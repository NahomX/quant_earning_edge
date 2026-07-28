"""Persistent inbox worker for restart-safe daily workflow run specifications."""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import ValidationError

from quant_earning_edge.orchestration.commands import (
    CommandExecutor,
    WorkflowRunSpec,
    execute_qee_command,
)
from quant_earning_edge.orchestration.workflow import DailyWorkflowRunner, DailyWorkflowStore

if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass(frozen=True)
class WorkerSpecResult:
    """One inbox specification outcome without captured command output."""

    spec_path: str
    spec_sha256: str
    trade_date: str | None
    workflow_state_sha256: str | None
    complete: bool
    error_type: str | None
    error_message: str | None

    def __post_init__(self) -> None:
        if not self.spec_path.strip() or not _is_sha256(self.spec_sha256):
            raise ValueError("worker spec result identity is invalid")
        if self.complete and (self.error_type is not None or self.error_message is not None):
            raise ValueError("complete worker result cannot retain an error")


@dataclass(frozen=True)
class WorkerCycleReport:
    """Immutable heartbeat and outcomes for one inbox scan."""

    schema_version: int
    worker_id: str
    evaluated_at: datetime
    inbox_path: str
    results: tuple[WorkerSpecResult, ...]

    def __post_init__(self) -> None:
        if self.schema_version != 1 or not self.worker_id.strip() or not self.inbox_path.strip():
            raise ValueError("worker cycle identity is invalid")
        if self.evaluated_at.tzinfo is None or self.evaluated_at.utcoffset() is None:
            raise ValueError("worker cycle time must be timezone-aware")
        spec_paths = tuple(item.spec_path for item in self.results)
        if spec_paths != tuple(sorted(set(spec_paths))):
            raise ValueError("worker cycle results must be unique and sorted")

    @property
    def canonical_bytes(self) -> bytes:
        return json.dumps(
            asdict(self),
            default=lambda item: item.isoformat(),
            sort_keys=True,
            separators=(",", ":"),
        ).encode()

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes).hexdigest()

    @property
    def all_complete(self) -> bool:
        return all(item.complete for item in self.results)

    @classmethod
    def load(cls, path: Path) -> WorkerCycleReport:
        """Strictly reload one canonical worker heartbeat."""
        try:
            raw = json.loads(path.read_bytes())
            allowed = {
                "schema_version",
                "worker_id",
                "evaluated_at",
                "inbox_path",
                "results",
            }
            if not isinstance(raw, dict) or set(raw) != allowed:
                raise ValueError("worker cycle fields are invalid")
            raw_results = raw["results"]
            if not isinstance(raw_results, list):
                raise ValueError("worker cycle results are invalid")
            report = cls(
                schema_version=int(raw["schema_version"]),
                worker_id=str(raw["worker_id"]),
                evaluated_at=datetime.fromisoformat(str(raw["evaluated_at"])),
                inbox_path=str(raw["inbox_path"]),
                results=tuple(
                    WorkerSpecResult(**item) for item in raw_results if isinstance(item, dict)
                ),
            )
            if len(report.results) != len(raw_results):
                raise ValueError("worker cycle result is invalid")
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid workflow worker cycle: {path}") from error
        if json.loads(report.canonical_bytes) != raw:
            raise ValueError("workflow worker cycle is not canonical")
        return report


class WorkflowWorkerStore:
    """Content-addressed worker-cycle heartbeat storage."""

    def __init__(self, root: Path) -> None:
        self._root = root.resolve() / "manifests" / "job=workflow-worker" / "cycles"

    def write(self, report: WorkerCycleReport) -> Path:
        timestamp = report.evaluated_at.strftime("%Y%m%dT%H%M%S%f%z")
        path = self._root / f"cycle-{timestamp}-{report.sha256[:16]}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with path.open("xb") as destination:
                destination.write(report.canonical_bytes)
        except FileExistsError:
            if path.read_bytes() != report.canonical_bytes:
                raise RuntimeError(f"workflow worker cycle collision at {path}") from None
        return path

    def load_latest(self) -> WorkerCycleReport | None:
        """Load the newest canonical worker heartbeat, if one exists."""
        paths = tuple(sorted(self._root.glob("cycle-*.json")))
        return WorkerCycleReport.load(paths[-1]) if paths else None


class WorkflowInboxWorker:
    """Scan immutable specs and resume each workflow until idle."""

    def __init__(
        self,
        *,
        data_lake_root: Path,
        worker_id: str,
        clock: Callable[[], datetime],
        executor: CommandExecutor = execute_qee_command,
    ) -> None:
        normalized = worker_id.strip()
        if not normalized:
            raise ValueError("workflow inbox worker_id must not be blank")
        self._store = DailyWorkflowStore(data_lake_root)
        self._cycle_store = WorkflowWorkerStore(data_lake_root)
        self._worker_id = normalized
        self._clock = clock
        self._executor = executor

    def run_once(self, inbox: Path) -> tuple[WorkerCycleReport, Path]:
        """Run every JSON spec in lexical order and persist one cycle heartbeat."""
        resolved_inbox = inbox.resolve()
        if not resolved_inbox.is_dir():
            raise ValueError(f"workflow inbox is not a directory: {resolved_inbox}")
        results = tuple(self._run_spec(path) for path in sorted(resolved_inbox.glob("*.json")))
        evaluated_at = self._clock()
        if evaluated_at.tzinfo is None or evaluated_at.utcoffset() is None:
            raise ValueError("workflow worker clock must be timezone-aware")
        report = WorkerCycleReport(
            schema_version=1,
            worker_id=self._worker_id,
            evaluated_at=evaluated_at,
            inbox_path=str(resolved_inbox),
            results=results,
        )
        return report, self._cycle_store.write(report)

    def _run_spec(self, path: Path) -> WorkerSpecResult:
        encoded = path.read_bytes()
        digest = hashlib.sha256(encoded).hexdigest()
        try:
            spec = WorkflowRunSpec.model_validate_json(encoded)
            state = DailyWorkflowRunner(
                store=self._store,
                handlers=spec.handlers(
                    working_directory=path.parent,
                    executor=self._executor,
                ),
                worker_id=spec.worker_id,
                clock=self._clock,
                trigger=spec.trigger,
                lease_duration=timedelta(seconds=spec.lease_seconds),
            ).run_until_idle(trade_date=spec.trade_date)
            current = next(
                (item for item in state.stages if item.status.value != "succeeded"),
                None,
            )
            finalization_error = (
                self._finalize_completed(spec=spec, spec_path=path, state_sha256=state.sha256)
                if state.complete
                else None
            )
            return WorkerSpecResult(
                spec_path=str(path.resolve()),
                spec_sha256=digest,
                trade_date=spec.trade_date.isoformat(),
                workflow_state_sha256=state.sha256,
                complete=state.complete and finalization_error is None,
                error_type=(
                    "Phase6FinalizationError"
                    if finalization_error is not None
                    else (current.error_type if current else None)
                ),
                error_message=(
                    finalization_error
                    if finalization_error is not None
                    else (current.error_message if current else None)
                ),
            )
        except (OSError, ValidationError, ValueError, RuntimeError) as error:
            return WorkerSpecResult(
                spec_path=str(path.resolve()),
                spec_sha256=digest,
                trade_date=None,
                workflow_state_sha256=None,
                complete=False,
                error_type=type(error).__name__,
                error_message=(str(error).strip() or "workflow spec failed")[:1000],
            )

    def _finalize_completed(
        self,
        *,
        spec: WorkflowRunSpec,
        spec_path: Path,
        state_sha256: str,
    ) -> str | None:
        command = spec.stages[-1].commands[-1]
        arguments = command.arguments
        aggregation = _option_value(arguments, "--aggregation-spec")
        phase6_output = _option_value(arguments, "--output")
        if aggregation is None or phase6_output is None:
            return None
        working_directory = spec_path.parent.resolve()
        aggregation_path = _resolved(aggregation, relative_to=working_directory)
        original_output = _resolved(phase6_output, relative_to=working_directory)
        daily_root = original_output.parent
        artifact_root = daily_root.parent
        output_directory = daily_root / "post-completion"
        marker = output_directory / f"finalization-{state_sha256}.json"
        if marker.is_file() and _finalization_marker_is_intact(
            marker,
            output_directory=output_directory,
            state_sha256=state_sha256,
        ):
            return None
        result = self._executor(
            (
                sys.executable,
                "-m",
                "quant_earning_edge.cli",
                "evaluation",
                "finalize-phase6",
                "--aggregation-spec",
                str(aggregation_path),
                "--current-trade-date",
                spec.trade_date.isoformat(),
                "--artifact-root",
                str(artifact_root),
                "--output-directory",
                str(output_directory),
            ),
            cwd=working_directory,
            timeout_seconds=spec.command_timeout_seconds,
        )
        if result.return_code:
            return f"post-completion Phase 6 finalizer exited {result.return_code}"
        if not _finalization_marker_is_intact(
            marker,
            output_directory=output_directory,
            state_sha256=state_sha256,
        ):
            return "post-completion Phase 6 finalizer produced no intact manifest"
        return None


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(item in "0123456789abcdef" for item in value)


def _option_value(arguments: tuple[str, ...], option: str) -> str | None:
    try:
        index = arguments.index(option)
    except ValueError:
        return None
    return arguments[index + 1] if index + 1 < len(arguments) else None


def _resolved(value: str, *, relative_to: Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (relative_to / path).resolve()


def _finalization_marker_is_intact(
    path: Path,
    *,
    output_directory: Path,
    state_sha256: str,
) -> bool:
    try:
        raw = json.loads(path.read_bytes())
        if (
            not isinstance(raw, dict)
            or raw.get("schema_version") != 1
            or raw.get("workflow_state_sha256") != state_sha256
        ):
            return False
        for path_key, hash_key in (
            ("health_path", "health_sha256"),
            ("aggregation_path", "aggregation_sha256"),
            ("gate_report_path", "gate_report_sha256"),
        ):
            artifact = Path(str(raw[path_key])).resolve()
            digest = str(raw[hash_key])
            if (
                not artifact.is_relative_to(output_directory.resolve())
                or not artifact.is_file()
                or not _is_sha256(digest)
                or hashlib.sha256(artifact.read_bytes()).hexdigest() != digest
            ):
                return False
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False
    return True
