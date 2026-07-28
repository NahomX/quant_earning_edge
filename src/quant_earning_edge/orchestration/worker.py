"""Persistent inbox worker for restart-safe daily workflow run specifications."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
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
    from pathlib import Path


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


@dataclass(frozen=True)
class WorkerCycleReport:
    """Immutable heartbeat and outcomes for one inbox scan."""

    schema_version: int
    worker_id: str
    evaluated_at: datetime
    inbox_path: str
    results: tuple[WorkerSpecResult, ...]

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
            return WorkerSpecResult(
                spec_path=str(path.resolve()),
                spec_sha256=digest,
                trade_date=spec.trade_date.isoformat(),
                workflow_state_sha256=state.sha256,
                complete=state.complete,
                error_type=current.error_type if current else None,
                error_message=current.error_message if current else None,
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
