"""Constrained shell-free adapters from workflow stages to existing qee commands."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import date  # noqa: TC003 - Pydantic resolves runtime annotations.
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator

from quant_earning_edge.orchestration.workflow import (
    STAGE_ORDER,
    DailyWorkflowState,
    StageHandler,
    WorkflowStage,
    WorkflowTrigger,
)

_ALLOWED_PREFIXES: dict[WorkflowStage, frozenset[tuple[str, str]]] = {
    WorkflowStage.FREEZE_INPUTS: frozenset(
        {
            ("calendar", "sessions"),
            ("ingest", "earnings"),
            ("ingest", "bars"),
            ("ingest", "minute-bars"),
            ("ingest", "corporate-actions"),
            ("universe", "build"),
            ("universe", "events"),
            ("features", "compute"),
        }
    ),
    WorkflowStage.GENERATE_ORDER_PLAN: frozenset({("model", "plan-event-backtest")}),
    WorkflowStage.EVALUATE_BREAKERS: frozenset({("monitoring", "circuit-breakers")}),
    WorkflowStage.SUBMIT_PAPER_ORDERS: frozenset(
        {
            ("paper", "submit-batch"),
            ("paper", "submit-order"),
        }
    ),
    WorkflowStage.CAPTURE_MARKET_EVENTS: frozenset({("ingest", "market-events")}),
    WorkflowStage.REPLAY_ORDERS: frozenset(
        {
            ("backtest", "materialize-replay-specs"),
            ("backtest", "replay-nbbo"),
            ("evaluation", "replay-session"),
        }
    ),
    WorkflowStage.RECONCILE_SESSION: frozenset({("paper", "reconcile")}),
    WorkflowStage.EVALUATE_PHASE6_PROGRESS: frozenset({("evaluation", "phase6-gate")}),
}
_SECRET_ARGUMENT_MARKERS = frozenset(
    {
        "--api-key",
        "--secret",
        "--secret-key",
        "--api-key-id",
        "--token",
        "--password",
    }
)


class CommandExecutor(Protocol):
    """Shell-free subprocess boundary injectable in tests."""

    def __call__(
        self,
        argv: tuple[str, ...],
        *,
        cwd: Path,
        timeout_seconds: float,
    ) -> QeeCommandResult: ...


@dataclass(frozen=True)
class QeeCommandResult:
    """Non-secret command result retained only long enough to resolve artifacts."""

    return_code: int
    stdout: str
    stderr: str = ""
    duration_ms: int = 0


@dataclass(frozen=True)
class CommandReceipt:
    """Non-secret evidence for one attempted qee stage command."""

    schema_version: int
    trade_date: date
    workflow_state_sha256: str
    stage: WorkflowStage
    stage_attempt: int
    command_index: int
    command_prefix: tuple[str, str]
    arguments_sha256: str
    return_code: int
    stdout_sha256: str
    stderr_sha256: str
    duration_ms: int

    @property
    def canonical_bytes(self) -> bytes:
        return json.dumps(
            asdict(self),
            default=lambda item: (
                item.value if isinstance(item, WorkflowStage) else item.isoformat()
            ),
            sort_keys=True,
            separators=(",", ":"),
        ).encode()

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes).hexdigest()

    def write(self, output: Path) -> None:
        encoded = self.canonical_bytes
        output.parent.mkdir(parents=True, exist_ok=True)
        try:
            with output.open("xb") as destination:
                destination.write(encoded)
        except FileExistsError:
            if output.read_bytes() != encoded:
                raise RuntimeError(f"command receipt collision at {output}") from None


def execute_qee_command(
    argv: tuple[str, ...],
    *,
    cwd: Path,
    timeout_seconds: float,
) -> QeeCommandResult:
    """Execute one validated qee argv without invoking a shell."""
    started = time.perf_counter()
    completed = subprocess.run(
        argv,
        cwd=cwd,
        shell=False,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
    )
    return QeeCommandResult(
        return_code=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
        duration_ms=max(0, round((time.perf_counter() - started) * 1000)),
    )


class _StrictSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")


class QeeCommandSpec(_StrictSpec):
    """Arguments after `qee`; secrets must come from the environment."""

    arguments: tuple[str, ...] = Field(min_length=2)
    artifact_json_keys: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_arguments(self) -> QeeCommandSpec:
        if any(not argument.strip() for argument in self.arguments):
            raise ValueError("qee command arguments must not be blank")
        lowered = {argument.lower().split("=", maxsplit=1)[0] for argument in self.arguments}
        if lowered & _SECRET_ARGUMENT_MARKERS:
            raise ValueError("qee command secrets must come from environment variables")
        if any(not key.strip() for key in self.artifact_json_keys):
            raise ValueError("artifact JSON keys must not be blank")
        if len(set(self.artifact_json_keys)) != len(self.artifact_json_keys):
            raise ValueError("artifact JSON keys must be unique")
        return self


class WorkflowStageCommandSpec(_StrictSpec):
    """Commands and durable outputs for one workflow stage."""

    stage: WorkflowStage
    commands: tuple[QeeCommandSpec, ...] = Field(min_length=1)
    output_files: tuple[Path, ...] = ()

    @model_validator(mode="after")
    def validate_stage_commands(self) -> WorkflowStageCommandSpec:
        allowed = _ALLOWED_PREFIXES[self.stage]
        for command in self.commands:
            prefix = (command.arguments[0], command.arguments[1])
            if prefix not in allowed:
                raise ValueError(
                    f"qee command {prefix[0]} {prefix[1]} is not allowed for {self.stage.value}"
                )
        if len(set(self.output_files)) != len(self.output_files):
            raise ValueError("workflow stage output files must be unique")
        if not self.output_files and not any(
            command.artifact_json_keys for command in self.commands
        ):
            raise ValueError("workflow stage requires output files or command artifact JSON keys")
        return self


class WorkflowRunSpec(_StrictSpec):
    """Complete concrete command plan for one daily workflow loop."""

    trade_date: date
    trigger: WorkflowTrigger
    worker_id: str = Field(min_length=1)
    lease_seconds: int = Field(default=900, ge=1, le=3600)
    command_timeout_seconds: float = Field(default=1800, gt=0, le=7200)
    stages: tuple[WorkflowStageCommandSpec, ...]

    @model_validator(mode="after")
    def validate_complete_workflow(self) -> WorkflowRunSpec:
        if tuple(item.stage for item in self.stages) != STAGE_ORDER:
            raise ValueError("workflow run spec must contain every stage in exact order")
        if not self.worker_id.strip():
            raise ValueError("workflow worker_id must not be blank")
        return self

    def handlers(
        self,
        *,
        working_directory: Path,
        executor: CommandExecutor = execute_qee_command,
    ) -> dict[WorkflowStage, StageHandler]:
        """Build one constrained handler for every required stage."""
        return {
            item.stage: ConfiguredQeeStageHandler(
                spec=item,
                working_directory=working_directory,
                timeout_seconds=self.command_timeout_seconds,
                executor=executor,
            )
            for item in self.stages
        }


class ConfiguredQeeStageHandler:
    """Run validated qee commands and return configured output artifacts."""

    def __init__(
        self,
        *,
        spec: WorkflowStageCommandSpec,
        working_directory: Path,
        timeout_seconds: float,
        executor: CommandExecutor = execute_qee_command,
    ) -> None:
        self._spec = spec
        self._working_directory = working_directory.resolve()
        self._timeout_seconds = timeout_seconds
        self._executor = executor

    def __call__(
        self,
        state: DailyWorkflowState,
        stage: WorkflowStage,
    ) -> tuple[Path, ...]:
        if stage is not self._spec.stage:
            raise ValueError(
                f"configured handler for {self._spec.stage.value} received {stage.value}"
            )
        outputs = [
            path if path.is_absolute() else self._working_directory / path
            for path in self._spec.output_files
        ]
        running = next(
            record
            for record in state.stages
            if record.stage is stage and record.status.value == "running"
        )
        for command_index, command in enumerate(self._spec.commands, start=1):
            argv = (
                sys.executable,
                "-m",
                "quant_earning_edge.cli",
                *command.arguments,
            )
            result = self._executor(
                argv,
                cwd=self._working_directory,
                timeout_seconds=self._timeout_seconds,
            )
            receipt_path = self._write_receipt(
                state=state,
                stage=stage,
                stage_attempt=running.attempts,
                command_index=command_index,
                command=command,
                result=result,
            )
            if result.return_code:
                prefix = " ".join(command.arguments[:2])
                raise RuntimeError(
                    f"qee {prefix} failed with exit code {result.return_code} "
                    f"for trade date {state.trade_date}; receipt={receipt_path}"
                )
            outputs.append(receipt_path)
            outputs.extend(
                self._artifact_paths_from_stdout(
                    stdout=result.stdout,
                    keys=command.artifact_json_keys,
                )
            )
        unique_outputs = tuple(dict.fromkeys(path.resolve() for path in outputs))
        if not unique_outputs:
            raise RuntimeError(f"workflow stage {stage.value} produced no artifact paths")
        return unique_outputs

    def _write_receipt(
        self,
        *,
        state: DailyWorkflowState,
        stage: WorkflowStage,
        stage_attempt: int,
        command_index: int,
        command: QeeCommandSpec,
        result: QeeCommandResult,
    ) -> Path:
        encoded_arguments = json.dumps(
            command.arguments,
            ensure_ascii=True,
            separators=(",", ":"),
        ).encode()
        receipt = CommandReceipt(
            schema_version=1,
            trade_date=state.trade_date,
            workflow_state_sha256=state.sha256,
            stage=stage,
            stage_attempt=stage_attempt,
            command_index=command_index,
            command_prefix=(command.arguments[0], command.arguments[1]),
            arguments_sha256=hashlib.sha256(encoded_arguments).hexdigest(),
            return_code=result.return_code,
            stdout_sha256=hashlib.sha256(result.stdout.encode()).hexdigest(),
            stderr_sha256=hashlib.sha256(result.stderr.encode()).hexdigest(),
            duration_ms=result.duration_ms,
        )
        path = (
            self._working_directory
            / ".qee"
            / "receipts"
            / f"trade_date={state.trade_date.isoformat()}"
            / f"stage={stage.value}"
            / f"attempt={stage_attempt:03d}"
            / f"command-{command_index:03d}.json"
        )
        receipt.write(path)
        return path

    def _artifact_paths_from_stdout(
        self,
        *,
        stdout: str,
        keys: tuple[str, ...],
    ) -> tuple[Path, ...]:
        if not keys:
            return ()
        try:
            payload = json.loads(stdout)
        except json.JSONDecodeError as error:
            raise RuntimeError(
                "qee command did not emit the required JSON artifact paths"
            ) from error
        if not isinstance(payload, dict):
            raise RuntimeError("qee command artifact output must be a JSON object")
        paths: list[Path] = []
        for key in keys:
            value = payload.get(key)
            raw_paths: tuple[str, ...]
            if isinstance(value, str) and value:
                raw_paths = (value,)
            elif (
                isinstance(value, list)
                and value
                and all(isinstance(item, str) and item for item in value)
            ):
                raw_paths = tuple(str(item) for item in value)
            else:
                raise RuntimeError(f"qee command JSON field {key!r} is not an artifact path")
            paths.extend(
                path if path.is_absolute() else self._working_directory / path
                for path in (Path(item) for item in raw_paths)
            )
        return tuple(paths)
