"""Constrained qee-command workflow handler tests."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from typing import TYPE_CHECKING

import pytest
from pydantic import ValidationError

from quant_earning_edge.orchestration import (
    ArtifactReference,
    ConfiguredQeeStageHandler,
    DailyWorkflowController,
    DailyWorkflowRunner,
    DailyWorkflowState,
    DailyWorkflowStore,
    QeeCommandResult,
    WorkflowRunSpec,
    WorkflowStage,
    WorkflowStageCommandSpec,
    WorkflowTrigger,
)

if TYPE_CHECKING:
    from pathlib import Path

_PREFIXES = {
    WorkflowStage.FREEZE_INPUTS: ("calendar", "sessions"),
    WorkflowStage.GENERATE_ORDER_PLAN: ("model", "plan-event-backtest"),
    WorkflowStage.EVALUATE_BREAKERS: ("monitoring", "circuit-breakers"),
    WorkflowStage.SUBMIT_PAPER_ORDERS: ("paper", "submit-order"),
    WorkflowStage.CAPTURE_MARKET_EVENTS: ("ingest", "market-events"),
    WorkflowStage.REPLAY_ORDERS: ("backtest", "replay-nbbo"),
    WorkflowStage.RECONCILE_SESSION: ("paper", "reconcile"),
    WorkflowStage.EVALUATE_PHASE6_PROGRESS: ("evaluation", "phase6-gate"),
}


def _run_spec() -> WorkflowRunSpec:
    return WorkflowRunSpec.model_validate(
        {
            "trade_date": "2026-07-28",
            "trigger": "scheduled",
            "worker_id": "production-worker-1",
            "stages": [
                {
                    "stage": stage,
                    "commands": [
                        {
                            "arguments": [
                                *_PREFIXES[stage],
                                "--output",
                                f"{stage.value}.json",
                            ]
                        }
                    ],
                    "output_files": [f"{stage.value}.json"],
                }
                for stage in WorkflowStage
            ],
        }
    )


def test_run_spec_requires_exact_stage_order_and_allowed_qee_prefixes() -> None:
    raw = _run_spec().model_dump(mode="json")
    raw["stages"][0]["commands"][0]["arguments"][:2] = ["workflow", "run"]

    with pytest.raises(ValidationError, match="not allowed"):
        WorkflowRunSpec.model_validate(raw)

    missing = _run_spec().model_dump(mode="json")
    missing["stages"].pop()
    with pytest.raises(ValidationError, match="every stage"):
        WorkflowRunSpec.model_validate(missing)


def test_run_spec_rejects_secret_command_arguments() -> None:
    raw = _run_spec().model_dump(mode="json")
    raw["stages"][0]["commands"][0]["arguments"].extend(["--api-key", "secret"])

    with pytest.raises(ValidationError, match="environment variables"):
        WorkflowRunSpec.model_validate(raw)


def test_configured_handlers_execute_without_shell_and_complete_loop(tmp_path: Path) -> None:
    calls: list[tuple[tuple[str, ...], Path, float]] = []

    def execute(
        argv: tuple[str, ...],
        *,
        cwd: Path,
        timeout_seconds: float,
    ) -> QeeCommandResult:
        calls.append((argv, cwd, timeout_seconds))
        output_index = argv.index("--output") + 1
        (cwd / argv[output_index]).write_text("{}", encoding="utf-8")
        return QeeCommandResult(return_code=0, stdout="{}")

    spec = _run_spec()
    runner = DailyWorkflowRunner(
        store=DailyWorkflowStore(tmp_path / "lake"),
        handlers=spec.handlers(working_directory=tmp_path, executor=execute),
        worker_id=spec.worker_id,
        clock=lambda: datetime.now(UTC),
        trigger=spec.trigger,
    )

    result = runner.run_until_idle(trade_date=spec.trade_date)

    assert result.complete
    assert len(calls) == len(WorkflowStage)
    assert all(call[0][1:3] == ("-m", "quant_earning_edge.cli") for call in calls)
    assert all(call[1] == tmp_path.resolve() for call in calls)
    receipts = tuple(tmp_path.rglob("command-*.json"))
    assert len(receipts) == len(WorkflowStage)
    assert all('"stdout":' not in path.read_text(encoding="utf-8") for path in receipts)


def test_nonzero_qee_exit_becomes_durable_retryable_stage_failure(tmp_path: Path) -> None:
    spec = _run_spec()

    def reject(
        argv: tuple[str, ...],
        *,
        cwd: Path,
        timeout_seconds: float,
    ) -> QeeCommandResult:
        del argv, cwd, timeout_seconds
        return QeeCommandResult(return_code=2, stdout="")

    result = DailyWorkflowRunner(
        store=DailyWorkflowStore(tmp_path / "lake"),
        handlers=spec.handlers(working_directory=tmp_path, executor=reject),
        worker_id=spec.worker_id,
        clock=lambda: datetime.now(UTC),
        trigger=spec.trigger,
    ).run_until_idle(trade_date=date(2026, 7, 28))

    assert not result.complete
    assert result.stages[0].error_type == "RuntimeError"
    assert "exit code 2" in (result.stages[0].error_message or "")
    assert len(tuple(tmp_path.rglob("command-*.json"))) == 1


def test_command_stdout_can_resolve_content_addressed_artifact_path(tmp_path: Path) -> None:
    artifact = tmp_path / "sessions-content-addressed.json"
    artifact.write_text("{}", encoding="utf-8")
    raw = _run_spec().model_dump(mode="json")
    first = raw["stages"][0]
    first["output_files"] = []
    first["commands"][0]["artifact_json_keys"] = ["path"]
    spec = WorkflowRunSpec.model_validate(raw)

    def execute(
        argv: tuple[str, ...],
        *,
        cwd: Path,
        timeout_seconds: float,
    ) -> QeeCommandResult:
        del argv, cwd, timeout_seconds
        return QeeCommandResult(
            return_code=0,
            stdout=f'{{"path":"{artifact.as_posix()}"}}',
        )

    handler = spec.handlers(working_directory=tmp_path, executor=execute)[
        WorkflowStage.FREEZE_INPUTS
    ]
    state = DailyWorkflowState.initialize(
        trade_date=spec.trade_date,
        now=datetime.now(UTC),
        trigger=spec.trigger,
    )
    claimed = DailyWorkflowController().claim_next(
        state,
        worker_id=spec.worker_id,
        now=datetime.now(UTC),
    )
    assert claimed is not None

    outputs = handler(claimed, WorkflowStage.FREEZE_INPUTS)
    assert artifact.resolve() in outputs
    assert any(path.name == "command-001.json" for path in outputs)


def _claimed_replay_state(tmp_path: Path) -> DailyWorkflowState:
    controller = DailyWorkflowController()
    now = datetime.now(UTC)
    state = DailyWorkflowState.initialize(
        trade_date=date(2026, 7, 28),
        now=now,
        trigger=WorkflowTrigger.SCHEDULED,
    )
    for index, stage in enumerate(tuple(WorkflowStage)[:5], start=1):
        claimed = controller.claim_next(
            state,
            worker_id="worker",
            now=now + timedelta(seconds=index * 2),
        )
        assert claimed is not None
        paths: tuple[Path, ...]
        if stage is WorkflowStage.CAPTURE_MARKET_EVENTS:
            paths = (
                tmp_path / "dataset=nbbo-quotes" / "quotes.parquet",
                tmp_path / "dataset=stock-trades" / "trades.parquet",
            )
        else:
            paths = (tmp_path / stage.value / "output.json",)
        for path in paths:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{}", encoding="utf-8")
        state = controller.succeed(
            claimed,
            worker_id="worker",
            stage=stage,
            artifacts=tuple(ArtifactReference.capture(path) for path in paths),
            now=now + timedelta(seconds=index * 2 + 1),
        )
    replay = controller.claim_next(
        state,
        worker_id="worker",
        now=now + timedelta(seconds=20),
    )
    assert replay is not None
    return replay


def test_artifact_bindings_expand_selected_prior_stage_files(tmp_path: Path) -> None:
    state = _claimed_replay_state(tmp_path)
    output = tmp_path / "manifest.json"
    output.write_text("{}", encoding="utf-8")
    seen: list[tuple[str, ...]] = []

    def execute(
        argv: tuple[str, ...],
        *,
        cwd: Path,
        timeout_seconds: float,
    ) -> QeeCommandResult:
        del cwd, timeout_seconds
        seen.append(argv)
        return QeeCommandResult(return_code=0, stdout="{}")

    spec = WorkflowStageCommandSpec.model_validate(
        {
            "stage": "replay_orders",
            "commands": [
                {
                    "arguments": [
                        "backtest",
                        "materialize-frozen-replay-specs",
                        "--frozen-orders",
                        "frozen.json",
                    ],
                    "artifact_bindings": [
                        {
                            "source_stage": "capture_market_events",
                            "option": "--quote-file",
                            "file_glob": "*.parquet",
                            "path_contains": ["dataset=nbbo-quotes"],
                            "maximum_matches": 1,
                        },
                        {
                            "source_stage": "capture_market_events",
                            "option": "--trade-file",
                            "file_glob": "*.parquet",
                            "path_contains": ["dataset=stock-trades"],
                            "maximum_matches": 1,
                        },
                    ],
                }
            ],
            "output_files": [output],
        }
    )
    ConfiguredQeeStageHandler(
        spec=spec,
        working_directory=tmp_path,
        timeout_seconds=30,
        executor=execute,
    )(state, WorkflowStage.REPLAY_ORDERS)

    argv = seen[0]
    quote = str((tmp_path / "dataset=nbbo-quotes" / "quotes.parquet").resolve())
    trade = str((tmp_path / "dataset=stock-trades" / "trades.parquet").resolve())
    assert argv[argv.index("--quote-file") + 1] == quote
    assert argv[argv.index("--trade-file") + 1] == trade


def test_artifact_binding_cardinality_failure_prevents_command(tmp_path: Path) -> None:
    state = _claimed_replay_state(tmp_path)
    spec = WorkflowStageCommandSpec.model_validate(
        {
            "stage": "replay_orders",
            "commands": [
                {
                    "arguments": ["backtest", "materialize-frozen-replay-specs"],
                    "artifact_bindings": [
                        {
                            "source_stage": "capture_market_events",
                            "option": "--quote-file",
                            "path_contains": ["does-not-exist"],
                        }
                    ],
                }
            ],
            "output_files": [tmp_path / "manifest.json"],
        }
    )

    with pytest.raises(RuntimeError, match="matched 0"):
        ConfiguredQeeStageHandler(
            spec=spec,
            working_directory=tmp_path,
            timeout_seconds=30,
        )(state, WorkflowStage.REPLAY_ORDERS)
