"""Restart-safe daily workflow state machine tests."""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
from typing import TYPE_CHECKING

import pytest
from typer.testing import CliRunner

from quant_earning_edge.cli import app
from quant_earning_edge.orchestration import (
    ArtifactReference,
    DailyWorkflowController,
    DailyWorkflowRunner,
    DailyWorkflowState,
    DailyWorkflowStore,
    StageStatus,
    WorkflowStage,
)

if TYPE_CHECKING:
    from pathlib import Path


class AdvancingClock:
    def __init__(self, now: datetime) -> None:
        self.now = now

    def __call__(self) -> datetime:
        result = self.now
        self.now += timedelta(seconds=1)
        return result


def _handlers(root: Path) -> dict[WorkflowStage, object]:
    handlers: dict[WorkflowStage, object] = {}
    for stage in WorkflowStage:

        def run(
            state: DailyWorkflowState,
            current: WorkflowStage,
            *,
            expected: WorkflowStage = stage,
        ) -> tuple[Path, ...]:
            assert current is expected
            output = root / f"{state.trade_date}-{current.value}.json"
            output.write_text(
                f'{{"stage":"{current.value}","revision":{state.revision}}}',
                encoding="utf-8",
            )
            return (output,)

        handlers[stage] = run
    return handlers


def test_runner_loops_through_every_required_stage(tmp_path: Path) -> None:
    store = DailyWorkflowStore(tmp_path)
    clock = AdvancingClock(datetime(2026, 7, 28, 1, 0, tzinfo=UTC))
    runner = DailyWorkflowRunner(
        store=store,
        handlers=_handlers(tmp_path),  # type: ignore[arg-type]
        worker_id="worker-1",
        clock=clock,
    )

    result = runner.run_until_idle(trade_date=date(2026, 7, 28))

    assert result.complete
    assert all(item.status is StageStatus.SUCCEEDED for item in result.stages)
    assert [item.attempts for item in result.stages] == [1] * len(WorkflowStage)
    assert store.load_latest(result.trade_date) == result
    for record in result.stages:
        assert record.output_artifacts
        record.output_artifacts[0].verify()


def test_unexpired_lease_prevents_a_second_worker_claim(tmp_path: Path) -> None:
    now = datetime(2026, 7, 28, 1, 0, tzinfo=UTC)
    state = DailyWorkflowState.initialize(trade_date=date(2026, 7, 28), now=now)
    controller = DailyWorkflowController()
    claimed = controller.claim_next(state, worker_id="first", now=now)
    assert claimed is not None

    duplicate = controller.claim_next(
        claimed,
        worker_id="second",
        now=now + timedelta(minutes=1),
    )

    assert duplicate is None


def test_expired_lease_is_reclaimed_and_attempt_count_increments(tmp_path: Path) -> None:
    store = DailyWorkflowStore(tmp_path)
    now = datetime(2026, 7, 28, 1, 0, tzinfo=UTC)
    initial = store.write(DailyWorkflowState.initialize(trade_date=date(2026, 7, 28), now=now))
    claimed = DailyWorkflowController().claim_next(
        initial,
        worker_id="crashed-worker",
        now=now,
        lease_duration=timedelta(minutes=5),
    )
    assert claimed is not None
    store.write(claimed)
    runner = DailyWorkflowRunner(
        store=store,
        handlers=_handlers(tmp_path),  # type: ignore[arg-type]
        worker_id="replacement",
        clock=AdvancingClock(now + timedelta(minutes=6)),
    )

    result = runner.run_until_idle(trade_date=initial.trade_date)

    assert result.complete
    assert result.stages[0].attempts == 2


def test_handler_failure_is_durable_and_retried_on_next_run(tmp_path: Path) -> None:
    store = DailyWorkflowStore(tmp_path)
    clock = AdvancingClock(datetime(2026, 7, 28, 1, 0, tzinfo=UTC))

    def fail_once(_: DailyWorkflowState, __: WorkflowStage) -> tuple[Path, ...]:
        raise RuntimeError("provider unavailable")

    failed = DailyWorkflowRunner(
        store=store,
        handlers={WorkflowStage.FREEZE_INPUTS: fail_once},
        worker_id="worker-1",
        clock=clock,
    ).run_until_idle(trade_date=date(2026, 7, 28))

    assert failed.stages[0].status is StageStatus.FAILED
    assert failed.stages[0].error_type == "RuntimeError"
    assert failed.stages[0].error_message == "provider unavailable"

    recovered = DailyWorkflowRunner(
        store=store,
        handlers=_handlers(tmp_path),  # type: ignore[arg-type]
        worker_id="worker-2",
        clock=clock,
    ).run_until_idle(trade_date=failed.trade_date)

    assert recovered.complete
    assert recovered.stages[0].attempts == 2


def test_tampered_artifact_cannot_complete_a_stage(tmp_path: Path) -> None:
    now = datetime(2026, 7, 28, 1, 0, tzinfo=UTC)
    controller = DailyWorkflowController()
    state = DailyWorkflowState.initialize(trade_date=date(2026, 7, 28), now=now)
    claimed = controller.claim_next(state, worker_id="worker", now=now)
    assert claimed is not None
    artifact_path = tmp_path / "artifact.json"
    artifact_path.write_text("original", encoding="utf-8")
    artifact = ArtifactReference.capture(artifact_path)
    artifact_path.write_text("changed", encoding="utf-8")

    with pytest.raises(ValueError, match="changed after capture"):
        controller.succeed(
            claimed,
            worker_id="worker",
            stage=WorkflowStage.FREEZE_INPUTS,
            artifacts=(artifact,),
            now=now + timedelta(seconds=1),
        )


def test_store_detects_tampered_revision(tmp_path: Path) -> None:
    store = DailyWorkflowStore(tmp_path)
    state = store.write(
        DailyWorkflowState.initialize(
            trade_date=date(2026, 7, 28),
            now=datetime(2026, 7, 28, 1, 0, tzinfo=UTC),
        )
    )
    revision = next(tmp_path.rglob("revision-*.json"))
    revision.write_bytes(state.canonical_bytes.replace(b'"revision":0', b'"revision":1'))

    with pytest.raises(ValueError, match="invalid daily workflow"):
        store.load_latest(state.trade_date)


def test_store_rejects_revision_gap_without_poisoning_chain(tmp_path: Path) -> None:
    store = DailyWorkflowStore(tmp_path)
    state = store.write(
        DailyWorkflowState.initialize(
            trade_date=date(2026, 7, 28),
            now=datetime(2026, 7, 28, 1, 0, tzinfo=UTC),
        )
    )
    gap = replace(state, revision=2, previous_sha256=state.sha256)

    with pytest.raises(RuntimeError, match="without gaps"):
        store.write(gap)

    assert store.load_latest(state.trade_date) == state


def test_workflow_cli_initialization_is_idempotent_and_status_is_visible(
    tmp_path: Path,
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(f"DATA_LAKE_ROOT={tmp_path / 'lake'}", encoding="utf-8")
    command = [
        "workflow",
        "initialize",
        "--trade-date",
        "2026-07-28",
        "--env-file",
        str(env_file),
    ]
    first = CliRunner().invoke(app, command)
    second = CliRunner().invoke(app, command)

    assert first.exit_code == 0
    assert json.loads(first.stdout)["created"]
    assert second.exit_code == 0
    assert not json.loads(second.stdout)["created"]

    status = CliRunner().invoke(
        app,
        [
            "workflow",
            "status",
            "--trade-date",
            "2026-07-28",
            "--env-file",
            str(env_file),
        ],
    )

    assert status.exit_code == 1
    payload = json.loads(status.stdout)
    assert payload["current_stage"] == "freeze_inputs"
    assert payload["current_status"] == "pending"
    assert payload["artifacts_intact"]
