"""Continuous proof-loop input discovery and next-session queuing tests."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

import pyarrow as pa
import pyarrow.parquet as pq

from quant_earning_edge.data import LakehouseLayout, SessionFileStore
from quant_earning_edge.data.clients import MarketSession
from quant_earning_edge.orchestration import (
    NextWorkflowQueuer,
    QeeCommandResult,
    WorkflowQueueStatus,
    WorkflowRunSpec,
    WorkflowStage,
)
from quant_earning_edge.signals import (
    ProductionModelTrainer,
    load_strategy_config,
)
from quant_earning_edge.universe import EVENT_CANDIDATE_SCHEMA

if TYPE_CHECKING:
    from collections.abc import Callable


_PREFIXES = {
    WorkflowStage.FREEZE_INPUTS: ("calendar", "sessions"),
    WorkflowStage.GENERATE_ORDER_PLAN: ("model", "plan-event-backtest"),
    WorkflowStage.EVALUATE_BREAKERS: ("monitoring", "circuit-breakers"),
    WorkflowStage.SUBMIT_PAPER_ORDERS: ("paper", "submit-batch"),
    WorkflowStage.CAPTURE_MARKET_EVENTS: ("ingest", "market-events"),
    WorkflowStage.REPLAY_ORDERS: ("backtest", "replay-nbbo"),
    WorkflowStage.RECONCILE_SESSION: ("paper", "reconcile"),
    WorkflowStage.EVALUATE_PHASE6_PROGRESS: ("evaluation", "phase6-gate"),
}


def _strategy_path() -> Path:
    return Path("configs/strategies/earnings_v1.yaml").resolve()


def _deployment(tmp_path: Path) -> tuple[Path, Path, Path]:
    lake = tmp_path / "lake"
    sessions = SessionFileStore(LakehouseLayout(lake)).write(
        (
            MarketSession(
                session_date=date(2026, 7, 27),
                open_at=datetime(2026, 7, 27, 13, 30, tzinfo=UTC),
                close_at=datetime(2026, 7, 27, 20, 0, tzinfo=UTC),
            ),
            MarketSession(
                session_date=date(2026, 7, 28),
                open_at=datetime(2026, 7, 28, 13, 30, tzinfo=UTC),
                close_at=datetime(2026, 7, 28, 20, 0, tzinfo=UTC),
            ),
        )
    )
    strategy = load_strategy_config(_strategy_path())
    training = tmp_path / "training.parquet"
    first = date(2025, 1, 2)
    rows = []
    for index in range(70):
        sign = 1.0 if index % 2 == 0 else -1.0
        row: dict[str, object] = {
            "asof_date": first + timedelta(days=index),
            "horizon_end_date": first + timedelta(days=index + 2),
            "forward_1d_close": sign * 0.01,
        }
        row.update({name: sign for name in strategy.features})
        rows.append(row)
    pq.write_table(pa.Table.from_pylist(rows), training)  # type: ignore[no-untyped-call]
    model = ProductionModelTrainer(
        feature_names=strategy.features,
        threshold=strategy.label.threshold,
        seed=strategy.seed,
        early_stopping_rounds=10,
    ).run(
        dataset_files=(training,),
        training_cutoff=date(2025, 3, 3),
        phase4_gate_sha256="f" * 64,
    )
    model_file, evidence = ProductionModelTrainer.write(model, tmp_path / "models")
    loop_spec = tmp_path / "loop.json"
    loop_spec.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "session_file": str(sessions.path),
                "strategy_config": str(_strategy_path()),
                "model_evidence": str(evidence),
                "model_file": str(model_file),
                "proof_start": "2026-07-28",
                "proof_end": "2026-07-28",
                "initial_cash": 100000,
                "artifact_root": str(tmp_path / "artifacts"),
                "staging_directory": str(tmp_path / "staging"),
                "worker_id": "paper-worker-1",
            }
        ),
        encoding="utf-8",
    )
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    return lake, loop_spec, inbox


def _executor(call_log: list[tuple[str, ...]]) -> Callable[..., QeeCommandResult]:
    def execute(
        argv: tuple[str, ...],
        *,
        cwd: Path,
        timeout_seconds: float,
    ) -> QeeCommandResult:
        del cwd, timeout_seconds
        call_log.append(argv)
        trade_date = argv[argv.index("--trade-date") + 1]
        output = Path(argv[argv.index("--output") + 1])
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(
                {
                    "trade_date": trade_date,
                    "trigger": "scheduled",
                    "worker_id": "paper-worker-1",
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
            ),
            encoding="utf-8",
        )
        return QeeCommandResult(return_code=0, stdout="{}")

    return execute


def test_queue_waits_for_authoritative_candidate_artifact(tmp_path: Path) -> None:
    lake, loop_spec, inbox = _deployment(tmp_path)
    calls: list[tuple[str, ...]] = []

    result = NextWorkflowQueuer(
        data_lake_root=lake,
        clock=lambda: datetime(2026, 7, 28, 1, 30, tzinfo=UTC),
        executor=_executor(calls),
    ).run_once(loop_spec=loop_spec, inbox=inbox)

    assert result.status is WorkflowQueueStatus.WAITING_FOR_INPUTS
    assert "event-candidate" in result.detail
    assert not calls


def test_queue_stages_zero_candidate_proof_start_without_fake_features(
    tmp_path: Path,
) -> None:
    lake, loop_spec, inbox = _deployment(tmp_path)
    candidate_root = lake / "gold" / "event-candidates" / "for_trade_date=2026-07-28"
    candidate_root.mkdir(parents=True)
    pq.write_table(  # type: ignore[no-untyped-call]
        pa.Table.from_pylist([], schema=EVENT_CANDIDATE_SCHEMA),
        candidate_root / "candidates-empty.parquet",
    )
    calls: list[tuple[str, ...]] = []
    queuer = NextWorkflowQueuer(
        data_lake_root=lake,
        clock=lambda: datetime(2026, 7, 28, 1, 30, tzinfo=UTC),
        executor=_executor(calls),
    )

    result = queuer.run_once(loop_spec=loop_spec, inbox=inbox)
    repeated = queuer.run_once(loop_spec=loop_spec, inbox=inbox)

    assert result.status is WorkflowQueueStatus.ADMISSION_REQUIRED
    assert result.workflow_spec is not None
    assert WorkflowRunSpec.model_validate_json(result.workflow_spec.read_bytes())
    assert "--stage-for-admission" in calls[0]
    assert "--feature-file" not in calls[0]
    assert repeated.status is WorkflowQueueStatus.ADMISSION_REQUIRED
    assert len(calls) == 1


def test_queue_invokes_provider_input_preparation_before_staging(
    tmp_path: Path,
) -> None:
    lake, loop_spec, inbox = _deployment(tmp_path)
    raw = json.loads(loop_spec.read_bytes())
    raw["universe_config"] = str(_strategy_path().parents[1] / "universe" / "default.yaml")
    halt_directory = tmp_path / "halts"
    halt_directory.mkdir()
    raw["halt_snapshot_directory"] = str(halt_directory)
    loop_spec.write_text(json.dumps(raw), encoding="utf-8")
    (halt_directory / "halt-2026-07-27.json").write_text("{}", encoding="utf-8")
    calls: list[tuple[str, ...]] = []

    def execute(
        argv: tuple[str, ...],
        *,
        cwd: Path,
        timeout_seconds: float,
    ) -> QeeCommandResult:
        if "prepare-session-inputs" in argv:
            calls.append(argv)
            root = lake / "gold" / "event-candidates" / "for_trade_date=2026-07-28"
            root.mkdir(parents=True)
            pq.write_table(  # type: ignore[no-untyped-call]
                pa.Table.from_pylist([], schema=EVENT_CANDIDATE_SCHEMA),
                root / "candidates-empty.parquet",
            )
            return QeeCommandResult(return_code=0, stdout="{}")
        return _executor(calls)(argv, cwd=cwd, timeout_seconds=timeout_seconds)

    result = NextWorkflowQueuer(
        data_lake_root=lake,
        clock=lambda: datetime(2026, 7, 28, 1, 30, tzinfo=UTC),
        executor=execute,
    ).run_once(loop_spec=loop_spec, inbox=inbox)

    assert result.status is WorkflowQueueStatus.ADMISSION_REQUIRED
    assert "prepare-session-inputs" in calls[0]
    assert "--stage-for-admission" in calls[1]
