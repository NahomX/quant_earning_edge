"""The isolated smoke path is manual, zero-order, and proof-excluded."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from typer.testing import CliRunner

from quant_earning_edge.cli import app
from quant_earning_edge.data import LakehouseLayout, SessionFileStore
from quant_earning_edge.data.clients import MarketSession
from quant_earning_edge.evaluation import ReplaySessionAggregator
from quant_earning_edge.orchestration import (
    WorkerCycleReport,
    WorkerSpecResult,
    WorkflowInboxWorker,
    WorkflowRunSpec,
)
from quant_earning_edge.signals import (
    DailyOrderPlanningSpec,
    LiveOrderPlanner,
    load_strategy_config,
    strategy_file_sha256,
)

if TYPE_CHECKING:
    import pytest


def _strategy_path() -> Path:
    return Path(__file__).parents[2] / "configs/strategies/earnings_v1.yaml"


def test_smoke_command_builds_isolated_manual_no_trade_workflow(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dates = (date(2026, 7, 27), date(2026, 7, 28))
    calendar = SessionFileStore(LakehouseLayout(tmp_path / "calendar")).write(
        tuple(
            MarketSession(
                session_date=item,
                open_at=datetime(item.year, item.month, item.day, 13, 30, tzinfo=UTC),
                close_at=datetime(item.year, item.month, item.day, 20, 0, tzinfo=UTC),
            )
            for item in dates
        )
    )
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            (
                "POLYGON_API_KEY=smoke-polygon",
                "APCA_API_KEY_ID=smoke-alpaca",
                "APCA_API_SECRET_KEY=smoke-secret",
            )
        ),
        encoding="utf-8",
    )
    for key in ("POLYGON_API_KEY", "APCA_API_KEY_ID", "APCA_API_SECRET_KEY"):
        monkeypatch.delenv(key, raising=False)
    smoke_root = tmp_path / "smoke"

    def run_once(
        _: object,
        inbox: Path,
    ) -> tuple[WorkerCycleReport, Path]:
        spec_path = next(inbox.glob("smoke-*.json"))
        spec = WorkflowRunSpec.model_validate_json(spec_path.read_bytes())
        planning_path = smoke_root / "current-no-trade-planning.json"
        planning = DailyOrderPlanningSpec.model_validate_json(planning_path.read_bytes())
        strategy_path = _strategy_path()
        frozen = LiveOrderPlanner(
            load_strategy_config(strategy_path),
            strategy_sha256=strategy_file_sha256(strategy_path),
        ).plan(planning)
        current_root = smoke_root / "artifacts" / "trade_date=2026-07-28"
        frozen.write(current_root / "frozen-daily-orders.json")
        ReplaySessionAggregator().evaluate(
            evidence=(),
            round_trips=(),
            session_date=dates[1],
            initial_cash=100_000,
        ).write(current_root / "replay-session.json")
        cycle = WorkerCycleReport(
            schema_version=1,
            worker_id="smoke-worker",
            evaluated_at=datetime(2026, 7, 28, 12, 0, tzinfo=UTC),
            inbox_path=str(inbox.resolve()),
            results=(
                WorkerSpecResult(
                    spec_path=str(spec_path.resolve()),
                    spec_sha256=spec.sha256,
                    trade_date=dates[1].isoformat(),
                    workflow_state_sha256="d" * 64,
                    complete=True,
                    error_type=None,
                    error_message=None,
                ),
            ),
        )
        cycle_path = smoke_root / "data-lake" / "fake-cycle.json"
        cycle_path.parent.mkdir(parents=True, exist_ok=True)
        cycle_path.write_bytes(cycle.canonical_bytes)
        return cycle, cycle_path

    monkeypatch.setattr(
        WorkflowInboxWorker,
        "run_once",
        run_once,
    )
    output = tmp_path / "smoke-result.json"

    result = CliRunner().invoke(
        app,
        [
            "workflow",
            "smoke-no-trade",
            "--session-file",
            str(calendar.path),
            "--smoke-date",
            dates[1].isoformat(),
            "--strategy-config",
            str(_strategy_path()),
            "--initial-cash",
            "100000",
            "--smoke-root",
            str(smoke_root),
            "--output",
            str(output),
            "--env-file",
            str(env_file),
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(output.read_bytes())
    assert payload["trigger"] == "manual"
    assert not payload["counts_toward_phase6"]
    assert payload["intended_order_count"] == 0
    spec = WorkflowRunSpec.model_validate_json(
        (smoke_root / "inbox" / "smoke-2026-07-28.json").read_bytes()
    )
    assert all(stage.not_before is None and stage.not_after is None for stage in spec.stages)
