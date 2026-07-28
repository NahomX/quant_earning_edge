"""Rolling Phase 6 control files are derived from durable daily evidence."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime
from typing import TYPE_CHECKING

from typer.testing import CliRunner

from quant_earning_edge.cli import app
from quant_earning_edge.data import LakehouseLayout, SessionFileStore
from quant_earning_edge.data.clients import MarketSession
from quant_earning_edge.evaluation import (
    Phase6AggregationSpec,
    ReplaySessionAggregator,
)
from quant_earning_edge.orchestration import WorkflowHealthReport

if TYPE_CHECKING:
    from pathlib import Path


def test_prepare_phase6_controls_includes_current_future_output_path(
    tmp_path: Path,
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
    artifact_root = tmp_path / "artifacts"
    prior_report = artifact_root / "trade_date=2026-07-27" / "replay-session.json"
    ReplaySessionAggregator().evaluate(
        evidence=(),
        round_trips=(),
        session_date=dates[0],
        initial_cash=100_000,
    ).write(prior_report)
    health_output = tmp_path / "controls" / "health.json"
    aggregation_output = tmp_path / "controls" / "phase6.json"
    env_file = tmp_path / ".env"
    env_file.write_text(
        f"DATA_LAKE_ROOT={tmp_path / 'lake'}",
        encoding="utf-8",
    )
    arguments = [
        "evaluation",
        "prepare-phase6-controls",
        "--session-file",
        str(calendar.path),
        "--proof-start",
        "2026-07-27",
        "--proof-end",
        "2026-07-28",
        "--current-trade-date",
        "2026-07-28",
        "--initial-cash",
        "100000",
        "--artifact-root",
        str(artifact_root),
        "--health-output",
        str(health_output),
        "--aggregation-output",
        str(aggregation_output),
        "--env-file",
        str(env_file),
    ]

    result = CliRunner().invoke(app, arguments)
    repeated = CliRunner().invoke(app, arguments)

    assert result.exit_code == 0
    assert repeated.exit_code == 0
    spec = Phase6AggregationSpec.model_validate_json(aggregation_output.read_bytes())
    expected_current = (artifact_root / "trade_date=2026-07-28" / "replay-session.json").resolve()
    assert spec.session_report_files == (prior_report.resolve(), expected_current)
    assert not expected_current.exists()
    health = WorkflowHealthReport.load(health_output)
    assert health.missing_dates == dates
    assert json.loads(result.stdout)["report_count"] == 2
