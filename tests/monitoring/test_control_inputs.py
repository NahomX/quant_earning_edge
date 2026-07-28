"""Breaker controls retain completed replay-session provenance."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

from typer.testing import CliRunner

from quant_earning_edge.cli import app
from quant_earning_edge.evaluation import ReplaySessionAggregator
from quant_earning_edge.monitoring import (
    CircuitBreakerEvaluationSpec,
    CircuitBreakerEvaluator,
)
from quant_earning_edge.signals import (
    DailyOrderPlanningSpec,
    LiveOrderPlanner,
    load_strategy_config,
    strategy_file_sha256,
)


def _strategy_path() -> Path:
    return Path(__file__).parents[2] / "configs/strategies/earnings_v1.yaml"


def _source(tmp_path: Path, session_date: date) -> tuple[Path, Path]:
    opened = datetime(
        session_date.year,
        session_date.month,
        session_date.day,
        13,
        30,
        tzinfo=UTC,
    )
    planning = DailyOrderPlanningSpec(
        trade_date=session_date,
        decision_at=opened - timedelta(hours=12),
        equity=100_000,
        entry_submitted_at=opened,
        entry_expires_at=opened + timedelta(minutes=5),
        exit_submitted_at=opened + timedelta(hours=6),
        exit_expires_at=opened + timedelta(hours=6, minutes=5),
    )
    strategy_path = _strategy_path()
    frozen = LiveOrderPlanner(
        load_strategy_config(strategy_path),
        strategy_sha256=strategy_file_sha256(strategy_path),
    ).plan(planning)
    frozen_path = tmp_path / f"{session_date}-frozen.json"
    frozen.write(frozen_path)
    report = ReplaySessionAggregator().evaluate(
        evidence=(),
        round_trips=(),
        session_date=session_date,
        initial_cash=100_000,
    )
    report_path = tmp_path / f"{session_date}-replay.json"
    report.write(report_path)
    return frozen_path, report_path


def test_prepare_breaker_controls_keeps_control_and_replay_dates_distinct(
    tmp_path: Path,
) -> None:
    first = _source(tmp_path, date(2026, 7, 27))
    latest = _source(tmp_path, date(2026, 7, 28))
    output = tmp_path / "breaker-controls.json"
    evaluated = datetime(2026, 7, 29, 13, 20, tzinfo=UTC)
    arguments = [
        "monitoring",
        "prepare-breaker-controls",
        "--control-date",
        "2026-07-29",
        "--evaluated-at",
        evaluated.isoformat(),
        "--polygon-data-observed-at",
        (evaluated - timedelta(minutes=1)).isoformat(),
        "--alpaca-data-observed-at",
        (evaluated - timedelta(minutes=2)).isoformat(),
        "--output",
        str(output),
    ]
    for frozen, report in (first, latest):
        arguments.extend(("--frozen-orders", str(frozen)))
        arguments.extend(("--replay-report", str(report)))

    result = CliRunner().invoke(app, arguments)
    repeated = CliRunner().invoke(app, arguments)

    assert result.exit_code == 0
    assert repeated.exit_code == 0
    spec = CircuitBreakerEvaluationSpec.model_validate_json(output.read_bytes())
    assert tuple(item.session_date for item in spec.observations) == (
        date(2026, 7, 27),
        date(2026, 7, 29),
    )
    assert tuple(item.replay_source_date for item in spec.observations) == (
        date(2026, 7, 27),
        date(2026, 7, 28),
    )
    decision = CircuitBreakerEvaluator().evaluate(
        tuple(item.to_domain() for item in spec.observations)
    )
    assert decision.replay_source_dates == (
        date(2026, 7, 27),
        date(2026, 7, 28),
    )
    assert json.loads(result.stdout)["control_date"] == "2026-07-29"
