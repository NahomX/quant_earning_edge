"""Standard metrics, bootstrap, and cost-attribution tests."""

from __future__ import annotations

import json
from datetime import date, timedelta
from typing import TYPE_CHECKING

import pytest

from quant_earning_edge.backtest import (
    BacktestResult,
    DailyMark,
    TradeIntent,
    VectorbtBacktestEngine,
)
from quant_earning_edge.evaluation import PerformanceEvaluator

if TYPE_CHECKING:
    from pathlib import Path


def _result() -> BacktestResult:
    first = date(2025, 1, 2)
    sessions = tuple(first + timedelta(days=offset) for offset in range(6))
    marks = tuple(
        DailyMark(symbol=symbol, session_date=session, close=close)
        for symbol, closes in (
            ("WIN", (100.0, 103.0, 106.0, 110.0, 110.0, 110.0)),
            ("LOSS", (100.0, 98.0, 96.0, 95.0, 95.0, 95.0)),
        )
        for session, close in zip(sessions, closes, strict=True)
    )
    trades = (
        TradeIntent(
            trade_id="winner",
            symbol="WIN",
            side="long",
            entry_date=sessions[0],
            exit_date=sessions[3],
            shares=10,
            entry_price=100.0,
            exit_price=110.0,
            entry_average_daily_volume_shares=1_000_000,
            exit_average_daily_volume_shares=1_000_000,
            holding_sessions=3,
        ),
        TradeIntent(
            trade_id="loser",
            symbol="LOSS",
            side="long",
            entry_date=sessions[1],
            exit_date=sessions[4],
            shares=10,
            entry_price=98.0,
            exit_price=95.0,
            entry_average_daily_volume_shares=1_000_000,
            exit_average_daily_volume_shares=1_000_000,
            holding_sessions=3,
        ),
    )
    return VectorbtBacktestEngine().run(
        trades=trades,
        marks=marks,
        sessions=sessions,
        initial_cash=10_000.0,
    )


def test_report_is_deterministic_and_cost_sharpe_reconciles() -> None:
    result = _result()
    evaluator = PerformanceEvaluator(bootstrap_resamples=1_000, seed=7)

    first = evaluator.evaluate(result)
    second = evaluator.evaluate(result)

    assert first == second
    assert first.trade_count == 2
    assert first.hit_rate == 0.5
    assert first.payoff is not None and first.payoff > 1
    assert first.max_drawdown >= 0
    assert first.bootstrap is not None
    assert first.bootstrap.resamples == 1_000
    assert sum(item.dollars for item in first.cost_attribution) == pytest.approx(
        result.final_gross_equity - result.final_net_equity
    )
    assert sum(item.marginal_sharpe_loss for item in first.cost_attribution) == pytest.approx(
        first.gross_sharpe - first.net_sharpe
    )


def test_report_persistence_is_content_addressed(tmp_path: Path) -> None:
    report = PerformanceEvaluator(bootstrap_resamples=100, seed=7).evaluate(_result())
    output = tmp_path / "evaluation" / "report.json"

    PerformanceEvaluator.write(report, output)
    PerformanceEvaluator.write(report, output)

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["trade_count"] == 2
    assert payload["bootstrap"]["seed"] == 7
    assert report.sha256
