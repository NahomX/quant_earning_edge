"""Vectorbt chassis determinism and accounting invariant tests."""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from quant_earning_edge.backtest import (
    DailyMark,
    TradeIntent,
    VectorbtBacktestEngine,
)


def _fixture() -> tuple[tuple[date, ...], tuple[DailyMark, ...], tuple[TradeIntent, ...]]:
    first = date(2025, 1, 2)
    sessions = tuple(first + timedelta(days=offset) for offset in range(4))
    marks = tuple(
        DailyMark(symbol=symbol, session_date=session, close=close)
        for symbol, closes in (
            ("AAA", (100.0, 105.0, 110.0, 108.0)),
            ("BBB", (50.0, 45.0, 40.0, 42.0)),
        )
        for session, close in zip(sessions, closes, strict=True)
    )
    trades = (
        TradeIntent(
            trade_id="long-aaa",
            symbol="AAA",
            side="long",
            entry_date=sessions[0],
            exit_date=sessions[2],
            shares=10,
            entry_price=100.0,
            exit_price=110.0,
            entry_average_daily_volume_shares=1_000_000,
            exit_average_daily_volume_shares=1_000_000,
            holding_sessions=2,
        ),
        TradeIntent(
            trade_id="short-bbb",
            symbol="BBB",
            side="short",
            entry_date=sessions[0],
            exit_date=sessions[2],
            shares=10,
            entry_price=50.0,
            exit_price=40.0,
            entry_average_daily_volume_shares=500_000,
            exit_average_daily_volume_shares=500_000,
            holding_sessions=2,
        ),
    )
    return sessions, marks, trades


def test_vectorbt_engine_reconciles_every_cost_and_final_equity() -> None:
    sessions, marks, trades = _fixture()

    result = VectorbtBacktestEngine().run(
        trades=trades,
        marks=marks,
        sessions=sessions,
        initial_cash=10_000.0,
    )

    expected_gross = 200.0
    total_cost = sum(item.total_cost for item in result.trades)
    assert result.engine.startswith("vectorbt-")
    assert len(result.input_sha256) == 64
    assert result.final_gross_equity == pytest.approx(10_000.0 + expected_gross)
    assert result.final_net_equity == pytest.approx(10_000.0 + expected_gross - total_cost)
    assert sum(item.gross_pnl for item in result.daily) == pytest.approx(expected_gross)
    assert sum(item.net_pnl for item in result.daily) == pytest.approx(expected_gross - total_cost)
    assert all(
        item.net_pnl == pytest.approx(item.gross_pnl - item.total_cost) for item in result.daily
    )
    assert result.daily[0].gross_exposure == 1_500.0
    assert result.daily[-1].gross_exposure == 0.0


def test_vectorbt_engine_is_deterministic() -> None:
    sessions, marks, trades = _fixture()
    engine = VectorbtBacktestEngine()

    first = engine.run(
        trades=trades,
        marks=marks,
        sessions=sessions,
        initial_cash=10_000.0,
    )
    second = engine.run(
        trades=trades,
        marks=marks,
        sessions=sessions,
        initial_cash=10_000.0,
    )

    assert first == second


def test_vectorbt_engine_rejects_missing_held_session_mark() -> None:
    sessions, marks, trades = _fixture()

    with pytest.raises(ValueError, match="missing AAA mark"):
        VectorbtBacktestEngine().run(
            trades=trades,
            marks=tuple(item for item in marks if item != marks[2]),
            sessions=sessions,
            initial_cash=10_000.0,
        )
