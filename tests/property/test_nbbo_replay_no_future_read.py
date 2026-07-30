"""NBBO replay must not consume market state from before the causal boundary."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest

from quant_earning_edge.backtest.nbbo_replay import (
    DecisionSnapshot,
    IntendedOrder,
    NbboQuote,
    TradePrint,
    replay_order,
)


def _inputs() -> tuple[
    IntendedOrder,
    DecisionSnapshot,
    tuple[NbboQuote, ...],
    tuple[TradePrint, ...],
]:
    decision = datetime(2025, 1, 2, 21, 30, tzinfo=UTC)
    submitted = datetime(2025, 1, 3, 14, 30, tzinfo=UTC)
    order = IntendedOrder(
        order_id="causal-order",
        ticker="AAA",
        side="buy",
        quantity=100,
        decision_time=decision,
        submitted_at=submitted,
        expires_at=submitted + timedelta(hours=6, minutes=30),
        average_daily_volume_shares=1_000_000,
        aggressiveness="mid",
        limit_price=101.0,
    )
    snapshot = DecisionSnapshot(
        ticker="AAA",
        observed_at=decision,
        bid_price=99.9,
        ask_price=100.1,
        bid_size=100,
        ask_size=100,
        last_trade_price=100.0,
        last_trade_at=decision - timedelta(seconds=1),
    )
    quotes = (
        NbboQuote("AAA", decision - timedelta(seconds=1), 1, 1.0, 2.0, 9999, 9999),
        NbboQuote("AAA", submitted - timedelta(seconds=1), 2, 2.0, 3.0, 9999, 9999),
        NbboQuote("AAA", submitted, 3, 100.0, 100.2, 100, 100),
    )
    trades = (
        TradePrint("AAA", decision - timedelta(seconds=1), 1, 1.5, 9999),
        TradePrint("AAA", submitted - timedelta(seconds=1), 2, 2.5, 9999),
        TradePrint("AAA", submitted + timedelta(seconds=1), 3, 100.1, 300),
    )
    return order, snapshot, quotes, trades


@pytest.mark.property
def test_replay_never_reads_quotes_before_decision_time() -> None:
    order, snapshot, quotes, trades = _inputs()

    result = replay_order(
        order,
        decision_snapshot=snapshot,
        quotes=quotes,
        trades=trades,
    )

    assert result.read_quote_timestamps
    assert result.read_trade_timestamps
    assert min(result.read_quote_timestamps) >= order.submitted_at >= order.decision_time
    assert min(result.read_trade_timestamps) >= order.submitted_at >= order.decision_time


@pytest.mark.property
def test_predecision_market_values_cannot_change_replay() -> None:
    order, snapshot, quotes, trades = _inputs()
    expected = replay_order(
        order,
        decision_snapshot=snapshot,
        quotes=quotes,
        trades=trades,
    )
    mutated_quotes = (
        replace(quotes[0], bid_price=50_000.0, ask_price=50_001.0),
        replace(quotes[1], bid_price=60_000.0, ask_price=60_001.0),
        quotes[2],
    )
    mutated_trades = (
        replace(trades[0], price=50_000.0, size=1_000_000),
        replace(trades[1], price=60_000.0, size=1_000_000),
        trades[2],
    )

    actual = replay_order(
        order,
        decision_snapshot=snapshot,
        quotes=mutated_quotes,
        trades=mutated_trades,
    )

    assert actual == expected
