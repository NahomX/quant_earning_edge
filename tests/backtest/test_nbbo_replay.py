"""Unit coverage for deterministic NBBO and trade-print replay."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from quant_earning_edge.backtest.nbbo_replay import (
    DecisionSnapshot,
    IntendedOrder,
    NbboQuote,
    ReplayConfig,
    TradePrint,
    replay_order,
)


@pytest.fixture
def times() -> tuple[datetime, datetime, datetime]:
    decision = datetime(2025, 1, 2, 21, 30, tzinfo=UTC)
    submitted = datetime(2025, 1, 3, 14, 30, tzinfo=UTC)
    return decision, submitted, submitted + timedelta(hours=6, minutes=30)


def _snapshot(decision: datetime) -> DecisionSnapshot:
    return DecisionSnapshot(
        ticker="AAA",
        observed_at=decision,
        bid_price=99.9,
        ask_price=100.1,
        bid_size=1_000,
        ask_size=1_000,
        last_trade_price=100.0,
        last_trade_at=decision - timedelta(seconds=1),
    )


def _order(
    times: tuple[datetime, datetime, datetime],
    *,
    side: str = "buy",
    quantity: int = 100,
    aggressiveness: str = "aggressive",
    limit_price: float | None = None,
) -> IntendedOrder:
    decision, submitted, expires = times
    return IntendedOrder(
        order_id="order-1",
        ticker="aaa",
        side=side,  # type: ignore[arg-type]
        quantity=quantity,
        decision_time=decision,
        submitted_at=submitted,
        expires_at=expires,
        average_daily_volume_shares=1_000_000,
        aggressiveness=aggressiveness,  # type: ignore[arg-type]
        limit_price=limit_price,
    )


def test_aggressive_order_consumes_displayed_size_and_records_partial_fill(
    times: tuple[datetime, datetime, datetime],
) -> None:
    order = _order(times, quantity=150)
    decision, submitted, _ = times
    quotes = (
        NbboQuote("AAA", submitted, 1, 99.9, 100.1, 20, 40),
        NbboQuote("AAA", submitted + timedelta(seconds=1), 2, 100.0, 100.2, 20, 60),
    )

    result = replay_order(order, decision_snapshot=_snapshot(decision), quotes=quotes)

    assert result.filled_qty == 100
    assert result.unfilled_qty == 50
    assert result.fill_rate == pytest.approx(2 / 3)
    assert result.notes == "partial fill"
    assert [fragment.quantity for fragment in result.fragments] == [40, 60]
    assert result.fill_price is not None
    assert result.fill_price > 100.1
    assert result.slippage_bps_realized is not None
    assert result.slippage_bps_predicted > 0


def test_sell_replay_uses_adverse_direction_for_slippage(
    times: tuple[datetime, datetime, datetime],
) -> None:
    order = _order(times, side="sell")
    decision, submitted, _ = times
    quote = NbboQuote("AAA", submitted, 1, 99.8, 100.2, 100, 100)

    result = replay_order(order, decision_snapshot=_snapshot(decision), quotes=(quote,))

    assert result.filled_qty == 100
    assert result.fill_price is not None
    assert result.fill_price < quote.bid_price
    assert result.slippage_bps_realized is not None
    assert result.slippage_bps_realized > 0


def test_mid_limit_uses_probability_weighted_trades_and_opening_skew(
    times: tuple[datetime, datetime, datetime],
) -> None:
    order = _order(times, quantity=100, aggressiveness="mid", limit_price=100.2)
    decision, submitted, _ = times
    quote = NbboQuote("AAA", submitted, 1, 99.9, 100.1, 100, 100)
    trades = (
        TradePrint("AAA", submitted, 1, 100.1, 200, is_opening_auction=True),
        TradePrint("AAA", submitted + timedelta(seconds=1), 2, 100.15, 200),
    )

    result = replay_order(
        order,
        decision_snapshot=_snapshot(decision),
        quotes=(quote,),
        trades=trades,
    )

    assert result.fill_probability_assumption == 0.35
    assert result.opening_auction_filled_qty == 35
    assert result.filled_qty == 100
    assert result.fragments[0].source == "opening_auction"
    assert result.opening_auction_skew_bps is not None


def test_passive_limit_can_be_missed(
    times: tuple[datetime, datetime, datetime],
) -> None:
    order = _order(times, aggressiveness="passive", limit_price=99.0)
    decision, submitted, _ = times
    quote = NbboQuote("AAA", submitted, 1, 99.9, 100.1, 100, 100)
    trade = TradePrint("AAA", submitted + timedelta(seconds=1), 1, 100.0, 10_000)

    result = replay_order(
        order,
        decision_snapshot=_snapshot(decision),
        quotes=(quote,),
        trades=(trade,),
    )

    assert result.filled_qty == 0
    assert result.fill_price is None
    assert result.slippage_bps_realized is None
    assert result.notes == "missed fill"


def test_limit_fill_never_violates_limit_price(
    times: tuple[datetime, datetime, datetime],
) -> None:
    order = _order(times, aggressiveness="mid", limit_price=100.0)
    decision, submitted, _ = times
    quote = NbboQuote("AAA", submitted, 1, 99.8, 100.2, 100, 100)
    trade = TradePrint("AAA", submitted + timedelta(seconds=1), 1, 99.99, 1_000)

    result = replay_order(
        order,
        decision_snapshot=_snapshot(decision),
        quotes=(quote,),
        trades=(trade,),
        config=ReplayConfig(market_impact_bps_coefficient=200),
    )

    assert result.fill_price == 100.0


def test_future_decision_snapshot_is_rejected(
    times: tuple[datetime, datetime, datetime],
) -> None:
    order = _order(times)
    decision, _, _ = times
    snapshot = _snapshot(decision + timedelta(seconds=1))

    with pytest.raises(ValueError, match="not observable"):
        replay_order(order, decision_snapshot=snapshot, quotes=())
