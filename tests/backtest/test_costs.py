"""Execution cost-model accounting tests."""

from __future__ import annotations

import pytest

from quant_earning_edge.backtest import CostModel, ExecutionCostInput


def test_cost_model_decomposes_documented_assumptions() -> None:
    result = CostModel().estimate(
        ExecutionCostInput(
            side="buy",
            shares=100,
            price=100.0,
            average_daily_volume_shares=1_000_000,
        )
    )

    assert result.notional == 10_000.0
    assert result.commission == 1.0
    assert result.half_spread == 2.0
    assert result.market_impact == pytest.approx(0.05)
    assert result.borrow == 0.0
    assert result.stop_slippage == 0.0
    assert result.total == pytest.approx(3.05)
    assert result.total_bps == pytest.approx(3.05)


@pytest.mark.parametrize(
    ("price", "expected_bps"),
    [(50.01, 2.0), (50.0, 5.0), (10.0, 5.0), (9.99, 15.0)],
)
def test_half_spread_price_tiers(price: float, expected_bps: float) -> None:
    result = CostModel().estimate(
        ExecutionCostInput(
            side="buy",
            shares=100,
            price=price,
            average_daily_volume_shares=1_000_000,
        )
    )

    assert result.half_spread / result.notional * 10_000 == pytest.approx(expected_bps)


def test_short_borrow_and_sell_stop_slippage_are_attributed() -> None:
    result = CostModel().estimate(
        ExecutionCostInput(
            side="sell",
            shares=200,
            price=25.0,
            average_daily_volume_shares=500_000,
            is_short_position=True,
            holding_days=10,
            triggered_stop_price=23.0,
            atr5=1.25,
        )
    )

    assert result.borrow == pytest.approx(5_000 * 0.005 * 10 / 252)
    assert result.stop_slippage == 250.0


def test_buy_stop_is_rejected_by_documented_long_stop_model() -> None:
    with pytest.raises(ValueError, match="sell stops"):
        ExecutionCostInput(
            side="buy",
            shares=100,
            price=25.0,
            average_daily_volume_shares=500_000,
            triggered_stop_price=27.0,
            atr5=1.0,
        )
