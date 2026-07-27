"""Property proof that every registered feature respects its as-of boundary."""

from __future__ import annotations

import math
from datetime import date, timedelta

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from quant_earning_edge.features import (
    FEATURE_REGISTRY,
    EarningsObservation,
    FeatureContext,
    FeatureSpec,
    PriceBar,
)

SPECS = FEATURE_REGISTRY.values()


@pytest.mark.property
@pytest.mark.parametrize("spec", SPECS, ids=lambda spec: spec.name)
@settings(max_examples=30, deadline=None)
@given(
    start_price=st.floats(min_value=5.0, max_value=500.0, allow_nan=False),
    returns=st.lists(
        st.floats(
            min_value=-0.08,
            max_value=0.08,
            allow_nan=False,
            allow_infinity=False,
        ),
        min_size=270,
        max_size=270,
    ),
)
def test_feature_is_point_in_time(
    spec: FeatureSpec,
    start_price: float,
    returns: list[float],
) -> None:
    """Appending observations after T cannot alter the exact value at T."""
    bars: list[PriceBar] = []
    price = start_price
    first_date = date(2025, 1, 1)
    for index, daily_return in enumerate(returns):
        price *= 1.0 + daily_return
        bars.append(
            PriceBar(
                session_date=first_date + timedelta(days=index),
                close=price,
                volume=1_000_000.0 + index,
                vwap=price * (1.0 + 0.001 * math.sin(index)),
            )
        )
    cutoff_index = 255
    asof_date = bars[cutoff_index].session_date
    target_date = asof_date + timedelta(days=1)
    known_earnings = (
        EarningsObservation(
            event_date=target_date - timedelta(days=90),
            effective_trade_date=target_date - timedelta(days=90),
            timing="amc",
            eps_actual=1.2,
            eps_estimate=1.0,
        ),
        EarningsObservation(
            event_date=target_date,
            effective_trade_date=target_date,
            timing="bmo",
        ),
    )
    truncated = FeatureContext(
        symbol="AAPL",
        asof_date=asof_date,
        bars=tuple(bars[: cutoff_index + 1]),
        target_date=target_date,
        earnings=known_earnings,
    )
    with_future = FeatureContext(
        symbol="AAPL",
        asof_date=asof_date,
        bars=tuple(bars),
        target_date=target_date,
        earnings=(
            *known_earnings,
            EarningsObservation(
                event_date=target_date + timedelta(days=30),
                effective_trade_date=target_date + timedelta(days=30),
                timing="bmo",
                eps_actual=9.0,
                eps_estimate=1.0,
            ),
        ),
    )

    assert spec.evaluate(truncated) == spec.evaluate(with_future)
