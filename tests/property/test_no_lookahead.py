"""Property proof that every registered feature respects its as-of boundary."""

from __future__ import annotations

import math
from datetime import date, timedelta

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from quant_earning_edge.features import FEATURE_REGISTRY, FeatureContext, FeatureSpec, PriceBar

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
        min_size=80,
        max_size=80,
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
    cutoff_index = 64
    asof_date = bars[cutoff_index].session_date
    truncated = FeatureContext(
        symbol="AAPL",
        asof_date=asof_date,
        bars=tuple(bars[: cutoff_index + 1]),
    )
    with_future = FeatureContext(
        symbol="AAPL",
        asof_date=asof_date,
        bars=tuple(bars),
    )

    assert spec.evaluate(truncated) == spec.evaluate(with_future)
