"""Causal split-vintage normalization tests."""

from datetime import date

import pytest

from quant_earning_edge.data import causally_adjust_daily_bar_rows
from quant_earning_edge.data.clients import (
    SplitAdjustmentType,
    StockSplit,
)


def _split(
    event_id: str,
    *,
    execution_date: date,
    split_from: float,
    split_to: float,
) -> StockSplit:
    return StockSplit(
        event_id=event_id,
        symbol="AAPL",
        execution_date=execution_date,
        adjustment_type=(
            SplitAdjustmentType.FORWARD_SPLIT
            if split_to > split_from
            else SplitAdjustmentType.REVERSE_SPLIT
        ),
        split_from=split_from,
        split_to=split_to,
    )


def test_unadjusted_bar_uses_only_splits_executed_by_requested_vintage() -> None:
    row = {
        "session_date": date(2025, 1, 2),
        "symbol": "AAPL",
        "open": 100.0,
        "high": 102.0,
        "low": 99.0,
        "close": 101.0,
        "vwap": 100.5,
        "volume": 1_000_000.0,
        "adjusted": False,
    }
    first = _split(
        "split-1",
        execution_date=date(2025, 2, 1),
        split_from=1,
        split_to=2,
    )
    future = _split(
        "split-2",
        execution_date=date(2025, 4, 1),
        split_from=10,
        split_to=1,
    )

    normalized = causally_adjust_daily_bar_rows(
        (row,),
        splits=(first, future),
        basis_date=date(2025, 3, 1),
    )[0]

    assert normalized["adjusted"] is True
    assert normalized["close"] == pytest.approx(50.5)
    assert normalized["vwap"] == pytest.approx(50.25)
    assert normalized["volume"] == pytest.approx(2_000_000)


def test_split_on_bar_session_does_not_readjust_post_split_trading() -> None:
    split = _split(
        "split-1",
        execution_date=date(2025, 2, 1),
        split_from=1,
        split_to=2,
    )
    normalized = causally_adjust_daily_bar_rows(
        (
            {
                "session_date": split.execution_date,
                "symbol": "AAPL",
                "close": 50.0,
                "vwap": None,
                "volume": 2_000_000.0,
                "adjusted": False,
            },
        ),
        splits=(split,),
        basis_date=split.execution_date,
    )[0]

    assert normalized["close"] == 50
    assert normalized["volume"] == 2_000_000


def test_duplicate_split_identifiers_fail_closed() -> None:
    split = _split(
        "duplicate",
        execution_date=date(2025, 2, 1),
        split_from=1,
        split_to=2,
    )
    with pytest.raises(ValueError, match="duplicate"):
        causally_adjust_daily_bar_rows(
            (
                {
                    "session_date": date(2025, 1, 1),
                    "symbol": "AAPL",
                    "close": 100.0,
                    "volume": 1_000_000.0,
                    "adjusted": False,
                },
            ),
            splits=(split, split),
            basis_date=date(2025, 2, 1),
        )
