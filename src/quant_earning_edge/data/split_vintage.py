"""Causal split normalization for provider-unadjusted historical daily bars."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence
    from datetime import date

    from quant_earning_edge.data.clients import EquityBar, StockSplit

_PRICE_FIELDS = ("open", "high", "low", "close", "vwap")


def causally_adjust_daily_bar_rows(
    rows: Sequence[dict[str, Any]],
    *,
    splits: Sequence[StockSplit],
    basis_date: date,
) -> tuple[dict[str, Any], ...]:
    """Return bars on the share basis established by splits executed by ``basis_date``."""
    identifiers = tuple(item.event_id for item in splits)
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("split history contains duplicate event identifiers")
    relevant: dict[str, tuple[StockSplit, ...]] = {}
    for symbol in {str(row["symbol"]).strip().upper() for row in rows}:
        relevant[symbol] = tuple(
            sorted(
                (
                    item
                    for item in splits
                    if item.symbol.strip().upper() == symbol and item.execution_date <= basis_date
                ),
                key=lambda item: (item.execution_date, item.event_id),
            )
        )
    output = []
    for source in rows:
        row = dict(source)
        session_date = row["session_date"]
        if session_date > basis_date:
            raise ValueError("daily bar occurs after the requested split-adjustment basis")
        if row["adjusted"]:
            output.append(row)
            continue
        factor = math.prod(
            item.split_from / item.split_to
            for item in relevant[str(row["symbol"]).strip().upper()]
            if session_date < item.execution_date
        )
        if not math.isfinite(factor) or factor <= 0:
            raise ValueError("split history produced an invalid adjustment factor")
        for field in _PRICE_FIELDS:
            if field in row and row[field] is not None:
                row[field] = float(row[field]) * factor
        row["volume"] = float(row["volume"]) / factor
        row["adjusted"] = True
        output.append(row)
    return tuple(output)


def causally_adjust_equity_bars(
    bars: Sequence[EquityBar],
    *,
    splits: Sequence[StockSplit],
    basis_date: date,
) -> tuple[EquityBar, ...]:
    """Normalize typed provider bars to the split basis known at one session."""
    from quant_earning_edge.data.clients import EquityBar  # noqa: PLC0415

    rows = tuple(
        {
            **bar.model_dump(),
            "session_date": bar.session_date,
        }
        for bar in bars
    )
    adjusted = causally_adjust_daily_bar_rows(
        rows,
        splits=splits,
        basis_date=basis_date,
    )
    return tuple(
        EquityBar.model_validate(
            {key: value for key, value in row.items() if key != "session_date"}
        )
        for row in adjusted
    )
