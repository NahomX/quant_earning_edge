"""Deterministic 60-session cross-sectional momentum tests."""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from quant_earning_edge.signals import CrossSectionalMomentum, MomentumPrice


def _prices(*, include_future: bool = False) -> tuple[MomentumPrice, ...]:
    first = date(2025, 1, 2)
    rows = [
        MomentumPrice(
            symbol=f"S{symbol_index:02d}",
            session_date=first + timedelta(days=session_index),
            close=100.0 + symbol_index * session_index,
        )
        for symbol_index in range(10)
        for session_index in range(61)
    ]
    if include_future:
        rows.extend(
            MomentumPrice(
                symbol=f"S{symbol_index:02d}",
                session_date=first + timedelta(days=61),
                close=1_000_000.0 - symbol_index,
            )
            for symbol_index in range(10)
        )
    return tuple(rows)


def test_momentum_ranks_top_and_bottom_deterministically() -> None:
    asof_date = date(2025, 3, 3)

    signals = CrossSectionalMomentum(selection_fraction=0.2).generate(
        _prices(),
        asof_date=asof_date,
    )

    assert [item.symbol for item in signals[:2]] == ["S09", "S08"]
    assert all(item.side == 1 for item in signals[:2])
    assert all(item.side == 0 for item in signals[2:-2])
    assert all(item.side == -1 for item in signals[-2:])
    assert [item.rank for item in signals] == list(range(1, 11))


def test_future_prices_cannot_change_asof_signal() -> None:
    asof_date = date(2025, 3, 3)
    model = CrossSectionalMomentum(selection_fraction=0.2)

    clipped = model.generate(_prices(), asof_date=asof_date)
    with_future = model.generate(_prices(include_future=True), asof_date=asof_date)

    assert clipped == with_future


def test_momentum_requires_full_60_session_return_window() -> None:
    asof_date = date(2025, 3, 3)

    with pytest.raises(ValueError, match="61 observations"):
        CrossSectionalMomentum().generate(
            tuple(item for item in _prices() if item.session_date != date(2025, 1, 2)),
            asof_date=asof_date,
        )
