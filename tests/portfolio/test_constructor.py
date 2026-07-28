"""Fractional-Kelly and hard-cap property tests."""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from quant_earning_edge.portfolio import (
    FractionalKellyPortfolioConstructor,
    PortfolioConfig,
    ScoredCandidate,
    TradeOutcome,
)


def _outcomes(count: int = 60) -> tuple[TradeOutcome, ...]:
    first = date(2025, 1, 1)
    return tuple(
        TradeOutcome(
            closed_date=first + timedelta(days=index),
            net_return=0.04 if index % 2 == 0 else -0.01,
        )
        for index in range(count)
    )


def _candidates(count: int, *, one_sector: bool = False) -> tuple[ScoredCandidate, ...]:
    return tuple(
        ScoredCandidate(
            symbol=f"S{index:02d}",
            sector="Technology" if one_sector else f"Sector-{index % 3}",
            side="long" if index % 2 == 0 else "short",
            score=1.0 - index / 100,
            price=100.0,
        )
        for index in range(count)
    )


def test_constructor_enforces_position_sector_and_gross_caps() -> None:
    constructor = FractionalKellyPortfolioConstructor(PortfolioConfig(top_k=20, minimum_history=20))

    plan = constructor.construct(
        candidates=_candidates(20),
        outcomes=_outcomes(),
        equity=1_000_000,
        decision_date=date(2025, 4, 1),
    )

    assert plan.raw_kelly == pytest.approx(0.375)
    assert plan.fractional_kelly == pytest.approx(0.09375)
    assert plan.gross_weight <= 0.50
    assert all(item.target_weight <= 0.05 for item in plan.positions)
    assert all(weight <= 0.20 for _, weight in plan.sector_weights)


def test_sector_cap_prevents_concentration() -> None:
    plan = FractionalKellyPortfolioConstructor(
        PortfolioConfig(top_k=10, minimum_history=20)
    ).construct(
        candidates=_candidates(10, one_sector=True),
        outcomes=_outcomes(),
        equity=1_000_000,
        decision_date=date(2025, 4, 1),
    )

    assert len(plan.positions) == 4
    assert plan.gross_weight == pytest.approx(0.20, abs=0.0001)
    assert plan.sector_weights[0][0] == "Technology"
    assert plan.sector_weights[0][1] == pytest.approx(0.20, abs=0.0001)


def test_insufficient_or_one_sided_history_produces_zero_risk() -> None:
    constructor = FractionalKellyPortfolioConstructor(PortfolioConfig(minimum_history=20))
    candidates = _candidates(3)

    insufficient = constructor.construct(
        candidates=candidates,
        outcomes=_outcomes(19),
        equity=100_000,
        decision_date=date(2025, 4, 1),
    )
    all_winners = constructor.construct(
        candidates=candidates,
        outcomes=tuple(
            TradeOutcome(date(2025, 1, 1) + timedelta(days=index), 0.01) for index in range(20)
        ),
        equity=100_000,
        decision_date=date(2025, 4, 1),
    )

    assert insufficient.positions == ()
    assert all_winners.positions == ()


def test_future_outcomes_cannot_change_position_targets() -> None:
    constructor = FractionalKellyPortfolioConstructor(PortfolioConfig(minimum_history=20))
    decision_date = date(2025, 4, 1)
    known = _outcomes()
    future = tuple(TradeOutcome(decision_date + timedelta(days=index), 0.50) for index in range(10))

    clipped = constructor.construct(
        candidates=_candidates(6),
        outcomes=known,
        equity=100_000,
        decision_date=decision_date,
    )
    with_future = constructor.construct(
        candidates=_candidates(6),
        outcomes=(*known, *future),
        equity=100_000,
        decision_date=decision_date,
    )

    assert clipped == with_future
