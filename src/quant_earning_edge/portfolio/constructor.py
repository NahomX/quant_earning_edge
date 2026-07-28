"""Deterministic fractional-Kelly portfolio construction with hard caps."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from collections.abc import Sequence
    from datetime import date

PositionSide = Literal["long", "short"]


@dataclass(frozen=True)
class TradeOutcome:
    """One realized strategy return available before the next decision."""

    closed_date: date
    net_return: float

    def __post_init__(self) -> None:
        if self.net_return <= -1 or not math.isfinite(self.net_return):
            raise ValueError("net_return must be finite and greater than -1")


@dataclass(frozen=True)
class ScoredCandidate:
    """One point-in-time candidate ranked by model conviction."""

    symbol: str
    sector: str
    side: PositionSide
    score: float
    price: float

    def __post_init__(self) -> None:
        symbol = self.symbol.strip().upper()
        sector = self.sector.strip()
        if not symbol or not sector:
            raise ValueError("symbol and sector must not be empty")
        object.__setattr__(self, "symbol", symbol)
        object.__setattr__(self, "sector", sector)
        if not math.isfinite(self.score):
            raise ValueError("score must be finite")
        if self.price <= 0 or not math.isfinite(self.price):
            raise ValueError("price must be finite and positive")


@dataclass(frozen=True)
class PortfolioConfig:
    """Precommitted sizing and concentration limits."""

    top_k: int = 10
    kelly_fraction: float = 0.25
    history_window: int = 60
    minimum_history: int = 20
    max_position_weight: float = 0.05
    max_sector_weight: float = 0.20
    max_gross_weight: float = 0.50

    def __post_init__(self) -> None:
        if self.top_k < 1:
            raise ValueError("top_k must be positive")
        if self.history_window < 1 or not 1 <= self.minimum_history <= self.history_window:
            raise ValueError("history requirements are inconsistent")
        fractions = (
            self.kelly_fraction,
            self.max_position_weight,
            self.max_sector_weight,
            self.max_gross_weight,
        )
        if any(not 0 < value <= 1 for value in fractions):
            raise ValueError("Kelly and risk fractions must be in (0, 1]")
        if self.max_position_weight > self.max_sector_weight:
            raise ValueError("position cap must not exceed sector cap")
        if self.max_sector_weight > self.max_gross_weight:
            raise ValueError("sector cap must not exceed gross cap")


@dataclass(frozen=True)
class PositionTarget:
    """Integer-share target after all portfolio caps."""

    rank: int
    symbol: str
    sector: str
    side: PositionSide
    score: float
    shares: int
    target_notional: float
    target_weight: float


@dataclass(frozen=True)
class PortfolioPlan:
    """Auditable output of one construction decision."""

    equity: float
    history_count: int
    raw_kelly: float
    fractional_kelly: float
    positions: tuple[PositionTarget, ...]
    gross_weight: float
    sector_weights: tuple[tuple[str, float], ...]


class FractionalKellyPortfolioConstructor:
    """Rank candidates and allocate without breaching any hard risk cap."""

    def __init__(self, config: PortfolioConfig | None = None) -> None:
        self._config = config or PortfolioConfig()

    def construct(
        self,
        *,
        candidates: Sequence[ScoredCandidate],
        outcomes: Sequence[TradeOutcome],
        equity: float,
        decision_date: date,
    ) -> PortfolioPlan:
        """Return integer-share targets or a documented zero-risk plan."""
        if equity <= 0 or not math.isfinite(equity):
            raise ValueError("equity must be finite and positive")
        symbols = tuple(item.symbol for item in candidates)
        if len(symbols) != len(set(symbols)):
            raise ValueError("candidate symbols must be unique")
        history = tuple(
            sorted(
                (item for item in outcomes if item.closed_date < decision_date),
                key=lambda item: item.closed_date,
            )[-self._config.history_window :]
        )
        raw_kelly = _kelly(history, minimum_history=self._config.minimum_history)
        fractional_kelly = raw_kelly * self._config.kelly_fraction
        per_position_weight = min(
            fractional_kelly,
            self._config.max_position_weight,
        )
        ranked = sorted(candidates, key=lambda item: (-abs(item.score), item.symbol))[
            : self._config.top_k
        ]
        positions: list[PositionTarget] = []
        sector_weights: dict[str, float] = {}
        gross_weight = 0.0
        for rank, candidate in enumerate(ranked, start=1):
            available_weight = min(
                per_position_weight,
                self._config.max_gross_weight - gross_weight,
                self._config.max_sector_weight - sector_weights.get(candidate.sector, 0.0),
            )
            shares = math.floor(equity * max(available_weight, 0.0) / candidate.price)
            if shares < 1:
                continue
            notional = shares * candidate.price
            realized_weight = notional / equity
            positions.append(
                PositionTarget(
                    rank=rank,
                    symbol=candidate.symbol,
                    sector=candidate.sector,
                    side=candidate.side,
                    score=candidate.score,
                    shares=shares,
                    target_notional=notional,
                    target_weight=realized_weight,
                )
            )
            gross_weight += realized_weight
            sector_weights[candidate.sector] = (
                sector_weights.get(candidate.sector, 0.0) + realized_weight
            )
        return PortfolioPlan(
            equity=equity,
            history_count=len(history),
            raw_kelly=raw_kelly,
            fractional_kelly=fractional_kelly,
            positions=tuple(positions),
            gross_weight=gross_weight,
            sector_weights=tuple(sorted(sector_weights.items())),
        )


def _kelly(outcomes: Sequence[TradeOutcome], *, minimum_history: int) -> float:
    if len(outcomes) < minimum_history:
        return 0.0
    wins = [item.net_return for item in outcomes if item.net_return > 0]
    losses = [item.net_return for item in outcomes if item.net_return < 0]
    if not wins or not losses:
        return 0.0
    win_probability = len(wins) / len(outcomes)
    payoff = (sum(wins) / len(wins)) / abs(sum(losses) / len(losses))
    return max(0.0, win_probability - (1.0 - win_probability) / payoff)
