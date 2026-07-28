"""Deterministic cross-sectional 60-session momentum baseline."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from collections.abc import Sequence
    from datetime import date

SignalSide = Literal[-1, 0, 1]


@dataclass(frozen=True)
class MomentumPrice:
    """One adjusted close used by the baseline signal."""

    symbol: str
    session_date: date
    close: float

    def __post_init__(self) -> None:
        normalized = self.symbol.strip().upper()
        if not normalized:
            raise ValueError("symbol must not be empty")
        object.__setattr__(self, "symbol", normalized)
        if self.close <= 0 or not math.isfinite(self.close):
            raise ValueError("close must be finite and positive")


@dataclass(frozen=True)
class MomentumSignal:
    """Ranked cross-sectional momentum decision."""

    symbol: str
    asof_date: date
    score: float
    rank: int
    side: SignalSide


class CrossSectionalMomentum:
    """Rank 60-session adjusted returns with deterministic symbol tie-breaking."""

    def __init__(self, *, selection_fraction: float = 0.1) -> None:
        if not 0 < selection_fraction <= 0.5:
            raise ValueError("selection_fraction must be in (0, 0.5]")
        self._selection_fraction = selection_fraction

    def generate(
        self,
        prices: Sequence[MomentumPrice],
        *,
        asof_date: date,
    ) -> tuple[MomentumSignal, ...]:
        """Generate long top-fraction and short bottom-fraction signals."""
        grouped: dict[str, list[MomentumPrice]] = {}
        for item in prices:
            if item.session_date <= asof_date:
                grouped.setdefault(item.symbol, []).append(item)
        if len(grouped) < 2:
            raise ValueError("cross-sectional momentum requires at least two symbols")
        scores: list[tuple[str, float]] = []
        for symbol, observations in grouped.items():
            ordered = sorted(observations, key=lambda item: item.session_date)
            dates = [item.session_date for item in ordered]
            if dates != sorted(set(dates)):
                raise ValueError(f"duplicate price sessions for {symbol}")
            if len(ordered) < 61 or ordered[-1].session_date != asof_date:
                raise ValueError(f"{symbol} requires 61 observations ending at asof_date")
            history = ordered[-61:]
            scores.append((symbol, history[-1].close / history[0].close - 1.0))
        ranked = sorted(scores, key=lambda item: (-item[1], item[0]))
        selected_count = max(1, math.floor(len(ranked) * self._selection_fraction))
        if selected_count * 2 > len(ranked):
            raise ValueError("selection fraction leaves no neutral cross-section")
        return tuple(
            MomentumSignal(
                symbol=symbol,
                asof_date=asof_date,
                score=score,
                rank=index + 1,
                side=(
                    1
                    if index < selected_count
                    else -1
                    if index >= len(ranked) - selected_count
                    else 0
                ),
            )
            for index, (symbol, score) in enumerate(ranked)
        )
