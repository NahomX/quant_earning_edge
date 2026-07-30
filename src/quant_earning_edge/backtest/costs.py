"""Fully decomposed execution and holding cost model."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

TradeSide = Literal["buy", "sell"]


@dataclass(frozen=True)
class CostModelConfig:
    """Documented default cost assumptions, expressed in basis points."""

    commission_bps_per_side: float = 1.0
    impact_coefficient_bps: float = 5.0
    borrow_bps_annualized: float = 50.0
    half_spread_by_price_tier: tuple[tuple[float, float], ...] = (
        (50.0, 2.0),
        (10.0, 5.0),
        (0.0, 15.0),
    )

    def __post_init__(self) -> None:
        if (
            min(
                self.commission_bps_per_side,
                self.impact_coefficient_bps,
                self.borrow_bps_annualized,
            )
            < 0
        ):
            raise ValueError("cost assumptions must be non-negative")
        minimums = tuple(item[0] for item in self.half_spread_by_price_tier)
        if (
            not minimums
            or minimums != tuple(sorted(set(minimums), reverse=True))
            or minimums[-1] != 0
        ):
            raise ValueError("spread tiers require unique descending floors ending at zero")
        if any(
            not math.isfinite(value) or value < 0
            for tier in self.half_spread_by_price_tier
            for value in tier
        ):
            raise ValueError("spread tier values must be finite and non-negative")


@dataclass(frozen=True)
class ExecutionCostInput:
    """Inputs for one intended execution."""

    side: TradeSide
    shares: int
    price: float
    average_daily_volume_shares: float
    is_short_position: bool = False
    holding_days: int = 0
    triggered_stop_price: float | None = None
    atr5: float | None = None

    def __post_init__(self) -> None:
        if self.shares < 1:
            raise ValueError("shares must be positive")
        if self.price <= 0 or not math.isfinite(self.price):
            raise ValueError("price must be finite and positive")
        if self.average_daily_volume_shares <= 0:
            raise ValueError("average daily volume must be positive")
        if self.holding_days < 0:
            raise ValueError("holding_days must not be negative")
        if (self.triggered_stop_price is None) != (self.atr5 is None):
            raise ValueError("triggered_stop_price and atr5 must be supplied together")
        if self.triggered_stop_price is not None and self.triggered_stop_price <= 0:
            raise ValueError("triggered_stop_price must be positive")
        if self.atr5 is not None and self.atr5 <= 0:
            raise ValueError("atr5 must be positive")
        if self.triggered_stop_price is not None and self.side != "sell":
            raise ValueError("documented stop slippage applies only to sell stops")


@dataclass(frozen=True)
class CostBreakdown:
    """Dollar cost attribution for one execution."""

    notional: float
    commission: float
    half_spread: float
    market_impact: float
    borrow: float
    stop_slippage: float

    @property
    def total(self) -> float:
        return (
            self.commission
            + self.half_spread
            + self.market_impact
            + self.borrow
            + self.stop_slippage
        )

    @property
    def total_bps(self) -> float:
        return self.total / self.notional * 10_000.0


class CostModel:
    """Apply the architecture's explicit commission/spread/impact assumptions."""

    def __init__(self, config: CostModelConfig | None = None) -> None:
        self._config = config or CostModelConfig()

    def estimate(self, order: ExecutionCostInput) -> CostBreakdown:
        """Return every cost component without netting away attribution."""
        notional = order.shares * order.price
        spread_bps = _half_spread_bps(
            order.price,
            tiers=self._config.half_spread_by_price_tier,
        )
        participation = order.shares / order.average_daily_volume_shares
        impact_bps = self._config.impact_coefficient_bps * math.sqrt(participation)
        borrow = (
            notional * self._config.borrow_bps_annualized / 10_000.0 * order.holding_days / 252.0
            if order.is_short_position
            else 0.0
        )
        stop_slippage = 0.0
        if order.triggered_stop_price is not None and order.atr5 is not None:
            modeled_fill = max(order.triggered_stop_price - order.atr5, 0.0)
            stop_slippage = abs(modeled_fill - order.triggered_stop_price) * order.shares
        return CostBreakdown(
            notional=notional,
            commission=notional * self._config.commission_bps_per_side / 10_000.0,
            half_spread=notional * spread_bps / 10_000.0,
            market_impact=notional * impact_bps / 10_000.0,
            borrow=borrow,
            stop_slippage=stop_slippage,
        )


def _half_spread_bps(
    price: float,
    *,
    tiers: tuple[tuple[float, float], ...],
) -> float:
    return next(spread_bps for minimum, spread_bps in tiers if price >= minimum)
