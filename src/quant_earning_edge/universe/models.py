"""Typed contracts for point-in-time universe construction."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from datetime import date


class RejectionReason(StrEnum):
    """Stable, auditable reasons a security is excluded."""

    INACTIVE = "inactive"
    NOT_YET_LISTED = "not_yet_listed"
    DELISTED = "delisted"
    HALTED = "halted"
    EXCHANGE = "exchange_not_allowed"
    SECURITY_TYPE = "security_type_not_allowed"
    PRICE = "price_below_minimum"
    MARKET_CAP = "market_cap_below_minimum"
    ADV = "adv_below_minimum"


@dataclass(frozen=True)
class UniverseConfig:
    """Eligibility thresholds evaluated using prior-close observations."""

    min_price: float = 5.0
    min_market_cap_usd: float = 500_000_000
    min_avg_daily_volume: float = 1_000_000
    allowed_exchanges: frozenset[str] = frozenset({"XNYS", "XNAS", "XASE"})
    allowed_security_types: frozenset[str] = frozenset({"CS"})
    exclude_halts: bool = True

    def __post_init__(self) -> None:
        if self.min_price <= 0:
            raise ValueError("min_price must be positive")
        if self.min_market_cap_usd <= 0:
            raise ValueError("min_market_cap_usd must be positive")
        if self.min_avg_daily_volume < 0:
            raise ValueError("min_avg_daily_volume must not be negative")
        if not self.allowed_exchanges:
            raise ValueError("allowed_exchanges must not be empty")
        if not self.allowed_security_types:
            raise ValueError("allowed_security_types must not be empty")


@dataclass(frozen=True)
class CandidateObservation:
    """All eligibility inputs frozen as of a single prior-close date."""

    symbol: str
    asof_date: date
    close: float
    avg_daily_volume: float
    market_cap_usd: float
    primary_exchange: str
    security_type: str
    active: bool
    halted: bool
    sector: str = "UNCLASSIFIED"
    list_date: date | None = None
    delisted_date: date | None = None

    def __post_init__(self) -> None:
        normalized = self.symbol.strip().upper()
        if not normalized:
            raise ValueError("symbol must not be empty")
        object.__setattr__(self, "symbol", normalized)
        if self.close <= 0:
            raise ValueError("close must be positive")
        if self.avg_daily_volume < 0:
            raise ValueError("avg_daily_volume must not be negative")
        if self.market_cap_usd <= 0:
            raise ValueError("market_cap_usd must be positive")
        if not self.primary_exchange:
            raise ValueError("primary_exchange must not be empty")
        if not self.security_type:
            raise ValueError("security_type must not be empty")
        sector = self.sector.strip().upper()
        if not sector:
            raise ValueError("sector must not be empty")
        object.__setattr__(self, "sector", sector)


@dataclass(frozen=True)
class UniverseDecision:
    """Eligibility result retaining inputs and every rejection reason."""

    candidate: CandidateObservation
    eligible: bool
    rejection_reasons: tuple[RejectionReason, ...]


def sector_from_sic_code(sic_code: str | None) -> str:
    """Map authoritative SIC divisions to conservative portfolio-cap buckets."""
    if sic_code is None or not sic_code.strip().isdigit():
        return "UNCLASSIFIED"
    value = int(sic_code)
    divisions = (
        (100, 999, "AGRICULTURE"),
        (1000, 1499, "MINING"),
        (1500, 1799, "CONSTRUCTION"),
        (2000, 3999, "MANUFACTURING"),
        (4000, 4999, "TRANSPORTATION_UTILITIES"),
        (5000, 5199, "WHOLESALE"),
        (5200, 5999, "RETAIL"),
        (6000, 6799, "FINANCE"),
        (7000, 8999, "SERVICES"),
        (9100, 9729, "PUBLIC_ADMINISTRATION"),
    )
    return next(
        (name for lower, upper, name in divisions if lower <= value <= upper),
        "UNCLASSIFIED",
    )
