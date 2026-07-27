"""Typed feature registry with an enforced point-in-time input boundary."""

from __future__ import annotations

import hashlib
import inspect
import math
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from collections.abc import Iterable
    from datetime import date

UpdateCadence = Literal["daily", "intraday", "event-driven"]
_FEATURE_NAME = re.compile(r"^[a-z][a-z0-9_]*$")


class InsufficientHistoryError(ValueError):
    """A feature cannot be computed from the available PIT observations."""


@dataclass(frozen=True)
class PriceBar:
    """Minimal adjusted daily-bar input used by baseline features."""

    session_date: date
    close: float
    volume: float
    vwap: float | None = None

    def __post_init__(self) -> None:
        if not math.isfinite(self.close) or self.close <= 0:
            raise ValueError("close must be finite and positive")
        if not math.isfinite(self.volume) or self.volume < 0:
            raise ValueError("volume must be finite and non-negative")
        if self.vwap is not None and (not math.isfinite(self.vwap) or self.vwap <= 0):
            raise ValueError("vwap must be finite and positive")


@dataclass(frozen=True)
class EarningsObservation:
    """One earnings event mapped to its first tradable session."""

    event_date: date
    effective_trade_date: date
    timing: Literal["bmo", "amc", "dmh"]
    eps_actual: float | None = None
    eps_estimate: float | None = None

    def __post_init__(self) -> None:
        if self.effective_trade_date < self.event_date:
            raise ValueError("effective_trade_date must not precede event_date")
        if self.timing not in {"bmo", "amc", "dmh"}:
            raise ValueError(f"invalid earnings timing: {self.timing!r}")
        for value in (self.eps_actual, self.eps_estimate):
            if value is not None and not math.isfinite(value):
                raise ValueError("earnings EPS values must be finite")


@dataclass(frozen=True)
class FeatureContext:
    """All observations plus the date boundary a feature is allowed to see."""

    symbol: str
    asof_date: date
    bars: tuple[PriceBar, ...]
    earnings: tuple[EarningsObservation, ...] = ()

    def __post_init__(self) -> None:
        normalized = self.symbol.strip().upper()
        if not normalized:
            raise ValueError("symbol must not be empty")
        object.__setattr__(self, "symbol", normalized)
        dates = tuple(item.session_date for item in self.bars)
        if dates != tuple(sorted(set(dates))):
            raise ValueError("bars must have unique ascending session dates")
        event_keys = tuple(
            (item.effective_trade_date, item.event_date, item.timing) for item in self.earnings
        )
        if event_keys != tuple(sorted(set(event_keys))):
            raise ValueError("earnings observations must have unique ascending event keys")

    def price_history(self, *, observations: int) -> tuple[PriceBar, ...]:
        """Return only the last N observations at or before ``asof_date``."""
        if observations < 1:
            raise ValueError("observations must be positive")
        known = tuple(item for item in self.bars if item.session_date <= self.asof_date)
        if len(known) < observations:
            raise InsufficientHistoryError(
                f"{self.symbol} has {len(known)} PIT bars; {observations} required"
            )
        return known[-observations:]

    def earnings_history(self) -> tuple[EarningsObservation, ...]:
        """Return only events effective at or before ``asof_date``."""
        return tuple(item for item in self.earnings if item.effective_trade_date <= self.asof_date)


FeatureFunction = Callable[[FeatureContext], float]


@dataclass(frozen=True)
class FeatureSpec:
    """Validated metadata and implementation identity for one feature."""

    name: str
    func: FeatureFunction
    lookback_days: int
    required_observations: int
    update_cadence: UpdateCadence
    code_hash: str
    dependencies: tuple[str, ...] = field(default_factory=tuple)

    def evaluate(self, context: FeatureContext) -> float:
        """Compute a finite scalar or fail loudly."""
        value = float(self.func(context))
        if not math.isfinite(value):
            raise ValueError(f"feature {self.name} produced a non-finite value")
        return value


class FeatureRegistry:
    """Ordered registry that rejects ambiguous feature definitions."""

    def __init__(self) -> None:
        self._specs: dict[str, FeatureSpec] = {}

    def register(self, spec: FeatureSpec) -> None:
        if spec.name in self._specs:
            raise ValueError(f"feature already registered: {spec.name}")
        self._specs[spec.name] = spec

    def get(self, name: str) -> FeatureSpec:
        try:
            return self._specs[name]
        except KeyError as error:
            raise KeyError(f"unknown feature: {name}") from error

    def values(self) -> tuple[FeatureSpec, ...]:
        return tuple(self._specs[name] for name in sorted(self._specs))

    def select(self, names: Iterable[str]) -> tuple[FeatureSpec, ...]:
        selected = tuple(self.get(name) for name in names)
        if not selected:
            raise ValueError("at least one feature must be selected")
        if len({item.name for item in selected}) != len(selected):
            raise ValueError("feature selection contains duplicates")
        return selected


FEATURE_REGISTRY = FeatureRegistry()


def feature(
    *,
    name: str,
    lookback_days: int,
    required_observations: int,
    update_cadence: UpdateCadence = "daily",
    dependencies: tuple[str, ...] = (),
) -> Callable[[FeatureFunction], FeatureFunction]:
    """Register a scalar feature and its reproducibility metadata."""
    if not _FEATURE_NAME.fullmatch(name):
        raise ValueError(f"invalid feature name: {name!r}")
    if lookback_days < 1 or required_observations < 1:
        raise ValueError("feature lookback and required observations must be positive")
    normalized_dependencies = tuple(sorted(set(dependencies)))

    def decorator(func: FeatureFunction) -> FeatureFunction:
        module = inspect.getmodule(func)
        source = inspect.getsource(module or func).encode()
        FEATURE_REGISTRY.register(
            FeatureSpec(
                name=name,
                func=func,
                lookback_days=lookback_days,
                required_observations=required_observations,
                update_cadence=update_cadence,
                code_hash=hashlib.sha256(source).hexdigest(),
                dependencies=normalized_dependencies,
            )
        )
        return func

    return decorator
