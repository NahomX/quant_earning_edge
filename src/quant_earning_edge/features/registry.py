"""Feature registry. Every feature must register itself here so PIT tests find it."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

UpdateCadence = Literal["daily", "intraday", "event-driven"]


@dataclass(frozen=True)
class FeatureSpec:
    """Metadata for a registered feature."""

    name: str
    func: Callable[..., Any]
    lookback_days: int
    update_cadence: UpdateCadence
    dependencies: tuple[str, ...] = field(default_factory=tuple)


FEATURE_REGISTRY: dict[str, FeatureSpec] = {}


def feature(
    *,
    name: str,
    lookback_days: int,
    update_cadence: UpdateCadence = "daily",
    dependencies: tuple[str, ...] = (),
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator that registers a feature with the global registry.

    Example:
        @feature(name="rsi_14", lookback_days=30)
        def rsi_14(ticker: str, asof_date: date, ...) -> float: ...
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        if name in FEATURE_REGISTRY:
            raise ValueError(f"Feature already registered: {name}")
        FEATURE_REGISTRY[name] = FeatureSpec(
            name=name,
            func=func,
            lookback_days=lookback_days,
            update_cadence=update_cadence,
            dependencies=dependencies,
        )
        return func

    return decorator
