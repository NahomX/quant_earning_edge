"""Causal baseline price features."""

from __future__ import annotations

import math
import statistics
from itertools import pairwise

from quant_earning_edge.features.registry import FeatureContext, feature

_BARS_DEPENDENCY = ("silver_daily_bars",)


def _return(context: FeatureContext, sessions: int) -> float:
    history = context.price_history(observations=sessions + 1)
    return history[-1].close / history[0].close - 1.0


@feature(
    name="return_1d",
    lookback_days=1,
    required_observations=2,
    dependencies=_BARS_DEPENDENCY,
)
def return_1d(context: FeatureContext) -> float:
    """Prior-close one-session adjusted return."""
    return _return(context, 1)


@feature(
    name="return_5d",
    lookback_days=5,
    required_observations=6,
    dependencies=_BARS_DEPENDENCY,
)
def return_5d(context: FeatureContext) -> float:
    """Prior-close five-session adjusted return."""
    return _return(context, 5)


@feature(
    name="return_20d",
    lookback_days=20,
    required_observations=21,
    dependencies=_BARS_DEPENDENCY,
)
def return_20d(context: FeatureContext) -> float:
    """Prior-close twenty-session adjusted return."""
    return _return(context, 20)


def _realized_volatility(context: FeatureContext, sessions: int) -> float:
    history = context.price_history(observations=sessions + 1)
    returns = [current.close / previous.close - 1.0 for previous, current in pairwise(history)]
    return statistics.stdev(returns) * math.sqrt(252.0)


@feature(
    name="realized_vol_20d",
    lookback_days=20,
    required_observations=21,
    dependencies=_BARS_DEPENDENCY,
)
def realized_vol_20d(context: FeatureContext) -> float:
    """Annualized sample volatility over twenty causal returns."""
    return _realized_volatility(context, 20)


@feature(
    name="realized_vol_60d",
    lookback_days=60,
    required_observations=61,
    dependencies=_BARS_DEPENDENCY,
)
def realized_vol_60d(context: FeatureContext) -> float:
    """Annualized sample volatility over sixty causal returns."""
    return _realized_volatility(context, 60)


@feature(
    name="distance_to_vwap_20d",
    lookback_days=20,
    required_observations=20,
    dependencies=_BARS_DEPENDENCY,
)
def distance_to_vwap_20d(context: FeatureContext) -> float:
    """Prior close's distance from the trailing volume-weighted daily VWAP."""
    history = context.price_history(observations=20)
    total_volume = sum(item.volume for item in history)
    if total_volume <= 0:
        raise ValueError("distance_to_vwap_20d requires positive trailing volume")
    trailing_vwap = sum((item.vwap or item.close) * item.volume for item in history) / total_volume
    return history[-1].close / trailing_vwap - 1.0
