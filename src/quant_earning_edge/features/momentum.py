"""Causal momentum features."""

from __future__ import annotations

from itertools import pairwise

from quant_earning_edge.features.registry import FeatureContext, feature

_BARS_DEPENDENCY = ("silver_daily_bars",)


@feature(
    name="rsi_14",
    lookback_days=14,
    required_observations=15,
    dependencies=_BARS_DEPENDENCY,
)
def rsi_14(context: FeatureContext) -> float:
    """Fourteen-period Wilder-style RSI from prior-close observations."""
    history = context.price_history(observations=15)
    changes = [current.close - previous.close for previous, current in pairwise(history)]
    average_gain = sum(max(change, 0.0) for change in changes) / 14.0
    average_loss = sum(max(-change, 0.0) for change in changes) / 14.0
    if average_loss == 0:
        return 100.0 if average_gain > 0 else 50.0
    relative_strength = average_gain / average_loss
    return 100.0 - 100.0 / (1.0 + relative_strength)


def _ema(values: list[float], span: int) -> list[float]:
    alpha = 2.0 / (span + 1.0)
    result = [values[0]]
    for value in values[1:]:
        result.append(alpha * value + (1.0 - alpha) * result[-1])
    return result


@feature(
    name="macd_signal_12_26_9",
    lookback_days=35,
    required_observations=35,
    dependencies=_BARS_DEPENDENCY,
)
def macd_signal_12_26_9(context: FeatureContext) -> float:
    """Scale-free MACD histogram using causal 12/26/9 exponential averages."""
    history = context.price_history(observations=35)
    closes = [item.close for item in history]
    fast = _ema(closes, 12)
    slow = _ema(closes, 26)
    macd = [fast_value - slow_value for fast_value, slow_value in zip(fast, slow, strict=True)]
    signal = _ema(macd, 9)
    return (macd[-1] - signal[-1]) / closes[-1]


@feature(
    name="distance_to_high_52w",
    lookback_days=252,
    required_observations=252,
    dependencies=_BARS_DEPENDENCY,
)
def distance_to_high_52w(context: FeatureContext) -> float:
    """Latest close divided by the trailing 252-session high, minus one."""
    history = context.price_history(observations=252)
    trailing_high = max(item.close for item in history)
    return history[-1].close / trailing_high - 1.0
