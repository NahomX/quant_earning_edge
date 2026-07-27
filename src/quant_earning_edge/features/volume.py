"""Causal local-level Kalman volume features."""

from __future__ import annotations

from quant_earning_edge.features.registry import FeatureContext, feature

_BARS_DEPENDENCY = ("silver_daily_bars",)


def _kalman_volume(context: FeatureContext, sessions: int) -> float:
    observations = [item.volume for item in context.price_history(observations=sessions)]
    estimate = observations[0]
    covariance = 1.0
    process_variance = 1.0 / sessions
    measurement_variance = 1.0
    for observation in observations[1:]:
        predicted_covariance = covariance + process_variance
        gain = predicted_covariance / (predicted_covariance + measurement_variance)
        estimate += gain * (observation - estimate)
        covariance = (1.0 - gain) * predicted_covariance
    return estimate


@feature(
    name="kalman_volume_7d",
    lookback_days=7,
    required_observations=7,
    dependencies=_BARS_DEPENDENCY,
)
def kalman_volume_7d(context: FeatureContext) -> float:
    """Final causal local-level Kalman estimate over seven volume observations."""
    return _kalman_volume(context, 7)


@feature(
    name="kalman_volume_30d",
    lookback_days=30,
    required_observations=30,
    dependencies=_BARS_DEPENDENCY,
)
def kalman_volume_30d(context: FeatureContext) -> float:
    """Final causal local-level Kalman estimate over thirty observations."""
    return _kalman_volume(context, 30)


@feature(
    name="relative_volume_30d",
    lookback_days=30,
    required_observations=30,
    dependencies=_BARS_DEPENDENCY,
)
def relative_volume_30d(context: FeatureContext) -> float:
    """Latest volume divided by its causal thirty-session Kalman estimate."""
    latest = context.price_history(observations=1)[0].volume
    baseline = _kalman_volume(context, 30)
    if baseline <= 0:
        raise ValueError("relative_volume_30d requires positive filtered volume")
    return latest / baseline
