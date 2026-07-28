"""Causal pre-market gap feature."""

from quant_earning_edge.features.registry import FeatureContext, feature


@feature(
    name="premarket_gap_pct",
    lookback_days=1,
    required_observations=1,
    update_cadence="intraday",
    dependencies=("silver_daily_bars", "silver_minute_bars"),
)
def premarket_gap_pct(context: FeatureContext) -> float:
    """Latest completed pre-market minute close versus the prior close."""
    observations = context.premarket_history()
    if not observations:
        raise ValueError("premarket_gap_pct requires a completed pre-market observation")
    prior_close = context.price_history(observations=1)[0].close
    return observations[-1].close / prior_close - 1.0
