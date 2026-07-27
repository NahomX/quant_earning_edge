"""Causal earnings-event features."""

from __future__ import annotations

from quant_earning_edge.features.registry import (
    EarningsObservation,
    FeatureContext,
    InsufficientHistoryError,
    feature,
)

_EVENT_DEPENDENCIES = ("silver_earnings_events", "gold_event_candidates")


def _current_event(context: FeatureContext) -> EarningsObservation:
    current = [
        item
        for item in context.earnings_history()
        if item.effective_trade_date == context.asof_date
    ]
    if len(current) != 1:
        raise ValueError(
            f"expected exactly one current earnings event for {context.symbol}; got {len(current)}"
        )
    return current[0]


def _prior_event(context: FeatureContext) -> EarningsObservation:
    current = _current_event(context)
    prior = [
        item
        for item in context.earnings_history()
        if item.event_date < current.event_date
        and item.eps_actual is not None
        and item.eps_estimate is not None
    ]
    if not prior:
        raise InsufficientHistoryError(f"{context.symbol} has no prior reported earnings")
    return max(prior, key=lambda item: item.event_date)


@feature(
    name="earnings_timing_flag",
    lookback_days=1,
    required_observations=1,
    update_cadence="event-driven",
    dependencies=_EVENT_DEPENDENCIES,
)
def earnings_timing_flag(context: FeatureContext) -> float:
    """Encode BMO as +1 and prior-session AMC as -1."""
    timing = _current_event(context).timing
    if timing == "bmo":
        return 1.0
    if timing == "amc":
        return -1.0
    raise ValueError("during-market-hours earnings are not supported candidates")


@feature(
    name="days_since_last_earnings",
    lookback_days=120,
    required_observations=1,
    update_cadence="event-driven",
    dependencies=_EVENT_DEPENDENCIES,
)
def days_since_last_earnings(context: FeatureContext) -> float:
    """Calendar days between the current and preceding reported event."""
    current = _current_event(context)
    prior = _prior_event(context)
    return float((current.event_date - prior.event_date).days)


@feature(
    name="prior_eps_surprise_pct",
    lookback_days=120,
    required_observations=1,
    update_cadence="event-driven",
    dependencies=_EVENT_DEPENDENCIES,
)
def prior_eps_surprise_pct(context: FeatureContext) -> float:
    """Most recent reported EPS surprise, never the current event's result."""
    prior = _prior_event(context)
    assert prior.eps_actual is not None
    assert prior.eps_estimate is not None
    if prior.eps_estimate == 0:
        raise ValueError("prior EPS estimate must be non-zero")
    return (prior.eps_actual - prior.eps_estimate) / abs(prior.eps_estimate)
