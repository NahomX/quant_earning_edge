"""Fail-closed circuit breakers for the paper-order workflow."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
from pathlib import Path  # noqa: TC003 - Pydantic resolves annotations.
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field, ValidationError

if TYPE_CHECKING:
    from collections.abc import Sequence

DAILY_LOSS_LIMIT = 0.02
LOW_FILL_RATE_LIMIT = 0.70
LOW_FILL_SESSION_LIMIT = 3
DATA_FRESHNESS_LIMIT = timedelta(minutes=30)
RECONCILIATION_T_PLUS_ONE_AGE = 1

DAILY_REPLAY_LOSS = "daily_replay_loss_above_2_percent"
THREE_DAY_LOW_FILL = "three_consecutive_replay_fill_rates_below_70_percent"
POLYGON_STALE = "polygon_data_more_than_30_minutes_stale"
ALPACA_STALE = "alpaca_data_more_than_30_minutes_stale"
RECONCILIATION_OVERDUE = "reconciliation_break_unresolved_at_t_plus_one_close"


@dataclass(frozen=True)
class CircuitBreakerObservation:
    """One close-time operational observation used by the breaker evaluator.

    ``reconciliation_break_age_sessions`` counts completed market closes after
    the close where the break was first detected: zero at T and one at T+1.
    """

    session_date: date
    evaluated_at: datetime
    replay_notional: float
    replay_net_pnl: float | None
    replay_fill_rate: float | None
    polygon_data_observed_at: datetime | None
    alpaca_data_observed_at: datetime | None
    reconciliation_break_age_sessions: int | None = None
    replay_source_date: date | None = None

    def __post_init__(self) -> None:
        source_date = self.replay_source_date or self.session_date
        if source_date > self.session_date:
            raise ValueError("replay source date cannot be after the control session date")
        object.__setattr__(self, "replay_source_date", source_date)
        _require_aware("evaluated_at", self.evaluated_at)
        if not math.isfinite(self.replay_notional) or self.replay_notional < 0:
            raise ValueError("replay_notional must be finite and non-negative")
        if self.replay_net_pnl is not None and not math.isfinite(self.replay_net_pnl):
            raise ValueError("replay_net_pnl must be finite when provided")
        if self.replay_notional == 0 and self.replay_net_pnl not in {None, 0.0}:
            raise ValueError("zero replay notional cannot have non-zero P&L")
        if self.replay_fill_rate is not None and (
            not math.isfinite(self.replay_fill_rate) or not 0.0 <= self.replay_fill_rate <= 1.0
        ):
            raise ValueError("replay_fill_rate must be within [0, 1]")
        for name, observed_at in (
            ("polygon_data_observed_at", self.polygon_data_observed_at),
            ("alpaca_data_observed_at", self.alpaca_data_observed_at),
        ):
            if observed_at is None:
                continue
            _require_aware(name, observed_at)
            if observed_at > self.evaluated_at:
                raise ValueError(f"{name} cannot be after evaluated_at")
        if (
            self.reconciliation_break_age_sessions is not None
            and self.reconciliation_break_age_sessions < 0
        ):
            raise ValueError("reconciliation break age cannot be negative")


@dataclass(frozen=True)
class CircuitBreakerDecision:
    """Immutable fail-closed decision controlling submission of new orders."""

    schema_version: int
    session_date: date
    evaluated_at: datetime
    observation_dates: tuple[date, ...]
    replay_source_dates: tuple[date, ...]
    halt_new_orders: bool
    triggered_breakers: tuple[str, ...]
    replay_loss_fraction: float | None
    consecutive_low_fill_sessions: int
    polygon_freshness_minutes: float | None
    alpaca_freshness_minutes: float | None
    reconciliation_break_age_sessions: int | None

    def __post_init__(self) -> None:
        _require_aware("evaluated_at", self.evaluated_at)
        if not self.observation_dates or self.observation_dates[-1] != self.session_date:
            raise ValueError("decision must identify the current observation date")
        if len(self.replay_source_dates) != len(self.observation_dates) or any(
            source > control
            for source, control in zip(
                self.replay_source_dates,
                self.observation_dates,
                strict=True,
            )
        ):
            raise ValueError("decision replay-source provenance is inconsistent")
        if self.halt_new_orders != bool(self.triggered_breakers):
            raise ValueError("halt flag must exactly match triggered breakers")
        if tuple(sorted(set(self.triggered_breakers))) != self.triggered_breakers:
            raise ValueError("triggered breakers must be unique and sorted")

    @property
    def canonical_bytes(self) -> bytes:
        return json.dumps(
            asdict(self),
            default=lambda item: item.isoformat(),
            sort_keys=True,
            separators=(",", ":"),
        ).encode()

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes).hexdigest()

    def write(self, output: Path) -> None:
        encoded = self.canonical_bytes
        output.parent.mkdir(parents=True, exist_ok=True)
        try:
            with output.open("xb") as destination:
                destination.write(encoded)
        except FileExistsError:
            if output.read_bytes() != encoded:
                raise RuntimeError(f"circuit-breaker decision collision at {output}") from None

    @classmethod
    def load(cls, path: Path) -> CircuitBreakerDecision:
        """Load a strict canonical decision and re-run domain validation."""
        try:
            raw = json.loads(path.read_bytes())
            decision = CircuitBreakerDecisionSpec.model_validate(raw).to_domain()
        except (json.JSONDecodeError, OSError, TypeError, ValidationError, ValueError) as error:
            raise ValueError(f"invalid circuit-breaker decision: {path}") from error
        if json.loads(decision.canonical_bytes) != raw:
            raise ValueError("circuit-breaker decision is not canonical or uses unsupported fields")
        return decision


class _StrictSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CircuitBreakerObservationSpec(_StrictSpec):
    """Strict JSON boundary for an operational observation."""

    session_date: date
    evaluated_at: datetime
    replay_notional: float = Field(ge=0)
    replay_net_pnl: float | None
    replay_fill_rate: float | None = Field(default=None, ge=0, le=1)
    polygon_data_observed_at: datetime | None
    alpaca_data_observed_at: datetime | None
    reconciliation_break_age_sessions: int | None = Field(default=None, ge=0)
    replay_source_date: date | None = None

    def to_domain(self) -> CircuitBreakerObservation:
        return CircuitBreakerObservation(**self.model_dump())


class CircuitBreakerEvaluationSpec(_StrictSpec):
    """Ordered session history required to decide whether submissions halt."""

    observations: tuple[CircuitBreakerObservationSpec, ...] = Field(min_length=1)


class CircuitBreakerDecisionSpec(_StrictSpec):
    """Strict loader schema for immutable breaker decisions."""

    schema_version: int = Field(ge=1)
    session_date: date
    evaluated_at: datetime
    observation_dates: tuple[date, ...] = Field(min_length=1)
    replay_source_dates: tuple[date, ...] = Field(min_length=1)
    halt_new_orders: bool
    triggered_breakers: tuple[str, ...]
    replay_loss_fraction: float | None = Field(default=None, ge=0)
    consecutive_low_fill_sessions: int = Field(ge=0)
    polygon_freshness_minutes: float | None = Field(default=None, ge=0)
    alpaca_freshness_minutes: float | None = Field(default=None, ge=0)
    reconciliation_break_age_sessions: int | None = Field(default=None, ge=0)

    def to_domain(self) -> CircuitBreakerDecision:
        return CircuitBreakerDecision(**self.model_dump())


class CircuitBreakerEvaluator:
    """Evaluate the documented loss, fill, freshness, and reconciliation halts."""

    def evaluate(
        self,
        observations: Sequence[CircuitBreakerObservation],
    ) -> CircuitBreakerDecision:
        if not observations:
            raise ValueError("at least one circuit-breaker observation is required")
        dates = tuple(item.session_date for item in observations)
        if dates != tuple(sorted(set(dates))):
            raise ValueError("observations must have unique, strictly increasing session dates")

        current = observations[-1]
        loss_fraction = _loss_fraction(current)
        low_fill_streak = _low_fill_streak(observations)
        polygon_freshness = _freshness_minutes(
            evaluated_at=current.evaluated_at,
            observed_at=current.polygon_data_observed_at,
        )
        alpaca_freshness = _freshness_minutes(
            evaluated_at=current.evaluated_at,
            observed_at=current.alpaca_data_observed_at,
        )

        triggered: list[str] = []
        if loss_fraction is not None and loss_fraction > DAILY_LOSS_LIMIT:
            triggered.append(DAILY_REPLAY_LOSS)
        if low_fill_streak >= LOW_FILL_SESSION_LIMIT:
            triggered.append(THREE_DAY_LOW_FILL)
        if (
            polygon_freshness is None
            or polygon_freshness > DATA_FRESHNESS_LIMIT.total_seconds() / 60
        ):
            triggered.append(POLYGON_STALE)
        if alpaca_freshness is None or alpaca_freshness > DATA_FRESHNESS_LIMIT.total_seconds() / 60:
            triggered.append(ALPACA_STALE)
        if (
            current.reconciliation_break_age_sessions is not None
            and current.reconciliation_break_age_sessions >= RECONCILIATION_T_PLUS_ONE_AGE
        ):
            triggered.append(RECONCILIATION_OVERDUE)
        reasons = tuple(sorted(triggered))
        return CircuitBreakerDecision(
            schema_version=2,
            session_date=current.session_date,
            evaluated_at=current.evaluated_at,
            observation_dates=dates,
            replay_source_dates=tuple(
                item.replay_source_date or item.session_date for item in observations
            ),
            halt_new_orders=bool(reasons),
            triggered_breakers=reasons,
            replay_loss_fraction=loss_fraction,
            consecutive_low_fill_sessions=low_fill_streak,
            polygon_freshness_minutes=polygon_freshness,
            alpaca_freshness_minutes=alpaca_freshness,
            reconciliation_break_age_sessions=current.reconciliation_break_age_sessions,
        )


def _require_aware(name: str, value: datetime) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")


def _loss_fraction(observation: CircuitBreakerObservation) -> float | None:
    if observation.replay_notional == 0 or observation.replay_net_pnl is None:
        return None
    return max(0.0, -observation.replay_net_pnl / observation.replay_notional)


def _low_fill_streak(observations: Sequence[CircuitBreakerObservation]) -> int:
    streak = 0
    for observation in reversed(observations):
        if (
            observation.replay_fill_rate is None
            or observation.replay_fill_rate >= LOW_FILL_RATE_LIMIT
        ):
            break
        streak += 1
    return streak


def _freshness_minutes(
    *,
    evaluated_at: datetime,
    observed_at: datetime | None,
) -> float | None:
    if observed_at is None:
        return None
    return (evaluated_at - observed_at).total_seconds() / 60.0
