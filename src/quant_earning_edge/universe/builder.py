"""Point-in-time universe evaluation with explicit temporal invariants."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from quant_earning_edge.universe.models import RejectionReason, UniverseDecision

if TYPE_CHECKING:
    from datetime import date

    from quant_earning_edge.universe.models import CandidateObservation, UniverseConfig


@dataclass(frozen=True)
class UniverseSnapshot:
    """Frozen decisions for orders intended on ``trade_date``."""

    trade_date: date
    asof_date: date
    generated_at: datetime
    config: UniverseConfig
    decisions: tuple[UniverseDecision, ...]

    @property
    def eligible_symbols(self) -> tuple[str, ...]:
        """Return deterministic eligible symbols only."""
        return tuple(decision.candidate.symbol for decision in self.decisions if decision.eligible)


class UniverseBuilder:
    """Evaluate every candidate using only an explicitly prior date."""

    def __init__(self, config: UniverseConfig) -> None:
        self._config = config

    def build(
        self,
        *,
        trade_date: date,
        asof_date: date,
        candidates: tuple[CandidateObservation, ...],
        generated_at: datetime | None = None,
    ) -> UniverseSnapshot:
        """Build a snapshot while rejecting lookahead or mixed-date inputs."""
        if asof_date >= trade_date:
            raise ValueError("asof_date must be before trade_date")
        observed_at = generated_at or datetime.now(UTC)
        if observed_at.tzinfo is None or observed_at.utcoffset() is None:
            raise ValueError("generated_at must be timezone-aware")

        symbols: set[str] = set()
        decisions: list[UniverseDecision] = []
        for candidate in candidates:
            if candidate.asof_date != asof_date:
                raise ValueError(
                    f"{candidate.symbol} observation date {candidate.asof_date} "
                    f"does not match snapshot asof_date {asof_date}"
                )
            if candidate.symbol in symbols:
                raise ValueError(f"duplicate universe candidate: {candidate.symbol}")
            symbols.add(candidate.symbol)
            reasons = self._rejection_reasons(candidate)
            decisions.append(
                UniverseDecision(
                    candidate=candidate,
                    eligible=not reasons,
                    rejection_reasons=tuple(reasons),
                )
            )

        return UniverseSnapshot(
            trade_date=trade_date,
            asof_date=asof_date,
            generated_at=observed_at.astimezone(UTC),
            config=self._config,
            decisions=tuple(sorted(decisions, key=lambda decision: decision.candidate.symbol)),
        )

    def _rejection_reasons(
        self,
        candidate: CandidateObservation,
    ) -> list[RejectionReason]:
        reasons: list[RejectionReason] = []
        config = self._config
        if not candidate.active:
            reasons.append(RejectionReason.INACTIVE)
        if candidate.list_date is not None and candidate.list_date > candidate.asof_date:
            reasons.append(RejectionReason.NOT_YET_LISTED)
        if candidate.delisted_date is not None and candidate.delisted_date <= candidate.asof_date:
            reasons.append(RejectionReason.DELISTED)
        if config.exclude_halts and candidate.halted:
            reasons.append(RejectionReason.HALTED)
        if candidate.primary_exchange not in config.allowed_exchanges:
            reasons.append(RejectionReason.EXCHANGE)
        if candidate.security_type not in config.allowed_security_types:
            reasons.append(RejectionReason.SECURITY_TYPE)
        if candidate.close < config.min_price:
            reasons.append(RejectionReason.PRICE)
        if candidate.market_cap_usd < config.min_market_cap_usd:
            reasons.append(RejectionReason.MARKET_CAP)
        if candidate.avg_daily_volume < config.min_avg_daily_volume:
            reasons.append(RejectionReason.ADV)
        return reasons
