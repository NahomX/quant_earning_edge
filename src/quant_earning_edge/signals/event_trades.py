"""Causal conversion of OOS probabilities into event-day backtest intents."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from datetime import date, datetime  # noqa: TC003 - Pydantic resolves types at runtime.
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

from quant_earning_edge.backtest import TradeIntent
from quant_earning_edge.portfolio import (
    FractionalKellyPortfolioConstructor,
    PortfolioPlan,
    ScoredCandidate,
    TradeOutcome,
)
from quant_earning_edge.signals.lgbm_model import OosPrediction

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path


@dataclass(frozen=True)
class EventExecutionObservation:
    """Decision-time sizing inputs plus later execution evidence."""

    row_index: int
    symbol: str
    sector: str
    asof_date: date
    trade_date: date
    decision_at: datetime
    sizing_price_observed_at: datetime
    sizing_price: float
    entry_at: datetime
    entry_price: float
    exit_at: datetime
    exit_price: float
    frozen_average_daily_volume_shares: float

    def __post_init__(self) -> None:
        symbol = self.symbol.strip().upper()
        sector = self.sector.strip()
        if not symbol or not sector:
            raise ValueError("symbol and sector must not be empty")
        object.__setattr__(self, "symbol", symbol)
        object.__setattr__(self, "sector", sector)
        timestamps = (
            self.decision_at,
            self.sizing_price_observed_at,
            self.entry_at,
            self.exit_at,
        )
        if any(item.tzinfo is None or item.utcoffset() is None for item in timestamps):
            raise ValueError("event execution timestamps must be timezone-aware")
        if self.asof_date >= self.trade_date:
            raise ValueError("asof_date must precede trade_date")
        if self.sizing_price_observed_at > self.decision_at:
            raise ValueError("sizing price was not known by decision_at")
        if not self.decision_at < self.entry_at < self.exit_at:
            raise ValueError("decision, entry, and exit timestamps are out of order")
        if self.entry_at.date() != self.trade_date or self.exit_at.date() != self.trade_date:
            raise ValueError("entry and exit timestamps must match trade_date")
        numeric = (
            self.sizing_price,
            self.entry_price,
            self.exit_price,
            self.frozen_average_daily_volume_shares,
        )
        if any(value <= 0 or not math.isfinite(value) for value in numeric):
            raise ValueError("prices and ADV must be finite and positive")


class _StrictSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")


class OosPredictionSpec(_StrictSpec):
    row_index: int = Field(ge=0)
    symbol: str
    asof_date: date
    probability_up: float = Field(ge=0, le=1)
    realized_label: int = Field(ge=0, le=1)

    def to_domain(self) -> OosPrediction:
        return OosPrediction(**self.model_dump())


class EventExecutionSpec(_StrictSpec):
    row_index: int = Field(ge=0)
    symbol: str
    sector: str
    asof_date: date
    trade_date: date
    decision_at: datetime
    sizing_price_observed_at: datetime
    sizing_price: float = Field(gt=0)
    entry_at: datetime
    entry_price: float = Field(gt=0)
    exit_at: datetime
    exit_price: float = Field(gt=0)
    frozen_average_daily_volume_shares: float = Field(gt=0)

    def to_domain(self) -> EventExecutionObservation:
        return EventExecutionObservation(**self.model_dump())


class TradeOutcomeSpec(_StrictSpec):
    closed_date: date
    net_return: float = Field(gt=-1)

    def to_domain(self) -> TradeOutcome:
        return TradeOutcome(**self.model_dump())


class EventTradePlanningSpec(_StrictSpec):
    """Serialized OOS backtest inputs for one event session."""

    equity: float = Field(gt=0)
    predictions: tuple[OosPredictionSpec, ...]
    observations: tuple[EventExecutionSpec, ...]
    outcomes: tuple[TradeOutcomeSpec, ...]

    def domain_inputs(
        self,
    ) -> tuple[
        float,
        tuple[OosPrediction, ...],
        tuple[EventExecutionObservation, ...],
        tuple[TradeOutcome, ...],
    ]:
        return (
            self.equity,
            tuple(item.to_domain() for item in self.predictions),
            tuple(item.to_domain() for item in self.observations),
            tuple(item.to_domain() for item in self.outcomes),
        )


@dataclass(frozen=True)
class PlannedEventTrades:
    """Portfolio targets and executable same-session round trips."""

    trade_date: date
    portfolio: PortfolioPlan
    intents: tuple[TradeIntent, ...]

    def to_json_bytes(self) -> bytes:
        """Serialize canonical decision and execution evidence."""
        return json.dumps(
            asdict(self),
            default=lambda item: item.isoformat(),
            sort_keys=True,
            separators=(",", ":"),
        ).encode()

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.to_json_bytes()).hexdigest()


class EventTradePlanner:
    """Select positive OOS probabilities and apply causal portfolio sizing."""

    def __init__(
        self,
        portfolio_constructor: FractionalKellyPortfolioConstructor | None = None,
        *,
        minimum_probability: float = 0.5,
    ) -> None:
        if not 0.5 <= minimum_probability < 1:
            raise ValueError("minimum_probability must be in [0.5, 1)")
        self._portfolio_constructor = portfolio_constructor or FractionalKellyPortfolioConstructor()
        self._minimum_probability = minimum_probability

    def plan(
        self,
        *,
        predictions: Sequence[OosPrediction],
        observations: Sequence[EventExecutionObservation],
        outcomes: Sequence[TradeOutcome],
        equity: float,
    ) -> PlannedEventTrades:
        """Create long-only event trades without sizing from future execution data."""
        if not predictions or not observations:
            raise ValueError("predictions and execution observations are required")
        observation_by_row = {item.row_index: item for item in observations}
        if len(observation_by_row) != len(observations):
            raise ValueError("execution observations contain duplicate row indices")
        if len({item.row_index for item in predictions}) != len(predictions):
            raise ValueError("predictions contain duplicate row indices")
        if set(observation_by_row) != {item.row_index for item in predictions}:
            raise ValueError("prediction and execution row-index sets differ")
        trade_dates = {item.trade_date for item in observations}
        decision_times = {item.decision_at for item in observations}
        if len(trade_dates) != 1 or len(decision_times) != 1:
            raise ValueError("one plan must contain one trade date and decision timestamp")
        trade_date = next(iter(trade_dates))
        candidates: list[ScoredCandidate] = []
        selected_observations: dict[str, EventExecutionObservation] = {}
        for prediction in predictions:
            observation = observation_by_row[prediction.row_index]
            if (
                prediction.symbol != observation.symbol
                or prediction.asof_date != observation.asof_date
            ):
                raise ValueError("prediction key does not match execution observation")
            if prediction.probability_up < self._minimum_probability:
                continue
            score = prediction.probability_up - 0.5
            candidates.append(
                ScoredCandidate(
                    symbol=prediction.symbol,
                    sector=observation.sector,
                    side="long",
                    score=score,
                    price=observation.sizing_price,
                )
            )
            selected_observations[prediction.symbol] = observation
        portfolio = self._portfolio_constructor.construct(
            candidates=candidates,
            outcomes=outcomes,
            equity=equity,
            decision_date=trade_date,
        )
        intents = tuple(
            _intent(
                position.symbol,
                shares=position.shares,
                score=position.score,
                observation=selected_observations[position.symbol],
            )
            for position in portfolio.positions
        )
        return PlannedEventTrades(
            trade_date=trade_date,
            portfolio=portfolio,
            intents=intents,
        )

    @staticmethod
    def write(plan: PlannedEventTrades, output: Path) -> None:
        """Persist an immutable event-trade plan."""
        encoded = plan.to_json_bytes()
        output.parent.mkdir(parents=True, exist_ok=True)
        try:
            with output.open("xb") as destination:
                destination.write(encoded)
        except FileExistsError:
            if output.read_bytes() != encoded:
                raise RuntimeError(f"event-trade plan collision at {output}") from None


def _intent(
    symbol: str,
    *,
    shares: int,
    score: float,
    observation: EventExecutionObservation,
) -> TradeIntent:
    identity = f"{observation.trade_date}|{symbol}|{observation.row_index}|{score:.17g}|{shares}"
    trade_id = f"earnings-{hashlib.sha256(identity.encode()).hexdigest()[:20]}"
    return TradeIntent(
        trade_id=trade_id,
        symbol=symbol,
        side="long",
        entry_date=observation.trade_date,
        exit_date=observation.trade_date,
        shares=shares,
        entry_price=observation.entry_price,
        exit_price=observation.exit_price,
        entry_average_daily_volume_shares=observation.frozen_average_daily_volume_shares,
        exit_average_daily_volume_shares=observation.frozen_average_daily_volume_shares,
        holding_sessions=0,
        entry_at=observation.entry_at,
        exit_at=observation.exit_at,
    )
