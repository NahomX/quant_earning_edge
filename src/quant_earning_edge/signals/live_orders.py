"""Live-safe conversion of decision-time scores into frozen paper/replay orders."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import date, datetime
from typing import TYPE_CHECKING, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, model_validator

from quant_earning_edge.backtest import DecisionSnapshotSpec, IntendedOrderSpec
from quant_earning_edge.live import PaperOrderBatchSpec, PaperOrderRequest
from quant_earning_edge.portfolio import (
    FractionalKellyPortfolioConstructor,
    PortfolioConfig,
    PortfolioPlan,
    PositionTarget,
    ScoredCandidate,
)
from quant_earning_edge.signals.event_trades import (  # noqa: TC001 - Pydantic annotation.
    TradeOutcomeSpec,
)

if TYPE_CHECKING:
    from pathlib import Path

    from quant_earning_edge.signals.config import EarningsStrategyConfig


class _StrictSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")


class LiveCandidateSpec(_StrictSpec):
    """Candidate fields observable no later than the frozen decision."""

    symbol: str
    sector: str
    probability_up: float = Field(ge=0, le=1)
    sizing_price: float = Field(gt=0)
    sizing_price_observed_at: datetime
    frozen_average_daily_volume_shares: float = Field(gt=0)
    decision_snapshot: DecisionSnapshotSpec

    @model_validator(mode="after")
    def normalize_candidate(self) -> LiveCandidateSpec:
        symbol = self.symbol.strip().upper()
        sector = self.sector.strip()
        if not symbol or not sector:
            raise ValueError("live candidate symbol and sector must not be blank")
        if self.decision_snapshot.ticker.strip().upper() != symbol:
            raise ValueError("live candidate and decision snapshot symbols differ")
        object.__setattr__(self, "symbol", symbol)
        object.__setattr__(self, "sector", sector)
        return self


class DailyOrderPlanningSpec(_StrictSpec):
    """Only causal inputs needed to freeze one session's complete order set."""

    trade_date: date
    decision_at: datetime
    equity: float = Field(gt=0)
    candidates: tuple[LiveCandidateSpec, ...] = ()
    outcomes: tuple[TradeOutcomeSpec, ...] = ()
    entry_submitted_at: datetime
    entry_expires_at: datetime
    exit_submitted_at: datetime
    exit_expires_at: datetime
    minimum_probability: float = Field(default=0.5, ge=0.5, lt=1)

    @model_validator(mode="after")
    def validate_causal_schedule(self) -> DailyOrderPlanningSpec:
        symbols = tuple(item.symbol for item in self.candidates)
        if symbols != tuple(sorted(set(symbols))):
            raise ValueError("live candidate symbols must be unique and sorted")
        timestamps = (
            self.decision_at,
            self.entry_submitted_at,
            self.entry_expires_at,
            self.exit_submitted_at,
            self.exit_expires_at,
        )
        if any(item.tzinfo is None or item.utcoffset() is None for item in timestamps):
            raise ValueError("daily order timestamps must be timezone-aware")
        if not (
            self.decision_at
            <= self.entry_submitted_at
            < self.entry_expires_at
            < self.exit_submitted_at
            < self.exit_expires_at
        ):
            raise ValueError("daily order execution schedule is not strictly causal")
        execution_times = timestamps[1:]
        if any(item.date() != self.trade_date for item in execution_times):
            raise ValueError("daily order execution timestamps must match trade_date")
        for candidate in self.candidates:
            if candidate.sizing_price_observed_at > self.decision_at:
                raise ValueError("sizing price was observed after decision_at")
            if candidate.decision_snapshot.observed_at > self.decision_at:
                raise ValueError("NBBO snapshot was observed after decision_at")
        if any(item.closed_date >= self.trade_date for item in self.outcomes):
            raise ValueError("Kelly outcomes must be closed before the trade date")
        return self

    @property
    def canonical_bytes(self) -> bytes:
        return json.dumps(
            self.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
        ).encode()


@dataclass(frozen=True)
class FrozenDailyOrders:
    """Immutable linked evidence for sizing, paper submission, and later replay."""

    schema_version: int
    input_sha256: str
    strategy_config_sha256: str
    trade_date: date
    decision_at: datetime
    portfolio: PortfolioPlan
    intended_orders: tuple[IntendedOrderSpec, ...]
    decision_snapshots: tuple[DecisionSnapshotSpec, ...]
    paper_batch: PaperOrderBatchSpec

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("unsupported frozen daily order schema version")
        for name, digest in (
            ("input_sha256", self.input_sha256),
            ("strategy_config_sha256", self.strategy_config_sha256),
        ):
            if len(digest) != 64 or any(
                character not in "0123456789abcdef" for character in digest
            ):
                raise ValueError(f"{name} must be lowercase SHA-256")
        if self.decision_at.tzinfo is None or self.decision_at.utcoffset() is None:
            raise ValueError("frozen daily decision_at must be timezone-aware")
        intended_ids = tuple(item.order_id for item in self.intended_orders)
        paper_ids = tuple(item.client_order_id for item in self.paper_batch.orders)
        if intended_ids != tuple(sorted(set(intended_ids))) or intended_ids != paper_ids:
            raise ValueError("frozen intended and paper order identities must exactly match")
        if self.paper_batch.session_date != self.trade_date:
            raise ValueError("frozen paper batch date differs from trade date")
        snapshots = tuple(item.ticker.strip().upper() for item in self.decision_snapshots)
        if snapshots != tuple(sorted(set(snapshots))):
            raise ValueError("frozen decision snapshots must have unique sorted symbols")
        position_symbols = tuple(sorted(position.symbol for position in self.portfolio.positions))
        if snapshots != position_symbols:
            raise ValueError("frozen snapshots must exactly match portfolio symbols")
        paper_by_id = {item.client_order_id: item for item in self.paper_batch.orders}
        for intended in self.intended_orders:
            order = intended.to_domain()
            paper = paper_by_id[order.order_id]
            if (
                order.decision_time != self.decision_at
                or order.submitted_at.date() != self.trade_date
                or order.expires_at.date() != self.trade_date
                or paper.symbol != order.ticker
                or paper.quantity != order.quantity
                or paper.side != order.side
            ):
                raise ValueError("frozen paper and replay order fields do not reconcile")

    @property
    def canonical_bytes(self) -> bytes:
        return json.dumps(
            {
                "schema_version": self.schema_version,
                "input_sha256": self.input_sha256,
                "strategy_config_sha256": self.strategy_config_sha256,
                "trade_date": self.trade_date.isoformat(),
                "decision_at": self.decision_at.isoformat(),
                "portfolio": asdict(self.portfolio),
                "intended_orders": [item.model_dump(mode="json") for item in self.intended_orders],
                "decision_snapshots": [
                    item.model_dump(mode="json") for item in self.decision_snapshots
                ],
                "paper_batch": self.paper_batch.model_dump(mode="json"),
            },
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
                raise RuntimeError(f"frozen daily order collision at {output}") from None

    @classmethod
    def load(cls, path: Path) -> FrozenDailyOrders:
        """Load canonical linked order evidence and re-run causal invariants."""
        try:
            raw = json.loads(path.read_bytes())
            portfolio = raw["portfolio"]
            artifact = cls(
                schema_version=int(raw["schema_version"]),
                input_sha256=str(raw["input_sha256"]),
                strategy_config_sha256=str(raw["strategy_config_sha256"]),
                trade_date=date.fromisoformat(raw["trade_date"]),
                decision_at=datetime.fromisoformat(raw["decision_at"]),
                portfolio=PortfolioPlan(
                    equity=float(portfolio["equity"]),
                    history_count=int(portfolio["history_count"]),
                    raw_kelly=float(portfolio["raw_kelly"]),
                    fractional_kelly=float(portfolio["fractional_kelly"]),
                    positions=tuple(PositionTarget(**item) for item in portfolio["positions"]),
                    gross_weight=float(portfolio["gross_weight"]),
                    sector_weights=tuple(
                        (str(item[0]), float(item[1])) for item in portfolio["sector_weights"]
                    ),
                    sizing_mode=cast(
                        "Literal['calibration', 'kelly']",
                        str(portfolio["sizing_mode"]),
                    ),
                    per_position_weight=float(portfolio["per_position_weight"]),
                ),
                intended_orders=tuple(
                    IntendedOrderSpec.model_validate(item) for item in raw["intended_orders"]
                ),
                decision_snapshots=tuple(
                    DecisionSnapshotSpec.model_validate(item) for item in raw["decision_snapshots"]
                ),
                paper_batch=PaperOrderBatchSpec.model_validate(raw["paper_batch"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"invalid frozen daily orders: {path}") from error
        if json.loads(artifact.canonical_bytes) != raw:
            raise ValueError("frozen daily orders are not canonical or use unsupported fields")
        return artifact


class LiveOrderPlanner:
    """Apply the precommitted portfolio policy and freeze linked order identities."""

    def __init__(self, strategy: EarningsStrategyConfig, *, strategy_sha256: str) -> None:
        self._strategy = strategy
        self._strategy_sha256 = strategy_sha256

    def plan(self, spec: DailyOrderPlanningSpec) -> FrozenDailyOrders:
        sizing = self._strategy.portfolio.sizing
        caps = self._strategy.portfolio.caps
        eligible = tuple(
            item for item in spec.candidates if item.probability_up >= spec.minimum_probability
        )
        portfolio = FractionalKellyPortfolioConstructor(
            PortfolioConfig(
                top_k=self._strategy.portfolio.top_k,
                kelly_fraction=sizing.kelly_fraction,
                history_window=sizing.rolling_window_days,
                minimum_history=min(20, sizing.rolling_window_days),
                calibration_position_weight=sizing.calibration_position_pct,
                max_position_weight=caps.max_position_pct,
                max_sector_weight=caps.max_sector_pct,
                max_gross_weight=caps.max_gross_exposure_pct,
            )
        ).construct(
            candidates=tuple(
                ScoredCandidate(
                    symbol=item.symbol,
                    sector=item.sector,
                    side="long",
                    score=item.probability_up - 0.5,
                    price=item.sizing_price,
                )
                for item in eligible
            ),
            outcomes=tuple(item.to_domain() for item in spec.outcomes),
            equity=spec.equity,
            decision_date=spec.trade_date,
        )
        candidates = {item.symbol: item for item in eligible}
        orders: list[IntendedOrderSpec] = []
        paper_orders: list[PaperOrderRequest] = []
        selected_snapshots: list[DecisionSnapshotSpec] = []
        for position in sorted(portfolio.positions, key=lambda item: item.symbol):
            candidate = candidates[position.symbol]
            trade_id = _trade_id(
                spec=spec,
                symbol=position.symbol,
                shares=position.shares,
                probability=candidate.probability_up,
            )
            entry_id = f"{trade_id}-entry"
            exit_id = f"{trade_id}-exit"
            orders.extend(
                (
                    IntendedOrderSpec(
                        order_id=entry_id,
                        ticker=position.symbol,
                        side="buy",
                        quantity=position.shares,
                        decision_time=spec.decision_at,
                        submitted_at=spec.entry_submitted_at,
                        expires_at=spec.entry_expires_at,
                        average_daily_volume_shares=(candidate.frozen_average_daily_volume_shares),
                    ),
                    IntendedOrderSpec(
                        order_id=exit_id,
                        ticker=position.symbol,
                        side="sell",
                        quantity=position.shares,
                        decision_time=spec.decision_at,
                        submitted_at=spec.exit_submitted_at,
                        expires_at=spec.exit_expires_at,
                        average_daily_volume_shares=(candidate.frozen_average_daily_volume_shares),
                    ),
                )
            )
            paper_orders.extend(
                (
                    PaperOrderRequest(
                        client_order_id=entry_id,
                        symbol=position.symbol,
                        quantity=position.shares,
                        side="buy",
                        order_type="market",
                        time_in_force="day",
                    ),
                    PaperOrderRequest(
                        client_order_id=exit_id,
                        symbol=position.symbol,
                        quantity=position.shares,
                        side="sell",
                        order_type="market",
                        time_in_force="cls",
                    ),
                )
            )
            selected_snapshots.append(candidate.decision_snapshot)
        sorted_orders = tuple(sorted(orders, key=lambda item: item.order_id))
        sorted_paper = tuple(sorted(paper_orders, key=lambda item: item.client_order_id))
        return FrozenDailyOrders(
            schema_version=1,
            input_sha256=hashlib.sha256(spec.canonical_bytes).hexdigest(),
            strategy_config_sha256=self._strategy_sha256,
            trade_date=spec.trade_date,
            decision_at=spec.decision_at,
            portfolio=portfolio,
            intended_orders=sorted_orders,
            decision_snapshots=tuple(sorted(selected_snapshots, key=lambda item: item.ticker)),
            paper_batch=PaperOrderBatchSpec(
                session_date=spec.trade_date,
                orders=sorted_paper,
            ),
        )


def _trade_id(
    *,
    spec: DailyOrderPlanningSpec,
    symbol: str,
    shares: int,
    probability: float,
) -> str:
    identity = (
        f"{spec.trade_date}|{spec.decision_at.isoformat()}|{symbol}|{shares}|{probability:.17g}"
    )
    return f"earnings-{hashlib.sha256(identity.encode()).hexdigest()[:20]}"


def strategy_file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
