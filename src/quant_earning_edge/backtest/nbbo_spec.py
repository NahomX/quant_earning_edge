"""Validated JSON boundary and immutable evidence for one NBBO replay."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime  # noqa: TC003 - Pydantic resolves runtime annotations.
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict, Field

from quant_earning_edge.backtest.nbbo_replay import (
    DecisionSnapshot,
    FillFragment,
    IntendedOrder,
    NbboQuote,
    ReplayConfig,
    ReplayFill,
    TradePrint,
)


class _StrictSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")


class IntendedOrderSpec(_StrictSpec):
    order_id: str
    ticker: str
    side: Literal["buy", "sell"]
    quantity: int = Field(gt=0)
    decision_time: datetime
    submitted_at: datetime
    expires_at: datetime
    average_daily_volume_shares: float = Field(gt=0)
    aggressiveness: Literal["aggressive", "mid", "passive"] = "aggressive"
    limit_price: float | None = Field(default=None, gt=0)

    def to_domain(self) -> IntendedOrder:
        return IntendedOrder(**self.model_dump())


class DecisionSnapshotSpec(_StrictSpec):
    ticker: str
    observed_at: datetime
    bid_price: float = Field(gt=0)
    ask_price: float = Field(gt=0)
    bid_size: int = Field(gt=0)
    ask_size: int = Field(gt=0)
    last_trade_price: float = Field(gt=0)
    last_trade_at: datetime

    def to_domain(self) -> DecisionSnapshot:
        return DecisionSnapshot(**self.model_dump())


class NbboQuoteSpec(_StrictSpec):
    ticker: str
    timestamp: datetime
    sequence: int = Field(ge=0)
    bid_price: float = Field(gt=0)
    ask_price: float = Field(gt=0)
    bid_size: int = Field(gt=0)
    ask_size: int = Field(gt=0)

    def to_domain(self) -> NbboQuote:
        return NbboQuote(**self.model_dump())


class TradePrintSpec(_StrictSpec):
    ticker: str
    timestamp: datetime
    sequence: int = Field(ge=0)
    price: float = Field(gt=0)
    size: int = Field(gt=0)
    is_opening_auction: bool = False

    def to_domain(self) -> TradePrint:
        return TradePrint(**self.model_dump())


class ReplayConfigSpec(_StrictSpec):
    market_impact_bps_coefficient: float = Field(default=5.0, ge=0)
    mid_fill_probability: float = Field(default=0.35, ge=0, le=1)
    passive_fill_probability: float = Field(default=0.10, ge=0, le=1)
    opening_auction_probability_multiplier: float = Field(default=0.50, ge=0, le=1)

    def to_domain(self) -> ReplayConfig:
        return ReplayConfig(**self.model_dump())


class NbboReplaySpec(_StrictSpec):
    """Self-contained normalized inputs for one deterministic replay."""

    order: IntendedOrderSpec
    decision_snapshot: DecisionSnapshotSpec
    quotes: tuple[NbboQuoteSpec, ...]
    trades: tuple[TradePrintSpec, ...] = ()
    config: ReplayConfigSpec = ReplayConfigSpec()

    @property
    def canonical_bytes(self) -> bytes:
        """Return stable semantic input bytes independent of source formatting."""
        return json.dumps(
            self.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
        ).encode()

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes).hexdigest()

    def domain_inputs(
        self,
    ) -> tuple[
        IntendedOrder,
        DecisionSnapshot,
        tuple[NbboQuote, ...],
        tuple[TradePrint, ...],
        ReplayConfig,
    ]:
        return (
            self.order.to_domain(),
            self.decision_snapshot.to_domain(),
            tuple(item.to_domain() for item in self.quotes),
            tuple(item.to_domain() for item in self.trades),
            self.config.to_domain(),
        )


class FillFragmentSpec(_StrictSpec):
    timestamp: datetime
    source: Literal["nbbo", "trade", "opening_auction"]
    quantity: int = Field(gt=0)
    price: float = Field(gt=0)
    source_sequence: int = Field(ge=0)

    def to_domain(self) -> FillFragment:
        return FillFragment(**self.model_dump())


class ReplayFillSpec(_StrictSpec):
    order: IntendedOrderSpec
    decision_snapshot: DecisionSnapshotSpec
    filled_qty: int = Field(ge=0)
    unfilled_qty: int = Field(ge=0)
    fill_price: float | None = Field(default=None, gt=0)
    fill_rate: float = Field(ge=0, le=1)
    slippage_bps_realized: float | None = None
    slippage_bps_predicted: float
    market_move_bps: float
    fill_probability_assumption: float = Field(ge=0, le=1)
    opening_auction_filled_qty: int = Field(ge=0)
    opening_auction_skew_bps: float | None = None
    fragments: tuple[FillFragmentSpec, ...]
    read_quote_timestamps: tuple[datetime, ...]
    read_trade_timestamps: tuple[datetime, ...]
    notes: str = ""

    def to_domain(self) -> ReplayFill:
        values = self.model_dump(
            exclude={"order", "decision_snapshot", "fragments"},
        )
        return ReplayFill(
            order=self.order.to_domain(),
            decision_snapshot=self.decision_snapshot.to_domain(),
            fragments=tuple(item.to_domain() for item in self.fragments),
            **values,
        )


class NbboReplayEvidenceSpec(_StrictSpec):
    schema_version: Literal[1]
    input_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    quote_event_count: int = Field(ge=0)
    trade_event_count: int = Field(ge=0)
    result: ReplayFillSpec

    def to_domain(self) -> NbboReplayEvidence:
        return NbboReplayEvidence(
            schema_version=self.schema_version,
            input_sha256=self.input_sha256,
            quote_event_count=self.quote_event_count,
            trade_event_count=self.trade_event_count,
            result=self.result.to_domain(),
        )


@dataclass(frozen=True)
class NbboReplayEvidence:
    """Canonical replay result linked to its complete semantic input."""

    schema_version: int
    input_sha256: str
    quote_event_count: int
    trade_event_count: int
    result: ReplayFill

    @classmethod
    def build(cls, *, spec: NbboReplaySpec, result: ReplayFill) -> NbboReplayEvidence:
        return cls(
            schema_version=1,
            input_sha256=spec.sha256,
            quote_event_count=len(spec.quotes),
            trade_event_count=len(spec.trades),
            result=result,
        )

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
        """Write once; allow an idempotent retry only for identical evidence."""
        encoded = self.canonical_bytes
        output.parent.mkdir(parents=True, exist_ok=True)
        try:
            with output.open("xb") as destination:
                destination.write(encoded)
        except FileExistsError:
            if output.read_bytes() != encoded:
                raise RuntimeError(f"NBBO replay evidence collision at {output}") from None

    @classmethod
    def load(cls, path: Path) -> NbboReplayEvidence:
        """Load strict canonical evidence and re-run all domain invariants."""
        try:
            raw = json.loads(path.read_bytes())
            evidence = NbboReplayEvidenceSpec.model_validate(raw).to_domain()
        except (json.JSONDecodeError, TypeError, ValueError) as error:
            raise ValueError(f"invalid NBBO replay evidence: {path}") from error
        if json.loads(evidence.canonical_bytes) != raw:
            raise ValueError("NBBO replay evidence is not canonical or uses unsupported fields")
        return evidence


if TYPE_CHECKING:
    from pathlib import Path
