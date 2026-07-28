"""Aggregate immutable order replays into one reconciled Phase 6 session."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from datetime import date  # noqa: TC003 - Pydantic resolves runtime annotations.
from pathlib import Path  # noqa: TC003 - Pydantic resolves runtime annotations.
from typing import TYPE_CHECKING, Literal
from zoneinfo import ZoneInfo

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from collections.abc import Sequence

    from quant_earning_edge.backtest import NbboReplayEvidence

_MARKET_TIMEZONE = ZoneInfo("America/New_York")
PositionSide = Literal["long", "short"]


@dataclass(frozen=True)
class ReplayRoundTrip:
    """Pair one entry and exit order into a same-session position lifecycle."""

    trade_id: str
    entry_order_id: str
    exit_order_id: str
    side: PositionSide

    def __post_init__(self) -> None:
        identifiers = tuple(
            value.strip() for value in (self.trade_id, self.entry_order_id, self.exit_order_id)
        )
        if any(not value for value in identifiers):
            raise ValueError("round-trip identifiers must not be empty")
        if identifiers[1] == identifiers[2]:
            raise ValueError("entry and exit order ids must differ")
        object.__setattr__(self, "trade_id", identifiers[0])
        object.__setattr__(self, "entry_order_id", identifiers[1])
        object.__setattr__(self, "exit_order_id", identifiers[2])


@dataclass(frozen=True)
class ReplayRoundTripResult:
    """Execution and P&L reconciliation for one position."""

    trade_id: str
    symbol: str
    side: PositionSide
    intended_quantity: int
    entry_filled_quantity: int
    exit_filled_quantity: int
    matched_quantity: int
    unmatched_quantity: int
    entry_fill_price: float | None
    exit_fill_price: float | None
    gross_pnl: float
    commission: float
    net_pnl_on_matched_quantity: float
    reconciled: bool


@dataclass(frozen=True)
class ReplaySessionReport:
    """Canonical daily execution proof input."""

    schema_version: int
    session_date: date
    initial_cash: float
    evidence_sha256: tuple[str, ...]
    intended_order_count: int
    fully_filled_order_count: int
    fully_filled_order_rate: float
    intended_share_count: int
    filled_share_count: int
    share_fill_rate: float
    realized_adverse_slippage_bps_p10: float | None
    realized_adverse_slippage_bps_p50: float | None
    realized_adverse_slippage_bps_p90: float | None
    predicted_adverse_slippage_bps_p90: float | None
    p90_realized_to_predicted_ratio: float | None
    opening_auction_filled_share_count: int
    reconciliation_break_count: int
    gross_pnl: float | None
    commission: float
    net_pnl: float | None
    net_return: float | None
    round_trips: tuple[ReplayRoundTripResult, ...]

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
                raise RuntimeError(f"replay-session report collision at {output}") from None


class _StrictSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ReplayRoundTripSpec(_StrictSpec):
    trade_id: str
    entry_order_id: str
    exit_order_id: str
    side: Literal["long", "short"]

    def to_domain(self) -> ReplayRoundTrip:
        return ReplayRoundTrip(**self.model_dump())


class ReplaySessionAggregationSpec(_StrictSpec):
    """Paths and lifecycle mapping needed to aggregate one session."""

    session_date: date
    initial_cash: float = Field(gt=0)
    commission_bps_per_side: float = Field(default=1.0, ge=0)
    evidence_files: tuple[Path, ...] = Field(min_length=1)
    round_trips: tuple[ReplayRoundTripSpec, ...] = Field(min_length=1)


class ReplaySessionAggregator:
    """Build daily metrics only when every evidence/order mapping is explicit."""

    def evaluate(
        self,
        *,
        evidence: Sequence[NbboReplayEvidence],
        round_trips: Sequence[ReplayRoundTrip],
        session_date: date,
        initial_cash: float,
        commission_bps_per_side: float = 1.0,
    ) -> ReplaySessionReport:
        if not evidence or not round_trips:
            raise ValueError("replay evidence and round trips are required")
        if not math.isfinite(initial_cash) or initial_cash <= 0:
            raise ValueError("initial_cash must be finite and positive")
        if not math.isfinite(commission_bps_per_side) or commission_bps_per_side < 0:
            raise ValueError("commission must be finite and non-negative")
        by_order = {item.result.order.order_id: item for item in evidence}
        if len(by_order) != len(evidence):
            raise ValueError("replay evidence contains duplicate order ids")
        trade_ids = {item.trade_id for item in round_trips}
        if len(trade_ids) != len(round_trips):
            raise ValueError("round trips contain duplicate trade ids")
        referenced = tuple(
            order_id
            for item in round_trips
            for order_id in (item.entry_order_id, item.exit_order_id)
        )
        if len(set(referenced)) != len(referenced):
            raise ValueError("an order cannot be used by multiple round trips")
        if set(referenced) != set(by_order):
            raise ValueError("round-trip order ids do not exactly match replay evidence")
        self._validate_session_dates(evidence, session_date=session_date)
        results = tuple(
            self._round_trip(
                item,
                by_order=by_order,
                commission_bps_per_side=commission_bps_per_side,
            )
            for item in round_trips
        )
        return self._report(
            evidence=evidence,
            results=results,
            session_date=session_date,
            initial_cash=initial_cash,
        )

    @staticmethod
    def _round_trip(
        trade: ReplayRoundTrip,
        *,
        by_order: dict[str, NbboReplayEvidence],
        commission_bps_per_side: float,
    ) -> ReplayRoundTripResult:
        entry = by_order[trade.entry_order_id].result
        exit_fill = by_order[trade.exit_order_id].result
        expected_entry_side = "buy" if trade.side == "long" else "sell"
        expected_exit_side = "sell" if trade.side == "long" else "buy"
        if entry.order.side != expected_entry_side or exit_fill.order.side != expected_exit_side:
            raise ValueError(f"round trip {trade.trade_id} has invalid order sides")
        if entry.order.ticker != exit_fill.order.ticker:
            raise ValueError(f"round trip {trade.trade_id} crosses symbols")
        if entry.order.quantity != exit_fill.order.quantity:
            raise ValueError(f"round trip {trade.trade_id} intended quantities differ")
        if entry.order.submitted_at >= exit_fill.order.submitted_at:
            raise ValueError(f"round trip {trade.trade_id} entry is not before exit")
        matched = min(entry.filled_qty, exit_fill.filled_qty)
        unmatched = abs(entry.filled_qty - exit_fill.filled_qty)
        entry_notional = (entry.fill_price or 0.0) * entry.filled_qty
        exit_notional = (exit_fill.fill_price or 0.0) * exit_fill.filled_qty
        commission = (entry_notional + exit_notional) * commission_bps_per_side / 10_000
        gross_pnl = 0.0
        if matched:
            if entry.fill_price is None or exit_fill.fill_price is None:
                raise ValueError("matched replay quantities require both fill prices")
            direction = 1.0 if trade.side == "long" else -1.0
            gross_pnl = direction * (exit_fill.fill_price - entry.fill_price) * matched
        return ReplayRoundTripResult(
            trade_id=trade.trade_id,
            symbol=entry.order.ticker,
            side=trade.side,
            intended_quantity=entry.order.quantity,
            entry_filled_quantity=entry.filled_qty,
            exit_filled_quantity=exit_fill.filled_qty,
            matched_quantity=matched,
            unmatched_quantity=unmatched,
            entry_fill_price=entry.fill_price,
            exit_fill_price=exit_fill.fill_price,
            gross_pnl=gross_pnl,
            commission=commission,
            net_pnl_on_matched_quantity=gross_pnl - commission,
            reconciled=unmatched == 0,
        )

    @staticmethod
    def _report(
        *,
        evidence: Sequence[NbboReplayEvidence],
        results: tuple[ReplayRoundTripResult, ...],
        session_date: date,
        initial_cash: float,
    ) -> ReplaySessionReport:
        intended_shares = sum(item.result.order.quantity for item in evidence)
        filled_shares = sum(item.result.filled_qty for item in evidence)
        fully_filled = sum(
            item.result.filled_qty == item.result.order.quantity for item in evidence
        )
        realized = tuple(
            max(item.result.slippage_bps_realized, 0.0)
            for item in evidence
            if item.result.slippage_bps_realized is not None
        )
        predicted = tuple(max(item.result.slippage_bps_predicted, 0.0) for item in evidence)
        realized_p90 = _percentile(realized, 0.90)
        predicted_p90 = _percentile(predicted, 0.90)
        ratio = (
            realized_p90 / predicted_p90
            if realized_p90 is not None and predicted_p90 is not None and predicted_p90 > 0
            else None
        )
        break_count = sum(not item.reconciled for item in results)
        commission = sum(item.commission for item in results)
        gross_pnl = sum(item.gross_pnl for item in results) if break_count == 0 else None
        net_pnl = gross_pnl - commission if gross_pnl is not None else None
        return ReplaySessionReport(
            schema_version=1,
            session_date=session_date,
            initial_cash=initial_cash,
            evidence_sha256=tuple(sorted(item.sha256 for item in evidence)),
            intended_order_count=len(evidence),
            fully_filled_order_count=fully_filled,
            fully_filled_order_rate=fully_filled / len(evidence),
            intended_share_count=intended_shares,
            filled_share_count=filled_shares,
            share_fill_rate=filled_shares / intended_shares,
            realized_adverse_slippage_bps_p10=_percentile(realized, 0.10),
            realized_adverse_slippage_bps_p50=_percentile(realized, 0.50),
            realized_adverse_slippage_bps_p90=realized_p90,
            predicted_adverse_slippage_bps_p90=predicted_p90,
            p90_realized_to_predicted_ratio=ratio,
            opening_auction_filled_share_count=sum(
                item.result.opening_auction_filled_qty for item in evidence
            ),
            reconciliation_break_count=break_count,
            gross_pnl=gross_pnl,
            commission=commission,
            net_pnl=net_pnl,
            net_return=net_pnl / initial_cash if net_pnl is not None else None,
            round_trips=results,
        )

    @staticmethod
    def _validate_session_dates(
        evidence: Sequence[NbboReplayEvidence],
        *,
        session_date: date,
    ) -> None:
        for item in evidence:
            order = item.result.order
            submitted_date = order.submitted_at.astimezone(_MARKET_TIMEZONE).date()
            expires_date = order.expires_at.astimezone(_MARKET_TIMEZONE).date()
            if submitted_date != session_date or expires_date != session_date:
                raise ValueError("replay order window does not match session_date")


def _percentile(values: Sequence[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    rank = (len(ordered) - 1) * probability
    lower_index = math.floor(rank)
    upper_index = math.ceil(rank)
    if lower_index == upper_index:
        return ordered[lower_index]
    weight = rank - lower_index
    return ordered[lower_index] * (1 - weight) + ordered[upper_index] * weight
