"""Deterministic, causally bounded NBBO and trade-print order replay.

The simulator is intentionally evidence-driven: it consumes recorded quote and
trade events only after an order became executable, retains every consumed
timestamp, and reports partial or missed fills rather than inventing liquidity.
It does not attempt to reconstruct exchange-level queue position from NBBO.
Instead, limit-order queue uncertainty is represented by an explicit,
conservative expected-participation assumption for each aggressiveness bracket.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from collections.abc import Iterable
    from datetime import datetime

OrderSide = Literal["buy", "sell"]
OrderAggressiveness = Literal["aggressive", "mid", "passive"]
FillSource = Literal["nbbo", "trade", "opening_auction"]


@dataclass(frozen=True)
class IntendedOrder:
    """An immutable order instruction frozen before it can execute."""

    order_id: str
    ticker: str
    side: OrderSide
    quantity: int
    decision_time: datetime
    submitted_at: datetime
    expires_at: datetime
    average_daily_volume_shares: float
    aggressiveness: OrderAggressiveness = "aggressive"
    limit_price: float | None = None

    def __post_init__(self) -> None:
        order_id = self.order_id.strip()
        ticker = self.ticker.strip().upper()
        if not order_id or not ticker:
            raise ValueError("order_id and ticker must not be empty")
        object.__setattr__(self, "order_id", order_id)
        object.__setattr__(self, "ticker", ticker)
        _require_aware(self.decision_time, field="decision_time")
        _require_aware(self.submitted_at, field="submitted_at")
        _require_aware(self.expires_at, field="expires_at")
        if not self.decision_time <= self.submitted_at < self.expires_at:
            raise ValueError("order timestamps must satisfy decision <= submitted < expires")
        if self.quantity <= 0:
            raise ValueError("quantity must be positive")
        _require_positive_finite(
            self.average_daily_volume_shares,
            field="average_daily_volume_shares",
        )
        if self.limit_price is not None:
            _require_positive_finite(self.limit_price, field="limit_price")


@dataclass(frozen=True)
class DecisionSnapshot:
    """NBBO and last-trade state actually known when the order was decided."""

    ticker: str
    observed_at: datetime
    bid_price: float
    ask_price: float
    bid_size: int
    ask_size: int
    last_trade_price: float
    last_trade_at: datetime

    def __post_init__(self) -> None:
        ticker = self.ticker.strip().upper()
        if not ticker:
            raise ValueError("ticker must not be empty")
        object.__setattr__(self, "ticker", ticker)
        _require_aware(self.observed_at, field="observed_at")
        _require_aware(self.last_trade_at, field="last_trade_at")
        _validate_market(
            bid_price=self.bid_price,
            ask_price=self.ask_price,
            bid_size=self.bid_size,
            ask_size=self.ask_size,
        )
        _require_positive_finite(self.last_trade_price, field="last_trade_price")
        if self.last_trade_at > self.observed_at:
            raise ValueError("last trade cannot be newer than the NBBO snapshot")

    @property
    def midpoint(self) -> float:
        return (self.bid_price + self.ask_price) / 2.0


@dataclass(frozen=True)
class NbboQuote:
    """One normalized NBBO update from the replay source."""

    ticker: str
    timestamp: datetime
    sequence: int
    bid_price: float
    ask_price: float
    bid_size: int
    ask_size: int

    def __post_init__(self) -> None:
        ticker = self.ticker.strip().upper()
        if not ticker:
            raise ValueError("ticker must not be empty")
        object.__setattr__(self, "ticker", ticker)
        _require_aware(self.timestamp, field="timestamp")
        if self.sequence < 0:
            raise ValueError("sequence must not be negative")
        _validate_market(
            bid_price=self.bid_price,
            ask_price=self.ask_price,
            bid_size=self.bid_size,
            ask_size=self.ask_size,
        )

    @property
    def midpoint(self) -> float:
        return (self.bid_price + self.ask_price) / 2.0


@dataclass(frozen=True)
class TradePrint:
    """One normalized eligible-market trade print."""

    ticker: str
    timestamp: datetime
    sequence: int
    price: float
    size: int
    is_opening_auction: bool = False

    def __post_init__(self) -> None:
        ticker = self.ticker.strip().upper()
        if not ticker:
            raise ValueError("ticker must not be empty")
        object.__setattr__(self, "ticker", ticker)
        _require_aware(self.timestamp, field="timestamp")
        if self.sequence < 0:
            raise ValueError("sequence must not be negative")
        _require_positive_finite(self.price, field="price")
        if self.size <= 0:
            raise ValueError("size must be positive")


@dataclass(frozen=True)
class ReplayConfig:
    """Explicit execution assumptions used where NBBO cannot reveal queue position."""

    market_impact_bps_coefficient: float = 5.0
    mid_fill_probability: float = 0.35
    passive_fill_probability: float = 0.10
    opening_auction_probability_multiplier: float = 0.50

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.market_impact_bps_coefficient)
            or self.market_impact_bps_coefficient < 0
        ):
            raise ValueError("market impact coefficient must be finite and non-negative")
        probabilities = (
            self.mid_fill_probability,
            self.passive_fill_probability,
            self.opening_auction_probability_multiplier,
        )
        if any(not math.isfinite(value) or not 0 <= value <= 1 for value in probabilities):
            raise ValueError("replay probabilities must be finite and in [0, 1]")
        if self.passive_fill_probability > self.mid_fill_probability:
            raise ValueError("passive probability must not exceed mid probability")


@dataclass(frozen=True)
class FillFragment:
    """One auditable component of a replayed fill."""

    timestamp: datetime
    source: FillSource
    quantity: int
    price: float
    source_sequence: int


@dataclass(frozen=True)
class ReplayFill:
    """Execution evidence produced by replaying one intended order."""

    order: IntendedOrder
    decision_snapshot: DecisionSnapshot
    filled_qty: int
    unfilled_qty: int
    fill_price: float | None
    fill_rate: float
    slippage_bps_realized: float | None
    slippage_bps_predicted: float
    market_move_bps: float
    fill_probability_assumption: float
    opening_auction_filled_qty: int
    opening_auction_skew_bps: float | None
    fragments: tuple[FillFragment, ...]
    read_quote_timestamps: tuple[datetime, ...]
    read_trade_timestamps: tuple[datetime, ...]
    notes: str = ""


def replay_order(
    order: IntendedOrder,
    *,
    decision_snapshot: DecisionSnapshot,
    quotes: Iterable[NbboQuote],
    trades: Iterable[TradePrint] = (),
    config: ReplayConfig | None = None,
) -> ReplayFill:
    """Replay one order without consulting pre-decision or pre-submission state.

    Aggressive orders consume recorded opposite-side NBBO size. Mid and passive
    limits consume a conservative expected fraction of eligible trade prints.
    The latter is an explicit probability-weighted queue approximation, not a
    claim that NBBO contains exchange queue position.
    """
    assumptions = config or ReplayConfig()
    _validate_snapshot(order, decision_snapshot)
    eligible_quotes = _eligible_quotes(order, quotes)
    if not eligible_quotes:
        return _missed_fill(
            order,
            decision_snapshot=decision_snapshot,
            probability=_fill_probability(order, assumptions),
            notes="no eligible NBBO at or after submission",
        )
    arrival = eligible_quotes[0]
    impact_bps = assumptions.market_impact_bps_coefficient * math.sqrt(
        order.quantity / order.average_daily_volume_shares
    )
    probability = _fill_probability(order, assumptions)
    if order.aggressiveness == "aggressive":
        fragments, read_quotes = _replay_aggressive(
            order,
            eligible_quotes,
            impact_bps=impact_bps,
        )
        read_trades: tuple[datetime, ...] = ()
    else:
        fragments, read_trades = _replay_limit(
            order,
            decision_snapshot=decision_snapshot,
            arrival=arrival,
            trades=trades,
            probability=probability,
            assumptions=assumptions,
            impact_bps=impact_bps,
        )
        read_quotes = (arrival.timestamp,)
    return _result(
        order,
        decision_snapshot=decision_snapshot,
        arrival=arrival,
        probability=probability,
        impact_bps=impact_bps,
        fragments=fragments,
        read_quotes=read_quotes,
        read_trades=read_trades,
    )


def _eligible_quotes(order: IntendedOrder, quotes: Iterable[NbboQuote]) -> tuple[NbboQuote, ...]:
    eligible = tuple(
        sorted(
            (
                quote
                for quote in quotes
                if quote.ticker == order.ticker
                and order.submitted_at <= quote.timestamp <= order.expires_at
            ),
            key=lambda quote: (quote.timestamp, quote.sequence),
        )
    )
    _require_unique_events(
        ((quote.timestamp, quote.sequence) for quote in eligible),
        source="quote",
    )
    return eligible


def _replay_aggressive(
    order: IntendedOrder,
    quotes: tuple[NbboQuote, ...],
    *,
    impact_bps: float,
) -> tuple[tuple[FillFragment, ...], tuple[datetime, ...]]:
    remaining = order.quantity
    fragments: list[FillFragment] = []
    read_timestamps: list[datetime] = []
    for quote in quotes:
        read_timestamps.append(quote.timestamp)
        base_price = quote.ask_price if order.side == "buy" else quote.bid_price
        if not _within_limit(order, base_price):
            continue
        displayed = quote.ask_size if order.side == "buy" else quote.bid_size
        quantity = min(remaining, displayed)
        if quantity <= 0:
            continue
        price = _impacted_price(order, base_price, impact_bps=impact_bps)
        if order.limit_price is not None:
            price = (
                min(price, order.limit_price)
                if order.side == "buy"
                else max(price, order.limit_price)
            )
        fragments.append(
            FillFragment(
                timestamp=quote.timestamp,
                source="nbbo",
                quantity=quantity,
                price=price,
                source_sequence=quote.sequence,
            )
        )
        remaining -= quantity
        if remaining == 0:
            break
    return tuple(fragments), tuple(read_timestamps)


def _replay_limit(
    order: IntendedOrder,
    *,
    decision_snapshot: DecisionSnapshot,
    arrival: NbboQuote,
    trades: Iterable[TradePrint],
    probability: float,
    assumptions: ReplayConfig,
    impact_bps: float,
) -> tuple[tuple[FillFragment, ...], tuple[datetime, ...]]:
    limit_price = order.limit_price or _default_limit(order, decision_snapshot)
    eligible = tuple(
        sorted(
            (
                trade
                for trade in trades
                if trade.ticker == order.ticker
                and max(order.submitted_at, arrival.timestamp)
                <= trade.timestamp
                <= order.expires_at
            ),
            key=lambda trade: (trade.timestamp, trade.sequence),
        )
    )
    _require_unique_events(
        ((trade.timestamp, trade.sequence) for trade in eligible),
        source="trade",
    )
    remaining = order.quantity
    fragments: list[FillFragment] = []
    read_timestamps: list[datetime] = []
    for trade in eligible:
        read_timestamps.append(trade.timestamp)
        if not _trade_reaches_limit(order, trade.price, limit_price=limit_price):
            continue
        event_probability = probability
        if trade.is_opening_auction:
            event_probability *= assumptions.opening_auction_probability_multiplier
        available = math.floor(trade.size * event_probability)
        quantity = min(remaining, available)
        if quantity <= 0:
            continue
        price = _impacted_price(order, trade.price, impact_bps=impact_bps)
        price = min(price, limit_price) if order.side == "buy" else max(price, limit_price)
        fragments.append(
            FillFragment(
                timestamp=trade.timestamp,
                source="opening_auction" if trade.is_opening_auction else "trade",
                quantity=quantity,
                price=price,
                source_sequence=trade.sequence,
            )
        )
        remaining -= quantity
        if remaining == 0:
            break
    return tuple(fragments), tuple(read_timestamps)


def _result(
    order: IntendedOrder,
    *,
    decision_snapshot: DecisionSnapshot,
    arrival: NbboQuote,
    probability: float,
    impact_bps: float,
    fragments: tuple[FillFragment, ...],
    read_quotes: tuple[datetime, ...],
    read_trades: tuple[datetime, ...],
) -> ReplayFill:
    filled = sum(fragment.quantity for fragment in fragments)
    fill_price = (
        sum(fragment.price * fragment.quantity for fragment in fragments) / filled
        if filled
        else None
    )
    direction = 1.0 if order.side == "buy" else -1.0
    arrival_midpoint = arrival.midpoint
    realized = (
        direction * (fill_price / arrival_midpoint - 1.0) * 10_000.0
        if fill_price is not None
        else None
    )
    predicted = _predicted_slippage_bps(
        order,
        arrival=arrival,
        impact_bps=impact_bps,
    )
    market_move = direction * (arrival_midpoint / decision_snapshot.midpoint - 1.0) * 10_000.0
    auction = tuple(item for item in fragments if item.source == "opening_auction")
    auction_qty = sum(item.quantity for item in auction)
    auction_price = (
        sum(item.price * item.quantity for item in auction) / auction_qty if auction_qty else None
    )
    auction_skew = (
        direction * (auction_price / arrival_midpoint - 1.0) * 10_000.0
        if auction_price is not None
        else None
    )
    notes = "filled" if filled == order.quantity else "partial fill" if filled else "missed fill"
    return ReplayFill(
        order=order,
        decision_snapshot=decision_snapshot,
        filled_qty=filled,
        unfilled_qty=order.quantity - filled,
        fill_price=fill_price,
        fill_rate=filled / order.quantity,
        slippage_bps_realized=realized,
        slippage_bps_predicted=predicted,
        market_move_bps=market_move,
        fill_probability_assumption=probability,
        opening_auction_filled_qty=auction_qty,
        opening_auction_skew_bps=auction_skew,
        fragments=fragments,
        read_quote_timestamps=read_quotes,
        read_trade_timestamps=read_trades,
        notes=notes,
    )


def _missed_fill(
    order: IntendedOrder,
    *,
    decision_snapshot: DecisionSnapshot,
    probability: float,
    notes: str,
) -> ReplayFill:
    return ReplayFill(
        order=order,
        decision_snapshot=decision_snapshot,
        filled_qty=0,
        unfilled_qty=order.quantity,
        fill_price=None,
        fill_rate=0.0,
        slippage_bps_realized=None,
        slippage_bps_predicted=0.0,
        market_move_bps=0.0,
        fill_probability_assumption=probability,
        opening_auction_filled_qty=0,
        opening_auction_skew_bps=None,
        fragments=(),
        read_quote_timestamps=(),
        read_trade_timestamps=(),
        notes=notes,
    )


def _validate_snapshot(order: IntendedOrder, snapshot: DecisionSnapshot) -> None:
    if snapshot.ticker != order.ticker:
        raise ValueError("decision snapshot ticker does not match order")
    if snapshot.observed_at > order.decision_time:
        raise ValueError("decision snapshot was not observable by decision_time")


def _fill_probability(order: IntendedOrder, config: ReplayConfig) -> float:
    if order.aggressiveness == "aggressive":
        return 1.0
    if order.aggressiveness == "mid":
        return config.mid_fill_probability
    return config.passive_fill_probability


def _default_limit(order: IntendedOrder, snapshot: DecisionSnapshot) -> float:
    if order.aggressiveness == "mid":
        return snapshot.midpoint
    return snapshot.bid_price if order.side == "buy" else snapshot.ask_price


def _within_limit(order: IntendedOrder, price: float) -> bool:
    if order.limit_price is None:
        return True
    return price <= order.limit_price if order.side == "buy" else price >= order.limit_price


def _trade_reaches_limit(order: IntendedOrder, price: float, *, limit_price: float) -> bool:
    return price <= limit_price if order.side == "buy" else price >= limit_price


def _impacted_price(order: IntendedOrder, price: float, *, impact_bps: float) -> float:
    multiplier = 1.0 + impact_bps / 10_000.0 if order.side == "buy" else 1.0 - impact_bps / 10_000.0
    return price * multiplier


def _predicted_slippage_bps(
    order: IntendedOrder,
    *,
    arrival: NbboQuote,
    impact_bps: float,
) -> float:
    half_spread_bps = (arrival.ask_price - arrival.bid_price) / (2 * arrival.midpoint) * 10_000
    if order.aggressiveness == "aggressive":
        return half_spread_bps + impact_bps
    if order.aggressiveness == "mid":
        return impact_bps
    return -half_spread_bps + impact_bps


def _require_unique_events(
    identities: Iterable[tuple[datetime, int]],
    *,
    source: str,
) -> None:
    values = tuple(identities)
    if len(set(values)) != len(values):
        raise ValueError(f"duplicate {source} timestamp/sequence identities")


def _validate_market(
    *,
    bid_price: float,
    ask_price: float,
    bid_size: int,
    ask_size: int,
) -> None:
    _require_positive_finite(bid_price, field="bid_price")
    _require_positive_finite(ask_price, field="ask_price")
    if ask_price < bid_price:
        raise ValueError("ask_price must not be below bid_price")
    if bid_size <= 0 or ask_size <= 0:
        raise ValueError("NBBO sizes must be positive")


def _require_aware(value: datetime, *, field: str) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field} must be timezone-aware")


def _require_positive_finite(value: float, *, field: str) -> None:
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{field} must be finite and positive")
