"""Load schema-stable Polygon market events into conservative replay inputs."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import UTC
from typing import TYPE_CHECKING, Any, ClassVar, TypeVar

import pyarrow.parquet as pq

from quant_earning_edge.backtest.nbbo_replay import NbboQuote, TradePrint

if TYPE_CHECKING:
    from collections.abc import Sequence
    from datetime import datetime
    from pathlib import Path

EventT = TypeVar("EventT", NbboQuote, TradePrint)


@dataclass(frozen=True)
class ReplayMarketEvents:
    """Replayable events plus explicit filtering and precision-loss evidence."""

    symbol: str
    start_at: datetime
    end_at: datetime
    quotes: tuple[NbboQuote, ...]
    trades: tuple[TradePrint, ...]
    source_quote_count: int
    rejected_one_sided_quote_count: int
    source_trade_count: int
    rejected_corrected_trade_count: int
    rejected_subshare_trade_count: int
    fractional_share_quantity_discarded: float
    auction_classification_configured: bool


class ReplayMarketDataLoader:
    """Convert silver Parquet to replay types without inventing missing liquidity."""

    _QUOTE_REQUIRED: ClassVar[set[str]] = {
        "timestamp",
        "sequence_number",
        "symbol",
        "bid_price",
        "ask_price",
        "bid_size",
        "ask_size",
        "is_replayable",
    }
    _TRADE_REQUIRED: ClassVar[set[str]] = {
        "timestamp",
        "sequence_number",
        "symbol",
        "price",
        "size",
        "conditions",
        "correction",
    }

    def load(
        self,
        *,
        quote_files: Sequence[Path],
        trade_files: Sequence[Path],
        symbol: str,
        start_at: datetime,
        end_at: datetime,
        opening_auction_condition_codes: frozenset[int] = frozenset(),
    ) -> ReplayMarketEvents:
        """Load an inclusive SIP-time interval and audit every rejected event."""
        normalized_symbol = symbol.strip().upper()
        if not normalized_symbol:
            raise ValueError("symbol must not be empty")
        if not quote_files:
            raise ValueError("at least one quote file is required")
        if any(value.tzinfo is None or value.utcoffset() is None for value in (start_at, end_at)):
            raise ValueError("market-event cutoffs must be timezone-aware")
        start_utc = start_at.astimezone(UTC)
        end_utc = end_at.astimezone(UTC)
        if end_utc <= start_utc:
            raise ValueError("end_at must be after start_at")
        if any(code < 0 for code in opening_auction_condition_codes):
            raise ValueError("opening-auction condition codes must not be negative")

        quote_rows = self._read_rows(quote_files, required=self._QUOTE_REQUIRED, kind="quote")
        trade_rows = self._read_rows(trade_files, required=self._TRADE_REQUIRED, kind="trade")
        quotes, source_quote_count, rejected_one_sided = self._load_quotes(
            quote_rows,
            symbol=normalized_symbol,
            start_at=start_utc,
            end_at=end_utc,
        )
        (
            trades,
            source_trade_count,
            rejected_corrections,
            rejected_subshare,
            fractional_discarded,
        ) = self._load_trades(
            trade_rows,
            symbol=normalized_symbol,
            start_at=start_utc,
            end_at=end_utc,
            opening_auction_condition_codes=opening_auction_condition_codes,
        )
        return ReplayMarketEvents(
            symbol=normalized_symbol,
            start_at=start_utc,
            end_at=end_utc,
            quotes=quotes,
            trades=trades,
            source_quote_count=source_quote_count,
            rejected_one_sided_quote_count=rejected_one_sided,
            source_trade_count=source_trade_count,
            rejected_corrected_trade_count=rejected_corrections,
            rejected_subshare_trade_count=rejected_subshare,
            fractional_share_quantity_discarded=fractional_discarded,
            auction_classification_configured=bool(opening_auction_condition_codes),
        )

    def _load_quotes(
        self,
        rows: tuple[dict[str, Any], ...],
        *,
        symbol: str,
        start_at: datetime,
        end_at: datetime,
    ) -> tuple[tuple[NbboQuote, ...], int, int]:
        quote_by_identity: dict[tuple[datetime, int], NbboQuote] = {}
        rejected_one_sided = 0
        source_quote_count = 0
        for row in rows:
            if not self._matches(row, symbol=symbol, start_at=start_at, end_at=end_at):
                continue
            source_quote_count += 1
            bid_price = float(row["bid_price"])
            ask_price = float(row["ask_price"])
            bid_size = math.floor(float(row["bid_size"]))
            ask_size = math.floor(float(row["ask_size"]))
            replayable = (
                bid_price > 0
                and ask_price > 0
                and bid_size > 0
                and ask_size > 0
                and ask_price >= bid_price
            )
            if bool(row["is_replayable"]) != replayable:
                raise ValueError("stored quote replayability flag conflicts with quote values")
            if not replayable:
                rejected_one_sided += 1
                continue
            quote_event = NbboQuote(
                ticker=symbol,
                timestamp=row["timestamp"].astimezone(UTC),
                sequence=int(row["sequence_number"]),
                bid_price=bid_price,
                ask_price=ask_price,
                bid_size=bid_size,
                ask_size=ask_size,
            )
            self._deduplicate(quote_event, target=quote_by_identity, kind="quote")
        quotes = tuple(
            sorted(
                quote_by_identity.values(),
                key=lambda item: (item.timestamp, item.sequence),
            )
        )
        return quotes, source_quote_count, rejected_one_sided

    def _load_trades(
        self,
        rows: tuple[dict[str, Any], ...],
        *,
        symbol: str,
        start_at: datetime,
        end_at: datetime,
        opening_auction_condition_codes: frozenset[int],
    ) -> tuple[tuple[TradePrint, ...], int, int, int, float]:
        trade_by_identity: dict[tuple[datetime, int], TradePrint] = {}
        source_trade_count = 0
        rejected_corrections = 0
        rejected_subshare = 0
        fractional_discarded = 0.0
        for row in rows:
            if not self._matches(row, symbol=symbol, start_at=start_at, end_at=end_at):
                continue
            source_trade_count += 1
            correction = row["correction"]
            if correction is not None and int(correction) != 0:
                rejected_corrections += 1
                continue
            raw_size = float(row["size"])
            whole_size = math.floor(raw_size)
            fractional_discarded += raw_size - whole_size
            if whole_size == 0:
                rejected_subshare += 1
                continue
            conditions = frozenset(int(item) for item in row["conditions"])
            trade_event = TradePrint(
                ticker=symbol,
                timestamp=row["timestamp"].astimezone(UTC),
                sequence=int(row["sequence_number"]),
                price=float(row["price"]),
                size=whole_size,
                is_opening_auction=bool(conditions & opening_auction_condition_codes),
            )
            self._deduplicate(trade_event, target=trade_by_identity, kind="trade")
        trades = tuple(
            sorted(
                trade_by_identity.values(),
                key=lambda item: (item.timestamp, item.sequence),
            )
        )
        return (
            trades,
            source_trade_count,
            rejected_corrections,
            rejected_subshare,
            fractional_discarded,
        )

    @staticmethod
    def _read_rows(
        paths: Sequence[Path],
        *,
        required: set[str],
        kind: str,
    ) -> tuple[dict[str, Any], ...]:
        rows: list[dict[str, Any]] = []
        for path in sorted(paths):
            table = pq.read_table(path)  # type: ignore[no-untyped-call]
            if not required.issubset(table.column_names):
                raise ValueError(f"{kind} file is missing required columns: {path}")
            rows.extend(table.select(sorted(required)).to_pylist())
        return tuple(rows)

    @staticmethod
    def _matches(
        row: dict[str, Any],
        *,
        symbol: str,
        start_at: datetime,
        end_at: datetime,
    ) -> bool:
        timestamp = row["timestamp"].astimezone(UTC)
        return str(row["symbol"]).strip().upper() == symbol and start_at <= timestamp <= end_at

    @staticmethod
    def _deduplicate(
        event: EventT,
        *,
        target: dict[tuple[datetime, int], EventT],
        kind: str,
    ) -> None:
        identity = (event.timestamp, event.sequence)
        previous = target.get(identity)
        if previous is not None and previous != event:
            raise ValueError(f"conflicting {kind} event for timestamp/sequence {identity}")
        target[identity] = event
