"""Schema-stable, partitioned Parquet persistence for validated records."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import pyarrow as pa
import pyarrow.parquet as pq

if TYPE_CHECKING:
    from datetime import date
    from pathlib import Path

    from quant_earning_edge.data.clients.finnhub import EarningsEvent
    from quant_earning_edge.data.clients.polygon import (
        CashDividend,
        EquityBar,
        MinuteBar,
        StockSplit,
    )
    from quant_earning_edge.data.layout import LakehouseLayout

DAILY_BARS_SCHEMA = pa.schema(
    [
        pa.field("session_date", pa.date32(), nullable=False),
        pa.field("timestamp", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("symbol", pa.string(), nullable=False),
        pa.field("open", pa.float64(), nullable=False),
        pa.field("high", pa.float64(), nullable=False),
        pa.field("low", pa.float64(), nullable=False),
        pa.field("close", pa.float64(), nullable=False),
        pa.field("volume", pa.float64(), nullable=False),
        pa.field("vwap", pa.float64()),
        pa.field("transactions", pa.int64()),
        pa.field("adjusted", pa.bool_(), nullable=False),
        pa.field("source", pa.string(), nullable=False),
        pa.field("ingested_at", pa.timestamp("us", tz="UTC"), nullable=False),
    ]
)

MINUTE_BARS_SCHEMA = pa.schema(
    [
        pa.field("timestamp", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("symbol", pa.string(), nullable=False),
        pa.field("open", pa.float64(), nullable=False),
        pa.field("high", pa.float64(), nullable=False),
        pa.field("low", pa.float64(), nullable=False),
        pa.field("close", pa.float64(), nullable=False),
        pa.field("volume", pa.float64(), nullable=False),
        pa.field("vwap", pa.float64()),
        pa.field("transactions", pa.int64()),
        pa.field("adjusted", pa.bool_(), nullable=False),
        pa.field("source", pa.string(), nullable=False),
        pa.field("ingested_at", pa.timestamp("us", tz="UTC"), nullable=False),
    ]
)

EARNINGS_SCHEMA = pa.schema(
    [
        pa.field("event_date", pa.date32(), nullable=False),
        pa.field("symbol", pa.string(), nullable=False),
        pa.field("timing", pa.string(), nullable=False),
        pa.field("year", pa.int16(), nullable=False),
        pa.field("quarter", pa.int8(), nullable=False),
        pa.field("eps_actual", pa.float64()),
        pa.field("eps_estimate", pa.float64()),
        pa.field("revenue_actual", pa.float64()),
        pa.field("revenue_estimate", pa.float64()),
        pa.field("source", pa.string(), nullable=False),
        pa.field("ingested_at", pa.timestamp("us", tz="UTC"), nullable=False),
    ]
)

SPLITS_SCHEMA = pa.schema(
    [
        pa.field("execution_date", pa.date32(), nullable=False),
        pa.field("event_id", pa.string(), nullable=False),
        pa.field("symbol", pa.string(), nullable=False),
        pa.field("adjustment_type", pa.string(), nullable=False),
        pa.field("split_from", pa.float64(), nullable=False),
        pa.field("split_to", pa.float64(), nullable=False),
        pa.field("historical_adjustment_factor", pa.float64()),
        pa.field("source", pa.string(), nullable=False),
        pa.field("ingested_at", pa.timestamp("us", tz="UTC"), nullable=False),
    ]
)

DIVIDENDS_SCHEMA = pa.schema(
    [
        pa.field("ex_dividend_date", pa.date32(), nullable=False),
        pa.field("event_id", pa.string(), nullable=False),
        pa.field("symbol", pa.string(), nullable=False),
        pa.field("distribution_type", pa.string(), nullable=False),
        pa.field("cash_amount", pa.float64(), nullable=False),
        pa.field("currency", pa.string(), nullable=False),
        pa.field("frequency", pa.int16(), nullable=False),
        pa.field("declaration_date", pa.date32()),
        pa.field("record_date", pa.date32()),
        pa.field("pay_date", pa.date32()),
        pa.field("split_adjusted_cash_amount", pa.float64()),
        pa.field("historical_adjustment_factor", pa.float64()),
        pa.field("source", pa.string(), nullable=False),
        pa.field("ingested_at", pa.timestamp("us", tz="UTC"), nullable=False),
    ]
)


@dataclass(frozen=True)
class SilverArtifact:
    """Identity and row metadata for one silver Parquet partition file."""

    path: Path
    sha256: str
    row_count: int
    schema: pa.Schema


class SilverWriter:
    """Write validated records to deterministic, content-addressed Parquet."""

    def __init__(self, layout: LakehouseLayout) -> None:
        self._layout = layout

    def write_earnings(
        self,
        events: tuple[EarningsEvent, ...],
        *,
        ingested_at: datetime | None = None,
    ) -> tuple[SilverArtifact, ...]:
        """Write one immutable file per event-date partition."""
        observed_at = ingested_at or datetime.now(UTC)
        if observed_at.tzinfo is None or observed_at.utcoffset() is None:
            raise ValueError("ingested_at must be timezone-aware")
        observed_at = observed_at.astimezone(UTC)

        grouped: dict[date, list[EarningsEvent]] = defaultdict(list)
        for event in events:
            grouped[event.event_date].append(event)

        artifacts = [
            self._write_earnings_partition(
                event_date=event_date,
                events=partition_events,
                ingested_at=observed_at,
            )
            for event_date, partition_events in sorted(grouped.items())
        ]
        return tuple(artifacts)

    def write_daily_bars(
        self,
        bars: tuple[EquityBar, ...],
        *,
        ingested_at: datetime | None = None,
    ) -> tuple[SilverArtifact, ...]:
        """Write split-adjusted bars to one immutable file per session."""
        observed_at = ingested_at or datetime.now(UTC)
        if observed_at.tzinfo is None or observed_at.utcoffset() is None:
            raise ValueError("ingested_at must be timezone-aware")
        observed_at = observed_at.astimezone(UTC)

        grouped: dict[date, list[EquityBar]] = defaultdict(list)
        for bar in bars:
            if not bar.adjusted:
                raise ValueError("silver daily bars must be split-adjusted")
            grouped[bar.session_date].append(bar)

        artifacts = [
            self._write_daily_bars_partition(
                session_date=session_date,
                bars=partition_bars,
                ingested_at=observed_at,
            )
            for session_date, partition_bars in sorted(grouped.items())
        ]
        return tuple(artifacts)

    def write_splits(
        self,
        events: tuple[StockSplit, ...],
        *,
        ingested_at: datetime | None = None,
    ) -> tuple[SilverArtifact, ...]:
        """Write one immutable split file per execution-date partition."""
        observed_at = self._observed_at(ingested_at)
        grouped: dict[date, list[StockSplit]] = defaultdict(list)
        for event in events:
            grouped[event.execution_date].append(event)
        return tuple(
            self._write_split_partition(
                execution_date=execution_date,
                events=partition_events,
                ingested_at=observed_at,
            )
            for execution_date, partition_events in sorted(grouped.items())
        )

    def write_minute_bars(
        self,
        bars: tuple[MinuteBar, ...],
        *,
        event_date: date,
        ingested_at: datetime | None = None,
    ) -> SilverArtifact:
        """Write one immutable minute-bar file for an explicit market date."""
        observed_at = self._observed_at(ingested_at)
        records = [
            {
                "timestamp": bar.timestamp.astimezone(UTC),
                "symbol": bar.symbol,
                "open": bar.open,
                "high": bar.high,
                "low": bar.low,
                "close": bar.close,
                "volume": bar.volume,
                "vwap": bar.vwap,
                "transactions": bar.transactions,
                "adjusted": bar.adjusted,
                "source": "polygon",
                "ingested_at": observed_at,
            }
            for bar in sorted(bars, key=lambda item: (item.symbol, item.timestamp))
        ]
        if not records:
            raise ValueError("minute-bar artifact must not be empty")
        return self._write_table(
            partition=self._layout.silver(
                asset_class="us-equity",
                dataset="minute-bars",
                event_date=event_date,
            ),
            digest=_records_digest(records),
            records=records,
            schema=MINUTE_BARS_SCHEMA,
        )

    def write_dividends(
        self,
        events: tuple[CashDividend, ...],
        *,
        ingested_at: datetime | None = None,
    ) -> tuple[SilverArtifact, ...]:
        """Write one immutable dividend file per ex-date partition."""
        observed_at = self._observed_at(ingested_at)
        grouped: dict[date, list[CashDividend]] = defaultdict(list)
        for event in events:
            grouped[event.ex_dividend_date].append(event)
        return tuple(
            self._write_dividend_partition(
                ex_dividend_date=ex_dividend_date,
                events=partition_events,
                ingested_at=observed_at,
            )
            for ex_dividend_date, partition_events in sorted(grouped.items())
        )

    def _write_daily_bars_partition(
        self,
        *,
        session_date: date,
        bars: list[EquityBar],
        ingested_at: datetime,
    ) -> SilverArtifact:
        records = [
            {
                "session_date": bar.session_date,
                "timestamp": bar.timestamp.astimezone(UTC),
                "symbol": bar.symbol,
                "open": bar.open,
                "high": bar.high,
                "low": bar.low,
                "close": bar.close,
                "volume": bar.volume,
                "vwap": bar.vwap,
                "transactions": bar.transactions,
                "adjusted": bar.adjusted,
                "source": "polygon",
                "ingested_at": ingested_at,
            }
            for bar in sorted(bars, key=lambda item: (item.symbol, item.timestamp))
        ]
        digest = _records_digest(records)
        partition = self._layout.silver(
            asset_class="us-equity",
            dataset="daily-bars",
            event_date=session_date,
        )
        return self._write_table(
            partition=partition,
            digest=digest,
            records=records,
            schema=DAILY_BARS_SCHEMA,
        )

    def _write_earnings_partition(
        self,
        *,
        event_date: date,
        events: list[EarningsEvent],
        ingested_at: datetime,
    ) -> SilverArtifact:
        records = [
            {
                "event_date": event.event_date,
                "symbol": event.symbol,
                "timing": event.timing.value,
                "year": event.year,
                "quarter": event.quarter,
                "eps_actual": event.eps_actual,
                "eps_estimate": event.eps_estimate,
                "revenue_actual": event.revenue_actual,
                "revenue_estimate": event.revenue_estimate,
                "source": "finnhub",
                "ingested_at": ingested_at,
            }
            for event in sorted(
                events,
                key=lambda item: (item.symbol, item.timing.value, item.year, item.quarter),
            )
        ]
        digest = _records_digest(records)
        partition = self._layout.silver(
            asset_class="us-equity",
            dataset="earnings-events",
            event_date=event_date,
        )
        return self._write_table(
            partition=partition,
            digest=digest,
            records=records,
            schema=EARNINGS_SCHEMA,
        )

    def _write_split_partition(
        self,
        *,
        execution_date: date,
        events: list[StockSplit],
        ingested_at: datetime,
    ) -> SilverArtifact:
        records = [
            {
                "execution_date": event.execution_date,
                "event_id": event.event_id,
                "symbol": event.symbol,
                "adjustment_type": event.adjustment_type.value,
                "split_from": event.split_from,
                "split_to": event.split_to,
                "historical_adjustment_factor": event.historical_adjustment_factor,
                "source": "polygon",
                "ingested_at": ingested_at,
            }
            for event in sorted(events, key=lambda item: (item.symbol, item.event_id))
        ]
        return self._write_table(
            partition=self._layout.silver(
                asset_class="us-equity",
                dataset="stock-splits",
                event_date=execution_date,
            ),
            digest=_records_digest(records),
            records=records,
            schema=SPLITS_SCHEMA,
        )

    def _write_dividend_partition(
        self,
        *,
        ex_dividend_date: date,
        events: list[CashDividend],
        ingested_at: datetime,
    ) -> SilverArtifact:
        records = [
            {
                "ex_dividend_date": event.ex_dividend_date,
                "event_id": event.event_id,
                "symbol": event.symbol,
                "distribution_type": event.distribution_type.value,
                "cash_amount": event.cash_amount,
                "currency": event.currency,
                "frequency": event.frequency,
                "declaration_date": event.declaration_date,
                "record_date": event.record_date,
                "pay_date": event.pay_date,
                "split_adjusted_cash_amount": event.split_adjusted_cash_amount,
                "historical_adjustment_factor": event.historical_adjustment_factor,
                "source": "polygon",
                "ingested_at": ingested_at,
            }
            for event in sorted(events, key=lambda item: (item.symbol, item.event_id))
        ]
        return self._write_table(
            partition=self._layout.silver(
                asset_class="us-equity",
                dataset="cash-dividends",
                event_date=ex_dividend_date,
            ),
            digest=_records_digest(records),
            records=records,
            schema=DIVIDENDS_SCHEMA,
        )

    @staticmethod
    def _observed_at(value: datetime | None) -> datetime:
        observed_at = value or datetime.now(UTC)
        if observed_at.tzinfo is None or observed_at.utcoffset() is None:
            raise ValueError("ingested_at must be timezone-aware")
        return observed_at.astimezone(UTC)

    @staticmethod
    def _write_table(
        *,
        partition: Path,
        digest: str,
        records: list[dict[str, Any]],
        schema: pa.Schema,
    ) -> SilverArtifact:
        partition.mkdir(parents=True, exist_ok=True)
        path = partition / f"part-{digest[:20]}.parquet"
        table = pa.Table.from_pylist(records, schema=schema)

        try:
            with path.open("xb") as sink:
                pq.write_table(table, sink, compression="zstd")  # type: ignore[no-untyped-call]
        except FileExistsError:
            existing = pq.read_schema(path)  # type: ignore[no-untyped-call]
            if existing != schema:
                raise RuntimeError(f"Silver artifact schema collision at {path}") from None

        return SilverArtifact(
            path=path,
            sha256=digest,
            row_count=table.num_rows,
            schema=table.schema,
        )


def _records_digest(records: list[dict[str, Any]]) -> str:
    serializable = [
        {
            key: value.isoformat() if hasattr(value, "isoformat") else value
            for key, value in record.items()
        }
        for record in records
    ]
    encoded = json.dumps(
        serializable,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()
