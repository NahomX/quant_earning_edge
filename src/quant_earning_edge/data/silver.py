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
    from quant_earning_edge.data.layout import LakehouseLayout

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
        partition.mkdir(parents=True, exist_ok=True)
        path = partition / f"part-{digest[:20]}.parquet"
        table = pa.Table.from_pylist(records, schema=EARNINGS_SCHEMA)

        try:
            with path.open("xb") as sink:
                pq.write_table(table, sink, compression="zstd")  # type: ignore[no-untyped-call]
        except FileExistsError:
            existing = pq.read_schema(path)  # type: ignore[no-untyped-call]
            if existing != EARNINGS_SCHEMA:
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
