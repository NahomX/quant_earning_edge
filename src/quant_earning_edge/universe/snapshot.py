"""Immutable Parquet persistence for point-in-time universe snapshots."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import pyarrow as pa
import pyarrow.parquet as pq

if TYPE_CHECKING:
    from pathlib import Path

    from quant_earning_edge.data.layout import LakehouseLayout
    from quant_earning_edge.universe.builder import UniverseSnapshot

UNIVERSE_SNAPSHOT_SCHEMA = pa.schema(
    [
        pa.field("trade_date", pa.date32(), nullable=False),
        pa.field("asof_date", pa.date32(), nullable=False),
        pa.field("generated_at", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("config_sha256", pa.string(), nullable=False),
        pa.field("symbol", pa.string(), nullable=False),
        pa.field("eligible", pa.bool_(), nullable=False),
        pa.field("rejection_reasons", pa.list_(pa.string()), nullable=False),
        pa.field("close", pa.float64(), nullable=False),
        pa.field("avg_daily_volume", pa.float64(), nullable=False),
        pa.field("market_cap_usd", pa.float64(), nullable=False),
        pa.field("primary_exchange", pa.string(), nullable=False),
        pa.field("security_type", pa.string(), nullable=False),
        pa.field("active", pa.bool_(), nullable=False),
        pa.field("halted", pa.bool_(), nullable=False),
        pa.field("sector", pa.string(), nullable=False),
        pa.field("list_date", pa.date32()),
        pa.field("delisted_date", pa.date32()),
    ]
)


@dataclass(frozen=True)
class UniverseSnapshotArtifact:
    """Identity and size of one frozen universe artifact."""

    path: Path
    sha256: str
    config_sha256: str
    row_count: int


class UniverseSnapshotWriter:
    """Persist complete decisions, not just survivors, for auditability."""

    def __init__(self, layout: LakehouseLayout) -> None:
        self._layout = layout

    def write(self, snapshot: UniverseSnapshot) -> UniverseSnapshotArtifact:
        """Write an immutable, content-addressed snapshot."""
        config_payload = {
            "min_price": snapshot.config.min_price,
            "min_market_cap_usd": snapshot.config.min_market_cap_usd,
            "min_avg_daily_volume": snapshot.config.min_avg_daily_volume,
            "allowed_exchanges": sorted(snapshot.config.allowed_exchanges),
            "allowed_security_types": sorted(snapshot.config.allowed_security_types),
            "exclude_halts": snapshot.config.exclude_halts,
        }
        config_sha256 = _digest(config_payload)
        records = [
            {
                "trade_date": snapshot.trade_date,
                "asof_date": snapshot.asof_date,
                "generated_at": snapshot.generated_at,
                "config_sha256": config_sha256,
                "symbol": decision.candidate.symbol,
                "eligible": decision.eligible,
                "rejection_reasons": [reason.value for reason in decision.rejection_reasons],
                "close": decision.candidate.close,
                "avg_daily_volume": decision.candidate.avg_daily_volume,
                "market_cap_usd": decision.candidate.market_cap_usd,
                "primary_exchange": decision.candidate.primary_exchange,
                "security_type": decision.candidate.security_type,
                "active": decision.candidate.active,
                "halted": decision.candidate.halted,
                "sector": decision.candidate.sector,
                "list_date": decision.candidate.list_date,
                "delisted_date": decision.candidate.delisted_date,
            }
            for decision in snapshot.decisions
        ]
        snapshot_sha256 = _digest(records)
        partition = self._layout.universe_snapshot(trade_date=snapshot.trade_date)
        partition.mkdir(parents=True, exist_ok=True)
        path = partition / f"snapshot-{snapshot_sha256[:20]}.parquet"
        table = pa.Table.from_pylist(records, schema=UNIVERSE_SNAPSHOT_SCHEMA)
        try:
            with path.open("xb") as sink:
                pq.write_table(table, sink, compression="zstd")  # type: ignore[no-untyped-call]
        except FileExistsError:
            existing = pq.ParquetFile(path).read()  # type: ignore[no-untyped-call]
            if existing.schema != UNIVERSE_SNAPSHOT_SCHEMA or not existing.equals(table):
                raise RuntimeError(f"Universe snapshot schema collision at {path}") from None
        return UniverseSnapshotArtifact(
            path=path,
            sha256=snapshot_sha256,
            config_sha256=config_sha256,
            row_count=table.num_rows,
        )


def _digest(value: Any) -> str:
    def default(item: Any) -> str:
        if hasattr(item, "isoformat"):
            return str(item.isoformat())
        raise TypeError(f"Unsupported digest value: {type(item).__name__}")

    encoded = json.dumps(
        value,
        default=default,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()
