"""Deterministic computation and long-form gold feature persistence."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import pyarrow as pa
import pyarrow.parquet as pq

from quant_earning_edge.features.registry import FEATURE_REGISTRY, FeatureContext

if TYPE_CHECKING:
    from collections.abc import Sequence
    from datetime import date
    from pathlib import Path

    from quant_earning_edge.data.layout import LakehouseLayout
    from quant_earning_edge.features.registry import FeatureRegistry

FEATURE_VALUE_SCHEMA = pa.schema(
    [
        pa.field("symbol", pa.string(), nullable=False),
        pa.field("asof_date", pa.date32(), nullable=False),
        pa.field("feature_name", pa.string(), nullable=False),
        pa.field("value", pa.float64(), nullable=False),
        pa.field("feature_code_hash", pa.string(), nullable=False),
        pa.field("input_sha256", pa.string(), nullable=False),
        pa.field("computed_at", pa.timestamp("us", tz="UTC"), nullable=False),
    ]
)


@dataclass(frozen=True)
class FeatureValue:
    """One scalar value with implementation and PIT-input lineage."""

    symbol: str
    asof_date: date
    feature_name: str
    value: float
    feature_code_hash: str
    input_sha256: str


@dataclass(frozen=True)
class FeatureArtifact:
    """Immutable gold feature artifact."""

    path: Path
    sha256: str
    row_count: int
    feature_names: tuple[str, ...]


class FeatureEngine:
    """Compute selected registered features from explicit contexts."""

    def __init__(self, registry: FeatureRegistry = FEATURE_REGISTRY) -> None:
        self._registry = registry

    def compute(
        self,
        contexts: Sequence[FeatureContext],
        *,
        feature_names: Sequence[str],
    ) -> tuple[FeatureValue, ...]:
        """Compute every selected feature for each unique symbol/date context."""
        if not contexts:
            raise ValueError("at least one feature context is required")
        specs = self._registry.select(feature_names)
        keys: set[tuple[str, date]] = set()
        values: list[FeatureValue] = []
        for context in sorted(contexts, key=lambda item: (item.asof_date, item.symbol)):
            key = (context.symbol, context.asof_date)
            if key in keys:
                raise ValueError(f"duplicate feature context: {key}")
            keys.add(key)
            input_hash = _context_hash(context)
            for spec in specs:
                values.append(
                    FeatureValue(
                        symbol=context.symbol,
                        asof_date=context.asof_date,
                        feature_name=spec.name,
                        value=spec.evaluate(context),
                        feature_code_hash=spec.code_hash,
                        input_sha256=input_hash,
                    )
                )
        return tuple(values)


class FeatureStore:
    """Write one content-addressed long-form file per feature group/month."""

    def __init__(self, layout: LakehouseLayout) -> None:
        self._layout = layout

    def write(
        self,
        *,
        feature_group: str,
        values: Sequence[FeatureValue],
        computed_at: datetime,
    ) -> FeatureArtifact:
        """Persist values after enforcing a single as-of month and unique keys."""
        if not feature_group.strip():
            raise ValueError("feature_group must not be empty")
        if computed_at.tzinfo is None or computed_at.utcoffset() is None:
            raise ValueError("computed_at must be timezone-aware")
        if not values:
            raise ValueError("feature artifact must contain values")
        normalized_time = computed_at.astimezone(UTC)
        months = {(item.asof_date.year, item.asof_date.month) for item in values}
        if len(months) != 1:
            raise ValueError("one feature artifact cannot span multiple as-of months")
        keys = [(item.symbol, item.asof_date, item.feature_name) for item in values]
        if len(keys) != len(set(keys)):
            raise ValueError("feature values contain duplicate symbol/date/name keys")
        records = [
            {
                "symbol": item.symbol,
                "asof_date": item.asof_date,
                "feature_name": item.feature_name,
                "value": item.value,
                "feature_code_hash": item.feature_code_hash,
                "input_sha256": item.input_sha256,
                "computed_at": normalized_time,
            }
            for item in sorted(
                values,
                key=lambda row: (row.asof_date, row.symbol, row.feature_name),
            )
        ]
        digest = _digest(records)
        partition = self._layout.gold(
            feature_group=feature_group,
            asof_month=values[0].asof_date,
        )
        partition.mkdir(parents=True, exist_ok=True)
        path = partition / f"part-{digest[:20]}.parquet"
        table = pa.Table.from_pylist(records, schema=FEATURE_VALUE_SCHEMA)
        try:
            with path.open("xb") as sink:
                pq.write_table(table, sink, compression="zstd")  # type: ignore[no-untyped-call]
        except FileExistsError:
            if pq.read_schema(path) != FEATURE_VALUE_SCHEMA:  # type: ignore[no-untyped-call]
                raise RuntimeError(f"feature artifact schema collision at {path}") from None
        return FeatureArtifact(
            path=path,
            sha256=digest,
            row_count=table.num_rows,
            feature_names=tuple(sorted({item.feature_name for item in values})),
        )


def _context_hash(context: FeatureContext) -> str:
    known = [
        {
            "session_date": item.session_date,
            "close": item.close,
            "volume": item.volume,
            "vwap": item.vwap,
        }
        for item in context.bars
        if item.session_date <= context.asof_date
    ]
    return _digest(
        {
            "symbol": context.symbol,
            "asof_date": context.asof_date,
            "target_date": context.target_date,
            "bars": known,
            "earnings": [
                {
                    "event_date": item.event_date,
                    "effective_trade_date": item.effective_trade_date,
                    "timing": item.timing,
                    "eps_actual": item.eps_actual,
                    "eps_estimate": item.eps_estimate,
                }
                for item in context.earnings_history()
            ],
        }
    )


def _digest(value: Any) -> str:
    encoded = json.dumps(
        value,
        default=lambda item: item.isoformat(),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()
