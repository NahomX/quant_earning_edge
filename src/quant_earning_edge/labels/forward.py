"""Session-indexed forward-return labels with explicit future-data lineage."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import pyarrow as pa
import pyarrow.parquet as pq

if TYPE_CHECKING:
    from collections.abc import Sequence
    from datetime import date
    from pathlib import Path

    from quant_earning_edge.data.layout import LakehouseLayout

FORWARD_LABEL_SCHEMA = pa.schema(
    [
        pa.field("symbol", pa.string(), nullable=False),
        pa.field("asof_date", pa.date32(), nullable=False),
        pa.field("target_date", pa.date32(), nullable=False),
        pa.field("horizon_end_date", pa.date32(), nullable=False),
        pa.field("forward_1d_open_to_close", pa.float64(), nullable=False),
        pa.field("forward_1d_close", pa.float64(), nullable=False),
        pa.field("forward_5d_close", pa.float64(), nullable=False),
        pa.field("input_sha256", pa.string(), nullable=False),
        pa.field("computed_at", pa.timestamp("us", tz="UTC"), nullable=False),
    ]
)


@dataclass(frozen=True)
class LabelBar:
    """Adjusted open/close observation for one symbol/session."""

    symbol: str
    session_date: date
    open: float
    close: float

    def __post_init__(self) -> None:
        normalized = self.symbol.strip().upper()
        if not normalized:
            raise ValueError("symbol must not be empty")
        object.__setattr__(self, "symbol", normalized)
        if not math.isfinite(self.open) or self.open <= 0:
            raise ValueError("open must be finite and positive")
        if not math.isfinite(self.close) or self.close <= 0:
            raise ValueError("close must be finite and positive")


@dataclass(frozen=True)
class ForwardLabel:
    """Three documented horizons for one feature key."""

    symbol: str
    asof_date: date
    target_date: date
    horizon_end_date: date
    forward_1d_open_to_close: float
    forward_1d_close: float
    forward_5d_close: float
    input_sha256: str


@dataclass(frozen=True)
class LabelArtifact:
    """Immutable gold label artifact."""

    path: Path
    sha256: str
    row_count: int


class ForwardLabelMaker:
    """Compute labels by explicit market-session offsets, never calendar days."""

    def compute(
        self,
        *,
        keys: Sequence[tuple[str, date]],
        sessions: Sequence[date],
        bars: Sequence[LabelBar],
    ) -> tuple[ForwardLabel, ...]:
        """Compute D+1 open-close, D+1 close, and D+5 close labels."""
        if not keys:
            raise ValueError("at least one symbol/asof key is required")
        session_dates = tuple(sessions)
        if session_dates != tuple(sorted(set(session_dates))):
            raise ValueError("sessions must be unique and ascending")
        normalized_keys = tuple((symbol.strip().upper(), asof_date) for symbol, asof_date in keys)
        if any(not symbol for symbol, _date in normalized_keys):
            raise ValueError("label symbols must not be empty")
        if len(normalized_keys) != len(set(normalized_keys)):
            raise ValueError("label keys contain duplicates")
        bar_map: dict[tuple[str, date], LabelBar] = {}
        for bar in bars:
            key = (bar.symbol, bar.session_date)
            if key in bar_map:
                raise ValueError(f"duplicate label bar: {key}")
            bar_map[key] = bar

        labels: list[ForwardLabel] = []
        for symbol, asof_date in sorted(normalized_keys, key=lambda item: (item[1], item[0])):
            try:
                asof_index = session_dates.index(asof_date)
            except ValueError as error:
                raise ValueError(f"asof_date {asof_date} is absent from sessions") from error
            if asof_index + 5 >= len(session_dates):
                raise ValueError(f"five future sessions are unavailable after {asof_date}")
            target_date = session_dates[asof_index + 1]
            horizon_end_date = session_dates[asof_index + 5]
            required_dates = (
                asof_date,
                target_date,
                session_dates[asof_index + 2],
                session_dates[asof_index + 3],
                session_dates[asof_index + 4],
                horizon_end_date,
            )
            try:
                required = tuple(bar_map[(symbol, item)] for item in required_dates)
            except KeyError as error:
                raise ValueError(f"missing required label bar: {error.args[0]}") from error
            asof_bar, target_bar, *_middle, horizon_bar = required
            labels.append(
                ForwardLabel(
                    symbol=symbol,
                    asof_date=asof_date,
                    target_date=target_date,
                    horizon_end_date=horizon_end_date,
                    forward_1d_open_to_close=target_bar.close / target_bar.open - 1.0,
                    forward_1d_close=target_bar.close / asof_bar.close - 1.0,
                    forward_5d_close=horizon_bar.close / asof_bar.close - 1.0,
                    input_sha256=_digest(
                        [
                            {
                                "symbol": item.symbol,
                                "session_date": item.session_date,
                                "open": item.open,
                                "close": item.close,
                            }
                            for item in required
                        ]
                    ),
                )
            )
        return tuple(labels)


class LabelStore:
    """Persist labels separately from features in monthly gold partitions."""

    def __init__(self, layout: LakehouseLayout) -> None:
        self._layout = layout

    def write(
        self,
        labels: Sequence[ForwardLabel],
        *,
        computed_at: datetime,
    ) -> LabelArtifact:
        """Write one content-addressed artifact for a single as-of month."""
        if not labels:
            raise ValueError("label artifact must contain labels")
        if computed_at.tzinfo is None or computed_at.utcoffset() is None:
            raise ValueError("computed_at must be timezone-aware")
        months = {(item.asof_date.year, item.asof_date.month) for item in labels}
        if len(months) != 1:
            raise ValueError("one label artifact cannot span multiple as-of months")
        normalized_time = computed_at.astimezone(UTC)
        records = [
            {
                "symbol": item.symbol,
                "asof_date": item.asof_date,
                "target_date": item.target_date,
                "horizon_end_date": item.horizon_end_date,
                "forward_1d_open_to_close": item.forward_1d_open_to_close,
                "forward_1d_close": item.forward_1d_close,
                "forward_5d_close": item.forward_5d_close,
                "input_sha256": item.input_sha256,
                "computed_at": normalized_time,
            }
            for item in sorted(labels, key=lambda row: (row.asof_date, row.symbol))
        ]
        if len(records) != len({(row["symbol"], row["asof_date"]) for row in records}):
            raise ValueError("labels contain duplicate symbol/asof keys")
        digest = _digest(records)
        partition = self._layout.gold(
            feature_group="forward-labels",
            asof_month=labels[0].asof_date,
        )
        partition.mkdir(parents=True, exist_ok=True)
        path = partition / f"part-{digest[:20]}.parquet"
        table = pa.Table.from_pylist(records, schema=FORWARD_LABEL_SCHEMA)
        try:
            with path.open("xb") as sink:
                pq.write_table(table, sink, compression="zstd")  # type: ignore[no-untyped-call]
        except FileExistsError:
            if pq.read_schema(path) != FORWARD_LABEL_SCHEMA:  # type: ignore[no-untyped-call]
                raise RuntimeError(f"label artifact schema collision at {path}") from None
        return LabelArtifact(path=path, sha256=digest, row_count=table.num_rows)


def _digest(value: Any) -> str:
    encoded = json.dumps(
        value,
        default=lambda item: item.isoformat(),
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()
