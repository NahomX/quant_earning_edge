"""Cutoff-aware silver input loader for forward labels."""

from __future__ import annotations

from datetime import UTC, date
from typing import TYPE_CHECKING, Any, ClassVar

import pyarrow.parquet as pq

from quant_earning_edge.data import (
    SplitHistorySourceCapture,
    SplitHistorySourceManifest,
    causally_adjust_daily_bar_rows,
)
from quant_earning_edge.labels.forward import LabelBar

if TYPE_CHECKING:
    from collections.abc import Sequence
    from datetime import datetime
    from pathlib import Path


class LabelBarsLoader:
    """Resolve latest bar revisions known when labels are materialized."""

    _REQUIRED: ClassVar[set[str]] = {
        "session_date",
        "symbol",
        "open",
        "close",
        "volume",
        "adjusted",
        "available_at",
        "ingested_at",
    }

    def load(  # noqa: PLR0912 - one strict causal input boundary.
        self,
        paths: Sequence[Path],
        *,
        symbols: Sequence[str],
        start_date: date,
        end_date: date,
        observed_at: datetime,
        split_source_manifest: Path | None = None,
    ) -> tuple[LabelBar, ...]:
        """Load an inclusive session interval at an explicit observation cutoff."""
        if end_date < start_date:
            raise ValueError("end_date must be on or after start_date")
        if observed_at.tzinfo is None or observed_at.utcoffset() is None:
            raise ValueError("observed_at must be timezone-aware")
        if not paths:
            raise ValueError("at least one daily-bars file is required")
        normalized_symbols = frozenset(item.strip().upper() for item in symbols)
        if not normalized_symbols or "" in normalized_symbols:
            raise ValueError("symbols must not be empty")
        cutoff = observed_at.astimezone(UTC)
        latest: dict[tuple[str, date], dict[str, Any]] = {}
        for path in sorted(paths):
            table = pq.read_table(path)  # type: ignore[no-untyped-call]
            if not self._REQUIRED.issubset(table.column_names):
                raise ValueError(f"daily-bars file is missing required columns: {path}")
            for row in table.select(sorted(self._REQUIRED)).to_pylist():
                symbol = str(row["symbol"]).strip().upper()
                session_date = row["session_date"]
                if (
                    symbol not in normalized_symbols
                    or not start_date <= session_date <= end_date
                    or row["available_at"] > cutoff
                ):
                    continue
                key = (symbol, session_date)
                previous = latest.get(key)
                if previous is None or previous["ingested_at"] < row["ingested_at"]:
                    latest[key] = row
                elif previous["ingested_at"] == row["ingested_at"] and previous != row:
                    raise ValueError(f"conflicting label bar revisions for {key}")
        rows = list(latest.values())
        adjustment_modes = {bool(row["adjusted"]) for row in rows}
        if len(adjustment_modes) > 1:
            raise ValueError("forward labels cannot mix daily-bar adjustment modes")
        adjusted = adjustment_modes.pop() if adjustment_modes else True
        if adjusted and split_source_manifest is not None:
            raise ValueError("split history is only valid with raw forward-label bars")
        if not adjusted:
            if split_source_manifest is None:
                raise ValueError("raw forward-label bars require a split-history source")
            split_source = SplitHistorySourceManifest.load(split_source_manifest)
            if (
                date.fromisoformat(split_source.raw["start_date"]) > start_date
                or date.fromisoformat(split_source.raw["end_date"]) < end_date
            ):
                raise ValueError("forward-label split history does not cover the label horizon")
            SplitHistorySourceCapture.reproduce(
                split_source,
                data_lake_root=split_source.data_lake_root,
            )
            rows = list(
                causally_adjust_daily_bar_rows(
                    rows,
                    splits=split_source.splits(data_lake_root=split_source.data_lake_root),
                    basis_date=end_date,
                )
            )
        normalized = {
            (str(row["symbol"]).strip().upper(), row["session_date"]): row for row in rows
        }
        return tuple(
            LabelBar(
                symbol=symbol,
                session_date=session_date,
                open=float(row["open"]),
                close=float(row["close"]),
            )
            for (symbol, session_date), row in sorted(
                normalized.items(),
                key=lambda item: (item[0][1], item[0][0]),
            )
        )
