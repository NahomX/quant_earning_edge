"""Cutoff-aware silver input loader for forward labels."""

from __future__ import annotations

from datetime import UTC
from typing import TYPE_CHECKING, Any, ClassVar

import pyarrow.parquet as pq

from quant_earning_edge.labels.forward import LabelBar

if TYPE_CHECKING:
    from collections.abc import Sequence
    from datetime import date, datetime
    from pathlib import Path


class LabelBarsLoader:
    """Resolve latest bar revisions known when labels are materialized."""

    _REQUIRED: ClassVar[set[str]] = {
        "session_date",
        "symbol",
        "open",
        "close",
        "ingested_at",
    }

    def load(
        self,
        paths: Sequence[Path],
        *,
        symbols: Sequence[str],
        start_date: date,
        end_date: date,
        observed_at: datetime,
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
                    or row["ingested_at"] > cutoff
                ):
                    continue
                key = (symbol, session_date)
                previous = latest.get(key)
                if previous is None or previous["ingested_at"] < row["ingested_at"]:
                    latest[key] = row
                elif previous["ingested_at"] == row["ingested_at"] and previous != row:
                    raise ValueError(f"conflicting label bar revisions for {key}")
        return tuple(
            LabelBar(
                symbol=symbol,
                session_date=session_date,
                open=float(row["open"]),
                close=float(row["close"]),
            )
            for (symbol, session_date), row in sorted(
                latest.items(),
                key=lambda item: (item[0][1], item[0][0]),
            )
        )
