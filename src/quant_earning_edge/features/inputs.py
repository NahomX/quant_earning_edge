"""Point-in-time loaders that turn silver observations into feature contexts."""

from __future__ import annotations

from datetime import UTC
from typing import TYPE_CHECKING, Any, ClassVar

import pyarrow.parquet as pq

from quant_earning_edge.features.registry import FeatureContext, PriceBar

if TYPE_CHECKING:
    from collections.abc import Sequence
    from datetime import date, datetime
    from pathlib import Path


class DailyBarsFeatureLoader:
    """Resolve the latest known silver bar revision at an explicit cutoff."""

    _REQUIRED: ClassVar[set[str]] = {
        "session_date",
        "symbol",
        "close",
        "volume",
        "vwap",
        "ingested_at",
    }

    def load(
        self,
        paths: Sequence[Path],
        *,
        symbols: Sequence[str],
        asof_date: date,
        observed_at: datetime,
    ) -> tuple[FeatureContext, ...]:
        """Build contexts while excluding future sessions and future ingestion."""
        if observed_at.tzinfo is None or observed_at.utcoffset() is None:
            raise ValueError("observed_at must be timezone-aware")
        if not paths:
            raise ValueError("at least one daily-bars file is required")
        normalized_symbols = tuple(sorted({item.strip().upper() for item in symbols}))
        if not normalized_symbols or any(not item for item in normalized_symbols):
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
                    or session_date > asof_date
                    or row["ingested_at"] > cutoff
                ):
                    continue
                key = (symbol, session_date)
                previous = latest.get(key)
                if previous is None or previous["ingested_at"] < row["ingested_at"]:
                    latest[key] = row
                elif previous["ingested_at"] == row["ingested_at"] and previous != row:
                    raise ValueError(f"conflicting daily-bar revisions for {key}")
        contexts: list[FeatureContext] = []
        for symbol in normalized_symbols:
            rows = sorted(
                (row for (row_symbol, _date), row in latest.items() if row_symbol == symbol),
                key=lambda row: row["session_date"],
            )
            if not rows:
                raise ValueError(f"no PIT daily bars found for {symbol}")
            contexts.append(
                FeatureContext(
                    symbol=symbol,
                    asof_date=asof_date,
                    bars=tuple(
                        PriceBar(
                            session_date=row["session_date"],
                            close=float(row["close"]),
                            volume=float(row["volume"]),
                            vwap=float(row["vwap"]) if row["vwap"] is not None else None,
                        )
                        for row in rows
                    ),
                )
            )
        return tuple(contexts)
