"""Point-in-time loaders that turn silver observations into feature contexts."""

from __future__ import annotations

from datetime import UTC
from typing import TYPE_CHECKING, Any, ClassVar, Literal

import pyarrow.parquet as pq

from quant_earning_edge.features.registry import (
    EarningsObservation,
    FeatureContext,
    PriceBar,
)

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


class EarningsFeatureLoader:
    """Attach current candidate timing and prior reported EPS without leakage."""

    _CANDIDATE_REQUIRED: ClassVar[set[str]] = {
        "trade_date",
        "symbol",
        "event_date",
        "timing",
        "decision_at",
    }
    _HISTORY_REQUIRED: ClassVar[set[str]] = {
        "symbol",
        "event_date",
        "timing",
        "eps_actual",
        "eps_estimate",
        "ingested_at",
    }

    def enrich(
        self,
        contexts: Sequence[FeatureContext],
        *,
        candidate_files: Sequence[Path],
        earnings_files: Sequence[Path],
        observed_at: datetime,
        target_date: date,
    ) -> tuple[FeatureContext, ...]:
        """Add exactly one current event plus prior known reported events."""
        if observed_at.tzinfo is None or observed_at.utcoffset() is None:
            raise ValueError("observed_at must be timezone-aware")
        if not candidate_files or not earnings_files:
            raise ValueError("candidate and earnings files are required")
        cutoff = observed_at.astimezone(UTC)
        candidates = self._read_candidates(candidate_files, cutoff=cutoff)
        history = self._read_history(earnings_files, cutoff=cutoff)
        enriched: list[FeatureContext] = []
        for context in contexts:
            if target_date <= context.asof_date:
                raise ValueError("target_date must be after every feature asof_date")
            key = (context.symbol, target_date)
            current = candidates.get(key)
            if current is None:
                raise ValueError(f"no event candidate found for {key}")
            prior = [
                EarningsObservation(
                    event_date=row["event_date"],
                    effective_trade_date=row["event_date"],
                    timing=_parse_timing(row["timing"]),
                    eps_actual=row["eps_actual"],
                    eps_estimate=row["eps_estimate"],
                )
                for row in history.get(context.symbol, ())
                if row["event_date"] < current.event_date
                and row["eps_actual"] is not None
                and row["eps_estimate"] is not None
            ]
            observations = tuple(
                sorted(
                    (
                        *prior,
                        current,
                    ),
                    key=lambda item: (
                        item.effective_trade_date,
                        item.event_date,
                        item.timing,
                    ),
                )
            )
            enriched.append(
                FeatureContext(
                    symbol=context.symbol,
                    asof_date=context.asof_date,
                    bars=context.bars,
                    target_date=target_date,
                    earnings=observations,
                )
            )
        return tuple(enriched)

    def _read_candidates(
        self,
        paths: Sequence[Path],
        *,
        cutoff: datetime,
    ) -> dict[tuple[str, date], EarningsObservation]:
        candidates: dict[tuple[str, date], EarningsObservation] = {}
        for path in sorted(paths):
            table = pq.read_table(path)  # type: ignore[no-untyped-call]
            if not self._CANDIDATE_REQUIRED.issubset(table.column_names):
                raise ValueError(f"candidate file is missing required columns: {path}")
            for row in table.select(sorted(self._CANDIDATE_REQUIRED)).to_pylist():
                if row["decision_at"] > cutoff:
                    continue
                key = (str(row["symbol"]).upper(), row["trade_date"])
                observation = EarningsObservation(
                    event_date=row["event_date"],
                    effective_trade_date=row["trade_date"],
                    timing=_parse_timing(row["timing"]),
                )
                previous = candidates.get(key)
                if previous is not None and previous != observation:
                    raise ValueError(f"conflicting current earnings candidates for {key}")
                candidates[key] = observation
        return candidates

    def _read_history(
        self,
        paths: Sequence[Path],
        *,
        cutoff: datetime,
    ) -> dict[str, tuple[dict[str, Any], ...]]:
        latest: dict[tuple[str, date, str], dict[str, Any]] = {}
        for path in sorted(paths):
            table = pq.read_table(path)  # type: ignore[no-untyped-call]
            if not self._HISTORY_REQUIRED.issubset(table.column_names):
                raise ValueError(f"earnings file is missing required columns: {path}")
            for row in table.select(sorted(self._HISTORY_REQUIRED)).to_pylist():
                if row["ingested_at"] > cutoff:
                    continue
                key = (str(row["symbol"]).upper(), row["event_date"], str(row["timing"]))
                previous = latest.get(key)
                if previous is None or previous["ingested_at"] < row["ingested_at"]:
                    latest[key] = row
                elif previous["ingested_at"] == row["ingested_at"] and previous != row:
                    raise ValueError(f"conflicting earnings history revisions for {key}")
        grouped: dict[str, list[dict[str, Any]]] = {}
        for (symbol, _event_date, _timing), row in latest.items():
            grouped.setdefault(symbol, []).append(row)
        return {
            symbol: tuple(sorted(rows, key=lambda row: row["event_date"]))
            for symbol, rows in grouped.items()
        }


def _parse_timing(value: object) -> Literal["bmo", "amc", "dmh"]:
    if value not in {"bmo", "amc", "dmh"}:
        raise ValueError(f"invalid earnings timing: {value!r}")
    return value
