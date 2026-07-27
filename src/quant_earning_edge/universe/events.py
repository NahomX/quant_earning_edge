"""Point-in-time earnings-event candidates from frozen universe inputs."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, date, datetime
from enum import StrEnum
from typing import TYPE_CHECKING, Any

import pyarrow as pa
import pyarrow.parquet as pq

from quant_earning_edge.data.calendar import SessionFileStore
from quant_earning_edge.data.clients import EarningsTiming

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from quant_earning_edge.data.layout import LakehouseLayout

EVENT_CANDIDATE_SCHEMA = pa.schema(
    [
        pa.field("trade_date", pa.date32(), nullable=False),
        pa.field("asof_date", pa.date32(), nullable=False),
        pa.field("decision_at", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("symbol", pa.string(), nullable=False),
        pa.field("event_date", pa.date32(), nullable=False),
        pa.field("timing", pa.string(), nullable=False),
        pa.field("year", pa.int16(), nullable=False),
        pa.field("quarter", pa.int8(), nullable=False),
        pa.field("eps_estimate", pa.float64()),
        pa.field("revenue_estimate", pa.float64()),
        pa.field("universe_snapshot_sha256", pa.string(), nullable=False),
        pa.field("session_file_sha256", pa.string(), nullable=False),
        pa.field("earnings_input_sha256", pa.string(), nullable=False),
    ]
)


class CandidateExclusion(StrEnum):
    """Auditable reasons a relevant event did not become a candidate."""

    NOT_IN_ELIGIBLE_UNIVERSE = "not_in_eligible_universe"
    UNSUPPORTED_DURING_MARKET_HOURS = "unsupported_during_market_hours"


@dataclass(frozen=True)
class EventCandidate:
    """A scheduled earnings event tradable from a frozen prior-close universe."""

    symbol: str
    trade_date: date
    asof_date: date
    decision_at: datetime
    event_date: date
    timing: EarningsTiming
    year: int
    quarter: int
    eps_estimate: float | None
    revenue_estimate: float | None


@dataclass(frozen=True)
class EventCandidateArtifact:
    """Immutable candidate output and its source identities."""

    path: Path
    manifest_path: Path
    sha256: str
    row_count: int
    excluded_counts: dict[CandidateExclusion, int]


class EventCandidateJob:
    """Join explicit sessions, known earnings observations, and one frozen universe."""

    def __init__(self, layout: LakehouseLayout) -> None:
        self._layout = layout

    def run(
        self,
        *,
        trade_date: date,
        decision_at: datetime,
        universe_snapshot: Path,
        session_file: Path,
        earnings_files: Sequence[Path],
    ) -> EventCandidateArtifact:
        """Build candidates without reading observations newer than ``decision_at``."""
        if decision_at.tzinfo is None or decision_at.utcoffset() is None:
            raise ValueError("decision_at must be timezone-aware")
        decision_at = decision_at.astimezone(UTC)
        if not earnings_files:
            raise ValueError("at least one earnings file is required")

        session_artifact = SessionFileStore.load(session_file)
        session_dates = tuple(item.session_date for item in session_artifact.sessions)
        if trade_date not in session_dates:
            raise ValueError(f"trade_date {trade_date} is absent from the session file")
        trade_index = session_dates.index(trade_date)
        if trade_index == 0:
            raise ValueError("session file must include the prior trading session")
        prior_session = session_artifact.sessions[trade_index - 1]
        trade_session = session_artifact.sessions[trade_index]
        asof_date = prior_session.session_date
        if decision_at < prior_session.close_at.astimezone(UTC):
            raise ValueError("decision_at must be on or after the prior session close")
        if decision_at >= trade_session.open_at.astimezone(UTC):
            raise ValueError("decision_at must be before the trade session open")

        universe_hash = _file_sha256(universe_snapshot)
        eligible_symbols = self._read_eligible_universe(
            universe_snapshot,
            trade_date=trade_date,
            asof_date=asof_date,
            decision_at=decision_at,
        )
        events, earnings_hash = self._read_known_events(
            earnings_files,
            decision_at=decision_at,
        )
        candidates: list[EventCandidate] = []
        exclusions = {reason: 0 for reason in CandidateExclusion}
        for event in events:
            timing = EarningsTiming(str(event["timing"]))
            event_date = event["event_date"]
            relevant = (
                timing is EarningsTiming.AFTER_MARKET_CLOSE and event_date == asof_date
            ) or (timing is EarningsTiming.BEFORE_MARKET_OPEN and event_date == trade_date)
            if timing is EarningsTiming.DURING_MARKET_HOURS and event_date == trade_date:
                exclusions[CandidateExclusion.UNSUPPORTED_DURING_MARKET_HOURS] += 1
                continue
            if not relevant:
                continue
            symbol = str(event["symbol"])
            if symbol not in eligible_symbols:
                exclusions[CandidateExclusion.NOT_IN_ELIGIBLE_UNIVERSE] += 1
                continue
            candidates.append(
                EventCandidate(
                    symbol=symbol,
                    trade_date=trade_date,
                    asof_date=asof_date,
                    decision_at=decision_at,
                    event_date=event_date,
                    timing=timing,
                    year=int(event["year"]),
                    quarter=int(event["quarter"]),
                    eps_estimate=event["eps_estimate"],
                    revenue_estimate=event["revenue_estimate"],
                )
            )

        candidates.sort(key=lambda item: (item.symbol, item.event_date, item.timing.value))
        if len({item.symbol for item in candidates}) != len(candidates):
            raise ValueError("multiple relevant earnings events exist for one symbol/trade date")
        return self._write(
            candidates,
            trade_date=trade_date,
            universe_hash=universe_hash,
            session_hash=session_artifact.sha256,
            earnings_hash=earnings_hash,
            exclusions=exclusions,
        )

    @staticmethod
    def _read_eligible_universe(
        path: Path,
        *,
        trade_date: date,
        asof_date: date,
        decision_at: datetime,
    ) -> frozenset[str]:
        table = pq.read_table(path)  # type: ignore[no-untyped-call]
        required = {"trade_date", "asof_date", "generated_at", "symbol", "eligible"}
        if not required.issubset(table.column_names):
            raise ValueError("universe snapshot is missing required columns")
        rows = table.select(sorted(required)).to_pylist()
        if not rows:
            raise ValueError("universe snapshot must not be empty")
        symbols: set[str] = set()
        eligible: set[str] = set()
        for row in rows:
            if row["trade_date"] != trade_date or row["asof_date"] != asof_date:
                raise ValueError("universe snapshot dates do not match the requested sessions")
            generated_at = row["generated_at"]
            if generated_at > decision_at:
                raise ValueError("universe snapshot was generated after decision_at")
            symbol = str(row["symbol"])
            if symbol in symbols:
                raise ValueError(f"duplicate universe symbol: {symbol}")
            symbols.add(symbol)
            if row["eligible"]:
                eligible.add(symbol)
        return frozenset(eligible)

    @staticmethod
    def _read_known_events(
        paths: Sequence[Path],
        *,
        decision_at: datetime,
    ) -> tuple[list[dict[str, Any]], str]:
        required = {
            "event_date",
            "symbol",
            "timing",
            "year",
            "quarter",
            "eps_estimate",
            "revenue_estimate",
            "ingested_at",
        }
        observations: list[dict[str, Any]] = []
        file_hashes: list[str] = []
        for path in sorted(paths):
            file_hashes.append(_file_sha256(path))
            table = pq.read_table(path)  # type: ignore[no-untyped-call]
            if not required.issubset(table.column_names):
                raise ValueError(f"earnings file is missing required columns: {path}")
            observations.extend(table.select(sorted(required)).to_pylist())

        latest: dict[tuple[str, date, str, int, int], dict[str, Any]] = {}
        for row in observations:
            observed_at = row["ingested_at"]
            if observed_at > decision_at:
                continue
            key = (
                str(row["symbol"]),
                row["event_date"],
                str(row["timing"]),
                int(row["year"]),
                int(row["quarter"]),
            )
            previous = latest.get(key)
            if previous is None or previous["ingested_at"] < observed_at:
                latest[key] = row
            elif previous["ingested_at"] == observed_at and previous != row:
                raise ValueError(f"conflicting earnings observations at {observed_at}")
        digest = hashlib.sha256("".join(sorted(file_hashes)).encode()).hexdigest()
        return list(latest.values()), digest

    def _write(
        self,
        candidates: Sequence[EventCandidate],
        *,
        trade_date: date,
        universe_hash: str,
        session_hash: str,
        earnings_hash: str,
        exclusions: dict[CandidateExclusion, int],
    ) -> EventCandidateArtifact:
        records = [
            {
                "trade_date": item.trade_date,
                "asof_date": item.asof_date,
                "decision_at": item.decision_at,
                "symbol": item.symbol,
                "event_date": item.event_date,
                "timing": item.timing.value,
                "year": item.year,
                "quarter": item.quarter,
                "eps_estimate": item.eps_estimate,
                "revenue_estimate": item.revenue_estimate,
                "universe_snapshot_sha256": universe_hash,
                "session_file_sha256": session_hash,
                "earnings_input_sha256": earnings_hash,
            }
            for item in candidates
        ]
        evidence = {
            "trade_date": trade_date,
            "records": records,
            "universe_snapshot_sha256": universe_hash,
            "session_file_sha256": session_hash,
            "earnings_input_sha256": earnings_hash,
            "excluded_counts": {
                reason.value: count for reason, count in exclusions.items() if count
            },
        }
        digest = _digest_records(evidence)
        root = self._layout.root / "gold" / "event-candidates" / f"for_trade_date={trade_date}"
        root.mkdir(parents=True, exist_ok=True)
        path = root / f"candidates-{digest[:20]}.parquet"
        table = pa.Table.from_pylist(records, schema=EVENT_CANDIDATE_SCHEMA)
        try:
            with path.open("xb") as sink:
                pq.write_table(table, sink, compression="zstd")  # type: ignore[no-untyped-call]
        except FileExistsError:
            if pq.read_schema(path) != EVENT_CANDIDATE_SCHEMA:  # type: ignore[no-untyped-call]
                raise RuntimeError(f"candidate artifact schema collision at {path}") from None
        manifest_path = root / f"manifest-{digest[:20]}.json"
        encoded_manifest = json.dumps(
            evidence,
            default=lambda item: item.isoformat(),
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        try:
            with manifest_path.open("xb") as destination:
                destination.write(encoded_manifest)
        except FileExistsError:
            if manifest_path.read_bytes() != encoded_manifest:
                raise RuntimeError(f"candidate manifest collision at {manifest_path}") from None
        return EventCandidateArtifact(
            path=path,
            manifest_path=manifest_path,
            sha256=digest,
            row_count=table.num_rows,
            excluded_counts={reason: count for reason, count in exclusions.items() if count},
        )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _digest_records(records: object) -> str:
    encoded = json.dumps(
        records,
        default=lambda item: item.isoformat(),
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()
