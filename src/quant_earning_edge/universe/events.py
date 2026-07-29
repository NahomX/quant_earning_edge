"""Point-in-time earnings-event candidates from frozen universe inputs."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, date, datetime
from enum import StrEnum
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Any

import pyarrow as pa
import pyarrow.parquet as pq

from quant_earning_edge.data.calendar import SessionFileStore
from quant_earning_edge.data.calendar_source import CalendarSourceManifest
from quant_earning_edge.data.clients import EarningsTiming
from quant_earning_edge.universe.event_source_capture import EventSourceCaptureManifest
from quant_earning_edge.universe.source_capture import UniverseSourceCaptureManifest

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
        pa.field("sector", pa.string(), nullable=False),
        pa.field("sizing_price", pa.float64(), nullable=False),
        pa.field("frozen_average_daily_volume_shares", pa.float64(), nullable=False),
        pa.field("event_date", pa.date32(), nullable=False),
        pa.field("timing", pa.string(), nullable=False),
        pa.field("year", pa.int16(), nullable=False),
        pa.field("quarter", pa.int8(), nullable=False),
        pa.field("eps_estimate", pa.float64()),
        pa.field("revenue_estimate", pa.float64()),
        pa.field("split_event_ids", pa.list_(pa.string()), nullable=False),
        pa.field("dividend_event_ids", pa.list_(pa.string()), nullable=False),
        pa.field("universe_snapshot_sha256", pa.string(), nullable=False),
        pa.field("session_file_sha256", pa.string(), nullable=False),
        pa.field("earnings_input_sha256", pa.string(), nullable=False),
        pa.field("corporate_actions_input_sha256", pa.string(), nullable=False),
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
    sector: str
    sizing_price: float
    frozen_average_daily_volume_shares: float
    trade_date: date
    asof_date: date
    decision_at: datetime
    event_date: date
    timing: EarningsTiming
    year: int
    quarter: int
    eps_estimate: float | None
    revenue_estimate: float | None
    split_event_ids: tuple[str, ...]
    dividend_event_ids: tuple[str, ...]


@dataclass(frozen=True)
class EventCandidateArtifact:
    """Immutable candidate output and its source identities."""

    path: Path
    manifest_path: Path
    sha256: str
    row_count: int
    excluded_counts: dict[CandidateExclusion, int]


@dataclass(frozen=True)
class EventCandidateManifest:
    """Strict candidate-generation inputs and expected output identity."""

    path: Path
    raw: dict[str, Any]

    @classmethod
    def load(cls, path: Path) -> EventCandidateManifest:  # noqa: PLR0912
        try:
            encoded = path.read_bytes()
            raw = json.loads(encoded)
        except (OSError, ValueError) as error:
            raise ValueError(f"invalid event-candidate manifest: {path}") from error
        required = {
            "schema_version",
            "trade_date",
            "decision_at",
            "records",
            "candidate_file_sha256",
            "universe_snapshot_sha256",
            "session_file_sha256",
            "earnings_input_sha256",
            "corporate_actions_input_sha256",
            "candidate_split_overlap_count",
            "candidate_dividend_overlap_count",
            "excluded_counts",
            "source_files",
        }
        if (
            not isinstance(raw, dict)
            or set(raw) != required
            or raw["schema_version"] not in {2, 3, 4, 5}
        ):
            raise ValueError("event-candidate manifest schema mismatch")
        sources = raw["source_files"]
        base_source_keys = {
            "universe_snapshot",
            "session_file",
            "earnings_files",
            "split_files",
            "dividend_files",
        }
        expected_source_keys = set(base_source_keys)
        if raw["schema_version"] >= 3:
            expected_source_keys |= {
                "universe_source_manifest",
                "universe_source_files",
            }
        if raw["schema_version"] >= 4:
            expected_source_keys |= {"event_source_manifest", "event_provider_files"}
        if raw["schema_version"] == 5:
            expected_source_keys |= {"calendar_source_manifest", "calendar_provider_files"}
        if not isinstance(sources, dict) or set(sources) != expected_source_keys:
            raise ValueError("event-candidate manifest source schema mismatch")
        if (
            not isinstance(sources["universe_snapshot"], dict)
            or not isinstance(sources["session_file"], dict)
            or not all(
                isinstance(sources[name], list)
                for name in ("earnings_files", "split_files", "dividend_files")
            )
            or not sources["earnings_files"]
            or not sources["split_files"]
            or not sources["dividend_files"]
            or (
                raw["schema_version"] >= 3
                and (
                    not isinstance(sources["universe_source_manifest"], dict)
                    or not isinstance(sources["universe_source_files"], list)
                    or not sources["universe_source_files"]
                )
            )
            or (
                raw["schema_version"] >= 4
                and (
                    not isinstance(sources["event_source_manifest"], dict)
                    or not isinstance(sources["event_provider_files"], list)
                    or not sources["event_provider_files"]
                )
            )
            or (
                raw["schema_version"] == 5
                and (
                    not isinstance(sources["calendar_source_manifest"], dict)
                    or not isinstance(sources["calendar_provider_files"], list)
                    or not sources["calendar_provider_files"]
                )
            )
        ):
            raise ValueError("event-candidate manifest source collection is invalid")
        entries = (
            *(
                (
                    sources["universe_source_manifest"],
                    *sources["universe_source_files"],
                )
                if raw["schema_version"] >= 3
                else ()
            ),
            *(
                (
                    sources["event_source_manifest"],
                    *sources["event_provider_files"],
                )
                if raw["schema_version"] >= 4
                else ()
            ),
            *(
                (
                    sources["calendar_source_manifest"],
                    *sources["calendar_provider_files"],
                )
                if raw["schema_version"] == 5
                else ()
            ),
            sources["universe_snapshot"],
            sources["session_file"],
            *sources["earnings_files"],
            *sources["split_files"],
            *sources["dividend_files"],
        )
        for entry in entries:
            if (
                not isinstance(entry, dict)
                or set(entry) != {"path", "sha256"}
                or not _is_sha256(entry["sha256"])
                or not str(entry["path"]).strip()
            ):
                raise ValueError("event-candidate manifest source entry is invalid")
            source_path = PurePosixPath(entry["path"])
            if (
                source_path.is_absolute()
                or ".." in source_path.parts
                or source_path.as_posix() != entry["path"]
            ):
                raise ValueError("event-candidate manifest source path is invalid")
        digest_fields = (
            "candidate_file_sha256",
            "universe_snapshot_sha256",
            "session_file_sha256",
            "earnings_input_sha256",
            "corporate_actions_input_sha256",
        )
        if any(not _is_sha256(raw[field]) for field in digest_fields):
            raise ValueError("event-candidate manifest digest is invalid")
        earnings_hash = _aggregate_file_hashes(sources["earnings_files"])
        split_hash = _aggregate_file_hashes(sources["split_files"])
        dividend_hash = _aggregate_file_hashes(sources["dividend_files"])
        corporate_actions_hash = hashlib.sha256(f"{split_hash}{dividend_hash}".encode()).hexdigest()
        if (
            raw["universe_snapshot_sha256"] != sources["universe_snapshot"]["sha256"]
            or raw["session_file_sha256"] != sources["session_file"]["sha256"]
            or raw["earnings_input_sha256"] != earnings_hash
            or raw["corporate_actions_input_sha256"] != corporate_actions_hash
        ):
            raise ValueError("event-candidate manifest aggregate digest differs from its sources")
        try:
            trade_date = date.fromisoformat(str(raw["trade_date"]))
            decision_at = datetime.fromisoformat(str(raw["decision_at"]))
        except ValueError as error:
            raise ValueError("event-candidate manifest timestamps are invalid") from error
        if (
            str(trade_date) != raw["trade_date"]
            or decision_at.tzinfo is None
            or decision_at.utcoffset() is None
            or not isinstance(raw["records"], list)
            or not isinstance(raw["excluded_counts"], dict)
            or not isinstance(raw["candidate_split_overlap_count"], int)
            or not isinstance(raw["candidate_dividend_overlap_count"], int)
        ):
            raise ValueError("event-candidate manifest metadata is invalid")
        canonical = json.dumps(raw, sort_keys=True, separators=(",", ":")).encode()
        if canonical != encoded:
            raise ValueError("event-candidate manifest is not canonical")
        return cls(path=path.resolve(), raw=raw)

    @property
    def source_entries(self) -> tuple[dict[str, str], ...]:
        return (
            *self.universe_lineage_entries,
            *self.event_lineage_entries,
            *self.calendar_lineage_entries,
            *self.candidate_source_entries,
        )

    @property
    def universe_lineage_entries(self) -> tuple[dict[str, str], ...]:
        if self.raw["schema_version"] < 3:
            return ()
        sources = self.raw["source_files"]
        return (
            sources["universe_source_manifest"],
            *sources["universe_source_files"],
        )

    @property
    def event_lineage_entries(self) -> tuple[dict[str, str], ...]:
        if self.raw["schema_version"] < 4:
            return ()
        sources = self.raw["source_files"]
        return (
            sources["event_source_manifest"],
            *sources["event_provider_files"],
        )

    @property
    def calendar_lineage_entries(self) -> tuple[dict[str, str], ...]:
        if self.raw["schema_version"] < 5:
            return ()
        sources = self.raw["source_files"]
        return (
            sources["calendar_source_manifest"],
            *sources["calendar_provider_files"],
        )

    @property
    def candidate_source_entries(self) -> tuple[dict[str, str], ...]:
        sources = self.raw["source_files"]
        return (
            sources["universe_snapshot"],
            sources["session_file"],
            *sources["earnings_files"],
            *sources["split_files"],
            *sources["dividend_files"],
        )

    def source_paths(self, *, data_lake_root: Path) -> tuple[Path, ...]:
        root = data_lake_root.resolve()
        paths = tuple((root / entry["path"]).resolve() for entry in self.source_entries)
        if any(path == root or root not in path.parents for path in paths):
            raise ValueError("event-candidate source escapes the data lake")
        if any(
            _file_sha256(path) != entry["sha256"]
            for path, entry in zip(paths, self.source_entries, strict=True)
        ):
            raise ValueError("event-candidate source file is missing or differs")
        return paths


class EventCandidateJob:
    """Join explicit sessions, known earnings observations, and one frozen universe."""

    def __init__(self, layout: LakehouseLayout, *, source_root: Path | None = None) -> None:
        self._layout = layout
        self._source_root = (source_root or layout.root).resolve()

    def run(  # noqa: PLR0912,PLR0915 - complete causal join boundary.
        self,
        *,
        trade_date: date,
        decision_at: datetime,
        universe_snapshot: Path,
        session_file: Path,
        earnings_files: Sequence[Path],
        split_files: Sequence[Path],
        dividend_files: Sequence[Path],
        universe_source_manifest: Path | None = None,
        event_source_manifest: Path | None = None,
        calendar_source_manifest: Path | None = None,
    ) -> EventCandidateArtifact:
        """Build candidates without reading observations newer than ``decision_at``."""
        if decision_at.tzinfo is None or decision_at.utcoffset() is None:
            raise ValueError("decision_at must be timezone-aware")
        decision_at = decision_at.astimezone(UTC)
        if not earnings_files:
            raise ValueError("at least one earnings file is required")
        if not split_files or not dividend_files:
            raise ValueError("split and dividend files are required for a complete event audit")

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
        universe_lineage = (
            UniverseSourceCaptureManifest.load(universe_source_manifest)
            if universe_source_manifest is not None
            else None
        )
        if universe_lineage is not None and (
            universe_lineage.raw["trade_date"] != trade_date.isoformat()
            or universe_lineage.raw["asof_date"] != asof_date.isoformat()
            or universe_lineage.raw["snapshot_file_sha256"] != universe_hash
        ):
            raise ValueError("universe source manifest differs from the candidate snapshot")
        event_lineage = (
            EventSourceCaptureManifest.load(event_source_manifest)
            if event_source_manifest is not None
            else None
        )
        if event_lineage is not None and universe_lineage is None:
            raise ValueError("event source lineage requires universe source lineage")
        if event_lineage is not None:
            event_silver_paths = event_lineage.silver_paths(data_lake_root=self._source_root)
            expected_event_paths = (
                *tuple(sorted(path.resolve() for path in earnings_files)),
                *tuple(sorted(path.resolve() for path in split_files)),
                *tuple(sorted(path.resolve() for path in dividend_files)),
            )
            if (
                event_lineage.raw["start_date"] != asof_date.isoformat()
                or event_lineage.raw["end_date"] != trade_date.isoformat()
                or datetime.fromisoformat(event_lineage.raw["ingested_at"]) != decision_at
                or event_silver_paths != expected_event_paths
            ):
                raise ValueError("event source manifest differs from candidate event inputs")
        calendar_lineage = (
            CalendarSourceManifest.load(calendar_source_manifest)
            if calendar_source_manifest is not None
            else None
        )
        if calendar_lineage is not None and event_lineage is None:
            raise ValueError("calendar source lineage requires event source lineage")
        if (
            calendar_lineage is not None
            and calendar_lineage.session_path(data_lake_root=self._source_root)
            != session_file.resolve()
        ):
            raise ValueError("calendar source manifest differs from candidate session file")
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
        splits, split_hash = self._read_known_corporate_actions(
            split_files,
            decision_at=decision_at,
            event_date_field="execution_date",
        )
        dividends, dividend_hash = self._read_known_corporate_actions(
            dividend_files,
            decision_at=decision_at,
            event_date_field="ex_dividend_date",
        )
        split_ids = self._actions_by_symbol(splits, event_date=trade_date)
        dividend_ids = self._actions_by_symbol(dividends, event_date=trade_date)
        corporate_actions_hash = hashlib.sha256(f"{split_hash}{dividend_hash}".encode()).hexdigest()
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
                    sector=eligible_symbols[symbol][0],
                    sizing_price=eligible_symbols[symbol][1],
                    frozen_average_daily_volume_shares=eligible_symbols[symbol][2],
                    trade_date=trade_date,
                    asof_date=asof_date,
                    decision_at=decision_at,
                    event_date=event_date,
                    timing=timing,
                    year=int(event["year"]),
                    quarter=int(event["quarter"]),
                    eps_estimate=event["eps_estimate"],
                    revenue_estimate=event["revenue_estimate"],
                    split_event_ids=split_ids.get(symbol, ()),
                    dividend_event_ids=dividend_ids.get(symbol, ()),
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
            corporate_actions_hash=corporate_actions_hash,
            exclusions=exclusions,
            decision_at=decision_at,
            universe_snapshot=universe_snapshot,
            session_file=session_file,
            earnings_files=earnings_files,
            split_files=split_files,
            dividend_files=dividend_files,
            universe_lineage=universe_lineage,
            event_lineage=event_lineage,
            calendar_lineage=calendar_lineage,
        )

    @staticmethod
    def _read_eligible_universe(
        path: Path,
        *,
        trade_date: date,
        asof_date: date,
        decision_at: datetime,
    ) -> dict[str, tuple[str, float, float]]:
        table = pq.ParquetFile(path).read()  # type: ignore[no-untyped-call]
        required = {
            "trade_date",
            "asof_date",
            "generated_at",
            "symbol",
            "eligible",
            "sector",
            "close",
            "avg_daily_volume",
        }
        if not required.issubset(table.column_names):
            raise ValueError("universe snapshot is missing required columns")
        rows = table.select(sorted(required)).to_pylist()
        if not rows:
            raise ValueError("universe snapshot must not be empty")
        symbols: set[str] = set()
        eligible: dict[str, tuple[str, float, float]] = {}
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
                sector = str(row["sector"]).strip().upper()
                close = float(row["close"])
                average_volume = float(row["avg_daily_volume"])
                if not sector or close <= 0 or average_volume <= 0:
                    raise ValueError(f"eligible universe sizing fields are invalid: {symbol}")
                eligible[symbol] = (sector, close, average_volume)
        return eligible

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
            table = pq.ParquetFile(path).read()  # type: ignore[no-untyped-call]
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

    @staticmethod
    def _read_known_corporate_actions(
        paths: Sequence[Path],
        *,
        decision_at: datetime,
        event_date_field: str,
    ) -> tuple[list[dict[str, Any]], str]:
        required = {"event_id", "symbol", event_date_field, "ingested_at"}
        latest: dict[str, dict[str, Any]] = {}
        file_hashes: list[str] = []
        for path in sorted(paths):
            file_hashes.append(_file_sha256(path))
            table = pq.ParquetFile(path).read()  # type: ignore[no-untyped-call]
            if not required.issubset(table.column_names):
                raise ValueError(f"corporate-action file is missing required columns: {path}")
            for row in table.select(sorted(required)).to_pylist():
                if row["ingested_at"] > decision_at:
                    continue
                event_id = str(row["event_id"])
                previous = latest.get(event_id)
                if previous is None or previous["ingested_at"] < row["ingested_at"]:
                    latest[event_id] = row
                elif previous["ingested_at"] == row["ingested_at"] and previous != row:
                    raise ValueError(f"conflicting corporate-action observations: {event_id}")
        digest = hashlib.sha256("".join(sorted(file_hashes)).encode()).hexdigest()
        return list(latest.values()), digest

    @staticmethod
    def _actions_by_symbol(
        actions: Sequence[dict[str, Any]],
        *,
        event_date: date,
    ) -> dict[str, tuple[str, ...]]:
        grouped: dict[str, list[str]] = {}
        for action in actions:
            action_date = action.get("execution_date", action.get("ex_dividend_date"))
            if action_date == event_date:
                grouped.setdefault(str(action["symbol"]), []).append(str(action["event_id"]))
        return {symbol: tuple(sorted(event_ids)) for symbol, event_ids in sorted(grouped.items())}

    def _write(
        self,
        candidates: Sequence[EventCandidate],
        *,
        trade_date: date,
        universe_hash: str,
        session_hash: str,
        earnings_hash: str,
        corporate_actions_hash: str,
        exclusions: dict[CandidateExclusion, int],
        decision_at: datetime,
        universe_snapshot: Path,
        session_file: Path,
        earnings_files: Sequence[Path],
        split_files: Sequence[Path],
        dividend_files: Sequence[Path],
        universe_lineage: UniverseSourceCaptureManifest | None,
        event_lineage: EventSourceCaptureManifest | None,
        calendar_lineage: CalendarSourceManifest | None,
    ) -> EventCandidateArtifact:
        records = [
            {
                "trade_date": item.trade_date,
                "asof_date": item.asof_date,
                "decision_at": item.decision_at,
                "symbol": item.symbol,
                "sector": item.sector,
                "sizing_price": item.sizing_price,
                "frozen_average_daily_volume_shares": (item.frozen_average_daily_volume_shares),
                "event_date": item.event_date,
                "timing": item.timing.value,
                "year": item.year,
                "quarter": item.quarter,
                "eps_estimate": item.eps_estimate,
                "revenue_estimate": item.revenue_estimate,
                "split_event_ids": list(item.split_event_ids),
                "dividend_event_ids": list(item.dividend_event_ids),
                "universe_snapshot_sha256": universe_hash,
                "session_file_sha256": session_hash,
                "earnings_input_sha256": earnings_hash,
                "corporate_actions_input_sha256": corporate_actions_hash,
            }
            for item in candidates
        ]
        core_evidence = {
            "trade_date": trade_date,
            "records": records,
            "universe_snapshot_sha256": universe_hash,
            "session_file_sha256": session_hash,
            "earnings_input_sha256": earnings_hash,
            "corporate_actions_input_sha256": corporate_actions_hash,
            "candidate_split_overlap_count": sum(bool(item.split_event_ids) for item in candidates),
            "candidate_dividend_overlap_count": sum(
                bool(item.dividend_event_ids) for item in candidates
            ),
            "excluded_counts": {
                reason.value: count for reason, count in exclusions.items() if count
            },
        }
        digest = _digest_records(
            {
                **core_evidence,
                "universe_source_manifest_sha256": _file_sha256(universe_lineage.path),
                **(
                    {"event_source_manifest_sha256": _file_sha256(event_lineage.path)}
                    if event_lineage is not None
                    else {}
                ),
                **(
                    {"calendar_source_manifest_sha256": _file_sha256(calendar_lineage.path)}
                    if calendar_lineage is not None
                    else {}
                ),
            }
            if universe_lineage is not None
            else core_evidence
        )
        root = self._layout.root / "gold" / "event-candidates" / f"for_trade_date={trade_date}"
        root.mkdir(parents=True, exist_ok=True)
        path = root / f"candidates-{digest[:20]}.parquet"
        table = pa.Table.from_pylist(records, schema=EVENT_CANDIDATE_SCHEMA)
        try:
            with path.open("xb") as sink:
                pq.write_table(table, sink, compression="zstd")  # type: ignore[no-untyped-call]
        except FileExistsError:
            existing = pq.ParquetFile(path).read()  # type: ignore[no-untyped-call]
            if existing.schema != EVENT_CANDIDATE_SCHEMA or not existing.equals(table):
                raise RuntimeError(f"candidate artifact schema collision at {path}") from None
        universe_source_files = (
            universe_lineage.source_paths(data_lake_root=self._source_root)
            if universe_lineage is not None
            else ()
        )
        event_provider_files = (
            event_lineage.provider_paths(data_lake_root=self._source_root)
            if event_lineage is not None
            else ()
        )
        calendar_provider_files = (
            calendar_lineage.provider_paths(data_lake_root=self._source_root)
            if calendar_lineage is not None
            else ()
        )
        evidence = {
            "schema_version": (
                5
                if calendar_lineage is not None
                and event_lineage is not None
                and universe_lineage is not None
                else 4
                if event_lineage is not None and universe_lineage is not None
                else 3
                if universe_lineage is not None
                else 2
            ),
            "trade_date": trade_date,
            "decision_at": decision_at,
            **core_evidence,
            "candidate_file_sha256": _file_sha256(path),
            "source_files": {
                **(
                    {
                        "universe_source_manifest": self._source_entry(universe_lineage.path),
                        "universe_source_files": [
                            self._source_entry(item) for item in universe_source_files
                        ],
                    }
                    if universe_lineage is not None
                    else {}
                ),
                **(
                    {
                        "calendar_source_manifest": self._source_entry(calendar_lineage.path),
                        "calendar_provider_files": [
                            self._source_entry(item) for item in calendar_provider_files
                        ],
                    }
                    if calendar_lineage is not None
                    else {}
                ),
                **(
                    {
                        "event_source_manifest": self._source_entry(event_lineage.path),
                        "event_provider_files": [
                            self._source_entry(item) for item in event_provider_files
                        ],
                    }
                    if event_lineage is not None
                    else {}
                ),
                "universe_snapshot": self._source_entry(universe_snapshot),
                "session_file": self._source_entry(session_file),
                "earnings_files": [self._source_entry(item) for item in sorted(earnings_files)],
                "split_files": [self._source_entry(item) for item in sorted(split_files)],
                "dividend_files": [self._source_entry(item) for item in sorted(dividend_files)],
            },
        }
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

    def _source_entry(self, path: Path) -> dict[str, str]:
        resolved = path.resolve()
        try:
            relative = resolved.relative_to(self._source_root)
        except ValueError as error:
            raise ValueError("event-candidate source must be inside the data lake") from error
        return {"path": relative.as_posix(), "sha256": _file_sha256(resolved)}


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


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(item in "0123456789abcdef" for item in value)
    )


def _aggregate_file_hashes(entries: Sequence[dict[str, str]]) -> str:
    return hashlib.sha256(
        "".join(sorted(entry["sha256"] for entry in entries)).encode()
    ).hexdigest()
