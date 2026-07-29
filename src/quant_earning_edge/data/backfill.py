"""Resumable, content-stable historical bar backfills and coverage evidence."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime
from enum import StrEnum
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any, Protocol, cast
from uuid import uuid4

import pyarrow.parquet as pq

from quant_earning_edge.data.bars_source import (
    DailyBarsSourceCapture,
    DailyBarsSourceManifest,
)
from quant_earning_edge.data.layout import LakehouseLayout
from quant_earning_edge.data.silver import (
    DAILY_BAR_SESSION_CLOSE_15M,
    DAILY_BARS_SCHEMA,
)

MIN_FIVE_YEAR_SESSIONS = 1_200

if TYPE_CHECKING:
    from collections.abc import Callable

    from quant_earning_edge.data.bronze import BronzeArtifact
    from quant_earning_edge.data.clients.polygon import EquityBar
    from quant_earning_edge.data.silver import SilverArtifact, SilverWriter


class BarsProvider(Protocol):
    """Provider operation required by a historical bar backfill."""

    def daily_bars(
        self,
        *,
        symbol: str,
        start_date: date,
        end_date: date,
        adjusted: bool = False,
    ) -> tuple[EquityBar, ...]: ...


class BackfillEventStatus(StrEnum):
    """Terminal state of one deterministic symbol batch attempt."""

    SUCCESS = "success"
    FAILURE = "failure"


@dataclass(frozen=True)
class BarBackfillPlan:
    """Immutable specification whose identity is independent of retry time."""

    plan_id: str
    symbols: tuple[str, ...]
    start_date: date
    end_date: date
    batch_size: int
    adjusted: bool
    created_at: datetime

    @property
    def batches(self) -> tuple[tuple[str, ...], ...]:
        """Return deterministic symbol batches."""
        return tuple(
            self.symbols[index : index + self.batch_size]
            for index in range(0, len(self.symbols), self.batch_size)
        )


@dataclass(frozen=True)
class BackfillBatchEvent:
    """Append-only evidence for one batch attempt."""

    attempt_id: str
    plan_id: str
    batch_index: int
    symbols: tuple[str, ...]
    status: BackfillEventStatus
    started_at: datetime
    completed_at: datetime
    bar_count: int
    artifact_count: int
    artifact_sha256: tuple[str, ...] = ()
    error_type: str | None = None
    error_message: str | None = None


@dataclass(frozen=True)
class BackfillRunResult:
    """Summary of one invocation, including batches skipped from prior success."""

    plan: BarBackfillPlan
    completed_batch_indices: tuple[int, ...]
    skipped_batch_indices: tuple[int, ...]
    failed_batch_indices: tuple[int, ...]


@dataclass(frozen=True)
class BarCoverageReport:
    """Coverage evidence against explicit expected market sessions."""

    plan_id: str
    expected_sessions: tuple[date, ...]
    completed_batch_indices: tuple[int, ...]
    complete_symbols: tuple[str, ...]
    missing_sessions_by_symbol: dict[str, tuple[date, ...]]
    covers_minimum_five_years: bool
    ready: bool


class BarBackfillStore:
    """Immutable plan plus append-only attempt/coverage evidence."""

    _JOB_NAME = "bars-backfill"

    def __init__(self, layout: LakehouseLayout) -> None:
        self._layout = layout

    def prepare_plan(
        self,
        *,
        symbols: tuple[str, ...],
        start_date: date,
        end_date: date,
        batch_size: int,
        adjusted: bool = False,
        created_at: datetime | None = None,
    ) -> BarBackfillPlan:
        """Create or load the deterministic plan for this exact specification."""
        if end_date < start_date:
            raise ValueError("end_date must be on or after start_date")
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        normalized = tuple(sorted({symbol.strip().upper() for symbol in symbols if symbol.strip()}))
        if not normalized:
            raise ValueError("backfill plan requires at least one symbol")
        identity = {
            "symbols": normalized,
            "start_date": start_date,
            "end_date": end_date,
            "batch_size": batch_size,
            "adjusted": adjusted,
        }
        plan_id = _digest(identity)
        root = self._plan_root(plan_id)
        path = root / "plan.json"
        if path.exists():
            return self.load_plan(plan_id)
        observed_at = created_at or datetime.now(UTC)
        if observed_at.tzinfo is None or observed_at.utcoffset() is None:
            raise ValueError("created_at must be timezone-aware")
        plan = BarBackfillPlan(
            plan_id=plan_id,
            symbols=normalized,
            start_date=start_date,
            end_date=end_date,
            batch_size=batch_size,
            adjusted=adjusted,
            created_at=observed_at.astimezone(UTC),
        )
        root.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as output:
            json.dump(
                asdict(plan),
                output,
                default=_json_default,
                sort_keys=True,
                separators=(",", ":"),
            )
        return plan

    def load_plan(self, plan_id: str) -> BarBackfillPlan:
        """Load one plan by its full content hash."""
        path = self._plan_root(plan_id) / "plan.json"
        encoded = path.read_bytes()
        try:
            raw = json.loads(encoded)
        except (OSError, ValueError) as error:
            raise ValueError(f"invalid backfill plan: {path}") from error
        required = {
            "plan_id",
            "symbols",
            "start_date",
            "end_date",
            "batch_size",
            "adjusted",
            "created_at",
        }
        if (
            not isinstance(raw, dict)
            or set(raw) != required
            or not isinstance(raw["symbols"], list)
            or not isinstance(raw["batch_size"], int)
            or isinstance(raw["batch_size"], bool)
            or raw["batch_size"] < 1
            or not isinstance(raw["adjusted"], bool)
        ):
            raise ValueError("backfill plan schema mismatch")
        plan = BarBackfillPlan(
            plan_id=str(raw["plan_id"]),
            symbols=tuple(str(item) for item in raw["symbols"]),
            start_date=date.fromisoformat(str(raw["start_date"])),
            end_date=date.fromisoformat(str(raw["end_date"])),
            batch_size=int(raw["batch_size"]),
            adjusted=raw["adjusted"],
            created_at=datetime.fromisoformat(str(raw["created_at"])),
        )
        identity = {
            "symbols": plan.symbols,
            "start_date": plan.start_date,
            "end_date": plan.end_date,
            "batch_size": plan.batch_size,
            "adjusted": plan.adjusted,
        }
        canonical = json.dumps(
            asdict(plan),
            default=_json_default,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        if (
            plan.plan_id != plan_id
            or plan.plan_id != _digest(identity)
            or plan.end_date < plan.start_date
            or plan.symbols
            != tuple(sorted({symbol.strip().upper() for symbol in plan.symbols if symbol.strip()}))
            or not plan.symbols
            or plan.created_at.tzinfo is None
            or plan.created_at.utcoffset() is None
            or canonical != encoded
        ):
            raise RuntimeError("backfill plan content does not match its immutable identity")
        return plan

    def write_event(self, event: BackfillBatchEvent) -> Path:
        """Append one batch attempt without replacing prior evidence."""
        plan = self.load_plan(event.plan_id)
        _validate_event(event, plan=plan)
        root = self._plan_root(event.plan_id) / "events" / f"batch={event.batch_index:06d}"
        root.mkdir(parents=True, exist_ok=True)
        path = root / f"{event.attempt_id}-{event.status.value}.json"
        encoded = _event_bytes(event)
        with path.open("x", encoding="utf-8") as output:
            output.write(encoded.decode())
        return path

    def read_events(self, plan_id: str) -> tuple[BackfillBatchEvent, ...]:
        """Load attempts in deterministic completion order."""
        plan = self.load_plan(plan_id)
        root = self._plan_root(plan_id) / "events"
        if not root.exists():
            return ()
        events = []
        for path in sorted(root.rglob("*.json")):
            encoded = path.read_bytes()
            try:
                event = _event_from_dict(json.loads(encoded))
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"invalid backfill event: {path}") from error
            _validate_event(event, plan=plan)
            expected_path = (
                root
                / f"batch={event.batch_index:06d}"
                / f"{event.attempt_id}-{event.status.value}.json"
            )
            if path.resolve() != expected_path.resolve() or encoded != _event_bytes(event):
                raise RuntimeError("backfill event differs from its immutable path or encoding")
            events.append(event)
        return tuple(sorted(events, key=lambda item: (item.completed_at, item.attempt_id)))

    def successful_batch_indices(self, plan_id: str) -> frozenset[int]:
        """Return batches with at least one durable success event."""
        return frozenset(
            event.batch_index
            for event in self.read_events(plan_id)
            if event.status is BackfillEventStatus.SUCCESS
        )

    def write_coverage_report(self, report: BarCoverageReport) -> Path:
        """Persist a content-addressed coverage audit."""
        payload = asdict(report)
        report_id = _digest(payload)
        root = self._plan_root(report.plan_id) / "coverage"
        root.mkdir(parents=True, exist_ok=True)
        path = root / f"coverage-{report_id[:20]}.json"
        try:
            with path.open("x", encoding="utf-8") as output:
                json.dump(
                    payload,
                    output,
                    default=_json_default,
                    sort_keys=True,
                    separators=(",", ":"),
                )
        except FileExistsError:
            expected = json.dumps(
                payload,
                default=_json_default,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
            if path.read_bytes() != expected:
                raise RuntimeError(f"backfill coverage collision at {path}") from None
        return path

    def _plan_root(self, plan_id: str) -> Path:
        if len(plan_id) != 64 or any(char not in "0123456789abcdef" for char in plan_id):
            raise ValueError("plan_id must be a lowercase SHA-256 hex digest")
        return self._layout.root / "manifests" / f"job={self._JOB_NAME}" / f"plan={plan_id}"


class BarBackfillJob:
    """Resume incomplete batches and keep retry output content-stable."""

    def __init__(
        self,
        *,
        provider: BarsProvider,
        silver_writer: SilverWriter,
        store: BarBackfillStore,
        source_capture: DailyBarsSourceCapture | None = None,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
        attempt_id_factory: Callable[[], str] = lambda: uuid4().hex,
    ) -> None:
        self._provider = provider
        self._silver_writer = silver_writer
        self._store = store
        self._source_capture = source_capture
        self._clock = clock
        self._attempt_id_factory = attempt_id_factory

    def run(
        self,
        plan: BarBackfillPlan,
        *,
        continue_on_error: bool = False,
    ) -> BackfillRunResult:
        """Run only batches without prior success evidence."""
        already_complete = self._store.successful_batch_indices(plan.plan_id)
        completed: list[int] = []
        skipped: list[int] = []
        failed: list[int] = []
        for batch_index, symbols in enumerate(plan.batches):
            if batch_index in already_complete:
                skipped.append(batch_index)
                continue
            try:
                event = self._run_batch(
                    plan=plan,
                    batch_index=batch_index,
                    symbols=symbols,
                )
            except Exception:
                failed.append(batch_index)
                if not continue_on_error:
                    raise
            else:
                if event.status is BackfillEventStatus.SUCCESS:
                    completed.append(batch_index)
        return BackfillRunResult(
            plan=plan,
            completed_batch_indices=tuple(completed),
            skipped_batch_indices=tuple(skipped),
            failed_batch_indices=tuple(failed),
        )

    def _run_batch(
        self,
        *,
        plan: BarBackfillPlan,
        batch_index: int,
        symbols: tuple[str, ...],
    ) -> BackfillBatchEvent:
        attempt_id = self._attempt_id_factory()
        started_at = self._aware_now()
        bars: list[EquityBar] = []
        observation_start = (
            len(self._provider_observations()) if self._source_capture is not None else 0
        )
        try:
            for symbol in symbols:
                fetched = self._provider.daily_bars(
                    symbol=symbol,
                    start_date=plan.start_date,
                    end_date=plan.end_date,
                    adjusted=plan.adjusted,
                )
                for bar in fetched:
                    if bar.symbol != symbol:
                        raise ValueError(
                            f"provider returned {bar.symbol} while backfilling {symbol}"
                        )
                    if not plan.start_date <= bar.session_date <= plan.end_date:
                        raise ValueError(f"{symbol} bar {bar.session_date} fell outside the plan")
                    if bar.adjusted is not plan.adjusted:
                        raise ValueError(
                            f"{symbol} bar adjustment mode differed from the backfill plan"
                        )
                bars.extend(fetched)
            artifacts = self._silver_writer.write_daily_bars(
                tuple(bars),
                ingested_at=plan.created_at,
                availability_policy=DAILY_BAR_SESSION_CLOSE_15M,
            )
            if self._source_capture is not None:
                self._source_capture.write(
                    symbols=symbols,
                    start_date=plan.start_date,
                    end_date=plan.end_date,
                    ingested_at=plan.created_at,
                    silver_files=artifacts,
                    provider_observations=self._provider_observations()[observation_start:],
                    availability_policy=DAILY_BAR_SESSION_CLOSE_15M,
                    adjusted=plan.adjusted,
                    backfill_plan_id=plan.plan_id,
                )
            event = self._success_event(
                plan=plan,
                attempt_id=attempt_id,
                batch_index=batch_index,
                symbols=symbols,
                started_at=started_at,
                bars=bars,
                artifacts=artifacts,
            )
            self._store.write_event(event)
            return event
        except Exception as error:
            failure = BackfillBatchEvent(
                attempt_id=attempt_id,
                plan_id=plan.plan_id,
                batch_index=batch_index,
                symbols=symbols,
                status=BackfillEventStatus.FAILURE,
                started_at=started_at,
                completed_at=self._aware_now(),
                bar_count=len(bars),
                artifact_count=0,
                error_type=type(error).__name__,
                error_message=str(error),
            )
            self._store.write_event(failure)
            raise

    def _success_event(
        self,
        *,
        plan: BarBackfillPlan,
        attempt_id: str,
        batch_index: int,
        symbols: tuple[str, ...],
        started_at: datetime,
        bars: list[EquityBar],
        artifacts: tuple[SilverArtifact, ...],
    ) -> BackfillBatchEvent:
        return BackfillBatchEvent(
            attempt_id=attempt_id,
            plan_id=plan.plan_id,
            batch_index=batch_index,
            symbols=symbols,
            status=BackfillEventStatus.SUCCESS,
            started_at=started_at,
            completed_at=self._aware_now(),
            bar_count=len(bars),
            artifact_count=len(artifacts),
            artifact_sha256=tuple(artifact.sha256 for artifact in artifacts),
        )

    def _aware_now(self) -> datetime:
        value = self._clock()
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("backfill clock must return timezone-aware datetimes")
        return value.astimezone(UTC)

    def _provider_observations(self) -> tuple[BronzeArtifact, ...]:
        observations = getattr(self._provider, "feature_observation_artifacts", None)
        if not isinstance(observations, tuple):
            raise ValueError("source-bound backfill provider does not expose observations")
        return cast("tuple[BronzeArtifact, ...]", observations)


class BarCoverageAuditor:
    """Compare persisted bars with explicit expected sessions per planned symbol."""

    def __init__(self, *, layout: LakehouseLayout, store: BarBackfillStore) -> None:
        self._layout = layout
        self._store = store

    def audit(
        self,
        plan: BarBackfillPlan,
        *,
        expected_sessions: tuple[date, ...],
    ) -> BarCoverageReport:
        """Return and persist evidence; never infer sessions from stored bars."""
        expected = tuple(sorted(set(expected_sessions)))
        if not expected:
            raise ValueError("coverage audit requires expected market sessions")
        if expected[0] < plan.start_date or expected[-1] > plan.end_date:
            raise ValueError("expected sessions must fall within the backfill plan")
        success_events = tuple(
            event
            for event in self._store.read_events(plan.plan_id)
            if event.status is BackfillEventStatus.SUCCESS
        )
        completed_batches = tuple(sorted({event.batch_index for event in success_events}))
        source_rows = self._reproduced_plan_rows(
            plan,
            success_events=success_events,
        )
        observed: dict[str, set[date]] = {symbol: set() for symbol in plan.symbols}
        for symbol, session_date, adjusted in source_rows:
            if (
                symbol not in observed
                or not plan.start_date <= session_date <= plan.end_date
                or adjusted is not plan.adjusted
            ):
                raise ValueError("backfill coverage source contains a row outside its plan")
            observed[symbol].add(session_date)
        expected_set = set(expected)
        missing = {
            symbol: tuple(sorted(expected_set - dates))
            for symbol, dates in observed.items()
            if expected_set - dates
        }
        complete_symbols = tuple(symbol for symbol in plan.symbols if symbol not in missing)
        all_batches_complete = len(completed_batches) == len(plan.batches)
        five_years = _covers_five_years(plan.start_date, plan.end_date)
        sufficient_sessions = len(expected) >= MIN_FIVE_YEAR_SESSIONS
        report = BarCoverageReport(
            plan_id=plan.plan_id,
            expected_sessions=expected,
            completed_batch_indices=completed_batches,
            complete_symbols=complete_symbols,
            missing_sessions_by_symbol=missing,
            covers_minimum_five_years=five_years,
            ready=(all_batches_complete and not missing and five_years and sufficient_sessions),
        )
        self._store.write_coverage_report(report)
        return report

    def _reproduced_plan_rows(
        self,
        plan: BarBackfillPlan,
        *,
        success_events: tuple[BackfillBatchEvent, ...],
    ) -> tuple[tuple[str, date, bool], ...]:
        if not success_events:
            return ()
        source_root = self._layout.root / "manifests" / "daily-bars-sources"
        sources = tuple(
            source
            for path in sorted(source_root.glob("source-*.json"))
            if (source := DailyBarsSourceManifest.load(path)).raw["backfill_plan_id"]
            == plan.plan_id
        )
        expected_batch_indices = {event.batch_index for event in success_events}
        batch_indices = {batch: index for index, batch in enumerate(plan.batches)}
        matched_events: set[tuple[int, str]] = set()
        reproduced_rows: list[tuple[str, date, bool]] = []
        with TemporaryDirectory(prefix="qee-backfill-coverage-reproduction-") as temporary:
            output_layout = LakehouseLayout(Path(temporary))
            for source in sources:
                symbols = tuple(source.raw["symbols"])
                batch_index = batch_indices.get(symbols)
                if batch_index not in expected_batch_indices:
                    raise ValueError("backfill source symbols differ from completed plan batches")
                if (
                    source.raw["start_date"] != plan.start_date.isoformat()
                    or source.raw["end_date"] != plan.end_date.isoformat()
                    or source.raw["ingested_at"] != plan.created_at.isoformat()
                    or source.raw["adjusted"] is not plan.adjusted
                    or source.raw["availability_policy"] != DAILY_BAR_SESSION_CLOSE_15M
                ):
                    raise ValueError("backfill source metadata differs from its immutable plan")
                reproduced = DailyBarsSourceCapture.reproduce(
                    source,
                    data_lake_root=self._layout.root,
                    output_layout=output_layout,
                )
                reproduced_hashes = tuple(sorted(artifact.sha256 for artifact in reproduced))
                matching_events = tuple(
                    event
                    for event in success_events
                    if event.batch_index == batch_index
                    and tuple(sorted(event.artifact_sha256)) == reproduced_hashes
                )
                if len(matching_events) != 1:
                    raise ValueError(
                        "backfill source does not uniquely match its success-event artifacts"
                    )
                rows: list[dict[str, Any]] = []
                for artifact in reproduced:
                    if pq.read_schema(artifact.path) != DAILY_BARS_SCHEMA:  # type: ignore[no-untyped-call]
                        raise ValueError(
                            f"backfill coverage daily-bar schema mismatch: {artifact.path}"
                        )
                    rows.extend(
                        pq.read_table(
                            artifact.path,
                            columns=["symbol", "session_date", "adjusted"],
                        ).to_pylist()  # type: ignore[no-untyped-call]
                    )
                event = matching_events[0]
                if len(rows) != event.bar_count:
                    raise ValueError("backfill source row count differs from its success event")
                matched_events.add((event.batch_index, event.attempt_id))
                reproduced_rows.extend(
                    (
                        str(row["symbol"]).strip().upper(),
                        cast("date", row["session_date"]),
                        bool(row["adjusted"]),
                    )
                    for row in rows
                )
        if matched_events != {(event.batch_index, event.attempt_id) for event in success_events}:
            raise ValueError("completed backfill batches lack plan-bound Polygon source lineage")
        return tuple(reproduced_rows)


def _covers_five_years(start_date: date, end_date: date) -> bool:
    try:
        threshold = start_date.replace(year=start_date.year + 5)
    except ValueError:
        threshold = start_date.replace(year=start_date.year + 5, day=28)
    return end_date >= threshold


def _digest(value: Any) -> str:
    encoded = json.dumps(
        value,
        default=_json_default,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _json_default(value: Any) -> str:
    if hasattr(value, "isoformat"):
        return str(value.isoformat())
    if isinstance(value, StrEnum):
        return value.value
    raise TypeError(f"Unsupported backfill value: {type(value).__name__}")


def _event_from_dict(raw: dict[str, Any]) -> BackfillBatchEvent:
    required = {
        "attempt_id",
        "plan_id",
        "batch_index",
        "symbols",
        "status",
        "started_at",
        "completed_at",
        "bar_count",
        "artifact_count",
        "artifact_sha256",
        "error_type",
        "error_message",
    }
    if (
        not isinstance(raw, dict)
        or set(raw) != required
        or not isinstance(raw["batch_index"], int)
        or isinstance(raw["batch_index"], bool)
        or not isinstance(raw["symbols"], list)
        or not isinstance(raw["bar_count"], int)
        or isinstance(raw["bar_count"], bool)
        or not isinstance(raw["artifact_count"], int)
        or isinstance(raw["artifact_count"], bool)
        or not isinstance(raw["artifact_sha256"], list)
    ):
        raise ValueError("backfill event schema mismatch")
    return BackfillBatchEvent(
        attempt_id=str(raw["attempt_id"]),
        plan_id=str(raw["plan_id"]),
        batch_index=int(raw["batch_index"]),
        symbols=tuple(str(item) for item in raw["symbols"]),
        status=BackfillEventStatus(raw["status"]),
        started_at=datetime.fromisoformat(str(raw["started_at"])),
        completed_at=datetime.fromisoformat(str(raw["completed_at"])),
        bar_count=int(raw["bar_count"]),
        artifact_count=int(raw["artifact_count"]),
        artifact_sha256=tuple(str(item) for item in raw["artifact_sha256"]),
        error_type=raw.get("error_type"),
        error_message=raw.get("error_message"),
    )


def _validate_event(event: BackfillBatchEvent, *, plan: BarBackfillPlan) -> None:
    valid_attempt = (
        bool(event.attempt_id)
        and "/" not in event.attempt_id
        and "\\" not in event.attempt_id
        and event.attempt_id not in {".", ".."}
    )
    valid_hashes = all(
        len(item) == 64 and all(character in "0123456789abcdef" for character in item)
        for item in event.artifact_sha256
    )
    valid_batch = (
        0 <= event.batch_index < len(plan.batches)
        and event.symbols == plan.batches[event.batch_index]
    )
    valid_times = (
        event.started_at.tzinfo is not None
        and event.started_at.utcoffset() is not None
        and event.completed_at.tzinfo is not None
        and event.completed_at.utcoffset() is not None
        and event.completed_at >= event.started_at
    )
    valid_success = event.status is not BackfillEventStatus.SUCCESS or (
        event.error_type is None
        and event.error_message is None
        and event.artifact_count == len(event.artifact_sha256)
    )
    valid_failure = event.status is not BackfillEventStatus.FAILURE or (
        isinstance(event.error_type, str)
        and bool(event.error_type)
        and isinstance(event.error_message, str)
        and event.artifact_count == 0
        and not event.artifact_sha256
    )
    if (
        event.plan_id != plan.plan_id
        or not valid_attempt
        or not valid_hashes
        or not valid_batch
        or not valid_times
        or event.bar_count < 0
        or event.artifact_count < 0
        or not valid_success
        or not valid_failure
    ):
        raise ValueError("backfill event differs from its immutable plan contract")


def _event_bytes(event: BackfillBatchEvent) -> bytes:
    return json.dumps(
        asdict(event),
        default=_json_default,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
