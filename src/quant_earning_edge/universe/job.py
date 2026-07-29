"""Production assembly and operational evidence for daily universe snapshots."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Protocol
from uuid import uuid4

from quant_earning_edge.universe.models import CandidateObservation, sector_from_sic_code

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from quant_earning_edge.data.clients.polygon import (
        EquityBar,
        TickerDetails,
        TickerReference,
    )
    from quant_earning_edge.data.layout import LakehouseLayout
    from quant_earning_edge.universe.builder import UniverseBuilder
    from quant_earning_edge.universe.snapshot import (
        UniverseSnapshotArtifact,
        UniverseSnapshotWriter,
    )


class UniverseMarketData(Protocol):
    """Provider operations required by the production universe job."""

    def list_tickers(
        self,
        *,
        asof_date: date,
        active: bool = True,
    ) -> tuple[TickerReference, ...]: ...

    def ticker_details(self, *, symbol: str, asof_date: date) -> TickerDetails: ...

    def daily_bars(
        self,
        *,
        symbol: str,
        start_date: date,
        end_date: date,
    ) -> tuple[EquityBar, ...]: ...


@dataclass(frozen=True)
class HaltSnapshot:
    """Symbols known halted as of the same prior-close date."""

    asof_date: date
    symbols: frozenset[str]
    captured_at: datetime

    def __post_init__(self) -> None:
        if self.captured_at.tzinfo is None or self.captured_at.utcoffset() is None:
            raise ValueError("halt snapshot captured_at must be timezone-aware")
        object.__setattr__(
            self,
            "symbols",
            frozenset(symbol.strip().upper() for symbol in self.symbols),
        )


class RunTrigger(StrEnum):
    """How a production job invocation was started."""

    MANUAL = "manual"
    SCHEDULED = "scheduled"


class RunStatus(StrEnum):
    """Terminal status recorded for an invocation."""

    SUCCESS = "success"
    FAILURE = "failure"


@dataclass(frozen=True)
class UniverseRunManifest:
    """Durable evidence for one completed or failed invocation."""

    run_id: str
    trade_date: date
    asof_date: date
    trigger: RunTrigger
    status: RunStatus
    started_at: datetime
    completed_at: datetime
    reference_count: int
    candidate_count: int
    eligible_count: int
    snapshot_path: str | None = None
    snapshot_sha256: str | None = None
    error_type: str | None = None
    error_message: str | None = None


@dataclass(frozen=True)
class UniverseJobResult:
    """Successful job output and its persisted operational evidence."""

    snapshot: UniverseSnapshotArtifact
    manifest: UniverseRunManifest
    manifest_path: Path


class UniverseManifestStore:
    """Append-only JSON manifest storage."""

    _JOB_NAME = "daily-universe"

    def __init__(self, layout: LakehouseLayout) -> None:
        self._layout = layout

    def write(self, manifest: UniverseRunManifest) -> Path:
        """Persist a manifest with exclusive-create semantics."""
        partition = self._layout.run_manifests(
            job_name=self._JOB_NAME,
            trade_date=manifest.trade_date,
        )
        partition.mkdir(parents=True, exist_ok=True)
        path = partition / f"{manifest.run_id}.json"
        encoded = json.dumps(
            asdict(manifest),
            default=_json_default,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        with path.open("xb") as output:
            output.write(encoded)
        return path

    def read_all(self) -> tuple[UniverseRunManifest, ...]:
        """Load all stored manifests in deterministic completion order."""
        root = self._layout.root / "manifests" / f"job={self._JOB_NAME}"
        if not root.exists():
            return ()
        manifests = [
            _manifest_from_dict(json.loads(path.read_text(encoding="utf-8")))
            for path in root.rglob("*.json")
        ]
        return tuple(sorted(manifests, key=lambda item: item.completed_at))


class DailyUniverseJob:
    """Build a complete prior-close universe or fail the invocation loudly."""

    def __init__(
        self,
        *,
        market_data: UniverseMarketData,
        builder: UniverseBuilder,
        snapshot_writer: UniverseSnapshotWriter,
        manifest_store: UniverseManifestStore,
        adv_sessions: int = 20,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
        run_id_factory: Callable[[], str] = lambda: uuid4().hex,
    ) -> None:
        if adv_sessions < 1:
            raise ValueError("adv_sessions must be at least 1")
        self._market_data = market_data
        self._builder = builder
        self._snapshot_writer = snapshot_writer
        self._manifest_store = manifest_store
        self._adv_sessions = adv_sessions
        self._clock = clock
        self._run_id_factory = run_id_factory

    def run(
        self,
        *,
        trade_date: date,
        asof_date: date,
        lookback_start: date,
        halt_snapshot: HaltSnapshot,
        trigger: RunTrigger,
    ) -> UniverseJobResult:
        """Execute one atomic logical run and always record terminal evidence."""
        run_id = self._run_id_factory()
        started_at = self._aware_now()
        reference_count = 0
        candidate_count = 0
        try:
            if halt_snapshot.asof_date != asof_date:
                raise ValueError("halt snapshot date must match asof_date")
            if halt_snapshot.captured_at > started_at:
                raise ValueError("halt snapshot cannot be captured after job start")
            references = self._market_data.list_tickers(
                asof_date=asof_date,
                active=True,
            )
            reference_count = len(references)
            candidates: list[CandidateObservation] = []
            for reference in references:
                candidates.append(
                    self._candidate_from_reference(
                        reference=reference,
                        asof_date=asof_date,
                        lookback_start=lookback_start,
                        halted=reference.symbol in halt_snapshot.symbols,
                    )
                )
                candidate_count = len(candidates)
            snapshot = self._builder.build(
                trade_date=trade_date,
                asof_date=asof_date,
                candidates=tuple(candidates),
                generated_at=self._aware_now(),
            )
            artifact = self._snapshot_writer.write(snapshot)
            manifest = UniverseRunManifest(
                run_id=run_id,
                trade_date=trade_date,
                asof_date=asof_date,
                trigger=trigger,
                status=RunStatus.SUCCESS,
                started_at=started_at,
                completed_at=self._aware_now(),
                reference_count=reference_count,
                candidate_count=candidate_count,
                eligible_count=len(snapshot.eligible_symbols),
                snapshot_path=str(artifact.path),
                snapshot_sha256=artifact.sha256,
            )
            manifest_path = self._manifest_store.write(manifest)
            return UniverseJobResult(
                snapshot=artifact,
                manifest=manifest,
                manifest_path=manifest_path,
            )
        except Exception as error:
            manifest = UniverseRunManifest(
                run_id=run_id,
                trade_date=trade_date,
                asof_date=asof_date,
                trigger=trigger,
                status=RunStatus.FAILURE,
                started_at=started_at,
                completed_at=self._aware_now(),
                reference_count=reference_count,
                candidate_count=candidate_count,
                eligible_count=0,
                error_type=type(error).__name__,
                error_message=str(error),
            )
            self._manifest_store.write(manifest)
            raise

    def _candidate_from_reference(
        self,
        *,
        reference: TickerReference,
        asof_date: date,
        lookback_start: date,
        halted: bool,
    ) -> CandidateObservation:
        if reference.asof_date != asof_date:
            raise ValueError(f"{reference.symbol} reference date did not match asof_date")
        details = self._market_data.ticker_details(
            symbol=reference.symbol,
            asof_date=asof_date,
        )
        if details.symbol != reference.symbol or details.asof_date != asof_date:
            raise ValueError(f"{reference.symbol} details identity/date did not match")
        if details.active != reference.active:
            raise ValueError(f"{reference.symbol} active state disagreed across endpoints")
        bars = self._market_data.daily_bars(
            symbol=reference.symbol,
            start_date=lookback_start,
            end_date=asof_date,
        )
        eligible_bars = sorted(
            (bar for bar in bars if bar.session_date <= asof_date),
            key=lambda bar: bar.session_date,
        )
        if len(eligible_bars) < self._adv_sessions:
            raise ValueError(
                f"{reference.symbol} has {len(eligible_bars)} bars; "
                f"{self._adv_sessions} required for ADV"
            )
        window = eligible_bars[-self._adv_sessions :]
        if window[-1].session_date != asof_date:
            raise ValueError(f"{reference.symbol} has no prior-close bar on {asof_date}")
        average_volume = sum(bar.volume for bar in window) / self._adv_sessions
        return CandidateObservation(
            symbol=reference.symbol,
            asof_date=asof_date,
            close=window[-1].close,
            avg_daily_volume=average_volume,
            market_cap_usd=details.market_cap,
            primary_exchange=details.primary_exchange,
            security_type=details.security_type,
            active=details.active,
            halted=halted,
            sector=sector_from_sic_code(details.sic_code),
            list_date=details.list_date,
            delisted_date=details.delisted_date,
        )

    def _aware_now(self) -> datetime:
        value = self._clock()
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("job clock must return timezone-aware datetimes")
        return value.astimezone(UTC)


@dataclass(frozen=True)
class ReadinessEvidence:
    """Evidence for the unattended-run exit gate."""

    expected_trade_dates: tuple[date, ...]
    successful_trade_dates: tuple[date, ...]
    ready: bool


def evaluate_unattended_readiness(
    manifests: tuple[UniverseRunManifest, ...],
    *,
    expected_trade_dates: tuple[date, ...],
) -> ReadinessEvidence:
    """Require a scheduled success for every externally supplied market date."""
    latest_by_date: dict[date, UniverseRunManifest] = {}
    for manifest in sorted(manifests, key=lambda item: item.completed_at):
        latest_by_date[manifest.trade_date] = manifest
    successful = tuple(
        trade_date
        for trade_date in expected_trade_dates
        if (
            (current := latest_by_date.get(trade_date)) is not None
            and current.trigger is RunTrigger.SCHEDULED
            and current.status is RunStatus.SUCCESS
            and current.snapshot_sha256 is not None
        )
    )
    return ReadinessEvidence(
        expected_trade_dates=expected_trade_dates,
        successful_trade_dates=successful,
        ready=bool(expected_trade_dates) and len(successful) == len(expected_trade_dates),
    )


def _json_default(value: Any) -> str:
    if hasattr(value, "isoformat"):
        return str(value.isoformat())
    if isinstance(value, StrEnum):
        return value.value
    raise TypeError(f"Unsupported manifest value: {type(value).__name__}")


def _manifest_from_dict(raw: dict[str, Any]) -> UniverseRunManifest:
    return UniverseRunManifest(
        run_id=str(raw["run_id"]),
        trade_date=date.fromisoformat(str(raw["trade_date"])),
        asof_date=date.fromisoformat(str(raw["asof_date"])),
        trigger=RunTrigger(raw["trigger"]),
        status=RunStatus(raw["status"]),
        started_at=datetime.fromisoformat(str(raw["started_at"])),
        completed_at=datetime.fromisoformat(str(raw["completed_at"])),
        reference_count=int(raw["reference_count"]),
        candidate_count=int(raw["candidate_count"]),
        eligible_count=int(raw["eligible_count"]),
        snapshot_path=raw.get("snapshot_path"),
        snapshot_sha256=raw.get("snapshot_sha256"),
        error_type=raw.get("error_type"),
        error_message=raw.get("error_message"),
    )
