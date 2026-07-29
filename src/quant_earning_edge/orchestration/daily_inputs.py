"""Provider-backed preparation of immutable candidates and live feature vectors."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from enum import StrEnum
from typing import TYPE_CHECKING

import pyarrow.parquet as pq

from quant_earning_edge.data import (
    BarsIngestor,
    CalendarSourceCapture,
    CorporateActionsIngestor,
    EarningsIngestor,
    SessionFileStore,
    SilverWriter,
)
from quant_earning_edge.features import (
    FEATURE_VALUE_SCHEMA,
    DailyBarsFeatureLoader,
    EarningsFeatureLoader,
    FeatureEngine,
    FeatureSourceCapture,
    FeatureStore,
    PremarketFeatureLoader,
)
from quant_earning_edge.universe import (
    DailyUniverseJob,
    EventCandidateJob,
    EventSourceCapture,
    RunTrigger,
    UniverseBuilder,
    UniverseManifestStore,
    UniverseSnapshotWriter,
    UniverseSourceCapture,
)
from quant_earning_edge.universe.config import load_halt_snapshot, load_universe_job_config
from quant_earning_edge.universe.events import (
    EVENT_CANDIDATE_SCHEMA,
    EventCandidateManifest,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from quant_earning_edge.data.clients import FinnhubClient, PolygonClient
    from quant_earning_edge.data.layout import LakehouseLayout


class DailyInputStatus(StrEnum):
    """Progress of one causal daily-input preparation attempt."""

    WAITING_FOR_PRIOR_CLOSE = "waiting_for_prior_close"
    CANDIDATES_READY = "candidates_ready"
    FEATURES_READY = "features_ready"


@dataclass(frozen=True)
class DailyInputPreparation:
    """Immutable outputs available after one preparation attempt."""

    status: DailyInputStatus
    trade_date: date
    candidate_file: Path | None
    feature_file: Path | None
    candidate_count: int
    detail: str


class DailyInputPreparer:
    """Build T-1 candidates, then causal pre-open features for those candidates."""

    def __init__(
        self,
        *,
        layout: LakehouseLayout,
        polygon: PolygonClient,
        finnhub: FinnhubClient,
        clock: Callable[[], datetime],
    ) -> None:
        self._layout = layout
        self._polygon = polygon
        self._finnhub = finnhub
        self._clock = clock

    def run(
        self,
        *,
        trade_date: date,
        session_file: Path,
        halt_snapshot_file: Path,
        universe_config_file: Path,
        feature_names: tuple[str, ...],
        feature_group: str,
        bar_lookback_calendar_days: int = 450,
        candidate_delay_after_close: timedelta = timedelta(hours=5),
        feature_lead_before_open: timedelta = timedelta(minutes=20),
        latest_safe_time_before_open: timedelta = timedelta(minutes=10),
    ) -> DailyInputPreparation:
        """Advance preparation without reading observations beyond the current clock."""
        if bar_lookback_calendar_days < 365:
            raise ValueError("live feature preparation requires at least 365 calendar days")
        if not feature_names or not feature_group.strip():
            raise ValueError("live feature preparation requires features and a feature group")
        now = self._aware_now()
        calendar = SessionFileStore.load(session_file)
        selected = next(
            (item for item in calendar.sessions if item.session_date == trade_date),
            None,
        )
        earlier = tuple(item for item in calendar.sessions if item.session_date < trade_date)
        if selected is None or not earlier:
            raise ValueError("daily input preparation requires trade and prior sessions")
        prior = earlier[-1]
        candidate = self._candidate_file(trade_date)
        if candidate is None:
            candidate_not_before = prior.close_at + candidate_delay_after_close
            if now < candidate_not_before:
                return DailyInputPreparation(
                    status=DailyInputStatus.WAITING_FOR_PRIOR_CLOSE,
                    trade_date=trade_date,
                    candidate_file=None,
                    feature_file=None,
                    candidate_count=0,
                    detail=f"candidate preparation opens at {candidate_not_before.isoformat()}",
                )
            candidate = self._prepare_candidates(
                trade_date=trade_date,
                decision_at=now,
                prior_date=prior.session_date,
                session_file=session_file,
                halt_snapshot_file=halt_snapshot_file,
                universe_config_file=universe_config_file,
            )
        symbols = self._candidate_symbols(candidate)
        if not symbols:
            return DailyInputPreparation(
                status=DailyInputStatus.FEATURES_READY,
                trade_date=trade_date,
                candidate_file=candidate,
                feature_file=None,
                candidate_count=0,
                detail="zero candidates require no feature artifact",
            )
        existing = self._feature_file(
            feature_group=feature_group,
            asof_date=prior.session_date,
            symbols=symbols,
            feature_names=feature_names,
            observed_at=now,
            target_open=selected.open_at,
        )
        if existing is not None:
            return DailyInputPreparation(
                status=DailyInputStatus.FEATURES_READY,
                trade_date=trade_date,
                candidate_file=candidate,
                feature_file=existing,
                candidate_count=len(symbols),
                detail="complete causal feature artifact already exists",
            )
        feature_not_before = selected.open_at - feature_lead_before_open
        if now < feature_not_before:
            return DailyInputPreparation(
                status=DailyInputStatus.CANDIDATES_READY,
                trade_date=trade_date,
                candidate_file=candidate,
                feature_file=None,
                candidate_count=len(symbols),
                detail=f"pre-open feature preparation opens at {feature_not_before.isoformat()}",
            )
        latest_safe = selected.open_at - latest_safe_time_before_open
        if now >= latest_safe:
            raise ValueError("daily feature preparation missed the safe pre-open window")
        feature = self._prepare_features(
            trade_date=trade_date,
            asof_date=prior.session_date,
            observed_at=now,
            market_open=selected.open_at,
            candidate_file=candidate,
            symbols=symbols,
            feature_names=feature_names,
            feature_group=feature_group,
            bar_lookback_calendar_days=bar_lookback_calendar_days,
        )
        return DailyInputPreparation(
            status=DailyInputStatus.FEATURES_READY,
            trade_date=trade_date,
            candidate_file=candidate,
            feature_file=feature,
            candidate_count=len(symbols),
            detail="provider-backed causal feature artifact prepared",
        )

    def _prepare_candidates(
        self,
        *,
        trade_date: date,
        decision_at: datetime,
        prior_date: date,
        session_file: Path,
        halt_snapshot_file: Path,
        universe_config_file: Path,
    ) -> Path:
        halt_snapshot = load_halt_snapshot(halt_snapshot_file)
        if halt_snapshot.asof_date != prior_date:
            raise ValueError("daily halt snapshot does not match the prior session")
        if halt_snapshot.captured_at > decision_at:
            raise ValueError("daily halt snapshot was captured after the decision")
        config = load_universe_job_config(universe_config_file)
        earnings_observation_start = len(self._finnhub.earnings_observation_artifacts)
        earnings = EarningsIngestor(
            client=self._finnhub,
            silver_writer=SilverWriter(self._layout),
        ).ingest(
            start_date=prior_date,
            end_date=trade_date,
            ingested_at=decision_at,
        )
        action_observation_start = len(self._polygon.corporate_action_observation_artifacts)
        actions = CorporateActionsIngestor(
            client=self._polygon,
            silver_writer=SilverWriter(self._layout),
        ).ingest(
            start_date=trade_date,
            end_date=trade_date,
            ingested_at=decision_at,
        )
        observation_start = len(self._polygon.universe_observation_artifacts)
        universe = DailyUniverseJob(
            market_data=self._polygon,
            builder=UniverseBuilder(config.eligibility.to_domain()),
            snapshot_writer=UniverseSnapshotWriter(self._layout),
            manifest_store=UniverseManifestStore(self._layout),
            adv_sessions=config.adv_sessions,
            clock=lambda: decision_at,
        ).run(
            trade_date=trade_date,
            asof_date=prior_date,
            lookback_start=prior_date - timedelta(days=90),
            halt_snapshot=halt_snapshot,
            trigger=RunTrigger.SCHEDULED,
        )
        universe_observations = self._polygon.universe_observation_artifacts[observation_start:]
        universe_source = UniverseSourceCapture(self._layout).write(
            trade_date=trade_date,
            asof_date=prior_date,
            lookback_start=prior_date - timedelta(days=90),
            decision_at=decision_at,
            adv_sessions=config.adv_sessions,
            snapshot=universe.snapshot,
            universe_config=universe_config_file,
            halt_snapshot=halt_snapshot_file,
            provider_observations=universe_observations,
        )
        split_artifacts = tuple(
            item for item in actions.silver_artifacts if "dataset=stock-splits" in str(item.path)
        )
        dividend_artifacts = tuple(
            item for item in actions.silver_artifacts if "dataset=cash-dividends" in str(item.path)
        )
        event_source = EventSourceCapture(self._layout).write(
            start_date=prior_date,
            end_date=trade_date,
            ingested_at=decision_at,
            earnings_files=earnings.silver_artifacts,
            split_files=split_artifacts,
            dividend_files=dividend_artifacts,
            earnings_observations=self._finnhub.earnings_observation_artifacts[
                earnings_observation_start:
            ],
            corporate_action_observations=(
                self._polygon.corporate_action_observation_artifacts[action_observation_start:]
            ),
        )
        calendar_source = CalendarSourceCapture.find_for_session(
            session_file,
            data_lake_root=self._layout.root,
        )
        artifact = EventCandidateJob(self._layout).run(
            trade_date=trade_date,
            decision_at=decision_at,
            universe_snapshot=universe.snapshot.path,
            session_file=session_file,
            earnings_files=tuple(item.path for item in earnings.silver_artifacts),
            split_files=tuple(item.path for item in split_artifacts),
            dividend_files=tuple(item.path for item in dividend_artifacts),
            universe_source_manifest=universe_source.path,
            event_source_manifest=event_source.path,
            calendar_source_manifest=calendar_source.path,
        )
        return artifact.path.resolve()

    def _prepare_features(
        self,
        *,
        trade_date: date,
        asof_date: date,
        observed_at: datetime,
        market_open: datetime,
        candidate_file: Path,
        symbols: tuple[str, ...],
        feature_names: tuple[str, ...],
        feature_group: str,
        bar_lookback_calendar_days: int,
    ) -> Path:
        writer = SilverWriter(self._layout)
        bars_files: list[Path] = []
        minute_files: list[Path] = []
        observation_start = len(self._polygon.feature_observation_artifacts)
        for symbol in symbols:
            bars = BarsIngestor(client=self._polygon, silver_writer=writer).ingest(
                symbol=symbol,
                start_date=asof_date - timedelta(days=bar_lookback_calendar_days),
                end_date=asof_date,
                ingested_at=observed_at,
            )
            bars_files.extend(item.path for item in bars.silver_artifacts)
            minute = self._polygon.minute_bars(
                symbol=symbol,
                start_at=market_open - timedelta(hours=5, minutes=30),
                end_at=observed_at - timedelta(minutes=1),
            )
            minute_files.append(
                writer.write_minute_bars(
                    minute,
                    event_date=trade_date,
                    ingested_at=observed_at,
                ).path
            )
        contexts = DailyBarsFeatureLoader().load(
            bars_files,
            symbols=symbols,
            asof_date=asof_date,
            observed_at=observed_at,
        )
        contexts = PremarketFeatureLoader().enrich(
            contexts,
            minute_files=minute_files,
            target_date=trade_date,
            observed_at=observed_at,
        )
        earnings_files = tuple(
            sorted(
                (
                    self._layout.root
                    / "silver"
                    / "asset_class=us-equity"
                    / "dataset=earnings-events"
                ).rglob("*.parquet")
            )
        )
        contexts = EarningsFeatureLoader().enrich(
            contexts,
            candidate_files=(candidate_file,),
            earnings_files=earnings_files,
            observed_at=observed_at,
            target_date=trade_date,
        )
        values = FeatureEngine().compute(contexts, feature_names=feature_names)
        artifact = FeatureStore(self._layout).write(
            feature_group=feature_group,
            values=values,
            computed_at=observed_at,
        )
        FeatureSourceCapture(self._layout).write(
            trade_date=trade_date,
            asof_date=asof_date,
            observed_at=observed_at,
            feature_group=feature_group,
            symbols=symbols,
            feature_names=feature_names,
            feature_file=artifact,
            candidate_files=(candidate_file,),
            daily_bar_files=bars_files,
            minute_bar_files=minute_files,
            earnings_files=earnings_files,
            provider_observations=self._polygon.feature_observation_artifacts[observation_start:],
        )
        return artifact.path.resolve()

    def _candidate_file(self, trade_date: date) -> Path | None:
        root = (
            self._layout.root
            / "gold"
            / "event-candidates"
            / f"for_trade_date={trade_date.isoformat()}"
        )
        paths = []
        for candidate in sorted(root.glob("candidates-*.parquet")):
            identity = candidate.stem.removeprefix("candidates-")
            manifest_path = candidate.with_name(f"manifest-{identity}.json")
            if not manifest_path.is_file():
                continue
            manifest = EventCandidateManifest.load(manifest_path)
            if (
                manifest.raw["schema_version"] == 5
                and manifest.universe_lineage_entries
                and manifest.event_lineage_entries
                and manifest.calendar_lineage_entries
            ):
                manifest.source_paths(data_lake_root=self._layout.root)
                paths.append(candidate)
        if not paths:
            return None
        if len(paths) > 1:
            raise ValueError("daily candidate artifact is ambiguous")
        self._candidate_symbols(paths[0])
        return paths[0].resolve()

    @staticmethod
    def _candidate_symbols(path: Path) -> tuple[str, ...]:
        if pq.read_schema(path) != EVENT_CANDIDATE_SCHEMA:  # type: ignore[no-untyped-call]
            raise ValueError("daily candidate artifact schema mismatch")
        rows = pq.ParquetFile(path).read(columns=["symbol"]).to_pylist()  # type: ignore[no-untyped-call]
        symbols = tuple(str(row["symbol"]).strip().upper() for row in rows)
        if symbols != tuple(sorted(set(symbols))):
            raise ValueError("daily candidate symbols must be unique and sorted")
        return symbols

    def _feature_file(
        self,
        *,
        feature_group: str,
        asof_date: date,
        symbols: tuple[str, ...],
        feature_names: tuple[str, ...],
        observed_at: datetime,
        target_open: datetime,
    ) -> Path | None:
        month = asof_date.isoformat()[:7]
        root = self._layout.root / "gold" / f"feature_group={feature_group}" / f"month={month}"
        expected = {(symbol, feature_name) for symbol in symbols for feature_name in feature_names}
        matches = []
        for path in sorted(root.glob("part-*.parquet")):
            if pq.read_schema(path) != FEATURE_VALUE_SCHEMA:  # type: ignore[no-untyped-call]
                continue
            rows = [
                row
                for row in pq.read_table(path).to_pylist()  # type: ignore[no-untyped-call]
                if row["asof_date"] == asof_date
            ]
            keys = {(str(row["symbol"]).strip().upper(), str(row["feature_name"])) for row in rows}
            computed = {row["computed_at"] for row in rows}
            if (
                keys == expected
                and len(rows) == len(expected)
                and len(computed) == 1
                and (computed_at := next(iter(computed))) <= observed_at
                and computed_at < target_open
            ):
                matches.append((computed_at, path.resolve()))
        if not matches:
            return None
        latest = max(item[0] for item in matches)
        paths = tuple(path for computed_at, path in matches if computed_at == latest)
        if len(paths) != 1:
            raise ValueError("latest daily feature artifact is ambiguous")
        return paths[0]

    def _aware_now(self) -> datetime:
        value = self._clock()
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("daily input preparation clock must be timezone-aware")
        return value
