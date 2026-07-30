"""Resumable backfill and explicit-session coverage tests."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime, timedelta
from typing import TYPE_CHECKING

import pyarrow.parquet as pq
import pytest

from quant_earning_edge.data import (
    BarBackfillJob,
    BarBackfillStore,
    BarCoverageAuditor,
    BronzeWriter,
    CalendarSourceCapture,
    DailyBarsSourceCapture,
    LakehouseLayout,
    SessionFileStore,
    SilverWriter,
    SplitHistorySourceCapture,
)
from quant_earning_edge.data.clients import AlpacaCalendarClient, EquityBar, PolygonClient
from quant_earning_edge.features import DailyBarsFeatureLoader

if TYPE_CHECKING:
    from pathlib import Path


def _bar(symbol: str, session: date) -> EquityBar:
    return EquityBar(
        symbol=symbol,
        timestamp=datetime.combine(session, datetime.min.time(), tzinfo=UTC) + timedelta(hours=5),
        open=100,
        high=102,
        low=99,
        close=101,
        volume=2_000_000,
        adjusted=True,
    )


class FakeBarsProvider:
    def __init__(
        self,
        *,
        sessions: tuple[date, ...],
        fail_symbols: frozenset[str] = frozenset(),
    ) -> None:
        self.sessions = sessions
        self.fail_symbols = fail_symbols
        self.calls: list[str] = []

    def daily_bars(
        self,
        *,
        symbol: str,
        start_date: date,
        end_date: date,
        adjusted: bool = True,
    ) -> tuple[EquityBar, ...]:
        self.calls.append(symbol)
        if symbol in self.fail_symbols:
            raise RuntimeError(f"intentional failure for {symbol}")
        return tuple(
            _bar(symbol, session).model_copy(update={"adjusted": adjusted})
            for session in self.sessions
            if start_date <= session <= end_date
        )


class Clock:
    def __init__(self) -> None:
        self.value = datetime(2026, 7, 27, tzinfo=UTC)

    def __call__(self) -> datetime:
        current = self.value
        self.value += timedelta(seconds=1)
        return current


class CapturingBarsProvider:
    def __init__(self, layout: LakehouseLayout, sessions: tuple[date, ...]) -> None:
        self._bronze = BronzeWriter(layout)
        self._sessions = sessions
        self._observations: list[object] = []

    @property
    def feature_observation_artifacts(self) -> tuple[object, ...]:
        return tuple(self._observations)

    def daily_bars(
        self,
        *,
        symbol: str,
        start_date: date,
        end_date: date,
        adjusted: bool = True,
    ) -> tuple[EquityBar, ...]:
        raw = {
            "ticker": symbol,
            "adjusted": adjusted,
            "status": "OK",
            "results": [
                {
                    "o": 100,
                    "h": 102,
                    "l": 99,
                    "c": 101,
                    "v": 2_000_000,
                    "n": 50_000,
                    "t": int(
                        (
                            datetime.combine(session, datetime.min.time(), tzinfo=UTC)
                            + timedelta(hours=16)
                        ).timestamp()
                        * 1000
                    ),
                }
                for session in self._sessions
                if start_date <= session <= end_date
            ],
        }
        self._observations.append(
            self._bronze.write_json(
                raw,
                source="polygon",
                dataset="daily-aggregate-bars",
                event_date=start_date,
            )
        )
        return PolygonClient.daily_bars_from_payload(
            raw,
            symbol=symbol,
            expected_adjusted=adjusted,
        )


def _calendar_source(
    layout: LakehouseLayout,
    *,
    sessions: tuple[date, ...],
    start_date: date,
    end_date: date,
) -> Path:
    raw = [
        {
            "date": session.isoformat(),
            "open": "09:30",
            "close": "16:00",
        }
        for session in sessions
    ]
    observation = BronzeWriter(layout).write_json(
        raw,
        source="alpaca",
        dataset="market-calendar",
        event_date=start_date,
        received_at=datetime(2026, 7, 27, tzinfo=UTC),
    )
    session_file = SessionFileStore(layout).write(
        AlpacaCalendarClient.sessions_from_payload(
            raw,
            start_date=start_date,
            end_date=end_date,
        )
    )
    return (
        CalendarSourceCapture(layout)
        .write(
            start_date=start_date,
            end_date=end_date,
            session_file=session_file,
            provider_observations=(observation,),
        )
        .path
    )


def test_plan_identity_is_normalized_and_stable_across_resume(tmp_path: Path) -> None:
    store = BarBackfillStore(LakehouseLayout(tmp_path))
    created_at = datetime(2026, 7, 27, tzinfo=UTC)

    first = store.prepare_plan(
        symbols=("msft", "AAPL", "aapl"),
        start_date=date(2021, 1, 1),
        end_date=date(2026, 1, 1),
        batch_size=2,
        created_at=created_at,
    )
    second = store.prepare_plan(
        symbols=("AAPL", "MSFT"),
        start_date=date(2021, 1, 1),
        end_date=date(2026, 1, 1),
        batch_size=2,
        created_at=created_at + timedelta(days=1),
    )
    adjusted = store.prepare_plan(
        symbols=("AAPL", "MSFT"),
        start_date=date(2021, 1, 1),
        end_date=date(2026, 1, 1),
        batch_size=2,
        adjusted=True,
        created_at=created_at,
    )

    assert first == second
    assert adjusted.plan_id != first.plan_id
    assert adjusted.adjusted
    assert first.symbols == ("AAPL", "MSFT")
    assert len(first.plan_id) == 64
    assert not first.adjusted
    assert first.created_at == created_at


def test_plan_load_rejects_rewritten_content_under_original_identity(tmp_path: Path) -> None:
    store = BarBackfillStore(LakehouseLayout(tmp_path))
    plan = store.prepare_plan(
        symbols=("AAPL",),
        start_date=date(2021, 1, 1),
        end_date=date(2026, 1, 1),
        batch_size=1,
        created_at=datetime(2026, 7, 27, tzinfo=UTC),
    )
    path = next(tmp_path.rglob("plan.json"))
    raw = json.loads(path.read_bytes())
    raw["end_date"] = "2027-01-01"
    path.write_bytes(json.dumps(raw, sort_keys=True, separators=(",", ":")).encode())

    with pytest.raises(RuntimeError, match="immutable identity"):
        store.load_plan(plan.plan_id)


def test_backfill_resumes_batches_and_keeps_silver_content_stable(tmp_path: Path) -> None:
    layout = LakehouseLayout(tmp_path)
    store = BarBackfillStore(layout)
    plan = store.prepare_plan(
        symbols=("A", "B", "C", "D"),
        start_date=date(2026, 7, 20),
        end_date=date(2026, 7, 21),
        batch_size=2,
        created_at=datetime(2026, 7, 27, tzinfo=UTC),
    )
    failing_provider = FakeBarsProvider(
        sessions=(date(2026, 7, 20), date(2026, 7, 21)),
        fail_symbols=frozenset({"C"}),
    )
    first = BarBackfillJob(
        provider=failing_provider,
        silver_writer=SilverWriter(layout),
        store=store,
        clock=Clock(),
        attempt_id_factory=iter(("attempt-1", "attempt-2")).__next__,
    ).run(plan, continue_on_error=True)

    assert first.completed_batch_indices == (0,)
    assert first.failed_batch_indices == (1,)
    initial_paths = tuple(sorted((tmp_path / "silver").rglob("*.parquet")))
    assert len(initial_paths) == 2

    healthy_provider = FakeBarsProvider(sessions=(date(2026, 7, 20), date(2026, 7, 21)))
    second = BarBackfillJob(
        provider=healthy_provider,
        silver_writer=SilverWriter(layout),
        store=store,
        clock=Clock(),
        attempt_id_factory=lambda: "attempt-3",
    ).run(plan)

    assert second.skipped_batch_indices == (0,)
    assert second.completed_batch_indices == (1,)
    assert healthy_provider.calls == ["C", "D"]
    final_paths = tuple(sorted((tmp_path / "silver").rglob("*.parquet")))
    assert len(final_paths) == 4
    assert set(initial_paths).issubset(final_paths)
    assert store.successful_batch_indices(plan.plan_id) == frozenset({0, 1})


def test_backfill_resume_rejects_event_rewritten_under_original_path(tmp_path: Path) -> None:
    layout = LakehouseLayout(tmp_path)
    store = BarBackfillStore(layout)
    session = date(2026, 7, 20)
    plan = store.prepare_plan(
        symbols=("AAPL",),
        start_date=session,
        end_date=session,
        batch_size=1,
        created_at=datetime(2026, 7, 27, tzinfo=UTC),
    )
    BarBackfillJob(
        provider=FakeBarsProvider(sessions=(session,)),
        silver_writer=SilverWriter(layout),
        store=store,
        clock=Clock(),
        attempt_id_factory=lambda: "immutable-attempt",
    ).run(plan)
    event_path = next((tmp_path / "manifests").rglob("*-success.json"))
    raw = json.loads(event_path.read_bytes())
    raw["symbols"] = ["MSFT"]
    event_path.write_bytes(json.dumps(raw, sort_keys=True, separators=(",", ":")).encode())

    with pytest.raises(ValueError, match="immutable plan contract"):
        store.successful_batch_indices(plan.plan_id)


def test_backfill_emits_reproducible_polygon_source_manifest(tmp_path: Path) -> None:
    layout = LakehouseLayout(tmp_path)
    store = BarBackfillStore(layout)
    sessions = (date(2026, 7, 20), date(2026, 7, 21))
    plan = store.prepare_plan(
        symbols=("AAPL", "MSFT"),
        start_date=sessions[0],
        end_date=sessions[-1],
        batch_size=2,
        created_at=datetime(2026, 7, 29, tzinfo=UTC),
    )
    BarBackfillJob(
        provider=CapturingBarsProvider(layout, sessions),
        silver_writer=SilverWriter(layout),
        store=store,
        source_capture=DailyBarsSourceCapture(layout),
        clock=Clock(),
        attempt_id_factory=lambda: "captured-attempt",
    ).run(plan)
    silver_files = tuple(sorted((tmp_path / "silver").rglob("*.parquet")))
    manifests = DailyBarsSourceCapture.find_for_files(
        silver_files,
        data_lake_root=tmp_path,
    )

    assert len(manifests) == 1
    assert manifests[0].raw["availability_policy"] == "session_close_plus_15m"
    with pytest.raises(ValueError, match="split-history source"):
        DailyBarsFeatureLoader().load(
            silver_files,
            symbols=("AAPL", "MSFT"),
            asof_date=sessions[-1],
            observed_at=datetime(2026, 7, 22, 13, tzinfo=UTC),
        )
    assert all(
        value == datetime(2026, 7, 29, tzinfo=UTC)
        for path in silver_files
        for value in pq.read_table(path).column("ingested_at").to_pylist()  # type: ignore[no-untyped-call]
    )
    assert manifests[0].raw["adjusted"] is False
    reproduced = DailyBarsSourceCapture.reproduce(
        manifests[0],
        data_lake_root=tmp_path,
        output_layout=LakehouseLayout(tmp_path / "reproduced"),
    )
    assert tuple(item.path.read_bytes() for item in reproduced) == tuple(
        path.read_bytes() for path in silver_files
    )


def test_unadjusted_backfill_is_source_bound_and_rejected_by_feature_loader(
    tmp_path: Path,
) -> None:
    layout = LakehouseLayout(tmp_path)
    store = BarBackfillStore(layout)
    session = date(2026, 7, 20)
    plan = store.prepare_plan(
        symbols=("AAPL",),
        start_date=session,
        end_date=session,
        batch_size=1,
        adjusted=False,
        created_at=datetime(2026, 7, 29, tzinfo=UTC),
    )
    BarBackfillJob(
        provider=CapturingBarsProvider(layout, (session,)),
        silver_writer=SilverWriter(layout),
        store=store,
        source_capture=DailyBarsSourceCapture(layout),
        clock=Clock(),
        attempt_id_factory=lambda: "raw-attempt",
    ).run(plan)
    silver_files = tuple(sorted((tmp_path / "silver").rglob("*.parquet")))
    manifest = DailyBarsSourceCapture.find_for_files(
        silver_files,
        data_lake_root=tmp_path,
    )[0]

    assert manifest.raw["adjusted"] is False
    assert pq.read_table(silver_files[0]).column("adjusted").to_pylist() == [False]  # type: ignore[no-untyped-call]
    with pytest.raises(ValueError, match="split-history source"):
        DailyBarsFeatureLoader().load(
            silver_files,
            symbols=("AAPL",),
            asof_date=session,
            observed_at=datetime(2026, 7, 21, 13, tzinfo=UTC),
        )
    split_observation = BronzeWriter(layout).write_json(
        {"status": "OK", "results": []},
        source="polygon",
        dataset="stock-splits",
        event_date=session,
    )
    split_source = SplitHistorySourceCapture(layout).write(
        plan_id=plan.plan_id,
        start_date=session,
        end_date=session,
        ingested_at=plan.created_at,
        split_files=(),
        provider_observations=(split_observation,),
    )
    contexts = DailyBarsFeatureLoader().load(
        silver_files,
        symbols=("AAPL",),
        asof_date=session,
        observed_at=datetime(2026, 7, 21, 13, tzinfo=UTC),
        split_source_manifest=split_source.path,
        data_lake_root=tmp_path,
    )
    assert contexts[0].bars[0].close == 101
    reproduced = DailyBarsSourceCapture.reproduce(
        manifest,
        data_lake_root=tmp_path,
        output_layout=LakehouseLayout(tmp_path / "reproduced"),
    )
    assert reproduced[0].path.read_bytes() == silver_files[0].read_bytes()


def test_coverage_requires_full_batches_five_years_and_1200_sessions(
    tmp_path: Path,
) -> None:
    layout = LakehouseLayout(tmp_path)
    store = BarBackfillStore(layout)
    plan = store.prepare_plan(
        symbols=("AAPL",),
        start_date=date(2021, 1, 1),
        end_date=date(2026, 1, 1),
        batch_size=1,
        created_at=datetime(2026, 1, 2, tzinfo=UTC),
    )
    sessions: list[date] = []
    cursor = plan.start_date
    while cursor <= plan.end_date and len(sessions) < 1_200:
        if cursor.weekday() < 5:
            sessions.append(cursor)
        cursor += timedelta(days=1)
    provider = CapturingBarsProvider(layout, tuple(sessions))
    BarBackfillJob(
        provider=provider,
        silver_writer=SilverWriter(layout),
        store=store,
        source_capture=DailyBarsSourceCapture(layout),
        clock=Clock(),
        attempt_id_factory=lambda: "coverage-attempt",
    ).run(plan)
    calendar_source = _calendar_source(
        layout,
        sessions=tuple(sessions),
        start_date=plan.start_date,
        end_date=plan.end_date,
    )

    report = BarCoverageAuditor(layout=layout, store=store).audit(
        plan,
        calendar_source_manifest=calendar_source,
    )

    assert report.ready
    assert len(report.calendar_source_sha256) == 64
    assert len(report.session_file_sha256) == 64
    assert report.complete_symbols == ("AAPL",)
    assert report.missing_sessions_by_symbol == {}
    assert report.covers_minimum_five_years
    coverage_root = (
        tmp_path / "manifests" / "job=bars-backfill" / f"plan={plan.plan_id}" / "coverage"
    )
    assert len(list(coverage_root.glob("coverage-*.json"))) == 1
    coverage_path = next(coverage_root.glob("coverage-*.json"))
    coverage_path.write_bytes(b"{}")
    with pytest.raises(RuntimeError, match="coverage collision"):
        BarCoverageAuditor(layout=layout, store=store).audit(
            plan,
            calendar_source_manifest=calendar_source,
        )


def test_coverage_rejects_completed_batch_without_plan_bound_provider_lineage(
    tmp_path: Path,
) -> None:
    layout = LakehouseLayout(tmp_path)
    store = BarBackfillStore(layout)
    session = date(2026, 7, 20)
    plan = store.prepare_plan(
        symbols=("AAPL",),
        start_date=session,
        end_date=session,
        batch_size=1,
        created_at=datetime(2026, 7, 27, tzinfo=UTC),
    )
    BarBackfillJob(
        provider=FakeBarsProvider(sessions=(session,)),
        silver_writer=SilverWriter(layout),
        store=store,
        clock=Clock(),
        attempt_id_factory=lambda: "unbound-attempt",
    ).run(plan)
    calendar_source = _calendar_source(
        layout,
        sessions=(session,),
        start_date=plan.start_date,
        end_date=plan.end_date,
    )

    with pytest.raises(ValueError, match="lack plan-bound Polygon source lineage"):
        BarCoverageAuditor(layout=layout, store=store).audit(
            plan,
            calendar_source_manifest=calendar_source,
        )


def test_coverage_rejects_success_event_detached_from_source_artifacts(
    tmp_path: Path,
) -> None:
    layout = LakehouseLayout(tmp_path)
    store = BarBackfillStore(layout)
    session = date(2026, 7, 20)
    plan = store.prepare_plan(
        symbols=("AAPL",),
        start_date=session,
        end_date=session,
        batch_size=1,
        created_at=datetime(2026, 7, 27, tzinfo=UTC),
    )
    BarBackfillJob(
        provider=CapturingBarsProvider(layout, (session,)),
        silver_writer=SilverWriter(layout),
        store=store,
        source_capture=DailyBarsSourceCapture(layout),
        clock=Clock(),
        attempt_id_factory=lambda: "detached-attempt",
    ).run(plan)
    event_path = next((tmp_path / "manifests").rglob("*-success.json"))
    raw = json.loads(event_path.read_bytes())
    raw["artifact_sha256"] = ["0" * 64]
    event_path.write_bytes(json.dumps(raw, sort_keys=True, separators=(",", ":")).encode())
    calendar_source = _calendar_source(
        layout,
        sessions=(session,),
        start_date=plan.start_date,
        end_date=plan.end_date,
    )

    with pytest.raises(ValueError, match="uniquely match its success-event artifacts"):
        BarCoverageAuditor(layout=layout, store=store).audit(
            plan,
            calendar_source_manifest=calendar_source,
        )


def test_coverage_reports_missing_sessions_and_rejects_mismatched_calendar_interval(
    tmp_path: Path,
) -> None:
    layout = LakehouseLayout(tmp_path)
    store = BarBackfillStore(layout)
    plan = store.prepare_plan(
        symbols=("AAPL",),
        start_date=date(2021, 1, 1),
        end_date=date(2026, 1, 1),
        batch_size=1,
        created_at=datetime(2026, 1, 2, tzinfo=UTC),
    )
    mismatched_calendar = _calendar_source(
        layout,
        sessions=(date(2021, 1, 4),),
        start_date=date(2021, 1, 4),
        end_date=plan.end_date,
    )
    with pytest.raises(ValueError, match="interval differs"):
        BarCoverageAuditor(layout=layout, store=store).audit(
            plan,
            calendar_source_manifest=mismatched_calendar,
        )

    calendar_source = _calendar_source(
        layout,
        sessions=(date(2021, 1, 4),),
        start_date=plan.start_date,
        end_date=plan.end_date,
    )
    report = BarCoverageAuditor(layout=layout, store=store).audit(
        plan,
        calendar_source_manifest=calendar_source,
    )
    assert not report.ready
    assert report.missing_sessions_by_symbol == {"AAPL": (date(2021, 1, 4),)}


def test_coverage_rejects_calendar_manifest_copied_outside_the_data_lake(
    tmp_path: Path,
) -> None:
    layout = LakehouseLayout(tmp_path / "lake")
    store = BarBackfillStore(layout)
    session = date(2026, 7, 20)
    plan = store.prepare_plan(
        symbols=("AAPL",),
        start_date=session,
        end_date=session,
        batch_size=1,
        created_at=datetime(2026, 7, 27, tzinfo=UTC),
    )
    calendar_source = _calendar_source(
        layout,
        sessions=(session,),
        start_date=session,
        end_date=session,
    )
    copied_source = tmp_path / calendar_source.name
    copied_source.write_bytes(calendar_source.read_bytes())

    with pytest.raises(ValueError, match="not canonical in-lake evidence"):
        BarCoverageAuditor(layout=layout, store=store).audit(
            plan,
            calendar_source_manifest=copied_source,
        )
