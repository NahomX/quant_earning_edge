"""Resumable backfill and explicit-session coverage tests."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from typing import TYPE_CHECKING

import pytest

from quant_earning_edge.data import (
    BarBackfillJob,
    BarBackfillStore,
    BarCoverageAuditor,
    LakehouseLayout,
    SilverWriter,
)
from quant_earning_edge.data.clients import EquityBar

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
    ) -> tuple[EquityBar, ...]:
        self.calls.append(symbol)
        if symbol in self.fail_symbols:
            raise RuntimeError(f"intentional failure for {symbol}")
        return tuple(
            _bar(symbol, session) for session in self.sessions if start_date <= session <= end_date
        )


class Clock:
    def __init__(self) -> None:
        self.value = datetime(2026, 7, 27, tzinfo=UTC)

    def __call__(self) -> datetime:
        current = self.value
        self.value += timedelta(seconds=1)
        return current


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

    assert first == second
    assert first.symbols == ("AAPL", "MSFT")
    assert len(first.plan_id) == 64
    assert first.created_at == created_at


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
    provider = FakeBarsProvider(sessions=tuple(sessions))
    BarBackfillJob(
        provider=provider,
        silver_writer=SilverWriter(layout),
        store=store,
        clock=Clock(),
        attempt_id_factory=lambda: "coverage-attempt",
    ).run(plan)

    report = BarCoverageAuditor(layout=layout, store=store).audit(
        plan,
        expected_sessions=tuple(sessions),
    )

    assert report.ready
    assert report.complete_symbols == ("AAPL",)
    assert report.missing_sessions_by_symbol == {}
    assert report.covers_minimum_five_years
    coverage_root = (
        tmp_path / "manifests" / "job=bars-backfill" / f"plan={plan.plan_id}" / "coverage"
    )
    assert len(list(coverage_root.glob("coverage-*.json"))) == 1


def test_coverage_reports_missing_sessions_and_rejects_inferred_empty_calendar(
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
    with pytest.raises(ValueError, match="expected market sessions"):
        BarCoverageAuditor(layout=layout, store=store).audit(
            plan,
            expected_sessions=(),
        )

    report = BarCoverageAuditor(layout=layout, store=store).audit(
        plan,
        expected_sessions=(date(2021, 1, 4),),
    )
    assert not report.ready
    assert report.missing_sessions_by_symbol == {"AAPL": (date(2021, 1, 4),)}
