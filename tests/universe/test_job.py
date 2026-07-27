"""Production universe job, manifests, and readiness evidence tests."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
from typing import TYPE_CHECKING

import pyarrow.parquet as pq
import pytest

from quant_earning_edge.data import LakehouseLayout
from quant_earning_edge.data.clients import EquityBar, TickerDetails, TickerReference
from quant_earning_edge.universe import (
    DailyUniverseJob,
    HaltSnapshot,
    RunStatus,
    RunTrigger,
    UniverseBuilder,
    UniverseConfig,
    UniverseManifestStore,
    UniverseRunManifest,
    UniverseSnapshotWriter,
    evaluate_unattended_readiness,
)

if TYPE_CHECKING:
    from pathlib import Path

ASOF_DATE = date(2026, 7, 27)
TRADE_DATE = date(2026, 7, 28)


class FakeMarketData:
    """Deterministic provider implementing the production protocol."""

    def __init__(self, *, omit_prior_close: bool = False) -> None:
        self._references = (
            TickerReference(
                symbol="AAPL",
                asof_date=ASOF_DATE,
                name="Apple",
                active=True,
                locale="us",
                market="stocks",
                primary_exchange="XNAS",
                security_type="CS",
            ),
            TickerReference(
                symbol="MSFT",
                asof_date=ASOF_DATE,
                name="Microsoft",
                active=True,
                locale="us",
                market="stocks",
                primary_exchange="XNAS",
                security_type="CS",
            ),
        )
        self._omit_prior_close = omit_prior_close

    def list_tickers(
        self,
        *,
        asof_date: date,
        active: bool = True,
    ) -> tuple[TickerReference, ...]:
        assert asof_date == ASOF_DATE
        assert active
        return self._references

    def ticker_details(self, *, symbol: str, asof_date: date) -> TickerDetails:
        return TickerDetails(
            symbol=symbol,
            asof_date=asof_date,
            name=symbol,
            active=True,
            locale="us",
            market="stocks",
            primary_exchange="XNAS",
            security_type="CS",
            market_cap=1_000_000_000,
            list_date=date(1980, 1, 1),
        )

    def daily_bars(
        self,
        *,
        symbol: str,
        start_date: date,
        end_date: date,
    ) -> tuple[EquityBar, ...]:
        del start_date
        sessions: list[date] = []
        cursor = end_date
        while len(sessions) < 20:
            if cursor.weekday() < 5:
                sessions.append(cursor)
            cursor -= timedelta(days=1)
        sessions.reverse()
        if self._omit_prior_close and symbol == "AAPL":
            sessions = sessions[:-1]
        return tuple(
            EquityBar(
                symbol=symbol,
                timestamp=datetime.combine(
                    session,
                    datetime.min.time(),
                    tzinfo=UTC,
                )
                + timedelta(hours=4),
                open=100,
                high=102,
                low=99,
                close=101,
                volume=2_000_000,
                adjusted=True,
            )
            for session in sessions
        )


class AdvancingClock:
    def __init__(self) -> None:
        self._value = datetime(2026, 7, 27, 22, tzinfo=UTC)

    def __call__(self) -> datetime:
        current = self._value
        self._value += timedelta(seconds=1)
        return current


def _job(
    tmp_path: Path,
    *,
    market_data: FakeMarketData,
) -> tuple[DailyUniverseJob, UniverseManifestStore]:
    layout = LakehouseLayout(tmp_path)
    store = UniverseManifestStore(layout)
    return (
        DailyUniverseJob(
            market_data=market_data,
            builder=UniverseBuilder(UniverseConfig()),
            snapshot_writer=UniverseSnapshotWriter(layout),
            manifest_store=store,
            clock=AdvancingClock(),
            run_id_factory=lambda: "run-001",
        ),
        store,
    )


def test_job_builds_prior_close_adv_snapshot_and_success_manifest(tmp_path: Path) -> None:
    job, store = _job(tmp_path, market_data=FakeMarketData())

    result = job.run(
        trade_date=TRADE_DATE,
        asof_date=ASOF_DATE,
        lookback_start=date(2026, 6, 1),
        halt_snapshot=HaltSnapshot(
            asof_date=ASOF_DATE,
            symbols=frozenset({"msft"}),
            captured_at=datetime(2026, 7, 27, 21, tzinfo=UTC),
        ),
        trigger=RunTrigger.SCHEDULED,
    )

    assert result.manifest.status is RunStatus.SUCCESS
    assert result.manifest.reference_count == 2
    assert result.manifest.candidate_count == 2
    assert result.manifest.eligible_count == 1
    assert result.snapshot.row_count == 2
    table = pq.ParquetFile(result.snapshot.path).read()  # type: ignore[no-untyped-call]
    rows = table.select(["symbol", "avg_daily_volume", "eligible", "halted"]).to_pylist()
    assert rows == [
        {
            "symbol": "AAPL",
            "avg_daily_volume": 2_000_000.0,
            "eligible": True,
            "halted": False,
        },
        {
            "symbol": "MSFT",
            "avg_daily_volume": 2_000_000.0,
            "eligible": False,
            "halted": True,
        },
    ]
    assert store.read_all() == (result.manifest,)


def test_missing_prior_close_aborts_and_persists_failure_manifest(tmp_path: Path) -> None:
    job, store = _job(
        tmp_path,
        market_data=FakeMarketData(omit_prior_close=True),
    )

    with pytest.raises(ValueError, match="bars; 20 required"):
        job.run(
            trade_date=TRADE_DATE,
            asof_date=ASOF_DATE,
            lookback_start=date(2026, 6, 1),
            halt_snapshot=HaltSnapshot(
                asof_date=ASOF_DATE,
                symbols=frozenset(),
                captured_at=datetime(2026, 7, 27, 21, tzinfo=UTC),
            ),
            trigger=RunTrigger.SCHEDULED,
        )

    manifests = store.read_all()
    assert len(manifests) == 1
    assert manifests[0].status is RunStatus.FAILURE
    assert manifests[0].reference_count == 2
    assert manifests[0].candidate_count == 0
    assert manifests[0].error_type == "ValueError"
    assert not list((tmp_path / "gold").rglob("*.parquet"))


def _manifest(
    trade_date: date,
    *,
    trigger: RunTrigger = RunTrigger.SCHEDULED,
    status: RunStatus = RunStatus.SUCCESS,
    offset: int = 0,
) -> UniverseRunManifest:
    completed = datetime(2026, 7, 1, tzinfo=UTC) + timedelta(days=offset)
    return UniverseRunManifest(
        run_id=f"run-{trade_date}-{offset}",
        trade_date=trade_date,
        asof_date=trade_date - timedelta(days=1),
        trigger=trigger,
        status=status,
        started_at=completed - timedelta(minutes=1),
        completed_at=completed,
        reference_count=100,
        candidate_count=100,
        eligible_count=50,
        snapshot_sha256="a" * 64 if status is RunStatus.SUCCESS else None,
    )


def test_readiness_requires_scheduled_success_for_every_expected_market_date() -> None:
    dates = tuple(date(2026, 7, day) for day in (20, 21, 22, 23, 24))
    manifests = tuple(_manifest(day, offset=index) for index, day in enumerate(dates))

    evidence = evaluate_unattended_readiness(
        manifests,
        expected_trade_dates=dates,
    )

    assert evidence.ready
    assert evidence.successful_trade_dates == dates

    manual_latest = replace(
        manifests[-1],
        run_id="manual-latest",
        trigger=RunTrigger.MANUAL,
        completed_at=manifests[-1].completed_at + timedelta(minutes=1),
    )
    not_ready = evaluate_unattended_readiness(
        (*manifests, manual_latest),
        expected_trade_dates=dates,
    )
    assert not not_ready.ready
    assert not_ready.successful_trade_dates == dates[:-1]
