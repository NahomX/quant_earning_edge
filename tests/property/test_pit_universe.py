"""Point-in-time invariants for tradable-universe snapshots."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from typing import TYPE_CHECKING

import pyarrow.parquet as pq
import pytest
from hypothesis import given
from hypothesis import strategies as st

from quant_earning_edge.data import LakehouseLayout
from quant_earning_edge.universe import (
    UNIVERSE_SNAPSHOT_SCHEMA,
    CandidateObservation,
    RejectionReason,
    UniverseBuilder,
    UniverseConfig,
    UniverseSnapshotWriter,
)

if TYPE_CHECKING:
    from pathlib import Path

    from quant_earning_edge.universe import UniverseSnapshot

ASOF_DATE = date(2026, 7, 27)
TRADE_DATE = date(2026, 7, 28)


def _candidate(
    *,
    symbol: str = "AAPL",
    asof_date: date = ASOF_DATE,
    close: float = 100,
    adv: float = 2_000_000,
    market_cap: float = 1_000_000_000,
    exchange: str = "XNAS",
    security_type: str = "CS",
    active: bool = True,
    halted: bool = False,
    list_date: date | None = date(1980, 12, 12),
    delisted_date: date | None = None,
) -> CandidateObservation:
    return CandidateObservation(
        symbol=symbol,
        asof_date=asof_date,
        close=close,
        avg_daily_volume=adv,
        market_cap_usd=market_cap,
        primary_exchange=exchange,
        security_type=security_type,
        active=active,
        halted=halted,
        list_date=list_date,
        delisted_date=delisted_date,
    )


def _build(*candidates: CandidateObservation) -> UniverseSnapshot:
    return UniverseBuilder(UniverseConfig()).build(
        trade_date=TRADE_DATE,
        asof_date=ASOF_DATE,
        candidates=tuple(candidates),
        generated_at=datetime(2026, 7, 27, 22, tzinfo=UTC),
    )


def test_threshold_boundaries_are_inclusive() -> None:
    snapshot = _build(
        _candidate(
            close=5,
            adv=1_000_000,
            market_cap=500_000_000,
        )
    )

    assert snapshot.eligible_symbols == ("AAPL",)
    assert snapshot.decisions[0].rejection_reasons == ()


def test_all_rejection_reasons_are_retained() -> None:
    candidate = _candidate(
        close=4.99,
        adv=999_999,
        market_cap=499_999_999,
        exchange="OTCM",
        security_type="ETF",
        active=False,
        halted=True,
        list_date=ASOF_DATE + timedelta(days=1),
        delisted_date=ASOF_DATE,
    )

    decision = _build(candidate).decisions[0]

    assert not decision.eligible
    assert set(decision.rejection_reasons) == set(RejectionReason)


def test_mixed_or_future_observation_date_is_rejected() -> None:
    with pytest.raises(ValueError, match="does not match"):
        _build(_candidate(asof_date=TRADE_DATE))


def test_snapshot_date_itself_must_precede_trade_date() -> None:
    with pytest.raises(ValueError, match="before trade_date"):
        UniverseBuilder(UniverseConfig()).build(
            trade_date=TRADE_DATE,
            asof_date=TRADE_DATE,
            candidates=(),
        )


def test_duplicate_symbols_are_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        _build(_candidate(), _candidate(symbol="aapl"))


@given(
    active=st.booleans(),
    list_offset=st.integers(min_value=-10, max_value=10),
    delist_offset=st.one_of(st.none(), st.integers(min_value=-10, max_value=10)),
)
def test_eligible_ticker_was_active_and_listed_at_prior_close(
    active: bool,
    list_offset: int,
    delist_offset: int | None,
) -> None:
    list_date = ASOF_DATE + timedelta(days=list_offset)
    delisted_date = None if delist_offset is None else ASOF_DATE + timedelta(days=delist_offset)
    decision = _build(
        _candidate(
            active=active,
            list_date=list_date,
            delisted_date=delisted_date,
        )
    ).decisions[0]

    if decision.eligible:
        assert active
        assert list_date <= ASOF_DATE
        assert delisted_date is None or delisted_date > ASOF_DATE


def test_snapshot_persists_all_decisions_and_is_idempotent(tmp_path: Path) -> None:
    snapshot = _build(
        _candidate(symbol="AAPL"),
        _candidate(symbol="PENNY", close=1),
    )
    writer = UniverseSnapshotWriter(LakehouseLayout(tmp_path))

    first = writer.write(snapshot)
    second = writer.write(snapshot)

    assert first == second
    assert first.row_count == 2
    assert len(first.config_sha256) == 64
    assert len(list((tmp_path / "gold").rglob("*.parquet"))) == 1
    table = pq.ParquetFile(first.path).read()  # type: ignore[no-untyped-call]
    assert table.schema == UNIVERSE_SNAPSHOT_SCHEMA
    rows = table.select(["symbol", "eligible"]).to_pylist()
    assert rows == [
        {"symbol": "AAPL", "eligible": True},
        {"symbol": "PENNY", "eligible": False},
    ]
