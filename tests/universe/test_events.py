"""Point-in-time earnings candidate join tests."""

from __future__ import annotations

from datetime import UTC, date, datetime
from typing import TYPE_CHECKING, Any
from zoneinfo import ZoneInfo

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from quant_earning_edge.data import LakehouseLayout, SessionFileStore, SilverWriter
from quant_earning_edge.data.clients import (
    CashDividend,
    DividendDistributionType,
    EarningsEvent,
    MarketSession,
    SplitAdjustmentType,
    StockSplit,
)
from quant_earning_edge.universe import CandidateExclusion, EventCandidateJob
from quant_earning_edge.universe.snapshot import UNIVERSE_SNAPSHOT_SCHEMA

if TYPE_CHECKING:
    from pathlib import Path

TRADE_DATE = date(2026, 7, 28)
ASOF_DATE = date(2026, 7, 27)
DECISION_AT = datetime(2026, 7, 28, 1, 30, tzinfo=UTC)


def _sessions(tmp_path: Path) -> Path:
    eastern = ZoneInfo("America/New_York")
    artifact = SessionFileStore(LakehouseLayout(tmp_path)).write(
        (
            MarketSession(
                session_date=ASOF_DATE,
                open_at=datetime(2026, 7, 27, 9, 30, tzinfo=eastern),
                close_at=datetime(2026, 7, 27, 16, 0, tzinfo=eastern),
            ),
            MarketSession(
                session_date=TRADE_DATE,
                open_at=datetime(2026, 7, 28, 9, 30, tzinfo=eastern),
                close_at=datetime(2026, 7, 28, 16, 0, tzinfo=eastern),
            ),
        )
    )
    return artifact.path


def _universe(tmp_path: Path, *, generated_at: datetime = DECISION_AT) -> Path:
    records: list[dict[str, Any]] = [
        {
            "trade_date": TRADE_DATE,
            "asof_date": ASOF_DATE,
            "generated_at": generated_at,
            "config_sha256": "a" * 64,
            "symbol": symbol,
            "eligible": eligible,
            "rejection_reasons": [],
            "close": 100.0,
            "avg_daily_volume": 2_000_000.0,
            "market_cap_usd": 1_000_000_000.0,
            "primary_exchange": "XNAS",
            "security_type": "CS",
            "active": True,
            "halted": False,
            "list_date": date(1980, 1, 1),
            "delisted_date": None,
        }
        for symbol, eligible in (("AAPL", True), ("MSFT", False), ("GOOG", True))
    ]
    path = tmp_path / "snapshot.parquet"
    pq.write_table(pa.Table.from_pylist(records, schema=UNIVERSE_SNAPSHOT_SCHEMA), path)  # type: ignore[no-untyped-call]
    return path


def _event(symbol: str, event_date: date, timing: str) -> EarningsEvent:
    return EarningsEvent.model_validate(
        {
            "date": event_date,
            "symbol": symbol,
            "hour": timing,
            "year": 2026,
            "quarter": 3,
            "epsEstimate": 1.25,
            "revenueEstimate": 100.0,
        }
    )


def _corporate_actions(
    tmp_path: Path,
    *,
    ingested_at: datetime = DECISION_AT,
) -> tuple[tuple[Path, ...], tuple[Path, ...]]:
    writer = SilverWriter(LakehouseLayout(tmp_path))
    splits = writer.write_splits(
        (
            StockSplit(
                event_id="split-aapl",
                symbol="AAPL",
                execution_date=TRADE_DATE,
                adjustment_type=SplitAdjustmentType.FORWARD_SPLIT,
                split_from=1,
                split_to=2,
            ),
        ),
        ingested_at=ingested_at,
    )
    dividends = writer.write_dividends(
        (
            CashDividend(
                event_id="dividend-goog",
                symbol="GOOG",
                ex_dividend_date=TRADE_DATE,
                distribution_type=DividendDistributionType.RECURRING,
                cash_amount=0.25,
                currency="USD",
                frequency=4,
            ),
        ),
        ingested_at=ingested_at,
    )
    return (
        tuple(item.path for item in splits),
        tuple(item.path for item in dividends),
    )


def test_join_selects_prior_amc_and_trade_date_bmo_only(tmp_path: Path) -> None:
    earnings = SilverWriter(LakehouseLayout(tmp_path)).write_earnings(
        (
            _event("AAPL", ASOF_DATE, "amc"),
            _event("GOOG", TRADE_DATE, "bmo"),
            _event("MSFT", TRADE_DATE, "bmo"),
            _event("TSLA", TRADE_DATE, "dmh"),
            _event("NVDA", TRADE_DATE, "amc"),
        ),
        ingested_at=DECISION_AT,
    )
    split_files, dividend_files = _corporate_actions(tmp_path)

    artifact = EventCandidateJob(LakehouseLayout(tmp_path)).run(
        trade_date=TRADE_DATE,
        decision_at=DECISION_AT,
        universe_snapshot=_universe(tmp_path),
        session_file=_sessions(tmp_path),
        earnings_files=tuple(item.path for item in earnings),
        split_files=split_files,
        dividend_files=dividend_files,
    )

    rows = pq.read_table(artifact.path).to_pylist()  # type: ignore[no-untyped-call]
    assert [(row["symbol"], row["event_date"], row["timing"]) for row in rows] == [
        ("AAPL", ASOF_DATE, "amc"),
        ("GOOG", TRADE_DATE, "bmo"),
    ]
    assert artifact.excluded_counts == {
        CandidateExclusion.NOT_IN_ELIGIBLE_UNIVERSE: 1,
        CandidateExclusion.UNSUPPORTED_DURING_MARKET_HOURS: 1,
    }
    assert rows[0]["split_event_ids"] == ["split-aapl"]
    assert rows[0]["dividend_event_ids"] == []
    assert rows[1]["split_event_ids"] == []
    assert rows[1]["dividend_event_ids"] == ["dividend-goog"]
    assert "eps_actual" not in rows[0]


def test_join_ignores_earnings_observations_ingested_after_decision(tmp_path: Path) -> None:
    future = SilverWriter(LakehouseLayout(tmp_path)).write_earnings(
        (_event("AAPL", ASOF_DATE, "amc"),),
        ingested_at=datetime(2026, 7, 28, 2, tzinfo=UTC),
    )
    split_files, dividend_files = _corporate_actions(tmp_path)

    artifact = EventCandidateJob(LakehouseLayout(tmp_path)).run(
        trade_date=TRADE_DATE,
        decision_at=DECISION_AT,
        universe_snapshot=_universe(tmp_path),
        session_file=_sessions(tmp_path),
        earnings_files=tuple(item.path for item in future),
        split_files=split_files,
        dividend_files=dividend_files,
    )

    assert artifact.row_count == 0
    assert artifact.manifest_path.exists()


def test_join_rejects_universe_created_after_decision(tmp_path: Path) -> None:
    earnings = SilverWriter(LakehouseLayout(tmp_path)).write_earnings(
        (_event("AAPL", ASOF_DATE, "amc"),),
        ingested_at=DECISION_AT,
    )
    split_files, dividend_files = _corporate_actions(tmp_path)

    with pytest.raises(ValueError, match="generated after decision_at"):
        EventCandidateJob(LakehouseLayout(tmp_path)).run(
            trade_date=TRADE_DATE,
            decision_at=DECISION_AT,
            universe_snapshot=_universe(
                tmp_path,
                generated_at=datetime(2026, 7, 28, 2, tzinfo=UTC),
            ),
            session_file=_sessions(tmp_path),
            earnings_files=tuple(item.path for item in earnings),
            split_files=split_files,
            dividend_files=dividend_files,
        )


def test_join_rejects_decision_during_trade_session(tmp_path: Path) -> None:
    earnings = SilverWriter(LakehouseLayout(tmp_path)).write_earnings(
        (_event("AAPL", ASOF_DATE, "amc"),),
        ingested_at=DECISION_AT,
    )
    split_files, dividend_files = _corporate_actions(tmp_path)

    with pytest.raises(ValueError, match="before the trade session open"):
        EventCandidateJob(LakehouseLayout(tmp_path)).run(
            trade_date=TRADE_DATE,
            decision_at=datetime(2026, 7, 28, 14, tzinfo=UTC),
            universe_snapshot=_universe(tmp_path),
            session_file=_sessions(tmp_path),
            earnings_files=tuple(item.path for item in earnings),
            split_files=split_files,
            dividend_files=dividend_files,
        )


def test_join_does_not_annotate_actions_observed_after_decision(tmp_path: Path) -> None:
    earnings = SilverWriter(LakehouseLayout(tmp_path)).write_earnings(
        (_event("AAPL", ASOF_DATE, "amc"),),
        ingested_at=DECISION_AT,
    )
    split_files, dividend_files = _corporate_actions(
        tmp_path,
        ingested_at=datetime(2026, 7, 28, 2, tzinfo=UTC),
    )

    artifact = EventCandidateJob(LakehouseLayout(tmp_path)).run(
        trade_date=TRADE_DATE,
        decision_at=DECISION_AT,
        universe_snapshot=_universe(tmp_path),
        session_file=_sessions(tmp_path),
        earnings_files=tuple(item.path for item in earnings),
        split_files=split_files,
        dividend_files=dividend_files,
    )

    row = pq.read_table(artifact.path).to_pylist()[0]  # type: ignore[no-untyped-call]
    assert row["split_event_ids"] == []
    assert row["dividend_event_ids"] == []
