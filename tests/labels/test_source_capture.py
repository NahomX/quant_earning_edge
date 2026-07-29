"""Provider-source reconstruction tests for forward-label artifacts."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from typing import TYPE_CHECKING

import pytest

from quant_earning_edge.data import (
    DAILY_BAR_SESSION_CLOSE_15M,
    BronzeWriter,
    CalendarSourceCapture,
    DailyBarsSourceCapture,
    LakehouseLayout,
    SessionFileStore,
    SilverWriter,
    SplitHistorySourceCapture,
)
from quant_earning_edge.data.clients import AlpacaCalendarClient, PolygonClient
from quant_earning_edge.labels import (
    ForwardLabelMaker,
    ForwardLabelSourceCapture,
    ForwardLabelSourceManifest,
    LabelBarsLoader,
    LabelStore,
)

if TYPE_CHECKING:
    from pathlib import Path

ASOF_DATE = date(2026, 7, 1)
SESSIONS = tuple(ASOF_DATE + timedelta(days=index) for index in range(6))
OBSERVED_AT = datetime(2026, 7, 8, 22, tzinfo=UTC)


def _aggregate(session_date: date, index: int) -> dict[str, float | int]:
    timestamp = datetime.combine(session_date, datetime.min.time(), tzinfo=UTC)
    timestamp += timedelta(hours=4)
    return {
        "o": 100.0 + index,
        "h": 103.0 + index,
        "l": 99.0 + index,
        "c": 101.0 + index,
        "v": 1_000_000,
        "vw": 100.5 + index,
        "n": 50_000,
        "t": int(timestamp.timestamp() * 1000),
    }


def _source_fixture(tmp_path: Path) -> tuple[ForwardLabelSourceManifest, Path]:
    layout = LakehouseLayout(tmp_path / "lake")
    bronze = BronzeWriter(layout)
    polygon_raw = {
        "ticker": "AAPL",
        "adjusted": True,
        "status": "OK",
        "results": [_aggregate(session_date, index) for index, session_date in enumerate(SESSIONS)],
    }
    polygon_observation = bronze.write_json(
        polygon_raw,
        source="polygon",
        dataset="daily-aggregate-bars",
        event_date=ASOF_DATE,
        received_at=OBSERVED_AT,
    )
    bars = PolygonClient.daily_bars_from_payload(polygon_raw, symbol="AAPL")
    daily_artifacts = SilverWriter(layout).write_daily_bars(
        bars,
        ingested_at=OBSERVED_AT,
    )
    DailyBarsSourceCapture(layout).write(
        symbols=("AAPL",),
        start_date=SESSIONS[0],
        end_date=SESSIONS[-1],
        ingested_at=OBSERVED_AT,
        silver_files=daily_artifacts,
        provider_observations=(polygon_observation,),
    )
    calendar_raw = [
        {"date": item.isoformat(), "open": "09:30", "close": "16:00"} for item in SESSIONS
    ]
    calendar_observation = bronze.write_json(
        calendar_raw,
        source="alpaca",
        dataset="market-calendar",
        event_date=SESSIONS[0],
        received_at=datetime(2026, 6, 30, 12, tzinfo=UTC),
    )
    market_sessions = AlpacaCalendarClient.sessions_from_payload(
        calendar_raw,
        start_date=SESSIONS[0],
        end_date=SESSIONS[-1],
    )
    session_file = SessionFileStore(layout).write(market_sessions)
    CalendarSourceCapture(layout).write(
        start_date=SESSIONS[0],
        end_date=SESSIONS[-1],
        session_file=session_file,
        provider_observations=(calendar_observation,),
    )
    loaded_bars = LabelBarsLoader().load(
        tuple(item.path for item in daily_artifacts),
        symbols=("AAPL",),
        start_date=ASOF_DATE,
        end_date=SESSIONS[-1],
        observed_at=OBSERVED_AT,
    )
    labels = ForwardLabelMaker().compute(
        keys=(("AAPL", ASOF_DATE),),
        sessions=SESSIONS,
        bars=loaded_bars,
    )
    label_artifact = LabelStore(layout).write(labels, computed_at=OBSERVED_AT)
    manifest = ForwardLabelSourceCapture(layout).write(
        asof_date=ASOF_DATE,
        observed_at=OBSERVED_AT,
        symbols=("AAPL",),
        label_file=label_artifact,
        daily_bar_files=tuple(item.path for item in daily_artifacts),
        session_file=session_file.path,
    )
    return manifest, polygon_observation.path


def test_forward_labels_reproduce_from_polygon_and_alpaca_sources(tmp_path: Path) -> None:
    manifest, _ = _source_fixture(tmp_path)
    lake = tmp_path / "lake"

    assert (
        ForwardLabelSourceCapture.find_for_label(
            manifest.source_paths(data_lake_root=lake)[0],
            data_lake_root=lake,
        )
        == manifest
    )
    assert (
        ForwardLabelSourceCapture.reproduce(
            manifest,
            data_lake_root=lake,
        )
        == manifest.source_paths(data_lake_root=lake)[0]
    )


def test_forward_label_reproduction_rejects_changed_polygon_payload(
    tmp_path: Path,
) -> None:
    manifest, polygon_observation = _source_fixture(tmp_path)
    polygon_observation.write_bytes(b"{}")

    with pytest.raises(ValueError, match="missing or differs"):
        ForwardLabelSourceCapture.reproduce(
            manifest,
            data_lake_root=tmp_path / "lake",
        )


def test_raw_forward_labels_normalize_only_through_horizon_split_vintage(
    tmp_path: Path,
) -> None:
    layout = LakehouseLayout(tmp_path / "lake")
    bronze = BronzeWriter(layout)
    plan_id = "7" * 64
    raw_prices = (
        (100.0, 101.0),
        (101.0, 102.0),
        (103.0, 104.0),
        (51.5, 52.0),
        (52.5, 53.0),
        (53.5, 54.0),
    )
    bars_raw = {
        "ticker": "AAPL",
        "adjusted": False,
        "status": "OK",
        "results": [
            {
                **_aggregate(session_date, index),
                "o": prices[0],
                "h": max(prices) + 1,
                "l": min(prices) - 1,
                "c": prices[1],
            }
            for index, (session_date, prices) in enumerate(zip(SESSIONS, raw_prices, strict=True))
        ],
    }
    bars_observation = bronze.write_json(
        bars_raw,
        source="polygon",
        dataset="daily-aggregate-bars",
        event_date=ASOF_DATE,
        received_at=OBSERVED_AT,
    )
    bars = PolygonClient.daily_bars_from_payload(
        bars_raw,
        symbol="AAPL",
        expected_adjusted=False,
    )
    daily_files = SilverWriter(layout).write_daily_bars(
        bars,
        ingested_at=OBSERVED_AT,
        availability_policy=DAILY_BAR_SESSION_CLOSE_15M,
    )
    DailyBarsSourceCapture(layout).write(
        symbols=("AAPL",),
        start_date=SESSIONS[0],
        end_date=SESSIONS[-1],
        ingested_at=OBSERVED_AT,
        silver_files=daily_files,
        provider_observations=(bars_observation,),
        availability_policy=DAILY_BAR_SESSION_CLOSE_15M,
        adjusted=False,
        backfill_plan_id=plan_id,
    )
    future_split_date = SESSIONS[-1] + timedelta(days=1)
    splits_raw = {
        "status": "OK",
        "results": [
            {
                "id": "split-inside-label-horizon",
                "ticker": "AAPL",
                "execution_date": SESSIONS[3].isoformat(),
                "adjustment_type": "forward_split",
                "split_from": 1,
                "split_to": 2,
            },
            {
                "id": "split-after-label-horizon",
                "ticker": "AAPL",
                "execution_date": future_split_date.isoformat(),
                "adjustment_type": "forward_split",
                "split_from": 1,
                "split_to": 10,
            },
        ],
    }
    split_observation = bronze.write_json(
        splits_raw,
        source="polygon",
        dataset="stock-splits",
        event_date=ASOF_DATE,
        received_at=OBSERVED_AT,
    )
    splits = PolygonClient.stock_splits_from_payloads(
        (splits_raw,),
        start_date=ASOF_DATE,
        end_date=future_split_date,
    )
    split_source = SplitHistorySourceCapture(layout).write(
        plan_id=plan_id,
        start_date=ASOF_DATE,
        end_date=future_split_date,
        ingested_at=OBSERVED_AT,
        split_files=SilverWriter(layout).write_splits(splits, ingested_at=OBSERVED_AT),
        provider_observations=(split_observation,),
    )
    calendar_raw = [
        {"date": item.isoformat(), "open": "09:30", "close": "16:00"} for item in SESSIONS
    ]
    calendar_observation = bronze.write_json(
        calendar_raw,
        source="alpaca",
        dataset="market-calendar",
        event_date=ASOF_DATE,
        received_at=OBSERVED_AT,
    )
    session_file = SessionFileStore(layout).write(
        AlpacaCalendarClient.sessions_from_payload(
            calendar_raw,
            start_date=SESSIONS[0],
            end_date=SESSIONS[-1],
        )
    )
    CalendarSourceCapture(layout).write(
        start_date=SESSIONS[0],
        end_date=SESSIONS[-1],
        session_file=session_file,
        provider_observations=(calendar_observation,),
    )

    with pytest.raises(ValueError, match="require a split-history source"):
        LabelBarsLoader().load(
            tuple(item.path for item in daily_files),
            symbols=("AAPL",),
            start_date=ASOF_DATE,
            end_date=SESSIONS[-1],
            observed_at=OBSERVED_AT,
        )
    loaded = LabelBarsLoader().load(
        tuple(item.path for item in daily_files),
        symbols=("AAPL",),
        start_date=ASOF_DATE,
        end_date=SESSIONS[-1],
        observed_at=OBSERVED_AT,
        split_source_manifest=split_source.path,
    )
    labels = ForwardLabelMaker().compute(
        keys=(("AAPL", ASOF_DATE),),
        sessions=SESSIONS,
        bars=loaded,
    )
    assert labels[0].forward_1d_close == pytest.approx(51 / 50.5 - 1)
    assert labels[0].forward_5d_close == pytest.approx(54 / 50.5 - 1)
    artifact = LabelStore(layout).write(labels, computed_at=OBSERVED_AT)
    manifest = ForwardLabelSourceCapture(layout).write(
        asof_date=ASOF_DATE,
        observed_at=OBSERVED_AT,
        symbols=("AAPL",),
        label_file=artifact,
        daily_bar_files=tuple(item.path for item in daily_files),
        session_file=session_file.path,
        split_source_manifest=split_source.path,
    )

    assert manifest.raw["schema_version"] == 2
    assert manifest.raw["split_source_manifest"] is not None
    assert (
        ForwardLabelSourceCapture.reproduce(
            manifest,
            data_lake_root=layout.root,
        )
        == artifact.path
    )
