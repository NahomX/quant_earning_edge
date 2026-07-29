"""Provider-source reconstruction tests for forward-label artifacts."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from typing import TYPE_CHECKING

import pytest

from quant_earning_edge.data import (
    BronzeWriter,
    CalendarSourceCapture,
    DailyBarsSourceCapture,
    LakehouseLayout,
    SessionFileStore,
    SilverWriter,
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
