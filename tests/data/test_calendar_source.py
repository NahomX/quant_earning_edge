"""Provider-source reconstruction for authoritative market sessions."""

from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from quant_earning_edge.data import (
    BronzeWriter,
    CalendarSourceCapture,
    CalendarSourceManifest,
    LakehouseLayout,
    SessionFileStore,
)
from quant_earning_edge.data.clients import AlpacaCalendarClient

START_DATE = date(2026, 7, 27)
END_DATE = date(2026, 7, 28)
CAPTURED_AT = datetime(2026, 7, 20, 12, tzinfo=UTC)


def _source_fixture(tmp_path: Path) -> tuple[CalendarSourceManifest, Path]:
    raw = [
        {"date": "2026-07-27", "open": "09:30", "close": "16:00"},
        {"date": "2026-07-28", "open": "09:30", "close": "16:00"},
    ]
    layout = LakehouseLayout(tmp_path / "lake")
    observation = BronzeWriter(layout).write_json(
        raw,
        source="alpaca",
        dataset="market-calendar",
        event_date=START_DATE,
        received_at=CAPTURED_AT,
    )
    sessions = AlpacaCalendarClient.sessions_from_payload(
        raw,
        start_date=START_DATE,
        end_date=END_DATE,
    )
    session_file = SessionFileStore(layout).write(sessions)
    manifest = CalendarSourceCapture(layout).write(
        start_date=START_DATE,
        end_date=END_DATE,
        session_file=session_file,
        provider_observations=(observation,),
    )
    return manifest, session_file.path


def test_calendar_reproduces_from_retained_alpaca_payload(tmp_path: Path) -> None:
    manifest, original_path = _source_fixture(tmp_path)

    discovered = CalendarSourceCapture.find_for_session(
        original_path,
        data_lake_root=tmp_path / "lake",
    )
    assert discovered.path == manifest.path

    with TemporaryDirectory(prefix="qee-calendar-test-") as temporary:
        reproduced = CalendarSourceCapture.reproduce(
            manifest,
            data_lake_root=tmp_path / "lake",
            output_layout=LakehouseLayout(Path(temporary)),
        )

        assert reproduced.path.read_bytes() == original_path.read_bytes()


def test_calendar_source_manifest_rejects_changed_provider_payload(
    tmp_path: Path,
) -> None:
    manifest, _ = _source_fixture(tmp_path)
    provider_path = manifest.provider_paths(data_lake_root=tmp_path / "lake")[0]
    provider_path.write_bytes(b"[]")

    with pytest.raises(ValueError, match="missing or differs"):
        CalendarSourceCapture.reproduce(
            manifest,
            data_lake_root=tmp_path / "lake",
            output_layout=LakehouseLayout(tmp_path / "reproduced"),
        )
