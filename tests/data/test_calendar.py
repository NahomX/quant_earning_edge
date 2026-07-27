"""Content-addressed market-session persistence tests."""

from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from quant_earning_edge.data.calendar import SessionFileStore
from quant_earning_edge.data.clients import MarketSession
from quant_earning_edge.data.layout import LakehouseLayout


def test_session_file_is_idempotent_and_round_trips(tmp_path: Path) -> None:
    eastern = ZoneInfo("America/New_York")
    sessions = (
        MarketSession(
            session_date=date(2026, 7, 27),
            open_at=datetime(2026, 7, 27, 9, 30, tzinfo=eastern),
            close_at=datetime(2026, 7, 27, 16, 0, tzinfo=eastern),
        ),
    )
    store = SessionFileStore(LakehouseLayout(tmp_path))

    first = store.write(sessions)
    second = store.write(sessions)
    loaded = store.load(first.path)

    assert first.path == second.path
    assert first.sha256 == second.sha256 == loaded.sha256
    assert loaded.sessions == sessions
