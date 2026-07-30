"""Daily controls discover complete prior workflow artifacts and revisions."""

from __future__ import annotations

from datetime import UTC, date, datetime
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest

from quant_earning_edge.data.calendar import SessionFile
from quant_earning_edge.data.clients import MarketSession
from quant_earning_edge.evaluation import ReplaySessionReport
from quant_earning_edge.live import PaperReconciliationReport
from quant_earning_edge.monitoring import DailyControlEvidenceDiscovery
from quant_earning_edge.signals import FrozenDailyOrders

if TYPE_CHECKING:
    from pathlib import Path


def _calendar(tmp_path: Path) -> SessionFile:
    dates = (date(2026, 7, 27), date(2026, 7, 28))
    return SessionFile(
        path=tmp_path / "sessions.json",
        sha256="a" * 64,
        sessions=tuple(
            MarketSession(
                session_date=item,
                open_at=datetime(item.year, item.month, item.day, 13, 30, tzinfo=UTC),
                close_at=datetime(item.year, item.month, item.day, 20, 0, tzinfo=UTC),
            )
            for item in dates
        ),
    )


def test_discovers_complete_replay_pair_and_all_reconciliation_revisions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = date(2026, 7, 27)
    daily = tmp_path / "trade_date=2026-07-27"
    daily.mkdir()
    frozen_path = daily / "frozen-daily-orders.json"
    replay_path = daily / "replay-session.json"
    first_revision = daily / "paper-reconciliation-a.json"
    second_revision = daily / "paper-reconciliation-b.json"
    for path in (frozen_path, replay_path, first_revision, second_revision):
        path.write_text("{}", encoding="utf-8")
    frozen = SimpleNamespace(trade_date=session)
    replay = SimpleNamespace(session_date=session)
    reports = {
        first_revision: SimpleNamespace(session_date=session),
        second_revision: SimpleNamespace(session_date=session),
    }
    monkeypatch.setattr(FrozenDailyOrders, "load", staticmethod(lambda _: frozen))
    monkeypatch.setattr(ReplaySessionReport, "load", staticmethod(lambda _: replay))
    monkeypatch.setattr(
        PaperReconciliationReport,
        "load",
        staticmethod(lambda path: reports[path]),
    )

    discovered = DailyControlEvidenceDiscovery().discover(
        calendar=_calendar(tmp_path),
        artifact_root=tmp_path,
        control_date=date(2026, 7, 28),
    )

    assert discovered.frozen_order_files == (frozen_path,)
    assert discovered.replay_report_files == (replay_path,)
    assert discovered.reconciliation_report_files == (
        first_revision,
        second_revision,
    )


def test_discovery_fails_closed_on_incomplete_prior_replay_pair(tmp_path: Path) -> None:
    daily = tmp_path / "trade_date=2026-07-27"
    daily.mkdir()
    (daily / "frozen-daily-orders.json").write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="incomplete prior replay"):
        DailyControlEvidenceDiscovery().discover(
            calendar=_calendar(tmp_path),
            artifact_root=tmp_path,
            control_date=date(2026, 7, 28),
        )
