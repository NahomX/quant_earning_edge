"""Reconciliation age follows latest revisions and authoritative closes."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from typing import TYPE_CHECKING

import pytest

from quant_earning_edge.data.calendar import SessionFile
from quant_earning_edge.data.clients import MarketSession
from quant_earning_edge.live import (
    PaperOrderReconciliation,
    PaperReconciliationReport,
)
from quant_earning_edge.monitoring import ReconciliationAgeEvaluator

if TYPE_CHECKING:
    from pathlib import Path


def _calendar(tmp_path: Path) -> SessionFile:
    dates = (date(2026, 7, 27), date(2026, 7, 28), date(2026, 7, 29))
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


def _report(*, evaluated_at: datetime, broken: bool) -> PaperReconciliationReport:
    order = PaperOrderReconciliation(
        client_order_id="order-1",
        symbol="AAA",
        side="buy",
        intended_quantity=10,
        replay_filled_quantity=10,
        replay_fill_price=100,
        paper_status="new" if broken else "filled",
        paper_filled_quantity=0 if broken else 10,
        paper_fill_price=None if broken else 100,
        fill_quantity_difference=-10 if broken else 0,
        paper_vs_replay_price_bps=None if broken else 0,
        terminal=not broken,
        break_reasons=("paper order is not terminal after close",) if broken else (),
    )
    return PaperReconciliationReport(
        schema_version=1,
        session_date=date(2026, 7, 27),
        evaluated_at=evaluated_at,
        replay_evidence_sha256=("b" * 64,),
        orders=(order,),
        reconciliation_break_count=int(broken),
        all_orders_terminal=not broken,
        paper_pnl_is_gate_input=False,
    )


def test_unresolved_break_ages_only_after_completed_intervening_close(
    tmp_path: Path,
) -> None:
    broken = _report(evaluated_at=datetime(2026, 7, 27, 20, 5, tzinfo=UTC), broken=True)

    evidence = ReconciliationAgeEvaluator().evaluate(
        calendar=_calendar(tmp_path),
        reports=(broken,),
        control_date=date(2026, 7, 29),
        evaluated_at=datetime(2026, 7, 29, 13, 20, tzinfo=UTC),
    )

    assert evidence.unresolved_session_dates == (date(2026, 7, 27),)
    assert evidence.reconciliation_break_age_sessions == 1


def test_later_clean_revision_resolves_prior_break(tmp_path: Path) -> None:
    broken = _report(evaluated_at=datetime(2026, 7, 27, 20, 5, tzinfo=UTC), broken=True)
    resolved = _report(
        evaluated_at=broken.evaluated_at + timedelta(minutes=5),
        broken=False,
    )
    path = tmp_path / "resolved.json"
    resolved.write(path)

    evidence = ReconciliationAgeEvaluator().evaluate(
        calendar=_calendar(tmp_path),
        reports=(broken, PaperReconciliationReport.load(path)),
        control_date=date(2026, 7, 29),
        evaluated_at=datetime(2026, 7, 29, 13, 20, tzinfo=UTC),
    )

    assert evidence.unresolved_session_dates == ()
    assert evidence.reconciliation_break_age_sessions is None


def test_reconciliation_age_rejects_evidence_not_available_at_control(
    tmp_path: Path,
) -> None:
    future = _report(evaluated_at=datetime(2026, 7, 29, 13, 21, tzinfo=UTC), broken=True)

    with pytest.raises(ValueError, match="after the control"):
        ReconciliationAgeEvaluator().evaluate(
            calendar=_calendar(tmp_path),
            reports=(future,),
            control_date=date(2026, 7, 29),
            evaluated_at=datetime(2026, 7, 29, 13, 20, tzinfo=UTC),
        )
