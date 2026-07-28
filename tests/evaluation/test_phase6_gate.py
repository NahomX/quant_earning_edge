"""Terminal 90-session NBBO replay gate aggregation."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, date, datetime, time, timedelta
from typing import TYPE_CHECKING

import pytest
from typer.testing import CliRunner

from quant_earning_edge.cli import app
from quant_earning_edge.data import LakehouseLayout, SessionFileStore
from quant_earning_edge.data.clients import MarketSession
from quant_earning_edge.evaluation import (
    Phase6GateEvaluator,
    ReplayRoundTripResult,
    ReplaySessionAggregator,
    ReplaySessionReport,
)

if TYPE_CHECKING:
    from pathlib import Path

    from quant_earning_edge.data.calendar import SessionFile


def _session_dates(count: int) -> tuple[date, ...]:
    values: list[date] = []
    current = date(2025, 1, 2)
    while len(values) < count:
        if current.weekday() < 5:
            values.append(current)
        current += timedelta(days=1)
    return tuple(values)


def _calendar(tmp_path: Path, count: int) -> SessionFile:
    sessions = tuple(
        MarketSession(
            session_date=session_date,
            open_at=datetime.combine(session_date, time(14, 30), tzinfo=UTC),
            close_at=datetime.combine(session_date, time(21), tzinfo=UTC),
        )
        for session_date in _session_dates(count)
    )
    return SessionFileStore(LakehouseLayout(tmp_path)).write(sessions)


def _daily_report(
    session_date: date,
    *,
    initial_cash: float,
    return_rate: float,
    index: int,
) -> ReplaySessionReport:
    pnl = initial_cash * return_rate
    entry_price = 100.0
    exit_price = entry_price + pnl / 100
    trip = ReplayRoundTripResult(
        trade_id=f"trade-{index}",
        symbol="AAA",
        side="long",
        intended_quantity=100,
        entry_filled_quantity=100,
        exit_filled_quantity=100,
        matched_quantity=100,
        unmatched_quantity=0,
        entry_fill_price=entry_price,
        exit_fill_price=exit_price,
        gross_pnl=pnl,
        commission=0.0,
        net_pnl_on_matched_quantity=pnl,
        reconciled=True,
    )
    hashes = tuple(
        sorted(
            hashlib.sha256(f"{session_date}|{leg}".encode()).hexdigest()
            for leg in ("entry", "exit")
        )
    )
    return ReplaySessionReport(
        schema_version=1,
        session_date=session_date,
        initial_cash=initial_cash,
        evidence_sha256=hashes,
        intended_order_count=2,
        fully_filled_order_count=2,
        fully_filled_order_rate=1.0,
        intended_share_count=200,
        filled_share_count=200,
        share_fill_rate=1.0,
        realized_adverse_slippage_bps=(1.0, 1.0),
        predicted_adverse_slippage_bps=(1.0, 1.0),
        realized_adverse_slippage_bps_p10=1.0,
        realized_adverse_slippage_bps_p50=1.0,
        realized_adverse_slippage_bps_p90=1.0,
        predicted_adverse_slippage_bps_p90=1.0,
        p90_realized_to_predicted_ratio=1.0,
        opening_auction_filled_share_count=0,
        reconciliation_break_count=0,
        gross_pnl=pnl,
        commission=0.0,
        net_pnl=pnl,
        net_return=return_rate,
        round_trips=(trip,),
    )


def _profitable_reports(dates: tuple[date, ...]) -> tuple[ReplaySessionReport, ...]:
    equity = 100_000.0
    reports = []
    for index, session_date in enumerate(dates):
        rate = 0.001 if index % 2 == 0 else 0.002
        report = _daily_report(
            session_date,
            initial_cash=equity,
            return_rate=rate,
            index=index,
        )
        reports.append(report)
        equity += report.net_pnl or 0.0
    return tuple(reports)


def test_phase6_gate_passes_only_with_all_locked_thresholds(tmp_path: Path) -> None:
    calendar = _calendar(tmp_path, 90)
    reports = _profitable_reports(tuple(item.session_date for item in calendar.sessions))

    result = Phase6GateEvaluator(bootstrap_resamples=500, seed=7).evaluate(
        calendar=calendar,
        reports=reports,
        proof_start=calendar.sessions[0].session_date,
        proof_end=calendar.sessions[-1].session_date,
        initial_cash=100_000,
    )

    assert result.authoritative_session_count == 90
    assert result.observed_session_count == 90
    assert result.operational_uptime == 1
    assert result.net_sharpe is not None and result.net_sharpe > 0.8
    assert result.bootstrap_sharpe is not None
    assert result.bootstrap_sharpe.lower > 0.3
    assert result.fully_filled_order_rate == 1
    assert result.p90_realized_to_predicted_ratio == 1
    assert result.passes_phase6_gate
    assert result.verdict == "pass"


def test_missing_sessions_fail_strict_uptime_gate(tmp_path: Path) -> None:
    calendar = _calendar(tmp_path, 90)
    all_reports = _profitable_reports(tuple(item.session_date for item in calendar.sessions))

    result = Phase6GateEvaluator(bootstrap_resamples=50).evaluate(
        calendar=calendar,
        reports=all_reports[:85],
        proof_start=calendar.sessions[0].session_date,
        proof_end=calendar.sessions[-1].session_date,
        initial_cash=100_000,
    )

    assert result.operational_uptime == pytest.approx(85 / 90)
    assert len(result.missing_session_dates) == 5
    assert not result.passes_uptime_gate
    assert not result.passes_phase6_gate
    assert result.verdict == "fail"


def test_phase6_gate_rejects_hidden_daily_capital_reset(tmp_path: Path) -> None:
    calendar = _calendar(tmp_path, 2)
    dates = tuple(item.session_date for item in calendar.sessions)
    reports = (
        _daily_report(dates[0], initial_cash=100_000, return_rate=0.01, index=0),
        _daily_report(dates[1], initial_cash=100_000, return_rate=0.01, index=1),
    )

    with pytest.raises(ValueError, match="capital continuity"):
        Phase6GateEvaluator(bootstrap_resamples=10).evaluate(
            calendar=calendar,
            reports=reports,
            proof_start=dates[0],
            proof_end=dates[-1],
            initial_cash=100_000,
        )


def test_phase6_cli_marks_short_fixture_as_insufficient(tmp_path: Path) -> None:
    calendar = _calendar(tmp_path, 2)
    report_paths = []
    for session in calendar.sessions:
        report = ReplaySessionAggregator().evaluate(
            evidence=(),
            round_trips=(),
            session_date=session.session_date,
            initial_cash=100_000,
        )
        path = tmp_path / f"{session.session_date}.json"
        report.write(path)
        report_paths.append(path.name)
    spec_path = tmp_path / "phase6.json"
    output = tmp_path / "gate.json"
    spec_path.write_text(
        json.dumps(
            {
                "session_file": calendar.path.name,
                "proof_start": calendar.sessions[0].session_date.isoformat(),
                "proof_end": calendar.sessions[-1].session_date.isoformat(),
                "initial_cash": 100_000,
                "session_report_files": report_paths,
                "bootstrap_resamples": 10,
            }
        ),
        encoding="utf-8",
    )
    relocated_calendar = tmp_path / calendar.path.name
    relocated_calendar.write_bytes(calendar.path.read_bytes())

    result = CliRunner().invoke(
        app,
        [
            "evaluation",
            "phase6-gate",
            "--aggregation-spec",
            str(spec_path),
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0
    command_result = json.loads(result.stdout)
    report = json.loads(output.read_bytes())
    assert command_result["verdict"] == "insufficient-evidence"
    assert not command_result["passes_phase6_gate"]
    assert report["required_session_count"] == 90
