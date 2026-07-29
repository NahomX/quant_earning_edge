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
    ReplaySessionReport,
)
from quant_earning_edge.orchestration import (
    DailyWorkflowState,
    DailyWorkflowStore,
    WorkflowHealthEvaluator,
    WorkflowHealthReport,
    WorkflowTrigger,
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


def _health(
    calendar: SessionFile,
    *,
    scheduled_count: int | None = None,
) -> WorkflowHealthReport:
    dates = tuple(item.session_date for item in calendar.sessions)
    count = len(dates) if scheduled_count is None else scheduled_count
    completed = dates[:count]
    return WorkflowHealthReport(
        schema_version=1,
        calendar_sha256=calendar.sha256,
        start_date=dates[0],
        end_date=dates[-1],
        authoritative_session_dates=dates,
        workflow_state_sha256=tuple(
            hashlib.sha256(item.isoformat().encode()).hexdigest() for item in completed
        ),
        scheduled_complete_dates=completed,
        manual_complete_dates=(),
        missing_dates=dates[count:],
        failed_dates=(),
        incomplete_dates=(),
        invalid_artifact_dates=(),
        operational_uptime=count / len(dates),
        maximum_consecutive_scheduled_successes=count,
        excess_stage_attempt_count=0,
        passes_five_session_unattended_gate=count >= 5,
    )


def _daily_report(
    session_date: date,
    *,
    initial_cash: float,
    return_rate: float,
    index: int,
) -> ReplaySessionReport:
    net_pnl = initial_cash * return_rate
    modeled_spread_cost = 1.0
    modeled_impact_cost = 0.5
    execution_residual_cost = 0.25
    execution_cost = modeled_spread_cost + modeled_impact_cost + execution_residual_cost
    commission = 0.25
    fill_gross_pnl = net_pnl + commission
    arrival_gross_pnl = fill_gross_pnl + execution_cost
    entry_price = 100.0
    exit_price = entry_price + fill_gross_pnl / 100
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
        arrival_gross_pnl=arrival_gross_pnl,
        realized_execution_slippage_cost=execution_cost,
        modeled_spread_cost=modeled_spread_cost,
        modeled_market_impact_cost=modeled_impact_cost,
        execution_residual_cost=execution_residual_cost,
        gross_pnl=fill_gross_pnl,
        commission=commission,
        net_pnl_on_matched_quantity=net_pnl,
        reconciled=True,
    )
    hashes = tuple(
        sorted(
            hashlib.sha256(f"{session_date}|{leg}".encode()).hexdigest()
            for leg in ("entry", "exit")
        )
    )
    return ReplaySessionReport(
        schema_version=2,
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
        arrival_gross_pnl=arrival_gross_pnl,
        realized_execution_slippage_cost=execution_cost,
        modeled_spread_cost=modeled_spread_cost,
        modeled_market_impact_cost=modeled_impact_cost,
        execution_residual_cost=execution_residual_cost,
        gross_pnl=fill_gross_pnl,
        commission=commission,
        net_pnl=net_pnl,
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
        workflow_health=_health(calendar),
        reports=reports,
        proof_start=calendar.sessions[0].session_date,
        proof_end=calendar.sessions[-1].session_date,
        initial_cash=100_000,
    )

    assert result.authoritative_session_count == 90
    assert result.observed_session_count == 90
    assert result.session_report_sha256 == tuple(item.sha256 for item in reports)
    assert result.bootstrap_resamples == 500
    assert result.seed == 7
    assert result.operational_uptime == 1
    assert result.net_sharpe is not None and result.net_sharpe > 0.8
    assert result.bootstrap_sharpe is not None
    assert result.bootstrap_sharpe.lower > 0.3
    assert result.fully_filled_order_rate == 1
    assert result.p90_realized_to_predicted_ratio == 1
    assert result.fill_gross_pnl == pytest.approx(
        (result.arrival_gross_pnl or 0.0)
        - result.modeled_spread_cost
        - result.modeled_market_impact_cost
        - result.execution_residual_cost
    )
    assert result.net_pnl == pytest.approx(
        (result.arrival_gross_pnl or 0.0)
        - result.modeled_spread_cost
        - result.modeled_market_impact_cost
        - result.execution_residual_cost
        - result.commission
    )
    assert [item.component for item in result.cost_attribution] == [
        "modeled_spread",
        "modeled_market_impact",
        "execution_residual",
        "commission",
    ]
    assert result.passes_phase6_gate
    assert result.verdict == "pass"


def test_phase6_report_hash_binds_exact_daily_replay_evidence(tmp_path: Path) -> None:
    calendar = _calendar(tmp_path, 90)
    dates = tuple(item.session_date for item in calendar.sessions)
    reports = _profitable_reports(dates)
    changed_reports = (
        _daily_report(
            dates[0],
            initial_cash=reports[0].initial_cash,
            return_rate=reports[0].net_return or 0.0,
            index=999,
        ),
        *reports[1:],
    )
    evaluator = Phase6GateEvaluator(bootstrap_resamples=50, seed=7)

    first = evaluator.evaluate(
        calendar=calendar,
        workflow_health=_health(calendar),
        reports=reports,
        proof_start=dates[0],
        proof_end=dates[-1],
        initial_cash=100_000,
    )
    changed = evaluator.evaluate(
        calendar=calendar,
        workflow_health=_health(calendar),
        reports=changed_reports,
        proof_start=dates[0],
        proof_end=dates[-1],
        initial_cash=100_000,
    )

    assert first.net_sharpe == changed.net_sharpe
    assert first.final_equity == changed.final_equity
    assert first.session_report_sha256 != changed.session_report_sha256
    assert first.sha256 != changed.sha256


def test_missing_sessions_fail_strict_uptime_gate(tmp_path: Path) -> None:
    calendar = _calendar(tmp_path, 90)
    all_reports = _profitable_reports(tuple(item.session_date for item in calendar.sessions))

    result = Phase6GateEvaluator(bootstrap_resamples=50).evaluate(
        calendar=calendar,
        workflow_health=_health(calendar, scheduled_count=85),
        reports=all_reports[:85],
        proof_start=calendar.sessions[0].session_date,
        proof_end=calendar.sessions[-1].session_date,
        initial_cash=100_000,
    )

    assert result.operational_uptime == pytest.approx(85 / 90)
    assert len(result.missing_session_dates) == 5
    assert not result.passes_uptime_gate
    assert not result.passes_phase6_gate
    assert result.verdict == "insufficient-evidence"


def test_complete_replay_reports_still_fail_without_scheduled_uptime(tmp_path: Path) -> None:
    calendar = _calendar(tmp_path, 90)
    reports = _profitable_reports(tuple(item.session_date for item in calendar.sessions))

    result = Phase6GateEvaluator(bootstrap_resamples=50).evaluate(
        calendar=calendar,
        workflow_health=_health(calendar, scheduled_count=85),
        reports=reports,
        proof_start=calendar.sessions[0].session_date,
        proof_end=calendar.sessions[-1].session_date,
        initial_cash=100_000,
    )

    assert result.observed_session_count == 90
    assert result.scheduled_complete_session_count == 85
    assert not result.passes_uptime_gate
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
            workflow_health=_health(calendar),
            reports=reports,
            proof_start=dates[0],
            proof_end=dates[-1],
            initial_cash=100_000,
        )


def test_phase6_cli_marks_short_fixture_as_insufficient(tmp_path: Path) -> None:
    calendar = _calendar(tmp_path, 2)
    workflow_root = tmp_path / "workflow-store"
    spec_path = tmp_path / "phase6.json"
    health_path = tmp_path / "workflow-health.json"
    workflow_store = DailyWorkflowStore(workflow_root)
    WorkflowHealthEvaluator().evaluate(
        calendar=calendar,
        store=workflow_store,
        start_date=calendar.sessions[0].session_date,
        end_date=calendar.sessions[-1].session_date,
    ).write(health_path)
    workflow_store.write(
        DailyWorkflowState.initialize(
            trade_date=calendar.sessions[0].session_date,
            now=datetime(2025, 1, 2, 12, tzinfo=UTC),
            trigger=WorkflowTrigger.SCHEDULED,
        )
    )
    output = tmp_path / "gate.json"
    spec_path.write_text(
        json.dumps(
            {
                "session_file": calendar.path.name,
                "workflow_store_root": workflow_root.name,
                "workflow_health_file": health_path.name,
                "proof_start": calendar.sessions[0].session_date.isoformat(),
                "proof_end": calendar.sessions[-1].session_date.isoformat(),
                "initial_cash": 100_000,
                "session_report_files": [],
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


def test_phase6_cli_rejects_forged_scheduled_uptime_summary(tmp_path: Path) -> None:
    calendar = _calendar(tmp_path, 2)
    workflow_root = tmp_path / "empty-workflow-store"
    health_path = tmp_path / "forged-health.json"
    _health(calendar).write(health_path)
    relocated_calendar = tmp_path / calendar.path.name
    relocated_calendar.write_bytes(calendar.path.read_bytes())
    spec_path = tmp_path / "phase6-forged.json"
    spec_path.write_text(
        json.dumps(
            {
                "session_file": relocated_calendar.name,
                "workflow_store_root": workflow_root.name,
                "workflow_health_file": health_path.name,
                "proof_start": calendar.sessions[0].session_date.isoformat(),
                "proof_end": calendar.sessions[-1].session_date.isoformat(),
                "initial_cash": 100_000,
                "session_report_files": [],
                "bootstrap_resamples": 10,
            }
        ),
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        app,
        [
            "evaluation",
            "phase6-gate",
            "--aggregation-spec",
            str(spec_path),
            "--output",
            str(tmp_path / "forged-gate.json"),
        ],
    )

    assert result.exit_code == 2
    plain_output = "".join(
        " " if "\u2500" <= character <= "\u257f" else character for character in result.output
    )
    assert "workflow health does not reproduce from the bound workflow store" in " ".join(
        plain_output.split()
    )
