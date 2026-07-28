"""Unattended readiness counts only intact scheduled workflow completions."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

from typer.testing import CliRunner

from quant_earning_edge.cli import app
from quant_earning_edge.data import LakehouseLayout, SessionFileStore
from quant_earning_edge.data.clients import MarketSession
from quant_earning_edge.orchestration import (
    DailyWorkflowRunner,
    DailyWorkflowState,
    DailyWorkflowStore,
    WorkflowHealthEvaluator,
    WorkflowStage,
    WorkflowTrigger,
)

if TYPE_CHECKING:
    from quant_earning_edge.data.calendar import SessionFile


def _calendar(tmp_path: Path, dates: tuple[date, ...]) -> SessionFile:
    sessions = tuple(
        MarketSession(
            session_date=item,
            open_at=datetime.combine(item, datetime.min.time(), tzinfo=UTC)
            + timedelta(hours=14, minutes=30),
            close_at=datetime.combine(item, datetime.min.time(), tzinfo=UTC) + timedelta(hours=21),
        )
        for item in dates
    )
    return SessionFileStore(LakehouseLayout(tmp_path)).write(sessions)


def _complete(
    *,
    store: DailyWorkflowStore,
    root: Path,
    trade_date: date,
    trigger: WorkflowTrigger,
) -> DailyWorkflowState:
    def handler(
        _: DailyWorkflowState,
        stage: WorkflowStage,
    ) -> tuple[Path, ...]:
        output = root / f"{trade_date}-{stage.value}.json"
        output.write_text("{}", encoding="utf-8")
        return (output,)

    return DailyWorkflowRunner(
        store=store,
        handlers={stage: handler for stage in WorkflowStage},
        worker_id=f"{trigger.value}-worker",
        clock=lambda: datetime.now(UTC),
        trigger=trigger,
    ).run_until_idle(trade_date=trade_date)


def test_five_consecutive_scheduled_completions_pass_readiness(tmp_path: Path) -> None:
    dates = tuple(date(2026, 7, 20) + timedelta(days=index) for index in range(7))
    calendar = _calendar(tmp_path / "calendar", dates)
    store = DailyWorkflowStore(tmp_path / "lake")
    for item in dates[:5]:
        _complete(
            store=store,
            root=tmp_path,
            trade_date=item,
            trigger=WorkflowTrigger.SCHEDULED,
        )
    _complete(
        store=store,
        root=tmp_path,
        trade_date=dates[5],
        trigger=WorkflowTrigger.MANUAL,
    )

    report = WorkflowHealthEvaluator().evaluate(
        calendar=calendar,
        store=store,
        start_date=dates[0],
        end_date=dates[-1],
    )

    assert report.scheduled_complete_dates == dates[:5]
    assert report.manual_complete_dates == (dates[5],)
    assert report.missing_dates == (dates[6],)
    assert report.operational_uptime == 5 / 7
    assert report.maximum_consecutive_scheduled_successes == 5
    assert report.passes_five_session_unattended_gate


def test_manual_runs_and_tampered_artifacts_never_count_as_uptime(tmp_path: Path) -> None:
    dates = (date(2026, 7, 20), date(2026, 7, 21))
    calendar = _calendar(tmp_path / "calendar", dates)
    store = DailyWorkflowStore(tmp_path / "lake")
    manual = _complete(
        store=store,
        root=tmp_path,
        trade_date=dates[0],
        trigger=WorkflowTrigger.MANUAL,
    )
    scheduled = _complete(
        store=store,
        root=tmp_path,
        trade_date=dates[1],
        trigger=WorkflowTrigger.SCHEDULED,
    )
    Path(scheduled.stages[0].output_artifacts[0].path).write_text(
        '{"tampered":true}',
        encoding="utf-8",
    )

    report = WorkflowHealthEvaluator().evaluate(
        calendar=calendar,
        store=store,
        start_date=dates[0],
        end_date=dates[-1],
    )

    assert manual.complete
    assert report.scheduled_complete_dates == ()
    assert report.manual_complete_dates == (dates[0],)
    assert report.invalid_artifact_dates == (dates[1],)
    assert report.operational_uptime == 0


def test_health_cli_persists_passing_scheduled_evidence(tmp_path: Path) -> None:
    dates = tuple(date(2026, 7, 20) + timedelta(days=index) for index in range(5))
    calendar = _calendar(tmp_path / "calendar", dates)
    lake = tmp_path / "lake"
    store = DailyWorkflowStore(lake)
    for item in dates:
        _complete(
            store=store,
            root=tmp_path,
            trade_date=item,
            trigger=WorkflowTrigger.SCHEDULED,
        )
    env_file = tmp_path / ".env"
    env_file.write_text(f"DATA_LAKE_ROOT={lake}", encoding="utf-8")
    output = tmp_path / "health.json"

    result = CliRunner().invoke(
        app,
        [
            "workflow",
            "health",
            "--session-file",
            str(calendar.path),
            "--start",
            dates[0].isoformat(),
            "--end",
            dates[-1].isoformat(),
            "--output",
            str(output),
            "--env-file",
            str(env_file),
        ],
    )

    assert result.exit_code == 0
    assert json.loads(result.stdout)["passes_five_session_unattended_gate"]
    assert json.loads(output.read_bytes())["scheduled_complete_dates"] == [
        item.isoformat() for item in dates
    ]
