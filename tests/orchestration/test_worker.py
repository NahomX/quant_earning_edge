"""Persistent inbox worker resumes specs and emits durable cycle heartbeats."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from typer.testing import CliRunner

from quant_earning_edge.cli import app
from quant_earning_edge.orchestration import (
    QeeCommandResult,
    WorkflowInboxWorker,
    WorkflowStage,
)

if TYPE_CHECKING:
    from pathlib import Path

_PREFIXES = {
    WorkflowStage.FREEZE_INPUTS: ("calendar", "sessions"),
    WorkflowStage.GENERATE_ORDER_PLAN: ("model", "plan-event-backtest"),
    WorkflowStage.EVALUATE_BREAKERS: ("monitoring", "circuit-breakers"),
    WorkflowStage.SUBMIT_PAPER_ORDERS: ("paper", "submit-batch"),
    WorkflowStage.CAPTURE_MARKET_EVENTS: ("ingest", "market-events"),
    WorkflowStage.REPLAY_ORDERS: ("backtest", "replay-nbbo"),
    WorkflowStage.RECONCILE_SESSION: ("paper", "reconcile"),
    WorkflowStage.EVALUATE_PHASE6_PROGRESS: ("evaluation", "phase6-gate"),
}


def _write_spec(inbox: Path) -> Path:
    path = inbox / "2026-07-28.json"
    path.write_text(
        json.dumps(
            {
                "trade_date": "2026-07-28",
                "trigger": "scheduled",
                "worker_id": "daily-worker",
                "stages": [
                    {
                        "stage": stage,
                        "commands": [
                            {
                                "arguments": [
                                    *_PREFIXES[stage],
                                    "--output",
                                    f"{stage.value}.json",
                                ]
                            }
                        ],
                        "output_files": [f"{stage.value}.json"],
                    }
                    for stage in WorkflowStage
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def test_worker_completes_inbox_and_persists_cycle_report(tmp_path: Path) -> None:
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    _write_spec(inbox)

    def execute(
        argv: tuple[str, ...],
        *,
        cwd: Path,
        timeout_seconds: float,
    ) -> QeeCommandResult:
        del timeout_seconds
        output = cwd / argv[argv.index("--output") + 1]
        output.write_text("{}", encoding="utf-8")
        return QeeCommandResult(return_code=0, stdout="{}")

    report, report_path = WorkflowInboxWorker(
        data_lake_root=tmp_path / "lake",
        worker_id="inbox-worker",
        clock=lambda: datetime.now(UTC),
        executor=execute,
    ).run_once(inbox)

    assert report.all_complete
    assert report.results[0].trade_date == "2026-07-28"
    assert report_path.exists()
    assert json.loads(report_path.read_bytes())["worker_id"] == "inbox-worker"


def test_invalid_spec_is_reported_without_stopping_other_scans(tmp_path: Path) -> None:
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    (inbox / "invalid.json").write_text('{"not":"a run spec"}', encoding="utf-8")

    report, _ = WorkflowInboxWorker(
        data_lake_root=tmp_path / "lake",
        worker_id="worker",
        clock=lambda: datetime.now(UTC),
    ).run_once(inbox)

    assert not report.all_complete
    assert report.results[0].error_type == "ValidationError"


def test_worker_cli_once_writes_empty_inbox_heartbeat(tmp_path: Path) -> None:
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    env_file = tmp_path / ".env"
    env_file.write_text(f"DATA_LAKE_ROOT={tmp_path / 'lake'}", encoding="utf-8")

    result = CliRunner().invoke(
        app,
        [
            "workflow",
            "worker",
            "--inbox",
            str(inbox),
            "--worker-id",
            "worker",
            "--once",
            "--env-file",
            str(env_file),
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["spec_count"] == 0
    assert payload["all_complete"]
