"""Persistent inbox worker resumes specs and emits durable cycle heartbeats."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, date, datetime
from pathlib import Path

from typer.testing import CliRunner

from quant_earning_edge.cli import app
from quant_earning_edge.orchestration import (
    DailyWorkflowStore,
    QeeCommandResult,
    WorkerCycleReport,
    WorkflowInboxWorker,
    WorkflowStage,
)

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
    assert WorkerCycleReport.load(report_path) == report
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
    assert len(tuple((tmp_path / "lake").rglob("attention-*.json"))) == 1


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


def test_worker_finalizes_phase6_once_after_workflow_completion(tmp_path: Path) -> None:
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    spec_path = _write_spec(inbox)
    raw = json.loads(spec_path.read_bytes())
    for stage in raw["stages"]:
        stage_output = tmp_path / "stage-artifacts" / f"{stage['stage']}.json"
        stage["commands"][0]["arguments"][-1] = str(stage_output)
        stage["output_files"] = [str(stage_output)]
    aggregation = tmp_path / "controls" / "phase6.json"
    aggregation.parent.mkdir()
    aggregation.write_text("{}", encoding="utf-8")
    daily_root = tmp_path / "artifacts" / "trade_date=2026-07-28"
    phase6_output = daily_root / "phase6-progress.json"
    raw["stages"][-1]["commands"][0]["arguments"] = [
        "evaluation",
        "phase6-gate",
        "--aggregation-spec",
        str(aggregation),
        "--output",
        str(phase6_output),
    ]
    raw["stages"][-1]["output_files"] = [str(phase6_output)]
    spec_path.write_text(json.dumps(raw), encoding="utf-8")
    finalizer_calls = 0
    data_lake = tmp_path / "lake"

    def execute(
        argv: tuple[str, ...],
        *,
        cwd: Path,
        timeout_seconds: float,
    ) -> QeeCommandResult:
        nonlocal finalizer_calls
        del timeout_seconds
        if argv[3:5] == ("evaluation", "finalize-phase6"):
            finalizer_calls += 1
            output_directory = Path(argv[argv.index("--output-directory") + 1])
            output_directory.mkdir(parents=True, exist_ok=True)
            state = DailyWorkflowStore(data_lake).load_latest(date(2026, 7, 28))
            assert state is not None
            artifacts = {}
            for name in ("health", "aggregation", "gate_report"):
                path = output_directory / f"{name}.json"
                path.write_text(f'{{"artifact":"{name}"}}', encoding="utf-8")
                artifacts[name] = (path, hashlib.sha256(path.read_bytes()).hexdigest())
            marker = output_directory / f"finalization-{state.sha256}.json"
            marker.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "workflow_state_sha256": state.sha256,
                        "health_path": str(artifacts["health"][0]),
                        "health_sha256": artifacts["health"][1],
                        "aggregation_path": str(artifacts["aggregation"][0]),
                        "aggregation_sha256": artifacts["aggregation"][1],
                        "gate_report_path": str(artifacts["gate_report"][0]),
                        "gate_report_sha256": artifacts["gate_report"][1],
                    }
                ),
                encoding="utf-8",
            )
            return QeeCommandResult(return_code=0, stdout="{}")
        output = Path(argv[argv.index("--output") + 1])
        if not output.is_absolute():
            output = cwd / output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("{}", encoding="utf-8")
        return QeeCommandResult(return_code=0, stdout="{}")

    worker = WorkflowInboxWorker(
        data_lake_root=data_lake,
        worker_id="worker",
        clock=lambda: datetime.now(UTC),
        executor=execute,
    )
    first, _ = worker.run_once(inbox)
    second, _ = worker.run_once(inbox)

    assert first.all_complete
    assert second.all_complete
    assert finalizer_calls == 1


def test_worker_emits_one_attention_record_for_expired_order_window(
    tmp_path: Path,
) -> None:
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    spec_path = _write_spec(inbox)
    raw = json.loads(spec_path.read_bytes())
    raw["stages"][0]["not_after"] = "2026-07-28T13:35:00Z"
    spec_path.write_text(json.dumps(raw), encoding="utf-8")
    now = datetime(2026, 7, 28, 14, 0, tzinfo=UTC)
    data_lake = tmp_path / "lake"
    worker = WorkflowInboxWorker(
        data_lake_root=data_lake,
        worker_id="worker",
        clock=lambda: now,
    )

    first, _ = worker.run_once(inbox)
    second, _ = worker.run_once(inbox)

    assert not first.all_complete
    assert not second.all_complete
    assert first.results[0].error_type == "WorkflowWindowExpired"
    attention = tuple(data_lake.rglob("attention-*.json"))
    assert len(attention) == 1
    state = DailyWorkflowStore(data_lake).load_latest(date(2026, 7, 28))
    assert state is not None
    assert state.stages[0].attempts == 1
