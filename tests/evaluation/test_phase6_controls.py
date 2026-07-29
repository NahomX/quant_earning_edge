"""Rolling Phase 6 control files are derived from durable daily evidence."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime
from pathlib import Path

from typer.testing import CliRunner

from quant_earning_edge.cli import app
from quant_earning_edge.data import LakehouseLayout, SessionFileStore
from quant_earning_edge.data.clients import MarketSession
from quant_earning_edge.evaluation import (
    Phase6AggregationSpec,
    ReplaySessionAggregator,
)
from quant_earning_edge.orchestration import (
    DailyWorkflowRunner,
    DailyWorkflowState,
    DailyWorkflowStore,
    WorkflowHealthReport,
    WorkflowStage,
    WorkflowTrigger,
)


def test_prepare_phase6_controls_includes_current_future_output_path(
    tmp_path: Path,
) -> None:
    dates = (date(2026, 7, 27), date(2026, 7, 28))
    calendar = SessionFileStore(LakehouseLayout(tmp_path / "calendar")).write(
        tuple(
            MarketSession(
                session_date=item,
                open_at=datetime(item.year, item.month, item.day, 13, 30, tzinfo=UTC),
                close_at=datetime(item.year, item.month, item.day, 20, 0, tzinfo=UTC),
            )
            for item in dates
        )
    )
    artifact_root = tmp_path / "artifacts"
    prior_report = artifact_root / "trade_date=2026-07-27" / "replay-session.json"
    ReplaySessionAggregator().evaluate(
        evidence=(),
        round_trips=(),
        session_date=dates[0],
        initial_cash=100_000,
    ).write(prior_report)
    health_output = tmp_path / "controls" / "health.json"
    aggregation_output = tmp_path / "controls" / "phase6.json"
    env_file = tmp_path / ".env"
    env_file.write_text(
        f"DATA_LAKE_ROOT={tmp_path / 'lake'}",
        encoding="utf-8",
    )
    arguments = [
        "evaluation",
        "prepare-phase6-controls",
        "--session-file",
        str(calendar.path),
        "--proof-start",
        "2026-07-27",
        "--proof-end",
        "2026-07-28",
        "--current-trade-date",
        "2026-07-28",
        "--initial-cash",
        "100000",
        "--artifact-root",
        str(artifact_root),
        "--health-output",
        str(health_output),
        "--aggregation-output",
        str(aggregation_output),
        "--env-file",
        str(env_file),
    ]

    result = CliRunner().invoke(app, arguments)
    repeated = CliRunner().invoke(app, arguments)

    assert result.exit_code == 0
    assert repeated.exit_code == 0
    spec = Phase6AggregationSpec.model_validate_json(aggregation_output.read_bytes())
    expected_current = (artifact_root / "trade_date=2026-07-28" / "replay-session.json").resolve()
    assert spec.session_report_files == (prior_report.resolve(), expected_current)
    assert spec.workflow_store_root == (tmp_path / "lake").resolve()
    assert not expected_current.exists()
    health = WorkflowHealthReport.load(health_output)
    assert health.missing_dates == dates
    assert json.loads(result.stdout)["report_count"] == 2


def test_finalize_phase6_refreshes_health_after_workflow_completion(
    tmp_path: Path,
) -> None:
    session_date = date(2026, 7, 28)
    calendar = SessionFileStore(LakehouseLayout(tmp_path / "calendar")).write(
        (
            MarketSession(
                session_date=session_date,
                open_at=datetime(2026, 7, 28, 13, 30, tzinfo=UTC),
                close_at=datetime(2026, 7, 28, 20, 0, tzinfo=UTC),
            ),
        )
    )
    artifact_root = tmp_path / "artifacts"
    replay_path = artifact_root / "trade_date=2026-07-28" / "replay-session.json"
    ReplaySessionAggregator().evaluate(
        evidence=(),
        round_trips=(),
        session_date=session_date,
        initial_cash=100_000,
    ).write(replay_path)
    data_lake = tmp_path / "lake"
    store = DailyWorkflowStore(data_lake)

    def handler(
        _: DailyWorkflowState,
        stage: WorkflowStage,
    ) -> tuple[Path, ...]:
        output = tmp_path / "stage-artifacts" / f"{stage.value}.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("{}", encoding="utf-8")
        return (output,)

    state = DailyWorkflowRunner(
        store=store,
        handlers={stage: handler for stage in WorkflowStage},
        worker_id="worker",
        clock=lambda: datetime.now(UTC),
        trigger=WorkflowTrigger.SCHEDULED,
    ).run_until_idle(trade_date=session_date)
    assert state.complete
    original = tmp_path / "pre-run-phase6.json"
    original.write_text(
        json.dumps(
            {
                "session_file": str(calendar.path),
                "workflow_store_root": str(data_lake),
                "workflow_health_file": "pre-run-health.json",
                "proof_start": session_date.isoformat(),
                "proof_end": session_date.isoformat(),
                "initial_cash": 100_000,
                "session_report_files": [str(replay_path)],
            }
        ),
        encoding="utf-8",
    )
    env_file = tmp_path / ".env"
    env_file.write_text(f"DATA_LAKE_ROOT={data_lake}", encoding="utf-8")
    output_directory = tmp_path / "post-completion"
    arguments = [
        "evaluation",
        "finalize-phase6",
        "--aggregation-spec",
        str(original),
        "--current-trade-date",
        session_date.isoformat(),
        "--artifact-root",
        str(artifact_root),
        "--output-directory",
        str(output_directory),
        "--env-file",
        str(env_file),
    ]

    result = CliRunner().invoke(app, arguments)
    repeated = CliRunner().invoke(app, arguments)

    assert result.exit_code == 0
    assert repeated.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload == json.loads(repeated.stdout)
    gate = json.loads(Path(payload["gate_report_path"]).read_bytes())
    assert gate["scheduled_complete_session_count"] == 1
    assert gate["observed_session_count"] == 1
    assert Path(payload["manifest_path"]).is_file()
