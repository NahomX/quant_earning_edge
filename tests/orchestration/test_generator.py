"""One validated command generates the complete daily operational loop."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from typer.testing import CliRunner

from quant_earning_edge.cli import app
from quant_earning_edge.evaluation import Phase6AggregationSpec
from quant_earning_edge.features import FEATURE_VALUE_SCHEMA
from quant_earning_edge.orchestration import (
    AutomatedPlanningInputs,
    DailyWorkflowSpecGenerator,
    WorkflowRunSpec,
    WorkflowStage,
    WorkflowTrigger,
)
from quant_earning_edge.signals import ProductionModelTrainer, load_strategy_config
from quant_earning_edge.universe import EVENT_CANDIDATE_SCHEMA


def _strategy_path() -> Path:
    return Path(__file__).parents[2] / "configs/strategies/earnings_v1.yaml"


def _inputs(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    trade_date = date(2026, 7, 28)
    decision = datetime(2026, 7, 27, 22, 0, tzinfo=UTC)
    planning = tmp_path / "planning.json"
    planning.write_text(
        json.dumps(
            {
                "trade_date": trade_date.isoformat(),
                "decision_at": decision.isoformat(),
                "equity": 100_000,
                "candidates": [],
                "outcomes": [],
                "entry_submitted_at": "2026-07-28T13:30:00Z",
                "entry_expires_at": "2026-07-28T13:35:00Z",
                "exit_submitted_at": "2026-07-28T19:50:00Z",
                "exit_expires_at": "2026-07-28T20:01:00Z",
            }
        ),
        encoding="utf-8",
    )
    evaluated = datetime(2026, 7, 28, 13, 20, tzinfo=UTC)
    breakers = tmp_path / "breakers.json"
    breakers.write_text(
        json.dumps(
            {
                "observations": [
                    {
                        "session_date": trade_date.isoformat(),
                        "evaluated_at": evaluated.isoformat(),
                        "replay_notional": 0,
                        "replay_net_pnl": None,
                        "replay_fill_rate": None,
                        "polygon_data_observed_at": (evaluated - timedelta(minutes=1)).isoformat(),
                        "alpaca_data_observed_at": (evaluated - timedelta(minutes=1)).isoformat(),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    artifact_root = tmp_path / "artifacts"
    expected_session = (artifact_root / "trade_date=2026-07-28" / "replay-session.json").resolve()
    phase6 = tmp_path / "phase6.json"
    phase6.write_text(
        json.dumps(
            {
                "session_file": "sessions.json",
                "workflow_store_root": "workflow-store",
                "workflow_health_file": "health.json",
                "proof_start": "2026-07-28",
                "proof_end": "2026-07-28",
                "initial_cash": 100_000,
                "session_report_files": [str(expected_session)],
                "minimum_session_count": 90,
            }
        ),
        encoding="utf-8",
    )
    return planning, breakers, phase6, artifact_root


def test_generate_cli_writes_complete_bound_workflow(tmp_path: Path) -> None:
    planning, breakers, phase6, artifact_root = _inputs(tmp_path)
    output = tmp_path / "inbox" / "daily.json"
    arguments = [
        "workflow",
        "generate",
        "--trade-date",
        "2026-07-28",
        "--planning-spec",
        str(planning),
        "--strategy-config",
        str(_strategy_path()),
        "--breaker-spec",
        str(breakers),
        "--phase6-spec",
        str(phase6),
        "--artifact-root",
        str(artifact_root),
        "--output",
        str(output),
        "--worker-id",
        "paper-worker-1",
    ]

    result = CliRunner().invoke(app, arguments)
    repeated = CliRunner().invoke(app, arguments)

    assert result.exit_code == 0
    assert repeated.exit_code == 0
    spec = WorkflowRunSpec.model_validate_json(output.read_bytes())
    assert tuple(item.stage for item in spec.stages) == tuple(WorkflowStage)
    assert spec.trigger.value == "scheduled"
    assert json.loads(result.stdout)["sha256"] == spec.sha256
    assert spec.stages[0].not_before == datetime(
        2026,
        7,
        28,
        13,
        20,
        tzinfo=UTC,
    )
    assert all(
        stage.not_after == datetime(2026, 7, 28, 13, 35, tzinfo=UTC) for stage in spec.stages[:4]
    )
    assert all(stage.not_after is None for stage in spec.stages[4:])
    assert tuple(stage.maximum_attempts for stage in spec.stages) == (
        6,
        3,
        6,
        6,
        12,
        6,
        24,
        6,
    )
    capture = spec.stages[4].commands[0]
    assert capture.arguments[:2] == ("ingest", "frozen-market-events")
    assert spec.stages[4].not_before == datetime(
        2026,
        7,
        28,
        20,
        6,
        tzinfo=UTC,
    )
    replay = spec.stages[5]
    assert tuple(item.arguments[:2] for item in replay.commands) == (
        ("backtest", "materialize-frozen-replay-specs"),
        ("backtest", "replay-materialization"),
        ("evaluation", "replay-frozen-session"),
    )
    assert replay.commands[2].artifact_bindings[0].source_stage is WorkflowStage.REPLAY_ORDERS
    assert spec.stages[3].commands[0].artifact_json_keys == ("broker_observation_paths",)
    assert spec.stages[6].commands[0].arguments[:2] == (
        "paper",
        "reconcile-frozen-revision",
    )
    assert spec.stages[6].commands[0].artifact_json_keys == (
        "output",
        "broker_observation_paths",
    )


def test_generate_cli_can_refresh_breakers_inside_preopen_stage(tmp_path: Path) -> None:
    planning, _, phase6, artifact_root = _inputs(tmp_path)
    artifact_root.mkdir()
    session_file = tmp_path / "sessions.json"
    session_file.write_text(
        json.dumps(
            {
                "provider": "alpaca",
                "sessions": [
                    {
                        "session_date": "2026-07-27",
                        "open_at": "2026-07-27T13:30:00Z",
                        "close_at": "2026-07-27T20:00:00Z",
                    },
                    {
                        "session_date": "2026-07-28",
                        "open_at": "2026-07-28T13:30:00Z",
                        "close_at": "2026-07-28T20:00:00Z",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "daily-controlled.json"

    result = CliRunner().invoke(
        app,
        [
            "workflow",
            "generate",
            "--trade-date",
            "2026-07-28",
            "--planning-spec",
            str(planning),
            "--strategy-config",
            str(_strategy_path()),
            "--breaker-session-file",
            str(session_file),
            "--phase6-spec",
            str(phase6),
            "--artifact-root",
            str(artifact_root),
            "--output",
            str(output),
            "--worker-id",
            "paper-worker-1",
        ],
    )

    assert result.exit_code == 0
    assert json.loads(result.stdout)["breaker_mode"] == "self_refreshing"
    spec = WorkflowRunSpec.model_validate_json(output.read_bytes())
    freeze = spec.stages[0].commands[0]
    assert freeze.arguments[:2] == ("monitoring", "prepare-breaker-bundle")
    assert freeze.artifact_json_keys == (
        "freshness_path",
        "freshness_observation_paths",
        "reconciliation_age_path",
        "reconciliation_age_source_paths",
        "breaker_spec_path",
    )
    evaluate = spec.stages[2].commands[0]
    assert "--spec-file" not in evaluate.arguments
    assert evaluate.artifact_bindings[0].source_stage is WorkflowStage.FREEZE_INPUTS
    assert evaluate.artifact_bindings[0].file_glob == "breaker-controls-*.json"


def test_generator_requires_current_session_in_phase6_spec(tmp_path: Path) -> None:
    planning, breakers, phase6, artifact_root = _inputs(tmp_path)
    raw = json.loads(phase6.read_bytes())
    raw["session_report_files"] = []
    phase6.write_text(json.dumps(raw), encoding="utf-8")

    result = CliRunner().invoke(
        app,
        [
            "workflow",
            "generate",
            "--trade-date",
            "2026-07-28",
            "--planning-spec",
            str(planning),
            "--strategy-config",
            str(_strategy_path()),
            "--breaker-spec",
            str(breakers),
            "--phase6-spec",
            str(phase6),
            "--artifact-root",
            str(artifact_root),
            "--output",
            str(tmp_path / "daily.json"),
            "--worker-id",
            "worker",
        ],
    )

    assert result.exit_code == 2
    assert "deterministicreplay-sessionpath" in "".join(result.stderr.split())


def test_prepare_cli_builds_phase6_controls_and_self_refreshing_workflow(
    tmp_path: Path,
) -> None:
    planning, _, _, artifact_root = _inputs(tmp_path)
    session_file = tmp_path / "sessions.json"
    session_file.write_text(
        json.dumps(
            {
                "provider": "alpaca",
                "sessions": [
                    {
                        "session_date": "2026-07-27",
                        "open_at": "2026-07-27T13:30:00Z",
                        "close_at": "2026-07-27T20:00:00Z",
                    },
                    {
                        "session_date": "2026-07-28",
                        "open_at": "2026-07-28T13:30:00Z",
                        "close_at": "2026-07-28T20:00:00Z",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    env_file = tmp_path / ".env"
    env_file.write_text(
        f"DATA_LAKE_ROOT={tmp_path / 'lake'}",
        encoding="utf-8",
    )
    output = tmp_path / "inbox" / "2026-07-28.json"
    arguments = [
        "workflow",
        "prepare",
        "--trade-date",
        "2026-07-28",
        "--planning-spec",
        str(planning),
        "--strategy-config",
        str(_strategy_path()),
        "--session-file",
        str(session_file),
        "--proof-start",
        "2026-07-28",
        "--proof-end",
        "2026-07-28",
        "--initial-cash",
        "100000",
        "--artifact-root",
        str(artifact_root),
        "--output",
        str(output),
        "--worker-id",
        "paper-worker-1",
        "--stage-for-admission",
        "--env-file",
        str(env_file),
    ]

    result = CliRunner().invoke(app, arguments)
    repeated = CliRunner().invoke(app, arguments)

    assert result.exit_code == 0
    assert repeated.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload == json.loads(repeated.stdout)
    workflow = WorkflowRunSpec.model_validate_json(output.read_bytes())
    assert workflow.stages[0].commands[0].arguments[:2] == (
        "monitoring",
        "prepare-breaker-bundle",
    )
    phase6 = Phase6AggregationSpec.model_validate_json(Path(payload["phase6_output"]).read_bytes())
    expected_report = artifact_root.resolve() / "trade_date=2026-07-28" / "replay-session.json"
    assert phase6.session_report_files == (expected_report,)


def test_generator_captures_and_scores_at_decision_before_refreshing_breakers(
    tmp_path: Path,
) -> None:
    _, _, phase6, artifact_root = _inputs(tmp_path)
    decision_at = datetime(2026, 7, 28, 1, 30, tzinfo=UTC)
    controls_at = datetime(2026, 7, 28, 13, 20, tzinfo=UTC)
    session_file = tmp_path / "sessions.json"
    automated = AutomatedPlanningInputs(
        candidate_file=tmp_path / "candidates.parquet",
        session_file=session_file,
        model_evidence=tmp_path / "production.json",
        model_file=tmp_path / "production.txt",
        feature_files=(tmp_path / "features.parquet",),
        prior_replay_files=(tmp_path / "prior-replay.json",),
        initial_cash=100_000,
        capture_not_before=decision_at,
    )

    spec = DailyWorkflowSpecGenerator().generate(
        trade_date=date(2026, 7, 28),
        trigger=WorkflowTrigger.SCHEDULED,
        worker_id="worker",
        planning_spec=None,
        strategy_config=_strategy_path(),
        breaker_spec=None,
        phase6_spec=phase6,
        artifact_root=artifact_root,
        order_controls_not_before=controls_at,
        order_submission_not_after=datetime(2026, 7, 28, 13, 35, tzinfo=UTC),
        market_events_not_before=datetime(2026, 7, 28, 20, 6, tzinfo=UTC),
        breaker_session_file=session_file,
        automated_planning=automated,
    )

    freeze = spec.stages[0]
    generation = spec.stages[1]
    breakers = spec.stages[2]
    assert freeze.not_before == decision_at
    assert freeze.commands[0].arguments[:2] == ("model", "capture-live-source")
    assert [command.arguments[:2] for command in generation.commands] == [
        ("model", "score-live-planning"),
        ("model", "plan-live-orders"),
    ]
    assert breakers.not_before == controls_at
    assert [command.arguments[:2] for command in breakers.commands] == [
        ("monitoring", "prepare-breaker-bundle"),
        ("monitoring", "circuit-breakers"),
    ]
    assert breakers.commands[1].artifact_bindings[0].source_stage is breakers.stage


def test_prepare_cli_can_generate_model_scored_planning(tmp_path: Path) -> None:
    _, _, _, artifact_root = _inputs(tmp_path)
    strategy = load_strategy_config(_strategy_path())
    training = tmp_path / "training.parquet"
    first = date(2025, 1, 2)
    training_rows = []
    for index in range(70):
        sign = 1.0 if index % 2 == 0 else -1.0
        row: dict[str, object] = {
            "asof_date": first + timedelta(days=index),
            "horizon_end_date": first + timedelta(days=index + 2),
            "forward_1d_open_to_close": sign * 0.01,
        }
        row.update({name: sign + offset / 100 for offset, name in enumerate(strategy.features)})
        training_rows.append(row)
    pq.write_table(  # type: ignore[no-untyped-call]
        pa.Table.from_pylist(training_rows),
        training,
    )
    model = ProductionModelTrainer(
        feature_names=strategy.features,
        label_name=strategy.label.column_name,
        threshold=strategy.label.threshold,
        seed=strategy.seed,
        early_stopping_rounds=10,
    ).run(
        dataset_files=(training,),
        training_cutoff=date(2025, 3, 3),
        phase4_gate_sha256="f" * 64,
    )
    model_file, model_evidence = ProductionModelTrainer.write(model, tmp_path / "models")
    decision = datetime(2026, 7, 27, 22, tzinfo=UTC)
    features = tmp_path / "features.parquet"
    pq.write_table(  # type: ignore[no-untyped-call]
        pa.Table.from_pylist(
            [
                {
                    "symbol": "IGNORED",
                    "asof_date": date(2026, 7, 27),
                    "feature_name": "not-a-model-feature",
                    "value": 0.0,
                    "feature_code_hash": "a" * 64,
                    "input_sha256": "b" * 64,
                    "computed_at": decision,
                }
            ],
            schema=FEATURE_VALUE_SCHEMA,
        ),
        features,
    )
    source = tmp_path / "source.json"
    source.write_text(
        json.dumps(
            {
                "trade_date": "2026-07-28",
                "feature_asof_date": "2026-07-27",
                "decision_at": decision.isoformat(),
                "equity": 100_000,
                "observations": [],
                "outcomes": [],
                "entry_submitted_at": "2026-07-28T13:30:00Z",
                "entry_expires_at": "2026-07-28T13:35:00Z",
                "exit_submitted_at": "2026-07-28T19:50:00Z",
                "exit_expires_at": "2026-07-28T20:01:00Z",
            }
        ),
        encoding="utf-8",
    )
    session_file = tmp_path / "sessions.json"
    session_file.write_text(
        json.dumps(
            {
                "provider": "alpaca",
                "sessions": [
                    {
                        "session_date": "2026-07-27",
                        "open_at": "2026-07-27T13:30:00Z",
                        "close_at": "2026-07-27T20:00:00Z",
                    },
                    {
                        "session_date": "2026-07-28",
                        "open_at": "2026-07-28T13:30:00Z",
                        "close_at": "2026-07-28T20:00:00Z",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    env_file = tmp_path / ".env"
    env_file.write_text(f"DATA_LAKE_ROOT={tmp_path / 'lake'}", encoding="utf-8")
    output = tmp_path / "inbox" / "2026-07-28.json"

    result = CliRunner().invoke(
        app,
        [
            "workflow",
            "prepare",
            "--trade-date",
            "2026-07-28",
            "--planning-source",
            str(source),
            "--model-evidence",
            str(model_evidence),
            "--model-file",
            str(model_file),
            "--strategy-config",
            str(_strategy_path()),
            "--session-file",
            str(session_file),
            "--proof-start",
            "2026-07-28",
            "--proof-end",
            "2026-07-28",
            "--initial-cash",
            "100000",
            "--artifact-root",
            str(artifact_root),
            "--output",
            str(output),
            "--worker-id",
            "paper-worker-1",
            "--stage-for-admission",
            "--env-file",
            str(env_file),
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert Path(payload["planning_output"]).exists()
    assert Path(payload["planning_evidence_output"]).exists()
    workflow = WorkflowRunSpec.model_validate_json(output.read_bytes())
    assert Path(payload["planning_output"]).resolve() in workflow.stages[0].output_files

    candidates = tmp_path / "candidates.parquet"
    pq.write_table(  # type: ignore[no-untyped-call]
        pa.Table.from_pylist([], schema=EVENT_CANDIDATE_SCHEMA),
        candidates,
    )
    automated_output = tmp_path / "staging" / "automated-2026-07-28.json"
    automated_result = CliRunner().invoke(
        app,
        [
            "workflow",
            "prepare",
            "--trade-date",
            "2026-07-28",
            "--candidate-file",
            str(candidates),
            "--model-evidence",
            str(model_evidence),
            "--model-file",
            str(model_file),
            "--strategy-config",
            str(_strategy_path()),
            "--session-file",
            str(session_file),
            "--proof-start",
            "2026-07-28",
            "--proof-end",
            "2026-07-28",
            "--initial-cash",
            "100000",
            "--artifact-root",
            str(artifact_root),
            "--output",
            str(automated_output),
            "--worker-id",
            "paper-worker-1",
            "--stage-for-admission",
            "--env-file",
            str(env_file),
        ],
    )

    assert automated_result.exit_code == 0
    automated_workflow = WorkflowRunSpec.model_validate_json(automated_output.read_bytes())
    assert automated_workflow.stages[0].commands[0].arguments[:2] == (
        "model",
        "capture-live-source",
    )
    assert automated_workflow.stages[0].not_before == datetime(2026, 7, 28, 1, 30, tzinfo=UTC)
