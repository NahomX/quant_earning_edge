"""Automatic causal live-planning assembly tests."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime, timedelta
from typing import TYPE_CHECKING

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from typer.testing import CliRunner

from quant_earning_edge.cli import app
from quant_earning_edge.features import FEATURE_VALUE_SCHEMA
from quant_earning_edge.signals import (
    LivePlanningAssembler,
    LivePlanningSourceSpec,
    ProductionModelTrainer,
    ScoredPlanningArtifact,
)

if TYPE_CHECKING:
    from pathlib import Path


def _training(path: Path) -> None:
    first = date(2025, 1, 2)
    rows = [
        {
            "asof_date": first + timedelta(days=index),
            "horizon_end_date": first + timedelta(days=index + 2),
            "signal": 1.0 if index % 2 == 0 else -1.0,
            "forward_1d_close": 0.01 if index % 2 == 0 else -0.01,
        }
        for index in range(70)
    ]
    pq.write_table(pa.Table.from_pylist(rows), path)  # type: ignore[no-untyped-call]


def _source(decision: datetime) -> LivePlanningSourceSpec:
    trade_date = date(2025, 3, 5)
    return LivePlanningSourceSpec.model_validate(
        {
            "trade_date": trade_date,
            "feature_asof_date": date(2025, 3, 4),
            "decision_at": decision,
            "equity": 100_000,
            "observations": [
                {
                    "symbol": "AAA",
                    "sector": "Technology",
                    "sizing_price": 100,
                    "sizing_price_observed_at": decision,
                    "frozen_average_daily_volume_shares": 1_000_000,
                    "decision_snapshot": {
                        "ticker": "AAA",
                        "observed_at": decision,
                        "bid_price": 99.9,
                        "ask_price": 100.1,
                        "bid_size": 100,
                        "ask_size": 100,
                        "last_trade_price": 100,
                        "last_trade_at": decision - timedelta(seconds=1),
                    },
                }
            ],
            "outcomes": [],
            "entry_submitted_at": datetime(2025, 3, 5, 14, 30, tzinfo=UTC),
            "entry_expires_at": datetime(2025, 3, 5, 14, 35, tzinfo=UTC),
            "exit_submitted_at": datetime(2025, 3, 5, 20, 50, tzinfo=UTC),
            "exit_expires_at": datetime(2025, 3, 5, 21, 1, tzinfo=UTC),
        }
    )


def _features(path: Path, computed_at: datetime) -> None:
    pq.write_table(  # type: ignore[no-untyped-call]
        pa.Table.from_pylist(
            [
                {
                    "symbol": "AAA",
                    "asof_date": date(2025, 3, 4),
                    "feature_name": "signal",
                    "value": 1.0,
                    "feature_code_hash": "a" * 64,
                    "input_sha256": "b" * 64,
                    "computed_at": computed_at,
                }
            ],
            schema=FEATURE_VALUE_SCHEMA,
        ),
        path,
    )


def test_live_planning_scores_features_without_manual_probability(tmp_path: Path) -> None:
    training = tmp_path / "training.parquet"
    features = tmp_path / "features.parquet"
    decision = datetime(2025, 3, 4, 22, tzinfo=UTC)
    _training(training)
    _features(features, decision)
    model = ProductionModelTrainer(feature_names=("signal",), early_stopping_rounds=10).run(
        dataset_files=(training,),
        training_cutoff=date(2025, 3, 3),
        phase4_gate_sha256="f" * 64,
    )

    artifact = LivePlanningAssembler().assemble(
        source=_source(decision),
        model=model,
        feature_files=(features,),
    )
    planning_path = tmp_path / "planning.json"
    evidence_path = tmp_path / "planning-evidence.json"
    LivePlanningAssembler.write(
        artifact,
        planning_output=planning_path,
        evidence_output=evidence_path,
    )

    assert len(artifact.planning.candidates) == 1
    assert 0 <= artifact.planning.candidates[0].probability_up <= 1
    assert b"probability_up" not in _source(decision).model_dump_json().encode()
    assert planning_path.read_bytes() == artifact.planning.canonical_bytes
    assert evidence_path.read_bytes() == artifact.canonical_bytes
    assert ScoredPlanningArtifact.load(evidence_path) == artifact


def test_live_planning_rejects_features_computed_after_decision(tmp_path: Path) -> None:
    training = tmp_path / "training.parquet"
    features = tmp_path / "features.parquet"
    decision = datetime(2025, 3, 4, 22, tzinfo=UTC)
    _training(training)
    _features(features, decision + timedelta(seconds=1))
    model = ProductionModelTrainer(feature_names=("signal",), early_stopping_rounds=10).run(
        dataset_files=(training,),
        training_cutoff=date(2025, 3, 3),
        phase4_gate_sha256="f" * 64,
    )

    with pytest.raises(ValueError, match="after decision_at"):
        LivePlanningAssembler().assemble(
            source=_source(decision),
            model=model,
            feature_files=(features,),
        )


def test_no_candidate_live_planning_requires_no_feature_artifact(tmp_path: Path) -> None:
    training = tmp_path / "training.parquet"
    decision = datetime(2025, 3, 4, 22, tzinfo=UTC)
    _training(training)
    model = ProductionModelTrainer(feature_names=("signal",), early_stopping_rounds=10).run(
        dataset_files=(training,),
        training_cutoff=date(2025, 3, 3),
        phase4_gate_sha256="f" * 64,
    )
    source = _source(decision).model_copy(update={"observations": ()})

    artifact = LivePlanningAssembler().assemble(
        source=source,
        model=model,
        feature_files=(),
    )

    assert artifact.planning.candidates == ()
    assert artifact.feature_file_sha256 == ()


def test_live_planning_cli_writes_linked_planning_and_evidence(tmp_path: Path) -> None:
    training = tmp_path / "training.parquet"
    features = tmp_path / "features.parquet"
    source_path = tmp_path / "source.json"
    planning_path = tmp_path / "planning.json"
    planning_evidence = tmp_path / "planning-evidence.json"
    decision = datetime(2025, 3, 4, 22, tzinfo=UTC)
    _training(training)
    _features(features, decision)
    source_path.write_text(_source(decision).model_dump_json(), encoding="utf-8")
    model = ProductionModelTrainer(feature_names=("signal",), early_stopping_rounds=10).run(
        dataset_files=(training,),
        training_cutoff=date(2025, 3, 3),
        phase4_gate_sha256="f" * 64,
    )
    model_path, model_evidence = ProductionModelTrainer.write(model, tmp_path / "models")

    result = CliRunner().invoke(
        app,
        [
            "model",
            "score-live-planning",
            "--source-spec",
            str(source_path),
            "--model-evidence",
            str(model_evidence),
            "--model-file",
            str(model_path),
            "--feature-file",
            str(features),
            "--planning-output",
            str(planning_path),
            "--evidence-output",
            str(planning_evidence),
        ],
    )

    assert result.exit_code == 0
    assert planning_path.exists()
    assert planning_evidence.exists()
    assert json.loads(result.stdout)["candidate_count"] == 1
