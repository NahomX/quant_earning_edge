"""Deterministic purged LightGBM walk-forward tests."""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from typer.testing import CliRunner

from quant_earning_edge.backtest import WalkForwardConfig, WalkForwardPlanner
from quant_earning_edge.cli import app
from quant_earning_edge.features import FEATURE_REGISTRY
from quant_earning_edge.signals import (
    LightgbmHyperparameters,
    LightgbmWalkForwardTrainer,
    WalkForwardModelRun,
)
from quant_earning_edge.signals.optuna_source import OptunaStudySourceCapture
from quant_earning_edge.signals.walkforward_source import WalkForwardModelSourceCapture


def _dataset(path: Path) -> None:
    first = date(2025, 1, 2)
    feature_names = tuple(item.name for item in FEATURE_REGISTRY.values())
    rows = []
    for session_index in range(70):
        session = first + timedelta(days=session_index)
        for symbol_index, symbol in enumerate(("AAA", "BBB")):
            label_sign = 1.0 if (session_index + symbol_index) % 2 == 0 else -1.0
            row: dict[str, object] = {
                "symbol": symbol,
                "asof_date": session,
                "horizon_end_date": session + timedelta(days=5),
                "information_cutoff_at": datetime(
                    session.year,
                    session.month,
                    session.day,
                    22,
                    tzinfo=UTC,
                ),
                "forward_1d_close": label_sign * 0.01,
                "forward_1d_open_to_close": label_sign * 0.01,
            }
            row.update(
                {
                    name: float(label_sign * (feature_index + 1) + session_index / 100)
                    for feature_index, name in enumerate(feature_names)
                }
            )
            rows.append(row)
    pq.write_table(  # type: ignore[no-untyped-call]
        pa.Table.from_pylist(rows),
        path,
    )


def test_walk_forward_models_and_oos_predictions_are_deterministic(tmp_path: Path) -> None:
    dataset = tmp_path / "training.parquet"
    plan_path = tmp_path / "plan.json"
    output = tmp_path / "models"
    _dataset(dataset)
    planner = WalkForwardPlanner()
    plan = planner.build(
        (dataset,),
        config=WalkForwardConfig(
            minimum_train_sessions=40,
            embargo_sessions=5,
            test_sessions=10,
        ),
    )
    planner.write(plan, plan_path)
    loaded_plan = planner.load(plan_path)
    trainer = LightgbmWalkForwardTrainer(
        feature_names=tuple(item.name for item in FEATURE_REGISTRY.values()),
        early_stopping_rounds=10,
    )

    first = trainer.run(dataset_files=(dataset,), plan=loaded_plan)
    second = trainer.run(dataset_files=(dataset,), plan=loaded_plan)
    trainer.write(first, output)
    trainer.write(first, output)

    assert first == second
    assert len(first.folds) == 2
    for fold_result, fold_plan in zip(first.folds, plan.folds, strict=True):
        assert {item.row_index for item in fold_result.predictions} == set(fold_plan.test_indices)
        assert set(fold_plan.train_indices).isdisjoint(fold_plan.test_indices)
        assert all(0 <= item.probability_up <= 1 for item in fold_result.predictions)
        assert all(
            item.information_cutoff_at.tzinfo is not None for item in fold_result.predictions
        )
        assert fold_result.fit_count < len(fold_plan.train_indices)
        assert fold_result.validation_count > 0
        assert len(fold_result.feature_attribution) == len(first.feature_names)
        assert {item.feature_name for item in fold_result.feature_attribution} == set(
            first.feature_names
        )
        assert all(item.mean_absolute_shap >= 0 for item in fold_result.feature_attribution)
    evidence = tuple(output.glob("run-*.json"))
    models = tuple(output.glob("fold-*.txt"))
    assert len(evidence) == 1
    assert len(models) == len(first.folds)
    assert json.loads(evidence[0].read_text(encoding="utf-8"))["seed"] == 20260427
    assert first.hyperparameters_sha256 == first.hyperparameters.sha256
    loaded = WalkForwardModelRun.load_evidence(evidence[0])
    assert loaded.evidence_json_bytes() == first.evidence_json_bytes()
    loaded.validate_predictions(
        tuple(item for fold in first.folds for item in fold.predictions),
        require_complete=True,
    )
    one_prediction = first.folds[0].predictions[0]
    with pytest.raises(ValueError, match="complete OOS prediction ledger"):
        loaded.validate_predictions((one_prediction,), require_complete=True)
    with pytest.raises(ValueError, match="differs from walk-forward"):
        loaded.validate_predictions(
            (replace(one_prediction, probability_up=1 - one_prediction.probability_up),)
        )


def test_walk_forward_run_binds_custom_hyperparameters(tmp_path: Path) -> None:
    dataset = tmp_path / "training.parquet"
    _dataset(dataset)
    plan = WalkForwardPlanner().build(
        (dataset,),
        config=WalkForwardConfig(40, 10, 5),
    )
    hyperparameters = LightgbmHyperparameters(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=7,
        min_child_samples=10,
        reg_alpha=0.25,
        reg_lambda=2.0,
        colsample_bytree=0.75,
    )
    run = LightgbmWalkForwardTrainer(
        feature_names=tuple(item.name for item in FEATURE_REGISTRY.values()),
        early_stopping_rounds=10,
        hyperparameters=hyperparameters,
    ).run(dataset_files=(dataset,), plan=plan)

    assert run.hyperparameters == hyperparameters
    assert run.hyperparameters_sha256 == hyperparameters.sha256
    assert json.loads(run.evidence_json_bytes())["hyperparameters"]["num_leaves"] == 7


def test_dataset_hash_mismatch_is_rejected(tmp_path: Path) -> None:
    dataset = tmp_path / "training.parquet"
    _dataset(dataset)
    plan = WalkForwardPlanner().build(
        (dataset,),
        config=WalkForwardConfig(40, 10, 5),
    )
    dataset.write_bytes(dataset.read_bytes() + b"tamper")

    trainer = LightgbmWalkForwardTrainer(
        feature_names=tuple(item.name for item in FEATURE_REGISTRY.values())
    )

    try:
        trainer.run(dataset_files=(dataset,), plan=plan)
    except ValueError as error:
        assert "does not match" in str(error)
    else:
        raise AssertionError("tampered dataset was accepted")


def test_walk_forward_training_cli_persists_models(tmp_path: Path) -> None:
    dataset = tmp_path / "training.parquet"
    plan_path = tmp_path / "plan.json"
    output = tmp_path / "models"
    strategy_config = tmp_path / "strategy.yaml"
    study_database = tmp_path / "study.sqlite3"
    study_output = tmp_path / "study.json"
    _dataset(dataset)
    planner = WalkForwardPlanner()
    plan = planner.build((dataset,), config=WalkForwardConfig(40, 10, 5))
    planner.write(plan, plan_path)
    strategy_config.write_text(
        (
            Path("configs/strategies/earnings_v1.yaml")
            .read_text(encoding="utf-8")
            .replace("n_trials: 200", "n_trials: 1")
        ),
        encoding="utf-8",
    )

    tune_result = CliRunner().invoke(
        app,
        [
            "model",
            "tune-walkforward",
            "--dataset-file",
            str(dataset),
            "--split-plan",
            str(plan_path),
            "--strategy-config",
            str(strategy_config),
            "--study-database",
            str(study_database),
            "--output",
            str(study_output),
        ],
    )

    result = CliRunner().invoke(
        app,
        [
            "model",
            "train-walkforward",
            "--dataset-file",
            str(dataset),
            "--split-plan",
            str(plan_path),
            "--strategy-config",
            str(strategy_config),
            "--hyperparameter-study",
            str(study_output),
            "--output-dir",
            str(output),
        ],
    )

    assert tune_result.exit_code == 0, tune_result.stdout
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    tune_payload = json.loads(tune_result.stdout)
    assert payload["fold_count"] == 2
    assert payload["prediction_count"] == 40
    assert payload["hyperparameter_study_sha256"] == tune_payload["artifact_sha256"]
    assert OptunaStudySourceCapture.find_for_study(study_output).path == Path(
        tune_payload["source_manifest"]
    )
    assert len(tuple(output.glob("fold-*.txt"))) == 2
    source_manifest = WalkForwardModelSourceCapture.find_for_run(next(output.glob("run-*.json")))
    assert Path(payload["source_manifest"]) == source_manifest.path
    assert WalkForwardModelSourceCapture.reproduce(source_manifest).sha256 == payload["run_sha256"]
    source_manifest.fold_model_paths()[0].write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="missing or differs"):
        WalkForwardModelSourceCapture.reproduce(source_manifest)
