"""Nested Optuna search and immutable selection evidence tests."""

from __future__ import annotations

from datetime import date, timedelta
from typing import TYPE_CHECKING

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from quant_earning_edge.backtest import WalkForwardConfig, WalkForwardPlanner
from quant_earning_edge.signals import OptunaLightgbmSearch, OptunaStudyArtifact
from quant_earning_edge.signals.hyperparameter_search import _top_k_daily_returns

if TYPE_CHECKING:
    from pathlib import Path


def _dataset(path: Path) -> None:
    first = date(2025, 1, 2)
    rows = []
    for session_index in range(70):
        session = first + timedelta(days=session_index)
        for symbol_index, symbol in enumerate(("AAA", "BBB", "CCC")):
            positive = (session_index + symbol_index) % 3 != 0
            sign = 1.0 if positive else -1.0
            magnitude = 0.004 + (session_index % 7) * 0.001
            rows.append(
                {
                    "symbol": symbol,
                    "asof_date": session,
                    "horizon_end_date": session + timedelta(days=5),
                    "signal": sign * (1 + symbol_index / 10) + session_index / 1_000,
                    "forward_1d_close": sign * magnitude,
                }
            )
    pq.write_table(pa.Table.from_pylist(rows), path)  # type: ignore[no-untyped-call]


def test_optuna_search_is_resumable_and_canonical(tmp_path: Path) -> None:
    dataset = tmp_path / "training.parquet"
    database = tmp_path / "study.sqlite3"
    output = tmp_path / "study.json"
    _dataset(dataset)
    plan = WalkForwardPlanner().build(
        (dataset,),
        config=WalkForwardConfig(
            minimum_train_sessions=40,
            embargo_sessions=5,
            test_sessions=10,
        ),
    )
    search = OptunaLightgbmSearch(
        feature_names=("signal",),
        early_stopping_rounds=5,
        top_k=2,
        requested_trials=2,
    )

    first = search.run(dataset_files=(dataset,), plan=plan, storage_path=database)
    second = search.run(dataset_files=(dataset,), plan=plan, storage_path=database)
    first.write(output)
    first.write(output)
    loaded = OptunaStudyArtifact.load(output)

    assert first == second == loaded
    assert len(first.trials) == 2
    assert first.best_trial_number in {item.number for item in first.trials}
    assert first.selected_hyperparameters_sha256 == first.selected_hyperparameters.sha256
    assert database.exists()
    first.validate_training_contract(
        dataset_files=(dataset,),
        plan=plan,
        feature_names=("signal",),
        label_name="forward_1d_close",
        threshold=0.0,
        seed=20260427,
        top_k=2,
        requested_trials=2,
    )


def test_optuna_artifact_rejects_different_contract(tmp_path: Path) -> None:
    dataset = tmp_path / "training.parquet"
    _dataset(dataset)
    plan = WalkForwardPlanner().build((dataset,), config=WalkForwardConfig(40, 10, 5))
    artifact = OptunaLightgbmSearch(
        feature_names=("signal",),
        early_stopping_rounds=5,
        requested_trials=1,
    ).run(dataset_files=(dataset,), plan=plan)

    try:
        artifact.validate_training_contract(
            dataset_files=(dataset,),
            plan=plan,
            feature_names=("signal",),
            label_name="forward_1d_close",
            threshold=0.0,
            seed=20260427,
            top_k=6,
            requested_trials=1,
        )
    except ValueError as error:
        assert "does not match" in str(error)
    else:
        raise AssertionError("mismatched Optuna contract was accepted")


def test_probability_ties_use_symbol_not_future_return() -> None:
    rows = [
        {"asof_date": date(2025, 1, 2), "symbol": "BBB"},
        {"asof_date": date(2025, 1, 2), "symbol": "AAA"},
    ]

    selected = _top_k_daily_returns(
        rows=rows,
        validation_indices=(0, 1),
        probabilities=(0.6, 0.6),
        continuous_labels=np.asarray((0.5, -0.2)),
        top_k=1,
    )

    assert selected == [-0.2]
