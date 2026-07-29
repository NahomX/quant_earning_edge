"""Production refit boundary and inference tests."""

from __future__ import annotations

import json
from datetime import date, timedelta
from typing import TYPE_CHECKING

import pyarrow as pa
import pyarrow.parquet as pq

from quant_earning_edge.signals import ProductionModelTrainer

if TYPE_CHECKING:
    from pathlib import Path


def _dataset(path: Path, *, mutate_future: bool = False) -> None:
    first = date(2025, 1, 2)
    rows = []
    for index in range(70):
        sign = 1.0 if index % 2 == 0 else -1.0
        label = sign * 0.01
        if mutate_future and index >= 60:
            label = -label
        rows.append(
            {
                "asof_date": first + timedelta(days=index),
                "horizon_end_date": first + timedelta(days=index + 2),
                "signal": sign + index / 100,
                "forward_1d_close": label,
            }
        )
    pq.write_table(pa.Table.from_pylist(rows), path)  # type: ignore[no-untyped-call]


def test_production_refit_is_deterministic_and_scores_exact_features(tmp_path: Path) -> None:
    dataset = tmp_path / "training.parquet"
    output = tmp_path / "models"
    _dataset(dataset)
    trainer = ProductionModelTrainer(feature_names=("signal",), early_stopping_rounds=10)
    cutoff = date(2025, 3, 3)

    first = trainer.run(dataset_files=(dataset,), training_cutoff=cutoff)
    second = trainer.run(dataset_files=(dataset,), training_cutoff=cutoff)
    paths = trainer.write(first, output)
    trainer.write(first, output)

    assert first == second
    assert first.fit_end_date < first.validation_start_date < cutoff
    assert first.validation_end_date < cutoff
    assert 0 <= first.predict_probability({"signal": 1.0}) <= 1
    assert all(path.exists() for path in paths)
    assert json.loads(paths[1].read_text(encoding="utf-8"))["training_cutoff"] == cutoff.isoformat()


def test_rows_whose_labels_close_after_cutoff_cannot_change_model(tmp_path: Path) -> None:
    original = tmp_path / "original.parquet"
    mutated = tmp_path / "mutated.parquet"
    _dataset(original)
    _dataset(mutated, mutate_future=True)
    trainer = ProductionModelTrainer(feature_names=("signal",), early_stopping_rounds=10)
    cutoff = date(2025, 3, 3)

    first = trainer.run(dataset_files=(original,), training_cutoff=cutoff)
    second = trainer.run(dataset_files=(mutated,), training_cutoff=cutoff)

    assert first.model_sha256 == second.model_sha256
    assert first.model_text == second.model_text
    assert first.dataset_sha256 != second.dataset_sha256


def test_inference_rejects_missing_or_extra_features(tmp_path: Path) -> None:
    dataset = tmp_path / "training.parquet"
    _dataset(dataset)
    artifact = ProductionModelTrainer(feature_names=("signal",), early_stopping_rounds=10).run(
        dataset_files=(dataset,), training_cutoff=date(2025, 3, 3)
    )

    for values in ({}, {"signal": 1.0, "extra": 2.0}):
        try:
            artifact.predict_probability(values)
        except ValueError as error:
            assert "exactly match" in str(error)
        else:
            raise AssertionError("invalid inference feature vector was accepted")
