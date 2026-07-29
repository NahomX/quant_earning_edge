"""Cutoff-safe deterministic LightGBM refit for future decision-time inference."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from datetime import date
    from pathlib import Path


@dataclass(frozen=True)
class ProductionModelArtifact:
    """Immutable model and the exact causal training boundary that produced it."""

    schema_version: int
    training_cutoff: date
    dataset_sha256: tuple[str, ...]
    feature_names: tuple[str, ...]
    label_name: str
    threshold: float
    seed: int
    lightgbm_version: str
    fit_start_date: date
    fit_end_date: date
    validation_start_date: date
    validation_end_date: date
    fit_count: int
    validation_count: int
    best_iteration: int
    model_sha256: str
    model_text: str = field(repr=False, compare=True)

    def evidence_json_bytes(self) -> bytes:
        payload = asdict(self)
        payload.pop("model_text")
        return json.dumps(
            payload,
            default=lambda item: item.isoformat(),
            sort_keys=True,
            separators=(",", ":"),
        ).encode()

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.evidence_json_bytes()).hexdigest()

    def predict_probability(self, features: Mapping[str, float]) -> float:
        """Score one exact feature vector using the frozen booster."""
        if tuple(sorted(features)) != tuple(sorted(self.feature_names)):
            raise ValueError("inference feature names do not exactly match the model")
        values = np.asarray([[float(features[name]) for name in self.feature_names]])
        if not np.all(np.isfinite(values)):
            raise ValueError("inference features must be finite")
        booster = _import_lightgbm().Booster(model_str=self.model_text)
        prediction = booster.predict(values, num_iteration=self.best_iteration)
        return float(prediction[0])


class ProductionModelTrainer:
    """Refit a model using only rows whose labels close before a declared cutoff."""

    def __init__(
        self,
        *,
        feature_names: Sequence[str],
        label_name: str = "forward_1d_close",
        threshold: float = 0.0,
        seed: int = 20260427,
        early_stopping_rounds: int = 50,
    ) -> None:
        names = tuple(feature_names)
        if not names or len(names) > 20 or len(set(names)) != len(names):
            raise ValueError("feature_names must contain 1-20 unique names")
        if label_name not in {
            "forward_1d_open_to_close",
            "forward_1d_close",
            "forward_5d_close",
        }:
            raise ValueError("unsupported forward label")
        if early_stopping_rounds < 1:
            raise ValueError("early_stopping_rounds must be positive")
        self._feature_names = names
        self._label_name = label_name
        self._threshold = threshold
        self._seed = seed
        self._early_stopping_rounds = early_stopping_rounds

    def run(
        self,
        *,
        dataset_files: Sequence[Path],
        training_cutoff: date,
    ) -> ProductionModelArtifact:
        """Fit on closed labels before cutoff and retain a purged final validation."""
        hashes = _dataset_hashes(dataset_files)
        table = _load_dataset(dataset_files)
        selected = ("asof_date", "horizon_end_date", self._label_name, *self._feature_names)
        missing = [name for name in selected if name not in table.column_names]
        if missing:
            raise ValueError(f"training dataset is missing columns: {missing}")
        all_rows: list[dict[str, Any]] = table.select(list(selected)).to_pylist()
        rows = [
            row
            for row in all_rows
            if row["asof_date"] < training_cutoff and row["horizon_end_date"] < training_cutoff
        ]
        dates = tuple(sorted({row["asof_date"] for row in rows}))
        validation_session_count = max(2, math.ceil(len(dates) * 0.2))
        if len(dates) <= validation_session_count + 5:
            raise ValueError("training history is too short for purged final validation")
        validation_dates = set(dates[-validation_session_count:])
        validation_start = min(validation_dates)
        fit_cutoff = dates[-validation_session_count - 5]
        fit_rows = [
            row
            for row in rows
            if row["asof_date"] < fit_cutoff and row["horizon_end_date"] < validation_start
        ]
        validation_rows = [row for row in rows if row["asof_date"] in validation_dates]
        if not fit_rows or not validation_rows:
            raise ValueError("production validation purge produced an empty partition")

        fit_x, fit_y = self._matrix(fit_rows)
        validation_x, validation_y = self._matrix(validation_rows)
        for labels, partition in ((fit_y, "fit"), (validation_y, "validation")):
            if len(set(labels.tolist())) < 2:
                raise ValueError(f"production {partition} partition needs both classes")
        lgb = _import_lightgbm()
        model = lgb.LGBMClassifier(
            objective="binary",
            n_estimators=1_000,
            learning_rate=0.03,
            num_leaves=15,
            reg_lambda=1.0,
            subsample=1.0,
            colsample_bytree=1.0,
            random_state=self._seed,
            deterministic=True,
            force_col_wise=True,
            n_jobs=1,
            verbosity=-1,
        )
        model.fit(
            fit_x,
            fit_y,
            eval_X=validation_x,
            eval_y=validation_y,
            callbacks=[lgb.early_stopping(self._early_stopping_rounds, verbose=False)],
        )
        model_text = str(model.booster_.model_to_string(num_iteration=model.best_iteration_))
        return ProductionModelArtifact(
            schema_version=1,
            training_cutoff=training_cutoff,
            dataset_sha256=hashes,
            feature_names=self._feature_names,
            label_name=self._label_name,
            threshold=self._threshold,
            seed=self._seed,
            lightgbm_version=str(lgb.__version__),
            fit_start_date=min(row["asof_date"] for row in fit_rows),
            fit_end_date=max(row["asof_date"] for row in fit_rows),
            validation_start_date=min(row["asof_date"] for row in validation_rows),
            validation_end_date=max(row["asof_date"] for row in validation_rows),
            fit_count=len(fit_rows),
            validation_count=len(validation_rows),
            best_iteration=int(model.best_iteration_),
            model_sha256=hashlib.sha256(model_text.encode()).hexdigest(),
            model_text=model_text,
        )

    def _matrix(self, rows: Sequence[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
        features = np.asarray(
            [[float(row[name]) for name in self._feature_names] for row in rows],
            dtype=np.float64,
        )
        continuous = np.asarray([float(row[self._label_name]) for row in rows])
        if not np.all(np.isfinite(features)) or not np.all(np.isfinite(continuous)):
            raise ValueError("features and labels must be finite")
        return features, (continuous > self._threshold).astype(np.int8)

    @staticmethod
    def write(artifact: ProductionModelArtifact, output_dir: Path) -> tuple[Path, Path]:
        output_dir.mkdir(parents=True, exist_ok=True)
        model_path = output_dir / f"production-{artifact.model_sha256[:20]}.txt"
        evidence_path = output_dir / f"production-{artifact.sha256[:20]}.json"
        _write_immutable(model_path, artifact.model_text.encode())
        _write_immutable(evidence_path, artifact.evidence_json_bytes())
        return model_path, evidence_path


def _dataset_hashes(paths: Sequence[Path]) -> tuple[str, ...]:
    if not paths:
        raise ValueError("at least one dataset file is required")
    return tuple(hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(paths))


def _load_dataset(paths: Sequence[Path]) -> pa.Table:
    return pa.concat_tables(
        [pq.read_table(path) for path in sorted(paths)]  # type: ignore[no-untyped-call]
    )


def _write_immutable(path: Path, encoded: bytes) -> None:
    try:
        with path.open("xb") as destination:
            destination.write(encoded)
    except FileExistsError:
        if path.read_bytes() != encoded:
            raise RuntimeError(f"model artifact collision at {path}") from None


def _import_lightgbm() -> Any:
    try:
        import lightgbm as lgb  # noqa: PLC0415
    except ImportError as error:
        raise RuntimeError(
            "LightGBM is required; install the project with the 'ml' extra"
        ) from error
    return lgb
