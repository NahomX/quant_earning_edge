"""Deterministic LightGBM walk-forward training and OOS evidence."""

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
    from collections.abc import Sequence
    from datetime import date
    from pathlib import Path

    from quant_earning_edge.backtest import WalkForwardFold, WalkForwardPlan


@dataclass(frozen=True)
class OosPrediction:
    """One prediction from a row never supplied to model fitting."""

    row_index: int
    symbol: str
    asof_date: date
    probability_up: float
    realized_label: int


@dataclass(frozen=True)
class FeatureAttribution:
    """Mean absolute OOS SHAP contribution for one feature."""

    feature_name: str
    mean_absolute_shap: float


@dataclass(frozen=True)
class FoldModelResult:
    """Fold model identity and OOS predictions."""

    fold_index: int
    model_sha256: str
    best_iteration: int
    fit_count: int
    validation_count: int
    predictions: tuple[OosPrediction, ...]
    feature_attribution: tuple[FeatureAttribution, ...]
    model_text: str = field(repr=False)


@dataclass(frozen=True)
class WalkForwardModelRun:
    """Content-linked evidence for every walk-forward fold."""

    plan_sha256: str
    dataset_sha256: tuple[str, ...]
    feature_names: tuple[str, ...]
    label_name: str
    threshold: float
    seed: int
    lightgbm_version: str
    folds: tuple[FoldModelResult, ...]

    def evidence_json_bytes(self) -> bytes:
        """Serialize canonical evidence while models remain separate artifacts."""
        payload = asdict(self)
        for item in payload["folds"]:
            item.pop("model_text")
        return json.dumps(
            payload,
            default=lambda item: item.isoformat(),
            sort_keys=True,
            separators=(",", ":"),
        ).encode()

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.evidence_json_bytes()).hexdigest()


class LightgbmWalkForwardTrainer:
    """Fit purged rows and emit probabilities only for declared test rows."""

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
        plan: WalkForwardPlan,
    ) -> WalkForwardModelRun:
        """Validate exact inputs, fit each fold, and return OOS evidence."""
        hashes = _dataset_hashes(dataset_files)
        if hashes != plan.dataset_sha256:
            raise ValueError("dataset content does not match the split plan")
        table = _load_dataset(dataset_files)
        if table.num_rows != plan.sample_count:
            raise ValueError("dataset content or row count does not match the split plan")
        identity = ("symbol", "asof_date", "horizon_end_date", self._label_name)
        selected = (*identity, *self._feature_names)
        missing = [name for name in selected if name not in table.column_names]
        if missing:
            raise ValueError(f"training dataset is missing columns: {missing}")
        rows: list[dict[str, Any]] = table.select(list(selected)).to_pylist()
        features = np.asarray(
            [[float(row[name]) for name in self._feature_names] for row in rows],
            dtype=np.float64,
        )
        continuous_labels = np.asarray(
            [float(row[self._label_name]) for row in rows],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(features)) or not np.all(np.isfinite(continuous_labels)):
            raise ValueError("features and labels must be finite")
        labels = (continuous_labels > self._threshold).astype(np.int8)
        folds = tuple(
            self._fit_fold(fold=fold, rows=rows, features=features, labels=labels)
            for fold in plan.folds
        )
        return WalkForwardModelRun(
            plan_sha256=plan.sha256,
            dataset_sha256=hashes,
            feature_names=self._feature_names,
            label_name=self._label_name,
            threshold=self._threshold,
            seed=self._seed,
            lightgbm_version=str(_import_lightgbm().__version__),
            folds=folds,
        )

    def _fit_fold(
        self,
        *,
        fold: WalkForwardFold,
        rows: list[dict[str, Any]],
        features: np.ndarray,
        labels: np.ndarray,
    ) -> FoldModelResult:
        fit_indices, validation_indices = _internal_validation_split(
            fold.train_indices,
            rows=rows,
        )
        for indices, partition in (
            (fit_indices, "fit"),
            (validation_indices, "validation"),
            (fold.test_indices, "test"),
        ):
            if len({int(labels[index]) for index in indices}) < 2:
                raise ValueError(f"fold {fold.fold_index} {partition} partition needs both classes")
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
            features[list(fit_indices)],
            labels[list(fit_indices)],
            eval_X=features[list(validation_indices)],
            eval_y=labels[list(validation_indices)],
            callbacks=[lgb.early_stopping(self._early_stopping_rounds, verbose=False)],
        )
        probabilities = model.predict_proba(
            features[list(fold.test_indices)],
            num_iteration=model.best_iteration_,
        )[:, 1]
        contributions = np.asarray(
            model.booster_.predict(
                features[list(fold.test_indices)],
                num_iteration=model.best_iteration_,
                pred_contrib=True,
            ),
            dtype=np.float64,
        )
        if contributions.shape != (
            len(fold.test_indices),
            len(self._feature_names) + 1,
        ):
            raise RuntimeError("unexpected LightGBM SHAP contribution shape")
        model_text = str(model.booster_.model_to_string(num_iteration=model.best_iteration_))
        return FoldModelResult(
            fold_index=fold.fold_index,
            model_sha256=hashlib.sha256(model_text.encode()).hexdigest(),
            best_iteration=int(model.best_iteration_),
            fit_count=len(fit_indices),
            validation_count=len(validation_indices),
            predictions=tuple(
                OosPrediction(
                    row_index=row_index,
                    symbol=str(rows[row_index]["symbol"]),
                    asof_date=rows[row_index]["asof_date"],
                    probability_up=float(probability),
                    realized_label=int(labels[row_index]),
                )
                for row_index, probability in zip(
                    fold.test_indices,
                    probabilities,
                    strict=True,
                )
            ),
            feature_attribution=tuple(
                FeatureAttribution(
                    feature_name=name,
                    mean_absolute_shap=float(np.mean(np.abs(contributions[:, index]))),
                )
                for index, name in enumerate(self._feature_names)
            ),
            model_text=model_text,
        )

    @staticmethod
    def write(run: WalkForwardModelRun, output_dir: Path) -> None:
        """Persist immutable boosters and run evidence."""
        output_dir.mkdir(parents=True, exist_ok=True)
        for fold in run.folds:
            _write_immutable(
                output_dir / f"fold-{fold.fold_index:03d}-{fold.model_sha256[:12]}.txt",
                fold.model_text.encode(),
            )
        _write_immutable(output_dir / f"run-{run.sha256[:20]}.json", run.evidence_json_bytes())


def _internal_validation_split(
    train_indices: Sequence[int],
    *,
    rows: Sequence[dict[str, Any]],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    dates = tuple(sorted({rows[index]["asof_date"] for index in train_indices}))
    validation_session_count = max(2, math.ceil(len(dates) * 0.2))
    if len(dates) <= validation_session_count + 5:
        raise ValueError("training fold is too short for purged internal validation")
    validation_dates = set(dates[-validation_session_count:])
    validation_start = min(validation_dates)
    fit_cutoff = dates[-validation_session_count - 5]
    validation = tuple(
        index for index in train_indices if rows[index]["asof_date"] in validation_dates
    )
    fit = tuple(
        index
        for index in train_indices
        if rows[index]["asof_date"] < fit_cutoff
        and rows[index]["horizon_end_date"] < validation_start
    )
    if not fit or not validation:
        raise ValueError("internal validation purge produced an empty partition")
    return fit, validation


def _dataset_hashes(paths: Sequence[Path]) -> tuple[str, ...]:
    if not paths:
        raise ValueError("at least one dataset file is required")
    return tuple(hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(paths))


def _load_dataset(paths: Sequence[Path]) -> pa.Table:
    tables: list[pa.Table] = []
    for path in sorted(paths):
        tables.append(pq.read_table(path))  # type: ignore[no-untyped-call]
    return pa.concat_tables(tables)


def _write_immutable(path: Path, encoded: bytes) -> None:
    try:
        with path.open("xb") as destination:
            destination.write(encoded)
    except FileExistsError:
        if path.read_bytes() != encoded:
            raise RuntimeError(f"model artifact collision at {path}") from None


def _import_lightgbm() -> Any:
    try:
        import lightgbm as lgb  # noqa: PLC0415 - optional ML dependency.
    except ImportError as error:
        raise RuntimeError(
            "LightGBM is required; install the project with the 'ml' extra"
        ) from error
    return lgb
