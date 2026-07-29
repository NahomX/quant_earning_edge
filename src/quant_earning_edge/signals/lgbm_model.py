"""Deterministic LightGBM walk-forward training and OOS evidence."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from datetime import date
from typing import TYPE_CHECKING, Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from quant_earning_edge.signals.lgbm_hyperparameters import LightgbmHyperparameters

if TYPE_CHECKING:
    from collections.abc import Sequence
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
    model_text: str = field(default="", repr=False)

    def __post_init__(self) -> None:
        if self.fold_index < 0 or not _is_sha256(self.model_sha256):
            raise ValueError("fold model identity is invalid")
        if min(self.best_iteration, self.fit_count, self.validation_count) < 1:
            raise ValueError("fold model counts must be positive")
        attribution_names = tuple(item.feature_name for item in self.feature_attribution)
        if (
            not attribution_names
            or len(set(attribution_names)) != len(attribution_names)
            or any(
                not item.feature_name.strip()
                or item.mean_absolute_shap < 0
                or not math.isfinite(item.mean_absolute_shap)
                for item in self.feature_attribution
            )
        ):
            raise ValueError("fold feature attribution is invalid")


@dataclass(frozen=True)
class WalkForwardModelRun:
    """Content-linked evidence for every walk-forward fold."""

    plan_sha256: str
    dataset_sha256: tuple[str, ...]
    feature_names: tuple[str, ...]
    label_name: str
    threshold: float
    seed: int
    hyperparameter_study_sha256: str | None
    hyperparameters: LightgbmHyperparameters
    hyperparameters_sha256: str
    lightgbm_version: str
    folds: tuple[FoldModelResult, ...]

    def __post_init__(self) -> None:
        if (
            not _is_sha256(self.plan_sha256)
            or not self.dataset_sha256
            or not all(_is_sha256(value) for value in self.dataset_sha256)
        ):
            raise ValueError("walk-forward model source digest is invalid")
        if self.hyperparameters.sha256 != self.hyperparameters_sha256:
            raise ValueError("walk-forward hyperparameters do not match their SHA-256")
        if (
            not self.feature_names
            or len(self.feature_names) > 20
            or len(set(self.feature_names)) != len(self.feature_names)
        ):
            raise ValueError("walk-forward feature names must contain 1-20 unique values")
        if self.hyperparameter_study_sha256 is not None and not _is_sha256(
            self.hyperparameter_study_sha256
        ):
            raise ValueError("walk-forward hyperparameter study digest is invalid")
        if tuple(item.fold_index for item in self.folds) != tuple(range(len(self.folds))):
            raise ValueError("walk-forward model folds must be consecutive")
        if any(
            tuple(item.feature_name for item in fold.feature_attribution) != self.feature_names
            for fold in self.folds
        ):
            raise ValueError("walk-forward fold attribution does not match feature order")
        predictions = tuple(item for fold in self.folds for item in fold.predictions)
        row_indices = tuple(item.row_index for item in predictions)
        if len(set(row_indices)) != len(row_indices):
            raise ValueError("walk-forward OOS prediction row indices must be unique")
        if any(
            item.row_index < 0
            or not item.symbol.strip()
            or not 0 <= item.probability_up <= 1
            or item.realized_label not in {0, 1}
            for item in predictions
        ):
            raise ValueError("walk-forward OOS prediction is invalid")

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

    @classmethod
    def load_evidence(cls, path: Path) -> WalkForwardModelRun:
        """Strictly reload canonical run evidence without loading boosters."""
        encoded = path.read_bytes()
        raw = json.loads(encoded)
        expected = {
            "plan_sha256",
            "dataset_sha256",
            "feature_names",
            "label_name",
            "threshold",
            "seed",
            "hyperparameter_study_sha256",
            "hyperparameters",
            "hyperparameters_sha256",
            "lightgbm_version",
            "folds",
        }
        if not isinstance(raw, dict) or set(raw) != expected:
            raise ValueError("walk-forward model run evidence schema mismatch")
        try:
            folds = tuple(_fold_from_evidence(item) for item in raw["folds"])
            run = cls(
                plan_sha256=str(raw["plan_sha256"]),
                dataset_sha256=tuple(str(item) for item in raw["dataset_sha256"]),
                feature_names=tuple(str(item) for item in raw["feature_names"]),
                label_name=str(raw["label_name"]),
                threshold=float(raw["threshold"]),
                seed=int(raw["seed"]),
                hyperparameter_study_sha256=(
                    None
                    if raw["hyperparameter_study_sha256"] is None
                    else str(raw["hyperparameter_study_sha256"])
                ),
                hyperparameters=LightgbmHyperparameters.from_dict(raw["hyperparameters"]),
                hyperparameters_sha256=str(raw["hyperparameters_sha256"]),
                lightgbm_version=str(raw["lightgbm_version"]),
                folds=folds,
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("invalid walk-forward model run evidence") from error
        if run.evidence_json_bytes() != encoded:
            raise ValueError("walk-forward model run evidence is not canonical")
        return run

    def validate_predictions(
        self,
        predictions: Sequence[OosPrediction],
        *,
        require_complete: bool = False,
    ) -> None:
        """Prove supplied probabilities are an exact subset or full run ledger."""
        expected = {item.row_index: item for fold in self.folds for item in fold.predictions}
        supplied = {item.row_index: item for item in predictions}
        if not supplied or len(supplied) != len(tuple(predictions)):
            raise ValueError("OOS prediction evidence is empty or contains duplicate rows")
        if any(expected.get(row_index) != item for row_index, item in supplied.items()):
            raise ValueError("supplied probability differs from walk-forward OOS evidence")
        if require_complete and set(supplied) != set(expected):
            raise ValueError("Phase 4 plans do not cover the complete OOS prediction ledger")


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
        hyperparameters: LightgbmHyperparameters | None = None,
        hyperparameter_study_sha256: str | None = None,
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
        self._hyperparameters = hyperparameters or LightgbmHyperparameters()
        if hyperparameter_study_sha256 is not None and not _is_sha256(hyperparameter_study_sha256):
            raise ValueError("hyperparameter study SHA-256 is invalid")
        self._hyperparameter_study_sha256 = hyperparameter_study_sha256

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
            hyperparameter_study_sha256=self._hyperparameter_study_sha256,
            hyperparameters=self._hyperparameters,
            hyperparameters_sha256=self._hyperparameters.sha256,
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
            **self._hyperparameters.classifier_kwargs(),
            subsample=1.0,
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


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _fold_from_evidence(raw: object) -> FoldModelResult:
    expected = {
        "fold_index",
        "model_sha256",
        "best_iteration",
        "fit_count",
        "validation_count",
        "predictions",
        "feature_attribution",
    }
    if not isinstance(raw, dict) or set(raw) != expected:
        raise ValueError("walk-forward fold evidence schema mismatch")
    predictions = tuple(_prediction_from_evidence(item) for item in raw["predictions"])
    attributions = tuple(_attribution_from_evidence(item) for item in raw["feature_attribution"])
    return FoldModelResult(
        fold_index=int(raw["fold_index"]),
        model_sha256=str(raw["model_sha256"]),
        best_iteration=int(raw["best_iteration"]),
        fit_count=int(raw["fit_count"]),
        validation_count=int(raw["validation_count"]),
        predictions=predictions,
        feature_attribution=attributions,
    )


def _prediction_from_evidence(raw: object) -> OosPrediction:
    expected = {
        "row_index",
        "symbol",
        "asof_date",
        "probability_up",
        "realized_label",
    }
    if not isinstance(raw, dict) or set(raw) != expected:
        raise ValueError("walk-forward prediction evidence schema mismatch")
    return OosPrediction(
        row_index=int(raw["row_index"]),
        symbol=str(raw["symbol"]),
        asof_date=date.fromisoformat(str(raw["asof_date"])),
        probability_up=float(raw["probability_up"]),
        realized_label=int(raw["realized_label"]),
    )


def _attribution_from_evidence(raw: object) -> FeatureAttribution:
    expected = {"feature_name", "mean_absolute_shap"}
    if not isinstance(raw, dict) or set(raw) != expected:
        raise ValueError("walk-forward feature-attribution schema mismatch")
    return FeatureAttribution(
        feature_name=str(raw["feature_name"]),
        mean_absolute_shap=float(raw["mean_absolute_shap"]),
    )
