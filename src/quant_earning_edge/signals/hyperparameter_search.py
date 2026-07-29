"""Deterministic nested Optuna search for the earnings LightGBM model."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np

from quant_earning_edge.signals.lgbm_hyperparameters import LightgbmHyperparameters
from quant_earning_edge.signals.lgbm_model import (
    _dataset_hashes,
    _import_lightgbm,
    _internal_validation_split,
    _load_dataset,
)

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from quant_earning_edge.backtest import WalkForwardPlan


@dataclass(frozen=True)
class OptunaTrialEvidence:
    """Portable evidence for one completed or pruned trial."""

    number: int
    state: Literal["COMPLETE", "PRUNED"]
    value: float | None
    parameters: tuple[tuple[str, int | float], ...]

    def __post_init__(self) -> None:
        names = tuple(name for name, _ in self.parameters)
        if self.number < 0:
            raise ValueError("Optuna trial numbers must be nonnegative")
        if self.state not in {"COMPLETE", "PRUNED"}:
            raise ValueError("unsupported Optuna trial state")
        if names != tuple(sorted(set(names))):
            raise ValueError("Optuna trial parameter names must be unique and sorted")
        if self.value is not None and not math.isfinite(self.value):
            raise ValueError("Optuna trial values must be finite")
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for _, value in self.parameters
        ):
            raise ValueError("Optuna trial parameters must be finite numeric values")


@dataclass(frozen=True)
class OptunaStudyArtifact:
    """Immutable selection evidence from a deterministic nested search."""

    schema_version: int
    plan_sha256: str
    dataset_sha256: tuple[str, ...]
    feature_names: tuple[str, ...]
    label_name: str
    threshold: float
    seed: int
    top_k: int
    requested_trials: int
    objective: Literal["mean_nested_oos_sharpe"]
    sampler: Literal["TPESampler"]
    pruner: Literal["MedianPruner"]
    optuna_version: str
    lightgbm_version: str
    best_trial_number: int
    best_value: float
    selected_hyperparameters: LightgbmHyperparameters
    selected_hyperparameters_sha256: str
    trials: tuple[OptunaTrialEvidence, ...]

    def __post_init__(self) -> None:  # noqa: PLR0912 - explicit evidence contract.
        if self.schema_version != 1:
            raise ValueError("unsupported Optuna study artifact schema version")
        if self.objective != "mean_nested_oos_sharpe":
            raise ValueError("unsupported Optuna objective")
        if self.sampler != "TPESampler" or self.pruner != "MedianPruner":
            raise ValueError("unsupported Optuna sampler or pruner")
        if not 1 <= self.requested_trials <= 200:
            raise ValueError("Optuna requested_trials must be between 1 and 200")
        if len(self.trials) != self.requested_trials:
            raise ValueError("Optuna artifact does not contain every requested trial")
        if (
            not self.feature_names
            or len(self.feature_names) > 20
            or len(set(self.feature_names)) != len(self.feature_names)
        ):
            raise ValueError("Optuna artifact must contain 1-20 unique features")
        if self.top_k < 1:
            raise ValueError("Optuna top_k must be positive")
        if (
            not _is_sha256(self.plan_sha256)
            or not self.dataset_sha256
            or not all(_is_sha256(value) for value in self.dataset_sha256)
        ):
            raise ValueError("Optuna source SHA-256 is invalid")
        if not self.optuna_version or not self.lightgbm_version:
            raise ValueError("Optuna artifact package versions must be nonempty")
        if not math.isfinite(self.best_value):
            raise ValueError("Optuna best value must be finite")
        if self.selected_hyperparameters.sha256 != self.selected_hyperparameters_sha256:
            raise ValueError("selected hyperparameters do not match their SHA-256")
        completed = {
            trial.number: trial
            for trial in self.trials
            if trial.state == "COMPLETE" and trial.value is not None
        }
        if self.best_trial_number not in completed:
            raise ValueError("Optuna best trial is not complete")
        if completed[self.best_trial_number].value != self.best_value:
            raise ValueError("Optuna best value does not match the best trial")
        if self.best_value != max(cast("float", trial.value) for trial in completed.values()):
            raise ValueError("Optuna best trial does not maximize the objective")
        best_parameters = dict(completed[self.best_trial_number].parameters)
        if best_parameters != self.selected_hyperparameters.classifier_kwargs():
            raise ValueError("selected hyperparameters differ from the best trial")
        if tuple(trial.number for trial in self.trials) != tuple(range(self.requested_trials)):
            raise ValueError("Optuna trial ledger must be ordered and contiguous")

    def canonical_json_bytes(self) -> bytes:
        return json.dumps(
            asdict(self),
            sort_keys=True,
            separators=(",", ":"),
        ).encode()

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_json_bytes()).hexdigest()

    def write(self, path: Path) -> None:
        encoded = self.canonical_json_bytes()
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with path.open("xb") as destination:
                destination.write(encoded)
        except FileExistsError:
            if path.read_bytes() != encoded:
                raise RuntimeError(f"Optuna artifact collision at {path}") from None

    @classmethod
    def load(cls, path: Path) -> OptunaStudyArtifact:
        raw = json.loads(path.read_bytes())
        expected = {
            "schema_version",
            "plan_sha256",
            "dataset_sha256",
            "feature_names",
            "label_name",
            "threshold",
            "seed",
            "top_k",
            "requested_trials",
            "objective",
            "sampler",
            "pruner",
            "optuna_version",
            "lightgbm_version",
            "best_trial_number",
            "best_value",
            "selected_hyperparameters",
            "selected_hyperparameters_sha256",
            "trials",
        }
        if not isinstance(raw, dict) or set(raw) != expected:
            raise ValueError("Optuna study artifact schema mismatch")
        try:
            trials = tuple(
                OptunaTrialEvidence(
                    number=int(item["number"]),
                    state=item["state"],
                    value=None if item["value"] is None else float(item["value"]),
                    parameters=tuple((str(name), value) for name, value in item["parameters"]),
                )
                for item in raw["trials"]
            )
            artifact = cls(
                schema_version=int(raw["schema_version"]),
                plan_sha256=str(raw["plan_sha256"]),
                dataset_sha256=tuple(str(item) for item in raw["dataset_sha256"]),
                feature_names=tuple(str(item) for item in raw["feature_names"]),
                label_name=str(raw["label_name"]),
                threshold=float(raw["threshold"]),
                seed=int(raw["seed"]),
                top_k=int(raw["top_k"]),
                requested_trials=int(raw["requested_trials"]),
                objective=raw["objective"],
                sampler=raw["sampler"],
                pruner=raw["pruner"],
                optuna_version=str(raw["optuna_version"]),
                lightgbm_version=str(raw["lightgbm_version"]),
                best_trial_number=int(raw["best_trial_number"]),
                best_value=float(raw["best_value"]),
                selected_hyperparameters=LightgbmHyperparameters.from_dict(
                    raw["selected_hyperparameters"]
                ),
                selected_hyperparameters_sha256=str(raw["selected_hyperparameters_sha256"]),
                trials=trials,
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("invalid Optuna study artifact") from error
        if artifact.canonical_json_bytes() != path.read_bytes():
            raise ValueError("Optuna study artifact is not canonical")
        return artifact

    def validate_training_contract(
        self,
        *,
        dataset_files: Sequence[Path],
        plan: WalkForwardPlan,
        feature_names: Sequence[str],
        label_name: str,
        threshold: float,
        seed: int,
        top_k: int,
        requested_trials: int,
    ) -> None:
        """Reject reuse against any different research contract."""
        actual = (
            _dataset_hashes(dataset_files),
            plan.sha256,
            tuple(feature_names),
            label_name,
            threshold,
            seed,
            top_k,
            requested_trials,
        )
        expected = (
            self.dataset_sha256,
            self.plan_sha256,
            self.feature_names,
            self.label_name,
            self.threshold,
            self.seed,
            self.top_k,
            self.requested_trials,
        )
        if actual != expected:
            raise ValueError("Optuna study artifact does not match the training contract")


class OptunaLightgbmSearch:
    """Tune on purged inner validation partitions, preserving outer OOS folds."""

    def __init__(
        self,
        *,
        feature_names: Sequence[str],
        label_name: str = "forward_1d_close",
        threshold: float = 0.0,
        seed: int = 20260427,
        early_stopping_rounds: int = 50,
        top_k: int = 5,
        requested_trials: int = 200,
    ) -> None:
        names = tuple(feature_names)
        if not names or len(names) > 20 or len(set(names)) != len(names):
            raise ValueError("feature_names must contain 1-20 unique names")
        if early_stopping_rounds < 1:
            raise ValueError("early_stopping_rounds must be positive")
        if top_k < 1:
            raise ValueError("top_k must be positive")
        if not 1 <= requested_trials <= 200:
            raise ValueError("requested_trials must be between 1 and 200")
        self._feature_names = names
        self._label_name = label_name
        self._threshold = threshold
        self._seed = seed
        self._early_stopping_rounds = early_stopping_rounds
        self._top_k = top_k
        self._requested_trials = requested_trials

    def run(
        self,
        *,
        dataset_files: Sequence[Path],
        plan: WalkForwardPlan,
        storage_path: Path | None = None,
    ) -> OptunaStudyArtifact:
        """Run or resume the exact deterministic study and freeze its evidence."""
        hashes = _dataset_hashes(dataset_files)
        if hashes != plan.dataset_sha256:
            raise ValueError("dataset content does not match the split plan")
        rows, features, continuous_labels, labels = self._load_inputs(dataset_files, plan)
        optuna = _import_optuna()
        contract_sha256 = self._contract_sha256(plan=plan, dataset_sha256=hashes)
        storage: str | None = None
        if storage_path is not None:
            storage_path.parent.mkdir(parents=True, exist_ok=True)
            storage = f"sqlite:///{storage_path.resolve().as_posix()}"
        study = optuna.create_study(
            study_name=f"qee-lgbm-{contract_sha256[:20]}",
            storage=storage,
            load_if_exists=True,
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=self._seed),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=1),
        )
        previous_contract = study.user_attrs.get("qee_contract_sha256")
        if previous_contract not in {None, contract_sha256}:
            raise ValueError("persistent Optuna study belongs to a different contract")
        study.set_user_attr("qee_contract_sha256", contract_sha256)
        if len(study.trials) > self._requested_trials:
            raise ValueError("persistent Optuna study has more trials than requested")

        def objective(trial: Any) -> float:
            parameters = _suggest_hyperparameters(trial)
            fold_sharpes: list[float] = []
            for step, fold in enumerate(plan.folds):
                fit_indices, validation_indices = _internal_validation_split(
                    fold.train_indices,
                    rows=rows,
                )
                score = self._fit_and_score(
                    parameters=parameters,
                    rows=rows,
                    features=features,
                    continuous_labels=continuous_labels,
                    labels=labels,
                    fit_indices=fit_indices,
                    validation_indices=validation_indices,
                )
                fold_sharpes.append(score)
                trial.report(float(np.mean(fold_sharpes)), step)
                if trial.should_prune():
                    raise optuna.TrialPruned
            return float(np.mean(fold_sharpes))

        remaining = self._requested_trials - len(study.trials)
        if remaining:
            study.optimize(objective, n_trials=remaining, n_jobs=1)
        trials = tuple(_trial_evidence(item) for item in study.trials)
        best_parameters = LightgbmHyperparameters(**study.best_trial.params)
        return OptunaStudyArtifact(
            schema_version=1,
            plan_sha256=plan.sha256,
            dataset_sha256=hashes,
            feature_names=self._feature_names,
            label_name=self._label_name,
            threshold=self._threshold,
            seed=self._seed,
            top_k=self._top_k,
            requested_trials=self._requested_trials,
            objective="mean_nested_oos_sharpe",
            sampler="TPESampler",
            pruner="MedianPruner",
            optuna_version=str(optuna.__version__),
            lightgbm_version=str(_import_lightgbm().__version__),
            best_trial_number=int(study.best_trial.number),
            best_value=float(study.best_value),
            selected_hyperparameters=best_parameters,
            selected_hyperparameters_sha256=best_parameters.sha256,
            trials=trials,
        )

    def _load_inputs(
        self,
        dataset_files: Sequence[Path],
        plan: WalkForwardPlan,
    ) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray, np.ndarray]:
        table = _load_dataset(dataset_files)
        if table.num_rows != plan.sample_count:
            raise ValueError("dataset content or row count does not match the split plan")
        selected = (
            "symbol",
            "asof_date",
            "horizon_end_date",
            self._label_name,
            *self._feature_names,
        )
        missing = [name for name in selected if name not in table.column_names]
        if missing:
            raise ValueError(f"training dataset is missing columns: {missing}")
        rows: list[dict[str, Any]] = table.select(list(selected)).to_pylist()
        features = np.asarray(
            [[float(row[name]) for name in self._feature_names] for row in rows],
            dtype=np.float64,
        )
        continuous = np.asarray(
            [float(row[self._label_name]) for row in rows],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(features)) or not np.all(np.isfinite(continuous)):
            raise ValueError("features and labels must be finite")
        return rows, features, continuous, (continuous > self._threshold).astype(np.int8)

    def _fit_and_score(
        self,
        *,
        parameters: LightgbmHyperparameters,
        rows: Sequence[dict[str, Any]],
        features: np.ndarray,
        continuous_labels: np.ndarray,
        labels: np.ndarray,
        fit_indices: Sequence[int],
        validation_indices: Sequence[int],
    ) -> float:
        for indices, partition in (
            (fit_indices, "fit"),
            (validation_indices, "validation"),
        ):
            if len({int(labels[index]) for index in indices}) < 2:
                raise ValueError(f"Optuna {partition} partition needs both classes")
        lgb = _import_lightgbm()
        model = lgb.LGBMClassifier(
            objective="binary",
            **parameters.classifier_kwargs(),
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
            features[list(validation_indices)],
            num_iteration=model.best_iteration_,
        )[:, 1]
        daily_returns = _top_k_daily_returns(
            rows=rows,
            validation_indices=validation_indices,
            probabilities=probabilities,
            continuous_labels=continuous_labels,
            top_k=self._top_k,
        )
        if len(daily_returns) < 2:
            raise ValueError("Optuna validation fold needs at least two sessions")
        deviation = float(np.std(daily_returns, ddof=1))
        if deviation <= 1e-12:
            return 0.0
        return float(np.mean(daily_returns) / deviation * math.sqrt(252))

    def _contract_sha256(
        self,
        *,
        plan: WalkForwardPlan,
        dataset_sha256: tuple[str, ...],
    ) -> str:
        payload = {
            "dataset_sha256": dataset_sha256,
            "early_stopping_rounds": self._early_stopping_rounds,
            "feature_names": self._feature_names,
            "label_name": self._label_name,
            "plan_sha256": plan.sha256,
            "requested_trials": self._requested_trials,
            "seed": self._seed,
            "threshold": self._threshold,
            "top_k": self._top_k,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()


def _suggest_hyperparameters(trial: Any) -> LightgbmHyperparameters:
    return LightgbmHyperparameters(
        n_estimators=trial.suggest_int("n_estimators", 500, 2_000, step=500),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        num_leaves=trial.suggest_int("num_leaves", 7, 63),
        min_child_samples=trial.suggest_int("min_child_samples", 10, 100),
        reg_alpha=trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
    )


def _top_k_daily_returns(
    *,
    rows: Sequence[dict[str, Any]],
    validation_indices: Sequence[int],
    probabilities: Sequence[float],
    continuous_labels: np.ndarray,
    top_k: int,
) -> list[float]:
    """Select probability ranks with a symbol tie-breaker, never realized returns."""
    by_date: dict[object, list[tuple[float, str, float]]] = {}
    for index, probability in zip(validation_indices, probabilities, strict=True):
        by_date.setdefault(rows[index]["asof_date"], []).append(
            (
                float(probability),
                str(rows[index]["symbol"]),
                float(continuous_labels[index]),
            )
        )
    daily_returns: list[float] = []
    for candidates in by_date.values():
        selected = sorted(candidates, key=lambda item: (-item[0], item[1]))[:top_k]
        positive = [realized for probability, _, realized in selected if probability >= 0.5]
        daily_returns.append(float(np.mean(positive)) if positive else 0.0)
    return daily_returns


def _trial_evidence(trial: Any) -> OptunaTrialEvidence:
    state = str(trial.state.name)
    if state not in {"COMPLETE", "PRUNED"}:
        raise ValueError(f"unsupported Optuna trial state: {state}")
    return OptunaTrialEvidence(
        number=int(trial.number),
        state=cast("Literal['COMPLETE', 'PRUNED']", state),
        value=None if trial.value is None else float(trial.value),
        parameters=tuple(sorted(trial.params.items())),
    )


def _import_optuna() -> Any:
    try:
        import optuna  # noqa: PLC0415 - optional ML dependency.
    except ImportError as error:
        raise RuntimeError("Optuna is required; install the project with the 'ml' extra") from error
    return optuna


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)
