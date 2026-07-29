"""Content-addressed reconstruction evidence for walk-forward OOS models."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from quant_earning_edge.backtest import WalkForwardPlanner
from quant_earning_edge.signals.config import load_strategy_config
from quant_earning_edge.signals.hyperparameter_search import OptunaStudyArtifact
from quant_earning_edge.signals.lgbm_model import (
    LightgbmWalkForwardTrainer,
    WalkForwardModelRun,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence


@dataclass(frozen=True)
class WalkForwardModelSourceManifest:
    """Exact datasets and research contract behind one OOS model run."""

    path: Path
    raw: dict[str, Any]

    @classmethod
    def load(cls, path: Path) -> WalkForwardModelSourceManifest:
        try:
            encoded = path.read_bytes()
            raw = json.loads(encoded)
        except (OSError, ValueError) as error:
            raise ValueError(f"invalid walk-forward source manifest: {path}") from error
        required = {
            "schema_version",
            "run_evidence",
            "fold_models",
            "dataset_files",
            "split_plan",
            "strategy_config",
            "hyperparameter_study",
        }
        if not isinstance(raw, dict) or set(raw) != required or raw["schema_version"] != 1:
            raise ValueError("walk-forward source manifest schema mismatch")
        singletons = ("run_evidence", "split_plan", "strategy_config", "hyperparameter_study")
        if any(not isinstance(raw[name], dict) for name in singletons) or any(
            not isinstance(raw[name], list) or not raw[name]
            for name in ("fold_models", "dataset_files")
        ):
            raise ValueError("walk-forward source manifest collections are invalid")
        entries = (
            *(raw[name] for name in singletons),
            *raw["fold_models"],
            *raw["dataset_files"],
        )
        for entry in entries:
            _validate_entry(entry)
        identities = tuple(entry["path"] for entry in entries)
        if len(identities) != len(set(identities)):
            raise ValueError("walk-forward source manifest paths are duplicated")
        if any(
            tuple(raw[name]) != tuple(sorted(raw[name], key=lambda item: item["path"]))
            for name in ("fold_models", "dataset_files")
        ):
            raise ValueError("walk-forward source manifest entries are not sorted")
        if json.dumps(raw, sort_keys=True, separators=(",", ":")).encode() != encoded:
            raise ValueError("walk-forward source manifest is not canonical")
        return cls(path=path.resolve(), raw=raw)

    def run_path(self) -> Path:
        return _resolve_entry(self.raw["run_evidence"])

    def fold_model_paths(self) -> tuple[Path, ...]:
        return _resolve_entries(self.raw["fold_models"])

    def dataset_paths(self) -> tuple[Path, ...]:
        return _resolve_entries(self.raw["dataset_files"])

    def split_plan_path(self) -> Path:
        return _resolve_entry(self.raw["split_plan"])

    def strategy_path(self) -> Path:
        return _resolve_entry(self.raw["strategy_config"])

    def study_path(self) -> Path:
        return _resolve_entry(self.raw["hyperparameter_study"])

    @property
    def lineage_paths(self) -> tuple[Path, ...]:
        return (
            self.path,
            self.run_path(),
            *self.fold_model_paths(),
            *self.dataset_paths(),
            self.split_plan_path(),
            self.strategy_path(),
            self.study_path(),
        )


class WalkForwardModelSourceCapture:
    """Persist and independently retrain every outer OOS fold."""

    @staticmethod
    def write(
        *,
        output_directory: Path,
        run: WalkForwardModelRun,
        dataset_files: Sequence[Path],
        split_plan: Path,
        strategy_config: Path,
        hyperparameter_study: Path,
    ) -> WalkForwardModelSourceManifest:
        run_path = output_directory / f"run-{run.sha256[:20]}.json"
        fold_paths = tuple(
            output_directory / f"fold-{fold.fold_index:03d}-{fold.model_sha256[:12]}.txt"
            for fold in run.folds
        )
        raw = {
            "schema_version": 1,
            "run_evidence": _entry(run_path),
            "fold_models": _entries(fold_paths),
            "dataset_files": _entries(dataset_files),
            "split_plan": _entry(split_plan),
            "strategy_config": _entry(strategy_config),
            "hyperparameter_study": _entry(hyperparameter_study),
        }
        encoded = json.dumps(raw, sort_keys=True, separators=(",", ":")).encode()
        digest = hashlib.sha256(encoded).hexdigest()
        path = output_directory / f"walkforward-source-{digest[:20]}.json"
        try:
            with path.open("xb") as destination:
                destination.write(encoded)
        except FileExistsError:
            if path.read_bytes() != encoded:
                raise RuntimeError(f"walk-forward source collision at {path}") from None
        manifest = WalkForwardModelSourceManifest.load(path)
        WalkForwardModelSourceCapture.reproduce(manifest)
        return manifest

    @staticmethod
    def find_for_run(run_evidence: Path) -> WalkForwardModelSourceManifest:
        """Resolve exactly one adjacent manifest for an OOS run."""
        target = run_evidence.resolve()
        matches = []
        for path in sorted(target.parent.glob("walkforward-source-*.json")):
            manifest = WalkForwardModelSourceManifest.load(path)
            if manifest.run_path() == target:
                matches.append(manifest)
        if len(matches) != 1:
            raise ValueError("walk-forward run lacks unique retained source lineage")
        return matches[0]

    @staticmethod
    def reproduce(
        manifest: WalkForwardModelSourceManifest,
    ) -> WalkForwardModelRun:
        """Retrain every fold and require exact OOS/model evidence."""
        run = WalkForwardModelRun.load_evidence(manifest.run_path())
        fold_paths = manifest.fold_model_paths()
        if len(fold_paths) != len(run.folds):
            raise ValueError("walk-forward source fold model count differs")
        bound_folds = []
        for fold, path in zip(run.folds, fold_paths, strict=True):
            model_text = path.read_text(encoding="utf-8")
            if hashlib.sha256(model_text.encode()).hexdigest() != fold.model_sha256:
                raise ValueError("walk-forward fold booster differs from run evidence")
            bound_folds.append(replace(fold, model_text=model_text))
        bound = replace(run, folds=tuple(bound_folds))
        strategy = load_strategy_config(manifest.strategy_path())
        plan = WalkForwardPlanner.load(manifest.split_plan_path())
        study = OptunaStudyArtifact.load(manifest.study_path())
        datasets = manifest.dataset_paths()
        study.validate_training_contract(
            dataset_files=datasets,
            plan=plan,
            feature_names=strategy.features,
            label_name=strategy.label.column_name,
            threshold=strategy.label.threshold,
            seed=strategy.seed,
            top_k=strategy.portfolio.top_k,
            requested_trials=strategy.model.hyperparam_search.n_trials,
        )
        if run.hyperparameter_study_sha256 != study.sha256:
            raise ValueError("walk-forward run differs from its Optuna study")
        reproduced = LightgbmWalkForwardTrainer(
            feature_names=strategy.features,
            label_name=strategy.label.column_name,
            threshold=strategy.label.threshold,
            seed=strategy.seed,
            early_stopping_rounds=strategy.model.early_stopping_rounds,
            hyperparameters=study.selected_hyperparameters,
            hyperparameter_study_sha256=study.sha256,
        ).run(
            dataset_files=datasets,
            plan=plan,
        )
        if reproduced != bound:
            raise ValueError("walk-forward run differs from reconstructed fold training")
        return reproduced


def _entries(paths: Iterable[Path]) -> list[dict[str, str]]:
    return [_entry(path) for path in sorted({path.resolve() for path in paths})]


def _entry(path: Path) -> dict[str, str]:
    resolved = path.resolve()
    return {
        "path": resolved.as_posix(),
        "sha256": _file_sha256(resolved),
    }


def _resolve_entries(entries: Sequence[dict[str, str]]) -> tuple[Path, ...]:
    return tuple(_resolve_entry(entry) for entry in entries)


def _resolve_entry(entry: dict[str, str]) -> Path:
    path = Path(entry["path"]).resolve()
    try:
        digest = _file_sha256(path)
    except OSError as error:
        raise ValueError("walk-forward source file is missing or differs") from error
    if digest != entry["sha256"]:
        raise ValueError("walk-forward source file is missing or differs")
    return path


def _validate_entry(entry: object) -> None:
    if (
        not isinstance(entry, dict)
        or set(entry) != {"path", "sha256"}
        or not _is_sha256(entry.get("sha256"))
    ):
        raise ValueError("walk-forward source entry is invalid")
    path = Path(str(entry.get("path", "")))
    if not path.is_absolute() or path.as_posix() != entry["path"]:
        raise ValueError("walk-forward source path is invalid")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(item in "0123456789abcdef" for item in value)
    )
