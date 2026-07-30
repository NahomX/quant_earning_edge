"""Content-addressed reconstruction evidence for nested Optuna selection."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from quant_earning_edge.backtest import WalkForwardPlanner
from quant_earning_edge.labels.dataset_source import (
    TrainingDatasetSourceCapture,
    TrainingDatasetSourceManifest,
)
from quant_earning_edge.signals.config import load_strategy_config
from quant_earning_edge.signals.hyperparameter_search import (
    OptunaLightgbmSearch,
    OptunaStudyArtifact,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence


@dataclass(frozen=True)
class OptunaStudySourceManifest:
    """Exact datasets and nested-validation contract behind one study."""

    path: Path
    raw: dict[str, Any]

    @classmethod
    def load(cls, path: Path) -> OptunaStudySourceManifest:
        try:
            encoded = path.read_bytes()
            raw = json.loads(encoded)
        except (OSError, ValueError) as error:
            raise ValueError(f"invalid Optuna source manifest: {path}") from error
        required = {
            "schema_version",
            "study_artifact",
            "dataset_files",
            "split_plan",
            "strategy_config",
            "dataset_source_manifests",
        }
        if not isinstance(raw, dict) or set(raw) != required or raw["schema_version"] != 2:
            raise ValueError("Optuna source manifest schema mismatch")
        if (
            not isinstance(raw["study_artifact"], dict)
            or not isinstance(raw["dataset_files"], list)
            or not raw["dataset_files"]
            or not isinstance(raw["split_plan"], dict)
            or not isinstance(raw["strategy_config"], dict)
            or not isinstance(raw["dataset_source_manifests"], list)
            or not raw["dataset_source_manifests"]
        ):
            raise ValueError("Optuna source manifest collections are invalid")
        entries = (
            raw["study_artifact"],
            *raw["dataset_files"],
            raw["split_plan"],
            raw["strategy_config"],
            *raw["dataset_source_manifests"],
        )
        for entry in entries:
            _validate_entry(entry)
        identities = tuple(entry["path"] for entry in entries)
        if len(identities) != len(set(identities)):
            raise ValueError("Optuna source manifest paths are duplicated")
        if tuple(raw["dataset_files"]) != tuple(
            sorted(raw["dataset_files"], key=lambda item: item["path"])
        ) or tuple(raw["dataset_source_manifests"]) != tuple(
            sorted(raw["dataset_source_manifests"], key=lambda item: item["path"])
        ):
            raise ValueError("Optuna source manifest datasets are not sorted")
        if json.dumps(raw, sort_keys=True, separators=(",", ":")).encode() != encoded:
            raise ValueError("Optuna source manifest is not canonical")
        return cls(path=path.resolve(), raw=raw)

    def study_path(self) -> Path:
        return _resolve_entry(self.raw["study_artifact"])

    def dataset_paths(self) -> tuple[Path, ...]:
        return _resolve_entries(self.raw["dataset_files"])

    def split_plan_path(self) -> Path:
        return _resolve_entry(self.raw["split_plan"])

    def strategy_path(self) -> Path:
        return _resolve_entry(self.raw["strategy_config"])

    def dataset_sources(self) -> tuple[TrainingDatasetSourceManifest, ...]:
        return tuple(
            TrainingDatasetSourceManifest.load(path)
            for path in _resolve_entries(self.raw["dataset_source_manifests"])
        )

    @property
    def lineage_paths(self) -> tuple[Path, ...]:
        return (
            self.path,
            self.study_path(),
            *self.dataset_paths(),
            self.split_plan_path(),
            self.strategy_path(),
            *(path for source in self.dataset_sources() for path in source.lineage_paths),
        )


class OptunaStudySourceCapture:
    """Persist and independently rerun a nested hyperparameter search."""

    @staticmethod
    def write(
        *,
        study_artifact: Path,
        dataset_files: Sequence[Path],
        split_plan: Path,
        strategy_config: Path,
    ) -> OptunaStudySourceManifest:
        study = OptunaStudyArtifact.load(study_artifact)
        strategy = load_strategy_config(strategy_config)
        plan = WalkForwardPlanner.load(split_plan)
        study.validate_training_contract(
            dataset_files=dataset_files,
            plan=plan,
            feature_names=strategy.features,
            label_name=strategy.label.column_name,
            threshold=strategy.label.threshold,
            seed=strategy.seed,
            top_k=strategy.portfolio.top_k,
            requested_trials=strategy.model.hyperparam_search.n_trials,
        )
        dataset_sources = TrainingDatasetSourceCapture.find_for_datasets(dataset_files)
        raw = {
            "schema_version": 2,
            "study_artifact": _entry(study_artifact),
            "dataset_files": _entries(dataset_files),
            "split_plan": _entry(split_plan),
            "strategy_config": _entry(strategy_config),
            "dataset_source_manifests": _entries(source.path for source in dataset_sources),
        }
        encoded = json.dumps(raw, sort_keys=True, separators=(",", ":")).encode()
        digest = hashlib.sha256(encoded).hexdigest()
        path = study_artifact.resolve().parent / f"optuna-source-{digest[:20]}.json"
        try:
            with path.open("xb") as destination:
                destination.write(encoded)
        except FileExistsError:
            if path.read_bytes() != encoded:
                raise RuntimeError(f"Optuna source collision at {path}") from None
        return OptunaStudySourceManifest.load(path)

    @staticmethod
    def find_for_study(study_artifact: Path) -> OptunaStudySourceManifest:
        """Resolve exactly one adjacent manifest for an Optuna artifact."""
        target = study_artifact.resolve()
        matches = []
        for path in sorted(target.parent.glob("optuna-source-*.json")):
            manifest = OptunaStudySourceManifest.load(path)
            if manifest.study_path() == target:
                matches.append(manifest)
        if len(matches) != 1:
            raise ValueError("Optuna study lacks unique retained source lineage")
        return matches[0]

    @staticmethod
    def reproduce(
        manifest: OptunaStudySourceManifest,
    ) -> OptunaStudyArtifact:
        """Rerun the deterministic nested search and require exact evidence."""
        expected = OptunaStudyArtifact.load(manifest.study_path())
        strategy = load_strategy_config(manifest.strategy_path())
        plan = WalkForwardPlanner.load(manifest.split_plan_path())
        datasets = manifest.dataset_paths()
        dataset_sources = manifest.dataset_sources()
        if tuple(source.dataset_path() for source in dataset_sources) != datasets:
            raise ValueError("Optuna training-dataset lineage differs")
        for source in dataset_sources:
            TrainingDatasetSourceCapture.reproduce(source)
        expected.validate_training_contract(
            dataset_files=datasets,
            plan=plan,
            feature_names=strategy.features,
            label_name=strategy.label.column_name,
            threshold=strategy.label.threshold,
            seed=strategy.seed,
            top_k=strategy.portfolio.top_k,
            requested_trials=strategy.model.hyperparam_search.n_trials,
        )
        reproduced = OptunaLightgbmSearch(
            feature_names=strategy.features,
            label_name=strategy.label.column_name,
            threshold=strategy.label.threshold,
            seed=strategy.seed,
            early_stopping_rounds=strategy.model.early_stopping_rounds,
            top_k=strategy.portfolio.top_k,
            requested_trials=strategy.model.hyperparam_search.n_trials,
        ).run(
            dataset_files=datasets,
            plan=plan,
        )
        if reproduced != expected:
            raise ValueError("Optuna study differs from reconstructed nested search")
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
        raise ValueError("Optuna source file is missing or differs") from error
    if digest != entry["sha256"]:
        raise ValueError("Optuna source file is missing or differs")
    return path


def _validate_entry(entry: object) -> None:
    if (
        not isinstance(entry, dict)
        or set(entry) != {"path", "sha256"}
        or not _is_sha256(entry.get("sha256"))
    ):
        raise ValueError("Optuna source entry is invalid")
    path = Path(str(entry.get("path", "")))
    if not path.is_absolute() or path.as_posix() != entry["path"]:
        raise ValueError("Optuna source path is invalid")


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
