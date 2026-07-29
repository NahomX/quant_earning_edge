"""Content-addressed reconstruction evidence for a production model."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from quant_earning_edge.backtest import WalkForwardPlanner
from quant_earning_edge.evaluation.phase4_verification import (
    Phase4GateReproduction,
    Phase4GateVerifier,
)
from quant_earning_edge.evaluation.strategy_gate import Phase4PromotionEvidence
from quant_earning_edge.signals.hyperparameter_search import OptunaStudyArtifact
from quant_earning_edge.signals.production_model import (
    ProductionModelArtifact,
    ProductionModelTrainer,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence


@dataclass(frozen=True)
class ProductionModelSourceManifest:
    """Exact research and training files behind one immutable booster."""

    path: Path
    raw: dict[str, Any]

    @classmethod
    def load(cls, path: Path) -> ProductionModelSourceManifest:
        try:
            encoded = path.read_bytes()
            raw = json.loads(encoded)
        except (OSError, ValueError) as error:
            raise ValueError(f"invalid production model source manifest: {path}") from error
        required = {
            "schema_version",
            "model_evidence",
            "model_file",
            "dataset_files",
            "phase4_gate",
            "phase4_aggregation",
            "phase4_source_files",
            "split_plan",
            "hyperparameter_study",
        }
        if not isinstance(raw, dict) or set(raw) != required or raw["schema_version"] != 1:
            raise ValueError("production model source manifest schema mismatch")
        singleton_names = (
            "model_evidence",
            "model_file",
            "phase4_gate",
            "phase4_aggregation",
            "split_plan",
            "hyperparameter_study",
        )
        lists = (raw["dataset_files"], raw["phase4_source_files"])
        if any(not isinstance(raw[name], dict) for name in singleton_names) or any(
            not isinstance(items, list) or not items for items in lists
        ):
            raise ValueError("production model source manifest collections are invalid")
        entries = (
            *(raw[name] for name in singleton_names),
            *raw["dataset_files"],
            *raw["phase4_source_files"],
        )
        for entry in entries:
            _validate_entry(entry)
        identities = tuple(entry["path"] for entry in entries)
        if len(identities) != len(set(identities)):
            raise ValueError("production model source manifest paths are duplicated")
        if any(
            tuple(raw[name]) != tuple(sorted(raw[name], key=lambda item: item["path"]))
            for name in ("dataset_files", "phase4_source_files")
        ):
            raise ValueError("production model source manifest entries are not sorted")
        canonical = json.dumps(raw, sort_keys=True, separators=(",", ":")).encode()
        if canonical != encoded:
            raise ValueError("production model source manifest is not canonical")
        return cls(path=path.resolve(), raw=raw)

    def model_paths(self) -> tuple[Path, Path]:
        return (
            _resolve_entry(self.raw["model_evidence"]),
            _resolve_entry(self.raw["model_file"]),
        )

    def dataset_paths(self) -> tuple[Path, ...]:
        return _resolve_entries(self.raw["dataset_files"])

    def phase4_gate_path(self) -> Path:
        return _resolve_entry(self.raw["phase4_gate"])

    def phase4_aggregation_path(self) -> Path:
        return _resolve_entry(self.raw["phase4_aggregation"])

    def phase4_source_paths(self) -> tuple[Path, ...]:
        return _resolve_entries(self.raw["phase4_source_files"])

    def split_plan_path(self) -> Path:
        return _resolve_entry(self.raw["split_plan"])

    def hyperparameter_study_path(self) -> Path:
        return _resolve_entry(self.raw["hyperparameter_study"])

    @property
    def lineage_paths(self) -> tuple[Path, ...]:
        return (
            self.path,
            *self.model_paths(),
            *self.dataset_paths(),
            self.phase4_gate_path(),
            self.phase4_aggregation_path(),
            *self.phase4_source_paths(),
            self.split_plan_path(),
            self.hyperparameter_study_path(),
        )


class ProductionModelSourceCapture:
    """Write and independently verify a complete production refit lineage."""

    @staticmethod
    def write(
        *,
        output_directory: Path,
        model_evidence: Path,
        model_file: Path,
        dataset_files: Sequence[Path],
        phase4_gate: Path,
        phase4_aggregation: Path,
        split_plan: Path,
        hyperparameter_study: Path,
    ) -> ProductionModelSourceManifest:
        if not dataset_files:
            raise ValueError("production model source requires training datasets")
        reproduction = Phase4GateVerifier.verify(
            report_path=phase4_gate,
            aggregation_spec=phase4_aggregation,
        )
        phase4_sources = reproduction.source_paths[1:]
        raw = {
            "schema_version": 1,
            "model_evidence": _entry(model_evidence),
            "model_file": _entry(model_file),
            "dataset_files": _entries(dataset_files),
            "phase4_gate": _entry(phase4_gate),
            "phase4_aggregation": _entry(phase4_aggregation),
            "phase4_source_files": _entries(phase4_sources),
            "split_plan": _entry(split_plan),
            "hyperparameter_study": _entry(hyperparameter_study),
        }
        encoded = json.dumps(raw, sort_keys=True, separators=(",", ":")).encode()
        digest = hashlib.sha256(encoded).hexdigest()
        output_directory.mkdir(parents=True, exist_ok=True)
        path = output_directory / f"production-source-{digest[:20]}.json"
        try:
            with path.open("xb") as destination:
                destination.write(encoded)
        except FileExistsError:
            if path.read_bytes() != encoded:
                raise RuntimeError(f"production model source collision at {path}") from None
        manifest = ProductionModelSourceManifest.load(path)
        ProductionModelSourceCapture._reproduce_with_phase4(
            manifest,
            reproduction=reproduction,
        )
        return manifest

    @staticmethod
    def reproduce(
        manifest: ProductionModelSourceManifest,
    ) -> ProductionModelArtifact:
        """Replay the gate and refit, requiring the exact retained booster."""
        phase4_gate = manifest.phase4_gate_path()
        reproduction = Phase4GateVerifier.verify(
            report_path=phase4_gate,
            aggregation_spec=manifest.phase4_aggregation_path(),
        )
        return ProductionModelSourceCapture._reproduce_with_phase4(
            manifest,
            reproduction=reproduction,
        )

    @staticmethod
    def _reproduce_with_phase4(
        manifest: ProductionModelSourceManifest,
        *,
        reproduction: Phase4GateReproduction,
    ) -> ProductionModelArtifact:
        model_evidence, model_file = manifest.model_paths()
        model = ProductionModelArtifact.load(
            evidence_path=model_evidence,
            model_path=model_file,
        )
        phase4_gate = manifest.phase4_gate_path()
        if manifest.phase4_source_paths() != tuple(sorted(reproduction.source_paths[1:])):
            raise ValueError("production model Phase 4 lineage is incomplete")
        promotion = Phase4PromotionEvidence.load(phase4_gate)
        strategy = reproduction.strategy
        strategy_path = reproduction.strategy_path
        strategy_sha256 = hashlib.sha256(strategy_path.read_bytes()).hexdigest()
        study = OptunaStudyArtifact.load(manifest.hyperparameter_study_path())
        plan = WalkForwardPlanner.load(manifest.split_plan_path())
        datasets = manifest.dataset_paths()
        if (
            promotion.report_sha256 != model.phase4_gate_sha256
            or promotion.strategy_sha256 != strategy_sha256
            or model.hyperparameter_study_sha256 != study.sha256
            or promotion.hyperparameter_study_sha256 != study.sha256
        ):
            raise ValueError("production model research bindings differ")
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
        reproduced = ProductionModelTrainer(
            feature_names=strategy.features,
            label_name=strategy.label.column_name,
            threshold=strategy.label.threshold,
            seed=strategy.seed,
            early_stopping_rounds=strategy.model.early_stopping_rounds,
            hyperparameters=study.selected_hyperparameters,
        ).run(
            dataset_files=datasets,
            training_cutoff=model.training_cutoff,
            phase4_gate_sha256=promotion.report_sha256,
            hyperparameter_study_sha256=study.sha256,
        )
        if reproduced != model:
            raise ValueError("production model differs from reconstructed training evidence")
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
        raise ValueError("production model source file is missing or differs") from error
    if digest != entry["sha256"]:
        raise ValueError("production model source file is missing or differs")
    return path


def _validate_entry(entry: object) -> None:
    if (
        not isinstance(entry, dict)
        or set(entry) != {"path", "sha256"}
        or not _is_sha256(entry.get("sha256"))
    ):
        raise ValueError("production model source entry is invalid")
    path = Path(str(entry.get("path", "")))
    if not path.is_absolute() or path.as_posix() != entry["path"]:
        raise ValueError("production model source path is invalid")


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
