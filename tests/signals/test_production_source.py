"""Production-model source-manifest reconstruction tests."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from quant_earning_edge.backtest import WalkForwardPlanner
from quant_earning_edge.evaluation.phase4_verification import Phase4GateVerifier
from quant_earning_edge.evaluation.strategy_gate import Phase4PromotionEvidence
from quant_earning_edge.signals import (
    LightgbmHyperparameters,
    OptunaStudyArtifact,
    ProductionModelArtifact,
    ProductionModelTrainer,
    load_strategy_config,
)
from quant_earning_edge.signals.production_source import (
    ProductionModelSourceCapture,
    ProductionModelSourceManifest,
)

if TYPE_CHECKING:
    from pytest import MonkeyPatch


@dataclass
class _Study:
    sha256: str
    selected_hyperparameters: LightgbmHyperparameters

    def validate_training_contract(self, **_: Any) -> None:
        return None


def _fixture(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> tuple[ProductionModelSourceManifest, ProductionModelArtifact, Path]:
    strategy_path = Path("configs/strategies/earnings_v1.yaml").resolve()
    strategy = load_strategy_config(strategy_path)
    hyperparameters = LightgbmHyperparameters()
    model = ProductionModelArtifact(
        schema_version=2,
        training_cutoff=date(2026, 7, 27),
        phase4_gate_sha256="f" * 64,
        dataset_sha256=("d" * 64,),
        feature_names=strategy.features,
        label_name=strategy.label.column_name,
        threshold=strategy.label.threshold,
        seed=strategy.seed,
        hyperparameter_study_sha256="e" * 64,
        hyperparameters=hyperparameters,
        hyperparameters_sha256=hyperparameters.sha256,
        lightgbm_version="test",
        fit_start_date=date(2024, 1, 2),
        fit_end_date=date(2025, 12, 1),
        validation_start_date=date(2026, 1, 2),
        validation_end_date=date(2026, 7, 24),
        fit_count=100,
        validation_count=20,
        best_iteration=10,
        model_sha256=hashlib.sha256(b"model").hexdigest(),
        model_text="model",
    )
    model_path, evidence_path = ProductionModelTrainer.write(model, tmp_path / "models")
    dataset = tmp_path / "training.parquet"
    dataset.write_bytes(b"dataset")
    phase4_gate = tmp_path / "phase4-gate.json"
    phase4_gate.write_bytes(b"gate")
    aggregation = tmp_path / "aggregation.json"
    aggregation.write_bytes(b"aggregation")
    assembly = tmp_path / "assembly.json"
    assembly.write_bytes(b"assembly")
    split_plan = tmp_path / "split.json"
    split_plan.write_bytes(b"split")
    study_path = tmp_path / "study.json"
    study_path.write_bytes(b"study")
    reproduction = type(
        "Reproduction",
        (),
        {
            "strategy": strategy,
            "strategy_path": strategy_path,
            "source_paths": (aggregation.resolve(), assembly.resolve(), strategy_path),
        },
    )()
    promotion = type(
        "Promotion",
        (),
        {
            "report_sha256": model.phase4_gate_sha256,
            "strategy_sha256": hashlib.sha256(strategy_path.read_bytes()).hexdigest(),
            "hyperparameter_study_sha256": model.hyperparameter_study_sha256,
        },
    )()
    study = _Study(
        sha256=model.hyperparameter_study_sha256 or "",
        selected_hyperparameters=hyperparameters,
    )
    monkeypatch.setattr(
        Phase4GateVerifier,
        "verify",
        staticmethod(lambda **_: reproduction),
    )
    monkeypatch.setattr(
        Phase4PromotionEvidence,
        "load",
        classmethod(lambda cls, path: promotion),
    )
    monkeypatch.setattr(
        OptunaStudyArtifact,
        "load",
        classmethod(lambda cls, path: study),
    )
    monkeypatch.setattr(
        WalkForwardPlanner,
        "load",
        staticmethod(lambda path: object()),
    )
    monkeypatch.setattr(
        ProductionModelTrainer,
        "run",
        lambda self, **kwargs: model,
    )
    manifest = ProductionModelSourceCapture.write(
        output_directory=tmp_path / "models",
        model_evidence=evidence_path,
        model_file=model_path,
        dataset_files=(dataset,),
        phase4_gate=phase4_gate,
        phase4_aggregation=aggregation,
        split_plan=split_plan,
        hyperparameter_study=study_path,
    )
    return manifest, model, dataset


def test_production_model_reproduces_from_complete_training_lineage(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    manifest, model, _ = _fixture(tmp_path, monkeypatch)

    reproduced = ProductionModelSourceCapture.reproduce(manifest)
    model_evidence, model_file = manifest.model_paths()
    discovered = ProductionModelSourceCapture.find_for_model(
        model_evidence=model_evidence,
        model_file=model_file,
    )

    assert reproduced == model
    assert discovered == manifest
    assert manifest.lineage_paths[0] == manifest.path


def test_production_model_source_rejects_changed_training_dataset(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    manifest, _, dataset = _fixture(tmp_path, monkeypatch)
    dataset.write_bytes(b"changed")

    with pytest.raises(ValueError, match="missing or differs"):
        ProductionModelSourceCapture.reproduce(manifest)
