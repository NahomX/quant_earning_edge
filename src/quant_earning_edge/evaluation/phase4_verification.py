"""Independent reconstruction of the complete Phase 4 research gate."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING

from pydantic import ValidationError

from quant_earning_edge.backtest import CostModel
from quant_earning_edge.evaluation.phase4_assembly import (
    Phase4AssemblyManifest,
    Phase4HistoricalAssembler,
)
from quant_earning_edge.evaluation.strategy_gate import (
    FoldBacktestResults,
    Phase4AggregationSpec,
    Phase4GateEvaluation,
    Phase4GateEvaluator,
)
from quant_earning_edge.signals.config import load_strategy_config
from quant_earning_edge.signals.event_trades import EventTradePlanner, run_event_plan
from quant_earning_edge.signals.lgbm_model import OosPrediction, WalkForwardModelRun

if TYPE_CHECKING:
    from quant_earning_edge.signals.config import EarningsStrategyConfig
    from quant_earning_edge.signals.event_trades import TradeCohort


@dataclass(frozen=True)
class Phase4GateReproduction:
    """A freshly rebuilt report plus every source path used to produce it."""

    report: Phase4GateEvaluation
    strategy: EarningsStrategyConfig
    aggregation_spec_path: Path
    assembly_manifest_path: Path
    walkforward_run_path: Path
    plan_paths: tuple[Path, ...]

    @property
    def strategy_path(self) -> Path:
        assembly = Phase4AssemblyManifest.load(self.assembly_manifest_path)
        return assembly.resolved_strategy_config(self.assembly_manifest_path)

    @property
    def source_paths(self) -> tuple[Path, ...]:
        assembly = Phase4AssemblyManifest.load(self.assembly_manifest_path)
        return (
            self.aggregation_spec_path,
            self.assembly_manifest_path,
            self.strategy_path,
            self.walkforward_run_path,
            assembly.resolved_session_file(self.assembly_manifest_path),
            *assembly.resolved_candidate_files(self.assembly_manifest_path),
            *assembly.resolved_daily_bar_files(self.assembly_manifest_path),
            *self.plan_paths,
        )


class Phase4GateVerifier:
    """Rebuild a Phase 4 report instead of trusting its serialized metrics."""

    @staticmethod
    def reproduce(
        aggregation_spec: Path,
        *,
        bootstrap_resamples: int,
        seed: int,
    ) -> Phase4GateReproduction:
        """Rebuild plans, ledgers, metrics, cohorts, and gate decisions."""
        aggregation_path = aggregation_spec.resolve()
        try:
            aggregation_bytes = aggregation_path.read_bytes()
            spec = Phase4AggregationSpec.model_validate_json(aggregation_bytes)
        except (OSError, ValidationError, ValueError) as error:
            raise ValueError(f"invalid Phase 4 aggregation spec: {aggregation_path}") from error
        if (
            json.dumps(
                spec.model_dump(mode="json"),
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
            != aggregation_bytes
        ):
            raise ValueError("Phase 4 aggregation spec is not canonical")
        manifest_path = _resolve(spec.assembly_manifest, base=aggregation_path.parent)
        assembly = Phase4AssemblyManifest.load(manifest_path)
        strategy = load_strategy_config(assembly.resolved_strategy_config(manifest_path))
        model_run_path = _resolve(
            spec.walkforward_run_evidence,
            base=aggregation_path.parent,
        )
        model_run = WalkForwardModelRun.load_evidence(model_run_path)
        if assembly.walkforward_run_evidence.sha256 != model_run.sha256:
            raise ValueError("Phase 4 assembly manifest differs from the walk-forward run")
        if (
            model_run.feature_names != strategy.features
            or model_run.label_name != strategy.label.column_name
            or model_run.threshold != strategy.label.threshold
            or model_run.seed != strategy.seed
        ):
            raise ValueError("Phase 4 strategy differs from the walk-forward run")
        if model_run.hyperparameter_study_sha256 is None:
            raise ValueError("walk-forward run lacks an Optuna study binding")
        _verify_plan_reproduction(
            manifest=assembly,
            manifest_path=manifest_path,
        )

        cost_model = CostModel(strategy.cost_model_config)
        fold_results: list[FoldBacktestResults] = []
        plan_paths: list[Path] = []
        all_predictions: list[OosPrediction] = []
        for fold in spec.folds:
            results = []
            cohorts: list[TradeCohort] = []
            for configured_path in fold.event_plan_files:
                plan_path = _resolve(configured_path, base=aggregation_path.parent)
                plan_paths.append(plan_path)
                plan = EventTradePlanner.load(plan_path)
                if plan.walkforward_run_sha256 != model_run.sha256:
                    raise ValueError("event plan differs from the Phase 4 walk-forward run")
                model_run.validate_predictions(plan.source_predictions)
                all_predictions.extend(plan.source_predictions)
                cohorts.extend(plan.cohorts)
                results.append(run_event_plan(plan, cost_model=cost_model))
            fold_results.append(
                FoldBacktestResults(
                    fold_index=fold.fold_index,
                    test_start_date=fold.test_start_date,
                    test_end_date=fold.test_end_date,
                    results=tuple(results),
                    cohorts=tuple(cohorts),
                )
            )
        if tuple(path.resolve() for path in plan_paths) != assembly.resolved_plan_files(
            manifest_path
        ):
            raise ValueError("Phase 4 aggregation plan files differ from the assembly manifest")
        model_run.validate_predictions(tuple(all_predictions), require_complete=True)
        report = Phase4GateEvaluator(
            bootstrap_resamples=bootstrap_resamples,
            seed=seed,
        ).evaluate(
            tuple(fold_results),
            strategy_sha256=assembly.strategy_config.sha256,
            assembly_manifest_sha256=assembly.sha256,
            walkforward_run_sha256=model_run.sha256,
            hyperparameter_study_sha256=model_run.hyperparameter_study_sha256,
        )
        return Phase4GateReproduction(
            report=report,
            strategy=strategy,
            aggregation_spec_path=aggregation_path,
            assembly_manifest_path=manifest_path,
            walkforward_run_path=model_run_path,
            plan_paths=tuple(plan_paths),
        )

    @staticmethod
    def verify(
        *,
        report_path: Path,
        aggregation_spec: Path,
    ) -> Phase4GateReproduction:
        """Require an existing passing report to match fresh reconstruction."""
        try:
            encoded = report_path.read_bytes()
            raw = json.loads(encoded)
            bootstrap = raw["overall"]["bootstrap"]
            bootstrap_resamples = int(bootstrap["resamples"])
            bootstrap_seed = int(bootstrap["seed"])
        except (KeyError, OSError, TypeError, ValueError) as error:
            raise ValueError(f"invalid Phase 4 gate report: {report_path}") from error
        reproduction = Phase4GateVerifier.reproduce(
            aggregation_spec,
            bootstrap_resamples=bootstrap_resamples,
            seed=bootstrap_seed,
        )
        if reproduction.report.to_json_bytes() != encoded:
            raise ValueError("Phase 4 gate report differs from reconstructed OOS evidence")
        return reproduction


def _verify_plan_reproduction(
    *,
    manifest: Phase4AssemblyManifest,
    manifest_path: Path,
) -> None:
    with TemporaryDirectory(prefix="qee-phase4-reproduction-") as temporary:
        reproduction_root = Path(temporary)
        reproduced = Phase4HistoricalAssembler().assemble(
            walkforward_run_evidence=manifest.resolved_walkforward_run(manifest_path),
            strategy_config=manifest.resolved_strategy_config(manifest_path),
            session_file=manifest.resolved_session_file(manifest_path),
            candidate_files=manifest.resolved_candidate_files(manifest_path),
            daily_bar_files=manifest.resolved_daily_bar_files(manifest_path),
            initial_cash=manifest.initial_cash,
            output_dir=reproduction_root / "plans",
            manifest_output=reproduction_root / "manifest.json",
            aggregation_output=reproduction_root / "aggregation.json",
            minimum_probability=manifest.minimum_probability,
        )
        reproduced_bytes = tuple(path.read_bytes() for path in reproduced.plan_files)
        bound_bytes = tuple(
            path.read_bytes() for path in manifest.resolved_plan_files(manifest_path)
        )
        if reproduced_bytes != bound_bytes:
            raise ValueError("Phase 4 plans do not reproduce from the assembly manifest sources")


def _resolve(path: Path, *, base: Path) -> Path:
    return (path if path.is_absolute() else base / path).resolve()
