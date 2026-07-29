"""Post-completion Phase 6 reevaluation with content-addressed controls."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

from quant_earning_edge.data import SessionFileStore
from quant_earning_edge.evaluation.phase6_controls import (
    Phase6ControlBuilder,
    encode_phase6_controls,
    write_phase6_controls,
)
from quant_earning_edge.evaluation.phase6_gate import (
    Phase6AggregationSpec,
    Phase6GateEvaluator,
    Phase6GateReport,
)
from quant_earning_edge.evaluation.phase6_sources import Phase6DailyReportVerifier
from quant_earning_edge.orchestration.health import WorkflowHealthEvaluator

if TYPE_CHECKING:
    from datetime import date
    from pathlib import Path

    from quant_earning_edge.orchestration.workflow import DailyWorkflowStore


@dataclass(frozen=True)
class Phase6FinalizationArtifacts:
    """Fresh post-completion health, aggregation, and gate evidence."""

    health_path: Path
    aggregation_path: Path
    gate_report_path: Path
    manifest_path: Path
    report: Phase6GateReport


@dataclass(frozen=True)
class Phase6FinalizationEvidence:
    """Canonical linkage from a complete workflow state to refreshed verdict."""

    schema_version: int
    trade_date: date
    workflow_state_sha256: str
    health_path: str
    health_sha256: str
    aggregation_path: str
    aggregation_sha256: str
    gate_report_path: str
    gate_report_sha256: str
    verdict: str
    passes_phase6_gate: bool

    @property
    def canonical_bytes(self) -> bytes:
        return json.dumps(
            asdict(self),
            default=lambda item: item.isoformat(),
            sort_keys=True,
            separators=(",", ":"),
        ).encode()

    def write(self, output: Path) -> None:
        encoded = self.canonical_bytes
        output.parent.mkdir(parents=True, exist_ok=True)
        try:
            with output.open("xb") as destination:
                destination.write(encoded)
        except FileExistsError:
            if output.read_bytes() != encoded:
                raise RuntimeError(f"Phase 6 finalization collision at {output}") from None


class Phase6CompletionFinalizer:
    """Reevaluate Phase 6 only after the current workflow is durably complete."""

    def finalize(
        self,
        *,
        original_aggregation_spec: Path,
        current_trade_date: date,
        artifact_root: Path,
        output_directory: Path,
        workflow_store: DailyWorkflowStore,
    ) -> Phase6FinalizationArtifacts:
        original = Phase6AggregationSpec.model_validate_json(original_aggregation_spec.read_bytes())
        configured_workflow_root = _resolve(
            original.workflow_store_root,
            relative_to=original_aggregation_spec.parent,
        )
        if configured_workflow_root != workflow_store.root:
            raise ValueError(
                "Phase 6 aggregation workflow store differs from the active workflow store"
            )
        session_path = _resolve(
            original.session_file,
            relative_to=original_aggregation_spec.parent,
        )
        calendar = SessionFileStore.load(session_path)
        state = workflow_store.load_latest(current_trade_date)
        if state is None or not state.complete:
            raise ValueError("Phase 6 finalization requires a complete current workflow")
        state.verify_artifacts()

        health = WorkflowHealthEvaluator().evaluate(
            calendar=calendar,
            store=workflow_store,
            start_date=original.proof_start,
            end_date=original.proof_end,
        )
        resolved_output = output_directory.resolve()
        health_path = resolved_output / f"workflow-health-{health.sha256}.json"
        controls = Phase6ControlBuilder().prepare(
            calendar=calendar,
            session_file=session_path,
            workflow_store_root=workflow_store.root,
            workflow_health=health,
            proof_start=original.proof_start,
            proof_end=original.proof_end,
            current_trade_date=current_trade_date,
            initial_cash=original.initial_cash,
            artifact_root=artifact_root,
            health_output=health_path,
            bootstrap_resamples=original.bootstrap_resamples,
            seed=original.seed,
        )
        aggregation_bytes = encode_phase6_controls(controls.aggregation_spec)
        aggregation_sha256 = hashlib.sha256(aggregation_bytes).hexdigest()
        aggregation_path = resolved_output / f"phase6-controls-{aggregation_sha256}.json"
        reports = tuple(
            Phase6DailyReportVerifier().verify(path, workflow_store=workflow_store)
            for path in controls.included_report_files
        )
        report = Phase6GateEvaluator(
            bootstrap_resamples=original.bootstrap_resamples,
            seed=original.seed,
        ).evaluate(
            calendar=calendar,
            workflow_health=health,
            reports=reports,
            proof_start=original.proof_start,
            proof_end=original.proof_end,
            initial_cash=original.initial_cash,
        )
        gate_path = resolved_output / f"phase6-gate-{report.sha256}.json"
        health.write(health_path)
        write_phase6_controls(controls.aggregation_spec, aggregation_path)
        report.write(gate_path)
        manifest_path = resolved_output / f"finalization-{state.sha256}.json"
        Phase6FinalizationEvidence(
            schema_version=1,
            trade_date=current_trade_date,
            workflow_state_sha256=state.sha256,
            health_path=str(health_path),
            health_sha256=health.sha256,
            aggregation_path=str(aggregation_path),
            aggregation_sha256=aggregation_sha256,
            gate_report_path=str(gate_path),
            gate_report_sha256=report.sha256,
            verdict=report.verdict,
            passes_phase6_gate=report.passes_phase6_gate,
        ).write(manifest_path)
        return Phase6FinalizationArtifacts(
            health_path=health_path,
            aggregation_path=aggregation_path,
            gate_report_path=gate_path,
            manifest_path=manifest_path,
            report=report,
        )


def _resolve(path: Path, *, relative_to: Path) -> Path:
    return path.resolve() if path.is_absolute() else (relative_to / path).resolve()
