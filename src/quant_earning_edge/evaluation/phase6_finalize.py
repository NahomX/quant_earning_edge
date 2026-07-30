"""Post-completion Phase 6 reevaluation with content-addressed controls."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
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

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("unsupported Phase 6 finalization schema")
        if self.verdict not in {"pass", "fail", "insufficient-evidence"}:
            raise ValueError("invalid Phase 6 finalization verdict")
        if self.passes_phase6_gate != (self.verdict == "pass"):
            raise ValueError("Phase 6 finalization verdict is inconsistent")
        for digest in (
            self.workflow_state_sha256,
            self.health_sha256,
            self.aggregation_sha256,
            self.gate_report_sha256,
        ):
            if len(digest) != 64 or any(item not in "0123456789abcdef" for item in digest):
                raise ValueError("invalid Phase 6 finalization SHA-256")
        for value in (self.health_path, self.aggregation_path, self.gate_report_path):
            if not value.strip():
                raise ValueError("Phase 6 finalization path must not be blank")

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

    @classmethod
    def load(cls, path: Path) -> Phase6FinalizationEvidence:
        """Strictly reload one canonical finalization linkage."""
        fields = {
            "schema_version",
            "trade_date",
            "workflow_state_sha256",
            "health_path",
            "health_sha256",
            "aggregation_path",
            "aggregation_sha256",
            "gate_report_path",
            "gate_report_sha256",
            "verdict",
            "passes_phase6_gate",
        }
        try:
            raw = json.loads(path.read_bytes())
            if not isinstance(raw, dict) or set(raw) != fields:
                raise ValueError("Phase 6 finalization fields are invalid")
            if not isinstance(raw["passes_phase6_gate"], bool):
                raise ValueError("Phase 6 finalization pass flag is invalid")
            evidence = cls(
                schema_version=int(raw["schema_version"]),
                trade_date=date.fromisoformat(str(raw["trade_date"])),
                workflow_state_sha256=str(raw["workflow_state_sha256"]),
                health_path=str(raw["health_path"]),
                health_sha256=str(raw["health_sha256"]),
                aggregation_path=str(raw["aggregation_path"]),
                aggregation_sha256=str(raw["aggregation_sha256"]),
                gate_report_path=str(raw["gate_report_path"]),
                gate_report_sha256=str(raw["gate_report_sha256"]),
                verdict=str(raw["verdict"]),
                passes_phase6_gate=raw["passes_phase6_gate"],
            )
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid Phase 6 finalization evidence: {path}") from error
        if json.loads(evidence.canonical_bytes) != raw:
            raise ValueError("Phase 6 finalization evidence is not canonical")
        return evidence


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

    def verify(
        self,
        *,
        manifest_path: Path,
        artifact_root: Path,
        output_directory: Path,
        workflow_store: DailyWorkflowStore,
        expected_state_sha256: str,
    ) -> Phase6FinalizationArtifacts:
        """Reproduce every artifact linked by a post-completion marker."""
        evidence = Phase6FinalizationEvidence.load(manifest_path)
        if evidence.workflow_state_sha256 != expected_state_sha256:
            raise ValueError("finalization workflow state differs from expected state")
        resolved_output = output_directory.resolve()
        if manifest_path.resolve() != (
            resolved_output / f"finalization-{expected_state_sha256}.json"
        ):
            raise ValueError("finalization manifest path is not content-addressed")
        health_path = _linked_artifact(
            evidence.health_path,
            digest=evidence.health_sha256,
            output_directory=resolved_output,
            expected_name=f"workflow-health-{evidence.health_sha256}.json",
        )
        aggregation_path = _linked_artifact(
            evidence.aggregation_path,
            digest=evidence.aggregation_sha256,
            output_directory=resolved_output,
            expected_name=f"phase6-controls-{evidence.aggregation_sha256}.json",
        )
        gate_path = _linked_artifact(
            evidence.gate_report_path,
            digest=evidence.gate_report_sha256,
            output_directory=resolved_output,
            expected_name=f"phase6-gate-{evidence.gate_report_sha256}.json",
        )
        state = workflow_store.load_latest(evidence.trade_date)
        if state is None or not state.complete or state.sha256 != expected_state_sha256:
            raise ValueError("finalization does not bind the complete latest workflow state")
        state.verify_artifacts()
        aggregation_bytes = aggregation_path.read_bytes()
        aggregation = Phase6AggregationSpec.model_validate_json(aggregation_bytes)
        if encode_phase6_controls(aggregation) != aggregation_bytes:
            raise ValueError("finalization aggregation controls are not canonical")
        workflow_root = _resolve(
            aggregation.workflow_store_root,
            relative_to=aggregation_path.parent,
        )
        if workflow_root != workflow_store.root:
            raise ValueError("finalization aggregation binds a different workflow store")
        session_path = _resolve(
            aggregation.session_file,
            relative_to=aggregation_path.parent,
        )
        calendar = SessionFileStore.load(session_path)
        health = WorkflowHealthEvaluator().evaluate(
            calendar=calendar,
            store=workflow_store,
            start_date=aggregation.proof_start,
            end_date=aggregation.proof_end,
        )
        if health.canonical_bytes != health_path.read_bytes():
            raise ValueError("finalization health does not reproduce from workflow state")
        controls = Phase6ControlBuilder().prepare(
            calendar=calendar,
            session_file=session_path,
            workflow_store_root=workflow_store.root,
            workflow_health=health,
            proof_start=aggregation.proof_start,
            proof_end=aggregation.proof_end,
            current_trade_date=evidence.trade_date,
            initial_cash=aggregation.initial_cash,
            artifact_root=artifact_root,
            health_output=health_path,
            bootstrap_resamples=aggregation.bootstrap_resamples,
            seed=aggregation.seed,
        )
        if encode_phase6_controls(controls.aggregation_spec) != aggregation_bytes:
            raise ValueError("finalization aggregation controls do not reproduce")
        reports = tuple(
            Phase6DailyReportVerifier().verify(path, workflow_store=workflow_store)
            for path in controls.included_report_files
        )
        report = Phase6GateEvaluator(
            bootstrap_resamples=aggregation.bootstrap_resamples,
            seed=aggregation.seed,
        ).evaluate(
            calendar=calendar,
            workflow_health=health,
            reports=reports,
            proof_start=aggregation.proof_start,
            proof_end=aggregation.proof_end,
            initial_cash=aggregation.initial_cash,
        )
        if report.canonical_bytes != gate_path.read_bytes():
            raise ValueError("finalization gate verdict does not reproduce")
        if (
            evidence.verdict != report.verdict
            or evidence.passes_phase6_gate != report.passes_phase6_gate
        ):
            raise ValueError("finalization manifest verdict differs from reproduced gate")
        return Phase6FinalizationArtifacts(
            health_path=health_path,
            aggregation_path=aggregation_path,
            gate_report_path=gate_path,
            manifest_path=manifest_path.resolve(),
            report=report,
        )


def _resolve(path: Path, *, relative_to: Path) -> Path:
    return path.resolve() if path.is_absolute() else (relative_to / path).resolve()


def _linked_artifact(
    value: str,
    *,
    digest: str,
    output_directory: Path,
    expected_name: str,
) -> Path:
    path = Path(value).resolve()
    if (
        path.parent != output_directory
        or path.name != expected_name
        or not path.is_file()
        or hashlib.sha256(path.read_bytes()).hexdigest() != digest
    ):
        raise ValueError("finalization linked artifact is missing or differs")
    return path
