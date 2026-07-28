"""Prepare rolling Phase 6 health and aggregation inputs without path handwork."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING

from quant_earning_edge.evaluation.phase6_gate import Phase6AggregationSpec
from quant_earning_edge.evaluation.replay_session import ReplaySessionReport
from quant_earning_edge.orchestration.health import WorkflowHealthEvaluator

if TYPE_CHECKING:
    from datetime import date
    from pathlib import Path

    from quant_earning_edge.data.calendar import SessionFile
    from quant_earning_edge.orchestration.health import WorkflowHealthReport
    from quant_earning_edge.orchestration.workflow import DailyWorkflowStore


@dataclass(frozen=True)
class Phase6ControlArtifacts:
    """Prepared immutable files consumed by the daily terminal workflow stage."""

    aggregation_spec: Phase6AggregationSpec
    health_sha256: str
    included_report_files: tuple[Path, ...]


class Phase6ControlBuilder:
    """Bind calendar, workflow health, and available deterministic daily reports."""

    def prepare(
        self,
        *,
        calendar: SessionFile,
        session_file: Path,
        workflow_health: WorkflowHealthReport,
        proof_start: date,
        proof_end: date,
        current_trade_date: date,
        initial_cash: float,
        artifact_root: Path,
        health_output: Path,
        bootstrap_resamples: int = 10_000,
        seed: int = 20260427,
    ) -> Phase6ControlArtifacts:
        session_dates = tuple(
            item.session_date
            for item in calendar.sessions
            if proof_start <= item.session_date <= proof_end
        )
        if current_trade_date not in session_dates:
            raise ValueError("current trade date is outside the authoritative proof window")
        report_files: list[Path] = []
        for session_date in session_dates:
            path = (
                artifact_root.resolve()
                / f"trade_date={session_date.isoformat()}"
                / "replay-session.json"
            )
            if path.is_file():
                report = ReplaySessionReport.load(path)
                if report.session_date != session_date:
                    raise ValueError(f"daily replay report date differs from path: {path}")
                report_files.append(path)
            elif session_date == current_trade_date:
                report_files.append(path)
        spec = Phase6AggregationSpec(
            session_file=session_file.resolve(),
            workflow_health_file=health_output.resolve(),
            proof_start=proof_start,
            proof_end=proof_end,
            initial_cash=initial_cash,
            session_report_files=tuple(report_files),
            bootstrap_resamples=bootstrap_resamples,
            seed=seed,
        )
        return Phase6ControlArtifacts(
            aggregation_spec=spec,
            health_sha256=workflow_health.sha256,
            included_report_files=tuple(report_files),
        )

    def build(
        self,
        *,
        calendar: SessionFile,
        session_file: Path,
        workflow_store: DailyWorkflowStore,
        proof_start: date,
        proof_end: date,
        current_trade_date: date,
        initial_cash: float,
        artifact_root: Path,
        health_output: Path,
        aggregation_output: Path,
        bootstrap_resamples: int = 10_000,
        seed: int = 20260427,
    ) -> Phase6ControlArtifacts:
        """Evaluate health and persist fixed-path rolling controls."""
        health = WorkflowHealthEvaluator().evaluate(
            calendar=calendar,
            store=workflow_store,
            start_date=proof_start,
            end_date=proof_end,
        )
        controls = self.prepare(
            calendar=calendar,
            session_file=session_file,
            workflow_health=health,
            proof_start=proof_start,
            proof_end=proof_end,
            current_trade_date=current_trade_date,
            initial_cash=initial_cash,
            artifact_root=artifact_root,
            health_output=health_output,
            bootstrap_resamples=bootstrap_resamples,
            seed=seed,
        )
        health.write(health_output)
        write_phase6_controls(controls.aggregation_spec, aggregation_output)
        return controls


def encode_phase6_controls(spec: Phase6AggregationSpec) -> bytes:
    """Encode one Phase 6 aggregation input canonically."""
    return json.dumps(
        spec.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
    ).encode()


def write_phase6_controls(spec: Phase6AggregationSpec, output: Path) -> None:
    """Persist canonical Phase 6 controls with collision checks."""
    _write_once(output, encode_phase6_controls(spec))


def _write_once(path: Path, encoded: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as destination:
            destination.write(encoded)
    except FileExistsError:
        if path.read_bytes() != encoded:
            raise RuntimeError(f"Phase 6 control-spec collision at {path}") from None
