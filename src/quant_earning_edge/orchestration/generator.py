"""Generate the concrete eight-stage daily paper/replay workflow."""

from __future__ import annotations

from typing import TYPE_CHECKING

from quant_earning_edge.orchestration.commands import (
    ArtifactArgumentBinding,
    QeeCommandSpec,
    WorkflowRunSpec,
    WorkflowStageCommandSpec,
)
from quant_earning_edge.orchestration.workflow import WorkflowStage

if TYPE_CHECKING:
    from datetime import date, datetime
    from pathlib import Path

    from quant_earning_edge.orchestration.workflow import WorkflowTrigger


class DailyWorkflowSpecGenerator:
    """Bind fixed daily inputs and dynamic content-addressed outputs."""

    def generate(
        self,
        *,
        trade_date: date,
        trigger: WorkflowTrigger,
        worker_id: str,
        planning_spec: Path,
        strategy_config: Path,
        breaker_spec: Path,
        phase6_spec: Path,
        artifact_root: Path,
        market_events_not_before: datetime,
        lease_seconds: int = 900,
        command_timeout_seconds: float = 1800,
    ) -> WorkflowRunSpec:
        planning = planning_spec.resolve()
        strategy = strategy_config.resolve()
        breakers = breaker_spec.resolve()
        phase6 = phase6_spec.resolve()
        root = artifact_root.resolve() / f"trade_date={trade_date.isoformat()}"
        frozen = root / "frozen-daily-orders.json"
        paper_batch = root / "paper-order-batch.json"
        breaker_decision = root / "circuit-breaker-decision.json"
        paper_submission = root / "paper-batch-submission.json"
        capture_manifest = root / "frozen-market-events.json"
        replay_spec_directory = root / "replay-specs"
        replay_manifest = root / "replay-materialization-manifest.json"
        evidence_directory = root / "replay-evidence"
        evidence_index = evidence_directory / "index.json"
        session_report = root / "replay-session.json"
        paper_reconciliation = root / "paper-reconciliation.json"
        phase6_report = root / "phase6-progress.json"

        stages = (
            WorkflowStageCommandSpec(
                stage=WorkflowStage.FREEZE_INPUTS,
                commands=(
                    QeeCommandSpec(
                        arguments=(
                            "calendar",
                            "sessions",
                            "--start",
                            trade_date.isoformat(),
                            "--end",
                            trade_date.isoformat(),
                        ),
                        artifact_json_keys=("path",),
                    ),
                ),
                output_files=(planning, strategy, breakers, phase6),
            ),
            WorkflowStageCommandSpec(
                stage=WorkflowStage.GENERATE_ORDER_PLAN,
                commands=(
                    QeeCommandSpec(
                        arguments=(
                            "model",
                            "plan-live-orders",
                            "--planning-spec",
                            str(planning),
                            "--strategy-config",
                            str(strategy),
                            "--output",
                            str(frozen),
                            "--paper-batch-output",
                            str(paper_batch),
                        ),
                    ),
                ),
                output_files=(frozen, paper_batch),
            ),
            WorkflowStageCommandSpec(
                stage=WorkflowStage.EVALUATE_BREAKERS,
                commands=(
                    QeeCommandSpec(
                        arguments=(
                            "monitoring",
                            "circuit-breakers",
                            "--spec-file",
                            str(breakers),
                            "--output",
                            str(breaker_decision),
                        ),
                    ),
                ),
                output_files=(breaker_decision,),
            ),
            WorkflowStageCommandSpec(
                stage=WorkflowStage.SUBMIT_PAPER_ORDERS,
                commands=(
                    QeeCommandSpec(
                        arguments=(
                            "paper",
                            "submit-batch",
                            "--spec-file",
                            str(paper_batch),
                            "--breaker-decision",
                            str(breaker_decision),
                            "--output",
                            str(paper_submission),
                        ),
                    ),
                ),
                output_files=(paper_submission,),
            ),
            WorkflowStageCommandSpec(
                stage=WorkflowStage.CAPTURE_MARKET_EVENTS,
                not_before=market_events_not_before,
                commands=(
                    QeeCommandSpec(
                        arguments=(
                            "ingest",
                            "frozen-market-events",
                            "--frozen-orders",
                            str(frozen),
                            "--manifest-output",
                            str(capture_manifest),
                        ),
                        artifact_json_keys=("quote_paths", "trade_paths"),
                    ),
                ),
                output_files=(capture_manifest,),
            ),
            WorkflowStageCommandSpec(
                stage=WorkflowStage.REPLAY_ORDERS,
                commands=(
                    QeeCommandSpec(
                        arguments=(
                            "backtest",
                            "materialize-frozen-replay-specs",
                            "--frozen-orders",
                            str(frozen),
                            "--strategy-config",
                            str(strategy),
                            "--output-dir",
                            str(replay_spec_directory),
                            "--manifest-output",
                            str(replay_manifest),
                        ),
                        artifact_json_keys=("replay_spec_paths",),
                        artifact_bindings=(
                            _binding(
                                source_stage=WorkflowStage.CAPTURE_MARKET_EVENTS,
                                option="--quote-file",
                                marker="dataset=nbbo-quotes",
                            ),
                            _binding(
                                source_stage=WorkflowStage.CAPTURE_MARKET_EVENTS,
                                option="--trade-file",
                                marker="dataset=stock-trades",
                            ),
                        ),
                    ),
                    QeeCommandSpec(
                        arguments=(
                            "backtest",
                            "replay-materialization",
                            "--manifest-file",
                            str(replay_manifest),
                            "--spec-directory",
                            str(replay_spec_directory),
                            "--output-directory",
                            str(evidence_directory),
                            "--index-output",
                            str(evidence_index),
                        ),
                        artifact_json_keys=("replay_evidence_paths",),
                    ),
                    QeeCommandSpec(
                        arguments=(
                            "evaluation",
                            "replay-frozen-session",
                            "--frozen-orders",
                            str(frozen),
                            "--strategy-config",
                            str(strategy),
                            "--output",
                            str(session_report),
                        ),
                        artifact_bindings=(
                            ArtifactArgumentBinding(
                                source_stage=WorkflowStage.REPLAY_ORDERS,
                                option="--evidence-file",
                                file_glob="replay-evidence-*.json",
                                minimum_matches=0,
                            ),
                        ),
                    ),
                ),
                output_files=(replay_manifest, evidence_index, session_report),
            ),
            WorkflowStageCommandSpec(
                stage=WorkflowStage.RECONCILE_SESSION,
                commands=(
                    QeeCommandSpec(
                        arguments=(
                            "paper",
                            "reconcile-frozen",
                            "--frozen-orders",
                            str(frozen),
                            "--output",
                            str(paper_reconciliation),
                        ),
                        artifact_bindings=(
                            ArtifactArgumentBinding(
                                source_stage=WorkflowStage.REPLAY_ORDERS,
                                option="--evidence-file",
                                file_glob="replay-evidence-*.json",
                                minimum_matches=0,
                            ),
                        ),
                    ),
                ),
                output_files=(paper_reconciliation,),
            ),
            WorkflowStageCommandSpec(
                stage=WorkflowStage.EVALUATE_PHASE6_PROGRESS,
                commands=(
                    QeeCommandSpec(
                        arguments=(
                            "evaluation",
                            "phase6-gate",
                            "--aggregation-spec",
                            str(phase6),
                            "--output",
                            str(phase6_report),
                        ),
                    ),
                ),
                output_files=(phase6_report,),
            ),
        )
        return WorkflowRunSpec(
            trade_date=trade_date,
            trigger=trigger,
            worker_id=worker_id,
            lease_seconds=lease_seconds,
            command_timeout_seconds=command_timeout_seconds,
            stages=stages,
        )


def _binding(
    *,
    source_stage: WorkflowStage,
    option: str,
    marker: str,
) -> ArtifactArgumentBinding:
    return ArtifactArgumentBinding(
        source_stage=source_stage,
        option=option,
        file_glob="*.parquet",
        path_contains=(marker,),
        minimum_matches=0,
    )
