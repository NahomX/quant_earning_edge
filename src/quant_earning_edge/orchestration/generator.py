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
        breaker_spec: Path | None,
        phase6_spec: Path,
        artifact_root: Path,
        order_controls_not_before: datetime,
        market_events_not_before: datetime,
        breaker_session_file: Path | None = None,
        freshness_symbol: str = "SPY",
        lease_seconds: int = 900,
        command_timeout_seconds: float = 1800,
    ) -> WorkflowRunSpec:
        planning = planning_spec.resolve()
        strategy = strategy_config.resolve()
        phase6 = phase6_spec.resolve()
        root = artifact_root.resolve() / f"trade_date={trade_date.isoformat()}"
        if (breaker_spec is None) == (breaker_session_file is None):
            raise ValueError(
                "workflow requires exactly one of a fixed breaker spec "
                "or an authoritative breaker session file"
            )
        breakers = breaker_spec.resolve() if breaker_spec is not None else None
        breaker_sessions = (
            breaker_session_file.resolve() if breaker_session_file is not None else None
        )
        control_evidence = root / "control-evidence"
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
        phase6_report = root / "phase6-progress.json"

        if breaker_sessions is None:
            assert breakers is not None
            freeze_commands = (
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
            )
            freeze_outputs = (planning, strategy, breakers, phase6)
            evaluate_breaker_command = QeeCommandSpec(
                arguments=(
                    "monitoring",
                    "circuit-breakers",
                    "--spec-file",
                    str(breakers),
                    "--output",
                    str(breaker_decision),
                ),
            )
        else:
            freeze_commands = (
                QeeCommandSpec(
                    arguments=(
                        "monitoring",
                        "prepare-breaker-bundle",
                        "--control-date",
                        trade_date.isoformat(),
                        "--session-file",
                        str(breaker_sessions),
                        "--artifact-root",
                        str(artifact_root.resolve()),
                        "--output-directory",
                        str(control_evidence),
                        "--symbol",
                        freshness_symbol,
                    ),
                    artifact_json_keys=(
                        "freshness_path",
                        "reconciliation_age_path",
                        "breaker_spec_path",
                    ),
                ),
            )
            freeze_outputs = (planning, strategy, phase6, breaker_sessions)
            evaluate_breaker_command = QeeCommandSpec(
                arguments=(
                    "monitoring",
                    "circuit-breakers",
                    "--output",
                    str(breaker_decision),
                ),
                artifact_bindings=(
                    ArtifactArgumentBinding(
                        source_stage=WorkflowStage.FREEZE_INPUTS,
                        option="--spec-file",
                        file_glob="breaker-controls-*.json",
                        path_contains=("control-evidence",),
                        maximum_matches=1,
                    ),
                ),
            )

        stages = (
            WorkflowStageCommandSpec(
                stage=WorkflowStage.FREEZE_INPUTS,
                not_before=order_controls_not_before,
                commands=freeze_commands,
                output_files=freeze_outputs,
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
                commands=(evaluate_breaker_command,),
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
                            "reconcile-frozen-revision",
                            "--frozen-orders",
                            str(frozen),
                            "--output-directory",
                            str(root),
                        ),
                        artifact_json_keys=("output",),
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
                output_files=(),
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
