"""Standardized backtest evaluation and evidence persistence."""

from quant_earning_edge.evaluation.folds import (
    FoldEvaluation,
    WalkForwardEvaluation,
    WalkForwardEvaluator,
)
from quant_earning_edge.evaluation.momentum_builder import (
    HISTORICAL_SPY_MEMBERSHIP_SCHEMA,
    MomentumBaselineBuild,
    MomentumBaselineBuilder,
    MomentumBaselineBuildSpec,
)
from quant_earning_edge.evaluation.momentum_gate import (
    MomentumBaselineManifest,
    MomentumBenchmarkGateEvaluator,
    MomentumBenchmarkGateReport,
    MomentumBenchmarkReferenceSpec,
)
from quant_earning_edge.evaluation.phase6_controls import (
    Phase6ControlArtifacts,
    Phase6ControlBuilder,
    encode_phase6_controls,
    write_phase6_controls,
)
from quant_earning_edge.evaluation.phase6_finalize import (
    Phase6CompletionFinalizer,
    Phase6FinalizationArtifacts,
    Phase6FinalizationEvidence,
)
from quant_earning_edge.evaluation.phase6_gate import (
    Phase6AggregationSpec,
    Phase6GateEvaluator,
    Phase6GateReport,
)
from quant_earning_edge.evaluation.replay_session import (
    ReplayRoundTrip,
    ReplayRoundTripResult,
    ReplayRoundTripSpec,
    ReplaySessionAggregationSpec,
    ReplaySessionAggregator,
    ReplaySessionReport,
)
from quant_earning_edge.evaluation.report import (
    BootstrapSummary,
    ConfidenceInterval,
    CostAttribution,
    PerformanceEvaluator,
    PerformanceReport,
)
from quant_earning_edge.evaluation.strategy_gate import (
    BacktestResultCombiner,
    FoldArtifactSpec,
    FoldBacktestResults,
    Phase4AggregationSpec,
    Phase4GateEvaluation,
    Phase4GateEvaluator,
    Phase4PromotionEvidence,
)
from quant_earning_edge.evaluation.tearsheet import HtmlTearsheetWriter
from quant_earning_edge.evaluation.tracking import (
    BacktestTrackingReference,
    default_artifact_location,
    default_tracking_uri,
    log_backtest_run,
    source_tree_sha256,
)

__all__ = [
    "HISTORICAL_SPY_MEMBERSHIP_SCHEMA",
    "BacktestResultCombiner",
    "BacktestTrackingReference",
    "BootstrapSummary",
    "ConfidenceInterval",
    "CostAttribution",
    "FoldArtifactSpec",
    "FoldBacktestResults",
    "FoldEvaluation",
    "HtmlTearsheetWriter",
    "MomentumBaselineBuild",
    "MomentumBaselineBuildSpec",
    "MomentumBaselineBuilder",
    "MomentumBaselineManifest",
    "MomentumBenchmarkGateEvaluator",
    "MomentumBenchmarkGateReport",
    "MomentumBenchmarkReferenceSpec",
    "PerformanceEvaluator",
    "PerformanceReport",
    "Phase4AggregationSpec",
    "Phase4GateEvaluation",
    "Phase4GateEvaluator",
    "Phase4PromotionEvidence",
    "Phase6AggregationSpec",
    "Phase6CompletionFinalizer",
    "Phase6ControlArtifacts",
    "Phase6ControlBuilder",
    "Phase6FinalizationArtifacts",
    "Phase6FinalizationEvidence",
    "Phase6GateEvaluator",
    "Phase6GateReport",
    "ReplayRoundTrip",
    "ReplayRoundTripResult",
    "ReplayRoundTripSpec",
    "ReplaySessionAggregationSpec",
    "ReplaySessionAggregator",
    "ReplaySessionReport",
    "WalkForwardEvaluation",
    "WalkForwardEvaluator",
    "default_artifact_location",
    "default_tracking_uri",
    "encode_phase6_controls",
    "log_backtest_run",
    "source_tree_sha256",
    "write_phase6_controls",
]
