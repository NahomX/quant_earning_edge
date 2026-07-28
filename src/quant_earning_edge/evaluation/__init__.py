"""Standardized backtest evaluation and evidence persistence."""

from quant_earning_edge.evaluation.folds import (
    FoldEvaluation,
    WalkForwardEvaluation,
    WalkForwardEvaluator,
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
)
from quant_earning_edge.evaluation.tearsheet import HtmlTearsheetWriter

__all__ = [
    "BacktestResultCombiner",
    "BootstrapSummary",
    "ConfidenceInterval",
    "CostAttribution",
    "FoldArtifactSpec",
    "FoldBacktestResults",
    "FoldEvaluation",
    "HtmlTearsheetWriter",
    "PerformanceEvaluator",
    "PerformanceReport",
    "Phase4AggregationSpec",
    "Phase4GateEvaluation",
    "Phase4GateEvaluator",
    "ReplayRoundTrip",
    "ReplayRoundTripResult",
    "ReplayRoundTripSpec",
    "ReplaySessionAggregationSpec",
    "ReplaySessionAggregator",
    "ReplaySessionReport",
    "WalkForwardEvaluation",
    "WalkForwardEvaluator",
]
