"""Deterministic baseline and strategy signal interfaces."""

from quant_earning_edge.signals.config import EarningsStrategyConfig, load_strategy_config
from quant_earning_edge.signals.event_trades import (
    EventExecutionObservation,
    EventTradePlanner,
    EventTradePlanningSpec,
    PlannedEventTrades,
    TradeCohort,
    run_event_plan,
)
from quant_earning_edge.signals.hyperparameter_search import (
    OptunaLightgbmSearch,
    OptunaStudyArtifact,
    OptunaTrialEvidence,
)
from quant_earning_edge.signals.lgbm_hyperparameters import LightgbmHyperparameters
from quant_earning_edge.signals.lgbm_model import (
    FeatureAttribution,
    FoldModelResult,
    LightgbmWalkForwardTrainer,
    OosPrediction,
    WalkForwardModelRun,
)
from quant_earning_edge.signals.live_capture import (
    LiveSourceCaptureArtifact,
    LiveSourceCaptureAssembler,
)
from quant_earning_edge.signals.live_orders import (
    DailyOrderPlanningSpec,
    FrozenDailyOrders,
    LiveCandidateSpec,
    LiveOrderPlanner,
    strategy_file_sha256,
)
from quant_earning_edge.signals.live_planning import (
    LiveMarketObservationSpec,
    LivePlanningAssembler,
    LivePlanningSourceSpec,
    ScoredPlanningArtifact,
)
from quant_earning_edge.signals.momentum import (
    CrossSectionalMomentum,
    MomentumPrice,
    MomentumSignal,
    SignalSide,
)
from quant_earning_edge.signals.production_model import (
    ProductionModelArtifact,
    ProductionModelTrainer,
)

__all__ = [
    "CrossSectionalMomentum",
    "DailyOrderPlanningSpec",
    "EarningsStrategyConfig",
    "EventExecutionObservation",
    "EventTradePlanner",
    "EventTradePlanningSpec",
    "FeatureAttribution",
    "FoldModelResult",
    "FrozenDailyOrders",
    "LightgbmHyperparameters",
    "LightgbmWalkForwardTrainer",
    "LiveCandidateSpec",
    "LiveMarketObservationSpec",
    "LiveOrderPlanner",
    "LivePlanningAssembler",
    "LivePlanningSourceSpec",
    "LiveSourceCaptureArtifact",
    "LiveSourceCaptureAssembler",
    "MomentumPrice",
    "MomentumSignal",
    "OosPrediction",
    "OptunaLightgbmSearch",
    "OptunaStudyArtifact",
    "OptunaTrialEvidence",
    "PlannedEventTrades",
    "ProductionModelArtifact",
    "ProductionModelTrainer",
    "ScoredPlanningArtifact",
    "SignalSide",
    "TradeCohort",
    "WalkForwardModelRun",
    "load_strategy_config",
    "run_event_plan",
    "strategy_file_sha256",
]
