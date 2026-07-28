"""Deterministic baseline and strategy signal interfaces."""

from quant_earning_edge.signals.config import EarningsStrategyConfig, load_strategy_config
from quant_earning_edge.signals.event_trades import (
    EventExecutionObservation,
    EventTradePlanner,
    EventTradePlanningSpec,
    PlannedEventTrades,
)
from quant_earning_edge.signals.lgbm_model import (
    FeatureAttribution,
    FoldModelResult,
    LightgbmWalkForwardTrainer,
    OosPrediction,
    WalkForwardModelRun,
)
from quant_earning_edge.signals.live_orders import (
    DailyOrderPlanningSpec,
    FrozenDailyOrders,
    LiveCandidateSpec,
    LiveOrderPlanner,
    strategy_file_sha256,
)
from quant_earning_edge.signals.momentum import (
    CrossSectionalMomentum,
    MomentumPrice,
    MomentumSignal,
    SignalSide,
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
    "LightgbmWalkForwardTrainer",
    "LiveCandidateSpec",
    "LiveOrderPlanner",
    "MomentumPrice",
    "MomentumSignal",
    "OosPrediction",
    "PlannedEventTrades",
    "SignalSide",
    "WalkForwardModelRun",
    "load_strategy_config",
    "strategy_file_sha256",
]
