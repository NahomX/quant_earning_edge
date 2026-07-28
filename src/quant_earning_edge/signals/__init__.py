"""Deterministic baseline and strategy signal interfaces."""

from quant_earning_edge.signals.config import EarningsStrategyConfig, load_strategy_config
from quant_earning_edge.signals.lgbm_model import (
    FoldModelResult,
    LightgbmWalkForwardTrainer,
    OosPrediction,
    WalkForwardModelRun,
)
from quant_earning_edge.signals.momentum import (
    CrossSectionalMomentum,
    MomentumPrice,
    MomentumSignal,
    SignalSide,
)

__all__ = [
    "CrossSectionalMomentum",
    "EarningsStrategyConfig",
    "FoldModelResult",
    "LightgbmWalkForwardTrainer",
    "MomentumPrice",
    "MomentumSignal",
    "OosPrediction",
    "SignalSide",
    "WalkForwardModelRun",
    "load_strategy_config",
]
