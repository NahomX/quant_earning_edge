"""Deterministic baseline and strategy signal interfaces."""

from quant_earning_edge.signals.momentum import (
    CrossSectionalMomentum,
    MomentumPrice,
    MomentumSignal,
    SignalSide,
)

__all__ = [
    "CrossSectionalMomentum",
    "EarningsStrategyConfig",
    "MomentumPrice",
    "MomentumSignal",
    "SignalSide",
    "load_strategy_config",
]
from quant_earning_edge.signals.config import EarningsStrategyConfig, load_strategy_config
