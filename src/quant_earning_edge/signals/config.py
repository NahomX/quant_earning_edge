"""Strict Phase 4 earnings-strategy configuration schema."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from quant_earning_edge.features import FEATURE_REGISTRY

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Self


class StrictModel(BaseModel):
    """Reject undeclared configuration fields."""

    model_config = ConfigDict(extra="forbid")


class UniverseSpec(StrictModel):
    min_price: float = Field(gt=0)
    min_market_cap_usd: float = Field(gt=0)
    min_avg_daily_volume: float = Field(gt=0)
    exchanges: tuple[Literal["NYSE", "NASDAQ", "AMEX"], ...]
    exclude_halts: bool


class EarningsWindowSpec(StrictModel):
    bmo: bool
    amc: bool
    dmt: bool


class EventSpec(StrictModel):
    source: Literal["finnhub"]
    earnings_window: EarningsWindowSpec


class LabelSpec(StrictModel):
    horizon: Literal["d1_open_to_close"]
    threshold: float

    @property
    def column_name(self) -> Literal["forward_1d_open_to_close"]:
        """Return the training target aligned with the executed holding window."""
        return "forward_1d_open_to_close"


class WalkForwardSpec(StrictModel):
    train_window_months: int = Field(gt=0)
    test_window_months: int = Field(gt=0)
    step_months: int = Field(gt=0)
    embargo_days: int = Field(ge=5)


class HyperparameterSearchSpec(StrictModel):
    backend: Literal["optuna"]
    n_trials: int = Field(gt=0, le=200)
    objective: Literal["oos_sharpe"]


class ModelSpec(StrictModel):
    type: Literal["lightgbm_binary"]
    early_stopping_rounds: int = Field(gt=0)
    hyperparam_search: HyperparameterSearchSpec


class SizingSpec(StrictModel):
    method: Literal["fractional_kelly"]
    kelly_fraction: float = Field(gt=0, le=1)
    rolling_window_days: int = Field(gt=0)


class CapsSpec(StrictModel):
    max_position_pct: float = Field(gt=0, le=1)
    max_sector_pct: float = Field(gt=0, le=1)
    max_gross_exposure_pct: float = Field(gt=0, le=1)

    @model_validator(mode="after")
    def validate_ordering(self) -> Self:
        if self.max_position_pct > self.max_sector_pct:
            raise ValueError("position cap must not exceed sector cap")
        if self.max_sector_pct > self.max_gross_exposure_pct:
            raise ValueError("sector cap must not exceed gross cap")
        return self


class PortfolioSpec(StrictModel):
    top_k: int = Field(gt=0)
    sizing: SizingSpec
    caps: CapsSpec


class SpreadTierSpec(StrictModel):
    min_price: float = Field(ge=0)
    half_spread_bps: float = Field(ge=0)


class CostsSpec(StrictModel):
    commission_bps_per_side: float = Field(ge=0)
    half_spread_by_price_tier: tuple[SpreadTierSpec, ...]
    market_impact_coef_bps: float = Field(ge=0)
    borrow_bps_annualized: float = Field(ge=0)

    @model_validator(mode="after")
    def validate_tiers(self) -> Self:
        minimums = tuple(item.min_price for item in self.half_spread_by_price_tier)
        if not minimums or minimums != tuple(sorted(set(minimums), reverse=True)):
            raise ValueError("spread tiers must have unique descending minimum prices")
        if minimums[-1] != 0:
            raise ValueError("spread tiers must end with a zero-price catch-all")
        return self


class EarningsStrategyConfig(StrictModel):
    """Fully validated training, portfolio, and cost contract."""

    name: Literal["earnings_v1"]
    description: str = Field(min_length=1)
    universe: UniverseSpec
    event: EventSpec
    features: tuple[str, ...] = Field(min_length=1, max_length=20)
    label: LabelSpec
    walkforward: WalkForwardSpec
    model: ModelSpec
    portfolio: PortfolioSpec
    costs: CostsSpec
    seed: int

    @model_validator(mode="after")
    def validate_features_and_windows(self) -> Self:
        FEATURE_REGISTRY.select(self.features)
        if not (self.event.earnings_window.bmo or self.event.earnings_window.amc):
            raise ValueError("at least one pre-session earnings window must be enabled")
        if self.portfolio.sizing.rolling_window_days != 60:
            raise ValueError("the precommitted Kelly history window is 60 days")
        return self


def load_strategy_config(path: Path) -> EarningsStrategyConfig:
    """Load YAML through the strict schema."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("strategy YAML root must be an object")
    return EarningsStrategyConfig.model_validate(raw)
