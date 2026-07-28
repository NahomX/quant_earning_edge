"""Validated JSON boundary for reproducible daily backtest runs."""

from __future__ import annotations

from datetime import date  # noqa: TC003 - Pydantic resolves field types at runtime.
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from quant_earning_edge.backtest.engine import DailyMark, TradeIntent


class DailyMarkSpec(BaseModel):
    """Serialized daily valuation mark."""

    model_config = ConfigDict(extra="forbid")

    symbol: str
    session_date: date
    close: float = Field(gt=0)

    def to_domain(self) -> DailyMark:
        return DailyMark(**self.model_dump())


class TradeIntentSpec(BaseModel):
    """Serialized daily trade instruction."""

    model_config = ConfigDict(extra="forbid")

    trade_id: str
    symbol: str
    side: Literal["long", "short"]
    entry_date: date
    exit_date: date
    shares: int = Field(gt=0)
    entry_price: float = Field(gt=0)
    exit_price: float = Field(gt=0)
    entry_average_daily_volume_shares: float = Field(gt=0)
    exit_average_daily_volume_shares: float = Field(gt=0)
    holding_sessions: int = Field(gt=0)
    triggered_stop_price: float | None = Field(default=None, gt=0)
    atr5: float | None = Field(default=None, gt=0)

    def to_domain(self) -> TradeIntent:
        return TradeIntent(**self.model_dump())


class BacktestSpec(BaseModel):
    """Complete deterministic input for one daily ledger run."""

    model_config = ConfigDict(extra="forbid")

    initial_cash: float = Field(gt=0)
    sessions: tuple[date, ...]
    marks: tuple[DailyMarkSpec, ...]
    trades: tuple[TradeIntentSpec, ...]

    def domain_inputs(
        self,
    ) -> tuple[float, tuple[date, ...], tuple[DailyMark, ...], tuple[TradeIntent, ...]]:
        """Convert validated serialized models into immutable domain inputs."""
        return (
            self.initial_cash,
            self.sessions,
            tuple(item.to_domain() for item in self.marks),
            tuple(item.to_domain() for item in self.trades),
        )
