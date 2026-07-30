"""Validated file contracts for universe jobs and halt snapshots."""

from __future__ import annotations

import json
from datetime import date, datetime  # noqa: TC003 - Pydantic resolves these at runtime.
from typing import TYPE_CHECKING

import yaml
from pydantic import BaseModel, ConfigDict, Field

from quant_earning_edge.universe.job import HaltSnapshot
from quant_earning_edge.universe.models import UniverseConfig

if TYPE_CHECKING:
    from pathlib import Path


class EligibilityFileConfig(BaseModel):
    """Strict YAML representation of eligibility thresholds."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    min_price: float = Field(gt=0)
    min_market_cap_usd: float = Field(gt=0)
    min_avg_daily_volume: float = Field(ge=0)
    allowed_exchanges: frozenset[str] = Field(min_length=1)
    allowed_security_types: frozenset[str] = Field(min_length=1)
    exclude_halts: bool = True

    def to_domain(self) -> UniverseConfig:
        """Convert validated file values to the domain configuration."""
        return UniverseConfig(
            min_price=self.min_price,
            min_market_cap_usd=self.min_market_cap_usd,
            min_avg_daily_volume=self.min_avg_daily_volume,
            allowed_exchanges=self.allowed_exchanges,
            allowed_security_types=self.allowed_security_types,
            exclude_halts=self.exclude_halts,
        )


class UniverseJobFileConfig(BaseModel):
    """Strict daily-universe job configuration."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    eligibility: EligibilityFileConfig
    adv_sessions: int = Field(default=20, ge=1)


class HaltSnapshotFile(BaseModel):
    """External, timestamped halt-state input contract."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    asof_date: date
    captured_at: datetime
    symbols: frozenset[str] = frozenset()

    def to_domain(self) -> HaltSnapshot:
        """Convert the validated JSON payload to a normalized snapshot."""
        return HaltSnapshot(
            asof_date=self.asof_date,
            captured_at=self.captured_at,
            symbols=self.symbols,
        )


def load_universe_job_config(path: Path) -> UniverseJobFileConfig:
    """Load a strict YAML job configuration."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return UniverseJobFileConfig.model_validate(raw)


def load_halt_snapshot(path: Path) -> HaltSnapshot:
    """Load an explicit JSON halt snapshot; an empty set must still be recorded."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    return HaltSnapshotFile.model_validate(raw).to_domain()
