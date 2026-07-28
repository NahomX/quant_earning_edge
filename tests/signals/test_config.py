"""Strict Phase 4 strategy configuration tests."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from quant_earning_edge.features import FEATURE_REGISTRY
from quant_earning_edge.signals import load_strategy_config


def test_committed_strategy_config_matches_registered_features() -> None:
    path = Path("configs/strategies/earnings_v1.yaml")

    config = load_strategy_config(path)

    assert config.name == "earnings_v1"
    assert len(config.features) == 16
    assert set(config.features) == {item.name for item in FEATURE_REGISTRY.values()}
    assert config.walkforward.embargo_days >= 5
    assert config.model.hyperparam_search.n_trials <= 200


def test_unknown_feature_is_rejected(tmp_path: Path) -> None:
    source = Path("configs/strategies/earnings_v1.yaml")
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    raw["features"].append("future_magic")
    target = tmp_path / "invalid.yaml"
    target.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(KeyError, match="unknown feature"):
        load_strategy_config(target)


def test_risk_cap_inversion_is_rejected(tmp_path: Path) -> None:
    source = Path("configs/strategies/earnings_v1.yaml")
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    raw["portfolio"]["caps"]["max_position_pct"] = 0.25
    target = tmp_path / "invalid.yaml"
    target.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(ValidationError, match="position cap"):
        load_strategy_config(target)
