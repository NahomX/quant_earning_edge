"""Universe YAML and halt-input validation tests."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from quant_earning_edge.universe import sector_from_sic_code
from quant_earning_edge.universe.config import (
    load_halt_snapshot,
    load_universe_job_config,
)


def test_default_universe_config_matches_committed_strategy() -> None:
    config = load_universe_job_config(Path("configs/universe/default.yaml"))
    domain = config.eligibility.to_domain()

    assert domain.min_price == 5
    assert domain.min_market_cap_usd == 500_000_000
    assert domain.min_avg_daily_volume == 1_000_000
    assert domain.allowed_exchanges == frozenset({"XNYS", "XNAS", "XASE"})
    assert config.adv_sessions == 20


def test_unknown_config_field_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "universe.yaml"
    path.write_text(
        """
eligibility:
  min_price: 5
  min_market_cap_usd: 500000000
  min_avg_daily_volume: 1000000
  allowed_exchanges: [XNAS]
  allowed_security_types: [CS]
  exclude_halts: true
unexpected: true
""",
        encoding="utf-8",
    )

    with pytest.raises(ValidationError, match="unexpected"):
        load_universe_job_config(path)


def test_halt_snapshot_requires_timestamp_and_normalizes_symbols(tmp_path: Path) -> None:
    path = tmp_path / "halts.json"
    path.write_text(
        """
{
  "asof_date": "2026-07-27",
  "captured_at": "2026-07-27T21:00:00Z",
  "symbols": [" aapl ", "MSFT"]
}
""",
        encoding="utf-8",
    )

    snapshot = load_halt_snapshot(path)

    assert snapshot.symbols == frozenset({"AAPL", "MSFT"})


def test_sic_divisions_create_conservative_exposure_buckets() -> None:
    assert sector_from_sic_code("3571") == "MANUFACTURING"
    assert sector_from_sic_code("6021") == "FINANCE"
    assert sector_from_sic_code(None) == "UNCLASSIFIED"
    assert sector_from_sic_code("unknown") == "UNCLASSIFIED"
