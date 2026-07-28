"""Live order planning excludes realized labels and future execution outcomes."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from quant_earning_edge.cli import app
from quant_earning_edge.signals import (
    DailyOrderPlanningSpec,
    FrozenDailyOrders,
    LiveOrderPlanner,
    load_strategy_config,
    strategy_file_sha256,
)


def _strategy_path() -> Path:
    return Path(__file__).parents[2] / "configs/strategies/earnings_v1.yaml"


def _raw_spec(*, outcome_count: int = 20) -> dict[str, Any]:
    trade_date = date(2026, 7, 28)
    decision = datetime(2026, 7, 27, 22, 0, tzinfo=UTC)
    candidates = []
    for symbol, probability, price in (("AAA", 0.8, 100.0), ("BBB", 0.7, 50.0)):
        candidates.append(
            {
                "symbol": symbol,
                "sector": "Technology",
                "probability_up": probability,
                "sizing_price": price,
                "sizing_price_observed_at": decision,
                "frozen_average_daily_volume_shares": 1_000_000,
                "decision_snapshot": {
                    "ticker": symbol,
                    "observed_at": decision,
                    "bid_price": price - 0.1,
                    "ask_price": price + 0.1,
                    "bid_size": 100,
                    "ask_size": 100,
                    "last_trade_price": price,
                    "last_trade_at": decision - timedelta(seconds=1),
                },
            }
        )
    return {
        "trade_date": trade_date,
        "decision_at": decision,
        "equity": 100_000,
        "candidates": candidates,
        "outcomes": [
            {
                "closed_date": trade_date - timedelta(days=index + 1),
                "net_return": 0.02 if index % 2 == 0 else -0.01,
            }
            for index in range(outcome_count)
        ],
        "entry_submitted_at": datetime(2026, 7, 28, 13, 30, tzinfo=UTC),
        "entry_expires_at": datetime(2026, 7, 28, 13, 35, tzinfo=UTC),
        "exit_submitted_at": datetime(2026, 7, 28, 19, 50, tzinfo=UTC),
        "exit_expires_at": datetime(2026, 7, 28, 20, 1, tzinfo=UTC),
    }


def _plan(spec: DailyOrderPlanningSpec) -> FrozenDailyOrders:
    strategy_path = _strategy_path()
    return LiveOrderPlanner(
        load_strategy_config(strategy_path),
        strategy_sha256=strategy_file_sha256(strategy_path),
    ).plan(spec)


def test_live_planner_freezes_linked_paper_and_replay_order_ids() -> None:
    artifact = _plan(DailyOrderPlanningSpec.model_validate(_raw_spec()))

    assert len(artifact.portfolio.positions) == 2
    assert len(artifact.intended_orders) == 4
    assert len(artifact.paper_batch.orders) == 4
    intended_ids = tuple(item.order_id for item in artifact.intended_orders)
    paper_ids = tuple(item.client_order_id for item in artifact.paper_batch.orders)
    assert intended_ids == paper_ids
    assert tuple(item.ticker for item in artifact.decision_snapshots) == ("AAA", "BBB")
    assert {item.time_in_force for item in artifact.paper_batch.orders if item.side == "sell"} == {
        "cls"
    }
    assert b"realized_label" not in artifact.canonical_bytes
    assert b"entry_price" not in artifact.canonical_bytes
    assert b"exit_price" not in artifact.canonical_bytes


def test_insufficient_closed_history_produces_explicit_no_trade_artifact() -> None:
    artifact = _plan(DailyOrderPlanningSpec.model_validate(_raw_spec(outcome_count=19)))

    assert artifact.portfolio.positions == ()
    assert artifact.intended_orders == ()
    assert artifact.paper_batch.orders == ()
    assert artifact.decision_snapshots == ()


def test_schema_rejects_realized_labels_and_future_observations() -> None:
    raw = _raw_spec()
    raw["candidates"][0]["realized_label"] = 1
    with pytest.raises(ValidationError, match="realized_label"):
        DailyOrderPlanningSpec.model_validate(raw)

    future = _raw_spec()
    future["outcomes"].append({"closed_date": future["trade_date"], "net_return": 0.5})
    with pytest.raises(ValidationError, match="closed before"):
        DailyOrderPlanningSpec.model_validate(future)


def test_live_order_cli_writes_immutable_artifact(tmp_path: Path) -> None:
    spec_file = tmp_path / "live-plan.json"
    spec_file.write_text(
        json.dumps(
            _raw_spec(),
            default=lambda item: item.isoformat(),
        ),
        encoding="utf-8",
    )
    output = tmp_path / "orders.json"
    paper_batch = tmp_path / "paper-batch.json"

    result = CliRunner().invoke(
        app,
        [
            "model",
            "plan-live-orders",
            "--planning-spec",
            str(spec_file),
            "--strategy-config",
            str(_strategy_path()),
            "--output",
            str(output),
            "--paper-batch-output",
            str(paper_batch),
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["position_count"] == 2
    assert payload["intended_order_count"] == 4
    assert json.loads(output.read_bytes())["strategy_config_sha256"]
    assert len(json.loads(paper_batch.read_bytes())["orders"]) == 4
    assert FrozenDailyOrders.load(output).sha256 == payload["sha256"]


def test_frozen_order_loader_rejects_paper_replay_identity_tampering(
    tmp_path: Path,
) -> None:
    artifact = _plan(DailyOrderPlanningSpec.model_validate(_raw_spec()))
    raw = json.loads(artifact.canonical_bytes)
    raw["paper_batch"]["orders"][0]["quantity"] += 1
    path = tmp_path / "tampered.json"
    path.write_text(
        json.dumps(raw, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="invalid frozen daily orders"):
        FrozenDailyOrders.load(path)


def test_no_trade_frozen_artifact_materializes_without_market_files(
    tmp_path: Path,
) -> None:
    artifact = _plan(DailyOrderPlanningSpec.model_validate(_raw_spec(outcome_count=19)))
    frozen_path = tmp_path / "frozen.json"
    artifact.write(frozen_path)
    output_dir = tmp_path / "replay-specs"
    manifest = tmp_path / "manifest.json"

    result = CliRunner().invoke(
        app,
        [
            "backtest",
            "materialize-frozen-replay-specs",
            "--frozen-orders",
            str(frozen_path),
            "--strategy-config",
            str(_strategy_path()),
            "--output-dir",
            str(output_dir),
            "--manifest-output",
            str(manifest),
        ],
    )

    assert result.exit_code == 0
    assert json.loads(result.stdout)["order_count"] == 0
    assert json.loads(manifest.read_bytes())["artifacts"] == []
