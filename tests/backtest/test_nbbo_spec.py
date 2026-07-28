"""Validated and immutable NBBO replay evidence boundaries."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from quant_earning_edge.backtest import NbboReplayEvidence, NbboReplaySpec, replay_order
from quant_earning_edge.cli import app

if TYPE_CHECKING:
    from pathlib import Path


def _payload() -> dict[str, object]:
    decision = datetime(2025, 1, 2, 21, 30, tzinfo=UTC)
    submitted = datetime(2025, 1, 3, 14, 30, tzinfo=UTC)
    return {
        "order": {
            "order_id": "proof-order",
            "ticker": "AAA",
            "side": "buy",
            "quantity": 100,
            "decision_time": decision.isoformat(),
            "submitted_at": submitted.isoformat(),
            "expires_at": (submitted + timedelta(hours=6, minutes=30)).isoformat(),
            "average_daily_volume_shares": 1_000_000,
            "aggressiveness": "aggressive",
        },
        "decision_snapshot": {
            "ticker": "AAA",
            "observed_at": decision.isoformat(),
            "bid_price": 99.9,
            "ask_price": 100.1,
            "bid_size": 100,
            "ask_size": 100,
            "last_trade_price": 100.0,
            "last_trade_at": (decision - timedelta(seconds=1)).isoformat(),
        },
        "quotes": [
            {
                "ticker": "AAA",
                "timestamp": submitted.isoformat(),
                "sequence": 1,
                "bid_price": 100.0,
                "ask_price": 100.2,
                "bid_size": 100,
                "ask_size": 100,
            }
        ],
        "trades": [],
    }


def test_spec_hash_is_semantic_and_replay_evidence_is_idempotent(tmp_path: Path) -> None:
    payload = _payload()
    compact = NbboReplaySpec.model_validate_json(json.dumps(payload))
    indented = NbboReplaySpec.model_validate_json(json.dumps(payload, indent=4))
    order, snapshot, quotes, trades, config = compact.domain_inputs()
    result = replay_order(
        order,
        decision_snapshot=snapshot,
        quotes=quotes,
        trades=trades,
        config=config,
    )
    evidence = NbboReplayEvidence.build(spec=compact, result=result)
    output = tmp_path / "evidence.json"

    evidence.write(output)
    evidence.write(output)

    assert compact.sha256 == indented.sha256
    assert json.loads(output.read_bytes())["input_sha256"] == compact.sha256
    assert evidence.sha256 == hashlib.sha256(output.read_bytes()).hexdigest()


def test_evidence_collision_is_rejected(tmp_path: Path) -> None:
    spec = NbboReplaySpec.model_validate(_payload())
    order, snapshot, quotes, trades, config = spec.domain_inputs()
    result = replay_order(
        order,
        decision_snapshot=snapshot,
        quotes=quotes,
        trades=trades,
        config=config,
    )
    evidence = NbboReplayEvidence.build(spec=spec, result=result)
    output = tmp_path / "evidence.json"
    output.write_text("different", encoding="utf-8")

    with pytest.raises(RuntimeError, match="collision"):
        evidence.write(output)


def test_spec_forbids_unknown_fields() -> None:
    payload = _payload()
    order = payload["order"]
    assert isinstance(order, dict)
    order["unknown"] = True

    with pytest.raises(ValidationError, match="Extra inputs"):
        NbboReplaySpec.model_validate(payload)


def test_replay_nbbo_cli_writes_hashed_evidence(tmp_path: Path) -> None:
    spec_file = tmp_path / "replay.json"
    output = tmp_path / "evidence.json"
    spec_file.write_text(json.dumps(_payload()), encoding="utf-8")

    result = CliRunner().invoke(
        app,
        [
            "backtest",
            "replay-nbbo",
            "--spec-file",
            str(spec_file),
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0
    command_result = json.loads(result.stdout)
    evidence = json.loads(output.read_bytes())
    assert command_result["sha256"]
    assert command_result["input_sha256"] == evidence["input_sha256"]
    assert command_result["filled_qty"] == 100
    assert evidence["result"]["read_quote_timestamps"]
