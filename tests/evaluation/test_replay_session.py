"""Daily lifecycle reconciliation for immutable NBBO replay evidence."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from quant_earning_edge.backtest import NbboReplayEvidence, NbboReplaySpec, replay_order
from quant_earning_edge.cli import app
from quant_earning_edge.evaluation import (
    ReplayRoundTrip,
    ReplaySessionAggregator,
    ReplaySessionReport,
)
from quant_earning_edge.signals import FrozenDailyOrders, strategy_file_sha256


def _evidence(
    *,
    order_id: str,
    side: str,
    submitted_at: datetime,
    bid: float,
    ask: float,
    displayed_size: int = 100,
) -> NbboReplayEvidence:
    decision = datetime(2025, 1, 2, 21, 30, tzinfo=UTC)
    spec = NbboReplaySpec.model_validate(
        {
            "order": {
                "order_id": order_id,
                "ticker": "AAA",
                "side": side,
                "quantity": 100,
                "decision_time": decision,
                "submitted_at": submitted_at,
                "expires_at": submitted_at + timedelta(minutes=5),
                "average_daily_volume_shares": 1_000_000,
            },
            "decision_snapshot": {
                "ticker": "AAA",
                "observed_at": decision,
                "bid_price": 99.9,
                "ask_price": 100.1,
                "bid_size": 100,
                "ask_size": 100,
                "last_trade_price": 100,
                "last_trade_at": decision - timedelta(seconds=1),
            },
            "quotes": [
                {
                    "ticker": "AAA",
                    "timestamp": submitted_at,
                    "sequence": 1,
                    "bid_price": bid,
                    "ask_price": ask,
                    "bid_size": displayed_size,
                    "ask_size": displayed_size,
                }
            ],
        }
    )
    order, snapshot, quotes, trades, config = spec.domain_inputs()
    result = replay_order(
        order,
        decision_snapshot=snapshot,
        quotes=quotes,
        trades=trades,
        config=config,
    )
    return NbboReplayEvidence.build(spec=spec, result=result)


def _full_round_trip() -> tuple[NbboReplayEvidence, NbboReplayEvidence]:
    return (
        _evidence(
            order_id="entry",
            side="buy",
            submitted_at=datetime(2025, 1, 3, 14, 30, tzinfo=UTC),
            bid=99.9,
            ask=100.1,
        ),
        _evidence(
            order_id="exit",
            side="sell",
            submitted_at=datetime(2025, 1, 3, 20, 0, tzinfo=UTC),
            bid=104.9,
            ask=105.1,
        ),
    )


def _frozen_round_trip() -> tuple[NbboReplayEvidence, NbboReplayEvidence]:
    return (
        _evidence(
            order_id="earnings-trade-1-entry",
            side="buy",
            submitted_at=datetime(2025, 1, 3, 14, 30, tzinfo=UTC),
            bid=99.9,
            ask=100.1,
        ),
        _evidence(
            order_id="earnings-trade-1-exit",
            side="sell",
            submitted_at=datetime(2025, 1, 3, 20, 0, tzinfo=UTC),
            bid=104.9,
            ask=105.1,
        ),
    )


def test_reconciled_session_computes_fill_slippage_and_net_pnl() -> None:
    evidence = _full_round_trip()

    report = ReplaySessionAggregator().evaluate(
        evidence=evidence,
        round_trips=(ReplayRoundTrip("trade-1", "entry", "exit", "long"),),
        session_date=date(2025, 1, 3),
        initial_cash=100_000,
    )

    assert report.intended_order_count == 2
    assert report.fully_filled_order_rate == 1
    assert report.share_fill_rate == 1
    assert report.reconciliation_break_count == 0
    assert report.gross_pnl is not None and report.gross_pnl > 0
    assert report.arrival_gross_pnl is not None
    assert report.arrival_gross_pnl - report.realized_execution_slippage_cost == pytest.approx(
        report.gross_pnl
    )
    assert report.realized_execution_slippage_cost == pytest.approx(
        report.modeled_spread_cost
        + report.modeled_market_impact_cost
        + report.execution_residual_cost
    )
    assert report.net_pnl is not None
    assert report.net_pnl == pytest.approx(report.gross_pnl - report.commission)
    assert report.net_return == pytest.approx(report.net_pnl / 100_000)
    assert report.realized_adverse_slippage_bps_p90 is not None
    assert report.predicted_adverse_slippage_bps_p90 is not None
    assert report.round_trips[0].matched_quantity == 100


def test_unmatched_partial_exit_creates_reconciliation_break() -> None:
    entry, _ = _full_round_trip()
    partial_exit = _evidence(
        order_id="exit",
        side="sell",
        submitted_at=datetime(2025, 1, 3, 20, 0, tzinfo=UTC),
        bid=104.9,
        ask=105.1,
        displayed_size=40,
    )

    report = ReplaySessionAggregator().evaluate(
        evidence=(entry, partial_exit),
        round_trips=(ReplayRoundTrip("trade-1", "entry", "exit", "long"),),
        session_date=date(2025, 1, 3),
        initial_cash=100_000,
    )

    assert report.share_fill_rate == 0.7
    assert report.reconciliation_break_count == 1
    assert report.round_trips[0].unmatched_quantity == 60
    assert report.gross_pnl is None
    assert report.net_pnl is None
    assert report.net_return is None


def test_round_trip_requires_exact_order_mapping() -> None:
    evidence = _full_round_trip()

    with pytest.raises(ValueError, match="exactly match"):
        ReplaySessionAggregator().evaluate(
            evidence=evidence,
            round_trips=(ReplayRoundTrip("trade-1", "entry", "missing", "long"),),
            session_date=date(2025, 1, 3),
            initial_cash=100_000,
        )


def test_no_trade_day_is_valid_zero_return_uptime_evidence() -> None:
    report = ReplaySessionAggregator().evaluate(
        evidence=(),
        round_trips=(),
        session_date=date(2025, 1, 3),
        initial_cash=100_000,
    )

    assert report.intended_order_count == 0
    assert report.fully_filled_order_rate is None
    assert report.share_fill_rate is None
    assert report.reconciliation_break_count == 0
    assert report.gross_pnl == 0
    assert report.net_pnl == 0
    assert report.net_return == 0


def test_frozen_orders_derive_exact_long_round_trip() -> None:
    evidence = _frozen_round_trip()

    report = ReplaySessionAggregator().evaluate_frozen_long_orders(
        evidence=evidence,
        intended_orders=tuple(item.result.order for item in evidence),
        session_date=date(2025, 1, 3),
        initial_cash=100_000,
        commission_bps_per_side=1,
    )

    assert report.reconciliation_break_count == 0
    assert len(report.round_trips) == 1
    assert report.round_trips[0].trade_id == "earnings-trade-1"
    assert report.round_trips[0].side == "long"


def test_frozen_session_rejects_non_lifecycle_order_id() -> None:
    evidence = _full_round_trip()

    with pytest.raises(ValueError, match="must end"):
        ReplaySessionAggregator().evaluate_frozen_long_orders(
            evidence=evidence,
            intended_orders=tuple(item.result.order for item in evidence),
            session_date=date(2025, 1, 3),
            initial_cash=100_000,
            commission_bps_per_side=1,
        )


def test_session_report_is_immutable_and_cli_reloads_evidence(tmp_path: Path) -> None:
    entry, exit_fill = _full_round_trip()
    entry_path = tmp_path / "entry.json"
    exit_path = tmp_path / "exit.json"
    entry.write(entry_path)
    exit_fill.write(exit_path)
    spec_path = tmp_path / "session.json"
    output = tmp_path / "report.json"
    spec_path.write_text(
        json.dumps(
            {
                "session_date": "2025-01-03",
                "initial_cash": 100_000,
                "evidence_files": ["entry.json", "exit.json"],
                "round_trips": [
                    {
                        "trade_id": "trade-1",
                        "entry_order_id": "entry",
                        "exit_order_id": "exit",
                        "side": "long",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        app,
        [
            "evaluation",
            "replay-session",
            "--aggregation-spec",
            str(spec_path),
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0
    command_result = json.loads(result.stdout)
    report = json.loads(output.read_bytes())
    assert command_result["sha256"]
    assert command_result["reconciliation_break_count"] == 0
    assert report["schema_version"] == 2
    assert report["evidence_sha256"] == sorted((entry.sha256, exit_fill.sha256))
    assert ReplaySessionReport.load(output).sha256 == command_result["sha256"]


def test_session_loader_rejects_tampered_reconciliation(tmp_path: Path) -> None:
    evidence = _full_round_trip()
    report = ReplaySessionAggregator().evaluate(
        evidence=evidence,
        round_trips=(ReplayRoundTrip("trade-1", "entry", "exit", "long"),),
        session_date=date(2025, 1, 3),
        initial_cash=100_000,
    )
    raw = json.loads(report.canonical_bytes)
    raw["filled_share_count"] -= 1
    output = tmp_path / "tampered.json"
    output.write_text(
        json.dumps(raw, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="invalid replay-session report"):
        ReplaySessionReport.load(output)


def test_frozen_replay_session_cli_uses_frozen_equity_and_strategy_cost(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence = _frozen_round_trip()
    evidence_paths = []
    for index, item in enumerate(evidence):
        path = tmp_path / f"evidence-{index}.json"
        item.write(path)
        evidence_paths.append(path)
    frozen_path = tmp_path / "frozen.json"
    frozen_path.write_text("{}", encoding="utf-8")
    strategy_path = Path(__file__).parents[2] / "configs/strategies/earnings_v1.yaml"
    frozen = SimpleNamespace(
        intended_orders=tuple(
            SimpleNamespace(to_domain=lambda order=item.result.order: order) for item in evidence
        ),
        trade_date=date(2025, 1, 3),
        portfolio=SimpleNamespace(equity=123_456.0),
        strategy_config_sha256=strategy_file_sha256(strategy_path),
    )
    monkeypatch.setattr(FrozenDailyOrders, "load", staticmethod(lambda _: frozen))
    output = tmp_path / "session-report.json"

    result = CliRunner().invoke(
        app,
        [
            "evaluation",
            "replay-frozen-session",
            "--frozen-orders",
            str(frozen_path),
            "--strategy-config",
            str(strategy_path),
            "--evidence-file",
            str(evidence_paths[0]),
            "--evidence-file",
            str(evidence_paths[1]),
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0
    report = ReplaySessionReport.load(output)
    assert report.initial_cash == 123_456
    assert report.round_trips[0].trade_id == "earnings-trade-1"
    assert json.loads(result.stdout)["sha256"] == report.sha256
