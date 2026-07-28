"""Paper orders are diagnostic smoke evidence beside primary NBBO replay."""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

import quant_earning_edge.cli as cli_module
from quant_earning_edge.backtest import NbboReplayEvidence, NbboReplaySpec, replay_order
from quant_earning_edge.cli import app
from quant_earning_edge.live import AlpacaPaperClient, BrokerOrder, PaperOrderReconciler
from quant_earning_edge.signals import FrozenDailyOrders


def _evidence() -> NbboReplayEvidence:
    decision = datetime(2025, 1, 2, 21, 30, tzinfo=UTC)
    submitted = datetime(2025, 1, 3, 14, 30, tzinfo=UTC)
    spec = NbboReplaySpec.model_validate(
        {
            "order": {
                "order_id": "entry-1",
                "ticker": "AAA",
                "side": "buy",
                "quantity": 100,
                "decision_time": decision,
                "submitted_at": submitted,
                "expires_at": submitted + timedelta(minutes=5),
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
                    "timestamp": submitted,
                    "sequence": 1,
                    "bid_price": 99.9,
                    "ask_price": 100.1,
                    "bid_size": 100,
                    "ask_size": 100,
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


def _broker_order(**overrides: object) -> BrokerOrder:
    values: dict[str, object] = {
        "id": "broker-1",
        "client_order_id": "entry-1",
        "symbol": "AAA",
        "asset_class": "us_equity",
        "qty": "100",
        "filled_qty": "100",
        "filled_avg_price": "100.12",
        "side": "buy",
        "type": "market",
        "time_in_force": "day",
        "status": "filled",
        "submitted_at": "2025-01-03T14:30:00Z",
        "filled_at": "2025-01-03T14:30:01Z",
    }
    values.update(overrides)
    return BrokerOrder.model_validate(values)


def test_terminal_paper_order_is_reconciled_as_diagnostic_only() -> None:
    evidence = _evidence()

    report = PaperOrderReconciler().evaluate(
        evidence=(evidence,),
        broker_orders=(_broker_order(),),
        session_date=date(2025, 1, 3),
        evaluated_at=datetime(2025, 1, 3, 21, 5, tzinfo=UTC),
    )

    order = report.orders[0]
    assert report.reconciliation_break_count == 0
    assert report.all_orders_terminal
    assert not report.paper_pnl_is_gate_input
    assert order.fill_quantity_difference == 0
    assert order.paper_vs_replay_price_bps is not None
    assert order.paper_vs_replay_price_bps > 0


def test_open_after_close_paper_order_is_an_operational_break() -> None:
    report = PaperOrderReconciler().evaluate(
        evidence=(_evidence(),),
        broker_orders=(
            _broker_order(
                status="partially_filled",
                filled_qty="50",
                filled_avg_price="100.11",
                filled_at=None,
            ),
        ),
        session_date=date(2025, 1, 3),
        evaluated_at=datetime(2025, 1, 3, 21, 5, tzinfo=UTC),
    )

    assert report.reconciliation_break_count == 1
    assert not report.all_orders_terminal
    assert report.orders[0].break_reasons == ("paper order is not terminal after close",)
    assert report.orders[0].fill_quantity_difference == -50


def test_reconciliation_requires_exact_client_order_id_set() -> None:
    with pytest.raises(ValueError, match="exactly match"):
        PaperOrderReconciler().evaluate(
            evidence=(_evidence(),),
            broker_orders=(_broker_order(client_order_id="different"),),
            session_date=date(2025, 1, 3),
            evaluated_at=datetime(2025, 1, 3, 21, 5, tzinfo=UTC),
        )


def test_frozen_reconciliation_fetches_only_verified_client_ids() -> None:
    evidence = _evidence()
    calls: list[str] = []

    class Lookup:
        def get_by_client_order_id(self, client_order_id: str) -> BrokerOrder:
            calls.append(client_order_id)
            return _broker_order(client_order_id=client_order_id)

    report = PaperOrderReconciler().fetch_and_evaluate(
        evidence=(evidence,),
        intended_orders=(evidence.result.order,),
        order_lookup=Lookup(),
        session_date=date(2025, 1, 3),
        evaluated_at=datetime(2025, 1, 3, 21, 5, tzinfo=UTC),
    )

    assert calls == ["entry-1"]
    assert report.reconciliation_break_count == 0


def test_frozen_reconciliation_refuses_changed_order_before_broker_fetch() -> None:
    evidence = _evidence()
    calls: list[str] = []

    class Lookup:
        def get_by_client_order_id(self, client_order_id: str) -> BrokerOrder:
            calls.append(client_order_id)
            return _broker_order()

    with pytest.raises(ValueError, match="order fields differ"):
        PaperOrderReconciler().fetch_and_evaluate(
            evidence=(evidence,),
            intended_orders=(replace(evidence.result.order, quantity=99),),
            order_lookup=Lookup(),
            session_date=date(2025, 1, 3),
            evaluated_at=datetime(2025, 1, 3, 21, 5, tzinfo=UTC),
        )

    assert calls == []


def test_paper_fill_decimal_fields_remain_exact() -> None:
    order = _broker_order(filled_avg_price="100.123456789")

    assert order.filled_average_price == Decimal("100.123456789")


def test_paper_reconciliation_cli_reloads_replay_evidence(tmp_path: Path) -> None:
    evidence = _evidence()
    evidence_file = tmp_path / "replay.json"
    evidence.write(evidence_file)
    spec_file = tmp_path / "paper-reconciliation.json"
    output = tmp_path / "paper-report.json"
    spec_file.write_text(
        json.dumps(
            {
                "session_date": "2025-01-03",
                "evaluated_at": "2025-01-03T21:05:00Z",
                "replay_evidence_files": ["replay.json"],
                "broker_orders": [_broker_order().model_dump(mode="json", by_alias=True)],
            }
        ),
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        app,
        [
            "paper",
            "reconcile",
            "--spec-file",
            str(spec_file),
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0
    command_result = json.loads(result.stdout)
    report = json.loads(output.read_bytes())
    assert command_result["reconciliation_break_count"] == 0
    assert not report["paper_pnl_is_gate_input"]
    assert report["replay_evidence_sha256"] == [evidence.sha256]


def test_frozen_reconciliation_cli_fetches_alpaca_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence = _evidence()
    evidence_file = tmp_path / "replay.json"
    evidence.write(evidence_file)
    frozen_path = tmp_path / "frozen.json"
    frozen_path.write_text("{}", encoding="utf-8")
    frozen = SimpleNamespace(
        intended_orders=(SimpleNamespace(to_domain=lambda: evidence.result.order),),
        trade_date=date(2025, 1, 3),
    )
    monkeypatch.setattr(FrozenDailyOrders, "load", staticmethod(lambda _: frozen))
    fetched: list[str] = []

    def fetch(_: AlpacaPaperClient, client_order_id: str) -> BrokerOrder:
        fetched.append(client_order_id)
        return _broker_order(client_order_id=client_order_id)

    monkeypatch.setattr(AlpacaPaperClient, "get_by_client_order_id", fetch)
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            (
                "APCA_API_KEY_ID=test-key",
                "APCA_API_SECRET_KEY=test-secret",
                f"DATA_LAKE_ROOT={tmp_path / 'lake'}",
            )
        ),
        encoding="utf-8",
    )
    output = tmp_path / "paper-report.json"

    result = CliRunner().invoke(
        app,
        [
            "paper",
            "reconcile-frozen",
            "--frozen-orders",
            str(frozen_path),
            "--evidence-file",
            str(evidence_file),
            "--output",
            str(output),
            "--env-file",
            str(env_file),
        ],
    )

    assert result.exit_code == 0
    assert fetched == ["entry-1"]

    revision_directory = tmp_path / "revisions"
    revision = CliRunner().invoke(
        app,
        [
            "paper",
            "reconcile-frozen-revision",
            "--frozen-orders",
            str(frozen_path),
            "--evidence-file",
            str(evidence_file),
            "--output-directory",
            str(revision_directory),
            "--env-file",
            str(env_file),
        ],
    )

    assert revision.exit_code == 0
    revision_path = Path(json.loads(revision.stdout)["output"])
    assert revision_path.parent == revision_directory.resolve()
    assert revision_path.name.startswith("paper-reconciliation-")
    assert FrozenDailyOrders.load(frozen_path) is frozen
    assert json.loads(result.stdout)["reconciliation_break_count"] == 0
    assert json.loads(output.read_bytes())["orders"][0]["client_order_id"] == "entry-1"


def test_content_addressed_reconciliation_can_resolve_on_later_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence = _evidence()
    broken = PaperOrderReconciler().evaluate(
        evidence=(evidence,),
        broker_orders=(
            _broker_order(
                status="partially_filled",
                filled_qty="50",
                filled_avg_price="100.11",
                filled_at=None,
            ),
        ),
        session_date=date(2025, 1, 3),
        evaluated_at=datetime(2025, 1, 3, 21, 5, tzinfo=UTC),
    )
    clean = PaperOrderReconciler().evaluate(
        evidence=(evidence,),
        broker_orders=(_broker_order(),),
        session_date=date(2025, 1, 3),
        evaluated_at=datetime(2025, 1, 3, 21, 10, tzinfo=UTC),
    )
    reports = iter((broken, clean))
    monkeypatch.setattr(
        cli_module,
        "_frozen_paper_reconciliation",
        lambda **_: next(reports),
    )
    frozen_path = tmp_path / "frozen.json"
    frozen_path.write_text("{}", encoding="utf-8")
    output_directory = tmp_path / "revisions"
    arguments = [
        "paper",
        "reconcile-frozen-revision",
        "--frozen-orders",
        str(frozen_path),
        "--output-directory",
        str(output_directory),
    ]

    first = CliRunner().invoke(app, arguments)
    second = CliRunner().invoke(app, arguments)

    assert first.exit_code == 1
    assert second.exit_code == 0
    revisions = tuple(output_directory.glob("paper-reconciliation-*.json"))
    assert len(revisions) == 2
