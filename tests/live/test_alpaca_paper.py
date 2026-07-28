"""Paper-host isolation and idempotent Alpaca order submission."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime
from typing import TYPE_CHECKING

import httpx
import pytest
from typer.testing import CliRunner

from quant_earning_edge.cli import app
from quant_earning_edge.data import BronzeWriter, LakehouseLayout
from quant_earning_edge.data.clients import ProviderResponseError
from quant_earning_edge.live import AlpacaPaperClient, PaperOrderRequest
from quant_earning_edge.monitoring import CircuitBreakerEvaluator, CircuitBreakerObservation

if TYPE_CHECKING:
    from pathlib import Path


def _order_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "id": "broker-order-1",
        "client_order_id": "qee-entry-1",
        "symbol": "AAPL",
        "asset_class": "us_equity",
        "qty": "10",
        "filled_qty": "0",
        "filled_avg_price": None,
        "side": "buy",
        "type": "limit",
        "time_in_force": "day",
        "limit_price": "100.25",
        "extended_hours": False,
        "status": "new",
        "submitted_at": "2026-07-28T13:29:59Z",
        "filled_at": None,
    }
    payload.update(overrides)
    return payload


def _request() -> PaperOrderRequest:
    return PaperOrderRequest(
        client_order_id="qee-entry-1",
        symbol="aapl",
        quantity=10,
        side="buy",
        order_type="limit",
        limit_price=100.25,
    )


def test_client_refuses_live_or_noncanonical_hosts() -> None:
    for base_url in (
        "https://api.alpaca.markets",
        "https://paper-api.alpaca.markets/proxy",
        "http://paper-api.alpaca.markets",
    ):
        with (
            httpx.Client(base_url=base_url) as http_client,
            pytest.raises(
                ValueError,
                match="paper order client requires exactly",
            ),
        ):
            AlpacaPaperClient(
                api_key_id="key",
                secret_key="secret",
                http_client=http_client,
            )


def test_submit_checks_client_id_then_places_one_paper_order(tmp_path: Path) -> None:
    requests: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        assert request.headers["APCA-API-KEY-ID"] == "key"
        assert request.headers["APCA-API-SECRET-KEY"] == "secret"
        if request.method == "GET":
            assert request.url.path == "/v2/orders:by_client_order_id"
            assert request.url.params["client_order_id"] == "qee-entry-1"
            return httpx.Response(404)
        assert request.method == "POST"
        assert request.url.path == "/v2/orders"
        body = json.loads(request.content)
        assert body == {
            "client_order_id": "qee-entry-1",
            "extended_hours": False,
            "limit_price": "100.25",
            "qty": "10",
            "side": "buy",
            "symbol": "AAPL",
            "time_in_force": "day",
            "type": "limit",
        }
        return httpx.Response(
            200,
            json=_order_payload(),
            headers={"X-Request-ID": "request-123"},
        )

    http_client = httpx.Client(
        base_url="https://paper-api.alpaca.markets",
        transport=httpx.MockTransport(respond),
    )
    with http_client:
        submission = AlpacaPaperClient(
            api_key_id="key",
            secret_key="secret",
            http_client=http_client,
        ).submit(_request())

    output = tmp_path / "submission.json"
    submission.write(output)
    submission.write(output)
    assert len(requests) == 2
    assert not submission.idempotent_reuse
    assert submission.provider_request_id == "request-123"
    assert submission.broker_order.symbol == "AAPL"
    assert json.loads(output.read_bytes())["request"]["client_order_id"] == "qee-entry-1"


def test_existing_identical_client_order_id_is_reused_without_post() -> None:
    methods: list[str] = []

    def respond(request: httpx.Request) -> httpx.Response:
        methods.append(request.method)
        return httpx.Response(200, json=_order_payload())

    http_client = httpx.Client(
        base_url="https://paper-api.alpaca.markets",
        transport=httpx.MockTransport(respond),
    )
    with http_client:
        submission = AlpacaPaperClient(
            api_key_id="key",
            secret_key="secret",
            http_client=http_client,
        ).submit(_request())

    assert methods == ["GET"]
    assert submission.idempotent_reuse


def test_existing_client_order_id_with_different_order_is_rejected() -> None:
    http_client = httpx.Client(
        base_url="https://paper-api.alpaca.markets",
        transport=httpx.MockTransport(lambda _: httpx.Response(200, json=_order_payload(qty="11"))),
    )
    with http_client, pytest.raises(ProviderResponseError, match="does not match"):
        AlpacaPaperClient(
            api_key_id="key",
            secret_key="secret",
            http_client=http_client,
        ).submit(_request())


def test_broker_fill_quantity_and_price_must_reconcile() -> None:
    http_client = httpx.Client(
        base_url="https://paper-api.alpaca.markets",
        transport=httpx.MockTransport(
            lambda _: httpx.Response(
                200,
                json=_order_payload(filled_qty="5", filled_avg_price=None),
            )
        ),
    )
    with http_client, pytest.raises(ProviderResponseError, match="failed validation"):
        AlpacaPaperClient(
            api_key_id="key",
            secret_key="secret",
            http_client=http_client,
        ).get_by_client_order_id("qee-entry-1")


def test_after_close_order_fetch_is_captured_in_bronze(tmp_path: Path) -> None:
    http_client = httpx.Client(
        base_url="https://paper-api.alpaca.markets",
        transport=httpx.MockTransport(
            lambda _: httpx.Response(200, json=_order_payload(status="filled"))
        ),
    )
    layout = LakehouseLayout(tmp_path / "lake")
    with http_client:
        order = AlpacaPaperClient(
            api_key_id="key",
            secret_key="secret",
            http_client=http_client,
            bronze_writer=BronzeWriter(layout),
        ).get_by_client_order_id("qee-entry-1")

    artifacts = tuple(
        layout.bronze(
            source="alpaca-paper",
            dataset="orders",
            event_date=order.submitted_at.date(),
        ).glob("*.json")
    )
    assert len(artifacts) == 1
    assert json.loads(artifacts[0].read_bytes())["client_order_id"] == "qee-entry-1"


def test_submit_cli_refuses_halted_breaker_before_network(tmp_path: Path) -> None:
    now = datetime.now(UTC)
    decision = CircuitBreakerEvaluator().evaluate(
        (
            CircuitBreakerObservation(
                session_date=date.today(),
                evaluated_at=now,
                replay_notional=100_000,
                replay_net_pnl=-3_000,
                replay_fill_rate=1.0,
                polygon_data_observed_at=now,
                alpaca_data_observed_at=now,
            ),
        )
    )
    breaker_file = tmp_path / "breaker.json"
    decision.write(breaker_file)
    spec_file = tmp_path / "order.json"
    spec_file.write_text(_request().model_dump_json(), encoding="utf-8")
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "APCA_API_KEY_ID=test-key",
                "APCA_API_SECRET_KEY=test-secret",
                f"DATA_LAKE_ROOT={tmp_path / 'lake'}",
            ]
        ),
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        app,
        [
            "paper",
            "submit-order",
            "--spec-file",
            str(spec_file),
            "--breaker-decision",
            str(breaker_file),
            "--output",
            str(tmp_path / "submission.json"),
            "--env-file",
            str(env_file),
        ],
    )

    assert result.exit_code == 2
    assert "halts new orders" in result.stderr
