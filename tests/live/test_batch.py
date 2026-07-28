"""Session-level paper submission is restart-safe and explicitly supports no-trade days."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime
from typing import TYPE_CHECKING, Any

import httpx
import pytest

from quant_earning_edge.data.clients import ProviderRequestError
from quant_earning_edge.live import (
    AlpacaPaperClient,
    PaperBatchSubmitter,
    PaperOrderBatchSpec,
)
from quant_earning_edge.monitoring import (
    CircuitBreakerDecision,
    CircuitBreakerEvaluator,
    CircuitBreakerObservation,
)

if TYPE_CHECKING:
    from pathlib import Path


def _decision(now: datetime, *, session_date: date) -> CircuitBreakerDecision:
    return CircuitBreakerEvaluator().evaluate(
        (
            CircuitBreakerObservation(
                session_date=session_date,
                evaluated_at=now,
                replay_notional=100_000,
                replay_net_pnl=0,
                replay_fill_rate=1,
                polygon_data_observed_at=now,
                alpaca_data_observed_at=now,
            ),
        )
    )


def _spec() -> PaperOrderBatchSpec:
    return PaperOrderBatchSpec.model_validate(
        {
            "session_date": "2026-07-28",
            "orders": [
                {
                    "client_order_id": "batch-001",
                    "symbol": "AAA",
                    "quantity": 10,
                    "side": "buy",
                },
                {
                    "client_order_id": "batch-002",
                    "symbol": "BBB",
                    "quantity": 20,
                    "side": "sell",
                },
            ],
        }
    )


def _broker_payload(request: dict[str, Any], *, sequence: int) -> dict[str, Any]:
    return {
        "id": f"broker-{sequence}",
        "client_order_id": request["client_order_id"],
        "symbol": request["symbol"],
        "asset_class": "us_equity",
        "qty": request["qty"],
        "filled_qty": "0",
        "filled_avg_price": None,
        "side": request["side"],
        "type": request["type"],
        "time_in_force": request["time_in_force"],
        "limit_price": request.get("limit_price"),
        "extended_hours": request["extended_hours"],
        "status": "new",
        "submitted_at": "2026-07-28T13:30:00Z",
        "filled_at": None,
    }


def test_batch_resumes_by_client_id_after_mid_batch_provider_failure(tmp_path: Path) -> None:
    broker_orders: dict[str, dict[str, Any]] = {}
    fail_second = True
    post_ids: list[str] = []

    def respond(request: httpx.Request) -> httpx.Response:
        nonlocal fail_second
        if request.method == "GET":
            client_id = request.url.params["client_order_id"]
            if client_id in broker_orders:
                return httpx.Response(200, json=broker_orders[client_id])
            return httpx.Response(404)
        body = json.loads(request.content)
        client_id = str(body["client_order_id"])
        post_ids.append(client_id)
        if client_id == "batch-002" and fail_second:
            return httpx.Response(503)
        payload = _broker_payload(body, sequence=len(broker_orders) + 1)
        broker_orders[client_id] = payload
        return httpx.Response(200, json=payload)

    now = datetime(2026, 7, 28, 13, 25, tzinfo=UTC)
    with httpx.Client(
        base_url="https://paper-api.alpaca.markets",
        transport=httpx.MockTransport(respond),
    ) as http_client:
        submitter = PaperBatchSubmitter(
            AlpacaPaperClient(
                api_key_id="key",
                secret_key="secret",
                http_client=http_client,
            )
        )
        with pytest.raises(ProviderRequestError):
            submitter.submit(
                _spec(),
                breaker_decision=_decision(now, session_date=date(2026, 7, 28)),
                evaluated_at=now,
            )
        fail_second = False
        batch = submitter.submit(
            _spec(),
            breaker_decision=_decision(now, session_date=date(2026, 7, 28)),
            evaluated_at=now,
        )

    assert post_ids == ["batch-001", "batch-002", "batch-002"]
    assert [item.idempotent_reuse for item in batch.submissions] == [True, False]
    output = tmp_path / "batch.json"
    batch.write(output)
    batch.write(output)
    assert (
        json.loads(output.read_bytes())["breaker_decision_sha256"] == batch.breaker_decision_sha256
    )


def test_no_trade_batch_writes_evidence_without_broker_requests(tmp_path: Path) -> None:
    def reject(_: httpx.Request) -> httpx.Response:
        raise AssertionError("no-trade batch must not call the broker")

    now = datetime(2026, 7, 28, 13, 25, tzinfo=UTC)
    with httpx.Client(
        base_url="https://paper-api.alpaca.markets",
        transport=httpx.MockTransport(reject),
    ) as http_client:
        batch = PaperBatchSubmitter(
            AlpacaPaperClient(
                api_key_id="key",
                secret_key="secret",
                http_client=http_client,
            )
        ).submit(
            PaperOrderBatchSpec(session_date=date(2026, 7, 28)),
            breaker_decision=_decision(now, session_date=date(2026, 7, 28)),
            evaluated_at=now,
        )

    output = tmp_path / "no-trade.json"
    batch.write(output)
    assert batch.submissions == ()
    assert json.loads(output.read_bytes())["submissions"] == []


def test_batch_requires_unique_sorted_ids_and_matching_fresh_breaker() -> None:
    raw = _spec().model_dump(mode="json")
    raw["orders"].reverse()
    with pytest.raises(ValueError, match="unique and sorted"):
        PaperOrderBatchSpec.model_validate(raw)

    now = datetime(2026, 7, 28, 13, 25, tzinfo=UTC)
    with httpx.Client(base_url="https://paper-api.alpaca.markets") as http_client:
        submitter = PaperBatchSubmitter(
            AlpacaPaperClient(
                api_key_id="key",
                secret_key="secret",
                http_client=http_client,
            )
        )
        with pytest.raises(ValueError, match="session dates"):
            submitter.submit(
                PaperOrderBatchSpec(session_date=date(2026, 7, 29)),
                breaker_decision=_decision(now, session_date=date(2026, 7, 28)),
                evaluated_at=now,
            )
