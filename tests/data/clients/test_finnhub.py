"""Contract and failure-path tests for the Finnhub client."""

from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING

import httpx
import pytest

from quant_earning_edge.data import BronzeWriter, LakehouseLayout
from quant_earning_edge.data.clients import (
    EarningsTiming,
    FinnhubClient,
    ProviderRequestError,
    ProviderResponseError,
)

if TYPE_CHECKING:
    from pathlib import Path


def _client(
    handler: httpx.MockTransport,
    *,
    bronze_writer: BronzeWriter | None = None,
    max_attempts: int = 3,
    sleeps: list[float] | None = None,
) -> tuple[FinnhubClient, httpx.Client]:
    http_client = httpx.Client(
        base_url="https://finnhub.io/api/v1",
        transport=handler,
        timeout=5,
    )
    client = FinnhubClient(
        api_key="test-key",
        http_client=http_client,
        bronze_writer=bronze_writer,
        max_attempts=max_attempts,
        retry_delay_seconds=0.5,
        sleeper=(sleeps.append if sleeps is not None else lambda _: None),
    )
    return client, http_client


def test_earnings_calendar_validates_contract_and_captures_bronze(tmp_path: Path) -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        assert request.url.params["from"] == "2026-07-27"
        assert request.url.params["to"] == "2026-07-28"
        assert request.url.params["symbol"] == "AAPL"
        assert request.url.params["international"] == "false"
        assert request.headers["X-Finnhub-Token"] == "test-key"
        return httpx.Response(
            200,
            json={
                "earningsCalendar": [
                    {
                        "date": "2026-07-28",
                        "epsActual": None,
                        "epsEstimate": 1.42,
                        "hour": "amc",
                        "quarter": 3,
                        "revenueActual": None,
                        "revenueEstimate": 98_000_000_000,
                        "symbol": "AAPL",
                        "year": 2026,
                    }
                ]
            },
        )

    client, http_client = _client(
        httpx.MockTransport(respond),
        bronze_writer=BronzeWriter(LakehouseLayout(tmp_path)),
    )
    with http_client:
        events = client.earnings_calendar(
            start_date=date(2026, 7, 27),
            end_date=date(2026, 7, 28),
            symbol="aapl",
        )

    assert len(events) == 1
    assert events[0].symbol == "AAPL"
    assert events[0].timing is EarningsTiming.AFTER_MARKET_CLOSE
    assert events[0].eps_estimate == 1.42
    assert len(client.earnings_observation_artifacts) == 1
    assert len(list((tmp_path / "bronze").rglob("*.json"))) == 1


def test_retries_rate_limit_with_bounded_backoff() -> None:
    attempts = 0
    sleeps: list[float] = []

    def respond(_: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            return httpx.Response(429, json={"error": "rate limit"})
        return httpx.Response(200, json={"earningsCalendar": []})

    client, http_client = _client(httpx.MockTransport(respond), sleeps=sleeps)
    with http_client:
        assert (
            client.earnings_calendar(
                start_date=date(2026, 7, 27),
                end_date=date(2026, 7, 27),
            )
            == ()
        )

    assert attempts == 3
    assert sleeps == [0.5, 1.0]


def test_exhausted_retryable_failures_raise() -> None:
    client, http_client = _client(
        httpx.MockTransport(lambda _: httpx.Response(503)),
        max_attempts=2,
    )
    with http_client, pytest.raises(ProviderRequestError, match="after 2 attempts"):
        client.earnings_calendar(
            start_date=date(2026, 7, 27),
            end_date=date(2026, 7, 27),
        )


def test_nonretryable_http_failure_raises_immediately() -> None:
    attempts = 0

    def respond(_: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(401)

    client, http_client = _client(httpx.MockTransport(respond))
    with http_client, pytest.raises(ProviderRequestError, match="HTTP 401"):
        client.earnings_calendar(
            start_date=date(2026, 7, 27),
            end_date=date(2026, 7, 27),
        )
    assert attempts == 1


@pytest.mark.parametrize(
    "payload",
    [
        {"unexpected": []},
        {"earningsCalendar": [{"date": "bad"}]},
    ],
)
def test_invalid_contract_raises(payload: object) -> None:
    client, http_client = _client(httpx.MockTransport(lambda _: httpx.Response(200, json=payload)))
    with http_client, pytest.raises(ProviderResponseError, match="failed validation"):
        client.earnings_calendar(
            start_date=date(2026, 7, 27),
            end_date=date(2026, 7, 27),
        )


def test_invalid_json_raises() -> None:
    client, http_client = _client(
        httpx.MockTransport(
            lambda _: httpx.Response(
                200,
                content=b"not-json",
                headers={"content-type": "application/json"},
            )
        )
    )
    with http_client, pytest.raises(ProviderResponseError, match="invalid JSON"):
        client.earnings_calendar(
            start_date=date(2026, 7, 27),
            end_date=date(2026, 7, 27),
        )


def test_rejects_inverted_date_range() -> None:
    client, http_client = _client(httpx.MockTransport(lambda _: httpx.Response(200, json={})))
    with http_client, pytest.raises(ValueError, match="end_date"):
        client.earnings_calendar(
            start_date=date(2026, 7, 28),
            end_date=date(2026, 7, 27),
        )
