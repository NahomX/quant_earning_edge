"""Contract tests for the authenticated Alpaca calendar boundary."""

from datetime import date

import httpx
import pytest

from quant_earning_edge.data.clients import AlpacaCalendarClient
from quant_earning_edge.data.clients.errors import ProviderResponseError


def test_calendar_preserves_early_close_and_auth_headers() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["APCA-API-KEY-ID"] == "key-id"
        assert request.headers["APCA-API-SECRET-KEY"] == "secret"
        assert request.url.params["start"] == "2026-11-27"
        return httpx.Response(
            200,
            json=[{"date": "2026-11-27", "open": "09:30", "close": "13:00"}],
        )

    with httpx.Client(
        base_url="https://paper-api.alpaca.markets",
        transport=httpx.MockTransport(handler),
    ) as http_client:
        sessions = AlpacaCalendarClient(
            api_key_id="key-id",
            secret_key="secret",
            http_client=http_client,
        ).sessions(start_date=date(2026, 11, 27), end_date=date(2026, 11, 27))

    assert len(sessions) == 1
    assert sessions[0].session_date == date(2026, 11, 27)
    assert sessions[0].open_at.hour == 9
    assert sessions[0].close_at.hour == 13
    assert sessions[0].open_at.utcoffset() is not None


def test_calendar_rejects_duplicate_or_unsorted_dates() -> None:
    payload = [
        {"date": "2026-07-28", "open": "09:30", "close": "16:00"},
        {"date": "2026-07-27", "open": "09:30", "close": "16:00"},
    ]
    transport = httpx.MockTransport(lambda _request: httpx.Response(200, json=payload))
    with (
        httpx.Client(
            base_url="https://paper-api.alpaca.markets",
            transport=transport,
        ) as http_client,
        pytest.raises(ProviderResponseError, match="unique and ascending"),
    ):
        AlpacaCalendarClient(
            api_key_id="key-id",
            secret_key="secret",
            http_client=http_client,
        ).sessions(start_date=date(2026, 7, 27), end_date=date(2026, 7, 28))
