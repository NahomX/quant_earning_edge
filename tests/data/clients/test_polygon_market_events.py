"""Contract coverage for Polygon historical stock NBBO and trade events."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import httpx
import pytest

from quant_earning_edge.data import BronzeWriter, LakehouseLayout
from quant_earning_edge.data.clients import PolygonClient, ProviderResponseError

if TYPE_CHECKING:
    from pathlib import Path


def _nanoseconds(timestamp: datetime) -> int:
    return int(timestamp.timestamp()) * 1_000_000_000 + timestamp.microsecond * 1_000


def test_stock_quotes_use_bounded_sip_time_and_capture_pages(tmp_path: Path) -> None:
    start = datetime(2026, 7, 28, 13, 30, 0, 123456, tzinfo=UTC)
    end = start + timedelta(seconds=10)
    requests: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if "cursor" not in request.url.params:
            assert request.url.path == "/v3/quotes/AAPL"
            assert request.url.params["timestamp.gte"] == str(_nanoseconds(start))
            assert request.url.params["timestamp.lte"] == str(_nanoseconds(end))
            assert request.url.params["sort"] == "timestamp"
            assert request.url.params["order"] == "asc"
            assert request.url.params["limit"] == "50000"
            return httpx.Response(
                200,
                json={
                    "status": "OK",
                    "results": [
                        {
                            "sip_timestamp": _nanoseconds(start),
                            "participant_timestamp": _nanoseconds(start) - 1_000,
                            "sequence_number": 10,
                            "bid_price": 100.0,
                            "ask_price": 100.1,
                            "bid_size": 200,
                            "ask_size": 300,
                            "bid_exchange": 11,
                            "ask_exchange": 12,
                            "conditions": [1],
                        }
                    ],
                    "next_url": "https://api.polygon.io/v3/quotes/AAPL?cursor=next",
                },
            )
        return httpx.Response(
            200,
            json={
                "status": "OK",
                "results": [
                    {
                        "sip_timestamp": _nanoseconds(start + timedelta(seconds=1)),
                        "sequence_number": 11,
                        "bid_price": 100.01,
                        "ask_price": 100.11,
                        "bid_size": 100,
                        "ask_size": 100,
                    }
                ],
            },
        )

    http_client = httpx.Client(
        base_url="https://api.polygon.io",
        transport=httpx.MockTransport(respond),
    )
    with http_client:
        quotes = PolygonClient(
            api_key="key",
            http_client=http_client,
            bronze_writer=BronzeWriter(LakehouseLayout(tmp_path)),
        ).stock_quotes(symbol=" aapl ", start_at=start, end_at=end)

    assert len(requests) == 2
    assert [quote.sequence_number for quote in quotes] == [10, 11]
    assert quotes[0].timestamp == start
    assert quotes[0].participant_timestamp == start - timedelta(microseconds=1)
    assert quotes[0].conditions == (1,)
    assert all(quote.is_replayable for quote in quotes)
    assert len(list((tmp_path / "bronze").rglob("*.json"))) == 2


def test_one_sided_quote_is_preserved_but_not_replayable() -> None:
    start = datetime(2026, 7, 28, 13, 30, tzinfo=UTC)
    response = {
        "status": "OK",
        "results": [
            {
                "sip_timestamp": _nanoseconds(start),
                "sequence_number": 1,
                "bid_price": 100,
                "bid_size": 100,
            }
        ],
    }
    http_client = httpx.Client(
        base_url="https://api.polygon.io",
        transport=httpx.MockTransport(lambda _: httpx.Response(200, json=response)),
    )
    with http_client:
        quotes = PolygonClient(api_key="key", http_client=http_client).stock_quotes(
            symbol="AAPL",
            start_at=start,
            end_at=start + timedelta(seconds=1),
        )

    assert len(quotes) == 1
    assert not quotes[0].is_replayable


def test_crossed_quote_is_rejected() -> None:
    start = datetime(2026, 7, 28, 13, 30, tzinfo=UTC)
    response = {
        "status": "OK",
        "results": [
            {
                "sip_timestamp": _nanoseconds(start),
                "sequence_number": 1,
                "bid_price": 101,
                "ask_price": 100,
                "bid_size": 100,
                "ask_size": 100,
            }
        ],
    }
    http_client = httpx.Client(
        base_url="https://api.polygon.io",
        transport=httpx.MockTransport(lambda _: httpx.Response(200, json=response)),
    )
    with http_client, pytest.raises(ProviderResponseError, match="failed validation"):
        PolygonClient(api_key="key", http_client=http_client).stock_quotes(
            symbol="AAPL",
            start_at=start,
            end_at=start + timedelta(seconds=1),
        )


def test_stock_trades_preserve_conditions_and_fractional_size() -> None:
    start = datetime(2026, 7, 28, 13, 30, tzinfo=UTC)
    response = {
        "status": "OK",
        "results": [
            {
                "sip_timestamp": _nanoseconds(start),
                "participant_timestamp": _nanoseconds(start) - 2_000,
                "sequence_number": 5,
                "price": 100.05,
                "size": 10.5,
                "exchange": 11,
                "id": "trade-1",
                "conditions": [12, 41],
                "correction": 0,
            }
        ],
    }
    http_client = httpx.Client(
        base_url="https://api.polygon.io",
        transport=httpx.MockTransport(lambda _: httpx.Response(200, json=response)),
    )
    with http_client:
        trades = PolygonClient(api_key="key", http_client=http_client).stock_trades(
            symbol="AAPL",
            start_at=start,
            end_at=start + timedelta(seconds=1),
        )

    assert len(trades) == 1
    assert trades[0].size == 10.5
    assert trades[0].conditions == (12, 41)
    assert trades[0].trade_id == "trade-1"
    assert trades[0].participant_timestamp == start - timedelta(microseconds=2)


def test_duplicate_market_event_identity_across_pages_is_rejected() -> None:
    start = datetime(2026, 7, 28, 13, 30, tzinfo=UTC)
    page = 0

    def respond(_: httpx.Request) -> httpx.Response:
        nonlocal page
        page += 1
        return httpx.Response(
            200,
            json={
                "status": "OK",
                "results": [
                    {
                        "sip_timestamp": _nanoseconds(start),
                        "sequence_number": 1,
                        "price": 100,
                        "size": 100,
                        "exchange": 11,
                        "id": f"trade-{page}",
                    }
                ],
                "next_url": (
                    "https://api.polygon.io/v3/trades/AAPL?cursor=next" if page == 1 else None
                ),
            },
        )

    http_client = httpx.Client(
        base_url="https://api.polygon.io",
        transport=httpx.MockTransport(respond),
    )
    with http_client, pytest.raises(ProviderResponseError, match="duplicate stock trade"):
        PolygonClient(api_key="key", http_client=http_client).stock_trades(
            symbol="AAPL",
            start_at=start,
            end_at=start + timedelta(seconds=1),
        )


def test_market_event_outside_requested_sip_interval_is_rejected() -> None:
    start = datetime(2026, 7, 28, 13, 30, tzinfo=UTC)
    response = {
        "status": "OK",
        "results": [
            {
                "sip_timestamp": _nanoseconds(start - timedelta(microseconds=1)),
                "sequence_number": 1,
                "bid_price": 100,
                "ask_price": 100.1,
                "bid_size": 100,
                "ask_size": 100,
            }
        ],
    }
    http_client = httpx.Client(
        base_url="https://api.polygon.io",
        transport=httpx.MockTransport(lambda _: httpx.Response(200, json=response)),
    )
    with http_client, pytest.raises(ProviderResponseError, match="out of range"):
        PolygonClient(api_key="key", http_client=http_client).stock_quotes(
            symbol="AAPL",
            start_at=start,
            end_at=start + timedelta(seconds=1),
        )
