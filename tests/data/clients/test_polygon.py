"""Contract, pagination, and failure-path tests for Polygon aggregates."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from typing import TYPE_CHECKING

import httpx
import pytest

from quant_earning_edge.data import BronzeWriter, LakehouseLayout
from quant_earning_edge.data.clients import (
    PolygonClient,
    ProviderRequestError,
    ProviderResponseError,
)

if TYPE_CHECKING:
    from pathlib import Path


def _aggregate(*, day: int, close: float = 101.0) -> dict[str, float | int]:
    timestamp = datetime(2026, 7, day, 4, tzinfo=UTC)
    return {
        "o": 100.0,
        "h": 102.0,
        "l": 99.0,
        "c": close,
        "v": 1_000_000,
        "vw": 100.5,
        "n": 50_000,
        "t": int(timestamp.timestamp() * 1000),
    }


def test_daily_bars_paginates_validates_and_captures_each_page(tmp_path: Path) -> None:
    requests: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        assert request.headers["Authorization"] == "Bearer test-key"
        if "cursor" not in request.url.params:
            assert request.url.path.endswith(
                "/v2/aggs/ticker/AAPL/range/1/day/2026-07-27/2026-07-28"
            )
            assert request.url.params["adjusted"] == "true"
            assert request.url.params["sort"] == "asc"
            assert request.url.params["limit"] == "50000"
            return httpx.Response(
                200,
                json={
                    "ticker": "AAPL",
                    "adjusted": True,
                    "status": "OK",
                    "results": [_aggregate(day=27)],
                    "next_url": "https://api.polygon.io/v2/aggs/ticker/AAPL?cursor=next",
                },
            )
        return httpx.Response(
            200,
            json={
                "ticker": "AAPL",
                "adjusted": True,
                "status": "OK",
                "results": [_aggregate(day=28, close=101.5)],
            },
        )

    http_client = httpx.Client(
        base_url="https://api.polygon.io",
        transport=httpx.MockTransport(respond),
    )
    with http_client:
        bars = PolygonClient(
            api_key="test-key",
            http_client=http_client,
            bronze_writer=BronzeWriter(LakehouseLayout(tmp_path)),
        ).daily_bars(
            symbol=" aapl ",
            start_date=date(2026, 7, 27),
            end_date=date(2026, 7, 28),
        )

    assert len(requests) == 2
    assert [bar.session_date for bar in bars] == [date(2026, 7, 27), date(2026, 7, 28)]
    assert bars[1].close == 101.5
    assert bars[0].adjusted
    assert len(list((tmp_path / "bronze").rglob("*.json"))) == 2


def test_daily_bars_retries_rate_limit() -> None:
    attempts = 0
    sleeps: list[float] = []

    def respond(_: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(429)
        return httpx.Response(
            200,
            json={"ticker": "AAPL", "adjusted": True, "status": "OK", "results": []},
        )

    http_client = httpx.Client(
        base_url="https://api.polygon.io",
        transport=httpx.MockTransport(respond),
    )
    with http_client:
        bars = PolygonClient(
            api_key="key",
            http_client=http_client,
            retry_delay_seconds=0.5,
            sleeper=sleeps.append,
        ).daily_bars(
            symbol="AAPL",
            start_date=date(2026, 7, 27),
            end_date=date(2026, 7, 27),
        )

    assert bars == ()
    assert attempts == 2
    assert sleeps == [0.5]


@pytest.mark.parametrize(
    ("response", "message"),
    [
        (
            {"ticker": "MSFT", "adjusted": True, "status": "OK", "results": []},
            "did not match",
        ),
        (
            {"ticker": "AAPL", "adjusted": False, "status": "OK", "results": []},
            "not split-adjusted",
        ),
        (
            {"ticker": "AAPL", "adjusted": True, "status": "ERROR", "results": []},
            "status",
        ),
    ],
)
def test_rejects_invalid_response_metadata(
    response: dict[str, object],
    message: str,
) -> None:
    http_client = httpx.Client(
        base_url="https://api.polygon.io",
        transport=httpx.MockTransport(lambda _: httpx.Response(200, json=response)),
    )
    with http_client, pytest.raises(ProviderResponseError, match=message):
        PolygonClient(api_key="key", http_client=http_client).daily_bars(
            symbol="AAPL",
            start_date=date(2026, 7, 27),
            end_date=date(2026, 7, 27),
        )


def test_rejects_cross_host_pagination_url() -> None:
    response = {
        "ticker": "AAPL",
        "adjusted": True,
        "status": "OK",
        "results": [],
        "next_url": "https://attacker.example/steal",
    }
    http_client = httpx.Client(
        base_url="https://api.polygon.io",
        transport=httpx.MockTransport(lambda _: httpx.Response(200, json=response)),
    )
    with http_client, pytest.raises(ProviderResponseError, match="API host"):
        PolygonClient(api_key="key", http_client=http_client).daily_bars(
            symbol="AAPL",
            start_date=date(2026, 7, 27),
            end_date=date(2026, 7, 27),
        )


def test_rejects_duplicate_timestamps_across_pages() -> None:
    page = 0

    def respond(_: httpx.Request) -> httpx.Response:
        nonlocal page
        page += 1
        return httpx.Response(
            200,
            json={
                "ticker": "AAPL",
                "adjusted": True,
                "status": "OK",
                "results": [_aggregate(day=27)],
                "next_url": ("https://api.polygon.io/page?cursor=next" if page == 1 else None),
            },
        )

    http_client = httpx.Client(
        base_url="https://api.polygon.io",
        transport=httpx.MockTransport(respond),
    )
    with http_client, pytest.raises(ProviderResponseError, match="duplicate"):
        PolygonClient(api_key="key", http_client=http_client).daily_bars(
            symbol="AAPL",
            start_date=date(2026, 7, 27),
            end_date=date(2026, 7, 28),
        )


def test_rejects_invalid_ohlc_envelope() -> None:
    invalid = _aggregate(day=27)
    invalid["h"] = 98.0
    response = {
        "ticker": "AAPL",
        "adjusted": True,
        "status": "OK",
        "results": [invalid],
    }
    http_client = httpx.Client(
        base_url="https://api.polygon.io",
        transport=httpx.MockTransport(lambda _: httpx.Response(200, json=response)),
    )
    with http_client, pytest.raises(ProviderResponseError, match="failed validation"):
        PolygonClient(api_key="key", http_client=http_client).daily_bars(
            symbol="AAPL",
            start_date=date(2026, 7, 27),
            end_date=date(2026, 7, 27),
        )


def test_nonretryable_http_error_uses_shared_provider_error() -> None:
    http_client = httpx.Client(
        base_url="https://api.polygon.io",
        transport=httpx.MockTransport(lambda _: httpx.Response(403)),
    )
    with http_client, pytest.raises(ProviderRequestError, match="HTTP 403"):
        PolygonClient(api_key="key", http_client=http_client).daily_bars(
            symbol="AAPL",
            start_date=date(2026, 7, 27),
            end_date=date(2026, 7, 27),
        )


def test_ticker_details_are_explicitly_point_in_time_and_captured(tmp_path: Path) -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v3/reference/tickers/AAPL"
        assert request.url.params["date"] == "2026-07-27"
        return httpx.Response(
            200,
            json={
                "status": "OK",
                "results": {
                    "ticker": "AAPL",
                    "name": "Apple Inc.",
                    "active": True,
                    "locale": "us",
                    "market": "stocks",
                    "primary_exchange": "XNAS",
                    "type": "CS",
                    "market_cap": 3_000_000_000_000,
                    "sic_code": "3571",
                    "list_date": "1980-12-12",
                },
            },
        )

    http_client = httpx.Client(
        base_url="https://api.polygon.io",
        transport=httpx.MockTransport(respond),
    )
    with http_client:
        details = PolygonClient(
            api_key="key",
            http_client=http_client,
            bronze_writer=BronzeWriter(LakehouseLayout(tmp_path)),
        ).ticker_details(symbol="aapl", asof_date=date(2026, 7, 27))

    assert details.symbol == "AAPL"
    assert details.asof_date == date(2026, 7, 27)
    assert details.primary_exchange == "XNAS"
    assert details.market_cap == 3_000_000_000_000
    assert details.sic_code == "3571"
    assert len(list((tmp_path / "bronze").rglob("*.json"))) == 1


def test_ticker_snapshot_uses_provider_nbbo_and_trade_timestamps(tmp_path: Path) -> None:
    captured = datetime(2026, 7, 28, 1, 30, tzinfo=UTC)
    quote_time = captured - timedelta(seconds=2)
    trade_time = captured - timedelta(seconds=3)

    def respond(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/AAPL")
        return httpx.Response(
            200,
            json={
                "status": "OK",
                "request_id": "snapshot-request",
                "ticker": {
                    "ticker": "AAPL",
                    "lastQuote": {
                        "P": 100.1,
                        "S": 200,
                        "p": 99.9,
                        "s": 150,
                        "t": int(quote_time.timestamp() * 1_000_000_000),
                    },
                    "lastTrade": {
                        "p": 100.0,
                        "t": int(trade_time.timestamp() * 1_000_000_000),
                    },
                },
            },
        )

    http_client = httpx.Client(
        base_url="https://api.polygon.io",
        transport=httpx.MockTransport(respond),
    )
    with http_client:
        snapshot = PolygonClient(
            api_key="key",
            http_client=http_client,
            bronze_writer=BronzeWriter(LakehouseLayout(tmp_path)),
        ).ticker_snapshot(symbol="aapl", captured_at=captured)

    assert snapshot.symbol == "AAPL"
    assert snapshot.observed_at == quote_time
    assert snapshot.last_trade_at == trade_time
    assert snapshot.bid_price == 99.9
    assert snapshot.ask_price == 100.1
    assert snapshot.request_id == "snapshot-request"
    assert len(snapshot.payload_sha256) == 64


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (
            {
                "status": "OK",
                "results": {
                    "ticker": "MSFT",
                    "name": "Microsoft",
                    "active": True,
                    "locale": "us",
                    "market": "stocks",
                    "primary_exchange": "XNAS",
                    "type": "CS",
                    "market_cap": 1_000_000_000,
                },
            },
            "did not match",
        ),
        (
            {
                "status": "ERROR",
                "results": {
                    "ticker": "AAPL",
                    "name": "Apple",
                    "active": True,
                    "locale": "us",
                    "market": "stocks",
                    "primary_exchange": "XNAS",
                    "type": "CS",
                    "market_cap": 1_000_000_000,
                },
            },
            "status",
        ),
        ({"status": "OK", "results": {"ticker": "AAPL"}}, "failed validation"),
    ],
)
def test_ticker_details_reject_invalid_contract(
    payload: dict[str, object],
    message: str,
) -> None:
    http_client = httpx.Client(
        base_url="https://api.polygon.io",
        transport=httpx.MockTransport(lambda _: httpx.Response(200, json=payload)),
    )
    with http_client, pytest.raises(ProviderResponseError, match=message):
        PolygonClient(api_key="key", http_client=http_client).ticker_details(
            symbol="AAPL",
            asof_date=date(2026, 7, 27),
        )


def test_list_tickers_paginates_historical_date_and_captures_bronze(
    tmp_path: Path,
) -> None:
    requests: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if "cursor" not in request.url.params:
            assert request.url.path == "/v3/reference/tickers"
            assert request.url.params["date"] == "2026-07-27"
            assert request.url.params["active"] == "true"
            assert request.url.params["market"] == "stocks"
            return httpx.Response(
                200,
                json={
                    "status": "OK",
                    "results": [
                        {
                            "ticker": "MSFT",
                            "name": "Microsoft",
                            "active": True,
                            "locale": "us",
                            "market": "stocks",
                            "primary_exchange": "XNAS",
                            "type": "CS",
                        }
                    ],
                    "next_url": "https://api.polygon.io/v3/reference/tickers?cursor=next",
                },
            )
        return httpx.Response(
            200,
            json={
                "status": "OK",
                "results": [
                    {
                        "ticker": "AAPL",
                        "name": "Apple",
                        "active": True,
                        "locale": "us",
                        "market": "stocks",
                        "primary_exchange": "XNAS",
                        "type": "CS",
                    }
                ],
            },
        )

    http_client = httpx.Client(
        base_url="https://api.polygon.io",
        transport=httpx.MockTransport(respond),
    )
    with http_client:
        references = PolygonClient(
            api_key="key",
            http_client=http_client,
            bronze_writer=BronzeWriter(LakehouseLayout(tmp_path)),
        ).list_tickers(asof_date=date(2026, 7, 27))

    assert [reference.symbol for reference in references] == ["AAPL", "MSFT"]
    assert all(reference.asof_date == date(2026, 7, 27) for reference in references)
    assert len(requests) == 2
    assert len(list((tmp_path / "bronze").rglob("*.json"))) == 2


def test_list_tickers_rejects_duplicate_symbol_across_pages() -> None:
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
                        "ticker": "AAPL",
                        "name": "Apple",
                        "active": True,
                        "locale": "us",
                        "market": "stocks",
                    }
                ],
                "next_url": (
                    "https://api.polygon.io/v3/reference/tickers?cursor=next" if page == 1 else None
                ),
            },
        )

    http_client = httpx.Client(
        base_url="https://api.polygon.io",
        transport=httpx.MockTransport(respond),
    )
    with http_client, pytest.raises(ProviderResponseError, match="duplicate ticker"):
        PolygonClient(api_key="key", http_client=http_client).list_tickers(
            asof_date=date(2026, 7, 27)
        )
