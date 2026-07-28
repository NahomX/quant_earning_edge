"""Polygon minute-aggregate contract tests."""

from datetime import UTC, datetime

import httpx

from quant_earning_edge.data.clients import PolygonClient


def test_minute_bars_use_millisecond_interval_and_validate_payload() -> None:
    start = datetime(2026, 7, 28, 8, tzinfo=UTC)
    end = datetime(2026, 7, 28, 13, 25, tzinfo=UTC)
    timestamp_ms = int(datetime(2026, 7, 28, 13, 20, tzinfo=UTC).timestamp() * 1000)
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "ticker": "AAPL",
                "adjusted": True,
                "status": "OK",
                "results": [
                    {
                        "o": 101,
                        "h": 103,
                        "l": 100,
                        "c": 102,
                        "v": 10_000,
                        "t": timestamp_ms,
                    }
                ],
            },
        )

    with httpx.Client(
        base_url="https://api.polygon.io",
        transport=httpx.MockTransport(handler),
    ) as http_client:
        bars = PolygonClient(api_key="secret", http_client=http_client).minute_bars(
            symbol="aapl",
            start_at=start,
            end_at=end,
        )

    assert len(bars) == 1
    assert bars[0].close == 102
    assert "/range/1/minute/" in requests[0].url.path
    assert requests[0].url.params["adjusted"] == "true"
