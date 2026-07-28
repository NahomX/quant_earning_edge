"""Provider freshness uses provider-native timestamps and canonical hosts."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

import httpx

from quant_earning_edge.data import BronzeWriter, LakehouseLayout
from quant_earning_edge.monitoring import (
    ProviderFreshnessEvidence,
    ProviderFreshnessProbe,
)

if TYPE_CHECKING:
    from pathlib import Path


def test_freshness_probe_validates_provider_timestamps_and_captures_bronze(
    tmp_path: Path,
) -> None:
    evaluated = datetime(2026, 7, 29, 13, 20, tzinfo=UTC)
    polygon_observed = datetime(2026, 7, 29, 13, 19, 30, tzinfo=UTC)
    alpaca_observed = datetime(2026, 7, 29, 13, 19, 45, tzinfo=UTC)

    def polygon_response(request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"] == "Bearer polygon-key"
        assert request.url.path.endswith("/SPY")
        return httpx.Response(
            200,
            json={
                "status": "OK",
                "request_id": "polygon-request",
                "ticker": {
                    "ticker": "SPY",
                    "updated": int(polygon_observed.timestamp() * 1_000_000_000),
                },
            },
        )

    def alpaca_response(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v2/clock"
        assert request.headers["APCA-API-KEY-ID"] == "alpaca-key"
        return httpx.Response(
            200,
            json={"timestamp": alpaca_observed.isoformat()},
            headers={"X-Request-ID": "alpaca-request"},
        )

    polygon_http = httpx.Client(
        base_url="https://api.polygon.io",
        transport=httpx.MockTransport(polygon_response),
    )
    alpaca_http = httpx.Client(
        base_url="https://paper-api.alpaca.markets",
        transport=httpx.MockTransport(alpaca_response),
    )
    with polygon_http, alpaca_http:
        evidence = ProviderFreshnessProbe(
            polygon_api_key="polygon-key",
            alpaca_api_key_id="alpaca-key",
            alpaca_secret_key="alpaca-secret",
            polygon_http=polygon_http,
            alpaca_http=alpaca_http,
            bronze_writer=BronzeWriter(LakehouseLayout(tmp_path / "lake")),
        ).probe(symbol="spy", evaluated_at=evaluated)

    output = tmp_path / "freshness.json"
    evidence.write(output)
    evidence.write(output)
    assert ProviderFreshnessEvidence.load(output) == evidence
    assert evidence.polygon_data_observed_at == polygon_observed
    assert evidence.alpaca_data_observed_at == alpaca_observed
    assert evidence.polygon_request_id == "polygon-request"
    assert len(tuple((tmp_path / "lake" / "bronze").rglob("*.json"))) == 2
