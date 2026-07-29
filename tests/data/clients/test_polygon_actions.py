"""Polygon/Massive corporate-action contract tests."""

from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING

import httpx
import pytest

from quant_earning_edge.data import BronzeWriter, LakehouseLayout
from quant_earning_edge.data.clients import (
    DividendDistributionType,
    PolygonClient,
    SplitAdjustmentType,
)
from quant_earning_edge.data.clients.errors import ProviderResponseError

if TYPE_CHECKING:
    from pathlib import Path


def test_splits_paginate_validate_sort_and_capture(tmp_path: Path) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if len(requests) == 1:
            return httpx.Response(
                200,
                json={
                    "status": "OK",
                    "results": [
                        {
                            "id": "split-2",
                            "ticker": "MSFT",
                            "execution_date": "2026-07-15",
                            "adjustment_type": "forward_split",
                            "split_from": 1,
                            "split_to": 2,
                            "historical_adjustment_factor": 0.5,
                        }
                    ],
                    "next_url": "https://api.polygon.io/stocks/v1/splits?cursor=abc",
                },
            )
        return httpx.Response(
            200,
            json={
                "status": "OK",
                "results": [
                    {
                        "id": "split-1",
                        "ticker": "AAPL",
                        "execution_date": "2026-07-10",
                        "adjustment_type": "reverse_split",
                        "split_from": 10,
                        "split_to": 1,
                    }
                ],
            },
        )

    with httpx.Client(
        base_url="https://api.polygon.io",
        transport=httpx.MockTransport(handler),
    ) as http_client:
        client = PolygonClient(
            api_key="secret",
            http_client=http_client,
            bronze_writer=BronzeWriter(LakehouseLayout(tmp_path)),
        )
        events = client.stock_splits(
            start_date=date(2026, 7, 1),
            end_date=date(2026, 7, 31),
        )

    assert [event.event_id for event in events] == ["split-1", "split-2"]
    assert events[0].adjustment_type is SplitAdjustmentType.REVERSE_SPLIT
    assert requests[0].url.params["execution_date.gte"] == "2026-07-01"
    assert requests[1].headers["Authorization"] == "Bearer secret"
    assert "cursor=abc" in str(requests[1].url)
    assert len(client.corporate_action_observation_artifacts) == 2


def test_dividends_validate_current_endpoint_fields() -> None:
    payload = {
        "status": "OK",
        "results": [
            {
                "id": "dividend-1",
                "ticker": "AAPL",
                "ex_dividend_date": "2026-07-10",
                "distribution_type": "recurring",
                "cash_amount": 0.26,
                "currency": "USD",
                "frequency": 4,
                "declaration_date": "2026-07-01",
                "record_date": "2026-07-10",
                "pay_date": "2026-07-15",
                "split_adjusted_cash_amount": 0.26,
                "historical_adjustment_factor": 0.997,
            }
        ],
    }
    with httpx.Client(
        base_url="https://api.polygon.io",
        transport=httpx.MockTransport(lambda _request: httpx.Response(200, json=payload)),
    ) as http_client:
        events = PolygonClient(api_key="secret", http_client=http_client).cash_dividends(
            start_date=date(2026, 7, 1),
            end_date=date(2026, 7, 31),
        )

    assert len(events) == 1
    assert events[0].distribution_type is DividendDistributionType.RECURRING
    assert events[0].cash_amount == 0.26


def test_corporate_actions_reject_duplicate_ids() -> None:
    item = {
        "id": "duplicate",
        "ticker": "AAPL",
        "execution_date": "2026-07-10",
        "adjustment_type": "forward_split",
        "split_from": 1,
        "split_to": 2,
    }
    payload = {"status": "OK", "results": [item, item]}
    with (
        httpx.Client(
            base_url="https://api.polygon.io",
            transport=httpx.MockTransport(lambda _request: httpx.Response(200, json=payload)),
        ) as http_client,
        pytest.raises(ProviderResponseError, match="duplicate corporate action"),
    ):
        PolygonClient(api_key="secret", http_client=http_client).stock_splits(
            start_date=date(2026, 7, 1),
            end_date=date(2026, 7, 31),
        )
