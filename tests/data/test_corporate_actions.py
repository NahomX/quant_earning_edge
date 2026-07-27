"""End-to-end corporate-action ingestion and silver-schema tests."""

from __future__ import annotations

from datetime import UTC, date, datetime
from typing import TYPE_CHECKING

import httpx
import pyarrow.parquet as pq
import pytest

from quant_earning_edge.data import (
    DIVIDENDS_SCHEMA,
    SPLITS_SCHEMA,
    BronzeWriter,
    CorporateActionsIngestor,
    DuckDBStore,
    LakehouseLayout,
    SilverDataset,
    SilverWriter,
)
from quant_earning_edge.data.clients import PolygonClient

if TYPE_CHECKING:
    from pathlib import Path


def test_ingestion_writes_bronze_and_partitioned_silver(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/splits"):
            return httpx.Response(
                200,
                json={
                    "status": "OK",
                    "results": [
                        {
                            "id": "split-1",
                            "ticker": "AAPL",
                            "execution_date": "2026-07-10",
                            "adjustment_type": "forward_split",
                            "split_from": 1,
                            "split_to": 2,
                        }
                    ],
                },
            )
        return httpx.Response(
            200,
            json={
                "status": "OK",
                "results": [
                    {
                        "id": "dividend-1",
                        "ticker": "MSFT",
                        "ex_dividend_date": "2026-07-11",
                        "distribution_type": "recurring",
                        "cash_amount": 0.75,
                        "currency": "USD",
                        "frequency": 4,
                    }
                ],
            },
        )

    layout = LakehouseLayout(tmp_path)
    with httpx.Client(
        base_url="https://api.polygon.io",
        transport=httpx.MockTransport(handler),
    ) as http_client:
        result = CorporateActionsIngestor(
            client=PolygonClient(
                api_key="secret",
                http_client=http_client,
                bronze_writer=BronzeWriter(layout),
            ),
            silver_writer=SilverWriter(layout),
        ).ingest(
            start_date=date(2026, 7, 1),
            end_date=date(2026, 7, 31),
            ingested_at=datetime(2026, 7, 31, 22, tzinfo=UTC),
        )

    assert result.split_count == result.dividend_count == 1
    assert len(result.silver_artifacts) == 2
    assert result.silver_artifacts[0].schema == SPLITS_SCHEMA
    assert result.silver_artifacts[1].schema == DIVIDENDS_SCHEMA
    assert len(list((tmp_path / "bronze").rglob("*.json"))) == 2
    split = pq.read_table(result.silver_artifacts[0].path)  # type: ignore[no-untyped-call]
    dividend = pq.read_table(result.silver_artifacts[1].path)  # type: ignore[no-untyped-call]
    assert split.column("event_id").to_pylist() == ["split-1"]
    assert dividend.column("event_id").to_pylist() == ["dividend-1"]

    database = tmp_path / "research.duckdb"
    with DuckDBStore(database) as store:
        views = store.register_silver_views(
            layout,
            datasets=(SilverDataset.STOCK_SPLITS, SilverDataset.CASH_DIVIDENDS),
        )
        split_row = store.execute(
            "SELECT symbol, split_from, split_to FROM silver_stock_splits"
        ).fetchone()
        dividend_row = store.execute(
            "SELECT symbol, cash_amount FROM silver_cash_dividends"
        ).fetchone()

    assert views == ("silver_stock_splits", "silver_cash_dividends")
    assert split_row == ("AAPL", 1.0, 2.0)
    assert dividend_row == ("MSFT", 0.75)


def test_silver_views_fail_when_required_dataset_is_missing(tmp_path: Path) -> None:
    with DuckDBStore() as store, pytest.raises(FileNotFoundError, match="earnings_events"):
        store.register_silver_views(
            LakehouseLayout(tmp_path),
            datasets=(SilverDataset.EARNINGS_EVENTS,),
        )
