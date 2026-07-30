"""Silver schema, persistence, and end-to-end earnings ingestion tests."""

from __future__ import annotations

from datetime import UTC, date, datetime
from typing import TYPE_CHECKING

import httpx
import pyarrow.parquet as pq
import pytest

from quant_earning_edge.data import (
    DAILY_BARS_SCHEMA,
    EARNINGS_SCHEMA,
    BarsIngestor,
    BronzeWriter,
    DailyBarsSourceCapture,
    DailyBarsSourceManifest,
    DuckDBStore,
    EarningsIngestor,
    EarningsSourceCapture,
    LakehouseLayout,
    SilverWriter,
)
from quant_earning_edge.data.clients import (
    EarningsEvent,
    EquityBar,
    FinnhubClient,
    PolygonClient,
)

if TYPE_CHECKING:
    from pathlib import Path


def _event(*, symbol: str, event_date: date) -> EarningsEvent:
    return EarningsEvent.model_validate(
        {
            "date": event_date.isoformat(),
            "symbol": symbol,
            "hour": "amc",
            "year": event_date.year,
            "quarter": 3,
            "epsActual": 1.5,
            "epsEstimate": 1.4,
            "revenueActual": 100.0,
            "revenueEstimate": 95.0,
        }
    )


def _bar(*, symbol: str, day: int) -> EquityBar:
    return EquityBar(
        symbol=symbol,
        timestamp=datetime(2026, 7, day, 4, tzinfo=UTC),
        open=100,
        high=102,
        low=99,
        close=101,
        volume=1_000_000,
        vwap=100.5,
        transactions=50_000,
        adjusted=True,
    )


def test_silver_writer_partitions_sorts_and_uses_explicit_schema(tmp_path: Path) -> None:
    writer = SilverWriter(LakehouseLayout(tmp_path))
    ingested_at = datetime(2026, 7, 27, 18, tzinfo=UTC)

    artifacts = writer.write_earnings(
        (
            _event(symbol="msft", event_date=date(2026, 7, 28)),
            _event(symbol="aapl", event_date=date(2026, 7, 28)),
            _event(symbol="goog", event_date=date(2026, 7, 29)),
        ),
        ingested_at=ingested_at,
    )

    assert len(artifacts) == 2
    assert artifacts[0].row_count == 2
    assert artifacts[0].schema == EARNINGS_SCHEMA
    assert "date=2026-07-28" in artifacts[0].path.as_posix()
    table = pq.read_table(artifacts[0].path)  # type: ignore[no-untyped-call]
    assert table.column("symbol").to_pylist() == ["AAPL", "MSFT"]
    assert table.column("source").to_pylist() == ["finnhub", "finnhub"]


def test_silver_writer_is_idempotent_for_same_observation(tmp_path: Path) -> None:
    writer = SilverWriter(LakehouseLayout(tmp_path))
    ingested_at = datetime(2026, 7, 27, 18, tzinfo=UTC)
    events = (_event(symbol="AAPL", event_date=date(2026, 7, 28)),)

    first = writer.write_earnings(events, ingested_at=ingested_at)
    second = writer.write_earnings(events, ingested_at=ingested_at)

    assert first == second
    assert len(list((tmp_path / "silver").rglob("*.parquet"))) == 1


def test_silver_writer_rejects_naive_ingestion_time(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        SilverWriter(LakehouseLayout(tmp_path)).write_earnings(
            (_event(symbol="AAPL", event_date=date(2026, 7, 28)),),
            ingested_at=datetime(2026, 7, 27),
        )


def test_duckdb_queries_partitioned_silver_parquet(tmp_path: Path) -> None:
    artifact = SilverWriter(LakehouseLayout(tmp_path)).write_earnings(
        (_event(symbol="AAPL", event_date=date(2026, 7, 28)),),
        ingested_at=datetime(2026, 7, 27, 18, tzinfo=UTC),
    )[0]

    with DuckDBStore() as store:
        row = store.parquet_relation(artifact.path).project("symbol, event_date, date").fetchone()

    assert row == ("AAPL", date(2026, 7, 28), date(2026, 7, 28))


def test_earnings_ingestor_captures_bronze_and_writes_silver(tmp_path: Path) -> None:
    payload = {
        "earningsCalendar": [
            {
                "date": "2026-07-28",
                "symbol": " aapl ",
                "hour": "bmo",
                "year": 2026,
                "quarter": 3,
            }
        ]
    }
    http_client = httpx.Client(
        base_url="https://finnhub.io/api/v1",
        transport=httpx.MockTransport(lambda _: httpx.Response(200, json=payload)),
    )
    layout = LakehouseLayout(tmp_path)
    client = FinnhubClient(
        api_key="test-key",
        http_client=http_client,
        bronze_writer=BronzeWriter(layout),
    )

    with http_client:
        result = EarningsIngestor(
            client=client,
            silver_writer=SilverWriter(layout),
            source_capture=EarningsSourceCapture(layout),
        ).ingest(
            start_date=date(2026, 7, 28),
            end_date=date(2026, 7, 28),
            ingested_at=datetime(2026, 7, 27, 18, tzinfo=UTC),
        )

    assert result.event_count == 1
    assert len(result.silver_artifacts) == 1
    assert result.source_manifest is not None
    assert len(list((tmp_path / "bronze").rglob("*.json"))) == 1
    silver = pq.read_table(  # type: ignore[no-untyped-call]
        result.silver_artifacts[0].path
    )
    assert silver.column("symbol").to_pylist() == ["AAPL"]


def test_empty_earnings_response_writes_explicit_audit_partition(tmp_path: Path) -> None:
    http_client = httpx.Client(
        base_url="https://finnhub.io/api/v1",
        transport=httpx.MockTransport(lambda _: httpx.Response(200, json={"earningsCalendar": []})),
    )
    layout = LakehouseLayout(tmp_path)
    with http_client:
        result = EarningsIngestor(
            client=FinnhubClient(
                api_key="test-key",
                http_client=http_client,
                bronze_writer=BronzeWriter(layout),
            ),
            silver_writer=SilverWriter(layout),
        ).ingest(
            start_date=date(2026, 7, 27),
            end_date=date(2026, 7, 28),
            ingested_at=datetime(2026, 7, 27, 22, tzinfo=UTC),
        )

    assert result.event_count == 0
    assert len(result.silver_artifacts) == 1
    assert result.silver_artifacts[0].row_count == 0
    assert "date=2026-07-28" in result.silver_artifacts[0].path.as_posix()
    assert pq.read_schema(result.silver_artifacts[0].path) == EARNINGS_SCHEMA  # type: ignore[no-untyped-call]


def test_daily_bars_writer_uses_schema_partitions_and_is_idempotent(tmp_path: Path) -> None:
    writer = SilverWriter(LakehouseLayout(tmp_path))
    ingested_at = datetime(2026, 7, 27, 18, tzinfo=UTC)
    bars = (
        _bar(symbol="MSFT", day=28),
        _bar(symbol="AAPL", day=28),
    )

    first = writer.write_daily_bars(bars, ingested_at=ingested_at)
    second = writer.write_daily_bars(bars, ingested_at=ingested_at)

    assert first == second
    assert first[0].schema == DAILY_BARS_SCHEMA
    assert first[0].row_count == 2
    table = pq.read_table(first[0].path)  # type: ignore[no-untyped-call]
    assert table.column("symbol").to_pylist() == ["AAPL", "MSFT"]
    assert table.column("adjusted").to_pylist() == [True, True]
    assert table.column("available_at").to_pylist() == [ingested_at, ingested_at]
    assert len(list((tmp_path / "silver").rglob("*.parquet"))) == 1


def test_bars_ingestor_captures_bronze_and_writes_silver(tmp_path: Path) -> None:
    timestamp = int(datetime(2026, 7, 28, 4, tzinfo=UTC).timestamp() * 1000)
    payload = {
        "ticker": "AAPL",
        "adjusted": True,
        "status": "OK",
        "results": [
            {
                "o": 100,
                "h": 102,
                "l": 99,
                "c": 101,
                "v": 1_000_000,
                "t": timestamp,
            }
        ],
    }
    http_client = httpx.Client(
        base_url="https://api.polygon.io",
        transport=httpx.MockTransport(lambda _: httpx.Response(200, json=payload)),
    )
    layout = LakehouseLayout(tmp_path)
    with http_client:
        result = BarsIngestor(
            client=PolygonClient(
                api_key="test-key",
                http_client=http_client,
                bronze_writer=BronzeWriter(layout),
            ),
            silver_writer=SilverWriter(layout),
            source_capture=DailyBarsSourceCapture(layout),
        ).ingest(
            symbol="aapl",
            start_date=date(2026, 7, 28),
            end_date=date(2026, 7, 28),
            ingested_at=datetime(2026, 7, 28, 22, tzinfo=UTC),
        )

    assert result.symbol == "AAPL"
    assert result.bar_count == 1
    assert len(result.silver_artifacts) == 1
    assert result.source_manifest is not None
    manifest = DailyBarsSourceManifest.load(result.source_manifest)
    assert DailyBarsSourceCapture.find_for_files(
        tuple(item.path for item in result.silver_artifacts),
        data_lake_root=tmp_path,
    ) == (manifest,)
    reproduced = DailyBarsSourceCapture.reproduce(
        manifest,
        data_lake_root=tmp_path,
        output_layout=LakehouseLayout(tmp_path / "reproduced"),
    )
    assert tuple(item.sha256 for item in reproduced) == tuple(
        item.sha256 for item in result.silver_artifacts
    )
    assert len(list((tmp_path / "bronze").rglob("*.json"))) == 1
    assert len(list((tmp_path / "silver").rglob("*.parquet"))) == 1
    manifest.provider_paths(data_lake_root=tmp_path)[0].write_bytes(b"changed")
    with pytest.raises(ValueError, match="missing or differs"):
        DailyBarsSourceCapture.reproduce(
            manifest,
            data_lake_root=tmp_path,
            output_layout=LakehouseLayout(tmp_path / "tampered"),
        )
