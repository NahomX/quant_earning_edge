"""Silver persistence and conservative replay normalization for tick data."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from typing import TYPE_CHECKING

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from quant_earning_edge.data import (
    NBBO_QUOTES_SCHEMA,
    STOCK_TRADES_SCHEMA,
    DuckDBStore,
    LakehouseLayout,
    ReplayMarketDataLoader,
    SilverDataset,
    SilverWriter,
)
from quant_earning_edge.data.clients import StockQuote, StockTrade

if TYPE_CHECKING:
    from pathlib import Path


def _events() -> tuple[tuple[StockQuote, ...], tuple[StockTrade, ...], datetime]:
    opened = datetime(2026, 7, 28, 13, 30, tzinfo=UTC)
    quotes = (
        StockQuote(
            symbol="AAPL",
            timestamp=opened,
            sequence_number=1,
            bid_price=100.0,
            ask_price=100.1,
            bid_size=100.9,
            ask_size=200.9,
            conditions=(1,),
        ),
        StockQuote(
            symbol="AAPL",
            timestamp=opened + timedelta(microseconds=1),
            sequence_number=2,
            bid_price=100.0,
            ask_price=0.0,
            bid_size=100,
            ask_size=0,
        ),
    )
    trades = (
        StockTrade(
            symbol="AAPL",
            timestamp=opened,
            sequence_number=10,
            price=100.05,
            size=100,
            exchange=11,
            trade_id="opening",
            conditions=(2,),
            correction=0,
        ),
        StockTrade(
            symbol="AAPL",
            timestamp=opened + timedelta(microseconds=1),
            sequence_number=11,
            price=100.06,
            size=200,
            exchange=11,
            trade_id="corrected",
            correction=1,
        ),
        StockTrade(
            symbol="AAPL",
            timestamp=opened + timedelta(microseconds=2),
            sequence_number=12,
            price=100.07,
            size=0.5,
            exchange=11,
            trade_id="subshare",
        ),
        StockTrade(
            symbol="AAPL",
            timestamp=opened + timedelta(microseconds=3),
            sequence_number=13,
            price=100.08,
            size=10.75,
            exchange=11,
            trade_id="fractional",
        ),
    )
    return quotes, trades, opened


def test_market_events_round_trip_to_conservative_replay_inputs(tmp_path: Path) -> None:
    quotes, trades, opened = _events()
    writer = SilverWriter(LakehouseLayout(tmp_path))
    ingested_at = opened + timedelta(hours=8)
    quote_artifact = writer.write_stock_quotes(
        quotes,
        event_date=date(2026, 7, 28),
        ingested_at=ingested_at,
    )
    trade_artifact = writer.write_stock_trades(
        trades,
        event_date=date(2026, 7, 28),
        ingested_at=ingested_at,
    )

    events = ReplayMarketDataLoader().load(
        quote_files=(quote_artifact.path,),
        trade_files=(trade_artifact.path,),
        symbol="aapl",
        start_at=opened,
        end_at=opened + timedelta(seconds=1),
        opening_auction_condition_codes=frozenset({2}),
    )

    assert quote_artifact.schema == NBBO_QUOTES_SCHEMA
    assert trade_artifact.schema == STOCK_TRADES_SCHEMA
    assert quote_artifact.row_count == 2
    assert trade_artifact.row_count == 4
    assert events.source_quote_count == 2
    assert events.rejected_one_sided_quote_count == 1
    assert len(events.quotes) == 1
    assert events.quotes[0].bid_size == 100
    assert events.quotes[0].ask_size == 200
    assert events.source_trade_count == 4
    assert events.rejected_corrected_trade_count == 1
    assert events.rejected_subshare_trade_count == 1
    assert events.fractional_share_quantity_discarded == pytest.approx(1.25)
    assert [trade.size for trade in events.trades] == [100, 10]
    assert events.trades[0].is_opening_auction
    assert not events.trades[1].is_opening_auction
    assert events.auction_classification_configured

    with DuckDBStore() as store:
        views = store.register_silver_views(
            LakehouseLayout(tmp_path),
            datasets=(SilverDataset.NBBO_QUOTES, SilverDataset.STOCK_TRADES),
        )
        quote_count = store.execute("SELECT count(*) FROM silver_nbbo_quotes").fetchone()
        trade_count = store.execute("SELECT count(*) FROM silver_stock_trades").fetchone()
    assert views == ("silver_nbbo_quotes", "silver_stock_trades")
    assert quote_count == (2,)
    assert trade_count == (4,)


def test_market_event_silver_is_idempotent_for_same_observation(tmp_path: Path) -> None:
    quotes, trades, opened = _events()
    writer = SilverWriter(LakehouseLayout(tmp_path))
    ingested_at = opened + timedelta(hours=8)

    first_quote = writer.write_stock_quotes(
        quotes,
        event_date=date(2026, 7, 28),
        ingested_at=ingested_at,
    )
    second_quote = writer.write_stock_quotes(
        tuple(reversed(quotes)),
        event_date=date(2026, 7, 28),
        ingested_at=ingested_at,
    )
    first_trade = writer.write_stock_trades(
        trades,
        event_date=date(2026, 7, 28),
        ingested_at=ingested_at,
    )
    second_trade = writer.write_stock_trades(
        tuple(reversed(trades)),
        event_date=date(2026, 7, 28),
        ingested_at=ingested_at,
    )

    assert first_quote.path == second_quote.path
    assert first_trade.path == second_trade.path
    assert pq.read_schema(first_quote.path) == NBBO_QUOTES_SCHEMA  # type: ignore[no-untyped-call]
    assert pq.read_schema(first_trade.path) == STOCK_TRADES_SCHEMA  # type: ignore[no-untyped-call]


def test_market_event_writer_rejects_wrong_partition_date(tmp_path: Path) -> None:
    quotes, _, opened = _events()

    with pytest.raises(ValueError, match="does not match event_date"):
        SilverWriter(LakehouseLayout(tmp_path)).write_stock_quotes(
            quotes,
            event_date=date(2026, 7, 29),
            ingested_at=opened,
        )


def test_loader_detects_tampered_replayability_flag(tmp_path: Path) -> None:
    quotes, _, opened = _events()
    artifact = SilverWriter(LakehouseLayout(tmp_path)).write_stock_quotes(
        quotes,
        event_date=date(2026, 7, 28),
        ingested_at=opened,
    )
    table = pq.read_table(artifact.path)  # type: ignore[no-untyped-call]
    rows = table.to_pylist()
    rows[0]["is_replayable"] = False
    tampered = tmp_path / "tampered.parquet"
    pq.write_table(  # type: ignore[no-untyped-call]
        pa.Table.from_pylist(rows, schema=NBBO_QUOTES_SCHEMA),
        tampered,
    )

    with pytest.raises(ValueError, match="replayability flag"):
        ReplayMarketDataLoader().load(
            quote_files=(tampered,),
            trade_files=(),
            symbol="AAPL",
            start_at=opened,
            end_at=opened + timedelta(seconds=1),
        )
