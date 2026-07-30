"""Orchestration services that move provider observations into lake tiers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from datetime import date
    from pathlib import Path

    from quant_earning_edge.data.bars_source import DailyBarsSourceCapture
    from quant_earning_edge.data.clients.finnhub import FinnhubClient
    from quant_earning_edge.data.clients.polygon import PolygonClient
    from quant_earning_edge.data.earnings_source import EarningsSourceCapture
    from quant_earning_edge.data.silver import SilverArtifact, SilverWriter


@dataclass(frozen=True)
class EarningsIngestionResult:
    """Auditable result of one Finnhub earnings interval ingestion."""

    start_date: date
    end_date: date
    event_count: int
    silver_artifacts: tuple[SilverArtifact, ...]
    source_manifest: Path | None = None


@dataclass(frozen=True)
class BarsIngestionResult:
    """Auditable result of one Polygon symbol/date-range ingestion."""

    symbol: str
    start_date: date
    end_date: date
    bar_count: int
    silver_artifacts: tuple[SilverArtifact, ...]
    source_manifest: Path | None = None


@dataclass(frozen=True)
class CorporateActionsIngestionResult:
    """Auditable result of one Polygon corporate-action interval."""

    start_date: date
    end_date: date
    split_count: int
    dividend_count: int
    silver_artifacts: tuple[SilverArtifact, ...]


@dataclass(frozen=True)
class MarketEventsIngestionResult:
    """Auditable result of one Polygon quote/trade interval ingestion."""

    symbol: str
    event_date: date
    start_at: datetime
    end_at: datetime
    quote_count: int
    trade_count: int
    quote_artifact: SilverArtifact
    trade_artifact: SilverArtifact


class EarningsIngestor:
    """Fetch validated events and persist them to the silver tier."""

    def __init__(
        self,
        *,
        client: FinnhubClient,
        silver_writer: SilverWriter,
        source_capture: EarningsSourceCapture | None = None,
    ) -> None:
        self._client = client
        self._silver_writer = silver_writer
        self._source_capture = source_capture

    def ingest(
        self,
        *,
        start_date: date,
        end_date: date,
        ingested_at: datetime | None = None,
    ) -> EarningsIngestionResult:
        """Ingest an inclusive date range and return its durable artifacts."""
        observation_start = len(self._client.earnings_observation_artifacts)
        effective_ingested_at = ingested_at or datetime.now(UTC)
        events = self._client.earnings_calendar(
            start_date=start_date,
            end_date=end_date,
        )
        artifacts = self._silver_writer.write_earnings(
            events,
            ingested_at=effective_ingested_at,
            empty_partition_date=end_date,
        )
        source_manifest = (
            self._source_capture.write(
                start_date=start_date,
                end_date=end_date,
                ingested_at=effective_ingested_at,
                silver_files=artifacts,
                provider_observations=self._client.earnings_observation_artifacts[
                    observation_start:
                ],
            )
            if self._source_capture is not None
            else None
        )
        return EarningsIngestionResult(
            start_date=start_date,
            end_date=end_date,
            event_count=len(events),
            silver_artifacts=artifacts,
            source_manifest=(source_manifest.path if source_manifest is not None else None),
        )


class BarsIngestor:
    """Fetch adjusted Polygon bars and persist them to the silver tier."""

    def __init__(
        self,
        *,
        client: PolygonClient,
        silver_writer: SilverWriter,
        source_capture: DailyBarsSourceCapture | None = None,
    ) -> None:
        self._client = client
        self._silver_writer = silver_writer
        self._source_capture = source_capture

    def ingest(
        self,
        *,
        symbol: str,
        start_date: date,
        end_date: date,
        ingested_at: datetime | None = None,
    ) -> BarsIngestionResult:
        """Ingest one symbol over an inclusive date range."""
        observation_start = len(self._client.feature_observation_artifacts)
        effective_ingested_at = ingested_at or datetime.now(UTC)
        bars = self._client.daily_bars(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
        )
        artifacts = self._silver_writer.write_daily_bars(
            bars,
            ingested_at=effective_ingested_at,
        )
        normalized_symbol = symbol.strip().upper()
        source_manifest = (
            self._source_capture.write(
                symbols=(normalized_symbol,),
                start_date=start_date,
                end_date=end_date,
                ingested_at=effective_ingested_at,
                silver_files=artifacts,
                provider_observations=self._client.feature_observation_artifacts[
                    observation_start:
                ],
            )
            if self._source_capture is not None
            else None
        )
        return BarsIngestionResult(
            symbol=normalized_symbol,
            start_date=start_date,
            end_date=end_date,
            bar_count=len(bars),
            silver_artifacts=artifacts,
            source_manifest=(source_manifest.path if source_manifest is not None else None),
        )


class CorporateActionsIngestor:
    """Fetch validated splits/dividends and persist both silver datasets."""

    def __init__(self, *, client: PolygonClient, silver_writer: SilverWriter) -> None:
        self._client = client
        self._silver_writer = silver_writer

    def ingest(
        self,
        *,
        start_date: date,
        end_date: date,
        ingested_at: datetime | None = None,
    ) -> CorporateActionsIngestionResult:
        """Ingest both corporate-action datasets for one inclusive interval."""
        splits = self._client.stock_splits(start_date=start_date, end_date=end_date)
        dividends = self._client.cash_dividends(start_date=start_date, end_date=end_date)
        split_artifacts = self._silver_writer.write_splits(
            splits,
            ingested_at=ingested_at,
            empty_partition_date=end_date,
        )
        dividend_artifacts = self._silver_writer.write_dividends(
            dividends,
            ingested_at=ingested_at,
            empty_partition_date=end_date,
        )
        return CorporateActionsIngestionResult(
            start_date=start_date,
            end_date=end_date,
            split_count=len(splits),
            dividend_count=len(dividends),
            silver_artifacts=(*split_artifacts, *dividend_artifacts),
        )


class MarketEventsIngestor:
    """Fetch normalized Polygon NBBO/trades and persist lossless silver records."""

    def __init__(self, *, client: PolygonClient, silver_writer: SilverWriter) -> None:
        self._client = client
        self._silver_writer = silver_writer

    def ingest(
        self,
        *,
        symbol: str,
        event_date: date,
        start_at: datetime,
        end_at: datetime,
        ingested_at: datetime | None = None,
    ) -> MarketEventsIngestionResult:
        """Ingest one symbol's inclusive SIP-time execution window."""
        quotes = self._client.stock_quotes(
            symbol=symbol,
            start_at=start_at,
            end_at=end_at,
        )
        trades = self._client.stock_trades(
            symbol=symbol,
            start_at=start_at,
            end_at=end_at,
        )
        quote_artifact = self._silver_writer.write_stock_quotes(
            quotes,
            event_date=event_date,
            ingested_at=ingested_at,
        )
        trade_artifact = self._silver_writer.write_stock_trades(
            trades,
            event_date=event_date,
            ingested_at=ingested_at,
        )
        return MarketEventsIngestionResult(
            symbol=symbol.strip().upper(),
            event_date=event_date,
            start_at=start_at,
            end_at=end_at,
            quote_count=len(quotes),
            trade_count=len(trades),
            quote_artifact=quote_artifact,
            trade_artifact=trade_artifact,
        )
