"""Orchestration services that move provider observations into lake tiers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from datetime import date, datetime

    from quant_earning_edge.data.clients.finnhub import FinnhubClient
    from quant_earning_edge.data.silver import SilverArtifact, SilverWriter


@dataclass(frozen=True)
class EarningsIngestionResult:
    """Auditable result of one Finnhub earnings interval ingestion."""

    start_date: date
    end_date: date
    event_count: int
    silver_artifacts: tuple[SilverArtifact, ...]


class EarningsIngestor:
    """Fetch validated events and persist them to the silver tier."""

    def __init__(self, *, client: FinnhubClient, silver_writer: SilverWriter) -> None:
        self._client = client
        self._silver_writer = silver_writer

    def ingest(
        self,
        *,
        start_date: date,
        end_date: date,
        ingested_at: datetime | None = None,
    ) -> EarningsIngestionResult:
        """Ingest an inclusive date range and return its durable artifacts."""
        events = self._client.earnings_calendar(
            start_date=start_date,
            end_date=end_date,
        )
        artifacts = self._silver_writer.write_earnings(
            events,
            ingested_at=ingested_at,
        )
        return EarningsIngestionResult(
            start_date=start_date,
            end_date=end_date,
            event_count=len(events),
            silver_artifacts=artifacts,
        )
