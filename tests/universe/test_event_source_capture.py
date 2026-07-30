"""Provider-source reconstruction for earnings and corporate-action silver."""

from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from quant_earning_edge.data import BronzeWriter, LakehouseLayout, SilverWriter
from quant_earning_edge.data.clients import FinnhubClient, PolygonClient
from quant_earning_edge.universe import EventSourceCapture, EventSourceCaptureManifest

START_DATE = date(2026, 7, 27)
END_DATE = date(2026, 7, 28)
INGESTED_AT = datetime(2026, 7, 28, 1, 30, tzinfo=UTC)


def _payloads() -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    earnings: dict[str, object] = {
        "earningsCalendar": [
            {
                "date": "2026-07-27",
                "symbol": "AAA",
                "hour": "amc",
                "year": 2026,
                "quarter": 3,
                "epsEstimate": 1.0,
                "revenueEstimate": 100.0,
            }
        ]
    }
    splits: dict[str, object] = {
        "status": "OK",
        "results": [
            {
                "id": "split-1",
                "ticker": "AAA",
                "execution_date": "2026-07-28",
                "adjustment_type": "forward_split",
                "split_from": 1,
                "split_to": 2,
            }
        ],
    }
    dividends: dict[str, object] = {
        "status": "OK",
        "results": [
            {
                "id": "dividend-1",
                "ticker": "AAA",
                "ex_dividend_date": "2026-07-28",
                "distribution_type": "recurring",
                "cash_amount": 0.25,
                "currency": "USD",
                "frequency": 4,
            }
        ],
    }
    return earnings, splits, dividends


def _source_fixture(
    tmp_path: Path,
) -> tuple[EventSourceCaptureManifest, tuple[Path, ...]]:
    layout = LakehouseLayout(tmp_path / "lake")
    earnings_raw, splits_raw, dividends_raw = _payloads()
    bronze = BronzeWriter(layout)
    earnings_observations = (
        bronze.write_json(
            earnings_raw,
            source="finnhub",
            dataset="earnings-calendar",
            event_date=END_DATE,
            received_at=INGESTED_AT,
        ),
    )
    action_observations = (
        bronze.write_json(
            splits_raw,
            source="polygon",
            dataset="stock-splits",
            event_date=END_DATE,
            received_at=INGESTED_AT,
        ),
        bronze.write_json(
            dividends_raw,
            source="polygon",
            dataset="cash-dividends",
            event_date=END_DATE,
            received_at=INGESTED_AT,
        ),
    )
    writer = SilverWriter(layout)
    earnings = writer.write_earnings(
        FinnhubClient.earnings_calendar_from_payload(earnings_raw),
        ingested_at=INGESTED_AT,
    )
    splits = writer.write_splits(
        PolygonClient.stock_splits_from_payloads(
            (splits_raw,),
            start_date=START_DATE,
            end_date=END_DATE,
        ),
        ingested_at=INGESTED_AT,
    )
    dividends = writer.write_dividends(
        PolygonClient.cash_dividends_from_payloads(
            (dividends_raw,),
            start_date=START_DATE,
            end_date=END_DATE,
        ),
        ingested_at=INGESTED_AT,
    )
    manifest = EventSourceCapture(layout).write(
        start_date=START_DATE,
        end_date=END_DATE,
        ingested_at=INGESTED_AT,
        earnings_files=earnings,
        split_files=splits,
        dividend_files=dividends,
        earnings_observations=earnings_observations,
        corporate_action_observations=action_observations,
    )
    return manifest, tuple(item.path for item in (*earnings, *splits, *dividends))


def test_event_silver_reproduces_from_retained_provider_payloads(
    tmp_path: Path,
) -> None:
    manifest, original_paths = _source_fixture(tmp_path)
    provider_paths = manifest.provider_paths(data_lake_root=tmp_path / "lake")
    recomposed = EventSourceCapture(LakehouseLayout(tmp_path / "lake")).write_paths(
        start_date=START_DATE,
        end_date=END_DATE,
        ingested_at=INGESTED_AT,
        earnings_files=original_paths[:1],
        split_files=original_paths[1:2],
        dividend_files=original_paths[2:],
        earnings_observations=provider_paths[:1],
        corporate_action_observations=provider_paths[1:],
    )

    with TemporaryDirectory(prefix="qee-events-test-") as temporary:
        reproduced = EventSourceCapture.reproduce(
            manifest,
            data_lake_root=tmp_path / "lake",
            output_layout=LakehouseLayout(Path(temporary)),
        )

        assert tuple(item.path.read_bytes() for item in reproduced) == tuple(
            path.read_bytes() for path in original_paths
        )
        assert recomposed == manifest


def test_event_source_manifest_rejects_changed_provider_payload(
    tmp_path: Path,
) -> None:
    manifest, _ = _source_fixture(tmp_path)
    provider_path = manifest.provider_paths(data_lake_root=tmp_path / "lake")[0]
    provider_path.write_bytes(b"{}")

    with pytest.raises(ValueError, match="missing or differs"):
        EventSourceCapture.reproduce(
            manifest,
            data_lake_root=tmp_path / "lake",
            output_layout=LakehouseLayout(tmp_path / "reproduced"),
        )
