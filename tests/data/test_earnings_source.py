"""Provider-source reconstruction for Finnhub earnings Silver."""

from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from quant_earning_edge.data import (
    BronzeWriter,
    EarningsSourceCapture,
    EarningsSourceManifest,
    LakehouseLayout,
    SilverWriter,
)
from quant_earning_edge.data.clients import FinnhubClient

START_DATE = date(2026, 7, 27)
END_DATE = date(2026, 7, 28)
INGESTED_AT = datetime(2026, 7, 28, 1, 30, tzinfo=UTC)


def _fixture(tmp_path: Path) -> tuple[EarningsSourceManifest, Path]:
    raw = {
        "earningsCalendar": [
            {
                "date": "2026-07-27",
                "symbol": "AAA",
                "hour": "amc",
                "year": 2026,
                "quarter": 3,
                "epsEstimate": 1.0,
            }
        ]
    }
    layout = LakehouseLayout(tmp_path / "lake")
    observation = BronzeWriter(layout).write_json(
        raw,
        source="finnhub",
        dataset="earnings-calendar",
        event_date=START_DATE,
        received_at=INGESTED_AT,
    )
    silver = SilverWriter(layout).write_earnings(
        FinnhubClient.earnings_calendar_from_payload(raw),
        ingested_at=INGESTED_AT,
    )
    manifest = EarningsSourceCapture(layout).write(
        start_date=START_DATE,
        end_date=END_DATE,
        ingested_at=INGESTED_AT,
        silver_files=silver,
        provider_observations=(observation,),
    )
    return manifest, silver[0].path


def test_earnings_silver_reproduces_from_retained_finnhub_payload(
    tmp_path: Path,
) -> None:
    manifest, original_path = _fixture(tmp_path)
    discovered = EarningsSourceCapture.find_for_files(
        (original_path,),
        data_lake_root=tmp_path / "lake",
    )
    assert discovered == (manifest,)

    with TemporaryDirectory(prefix="qee-earnings-test-") as temporary:
        reproduced = EarningsSourceCapture.reproduce(
            manifest,
            data_lake_root=tmp_path / "lake",
            output_layout=LakehouseLayout(Path(temporary)),
        )

        assert reproduced[0].path.read_bytes() == original_path.read_bytes()


def test_earnings_source_rejects_changed_finnhub_payload(tmp_path: Path) -> None:
    manifest, _ = _fixture(tmp_path)
    provider_path = manifest.provider_paths(data_lake_root=tmp_path / "lake")[0]
    provider_path.write_bytes(b"{}")

    with pytest.raises(ValueError, match="missing or differs"):
        EarningsSourceCapture.reproduce(
            manifest,
            data_lake_root=tmp_path / "lake",
            output_layout=LakehouseLayout(tmp_path / "reproduced"),
        )
