"""Provider-source reconstruction for point-in-time universe snapshots."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING

import pytest

from quant_earning_edge.data import BronzeWriter, LakehouseLayout
from quant_earning_edge.data.clients import PolygonClient
from quant_earning_edge.universe import (
    DailyUniverseJob,
    RunTrigger,
    UniverseBuilder,
    UniverseManifestStore,
    UniverseSnapshotWriter,
    UniverseSourceCapture,
    UniverseSourceCaptureManifest,
)
from quant_earning_edge.universe.config import (
    load_halt_snapshot,
    load_universe_job_config,
)

if TYPE_CHECKING:
    from quant_earning_edge.data.clients import EquityBar, TickerDetails, TickerReference

TRADE_DATE = date(2026, 7, 28)
ASOF_DATE = date(2026, 7, 27)
DECISION_AT = datetime(2026, 7, 28, 1, 30, tzinfo=UTC)


class _PayloadMarketData:
    def __init__(
        self,
        *,
        references: tuple[TickerReference, ...],
        details: TickerDetails,
        bars: tuple[EquityBar, ...],
    ) -> None:
        self._references = references
        self._details = details
        self._bars = bars

    def list_tickers(
        self,
        *,
        asof_date: date,
        active: bool = True,
    ) -> tuple[TickerReference, ...]:
        assert asof_date == ASOF_DATE
        assert active
        return self._references

    def ticker_details(self, *, symbol: str, asof_date: date) -> TickerDetails:
        assert symbol == "AAA"
        assert asof_date == ASOF_DATE
        return self._details

    def daily_bars(
        self,
        *,
        symbol: str,
        start_date: date,
        end_date: date,
    ) -> tuple[EquityBar, ...]:
        assert symbol == "AAA"
        assert start_date == ASOF_DATE
        assert end_date == ASOF_DATE
        return self._bars


def _payloads() -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    references = {
        "status": "OK",
        "results": [
            {
                "ticker": "AAA",
                "name": "AAA Corp",
                "active": True,
                "locale": "us",
                "market": "stocks",
                "primary_exchange": "XNAS",
                "type": "CS",
            }
        ],
    }
    details = {
        "status": "OK",
        "results": {
            "ticker": "AAA",
            "name": "AAA Corp",
            "active": True,
            "locale": "us",
            "market": "stocks",
            "primary_exchange": "XNAS",
            "type": "CS",
            "market_cap": 2_000_000_000,
            "sic_code": "3571",
            "list_date": "2000-01-01",
        },
    }
    bars = {
        "ticker": "AAA",
        "adjusted": True,
        "status": "OK",
        "results": [
            {
                "t": int(datetime(2026, 7, 27, 20, tzinfo=UTC).timestamp() * 1000),
                "o": 99,
                "h": 101,
                "l": 98,
                "c": 100,
                "v": 2_000_000,
                "vw": 100,
                "n": 1000,
            }
        ],
    }
    return references, details, bars


def _source_fixture(tmp_path: Path) -> tuple[UniverseSourceCaptureManifest, Path]:
    layout = LakehouseLayout(tmp_path / "lake")
    reference_raw, details_raw, bars_raw = _payloads()
    writer = BronzeWriter(layout)
    observations = (
        writer.write_json(
            reference_raw,
            source="polygon",
            dataset="ticker-reference",
            event_date=ASOF_DATE,
            received_at=DECISION_AT,
        ),
        writer.write_json(
            details_raw,
            source="polygon",
            dataset="ticker-details",
            event_date=ASOF_DATE,
            received_at=DECISION_AT,
        ),
        writer.write_json(
            bars_raw,
            source="polygon",
            dataset="daily-aggregate-bars",
            event_date=ASOF_DATE,
            received_at=DECISION_AT,
        ),
    )
    config_path = tmp_path / "universe.yaml"
    config_path.write_text(
        "\n".join(
            (
                "adv_sessions: 1",
                "eligibility:",
                "  min_price: 5",
                "  min_market_cap_usd: 100000000",
                "  min_avg_daily_volume: 500000",
                "  allowed_exchanges: [XNAS]",
                "  allowed_security_types: [CS]",
                "  exclude_halts: true",
            )
        ),
        encoding="utf-8",
    )
    halt_path = tmp_path / "halts.json"
    halt_path.write_text(
        json.dumps(
            {
                "asof_date": ASOF_DATE.isoformat(),
                "captured_at": "2026-07-27T21:00:00+00:00",
                "symbols": [],
            }
        ),
        encoding="utf-8",
    )
    references = PolygonClient.ticker_references_from_payload(
        reference_raw,
        asof_date=ASOF_DATE,
    )
    details = PolygonClient.ticker_details_from_payload(
        details_raw,
        symbol="AAA",
        asof_date=ASOF_DATE,
    )
    bars = PolygonClient.daily_bars_from_payload(bars_raw, symbol="AAA")
    config = load_universe_job_config(config_path)
    result = DailyUniverseJob(
        market_data=_PayloadMarketData(
            references=references,
            details=details,
            bars=bars,
        ),
        builder=UniverseBuilder(config.eligibility.to_domain()),
        snapshot_writer=UniverseSnapshotWriter(layout),
        manifest_store=UniverseManifestStore(layout),
        adv_sessions=1,
        clock=lambda: DECISION_AT,
        run_id_factory=lambda: "original",
    ).run(
        trade_date=TRADE_DATE,
        asof_date=ASOF_DATE,
        lookback_start=ASOF_DATE,
        halt_snapshot=load_halt_snapshot(halt_path),
        trigger=RunTrigger.SCHEDULED,
    )
    manifest = UniverseSourceCapture(layout).write(
        trade_date=TRADE_DATE,
        asof_date=ASOF_DATE,
        lookback_start=ASOF_DATE,
        decision_at=DECISION_AT,
        adv_sessions=1,
        snapshot=result.snapshot,
        universe_config=config_path,
        halt_snapshot=halt_path,
        provider_observations=observations,
    )
    return manifest, result.snapshot.path


def test_universe_snapshot_reproduces_from_retained_provider_payloads(
    tmp_path: Path,
) -> None:
    manifest, original_path = _source_fixture(tmp_path)

    with TemporaryDirectory(prefix="qee-universe-test-") as temporary:
        reproduced = UniverseSourceCapture.reproduce(
            manifest,
            data_lake_root=tmp_path / "lake",
            output_layout=LakehouseLayout(Path(temporary)),
        )

        assert reproduced.path.read_bytes() == original_path.read_bytes()
        assert reproduced.sha256 == manifest.raw["snapshot_semantic_sha256"]


def test_universe_source_manifest_rejects_changed_provider_payload(
    tmp_path: Path,
) -> None:
    manifest, _ = _source_fixture(tmp_path)
    provider_path = manifest.source_paths(data_lake_root=tmp_path / "lake")[2]
    provider_path.write_bytes(b"{}")

    with pytest.raises(ValueError, match="missing or differs"):
        UniverseSourceCapture.reproduce(
            manifest,
            data_lake_root=tmp_path / "lake",
            output_layout=LakehouseLayout(tmp_path / "reproduced"),
        )
