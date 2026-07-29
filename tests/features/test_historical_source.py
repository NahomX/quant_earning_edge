"""Provider-source reconstruction tests for historical research features."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from typing import TYPE_CHECKING

import pytest

from quant_earning_edge.data import (
    BronzeWriter,
    DailyBarsSourceCapture,
    LakehouseLayout,
    SilverWriter,
)
from quant_earning_edge.data.clients import PolygonClient
from quant_earning_edge.features import (
    DailyBarsFeatureLoader,
    FeatureEngine,
    FeatureStore,
    HistoricalFeatureSourceCapture,
    HistoricalFeatureSourceManifest,
)

if TYPE_CHECKING:
    from pathlib import Path

ASOF_DATE = date(2026, 7, 27)
START_DATE = ASOF_DATE - timedelta(days=20)
OBSERVED_AT = datetime(2026, 7, 27, 21, tzinfo=UTC)


def _source_fixture(tmp_path: Path) -> tuple[HistoricalFeatureSourceManifest, Path]:
    layout = LakehouseLayout(tmp_path / "lake")
    raw = {
        "ticker": "AAPL",
        "adjusted": True,
        "status": "OK",
        "results": [
            {
                "o": 99.0 + index,
                "h": 102.0 + index,
                "l": 98.0 + index,
                "c": 100.0 + index,
                "v": 1_000_000 + index,
                "vw": 99.5 + index,
                "n": 50_000,
                "t": int(
                    (
                        datetime.combine(
                            START_DATE + timedelta(days=index),
                            datetime.min.time(),
                            tzinfo=UTC,
                        )
                        + timedelta(hours=4)
                    ).timestamp()
                    * 1000
                ),
            }
            for index in range(21)
        ],
    }
    observation = BronzeWriter(layout).write_json(
        raw,
        source="polygon",
        dataset="daily-aggregate-bars",
        event_date=START_DATE,
        received_at=OBSERVED_AT,
    )
    bars = PolygonClient.daily_bars_from_payload(raw, symbol="AAPL")
    silver = SilverWriter(layout).write_daily_bars(bars, ingested_at=OBSERVED_AT)
    DailyBarsSourceCapture(layout).write(
        symbols=("AAPL",),
        start_date=START_DATE,
        end_date=ASOF_DATE,
        ingested_at=OBSERVED_AT,
        silver_files=silver,
        provider_observations=(observation,),
    )
    contexts = DailyBarsFeatureLoader().load(
        tuple(item.path for item in silver),
        symbols=("AAPL",),
        asof_date=ASOF_DATE,
        observed_at=OBSERVED_AT,
    )
    names = ("return_1d", "return_20d")
    artifact = FeatureStore(layout).write(
        feature_group="price",
        values=FeatureEngine().compute(contexts, feature_names=names),
        computed_at=OBSERVED_AT,
    )
    manifest = HistoricalFeatureSourceCapture(layout).write(
        asof_date=ASOF_DATE,
        observed_at=OBSERVED_AT,
        target_date=None,
        feature_group="price",
        symbols=("AAPL",),
        feature_names=names,
        feature_file=artifact,
        daily_bar_files=tuple(item.path for item in silver),
    )
    return manifest, observation.path


def test_historical_features_reproduce_from_polygon_sources(tmp_path: Path) -> None:
    manifest, _ = _source_fixture(tmp_path)
    lake = tmp_path / "lake"
    feature = manifest.feature_path(data_lake_root=lake)

    assert (
        HistoricalFeatureSourceCapture.find_for_feature(
            feature,
            data_lake_root=lake,
        )
        == manifest
    )
    assert (
        HistoricalFeatureSourceCapture.reproduce(
            manifest,
            data_lake_root=lake,
        )
        == feature
    )


def test_historical_feature_reproduction_rejects_changed_provider_payload(
    tmp_path: Path,
) -> None:
    manifest, observation = _source_fixture(tmp_path)
    observation.write_bytes(b"{}")

    with pytest.raises(ValueError, match="missing or differs"):
        HistoricalFeatureSourceCapture.reproduce(
            manifest,
            data_lake_root=tmp_path / "lake",
        )
