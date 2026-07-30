"""Independent reconstruction of causal live feature artifacts."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from quant_earning_edge.data import (
    BronzeWriter,
    EarningsSourceCapture,
    LakehouseLayout,
    SilverWriter,
)
from quant_earning_edge.data.clients import EarningsEvent, EquityBar, MinuteBar
from quant_earning_edge.features import (
    DailyBarsFeatureLoader,
    EarningsFeatureLoader,
    FeatureEngine,
    FeatureSourceCapture,
    FeatureSourceManifest,
    FeatureStore,
    PremarketFeatureLoader,
)
from quant_earning_edge.universe import EVENT_CANDIDATE_SCHEMA

ASOF_DATE = date(2026, 7, 27)
TRADE_DATE = date(2026, 7, 28)
OBSERVED_AT = datetime(2026, 7, 28, 12, tzinfo=UTC)


def _fixture(tmp_path: Path) -> tuple[FeatureSourceManifest, Path]:
    layout = LakehouseLayout(tmp_path / "lake")
    writer = SilverWriter(layout)
    first = ASOF_DATE - timedelta(days=80)
    daily_models = tuple(
        EquityBar(
            symbol="AAA",
            timestamp=datetime.combine(
                first + timedelta(days=index),
                datetime.min.time(),
                tzinfo=UTC,
            )
            + timedelta(hours=20),
            open=99 + index,
            high=102 + index,
            low=98 + index,
            close=100 + index,
            volume=1_000_000 + index * 1_000,
            vwap=99.5 + index,
            adjusted=True,
        )
        for index in range(81)
    )
    daily_raw = {
        "ticker": "AAA",
        "adjusted": True,
        "status": "OK",
        "results": [
            {
                "t": int(item.timestamp.timestamp() * 1000),
                "o": item.open,
                "h": item.high,
                "l": item.low,
                "c": item.close,
                "v": item.volume,
                "vw": item.vwap,
            }
            for item in daily_models
        ],
    }
    daily = writer.write_daily_bars(
        daily_models,
        ingested_at=OBSERVED_AT,
    )
    minute_model = MinuteBar(
        symbol="AAA",
        timestamp=OBSERVED_AT - timedelta(minutes=2),
        open=180,
        high=181,
        low=179,
        close=180.5,
        volume=10_000,
        vwap=180.2,
        adjusted=True,
    )
    minute_raw = {
        "ticker": "AAA",
        "adjusted": True,
        "status": "OK",
        "results": [
            {
                "t": int(minute_model.timestamp.timestamp() * 1000),
                "o": minute_model.open,
                "h": minute_model.high,
                "l": minute_model.low,
                "c": minute_model.close,
                "v": minute_model.volume,
                "vw": minute_model.vwap,
            }
        ],
    }
    minute = writer.write_minute_bars(
        (minute_model,),
        event_date=TRADE_DATE,
        ingested_at=OBSERVED_AT,
    )
    bronze = BronzeWriter(layout)
    provider_observations = (
        bronze.write_json(
            daily_raw,
            source="polygon",
            dataset="daily-aggregate-bars",
            event_date=ASOF_DATE,
            received_at=OBSERVED_AT,
        ),
        bronze.write_json(
            minute_raw,
            source="polygon",
            dataset="minute-aggregate-bars",
            event_date=TRADE_DATE,
            received_at=OBSERVED_AT,
        ),
    )
    earnings_raw = {
        "earningsCalendar": [
            {
                "date": (ASOF_DATE - timedelta(days=70)).isoformat(),
                "symbol": "AAA",
                "hour": "amc",
                "year": 2026,
                "quarter": 1,
                "epsActual": 1.2,
                "epsEstimate": 1.0,
            }
        ]
    }
    earnings = writer.write_earnings(
        (EarningsEvent.model_validate(earnings_raw["earningsCalendar"][0]),),
        ingested_at=OBSERVED_AT,
    )
    earnings_observation = bronze.write_json(
        earnings_raw,
        source="finnhub",
        dataset="earnings-calendar",
        event_date=ASOF_DATE - timedelta(days=70),
        received_at=OBSERVED_AT,
    )
    EarningsSourceCapture(layout).write(
        start_date=ASOF_DATE - timedelta(days=70),
        end_date=ASOF_DATE - timedelta(days=70),
        ingested_at=OBSERVED_AT,
        silver_files=earnings,
        provider_observations=(earnings_observation,),
    )
    candidate = layout.root / "gold" / "event-candidates" / "candidate.parquet"
    candidate.parent.mkdir(parents=True)
    pq.write_table(  # type: ignore[no-untyped-call]
        pa.Table.from_pylist(
            [
                {
                    "trade_date": TRADE_DATE,
                    "asof_date": ASOF_DATE,
                    "decision_at": OBSERVED_AT - timedelta(hours=10),
                    "symbol": "AAA",
                    "sector": "technology",
                    "sizing_price": 180.0,
                    "frozen_average_daily_volume_shares": 1_000_000.0,
                    "event_date": ASOF_DATE,
                    "timing": "amc",
                    "year": 2026,
                    "quarter": 3,
                    "eps_estimate": 1.1,
                    "revenue_estimate": 100.0,
                    "split_event_ids": [],
                    "dividend_event_ids": [],
                    "universe_snapshot_sha256": "a" * 64,
                    "session_file_sha256": "b" * 64,
                    "earnings_input_sha256": "c" * 64,
                    "corporate_actions_input_sha256": "d" * 64,
                }
            ],
            schema=EVENT_CANDIDATE_SCHEMA,
        ),
        candidate,
    )
    daily_paths = tuple(item.path for item in daily)
    earnings_paths = tuple(item.path for item in earnings)
    contexts = DailyBarsFeatureLoader().load(
        daily_paths,
        symbols=("AAA",),
        asof_date=ASOF_DATE,
        observed_at=OBSERVED_AT,
    )
    contexts = PremarketFeatureLoader().enrich(
        contexts,
        minute_files=(minute.path,),
        target_date=TRADE_DATE,
        observed_at=OBSERVED_AT,
    )
    contexts = EarningsFeatureLoader().enrich(
        contexts,
        candidate_files=(candidate,),
        earnings_files=earnings_paths,
        observed_at=OBSERVED_AT,
        target_date=TRADE_DATE,
    )
    names = ("earnings_timing_flag", "premarket_gap_pct", "return_1d")
    values = FeatureEngine().compute(contexts, feature_names=names)
    feature = FeatureStore(layout).write(
        feature_group="live",
        values=values,
        computed_at=OBSERVED_AT,
    )
    manifest = FeatureSourceCapture(layout).write(
        trade_date=TRADE_DATE,
        asof_date=ASOF_DATE,
        observed_at=OBSERVED_AT,
        feature_group="live",
        symbols=("AAA",),
        feature_names=names,
        feature_file=feature,
        candidate_files=(candidate,),
        daily_bar_files=daily_paths,
        minute_bar_files=(minute.path,),
        earnings_files=earnings_paths,
        provider_observations=provider_observations,
    )
    return manifest, feature.path


def test_feature_artifact_reproduces_from_exact_causal_inputs(tmp_path: Path) -> None:
    manifest, original_path = _fixture(tmp_path)
    discovered = FeatureSourceCapture.find_for_feature(
        original_path,
        data_lake_root=tmp_path / "lake",
    )
    assert discovered.path == manifest.path

    with TemporaryDirectory(prefix="qee-feature-test-") as temporary:
        reproduced = FeatureSourceCapture.reproduce(
            manifest,
            data_lake_root=tmp_path / "lake",
            output_layout=LakehouseLayout(Path(temporary)),
        )

        assert reproduced.path.read_bytes() == original_path.read_bytes()


def test_feature_source_rejects_changed_causal_input(tmp_path: Path) -> None:
    manifest, _ = _fixture(tmp_path)
    input_path = manifest.input_paths(data_lake_root=tmp_path / "lake")[1]
    input_path.write_bytes(b"changed")

    with pytest.raises(ValueError, match="missing or differs"):
        FeatureSourceCapture.reproduce(
            manifest,
            data_lake_root=tmp_path / "lake",
            output_layout=LakehouseLayout(tmp_path / "reproduced"),
        )


def test_feature_source_rejects_changed_provider_observation(tmp_path: Path) -> None:
    manifest, _ = _fixture(tmp_path)
    provider_path = manifest.provider_paths(data_lake_root=tmp_path / "lake")[0]
    provider_path.write_bytes(b"{}")

    with pytest.raises(ValueError, match="missing or differs"):
        FeatureSourceCapture.reproduce(
            manifest,
            data_lake_root=tmp_path / "lake",
            output_layout=LakehouseLayout(tmp_path / "reproduced"),
        )
