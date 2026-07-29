"""Session-indexed label and leakage-guarded dataset tests."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from typing import TYPE_CHECKING

import pyarrow.parquet as pq
import pytest

from quant_earning_edge.data import (
    BronzeWriter,
    CalendarSourceCapture,
    DailyBarsSourceCapture,
    LakehouseLayout,
    SessionFileStore,
    SilverWriter,
)
from quant_earning_edge.data.clients import AlpacaCalendarClient, PolygonClient
from quant_earning_edge.features import (
    DailyBarsFeatureLoader,
    FeatureEngine,
    FeatureStore,
    HistoricalFeatureSourceCapture,
)
from quant_earning_edge.labels import (
    FORWARD_LABEL_SCHEMA,
    ForwardLabelMaker,
    ForwardLabelSourceCapture,
    LabelBar,
    LabelBarsLoader,
    LabelStore,
    TrainingDatasetAssembler,
)
from quant_earning_edge.labels.dataset_source import (
    TrainingDatasetSourceCapture,
    TrainingDatasetSourceManifest,
)

if TYPE_CHECKING:
    from pathlib import Path

ASOF_DATE = date(2026, 7, 1)
SESSION_DATES = tuple(ASOF_DATE + timedelta(days=index) for index in range(7))


def _session_file(tmp_path: Path) -> Path:
    layout = LakehouseLayout(tmp_path)
    raw = [
        {"date": session_date.isoformat(), "open": "09:30", "close": "16:00"}
        for session_date in SESSION_DATES
    ]
    observation = BronzeWriter(layout).write_json(
        raw,
        source="alpaca",
        dataset="market-calendar",
        event_date=SESSION_DATES[0],
        received_at=datetime(2026, 6, 30, 12, tzinfo=UTC),
    )
    sessions = AlpacaCalendarClient.sessions_from_payload(
        raw,
        start_date=SESSION_DATES[0],
        end_date=SESSION_DATES[-1],
    )
    artifact = SessionFileStore(layout).write(sessions)
    CalendarSourceCapture(layout).write(
        start_date=SESSION_DATES[0],
        end_date=SESSION_DATES[-1],
        session_file=artifact,
        provider_observations=(observation,),
    )
    return artifact.path


def _label_bars() -> tuple[LabelBar, ...]:
    return tuple(
        LabelBar(
            symbol="AAPL",
            session_date=session_date,
            open=100.0 + index,
            close=101.0 + index,
        )
        for index, session_date in enumerate(SESSION_DATES[:6])
    )


def test_label_maker_uses_explicit_session_offsets_and_store_is_idempotent(
    tmp_path: Path,
) -> None:
    labels = ForwardLabelMaker().compute(
        keys=(("aapl", ASOF_DATE),),
        sessions=SESSION_DATES,
        bars=_label_bars(),
    )

    assert len(labels) == 1
    label = labels[0]
    assert label.target_date == SESSION_DATES[1]
    assert label.horizon_end_date == SESSION_DATES[5]
    assert label.forward_1d_open_to_close == pytest.approx(102 / 101 - 1)
    assert label.forward_1d_close == pytest.approx(102 / 101 - 1)
    assert label.forward_5d_close == pytest.approx(106 / 101 - 1)
    store = LabelStore(LakehouseLayout(tmp_path))
    computed_at = datetime(2026, 7, 8, 22, tzinfo=UTC)
    first = store.write(labels, computed_at=computed_at)
    second = store.write(labels, computed_at=computed_at)

    assert first == second
    assert pq.read_schema(first.path) == FORWARD_LABEL_SCHEMA  # type: ignore[no-untyped-call]
    assert len(list((tmp_path / "gold").rglob("*.parquet"))) == 1


def test_label_maker_fails_when_a_required_future_session_bar_is_missing() -> None:
    with pytest.raises(ValueError, match="missing required label bar"):
        ForwardLabelMaker().compute(
            keys=(("AAPL", ASOF_DATE),),
            sessions=SESSION_DATES,
            bars=_label_bars()[:-1],
        )


def _feature_artifact(tmp_path: Path, *, computed_at: datetime) -> Path:
    layout = LakehouseLayout(tmp_path)
    start_date = ASOF_DATE - timedelta(days=20)
    raw = {
        "ticker": "AAPL",
        "adjusted": True,
        "status": "OK",
        "results": [
            {
                "o": 99 + index,
                "h": 102 + index,
                "l": 98 + index,
                "c": 100 + index,
                "v": 1_000_000,
                "vw": 99.5 + index,
                "n": 50_000,
                "t": int(
                    (
                        datetime.combine(
                            start_date + timedelta(days=index),
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
        event_date=start_date,
        received_at=computed_at,
    )
    bars = PolygonClient.daily_bars_from_payload(raw, symbol="AAPL")
    silver = SilverWriter(layout).write_daily_bars(bars, ingested_at=computed_at)
    DailyBarsSourceCapture(layout).write(
        symbols=("AAPL",),
        start_date=start_date,
        end_date=ASOF_DATE,
        ingested_at=computed_at,
        silver_files=silver,
        provider_observations=(observation,),
    )
    contexts = DailyBarsFeatureLoader().load(
        tuple(item.path for item in silver),
        symbols=("AAPL",),
        asof_date=ASOF_DATE,
        observed_at=computed_at,
    )
    values = FeatureEngine().compute(
        contexts,
        feature_names=("return_1d", "return_20d"),
    )
    artifact = FeatureStore(layout).write(
        feature_group="price",
        values=values,
        computed_at=computed_at,
    )
    HistoricalFeatureSourceCapture(layout).write(
        asof_date=ASOF_DATE,
        observed_at=computed_at,
        target_date=None,
        feature_group="price",
        symbols=("AAPL",),
        feature_names=("return_1d", "return_20d"),
        feature_file=artifact,
        daily_bar_files=tuple(item.path for item in silver),
    )
    return artifact.path


def _label_artifact(tmp_path: Path, *, session_file: Path) -> Path:
    layout = LakehouseLayout(tmp_path)
    computed_at = datetime(2026, 7, 8, 22, tzinfo=UTC)
    raw = {
        "ticker": "AAPL",
        "adjusted": True,
        "status": "OK",
        "results": [
            {
                "o": bar.open,
                "h": max(bar.open, bar.close) + 1,
                "l": min(bar.open, bar.close) - 1,
                "c": bar.close,
                "v": 1_000_000,
                "vw": (bar.open + bar.close) / 2,
                "n": 50_000,
                "t": int(
                    (
                        datetime.combine(
                            bar.session_date,
                            datetime.min.time(),
                            tzinfo=UTC,
                        )
                        + timedelta(hours=4)
                    ).timestamp()
                    * 1000
                ),
            }
            for bar in _label_bars()
        ],
    }
    observation = BronzeWriter(layout).write_json(
        raw,
        source="polygon",
        dataset="daily-aggregate-bars",
        event_date=ASOF_DATE,
        received_at=computed_at,
    )
    bars = PolygonClient.daily_bars_from_payload(raw, symbol="AAPL")
    silver = SilverWriter(layout).write_daily_bars(bars, ingested_at=computed_at)
    DailyBarsSourceCapture(layout).write(
        symbols=("AAPL",),
        start_date=ASOF_DATE,
        end_date=SESSION_DATES[5],
        ingested_at=computed_at,
        silver_files=silver,
        provider_observations=(observation,),
    )
    loaded = LabelBarsLoader().load(
        tuple(item.path for item in silver),
        symbols=("AAPL",),
        start_date=ASOF_DATE,
        end_date=SESSION_DATES[5],
        observed_at=computed_at,
    )
    labels = ForwardLabelMaker().compute(
        keys=(("AAPL", ASOF_DATE),),
        sessions=SESSION_DATES,
        bars=loaded,
    )
    artifact = LabelStore(layout).write(labels, computed_at=computed_at)
    ForwardLabelSourceCapture(layout).write(
        asof_date=ASOF_DATE,
        observed_at=computed_at,
        symbols=("AAPL",),
        label_file=artifact,
        daily_bar_files=tuple(item.path for item in silver),
        session_file=session_file,
    )
    return artifact.path


def test_training_assembly_requires_complete_keys_and_preopen_features(
    tmp_path: Path,
) -> None:
    session_file = _session_file(tmp_path)
    feature_file = _feature_artifact(
        tmp_path,
        computed_at=datetime(2026, 7, 1, 21, tzinfo=UTC),
    )
    label_file = _label_artifact(tmp_path, session_file=session_file)

    artifact = TrainingDatasetAssembler(LakehouseLayout(tmp_path)).assemble(
        feature_files=(feature_file,),
        label_files=(label_file,),
        session_file=session_file,
        assembled_at=datetime(2026, 7, 8, 23, tzinfo=UTC),
    )

    table = pq.read_table(artifact.path)  # type: ignore[no-untyped-call]
    assert artifact.row_count == 1
    assert artifact.feature_names == ("return_1d", "return_20d")
    assert table.column("target_date").to_pylist() == [SESSION_DATES[1]]
    assert table.column("information_cutoff_at").to_pylist() == [
        datetime(2026, 7, 1, 21, tzinfo=UTC)
    ]
    assert table.column("return_20d").to_pylist()[0] == pytest.approx(120 / 100 - 1)
    assert artifact.manifest_path.exists()
    source = TrainingDatasetSourceManifest.load(artifact.source_manifest_path)
    assert TrainingDatasetSourceCapture.reproduce(source) == artifact.path
    provider_path = next(path for path in source.feature_source_paths() if "bronze" in path.parts)
    provider_bytes = provider_path.read_bytes()
    provider_path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="missing or differs"):
        TrainingDatasetSourceCapture.reproduce(source)
    provider_path.write_bytes(provider_bytes)
    feature_file.write_bytes(b"changed")
    with pytest.raises(ValueError, match="missing or differs"):
        TrainingDatasetSourceCapture.reproduce(source)


def test_training_assembly_rejects_features_computed_after_target_open(
    tmp_path: Path,
) -> None:
    session_file = _session_file(tmp_path)
    feature_file = _feature_artifact(
        tmp_path,
        computed_at=datetime(2026, 7, 2, 15, tzinfo=UTC),
    )
    label_file = _label_artifact(tmp_path, session_file=session_file)

    with pytest.raises(ValueError, match="not frozen before target open"):
        TrainingDatasetAssembler(LakehouseLayout(tmp_path)).assemble(
            feature_files=(feature_file,),
            label_files=(label_file,),
            session_file=session_file,
            assembled_at=datetime(2026, 7, 8, 23, tzinfo=UTC),
        )
