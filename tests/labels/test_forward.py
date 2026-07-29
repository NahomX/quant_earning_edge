"""Session-indexed label and leakage-guarded dataset tests."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from typing import TYPE_CHECKING
from zoneinfo import ZoneInfo

import pyarrow.parquet as pq
import pytest

from quant_earning_edge.data import LakehouseLayout, SessionFileStore
from quant_earning_edge.data.clients import MarketSession
from quant_earning_edge.features import FeatureContext, FeatureEngine, FeatureStore, PriceBar
from quant_earning_edge.labels import (
    FORWARD_LABEL_SCHEMA,
    ForwardLabelMaker,
    LabelBar,
    LabelStore,
    TrainingDatasetAssembler,
)

if TYPE_CHECKING:
    from pathlib import Path

ASOF_DATE = date(2026, 7, 1)
SESSION_DATES = tuple(ASOF_DATE + timedelta(days=index) for index in range(7))


def _session_file(tmp_path: Path) -> Path:
    eastern = ZoneInfo("America/New_York")
    sessions = tuple(
        MarketSession(
            session_date=session_date,
            open_at=datetime.combine(
                session_date,
                datetime.min.time(),
                tzinfo=eastern,
            )
            + timedelta(hours=9, minutes=30),
            close_at=datetime.combine(
                session_date,
                datetime.min.time(),
                tzinfo=eastern,
            )
            + timedelta(hours=16),
        )
        for session_date in SESSION_DATES
    )
    return SessionFileStore(LakehouseLayout(tmp_path)).write(sessions).path


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
    bars = tuple(
        PriceBar(
            session_date=ASOF_DATE - timedelta(days=20 - index),
            close=100 + index,
            volume=1_000_000,
            vwap=99.5 + index,
        )
        for index in range(21)
    )
    values = FeatureEngine().compute(
        (FeatureContext(symbol="AAPL", asof_date=ASOF_DATE, bars=bars),),
        feature_names=("return_1d", "return_20d"),
    )
    return (
        FeatureStore(LakehouseLayout(tmp_path))
        .write(
            feature_group="price",
            values=values,
            computed_at=computed_at,
        )
        .path
    )


def _label_artifact(tmp_path: Path) -> Path:
    labels = ForwardLabelMaker().compute(
        keys=(("AAPL", ASOF_DATE),),
        sessions=SESSION_DATES,
        bars=_label_bars(),
    )
    return (
        LabelStore(LakehouseLayout(tmp_path))
        .write(
            labels,
            computed_at=datetime(2026, 7, 8, 22, tzinfo=UTC),
        )
        .path
    )


def test_training_assembly_requires_complete_keys_and_preopen_features(
    tmp_path: Path,
) -> None:
    session_file = _session_file(tmp_path)
    feature_file = _feature_artifact(
        tmp_path,
        computed_at=datetime(2026, 7, 1, 21, tzinfo=UTC),
    )
    label_file = _label_artifact(tmp_path)

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


def test_training_assembly_rejects_features_computed_after_target_open(
    tmp_path: Path,
) -> None:
    session_file = _session_file(tmp_path)
    feature_file = _feature_artifact(
        tmp_path,
        computed_at=datetime(2026, 7, 2, 15, tzinfo=UTC),
    )
    label_file = _label_artifact(tmp_path)

    with pytest.raises(ValueError, match="not frozen before target open"):
        TrainingDatasetAssembler(LakehouseLayout(tmp_path)).assemble(
            feature_files=(feature_file,),
            label_files=(label_file,),
            session_file=session_file,
            assembled_at=datetime(2026, 7, 8, 23, tzinfo=UTC),
        )
