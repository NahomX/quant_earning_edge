"""Feature registry, engine, and deterministic gold-store tests."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from typing import TYPE_CHECKING

import pyarrow.parquet as pq
import pytest

from quant_earning_edge.data import LakehouseLayout, SilverWriter
from quant_earning_edge.data.clients import EquityBar
from quant_earning_edge.features import (
    FEATURE_REGISTRY,
    FEATURE_VALUE_SCHEMA,
    DailyBarsFeatureLoader,
    FeatureContext,
    FeatureEngine,
    FeatureStore,
    InsufficientHistoryError,
    PriceBar,
)

if TYPE_CHECKING:
    from pathlib import Path


def _context(*, include_future: bool = False) -> FeatureContext:
    first = date(2026, 1, 1)
    bars = tuple(
        PriceBar(
            session_date=first + timedelta(days=index),
            close=100.0 + index,
            volume=1_000_000.0 + index * 100,
            vwap=99.5 + index,
        )
        for index in range(70 if include_future else 61)
    )
    return FeatureContext(
        symbol="aapl",
        asof_date=first + timedelta(days=60),
        bars=bars,
    )


def test_registry_has_real_metadata_and_six_baseline_features() -> None:
    specs = FEATURE_REGISTRY.values()

    assert len(specs) == 6
    assert {item.name for item in specs} == {
        "distance_to_vwap_20d",
        "realized_vol_20d",
        "realized_vol_60d",
        "return_1d",
        "return_20d",
        "return_5d",
    }
    assert all(len(item.code_hash) == 64 for item in specs)
    assert all(item.dependencies == ("silver_daily_bars",) for item in specs)


def test_engine_lineage_is_unchanged_by_future_rows() -> None:
    names = ("return_1d", "return_20d", "realized_vol_60d")
    engine = FeatureEngine()

    truncated = engine.compute((_context(),), feature_names=names)
    with_future = engine.compute((_context(include_future=True),), feature_names=names)

    assert truncated == with_future


def test_store_writes_idempotent_schema_stable_month_partition(tmp_path: Path) -> None:
    values = FeatureEngine().compute(
        (_context(),),
        feature_names=("return_1d", "realized_vol_20d"),
    )
    store = FeatureStore(LakehouseLayout(tmp_path))
    computed_at = datetime(2026, 3, 3, 22, tzinfo=UTC)

    first = store.write(
        feature_group="price",
        values=values,
        computed_at=computed_at,
    )
    second = store.write(
        feature_group="price",
        values=values,
        computed_at=computed_at,
    )

    assert first == second
    assert "feature_group=price/month=2026-03" in first.path.as_posix()
    assert pq.read_schema(first.path) == FEATURE_VALUE_SCHEMA  # type: ignore[no-untyped-call]
    rows = pq.read_table(first.path).to_pylist()  # type: ignore[no-untyped-call]
    assert [row["feature_name"] for row in rows] == ["realized_vol_20d", "return_1d"]
    assert len(list(tmp_path.rglob("*.parquet"))) == 1


def test_insufficient_history_fails_loudly() -> None:
    context = FeatureContext(
        symbol="AAPL",
        asof_date=date(2026, 1, 2),
        bars=(
            PriceBar(
                session_date=date(2026, 1, 2),
                close=100,
                volume=1_000_000,
            ),
        ),
    )

    with pytest.raises(InsufficientHistoryError, match="61 required"):
        FEATURE_REGISTRY.get("realized_vol_60d").evaluate(context)


def test_daily_bar_loader_resolves_only_revisions_known_at_cutoff(tmp_path: Path) -> None:
    first = date(2026, 1, 1)
    bars = tuple(
        EquityBar(
            symbol="AAPL",
            timestamp=datetime.combine(
                first + timedelta(days=index),
                datetime.min.time(),
                tzinfo=UTC,
            )
            + timedelta(hours=17),
            open=100 + index,
            high=102 + index,
            low=99 + index,
            close=101 + index,
            volume=1_000_000,
            vwap=100.5 + index,
            adjusted=True,
        )
        for index in range(62)
    )
    writer = SilverWriter(LakehouseLayout(tmp_path))
    original = writer.write_daily_bars(
        bars[:61],
        ingested_at=datetime(2026, 3, 2, 20, tzinfo=UTC),
    )
    late_revision = writer.write_daily_bars(
        (
            EquityBar(
                symbol="AAPL",
                timestamp=bars[60].timestamp,
                open=200,
                high=202,
                low=199,
                close=201,
                volume=2_000_000,
                vwap=200.5,
                adjusted=True,
            ),
        ),
        ingested_at=datetime(2026, 3, 2, 23, tzinfo=UTC),
    )
    future = writer.write_daily_bars(
        (bars[61],),
        ingested_at=datetime(2026, 3, 2, 20, tzinfo=UTC),
    )

    contexts = DailyBarsFeatureLoader().load(
        tuple(item.path for item in (*original, *late_revision, *future)),
        symbols=("aapl",),
        asof_date=first + timedelta(days=60),
        observed_at=datetime(2026, 3, 2, 21, tzinfo=UTC),
    )

    assert len(contexts) == 1
    assert len(contexts[0].bars) == 61
    assert contexts[0].bars[-1].close == 161
