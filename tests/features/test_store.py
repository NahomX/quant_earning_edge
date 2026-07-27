"""Feature registry, engine, and deterministic gold-store tests."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from typing import TYPE_CHECKING

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from quant_earning_edge.data import LakehouseLayout, SilverWriter
from quant_earning_edge.data.clients import EarningsEvent, EquityBar
from quant_earning_edge.features import (
    FEATURE_REGISTRY,
    FEATURE_VALUE_SCHEMA,
    DailyBarsFeatureLoader,
    EarningsFeatureLoader,
    EarningsObservation,
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


def test_registry_has_real_metadata_and_fifteen_baseline_features() -> None:
    specs = FEATURE_REGISTRY.values()

    assert len(specs) == 15
    assert {item.name for item in specs} == {
        "days_since_last_earnings",
        "distance_to_vwap_20d",
        "distance_to_high_52w",
        "earnings_timing_flag",
        "kalman_volume_30d",
        "kalman_volume_7d",
        "macd_signal_12_26_9",
        "prior_eps_surprise_pct",
        "realized_vol_20d",
        "realized_vol_60d",
        "relative_volume_30d",
        "return_1d",
        "return_20d",
        "return_5d",
        "rsi_14",
    }
    assert all(len(item.code_hash) == 64 for item in specs)
    assert all(item.dependencies for item in specs)


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


def test_volume_momentum_and_event_features_are_causal_and_well_defined() -> None:
    first = date(2025, 1, 1)
    bars = tuple(
        PriceBar(
            session_date=first + timedelta(days=index),
            close=100.0 + index,
            volume=1_000_000.0 + index * 1_000,
            vwap=99.5 + index,
        )
        for index in range(252)
    )
    asof_date = bars[-1].session_date
    context = FeatureContext(
        symbol="AAPL",
        asof_date=asof_date,
        bars=bars,
        earnings=(
            EarningsObservation(
                event_date=asof_date - timedelta(days=90),
                effective_trade_date=asof_date - timedelta(days=90),
                timing="amc",
                eps_actual=1.2,
                eps_estimate=1.0,
            ),
            EarningsObservation(
                event_date=asof_date,
                effective_trade_date=asof_date,
                timing="bmo",
            ),
        ),
    )
    names = (
        "kalman_volume_7d",
        "kalman_volume_30d",
        "relative_volume_30d",
        "rsi_14",
        "macd_signal_12_26_9",
        "distance_to_high_52w",
        "earnings_timing_flag",
        "days_since_last_earnings",
        "prior_eps_surprise_pct",
    )
    values = {
        item.feature_name: item.value
        for item in FeatureEngine().compute((context,), feature_names=names)
    }

    assert values["kalman_volume_7d"] > values["kalman_volume_30d"] > 0
    assert values["relative_volume_30d"] > 1
    assert values["rsi_14"] == 100
    assert values["distance_to_high_52w"] == 0
    assert values["earnings_timing_flag"] == 1
    assert values["days_since_last_earnings"] == 90
    assert values["prior_eps_surprise_pct"] == pytest.approx(0.2)


def test_earnings_loader_uses_candidate_timing_and_only_prior_known_results(
    tmp_path: Path,
) -> None:
    trade_date = date(2026, 7, 28)
    cutoff = datetime(2026, 7, 28, 1, 30, tzinfo=UTC)
    candidate_path = tmp_path / "candidate.parquet"
    pq.write_table(  # type: ignore[no-untyped-call]
        pa.Table.from_pylist(
            [
                {
                    "trade_date": trade_date,
                    "symbol": "AAPL",
                    "event_date": trade_date,
                    "timing": "bmo",
                    "decision_at": cutoff,
                }
            ]
        ),
        candidate_path,
    )
    writer = SilverWriter(LakehouseLayout(tmp_path))
    prior = EarningsEvent.model_validate(
        {
            "date": "2026-04-28",
            "symbol": "AAPL",
            "hour": "amc",
            "year": 2026,
            "quarter": 2,
            "epsActual": 1.2,
            "epsEstimate": 1.0,
        }
    )
    current_result = EarningsEvent.model_validate(
        {
            "date": trade_date,
            "symbol": "AAPL",
            "hour": "bmo",
            "year": 2026,
            "quarter": 3,
            "epsActual": 9.0,
            "epsEstimate": 1.0,
        }
    )
    prior_files = writer.write_earnings(
        (prior,),
        ingested_at=datetime(2026, 4, 28, 22, tzinfo=UTC),
    )
    future_files = writer.write_earnings(
        (current_result,),
        ingested_at=datetime(2026, 7, 28, 14, tzinfo=UTC),
    )
    base = FeatureContext(
        symbol="AAPL",
        asof_date=trade_date,
        bars=_context().bars,
    )

    enriched = EarningsFeatureLoader().enrich(
        (base,),
        candidate_files=(candidate_path,),
        earnings_files=tuple(item.path for item in (*prior_files, *future_files)),
        observed_at=cutoff,
    )[0]

    assert len(enriched.earnings) == 2
    assert enriched.earnings[-1].eps_actual is None
    assert FEATURE_REGISTRY.get("prior_eps_surprise_pct").evaluate(enriched) == pytest.approx(0.2)
