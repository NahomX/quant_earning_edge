"""Tests for deterministic layout and immutable bronze persistence."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from quant_earning_edge.data import BronzeWriter, DuckDBStore, LakehouseLayout


def test_layout_builds_expected_partitions(tmp_path: Path) -> None:
    layout = LakehouseLayout(tmp_path)

    assert layout.bronze(
        source="polygon",
        dataset="daily-bars",
        event_date=date(2026, 7, 27),
    ).relative_to(tmp_path.resolve()) == Path(
        "bronze/source=polygon/dataset=daily-bars/date=2026-07-27"
    )
    assert layout.silver(
        asset_class="us-equity",
        dataset="bars",
        event_date=date(2026, 7, 27),
    ).relative_to(tmp_path.resolve()) == Path(
        "silver/asset_class=us-equity/dataset=bars/date=2026-07-27"
    )
    assert layout.gold(
        feature_group="price",
        asof_month=date(2026, 7, 27),
    ).relative_to(tmp_path.resolve()) == Path(
        "gold/feature_group=price/month=2026-07"
    )


@pytest.mark.parametrize("unsafe", ["../escape", "a/b", "", "with space"])
def test_layout_rejects_unsafe_partition_values(tmp_path: Path, unsafe: str) -> None:
    with pytest.raises(ValueError, match="unsafe path"):
        LakehouseLayout(tmp_path).bronze(
            source=unsafe,
            dataset="bars",
            event_date=date(2026, 7, 27),
        )


def test_bronze_writer_is_canonical_and_idempotent(tmp_path: Path) -> None:
    writer = BronzeWriter(LakehouseLayout(tmp_path))
    received_at = datetime(2026, 7, 27, 12, 30, tzinfo=UTC)

    first = writer.write_json(
        {"ticker": "AAPL", "close": 210.5},
        source="polygon",
        dataset="daily-bars",
        event_date=date(2026, 7, 26),
        received_at=received_at,
    )
    second = writer.write_json(
        {"close": 210.5, "ticker": "AAPL"},
        source="polygon",
        dataset="daily-bars",
        event_date=date(2026, 7, 26),
        received_at=received_at,
    )

    assert first == second
    assert json.loads(first.path.read_text(encoding="utf-8")) == {
        "close": 210.5,
        "ticker": "AAPL",
    }
    assert len(list(first.path.parent.glob("*.json"))) == 1
    assert first.byte_count == first.path.stat().st_size


def test_bronze_writer_requires_aware_timestamp(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        BronzeWriter(LakehouseLayout(tmp_path)).write_json(
            {},
            source="polygon",
            dataset="bars",
            event_date=date(2026, 7, 27),
            received_at=datetime(2026, 7, 27),
        )


def test_duckdb_store_persists_and_queries(tmp_path: Path) -> None:
    database = tmp_path / "catalog.duckdb"

    with DuckDBStore(database) as store:
        store.execute("CREATE TABLE observations (ticker VARCHAR, close DOUBLE)")
        store.execute("INSERT INTO observations VALUES (?, ?)", ("AAPL", 210.5))

    with DuckDBStore(database) as store:
        row = store.execute(
            "SELECT ticker, close FROM observations WHERE ticker = ?",
            ("AAPL",),
        ).fetchone()

    assert row == ("AAPL", 210.5)
