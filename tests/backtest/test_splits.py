"""Leakage-safe walk-forward split and manifest tests."""

from __future__ import annotations

import json
from datetime import date, timedelta
from typing import TYPE_CHECKING

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from quant_earning_edge.backtest import (
    LabeledSample,
    PurgedWalkForwardSplitter,
    WalkForwardConfig,
    WalkForwardPlanner,
)

if TYPE_CHECKING:
    from pathlib import Path


def _samples(session_count: int = 30) -> tuple[LabeledSample, ...]:
    first = date(2025, 1, 2)
    sessions = tuple(first + timedelta(days=index) for index in range(session_count))
    return tuple(
        LabeledSample(
            symbol=symbol,
            asof_date=session,
            horizon_end_date=session + timedelta(days=5),
        )
        for session in sessions
        for symbol in ("AAA", "BBB")
    )


def test_purged_walk_forward_has_no_horizon_overlap() -> None:
    samples = _samples()
    config = WalkForwardConfig(
        minimum_train_sessions=10,
        embargo_sessions=5,
        test_sessions=5,
    )

    folds = PurgedWalkForwardSplitter(config).split(samples)

    assert len(folds) == 3
    for fold in folds:
        train = tuple(samples[index] for index in fold.train_indices)
        test = tuple(samples[index] for index in fold.test_indices)
        assert len(fold.embargo_dates) == config.embargo_sessions
        assert len(test) == config.test_sessions * 2
        assert max(item.horizon_end_date for item in train) < fold.test_start_date
        assert {item.asof_date for item in train}.isdisjoint(item.asof_date for item in test)


def test_splitter_is_deterministic_for_unsorted_samples() -> None:
    samples = _samples()
    reordered = tuple(reversed(samples))
    config = WalkForwardConfig(10, 5, 5)

    first = PurgedWalkForwardSplitter(config).split(reordered)
    second = PurgedWalkForwardSplitter(config).split(reordered)

    assert first == second


def test_splitter_rejects_duplicate_keys() -> None:
    samples = _samples()

    with pytest.raises(ValueError, match="duplicate"):
        PurgedWalkForwardSplitter(WalkForwardConfig(10, 5, 5)).split((*samples, samples[0]))


def test_planner_persists_content_addressed_manifest(tmp_path: Path) -> None:
    dataset = tmp_path / "training.parquet"
    output = tmp_path / "plans" / "walk-forward.json"
    samples = _samples()
    table = pa.table(
        {
            "symbol": [item.symbol for item in samples],
            "asof_date": pa.array(
                [item.asof_date for item in samples],
                type=pa.date32(),
            ),
            "horizon_end_date": pa.array(
                [item.horizon_end_date for item in samples],
                type=pa.date32(),
            ),
        }
    )
    pq.write_table(table, dataset)  # type: ignore[no-untyped-call]
    planner = WalkForwardPlanner()

    plan = planner.build(dataset_files=(dataset,), config=WalkForwardConfig(10, 5, 5))
    planner.write(plan, output)
    planner.write(plan, output)

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["sample_count"] == 60
    assert len(payload["folds"]) == 3
    assert plan.sha256


def test_planner_rejects_schema_mismatch(tmp_path: Path) -> None:
    dataset = tmp_path / "invalid.parquet"
    pq.write_table(  # type: ignore[no-untyped-call]
        pa.table({"symbol": ["AAA"]}),
        dataset,
    )

    with pytest.raises(ValueError, match="asof_date"):
        WalkForwardPlanner().build(
            dataset_files=(dataset,),
            config=WalkForwardConfig(1, 1, 1),
        )
