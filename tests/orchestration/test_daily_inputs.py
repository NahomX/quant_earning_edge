"""State-driven provider-backed daily input preparation tests."""

from __future__ import annotations

from datetime import UTC, date, datetime
from typing import TYPE_CHECKING, Any, cast

import pyarrow as pa
import pyarrow.parquet as pq

from quant_earning_edge.data import LakehouseLayout, SessionFileStore
from quant_earning_edge.data.clients import MarketSession
from quant_earning_edge.orchestration.daily_inputs import (
    DailyInputPreparer,
    DailyInputStatus,
)
from quant_earning_edge.universe import EVENT_CANDIDATE_SCHEMA

if TYPE_CHECKING:
    from pathlib import Path

    from quant_earning_edge.data.clients import FinnhubClient, PolygonClient


def _sessions(tmp_path: Path) -> Path:
    return (
        SessionFileStore(LakehouseLayout(tmp_path))
        .write(
            (
                MarketSession(
                    session_date=date(2026, 7, 27),
                    open_at=datetime(2026, 7, 27, 13, 30, tzinfo=UTC),
                    close_at=datetime(2026, 7, 27, 20, 0, tzinfo=UTC),
                ),
                MarketSession(
                    session_date=date(2026, 7, 28),
                    open_at=datetime(2026, 7, 28, 13, 30, tzinfo=UTC),
                    close_at=datetime(2026, 7, 28, 20, 0, tzinfo=UTC),
                ),
            )
        )
        .path
    )


def _preparer(tmp_path: Path, now: datetime) -> DailyInputPreparer:
    unavailable = cast("Any", object())
    return DailyInputPreparer(
        layout=LakehouseLayout(tmp_path),
        polygon=cast("PolygonClient", unavailable),
        finnhub=cast("FinnhubClient", unavailable),
        clock=lambda: now,
    )


def test_daily_inputs_wait_before_candidate_boundary(tmp_path: Path) -> None:
    result = _preparer(
        tmp_path,
        datetime(2026, 7, 27, 23, 0, tzinfo=UTC),
    ).run(
        trade_date=date(2026, 7, 28),
        session_file=_sessions(tmp_path),
        halt_snapshot_file=tmp_path / "unused-halts.json",
        universe_config_file=tmp_path / "unused-universe.yaml",
        feature_names=("signal",),
        feature_group="live",
    )

    assert result.status is DailyInputStatus.WAITING_FOR_PRIOR_CLOSE
    assert result.candidate_file is None


def test_zero_candidate_day_is_feature_ready_without_provider_calls(
    tmp_path: Path,
) -> None:
    candidate_root = tmp_path / "gold" / "event-candidates" / "for_trade_date=2026-07-28"
    candidate_root.mkdir(parents=True)
    candidate = candidate_root / "candidates-empty.parquet"
    pq.write_table(  # type: ignore[no-untyped-call]
        pa.Table.from_pylist([], schema=EVENT_CANDIDATE_SCHEMA),
        candidate,
    )

    result = _preparer(
        tmp_path,
        datetime(2026, 7, 28, 1, 30, tzinfo=UTC),
    ).run(
        trade_date=date(2026, 7, 28),
        session_file=_sessions(tmp_path),
        halt_snapshot_file=tmp_path / "unused-halts.json",
        universe_config_file=tmp_path / "unused-universe.yaml",
        feature_names=("signal",),
        feature_group="live",
    )

    assert result.status is DailyInputStatus.FEATURES_READY
    assert result.candidate_file == candidate.resolve()
    assert result.feature_file is None
    assert result.candidate_count == 0
