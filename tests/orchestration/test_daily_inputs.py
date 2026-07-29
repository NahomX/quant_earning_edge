"""State-driven provider-backed daily input preparation tests."""

from __future__ import annotations

import hashlib
import json
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
    session_file = _sessions(tmp_path)
    candidate_root = tmp_path / "gold" / "event-candidates" / "for_trade_date=2026-07-28"
    candidate_root.mkdir(parents=True)
    candidate = candidate_root / "candidates-empty.parquet"
    pq.write_table(  # type: ignore[no-untyped-call]
        pa.Table.from_pylist([], schema=EVENT_CANDIDATE_SCHEMA),
        candidate,
    )
    source_paths = tuple(
        tmp_path / f"{name}.source"
        for name in (
            "universe",
            "earnings",
            "splits",
            "dividends",
            "universe-manifest",
            "universe-provider",
            "event-manifest",
            "event-provider",
            "calendar-manifest",
            "calendar-provider",
        )
    )
    for index, path in enumerate(source_paths):
        path.write_bytes(f"source-{index}".encode())

    def entry(path: Path) -> dict[str, str]:
        return {
            "path": path.relative_to(tmp_path).as_posix(),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }

    (
        universe,
        earnings,
        splits,
        dividends,
        universe_manifest,
        universe_provider,
        event_manifest,
        event_provider,
        calendar_manifest,
        calendar_provider,
    ) = (entry(path) for path in source_paths)
    session = entry(session_file)
    split_hash = hashlib.sha256(splits["sha256"].encode()).hexdigest()
    dividend_hash = hashlib.sha256(dividends["sha256"].encode()).hexdigest()
    manifest = {
        "schema_version": 5,
        "trade_date": "2026-07-28",
        "decision_at": "2026-07-28T01:30:00+00:00",
        "records": [],
        "candidate_file_sha256": hashlib.sha256(candidate.read_bytes()).hexdigest(),
        "universe_snapshot_sha256": universe["sha256"],
        "session_file_sha256": session["sha256"],
        "earnings_input_sha256": hashlib.sha256(earnings["sha256"].encode()).hexdigest(),
        "corporate_actions_input_sha256": hashlib.sha256(
            f"{split_hash}{dividend_hash}".encode()
        ).hexdigest(),
        "candidate_split_overlap_count": 0,
        "candidate_dividend_overlap_count": 0,
        "excluded_counts": {},
        "source_files": {
            "universe_source_manifest": universe_manifest,
            "universe_source_files": [universe_provider],
            "event_source_manifest": event_manifest,
            "event_provider_files": [event_provider],
            "calendar_source_manifest": calendar_manifest,
            "calendar_provider_files": [calendar_provider],
            "universe_snapshot": universe,
            "session_file": session,
            "earnings_files": [earnings],
            "split_files": [splits],
            "dividend_files": [dividends],
        },
    }
    candidate.with_name("manifest-empty.json").write_bytes(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    )

    result = _preparer(
        tmp_path,
        datetime(2026, 7, 28, 1, 30, tzinfo=UTC),
    ).run(
        trade_date=date(2026, 7, 28),
        session_file=session_file,
        halt_snapshot_file=tmp_path / "unused-halts.json",
        universe_config_file=tmp_path / "unused-universe.yaml",
        feature_names=("signal",),
        feature_group="live",
    )

    assert result.status is DailyInputStatus.FEATURES_READY
    assert result.candidate_file == candidate.resolve()
    assert result.feature_file is None
    assert result.candidate_count == 0
