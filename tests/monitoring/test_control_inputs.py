"""Breaker controls retain completed replay-session provenance."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

from typer.testing import CliRunner

import quant_earning_edge.cli as cli_module
from quant_earning_edge.cli import app
from quant_earning_edge.data import LakehouseLayout, SessionFileStore
from quant_earning_edge.data.clients import MarketSession
from quant_earning_edge.evaluation import ReplaySessionAggregator
from quant_earning_edge.monitoring import (
    CircuitBreakerEvaluationSpec,
    CircuitBreakerEvaluator,
    ProviderFreshnessEvidence,
)
from quant_earning_edge.signals import (
    DailyOrderPlanningSpec,
    LiveOrderPlanner,
    load_strategy_config,
    strategy_file_sha256,
)

if TYPE_CHECKING:
    import pytest


def _strategy_path() -> Path:
    return Path(__file__).parents[2] / "configs/strategies/earnings_v1.yaml"


def _source(tmp_path: Path, session_date: date) -> tuple[Path, Path]:
    opened = datetime(
        session_date.year,
        session_date.month,
        session_date.day,
        13,
        30,
        tzinfo=UTC,
    )
    planning = DailyOrderPlanningSpec(
        trade_date=session_date,
        decision_at=opened - timedelta(hours=12),
        equity=100_000,
        entry_submitted_at=opened,
        entry_expires_at=opened + timedelta(minutes=5),
        exit_submitted_at=opened + timedelta(hours=6),
        exit_expires_at=opened + timedelta(hours=6, minutes=5),
    )
    strategy_path = _strategy_path()
    frozen = LiveOrderPlanner(
        load_strategy_config(strategy_path),
        strategy_sha256=strategy_file_sha256(strategy_path),
    ).plan(planning)
    frozen_path = tmp_path / f"{session_date}-frozen.json"
    frozen.write(frozen_path)
    report = ReplaySessionAggregator().evaluate(
        evidence=(),
        round_trips=(),
        session_date=session_date,
        initial_cash=100_000,
    )
    report_path = tmp_path / f"{session_date}-replay.json"
    report.write(report_path)
    return frozen_path, report_path


def test_prepare_breaker_controls_keeps_control_and_replay_dates_distinct(
    tmp_path: Path,
) -> None:
    first = _source(tmp_path, date(2026, 7, 27))
    latest = _source(tmp_path, date(2026, 7, 28))
    output = tmp_path / "breaker-controls.json"
    evaluated = datetime(2026, 7, 29, 13, 20, tzinfo=UTC)
    arguments = [
        "monitoring",
        "prepare-breaker-controls",
        "--control-date",
        "2026-07-29",
        "--evaluated-at",
        evaluated.isoformat(),
        "--polygon-data-observed-at",
        (evaluated - timedelta(minutes=1)).isoformat(),
        "--alpaca-data-observed-at",
        (evaluated - timedelta(minutes=2)).isoformat(),
        "--output",
        str(output),
    ]
    for frozen, report in (first, latest):
        arguments.extend(("--frozen-orders", str(frozen)))
        arguments.extend(("--replay-report", str(report)))

    result = CliRunner().invoke(app, arguments)
    repeated = CliRunner().invoke(app, arguments)

    assert result.exit_code == 0
    assert repeated.exit_code == 0
    spec = CircuitBreakerEvaluationSpec.model_validate_json(output.read_bytes())
    assert tuple(item.session_date for item in spec.observations) == (
        date(2026, 7, 27),
        date(2026, 7, 29),
    )
    assert tuple(item.replay_source_date for item in spec.observations) == (
        date(2026, 7, 27),
        date(2026, 7, 28),
    )
    decision = CircuitBreakerEvaluator().evaluate(
        tuple(item.to_domain() for item in spec.observations)
    )
    assert decision.replay_source_dates == (
        date(2026, 7, 27),
        date(2026, 7, 28),
    )
    assert json.loads(result.stdout)["control_date"] == "2026-07-29"


def test_prepare_breaker_evidence_consumes_freshness_and_calendar(
    tmp_path: Path,
) -> None:
    frozen, replay = _source(tmp_path, date(2026, 7, 28))
    evaluated = datetime(2026, 7, 29, 13, 20, tzinfo=UTC)
    freshness_path = tmp_path / "freshness.json"
    ProviderFreshnessEvidence(
        schema_version=1,
        evaluated_at=evaluated,
        polygon_symbol="SPY",
        polygon_data_observed_at=evaluated - timedelta(minutes=1),
        polygon_payload_sha256="a" * 64,
        polygon_request_id="polygon",
        alpaca_data_observed_at=evaluated - timedelta(minutes=2),
        alpaca_payload_sha256="b" * 64,
        alpaca_request_id="alpaca",
    ).write(freshness_path)
    calendar = SessionFileStore(LakehouseLayout(tmp_path / "calendar")).write(
        (
            MarketSession(
                session_date=date(2026, 7, 28),
                open_at=datetime(2026, 7, 28, 13, 30, tzinfo=UTC),
                close_at=datetime(2026, 7, 28, 20, 0, tzinfo=UTC),
            ),
            MarketSession(
                session_date=date(2026, 7, 29),
                open_at=datetime(2026, 7, 29, 13, 30, tzinfo=UTC),
                close_at=datetime(2026, 7, 29, 20, 0, tzinfo=UTC),
            ),
        )
    )
    output = tmp_path / "breaker.json"
    age_output = tmp_path / "age.json"

    result = CliRunner().invoke(
        app,
        [
            "monitoring",
            "prepare-breaker-evidence",
            "--control-date",
            "2026-07-29",
            "--freshness-file",
            str(freshness_path),
            "--session-file",
            str(calendar.path),
            "--frozen-orders",
            str(frozen),
            "--replay-report",
            str(replay),
            "--reconciliation-age-output",
            str(age_output),
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0
    spec = CircuitBreakerEvaluationSpec.model_validate_json(output.read_bytes())
    assert spec.observations[-1].replay_source_date == date(2026, 7, 28)
    assert spec.observations[-1].session_date == date(2026, 7, 29)
    assert json.loads(age_output.read_bytes())["reconciliation_break_age_sessions"] is None


def test_prepare_breaker_bundle_discovers_sources_and_writes_content_addresses(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_frozen, source_replay = _source(tmp_path, date(2026, 7, 28))
    artifact_root = tmp_path / "artifacts"
    daily = artifact_root / "trade_date=2026-07-28"
    daily.mkdir(parents=True)
    (daily / "frozen-daily-orders.json").write_bytes(source_frozen.read_bytes())
    (daily / "replay-session.json").write_bytes(source_replay.read_bytes())
    evaluated = datetime(2026, 7, 29, 13, 20, tzinfo=UTC)
    freshness = ProviderFreshnessEvidence(
        schema_version=1,
        evaluated_at=evaluated,
        polygon_symbol="SPY",
        polygon_data_observed_at=evaluated - timedelta(minutes=1),
        polygon_payload_sha256="a" * 64,
        polygon_request_id="polygon",
        alpaca_data_observed_at=evaluated - timedelta(minutes=2),
        alpaca_payload_sha256="b" * 64,
        alpaca_request_id="alpaca",
    )
    raw_paths = (tmp_path / "polygon-freshness.json", tmp_path / "alpaca-clock.json")
    for raw_path in raw_paths:
        raw_path.write_text("{}", encoding="utf-8")

    def capture_freshness(
        *,
        observation_paths: list[Path] | None = None,
        **_: object,
    ) -> ProviderFreshnessEvidence:
        if observation_paths is not None:
            observation_paths.extend(raw_paths)
        return freshness

    monkeypatch.setattr(cli_module, "_capture_provider_freshness", capture_freshness)
    calendar = SessionFileStore(LakehouseLayout(tmp_path / "calendar")).write(
        (
            MarketSession(
                session_date=date(2026, 7, 28),
                open_at=datetime(2026, 7, 28, 13, 30, tzinfo=UTC),
                close_at=datetime(2026, 7, 28, 20, 0, tzinfo=UTC),
            ),
            MarketSession(
                session_date=date(2026, 7, 29),
                open_at=datetime(2026, 7, 29, 13, 30, tzinfo=UTC),
                close_at=datetime(2026, 7, 29, 20, 0, tzinfo=UTC),
            ),
        )
    )
    output_directory = daily.parent / "trade_date=2026-07-29" / "control-evidence"
    arguments = [
        "monitoring",
        "prepare-breaker-bundle",
        "--control-date",
        "2026-07-29",
        "--session-file",
        str(calendar.path),
        "--artifact-root",
        str(artifact_root),
        "--output-directory",
        str(output_directory),
    ]

    result = CliRunner().invoke(app, arguments)
    repeated = CliRunner().invoke(app, arguments)

    assert result.exit_code == 0
    assert repeated.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload == json.loads(repeated.stdout)
    assert Path(payload["freshness_path"]).name.startswith("provider-freshness-")
    assert payload["freshness_observation_paths"] == [str(path.resolve()) for path in raw_paths]
    assert Path(payload["reconciliation_age_path"]).name.startswith("reconciliation-age-")
    breaker_path = Path(payload["breaker_spec_path"])
    assert breaker_path.name.startswith("breaker-controls-")
    spec = CircuitBreakerEvaluationSpec.model_validate_json(breaker_path.read_bytes())
    assert spec.observations[-1].session_date == date(2026, 7, 29)
