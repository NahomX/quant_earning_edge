"""Operational readiness is explicit, immutable, and secret-free."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime, timedelta
from typing import TYPE_CHECKING

from typer.testing import CliRunner

from quant_earning_edge.cli import app
from quant_earning_edge.data.calendar import SessionFile
from quant_earning_edge.data.clients import MarketSession
from quant_earning_edge.monitoring import ProviderFreshnessEvidence
from quant_earning_edge.orchestration import (
    OperationalReadinessEvaluator,
    WorkerCycleReport,
)

if TYPE_CHECKING:
    from pathlib import Path


def _calendar(tmp_path: Path, *, count: int = 90) -> SessionFile:
    start = date(2026, 1, 1)
    dates = tuple(start + timedelta(days=index) for index in range(count))
    return SessionFile(
        path=tmp_path / "sessions.json",
        sha256="a" * 64,
        sessions=tuple(
            MarketSession(
                session_date=item,
                open_at=datetime(item.year, item.month, item.day, 14, 30, tzinfo=UTC),
                close_at=datetime(item.year, item.month, item.day, 21, 0, tzinfo=UTC),
            )
            for item in dates
        ),
    )


def test_all_operational_readiness_checks_can_pass(tmp_path: Path) -> None:
    now = datetime(2026, 1, 2, 14, 20, tzinfo=UTC)
    calendar = _calendar(tmp_path)
    data_lake = tmp_path / "lake"
    artifacts = tmp_path / "artifacts"
    inbox = tmp_path / "inbox"
    for path in (data_lake, artifacts, inbox):
        path.mkdir()
    freshness = ProviderFreshnessEvidence(
        schema_version=1,
        evaluated_at=now,
        polygon_symbol="SPY",
        polygon_data_observed_at=now - timedelta(minutes=1),
        polygon_payload_sha256="b" * 64,
        polygon_request_id="polygon",
        alpaca_data_observed_at=now - timedelta(minutes=1),
        alpaca_payload_sha256="c" * 64,
        alpaca_request_id="alpaca",
    )
    heartbeat = WorkerCycleReport(
        schema_version=1,
        worker_id="worker-1",
        evaluated_at=now - timedelta(seconds=30),
        inbox_path=str(inbox),
        results=(),
    )

    report = OperationalReadinessEvaluator().evaluate(
        calendar=calendar,
        control_date=calendar.sessions[1].session_date,
        evaluated_at=now,
        minimum_calendar_sessions=90,
        data_lake_root=data_lake,
        artifact_root=artifacts,
        inbox=inbox,
        polygon_base_url="https://api.polygon.io",
        finnhub_base_url="https://finnhub.io/api/v1",
        alpaca_base_url="https://paper-api.alpaca.markets",
        polygon_credential_configured=True,
        finnhub_credential_configured=True,
        alpaca_credentials_configured=True,
        provider_freshness=freshness,
        provider_probe_error=None,
        polygon_nbbo_entitlement_verified=True,
        polygon_nbbo_entitlement_detail="historical quote probe returned data",
        worker_id="worker-1",
        latest_worker_cycle=heartbeat,
        maximum_heartbeat_age=timedelta(minutes=5),
        bootstrap_source_count=1,
        bootstrap_error=None,
    )
    output = tmp_path / "readiness.json"
    report.write(output)
    report.write(output)

    assert report.ready
    assert all(item.passed for item in report.checks)
    assert json.loads(output.read_bytes())["ready"]


def test_readiness_cli_persists_failures_without_credentials(
    tmp_path: Path,
) -> None:
    calendar = _calendar(tmp_path, count=1)
    calendar.path.write_text(
        json.dumps(
            {
                "provider": "alpaca",
                "sessions": [
                    {
                        "session_date": calendar.sessions[0].session_date.isoformat(),
                        "open_at": calendar.sessions[0].open_at.isoformat(),
                        "close_at": calendar.sessions[0].close_at.isoformat(),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    data_lake = tmp_path / "lake"
    artifacts = tmp_path / "artifacts"
    inbox = tmp_path / "inbox"
    for path in (data_lake, artifacts, inbox):
        path.mkdir()
    env_file = tmp_path / ".env"
    env_file.write_text(f"DATA_LAKE_ROOT={data_lake}", encoding="utf-8")
    output = tmp_path / "failed-readiness.json"

    result = CliRunner().invoke(
        app,
        [
            "workflow",
            "audit-readiness",
            "--session-file",
            str(calendar.path),
            "--control-date",
            calendar.sessions[0].session_date.isoformat(),
            "--artifact-root",
            str(artifacts),
            "--inbox",
            str(inbox),
            "--worker-id",
            "worker-1",
            "--minimum-calendar-sessions",
            "1",
            "--output",
            str(output),
            "--env-file",
            str(env_file),
        ],
    )

    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert not payload["ready"]
    assert "polygon_credential_configured" in payload["failed_checks"]
    encoded = output.read_text(encoding="utf-8")
    assert "POLYGON_API_KEY" not in encoded
