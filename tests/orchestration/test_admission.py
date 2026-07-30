"""The first scheduled proof session is fail-closed behind readiness and smoke."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime, timedelta
from typing import TYPE_CHECKING

from typer.testing import CliRunner

from quant_earning_edge.cli import app
from quant_earning_edge.data import LakehouseLayout, SessionFileStore
from quant_earning_edge.data.clients import MarketSession
from quant_earning_edge.orchestration import (
    DailyWorkflowSpecGenerator,
    NoTradeSmokeEvidence,
    OperationalReadinessReport,
    ReadinessCheck,
    WorkflowTrigger,
)

if TYPE_CHECKING:
    from pathlib import Path


def _inputs(
    tmp_path: Path,
    *,
    ready: bool = True,
) -> tuple[Path, Path, Path, date, datetime]:
    smoke_date = date(2026, 7, 27)
    proof_start = date(2026, 7, 28)
    sessions = tuple(
        MarketSession(
            session_date=item,
            open_at=datetime(item.year, item.month, item.day, 13, 30, tzinfo=UTC),
            close_at=datetime(item.year, item.month, item.day, 20, 0, tzinfo=UTC),
        )
        for item in (smoke_date, proof_start)
    )
    calendar = SessionFileStore(LakehouseLayout(tmp_path / "lake")).write(sessions)
    evaluated_at = datetime(2026, 7, 28, 12, 0, tzinfo=UTC)
    readiness = OperationalReadinessReport(
        schema_version=1,
        evaluated_at=evaluated_at,
        control_date=proof_start,
        session_file_sha256=calendar.sha256,
        provider_freshness_sha256="a" * 64 if ready else None,
        latest_worker_cycle_sha256="b" * 64 if ready else None,
        checks=(ReadinessCheck(name="all_prerequisites", passed=ready, detail="test"),),
        ready=ready,
    )
    readiness_path = tmp_path / "readiness.json"
    readiness.write(readiness_path)
    smoke = NoTradeSmokeEvidence(
        schema_version=1,
        smoke_date=smoke_date,
        trigger="manual",
        counts_toward_phase6=False,
        intended_order_count=0,
        workflow_spec_sha256="c" * 64,
        workflow_state_sha256="d" * 64,
        worker_cycle_sha256="e" * 64,
        artifact_root=str((tmp_path / "smoke").resolve()),
    )
    smoke_path = tmp_path / "smoke.json"
    smoke_path.write_bytes(smoke.canonical_bytes)
    staged = DailyWorkflowSpecGenerator().generate(
        trade_date=proof_start,
        trigger=WorkflowTrigger.SCHEDULED,
        worker_id="worker",
        planning_spec=tmp_path / "planning.json",
        strategy_config=tmp_path / "strategy.yaml",
        breaker_spec=tmp_path / "breakers.json",
        phase6_spec=tmp_path / "phase6.json",
        artifact_root=tmp_path / "artifacts",
        order_controls_not_before=evaluated_at,
        order_submission_not_after=evaluated_at + timedelta(hours=1),
        market_events_not_before=evaluated_at + timedelta(hours=9),
    )
    staged_path = tmp_path / "staged.json"
    staged.write(staged_path)
    return calendar.path, readiness_path, smoke_path, proof_start, evaluated_at


def test_admission_publishes_first_scheduled_spec_only_after_both_gates(
    tmp_path: Path,
) -> None:
    calendar, readiness, smoke, proof_start, evaluated_at = _inputs(tmp_path)
    staged = tmp_path / "staged.json"
    inbox = tmp_path / "inbox" / "first.json"
    evidence = tmp_path / "admission.json"

    result = CliRunner().invoke(
        app,
        [
            "workflow",
            "admit-proof-start",
            "--session-file",
            str(calendar),
            "--proof-start",
            proof_start.isoformat(),
            "--readiness-file",
            str(readiness),
            "--smoke-file",
            str(smoke),
            "--workflow-spec",
            str(staged),
            "--inbox-output",
            str(inbox),
            "--output",
            str(evidence),
            "--admitted-at",
            (evaluated_at + timedelta(minutes=5)).isoformat(),
        ],
    )

    assert result.exit_code == 0
    assert inbox.read_bytes() == staged.read_bytes()
    payload = json.loads(evidence.read_bytes())
    assert payload["proof_start"] == proof_start.isoformat()
    assert payload["readiness_sha256"]
    assert payload["smoke_sha256"]


def test_failed_readiness_never_publishes_first_spec(tmp_path: Path) -> None:
    calendar, readiness, smoke, proof_start, evaluated_at = _inputs(tmp_path, ready=False)
    inbox = tmp_path / "inbox" / "first.json"

    result = CliRunner().invoke(
        app,
        [
            "workflow",
            "admit-proof-start",
            "--session-file",
            str(calendar),
            "--proof-start",
            proof_start.isoformat(),
            "--readiness-file",
            str(readiness),
            "--smoke-file",
            str(smoke),
            "--workflow-spec",
            str(tmp_path / "staged.json"),
            "--inbox-output",
            str(inbox),
            "--output",
            str(tmp_path / "admission.json"),
            "--admitted-at",
            (evaluated_at + timedelta(minutes=5)).isoformat(),
        ],
    )

    assert result.exit_code == 2
    compact_error = "".join(result.stderr.split())
    assert "operationalreadinessreportdid" in compact_error
    assert "notpass" in compact_error
    assert not inbox.exists()
