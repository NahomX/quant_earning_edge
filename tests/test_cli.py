"""CLI validation and exit-code tests without external network access."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from typer.testing import CliRunner

from quant_earning_edge import __version__
from quant_earning_edge.cli import app

if TYPE_CHECKING:
    from pathlib import Path

    import pytest

runner = CliRunner()


def test_version_command() -> None:
    result = runner.invoke(app, ["version"])

    assert result.exit_code == 0
    assert result.stdout.strip() == __version__


def test_bars_command_missing_credential_exits_nonzero(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("POLYGON_API_KEY", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text(f"DATA_LAKE_ROOT={tmp_path / 'lake'}", encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "ingest",
            "bars",
            "--symbol",
            "AAPL",
            "--start",
            "2026-07-01",
            "--end",
            "2026-07-27",
            "--env-file",
            str(env_file),
        ],
    )

    assert result.exit_code == 2
    assert "POLYGON_API_KEY is required" in result.stderr


def test_invalid_date_exits_nonzero_before_network(
    tmp_path: Path,
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "POLYGON_API_KEY=test-only",
                f"DATA_LAKE_ROOT={tmp_path / 'lake'}",
            ]
        ),
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        [
            "ingest",
            "bars",
            "--symbol",
            "AAPL",
            "--start",
            "not-a-date",
            "--end",
            "2026-07-27",
            "--env-file",
            str(env_file),
        ],
    )

    assert result.exit_code == 2
    assert "YYYY-MM-DD" in result.stderr


def test_readiness_command_reports_missing_real_runs(
    tmp_path: Path,
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(f"DATA_LAKE_ROOT={tmp_path / 'lake'}", encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "universe",
            "readiness",
            "--trade-date",
            "2026-07-20",
            "--trade-date",
            "2026-07-21",
            "--env-file",
            str(env_file),
        ],
    )

    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert payload["ready"] is False
    assert payload["successful_trade_dates"] == []


def test_calendar_command_requires_both_credentials(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "APCA_API_KEY_ID=test-only",
                f"DATA_LAKE_ROOT={tmp_path / 'lake'}",
            ]
        ),
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        [
            "calendar",
            "sessions",
            "--start",
            "2026-07-01",
            "--end",
            "2026-07-31",
            "--env-file",
            str(env_file),
        ],
    )

    assert result.exit_code == 2
    assert "APCA_API_KEY_ID" in result.stderr
    assert "APCA_API_SECRET_KEY" in result.stderr
    assert "required" in result.stderr
