"""Deployment assets keep the persistent worker constrained and restartable."""

from __future__ import annotations

from pathlib import Path


def test_systemd_worker_is_persistent_non_root_and_filesystem_constrained() -> None:
    root = Path(__file__).parents[2]
    service = (root / "ops/systemd/qee-workflow-worker.service").read_text(encoding="utf-8")

    assert "User=qee" in service
    assert "workflow worker" in service
    assert "--loop-spec /etc/quant_earning_edge/phase6-loop.json" in service
    assert "Restart=always" in service
    assert "NoNewPrivileges=true" in service
    assert "ProtectSystem=strict" in service
    assert "ReadWritePaths=/var/lib/quant_earning_edge" in service


def test_windows_worker_uses_project_venv_and_no_embedded_secrets() -> None:
    root = Path(__file__).parents[2]
    script = (root / "ops/run-workflow-worker.ps1").read_text(encoding="utf-8")

    assert ".venv\\Scripts\\qee.exe" in script
    assert "workflow worker" in script
    assert "[string]$LoopSpec" in script
    assert "--loop-spec $ResolvedLoopSpec" in script
    assert "--env-file" in script
    assert "while ($true)" in script
    assert "Start-Sleep -Seconds $RestartDelaySeconds" in script
    assert "[int]$RestartDelaySeconds = 5" in script
    assert "APCA_API_SECRET_KEY" not in script
    assert "POLYGON_API_KEY=" not in script
