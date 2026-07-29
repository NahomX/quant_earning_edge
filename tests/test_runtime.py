"""Secret-safe runtime environment tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from quant_earning_edge.runtime import (
    RuntimeConfigurationError,
    load_runtime_environment,
    load_subprocess_environment,
)


def test_environment_file_loads_without_exposing_secrets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("POLYGON_API_KEY", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "POLYGON_API_KEY=polygon-secret",
                "FINNHUB_API_KEY=finnhub-secret",
                f"DATA_LAKE_ROOT={tmp_path / 'lake'}",
            ]
        ),
        encoding="utf-8",
    )

    environment = load_runtime_environment(env_file=env_file)

    assert environment.require_polygon_api_key() == "polygon-secret"
    assert environment.require_finnhub_api_key() == "finnhub-secret"
    assert environment.data_lake_root == (tmp_path / "lake").resolve()
    assert "polygon-secret" not in repr(environment)
    assert "finnhub-secret" not in repr(environment)


def test_process_environment_takes_precedence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("POLYGON_API_KEY=file-key", encoding="utf-8")
    monkeypatch.setenv("POLYGON_API_KEY", "process-key")

    environment = load_runtime_environment(env_file=env_file)
    child = load_subprocess_environment(env_file=env_file)

    assert environment.require_polygon_api_key() == "process-key"
    assert child["POLYGON_API_KEY"] == "process-key"


def test_missing_key_and_unsafe_endpoint_fail_cleanly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("FINNHUB_API_KEY", raising=False)
    empty = tmp_path / "missing.env"
    with pytest.raises(RuntimeConfigurationError, match="FINNHUB_API_KEY"):
        load_runtime_environment(env_file=empty).require_finnhub_api_key()

    unsafe = tmp_path / ".env"
    unsafe.write_text("POLYGON_BASE_URL=http://example.test", encoding="utf-8")
    with pytest.raises(RuntimeConfigurationError, match="HTTPS"):
        load_runtime_environment(env_file=unsafe)


def test_example_uses_runtime_alpaca_credential_names() -> None:
    example = (Path(__file__).parents[1] / ".env.example").read_text(encoding="utf-8")

    assert "APCA_API_KEY_ID=" in example
    assert "APCA_API_SECRET_KEY=" in example
    assert "ALPACA_API_KEY_ID=" not in example
    assert "ALPACA_API_SECRET_KEY=" not in example
