"""Environment loading with secret-safe validation."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import urlparse

from dotenv import dotenv_values

if TYPE_CHECKING:
    from collections.abc import Mapping


class RuntimeConfigurationError(RuntimeError):
    """Required operational configuration is absent or unsafe."""


@dataclass(frozen=True)
class RuntimeEnvironment:
    """Non-secret runtime settings plus redacted provider credentials."""

    data_lake_root: Path
    polygon_base_url: str
    finnhub_base_url: str
    alpaca_trading_base_url: str
    http_timeout_seconds: float
    _polygon_api_key: str | None = field(default=None, repr=False)
    _finnhub_api_key: str | None = field(default=None, repr=False)
    _alpaca_api_key_id: str | None = field(default=None, repr=False)
    _alpaca_secret_key: str | None = field(default=None, repr=False)

    def require_polygon_api_key(self) -> str:
        """Return the Polygon key or fail without echoing secret material."""
        if not self._polygon_api_key:
            raise RuntimeConfigurationError("POLYGON_API_KEY is required for this command")
        return self._polygon_api_key

    def require_finnhub_api_key(self) -> str:
        """Return the Finnhub key or fail without echoing secret material."""
        if not self._finnhub_api_key:
            raise RuntimeConfigurationError("FINNHUB_API_KEY is required for this command")
        return self._finnhub_api_key

    def require_alpaca_credentials(self) -> tuple[str, str]:
        """Return paper-account credentials or fail without exposing them."""
        if not self._alpaca_api_key_id or not self._alpaca_secret_key:
            raise RuntimeConfigurationError(
                "APCA_API_KEY_ID and APCA_API_SECRET_KEY are required for this command"
            )
        return self._alpaca_api_key_id, self._alpaca_secret_key


def load_runtime_environment(*, env_file: Path | None = None) -> RuntimeEnvironment:
    """Load `.env` without overriding process variables and validate endpoints."""
    selected_file = env_file or Path(".env")
    file_values = dotenv_values(selected_file) if selected_file.exists() else {}

    def setting(name: str, default: str | None = None) -> str | None:
        process_value = os.getenv(name)
        if process_value is not None:
            return process_value
        file_value = file_values.get(name)
        return str(file_value) if file_value is not None else default

    timeout = _positive_float(setting("HTTP_TIMEOUT_SECONDS", "30") or "")
    polygon_base_url = (
        setting(
            "POLYGON_BASE_URL",
            "https://api.polygon.io",
        )
        or ""
    )
    finnhub_base_url = (
        setting(
            "FINNHUB_BASE_URL",
            "https://finnhub.io/api/v1",
        )
        or ""
    )
    alpaca_trading_base_url = setting("ALPACA_BASE_URL", "https://paper-api.alpaca.markets") or ""
    _require_https_url(polygon_base_url, field_name="POLYGON_BASE_URL")
    _require_https_url(finnhub_base_url, field_name="FINNHUB_BASE_URL")
    _require_https_url(alpaca_trading_base_url, field_name="ALPACA_BASE_URL")
    return RuntimeEnvironment(
        data_lake_root=Path(setting("DATA_LAKE_ROOT", "./data") or "").expanduser().resolve(),
        polygon_base_url=polygon_base_url,
        finnhub_base_url=finnhub_base_url,
        alpaca_trading_base_url=alpaca_trading_base_url,
        http_timeout_seconds=timeout,
        _polygon_api_key=_clean_secret(setting("POLYGON_API_KEY")),
        _finnhub_api_key=_clean_secret(setting("FINNHUB_API_KEY")),
        _alpaca_api_key_id=_clean_secret(setting("APCA_API_KEY_ID")),
        _alpaca_secret_key=_clean_secret(setting("APCA_API_SECRET_KEY")),
    )


def load_subprocess_environment(*, env_file: Path | None = None) -> Mapping[str, str]:
    """Merge dotenv values into a child-only environment without mutating this process."""
    selected_file = env_file or Path(".env")
    child = dict(os.environ)
    if selected_file.exists():
        for key, value in dotenv_values(selected_file).items():
            if value is not None and key not in child:
                child[key] = str(value)
    return child


def _clean_secret(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = value.strip()
    return cleaned or None


def _positive_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as error:
        raise RuntimeConfigurationError("HTTP_TIMEOUT_SECONDS must be numeric") from error
    if parsed <= 0:
        raise RuntimeConfigurationError("HTTP_TIMEOUT_SECONDS must be positive")
    return parsed


def _require_https_url(value: str, *, field_name: str) -> None:
    parsed = urlparse(value)
    if parsed.scheme != "https" or not parsed.hostname:
        raise RuntimeConfigurationError(f"{field_name} must be an HTTPS URL")
