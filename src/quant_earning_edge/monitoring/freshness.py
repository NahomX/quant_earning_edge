"""Immutable provider-timestamp evidence for pre-submit freshness controls."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from quant_earning_edge.data.clients.errors import ProviderRequestError, ProviderResponseError

if TYPE_CHECKING:
    from pathlib import Path

    from quant_earning_edge.data.bronze import BronzeWriter


class _AlpacaClock(BaseModel):
    model_config = ConfigDict(extra="ignore")

    timestamp: datetime


class _PolygonSnapshotTicker(BaseModel):
    model_config = ConfigDict(extra="ignore")

    ticker: str
    updated: int = Field(gt=0)


class _PolygonSnapshot(BaseModel):
    model_config = ConfigDict(extra="ignore")

    status: str
    request_id: str | None = None
    ticker: _PolygonSnapshotTicker


@dataclass(frozen=True)
class ProviderFreshnessEvidence:
    """Canonical provider-native timestamps and payload identities."""

    schema_version: int
    evaluated_at: datetime
    polygon_symbol: str
    polygon_data_observed_at: datetime
    polygon_payload_sha256: str
    polygon_request_id: str | None
    alpaca_data_observed_at: datetime
    alpaca_payload_sha256: str
    alpaca_request_id: str | None

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("unsupported provider freshness schema")
        for name, value in (
            ("evaluated_at", self.evaluated_at),
            ("polygon_data_observed_at", self.polygon_data_observed_at),
            ("alpaca_data_observed_at", self.alpaca_data_observed_at),
        ):
            if value.tzinfo is None or value.utcoffset() is None:
                raise ValueError(f"{name} must be timezone-aware")
        if (
            self.polygon_data_observed_at > self.evaluated_at
            or self.alpaca_data_observed_at > self.evaluated_at
        ):
            raise ValueError("provider observation cannot be after evaluation")
        if not self.polygon_symbol.strip() or self.polygon_symbol != self.polygon_symbol.upper():
            raise ValueError("Polygon probe symbol must be normalized")
        for digest in (self.polygon_payload_sha256, self.alpaca_payload_sha256):
            if len(digest) != 64 or any(item not in "0123456789abcdef" for item in digest):
                raise ValueError("provider payload hash must be SHA-256")

    @property
    def canonical_bytes(self) -> bytes:
        return json.dumps(
            asdict(self),
            default=lambda item: item.isoformat(),
            sort_keys=True,
            separators=(",", ":"),
        ).encode()

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes).hexdigest()

    def write(self, output: Path) -> None:
        encoded = self.canonical_bytes
        output.parent.mkdir(parents=True, exist_ok=True)
        try:
            with output.open("xb") as destination:
                destination.write(encoded)
        except FileExistsError:
            if output.read_bytes() != encoded:
                raise RuntimeError(f"provider freshness collision at {output}") from None

    @classmethod
    def load(cls, path: Path) -> ProviderFreshnessEvidence:
        try:
            raw = json.loads(path.read_bytes())
            evidence = cls(
                schema_version=int(raw["schema_version"]),
                evaluated_at=datetime.fromisoformat(raw["evaluated_at"]),
                polygon_symbol=str(raw["polygon_symbol"]),
                polygon_data_observed_at=datetime.fromisoformat(raw["polygon_data_observed_at"]),
                polygon_payload_sha256=str(raw["polygon_payload_sha256"]),
                polygon_request_id=raw["polygon_request_id"],
                alpaca_data_observed_at=datetime.fromisoformat(raw["alpaca_data_observed_at"]),
                alpaca_payload_sha256=str(raw["alpaca_payload_sha256"]),
                alpaca_request_id=raw["alpaca_request_id"],
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"invalid provider freshness evidence: {path}") from error
        if json.loads(evidence.canonical_bytes) != raw:
            raise ValueError("provider freshness evidence is not canonical")
        return evidence


class ProviderFreshnessProbe:
    """Read provider-native clocks without treating local receipt time as data time."""

    def __init__(
        self,
        *,
        polygon_api_key: str,
        alpaca_api_key_id: str,
        alpaca_secret_key: str,
        polygon_http: httpx.Client,
        alpaca_http: httpx.Client,
        bronze_writer: BronzeWriter | None = None,
    ) -> None:
        if not all(
            item.strip() for item in (polygon_api_key, alpaca_api_key_id, alpaca_secret_key)
        ):
            raise ValueError("provider freshness credentials must not be blank")
        if _host(polygon_http) != "api.polygon.io":
            raise ValueError("freshness probe requires Polygon's canonical API host")
        if _host(alpaca_http) != "paper-api.alpaca.markets":
            raise ValueError("freshness probe requires Alpaca's canonical paper host")
        self._polygon_key = polygon_api_key
        self._alpaca_key = alpaca_api_key_id
        self._alpaca_secret = alpaca_secret_key
        self._polygon_http = polygon_http
        self._alpaca_http = alpaca_http
        self._bronze = bronze_writer

    def probe(
        self,
        *,
        symbol: str,
        evaluated_at: datetime | None = None,
    ) -> ProviderFreshnessEvidence:
        normalized = symbol.strip().upper()
        if not normalized:
            raise ValueError("Polygon freshness symbol must not be blank")
        polygon_response = self._polygon_http.get(
            f"/v2/snapshot/locale/us/markets/stocks/tickers/{normalized}",
            headers={"Authorization": f"Bearer {self._polygon_key}"},
        )
        alpaca_response = self._alpaca_http.get(
            "/v2/clock",
            headers={
                "APCA-API-KEY-ID": self._alpaca_key,
                "APCA-API-SECRET-KEY": self._alpaca_secret,
            },
        )
        polygon_raw = _decode(polygon_response, provider="Polygon")
        alpaca_raw = _decode(alpaca_response, provider="Alpaca")
        try:
            polygon = _PolygonSnapshot.model_validate(polygon_raw)
            alpaca = _AlpacaClock.model_validate(alpaca_raw)
        except ValidationError as error:
            raise ProviderResponseError(
                f"provider freshness response failed validation: {error}"
            ) from error
        if polygon.status.upper() != "OK" or polygon.ticker.ticker.strip().upper() != normalized:
            raise ProviderResponseError("Polygon freshness snapshot identity is invalid")
        observed_at = _from_nanoseconds(polygon.ticker.updated)
        evaluated = evaluated_at or datetime.now(UTC)
        evidence = ProviderFreshnessEvidence(
            schema_version=1,
            evaluated_at=evaluated,
            polygon_symbol=normalized,
            polygon_data_observed_at=observed_at,
            polygon_payload_sha256=_payload_hash(polygon_raw),
            polygon_request_id=polygon.request_id,
            alpaca_data_observed_at=alpaca.timestamp,
            alpaca_payload_sha256=_payload_hash(alpaca_raw),
            alpaca_request_id=alpaca_response.headers.get("X-Request-ID"),
        )
        if self._bronze is not None:
            self._bronze.write_json(
                polygon_raw,
                source="polygon",
                dataset="freshness-snapshot",
                event_date=evaluated.date(),
            )
            self._bronze.write_json(
                alpaca_raw,
                source="alpaca",
                dataset="market-clock",
                event_date=evaluated.date(),
            )
        return evidence


def _host(client: httpx.Client) -> str:
    parsed = urlparse(str(client.base_url).rstrip("/"))
    return parsed.netloc


def _decode(response: httpx.Response, *, provider: str) -> Any:
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as error:
        raise ProviderRequestError(
            f"{provider} freshness request failed with HTTP {response.status_code}"
        ) from error
    try:
        return response.json()
    except ValueError as error:
        raise ProviderResponseError(f"{provider} freshness response was not JSON") from error


def _from_nanoseconds(value: int) -> datetime:
    seconds, nanoseconds = divmod(value, 1_000_000_000)
    return datetime.fromtimestamp(seconds, tz=UTC).replace(microsecond=nanoseconds // 1_000)


def _payload_hash(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
