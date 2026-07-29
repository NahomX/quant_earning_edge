"""Paper-only Alpaca order submission with idempotent client identifiers."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from datetime import UTC, date, datetime
from decimal import Decimal
from typing import TYPE_CHECKING, Any, Literal
from urllib.parse import urlparse

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from quant_earning_edge.data.clients.errors import ProviderRequestError, ProviderResponseError

if TYPE_CHECKING:
    from pathlib import Path

    from quant_earning_edge.data.bronze import BronzeArtifact, BronzeWriter
    from quant_earning_edge.monitoring import CircuitBreakerDecision

_PAPER_BASE_URL = "https://paper-api.alpaca.markets"


class PaperOrderRequest(BaseModel):
    """A simple whole-share US-equity paper order."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    client_order_id: str = Field(min_length=1, max_length=128)
    symbol: str = Field(min_length=1)
    quantity: int = Field(gt=0)
    side: Literal["buy", "sell"]
    order_type: Literal["market", "limit"] = "market"
    time_in_force: Literal["day", "opg", "cls", "ioc", "fok"] = "day"
    limit_price: float | None = Field(default=None, gt=0)
    extended_hours: bool = False

    @model_validator(mode="after")
    def validate_order(self) -> PaperOrderRequest:
        symbol = self.symbol.strip().upper()
        client_order_id = self.client_order_id.strip()
        if not symbol or not client_order_id:
            raise ValueError("symbol and client_order_id must not be blank")
        object.__setattr__(self, "symbol", symbol)
        object.__setattr__(self, "client_order_id", client_order_id)
        if (self.order_type == "limit") != (self.limit_price is not None):
            raise ValueError("limit_price must be supplied exactly for limit orders")
        if self.extended_hours and (self.order_type != "limit" or self.time_in_force != "day"):
            raise ValueError("extended-hours orders must be day limit orders")
        if self.limit_price is not None and not math.isfinite(self.limit_price):
            raise ValueError("limit_price must be finite")
        return self

    def payload(self) -> dict[str, str | bool]:
        payload: dict[str, str | bool] = {
            "client_order_id": self.client_order_id,
            "symbol": self.symbol,
            "qty": str(self.quantity),
            "side": self.side,
            "type": self.order_type,
            "time_in_force": self.time_in_force,
            "extended_hours": self.extended_hours,
        }
        if self.limit_price is not None:
            payload["limit_price"] = format(self.limit_price, ".15g")
        return payload


class BrokerOrder(BaseModel):
    """Validated subset of Alpaca's order resource needed for reconciliation."""

    model_config = ConfigDict(extra="ignore", frozen=True, populate_by_name=True)

    order_id: str = Field(alias="id", min_length=1)
    client_order_id: str = Field(min_length=1)
    symbol: str = Field(min_length=1)
    asset_class: Literal["us_equity"]
    quantity: Decimal = Field(alias="qty", gt=0)
    filled_quantity: Decimal = Field(alias="filled_qty", ge=0)
    filled_average_price: Decimal | None = Field(default=None, alias="filled_avg_price", gt=0)
    side: Literal["buy", "sell"]
    order_type: Literal["market", "limit"] = Field(alias="type")
    time_in_force: str = Field(min_length=1)
    limit_price: Decimal | None = Field(default=None, gt=0)
    extended_hours: bool = False
    status: str = Field(min_length=1)
    submitted_at: datetime
    filled_at: datetime | None = None

    @model_validator(mode="after")
    def validate_order(self) -> BrokerOrder:
        if self.submitted_at.tzinfo is None or self.submitted_at.utcoffset() is None:
            raise ValueError("broker submitted_at must be timezone-aware")
        if self.filled_at is not None and (
            self.filled_at.tzinfo is None or self.filled_at.utcoffset() is None
        ):
            raise ValueError("broker filled_at must be timezone-aware")
        if self.filled_quantity > self.quantity:
            raise ValueError("broker filled quantity exceeds order quantity")
        if (self.filled_quantity > 0) != (self.filled_average_price is not None):
            raise ValueError("broker average fill price is inconsistent with filled quantity")
        if self.quantity != self.quantity.to_integral_value():
            raise ValueError("broker order quantity must be whole-share")
        if self.filled_quantity != self.filled_quantity.to_integral_value():
            raise ValueError("broker filled quantity must be whole-share")
        object.__setattr__(self, "symbol", self.symbol.strip().upper())
        return self


class PaperAccountSnapshot(BaseModel):
    """Causal paper-account equity used for portfolio sizing."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    captured_at: datetime
    equity: float = Field(gt=0)
    buying_power: float = Field(ge=0)
    status: Literal["ACTIVE"]
    trading_blocked: bool
    payload_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    provider_request_id: str | None = None

    @model_validator(mode="after")
    def validate_account(self) -> PaperAccountSnapshot:
        if self.captured_at.tzinfo is None or self.captured_at.utcoffset() is None:
            raise ValueError("paper account capture time must be timezone-aware")
        if self.trading_blocked:
            raise ValueError("paper account is blocked from trading")
        return self


class _PaperAccountPayload(BaseModel):
    model_config = ConfigDict(extra="ignore")

    equity: Decimal = Field(gt=0)
    buying_power: Decimal = Field(ge=0)
    status: Literal["ACTIVE"]
    trading_blocked: bool


@dataclass(frozen=True)
class PaperSubmission:
    """Immutable paper-order request/response audit."""

    schema_version: int
    request: PaperOrderRequest
    broker_order: BrokerOrder
    provider_request_id: str | None
    idempotent_reuse: bool

    @property
    def canonical_bytes(self) -> bytes:
        return json.dumps(
            {
                "schema_version": self.schema_version,
                "request": self.request.model_dump(mode="json"),
                "broker_order": self.broker_order.model_dump(mode="json"),
                "provider_request_id": self.provider_request_id,
                "idempotent_reuse": self.idempotent_reuse,
            },
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
                raise RuntimeError(f"paper-submission evidence collision at {output}") from None


class PaperOrderBatchSpec(BaseModel):
    """A complete, deterministic session order batch, including explicit no-trade days."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    session_date: date
    orders: tuple[PaperOrderRequest, ...] = ()

    @model_validator(mode="after")
    def validate_batch(self) -> PaperOrderBatchSpec:
        client_ids = tuple(order.client_order_id for order in self.orders)
        if client_ids != tuple(sorted(set(client_ids))):
            raise ValueError("batch client_order_id values must be unique and sorted")
        return self

    @property
    def canonical_bytes(self) -> bytes:
        return json.dumps(
            self.model_dump(mode="json"),
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
                raise RuntimeError(f"paper-order batch collision at {output}") from None


@dataclass(frozen=True)
class PaperBatchSubmission:
    """All-or-resume session evidence; written only after every order is verified."""

    schema_version: int
    session_date: date
    breaker_decision_sha256: str
    submissions: tuple[PaperSubmission, ...]

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("unsupported paper batch schema version")
        if len(self.breaker_decision_sha256) != 64:
            raise ValueError("paper batch breaker digest must be SHA-256")
        client_ids = tuple(item.request.client_order_id for item in self.submissions)
        if client_ids != tuple(sorted(set(client_ids))):
            raise ValueError("paper batch submissions must have unique sorted client ids")
        if any(
            item.broker_order.submitted_at.date() != self.session_date for item in self.submissions
        ):
            raise ValueError("paper batch broker submission dates must match the session date")

    @property
    def canonical_bytes(self) -> bytes:
        return json.dumps(
            {
                "schema_version": self.schema_version,
                "session_date": self.session_date.isoformat(),
                "breaker_decision_sha256": self.breaker_decision_sha256,
                "submissions": [
                    {
                        "schema_version": item.schema_version,
                        "request": item.request.model_dump(mode="json"),
                        "broker_order": item.broker_order.model_dump(mode="json"),
                        "provider_request_id": item.provider_request_id,
                        "idempotent_reuse": item.idempotent_reuse,
                    }
                    for item in self.submissions
                ],
            },
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
                raise RuntimeError(f"paper-batch evidence collision at {output}") from None


class PaperBatchSubmitter:
    """Submit a sorted session plan and safely resume after partial process failure."""

    def __init__(self, client: AlpacaPaperClient) -> None:
        self._client = client

    def submit(
        self,
        spec: PaperOrderBatchSpec,
        *,
        breaker_decision: CircuitBreakerDecision,
        evaluated_at: datetime,
    ) -> PaperBatchSubmission:
        if evaluated_at.tzinfo is None or evaluated_at.utcoffset() is None:
            raise ValueError("paper batch evaluated_at must be timezone-aware")
        if breaker_decision.halt_new_orders:
            raise ValueError("paper batch refused because circuit breakers halt new orders")
        if breaker_decision.session_date != spec.session_date:
            raise ValueError("paper batch and circuit-breaker session dates must match")
        decision_age = (
            evaluated_at - breaker_decision.evaluated_at.astimezone(evaluated_at.tzinfo)
        ).total_seconds()
        if not 0 <= decision_age <= 30 * 60:
            raise ValueError("paper batch requires a breaker decision no more than 30 minutes old")
        submissions = tuple(self._client.submit(order) for order in spec.orders)
        return PaperBatchSubmission(
            schema_version=1,
            session_date=spec.session_date,
            breaker_decision_sha256=breaker_decision.sha256,
            submissions=submissions,
        )


class AlpacaPaperClient:
    """Submit and query orders only against Alpaca's dedicated paper host."""

    def __init__(
        self,
        *,
        api_key_id: str,
        secret_key: str,
        http_client: httpx.Client,
        bronze_writer: BronzeWriter | None = None,
    ) -> None:
        if not api_key_id.strip() or not secret_key.strip():
            raise ValueError("Alpaca paper credentials must not be empty")
        parsed = urlparse(str(http_client.base_url).rstrip("/"))
        if parsed.scheme != "https" or parsed.netloc != "paper-api.alpaca.markets" or parsed.path:
            raise ValueError("paper order client requires exactly https://paper-api.alpaca.markets")
        self._api_key_id = api_key_id
        self._secret_key = secret_key
        self._http = http_client
        self._bronze_writer = bronze_writer
        self._observation_artifacts: list[BronzeArtifact] = []

    @property
    def observation_artifacts(self) -> tuple[BronzeArtifact, ...]:
        """Return raw provider observations captured by this client instance."""
        return tuple(self._observation_artifacts)

    def submit(self, request: PaperOrderRequest) -> PaperSubmission:
        """Reuse an identical client ID or place exactly one new paper order."""
        existing_response = self._http.get(
            "/v2/orders:by_client_order_id",
            params={"client_order_id": request.client_order_id},
            headers=self._headers,
        )
        if existing_response.status_code == 404:
            response = self._http.post(
                "/v2/orders",
                json=request.payload(),
                headers=self._headers,
            )
            self._raise_for_status(response)
            raw = self._decode(response)
            reused = False
        else:
            self._raise_for_status(existing_response)
            response = existing_response
            raw = self._decode(response)
            reused = True
        broker_order = self._broker_order(raw)
        self._validate_match(request, broker_order)
        if self._bronze_writer is not None:
            self._observation_artifacts.append(
                self._bronze_writer.write_json(
                    raw,
                    source="alpaca-paper",
                    dataset="orders",
                    event_date=broker_order.submitted_at.date(),
                )
            )
        return PaperSubmission(
            schema_version=1,
            request=request,
            broker_order=broker_order,
            provider_request_id=response.headers.get("X-Request-ID"),
            idempotent_reuse=reused,
        )

    def get_by_client_order_id(self, client_order_id: str) -> BrokerOrder:
        """Retrieve one paper order for after-close reconciliation."""
        normalized = client_order_id.strip()
        if not normalized:
            raise ValueError("client_order_id must not be empty")
        response = self._http.get(
            "/v2/orders:by_client_order_id",
            params={"client_order_id": normalized},
            headers=self._headers,
        )
        self._raise_for_status(response)
        raw = self._decode(response)
        broker_order = self._broker_order(raw)
        if self._bronze_writer is not None:
            self._observation_artifacts.append(
                self._bronze_writer.write_json(
                    raw,
                    source="alpaca-paper",
                    dataset="orders",
                    event_date=broker_order.submitted_at.date(),
                )
            )
        return broker_order

    def account_snapshot(
        self,
        *,
        captured_at: datetime | None = None,
    ) -> PaperAccountSnapshot:
        """Capture current paper equity and fail closed on blocked/non-active accounts."""
        captured = (captured_at or datetime.now(UTC)).astimezone(UTC)
        response = self._http.get("/v2/account", headers=self._headers)
        self._raise_for_status(response)
        raw = self._decode(response)
        try:
            account = _PaperAccountPayload.model_validate(raw)
            payload = json.dumps(raw, sort_keys=True, separators=(",", ":")).encode()
            snapshot = PaperAccountSnapshot(
                captured_at=captured,
                equity=float(account.equity),
                buying_power=float(account.buying_power),
                status=account.status,
                trading_blocked=account.trading_blocked,
                payload_sha256=hashlib.sha256(payload).hexdigest(),
                provider_request_id=response.headers.get("X-Request-ID"),
            )
        except ValidationError as error:
            raise ProviderResponseError(
                f"Alpaca paper account failed validation: {error}"
            ) from error
        if self._bronze_writer is not None:
            self._bronze_writer.write_json(
                raw,
                source="alpaca-paper",
                dataset="account",
                event_date=captured.date(),
            )
        return snapshot

    @property
    def _headers(self) -> dict[str, str]:
        return {
            "APCA-API-KEY-ID": self._api_key_id,
            "APCA-API-SECRET-KEY": self._secret_key,
        }

    @staticmethod
    def _validate_match(request: PaperOrderRequest, order: BrokerOrder) -> None:
        expected_limit = Decimal(str(request.limit_price)) if request.limit_price else None
        if (
            order.client_order_id != request.client_order_id
            or order.symbol != request.symbol
            or order.quantity != Decimal(request.quantity)
            or order.side != request.side
            or order.order_type != request.order_type
            or order.time_in_force != request.time_in_force
            or order.limit_price != expected_limit
            or order.extended_hours != request.extended_hours
        ):
            raise ProviderResponseError(
                "existing Alpaca paper order does not match the immutable request"
            )

    @staticmethod
    def _broker_order(raw: Any) -> BrokerOrder:
        try:
            return BrokerOrder.model_validate(raw)
        except ValidationError as error:
            raise ProviderResponseError(f"Alpaca paper order failed validation: {error}") from error

    @staticmethod
    def _decode(response: httpx.Response) -> Any:
        try:
            return response.json()
        except ValueError as error:
            raise ProviderResponseError("Alpaca paper order response was not JSON") from error

    @staticmethod
    def _raise_for_status(response: httpx.Response) -> None:
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as error:
            raise ProviderRequestError(
                f"Alpaca paper order request failed with HTTP {response.status_code}"
            ) from error
