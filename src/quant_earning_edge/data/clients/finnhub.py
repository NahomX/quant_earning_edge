"""Finnhub earnings-calendar client with validation and raw capture."""

from __future__ import annotations

import time
from datetime import date  # noqa: TC003 - Pydantic resolves this field type at runtime.
from enum import StrEnum
from typing import TYPE_CHECKING, Any

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from quant_earning_edge.data.clients.errors import ProviderRequestError, ProviderResponseError

if TYPE_CHECKING:
    from collections.abc import Callable

    from quant_earning_edge.data.bronze import BronzeWriter

_RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})


class EarningsTiming(StrEnum):
    """Finnhub's documented release-time values."""

    BEFORE_MARKET_OPEN = "bmo"
    AFTER_MARKET_CLOSE = "amc"
    DURING_MARKET_HOURS = "dmh"


class EarningsEvent(BaseModel):
    """Validated earnings-calendar observation."""

    model_config = ConfigDict(extra="ignore", frozen=True, populate_by_name=True)

    event_date: date = Field(alias="date")
    symbol: str = Field(min_length=1)
    timing: EarningsTiming = Field(alias="hour")
    year: int = Field(ge=1900)
    quarter: int = Field(ge=1, le=4)
    eps_actual: float | None = Field(default=None, alias="epsActual")
    eps_estimate: float | None = Field(default=None, alias="epsEstimate")
    revenue_actual: float | None = Field(default=None, alias="revenueActual")
    revenue_estimate: float | None = Field(default=None, alias="revenueEstimate")

    @field_validator("symbol")
    @classmethod
    def normalize_symbol(cls, value: str) -> str:
        """Store a normalized symbol key across provider responses."""
        normalized = value.strip().upper()
        if not normalized:
            raise ValueError("symbol must not be blank")
        return normalized


class _EarningsCalendarResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    earnings_calendar: list[EarningsEvent] = Field(alias="earningsCalendar")


class FinnhubClient:
    """Read and validate Finnhub earnings events.

    The caller owns the injected HTTP client. Transient request failures and
    documented rate-limit/server statuses are retried with bounded backoff.
    """

    _ENDPOINT = "/calendar/earnings"

    def __init__(
        self,
        *,
        api_key: str,
        http_client: httpx.Client,
        bronze_writer: BronzeWriter | None = None,
        max_attempts: int = 3,
        retry_delay_seconds: float = 0.25,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        if not api_key.strip():
            raise ValueError("Finnhub API key must not be empty")
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        if retry_delay_seconds < 0:
            raise ValueError("retry_delay_seconds must not be negative")
        self._api_key = api_key
        self._http = http_client
        self._bronze_writer = bronze_writer
        self._max_attempts = max_attempts
        self._retry_delay_seconds = retry_delay_seconds
        self._sleeper = sleeper

    def earnings_calendar(
        self,
        *,
        start_date: date,
        end_date: date,
        symbol: str | None = None,
        include_international: bool = False,
    ) -> tuple[EarningsEvent, ...]:
        """Fetch events in an inclusive date interval."""
        if end_date < start_date:
            raise ValueError("end_date must be on or after start_date")

        params: dict[str, str] = {
            "from": start_date.isoformat(),
            "to": end_date.isoformat(),
            "international": str(include_international).lower(),
        }
        if symbol:
            params["symbol"] = symbol.upper()

        response = self._request(params)
        raw = self._decode_json(response)
        if self._bronze_writer is not None:
            self._bronze_writer.write_json(
                raw,
                source="finnhub",
                dataset="earnings-calendar",
                event_date=start_date,
            )

        try:
            validated = _EarningsCalendarResponse.model_validate(raw)
        except ValidationError as error:
            raise ProviderResponseError(
                f"Finnhub earnings response failed validation: {error}"
            ) from error
        return tuple(validated.earnings_calendar)

    def _request(self, params: dict[str, str]) -> httpx.Response:
        headers = {"X-Finnhub-Token": self._api_key}
        last_error: httpx.RequestError | None = None

        for attempt in range(1, self._max_attempts + 1):
            try:
                response = self._http.get(self._ENDPOINT, params=params, headers=headers)
            except httpx.RequestError as error:
                last_error = error
            else:
                if response.status_code not in _RETRYABLE_STATUS_CODES:
                    try:
                        response.raise_for_status()
                    except httpx.HTTPStatusError as error:
                        raise ProviderRequestError(
                            f"Finnhub request failed with HTTP {response.status_code}"
                        ) from error
                    return response
                last_error = None

            if attempt < self._max_attempts:
                self._sleeper(self._retry_delay_seconds * attempt)

        detail = f": {last_error}" if last_error is not None else " after retryable HTTP responses"
        raise ProviderRequestError(
            f"Finnhub request failed after {self._max_attempts} attempts{detail}"
        )

    @staticmethod
    def _decode_json(response: httpx.Response) -> Any:
        try:
            return response.json()
        except ValueError as error:
            raise ProviderResponseError("Finnhub returned invalid JSON") from error
