"""Alpaca trading-calendar client with validated session boundaries."""

from __future__ import annotations

import time
from datetime import date, datetime
from datetime import time as wall_time
from typing import TYPE_CHECKING, Any
from zoneinfo import ZoneInfo

import httpx
from pydantic import BaseModel, ConfigDict, ValidationError, field_validator

from quant_earning_edge.data.clients.errors import ProviderRequestError, ProviderResponseError

if TYPE_CHECKING:
    from collections.abc import Callable

    from quant_earning_edge.data.bronze import BronzeArtifact, BronzeWriter

_EASTERN = ZoneInfo("America/New_York")
_RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})


class MarketSession(BaseModel):
    """One authoritative regular US-equity trading session."""

    model_config = ConfigDict(frozen=True)

    session_date: date
    open_at: datetime
    close_at: datetime

    @field_validator("open_at", "close_at")
    @classmethod
    def require_aware_timestamp(cls, value: datetime) -> datetime:
        """Reject timestamps without an explicit offset."""
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("session timestamps must be timezone-aware")
        return value

    @field_validator("close_at")
    @classmethod
    def require_close_after_open(
        cls,
        value: datetime,
        info: Any,
    ) -> datetime:
        """Reject inverted or empty market sessions."""
        open_at = info.data.get("open_at")
        if open_at is not None and value <= open_at:
            raise ValueError("close_at must be after open_at")
        return value


class _CalendarDay(BaseModel):
    model_config = ConfigDict(extra="ignore")

    date: date
    open: str
    close: str


class AlpacaCalendarClient:
    """Fetch the account-authenticated Alpaca market calendar."""

    _ENDPOINT = "/v2/calendar"

    def __init__(
        self,
        *,
        api_key_id: str,
        secret_key: str,
        http_client: httpx.Client,
        bronze_writer: BronzeWriter | None = None,
        max_attempts: int = 3,
        retry_delay_seconds: float = 0.25,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        if not api_key_id.strip() or not secret_key.strip():
            raise ValueError("Alpaca API credentials must not be empty")
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        if retry_delay_seconds < 0:
            raise ValueError("retry_delay_seconds must not be negative")
        self._api_key_id = api_key_id
        self._secret_key = secret_key
        self._http = http_client
        self._bronze_writer = bronze_writer
        self._max_attempts = max_attempts
        self._retry_delay_seconds = retry_delay_seconds
        self._sleeper = sleeper
        self._calendar_observation_artifacts: list[BronzeArtifact] = []

    @property
    def calendar_observation_artifacts(self) -> tuple[BronzeArtifact, ...]:
        """Return retained raw market-calendar responses."""
        return tuple(self._calendar_observation_artifacts)

    @classmethod
    def sessions_from_payload(
        cls,
        raw: Any,
        *,
        start_date: date,
        end_date: date,
    ) -> tuple[MarketSession, ...]:
        """Reconstruct authoritative sessions without a provider request."""
        try:
            days = tuple(_CalendarDay.model_validate(item) for item in raw)
        except (TypeError, ValidationError) as error:
            raise ProviderResponseError(
                f"Alpaca calendar response failed validation: {error}"
            ) from error
        sessions = tuple(cls._to_session(day) for day in days)
        dates = tuple(item.session_date for item in sessions)
        if dates != tuple(sorted(set(dates))):
            raise ProviderResponseError("Alpaca calendar dates must be unique and ascending")
        if any(item < start_date or item > end_date for item in dates):
            raise ProviderResponseError("Alpaca returned a session outside the requested range")
        return sessions

    def sessions(self, *, start_date: date, end_date: date) -> tuple[MarketSession, ...]:
        """Return regular sessions in an inclusive calendar-date range."""
        if end_date < start_date:
            raise ValueError("end_date must be on or after start_date")
        response = self._request(
            params={"start": start_date.isoformat(), "end": end_date.isoformat()}
        )
        try:
            raw = response.json()
        except ValueError as error:
            raise ProviderResponseError("Alpaca returned invalid JSON") from error
        if self._bronze_writer is not None:
            self._calendar_observation_artifacts.append(
                self._bronze_writer.write_json(
                    raw,
                    source="alpaca",
                    dataset="market-calendar",
                    event_date=start_date,
                )
            )
        return self.sessions_from_payload(
            raw,
            start_date=start_date,
            end_date=end_date,
        )

    def _request(self, *, params: dict[str, str]) -> httpx.Response:
        headers = {
            "APCA-API-KEY-ID": self._api_key_id,
            "APCA-API-SECRET-KEY": self._secret_key,
        }
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
                            f"Alpaca request failed with HTTP {response.status_code}"
                        ) from error
                    return response
                last_error = None
            if attempt < self._max_attempts:
                self._sleeper(self._retry_delay_seconds * attempt)
        detail = f": {last_error}" if last_error is not None else " after retryable HTTP responses"
        raise ProviderRequestError(
            f"Alpaca request failed after {self._max_attempts} attempts{detail}"
        )

    @staticmethod
    def _to_session(day: _CalendarDay) -> MarketSession:
        try:
            open_time = wall_time.fromisoformat(day.open)
            close_time = wall_time.fromisoformat(day.close)
        except ValueError as error:
            raise ProviderResponseError("Alpaca returned an invalid session time") from error
        return MarketSession(
            session_date=day.date,
            open_at=datetime.combine(day.date, open_time, tzinfo=_EASTERN),
            close_at=datetime.combine(day.date, close_time, tzinfo=_EASTERN),
        )
