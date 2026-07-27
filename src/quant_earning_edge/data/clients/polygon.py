"""Polygon/Massive daily aggregate-bars client."""

from __future__ import annotations

import time
from datetime import UTC, date, datetime
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Literal
from urllib.parse import urlparse
from zoneinfo import ZoneInfo

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from quant_earning_edge.data.clients.errors import ProviderRequestError, ProviderResponseError

if TYPE_CHECKING:
    from collections.abc import Callable

    from quant_earning_edge.data.bronze import BronzeWriter

_MARKET_TIMEZONE = ZoneInfo("America/New_York")
_RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})


class EquityBar(BaseModel):
    """Validated, split-adjusted US-equity daily aggregate."""

    model_config = ConfigDict(frozen=True)

    symbol: str = Field(min_length=1)
    timestamp: datetime
    open: float = Field(gt=0)
    high: float = Field(gt=0)
    low: float = Field(gt=0)
    close: float = Field(gt=0)
    volume: float = Field(ge=0)
    vwap: float | None = Field(default=None, gt=0)
    transactions: int | None = Field(default=None, ge=0)
    adjusted: bool

    @model_validator(mode="after")
    def validate_price_bounds(self) -> EquityBar:
        """Require the high/low envelope to contain open and close."""
        if self.low > self.high:
            raise ValueError("low must not exceed high")
        if not self.low <= self.open <= self.high:
            raise ValueError("open must be within low/high")
        if not self.low <= self.close <= self.high:
            raise ValueError("close must be within low/high")
        if self.timestamp.tzinfo is None or self.timestamp.utcoffset() is None:
            raise ValueError("timestamp must be timezone-aware")
        return self

    @property
    def session_date(self) -> date:
        """Trading-session date in the provider's documented Eastern Time."""
        return self.timestamp.astimezone(_MARKET_TIMEZONE).date()


class TickerDetails(BaseModel):
    """Point-in-time security metadata used by universe construction."""

    model_config = ConfigDict(frozen=True)

    symbol: str = Field(min_length=1)
    asof_date: date
    name: str = Field(min_length=1)
    active: bool
    locale: Literal["us"]
    market: Literal["stocks"]
    primary_exchange: str = Field(min_length=1)
    security_type: str = Field(min_length=1)
    market_cap: float = Field(gt=0)
    list_date: date | None = None
    delisted_date: date | None = None


class TickerReference(BaseModel):
    """Historical ticker identity returned by the all-tickers endpoint."""

    model_config = ConfigDict(frozen=True)

    symbol: str = Field(min_length=1)
    asof_date: date
    name: str = Field(min_length=1)
    active: bool
    locale: Literal["us"]
    market: Literal["stocks"]
    primary_exchange: str
    security_type: str
    delisted_date: date | None = None


class SplitAdjustmentType(StrEnum):
    """Massive's current share-change classifications."""

    FORWARD_SPLIT = "forward_split"
    REVERSE_SPLIT = "reverse_split"
    STOCK_DIVIDEND = "stock_dividend"


class DividendDistributionType(StrEnum):
    """Massive's current cash-distribution classifications."""

    RECURRING = "recurring"
    SPECIAL = "special"
    SUPPLEMENTAL = "supplemental"
    IRREGULAR = "irregular"
    UNKNOWN = "unknown"


class StockSplit(BaseModel):
    """Validated corporate-action split observation."""

    model_config = ConfigDict(frozen=True)

    event_id: str = Field(min_length=1)
    symbol: str = Field(min_length=1)
    execution_date: date
    adjustment_type: SplitAdjustmentType
    split_from: float = Field(gt=0)
    split_to: float = Field(gt=0)
    historical_adjustment_factor: float | None = Field(default=None, gt=0)


class CashDividend(BaseModel):
    """Validated cash-dividend observation."""

    model_config = ConfigDict(frozen=True)

    event_id: str = Field(min_length=1)
    symbol: str = Field(min_length=1)
    ex_dividend_date: date
    distribution_type: DividendDistributionType
    cash_amount: float = Field(gt=0)
    currency: str = Field(min_length=3, max_length=3)
    frequency: int = Field(ge=0)
    declaration_date: date | None = None
    record_date: date | None = None
    pay_date: date | None = None
    split_adjusted_cash_amount: float | None = Field(default=None, gt=0)
    historical_adjustment_factor: float | None = Field(default=None, gt=0)


class _Aggregate(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    open: float = Field(alias="o")
    high: float = Field(alias="h")
    low: float = Field(alias="l")
    close: float = Field(alias="c")
    volume: float = Field(alias="v")
    timestamp_ms: int = Field(alias="t", ge=0)
    vwap: float | None = Field(default=None, alias="vw")
    transactions: int | None = Field(default=None, alias="n")


class _AggregatesResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    ticker: str = Field(min_length=1)
    adjusted: bool
    status: str
    results: list[_Aggregate] = Field(default_factory=list)
    next_url: str | None = None


class _TickerDetailsPayload(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    ticker: str = Field(min_length=1)
    name: str = Field(min_length=1)
    active: bool
    locale: Literal["us"]
    market: Literal["stocks"]
    primary_exchange: str = Field(min_length=1)
    security_type: str = Field(alias="type", min_length=1)
    market_cap: float = Field(gt=0)
    list_date: date | None = None
    delisted_date: date | None = Field(default=None, alias="delisted_utc")


class _TickerDetailsResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    status: str
    results: _TickerDetailsPayload


class _TickerReferencePayload(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    ticker: str = Field(min_length=1)
    name: str = Field(min_length=1)
    active: bool
    locale: Literal["us"]
    market: Literal["stocks"]
    primary_exchange: str = ""
    security_type: str = Field(default="", alias="type")
    delisted_date: date | None = Field(default=None, alias="delisted_utc")


class _TickersResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    status: str
    results: list[_TickerReferencePayload] = Field(default_factory=list)
    next_url: str | None = None


class _SplitPayload(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    event_id: str = Field(alias="id", min_length=1)
    symbol: str = Field(alias="ticker", min_length=1)
    execution_date: date
    adjustment_type: SplitAdjustmentType
    split_from: float = Field(gt=0)
    split_to: float = Field(gt=0)
    historical_adjustment_factor: float | None = Field(default=None, gt=0)


class _DividendPayload(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    event_id: str = Field(alias="id", min_length=1)
    symbol: str = Field(alias="ticker", min_length=1)
    ex_dividend_date: date
    distribution_type: DividendDistributionType
    cash_amount: float = Field(gt=0)
    currency: str = Field(min_length=3, max_length=3)
    frequency: int = Field(ge=0)
    declaration_date: date | None = None
    record_date: date | None = None
    pay_date: date | None = None
    split_adjusted_cash_amount: float | None = Field(default=None, gt=0)
    historical_adjustment_factor: float | None = Field(default=None, gt=0)


class _CorporateActionsResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    status: str
    results: list[dict[str, Any]] = Field(default_factory=list)
    next_url: str | None = None


class PolygonClient:
    """Read adjusted daily stock aggregates with bounded pagination/retries."""

    def __init__(
        self,
        *,
        api_key: str,
        http_client: httpx.Client,
        bronze_writer: BronzeWriter | None = None,
        max_attempts: int = 3,
        retry_delay_seconds: float = 0.25,
        max_pages: int = 100,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        if not api_key.strip():
            raise ValueError("Polygon API key must not be empty")
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        if max_pages < 1:
            raise ValueError("max_pages must be at least 1")
        self._api_key = api_key
        self._http = http_client
        self._bronze_writer = bronze_writer
        self._max_attempts = max_attempts
        self._retry_delay_seconds = retry_delay_seconds
        self._max_pages = max_pages
        self._sleeper = sleeper

    def daily_bars(
        self,
        *,
        symbol: str,
        start_date: date,
        end_date: date,
    ) -> tuple[EquityBar, ...]:
        """Fetch split-adjusted daily aggregates over an inclusive interval."""
        normalized_symbol = symbol.strip().upper()
        if not normalized_symbol:
            raise ValueError("symbol must not be empty")
        if end_date < start_date:
            raise ValueError("end_date must be on or after start_date")

        url = (
            f"/v2/aggs/ticker/{normalized_symbol}/range/1/day/"
            f"{start_date.isoformat()}/{end_date.isoformat()}"
        )
        params: dict[str, str] | None = {
            "adjusted": "true",
            "sort": "asc",
            "limit": "50000",
        }
        bars: list[EquityBar] = []
        seen_timestamps: set[datetime] = set()

        for _page_number in range(1, self._max_pages + 1):
            response = self._request(url=url, params=params)
            raw = self._decode_json(response)
            if self._bronze_writer is not None:
                self._bronze_writer.write_json(
                    raw,
                    source="polygon",
                    dataset="daily-aggregate-bars",
                    event_date=start_date,
                )
            page = self._validate_page(raw, expected_symbol=normalized_symbol)
            for aggregate in page.results:
                timestamp = datetime.fromtimestamp(aggregate.timestamp_ms / 1000, tz=UTC)
                if timestamp in seen_timestamps:
                    raise ProviderResponseError(
                        f"Polygon returned duplicate aggregate timestamp: {timestamp.isoformat()}"
                    )
                seen_timestamps.add(timestamp)
                bars.append(
                    self._to_equity_bar(
                        aggregate=aggregate,
                        symbol=normalized_symbol,
                        adjusted=page.adjusted,
                        timestamp=timestamp,
                    )
                )

            if page.next_url is None:
                return tuple(sorted(bars, key=lambda bar: bar.timestamp))
            url = self._validated_next_url(page.next_url)
            params = None

        raise ProviderResponseError(f"Polygon pagination exceeded max_pages={self._max_pages}")

    def ticker_details(self, *, symbol: str, asof_date: date) -> TickerDetails:
        """Fetch security metadata explicitly as it was known on ``asof_date``."""
        normalized_symbol = symbol.strip().upper()
        if not normalized_symbol:
            raise ValueError("symbol must not be empty")
        response = self._request(
            url=f"/v3/reference/tickers/{normalized_symbol}",
            params={"date": asof_date.isoformat()},
        )
        raw = self._decode_json(response)
        if self._bronze_writer is not None:
            self._bronze_writer.write_json(
                raw,
                source="polygon",
                dataset="ticker-details",
                event_date=asof_date,
            )
        try:
            envelope = _TickerDetailsResponse.model_validate(raw)
        except ValidationError as error:
            raise ProviderResponseError(
                f"Polygon ticker-details response failed validation: {error}"
            ) from error
        if envelope.status != "OK":
            raise ProviderResponseError(f"Polygon ticker-details status was {envelope.status!r}")
        details = envelope.results
        if details.ticker != normalized_symbol:
            raise ProviderResponseError(
                f"Polygon response ticker {details.ticker!r} did not match {normalized_symbol!r}"
            )
        return TickerDetails(
            symbol=normalized_symbol,
            asof_date=asof_date,
            name=details.name,
            active=details.active,
            locale=details.locale,
            market=details.market,
            primary_exchange=details.primary_exchange,
            security_type=details.security_type,
            market_cap=details.market_cap,
            list_date=details.list_date,
            delisted_date=details.delisted_date,
        )

    def list_tickers(
        self,
        *,
        asof_date: date,
        active: bool = True,
    ) -> tuple[TickerReference, ...]:
        """Enumerate US stock tickers explicitly available on a historical date."""
        url = "/v3/reference/tickers"
        params: dict[str, str] | None = {
            "market": "stocks",
            "date": asof_date.isoformat(),
            "active": str(active).lower(),
            "sort": "ticker",
            "order": "asc",
            "limit": "1000",
        }
        references: list[TickerReference] = []
        symbols: set[str] = set()
        for _page_number in range(1, self._max_pages + 1):
            response = self._request(url=url, params=params)
            raw = self._decode_json(response)
            if self._bronze_writer is not None:
                self._bronze_writer.write_json(
                    raw,
                    source="polygon",
                    dataset="ticker-reference",
                    event_date=asof_date,
                )
            try:
                page = _TickersResponse.model_validate(raw)
            except ValidationError as error:
                raise ProviderResponseError(
                    f"Polygon tickers response failed validation: {error}"
                ) from error
            if page.status != "OK":
                raise ProviderResponseError(f"Polygon tickers response status was {page.status!r}")
            for item in page.results:
                symbol = item.ticker.strip().upper()
                if symbol in symbols:
                    raise ProviderResponseError(
                        f"Polygon returned duplicate ticker reference: {symbol}"
                    )
                symbols.add(symbol)
                references.append(
                    TickerReference(
                        symbol=symbol,
                        asof_date=asof_date,
                        name=item.name,
                        active=item.active,
                        locale=item.locale,
                        market=item.market,
                        primary_exchange=item.primary_exchange,
                        security_type=item.security_type,
                        delisted_date=item.delisted_date,
                    )
                )
            if page.next_url is None:
                return tuple(sorted(references, key=lambda item: item.symbol))
            url = self._validated_next_url(page.next_url)
            params = None
        raise ProviderResponseError(
            f"Polygon ticker pagination exceeded max_pages={self._max_pages}"
        )

    def stock_splits(
        self,
        *,
        start_date: date,
        end_date: date,
    ) -> tuple[StockSplit, ...]:
        """Fetch all split events by execution date over an inclusive interval."""
        if end_date < start_date:
            raise ValueError("end_date must be on or after start_date")
        pages = self._corporate_action_pages(
            url="/stocks/v1/splits",
            params={
                "execution_date.gte": start_date.isoformat(),
                "execution_date.lte": end_date.isoformat(),
                "limit": "5000",
                "sort": "execution_date.asc",
            },
            dataset="stock-splits",
            event_date=start_date,
        )
        results: list[StockSplit] = []
        for page in pages:
            for raw_item in page.results:
                try:
                    item = _SplitPayload.model_validate(raw_item)
                except ValidationError as error:
                    raise ProviderResponseError(
                        f"Polygon stock-splits response failed validation: {error}"
                    ) from error
                results.append(StockSplit.model_validate(item.model_dump()))
        self._validate_corporate_actions(
            results,
            start_date=start_date,
            end_date=end_date,
            date_field="execution_date",
        )
        return tuple(sorted(results, key=lambda item: (item.execution_date, item.symbol)))

    def cash_dividends(
        self,
        *,
        start_date: date,
        end_date: date,
    ) -> tuple[CashDividend, ...]:
        """Fetch all cash dividends by ex-date over an inclusive interval."""
        if end_date < start_date:
            raise ValueError("end_date must be on or after start_date")
        pages = self._corporate_action_pages(
            url="/stocks/v1/dividends",
            params={
                "ex_dividend_date.gte": start_date.isoformat(),
                "ex_dividend_date.lte": end_date.isoformat(),
                "limit": "5000",
                "sort": "ex_dividend_date.asc",
            },
            dataset="cash-dividends",
            event_date=start_date,
        )
        results: list[CashDividend] = []
        for page in pages:
            for raw_item in page.results:
                try:
                    item = _DividendPayload.model_validate(raw_item)
                except ValidationError as error:
                    raise ProviderResponseError(
                        f"Polygon cash-dividends response failed validation: {error}"
                    ) from error
                results.append(CashDividend.model_validate(item.model_dump()))
        self._validate_corporate_actions(
            results,
            start_date=start_date,
            end_date=end_date,
            date_field="ex_dividend_date",
        )
        return tuple(sorted(results, key=lambda item: (item.ex_dividend_date, item.symbol)))

    def _corporate_action_pages(
        self,
        *,
        url: str,
        params: dict[str, str],
        dataset: str,
        event_date: date,
    ) -> list[_CorporateActionsResponse]:
        pages: list[_CorporateActionsResponse] = []
        request_params: dict[str, str] | None = params
        for _page_number in range(1, self._max_pages + 1):
            response = self._request(url=url, params=request_params)
            raw = self._decode_json(response)
            if self._bronze_writer is not None:
                self._bronze_writer.write_json(
                    raw,
                    source="polygon",
                    dataset=dataset,
                    event_date=event_date,
                )
            try:
                page = _CorporateActionsResponse.model_validate(raw)
            except ValidationError as error:
                raise ProviderResponseError(
                    f"Polygon {dataset} response failed validation: {error}"
                ) from error
            if page.status != "OK":
                raise ProviderResponseError(f"Polygon {dataset} status was {page.status!r}")
            pages.append(page)
            if page.next_url is None:
                return pages
            url = self._validated_next_url(page.next_url)
            request_params = None
        raise ProviderResponseError(
            f"Polygon {dataset} pagination exceeded max_pages={self._max_pages}"
        )

    @staticmethod
    def _validate_corporate_actions(
        events: list[StockSplit] | list[CashDividend],
        *,
        start_date: date,
        end_date: date,
        date_field: str,
    ) -> None:
        if end_date < start_date:
            raise ValueError("end_date must be on or after start_date")
        identifiers: set[str] = set()
        for event in events:
            event_date = getattr(event, date_field)
            if not start_date <= event_date <= end_date:
                raise ProviderResponseError("Polygon returned a corporate action out of range")
            if event.event_id in identifiers:
                raise ProviderResponseError(
                    f"Polygon returned duplicate corporate action id: {event.event_id}"
                )
            identifiers.add(event.event_id)

    @staticmethod
    def _to_equity_bar(
        *,
        aggregate: _Aggregate,
        symbol: str,
        adjusted: bool,
        timestamp: datetime,
    ) -> EquityBar:
        try:
            return EquityBar(
                symbol=symbol,
                timestamp=timestamp,
                open=aggregate.open,
                high=aggregate.high,
                low=aggregate.low,
                close=aggregate.close,
                volume=aggregate.volume,
                vwap=aggregate.vwap,
                transactions=aggregate.transactions,
                adjusted=adjusted,
            )
        except ValidationError as error:
            raise ProviderResponseError(f"Polygon aggregate failed validation: {error}") from error

    def _request(
        self,
        *,
        url: str,
        params: dict[str, str] | None,
    ) -> httpx.Response:
        headers = {"Authorization": f"Bearer {self._api_key}"}
        last_error: httpx.RequestError | None = None
        for attempt in range(1, self._max_attempts + 1):
            try:
                response = self._http.get(url, params=params, headers=headers)
            except httpx.RequestError as error:
                last_error = error
            else:
                if response.status_code not in _RETRYABLE_STATUS_CODES:
                    try:
                        response.raise_for_status()
                    except httpx.HTTPStatusError as error:
                        raise ProviderRequestError(
                            f"Polygon request failed with HTTP {response.status_code}"
                        ) from error
                    return response
                last_error = None
            if attempt < self._max_attempts:
                self._sleeper(self._retry_delay_seconds * attempt)

        detail = f": {last_error}" if last_error is not None else " after retryable HTTP responses"
        raise ProviderRequestError(
            f"Polygon request failed after {self._max_attempts} attempts{detail}"
        )

    def _validated_next_url(self, next_url: str) -> str:
        parsed = urlparse(next_url)
        base_host = self._http.base_url.host
        if parsed.scheme != "https" or parsed.hostname != base_host:
            raise ProviderResponseError("Polygon next_url must use HTTPS and the API host")
        return next_url

    @staticmethod
    def _validate_page(raw: Any, *, expected_symbol: str) -> _AggregatesResponse:
        try:
            page = _AggregatesResponse.model_validate(raw)
        except ValidationError as error:
            raise ProviderResponseError(
                f"Polygon aggregates response failed validation: {error}"
            ) from error
        if page.status != "OK":
            raise ProviderResponseError(f"Polygon response status was {page.status!r}")
        if page.ticker != expected_symbol:
            raise ProviderResponseError(
                f"Polygon response ticker {page.ticker!r} did not match {expected_symbol!r}"
            )
        if not page.adjusted:
            raise ProviderResponseError("Polygon response was not split-adjusted")
        return page

    @staticmethod
    def _decode_json(response: httpx.Response) -> Any:
        try:
            return response.json()
        except ValueError as error:
            raise ProviderResponseError("Polygon returned invalid JSON") from error
