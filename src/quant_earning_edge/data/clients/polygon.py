"""Polygon/Massive daily aggregate-bars client."""

from __future__ import annotations

import hashlib
import json
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

    from quant_earning_edge.data.bronze import BronzeArtifact, BronzeWriter

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


class MinuteBar(BaseModel):
    """Validated split-adjusted one-minute stock aggregate."""

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
    def validate_bounds(self) -> MinuteBar:
        """Require a timezone-aware timestamp and valid OHLC envelope."""
        if self.timestamp.tzinfo is None or self.timestamp.utcoffset() is None:
            raise ValueError("timestamp must be timezone-aware")
        if self.low > self.high or not self.low <= self.open <= self.high:
            raise ValueError("invalid minute-bar open/high/low")
        if not self.low <= self.close <= self.high:
            raise ValueError("invalid minute-bar close/high/low")
        if not self.adjusted:
            raise ValueError("minute bar must be split-adjusted")
        return self


class StockQuote(BaseModel):
    """Normalized historical stock NBBO update using the SIP receipt timestamp."""

    model_config = ConfigDict(frozen=True)

    symbol: str = Field(min_length=1)
    timestamp: datetime
    sequence_number: int = Field(ge=0)
    bid_price: float = Field(ge=0)
    ask_price: float = Field(ge=0)
    bid_size: float = Field(ge=0)
    ask_size: float = Field(ge=0)
    bid_exchange: int | None = Field(default=None, ge=0)
    ask_exchange: int | None = Field(default=None, ge=0)
    participant_timestamp: datetime | None = None
    conditions: tuple[int, ...] = ()

    @model_validator(mode="after")
    def validate_quote(self) -> StockQuote:
        """Accept one-sided provider updates but reject crossed two-sided quotes."""
        if self.timestamp.tzinfo is None or self.timestamp.utcoffset() is None:
            raise ValueError("quote timestamp must be timezone-aware")
        if self.participant_timestamp is not None and (
            self.participant_timestamp.tzinfo is None
            or self.participant_timestamp.utcoffset() is None
        ):
            raise ValueError("participant timestamp must be timezone-aware")
        if self.bid_price > 0 and self.ask_price > 0 and self.ask_price < self.bid_price:
            raise ValueError("two-sided NBBO must not be crossed")
        return self

    @property
    def is_replayable(self) -> bool:
        """Whether both sides carry positive price and displayed size."""
        return self.bid_price > 0 and self.ask_price > 0 and self.bid_size > 0 and self.ask_size > 0


class StockTrade(BaseModel):
    """Normalized historical stock trade using the SIP receipt timestamp."""

    model_config = ConfigDict(frozen=True)

    symbol: str = Field(min_length=1)
    timestamp: datetime
    sequence_number: int = Field(ge=0)
    price: float = Field(gt=0)
    size: float = Field(gt=0)
    exchange: int = Field(ge=0)
    trade_id: str = Field(min_length=1)
    participant_timestamp: datetime | None = None
    conditions: tuple[int, ...] = ()
    correction: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_timestamps(self) -> StockTrade:
        if self.timestamp.tzinfo is None or self.timestamp.utcoffset() is None:
            raise ValueError("trade timestamp must be timezone-aware")
        if self.participant_timestamp is not None and (
            self.participant_timestamp.tzinfo is None
            or self.participant_timestamp.utcoffset() is None
        ):
            raise ValueError("participant timestamp must be timezone-aware")
        return self


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
    sic_code: str | None = None
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


class TickerSnapshot(BaseModel):
    """Provider-timestamped decision NBBO and last trade from Polygon snapshot."""

    model_config = ConfigDict(frozen=True)

    symbol: str = Field(min_length=1)
    captured_at: datetime
    observed_at: datetime
    bid_price: float = Field(gt=0)
    ask_price: float = Field(gt=0)
    bid_size: int = Field(gt=0)
    ask_size: int = Field(gt=0)
    last_trade_price: float = Field(gt=0)
    last_trade_at: datetime
    payload_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    request_id: str | None = None

    @model_validator(mode="after")
    def validate_snapshot(self) -> TickerSnapshot:
        if any(
            item.tzinfo is None or item.utcoffset() is None
            for item in (self.captured_at, self.observed_at, self.last_trade_at)
        ):
            raise ValueError("ticker snapshot timestamps must be timezone-aware")
        if self.ask_price < self.bid_price:
            raise ValueError("ticker snapshot NBBO must not be crossed")
        if self.observed_at > self.captured_at or self.last_trade_at > self.captured_at:
            raise ValueError("ticker snapshot observation cannot follow capture")
        object.__setattr__(self, "symbol", self.symbol.strip().upper())
        return self


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
    sic_code: str | None = None
    list_date: date | None = None
    delisted_date: date | None = Field(default=None, alias="delisted_utc")


class _TickerDetailsResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    status: str
    results: _TickerDetailsPayload


class _SnapshotQuote(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    ask_price: float = Field(alias="P", gt=0)
    ask_size: int = Field(alias="S", gt=0)
    bid_price: float = Field(alias="p", gt=0)
    bid_size: int = Field(alias="s", gt=0)
    timestamp_ns: int = Field(alias="t", gt=0)


class _SnapshotTrade(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    price: float = Field(alias="p", gt=0)
    timestamp_ns: int = Field(alias="t", gt=0)


class _SnapshotTickerPayload(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    ticker: str
    quote: _SnapshotQuote = Field(alias="lastQuote")
    trade: _SnapshotTrade = Field(alias="lastTrade")


class _SnapshotResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    status: str
    request_id: str | None = None
    ticker: _SnapshotTickerPayload


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


class _QuotePayload(BaseModel):
    model_config = ConfigDict(extra="ignore")

    sip_timestamp: int = Field(ge=0)
    participant_timestamp: int | None = Field(default=None, ge=0)
    sequence_number: int = Field(ge=0)
    bid_price: float = Field(default=0, ge=0)
    ask_price: float = Field(default=0, ge=0)
    bid_size: float = Field(default=0, ge=0)
    ask_size: float = Field(default=0, ge=0)
    bid_exchange: int | None = Field(default=None, ge=0)
    ask_exchange: int | None = Field(default=None, ge=0)
    conditions: tuple[int, ...] = ()


class _TradePayload(BaseModel):
    model_config = ConfigDict(extra="ignore")

    sip_timestamp: int = Field(ge=0)
    participant_timestamp: int | None = Field(default=None, ge=0)
    sequence_number: int = Field(ge=0)
    price: float = Field(gt=0)
    size: float = Field(gt=0)
    exchange: int = Field(ge=0)
    trade_id: str = Field(alias="id", min_length=1)
    conditions: tuple[int, ...] = ()
    correction: int | None = Field(default=None, ge=0)


class _QuotesResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    status: str
    results: tuple[_QuotePayload, ...] = ()
    next_url: str | None = None


class _TradesResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    status: str
    results: tuple[_TradePayload, ...] = ()
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
        self._decision_snapshot_artifacts: list[BronzeArtifact] = []
        self._universe_observation_artifacts: list[BronzeArtifact] = []
        self._corporate_action_observation_artifacts: list[BronzeArtifact] = []

    @property
    def decision_snapshot_artifacts(self) -> tuple[BronzeArtifact, ...]:
        """Return raw decision snapshots captured by this client instance."""
        return tuple(self._decision_snapshot_artifacts)

    @property
    def universe_observation_artifacts(self) -> tuple[BronzeArtifact, ...]:
        """Return raw reference, details, and daily-bar universe responses."""
        return tuple(self._universe_observation_artifacts)

    @property
    def corporate_action_observation_artifacts(self) -> tuple[BronzeArtifact, ...]:
        """Return raw split and dividend responses retained by this client."""
        return tuple(self._corporate_action_observation_artifacts)

    @classmethod
    def stock_splits_from_payloads(
        cls,
        raws: tuple[Any, ...],
        *,
        start_date: date,
        end_date: date,
    ) -> tuple[StockSplit, ...]:
        """Reconstruct retained split pages without provider requests."""
        results: list[StockSplit] = []
        for raw in raws:
            page = cls._corporate_action_page_from_payload(raw, dataset="stock-splits")
            for raw_item in page.results:
                try:
                    item = _SplitPayload.model_validate(raw_item)
                except ValidationError as error:
                    raise ProviderResponseError(
                        f"Polygon stock-splits response failed validation: {error}"
                    ) from error
                results.append(StockSplit.model_validate(item.model_dump()))
        cls._validate_corporate_actions(
            results,
            start_date=start_date,
            end_date=end_date,
            date_field="execution_date",
        )
        return tuple(sorted(results, key=lambda item: (item.execution_date, item.symbol)))

    @classmethod
    def cash_dividends_from_payloads(
        cls,
        raws: tuple[Any, ...],
        *,
        start_date: date,
        end_date: date,
    ) -> tuple[CashDividend, ...]:
        """Reconstruct retained dividend pages without provider requests."""
        results: list[CashDividend] = []
        for raw in raws:
            page = cls._corporate_action_page_from_payload(raw, dataset="cash-dividends")
            for raw_item in page.results:
                try:
                    item = _DividendPayload.model_validate(raw_item)
                except ValidationError as error:
                    raise ProviderResponseError(
                        f"Polygon cash-dividends response failed validation: {error}"
                    ) from error
                results.append(CashDividend.model_validate(item.model_dump()))
        cls._validate_corporate_actions(
            results,
            start_date=start_date,
            end_date=end_date,
            date_field="ex_dividend_date",
        )
        return tuple(sorted(results, key=lambda item: (item.ex_dividend_date, item.symbol)))

    @staticmethod
    def _corporate_action_page_from_payload(
        raw: Any,
        *,
        dataset: str,
    ) -> _CorporateActionsResponse:
        try:
            page = _CorporateActionsResponse.model_validate(raw)
        except ValidationError as error:
            raise ProviderResponseError(
                f"Polygon {dataset} response failed validation: {error}"
            ) from error
        if page.status != "OK":
            raise ProviderResponseError(f"Polygon {dataset} status was {page.status!r}")
        return page

    @classmethod
    def daily_bars_from_payload(
        cls,
        raw: Any,
        *,
        symbol: str,
    ) -> tuple[EquityBar, ...]:
        """Reconstruct one retained aggregate page without a provider request."""
        normalized = symbol.strip().upper()
        page = cls._validate_page(raw, expected_symbol=normalized)
        bars = tuple(
            cls._to_equity_bar(
                aggregate=aggregate,
                symbol=normalized,
                adjusted=page.adjusted,
                timestamp=datetime.fromtimestamp(aggregate.timestamp_ms / 1000, tz=UTC),
            )
            for aggregate in page.results
        )
        timestamps = tuple(item.timestamp for item in bars)
        if timestamps != tuple(sorted(set(timestamps))):
            raise ProviderResponseError("Polygon aggregate payload timestamps are not unique")
        return bars

    @staticmethod
    def ticker_details_from_payload(
        raw: Any,
        *,
        symbol: str,
        asof_date: date,
    ) -> TickerDetails:
        """Reconstruct retained point-in-time ticker details."""
        normalized = symbol.strip().upper()
        try:
            envelope = _TickerDetailsResponse.model_validate(raw)
        except ValidationError as error:
            raise ProviderResponseError(
                f"Polygon ticker-details response failed validation: {error}"
            ) from error
        if envelope.status != "OK":
            raise ProviderResponseError(f"Polygon ticker-details status was {envelope.status!r}")
        if envelope.results.ticker != normalized:
            raise ProviderResponseError(
                f"Polygon response ticker {envelope.results.ticker!r} did not match {normalized!r}"
            )
        details = envelope.results
        return TickerDetails(
            symbol=normalized,
            asof_date=asof_date,
            name=details.name,
            active=details.active,
            locale=details.locale,
            market=details.market,
            primary_exchange=details.primary_exchange,
            security_type=details.security_type,
            market_cap=details.market_cap,
            sic_code=details.sic_code,
            list_date=details.list_date,
            delisted_date=details.delisted_date,
        )

    @staticmethod
    def ticker_references_from_payload(
        raw: Any,
        *,
        asof_date: date,
    ) -> tuple[TickerReference, ...]:
        """Reconstruct one retained ticker-reference page."""
        try:
            page = _TickersResponse.model_validate(raw)
        except ValidationError as error:
            raise ProviderResponseError(
                f"Polygon tickers response failed validation: {error}"
            ) from error
        if page.status != "OK":
            raise ProviderResponseError(f"Polygon tickers response status was {page.status!r}")
        references = tuple(
            TickerReference(
                symbol=item.ticker.strip().upper(),
                asof_date=asof_date,
                name=item.name,
                active=item.active,
                locale=item.locale,
                market=item.market,
                primary_exchange=item.primary_exchange,
                security_type=item.security_type,
                delisted_date=item.delisted_date,
            )
            for item in page.results
        )
        symbols = tuple(item.symbol for item in references)
        if len(symbols) != len(set(symbols)):
            raise ProviderResponseError("Polygon retained ticker references contain duplicates")
        return tuple(sorted(references, key=lambda item: item.symbol))

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
                self._universe_observation_artifacts.append(
                    self._bronze_writer.write_json(
                        raw,
                        source="polygon",
                        dataset="daily-aggregate-bars",
                        event_date=start_date,
                    )
                )
            page = self._validate_page(raw, expected_symbol=normalized_symbol)
            for bar in self.daily_bars_from_payload(raw, symbol=normalized_symbol):
                if bar.timestamp in seen_timestamps:
                    raise ProviderResponseError(
                        "Polygon returned duplicate aggregate timestamp: "
                        f"{bar.timestamp.isoformat()}"
                    )
                seen_timestamps.add(bar.timestamp)
                bars.append(bar)

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
            self._universe_observation_artifacts.append(
                self._bronze_writer.write_json(
                    raw,
                    source="polygon",
                    dataset="ticker-details",
                    event_date=asof_date,
                )
            )
        return self.ticker_details_from_payload(
            raw,
            symbol=normalized_symbol,
            asof_date=asof_date,
        )

    def ticker_snapshot(
        self,
        *,
        symbol: str,
        captured_at: datetime | None = None,
    ) -> TickerSnapshot:
        """Capture the current two-sided NBBO and last trade with provider timestamps."""
        normalized_symbol = symbol.strip().upper()
        if not normalized_symbol:
            raise ValueError("symbol must not be empty")
        captured = (captured_at or datetime.now(UTC)).astimezone(UTC)
        response = self._request(
            url=f"/v2/snapshot/locale/us/markets/stocks/tickers/{normalized_symbol}",
            params=None,
        )
        raw = self._decode_json(response)
        if self._bronze_writer is not None:
            self._decision_snapshot_artifacts.append(
                self._bronze_writer.write_json(
                    raw,
                    source="polygon",
                    dataset="decision-snapshot",
                    event_date=captured.date(),
                )
            )
        return self.ticker_snapshot_from_payload(
            raw,
            symbol=normalized_symbol,
            captured_at=captured,
        )

    @staticmethod
    def ticker_snapshot_from_payload(
        raw: Any,
        *,
        symbol: str,
        captured_at: datetime,
    ) -> TickerSnapshot:
        """Rebuild one decision snapshot from its exact provider payload."""
        normalized_symbol = symbol.strip().upper()
        try:
            envelope = _SnapshotResponse.model_validate(raw)
        except ValidationError as error:
            raise ProviderResponseError(
                f"Polygon ticker snapshot failed validation: {error}"
            ) from error
        if envelope.status.upper() != "OK":
            raise ProviderResponseError(f"Polygon ticker snapshot status was {envelope.status!r}")
        if envelope.ticker.ticker.strip().upper() != normalized_symbol:
            raise ProviderResponseError("Polygon ticker snapshot identity did not match")
        quote = envelope.ticker.quote
        trade = envelope.ticker.trade
        payload = json.dumps(
            raw,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        try:
            return TickerSnapshot(
                symbol=normalized_symbol,
                captured_at=captured_at,
                observed_at=PolygonClient._nanoseconds_to_datetime(quote.timestamp_ns),
                bid_price=quote.bid_price,
                ask_price=quote.ask_price,
                bid_size=quote.bid_size,
                ask_size=quote.ask_size,
                last_trade_price=trade.price,
                last_trade_at=PolygonClient._nanoseconds_to_datetime(trade.timestamp_ns),
                payload_sha256=hashlib.sha256(payload).hexdigest(),
                request_id=envelope.request_id,
            )
        except ValidationError as error:
            raise ProviderResponseError(
                f"Polygon decision snapshot failed validation: {error}"
            ) from error

    def minute_bars(
        self,
        *,
        symbol: str,
        start_at: datetime,
        end_at: datetime,
    ) -> tuple[MinuteBar, ...]:
        """Fetch split-adjusted minute aggregates over an aware timestamp interval."""
        normalized_symbol = symbol.strip().upper()
        if not normalized_symbol:
            raise ValueError("symbol must not be empty")
        if any(value.tzinfo is None or value.utcoffset() is None for value in (start_at, end_at)):
            raise ValueError("minute-bar interval timestamps must be timezone-aware")
        start_utc = start_at.astimezone(UTC)
        end_utc = end_at.astimezone(UTC)
        if end_utc <= start_utc:
            raise ValueError("end_at must be after start_at")
        start_ms = int(start_utc.timestamp() * 1000)
        end_ms = int(end_utc.timestamp() * 1000)
        url = f"/v2/aggs/ticker/{normalized_symbol}/range/1/minute/{start_ms}/{end_ms}"
        params: dict[str, str] | None = {
            "adjusted": "true",
            "sort": "asc",
            "limit": "50000",
        }
        bars: list[MinuteBar] = []
        timestamps: set[datetime] = set()
        for _page_number in range(1, self._max_pages + 1):
            response = self._request(url=url, params=params)
            raw = self._decode_json(response)
            if self._bronze_writer is not None:
                self._bronze_writer.write_json(
                    raw,
                    source="polygon",
                    dataset="minute-aggregate-bars",
                    event_date=start_utc.astimezone(_MARKET_TIMEZONE).date(),
                )
            page = self._validate_page(raw, expected_symbol=normalized_symbol)
            for aggregate in page.results:
                timestamp = datetime.fromtimestamp(aggregate.timestamp_ms / 1000, tz=UTC)
                if timestamp in timestamps:
                    raise ProviderResponseError(
                        f"Polygon returned duplicate minute timestamp: {timestamp.isoformat()}"
                    )
                if not start_utc <= timestamp <= end_utc:
                    raise ProviderResponseError("Polygon returned a minute bar out of range")
                timestamps.add(timestamp)
                try:
                    bars.append(
                        MinuteBar(
                            symbol=normalized_symbol,
                            timestamp=timestamp,
                            open=aggregate.open,
                            high=aggregate.high,
                            low=aggregate.low,
                            close=aggregate.close,
                            volume=aggregate.volume,
                            vwap=aggregate.vwap,
                            transactions=aggregate.transactions,
                            adjusted=page.adjusted,
                        )
                    )
                except ValidationError as error:
                    raise ProviderResponseError(
                        f"Polygon minute aggregate failed validation: {error}"
                    ) from error
            if page.next_url is None:
                return tuple(sorted(bars, key=lambda item: item.timestamp))
            url = self._validated_next_url(page.next_url)
            params = None
        raise ProviderResponseError(
            f"Polygon minute pagination exceeded max_pages={self._max_pages}"
        )

    def stock_quotes(
        self,
        *,
        symbol: str,
        start_at: datetime,
        end_at: datetime,
    ) -> tuple[StockQuote, ...]:
        """Fetch historical NBBO updates over an inclusive SIP-time interval."""
        normalized_symbol, start_utc, end_utc = self._market_event_interval(
            symbol=symbol,
            start_at=start_at,
            end_at=end_at,
        )
        url = f"/v3/quotes/{normalized_symbol}"
        params: dict[str, str] | None = self._market_event_params(start_utc, end_utc)
        quotes: list[StockQuote] = []
        identities: set[tuple[datetime, int]] = set()
        for _page_number in range(1, self._max_pages + 1):
            response = self._request(url=url, params=params)
            raw = self._decode_json(response)
            self._capture_market_events(
                raw,
                dataset="stock-nbbo-quotes",
                event_date=start_utc.astimezone(_MARKET_TIMEZONE).date(),
            )
            try:
                page = _QuotesResponse.model_validate(raw)
            except ValidationError as error:
                raise ProviderResponseError(
                    f"Polygon stock-quotes response failed validation: {error}"
                ) from error
            if page.status != "OK":
                raise ProviderResponseError(f"Polygon stock-quotes status was {page.status!r}")
            for item in page.results:
                timestamp = self._nanoseconds_to_datetime(item.sip_timestamp)
                identity = (timestamp, item.sequence_number)
                self._validate_market_event_identity(
                    identity,
                    identities=identities,
                    start_at=start_utc,
                    end_at=end_utc,
                    dataset="stock quote",
                )
                try:
                    quotes.append(
                        StockQuote(
                            symbol=normalized_symbol,
                            timestamp=timestamp,
                            sequence_number=item.sequence_number,
                            bid_price=item.bid_price,
                            ask_price=item.ask_price,
                            bid_size=item.bid_size,
                            ask_size=item.ask_size,
                            bid_exchange=item.bid_exchange,
                            ask_exchange=item.ask_exchange,
                            participant_timestamp=(
                                self._nanoseconds_to_datetime(item.participant_timestamp)
                                if item.participant_timestamp is not None
                                else None
                            ),
                            conditions=item.conditions,
                        )
                    )
                except ValidationError as error:
                    raise ProviderResponseError(
                        f"Polygon stock quote failed validation: {error}"
                    ) from error
            if page.next_url is None:
                return tuple(
                    sorted(quotes, key=lambda item: (item.timestamp, item.sequence_number))
                )
            url = self._validated_next_url(page.next_url)
            params = None
        raise ProviderResponseError(
            f"Polygon stock-quotes pagination exceeded max_pages={self._max_pages}"
        )

    def stock_trades(
        self,
        *,
        symbol: str,
        start_at: datetime,
        end_at: datetime,
    ) -> tuple[StockTrade, ...]:
        """Fetch historical trades over an inclusive SIP-time interval."""
        normalized_symbol, start_utc, end_utc = self._market_event_interval(
            symbol=symbol,
            start_at=start_at,
            end_at=end_at,
        )
        url = f"/v3/trades/{normalized_symbol}"
        params: dict[str, str] | None = self._market_event_params(start_utc, end_utc)
        trades: list[StockTrade] = []
        identities: set[tuple[datetime, int]] = set()
        for _page_number in range(1, self._max_pages + 1):
            response = self._request(url=url, params=params)
            raw = self._decode_json(response)
            self._capture_market_events(
                raw,
                dataset="stock-trades",
                event_date=start_utc.astimezone(_MARKET_TIMEZONE).date(),
            )
            try:
                page = _TradesResponse.model_validate(raw)
            except ValidationError as error:
                raise ProviderResponseError(
                    f"Polygon stock-trades response failed validation: {error}"
                ) from error
            if page.status != "OK":
                raise ProviderResponseError(f"Polygon stock-trades status was {page.status!r}")
            for item in page.results:
                timestamp = self._nanoseconds_to_datetime(item.sip_timestamp)
                identity = (timestamp, item.sequence_number)
                self._validate_market_event_identity(
                    identity,
                    identities=identities,
                    start_at=start_utc,
                    end_at=end_utc,
                    dataset="stock trade",
                )
                try:
                    trades.append(
                        StockTrade(
                            symbol=normalized_symbol,
                            timestamp=timestamp,
                            sequence_number=item.sequence_number,
                            price=item.price,
                            size=item.size,
                            exchange=item.exchange,
                            trade_id=item.trade_id,
                            participant_timestamp=(
                                self._nanoseconds_to_datetime(item.participant_timestamp)
                                if item.participant_timestamp is not None
                                else None
                            ),
                            conditions=item.conditions,
                            correction=item.correction,
                        )
                    )
                except ValidationError as error:
                    raise ProviderResponseError(
                        f"Polygon stock trade failed validation: {error}"
                    ) from error
            if page.next_url is None:
                return tuple(
                    sorted(trades, key=lambda item: (item.timestamp, item.sequence_number))
                )
            url = self._validated_next_url(page.next_url)
            params = None
        raise ProviderResponseError(
            f"Polygon stock-trades pagination exceeded max_pages={self._max_pages}"
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
                self._universe_observation_artifacts.append(
                    self._bronze_writer.write_json(
                        raw,
                        source="polygon",
                        dataset="ticker-reference",
                        event_date=asof_date,
                    )
                )
            try:
                page = _TickersResponse.model_validate(raw)
            except ValidationError as error:
                raise ProviderResponseError(
                    f"Polygon tickers response failed validation: {error}"
                ) from error
            for reference in self.ticker_references_from_payload(raw, asof_date=asof_date):
                symbol = reference.symbol
                if symbol in symbols:
                    raise ProviderResponseError(
                        f"Polygon returned duplicate ticker reference: {symbol}"
                    )
                symbols.add(symbol)
                references.append(reference)
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
                self._corporate_action_observation_artifacts.append(
                    self._bronze_writer.write_json(
                        raw,
                        source="polygon",
                        dataset=dataset,
                        event_date=event_date,
                    )
                )
            page = self._corporate_action_page_from_payload(raw, dataset=dataset)
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

    def _capture_market_events(self, raw: Any, *, dataset: str, event_date: date) -> None:
        if self._bronze_writer is not None:
            self._bronze_writer.write_json(
                raw,
                source="polygon",
                dataset=dataset,
                event_date=event_date,
            )

    @staticmethod
    def _market_event_interval(
        *,
        symbol: str,
        start_at: datetime,
        end_at: datetime,
    ) -> tuple[str, datetime, datetime]:
        normalized_symbol = symbol.strip().upper()
        if not normalized_symbol:
            raise ValueError("symbol must not be empty")
        if any(value.tzinfo is None or value.utcoffset() is None for value in (start_at, end_at)):
            raise ValueError("market-event interval timestamps must be timezone-aware")
        start_utc = start_at.astimezone(UTC)
        end_utc = end_at.astimezone(UTC)
        if end_utc <= start_utc:
            raise ValueError("end_at must be after start_at")
        return normalized_symbol, start_utc, end_utc

    @classmethod
    def _market_event_params(cls, start_at: datetime, end_at: datetime) -> dict[str, str]:
        return {
            "timestamp.gte": str(cls._datetime_to_nanoseconds(start_at)),
            "timestamp.lte": str(cls._datetime_to_nanoseconds(end_at)),
            "sort": "timestamp",
            "order": "asc",
            "limit": "50000",
        }

    @staticmethod
    def _validate_market_event_identity(
        identity: tuple[datetime, int],
        *,
        identities: set[tuple[datetime, int]],
        start_at: datetime,
        end_at: datetime,
        dataset: str,
    ) -> None:
        timestamp, _sequence = identity
        if not start_at <= timestamp <= end_at:
            raise ProviderResponseError(f"Polygon returned a {dataset} out of range")
        if identity in identities:
            raise ProviderResponseError(f"Polygon returned duplicate {dataset} identity")
        identities.add(identity)

    @staticmethod
    def _datetime_to_nanoseconds(value: datetime) -> int:
        utc = value.astimezone(UTC)
        return int(utc.timestamp()) * 1_000_000_000 + utc.microsecond * 1_000

    @staticmethod
    def _nanoseconds_to_datetime(value: int) -> datetime:
        seconds, nanoseconds = divmod(value, 1_000_000_000)
        return datetime.fromtimestamp(seconds, tz=UTC).replace(microsecond=nanoseconds // 1_000)

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
