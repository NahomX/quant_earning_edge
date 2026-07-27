"""Typed external data-provider clients."""

from quant_earning_edge.data.clients.alpaca import AlpacaCalendarClient, MarketSession
from quant_earning_edge.data.clients.errors import ProviderRequestError, ProviderResponseError
from quant_earning_edge.data.clients.finnhub import (
    EarningsEvent,
    EarningsTiming,
    FinnhubClient,
)
from quant_earning_edge.data.clients.polygon import (
    CashDividend,
    DividendDistributionType,
    EquityBar,
    PolygonClient,
    SplitAdjustmentType,
    StockSplit,
    TickerDetails,
    TickerReference,
)

__all__ = [
    "AlpacaCalendarClient",
    "CashDividend",
    "DividendDistributionType",
    "EarningsEvent",
    "EarningsTiming",
    "EquityBar",
    "FinnhubClient",
    "MarketSession",
    "PolygonClient",
    "ProviderRequestError",
    "ProviderResponseError",
    "SplitAdjustmentType",
    "StockSplit",
    "TickerDetails",
    "TickerReference",
]
