"""Typed external data-provider clients."""

from quant_earning_edge.data.clients.alpaca import AlpacaCalendarClient, MarketSession
from quant_earning_edge.data.clients.errors import ProviderRequestError, ProviderResponseError
from quant_earning_edge.data.clients.finnhub import (
    EarningsEvent,
    EarningsTiming,
    FinnhubClient,
)
from quant_earning_edge.data.clients.polygon import (
    EquityBar,
    PolygonClient,
    TickerDetails,
    TickerReference,
)

__all__ = [
    "AlpacaCalendarClient",
    "EarningsEvent",
    "EarningsTiming",
    "EquityBar",
    "FinnhubClient",
    "MarketSession",
    "PolygonClient",
    "ProviderRequestError",
    "ProviderResponseError",
    "TickerDetails",
    "TickerReference",
]
