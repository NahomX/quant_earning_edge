"""Typed external data-provider clients."""

from quant_earning_edge.data.clients.finnhub import (
    EarningsEvent,
    EarningsTiming,
    FinnhubClient,
    ProviderRequestError,
    ProviderResponseError,
)

__all__ = [
    "EarningsEvent",
    "EarningsTiming",
    "FinnhubClient",
    "ProviderRequestError",
    "ProviderResponseError",
]
