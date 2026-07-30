"""Causal split-vintage adapter for provider-backed universe construction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from quant_earning_edge.data.split_vintage import causally_adjust_equity_bars

if TYPE_CHECKING:
    from datetime import date

    from quant_earning_edge.data.clients import (
        EquityBar,
        PolygonClient,
        StockSplit,
        TickerDetails,
        TickerReference,
    )


@dataclass(frozen=True)
class SplitNormalizedUniverseMarketData:
    """Fetch raw daily bars and normalize only through the universe as-of date."""

    provider: PolygonClient
    splits: tuple[StockSplit, ...]
    basis_date: date

    def list_tickers(
        self,
        *,
        asof_date: date,
        active: bool = True,
    ) -> tuple[TickerReference, ...]:
        if asof_date != self.basis_date:
            raise ValueError("universe ticker date differs from the split basis")
        return self.provider.list_tickers(asof_date=asof_date, active=active)

    def ticker_details(self, *, symbol: str, asof_date: date) -> TickerDetails:
        if asof_date != self.basis_date:
            raise ValueError("universe details date differs from the split basis")
        return self.provider.ticker_details(symbol=symbol, asof_date=asof_date)

    def daily_bars(
        self,
        *,
        symbol: str,
        start_date: date,
        end_date: date,
    ) -> tuple[EquityBar, ...]:
        if end_date != self.basis_date:
            raise ValueError("universe bar end date differs from the split basis")
        raw = self.provider.daily_bars(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            adjusted=False,
        )
        return causally_adjust_equity_bars(
            raw,
            splits=self.splits,
            basis_date=self.basis_date,
        )
