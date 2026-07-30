"""Vectorbt-backed daily round-trip engine with exact cost reconciliation."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd

from quant_earning_edge.backtest.costs import (
    CostBreakdown,
    CostModel,
    ExecutionCostInput,
)

if TYPE_CHECKING:
    from collections.abc import Sequence
    from datetime import date, datetime

PositionSide = Literal["long", "short"]


@dataclass(frozen=True)
class DailyMark:
    """One causal end-of-session valuation mark."""

    symbol: str
    session_date: date
    close: float

    def __post_init__(self) -> None:
        normalized = self.symbol.strip().upper()
        if not normalized:
            raise ValueError("symbol must not be empty")
        object.__setattr__(self, "symbol", normalized)
        if self.close <= 0 or not math.isfinite(self.close):
            raise ValueError("close must be finite and positive")


@dataclass(frozen=True)
class TradeIntent:
    """One fully specified daily round trip."""

    trade_id: str
    symbol: str
    side: PositionSide
    entry_date: date
    exit_date: date
    shares: int
    entry_price: float
    exit_price: float
    entry_average_daily_volume_shares: float
    exit_average_daily_volume_shares: float
    holding_sessions: int
    triggered_stop_price: float | None = None
    atr5: float | None = None
    entry_at: datetime | None = None
    exit_at: datetime | None = None

    def __post_init__(self) -> None:  # noqa: PLR0912 - validates daily and intraday forms.
        trade_id = self.trade_id.strip()
        symbol = self.symbol.strip().upper()
        if not trade_id:
            raise ValueError("trade_id must not be empty")
        if not symbol:
            raise ValueError("symbol must not be empty")
        object.__setattr__(self, "trade_id", trade_id)
        object.__setattr__(self, "symbol", symbol)
        if (self.entry_at is None) != (self.exit_at is None):
            raise ValueError("entry_at and exit_at must be supplied together")
        if self.entry_at is None:
            if self.entry_date >= self.exit_date:
                raise ValueError("daily trade entry_date must precede exit_date")
        else:
            if (
                self.entry_at.tzinfo is None
                or self.entry_at.utcoffset() is None
                or self.exit_at is None
                or self.exit_at.tzinfo is None
                or self.exit_at.utcoffset() is None
            ):
                raise ValueError("execution timestamps must be timezone-aware")
            if self.entry_at >= self.exit_at:
                raise ValueError("entry_at must precede exit_at")
            if self.entry_at.date() != self.entry_date or self.exit_at.date() != self.exit_date:
                raise ValueError("execution timestamp dates must match trade dates")
        if self.shares < 1:
            raise ValueError("shares must be positive")
        numeric = (
            self.entry_price,
            self.exit_price,
            self.entry_average_daily_volume_shares,
            self.exit_average_daily_volume_shares,
        )
        if any(value <= 0 or not math.isfinite(value) for value in numeric):
            raise ValueError("prices and average daily volumes must be finite and positive")
        expected_minimum_holding = 0 if self.entry_date == self.exit_date else 1
        if self.holding_sessions < expected_minimum_holding:
            raise ValueError("holding_sessions is inconsistent with trade dates")
        if self.side == "short" and self.triggered_stop_price is not None:
            raise ValueError("short buy-stop slippage is not defined by the architecture")


@dataclass(frozen=True)
class TradeLedger:
    """Gross and component-level net result for one round trip."""

    intent: TradeIntent
    entry_cost: CostBreakdown
    exit_cost: CostBreakdown

    @property
    def gross_pnl(self) -> float:
        direction = 1.0 if self.intent.side == "long" else -1.0
        return direction * (self.intent.exit_price - self.intent.entry_price) * self.intent.shares

    @property
    def total_cost(self) -> float:
        return self.entry_cost.total + self.exit_cost.total

    @property
    def net_pnl(self) -> float:
        return self.gross_pnl - self.total_cost

    @property
    def entry_notional(self) -> float:
        return self.intent.entry_price * self.intent.shares

    @property
    def net_return(self) -> float:
        return self.net_pnl / self.entry_notional


@dataclass(frozen=True)
class DailyLedger:
    """Daily portfolio equity and accounting attribution."""

    session_date: date
    gross_pnl: float
    commission: float
    half_spread: float
    market_impact: float
    borrow: float
    stop_slippage: float
    net_pnl: float
    gross_equity: float
    net_equity: float
    gross_exposure: float

    @property
    def total_cost(self) -> float:
        return (
            self.commission
            + self.half_spread
            + self.market_impact
            + self.borrow
            + self.stop_slippage
        )


@dataclass(frozen=True)
class BacktestResult:
    """Deterministic vectorbt result reduced to stable typed evidence."""

    engine: str
    input_sha256: str
    initial_cash: float
    trades: tuple[TradeLedger, ...]
    daily: tuple[DailyLedger, ...]

    @property
    def final_gross_equity(self) -> float:
        return self.daily[-1].gross_equity

    @property
    def final_net_equity(self) -> float:
        return self.daily[-1].net_equity


class VectorbtBacktestEngine:
    """Run independent round trips in one cash-sharing vectorbt portfolio."""

    def __init__(self, cost_model: CostModel | None = None) -> None:
        self._cost_model = cost_model or CostModel()

    def run(  # noqa: PLR0915 - atomic vectorbt setup and reconciliation boundary.
        self,
        *,
        trades: Sequence[TradeIntent],
        marks: Sequence[DailyMark],
        sessions: Sequence[date],
        initial_cash: float,
    ) -> BacktestResult:
        """Value daily positions and prove exact gross-minus-cost reconciliation."""
        if initial_cash <= 0 or not math.isfinite(initial_cash):
            raise ValueError("initial_cash must be finite and positive")
        session_dates = tuple(sessions)
        if not session_dates or tuple(sorted(set(session_dates))) != session_dates:
            raise ValueError("sessions must be non-empty, unique, and increasing")
        if not trades:
            raise ValueError("at least one trade is required")
        trade_ids = tuple(item.trade_id for item in trades)
        if len(set(trade_ids)) != len(trade_ids):
            raise ValueError("trade_id values must be unique")
        session_set = set(session_dates)
        if any(
            item.entry_date not in session_set or item.exit_date not in session_set
            for item in trades
        ):
            raise ValueError("every trade entry and exit must be in sessions")
        if any(item.entry_at is not None or item.exit_at is not None for item in trades):
            raise ValueError("timestamped trades require VectorbtIntradayEngine")

        marks_by_key = self._marks_by_key(marks)
        ledgers = tuple(self._trade_ledger(item) for item in trades)
        index = pd.DatetimeIndex(session_dates)
        columns = list(trade_ids)
        close = pd.DataFrame(index=index, columns=columns, dtype=float)
        size = pd.DataFrame(np.nan, index=index, columns=columns, dtype=float)
        execution = pd.DataFrame(np.nan, index=index, columns=columns, dtype=float)
        fixed_costs = pd.DataFrame(0.0, index=index, columns=columns, dtype=float)

        for ledger in ledgers:
            intent = ledger.intent
            entry_index = session_dates.index(intent.entry_date)
            exit_index = session_dates.index(intent.exit_date)
            for offset, session_date in enumerate(session_dates):
                if offset < entry_index:
                    mark = intent.entry_price
                elif offset > exit_index:
                    mark = intent.exit_price
                else:
                    key = (intent.symbol, session_date)
                    try:
                        mark = marks_by_key[key]
                    except KeyError as error:
                        raise ValueError(
                            f"missing {intent.symbol} mark for held session {session_date}"
                        ) from error
                close.at[pd.Timestamp(session_date), intent.trade_id] = mark
            direction = 1.0 if intent.side == "long" else -1.0
            size.at[pd.Timestamp(intent.entry_date), intent.trade_id] = direction * intent.shares
            size.at[pd.Timestamp(intent.exit_date), intent.trade_id] = -direction * intent.shares
            execution.at[pd.Timestamp(intent.entry_date), intent.trade_id] = intent.entry_price
            execution.at[pd.Timestamp(intent.exit_date), intent.trade_id] = intent.exit_price
            fixed_costs.at[pd.Timestamp(intent.entry_date), intent.trade_id] = (
                ledger.entry_cost.total
            )
            fixed_costs.at[pd.Timestamp(intent.exit_date), intent.trade_id] = ledger.exit_cost.total

        vbt = _import_vectorbt()
        common = {
            "close": close,
            "size": size,
            "size_type": "amount",
            "direction": "both",
            "price": execution,
            "init_cash": initial_cash,
            "cash_sharing": True,
            "group_by": True,
            "call_seq": "auto",
            "allow_partial": False,
            "raise_reject": True,
            "freq": "1D",
        }
        gross_portfolio = vbt.Portfolio.from_orders(**common)
        net_portfolio = vbt.Portfolio.from_orders(**common, fixed_fees=fixed_costs)
        gross_values = gross_portfolio.value()
        net_values = net_portfolio.value()
        daily_costs = self._daily_costs(ledgers, session_dates)
        exposure = self._daily_exposure(
            trades=trades,
            marks_by_key=marks_by_key,
            sessions=session_dates,
        )
        daily = self._daily_ledger(
            sessions=session_dates,
            gross_values=gross_values,
            net_values=net_values,
            daily_costs=daily_costs,
            exposure=exposure,
            initial_cash=initial_cash,
        )
        self._validate_reconciliation(
            initial_cash=initial_cash,
            trades=ledgers,
            daily=daily,
        )
        return BacktestResult(
            engine=f"vectorbt-{vbt.__version__}",
            input_sha256=backtest_input_sha256(
                trades=trades,
                marks=marks,
                sessions=session_dates,
                initial_cash=initial_cash,
            ),
            initial_cash=initial_cash,
            trades=ledgers,
            daily=daily,
        )

    def _trade_ledger(self, intent: TradeIntent) -> TradeLedger:
        return _trade_ledger(intent, cost_model=self._cost_model)

    @staticmethod
    def _marks_by_key(marks: Sequence[DailyMark]) -> dict[tuple[str, date], float]:
        result: dict[tuple[str, date], float] = {}
        for item in marks:
            key = (item.symbol, item.session_date)
            if key in result:
                raise ValueError(f"duplicate daily mark for {key}")
            result[key] = item.close
        return result

    @staticmethod
    def _daily_costs(
        trades: Sequence[TradeLedger],
        sessions: Sequence[date],
    ) -> dict[date, list[float]]:
        result = {session: [0.0] * 5 for session in sessions}
        for trade in trades:
            for session, cost in (
                (trade.intent.entry_date, trade.entry_cost),
                (trade.intent.exit_date, trade.exit_cost),
            ):
                target = result[session]
                target[0] += cost.commission
                target[1] += cost.half_spread
                target[2] += cost.market_impact
                target[3] += cost.borrow
                target[4] += cost.stop_slippage
        return result

    @staticmethod
    def _daily_exposure(
        *,
        trades: Sequence[TradeIntent],
        marks_by_key: dict[tuple[str, date], float],
        sessions: Sequence[date],
    ) -> dict[date, float]:
        return {
            session: sum(
                item.shares * marks_by_key[(item.symbol, session)]
                for item in trades
                if item.entry_date <= session < item.exit_date
            )
            for session in sessions
        }

    @staticmethod
    def _daily_ledger(
        *,
        sessions: Sequence[date],
        gross_values: pd.Series,
        net_values: pd.Series,
        daily_costs: dict[date, list[float]],
        exposure: dict[date, float],
        initial_cash: float,
    ) -> tuple[DailyLedger, ...]:
        output: list[DailyLedger] = []
        previous_gross = initial_cash
        previous_net = initial_cash
        for index, session in enumerate(sessions):
            gross_equity = float(gross_values.iloc[index])
            net_equity = float(net_values.iloc[index])
            costs = daily_costs[session]
            output.append(
                DailyLedger(
                    session_date=session,
                    gross_pnl=gross_equity - previous_gross,
                    commission=costs[0],
                    half_spread=costs[1],
                    market_impact=costs[2],
                    borrow=costs[3],
                    stop_slippage=costs[4],
                    net_pnl=net_equity - previous_net,
                    gross_equity=gross_equity,
                    net_equity=net_equity,
                    gross_exposure=exposure[session],
                )
            )
            previous_gross = gross_equity
            previous_net = net_equity
        return tuple(output)

    @staticmethod
    def _validate_reconciliation(
        *,
        initial_cash: float,
        trades: Sequence[TradeLedger],
        daily: Sequence[DailyLedger],
    ) -> None:
        expected_gross = initial_cash + sum(item.gross_pnl for item in trades)
        expected_net = initial_cash + sum(item.net_pnl for item in trades)
        if not math.isclose(daily[-1].gross_equity, expected_gross, abs_tol=1e-8):
            raise RuntimeError("vectorbt gross equity does not reconcile to trade ledger")
        if not math.isclose(daily[-1].net_equity, expected_net, abs_tol=1e-8):
            raise RuntimeError("vectorbt net equity does not reconcile to trade ledger")
        for item in daily:
            if not math.isclose(
                item.net_pnl,
                item.gross_pnl - item.total_cost,
                abs_tol=1e-8,
            ):
                raise RuntimeError(
                    f"daily cost attribution does not reconcile on {item.session_date}"
                )


class VectorbtIntradayEngine:
    """Run timestamped same-session round trips and aggregate daily evidence."""

    def __init__(self, cost_model: CostModel | None = None) -> None:
        self._cost_model = cost_model or CostModel()

    def run(  # noqa: PLR0912 - atomic timestamp validation and vectorbt execution.
        self,
        *,
        trades: Sequence[TradeIntent],
        sessions: Sequence[date],
        initial_cash: float,
    ) -> BacktestResult:
        """Execute distinct intraday entry/exit orders with exact reconciliation."""
        if initial_cash <= 0 or not math.isfinite(initial_cash):
            raise ValueError("initial_cash must be finite and positive")
        session_dates = tuple(sessions)
        if not session_dates or tuple(sorted(set(session_dates))) != session_dates:
            raise ValueError("sessions must be non-empty, unique, and increasing")
        if not trades:
            raise ValueError("at least one trade is required")
        if len({item.trade_id for item in trades}) != len(trades):
            raise ValueError("trade_id values must be unique")
        session_set = set(session_dates)
        for item in trades:
            if item.entry_at is None or item.exit_at is None:
                raise ValueError("intraday trades require entry_at and exit_at")
            if item.entry_date != item.exit_date:
                raise ValueError("intraday engine accepts same-session round trips only")
            if item.entry_date not in session_set:
                raise ValueError("every intraday trade date must be in sessions")

        ledgers = tuple(_trade_ledger(item, cost_model=self._cost_model) for item in trades)
        timeline = tuple(
            sorted(
                {
                    timestamp
                    for item in trades
                    for timestamp in (item.entry_at, item.exit_at)
                    if timestamp is not None
                }
            )
        )
        index = pd.DatetimeIndex(timeline)
        columns = [item.trade_id for item in trades]
        close = pd.DataFrame(index=index, columns=columns, dtype=float)
        size = pd.DataFrame(np.nan, index=index, columns=columns, dtype=float)
        execution = pd.DataFrame(np.nan, index=index, columns=columns, dtype=float)
        fixed_costs = pd.DataFrame(0.0, index=index, columns=columns, dtype=float)
        for ledger in ledgers:
            intent = ledger.intent
            entry_at = intent.entry_at
            exit_at = intent.exit_at
            if entry_at is None or exit_at is None:
                raise RuntimeError("validated intraday timestamp disappeared")
            for timestamp in timeline:
                close.at[pd.Timestamp(timestamp), intent.trade_id] = (
                    intent.entry_price if timestamp < exit_at else intent.exit_price
                )
            direction = 1.0 if intent.side == "long" else -1.0
            size.at[pd.Timestamp(entry_at), intent.trade_id] = direction * intent.shares
            size.at[pd.Timestamp(exit_at), intent.trade_id] = -direction * intent.shares
            execution.at[pd.Timestamp(entry_at), intent.trade_id] = intent.entry_price
            execution.at[pd.Timestamp(exit_at), intent.trade_id] = intent.exit_price
            fixed_costs.at[pd.Timestamp(entry_at), intent.trade_id] = ledger.entry_cost.total
            fixed_costs.at[pd.Timestamp(exit_at), intent.trade_id] = ledger.exit_cost.total
        vbt = _import_vectorbt()
        common = {
            "close": close,
            "size": size,
            "size_type": "amount",
            "direction": "both",
            "price": execution,
            "init_cash": initial_cash,
            "cash_sharing": True,
            "group_by": True,
            "call_seq": "auto",
            "allow_partial": False,
            "raise_reject": True,
        }
        gross_final = float(vbt.Portfolio.from_orders(**common).value().iloc[-1])
        net_final = float(
            vbt.Portfolio.from_orders(**common, fixed_fees=fixed_costs).value().iloc[-1]
        )
        daily = _intraday_daily_ledger(
            trades=ledgers,
            sessions=session_dates,
            initial_cash=initial_cash,
        )
        expected_gross = initial_cash + sum(item.gross_pnl for item in ledgers)
        expected_net = initial_cash + sum(item.net_pnl for item in ledgers)
        if not math.isclose(gross_final, expected_gross, abs_tol=1e-8):
            raise RuntimeError("intraday vectorbt gross equity does not reconcile")
        if not math.isclose(net_final, expected_net, abs_tol=1e-8):
            raise RuntimeError("intraday vectorbt net equity does not reconcile")
        return BacktestResult(
            engine=f"vectorbt-intraday-{vbt.__version__}",
            input_sha256=backtest_input_sha256(
                trades=trades,
                marks=(),
                sessions=session_dates,
                initial_cash=initial_cash,
            ),
            initial_cash=initial_cash,
            trades=ledgers,
            daily=daily,
        )


def _trade_ledger(intent: TradeIntent, *, cost_model: CostModel) -> TradeLedger:
    is_short = intent.side == "short"
    entry = cost_model.estimate(
        ExecutionCostInput(
            side="sell" if is_short else "buy",
            shares=intent.shares,
            price=intent.entry_price,
            average_daily_volume_shares=intent.entry_average_daily_volume_shares,
            is_short_position=is_short,
            holding_days=intent.holding_sessions if is_short else 0,
        )
    )
    exit_cost = cost_model.estimate(
        ExecutionCostInput(
            side="buy" if is_short else "sell",
            shares=intent.shares,
            price=intent.exit_price,
            average_daily_volume_shares=intent.exit_average_daily_volume_shares,
            triggered_stop_price=intent.triggered_stop_price,
            atr5=intent.atr5,
        )
    )
    return TradeLedger(intent=intent, entry_cost=entry, exit_cost=exit_cost)


def _intraday_daily_ledger(
    *,
    trades: Sequence[TradeLedger],
    sessions: Sequence[date],
    initial_cash: float,
) -> tuple[DailyLedger, ...]:
    output: list[DailyLedger] = []
    gross_equity = initial_cash
    net_equity = initial_cash
    for session in sessions:
        session_trades = tuple(item for item in trades if item.intent.entry_date == session)
        gross_pnl = sum(item.gross_pnl for item in session_trades)
        commission = sum(
            item.entry_cost.commission + item.exit_cost.commission for item in session_trades
        )
        half_spread = sum(
            item.entry_cost.half_spread + item.exit_cost.half_spread for item in session_trades
        )
        market_impact = sum(
            item.entry_cost.market_impact + item.exit_cost.market_impact for item in session_trades
        )
        borrow = sum(item.entry_cost.borrow + item.exit_cost.borrow for item in session_trades)
        stop_slippage = sum(
            item.entry_cost.stop_slippage + item.exit_cost.stop_slippage for item in session_trades
        )
        total_cost = commission + half_spread + market_impact + borrow + stop_slippage
        net_pnl = gross_pnl - total_cost
        gross_equity += gross_pnl
        net_equity += net_pnl
        output.append(
            DailyLedger(
                session_date=session,
                gross_pnl=gross_pnl,
                commission=commission,
                half_spread=half_spread,
                market_impact=market_impact,
                borrow=borrow,
                stop_slippage=stop_slippage,
                net_pnl=net_pnl,
                gross_equity=gross_equity,
                net_equity=net_equity,
                gross_exposure=sum(item.entry_notional for item in session_trades),
            )
        )
    return tuple(output)


def _import_vectorbt() -> Any:
    try:
        import vectorbt as vbt  # noqa: PLC0415 - optional backtest dependency.
    except ImportError as error:
        raise RuntimeError(
            "vectorbt is required; install the project with the 'backtest' extra"
        ) from error
    return vbt


def backtest_input_sha256(
    *,
    trades: Sequence[TradeIntent],
    marks: Sequence[DailyMark],
    sessions: Sequence[date],
    initial_cash: float,
) -> str:
    evidence = {
        "initial_cash": initial_cash,
        "sessions": sessions,
        "trades": trades,
        "marks": sorted(marks, key=lambda item: (item.session_date, item.symbol)),
    }
    encoded = json.dumps(
        evidence,
        default=lambda item: (
            asdict(item) if hasattr(item, "__dataclass_fields__") else item.isoformat()
        ),
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()
