"""Deterministic standardized performance and cost-attribution reports."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pathlib import Path

    from quant_earning_edge.backtest import BacktestResult

_TRADING_SESSIONS_PER_YEAR = 252.0
_COST_COMPONENTS = (
    "commission",
    "half_spread",
    "market_impact",
    "borrow",
    "stop_slippage",
)


@dataclass(frozen=True)
class ConfidenceInterval:
    """Point estimate and percentile confidence bounds."""

    lower: float
    point: float
    upper: float


@dataclass(frozen=True)
class BootstrapSummary:
    """Trade-resampled uncertainty evidence."""

    seed: int
    resamples: int
    sharpe: ConfidenceInterval
    annualized_return: ConfidenceInterval


@dataclass(frozen=True)
class CostAttribution:
    """Dollar cost and sequential marginal Sharpe loss."""

    component: str
    dollars: float
    marginal_sharpe_loss: float


@dataclass(frozen=True)
class PerformanceReport:
    """Stable, machine-readable Phase 3 evaluation evidence."""

    engine: str
    input_sha256: str
    session_count: int
    trade_count: int
    initial_cash: float
    final_gross_equity: float
    final_net_equity: float
    gross_sharpe: float
    net_sharpe: float
    annualized_return: float
    max_drawdown: float
    hit_rate: float
    payoff: float | None
    average_gross_exposure: float
    turnover: float
    cost_attribution: tuple[CostAttribution, ...]
    bootstrap: BootstrapSummary | None

    def to_json_bytes(self) -> bytes:
        """Serialize canonical report evidence."""
        return json.dumps(
            asdict(self),
            sort_keys=True,
            separators=(",", ":"),
        ).encode()

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.to_json_bytes()).hexdigest()

    @classmethod
    def load(cls, path: Path) -> PerformanceReport:
        """Strictly reload canonical standardized performance evidence."""
        try:
            encoded = path.read_bytes()
            raw = json.loads(encoded)
            expected = {
                "engine",
                "input_sha256",
                "session_count",
                "trade_count",
                "initial_cash",
                "final_gross_equity",
                "final_net_equity",
                "gross_sharpe",
                "net_sharpe",
                "annualized_return",
                "max_drawdown",
                "hit_rate",
                "payoff",
                "average_gross_exposure",
                "turnover",
                "cost_attribution",
                "bootstrap",
            }
            if not isinstance(raw, dict) or set(raw) != expected:
                raise ValueError("performance report schema mismatch")
            costs_raw = raw["cost_attribution"]
            if not isinstance(costs_raw, list):
                raise TypeError
            costs = tuple(
                CostAttribution(
                    component=str(item["component"]),
                    dollars=float(item["dollars"]),
                    marginal_sharpe_loss=float(item["marginal_sharpe_loss"]),
                )
                for item in costs_raw
                if isinstance(item, dict)
                and set(item) == {"component", "dollars", "marginal_sharpe_loss"}
            )
            if len(costs) != len(costs_raw):
                raise ValueError("invalid performance cost attribution")
            bootstrap_raw = raw["bootstrap"]
            bootstrap = (
                _bootstrap_from_dict(bootstrap_raw) if isinstance(bootstrap_raw, dict) else None
            )
            if bootstrap_raw is not None and bootstrap is None:
                raise ValueError("invalid performance bootstrap")
            report = cls(
                engine=str(raw["engine"]),
                input_sha256=str(raw["input_sha256"]),
                session_count=int(raw["session_count"]),
                trade_count=int(raw["trade_count"]),
                initial_cash=float(raw["initial_cash"]),
                final_gross_equity=float(raw["final_gross_equity"]),
                final_net_equity=float(raw["final_net_equity"]),
                gross_sharpe=float(raw["gross_sharpe"]),
                net_sharpe=float(raw["net_sharpe"]),
                annualized_return=float(raw["annualized_return"]),
                max_drawdown=float(raw["max_drawdown"]),
                hit_rate=float(raw["hit_rate"]),
                payoff=(float(raw["payoff"]) if raw["payoff"] is not None else None),
                average_gross_exposure=float(raw["average_gross_exposure"]),
                turnover=float(raw["turnover"]),
                cost_attribution=costs,
                bootstrap=bootstrap,
            )
        except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid performance report: {path}") from error
        if report.to_json_bytes() != encoded:
            raise ValueError("performance report is not canonical")
        return report


class PerformanceEvaluator:
    """Compute standardized metrics from an already reconciled ledger."""

    def __init__(self, *, bootstrap_resamples: int = 10_000, seed: int = 20260427) -> None:
        if bootstrap_resamples < 1:
            raise ValueError("bootstrap_resamples must be positive")
        self._bootstrap_resamples = bootstrap_resamples
        self._seed = seed

    def evaluate(self, result: BacktestResult) -> PerformanceReport:
        """Return headline metrics, cost attribution, and bootstrap intervals."""
        gross_pnl = np.asarray([item.gross_pnl for item in result.daily], dtype=float)
        net_pnl = np.asarray([item.net_pnl for item in result.daily], dtype=float)
        gross_returns = _returns_from_pnl(gross_pnl, initial_cash=result.initial_cash)
        net_returns = _returns_from_pnl(net_pnl, initial_cash=result.initial_cash)
        gross_sharpe = _sharpe(gross_returns)
        net_sharpe = _sharpe(net_returns)
        net_equity = np.asarray([item.net_equity for item in result.daily], dtype=float)
        if np.any(net_equity <= 0):
            raise ValueError("performance metrics require strictly positive net equity")
        trade_returns = np.asarray([item.net_return for item in result.trades], dtype=float)
        wins = trade_returns[trade_returns > 0]
        losses = trade_returns[trade_returns < 0]
        prior_equity = np.concatenate(([result.initial_cash], net_equity[:-1]))
        exposure = np.asarray([item.gross_exposure for item in result.daily], dtype=float)
        traded_notional = sum(
            item.entry_notional + item.intent.exit_price * item.intent.shares
            for item in result.trades
        )
        session_count = len(result.daily)
        annualized_return = (result.final_net_equity / result.initial_cash) ** (
            _TRADING_SESSIONS_PER_YEAR / session_count
        ) - 1.0
        return PerformanceReport(
            engine=result.engine,
            input_sha256=result.input_sha256,
            session_count=session_count,
            trade_count=len(result.trades),
            initial_cash=result.initial_cash,
            final_gross_equity=result.final_gross_equity,
            final_net_equity=result.final_net_equity,
            gross_sharpe=gross_sharpe,
            net_sharpe=net_sharpe,
            annualized_return=annualized_return,
            max_drawdown=_max_drawdown(net_equity, initial_cash=result.initial_cash),
            hit_rate=(float(np.mean(trade_returns > 0)) if trade_returns.size else 0.0),
            payoff=(
                float(np.mean(wins) / abs(np.mean(losses))) if wins.size and losses.size else None
            ),
            average_gross_exposure=float(np.mean(exposure / prior_equity)),
            turnover=traded_notional / float(np.mean(prior_equity)),
            cost_attribution=_cost_attribution(
                result,
                gross_pnl=gross_pnl,
                gross_sharpe=gross_sharpe,
            ),
            bootstrap=(
                _bootstrap(
                    trade_returns,
                    session_count=session_count,
                    seed=self._seed,
                    resamples=self._bootstrap_resamples,
                )
                if trade_returns.size >= 2
                else None
            ),
        )

    @staticmethod
    def write(report: PerformanceReport, output: Path) -> None:
        """Persist immutable report JSON, accepting identical re-runs."""
        encoded = report.to_json_bytes()
        output.parent.mkdir(parents=True, exist_ok=True)
        try:
            with output.open("xb") as destination:
                destination.write(encoded)
        except FileExistsError:
            if output.read_bytes() != encoded:
                raise RuntimeError(f"performance report collision at {output}") from None


def _returns_from_pnl(pnl: np.ndarray, *, initial_cash: float) -> np.ndarray:
    equity_before = initial_cash + np.concatenate(([0.0], np.cumsum(pnl)[:-1]))
    if np.any(equity_before <= 0):
        raise ValueError("return calculation requires strictly positive prior equity")
    return pnl / equity_before


def _sharpe(returns: np.ndarray, *, periods_per_year: float = 252.0) -> float:
    if returns.size < 2:
        return 0.0
    standard_deviation = float(np.std(returns, ddof=1))
    if math.isclose(standard_deviation, 0.0, abs_tol=1e-15):
        return 0.0
    return float(np.mean(returns) / standard_deviation * math.sqrt(periods_per_year))


def _max_drawdown(equity: np.ndarray, *, initial_cash: float) -> float:
    full_path = np.concatenate(([initial_cash], equity))
    running_high = np.maximum.accumulate(full_path)
    return float(abs(np.min(full_path / running_high - 1.0)))


def _cost_attribution(
    result: BacktestResult,
    *,
    gross_pnl: np.ndarray,
    gross_sharpe: float,
) -> tuple[CostAttribution, ...]:
    running_pnl = gross_pnl.copy()
    before_sharpe = gross_sharpe
    output: list[CostAttribution] = []
    for component in _COST_COMPONENTS:
        component_cost = np.asarray(
            [float(getattr(item, component)) for item in result.daily],
            dtype=float,
        )
        after_pnl = running_pnl - component_cost
        after_sharpe = _sharpe(_returns_from_pnl(after_pnl, initial_cash=result.initial_cash))
        output.append(
            CostAttribution(
                component=component,
                dollars=float(np.sum(component_cost)),
                marginal_sharpe_loss=before_sharpe - after_sharpe,
            )
        )
        running_pnl = after_pnl
        before_sharpe = after_sharpe
    return tuple(output)


def _bootstrap(
    trade_returns: np.ndarray,
    *,
    session_count: int,
    seed: int,
    resamples: int,
) -> BootstrapSummary:
    generator = np.random.default_rng(seed)
    trades_per_year = trade_returns.size * _TRADING_SESSIONS_PER_YEAR / session_count
    sharpe_values = np.empty(resamples, dtype=float)
    annualized_values = np.empty(resamples, dtype=float)
    batch_size = 512
    for start in range(0, resamples, batch_size):
        stop = min(start + batch_size, resamples)
        sample = generator.choice(
            trade_returns,
            size=(stop - start, trade_returns.size),
            replace=True,
        )
        means = np.mean(sample, axis=1)
        standard_deviations = np.std(sample, axis=1, ddof=1)
        sharpe_values[start:stop] = np.divide(
            means * math.sqrt(trades_per_year),
            standard_deviations,
            out=np.zeros_like(means),
            where=~np.isclose(standard_deviations, 0.0, atol=1e-15),
        )
        annualized_values[start:stop] = means * trades_per_year
    point_sharpe = _sharpe(trade_returns, periods_per_year=trades_per_year)
    point_annualized = float(np.mean(trade_returns) * trades_per_year)
    return BootstrapSummary(
        seed=seed,
        resamples=resamples,
        sharpe=_interval(sharpe_values, point=point_sharpe),
        annualized_return=_interval(annualized_values, point=point_annualized),
    )


def _interval(values: np.ndarray, *, point: float) -> ConfidenceInterval:
    lower, upper = np.percentile(values, (2.5, 97.5))
    return ConfidenceInterval(lower=float(lower), point=point, upper=float(upper))


def _bootstrap_from_dict(raw: dict[str, object]) -> BootstrapSummary | None:
    if set(raw) != {"seed", "resamples", "sharpe", "annualized_return"}:
        return None
    sharpe = raw["sharpe"]
    annualized = raw["annualized_return"]
    if not isinstance(sharpe, dict) or not isinstance(annualized, dict):
        return None
    expected = {"lower", "point", "upper"}
    if set(sharpe) != expected or set(annualized) != expected:
        return None
    return BootstrapSummary(
        seed=int(str(raw["seed"])),
        resamples=int(str(raw["resamples"])),
        sharpe=ConfidenceInterval(
            lower=float(sharpe["lower"]),
            point=float(sharpe["point"]),
            upper=float(sharpe["upper"]),
        ),
        annualized_return=ConfidenceInterval(
            lower=float(annualized["lower"]),
            point=float(annualized["point"]),
            upper=float(annualized["upper"]),
        ),
    )
