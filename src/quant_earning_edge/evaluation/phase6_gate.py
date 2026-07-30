"""Aggregate authoritative daily replay records into the terminal Phase 6 gate."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from datetime import date  # noqa: TC003 - Pydantic resolves runtime annotations.
from pathlib import Path  # noqa: TC003 - Pydantic resolves runtime annotations.
from typing import TYPE_CHECKING, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from quant_earning_edge.evaluation.report import ConfidenceInterval, CostAttribution

if TYPE_CHECKING:
    from collections.abc import Sequence

    from quant_earning_edge.data.calendar import SessionFile
    from quant_earning_edge.evaluation.replay_session import ReplaySessionReport
    from quant_earning_edge.orchestration import WorkflowHealthReport

_TRADING_SESSIONS_PER_YEAR = 252.0


class _StrictSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Phase6AggregationSpec(_StrictSpec):
    """Immutable daily files and calendar bounds for one terminal proof."""

    session_file: Path
    workflow_store_root: Path
    workflow_health_file: Path
    proof_start: date
    proof_end: date
    initial_cash: float = Field(gt=0)
    session_report_files: tuple[Path, ...]
    minimum_session_count: Literal[90] = 90
    bootstrap_resamples: int = Field(default=10_000, ge=1)
    seed: int = 20260427

    @model_validator(mode="after")
    def validate_dates(self) -> Phase6AggregationSpec:
        if self.proof_end < self.proof_start:
            raise ValueError("proof_end must be on or after proof_start")
        return self


@dataclass(frozen=True)
class Phase6GateReport:
    """Machine-readable verdict for the documented terminal thresholds."""

    schema_version: int
    calendar_sha256: str
    workflow_health_sha256: str
    session_report_sha256: tuple[str, ...]
    bootstrap_resamples: int
    seed: int
    proof_start: date
    proof_end: date
    required_session_count: int
    authoritative_session_count: int
    observed_session_count: int
    scheduled_complete_session_count: int
    missing_session_dates: tuple[date, ...]
    operational_uptime: float
    initial_cash: float
    final_equity: float | None
    net_sharpe: float | None
    bootstrap_sharpe: ConfidenceInterval | None
    max_drawdown: float | None
    intended_order_count: int
    fully_filled_order_count: int
    fully_filled_order_rate: float | None
    intended_share_count: int
    filled_share_count: int
    share_fill_rate: float | None
    realized_adverse_slippage_bps_p10: float | None
    realized_adverse_slippage_bps_p50: float | None
    realized_adverse_slippage_bps_p90: float | None
    predicted_adverse_slippage_bps_p90: float | None
    p90_realized_to_predicted_ratio: float | None
    reconciliation_break_dates: tuple[date, ...]
    arrival_gross_pnl: float | None
    modeled_spread_cost: float
    modeled_market_impact_cost: float
    execution_residual_cost: float
    fill_gross_pnl: float | None
    commission: float
    net_pnl: float | None
    cost_attribution: tuple[CostAttribution, ...]
    passes_session_count_gate: bool
    passes_net_sharpe_gate: bool
    passes_fill_rate_gate: bool
    passes_slippage_gate: bool
    passes_uptime_gate: bool
    passes_reconciliation_gate: bool
    passes_phase6_gate: bool
    verdict: Literal["pass", "fail", "insufficient-evidence"]

    @property
    def canonical_bytes(self) -> bytes:
        return json.dumps(
            asdict(self),
            default=lambda item: item.isoformat(),
            sort_keys=True,
            separators=(",", ":"),
        ).encode()

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes).hexdigest()

    def write(self, output: Path) -> None:
        encoded = self.canonical_bytes
        output.parent.mkdir(parents=True, exist_ok=True)
        try:
            with output.open("xb") as destination:
                destination.write(encoded)
        except FileExistsError:
            if output.read_bytes() != encoded:
                raise RuntimeError(f"Phase 6 gate report collision at {output}") from None


class Phase6GateEvaluator:
    """Apply the locked 90-session NBBO replay thresholds without paper P&L."""

    def __init__(self, *, bootstrap_resamples: int = 10_000, seed: int = 20260427) -> None:
        if bootstrap_resamples < 1:
            raise ValueError("bootstrap_resamples must be positive")
        self._bootstrap_resamples = bootstrap_resamples
        self._seed = seed

    def evaluate(
        self,
        *,
        calendar: SessionFile,
        workflow_health: WorkflowHealthReport,
        reports: Sequence[ReplaySessionReport],
        proof_start: date,
        proof_end: date,
        initial_cash: float,
    ) -> Phase6GateReport:
        if proof_end < proof_start:
            raise ValueError("proof_end must be on or after proof_start")
        if not math.isfinite(initial_cash) or initial_cash <= 0:
            raise ValueError("initial_cash must be finite and positive")
        session_dates = tuple(
            item.session_date
            for item in calendar.sessions
            if proof_start <= item.session_date <= proof_end
        )
        if not session_dates or session_dates[0] != proof_start or session_dates[-1] != proof_end:
            raise ValueError("proof bounds must both be authoritative trading sessions")
        if workflow_health.calendar_sha256 != calendar.sha256:
            raise ValueError("workflow health calendar does not match Phase 6 calendar")
        if (
            workflow_health.start_date != proof_start
            or workflow_health.end_date != proof_end
            or workflow_health.authoritative_session_dates != session_dates
        ):
            raise ValueError("workflow health range does not exactly match Phase 6 proof sessions")
        by_date = {item.session_date: item for item in reports}
        if len(by_date) != len(reports):
            raise ValueError("daily replay reports contain duplicate session dates")
        if not set(by_date).issubset(session_dates):
            raise ValueError("daily replay report falls outside authoritative proof sessions")
        missing = tuple(item for item in session_dates if item not in by_date)
        ordered_reports = tuple(by_date[item] for item in session_dates if item in by_date)
        performance = self._performance(
            session_dates=session_dates,
            by_date=by_date,
            initial_cash=initial_cash,
        )
        return self._report(
            calendar=calendar,
            workflow_health=workflow_health,
            session_dates=session_dates,
            reports=ordered_reports,
            missing=missing,
            proof_start=proof_start,
            proof_end=proof_end,
            initial_cash=initial_cash,
            performance=performance,
            bootstrap_resamples=self._bootstrap_resamples,
            seed=self._seed,
        )

    def _performance(
        self,
        *,
        session_dates: tuple[date, ...],
        by_date: dict[date, ReplaySessionReport],
        initial_cash: float,
    ) -> tuple[float | None, float | None, ConfidenceInterval | None, float | None]:
        current_equity = initial_cash
        returns: list[float] = []
        equity_path: list[float] = []
        valid = True
        for session_date in session_dates:
            report = by_date.get(session_date)
            if report is None:
                returns.append(0.0)
                equity_path.append(current_equity)
                continue
            if not math.isclose(
                report.initial_cash,
                current_equity,
                rel_tol=1e-10,
                abs_tol=1e-8,
            ):
                raise ValueError(f"daily capital continuity breaks on {session_date}")
            if report.net_pnl is None or report.net_return is None:
                valid = False
                returns.append(0.0)
                equity_path.append(current_equity)
                continue
            session_return = report.net_pnl / current_equity
            if not math.isclose(
                report.net_return,
                session_return,
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                raise ValueError(f"daily return does not reconcile on {session_date}")
            current_equity += report.net_pnl
            if current_equity <= 0:
                raise ValueError("Phase 6 equity must remain positive")
            returns.append(session_return)
            equity_path.append(current_equity)
        if not valid:
            return None, None, None, None
        values = np.asarray(returns, dtype=float)
        sharpe = _sharpe(values)
        interval = _bootstrap_sharpe(
            values,
            point=sharpe,
            resamples=self._bootstrap_resamples,
            seed=self._seed,
        )
        drawdown = _max_drawdown(
            np.asarray(equity_path, dtype=float),
            initial_cash=initial_cash,
        )
        return current_equity, sharpe, interval, drawdown

    @staticmethod
    def _report(
        *,
        calendar: SessionFile,
        workflow_health: WorkflowHealthReport,
        session_dates: tuple[date, ...],
        reports: tuple[ReplaySessionReport, ...],
        missing: tuple[date, ...],
        proof_start: date,
        proof_end: date,
        initial_cash: float,
        bootstrap_resamples: int,
        seed: int,
        performance: tuple[
            float | None,
            float | None,
            ConfidenceInterval | None,
            float | None,
        ],
    ) -> Phase6GateReport:
        final_equity, net_sharpe, bootstrap, max_drawdown = performance
        intended_orders = sum(item.intended_order_count for item in reports)
        fully_filled = sum(item.fully_filled_order_count for item in reports)
        intended_shares = sum(item.intended_share_count for item in reports)
        filled_shares = sum(item.filled_share_count for item in reports)
        realized = tuple(
            value for report in reports for value in report.realized_adverse_slippage_bps
        )
        predicted = tuple(
            value for report in reports for value in report.predicted_adverse_slippage_bps
        )
        realized_p90 = _percentile(realized, 0.90)
        predicted_p90 = _percentile(predicted, 0.90)
        ratio = (
            realized_p90 / predicted_p90
            if realized_p90 is not None and predicted_p90 is not None and predicted_p90 > 0
            else None
        )
        breaks = tuple(
            report.session_date for report in reports if report.reconciliation_break_count
        )
        arrival_gross_pnl = (
            sum(item.arrival_gross_pnl or 0.0 for item in reports) if not breaks else None
        )
        fill_gross_pnl = sum(item.gross_pnl or 0.0 for item in reports) if not breaks else None
        modeled_spread_cost = sum(item.modeled_spread_cost for item in reports)
        modeled_impact_cost = sum(item.modeled_market_impact_cost for item in reports)
        execution_residual_cost = sum(item.execution_residual_cost for item in reports)
        commission = sum(item.commission for item in reports)
        session_count_gate = len(session_dates) >= 90 and not missing
        sharpe_gate = (
            net_sharpe is not None
            and bootstrap is not None
            and net_sharpe > 0.8
            and bootstrap.lower > 0.3
        )
        fill_rate = fully_filled / intended_orders if intended_orders else None
        fill_gate = fill_rate is not None and fill_rate > 0.90
        slippage_gate = (
            realized_p90 is not None
            and predicted_p90 is not None
            and realized_p90 < 2 * predicted_p90
        )
        uptime = workflow_health.operational_uptime
        uptime_gate = uptime > 0.95
        reconciliation_gate = not breaks
        phase6_gate = all(
            (
                session_count_gate,
                sharpe_gate,
                fill_gate,
                slippage_gate,
                uptime_gate,
                reconciliation_gate,
            )
        )
        return Phase6GateReport(
            schema_version=4,
            calendar_sha256=calendar.sha256,
            workflow_health_sha256=workflow_health.sha256,
            session_report_sha256=tuple(item.sha256 for item in reports),
            bootstrap_resamples=bootstrap_resamples,
            seed=seed,
            proof_start=proof_start,
            proof_end=proof_end,
            required_session_count=90,
            authoritative_session_count=len(session_dates),
            observed_session_count=len(reports),
            scheduled_complete_session_count=len(workflow_health.scheduled_complete_dates),
            missing_session_dates=missing,
            operational_uptime=uptime,
            initial_cash=initial_cash,
            final_equity=final_equity,
            net_sharpe=net_sharpe,
            bootstrap_sharpe=bootstrap,
            max_drawdown=max_drawdown,
            intended_order_count=intended_orders,
            fully_filled_order_count=fully_filled,
            fully_filled_order_rate=fill_rate,
            intended_share_count=intended_shares,
            filled_share_count=filled_shares,
            share_fill_rate=filled_shares / intended_shares if intended_shares else None,
            realized_adverse_slippage_bps_p10=_percentile(realized, 0.10),
            realized_adverse_slippage_bps_p50=_percentile(realized, 0.50),
            realized_adverse_slippage_bps_p90=realized_p90,
            predicted_adverse_slippage_bps_p90=predicted_p90,
            p90_realized_to_predicted_ratio=ratio,
            reconciliation_break_dates=breaks,
            arrival_gross_pnl=arrival_gross_pnl,
            modeled_spread_cost=modeled_spread_cost,
            modeled_market_impact_cost=modeled_impact_cost,
            execution_residual_cost=execution_residual_cost,
            fill_gross_pnl=fill_gross_pnl,
            commission=commission,
            net_pnl=(final_equity - initial_cash if final_equity is not None else None),
            cost_attribution=(
                _cost_attribution(
                    session_dates=session_dates,
                    reports=reports,
                    initial_cash=initial_cash,
                )
                if not breaks
                else ()
            ),
            passes_session_count_gate=session_count_gate,
            passes_net_sharpe_gate=sharpe_gate,
            passes_fill_rate_gate=fill_gate,
            passes_slippage_gate=slippage_gate,
            passes_uptime_gate=uptime_gate,
            passes_reconciliation_gate=reconciliation_gate,
            passes_phase6_gate=phase6_gate,
            verdict=(
                "insufficient-evidence"
                if not session_count_gate
                else "pass"
                if phase6_gate
                else "fail"
            ),
        )


def _sharpe(returns: np.ndarray) -> float:
    if returns.size < 2:
        return 0.0
    deviation = float(np.std(returns, ddof=1))
    if math.isclose(deviation, 0.0, abs_tol=1e-15):
        return 0.0
    return float(np.mean(returns) / deviation * math.sqrt(_TRADING_SESSIONS_PER_YEAR))


def _bootstrap_sharpe(
    returns: np.ndarray,
    *,
    point: float,
    resamples: int,
    seed: int,
) -> ConfidenceInterval:
    if returns.size < 2:
        return ConfidenceInterval(lower=point, point=point, upper=point)
    generator = np.random.default_rng(seed)
    values = np.empty(resamples, dtype=float)
    batch_size = 512
    for start in range(0, resamples, batch_size):
        stop = min(start + batch_size, resamples)
        samples = generator.choice(
            returns,
            size=(stop - start, returns.size),
            replace=True,
        )
        means = np.mean(samples, axis=1)
        deviations = np.std(samples, axis=1, ddof=1)
        values[start:stop] = np.divide(
            means * math.sqrt(_TRADING_SESSIONS_PER_YEAR),
            deviations,
            out=np.zeros_like(means),
            where=~np.isclose(deviations, 0.0, atol=1e-15),
        )
    lower, upper = np.percentile(values, (2.5, 97.5))
    return ConfidenceInterval(lower=float(lower), point=point, upper=float(upper))


def _max_drawdown(equity: np.ndarray, *, initial_cash: float) -> float:
    full_path = np.concatenate(([initial_cash], equity))
    running_high = np.maximum.accumulate(full_path)
    return float(abs(np.min(full_path / running_high - 1.0)))


def _cost_attribution(
    *,
    session_dates: tuple[date, ...],
    reports: tuple[ReplaySessionReport, ...],
    initial_cash: float,
) -> tuple[CostAttribution, ...]:
    by_date = {item.session_date: item for item in reports}
    arrival_pnl = np.asarray(
        [
            (by_date[item].arrival_gross_pnl or 0.0) if item in by_date else 0.0
            for item in session_dates
        ],
        dtype=float,
    )
    component_values = (
        ("modeled_spread", "modeled_spread_cost"),
        ("modeled_market_impact", "modeled_market_impact_cost"),
        ("execution_residual", "execution_residual_cost"),
        ("commission", "commission"),
    )
    running_pnl = arrival_pnl
    before_sharpe = _sharpe(_returns_from_pnl(running_pnl, initial_cash=initial_cash))
    output: list[CostAttribution] = []
    for component, field in component_values:
        costs = np.asarray(
            [
                float(getattr(by_date[item], field)) if item in by_date else 0.0
                for item in session_dates
            ],
            dtype=float,
        )
        after_pnl = running_pnl - costs
        after_sharpe = _sharpe(_returns_from_pnl(after_pnl, initial_cash=initial_cash))
        output.append(
            CostAttribution(
                component=component,
                dollars=float(np.sum(costs)),
                marginal_sharpe_loss=before_sharpe - after_sharpe,
            )
        )
        running_pnl = after_pnl
        before_sharpe = after_sharpe
    return tuple(output)


def _returns_from_pnl(pnl: np.ndarray, *, initial_cash: float) -> np.ndarray:
    equity_before = initial_cash + np.concatenate(([0.0], np.cumsum(pnl)[:-1]))
    if np.any(equity_before <= 0):
        raise ValueError("cost attribution requires positive prior equity")
    return pnl / equity_before


def _percentile(values: Sequence[float], probability: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=float), probability * 100))
