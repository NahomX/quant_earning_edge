"""Chronological strategy aggregation and documented Phase 4 gates."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, replace
from datetime import date  # noqa: TC003 - Pydantic resolves runtime fields.
from pathlib import Path  # noqa: TC003 - Pydantic resolves runtime fields.
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict

from quant_earning_edge.backtest import BacktestResult
from quant_earning_edge.evaluation.folds import (
    WalkForwardEvaluation,
    WalkForwardEvaluator,
)
from quant_earning_edge.evaluation.report import PerformanceEvaluator, PerformanceReport

if TYPE_CHECKING:
    from collections.abc import Sequence


class FoldArtifactSpec(BaseModel):
    """Serialized fold assignment for immutable event plans."""

    model_config = ConfigDict(extra="forbid")

    fold_index: int
    test_start_date: date
    test_end_date: date
    event_plan_files: tuple[Path, ...]


class Phase4AggregationSpec(BaseModel):
    """Complete fold mapping for a Phase 4 gate run."""

    model_config = ConfigDict(extra="forbid")

    folds: tuple[FoldArtifactSpec, ...]


@dataclass(frozen=True)
class FoldBacktestResults:
    """Chronological event-session results assigned to one OOS fold."""

    fold_index: int
    test_start_date: date
    test_end_date: date
    results: tuple[BacktestResult, ...]


@dataclass(frozen=True)
class Phase4GateEvaluation:
    """Overall/fold metrics and both precommitted research gates."""

    overall: PerformanceReport
    walk_forward: WalkForwardEvaluation
    passes_phase4_research_gate: bool
    passes_pre_paper_backtest_gate: bool

    def to_json_bytes(self) -> bytes:
        return json.dumps(
            asdict(self),
            default=lambda item: item.isoformat(),
            sort_keys=True,
            separators=(",", ":"),
        ).encode()

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.to_json_bytes()).hexdigest()


@dataclass(frozen=True)
class Phase4PromotionEvidence:
    """Minimal verified decision needed to authorize a deployable model refit."""

    report_sha256: str
    net_sharpe: float
    lower_sharpe: float
    max_drawdown: float
    positive_fold_gate: bool

    @classmethod
    def load(cls, path: Path) -> Phase4PromotionEvidence:
        """Reload canonical Phase 4 JSON and independently recompute both gates."""
        encoded = path.read_bytes()
        raw = json.loads(encoded)
        expected = {
            "overall",
            "walk_forward",
            "passes_phase4_research_gate",
            "passes_pre_paper_backtest_gate",
        }
        if not isinstance(raw, dict) or set(raw) != expected:
            raise ValueError("Phase 4 gate report schema mismatch")
        if json.dumps(raw, sort_keys=True, separators=(",", ":")).encode() != encoded:
            raise ValueError("Phase 4 gate report is not canonical")
        try:
            overall = raw["overall"]
            walk_forward = raw["walk_forward"]
            bootstrap = overall["bootstrap"]
            net_sharpe = float(overall["net_sharpe"])
            max_drawdown = float(overall["max_drawdown"])
            lower_sharpe = float(bootstrap["sharpe"]["lower"])
            positive_fold_gate = bool(walk_forward["passes_positive_fold_gate"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("invalid Phase 4 gate report") from error
        research = net_sharpe >= 0.8 and lower_sharpe >= 0.3 and max_drawdown <= 0.20
        pre_paper = (
            net_sharpe > 1.0 and lower_sharpe > 0.5 and max_drawdown < 0.15 and positive_fold_gate
        )
        if bool(raw["passes_phase4_research_gate"]) != research:
            raise ValueError("Phase 4 research gate verdict is inconsistent")
        if bool(raw["passes_pre_paper_backtest_gate"]) != pre_paper:
            raise ValueError("Phase 4 pre-paper gate verdict is inconsistent")
        if not pre_paper:
            raise ValueError("Phase 4 pre-paper backtest gate did not pass")
        return cls(
            report_sha256=hashlib.sha256(encoded).hexdigest(),
            net_sharpe=net_sharpe,
            lower_sharpe=lower_sharpe,
            max_drawdown=max_drawdown,
            positive_fold_gate=positive_fold_gate,
        )


class BacktestResultCombiner:
    """Combine event-session ledgers only when equity and time are continuous."""

    def combine(self, results: Sequence[BacktestResult]) -> BacktestResult:
        """Return one ledger without resetting capital between sessions."""
        if not results:
            raise ValueError("at least one backtest result is required")
        previous_date: date | None = None
        expected_initial = results[0].initial_cash
        trade_ids: set[str] = set()
        all_trades = []
        all_daily = []
        gross_equity = results[0].initial_cash
        net_equity = results[0].initial_cash
        for result in results:
            if not math.isclose(result.initial_cash, expected_initial, abs_tol=1e-8):
                raise ValueError("backtest result equity is not chronologically chained")
            if not result.daily:
                raise ValueError("backtest result has no daily ledger")
            dates = tuple(item.session_date for item in result.daily)
            if dates != tuple(sorted(set(dates))):
                raise ValueError("backtest result dates must be unique and increasing")
            if previous_date is not None and dates[0] <= previous_date:
                raise ValueError("backtest results overlap or are out of order")
            for trade in result.trades:
                if trade.intent.trade_id in trade_ids:
                    raise ValueError("trade IDs must be unique across combined results")
                trade_ids.add(trade.intent.trade_id)
                all_trades.append(trade)
            for daily in result.daily:
                gross_equity += daily.gross_pnl
                net_equity += daily.net_pnl
                all_daily.append(
                    replace(
                        daily,
                        gross_equity=gross_equity,
                        net_equity=net_equity,
                    )
                )
            previous_date = dates[-1]
            expected_initial = result.final_net_equity
        digest = hashlib.sha256(
            "|".join(item.input_sha256 for item in results).encode()
        ).hexdigest()
        engines = ",".join(sorted({item.engine for item in results}))
        return BacktestResult(
            engine=f"combined[{engines}]",
            input_sha256=digest,
            initial_cash=results[0].initial_cash,
            trades=tuple(all_trades),
            daily=tuple(all_daily),
        )


class Phase4GateEvaluator:
    """Evaluate fold and overall ledgers against both documented thresholds."""

    def __init__(self, *, bootstrap_resamples: int = 10_000, seed: int = 20260427) -> None:
        self._performance = PerformanceEvaluator(
            bootstrap_resamples=bootstrap_resamples,
            seed=seed,
        )
        self._combiner = BacktestResultCombiner()

    def evaluate(self, folds: Sequence[FoldBacktestResults]) -> Phase4GateEvaluation:
        """Aggregate OOS folds and expose research/pre-paper decisions."""
        if not folds:
            raise ValueError("at least one fold result is required")
        fold_reports = []
        all_results: list[BacktestResult] = []
        for fold in folds:
            combined = self._combiner.combine(fold.results)
            if (
                combined.daily[0].session_date < fold.test_start_date
                or combined.daily[-1].session_date > fold.test_end_date
            ):
                raise ValueError(f"fold {fold.fold_index} result lies outside its test window")
            report = self._performance.evaluate(combined)
            fold_reports.append(
                (
                    fold.fold_index,
                    fold.test_start_date,
                    fold.test_end_date,
                    report,
                )
            )
            all_results.extend(fold.results)
        overall = self._performance.evaluate(self._combiner.combine(all_results))
        walk_forward = WalkForwardEvaluator().aggregate(fold_reports)
        lower_sharpe = (
            overall.bootstrap.sharpe.lower if overall.bootstrap is not None else float("-inf")
        )
        phase4 = overall.net_sharpe >= 0.8 and lower_sharpe >= 0.3 and overall.max_drawdown <= 0.20
        pre_paper = (
            overall.net_sharpe > 1.0
            and lower_sharpe > 0.5
            and overall.max_drawdown < 0.15
            and walk_forward.passes_positive_fold_gate
        )
        return Phase4GateEvaluation(
            overall=overall,
            walk_forward=walk_forward,
            passes_phase4_research_gate=phase4,
            passes_pre_paper_backtest_gate=pre_paper,
        )

    @staticmethod
    def write(report: Phase4GateEvaluation, output: Path) -> None:
        encoded = report.to_json_bytes()
        output.parent.mkdir(parents=True, exist_ok=True)
        try:
            with output.open("xb") as destination:
                destination.write(encoded)
        except FileExistsError:
            if output.read_bytes() != encoded:
                raise RuntimeError(f"Phase 4 gate report collision at {output}") from None
