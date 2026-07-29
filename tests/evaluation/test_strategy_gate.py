"""Chronological Phase 4 strategy gate tests."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pytest
from typer.testing import CliRunner

from quant_earning_edge.backtest import (
    BacktestResult,
    CostBreakdown,
    DailyLedger,
    TradeIntent,
    TradeLedger,
    VectorbtIntradayEngine,
)
from quant_earning_edge.cli import app
from quant_earning_edge.evaluation import (
    BacktestResultCombiner,
    FoldBacktestResults,
    Phase4GateEvaluator,
    Phase4PromotionEvidence,
)
from quant_earning_edge.evaluation.phase4_assembly import Phase4AssemblyManifest
from quant_earning_edge.portfolio import PortfolioPlan
from quant_earning_edge.signals import (
    EventTradePlanner,
    FeatureAttribution,
    FoldModelResult,
    LightgbmHyperparameters,
    OosPrediction,
    PlannedEventTrades,
    TradeCohort,
    WalkForwardModelRun,
    load_strategy_config,
)


def _source_reference(path: Path) -> dict[str, str]:
    return {
        "path": path.name,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _result(session: date, *, initial_cash: float, pnl: float, index: int) -> BacktestResult:
    entry = 100.0
    shares = 10
    exit_price = entry + pnl / shares
    intent = TradeIntent(
        trade_id=f"trade-{index}",
        symbol="AAA",
        side="long",
        entry_date=session,
        exit_date=session,
        shares=shares,
        entry_price=entry,
        exit_price=exit_price,
        entry_average_daily_volume_shares=1_000_000,
        exit_average_daily_volume_shares=1_000_000,
        holding_sessions=0,
        entry_at=datetime.combine(session, datetime.min.time(), UTC).replace(hour=14),
        exit_at=datetime.combine(session, datetime.min.time(), UTC).replace(hour=21),
    )
    zero_entry = CostBreakdown(entry * shares, 0, 0, 0, 0, 0)
    zero_exit = CostBreakdown(exit_price * shares, 0, 0, 0, 0, 0)
    trade = TradeLedger(intent=intent, entry_cost=zero_entry, exit_cost=zero_exit)
    daily = DailyLedger(
        session_date=session,
        gross_pnl=pnl,
        commission=0,
        half_spread=0,
        market_impact=0,
        borrow=0,
        stop_slippage=0,
        net_pnl=pnl,
        gross_equity=initial_cash + pnl,
        net_equity=initial_cash + pnl,
        gross_exposure=entry * shares,
    )
    return BacktestResult(
        engine="fixture",
        input_sha256=f"{index:064x}",
        initial_cash=initial_cash,
        trades=(trade,),
        daily=(daily,),
    )


def _chained_results() -> tuple[BacktestResult, ...]:
    first = date(2025, 1, 2)
    pnls = (100, -25, 80, 40, -15, 110, -30, 70, 60, -20) * 2
    results = []
    equity = 100_000.0
    for index, pnl in enumerate(pnls):
        result = _result(
            first + timedelta(days=index),
            initial_cash=equity,
            pnl=float(pnl),
            index=index,
        )
        results.append(result)
        equity = result.final_net_equity
    return tuple(results)


def _cohorts(results: tuple[BacktestResult, ...]) -> tuple[TradeCohort, ...]:
    return tuple(
        TradeCohort(
            trade_id=result.trades[0].intent.trade_id,
            event_timing="bmo" if index % 2 == 0 else "amc",
            sector="Technology" if index % 3 else "Health Care",
            iv_regime=("low", "medium", "high")[index % 3],
        )
        for index, result in enumerate(results)
    )


def _promotion_cohort_rows() -> list[dict[str, object]]:
    return [
        {
            "dimension": dimension,
            "value": value,
            "trade_count": 10,
            "session_count": 10,
            "total_net_pnl": 500.0,
            "mean_net_return": 0.01,
            "net_sharpe": 1.1,
            "hit_rate": 0.6,
            "payoff": 1.5,
        }
        for dimension, value in (
            ("event_timing", "bmo"),
            ("sector", "Technology"),
            ("iv_regime", "unavailable"),
        )
    ]


def _walkforward_run(predictions: tuple[OosPrediction, ...]) -> WalkForwardModelRun:
    strategy = load_strategy_config(Path("configs/strategies/earnings_v1.yaml"))
    hyperparameters = LightgbmHyperparameters()
    return WalkForwardModelRun(
        plan_sha256="a" * 64,
        dataset_sha256=("b" * 64,),
        feature_names=strategy.features,
        label_name=strategy.label.column_name,
        threshold=strategy.label.threshold,
        seed=strategy.seed,
        hyperparameter_study_sha256="c" * 64,
        hyperparameters=hyperparameters,
        hyperparameters_sha256=hyperparameters.sha256,
        lightgbm_version="test",
        folds=(
            FoldModelResult(
                fold_index=0,
                model_sha256="d" * 64,
                best_iteration=1,
                fit_count=10,
                validation_count=2,
                predictions=predictions,
                feature_attribution=tuple(
                    FeatureAttribution(name, 0.0) for name in strategy.features
                ),
            ),
        ),
    )


def test_phase4_report_is_deterministic_and_persisted(tmp_path: Path) -> None:
    results = _chained_results()
    folds = (
        FoldBacktestResults(
            0,
            results[0].daily[0].session_date,
            results[9].daily[0].session_date,
            results[:10],
            _cohorts(results[:10]),
        ),
        FoldBacktestResults(
            1,
            results[10].daily[0].session_date,
            results[-1].daily[0].session_date,
            results[10:],
            _cohorts(results[10:]),
        ),
    )
    evaluator = Phase4GateEvaluator(bootstrap_resamples=100, seed=7)

    first = evaluator.evaluate(
        folds,
        strategy_sha256="a" * 64,
        assembly_manifest_sha256="b" * 64,
        walkforward_run_sha256="e" * 64,
        hyperparameter_study_sha256="f" * 64,
    )
    second = evaluator.evaluate(
        folds,
        strategy_sha256="a" * 64,
        assembly_manifest_sha256="b" * 64,
        walkforward_run_sha256="e" * 64,
        hyperparameter_study_sha256="f" * 64,
    )
    output = tmp_path / "phase4-gate.json"
    evaluator.write(first, output)
    evaluator.write(first, output)

    assert first == second
    assert first.overall.trade_count == 20
    assert first.walk_forward.positive_sharpe_fraction == 1.0
    assert {item.dimension for item in first.cohorts} == {
        "event_timing",
        "sector",
        "iv_regime",
    }
    assert first.passes_phase4_research_gate == (
        first.overall.net_sharpe >= 0.8
        and first.overall.bootstrap is not None
        and first.overall.bootstrap.sharpe.lower >= 0.3
        and first.overall.max_drawdown <= 0.20
    )
    assert output.exists()


def test_combiner_rejects_equity_discontinuity() -> None:
    results = _chained_results()
    broken = replace_result_initial(results[1], results[1].initial_cash + 1)

    with pytest.raises(ValueError, match="equity"):
        BacktestResultCombiner().combine((results[0], broken))


def test_phase4_gate_rejects_missing_cohort_trade() -> None:
    results = _chained_results()[:2]
    folds = (
        FoldBacktestResults(
            0,
            results[0].daily[0].session_date,
            results[-1].daily[0].session_date,
            results,
            _cohorts(results)[:1],
        ),
    )

    with pytest.raises(ValueError, match="does not cover every trade"):
        Phase4GateEvaluator(bootstrap_resamples=10).evaluate(
            folds,
            strategy_sha256="a" * 64,
            assembly_manifest_sha256="b" * 64,
            walkforward_run_sha256="e" * 64,
            hyperparameter_study_sha256="f" * 64,
        )


def replace_result_initial(result: BacktestResult, value: float) -> BacktestResult:
    return BacktestResult(
        engine=result.engine,
        input_sha256=result.input_sha256,
        initial_cash=value,
        trades=result.trades,
        daily=result.daily,
    )


def test_phase4_gate_cli_rejects_nonreproducible_manual_sources(tmp_path: Path) -> None:
    first_date = date(2025, 1, 2)
    second_date = first_date + timedelta(days=1)
    first_fixture = _result(first_date, initial_cash=100_000, pnl=100, index=0)
    first_prediction = OosPrediction(0, "AAA", first_date, 0.8, 1)
    second_prediction = OosPrediction(1, "AAA", second_date, 0.7, 1)
    model_run = _walkforward_run((first_prediction, second_prediction))
    run_path = tmp_path / "walkforward-run.json"
    run_path.write_bytes(model_run.evidence_json_bytes())
    first_plan = PlannedEventTrades(
        trade_date=first_date,
        portfolio=PortfolioPlan(100_000, 20, 0.1, 0.025, (), 0.01, ()),
        intents=(first_fixture.trades[0].intent,),
        cohorts=(
            TradeCohort(
                first_fixture.trades[0].intent.trade_id,
                "bmo",
                "Technology",
                "unavailable",
            ),
        ),
        walkforward_run_sha256=model_run.sha256,
        source_predictions=(first_prediction,),
    )
    first_path = tmp_path / "first.json"
    EventTradePlanner.write(first_plan, first_path)
    first_run = VectorbtIntradayEngine().run(
        trades=first_plan.intents,
        sessions=(first_date,),
        initial_cash=first_plan.portfolio.equity,
    )
    second_fixture = _result(
        second_date,
        initial_cash=first_run.final_net_equity,
        pnl=80,
        index=1,
    )
    second_plan = PlannedEventTrades(
        trade_date=second_date,
        portfolio=PortfolioPlan(
            first_run.final_net_equity,
            20,
            0.1,
            0.025,
            (),
            0.01,
            (),
        ),
        intents=(second_fixture.trades[0].intent,),
        cohorts=(
            TradeCohort(
                second_fixture.trades[0].intent.trade_id,
                "amc",
                "Health Care",
                "unavailable",
            ),
        ),
        walkforward_run_sha256=model_run.sha256,
        source_predictions=(second_prediction,),
    )
    second_path = tmp_path / "second.json"
    EventTradePlanner.write(second_plan, second_path)
    strategy_source = tmp_path / "earnings_v1.yaml"
    strategy_source.write_bytes(Path("configs/strategies/earnings_v1.yaml").read_bytes())
    session_source = tmp_path / "sessions.json"
    candidate_source = tmp_path / "candidates.parquet"
    bar_source = tmp_path / "bars.parquet"
    for path, content in (
        (session_source, b"sessions"),
        (candidate_source, b"candidates"),
        (bar_source, b"bars"),
    ):
        path.write_bytes(content)
    assembly = Phase4AssemblyManifest.model_validate(
        {
            "schema_version": 1,
            "strategy_config": _source_reference(strategy_source),
            "walkforward_run_evidence": _source_reference(run_path),
            "session_file": _source_reference(session_source),
            "candidate_files": [_source_reference(candidate_source)],
            "daily_bar_files": [_source_reference(bar_source)],
            "initial_cash": 100_000,
            "minimum_probability": 0.5,
            "execution_price_contract": "adjusted_session_open_to_close",
            "iv_regime_contract": "unavailable",
            "plan_files": [
                _source_reference(first_path),
                _source_reference(second_path),
            ],
        }
    )
    assembly_path = tmp_path / "assembly.json"
    assembly_path.write_bytes(assembly.canonical_bytes)
    spec = tmp_path / "aggregation.json"
    output = tmp_path / "gate.json"
    tearsheet = tmp_path / "gate.html"
    spec.write_text(
        json.dumps(
            {
                "assembly_manifest": assembly_path.name,
                "walkforward_run_evidence": run_path.name,
                "folds": [
                    {
                        "fold_index": 0,
                        "test_start_date": first_date.isoformat(),
                        "test_end_date": first_date.isoformat(),
                        "event_plan_files": [first_path.name],
                    },
                    {
                        "fold_index": 1,
                        "test_start_date": second_date.isoformat(),
                        "test_end_date": second_date.isoformat(),
                        "event_plan_files": [second_path.name],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        app,
        [
            "evaluation",
            "phase4-gate",
            "--aggregation-spec",
            str(spec),
            "--output",
            str(output),
            "--tearsheet-output",
            str(tearsheet),
            "--bootstrap-resamples",
            "10",
        ],
    )

    assert result.exit_code == 2
    assert not output.exists()
    assert not tearsheet.exists()


def test_production_promotion_recomputes_phase4_gate(tmp_path: Path) -> None:
    report = {
        "overall": {
            "net_sharpe": 1.2,
            "max_drawdown": 0.10,
            "bootstrap": {"sharpe": {"lower": 0.6}},
        },
        "walk_forward": {"passes_positive_fold_gate": True},
        "cohorts": _promotion_cohort_rows(),
        "strategy_sha256": "a" * 64,
        "assembly_manifest_sha256": "b" * 64,
        "walkforward_run_sha256": "e" * 64,
        "hyperparameter_study_sha256": "f" * 64,
        "passes_phase4_research_gate": True,
        "passes_pre_paper_backtest_gate": True,
    }
    path = tmp_path / "passing-gate.json"
    encoded = json.dumps(report, sort_keys=True, separators=(",", ":")).encode()
    path.write_bytes(encoded)

    promotion = Phase4PromotionEvidence.load(path)

    assert promotion.report_sha256
    assert promotion.net_sharpe == 1.2
    assert promotion.strategy_sha256 == "a" * 64
    assert promotion.assembly_manifest_sha256 == "b" * 64
    assert promotion.walkforward_run_sha256 == "e" * 64
    assert promotion.hyperparameter_study_sha256 == "f" * 64

    report["overall"]["net_sharpe"] = 0.9
    failing = tmp_path / "inconsistent-gate.json"
    failing.write_bytes(json.dumps(report, sort_keys=True, separators=(",", ":")).encode())
    with pytest.raises(ValueError, match="pre-paper gate verdict is inconsistent"):
        Phase4PromotionEvidence.load(failing)


def test_production_promotion_rejects_missing_cohort_evidence(tmp_path: Path) -> None:
    report = {
        "overall": {
            "net_sharpe": 1.2,
            "max_drawdown": 0.10,
            "bootstrap": {"sharpe": {"lower": 0.6}},
        },
        "walk_forward": {"passes_positive_fold_gate": True},
        "cohorts": [],
        "strategy_sha256": "a" * 64,
        "assembly_manifest_sha256": "b" * 64,
        "walkforward_run_sha256": "e" * 64,
        "hyperparameter_study_sha256": "f" * 64,
        "passes_phase4_research_gate": True,
        "passes_pre_paper_backtest_gate": True,
    }
    path = tmp_path / "missing-cohorts.json"
    path.write_bytes(json.dumps(report, sort_keys=True, separators=(",", ":")).encode())

    with pytest.raises(ValueError, match="invalid Phase 4 gate report"):
        Phase4PromotionEvidence.load(path)
