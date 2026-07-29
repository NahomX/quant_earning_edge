"""Causal OOS prediction-to-event-trade tests."""

from __future__ import annotations

import json
from dataclasses import asdict, replace
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

from typer.testing import CliRunner

from quant_earning_edge.backtest import VectorbtIntradayEngine
from quant_earning_edge.cli import app
from quant_earning_edge.evaluation import PerformanceEvaluator
from quant_earning_edge.portfolio import (
    FractionalKellyPortfolioConstructor,
    PortfolioConfig,
    TradeOutcome,
)
from quant_earning_edge.signals import (
    EventExecutionObservation,
    EventTradePlanner,
    FeatureAttribution,
    FoldModelResult,
    LightgbmHyperparameters,
    OosPrediction,
    WalkForwardModelRun,
    load_strategy_config,
)


def _outcomes() -> tuple[TradeOutcome, ...]:
    first = date(2025, 1, 1)
    return tuple(
        TradeOutcome(
            closed_date=first + timedelta(days=index),
            net_return=0.04 if index % 2 == 0 else -0.01,
        )
        for index in range(20)
    )


def _inputs() -> tuple[
    tuple[OosPrediction, ...],
    tuple[EventExecutionObservation, ...],
]:
    asof = date(2025, 2, 2)
    trade_date = date(2025, 2, 3)
    decision = datetime(2025, 2, 3, 2, 30, tzinfo=UTC)
    predictions = tuple(
        OosPrediction(
            row_index=index,
            symbol=symbol,
            asof_date=asof,
            probability_up=probability,
            realized_label=index % 2,
        )
        for index, (symbol, probability) in enumerate((("AAA", 0.80), ("BBB", 0.65), ("CCC", 0.40)))
    )
    observations = tuple(
        EventExecutionObservation(
            row_index=index,
            symbol=prediction.symbol,
            sector="Technology" if index < 2 else "Health Care",
            asof_date=asof,
            trade_date=trade_date,
            decision_at=decision,
            sizing_price_observed_at=decision - timedelta(minutes=30),
            sizing_price=100.0 + index * 10,
            entry_at=datetime(2025, 2, 3, 14, 30, tzinfo=UTC),
            entry_price=101.0 + index * 10,
            exit_at=datetime(2025, 2, 3, 21, 0, tzinfo=UTC),
            exit_price=103.0 + index * 10,
            frozen_average_daily_volume_shares=1_000_000,
            event_timing="bmo" if index % 2 == 0 else "amc",
        )
        for index, prediction in enumerate(predictions)
    )
    return predictions, observations


def _write_run_evidence(path: Path, predictions: tuple[OosPrediction, ...]) -> WalkForwardModelRun:
    feature_names = load_strategy_config(Path("configs/strategies/earnings_v1.yaml")).features
    hyperparameters = LightgbmHyperparameters()
    run = WalkForwardModelRun(
        plan_sha256="a" * 64,
        dataset_sha256=("b" * 64,),
        feature_names=feature_names,
        label_name="forward_1d_close",
        threshold=0.0,
        seed=20260427,
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
                feature_attribution=tuple(FeatureAttribution(name, 0.0) for name in feature_names),
            ),
        ),
    )
    path.write_bytes(run.evidence_json_bytes())
    return run


def test_event_trade_plan_is_deterministic_and_long_only() -> None:
    predictions, observations = _inputs()
    planner = EventTradePlanner(
        FractionalKellyPortfolioConstructor(PortfolioConfig(top_k=5, minimum_history=20))
    )

    first = planner.plan(
        predictions=predictions,
        observations=observations,
        outcomes=_outcomes(),
        equity=100_000,
        walkforward_run_sha256="f" * 64,
    )
    second = planner.plan(
        predictions=predictions,
        observations=observations,
        outcomes=_outcomes(),
        equity=100_000,
        walkforward_run_sha256="f" * 64,
    )

    assert first == second
    assert [item.symbol for item in first.intents] == ["AAA", "BBB"]
    assert all(item.side == "long" for item in first.intents)
    assert all(item.entry_date == item.exit_date for item in first.intents)
    assert first.portfolio.gross_weight <= 0.50
    assert tuple(item.event_timing for item in first.cohorts) == ("bmo", "amc")
    assert all(item.iv_regime == "unavailable" for item in first.cohorts)


def test_future_labels_and_exit_prices_cannot_change_selection_or_size() -> None:
    predictions, observations = _inputs()
    planner = EventTradePlanner(
        FractionalKellyPortfolioConstructor(PortfolioConfig(top_k=5, minimum_history=20))
    )
    baseline = planner.plan(
        predictions=predictions,
        observations=observations,
        outcomes=_outcomes(),
        equity=100_000,
        walkforward_run_sha256="f" * 64,
    )
    changed_predictions = tuple(
        replace(item, realized_label=1 - item.realized_label) for item in predictions
    )
    changed_observations = tuple(
        replace(item, exit_price=item.exit_price * 10) for item in observations
    )

    changed = planner.plan(
        predictions=changed_predictions,
        observations=changed_observations,
        outcomes=_outcomes(),
        equity=100_000,
        walkforward_run_sha256="f" * 64,
    )

    assert baseline.portfolio == changed.portfolio
    assert tuple((item.symbol, item.shares) for item in baseline.intents) == tuple(
        (item.symbol, item.shares) for item in changed.intents
    )


def test_planned_trades_persist_and_run_through_evaluation(tmp_path: Path) -> None:
    predictions, observations = _inputs()
    planner = EventTradePlanner(
        FractionalKellyPortfolioConstructor(PortfolioConfig(top_k=5, minimum_history=20))
    )
    plan = planner.plan(
        predictions=predictions,
        observations=observations,
        outcomes=_outcomes(),
        equity=100_000,
        walkforward_run_sha256="f" * 64,
    )
    artifact = tmp_path / "event-trades.json"

    planner.write(plan, artifact)
    planner.write(plan, artifact)
    result = VectorbtIntradayEngine().run(
        trades=plan.intents,
        sessions=(plan.trade_date,),
        initial_cash=plan.portfolio.equity,
    )
    report = PerformanceEvaluator(bootstrap_resamples=10).evaluate(result)

    assert artifact.exists()
    assert len(plan.sha256) == 64
    assert report.trade_count == len(plan.intents)
    assert report.final_net_equity < report.final_gross_equity


def test_event_backtest_cli_writes_plan_and_evaluation(tmp_path: Path) -> None:
    predictions, observations = _inputs()
    spec = tmp_path / "planning.json"
    plan_output = tmp_path / "plan.json"
    evaluation_output = tmp_path / "evaluation.json"
    run_evidence = tmp_path / "walkforward-run.json"
    _write_run_evidence(run_evidence, predictions)
    spec.write_text(
        json.dumps(
            {
                "equity": 100_000,
                "predictions": [asdict(item) for item in predictions],
                "observations": [asdict(item) for item in observations],
                "outcomes": [asdict(item) for item in _outcomes()],
            },
            default=lambda item: item.isoformat(),
        ),
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        app,
        [
            "model",
            "plan-event-backtest",
            "--planning-spec",
            str(spec),
            "--strategy-config",
            "configs/strategies/earnings_v1.yaml",
            "--plan-output",
            str(plan_output),
            "--evaluation-output",
            str(evaluation_output),
            "--walkforward-run-evidence",
            str(run_evidence),
        ],
    )

    assert result.exit_code == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["trade_count"] == 2
    assert plan_output.exists()
    assert evaluation_output.exists()


def test_event_backtest_cli_persists_model_abstention_session(tmp_path: Path) -> None:
    predictions, observations = _inputs()
    abstentions = tuple(replace(item, probability_up=0.4) for item in predictions)
    spec = tmp_path / "planning.json"
    plan_output = tmp_path / "plan.json"
    evaluation_output = tmp_path / "evaluation.json"
    run_evidence = tmp_path / "walkforward-run.json"
    _write_run_evidence(run_evidence, abstentions)
    spec.write_text(
        json.dumps(
            {
                "equity": 100_000,
                "predictions": [asdict(item) for item in abstentions],
                "observations": [asdict(item) for item in observations],
                "outcomes": [asdict(item) for item in _outcomes()],
            },
            default=lambda item: item.isoformat(),
        ),
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        app,
        [
            "model",
            "plan-event-backtest",
            "--planning-spec",
            str(spec),
            "--strategy-config",
            "configs/strategies/earnings_v1.yaml",
            "--plan-output",
            str(plan_output),
            "--evaluation-output",
            str(evaluation_output),
            "--walkforward-run-evidence",
            str(run_evidence),
        ],
    )

    assert result.exit_code == 0, result.stderr
    payload = json.loads(result.stdout)
    report = json.loads(evaluation_output.read_bytes())
    assert payload["trade_count"] == 0
    assert report["session_count"] == 1
    assert report["hit_rate"] == 0.0
