"""Operational CLI for ingestion, universe snapshots, and readiness evidence."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import httpx
import typer
from pydantic import ValidationError

from quant_earning_edge import __version__
from quant_earning_edge.backtest import (
    BacktestSpec,
    NbboReplayEvidence,
    NbboReplaySpec,
    VectorbtBacktestEngine,
    VectorbtIntradayEngine,
    WalkForwardConfig,
    WalkForwardPlanner,
    replay_order,
)
from quant_earning_edge.data import (
    BarBackfillJob,
    BarBackfillStore,
    BarCoverageAuditor,
    BarsIngestor,
    BronzeWriter,
    CorporateActionsIngestor,
    DuckDBStore,
    EarningsIngestor,
    LakehouseLayout,
    MarketEventsIngestor,
    SessionFileStore,
    SilverDataset,
    SilverWriter,
)
from quant_earning_edge.data.clients import AlpacaCalendarClient, FinnhubClient, PolygonClient
from quant_earning_edge.evaluation import (
    FoldBacktestResults,
    HtmlTearsheetWriter,
    PerformanceEvaluator,
    Phase4AggregationSpec,
    Phase4GateEvaluator,
    ReplaySessionAggregationSpec,
    ReplaySessionAggregator,
)
from quant_earning_edge.features import (
    DailyBarsFeatureLoader,
    EarningsFeatureLoader,
    FeatureEngine,
    FeatureStore,
    PremarketFeatureLoader,
)
from quant_earning_edge.labels import (
    ForwardLabelMaker,
    LabelBarsLoader,
    LabelStore,
    TrainingDatasetAssembler,
)
from quant_earning_edge.portfolio import (
    FractionalKellyPortfolioConstructor,
    PortfolioConfig,
)
from quant_earning_edge.runtime import (
    RuntimeConfigurationError,
    RuntimeEnvironment,
    load_runtime_environment,
)
from quant_earning_edge.signals import (
    EventTradePlanner,
    EventTradePlanningSpec,
    LightgbmWalkForwardTrainer,
    load_strategy_config,
)
from quant_earning_edge.universe import (
    DailyUniverseJob,
    EventCandidateJob,
    RunTrigger,
    UniverseBuilder,
    UniverseManifestStore,
    UniverseSnapshotWriter,
    evaluate_unattended_readiness,
)
from quant_earning_edge.universe.config import (
    load_halt_snapshot,
    load_universe_job_config,
)

if TYPE_CHECKING:
    from collections.abc import Callable

app = typer.Typer(no_args_is_help=True, help="quant_earning_edge command-line interface.")
ingest_app = typer.Typer(no_args_is_help=True, help="Ingest provider data.")
universe_app = typer.Typer(no_args_is_help=True, help="Build and inspect universes.")
backfill_app = typer.Typer(no_args_is_help=True, help="Plan and resume historical backfills.")
calendar_app = typer.Typer(no_args_is_help=True, help="Fetch authoritative market sessions.")
data_app = typer.Typer(no_args_is_help=True, help="Manage local lake query surfaces.")
features_app = typer.Typer(no_args_is_help=True, help="Compute point-in-time features.")
labels_app = typer.Typer(no_args_is_help=True, help="Materialize forward labels and datasets.")
backtest_app = typer.Typer(no_args_is_help=True, help="Plan and run reproducible backtests.")
model_app = typer.Typer(no_args_is_help=True, help="Train deterministic signal models.")
evaluation_app = typer.Typer(no_args_is_help=True, help="Aggregate strategy gate evidence.")
app.add_typer(ingest_app, name="ingest")
app.add_typer(universe_app, name="universe")
app.add_typer(backfill_app, name="backfill")
app.add_typer(calendar_app, name="calendar")
app.add_typer(data_app, name="data")
app.add_typer(features_app, name="features")
app.add_typer(labels_app, name="labels")
app.add_typer(backtest_app, name="backtest")
app.add_typer(model_app, name="model")
app.add_typer(evaluation_app, name="evaluation")

EnvFileOption = Annotated[
    Path | None,
    typer.Option(
        "--env-file",
        help="Optional dotenv file. Process environment variables take precedence.",
    ),
]


@app.command()
def version() -> None:
    """Print the installed package version."""
    typer.echo(__version__)


@calendar_app.command("sessions")
def calendar_sessions(
    start: Annotated[str, typer.Option(help="Inclusive calendar start (YYYY-MM-DD).")],
    end: Annotated[str, typer.Option(help="Inclusive calendar end (YYYY-MM-DD).")],
    env_file: EnvFileOption = None,
) -> None:
    """Fetch Alpaca's trading calendar into an immutable session file."""
    environment = _environment(env_file)
    try:
        api_key_id, secret_key = environment.require_alpaca_credentials()
    except RuntimeConfigurationError as error:
        raise typer.BadParameter(str(error), param_hint="environment") from error
    start_date = _parse_date(start, option="--start")
    end_date = _parse_date(end, option="--end")
    layout = LakehouseLayout(environment.data_lake_root)
    with httpx.Client(
        base_url=environment.alpaca_trading_base_url,
        timeout=environment.http_timeout_seconds,
    ) as http_client:
        sessions = AlpacaCalendarClient(
            api_key_id=api_key_id,
            secret_key=secret_key,
            http_client=http_client,
            bronze_writer=BronzeWriter(layout),
        ).sessions(start_date=start_date, end_date=end_date)
    artifact = SessionFileStore(layout).write(sessions)
    _echo_json(
        {
            "path": str(artifact.path),
            "sha256": artifact.sha256,
            "session_count": len(artifact.sessions),
            "first_session": artifact.sessions[0].session_date,
            "last_session": artifact.sessions[-1].session_date,
        }
    )


@data_app.command("register-views")
def register_views(
    database: Annotated[
        Path,
        typer.Option(help="Persistent DuckDB database to create or update."),
    ],
    datasets: Annotated[
        list[SilverDataset] | None,
        typer.Option(
            "--dataset",
            help="Required silver dataset; repeat as needed. Defaults to all datasets.",
        ),
    ] = None,
    env_file: EnvFileOption = None,
) -> None:
    """Validate silver schemas and register stable DuckDB views."""
    environment = _environment(env_file)
    selected = tuple(datasets) if datasets else tuple(SilverDataset)
    with DuckDBStore(database) as store:
        views = store.register_silver_views(
            LakehouseLayout(environment.data_lake_root),
            datasets=selected,
        )
    _echo_json({"database": str(database.resolve()), "views": views})


@features_app.command("compute")
def compute_features(  # noqa: PLR0917 - CLI options are the PIT compute contract.
    asof_date: Annotated[str, typer.Option(help="Last observable session (YYYY-MM-DD).")],
    observed_at: Annotated[
        str,
        typer.Option(help="Offset-aware cutoff for silver ingestion revisions."),
    ],
    bars_files: Annotated[
        list[Path],
        typer.Option(
            "--bars-file",
            exists=True,
            dir_okay=False,
            help="Silver daily-bars Parquet; repeat for all required partitions.",
        ),
    ],
    symbols: Annotated[
        list[str],
        typer.Option("--symbol", help="Ticker to compute; repeat for multiple symbols."),
    ],
    feature_names: Annotated[
        list[str],
        typer.Option(
            "--feature",
            help="Registered price feature; repeat as needed.",
        ),
    ],
    candidate_files: Annotated[
        list[Path] | None,
        typer.Option(
            "--candidate-file",
            exists=True,
            dir_okay=False,
            help="Gold event-candidate Parquet; required for event features.",
        ),
    ] = None,
    minute_files: Annotated[
        list[Path] | None,
        typer.Option(
            "--minute-file",
            exists=True,
            dir_okay=False,
            help="Silver target-date minute bars; required for premarket gap.",
        ),
    ] = None,
    earnings_files: Annotated[
        list[Path] | None,
        typer.Option(
            "--earnings-file",
            exists=True,
            dir_okay=False,
            help="Silver earnings Parquet history; required for event features.",
        ),
    ] = None,
    target_date: Annotated[
        str | None,
        typer.Option(
            help="Next trading session for event features (YYYY-MM-DD).",
        ),
    ] = None,
    feature_group: Annotated[
        str,
        typer.Option(help="Gold feature-group partition name."),
    ] = "price",
    env_file: EnvFileOption = None,
) -> None:
    """Compute registered causal features and persist lineage."""
    environment = _environment(env_file)
    cutoff = _parse_datetime(observed_at, option="--observed-at")
    contexts = DailyBarsFeatureLoader().load(
        bars_files,
        symbols=symbols,
        asof_date=_parse_date(asof_date, option="--asof-date"),
        observed_at=cutoff,
    )
    if (candidate_files is None) != (earnings_files is None):
        raise typer.BadParameter(
            "--candidate-file and --earnings-file must be supplied together",
            param_hint="event feature inputs",
        )
    if minute_files is not None:
        if target_date is None:
            raise typer.BadParameter(
                "--target-date is required with minute-bar inputs",
                param_hint="--target-date",
            )
        contexts = PremarketFeatureLoader().enrich(
            contexts,
            minute_files=minute_files,
            target_date=_parse_date(target_date, option="--target-date"),
            observed_at=cutoff,
        )
    if candidate_files is not None and earnings_files is not None:
        if target_date is None:
            raise typer.BadParameter(
                "--target-date is required with event feature inputs",
                param_hint="--target-date",
            )
        contexts = EarningsFeatureLoader().enrich(
            contexts,
            candidate_files=candidate_files,
            earnings_files=earnings_files,
            observed_at=cutoff,
            target_date=_parse_date(target_date, option="--target-date"),
        )
    values = FeatureEngine().compute(contexts, feature_names=feature_names)
    artifact = FeatureStore(LakehouseLayout(environment.data_lake_root)).write(
        feature_group=feature_group,
        values=values,
        computed_at=cutoff,
    )
    _echo_json(
        {
            "path": str(artifact.path),
            "sha256": artifact.sha256,
            "row_count": artifact.row_count,
            "feature_names": artifact.feature_names,
        }
    )


@labels_app.command("compute")
def compute_labels(  # noqa: PLR0917 - CLI options are the label contract.
    asof_date: Annotated[str, typer.Option(help="Feature as-of session (YYYY-MM-DD).")],
    observed_at: Annotated[
        str,
        typer.Option(help="Offset-aware cutoff after the label horizon completed."),
    ],
    bars_files: Annotated[
        list[Path],
        typer.Option(
            "--bars-file",
            exists=True,
            dir_okay=False,
            help="Silver daily-bars Parquet; repeat for the full horizon.",
        ),
    ],
    symbols: Annotated[
        list[str],
        typer.Option("--symbol", help="Ticker key; repeat for multiple symbols."),
    ],
    session_file: Annotated[
        Path,
        typer.Option(exists=True, dir_okay=False, help="Immutable market-session file."),
    ],
    env_file: EnvFileOption = None,
) -> None:
    """Compute the three documented forward-return labels."""
    environment = _environment(env_file)
    asof = _parse_date(asof_date, option="--asof-date")
    cutoff = _parse_datetime(observed_at, option="--observed-at")
    session_artifact = SessionFileStore.load(session_file)
    sessions = tuple(item.session_date for item in session_artifact.sessions)
    try:
        asof_index = sessions.index(asof)
        horizon_end = sessions[asof_index + 5]
    except (ValueError, IndexError) as error:
        raise typer.BadParameter(
            "session file must include asof_date and five later sessions",
            param_hint="--session-file",
        ) from error
    bars = LabelBarsLoader().load(
        bars_files,
        symbols=symbols,
        start_date=asof,
        end_date=horizon_end,
        observed_at=cutoff,
    )
    labels = ForwardLabelMaker().compute(
        keys=tuple((symbol, asof) for symbol in symbols),
        sessions=sessions,
        bars=bars,
    )
    artifact = LabelStore(LakehouseLayout(environment.data_lake_root)).write(
        labels,
        computed_at=cutoff,
    )
    _echo_json(
        {
            "path": str(artifact.path),
            "sha256": artifact.sha256,
            "row_count": artifact.row_count,
        }
    )


@labels_app.command("assemble")
def assemble_training_dataset(
    feature_files: Annotated[
        list[Path],
        typer.Option(
            "--feature-file",
            exists=True,
            dir_okay=False,
            help="Gold long-form feature artifact; repeat for all feature groups.",
        ),
    ],
    label_files: Annotated[
        list[Path],
        typer.Option(
            "--label-file",
            exists=True,
            dir_okay=False,
            help="Gold forward-label artifact; repeat for all partitions.",
        ),
    ],
    session_file: Annotated[
        Path,
        typer.Option(exists=True, dir_okay=False, help="Immutable market-session file."),
    ],
    assembled_at: Annotated[
        str,
        typer.Option(help="Offset-aware dataset assembly timestamp."),
    ],
    env_file: EnvFileOption = None,
) -> None:
    """Build an exact-key, pre-open-frozen wide training dataset."""
    environment = _environment(env_file)
    artifact = TrainingDatasetAssembler(LakehouseLayout(environment.data_lake_root)).assemble(
        feature_files=feature_files,
        label_files=label_files,
        session_file=session_file,
        assembled_at=_parse_datetime(assembled_at, option="--assembled-at"),
    )
    _echo_json(
        {
            "path": str(artifact.path),
            "manifest_path": str(artifact.manifest_path),
            "sha256": artifact.sha256,
            "row_count": artifact.row_count,
            "feature_names": artifact.feature_names,
        }
    )


@backtest_app.command("plan-splits")
def plan_backtest_splits(  # noqa: PLR0917 - CLI options define the split contract.
    dataset_files: Annotated[
        list[Path],
        typer.Option(
            "--dataset-file",
            exists=True,
            dir_okay=False,
            help="Assembled training Parquet; repeat for multiple partitions.",
        ),
    ],
    output: Annotated[
        Path,
        typer.Option(dir_okay=False, help="Immutable JSON split-manifest path."),
    ],
    minimum_train_sessions: Annotated[
        int,
        typer.Option(min=1, help="Minimum expanding-window training sessions."),
    ],
    test_sessions: Annotated[
        int,
        typer.Option(min=1, help="Consecutive sessions in each test fold."),
    ],
    embargo_sessions: Annotated[
        int,
        typer.Option(min=1, help="Sessions embargoed before every test fold."),
    ] = 5,
    step_sessions: Annotated[
        int | None,
        typer.Option(min=1, help="Fold-start step; defaults to test-session count."),
    ] = None,
) -> None:
    """Build a content-addressed purged expanding walk-forward plan."""
    config = WalkForwardConfig(
        minimum_train_sessions=minimum_train_sessions,
        test_sessions=test_sessions,
        embargo_sessions=embargo_sessions,
        step_sessions=step_sessions,
    )
    planner = WalkForwardPlanner()
    plan = planner.build(dataset_files, config=config)
    planner.write(plan, output)
    _echo_json(
        {
            "output": str(output.resolve()),
            "sha256": plan.sha256,
            "sample_count": plan.sample_count,
            "fold_count": len(plan.folds),
        }
    )


@backtest_app.command("run-ledger")
def run_backtest_ledger(
    spec_file: Annotated[
        Path,
        typer.Option(exists=True, dir_okay=False, help="Validated daily backtest JSON spec."),
    ],
    output: Annotated[
        Path,
        typer.Option(dir_okay=False, help="Immutable standardized evaluation JSON."),
    ],
    tearsheet_output: Annotated[
        Path | None,
        typer.Option(dir_okay=False, help="Optional immutable self-contained HTML tearsheet."),
    ] = None,
    bootstrap_resamples: Annotated[
        int,
        typer.Option(min=1, help="Trade-vector bootstrap resamples."),
    ] = 10_000,
    seed: Annotated[
        int,
        typer.Option(help="Deterministic bootstrap random seed."),
    ] = 20260427,
) -> None:
    """Run vectorbt, reconcile costs, and persist standardized evidence."""
    try:
        spec = BacktestSpec.model_validate_json(spec_file.read_bytes())
        initial_cash, sessions, marks, trades = spec.domain_inputs()
        if any(item.entry_at is not None for item in trades):
            if marks:
                raise ValueError("intraday ledger specs must not supply daily marks")
            result = VectorbtIntradayEngine().run(
                trades=trades,
                sessions=sessions,
                initial_cash=initial_cash,
            )
        else:
            result = VectorbtBacktestEngine().run(
                trades=trades,
                marks=marks,
                sessions=sessions,
                initial_cash=initial_cash,
            )
    except (ValidationError, ValueError, RuntimeError) as error:
        raise typer.BadParameter(str(error), param_hint="--spec-file") from error
    evaluator = PerformanceEvaluator(
        bootstrap_resamples=bootstrap_resamples,
        seed=seed,
    )
    report = evaluator.evaluate(result)
    evaluator.write(report, output)
    if tearsheet_output is not None:
        HtmlTearsheetWriter().write(
            report=report,
            result=result,
            output=tearsheet_output,
        )
    _echo_json(
        {
            "output": str(output.resolve()),
            "sha256": report.sha256,
            "input_sha256": report.input_sha256,
            "trade_count": report.trade_count,
            "session_count": report.session_count,
            "tearsheet_output": (
                str(tearsheet_output.resolve()) if tearsheet_output is not None else None
            ),
        }
    )


@backtest_app.command("replay-nbbo")
def replay_nbbo(
    spec_file: Annotated[
        Path,
        typer.Option(
            exists=True,
            dir_okay=False,
            help="Validated order, decision snapshot, and normalized market events.",
        ),
    ],
    output: Annotated[
        Path,
        typer.Option(dir_okay=False, help="Immutable causal replay evidence JSON."),
    ],
) -> None:
    """Replay one intended order and persist content-addressed execution evidence."""
    try:
        spec = NbboReplaySpec.model_validate_json(spec_file.read_bytes())
        order, snapshot, quotes, trades, config = spec.domain_inputs()
        result = replay_order(
            order,
            decision_snapshot=snapshot,
            quotes=quotes,
            trades=trades,
            config=config,
        )
        evidence = NbboReplayEvidence.build(spec=spec, result=result)
        evidence.write(output)
    except (ValidationError, ValueError, RuntimeError) as error:
        raise typer.BadParameter(str(error), param_hint="--spec-file") from error
    _echo_json(
        {
            "output": str(output.resolve()),
            "sha256": evidence.sha256,
            "input_sha256": evidence.input_sha256,
            "order_id": order.order_id,
            "filled_qty": result.filled_qty,
            "unfilled_qty": result.unfilled_qty,
            "fill_rate": result.fill_rate,
        }
    )


@model_app.command("train-walkforward")
def train_walkforward_model(
    dataset_files: Annotated[
        list[Path],
        typer.Option(
            "--dataset-file",
            exists=True,
            dir_okay=False,
            help="Assembled training Parquet; repeat in the split-plan file set.",
        ),
    ],
    split_plan: Annotated[
        Path,
        typer.Option(exists=True, dir_okay=False, help="Canonical walk-forward JSON plan."),
    ],
    strategy_config: Annotated[
        Path,
        typer.Option(exists=True, dir_okay=False, help="Validated earnings strategy YAML."),
    ],
    output_dir: Annotated[
        Path,
        typer.Option(file_okay=False, help="Immutable model artifact directory."),
    ],
) -> None:
    """Train fold models without exposing OOS rows to fit or early stopping."""
    try:
        config = load_strategy_config(strategy_config)
        plan = WalkForwardPlanner.load(split_plan)
        trainer = LightgbmWalkForwardTrainer(
            feature_names=config.features,
            label_name="forward_1d_close",
            threshold=config.label.threshold,
            seed=config.seed,
            early_stopping_rounds=config.model.early_stopping_rounds,
        )
        run = trainer.run(dataset_files=dataset_files, plan=plan)
        trainer.write(run, output_dir)
    except (KeyError, ValidationError, ValueError, RuntimeError) as error:
        raise typer.BadParameter(str(error), param_hint="model inputs") from error
    _echo_json(
        {
            "output_dir": str(output_dir.resolve()),
            "run_sha256": run.sha256,
            "plan_sha256": run.plan_sha256,
            "fold_count": len(run.folds),
            "prediction_count": sum(len(item.predictions) for item in run.folds),
        }
    )


@model_app.command("plan-event-backtest")
def plan_event_backtest(
    planning_spec: Annotated[
        Path,
        typer.Option(
            exists=True,
            dir_okay=False,
            help="One-session OOS predictions, PIT sizing inputs, and execution evidence.",
        ),
    ],
    strategy_config: Annotated[
        Path,
        typer.Option(exists=True, dir_okay=False, help="Validated earnings strategy YAML."),
    ],
    plan_output: Annotated[
        Path,
        typer.Option(dir_okay=False, help="Immutable event-trade plan JSON."),
    ],
    evaluation_output: Annotated[
        Path,
        typer.Option(dir_okay=False, help="Immutable standardized evaluation JSON."),
    ],
) -> None:
    """Create causal event trades and evaluate their timestamped executions."""
    try:
        config = load_strategy_config(strategy_config)
        spec = EventTradePlanningSpec.model_validate_json(planning_spec.read_bytes())
        equity, predictions, observations, outcomes = spec.domain_inputs()
        caps = config.portfolio.caps
        sizing = config.portfolio.sizing
        planner = EventTradePlanner(
            FractionalKellyPortfolioConstructor(
                PortfolioConfig(
                    top_k=config.portfolio.top_k,
                    kelly_fraction=sizing.kelly_fraction,
                    history_window=sizing.rolling_window_days,
                    minimum_history=min(20, sizing.rolling_window_days),
                    max_position_weight=caps.max_position_pct,
                    max_sector_weight=caps.max_sector_pct,
                    max_gross_weight=caps.max_gross_exposure_pct,
                )
            )
        )
        plan = planner.plan(
            predictions=predictions,
            observations=observations,
            outcomes=outcomes,
            equity=equity,
        )
        if not plan.intents:
            raise ValueError("event plan produced no trades")
        planner.write(plan, plan_output)
        result = VectorbtIntradayEngine().run(
            trades=plan.intents,
            sessions=(plan.trade_date,),
            initial_cash=equity,
        )
        evaluator = PerformanceEvaluator()
        report = evaluator.evaluate(result)
        evaluator.write(report, evaluation_output)
    except (KeyError, ValidationError, ValueError, RuntimeError) as error:
        raise typer.BadParameter(str(error), param_hint="event backtest inputs") from error
    _echo_json(
        {
            "plan_output": str(plan_output.resolve()),
            "plan_sha256": plan.sha256,
            "evaluation_output": str(evaluation_output.resolve()),
            "evaluation_sha256": report.sha256,
            "trade_count": len(plan.intents),
        }
    )


@evaluation_app.command("phase4-gate")
def evaluate_phase4_gate(
    aggregation_spec: Annotated[
        Path,
        typer.Option(
            exists=True,
            dir_okay=False,
            help="Fold mapping to immutable event-trade plan files.",
        ),
    ],
    output: Annotated[
        Path,
        typer.Option(dir_okay=False, help="Immutable combined Phase 4 gate JSON."),
    ],
    bootstrap_resamples: Annotated[
        int,
        typer.Option(min=1, help="Trade bootstrap resamples."),
    ] = 10_000,
) -> None:
    """Replay event plans and evaluate both documented strategy gates."""
    try:
        spec = Phase4AggregationSpec.model_validate_json(aggregation_spec.read_bytes())
        fold_results = []
        for fold in spec.folds:
            results = []
            for configured_path in fold.event_plan_files:
                plan_path = (
                    configured_path
                    if configured_path.is_absolute()
                    else aggregation_spec.parent / configured_path
                )
                plan = EventTradePlanner.load(plan_path)
                results.append(
                    VectorbtIntradayEngine().run(
                        trades=plan.intents,
                        sessions=(plan.trade_date,),
                        initial_cash=plan.portfolio.equity,
                    )
                )
            fold_results.append(
                FoldBacktestResults(
                    fold_index=fold.fold_index,
                    test_start_date=fold.test_start_date,
                    test_end_date=fold.test_end_date,
                    results=tuple(results),
                )
            )
        evaluator = Phase4GateEvaluator(
            bootstrap_resamples=bootstrap_resamples,
        )
        report = evaluator.evaluate(tuple(fold_results))
        evaluator.write(report, output)
    except (KeyError, ValidationError, ValueError, RuntimeError) as error:
        raise typer.BadParameter(str(error), param_hint="Phase 4 aggregation") from error
    _echo_json(
        {
            "output": str(output.resolve()),
            "sha256": report.sha256,
            "trade_count": report.overall.trade_count,
            "fold_count": len(report.walk_forward.folds),
            "passes_phase4_research_gate": report.passes_phase4_research_gate,
            "passes_pre_paper_backtest_gate": report.passes_pre_paper_backtest_gate,
        }
    )


@evaluation_app.command("replay-session")
def aggregate_replay_session(
    aggregation_spec: Annotated[
        Path,
        typer.Option(
            exists=True,
            dir_okay=False,
            help="Session, replay-evidence paths, and round-trip lifecycle mapping.",
        ),
    ],
    output: Annotated[
        Path,
        typer.Option(dir_okay=False, help="Immutable daily Phase 6 replay report."),
    ],
) -> None:
    """Reconcile entry/exit fills and aggregate one daily replay proof record."""
    try:
        spec = ReplaySessionAggregationSpec.model_validate_json(aggregation_spec.read_bytes())
        evidence = tuple(
            NbboReplayEvidence.load(
                configured_path
                if configured_path.is_absolute()
                else aggregation_spec.parent / configured_path
            )
            for configured_path in spec.evidence_files
        )
        aggregator = ReplaySessionAggregator()
        report = aggregator.evaluate(
            evidence=evidence,
            round_trips=tuple(item.to_domain() for item in spec.round_trips),
            session_date=spec.session_date,
            initial_cash=spec.initial_cash,
            commission_bps_per_side=spec.commission_bps_per_side,
        )
        report.write(output)
    except (OSError, ValidationError, ValueError, RuntimeError) as error:
        raise typer.BadParameter(str(error), param_hint="replay-session inputs") from error
    _echo_json(
        {
            "output": str(output.resolve()),
            "sha256": report.sha256,
            "session_date": report.session_date,
            "order_count": report.intended_order_count,
            "share_fill_rate": report.share_fill_rate,
            "reconciliation_break_count": report.reconciliation_break_count,
            "net_pnl": report.net_pnl,
        }
    )


@ingest_app.command("earnings")
def ingest_earnings(
    start: Annotated[str, typer.Option(help="Inclusive start date (YYYY-MM-DD).")],
    end: Annotated[str, typer.Option(help="Inclusive end date (YYYY-MM-DD).")],
    env_file: EnvFileOption = None,
) -> None:
    """Fetch Finnhub earnings into immutable bronze and silver storage."""
    environment = _environment(env_file)
    api_key = _required_key(environment.require_finnhub_api_key)
    start_date = _parse_date(start, option="--start")
    end_date = _parse_date(end, option="--end")
    layout = LakehouseLayout(environment.data_lake_root)
    with httpx.Client(
        base_url=environment.finnhub_base_url,
        timeout=environment.http_timeout_seconds,
    ) as http_client:
        result = EarningsIngestor(
            client=FinnhubClient(
                api_key=api_key,
                http_client=http_client,
                bronze_writer=BronzeWriter(layout),
            ),
            silver_writer=SilverWriter(layout),
        ).ingest(start_date=start_date, end_date=end_date)
    _echo_json(
        {
            "event_count": result.event_count,
            "silver_artifacts": [str(item.path) for item in result.silver_artifacts],
        }
    )


@ingest_app.command("bars")
def ingest_bars(
    symbol: Annotated[str, typer.Option(help="US-equity ticker.")],
    start: Annotated[str, typer.Option(help="Inclusive start date (YYYY-MM-DD).")],
    end: Annotated[str, typer.Option(help="Inclusive end date (YYYY-MM-DD).")],
    env_file: EnvFileOption = None,
) -> None:
    """Fetch Polygon adjusted daily bars into bronze and silver storage."""
    environment = _environment(env_file)
    api_key = _required_key(environment.require_polygon_api_key)
    start_date = _parse_date(start, option="--start")
    end_date = _parse_date(end, option="--end")
    layout = LakehouseLayout(environment.data_lake_root)
    with httpx.Client(
        base_url=environment.polygon_base_url,
        timeout=environment.http_timeout_seconds,
    ) as http_client:
        result = BarsIngestor(
            client=PolygonClient(
                api_key=api_key,
                http_client=http_client,
                bronze_writer=BronzeWriter(layout),
            ),
            silver_writer=SilverWriter(layout),
        ).ingest(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
        )
    _echo_json(
        {
            "symbol": result.symbol,
            "bar_count": result.bar_count,
            "silver_artifacts": [str(item.path) for item in result.silver_artifacts],
        }
    )


@ingest_app.command("minute-bars")
def ingest_minute_bars(
    symbol: Annotated[str, typer.Option(help="US-equity ticker.")],
    start_at: Annotated[str, typer.Option(help="Offset-aware interval start.")],
    end_at: Annotated[str, typer.Option(help="Offset-aware interval end.")],
    event_date: Annotated[str, typer.Option(help="Market-date partition (YYYY-MM-DD).")],
    env_file: EnvFileOption = None,
) -> None:
    """Fetch adjusted Polygon minute aggregates into bronze and silver."""
    environment = _environment(env_file)
    api_key = _required_key(environment.require_polygon_api_key)
    layout = LakehouseLayout(environment.data_lake_root)
    partition_date = _parse_date(event_date, option="--event-date")
    with httpx.Client(
        base_url=environment.polygon_base_url,
        timeout=environment.http_timeout_seconds,
    ) as http_client:
        bars = PolygonClient(
            api_key=api_key,
            http_client=http_client,
            bronze_writer=BronzeWriter(layout),
        ).minute_bars(
            symbol=symbol,
            start_at=_parse_datetime(start_at, option="--start-at"),
            end_at=_parse_datetime(end_at, option="--end-at"),
        )
    artifact = SilverWriter(layout).write_minute_bars(
        bars,
        event_date=partition_date,
    )
    _echo_json(
        {
            "path": str(artifact.path),
            "sha256": artifact.sha256,
            "row_count": artifact.row_count,
        }
    )


@ingest_app.command("market-events")
def ingest_market_events(
    symbol: Annotated[str, typer.Option(help="US-equity ticker.")],
    start_at: Annotated[str, typer.Option(help="Offset-aware inclusive SIP-time start.")],
    end_at: Annotated[str, typer.Option(help="Offset-aware inclusive SIP-time end.")],
    event_date: Annotated[str, typer.Option(help="Market-date partition (YYYY-MM-DD).")],
    env_file: EnvFileOption = None,
) -> None:
    """Fetch Polygon historical NBBO and trades into bronze and silver."""
    environment = _environment(env_file)
    api_key = _required_key(environment.require_polygon_api_key)
    layout = LakehouseLayout(environment.data_lake_root)
    with httpx.Client(
        base_url=environment.polygon_base_url,
        timeout=environment.http_timeout_seconds,
    ) as http_client:
        result = MarketEventsIngestor(
            client=PolygonClient(
                api_key=api_key,
                http_client=http_client,
                bronze_writer=BronzeWriter(layout),
            ),
            silver_writer=SilverWriter(layout),
        ).ingest(
            symbol=symbol,
            event_date=_parse_date(event_date, option="--event-date"),
            start_at=_parse_datetime(start_at, option="--start-at"),
            end_at=_parse_datetime(end_at, option="--end-at"),
        )
    _echo_json(
        {
            "symbol": result.symbol,
            "event_date": result.event_date,
            "quote_count": result.quote_count,
            "trade_count": result.trade_count,
            "quote_path": str(result.quote_artifact.path),
            "quote_sha256": result.quote_artifact.sha256,
            "trade_path": str(result.trade_artifact.path),
            "trade_sha256": result.trade_artifact.sha256,
        }
    )


@ingest_app.command("corporate-actions")
def ingest_corporate_actions(
    start: Annotated[str, typer.Option(help="Inclusive action date (YYYY-MM-DD).")],
    end: Annotated[str, typer.Option(help="Inclusive action date (YYYY-MM-DD).")],
    env_file: EnvFileOption = None,
) -> None:
    """Fetch Polygon splits and dividends into bronze and silver storage."""
    environment = _environment(env_file)
    api_key = _required_key(environment.require_polygon_api_key)
    start_date = _parse_date(start, option="--start")
    end_date = _parse_date(end, option="--end")
    layout = LakehouseLayout(environment.data_lake_root)
    with httpx.Client(
        base_url=environment.polygon_base_url,
        timeout=environment.http_timeout_seconds,
    ) as http_client:
        result = CorporateActionsIngestor(
            client=PolygonClient(
                api_key=api_key,
                http_client=http_client,
                bronze_writer=BronzeWriter(layout),
            ),
            silver_writer=SilverWriter(layout),
        ).ingest(start_date=start_date, end_date=end_date)
    _echo_json(
        {
            "split_count": result.split_count,
            "dividend_count": result.dividend_count,
            "silver_artifacts": [str(item.path) for item in result.silver_artifacts],
        }
    )


@universe_app.command("build")
def build_universe(  # noqa: PLR0917 - CLI options are an explicit operational contract.
    trade_date: Annotated[
        str,
        typer.Option(help="Intended order date (YYYY-MM-DD)."),
    ],
    asof_date: Annotated[
        str,
        typer.Option(help="Prior market-close date (YYYY-MM-DD)."),
    ],
    lookback_start: Annotated[
        str,
        typer.Option(help="Bar lookback start date (YYYY-MM-DD)."),
    ],
    halt_snapshot_file: Annotated[
        Path,
        typer.Option(
            exists=True,
            dir_okay=False,
            help="Timestamped JSON halt snapshot, including an explicit empty set.",
        ),
    ],
    config_file: Annotated[
        Path,
        typer.Option(
            exists=True,
            dir_okay=False,
            help="Universe job YAML configuration.",
        ),
    ] = Path("configs/universe/default.yaml"),
    trigger: Annotated[
        RunTrigger,
        typer.Option(help="Invocation source recorded in the run manifest."),
    ] = RunTrigger.MANUAL,
    env_file: EnvFileOption = None,
) -> None:
    """Build a frozen next-session universe from strict prior-close inputs."""
    environment = _environment(env_file)
    api_key = _required_key(environment.require_polygon_api_key)
    job_config = load_universe_job_config(config_file)
    halt_snapshot = load_halt_snapshot(halt_snapshot_file)
    layout = LakehouseLayout(environment.data_lake_root)
    with httpx.Client(
        base_url=environment.polygon_base_url,
        timeout=environment.http_timeout_seconds,
    ) as http_client:
        result = DailyUniverseJob(
            market_data=PolygonClient(
                api_key=api_key,
                http_client=http_client,
                bronze_writer=BronzeWriter(layout),
            ),
            builder=UniverseBuilder(job_config.eligibility.to_domain()),
            snapshot_writer=UniverseSnapshotWriter(layout),
            manifest_store=UniverseManifestStore(layout),
            adv_sessions=job_config.adv_sessions,
        ).run(
            trade_date=_parse_date(trade_date, option="--trade-date"),
            asof_date=_parse_date(asof_date, option="--asof-date"),
            lookback_start=_parse_date(
                lookback_start,
                option="--lookback-start",
            ),
            halt_snapshot=halt_snapshot,
            trigger=trigger,
        )
    _echo_json(
        {
            "run_id": result.manifest.run_id,
            "status": result.manifest.status,
            "reference_count": result.manifest.reference_count,
            "candidate_count": result.manifest.candidate_count,
            "eligible_count": result.manifest.eligible_count,
            "snapshot_path": str(result.snapshot.path),
            "snapshot_sha256": result.snapshot.sha256,
            "manifest_path": str(result.manifest_path),
        }
    )


@universe_app.command("readiness")
def universe_readiness(
    trade_dates: Annotated[
        list[str],
        typer.Option(
            "--trade-date",
            help="Expected real market date; repeat once per required session.",
        ),
    ],
    env_file: EnvFileOption = None,
) -> None:
    """Evaluate persisted scheduled-run evidence for explicit market dates."""
    environment = _environment(env_file)
    store = UniverseManifestStore(LakehouseLayout(environment.data_lake_root))
    expected = tuple(_parse_date(value, option="--trade-date") for value in trade_dates)
    evidence = evaluate_unattended_readiness(
        store.read_all(),
        expected_trade_dates=expected,
    )
    _echo_json(
        {
            "ready": evidence.ready,
            "expected_trade_dates": evidence.expected_trade_dates,
            "successful_trade_dates": evidence.successful_trade_dates,
        }
    )
    if not evidence.ready:
        raise typer.Exit(code=1)


@universe_app.command("events")
def universe_events(  # noqa: PLR0917 - CLI options are the event-join contract.
    trade_date: Annotated[str, typer.Option(help="Intended order date (YYYY-MM-DD).")],
    decision_at: Annotated[
        str,
        typer.Option(help="Point-in-time cutoff as an offset-aware ISO-8601 timestamp."),
    ],
    universe_snapshot: Annotated[
        Path,
        typer.Option(exists=True, dir_okay=False, help="Frozen universe Parquet artifact."),
    ],
    session_file: Annotated[
        Path,
        typer.Option(exists=True, dir_okay=False, help="Immutable Alpaca session JSON file."),
    ],
    earnings_files: Annotated[
        list[Path],
        typer.Option(
            "--earnings-file",
            exists=True,
            dir_okay=False,
            help="Silver earnings Parquet; repeat for every relevant partition/revision.",
        ),
    ],
    split_files: Annotated[
        list[Path],
        typer.Option(
            "--split-file",
            exists=True,
            dir_okay=False,
            help="Silver split Parquet; repeat for every audited partition/revision.",
        ),
    ],
    dividend_files: Annotated[
        list[Path],
        typer.Option(
            "--dividend-file",
            exists=True,
            dir_okay=False,
            help="Silver dividend Parquet; repeat for every audited partition/revision.",
        ),
    ],
    env_file: EnvFileOption = None,
) -> None:
    """Join scheduled earnings to a frozen eligible universe without lookahead."""
    environment = _environment(env_file)
    artifact = EventCandidateJob(LakehouseLayout(environment.data_lake_root)).run(
        trade_date=_parse_date(trade_date, option="--trade-date"),
        decision_at=_parse_datetime(decision_at, option="--decision-at"),
        universe_snapshot=universe_snapshot,
        session_file=session_file,
        earnings_files=earnings_files,
        split_files=split_files,
        dividend_files=dividend_files,
    )
    _echo_json(
        {
            "path": str(artifact.path),
            "manifest_path": str(artifact.manifest_path),
            "sha256": artifact.sha256,
            "candidate_count": artifact.row_count,
            "excluded_counts": artifact.excluded_counts,
        }
    )


@backfill_app.command("symbols")
def backfill_symbols(
    asof_dates: Annotated[
        list[str],
        typer.Option(
            "--asof-date",
            help="Historical reference date; repeat to build a survivorship-aware union.",
        ),
    ],
    output: Annotated[
        Path,
        typer.Option(help="New JSON file to create; existing files are never replaced."),
    ],
    env_file: EnvFileOption = None,
) -> None:
    """Create a dated union of active historical Polygon ticker symbols."""
    environment = _environment(env_file)
    api_key = _required_key(environment.require_polygon_api_key)
    dates = tuple(_parse_date(item, option="--asof-date") for item in asof_dates)
    if not dates:
        raise typer.BadParameter(
            "at least one --asof-date is required",
            param_hint="--asof-date",
        )
    layout = LakehouseLayout(environment.data_lake_root)
    symbols: set[str] = set()
    with httpx.Client(
        base_url=environment.polygon_base_url,
        timeout=environment.http_timeout_seconds,
    ) as http_client:
        client = PolygonClient(
            api_key=api_key,
            http_client=http_client,
            bronze_writer=BronzeWriter(layout),
        )
        for asof_date in dates:
            symbols.update(
                reference.symbol for reference in client.list_tickers(asof_date=asof_date)
            )
    payload = {
        "asof_dates": dates,
        "generated_at": datetime.now(UTC),
        "symbols": sorted(symbols),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as destination:
        json.dump(
            payload,
            destination,
            default=lambda item: item.isoformat(),
            sort_keys=True,
            separators=(",", ":"),
        )
    _echo_json({"output": str(output), "symbol_count": len(symbols)})


@backfill_app.command("bars")
def backfill_bars(  # noqa: PLR0917 - CLI options are the backfill contract.
    symbols_file: Annotated[
        Path,
        typer.Option(exists=True, dir_okay=False, help="JSON or line-delimited symbols."),
    ],
    start: Annotated[str, typer.Option(help="Inclusive start date (YYYY-MM-DD).")],
    end: Annotated[str, typer.Option(help="Inclusive end date (YYYY-MM-DD).")],
    batch_size: Annotated[
        int,
        typer.Option(min=1, help="Symbols fetched before one partitioned silver write."),
    ] = 100,
    continue_on_error: Annotated[
        bool,
        typer.Option(help="Record a failed batch and continue with later batches."),
    ] = False,
    env_file: EnvFileOption = None,
) -> None:
    """Create/resume a deterministic multi-symbol adjusted-bars backfill."""
    environment = _environment(env_file)
    api_key = _required_key(environment.require_polygon_api_key)
    layout = LakehouseLayout(environment.data_lake_root)
    store = BarBackfillStore(layout)
    plan = store.prepare_plan(
        symbols=_load_string_list(symbols_file, key="symbols"),
        start_date=_parse_date(start, option="--start"),
        end_date=_parse_date(end, option="--end"),
        batch_size=batch_size,
    )
    with httpx.Client(
        base_url=environment.polygon_base_url,
        timeout=environment.http_timeout_seconds,
    ) as http_client:
        result = BarBackfillJob(
            provider=PolygonClient(
                api_key=api_key,
                http_client=http_client,
                bronze_writer=BronzeWriter(layout),
            ),
            silver_writer=SilverWriter(layout),
            store=store,
        ).run(plan, continue_on_error=continue_on_error)
    _echo_json(
        {
            "plan_id": plan.plan_id,
            "batch_count": len(plan.batches),
            "completed_batch_indices": result.completed_batch_indices,
            "skipped_batch_indices": result.skipped_batch_indices,
            "failed_batch_indices": result.failed_batch_indices,
        }
    )
    if result.failed_batch_indices:
        raise typer.Exit(code=1)


@backfill_app.command("coverage")
def backfill_coverage(
    plan_id: Annotated[str, typer.Option(help="SHA-256 plan identifier.")],
    sessions_file: Annotated[
        Path,
        typer.Option(
            exists=True,
            dir_okay=False,
            help="JSON or line-delimited authoritative market sessions.",
        ),
    ],
    env_file: EnvFileOption = None,
) -> None:
    """Audit a backfill against explicit market sessions and persist evidence."""
    environment = _environment(env_file)
    layout = LakehouseLayout(environment.data_lake_root)
    store = BarBackfillStore(layout)
    plan = store.load_plan(plan_id)
    sessions = tuple(
        _parse_date(item, option="sessions file")
        for item in _load_string_list(sessions_file, key="sessions")
    )
    report = BarCoverageAuditor(layout=layout, store=store).audit(
        plan,
        expected_sessions=sessions,
    )
    _echo_json(
        {
            "plan_id": report.plan_id,
            "ready": report.ready,
            "expected_session_count": len(report.expected_sessions),
            "complete_symbol_count": len(report.complete_symbols),
            "missing_symbol_count": len(report.missing_sessions_by_symbol),
            "missing_session_count": sum(
                len(items) for items in report.missing_sessions_by_symbol.values()
            ),
            "covers_minimum_five_years": report.covers_minimum_five_years,
            "completed_batch_indices": report.completed_batch_indices,
        }
    )
    if not report.ready:
        raise typer.Exit(code=1)


def _environment(env_file: Path | None) -> RuntimeEnvironment:
    try:
        return load_runtime_environment(env_file=env_file)
    except RuntimeConfigurationError as error:
        raise typer.BadParameter(str(error), param_hint="environment") from error


def _required_key(getter: Callable[[], str]) -> str:
    try:
        return getter()
    except RuntimeConfigurationError as error:
        raise typer.BadParameter(str(error), param_hint="environment") from error


def _parse_date(value: str, *, option: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as error:
        raise typer.BadParameter(
            f"{option} must use YYYY-MM-DD",
            param_hint=option,
        ) from error


def _parse_datetime(value: str, *, option: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise typer.BadParameter(
            f"{option} must be an ISO-8601 timestamp",
            param_hint=option,
        ) from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise typer.BadParameter(
            f"{option} must include a UTC offset",
            param_hint=option,
        )
    return parsed


def _load_string_list(path: Path, *, key: str) -> tuple[str, ...]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        raw = json.loads(text)
        if not isinstance(raw, dict) or not isinstance(raw.get(key), list):
            raise typer.BadParameter(
                f"JSON file must contain a {key!r} list",
                param_hint=str(path),
            )
        values = raw[key]
    else:
        values = [
            line.strip()
            for line in text.splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
    if not all(isinstance(item, str) for item in values):
        raise typer.BadParameter(
            f"{key!r} values must all be strings",
            param_hint=str(path),
        )
    return tuple(str(item) for item in values)


def _echo_json(value: object) -> None:
    typer.echo(
        json.dumps(
            value,
            default=lambda item: item.isoformat() if hasattr(item, "isoformat") else str(item),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    app()
