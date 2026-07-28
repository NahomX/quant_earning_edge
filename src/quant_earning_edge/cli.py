"""Operational CLI for ingestion, universe snapshots, and readiness evidence."""

from __future__ import annotations

import json
import time
from datetime import UTC, date, datetime, timedelta
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
    ReplayConfigSpec,
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
    FrozenMarketEventsIngestor,
    FrozenMarketEventsManifest,
    LakehouseLayout,
    MarketEventsIngestor,
    ReplayManifestRunner,
    ReplayMaterializationManifest,
    ReplayMaterializationSpec,
    ReplaySpecMaterializer,
    SessionFileStore,
    SilverDataset,
    SilverWriter,
    replay_sources_from_files,
)
from quant_earning_edge.data.clients import AlpacaCalendarClient, FinnhubClient, PolygonClient
from quant_earning_edge.evaluation import (
    FoldBacktestResults,
    HtmlTearsheetWriter,
    PerformanceEvaluator,
    Phase4AggregationSpec,
    Phase4GateEvaluator,
    Phase6AggregationSpec,
    Phase6ControlBuilder,
    Phase6GateEvaluator,
    ReplaySessionAggregationSpec,
    ReplaySessionAggregator,
    ReplaySessionReport,
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
from quant_earning_edge.live import (
    AlpacaPaperClient,
    PaperBatchSubmitter,
    PaperOrderBatchSpec,
    PaperOrderReconciler,
    PaperOrderRequest,
    PaperReconciliationSpec,
)
from quant_earning_edge.monitoring import (
    CircuitBreakerControlBuilder,
    CircuitBreakerDecision,
    CircuitBreakerEvaluationSpec,
    CircuitBreakerEvaluator,
    CompletedReplayControlSource,
)
from quant_earning_edge.orchestration import (
    DailyWorkflowRunner,
    DailyWorkflowSpecGenerator,
    DailyWorkflowState,
    DailyWorkflowStore,
    StageStatus,
    WorkflowHealthEvaluator,
    WorkflowHealthReport,
    WorkflowInboxWorker,
    WorkflowRunSpec,
    WorkflowTrigger,
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
    DailyOrderPlanningSpec,
    EventTradePlanner,
    EventTradePlanningSpec,
    FrozenDailyOrders,
    LightgbmWalkForwardTrainer,
    LiveOrderPlanner,
    load_strategy_config,
    strategy_file_sha256,
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
monitoring_app = typer.Typer(no_args_is_help=True, help="Evaluate operational safety gates.")
paper_app = typer.Typer(no_args_is_help=True, help="Operate the isolated Alpaca paper account.")
workflow_app = typer.Typer(no_args_is_help=True, help="Inspect restart-safe daily workflow state.")
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
app.add_typer(monitoring_app, name="monitoring")
app.add_typer(paper_app, name="paper")
app.add_typer(workflow_app, name="workflow")

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


@backtest_app.command("materialize-replay-specs")
def materialize_replay_specs(
    materialization_spec: Annotated[
        Path,
        typer.Option(
            exists=True,
            dir_okay=False,
            help="Frozen orders, decision snapshots, and silver event sources.",
        ),
    ],
    output_dir: Annotated[
        Path,
        typer.Option(file_okay=False, help="Directory for canonical per-order replay specs."),
    ],
    manifest_output: Annotated[
        Path,
        typer.Option(dir_okay=False, help="Immutable source/filtering audit manifest."),
    ],
) -> None:
    """Bridge frozen orders and silver events into self-contained causal replay specs."""
    try:
        spec = ReplayMaterializationSpec.model_validate_json(
            materialization_spec.read_bytes()
        ).resolve_paths(materialization_spec.parent)
        manifest = ReplaySpecMaterializer().materialize(
            spec,
            output_dir=output_dir,
            manifest_output=manifest_output,
        )
    except (OSError, ValidationError, ValueError, RuntimeError) as error:
        raise typer.BadParameter(str(error), param_hint="replay materialization inputs") from error
    _echo_json(
        {
            "manifest_path": str(manifest_output.resolve()),
            "manifest_sha256": manifest.sha256,
            "input_sha256": manifest.input_sha256,
            "order_count": len(manifest.artifacts),
            "replay_spec_paths": [
                str((output_dir / item.file_name).resolve()) for item in manifest.artifacts
            ],
        }
    )


@backtest_app.command("materialize-frozen-replay-specs")
def materialize_frozen_replay_specs(  # noqa: PLR0917 - explicit replay source contract.
    frozen_orders: Annotated[
        Path,
        typer.Option(exists=True, dir_okay=False, help="Canonical live-safe order artifact."),
    ],
    strategy_config: Annotated[
        Path,
        typer.Option(exists=True, dir_okay=False, help="Matching earnings strategy YAML."),
    ],
    quote_files: Annotated[
        list[Path] | None,
        typer.Option("--quote-file", exists=True, dir_okay=False, help="Silver NBBO file."),
    ] = None,
    trade_files: Annotated[
        list[Path] | None,
        typer.Option("--trade-file", exists=True, dir_okay=False, help="Silver trades file."),
    ] = None,
    opening_auction_condition_codes: Annotated[
        list[int] | None,
        typer.Option("--opening-auction-code", min=0, help="Polygon opening-auction condition."),
    ] = None,
    output_dir: Annotated[
        Path,
        typer.Option(file_okay=False, help="Directory for canonical per-order replay specs."),
    ] = Path("replay-specs"),
    manifest_output: Annotated[
        Path,
        typer.Option(dir_okay=False, help="Immutable source/filtering audit manifest."),
    ] = Path("replay-materialization-manifest.json"),
) -> None:
    """Materialize replay specs directly from the frozen live-order artifact."""
    try:
        frozen = FrozenDailyOrders.load(frozen_orders)
        strategy = load_strategy_config(strategy_config)
        if strategy_file_sha256(strategy_config) != frozen.strategy_config_sha256:
            raise ValueError("strategy config does not match frozen daily orders")
        symbols = tuple(item.ticker for item in frozen.decision_snapshots)
        sources = replay_sources_from_files(
            quote_files=tuple(quote_files or ()),
            trade_files=tuple(trade_files or ()),
            expected_symbols=symbols,
            opening_auction_condition_codes=frozenset(opening_auction_condition_codes or ()),
        )
        spec = ReplayMaterializationSpec(
            orders=frozen.intended_orders,
            decision_snapshots=frozen.decision_snapshots,
            event_sources=sources,
            config=ReplayConfigSpec(
                market_impact_bps_coefficient=(strategy.costs.market_impact_coef_bps)
            ),
        )
        manifest = ReplaySpecMaterializer().materialize(
            spec,
            output_dir=output_dir,
            manifest_output=manifest_output,
        )
    except (OSError, ValidationError, ValueError, RuntimeError) as error:
        raise typer.BadParameter(
            str(error), param_hint="frozen replay materialization inputs"
        ) from error
    _echo_json(
        {
            "manifest_path": str(manifest_output.resolve()),
            "manifest_sha256": manifest.sha256,
            "input_sha256": manifest.input_sha256,
            "order_count": len(manifest.artifacts),
            "replay_spec_paths": [
                str((output_dir / item.file_name).resolve()) for item in manifest.artifacts
            ],
        }
    )


@backtest_app.command("replay-materialization")
def replay_materialization(
    manifest_file: Annotated[
        Path,
        typer.Option(
            exists=True,
            dir_okay=False,
            help="Canonical replay materialization manifest.",
        ),
    ],
    spec_directory: Annotated[
        Path,
        typer.Option(
            exists=True,
            file_okay=False,
            help="Directory containing the manifest's replay specs.",
        ),
    ],
    output_directory: Annotated[
        Path,
        typer.Option(file_okay=False, help="Directory for immutable replay evidence."),
    ],
    index_output: Annotated[
        Path,
        typer.Option(dir_okay=False, help="Immutable replay evidence index."),
    ],
) -> None:
    """Verify and replay every order in one materialization manifest."""
    try:
        manifest = ReplayMaterializationManifest.load(manifest_file)
        index = ReplayManifestRunner().run(
            manifest,
            spec_directory=spec_directory,
            output_directory=output_directory,
            index_output=index_output,
        )
    except (OSError, ValidationError, ValueError, RuntimeError) as error:
        raise typer.BadParameter(str(error), param_hint="replay materialization inputs") from error
    _echo_json(
        {
            "index_path": str(index_output.resolve()),
            "index_sha256": index.sha256,
            "materialization_manifest_sha256": (index.materialization_manifest_sha256),
            "order_count": len(index.evidence_files),
            "replay_evidence_paths": [
                str((output_directory / item).resolve()) for item in index.evidence_files
            ],
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


@model_app.command("plan-live-orders")
def plan_live_orders(
    planning_spec: Annotated[
        Path,
        typer.Option(
            exists=True,
            dir_okay=False,
            help="Decision-time scores, sizing observations, outcomes, and NBBO snapshots.",
        ),
    ],
    strategy_config: Annotated[
        Path,
        typer.Option(exists=True, dir_okay=False, help="Validated earnings strategy YAML."),
    ],
    output: Annotated[
        Path,
        typer.Option(dir_okay=False, help="Immutable linked paper/replay order artifact."),
    ],
    paper_batch_output: Annotated[
        Path,
        typer.Option(dir_okay=False, help="Direct input for `paper submit-batch`."),
    ],
) -> None:
    """Freeze live-safe orders without realized labels or execution prices."""
    try:
        spec = DailyOrderPlanningSpec.model_validate_json(planning_spec.read_bytes())
        artifact = LiveOrderPlanner(
            load_strategy_config(strategy_config),
            strategy_sha256=strategy_file_sha256(strategy_config),
        ).plan(spec)
        artifact.write(output)
        artifact.paper_batch.write(paper_batch_output)
    except (KeyError, OSError, ValidationError, ValueError, RuntimeError) as error:
        raise typer.BadParameter(str(error), param_hint="live order planning inputs") from error
    _echo_json(
        {
            "output": str(output.resolve()),
            "sha256": artifact.sha256,
            "input_sha256": artifact.input_sha256,
            "trade_date": artifact.trade_date,
            "position_count": len(artifact.portfolio.positions),
            "intended_order_count": len(artifact.intended_orders),
            "paper_order_count": len(artifact.paper_batch.orders),
            "paper_batch_output": str(paper_batch_output.resolve()),
            "paper_batch_sha256": artifact.paper_batch.sha256,
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


@evaluation_app.command("replay-frozen-session")
def aggregate_frozen_replay_session(
    frozen_orders: Annotated[
        Path,
        typer.Option(exists=True, dir_okay=False, help="Canonical frozen daily orders."),
    ],
    strategy_config: Annotated[
        Path,
        typer.Option(exists=True, dir_okay=False, help="Matching earnings strategy YAML."),
    ],
    evidence_files: Annotated[
        list[Path] | None,
        typer.Option(
            "--evidence-file",
            exists=True,
            dir_okay=False,
            help="Immutable replay evidence; repeat for every frozen order.",
        ),
    ] = None,
    output: Annotated[
        Path,
        typer.Option(dir_okay=False, help="Immutable daily Phase 6 replay report."),
    ] = Path("replay-session.json"),
) -> None:
    """Derive frozen entry/exit lifecycles and aggregate daily replay evidence."""
    try:
        frozen = FrozenDailyOrders.load(frozen_orders)
        strategy = load_strategy_config(strategy_config)
        if strategy_file_sha256(strategy_config) != frozen.strategy_config_sha256:
            raise ValueError("strategy config does not match frozen daily orders")
        evidence = tuple(NbboReplayEvidence.load(path) for path in tuple(evidence_files or ()))
        report = ReplaySessionAggregator().evaluate_frozen_long_orders(
            evidence=evidence,
            intended_orders=tuple(item.to_domain() for item in frozen.intended_orders),
            session_date=frozen.trade_date,
            initial_cash=frozen.portfolio.equity,
            commission_bps_per_side=strategy.costs.commission_bps_per_side,
        )
        report.write(output)
    except (OSError, ValidationError, ValueError, RuntimeError) as error:
        raise typer.BadParameter(str(error), param_hint="frozen replay-session inputs") from error
    _echo_json(
        {
            "output": str(output.resolve()),
            "sha256": report.sha256,
            "session_date": report.session_date,
            "intended_order_count": report.intended_order_count,
            "reconciliation_break_count": report.reconciliation_break_count,
            "net_return": report.net_return,
        }
    )
    if report.reconciliation_break_count:
        raise typer.Exit(code=1)


@evaluation_app.command("phase6-gate")
def evaluate_phase6_gate(
    aggregation_spec: Annotated[
        Path,
        typer.Option(
            exists=True,
            dir_okay=False,
            help="Calendar, proof bounds, capital, and immutable daily report paths.",
        ),
    ],
    output: Annotated[
        Path,
        typer.Option(dir_okay=False, help="Immutable terminal Phase 6 gate JSON."),
    ],
) -> None:
    """Evaluate the locked 90-session NBBO replay terminal thresholds."""
    try:
        spec = Phase6AggregationSpec.model_validate_json(aggregation_spec.read_bytes())
        session_path = (
            spec.session_file
            if spec.session_file.is_absolute()
            else aggregation_spec.parent / spec.session_file
        )
        calendar = SessionFileStore.load(session_path)
        health_path = (
            spec.workflow_health_file
            if spec.workflow_health_file.is_absolute()
            else aggregation_spec.parent / spec.workflow_health_file
        )
        workflow_health = WorkflowHealthReport.load(health_path)
        reports = tuple(
            ReplaySessionReport.load(
                configured_path
                if configured_path.is_absolute()
                else aggregation_spec.parent / configured_path
            )
            for configured_path in spec.session_report_files
        )
        evaluator = Phase6GateEvaluator(
            bootstrap_resamples=spec.bootstrap_resamples,
            seed=spec.seed,
        )
        report = evaluator.evaluate(
            calendar=calendar,
            workflow_health=workflow_health,
            reports=reports,
            proof_start=spec.proof_start,
            proof_end=spec.proof_end,
            initial_cash=spec.initial_cash,
        )
        report.write(output)
    except (OSError, ValidationError, ValueError, RuntimeError) as error:
        raise typer.BadParameter(str(error), param_hint="Phase 6 aggregation") from error
    _echo_json(
        {
            "output": str(output.resolve()),
            "sha256": report.sha256,
            "verdict": report.verdict,
            "authoritative_session_count": report.authoritative_session_count,
            "observed_session_count": report.observed_session_count,
            "scheduled_complete_session_count": report.scheduled_complete_session_count,
            "operational_uptime": report.operational_uptime,
            "passes_phase6_gate": report.passes_phase6_gate,
        }
    )


@evaluation_app.command("prepare-phase6-controls")
def prepare_phase6_controls(  # noqa: PLR0917 - explicit proof-control contract.
    session_file: Annotated[
        Path,
        typer.Option(exists=True, dir_okay=False, help="Authoritative market sessions."),
    ],
    proof_start: Annotated[str, typer.Option(help="First proof session date.")],
    proof_end: Annotated[str, typer.Option(help="Current proof session date.")],
    current_trade_date: Annotated[
        str,
        typer.Option(help="Daily report path to include before it exists."),
    ],
    initial_cash: Annotated[
        float,
        typer.Option(min=0.01, help="Proof starting equity."),
    ],
    artifact_root: Annotated[
        Path,
        typer.Option(file_okay=False, help="Root of deterministic daily artifacts."),
    ],
    health_output: Annotated[
        Path,
        typer.Option(dir_okay=False, help="Immutable pre-run workflow health report."),
    ],
    aggregation_output: Annotated[
        Path,
        typer.Option(dir_okay=False, help="Canonical Phase 6 aggregation input."),
    ],
    bootstrap_resamples: Annotated[
        int,
        typer.Option(min=1, help="Terminal Sharpe bootstrap resamples."),
    ] = 10_000,
    seed: Annotated[int, typer.Option(help="Deterministic bootstrap seed.")] = 20260427,
    env_file: EnvFileOption = None,
) -> None:
    """Prepare rolling Phase 6 controls from durable workflow/report evidence."""
    environment = _environment(env_file)
    try:
        calendar = SessionFileStore.load(session_file)
        controls = Phase6ControlBuilder().build(
            calendar=calendar,
            session_file=session_file,
            workflow_store=DailyWorkflowStore(environment.data_lake_root),
            proof_start=_parse_date(proof_start, option="--proof-start"),
            proof_end=_parse_date(proof_end, option="--proof-end"),
            current_trade_date=_parse_date(
                current_trade_date,
                option="--current-trade-date",
            ),
            initial_cash=initial_cash,
            artifact_root=artifact_root,
            health_output=health_output,
            aggregation_output=aggregation_output,
            bootstrap_resamples=bootstrap_resamples,
            seed=seed,
        )
    except (OSError, ValidationError, ValueError, RuntimeError) as error:
        raise typer.BadParameter(str(error), param_hint="Phase 6 control inputs") from error
    _echo_json(
        {
            "aggregation_output": str(aggregation_output.resolve()),
            "health_output": str(health_output.resolve()),
            "health_sha256": controls.health_sha256,
            "proof_start": controls.aggregation_spec.proof_start,
            "proof_end": controls.aggregation_spec.proof_end,
            "report_count": len(controls.included_report_files),
            "session_report_files": [str(item) for item in controls.included_report_files],
        }
    )


@monitoring_app.command("circuit-breakers")
def evaluate_circuit_breakers(
    spec_file: Annotated[
        Path,
        typer.Option(
            exists=True,
            dir_okay=False,
            help="Ordered replay, provider-freshness, and reconciliation observations.",
        ),
    ],
    output: Annotated[
        Path,
        typer.Option(dir_okay=False, help="Immutable circuit-breaker decision JSON."),
    ],
) -> None:
    """Fail closed when any documented operational safety threshold is crossed."""
    try:
        spec = CircuitBreakerEvaluationSpec.model_validate_json(spec_file.read_bytes())
        decision = CircuitBreakerEvaluator().evaluate(
            tuple(item.to_domain() for item in spec.observations)
        )
        decision.write(output)
    except (OSError, ValidationError, ValueError, RuntimeError) as error:
        raise typer.BadParameter(str(error), param_hint="circuit-breaker inputs") from error
    _echo_json(
        {
            "output": str(output.resolve()),
            "sha256": decision.sha256,
            "session_date": decision.session_date,
            "halt_new_orders": decision.halt_new_orders,
            "triggered_breakers": decision.triggered_breakers,
        }
    )
    if decision.halt_new_orders:
        raise typer.Exit(code=1)


@monitoring_app.command("prepare-breaker-controls")
def prepare_breaker_controls(  # noqa: PLR0917 - explicit control provenance contract.
    control_date: Annotated[str, typer.Option(help="Session being authorized.")],
    evaluated_at: Annotated[
        str,
        typer.Option(help="Offset-aware control evaluation timestamp."),
    ],
    polygon_data_observed_at: Annotated[
        str,
        typer.Option(help="Offset-aware latest Polygon observation."),
    ],
    alpaca_data_observed_at: Annotated[
        str,
        typer.Option(help="Offset-aware latest Alpaca observation."),
    ],
    frozen_order_files: Annotated[
        list[Path],
        typer.Option(
            "--frozen-orders",
            exists=True,
            dir_okay=False,
            help="Completed frozen session; repeat in date order.",
        ),
    ],
    replay_report_files: Annotated[
        list[Path],
        typer.Option(
            "--replay-report",
            exists=True,
            dir_okay=False,
            help="Matching completed replay report; repeat in date order.",
        ),
    ],
    output: Annotated[
        Path,
        typer.Option(dir_okay=False, help="Canonical breaker evaluation input."),
    ],
    reconciliation_break_age_sessions: Annotated[
        int | None,
        typer.Option(min=0, help="Age of any unresolved operational break."),
    ] = None,
) -> None:
    """Build breaker controls without relabeling prior-session replay evidence."""
    try:
        if len(frozen_order_files) != len(replay_report_files):
            raise ValueError("frozen-order and replay-report counts must match")
        spec = CircuitBreakerControlBuilder().build(
            control_date=_parse_date(control_date, option="--control-date"),
            evaluated_at=_parse_datetime(evaluated_at, option="--evaluated-at"),
            polygon_data_observed_at=_parse_datetime(
                polygon_data_observed_at,
                option="--polygon-data-observed-at",
            ),
            alpaca_data_observed_at=_parse_datetime(
                alpaca_data_observed_at,
                option="--alpaca-data-observed-at",
            ),
            sources=tuple(
                CompletedReplayControlSource(
                    frozen_orders=FrozenDailyOrders.load(frozen_path),
                    replay_report=ReplaySessionReport.load(report_path),
                )
                for frozen_path, report_path in zip(
                    frozen_order_files,
                    replay_report_files,
                    strict=True,
                )
            ),
            reconciliation_break_age_sessions=reconciliation_break_age_sessions,
            output=output,
        )
    except (OSError, ValidationError, ValueError, RuntimeError) as error:
        raise typer.BadParameter(str(error), param_hint="breaker control inputs") from error
    _echo_json(
        {
            "output": str(output.resolve()),
            "control_date": spec.observations[-1].session_date,
            "replay_source_dates": [item.replay_source_date for item in spec.observations],
            "observation_count": len(spec.observations),
        }
    )


@paper_app.command("submit-order")
def submit_paper_order(
    spec_file: Annotated[
        Path,
        typer.Option(
            exists=True,
            dir_okay=False,
            help="Strict immutable simple-equity paper order request JSON.",
        ),
    ],
    breaker_decision: Annotated[
        Path,
        typer.Option(
            exists=True,
            dir_okay=False,
            help="Current allow decision from `monitoring circuit-breakers`.",
        ),
    ],
    output: Annotated[
        Path,
        typer.Option(dir_okay=False, help="Immutable paper submission evidence JSON."),
    ],
    env_file: EnvFileOption = None,
) -> None:
    """Submit one idempotent order only to Alpaca's canonical paper host."""
    environment = _environment(env_file)
    try:
        api_key_id, secret_key = environment.require_alpaca_credentials()
    except RuntimeConfigurationError as error:
        raise typer.BadParameter(str(error), param_hint="environment") from error
    try:
        request = PaperOrderRequest.model_validate_json(spec_file.read_bytes())
        decision = CircuitBreakerDecision.load(breaker_decision)
        if decision.halt_new_orders:
            raise ValueError(
                "paper submission refused because the circuit-breaker decision halts new orders"
            )
        decision_age_seconds = (
            datetime.now(UTC) - decision.evaluated_at.astimezone(UTC)
        ).total_seconds()
        if not 0 <= decision_age_seconds <= 30 * 60:
            raise ValueError(
                "paper submission requires a circuit-breaker decision no more than 30 minutes old"
            )
        layout = LakehouseLayout(environment.data_lake_root)
        with httpx.Client(
            base_url=environment.alpaca_trading_base_url,
            timeout=environment.http_timeout_seconds,
        ) as http_client:
            submission = AlpacaPaperClient(
                api_key_id=api_key_id,
                secret_key=secret_key,
                http_client=http_client,
                bronze_writer=BronzeWriter(layout),
            ).submit(request)
        submission.write(output)
    except (OSError, ValidationError, ValueError, RuntimeError) as error:
        raise typer.BadParameter(str(error), param_hint="paper submission inputs") from error
    _echo_json(
        {
            "output": str(output.resolve()),
            "sha256": submission.sha256,
            "client_order_id": submission.request.client_order_id,
            "broker_order_id": submission.broker_order.order_id,
            "status": submission.broker_order.status,
            "idempotent_reuse": submission.idempotent_reuse,
        }
    )


@paper_app.command("submit-batch")
def submit_paper_batch(
    spec_file: Annotated[
        Path,
        typer.Option(exists=True, dir_okay=False, help="Sorted session paper-order batch JSON."),
    ],
    breaker_decision: Annotated[
        Path,
        typer.Option(exists=True, dir_okay=False, help="Fresh non-halted breaker decision."),
    ],
    output: Annotated[
        Path,
        typer.Option(dir_okay=False, help="Immutable complete session submission evidence."),
    ],
    env_file: EnvFileOption = None,
) -> None:
    """Submit or safely resume a complete paper session order batch."""
    environment = _environment(env_file)
    try:
        api_key_id, secret_key = environment.require_alpaca_credentials()
    except RuntimeConfigurationError as error:
        raise typer.BadParameter(str(error), param_hint="environment") from error
    try:
        spec = PaperOrderBatchSpec.model_validate_json(spec_file.read_bytes())
        decision = CircuitBreakerDecision.load(breaker_decision)
        layout = LakehouseLayout(environment.data_lake_root)
        with httpx.Client(
            base_url=environment.alpaca_trading_base_url,
            timeout=environment.http_timeout_seconds,
        ) as http_client:
            batch = PaperBatchSubmitter(
                AlpacaPaperClient(
                    api_key_id=api_key_id,
                    secret_key=secret_key,
                    http_client=http_client,
                    bronze_writer=BronzeWriter(layout),
                )
            ).submit(
                spec,
                breaker_decision=decision,
                evaluated_at=datetime.now(UTC),
            )
        batch.write(output)
    except (OSError, ValidationError, ValueError, RuntimeError) as error:
        raise typer.BadParameter(str(error), param_hint="paper batch inputs") from error
    _echo_json(
        {
            "output": str(output.resolve()),
            "sha256": batch.sha256,
            "session_date": batch.session_date,
            "order_count": len(batch.submissions),
            "idempotent_reuse_count": sum(item.idempotent_reuse for item in batch.submissions),
        }
    )


@paper_app.command("reconcile")
def reconcile_paper_orders(
    spec_file: Annotated[
        Path,
        typer.Option(
            exists=True,
            dir_okay=False,
            help="Replay evidence paths and after-close Alpaca paper order resources.",
        ),
    ],
    output: Annotated[
        Path,
        typer.Option(dir_okay=False, help="Immutable paper/replay reconciliation JSON."),
    ],
) -> None:
    """Reconcile paper operations without using paper P&L in strategy gates."""
    try:
        spec = PaperReconciliationSpec.model_validate_json(spec_file.read_bytes())
        evidence = tuple(
            NbboReplayEvidence.load(
                configured_path
                if configured_path.is_absolute()
                else spec_file.parent / configured_path
            )
            for configured_path in spec.replay_evidence_files
        )
        report = PaperOrderReconciler().evaluate(
            evidence=evidence,
            broker_orders=spec.broker_orders,
            session_date=spec.session_date,
            evaluated_at=spec.evaluated_at,
        )
        report.write(output)
    except (OSError, ValidationError, ValueError, RuntimeError) as error:
        raise typer.BadParameter(str(error), param_hint="paper reconciliation inputs") from error
    _echo_json(
        {
            "output": str(output.resolve()),
            "sha256": report.sha256,
            "session_date": report.session_date,
            "order_count": len(report.orders),
            "reconciliation_break_count": report.reconciliation_break_count,
            "all_orders_terminal": report.all_orders_terminal,
            "paper_pnl_is_gate_input": report.paper_pnl_is_gate_input,
        }
    )
    if report.reconciliation_break_count:
        raise typer.Exit(code=1)


@paper_app.command("reconcile-frozen")
def reconcile_frozen_paper_orders(
    frozen_orders: Annotated[
        Path,
        typer.Option(exists=True, dir_okay=False, help="Canonical frozen daily orders."),
    ],
    evidence_files: Annotated[
        list[Path] | None,
        typer.Option(
            "--evidence-file",
            exists=True,
            dir_okay=False,
            help="Immutable per-order replay evidence; repeat for every frozen order.",
        ),
    ] = None,
    output: Annotated[
        Path,
        typer.Option(dir_okay=False, help="Immutable paper/replay reconciliation JSON."),
    ] = Path("paper-reconciliation.json"),
    env_file: EnvFileOption = None,
) -> None:
    """Fetch frozen orders from Alpaca paper and reconcile them to replay."""
    try:
        frozen = FrozenDailyOrders.load(frozen_orders)
        evidence = tuple(NbboReplayEvidence.load(path) for path in tuple(evidence_files or ()))
        reconciler = PaperOrderReconciler()
        evaluated_at = datetime.now(UTC)
        if frozen.intended_orders:
            environment = _environment(env_file)
            api_key_id, secret_key = environment.require_alpaca_credentials()
            layout = LakehouseLayout(environment.data_lake_root)
            with httpx.Client(
                base_url=environment.alpaca_trading_base_url,
                timeout=environment.http_timeout_seconds,
            ) as http_client:
                report = reconciler.fetch_and_evaluate(
                    evidence=evidence,
                    intended_orders=tuple(item.to_domain() for item in frozen.intended_orders),
                    order_lookup=AlpacaPaperClient(
                        api_key_id=api_key_id,
                        secret_key=secret_key,
                        http_client=http_client,
                        bronze_writer=BronzeWriter(layout),
                    ),
                    session_date=frozen.trade_date,
                    evaluated_at=evaluated_at,
                )
        else:
            if evidence:
                raise ValueError("no-trade frozen orders must not have replay evidence")
            report = reconciler.evaluate(
                evidence=(),
                broker_orders=(),
                session_date=frozen.trade_date,
                evaluated_at=evaluated_at,
            )
        report.write(output)
    except (
        OSError,
        RuntimeConfigurationError,
        ValidationError,
        ValueError,
        RuntimeError,
    ) as error:
        raise typer.BadParameter(
            str(error), param_hint="frozen paper reconciliation inputs"
        ) from error
    _echo_json(
        {
            "output": str(output.resolve()),
            "sha256": report.sha256,
            "session_date": report.session_date,
            "order_count": len(report.orders),
            "reconciliation_break_count": report.reconciliation_break_count,
            "all_orders_terminal": report.all_orders_terminal,
            "paper_pnl_is_gate_input": report.paper_pnl_is_gate_input,
        }
    )
    if report.reconciliation_break_count:
        raise typer.Exit(code=1)


@workflow_app.command("run")
def run_daily_workflow(
    spec_file: Annotated[
        Path,
        typer.Option(
            exists=True,
            dir_okay=False,
            help="Complete stage-to-qee-command mapping for one trade date.",
        ),
    ],
    env_file: EnvFileOption = None,
) -> None:
    """Loop through concrete qee stages until complete, leased, or failed."""
    environment = _environment(env_file)
    try:
        spec = WorkflowRunSpec.model_validate_json(spec_file.read_bytes())
        state = DailyWorkflowRunner(
            store=DailyWorkflowStore(environment.data_lake_root),
            handlers=spec.handlers(working_directory=spec_file.parent),
            worker_id=spec.worker_id,
            clock=lambda: datetime.now(UTC),
            trigger=spec.trigger,
            lease_duration=timedelta(seconds=spec.lease_seconds),
        ).run_until_idle(trade_date=spec.trade_date)
    except (OSError, ValidationError, ValueError, RuntimeError) as error:
        raise typer.BadParameter(str(error), param_hint="workflow run inputs") from error
    current = next(
        (record for record in state.stages if record.status is not StageStatus.SUCCEEDED),
        None,
    )
    _echo_json(
        {
            "trade_date": state.trade_date,
            "revision": state.revision,
            "sha256": state.sha256,
            "complete": state.complete,
            "current_stage": current.stage if current else None,
            "current_status": current.status if current else None,
            "attempts": current.attempts if current else None,
            "error_type": current.error_type if current else None,
            "error_message": current.error_message if current else None,
        }
    )
    if not state.complete:
        raise typer.Exit(code=1)


@workflow_app.command("generate")
def generate_daily_workflow(  # noqa: PLR0917 - explicit operational inputs.
    trade_date: Annotated[str, typer.Option(help="Trading session date (YYYY-MM-DD).")],
    planning_spec: Annotated[
        Path,
        typer.Option(exists=True, dir_okay=False, help="Live-safe daily planning input."),
    ],
    strategy_config: Annotated[
        Path,
        typer.Option(exists=True, dir_okay=False, help="Validated strategy YAML."),
    ],
    breaker_spec: Annotated[
        Path,
        typer.Option(exists=True, dir_okay=False, help="Current circuit-breaker input."),
    ],
    phase6_spec: Annotated[
        Path,
        typer.Option(exists=True, dir_okay=False, help="Current Phase 6 aggregation input."),
    ],
    artifact_root: Annotated[
        Path,
        typer.Option(file_okay=False, help="Root for deterministic daily artifacts."),
    ],
    output: Annotated[
        Path,
        typer.Option(dir_okay=False, help="Canonical eight-stage workflow run spec."),
    ],
    worker_id: Annotated[str, typer.Option(help="Persistent worker identity.")],
    trigger: Annotated[
        WorkflowTrigger,
        typer.Option(help="Workflow invocation provenance."),
    ] = WorkflowTrigger.SCHEDULED,
    lease_seconds: Annotated[
        int,
        typer.Option(min=1, max=3600, help="Per-stage lease duration."),
    ] = 900,
    command_timeout_seconds: Annotated[
        float,
        typer.Option(min=1, max=7200, help="Per-command timeout."),
    ] = 1800,
) -> None:
    """Generate the complete daily loop with typed cross-stage bindings."""
    selected_date = _parse_date(trade_date, option="--trade-date")
    try:
        planning = DailyOrderPlanningSpec.model_validate_json(planning_spec.read_bytes())
        if planning.trade_date != selected_date:
            raise ValueError("planning spec trade date differs from workflow trade date")
        load_strategy_config(strategy_config)
        breakers = CircuitBreakerEvaluationSpec.model_validate_json(breaker_spec.read_bytes())
        if breakers.observations[-1].session_date != selected_date:
            raise ValueError("breaker spec latest date differs from workflow trade date")
        phase6 = Phase6AggregationSpec.model_validate_json(phase6_spec.read_bytes())
        expected_session_report = (
            artifact_root.resolve()
            / f"trade_date={selected_date.isoformat()}"
            / "replay-session.json"
        )
        configured_reports = {
            (path.resolve() if path.is_absolute() else (phase6_spec.parent / path).resolve())
            for path in phase6.session_report_files
        }
        if expected_session_report not in configured_reports:
            raise ValueError(
                "Phase 6 spec must include this workflow's deterministic replay-session path"
            )
        spec = DailyWorkflowSpecGenerator().generate(
            trade_date=selected_date,
            trigger=trigger,
            worker_id=worker_id,
            planning_spec=planning_spec,
            strategy_config=strategy_config,
            breaker_spec=breaker_spec,
            phase6_spec=phase6_spec,
            artifact_root=artifact_root,
            market_events_not_before=(planning.exit_expires_at + timedelta(minutes=5)),
            lease_seconds=lease_seconds,
            command_timeout_seconds=command_timeout_seconds,
        )
        spec.write(output)
    except (OSError, ValidationError, ValueError, RuntimeError) as error:
        raise typer.BadParameter(str(error), param_hint="workflow generation inputs") from error
    _echo_json(
        {
            "output": str(output.resolve()),
            "sha256": spec.sha256,
            "trade_date": spec.trade_date,
            "trigger": spec.trigger,
            "stage_count": len(spec.stages),
        }
    )


@workflow_app.command("initialize")
def initialize_daily_workflow(
    trade_date: Annotated[str, typer.Option(help="Trading session date (YYYY-MM-DD).")],
    env_file: EnvFileOption = None,
) -> None:
    """Create the initial append-only workflow revision if it does not exist."""
    selected_date = _parse_date(trade_date, option="--trade-date")
    environment = _environment(env_file)
    store = DailyWorkflowStore(environment.data_lake_root)
    state = store.load_latest(selected_date)
    created = state is None
    if state is None:
        state = store.write(
            DailyWorkflowState.initialize(
                trade_date=selected_date,
                now=datetime.now(UTC),
                trigger=WorkflowTrigger.MANUAL,
            )
        )
    _echo_json(
        {
            "trade_date": state.trade_date,
            "created": created,
            "revision": state.revision,
            "sha256": state.sha256,
            "complete": state.complete,
        }
    )


@workflow_app.command("status")
def daily_workflow_status(
    trade_date: Annotated[str, typer.Option(help="Trading session date (YYYY-MM-DD).")],
    env_file: EnvFileOption = None,
) -> None:
    """Report the latest verified revision and the stage requiring attention."""
    selected_date = _parse_date(trade_date, option="--trade-date")
    environment = _environment(env_file)
    state = DailyWorkflowStore(environment.data_lake_root).load_latest(selected_date)
    if state is None:
        _echo_json(
            {
                "trade_date": selected_date,
                "initialized": False,
                "complete": False,
            }
        )
        raise typer.Exit(code=1)
    artifact_error: str | None = None
    try:
        state.verify_artifacts()
    except ValueError as error:
        artifact_error = str(error)
    current = next(
        (record for record in state.stages if record.status is not StageStatus.SUCCEEDED),
        None,
    )
    _echo_json(
        {
            "trade_date": state.trade_date,
            "initialized": True,
            "revision": state.revision,
            "sha256": state.sha256,
            "trigger": state.trigger,
            "complete": state.complete,
            "artifacts_intact": artifact_error is None,
            "artifact_error": artifact_error,
            "current_stage": current.stage if current else None,
            "current_status": current.status if current else None,
            "attempts": current.attempts if current else None,
            "worker_id": current.worker_id if current else None,
            "lease_expires_at": current.lease_expires_at if current else None,
        }
    )
    if not state.complete or artifact_error is not None:
        raise typer.Exit(code=1)


@workflow_app.command("health")
def daily_workflow_health(
    session_file: Annotated[
        Path,
        typer.Option(exists=True, dir_okay=False, help="Authoritative market-session JSON."),
    ],
    start: Annotated[str, typer.Option(help="Inclusive health range start (YYYY-MM-DD).")],
    end: Annotated[str, typer.Option(help="Inclusive health range end (YYYY-MM-DD).")],
    output: Annotated[
        Path,
        typer.Option(dir_okay=False, help="Immutable unattended workflow health JSON."),
    ],
    env_file: EnvFileOption = None,
) -> None:
    """Evaluate scheduled operational uptime and five-session readiness."""
    environment = _environment(env_file)
    try:
        report = WorkflowHealthEvaluator().evaluate(
            calendar=SessionFileStore.load(session_file),
            store=DailyWorkflowStore(environment.data_lake_root),
            start_date=_parse_date(start, option="--start"),
            end_date=_parse_date(end, option="--end"),
        )
        report.write(output)
    except (OSError, ValueError, RuntimeError) as error:
        raise typer.BadParameter(str(error), param_hint="workflow health inputs") from error
    _echo_json(
        {
            "output": str(output.resolve()),
            "sha256": report.sha256,
            "operational_uptime": report.operational_uptime,
            "maximum_consecutive_scheduled_successes": (
                report.maximum_consecutive_scheduled_successes
            ),
            "passes_five_session_unattended_gate": (report.passes_five_session_unattended_gate),
        }
    )
    if not report.passes_five_session_unattended_gate:
        raise typer.Exit(code=1)


@workflow_app.command("worker")
def run_workflow_worker(
    inbox: Annotated[
        Path,
        typer.Option(exists=True, file_okay=False, help="Directory of immutable run specs."),
    ],
    worker_id: Annotated[str, typer.Option(help="Persistent worker identity.")],
    poll_seconds: Annotated[
        float,
        typer.Option(min=1, max=60, help="Inbox polling interval."),
    ] = 10,
    once: Annotated[
        bool,
        typer.Option(help="Run one scan for verification instead of polling continuously."),
    ] = False,
    env_file: EnvFileOption = None,
) -> None:
    """Continuously resume every workflow specification in an inbox."""
    environment = _environment(env_file)
    try:
        worker = WorkflowInboxWorker(
            data_lake_root=environment.data_lake_root,
            worker_id=worker_id,
            clock=lambda: datetime.now(UTC),
        )
        while True:
            report, report_path = worker.run_once(inbox)
            _echo_json(
                {
                    "report": str(report_path),
                    "sha256": report.sha256,
                    "evaluated_at": report.evaluated_at,
                    "spec_count": len(report.results),
                    "complete_count": sum(item.complete for item in report.results),
                    "all_complete": report.all_complete,
                }
            )
            if once:
                if not report.all_complete:
                    raise typer.Exit(code=1)
                return
            time.sleep(poll_seconds)
    except (OSError, ValidationError, ValueError, RuntimeError) as error:
        raise typer.BadParameter(str(error), param_hint="workflow worker inputs") from error


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


@ingest_app.command("frozen-market-events")
def ingest_frozen_market_events(
    frozen_orders: Annotated[
        Path,
        typer.Option(exists=True, dir_okay=False, help="Canonical frozen daily orders."),
    ],
    manifest_output: Annotated[
        Path,
        typer.Option(dir_okay=False, help="Immutable frozen capture manifest."),
    ],
    env_file: EnvFileOption = None,
) -> None:
    """Capture all selected symbols over their complete frozen order windows."""
    try:
        frozen = FrozenDailyOrders.load(frozen_orders)
        captured_at = datetime.now(UTC)
        if frozen.intended_orders:
            environment = _environment(env_file)
            api_key = _required_key(environment.require_polygon_api_key)
            layout = LakehouseLayout(environment.data_lake_root)
            with httpx.Client(
                base_url=environment.polygon_base_url,
                timeout=environment.http_timeout_seconds,
            ) as http_client:
                manifest = FrozenMarketEventsIngestor(
                    MarketEventsIngestor(
                        client=PolygonClient(
                            api_key=api_key,
                            http_client=http_client,
                            bronze_writer=BronzeWriter(layout),
                        ),
                        silver_writer=SilverWriter(layout),
                    )
                ).ingest(
                    intended_orders=tuple(item.to_domain() for item in frozen.intended_orders),
                    trade_date=frozen.trade_date,
                    frozen_orders_sha256=frozen.sha256,
                    manifest_output=manifest_output,
                    captured_at=captured_at,
                )
        else:
            manifest = FrozenMarketEventsManifest(
                schema_version=1,
                trade_date=frozen.trade_date,
                captured_at=captured_at,
                frozen_orders_sha256=frozen.sha256,
                artifacts=(),
            )
            manifest.write(manifest_output)
    except (
        OSError,
        RuntimeConfigurationError,
        ValidationError,
        ValueError,
        RuntimeError,
    ) as error:
        raise typer.BadParameter(str(error), param_hint="frozen market-event inputs") from error
    _echo_json(
        {
            "manifest_path": str(manifest_output.resolve()),
            "manifest_sha256": manifest.sha256,
            "symbol_count": len(manifest.artifacts),
            "quote_paths": [item.quote_path for item in manifest.artifacts],
            "trade_paths": [item.trade_path for item in manifest.artifacts],
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
