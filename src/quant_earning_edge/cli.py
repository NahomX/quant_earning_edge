"""Operational CLI for ingestion, universe snapshots, and readiness evidence."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import httpx
import typer

from quant_earning_edge import __version__
from quant_earning_edge.data import (
    BarBackfillJob,
    BarBackfillStore,
    BarCoverageAuditor,
    BarsIngestor,
    BronzeWriter,
    EarningsIngestor,
    LakehouseLayout,
    SessionFileStore,
    SilverWriter,
)
from quant_earning_edge.data.clients import AlpacaCalendarClient, FinnhubClient, PolygonClient
from quant_earning_edge.runtime import (
    RuntimeConfigurationError,
    RuntimeEnvironment,
    load_runtime_environment,
)
from quant_earning_edge.universe import (
    DailyUniverseJob,
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
app.add_typer(ingest_app, name="ingest")
app.add_typer(universe_app, name="universe")
app.add_typer(backfill_app, name="backfill")
app.add_typer(calendar_app, name="calendar")

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
