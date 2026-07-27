"""Operational CLI for ingestion, universe snapshots, and readiness evidence."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import httpx
import typer

from quant_earning_edge import __version__
from quant_earning_edge.data import (
    BarsIngestor,
    BronzeWriter,
    EarningsIngestor,
    LakehouseLayout,
    SilverWriter,
)
from quant_earning_edge.data.clients import FinnhubClient, PolygonClient
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
app.add_typer(ingest_app, name="ingest")
app.add_typer(universe_app, name="universe")

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
