"""CLI entry point. Every backtest, ingest, or live action is invoked through here.

Each command logs to MLflow with a manifest (code hash, data hash, params, seed).
"""

from __future__ import annotations

import typer

app = typer.Typer(no_args_is_help=True, help="quant_earning_edge command-line interface.")


@app.command()
def version() -> None:
    """Print the installed package version."""
    from quant_earning_edge import __version__

    typer.echo(__version__)


if __name__ == "__main__":
    app()
