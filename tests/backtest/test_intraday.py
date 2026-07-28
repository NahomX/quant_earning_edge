"""Timestamped same-session vectorbt execution tests."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime
from typing import TYPE_CHECKING

import pytest
from typer.testing import CliRunner

from quant_earning_edge.backtest import TradeIntent, VectorbtIntradayEngine
from quant_earning_edge.cli import app

if TYPE_CHECKING:
    from pathlib import Path


def _trade(
    *,
    trade_id: str,
    symbol: str,
    session: date,
    side: str,
    entry_price: float,
    exit_price: float,
) -> TradeIntent:
    return TradeIntent(
        trade_id=trade_id,
        symbol=symbol,
        side=side,  # type: ignore[arg-type]
        entry_date=session,
        exit_date=session,
        shares=100,
        entry_price=entry_price,
        exit_price=exit_price,
        entry_average_daily_volume_shares=1_000_000,
        exit_average_daily_volume_shares=1_000_000,
        holding_sessions=0,
        entry_at=datetime.combine(session, datetime.min.time(), UTC).replace(
            hour=14,
            minute=30,
        ),
        exit_at=datetime.combine(session, datetime.min.time(), UTC).replace(
            hour=21,
        ),
    )


def test_intraday_engine_reconciles_same_session_long_and_short() -> None:
    first = date(2025, 1, 2)
    second = date(2025, 1, 3)
    trades = (
        _trade(
            trade_id="long",
            symbol="AAA",
            session=first,
            side="long",
            entry_price=100.0,
            exit_price=105.0,
        ),
        _trade(
            trade_id="short",
            symbol="BBB",
            session=second,
            side="short",
            entry_price=50.0,
            exit_price=48.0,
        ),
    )

    result = VectorbtIntradayEngine().run(
        trades=trades,
        sessions=(first, second),
        initial_cash=100_000.0,
    )

    total_cost = sum(item.total_cost for item in result.trades)
    assert result.engine.startswith("vectorbt-intraday-")
    assert result.final_gross_equity == pytest.approx(100_700.0)
    assert result.final_net_equity == pytest.approx(100_700.0 - total_cost)
    assert all(
        item.net_pnl == pytest.approx(item.gross_pnl - item.total_cost) for item in result.daily
    )


def test_run_ledger_cli_accepts_timestamped_round_trip(tmp_path: Path) -> None:
    session = date(2025, 1, 2)
    trade = _trade(
        trade_id="long",
        symbol="AAA",
        session=session,
        side="long",
        entry_price=100.0,
        exit_price=105.0,
    )
    spec = tmp_path / "intraday.json"
    report = tmp_path / "report.json"
    spec.write_text(
        json.dumps(
            {
                "initial_cash": 100000,
                "sessions": [session.isoformat()],
                "marks": [],
                "trades": [
                    {
                        **trade.__dict__,
                        "entry_date": trade.entry_date.isoformat(),
                        "exit_date": trade.exit_date.isoformat(),
                        "entry_at": trade.entry_at.isoformat() if trade.entry_at else None,
                        "exit_at": trade.exit_at.isoformat() if trade.exit_at else None,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        app,
        [
            "backtest",
            "run-ledger",
            "--spec-file",
            str(spec),
            "--output",
            str(report),
            "--bootstrap-resamples",
            "10",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["engine"].startswith("vectorbt-intraday-")
    assert payload["trade_count"] == 1
