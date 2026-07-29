"""Frozen order batches drive exact per-symbol market-event capture windows."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, date, datetime, timedelta
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest
from typer.testing import CliRunner

from quant_earning_edge.backtest import IntendedOrder
from quant_earning_edge.cli import app
from quant_earning_edge.data import FrozenMarketEventsIngestor
from quant_earning_edge.signals import FrozenDailyOrders

if TYPE_CHECKING:
    from pathlib import Path


def _order(
    order_id: str,
    *,
    symbol: str,
    submitted_at: datetime,
    expires_at: datetime,
) -> IntendedOrder:
    return IntendedOrder(
        order_id=order_id,
        ticker=symbol,
        side="buy" if order_id.endswith("-entry") else "sell",
        quantity=10,
        decision_time=submitted_at - timedelta(hours=12),
        submitted_at=submitted_at,
        expires_at=expires_at,
        average_daily_volume_shares=1_000_000,
    )


def test_batch_ingestor_combines_each_symbols_frozen_windows(tmp_path: Path) -> None:
    opened = datetime(2026, 7, 28, 13, 30, tzinfo=UTC)
    closed = datetime(2026, 7, 28, 20, 0, tzinfo=UTC)
    calls: list[tuple[str, datetime, datetime]] = []

    class Ingestor:
        def ingest(
            self,
            *,
            symbol: str,
            event_date: date,
            start_at: datetime,
            end_at: datetime,
        ) -> SimpleNamespace:
            calls.append((symbol, start_at, end_at))
            quote_path = tmp_path / f"{symbol}-quotes.parquet"
            trade_path = tmp_path / f"{symbol}-trades.parquet"
            quote_path.write_bytes(f"{symbol}-quotes".encode())
            trade_path.write_bytes(f"{symbol}-trades".encode())
            return SimpleNamespace(
                symbol=symbol,
                event_date=event_date,
                start_at=start_at,
                end_at=end_at,
                quote_count=100,
                trade_count=50,
                quote_artifact=SimpleNamespace(
                    path=quote_path,
                    sha256=hashlib.sha256(quote_path.read_bytes()).hexdigest(),
                ),
                trade_artifact=SimpleNamespace(
                    path=trade_path,
                    sha256=hashlib.sha256(trade_path.read_bytes()).hexdigest(),
                ),
            )

    orders = (
        _order(
            "aaa-entry",
            symbol="AAA",
            submitted_at=opened,
            expires_at=opened + timedelta(minutes=5),
        ),
        _order(
            "aaa-exit",
            symbol="AAA",
            submitted_at=closed - timedelta(minutes=10),
            expires_at=closed,
        ),
        _order(
            "bbb-entry",
            symbol="BBB",
            submitted_at=opened + timedelta(minutes=1),
            expires_at=opened + timedelta(minutes=6),
        ),
        _order(
            "bbb-exit",
            symbol="BBB",
            submitted_at=closed - timedelta(minutes=9),
            expires_at=closed - timedelta(minutes=1),
        ),
    )
    manifest_path = tmp_path / "capture.json"

    manifest = FrozenMarketEventsIngestor(Ingestor()).ingest(  # type: ignore[arg-type]
        intended_orders=orders,
        trade_date=date(2026, 7, 28),
        frozen_orders_sha256="c" * 64,
        manifest_output=manifest_path,
        captured_at=closed + timedelta(minutes=10),
    )
    repeated = FrozenMarketEventsIngestor(Ingestor()).ingest(  # type: ignore[arg-type]
        intended_orders=orders,
        trade_date=date(2026, 7, 28),
        frozen_orders_sha256="c" * 64,
        manifest_output=manifest_path,
        captured_at=closed + timedelta(minutes=10),
    )

    assert repeated == manifest
    assert calls[:2] == [
        ("AAA", opened, closed),
        ("BBB", opened + timedelta(minutes=1), closed - timedelta(minutes=1)),
    ]
    assert tuple(item.symbol for item in manifest.artifacts) == ("AAA", "BBB")
    assert json.loads(manifest_path.read_bytes())["frozen_orders_sha256"] == "c" * 64


def test_no_trade_capture_cli_needs_no_provider_credentials(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frozen_path = tmp_path / "frozen.json"
    frozen_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        FrozenDailyOrders,
        "load",
        staticmethod(
            lambda _: SimpleNamespace(
                intended_orders=(),
                trade_date=date(2026, 7, 28),
                sha256="d" * 64,
            )
        ),
    )
    manifest_path = tmp_path / "capture.json"

    result = CliRunner().invoke(
        app,
        [
            "ingest",
            "frozen-market-events",
            "--frozen-orders",
            str(frozen_path),
            "--manifest-output",
            str(manifest_path),
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["symbol_count"] == 0
    assert payload["quote_paths"] == []
    assert json.loads(manifest_path.read_bytes())["artifacts"] == []


def test_batch_capture_refuses_to_read_before_every_order_expires(
    tmp_path: Path,
) -> None:
    opened = datetime(2026, 7, 28, 13, 30, tzinfo=UTC)
    calls: list[str] = []

    class Ingestor:
        def ingest(self, **kwargs: object) -> None:
            calls.append(str(kwargs["symbol"]))

    with pytest.raises(ValueError, match="post-expiry settlement delay"):
        FrozenMarketEventsIngestor(Ingestor()).ingest(  # type: ignore[arg-type]
            intended_orders=(
                _order(
                    "aaa-entry",
                    symbol="AAA",
                    submitted_at=opened,
                    expires_at=opened + timedelta(minutes=5),
                ),
            ),
            trade_date=date(2026, 7, 28),
            frozen_orders_sha256="e" * 64,
            manifest_output=tmp_path / "capture.json",
            captured_at=opened,
        )

    assert calls == []
