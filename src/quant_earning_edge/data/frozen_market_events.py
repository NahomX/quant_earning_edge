"""Batch Polygon market-event capture for one frozen daily order artifact."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import timedelta
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence
    from datetime import date, datetime
    from pathlib import Path

    from quant_earning_edge.backtest import IntendedOrder
    from quant_earning_edge.data.ingest import MarketEventsIngestor

_CAPTURE_SETTLE_DELAY = timedelta(minutes=5)


@dataclass(frozen=True)
class FrozenMarketEventArtifact:
    """Silver quote/trade outputs for one frozen symbol window."""

    symbol: str
    start_at: datetime
    end_at: datetime
    quote_count: int
    trade_count: int
    quote_path: str
    quote_sha256: str
    trade_path: str
    trade_sha256: str

    def __post_init__(self) -> None:
        if self.start_at.tzinfo is None or self.start_at.utcoffset() is None:
            raise ValueError("frozen market-event start_at must be timezone-aware")
        if self.end_at.tzinfo is None or self.end_at.utcoffset() is None:
            raise ValueError("frozen market-event end_at must be timezone-aware")
        if self.start_at >= self.end_at:
            raise ValueError("frozen market-event window must be increasing")
        if not self.symbol.strip() or self.symbol != self.symbol.strip().upper():
            raise ValueError("frozen market-event symbol must be normalized")
        if self.quote_count < 0 or self.trade_count < 0:
            raise ValueError("frozen market-event counts cannot be negative")
        for name, value in (
            ("quote", self.quote_sha256),
            ("trade", self.trade_sha256),
        ):
            if len(value) != 64 or any(item not in "0123456789abcdef" for item in value):
                raise ValueError(f"frozen market-event {name} hash must be SHA-256")
        if not self.quote_path.strip() or not self.trade_path.strip():
            raise ValueError("frozen market-event paths must not be blank")


@dataclass(frozen=True)
class FrozenMarketEventsManifest:
    """Immutable link from frozen orders to exact silver execution windows."""

    schema_version: int
    trade_date: date
    captured_at: datetime
    frozen_orders_sha256: str
    artifacts: tuple[FrozenMarketEventArtifact, ...]

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("unsupported frozen market-events manifest schema")
        if self.captured_at.tzinfo is None or self.captured_at.utcoffset() is None:
            raise ValueError("frozen market-events captured_at must be timezone-aware")
        if len(self.frozen_orders_sha256) != 64 or any(
            item not in "0123456789abcdef" for item in self.frozen_orders_sha256
        ):
            raise ValueError("frozen orders hash must be SHA-256")
        symbols = tuple(item.symbol for item in self.artifacts)
        if symbols != tuple(sorted(set(symbols))):
            raise ValueError("frozen market-event artifacts must have unique sorted symbols")

    @property
    def canonical_bytes(self) -> bytes:
        return json.dumps(
            asdict(self),
            default=lambda item: item.isoformat(),
            sort_keys=True,
            separators=(",", ":"),
        ).encode()

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes).hexdigest()

    def write(self, output: Path) -> None:
        encoded = self.canonical_bytes
        output.parent.mkdir(parents=True, exist_ok=True)
        try:
            with output.open("xb") as destination:
                destination.write(encoded)
        except FileExistsError:
            if output.read_bytes() != encoded:
                raise RuntimeError(f"frozen market-events collision at {output}") from None


class FrozenMarketEventsIngestor:
    """Capture each selected symbol once over its complete execution window."""

    def __init__(self, ingestor: MarketEventsIngestor) -> None:
        self._ingestor = ingestor

    def ingest(
        self,
        *,
        intended_orders: Sequence[IntendedOrder],
        trade_date: date,
        frozen_orders_sha256: str,
        manifest_output: Path,
        captured_at: datetime,
    ) -> FrozenMarketEventsManifest:
        if captured_at.tzinfo is None or captured_at.utcoffset() is None:
            raise ValueError("frozen market-event captured_at must be timezone-aware")
        order_ids = tuple(item.order_id for item in intended_orders)
        if order_ids != tuple(sorted(set(order_ids))):
            raise ValueError("frozen market-event order ids must be unique and sorted")
        if intended_orders and captured_at < (
            max(item.expires_at for item in intended_orders) + _CAPTURE_SETTLE_DELAY
        ):
            raise ValueError(
                "frozen market events require a five-minute post-expiry settlement delay"
            )
        by_symbol: dict[str, list[IntendedOrder]] = {}
        for order in intended_orders:
            if order.submitted_at.date() != trade_date or order.expires_at.date() != trade_date:
                raise ValueError("frozen market-event order window differs from trade date")
            by_symbol.setdefault(order.ticker, []).append(order)
        artifacts: list[FrozenMarketEventArtifact] = []
        for symbol, orders in sorted(by_symbol.items()):
            start_at = min(item.submitted_at for item in orders)
            end_at = max(item.expires_at for item in orders)
            result = self._ingestor.ingest(
                symbol=symbol,
                event_date=trade_date,
                start_at=start_at,
                end_at=end_at,
            )
            if (
                result.symbol != symbol
                or result.event_date != trade_date
                or result.start_at != start_at
                or result.end_at != end_at
            ):
                raise ValueError("market-event ingestion result differs from frozen request")
            _verify_artifact(result.quote_artifact.path, result.quote_artifact.sha256)
            _verify_artifact(result.trade_artifact.path, result.trade_artifact.sha256)
            artifacts.append(
                FrozenMarketEventArtifact(
                    symbol=result.symbol,
                    start_at=result.start_at,
                    end_at=result.end_at,
                    quote_count=result.quote_count,
                    trade_count=result.trade_count,
                    quote_path=str(result.quote_artifact.path.resolve()),
                    quote_sha256=result.quote_artifact.sha256,
                    trade_path=str(result.trade_artifact.path.resolve()),
                    trade_sha256=result.trade_artifact.sha256,
                )
            )
        manifest = FrozenMarketEventsManifest(
            schema_version=1,
            trade_date=trade_date,
            captured_at=captured_at,
            frozen_orders_sha256=frozen_orders_sha256,
            artifacts=tuple(artifacts),
        )
        manifest.write(manifest_output)
        return manifest


def _verify_artifact(path: Path, expected_sha256: str) -> None:
    if not path.is_file():
        raise ValueError(f"frozen market-event artifact does not exist: {path}")
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha256:
        raise ValueError(f"frozen market-event artifact hash differs: {path}")
