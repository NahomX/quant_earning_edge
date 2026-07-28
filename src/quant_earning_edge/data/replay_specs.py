"""Materialize causal replay specifications from frozen orders and silver events."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path  # noqa: TC003 - Pydantic resolves runtime annotations.

from pydantic import BaseModel, ConfigDict, Field, model_validator

from quant_earning_edge.backtest.nbbo_spec import (
    DecisionSnapshotSpec,
    IntendedOrderSpec,
    NbboQuoteSpec,
    NbboReplaySpec,
    ReplayConfigSpec,
    TradePrintSpec,
)
from quant_earning_edge.data.market_events import ReplayMarketDataLoader


class _StrictSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ReplayEventSourceSpec(_StrictSpec):
    """Silver quote/trade files for one normalized symbol."""

    symbol: str
    quote_files: tuple[Path, ...] = Field(min_length=1)
    trade_files: tuple[Path, ...] = ()
    opening_auction_condition_codes: frozenset[int] = frozenset()

    @model_validator(mode="after")
    def normalize_source(self) -> ReplayEventSourceSpec:
        symbol = self.symbol.strip().upper()
        if not symbol:
            raise ValueError("replay event source symbol must not be blank")
        if any(code < 0 for code in self.opening_auction_condition_codes):
            raise ValueError("opening-auction condition codes cannot be negative")
        quote_files = tuple(sorted(set(self.quote_files), key=str))
        trade_files = tuple(sorted(set(self.trade_files), key=str))
        object.__setattr__(self, "symbol", symbol)
        object.__setattr__(self, "quote_files", quote_files)
        object.__setattr__(self, "trade_files", trade_files)
        return self


class ReplayMaterializationSpec(_StrictSpec):
    """Frozen order/snapshot identities and the silver files that complete them."""

    orders: tuple[IntendedOrderSpec, ...] = ()
    decision_snapshots: tuple[DecisionSnapshotSpec, ...] = ()
    event_sources: tuple[ReplayEventSourceSpec, ...] = ()
    config: ReplayConfigSpec = ReplayConfigSpec()

    @model_validator(mode="after")
    def validate_identity_sets(self) -> ReplayMaterializationSpec:
        order_ids = tuple(item.order_id for item in self.orders)
        if order_ids != tuple(sorted(set(order_ids))):
            raise ValueError("replay materialization order ids must be unique and sorted")
        order_symbols = {item.ticker.strip().upper() for item in self.orders}
        snapshot_symbols = tuple(item.ticker.strip().upper() for item in self.decision_snapshots)
        source_symbols = tuple(item.symbol for item in self.event_sources)
        if snapshot_symbols != tuple(sorted(set(snapshot_symbols))):
            raise ValueError("decision snapshots must have unique sorted symbols")
        if source_symbols != tuple(sorted(set(source_symbols))):
            raise ValueError("replay event sources must have unique sorted symbols")
        if set(snapshot_symbols) != order_symbols or set(source_symbols) != order_symbols:
            raise ValueError("orders, decision snapshots, and event sources must share symbol sets")
        snapshots = {
            item.ticker.strip().upper(): item.to_domain() for item in self.decision_snapshots
        }
        for item in self.orders:
            order = item.to_domain()
            snapshot = snapshots[order.ticker]
            if snapshot.observed_at > order.decision_time:
                raise ValueError("decision snapshot was observed after the order decision")
        self.config.to_domain()
        return self

    def resolve_paths(self, base_directory: Path) -> ReplayMaterializationSpec:
        """Resolve source files relative to the immutable materialization spec."""
        return self.model_copy(
            update={
                "event_sources": tuple(
                    source.model_copy(
                        update={
                            "quote_files": tuple(
                                path if path.is_absolute() else base_directory / path
                                for path in source.quote_files
                            ),
                            "trade_files": tuple(
                                path if path.is_absolute() else base_directory / path
                                for path in source.trade_files
                            ),
                        }
                    )
                    for source in self.event_sources
                )
            }
        )


@dataclass(frozen=True)
class ReplaySpecArtifact:
    """Content identity and filtering audit for one materialized order."""

    order_id: str
    symbol: str
    file_name: str
    sha256: str
    source_quote_count: int
    replay_quote_count: int
    rejected_one_sided_quote_count: int
    source_trade_count: int
    replay_trade_count: int
    rejected_corrected_trade_count: int
    rejected_subshare_trade_count: int
    fractional_share_quantity_discarded: float
    auction_classification_configured: bool


@dataclass(frozen=True)
class ReplayMaterializationManifest:
    """Immutable bridge from silver source hashes to self-contained replay specs."""

    schema_version: int
    input_sha256: str
    artifacts: tuple[ReplaySpecArtifact, ...]

    @property
    def canonical_bytes(self) -> bytes:
        return json.dumps(
            asdict(self),
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
                raise RuntimeError(f"replay materialization collision at {output}") from None


class ReplaySpecMaterializer:
    """Build one canonical replay spec per intended order without future reads."""

    def materialize(
        self,
        spec: ReplayMaterializationSpec,
        *,
        output_dir: Path,
        manifest_output: Path,
    ) -> ReplayMaterializationManifest:
        output_dir.mkdir(parents=True, exist_ok=True)
        snapshots = {item.ticker.strip().upper(): item for item in spec.decision_snapshots}
        sources = {item.symbol: item for item in spec.event_sources}
        artifacts: list[ReplaySpecArtifact] = []
        for order_spec in spec.orders:
            order = order_spec.to_domain()
            source = sources[order.ticker]
            events = ReplayMarketDataLoader().load(
                quote_files=source.quote_files,
                trade_files=source.trade_files,
                symbol=order.ticker,
                start_at=order.submitted_at,
                end_at=order.expires_at,
                opening_auction_condition_codes=source.opening_auction_condition_codes,
            )
            replay_spec = NbboReplaySpec(
                order=order_spec,
                decision_snapshot=snapshots[order.ticker],
                quotes=tuple(NbboQuoteSpec(**asdict(item)) for item in events.quotes),
                trades=tuple(TradePrintSpec(**asdict(item)) for item in events.trades),
                config=spec.config,
            )
            file_name = f"replay-spec-{_slug(order.order_id)}-{replay_spec.sha256[:16]}.json"
            path = output_dir / file_name
            _write_once(path, replay_spec.canonical_bytes, kind="replay spec")
            artifacts.append(
                ReplaySpecArtifact(
                    order_id=order.order_id,
                    symbol=order.ticker,
                    file_name=file_name,
                    sha256=replay_spec.sha256,
                    source_quote_count=events.source_quote_count,
                    replay_quote_count=len(events.quotes),
                    rejected_one_sided_quote_count=events.rejected_one_sided_quote_count,
                    source_trade_count=events.source_trade_count,
                    replay_trade_count=len(events.trades),
                    rejected_corrected_trade_count=events.rejected_corrected_trade_count,
                    rejected_subshare_trade_count=events.rejected_subshare_trade_count,
                    fractional_share_quantity_discarded=events.fractional_share_quantity_discarded,
                    auction_classification_configured=(events.auction_classification_configured),
                )
            )
        manifest = ReplayMaterializationManifest(
            schema_version=1,
            input_sha256=_semantic_input_sha256(spec),
            artifacts=tuple(artifacts),
        )
        manifest.write(manifest_output)
        return manifest


def _semantic_input_sha256(spec: ReplayMaterializationSpec) -> str:
    sources = []
    for source in spec.event_sources:
        sources.append(
            {
                "symbol": source.symbol,
                "quote_sha256": tuple(_file_sha256(path) for path in source.quote_files),
                "trade_sha256": tuple(_file_sha256(path) for path in source.trade_files),
                "opening_auction_condition_codes": sorted(source.opening_auction_condition_codes),
            }
        )
    payload = {
        "orders": [item.model_dump(mode="json") for item in spec.orders],
        "decision_snapshots": [item.model_dump(mode="json") for item in spec.decision_snapshots],
        "event_sources": sources,
        "config": spec.config.model_dump(mode="json"),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_once(path: Path, encoded: bytes, *, kind: str) -> None:
    try:
        with path.open("xb") as destination:
            destination.write(encoded)
    except FileExistsError:
        if path.read_bytes() != encoded:
            raise RuntimeError(f"{kind} collision at {path}") from None


def _slug(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip()).strip("-")
    return (normalized or "order")[:64]
