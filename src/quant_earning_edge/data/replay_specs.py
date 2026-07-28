"""Materialize causal replay specifications from frozen orders and silver events."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import pyarrow.parquet as pq
from pydantic import BaseModel, ConfigDict, Field, model_validator

from quant_earning_edge.backtest.nbbo_replay import replay_order
from quant_earning_edge.backtest.nbbo_spec import (
    DecisionSnapshotSpec,
    IntendedOrderSpec,
    NbboQuoteSpec,
    NbboReplayEvidence,
    NbboReplaySpec,
    ReplayConfigSpec,
    TradePrintSpec,
)
from quant_earning_edge.data.market_events import ReplayMarketDataLoader

if TYPE_CHECKING:
    from collections.abc import Sequence

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


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

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("unsupported replay materialization schema")
        if not _SHA256_PATTERN.fullmatch(self.input_sha256):
            raise ValueError("invalid replay materialization input hash")
        identities = tuple(item.order_id for item in self.artifacts)
        if identities != tuple(sorted(set(identities))):
            raise ValueError("replay artifacts must have unique sorted order ids")
        file_names = tuple(item.file_name for item in self.artifacts)
        if len(file_names) != len(set(file_names)):
            raise ValueError("replay artifact filenames must be unique")
        for artifact in self.artifacts:
            if not artifact.order_id.strip() or not artifact.symbol.strip():
                raise ValueError("replay artifact identities must not be blank")
            if Path(artifact.file_name).name != artifact.file_name:
                raise ValueError("replay artifact filenames must be safe leaf names")
            if not _SHA256_PATTERN.fullmatch(artifact.sha256):
                raise ValueError("invalid replay artifact hash")
            counts = (
                artifact.source_quote_count,
                artifact.replay_quote_count,
                artifact.rejected_one_sided_quote_count,
                artifact.source_trade_count,
                artifact.replay_trade_count,
                artifact.rejected_corrected_trade_count,
                artifact.rejected_subshare_trade_count,
            )
            if any(item < 0 for item in counts):
                raise ValueError("replay artifact counts cannot be negative")
            if artifact.fractional_share_quantity_discarded < 0:
                raise ValueError("discarded fractional quantity cannot be negative")

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

    @classmethod
    def load(cls, path: Path) -> ReplayMaterializationManifest:
        """Load strict canonical materialization evidence."""
        try:
            raw = json.loads(path.read_bytes())
            if set(raw) != {"schema_version", "input_sha256", "artifacts"}:
                raise ValueError("unsupported replay materialization fields")
            manifest = cls(
                schema_version=int(raw["schema_version"]),
                input_sha256=str(raw["input_sha256"]),
                artifacts=tuple(ReplaySpecArtifact(**item) for item in raw["artifacts"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"invalid replay materialization manifest: {path}") from error
        if json.loads(manifest.canonical_bytes) != raw:
            raise ValueError("replay materialization manifest is not canonical")
        return manifest


@dataclass(frozen=True)
class ReplayEvidenceIndex:
    """Immutable index linking materialized specs to replay evidence files."""

    schema_version: int
    materialization_manifest_sha256: str
    evidence_files: tuple[str, ...]
    evidence_sha256: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("unsupported replay evidence index schema")
        if not _SHA256_PATTERN.fullmatch(self.materialization_manifest_sha256):
            raise ValueError("invalid replay materialization manifest hash")
        if len(self.evidence_files) != len(self.evidence_sha256):
            raise ValueError("replay evidence filenames and hashes must align")
        if len(self.evidence_files) != len(set(self.evidence_files)):
            raise ValueError("replay evidence filenames must be unique")
        if any(Path(item).name != item for item in self.evidence_files):
            raise ValueError("replay evidence filenames must be safe leaf names")
        if any(not _SHA256_PATTERN.fullmatch(item) for item in self.evidence_sha256):
            raise ValueError("invalid replay evidence hash")

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
        _write_once(output, self.canonical_bytes, kind="replay evidence index")

    @classmethod
    def load(cls, path: Path) -> ReplayEvidenceIndex:
        """Load a strict canonical replay evidence index."""
        try:
            raw = json.loads(path.read_bytes())
            if set(raw) != {
                "schema_version",
                "materialization_manifest_sha256",
                "evidence_files",
                "evidence_sha256",
            }:
                raise ValueError("unsupported replay evidence index fields")
            index = cls(
                schema_version=int(raw["schema_version"]),
                materialization_manifest_sha256=str(raw["materialization_manifest_sha256"]),
                evidence_files=tuple(str(item) for item in raw["evidence_files"]),
                evidence_sha256=tuple(str(item) for item in raw["evidence_sha256"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"invalid replay evidence index: {path}") from error
        if json.loads(index.canonical_bytes) != raw:
            raise ValueError("replay evidence index is not canonical")
        return index


class ReplayManifestRunner:
    """Replay every verified spec in one materialization manifest."""

    def run(
        self,
        manifest: ReplayMaterializationManifest,
        *,
        spec_directory: Path,
        output_directory: Path,
        index_output: Path,
    ) -> ReplayEvidenceIndex:
        output_directory.mkdir(parents=True, exist_ok=True)
        evidence_files: list[str] = []
        evidence_hashes: list[str] = []
        for artifact in manifest.artifacts:
            spec_path = spec_directory / artifact.file_name
            encoded_spec = spec_path.read_bytes()
            replay_spec = NbboReplaySpec.model_validate_json(encoded_spec)
            if replay_spec.sha256 != artifact.sha256:
                raise ValueError(f"replay spec hash differs from manifest: {spec_path}")
            if replay_spec.canonical_bytes != encoded_spec:
                raise ValueError(f"replay spec is not canonical: {spec_path}")
            if (
                replay_spec.order.order_id != artifact.order_id
                or replay_spec.order.ticker.strip().upper() != artifact.symbol
            ):
                raise ValueError(f"replay spec identity differs from manifest: {spec_path}")
            order, snapshot, quotes, trades, config = replay_spec.domain_inputs()
            result = replay_order(
                order,
                decision_snapshot=snapshot,
                quotes=quotes,
                trades=trades,
                config=config,
            )
            evidence = NbboReplayEvidence.build(spec=replay_spec, result=result)
            file_name = f"replay-evidence-{_slug(order.order_id)}-{evidence.sha256[:16]}.json"
            evidence.write(output_directory / file_name)
            evidence_files.append(file_name)
            evidence_hashes.append(evidence.sha256)
        index = ReplayEvidenceIndex(
            schema_version=1,
            materialization_manifest_sha256=manifest.sha256,
            evidence_files=tuple(evidence_files),
            evidence_sha256=tuple(evidence_hashes),
        )
        index.write(index_output)
        return index


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
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as destination:
            destination.write(encoded)
    except FileExistsError:
        if path.read_bytes() != encoded:
            raise RuntimeError(f"{kind} collision at {path}") from None


def _slug(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip()).strip("-")
    return (normalized or "order")[:64]


def replay_sources_from_files(
    *,
    quote_files: Sequence[Path],
    trade_files: Sequence[Path],
    expected_symbols: Sequence[str],
    opening_auction_condition_codes: frozenset[int] = frozenset(),
) -> tuple[ReplayEventSourceSpec, ...]:
    """Group silver files by their single stored symbol without path conventions."""
    normalized_expected = tuple(sorted({item.strip().upper() for item in expected_symbols}))
    if any(not item for item in normalized_expected):
        raise ValueError("expected replay symbols must not be blank")
    quotes = _group_files_by_symbol(quote_files, kind="quote")
    trades = _group_files_by_symbol(trade_files, kind="trade")
    if set(quotes) != set(normalized_expected):
        raise ValueError("quote-file symbols do not exactly match frozen order symbols")
    if not set(trades).issubset(normalized_expected):
        raise ValueError("trade-file symbol is not present in frozen orders")
    return tuple(
        ReplayEventSourceSpec(
            symbol=symbol,
            quote_files=tuple(quotes[symbol]),
            trade_files=tuple(trades.get(symbol, ())),
            opening_auction_condition_codes=opening_auction_condition_codes,
        )
        for symbol in normalized_expected
    )


def _group_files_by_symbol(
    files: Sequence[Path],
    *,
    kind: str,
) -> dict[str, list[Path]]:
    grouped: dict[str, list[Path]] = {}
    for path in sorted(set(files), key=str):
        table = pq.read_table(path, columns=["symbol"])  # type: ignore[no-untyped-call]
        symbols = {
            str(item).strip().upper()
            for item in table.column("symbol").to_pylist()
            if item is not None
        }
        if len(symbols) != 1:
            raise ValueError(f"{kind} file must contain exactly one symbol: {path}")
        symbol = next(iter(symbols))
        grouped.setdefault(symbol, []).append(path)
    return grouped
