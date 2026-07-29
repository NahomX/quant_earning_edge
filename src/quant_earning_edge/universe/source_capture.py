"""Source-bound reconstruction of provider-backed universe snapshots."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any

from quant_earning_edge.data.clients import PolygonClient
from quant_earning_edge.universe.builder import UniverseBuilder
from quant_earning_edge.universe.config import (
    load_halt_snapshot,
    load_universe_job_config,
)
from quant_earning_edge.universe.job import (
    DailyUniverseJob,
    RunTrigger,
    UniverseManifestStore,
)
from quant_earning_edge.universe.snapshot import UniverseSnapshotWriter

if TYPE_CHECKING:
    from collections.abc import Sequence

    from quant_earning_edge.data.bronze import BronzeArtifact
    from quant_earning_edge.data.clients import EquityBar, TickerDetails, TickerReference
    from quant_earning_edge.data.layout import LakehouseLayout
    from quant_earning_edge.universe.snapshot import UniverseSnapshotArtifact

_DATASETS = frozenset({"ticker-reference", "ticker-details", "daily-aggregate-bars"})


@dataclass(frozen=True)
class UniverseSourceCaptureManifest:
    """Strict identities needed to independently rebuild one universe snapshot."""

    path: Path
    raw: dict[str, Any]

    @classmethod
    def load(cls, path: Path) -> UniverseSourceCaptureManifest:
        try:
            encoded = path.read_bytes()
            raw = json.loads(encoded)
        except (OSError, ValueError) as error:
            raise ValueError(f"invalid universe source manifest: {path}") from error
        required = {
            "schema_version",
            "trade_date",
            "asof_date",
            "lookback_start",
            "decision_at",
            "adv_sessions",
            "snapshot_file_sha256",
            "snapshot_semantic_sha256",
            "universe_config",
            "halt_snapshot",
            "provider_observations",
        }
        if not isinstance(raw, dict) or set(raw) != required or raw["schema_version"] != 1:
            raise ValueError("universe source manifest schema mismatch")
        for name in ("universe_config", "halt_snapshot"):
            _validate_source_entry(raw[name])
        observations = raw["provider_observations"]
        if not isinstance(observations, list) or not observations:
            raise ValueError("universe source manifest has no provider observations")
        for entry in observations:
            _validate_source_entry(entry, include_dataset=True)
            if entry["dataset"] not in _DATASETS:
                raise ValueError("universe source manifest provider dataset is unsupported")
        identities = tuple((entry["dataset"], entry["path"]) for entry in observations)
        if identities != tuple(sorted(set(identities))):
            raise ValueError("universe source manifest observations are not unique and sorted")
        if {entry["dataset"] for entry in observations} != _DATASETS:
            raise ValueError("universe source manifest provider datasets are incomplete")
        try:
            trade_date = date.fromisoformat(str(raw["trade_date"]))
            asof_date = date.fromisoformat(str(raw["asof_date"]))
            lookback_start = date.fromisoformat(str(raw["lookback_start"]))
            decision_at = datetime.fromisoformat(str(raw["decision_at"]))
        except ValueError as error:
            raise ValueError("universe source manifest timestamps are invalid") from error
        if (
            trade_date.isoformat() != raw["trade_date"]
            or asof_date.isoformat() != raw["asof_date"]
            or lookback_start.isoformat() != raw["lookback_start"]
            or lookback_start > asof_date
            or decision_at.tzinfo is None
            or decision_at.utcoffset() is None
            or not isinstance(raw["adv_sessions"], int)
            or raw["adv_sessions"] < 1
            or not _is_sha256(raw["snapshot_file_sha256"])
            or not _is_sha256(raw["snapshot_semantic_sha256"])
        ):
            raise ValueError("universe source manifest metadata is invalid")
        canonical = json.dumps(raw, sort_keys=True, separators=(",", ":")).encode()
        if canonical != encoded:
            raise ValueError("universe source manifest is not canonical")
        return cls(path=path.resolve(), raw=raw)

    @property
    def source_entries(self) -> tuple[dict[str, str], ...]:
        return (
            self.raw["universe_config"],
            self.raw["halt_snapshot"],
            *self.raw["provider_observations"],
        )

    def source_paths(self, *, data_lake_root: Path) -> tuple[Path, ...]:
        root = data_lake_root.resolve()
        paths = tuple((root / entry["path"]).resolve() for entry in self.source_entries)
        if any(path == root or root not in path.parents for path in paths):
            raise ValueError("universe source escapes the data lake")
        if any(
            _file_sha256(path) != entry["sha256"]
            for path, entry in zip(paths, self.source_entries, strict=True)
        ):
            raise ValueError("universe source file is missing or differs")
        return paths


class UniverseSourceCapture:
    """Write and independently reproduce exact universe-generation evidence."""

    def __init__(self, layout: LakehouseLayout) -> None:
        self._layout = layout

    def write(
        self,
        *,
        trade_date: date,
        asof_date: date,
        lookback_start: date,
        decision_at: datetime,
        adv_sessions: int,
        snapshot: UniverseSnapshotArtifact,
        universe_config: Path,
        halt_snapshot: Path,
        provider_observations: Sequence[BronzeArtifact],
    ) -> UniverseSourceCaptureManifest:
        if decision_at.tzinfo is None or decision_at.utcoffset() is None:
            raise ValueError("universe source decision_at must be timezone-aware")
        entries = tuple(
            sorted(
                (
                    {
                        "dataset": _dataset_from_path(item.path),
                        **self._source_entry(item.path),
                    }
                    for item in provider_observations
                ),
                key=lambda item: (item["dataset"], item["path"]),
            )
        )
        if {entry["dataset"] for entry in entries} != _DATASETS:
            raise ValueError("universe source capture lacks required provider observations")
        raw = {
            "schema_version": 1,
            "trade_date": trade_date.isoformat(),
            "asof_date": asof_date.isoformat(),
            "lookback_start": lookback_start.isoformat(),
            "decision_at": decision_at.isoformat(),
            "adv_sessions": adv_sessions,
            "snapshot_file_sha256": _file_sha256(snapshot.path),
            "snapshot_semantic_sha256": snapshot.sha256,
            "universe_config": self._source_entry(
                self._retain_input(universe_config, dataset="config")
            ),
            "halt_snapshot": self._source_entry(self._retain_input(halt_snapshot, dataset="halts")),
            "provider_observations": entries,
        }
        encoded = json.dumps(raw, sort_keys=True, separators=(",", ":")).encode()
        digest = hashlib.sha256(encoded).hexdigest()
        path = snapshot.path.with_name(f"source-{digest[:20]}.json")
        try:
            with path.open("xb") as destination:
                destination.write(encoded)
        except FileExistsError:
            if path.read_bytes() != encoded:
                raise RuntimeError(f"universe source manifest collision at {path}") from None
        return UniverseSourceCaptureManifest.load(path)

    def _retain_input(self, path: Path, *, dataset: str) -> Path:
        encoded = path.read_bytes()
        digest = hashlib.sha256(encoded).hexdigest()
        suffix = path.suffix.lower()
        retained = (
            self._layout.root
            / "manifests"
            / "universe-inputs"
            / f"dataset={dataset}"
            / f"{digest}{suffix}"
        )
        retained.parent.mkdir(parents=True, exist_ok=True)
        try:
            with retained.open("xb") as destination:
                destination.write(encoded)
        except FileExistsError:
            if retained.read_bytes() != encoded:
                raise RuntimeError(f"universe retained-input collision at {retained}") from None
        return retained

    @staticmethod
    def reproduce(
        manifest: UniverseSourceCaptureManifest,
        *,
        data_lake_root: Path,
        output_layout: LakehouseLayout,
    ) -> UniverseSnapshotArtifact:
        paths = manifest.source_paths(data_lake_root=data_lake_root)
        config_path, halt_path, *provider_paths = paths
        config = load_universe_job_config(config_path)
        halt = load_halt_snapshot(halt_path)
        market_data = _CapturedUniverseMarketData.from_files(
            tuple(
                (
                    entry["dataset"],
                    path,
                )
                for entry, path in zip(
                    manifest.raw["provider_observations"],
                    provider_paths,
                    strict=True,
                )
            ),
            asof_date=date.fromisoformat(manifest.raw["asof_date"]),
            lookback_start=date.fromisoformat(manifest.raw["lookback_start"]),
        )
        decision_at = datetime.fromisoformat(manifest.raw["decision_at"])
        result = DailyUniverseJob(
            market_data=market_data,
            builder=UniverseBuilder(config.eligibility.to_domain()),
            snapshot_writer=UniverseSnapshotWriter(output_layout),
            manifest_store=UniverseManifestStore(output_layout),
            adv_sessions=int(manifest.raw["adv_sessions"]),
            clock=lambda: decision_at,
            run_id_factory=lambda: "reconstruction",
        ).run(
            trade_date=date.fromisoformat(manifest.raw["trade_date"]),
            asof_date=date.fromisoformat(manifest.raw["asof_date"]),
            lookback_start=date.fromisoformat(manifest.raw["lookback_start"]),
            halt_snapshot=halt,
            trigger=RunTrigger.SCHEDULED,
        )
        reproduced_file_sha256 = _file_sha256(result.snapshot.path)
        if (
            result.snapshot.sha256 != manifest.raw["snapshot_semantic_sha256"]
            or reproduced_file_sha256 != manifest.raw["snapshot_file_sha256"]
        ):
            raise ValueError(
                "universe snapshot differs from captured provider inputs: "
                f"semantic={result.snapshot.sha256}, file={reproduced_file_sha256}"
            )
        return result.snapshot

    def _source_entry(self, path: Path) -> dict[str, str]:
        resolved = path.resolve()
        try:
            relative = resolved.relative_to(self._layout.root.resolve())
        except ValueError as error:
            raise ValueError("universe source must be inside the data lake") from error
        return {"path": relative.as_posix(), "sha256": _file_sha256(resolved)}


@dataclass(frozen=True)
class _CapturedUniverseMarketData:
    references: tuple[TickerReference, ...]
    details: dict[str, TickerDetails]
    bars: dict[str, tuple[EquityBar, ...]]
    asof_date: date
    lookback_start: date

    @classmethod
    def from_files(
        cls,
        files: tuple[tuple[str, Path], ...],
        *,
        asof_date: date,
        lookback_start: date,
    ) -> _CapturedUniverseMarketData:
        references: list[TickerReference] = []
        details: dict[str, TickerDetails] = {}
        bars: dict[str, list[EquityBar]] = {}
        for dataset, path in files:
            raw = json.loads(path.read_bytes())
            if dataset == "ticker-reference":
                references.extend(
                    PolygonClient.ticker_references_from_payload(raw, asof_date=asof_date)
                )
            elif dataset == "ticker-details":
                symbol = str(raw.get("results", {}).get("ticker", "")).strip().upper()
                if symbol in details:
                    raise ValueError(f"duplicate retained ticker details: {symbol}")
                details[symbol] = PolygonClient.ticker_details_from_payload(
                    raw,
                    symbol=symbol,
                    asof_date=asof_date,
                )
            elif dataset == "daily-aggregate-bars":
                symbol = str(raw.get("ticker", "")).strip().upper()
                bars.setdefault(symbol, []).extend(
                    PolygonClient.daily_bars_from_payload(raw, symbol=symbol)
                )
        ordered_references = tuple(sorted(references, key=lambda item: item.symbol))
        symbols = tuple(item.symbol for item in ordered_references)
        if len(symbols) != len(set(symbols)):
            raise ValueError("retained ticker-reference pages overlap")
        if set(details) != set(symbols) or set(bars) != set(symbols):
            raise ValueError("retained universe provider observations are incomplete")
        normalized_bars = {}
        for symbol, values in bars.items():
            ordered = tuple(sorted(values, key=lambda item: item.timestamp))
            dates = tuple(item.session_date for item in ordered)
            if len(dates) != len(set(dates)) or any(
                not lookback_start <= item <= asof_date for item in dates
            ):
                raise ValueError(f"retained daily bars are invalid for {symbol}")
            normalized_bars[symbol] = ordered
        return cls(
            references=ordered_references,
            details=details,
            bars=normalized_bars,
            asof_date=asof_date,
            lookback_start=lookback_start,
        )

    def list_tickers(
        self,
        *,
        asof_date: date,
        active: bool = True,
    ) -> tuple[TickerReference, ...]:
        if asof_date != self.asof_date or not active:
            raise ValueError("captured ticker-reference request differs")
        return self.references

    def ticker_details(self, *, symbol: str, asof_date: date) -> TickerDetails:
        if asof_date != self.asof_date:
            raise ValueError("captured ticker-details date differs")
        return self.details[symbol]

    def daily_bars(
        self,
        *,
        symbol: str,
        start_date: date,
        end_date: date,
    ) -> tuple[EquityBar, ...]:
        if start_date != self.lookback_start or end_date != self.asof_date:
            raise ValueError("captured daily-bar request interval differs")
        return self.bars[symbol]


def _validate_source_entry(entry: object, *, include_dataset: bool = False) -> None:
    keys = {"path", "sha256", *(("dataset",) if include_dataset else ())}
    if not isinstance(entry, dict) or set(entry) != keys or not _is_sha256(entry.get("sha256")):
        raise ValueError("universe source manifest entry is invalid")
    source_path = PurePosixPath(str(entry.get("path", "")))
    if (
        source_path.is_absolute()
        or ".." in source_path.parts
        or not source_path.parts
        or source_path.as_posix() != entry["path"]
    ):
        raise ValueError("universe source manifest path is invalid")


def _dataset_from_path(path: Path) -> str:
    datasets = tuple(
        part.removeprefix("dataset=") for part in path.parts if part.startswith("dataset=")
    )
    if len(datasets) != 1 or datasets[0] not in _DATASETS:
        raise ValueError(f"unsupported universe provider observation path: {path}")
    return datasets[0]


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(item in "0123456789abcdef" for item in value)
    )
