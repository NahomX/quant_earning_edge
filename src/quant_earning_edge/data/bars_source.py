"""Source-bound reconstruction of Polygon adjusted daily-bar Silver files."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any

from quant_earning_edge.data.clients import PolygonClient
from quant_earning_edge.data.silver import SilverWriter

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from quant_earning_edge.data.bronze import BronzeArtifact
    from quant_earning_edge.data.layout import LakehouseLayout
    from quant_earning_edge.data.silver import SilverArtifact


@dataclass(frozen=True)
class DailyBarsSourceManifest:
    """Exact Polygon responses behind one adjusted daily-bar materialization."""

    path: Path
    raw: dict[str, Any]

    @classmethod
    def load(cls, path: Path) -> DailyBarsSourceManifest:
        try:
            encoded = path.read_bytes()
            raw = json.loads(encoded)
        except (OSError, ValueError) as error:
            raise ValueError(f"invalid daily-bars source manifest: {path}") from error
        required = {
            "schema_version",
            "symbols",
            "start_date",
            "end_date",
            "ingested_at",
            "silver_files",
            "provider_observations",
        }
        if not isinstance(raw, dict) or set(raw) != required or raw["schema_version"] != 1:
            raise ValueError("daily-bars source manifest schema mismatch")
        symbols = raw["symbols"]
        silver = raw["silver_files"]
        providers = raw["provider_observations"]
        if (
            not isinstance(symbols, list)
            or not symbols
            or symbols != sorted(set(symbols))
            or any(not isinstance(item, str) or not item for item in symbols)
            or not isinstance(silver, list)
            or not silver
            or not isinstance(providers, list)
            or not providers
        ):
            raise ValueError("daily-bars source manifest collections are invalid")
        entries = (*silver, *providers)
        for entry in entries:
            _validate_entry(entry)
        identities = tuple(entry["path"] for entry in entries)
        if len(identities) != len(set(identities)):
            raise ValueError("daily-bars source manifest paths are duplicated")
        if tuple(silver) != tuple(sorted(silver, key=lambda item: item["path"])) or tuple(
            providers
        ) != tuple(sorted(providers, key=lambda item: item["path"])):
            raise ValueError("daily-bars source manifest entries are not sorted")
        try:
            start_date = date.fromisoformat(str(raw["start_date"]))
            end_date = date.fromisoformat(str(raw["end_date"]))
            ingested_at = datetime.fromisoformat(str(raw["ingested_at"]))
        except ValueError as error:
            raise ValueError("daily-bars source manifest timestamps are invalid") from error
        if (
            start_date.isoformat() != raw["start_date"]
            or end_date.isoformat() != raw["end_date"]
            or end_date < start_date
            or ingested_at.tzinfo is None
            or ingested_at.utcoffset() is None
            or ingested_at.isoformat() != raw["ingested_at"]
        ):
            raise ValueError("daily-bars source manifest metadata is invalid")
        if json.dumps(raw, sort_keys=True, separators=(",", ":")).encode() != encoded:
            raise ValueError("daily-bars source manifest is not canonical")
        return cls(path=path.resolve(), raw=raw)

    @property
    def silver_entries(self) -> tuple[dict[str, str], ...]:
        return tuple(self.raw["silver_files"])

    @property
    def provider_entries(self) -> tuple[dict[str, str], ...]:
        return tuple(self.raw["provider_observations"])

    def silver_paths(self, *, data_lake_root: Path) -> tuple[Path, ...]:
        return _resolve_entries(self.silver_entries, data_lake_root=data_lake_root)

    def provider_paths(self, *, data_lake_root: Path) -> tuple[Path, ...]:
        return _resolve_entries(self.provider_entries, data_lake_root=data_lake_root)


class DailyBarsSourceCapture:
    """Persist and independently reproduce raw-to-Silver daily-bar lineage."""

    def __init__(self, layout: LakehouseLayout) -> None:
        self._layout = layout

    def write(
        self,
        *,
        symbols: Sequence[str],
        start_date: date,
        end_date: date,
        ingested_at: datetime,
        silver_files: Sequence[SilverArtifact],
        provider_observations: Sequence[BronzeArtifact],
    ) -> DailyBarsSourceManifest:
        normalized_symbols = tuple(sorted({item.strip().upper() for item in symbols}))
        if (
            not normalized_symbols
            or any(not item for item in normalized_symbols)
            or end_date < start_date
            or ingested_at.tzinfo is None
            or ingested_at.utcoffset() is None
            or not silver_files
            or not provider_observations
        ):
            raise ValueError("daily-bars source capture metadata is invalid")
        if any(
            _path_partition(item.path, prefix="source=") != "polygon"
            or _path_partition(item.path, prefix="dataset=") != "daily-aggregate-bars"
            for item in provider_observations
        ):
            raise ValueError("daily-bars source contains a non-Polygon observation")
        raw = {
            "schema_version": 1,
            "symbols": list(normalized_symbols),
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "ingested_at": ingested_at.isoformat(),
            "silver_files": self._entries(item.path for item in silver_files),
            "provider_observations": self._entries(item.path for item in provider_observations),
        }
        encoded = json.dumps(raw, sort_keys=True, separators=(",", ":")).encode()
        digest = hashlib.sha256(encoded).hexdigest()
        root = self._layout.root / "manifests" / "daily-bars-sources"
        root.mkdir(parents=True, exist_ok=True)
        path = root / f"source-{digest[:20]}.json"
        try:
            with path.open("xb") as destination:
                destination.write(encoded)
        except FileExistsError:
            if path.read_bytes() != encoded:
                raise RuntimeError(f"daily-bars source manifest collision at {path}") from None
        return DailyBarsSourceManifest.load(path)

    @staticmethod
    def find_for_files(
        daily_bar_files: Sequence[Path],
        *,
        data_lake_root: Path,
    ) -> tuple[DailyBarsSourceManifest, ...]:
        """Resolve exactly one retained source manifest per daily-bar file."""
        targets = {path.resolve() for path in daily_bar_files}
        matches: dict[Path, list[DailyBarsSourceManifest]] = {path: [] for path in targets}
        source_root = data_lake_root.resolve() / "manifests" / "daily-bars-sources"
        for path in sorted(source_root.glob("source-*.json")):
            manifest = DailyBarsSourceManifest.load(path)
            manifest_paths = set(manifest.silver_paths(data_lake_root=data_lake_root))
            for target in targets & manifest_paths:
                manifest.provider_paths(data_lake_root=data_lake_root)
                matches[target].append(manifest)
        if not matches or any(len(items) != 1 for items in matches.values()):
            raise ValueError("daily-bar Silver files lack unique Polygon source lineage")
        return tuple(
            sorted(
                {items[0].path: items[0] for items in matches.values()}.values(),
                key=lambda item: str(item.path),
            )
        )

    @staticmethod
    def reproduce(
        manifest: DailyBarsSourceManifest,
        *,
        data_lake_root: Path,
        output_layout: LakehouseLayout,
    ) -> tuple[SilverArtifact, ...]:
        start_date = date.fromisoformat(manifest.raw["start_date"])
        end_date = date.fromisoformat(manifest.raw["end_date"])
        ingested_at = datetime.fromisoformat(manifest.raw["ingested_at"])
        expected_symbols = set(manifest.raw["symbols"])
        bars = []
        observed_symbols: set[str] = set()
        seen: set[tuple[str, datetime]] = set()
        for path in manifest.provider_paths(data_lake_root=data_lake_root):
            raw = json.loads(path.read_bytes())
            if not isinstance(raw, dict) or not isinstance(raw.get("ticker"), str):
                raise ValueError("daily-bars Polygon observation lacks a ticker")
            symbol = raw["ticker"].strip().upper()
            observed_symbols.add(symbol)
            for bar in PolygonClient.daily_bars_from_payload(raw, symbol=symbol):
                key = (bar.symbol, bar.timestamp)
                if key in seen:
                    raise ValueError("daily-bars provider observations contain duplicate bars")
                seen.add(key)
                bars.append(bar)
        if observed_symbols != expected_symbols or any(
            not start_date <= item.session_date <= end_date for item in bars
        ):
            raise ValueError("daily-bars provider observations differ from captured metadata")
        reproduced = SilverWriter(output_layout).write_daily_bars(
            tuple(sorted(bars, key=lambda item: (item.timestamp, item.symbol))),
            ingested_at=ingested_at,
        )
        expected = manifest.silver_paths(data_lake_root=data_lake_root)
        if _byte_digests(item.path for item in reproduced) != _byte_digests(expected):
            raise ValueError("daily-bar Silver differs from retained Polygon observations")
        return reproduced

    def _entries(self, paths: Iterable[Path]) -> list[dict[str, str]]:
        return [self._entry(path) for path in sorted(paths)]

    def _entry(self, path: Path) -> dict[str, str]:
        resolved = path.resolve()
        try:
            relative = resolved.relative_to(self._layout.root.resolve())
        except ValueError as error:
            raise ValueError("daily-bars source must be inside the data lake") from error
        return {"path": relative.as_posix(), "sha256": _file_sha256(resolved)}


def _resolve_entries(
    entries: tuple[dict[str, str], ...],
    *,
    data_lake_root: Path,
) -> tuple[Path, ...]:
    root = data_lake_root.resolve()
    paths = tuple((root / entry["path"]).resolve() for entry in entries)
    if any(path == root or root not in path.parents for path in paths):
        raise ValueError("daily-bars source escapes the data lake")
    try:
        valid = all(
            _file_sha256(path) == entry["sha256"]
            for path, entry in zip(paths, entries, strict=True)
        )
    except OSError as error:
        raise ValueError("daily-bars source file is missing or differs") from error
    if not valid:
        raise ValueError("daily-bars source file is missing or differs")
    return paths


def _validate_entry(entry: object) -> None:
    if (
        not isinstance(entry, dict)
        or set(entry) != {"path", "sha256"}
        or not _is_sha256(entry.get("sha256"))
    ):
        raise ValueError("daily-bars source manifest entry is invalid")
    path = PurePosixPath(str(entry.get("path", "")))
    if (
        path.is_absolute()
        or ".." in path.parts
        or not path.parts
        or path.as_posix() != entry["path"]
    ):
        raise ValueError("daily-bars source manifest path is invalid")


def _path_partition(path: Path, *, prefix: str) -> str:
    matches = tuple(part.removeprefix(prefix) for part in path.parts if part.startswith(prefix))
    if len(matches) != 1:
        raise ValueError(f"daily-bars source path has ambiguous {prefix.rstrip('=')}: {path}")
    return matches[0]


def _byte_digests(paths: Iterable[Path]) -> tuple[str, ...]:
    return tuple(sorted(_file_sha256(path) for path in paths))


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
