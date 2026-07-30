"""Source-bound reconstruction of Polygon minute-bar Silver files."""

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
class MinuteBarsSourceManifest:
    """Exact Polygon pages behind one minute-bar Silver artifact."""

    path: Path
    raw: dict[str, Any]

    @classmethod
    def load(cls, path: Path) -> MinuteBarsSourceManifest:
        try:
            encoded = path.read_bytes()
            raw = json.loads(encoded)
        except (OSError, ValueError) as error:
            raise ValueError(f"invalid minute-bars source manifest: {path}") from error
        required = {
            "schema_version",
            "symbol",
            "start_at",
            "end_at",
            "event_date",
            "ingested_at",
            "silver_file",
            "provider_observations",
        }
        if not isinstance(raw, dict) or set(raw) != required or raw["schema_version"] != 1:
            raise ValueError("minute-bars source manifest schema mismatch")
        providers = raw["provider_observations"]
        if (
            not isinstance(raw["symbol"], str)
            or not raw["symbol"]
            or raw["symbol"] != raw["symbol"].strip().upper()
            or not isinstance(raw["silver_file"], dict)
            or not isinstance(providers, list)
            or not providers
        ):
            raise ValueError("minute-bars source manifest collections are invalid")
        entries = (raw["silver_file"], *providers)
        for entry in entries:
            _validate_entry(entry)
        identities = tuple(entry["path"] for entry in entries)
        if len(identities) != len(set(identities)):
            raise ValueError("minute-bars source manifest paths are duplicated")
        if tuple(providers) != tuple(sorted(providers, key=lambda item: item["path"])):
            raise ValueError("minute-bars source observations are not sorted")
        try:
            start_at = datetime.fromisoformat(str(raw["start_at"]))
            end_at = datetime.fromisoformat(str(raw["end_at"]))
            event_date = date.fromisoformat(str(raw["event_date"]))
            ingested_at = datetime.fromisoformat(str(raw["ingested_at"]))
        except ValueError as error:
            raise ValueError("minute-bars source timestamps are invalid") from error
        if (
            any(
                item.tzinfo is None or item.utcoffset() is None
                for item in (start_at, end_at, ingested_at)
            )
            or end_at <= start_at
            or event_date.isoformat() != raw["event_date"]
            or start_at.isoformat() != raw["start_at"]
            or end_at.isoformat() != raw["end_at"]
            or ingested_at.isoformat() != raw["ingested_at"]
        ):
            raise ValueError("minute-bars source metadata is invalid")
        if json.dumps(raw, sort_keys=True, separators=(",", ":")).encode() != encoded:
            raise ValueError("minute-bars source manifest is not canonical")
        return cls(path=path.resolve(), raw=raw)

    def silver_path(self, *, data_lake_root: Path) -> Path:
        return _resolve_entries(
            (self.raw["silver_file"],),
            data_lake_root=data_lake_root,
        )[0]

    def provider_paths(self, *, data_lake_root: Path) -> tuple[Path, ...]:
        return _resolve_entries(
            tuple(self.raw["provider_observations"]),
            data_lake_root=data_lake_root,
        )


class MinuteBarsSourceCapture:
    """Persist and independently reproduce raw-to-Silver minute bars."""

    def __init__(self, layout: LakehouseLayout) -> None:
        self._layout = layout

    def write(
        self,
        *,
        symbol: str,
        start_at: datetime,
        end_at: datetime,
        event_date: date,
        ingested_at: datetime,
        silver_file: SilverArtifact,
        provider_observations: Sequence[BronzeArtifact],
    ) -> MinuteBarsSourceManifest:
        normalized_symbol = symbol.strip().upper()
        if (
            not normalized_symbol
            or any(
                item.tzinfo is None or item.utcoffset() is None
                for item in (start_at, end_at, ingested_at)
            )
            or end_at <= start_at
            or not provider_observations
        ):
            raise ValueError("minute-bars source capture metadata is invalid")
        if any(
            _path_partition(item.path, prefix="source=") != "polygon"
            or _path_partition(item.path, prefix="dataset=") != "minute-aggregate-bars"
            for item in provider_observations
        ):
            raise ValueError("minute-bars source contains a non-Polygon observation")
        raw = {
            "schema_version": 1,
            "symbol": normalized_symbol,
            "start_at": start_at.isoformat(),
            "end_at": end_at.isoformat(),
            "event_date": event_date.isoformat(),
            "ingested_at": ingested_at.isoformat(),
            "silver_file": self._entry(silver_file.path),
            "provider_observations": self._entries(item.path for item in provider_observations),
        }
        encoded = json.dumps(raw, sort_keys=True, separators=(",", ":")).encode()
        digest = hashlib.sha256(encoded).hexdigest()
        root = self._layout.root / "manifests" / "minute-bars-sources"
        root.mkdir(parents=True, exist_ok=True)
        path = root / f"source-{digest[:20]}.json"
        try:
            with path.open("xb") as destination:
                destination.write(encoded)
        except FileExistsError:
            if path.read_bytes() != encoded:
                raise RuntimeError(f"minute-bars source manifest collision at {path}") from None
        return MinuteBarsSourceManifest.load(path)

    @staticmethod
    def find_for_files(
        minute_bar_files: Sequence[Path],
        *,
        data_lake_root: Path,
    ) -> tuple[MinuteBarsSourceManifest, ...]:
        targets = {path.resolve() for path in minute_bar_files}
        matches: dict[Path, list[MinuteBarsSourceManifest]] = {path: [] for path in targets}
        root = data_lake_root.resolve() / "manifests" / "minute-bars-sources"
        for path in sorted(root.glob("source-*.json")):
            manifest = MinuteBarsSourceManifest.load(path)
            target = manifest.silver_path(data_lake_root=data_lake_root)
            if target in matches:
                manifest.provider_paths(data_lake_root=data_lake_root)
                matches[target].append(manifest)
        if not matches or any(len(items) != 1 for items in matches.values()):
            raise ValueError("minute-bar Silver files lack unique Polygon source lineage")
        return tuple(
            sorted(
                (items[0] for items in matches.values()),
                key=lambda item: str(item.path),
            )
        )

    @staticmethod
    def reproduce(
        manifest: MinuteBarsSourceManifest,
        *,
        data_lake_root: Path,
        output_layout: LakehouseLayout,
    ) -> SilverArtifact:
        symbol = manifest.raw["symbol"]
        start_at = datetime.fromisoformat(manifest.raw["start_at"])
        end_at = datetime.fromisoformat(manifest.raw["end_at"])
        event_date = date.fromisoformat(manifest.raw["event_date"])
        ingested_at = datetime.fromisoformat(manifest.raw["ingested_at"])
        raws = tuple(
            json.loads(path.read_bytes())
            for path in manifest.provider_paths(data_lake_root=data_lake_root)
        )
        bars = PolygonClient.minute_bars_from_payloads(raws, symbol=symbol)
        timestamps = tuple(item.timestamp for item in bars)
        if (
            len(timestamps) != len(set(timestamps))
            or any(not start_at <= item <= end_at for item in timestamps)
            or not bars
        ):
            raise ValueError("minute-bars provider observations differ from captured metadata")
        reproduced = SilverWriter(output_layout).write_minute_bars(
            bars,
            event_date=event_date,
            ingested_at=ingested_at,
        )
        expected = manifest.silver_path(data_lake_root=data_lake_root)
        if reproduced.path.read_bytes() != expected.read_bytes():
            raise ValueError("minute-bar Silver differs from retained Polygon observations")
        return reproduced

    def _entries(self, paths: Iterable[Path]) -> list[dict[str, str]]:
        return [self._entry(path) for path in sorted(paths)]

    def _entry(self, path: Path) -> dict[str, str]:
        resolved = path.resolve()
        try:
            relative = resolved.relative_to(self._layout.root.resolve())
        except ValueError as error:
            raise ValueError("minute-bars source must be inside the data lake") from error
        return {"path": relative.as_posix(), "sha256": _file_sha256(resolved)}


def _resolve_entries(
    entries: tuple[dict[str, str], ...],
    *,
    data_lake_root: Path,
) -> tuple[Path, ...]:
    root = data_lake_root.resolve()
    paths = tuple((root / entry["path"]).resolve() for entry in entries)
    if any(path == root or root not in path.parents for path in paths):
        raise ValueError("minute-bars source escapes the data lake")
    try:
        valid = all(
            _file_sha256(path) == entry["sha256"]
            for path, entry in zip(paths, entries, strict=True)
        )
    except OSError as error:
        raise ValueError("minute-bars source file is missing or differs") from error
    if not valid:
        raise ValueError("minute-bars source file is missing or differs")
    return paths


def _validate_entry(entry: object) -> None:
    if (
        not isinstance(entry, dict)
        or set(entry) != {"path", "sha256"}
        or not _is_sha256(entry.get("sha256"))
    ):
        raise ValueError("minute-bars source manifest entry is invalid")
    path = PurePosixPath(str(entry.get("path", "")))
    if (
        path.is_absolute()
        or ".." in path.parts
        or not path.parts
        or path.as_posix() != entry["path"]
    ):
        raise ValueError("minute-bars source manifest path is invalid")


def _path_partition(path: Path, *, prefix: str) -> str:
    matches = tuple(part.removeprefix(prefix) for part in path.parts if part.startswith(prefix))
    if len(matches) != 1:
        raise ValueError(f"minute-bars source path has ambiguous {prefix.rstrip('=')}: {path}")
    return matches[0]


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
