"""Source-bound reconstruction of Finnhub earnings Silver partitions."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any

from quant_earning_edge.data.clients import FinnhubClient
from quant_earning_edge.data.silver import SilverWriter

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from quant_earning_edge.data.bronze import BronzeArtifact
    from quant_earning_edge.data.layout import LakehouseLayout
    from quant_earning_edge.data.silver import SilverArtifact


@dataclass(frozen=True)
class EarningsSourceManifest:
    """Exact Finnhub responses behind one earnings ingestion interval."""

    path: Path
    raw: dict[str, Any]

    @classmethod
    def load(cls, path: Path) -> EarningsSourceManifest:
        try:
            encoded = path.read_bytes()
            raw = json.loads(encoded)
        except (OSError, ValueError) as error:
            raise ValueError(f"invalid earnings source manifest: {path}") from error
        required = {
            "schema_version",
            "start_date",
            "end_date",
            "ingested_at",
            "silver_files",
            "provider_observations",
        }
        if not isinstance(raw, dict) or set(raw) != required or raw["schema_version"] != 1:
            raise ValueError("earnings source manifest schema mismatch")
        silver = raw["silver_files"]
        providers = raw["provider_observations"]
        if (
            not isinstance(silver, list)
            or not silver
            or not isinstance(providers, list)
            or not providers
        ):
            raise ValueError("earnings source manifest collections are invalid")
        entries = (*silver, *providers)
        for entry in entries:
            _validate_entry(entry)
        identities = tuple(entry["path"] for entry in entries)
        if len(identities) != len(set(identities)):
            raise ValueError("earnings source manifest paths are duplicated")
        if tuple(silver) != tuple(sorted(silver, key=lambda item: item["path"])) or tuple(
            providers
        ) != tuple(sorted(providers, key=lambda item: item["path"])):
            raise ValueError("earnings source manifest entries are not sorted")
        try:
            start_date = date.fromisoformat(str(raw["start_date"]))
            end_date = date.fromisoformat(str(raw["end_date"]))
            ingested_at = datetime.fromisoformat(str(raw["ingested_at"]))
        except ValueError as error:
            raise ValueError("earnings source manifest timestamps are invalid") from error
        if (
            start_date.isoformat() != raw["start_date"]
            or end_date.isoformat() != raw["end_date"]
            or end_date < start_date
            or ingested_at.tzinfo is None
            or ingested_at.utcoffset() is None
        ):
            raise ValueError("earnings source manifest metadata is invalid")
        canonical = json.dumps(raw, sort_keys=True, separators=(",", ":")).encode()
        if canonical != encoded:
            raise ValueError("earnings source manifest is not canonical")
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


class EarningsSourceCapture:
    """Persist and independently reproduce raw-to-Silver earnings lineage."""

    def __init__(self, layout: LakehouseLayout) -> None:
        self._layout = layout

    def write(
        self,
        *,
        start_date: date,
        end_date: date,
        ingested_at: datetime,
        silver_files: Sequence[SilverArtifact],
        provider_observations: Sequence[BronzeArtifact],
    ) -> EarningsSourceManifest:
        if (
            end_date < start_date
            or ingested_at.tzinfo is None
            or ingested_at.utcoffset() is None
            or not silver_files
            or not provider_observations
        ):
            raise ValueError("earnings source capture metadata is invalid")
        if any(
            _path_partition(item.path, prefix="source=") != "finnhub"
            or _path_partition(item.path, prefix="dataset=") != "earnings-calendar"
            for item in provider_observations
        ):
            raise ValueError("earnings source contains a non-Finnhub observation")
        raw = {
            "schema_version": 1,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "ingested_at": ingested_at.isoformat(),
            "silver_files": self._entries(item.path for item in silver_files),
            "provider_observations": self._entries(item.path for item in provider_observations),
        }
        encoded = json.dumps(raw, sort_keys=True, separators=(",", ":")).encode()
        digest = hashlib.sha256(encoded).hexdigest()
        root = self._layout.root / "manifests" / "earnings-sources"
        root.mkdir(parents=True, exist_ok=True)
        path = root / f"source-{digest[:20]}.json"
        try:
            with path.open("xb") as destination:
                destination.write(encoded)
        except FileExistsError:
            if path.read_bytes() != encoded:
                raise RuntimeError(f"earnings source manifest collision at {path}") from None
        return EarningsSourceManifest.load(path)

    @staticmethod
    def find_for_files(
        earnings_files: Sequence[Path],
        *,
        data_lake_root: Path,
    ) -> tuple[EarningsSourceManifest, ...]:
        """Resolve exactly one retained source manifest per earnings file."""
        targets = {path.resolve() for path in earnings_files}
        matches: dict[Path, list[EarningsSourceManifest]] = {path: [] for path in targets}
        source_root = data_lake_root.resolve() / "manifests" / "earnings-sources"
        for path in sorted(source_root.glob("source-*.json")):
            manifest = EarningsSourceManifest.load(path)
            manifest_paths = set(manifest.silver_paths(data_lake_root=data_lake_root))
            for target in targets & manifest_paths:
                manifest.provider_paths(data_lake_root=data_lake_root)
                matches[target].append(manifest)
        if any(len(items) != 1 for items in matches.values()):
            raise ValueError("earnings Silver files lack unique Finnhub source lineage")
        return tuple(
            sorted(
                {items[0].path: items[0] for items in matches.values()}.values(),
                key=lambda item: str(item.path),
            )
        )

    @staticmethod
    def reproduce(
        manifest: EarningsSourceManifest,
        *,
        data_lake_root: Path,
        output_layout: LakehouseLayout,
    ) -> tuple[SilverArtifact, ...]:
        start_date = date.fromisoformat(manifest.raw["start_date"])
        end_date = date.fromisoformat(manifest.raw["end_date"])
        ingested_at = datetime.fromisoformat(manifest.raw["ingested_at"])
        events = tuple(
            event
            for path in manifest.provider_paths(data_lake_root=data_lake_root)
            for event in FinnhubClient.earnings_calendar_from_payload(json.loads(path.read_bytes()))
        )
        if any(not start_date <= item.event_date <= end_date for item in events):
            raise ValueError("earnings provider event is outside its captured interval")
        reproduced = SilverWriter(output_layout).write_earnings(
            events,
            ingested_at=ingested_at,
            empty_partition_date=end_date,
        )
        expected = manifest.silver_paths(data_lake_root=data_lake_root)
        if _byte_digests(item.path for item in reproduced) != _byte_digests(expected):
            raise ValueError("earnings Silver differs from retained Finnhub observations")
        return reproduced

    def _entries(self, paths: Iterable[Path]) -> list[dict[str, str]]:
        return [self._entry(path) for path in sorted(paths)]

    def _entry(self, path: Path) -> dict[str, str]:
        resolved = path.resolve()
        try:
            relative = resolved.relative_to(self._layout.root.resolve())
        except ValueError as error:
            raise ValueError("earnings source must be inside the data lake") from error
        return {"path": relative.as_posix(), "sha256": _file_sha256(resolved)}


def _resolve_entries(
    entries: tuple[dict[str, str], ...],
    *,
    data_lake_root: Path,
) -> tuple[Path, ...]:
    root = data_lake_root.resolve()
    paths = tuple((root / entry["path"]).resolve() for entry in entries)
    if any(path == root or root not in path.parents for path in paths):
        raise ValueError("earnings source escapes the data lake")
    if any(
        _file_sha256(path) != entry["sha256"] for path, entry in zip(paths, entries, strict=True)
    ):
        raise ValueError("earnings source file is missing or differs")
    return paths


def _validate_entry(entry: object) -> None:
    if (
        not isinstance(entry, dict)
        or set(entry) != {"path", "sha256"}
        or not _is_sha256(entry.get("sha256"))
    ):
        raise ValueError("earnings source manifest entry is invalid")
    path = PurePosixPath(str(entry.get("path", "")))
    if (
        path.is_absolute()
        or ".." in path.parts
        or not path.parts
        or path.as_posix() != entry["path"]
    ):
        raise ValueError("earnings source manifest path is invalid")


def _path_partition(path: Path, *, prefix: str) -> str:
    matches = tuple(part.removeprefix(prefix) for part in path.parts if part.startswith(prefix))
    if len(matches) != 1:
        raise ValueError(f"earnings source path has ambiguous {prefix.rstrip('=')}: {path}")
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
