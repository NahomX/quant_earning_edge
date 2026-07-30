"""Provider-bound split history used to normalize raw bars at a causal vintage."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any

from quant_earning_edge.data.clients import PolygonClient

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from quant_earning_edge.data.bronze import BronzeArtifact
    from quant_earning_edge.data.clients import StockSplit
    from quant_earning_edge.data.layout import LakehouseLayout
    from quant_earning_edge.data.silver import SilverArtifact, SilverWriter


@dataclass(frozen=True)
class SplitHistorySourceManifest:
    """Exact Polygon split pages behind one complete execution-date interval."""

    path: Path
    raw: dict[str, Any]

    @classmethod
    def load(cls, path: Path) -> SplitHistorySourceManifest:
        try:
            encoded = path.read_bytes()
            raw = json.loads(encoded)
        except (OSError, ValueError) as error:
            raise ValueError(f"invalid split-history source manifest: {path}") from error
        required = {
            "schema_version",
            "plan_id",
            "start_date",
            "end_date",
            "ingested_at",
            "split_files",
            "provider_observations",
        }
        if not isinstance(raw, dict) or set(raw) != required or raw["schema_version"] != 1:
            raise ValueError("split-history source manifest schema mismatch")
        split_files = raw["split_files"]
        providers = raw["provider_observations"]
        if not isinstance(split_files, list) or not isinstance(providers, list) or not providers:
            raise ValueError("split-history source manifest collections are invalid")
        for entries in (split_files, providers):
            for entry in entries:
                _validate_entry(entry)
            if entries != sorted(entries, key=lambda item: item["path"]) or len(entries) != len(
                {entry["path"] for entry in entries}
            ):
                raise ValueError("split-history source manifest entries are invalid")
        try:
            start_date = date.fromisoformat(str(raw["start_date"]))
            end_date = date.fromisoformat(str(raw["end_date"]))
            ingested_at = datetime.fromisoformat(str(raw["ingested_at"]))
        except ValueError as error:
            raise ValueError("split-history source manifest timestamps are invalid") from error
        if (
            not _is_sha256(raw["plan_id"])
            or start_date.isoformat() != raw["start_date"]
            or end_date.isoformat() != raw["end_date"]
            or end_date < start_date
            or ingested_at.tzinfo is None
            or ingested_at.utcoffset() is None
            or ingested_at.isoformat() != raw["ingested_at"]
        ):
            raise ValueError("split-history source manifest metadata is invalid")
        if json.dumps(raw, sort_keys=True, separators=(",", ":")).encode() != encoded:
            raise ValueError("split-history source manifest is not canonical")
        return cls(path=path.resolve(), raw=raw)

    def split_paths(self, *, data_lake_root: Path) -> tuple[Path, ...]:
        return _resolve_entries(tuple(self.raw["split_files"]), data_lake_root=data_lake_root)

    def provider_paths(self, *, data_lake_root: Path) -> tuple[Path, ...]:
        return _resolve_entries(
            tuple(self.raw["provider_observations"]),
            data_lake_root=data_lake_root,
        )

    @property
    def data_lake_root(self) -> Path:
        """Infer the validated lake root from the canonical manifest location."""
        if (
            self.path.parent.name != "split-history-sources"
            or self.path.parent.parent.name != "manifests"
        ):
            raise ValueError("split-history source manifest is outside its canonical lake path")
        return self.path.parent.parent.parent.resolve()

    def splits(self, *, data_lake_root: Path) -> tuple[StockSplit, ...]:
        raws = tuple(
            json.loads(path.read_bytes())
            for path in self.provider_paths(data_lake_root=data_lake_root)
        )
        return PolygonClient.stock_splits_from_payloads(
            raws,
            start_date=date.fromisoformat(self.raw["start_date"]),
            end_date=date.fromisoformat(self.raw["end_date"]),
        )


class SplitHistorySourceCapture:
    """Persist and independently reproduce a complete Polygon split interval."""

    def __init__(self, layout: LakehouseLayout) -> None:
        self._layout = layout

    def write(
        self,
        *,
        plan_id: str,
        start_date: date,
        end_date: date,
        ingested_at: datetime,
        split_files: Sequence[SilverArtifact],
        provider_observations: Sequence[BronzeArtifact],
    ) -> SplitHistorySourceManifest:
        if (
            not _is_sha256(plan_id)
            or end_date < start_date
            or ingested_at.tzinfo is None
            or ingested_at.utcoffset() is None
            or not provider_observations
            or any(
                _path_partition(item.path, prefix="source=") != "polygon"
                or _path_partition(item.path, prefix="dataset=") != "stock-splits"
                for item in provider_observations
            )
        ):
            raise ValueError("split-history source capture metadata is invalid")
        raws = tuple(json.loads(item.path.read_bytes()) for item in provider_observations)
        splits = PolygonClient.stock_splits_from_payloads(
            raws,
            start_date=start_date,
            end_date=end_date,
        )
        if len(splits) != sum(item.row_count for item in split_files):
            raise ValueError("split-history Silver rows differ from Polygon observations")
        raw = {
            "schema_version": 1,
            "plan_id": plan_id,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "ingested_at": ingested_at.isoformat(),
            "split_files": self._entries(item.path for item in split_files),
            "provider_observations": self._entries(item.path for item in provider_observations),
        }
        encoded = json.dumps(raw, sort_keys=True, separators=(",", ":")).encode()
        digest = hashlib.sha256(encoded).hexdigest()
        root = self._layout.root / "manifests" / "split-history-sources"
        root.mkdir(parents=True, exist_ok=True)
        path = root / f"source-{digest[:20]}.json"
        try:
            with path.open("xb") as destination:
                destination.write(encoded)
        except FileExistsError:
            if path.read_bytes() != encoded:
                raise RuntimeError(f"split-history source manifest collision at {path}") from None
        return SplitHistorySourceManifest.load(path)

    def ensure(
        self,
        *,
        plan_id: str,
        start_date: date,
        end_date: date,
        ingested_at: datetime,
        provider: PolygonClient,
        silver_writer: SilverWriter,
    ) -> SplitHistorySourceManifest:
        """Reuse a retained scope or capture its complete provider interval once."""
        existing = self.find_for_plan(plan_id, data_lake_root=self._layout.root)
        if existing is not None:
            if (
                existing.raw["start_date"] != start_date.isoformat()
                or existing.raw["end_date"] != end_date.isoformat()
            ):
                raise ValueError("split-history plan identity was reused for another interval")
            self.reproduce(existing, data_lake_root=self._layout.root)
            return existing
        observation_start = len(provider.corporate_action_observation_artifacts)
        splits = provider.stock_splits(start_date=start_date, end_date=end_date)
        captured = self.write(
            plan_id=plan_id,
            start_date=start_date,
            end_date=end_date,
            ingested_at=ingested_at,
            split_files=silver_writer.write_splits(splits, ingested_at=ingested_at),
            provider_observations=provider.corporate_action_observation_artifacts[
                observation_start:
            ],
        )
        self.reproduce(captured, data_lake_root=self._layout.root)
        return captured

    @staticmethod
    def find_for_plan(
        plan_id: str,
        *,
        data_lake_root: Path,
    ) -> SplitHistorySourceManifest | None:
        matches = []
        root = data_lake_root.resolve() / "manifests" / "split-history-sources"
        for path in sorted(root.glob("source-*.json")):
            manifest = SplitHistorySourceManifest.load(path)
            if manifest.raw["plan_id"] == plan_id:
                manifest.provider_paths(data_lake_root=data_lake_root)
                manifest.split_paths(data_lake_root=data_lake_root)
                matches.append(manifest)
        if len(matches) > 1:
            raise ValueError("backfill plan has multiple retained split-history sources")
        return matches[0] if matches else None

    @staticmethod
    def reproduce(
        manifest: SplitHistorySourceManifest,
        *,
        data_lake_root: Path,
    ) -> tuple[Path, ...]:
        from quant_earning_edge.data.layout import LakehouseLayout  # noqa: PLC0415
        from quant_earning_edge.data.silver import SilverWriter  # noqa: PLC0415

        with TemporaryDirectory(prefix="qee-split-history-reproduction-") as temporary:
            reproduced = SilverWriter(LakehouseLayout(Path(temporary))).write_splits(
                manifest.splits(data_lake_root=data_lake_root),
                ingested_at=datetime.fromisoformat(manifest.raw["ingested_at"]),
            )
            expected = manifest.split_paths(data_lake_root=data_lake_root)
            if _byte_digests(item.path for item in reproduced) != _byte_digests(expected):
                raise ValueError("split-history Silver differs from Polygon observations")
        return expected

    def _entries(self, paths: Iterable[Path]) -> list[dict[str, str]]:
        return [self._entry(path) for path in sorted({path.resolve() for path in paths})]

    def _entry(self, path: Path) -> dict[str, str]:
        resolved = path.resolve()
        try:
            relative = resolved.relative_to(self._layout.root.resolve())
        except ValueError as error:
            raise ValueError("split-history source must be inside the data lake") from error
        return {"path": relative.as_posix(), "sha256": _file_sha256(resolved)}


def split_history_plan_id(
    *,
    scope: str,
    start_date: date,
    end_date: date,
    discriminator: str,
) -> str:
    """Return a stable identity for one complete split-history query scope."""
    if not scope.strip() or not discriminator.strip() or end_date < start_date:
        raise ValueError("split-history plan identity metadata is invalid")
    encoded = json.dumps(
        {
            "scope": scope.strip(),
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "discriminator": discriminator.strip(),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _resolve_entries(
    entries: tuple[dict[str, str], ...],
    *,
    data_lake_root: Path,
) -> tuple[Path, ...]:
    root = data_lake_root.resolve()
    paths = tuple((root / entry["path"]).resolve() for entry in entries)
    if any(path == root or root not in path.parents for path in paths):
        raise ValueError("split-history source escapes the data lake")
    try:
        valid = all(
            _file_sha256(path) == entry["sha256"]
            for path, entry in zip(paths, entries, strict=True)
        )
    except OSError as error:
        raise ValueError("split-history source file is missing or differs") from error
    if not valid:
        raise ValueError("split-history source file is missing or differs")
    return paths


def _validate_entry(entry: object) -> None:
    if (
        not isinstance(entry, dict)
        or set(entry) != {"path", "sha256"}
        or not _is_sha256(entry.get("sha256"))
    ):
        raise ValueError("split-history source manifest entry is invalid")
    path = PurePosixPath(str(entry.get("path", "")))
    if (
        path.is_absolute()
        or ".." in path.parts
        or not path.parts
        or path.as_posix() != entry["path"]
    ):
        raise ValueError("split-history source manifest path is invalid")


def _path_partition(path: Path, *, prefix: str) -> str:
    matches = [part.removeprefix(prefix) for part in path.parts if part.startswith(prefix)]
    if len(matches) != 1 or not matches[0]:
        raise ValueError(f"split-history source path has ambiguous {prefix.rstrip('=')}: {path}")
    return matches[0]


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _byte_digests(paths: Iterable[Path]) -> tuple[str, ...]:
    return tuple(sorted(_file_sha256(path) for path in paths))


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )
