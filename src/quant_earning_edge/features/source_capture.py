"""Source-bound reconstruction of causal live feature artifacts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any

import pyarrow.parquet as pq

from quant_earning_edge.features.inputs import (
    DailyBarsFeatureLoader,
    EarningsFeatureLoader,
    PremarketFeatureLoader,
)
from quant_earning_edge.features.store import (
    FEATURE_VALUE_SCHEMA,
    FeatureEngine,
    FeatureStore,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from quant_earning_edge.data.layout import LakehouseLayout
    from quant_earning_edge.features.store import FeatureArtifact

_INPUT_GROUPS = (
    "candidate_files",
    "daily_bar_files",
    "minute_bar_files",
    "earnings_files",
)


@dataclass(frozen=True)
class FeatureSourceManifest:
    """Exact causal inputs behind one immutable live feature file."""

    path: Path
    raw: dict[str, Any]

    @classmethod
    def load(cls, path: Path) -> FeatureSourceManifest:
        try:
            encoded = path.read_bytes()
            raw = json.loads(encoded)
        except (OSError, ValueError) as error:
            raise ValueError(f"invalid feature source manifest: {path}") from error
        required = {
            "schema_version",
            "trade_date",
            "asof_date",
            "observed_at",
            "feature_group",
            "symbols",
            "feature_names",
            "feature_file",
            "source_files",
        }
        if not isinstance(raw, dict) or set(raw) != required or raw["schema_version"] != 1:
            raise ValueError("feature source manifest schema mismatch")
        if (
            not isinstance(raw["feature_file"], dict)
            or not isinstance(raw["source_files"], dict)
            or set(raw["source_files"]) != set(_INPUT_GROUPS)
            or any(
                not isinstance(raw["source_files"][name], list) or not raw["source_files"][name]
                for name in _INPUT_GROUPS
            )
        ):
            raise ValueError("feature source manifest collections are invalid")
        entries = (
            raw["feature_file"],
            *(entry for name in _INPUT_GROUPS for entry in raw["source_files"][name]),
        )
        for entry in entries:
            _validate_entry(entry)
        identities = tuple(entry["path"] for entry in entries)
        if len(identities) != len(set(identities)):
            raise ValueError("feature source manifest paths are duplicated")
        if any(
            tuple(raw["source_files"][name])
            != tuple(sorted(raw["source_files"][name], key=lambda item: item["path"]))
            for name in _INPUT_GROUPS
        ):
            raise ValueError("feature source manifest inputs are not sorted")
        try:
            trade_date = date.fromisoformat(str(raw["trade_date"]))
            asof_date = date.fromisoformat(str(raw["asof_date"]))
            observed_at = datetime.fromisoformat(str(raw["observed_at"]))
        except ValueError as error:
            raise ValueError("feature source manifest timestamps are invalid") from error
        symbols = raw["symbols"]
        names = raw["feature_names"]
        if (
            trade_date.isoformat() != raw["trade_date"]
            or asof_date.isoformat() != raw["asof_date"]
            or trade_date <= asof_date
            or observed_at.tzinfo is None
            or observed_at.utcoffset() is None
            or not isinstance(raw["feature_group"], str)
            or not raw["feature_group"].strip()
            or not isinstance(symbols, list)
            or not symbols
            or symbols != sorted(set(symbols))
            or any(not isinstance(item, str) or not item for item in symbols)
            or not isinstance(names, list)
            or not names
            or names != sorted(set(names))
            or any(not isinstance(item, str) or not item for item in names)
        ):
            raise ValueError("feature source manifest metadata is invalid")
        canonical = json.dumps(raw, sort_keys=True, separators=(",", ":")).encode()
        if canonical != encoded:
            raise ValueError("feature source manifest is not canonical")
        return cls(path=path.resolve(), raw=raw)

    @property
    def input_entries(self) -> tuple[dict[str, str], ...]:
        return tuple(entry for name in _INPUT_GROUPS for entry in self.raw["source_files"][name])

    def feature_path(self, *, data_lake_root: Path) -> Path:
        return _resolve_entries(
            (self.raw["feature_file"],),
            data_lake_root=data_lake_root,
        )[0]

    def input_paths(self, *, data_lake_root: Path) -> tuple[Path, ...]:
        return _resolve_entries(self.input_entries, data_lake_root=data_lake_root)

    def grouped_input_paths(
        self,
        *,
        data_lake_root: Path,
    ) -> tuple[tuple[Path, ...], ...]:
        paths = self.input_paths(data_lake_root=data_lake_root)
        counts = tuple(len(self.raw["source_files"][name]) for name in _INPUT_GROUPS)
        groups = []
        start = 0
        for count in counts:
            groups.append(paths[start : start + count])
            start += count
        return tuple(groups)


class FeatureSourceCapture:
    """Persist and independently reproduce live feature generation."""

    def __init__(self, layout: LakehouseLayout) -> None:
        self._layout = layout

    def write(
        self,
        *,
        trade_date: date,
        asof_date: date,
        observed_at: datetime,
        feature_group: str,
        symbols: Sequence[str],
        feature_names: Sequence[str],
        feature_file: FeatureArtifact,
        candidate_files: Sequence[Path],
        daily_bar_files: Sequence[Path],
        minute_bar_files: Sequence[Path],
        earnings_files: Sequence[Path],
    ) -> FeatureSourceManifest:
        normalized_symbols = tuple(sorted({item.strip().upper() for item in symbols}))
        normalized_names = tuple(sorted(set(feature_names)))
        if (
            trade_date <= asof_date
            or observed_at.tzinfo is None
            or observed_at.utcoffset() is None
            or not feature_group.strip()
            or not normalized_symbols
            or any(not item for item in normalized_symbols)
            or not normalized_names
        ):
            raise ValueError("feature source metadata is invalid")
        groups = (candidate_files, daily_bar_files, minute_bar_files, earnings_files)
        if any(not group for group in groups):
            raise ValueError("feature source input groups must not be empty")
        table = pq.ParquetFile(feature_file.path).read()  # type: ignore[no-untyped-call]
        rows = table.to_pylist()
        if (
            table.schema != FEATURE_VALUE_SCHEMA
            or {str(row["symbol"]) for row in rows} != set(normalized_symbols)
            or {str(row["feature_name"]) for row in rows} != set(normalized_names)
            or {row["asof_date"] for row in rows} != {asof_date}
            or {row["computed_at"] for row in rows} != {observed_at}
            or tuple(sorted(feature_file.feature_names)) != normalized_names
        ):
            raise ValueError("feature artifact differs from source metadata")
        raw = {
            "schema_version": 1,
            "trade_date": trade_date.isoformat(),
            "asof_date": asof_date.isoformat(),
            "observed_at": observed_at.isoformat(),
            "feature_group": feature_group,
            "symbols": list(normalized_symbols),
            "feature_names": list(normalized_names),
            "feature_file": self._entry(feature_file.path),
            "source_files": {
                name: self._entries(paths)
                for name, paths in zip(_INPUT_GROUPS, groups, strict=True)
            },
        }
        encoded = json.dumps(raw, sort_keys=True, separators=(",", ":")).encode()
        digest = hashlib.sha256(encoded).hexdigest()
        path = feature_file.path.with_name(f"source-{digest[:20]}.json")
        try:
            with path.open("xb") as destination:
                destination.write(encoded)
        except FileExistsError:
            if path.read_bytes() != encoded:
                raise RuntimeError(f"feature source manifest collision at {path}") from None
        return FeatureSourceManifest.load(path)

    @staticmethod
    def find_for_feature(
        feature_file: Path,
        *,
        data_lake_root: Path,
    ) -> FeatureSourceManifest:
        """Select valid retained provenance for an exact feature artifact."""
        matches = []
        for path in sorted(feature_file.parent.glob("source-*.json")):
            manifest = FeatureSourceManifest.load(path)
            if manifest.feature_path(data_lake_root=data_lake_root) == feature_file.resolve():
                manifest.input_paths(data_lake_root=data_lake_root)
                matches.append(manifest)
        if not matches:
            raise ValueError("feature file lacks retained source lineage")
        return matches[0]

    @staticmethod
    def reproduce(
        manifest: FeatureSourceManifest,
        *,
        data_lake_root: Path,
        output_layout: LakehouseLayout,
    ) -> FeatureArtifact:
        candidate_files, daily_bar_files, minute_bar_files, earnings_files = (
            manifest.grouped_input_paths(data_lake_root=data_lake_root)
        )
        observed_at = datetime.fromisoformat(manifest.raw["observed_at"])
        asof_date = date.fromisoformat(manifest.raw["asof_date"])
        trade_date = date.fromisoformat(manifest.raw["trade_date"])
        symbols = tuple(manifest.raw["symbols"])
        contexts = DailyBarsFeatureLoader().load(
            daily_bar_files,
            symbols=symbols,
            asof_date=asof_date,
            observed_at=observed_at,
        )
        contexts = PremarketFeatureLoader().enrich(
            contexts,
            minute_files=minute_bar_files,
            target_date=trade_date,
            observed_at=observed_at,
        )
        contexts = EarningsFeatureLoader().enrich(
            contexts,
            candidate_files=candidate_files,
            earnings_files=earnings_files,
            observed_at=observed_at,
            target_date=trade_date,
        )
        values = FeatureEngine().compute(
            contexts,
            feature_names=tuple(manifest.raw["feature_names"]),
        )
        reproduced = FeatureStore(output_layout).write(
            feature_group=manifest.raw["feature_group"],
            values=values,
            computed_at=observed_at,
        )
        expected = manifest.feature_path(data_lake_root=data_lake_root)
        if reproduced.path.read_bytes() != expected.read_bytes():
            raise ValueError("feature artifact differs from retained causal inputs")
        return reproduced

    def _entries(self, paths: Iterable[Path]) -> list[dict[str, str]]:
        return [self._entry(path) for path in sorted(paths)]

    def _entry(self, path: Path) -> dict[str, str]:
        resolved = path.resolve()
        try:
            relative = resolved.relative_to(self._layout.root.resolve())
        except ValueError as error:
            raise ValueError("feature source must be inside the data lake") from error
        return {"path": relative.as_posix(), "sha256": _file_sha256(resolved)}


def _resolve_entries(
    entries: tuple[dict[str, str], ...],
    *,
    data_lake_root: Path,
) -> tuple[Path, ...]:
    root = data_lake_root.resolve()
    paths = tuple((root / entry["path"]).resolve() for entry in entries)
    if any(path == root or root not in path.parents for path in paths):
        raise ValueError("feature source escapes the data lake")
    if any(
        _file_sha256(path) != entry["sha256"] for path, entry in zip(paths, entries, strict=True)
    ):
        raise ValueError("feature source file is missing or differs")
    return paths


def _validate_entry(entry: object) -> None:
    if (
        not isinstance(entry, dict)
        or set(entry) != {"path", "sha256"}
        or not _is_sha256(entry.get("sha256"))
    ):
        raise ValueError("feature source manifest entry is invalid")
    path = PurePosixPath(str(entry.get("path", "")))
    if (
        path.is_absolute()
        or ".." in path.parts
        or not path.parts
        or path.as_posix() != entry["path"]
    ):
        raise ValueError("feature source manifest path is invalid")


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
