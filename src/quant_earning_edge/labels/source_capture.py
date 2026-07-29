"""Provider-source reconstruction of forward-label artifacts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any

import pyarrow.parquet as pq

from quant_earning_edge.data.bars_source import (
    DailyBarsSourceCapture,
    DailyBarsSourceManifest,
)
from quant_earning_edge.data.calendar import SessionFileStore
from quant_earning_edge.data.calendar_source import (
    CalendarSourceCapture,
    CalendarSourceManifest,
)
from quant_earning_edge.labels.forward import (
    FORWARD_LABEL_SCHEMA,
    ForwardLabelMaker,
    LabelStore,
)
from quant_earning_edge.labels.inputs import LabelBarsLoader

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from quant_earning_edge.data.layout import LakehouseLayout
    from quant_earning_edge.labels.forward import LabelArtifact


@dataclass(frozen=True)
class ForwardLabelSourceManifest:
    """Exact bars, calendar, and provider observations behind one label file."""

    path: Path
    raw: dict[str, Any]

    @classmethod
    def load(cls, path: Path) -> ForwardLabelSourceManifest:
        try:
            encoded = path.read_bytes()
            raw = json.loads(encoded)
        except (OSError, ValueError) as error:
            raise ValueError(f"invalid forward-label source manifest: {path}") from error
        required = {
            "schema_version",
            "asof_date",
            "observed_at",
            "symbols",
            "label_file",
            "daily_bar_files",
            "session_file",
            "daily_bar_source_manifests",
            "daily_bar_provider_observations",
            "calendar_source_manifest",
            "calendar_provider_observations",
        }
        if not isinstance(raw, dict) or set(raw) != required or raw["schema_version"] != 1:
            raise ValueError("forward-label source manifest schema mismatch")
        symbols = raw["symbols"]
        collections = (
            "daily_bar_files",
            "daily_bar_source_manifests",
            "daily_bar_provider_observations",
            "calendar_provider_observations",
        )
        if (
            not isinstance(symbols, list)
            or not symbols
            or symbols != sorted(set(symbols))
            or any(not isinstance(item, str) or not item for item in symbols)
            or not isinstance(raw["label_file"], dict)
            or not isinstance(raw["session_file"], dict)
            or not isinstance(raw["calendar_source_manifest"], dict)
            or any(not isinstance(raw[name], list) or not raw[name] for name in collections)
        ):
            raise ValueError("forward-label source manifest collections are invalid")
        entries = (
            raw["label_file"],
            *raw["daily_bar_files"],
            raw["session_file"],
            *raw["daily_bar_source_manifests"],
            *raw["daily_bar_provider_observations"],
            raw["calendar_source_manifest"],
            *raw["calendar_provider_observations"],
        )
        for entry in entries:
            _validate_entry(entry)
        identities = tuple(entry["path"] for entry in entries)
        if len(identities) != len(set(identities)):
            raise ValueError("forward-label source manifest paths are duplicated")
        if any(
            tuple(raw[name]) != tuple(sorted(raw[name], key=lambda item: item["path"]))
            for name in collections
        ):
            raise ValueError("forward-label source manifest entries are not sorted")
        try:
            asof_date = date.fromisoformat(str(raw["asof_date"]))
            observed_at = datetime.fromisoformat(str(raw["observed_at"]))
        except ValueError as error:
            raise ValueError("forward-label source timestamps are invalid") from error
        if (
            asof_date.isoformat() != raw["asof_date"]
            or observed_at.tzinfo is None
            or observed_at.utcoffset() is None
            or observed_at.isoformat() != raw["observed_at"]
        ):
            raise ValueError("forward-label source metadata is invalid")
        if json.dumps(raw, sort_keys=True, separators=(",", ":")).encode() != encoded:
            raise ValueError("forward-label source manifest is not canonical")
        return cls(path=path.resolve(), raw=raw)

    @property
    def source_entries(self) -> tuple[dict[str, str], ...]:
        return (
            self.raw["label_file"],
            *self.raw["daily_bar_files"],
            self.raw["session_file"],
            *self.raw["daily_bar_source_manifests"],
            *self.raw["daily_bar_provider_observations"],
            self.raw["calendar_source_manifest"],
            *self.raw["calendar_provider_observations"],
        )

    def source_paths(self, *, data_lake_root: Path) -> tuple[Path, ...]:
        return _resolve_entries(self.source_entries, data_lake_root=data_lake_root)

    @property
    def lineage_paths(self) -> tuple[Path, ...]:
        root = _infer_data_lake_root(self.path)
        return (self.path, *self.source_paths(data_lake_root=root))


class ForwardLabelSourceCapture:
    """Persist and independently reproduce forward-label generation."""

    def __init__(self, layout: LakehouseLayout) -> None:
        self._layout = layout

    def write(
        self,
        *,
        asof_date: date,
        observed_at: datetime,
        symbols: Sequence[str],
        label_file: LabelArtifact,
        daily_bar_files: Sequence[Path],
        session_file: Path,
    ) -> ForwardLabelSourceManifest:
        normalized_symbols = tuple(sorted({item.strip().upper() for item in symbols}))
        if (
            observed_at.tzinfo is None
            or observed_at.utcoffset() is None
            or not normalized_symbols
            or any(not item for item in normalized_symbols)
            or not daily_bar_files
        ):
            raise ValueError("forward-label source metadata is invalid")
        table = pq.ParquetFile(label_file.path).read()  # type: ignore[no-untyped-call]
        rows = table.to_pylist()
        if (
            table.schema != FORWARD_LABEL_SCHEMA
            or {str(row["symbol"]) for row in rows} != set(normalized_symbols)
            or {row["asof_date"] for row in rows} != {asof_date}
            or {row["computed_at"] for row in rows} != {observed_at}
        ):
            raise ValueError("forward-label artifact differs from source metadata")
        bar_sources = DailyBarsSourceCapture.find_for_files(
            daily_bar_files,
            data_lake_root=self._layout.root,
        )
        bar_providers = tuple(
            sorted(
                {
                    path
                    for source in bar_sources
                    for path in source.provider_paths(data_lake_root=self._layout.root)
                }
            )
        )
        calendar_source = CalendarSourceCapture.find_for_session(
            session_file,
            data_lake_root=self._layout.root,
        )
        calendar_providers = calendar_source.provider_paths(data_lake_root=self._layout.root)
        raw = {
            "schema_version": 1,
            "asof_date": asof_date.isoformat(),
            "observed_at": observed_at.isoformat(),
            "symbols": list(normalized_symbols),
            "label_file": self._entry(label_file.path),
            "daily_bar_files": self._entries(daily_bar_files),
            "session_file": self._entry(session_file),
            "daily_bar_source_manifests": self._entries(source.path for source in bar_sources),
            "daily_bar_provider_observations": self._entries(bar_providers),
            "calendar_source_manifest": self._entry(calendar_source.path),
            "calendar_provider_observations": self._entries(calendar_providers),
        }
        encoded = json.dumps(raw, sort_keys=True, separators=(",", ":")).encode()
        digest = hashlib.sha256(encoded).hexdigest()
        path = label_file.path.with_name(f"label-source-{digest[:20]}.json")
        try:
            with path.open("xb") as destination:
                destination.write(encoded)
        except FileExistsError:
            if path.read_bytes() != encoded:
                raise RuntimeError(f"forward-label source collision at {path}") from None
        return ForwardLabelSourceManifest.load(path)

    @staticmethod
    def find_for_label(
        label_file: Path,
        *,
        data_lake_root: Path,
    ) -> ForwardLabelSourceManifest:
        """Resolve exactly one valid adjacent source manifest for a label file."""
        matches = []
        for path in sorted(label_file.parent.glob("label-source-*.json")):
            manifest = ForwardLabelSourceManifest.load(path)
            if manifest.source_paths(data_lake_root=data_lake_root)[0] == label_file.resolve():
                matches.append(manifest)
        if len(matches) != 1:
            raise ValueError("forward-label file lacks unique retained source lineage")
        return matches[0]

    @staticmethod
    def reproduce(
        manifest: ForwardLabelSourceManifest,
        *,
        data_lake_root: Path,
    ) -> Path:
        """Rebuild provider Silver, sessions, and the exact label Parquet."""
        from quant_earning_edge.data.layout import LakehouseLayout  # noqa: PLC0415

        paths = manifest.source_paths(data_lake_root=data_lake_root)
        daily_count = len(manifest.raw["daily_bar_files"])
        daily_source_count = len(manifest.raw["daily_bar_source_manifests"])
        daily_provider_count = len(manifest.raw["daily_bar_provider_observations"])
        calendar_provider_count = len(manifest.raw["calendar_provider_observations"])
        cursor = 1
        daily_files = paths[cursor : cursor + daily_count]
        cursor += daily_count
        session_file = paths[cursor]
        cursor += 1
        daily_source_paths = paths[cursor : cursor + daily_source_count]
        cursor += daily_source_count
        daily_provider_paths = paths[cursor : cursor + daily_provider_count]
        cursor += daily_provider_count
        calendar_source_path = paths[cursor]
        cursor += 1
        calendar_provider_paths = paths[cursor : cursor + calendar_provider_count]
        daily_sources = tuple(DailyBarsSourceManifest.load(path) for path in daily_source_paths)
        if not set(daily_files).issubset(
            {
                path
                for source in daily_sources
                for path in source.silver_paths(data_lake_root=data_lake_root)
            }
        ) or set(daily_provider_paths) != {
            path
            for source in daily_sources
            for path in source.provider_paths(data_lake_root=data_lake_root)
        }:
            raise ValueError("forward-label daily-bar lineage differs from its inputs")
        calendar_source = CalendarSourceManifest.load(calendar_source_path)
        if calendar_source.session_path(data_lake_root=data_lake_root) != session_file or set(
            calendar_source.provider_paths(data_lake_root=data_lake_root)
        ) != set(calendar_provider_paths):
            raise ValueError("forward-label calendar lineage differs from its input")
        with TemporaryDirectory(prefix="qee-forward-label-reproduction-") as temporary:
            output_layout = LakehouseLayout(Path(temporary))
            for source in daily_sources:
                DailyBarsSourceCapture.reproduce(
                    source,
                    data_lake_root=data_lake_root,
                    output_layout=output_layout,
                )
            CalendarSourceCapture.reproduce(
                calendar_source,
                data_lake_root=data_lake_root,
                output_layout=output_layout,
            )
            asof_date = date.fromisoformat(manifest.raw["asof_date"])
            observed_at = datetime.fromisoformat(manifest.raw["observed_at"])
            sessions = tuple(
                item.session_date for item in SessionFileStore.load(session_file).sessions
            )
            try:
                horizon_end = sessions[sessions.index(asof_date) + 5]
            except (ValueError, IndexError) as error:
                raise ValueError("forward-label calendar lacks the captured horizon") from error
            symbols = tuple(manifest.raw["symbols"])
            bars = LabelBarsLoader().load(
                daily_files,
                symbols=symbols,
                start_date=asof_date,
                end_date=horizon_end,
                observed_at=observed_at,
            )
            labels = ForwardLabelMaker().compute(
                keys=tuple((symbol, asof_date) for symbol in symbols),
                sessions=sessions,
                bars=bars,
            )
            reproduced = LabelStore(output_layout).write(labels, computed_at=observed_at)
            if reproduced.path.read_bytes() != paths[0].read_bytes():
                raise ValueError("forward-label artifact differs from retained causal inputs")
        return paths[0]

    def _entries(self, paths: Iterable[Path]) -> list[dict[str, str]]:
        return [self._entry(path) for path in sorted({path.resolve() for path in paths})]

    def _entry(self, path: Path) -> dict[str, str]:
        resolved = path.resolve()
        try:
            relative = resolved.relative_to(self._layout.root.resolve())
        except ValueError as error:
            raise ValueError("forward-label source must be inside the data lake") from error
        return {"path": relative.as_posix(), "sha256": _file_sha256(resolved)}


def _resolve_entries(
    entries: tuple[dict[str, str], ...],
    *,
    data_lake_root: Path,
) -> tuple[Path, ...]:
    root = data_lake_root.resolve()
    paths = tuple((root / entry["path"]).resolve() for entry in entries)
    if any(path == root or root not in path.parents for path in paths):
        raise ValueError("forward-label source escapes the data lake")
    try:
        valid = all(
            _file_sha256(path) == entry["sha256"]
            for path, entry in zip(paths, entries, strict=True)
        )
    except OSError as error:
        raise ValueError("forward-label source file is missing or differs") from error
    if not valid:
        raise ValueError("forward-label source file is missing or differs")
    return paths


def _infer_data_lake_root(path: Path) -> Path:
    for parent in path.resolve().parents:
        if parent.name == "gold":
            return parent.parent
    raise ValueError("forward-label source is outside a data lake")


def _validate_entry(entry: object) -> None:
    if (
        not isinstance(entry, dict)
        or set(entry) != {"path", "sha256"}
        or not _is_sha256(entry.get("sha256"))
    ):
        raise ValueError("forward-label source manifest entry is invalid")
    path = PurePosixPath(str(entry.get("path", "")))
    if (
        path.is_absolute()
        or ".." in path.parts
        or not path.parts
        or path.as_posix() != entry["path"]
    ):
        raise ValueError("forward-label source manifest path is invalid")


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
