"""Provider-source reconstruction of historical research feature artifacts."""

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
from quant_earning_edge.data.earnings_source import (
    EarningsSourceCapture,
    EarningsSourceManifest,
)
from quant_earning_edge.data.minute_bars_source import (
    MinuteBarsSourceCapture,
    MinuteBarsSourceManifest,
)
from quant_earning_edge.data.split_source import (
    SplitHistorySourceCapture,
    SplitHistorySourceManifest,
)
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
from quant_earning_edge.universe.candidate_source import EventCandidateSourceCapture
from quant_earning_edge.universe.events import EventCandidateManifest

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from quant_earning_edge.data.layout import LakehouseLayout
    from quant_earning_edge.features.store import FeatureArtifact

_GROUPS = (
    "daily_bar_files",
    "candidate_files",
    "minute_bar_files",
    "earnings_files",
    "daily_bar_source_manifests",
    "daily_bar_provider_observations",
    "split_files",
    "split_source_manifests",
    "split_provider_observations",
    "minute_bar_source_manifests",
    "minute_bar_provider_observations",
    "earnings_source_manifests",
    "earnings_provider_observations",
    "candidate_manifests",
    "candidate_lineage_files",
)


@dataclass(frozen=True)
class HistoricalFeatureSourceManifest:
    """All causal inputs and provider lineage behind one research feature file."""

    path: Path
    raw: dict[str, Any]

    @classmethod
    def load(cls, path: Path) -> HistoricalFeatureSourceManifest:
        try:
            encoded = path.read_bytes()
            raw = json.loads(encoded)
        except (OSError, ValueError) as error:
            raise ValueError(f"invalid historical-feature source manifest: {path}") from error
        required = {
            "schema_version",
            "asof_date",
            "observed_at",
            "target_date",
            "feature_group",
            "symbols",
            "feature_names",
            "feature_file",
            "source_files",
        }
        if not isinstance(raw, dict) or set(raw) != required or raw["schema_version"] != 2:
            raise ValueError("historical-feature source manifest schema mismatch")
        sources = raw["source_files"]
        if (
            not isinstance(raw["feature_file"], dict)
            or not isinstance(sources, dict)
            or set(sources) != set(_GROUPS)
            or any(not isinstance(sources[name], list) for name in _GROUPS)
            or not sources["daily_bar_files"]
            or not sources["daily_bar_source_manifests"]
            or not sources["daily_bar_provider_observations"]
        ):
            raise ValueError("historical-feature source collections are invalid")
        for name in _GROUPS:
            entries = sources[name]
            for entry in entries:
                _validate_entry(entry)
            if tuple(entries) != tuple(sorted(entries, key=lambda item: item["path"])) or len(
                entries
            ) != len({entry["path"] for entry in entries}):
                raise ValueError("historical-feature source entries are invalid")
        _validate_entry(raw["feature_file"])
        optional_pairs = (
            ("minute_bar_files", "minute_bar_source_manifests"),
            ("minute_bar_files", "minute_bar_provider_observations"),
            ("earnings_files", "earnings_source_manifests"),
            ("earnings_files", "earnings_provider_observations"),
            ("candidate_files", "candidate_manifests"),
            ("candidate_files", "candidate_lineage_files"),
            ("split_source_manifests", "split_provider_observations"),
        )
        if any(bool(sources[left]) != bool(sources[right]) for left, right in optional_pairs):
            raise ValueError("historical-feature optional lineage is incomplete")
        try:
            asof_date = date.fromisoformat(str(raw["asof_date"]))
            observed_at = datetime.fromisoformat(str(raw["observed_at"]))
            target_date = (
                date.fromisoformat(str(raw["target_date"]))
                if raw["target_date"] is not None
                else None
            )
        except ValueError as error:
            raise ValueError("historical-feature source timestamps are invalid") from error
        symbols = raw["symbols"]
        feature_names = raw["feature_names"]
        if (
            asof_date.isoformat() != raw["asof_date"]
            or observed_at.tzinfo is None
            or observed_at.utcoffset() is None
            or observed_at.isoformat() != raw["observed_at"]
            or (target_date is not None and target_date <= asof_date)
            or (target_date is not None and target_date.isoformat() != raw["target_date"])
            or ((sources["minute_bar_files"] or sources["candidate_files"]) and target_date is None)
            or bool(sources["candidate_files"]) != bool(sources["earnings_files"])
            or not isinstance(raw["feature_group"], str)
            or not raw["feature_group"].strip()
            or not isinstance(symbols, list)
            or not symbols
            or symbols != sorted(set(symbols))
            or not isinstance(feature_names, list)
            or not feature_names
            or feature_names != sorted(set(feature_names))
        ):
            raise ValueError("historical-feature source metadata is invalid")
        if json.dumps(raw, sort_keys=True, separators=(",", ":")).encode() != encoded:
            raise ValueError("historical-feature source manifest is not canonical")
        return cls(path=path.resolve(), raw=raw)

    def grouped_paths(self, *, data_lake_root: Path) -> dict[str, tuple[Path, ...]]:
        return {
            name: _resolve_entries(
                tuple(self.raw["source_files"][name]),
                data_lake_root=data_lake_root,
            )
            for name in _GROUPS
        }

    def feature_path(self, *, data_lake_root: Path) -> Path:
        return _resolve_entries(
            (self.raw["feature_file"],),
            data_lake_root=data_lake_root,
        )[0]

    def lineage_paths(self, *, data_lake_root: Path) -> tuple[Path, ...]:
        groups = self.grouped_paths(data_lake_root=data_lake_root)
        return tuple(
            dict.fromkeys(
                (
                    self.path,
                    self.feature_path(data_lake_root=data_lake_root),
                    *(path for name in _GROUPS for path in groups[name]),
                )
            )
        )


class HistoricalFeatureSourceCapture:
    """Persist and independently replay historical feature computation."""

    def __init__(self, layout: LakehouseLayout) -> None:
        self._layout = layout

    def write(
        self,
        *,
        asof_date: date,
        observed_at: datetime,
        target_date: date | None,
        feature_group: str,
        symbols: Sequence[str],
        feature_names: Sequence[str],
        feature_file: FeatureArtifact,
        daily_bar_files: Sequence[Path],
        candidate_files: Sequence[Path] = (),
        minute_bar_files: Sequence[Path] = (),
        earnings_files: Sequence[Path] = (),
        split_source_manifest: Path | None = None,
    ) -> HistoricalFeatureSourceManifest:
        normalized_symbols = tuple(sorted({item.strip().upper() for item in symbols}))
        normalized_names = tuple(sorted(set(feature_names)))
        if (
            observed_at.tzinfo is None
            or observed_at.utcoffset() is None
            or not normalized_symbols
            or any(not item for item in normalized_symbols)
            or not normalized_names
            or not feature_group.strip()
            or not daily_bar_files
            or bool(candidate_files) != bool(earnings_files)
            or ((minute_bar_files or candidate_files) and target_date is None)
            or (target_date is not None and target_date <= asof_date)
        ):
            raise ValueError("historical-feature source metadata is invalid")
        table = pq.ParquetFile(feature_file.path).read()  # type: ignore[no-untyped-call]
        rows = table.to_pylist()
        if (
            table.schema != FEATURE_VALUE_SCHEMA
            or {str(row["symbol"]) for row in rows} != set(normalized_symbols)
            or {str(row["feature_name"]) for row in rows} != set(normalized_names)
            or {row["asof_date"] for row in rows} != {asof_date}
            or {row["computed_at"] for row in rows} != {observed_at}
        ):
            raise ValueError("historical feature artifact differs from source metadata")
        daily_sources = DailyBarsSourceCapture.find_for_files(
            daily_bar_files,
            data_lake_root=self._layout.root,
        )
        adjustment_modes = {bool(item.raw["adjusted"]) for item in daily_sources}
        if len(adjustment_modes) != 1:
            raise ValueError("historical features cannot mix daily-bar adjustment modes")
        adjusted = adjustment_modes.pop()
        if adjusted and any(
            item.raw["availability_policy"] == "session_close_plus_15m" for item in daily_sources
        ):
            raise ValueError(
                "retroactively adjusted backfill bars are not point-in-time feature inputs"
            )
        if adjusted == (split_source_manifest is not None):
            raise ValueError(
                "unadjusted daily bars require exactly one split-history source manifest"
            )
        split_source = (
            SplitHistorySourceManifest.load(split_source_manifest)
            if split_source_manifest is not None
            else None
        )
        if split_source is not None:
            if {item.raw["backfill_plan_id"] for item in daily_sources} != {
                split_source.raw["plan_id"]
            }:
                raise ValueError("daily bars and split history differ from the backfill plan")
            SplitHistorySourceCapture.reproduce(
                split_source,
                data_lake_root=self._layout.root,
            )
        minute_sources = (
            MinuteBarsSourceCapture.find_for_files(
                minute_bar_files,
                data_lake_root=self._layout.root,
            )
            if minute_bar_files
            else ()
        )
        earnings_sources = (
            EarningsSourceCapture.find_for_files(
                earnings_files,
                data_lake_root=self._layout.root,
            )
            if earnings_files
            else ()
        )
        candidate_manifests = tuple(
            EventCandidateSourceCapture.find_for_candidate(
                path,
                data_lake_root=self._layout.root,
            )
            for path in sorted(candidate_files)
        )
        raw = {
            "schema_version": 2,
            "asof_date": asof_date.isoformat(),
            "observed_at": observed_at.isoformat(),
            "target_date": target_date.isoformat() if target_date is not None else None,
            "feature_group": feature_group,
            "symbols": list(normalized_symbols),
            "feature_names": list(normalized_names),
            "feature_file": self._entry(feature_file.path),
            "source_files": {
                "daily_bar_files": self._entries(daily_bar_files),
                "candidate_files": self._entries(candidate_files),
                "minute_bar_files": self._entries(minute_bar_files),
                "earnings_files": self._entries(earnings_files),
                "daily_bar_source_manifests": self._entries(item.path for item in daily_sources),
                "daily_bar_provider_observations": self._entries(
                    path
                    for item in daily_sources
                    for path in item.provider_paths(data_lake_root=self._layout.root)
                ),
                "split_files": self._entries(
                    split_source.split_paths(data_lake_root=self._layout.root)
                    if split_source is not None
                    else ()
                ),
                "split_source_manifests": self._entries(
                    (split_source.path,) if split_source is not None else ()
                ),
                "split_provider_observations": self._entries(
                    split_source.provider_paths(data_lake_root=self._layout.root)
                    if split_source is not None
                    else ()
                ),
                "minute_bar_source_manifests": self._entries(item.path for item in minute_sources),
                "minute_bar_provider_observations": self._entries(
                    path
                    for item in minute_sources
                    for path in item.provider_paths(data_lake_root=self._layout.root)
                ),
                "earnings_source_manifests": self._entries(item.path for item in earnings_sources),
                "earnings_provider_observations": self._entries(
                    path
                    for item in earnings_sources
                    for path in item.provider_paths(data_lake_root=self._layout.root)
                ),
                "candidate_manifests": self._entries(item.path for item in candidate_manifests),
                "candidate_lineage_files": self._entries(
                    path
                    for item in candidate_manifests
                    for path in item.source_paths(data_lake_root=self._layout.root)
                ),
            },
        }
        encoded = json.dumps(raw, sort_keys=True, separators=(",", ":")).encode()
        digest = hashlib.sha256(encoded).hexdigest()
        path = feature_file.path.with_name(f"historical-source-{digest[:20]}.json")
        try:
            with path.open("xb") as destination:
                destination.write(encoded)
        except FileExistsError:
            if path.read_bytes() != encoded:
                raise RuntimeError(f"historical-feature source collision at {path}") from None
        return HistoricalFeatureSourceManifest.load(path)

    @staticmethod
    def find_for_feature(
        feature_file: Path,
        *,
        data_lake_root: Path,
    ) -> HistoricalFeatureSourceManifest:
        matches = []
        for path in sorted(feature_file.parent.glob("historical-source-*.json")):
            manifest = HistoricalFeatureSourceManifest.load(path)
            if manifest.feature_path(data_lake_root=data_lake_root) == feature_file.resolve():
                manifest.grouped_paths(data_lake_root=data_lake_root)
                matches.append(manifest)
        if len(matches) != 1:
            raise ValueError("historical feature lacks unique retained source lineage")
        return matches[0]

    @staticmethod
    def reproduce(
        manifest: HistoricalFeatureSourceManifest,
        *,
        data_lake_root: Path,
    ) -> Path:
        from quant_earning_edge.data.layout import LakehouseLayout  # noqa: PLC0415

        groups = manifest.grouped_paths(data_lake_root=data_lake_root)
        daily_sources = tuple(
            DailyBarsSourceManifest.load(path) for path in groups["daily_bar_source_manifests"]
        )
        minute_sources = tuple(
            MinuteBarsSourceManifest.load(path) for path in groups["minute_bar_source_manifests"]
        )
        earnings_sources = tuple(
            EarningsSourceManifest.load(path) for path in groups["earnings_source_manifests"]
        )
        split_sources = tuple(
            SplitHistorySourceManifest.load(path) for path in groups["split_source_manifests"]
        )
        if len(split_sources) > 1:
            raise ValueError("historical feature has multiple split-history sources")
        HistoricalFeatureSourceCapture._validate_lineage(
            groups=groups,
            daily_sources=daily_sources,
            minute_sources=minute_sources,
            earnings_sources=earnings_sources,
            data_lake_root=data_lake_root,
        )
        split_source = split_sources[0] if split_sources else None
        if split_source is not None:
            if set(groups["split_files"]) != set(
                split_source.split_paths(data_lake_root=data_lake_root)
            ) or set(groups["split_provider_observations"]) != set(
                split_source.provider_paths(data_lake_root=data_lake_root)
            ):
                raise ValueError("historical split lineage differs from its source manifest")
            SplitHistorySourceCapture.reproduce(
                split_source,
                data_lake_root=data_lake_root,
            )
        candidate_manifests = tuple(
            EventCandidateManifest.load(path) for path in groups["candidate_manifests"]
        )
        if set(groups["candidate_lineage_files"]) != {
            path
            for item in candidate_manifests
            for path in item.source_paths(data_lake_root=data_lake_root)
        }:
            raise ValueError("historical candidate lineage differs from its manifests")
        for candidate in groups["candidate_files"]:
            digest = _file_sha256(candidate)
            matches = [
                item for item in candidate_manifests if item.raw["candidate_file_sha256"] == digest
            ]
            if len(matches) != 1:
                raise ValueError("historical candidate lacks one matching manifest")
            EventCandidateSourceCapture.reproduce(
                matches[0],
                candidate_file=candidate,
                data_lake_root=data_lake_root,
            )
        with TemporaryDirectory(prefix="qee-historical-feature-reproduction-") as temporary:
            output_layout = LakehouseLayout(Path(temporary))
            for daily_source in daily_sources:
                DailyBarsSourceCapture.reproduce(
                    daily_source,
                    data_lake_root=data_lake_root,
                    output_layout=output_layout,
                )
            for minute_source in minute_sources:
                MinuteBarsSourceCapture.reproduce(
                    minute_source,
                    data_lake_root=data_lake_root,
                    output_layout=output_layout,
                )
            for earnings_source in earnings_sources:
                EarningsSourceCapture.reproduce(
                    earnings_source,
                    data_lake_root=data_lake_root,
                    output_layout=output_layout,
                )
            observed_at = datetime.fromisoformat(manifest.raw["observed_at"])
            asof_date = date.fromisoformat(manifest.raw["asof_date"])
            target_date = (
                date.fromisoformat(manifest.raw["target_date"])
                if manifest.raw["target_date"] is not None
                else None
            )
            contexts = DailyBarsFeatureLoader().load(
                groups["daily_bar_files"],
                symbols=tuple(manifest.raw["symbols"]),
                asof_date=asof_date,
                observed_at=observed_at,
                split_source_manifest=(split_source.path if split_source is not None else None),
                data_lake_root=data_lake_root,
            )
            if groups["minute_bar_files"]:
                assert target_date is not None
                contexts = PremarketFeatureLoader().enrich(
                    contexts,
                    minute_files=groups["minute_bar_files"],
                    target_date=target_date,
                    observed_at=observed_at,
                )
            if groups["candidate_files"]:
                assert target_date is not None
                contexts = EarningsFeatureLoader().enrich(
                    contexts,
                    candidate_files=groups["candidate_files"],
                    earnings_files=groups["earnings_files"],
                    observed_at=observed_at,
                    target_date=target_date,
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
                raise ValueError("historical feature differs from retained causal inputs")
        return expected

    @staticmethod
    def _validate_lineage(
        *,
        groups: dict[str, tuple[Path, ...]],
        daily_sources: tuple[DailyBarsSourceManifest, ...],
        minute_sources: tuple[MinuteBarsSourceManifest, ...],
        earnings_sources: tuple[EarningsSourceManifest, ...],
        data_lake_root: Path,
    ) -> None:
        if not set(groups["daily_bar_files"]).issubset(
            {
                path
                for item in daily_sources
                for path in item.silver_paths(data_lake_root=data_lake_root)
            }
        ) or set(groups["daily_bar_provider_observations"]) != {
            path
            for item in daily_sources
            for path in item.provider_paths(data_lake_root=data_lake_root)
        }:
            raise ValueError("historical daily-bar lineage differs from its inputs")
        if {item.silver_path(data_lake_root=data_lake_root) for item in minute_sources} != set(
            groups["minute_bar_files"]
        ) or set(groups["minute_bar_provider_observations"]) != {
            path
            for item in minute_sources
            for path in item.provider_paths(data_lake_root=data_lake_root)
        }:
            raise ValueError("historical minute-bar lineage differs from its inputs")
        if not set(groups["earnings_files"]).issubset(
            {
                path
                for item in earnings_sources
                for path in item.silver_paths(data_lake_root=data_lake_root)
            }
        ) or set(groups["earnings_provider_observations"]) != {
            path
            for item in earnings_sources
            for path in item.provider_paths(data_lake_root=data_lake_root)
        }:
            raise ValueError("historical earnings lineage differs from its inputs")

    def _entries(self, paths: Iterable[Path]) -> list[dict[str, str]]:
        return [self._entry(path) for path in sorted({path.resolve() for path in paths})]

    def _entry(self, path: Path) -> dict[str, str]:
        resolved = path.resolve()
        try:
            relative = resolved.relative_to(self._layout.root.resolve())
        except ValueError as error:
            raise ValueError("historical-feature source must be inside the data lake") from error
        return {"path": relative.as_posix(), "sha256": _file_sha256(resolved)}


def _resolve_entries(
    entries: tuple[dict[str, str], ...],
    *,
    data_lake_root: Path,
) -> tuple[Path, ...]:
    root = data_lake_root.resolve()
    paths = tuple((root / entry["path"]).resolve() for entry in entries)
    if any(path == root or root not in path.parents for path in paths):
        raise ValueError("historical-feature source escapes the data lake")
    try:
        valid = all(
            _file_sha256(path) == entry["sha256"]
            for path, entry in zip(paths, entries, strict=True)
        )
    except OSError as error:
        raise ValueError("historical-feature source file is missing or differs") from error
    if not valid:
        raise ValueError("historical-feature source file is missing or differs")
    return paths


def _validate_entry(entry: object) -> None:
    if (
        not isinstance(entry, dict)
        or set(entry) != {"path", "sha256"}
        or not _is_sha256(entry.get("sha256"))
    ):
        raise ValueError("historical-feature source entry is invalid")
    path = PurePosixPath(str(entry.get("path", "")))
    if (
        path.is_absolute()
        or ".." in path.parts
        or not path.parts
        or path.as_posix() != entry["path"]
    ):
        raise ValueError("historical-feature source path is invalid")


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
