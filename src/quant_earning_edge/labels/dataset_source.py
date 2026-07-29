"""Source-bound reconstruction of assembled training datasets."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence


@dataclass(frozen=True)
class TrainingDatasetSourceManifest:
    """Exact features, labels, and calendar behind one wide dataset."""

    path: Path
    raw: dict[str, Any]

    @classmethod
    def load(cls, path: Path) -> TrainingDatasetSourceManifest:
        try:
            encoded = path.read_bytes()
            raw = json.loads(encoded)
        except (OSError, ValueError) as error:
            raise ValueError(f"invalid training dataset source manifest: {path}") from error
        required = {
            "schema_version",
            "assembled_at",
            "dataset_file",
            "feature_files",
            "label_files",
            "session_file",
            "feature_source_manifests",
            "feature_source_files",
            "label_source_manifests",
            "label_source_files",
        }
        if not isinstance(raw, dict) or set(raw) != required or raw["schema_version"] != 2:
            raise ValueError("training dataset source manifest schema mismatch")
        collections = (
            "feature_files",
            "label_files",
            "feature_source_manifests",
            "feature_source_files",
            "label_source_manifests",
            "label_source_files",
        )
        if (
            not isinstance(raw["dataset_file"], dict)
            or not isinstance(raw["session_file"], dict)
            or any(not isinstance(raw[name], list) or not raw[name] for name in collections)
        ):
            raise ValueError("training dataset source manifest collections are invalid")
        entries = (raw["dataset_file"], raw["session_file"])
        for entry in entries:
            _validate_entry(entry)
        for name in collections:
            for entry in raw[name]:
                _validate_entry(entry)
            if tuple(raw[name]) != tuple(sorted(raw[name], key=lambda item: item["path"])) or len(
                raw[name]
            ) != len({entry["path"] for entry in raw[name]}):
                raise ValueError("training dataset source manifest entries are invalid")
        try:
            assembled_at = datetime.fromisoformat(str(raw["assembled_at"]))
        except ValueError as error:
            raise ValueError("training dataset source timestamp is invalid") from error
        if (
            assembled_at.tzinfo is None
            or assembled_at.utcoffset() is None
            or assembled_at.isoformat() != raw["assembled_at"]
        ):
            raise ValueError("training dataset source timestamp is invalid")
        if json.dumps(raw, sort_keys=True, separators=(",", ":")).encode() != encoded:
            raise ValueError("training dataset source manifest is not canonical")
        return cls(path=path.resolve(), raw=raw)

    def dataset_path(self) -> Path:
        return _resolve_entry(self.raw["dataset_file"])

    def feature_paths(self) -> tuple[Path, ...]:
        return _resolve_entries(self.raw["feature_files"])

    def label_paths(self) -> tuple[Path, ...]:
        return _resolve_entries(self.raw["label_files"])

    def session_path(self) -> Path:
        return _resolve_entry(self.raw["session_file"])

    def feature_source_manifest_paths(self) -> tuple[Path, ...]:
        return _resolve_entries(self.raw["feature_source_manifests"])

    def feature_source_paths(self) -> tuple[Path, ...]:
        return _resolve_entries(self.raw["feature_source_files"])

    def label_source_manifest_paths(self) -> tuple[Path, ...]:
        return _resolve_entries(self.raw["label_source_manifests"])

    def label_source_paths(self) -> tuple[Path, ...]:
        return _resolve_entries(self.raw["label_source_files"])

    @property
    def lineage_paths(self) -> tuple[Path, ...]:
        return tuple(
            dict.fromkeys(
                (
                    self.path,
                    self.dataset_path(),
                    *self.feature_paths(),
                    *self.label_paths(),
                    self.session_path(),
                    *self.feature_source_manifest_paths(),
                    *self.feature_source_paths(),
                    *self.label_source_manifest_paths(),
                    *self.label_source_paths(),
                )
            )
        )


class TrainingDatasetSourceCapture:
    """Persist and independently rebuild wide training Parquet."""

    @staticmethod
    def write(
        *,
        dataset_file: Path,
        feature_files: Sequence[Path],
        label_files: Sequence[Path],
        session_file: Path,
        assembled_at: datetime,
    ) -> TrainingDatasetSourceManifest:
        from quant_earning_edge.features.historical_source import (  # noqa: PLC0415
            HistoricalFeatureSourceCapture,
        )
        from quant_earning_edge.labels.source_capture import (  # noqa: PLC0415
            ForwardLabelSourceCapture,
        )

        if assembled_at.tzinfo is None or assembled_at.utcoffset() is None:
            raise ValueError("training dataset source timestamp must be timezone-aware")
        feature_manifests = tuple(
            HistoricalFeatureSourceCapture.find_for_feature(
                path,
                data_lake_root=_data_lake_root(path),
            )
            for path in sorted({item.resolve() for item in feature_files})
        )
        label_manifests = tuple(
            ForwardLabelSourceCapture.find_for_label(
                path,
                data_lake_root=_data_lake_root(path),
            )
            for path in sorted({item.resolve() for item in label_files})
        )
        feature_direct = {item.resolve() for item in feature_files}
        label_direct = {item.resolve() for item in label_files}
        raw = {
            "schema_version": 2,
            "assembled_at": assembled_at.isoformat(),
            "dataset_file": _entry(dataset_file),
            "feature_files": _entries(feature_files),
            "label_files": _entries(label_files),
            "session_file": _entry(session_file),
            "feature_source_manifests": _entries(item.path for item in feature_manifests),
            "feature_source_files": _entries(
                path
                for item in feature_manifests
                for path in item.lineage_paths(data_lake_root=_data_lake_root(item.path))
                if path != item.path and path not in feature_direct
            ),
            "label_source_manifests": _entries(item.path for item in label_manifests),
            "label_source_files": _entries(
                path
                for item in label_manifests
                for path in item.lineage_paths
                if path != item.path and path not in label_direct
            ),
        }
        encoded = json.dumps(raw, sort_keys=True, separators=(",", ":")).encode()
        digest = hashlib.sha256(encoded).hexdigest()
        path = dataset_file.resolve().parent / f"training-source-{digest[:20]}.json"
        try:
            with path.open("xb") as destination:
                destination.write(encoded)
        except FileExistsError:
            if path.read_bytes() != encoded:
                raise RuntimeError(f"training dataset source collision at {path}") from None
        return TrainingDatasetSourceManifest.load(path)

    @staticmethod
    def find_for_datasets(
        dataset_files: Sequence[Path],
    ) -> tuple[TrainingDatasetSourceManifest, ...]:
        """Resolve exactly one adjacent source manifest per dataset."""
        output = []
        for dataset in sorted({path.resolve() for path in dataset_files}):
            matches = []
            for path in sorted(dataset.parent.glob("training-source-*.json")):
                manifest = TrainingDatasetSourceManifest.load(path)
                if manifest.dataset_path() == dataset:
                    matches.append(manifest)
            if len(matches) != 1:
                raise ValueError("training dataset lacks unique retained source lineage")
            output.append(matches[0])
        return tuple(output)

    @staticmethod
    def reproduce(
        manifest: TrainingDatasetSourceManifest,
    ) -> Path:
        """Reassemble one dataset and require exact Parquet bytes."""
        from quant_earning_edge.data import LakehouseLayout  # noqa: PLC0415
        from quant_earning_edge.labels.dataset import (  # noqa: PLC0415
            TrainingDatasetAssembler,
        )

        expected = manifest.dataset_path()
        TrainingDatasetSourceCapture._reproduce_feature_sources(manifest)
        TrainingDatasetSourceCapture._reproduce_label_sources(manifest)
        with TemporaryDirectory(prefix="qee-training-dataset-reproduction-") as temporary:
            artifact = TrainingDatasetAssembler(LakehouseLayout(Path(temporary))).assemble(
                feature_files=manifest.feature_paths(),
                label_files=manifest.label_paths(),
                session_file=manifest.session_path(),
                assembled_at=datetime.fromisoformat(manifest.raw["assembled_at"]),
            )
            if artifact.path.read_bytes() != expected.read_bytes():
                raise ValueError("training dataset differs from reconstructed feature/label inputs")
        return expected

    @staticmethod
    def _reproduce_feature_sources(manifest: TrainingDatasetSourceManifest) -> None:
        from quant_earning_edge.features.historical_source import (  # noqa: PLC0415
            HistoricalFeatureSourceCapture,
            HistoricalFeatureSourceManifest,
        )

        feature_paths = set(manifest.feature_paths())
        sources = tuple(
            HistoricalFeatureSourceManifest.load(path)
            for path in manifest.feature_source_manifest_paths()
        )
        captured_features = {
            source.feature_path(data_lake_root=_data_lake_root(source.path)) for source in sources
        }
        if captured_features != feature_paths:
            raise ValueError("training feature lineage differs from its feature files")
        expected_lineage = {
            path
            for source in sources
            for path in source.lineage_paths(
                data_lake_root=_data_lake_root(source.path),
            )
            if path != source.path and path not in feature_paths
        }
        if set(manifest.feature_source_paths()) != expected_lineage:
            raise ValueError("training feature lineage differs from captured source files")
        for source in sources:
            HistoricalFeatureSourceCapture.reproduce(
                source,
                data_lake_root=_data_lake_root(source.path),
            )

    @staticmethod
    def _reproduce_label_sources(manifest: TrainingDatasetSourceManifest) -> None:
        from quant_earning_edge.labels.source_capture import (  # noqa: PLC0415
            ForwardLabelSourceCapture,
            ForwardLabelSourceManifest,
        )

        label_paths = set(manifest.label_paths())
        sources = tuple(
            ForwardLabelSourceManifest.load(path) for path in manifest.label_source_manifest_paths()
        )
        captured_labels = {
            source.source_paths(data_lake_root=_data_lake_root(source.path))[0]
            for source in sources
        }
        if captured_labels != label_paths:
            raise ValueError("training label lineage differs from its label files")
        expected_lineage = {
            path
            for source in sources
            for path in source.lineage_paths
            if path != source.path and path not in label_paths
        }
        if set(manifest.label_source_paths()) != expected_lineage:
            raise ValueError("training label lineage differs from captured source files")
        for source in sources:
            ForwardLabelSourceCapture.reproduce(
                source,
                data_lake_root=_data_lake_root(source.path),
            )


def _entries(paths: Iterable[Path]) -> list[dict[str, str]]:
    return [_entry(path) for path in sorted({path.resolve() for path in paths})]


def _entry(path: Path) -> dict[str, str]:
    resolved = path.resolve()
    return {
        "path": resolved.as_posix(),
        "sha256": _file_sha256(resolved),
    }


def _resolve_entries(entries: Sequence[dict[str, str]]) -> tuple[Path, ...]:
    return tuple(_resolve_entry(entry) for entry in entries)


def _resolve_entry(entry: dict[str, str]) -> Path:
    path = Path(entry["path"]).resolve()
    try:
        digest = _file_sha256(path)
    except OSError as error:
        raise ValueError("training dataset source file is missing or differs") from error
    if digest != entry["sha256"]:
        raise ValueError("training dataset source file is missing or differs")
    return path


def _validate_entry(entry: object) -> None:
    if (
        not isinstance(entry, dict)
        or set(entry) != {"path", "sha256"}
        or not _is_sha256(entry.get("sha256"))
    ):
        raise ValueError("training dataset source entry is invalid")
    path = Path(str(entry.get("path", "")))
    if not path.is_absolute() or path.as_posix() != entry["path"]:
        raise ValueError("training dataset source path is invalid")


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


def _data_lake_root(path: Path) -> Path:
    resolved = path.resolve()
    for parent in (resolved, *resolved.parents):
        if parent.name in {"gold", "silver", "bronze", "manifests"}:
            return parent.parent
    raise ValueError("training source artifact is outside a data lake")
