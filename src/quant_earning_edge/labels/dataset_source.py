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
        }
        if not isinstance(raw, dict) or set(raw) != required or raw["schema_version"] != 1:
            raise ValueError("training dataset source manifest schema mismatch")
        if (
            not isinstance(raw["dataset_file"], dict)
            or not isinstance(raw["session_file"], dict)
            or not isinstance(raw["feature_files"], list)
            or not raw["feature_files"]
            or not isinstance(raw["label_files"], list)
            or not raw["label_files"]
        ):
            raise ValueError("training dataset source manifest collections are invalid")
        entries = (
            raw["dataset_file"],
            *raw["feature_files"],
            *raw["label_files"],
            raw["session_file"],
        )
        for entry in entries:
            _validate_entry(entry)
        identities = tuple(entry["path"] for entry in entries)
        if len(identities) != len(set(identities)):
            raise ValueError("training dataset source manifest paths are duplicated")
        if any(
            tuple(raw[name]) != tuple(sorted(raw[name], key=lambda item: item["path"]))
            for name in ("feature_files", "label_files")
        ):
            raise ValueError("training dataset source manifest entries are not sorted")
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

    @property
    def lineage_paths(self) -> tuple[Path, ...]:
        return (
            self.path,
            self.dataset_path(),
            *self.feature_paths(),
            *self.label_paths(),
            self.session_path(),
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
        if assembled_at.tzinfo is None or assembled_at.utcoffset() is None:
            raise ValueError("training dataset source timestamp must be timezone-aware")
        raw = {
            "schema_version": 1,
            "assembled_at": assembled_at.isoformat(),
            "dataset_file": _entry(dataset_file),
            "feature_files": _entries(feature_files),
            "label_files": _entries(label_files),
            "session_file": _entry(session_file),
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
