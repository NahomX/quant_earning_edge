"""Source-bound reconstruction of authoritative Alpaca session files."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any

from quant_earning_edge.data.calendar import SessionFileStore
from quant_earning_edge.data.clients.alpaca import AlpacaCalendarClient

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from quant_earning_edge.data.bronze import BronzeArtifact
    from quant_earning_edge.data.calendar import SessionFile
    from quant_earning_edge.data.layout import LakehouseLayout


@dataclass(frozen=True)
class CalendarSourceManifest:
    """Exact raw provider observations behind one immutable session file."""

    path: Path
    raw: dict[str, Any]

    @classmethod
    def load(cls, path: Path) -> CalendarSourceManifest:
        try:
            encoded = path.read_bytes()
            raw = json.loads(encoded)
        except (OSError, ValueError) as error:
            raise ValueError(f"invalid calendar source manifest: {path}") from error
        required = {
            "schema_version",
            "start_date",
            "end_date",
            "session_file",
            "provider_observations",
        }
        if not isinstance(raw, dict) or set(raw) != required or raw["schema_version"] != 1:
            raise ValueError("calendar source manifest schema mismatch")
        observations = raw["provider_observations"]
        if not isinstance(observations, list) or not observations:
            raise ValueError("calendar source manifest has no provider observations")
        entries = (raw["session_file"], *observations)
        for entry in entries:
            _validate_entry(entry)
        identities = tuple(entry["path"] for entry in entries)
        if len(identities) != len(set(identities)):
            raise ValueError("calendar source manifest paths are duplicated")
        if tuple(observations) != tuple(sorted(observations, key=lambda item: item["path"])):
            raise ValueError("calendar provider observations are not sorted")
        try:
            start_date = date.fromisoformat(str(raw["start_date"]))
            end_date = date.fromisoformat(str(raw["end_date"]))
        except ValueError as error:
            raise ValueError("calendar source manifest dates are invalid") from error
        if (
            start_date.isoformat() != raw["start_date"]
            or end_date.isoformat() != raw["end_date"]
            or end_date < start_date
        ):
            raise ValueError("calendar source manifest interval is invalid")
        canonical = json.dumps(raw, sort_keys=True, separators=(",", ":")).encode()
        if canonical != encoded:
            raise ValueError("calendar source manifest is not canonical")
        return cls(path=path.resolve(), raw=raw)

    @property
    def source_entries(self) -> tuple[dict[str, str], ...]:
        return (self.raw["session_file"], *self.raw["provider_observations"])

    @property
    def provider_entries(self) -> tuple[dict[str, str], ...]:
        return tuple(self.raw["provider_observations"])

    def session_path(self, *, data_lake_root: Path) -> Path:
        return _resolve_entries(
            (self.raw["session_file"],),
            data_lake_root=data_lake_root,
        )[0]

    def provider_paths(self, *, data_lake_root: Path) -> tuple[Path, ...]:
        return _resolve_entries(
            self.provider_entries,
            data_lake_root=data_lake_root,
        )


class CalendarSourceCapture:
    """Persist and independently reproduce raw-to-session-file lineage."""

    def __init__(self, layout: LakehouseLayout) -> None:
        self._layout = layout

    def write(
        self,
        *,
        start_date: date,
        end_date: date,
        session_file: SessionFile,
        provider_observations: Sequence[BronzeArtifact],
    ) -> CalendarSourceManifest:
        if end_date < start_date:
            raise ValueError("calendar source end_date must be on or after start_date")
        if not provider_observations:
            raise ValueError("calendar source capture requires provider observations")
        if any(
            _path_partition(item.path, prefix="source=") != "alpaca"
            or _path_partition(item.path, prefix="dataset=") != "market-calendar"
            for item in provider_observations
        ):
            raise ValueError("calendar source capture contains a non-calendar observation")
        loaded = SessionFileStore.load(session_file.path)
        if (
            loaded.sha256 != session_file.sha256
            or loaded.sessions != session_file.sessions
            or any(
                item.session_date < start_date or item.session_date > end_date
                for item in loaded.sessions
            )
        ):
            raise ValueError("calendar session file differs from its captured interval")
        raw = {
            "schema_version": 1,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "session_file": self._entry(session_file.path),
            "provider_observations": self._entries(item.path for item in provider_observations),
        }
        encoded = json.dumps(raw, sort_keys=True, separators=(",", ":")).encode()
        digest = hashlib.sha256(encoded).hexdigest()
        root = self._layout.root / "manifests" / "market-calendar" / "sources"
        root.mkdir(parents=True, exist_ok=True)
        path = root / f"source-{digest[:20]}.json"
        try:
            with path.open("xb") as destination:
                destination.write(encoded)
        except FileExistsError:
            if path.read_bytes() != encoded:
                raise RuntimeError(f"calendar source manifest collision at {path}") from None
        return CalendarSourceManifest.load(path)

    @staticmethod
    def find_for_session(
        session_file: Path,
        *,
        data_lake_root: Path,
    ) -> CalendarSourceManifest:
        """Select retained valid provenance for an exact session file."""
        root = data_lake_root.resolve()
        source_root = root / "manifests" / "market-calendar" / "sources"
        matches = []
        for path in sorted(source_root.glob("source-*.json")):
            manifest = CalendarSourceManifest.load(path)
            if manifest.session_path(data_lake_root=root) == session_file.resolve():
                manifest.provider_paths(data_lake_root=root)
                matches.append(manifest)
        if not matches:
            raise ValueError("session file lacks retained Alpaca source lineage")
        return matches[0]

    @staticmethod
    def reproduce(
        manifest: CalendarSourceManifest,
        *,
        data_lake_root: Path,
        output_layout: LakehouseLayout,
    ) -> SessionFile:
        start_date = date.fromisoformat(manifest.raw["start_date"])
        end_date = date.fromisoformat(manifest.raw["end_date"])
        sessions = tuple(
            session
            for path in manifest.provider_paths(data_lake_root=data_lake_root)
            for session in AlpacaCalendarClient.sessions_from_payload(
                json.loads(path.read_bytes()),
                start_date=start_date,
                end_date=end_date,
            )
        )
        reproduced = SessionFileStore(output_layout).write(sessions)
        expected = manifest.session_path(data_lake_root=data_lake_root)
        if reproduced.path.read_bytes() != expected.read_bytes():
            raise ValueError("calendar session file differs from retained provider observations")
        return reproduced

    def _entries(self, paths: Iterable[Path]) -> list[dict[str, str]]:
        return [self._entry(path) for path in sorted(paths)]

    def _entry(self, path: Path) -> dict[str, str]:
        resolved = path.resolve()
        try:
            relative = resolved.relative_to(self._layout.root.resolve())
        except ValueError as error:
            raise ValueError("calendar source must be inside the data lake") from error
        return {"path": relative.as_posix(), "sha256": _file_sha256(resolved)}


def _resolve_entries(
    entries: tuple[dict[str, str], ...],
    *,
    data_lake_root: Path,
) -> tuple[Path, ...]:
    root = data_lake_root.resolve()
    paths = tuple((root / entry["path"]).resolve() for entry in entries)
    if any(path == root or root not in path.parents for path in paths):
        raise ValueError("calendar source escapes the data lake")
    if any(
        _file_sha256(path) != entry["sha256"] for path, entry in zip(paths, entries, strict=True)
    ):
        raise ValueError("calendar source file is missing or differs")
    return paths


def _validate_entry(entry: object) -> None:
    if (
        not isinstance(entry, dict)
        or set(entry) != {"path", "sha256"}
        or not _is_sha256(entry.get("sha256"))
    ):
        raise ValueError("calendar source manifest entry is invalid")
    path = PurePosixPath(str(entry.get("path", "")))
    if (
        path.is_absolute()
        or ".." in path.parts
        or not path.parts
        or path.as_posix() != entry["path"]
    ):
        raise ValueError("calendar source manifest path is invalid")


def _path_partition(path: Path, *, prefix: str) -> str:
    matches = tuple(part.removeprefix(prefix) for part in path.parts if part.startswith(prefix))
    if len(matches) != 1:
        raise ValueError(f"calendar source path has ambiguous {prefix.rstrip('=')}: {path}")
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
