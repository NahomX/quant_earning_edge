"""Source-bound reconstruction of earnings and corporate-action silver inputs."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any

from quant_earning_edge.data.clients import FinnhubClient, PolygonClient
from quant_earning_edge.data.silver import SilverWriter

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from quant_earning_edge.data.bronze import BronzeArtifact
    from quant_earning_edge.data.layout import LakehouseLayout
    from quant_earning_edge.data.silver import SilverArtifact

_SILVER_GROUPS = ("earnings_files", "split_files", "dividend_files")
_PROVIDER_GROUPS = ("earnings_observations", "split_observations", "dividend_observations")


@dataclass(frozen=True)
class EventSourceCaptureManifest:
    """Exact provider observations and expected event silver outputs."""

    path: Path
    raw: dict[str, Any]

    @classmethod
    def load(cls, path: Path) -> EventSourceCaptureManifest:
        try:
            encoded = path.read_bytes()
            raw = json.loads(encoded)
        except (OSError, ValueError) as error:
            raise ValueError(f"invalid event source manifest: {path}") from error
        required = {
            "schema_version",
            "start_date",
            "end_date",
            "ingested_at",
            "silver_files",
            "provider_observations",
        }
        if not isinstance(raw, dict) or set(raw) != required or raw["schema_version"] != 1:
            raise ValueError("event source manifest schema mismatch")
        silver = raw["silver_files"]
        providers = raw["provider_observations"]
        if (
            not isinstance(silver, dict)
            or set(silver) != set(_SILVER_GROUPS)
            or not isinstance(providers, dict)
            or set(providers) != set(_PROVIDER_GROUPS)
            or any(
                not isinstance(silver[name], list) or not silver[name] for name in _SILVER_GROUPS
            )
            or any(
                not isinstance(providers[name], list) or not providers[name]
                for name in _PROVIDER_GROUPS
            )
        ):
            raise ValueError("event source manifest collections are invalid")
        entries = tuple(
            entry
            for collection in (
                *(silver[name] for name in _SILVER_GROUPS),
                *(providers[name] for name in _PROVIDER_GROUPS),
            )
            for entry in collection
        )
        for entry in entries:
            _validate_entry(entry)
        identities = tuple(entry["path"] for entry in entries)
        if len(identities) != len(set(identities)):
            raise ValueError("event source manifest paths are duplicated")
        try:
            start_date = date.fromisoformat(str(raw["start_date"]))
            end_date = date.fromisoformat(str(raw["end_date"]))
            ingested_at = datetime.fromisoformat(str(raw["ingested_at"]))
        except ValueError as error:
            raise ValueError("event source manifest timestamps are invalid") from error
        if (
            start_date.isoformat() != raw["start_date"]
            or end_date.isoformat() != raw["end_date"]
            or end_date < start_date
            or ingested_at.tzinfo is None
            or ingested_at.utcoffset() is None
        ):
            raise ValueError("event source manifest metadata is invalid")
        canonical = json.dumps(raw, sort_keys=True, separators=(",", ":")).encode()
        if canonical != encoded:
            raise ValueError("event source manifest is not canonical")
        return cls(path=path.resolve(), raw=raw)

    @property
    def silver_entries(self) -> tuple[dict[str, str], ...]:
        return tuple(entry for name in _SILVER_GROUPS for entry in self.raw["silver_files"][name])

    @property
    def provider_entries(self) -> tuple[dict[str, str], ...]:
        return tuple(
            entry for name in _PROVIDER_GROUPS for entry in self.raw["provider_observations"][name]
        )

    def silver_paths(self, *, data_lake_root: Path) -> tuple[Path, ...]:
        return _resolve_entries(self.silver_entries, data_lake_root=data_lake_root)

    def provider_paths(self, *, data_lake_root: Path) -> tuple[Path, ...]:
        return _resolve_entries(self.provider_entries, data_lake_root=data_lake_root)


class EventSourceCapture:
    """Persist and independently reproduce raw-to-silver event lineage."""

    def __init__(self, layout: LakehouseLayout) -> None:
        self._layout = layout

    def write(
        self,
        *,
        start_date: date,
        end_date: date,
        ingested_at: datetime,
        earnings_files: Sequence[SilverArtifact],
        split_files: Sequence[SilverArtifact],
        dividend_files: Sequence[SilverArtifact],
        earnings_observations: Sequence[BronzeArtifact],
        corporate_action_observations: Sequence[BronzeArtifact],
    ) -> EventSourceCaptureManifest:
        if ingested_at.tzinfo is None or ingested_at.utcoffset() is None:
            raise ValueError("event source ingested_at must be timezone-aware")
        split_observations = tuple(
            item
            for item in corporate_action_observations
            if _dataset_from_path(item.path) == "stock-splits"
        )
        dividend_observations = tuple(
            item
            for item in corporate_action_observations
            if _dataset_from_path(item.path) == "cash-dividends"
        )
        collections = (
            earnings_files,
            split_files,
            dividend_files,
            earnings_observations,
            split_observations,
            dividend_observations,
        )
        if any(not items for items in collections):
            raise ValueError("event source capture collections must not be empty")
        raw = {
            "schema_version": 1,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "ingested_at": ingested_at.isoformat(),
            "silver_files": {
                "earnings_files": self._entries(item.path for item in earnings_files),
                "split_files": self._entries(item.path for item in split_files),
                "dividend_files": self._entries(item.path for item in dividend_files),
            },
            "provider_observations": {
                "earnings_observations": self._entries(item.path for item in earnings_observations),
                "split_observations": self._entries(item.path for item in split_observations),
                "dividend_observations": self._entries(item.path for item in dividend_observations),
            },
        }
        encoded = json.dumps(raw, sort_keys=True, separators=(",", ":")).encode()
        digest = hashlib.sha256(encoded).hexdigest()
        root = (
            self._layout.root
            / "manifests"
            / "event-sources"
            / f"for_trade_date={end_date.isoformat()}"
        )
        root.mkdir(parents=True, exist_ok=True)
        path = root / f"source-{digest[:20]}.json"
        try:
            with path.open("xb") as destination:
                destination.write(encoded)
        except FileExistsError:
            if path.read_bytes() != encoded:
                raise RuntimeError(f"event source manifest collision at {path}") from None
        return EventSourceCaptureManifest.load(path)

    @staticmethod
    def reproduce(
        manifest: EventSourceCaptureManifest,
        *,
        data_lake_root: Path,
        output_layout: LakehouseLayout,
    ) -> tuple[SilverArtifact, ...]:
        start_date = date.fromisoformat(manifest.raw["start_date"])
        end_date = date.fromisoformat(manifest.raw["end_date"])
        ingested_at = datetime.fromisoformat(manifest.raw["ingested_at"])
        provider_paths = manifest.provider_paths(data_lake_root=data_lake_root)
        provider_counts = tuple(
            len(manifest.raw["provider_observations"][name]) for name in _PROVIDER_GROUPS
        )
        earnings_end = provider_counts[0]
        splits_end = earnings_end + provider_counts[1]
        earnings_raw = tuple(
            json.loads(path.read_bytes()) for path in provider_paths[:earnings_end]
        )
        split_raw = tuple(
            json.loads(path.read_bytes()) for path in provider_paths[earnings_end:splits_end]
        )
        dividend_raw = tuple(json.loads(path.read_bytes()) for path in provider_paths[splits_end:])
        earnings = tuple(
            event
            for raw in earnings_raw
            for event in FinnhubClient.earnings_calendar_from_payload(raw)
        )
        if any(not start_date <= item.event_date <= end_date for item in earnings):
            raise ValueError("retained earnings event is outside the captured interval")
        splits = PolygonClient.stock_splits_from_payloads(
            split_raw,
            start_date=start_date,
            end_date=end_date,
        )
        dividends = PolygonClient.cash_dividends_from_payloads(
            dividend_raw,
            start_date=start_date,
            end_date=end_date,
        )
        writer = SilverWriter(output_layout)
        reproduced = (
            *writer.write_earnings(
                earnings,
                ingested_at=ingested_at,
                empty_partition_date=end_date,
            ),
            *writer.write_splits(
                splits,
                ingested_at=ingested_at,
                empty_partition_date=end_date,
            ),
            *writer.write_dividends(
                dividends,
                ingested_at=ingested_at,
                empty_partition_date=end_date,
            ),
        )
        expected_paths = manifest.silver_paths(data_lake_root=data_lake_root)
        if len(reproduced) != len(expected_paths) or any(
            artifact.path.read_bytes() != expected.read_bytes()
            for artifact, expected in zip(reproduced, expected_paths, strict=True)
        ):
            raise ValueError("event silver inputs differ from retained provider observations")
        return reproduced

    def _entries(self, paths: Iterable[Path]) -> list[dict[str, str]]:
        return [self._entry(path) for path in sorted(paths)]

    def _entry(self, path: Path) -> dict[str, str]:
        resolved = path.resolve()
        try:
            relative = resolved.relative_to(self._layout.root.resolve())
        except ValueError as error:
            raise ValueError("event source must be inside the data lake") from error
        return {"path": relative.as_posix(), "sha256": _file_sha256(resolved)}


def _resolve_entries(
    entries: tuple[dict[str, str], ...],
    *,
    data_lake_root: Path,
) -> tuple[Path, ...]:
    root = data_lake_root.resolve()
    paths = tuple((root / entry["path"]).resolve() for entry in entries)
    if any(path == root or root not in path.parents for path in paths):
        raise ValueError("event source escapes the data lake")
    if any(
        _file_sha256(path) != entry["sha256"] for path, entry in zip(paths, entries, strict=True)
    ):
        raise ValueError("event source file is missing or differs")
    return paths


def _validate_entry(entry: object) -> None:
    if (
        not isinstance(entry, dict)
        or set(entry) != {"path", "sha256"}
        or not _is_sha256(entry.get("sha256"))
    ):
        raise ValueError("event source manifest entry is invalid")
    path = PurePosixPath(str(entry.get("path", "")))
    if (
        path.is_absolute()
        or ".." in path.parts
        or not path.parts
        or path.as_posix() != entry["path"]
    ):
        raise ValueError("event source manifest path is invalid")


def _dataset_from_path(path: Path) -> str:
    datasets = tuple(
        part.removeprefix("dataset=") for part in path.parts if part.startswith("dataset=")
    )
    if len(datasets) != 1:
        raise ValueError(f"event provider observation dataset is ambiguous: {path}")
    return datasets[0]


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
