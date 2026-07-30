"""Independent reconstruction of fully source-bound event candidates."""

from __future__ import annotations

import hashlib
from datetime import date, datetime
from pathlib import Path
from tempfile import TemporaryDirectory

from quant_earning_edge.data import (
    CalendarSourceCapture,
    CalendarSourceManifest,
    LakehouseLayout,
)
from quant_earning_edge.universe.event_source_capture import (
    EventSourceCapture,
    EventSourceCaptureManifest,
)
from quant_earning_edge.universe.events import (
    EventCandidateJob,
    EventCandidateManifest,
)
from quant_earning_edge.universe.source_capture import (
    UniverseSourceCapture,
    UniverseSourceCaptureManifest,
)


class EventCandidateSourceCapture:
    """Discover and replay a schema-v5 candidate manifest and its provider chain."""

    @staticmethod
    def find_for_candidate(
        candidate_file: Path,
        *,
        data_lake_root: Path,
    ) -> EventCandidateManifest:
        matches = []
        digest = _file_sha256(candidate_file)
        for path in sorted(candidate_file.parent.glob("manifest-*.json")):
            manifest = EventCandidateManifest.load(path)
            if manifest.raw["candidate_file_sha256"] == digest:
                manifest.source_paths(data_lake_root=data_lake_root)
                matches.append(manifest)
        if len(matches) != 1 or matches[0].raw["schema_version"] != 5:
            raise ValueError("candidate file lacks unique complete provider source lineage")
        return matches[0]

    @staticmethod
    def reproduce(
        manifest: EventCandidateManifest,
        *,
        candidate_file: Path,
        data_lake_root: Path,
    ) -> Path:
        if (
            manifest.raw["schema_version"] != 5
            or _file_sha256(candidate_file) != manifest.raw["candidate_file_sha256"]
        ):
            raise ValueError("candidate generation lacks complete provider source lineage")
        sources = manifest.source_paths(data_lake_root=data_lake_root)
        universe_end = len(manifest.universe_lineage_entries)
        event_end = universe_end + len(manifest.event_lineage_entries)
        calendar_end = event_end + len(manifest.calendar_lineage_entries)
        universe_manifest_path = sources[0]
        event_manifest_path = sources[universe_end]
        calendar_manifest_path = sources[event_end]
        universe_manifest = UniverseSourceCaptureManifest.load(universe_manifest_path)
        event_manifest = EventSourceCaptureManifest.load(event_manifest_path)
        calendar_manifest = CalendarSourceManifest.load(calendar_manifest_path)
        if (
            universe_manifest.source_paths(data_lake_root=data_lake_root) != sources[1:universe_end]
            or event_manifest.provider_paths(data_lake_root=data_lake_root)
            != sources[universe_end + 1 : event_end]
            or calendar_manifest.provider_paths(data_lake_root=data_lake_root)
            != sources[event_end + 1 : calendar_end]
        ):
            raise ValueError("candidate lineage differs from its upstream manifests")
        candidate_sources = sources[calendar_end:]
        if (
            event_manifest.silver_paths(data_lake_root=data_lake_root) != candidate_sources[2:]
            or calendar_manifest.session_path(data_lake_root=data_lake_root) != candidate_sources[1]
        ):
            raise ValueError("candidate inputs differ from their provider manifests")
        source_groups = manifest.raw["source_files"]
        earnings_end = 2 + len(source_groups["earnings_files"])
        splits_end = earnings_end + len(source_groups["split_files"])
        with TemporaryDirectory(prefix="qee-candidate-source-reproduction-") as temporary:
            temporary_root = Path(temporary)
            reproduced_universe = UniverseSourceCapture.reproduce(
                universe_manifest,
                data_lake_root=data_lake_root,
                output_layout=LakehouseLayout(temporary_root / "universe"),
            )
            if reproduced_universe.path.read_bytes() != candidate_sources[0].read_bytes():
                raise ValueError("candidate universe differs from provider reconstruction")
            EventSourceCapture.reproduce(
                event_manifest,
                data_lake_root=data_lake_root,
                output_layout=LakehouseLayout(temporary_root / "events"),
            )
            CalendarSourceCapture.reproduce(
                calendar_manifest,
                data_lake_root=data_lake_root,
                output_layout=LakehouseLayout(temporary_root / "calendar"),
            )
            reproduced = EventCandidateJob(
                LakehouseLayout(temporary_root / "candidates"),
                source_root=data_lake_root,
            ).run(
                trade_date=date.fromisoformat(manifest.raw["trade_date"]),
                decision_at=datetime.fromisoformat(manifest.raw["decision_at"]),
                universe_snapshot=candidate_sources[0],
                session_file=candidate_sources[1],
                earnings_files=candidate_sources[2:earnings_end],
                split_files=candidate_sources[earnings_end:splits_end],
                dividend_files=candidate_sources[splits_end:],
                universe_source_manifest=universe_manifest_path,
                event_source_manifest=event_manifest_path,
                calendar_source_manifest=calendar_manifest_path,
            )
            if (
                reproduced.path.read_bytes() != candidate_file.read_bytes()
                or EventCandidateManifest.load(reproduced.manifest_path).raw != manifest.raw
            ):
                raise ValueError("candidate artifact differs from reconstructed upstream inputs")
        return candidate_file.resolve()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
