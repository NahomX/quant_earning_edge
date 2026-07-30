"""Immutable raw-payload writer for the bronze data tier."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

    from quant_earning_edge.data.layout import LakehouseLayout


@dataclass(frozen=True)
class BronzeArtifact:
    """Identity and integrity metadata for one persisted raw payload."""

    path: Path
    sha256: str
    byte_count: int
    received_at: datetime


class BronzeWriter:
    """Persist canonical JSON without ever replacing an existing artifact."""

    def __init__(self, layout: LakehouseLayout) -> None:
        self._layout = layout

    def write_json(
        self,
        payload: Any,
        *,
        source: str,
        dataset: str,
        event_date: date,
        received_at: datetime | None = None,
    ) -> BronzeArtifact:
        """Write a canonical JSON payload using an exclusive create.

        Repeating the exact same observation is idempotent. A filename collision
        with different content fails loudly instead of mutating bronze history.
        """
        observed_at = received_at or datetime.now(UTC)
        if observed_at.tzinfo is None or observed_at.utcoffset() is None:
            raise ValueError("received_at must be timezone-aware")

        body = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        digest = hashlib.sha256(body).hexdigest()
        timestamp = observed_at.astimezone(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
        partition = self._layout.bronze(
            source=source,
            dataset=dataset,
            event_date=event_date,
        )
        partition.mkdir(parents=True, exist_ok=True)
        path = partition / f"{timestamp}_{digest[:16]}.json"

        try:
            with path.open("xb") as artifact:
                artifact.write(body)
        except FileExistsError:
            if path.read_bytes() != body:
                raise RuntimeError(f"Bronze artifact collision at {path}") from None

        return BronzeArtifact(
            path=path,
            sha256=digest,
            byte_count=len(body),
            received_at=observed_at,
        )
