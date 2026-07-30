"""Immutable, content-addressed market-session files."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING

from quant_earning_edge.data.clients.alpaca import MarketSession

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from quant_earning_edge.data.layout import LakehouseLayout


@dataclass(frozen=True)
class SessionFile:
    """Persisted calendar identity and location."""

    path: Path
    sha256: str
    sessions: tuple[MarketSession, ...]


class SessionFileStore:
    """Write and read explicit session sets without mutable aliases."""

    def __init__(self, layout: LakehouseLayout) -> None:
        self._root = layout.root / "manifests" / "market-calendar"

    def write(self, sessions: Sequence[MarketSession]) -> SessionFile:
        """Persist a non-empty, ordered session range content-addressably."""
        normalized = tuple(sessions)
        self._validate(normalized)
        payload = {
            "provider": "alpaca",
            "sessions": [
                {
                    "session_date": item.session_date.isoformat(),
                    "open_at": item.open_at.isoformat(),
                    "close_at": item.close_at.isoformat(),
                }
                for item in normalized
            ],
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        digest = hashlib.sha256(encoded).hexdigest()
        path = self._root / f"sessions-{digest}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with path.open("xb") as destination:
                destination.write(encoded)
        except FileExistsError:
            if path.read_bytes() != encoded:
                raise RuntimeError(f"calendar digest collision at {path}") from None
        return SessionFile(path=path, sha256=digest, sessions=normalized)

    @staticmethod
    def load(path: Path) -> SessionFile:
        """Validate and load a previously persisted session file."""
        encoded = path.read_bytes()
        digest = hashlib.sha256(encoded).hexdigest()
        raw = json.loads(encoded)
        if raw.get("provider") != "alpaca" or not isinstance(raw.get("sessions"), list):
            raise ValueError("invalid Alpaca session file")
        sessions = tuple(MarketSession.model_validate(item) for item in raw["sessions"])
        SessionFileStore._validate(sessions)
        return SessionFile(path=path, sha256=digest, sessions=sessions)

    @staticmethod
    def _validate(sessions: tuple[MarketSession, ...]) -> None:
        if not sessions:
            raise ValueError("session file must not be empty")
        dates = tuple(item.session_date for item in sessions)
        if dates != tuple(sorted(set(dates))):
            raise ValueError("sessions must have unique ascending dates")
