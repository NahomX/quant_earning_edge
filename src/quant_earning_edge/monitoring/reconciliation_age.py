"""Derive unresolved reconciliation age from immutable report revisions."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence
    from datetime import date, datetime
    from pathlib import Path

    from quant_earning_edge.data.calendar import SessionFile
    from quant_earning_edge.live import PaperReconciliationReport


@dataclass(frozen=True)
class ReconciliationAgeEvidence:
    """Latest-report selection and authoritative completed-close count."""

    schema_version: int
    control_date: date
    evaluated_at: datetime
    calendar_sha256: str
    input_report_sha256: tuple[str, ...]
    latest_report_sha256: tuple[str, ...]
    unresolved_session_dates: tuple[date, ...]
    reconciliation_break_age_sessions: int | None

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("unsupported reconciliation-age schema")
        if self.evaluated_at.tzinfo is None or self.evaluated_at.utcoffset() is None:
            raise ValueError("reconciliation-age evaluated_at must be timezone-aware")
        if self.unresolved_session_dates != tuple(sorted(set(self.unresolved_session_dates))):
            raise ValueError("unresolved reconciliation dates must be unique and sorted")
        if bool(self.unresolved_session_dates) != (
            self.reconciliation_break_age_sessions is not None
        ):
            raise ValueError("reconciliation age and unresolved dates are inconsistent")
        for digest in (
            self.calendar_sha256,
            *self.input_report_sha256,
            *self.latest_report_sha256,
        ):
            if len(digest) != 64 or any(item not in "0123456789abcdef" for item in digest):
                raise ValueError("reconciliation-age digest must be SHA-256")

    @property
    def canonical_bytes(self) -> bytes:
        return json.dumps(
            asdict(self),
            default=lambda item: item.isoformat(),
            sort_keys=True,
            separators=(",", ":"),
        ).encode()

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes).hexdigest()

    def write(self, output: Path) -> None:
        encoded = self.canonical_bytes
        output.parent.mkdir(parents=True, exist_ok=True)
        try:
            with output.open("xb") as destination:
                destination.write(encoded)
        except FileExistsError:
            if output.read_bytes() != encoded:
                raise RuntimeError(f"reconciliation-age collision at {output}") from None


class ReconciliationAgeEvaluator:
    """Select latest revisions and count only intervening completed sessions."""

    def evaluate(
        self,
        *,
        calendar: SessionFile,
        reports: Sequence[PaperReconciliationReport],
        control_date: date,
        evaluated_at: datetime,
    ) -> ReconciliationAgeEvidence:
        session_dates = tuple(item.session_date for item in calendar.sessions)
        if control_date not in session_dates:
            raise ValueError("reconciliation control date is not authoritative")
        by_session: dict[date, list[PaperReconciliationReport]] = {}
        for report in reports:
            if report.session_date not in session_dates:
                raise ValueError("reconciliation report session is not authoritative")
            if report.session_date >= control_date:
                raise ValueError("reconciliation report must precede the control date")
            if report.evaluated_at > evaluated_at:
                raise ValueError("reconciliation report was evaluated after the control")
            by_session.setdefault(report.session_date, []).append(report)
        latest = []
        for session_date, revisions in sorted(by_session.items()):
            ordered = sorted(revisions, key=lambda item: item.evaluated_at)
            timestamps = tuple(item.evaluated_at for item in ordered)
            if len(timestamps) != len(set(timestamps)):
                raise ValueError("reconciliation revisions have duplicate evaluation times")
            identities = {
                (
                    item.replay_evidence_sha256,
                    tuple(order.client_order_id for order in item.orders),
                )
                for item in ordered
            }
            if len(identities) != 1:
                raise ValueError(
                    f"reconciliation revisions change session identity: {session_date}"
                )
            latest.append(ordered[-1])
        unresolved = tuple(item.session_date for item in latest if item.reconciliation_break_count)
        age = (
            sum(unresolved[0] < session_date < control_date for session_date in session_dates)
            if unresolved
            else None
        )
        return ReconciliationAgeEvidence(
            schema_version=1,
            control_date=control_date,
            evaluated_at=evaluated_at,
            calendar_sha256=calendar.sha256,
            input_report_sha256=tuple(sorted(item.sha256 for item in reports)),
            latest_report_sha256=tuple(item.sha256 for item in latest),
            unresolved_session_dates=unresolved,
            reconciliation_break_age_sessions=age,
        )
