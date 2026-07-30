"""Fail-closed admission of the first scheduled execution-realism proof session."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import date, datetime
from typing import TYPE_CHECKING, Any

from quant_earning_edge.orchestration.workflow import WorkflowTrigger

if TYPE_CHECKING:
    from datetime import timedelta
    from pathlib import Path

    from quant_earning_edge.data.calendar import SessionFile
    from quant_earning_edge.orchestration.commands import WorkflowRunSpec
    from quant_earning_edge.orchestration.readiness import OperationalReadinessReport


@dataclass(frozen=True)
class NoTradeSmokeEvidence:
    """Strict successful result from the isolated credentialed smoke workflow."""

    schema_version: int
    smoke_date: date
    trigger: str
    counts_toward_phase6: bool
    intended_order_count: int
    workflow_spec_sha256: str
    workflow_state_sha256: str
    worker_cycle_sha256: str
    artifact_root: str

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("unsupported no-trade smoke schema")
        if self.trigger != "manual" or self.counts_toward_phase6 or self.intended_order_count != 0:
            raise ValueError("smoke evidence is not a successful proof-excluded no-trade run")
        for digest in (
            self.workflow_spec_sha256,
            self.workflow_state_sha256,
            self.worker_cycle_sha256,
        ):
            if not _is_sha256(digest):
                raise ValueError("no-trade smoke digest is invalid")
        if not self.artifact_root.strip():
            raise ValueError("no-trade smoke artifact root is blank")

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

    @classmethod
    def load(cls, path: Path) -> NoTradeSmokeEvidence:
        raw: Any = json.loads(path.read_bytes())
        expected = {
            "schema_version",
            "smoke_date",
            "trigger",
            "counts_toward_phase6",
            "intended_order_count",
            "workflow_spec_sha256",
            "workflow_state_sha256",
            "worker_cycle_sha256",
            "artifact_root",
        }
        if not isinstance(raw, dict) or set(raw) != expected:
            raise ValueError("no-trade smoke evidence schema mismatch")
        try:
            evidence = cls(
                schema_version=int(raw["schema_version"]),
                smoke_date=date.fromisoformat(str(raw["smoke_date"])),
                trigger=str(raw["trigger"]),
                counts_toward_phase6=bool(raw["counts_toward_phase6"]),
                intended_order_count=int(raw["intended_order_count"]),
                workflow_spec_sha256=str(raw["workflow_spec_sha256"]),
                workflow_state_sha256=str(raw["workflow_state_sha256"]),
                worker_cycle_sha256=str(raw["worker_cycle_sha256"]),
                artifact_root=str(raw["artifact_root"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("invalid no-trade smoke evidence") from error
        if evidence.canonical_bytes != path.read_bytes():
            raise ValueError("no-trade smoke evidence is not canonical")
        return evidence


@dataclass(frozen=True)
class ProofStartAdmission:
    """Content-linked authorization to place one first-session spec in the inbox."""

    schema_version: int
    admitted_at: datetime
    proof_start: date
    session_file_sha256: str
    readiness_sha256: str
    smoke_sha256: str
    workflow_spec_sha256: str

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("unsupported proof-start admission schema")
        if self.admitted_at.tzinfo is None or self.admitted_at.utcoffset() is None:
            raise ValueError("proof-start admission time must be timezone-aware")
        for digest in (
            self.session_file_sha256,
            self.readiness_sha256,
            self.smoke_sha256,
            self.workflow_spec_sha256,
        ):
            if not _is_sha256(digest):
                raise ValueError("proof-start admission digest is invalid")

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


class ProofStartAdmitter:
    """Require fresh readiness and successful smoke before inbox publication."""

    def admit(
        self,
        *,
        calendar: SessionFile,
        proof_start: date,
        admitted_at: datetime,
        maximum_readiness_age: timedelta,
        readiness: OperationalReadinessReport,
        smoke: NoTradeSmokeEvidence,
        workflow_spec: WorkflowRunSpec,
    ) -> ProofStartAdmission:
        if admitted_at.tzinfo is None or admitted_at.utcoffset() is None:
            raise ValueError("proof-start admission time must be timezone-aware")
        if maximum_readiness_age.total_seconds() <= 0:
            raise ValueError("maximum readiness age must be positive")
        if proof_start not in {item.session_date for item in calendar.sessions}:
            raise ValueError("proof start is not an authoritative market session")
        if not readiness.ready:
            raise ValueError("operational readiness report did not pass")
        if readiness.control_date != proof_start:
            raise ValueError("operational readiness control date differs from proof start")
        if readiness.session_file_sha256 != calendar.sha256:
            raise ValueError("operational readiness used a different session file")
        readiness_age = admitted_at - readiness.evaluated_at
        if readiness_age.total_seconds() < 0 or readiness_age > maximum_readiness_age:
            raise ValueError("operational readiness report is stale or from the future")
        if smoke.smoke_date >= proof_start:
            raise ValueError("credentialed no-trade smoke must precede proof start")
        if workflow_spec.trade_date != proof_start:
            raise ValueError("workflow specification date differs from proof start")
        if workflow_spec.trigger is not WorkflowTrigger.SCHEDULED:
            raise ValueError("first proof workflow must use the scheduled trigger")
        return ProofStartAdmission(
            schema_version=1,
            admitted_at=admitted_at,
            proof_start=proof_start,
            session_file_sha256=calendar.sha256,
            readiness_sha256=readiness.sha256,
            smoke_sha256=smoke.sha256,
            workflow_spec_sha256=workflow_spec.sha256,
        )

    @staticmethod
    def write(
        admission: ProofStartAdmission,
        *,
        workflow_spec: WorkflowRunSpec,
        inbox_output: Path,
        evidence_output: Path,
    ) -> None:
        workflow_spec.write(inbox_output)
        _write_immutable(evidence_output, admission.canonical_bytes)


def _write_immutable(path: Path, encoded: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as destination:
            destination.write(encoded)
    except FileExistsError:
        if path.read_bytes() != encoded:
            raise RuntimeError(f"proof-start admission collision at {path}") from None


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(item in "0123456789abcdef" for item in value)
