"""Credential-safe operational readiness evidence for the unattended workflow."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING
from urllib.parse import urlparse

if TYPE_CHECKING:
    from datetime import date, datetime, timedelta
    from pathlib import Path

    from quant_earning_edge.data.calendar import SessionFile
    from quant_earning_edge.monitoring import ProviderFreshnessEvidence
    from quant_earning_edge.orchestration.worker import WorkerCycleReport


@dataclass(frozen=True)
class ReadinessCheck:
    """One secret-free, deterministic readiness assertion."""

    name: str
    passed: bool
    detail: str

    def __post_init__(self) -> None:
        if not self.name.strip() or not self.detail.strip():
            raise ValueError("readiness check name and detail must not be blank")


@dataclass(frozen=True)
class OperationalReadinessReport:
    """Immutable audit of prerequisites for a real unattended proof run."""

    schema_version: int
    evaluated_at: datetime
    control_date: date
    session_file_sha256: str
    provider_freshness_sha256: str | None
    latest_worker_cycle_sha256: str | None
    checks: tuple[ReadinessCheck, ...]
    ready: bool

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("unsupported operational readiness schema")
        if self.evaluated_at.tzinfo is None or self.evaluated_at.utcoffset() is None:
            raise ValueError("operational readiness time must be timezone-aware")
        names = tuple(item.name for item in self.checks)
        if names != tuple(sorted(set(names))):
            raise ValueError("operational readiness checks must be unique and sorted")
        if self.ready != all(item.passed for item in self.checks):
            raise ValueError("operational readiness verdict is inconsistent")
        for digest in (
            self.session_file_sha256,
            self.provider_freshness_sha256,
            self.latest_worker_cycle_sha256,
        ):
            if digest is not None and not _is_sha256(digest):
                raise ValueError("operational readiness digest is invalid")

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
                raise RuntimeError(f"operational readiness collision at {output}") from None


class OperationalReadinessEvaluator:
    """Evaluate local, credential, heartbeat, evidence, and live-provider gates."""

    def evaluate(
        self,
        *,
        calendar: SessionFile,
        control_date: date,
        evaluated_at: datetime,
        minimum_calendar_sessions: int,
        data_lake_root: Path,
        artifact_root: Path,
        inbox: Path,
        polygon_base_url: str,
        finnhub_base_url: str,
        alpaca_base_url: str,
        polygon_credential_configured: bool,
        finnhub_credential_configured: bool,
        alpaca_credentials_configured: bool,
        provider_freshness: ProviderFreshnessEvidence | None,
        provider_probe_error: str | None,
        polygon_nbbo_entitlement_verified: bool,
        polygon_nbbo_entitlement_detail: str,
        worker_id: str,
        latest_worker_cycle: WorkerCycleReport | None,
        maximum_heartbeat_age: timedelta,
        bootstrap_source_count: int,
        bootstrap_error: str | None,
    ) -> OperationalReadinessReport:
        if evaluated_at.tzinfo is None or evaluated_at.utcoffset() is None:
            raise ValueError("operational readiness time must be timezone-aware")
        if minimum_calendar_sessions < 1 or maximum_heartbeat_age.total_seconds() <= 0:
            raise ValueError("operational readiness thresholds must be positive")
        session_dates = tuple(item.session_date for item in calendar.sessions)
        heartbeat_age = (
            evaluated_at - latest_worker_cycle.evaluated_at
            if latest_worker_cycle is not None
            else None
        )
        polygon_age = (
            evaluated_at - provider_freshness.polygon_data_observed_at
            if provider_freshness is not None
            else None
        )
        alpaca_age = (
            evaluated_at - provider_freshness.alpaca_data_observed_at
            if provider_freshness is not None
            else None
        )
        checks = (
            _check(
                "alpaca_credentials_configured",
                alpaca_credentials_configured,
                "configured" if alpaca_credentials_configured else "missing",
            ),
            _check(
                "alpaca_paper_host",
                _canonical_host(alpaca_base_url) == "paper-api.alpaca.markets",
                _canonical_host(alpaca_base_url) or "invalid",
            ),
            _check(
                "artifact_root_writable",
                _writable_directory(artifact_root),
                str(artifact_root.resolve()),
            ),
            _check(
                "bootstrap_evidence",
                bootstrap_error is None and bootstrap_source_count > 0,
                (
                    f"{bootstrap_source_count} completed prior session(s)"
                    if bootstrap_error is None
                    else bootstrap_error
                ),
            ),
            _check(
                "calendar_control_date",
                control_date in session_dates,
                control_date.isoformat(),
            ),
            _check(
                "calendar_session_span",
                len(session_dates) >= minimum_calendar_sessions,
                f"{len(session_dates)}/{minimum_calendar_sessions} sessions",
            ),
            _check(
                "data_lake_root_writable",
                _writable_directory(data_lake_root),
                str(data_lake_root.resolve()),
            ),
            _check(
                "finnhub_credential_configured",
                finnhub_credential_configured,
                "configured" if finnhub_credential_configured else "missing",
            ),
            _check(
                "finnhub_host",
                _canonical_host(finnhub_base_url) == "finnhub.io",
                _canonical_host(finnhub_base_url) or "invalid",
            ),
            _check(
                "inbox_writable",
                _writable_directory(inbox),
                str(inbox.resolve()),
            ),
            _check(
                "polygon_credential_configured",
                polygon_credential_configured,
                "configured" if polygon_credential_configured else "missing",
            ),
            _check(
                "polygon_host",
                _canonical_host(polygon_base_url) == "api.polygon.io",
                _canonical_host(polygon_base_url) or "invalid",
            ),
            _check(
                "polygon_nbbo_entitlement",
                polygon_nbbo_entitlement_verified,
                polygon_nbbo_entitlement_detail,
            ),
            _check(
                "provider_live_probe",
                (
                    provider_freshness is not None
                    and provider_probe_error is None
                    and polygon_age is not None
                    and alpaca_age is not None
                    and polygon_age.total_seconds() <= 30 * 60
                    and alpaca_age.total_seconds() <= 30 * 60
                    and polygon_age.total_seconds() >= 0
                    and alpaca_age.total_seconds() >= 0
                ),
                (
                    provider_probe_error
                    or (
                        f"freshness={provider_freshness.sha256}"
                        if provider_freshness is not None
                        else "not executed"
                    )
                ),
            ),
            _check(
                "worker_heartbeat",
                (
                    latest_worker_cycle is not None
                    and latest_worker_cycle.worker_id == worker_id
                    and heartbeat_age is not None
                    and heartbeat_age.total_seconds() >= 0
                    and heartbeat_age <= maximum_heartbeat_age
                ),
                (
                    f"worker={latest_worker_cycle.worker_id}; age={heartbeat_age}"
                    if latest_worker_cycle is not None
                    else "missing"
                ),
            ),
        )
        ordered = tuple(sorted(checks, key=lambda item: item.name))
        return OperationalReadinessReport(
            schema_version=1,
            evaluated_at=evaluated_at,
            control_date=control_date,
            session_file_sha256=calendar.sha256,
            provider_freshness_sha256=(
                provider_freshness.sha256 if provider_freshness is not None else None
            ),
            latest_worker_cycle_sha256=(
                latest_worker_cycle.sha256 if latest_worker_cycle is not None else None
            ),
            checks=ordered,
            ready=all(item.passed for item in ordered),
        )


def _check(name: str, passed: bool, detail: str) -> ReadinessCheck:
    return ReadinessCheck(
        name=name,
        passed=passed,
        detail=(detail.strip() or "unavailable")[:500],
    )


def _canonical_host(url: str) -> str:
    parsed = urlparse(url)
    return parsed.hostname or ""


def _writable_directory(path: Path) -> bool:
    resolved = path.resolve()
    return resolved.is_dir() and os.access(resolved, os.W_OK)


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(item in "0123456789abcdef" for item in value)
