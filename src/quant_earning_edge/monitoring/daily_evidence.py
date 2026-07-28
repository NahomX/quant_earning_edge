"""Discover prior immutable workflow evidence for pre-open controls."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import TYPE_CHECKING

from quant_earning_edge.evaluation import ReplaySessionReport
from quant_earning_edge.live import PaperReconciliationReport
from quant_earning_edge.monitoring.control_inputs import CompletedReplayControlSource
from quant_earning_edge.signals import FrozenDailyOrders

if TYPE_CHECKING:
    from pathlib import Path

    from quant_earning_edge.data.calendar import SessionFile


@dataclass(frozen=True)
class DiscoveredDailyControlEvidence:
    """Validated completed replay pairs and all reconciliation revisions."""

    replay_sources: tuple[CompletedReplayControlSource, ...]
    reconciliation_reports: tuple[PaperReconciliationReport, ...]
    frozen_order_files: tuple[Path, ...]
    replay_report_files: tuple[Path, ...]
    reconciliation_report_files: tuple[Path, ...]


class DailyControlEvidenceDiscovery:
    """Discover deterministic prior-session artifacts without accepting partial pairs."""

    def discover(
        self,
        *,
        calendar: SessionFile,
        artifact_root: Path,
        control_date: date,
    ) -> DiscoveredDailyControlEvidence:
        authoritative_dates = {item.session_date for item in calendar.sessions}
        if control_date not in authoritative_dates:
            raise ValueError("daily control date is not authoritative")
        frozen_files: list[Path] = []
        replay_files: list[Path] = []
        sources: list[CompletedReplayControlSource] = []
        reconciliation_files: list[Path] = []
        reconciliation_reports: list[PaperReconciliationReport] = []

        for directory in sorted(artifact_root.resolve().glob("trade_date=*"), key=str):
            if not directory.is_dir():
                continue
            session_date = _session_date(directory.name)
            if session_date >= control_date:
                continue
            if session_date not in authoritative_dates:
                raise ValueError(f"workflow artifact date is not authoritative: {session_date}")
            frozen_path = directory / "frozen-daily-orders.json"
            replay_path = directory / "replay-session.json"
            if frozen_path.is_file() != replay_path.is_file():
                raise ValueError(f"incomplete prior replay control pair for session {session_date}")
            if frozen_path.is_file():
                frozen = FrozenDailyOrders.load(frozen_path)
                replay = ReplaySessionReport.load(replay_path)
                if frozen.trade_date != session_date or replay.session_date != session_date:
                    raise ValueError(
                        f"prior replay control path identity differs for {session_date}"
                    )
                frozen_files.append(frozen_path)
                replay_files.append(replay_path)
                sources.append(
                    CompletedReplayControlSource(
                        frozen_orders=frozen,
                        replay_report=replay,
                    )
                )

            report_paths = set(directory.glob("paper-reconciliation-*.json"))
            legacy_report = directory / "paper-reconciliation.json"
            if legacy_report.is_file():
                report_paths.add(legacy_report)
            for report_path in sorted(report_paths, key=str):
                report = PaperReconciliationReport.load(report_path)
                if report.session_date != session_date:
                    raise ValueError(
                        f"paper reconciliation path identity differs for {session_date}"
                    )
                reconciliation_files.append(report_path)
                reconciliation_reports.append(report)

        if not sources:
            raise ValueError("no completed prior replay controls were discovered")
        return DiscoveredDailyControlEvidence(
            replay_sources=tuple(sources),
            reconciliation_reports=tuple(reconciliation_reports),
            frozen_order_files=tuple(frozen_files),
            replay_report_files=tuple(replay_files),
            reconciliation_report_files=tuple(reconciliation_files),
        )


def _session_date(directory_name: str) -> date:
    try:
        return date.fromisoformat(directory_name.removeprefix("trade_date="))
    except ValueError as error:
        raise ValueError(f"invalid workflow artifact directory: {directory_name}") from error
