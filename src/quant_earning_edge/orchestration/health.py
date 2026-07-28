"""Authoritative unattended-workflow uptime and readiness evidence."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

from quant_earning_edge.orchestration.workflow import (
    DailyWorkflowState,
    DailyWorkflowStore,
    StageStatus,
    WorkflowTrigger,
)

if TYPE_CHECKING:
    from datetime import date
    from pathlib import Path

    from quant_earning_edge.data.calendar import SessionFile


@dataclass(frozen=True)
class WorkflowHealthReport:
    """Immutable scheduled-run evidence over authoritative market sessions."""

    schema_version: int
    calendar_sha256: str
    start_date: date
    end_date: date
    authoritative_session_dates: tuple[date, ...]
    workflow_state_sha256: tuple[str, ...]
    scheduled_complete_dates: tuple[date, ...]
    manual_complete_dates: tuple[date, ...]
    missing_dates: tuple[date, ...]
    failed_dates: tuple[date, ...]
    incomplete_dates: tuple[date, ...]
    invalid_artifact_dates: tuple[date, ...]
    operational_uptime: float
    maximum_consecutive_scheduled_successes: int
    excess_stage_attempt_count: int
    passes_five_session_unattended_gate: bool

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
                raise RuntimeError(f"workflow health report collision at {output}") from None


class WorkflowHealthEvaluator:
    """Count only intact scheduled completions as unattended operational uptime."""

    def evaluate(  # noqa: PLR0912 - exhaustive operational-state classification.
        self,
        *,
        calendar: SessionFile,
        store: DailyWorkflowStore,
        start_date: date,
        end_date: date,
    ) -> WorkflowHealthReport:
        if end_date < start_date:
            raise ValueError("workflow health end date must not precede start date")
        dates = tuple(
            session.session_date
            for session in calendar.sessions
            if start_date <= session.session_date <= end_date
        )
        if not dates:
            raise ValueError("workflow health range contains no authoritative sessions")

        states: list[DailyWorkflowState] = []
        missing: list[date] = []
        for session_date in dates:
            state = store.load_latest(session_date)
            if state is None:
                missing.append(session_date)
            else:
                states.append(state)

        scheduled_complete: list[date] = []
        manual_complete: list[date] = []
        failed: list[date] = []
        incomplete: list[date] = []
        invalid_artifacts: list[date] = []
        excess_attempts = 0
        by_date = {state.trade_date: state for state in states}
        for state in states:
            excess_attempts += sum(max(0, stage.attempts - 1) for stage in state.stages)
            artifact_valid = True
            try:
                state.verify_artifacts()
            except ValueError:
                artifact_valid = False
                invalid_artifacts.append(state.trade_date)
            if state.complete and artifact_valid:
                if state.trigger is WorkflowTrigger.SCHEDULED:
                    scheduled_complete.append(state.trade_date)
                else:
                    manual_complete.append(state.trade_date)
                continue
            if state.complete:
                continue
            current = next(
                stage for stage in state.stages if stage.status is not StageStatus.SUCCEEDED
            )
            if current.status is StageStatus.FAILED:
                failed.append(state.trade_date)
            else:
                incomplete.append(state.trade_date)

        scheduled_set = set(scheduled_complete)
        maximum_streak = _maximum_streak(dates, scheduled_set)
        return WorkflowHealthReport(
            schema_version=1,
            calendar_sha256=calendar.sha256,
            start_date=start_date,
            end_date=end_date,
            authoritative_session_dates=dates,
            workflow_state_sha256=tuple(by_date[item].sha256 for item in dates if item in by_date),
            scheduled_complete_dates=tuple(scheduled_complete),
            manual_complete_dates=tuple(manual_complete),
            missing_dates=tuple(missing),
            failed_dates=tuple(failed),
            incomplete_dates=tuple(incomplete),
            invalid_artifact_dates=tuple(invalid_artifacts),
            operational_uptime=len(scheduled_complete) / len(dates),
            maximum_consecutive_scheduled_successes=maximum_streak,
            excess_stage_attempt_count=excess_attempts,
            passes_five_session_unattended_gate=maximum_streak >= 5,
        )


def _maximum_streak(dates: tuple[date, ...], successes: set[date]) -> int:
    maximum = 0
    current = 0
    for session_date in dates:
        current = current + 1 if session_date in successes else 0
        maximum = max(maximum, current)
    return maximum
