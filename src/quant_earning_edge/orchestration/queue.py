"""Discover immutable daily inputs and queue the next authoritative workflow."""

from __future__ import annotations

import hashlib
import sys
from dataclasses import dataclass
from datetime import date, datetime  # noqa: TC003 - Pydantic resolves runtime fields.
from enum import StrEnum
from pathlib import Path  # noqa: TC003 - Pydantic resolves runtime fields.
from typing import TYPE_CHECKING, Self

import pyarrow.parquet as pq
from pydantic import BaseModel, ConfigDict, Field, model_validator

from quant_earning_edge.data.calendar import SessionFileStore
from quant_earning_edge.features import FEATURE_VALUE_SCHEMA
from quant_earning_edge.orchestration.commands import (
    CommandExecutor,
    WorkflowRunSpec,
    execute_qee_command,
)
from quant_earning_edge.orchestration.workflow import DailyWorkflowStore
from quant_earning_edge.universe.events import (
    EVENT_CANDIDATE_SCHEMA,
    EventCandidateManifest,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from quant_earning_edge.data.clients import MarketSession


class WorkflowQueueStatus(StrEnum):
    """One persistent-loop attempt to advance the proof queue."""

    WAITING_FOR_INPUTS = "waiting_for_inputs"
    ADMISSION_REQUIRED = "admission_required"
    QUEUED = "queued"
    ALREADY_QUEUED = "already_queued"
    PROOF_COMPLETE = "proof_complete"


@dataclass(frozen=True)
class WorkflowQueueResult:
    """Observable outcome of one next-session queue attempt."""

    status: WorkflowQueueStatus
    trade_date: date | None
    workflow_spec: Path | None
    detail: str

    @property
    def requires_attention(self) -> bool:
        return self.status in {
            WorkflowQueueStatus.WAITING_FOR_INPUTS,
            WorkflowQueueStatus.ADMISSION_REQUIRED,
        }


class WorkflowLoopSpec(BaseModel):
    """Stable deployment inputs reused across every proof session."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int = 1
    session_file: Path
    strategy_config: Path
    model_evidence: Path
    model_file: Path
    phase4_gate_file: Path
    proof_start: date
    proof_end: date
    initial_cash: float = Field(gt=0)
    artifact_root: Path
    staging_directory: Path
    worker_id: str = Field(min_length=1)
    universe_config: Path | None = None
    halt_snapshot_directory: Path | None = None
    feature_group: str = Field(default="earnings-v1", min_length=1)
    bar_lookback_calendar_days: int = Field(default=450, ge=365, le=730)
    freshness_symbol: str = Field(default="SPY", min_length=1)
    maximum_model_age_calendar_days: int = Field(default=180, ge=90, le=365)
    lease_seconds: int = Field(default=900, ge=1, le=3600)
    command_timeout_seconds: float = Field(default=1800, gt=0, le=7200)

    @model_validator(mode="after")
    def validate_contract(self) -> WorkflowLoopSpec:
        if self.schema_version != 1:
            raise ValueError("workflow loop schema_version must be 1")
        if self.proof_end < self.proof_start:
            raise ValueError("workflow loop proof_end precedes proof_start")
        if not self.worker_id.strip():
            raise ValueError("workflow loop worker_id must not be blank")
        if (self.universe_config is None) != (self.halt_snapshot_directory is None):
            raise ValueError(
                "workflow loop universe_config and halt_snapshot_directory are required together"
            )
        return self

    @classmethod
    def load(cls, path: Path) -> Self:
        """Load a loop spec and resolve deployment paths against its directory."""
        spec = cls.model_validate_json(path.read_bytes())
        base = path.resolve().parent
        updates = {}
        for field in (
            "session_file",
            "strategy_config",
            "model_evidence",
            "model_file",
            "phase4_gate_file",
            "artifact_root",
            "staging_directory",
            "universe_config",
            "halt_snapshot_directory",
        ):
            value = getattr(spec, field)
            if value is not None:
                updates[field] = _resolve(value, relative_to=base)
        return spec.model_copy(update=updates)


class NextWorkflowQueuer:
    """Advance at most one authoritative proof session per worker cycle."""

    def __init__(
        self,
        *,
        data_lake_root: Path,
        clock: Callable[[], datetime],
        executor: CommandExecutor = execute_qee_command,
    ) -> None:
        self._data_lake_root = data_lake_root.resolve()
        self._store = DailyWorkflowStore(self._data_lake_root)
        self._clock = clock
        self._executor = executor

    def run_once(  # noqa: PLR0911,PLR0912,PLR0915 - fail-closed queue boundaries.
        self,
        *,
        loop_spec: Path,
        inbox: Path,
    ) -> WorkflowQueueResult:
        """Discover, prepare, and queue the first unfinished proof session."""
        deployment = WorkflowLoopSpec.load(loop_spec)
        resolved_inbox = inbox.resolve()
        if not resolved_inbox.is_dir():
            raise ValueError(f"workflow inbox is not a directory: {resolved_inbox}")
        calendar = SessionFileStore.load(deployment.session_file)
        sessions = self._proof_sessions(
            calendar.sessions,
            proof_start=deployment.proof_start,
            proof_end=deployment.proof_end,
        )
        # Lazy imports avoid an evaluation -> orchestration -> signals ->
        # evaluation cycle while the CLI package graph is initializing.
        from quant_earning_edge.signals.config import (  # noqa: PLC0415
            load_strategy_config,
        )
        from quant_earning_edge.signals.production_model import (  # noqa: PLC0415
            ProductionModelArtifact,
        )

        strategy = load_strategy_config(deployment.strategy_config)
        model = ProductionModelArtifact.load(
            evidence_path=deployment.model_evidence,
            model_path=deployment.model_file,
        )
        from quant_earning_edge.evaluation.strategy_gate import (  # noqa: PLC0415
            Phase4PromotionEvidence,
        )

        promotion = Phase4PromotionEvidence.load(deployment.phase4_gate_file)
        if promotion.report_sha256 != model.phase4_gate_sha256:
            raise ValueError("workflow loop Phase 4 gate differs from production model")
        if model.hyperparameter_study_sha256 is None:
            raise ValueError("workflow loop production model lacks an Optuna study binding")
        if promotion.hyperparameter_study_sha256 != model.hyperparameter_study_sha256:
            raise ValueError("workflow loop Phase 4 Optuna study differs from production model")
        strategy_sha256 = hashlib.sha256(deployment.strategy_config.read_bytes()).hexdigest()
        if promotion.strategy_sha256 != strategy_sha256:
            raise ValueError("workflow loop Phase 4 strategy differs from strategy config")
        if model.training_cutoff > deployment.proof_start:
            raise ValueError("workflow loop model training cutoff follows proof start")
        model_age_at_end = (deployment.proof_end - model.training_cutoff).days
        if model_age_at_end > deployment.maximum_model_age_calendar_days:
            raise ValueError("workflow loop model will be stale before proof end")
        if (
            model.feature_names != strategy.features
            or model.label_name != strategy.label.column_name
            or model.threshold != strategy.label.threshold
            or model.seed != strategy.seed
        ):
            raise ValueError("workflow loop model contract differs from strategy config")
        now = self._aware_now()

        for index, session in enumerate(sessions):
            trade_date = session.session_date
            inbox_path = resolved_inbox / f"{trade_date.isoformat()}.json"
            staged_path = deployment.staging_directory.resolve() / f"{trade_date.isoformat()}.json"
            state = self._store.load_latest(trade_date)
            if inbox_path.is_file():
                self._validate_daily_spec(inbox_path, trade_date=trade_date)
                if state is not None and state.complete:
                    continue
                return WorkflowQueueResult(
                    status=WorkflowQueueStatus.ALREADY_QUEUED,
                    trade_date=trade_date,
                    workflow_spec=inbox_path,
                    detail="the next proof session is already in the worker inbox",
                )
            if state is not None and state.complete:
                continue
            if index and not self._prior_complete(sessions[index - 1].session_date):
                return WorkflowQueueResult(
                    status=WorkflowQueueStatus.WAITING_FOR_INPUTS,
                    trade_date=trade_date,
                    workflow_spec=None,
                    detail="the prior authoritative proof session is not complete",
                )
            if index == 0 and staged_path.is_file():
                self._validate_daily_spec(staged_path, trade_date=trade_date)
                return WorkflowQueueResult(
                    status=WorkflowQueueStatus.ADMISSION_REQUIRED,
                    trade_date=trade_date,
                    workflow_spec=staged_path,
                    detail="the first proof workflow is staged and requires proof-start admission",
                )
            prior_session = self._prior_session(calendar.sessions, trade_date=trade_date)
            if now < prior_session.close_at:
                return WorkflowQueueResult(
                    status=WorkflowQueueStatus.WAITING_FOR_INPUTS,
                    trade_date=trade_date,
                    workflow_spec=None,
                    detail="the authoritative prior session has not closed",
                )
            candidate_file = self._candidate_file(trade_date)
            if candidate_file is None and deployment.universe_config is not None:
                halt_file = self._halt_file(
                    deployment,
                    prior_date=prior_session.session_date,
                )
                if halt_file is None:
                    return WorkflowQueueResult(
                        status=WorkflowQueueStatus.WAITING_FOR_INPUTS,
                        trade_date=trade_date,
                        workflow_spec=None,
                        detail="the authoritative prior-close halt snapshot is not available",
                    )
                self._advance_upstream(
                    deployment=deployment,
                    trade_date=trade_date,
                    halt_snapshot_file=halt_file,
                    working_directory=loop_spec.resolve().parent,
                )
                candidate_file = self._candidate_file(trade_date)
            if candidate_file is None:
                return WorkflowQueueResult(
                    status=WorkflowQueueStatus.WAITING_FOR_INPUTS,
                    trade_date=trade_date,
                    workflow_spec=None,
                    detail="the immutable event-candidate artifact is not available",
                )
            symbols = self._candidate_symbols(candidate_file)
            feature_files = self._feature_files(
                feature_group=deployment.feature_group,
                asof_date=prior_session.session_date,
                symbols=symbols,
                feature_names=model.feature_names,
                observed_at=now,
                target_open=session.open_at,
            )
            if symbols and not feature_files and deployment.universe_config is not None:
                halt_file = self._halt_file(
                    deployment,
                    prior_date=prior_session.session_date,
                )
                if halt_file is None:
                    return WorkflowQueueResult(
                        status=WorkflowQueueStatus.WAITING_FOR_INPUTS,
                        trade_date=trade_date,
                        workflow_spec=None,
                        detail="the retained prior-close halt snapshot is not available",
                    )
                self._advance_upstream(
                    deployment=deployment,
                    trade_date=trade_date,
                    halt_snapshot_file=halt_file,
                    working_directory=loop_spec.resolve().parent,
                )
                feature_files = self._feature_files(
                    feature_group=deployment.feature_group,
                    asof_date=prior_session.session_date,
                    symbols=symbols,
                    feature_names=model.feature_names,
                    observed_at=self._aware_now(),
                    target_open=session.open_at,
                )
            if symbols and not feature_files:
                return WorkflowQueueResult(
                    status=WorkflowQueueStatus.WAITING_FOR_INPUTS,
                    trade_date=trade_date,
                    workflow_spec=None,
                    detail="a complete causal live-feature artifact is not available",
                )
            output = staged_path if index == 0 else inbox_path
            self._prepare(
                deployment=deployment,
                trade_date=trade_date,
                candidate_file=candidate_file,
                feature_files=feature_files,
                prior_replay_files=self._prior_replays(
                    deployment.artifact_root,
                    sessions=sessions[:index],
                ),
                output=output,
                stage_for_admission=index == 0,
                working_directory=loop_spec.resolve().parent,
            )
            self._validate_daily_spec(output, trade_date=trade_date)
            return WorkflowQueueResult(
                status=(
                    WorkflowQueueStatus.ADMISSION_REQUIRED
                    if index == 0
                    else WorkflowQueueStatus.QUEUED
                ),
                trade_date=trade_date,
                workflow_spec=output,
                detail=(
                    "the first proof workflow is staged and requires proof-start admission"
                    if index == 0
                    else "the next authoritative proof session was queued"
                ),
            )

        return WorkflowQueueResult(
            status=WorkflowQueueStatus.PROOF_COMPLETE,
            trade_date=None,
            workflow_spec=None,
            detail="every authoritative proof session is complete",
        )

    @staticmethod
    def _proof_sessions(
        sessions: Sequence[MarketSession],
        *,
        proof_start: date,
        proof_end: date,
    ) -> tuple[MarketSession, ...]:
        selected = tuple(item for item in sessions if proof_start <= item.session_date <= proof_end)
        if (
            not selected
            or selected[0].session_date != proof_start
            or selected[-1].session_date != proof_end
        ):
            raise ValueError("workflow loop proof boundaries are absent from the session file")
        return selected

    @staticmethod
    def _prior_session(
        sessions: Sequence[MarketSession],
        *,
        trade_date: date,
    ) -> MarketSession:
        earlier = tuple(item for item in sessions if item.session_date < trade_date)
        if not earlier:
            raise ValueError("workflow loop requires the prior authoritative session")
        return earlier[-1]

    def _prior_complete(self, trade_date: date) -> bool:
        state = self._store.load_latest(trade_date)
        return state is not None and state.complete

    def _candidate_file(self, trade_date: date) -> Path | None:
        root = (
            self._data_lake_root
            / "gold"
            / "event-candidates"
            / f"for_trade_date={trade_date.isoformat()}"
        )
        candidates = []
        for candidate in sorted(root.glob("candidates-*.parquet")):
            identity = candidate.stem.removeprefix("candidates-")
            manifest_path = candidate.with_name(f"manifest-{identity}.json")
            if not manifest_path.is_file():
                continue
            manifest = EventCandidateManifest.load(manifest_path)
            if (
                manifest.raw["schema_version"] == 5
                and manifest.universe_lineage_entries
                and manifest.event_lineage_entries
                and manifest.calendar_lineage_entries
            ):
                candidates.append(candidate)
        if not candidates:
            return None
        if len(candidates) > 1:
            raise ValueError(
                f"multiple event-candidate artifacts exist for {trade_date}; "
                "the authoritative revision is ambiguous"
            )
        candidate = candidates[0].resolve()
        identity = candidate.stem.removeprefix("candidates-")
        manifest_path = candidate.with_name(f"manifest-{identity}.json")
        if not manifest_path.is_file():
            return None
        manifest = EventCandidateManifest.load(manifest_path)
        if (
            manifest.raw["trade_date"] != trade_date.isoformat()
            or manifest.raw["candidate_file_sha256"]
            != hashlib.sha256(candidate.read_bytes()).hexdigest()
        ):
            raise ValueError("workflow queue candidate manifest differs from its artifact")
        manifest.source_paths(data_lake_root=self._data_lake_root)
        self._candidate_symbols(candidate)
        return candidate

    @staticmethod
    def _candidate_symbols(path: Path) -> tuple[str, ...]:
        if pq.read_schema(path) != EVENT_CANDIDATE_SCHEMA:  # type: ignore[no-untyped-call]
            raise ValueError("workflow queue candidate artifact schema mismatch")
        rows = pq.ParquetFile(path).read(columns=["symbol"]).to_pylist()  # type: ignore[no-untyped-call]
        symbols = tuple(str(row["symbol"]).strip().upper() for row in rows)
        if symbols != tuple(sorted(set(symbols))):
            raise ValueError("workflow queue candidate symbols must be unique and sorted")
        return symbols

    def _feature_files(
        self,
        *,
        feature_group: str,
        asof_date: date,
        symbols: tuple[str, ...],
        feature_names: tuple[str, ...],
        observed_at: datetime,
        target_open: datetime,
    ) -> tuple[Path, ...]:
        if not symbols:
            return ()
        month = asof_date.isoformat()[:7]
        root = self._data_lake_root / "gold" / f"feature_group={feature_group}" / f"month={month}"
        matches: list[tuple[datetime, Path]] = []
        expected_keys = {
            (symbol, feature_name) for symbol in symbols for feature_name in feature_names
        }
        for path in sorted(root.glob("part-*.parquet")):
            if pq.read_schema(path) != FEATURE_VALUE_SCHEMA:  # type: ignore[no-untyped-call]
                continue
            rows = [
                row
                for row in pq.read_table(path).to_pylist()  # type: ignore[no-untyped-call]
                if row["asof_date"] == asof_date
            ]
            keys = {(str(row["symbol"]).strip().upper(), str(row["feature_name"])) for row in rows}
            computed = {row["computed_at"] for row in rows}
            if (
                keys == expected_keys
                and len(rows) == len(expected_keys)
                and len(computed) == 1
                and (computed_at := next(iter(computed))) <= observed_at
                and computed_at < target_open
            ):
                matches.append((computed_at, path.resolve()))
        if not matches:
            return ()
        latest = max(item[0] for item in matches)
        latest_paths = tuple(path for computed_at, path in matches if computed_at == latest)
        if len(latest_paths) > 1:
            raise ValueError("latest causal live-feature artifact is ambiguous")
        return latest_paths

    @staticmethod
    def _prior_replays(
        artifact_root: Path,
        *,
        sessions: Sequence[MarketSession],
    ) -> tuple[Path, ...]:
        paths = tuple(
            artifact_root.resolve()
            / f"trade_date={session.session_date.isoformat()}"
            / "replay-session.json"
            for session in sessions
        )
        missing = tuple(path for path in paths if not path.is_file())
        if missing:
            raise ValueError(f"prior proof replay is missing: {missing[0]}")
        return paths

    def _prepare(
        self,
        *,
        deployment: WorkflowLoopSpec,
        trade_date: date,
        candidate_file: Path,
        feature_files: tuple[Path, ...],
        prior_replay_files: tuple[Path, ...],
        output: Path,
        stage_for_admission: bool,
        working_directory: Path,
    ) -> None:
        arguments = [
            sys.executable,
            "-m",
            "quant_earning_edge.cli",
            "workflow",
            "prepare",
            "--trade-date",
            trade_date.isoformat(),
            "--candidate-file",
            str(candidate_file),
            "--model-evidence",
            str(deployment.model_evidence),
            "--model-file",
            str(deployment.model_file),
            "--strategy-config",
            str(deployment.strategy_config),
            "--session-file",
            str(deployment.session_file),
            "--proof-start",
            deployment.proof_start.isoformat(),
            "--proof-end",
            deployment.proof_end.isoformat(),
            "--initial-cash",
            str(deployment.initial_cash),
            "--artifact-root",
            str(deployment.artifact_root),
            "--output",
            str(output.resolve()),
            "--worker-id",
            deployment.worker_id,
            "--freshness-symbol",
            deployment.freshness_symbol,
            "--lease-seconds",
            str(deployment.lease_seconds),
            "--command-timeout-seconds",
            str(deployment.command_timeout_seconds),
        ]
        for path in feature_files:
            arguments.extend(("--feature-file", str(path)))
        for path in prior_replay_files:
            arguments.extend(("--prior-replay-file", str(path)))
        if stage_for_admission:
            arguments.append("--stage-for-admission")
        result = self._executor(
            tuple(arguments),
            cwd=working_directory,
            timeout_seconds=deployment.command_timeout_seconds,
        )
        if result.return_code:
            detail = result.stderr.strip() or result.stdout.strip() or "unknown error"
            raise RuntimeError(f"workflow prepare exited {result.return_code}: {detail[:1000]}")

    def _advance_upstream(
        self,
        *,
        deployment: WorkflowLoopSpec,
        trade_date: date,
        halt_snapshot_file: Path,
        working_directory: Path,
    ) -> None:
        assert deployment.universe_config is not None
        result = self._executor(
            (
                sys.executable,
                "-m",
                "quant_earning_edge.cli",
                "workflow",
                "prepare-session-inputs",
                "--trade-date",
                trade_date.isoformat(),
                "--session-file",
                str(deployment.session_file),
                "--halt-snapshot-file",
                str(halt_snapshot_file),
                "--strategy-config",
                str(deployment.strategy_config),
                "--universe-config",
                str(deployment.universe_config),
                "--feature-group",
                deployment.feature_group,
                "--bar-lookback-calendar-days",
                str(deployment.bar_lookback_calendar_days),
            ),
            cwd=working_directory,
            timeout_seconds=deployment.command_timeout_seconds,
        )
        if result.return_code:
            detail = result.stderr.strip() or result.stdout.strip() or "unknown error"
            raise RuntimeError(
                f"daily input preparation exited {result.return_code}: {detail[:1000]}"
            )

    @staticmethod
    def _halt_file(
        deployment: WorkflowLoopSpec,
        *,
        prior_date: date,
    ) -> Path | None:
        assert deployment.halt_snapshot_directory is not None
        path = deployment.halt_snapshot_directory.resolve() / f"halt-{prior_date.isoformat()}.json"
        return path if path.is_file() else None

    @staticmethod
    def _validate_daily_spec(path: Path, *, trade_date: date) -> None:
        spec = WorkflowRunSpec.model_validate_json(path.read_bytes())
        if spec.trade_date != trade_date:
            raise ValueError("queued workflow spec trade date differs from its session")

    def _aware_now(self) -> datetime:
        value = self._clock()
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("workflow queue clock must be timezone-aware")
        return value


def _resolve(path: Path, *, relative_to: Path) -> Path:
    return path.resolve() if path.is_absolute() else (relative_to / path).resolve()
