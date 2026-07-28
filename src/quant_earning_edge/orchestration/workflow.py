"""Append-only state machine and execution loop for one trading session."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, replace
from datetime import date, datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, ValidationError

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


class WorkflowStage(StrEnum):
    """Required stages in causal operational order."""

    FREEZE_INPUTS = "freeze_inputs"
    GENERATE_ORDER_PLAN = "generate_order_plan"
    EVALUATE_BREAKERS = "evaluate_breakers"
    SUBMIT_PAPER_ORDERS = "submit_paper_orders"
    CAPTURE_MARKET_EVENTS = "capture_market_events"
    REPLAY_ORDERS = "replay_orders"
    RECONCILE_SESSION = "reconcile_session"
    EVALUATE_PHASE6_PROGRESS = "evaluate_phase6_progress"


STAGE_ORDER = tuple(WorkflowStage)


class StageStatus(StrEnum):
    """Durable lifecycle of one required stage."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class WorkflowTrigger(StrEnum):
    """Invocation provenance; only scheduled runs count as unattended proof."""

    MANUAL = "manual"
    SCHEDULED = "scheduled"


@dataclass(frozen=True)
class ArtifactReference:
    """Content identity for one stage output."""

    path: str
    sha256: str
    size_bytes: int

    def __post_init__(self) -> None:
        if not self.path.strip():
            raise ValueError("artifact path must not be blank")
        _validate_digest("artifact sha256", self.sha256)
        if self.size_bytes < 0:
            raise ValueError("artifact size cannot be negative")

    @classmethod
    def capture(cls, path: Path) -> ArtifactReference:
        """Hash an existing file and retain its absolute path."""
        resolved = path.resolve()
        if not resolved.is_file():
            raise ValueError(f"workflow artifact does not exist or is not a file: {resolved}")
        payload = resolved.read_bytes()
        return cls(
            path=str(resolved),
            sha256=hashlib.sha256(payload).hexdigest(),
            size_bytes=len(payload),
        )

    def verify(self) -> None:
        """Reject a missing or changed artifact."""
        path = Path(self.path)
        if not path.is_file():
            raise ValueError(f"workflow artifact is missing: {path}")
        payload = path.read_bytes()
        if len(payload) != self.size_bytes or hashlib.sha256(payload).hexdigest() != self.sha256:
            raise ValueError(f"workflow artifact changed after capture: {path}")


@dataclass(frozen=True)
class StageRecord:
    """One required stage and its latest retry state."""

    stage: WorkflowStage
    status: StageStatus
    attempts: int
    worker_id: str | None
    started_at: datetime | None
    lease_expires_at: datetime | None
    completed_at: datetime | None
    input_sha256: tuple[str, ...]
    output_artifacts: tuple[ArtifactReference, ...]
    error_type: str | None
    error_message: str | None

    def __post_init__(self) -> None:
        _validate_stage_timestamps(self)
        if self.attempts < 0:
            raise ValueError("stage attempts cannot be negative")
        _validate_stage_status(self)
        for digest in self.input_sha256:
            _validate_digest("stage input sha256", digest)


@dataclass(frozen=True)
class DailyWorkflowState:
    """One immutable revision of the full daily operational workflow."""

    schema_version: int
    trade_date: date
    trigger: WorkflowTrigger
    revision: int
    previous_sha256: str | None
    created_at: datetime
    updated_at: datetime
    stages: tuple[StageRecord, ...]

    def __post_init__(self) -> None:
        _require_aware("created_at", self.created_at)
        _require_aware("updated_at", self.updated_at)
        if self.updated_at < self.created_at:
            raise ValueError("workflow update cannot precede creation")
        if self.revision < 0:
            raise ValueError("workflow revision cannot be negative")
        if self.schema_version != 2:
            raise ValueError("unsupported workflow schema version")
        if (self.revision == 0) != (self.previous_sha256 is None):
            raise ValueError("only the initial workflow revision can omit previous_sha256")
        if self.previous_sha256 is not None:
            _validate_digest("workflow previous_sha256", self.previous_sha256)
        if tuple(item.stage for item in self.stages) != STAGE_ORDER:
            raise ValueError("workflow must contain every required stage in exact order")
        running_count = sum(item.status is StageStatus.RUNNING for item in self.stages)
        if running_count > 1:
            raise ValueError("only one workflow stage may run at a time")
        incomplete_seen = False
        for stage in self.stages:
            if incomplete_seen and stage.status is not StageStatus.PENDING:
                raise ValueError("later workflow stages must remain pending")
            if stage.status is not StageStatus.SUCCEEDED:
                incomplete_seen = True

    @classmethod
    def initialize(
        cls,
        *,
        trade_date: date,
        now: datetime,
        trigger: WorkflowTrigger = WorkflowTrigger.MANUAL,
    ) -> DailyWorkflowState:
        """Create the first pending revision."""
        _require_aware("now", now)
        return cls(
            schema_version=2,
            trade_date=trade_date,
            trigger=trigger,
            revision=0,
            previous_sha256=None,
            created_at=now,
            updated_at=now,
            stages=tuple(
                StageRecord(
                    stage=stage,
                    status=StageStatus.PENDING,
                    attempts=0,
                    worker_id=None,
                    started_at=None,
                    lease_expires_at=None,
                    completed_at=None,
                    input_sha256=(),
                    output_artifacts=(),
                    error_type=None,
                    error_message=None,
                )
                for stage in STAGE_ORDER
            ),
        )

    @property
    def complete(self) -> bool:
        return all(item.status is StageStatus.SUCCEEDED for item in self.stages)

    def verify_artifacts(self) -> None:
        """Verify every artifact retained by completed stages."""
        for stage in self.stages:
            for artifact in stage.output_artifacts:
                artifact.verify()

    @property
    def canonical_bytes(self) -> bytes:
        return json.dumps(
            asdict(self),
            default=lambda item: item.value if isinstance(item, StrEnum) else item.isoformat(),
            sort_keys=True,
            separators=(",", ":"),
        ).encode()

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes).hexdigest()

    @classmethod
    def load(cls, path: Path) -> DailyWorkflowState:
        """Load strict state and reject noncanonical or internally invalid evidence."""
        try:
            raw = json.loads(path.read_bytes())
            state = DailyWorkflowStateSpec.model_validate(raw).to_domain()
        except (json.JSONDecodeError, OSError, TypeError, ValidationError, ValueError) as error:
            raise ValueError(f"invalid daily workflow state: {path}") from error
        if json.loads(state.canonical_bytes) != raw:
            raise ValueError("daily workflow state is not canonical or uses unsupported fields")
        return state


class DailyWorkflowController:
    """Apply deterministic stage claims, completions, and retryable failures."""

    def claim_next(
        self,
        state: DailyWorkflowState,
        *,
        worker_id: str,
        now: datetime,
        lease_duration: timedelta = timedelta(minutes=15),
    ) -> DailyWorkflowState | None:
        _require_aware("now", now)
        normalized_worker = worker_id.strip()
        if not normalized_worker:
            raise ValueError("worker_id must not be blank")
        if lease_duration <= timedelta(0):
            raise ValueError("lease duration must be positive")
        if state.complete:
            return None
        index = next(
            index
            for index, record in enumerate(state.stages)
            if record.status is not StageStatus.SUCCEEDED
        )
        current = state.stages[index]
        if (
            current.status is StageStatus.RUNNING
            and current.lease_expires_at is not None
            and current.lease_expires_at > now
        ):
            return None
        input_hashes = (
            tuple(item.sha256 for item in state.stages[index - 1].output_artifacts) if index else ()
        )
        claimed = StageRecord(
            stage=current.stage,
            status=StageStatus.RUNNING,
            attempts=current.attempts + 1,
            worker_id=normalized_worker,
            started_at=now,
            lease_expires_at=now + lease_duration,
            completed_at=None,
            input_sha256=input_hashes,
            output_artifacts=(),
            error_type=None,
            error_message=None,
        )
        return _advance(state, now=now, stages=_replace_stage(state.stages, index, claimed))

    def succeed(
        self,
        state: DailyWorkflowState,
        *,
        worker_id: str,
        stage: WorkflowStage,
        artifacts: Sequence[ArtifactReference],
        now: datetime,
    ) -> DailyWorkflowState:
        index, current = _owned_running_stage(state, worker_id=worker_id, stage=stage)
        captured = tuple(artifacts)
        if not captured:
            raise ValueError("successful workflow stage requires at least one artifact")
        paths = tuple(item.path for item in captured)
        if len(set(paths)) != len(paths):
            raise ValueError("successful workflow stage contains duplicate artifact paths")
        for artifact in captured:
            artifact.verify()
        succeeded = replace(
            current,
            status=StageStatus.SUCCEEDED,
            lease_expires_at=None,
            completed_at=now,
            output_artifacts=captured,
            error_type=None,
            error_message=None,
        )
        return _advance(state, now=now, stages=_replace_stage(state.stages, index, succeeded))

    def fail(
        self,
        state: DailyWorkflowState,
        *,
        worker_id: str,
        stage: WorkflowStage,
        error: Exception,
        now: datetime,
    ) -> DailyWorkflowState:
        index, current = _owned_running_stage(state, worker_id=worker_id, stage=stage)
        message = str(error).strip() or "stage failed without an error message"
        failed = replace(
            current,
            status=StageStatus.FAILED,
            lease_expires_at=None,
            completed_at=now,
            output_artifacts=(),
            error_type=type(error).__name__,
            error_message=message[:1000],
        )
        return _advance(state, now=now, stages=_replace_stage(state.stages, index, failed))


class StageHandler(Protocol):
    """Produce durable artifact files for one claimed workflow stage."""

    def __call__(
        self,
        state: DailyWorkflowState,
        stage: WorkflowStage,
        /,
    ) -> Sequence[Path]: ...


class Clock(Protocol):
    """Timezone-aware clock used by the deterministic runner."""

    def __call__(self) -> datetime: ...


class DailyWorkflowRunner:
    """Continuously advance a trade date until complete, leased, or failed."""

    def __init__(
        self,
        *,
        store: DailyWorkflowStore,
        handlers: Mapping[WorkflowStage, StageHandler],
        worker_id: str,
        clock: Clock,
        trigger: WorkflowTrigger = WorkflowTrigger.MANUAL,
        lease_duration: timedelta = timedelta(minutes=15),
    ) -> None:
        self._store = store
        self._handlers = handlers
        self._worker_id = worker_id
        self._clock = clock
        self._trigger = trigger
        self._lease_duration = lease_duration
        self._controller = DailyWorkflowController()

    def run_until_idle(self, *, trade_date: date) -> DailyWorkflowState:
        """Loop over required stages and persist every transition before continuing."""
        state = self._store.load_latest(trade_date)
        if state is None:
            state = self._store.write(
                DailyWorkflowState.initialize(
                    trade_date=trade_date,
                    now=self._clock(),
                    trigger=self._trigger,
                )
            )
        elif state.trigger is not self._trigger:
            raise ValueError(
                f"existing workflow trigger {state.trigger.value} does not match "
                f"runner trigger {self._trigger.value}"
            )
        while not state.complete:
            pending_stage = next(
                item.stage for item in state.stages if item.status is not StageStatus.SUCCEEDED
            )
            pending_handler = self._handlers.get(pending_stage)
            claim_time = self._transition_time(state)
            readiness = (
                getattr(pending_handler, "is_ready", None) if pending_handler is not None else None
            )
            if callable(readiness) and not readiness(claim_time):
                return state
            claimed = self._controller.claim_next(
                state,
                worker_id=self._worker_id,
                now=claim_time,
                lease_duration=self._lease_duration,
            )
            if claimed is None:
                return state
            state = self._store.write(claimed)
            stage = next(item.stage for item in state.stages if item.status is StageStatus.RUNNING)
            handler = self._handlers.get(stage)
            if handler is None:
                error = RuntimeError(f"no workflow handler registered for {stage.value}")
                state = self._store.write(
                    self._controller.fail(
                        state,
                        worker_id=self._worker_id,
                        stage=stage,
                        error=error,
                        now=self._transition_time(state),
                    )
                )
                return state
            try:
                paths = handler(state, stage)
                artifacts = tuple(ArtifactReference.capture(path) for path in paths)
                state = self._store.write(
                    self._controller.succeed(
                        state,
                        worker_id=self._worker_id,
                        stage=stage,
                        artifacts=artifacts,
                        now=self._transition_time(state),
                    )
                )
            except Exception as error:
                state = self._store.write(
                    self._controller.fail(
                        state,
                        worker_id=self._worker_id,
                        stage=stage,
                        error=error,
                        now=self._transition_time(state),
                    )
                )
                return state
        return state

    def _transition_time(self, state: DailyWorkflowState) -> datetime:
        """Clamp wall-clock regressions to the append-only logical timeline."""
        return max(self._clock(), state.updated_at)


class DailyWorkflowStore:
    """Append-only revision store with a verified hash chain."""

    def __init__(self, root: Path) -> None:
        self._root = root.resolve()

    def write(self, state: DailyWorkflowState) -> DailyWorkflowState:
        partition = self._partition(state.trade_date)
        partition.mkdir(parents=True, exist_ok=True)
        states = self._validate_chain(state.trade_date)
        if state.revision < len(states):
            if states[state.revision].sha256 == state.sha256:
                return state
            raise RuntimeError(
                f"workflow revision {state.revision} already exists with different content"
            )
        if state.revision > len(states):
            raise RuntimeError("workflow revisions must be written without gaps")
        expected_previous = states[-1].sha256 if states else None
        if state.previous_sha256 != expected_previous:
            raise RuntimeError("workflow revision does not extend the current hash chain")
        existing = tuple(partition.glob(f"revision-{state.revision:06d}-*.json"))
        path = partition / f"revision-{state.revision:06d}-{state.sha256[:16]}.json"
        if existing and path not in existing:
            raise RuntimeError(
                f"concurrent workflow revision collision for {state.trade_date} "
                f"revision {state.revision}"
            )
        try:
            with path.open("xb") as destination:
                destination.write(state.canonical_bytes)
        except FileExistsError:
            if path.read_bytes() != state.canonical_bytes:
                raise RuntimeError(f"workflow state collision at {path}") from None
        self._validate_chain(state.trade_date)
        return state

    def load_latest(self, trade_date: date) -> DailyWorkflowState | None:
        states = self._validate_chain(trade_date)
        return states[-1] if states else None

    def _validate_chain(self, trade_date: date) -> tuple[DailyWorkflowState, ...]:
        paths = sorted(self._partition(trade_date).glob("revision-*.json"))
        states = tuple(DailyWorkflowState.load(path) for path in paths)
        for index, (path, state) in enumerate(zip(paths, states, strict=True)):
            if state.trade_date != trade_date or state.revision != index:
                raise ValueError(f"workflow revision sequence is invalid at {path}")
            expected_name = f"revision-{index:06d}-{state.sha256[:16]}.json"
            if path.name != expected_name:
                raise ValueError(f"workflow revision filename hash is invalid at {path}")
            expected_previous = states[index - 1].sha256 if index else None
            if state.previous_sha256 != expected_previous:
                raise ValueError(f"workflow hash chain is invalid at {path}")
        return states

    def _partition(self, trade_date: date) -> Path:
        return (
            self._root
            / "manifests"
            / "job=daily-paper-workflow"
            / f"trade_date={trade_date.isoformat()}"
        )


class _StrictSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ArtifactReferenceSpec(_StrictSpec):
    path: str
    sha256: str
    size_bytes: int = Field(ge=0)

    def to_domain(self) -> ArtifactReference:
        return ArtifactReference(**self.model_dump())


class StageRecordSpec(_StrictSpec):
    stage: WorkflowStage
    status: StageStatus
    attempts: int = Field(ge=0)
    worker_id: str | None
    started_at: datetime | None
    lease_expires_at: datetime | None
    completed_at: datetime | None
    input_sha256: tuple[str, ...]
    output_artifacts: tuple[ArtifactReferenceSpec, ...]
    error_type: str | None
    error_message: str | None

    def to_domain(self) -> StageRecord:
        values = self.model_dump(exclude={"output_artifacts"})
        return StageRecord(
            **values,
            output_artifacts=tuple(item.to_domain() for item in self.output_artifacts),
        )


class DailyWorkflowStateSpec(_StrictSpec):
    schema_version: Literal[2]
    trade_date: date
    trigger: WorkflowTrigger
    revision: int = Field(ge=0)
    previous_sha256: str | None
    created_at: datetime
    updated_at: datetime
    stages: tuple[StageRecordSpec, ...]

    def to_domain(self) -> DailyWorkflowState:
        values = self.model_dump(exclude={"stages"})
        return DailyWorkflowState(
            **values,
            stages=tuple(item.to_domain() for item in self.stages),
        )


def _owned_running_stage(
    state: DailyWorkflowState,
    *,
    worker_id: str,
    stage: WorkflowStage,
) -> tuple[int, StageRecord]:
    index = STAGE_ORDER.index(stage)
    current = state.stages[index]
    if current.status is not StageStatus.RUNNING:
        raise ValueError(f"workflow stage {stage.value} is not running")
    if current.worker_id != worker_id.strip():
        raise ValueError(f"workflow stage {stage.value} is owned by another worker")
    return index, current


def _validate_stage_timestamps(record: StageRecord) -> None:
    for name, value in (
        ("started_at", record.started_at),
        ("lease_expires_at", record.lease_expires_at),
        ("completed_at", record.completed_at),
    ):
        if value is not None:
            _require_aware(name, value)
    if (
        record.started_at is not None
        and record.completed_at is not None
        and record.completed_at < record.started_at
    ):
        raise ValueError("stage completion cannot precede its start")


def _validate_stage_status(record: StageRecord) -> None:
    if record.status is StageStatus.PENDING:
        execution_values = (
            record.worker_id,
            record.started_at,
            record.lease_expires_at,
            record.completed_at,
            record.error_type,
            record.error_message,
        )
        if any(value is not None for value in execution_values) or record.attempts:
            raise ValueError("pending stage cannot contain execution state")
        return
    if record.status is StageStatus.RUNNING:
        if (
            not record.worker_id
            or record.started_at is None
            or record.lease_expires_at is None
            or record.completed_at is not None
            or record.output_artifacts
            or record.error_type
            or record.error_message
        ):
            raise ValueError("running stage requires a worker and active lease")
        if record.lease_expires_at <= record.started_at:
            raise ValueError("stage lease must end after its start")
        return
    if record.status is StageStatus.SUCCEEDED:
        if (
            record.attempts < 1
            or not record.worker_id
            or record.started_at is None
            or record.completed_at is None
            or not record.output_artifacts
        ):
            raise ValueError("successful stage requires completion time and output artifacts")
        if record.lease_expires_at is not None or record.error_type or record.error_message:
            raise ValueError("successful stage cannot retain lease or error state")
        return
    if (
        record.attempts < 1
        or not record.worker_id
        or record.started_at is None
        or record.completed_at is None
        or not record.error_type
        or not record.error_message
    ):
        raise ValueError("failed stage requires completion time and error details")
    if record.lease_expires_at is not None or record.output_artifacts:
        raise ValueError("failed stage cannot retain a lease or output artifacts")


def _replace_stage(
    stages: tuple[StageRecord, ...],
    index: int,
    record: StageRecord,
) -> tuple[StageRecord, ...]:
    return (*stages[:index], record, *stages[index + 1 :])


def _advance(
    state: DailyWorkflowState,
    *,
    now: datetime,
    stages: tuple[StageRecord, ...],
) -> DailyWorkflowState:
    _require_aware("now", now)
    if now < state.updated_at:
        raise ValueError("workflow transition time cannot move backward")
    return DailyWorkflowState(
        schema_version=state.schema_version,
        trade_date=state.trade_date,
        trigger=state.trigger,
        revision=state.revision + 1,
        previous_sha256=state.sha256,
        created_at=state.created_at,
        updated_at=now,
        stages=stages,
    )


def _require_aware(name: str, value: datetime) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")


def _validate_digest(name: str, value: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{name} must be lowercase hexadecimal")
