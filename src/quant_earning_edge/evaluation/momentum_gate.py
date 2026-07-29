"""Source-bound Phase 3 momentum benchmark reproduction gate."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path  # noqa: TC003 - Pydantic resolves runtime fields.
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, model_validator

from quant_earning_edge.backtest import BacktestSpec, VectorbtBacktestEngine
from quant_earning_edge.evaluation.momentum_builder import (
    MomentumBaselineBuilder,
    MomentumBaselineBuildSpec,
)
from quant_earning_edge.evaluation.report import PerformanceEvaluator, PerformanceReport

if TYPE_CHECKING:
    from collections.abc import Sequence

    from quant_earning_edge.data.calendar import SessionFile

_SHA256_LENGTH = 64
_MOMENTUM_STRATEGY = "cross_sectional_momentum_60_session"
_MOMENTUM_UNIVERSE = "historical_spy_components"


class _StrictSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")


class MomentumBenchmarkReferenceSpec(_StrictSpec):
    """Pinned published result and the exact source artifact carrying it."""

    schema_version: int = 1
    benchmark_name: str = Field(min_length=1)
    source_url: HttpUrl
    source_artifact_sha256: str
    universe_source_url: HttpUrl
    baseline_build_spec_sha256: str
    published_net_sharpe: float
    tolerance: float = Field(default=0.1, gt=0, le=0.1)
    minimum_session_count: int = Field(default=252, ge=252)

    @model_validator(mode="after")
    def validate_contract(self) -> MomentumBenchmarkReferenceSpec:
        if self.schema_version != 1 or not all(
            _is_sha256(item)
            for item in (
                self.source_artifact_sha256,
                self.baseline_build_spec_sha256,
            )
        ):
            raise ValueError("momentum benchmark reference identity is invalid")
        if not math.isfinite(self.published_net_sharpe):
            raise ValueError("published momentum Sharpe must be finite")
        return self

    @property
    def canonical_bytes(self) -> bytes:
        return json.dumps(
            self.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
        ).encode()

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes).hexdigest()

    def write(self, output: Path) -> None:
        output.parent.mkdir(parents=True, exist_ok=True)
        try:
            with output.open("xb") as destination:
                destination.write(self.canonical_bytes)
        except FileExistsError:
            if output.read_bytes() != self.canonical_bytes:
                raise RuntimeError(f"momentum benchmark reference collision at {output}") from None

    @classmethod
    def load(cls, path: Path) -> MomentumBenchmarkReferenceSpec:
        encoded = path.read_bytes()
        spec = cls.model_validate_json(encoded)
        if spec.canonical_bytes != encoded:
            raise ValueError("momentum benchmark reference is not canonical")
        return spec


class MomentumBaselineManifest(_StrictSpec):
    """Actual baseline identity linked to the standardized backtest input."""

    schema_version: int = 2
    strategy: str
    lookback_sessions: int
    selection_fraction: float = Field(gt=0, le=0.5)
    universe: str
    universe_artifact_sha256: str
    trade_plan_sha256: str
    backtest_input_sha256: str
    build_spec_sha256: str
    session_file_sha256: str
    daily_bar_sha256: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_contract(self) -> MomentumBaselineManifest:
        if self.schema_version != 2:
            raise ValueError("momentum baseline schema_version must be 2")
        if self.strategy != _MOMENTUM_STRATEGY or self.lookback_sessions != 60:
            raise ValueError("momentum baseline strategy contract differs")
        if self.universe != _MOMENTUM_UNIVERSE:
            raise ValueError("momentum baseline must use historical SPY components")
        for digest in (
            self.universe_artifact_sha256,
            self.trade_plan_sha256,
            self.backtest_input_sha256,
            self.build_spec_sha256,
            self.session_file_sha256,
            *self.daily_bar_sha256,
        ):
            if not _is_sha256(digest):
                raise ValueError("momentum baseline artifact identity is invalid")
        return self

    @property
    def canonical_bytes(self) -> bytes:
        return json.dumps(
            self.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
        ).encode()

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes).hexdigest()

    def write(self, output: Path) -> None:
        output.parent.mkdir(parents=True, exist_ok=True)
        try:
            with output.open("xb") as destination:
                destination.write(self.canonical_bytes)
        except FileExistsError:
            if output.read_bytes() != self.canonical_bytes:
                raise RuntimeError(f"momentum baseline manifest collision at {output}") from None

    @classmethod
    def load(cls, path: Path) -> MomentumBaselineManifest:
        encoded = path.read_bytes()
        manifest = cls.model_validate_json(encoded)
        if manifest.canonical_bytes != encoded:
            raise ValueError("momentum baseline manifest is not canonical")
        return manifest


@dataclass(frozen=True)
class MomentumBenchmarkGateReport:
    """Immutable verdict for the Phase 3 published-comparison exit gate."""

    schema_version: int
    reference_spec_sha256: str
    reference_artifact_sha256: str
    baseline_manifest_sha256: str
    performance_report_sha256: str
    published_net_sharpe: float
    actual_net_sharpe: float
    absolute_difference: float
    tolerance: float
    session_count: int
    passes: bool

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("momentum benchmark gate schema_version must be 1")
        for digest in (
            self.reference_spec_sha256,
            self.reference_artifact_sha256,
            self.baseline_manifest_sha256,
            self.performance_report_sha256,
        ):
            if not _is_sha256(digest):
                raise ValueError("momentum benchmark gate artifact identity is invalid")
        expected = self.absolute_difference <= self.tolerance
        if self.passes != expected:
            raise ValueError("momentum benchmark gate verdict is inconsistent")

    @property
    def canonical_bytes(self) -> bytes:
        return json.dumps(
            asdict(self),
            sort_keys=True,
            separators=(",", ":"),
        ).encode()

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes).hexdigest()

    def write(self, output: Path) -> None:
        output.parent.mkdir(parents=True, exist_ok=True)
        try:
            with output.open("xb") as destination:
                destination.write(self.canonical_bytes)
        except FileExistsError:
            if output.read_bytes() != self.canonical_bytes:
                raise RuntimeError(f"momentum benchmark gate collision at {output}") from None


class MomentumBenchmarkGateEvaluator:
    """Verify source, strategy, and performance identities before comparison."""

    def evaluate(  # noqa: PLR0912 - strict source/reproduction gate.
        self,
        *,
        reference_spec: MomentumBenchmarkReferenceSpec,
        reference_artifact: Path,
        universe_artifact: Path,
        trade_plan: Path,
        build_spec: MomentumBaselineBuildSpec,
        calendar: SessionFile,
        daily_bar_files: Sequence[Path],
        baseline_manifest: MomentumBaselineManifest,
        performance_report: PerformanceReport,
    ) -> MomentumBenchmarkGateReport:
        reference_hash = _file_sha256(reference_artifact)
        if reference_hash != reference_spec.source_artifact_sha256:
            raise ValueError("published momentum reference artifact hash differs")
        if build_spec.sha256 != reference_spec.baseline_build_spec_sha256:
            raise ValueError("published momentum reference build methodology differs")
        if _file_sha256(universe_artifact) != baseline_manifest.universe_artifact_sha256:
            raise ValueError("momentum historical-universe artifact hash differs")
        if _file_sha256(trade_plan) != baseline_manifest.trade_plan_sha256:
            raise ValueError("momentum trade-plan artifact hash differs")
        if build_spec.sha256 != baseline_manifest.build_spec_sha256:
            raise ValueError("momentum build-spec artifact hash differs")
        if calendar.sha256 != baseline_manifest.session_file_sha256:
            raise ValueError("momentum session-file artifact hash differs")
        bar_hashes = tuple(sorted(_file_sha256(path) for path in daily_bar_files))
        if bar_hashes != baseline_manifest.daily_bar_sha256:
            raise ValueError("momentum daily-bar artifact hashes differ")
        rebuilt = MomentumBaselineBuilder().build(
            spec=build_spec,
            calendar=calendar,
            universe_artifact=universe_artifact,
            daily_bar_files=daily_bar_files,
        )
        if rebuilt.trade_plan_bytes != trade_plan.read_bytes():
            raise ValueError("momentum trade plan does not reproduce from source artifacts")
        expected_manifest = MomentumBaselineManifest(
            schema_version=2,
            strategy=_MOMENTUM_STRATEGY,
            lookback_sessions=build_spec.lookback_sessions,
            selection_fraction=build_spec.selection_fraction,
            universe=_MOMENTUM_UNIVERSE,
            universe_artifact_sha256=rebuilt.universe_artifact_sha256,
            trade_plan_sha256=rebuilt.trade_plan_sha256,
            backtest_input_sha256=rebuilt.trade_plan.input_sha256,
            build_spec_sha256=rebuilt.build_spec_sha256,
            session_file_sha256=rebuilt.session_file_sha256,
            daily_bar_sha256=rebuilt.daily_bar_sha256,
        )
        if expected_manifest != baseline_manifest:
            raise ValueError("momentum baseline manifest does not reproduce from source artifacts")
        reproduced = _reproduce_performance(
            trade_plan=trade_plan,
            claimed=performance_report,
        )
        if reproduced.input_sha256 != baseline_manifest.backtest_input_sha256:
            raise ValueError("momentum baseline manifest and performance input differ")
        if reproduced.engine.startswith("vectorbt-intraday-"):
            raise ValueError("momentum benchmark requires the daily vectorbt engine")
        if not reproduced.engine.startswith("vectorbt-"):
            raise ValueError("momentum benchmark requires the standardized vectorbt engine")
        if reproduced.session_count < reference_spec.minimum_session_count:
            raise ValueError("momentum benchmark has insufficient evaluated sessions")
        if reproduced.trade_count < 1:
            raise ValueError("momentum benchmark performance contains no trades")
        if not math.isfinite(reproduced.net_sharpe):
            raise ValueError("momentum benchmark Sharpe is not finite")
        difference = abs(reproduced.net_sharpe - reference_spec.published_net_sharpe)
        return MomentumBenchmarkGateReport(
            schema_version=1,
            reference_spec_sha256=reference_spec.sha256,
            reference_artifact_sha256=reference_hash,
            baseline_manifest_sha256=baseline_manifest.sha256,
            performance_report_sha256=performance_report.sha256,
            published_net_sharpe=reference_spec.published_net_sharpe,
            actual_net_sharpe=reproduced.net_sharpe,
            absolute_difference=difference,
            tolerance=reference_spec.tolerance,
            session_count=reproduced.session_count,
            passes=difference <= reference_spec.tolerance,
        )


def _reproduce_performance(
    *,
    trade_plan: Path,
    claimed: PerformanceReport,
) -> PerformanceReport:
    spec = BacktestSpec.model_validate_json(trade_plan.read_bytes())
    initial_cash, sessions, marks, trades = spec.domain_inputs()
    if any(item.entry_at is not None or item.exit_at is not None for item in trades):
        raise ValueError("momentum trade plan must use the daily backtest contract")
    result = VectorbtBacktestEngine().run(
        trades=trades,
        marks=marks,
        sessions=sessions,
        initial_cash=initial_cash,
    )
    bootstrap = claimed.bootstrap
    evaluator = PerformanceEvaluator(
        bootstrap_resamples=bootstrap.resamples if bootstrap is not None else 1,
        seed=bootstrap.seed if bootstrap is not None else 0,
    )
    reproduced = evaluator.evaluate(result)
    if reproduced != claimed:
        raise ValueError("momentum performance report does not reproduce from trade plan")
    return reproduced


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: str) -> bool:
    return len(value) == _SHA256_LENGTH and all(item in "0123456789abcdef" for item in value)
