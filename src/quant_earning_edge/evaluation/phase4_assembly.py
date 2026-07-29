"""Deterministic source-to-plan assembly for the complete Phase 4 OOS run."""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

import pyarrow.parquet as pq
from pydantic import BaseModel, ConfigDict, Field

from quant_earning_edge.backtest import CostModel
from quant_earning_edge.data import (
    DAILY_BARS_SCHEMA,
    SessionFileStore,
    SplitHistorySourceCapture,
    SplitHistorySourceManifest,
)
from quant_earning_edge.evaluation.strategy_gate import (
    FoldArtifactSpec,
    Phase4AggregationSpec,
)
from quant_earning_edge.portfolio import (
    FractionalKellyPortfolioConstructor,
    PortfolioConfig,
    TradeOutcome,
)
from quant_earning_edge.signals.config import load_strategy_config
from quant_earning_edge.signals.event_trades import (
    EventExecutionObservation,
    EventTradePlanner,
    run_event_plan,
)
from quant_earning_edge.signals.lgbm_model import OosPrediction, WalkForwardModelRun
from quant_earning_edge.universe import EVENT_CANDIDATE_SCHEMA

if TYPE_CHECKING:
    from collections.abc import Sequence
    from datetime import datetime

    from quant_earning_edge.data.clients import MarketSession


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Phase4FileReference(_StrictModel):
    """Relative path and exact content identity for one assembly file."""

    path: Path
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class Phase4AssemblyManifest(_StrictModel):
    """Canonical binding from raw research inputs to generated event plans."""

    schema_version: Literal[2] = 2
    strategy_config: Phase4FileReference
    walkforward_run_evidence: Phase4FileReference
    session_file: Phase4FileReference
    candidate_files: tuple[Phase4FileReference, ...]
    daily_bar_files: tuple[Phase4FileReference, ...]
    split_source_manifest: Phase4FileReference
    initial_cash: float = Field(gt=0)
    minimum_probability: float = Field(ge=0.5, lt=1)
    execution_price_contract: Literal["raw_session_open_to_close_no_trade_date_split"]
    iv_regime_contract: Literal["unavailable"]
    plan_files: tuple[Phase4FileReference, ...]

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

    @classmethod
    def load(cls, path: Path, *, verify_files: bool = True) -> Phase4AssemblyManifest:
        """Reload canonical assembly evidence and optionally re-hash every file."""
        encoded = path.read_bytes()
        try:
            manifest = cls.model_validate_json(encoded)
        except (TypeError, ValueError) as error:
            raise ValueError(f"invalid Phase 4 assembly manifest: {path}") from error
        if manifest.canonical_bytes != encoded:
            raise ValueError("Phase 4 assembly manifest is not canonical")
        if verify_files:
            manifest.verify_files(path)
        return manifest

    def verify_files(self, manifest_path: Path) -> None:
        """Verify unique references and every bound file digest."""
        references = (
            self.strategy_config,
            self.walkforward_run_evidence,
            self.session_file,
            *self.candidate_files,
            *self.daily_bar_files,
            self.split_source_manifest,
            *self.plan_files,
        )
        resolved = tuple(
            _resolve(reference.path, base=manifest_path.parent) for reference in references
        )
        if len(set(resolved)) != len(resolved):
            raise ValueError("Phase 4 assembly manifest contains duplicate file references")
        for reference, path in zip(references, resolved, strict=True):
            if not path.is_file():
                raise ValueError(f"Phase 4 assembly source is missing: {path}")
            if _file_sha256(path) != reference.sha256:
                raise ValueError(f"Phase 4 assembly source hash differs: {path}")

    def resolved_plan_files(self, manifest_path: Path) -> tuple[Path, ...]:
        return tuple(_resolve(item.path, base=manifest_path.parent) for item in self.plan_files)

    def resolved_strategy_config(self, manifest_path: Path) -> Path:
        return _resolve(self.strategy_config.path, base=manifest_path.parent)

    def resolved_walkforward_run(self, manifest_path: Path) -> Path:
        return _resolve(self.walkforward_run_evidence.path, base=manifest_path.parent)

    def resolved_session_file(self, manifest_path: Path) -> Path:
        return _resolve(self.session_file.path, base=manifest_path.parent)

    def resolved_candidate_files(self, manifest_path: Path) -> tuple[Path, ...]:
        return tuple(
            _resolve(item.path, base=manifest_path.parent) for item in self.candidate_files
        )

    def resolved_daily_bar_files(self, manifest_path: Path) -> tuple[Path, ...]:
        return tuple(
            _resolve(item.path, base=manifest_path.parent) for item in self.daily_bar_files
        )

    def resolved_split_source_manifest(self, manifest_path: Path) -> Path:
        return _resolve(self.split_source_manifest.path, base=manifest_path.parent)


@dataclass(frozen=True)
class Phase4AssemblyResult:
    """Paths and identities emitted by one historical assembly."""

    manifest_path: Path
    manifest_sha256: str
    aggregation_path: Path
    plan_files: tuple[Path, ...]
    session_count: int
    trade_count: int
    final_equity: float


class Phase4HistoricalAssembler:
    """Build every OOS event plan directly from pinned canonical sources."""

    def assemble(  # noqa: PLR0915 - complete evidence boundary.
        self,
        *,
        walkforward_run_evidence: Path,
        strategy_config: Path,
        session_file: Path,
        candidate_files: Sequence[Path],
        daily_bar_files: Sequence[Path],
        split_source_manifest: Path,
        initial_cash: float,
        output_dir: Path,
        manifest_output: Path,
        aggregation_output: Path,
        minimum_probability: float = 0.5,
    ) -> Phase4AssemblyResult:
        """Generate chronologically chained plans and the Phase 4 fold map."""
        if initial_cash <= 0 or not math.isfinite(initial_cash):
            raise ValueError("Phase 4 initial cash must be finite and positive")
        if not 0.5 <= minimum_probability < 1:
            raise ValueError("Phase 4 minimum probability must be in [0.5, 1)")
        candidate_paths = _unique_files(candidate_files, kind="candidate")
        daily_paths = _unique_files(daily_bar_files, kind="daily bar")
        split_source = SplitHistorySourceManifest.load(split_source_manifest)
        SplitHistorySourceCapture.reproduce(
            split_source,
            data_lake_root=split_source.data_lake_root,
        )
        strategy = load_strategy_config(strategy_config)
        run = WalkForwardModelRun.load_evidence(walkforward_run_evidence)
        if run.hyperparameter_study_sha256 is None:
            raise ValueError("walk-forward run lacks an Optuna study binding")
        if (
            run.feature_names != strategy.features
            or run.label_name != strategy.label.column_name
            or run.threshold != strategy.label.threshold
            or run.seed != strategy.seed
        ):
            raise ValueError("walk-forward run differs from the strategy configuration")

        calendar = SessionFileStore.load(session_file)
        sessions = calendar.sessions
        session_index = {item.session_date: index for index, item in enumerate(sessions)}
        candidates = _load_candidates(candidate_paths, session_sha256=calendar.sha256)
        bars = _load_daily_bars(daily_paths)
        bar_dates = tuple(key[1] for key in bars)
        if date.fromisoformat(split_source.raw["start_date"]) > min(
            bar_dates
        ) or date.fromisoformat(split_source.raw["end_date"]) < max(bar_dates):
            raise ValueError("Phase 4 split history does not cover the execution bars")
        predictions = tuple(item for fold in run.folds for item in fold.predictions)
        prediction_keys = {(item.symbol, item.asof_date) for item in predictions}
        if len(prediction_keys) != len(predictions):
            raise ValueError("walk-forward predictions contain duplicate symbol/as-of keys")
        if set(candidates) != prediction_keys:
            missing = prediction_keys - set(candidates)
            extra = set(candidates) - prediction_keys
            raise ValueError(
                "candidate keys differ from the complete OOS prediction ledger "
                f"(missing={len(missing)}, extra={len(extra)})"
            )

        planner = EventTradePlanner(
            FractionalKellyPortfolioConstructor(
                PortfolioConfig(
                    top_k=strategy.portfolio.top_k,
                    kelly_fraction=strategy.portfolio.sizing.kelly_fraction,
                    history_window=strategy.portfolio.sizing.rolling_window_days,
                    minimum_history=min(20, strategy.portfolio.sizing.rolling_window_days),
                    calibration_position_weight=(
                        strategy.portfolio.sizing.calibration_position_pct
                    ),
                    max_position_weight=strategy.portfolio.caps.max_position_pct,
                    max_sector_weight=strategy.portfolio.caps.max_sector_pct,
                    max_gross_weight=strategy.portfolio.caps.max_gross_exposure_pct,
                )
            ),
            minimum_probability=minimum_probability,
        )
        cost_model = CostModel(strategy.cost_model_config)
        output_root = output_dir.resolve()
        output_root.mkdir(parents=True, exist_ok=True)
        equity = float(initial_cash)
        outcomes: list[TradeOutcome] = []
        plan_paths: list[Path] = []
        fold_specs: list[FoldArtifactSpec] = []
        used_trade_dates: set[date] = set()
        trade_count = 0

        for fold in run.folds:
            predictions_by_trade_date: dict[date, list[OosPrediction]] = {}
            for prediction in fold.predictions:
                candidate = candidates[(prediction.symbol, prediction.asof_date)]
                trade_date = candidate["trade_date"]
                predictions_by_trade_date.setdefault(trade_date, []).append(prediction)
            fold_plan_paths: list[Path] = []
            for trade_date in sorted(predictions_by_trade_date):
                if trade_date in used_trade_dates:
                    raise ValueError("walk-forward folds overlap on an event trade date")
                used_trade_dates.add(trade_date)
                session, decision_at = _validated_trade_session(
                    trade_date=trade_date,
                    prediction_rows=predictions_by_trade_date[trade_date],
                    candidates=candidates,
                    session_index=session_index,
                    sessions=sessions,
                )
                observations = tuple(
                    _observation(
                        prediction=item,
                        candidate=candidates[(item.symbol, item.asof_date)],
                        decision_at=decision_at,
                        prior_close_at=sessions[session_index[trade_date] - 1].close_at,
                        entry_at=session.open_at,
                        exit_at=session.close_at,
                        bar=bars.get((item.symbol, trade_date)),
                    )
                    for item in sorted(
                        predictions_by_trade_date[trade_date],
                        key=lambda row: row.row_index,
                    )
                )
                selected_predictions = tuple(
                    sorted(
                        predictions_by_trade_date[trade_date],
                        key=lambda row: row.row_index,
                    )
                )
                plan = planner.plan(
                    predictions=selected_predictions,
                    observations=observations,
                    outcomes=tuple(outcomes),
                    equity=equity,
                    walkforward_run_sha256=run.sha256,
                )
                plan_path = (
                    output_root
                    / f"fold-{fold.fold_index:03d}"
                    / f"event-plan-{trade_date.isoformat()}.json"
                )
                EventTradePlanner.write(plan, plan_path)
                result = run_event_plan(plan, cost_model=cost_model)
                outcomes.extend(
                    TradeOutcome(closed_date=trade_date, net_return=trade.net_return)
                    for trade in result.trades
                )
                equity = result.final_net_equity
                trade_count += len(plan.intents)
                plan_paths.append(plan_path)
                fold_plan_paths.append(plan_path)
            if not fold_plan_paths:
                raise ValueError(f"walk-forward fold {fold.fold_index} produced no event plans")
            fold_dates = tuple(EventTradePlanner.load(path).trade_date for path in fold_plan_paths)
            fold_specs.append(
                FoldArtifactSpec(
                    fold_index=fold.fold_index,
                    test_start_date=min(fold_dates),
                    test_end_date=max(fold_dates),
                    event_plan_files=tuple(
                        _relative(path, base=aggregation_output.parent) for path in fold_plan_paths
                    ),
                )
            )

        manifest = Phase4AssemblyManifest(
            strategy_config=_reference(strategy_config, base=manifest_output.parent),
            walkforward_run_evidence=_reference(
                walkforward_run_evidence,
                base=manifest_output.parent,
            ),
            session_file=_reference(session_file, base=manifest_output.parent),
            candidate_files=tuple(
                _reference(path, base=manifest_output.parent) for path in candidate_paths
            ),
            daily_bar_files=tuple(
                _reference(path, base=manifest_output.parent) for path in daily_paths
            ),
            split_source_manifest=_reference(
                split_source.path,
                base=manifest_output.parent,
            ),
            initial_cash=initial_cash,
            minimum_probability=minimum_probability,
            execution_price_contract="raw_session_open_to_close_no_trade_date_split",
            iv_regime_contract="unavailable",
            plan_files=tuple(_reference(path, base=manifest_output.parent) for path in plan_paths),
        )
        _write_once(manifest_output, manifest.canonical_bytes, kind="assembly manifest")
        aggregation = Phase4AggregationSpec(
            assembly_manifest=_relative(manifest_output, base=aggregation_output.parent),
            walkforward_run_evidence=_relative(
                walkforward_run_evidence,
                base=aggregation_output.parent,
            ),
            folds=tuple(fold_specs),
        )
        aggregation_bytes = json.dumps(
            aggregation.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        _write_once(aggregation_output, aggregation_bytes, kind="aggregation")
        return Phase4AssemblyResult(
            manifest_path=manifest_output.resolve(),
            manifest_sha256=manifest.sha256,
            aggregation_path=aggregation_output.resolve(),
            plan_files=tuple(path.resolve() for path in plan_paths),
            session_count=len(plan_paths),
            trade_count=trade_count,
            final_equity=equity,
        )


def _load_candidates(
    paths: Sequence[Path],
    *,
    session_sha256: str,
) -> dict[tuple[str, date], dict[str, Any]]:
    output: dict[tuple[str, date], dict[str, Any]] = {}
    for path in paths:
        if pq.read_schema(path) != EVENT_CANDIDATE_SCHEMA:  # type: ignore[no-untyped-call]
            raise ValueError(f"event candidate schema mismatch: {path}")
        rows: list[dict[str, Any]] = pq.read_table(path).to_pylist()  # type: ignore[no-untyped-call]
        if not rows:
            raise ValueError(f"event candidate file is empty: {path}")
        file_keys = tuple((str(row["symbol"]).strip().upper(), row["asof_date"]) for row in rows)
        if file_keys != tuple(sorted(file_keys)):
            raise ValueError(f"event candidate rows are not sorted: {path}")
        for row, key in zip(rows, file_keys, strict=True):
            if key in output:
                raise ValueError(f"duplicate event candidate key: {key}")
            if row["session_file_sha256"] != session_sha256:
                raise ValueError("event candidate differs from the supplied session file")
            if row["timing"] not in {"bmo", "amc"}:
                raise ValueError("event candidate timing is unsupported")
            if row["split_event_ids"]:
                raise ValueError("Phase 4 excludes earnings trades on split execution dates")
            output[key] = row
    return output


def _load_daily_bars(paths: Sequence[Path]) -> dict[tuple[str, date], dict[str, Any]]:
    output: dict[tuple[str, date], dict[str, Any]] = {}
    for path in paths:
        if pq.read_schema(path) != DAILY_BARS_SCHEMA:  # type: ignore[no-untyped-call]
            raise ValueError(f"daily bar schema mismatch: {path}")
        rows: list[dict[str, Any]] = pq.read_table(path).to_pylist()  # type: ignore[no-untyped-call]
        for row in rows:
            symbol = str(row["symbol"]).strip().upper()
            key = (symbol, row["session_date"])
            if key in output:
                raise ValueError(f"duplicate daily bar key: {key}")
            if row["adjusted"]:
                raise ValueError("Phase 4 execution requires raw daily bars")
            for field in ("open", "close"):
                value = float(row[field])
                if value <= 0 or not math.isfinite(value):
                    raise ValueError(f"daily bar {field} must be finite and positive")
            output[key] = row
    return output


def _validated_trade_session(
    *,
    trade_date: date,
    prediction_rows: Sequence[OosPrediction],
    candidates: dict[tuple[str, date], dict[str, Any]],
    session_index: dict[date, int],
    sessions: Sequence[MarketSession],
) -> tuple[MarketSession, datetime]:
    try:
        index = session_index[trade_date]
    except KeyError as error:
        raise ValueError(
            f"event trade date is absent from the session file: {trade_date}"
        ) from error
    if index == 0:
        raise ValueError("session file lacks the prior session for an event trade")
    session = sessions[index]
    prior = sessions[index - 1]
    information_cutoffs = []
    for prediction in prediction_rows:
        candidate = candidates[(prediction.symbol, prediction.asof_date)]
        if candidate["trade_date"] != trade_date or candidate["asof_date"] != prior.session_date:
            raise ValueError("event candidate does not use the immediately prior session")
        candidate_frozen_at = candidate["decision_at"]
        if not prior.close_at <= candidate_frozen_at < session.open_at:
            raise ValueError("event candidate decision timestamp is outside the causal window")
        if not candidate_frozen_at <= prediction.information_cutoff_at < session.open_at:
            raise ValueError(
                "OOS prediction information cutoff is outside the causal pre-open window"
            )
        information_cutoffs.append(prediction.information_cutoff_at)
    return session, max(information_cutoffs)


def _observation(
    *,
    prediction: OosPrediction,
    candidate: dict[str, Any],
    decision_at: datetime,
    prior_close_at: datetime,
    entry_at: datetime,
    exit_at: datetime,
    bar: dict[str, Any] | None,
) -> EventExecutionObservation:
    if bar is None:
        raise ValueError(
            f"missing raw execution bar for {prediction.symbol} on {candidate['trade_date']}"
        )
    return EventExecutionObservation(
        row_index=prediction.row_index,
        symbol=prediction.symbol,
        sector=str(candidate["sector"]),
        asof_date=prediction.asof_date,
        trade_date=candidate["trade_date"],
        decision_at=decision_at,
        sizing_price_observed_at=prior_close_at,
        sizing_price=float(candidate["sizing_price"]),
        entry_at=entry_at,
        entry_price=float(bar["open"]),
        exit_at=exit_at,
        exit_price=float(bar["close"]),
        frozen_average_daily_volume_shares=float(candidate["frozen_average_daily_volume_shares"]),
        event_timing=cast("Literal['bmo', 'amc']", candidate["timing"]),
        iv_regime="unavailable",
    )


def _unique_files(paths: Sequence[Path], *, kind: str) -> tuple[Path, ...]:
    resolved = tuple(sorted((path.resolve() for path in paths), key=str))
    if not resolved:
        raise ValueError(f"at least one {kind} file is required")
    if len(set(resolved)) != len(resolved):
        raise ValueError(f"{kind} files contain duplicates")
    if any(not path.is_file() for path in resolved):
        raise ValueError(f"{kind} file is missing")
    return resolved


def _reference(path: Path, *, base: Path) -> Phase4FileReference:
    resolved = path.resolve()
    return Phase4FileReference(
        path=_relative(resolved, base=base),
        sha256=_file_sha256(resolved),
    )


def _relative(path: Path, *, base: Path) -> Path:
    return Path(os.path.relpath(path.resolve(), start=base.resolve()))


def _resolve(path: Path, *, base: Path) -> Path:
    return (path if path.is_absolute() else base / path).resolve()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_once(path: Path, encoded: bytes, *, kind: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as destination:
            destination.write(encoded)
    except FileExistsError:
        if path.read_bytes() != encoded:
            raise RuntimeError(f"Phase 4 {kind} collision at {path}") from None
