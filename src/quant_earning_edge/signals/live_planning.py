"""Causal assembly of live planning inputs from frozen features and a booster."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import date, datetime  # noqa: TC003 - Pydantic resolves runtime fields.
from typing import TYPE_CHECKING, Any

import pyarrow.parquet as pq
from pydantic import BaseModel, ConfigDict, Field, model_validator

from quant_earning_edge.backtest import (  # noqa: TC001 - Pydantic annotation.
    DecisionSnapshotSpec,
)
from quant_earning_edge.features import FEATURE_VALUE_SCHEMA
from quant_earning_edge.signals.event_trades import (  # noqa: TC001 - Pydantic annotation.
    TradeOutcomeSpec,
)
from quant_earning_edge.signals.live_orders import (
    DailyOrderPlanningSpec,
    LiveCandidateSpec,
)

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from quant_earning_edge.signals.production_model import ProductionModelArtifact


class _StrictSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")


class LiveMarketObservationSpec(_StrictSpec):
    """Non-model fields frozen for one candidate at the decision boundary."""

    symbol: str
    sector: str
    sizing_price: float = Field(gt=0)
    sizing_price_observed_at: datetime
    frozen_average_daily_volume_shares: float = Field(gt=0)
    decision_snapshot: DecisionSnapshotSpec

    @model_validator(mode="after")
    def normalize(self) -> LiveMarketObservationSpec:
        symbol = self.symbol.strip().upper()
        sector = self.sector.strip()
        if not symbol or not sector:
            raise ValueError("live observation symbol and sector must not be blank")
        if self.decision_snapshot.ticker.strip().upper() != symbol:
            raise ValueError("live observation and decision snapshot symbols differ")
        object.__setattr__(self, "symbol", symbol)
        object.__setattr__(self, "sector", sector)
        return self


class LivePlanningSourceSpec(_StrictSpec):
    """Human-independent schedule and market observations, excluding model scores."""

    trade_date: date
    feature_asof_date: date
    decision_at: datetime
    equity: float = Field(gt=0)
    observations: tuple[LiveMarketObservationSpec, ...] = ()
    outcomes: tuple[TradeOutcomeSpec, ...] = ()
    entry_submitted_at: datetime
    entry_expires_at: datetime
    exit_submitted_at: datetime
    exit_expires_at: datetime
    minimum_probability: float = Field(default=0.5, ge=0.5, lt=1)

    @model_validator(mode="after")
    def validate_source(self) -> LivePlanningSourceSpec:
        symbols = tuple(item.symbol for item in self.observations)
        if symbols != tuple(sorted(set(symbols))):
            raise ValueError("live observation symbols must be unique and sorted")
        if self.feature_asof_date >= self.trade_date:
            raise ValueError("feature_asof_date must precede trade_date")
        return self

    @property
    def canonical_bytes(self) -> bytes:
        return json.dumps(
            self.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
        ).encode()


@dataclass(frozen=True)
class ScoredPlanningArtifact:
    """Lineage wrapper around the generated legacy-compatible planning spec."""

    schema_version: int
    source_sha256: str
    model_artifact_sha256: str
    model_sha256: str
    feature_file_sha256: tuple[str, ...]
    planning: DailyOrderPlanningSpec

    def __post_init__(self) -> None:
        if self.schema_version != 2:
            raise ValueError("unsupported scored planning schema version")
        for digest in (
            self.source_sha256,
            self.model_artifact_sha256,
            self.model_sha256,
            *self.feature_file_sha256,
        ):
            if len(digest) != 64 or any(item not in "0123456789abcdef" for item in digest):
                raise ValueError("scored planning digest must be SHA-256")

    @property
    def canonical_bytes(self) -> bytes:
        return json.dumps(
            {
                "schema_version": self.schema_version,
                "source_sha256": self.source_sha256,
                "model_artifact_sha256": self.model_artifact_sha256,
                "model_sha256": self.model_sha256,
                "feature_file_sha256": self.feature_file_sha256,
                "planning": self.planning.model_dump(mode="json"),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes).hexdigest()

    @classmethod
    def load(cls, path: Path) -> ScoredPlanningArtifact:
        try:
            raw = json.loads(path.read_bytes())
            artifact = cls(
                schema_version=int(raw["schema_version"]),
                source_sha256=str(raw["source_sha256"]),
                model_artifact_sha256=str(raw["model_artifact_sha256"]),
                model_sha256=str(raw["model_sha256"]),
                feature_file_sha256=tuple(str(item) for item in raw["feature_file_sha256"]),
                planning=DailyOrderPlanningSpec.model_validate(raw["planning"]),
            )
        except (KeyError, OSError, TypeError, ValueError) as error:
            raise ValueError(f"invalid scored planning evidence: {path}") from error
        if artifact.canonical_bytes != path.read_bytes():
            raise ValueError("scored planning evidence is not canonical")
        return artifact


class LivePlanningAssembler:
    """Score exact point-in-time feature vectors and freeze daily planning."""

    def assemble(
        self,
        *,
        source: LivePlanningSourceSpec,
        model: ProductionModelArtifact,
        feature_files: Sequence[Path],
    ) -> ScoredPlanningArtifact:
        if model.training_cutoff > source.feature_asof_date:
            raise ValueError("production model cutoff is after the live feature as-of date")
        observation_symbols = tuple(item.symbol for item in source.observations)
        if observation_symbols:
            vectors, hashes = self._load_vectors(
                feature_files=feature_files,
                asof_date=source.feature_asof_date,
                decision_at=source.decision_at,
                feature_names=model.feature_names,
            )
        else:
            if feature_files:
                raise ValueError("no-candidate live planning must not bind feature artifacts")
            vectors, hashes = {}, ()
        if tuple(sorted(vectors)) != observation_symbols:
            raise ValueError("live feature and market-observation symbol sets differ")
        candidates = tuple(
            LiveCandidateSpec(
                symbol=observation.symbol,
                sector=observation.sector,
                probability_up=model.predict_probability(vectors[observation.symbol]),
                sizing_price=observation.sizing_price,
                sizing_price_observed_at=observation.sizing_price_observed_at,
                frozen_average_daily_volume_shares=(observation.frozen_average_daily_volume_shares),
                decision_snapshot=observation.decision_snapshot,
            )
            for observation in source.observations
        )
        planning = DailyOrderPlanningSpec(
            trade_date=source.trade_date,
            decision_at=source.decision_at,
            equity=source.equity,
            candidates=candidates,
            outcomes=source.outcomes,
            entry_submitted_at=source.entry_submitted_at,
            entry_expires_at=source.entry_expires_at,
            exit_submitted_at=source.exit_submitted_at,
            exit_expires_at=source.exit_expires_at,
            minimum_probability=source.minimum_probability,
        )
        return ScoredPlanningArtifact(
            schema_version=2,
            source_sha256=hashlib.sha256(source.canonical_bytes).hexdigest(),
            model_artifact_sha256=model.sha256,
            model_sha256=model.model_sha256,
            feature_file_sha256=hashes,
            planning=planning,
        )

    @staticmethod
    def _load_vectors(
        *,
        feature_files: Sequence[Path],
        asof_date: date,
        decision_at: datetime,
        feature_names: tuple[str, ...],
    ) -> tuple[dict[str, dict[str, float]], tuple[str, ...]]:
        if not feature_files:
            raise ValueError("at least one live feature file is required")
        rows: list[dict[str, Any]] = []
        hashes = []
        for path in sorted(feature_files):
            if pq.read_schema(path) != FEATURE_VALUE_SCHEMA:  # type: ignore[no-untyped-call]
                raise ValueError(f"live feature artifact schema mismatch: {path}")
            hashes.append(hashlib.sha256(path.read_bytes()).hexdigest())
            rows.extend(
                row
                for row in pq.read_table(path).to_pylist()  # type: ignore[no-untyped-call]
                if row["asof_date"] == asof_date
            )
        vectors: dict[str, dict[str, float]] = {}
        lineage: dict[str, str] = {}
        for row in rows:
            symbol = str(row["symbol"]).strip().upper()
            name = str(row["feature_name"])
            if name not in feature_names:
                continue
            if row["computed_at"] > decision_at:
                raise ValueError(f"live feature was computed after decision_at: {symbol}/{name}")
            if name in vectors.setdefault(symbol, {}):
                raise ValueError(f"duplicate live feature value: {symbol}/{name}")
            vectors[symbol][name] = float(row["value"])
            previous = lineage.setdefault(symbol, str(row["input_sha256"]))
            if previous != row["input_sha256"]:
                raise ValueError(f"mixed live feature input lineage for {symbol}")
        expected = set(feature_names)
        for symbol, values in vectors.items():
            if set(values) != expected:
                raise ValueError(f"incomplete live feature vector for {symbol}")
        return vectors, tuple(hashes)

    @staticmethod
    def write(
        artifact: ScoredPlanningArtifact,
        *,
        planning_output: Path,
        evidence_output: Path,
    ) -> None:
        _write_immutable(planning_output, artifact.planning.canonical_bytes)
        _write_immutable(evidence_output, artifact.canonical_bytes)


def _write_immutable(path: Path, encoded: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as destination:
            destination.write(encoded)
    except FileExistsError:
        if path.read_bytes() != encoded:
            raise RuntimeError(f"live planning artifact collision at {path}") from None
