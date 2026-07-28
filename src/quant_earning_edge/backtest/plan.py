"""Auditable walk-forward plans built from immutable training artifacts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import date
from typing import TYPE_CHECKING, Any, ClassVar

import pyarrow as pa
import pyarrow.parquet as pq

from quant_earning_edge.backtest.splits import (
    LabeledSample,
    PurgedWalkForwardSplitter,
    WalkForwardConfig,
    WalkForwardFold,
)

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path


@dataclass(frozen=True)
class WalkForwardPlan:
    """Split evidence tied to exact input file content."""

    dataset_sha256: tuple[str, ...]
    sample_count: int
    config: WalkForwardConfig
    folds: tuple[WalkForwardFold, ...]

    def to_json_bytes(self) -> bytes:
        """Serialize canonical evidence for reproducible comparison."""
        return json.dumps(
            asdict(self),
            default=lambda item: item.isoformat(),
            sort_keys=True,
            separators=(",", ":"),
        ).encode()

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.to_json_bytes()).hexdigest()


class WalkForwardPlanner:
    """Load training artifacts and persist a deterministic split manifest."""

    _required_types: ClassVar[dict[str, pa.DataType]] = {
        "symbol": pa.string(),
        "asof_date": pa.date32(),
        "horizon_end_date": pa.date32(),
    }

    def build(
        self,
        dataset_files: Sequence[Path],
        *,
        config: WalkForwardConfig,
    ) -> WalkForwardPlan:
        """Create a plan using row indices in sorted-file concatenation order."""
        if not dataset_files:
            raise ValueError("at least one training dataset file is required")
        samples: list[LabeledSample] = []
        file_hashes: list[str] = []
        for path in sorted(dataset_files):
            schema = pq.read_schema(path)  # type: ignore[no-untyped-call]
            self._validate_schema(schema, path=path)
            file_hashes.append(_file_hash(path))
            rows: list[dict[str, Any]] = pq.read_table(  # type: ignore[no-untyped-call]
                path,
                columns=list(self._required_types),
            ).to_pylist()
            samples.extend(
                LabeledSample(
                    symbol=str(row["symbol"]),
                    asof_date=row["asof_date"],
                    horizon_end_date=row["horizon_end_date"],
                )
                for row in rows
            )
        folds = PurgedWalkForwardSplitter(config).split(samples)
        return WalkForwardPlan(
            dataset_sha256=tuple(file_hashes),
            sample_count=len(samples),
            config=config,
            folds=folds,
        )

    @staticmethod
    def write(plan: WalkForwardPlan, output: Path) -> None:
        """Write immutable canonical JSON, accepting an identical prior write."""
        encoded = plan.to_json_bytes()
        output.parent.mkdir(parents=True, exist_ok=True)
        try:
            with output.open("xb") as destination:
                destination.write(encoded)
        except FileExistsError:
            if output.read_bytes() != encoded:
                raise RuntimeError(f"walk-forward plan collision at {output}") from None

    @staticmethod
    def load(path: Path) -> WalkForwardPlan:
        """Load canonical JSON split evidence."""
        raw = json.loads(path.read_text(encoding="utf-8"))
        try:
            plan = WalkForwardPlan(
                dataset_sha256=tuple(raw["dataset_sha256"]),
                sample_count=int(raw["sample_count"]),
                config=WalkForwardConfig(**raw["config"]),
                folds=tuple(
                    WalkForwardFold(
                        fold_index=int(item["fold_index"]),
                        train_indices=tuple(item["train_indices"]),
                        embargo_dates=tuple(
                            date.fromisoformat(value) for value in item["embargo_dates"]
                        ),
                        test_indices=tuple(item["test_indices"]),
                        train_end_date=date.fromisoformat(item["train_end_date"]),
                        test_start_date=date.fromisoformat(item["test_start_date"]),
                        test_end_date=date.fromisoformat(item["test_end_date"]),
                    )
                    for item in raw["folds"]
                ),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"invalid walk-forward plan: {path}") from error
        if plan.to_json_bytes() != path.read_bytes():
            raise ValueError("walk-forward plan is not canonical or uses unsupported fields")
        return plan

    @classmethod
    def _validate_schema(cls, schema: pa.Schema, *, path: Path) -> None:
        for name, expected_type in cls._required_types.items():
            index = schema.get_field_index(name)
            if index < 0 or schema.field(index).type != expected_type:
                raise ValueError(f"training dataset {path} requires {name}:{expected_type}")


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
