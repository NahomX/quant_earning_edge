"""Walk-forward fold aggregation and gate evidence."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence
    from datetime import date
    from pathlib import Path

    from quant_earning_edge.evaluation.report import PerformanceReport


@dataclass(frozen=True)
class FoldEvaluation:
    """Metrics and temporal identity for one out-of-sample fold."""

    fold_index: int
    test_start_date: date
    test_end_date: date
    input_sha256: str
    trade_count: int
    net_sharpe: float
    annualized_return: float
    max_drawdown: float


@dataclass(frozen=True)
class WalkForwardEvaluation:
    """Aggregate evidence for the documented positive-fold gate."""

    folds: tuple[FoldEvaluation, ...]
    mean_net_sharpe: float
    positive_sharpe_fold_count: int
    positive_sharpe_fraction: float
    passes_positive_fold_gate: bool

    def to_json_bytes(self) -> bytes:
        """Return canonical JSON for durable evidence."""
        return json.dumps(
            asdict(self),
            default=lambda item: item.isoformat(),
            sort_keys=True,
            separators=(",", ":"),
        ).encode()

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.to_json_bytes()).hexdigest()


class WalkForwardEvaluator:
    """Combine independently evaluated OOS folds without hiding weak folds."""

    def aggregate(
        self,
        folds: Sequence[tuple[int, date, date, PerformanceReport]],
    ) -> WalkForwardEvaluation:
        """Validate ordered non-overlapping folds and calculate gate evidence."""
        if not folds:
            raise ValueError("at least one evaluated fold is required")
        expected_indices = tuple(range(len(folds)))
        actual_indices = tuple(item[0] for item in folds)
        if actual_indices != expected_indices:
            raise ValueError("fold indices must be consecutive and start at zero")
        output: list[FoldEvaluation] = []
        previous_end: date | None = None
        for fold_index, test_start, test_end, report in folds:
            if test_start > test_end:
                raise ValueError(f"fold {fold_index} starts after it ends")
            if previous_end is not None and test_start <= previous_end:
                raise ValueError("fold test windows must be increasing and non-overlapping")
            output.append(
                FoldEvaluation(
                    fold_index=fold_index,
                    test_start_date=test_start,
                    test_end_date=test_end,
                    input_sha256=report.input_sha256,
                    trade_count=report.trade_count,
                    net_sharpe=report.net_sharpe,
                    annualized_return=report.annualized_return,
                    max_drawdown=report.max_drawdown,
                )
            )
            previous_end = test_end
        positive_count = sum(item.net_sharpe > 0 for item in output)
        positive_fraction = positive_count / len(output)
        return WalkForwardEvaluation(
            folds=tuple(output),
            mean_net_sharpe=sum(item.net_sharpe for item in output) / len(output),
            positive_sharpe_fold_count=positive_count,
            positive_sharpe_fraction=positive_fraction,
            passes_positive_fold_gate=positive_fraction >= 0.75,
        )

    @staticmethod
    def write(report: WalkForwardEvaluation, output: Path) -> None:
        """Persist immutable fold evidence."""
        encoded = report.to_json_bytes()
        output.parent.mkdir(parents=True, exist_ok=True)
        try:
            with output.open("xb") as destination:
                destination.write(encoded)
        except FileExistsError:
            if output.read_bytes() != encoded:
                raise RuntimeError(f"walk-forward report collision at {output}") from None
