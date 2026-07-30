"""Purged expanding walk-forward splits for overlapping forward labels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence
    from datetime import date


@dataclass(frozen=True)
class LabeledSample:
    """Minimal temporal identity needed to prevent label overlap."""

    symbol: str
    asof_date: date
    horizon_end_date: date

    def __post_init__(self) -> None:
        normalized = self.symbol.strip().upper()
        if not normalized:
            raise ValueError("symbol must not be empty")
        object.__setattr__(self, "symbol", normalized)
        if self.horizon_end_date <= self.asof_date:
            raise ValueError("horizon_end_date must be after asof_date")


@dataclass(frozen=True)
class WalkForwardConfig:
    """Session-count configuration for an expanding temporal split."""

    minimum_train_sessions: int
    test_sessions: int
    embargo_sessions: int = 5
    step_sessions: int | None = None

    def __post_init__(self) -> None:
        if self.minimum_train_sessions < 1:
            raise ValueError("minimum_train_sessions must be positive")
        if self.test_sessions < 1:
            raise ValueError("test_sessions must be positive")
        if self.embargo_sessions < 1:
            raise ValueError("embargo_sessions must be positive")
        if self.step_sessions is not None and self.step_sessions < 1:
            raise ValueError("step_sessions must be positive")


@dataclass(frozen=True)
class WalkForwardFold:
    """One fully auditable train/embargo/test partition."""

    fold_index: int
    train_indices: tuple[int, ...]
    embargo_dates: tuple[date, ...]
    test_indices: tuple[int, ...]
    train_end_date: date
    test_start_date: date
    test_end_date: date


class PurgedWalkForwardSplitter:
    """Build expanding folds without label-horizon overlap."""

    def __init__(self, config: WalkForwardConfig) -> None:
        self._config = config

    def split(self, samples: Sequence[LabeledSample]) -> tuple[WalkForwardFold, ...]:
        """Return folds indexed into the caller's original sample sequence."""
        if not samples:
            raise ValueError("at least one labeled sample is required")
        keys = [(item.symbol, item.asof_date) for item in samples]
        if len(keys) != len(set(keys)):
            raise ValueError("labeled samples contain duplicate symbol/asof keys")
        dates = tuple(sorted({item.asof_date for item in samples}))
        config = self._config
        first_test_index = config.minimum_train_sessions + config.embargo_sessions
        if first_test_index + config.test_sessions > len(dates):
            raise ValueError("insufficient sessions for one configured walk-forward fold")
        step = config.step_sessions or config.test_sessions
        folds: list[WalkForwardFold] = []
        test_start_index = first_test_index
        while test_start_index + config.test_sessions <= len(dates):
            test_dates = dates[test_start_index : test_start_index + config.test_sessions]
            embargo_dates = dates[test_start_index - config.embargo_sessions : test_start_index]
            train_end_date = dates[test_start_index - config.embargo_sessions - 1]
            test_start_date = test_dates[0]
            train_indices = tuple(
                index
                for index, sample in enumerate(samples)
                if sample.asof_date <= train_end_date and sample.horizon_end_date < test_start_date
            )
            test_indices = tuple(
                index for index, sample in enumerate(samples) if sample.asof_date in test_dates
            )
            if not train_indices or not test_indices:
                raise ValueError("purging produced an empty train or test partition")
            folds.append(
                WalkForwardFold(
                    fold_index=len(folds),
                    train_indices=train_indices,
                    embargo_dates=embargo_dates,
                    test_indices=test_indices,
                    train_end_date=train_end_date,
                    test_start_date=test_start_date,
                    test_end_date=test_dates[-1],
                )
            )
            test_start_index += step
        return tuple(folds)
