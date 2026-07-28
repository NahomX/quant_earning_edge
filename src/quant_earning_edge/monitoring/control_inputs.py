"""Prepare pre-submit circuit-breaker controls from completed replay evidence."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING

from quant_earning_edge.monitoring.breakers import (
    CircuitBreakerEvaluationSpec,
    CircuitBreakerObservation,
    CircuitBreakerObservationSpec,
)

if TYPE_CHECKING:
    from collections.abc import Sequence
    from datetime import date, datetime
    from pathlib import Path

    from quant_earning_edge.evaluation import ReplaySessionReport
    from quant_earning_edge.signals import FrozenDailyOrders


@dataclass(frozen=True)
class CompletedReplayControlSource:
    """One completed session's frozen sizing and replay outcome."""

    frozen_orders: FrozenDailyOrders
    replay_report: ReplaySessionReport


class CircuitBreakerControlBuilder:
    """Map completed replay sessions onto an explicit new control date."""

    def build(
        self,
        *,
        control_date: date,
        evaluated_at: datetime,
        polygon_data_observed_at: datetime | None,
        alpaca_data_observed_at: datetime | None,
        sources: Sequence[CompletedReplayControlSource],
        reconciliation_break_age_sessions: int | None,
        output: Path,
    ) -> CircuitBreakerEvaluationSpec:
        if not sources:
            raise ValueError("at least one completed replay source is required")
        source_dates = tuple(item.replay_report.session_date for item in sources)
        if source_dates != tuple(sorted(set(source_dates))):
            raise ValueError("completed replay source dates must be unique and sorted")
        if source_dates[-1] >= control_date:
            raise ValueError("breaker replay sources must close before the control date")
        observations: list[CircuitBreakerObservation] = []
        for index, source in enumerate(sources):
            frozen = source.frozen_orders
            report = source.replay_report
            if (
                frozen.trade_date != report.session_date
                or frozen.portfolio.equity != report.initial_cash
            ):
                raise ValueError("frozen sizing and replay report do not reconcile")
            current = index == len(sources) - 1
            observation_time = evaluated_at
            observations.append(
                CircuitBreakerObservation(
                    session_date=control_date if current else report.session_date,
                    replay_source_date=report.session_date,
                    evaluated_at=observation_time,
                    replay_notional=sum(
                        item.target_notional for item in frozen.portfolio.positions
                    ),
                    replay_net_pnl=report.net_pnl,
                    replay_fill_rate=report.fully_filled_order_rate,
                    polygon_data_observed_at=(
                        polygon_data_observed_at if current else observation_time
                    ),
                    alpaca_data_observed_at=(
                        alpaca_data_observed_at if current else observation_time
                    ),
                    reconciliation_break_age_sessions=(
                        reconciliation_break_age_sessions if current else None
                    ),
                )
            )
        spec = CircuitBreakerEvaluationSpec(
            observations=tuple(
                CircuitBreakerObservationSpec.model_validate(item.__dict__) for item in observations
            )
        )
        _write_once(
            output,
            json.dumps(
                spec.model_dump(mode="json"),
                sort_keys=True,
                separators=(",", ":"),
            ).encode(),
        )
        return spec


def _write_once(path: Path, encoded: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as destination:
            destination.write(encoded)
    except FileExistsError:
        if path.read_bytes() != encoded:
            raise RuntimeError(f"circuit-breaker control collision at {path}") from None
