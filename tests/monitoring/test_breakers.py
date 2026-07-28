"""Operational circuit-breaker boundary tests."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime, timedelta
from typing import TYPE_CHECKING

import pytest
from typer.testing import CliRunner

from quant_earning_edge.cli import app
from quant_earning_edge.monitoring import (
    CircuitBreakerDecision,
    CircuitBreakerEvaluator,
    CircuitBreakerObservation,
)
from quant_earning_edge.monitoring.breakers import (
    ALPACA_STALE,
    DAILY_REPLAY_LOSS,
    POLYGON_STALE,
    RECONCILIATION_OVERDUE,
    THREE_DAY_LOW_FILL,
)

if TYPE_CHECKING:
    from pathlib import Path


def _observation(
    *,
    day: int = 3,
    loss_fraction: float = 0.0,
    fill_rate: float | None = 1.0,
    polygon_age_minutes: int | None = 0,
    alpaca_age_minutes: int | None = 0,
    reconciliation_age: int | None = None,
) -> CircuitBreakerObservation:
    evaluated_at = datetime(2025, 1, day, 21, 5, tzinfo=UTC)
    return CircuitBreakerObservation(
        session_date=date(2025, 1, day),
        evaluated_at=evaluated_at,
        replay_notional=100_000,
        replay_net_pnl=-100_000 * loss_fraction,
        replay_fill_rate=fill_rate,
        polygon_data_observed_at=(
            None
            if polygon_age_minutes is None
            else evaluated_at - timedelta(minutes=polygon_age_minutes)
        ),
        alpaca_data_observed_at=(
            None
            if alpaca_age_minutes is None
            else evaluated_at - timedelta(minutes=alpaca_age_minutes)
        ),
        reconciliation_break_age_sessions=reconciliation_age,
    )


def test_healthy_observation_allows_new_orders() -> None:
    decision = CircuitBreakerEvaluator().evaluate((_observation(),))

    assert not decision.halt_new_orders
    assert decision.triggered_breakers == ()
    assert decision.replay_loss_fraction == 0


@pytest.mark.parametrize(
    ("loss_fraction", "halts"),
    [(0.02, False), (0.020_001, True)],
)
def test_daily_loss_boundary_is_strict(loss_fraction: float, *, halts: bool) -> None:
    decision = CircuitBreakerEvaluator().evaluate(
        (_observation(loss_fraction=loss_fraction),),
    )

    assert (DAILY_REPLAY_LOSS in decision.triggered_breakers) is halts


def test_low_fill_requires_three_consecutive_applicable_sessions() -> None:
    evaluator = CircuitBreakerEvaluator()
    two_days = (_observation(day=1, fill_rate=0.69), _observation(day=2, fill_rate=0.69))
    three_days = (*two_days, _observation(day=3, fill_rate=0.69))

    assert THREE_DAY_LOW_FILL not in evaluator.evaluate(two_days).triggered_breakers
    assert THREE_DAY_LOW_FILL in evaluator.evaluate(three_days).triggered_breakers


def test_fill_rate_boundary_and_no_trade_day_interrupt_streak() -> None:
    observations = (
        _observation(day=1, fill_rate=0.69),
        _observation(day=2, fill_rate=None),
        _observation(day=3, fill_rate=0.69),
        _observation(day=4, fill_rate=0.69),
    )
    at_boundary = (*observations, _observation(day=5, fill_rate=0.70))

    assert CircuitBreakerEvaluator().evaluate(observations).consecutive_low_fill_sessions == 2
    assert CircuitBreakerEvaluator().evaluate(at_boundary).consecutive_low_fill_sessions == 0


@pytest.mark.parametrize(
    ("age_minutes", "halts"),
    [(30, False), (31, True), (None, True)],
)
def test_provider_freshness_is_fail_closed(
    age_minutes: int | None,
    *,
    halts: bool,
) -> None:
    decision = CircuitBreakerEvaluator().evaluate(
        (
            _observation(
                polygon_age_minutes=age_minutes,
                alpaca_age_minutes=age_minutes,
            ),
        ),
    )

    assert (POLYGON_STALE in decision.triggered_breakers) is halts
    assert (ALPACA_STALE in decision.triggered_breakers) is halts


@pytest.mark.parametrize(
    ("age_sessions", "halts"),
    [(None, False), (0, False), (1, True)],
)
def test_reconciliation_halts_at_t_plus_one_close(
    age_sessions: int | None,
    *,
    halts: bool,
) -> None:
    decision = CircuitBreakerEvaluator().evaluate(
        (_observation(reconciliation_age=age_sessions),),
    )

    assert (RECONCILIATION_OVERDUE in decision.triggered_breakers) is halts


def test_decision_is_collision_safe_and_hashes_canonical_evidence(tmp_path: Path) -> None:
    output = tmp_path / "breaker.json"
    decision = CircuitBreakerEvaluator().evaluate((_observation(),))
    decision.write(output)
    decision.write(output)

    assert decision.sha256
    assert output.read_bytes() == decision.canonical_bytes
    conflicting = CircuitBreakerDecision(
        **{
            **decision.__dict__,
            "triggered_breakers": (POLYGON_STALE,),
            "halt_new_orders": True,
        }
    )
    with pytest.raises(RuntimeError, match="collision"):
        conflicting.write(output)
    assert CircuitBreakerDecision.load(output).sha256 == decision.sha256


def test_circuit_breaker_cli_writes_halt_and_exits_nonzero(tmp_path: Path) -> None:
    spec_file = tmp_path / "observations.json"
    output = tmp_path / "decision.json"
    observation = _observation(loss_fraction=0.03)
    spec_file.write_text(
        json.dumps(
            {
                "observations": [
                    {
                        **observation.__dict__,
                        "session_date": observation.session_date.isoformat(),
                        "evaluated_at": observation.evaluated_at.isoformat(),
                        "polygon_data_observed_at": (
                            observation.polygon_data_observed_at.isoformat()
                            if observation.polygon_data_observed_at
                            else None
                        ),
                        "alpaca_data_observed_at": (
                            observation.alpaca_data_observed_at.isoformat()
                            if observation.alpaca_data_observed_at
                            else None
                        ),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        app,
        [
            "monitoring",
            "circuit-breakers",
            "--spec-file",
            str(spec_file),
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 1
    command_result = json.loads(result.stdout)
    assert command_result["halt_new_orders"]
    assert DAILY_REPLAY_LOSS in command_result["triggered_breakers"]
    assert CircuitBreakerDecision.load(output).halt_new_orders


def test_decision_loader_rejects_tampered_halt_flag(tmp_path: Path) -> None:
    output = tmp_path / "decision.json"
    decision = CircuitBreakerEvaluator().evaluate((_observation(),))
    raw = json.loads(decision.canonical_bytes)
    raw["halt_new_orders"] = True
    output.write_text(
        json.dumps(raw, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="invalid circuit-breaker"):
        CircuitBreakerDecision.load(output)


def test_observations_must_be_ordered_and_provider_times_cannot_be_future() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        CircuitBreakerEvaluator().evaluate(
            (_observation(day=2), _observation(day=1)),
        )
    with pytest.raises(ValueError, match="cannot be after"):
        current = _observation()
        CircuitBreakerObservation(
            **{
                **current.__dict__,
                "polygon_data_observed_at": current.evaluated_at + timedelta(seconds=1),
            }
        )
