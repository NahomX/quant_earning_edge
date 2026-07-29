"""Source-bound Phase 3 published momentum benchmark gate tests."""

from __future__ import annotations

import hashlib
import json
from datetime import date, timedelta
from typing import TYPE_CHECKING

import pytest
from typer.testing import CliRunner

from quant_earning_edge.backtest import (
    BacktestSpec,
    VectorbtBacktestEngine,
)
from quant_earning_edge.cli import app
from quant_earning_edge.evaluation import (
    MomentumBaselineManifest,
    MomentumBenchmarkGateEvaluator,
    MomentumBenchmarkReferenceSpec,
    PerformanceEvaluator,
    PerformanceReport,
)

if TYPE_CHECKING:
    from pathlib import Path


def _inputs(
    tmp_path: Path,
    *,
    reference_difference: float = 0.05,
) -> tuple[Path, Path, Path, Path, Path, Path]:
    sessions = tuple(date(2024, 1, 1) + timedelta(days=index) for index in range(300))
    spec = BacktestSpec.model_validate(
        {
            "initial_cash": 100_000.0,
            "sessions": sessions,
            "marks": [
                {
                    "symbol": symbol,
                    "session_date": session,
                    "close": (
                        102.0
                        if symbol == "WIN" and session >= sessions[61]
                        else 98.0
                        if symbol == "LOSS" and session >= sessions[121]
                        else 100.0
                    ),
                }
                for session in sessions
                for symbol in ("LOSS", "WIN")
            ],
            "trades": [
                {
                    "trade_id": "winner",
                    "symbol": "WIN",
                    "side": "long",
                    "entry_date": sessions[60],
                    "exit_date": sessions[61],
                    "shares": 100,
                    "entry_price": 100.0,
                    "exit_price": 102.0,
                    "entry_average_daily_volume_shares": 1_000_000.0,
                    "exit_average_daily_volume_shares": 1_000_000.0,
                    "holding_sessions": 1,
                },
                {
                    "trade_id": "loser",
                    "symbol": "LOSS",
                    "side": "long",
                    "entry_date": sessions[120],
                    "exit_date": sessions[121],
                    "shares": 100,
                    "entry_price": 100.0,
                    "exit_price": 98.0,
                    "entry_average_daily_volume_shares": 1_000_000.0,
                    "exit_average_daily_volume_shares": 1_000_000.0,
                    "holding_sessions": 1,
                },
            ],
        }
    )
    initial_cash, domain_sessions, marks, trades = spec.domain_inputs()
    result = VectorbtBacktestEngine().run(
        trades=trades,
        marks=marks,
        sessions=domain_sessions,
        initial_cash=initial_cash,
    )
    report = PerformanceEvaluator(bootstrap_resamples=20, seed=7).evaluate(result)
    reference_artifact = tmp_path / "published-paper.pdf"
    reference_artifact.write_bytes(b"pinned published momentum result")
    reference = MomentumBenchmarkReferenceSpec.model_validate(
        {
            "benchmark_name": "Published SPY-component 60-session momentum",
            "source_url": "https://example.org/published-momentum.pdf",
            "source_artifact_sha256": hashlib.sha256(reference_artifact.read_bytes()).hexdigest(),
            "published_net_sharpe": report.net_sharpe + reference_difference,
            "minimum_session_count": 252,
        }
    )
    reference_path = tmp_path / "reference.json"
    reference_path.write_bytes(reference.canonical_bytes)
    universe = tmp_path / "historical-spy-components.parquet"
    universe.write_bytes(b"historical membership")
    trade_plan = tmp_path / "momentum-trade-plan.json"
    trade_plan.write_bytes(
        json.dumps(
            spec.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    )
    manifest = MomentumBaselineManifest(
        schema_version=1,
        strategy="cross_sectional_momentum_60_session",
        lookback_sessions=60,
        selection_fraction=0.1,
        universe="historical_spy_components",
        universe_artifact_sha256=hashlib.sha256(universe.read_bytes()).hexdigest(),
        trade_plan_sha256=hashlib.sha256(trade_plan.read_bytes()).hexdigest(),
        backtest_input_sha256=report.input_sha256,
    )
    manifest_path = tmp_path / "baseline.json"
    manifest_path.write_bytes(manifest.canonical_bytes)
    report_path = tmp_path / "performance.json"
    PerformanceEvaluator.write(report, report_path)
    return (
        reference_path,
        reference_artifact,
        manifest_path,
        universe,
        trade_plan,
        report_path,
    )


def test_momentum_gate_passes_only_source_bound_matching_report(tmp_path: Path) -> None:
    reference, artifact, manifest, universe, trade_plan, performance = _inputs(tmp_path)

    result = MomentumBenchmarkGateEvaluator().evaluate(
        reference_spec=MomentumBenchmarkReferenceSpec.load(reference),
        reference_artifact=artifact,
        universe_artifact=universe,
        trade_plan=trade_plan,
        baseline_manifest=MomentumBaselineManifest.load(manifest),
        performance_report=PerformanceReport.load(performance),
    )

    assert result.passes
    assert result.absolute_difference == pytest.approx(0.05)
    assert result.performance_report_sha256 == hashlib.sha256(performance.read_bytes()).hexdigest()


def test_momentum_gate_rejects_tampered_published_artifact(tmp_path: Path) -> None:
    reference, artifact, manifest, universe, trade_plan, performance = _inputs(tmp_path)
    artifact.write_bytes(b"changed")

    with pytest.raises(ValueError, match="reference artifact hash differs"):
        MomentumBenchmarkGateEvaluator().evaluate(
            reference_spec=MomentumBenchmarkReferenceSpec.load(reference),
            reference_artifact=artifact,
            universe_artifact=universe,
            trade_plan=trade_plan,
            baseline_manifest=MomentumBaselineManifest.load(manifest),
            performance_report=PerformanceReport.load(performance),
        )


@pytest.mark.parametrize(
    ("target_index", "message"),
    (
        (3, "historical-universe artifact hash differs"),
        (4, "trade-plan artifact hash differs"),
    ),
)
def test_momentum_gate_rejects_tampered_baseline_artifacts(
    tmp_path: Path,
    target_index: int,
    message: str,
) -> None:
    inputs = _inputs(tmp_path)
    reference, artifact, manifest, universe, trade_plan, performance = inputs
    inputs[target_index].write_bytes(b"changed")

    with pytest.raises(ValueError, match=message):
        MomentumBenchmarkGateEvaluator().evaluate(
            reference_spec=MomentumBenchmarkReferenceSpec.load(reference),
            reference_artifact=artifact,
            universe_artifact=universe,
            trade_plan=trade_plan,
            baseline_manifest=MomentumBaselineManifest.load(manifest),
            performance_report=PerformanceReport.load(performance),
        )


def test_momentum_gate_rejects_nonreproducing_performance(tmp_path: Path) -> None:
    reference, artifact, manifest, universe, trade_plan, performance = _inputs(tmp_path)
    payload = json.loads(performance.read_bytes())
    payload["net_sharpe"] += 0.01
    performance.write_bytes(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode())

    with pytest.raises(ValueError, match="does not reproduce from trade plan"):
        MomentumBenchmarkGateEvaluator().evaluate(
            reference_spec=MomentumBenchmarkReferenceSpec.load(reference),
            reference_artifact=artifact,
            universe_artifact=universe,
            trade_plan=trade_plan,
            baseline_manifest=MomentumBaselineManifest.load(manifest),
            performance_report=PerformanceReport.load(performance),
        )


def test_momentum_gate_cli_persists_failed_comparison(tmp_path: Path) -> None:
    reference, artifact, manifest, universe, trade_plan, performance = _inputs(
        tmp_path,
        reference_difference=0.25,
    )
    output = tmp_path / "gate.json"

    result = CliRunner().invoke(
        app,
        [
            "evaluation",
            "momentum-benchmark-gate",
            "--reference-spec",
            str(reference),
            "--reference-artifact",
            str(artifact),
            "--baseline-manifest",
            str(manifest),
            "--universe-artifact",
            str(universe),
            "--trade-plan",
            str(trade_plan),
            "--performance-report",
            str(performance),
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 1
    assert not json.loads(result.stdout)["passes"]
    assert not json.loads(output.read_bytes())["passes"]
