"""Source-rebuilt Phase 3 published momentum benchmark gate tests."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from typer.testing import CliRunner

from quant_earning_edge.backtest import VectorbtBacktestEngine
from quant_earning_edge.cli import app
from quant_earning_edge.data import (
    BronzeWriter,
    LakehouseLayout,
    SessionFileStore,
    SplitHistorySourceCapture,
)
from quant_earning_edge.data.clients import MarketSession
from quant_earning_edge.evaluation import (
    HISTORICAL_SPY_MEMBERSHIP_SCHEMA,
    MomentumBaselineBuilder,
    MomentumBaselineBuildSpec,
    MomentumBaselineManifest,
    MomentumBenchmarkGateEvaluator,
    MomentumBenchmarkGateReport,
    MomentumBenchmarkReferenceSpec,
    PerformanceEvaluator,
    PerformanceReport,
)


@dataclass(frozen=True)
class _Inputs:
    reference: Path
    reference_artifact: Path
    manifest: Path
    universe: Path
    trade_plan: Path
    performance: Path
    build_spec: Path
    session_file: Path
    daily_bars: Path
    split_source: Path


def _inputs(
    tmp_path: Path,
    *,
    reference_difference: float = 0.05,
) -> _Inputs:
    dates = tuple(date(2024, 1, 1) + timedelta(days=index) for index in range(361))
    layout = LakehouseLayout(tmp_path / "lake")
    calendar = SessionFileStore(layout).write(
        tuple(
            MarketSession(
                session_date=item,
                open_at=datetime(item.year, item.month, item.day, 14, 30, tzinfo=UTC),
                close_at=datetime(item.year, item.month, item.day, 21, 0, tzinfo=UTC),
            )
            for item in dates
        )
    )
    universe = tmp_path / "historical-spy-components.parquet"
    pq.write_table(  # type: ignore[no-untyped-call]
        pa.Table.from_pylist(
            [
                {
                    "symbol": symbol,
                    "effective_from": dates[0],
                    "effective_through": dates[-1],
                }
                for symbol in ("LOSS", "WIN")
            ],
            schema=HISTORICAL_SPY_MEMBERSHIP_SCHEMA,
        ),
        universe,
    )
    daily_bars = tmp_path / "daily-bars.parquet"
    pq.write_table(  # type: ignore[no-untyped-call]
        pa.Table.from_pylist(
            [
                {
                    "session_date": session,
                    "symbol": symbol,
                    "open": close,
                    "close": close,
                    "volume": 1_000_000.0,
                    "adjusted": False,
                }
                for index, session in enumerate(dates)
                for symbol, close in (
                    ("LOSS", 100.0 - index * 0.05),
                    ("WIN", 100.0 + index * 0.10),
                )
            ]
        ),
        daily_bars,
    )
    methodology = MomentumBaselineBuildSpec(
        signal_start_date=dates[60],
        signal_end_date=dates[358],
        rebalance_interval_sessions=63,
        holding_sessions=1,
    )
    build_spec = tmp_path / "momentum-build.json"
    build_spec.write_bytes(methodology.canonical_bytes)
    split_observation = BronzeWriter(layout).write_json(
        {"status": "OK", "results": []},
        source="polygon",
        dataset="stock-splits",
        event_date=dates[0],
    )
    split_source = SplitHistorySourceCapture(layout).write(
        plan_id="a" * 64,
        start_date=dates[0],
        end_date=dates[-1],
        ingested_at=datetime(2025, 1, 1, tzinfo=UTC),
        split_files=(),
        provider_observations=(split_observation,),
    )
    built = MomentumBaselineBuilder().build(
        spec=methodology,
        calendar=calendar,
        universe_artifact=universe,
        daily_bar_files=(daily_bars,),
        split_source_manifest=split_source.path,
    )
    trade_plan = tmp_path / "momentum-trade-plan.json"
    built.write_trade_plan(trade_plan)
    manifest = MomentumBaselineManifest(
        schema_version=3,
        strategy="cross_sectional_momentum_60_session",
        lookback_sessions=60,
        selection_fraction=0.1,
        universe="historical_spy_components",
        universe_artifact_sha256=built.universe_artifact_sha256,
        trade_plan_sha256=built.trade_plan_sha256,
        backtest_input_sha256=built.trade_plan.input_sha256,
        build_spec_sha256=built.build_spec_sha256,
        session_file_sha256=built.session_file_sha256,
        daily_bar_sha256=built.daily_bar_sha256,
        split_source_sha256=built.split_source_sha256 or "",
    )
    manifest_path = tmp_path / "baseline.json"
    manifest.write(manifest_path)
    initial_cash, sessions, marks, trades = built.trade_plan.domain_inputs()
    result = VectorbtBacktestEngine().run(
        trades=trades,
        marks=marks,
        sessions=sessions,
        initial_cash=initial_cash,
    )
    report = PerformanceEvaluator(bootstrap_resamples=20, seed=7).evaluate(result)
    performance = tmp_path / "performance.json"
    PerformanceEvaluator.write(report, performance)
    reference_artifact = tmp_path / "published-paper.pdf"
    reference_artifact.write_bytes(b"pinned published momentum result")
    reference_spec = MomentumBenchmarkReferenceSpec.model_validate(
        {
            "benchmark_name": "Published SPY-component 60-session momentum",
            "source_url": "https://example.org/published-momentum.pdf",
            "universe_source_url": "https://example.org/historical-spy-membership",
            "source_artifact_sha256": hashlib.sha256(reference_artifact.read_bytes()).hexdigest(),
            "baseline_build_spec_sha256": methodology.sha256,
            "published_net_sharpe": report.net_sharpe + reference_difference,
            "minimum_session_count": 252,
        }
    )
    reference = tmp_path / "reference.json"
    reference_spec.write(reference)
    return _Inputs(
        reference=reference,
        reference_artifact=reference_artifact,
        manifest=manifest_path,
        universe=universe,
        trade_plan=trade_plan,
        performance=performance,
        build_spec=build_spec,
        session_file=calendar.path,
        daily_bars=daily_bars,
        split_source=split_source.path,
    )


def _evaluate(inputs: _Inputs) -> MomentumBenchmarkGateReport:
    return MomentumBenchmarkGateEvaluator().evaluate(
        reference_spec=MomentumBenchmarkReferenceSpec.load(inputs.reference),
        reference_artifact=inputs.reference_artifact,
        universe_artifact=inputs.universe,
        trade_plan=inputs.trade_plan,
        build_spec=MomentumBaselineBuildSpec.load(inputs.build_spec),
        calendar=SessionFileStore.load(inputs.session_file),
        daily_bar_files=(inputs.daily_bars,),
        split_source_manifest=inputs.split_source,
        baseline_manifest=MomentumBaselineManifest.load(inputs.manifest),
        performance_report=PerformanceReport.load(inputs.performance),
    )


def test_momentum_build_example_is_canonical() -> None:
    example = Path(__file__).parents[2] / "configs" / "evaluation" / "momentum_build.example.json"

    assert MomentumBaselineBuildSpec.load(example).lookback_sessions == 60


def test_momentum_gate_passes_only_rebuilt_matching_report(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)

    result = _evaluate(inputs)

    assert result.passes
    assert result.absolute_difference == pytest.approx(0.05)
    assert (
        result.performance_report_sha256
        == hashlib.sha256(inputs.performance.read_bytes()).hexdigest()
    )


def test_momentum_baseline_cli_rebuilds_exact_artifacts(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    trade_plan = tmp_path / "rebuilt-plan.json"
    manifest = tmp_path / "rebuilt-manifest.json"

    result = CliRunner().invoke(
        app,
        [
            "evaluation",
            "build-momentum-baseline",
            "--build-spec",
            str(inputs.build_spec),
            "--session-file",
            str(inputs.session_file),
            "--universe-artifact",
            str(inputs.universe),
            "--daily-bar-file",
            str(inputs.daily_bars),
            "--split-source-manifest",
            str(inputs.split_source),
            "--trade-plan-output",
            str(trade_plan),
            "--manifest-output",
            str(manifest),
        ],
    )

    assert result.exit_code == 0, result.stderr
    assert trade_plan.read_bytes() == inputs.trade_plan.read_bytes()
    assert manifest.read_bytes() == inputs.manifest.read_bytes()


def test_future_membership_and_bars_cannot_change_trade_plan(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    methodology = MomentumBaselineBuildSpec.load(inputs.build_spec)
    calendar = SessionFileStore.load(inputs.session_file)
    original = MomentumBaselineBuilder().build(
        spec=methodology,
        calendar=calendar,
        universe_artifact=inputs.universe,
        daily_bar_files=(inputs.daily_bars,),
        split_source_manifest=inputs.split_source,
    )
    future_date = calendar.sessions[-1].session_date + timedelta(days=1)
    extended_universe = tmp_path / "extended-universe.parquet"
    membership = pq.read_table(inputs.universe)  # type: ignore[no-untyped-call]
    pq.write_table(  # type: ignore[no-untyped-call]
        pa.concat_tables(
            [
                membership,
                pa.Table.from_pylist(
                    [
                        {
                            "symbol": "FUTURE",
                            "effective_from": calendar.sessions[-2].session_date,
                            "effective_through": future_date,
                        }
                    ],
                    schema=HISTORICAL_SPY_MEMBERSHIP_SCHEMA,
                ),
            ]
        ),
        extended_universe,
    )
    extended_bars = tmp_path / "extended-bars.parquet"
    bars = pq.read_table(inputs.daily_bars)  # type: ignore[no-untyped-call]
    pq.write_table(  # type: ignore[no-untyped-call]
        pa.concat_tables(
            [
                bars,
                pa.Table.from_pylist(
                    [
                        {
                            "session_date": future_date,
                            "symbol": symbol,
                            "open": 1_000_000.0,
                            "close": 1_000_000.0,
                            "volume": 1_000_000.0,
                            "adjusted": False,
                        }
                        for symbol in ("LOSS", "WIN")
                    ],
                    schema=bars.schema,
                ),
            ]
        ),
        extended_bars,
    )

    extended = MomentumBaselineBuilder().build(
        spec=methodology,
        calendar=calendar,
        universe_artifact=extended_universe,
        daily_bar_files=(extended_bars,),
        split_source_manifest=inputs.split_source,
    )

    assert extended.trade_plan_bytes == original.trade_plan_bytes


def test_momentum_gate_rejects_tampered_published_artifact(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    inputs.reference_artifact.write_bytes(b"changed")

    with pytest.raises(ValueError, match="reference artifact hash differs"):
        _evaluate(inputs)


@pytest.mark.parametrize(
    ("target", "message"),
    (
        ("universe", "historical-universe artifact hash differs"),
        ("trade_plan", "trade-plan artifact hash differs"),
        ("daily_bars", "daily-bar artifact hashes differ"),
        ("split_source", "split-history source hash differs"),
    ),
)
def test_momentum_gate_rejects_tampered_source_artifacts(
    tmp_path: Path,
    target: str,
    message: str,
) -> None:
    inputs = _inputs(tmp_path)
    getattr(inputs, target).write_bytes(b"changed")

    with pytest.raises(ValueError, match=message):
        _evaluate(inputs)


def test_momentum_gate_rejects_nonreproducing_performance(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    payload = json.loads(inputs.performance.read_bytes())
    payload["net_sharpe"] += 0.01
    inputs.performance.write_bytes(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    )

    with pytest.raises(ValueError, match="does not reproduce from trade plan"):
        _evaluate(inputs)


def test_momentum_gate_cli_persists_failed_comparison(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path, reference_difference=0.25)
    output = tmp_path / "gate.json"

    result = CliRunner().invoke(
        app,
        [
            "evaluation",
            "momentum-benchmark-gate",
            "--reference-spec",
            str(inputs.reference),
            "--reference-artifact",
            str(inputs.reference_artifact),
            "--build-spec",
            str(inputs.build_spec),
            "--session-file",
            str(inputs.session_file),
            "--daily-bar-file",
            str(inputs.daily_bars),
            "--split-source-manifest",
            str(inputs.split_source),
            "--baseline-manifest",
            str(inputs.manifest),
            "--universe-artifact",
            str(inputs.universe),
            "--trade-plan",
            str(inputs.trade_plan),
            "--performance-report",
            str(inputs.performance),
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 1, result.stderr
    assert not json.loads(result.stdout)["passes"]
    assert not json.loads(output.read_bytes())["passes"]
