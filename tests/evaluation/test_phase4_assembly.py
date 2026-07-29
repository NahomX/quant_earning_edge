"""Automated, source-bound Phase 4 historical assembly tests."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from datetime import UTC, date, datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from typer.testing import CliRunner

from quant_earning_edge.cli import app
from quant_earning_edge.data import DAILY_BARS_SCHEMA, LakehouseLayout, SessionFileStore
from quant_earning_edge.data.clients import MarketSession
from quant_earning_edge.evaluation import (
    Phase4AggregationSpec,
)
from quant_earning_edge.evaluation.phase4_assembly import (
    Phase4AssemblyManifest,
    Phase4AssemblyResult,
    Phase4HistoricalAssembler,
)
from quant_earning_edge.evaluation.phase4_verification import Phase4GateVerifier
from quant_earning_edge.signals import (
    EventTradePlanner,
    FeatureAttribution,
    FoldModelResult,
    LightgbmHyperparameters,
    OosPrediction,
    WalkForwardModelRun,
    load_strategy_config,
)
from quant_earning_edge.universe import EVENT_CANDIDATE_SCHEMA


def _sources(tmp_path: Path) -> dict[str, object]:
    strategy_path = Path("configs/strategies/earnings_v1.yaml").resolve()
    strategy = load_strategy_config(strategy_path)
    sessions = (
        MarketSession(
            session_date=date(2025, 1, 2),
            open_at=datetime(2025, 1, 2, 14, 30, tzinfo=UTC),
            close_at=datetime(2025, 1, 2, 21, 0, tzinfo=UTC),
        ),
        MarketSession(
            session_date=date(2025, 1, 3),
            open_at=datetime(2025, 1, 3, 14, 30, tzinfo=UTC),
            close_at=datetime(2025, 1, 3, 21, 0, tzinfo=UTC),
        ),
        MarketSession(
            session_date=date(2025, 1, 6),
            open_at=datetime(2025, 1, 6, 14, 30, tzinfo=UTC),
            close_at=datetime(2025, 1, 6, 21, 0, tzinfo=UTC),
        ),
    )
    session_artifact = SessionFileStore(LakehouseLayout(tmp_path / "lake")).write(sessions)
    predictions = (
        OosPrediction(
            0,
            "AAA",
            date(2025, 1, 2),
            datetime(2025, 1, 3, 14, 10, tzinfo=UTC),
            0.8,
            1,
        ),
        OosPrediction(
            1,
            "BBB",
            date(2025, 1, 3),
            datetime(2025, 1, 6, 14, 10, tzinfo=UTC),
            0.7,
            0,
        ),
    )
    hyperparameters = LightgbmHyperparameters()
    run = WalkForwardModelRun(
        plan_sha256="a" * 64,
        dataset_sha256=("b" * 64,),
        feature_names=strategy.features,
        label_name=strategy.label.column_name,
        threshold=strategy.label.threshold,
        seed=strategy.seed,
        hyperparameter_study_sha256="c" * 64,
        hyperparameters=hyperparameters,
        hyperparameters_sha256=hyperparameters.sha256,
        lightgbm_version="test",
        folds=(
            FoldModelResult(
                fold_index=0,
                model_sha256="d" * 64,
                best_iteration=1,
                fit_count=20,
                validation_count=5,
                predictions=predictions,
                feature_attribution=tuple(
                    FeatureAttribution(name, 0.0) for name in strategy.features
                ),
            ),
        ),
    )
    run_path = tmp_path / "walkforward-run.json"
    run_path.write_bytes(run.evidence_json_bytes())
    candidate_path = tmp_path / "candidates.parquet"
    candidate_rows = [
        _candidate(
            symbol="AAA",
            sector="Technology",
            trade_date=date(2025, 1, 3),
            asof_date=date(2025, 1, 2),
            decision_at=sessions[0].close_at,
            session_sha256=session_artifact.sha256,
            timing="bmo",
        ),
        _candidate(
            symbol="BBB",
            sector="Health Care",
            trade_date=date(2025, 1, 6),
            asof_date=date(2025, 1, 3),
            decision_at=sessions[1].close_at,
            session_sha256=session_artifact.sha256,
            timing="amc",
        ),
    ]
    pq.write_table(  # type: ignore[no-untyped-call]
        pa.Table.from_pylist(candidate_rows, schema=EVENT_CANDIDATE_SCHEMA),
        candidate_path,
    )
    bar_path = tmp_path / "bars.parquet"
    bar_rows = [
        _bar("AAA", date(2025, 1, 3), 100.0, 102.0),
        _bar("BBB", date(2025, 1, 6), 50.0, 49.0),
    ]
    pq.write_table(  # type: ignore[no-untyped-call]
        pa.Table.from_pylist(bar_rows, schema=DAILY_BARS_SCHEMA),
        bar_path,
    )
    return {
        "strategy": strategy_path,
        "sessions": session_artifact.path,
        "run": run_path,
        "candidates": candidate_path,
        "bars": bar_path,
        "bar_rows": bar_rows,
    }


def _candidate(
    *,
    symbol: str,
    sector: str,
    trade_date: date,
    asof_date: date,
    decision_at: datetime,
    session_sha256: str,
    timing: str,
) -> dict[str, object]:
    return {
        "trade_date": trade_date,
        "asof_date": asof_date,
        "decision_at": decision_at,
        "symbol": symbol,
        "sector": sector,
        "sizing_price": 100.0 if symbol == "AAA" else 50.0,
        "frozen_average_daily_volume_shares": 1_000_000.0,
        "event_date": trade_date if timing == "bmo" else asof_date,
        "timing": timing,
        "year": 2025,
        "quarter": 1,
        "eps_estimate": 1.0,
        "revenue_estimate": 1_000_000.0,
        "split_event_ids": [],
        "dividend_event_ids": [],
        "universe_snapshot_sha256": "e" * 64,
        "session_file_sha256": session_sha256,
        "earnings_input_sha256": "f" * 64,
        "corporate_actions_input_sha256": "0" * 64,
    }


def _bar(symbol: str, session_date: date, open_price: float, close: float) -> dict[str, object]:
    return {
        "session_date": session_date,
        "timestamp": datetime(
            session_date.year,
            session_date.month,
            session_date.day,
            21,
            tzinfo=UTC,
        ),
        "symbol": symbol,
        "open": open_price,
        "high": max(open_price, close),
        "low": min(open_price, close),
        "close": close,
        "volume": 1_000_000.0,
        "vwap": (open_price + close) / 2,
        "transactions": 10_000,
        "adjusted": True,
        "source": "polygon",
        "ingested_at": datetime(2025, 2, 1, tzinfo=UTC),
    }


def _assemble(
    tmp_path: Path,
    sources: dict[str, object],
    *,
    suffix: str = "",
) -> Phase4AssemblyResult:
    return Phase4HistoricalAssembler().assemble(
        walkforward_run_evidence=sources["run"],
        strategy_config=sources["strategy"],
        session_file=sources["sessions"],
        candidate_files=(sources["candidates"],),
        daily_bar_files=(sources["bars"],),
        initial_cash=100_000,
        output_dir=tmp_path / f"plans{suffix}",
        manifest_output=tmp_path / f"manifest{suffix}.json",
        aggregation_output=tmp_path / f"aggregation{suffix}.json",
    )


def test_assembler_chains_equity_and_writes_verified_manifest(tmp_path: Path) -> None:
    sources = _sources(tmp_path)

    result = _assemble(tmp_path, sources)

    assert result.session_count == 2
    assert result.trade_count == 2
    assert result.final_equity != 100_000
    manifest = Phase4AssemblyManifest.load(result.manifest_path)
    aggregation = Phase4AggregationSpec.model_validate_json(result.aggregation_path.read_bytes())
    plans = tuple(EventTradePlanner.load(path) for path in result.plan_files)
    assert manifest.sha256 == result.manifest_sha256
    assert len(aggregation.folds) == 1
    assert plans[0].portfolio.sizing_mode == "calibration"
    assert plans[0].source_predictions[0].information_cutoff_at == datetime(
        2025, 1, 3, 14, 10, tzinfo=UTC
    )
    assert plans[0].intents[0].entry_at > plans[0].source_predictions[0].information_cutoff_at
    assert plans[0].portfolio.history_count == 0
    assert plans[1].portfolio.history_count == 1
    assert plans[1].portfolio.equity != plans[0].portfolio.equity


def test_future_unrelated_bar_cannot_change_generated_plans(tmp_path: Path) -> None:
    sources = _sources(tmp_path)
    baseline = _assemble(tmp_path, sources, suffix="-baseline")
    future_bar_path = tmp_path / "bars-with-future.parquet"
    pq.write_table(  # type: ignore[no-untyped-call]
        pa.Table.from_pylist(
            [*sources["bar_rows"], _bar("ZZZ", date(2025, 1, 7), 10.0, 99.0)],
            schema=DAILY_BARS_SCHEMA,
        ),
        future_bar_path,
    )
    sources["bars"] = future_bar_path

    changed = _assemble(tmp_path, sources, suffix="-future")

    assert tuple(path.read_bytes() for path in baseline.plan_files) == tuple(
        path.read_bytes() for path in changed.plan_files
    )
    assert baseline.manifest_sha256 != changed.manifest_sha256


def test_assembler_rejects_missing_execution_bar(tmp_path: Path) -> None:
    sources = _sources(tmp_path)
    missing_path = tmp_path / "bars-missing.parquet"
    pq.write_table(  # type: ignore[no-untyped-call]
        pa.Table.from_pylist(sources["bar_rows"][:1], schema=DAILY_BARS_SCHEMA),
        missing_path,
    )
    sources["bars"] = missing_path

    with pytest.raises(ValueError, match="missing adjusted execution bar"):
        _assemble(tmp_path, sources)


def test_assembler_rejects_prediction_information_available_after_open(
    tmp_path: Path,
) -> None:
    sources = _sources(tmp_path)
    run = WalkForwardModelRun.load_evidence(sources["run"])
    first_fold = run.folds[0]
    late = replace(
        first_fold.predictions[0],
        information_cutoff_at=datetime(2025, 1, 3, 14, 30, tzinfo=UTC),
    )
    changed = replace(
        run,
        folds=(
            replace(
                first_fold,
                predictions=(late, *first_fold.predictions[1:]),
            ),
        ),
    )
    Path(sources["run"]).write_bytes(changed.evidence_json_bytes())

    with pytest.raises(ValueError, match="information cutoff"):
        _assemble(tmp_path, sources)


def test_manifest_rejects_source_mutation_after_assembly(tmp_path: Path) -> None:
    sources = _sources(tmp_path)
    result = _assemble(tmp_path, sources)
    Path(sources["bars"]).write_bytes(b"changed after assembly")

    with pytest.raises(ValueError, match="source hash differs"):
        Phase4AssemblyManifest.load(result.manifest_path)


def test_phase4_gate_rejects_rehashed_but_nonreproducible_plan(tmp_path: Path) -> None:
    sources = _sources(tmp_path)
    result = _assemble(tmp_path, sources)
    plan_path = result.plan_files[0]
    plan = json.loads(plan_path.read_bytes())
    plan["portfolio"]["sizing_mode"] = "kelly"
    plan_path.write_bytes(json.dumps(plan, sort_keys=True, separators=(",", ":")).encode())
    manifest = json.loads(result.manifest_path.read_bytes())
    manifest["plan_files"][0]["sha256"] = hashlib.sha256(plan_path.read_bytes()).hexdigest()
    result.manifest_path.write_bytes(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    )

    gate = CliRunner().invoke(
        app,
        [
            "evaluation",
            "phase4-gate",
            "--aggregation-spec",
            str(result.aggregation_path),
            "--output",
            str(tmp_path / "tampered-gate.json"),
            "--tearsheet-output",
            str(tmp_path / "tampered-tearsheet.html"),
            "--bootstrap-resamples",
            "10",
        ],
    )

    assert gate.exit_code == 2
    assert "plans do not reproduce" in gate.output


def test_assemble_phase4_cli_materializes_complete_fold_map(tmp_path: Path) -> None:
    sources = _sources(tmp_path)
    result = CliRunner().invoke(
        app,
        [
            "evaluation",
            "assemble-phase4",
            "--walkforward-run-evidence",
            str(sources["run"]),
            "--strategy-config",
            str(sources["strategy"]),
            "--session-file",
            str(sources["sessions"]),
            "--candidate-file",
            str(sources["candidates"]),
            "--daily-bar-file",
            str(sources["bars"]),
            "--initial-cash",
            "100000",
            "--output-dir",
            str(tmp_path / "cli-plans"),
            "--manifest-output",
            str(tmp_path / "cli-manifest.json"),
            "--aggregation-output",
            str(tmp_path / "cli-aggregation.json"),
        ],
    )

    assert result.exit_code == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["session_count"] == 2
    assert payload["trade_count"] == 2
    assert Path(payload["manifest_output"]).is_file()

    gate_path = tmp_path / "phase4-gate.json"
    gate = CliRunner().invoke(
        app,
        [
            "evaluation",
            "phase4-gate",
            "--aggregation-spec",
            str(tmp_path / "cli-aggregation.json"),
            "--output",
            str(gate_path),
            "--tearsheet-output",
            str(tmp_path / "phase4-tearsheet.html"),
            "--bootstrap-resamples",
            "10",
        ],
    )

    assert gate.exit_code == 0, gate.stderr
    assert json.loads(gate.stdout)["trade_count"] == 2
    verified = Phase4GateVerifier.verify(
        report_path=gate_path,
        aggregation_spec=tmp_path / "cli-aggregation.json",
    )
    assert verified.report.to_json_bytes() == gate_path.read_bytes()

    tampered = json.loads(gate_path.read_bytes())
    tampered["overall"]["final_net_equity"] += 1
    tampered_path = tmp_path / "tampered-phase4-gate.json"
    tampered_path.write_bytes(json.dumps(tampered, sort_keys=True, separators=(",", ":")).encode())
    with pytest.raises(ValueError, match="differs from reconstructed"):
        Phase4GateVerifier.verify(
            report_path=tampered_path,
            aggregation_spec=tmp_path / "cli-aggregation.json",
        )
