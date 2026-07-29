"""Backtest MLflow provenance is complete and fail-closed."""

from __future__ import annotations

import json
from pathlib import Path

import mlflow
import pytest

from quant_earning_edge.evaluation import (
    default_artifact_location,
    default_tracking_uri,
    log_backtest_run,
)


def test_backtest_run_logs_hashes_parameters_and_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    source = tmp_path / "backtest.json"
    source.write_text('{"input":"fixed"}', encoding="utf-8")
    output = tmp_path / "report.json"
    tearsheet = tmp_path / "tearsheet.html"
    tearsheet.write_text("<html>fixed</html>", encoding="utf-8")
    report = b'{"net_sharpe":1.25}'
    environment: dict[str, str] = {}
    tracking_uri = default_tracking_uri(output, environment=environment)

    reference = log_backtest_run(
        tracking_uri=tracking_uri,
        artifact_location=default_artifact_location(
            output,
            environment=environment,
        ),
        experiment_name="qee-test",
        run_kind="ledger",
        report_bytes=report,
        report_sha256="a" * 64,
        input_sha256="b" * 64,
        engine="vectorbt-test",
        trade_count=7,
        session_count=300,
        bootstrap_resamples=10_000,
        seed=42,
        source_files=(source,),
        artifact_files=(tearsheet,),
    )

    run = mlflow.MlflowClient(tracking_uri).get_run(reference.run_id)
    assert run.data.params["code_sha256"] == reference.code_sha256
    assert run.data.params["input_sha256"] == "b" * 64
    assert run.data.params["seed"] == "42"
    assert run.data.params["bootstrap_resamples"] == "10000"
    assert run.data.params["cost_impact_coefficient_bps"] == "5.0"
    assert {
        item.path for item in mlflow.MlflowClient(tracking_uri).list_artifacts(reference.run_id)
    } == {
        "performance-report.json",
        "source-manifest.json",
        "supplemental",
    }
    supplemental = mlflow.MlflowClient(tracking_uri).list_artifacts(
        reference.run_id,
        "supplemental",
    )
    assert {item.path for item in supplemental} == {"supplemental/tearsheet.html"}
    downloaded = mlflow.MlflowClient(tracking_uri).download_artifacts(
        reference.run_id,
        "performance-report.json",
        tmp_path / "download",
    )
    assert json.loads(Path(downloaded).read_bytes())["net_sharpe"] == 1.25


def test_backtest_tracking_failure_is_not_silently_ignored(tmp_path: Path) -> None:
    source = tmp_path / "backtest.json"
    source.write_text("{}", encoding="utf-8")
    prior_tracking_uri = mlflow.get_tracking_uri()
    prior_registry_uri = mlflow.get_registry_uri()

    with pytest.raises(RuntimeError, match="MLflow backtest tracking failed"):
        log_backtest_run(
            tracking_uri="unsupported://tracking",
            artifact_location=None,
            experiment_name="qee-test",
            run_kind="ledger",
            report_bytes=b"{}",
            report_sha256="a" * 64,
            input_sha256="b" * 64,
            engine="vectorbt-test",
            trade_count=1,
            session_count=1,
            bootstrap_resamples=1,
            seed=1,
            source_files=(source,),
        )

    assert mlflow.get_tracking_uri() == prior_tracking_uri
    assert mlflow.get_registry_uri() == prior_registry_uri
