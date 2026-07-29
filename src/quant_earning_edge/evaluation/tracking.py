"""MLflow provenance for every research backtest entry point."""

from __future__ import annotations

import hashlib
import json
import os
from contextlib import suppress
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from quant_earning_edge.backtest import CostModelConfig

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


@dataclass(frozen=True)
class BacktestTrackingReference:
    """Non-secret identity returned by a completed MLflow write."""

    run_id: str
    experiment_id: str
    code_sha256: str
    source_manifest_sha256: str


def source_tree_sha256() -> str:
    """Hash every package Python source using stable relative paths."""
    package_root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for path in sorted(
        package_root.rglob("*.py"), key=lambda item: item.relative_to(package_root).as_posix()
    ):
        relative = path.relative_to(package_root).as_posix().encode()
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        content = path.read_bytes()
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def log_backtest_run(
    *,
    tracking_uri: str,
    artifact_location: str | None,
    experiment_name: str,
    run_kind: str,
    report_bytes: bytes,
    report_sha256: str,
    input_sha256: str,
    engine: str,
    trade_count: int,
    session_count: int,
    bootstrap_resamples: int,
    seed: int,
    source_files: Sequence[Path],
    artifact_files: Sequence[Path] = (),
    extra_parameters: Mapping[str, str | int | float | bool] | None = None,
) -> BacktestTrackingReference:
    """Persist hashes, parameters, seed, and report to MLflow or fail."""
    if not experiment_name.strip() or not run_kind.strip():
        raise ValueError("MLflow experiment and run kind must not be blank")
    if bootstrap_resamples < 1:
        raise ValueError("bootstrap resamples must be positive")
    code_sha256 = source_tree_sha256()
    source_manifest = _source_manifest(source_files)
    source_manifest_bytes = json.dumps(
        source_manifest,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    source_manifest_sha256 = hashlib.sha256(source_manifest_bytes).hexdigest()
    cost_parameters = asdict(CostModelConfig())
    parameters: dict[str, str | int | float | bool] = {
        "run_kind": run_kind,
        "input_sha256": input_sha256,
        "report_sha256": report_sha256,
        "code_sha256": code_sha256,
        "source_manifest_sha256": source_manifest_sha256,
        "engine": engine,
        "trade_count": trade_count,
        "session_count": session_count,
        "bootstrap_resamples": bootstrap_resamples,
        "seed": seed,
        **{f"cost_{key}": value for key, value in cost_parameters.items()},
        **(dict(extra_parameters) if extra_parameters is not None else {}),
    }
    prior_file_store_override = os.environ.get("MLFLOW_ALLOW_FILE_STORE")
    if tracking_uri.startswith("file:"):
        os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    client: Any | None = None
    run_id: str | None = None
    try:
        try:
            mlflow = _import_mlflow()
            client = mlflow.MlflowClient(
                tracking_uri=tracking_uri,
                registry_uri=tracking_uri,
            )
            experiment = client.get_experiment_by_name(experiment_name)
            experiment_id = (
                str(experiment.experiment_id)
                if experiment is not None
                else str(
                    client.create_experiment(
                        experiment_name,
                        artifact_location=artifact_location,
                    )
                )
            )
            active = client.create_run(
                experiment_id,
                tags={
                    "mlflow.runName": f"{run_kind}-{report_sha256[:12]}",
                    "qee.run_kind": run_kind,
                    "qee.code_sha256": code_sha256,
                    "qee.input_sha256": input_sha256,
                    "qee.report_sha256": report_sha256,
                },
            )
            run_id = str(active.info.run_id)
            for key, value in parameters.items():
                client.log_param(run_id, key, value)
            client.log_dict(run_id, json.loads(report_bytes), "performance-report.json")
            client.log_dict(run_id, source_manifest, "source-manifest.json")
            for artifact_file in artifact_files:
                client.log_artifact(
                    run_id,
                    str(artifact_file.resolve()),
                    artifact_path="supplemental",
                )
            client.set_terminated(run_id, status="FINISHED")
        except Exception as error:
            if client is not None and run_id is not None:
                with suppress(Exception):
                    client.set_terminated(run_id, status="FAILED")
            raise RuntimeError(
                f"MLflow backtest tracking failed ({type(error).__name__})"
            ) from error
    finally:
        if prior_file_store_override is None:
            os.environ.pop("MLFLOW_ALLOW_FILE_STORE", None)
        else:
            os.environ["MLFLOW_ALLOW_FILE_STORE"] = prior_file_store_override
    if run_id is None:
        raise RuntimeError("MLflow backtest tracking did not create a run")
    return BacktestTrackingReference(
        run_id=run_id,
        experiment_id=experiment_id,
        code_sha256=code_sha256,
        source_manifest_sha256=source_manifest_sha256,
    )


def default_tracking_uri(output: Path, *, environment: Mapping[str, str]) -> str:
    """Use configured MLflow, otherwise an output-local file store."""
    configured = environment.get("MLFLOW_TRACKING_URI", "").strip()
    if configured:
        return configured
    return (output.resolve().parent / ".mlflow").as_uri()


def default_artifact_location(
    output: Path,
    *,
    environment: Mapping[str, str],
) -> str | None:
    """Keep fallback artifacts beside fallback tracking; defer to remote servers."""
    if environment.get("MLFLOW_TRACKING_URI", "").strip():
        return None
    artifacts = output.resolve().parent / ".mlflow-artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    return artifacts.as_uri()


def _source_manifest(paths: Sequence[Path]) -> tuple[dict[str, str], ...]:
    if not paths:
        raise ValueError("MLflow backtest provenance requires source files")
    resolved = tuple(sorted({path.resolve() for path in paths}, key=str))
    return tuple(
        {
            "name": path.name,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in resolved
    )


def _import_mlflow() -> Any:
    try:
        import mlflow  # noqa: PLC0415 - expensive optional runtime boundary.
    except ImportError as error:
        raise RuntimeError("mlflow is required for reproducible backtest tracking") from error
    return mlflow
