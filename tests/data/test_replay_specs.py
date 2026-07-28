"""Frozen orders and silver events materialize into causal self-contained replay specs."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime, timedelta
from typing import TYPE_CHECKING

from typer.testing import CliRunner

from quant_earning_edge.backtest import NbboReplayEvidence, NbboReplaySpec
from quant_earning_edge.cli import app
from quant_earning_edge.data import (
    LakehouseLayout,
    ReplayEvidenceIndex,
    ReplayManifestRunner,
    ReplayMaterializationManifest,
    ReplayMaterializationSpec,
    ReplaySpecMaterializer,
    SilverWriter,
)
from quant_earning_edge.data.clients import StockQuote, StockTrade

if TYPE_CHECKING:
    from pathlib import Path


def _inputs(tmp_path: Path) -> ReplayMaterializationSpec:
    opened = datetime(2026, 7, 28, 13, 30, tzinfo=UTC)
    closing = datetime(2026, 7, 28, 19, 55, tzinfo=UTC)
    quotes = tuple(
        StockQuote(
            symbol="AAA",
            timestamp=timestamp,
            sequence_number=index,
            bid_price=99.9 + index,
            ask_price=100.1 + index,
            bid_size=100,
            ask_size=100,
        )
        for index, timestamp in enumerate((opened, closing), start=1)
    )
    trades = tuple(
        StockTrade(
            symbol="AAA",
            timestamp=timestamp,
            sequence_number=index + 10,
            price=100 + index,
            size=100,
            exchange=11,
            trade_id=f"trade-{index}",
            correction=0,
        )
        for index, timestamp in enumerate((opened, closing), start=1)
    )
    writer = SilverWriter(LakehouseLayout(tmp_path / "lake"))
    quote_artifact = writer.write_stock_quotes(
        quotes,
        event_date=date(2026, 7, 28),
        ingested_at=closing + timedelta(hours=1),
    )
    trade_artifact = writer.write_stock_trades(
        trades,
        event_date=date(2026, 7, 28),
        ingested_at=closing + timedelta(hours=1),
    )
    decision = datetime(2026, 7, 27, 22, 0, tzinfo=UTC)
    return ReplayMaterializationSpec.model_validate(
        {
            "orders": [
                {
                    "order_id": "AAA-entry",
                    "ticker": "AAA",
                    "side": "buy",
                    "quantity": 100,
                    "decision_time": decision,
                    "submitted_at": opened,
                    "expires_at": opened + timedelta(minutes=1),
                    "average_daily_volume_shares": 1_000_000,
                },
                {
                    "order_id": "AAA-exit",
                    "ticker": "AAA",
                    "side": "sell",
                    "quantity": 100,
                    "decision_time": decision,
                    "submitted_at": closing,
                    "expires_at": closing + timedelta(minutes=1),
                    "average_daily_volume_shares": 1_000_000,
                },
            ],
            "decision_snapshots": [
                {
                    "ticker": "AAA",
                    "observed_at": decision,
                    "bid_price": 99.9,
                    "ask_price": 100.1,
                    "bid_size": 100,
                    "ask_size": 100,
                    "last_trade_price": 100,
                    "last_trade_at": decision - timedelta(seconds=1),
                }
            ],
            "event_sources": [
                {
                    "symbol": "AAA",
                    "quote_files": [quote_artifact.path],
                    "trade_files": [trade_artifact.path],
                }
            ],
        }
    )


def test_materializer_writes_one_filtered_spec_per_sorted_order(tmp_path: Path) -> None:
    spec = _inputs(tmp_path)
    output_dir = tmp_path / "specs"
    manifest_path = tmp_path / "manifest.json"

    manifest = ReplaySpecMaterializer().materialize(
        spec,
        output_dir=output_dir,
        manifest_output=manifest_path,
    )
    repeated = ReplaySpecMaterializer().materialize(
        spec,
        output_dir=output_dir,
        manifest_output=manifest_path,
    )

    assert repeated == manifest
    assert [item.order_id for item in manifest.artifacts] == ["AAA-entry", "AAA-exit"]
    for artifact in manifest.artifacts:
        replay_spec = NbboReplaySpec.model_validate_json(
            (output_dir / artifact.file_name).read_bytes()
        )
        assert replay_spec.sha256 == artifact.sha256
        assert len(replay_spec.quotes) == 1
        assert len(replay_spec.trades) == 1
    assert json.loads(manifest_path.read_bytes())["input_sha256"] == manifest.input_sha256


def test_materialization_cli_emits_paths_for_workflow_chaining(tmp_path: Path) -> None:
    spec = _inputs(tmp_path)
    spec_path = tmp_path / "materialize.json"
    spec_path.write_bytes(json.dumps(spec.model_dump(mode="json"), sort_keys=True).encode())
    output_dir = tmp_path / "replay-specs"
    manifest = tmp_path / "manifest.json"

    result = CliRunner().invoke(
        app,
        [
            "backtest",
            "materialize-replay-specs",
            "--materialization-spec",
            str(spec_path),
            "--output-dir",
            str(output_dir),
            "--manifest-output",
            str(manifest),
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["order_count"] == 2
    assert len(payload["replay_spec_paths"]) == 2
    assert payload["manifest_path"] == str(manifest.resolve())


def test_explicit_no_trade_materialization_needs_no_market_files(tmp_path: Path) -> None:
    manifest = ReplaySpecMaterializer().materialize(
        ReplayMaterializationSpec(),
        output_dir=tmp_path / "specs",
        manifest_output=tmp_path / "manifest.json",
    )

    assert manifest.artifacts == ()
    assert (tmp_path / "manifest.json").exists()


def test_manifest_runner_replays_every_verified_spec_idempotently(tmp_path: Path) -> None:
    spec = _inputs(tmp_path)
    spec_directory = tmp_path / "specs"
    manifest_path = tmp_path / "manifest.json"
    evidence_directory = tmp_path / "evidence"
    index_path = tmp_path / "indexes" / "replay-index.json"
    materialized = ReplaySpecMaterializer().materialize(
        spec,
        output_dir=spec_directory,
        manifest_output=manifest_path,
    )

    manifest = ReplayMaterializationManifest.load(manifest_path)
    index = ReplayManifestRunner().run(
        manifest,
        spec_directory=spec_directory,
        output_directory=evidence_directory,
        index_output=index_path,
    )
    repeated = ReplayManifestRunner().run(
        manifest,
        spec_directory=spec_directory,
        output_directory=evidence_directory,
        index_output=index_path,
    )

    assert manifest == materialized
    assert repeated == index
    assert ReplayEvidenceIndex.load(index_path) == index
    assert index.materialization_manifest_sha256 == manifest.sha256
    assert len(index.evidence_files) == 2
    for file_name, expected_hash in zip(
        index.evidence_files,
        index.evidence_sha256,
        strict=True,
    ):
        evidence = NbboReplayEvidence.load(evidence_directory / file_name)
        assert evidence.sha256 == expected_hash


def test_manifest_runner_rejects_tampered_spec(tmp_path: Path) -> None:
    spec_directory = tmp_path / "specs"
    manifest = ReplaySpecMaterializer().materialize(
        _inputs(tmp_path),
        output_dir=spec_directory,
        manifest_output=tmp_path / "manifest.json",
    )
    spec_path = spec_directory / manifest.artifacts[0].file_name
    spec_path.write_bytes(spec_path.read_bytes() + b"\n")

    try:
        ReplayManifestRunner().run(
            manifest,
            spec_directory=spec_directory,
            output_directory=tmp_path / "evidence",
            index_output=tmp_path / "index.json",
        )
    except ValueError as error:
        assert "not canonical" in str(error)
    else:
        raise AssertionError("tampered replay spec should fail closed")


def test_no_trade_manifest_runner_writes_empty_index(tmp_path: Path) -> None:
    spec_directory = tmp_path / "specs"
    manifest = ReplaySpecMaterializer().materialize(
        ReplayMaterializationSpec(),
        output_dir=spec_directory,
        manifest_output=tmp_path / "manifest.json",
    )

    index = ReplayManifestRunner().run(
        manifest,
        spec_directory=spec_directory,
        output_directory=tmp_path / "evidence",
        index_output=tmp_path / "index.json",
    )

    assert index.evidence_files == ()
    assert index.evidence_sha256 == ()
    assert ReplayEvidenceIndex.load(tmp_path / "index.json") == index


def test_replay_materialization_cli_emits_batch_evidence_paths(tmp_path: Path) -> None:
    spec_directory = tmp_path / "specs"
    manifest_path = tmp_path / "manifest.json"
    manifest = ReplaySpecMaterializer().materialize(
        _inputs(tmp_path),
        output_dir=spec_directory,
        manifest_output=manifest_path,
    )
    evidence_directory = tmp_path / "evidence"
    index_path = tmp_path / "index.json"

    result = CliRunner().invoke(
        app,
        [
            "backtest",
            "replay-materialization",
            "--manifest-file",
            str(manifest_path),
            "--spec-directory",
            str(spec_directory),
            "--output-directory",
            str(evidence_directory),
            "--index-output",
            str(index_path),
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["order_count"] == 2
    assert len(payload["replay_evidence_paths"]) == 2
    assert payload["materialization_manifest_sha256"] == manifest.sha256
    assert ReplayEvidenceIndex.load(index_path).sha256 == payload["index_sha256"]
