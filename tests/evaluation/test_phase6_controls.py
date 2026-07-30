"""Rolling Phase 6 control files are derived from durable daily evidence."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, date, datetime, timedelta
from functools import cache
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from typer.testing import CliRunner

from quant_earning_edge.backtest import (
    DecisionSnapshotSpec,
    NbboReplayEvidence,
    ReplayConfigSpec,
)
from quant_earning_edge.cli import app
from quant_earning_edge.data import (
    BronzeWriter,
    CalendarSourceCapture,
    EarningsSourceCapture,
    LakehouseLayout,
    ReplayEventSourceSpec,
    ReplayEvidenceIndex,
    ReplayManifestRunner,
    ReplayMaterializationSpec,
    ReplaySpecMaterializer,
    SessionFileStore,
    SilverWriter,
    SplitHistorySourceCapture,
)
from quant_earning_edge.data.clients import (
    AlpacaCalendarClient,
    EquityBar,
    FinnhubClient,
    MarketSession,
    MinuteBar,
    PolygonClient,
    StockQuote,
    TickerSnapshot,
)
from quant_earning_edge.evaluation import (
    Phase6AggregationSpec,
    Phase6CompletionFinalizer,
    Phase6DailyReportVerifier,
    ReplaySessionAggregator,
)
from quant_earning_edge.features import (
    DailyBarsFeatureLoader,
    EarningsFeatureLoader,
    FeatureEngine,
    FeatureSourceCapture,
    FeatureStore,
    PremarketFeatureLoader,
)
from quant_earning_edge.live import (
    BrokerOrder,
    PaperAccountSnapshot,
    PaperBatchSubmission,
    PaperOrderReconciler,
    PaperSubmission,
)
from quant_earning_edge.monitoring import (
    CircuitBreakerEvaluationSpec,
    CircuitBreakerEvaluator,
    CircuitBreakerObservation,
    CircuitBreakerObservationSpec,
    ProviderFreshnessEvidence,
    ReconciliationAgeEvaluator,
    encode_circuit_breaker_controls,
)
from quant_earning_edge.orchestration import (
    DailyWorkflowRunner,
    DailyWorkflowState,
    DailyWorkflowStore,
    WorkflowHealthReport,
    WorkflowStage,
    WorkflowTrigger,
)
from quant_earning_edge.signals import (
    DailyOrderPlanningSpec,
    LiveOrderPlanner,
    LivePlanningAssembler,
    LivePlanningSourceSpec,
    LiveSourceCaptureAssembler,
    ProductionModelArtifact,
    ProductionModelTrainer,
    load_strategy_config,
    strategy_file_sha256,
)
from quant_earning_edge.signals.production_source import (
    ProductionModelSourceCapture,
    ProductionModelSourceManifest,
)
from quant_earning_edge.universe import (
    CandidateObservation,
    EventCandidateJob,
    EventSourceCapture,
    UniverseBuilder,
    UniverseConfig,
    UniverseSnapshotWriter,
    UniverseSourceCapture,
    sector_from_sic_code,
)

_MODEL_TEMP = TemporaryDirectory(prefix="qee-phase6-model-")


@cache
def _production_model() -> ProductionModelArtifact:
    dataset = Path(_MODEL_TEMP.name) / "training.parquet"
    first = date(2025, 1, 2)
    rows = [
        {
            "asof_date": first + timedelta(days=index),
            "horizon_end_date": first + timedelta(days=index + 2),
            "return_1d": 1.0 if index % 2 == 0 else -1.0,
            "forward_1d_close": 0.01 if index % 2 == 0 else -0.01,
        }
        for index in range(70)
    ]
    pq.write_table(pa.Table.from_pylist(rows), dataset)  # type: ignore[no-untyped-call]
    return ProductionModelTrainer(
        feature_names=("return_1d",),
        early_stopping_rounds=10,
    ).run(
        dataset_files=(dataset,),
        training_cutoff=date(2025, 3, 20),
        phase4_gate_sha256="f" * 64,
    )


@pytest.fixture(autouse=True)
def _use_fixture_model_reconstruction(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        ProductionModelSourceCapture,
        "reproduce",
        staticmethod(lambda manifest: _production_model()),
    )


def _write_model_source_fixture(
    daily: Path,
    *,
    model_evidence: Path,
    model_file: Path,
) -> ProductionModelSourceManifest:
    def entry(path: Path) -> dict[str, str]:
        resolved = path.resolve()
        return {
            "path": resolved.as_posix(),
            "sha256": hashlib.sha256(resolved.read_bytes()).hexdigest(),
        }

    root = daily / "production-model-lineage"
    root.mkdir(parents=True, exist_ok=True)
    sources = []
    for name in ("gate", "aggregation", "assembly", "split", "study"):
        path = root / f"{name}.json"
        path.write_text(name, encoding="utf-8")
        sources.append(path)
    strategy = Path("configs/strategies/earnings_v1.yaml").resolve()
    dataset = Path(_MODEL_TEMP.name) / "training.parquet"
    raw = {
        "schema_version": 1,
        "model_evidence": entry(model_evidence),
        "model_file": entry(model_file),
        "dataset_files": [entry(dataset)],
        "phase4_gate": entry(sources[0]),
        "phase4_aggregation": entry(sources[1]),
        "phase4_source_files": sorted(
            (entry(sources[2]), entry(strategy)),
            key=lambda item: item["path"],
        ),
        "split_plan": entry(sources[3]),
        "hyperparameter_study": entry(sources[4]),
    }
    encoded = json.dumps(raw, sort_keys=True, separators=(",", ":")).encode()
    path = model_evidence.parent / "production-source-fixture.json"
    path.write_bytes(encoded)
    return ProductionModelSourceManifest.load(path)


def _write_scored_planning(
    daily: Path,
    *,
    source: LivePlanningSourceSpec,
    feature_files: tuple[Path, ...] = (),
) -> tuple[DailyOrderPlanningSpec, tuple[Path, ...], Path, tuple[Path, ...]]:
    model = _production_model()
    model_path, model_evidence_path = ProductionModelTrainer.write(
        model,
        daily / "production-model",
    )
    model_source = _write_model_source_fixture(
        daily,
        model_evidence=model_evidence_path,
        model_file=model_path,
    )
    source_path = daily / "live-source.json"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_bytes(source.canonical_bytes)
    scored = LivePlanningAssembler().assemble(
        source=source,
        model=model,
        feature_files=feature_files,
    )
    planning_path = daily / "daily-order-planning.json"
    scored_path = daily / "scored-live-planning-evidence.json"
    LivePlanningAssembler.write(
        scored,
        planning_output=planning_path,
        evidence_output=scored_path,
    )
    return (
        scored.planning,
        (source_path, model_evidence_path, model_path, *feature_files),
        scored_path,
        tuple(
            path
            for path in model_source.lineage_paths
            if path
            not in {
                model_evidence_path.resolve(),
                model_path.resolve(),
                Path("configs/strategies/earnings_v1.yaml").resolve(),
            }
        ),
    )


def _write_live_source_capture(  # noqa: PLR0915 - complete source fixture.
    daily: Path,
    *,
    trade_date: date,
    captured_at: datetime,
    snapshot: DecisionSnapshotSpec | None,
) -> tuple[LivePlanningSourceSpec, tuple[Path, ...], Path]:
    prior_date = trade_date - timedelta(days=1)
    candidate_root = daily / "candidate-lake"
    candidate_layout = LakehouseLayout(candidate_root)
    bronze = BronzeWriter(candidate_layout)
    calendar_raw = [
        {"date": prior_date.isoformat(), "open": "09:30", "close": "16:00"},
        {"date": trade_date.isoformat(), "open": "09:30", "close": "16:00"},
    ]
    calendar_observation = bronze.write_json(
        calendar_raw,
        source="alpaca",
        dataset="market-calendar",
        event_date=prior_date,
        received_at=captured_at,
    )
    calendar = SessionFileStore(candidate_layout).write(
        AlpacaCalendarClient.sessions_from_payload(
            calendar_raw,
            start_date=prior_date,
            end_date=trade_date,
        )
    )
    calendar_source = CalendarSourceCapture(candidate_layout).write(
        start_date=prior_date,
        end_date=trade_date,
        session_file=calendar,
        provider_observations=(calendar_observation,),
    )
    universe_config = UniverseConfig(
        min_price=5.0,
        min_market_cap_usd=100_000_000.0,
        min_avg_daily_volume=500_000.0,
        allowed_exchanges=frozenset({"XNAS"}),
        allowed_security_types=frozenset({"CS"}),
        exclude_halts=True,
    )
    universe_snapshot = UniverseSnapshotWriter(candidate_layout).write(
        UniverseBuilder(universe_config).build(
            trade_date=trade_date,
            asof_date=prior_date,
            generated_at=captured_at,
            candidates=(
                CandidateObservation(
                    symbol="AAA",
                    asof_date=prior_date,
                    close=100.0,
                    avg_daily_volume=1_000_000.0,
                    market_cap_usd=1_000_000_000.0,
                    primary_exchange="XNAS",
                    security_type="CS",
                    active=True,
                    halted=False,
                    sector=sector_from_sic_code("3571"),
                    list_date=date(2000, 1, 1),
                    delisted_date=None,
                ),
            ),
        )
    )
    config_path = candidate_root / "universe.yaml"
    config_path.write_text(
        "adv_sessions: 1\neligibility:\n"
        "  min_price: 5\n  min_market_cap_usd: 100000000\n"
        "  min_avg_daily_volume: 500000\n  allowed_exchanges: [XNAS]\n"
        "  allowed_security_types: [CS]\n  exclude_halts: true\n",
        encoding="utf-8",
    )
    halt_path = candidate_root / "halts.json"
    halt_path.write_text(
        json.dumps(
            {
                "asof_date": prior_date.isoformat(),
                "captured_at": captured_at.isoformat(),
                "symbols": [],
            }
        ),
        encoding="utf-8",
    )
    reference_raw = {
        "status": "OK",
        "results": [
            {
                "ticker": "AAA",
                "name": "AAA Corp",
                "active": True,
                "locale": "us",
                "market": "stocks",
                "primary_exchange": "XNAS",
                "type": "CS",
            }
        ],
    }
    details_raw = {
        "status": "OK",
        "results": {
            "ticker": "AAA",
            "name": "AAA Corp",
            "active": True,
            "locale": "us",
            "market": "stocks",
            "primary_exchange": "XNAS",
            "type": "CS",
            "market_cap": 1_000_000_000,
            "sic_code": "3571",
            "list_date": "2000-01-01",
        },
    }
    bars_raw = {
        "ticker": "AAA",
        "adjusted": False,
        "status": "OK",
        "results": [
            {
                "t": int(
                    datetime(
                        prior_date.year, prior_date.month, prior_date.day, 20, tzinfo=UTC
                    ).timestamp()
                    * 1000
                ),
                "o": 99,
                "h": 101,
                "l": 98,
                "c": 100,
                "v": 1_000_000,
                "vw": 100,
                "n": 1000,
            }
        ],
    }
    provider_observations = tuple(
        bronze.write_json(
            payload,
            source="polygon",
            dataset=dataset,
            event_date=prior_date,
            received_at=captured_at,
        )
        for dataset, payload in (
            ("ticker-reference", reference_raw),
            ("ticker-details", details_raw),
            ("daily-aggregate-bars", bars_raw),
        )
    )
    writer = SilverWriter(candidate_layout)
    split_observation = bronze.write_json(
        {"status": "OK", "results": []},
        source="polygon",
        dataset="stock-splits",
        event_date=prior_date,
        received_at=captured_at,
    )
    split_source = SplitHistorySourceCapture(candidate_layout).write(
        plan_id="a" * 64,
        start_date=prior_date,
        end_date=prior_date,
        ingested_at=captured_at,
        split_files=(),
        provider_observations=(split_observation,),
    )
    universe_source = UniverseSourceCapture(candidate_layout).write(
        trade_date=trade_date,
        asof_date=prior_date,
        lookback_start=prior_date,
        decision_at=captured_at,
        adv_sessions=1,
        snapshot=universe_snapshot,
        universe_config=config_path,
        halt_snapshot=halt_path,
        split_source_manifest=split_source.path,
        provider_observations=provider_observations,
    )
    universe_path = universe_snapshot.path
    earnings_raw = {
        "earningsCalendar": [
            {
                "date": prior_date.isoformat(),
                "symbol": "AAA" if snapshot is not None else "ZZZ",
                "hour": "amc",
                "year": trade_date.year,
                "quarter": 3,
                "epsEstimate": 1.0,
                "revenueEstimate": 100.0,
            }
        ]
    }
    splits_raw = {
        "status": "OK",
        "results": [
            {
                "id": "irrelevant-split",
                "ticker": "ZZZ",
                "execution_date": trade_date.isoformat(),
                "adjustment_type": "forward_split",
                "split_from": 1,
                "split_to": 2,
            }
        ],
    }
    dividends_raw = {
        "status": "OK",
        "results": [
            {
                "id": "irrelevant-dividend",
                "ticker": "ZZZ",
                "ex_dividend_date": trade_date.isoformat(),
                "distribution_type": "recurring",
                "cash_amount": 0.25,
                "currency": "USD",
                "frequency": 4,
            }
        ],
    }
    earnings_observations = (
        bronze.write_json(
            earnings_raw,
            source="finnhub",
            dataset="earnings-calendar",
            event_date=prior_date,
            received_at=captured_at,
        ),
    )
    action_observations = tuple(
        bronze.write_json(
            payload,
            source="polygon",
            dataset=dataset,
            event_date=trade_date,
            received_at=captured_at,
        )
        for dataset, payload in (
            ("stock-splits", splits_raw),
            ("cash-dividends", dividends_raw),
        )
    )
    earnings = writer.write_earnings(
        FinnhubClient.earnings_calendar_from_payload(earnings_raw),
        ingested_at=captured_at,
    )
    splits = writer.write_splits(
        PolygonClient.stock_splits_from_payloads(
            (splits_raw,),
            start_date=prior_date,
            end_date=trade_date,
        ),
        ingested_at=captured_at,
    )
    dividends = writer.write_dividends(
        PolygonClient.cash_dividends_from_payloads(
            (dividends_raw,),
            start_date=prior_date,
            end_date=trade_date,
        ),
        ingested_at=captured_at,
    )
    event_source = EventSourceCapture(candidate_layout).write(
        start_date=prior_date,
        end_date=trade_date,
        ingested_at=captured_at,
        earnings_files=earnings,
        split_files=splits,
        dividend_files=dividends,
        earnings_observations=earnings_observations,
        corporate_action_observations=action_observations,
    )
    EarningsSourceCapture(candidate_layout).write(
        start_date=prior_date,
        end_date=trade_date,
        ingested_at=captured_at,
        silver_files=earnings,
        provider_observations=earnings_observations,
    )
    candidate = EventCandidateJob(candidate_layout).run(
        trade_date=trade_date,
        decision_at=captured_at,
        universe_snapshot=universe_path,
        session_file=calendar.path,
        earnings_files=tuple(item.path for item in earnings),
        split_files=tuple(item.path for item in splits),
        dividend_files=tuple(item.path for item in dividends),
        universe_source_manifest=universe_source.path,
        event_source_manifest=event_source.path,
        calendar_source_manifest=calendar_source.path,
    )

    def write_raw(path: Path, payload: object) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(
            json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        )
        return path

    account_raw = {
        "equity": "100000",
        "buying_power": "200000",
        "status": "ACTIVE",
        "trading_blocked": False,
    }
    account_path = write_raw(daily / "raw-account.json", account_raw)
    account = PaperAccountSnapshot.from_payload(
        account_raw,
        captured_at=captured_at,
    )
    snapshots: tuple[TickerSnapshot, ...] = ()
    snapshot_paths: tuple[Path, ...] = ()
    if snapshot is not None:
        quote_nanoseconds = int(snapshot.observed_at.timestamp()) * 1_000_000_000
        trade_nanoseconds = int(snapshot.last_trade_at.timestamp()) * 1_000_000_000
        snapshot_raw = {
            "status": "OK",
            "request_id": "snapshot-request",
            "ticker": {
                "ticker": "AAA",
                "lastQuote": {
                    "P": snapshot.ask_price,
                    "S": snapshot.ask_size,
                    "p": snapshot.bid_price,
                    "s": snapshot.bid_size,
                    "t": quote_nanoseconds,
                },
                "lastTrade": {
                    "p": snapshot.last_trade_price,
                    "t": trade_nanoseconds,
                },
            },
        }
        snapshot_path = write_raw(daily / "raw-polygon-snapshot.json", snapshot_raw)
        snapshots = (
            PolygonClient.ticker_snapshot_from_payload(
                snapshot_raw,
                symbol="AAA",
                captured_at=captured_at,
            ),
        )
        snapshot_paths = (snapshot_path,)
    capture = LiveSourceCaptureAssembler().assemble(
        trade_date=trade_date,
        captured_at=captured_at,
        candidate_file=candidate.path,
        session_file=calendar.path,
        account=account,
        initial_cash=100_000,
        snapshots=snapshots,
        prior_replay_files=(),
    )
    source_path = daily / "live-source.json"
    evidence_path = daily / "live-source-evidence.json"
    LiveSourceCaptureAssembler.write(
        capture,
        source_output=source_path,
        evidence_output=evidence_path,
    )
    return (
        capture.source,
        (
            candidate.path,
            calendar.path,
            account_path,
            *snapshot_paths,
            evidence_path,
            candidate.manifest_path,
            universe_source.path,
            *universe_source.source_paths(data_lake_root=candidate_root),
            event_source.path,
            *event_source.provider_paths(data_lake_root=candidate_root),
            calendar_source.path,
            *calendar_source.provider_paths(data_lake_root=candidate_root),
            universe_path,
            *(item.path for item in earnings),
            *(item.path for item in splits),
            *(item.path for item in dividends),
        ),
        source_path,
    )


def _write_breaker_auxiliary_evidence(
    daily: Path,
    *,
    observation: CircuitBreakerObservation,
) -> tuple[Path, Path, Path, Path, Path, Path]:
    assert observation.polygon_data_observed_at is not None
    assert observation.alpaca_data_observed_at is not None
    polygon_observed = observation.polygon_data_observed_at
    polygon_nanoseconds = (
        int(polygon_observed.timestamp()) * 1_000_000_000 + polygon_observed.microsecond * 1_000
    )
    polygon_payload = {
        "status": "OK",
        "request_id": None,
        "ticker": {
            "ticker": "SPY",
            "updated": polygon_nanoseconds,
        },
    }
    alpaca_payload = {"timestamp": observation.alpaca_data_observed_at.isoformat()}

    def write_payload(path: Path, payload: object) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(
            json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        )
        return path

    polygon_path = write_payload(
        daily / "source=polygon" / "dataset=freshness-snapshot" / "polygon.json",
        polygon_payload,
    )
    alpaca_path = write_payload(
        daily / "source=alpaca" / "dataset=market-clock" / "alpaca.json",
        alpaca_payload,
    )
    freshness = ProviderFreshnessEvidence.from_payloads(
        polygon_payload=polygon_payload,
        alpaca_payload=alpaca_payload,
        evaluated_at=observation.evaluated_at,
        polygon_symbol="SPY",
        alpaca_request_id=None,
    )
    freshness_path = daily / f"provider-freshness-{freshness.sha256}.json"
    freshness.write(freshness_path)
    prior_date = observation.session_date - timedelta(days=1)
    calendar = SessionFileStore(LakehouseLayout(daily / "control-calendar")).write(
        (
            MarketSession(
                session_date=prior_date,
                open_at=datetime(
                    prior_date.year,
                    prior_date.month,
                    prior_date.day,
                    13,
                    30,
                    tzinfo=UTC,
                ),
                close_at=datetime(
                    prior_date.year,
                    prior_date.month,
                    prior_date.day,
                    20,
                    tzinfo=UTC,
                ),
            ),
            MarketSession(
                session_date=observation.session_date,
                open_at=datetime(
                    observation.session_date.year,
                    observation.session_date.month,
                    observation.session_date.day,
                    13,
                    30,
                    tzinfo=UTC,
                ),
                close_at=datetime(
                    observation.session_date.year,
                    observation.session_date.month,
                    observation.session_date.day,
                    20,
                    tzinfo=UTC,
                ),
            ),
        )
    )
    source_report = PaperOrderReconciler().evaluate(
        evidence=(),
        broker_orders=(),
        session_date=prior_date,
        evaluated_at=observation.evaluated_at - timedelta(minutes=1),
    )
    source_report_path = (
        daily / "control-sources" / f"paper-reconciliation-{source_report.sha256}.json"
    )
    source_report.write(source_report_path)
    age = ReconciliationAgeEvaluator().evaluate(
        calendar=calendar,
        reports=(source_report,),
        control_date=observation.session_date,
        evaluated_at=observation.evaluated_at,
    )
    age_path = daily / f"reconciliation-age-{age.sha256}.json"
    age.write(age_path)
    return (
        freshness_path,
        age_path,
        polygon_path,
        alpaca_path,
        calendar.path,
        source_report_path,
    )


def _no_trade_replay_sources(
    tmp_path: Path,
    *,
    session_date: date,
    report_initial_cash: float = 100_000,
) -> dict[str, Any]:
    strategy = Path("configs/strategies/earnings_v1.yaml").resolve()
    daily = tmp_path / "artifacts" / f"trade_date={session_date.isoformat()}"
    frozen_path = daily / "frozen-daily-orders.json"
    planning_source, live_capture_paths, source_path = _write_live_source_capture(
        daily,
        trade_date=session_date,
        captured_at=datetime(
            session_date.year,
            session_date.month,
            session_date.day,
            12,
            tzinfo=UTC,
        ),
        snapshot=None,
    )
    (
        candidate_path,
        live_calendar_path,
        account_path,
        live_evidence_path,
        candidate_manifest_path,
        universe_source_manifest_path,
        universe_config_path,
        universe_halt_path,
        universe_split_source_manifest_path,
        universe_bar_raw_path,
        universe_details_raw_path,
        universe_reference_raw_path,
        universe_split_raw_path,
        event_source_manifest_path,
        event_earnings_raw_path,
        event_split_raw_path,
        event_dividend_raw_path,
        calendar_source_manifest_path,
        calendar_raw_path,
        universe_path,
        earnings_path,
        split_path,
        dividend_path,
    ) = live_capture_paths
    planning, planning_sources, planning_evidence_path, model_lineage_paths = (
        _write_scored_planning(
            daily,
            source=planning_source,
        )
    )
    _, model_evidence_path, model_path = planning_sources
    planning_path = daily / "daily-order-planning.json"
    frozen = LiveOrderPlanner(
        load_strategy_config(strategy),
        strategy_sha256=strategy_file_sha256(strategy),
    ).plan(planning)
    frozen.write(frozen_path)
    breaker_evaluated_at = datetime.now(UTC)
    breaker_observation = CircuitBreakerObservation(
        session_date=session_date,
        evaluated_at=breaker_evaluated_at,
        replay_notional=0,
        replay_net_pnl=0,
        replay_fill_rate=None,
        polygon_data_observed_at=breaker_evaluated_at,
        alpaca_data_observed_at=breaker_evaluated_at,
        reconciliation_break_age_sessions=None,
        replay_source_date=session_date,
    )
    breaker_spec = CircuitBreakerEvaluationSpec(
        observations=(CircuitBreakerObservationSpec.model_validate(breaker_observation.__dict__),)
    )
    breaker = CircuitBreakerEvaluator().evaluate((breaker_observation,))
    breaker_spec_path = daily / "breaker-controls.json"
    breaker_spec_path.write_bytes(encode_circuit_breaker_controls(breaker_spec))
    (
        freshness_path,
        age_path,
        polygon_path,
        alpaca_path,
        calendar_path,
        age_report_path,
    ) = _write_breaker_auxiliary_evidence(
        daily,
        observation=breaker_observation,
    )
    breaker_path = daily / "breaker-decision.json"
    breaker.write(breaker_path)
    submission_path = daily / "paper-batch-submission.json"
    PaperBatchSubmission(
        schema_version=1,
        session_date=session_date,
        breaker_decision_sha256=breaker.sha256,
        submissions=(),
    ).write(submission_path)
    manifest_path = daily / "replay-materialization-manifest.json"
    loaded_strategy = load_strategy_config(strategy)
    manifest = ReplaySpecMaterializer().materialize(
        ReplayMaterializationSpec(
            orders=frozen.intended_orders,
            decision_snapshots=frozen.decision_snapshots,
            event_sources=(),
            config=ReplayConfigSpec(
                market_impact_bps_coefficient=(loaded_strategy.costs.market_impact_coef_bps)
            ),
        ),
        output_dir=daily / "replay-specs",
        manifest_output=manifest_path,
    )
    index_path = daily / "replay-evidence" / "index.json"
    ReplayEvidenceIndex(
        schema_version=1,
        materialization_manifest_sha256=manifest.sha256,
        evidence_files=(),
        evidence_sha256=(),
    ).write(index_path)
    reconciliation = PaperOrderReconciler().evaluate(
        evidence=(),
        broker_orders=(),
        session_date=session_date,
        evaluated_at=datetime.now(UTC),
    )
    reconciliation_path = daily / f"paper-reconciliation-{reconciliation.sha256}.json"
    reconciliation.write(reconciliation_path)
    report_path = daily / "replay-session.json"
    ReplaySessionAggregator().evaluate_frozen_long_orders(
        evidence=(),
        intended_orders=tuple(item.to_domain() for item in frozen.intended_orders),
        session_date=session_date,
        initial_cash=report_initial_cash,
        commission_bps_per_side=loaded_strategy.costs.commission_bps_per_side,
    ).write(report_path)
    return {
        "strategy": strategy,
        "planning": planning_path,
        "planning_source": source_path,
        "candidate_file": candidate_path,
        "candidate_manifest": candidate_manifest_path,
        "universe_source_manifest": universe_source_manifest_path,
        "universe_config": universe_config_path,
        "universe_halts": universe_halt_path,
        "universe_split_source_manifest": universe_split_source_manifest_path,
        "universe_bar_raw": universe_bar_raw_path,
        "universe_details_raw": universe_details_raw_path,
        "universe_reference_raw": universe_reference_raw_path,
        "universe_split_raw": universe_split_raw_path,
        "event_source_manifest": event_source_manifest_path,
        "event_earnings_raw": event_earnings_raw_path,
        "event_split_raw": event_split_raw_path,
        "event_dividend_raw": event_dividend_raw_path,
        "calendar_source_manifest": calendar_source_manifest_path,
        "calendar_raw": calendar_raw_path,
        "candidate_universe": universe_path,
        "candidate_earnings": earnings_path,
        "candidate_splits": split_path,
        "candidate_dividends": dividend_path,
        "live_calendar": live_calendar_path,
        "account_observation": account_path,
        "live_source_evidence": live_evidence_path,
        "model_evidence": model_evidence_path,
        "model_file": model_path,
        "model_lineage": model_lineage_paths,
        "planning_evidence": planning_evidence_path,
        "frozen": frozen_path,
        "breaker": breaker_path,
        "breaker_spec": breaker_spec_path,
        "freshness": freshness_path,
        "age": age_path,
        "polygon_freshness": polygon_path,
        "alpaca_freshness": alpaca_path,
        "reconciliation_calendar": calendar_path,
        "reconciliation_age_report": age_report_path,
        "submission": submission_path,
        "manifest": manifest_path,
        "index": index_path,
        "reconciliation": reconciliation_path,
        "report": report_path,
    }


def _complete_source_workflow(
    *,
    store: DailyWorkflowStore,
    tmp_path: Path,
    session_date: date,
    sources: dict[str, Any],
    reconciliation_succeeds: bool = True,
) -> DailyWorkflowState:
    def handler(  # noqa: PLR0911 - explicit workflow-stage fixture.
        _: DailyWorkflowState,
        stage: WorkflowStage,
    ) -> tuple[Path, ...]:
        if stage is WorkflowStage.FREEZE_INPUTS:
            return (
                sources["strategy"],
                sources["planning_source"],
                sources["candidate_file"],
                sources["candidate_manifest"],
                sources["universe_source_manifest"],
                sources["universe_config"],
                sources["universe_halts"],
                sources["universe_split_source_manifest"],
                sources["universe_bar_raw"],
                sources["universe_details_raw"],
                sources["universe_reference_raw"],
                sources["universe_split_raw"],
                sources["event_source_manifest"],
                sources["event_earnings_raw"],
                sources["event_split_raw"],
                sources["event_dividend_raw"],
                sources["calendar_source_manifest"],
                sources["calendar_raw"],
                sources["candidate_universe"],
                sources["candidate_earnings"],
                sources["candidate_splits"],
                sources["candidate_dividends"],
                sources["live_calendar"],
                sources["account_observation"],
                sources["live_source_evidence"],
                sources["model_evidence"],
                sources["model_file"],
                *sources["model_lineage"],
            )
        if stage is WorkflowStage.GENERATE_ORDER_PLAN:
            return (
                sources["planning"],
                sources["planning_evidence"],
                sources["frozen"],
            )
        if stage is WorkflowStage.EVALUATE_BREAKERS:
            return (
                sources["freshness"],
                sources["age"],
                sources["polygon_freshness"],
                sources["alpaca_freshness"],
                sources["reconciliation_calendar"],
                sources["reconciliation_age_report"],
                sources["breaker_spec"],
                sources["breaker"],
            )
        if stage is WorkflowStage.SUBMIT_PAPER_ORDERS:
            return (sources["submission"],)
        if stage is WorkflowStage.REPLAY_ORDERS:
            return (sources["manifest"], sources["index"], sources["report"])
        if stage is WorkflowStage.RECONCILE_SESSION:
            if not reconciliation_succeeds:
                raise RuntimeError("paper reconciliation is still unresolved")
            return (sources["reconciliation"],)
        output = tmp_path / "stage-artifacts" / f"{stage.value}.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("{}", encoding="utf-8")
        return (output,)

    return DailyWorkflowRunner(
        store=store,
        handlers={stage: handler for stage in WorkflowStage},
        worker_id="worker",
        clock=lambda: datetime.now(UTC),
        trigger=WorkflowTrigger.SCHEDULED,
    ).run_until_idle(trade_date=session_date)


def _trade_source_workflow(  # noqa: PLR0915 - complete source-bound trade fixture.
    *,
    store: DailyWorkflowStore,
    tmp_path: Path,
    session_date: date,
    capture_broker_observations: bool = True,
    capture_submission_observations: bool = True,
    capture_breaker_spec: bool = True,
    capture_breaker_auxiliary: bool = True,
    capture_freshness_payloads: bool = True,
    capture_reconciliation_age_sources: bool = True,
    capture_planning_input: bool = True,
    capture_scored_planning_evidence: bool = True,
    capture_live_source_evidence: bool = True,
    capture_live_provider_inputs: bool = True,
    capture_candidate_lineage: bool = True,
    capture_universe_lineage: bool = True,
    capture_event_lineage: bool = True,
    capture_calendar_lineage: bool = True,
    capture_feature_lineage: bool = True,
    capture_model_lineage: bool = True,
) -> Path:
    strategy_path = Path("configs/strategies/earnings_v1.yaml").resolve()
    strategy = load_strategy_config(strategy_path)
    decision_at = datetime(2026, 7, 28, 12, tzinfo=UTC)
    entry_at = datetime(2026, 7, 28, 13, 30, tzinfo=UTC)
    exit_at = datetime(2026, 7, 28, 19, 55, tzinfo=UTC)
    snapshot = DecisionSnapshotSpec(
        ticker="AAA",
        observed_at=decision_at,
        bid_price=99.9,
        ask_price=100.1,
        bid_size=100,
        ask_size=100,
        last_trade_price=100,
        last_trade_at=decision_at - timedelta(seconds=1),
    )
    daily = tmp_path / "artifacts" / f"trade_date={session_date.isoformat()}"
    planning_source, live_capture_paths, source_path = _write_live_source_capture(
        daily,
        trade_date=session_date,
        captured_at=decision_at,
        snapshot=snapshot,
    )
    (
        candidate_path,
        live_calendar_path,
        account_path,
        live_snapshot_path,
        live_evidence_path,
        candidate_manifest_path,
        universe_source_manifest_path,
        universe_config_path,
        universe_halt_path,
        universe_split_source_manifest_path,
        universe_bar_raw_path,
        universe_details_raw_path,
        universe_reference_raw_path,
        universe_split_raw_path,
        event_source_manifest_path,
        event_earnings_raw_path,
        event_split_raw_path,
        event_dividend_raw_path,
        calendar_source_manifest_path,
        calendar_raw_path,
        universe_path,
        earnings_path,
        split_path,
        dividend_path,
    ) = live_capture_paths
    feature_layout = LakehouseLayout(daily / "candidate-lake")
    feature_writer = SilverWriter(feature_layout)
    feature_bar_models = tuple(
        EquityBar(
            symbol="AAA",
            timestamp=datetime(
                day.year,
                day.month,
                day.day,
                20,
                tzinfo=UTC,
            ),
            open=close - 1,
            high=close + 1,
            low=close - 2,
            close=close,
            volume=1_000_000,
            vwap=close - 0.5,
            adjusted=True,
        )
        for day, close in (
            (planning_source.feature_asof_date - timedelta(days=1), 99.0),
            (planning_source.feature_asof_date, 100.0),
        )
    )
    feature_bars = feature_writer.write_daily_bars(
        feature_bar_models,
        ingested_at=decision_at,
    )
    feature_minute_model = MinuteBar(
        symbol="AAA",
        timestamp=decision_at - timedelta(minutes=2),
        open=100,
        high=101,
        low=99,
        close=100.5,
        volume=10_000,
        vwap=100.25,
        adjusted=True,
    )
    feature_minute = feature_writer.write_minute_bars(
        (feature_minute_model,),
        event_date=session_date,
        ingested_at=decision_at,
    )
    feature_bronze = BronzeWriter(feature_layout)
    feature_provider_observations = (
        feature_bronze.write_json(
            {
                "ticker": "AAA",
                "adjusted": True,
                "status": "OK",
                "results": [
                    {
                        "t": int(item.timestamp.timestamp() * 1000),
                        "o": item.open,
                        "h": item.high,
                        "l": item.low,
                        "c": item.close,
                        "v": item.volume,
                        "vw": item.vwap,
                    }
                    for item in feature_bar_models
                ],
            },
            source="polygon",
            dataset="daily-aggregate-bars",
            event_date=planning_source.feature_asof_date,
            received_at=decision_at,
        ),
        feature_bronze.write_json(
            {
                "ticker": "AAA",
                "adjusted": True,
                "status": "OK",
                "results": [
                    {
                        "t": int(feature_minute_model.timestamp.timestamp() * 1000),
                        "o": feature_minute_model.open,
                        "h": feature_minute_model.high,
                        "l": feature_minute_model.low,
                        "c": feature_minute_model.close,
                        "v": feature_minute_model.volume,
                        "vw": feature_minute_model.vwap,
                    }
                ],
            },
            source="polygon",
            dataset="minute-aggregate-bars",
            event_date=session_date,
            received_at=decision_at,
        ),
    )
    bar_paths = tuple(item.path for item in feature_bars)
    contexts = DailyBarsFeatureLoader().load(
        bar_paths,
        symbols=("AAA",),
        asof_date=planning_source.feature_asof_date,
        observed_at=decision_at,
    )
    contexts = PremarketFeatureLoader().enrich(
        contexts,
        minute_files=(feature_minute.path,),
        target_date=session_date,
        observed_at=decision_at,
    )
    contexts = EarningsFeatureLoader().enrich(
        contexts,
        candidate_files=(candidate_path,),
        earnings_files=(earnings_path,),
        observed_at=decision_at,
        target_date=session_date,
    )
    feature = FeatureStore(feature_layout).write(
        feature_group="live",
        values=FeatureEngine().compute(contexts, feature_names=("return_1d",)),
        computed_at=decision_at,
    )
    feature_path = feature.path
    feature_source = FeatureSourceCapture(feature_layout).write(
        trade_date=session_date,
        asof_date=planning_source.feature_asof_date,
        observed_at=decision_at,
        feature_group="live",
        symbols=("AAA",),
        feature_names=("return_1d",),
        feature_file=feature,
        candidate_files=(candidate_path,),
        daily_bar_files=bar_paths,
        minute_bar_files=(feature_minute.path,),
        earnings_files=(earnings_path,),
        provider_observations=feature_provider_observations,
    )
    feature_lineage_paths = (
        feature_source.path,
        *feature_source.input_paths(data_lake_root=feature_layout.root),
        *feature_source.provider_paths(data_lake_root=feature_layout.root),
        *feature_source.earnings_lineage_paths(data_lake_root=feature_layout.root),
    )
    planning, planning_sources, planning_evidence_path, model_lineage_paths = (
        _write_scored_planning(
            daily,
            source=planning_source,
            feature_files=(feature_path,),
        )
    )
    _, model_evidence_path, model_path, captured_feature_path = planning_sources
    frozen = LiveOrderPlanner(
        strategy,
        strategy_sha256=strategy_file_sha256(strategy_path),
    ).plan(planning)
    planning_path = daily / "daily-order-planning.json"
    frozen_path = daily / "frozen-daily-orders.json"
    frozen.write(frozen_path)
    breaker_observation = CircuitBreakerObservation(
        session_date=session_date,
        evaluated_at=entry_at - timedelta(minutes=5),
        replay_notional=0,
        replay_net_pnl=0,
        replay_fill_rate=None,
        polygon_data_observed_at=entry_at - timedelta(minutes=5),
        alpaca_data_observed_at=entry_at - timedelta(minutes=5),
        reconciliation_break_age_sessions=None,
        replay_source_date=session_date,
    )
    breaker_spec = CircuitBreakerEvaluationSpec(
        observations=(CircuitBreakerObservationSpec.model_validate(breaker_observation.__dict__),)
    )
    breaker = CircuitBreakerEvaluator().evaluate((breaker_observation,))
    breaker_spec_path = daily / "breaker-controls.json"
    breaker_spec_path.write_bytes(encode_circuit_breaker_controls(breaker_spec))
    (
        freshness_path,
        age_path,
        polygon_path,
        alpaca_path,
        calendar_path,
        age_report_path,
    ) = _write_breaker_auxiliary_evidence(
        daily,
        observation=breaker_observation,
    )
    breaker_path = daily / "breaker-decision.json"
    breaker.write(breaker_path)
    quote_artifact = SilverWriter(LakehouseLayout(tmp_path / "market-lake")).write_stock_quotes(
        (
            StockQuote(
                symbol="AAA",
                timestamp=entry_at,
                sequence_number=1,
                bid_price=99.9,
                ask_price=100.1,
                bid_size=100,
                ask_size=100,
            ),
            StockQuote(
                symbol="AAA",
                timestamp=exit_at,
                sequence_number=2,
                bid_price=101.9,
                ask_price=102.1,
                bid_size=100,
                ask_size=100,
            ),
        ),
        event_date=session_date,
        ingested_at=exit_at + timedelta(hours=1),
    )
    spec_directory = daily / "replay-specs"
    manifest_path = daily / "replay-materialization-manifest.json"
    manifest = ReplaySpecMaterializer().materialize(
        ReplayMaterializationSpec(
            orders=frozen.intended_orders,
            decision_snapshots=frozen.decision_snapshots,
            event_sources=(
                ReplayEventSourceSpec(
                    symbol="AAA",
                    quote_files=(quote_artifact.path,),
                ),
            ),
            config=ReplayConfigSpec(
                market_impact_bps_coefficient=strategy.costs.market_impact_coef_bps
            ),
        ),
        output_dir=spec_directory,
        manifest_output=manifest_path,
    )
    evidence_directory = daily / "replay-evidence"
    index_path = evidence_directory / "index.json"
    index = ReplayManifestRunner().run(
        manifest,
        spec_directory=spec_directory,
        output_directory=evidence_directory,
        index_output=index_path,
    )
    evidence_paths = tuple(evidence_directory / item for item in index.evidence_files)
    evidence = tuple(NbboReplayEvidence.load(path) for path in evidence_paths)
    report_path = daily / "replay-session.json"
    ReplaySessionAggregator().evaluate_frozen_long_orders(
        evidence=evidence,
        intended_orders=tuple(item.to_domain() for item in frozen.intended_orders),
        session_date=session_date,
        initial_cash=frozen.portfolio.equity,
        commission_bps_per_side=strategy.costs.commission_bps_per_side,
    ).write(report_path)
    broker_orders = tuple(
        BrokerOrder.model_validate(
            {
                "id": f"broker-{item.result.order.order_id}",
                "client_order_id": item.result.order.order_id,
                "symbol": item.result.order.ticker,
                "asset_class": "us_equity",
                "qty": str(item.result.order.quantity),
                "filled_qty": str(item.result.filled_qty),
                "filled_avg_price": item.result.fill_price,
                "side": item.result.order.side,
                "type": "market",
                "time_in_force": "day",
                "status": "filled",
                "submitted_at": item.result.order.submitted_at,
                "filled_at": item.result.fragments[-1].timestamp,
            }
        )
        for item in evidence
    )
    broker_observation_paths = []
    for broker_order in broker_orders:
        observation_path = (
            daily
            / "source=alpaca-paper"
            / "dataset=orders"
            / f"{broker_order.client_order_id}.json"
        )
        observation_path.parent.mkdir(parents=True, exist_ok=True)
        observation_path.write_text(
            json.dumps(
                broker_order.model_dump(mode="json", by_alias=True),
                sort_keys=True,
                separators=(",", ":"),
            ),
            encoding="utf-8",
        )
        broker_observation_paths.append(observation_path)
    request_by_id = {item.client_order_id: item for item in frozen.paper_batch.orders}
    paper_submission = PaperBatchSubmission(
        schema_version=1,
        session_date=session_date,
        breaker_decision_sha256=breaker.sha256,
        submissions=tuple(
            PaperSubmission(
                schema_version=1,
                request=request_by_id[item.client_order_id],
                broker_order=item,
                provider_request_id=None,
                idempotent_reuse=False,
            )
            for item in broker_orders
        ),
    )
    paper_submission_path = daily / "paper-batch-submission.json"
    paper_submission.write(paper_submission_path)
    submit_observation_paths = []
    for path in broker_observation_paths:
        submit_path = daily / "submission-observations" / path.name
        submit_path.parent.mkdir(parents=True, exist_ok=True)
        submit_path.write_bytes(path.read_bytes())
        submit_observation_paths.append(submit_path)
    reconciliation = PaperOrderReconciler().evaluate(
        evidence=evidence,
        broker_orders=broker_orders,
        session_date=session_date,
        evaluated_at=exit_at + timedelta(minutes=10),
    )
    reconciliation_path = daily / f"paper-reconciliation-{reconciliation.sha256}.json"
    reconciliation.write(reconciliation_path)

    def handler(  # noqa: PLR0911 - explicit workflow-stage fixture.
        _: DailyWorkflowState,
        stage: WorkflowStage,
    ) -> tuple[Path, ...]:
        if stage is WorkflowStage.FREEZE_INPUTS:
            return tuple(
                dict.fromkeys(
                    (
                        strategy_path,
                        source_path,
                        *(
                            (
                                candidate_path,
                                live_calendar_path,
                                account_path,
                                live_snapshot_path,
                            )
                            if capture_live_provider_inputs
                            else ()
                        ),
                        *(
                            (
                                candidate_manifest_path,
                                universe_path,
                                earnings_path,
                                split_path,
                                dividend_path,
                            )
                            if capture_candidate_lineage
                            else ()
                        ),
                        *(
                            (
                                universe_source_manifest_path,
                                universe_config_path,
                                universe_halt_path,
                                universe_split_source_manifest_path,
                                universe_bar_raw_path,
                                universe_details_raw_path,
                                universe_reference_raw_path,
                                universe_split_raw_path,
                            )
                            if capture_universe_lineage
                            else ()
                        ),
                        *(
                            (
                                event_source_manifest_path,
                                event_earnings_raw_path,
                                event_split_raw_path,
                                event_dividend_raw_path,
                            )
                            if capture_event_lineage
                            else ()
                        ),
                        *(
                            (calendar_source_manifest_path, calendar_raw_path)
                            if capture_calendar_lineage
                            else ()
                        ),
                        *((live_evidence_path,) if capture_live_source_evidence else ()),
                        model_evidence_path,
                        model_path,
                        *(model_lineage_paths if capture_model_lineage else ()),
                        captured_feature_path,
                        *(feature_lineage_paths if capture_feature_lineage else ()),
                    )
                    if capture_planning_input
                    else (strategy_path,)
                )
            )
        if stage is WorkflowStage.GENERATE_ORDER_PLAN:
            return (
                (
                    planning_path,
                    *((planning_evidence_path,) if capture_scored_planning_evidence else ()),
                    frozen_path,
                )
                if capture_planning_input
                else (frozen_path,)
            )
        if stage is WorkflowStage.EVALUATE_BREAKERS:
            return (
                *((freshness_path, age_path) if capture_breaker_auxiliary else ()),
                *(
                    (polygon_path, alpaca_path)
                    if capture_breaker_auxiliary and capture_freshness_payloads
                    else ()
                ),
                *(
                    (calendar_path, age_report_path)
                    if capture_breaker_auxiliary and capture_reconciliation_age_sources
                    else ()
                ),
                *((breaker_spec_path,) if capture_breaker_spec else ()),
                breaker_path,
            )
        if stage is WorkflowStage.SUBMIT_PAPER_ORDERS:
            return (
                (paper_submission_path, *submit_observation_paths)
                if capture_submission_observations
                else (paper_submission_path,)
            )
        if stage is WorkflowStage.CAPTURE_MARKET_EVENTS:
            return (quote_artifact.path,)
        if stage is WorkflowStage.REPLAY_ORDERS:
            return (
                manifest_path,
                *(spec_directory / item.file_name for item in manifest.artifacts),
                index_path,
                *evidence_paths,
                report_path,
            )
        if stage is WorkflowStage.RECONCILE_SESSION:
            return (
                (reconciliation_path, *broker_observation_paths)
                if capture_broker_observations
                else (reconciliation_path,)
            )
        output = tmp_path / "trade-stage-artifacts" / f"{stage.value}.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("{}", encoding="utf-8")
        return (output,)

    state = DailyWorkflowRunner(
        store=store,
        handlers={stage: handler for stage in WorkflowStage},
        worker_id="worker",
        clock=lambda: datetime.now(UTC),
        trigger=WorkflowTrigger.SCHEDULED,
    ).run_until_idle(trade_date=session_date)
    assert state.complete
    return report_path


def test_prepare_phase6_controls_includes_current_future_output_path(
    tmp_path: Path,
) -> None:
    dates = (date(2026, 7, 27), date(2026, 7, 28))
    calendar = SessionFileStore(LakehouseLayout(tmp_path / "calendar")).write(
        tuple(
            MarketSession(
                session_date=item,
                open_at=datetime(item.year, item.month, item.day, 13, 30, tzinfo=UTC),
                close_at=datetime(item.year, item.month, item.day, 20, 0, tzinfo=UTC),
            )
            for item in dates
        )
    )
    artifact_root = tmp_path / "artifacts"
    prior_report = artifact_root / "trade_date=2026-07-27" / "replay-session.json"
    ReplaySessionAggregator().evaluate(
        evidence=(),
        round_trips=(),
        session_date=dates[0],
        initial_cash=100_000,
    ).write(prior_report)
    health_output = tmp_path / "controls" / "health.json"
    aggregation_output = tmp_path / "controls" / "phase6.json"
    env_file = tmp_path / ".env"
    env_file.write_text(
        f"DATA_LAKE_ROOT={tmp_path / 'lake'}",
        encoding="utf-8",
    )
    arguments = [
        "evaluation",
        "prepare-phase6-controls",
        "--session-file",
        str(calendar.path),
        "--proof-start",
        "2026-07-27",
        "--proof-end",
        "2026-07-28",
        "--current-trade-date",
        "2026-07-28",
        "--initial-cash",
        "100000",
        "--artifact-root",
        str(artifact_root),
        "--health-output",
        str(health_output),
        "--aggregation-output",
        str(aggregation_output),
        "--env-file",
        str(env_file),
    ]

    result = CliRunner().invoke(app, arguments)
    repeated = CliRunner().invoke(app, arguments)

    assert result.exit_code == 0, result.output
    assert repeated.exit_code == 0, repeated.output
    spec = Phase6AggregationSpec.model_validate_json(aggregation_output.read_bytes())
    expected_current = (artifact_root / "trade_date=2026-07-28" / "replay-session.json").resolve()
    assert spec.session_report_files == (prior_report.resolve(), expected_current)
    assert spec.workflow_store_root == (tmp_path / "lake").resolve()
    assert not expected_current.exists()
    health = WorkflowHealthReport.load(health_output)
    assert health.missing_dates == dates
    assert json.loads(result.stdout)["report_count"] == 2


def test_finalize_phase6_refreshes_health_after_workflow_completion(
    tmp_path: Path,
) -> None:
    session_date = date(2026, 7, 28)
    calendar = SessionFileStore(LakehouseLayout(tmp_path / "calendar")).write(
        (
            MarketSession(
                session_date=session_date,
                open_at=datetime(2026, 7, 28, 13, 30, tzinfo=UTC),
                close_at=datetime(2026, 7, 28, 20, 0, tzinfo=UTC),
            ),
        )
    )
    sources = _no_trade_replay_sources(tmp_path, session_date=session_date)
    artifact_root = tmp_path / "artifacts"
    replay_path = sources["report"]
    data_lake = tmp_path / "lake"
    store = DailyWorkflowStore(data_lake)
    state = _complete_source_workflow(
        store=store,
        tmp_path=tmp_path,
        session_date=session_date,
        sources=sources,
    )
    assert state.complete
    original = tmp_path / "pre-run-phase6.json"
    original.write_text(
        json.dumps(
            {
                "session_file": str(calendar.path),
                "workflow_store_root": str(data_lake),
                "workflow_health_file": "pre-run-health.json",
                "proof_start": session_date.isoformat(),
                "proof_end": session_date.isoformat(),
                "initial_cash": 100_000,
                "session_report_files": [str(replay_path)],
            }
        ),
        encoding="utf-8",
    )
    env_file = tmp_path / ".env"
    env_file.write_text(f"DATA_LAKE_ROOT={data_lake}", encoding="utf-8")
    output_directory = tmp_path / "post-completion"
    arguments = [
        "evaluation",
        "finalize-phase6",
        "--aggregation-spec",
        str(original),
        "--current-trade-date",
        session_date.isoformat(),
        "--artifact-root",
        str(artifact_root),
        "--output-directory",
        str(output_directory),
        "--env-file",
        str(env_file),
    ]

    result = CliRunner().invoke(app, arguments)
    repeated = CliRunner().invoke(app, arguments)

    assert result.exit_code == 0, result.output
    assert repeated.exit_code == 0, repeated.output
    payload = json.loads(result.stdout)
    assert payload == json.loads(repeated.stdout)
    gate = json.loads(Path(payload["gate_report_path"]).read_bytes())
    assert gate["scheduled_complete_session_count"] == 1
    assert gate["observed_session_count"] == 1
    manifest_path = Path(payload["manifest_path"])
    verified = Phase6CompletionFinalizer().verify(
        manifest_path=manifest_path,
        artifact_root=artifact_root,
        output_directory=output_directory,
        workflow_store=store,
        expected_state_sha256=state.sha256,
    )
    assert verified.report.canonical_bytes == Path(payload["gate_report_path"]).read_bytes()
    verify_arguments = [
        "evaluation",
        "verify-phase6-finalization",
        "--manifest",
        str(manifest_path),
        "--artifact-root",
        str(artifact_root),
        "--env-file",
        str(env_file),
    ]
    cli_verification = CliRunner().invoke(app, verify_arguments)
    assert cli_verification.exit_code == 0, cli_verification.output
    assert json.loads(cli_verification.stdout)["verified"] is True

    forged_gate = output_directory / f"phase6-gate-{hashlib.sha256(b'{}').hexdigest()}.json"
    forged_gate.write_bytes(b"{}")
    forged_manifest = json.loads(manifest_path.read_bytes())
    forged_manifest["gate_report_path"] = str(forged_gate)
    forged_manifest["gate_report_sha256"] = hashlib.sha256(b"{}").hexdigest()
    manifest_path.write_text(
        json.dumps(forged_manifest, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="gate verdict does not reproduce"):
        Phase6CompletionFinalizer().verify(
            manifest_path=manifest_path,
            artifact_root=artifact_root,
            output_directory=output_directory,
            workflow_store=store,
            expected_state_sha256=state.sha256,
        )
    cli_rejection = CliRunner().invoke(app, verify_arguments)
    assert cli_rejection.exit_code == 2


def test_daily_report_verifier_rejects_rehashed_summary_not_matching_sources(
    tmp_path: Path,
) -> None:
    session_date = date(2026, 7, 28)
    sources = _no_trade_replay_sources(
        tmp_path,
        session_date=session_date,
        report_initial_cash=200_000,
    )
    store = DailyWorkflowStore(tmp_path / "lake")
    _complete_source_workflow(
        store=store,
        tmp_path=tmp_path,
        session_date=session_date,
        sources=sources,
    )

    with pytest.raises(ValueError, match="differs from independent reconstruction"):
        Phase6DailyReportVerifier().verify(
            sources["report"],
            workflow_store=store,
        )


def test_daily_report_verifier_rejects_unresolved_paper_reconciliation(
    tmp_path: Path,
) -> None:
    session_date = date(2026, 7, 28)
    sources = _no_trade_replay_sources(tmp_path, session_date=session_date)
    store = DailyWorkflowStore(tmp_path / "lake")
    state = _complete_source_workflow(
        store=store,
        tmp_path=tmp_path,
        session_date=session_date,
        sources=sources,
        reconciliation_succeeds=False,
    )
    assert not state.complete

    with pytest.raises(ValueError, match="paper reconciliation stage is not complete"):
        Phase6DailyReportVerifier().verify(
            sources["report"],
            workflow_store=store,
        )


def test_daily_report_verifier_rebuilds_trade_report_from_market_sources(
    tmp_path: Path,
) -> None:
    session_date = date(2026, 7, 28)
    store = DailyWorkflowStore(tmp_path / "lake")
    report_path = _trade_source_workflow(
        store=store,
        tmp_path=tmp_path,
        session_date=session_date,
    )

    verified = Phase6DailyReportVerifier().verify(
        report_path,
        workflow_store=store,
    )

    assert verified.intended_order_count == 2
    assert verified.fully_filled_order_count == 2
    assert verified.reconciliation_break_count == 0


def test_daily_report_verifier_requires_raw_broker_observations(
    tmp_path: Path,
) -> None:
    session_date = date(2026, 7, 28)
    store = DailyWorkflowStore(tmp_path / "lake")
    report_path = _trade_source_workflow(
        store=store,
        tmp_path=tmp_path,
        session_date=session_date,
        capture_broker_observations=False,
    )

    with pytest.raises(ValueError, match="lacks exact raw broker observations"):
        Phase6DailyReportVerifier().verify(
            report_path,
            workflow_store=store,
        )


def test_daily_report_verifier_requires_raw_submission_observations(
    tmp_path: Path,
) -> None:
    session_date = date(2026, 7, 28)
    store = DailyWorkflowStore(tmp_path / "lake")
    report_path = _trade_source_workflow(
        store=store,
        tmp_path=tmp_path,
        session_date=session_date,
        capture_submission_observations=False,
    )

    with pytest.raises(ValueError, match="submission lacks exact raw broker observations"):
        Phase6DailyReportVerifier().verify(
            report_path,
            workflow_store=store,
        )


def test_daily_report_verifier_requires_reproducible_breaker_decision(
    tmp_path: Path,
) -> None:
    session_date = date(2026, 7, 28)
    store = DailyWorkflowStore(tmp_path / "lake")
    report_path = _trade_source_workflow(
        store=store,
        tmp_path=tmp_path,
        session_date=session_date,
        capture_breaker_spec=False,
    )

    with pytest.raises(ValueError, match="does not reproduce from one captured control"):
        Phase6DailyReportVerifier().verify(
            report_path,
            workflow_store=store,
        )


def test_daily_report_verifier_requires_breaker_safety_evidence(
    tmp_path: Path,
) -> None:
    session_date = date(2026, 7, 28)
    store = DailyWorkflowStore(tmp_path / "lake")
    report_path = _trade_source_workflow(
        store=store,
        tmp_path=tmp_path,
        session_date=session_date,
        capture_breaker_auxiliary=False,
    )

    with pytest.raises(ValueError, match="exactly one provider-freshness-"):
        Phase6DailyReportVerifier().verify(
            report_path,
            workflow_store=store,
        )


def test_daily_report_verifier_requires_raw_freshness_payloads(
    tmp_path: Path,
) -> None:
    session_date = date(2026, 7, 28)
    store = DailyWorkflowStore(tmp_path / "lake")
    report_path = _trade_source_workflow(
        store=store,
        tmp_path=tmp_path,
        session_date=session_date,
        capture_freshness_payloads=False,
    )

    with pytest.raises(ValueError, match="exactly one captured raw Polygon and Alpaca"):
        Phase6DailyReportVerifier().verify(
            report_path,
            workflow_store=store,
        )


def test_daily_report_verifier_requires_reconciliation_age_sources(
    tmp_path: Path,
) -> None:
    session_date = date(2026, 7, 28)
    store = DailyWorkflowStore(tmp_path / "lake")
    report_path = _trade_source_workflow(
        store=store,
        tmp_path=tmp_path,
        session_date=session_date,
        capture_reconciliation_age_sources=False,
    )

    with pytest.raises(ValueError, match="captured calendar"):
        Phase6DailyReportVerifier().verify(
            report_path,
            workflow_store=store,
        )


def test_daily_report_verifier_requires_captured_planning_input(
    tmp_path: Path,
) -> None:
    session_date = date(2026, 7, 28)
    store = DailyWorkflowStore(tmp_path / "lake")
    report_path = _trade_source_workflow(
        store=store,
        tmp_path=tmp_path,
        session_date=session_date,
        capture_planning_input=False,
    )

    with pytest.raises(ValueError, match="exactly one captured planning input"):
        Phase6DailyReportVerifier().verify(
            report_path,
            workflow_store=store,
        )


def test_daily_report_verifier_rejects_unscored_compatibility_planning(
    tmp_path: Path,
) -> None:
    session_date = date(2026, 7, 28)
    store = DailyWorkflowStore(tmp_path / "lake")
    report_path = _trade_source_workflow(
        store=store,
        tmp_path=tmp_path,
        session_date=session_date,
        capture_scored_planning_evidence=False,
    )

    with pytest.raises(ValueError, match="exactly one captured scored-planning"):
        Phase6DailyReportVerifier().verify(
            report_path,
            workflow_store=store,
        )


def test_daily_report_verifier_requires_feature_generation_lineage(
    tmp_path: Path,
) -> None:
    session_date = date(2026, 7, 28)
    store = DailyWorkflowStore(tmp_path / "lake")
    report_path = _trade_source_workflow(
        store=store,
        tmp_path=tmp_path,
        session_date=session_date,
        capture_feature_lineage=False,
    )

    with pytest.raises(ValueError, match="captured source manifest"):
        Phase6DailyReportVerifier().verify(
            report_path,
            workflow_store=store,
        )


def test_daily_report_verifier_requires_production_model_lineage(
    tmp_path: Path,
) -> None:
    session_date = date(2026, 7, 28)
    store = DailyWorkflowStore(tmp_path / "lake")
    report_path = _trade_source_workflow(
        store=store,
        tmp_path=tmp_path,
        session_date=session_date,
        capture_model_lineage=False,
    )

    with pytest.raises(ValueError, match="captured source manifest"):
        Phase6DailyReportVerifier().verify(
            report_path,
            workflow_store=store,
        )


def test_daily_report_verifier_requires_live_source_evidence(
    tmp_path: Path,
) -> None:
    session_date = date(2026, 7, 28)
    store = DailyWorkflowStore(tmp_path / "lake")
    report_path = _trade_source_workflow(
        store=store,
        tmp_path=tmp_path,
        session_date=session_date,
        capture_live_source_evidence=False,
    )

    with pytest.raises(ValueError, match="exactly one captured source evidence"):
        Phase6DailyReportVerifier().verify(
            report_path,
            workflow_store=store,
        )


def test_daily_report_verifier_requires_live_provider_inputs(
    tmp_path: Path,
) -> None:
    session_date = date(2026, 7, 28)
    store = DailyWorkflowStore(tmp_path / "lake")
    report_path = _trade_source_workflow(
        store=store,
        tmp_path=tmp_path,
        session_date=session_date,
        capture_live_provider_inputs=False,
    )

    with pytest.raises(ValueError, match="exact captured provider and workflow inputs"):
        Phase6DailyReportVerifier().verify(
            report_path,
            workflow_store=store,
        )


def test_daily_report_verifier_requires_candidate_generation_lineage(
    tmp_path: Path,
) -> None:
    session_date = date(2026, 7, 28)
    store = DailyWorkflowStore(tmp_path / "lake")
    report_path = _trade_source_workflow(
        store=store,
        tmp_path=tmp_path,
        session_date=session_date,
        capture_candidate_lineage=False,
    )

    with pytest.raises(ValueError, match="candidate-generation manifest"):
        Phase6DailyReportVerifier().verify(
            report_path,
            workflow_store=store,
        )


def test_daily_report_verifier_requires_universe_generation_lineage(
    tmp_path: Path,
) -> None:
    session_date = date(2026, 7, 28)
    store = DailyWorkflowStore(tmp_path / "lake")
    report_path = _trade_source_workflow(
        store=store,
        tmp_path=tmp_path,
        session_date=session_date,
        capture_universe_lineage=False,
    )

    with pytest.raises(ValueError, match="exact captured data-lake sources"):
        Phase6DailyReportVerifier().verify(
            report_path,
            workflow_store=store,
        )


def test_daily_report_verifier_requires_event_generation_lineage(
    tmp_path: Path,
) -> None:
    session_date = date(2026, 7, 28)
    store = DailyWorkflowStore(tmp_path / "lake")
    report_path = _trade_source_workflow(
        store=store,
        tmp_path=tmp_path,
        session_date=session_date,
        capture_event_lineage=False,
    )

    with pytest.raises(ValueError, match="exact captured data-lake sources"):
        Phase6DailyReportVerifier().verify(
            report_path,
            workflow_store=store,
        )


def test_daily_report_verifier_requires_calendar_generation_lineage(
    tmp_path: Path,
) -> None:
    session_date = date(2026, 7, 28)
    store = DailyWorkflowStore(tmp_path / "lake")
    report_path = _trade_source_workflow(
        store=store,
        tmp_path=tmp_path,
        session_date=session_date,
        capture_calendar_lineage=False,
    )

    with pytest.raises(ValueError, match="exact captured data-lake sources"):
        Phase6DailyReportVerifier().verify(
            report_path,
            workflow_store=store,
        )


def test_daily_report_verifier_rejects_changed_candidate_source(
    tmp_path: Path,
) -> None:
    session_date = date(2026, 7, 28)
    store = DailyWorkflowStore(tmp_path / "lake")
    report_path = _trade_source_workflow(
        store=store,
        tmp_path=tmp_path,
        session_date=session_date,
    )
    universe_path = next(tmp_path.glob("**/candidate-lake/**/snapshot-*.parquet"))
    universe_path.write_bytes(b"changed after candidate generation")

    with pytest.raises(ValueError, match="workflow artifact changed after capture"):
        Phase6DailyReportVerifier().verify(
            report_path,
            workflow_store=store,
        )


def test_daily_report_verifier_rejects_changed_event_provider_source(
    tmp_path: Path,
) -> None:
    session_date = date(2026, 7, 28)
    store = DailyWorkflowStore(tmp_path / "lake")
    report_path = _trade_source_workflow(
        store=store,
        tmp_path=tmp_path,
        session_date=session_date,
    )
    event_path = next(
        tmp_path.glob("**/candidate-lake/bronze/source=finnhub/dataset=earnings-calendar/**/*.json")
    )
    event_path.write_bytes(b"{}")

    with pytest.raises(ValueError, match="workflow artifact changed after capture"):
        Phase6DailyReportVerifier().verify(
            report_path,
            workflow_store=store,
        )


def test_daily_report_verifier_rejects_changed_calendar_provider_source(
    tmp_path: Path,
) -> None:
    session_date = date(2026, 7, 28)
    store = DailyWorkflowStore(tmp_path / "lake")
    report_path = _trade_source_workflow(
        store=store,
        tmp_path=tmp_path,
        session_date=session_date,
    )
    calendar_path = next(
        tmp_path.glob("**/candidate-lake/bronze/source=alpaca/dataset=market-calendar/**/*.json")
    )
    calendar_path.write_bytes(b"[]")

    with pytest.raises(ValueError, match="workflow artifact changed after capture"):
        Phase6DailyReportVerifier().verify(
            report_path,
            workflow_store=store,
        )
