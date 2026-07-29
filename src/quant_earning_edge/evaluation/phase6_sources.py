"""Independent reconstruction of daily Phase 6 replay reports."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING

from quant_earning_edge.backtest import NbboReplayEvidence, ReplayConfigSpec
from quant_earning_edge.data import (
    ReplayEvidenceIndex,
    ReplayManifestRunner,
    ReplayMaterializationManifest,
    ReplayMaterializationSpec,
    ReplaySpecMaterializer,
    replay_sources_from_files,
)
from quant_earning_edge.evaluation.replay_session import (
    ReplaySessionAggregator,
    ReplaySessionReport,
)
from quant_earning_edge.live import (
    BrokerOrder,
    PaperBatchSubmission,
    PaperOrderReconciler,
    PaperReconciliationReport,
)
from quant_earning_edge.orchestration.workflow import (
    DailyWorkflowStore,
    StageStatus,
    WorkflowStage,
)

if TYPE_CHECKING:
    from quant_earning_edge.monitoring.breakers import CircuitBreakerDecision
    from quant_earning_edge.orchestration.workflow import (
        ArtifactReference,
        DailyWorkflowState,
        StageRecord,
    )
    from quant_earning_edge.signals.config import EarningsStrategyConfig
    from quant_earning_edge.signals.live_capture import LiveSourceCaptureArtifact
    from quant_earning_edge.signals.live_orders import (
        DailyOrderPlanningSpec,
        FrozenDailyOrders,
    )
    from quant_earning_edge.signals.live_planning import LivePlanningSourceSpec
    from quant_earning_edge.universe import EventCandidateManifest


class Phase6DailyReportVerifier:
    """Re-run order replays and daily aggregation from captured workflow sources."""

    def verify(
        self,
        report_path: Path,
        *,
        workflow_store: DailyWorkflowStore,
    ) -> ReplaySessionReport:
        """Return a report only after byte-exact source reconstruction."""
        from quant_earning_edge.signals.config import load_strategy_config  # noqa: PLC0415
        from quant_earning_edge.signals.live_orders import (  # noqa: PLC0415
            FrozenDailyOrders,
        )

        resolved_report = report_path.resolve()
        report = ReplaySessionReport.load(resolved_report)
        state = workflow_store.load_latest(report.session_date)
        if state is None:
            raise ValueError(f"daily replay report has no workflow state: {report.session_date}")
        state.verify_artifacts()
        self._require_report_artifact(resolved_report, state=state)
        frozen_path = self._unique_named_artifact(
            self._stage(state, WorkflowStage.GENERATE_ORDER_PLAN),
            "frozen-daily-orders.json",
        )
        frozen = FrozenDailyOrders.load(frozen_path)
        if frozen.trade_date != report.session_date:
            raise ValueError("frozen orders and daily replay report dates differ")
        strategy_path = self._strategy_artifact(state=state, frozen=frozen)
        strategy = load_strategy_config(strategy_path)
        self._verify_frozen_orders(
            state=state,
            frozen=frozen,
            strategy_path=strategy_path,
        )
        self._verify_paper_submission(state=state, frozen=frozen)
        replay_stage = self._stage(state, WorkflowStage.REPLAY_ORDERS)
        manifest_path = self._unique_named_artifact(
            replay_stage,
            "replay-materialization-manifest.json",
        )
        index_path = self._unique_named_artifact(replay_stage, "index.json")
        manifest = ReplayMaterializationManifest.load(manifest_path)
        captured_index = ReplayEvidenceIndex.load(index_path)
        captured_evidence = self._captured_evidence(
            replay_stage,
            index=captured_index,
        )
        self._verify_paper_reconciliation(
            state=state,
            frozen=frozen,
            evidence=captured_evidence,
        )
        reproduced_evidence = self._reproduce_order_evidence(
            state=state,
            replay_stage=replay_stage,
            manifest=manifest,
            captured_index=captured_index,
            frozen=frozen,
            strategy=strategy,
        )
        if tuple(item.sha256 for item in reproduced_evidence) != tuple(
            item.sha256 for item in captured_evidence
        ):
            raise ValueError("captured replay evidence differs from independent reconstruction")
        reproduced_report = ReplaySessionAggregator().evaluate_frozen_long_orders(
            evidence=reproduced_evidence,
            intended_orders=tuple(item.to_domain() for item in frozen.intended_orders),
            session_date=frozen.trade_date,
            initial_cash=frozen.portfolio.equity,
            commission_bps_per_side=strategy.costs.commission_bps_per_side,
        )
        if reproduced_report.canonical_bytes != report.canonical_bytes:
            supplied = json.loads(report.canonical_bytes)
            reproduced = json.loads(reproduced_report.canonical_bytes)
            fields = tuple(
                sorted(
                    key
                    for key in set(supplied) | set(reproduced)
                    if supplied.get(key) != reproduced.get(key)
                )
            )
            raise ValueError(
                f"daily replay report differs from independent reconstruction (fields={fields})"
            )
        return report

    @staticmethod
    def _verify_frozen_orders(
        *,
        state: DailyWorkflowState,
        frozen: FrozenDailyOrders,
        strategy_path: Path,
    ) -> None:
        from quant_earning_edge.signals.config import load_strategy_config  # noqa: PLC0415
        from quant_earning_edge.signals.live_orders import (  # noqa: PLC0415
            DailyOrderPlanningSpec,
            LiveOrderPlanner,
            strategy_file_sha256,
        )

        paths = Phase6DailyReportVerifier._artifact_paths_by_sha(state).get(
            frozen.input_sha256,
            [],
        )
        if len(paths) != 1:
            raise ValueError("frozen daily orders must bind to exactly one captured planning input")
        encoded = paths[0].read_bytes()
        planning = DailyOrderPlanningSpec.model_validate_json(encoded)
        if planning.canonical_bytes != encoded:
            raise ValueError("captured daily planning input is not canonical")
        Phase6DailyReportVerifier._verify_scored_planning(
            state=state,
            planning=planning,
        )
        reproduced = LiveOrderPlanner(
            load_strategy_config(strategy_path),
            strategy_sha256=strategy_file_sha256(strategy_path),
        ).plan(planning)
        if reproduced.canonical_bytes != frozen.canonical_bytes:
            raise ValueError("frozen daily orders differ from captured planning input")

    @staticmethod
    def _verify_scored_planning(
        *,
        state: DailyWorkflowState,
        planning: DailyOrderPlanningSpec,
    ) -> None:
        from quant_earning_edge.signals.live_planning import (  # noqa: PLC0415
            LivePlanningAssembler,
            LivePlanningSourceSpec,
            ScoredPlanningArtifact,
        )
        from quant_earning_edge.signals.production_model import (  # noqa: PLC0415
            ProductionModelArtifact,
        )

        evidence = []
        for stage in state.stages:
            for artifact in stage.output_artifacts:
                candidate = Path(artifact.path).resolve()
                if candidate.suffix != ".json":
                    continue
                try:
                    scored = ScoredPlanningArtifact.load(candidate)
                except (OSError, ValueError):
                    continue
                if scored.planning.canonical_bytes == planning.canonical_bytes:
                    evidence.append(scored)
        if len(evidence) != 1:
            raise ValueError(
                "planning input must bind to exactly one captured scored-planning evidence"
            )
        scored = evidence[0]
        paths_by_sha = Phase6DailyReportVerifier._artifact_paths_by_sha(state)
        source_paths = paths_by_sha.get(scored.source_sha256, [])
        model_evidence_paths = paths_by_sha.get(scored.model_artifact_sha256, [])
        model_paths = paths_by_sha.get(scored.model_sha256, [])
        feature_matches = tuple(
            paths_by_sha.get(digest, []) for digest in scored.feature_file_sha256
        )
        if (
            len(source_paths) != 1
            or len(model_evidence_paths) != 1
            or len(model_paths) != 1
            or any(len(paths) != 1 for paths in feature_matches)
        ):
            raise ValueError("scored planning lacks exact captured source/model/features")
        source_encoded = source_paths[0].read_bytes()
        source = LivePlanningSourceSpec.model_validate_json(source_encoded)
        if source.canonical_bytes != source_encoded:
            raise ValueError("captured live planning source is not canonical")
        Phase6DailyReportVerifier._verify_live_source(
            state=state,
            source=source,
        )
        model = ProductionModelArtifact.load(
            evidence_path=model_evidence_paths[0],
            model_path=model_paths[0],
        )
        feature_paths = tuple(sorted((paths[0] for paths in feature_matches), key=str))
        reproduced = LivePlanningAssembler().assemble(
            source=source,
            model=model,
            feature_files=feature_paths,
        )
        if reproduced.canonical_bytes != scored.canonical_bytes:
            raise ValueError("scored planning differs from captured source/model/features")

    @staticmethod
    def _verify_live_source(
        *,
        state: DailyWorkflowState,
        source: LivePlanningSourceSpec,
    ) -> None:
        from quant_earning_edge.data.clients import PolygonClient  # noqa: PLC0415
        from quant_earning_edge.live import PaperAccountSnapshot  # noqa: PLC0415
        from quant_earning_edge.signals.live_capture import (  # noqa: PLC0415
            LiveSourceCaptureArtifact,
            LiveSourceCaptureAssembler,
        )

        evidence = []
        for stage in state.stages:
            for artifact in stage.output_artifacts:
                candidate = Path(artifact.path).resolve()
                if candidate.suffix != ".json":
                    continue
                try:
                    captured = LiveSourceCaptureArtifact.load(candidate)
                except (OSError, ValueError):
                    continue
                if captured.source.canonical_bytes == source.canonical_bytes:
                    evidence.append(captured)
        if len(evidence) != 1:
            raise ValueError(
                "live planning source must bind to exactly one captured source evidence"
            )
        captured = evidence[0]
        paths_by_sha = Phase6DailyReportVerifier._artifact_paths_by_sha(state)
        candidate_paths = paths_by_sha.get(captured.candidate_file_sha256, [])
        session_paths = paths_by_sha.get(captured.session_file_sha256, [])
        account_paths = paths_by_sha.get(captured.account_payload_sha256, [])
        snapshot_matches = tuple(
            paths_by_sha.get(digest, []) for digest in captured.snapshot_payload_sha256
        )
        replay_matches = tuple(
            paths_by_sha.get(digest, []) for digest in captured.prior_replay_sha256
        )
        if (
            not candidate_paths
            or not session_paths
            or not account_paths
            or any(not paths for paths in snapshot_matches)
            or any(not paths for paths in replay_matches)
        ):
            raise ValueError("live source lacks exact captured provider and workflow inputs")
        Phase6DailyReportVerifier._verify_candidate_generation(
            state=state,
            source=source,
            captured=captured,
            candidate_path=candidate_paths[0],
            paths_by_sha=paths_by_sha,
        )
        account = PaperAccountSnapshot.from_payload(
            json.loads(account_paths[0].read_bytes()),
            captured_at=source.decision_at,
        )
        if len(snapshot_matches) != len(source.observations):
            raise ValueError("live source snapshot count differs from captured observations")
        snapshots = tuple(
            PolygonClient.ticker_snapshot_from_payload(
                json.loads(paths[0].read_bytes()),
                symbol=observation.symbol,
                captured_at=source.decision_at,
            )
            for observation, paths in zip(
                source.observations,
                snapshot_matches,
                strict=True,
            )
        )
        reproduced = LiveSourceCaptureAssembler().assemble(
            trade_date=source.trade_date,
            captured_at=source.decision_at,
            candidate_file=candidate_paths[0],
            session_file=session_paths[0],
            account=account,
            initial_cash=captured.initial_cash,
            snapshots=snapshots,
            prior_replay_files=tuple(paths[0] for paths in replay_matches),
            minimum_probability=source.minimum_probability,
        )
        if reproduced.canonical_bytes != captured.canonical_bytes:
            raise ValueError("live source differs from captured provider or workflow inputs")

    @staticmethod
    def _verify_candidate_generation(
        *,
        state: DailyWorkflowState,
        source: LivePlanningSourceSpec,
        captured: LiveSourceCaptureArtifact,
        candidate_path: Path,
        paths_by_sha: dict[str, list[Path]],
    ) -> None:
        from quant_earning_edge.data.layout import LakehouseLayout  # noqa: PLC0415
        from quant_earning_edge.universe import (  # noqa: PLC0415
            EventCandidateJob,
            EventCandidateManifest,
            UniverseSourceCapture,
            UniverseSourceCaptureManifest,
        )

        manifests = []
        for stage in state.stages:
            for artifact in stage.output_artifacts:
                path = Path(artifact.path).resolve()
                if path.suffix != ".json":
                    continue
                try:
                    manifest = EventCandidateManifest.load(path)
                except (OSError, ValueError):
                    continue
                if manifest.raw["candidate_file_sha256"] == captured.candidate_file_sha256:
                    manifests.append(manifest)
        if len(manifests) != 1:
            raise ValueError(
                "live source must bind to exactly one captured candidate-generation manifest"
            )
        manifest = manifests[0]
        if manifest.raw["schema_version"] != 3:
            raise ValueError("candidate generation lacks source-bound universe lineage")
        source_root, sources = Phase6DailyReportVerifier._candidate_sources(
            manifest=manifest,
            paths_by_sha=paths_by_sha,
        )
        if (
            manifest.raw["trade_date"] != source.trade_date.isoformat()
            or manifest.raw["candidate_file_sha256"]
            != hashlib.sha256(candidate_path.read_bytes()).hexdigest()
        ):
            raise ValueError("live source candidate identity differs from its generation manifest")
        source_groups = manifest.raw["source_files"]
        lineage_count = len(manifest.universe_lineage_entries)
        universe_manifest_path = sources[0]
        universe_manifest = UniverseSourceCaptureManifest.load(universe_manifest_path)
        if universe_manifest.source_paths(data_lake_root=source_root) != sources[1:lineage_count]:
            raise ValueError("candidate universe lineage differs from its source manifest")
        candidate_sources = sources[lineage_count:]
        earnings_end = 2 + len(source_groups["earnings_files"])
        splits_end = earnings_end + len(source_groups["split_files"])
        with TemporaryDirectory(prefix="qee-candidate-reconstruction-") as temporary:
            universe_output = LakehouseLayout(Path(temporary) / "universe")
            reproduced_universe = UniverseSourceCapture.reproduce(
                universe_manifest,
                data_lake_root=source_root,
                output_layout=universe_output,
            )
            if reproduced_universe.path.read_bytes() != candidate_sources[0].read_bytes():
                raise ValueError("candidate universe differs from independent reconstruction")
            reproduced = EventCandidateJob(
                LakehouseLayout(Path(temporary) / "candidates"),
                source_root=source_root,
            ).run(
                trade_date=source.trade_date,
                decision_at=datetime.fromisoformat(manifest.raw["decision_at"]),
                universe_snapshot=candidate_sources[0],
                session_file=candidate_sources[1],
                earnings_files=candidate_sources[2:earnings_end],
                split_files=candidate_sources[earnings_end:splits_end],
                dividend_files=candidate_sources[splits_end:],
                universe_source_manifest=universe_manifest_path,
            )
            if (
                hashlib.sha256(reproduced.path.read_bytes()).hexdigest()
                != captured.candidate_file_sha256
                or EventCandidateManifest.load(reproduced.manifest_path).raw != manifest.raw
            ):
                raise ValueError(
                    "candidate artifact differs from reconstructed captured upstream inputs"
                )

    @staticmethod
    def _candidate_sources(
        *,
        manifest: EventCandidateManifest,
        paths_by_sha: dict[str, list[Path]],
    ) -> tuple[Path, tuple[Path, ...]]:
        first_entry = manifest.source_entries[0]
        first_paths = paths_by_sha.get(first_entry["sha256"], [])
        relative = Path(first_entry["path"])
        roots = []
        for candidate in first_paths:
            root = candidate
            for _ in relative.parts:
                root = root.parent
            if (root / relative).resolve() == candidate.resolve():
                roots.append(root.resolve())
        for root in sorted(set(roots), key=str):
            sources = tuple((root / entry["path"]).resolve() for entry in manifest.source_entries)
            if all(
                path in paths_by_sha.get(entry["sha256"], [])
                for path, entry in zip(sources, manifest.source_entries, strict=True)
            ):
                return root, sources
        raise ValueError("candidate manifest lacks its exact captured data-lake sources")

    @staticmethod
    def _stage(
        state: DailyWorkflowState,
        stage: WorkflowStage,
    ) -> StageRecord:
        return next(item for item in state.stages if item.stage is stage)

    @staticmethod
    def _unique_named_artifact(stage: StageRecord, file_name: str) -> Path:
        paths = tuple(
            Path(item.path).resolve()
            for item in stage.output_artifacts
            if Path(item.path).name == file_name
        )
        if len(paths) != 1:
            raise ValueError(f"workflow stage must capture exactly one {file_name}")
        return paths[0]

    @staticmethod
    def _require_report_artifact(
        report_path: Path,
        *,
        state: DailyWorkflowState,
    ) -> None:
        replay = Phase6DailyReportVerifier._stage(state, WorkflowStage.REPLAY_ORDERS)
        captured = {
            Path(item.path).resolve()
            for item in replay.output_artifacts
            if Path(item.path).name == "replay-session.json"
        }
        if captured != {report_path}:
            raise ValueError("daily replay report is not the workflow-captured report")

    @staticmethod
    def _strategy_artifact(
        *,
        state: DailyWorkflowState,
        frozen: FrozenDailyOrders,
    ) -> Path:
        from quant_earning_edge.signals.live_orders import (  # noqa: PLC0415
            strategy_file_sha256,
        )

        freeze = Phase6DailyReportVerifier._stage(state, WorkflowStage.FREEZE_INPUTS)
        paths = tuple(
            sorted(
                {
                    Path(item.path).resolve()
                    for item in freeze.output_artifacts
                    if item.sha256 == frozen.strategy_config_sha256
                },
                key=str,
            )
        )
        if not paths:
            raise ValueError("workflow did not capture the frozen strategy configuration")
        path = paths[0]
        if strategy_file_sha256(path) != frozen.strategy_config_sha256:
            raise ValueError("captured strategy differs from frozen daily orders")
        return path

    @staticmethod
    def _captured_evidence(
        stage: StageRecord,
        *,
        index: ReplayEvidenceIndex,
    ) -> tuple[NbboReplayEvidence, ...]:
        by_name: dict[str, tuple[ArtifactReference, Path]] = {}
        for artifact in stage.output_artifacts:
            path = Path(artifact.path).resolve()
            if path.name.startswith("replay-evidence-") and path.suffix == ".json":
                if path.name in by_name:
                    raise ValueError("workflow captured duplicate replay evidence filenames")
                by_name[path.name] = (artifact, path)
        if set(by_name) != set(index.evidence_files):
            raise ValueError("workflow replay evidence files differ from their index")
        evidence = tuple(
            NbboReplayEvidence.load(by_name[file_name][1]) for file_name in index.evidence_files
        )
        if tuple(item.sha256 for item in evidence) != index.evidence_sha256:
            raise ValueError("workflow replay evidence hashes differ from their index")
        return evidence

    @staticmethod
    def _verify_paper_submission(
        *,
        state: DailyWorkflowState,
        frozen: FrozenDailyOrders,
    ) -> None:
        breaker_stage = Phase6DailyReportVerifier._stage(
            state,
            WorkflowStage.EVALUATE_BREAKERS,
        )
        submit_stage = Phase6DailyReportVerifier._stage(
            state,
            WorkflowStage.SUBMIT_PAPER_ORDERS,
        )
        if (
            breaker_stage.status is not StageStatus.SUCCEEDED
            or submit_stage.status is not StageStatus.SUCCEEDED
        ):
            raise ValueError("paper submission controls are not complete")
        breaker_path = Phase6DailyReportVerifier._unique_named_artifact(
            breaker_stage,
            "breaker-decision.json",
        )
        submission_path = Phase6DailyReportVerifier._unique_named_artifact(
            submit_stage,
            "paper-batch-submission.json",
        )
        breaker = Phase6DailyReportVerifier._reproduced_breaker(
            state=state,
            breaker_path=breaker_path,
        )
        submission = PaperBatchSubmission.load(submission_path)
        if breaker.halt_new_orders:
            raise ValueError("paper submission used a halted breaker decision")
        if (
            breaker.session_date != frozen.trade_date
            or submission.session_date != frozen.trade_date
            or submission.breaker_decision_sha256 != breaker.sha256
        ):
            raise ValueError("paper submission date or breaker binding differs")
        requests = {item.client_order_id: item for item in frozen.paper_batch.orders}
        submitted = {item.request.client_order_id: item for item in submission.submissions}
        if set(submitted) != set(requests):
            raise ValueError("paper submission identities differ from frozen orders")
        observation_paths = tuple(
            Path(item.path).resolve()
            for item in submit_stage.output_artifacts
            if Path(item.path).suffix == ".json" and Path(item.path).resolve() != submission_path
        )
        if len(observation_paths) != len(submission.submissions):
            raise ValueError("workflow paper submission lacks exact raw broker observations")
        observed = {
            order.client_order_id: order
            for order in (
                Phase6DailyReportVerifier._load_raw_broker_observation(path)
                for path in observation_paths
            )
        }
        if len(observed) != len(observation_paths) or set(observed) != set(requests):
            raise ValueError("raw broker submission identities differ from frozen orders")
        for order_id, record in submitted.items():
            if record.request != requests[order_id] or record.broker_order != observed[order_id]:
                raise ValueError("paper submission differs from frozen or raw broker evidence")

    @staticmethod
    def _reproduced_breaker(
        *,
        state: DailyWorkflowState,
        breaker_path: Path,
    ) -> CircuitBreakerDecision:
        from quant_earning_edge.data import SessionFileStore  # noqa: PLC0415
        from quant_earning_edge.monitoring.breakers import (  # noqa: PLC0415
            CircuitBreakerDecision,
            CircuitBreakerEvaluationSpec,
            CircuitBreakerEvaluator,
        )
        from quant_earning_edge.monitoring.control_inputs import (  # noqa: PLC0415
            encode_circuit_breaker_controls,
        )
        from quant_earning_edge.monitoring.freshness import (  # noqa: PLC0415
            ProviderFreshnessEvidence,
        )
        from quant_earning_edge.monitoring.reconciliation_age import (  # noqa: PLC0415
            ReconciliationAgeEvaluator,
            ReconciliationAgeEvidence,
        )

        breaker = CircuitBreakerDecision.load(breaker_path)
        reproduced_paths = []
        reproduced_specs = []
        for stage in state.stages:
            for artifact in stage.output_artifacts:
                candidate = Path(artifact.path).resolve()
                if candidate == breaker_path or candidate.suffix != ".json":
                    continue
                try:
                    encoded = candidate.read_bytes()
                    spec = CircuitBreakerEvaluationSpec.model_validate_json(encoded)
                    if encode_circuit_breaker_controls(spec) != encoded:
                        continue
                    reproduced = CircuitBreakerEvaluator().evaluate(
                        tuple(item.to_domain() for item in spec.observations)
                    )
                except (OSError, ValueError):
                    continue
                if reproduced.canonical_bytes == breaker.canonical_bytes:
                    reproduced_paths.append(candidate)
                    reproduced_specs.append(spec)
        if len(reproduced_paths) != 1:
            raise ValueError(
                "breaker decision does not reproduce from one captured control specification"
            )
        freshness_path = Phase6DailyReportVerifier._unique_prefixed_artifact(
            state,
            "provider-freshness-",
        )
        age_path = Phase6DailyReportVerifier._unique_prefixed_artifact(
            state,
            "reconciliation-age-",
        )
        freshness = ProviderFreshnessEvidence.load(freshness_path)
        age = ReconciliationAgeEvidence.load(age_path)
        artifact_paths = Phase6DailyReportVerifier._artifact_paths_by_sha(state)
        calendar_paths = artifact_paths.get(age.calendar_sha256, [])
        report_paths = tuple(artifact_paths.get(digest, []) for digest in age.input_report_sha256)
        if not calendar_paths or any(not paths for paths in report_paths):
            raise ValueError(
                "reconciliation-age evidence must bind to a captured calendar "
                "and each source report"
            )
        reproduced_age = ReconciliationAgeEvaluator().evaluate(
            calendar=SessionFileStore.load(calendar_paths[0]),
            reports=tuple(PaperReconciliationReport.load(paths[0]) for paths in report_paths),
            control_date=age.control_date,
            evaluated_at=age.evaluated_at,
        )
        if reproduced_age.canonical_bytes != age.canonical_bytes:
            raise ValueError(
                "reconciliation-age evidence differs from captured calendar or reports"
            )
        canonical_payloads = Phase6DailyReportVerifier._canonical_payloads_by_sha(state)
        polygon_payloads = canonical_payloads.get(freshness.polygon_payload_sha256, [])
        alpaca_payloads = canonical_payloads.get(freshness.alpaca_payload_sha256, [])
        if len(polygon_payloads) != 1 or len(alpaca_payloads) != 1:
            raise ValueError(
                "freshness evidence must bind to exactly one captured raw "
                "Polygon and Alpaca payload"
            )
        reproduced_freshness = ProviderFreshnessEvidence.from_payloads(
            polygon_payload=polygon_payloads[0],
            alpaca_payload=alpaca_payloads[0],
            evaluated_at=freshness.evaluated_at,
            polygon_symbol=freshness.polygon_symbol,
            alpaca_request_id=freshness.alpaca_request_id,
        )
        if reproduced_freshness.canonical_bytes != freshness.canonical_bytes:
            raise ValueError("freshness evidence differs from captured raw provider payloads")
        current = reproduced_specs[0].observations[-1]
        if (
            current.session_date != age.control_date
            or current.evaluated_at != freshness.evaluated_at
            or current.evaluated_at != age.evaluated_at
            or current.polygon_data_observed_at != freshness.polygon_data_observed_at
            or current.alpaca_data_observed_at != freshness.alpaca_data_observed_at
            or current.reconciliation_break_age_sessions != age.reconciliation_break_age_sessions
        ):
            raise ValueError(
                "breaker controls differ from captured freshness or reconciliation age"
            )
        return breaker

    @staticmethod
    def _artifact_paths_by_sha(state: DailyWorkflowState) -> dict[str, list[Path]]:
        paths: dict[str, list[Path]] = {}
        for candidate in {
            Path(artifact.path).resolve()
            for stage in state.stages
            for artifact in stage.output_artifacts
        }:
            try:
                digest = hashlib.sha256(candidate.read_bytes()).hexdigest()
            except OSError:
                continue
            paths.setdefault(digest, []).append(candidate)
        return paths

    @staticmethod
    def _canonical_payloads_by_sha(
        state: DailyWorkflowState,
    ) -> dict[str, list[object]]:
        payloads: dict[str, list[object]] = {}
        artifact_paths = {
            Path(artifact.path).resolve()
            for stage in state.stages
            for artifact in stage.output_artifacts
            if Path(artifact.path).suffix == ".json"
        }
        for candidate in artifact_paths:
            try:
                encoded = candidate.read_bytes()
                payload = json.loads(encoded)
                canonical = json.dumps(
                    payload,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
            except (OSError, ValueError):
                continue
            if canonical != encoded:
                continue
            digest = hashlib.sha256(canonical).hexdigest()
            payloads.setdefault(digest, []).append(payload)
        return payloads

    @staticmethod
    def _unique_prefixed_artifact(
        state: DailyWorkflowState,
        prefix: str,
    ) -> Path:
        paths = tuple(
            Path(artifact.path).resolve()
            for stage in state.stages
            for artifact in stage.output_artifacts
            if Path(artifact.path).name.startswith(prefix) and Path(artifact.path).suffix == ".json"
        )
        if len(paths) != 1:
            raise ValueError(f"workflow must capture exactly one {prefix} artifact")
        return paths[0]

    @staticmethod
    def _verify_paper_reconciliation(
        *,
        state: DailyWorkflowState,
        frozen: FrozenDailyOrders,
        evidence: tuple[NbboReplayEvidence, ...],
    ) -> None:
        stage = Phase6DailyReportVerifier._stage(
            state,
            WorkflowStage.RECONCILE_SESSION,
        )
        if stage.status is not StageStatus.SUCCEEDED:
            raise ValueError("paper reconciliation stage is not complete")
        paths = tuple(
            Path(item.path).resolve()
            for item in stage.output_artifacts
            if Path(item.path).name.startswith("paper-reconciliation-")
            and Path(item.path).suffix == ".json"
        )
        if len(paths) != 1:
            raise ValueError("workflow must capture exactly one paper reconciliation revision")
        report = PaperReconciliationReport.load(paths[0])
        if report.session_date != frozen.trade_date:
            raise ValueError("paper reconciliation and frozen-order dates differ")
        expected_hashes = tuple(sorted(item.sha256 for item in evidence))
        if report.replay_evidence_sha256 != expected_hashes:
            raise ValueError("paper reconciliation is not bound to captured replay evidence")
        if report.reconciliation_break_count or not report.all_orders_terminal:
            raise ValueError("paper reconciliation has an unresolved operational break")
        replay_by_id = {item.result.order.order_id: item.result for item in evidence}
        paper_by_id = {item.client_order_id: item for item in report.orders}
        if set(paper_by_id) != set(replay_by_id):
            raise ValueError("paper reconciliation order identities differ from replay evidence")
        for order_id, replay in replay_by_id.items():
            paper = paper_by_id[order_id]
            if (
                paper.symbol != replay.order.ticker
                or paper.side != replay.order.side
                or paper.intended_quantity != replay.order.quantity
                or paper.replay_filled_quantity != replay.filled_qty
                or paper.replay_fill_price != replay.fill_price
            ):
                raise ValueError("paper reconciliation fields differ from replay evidence")
        observation_paths = tuple(
            Path(item.path).resolve()
            for item in stage.output_artifacts
            if Path(item.path).suffix == ".json" and Path(item.path).resolve() != paths[0]
        )
        if len(observation_paths) != len(report.orders):
            raise ValueError("workflow paper reconciliation lacks exact raw broker observations")
        broker_orders = tuple(
            Phase6DailyReportVerifier._load_raw_broker_observation(path)
            for path in observation_paths
        )
        reproduced = PaperOrderReconciler().evaluate(
            evidence=evidence,
            broker_orders=broker_orders,
            session_date=report.session_date,
            evaluated_at=report.evaluated_at,
        )
        if reproduced.canonical_bytes != report.canonical_bytes:
            raise ValueError("paper reconciliation differs from raw broker observations")

    @staticmethod
    def _load_raw_broker_observation(path: Path) -> BrokerOrder:
        encoded = path.read_bytes()
        raw = json.loads(encoded)
        if (
            json.dumps(
                raw,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
            != encoded
        ):
            raise ValueError("raw broker observation is not canonical")
        return BrokerOrder.model_validate(raw)

    @staticmethod
    def _reproduce_order_evidence(
        *,
        state: DailyWorkflowState,
        replay_stage: StageRecord,
        manifest: ReplayMaterializationManifest,
        captured_index: ReplayEvidenceIndex,
        frozen: FrozenDailyOrders,
        strategy: EarningsStrategyConfig,
    ) -> tuple[NbboReplayEvidence, ...]:
        spec_paths: dict[str, Path] = {}
        for artifact in replay_stage.output_artifacts:
            path = Path(artifact.path).resolve()
            if path.name.startswith("replay-spec-") and path.suffix == ".json":
                if path.name in spec_paths:
                    raise ValueError("workflow captured duplicate replay spec filenames")
                spec_paths[path.name] = path
        expected_names = {item.file_name for item in manifest.artifacts}
        if set(spec_paths) != expected_names:
            raise ValueError("workflow replay specs differ from their materialization manifest")
        capture_stage = Phase6DailyReportVerifier._stage(
            state,
            WorkflowStage.CAPTURE_MARKET_EVENTS,
        )
        quote_files = tuple(
            Path(item.path).resolve()
            for item in capture_stage.output_artifacts
            if "dataset=nbbo-quotes" in Path(item.path).parts
        )
        trade_files = tuple(
            Path(item.path).resolve()
            for item in capture_stage.output_artifacts
            if "dataset=stock-trades" in Path(item.path).parts
        )
        symbols = tuple(item.ticker for item in frozen.decision_snapshots)
        sources = replay_sources_from_files(
            quote_files=quote_files,
            trade_files=trade_files,
            expected_symbols=symbols,
        )
        materialization_spec = ReplayMaterializationSpec(
            orders=frozen.intended_orders,
            decision_snapshots=frozen.decision_snapshots,
            event_sources=sources,
            config=ReplayConfigSpec(
                market_impact_bps_coefficient=strategy.costs.market_impact_coef_bps
            ),
        )
        with TemporaryDirectory(prefix="qee-phase6-replay-") as temporary:
            root = Path(temporary)
            spec_directory = root / "specs"
            reproduced_manifest = ReplaySpecMaterializer().materialize(
                materialization_spec,
                output_dir=spec_directory,
                manifest_output=root / "manifest.json",
            )
            if reproduced_manifest.canonical_bytes != manifest.canonical_bytes:
                raise ValueError(
                    "replay materialization manifest differs from captured market sources"
                )
            for file_name, captured_path in spec_paths.items():
                if (spec_directory / file_name).read_bytes() != captured_path.read_bytes():
                    raise ValueError(
                        "replay spec differs from captured market-source reconstruction"
                    )
            evidence_directory = root / "evidence"
            reproduced_index = ReplayManifestRunner().run(
                reproduced_manifest,
                spec_directory=spec_directory,
                output_directory=evidence_directory,
                index_output=root / "index.json",
            )
            if reproduced_index.canonical_bytes != captured_index.canonical_bytes:
                raise ValueError("replay evidence index differs from independent reconstruction")
            return tuple(
                NbboReplayEvidence.load(evidence_directory / file_name)
                for file_name in reproduced_index.evidence_files
            )
