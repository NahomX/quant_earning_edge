"""Independent reconstruction of daily Phase 6 replay reports."""

from __future__ import annotations

import json
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
from quant_earning_edge.live import PaperReconciliationReport
from quant_earning_edge.orchestration.workflow import (
    DailyWorkflowStore,
    StageStatus,
    WorkflowStage,
)

if TYPE_CHECKING:
    from quant_earning_edge.orchestration.workflow import (
        ArtifactReference,
        DailyWorkflowState,
        StageRecord,
    )
    from quant_earning_edge.signals.config import EarningsStrategyConfig
    from quant_earning_edge.signals.live_orders import FrozenDailyOrders


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
        from quant_earning_edge.signals.live_orders import FrozenDailyOrders  # noqa: PLC0415

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
