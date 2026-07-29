"""Assemble probability-free live planning sources from authoritative artifacts."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from datetime import timedelta
from typing import TYPE_CHECKING, Any

import pyarrow.parquet as pq

from quant_earning_edge.backtest import DecisionSnapshotSpec
from quant_earning_edge.data.calendar import SessionFileStore
from quant_earning_edge.evaluation import ReplaySessionReport
from quant_earning_edge.signals.event_trades import TradeOutcomeSpec
from quant_earning_edge.signals.live_planning import (
    LiveMarketObservationSpec,
    LivePlanningSourceSpec,
)
from quant_earning_edge.universe.events import EVENT_CANDIDATE_SCHEMA

if TYPE_CHECKING:
    from collections.abc import Sequence
    from datetime import date, datetime
    from pathlib import Path

    from quant_earning_edge.data.clients import TickerSnapshot
    from quant_earning_edge.live import PaperAccountSnapshot


@dataclass(frozen=True)
class LiveSourceCaptureArtifact:
    """Content-linked source consumed by production-model live scoring."""

    schema_version: int
    initial_cash: float
    candidate_file_sha256: str
    session_file_sha256: str
    account_payload_sha256: str
    paper_account_equity: float
    snapshot_payload_sha256: tuple[str, ...]
    prior_replay_sha256: tuple[str, ...]
    source: LivePlanningSourceSpec

    def __post_init__(self) -> None:
        if self.schema_version != 2:
            raise ValueError("unsupported live source capture schema version")
        if not math.isfinite(self.initial_cash) or self.initial_cash <= 0:
            raise ValueError("live source initial cash must be finite and positive")
        for digest in (
            self.candidate_file_sha256,
            self.session_file_sha256,
            self.account_payload_sha256,
            *self.snapshot_payload_sha256,
            *self.prior_replay_sha256,
        ):
            if len(digest) != 64 or any(item not in "0123456789abcdef" for item in digest):
                raise ValueError("live source capture digest must be SHA-256")

    @property
    def canonical_bytes(self) -> bytes:
        return json.dumps(
            {
                "schema_version": self.schema_version,
                "initial_cash": float(self.initial_cash),
                "candidate_file_sha256": self.candidate_file_sha256,
                "session_file_sha256": self.session_file_sha256,
                "account_payload_sha256": self.account_payload_sha256,
                "paper_account_equity": self.paper_account_equity,
                "snapshot_payload_sha256": self.snapshot_payload_sha256,
                "prior_replay_sha256": self.prior_replay_sha256,
                "source": self.source.model_dump(mode="json"),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes).hexdigest()

    @classmethod
    def load(cls, path: Path) -> LiveSourceCaptureArtifact:
        try:
            raw = json.loads(path.read_bytes())
            artifact = cls(
                schema_version=int(raw["schema_version"]),
                initial_cash=float(raw["initial_cash"]),
                candidate_file_sha256=str(raw["candidate_file_sha256"]),
                session_file_sha256=str(raw["session_file_sha256"]),
                account_payload_sha256=str(raw["account_payload_sha256"]),
                paper_account_equity=float(raw["paper_account_equity"]),
                snapshot_payload_sha256=tuple(str(item) for item in raw["snapshot_payload_sha256"]),
                prior_replay_sha256=tuple(str(item) for item in raw["prior_replay_sha256"]),
                source=LivePlanningSourceSpec.model_validate(raw["source"]),
            )
        except (KeyError, OSError, TypeError, ValueError) as error:
            raise ValueError(f"invalid live source capture evidence: {path}") from error
        if artifact.canonical_bytes != path.read_bytes():
            raise ValueError("live source capture evidence is not canonical")
        return artifact


class LiveSourceCaptureAssembler:
    """Freeze candidates, provider observations, account equity, and outcomes."""

    @staticmethod
    def candidate_symbols(path: Path) -> tuple[str, ...]:
        """Read the exact sorted symbol set needed for provider snapshot capture."""
        if pq.read_schema(path) != EVENT_CANDIDATE_SCHEMA:  # type: ignore[no-untyped-call]
            raise ValueError("live source candidate artifact schema mismatch")
        rows = pq.ParquetFile(path).read(columns=["symbol"]).to_pylist()  # type: ignore[no-untyped-call]
        symbols = tuple(str(row["symbol"]).strip().upper() for row in rows)
        if symbols != tuple(sorted(set(symbols))):
            raise ValueError("live source candidate symbols must be unique and sorted")
        return symbols

    def assemble(
        self,
        *,
        trade_date: date,
        captured_at: datetime,
        candidate_file: Path,
        session_file: Path,
        account: PaperAccountSnapshot,
        initial_cash: float,
        snapshots: Sequence[TickerSnapshot],
        prior_replay_files: Sequence[Path],
        minimum_probability: float = 0.5,
    ) -> LiveSourceCaptureArtifact:
        if captured_at.tzinfo is None or captured_at.utcoffset() is None:
            raise ValueError("live source capture time must be timezone-aware")
        calendar = SessionFileStore.load(session_file)
        sessions = {item.session_date: item for item in calendar.sessions}
        session = sessions.get(trade_date)
        if session is None:
            raise ValueError("live source trade date is not an authoritative session")
        earlier = tuple(item for item in calendar.sessions if item.session_date < trade_date)
        if not earlier:
            raise ValueError("live source requires an authoritative prior session")
        prior = earlier[-1]
        if not prior.close_at <= captured_at < session.open_at:
            raise ValueError("live source capture must occur after prior close and before open")
        if account.captured_at != captured_at:
            raise ValueError("paper account and live source capture times differ")
        if not math.isfinite(initial_cash) or initial_cash <= 0:
            raise ValueError("live source initial proof cash must be positive")

        candidates, candidate_hash = self._load_candidates(
            candidate_file,
            trade_date=trade_date,
            feature_asof_date=prior.session_date,
            captured_at=captured_at,
        )
        by_symbol = {item.symbol: item for item in snapshots}
        expected_symbols = tuple(row["symbol"] for row in candidates)
        if tuple(sorted(by_symbol)) != expected_symbols:
            raise ValueError("candidate and provider snapshot symbol sets differ")
        observations = []
        snapshot_hashes = []
        for row in candidates:
            symbol = str(row["symbol"])
            snapshot = by_symbol[symbol]
            if snapshot.captured_at != captured_at:
                raise ValueError(f"provider snapshot capture time differs for {symbol}")
            observations.append(
                LiveMarketObservationSpec(
                    symbol=symbol,
                    sector=str(row["sector"]),
                    sizing_price=float(row["sizing_price"]),
                    sizing_price_observed_at=prior.close_at,
                    frozen_average_daily_volume_shares=float(
                        row["frozen_average_daily_volume_shares"]
                    ),
                    decision_snapshot=DecisionSnapshotSpec(
                        ticker=symbol,
                        observed_at=snapshot.observed_at,
                        bid_price=snapshot.bid_price,
                        ask_price=snapshot.ask_price,
                        bid_size=snapshot.bid_size,
                        ask_size=snapshot.ask_size,
                        last_trade_price=snapshot.last_trade_price,
                        last_trade_at=snapshot.last_trade_at,
                    ),
                )
            )
            snapshot_hashes.append(snapshot.payload_sha256)
        outcomes, replay_hashes, proof_equity = self._load_outcomes(
            prior_replay_files,
            trade_date=trade_date,
            initial_cash=initial_cash,
        )
        source = LivePlanningSourceSpec(
            trade_date=trade_date,
            feature_asof_date=prior.session_date,
            decision_at=captured_at,
            equity=proof_equity,
            observations=tuple(observations),
            outcomes=outcomes,
            entry_submitted_at=session.open_at,
            entry_expires_at=session.open_at + _minutes(5),
            exit_submitted_at=session.close_at - _minutes(10),
            exit_expires_at=session.close_at + _minutes(1),
            minimum_probability=minimum_probability,
        )
        return LiveSourceCaptureArtifact(
            schema_version=2,
            initial_cash=initial_cash,
            candidate_file_sha256=candidate_hash,
            session_file_sha256=calendar.sha256,
            account_payload_sha256=account.payload_sha256,
            paper_account_equity=account.equity,
            snapshot_payload_sha256=tuple(snapshot_hashes),
            prior_replay_sha256=replay_hashes,
            source=source,
        )

    @staticmethod
    def _load_candidates(
        path: Path,
        *,
        trade_date: date,
        feature_asof_date: date,
        captured_at: datetime,
    ) -> tuple[list[dict[str, Any]], str]:
        if pq.read_schema(path) != EVENT_CANDIDATE_SCHEMA:  # type: ignore[no-untyped-call]
            raise ValueError("live source candidate artifact schema mismatch")
        rows: list[dict[str, Any]] = pq.ParquetFile(path).read().to_pylist()  # type: ignore[no-untyped-call]
        symbols = tuple(str(row["symbol"]).strip().upper() for row in rows)
        if symbols != tuple(sorted(set(symbols))):
            raise ValueError("live source candidate symbols must be unique and sorted")
        for row in rows:
            if row["trade_date"] != trade_date:
                raise ValueError("live source candidate trade date differs")
            if row["asof_date"] != feature_asof_date:
                raise ValueError("live source candidate feature as-of date differs")
            if row["decision_at"] > captured_at:
                raise ValueError("live source candidate was generated after capture")
        return rows, hashlib.sha256(path.read_bytes()).hexdigest()

    @staticmethod
    def _load_outcomes(
        paths: Sequence[Path],
        *,
        trade_date: date,
        initial_cash: float,
    ) -> tuple[tuple[TradeOutcomeSpec, ...], tuple[str, ...], float]:
        reports = [ReplaySessionReport.load(path) for path in sorted(paths)]
        dates = tuple(item.session_date for item in reports)
        if dates != tuple(sorted(set(dates))):
            raise ValueError("prior replay sessions must be unique and chronological")
        outcomes: list[TradeOutcomeSpec] = []
        current_equity = initial_cash
        for report in reports:
            if report.session_date >= trade_date:
                raise ValueError("prior replay report does not precede live trade date")
            if report.reconciliation_break_count or report.net_return is None:
                raise ValueError("prior replay report is not cleanly reconciled")
            if not math.isclose(report.initial_cash, current_equity, abs_tol=1e-8):
                raise ValueError("prior replay equity is not chronologically chained")
            assert report.net_pnl is not None
            outcomes.extend(
                TradeOutcomeSpec(
                    closed_date=report.session_date,
                    net_return=(
                        item.net_pnl_on_matched_quantity
                        / (item.entry_fill_price * item.matched_quantity)
                    ),
                )
                for item in report.round_trips
                if item.matched_quantity > 0 and item.entry_fill_price is not None
            )
            current_equity += report.net_pnl
        return tuple(outcomes), tuple(item.sha256 for item in reports), current_equity

    @staticmethod
    def write(
        artifact: LiveSourceCaptureArtifact,
        *,
        source_output: Path,
        evidence_output: Path,
    ) -> None:
        _write_immutable(source_output, artifact.source.canonical_bytes)
        _write_immutable(evidence_output, artifact.canonical_bytes)


def _minutes(value: int) -> timedelta:
    return timedelta(minutes=value)


def _write_immutable(path: Path, encoded: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as destination:
            destination.write(encoded)
    except FileExistsError:
        if path.read_bytes() != encoded:
            raise RuntimeError(f"live source capture collision at {path}") from None
