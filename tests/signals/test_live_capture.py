"""Provider-backed live-source assembly removes hand-authored market fields."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import httpx
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from typer.testing import CliRunner

import quant_earning_edge.cli as cli_module
from quant_earning_edge.backtest import NbboReplayEvidence, NbboReplaySpec, replay_order
from quant_earning_edge.cli import app
from quant_earning_edge.data import LakehouseLayout, SessionFileStore
from quant_earning_edge.data.clients import MarketSession, TickerSnapshot
from quant_earning_edge.evaluation import ReplayRoundTrip, ReplaySessionAggregator
from quant_earning_edge.live import PaperAccountSnapshot
from quant_earning_edge.signals import (
    LiveSourceCaptureArtifact,
    LiveSourceCaptureAssembler,
)
from quant_earning_edge.universe import EVENT_CANDIDATE_SCHEMA

TRADE_DATE = date(2026, 7, 28)
ASOF_DATE = date(2026, 7, 27)
CAPTURED_AT = datetime(2026, 7, 28, 1, 30, tzinfo=UTC)


def _calendar(tmp_path: Path) -> Path:
    return (
        SessionFileStore(LakehouseLayout(tmp_path / "calendar"))
        .write(
            (
                MarketSession(
                    session_date=ASOF_DATE,
                    open_at=datetime(2026, 7, 27, 13, 30, tzinfo=UTC),
                    close_at=datetime(2026, 7, 27, 20, 0, tzinfo=UTC),
                ),
                MarketSession(
                    session_date=TRADE_DATE,
                    open_at=datetime(2026, 7, 28, 13, 30, tzinfo=UTC),
                    close_at=datetime(2026, 7, 28, 20, 0, tzinfo=UTC),
                ),
            )
        )
        .path
    )


def _candidates(tmp_path: Path) -> Path:
    path = tmp_path / "candidates.parquet"
    pq.write_table(  # type: ignore[no-untyped-call]
        pa.Table.from_pylist(
            [
                {
                    "trade_date": TRADE_DATE,
                    "asof_date": ASOF_DATE,
                    "decision_at": datetime(2026, 7, 28, 1, 0, tzinfo=UTC),
                    "symbol": "AAPL",
                    "sector": "MANUFACTURING",
                    "sizing_price": 100.0,
                    "frozen_average_daily_volume_shares": 2_000_000.0,
                    "event_date": ASOF_DATE,
                    "timing": "amc",
                    "year": 2026,
                    "quarter": 3,
                    "eps_estimate": 1.25,
                    "revenue_estimate": 100.0,
                    "split_event_ids": [],
                    "dividend_event_ids": [],
                    "universe_snapshot_sha256": "a" * 64,
                    "session_file_sha256": "b" * 64,
                    "earnings_input_sha256": "c" * 64,
                    "corporate_actions_input_sha256": "d" * 64,
                }
            ],
            schema=EVENT_CANDIDATE_SCHEMA,
        ),
        path,
    )
    return path


def _account() -> PaperAccountSnapshot:
    return PaperAccountSnapshot(
        captured_at=CAPTURED_AT,
        equity=100_123.45,
        buying_power=200_246.90,
        status="ACTIVE",
        trading_blocked=False,
        payload_sha256="e" * 64,
        provider_request_id="account-request",
    )


def _snapshot(symbol: str = "AAPL") -> TickerSnapshot:
    return TickerSnapshot(
        symbol=symbol,
        captured_at=CAPTURED_AT,
        observed_at=datetime(2026, 7, 28, 1, 29, 58, tzinfo=UTC),
        bid_price=99.9,
        ask_price=100.1,
        bid_size=150,
        ask_size=200,
        last_trade_price=100,
        last_trade_at=datetime(2026, 7, 28, 1, 29, 57, tzinfo=UTC),
        payload_sha256="f" * 64,
        request_id="snapshot-request",
    )


def _filled_replay_evidence(
    *,
    order_id: str,
    side: str,
    submitted_at: datetime,
    bid: float,
    ask: float,
) -> NbboReplayEvidence:
    decision = datetime(2026, 7, 26, 21, 30, tzinfo=UTC)
    spec = NbboReplaySpec.model_validate(
        {
            "order": {
                "order_id": order_id,
                "ticker": "AAPL",
                "side": side,
                "quantity": 100,
                "decision_time": decision,
                "submitted_at": submitted_at,
                "expires_at": submitted_at + timedelta(minutes=5),
                "average_daily_volume_shares": 1_000_000,
            },
            "decision_snapshot": {
                "ticker": "AAPL",
                "observed_at": decision,
                "bid_price": 99.9,
                "ask_price": 100.1,
                "bid_size": 100,
                "ask_size": 100,
                "last_trade_price": 100,
                "last_trade_at": decision - timedelta(seconds=1),
            },
            "quotes": [
                {
                    "ticker": "AAPL",
                    "timestamp": submitted_at,
                    "sequence": 1,
                    "bid_price": bid,
                    "ask_price": ask,
                    "bid_size": 100,
                    "ask_size": 100,
                }
            ],
        }
    )
    order, snapshot, quotes, trades, config = spec.domain_inputs()
    return NbboReplayEvidence.build(
        spec=spec,
        result=replay_order(
            order,
            decision_snapshot=snapshot,
            quotes=quotes,
            trades=trades,
            config=config,
        ),
    )


def test_capture_builds_schedule_and_excludes_no_trade_session_from_kelly(
    tmp_path: Path,
) -> None:
    candidate_file = _candidates(tmp_path)
    session_file = _calendar(tmp_path)
    replay = ReplaySessionAggregator().evaluate(
        evidence=(),
        round_trips=(),
        session_date=ASOF_DATE,
        initial_cash=100_000,
    )
    replay_path = tmp_path / "prior-replay.json"
    replay.write(replay_path)

    artifact = LiveSourceCaptureAssembler().assemble(
        trade_date=TRADE_DATE,
        captured_at=CAPTURED_AT,
        candidate_file=candidate_file,
        session_file=session_file,
        account=_account(),
        initial_cash=100_000,
        snapshots=(_snapshot(),),
        prior_replay_files=(replay_path,),
    )
    source_path = tmp_path / "source.json"
    evidence_path = tmp_path / "evidence.json"
    LiveSourceCaptureAssembler.write(
        artifact,
        source_output=source_path,
        evidence_output=evidence_path,
    )

    source = artifact.source
    assert source.equity == 100_000
    assert artifact.paper_account_equity == 100_123.45
    assert source.entry_submitted_at == datetime(2026, 7, 28, 13, 30, tzinfo=UTC)
    assert source.exit_submitted_at == datetime(2026, 7, 28, 19, 50, tzinfo=UTC)
    assert source.observations[0].sector == "MANUFACTURING"
    assert source.observations[0].sizing_price == 100
    assert source.observations[0].decision_snapshot.bid_price == 99.9
    assert source.outcomes == ()
    assert artifact.candidate_file_sha256 == hashlib.sha256(candidate_file.read_bytes()).hexdigest()
    assert json.loads(source_path.read_bytes())["observations"][0]["symbol"] == "AAPL"
    assert evidence_path.read_bytes() == artifact.canonical_bytes


def test_capture_uses_realized_round_trip_return_for_kelly(tmp_path: Path) -> None:
    entry = _filled_replay_evidence(
        order_id="entry",
        side="buy",
        submitted_at=datetime(2026, 7, 27, 13, 30, tzinfo=UTC),
        bid=99.9,
        ask=100.1,
    )
    exit_evidence = _filled_replay_evidence(
        order_id="exit",
        side="sell",
        submitted_at=datetime(2026, 7, 27, 19, 55, tzinfo=UTC),
        bid=104.9,
        ask=105.1,
    )
    replay = ReplaySessionAggregator().evaluate(
        evidence=(entry, exit_evidence),
        round_trips=(ReplayRoundTrip("trade-1", "entry", "exit", "long"),),
        session_date=ASOF_DATE,
        initial_cash=100_000,
    )
    replay_path = tmp_path / "prior-replay.json"
    replay.write(replay_path)

    artifact = LiveSourceCaptureAssembler().assemble(
        trade_date=TRADE_DATE,
        captured_at=CAPTURED_AT,
        candidate_file=_candidates(tmp_path),
        session_file=_calendar(tmp_path),
        account=_account(),
        initial_cash=100_000,
        snapshots=(_snapshot(),),
        prior_replay_files=(replay_path,),
    )

    round_trip = replay.round_trips[0]
    assert round_trip.entry_fill_price is not None
    expected_return = round_trip.net_pnl_on_matched_quantity / (
        round_trip.entry_fill_price * round_trip.matched_quantity
    )
    assert len(artifact.source.outcomes) == 1
    assert artifact.source.outcomes[0].closed_date == ASOF_DATE
    assert artifact.source.outcomes[0].net_return == pytest.approx(expected_return)


def test_capture_rejects_snapshot_symbol_mismatch(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="symbol sets differ"):
        LiveSourceCaptureAssembler().assemble(
            trade_date=TRADE_DATE,
            captured_at=CAPTURED_AT,
            candidate_file=_candidates(tmp_path),
            session_file=_calendar(tmp_path),
            account=_account(),
            initial_cash=100_000,
            snapshots=(_snapshot("MSFT"),),
            prior_replay_files=(),
        )


def test_capture_cli_fetches_only_paper_account_and_polygon_snapshots(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate_file = _candidates(tmp_path)
    session_file = _calendar(tmp_path)
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            (
                "POLYGON_API_KEY=polygon-key",
                "APCA_API_KEY_ID=alpaca-key",
                "APCA_API_SECRET_KEY=alpaca-secret",
                f"DATA_LAKE_ROOT={tmp_path / 'lake'}",
            )
        ),
        encoding="utf-8",
    )
    real_client = httpx.Client

    def client_factory(*args: object, **kwargs: object) -> httpx.Client:
        base_url = str(kwargs["base_url"])

        def respond(request: httpx.Request) -> httpx.Response:
            if base_url == "https://paper-api.alpaca.markets":
                assert request.url.path == "/v2/account"
                return httpx.Response(
                    200,
                    json={
                        "equity": "100123.45",
                        "buying_power": "200246.90",
                        "status": "ACTIVE",
                        "trading_blocked": False,
                    },
                )
            assert request.url.path.endswith("/AAPL")
            return httpx.Response(
                200,
                json={
                    "status": "OK",
                    "ticker": {
                        "ticker": "AAPL",
                        "lastQuote": {
                            "P": 100.1,
                            "S": 200,
                            "p": 99.9,
                            "s": 150,
                            "t": int(
                                datetime(2026, 7, 28, 1, 29, 58, tzinfo=UTC).timestamp()
                                * 1_000_000_000
                            ),
                        },
                        "lastTrade": {
                            "p": 100,
                            "t": int(
                                datetime(2026, 7, 28, 1, 29, 57, tzinfo=UTC).timestamp()
                                * 1_000_000_000
                            ),
                        },
                    },
                },
            )

        return real_client(
            *args,
            **kwargs,
            transport=httpx.MockTransport(respond),
        )

    class FixedDateTime(datetime):
        @classmethod
        def now(cls, tz: object = None) -> datetime:
            del tz
            return CAPTURED_AT

    monkeypatch.setattr(cli_module.httpx, "Client", client_factory)
    monkeypatch.setattr(cli_module, "datetime", FixedDateTime)
    source = tmp_path / "source.json"
    evidence = tmp_path / "evidence.json"

    result = CliRunner().invoke(
        app,
        [
            "model",
            "capture-live-source",
            "--trade-date",
            TRADE_DATE.isoformat(),
            "--candidate-file",
            str(candidate_file),
            "--session-file",
            str(session_file),
            "--initial-cash",
            "100000",
            "--source-output",
            str(source),
            "--evidence-output",
            str(evidence),
            "--env-file",
            str(env_file),
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["candidate_count"] == 1
    assert len(payload["provider_observation_paths"]) == 2
    assert all(Path(path).is_file() for path in payload["provider_observation_paths"])
    assert json.loads(source.read_bytes())["equity"] == 100_000
    assert json.loads(evidence.read_bytes())["paper_account_equity"] == 100_123.45
    assert LiveSourceCaptureArtifact.load(evidence).schema_version == 2
