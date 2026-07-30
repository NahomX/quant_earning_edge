"""Deterministic point-in-time construction of the Phase 3 momentum baseline."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from datetime import date
from itertools import pairwise
from pathlib import Path  # noqa: TC003 - Pydantic resolves runtime fields.
from typing import TYPE_CHECKING, Any, Self

import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import BaseModel, ConfigDict, Field, model_validator

from quant_earning_edge.backtest import BacktestSpec
from quant_earning_edge.data.split_source import (
    SplitHistorySourceCapture,
    SplitHistorySourceManifest,
)
from quant_earning_edge.data.split_vintage import causally_adjust_daily_bar_rows
from quant_earning_edge.momentum import CrossSectionalMomentum, MomentumPrice

if TYPE_CHECKING:
    from collections.abc import Sequence

    from quant_earning_edge.data.calendar import SessionFile

HISTORICAL_SPY_MEMBERSHIP_SCHEMA = pa.schema(
    [
        pa.field("symbol", pa.string(), nullable=False),
        pa.field("effective_from", pa.date32(), nullable=False),
        pa.field("effective_through", pa.date32(), nullable=False),
    ]
)


class MomentumBaselineBuildSpec(BaseModel):
    """Precommitted methodology for one historical momentum reproduction."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int = 1
    signal_start_date: date
    signal_end_date: date
    lookback_sessions: int = Field(default=60, ge=60, le=60)
    adv_sessions: int = Field(default=20, ge=20, le=20)
    selection_fraction: float = Field(default=0.1, gt=0, le=0.5)
    rebalance_interval_sessions: int = Field(default=1, ge=1, le=63)
    holding_sessions: int = Field(default=1, ge=1, le=63)
    gross_exposure: float = Field(default=1.0, gt=0, le=1)
    initial_cash: float = Field(default=100_000.0, gt=0)

    @model_validator(mode="after")
    def validate_contract(self) -> MomentumBaselineBuildSpec:
        if self.schema_version != 1:
            raise ValueError("momentum build schema_version must be 1")
        if self.signal_end_date < self.signal_start_date:
            raise ValueError("momentum signal end precedes start")
        return self

    @property
    def canonical_bytes(self) -> bytes:
        return json.dumps(
            self.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
        ).encode()

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes).hexdigest()

    @classmethod
    def load(cls, path: Path) -> Self:
        encoded = path.read_bytes()
        spec = cls.model_validate_json(encoded)
        if encoded not in {
            spec.canonical_bytes,
            spec.canonical_bytes + b"\n",
            spec.canonical_bytes + b"\r\n",
        }:
            raise ValueError("momentum build specification is not canonical")
        return spec


@dataclass(frozen=True)
class MomentumBaselineBuild:
    """Generated BacktestSpec plus complete source identities."""

    trade_plan: BacktestSpec
    build_spec_sha256: str
    session_file_sha256: str
    universe_artifact_sha256: str
    daily_bar_sha256: tuple[str, ...]
    split_source_sha256: str | None

    @property
    def trade_plan_bytes(self) -> bytes:
        return json.dumps(
            self.trade_plan.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
        ).encode()

    @property
    def trade_plan_sha256(self) -> str:
        return hashlib.sha256(self.trade_plan_bytes).hexdigest()

    def write_trade_plan(self, output: Path) -> None:
        _write_immutable(output, self.trade_plan_bytes, kind="momentum trade plan")


@dataclass(frozen=True)
class _Membership:
    symbol: str
    effective_from: date
    effective_through: date


@dataclass(frozen=True)
class _Bar:
    symbol: str
    session_date: date
    open: float
    close: float
    volume: float


class MomentumBaselineBuilder:
    """Build causal signals, next-open executions, and a frozen daily ledger."""

    def build(  # noqa: PLR0915 - one atomic causal construction boundary.
        self,
        *,
        spec: MomentumBaselineBuildSpec,
        calendar: SessionFile,
        universe_artifact: Path,
        daily_bar_files: Sequence[Path],
        split_source_manifest: Path | None = None,
    ) -> MomentumBaselineBuild:
        if not daily_bar_files:
            raise ValueError("momentum baseline requires daily bar files")
        dates = tuple(item.session_date for item in calendar.sessions)
        try:
            start_index = dates.index(spec.signal_start_date)
            end_index = dates.index(spec.signal_end_date)
        except ValueError as error:
            raise ValueError("momentum signal boundary is absent from session file") from error
        if start_index < spec.lookback_sessions:
            raise ValueError("session file lacks the momentum lookback window")
        final_index = end_index + 1 + spec.holding_sessions
        if final_index >= len(dates):
            raise ValueError("session file lacks post-signal execution sessions")

        memberships = _load_memberships(universe_artifact)
        bars, split_source_sha256 = _load_bars(
            daily_bar_files,
            basis_date=dates[final_index],
            split_source_manifest=split_source_manifest,
        )
        model = CrossSectionalMomentum(selection_fraction=spec.selection_fraction)
        trade_rows: list[dict[str, Any]] = []
        mark_keys: set[tuple[str, date]] = set()
        concurrent_cohorts = math.ceil(spec.holding_sessions / spec.rebalance_interval_sessions)
        cohort_notional = spec.initial_cash * spec.gross_exposure / concurrent_cohorts

        for asof_index in range(
            start_index,
            end_index + 1,
            spec.rebalance_interval_sessions,
        ):
            asof_date = dates[asof_index]
            history_dates = dates[asof_index - spec.lookback_sessions : asof_index + 1]
            active = tuple(
                item.symbol
                for item in memberships
                if item.effective_from <= asof_date <= item.effective_through
            )
            prices = tuple(
                MomentumPrice(
                    symbol=symbol,
                    session_date=session_date,
                    close=bars[(symbol, session_date)].close,
                )
                for symbol in active
                if all((symbol, session) in bars for session in history_dates)
                for session_date in history_dates
            )
            signals = model.generate(prices, asof_date=asof_date)
            selected = tuple(item for item in signals if item.side)
            selected_per_side = sum(item.side == 1 for item in selected)
            if selected_per_side < 1 or selected_per_side != sum(
                item.side == -1 for item in selected
            ):
                raise ValueError("momentum selection is not balanced long/short")
            target_notional = cohort_notional / (2 * selected_per_side)
            entry_index = asof_index + 1
            exit_index = entry_index + spec.holding_sessions
            entry_date = dates[entry_index]
            exit_date = dates[exit_index]
            for signal in selected:
                symbol = signal.symbol
                entry = _required_bar(bars, symbol=symbol, session_date=entry_date)
                exit_bar = _required_bar(bars, symbol=symbol, session_date=exit_date)
                history = tuple(
                    _required_bar(bars, symbol=symbol, session_date=item)
                    for item in history_dates[-spec.adv_sessions :]
                )
                average_volume = sum(item.volume for item in history) / len(history)
                if average_volume <= 0:
                    raise ValueError(f"{symbol} has non-positive causal ADV")
                sizing_close = _required_bar(
                    bars,
                    symbol=symbol,
                    session_date=asof_date,
                ).close
                shares = math.floor(target_notional / sizing_close)
                if shares < 1:
                    raise ValueError(f"{symbol} momentum target rounds to zero shares")
                identity = (
                    f"{spec.sha256}|{asof_date}|{symbol}|{signal.side}|"
                    f"{entry_date}|{exit_date}|{shares}"
                )
                trade_rows.append(
                    {
                        "trade_id": (
                            f"momentum-{hashlib.sha256(identity.encode()).hexdigest()[:20]}"
                        ),
                        "symbol": symbol,
                        "side": "long" if signal.side == 1 else "short",
                        "entry_date": entry_date,
                        "exit_date": exit_date,
                        "shares": shares,
                        "entry_price": entry.open,
                        "exit_price": exit_bar.close,
                        "entry_average_daily_volume_shares": average_volume,
                        "exit_average_daily_volume_shares": average_volume,
                        "holding_sessions": spec.holding_sessions,
                    }
                )
                mark_keys.update(
                    (symbol, dates[index]) for index in range(entry_index, exit_index + 1)
                )

        if not trade_rows:
            raise ValueError("momentum build produced no trades")
        plan = BacktestSpec.model_validate(
            {
                "initial_cash": spec.initial_cash,
                "sessions": dates[start_index + 1 : final_index + 1],
                "marks": [
                    {
                        "symbol": symbol,
                        "session_date": session_date,
                        "close": _required_bar(
                            bars,
                            symbol=symbol,
                            session_date=session_date,
                        ).close,
                    }
                    for symbol, session_date in sorted(
                        mark_keys, key=lambda item: (item[1], item[0])
                    )
                ],
                "trades": sorted(
                    trade_rows,
                    key=lambda item: (
                        item["entry_date"],
                        item["symbol"],
                        item["side"],
                    ),
                ),
            }
        )
        return MomentumBaselineBuild(
            trade_plan=plan,
            build_spec_sha256=spec.sha256,
            session_file_sha256=calendar.sha256,
            universe_artifact_sha256=_file_sha256(universe_artifact),
            daily_bar_sha256=tuple(sorted(_file_sha256(path) for path in daily_bar_files)),
            split_source_sha256=split_source_sha256,
        )


def _load_memberships(path: Path) -> tuple[_Membership, ...]:
    table = pq.read_table(path)  # type: ignore[no-untyped-call]
    if table.schema != HISTORICAL_SPY_MEMBERSHIP_SCHEMA:
        raise ValueError("historical SPY membership schema differs")
    rows = tuple(
        _Membership(
            symbol=str(row["symbol"]).strip().upper(),
            effective_from=row["effective_from"],
            effective_through=row["effective_through"],
        )
        for row in table.to_pylist()
    )
    if not rows or any(
        not item.symbol or item.effective_through < item.effective_from for item in rows
    ):
        raise ValueError("historical SPY membership interval is invalid")
    by_symbol: dict[str, list[_Membership]] = {}
    for item in rows:
        by_symbol.setdefault(item.symbol, []).append(item)
    for symbol, intervals in by_symbol.items():
        ordered = sorted(intervals, key=lambda item: item.effective_from)
        if any(
            current.effective_from <= prior.effective_through
            for prior, current in pairwise(ordered)
        ):
            raise ValueError(f"historical SPY membership intervals overlap for {symbol}")
    return tuple(
        sorted(rows, key=lambda item: (item.effective_from, item.symbol, item.effective_through))
    )


def _load_bars(
    paths: Sequence[Path],
    *,
    basis_date: date,
    split_source_manifest: Path | None,
) -> tuple[dict[tuple[str, date], _Bar], str | None]:
    output: dict[tuple[str, date], _Bar] = {}
    required = {"session_date", "symbol", "open", "close", "volume", "adjusted"}
    rows: list[dict[str, Any]] = []
    for path in sorted(paths):
        table = pq.read_table(path)  # type: ignore[no-untyped-call]
        if not required.issubset(table.column_names):
            raise ValueError(f"daily bar artifact is missing columns: {path}")
        rows.extend(
            row
            for row in table.select(sorted(required)).to_pylist()
            if row["session_date"] <= basis_date
        )
    modes = {bool(row["adjusted"]) for row in rows}
    if len(modes) != 1:
        raise ValueError("momentum daily bars mix split-adjustment modes")
    adjusted = modes.pop() if modes else True
    split_source_sha256 = None
    if not adjusted:
        if split_source_manifest is None:
            raise ValueError("raw momentum bars require a complete split-history source")
        source = SplitHistorySourceManifest.load(split_source_manifest)
        root = source.data_lake_root
        if (
            date.fromisoformat(source.raw["start_date"]) > min(row["session_date"] for row in rows)
            or date.fromisoformat(source.raw["end_date"]) < basis_date
        ):
            raise ValueError("momentum split history does not cover the evaluation interval")
        SplitHistorySourceCapture.reproduce(source, data_lake_root=root)
        rows = list(
            causally_adjust_daily_bar_rows(
                rows,
                splits=source.splits(data_lake_root=root),
                basis_date=basis_date,
            )
        )
        split_source_sha256 = _file_sha256(source.path)
    elif split_source_manifest is not None:
        raise ValueError("split-history source is only valid with raw momentum bars")
    for row in rows:
        if row["adjusted"] is not True:
            raise ValueError("momentum baseline requires split-normalized daily bars")
        bar = _Bar(
            symbol=str(row["symbol"]).strip().upper(),
            session_date=row["session_date"],
            open=float(row["open"]),
            close=float(row["close"]),
            volume=float(row["volume"]),
        )
        if (
            not bar.symbol
            or min(bar.open, bar.close) <= 0
            or not all(math.isfinite(item) for item in (bar.open, bar.close, bar.volume))
            or bar.volume < 0
        ):
            raise ValueError("momentum daily bar is invalid")
        key = (bar.symbol, bar.session_date)
        if key in output:
            raise ValueError(f"duplicate momentum daily bar for {bar.symbol} {bar.session_date}")
        output[key] = bar
    return output, split_source_sha256


def _required_bar(
    bars: dict[tuple[str, date], _Bar],
    *,
    symbol: str,
    session_date: date,
) -> _Bar:
    try:
        return bars[(symbol, session_date)]
    except KeyError as error:
        raise ValueError(f"missing {symbol} momentum bar for {session_date}") from error


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_immutable(path: Path, encoded: bytes, *, kind: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as output:
            output.write(encoded)
    except FileExistsError:
        if path.read_bytes() != encoded:
            raise RuntimeError(f"{kind} collision at {path}") from None
