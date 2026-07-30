"""After-close diagnostic reconciliation of paper orders to NBBO replay."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from datetime import date, datetime  # noqa: TC003 - Pydantic resolves runtime annotations.
from pathlib import Path  # noqa: TC003 - Pydantic resolves runtime annotations.
from typing import TYPE_CHECKING, Literal
from zoneinfo import ZoneInfo

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from quant_earning_edge.live.alpaca_paper import (  # noqa: TC001 - Pydantic resolves annotations.
    BrokerOrder,
)

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Protocol

    from quant_earning_edge.backtest import IntendedOrder, NbboReplayEvidence

    class PaperOrderLookup(Protocol):
        """Read-only broker boundary used by automated reconciliation."""

        def get_by_client_order_id(self, client_order_id: str) -> BrokerOrder: ...


_MARKET_TIMEZONE = ZoneInfo("America/New_York")
_TERMINAL_STATUSES = frozenset({"filled", "canceled", "expired", "rejected"})


@dataclass(frozen=True)
class PaperOrderReconciliation:
    """One paper order compared with the corresponding replay result."""

    client_order_id: str
    symbol: str
    side: str
    intended_quantity: int
    replay_filled_quantity: int
    replay_fill_price: float | None
    paper_status: str
    paper_filled_quantity: int
    paper_fill_price: float | None
    fill_quantity_difference: int
    paper_vs_replay_price_bps: float | None
    terminal: bool
    break_reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.client_order_id.strip() or not self.symbol.strip():
            raise ValueError("paper reconciliation identities must not be blank")
        if self.symbol != self.symbol.strip().upper() or self.side not in {"buy", "sell"}:
            raise ValueError("paper reconciliation symbol or side is invalid")
        if (
            self.intended_quantity <= 0
            or not 0 <= self.replay_filled_quantity <= self.intended_quantity
            or not 0 <= self.paper_filled_quantity <= self.intended_quantity
        ):
            raise ValueError("paper reconciliation quantities are invalid")
        if (
            self.fill_quantity_difference
            != self.paper_filled_quantity - self.replay_filled_quantity
        ):
            raise ValueError("paper reconciliation fill difference is inconsistent")


@dataclass(frozen=True)
class PaperReconciliationReport:
    """Immutable smoke-test evidence; paper results never enter strategy gates."""

    schema_version: int
    session_date: date
    evaluated_at: datetime
    replay_evidence_sha256: tuple[str, ...]
    orders: tuple[PaperOrderReconciliation, ...]
    reconciliation_break_count: int
    all_orders_terminal: bool
    paper_pnl_is_gate_input: bool

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("unsupported paper reconciliation schema")
        if self.evaluated_at.tzinfo is None or self.evaluated_at.utcoffset() is None:
            raise ValueError("reconciliation evaluated_at must be timezone-aware")
        if self.paper_pnl_is_gate_input:
            raise ValueError("paper P&L must never be a strategy gate input")
        breaks = sum(bool(item.break_reasons) for item in self.orders)
        if breaks != self.reconciliation_break_count:
            raise ValueError("reconciliation break count is inconsistent")
        if self.all_orders_terminal != all(item.terminal for item in self.orders):
            raise ValueError("all-orders-terminal flag is inconsistent")
        hashes = self.replay_evidence_sha256
        if hashes != tuple(sorted(set(hashes))) or any(
            len(item) != 64 or any(character not in "0123456789abcdef" for character in item)
            for item in hashes
        ):
            raise ValueError("paper reconciliation evidence hashes are invalid")
        order_ids = tuple(item.client_order_id for item in self.orders)
        if order_ids != tuple(sorted(set(order_ids))):
            raise ValueError("paper reconciliation orders must be unique and sorted")

    @property
    def canonical_bytes(self) -> bytes:
        return json.dumps(
            asdict(self),
            default=lambda item: item.isoformat(),
            sort_keys=True,
            separators=(",", ":"),
        ).encode()

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes).hexdigest()

    def write(self, output: Path) -> None:
        encoded = self.canonical_bytes
        output.parent.mkdir(parents=True, exist_ok=True)
        try:
            with output.open("xb") as destination:
                destination.write(encoded)
        except FileExistsError:
            if output.read_bytes() != encoded:
                raise RuntimeError(f"paper reconciliation collision at {output}") from None

    @classmethod
    def load(cls, path: Path) -> PaperReconciliationReport:
        """Reload canonical reconciliation evidence for rolling controls."""
        try:
            raw = json.loads(path.read_bytes())
            report = _PaperReconciliationReportSpec.model_validate(raw).to_domain()
        except (
            json.JSONDecodeError,
            OSError,
            TypeError,
            ValidationError,
            ValueError,
        ) as error:
            raise ValueError(f"invalid paper reconciliation report: {path}") from error
        if json.loads(report.canonical_bytes) != raw:
            raise ValueError("paper reconciliation report is not canonical")
        return report


class PaperOrderReconciler:
    """Require exact paper/replay identities and retain divergence as diagnostics."""

    def fetch_and_evaluate(
        self,
        *,
        evidence: Sequence[NbboReplayEvidence],
        intended_orders: Sequence[IntendedOrder],
        order_lookup: PaperOrderLookup,
        session_date: date,
        evaluated_at: datetime,
    ) -> PaperReconciliationReport:
        """Verify frozen identities, fetch exact broker orders, and reconcile them."""
        intended_ids = tuple(item.order_id for item in intended_orders)
        if intended_ids != tuple(sorted(set(intended_ids))):
            raise ValueError("frozen intended order ids must be unique and sorted")
        replay_by_id = {item.result.order.order_id: item.result.order for item in evidence}
        if len(replay_by_id) != len(evidence) or set(replay_by_id) != set(intended_ids):
            raise ValueError("replay evidence does not exactly match frozen intended order ids")
        intended_by_id = {item.order_id: item for item in intended_orders}
        if any(replay_by_id[order_id] != intended_by_id[order_id] for order_id in intended_ids):
            raise ValueError("replay evidence order fields differ from frozen intended orders")
        broker_orders = tuple(
            order_lookup.get_by_client_order_id(order_id) for order_id in intended_ids
        )
        return self.evaluate(
            evidence=evidence,
            broker_orders=broker_orders,
            session_date=session_date,
            evaluated_at=evaluated_at,
        )

    def evaluate(
        self,
        *,
        evidence: Sequence[NbboReplayEvidence],
        broker_orders: Sequence[BrokerOrder],
        session_date: date,
        evaluated_at: datetime,
    ) -> PaperReconciliationReport:
        if evaluated_at.tzinfo is None or evaluated_at.utcoffset() is None:
            raise ValueError("evaluated_at must be timezone-aware")
        by_client_id = {item.client_order_id: item for item in broker_orders}
        if len(by_client_id) != len(broker_orders):
            raise ValueError("paper orders contain duplicate client_order_id values")
        by_order_id = {item.result.order.order_id: item for item in evidence}
        if len(by_order_id) != len(evidence):
            raise ValueError("replay evidence contains duplicate order ids")
        if set(by_client_id) != set(by_order_id):
            raise ValueError("paper client ids do not exactly match replay order ids")
        reconciliations = tuple(
            self._order(
                evidence=by_order_id[order_id],
                broker_order=by_client_id[order_id],
                session_date=session_date,
            )
            for order_id in sorted(by_order_id)
        )
        return PaperReconciliationReport(
            schema_version=1,
            session_date=session_date,
            evaluated_at=evaluated_at,
            replay_evidence_sha256=tuple(sorted(item.sha256 for item in evidence)),
            orders=reconciliations,
            reconciliation_break_count=sum(bool(item.break_reasons) for item in reconciliations),
            all_orders_terminal=all(item.terminal for item in reconciliations),
            paper_pnl_is_gate_input=False,
        )

    @staticmethod
    def _order(
        *,
        evidence: NbboReplayEvidence,
        broker_order: BrokerOrder,
        session_date: date,
    ) -> PaperOrderReconciliation:
        replay = evidence.result
        intended = replay.order
        reasons: list[str] = []
        submitted_date = broker_order.submitted_at.astimezone(_MARKET_TIMEZONE).date()
        if submitted_date != session_date:
            reasons.append("paper submission date differs from session")
        if broker_order.symbol != intended.ticker:
            reasons.append("paper symbol differs from intended order")
        if broker_order.side != intended.side:
            reasons.append("paper side differs from intended order")
        paper_quantity = int(broker_order.quantity)
        if paper_quantity != intended.quantity:
            reasons.append("paper quantity differs from intended order")
        terminal = broker_order.status in _TERMINAL_STATUSES
        if not terminal:
            reasons.append("paper order is not terminal after close")
        paper_filled = int(broker_order.filled_quantity)
        paper_price = (
            float(broker_order.filled_average_price)
            if broker_order.filled_average_price is not None
            else None
        )
        divergence = _price_divergence_bps(
            side=intended.side,
            paper_price=paper_price,
            replay_price=replay.fill_price,
        )
        return PaperOrderReconciliation(
            client_order_id=broker_order.client_order_id,
            symbol=intended.ticker,
            side=intended.side,
            intended_quantity=intended.quantity,
            replay_filled_quantity=replay.filled_qty,
            replay_fill_price=replay.fill_price,
            paper_status=broker_order.status,
            paper_filled_quantity=paper_filled,
            paper_fill_price=paper_price,
            fill_quantity_difference=paper_filled - replay.filled_qty,
            paper_vs_replay_price_bps=divergence,
            terminal=terminal,
            break_reasons=tuple(reasons),
        )


class PaperReconciliationSpec(BaseModel):
    """Strict JSON boundary for after-close paper/replay reconciliation."""

    model_config = ConfigDict(extra="forbid")

    session_date: date
    evaluated_at: datetime
    replay_evidence_files: tuple[Path, ...] = Field(min_length=1)
    broker_orders: tuple[BrokerOrder, ...] = Field(min_length=1)


class _PaperOrderReconciliationSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    client_order_id: str = Field(min_length=1)
    symbol: str = Field(min_length=1)
    side: Literal["buy", "sell"]
    intended_quantity: int = Field(gt=0)
    replay_filled_quantity: int = Field(ge=0)
    replay_fill_price: float | None = Field(default=None, gt=0)
    paper_status: str = Field(min_length=1)
    paper_filled_quantity: int = Field(ge=0)
    paper_fill_price: float | None = Field(default=None, gt=0)
    fill_quantity_difference: int
    paper_vs_replay_price_bps: float | None
    terminal: bool
    break_reasons: tuple[str, ...]

    def to_domain(self) -> PaperOrderReconciliation:
        return PaperOrderReconciliation(**self.model_dump())


class _PaperReconciliationReportSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1]
    session_date: date
    evaluated_at: datetime
    replay_evidence_sha256: tuple[str, ...]
    orders: tuple[_PaperOrderReconciliationSpec, ...]
    reconciliation_break_count: int = Field(ge=0)
    all_orders_terminal: bool
    paper_pnl_is_gate_input: Literal[False]

    def to_domain(self) -> PaperReconciliationReport:
        values = self.model_dump(exclude={"orders"})
        return PaperReconciliationReport(
            **values,
            orders=tuple(item.to_domain() for item in self.orders),
        )


def _price_divergence_bps(
    *,
    side: str,
    paper_price: float | None,
    replay_price: float | None,
) -> float | None:
    if paper_price is None or replay_price is None:
        return None
    if not math.isfinite(paper_price) or paper_price <= 0:
        raise ValueError("paper fill price must be finite and positive")
    direction = 1.0 if side == "buy" else -1.0
    return direction * (paper_price / replay_price - 1.0) * 10_000.0
