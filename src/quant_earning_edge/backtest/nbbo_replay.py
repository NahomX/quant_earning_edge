"""NBBO-replay simulator stub.

Phase 6 deliverable. Replays intended orders against recorded Polygon NBBO/Trade
data to produce execution-realism metrics (fill rate, realized slippage, partial
fills, opening-auction skew). Output is the primary input to the terminal proof;
Alpaca paper P&L is a smoke-test signal only.

Invariant — enforced by ``tests/property/test_nbbo_replay_no_future_read.py``:
the simulator may only consult quotes timestamped at or after the order's
decision time. Reading any earlier quote is a look-ahead bug.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal

OrderSide = Literal["buy", "sell"]
OrderAggressiveness = Literal["aggressive", "mid", "passive"]


@dataclass(frozen=True)
class IntendedOrder:
    """An order as it was decided at ``decision_time`` (T-1 21:30 ET typically)."""

    ticker: str
    side: OrderSide
    quantity: int
    decision_time: datetime
    aggressiveness: OrderAggressiveness = "aggressive"


@dataclass(frozen=True)
class ReplayFill:
    """Result of replaying one IntendedOrder against recorded NBBO."""

    order: IntendedOrder
    filled_qty: int
    fill_price: float | None
    slippage_bps_realized: float | None
    slippage_bps_predicted: float | None
    notes: str = ""


def replay_order(order: IntendedOrder) -> ReplayFill:
    """Replay a single intended order against recorded NBBO.

    NOT YET IMPLEMENTED. The first implementation lands in Phase 6 alongside
    Polygon Advanced data ingestion. The signature is fixed now so the property
    test scaffold can reference it.
    """
    raise NotImplementedError("NBBO replay implementation lands in Phase 6.")
