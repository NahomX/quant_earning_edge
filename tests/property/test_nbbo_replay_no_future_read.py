"""NBBO-replay simulator must not read quotes from before the decision time.

For an order decided at ``decision_time`` T_d, the replay simulator may only
consult quotes timestamped >= T_d. Any earlier quote read would let the
strategy peek into a market state that wasn't observable at decision time.

This test will be filled in alongside the Phase 6 implementation. The contract
is staged here so the invariant is fixed before code is written.
"""

from __future__ import annotations

import pytest


@pytest.mark.property
def test_replay_never_reads_quotes_before_decision_time() -> None:
    """Phase 6: assert simulator's quote-read indices are all >= decision_time."""
    pytest.skip("NBBO replay implementation lands in Phase 6; scaffold only.")
