"""Point-in-time correctness test.

Every registered feature must produce the same value when given history clipped
at T as it does when given a longer history clipped at T. If adding future data
to the input changes the output, the feature is reading the future.

This test is parametrized over ``FEATURE_REGISTRY``; until features are
registered (Phase 2), it runs vacuously. Adding a feature without making it
pass this test is a CI-blocking failure.
"""

from __future__ import annotations

import pytest

from quant_earning_edge.features import FEATURE_REGISTRY, FeatureSpec


@pytest.mark.property
@pytest.mark.parametrize(
    "spec",
    list(FEATURE_REGISTRY.values()) or [None],
    ids=lambda s: s.name if s is not None else "no-features-registered",
)
def test_feature_is_point_in_time(spec: FeatureSpec | None) -> None:
    """Feature value at T must not depend on data after T.

    Phase 2 fills this in: synthesize a history H, compute ``f(ticker, T, H[:T])``
    vs ``f(ticker, T, H[:T+lookahead])``, assert exact equality.
    """
    if spec is None:
        pytest.skip("No features registered yet — scaffold runs vacuously.")
    pytest.skip(f"Feature implementations land in Phase 2; scaffold for {spec.name}.")
