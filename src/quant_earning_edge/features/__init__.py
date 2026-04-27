"""Feature registry and definitions.

Every feature is a pure function ``(ticker, asof_date) -> scalar/vector`` registered
via the :func:`registry.feature` decorator. The registry is consulted by the
no-lookahead property test in ``tests/property/test_no_lookahead.py``.
"""

from quant_earning_edge.features.registry import FEATURE_REGISTRY, FeatureSpec, feature

__all__ = ["FEATURE_REGISTRY", "FeatureSpec", "feature"]
