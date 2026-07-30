"""Point-in-time feature contracts and registered baseline definitions."""

from quant_earning_edge.features import event as _event  # noqa: F401 - registration side effect
from quant_earning_edge.features import gap as _gap  # noqa: F401 - registration side effect
from quant_earning_edge.features import (
    momentum as _momentum,  # noqa: F401 - registration side effect
)
from quant_earning_edge.features import price as _price  # noqa: F401 - registration side effect
from quant_earning_edge.features import volume as _volume  # noqa: F401 - registration side effect
from quant_earning_edge.features.historical_source import (
    HistoricalFeatureSourceCapture,
    HistoricalFeatureSourceManifest,
)
from quant_earning_edge.features.inputs import (
    DailyBarsFeatureLoader,
    EarningsFeatureLoader,
    PremarketFeatureLoader,
)
from quant_earning_edge.features.registry import (
    FEATURE_REGISTRY,
    EarningsObservation,
    FeatureContext,
    FeatureRegistry,
    FeatureSpec,
    InsufficientHistoryError,
    PremarketObservation,
    PriceBar,
    feature,
)
from quant_earning_edge.features.source_capture import (
    FeatureSourceCapture,
    FeatureSourceManifest,
)
from quant_earning_edge.features.store import (
    FEATURE_VALUE_SCHEMA,
    FeatureArtifact,
    FeatureEngine,
    FeatureStore,
    FeatureValue,
)

__all__ = [
    "FEATURE_REGISTRY",
    "FEATURE_VALUE_SCHEMA",
    "DailyBarsFeatureLoader",
    "EarningsFeatureLoader",
    "EarningsObservation",
    "FeatureArtifact",
    "FeatureContext",
    "FeatureEngine",
    "FeatureRegistry",
    "FeatureSourceCapture",
    "FeatureSourceManifest",
    "FeatureSpec",
    "FeatureStore",
    "FeatureValue",
    "HistoricalFeatureSourceCapture",
    "HistoricalFeatureSourceManifest",
    "InsufficientHistoryError",
    "PremarketFeatureLoader",
    "PremarketObservation",
    "PriceBar",
    "feature",
]
