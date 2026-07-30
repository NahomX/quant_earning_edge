"""Validated, content-addressed LightGBM hyperparameters."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class LightgbmHyperparameters:
    """The bounded model parameters shared by research and production refits."""

    schema_version: int = 1
    n_estimators: int = 1_000
    learning_rate: float = 0.03
    num_leaves: int = 15
    min_child_samples: int = 20
    reg_alpha: float = 0.0
    reg_lambda: float = 1.0
    colsample_bytree: float = 1.0

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("unsupported LightGBM hyperparameter schema version")
        if not 100 <= self.n_estimators <= 5_000:
            raise ValueError("n_estimators must be between 100 and 5000")
        if not 0.001 <= self.learning_rate <= 0.3:
            raise ValueError("learning_rate must be between 0.001 and 0.3")
        if not 2 <= self.num_leaves <= 255:
            raise ValueError("num_leaves must be between 2 and 255")
        if not 5 <= self.min_child_samples <= 500:
            raise ValueError("min_child_samples must be between 5 and 500")
        if not 0 <= self.reg_alpha <= 100:
            raise ValueError("reg_alpha must be between 0 and 100")
        if not 0 <= self.reg_lambda <= 100:
            raise ValueError("reg_lambda must be between 0 and 100")
        if not 0.25 <= self.colsample_bytree <= 1:
            raise ValueError("colsample_bytree must be between 0.25 and 1")
        if not all(
            math.isfinite(value)
            for value in (
                self.learning_rate,
                self.reg_alpha,
                self.reg_lambda,
                self.colsample_bytree,
            )
        ):
            raise ValueError("LightGBM hyperparameters must be finite")

    def canonical_json_bytes(self) -> bytes:
        """Return the exact portable representation used for identity."""
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":")).encode()

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_json_bytes()).hexdigest()

    def classifier_kwargs(self) -> dict[str, Any]:
        """Return only parameters controlled by the hyperparameter contract."""
        payload = asdict(self)
        payload.pop("schema_version")
        return payload

    @classmethod
    def from_dict(cls, raw: object) -> LightgbmHyperparameters:
        """Strictly parse the nested representation stored in model evidence."""
        expected = {
            "schema_version",
            "n_estimators",
            "learning_rate",
            "num_leaves",
            "min_child_samples",
            "reg_alpha",
            "reg_lambda",
            "colsample_bytree",
        }
        if not isinstance(raw, dict) or set(raw) != expected:
            raise ValueError("LightGBM hyperparameter schema mismatch")
        return cls(
            schema_version=int(raw["schema_version"]),
            n_estimators=int(raw["n_estimators"]),
            learning_rate=float(raw["learning_rate"]),
            num_leaves=int(raw["num_leaves"]),
            min_child_samples=int(raw["min_child_samples"]),
            reg_alpha=float(raw["reg_alpha"]),
            reg_lambda=float(raw["reg_lambda"]),
            colsample_bytree=float(raw["colsample_bytree"]),
        )
