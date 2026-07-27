"""Deterministic paths for the bronze/silver/gold lakehouse."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from enum import StrEnum
from pathlib import Path

_PARTITION_VALUE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


class DataTier(StrEnum):
    """Storage tiers with progressively stronger data contracts."""

    BRONZE = "bronze"
    SILVER = "silver"
    GOLD = "gold"


def _safe_partition(value: str, *, field: str) -> str:
    if not _PARTITION_VALUE.fullmatch(value):
        raise ValueError(f"{field} contains unsafe path characters: {value!r}")
    return value


@dataclass(frozen=True)
class LakehouseLayout:
    """Build partition paths without allowing traversal outside the data root."""

    root: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", self.root.expanduser().resolve())

    def bronze(self, *, source: str, dataset: str, event_date: date) -> Path:
        """Raw, immutable API payload partition."""
        return (
            self.root
            / DataTier.BRONZE
            / f"source={_safe_partition(source, field='source')}"
            / f"dataset={_safe_partition(dataset, field='dataset')}"
            / f"date={event_date.isoformat()}"
        )

    def silver(self, *, asset_class: str, dataset: str, event_date: date) -> Path:
        """Cleaned, schema-stable record partition."""
        return (
            self.root
            / DataTier.SILVER
            / f"asset_class={_safe_partition(asset_class, field='asset_class')}"
            / f"dataset={_safe_partition(dataset, field='dataset')}"
            / f"date={event_date.isoformat()}"
        )

    def gold(self, *, feature_group: str, asof_month: date) -> Path:
        """Point-in-time feature partition, grouped monthly."""
        month = asof_month.replace(day=1).isoformat()[:7]
        return (
            self.root
            / DataTier.GOLD
            / f"feature_group={_safe_partition(feature_group, field='feature_group')}"
            / f"month={month}"
        )
