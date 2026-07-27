"""DuckDB query boundary for local lakehouse analytics."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from quant_earning_edge.data.silver import (
    DAILY_BARS_SCHEMA,
    DIVIDENDS_SCHEMA,
    EARNINGS_SCHEMA,
    SPLITS_SCHEMA,
)

if TYPE_CHECKING:
    from types import TracebackType

    from quant_earning_edge.data.layout import LakehouseLayout


class SilverDataset(StrEnum):
    """Stable names for validated silver query surfaces."""

    DAILY_BARS = "daily_bars"
    EARNINGS_EVENTS = "earnings_events"
    STOCK_SPLITS = "stock_splits"
    CASH_DIVIDENDS = "cash_dividends"


_SILVER_SPECS: dict[SilverDataset, tuple[str, pa.Schema]] = {
    SilverDataset.DAILY_BARS: ("daily-bars", DAILY_BARS_SCHEMA),
    SilverDataset.EARNINGS_EVENTS: ("earnings-events", EARNINGS_SCHEMA),
    SilverDataset.STOCK_SPLITS: ("stock-splits", SPLITS_SCHEMA),
    SilverDataset.CASH_DIVIDENDS: ("cash-dividends", DIVIDENDS_SCHEMA),
}


class DuckDBStore:
    """Own a DuckDB connection and expose parameterized query operations."""

    def __init__(self, database: Path | str = ":memory:") -> None:
        target = str(database)
        if target != ":memory:":
            path = Path(target).expanduser().resolve()
            path.parent.mkdir(parents=True, exist_ok=True)
            target = str(path)
        self._connection = duckdb.connect(target)

    def execute(
        self,
        sql: str,
        parameters: tuple[Any, ...] = (),
    ) -> duckdb.DuckDBPyConnection:
        """Execute SQL with positional parameters."""
        return self._connection.execute(sql, parameters)

    def parquet_relation(self, path_glob: Path | str) -> duckdb.DuckDBPyRelation:
        """Return a relation over partitioned Parquet with Hive keys enabled."""
        return self._connection.from_parquet(
            str(path_glob),
            hive_partitioning=True,
        )

    def register_silver_views(
        self,
        layout: LakehouseLayout,
        *,
        datasets: tuple[SilverDataset, ...] = tuple(SilverDataset),
    ) -> tuple[str, ...]:
        """Validate every artifact schema before replacing stable SQL views."""
        if not datasets:
            raise ValueError("at least one silver dataset is required")
        registered: list[str] = []
        for dataset in datasets:
            partition_name, expected_schema = _SILVER_SPECS[dataset]
            root = layout.root / "silver" / "asset_class=us-equity" / f"dataset={partition_name}"
            paths = sorted(root.rglob("*.parquet"))
            if not paths:
                raise FileNotFoundError(f"no silver artifacts found for {dataset.value}")
            for path in paths:
                actual_schema = pq.read_schema(path)  # type: ignore[no-untyped-call]
                if actual_schema != expected_schema:
                    raise ValueError(f"schema mismatch for {dataset.value}: {path}")
            relation = self.parquet_relation(root / "**" / "*.parquet")
            view_name = f"silver_{dataset.value}"
            relation.create_view(view_name, replace=True)
            registered.append(view_name)
        return tuple(registered)

    def close(self) -> None:
        """Close the owned connection."""
        self._connection.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()
