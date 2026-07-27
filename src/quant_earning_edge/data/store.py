"""DuckDB query boundary for local lakehouse analytics."""

from __future__ import annotations

from pathlib import Path
from types import TracebackType
from typing import Any, Self

import duckdb


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
