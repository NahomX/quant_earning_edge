"""Leakage-guarded assembly of long-form features and forward labels."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import pyarrow as pa
import pyarrow.parquet as pq

from quant_earning_edge.data.calendar import SessionFileStore
from quant_earning_edge.features import FEATURE_VALUE_SCHEMA
from quant_earning_edge.labels.forward import FORWARD_LABEL_SCHEMA

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from quant_earning_edge.data.layout import LakehouseLayout


@dataclass(frozen=True)
class TrainingDatasetArtifact:
    """Immutable wide training table plus source manifest."""

    path: Path
    manifest_path: Path
    sha256: str
    row_count: int
    feature_names: tuple[str, ...]


class TrainingDatasetAssembler:
    """Create a complete-key training matrix while enforcing decision-time cutoffs."""

    def __init__(self, layout: LakehouseLayout) -> None:
        self._layout = layout

    def assemble(  # noqa: PLR0912, PLR0915 - one atomic validation/persistence boundary.
        self,
        *,
        feature_files: Sequence[Path],
        label_files: Sequence[Path],
        session_file: Path,
        assembled_at: datetime,
    ) -> TrainingDatasetArtifact:
        """Join exact feature/label key sets and persist a wide monthly table."""
        if not feature_files or not label_files:
            raise ValueError("feature and label files are required")
        if assembled_at.tzinfo is None or assembled_at.utcoffset() is None:
            raise ValueError("assembled_at must be timezone-aware")
        session_artifact = SessionFileStore.load(session_file)
        sessions = session_artifact.sessions
        session_dates = tuple(item.session_date for item in sessions)
        opens = {item.session_date: item.open_at.astimezone(UTC) for item in sessions}

        feature_rows, feature_hashes = self._load_features(feature_files)
        label_rows, label_hashes = self._load_labels(label_files)
        feature_keys = {(row["symbol"], row["asof_date"]) for row in feature_rows}
        label_by_key = {(row["symbol"], row["asof_date"]): row for row in label_rows}
        if len(label_by_key) != len(label_rows):
            raise ValueError("labels contain duplicate symbol/asof keys")
        if feature_keys != set(label_by_key):
            missing_labels = sorted(feature_keys - set(label_by_key))
            missing_features = sorted(set(label_by_key) - feature_keys)
            raise ValueError(
                "feature/label key sets differ: "
                f"missing_labels={missing_labels}, missing_features={missing_features}"
            )

        feature_names = tuple(sorted({str(row["feature_name"]) for row in feature_rows}))
        expected_names = set(feature_names)
        code_hash_by_name: dict[str, str] = {}
        grouped: dict[tuple[str, Any], dict[str, dict[str, Any]]] = {}
        for row in feature_rows:
            key = (str(row["symbol"]), row["asof_date"])
            name = str(row["feature_name"])
            if name in grouped.setdefault(key, {}):
                raise ValueError(f"duplicate feature value for {key}/{name}")
            grouped[key][name] = row
            previous_hash = code_hash_by_name.setdefault(name, str(row["feature_code_hash"]))
            if previous_hash != row["feature_code_hash"]:
                raise ValueError(f"mixed feature code versions for {name}")

        records: list[dict[str, Any]] = []
        for key in sorted(grouped, key=lambda item: (item[1], item[0])):
            rows_by_name = grouped[key]
            if set(rows_by_name) != expected_names:
                raise ValueError(f"incomplete feature vector for {key}")
            label = label_by_key[key]
            asof_date = key[1]
            target_date = label["target_date"]
            try:
                asof_index = session_dates.index(asof_date)
            except ValueError as error:
                raise ValueError(f"feature asof_date absent from sessions: {asof_date}") from error
            if asof_index + 1 >= len(session_dates) or session_dates[asof_index + 1] != target_date:
                raise ValueError(f"label target is not the next session for {key}")
            target_open = opens[target_date]
            input_hashes = {str(row["input_sha256"]) for row in rows_by_name.values()}
            if len(input_hashes) != 1:
                raise ValueError(f"mixed feature input lineage for {key}")
            if any(row["computed_at"] >= target_open for row in rows_by_name.values()):
                raise ValueError(f"feature computation was not frozen before target open: {key}")
            information_cutoff_at = max(
                row["computed_at"].astimezone(UTC) for row in rows_by_name.values()
            )
            record: dict[str, Any] = {
                "symbol": key[0],
                "asof_date": asof_date,
                "target_date": target_date,
                "horizon_end_date": label["horizon_end_date"],
                "information_cutoff_at": information_cutoff_at,
                "feature_input_sha256": next(iter(input_hashes)),
                "label_input_sha256": label["input_sha256"],
                "forward_1d_open_to_close": label["forward_1d_open_to_close"],
                "forward_1d_close": label["forward_1d_close"],
                "forward_5d_close": label["forward_5d_close"],
            }
            record.update({name: float(rows_by_name[name]["value"]) for name in feature_names})
            records.append(record)

        months = {(row["asof_date"].year, row["asof_date"].month) for row in records}
        if len(months) != 1:
            raise ValueError("one training artifact cannot span multiple as-of months")
        schema = _training_schema(feature_names)
        evidence = {
            "assembled_at": assembled_at.astimezone(UTC),
            "session_file_sha256": session_artifact.sha256,
            "feature_file_sha256": feature_hashes,
            "label_file_sha256": label_hashes,
            "feature_code_hashes": code_hash_by_name,
            "records": records,
        }
        digest = _digest(evidence)
        partition = self._layout.gold(
            feature_group="training-dataset",
            asof_month=records[0]["asof_date"],
        )
        partition.mkdir(parents=True, exist_ok=True)
        path = partition / f"part-{digest[:20]}.parquet"
        table = pa.Table.from_pylist(records, schema=schema)
        try:
            with path.open("xb") as sink:
                pq.write_table(table, sink, compression="zstd")  # type: ignore[no-untyped-call]
        except FileExistsError:
            if pq.read_schema(path) != schema:  # type: ignore[no-untyped-call]
                raise RuntimeError(f"training dataset schema collision at {path}") from None
        manifest_path = partition / f"manifest-{digest[:20]}.json"
        encoded = json.dumps(
            evidence,
            default=lambda item: item.isoformat(),
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        try:
            with manifest_path.open("xb") as destination:
                destination.write(encoded)
        except FileExistsError:
            if manifest_path.read_bytes() != encoded:
                raise RuntimeError(f"training manifest collision at {manifest_path}") from None
        return TrainingDatasetArtifact(
            path=path,
            manifest_path=manifest_path,
            sha256=digest,
            row_count=table.num_rows,
            feature_names=feature_names,
        )

    @staticmethod
    def _load_features(paths: Sequence[Path]) -> tuple[list[dict[str, Any]], tuple[str, ...]]:
        rows: list[dict[str, Any]] = []
        hashes: list[str] = []
        for path in sorted(paths):
            if pq.read_schema(path) != FEATURE_VALUE_SCHEMA:  # type: ignore[no-untyped-call]
                raise ValueError(f"feature artifact schema mismatch: {path}")
            hashes.append(_file_hash(path))
            rows.extend(pq.read_table(path).to_pylist())  # type: ignore[no-untyped-call]
        return rows, tuple(hashes)

    @staticmethod
    def _load_labels(paths: Sequence[Path]) -> tuple[list[dict[str, Any]], tuple[str, ...]]:
        rows: list[dict[str, Any]] = []
        hashes: list[str] = []
        for path in sorted(paths):
            if pq.read_schema(path) != FORWARD_LABEL_SCHEMA:  # type: ignore[no-untyped-call]
                raise ValueError(f"label artifact schema mismatch: {path}")
            hashes.append(_file_hash(path))
            rows.extend(pq.read_table(path).to_pylist())  # type: ignore[no-untyped-call]
        return rows, tuple(hashes)


def _training_schema(feature_names: tuple[str, ...]) -> pa.Schema:
    return pa.schema(
        [
            pa.field("symbol", pa.string(), nullable=False),
            pa.field("asof_date", pa.date32(), nullable=False),
            pa.field("target_date", pa.date32(), nullable=False),
            pa.field("horizon_end_date", pa.date32(), nullable=False),
            pa.field("information_cutoff_at", pa.timestamp("us", tz="UTC"), nullable=False),
            pa.field("feature_input_sha256", pa.string(), nullable=False),
            pa.field("label_input_sha256", pa.string(), nullable=False),
            *[pa.field(name, pa.float64(), nullable=False) for name in feature_names],
            pa.field("forward_1d_open_to_close", pa.float64(), nullable=False),
            pa.field("forward_1d_close", pa.float64(), nullable=False),
            pa.field("forward_5d_close", pa.float64(), nullable=False),
        ]
    )


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _digest(value: object) -> str:
    encoded = json.dumps(
        value,
        default=lambda item: item.isoformat(),
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()
