from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from src.regimes.core.paths import default_foundation_report_root, normalized_path_parts, require_foundation_report_root
from src.regimes.core.serialization import dumps_json, to_jsonable
from src.regimes.market_state.contracts import MARKET_STATE_SCHEMA_VERSION, MarketStateSchemaVersion, _schema_version
from src.regimes.market_state.output_contract import (
    MarketStateOutputRow,
    MarketStateSandboxOutputSchema,
    default_market_state_sandbox_output_schema,
    validate_market_state_output_rows,
)


SANDBOX_OUTPUT_DIR_NAME = "sandbox_outputs"
SANDBOX_WRITER_STATUS_WRITTEN = "written"
SANDBOX_WRITER_SUPPORTED_FORMATS: tuple[str, ...] = ("auto", "parquet", "jsonl", "csv")
PRODUCTION_LIKE_SANDBOX_PARTS: frozenset[str] = frozenset(
    {"production", "prod", "live", "regime_labels", "regime-labels", "prod_outputs", "production_outputs"}
)


@dataclass(frozen=True)
class MarketStateSandboxWriteRequest:
    rows: Sequence[MarketStateOutputRow | Mapping[str, Any]]
    output_root: Path | str | None = None
    run_id: str | None = None
    file_format: str = "auto"
    write_metadata: bool = True
    schema_version: int | MarketStateSchemaVersion = MARKET_STATE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        fmt = str(self.file_format).strip().lower()
        if fmt not in SANDBOX_WRITER_SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported market-state sandbox output format {fmt!r}")
        object.__setattr__(self, "schema_version", _schema_version(self.schema_version))
        object.__setattr__(self, "output_root", None if self.output_root is None else Path(self.output_root))
        object.__setattr__(self, "run_id", None if self.run_id is None else str(self.run_id).strip())
        object.__setattr__(self, "file_format", fmt)


@dataclass(frozen=True)
class MarketStateSandboxWriteResult:
    status: str
    output_root: Path
    run_id: str
    row_count: int
    file_format: str
    artifact_paths: Mapping[str, str]
    schema: MarketStateSandboxOutputSchema = field(default_factory=default_market_state_sandbox_output_schema)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int | MarketStateSchemaVersion = MARKET_STATE_SCHEMA_VERSION

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "artifact_kind": "market_state_sandbox_write_result",
            "status": self.status,
            "output_root": str(self.output_root),
            "run_id": self.run_id,
            "row_count": int(self.row_count),
            "file_format": self.file_format,
            "artifact_paths": dict(self.artifact_paths),
            "schema": self.schema.as_dict(),
            "metadata": to_jsonable(dict(self.metadata)),
            "production_outputs_written": False,
        }

    def to_json(self, **kwargs: Any) -> str:
        return dumps_json(self.as_dict(), **kwargs)


def resolve_market_state_sandbox_output_root(output_root: Path | str | None = None) -> Path:
    raw = Path(output_root) if output_root is not None else default_foundation_report_root("market_state", SANDBOX_OUTPUT_DIR_NAME)
    safe = require_foundation_report_root(
        raw,
        allow_foundation_descendant=True,
        error_prefix="Market-state sandbox output root",
    )
    parts = normalized_path_parts(safe)
    if "market_state" not in parts:
        raise ValueError("Market-state sandbox output root must be under reports/regimes/foundation/market_state")
    if any(part in PRODUCTION_LIKE_SANDBOX_PARTS for part in parts):
        raise ValueError("Market-state sandbox output root is production-like and is not allowed")
    if SANDBOX_OUTPUT_DIR_NAME not in parts:
        safe = safe / SANDBOX_OUTPUT_DIR_NAME
    return safe.resolve()


def write_market_state_sandbox_outputs(request: MarketStateSandboxWriteRequest) -> MarketStateSandboxWriteResult:
    rows = validate_market_state_output_rows(request.rows)
    run_id = _resolve_run_id(request.run_id, rows)
    output_root = resolve_market_state_sandbox_output_root(request.output_root)
    file_format = _resolve_file_format(request.file_format)
    schema = default_market_state_sandbox_output_schema()

    grouped: dict[tuple[str, str, int], list[MarketStateOutputRow]] = {}
    for row in rows:
        grouped.setdefault((row.axis, row.band, int(row.interval)), []).append(row)

    artifact_paths: dict[str, str] = {}
    for (axis, band, interval), group_rows in sorted(grouped.items()):
        partition = output_root / f"run_id={_safe_part(run_id)}" / f"axis={_safe_part(axis)}" / f"band={_safe_part(band)}" / f"interval={int(interval)}"
        path = partition / f"market_state_regime_labels.{file_format}"
        _write_rows(path, group_rows, file_format=file_format)
        artifact_paths[f"{axis}/{band}/{interval}"] = str(path)

    metadata: dict[str, Any] = {
        "schema": schema.as_dict(),
        "run_id": run_id,
        "row_count": int(len(rows)),
        "partition_count": int(len(grouped)),
        "no_production_write_flags": {
            "production_outputs_written": False,
            "production_parquet_written": False,
            "production_regime_labels_written": False,
            "production_profile_promotion": False,
        },
        "write_root_policy": "reports/regimes/foundation/market_state/sandbox_outputs only",
    }
    if request.write_metadata:
        metadata_path = output_root / f"run_id={_safe_part(run_id)}" / "market_state_sandbox_output_metadata.json"
        _write_json_atomic(metadata_path, metadata)
        artifact_paths["metadata"] = str(metadata_path)

    return MarketStateSandboxWriteResult(
        schema_version=MARKET_STATE_SCHEMA_VERSION,
        status=SANDBOX_WRITER_STATUS_WRITTEN,
        output_root=output_root,
        run_id=run_id,
        row_count=len(rows),
        file_format=file_format,
        artifact_paths=artifact_paths,
        schema=schema,
        metadata=metadata,
    )


def _resolve_run_id(requested: str | None, rows: Sequence[MarketStateOutputRow]) -> str:
    run_id = requested or rows[0].run_id
    if not str(run_id).strip():
        raise ValueError("Market-state sandbox writer run_id must be non-empty")
    row_run_ids = {row.run_id for row in rows}
    if requested is None and len(row_run_ids) > 1:
        raise ValueError("Market-state sandbox writer requires run_id when rows contain multiple run ids")
    return str(run_id).strip()


def _resolve_file_format(file_format: str) -> str:
    fmt = str(file_format).strip().lower()
    if fmt == "auto":
        return "parquet" if _parquet_available() else "jsonl"
    if fmt == "parquet" and not _parquet_available():
        raise ValueError("Parquet output requested but no parquet engine is available")
    return fmt


def _parquet_available() -> bool:
    try:
        import pyarrow  # noqa: F401

        return True
    except Exception:
        pass
    try:
        import fastparquet  # noqa: F401

        return True
    except Exception:
        return False


def _write_rows(path: Path, rows: Sequence[MarketStateOutputRow], *, file_format: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [row.storage_dict() for row in rows]
    if file_format == "parquet":
        _write_parquet_atomic(path, payload)
    elif file_format == "jsonl":
        _write_jsonl_atomic(path, payload)
    elif file_format == "csv":
        _write_csv_atomic(path, payload)
    else:
        raise ValueError(f"Unsupported market-state sandbox output format {file_format!r}")


def _write_parquet_atomic(path: Path, payload: Sequence[Mapping[str, Any]]) -> None:
    tmp = path.with_name(f".{path.stem}.{os.getpid()}.tmp{path.suffix}")
    pd.DataFrame([dict(row) for row in payload]).to_parquet(tmp, index=False)
    os.replace(tmp, path)


def _write_jsonl_atomic(path: Path, payload: Sequence[Mapping[str, Any]]) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in payload:
            f.write(json.dumps(to_jsonable(dict(row)), sort_keys=True))
            f.write("\n")
    os.replace(tmp, path)


def _write_csv_atomic(path: Path, payload: Sequence[Mapping[str, Any]]) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    pd.DataFrame([dict(row) for row in payload]).to_csv(tmp, index=False)
    os.replace(tmp, path)


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(to_jsonable(dict(payload)), f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp, path)


def _safe_part(value: object) -> str:
    text = str(value).strip()
    if not text:
        return "unknown"
    return re.sub(r"[^A-Za-z0-9_.=-]+", "_", text)


__all__ = [
    "SANDBOX_OUTPUT_DIR_NAME",
    "SANDBOX_WRITER_STATUS_WRITTEN",
    "SANDBOX_WRITER_SUPPORTED_FORMATS",
    "MarketStateSandboxWriteRequest",
    "MarketStateSandboxWriteResult",
    "resolve_market_state_sandbox_output_root",
    "write_market_state_sandbox_outputs",
]
