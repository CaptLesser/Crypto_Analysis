from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from src.regimes.core.artifact_refs import make_artifact_ref, validate_portable_relative_path
from src.regimes.core.serialization import to_jsonable
from src.regimes.market_state.axis_panel import MARKET_STATE_AXIS_PANEL_SCHEMA_ID


MARKET_STATE_FEATURE_WRITE_STATUS_WRITTEN = "written"
MARKET_STATE_FEATURE_WRITE_STATUS_NO_ROWS = "no_rows"
MARKET_STATE_FEATURE_WRITE_FORMAT_AUTO = "auto"
MARKET_STATE_FEATURE_WRITE_FORMAT_PARQUET = "parquet"
MARKET_STATE_FEATURE_WRITE_FORMAT_JSONL = "jsonl"
MARKET_STATE_FEATURE_MANIFEST_SCHEMA_VERSION = 1

PRODUCTION_LIKE_WRITE_PARTS: frozenset[str] = frozenset(
    {
        "production",
        "prod",
        "live",
        "promoted",
        "promotion",
        "production_outputs",
        "prod_outputs",
        "regime_labels",
        "market_state_labels",
    }
)

FEATURE_REQUIRED_COLUMNS: tuple[str, ...] = (
    "ts",
    "interval",
    "band",
    "feature_family_id",
    "feature_set_id",
    "known_at_ts",
    "source_tail_ts",
    "lineage_id",
    "schema_version",
)
AXIS_REQUIRED_COLUMNS: tuple[str, ...] = (
    "ts",
    "axis",
    "interval",
    "band",
    "known_at_ts",
    "lineage_id",
    "feature_schema_id",
)


@dataclass(frozen=True)
class MarketStateFeatureMaterializationRequest:
    output_root: str | Path
    run_id: str
    market_feature_rows: Mapping[str, pd.DataFrame] | Sequence[pd.DataFrame] = field(default_factory=dict)
    axis_panel_rows: Mapping[str, pd.DataFrame] | Sequence[pd.DataFrame] = field(default_factory=dict)
    universe_manifest_reference: Mapping[str, Any] | str | Path | None = None
    file_format: str = MARKET_STATE_FEATURE_WRITE_FORMAT_AUTO
    production_enabled: bool = False
    write_manifest: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "output_root", Path(self.output_root))
        object.__setattr__(self, "run_id", _safe_part(self.run_id, field_name="run_id"))
        object.__setattr__(self, "file_format", _file_format(self.file_format))
        object.__setattr__(self, "production_enabled", bool(self.production_enabled))
        object.__setattr__(self, "write_manifest", bool(self.write_manifest))
        object.__setattr__(self, "metadata", to_jsonable(dict(self.metadata)))


@dataclass(frozen=True)
class MarketStateFeatureMaterializationResult:
    status: str
    output_root: Path
    run_id: str
    market_feature_row_count: int = 0
    axis_panel_row_count: int = 0
    written_paths: Sequence[Path] = ()
    manifest_path: Path | None = None
    build_summary_path: Path | None = None
    universe_reference_path: Path | None = None
    file_format_counts: Mapping[str, int] = field(default_factory=dict)
    artifact_boundary: Mapping[str, Any] = field(default_factory=dict)
    manifest: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": MARKET_STATE_FEATURE_MANIFEST_SCHEMA_VERSION,
            "artifact_kind": "market_state_feature_materialization_result",
            "status": self.status,
            "output_root": str(self.output_root),
            "run_id": self.run_id,
            "market_feature_row_count": int(self.market_feature_row_count),
            "axis_panel_row_count": int(self.axis_panel_row_count),
            "written_paths": [str(path) for path in self.written_paths],
            "manifest_path": str(self.manifest_path) if self.manifest_path is not None else None,
            "build_summary_path": str(self.build_summary_path) if self.build_summary_path is not None else None,
            "universe_reference_path": str(self.universe_reference_path) if self.universe_reference_path is not None else None,
            "file_format_counts": dict(self.file_format_counts),
            "artifact_boundary": to_jsonable(dict(self.artifact_boundary)),
            "manifest": to_jsonable(dict(self.manifest)),
            "metadata": to_jsonable(dict(self.metadata)),
            "production_enabled": False,
            "production_outputs_written": False,
        }


def write_market_state_v1_feature_materialization(
    request: MarketStateFeatureMaterializationRequest,
) -> MarketStateFeatureMaterializationResult:
    root = validate_market_state_feature_write_root(
        request.output_root,
        production_enabled=bool(request.production_enabled),
    )
    market_rows = _normalize_market_feature_rows(request.market_feature_rows)
    axis_rows = _normalize_axis_panel_rows(request.axis_panel_rows)
    market_count = int(sum(frame.shape[0] for frame in market_rows.values()))
    axis_count = int(sum(frame.shape[0] for frame in axis_rows.values()))
    boundary = _artifact_boundary()
    if market_count <= 0 and axis_count <= 0:
        return MarketStateFeatureMaterializationResult(
            status=MARKET_STATE_FEATURE_WRITE_STATUS_NO_ROWS,
            output_root=root,
            run_id=request.run_id,
            artifact_boundary=boundary,
            metadata={"request_metadata": dict(request.metadata)},
        )

    written: list[Path] = []
    format_counts: dict[str, int] = {}
    for family, frame in sorted(market_rows.items()):
        _validate_market_feature_frame(frame, family=family)
        for path, fmt in _write_market_feature_partitions(frame, root=root, family=family, file_format=request.file_format):
            written.append(path)
            format_counts[fmt] = format_counts.get(fmt, 0) + 1
    for axis, frame in sorted(axis_rows.items()):
        _validate_axis_panel_frame(frame, axis=axis)
        for path, fmt in _write_axis_panel_partitions(frame, root=root, axis=axis, file_format=request.file_format):
            written.append(path)
            format_counts[fmt] = format_counts.get(fmt, 0) + 1

    manifest_dir = root / "market_state_feature_manifests" / request.run_id
    _ensure_within_root(manifest_dir, root)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    universe_ref = _universe_reference_payload(request.universe_manifest_reference)
    universe_ref_path = manifest_dir / "universe_manifest_reference.json"
    build_summary_path = manifest_dir / "build_summary.json"
    manifest_path = manifest_dir / "manifest.json"
    manifest = _manifest_payload(
        request,
        root=root,
        market_rows=market_rows,
        axis_rows=axis_rows,
        written=written,
        file_format_counts=format_counts,
        artifact_boundary=boundary,
        universe_reference=universe_ref,
    )
    build_summary = {
        "artifact_kind": "market_state_v1_feature_build_summary",
        "run_id": request.run_id,
        "market_feature_row_count": market_count,
        "axis_panel_row_count": axis_count,
        "partition_count": len(written),
        "file_format_counts": dict(sorted(format_counts.items())),
        "artifact_boundary": boundary,
    }
    if request.write_manifest:
        _write_json(universe_ref_path, universe_ref)
        _write_json(build_summary_path, build_summary)
        _write_json(manifest_path, manifest)
        written.extend([universe_ref_path, build_summary_path, manifest_path])
    return MarketStateFeatureMaterializationResult(
        status=MARKET_STATE_FEATURE_WRITE_STATUS_WRITTEN,
        output_root=root,
        run_id=request.run_id,
        market_feature_row_count=market_count,
        axis_panel_row_count=axis_count,
        written_paths=tuple(written),
        manifest_path=manifest_path if request.write_manifest else None,
        build_summary_path=build_summary_path if request.write_manifest else None,
        universe_reference_path=universe_ref_path if request.write_manifest else None,
        file_format_counts=dict(sorted(format_counts.items())),
        artifact_boundary=boundary,
        manifest=manifest,
        metadata={"request_metadata": dict(request.metadata)},
    )


def validate_market_state_feature_write_root(
    output_root: str | Path,
    *,
    production_enabled: bool = False,
) -> Path:
    if production_enabled is not False:
        raise ValueError("Market-State v1 feature writer production writes are disabled")
    root = Path(output_root).expanduser()
    if not str(root).strip():
        raise ValueError("Market-State v1 feature writer output_root must be non-empty")
    parts = {part.lower() for part in root.parts}
    if parts.intersection(PRODUCTION_LIKE_WRITE_PARTS):
        raise ValueError("Market-State v1 feature writer refusing production-like output root")
    return root.resolve()


def _write_market_feature_partitions(
    frame: pd.DataFrame,
    *,
    root: Path,
    family: str,
    file_format: str,
) -> list[tuple[Path, str]]:
    work = _with_time_partitions(frame)
    written: list[tuple[Path, str]] = []
    for (interval, band, year, month), group in work.groupby(["interval", "band", "_year", "_month"], sort=True):
        partition = (
            root
            / f"regime_features_market_{int(interval)}"
            / f"band={_safe_part(band, field_name='band')}"
            / f"feature_family={_safe_part(family, field_name='feature_family')}"
            / f"year={int(year):04d}"
            / f"month={int(month):02d}"
        )
        written.append(_write_frame(group.drop(columns=["_year", "_month"], errors="ignore"), partition, root=root, file_format=file_format))
    return written


def _write_axis_panel_partitions(
    frame: pd.DataFrame,
    *,
    root: Path,
    axis: str,
    file_format: str,
) -> list[tuple[Path, str]]:
    work = _with_time_partitions(frame)
    written: list[tuple[Path, str]] = []
    for (interval, band, year, month), group in work.groupby(["interval", "band", "_year", "_month"], sort=True):
        partition = (
            root
            / f"market_state_axis_panel_{int(interval)}"
            / f"band={_safe_part(band, field_name='band')}"
            / f"axis={_safe_part(axis, field_name='axis')}"
            / f"year={int(year):04d}"
            / f"month={int(month):02d}"
        )
        written.append(_write_frame(group.drop(columns=["_year", "_month"], errors="ignore"), partition, root=root, file_format=file_format))
    return written


def _write_frame(frame: pd.DataFrame, partition: Path, *, root: Path, file_format: str) -> tuple[Path, str]:
    _ensure_within_root(partition, root)
    partition.mkdir(parents=True, exist_ok=True)
    if file_format == MARKET_STATE_FEATURE_WRITE_FORMAT_JSONL:
        path = partition / "part-000.jsonl"
        _write_jsonl_frame(path, frame)
        return path, MARKET_STATE_FEATURE_WRITE_FORMAT_JSONL
    if file_format == MARKET_STATE_FEATURE_WRITE_FORMAT_PARQUET:
        path = partition / "part-000.parquet"
        frame.to_parquet(path, index=False)
        return path, MARKET_STATE_FEATURE_WRITE_FORMAT_PARQUET
    parquet_path = partition / "part-000.parquet"
    try:
        frame.to_parquet(parquet_path, index=False)
        return parquet_path, MARKET_STATE_FEATURE_WRITE_FORMAT_PARQUET
    except Exception:
        jsonl_path = partition / "part-000.jsonl"
        _write_jsonl_frame(jsonl_path, frame)
        return jsonl_path, MARKET_STATE_FEATURE_WRITE_FORMAT_JSONL


def _validate_market_feature_frame(frame: pd.DataFrame, *, family: str) -> None:
    missing = [column for column in FEATURE_REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Market-State feature rows for {family!r} missing required columns: {missing}")
    observed = set(frame["feature_family_id"].dropna().astype(str))
    if observed and observed != {str(family)}:
        raise ValueError(f"Market-State feature family key {family!r} does not match frame feature_family_id values {sorted(observed)}")


def _validate_axis_panel_frame(frame: pd.DataFrame, *, axis: str) -> None:
    missing = [column for column in AXIS_REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Market-State axis panel rows for {axis!r} missing required columns: {missing}")
    observed = set(frame["axis"].dropna().astype(str))
    if observed and observed != {str(axis)}:
        raise ValueError(f"Market-State axis key {axis!r} does not match frame axis values {sorted(observed)}")
    if "market_state" in frame.columns:
        raise ValueError("Market-State axis panel writer refuses composite market_state label columns")
    if not frame["feature_schema_id"].dropna().astype(str).eq(MARKET_STATE_AXIS_PANEL_SCHEMA_ID).all():
        raise ValueError("Market-State axis panel rows must carry the Market-State v1 axis panel schema id")


def _normalize_market_feature_rows(rows: Mapping[str, pd.DataFrame] | Sequence[pd.DataFrame]) -> dict[str, pd.DataFrame]:
    if isinstance(rows, Mapping):
        return {str(family): frame.copy() for family, frame in rows.items() if isinstance(frame, pd.DataFrame) and not frame.empty}
    out: dict[str, list[pd.DataFrame]] = {}
    for frame in rows:
        if not isinstance(frame, pd.DataFrame) or frame.empty or "feature_family_id" not in frame.columns:
            continue
        for family, group in frame.groupby(frame["feature_family_id"].astype(str), dropna=False):
            out.setdefault(str(family), []).append(group.copy())
    return {family: pd.concat(frames, ignore_index=True) for family, frames in out.items()}


def _normalize_axis_panel_rows(rows: Mapping[str, pd.DataFrame] | Sequence[pd.DataFrame]) -> dict[str, pd.DataFrame]:
    if isinstance(rows, Mapping):
        return {str(axis): frame.copy() for axis, frame in rows.items() if isinstance(frame, pd.DataFrame) and not frame.empty}
    out: dict[str, list[pd.DataFrame]] = {}
    for frame in rows:
        if not isinstance(frame, pd.DataFrame) or frame.empty or "axis" not in frame.columns:
            continue
        for axis, group in frame.groupby(frame["axis"].astype(str), dropna=False):
            out.setdefault(str(axis), []).append(group.copy())
    return {axis: pd.concat(frames, ignore_index=True) for axis, frames in out.items()}


def _with_time_partitions(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.copy()
    work["ts"] = pd.to_numeric(work["ts"], errors="coerce")
    work["interval"] = pd.to_numeric(work["interval"], errors="coerce")
    work = work.dropna(subset=["ts", "interval", "band"]).copy()
    if work.empty:
        raise ValueError("Market-State v1 feature writer has no rows with finite ts/interval/band")
    work["ts"] = work["ts"].astype("int64")
    work["interval"] = work["interval"].astype("int64")
    dt = pd.to_datetime(work["ts"], unit="s", utc=True)
    work["_year"] = dt.dt.year.astype("int64")
    work["_month"] = dt.dt.month.astype("int64")
    return work


def _manifest_payload(
    request: MarketStateFeatureMaterializationRequest,
    *,
    root: Path,
    market_rows: Mapping[str, pd.DataFrame],
    axis_rows: Mapping[str, pd.DataFrame],
    written: Sequence[Path],
    file_format_counts: Mapping[str, int],
    artifact_boundary: Mapping[str, Any],
    universe_reference: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": MARKET_STATE_FEATURE_MANIFEST_SCHEMA_VERSION,
        "artifact_kind": "market_state_v1_feature_materialization_manifest",
        "run_id": request.run_id,
        "output_root": "runtime_only_not_serialized",
        "artifact_boundary": to_jsonable(dict(artifact_boundary)),
        "universe_manifest_reference": to_jsonable(dict(universe_reference)),
        "market_feature_families": {family: _frame_summary(frame) for family, frame in market_rows.items()},
        "axis_panels": {axis: _frame_summary(frame) for axis, frame in axis_rows.items()},
        "written_paths": [_relative_path(path, root) for path in written],
        "artifact_refs": [
            make_artifact_ref(
                path,
                artifact_kind=_artifact_kind(path),
                artifact_root=root,
                producer="src.regimes.market_state.feature_writer",
            ).as_dict()
            for path in written
            if path.exists() and path.is_file()
        ],
        "file_format_counts": dict(sorted(file_format_counts.items())),
        "feature_schema_ids": _feature_schema_ids(market_rows, axis_rows),
        "lineage_ids": sorted(_collect_values((*market_rows.values(), *axis_rows.values()), "lineage_id")),
        "known_at": _known_at_span((*market_rows.values(), *axis_rows.values())),
        "metadata": to_jsonable(dict(request.metadata)),
    }


def _frame_summary(frame: pd.DataFrame) -> dict[str, Any]:
    return {
        "row_count": int(frame.shape[0]),
        "bands": sorted(str(value) for value in frame.get("band", pd.Series(dtype=object)).dropna().unique()),
        "intervals": sorted(int(value) for value in pd.to_numeric(frame.get("interval", pd.Series(dtype=float)), errors="coerce").dropna().unique()),
        "schema_versions": sorted(int(value) for value in pd.to_numeric(frame.get("schema_version", pd.Series(dtype=float)), errors="coerce").dropna().unique()),
        "universe_snapshot_ids": sorted(str(value) for value in frame.get("universe_snapshot_id", pd.Series(dtype=object)).dropna().unique()),
        "universe_snapshot_hashes": sorted(str(value) for value in frame.get("universe_snapshot_hash", pd.Series(dtype=object)).dropna().unique()),
    }


def _feature_schema_ids(market_rows: Mapping[str, pd.DataFrame], axis_rows: Mapping[str, pd.DataFrame]) -> dict[str, list[str]]:
    market_versions = sorted(
        {str(value) for frame in market_rows.values() for value in frame.get("schema_version", pd.Series(dtype=object)).dropna().unique()}
    )
    axis_schema_ids = sorted(
        {str(value) for frame in axis_rows.values() for value in frame.get("feature_schema_id", pd.Series(dtype=object)).dropna().unique()}
    )
    return {"market_feature_schema_versions": market_versions, "axis_panel_schema_ids": axis_schema_ids}


def _known_at_span(frames: Sequence[pd.DataFrame]) -> dict[str, int | None]:
    values: list[int] = []
    for frame in frames:
        if "known_at_ts" in frame.columns:
            values.extend(int(value) for value in pd.to_numeric(frame["known_at_ts"], errors="coerce").dropna())
    return {"min_known_at_ts": min(values) if values else None, "max_known_at_ts": max(values) if values else None}


def _collect_values(frames: Sequence[pd.DataFrame], column: str) -> set[str]:
    out: set[str] = set()
    for frame in frames:
        if column in frame.columns:
            out.update(str(value) for value in frame[column].dropna().unique() if str(value).strip())
    return out


def _universe_reference_payload(reference: Mapping[str, Any] | str | Path | None) -> dict[str, Any]:
    if reference is None:
        return {"status": "not_supplied"}
    if isinstance(reference, Mapping):
        return dict(to_jsonable(dict(reference)))
    path = Path(reference)
    return {"manifest_path": str(path), "manifest_name": path.name}


def _artifact_boundary() -> dict[str, Any]:
    return {
        "production_enabled": False,
        "production_outputs_written": False,
        "clustering_performed": False,
        "promotion_allowed": False,
        "production_promotion_performed": False,
        "labels_written": False,
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(to_jsonable(dict(payload)), handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(tmp, path)


def _write_jsonl_frame(path: Path, frame: pd.DataFrame) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in frame.to_dict(orient="records"):
            handle.write(json.dumps(to_jsonable(dict(row)), sort_keys=True))
            handle.write("\n")
    os.replace(tmp, path)


def _file_format(value: str) -> str:
    text = str(value).strip().lower()
    valid = {
        MARKET_STATE_FEATURE_WRITE_FORMAT_AUTO,
        MARKET_STATE_FEATURE_WRITE_FORMAT_PARQUET,
        MARKET_STATE_FEATURE_WRITE_FORMAT_JSONL,
    }
    if text not in valid:
        raise ValueError(f"Unsupported Market-State feature writer file_format {value!r}")
    return text


def _safe_part(value: object, *, field_name: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"Market-State feature writer {field_name} must be non-empty")
    if any(part in text for part in ("/", "\\", ":", "\x00")):
        raise ValueError(f"Market-State feature writer {field_name} contains unsafe path characters")
    return re.sub(r"[^A-Za-z0-9_.=-]+", "_", text)


def _ensure_within_root(path: Path, root: Path) -> None:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"Market-State feature writer refusing to write outside output root: {path}") from exc


def _relative_path(path: Path, root: Path) -> str:
    return validate_portable_relative_path(path.resolve().relative_to(root.resolve()).as_posix())


def _artifact_kind(path: Path) -> str:
    name = path.name
    if name == "manifest.json":
        return "market_state_feature_manifest"
    if name == "build_summary.json":
        return "market_state_feature_build_summary"
    if name == "universe_manifest_reference.json":
        return "market_state_universe_reference"
    if "market_state_axis_panel" in str(path):
        return "market_state_axis_panel"
    return "market_state_feature_rows"


__all__ = [
    "AXIS_REQUIRED_COLUMNS",
    "FEATURE_REQUIRED_COLUMNS",
    "MARKET_STATE_FEATURE_MANIFEST_SCHEMA_VERSION",
    "MARKET_STATE_FEATURE_WRITE_FORMAT_AUTO",
    "MARKET_STATE_FEATURE_WRITE_FORMAT_JSONL",
    "MARKET_STATE_FEATURE_WRITE_FORMAT_PARQUET",
    "MARKET_STATE_FEATURE_WRITE_STATUS_NO_ROWS",
    "MARKET_STATE_FEATURE_WRITE_STATUS_WRITTEN",
    "MarketStateFeatureMaterializationRequest",
    "MarketStateFeatureMaterializationResult",
    "validate_market_state_feature_write_root",
    "write_market_state_v1_feature_materialization",
]
