from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from src.regimes.asset_state.contracts import (
    ASSET_STATE_SCHEMA_VERSION,
    AssetStateAxis,
    AssetStateBand,
    AssetStateSchemaVersion,
)
from src.regimes.asset_state.contracts import _enum_value, _schema_version, _string_tuple
from src.regimes.asset_state.feature_registry import (
    canonical_interval_feature_name,
    default_asset_state_feature_pool_registry,
)
from src.regimes.asset_state.taxonomy import default_asset_state_taxonomy
from src.regimes.core.serialization import to_jsonable


class AssetStateDataError(ValueError):
    pass


LEAKAGE_COLUMN_TOKENS: tuple[str, ...] = (
    "future_",
    "forward_",
    "_target",
    "target_",
    "_label",
    "label_",
    "regime_",
)


@dataclass(frozen=True)
class AssetStateDatasetBuildRequest:
    axis: str | AssetStateAxis
    band: str | AssetStateBand
    assets: Sequence[str]
    source_feature_root: Path | str
    start_ts: int | None = None
    end_ts: int | None = None
    feature_pool_id: str | None = None
    require_all_member_intervals: bool = True
    min_rows: int = 32
    schema_version: int | AssetStateSchemaVersion = ASSET_STATE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        taxonomy = default_asset_state_taxonomy()
        axis = _enum_value(self.axis, AssetStateAxis, field_name="axis")
        band = _enum_value(self.band, AssetStateBand, field_name="band")
        taxonomy.validate_axis_band(axis, band)
        assets = _string_tuple(self.assets, field_name="assets", require_non_empty=True)
        if self.start_ts is not None and self.end_ts is not None and int(self.start_ts) > int(self.end_ts):
            raise ValueError("Asset-state dataset start_ts must be <= end_ts")
        if int(self.min_rows) < 1:
            raise ValueError("Asset-state dataset min_rows must be positive")
        object.__setattr__(self, "schema_version", _schema_version(self.schema_version))
        object.__setattr__(self, "axis", axis)
        object.__setattr__(self, "band", band)
        object.__setattr__(self, "assets", assets)
        object.__setattr__(self, "source_feature_root", Path(self.source_feature_root))
        object.__setattr__(self, "min_rows", int(self.min_rows))


@dataclass(frozen=True)
class AssetStateDataset:
    frame: pd.DataFrame = field(repr=False, compare=False)
    feature_columns: tuple[str, ...]
    metadata: Mapping[str, Any]
    schema_version: int | AssetStateSchemaVersion = ASSET_STATE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "schema_version", _schema_version(self.schema_version))
        object.__setattr__(self, "feature_columns", tuple(str(column) for column in self.feature_columns))
        object.__setattr__(self, "metadata", to_jsonable(dict(self.metadata)))

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "row_count": int(len(self.frame)),
            "feature_columns": list(self.feature_columns),
            "metadata": to_jsonable(dict(self.metadata)),
        }


def _path_has_part(path: Path, part: str) -> bool:
    return any(piece.lower() == part.lower() for piece in path.parts)


def _partition_value(path: Path, key: str) -> str | None:
    prefix = f"{key}="
    for part in path.parts:
        if part.startswith(prefix):
            return part[len(prefix) :]
    return None


def _interval_root_candidates(source_root: Path, interval: int) -> tuple[Path, ...]:
    name = f"scalar_features_{int(interval)}"
    candidates = [
        source_root / name,
        source_root / "scalar_features" / name,
        source_root / "model_states" / "scalar_features" / name,
    ]
    if source_root.name == name:
        candidates.insert(0, source_root)
    return tuple(dict.fromkeys(path for path in candidates if path.exists() and path.is_dir()))


def _parquet_parts_for_interval(source_root: Path, interval: int, asset: str) -> tuple[Path, ...]:
    candidates = _interval_root_candidates(source_root, interval)
    if not candidates:
        return ()
    out: list[Path] = []
    for root in candidates:
        for path in sorted(root.rglob("*.parquet")):
            partition_asset = _partition_value(path, "asset")
            if partition_asset is None or partition_asset == asset:
                out.append(path)
    return tuple(dict.fromkeys(out))


def _detect_leakage_columns(columns: Sequence[str]) -> tuple[str, ...]:
    unsafe: list[str] = []
    for column in columns:
        lowered = str(column).lower()
        if column in {"ts", "asset", "year", "month"}:
            continue
        if any(token in lowered for token in LEAKAGE_COLUMN_TOKENS):
            unsafe.append(str(column))
    return tuple(unsafe)


def _read_interval_asset_frame(
    *,
    source_root: Path,
    interval: int,
    asset: str,
    start_ts: int | None,
    end_ts: int | None,
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    parts = _parquet_parts_for_interval(source_root, interval, asset)
    if not parts:
        raise AssetStateDataError(f"missing_scalar_feature_parquet interval={interval} asset={asset}")
    frames: list[pd.DataFrame] = []
    artifacts: list[str] = []
    for path in parts:
        try:
            frame = pd.read_parquet(path)
        except Exception as exc:
            raise AssetStateDataError(f"failed_reading_scalar_feature_parquet path={path}: {exc}") from exc
        if "ts" not in frame.columns:
            raise AssetStateDataError(f"malformed_scalar_feature_parquet_missing_ts path={path}")
        unsafe = _detect_leakage_columns(tuple(str(column) for column in frame.columns))
        if unsafe:
            raise AssetStateDataError(f"leakage_risk_columns interval={interval} asset={asset}: {', '.join(unsafe)}")
        frame = frame.copy()
        if "asset" not in frame.columns:
            partition_asset = _partition_value(path, "asset")
            frame["asset"] = partition_asset or str(asset)
        frame["asset"] = frame["asset"].astype(str)
        frame = frame.loc[frame["asset"] == str(asset)].copy()
        frame["ts"] = pd.to_numeric(frame["ts"], errors="coerce")
        frame = frame.loc[frame["ts"].notna()].copy()
        frame["ts"] = frame["ts"].astype("int64")
        if start_ts is not None:
            frame = frame.loc[frame["ts"] >= int(start_ts)]
        if end_ts is not None:
            frame = frame.loc[frame["ts"] <= int(end_ts)]
        if not frame.empty:
            frames.append(frame)
            artifacts.append(str(path.resolve()))
    if not frames:
        raise AssetStateDataError(f"empty_scalar_feature_window interval={interval} asset={asset}")
    return pd.concat(frames, ignore_index=True).drop_duplicates(subset=["asset", "ts"], keep="last"), tuple(artifacts)


def _canonicalize_interval_columns(frame: pd.DataFrame, *, interval: int, feature_bases: Sequence[str]) -> tuple[pd.DataFrame, tuple[str, ...], tuple[str, ...]]:
    out = frame[["asset", "ts"]].copy()
    selected: list[str] = []
    missing: list[str] = []
    for base in feature_bases:
        canonical = canonical_interval_feature_name(interval, base)
        source = canonical if canonical in frame.columns else str(base) if str(base) in frame.columns else None
        if source is None:
            missing.append(str(base))
            continue
        out[canonical] = pd.to_numeric(frame[source], errors="coerce")
        selected.append(canonical)
    return out, tuple(selected), tuple(missing)


def _feature_schema_hash(columns: Sequence[str]) -> str:
    payload = "\n".join(str(column) for column in columns).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def build_asset_state_scalar_feature_dataset(request: AssetStateDatasetBuildRequest) -> AssetStateDataset:
    source_root = Path(request.source_feature_root).expanduser()
    if not source_root.exists() or not source_root.is_dir():
        raise AssetStateDataError(f"missing_source_feature_root path={source_root}")
    if _path_has_part(source_root.resolve(), "regime_definitions"):
        raise AssetStateDataError("unsafe_source_feature_root regime_definitions is not a scalar-feature source root")
    taxonomy = default_asset_state_taxonomy()
    band_spec = taxonomy.band_spec(request.band)
    registry = default_asset_state_feature_pool_registry()
    pool = registry.get(request.feature_pool_id) if request.feature_pool_id else registry.default_for_axis(request.axis, band=request.band)
    if not pool.supports(axis=request.axis, band=request.band):
        raise AssetStateDataError(f"incompatible_feature_pool pool_id={pool.pool_id} axis={request.axis} band={request.band}")
    all_frames: list[pd.DataFrame] = []
    all_feature_columns: list[str] = []
    source_artifacts: list[str] = []
    missing_intervals: dict[str, list[int]] = {}
    missing_bases_by_interval: dict[str, dict[str, list[str]]] = {}
    for asset in request.assets:
        asset_frame: pd.DataFrame | None = None
        for interval in band_spec.member_intervals:
            try:
                raw, artifacts = _read_interval_asset_frame(
                    source_root=source_root,
                    interval=int(interval),
                    asset=str(asset),
                    start_ts=request.start_ts,
                    end_ts=request.end_ts,
                )
            except AssetStateDataError as exc:
                message = str(exc)
                hard_block_tokens = (
                    "leakage_risk",
                    "malformed_scalar_feature",
                    "failed_reading_scalar_feature",
                    "unsafe_source_feature_root",
                )
                if request.require_all_member_intervals or any(token in message for token in hard_block_tokens):
                    raise
                missing_intervals.setdefault(str(asset), []).append(int(interval))
                continue
            canonical, selected, missing_bases = _canonicalize_interval_columns(
                raw,
                interval=int(interval),
                feature_bases=pool.feature_bases,
            )
            if not selected:
                if request.require_all_member_intervals:
                    raise AssetStateDataError(
                        f"missing_intended_feature_bases interval={interval} asset={asset} pool_id={pool.pool_id}"
                    )
                missing_intervals.setdefault(str(asset), []).append(int(interval))
                missing_bases_by_interval.setdefault(str(asset), {})[str(interval)] = list(missing_bases)
                continue
            missing_bases_by_interval.setdefault(str(asset), {})[str(interval)] = list(missing_bases)
            source_artifacts.extend(artifacts)
            all_feature_columns.extend(selected)
            asset_frame = canonical if asset_frame is None else asset_frame.merge(canonical, on=["asset", "ts"], how="outer")
        if asset_frame is not None and not asset_frame.empty:
            all_frames.append(asset_frame)
    if not all_frames:
        raise AssetStateDataError("no_asset_state_feature_rows_built")
    frame = pd.concat(all_frames, ignore_index=True).sort_values(["asset", "ts"]).reset_index(drop=True)
    feature_columns = tuple(dict.fromkeys(all_feature_columns))
    if not feature_columns:
        raise AssetStateDataError("no_asset_state_feature_columns_built")
    if int(len(frame)) < int(request.min_rows):
        raise AssetStateDataError(f"insufficient_dataset_rows row_count={len(frame)} min_rows={request.min_rows}")
    metadata = {
        "artifact_kind": "asset_state_scalar_feature_dataset",
        "pathway": "asset_state",
        "axis": request.axis,
        "band": request.band,
        "assets": list(request.assets),
        "asset_count": int(len(request.assets)),
        "source_feature_root": str(source_root.resolve()),
        "source_artifacts": sorted(set(source_artifacts)),
        "real_scalar_feature_parquet_used": True,
        "feature_pool_id": pool.pool_id,
        "feature_schema_hash": _feature_schema_hash(feature_columns),
        "member_intervals": list(band_spec.member_intervals),
        "missing_intervals": missing_intervals,
        "missing_bases_by_interval": missing_bases_by_interval,
        "start_ts": request.start_ts,
        "end_ts": request.end_ts,
        "row_count": int(len(frame)),
        "feature_count": int(len(feature_columns)),
        "production_outputs_written": False,
    }
    return AssetStateDataset(frame=frame, feature_columns=feature_columns, metadata=metadata)


__all__ = [
    "AssetStateDataError",
    "AssetStateDataset",
    "AssetStateDatasetBuildRequest",
    "LEAKAGE_COLUMN_TOKENS",
    "build_asset_state_scalar_feature_dataset",
]
