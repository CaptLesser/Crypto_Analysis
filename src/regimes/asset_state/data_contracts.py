from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .contracts import (
    ASSET_STATE_SCHEMA_VERSION,
    AssetStateAxis,
    AssetStateBand,
    _enum_value,
)


DATASET_BUILD_STATUS_READY = "ready"
DATASET_BUILD_STATUS_BLOCKED = "blocked"

DATASET_REASON_MISSING_SOURCE_ROOT = "missing_source_feature_root"
DATASET_REASON_MISSING_INTERVAL_ROOT = "missing_interval_root"
DATASET_REASON_MISSING_ASSET_PARTITION = "missing_asset_partition"
DATASET_REASON_MISSING_PARTITION = "missing_partition"
DATASET_REASON_EMPTY_WINDOW = "empty_window"
DATASET_REASON_LEAKAGE_RISK_COLUMNS = "leakage_risk_columns"
DATASET_REASON_MISSING_FEATURE_COLUMNS = "missing_feature_columns"
DATASET_REASON_INSUFFICIENT_TRAIN_ROWS = "insufficient_train_rows"
DATASET_REASON_INSUFFICIENT_VALIDATION_ROWS = "insufficient_validation_rows"
DATASET_REASON_INSUFFICIENT_HOLDOUT_ROWS = "insufficient_holdout_rows"
DATASET_REASON_NO_RETAINED_TRAIN_FEATURES = "no_retained_train_features"
DATASET_REASON_NONFINITE_FEATURE_MATRIX = "nonfinite_feature_matrix"
DATASET_REASON_PREPROCESSING_FAILED = "preprocessing_failed"
DATASET_REASON_UNSAFE_DIAGNOSTICS_ROOT = "unsafe_diagnostics_root"
DATASET_REASON_MALFORMED_PARTITION = "malformed_partition"
DATASET_REASON_BUILDER_EXCEPTION = "builder_exception"


@dataclass(frozen=True)
class AssetStateSplitSpec:
    train_fraction: float = 0.6
    validation_fraction: float = 0.2
    holdout_fraction: float = 0.2
    train_start_ts: int | None = None
    train_end_ts: int | None = None
    validation_start_ts: int | None = None
    validation_end_ts: int | None = None
    holdout_start_ts: int | None = None
    holdout_end_ts: int | None = None
    timestamp_column: str = "ts"

    def __post_init__(self) -> None:
        if not self.timestamp_column:
            raise ValueError("timestamp_column is required")
        if self.uses_explicit_windows:
            for start, end, name in (
                (self.train_start_ts, self.train_end_ts, "train"),
                (
                    self.validation_start_ts,
                    self.validation_end_ts,
                    "validation",
                ),
                (self.holdout_start_ts, self.holdout_end_ts, "holdout"),
            ):
                if (start is None) != (end is None):
                    raise ValueError(f"{name} split requires both start and end timestamps")
                if start is not None and end is not None and int(start) >= int(end):
                    raise ValueError(f"{name} split start must be before end")
            return

        total = self.train_fraction + self.validation_fraction + self.holdout_fraction
        if self.train_fraction <= 0:
            raise ValueError("train_fraction must be positive")
        if self.validation_fraction < 0 or self.holdout_fraction < 0:
            raise ValueError("validation_fraction and holdout_fraction cannot be negative")
        if total <= 0:
            raise ValueError("at least one split fraction must be positive")

    @property
    def uses_explicit_windows(self) -> bool:
        return (
            self.train_start_ts is not None
            or self.train_end_ts is not None
            or self.validation_start_ts is not None
            or self.validation_end_ts is not None
            or self.holdout_start_ts is not None
            or self.holdout_end_ts is not None
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "train_fraction": self.train_fraction,
            "validation_fraction": self.validation_fraction,
            "holdout_fraction": self.holdout_fraction,
            "train_start_ts": self.train_start_ts,
            "train_end_ts": self.train_end_ts,
            "validation_start_ts": self.validation_start_ts,
            "validation_end_ts": self.validation_end_ts,
            "holdout_start_ts": self.holdout_start_ts,
            "holdout_end_ts": self.holdout_end_ts,
            "timestamp_column": self.timestamp_column,
            "uses_explicit_windows": self.uses_explicit_windows,
        }


@dataclass(frozen=True)
class AssetStateStudyDatasetRequest:
    asset: str
    axis: AssetStateAxis | str
    band: AssetStateBand | str
    source_feature_root: str | Path | None = None
    start_ts: int | None = None
    end_ts: int | None = None
    split: AssetStateSplitSpec = field(default_factory=AssetStateSplitSpec)
    feature_pool_id: str | None = None
    preprocess: str = "robust_scale"
    preprocess_params: Mapping[str, Any] = field(default_factory=dict)
    forward_horizon_min: int | None = None
    require_all_member_intervals: bool = True
    min_train_rows: int = 32
    min_validation_rows: int = 1
    min_holdout_rows: int = 0
    min_feature_count: int = 1
    diagnostics_root: str | Path | None = None
    write_diagnostics: bool = False
    schema_version: str = ASSET_STATE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.asset or "," in self.asset:
            raise ValueError("asset must name one asset")
        if self.start_ts is not None and self.end_ts is not None and int(self.start_ts) >= int(self.end_ts):
            raise ValueError("start_ts must be before end_ts")
        if self.min_train_rows < 1:
            raise ValueError("min_train_rows must be positive")
        if self.min_validation_rows < 0 or self.min_holdout_rows < 0:
            raise ValueError("minimum split rows cannot be negative")
        if self.min_feature_count < 1:
            raise ValueError("min_feature_count must be positive")

    @property
    def axis_id(self) -> str:
        return _enum_value(self.axis, AssetStateAxis, field_name="axis")

    @property
    def band_id(self) -> str:
        return _enum_value(self.band, AssetStateBand, field_name="band")

    @property
    def source_root(self) -> Path:
        if self.source_feature_root is None or not str(self.source_feature_root).strip():
            return Path()
        return Path(self.source_feature_root)

    def as_dict(self) -> dict[str, Any]:
        return {
            "asset": self.asset,
            "axis": self.axis_id,
            "band": self.band_id,
            "source_feature_root": str(self.source_feature_root) if self.source_feature_root is not None else None,
            "start_ts": self.start_ts,
            "end_ts": self.end_ts,
            "split": self.split.as_dict(),
            "feature_pool_id": self.feature_pool_id,
            "preprocess": self.preprocess,
            "preprocess_params": dict(self.preprocess_params),
            "forward_horizon_min": self.forward_horizon_min,
            "require_all_member_intervals": self.require_all_member_intervals,
            "min_train_rows": self.min_train_rows,
            "min_validation_rows": self.min_validation_rows,
            "min_holdout_rows": self.min_holdout_rows,
            "min_feature_count": self.min_feature_count,
            "diagnostics_root": str(self.diagnostics_root) if self.diagnostics_root is not None else None,
            "write_diagnostics": self.write_diagnostics,
            "schema_version": self.schema_version,
        }


@dataclass(frozen=True)
class PartitionLineage:
    interval: int
    asset: str
    path: str
    root: str
    row_count: int
    min_ts: int | None
    max_ts: int | None
    columns: tuple[str, ...]
    year: int | None = None
    month: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "interval": self.interval,
            "asset": self.asset,
            "path": self.path,
            "root": self.root,
            "row_count": self.row_count,
            "min_ts": self.min_ts,
            "max_ts": self.max_ts,
            "columns": list(self.columns),
            "year": self.year,
            "month": self.month,
        }


@dataclass
class SplitMatrix:
    name: str
    x: np.ndarray = field(repr=False)
    timestamps: tuple[int, ...]
    frame: pd.DataFrame = field(repr=False)
    targets: pd.DataFrame | None = field(default=None, repr=False)
    row_count_before_cleaning: int = 0
    dropped_rows: int = 0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def row_count(self) -> int:
        return int(self.x.shape[0])

    @property
    def feature_count(self) -> int:
        if self.x.ndim != 2:
            return 0
        return int(self.x.shape[1])

    def as_dict(self) -> dict[str, Any]:
        target_columns: list[str] = []
        target_missing_fraction: dict[str, float] = {}
        if self.targets is not None and not self.targets.empty:
            target_columns = [c for c in self.targets.columns if c != "ts"]
            target_missing_fraction = {
                c: float(self.targets[c].isna().mean())
                for c in target_columns
            }
        return {
            "name": self.name,
            "shape": [self.row_count, self.feature_count],
            "row_count_before_cleaning": self.row_count_before_cleaning,
            "dropped_rows": self.dropped_rows,
            "timestamp_start": self.timestamps[0] if self.timestamps else None,
            "timestamp_end": self.timestamps[-1] if self.timestamps else None,
            "timestamp_count": len(self.timestamps),
            "target_columns": target_columns,
            "target_missing_fraction": target_missing_fraction,
            "metadata": dict(self.metadata),
        }


@dataclass
class DatasetBuildResult:
    status: str
    reason_codes: tuple[str, ...]
    asset: str
    axis: str
    band: str
    schema_version: str
    train: SplitMatrix
    validation: SplitMatrix
    holdout: SplitMatrix | None = None
    feature_columns: tuple[str, ...] = ()
    output_feature_names: tuple[str, ...] = ()
    scalar_feature_columns_available: tuple[str, ...] = ()
    partition_lineage: tuple[PartitionLineage, ...] = ()
    missingness_summary: Mapping[str, Any] = field(default_factory=dict)
    row_counts: Mapping[str, int] = field(default_factory=dict)
    dropped_rows: Mapping[str, int] = field(default_factory=dict)
    feature_count: int = 0
    diagnostics_path: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    message: str | None = None

    @property
    def usable(self) -> bool:
        return self.status == DATASET_BUILD_STATUS_READY

    @property
    def X_train(self) -> np.ndarray:
        return self.train.x

    @property
    def X_validation(self) -> np.ndarray:
        return self.validation.x

    @property
    def X_score(self) -> np.ndarray:
        return self.validation.x

    @property
    def X_holdout(self) -> np.ndarray | None:
        return self.holdout.x if self.holdout is not None else None

    @property
    def train_timestamps(self) -> tuple[int, ...]:
        return self.train.timestamps

    @property
    def validation_timestamps(self) -> tuple[int, ...]:
        return self.validation.timestamps

    @property
    def holdout_timestamps(self) -> tuple[int, ...]:
        return self.holdout.timestamps if self.holdout is not None else ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "reason_codes": list(self.reason_codes),
            "asset": self.asset,
            "axis": self.axis,
            "band": self.band,
            "schema_version": self.schema_version,
            "feature_columns": list(self.feature_columns),
            "output_feature_names": list(self.output_feature_names),
            "scalar_feature_columns_available": list(self.scalar_feature_columns_available),
            "feature_count": self.feature_count,
            "row_counts": dict(self.row_counts),
            "dropped_rows": dict(self.dropped_rows),
            "missingness_summary": dict(self.missingness_summary),
            "partition_lineage": [p.as_dict() for p in self.partition_lineage],
            "splits": {
                "train": self.train.as_dict(),
                "validation": self.validation.as_dict(),
                "holdout": self.holdout.as_dict() if self.holdout is not None else None,
            },
            "diagnostics_path": self.diagnostics_path,
            "metadata": dict(self.metadata),
            "message": self.message,
        }


def empty_split_matrix(name: str) -> SplitMatrix:
    return SplitMatrix(
        name=name,
        x=np.empty((0, 0), dtype=float),
        timestamps=(),
        frame=pd.DataFrame(),
        targets=None,
        row_count_before_cleaning=0,
        dropped_rows=0,
        metadata={},
    )


def blocked_dataset_result(
    request: AssetStateStudyDatasetRequest,
    reason_codes: Sequence[str],
    *,
    message: str | None = None,
    partition_lineage: Sequence[PartitionLineage] = (),
    metadata: Mapping[str, Any] | None = None,
    missingness_summary: Mapping[str, Any] | None = None,
) -> DatasetBuildResult:
    return DatasetBuildResult(
        status=DATASET_BUILD_STATUS_BLOCKED,
        reason_codes=tuple(dict.fromkeys(reason_codes)),
        asset=request.asset,
        axis=request.axis_id,
        band=request.band_id,
        schema_version=request.schema_version,
        train=empty_split_matrix("train"),
        validation=empty_split_matrix("validation"),
        holdout=None,
        partition_lineage=tuple(partition_lineage),
        missingness_summary=missingness_summary or {},
        row_counts={},
        dropped_rows={},
        feature_count=0,
        metadata=metadata or {},
        message=message,
    )
