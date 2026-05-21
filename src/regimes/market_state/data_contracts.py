from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from src.regimes.core.clamp_policy import RegimeClampPolicy
from src.regimes.core.serialization import dumps_json, loads_json, require_json_object, to_jsonable
from src.regimes.market_state.contracts import (
    MARKET_STATE_SCHEMA_VERSION,
    MarketStateBand,
    MarketStateSchemaVersion,
    _enum_value,
    _schema_version,
    _string_tuple,
)


MARKET_STATE_DATASET_STATUS_READY = "ready"
MARKET_STATE_DATASET_STATUS_BLOCKED = "blocked"

MARKET_STATE_DATASET_REASON_MISSING_SOURCE_ROOT = "missing_source_root"
MARKET_STATE_DATASET_REASON_EMPTY_WINDOW = "empty_window"
MARKET_STATE_DATASET_REASON_TOO_FEW_TIMESTAMPS = "too_few_timestamps"
MARKET_STATE_DATASET_REASON_TOO_FEW_CORE_ASSETS = "too_few_core_assets"
MARKET_STATE_DATASET_REASON_INSUFFICIENT_BROAD_UNIVERSE = "insufficient_broad_universe_coverage"
MARKET_STATE_DATASET_REASON_LEAKAGE_RISK_COLUMNS = "leakage_risk_columns"
MARKET_STATE_DATASET_REASON_MALFORMED_DATA = "malformed_data"
MARKET_STATE_DATASET_REASON_BUILDER_EXCEPTION = "builder_exception"

MARKET_STATE_DATASET_SOURCE_AUTO = "auto"
MARKET_STATE_DATASET_SOURCE_OHLCVT = "ohlcvt"
MARKET_STATE_DATASET_SOURCE_SCALAR_FEATURES = "scalar_features"
MARKET_STATE_DATASET_SOURCE_KINDS: tuple[str, ...] = (
    MARKET_STATE_DATASET_SOURCE_AUTO,
    MARKET_STATE_DATASET_SOURCE_OHLCVT,
    MARKET_STATE_DATASET_SOURCE_SCALAR_FEATURES,
)


@dataclass(frozen=True)
class MarketStateSplitPolicy:
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
        if not str(self.timestamp_column).strip():
            raise ValueError("Market-state split timestamp_column is required")
        if self.uses_explicit_windows:
            for name, start, end in (
                ("train", self.train_start_ts, self.train_end_ts),
                ("validation", self.validation_start_ts, self.validation_end_ts),
                ("holdout", self.holdout_start_ts, self.holdout_end_ts),
            ):
                if (start is None) != (end is None):
                    raise ValueError(f"Market-state {name} split requires both start and end timestamps")
                if start is not None and end is not None and int(start) >= int(end):
                    raise ValueError(f"Market-state {name} split start must be before end")
            return
        total = float(self.train_fraction) + float(self.validation_fraction) + float(self.holdout_fraction)
        if float(self.train_fraction) <= 0.0:
            raise ValueError("Market-state train_fraction must be positive")
        if float(self.validation_fraction) < 0.0 or float(self.holdout_fraction) < 0.0:
            raise ValueError("Market-state validation_fraction and holdout_fraction cannot be negative")
        if total <= 0.0:
            raise ValueError("Market-state split fractions must sum to a positive value")

    @property
    def uses_explicit_windows(self) -> bool:
        return any(
            value is not None
            for value in (
                self.train_start_ts,
                self.train_end_ts,
                self.validation_start_ts,
                self.validation_end_ts,
                self.holdout_start_ts,
                self.holdout_end_ts,
            )
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "train_fraction": float(self.train_fraction),
            "validation_fraction": float(self.validation_fraction),
            "holdout_fraction": float(self.holdout_fraction),
            "train_start_ts": self.train_start_ts,
            "train_end_ts": self.train_end_ts,
            "validation_start_ts": self.validation_start_ts,
            "validation_end_ts": self.validation_end_ts,
            "holdout_start_ts": self.holdout_start_ts,
            "holdout_end_ts": self.holdout_end_ts,
            "timestamp_column": self.timestamp_column,
            "uses_explicit_windows": self.uses_explicit_windows,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "MarketStateSplitPolicy":
        obj = require_json_object(payload, context="MarketStateSplitPolicy")
        obj.pop("uses_explicit_windows", None)
        return cls(**obj)


@dataclass(frozen=True)
class MarketStateDatasetBuildRequest:
    source_root: str | Path
    interval: int
    band: str | MarketStateBand
    core_basket_assets: Sequence[str]
    broad_universe_assets: Sequence[str]
    start_ts: int | None = None
    end_ts: int | None = None
    split: MarketStateSplitPolicy | Mapping[str, Any] = field(default_factory=MarketStateSplitPolicy)
    source_kind: str = MARKET_STATE_DATASET_SOURCE_AUTO
    min_timestamp_count: int = 32
    min_core_asset_count: int = 2
    min_broad_asset_count: int = 2
    min_core_asset_timestamp_coverage: float = 0.8
    min_broad_asset_timestamp_coverage: float = 0.5
    min_broad_timestamp_coverage: float = 0.5
    clamp_policy: RegimeClampPolicy | Mapping[str, Any] | None = None
    schema_version: int | MarketStateSchemaVersion = MARKET_STATE_SCHEMA_VERSION
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        interval = int(self.interval)
        if interval <= 0:
            raise ValueError("Market-state dataset interval must be positive")
        if self.start_ts is not None and self.end_ts is not None and int(self.start_ts) >= int(self.end_ts):
            raise ValueError("Market-state dataset start_ts must be before end_ts")
        core = _string_tuple(self.core_basket_assets, field_name="core_basket_assets", require_non_empty=True)
        broad = _string_tuple(self.broad_universe_assets, field_name="broad_universe_assets", require_non_empty=True)
        source_kind = str(self.source_kind).strip().lower()
        if source_kind not in MARKET_STATE_DATASET_SOURCE_KINDS:
            raise ValueError(f"Unsupported market-state dataset source_kind {source_kind!r}")
        split = self.split if isinstance(self.split, MarketStateSplitPolicy) else MarketStateSplitPolicy.from_dict(self.split)
        min_timestamp_count = int(self.min_timestamp_count)
        min_core_assets = int(self.min_core_asset_count)
        min_broad_assets = int(self.min_broad_asset_count)
        if min_timestamp_count <= 0:
            raise ValueError("Market-state min_timestamp_count must be positive")
        if min_core_assets <= 0 or min_broad_assets <= 0:
            raise ValueError("Market-state minimum asset counts must be positive")
        clamp = None
        if self.clamp_policy is not None:
            clamp = (
                self.clamp_policy
                if isinstance(self.clamp_policy, RegimeClampPolicy)
                else RegimeClampPolicy.from_dict(self.clamp_policy)
            )
            if "market_state" not in clamp.applies_to_pathways:
                raise ValueError("Market-state dataset clamp_policy must apply to market_state")
        object.__setattr__(self, "schema_version", _schema_version(self.schema_version))
        object.__setattr__(self, "source_root", Path(self.source_root))
        object.__setattr__(self, "interval", interval)
        object.__setattr__(self, "band", _enum_value(self.band, MarketStateBand, field_name="band"))
        object.__setattr__(self, "core_basket_assets", core)
        object.__setattr__(self, "broad_universe_assets", tuple(dict.fromkeys((*core, *broad))))
        object.__setattr__(self, "split", split)
        object.__setattr__(self, "source_kind", source_kind)
        object.__setattr__(self, "min_timestamp_count", min_timestamp_count)
        object.__setattr__(self, "min_core_asset_count", min_core_assets)
        object.__setattr__(self, "min_broad_asset_count", min_broad_assets)
        object.__setattr__(self, "min_core_asset_timestamp_coverage", _coverage(self.min_core_asset_timestamp_coverage, field_name="min_core_asset_timestamp_coverage"))
        object.__setattr__(self, "min_broad_asset_timestamp_coverage", _coverage(self.min_broad_asset_timestamp_coverage, field_name="min_broad_asset_timestamp_coverage"))
        object.__setattr__(self, "min_broad_timestamp_coverage", _coverage(self.min_broad_timestamp_coverage, field_name="min_broad_timestamp_coverage"))
        object.__setattr__(self, "clamp_policy", clamp)
        object.__setattr__(self, "metadata", to_jsonable(dict(self.metadata)))

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "artifact_kind": "market_state_dataset_build_request",
            "source_root": str(self.source_root),
            "interval": int(self.interval),
            "band": self.band,
            "core_basket_assets": list(self.core_basket_assets),
            "broad_universe_assets": list(self.broad_universe_assets),
            "start_ts": self.start_ts,
            "end_ts": self.end_ts,
            "split": self.split.as_dict(),
            "source_kind": self.source_kind,
            "min_timestamp_count": int(self.min_timestamp_count),
            "min_core_asset_count": int(self.min_core_asset_count),
            "min_broad_asset_count": int(self.min_broad_asset_count),
            "min_core_asset_timestamp_coverage": float(self.min_core_asset_timestamp_coverage),
            "min_broad_asset_timestamp_coverage": float(self.min_broad_asset_timestamp_coverage),
            "min_broad_timestamp_coverage": float(self.min_broad_timestamp_coverage),
            "clamp_policy": self.clamp_policy.as_dict() if self.clamp_policy is not None else None,
            "metadata": to_jsonable(dict(self.metadata)),
        }


@dataclass(frozen=True)
class MarketStateSourcePartitionLineage:
    interval: int
    asset: str
    source_kind: str
    path: str
    root: str
    row_count: int
    min_ts: int | None
    max_ts: int | None
    columns: Sequence[str]
    year: int | None = None
    month: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "interval": int(self.interval),
            "asset": self.asset,
            "source_kind": self.source_kind,
            "path": self.path,
            "root": self.root,
            "row_count": int(self.row_count),
            "min_ts": self.min_ts,
            "max_ts": self.max_ts,
            "columns": list(self.columns),
            "year": self.year,
            "month": self.month,
        }


@dataclass(frozen=True)
class MarketStateDatasetSplit:
    name: str
    timestamps: Sequence[int] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", str(self.name).strip())
        object.__setattr__(self, "timestamps", tuple(int(ts) for ts in self.timestamps))

    @property
    def timestamp_count(self) -> int:
        return len(self.timestamps)

    def as_dict(self) -> dict[str, Any]:
        timestamps = tuple(self.timestamps)
        return {
            "name": self.name,
            "timestamp_count": len(timestamps),
            "timestamp_start": timestamps[0] if timestamps else None,
            "timestamp_end": timestamps[-1] if timestamps else None,
        }


@dataclass
class MarketStateDatasetBuildResult:
    status: str
    reason_codes: Sequence[str]
    interval: int
    band: str
    timestamp_count: int
    core_asset_count: int
    broad_asset_count: int
    split_train: MarketStateDatasetSplit
    split_validation: MarketStateDatasetSplit
    split_holdout: MarketStateDatasetSplit
    per_asset_coverage_diagnostics: Mapping[str, Any]
    missingness_summary: Mapping[str, Any]
    source_partition_lineage: Sequence[MarketStateSourcePartitionLineage]
    panel: pd.DataFrame = field(default_factory=pd.DataFrame, repr=False)
    long_panel: pd.DataFrame = field(default_factory=pd.DataFrame, repr=False)
    selected_core_assets: Sequence[str] = ()
    selected_broad_assets: Sequence[str] = ()
    excluded_assets: Mapping[str, Sequence[str]] = field(default_factory=dict)
    base_series_columns: Sequence[str] = ()
    known_at_metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int | MarketStateSchemaVersion = MARKET_STATE_SCHEMA_VERSION
    metadata: Mapping[str, Any] = field(default_factory=dict)
    message: str | None = None

    def __post_init__(self) -> None:
        self.schema_version = _schema_version(self.schema_version)
        self.reason_codes = tuple(dict.fromkeys(str(reason) for reason in self.reason_codes))
        self.selected_core_assets = tuple(str(asset) for asset in self.selected_core_assets)
        self.selected_broad_assets = tuple(str(asset) for asset in self.selected_broad_assets)
        self.base_series_columns = tuple(str(column) for column in self.base_series_columns)

    @property
    def usable(self) -> bool:
        return self.status == MARKET_STATE_DATASET_STATUS_READY

    @property
    def train_timestamps(self) -> tuple[int, ...]:
        return tuple(self.split_train.timestamps)

    @property
    def validation_timestamps(self) -> tuple[int, ...]:
        return tuple(self.split_validation.timestamps)

    @property
    def holdout_timestamps(self) -> tuple[int, ...]:
        return tuple(self.split_holdout.timestamps)

    @property
    def split_boundaries(self) -> dict[str, dict[str, Any]]:
        return {
            "train": self.split_train.as_dict(),
            "validation": self.split_validation.as_dict(),
            "holdout": self.split_holdout.as_dict(),
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "artifact_kind": "market_state_dataset_build_result",
            "status": self.status,
            "reason_codes": list(self.reason_codes),
            "interval": int(self.interval),
            "band": self.band,
            "timestamp_count": int(self.timestamp_count),
            "core_asset_count": int(self.core_asset_count),
            "broad_asset_count": int(self.broad_asset_count),
            "selected_core_assets": list(self.selected_core_assets),
            "selected_broad_assets": list(self.selected_broad_assets),
            "excluded_assets": {asset: list(reasons) for asset, reasons in sorted(self.excluded_assets.items())},
            "base_series_columns": list(self.base_series_columns),
            "split_boundaries": self.split_boundaries,
            "per_asset_coverage_diagnostics": to_jsonable(dict(self.per_asset_coverage_diagnostics)),
            "missingness_summary": to_jsonable(dict(self.missingness_summary)),
            "source_partition_lineage": [lineage.as_dict() for lineage in self.source_partition_lineage],
            "known_at_metadata": to_jsonable(dict(self.known_at_metadata)),
            "panel_shape": [int(self.panel.shape[0]), int(self.panel.shape[1])],
            "long_panel_shape": [int(self.long_panel.shape[0]), int(self.long_panel.shape[1])],
            "metadata": to_jsonable(dict(self.metadata)),
            "message": self.message,
            "production_feature_materialization": False,
            "clustering_enabled": False,
        }

    def to_json(self, **kwargs: Any) -> str:
        return dumps_json(self.as_dict(), **kwargs)

    @classmethod
    def from_json(cls, text: str) -> dict[str, Any]:
        return require_json_object(loads_json(text), context="MarketStateDatasetBuildResult JSON")


def empty_market_state_dataset_split(name: str) -> MarketStateDatasetSplit:
    return MarketStateDatasetSplit(name=name, timestamps=())


def blocked_market_state_dataset_result(
    request: MarketStateDatasetBuildRequest,
    reason_codes: Sequence[str],
    *,
    message: str | None = None,
    source_partition_lineage: Sequence[MarketStateSourcePartitionLineage] = (),
    per_asset_coverage_diagnostics: Mapping[str, Any] | None = None,
    missingness_summary: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> MarketStateDatasetBuildResult:
    return MarketStateDatasetBuildResult(
        status=MARKET_STATE_DATASET_STATUS_BLOCKED,
        reason_codes=tuple(reason_codes),
        interval=request.interval,
        band=request.band,
        timestamp_count=0,
        core_asset_count=0,
        broad_asset_count=0,
        split_train=empty_market_state_dataset_split("train"),
        split_validation=empty_market_state_dataset_split("validation"),
        split_holdout=empty_market_state_dataset_split("holdout"),
        per_asset_coverage_diagnostics=per_asset_coverage_diagnostics or {},
        missingness_summary=missingness_summary or {},
        source_partition_lineage=tuple(source_partition_lineage),
        schema_version=request.schema_version,
        metadata=metadata or {"request": request.as_dict()},
        message=message,
    )


def _coverage(value: object, *, field_name: str) -> float:
    try:
        out = float(value)
    except Exception as exc:
        raise ValueError(f"Market-state dataset {field_name} must be numeric") from exc
    if out < 0.0 or out > 1.0:
        raise ValueError(f"Market-state dataset {field_name} must be within [0, 1]")
    return out


__all__ = [
    "MARKET_STATE_DATASET_REASON_BUILDER_EXCEPTION",
    "MARKET_STATE_DATASET_REASON_EMPTY_WINDOW",
    "MARKET_STATE_DATASET_REASON_INSUFFICIENT_BROAD_UNIVERSE",
    "MARKET_STATE_DATASET_REASON_LEAKAGE_RISK_COLUMNS",
    "MARKET_STATE_DATASET_REASON_MALFORMED_DATA",
    "MARKET_STATE_DATASET_REASON_MISSING_SOURCE_ROOT",
    "MARKET_STATE_DATASET_REASON_TOO_FEW_CORE_ASSETS",
    "MARKET_STATE_DATASET_REASON_TOO_FEW_TIMESTAMPS",
    "MARKET_STATE_DATASET_SOURCE_AUTO",
    "MARKET_STATE_DATASET_SOURCE_KINDS",
    "MARKET_STATE_DATASET_SOURCE_OHLCVT",
    "MARKET_STATE_DATASET_SOURCE_SCALAR_FEATURES",
    "MARKET_STATE_DATASET_STATUS_BLOCKED",
    "MARKET_STATE_DATASET_STATUS_READY",
    "MarketStateDatasetBuildRequest",
    "MarketStateDatasetBuildResult",
    "MarketStateDatasetSplit",
    "MarketStateSourcePartitionLineage",
    "MarketStateSplitPolicy",
    "blocked_market_state_dataset_result",
    "empty_market_state_dataset_split",
]
