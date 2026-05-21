from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import pandas as pd

from src.regimes.core.feature_preprocessing import FeatureDropRecord, filter_regime_feature_frame
from src.regimes.core.serialization import dumps_json, to_jsonable

from .data_contracts import DATASET_BUILD_STATUS_READY, DatasetBuildResult
from .feature_pools import (
    AssetStateFeaturePoolRegistry,
    AssetStateFeaturePoolSpec,
    default_asset_state_feature_pool_registry,
)
from .feature_column_catalog import (
    FEATURE_COLUMN_STATUS_AVAILABLE,
    FEATURE_COLUMN_STATUS_PENDING_SCALAR_FEATURE,
    AssetStateFeatureColumnCatalog,
    default_asset_state_feature_column_catalog,
)


FEATURE_POOL_RESOLUTION_STATUS_USABLE = "usable"
FEATURE_POOL_RESOLUTION_STATUS_BLOCKED = "blocked"
FEATURE_POOL_SCHEMA_RESOLUTION_ARTIFACT_KIND = "asset_state_feature_pool_schema_resolution"


@dataclass(frozen=True)
class FeaturePoolResolutionResult:
    feature_pool_id: str
    axis: str
    band: str
    status: str
    retained_columns: tuple[str, ...]
    missing_required_columns: tuple[str, ...]
    available_required_columns: Mapping[str, tuple[str, ...]]
    available_optional_columns: Mapping[str, tuple[str, ...]]
    dropped_columns: tuple[str, ...]
    dropped_column_details: tuple[Mapping[str, Any], ...]
    final_feature_count: int
    usable: bool
    train_only_filter_enforced: bool = True
    reason_codes: tuple[str, ...] = ()
    filter_metadata: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "feature_pool_id": self.feature_pool_id,
            "axis": self.axis,
            "band": self.band,
            "status": self.status,
            "usable": self.usable,
            "reason_codes": list(self.reason_codes),
            "retained_columns": list(self.retained_columns),
            "missing_required_columns": list(self.missing_required_columns),
            "available_required_columns": {
                key: list(value) for key, value in self.available_required_columns.items()
            },
            "available_optional_columns": {
                key: list(value) for key, value in self.available_optional_columns.items()
            },
            "dropped_columns": list(self.dropped_columns),
            "dropped_column_details": [to_jsonable(dict(item)) for item in self.dropped_column_details],
            "final_feature_count": self.final_feature_count,
            "train_only_filter_enforced": self.train_only_filter_enforced,
            "filter_metadata": to_jsonable(dict(self.filter_metadata)),
            "metadata": to_jsonable(dict(self.metadata)),
        }

    def to_json(self, **kwargs: Any) -> str:
        return dumps_json(self.as_dict(), **kwargs)


@dataclass(frozen=True)
class FeaturePoolSchemaResolutionResult:
    feature_pool_id: str
    axis: str
    pool_style: str
    status: str
    usable: bool
    required_column_statuses: Mapping[str, str]
    optional_column_statuses: Mapping[str, str]
    pending_supported_column_statuses: Mapping[str, str] = field(default_factory=dict)
    sourced_from_ohlcvt_column_statuses: Mapping[str, str] = field(default_factory=dict)
    validation_target_only_column_statuses: Mapping[str, str] = field(default_factory=dict)
    unsupported_for_now_column_statuses: Mapping[str, str] = field(default_factory=dict)
    available_required_columns: Sequence[str] = ()
    available_optional_columns: Sequence[str] = ()
    missing_required_columns: Sequence[str] = ()
    missing_optional_columns: Sequence[str] = ()
    pending_scalar_feature_columns: Sequence[str] = ()
    leakage_risk_input_columns: Sequence[str] = ()
    reason_codes: Sequence[str] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "usable", bool(self.usable))
        object.__setattr__(self, "available_required_columns", _string_tuple(self.available_required_columns))
        object.__setattr__(self, "available_optional_columns", _string_tuple(self.available_optional_columns))
        object.__setattr__(self, "missing_required_columns", _string_tuple(self.missing_required_columns))
        object.__setattr__(self, "missing_optional_columns", _string_tuple(self.missing_optional_columns))
        object.__setattr__(self, "pending_scalar_feature_columns", _string_tuple(self.pending_scalar_feature_columns))
        object.__setattr__(self, "leakage_risk_input_columns", _string_tuple(self.leakage_risk_input_columns))
        object.__setattr__(self, "reason_codes", _string_tuple(self.reason_codes))

    def as_dict(self) -> dict[str, Any]:
        return {
            "feature_pool_id": self.feature_pool_id,
            "axis": self.axis,
            "pool_style": self.pool_style,
            "status": self.status,
            "usable": bool(self.usable),
            "required_column_statuses": dict(self.required_column_statuses),
            "optional_column_statuses": dict(self.optional_column_statuses),
            "pending_supported_column_statuses": dict(self.pending_supported_column_statuses),
            "sourced_from_ohlcvt_column_statuses": dict(self.sourced_from_ohlcvt_column_statuses),
            "validation_target_only_column_statuses": dict(self.validation_target_only_column_statuses),
            "unsupported_for_now_column_statuses": dict(self.unsupported_for_now_column_statuses),
            "available_required_columns": list(self.available_required_columns),
            "available_optional_columns": list(self.available_optional_columns),
            "missing_required_columns": list(self.missing_required_columns),
            "missing_optional_columns": list(self.missing_optional_columns),
            "pending_scalar_feature_columns": list(self.pending_scalar_feature_columns),
            "leakage_risk_input_columns": list(self.leakage_risk_input_columns),
            "reason_codes": list(self.reason_codes),
        }


def resolve_feature_pool_against_dataset(
    dataset: DatasetBuildResult,
    feature_pool: AssetStateFeaturePoolSpec | str,
    *,
    registry: AssetStateFeaturePoolRegistry | None = None,
) -> FeaturePoolResolutionResult:
    pool = _resolve_pool(feature_pool, registry=registry)
    if dataset.status != DATASET_BUILD_STATUS_READY:
        return _blocked_result(
            dataset,
            pool,
            reason_codes=("dataset_unusable",),
            metadata={"dataset_status": dataset.status, "dataset_reason_codes": list(dataset.reason_codes)},
        )
    if not pool.supports(axis=dataset.axis, band=dataset.band):
        return _blocked_result(
            dataset,
            pool,
            reason_codes=("axis_band_incompatible",),
            metadata={"dataset_axis": dataset.axis, "dataset_band": dataset.band},
        )

    train_frame = dataset.train.frame
    if train_frame.empty:
        return _blocked_result(
            dataset,
            pool,
            reason_codes=("empty_train_frame",),
            metadata={"dataset_row_counts": dict(dataset.row_counts)},
        )

    columns_by_base = _columns_by_source_base(train_frame.columns)
    available_required = {
        base: tuple(columns_by_base.get(base, ()))
        for base in pool.required_source_columns
    }
    available_optional = {
        base: tuple(columns_by_base.get(base, ()))
        for base in pool.optional_source_columns
        if columns_by_base.get(base)
    }
    missing_required = tuple(
        base for base, columns in available_required.items() if not columns
    )

    candidate_columns = _candidate_columns_in_dataset_order(
        train_frame,
        required=pool.required_source_columns,
        optional=pool.optional_source_columns,
    )
    if not candidate_columns:
        return _blocked_result(
            dataset,
            pool,
            reason_codes=("missing_required_columns", "no_candidate_columns"),
            missing_required_columns=missing_required or tuple(pool.required_source_columns),
            available_required_columns=available_required,
            available_optional_columns=available_optional,
            metadata={"source": "train_split_only"},
        )

    filter_result = filter_regime_feature_frame(
        train_frame,
        candidate_columns,
        config=pool.filter_config(),
    )
    retained = tuple(filter_result.retained_columns)
    dropped = tuple(filter_result.dropped_columns)
    dropped_details = tuple(_drop_record_as_dict(record) for record in filter_result.dropped_features)

    reason_codes: list[str] = []
    if missing_required:
        reason_codes.append("missing_required_columns")
    if len(retained) < pool.min_retained_columns:
        reason_codes.append("insufficient_retained_columns")

    usable = not reason_codes
    return FeaturePoolResolutionResult(
        feature_pool_id=pool.feature_pool_id,
        axis=pool.axis,
        band=dataset.band,
        status=FEATURE_POOL_RESOLUTION_STATUS_USABLE if usable else FEATURE_POOL_RESOLUTION_STATUS_BLOCKED,
        retained_columns=retained,
        missing_required_columns=missing_required,
        available_required_columns=available_required,
        available_optional_columns=available_optional,
        dropped_columns=dropped,
        dropped_column_details=dropped_details,
        final_feature_count=len(retained),
        usable=usable,
        train_only_filter_enforced=True,
        reason_codes=tuple(reason_codes),
        filter_metadata=filter_result.to_metadata(),
        metadata={
            "fit_scope": "train_only",
            "feature_pool": pool.as_dict(),
            "candidate_columns": list(candidate_columns),
            "dataset_feature_columns": list(dataset.feature_columns),
            "dataset_output_feature_names": list(dataset.output_feature_names),
        },
    )


def resolve_axis_feature_pools_against_dataset(
    dataset: DatasetBuildResult,
    *,
    registry: AssetStateFeaturePoolRegistry | None = None,
) -> tuple[FeaturePoolResolutionResult, ...]:
    active_registry = registry or default_asset_state_feature_pool_registry()
    pools = active_registry.for_axis(dataset.axis, band=dataset.band)
    return tuple(resolve_feature_pool_against_dataset(dataset, pool, registry=active_registry) for pool in pools)


def resolve_feature_pool_against_schema(
    source_feature_columns: Sequence[str],
    feature_pool: AssetStateFeaturePoolSpec | str,
    *,
    registry: AssetStateFeaturePoolRegistry | None = None,
    catalog: AssetStateFeatureColumnCatalog | None = None,
) -> FeaturePoolSchemaResolutionResult:
    pool = _resolve_pool(feature_pool, registry=registry)
    active_catalog = catalog or default_asset_state_feature_column_catalog(source_feature_columns, source="feature_pool_schema_resolution")
    required_statuses = {base: active_catalog.classify(base).status for base in pool.required_source_columns}
    optional_statuses = {base: active_catalog.classify(base).status for base in pool.optional_source_columns}
    pending_statuses = {base: active_catalog.classify(base).status for base in pool.pending_supported_columns}
    sourced_statuses = {base: active_catalog.classify(base).status for base in pool.sourced_from_ohlcvt_columns}
    validation_statuses = {base: active_catalog.classify(base).status for base in pool.validation_target_only_columns}
    unsupported_statuses = {base: active_catalog.classify(base).status for base in pool.unsupported_for_now_columns}

    missing_required = tuple(base for base, status in required_statuses.items() if status != FEATURE_COLUMN_STATUS_AVAILABLE)
    missing_optional = tuple(base for base, status in optional_statuses.items() if status != FEATURE_COLUMN_STATUS_AVAILABLE)
    pending_columns = tuple(
        base for base, status in pending_statuses.items() if status == FEATURE_COLUMN_STATUS_PENDING_SCALAR_FEATURE
    )
    leakage_risk = tuple(
        base
        for base in (*pool.required_source_columns, *pool.optional_source_columns, *pool.pending_supported_columns)
        if active_catalog.classify(base).leakage_risk
    )
    reason_codes: list[str] = []
    if missing_required:
        reason_codes.append("missing_required_columns")
    if leakage_risk:
        reason_codes.append("leakage_risk_input_columns")
    usable = not reason_codes
    return FeaturePoolSchemaResolutionResult(
        feature_pool_id=pool.feature_pool_id,
        axis=pool.axis,
        pool_style=pool.pool_style,
        status=FEATURE_POOL_RESOLUTION_STATUS_USABLE if usable else FEATURE_POOL_RESOLUTION_STATUS_BLOCKED,
        usable=usable,
        required_column_statuses=required_statuses,
        optional_column_statuses=optional_statuses,
        pending_supported_column_statuses=pending_statuses,
        sourced_from_ohlcvt_column_statuses=sourced_statuses,
        validation_target_only_column_statuses=validation_statuses,
        unsupported_for_now_column_statuses=unsupported_statuses,
        available_required_columns=tuple(base for base, status in required_statuses.items() if status == FEATURE_COLUMN_STATUS_AVAILABLE),
        available_optional_columns=tuple(base for base, status in optional_statuses.items() if status == FEATURE_COLUMN_STATUS_AVAILABLE),
        missing_required_columns=missing_required,
        missing_optional_columns=missing_optional,
        pending_scalar_feature_columns=pending_columns,
        leakage_risk_input_columns=leakage_risk,
        reason_codes=tuple(reason_codes),
    )


def resolve_feature_pool_registry_against_schema(
    source_feature_columns: Sequence[str],
    *,
    registry: AssetStateFeaturePoolRegistry | None = None,
    band: str | None = None,
    catalog: AssetStateFeatureColumnCatalog | None = None,
) -> dict[str, Any]:
    active_registry = registry or default_asset_state_feature_pool_registry()
    active_catalog = catalog or default_asset_state_feature_column_catalog(source_feature_columns, source="feature_pool_registry_schema_resolution")
    results = tuple(
        resolve_feature_pool_against_schema(
            source_feature_columns,
            pool,
            registry=active_registry,
            catalog=active_catalog,
        )
        for pool in active_registry.pools.values()
        if band is None or str(band).lower() in pool.compatible_bands
    )
    axis_summaries: dict[str, dict[str, Any]] = {}
    for result in results:
        summary = axis_summaries.setdefault(
            result.axis,
            {
                "axis": result.axis,
                "pool_ids": [],
                "usable_pool_ids": [],
                "blocked_pool_ids": [],
                "missing_required_columns": set(),
                "missing_optional_columns": set(),
                "pending_scalar_feature_columns": set(),
            },
        )
        summary["pool_ids"].append(result.feature_pool_id)
        if result.usable:
            summary["usable_pool_ids"].append(result.feature_pool_id)
        else:
            summary["blocked_pool_ids"].append(result.feature_pool_id)
        summary["missing_required_columns"].update(result.missing_required_columns)
        summary["missing_optional_columns"].update(result.missing_optional_columns)
        summary["pending_scalar_feature_columns"].update(result.pending_scalar_feature_columns)

    normalized_axis = {
        axis: {
            "axis": axis,
            "status": "usable_alternatives" if summary["usable_pool_ids"] else "blocked",
            "pool_ids": list(summary["pool_ids"]),
            "usable_pool_ids": list(summary["usable_pool_ids"]),
            "blocked_pool_ids": list(summary["blocked_pool_ids"]),
            "usable_pool_count": int(len(summary["usable_pool_ids"])),
            "blocked_pool_count": int(len(summary["blocked_pool_ids"])),
            "missing_required_columns": sorted(summary["missing_required_columns"]),
            "missing_optional_columns": sorted(summary["missing_optional_columns"]),
            "pending_scalar_feature_columns": sorted(summary["pending_scalar_feature_columns"]),
            "axis_not_broken_if_any_usable_pool": bool(summary["usable_pool_ids"]),
        }
        for axis, summary in sorted(axis_summaries.items())
    }
    return {
        "artifact_kind": FEATURE_POOL_SCHEMA_RESOLUTION_ARTIFACT_KIND,
        "pool_count": int(len(results)),
        "band": band,
        "source_feature_count": int(len(set(_source_base(column) for column in source_feature_columns))),
        "catalog_status_counts": active_catalog.status_counts,
        "all_axes_have_usable_pool": all(summary["usable_pool_count"] > 0 for summary in normalized_axis.values()),
        "axis_summaries": normalized_axis,
        "pools": {result.feature_pool_id: result.as_dict() for result in results},
        "production_profile_selection_enabled": False,
        "benchmarking_performed": False,
    }


def _resolve_pool(
    feature_pool: AssetStateFeaturePoolSpec | str,
    *,
    registry: AssetStateFeaturePoolRegistry | None,
) -> AssetStateFeaturePoolSpec:
    if isinstance(feature_pool, AssetStateFeaturePoolSpec):
        return feature_pool
    active_registry = registry or default_asset_state_feature_pool_registry()
    return active_registry.get(str(feature_pool))


def _blocked_result(
    dataset: DatasetBuildResult,
    pool: AssetStateFeaturePoolSpec,
    *,
    reason_codes: Sequence[str],
    missing_required_columns: Sequence[str] = (),
    available_required_columns: Mapping[str, tuple[str, ...]] | None = None,
    available_optional_columns: Mapping[str, tuple[str, ...]] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> FeaturePoolResolutionResult:
    return FeaturePoolResolutionResult(
        feature_pool_id=pool.feature_pool_id,
        axis=pool.axis,
        band=dataset.band,
        status=FEATURE_POOL_RESOLUTION_STATUS_BLOCKED,
        retained_columns=(),
        missing_required_columns=tuple(missing_required_columns),
        available_required_columns=available_required_columns or {},
        available_optional_columns=available_optional_columns or {},
        dropped_columns=(),
        dropped_column_details=(),
        final_feature_count=0,
        usable=False,
        train_only_filter_enforced=True,
        reason_codes=tuple(dict.fromkeys(reason_codes)),
        filter_metadata={},
        metadata=metadata or {},
    )


def _columns_by_source_base(columns: Sequence[object]) -> dict[str, tuple[str, ...]]:
    grouped: dict[str, list[str]] = {}
    for column in columns:
        text = str(column)
        base = _source_base(text)
        grouped.setdefault(base, []).append(text)
    return {base: tuple(values) for base, values in grouped.items()}


def _source_base(column: str) -> str:
    text = str(column)
    if text.startswith("i"):
        prefix, sep, suffix = text.partition("_")
        if sep and prefix[1:].isdigit() and suffix:
            return suffix
    return text


def _candidate_columns_in_dataset_order(
    frame: pd.DataFrame,
    *,
    required: Sequence[str],
    optional: Sequence[str],
) -> tuple[str, ...]:
    wanted = set(str(base) for base in (*required, *optional))
    candidates: list[str] = []
    for column in frame.columns:
        text = str(column)
        if text in {"asset", "ts"}:
            continue
        if _source_base(text) in wanted:
            candidates.append(text)
    return tuple(candidates)


def _drop_record_as_dict(record: FeatureDropRecord) -> dict[str, Any]:
    return {
        "column": record.column,
        "reason": record.reason,
        "metric_name": record.metric_name,
        "metric_value": record.metric_value,
        "reference_column": record.reference_column,
    }


def _string_tuple(values: Sequence[Any]) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        return (str(values),)
    return tuple(str(value) for value in values)


__all__ = [
    "FEATURE_POOL_RESOLUTION_STATUS_BLOCKED",
    "FEATURE_POOL_RESOLUTION_STATUS_USABLE",
    "FEATURE_POOL_SCHEMA_RESOLUTION_ARTIFACT_KIND",
    "FeaturePoolResolutionResult",
    "FeaturePoolSchemaResolutionResult",
    "resolve_axis_feature_pools_against_dataset",
    "resolve_feature_pool_against_dataset",
    "resolve_feature_pool_against_schema",
    "resolve_feature_pool_registry_against_schema",
]
