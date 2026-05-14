from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.regimes.contracts import REGIME_BANDS
from src.regimes.core import (
    PATHWAY_DIAGNOSTICS_ROOT_EXPLICIT,
    PATHWAY_DIAGNOSTICS_ROOT_POLICY_VERSION,
    PATHWAY_DIAGNOSTICS_ROOT_REPORT,
    PATHWAY_DIAGNOSTICS_ROOT_SANDBOX_TEMP,
    PATHWAY_DIAGNOSTICS_ROOT_UNSAFE_PRODUCTION,
    PathwayDiagnosticsRootPolicy,
    SourceArtifactLineage,
    classify_pathway_diagnostics_root,
    require_pathway_diagnostics_root,
    safe_path_part,
    write_json,
)


MARKET_AGGREGATE_SCHEMA_VERSION = 1
MARKET_AGGREGATE_SCHEMA_ARTIFACT_KIND = "market_aggregate_feature_schema_contract"
MARKET_AGGREGATE_SCHEMA_STATUS = "schema_only"
MARKET_AGGREGATE_SCHEMA_MODE = "metadata_only"
MARKET_STATE_PATHWAY = "market_state"
MARKET_FEATURE_FAMILY_CONTRACT_ARTIFACT_KIND = "market_state_feature_family_contract"
MARKET_METADATA_PRODUCTION_CLASSIFICATIONS: tuple[str, ...] = ("diagnostics_only", "scaffold", "schema_only")
MARKET_COVARIANCE_CORRELATION_ESTIMATION_METHODS: tuple[str, ...] = (
    "not_applicable",
    "return_summary",
    "realized_volatility_summary",
    "breadth_participation",
    "cross_sectional_dispersion",
    "empirical_correlation",
    "empirical_covariance",
    "pca_eigendecomposition",
    "shrinkage_covariance_metadata",
)
MARKET_SHRINKAGE_METHODS: tuple[str, ...] = (
    "not_applicable",
    "empirical",
    "ledoit_wolf",
    "oas",
    "method_catalog",
    "future_sparse_precision_placeholder",
)
MARKET_FEATURE_IMPLEMENTATION_STATUSES: tuple[str, ...] = ("declared", "placeholder", "metadata_only")
DEFAULT_SOURCE_LABEL = "scalar_features"
DEFAULT_MEMBERSHIP_BASIS = "explicit_member_assets_only"
DEFAULT_UNIVERSE_MEMBERSHIP_SOURCE = "config.member_assets"
DEFAULT_AGGREGATION_GRAIN = "universe_band_timestamp"
DEFAULT_MINIMUM_COVERAGE_THRESHOLD = 0.8
DEFAULT_LOOKBACK_WINDOW = 60
DEFAULT_STALE_MAX_INTERVALS = 1
DEFAULT_MARKET_COVARIANCE_FEATURE_FAMILIES: tuple[str, ...] = (
    "market_return_summary",
    "realized_volatility_summary",
    "breadth_participation",
    "cross_sectional_dispersion",
    "median_pairwise_correlation",
    "first_principal_component_concentration",
    "covariance_summary",
    "correlation_summary",
    "shrinkage_covariance_metadata",
)


def _jsonable(value: Any) -> Any:
    if hasattr(value, "as_dict"):
        return value.as_dict()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(val) for key, val in value.items()}
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, set):
        return sorted(_jsonable(item) for item in value)
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            return None
        return value
    return str(value)


def _normalize_token(value: object, *, field_name: str) -> str:
    token = str(value).strip().lower()
    if not token:
        raise ValueError(f"Market aggregate schema {field_name} must be non-empty")
    return token


def _require_member(value: object, allowed: Sequence[str], *, field_name: str) -> str:
    token = _normalize_token(value, field_name=field_name)
    if token not in allowed:
        valid = ", ".join(str(item) for item in allowed)
        raise ValueError(f"Unsupported market aggregate schema {field_name} {token!r}; expected one of: {valid}")
    return token


def _unique_texts(values: Sequence[object], *, field_name: str, require_nonempty: bool = True) -> tuple[str, ...]:
    out = tuple(dict.fromkeys(str(value).strip() for value in values if str(value).strip()))
    if require_nonempty and not out:
        raise ValueError(f"Market aggregate schema {field_name} must include at least one value")
    return out


def _default_market_metadata_artifact_boundary() -> dict[str, Any]:
    return {
        "write_mode": "market_state_feature_schema_metadata_only",
        "metadata_only": True,
        "schema_only": True,
        "production_writes_enabled": False,
        "parquet_writes_enabled": False,
        "definition_writes_enabled": False,
        "state_writes_enabled": False,
        "production_outputs_written": False,
        "aggregation_frame_materialized": False,
        "aggregation_values_computed": False,
        "source_reader_enabled": False,
        "downstream_forecast_readers_enabled": False,
    }


def _validate_market_metadata_artifact_boundary(boundary: Mapping[str, Any], *, context: str) -> dict[str, Any]:
    checked = dict(boundary)
    if checked.get("metadata_only") is not True:
        raise ValueError(f"{context} must declare metadata_only=true")
    if checked.get("schema_only") is not True:
        raise ValueError(f"{context} must declare schema_only=true")
    for field_name in (
        "production_writes_enabled",
        "parquet_writes_enabled",
        "definition_writes_enabled",
        "state_writes_enabled",
        "production_outputs_written",
        "aggregation_frame_materialized",
        "aggregation_values_computed",
        "source_reader_enabled",
        "downstream_forecast_readers_enabled",
    ):
        if checked.get(field_name) is not False:
            raise ValueError(f"{context} cannot enable {field_name}")
    return checked


@dataclass(frozen=True)
class MarketAggregateFeatureFamilyContract:
    feature_family_name: str
    source_label: str
    required_source_columns: tuple[str, ...]
    derived_feature_columns: tuple[str, ...]
    universe_membership_source: str
    included_asset_count: int
    excluded_asset_count: int
    coverage_by_timestamp_window: Mapping[str, Any]
    minimum_coverage_threshold: float
    stale_null_policy: Mapping[str, Any]
    covariance_correlation_estimation_method: str
    shrinkage_method: str
    lookback_window: int
    asset_universe_membership_basis: str
    aggregation_grain: str
    coverage_policy: Mapping[str, Any]
    null_staleness_policy: Mapping[str, Any]
    source_lineage: tuple[SourceArtifactLineage, ...]
    lineage_fields: tuple[str, ...]
    diagnostics_root_constraints: Mapping[str, Any]
    shrinkage_method_catalog: tuple[str, ...] = ()
    placeholder_methods: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()
    production_classification: str = "diagnostics_only"
    implementation_status: str = "declared"
    artifact_boundary: Mapping[str, Any] = field(default_factory=_default_market_metadata_artifact_boundary)
    schema_version: int = MARKET_AGGREGATE_SCHEMA_VERSION
    mode: str = MARKET_AGGREGATE_SCHEMA_MODE
    status: str = MARKET_AGGREGATE_SCHEMA_STATUS
    artifact_kind: str = MARKET_FEATURE_FAMILY_CONTRACT_ARTIFACT_KIND

    def __post_init__(self) -> None:
        family = _normalize_token(self.feature_family_name, field_name="feature family name")
        source_columns = _unique_texts(self.required_source_columns, field_name="required source columns")
        derived_columns = _unique_texts(self.derived_feature_columns, field_name="derived feature columns")
        membership_source = str(self.universe_membership_source).strip()
        if not membership_source:
            raise ValueError("Market aggregate schema universe_membership_source must be non-empty")
        included = int(self.included_asset_count)
        excluded = int(self.excluded_asset_count)
        if included < 0 or excluded < 0:
            raise ValueError("Market aggregate schema included/excluded asset counts must be non-negative")
        threshold = float(self.minimum_coverage_threshold)
        if threshold < 0.0 or threshold > 1.0:
            raise ValueError("Market aggregate schema minimum_coverage_threshold must be between 0 and 1")
        method = _require_member(
            self.covariance_correlation_estimation_method,
            MARKET_COVARIANCE_CORRELATION_ESTIMATION_METHODS,
            field_name="covariance/correlation estimation method",
        )
        shrinkage = _require_member(self.shrinkage_method, MARKET_SHRINKAGE_METHODS, field_name="shrinkage method")
        lookback = int(self.lookback_window)
        if lookback <= 0:
            raise ValueError("Market aggregate schema lookback_window must be positive")
        classification = _require_member(
            self.production_classification,
            MARKET_METADATA_PRODUCTION_CLASSIFICATIONS,
            field_name="production classification",
        )
        status = _require_member(
            self.implementation_status,
            MARKET_FEATURE_IMPLEMENTATION_STATUSES,
            field_name="implementation status",
        )
        if str(self.mode) != MARKET_AGGREGATE_SCHEMA_MODE:
            raise ValueError(f"Market aggregate schema mode must be {MARKET_AGGREGATE_SCHEMA_MODE!r}")
        if str(self.status) != MARKET_AGGREGATE_SCHEMA_STATUS:
            raise ValueError(f"Market aggregate schema status must be {MARKET_AGGREGATE_SCHEMA_STATUS!r}")
        if not self.source_lineage:
            raise ValueError("Market aggregate schema source_lineage must be non-empty")
        boundary = _validate_market_metadata_artifact_boundary(
            self.artifact_boundary,
            context=f"Market feature family {family}",
        )
        object.__setattr__(self, "feature_family_name", family)
        object.__setattr__(self, "required_source_columns", source_columns)
        object.__setattr__(self, "derived_feature_columns", derived_columns)
        object.__setattr__(self, "universe_membership_source", membership_source)
        object.__setattr__(self, "included_asset_count", included)
        object.__setattr__(self, "excluded_asset_count", excluded)
        object.__setattr__(self, "minimum_coverage_threshold", threshold)
        object.__setattr__(self, "covariance_correlation_estimation_method", method)
        object.__setattr__(self, "shrinkage_method", shrinkage)
        object.__setattr__(self, "lookback_window", lookback)
        object.__setattr__(self, "production_classification", classification)
        object.__setattr__(self, "implementation_status", status)
        object.__setattr__(self, "source_lineage", tuple(self.source_lineage))
        object.__setattr__(self, "shrinkage_method_catalog", tuple(str(item) for item in self.shrinkage_method_catalog))
        object.__setattr__(self, "placeholder_methods", tuple(str(item) for item in self.placeholder_methods))
        object.__setattr__(self, "notes", tuple(str(item) for item in self.notes))
        object.__setattr__(self, "artifact_boundary", boundary)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "artifact_kind": self.artifact_kind,
            "status": self.status,
            "mode": self.mode,
            "pathway": MARKET_STATE_PATHWAY,
            "feature_family_name": self.feature_family_name,
            "source_label": self.source_label,
            "required_source_columns": list(self.required_source_columns),
            "derived_feature_columns": list(self.derived_feature_columns),
            "universe_membership_source": self.universe_membership_source,
            "included_asset_count": int(self.included_asset_count),
            "excluded_asset_count": int(self.excluded_asset_count),
            "coverage_by_timestamp_window": _jsonable(dict(self.coverage_by_timestamp_window)),
            "minimum_coverage_threshold": float(self.minimum_coverage_threshold),
            "stale_null_policy": _jsonable(dict(self.stale_null_policy)),
            "covariance_correlation_estimation_method": self.covariance_correlation_estimation_method,
            "shrinkage_method": self.shrinkage_method,
            "shrinkage_method_catalog": list(self.shrinkage_method_catalog),
            "placeholder_methods": list(self.placeholder_methods),
            "lookback_window": int(self.lookback_window),
            "asset_universe_membership_basis": self.asset_universe_membership_basis,
            "aggregation_grain": self.aggregation_grain,
            "coverage_policy": _jsonable(dict(self.coverage_policy)),
            "null_staleness_policy": _jsonable(dict(self.null_staleness_policy)),
            "source_lineage": [lineage.as_dict() for lineage in self.source_lineage],
            "lineage_fields": list(self.lineage_fields),
            "diagnostics_root_constraints": _jsonable(dict(self.diagnostics_root_constraints)),
            "production_classification": self.production_classification,
            "implementation_status": self.implementation_status,
            "notes": list(self.notes),
            "computed_value_columns": [],
            "aggregation_values_computed": False,
            "artifact_boundary": dict(self.artifact_boundary),
        }


@dataclass(frozen=True)
class MarketAggregationFrameInputSchemaContract:
    universe: str
    bands: tuple[str, ...]
    source_label: str
    timestamp_column: str
    asset_column: str
    member_assets: tuple[str, ...]
    excluded_assets: tuple[str, ...]
    min_assets: int
    universe_membership_source: str
    minimum_coverage_threshold: float
    coverage_by_timestamp_window: Mapping[str, Any]
    stale_null_policy: Mapping[str, Any]
    asset_universe_membership_basis: str
    aggregation_grain: str
    feature_families: tuple[MarketAggregateFeatureFamilyContract, ...]
    source_lineage: tuple[SourceArtifactLineage, ...]
    diagnostics_root_constraints: Mapping[str, Any]
    production_classification: str = "diagnostics_only"
    schema_version: int = MARKET_AGGREGATE_SCHEMA_VERSION
    mode: str = MARKET_AGGREGATE_SCHEMA_MODE
    status: str = MARKET_AGGREGATE_SCHEMA_STATUS

    @property
    def key_columns(self) -> tuple[str, ...]:
        return (self.timestamp_column, self.asset_column, "universe", "band", "ceiling_interval_min")

    @property
    def required_source_columns(self) -> tuple[str, ...]:
        columns: list[str] = list(self.key_columns)
        for family in self.feature_families:
            for column in family.required_source_columns:
                if column not in columns:
                    columns.append(column)
        return tuple(columns)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "status": self.status,
            "mode": self.mode,
            "pathway": MARKET_STATE_PATHWAY,
            "universe": self.universe,
            "bands": list(self.bands),
            "source_label": self.source_label,
            "timestamp_column": self.timestamp_column,
            "asset_column": self.asset_column,
            "member_assets": list(self.member_assets),
            "excluded_assets": list(self.excluded_assets),
            "member_assets_count": int(len(self.member_assets)),
            "included_asset_count": int(len(self.member_assets)),
            "excluded_asset_count": int(len(self.excluded_assets)),
            "min_assets": int(self.min_assets),
            "universe_membership_source": self.universe_membership_source,
            "minimum_coverage_threshold": float(self.minimum_coverage_threshold),
            "coverage_by_timestamp_window": _jsonable(dict(self.coverage_by_timestamp_window)),
            "stale_null_policy": _jsonable(dict(self.stale_null_policy)),
            "asset_universe_membership_basis": self.asset_universe_membership_basis,
            "aggregation_grain": self.aggregation_grain,
            "key_columns": list(self.key_columns),
            "required_source_columns": list(self.required_source_columns),
            "required_source_columns_by_family": {
                family.feature_family_name: list(family.required_source_columns)
                for family in self.feature_families
            },
            "feature_families": [family.as_dict() for family in self.feature_families],
            "source_lineage": [lineage.as_dict() for lineage in self.source_lineage],
            "diagnostics_root_constraints": _jsonable(dict(self.diagnostics_root_constraints)),
            "production_classification": self.production_classification,
            "aggregation_frame_materialized": False,
            "aggregation_values_computed": False,
            "parquet_output_declared": False,
            "artifact_boundary": _default_market_metadata_artifact_boundary(),
        }


@dataclass(frozen=True)
class MarketAggregateSchemaConfig:
    universe: str
    member_assets: tuple[str, ...] = ()
    excluded_assets: tuple[str, ...] = ()
    bands: tuple[str, ...] = tuple(REGIME_BANDS.keys())
    feature_families: tuple[str, ...] = DEFAULT_MARKET_COVARIANCE_FEATURE_FAMILIES
    source_label: str = DEFAULT_SOURCE_LABEL
    timestamp_column: str = "ts"
    asset_column: str = "asset"
    min_assets: int = 1
    universe_membership_source: str = DEFAULT_UNIVERSE_MEMBERSHIP_SOURCE
    minimum_coverage_threshold: float = DEFAULT_MINIMUM_COVERAGE_THRESHOLD
    lookback_window: int = DEFAULT_LOOKBACK_WINDOW
    stale_max_intervals: int = DEFAULT_STALE_MAX_INTERVALS
    coverage_window_label: str = "schema_window"
    asset_universe_membership_basis: str = DEFAULT_MEMBERSHIP_BASIS
    aggregation_grain: str = DEFAULT_AGGREGATION_GRAIN
    source_feature_root: str | None = None
    production_classification: str = "diagnostics_only"
    schema_version: int = MARKET_AGGREGATE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not str(self.universe).strip():
            raise ValueError("Market aggregate schema universe must be non-empty")
        if int(self.min_assets) < 1:
            raise ValueError("Market aggregate schema min_assets must be positive")
        members = _unique_texts(self.member_assets, field_name="member_assets", require_nonempty=False)
        excluded = _unique_texts(self.excluded_assets, field_name="excluded_assets", require_nonempty=False)
        if self.member_assets and int(self.min_assets) > len(members):
            raise ValueError("Market aggregate schema min_assets cannot exceed member_assets count")
        overlap = set(members).intersection(excluded)
        if overlap:
            raise ValueError("Market aggregate schema member_assets and excluded_assets must not overlap")
        for field_name in ("source_label", "timestamp_column", "asset_column"):
            if not str(getattr(self, field_name)).strip():
                raise ValueError(f"Market aggregate schema {field_name} must be non-empty")
        if not str(self.universe_membership_source).strip():
            raise ValueError("Market aggregate schema universe_membership_source must be non-empty")
        threshold = float(self.minimum_coverage_threshold)
        if threshold < 0.0 or threshold > 1.0:
            raise ValueError("Market aggregate schema minimum_coverage_threshold must be between 0 and 1")
        if int(self.lookback_window) <= 0:
            raise ValueError("Market aggregate schema lookback_window must be positive")
        if int(self.stale_max_intervals) < 0:
            raise ValueError("Market aggregate schema stale_max_intervals must be non-negative")
        _require_member(
            self.production_classification,
            MARKET_METADATA_PRODUCTION_CLASSIFICATIONS,
            field_name="production classification",
        )
        invalid_bands = [str(band) for band in self.bands if str(band) not in REGIME_BANDS]
        if invalid_bands:
            valid = ", ".join(REGIME_BANDS)
            raise ValueError(f"Unsupported Regime bands {invalid_bands}; expected one of: {valid}")
        invalid_families = [name for name in self.feature_families if name not in DEFAULT_FAMILY_SOURCE_COLUMNS]
        if invalid_families:
            valid = ", ".join(DEFAULT_FAMILY_SOURCE_COLUMNS)
            raise ValueError(f"Unsupported market aggregate feature families {invalid_families}; expected one of: {valid}")
        object.__setattr__(self, "member_assets", members)
        object.__setattr__(self, "excluded_assets", excluded)
        object.__setattr__(self, "minimum_coverage_threshold", threshold)
        object.__setattr__(self, "lookback_window", int(self.lookback_window))
        object.__setattr__(self, "stale_max_intervals", int(self.stale_max_intervals))
        object.__setattr__(
            self,
            "production_classification",
            _require_member(
                self.production_classification,
                MARKET_METADATA_PRODUCTION_CLASSIFICATIONS,
                field_name="production classification",
            ),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "mode": MARKET_AGGREGATE_SCHEMA_MODE,
            "universe": self.universe,
            "member_assets": list(self.member_assets),
            "excluded_assets": list(self.excluded_assets),
            "member_assets_count": int(len(self.member_assets)),
            "included_asset_count": int(len(self.member_assets)),
            "excluded_asset_count": int(len(self.excluded_assets)),
            "bands": list(self.bands),
            "feature_families": list(self.feature_families),
            "source_label": self.source_label,
            "timestamp_column": self.timestamp_column,
            "asset_column": self.asset_column,
            "min_assets": int(self.min_assets),
            "universe_membership_source": self.universe_membership_source,
            "minimum_coverage_threshold": float(self.minimum_coverage_threshold),
            "lookback_window": int(self.lookback_window),
            "stale_max_intervals": int(self.stale_max_intervals),
            "coverage_window_label": self.coverage_window_label,
            "asset_universe_membership_basis": self.asset_universe_membership_basis,
            "aggregation_grain": self.aggregation_grain,
            "source_feature_root": self.source_feature_root,
            "production_classification": self.production_classification,
        }


DEFAULT_FAMILY_SOURCE_COLUMNS: Mapping[str, tuple[str, ...]] = {
    "market_return_summary": ("ts", "asset", "log_return", "close"),
    "realized_volatility_summary": ("ts", "asset", "atr_14", "ret_std_20", "cv_20"),
    "breadth_participation": ("ts", "asset", "log_return", "close", "volume"),
    "cross_sectional_dispersion": ("ts", "asset", "log_return", "ret_std_20"),
    "median_pairwise_correlation": ("ts", "asset", "log_return"),
    "first_principal_component_concentration": ("ts", "asset", "log_return"),
    "covariance_summary": ("ts", "asset", "log_return"),
    "correlation_summary": ("ts", "asset", "log_return"),
    "shrinkage_covariance_metadata": ("ts", "asset", "log_return"),
    "breadth": ("ts", "asset", "log_return", "close"),
    "dispersion": ("ts", "asset", "log_return", "ret_std_20"),
    "correlation": ("ts", "asset", "log_return"),
    "market_vol": ("ts", "asset", "atr_14", "ret_std_20", "cv_20"),
    "leadership": ("ts", "asset", "log_return", "volume", "trade_intensity"),
}

DEFAULT_FAMILY_DERIVED_COLUMNS: Mapping[str, tuple[str, ...]] = {
    "market_return_summary": ("market_return_equal_weight", "market_return_member_weighted"),
    "realized_volatility_summary": ("market_realized_vol_median", "market_realized_vol_iqr"),
    "breadth_participation": ("positive_return_share", "advance_decline", "active_volume_share"),
    "cross_sectional_dispersion": ("cross_sectional_return_std", "return_iqr", "cross_sectional_vol_std"),
    "median_pairwise_correlation": ("median_pairwise_correlation",),
    "first_principal_component_concentration": ("first_pc_variance_share", "first_pc_loading_concentration"),
    "covariance_summary": ("covariance_trace", "covariance_condition_number", "covariance_offdiag_mean_abs"),
    "correlation_summary": ("mean_pairwise_correlation", "median_pairwise_correlation", "correlation_breadth"),
    "shrinkage_covariance_metadata": (
        "covariance_estimator_name",
        "covariance_shrinkage_method",
        "covariance_shrinkage_intensity",
    ),
    "breadth": ("advance_decline", "positive_return_share", "above_moving_average_share"),
    "dispersion": ("cross_sectional_return_std", "cross_sectional_vol_std", "return_iqr"),
    "correlation": ("mean_pairwise_correlation", "median_pairwise_correlation", "correlation_breadth"),
    "market_vol": ("index_realized_vol", "median_asset_realized_vol", "vol_of_vol"),
    "leadership": ("top_decile_contribution", "sector_leadership_share", "leader_rotation_rate"),
}

DEFAULT_FAMILY_ESTIMATION_METHOD: Mapping[str, str] = {
    "market_return_summary": "return_summary",
    "realized_volatility_summary": "realized_volatility_summary",
    "breadth_participation": "breadth_participation",
    "cross_sectional_dispersion": "cross_sectional_dispersion",
    "median_pairwise_correlation": "empirical_correlation",
    "first_principal_component_concentration": "pca_eigendecomposition",
    "covariance_summary": "empirical_covariance",
    "correlation_summary": "empirical_correlation",
    "shrinkage_covariance_metadata": "shrinkage_covariance_metadata",
    "breadth": "breadth_participation",
    "dispersion": "cross_sectional_dispersion",
    "correlation": "empirical_correlation",
    "market_vol": "realized_volatility_summary",
    "leadership": "breadth_participation",
}

DEFAULT_FAMILY_SHRINKAGE_METHOD: Mapping[str, str] = {
    "market_return_summary": "not_applicable",
    "realized_volatility_summary": "not_applicable",
    "breadth_participation": "not_applicable",
    "cross_sectional_dispersion": "not_applicable",
    "median_pairwise_correlation": "empirical",
    "first_principal_component_concentration": "empirical",
    "covariance_summary": "empirical",
    "correlation_summary": "empirical",
    "shrinkage_covariance_metadata": "method_catalog",
    "breadth": "not_applicable",
    "dispersion": "not_applicable",
    "correlation": "empirical",
    "market_vol": "not_applicable",
    "leadership": "not_applicable",
}


def _csv(raw: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in str(raw or "").split(",") if part.strip())


def _created_at(value: str | None = None) -> str:
    text = str(value or "").strip()
    if text:
        return text
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def diagnostics_root_constraints() -> dict[str, Any]:
    return {
        "schema_version": PATHWAY_DIAGNOSTICS_ROOT_POLICY_VERSION,
        "allowed_for_schema_json_classifications": [
            PATHWAY_DIAGNOSTICS_ROOT_REPORT,
            PATHWAY_DIAGNOSTICS_ROOT_SANDBOX_TEMP,
            PATHWAY_DIAGNOSTICS_ROOT_EXPLICIT,
        ],
        "rejected_classifications": [PATHWAY_DIAGNOSTICS_ROOT_UNSAFE_PRODUCTION],
        "production_adjacent_roots_rejected": True,
        "diagnostic_writes_limited_to_json": True,
        "parquet_output_allowed": False,
        "definition_output_allowed": False,
        "state_output_allowed": False,
    }


def shrinkage_method_catalog() -> dict[str, Any]:
    return {
        "schema_version": MARKET_AGGREGATE_SCHEMA_VERSION,
        "methods": {
            "empirical": {
                "status": "declared",
                "dependency": "numpy_or_pandas_covariance",
                "production_ready": False,
            },
            "ledoit_wolf": {
                "status": "declared",
                "dependency": "sklearn.covariance.LedoitWolf",
                "production_ready": False,
            },
            "oas": {
                "status": "declared",
                "dependency": "sklearn.covariance.OAS",
                "production_ready": False,
            },
            "future_sparse_precision_placeholder": {
                "status": "placeholder",
                "dependency": "future_sparse_precision_policy",
                "production_ready": False,
            },
        },
        "selection_policy": "schema_only_no_estimator_selection",
    }


def _coverage_policy(config: MarketAggregateSchemaConfig) -> dict[str, Any]:
    return {
        "coverage_basis": "explicit_member_assets",
        "min_assets": int(config.min_assets),
        "included_asset_count": int(len(config.member_assets)),
        "excluded_asset_count": int(len(config.excluded_assets)),
        "universe_membership_source": config.universe_membership_source,
        "member_assets_required_for_values": True,
        "member_assets_supplied": bool(config.member_assets),
        "min_coverage_pct": float(config.minimum_coverage_threshold),
        "insufficient_coverage_behavior": "emit_schema_diagnostic_only_fail_closed_no_values",
        "coverage_columns_declared": [
            "member_asset_count",
            "contributing_asset_count",
            "coverage_pct",
        ],
    }


def _coverage_by_timestamp_window(config: MarketAggregateSchemaConfig) -> dict[str, Any]:
    return {
        "status": "not_computed_metadata_only",
        "coverage_window_label": str(config.coverage_window_label),
        "coverage_by_timestamp": {},
        "coverage_by_window": {},
        "included_asset_count": int(len(config.member_assets)),
        "excluded_asset_count": int(len(config.excluded_assets)),
        "minimum_coverage_threshold": float(config.minimum_coverage_threshold),
        "coverage_values_materialized": False,
    }


def _null_staleness_policy(config: MarketAggregateSchemaConfig) -> dict[str, Any]:
    return {
        "null_policy": "required_source_columns_fail_closed_before_any_aggregation_values",
        "staleness_policy": "max_stale_intervals_reserved_schema_only",
        "stale_max_intervals": int(config.stale_max_intervals),
        "max_staleness_seconds": None,
        "forward_fill_allowed": False,
        "drop_timestamp_allowed": False,
        "runtime_staleness_evaluation_enabled": False,
    }


def _lineage_fields() -> tuple[str, ...]:
    return (
        "schema_version",
        "run_id",
        "created_at_utc",
        "feature_family_name",
        "source_label",
        "source_feature_root",
        "universe",
        "member_assets",
        "excluded_assets",
        "universe_membership_source",
        "included_asset_count",
        "excluded_asset_count",
        "minimum_coverage_threshold",
        "coverage_by_timestamp_window",
        "stale_null_policy",
        "covariance_correlation_estimation_method",
        "shrinkage_method",
        "lookback_window",
        "asset_universe_membership_basis",
        "aggregation_grain",
        "required_source_columns",
        "diagnostics_root_policy",
    )


def _source_lineage(config: MarketAggregateSchemaConfig) -> tuple[SourceArtifactLineage, ...]:
    return (
        SourceArtifactLineage(
            artifact_kind="market_state_feature_source_contract",
            artifact_path=f"metadata://market_state/{config.universe}/{config.source_label}",
            schema_version=MARKET_AGGREGATE_SCHEMA_VERSION,
            produced_by="src.regimes.market_aggregate_schema",
            metadata={
                "pathway": MARKET_STATE_PATHWAY,
                "mode": MARKET_AGGREGATE_SCHEMA_MODE,
                "schema_only": True,
                "source_feature_root": config.source_feature_root,
                "universe_membership_source": config.universe_membership_source,
            },
        ),
    )


def _family_shrinkage_catalog(family: str) -> tuple[str, ...]:
    if family == "shrinkage_covariance_metadata":
        return ("empirical", "ledoit_wolf", "oas")
    return ()


def _family_placeholder_methods(family: str) -> tuple[str, ...]:
    if family == "shrinkage_covariance_metadata":
        return ("future_sparse_precision_placeholder",)
    return ()


def _family_notes(family: str) -> tuple[str, ...]:
    if family in {"covariance_summary", "correlation_summary", "median_pairwise_correlation"}:
        return ("Schema reserves covariance/correlation fields only; no matrix values are computed here.",)
    if family == "first_principal_component_concentration":
        return ("Schema reserves first-principal-component concentration fields; no PCA is fit here.",)
    if family == "shrinkage_covariance_metadata":
        return ("Ledoit-Wolf and OAS are declared for future estimator metadata; sparse precision remains a placeholder.",)
    return ()


def feature_family_contracts(config: MarketAggregateSchemaConfig) -> tuple[MarketAggregateFeatureFamilyContract, ...]:
    constraints = diagnostics_root_constraints()
    coverage = _coverage_policy(config)
    null_staleness = _null_staleness_policy(config)
    coverage_by_window = _coverage_by_timestamp_window(config)
    lineage = _lineage_fields()
    source_lineage = _source_lineage(config)
    return tuple(
        MarketAggregateFeatureFamilyContract(
            feature_family_name=family,
            source_label=config.source_label,
            required_source_columns=tuple(DEFAULT_FAMILY_SOURCE_COLUMNS[family]),
            derived_feature_columns=tuple(DEFAULT_FAMILY_DERIVED_COLUMNS[family]),
            universe_membership_source=config.universe_membership_source,
            included_asset_count=len(config.member_assets),
            excluded_asset_count=len(config.excluded_assets),
            coverage_by_timestamp_window=coverage_by_window,
            minimum_coverage_threshold=config.minimum_coverage_threshold,
            stale_null_policy=null_staleness,
            covariance_correlation_estimation_method=DEFAULT_FAMILY_ESTIMATION_METHOD[family],
            shrinkage_method=DEFAULT_FAMILY_SHRINKAGE_METHOD[family],
            lookback_window=config.lookback_window,
            asset_universe_membership_basis=config.asset_universe_membership_basis,
            aggregation_grain=config.aggregation_grain,
            coverage_policy=coverage,
            null_staleness_policy=null_staleness,
            source_lineage=source_lineage,
            lineage_fields=lineage,
            diagnostics_root_constraints=constraints,
            shrinkage_method_catalog=_family_shrinkage_catalog(family),
            placeholder_methods=_family_placeholder_methods(family),
            notes=_family_notes(family),
            production_classification=config.production_classification,
            schema_version=config.schema_version,
        )
        for family in config.feature_families
    )


def aggregation_frame_input_schema(config: MarketAggregateSchemaConfig) -> MarketAggregationFrameInputSchemaContract:
    return MarketAggregationFrameInputSchemaContract(
        universe=config.universe,
        bands=config.bands,
        source_label=config.source_label,
        timestamp_column=config.timestamp_column,
        asset_column=config.asset_column,
        member_assets=config.member_assets,
        excluded_assets=config.excluded_assets,
        min_assets=config.min_assets,
        universe_membership_source=config.universe_membership_source,
        minimum_coverage_threshold=config.minimum_coverage_threshold,
        coverage_by_timestamp_window=_coverage_by_timestamp_window(config),
        stale_null_policy=_null_staleness_policy(config),
        asset_universe_membership_basis=config.asset_universe_membership_basis,
        aggregation_grain=config.aggregation_grain,
        feature_families=feature_family_contracts(config),
        source_lineage=_source_lineage(config),
        diagnostics_root_constraints=diagnostics_root_constraints(),
        production_classification=config.production_classification,
        schema_version=config.schema_version,
    )


def market_aggregate_schema_diagnostic_path(
    diagnostics_root: Path,
    *,
    universe: str,
    run_id: str,
    filename: str = "schema_diagnostic.json",
) -> Path:
    return (
        Path(diagnostics_root)
        / "market_aggregate_schema_diagnostics"
        / safe_path_part(universe, context="Market aggregate schema universe")
        / safe_path_part(run_id, context="Market aggregate schema run id")
        / safe_path_part(filename, context="Market aggregate schema diagnostic filename")
    )


def build_market_aggregate_schema_diagnostic(
    config: MarketAggregateSchemaConfig,
    *,
    run_id: str = "market_aggregate_schema",
    created_at_utc: str | None = None,
    diagnostics_root_policy: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    created = _created_at(created_at_utc)
    frame_schema = aggregation_frame_input_schema(config).as_dict()
    coverage_by_window = _coverage_by_timestamp_window(config)
    stale_null_policy = _null_staleness_policy(config)
    return {
        "schema_version": int(config.schema_version),
        "artifact_kind": MARKET_AGGREGATE_SCHEMA_ARTIFACT_KIND,
        "status": MARKET_AGGREGATE_SCHEMA_STATUS,
        "mode": MARKET_AGGREGATE_SCHEMA_MODE,
        "pathway": MARKET_STATE_PATHWAY,
        "run_id": str(run_id),
        "created_at_utc": created,
        "universe": config.universe,
        "universe_membership_source": config.universe_membership_source,
        "included_asset_count": int(len(config.member_assets)),
        "excluded_asset_count": int(len(config.excluded_assets)),
        "coverage_by_timestamp_window": coverage_by_window,
        "minimum_coverage_threshold": float(config.minimum_coverage_threshold),
        "stale_null_policy": stale_null_policy,
        "aggregation_grain": config.aggregation_grain,
        "source_lineage": [lineage.as_dict() for lineage in _source_lineage(config)],
        "production_classification": config.production_classification,
        "config": config.as_dict(),
        "feature_family_contracts": frame_schema["feature_families"],
        "aggregation_frame_input_schema": frame_schema,
        "shrinkage_method_catalog": shrinkage_method_catalog(),
        "lineage_fields": list(_lineage_fields()),
        "diagnostics_root_constraints": diagnostics_root_constraints(),
        "diagnostics_root_policy": dict(diagnostics_root_policy or {}),
        "diagnostic_assertions": {
            "metadata_only": True,
            "schema_only": True,
            "market_aggregation_values_computed": False,
            "production_writes_disabled": True,
            "parquet_writes_disabled": True,
            "definition_writes_disabled": True,
            "state_writes_disabled": True,
            "downstream_forecast_readers_disabled": True,
        },
        "artifact_boundary": {
            "write_mode": "market_aggregate_schema_json_only",
            "metadata_only": True,
            "schema_only": True,
            "production_writes_enabled": False,
            "parquet_writes_enabled": False,
            "definition_writes_enabled": False,
            "state_writes_enabled": False,
            "production_outputs_written": False,
            "aggregation_frame_materialized": False,
            "aggregation_values_computed": False,
            "market_aggregate_parquet_output_produced": False,
            "source_reader_enabled": False,
            "relative_readers_enabled": False,
            "downstream_forecast_readers_enabled": False,
        },
        "computed_feature_values": [],
        "value_rows_written": 0,
    }


def write_market_aggregate_schema_diagnostic(
    diagnostics_root: Path,
    config: MarketAggregateSchemaConfig,
    *,
    run_id: str = "market_aggregate_schema",
    created_at_utc: str | None = None,
    project_root: Path | None = None,
    env: Mapping[str, str] | None = None,
) -> Path:
    policy = require_pathway_diagnostics_root(Path(diagnostics_root), project_root=project_root, env=env)
    path = market_aggregate_schema_diagnostic_path(
        Path(diagnostics_root),
        universe=config.universe,
        run_id=str(run_id),
    )
    payload = build_market_aggregate_schema_diagnostic(
        config,
        run_id=str(run_id),
        created_at_utc=created_at_utc,
        diagnostics_root_policy=policy.as_dict(),
    )
    write_json(path, payload, write_kind="Regime market aggregate schema diagnostic")
    return path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Metadata-only market aggregate schema scaffold.")
    parser.add_argument("--run-id", type=str, default="market_aggregate_schema")
    parser.add_argument("--universe", type=str, default="global")
    parser.add_argument("--member-assets", type=str, default="")
    parser.add_argument("--excluded-assets", type=str, default="")
    parser.add_argument("--bands", type=str, default="micro,meso,macro")
    parser.add_argument("--feature-families", type=str, default=",".join(DEFAULT_MARKET_COVARIANCE_FEATURE_FAMILIES))
    parser.add_argument("--source-label", type=str, default=DEFAULT_SOURCE_LABEL)
    parser.add_argument("--source-feature-root", type=str, default="")
    parser.add_argument("--timestamp-column", type=str, default="ts")
    parser.add_argument("--asset-column", type=str, default="asset")
    parser.add_argument("--min-assets", type=int, default=1)
    parser.add_argument("--universe-membership-source", type=str, default=DEFAULT_UNIVERSE_MEMBERSHIP_SOURCE)
    parser.add_argument("--minimum-coverage-threshold", type=float, default=DEFAULT_MINIMUM_COVERAGE_THRESHOLD)
    parser.add_argument("--lookback-window", type=int, default=DEFAULT_LOOKBACK_WINDOW)
    parser.add_argument("--stale-max-intervals", type=int, default=DEFAULT_STALE_MAX_INTERVALS)
    parser.add_argument("--coverage-window-label", type=str, default="schema_window")
    parser.add_argument("--membership-basis", type=str, default=DEFAULT_MEMBERSHIP_BASIS)
    parser.add_argument("--aggregation-grain", type=str, default=DEFAULT_AGGREGATION_GRAIN)
    parser.add_argument("--production-classification", type=str, default="diagnostics_only")
    parser.add_argument("--created-at-utc", type=str, default="")
    parser.add_argument("--diagnostics-root", type=Path, default=None)
    return parser


def config_from_args(args: argparse.Namespace) -> MarketAggregateSchemaConfig:
    return MarketAggregateSchemaConfig(
        universe=str(getattr(args, "universe", "global") or "global"),
        member_assets=_csv(getattr(args, "member_assets", "")),
        excluded_assets=_csv(getattr(args, "excluded_assets", "")),
        bands=_csv(getattr(args, "bands", "")) or tuple(REGIME_BANDS.keys()),
        feature_families=_csv(getattr(args, "feature_families", "")) or DEFAULT_MARKET_COVARIANCE_FEATURE_FAMILIES,
        source_label=str(getattr(args, "source_label", DEFAULT_SOURCE_LABEL) or DEFAULT_SOURCE_LABEL),
        timestamp_column=str(getattr(args, "timestamp_column", "ts") or "ts"),
        asset_column=str(getattr(args, "asset_column", "asset") or "asset"),
        min_assets=int(getattr(args, "min_assets", 1) or 1),
        universe_membership_source=str(
            getattr(args, "universe_membership_source", DEFAULT_UNIVERSE_MEMBERSHIP_SOURCE)
            or DEFAULT_UNIVERSE_MEMBERSHIP_SOURCE
        ),
        minimum_coverage_threshold=float(
            getattr(args, "minimum_coverage_threshold", DEFAULT_MINIMUM_COVERAGE_THRESHOLD)
        ),
        lookback_window=int(getattr(args, "lookback_window", DEFAULT_LOOKBACK_WINDOW) or DEFAULT_LOOKBACK_WINDOW),
        stale_max_intervals=int(getattr(args, "stale_max_intervals", DEFAULT_STALE_MAX_INTERVALS)),
        coverage_window_label=str(getattr(args, "coverage_window_label", "schema_window") or "schema_window"),
        asset_universe_membership_basis=str(
            getattr(args, "membership_basis", DEFAULT_MEMBERSHIP_BASIS) or DEFAULT_MEMBERSHIP_BASIS
        ),
        aggregation_grain=str(
            getattr(args, "aggregation_grain", DEFAULT_AGGREGATION_GRAIN) or DEFAULT_AGGREGATION_GRAIN
        ),
        source_feature_root=str(getattr(args, "source_feature_root", "") or "") or None,
        production_classification=str(getattr(args, "production_classification", "diagnostics_only")),
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    config = config_from_args(args)
    diagnostics_root = getattr(args, "diagnostics_root", None)
    created_at = str(getattr(args, "created_at_utc", "") or "").strip() or None
    policy: PathwayDiagnosticsRootPolicy | None = None
    manifest = build_market_aggregate_schema_diagnostic(
        config,
        run_id=str(args.run_id),
        created_at_utc=created_at,
    )
    if diagnostics_root is not None:
        policy = classify_pathway_diagnostics_root(Path(diagnostics_root))
        path = write_market_aggregate_schema_diagnostic(
            Path(diagnostics_root),
            config,
            run_id=str(args.run_id),
            created_at_utc=created_at,
        )
        manifest["diagnostics_root_policy"] = policy.as_dict()
        manifest["schema_diagnostic_path"] = str(path)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
