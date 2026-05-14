from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from src.regimes.contracts import REGIME_BANDS, RegimeBandContract, band_for_ceiling
from src.regimes.core.pathway_artifacts import (
    PathwayArtifactMetadataSchema,
    PathwayDiagnosticReportSchema,
    PathwaySourceProbeRecord,
    SCALAR_FEATURE_SOURCE_PROBE_KIND,
    artifact_metadata_schema as build_pathway_artifact_metadata_schema,
    band_metadata,
    diagnostic_report_schema as build_pathway_diagnostic_report_schema,
    dry_run_manifest,
    market_aggregation_source_read_precondition as build_market_aggregation_source_read_precondition,
    market_aggregation_source_read_precondition_path as build_market_aggregation_source_read_precondition_path,
    market_membership_snapshot_provenance as build_market_membership_snapshot_provenance,
    market_membership_source_policy as build_market_membership_source_policy,
    market_source_coverage_diagnostic_path as build_market_source_coverage_diagnostic_path,
    market_source_coverage_diagnostic_record as build_market_source_coverage_diagnostic_record,
    market_universe_lifecycle_policy as build_market_universe_lifecycle_policy,
    classify_market_universe_membership_input_path as build_market_universe_membership_input_path_policy,
    load_market_universe_membership_input_file as build_load_market_universe_membership_input_file,
    market_universe_membership_input_from_payload as build_market_universe_membership_input_from_payload,
    market_universe_membership_snapshot as build_market_universe_membership_snapshot,
    market_universe_membership_snapshot_path as build_market_universe_membership_snapshot_path,
    market_universe_snapshot_metadata,
    pathway_source_probe_record,
    relative_benchmark_source_policy as build_relative_benchmark_source_policy,
    relative_peer_basket_lifecycle_policy as build_relative_peer_basket_lifecycle_policy,
    relative_peer_basket_source_policy as build_relative_peer_basket_source_policy,
    relative_source_read_precondition as build_relative_source_read_precondition,
    relative_source_read_precondition_path as build_relative_source_read_precondition_path,
    pathway_month_dir,
    pathway_part_path,
    pathway_table_dir,
    summarize_scalar_feature_source_partitions,
)


UNKNOWN_LABEL = "unknown"
PATHWAY_SCHEMA_VERSION = 1
DEFAULT_TIMESTAMP_COLUMN = "ts"

MARKET_STATE_COVERAGE_COLUMNS: tuple[str, ...] = (
    "member_asset_count",
    "contributing_asset_count",
    "coverage_pct",
)
RELATIVE_STATE_COVERAGE_COLUMNS: tuple[str, ...] = (
    "peer_asset_count",
    "contributing_peer_asset_count",
    "benchmark_present",
    "coverage_pct",
)
MISSING_BENCHMARK_POLICIES: tuple[str, ...] = (
    "require",
    "drop_timestamp",
    "carry_forward",
    "use_universe_proxy",
)


@dataclass(frozen=True)
class PathwayAxisContract:
    name: str
    labels: tuple[str, ...]
    neutral_label: str

    @property
    def label_to_id(self) -> dict[str, int]:
        return {label: idx for idx, label in enumerate(self.labels)}

    @property
    def id_to_label(self) -> dict[int, str]:
        return {idx: label for idx, label in enumerate(self.labels)}

    @property
    def cluster_column(self) -> str:
        return f"{self.name}_cluster_id"

    @property
    def label_column(self) -> str:
        return f"{self.name}_label"

    @property
    def confidence_column(self) -> str:
        return f"{self.name}_confidence_pct"

    @property
    def intensity_column(self) -> str:
        return f"{self.name}_intensity_pct"

    @property
    def output_columns(self) -> tuple[str, ...]:
        return (
            self.cluster_column,
            self.label_column,
            self.confidence_column,
            self.intensity_column,
        )


@dataclass(frozen=True)
class PathwayContract:
    name: str
    table_prefix: str
    key_columns: tuple[str, ...]
    partition_columns: tuple[str, ...]
    axis_order: tuple[str, ...]
    axes: Mapping[str, PathwayAxisContract]
    diagnostic_columns: tuple[str, ...] = ("feature_schema_hash",)

    @property
    def base_output_columns(self) -> tuple[str, ...]:
        return (*self.key_columns, "ceiling_interval_min")

    @property
    def required_output_columns(self) -> tuple[str, ...]:
        return (
            *self.base_output_columns,
            *(column for axis in self.axis_order for column in self.axes[axis].output_columns),
            *self.diagnostic_columns,
        )

    def axis(self, axis_name: str) -> PathwayAxisContract:
        try:
            return self.axes[str(axis_name)]
        except KeyError as exc:
            valid = ", ".join(self.axis_order)
            raise ValueError(f"Unsupported {self.name} axis {axis_name!r}; expected one of: {valid}") from exc

    def table_dir(self, ceiling_interval_min: int) -> str:
        band_for_ceiling(int(ceiling_interval_min))
        return f"{self.table_prefix}_{int(ceiling_interval_min)}"


ASSET_STATE_PATHWAY = "asset_state"
MARKET_STATE_PATHWAY = "market_state"
RELATIVE_STATE_PATHWAY = "relative_state"


MARKET_STATE_AXIS_ORDER: tuple[str, ...] = (
    "breadth",
    "dispersion",
    "correlation",
    "market_vol",
    "leadership",
)
MARKET_STATE_AXES: Mapping[str, PathwayAxisContract] = {
    "breadth": PathwayAxisContract("breadth", ("low", "normal", "high"), "normal"),
    "dispersion": PathwayAxisContract("dispersion", ("low", "normal", "high"), "normal"),
    "correlation": PathwayAxisContract("correlation", ("low", "normal", "high"), "normal"),
    "market_vol": PathwayAxisContract("market_vol", ("low", "normal", "high"), "normal"),
    "leadership": PathwayAxisContract("leadership", ("concentrated", "balanced", "rotating"), "balanced"),
}

RELATIVE_STATE_AXIS_ORDER: tuple[str, ...] = (
    "beta",
    "correlation",
    "relative_strength",
    "relative_dispersion",
)
RELATIVE_STATE_AXES: Mapping[str, PathwayAxisContract] = {
    "beta": PathwayAxisContract("beta", ("low", "neutral", "high"), "neutral"),
    "correlation": PathwayAxisContract("correlation", ("low", "normal", "high"), "normal"),
    "relative_strength": PathwayAxisContract("relative_strength", ("lagging", "neutral", "leading"), "neutral"),
    "relative_dispersion": PathwayAxisContract("relative_dispersion", ("compressed", "normal", "dispersed"), "normal"),
}


MARKET_STATE_CONTRACT = PathwayContract(
    name=MARKET_STATE_PATHWAY,
    table_prefix="market_regimes",
    key_columns=("ts", "universe", "band"),
    partition_columns=("universe", "year", "month"),
    axis_order=MARKET_STATE_AXIS_ORDER,
    axes=MARKET_STATE_AXES,
)

RELATIVE_STATE_CONTRACT = PathwayContract(
    name=RELATIVE_STATE_PATHWAY,
    table_prefix="relative_regimes",
    key_columns=("ts", "asset", "universe", "benchmark", "band"),
    partition_columns=("universe", "benchmark", "asset", "year", "month"),
    axis_order=RELATIVE_STATE_AXIS_ORDER,
    axes=RELATIVE_STATE_AXES,
)

REGIME_PATHWAYS: Mapping[str, PathwayContract] = {
    MARKET_STATE_CONTRACT.name: MARKET_STATE_CONTRACT,
    RELATIVE_STATE_CONTRACT.name: RELATIVE_STATE_CONTRACT,
}


@dataclass(frozen=True)
class FeatureGroupSpec:
    name: str
    candidate_columns: tuple[str, ...]
    description: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "candidate_columns": list(self.candidate_columns),
            "description": self.description,
        }


@dataclass(frozen=True)
class FeatureManifestSpec:
    pathway: str
    groups: tuple[FeatureGroupSpec, ...]

    @property
    def group_names(self) -> tuple[str, ...]:
        return tuple(group.name for group in self.groups)

    def as_dict(self) -> dict[str, Any]:
        return {
            "pathway": self.pathway,
            "groups": [group.as_dict() for group in self.groups],
        }


MARKET_STATE_FEATURE_MANIFEST = FeatureManifestSpec(
    pathway=MARKET_STATE_PATHWAY,
    groups=(
        FeatureGroupSpec(
            "breadth",
            ("advance_decline", "positive_return_share", "above_moving_average_share"),
            "Universe-wide participation and directional agreement candidates.",
        ),
        FeatureGroupSpec(
            "dispersion",
            ("cross_sectional_return_std", "cross_sectional_vol_std", "return_iqr"),
            "Cross-sectional spread candidates across member assets.",
        ),
        FeatureGroupSpec(
            "correlation",
            ("mean_pairwise_correlation", "median_pairwise_correlation", "correlation_breadth"),
            "Universe synchronization candidates.",
        ),
        FeatureGroupSpec(
            "market_vol",
            ("index_realized_vol", "median_asset_realized_vol", "vol_of_vol"),
            "Aggregate realized volatility candidates.",
        ),
        FeatureGroupSpec(
            "leadership",
            ("top_decile_contribution", "sector_leadership_share", "leader_rotation_rate"),
            "Concentration and rotation candidates for market leadership.",
        ),
    ),
)

RELATIVE_STATE_FEATURE_MANIFEST = FeatureManifestSpec(
    pathway=RELATIVE_STATE_PATHWAY,
    groups=(
        FeatureGroupSpec(
            "beta",
            ("rolling_beta_to_benchmark", "downside_beta", "upside_beta"),
            "Asset sensitivity candidates relative to the benchmark or universe.",
        ),
        FeatureGroupSpec(
            "correlation",
            ("rolling_corr_to_benchmark", "rolling_corr_to_universe", "correlation_stability"),
            "Relative co-movement candidates.",
        ),
        FeatureGroupSpec(
            "relative_strength",
            ("relative_return", "relative_momentum_rank", "excess_return_zscore"),
            "Outperformance and underperformance candidates.",
        ),
        FeatureGroupSpec(
            "relative_dispersion",
            ("distance_from_universe_median", "rank_volatility", "cross_sectional_zscore"),
            "Asset position candidates inside the current universe distribution.",
        ),
    ),
)


@dataclass(frozen=True)
class PathwayInputValidationResult:
    pathway: str
    row_count: int
    required_columns: tuple[str, ...]
    missing_columns: tuple[str, ...]
    feature_columns_present: Mapping[str, tuple[str, ...]]
    feature_columns_missing: Mapping[str, tuple[str, ...]]
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return not self.errors

    def raise_for_errors(self) -> None:
        if self.errors:
            raise ValueError("; ".join(self.errors))

    def as_dict(self) -> dict[str, Any]:
        return {
            "pathway": self.pathway,
            "row_count": int(self.row_count),
            "required_columns": list(self.required_columns),
            "missing_columns": list(self.missing_columns),
            "feature_columns_present": {
                str(group): list(columns) for group, columns in self.feature_columns_present.items()
            },
            "feature_columns_missing": {
                str(group): list(columns) for group, columns in self.feature_columns_missing.items()
            },
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "ok": self.ok,
        }


@dataclass(frozen=True)
class MarketStateInputFrameContract:
    universe: str
    band: str
    member_assets: tuple[str, ...]
    feature_groups: tuple[str, ...] = MARKET_STATE_FEATURE_MANIFEST.group_names
    timestamp_column: str = DEFAULT_TIMESTAMP_COLUMN
    min_assets: int = 10
    schema_version: int = PATHWAY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_bands((self.band,))
        _validate_feature_groups(self.feature_groups, MARKET_STATE_FEATURE_MANIFEST)
        if not str(self.universe).strip():
            raise ValueError("Market-state input universe must be non-empty")
        if not tuple(str(asset).strip() for asset in self.member_assets if str(asset).strip()):
            raise ValueError("Market-state input member_assets must be non-empty")
        if int(self.min_assets) < 1:
            raise ValueError("Market-state input min_assets must be positive")
        if int(self.min_assets) > len(self.member_assets):
            raise ValueError("Market-state input min_assets cannot exceed member_assets count")
        if not str(self.timestamp_column).strip():
            raise ValueError("Market-state input timestamp_column must be non-empty")

    @property
    def band_contract(self) -> RegimeBandContract:
        return REGIME_BANDS[str(self.band)]

    @property
    def source_intervals(self) -> tuple[int, ...]:
        return tuple(int(interval) for interval in self.band_contract.member_intervals)

    @property
    def base_columns(self) -> tuple[str, ...]:
        return (self.timestamp_column, "universe", "band", "ceiling_interval_min")

    @property
    def required_columns(self) -> tuple[str, ...]:
        return (*self.base_columns, *MARKET_STATE_COVERAGE_COLUMNS)

    @property
    def candidate_columns_by_group(self) -> dict[str, tuple[str, ...]]:
        return _candidate_columns_by_group(MARKET_STATE_FEATURE_MANIFEST, self.feature_groups)

    def validate_frame(self, frame: pd.DataFrame) -> PathwayInputValidationResult:
        return validate_market_state_input_frame(frame, self)

    def as_dict(self) -> dict[str, Any]:
        return {
            "pathway": MARKET_STATE_PATHWAY,
            "schema_version": int(self.schema_version),
            "universe": self.universe,
            "band": self.band,
            "ceiling_interval_min": int(self.band_contract.ceiling_interval_min),
            "member_assets": list(self.member_assets),
            "timestamp_column": self.timestamp_column,
            "source_intervals": list(self.source_intervals),
            "min_assets": int(self.min_assets),
            "coverage_columns": list(MARKET_STATE_COVERAGE_COLUMNS),
            "required_columns": list(self.required_columns),
            "feature_columns_by_group": {
                group: list(columns) for group, columns in self.candidate_columns_by_group.items()
            },
        }


@dataclass(frozen=True)
class RelativeStateInputFrameContract:
    universe: str
    benchmark: str
    asset: str
    band: str
    peer_assets: tuple[str, ...]
    feature_groups: tuple[str, ...] = RELATIVE_STATE_FEATURE_MANIFEST.group_names
    timestamp_column: str = DEFAULT_TIMESTAMP_COLUMN
    min_peer_assets: int = 5
    missing_benchmark_policy: str = "require"
    schema_version: int = PATHWAY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_bands((self.band,))
        _validate_feature_groups(self.feature_groups, RELATIVE_STATE_FEATURE_MANIFEST)
        if not str(self.universe).strip():
            raise ValueError("Relative-state input universe must be non-empty")
        if not str(self.benchmark).strip():
            raise ValueError("Relative-state input benchmark must be non-empty")
        if not str(self.asset).strip():
            raise ValueError("Relative-state input asset must be non-empty")
        if not tuple(str(asset).strip() for asset in self.peer_assets if str(asset).strip()):
            raise ValueError("Relative-state input peer_assets must be non-empty")
        if int(self.min_peer_assets) < 1:
            raise ValueError("Relative-state input min_peer_assets must be positive")
        if int(self.min_peer_assets) > len(self.peer_assets):
            raise ValueError("Relative-state input min_peer_assets cannot exceed peer_assets count")
        if str(self.missing_benchmark_policy) not in MISSING_BENCHMARK_POLICIES:
            valid = ", ".join(MISSING_BENCHMARK_POLICIES)
            raise ValueError(
                f"Unsupported missing benchmark policy {self.missing_benchmark_policy!r}; expected one of: {valid}"
            )
        if not str(self.timestamp_column).strip():
            raise ValueError("Relative-state input timestamp_column must be non-empty")

    @property
    def band_contract(self) -> RegimeBandContract:
        return REGIME_BANDS[str(self.band)]

    @property
    def source_intervals(self) -> tuple[int, ...]:
        return tuple(int(interval) for interval in self.band_contract.member_intervals)

    @property
    def base_columns(self) -> tuple[str, ...]:
        return (self.timestamp_column, "asset", "universe", "benchmark", "band", "ceiling_interval_min")

    @property
    def required_columns(self) -> tuple[str, ...]:
        return (*self.base_columns, *RELATIVE_STATE_COVERAGE_COLUMNS)

    @property
    def candidate_columns_by_group(self) -> dict[str, tuple[str, ...]]:
        return _candidate_columns_by_group(RELATIVE_STATE_FEATURE_MANIFEST, self.feature_groups)

    def validate_frame(self, frame: pd.DataFrame) -> PathwayInputValidationResult:
        return validate_relative_state_input_frame(frame, self)

    def as_dict(self) -> dict[str, Any]:
        return {
            "pathway": RELATIVE_STATE_PATHWAY,
            "schema_version": int(self.schema_version),
            "universe": self.universe,
            "benchmark": self.benchmark,
            "asset": self.asset,
            "band": self.band,
            "ceiling_interval_min": int(self.band_contract.ceiling_interval_min),
            "peer_assets": list(self.peer_assets),
            "timestamp_column": self.timestamp_column,
            "source_intervals": list(self.source_intervals),
            "min_peer_assets": int(self.min_peer_assets),
            "missing_benchmark_policy": self.missing_benchmark_policy,
            "coverage_columns": list(RELATIVE_STATE_COVERAGE_COLUMNS),
            "required_columns": list(self.required_columns),
            "feature_columns_by_group": {
                group: list(columns) for group, columns in self.candidate_columns_by_group.items()
            },
        }


@dataclass(frozen=True)
class MarketStateConfig:
    universe: str
    member_assets: tuple[str, ...] = ()
    bands: tuple[str, ...] = tuple(REGIME_BANDS.keys())
    feature_groups: tuple[str, ...] = MARKET_STATE_FEATURE_MANIFEST.group_names
    timestamp_column: str = DEFAULT_TIMESTAMP_COLUMN
    min_assets: int = 10
    pathway: str = MARKET_STATE_PATHWAY
    schema_version: int = PATHWAY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_bands(self.bands)
        _validate_feature_groups(self.feature_groups, MARKET_STATE_FEATURE_MANIFEST)
        if int(self.min_assets) < 1:
            raise ValueError("Market-state min_assets must be positive")
        if not str(self.universe).strip():
            raise ValueError("Market-state universe must be non-empty")
        if self.member_assets and int(self.min_assets) > len(self.member_assets):
            raise ValueError("Market-state min_assets cannot exceed member_assets count")
        if not str(self.timestamp_column).strip():
            raise ValueError("Market-state timestamp_column must be non-empty")

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "pathway": self.pathway,
            "universe": self.universe,
            "member_assets": list(self.member_assets),
            "bands": list(self.bands),
            "feature_groups": list(self.feature_groups),
            "timestamp_column": self.timestamp_column,
            "min_assets": int(self.min_assets),
        }


@dataclass(frozen=True)
class RelativeStateConfig:
    universe: str
    benchmark: str
    assets: tuple[str, ...] = ()
    peer_assets: tuple[str, ...] = ()
    bands: tuple[str, ...] = tuple(REGIME_BANDS.keys())
    feature_groups: tuple[str, ...] = RELATIVE_STATE_FEATURE_MANIFEST.group_names
    timestamp_column: str = DEFAULT_TIMESTAMP_COLUMN
    min_peer_assets: int = 5
    missing_benchmark_policy: str = "require"
    pathway: str = RELATIVE_STATE_PATHWAY
    schema_version: int = PATHWAY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_bands(self.bands)
        _validate_feature_groups(self.feature_groups, RELATIVE_STATE_FEATURE_MANIFEST)
        if int(self.min_peer_assets) < 1:
            raise ValueError("Relative-state min_peer_assets must be positive")
        if not str(self.universe).strip():
            raise ValueError("Relative-state universe must be non-empty")
        if not str(self.benchmark).strip():
            raise ValueError("Relative-state benchmark must be non-empty")
        if self.peer_assets and int(self.min_peer_assets) > len(self.peer_assets):
            raise ValueError("Relative-state min_peer_assets cannot exceed peer_assets count")
        if str(self.missing_benchmark_policy) not in MISSING_BENCHMARK_POLICIES:
            valid = ", ".join(MISSING_BENCHMARK_POLICIES)
            raise ValueError(f"Unsupported missing benchmark policy {self.missing_benchmark_policy!r}; expected one of: {valid}")
        if not str(self.timestamp_column).strip():
            raise ValueError("Relative-state timestamp_column must be non-empty")

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "pathway": self.pathway,
            "universe": self.universe,
            "benchmark": self.benchmark,
            "assets": list(self.assets),
            "peer_assets": list(self.peer_assets),
            "bands": list(self.bands),
            "feature_groups": list(self.feature_groups),
            "timestamp_column": self.timestamp_column,
            "min_peer_assets": int(self.min_peer_assets),
            "missing_benchmark_policy": self.missing_benchmark_policy,
        }


ArtifactMetadataSchema = PathwayArtifactMetadataSchema
DiagnosticReportSchema = PathwayDiagnosticReportSchema
SourceProbeRecord = PathwaySourceProbeRecord


def _validate_bands(bands: Sequence[str]) -> None:
    invalid = [str(band) for band in bands if str(band) not in REGIME_BANDS]
    if invalid:
        valid = ", ".join(REGIME_BANDS)
        raise ValueError(f"Unsupported Regime bands {invalid}; expected one of: {valid}")


def _validate_feature_groups(feature_groups: Sequence[str], manifest: FeatureManifestSpec) -> None:
    valid = set(manifest.group_names)
    invalid = [str(group) for group in feature_groups if str(group) not in valid]
    if invalid:
        valid_text = ", ".join(manifest.group_names)
        raise ValueError(f"Unsupported {manifest.pathway} feature groups {invalid}; expected one of: {valid_text}")


def _candidate_columns_by_group(
    manifest: FeatureManifestSpec,
    feature_groups: Sequence[str],
) -> dict[str, tuple[str, ...]]:
    wanted = tuple(str(group) for group in feature_groups)
    by_name = {group.name: group for group in manifest.groups}
    return {group: tuple(by_name[group].candidate_columns) for group in wanted}


def _feature_group_presence(
    frame: pd.DataFrame,
    expected_by_group: Mapping[str, Sequence[str]],
) -> tuple[dict[str, tuple[str, ...]], dict[str, tuple[str, ...]], list[str]]:
    present: dict[str, tuple[str, ...]] = {}
    missing: dict[str, tuple[str, ...]] = {}
    errors: list[str] = []
    for group, columns in expected_by_group.items():
        expected = tuple(str(column) for column in columns)
        present_cols = tuple(column for column in expected if column in frame.columns)
        missing_cols = tuple(column for column in expected if column not in frame.columns)
        present[str(group)] = present_cols
        missing[str(group)] = missing_cols
        if not present_cols:
            errors.append(f"Feature group {group!r} has no candidate columns present")
            continue
        for column in present_cols:
            numeric = pd.to_numeric(frame[column], errors="coerce")
            if not bool(numeric.notna().any()):
                errors.append(f"Feature column {column!r} has no numeric values")
    return present, missing, errors


def _validate_timestamp_series(frame: pd.DataFrame, timestamp_column: str) -> list[str]:
    if timestamp_column not in frame.columns:
        return []
    ts = pd.to_numeric(frame[timestamp_column], errors="coerce")
    errors: list[str] = []
    if bool(ts.isna().any()):
        errors.append(f"Timestamp column {timestamp_column!r} contains non-numeric values")
    if bool(ts.duplicated().any()):
        errors.append(f"Timestamp column {timestamp_column!r} contains duplicate timestamps")
    if len(ts) > 1 and not bool(ts.is_monotonic_increasing):
        errors.append(f"Timestamp column {timestamp_column!r} must be monotonic increasing")
    return errors


def _coverage_errors(frame: pd.DataFrame, *, coverage_column: str = "coverage_pct") -> list[str]:
    if coverage_column not in frame.columns:
        return []
    coverage = pd.to_numeric(frame[coverage_column], errors="coerce")
    errors: list[str] = []
    if bool(coverage.isna().any()):
        errors.append(f"Coverage column {coverage_column!r} contains non-numeric values")
    if bool(((coverage < 0.0) | (coverage > 1.0)).any()):
        errors.append(f"Coverage column {coverage_column!r} must be within [0, 1]")
    return errors


def _bool_series(
    frame: pd.DataFrame,
    column: str,
) -> tuple[pd.Series, list[str]]:
    if column not in frame.columns:
        return pd.Series(dtype=bool), []
    raw = frame[column]
    if pd.api.types.is_bool_dtype(raw):
        return raw.fillna(False).astype(bool), []
    if pd.api.types.is_numeric_dtype(raw):
        numeric = pd.to_numeric(raw, errors="coerce")
        invalid = numeric.isna() | ~numeric.isin((0, 1))
        errors = [f"Boolean column {column!r} must contain bool, 0/1, or true/false values"] if bool(invalid.any()) else []
        return numeric.fillna(0).astype(int).astype(bool), errors
    normalized = raw.astype(str).str.strip().str.lower()
    true_values = {"true", "1", "yes", "y"}
    false_values = {"false", "0", "no", "n"}
    invalid = ~normalized.isin(true_values | false_values)
    errors = [f"Boolean column {column!r} must contain bool, 0/1, or true/false values"] if bool(invalid.any()) else []
    parsed = normalized.isin(true_values)
    return parsed.astype(bool), errors


def validate_market_state_input_frame(
    frame: pd.DataFrame,
    contract: MarketStateInputFrameContract,
) -> PathwayInputValidationResult:
    required = tuple(contract.required_columns)
    missing = tuple(column for column in required if column not in frame.columns)
    present, feature_missing, feature_errors = _feature_group_presence(frame, contract.candidate_columns_by_group)
    errors = [f"Missing required columns: {list(missing)}"] if missing else []
    errors.extend(feature_errors)
    warnings: list[str] = []
    if frame.empty:
        warnings.append("Market-state input frame is empty")
    if not missing:
        errors.extend(_validate_timestamp_series(frame, contract.timestamp_column))
        errors.extend(_coverage_errors(frame))
        if not frame.empty:
            if not frame["universe"].astype(str).eq(str(contract.universe)).all():
                errors.append("Market-state input universe values do not match contract")
            if not frame["band"].astype(str).eq(str(contract.band)).all():
                errors.append("Market-state input band values do not match contract")
            ceiling = pd.to_numeric(frame["ceiling_interval_min"], errors="coerce")
            if not bool(ceiling.eq(int(contract.band_contract.ceiling_interval_min)).all()):
                errors.append("Market-state input ceiling_interval_min values do not match band")
            member_count = pd.to_numeric(frame["member_asset_count"], errors="coerce")
            contributing = pd.to_numeric(frame["contributing_asset_count"], errors="coerce")
            if not bool(member_count.eq(len(contract.member_assets)).all()):
                errors.append("Market-state input member_asset_count does not match member_assets count")
            if bool((contributing < int(contract.min_assets)).any()):
                errors.append("Market-state input contributing_asset_count is below min_assets")
            if bool((contributing > member_count).any()):
                errors.append("Market-state input contributing_asset_count exceeds member_asset_count")
    return PathwayInputValidationResult(
        pathway=MARKET_STATE_PATHWAY,
        row_count=int(len(frame)),
        required_columns=required,
        missing_columns=missing,
        feature_columns_present=present,
        feature_columns_missing=feature_missing,
        errors=tuple(errors),
        warnings=tuple(warnings),
    )


def validate_relative_state_input_frame(
    frame: pd.DataFrame,
    contract: RelativeStateInputFrameContract,
) -> PathwayInputValidationResult:
    required = tuple(contract.required_columns)
    missing = tuple(column for column in required if column not in frame.columns)
    present, feature_missing, feature_errors = _feature_group_presence(frame, contract.candidate_columns_by_group)
    errors = [f"Missing required columns: {list(missing)}"] if missing else []
    errors.extend(feature_errors)
    warnings: list[str] = []
    if frame.empty:
        warnings.append("Relative-state input frame is empty")
    if not missing:
        errors.extend(_validate_timestamp_series(frame, contract.timestamp_column))
        errors.extend(_coverage_errors(frame))
        if not frame.empty:
            if not frame["asset"].astype(str).eq(str(contract.asset)).all():
                errors.append("Relative-state input asset values do not match contract")
            if not frame["universe"].astype(str).eq(str(contract.universe)).all():
                errors.append("Relative-state input universe values do not match contract")
            if not frame["benchmark"].astype(str).eq(str(contract.benchmark)).all():
                errors.append("Relative-state input benchmark values do not match contract")
            if not frame["band"].astype(str).eq(str(contract.band)).all():
                errors.append("Relative-state input band values do not match contract")
            ceiling = pd.to_numeric(frame["ceiling_interval_min"], errors="coerce")
            if not bool(ceiling.eq(int(contract.band_contract.ceiling_interval_min)).all()):
                errors.append("Relative-state input ceiling_interval_min values do not match band")
            peer_count = pd.to_numeric(frame["peer_asset_count"], errors="coerce")
            contributing = pd.to_numeric(frame["contributing_peer_asset_count"], errors="coerce")
            if not bool(peer_count.eq(len(contract.peer_assets)).all()):
                errors.append("Relative-state input peer_asset_count does not match peer_assets count")
            if bool((contributing < int(contract.min_peer_assets)).any()):
                errors.append("Relative-state input contributing_peer_asset_count is below min_peer_assets")
            if bool((contributing > peer_count).any()):
                errors.append("Relative-state input contributing_peer_asset_count exceeds peer_asset_count")
            benchmark_present, benchmark_errors = _bool_series(frame, "benchmark_present")
            errors.extend(benchmark_errors)
            if contract.missing_benchmark_policy == "require" and not bool(benchmark_present.all()):
                errors.append("Relative-state input benchmark_present must be true when missing_benchmark_policy='require'")
    return PathwayInputValidationResult(
        pathway=RELATIVE_STATE_PATHWAY,
        row_count=int(len(frame)),
        required_columns=required,
        missing_columns=missing,
        feature_columns_present=present,
        feature_columns_missing=feature_missing,
        errors=tuple(errors),
        warnings=tuple(warnings),
    )


def _json_scalar(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def _numeric_min_max(frame: pd.DataFrame, column: str) -> dict[str, Any]:
    if column not in frame.columns or frame.empty:
        return {"min": None, "max": None}
    numeric = pd.to_numeric(frame[column], errors="coerce").dropna()
    if numeric.empty:
        return {"min": None, "max": None}
    return {"min": _json_scalar(numeric.min()), "max": _json_scalar(numeric.max())}


def _unique_values(frame: pd.DataFrame, column: str, *, limit: int = 20) -> list[Any]:
    if column not in frame.columns or frame.empty:
        return []
    values = frame[column].dropna().unique().tolist()
    values = [_json_scalar(value) for value in values]
    return values[: max(1, int(limit))]


def _market_member_assets_for_probe(config: MarketStateConfig, frame: pd.DataFrame) -> tuple[tuple[str, ...], str]:
    configured = tuple(str(asset) for asset in config.member_assets if str(asset).strip())
    if configured:
        return configured, "config.member_assets"
    member_count = int(config.min_assets)
    if "member_asset_count" in frame.columns and not frame.empty:
        counts = pd.to_numeric(frame["member_asset_count"], errors="coerce").dropna()
        if not counts.empty:
            member_count = max(member_count, int(counts.max()))
    return tuple(f"FRAME_MEMBER_{idx:02d}" for idx in range(max(member_count, 1))), "frame.member_asset_count"


def _market_probe_source_summary(
    frame: pd.DataFrame,
    contract: MarketStateInputFrameContract,
    *,
    membership_source: str,
    snapshot_timestamp_utc: str | None = None,
) -> dict[str, Any]:
    snapshot = market_universe_snapshot_metadata(
        universe=contract.universe,
        member_assets=contract.member_assets,
        membership_source=membership_source,
        snapshot_timestamp_utc=snapshot_timestamp_utc,
        snapshot_scope="market_state_aggregation_source_probe",
        min_assets=contract.min_assets,
    ).as_dict()
    try:
        lifecycle_policy: dict[str, Any] | None = market_universe_lifecycle_policy(
            MarketStateConfig(
                universe=contract.universe,
                member_assets=contract.member_assets,
                bands=(contract.band,),
                feature_groups=contract.feature_groups,
                timestamp_column=contract.timestamp_column,
                min_assets=contract.min_assets,
                schema_version=contract.schema_version,
            ),
            membership_source=membership_source,
        )
    except ValueError:
        lifecycle_policy = None
    return {
        "pathway": MARKET_STATE_PATHWAY,
        "probe_kind": "aggregation_frame",
        "status": "scaffold_only",
        "row_count": int(len(frame)),
        "column_count": int(len(frame.columns)),
        "columns": [str(column) for column in frame.columns],
        "timestamp_column": contract.timestamp_column,
        "timestamp_range": _numeric_min_max(frame, contract.timestamp_column),
        "universe": contract.universe,
        "universe_values": _unique_values(frame, "universe"),
        "band": contract.band,
        "band_values": _unique_values(frame, "band"),
        "ceiling_interval_min": int(contract.band_contract.ceiling_interval_min),
        "ceiling_interval_min_values": _unique_values(frame, "ceiling_interval_min"),
        "source_intervals": list(contract.source_intervals),
        "member_assets_count": int(len(contract.member_assets)),
        "member_assets_source": membership_source,
        "lifecycle_policy": lifecycle_policy,
        "universe_snapshot": snapshot,
        "min_assets": int(contract.min_assets),
        "member_asset_count_range": _numeric_min_max(frame, "member_asset_count"),
        "contributing_asset_count_range": _numeric_min_max(frame, "contributing_asset_count"),
        "coverage_pct_range": _numeric_min_max(frame, "coverage_pct"),
        "feature_groups": list(contract.feature_groups),
    }


def market_state_aggregation_source_probe(
    config: MarketStateConfig,
    frame: pd.DataFrame,
    *,
    band: str | None = None,
    run_id: str = "market_state_source_probe",
    created_at_utc: str | None = None,
    membership_source: str | None = None,
) -> SourceProbeRecord:
    probe_band = str(band or (config.bands[0] if config.bands else ""))
    member_assets, derived_membership_source = _market_member_assets_for_probe(config, frame)
    snapshot_membership_source = str(membership_source).strip() if membership_source else derived_membership_source
    contract = market_state_input_contract(config, band=probe_band, member_assets=member_assets)
    validation = contract.validate_frame(frame).as_dict()
    created = created_at_utc or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    source_summary = _market_probe_source_summary(
        frame,
        contract,
        membership_source=snapshot_membership_source,
        snapshot_timestamp_utc=created,
    )
    source_summary["member_assets_source_derived"] = derived_membership_source
    return pathway_source_probe_record(
        pathway=MARKET_STATE_PATHWAY,
        run_id=str(run_id),
        source_summary=source_summary,
        input_validation=validation,
        created_at_utc=created,
    )


def _market_scalar_partition_validation(
    partition_summary: Mapping[str, Any],
    *,
    min_assets: int,
) -> dict[str, Any]:
    missing_assets = tuple(str(asset) for asset in partition_summary.get("missing_assets", ()))
    source_root_exists = bool(partition_summary.get("source_root_exists"))
    assets_with_all_intervals = int(partition_summary.get("assets_with_all_intervals", 0) or 0)
    errors: list[str] = []
    warnings: list[str] = []
    if not source_root_exists:
        errors.append("Scalar feature source root does not exist")
    if missing_assets:
        errors.append(f"Scalar feature source partitions are missing for assets: {list(missing_assets)}")
    if assets_with_all_intervals < int(min_assets):
        errors.append("Scalar feature source partition coverage is below min_assets")
    if not bool(partition_summary.get("row_count_estimate_complete", False)):
        warnings.append("Scalar feature row-count estimate is incomplete")
    return {
        "pathway": MARKET_STATE_PATHWAY,
        "probe_kind": SCALAR_FEATURE_SOURCE_PROBE_KIND,
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "source_root_exists": source_root_exists,
        "member_assets_required": True,
        "membership_inferred_from_disk": False,
        "member_assets_count": int(partition_summary.get("member_assets_count", 0) or 0),
        "assets_with_all_intervals": int(assets_with_all_intervals),
        "min_assets": int(min_assets),
        "missing_assets": list(missing_assets),
        "missing_intervals_by_asset": dict(partition_summary.get("missing_intervals_by_asset", {})),
    }


def market_state_scalar_partition_source_probe(
    config: MarketStateConfig,
    *,
    source_feature_root: Path,
    band: str | None = None,
    run_id: str = "market_state_scalar_partition_probe",
    created_at_utc: str | None = None,
    membership_source: str = "config.member_assets",
) -> SourceProbeRecord:
    member_assets = tuple(str(asset).strip() for asset in config.member_assets if str(asset).strip())
    if not member_assets:
        raise ValueError("Market-state scalar partition source probe requires explicit member_assets")
    probe_band = str(band or (config.bands[0] if config.bands else ""))
    contract = market_state_input_contract(config, band=probe_band, member_assets=member_assets)
    created = created_at_utc or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    partition_summary = summarize_scalar_feature_source_partitions(
        source_root=Path(source_feature_root),
        band=probe_band,
        member_assets=member_assets,
        timestamp_column=contract.timestamp_column,
        membership_source=str(membership_source).strip() or "config.member_assets",
    ).as_dict()
    snapshot = market_universe_snapshot_metadata(
        universe=contract.universe,
        member_assets=contract.member_assets,
        membership_source=str(membership_source).strip() or "config.member_assets",
        snapshot_timestamp_utc=created,
        snapshot_scope="market_state_scalar_partition_source_probe",
        min_assets=contract.min_assets,
    ).as_dict()
    source_summary = {
        "pathway": MARKET_STATE_PATHWAY,
        "probe_kind": SCALAR_FEATURE_SOURCE_PROBE_KIND,
        "status": "scaffold_only",
        "universe": contract.universe,
        "band": contract.band,
        "ceiling_interval_min": int(contract.band_contract.ceiling_interval_min),
        "source_intervals": list(contract.source_intervals),
        "feature_groups": list(contract.feature_groups),
        "min_assets": int(contract.min_assets),
        "lifecycle_policy": market_universe_lifecycle_policy(
            config,
            membership_source=str(membership_source).strip() or "config.member_assets",
        ),
        "universe_snapshot": snapshot,
        "aggregation_frame_built": False,
        "aggregation_values_built": False,
        "membership_inferred_from_disk": False,
        **partition_summary,
    }
    return pathway_source_probe_record(
        pathway=MARKET_STATE_PATHWAY,
        run_id=str(run_id),
        source_summary=source_summary,
        input_validation=_market_scalar_partition_validation(partition_summary, min_assets=contract.min_assets),
        created_at_utc=created,
    )


def pathway_contract(pathway: str) -> PathwayContract:
    try:
        return REGIME_PATHWAYS[str(pathway)]
    except KeyError as exc:
        valid = ", ".join(REGIME_PATHWAYS)
        raise ValueError(f"Unsupported Regime pathway {pathway!r}; expected one of: {valid}") from exc


def feature_manifest(pathway: str) -> FeatureManifestSpec:
    if str(pathway) == MARKET_STATE_PATHWAY:
        return MARKET_STATE_FEATURE_MANIFEST
    if str(pathway) == RELATIVE_STATE_PATHWAY:
        return RELATIVE_STATE_FEATURE_MANIFEST
    valid = ", ".join(REGIME_PATHWAYS)
    raise ValueError(f"Unsupported Regime feature manifest pathway {pathway!r}; expected one of: {valid}")


def market_state_input_contract(
    config: MarketStateConfig,
    *,
    band: str,
    member_assets: Sequence[str] | None = None,
) -> MarketStateInputFrameContract:
    return MarketStateInputFrameContract(
        universe=config.universe,
        band=str(band),
        member_assets=tuple(str(asset) for asset in (member_assets or config.member_assets)),
        feature_groups=config.feature_groups,
        timestamp_column=config.timestamp_column,
        min_assets=config.min_assets,
        schema_version=config.schema_version,
    )


def relative_state_input_contract(
    config: RelativeStateConfig,
    *,
    asset: str,
    band: str,
    peer_assets: Sequence[str] | None = None,
) -> RelativeStateInputFrameContract:
    return RelativeStateInputFrameContract(
        universe=config.universe,
        benchmark=config.benchmark,
        asset=str(asset),
        band=str(band),
        peer_assets=tuple(str(peer) for peer in (peer_assets or config.peer_assets)),
        feature_groups=config.feature_groups,
        timestamp_column=config.timestamp_column,
        min_peer_assets=config.min_peer_assets,
        missing_benchmark_policy=config.missing_benchmark_policy,
        schema_version=config.schema_version,
    )


def market_state_input_contract_template(config: MarketStateConfig) -> dict[str, Any]:
    return {
        "pathway": MARKET_STATE_PATHWAY,
        "schema_version": int(config.schema_version),
        "universe": config.universe,
        "member_assets": list(config.member_assets),
        "timestamp_column": config.timestamp_column,
        "min_assets": int(config.min_assets),
        "coverage_columns": list(MARKET_STATE_COVERAGE_COLUMNS),
        "bands": [
            {
                "band": band_name,
                "ceiling_interval_min": int(REGIME_BANDS[str(band_name)].ceiling_interval_min),
                "source_intervals": list(REGIME_BANDS[str(band_name)].member_intervals),
                "required_columns": [
                    config.timestamp_column,
                    "universe",
                    "band",
                    "ceiling_interval_min",
                    *MARKET_STATE_COVERAGE_COLUMNS,
                ],
            }
            for band_name in config.bands
        ],
        "feature_columns_by_group": _candidate_columns_by_group(MARKET_STATE_FEATURE_MANIFEST, config.feature_groups),
    }


def relative_state_input_contract_template(config: RelativeStateConfig) -> dict[str, Any]:
    return {
        "pathway": RELATIVE_STATE_PATHWAY,
        "schema_version": int(config.schema_version),
        "universe": config.universe,
        "benchmark": config.benchmark,
        "assets": list(config.assets),
        "peer_assets": list(config.peer_assets),
        "timestamp_column": config.timestamp_column,
        "min_peer_assets": int(config.min_peer_assets),
        "missing_benchmark_policy": config.missing_benchmark_policy,
        "coverage_columns": list(RELATIVE_STATE_COVERAGE_COLUMNS),
        "bands": [
            {
                "band": band_name,
                "ceiling_interval_min": int(REGIME_BANDS[str(band_name)].ceiling_interval_min),
                "source_intervals": list(REGIME_BANDS[str(band_name)].member_intervals),
                "required_columns": [
                    config.timestamp_column,
                    "asset",
                    "universe",
                    "benchmark",
                    "band",
                    "ceiling_interval_min",
                    *RELATIVE_STATE_COVERAGE_COLUMNS,
                ],
            }
            for band_name in config.bands
        ],
        "feature_columns_by_group": _candidate_columns_by_group(RELATIVE_STATE_FEATURE_MANIFEST, config.feature_groups),
    }


def relative_benchmark_source_policy(
    config: RelativeStateConfig,
    *,
    benchmark_source_kind: str = "config.benchmark",
    benchmark_source_detail: str | None = None,
    substitution_policy: str = "none",
) -> dict[str, Any]:
    return build_relative_benchmark_source_policy(
        benchmark=config.benchmark,
        benchmark_source_kind=benchmark_source_kind,
        benchmark_source_detail=benchmark_source_detail,
        substitution_policy=substitution_policy,
        missing_benchmark_policy=config.missing_benchmark_policy,
    ).as_dict()


def relative_peer_basket_lifecycle_policy(
    config: RelativeStateConfig,
    *,
    peer_source_kind: str = "config.peer_assets",
) -> dict[str, Any]:
    return build_relative_peer_basket_lifecycle_policy(
        universe=config.universe,
        peer_assets=config.peer_assets,
        min_peer_assets=config.min_peer_assets,
        peer_source_kind=peer_source_kind,
    ).as_dict()


def relative_peer_basket_source_policy(
    config: RelativeStateConfig,
    *,
    peer_source_kind: str = "config.peer_assets",
    peer_source_detail: str | None = None,
) -> dict[str, Any]:
    return build_relative_peer_basket_source_policy(
        universe=config.universe,
        peer_assets=config.peer_assets,
        min_peer_assets=config.min_peer_assets,
        peer_source_kind=peer_source_kind,
        peer_source_detail=peer_source_detail,
    ).as_dict()


def relative_source_read_precondition_path(root: Path, universe: str, benchmark: str, run_id: str) -> Path:
    return build_relative_source_read_precondition_path(
        Path(root),
        universe=universe,
        benchmark=benchmark,
        run_id=run_id,
    )


def relative_source_read_precondition(
    config: RelativeStateConfig,
    *,
    run_id: str,
    band: str | None = None,
    benchmark_source_policy: Mapping[str, Any] | None = None,
    peer_basket_lifecycle_policy: Mapping[str, Any] | None = None,
    peer_basket_source_policy: Mapping[str, Any] | None = None,
    created_at_utc: str | None = None,
) -> dict[str, Any]:
    selected_band = str(band or config.bands[0])
    benchmark_policy = benchmark_source_policy or relative_benchmark_source_policy(config)
    peer_lifecycle = peer_basket_lifecycle_policy or relative_peer_basket_lifecycle_policy(config)
    peer_source = peer_basket_source_policy or relative_peer_basket_source_policy(config)
    return build_relative_source_read_precondition(
        run_id=str(run_id),
        universe=config.universe,
        benchmark=config.benchmark,
        assets=config.assets,
        peer_assets=config.peer_assets,
        min_peer_assets=config.min_peer_assets,
        band=selected_band,
        missing_benchmark_policy=config.missing_benchmark_policy,
        benchmark_source_policy=benchmark_policy,
        peer_basket_lifecycle_policy=peer_lifecycle,
        peer_basket_source_policy=peer_source,
        created_at_utc=created_at_utc,
    ).as_dict()


def artifact_metadata_schema(pathway: str) -> ArtifactMetadataSchema:
    contract = pathway_contract(pathway)
    manifest = feature_manifest(pathway)
    return build_pathway_artifact_metadata_schema(
        contract,
        schema_version=PATHWAY_SCHEMA_VERSION,
        feature_manifest_groups=manifest.group_names,
    )


def diagnostic_report_schema(pathway: str) -> DiagnosticReportSchema:
    contract = pathway_contract(pathway)
    return build_pathway_diagnostic_report_schema(contract)


def market_state_table_dir(ceiling_interval_min: int) -> str:
    return pathway_table_dir(MARKET_STATE_CONTRACT.table_prefix, int(ceiling_interval_min))


def market_state_month_dir(root: Path, ceiling_interval_min: int, universe: str, year: int, month: int) -> Path:
    return pathway_month_dir(
        Path(root),
        table_prefix=MARKET_STATE_CONTRACT.table_prefix,
        ceiling_interval_min=int(ceiling_interval_min),
        partitions={"universe": universe},
        year=int(year),
        month=int(month),
    )


def market_state_part_path(root: Path, ceiling_interval_min: int, universe: str, year: int, month: int) -> Path:
    return pathway_part_path(
        Path(root),
        table_prefix=MARKET_STATE_CONTRACT.table_prefix,
        ceiling_interval_min=int(ceiling_interval_min),
        partitions={"universe": universe},
        year=int(year),
        month=int(month),
    )


def market_universe_membership_snapshot_path(root: Path, universe: str, run_id: str) -> Path:
    return build_market_universe_membership_snapshot_path(
        Path(root),
        universe=universe,
        run_id=run_id,
    )


def market_universe_membership_snapshot(
    config: MarketStateConfig,
    *,
    membership_source: str,
    provenance: Mapping[str, Any],
    snapshot_timestamp_utc: str | None = None,
    snapshot_scope: str = "market_state_universe_membership",
) -> dict[str, Any]:
    return build_market_universe_membership_snapshot(
        universe=config.universe,
        member_assets=config.member_assets,
        membership_source=membership_source,
        provenance=provenance,
        snapshot_timestamp_utc=snapshot_timestamp_utc,
        snapshot_scope=snapshot_scope,
        min_assets=config.min_assets,
    ).as_dict()


def market_membership_source_policy(
    config: MarketStateConfig,
    *,
    membership_source: str = "config.member_assets",
    source_detail: str | None = None,
    source_feature_root: Path | str | None = None,
) -> dict[str, Any]:
    return build_market_membership_source_policy(
        membership_source=membership_source,
        member_assets=config.member_assets,
        source_detail=source_detail,
        source_feature_root=source_feature_root,
    ).as_dict()


def market_universe_lifecycle_policy(
    config: MarketStateConfig,
    *,
    membership_source: str = "config.member_assets",
) -> dict[str, Any]:
    return build_market_universe_lifecycle_policy(
        membership_source=membership_source,
    ).as_dict()


def market_membership_snapshot_provenance(
    config: MarketStateConfig,
    *,
    run_id: str,
    membership_source: str = "config.member_assets",
    source_detail: str | None = None,
    source_feature_root: Path | str | None = None,
    extra_provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return build_market_membership_snapshot_provenance(
        membership_source=membership_source,
        member_assets=config.member_assets,
        run_id=run_id,
        source_detail=source_detail,
        source_feature_root=source_feature_root,
        extra_provenance=extra_provenance,
    )


def market_universe_membership_input_path_policy(
    input_path: Path,
    *,
    project_root: Path | None = None,
    env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    return build_market_universe_membership_input_path_policy(
        Path(input_path),
        project_root=project_root,
        env=env,
    ).as_dict()


def market_universe_membership_input_from_payload(
    payload: Mapping[str, Any],
    *,
    expected_universe: str | None = None,
    source_path: Path | str | None = None,
    input_path_policy: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return build_market_universe_membership_input_from_payload(
        payload,
        expected_universe=expected_universe,
        source_path=source_path,
        input_path_policy=input_path_policy,
    ).as_dict()


def load_market_universe_membership_input_file(
    input_path: Path,
    *,
    expected_universe: str | None = None,
    project_root: Path | None = None,
    env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    return build_load_market_universe_membership_input_file(
        Path(input_path),
        expected_universe=expected_universe,
        project_root=project_root,
        env=env,
    ).as_dict()


def market_source_coverage_diagnostic_path(root: Path, universe: str, run_id: str) -> Path:
    return build_market_source_coverage_diagnostic_path(
        Path(root),
        universe=universe,
        run_id=run_id,
    )


def market_aggregation_source_read_precondition_path(root: Path, universe: str, run_id: str) -> Path:
    return build_market_aggregation_source_read_precondition_path(
        Path(root),
        universe=universe,
        run_id=run_id,
    )


def market_aggregation_source_read_precondition(
    config: MarketStateConfig,
    *,
    run_id: str,
    membership_source: str,
    source_probe: SourceProbeRecord | Mapping[str, Any],
    source_probe_path: Path | str | None = None,
    membership_snapshot: Mapping[str, Any] | None = None,
    membership_snapshot_path: Path | str | None = None,
    source_coverage_diagnostic: Mapping[str, Any] | None = None,
    source_coverage_diagnostic_path: Path | str | None = None,
    band: str | None = None,
    created_at_utc: str | None = None,
) -> dict[str, Any]:
    if not config.member_assets:
        raise ValueError("Market aggregation source-read precondition requires explicit member_assets")
    probe_payload: Mapping[str, Any]
    if hasattr(source_probe, "as_dict"):
        probe_payload = source_probe.as_dict()
    else:
        probe_payload = dict(source_probe)
    return build_market_aggregation_source_read_precondition(
        run_id=str(run_id),
        universe=config.universe,
        band=str(band or config.bands[0]),
        member_assets=config.member_assets,
        min_assets=config.min_assets,
        membership_source=membership_source,
        source_probe=probe_payload,
        source_probe_path=source_probe_path,
        membership_snapshot=membership_snapshot,
        membership_snapshot_path=membership_snapshot_path,
        source_coverage_diagnostic=source_coverage_diagnostic,
        source_coverage_diagnostic_path=source_coverage_diagnostic_path,
        created_at_utc=created_at_utc,
    ).as_dict()


def market_source_coverage_diagnostic(
    config: MarketStateConfig,
    *,
    run_id: str,
    membership_source: str,
    membership_snapshot: Mapping[str, Any] | None = None,
    membership_snapshot_path: Path | str | None = None,
    source_probe: SourceProbeRecord | Mapping[str, Any] | None = None,
    source_probe_path: Path | str | None = None,
    band: str | None = None,
    created_at_utc: str | None = None,
) -> dict[str, Any]:
    if not config.member_assets:
        raise ValueError("Market source coverage diagnostic requires explicit member_assets")
    probe_payload: Mapping[str, Any] | None
    if source_probe is None:
        probe_payload = None
    elif hasattr(source_probe, "as_dict"):
        probe_payload = source_probe.as_dict()
    else:
        probe_payload = dict(source_probe)
    return build_market_source_coverage_diagnostic_record(
        run_id=str(run_id),
        universe=config.universe,
        band=str(band or config.bands[0]),
        member_assets=config.member_assets,
        min_assets=config.min_assets,
        membership_source=membership_source,
        membership_snapshot=membership_snapshot,
        membership_snapshot_path=membership_snapshot_path,
        source_probe=probe_payload,
        source_probe_path=source_probe_path,
        created_at_utc=created_at_utc,
    ).as_dict()


def relative_state_table_dir(ceiling_interval_min: int) -> str:
    return pathway_table_dir(RELATIVE_STATE_CONTRACT.table_prefix, int(ceiling_interval_min))


def relative_state_month_dir(
    root: Path,
    ceiling_interval_min: int,
    universe: str,
    benchmark: str,
    asset: str,
    year: int,
    month: int,
) -> Path:
    return pathway_month_dir(
        Path(root),
        table_prefix=RELATIVE_STATE_CONTRACT.table_prefix,
        ceiling_interval_min=int(ceiling_interval_min),
        partitions={"universe": universe, "benchmark": benchmark, "asset": asset},
        year=int(year),
        month=int(month),
    )


def relative_state_part_path(
    root: Path,
    ceiling_interval_min: int,
    universe: str,
    benchmark: str,
    asset: str,
    year: int,
    month: int,
) -> Path:
    return pathway_part_path(
        Path(root),
        table_prefix=RELATIVE_STATE_CONTRACT.table_prefix,
        ceiling_interval_min=int(ceiling_interval_min),
        partitions={"universe": universe, "benchmark": benchmark, "asset": asset},
        year=int(year),
        month=int(month),
    )


def market_state_run_manifest(
    config: MarketStateConfig,
    *,
    run_id: str = "market_state_scaffold",
    membership_source: str = "config.member_assets",
) -> dict[str, Any]:
    manifest = dry_run_manifest(
        run_id=str(run_id),
        config=config.as_dict(),
        contract=artifact_metadata_schema(MARKET_STATE_PATHWAY),
        feature_manifest=MARKET_STATE_FEATURE_MANIFEST.as_dict(),
        diagnostics=diagnostic_report_schema(MARKET_STATE_PATHWAY),
        bands=config.bands,
    )
    manifest["lifecycle_policy"] = market_universe_lifecycle_policy(
        config,
        membership_source=str(membership_source).strip() or "config.member_assets",
    )
    manifest["input_frame_contract"] = market_state_input_contract_template(config)
    return manifest


def relative_state_run_manifest(config: RelativeStateConfig, *, run_id: str = "relative_state_scaffold") -> dict[str, Any]:
    manifest = dry_run_manifest(
        run_id=str(run_id),
        config=config.as_dict(),
        contract=artifact_metadata_schema(RELATIVE_STATE_PATHWAY),
        feature_manifest=RELATIVE_STATE_FEATURE_MANIFEST.as_dict(),
        diagnostics=diagnostic_report_schema(RELATIVE_STATE_PATHWAY),
        bands=config.bands,
    )
    benchmark_policy = relative_benchmark_source_policy(config)
    peer_lifecycle = relative_peer_basket_lifecycle_policy(config)
    peer_source = relative_peer_basket_source_policy(config)
    manifest["benchmark_source_policy"] = benchmark_policy
    manifest["peer_basket_lifecycle_policy"] = peer_lifecycle
    manifest["peer_basket_source_policy"] = peer_source
    manifest["source_read_precondition"] = relative_source_read_precondition(
        config,
        run_id=str(run_id),
        benchmark_source_policy=benchmark_policy,
        peer_basket_lifecycle_policy=peer_lifecycle,
        peer_basket_source_policy=peer_source,
    )
    manifest["input_frame_contract"] = relative_state_input_contract_template(config)
    return manifest


__all__ = [
    "ASSET_STATE_PATHWAY",
    "ArtifactMetadataSchema",
    "DiagnosticReportSchema",
    "FeatureGroupSpec",
    "FeatureManifestSpec",
    "MARKET_STATE_COVERAGE_COLUMNS",
    "MARKET_STATE_AXES",
    "MARKET_STATE_AXIS_ORDER",
    "MARKET_STATE_CONTRACT",
    "MARKET_STATE_FEATURE_MANIFEST",
    "MARKET_STATE_PATHWAY",
    "MISSING_BENCHMARK_POLICIES",
    "MarketStateConfig",
    "MarketStateInputFrameContract",
    "PATHWAY_SCHEMA_VERSION",
    "PathwayAxisContract",
    "PathwayContract",
    "PathwayInputValidationResult",
    "REGIME_PATHWAYS",
    "RELATIVE_STATE_COVERAGE_COLUMNS",
    "RELATIVE_STATE_AXES",
    "RELATIVE_STATE_AXIS_ORDER",
    "RELATIVE_STATE_CONTRACT",
    "RELATIVE_STATE_FEATURE_MANIFEST",
    "RELATIVE_STATE_PATHWAY",
    "RelativeStateConfig",
    "RelativeStateInputFrameContract",
    "SourceProbeRecord",
    "artifact_metadata_schema",
    "band_metadata",
    "diagnostic_report_schema",
    "feature_manifest",
    "market_state_input_contract",
    "market_state_input_contract_template",
    "market_state_aggregation_source_probe",
    "market_state_scalar_partition_source_probe",
    "market_membership_snapshot_provenance",
    "market_membership_source_policy",
    "market_universe_lifecycle_policy",
    "market_universe_membership_snapshot",
    "market_universe_membership_snapshot_path",
    "market_state_month_dir",
    "market_state_part_path",
    "market_state_run_manifest",
    "market_state_table_dir",
    "market_aggregation_source_read_precondition",
    "market_aggregation_source_read_precondition_path",
    "market_source_coverage_diagnostic",
    "market_source_coverage_diagnostic_path",
    "load_market_universe_membership_input_file",
    "market_universe_membership_input_from_payload",
    "market_universe_membership_input_path_policy",
    "pathway_contract",
    "relative_benchmark_source_policy",
    "relative_peer_basket_lifecycle_policy",
    "relative_peer_basket_source_policy",
    "relative_source_read_precondition",
    "relative_source_read_precondition_path",
    "relative_state_input_contract",
    "relative_state_input_contract_template",
    "relative_state_month_dir",
    "relative_state_part_path",
    "relative_state_run_manifest",
    "relative_state_table_dir",
    "validate_market_state_input_frame",
    "validate_relative_state_input_frame",
]
