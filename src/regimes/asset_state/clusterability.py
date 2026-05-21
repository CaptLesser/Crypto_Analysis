from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from src.regimes.asset_state.contracts import (
    ASSET_STATE_SCHEMA_VERSION,
    AssetStateAxis,
    AssetStateBand,
    AssetStateClusterabilityStatus,
    AssetStateFallbackStatus,
    AssetStateSchemaVersion,
)
from src.regimes.asset_state.contracts import _enum_value, _schema_version
from src.regimes.asset_state.data_contracts import (
    DATASET_BUILD_STATUS_READY,
    DatasetBuildResult,
)
from src.regimes.asset_state.dataset import AssetStateDataset
from src.regimes.asset_state.taxonomy import default_asset_state_taxonomy
from src.regimes.core.serialization import to_jsonable


CLUSTERABILITY_STATUS_CLUSTERABLE_CANDIDATE = "clusterable_candidate"
CLUSTERABILITY_STATUS_VALID_FLAT_SINGLE_STATE = "valid_flat_single_state"
CLUSTERABILITY_STATUS_NEAR_FLAT_NEEDS_MORE_EVIDENCE = "near_flat_needs_more_evidence"
CLUSTERABILITY_STATUS_INSUFFICIENT_HISTORY = "insufficient_history"
CLUSTERABILITY_STATUS_INSUFFICIENT_FINITE_ROWS = "insufficient_finite_rows"
CLUSTERABILITY_STATUS_AXIS_NOT_CLUSTERABLE = "axis_not_clusterable"
CLUSTERABILITY_STATUS_FRAGMENTED_NOISE_CANDIDATE = "fragmented_noise_candidate"
CLUSTERABILITY_STATUS_UNKNOWN_ERROR = "unknown_error"

CLUSTERABILITY_STATUSES: tuple[str, ...] = (
    CLUSTERABILITY_STATUS_CLUSTERABLE_CANDIDATE,
    CLUSTERABILITY_STATUS_VALID_FLAT_SINGLE_STATE,
    CLUSTERABILITY_STATUS_NEAR_FLAT_NEEDS_MORE_EVIDENCE,
    CLUSTERABILITY_STATUS_INSUFFICIENT_HISTORY,
    CLUSTERABILITY_STATUS_INSUFFICIENT_FINITE_ROWS,
    CLUSTERABILITY_STATUS_AXIS_NOT_CLUSTERABLE,
    CLUSTERABILITY_STATUS_FRAGMENTED_NOISE_CANDIDATE,
    CLUSTERABILITY_STATUS_UNKNOWN_ERROR,
)

FALLBACK_STATUS_NO_FALLBACK_NEEDED = "no_fallback_needed"
FALLBACK_STATUS_NEUTRAL_FLAT_FALLBACK = "neutral_flat_fallback"
FALLBACK_STATUS_INSUFFICIENT_DATA_NO_LABEL = "insufficient_data_no_label"
FALLBACK_STATUS_AXIS_NOT_APPLICABLE = "axis_not_applicable"
FALLBACK_STATUS_NEEDS_MANUAL_REVIEW = "needs_manual_review"

CLUSTERABILITY_FALLBACK_STATUSES: tuple[str, ...] = (
    FALLBACK_STATUS_NO_FALLBACK_NEEDED,
    FALLBACK_STATUS_NEUTRAL_FLAT_FALLBACK,
    FALLBACK_STATUS_INSUFFICIENT_DATA_NO_LABEL,
    FALLBACK_STATUS_AXIS_NOT_APPLICABLE,
    FALLBACK_STATUS_NEEDS_MANUAL_REVIEW,
)


@dataclass(frozen=True)
class AssetStateClusterabilityPolicy:
    min_rows: int = 32
    min_finite_rows: int = 32
    max_missing_fraction: float = 0.35
    min_nonzero_variance_features: int = 2
    near_flat_fraction_threshold: float = 0.98
    min_clusterable_assets: int = 1
    schema_version: int | AssetStateSchemaVersion = ASSET_STATE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if int(self.min_rows) < 1 or int(self.min_finite_rows) < 1:
            raise ValueError("Asset-state clusterability row thresholds must be positive")
        for field_name in ("max_missing_fraction", "near_flat_fraction_threshold"):
            value = float(getattr(self, field_name))
            if value < 0.0 or value > 1.0:
                raise ValueError(f"Asset-state clusterability {field_name} must be between 0 and 1")
            object.__setattr__(self, field_name, value)
        object.__setattr__(self, "schema_version", _schema_version(self.schema_version))
        object.__setattr__(self, "min_rows", int(self.min_rows))
        object.__setattr__(self, "min_finite_rows", int(self.min_finite_rows))
        object.__setattr__(self, "min_nonzero_variance_features", int(self.min_nonzero_variance_features))
        object.__setattr__(self, "min_clusterable_assets", int(self.min_clusterable_assets))

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "min_rows": int(self.min_rows),
            "min_finite_rows": int(self.min_finite_rows),
            "max_missing_fraction": float(self.max_missing_fraction),
            "min_nonzero_variance_features": int(self.min_nonzero_variance_features),
            "near_flat_fraction_threshold": float(self.near_flat_fraction_threshold),
            "min_clusterable_assets": int(self.min_clusterable_assets),
            "fail_closed": True,
        }


@dataclass(frozen=True)
class AssetClusterabilityResult:
    asset: str
    axis: str
    band: str
    status: str
    final_label: str
    fallback_status: str
    row_count: int
    finite_row_count: int
    feature_count: int
    missing_fraction: float | None
    nonzero_variance_feature_count: int
    near_zero_movement_fraction: float | None
    reason: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int | AssetStateSchemaVersion = ASSET_STATE_SCHEMA_VERSION

    @property
    def clusterable_candidate(self) -> bool:
        return self.status == AssetStateClusterabilityStatus.CLUSTERABLE_CANDIDATE.value

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "asset": self.asset,
            "axis": self.axis,
            "band": self.band,
            "status": self.status,
            "final_label": self.final_label,
            "fallback_status": self.fallback_status,
            "clusterable_candidate": self.clusterable_candidate,
            "row_count": int(self.row_count),
            "finite_row_count": int(self.finite_row_count),
            "feature_count": int(self.feature_count),
            "missing_fraction": self.missing_fraction,
            "nonzero_variance_feature_count": int(self.nonzero_variance_feature_count),
            "near_zero_movement_fraction": self.near_zero_movement_fraction,
            "reason": self.reason,
            "metadata": to_jsonable(dict(self.metadata)),
        }


@dataclass(frozen=True)
class UniverseClusterabilityResult:
    axis: str
    band: str
    status: str
    asset_results: tuple[AssetClusterabilityResult, ...]
    policy: AssetStateClusterabilityPolicy
    schema_version: int | AssetStateSchemaVersion = ASSET_STATE_SCHEMA_VERSION

    def as_dict(self) -> dict[str, Any]:
        counts: dict[str, int] = {}
        for result in self.asset_results:
            counts[result.status] = int(counts.get(result.status, 0) + 1)
        return {
            "schema_version": int(self.schema_version),
            "artifact_kind": "asset_state_universe_clusterability_result",
            "axis": self.axis,
            "band": self.band,
            "status": self.status,
            "policy": self.policy.as_dict(),
            "summary": {
                "asset_count": int(len(self.asset_results)),
                "clusterable_asset_count": int(sum(1 for result in self.asset_results if result.clusterable_candidate)),
                "status_counts": counts,
            },
            "asset_results": [result.as_dict() for result in self.asset_results],
            "production_outputs_written": False,
        }


@dataclass(frozen=True)
class AssetStateClusterabilityThresholdProfile:
    profile_id: str
    axis: str | AssetStateAxis
    min_history_rows: int = 32
    min_finite_rows: int = 24
    max_missingness_share: float = 0.25
    low_variance_threshold: float = 1e-12
    low_variance_feature_share_threshold: float = 0.75
    flat_movement_abs_threshold: float = 1e-12
    flat_movement_share_threshold: float = 0.98
    near_flat_movement_share_threshold: float = 0.85
    duplicate_row_share_threshold: float = 0.75
    min_effective_sample_diversity: float = 0.2
    fragmentation_noise_score_threshold: float = 0.8
    min_activity_nonzero_share: float = 0.1
    valid_flat_single_state_allowed: bool = True
    schema_version: int | AssetStateSchemaVersion = ASSET_STATE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        axis_value = _enum_value(self.axis, AssetStateAxis, field_name="axis")
        if int(self.min_history_rows) < 1 or int(self.min_finite_rows) < 1:
            raise ValueError("Asset-state clusterability threshold row counts must be positive")
        for field_name in (
            "max_missingness_share",
            "low_variance_feature_share_threshold",
            "flat_movement_share_threshold",
            "near_flat_movement_share_threshold",
            "duplicate_row_share_threshold",
            "min_effective_sample_diversity",
            "fragmentation_noise_score_threshold",
            "min_activity_nonzero_share",
        ):
            value = float(getattr(self, field_name))
            if value < 0.0 or value > 1.0:
                raise ValueError(f"Asset-state clusterability threshold {field_name} must be between 0 and 1")
            object.__setattr__(self, field_name, value)
        if float(self.low_variance_threshold) < 0.0 or float(self.flat_movement_abs_threshold) < 0.0:
            raise ValueError("Asset-state clusterability variance and flat movement thresholds must be non-negative")
        object.__setattr__(self, "schema_version", _schema_version(self.schema_version))
        object.__setattr__(self, "axis", axis_value)
        object.__setattr__(self, "profile_id", str(self.profile_id).strip())
        object.__setattr__(self, "min_history_rows", int(self.min_history_rows))
        object.__setattr__(self, "min_finite_rows", int(self.min_finite_rows))
        object.__setattr__(self, "low_variance_threshold", float(self.low_variance_threshold))
        object.__setattr__(self, "flat_movement_abs_threshold", float(self.flat_movement_abs_threshold))
        object.__setattr__(self, "valid_flat_single_state_allowed", bool(self.valid_flat_single_state_allowed))

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "profile_id": self.profile_id,
            "axis": self.axis,
            "min_history_rows": int(self.min_history_rows),
            "min_finite_rows": int(self.min_finite_rows),
            "max_missingness_share": float(self.max_missingness_share),
            "low_variance_threshold": float(self.low_variance_threshold),
            "low_variance_feature_share_threshold": float(self.low_variance_feature_share_threshold),
            "flat_movement_abs_threshold": float(self.flat_movement_abs_threshold),
            "flat_movement_share_threshold": float(self.flat_movement_share_threshold),
            "near_flat_movement_share_threshold": float(self.near_flat_movement_share_threshold),
            "duplicate_row_share_threshold": float(self.duplicate_row_share_threshold),
            "min_effective_sample_diversity": float(self.min_effective_sample_diversity),
            "fragmentation_noise_score_threshold": float(self.fragmentation_noise_score_threshold),
            "min_activity_nonzero_share": float(self.min_activity_nonzero_share),
            "valid_flat_single_state_allowed": bool(self.valid_flat_single_state_allowed),
            "fail_closed": True,
        }


@dataclass(frozen=True)
class AssetStateClusterabilityDiagnostics:
    row_count: int
    finite_row_count: int
    feature_count: int
    missingness_share: float | None
    low_variance_feature_count: int
    low_variance_feature_share: float | None
    near_constant_behavior: bool
    mostly_zero_movement_share: float | None
    mostly_zero_movement: bool
    duplicate_row_share: float | None
    duplicate_heavy_rows: bool
    effective_sample_diversity: float | None
    fragmentation_noise_score: float | None
    fragmentation_noise_risk: bool
    activity_nonzero_share: float | None
    activity_finite_row_count: int | None
    activity_sufficient: bool | None
    movement_columns: tuple[str, ...] = ()
    activity_columns: tuple[str, ...] = ()
    variance_summary: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "row_count": int(self.row_count),
            "finite_row_count": int(self.finite_row_count),
            "feature_count": int(self.feature_count),
            "missingness_share": self.missingness_share,
            "low_variance_feature_count": int(self.low_variance_feature_count),
            "low_variance_feature_share": self.low_variance_feature_share,
            "near_constant_behavior": bool(self.near_constant_behavior),
            "mostly_zero_movement_share": self.mostly_zero_movement_share,
            "mostly_zero_movement": bool(self.mostly_zero_movement),
            "duplicate_row_share": self.duplicate_row_share,
            "duplicate_heavy_rows": bool(self.duplicate_heavy_rows),
            "effective_sample_diversity": self.effective_sample_diversity,
            "fragmentation_noise_score": self.fragmentation_noise_score,
            "fragmentation_noise_risk": bool(self.fragmentation_noise_risk),
            "activity_nonzero_share": self.activity_nonzero_share,
            "activity_finite_row_count": self.activity_finite_row_count,
            "activity_sufficient": self.activity_sufficient,
            "movement_columns": list(self.movement_columns),
            "activity_columns": list(self.activity_columns),
            "variance_summary": to_jsonable(dict(self.variance_summary)),
        }


@dataclass(frozen=True)
class AssetStateClusterabilityAssessment:
    asset: str
    axis: str
    band: str
    status: str
    fallback_status: str
    diagnostics: AssetStateClusterabilityDiagnostics
    threshold_profile: AssetStateClusterabilityThresholdProfile
    source_data_lineage: tuple[Mapping[str, Any], ...] = ()
    reason: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int | AssetStateSchemaVersion = ASSET_STATE_SCHEMA_VERSION

    @property
    def clusterable_candidate(self) -> bool:
        return self.status == CLUSTERABILITY_STATUS_CLUSTERABLE_CANDIDATE

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "asset": self.asset,
            "axis": self.axis,
            "band": self.band,
            "status": self.status,
            "fallback_status": self.fallback_status,
            "clusterable_candidate": self.clusterable_candidate,
            "reason": self.reason,
            "diagnostics": self.diagnostics.as_dict(),
            "source_data_lineage": [to_jsonable(dict(item)) for item in self.source_data_lineage],
            "threshold_profile": self.threshold_profile.as_dict(),
            "metadata": to_jsonable(dict(self.metadata)),
        }


@dataclass(frozen=True)
class AssetStateUniverseClusterabilityManifest:
    assessments: tuple[AssetStateClusterabilityAssessment, ...]
    run_config: Mapping[str, Any]
    artifact_kind: str = "asset_state_universe_clusterability_manifest"
    schema_version: int | AssetStateSchemaVersion = ASSET_STATE_SCHEMA_VERSION

    def as_dict(self) -> dict[str, Any]:
        status_counts: dict[str, int] = {}
        fallback_counts: dict[str, int] = {}
        for assessment in self.assessments:
            status_counts[assessment.status] = int(status_counts.get(assessment.status, 0) + 1)
            fallback_counts[assessment.fallback_status] = int(fallback_counts.get(assessment.fallback_status, 0) + 1)
        return {
            "schema_version": int(self.schema_version),
            "artifact_kind": self.artifact_kind,
            "summary": {
                "assessment_count": len(self.assessments),
                "asset_count": len({item.asset for item in self.assessments}),
                "clusterable_candidate_count": sum(1 for item in self.assessments if item.clusterable_candidate),
                "status_counts": status_counts,
                "fallback_status_counts": fallback_counts,
                "production_fallback_labels_written": False,
                "production_outputs_written": False,
            },
            "run_config": to_jsonable(dict(self.run_config)),
            "assessments": [assessment.as_dict() for assessment in self.assessments],
        }


def default_clusterability_threshold_profiles() -> dict[str, AssetStateClusterabilityThresholdProfile]:
    taxonomy = default_asset_state_taxonomy()
    return {
        AssetStateAxis.TREND.value: AssetStateClusterabilityThresholdProfile(
            profile_id="asset_state_trend_clusterability_v1",
            axis=AssetStateAxis.TREND,
            valid_flat_single_state_allowed=taxonomy.axis_spec(AssetStateAxis.TREND).allow_single_state_output,
        ),
        AssetStateAxis.VOLATILITY.value: AssetStateClusterabilityThresholdProfile(
            profile_id="asset_state_volatility_clusterability_v1",
            axis=AssetStateAxis.VOLATILITY,
            flat_movement_abs_threshold=1e-10,
            flat_movement_share_threshold=0.97,
            near_flat_movement_share_threshold=0.82,
            fragmentation_noise_score_threshold=0.9,
            valid_flat_single_state_allowed=taxonomy.axis_spec(AssetStateAxis.VOLATILITY).allow_single_state_output,
        ),
        AssetStateAxis.ACTIVITY.value: AssetStateClusterabilityThresholdProfile(
            profile_id="asset_state_activity_clusterability_v1",
            axis=AssetStateAxis.ACTIVITY,
            min_activity_nonzero_share=0.2,
            valid_flat_single_state_allowed=taxonomy.axis_spec(AssetStateAxis.ACTIVITY).allow_single_state_output,
        ),
        AssetStateAxis.MEAN_REVERSION.value: AssetStateClusterabilityThresholdProfile(
            profile_id="asset_state_mean_reversion_clusterability_v1",
            axis=AssetStateAxis.MEAN_REVERSION,
            near_flat_movement_share_threshold=0.8,
            valid_flat_single_state_allowed=taxonomy.axis_spec(AssetStateAxis.MEAN_REVERSION).allow_single_state_output,
        ),
        AssetStateAxis.DRAWDOWN.value: AssetStateClusterabilityThresholdProfile(
            profile_id="asset_state_drawdown_clusterability_v1",
            axis=AssetStateAxis.DRAWDOWN,
            near_flat_movement_share_threshold=0.8,
            valid_flat_single_state_allowed=taxonomy.axis_spec(AssetStateAxis.DRAWDOWN).allow_single_state_output,
        ),
        AssetStateAxis.RANGE_EFFICIENCY.value: AssetStateClusterabilityThresholdProfile(
            profile_id="asset_state_range_efficiency_clusterability_v1",
            axis=AssetStateAxis.RANGE_EFFICIENCY,
            valid_flat_single_state_allowed=taxonomy.axis_spec(AssetStateAxis.RANGE_EFFICIENCY).allow_single_state_output,
        ),
    }


def threshold_profile_for_axis(
    axis: str | AssetStateAxis,
    *,
    overrides: Mapping[str, Any] | None = None,
) -> AssetStateClusterabilityThresholdProfile:
    axis_value = _enum_value(axis, AssetStateAxis, field_name="axis")
    base = default_clusterability_threshold_profiles()[axis_value]
    if not overrides:
        return base
    payload = {**base.as_dict(), **dict(overrides)}
    payload.pop("schema_version", None)
    payload.pop("fail_closed", None)
    return AssetStateClusterabilityThresholdProfile(**payload)


def _movement_columns(feature_columns: Sequence[str]) -> tuple[str, ...]:
    markers = (
        "_log_return",
        "_roc_",
        "_mom_",
        "_d_close_",
        "_range_efficiency",
        "_range_hl",
        "_range_co",
        "_true_range",
        "_drawdown",
        "_runup",
    )
    return tuple(column for column in feature_columns if any(marker in str(column) for marker in markers))


def _activity_columns(feature_columns: Sequence[str]) -> tuple[str, ...]:
    markers = (
        "_volume",
        "_trades",
        "_trade_intensity",
        "_avg_trade_size",
        "_vroc_",
        "_prr",
        "_obv",
        "_adl",
        "_force_index",
        "_vpt",
        "_eom_",
    )
    return tuple(column for column in feature_columns if any(marker in str(column) for marker in markers))


def _safe_ratio(numerator: int | float, denominator: int | float) -> float | None:
    denominator = float(denominator)
    if denominator <= 0:
        return None
    return float(numerator) / denominator


def _rough_fragmentation_noise_score(frame: pd.DataFrame, movement_columns: Sequence[str]) -> float | None:
    if frame.empty:
        return None
    if movement_columns:
        selected = [column for column in movement_columns if column in frame.columns]
        if not selected:
            return None
        signal = frame[selected].mean(axis=1)
    else:
        signal = frame.iloc[:, 0]
    signal = pd.to_numeric(signal, errors="coerce").dropna()
    if len(signal) < 3:
        return None
    signs = np.sign(signal.to_numpy(dtype=float))
    signs = signs[signs != 0]
    if len(signs) < 3:
        return 0.0
    return float(np.mean(signs[1:] != signs[:-1]))


def _clusterability_diagnostics(
    dataset: DatasetBuildResult,
    *,
    profile: AssetStateClusterabilityThresholdProfile,
) -> AssetStateClusterabilityDiagnostics:
    train_frame = dataset.train.frame
    candidate_columns = tuple(str(column) for column in dataset.feature_columns if str(column) in train_frame.columns)
    matrix_columns = tuple(str(column) for column in dataset.output_feature_names)
    if (
        dataset.X_train.ndim == 2
        and dataset.X_train.shape[0] > 0
        and dataset.X_train.shape[1] > 0
        and len(matrix_columns) == dataset.X_train.shape[1]
    ):
        analysis_frame = pd.DataFrame(dataset.X_train, columns=matrix_columns)
    elif candidate_columns:
        analysis_frame = train_frame[list(candidate_columns)].copy()
    else:
        analysis_frame = pd.DataFrame()
    row_count = int(dataset.row_counts.get("train_before_cleaning", dataset.train.row_count_before_cleaning or len(train_frame)))
    feature_count = int(analysis_frame.shape[1])
    if analysis_frame.empty or feature_count == 0:
        return AssetStateClusterabilityDiagnostics(
            row_count=row_count,
            finite_row_count=0,
            feature_count=feature_count,
            missingness_share=1.0,
            low_variance_feature_count=0,
            low_variance_feature_share=None,
            near_constant_behavior=False,
            mostly_zero_movement_share=None,
            mostly_zero_movement=False,
            duplicate_row_share=None,
            duplicate_heavy_rows=False,
            effective_sample_diversity=None,
            fragmentation_noise_score=None,
            fragmentation_noise_risk=False,
            activity_nonzero_share=None,
            activity_finite_row_count=None,
            activity_sufficient=None,
        )

    numeric = analysis_frame.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    missingness_share = float(numeric.isna().to_numpy().mean()) if len(numeric) else 1.0
    finite = numeric.dropna(axis=0, how="any")
    finite_row_count = int(len(finite))
    variances = finite.var(ddof=0) if finite_row_count else pd.Series(dtype=float)
    low_variance = variances <= float(profile.low_variance_threshold) if not variances.empty else pd.Series(dtype=bool)
    low_variance_count = int(low_variance.sum()) if not low_variance.empty else 0
    low_variance_share = _safe_ratio(low_variance_count, feature_count)
    near_constant_behavior = bool(
        low_variance_share is not None
        and low_variance_share >= float(profile.low_variance_feature_share_threshold)
    )

    rounded = finite.round(12) if not finite.empty else finite
    unique_count = int(len(rounded.drop_duplicates())) if not rounded.empty else 0
    duplicate_row_share = None if finite_row_count == 0 else float(1.0 - unique_count / finite_row_count)
    duplicate_heavy_rows = bool(
        duplicate_row_share is not None
        and duplicate_row_share >= float(profile.duplicate_row_share_threshold)
    )
    effective_sample_diversity = _safe_ratio(unique_count, finite_row_count)

    raw_numeric = (
        train_frame[list(candidate_columns)].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
        if candidate_columns
        else numeric
    )
    movement_columns = _movement_columns(tuple(candidate_columns) or tuple(str(column) for column in numeric.columns))
    selected_movement = [column for column in movement_columns if column in raw_numeric.columns]
    mostly_zero_movement_share: float | None = None
    mostly_zero_movement = False
    movement_frame = raw_numeric[selected_movement].dropna(axis=0, how="any") if selected_movement else pd.DataFrame()
    if selected_movement and not movement_frame.empty:
        movement_values = movement_frame.abs()
        mostly_zero_movement_share = float((movement_values <= float(profile.flat_movement_abs_threshold)).all(axis=1).mean())
        mostly_zero_movement = bool(mostly_zero_movement_share >= float(profile.flat_movement_share_threshold))
    elif near_constant_behavior and finite_row_count:
        mostly_zero_movement_share = 1.0
        mostly_zero_movement = True

    fragmentation_noise_score = _rough_fragmentation_noise_score(movement_frame, selected_movement)
    fragmentation_noise_risk = bool(
        fragmentation_noise_score is not None
        and fragmentation_noise_score >= float(profile.fragmentation_noise_score_threshold)
    )

    activity_columns = _activity_columns(tuple(str(column) for column in numeric.columns))
    selected_activity = [column for column in activity_columns if column in numeric.columns]
    activity_nonzero_share: float | None = None
    activity_finite_row_count: int | None = None
    activity_sufficient: bool | None = None
    if selected_activity:
        activity_numeric = numeric[selected_activity].dropna(axis=0, how="any")
        activity_finite_row_count = int(len(activity_numeric))
        if activity_finite_row_count:
            activity_nonzero_share = float((activity_numeric.abs() > 0.0).any(axis=1).mean())
            activity_sufficient = bool(activity_nonzero_share >= float(profile.min_activity_nonzero_share))
        else:
            activity_nonzero_share = 0.0
            activity_sufficient = False

    return AssetStateClusterabilityDiagnostics(
        row_count=row_count,
        finite_row_count=finite_row_count,
        feature_count=feature_count,
        missingness_share=missingness_share,
        low_variance_feature_count=low_variance_count,
        low_variance_feature_share=low_variance_share,
        near_constant_behavior=near_constant_behavior,
        mostly_zero_movement_share=mostly_zero_movement_share,
        mostly_zero_movement=mostly_zero_movement,
        duplicate_row_share=duplicate_row_share,
        duplicate_heavy_rows=duplicate_heavy_rows,
        effective_sample_diversity=effective_sample_diversity,
        fragmentation_noise_score=fragmentation_noise_score,
        fragmentation_noise_risk=fragmentation_noise_risk,
        activity_nonzero_share=activity_nonzero_share,
        activity_finite_row_count=activity_finite_row_count,
        activity_sufficient=activity_sufficient,
        movement_columns=tuple(selected_movement),
        activity_columns=tuple(selected_activity),
        variance_summary={str(key): float(value) for key, value in variances.items()} if not variances.empty else {},
    )


def _lineage_from_dataset(dataset: DatasetBuildResult) -> tuple[Mapping[str, Any], ...]:
    return tuple(item.as_dict() for item in dataset.partition_lineage)


def _blocked_dataset_assessment(
    dataset: DatasetBuildResult,
    *,
    profile: AssetStateClusterabilityThresholdProfile,
) -> AssetStateClusterabilityAssessment:
    reason_codes = tuple(dataset.reason_codes)
    status = CLUSTERABILITY_STATUS_INSUFFICIENT_HISTORY
    if any("finite" in code or "preprocessing" in code for code in reason_codes):
        status = CLUSTERABILITY_STATUS_INSUFFICIENT_FINITE_ROWS
    diagnostics = AssetStateClusterabilityDiagnostics(
        row_count=int(dataset.row_counts.get("raw", 0)),
        finite_row_count=0,
        feature_count=0,
        missingness_share=1.0,
        low_variance_feature_count=0,
        low_variance_feature_share=None,
        near_constant_behavior=False,
        mostly_zero_movement_share=None,
        mostly_zero_movement=False,
        duplicate_row_share=None,
        duplicate_heavy_rows=False,
        effective_sample_diversity=None,
        fragmentation_noise_score=None,
        fragmentation_noise_risk=False,
        activity_nonzero_share=None,
        activity_finite_row_count=None,
        activity_sufficient=None,
    )
    return AssetStateClusterabilityAssessment(
        schema_version=ASSET_STATE_SCHEMA_VERSION,
        asset=dataset.asset,
        axis=dataset.axis,
        band=dataset.band,
        status=status,
        fallback_status=FALLBACK_STATUS_INSUFFICIENT_DATA_NO_LABEL,
        diagnostics=diagnostics,
        threshold_profile=profile,
        source_data_lineage=_lineage_from_dataset(dataset),
        reason=f"dataset builder blocked: {', '.join(reason_codes) or dataset.message or 'unknown'}",
        metadata={"dataset_status": dataset.status, "dataset_reason_codes": list(reason_codes)},
    )


def evaluate_dataset_clusterability(
    dataset: DatasetBuildResult,
    *,
    threshold_profile: AssetStateClusterabilityThresholdProfile | None = None,
    threshold_overrides: Mapping[str, Any] | None = None,
) -> AssetStateClusterabilityAssessment:
    profile = threshold_profile or threshold_profile_for_axis(dataset.axis, overrides=threshold_overrides)
    try:
        if dataset.status != DATASET_BUILD_STATUS_READY:
            return _blocked_dataset_assessment(dataset, profile=profile)
        axis_spec = default_asset_state_taxonomy().axis_spec(dataset.axis)
        diagnostics = _clusterability_diagnostics(dataset, profile=profile)
        status = CLUSTERABILITY_STATUS_CLUSTERABLE_CANDIDATE
        fallback = FALLBACK_STATUS_NO_FALLBACK_NEEDED
        reason: str | None = None

        if diagnostics.row_count < int(profile.min_history_rows):
            status = CLUSTERABILITY_STATUS_INSUFFICIENT_HISTORY
            fallback = FALLBACK_STATUS_INSUFFICIENT_DATA_NO_LABEL
            reason = f"row_count {diagnostics.row_count} < min_history_rows {profile.min_history_rows}"
        elif (
            diagnostics.finite_row_count < int(profile.min_finite_rows)
            or (
                diagnostics.missingness_share is not None
                and diagnostics.missingness_share > float(profile.max_missingness_share)
            )
        ):
            status = CLUSTERABILITY_STATUS_INSUFFICIENT_FINITE_ROWS
            fallback = FALLBACK_STATUS_INSUFFICIENT_DATA_NO_LABEL
            reason = "finite row count or missingness failed threshold profile"
        elif dataset.axis == AssetStateAxis.ACTIVITY.value and diagnostics.activity_sufficient is False:
            status = CLUSTERABILITY_STATUS_AXIS_NOT_CLUSTERABLE
            fallback = FALLBACK_STATUS_AXIS_NOT_APPLICABLE
            reason = "activity/volume columns indicate insufficient participation"
        elif diagnostics.mostly_zero_movement:
            if axis_spec.allow_single_state_output and profile.valid_flat_single_state_allowed:
                status = CLUSTERABILITY_STATUS_VALID_FLAT_SINGLE_STATE
                fallback = FALLBACK_STATUS_NEUTRAL_FLAT_FALLBACK
                reason = "valid flat single-state fallback allowed by axis policy"
            else:
                status = CLUSTERABILITY_STATUS_AXIS_NOT_CLUSTERABLE
                fallback = FALLBACK_STATUS_AXIS_NOT_APPLICABLE
                reason = "flat single-state fallback is not allowed for this axis"
        elif (
            diagnostics.mostly_zero_movement_share is not None
            and diagnostics.mostly_zero_movement_share >= float(profile.near_flat_movement_share_threshold)
        ):
            status = CLUSTERABILITY_STATUS_NEAR_FLAT_NEEDS_MORE_EVIDENCE
            fallback = FALLBACK_STATUS_NEEDS_MANUAL_REVIEW
            reason = "near-flat movement share requires more evidence before clustering"
        elif diagnostics.near_constant_behavior:
            status = CLUSTERABILITY_STATUS_AXIS_NOT_CLUSTERABLE
            fallback = FALLBACK_STATUS_AXIS_NOT_APPLICABLE
            reason = "too many near-constant features for clusterable axis behavior"
        elif (
            diagnostics.effective_sample_diversity is not None
            and diagnostics.effective_sample_diversity < float(profile.min_effective_sample_diversity)
        ):
            status = CLUSTERABILITY_STATUS_AXIS_NOT_CLUSTERABLE
            fallback = FALLBACK_STATUS_AXIS_NOT_APPLICABLE
            reason = "effective sample diversity is below threshold"
        elif diagnostics.duplicate_heavy_rows or diagnostics.fragmentation_noise_risk:
            status = CLUSTERABILITY_STATUS_FRAGMENTED_NOISE_CANDIDATE
            fallback = FALLBACK_STATUS_NEEDS_MANUAL_REVIEW
            reason = "duplicate-heavy rows or fragmented noise risk requires review"

        return AssetStateClusterabilityAssessment(
            schema_version=ASSET_STATE_SCHEMA_VERSION,
            asset=dataset.asset,
            axis=dataset.axis,
            band=dataset.band,
            status=status,
            fallback_status=fallback,
            diagnostics=diagnostics,
            threshold_profile=profile,
            source_data_lineage=_lineage_from_dataset(dataset),
            reason=reason,
            metadata={
                "dataset_row_counts": dict(dataset.row_counts),
                "dataset_feature_count": dataset.feature_count,
                "production_fallback_label_written": False,
            },
        )
    except Exception as exc:
        diagnostics = AssetStateClusterabilityDiagnostics(
            row_count=int(dataset.row_counts.get("raw", 0)),
            finite_row_count=0,
            feature_count=0,
            missingness_share=None,
            low_variance_feature_count=0,
            low_variance_feature_share=None,
            near_constant_behavior=False,
            mostly_zero_movement_share=None,
            mostly_zero_movement=False,
            duplicate_row_share=None,
            duplicate_heavy_rows=False,
            effective_sample_diversity=None,
            fragmentation_noise_score=None,
            fragmentation_noise_risk=False,
            activity_nonzero_share=None,
            activity_finite_row_count=None,
            activity_sufficient=None,
        )
        return AssetStateClusterabilityAssessment(
            schema_version=ASSET_STATE_SCHEMA_VERSION,
            asset=dataset.asset,
            axis=dataset.axis,
            band=dataset.band,
            status=CLUSTERABILITY_STATUS_UNKNOWN_ERROR,
            fallback_status=FALLBACK_STATUS_NEEDS_MANUAL_REVIEW,
            diagnostics=diagnostics,
            threshold_profile=profile,
            source_data_lineage=_lineage_from_dataset(dataset),
            reason=f"clusterability evaluation failed closed: {exc}",
        )


def evaluate_asset_clusterability(
    frame: pd.DataFrame,
    *,
    asset: str,
    axis: str | AssetStateAxis,
    band: str | AssetStateBand,
    feature_columns: Sequence[str],
    policy: AssetStateClusterabilityPolicy | None = None,
) -> AssetClusterabilityResult:
    cfg = policy or AssetStateClusterabilityPolicy()
    axis_value = _enum_value(axis, AssetStateAxis, field_name="axis")
    band_value = _enum_value(band, AssetStateBand, field_name="band")
    axis_spec = default_asset_state_taxonomy().axis_spec(axis_value)
    asset_frame = frame.loc[frame["asset"].astype(str) == str(asset)].copy() if "asset" in frame.columns else frame.copy()
    columns = tuple(str(column) for column in feature_columns if str(column) in asset_frame.columns)
    row_count = int(len(asset_frame))
    if not columns:
        return AssetClusterabilityResult(
            schema_version=ASSET_STATE_SCHEMA_VERSION,
            asset=str(asset),
            axis=axis_value,
            band=band_value,
            status=AssetStateClusterabilityStatus.MISSING_FEATURES.value,
            final_label="missing_features",
            fallback_status=AssetStateFallbackStatus.MISSING_DATA.value,
            row_count=row_count,
            finite_row_count=0,
            feature_count=0,
            missing_fraction=1.0,
            nonzero_variance_feature_count=0,
            near_zero_movement_fraction=None,
            reason="no feature columns available",
        )
    numeric = asset_frame[list(columns)].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    missing_fraction = float(numeric.isna().to_numpy().mean()) if row_count else 1.0
    finite_mask = numeric.notna().all(axis=1)
    finite = numeric.loc[finite_mask]
    finite_row_count = int(len(finite))
    variances = finite.var(ddof=0) if finite_row_count else pd.Series(dtype=float)
    nonzero = int((variances > 1e-12).sum()) if not variances.empty else 0
    movement = _movement_columns(columns)
    near_zero_movement_fraction: float | None = None
    if movement and finite_row_count:
        movement_values = finite[[column for column in movement if column in finite.columns]].abs()
        if not movement_values.empty:
            near_zero_movement_fraction = float((movement_values <= 1e-12).all(axis=1).mean())
    if row_count < cfg.min_rows:
        status = AssetStateClusterabilityStatus.INSUFFICIENT_DATA.value
        final = "insufficient_data"
        fallback = AssetStateFallbackStatus.MISSING_DATA.value
        reason = f"row_count {row_count} < min_rows {cfg.min_rows}"
    elif missing_fraction > cfg.max_missing_fraction or finite_row_count < cfg.min_finite_rows:
        status = AssetStateClusterabilityStatus.INSUFFICIENT_DATA.value
        final = "insufficient_data"
        fallback = AssetStateFallbackStatus.MISSING_DATA.value
        reason = "missing or nonfinite rows exceed clusterability policy"
    elif near_zero_movement_fraction is not None and near_zero_movement_fraction >= cfg.near_flat_fraction_threshold:
        if axis_spec.allow_single_state_output:
            status = AssetStateClusterabilityStatus.SINGLE_STATE_FALLBACK.value
            final = "neutral_single_state_fallback"
            fallback = AssetStateFallbackStatus.APPLIED_NEUTRAL_SINGLE_STATE.value
            reason = "valid near-zero movement single-state fallback"
        else:
            status = AssetStateClusterabilityStatus.NOT_CLUSTERABLE.value
            final = "axis_not_clusterable"
            fallback = AssetStateFallbackStatus.BLOCKED.value
            reason = "single-state fallback is not allowed for this axis"
    elif nonzero < cfg.min_nonzero_variance_features:
        status = AssetStateClusterabilityStatus.NOT_CLUSTERABLE.value
        final = "axis_not_clusterable"
        fallback = AssetStateFallbackStatus.BLOCKED.value
        reason = "insufficient nonzero-variance features"
    else:
        status = AssetStateClusterabilityStatus.CLUSTERABLE_CANDIDATE.value
        final = "clusterable"
        fallback = AssetStateFallbackStatus.ALLOWED_NOT_USED.value if axis_spec.allow_single_state_output else AssetStateFallbackStatus.NOT_REQUIRED.value
        reason = None
    return AssetClusterabilityResult(
        schema_version=ASSET_STATE_SCHEMA_VERSION,
        asset=str(asset),
        axis=axis_value,
        band=band_value,
        status=status,
        final_label=final,
        fallback_status=fallback,
        row_count=row_count,
        finite_row_count=finite_row_count,
        feature_count=int(len(columns)),
        missing_fraction=missing_fraction,
        nonzero_variance_feature_count=nonzero,
        near_zero_movement_fraction=near_zero_movement_fraction,
        reason=reason,
        metadata={"variance_summary": {str(k): float(v) for k, v in variances.items()} if not variances.empty else {}},
    )


def evaluate_universe_clusterability(
    dataset: AssetStateDataset,
    *,
    axis: str | AssetStateAxis,
    band: str | AssetStateBand,
    assets: Sequence[str] | None = None,
    policy: AssetStateClusterabilityPolicy | None = None,
) -> UniverseClusterabilityResult:
    cfg = policy or AssetStateClusterabilityPolicy()
    axis_value = _enum_value(axis, AssetStateAxis, field_name="axis")
    band_value = _enum_value(band, AssetStateBand, field_name="band")
    resolved_assets = tuple(str(asset) for asset in (assets or dataset.metadata.get("assets", ())))
    if not resolved_assets:
        resolved_assets = tuple(sorted(str(value) for value in dataset.frame["asset"].dropna().astype(str).unique()))
    asset_results = tuple(
        evaluate_asset_clusterability(
            dataset.frame,
            asset=asset,
            axis=axis_value,
            band=band_value,
            feature_columns=dataset.feature_columns,
            policy=cfg,
        )
        for asset in resolved_assets
    )
    clusterable_count = int(sum(1 for result in asset_results if result.clusterable_candidate))
    fallback_count = int(sum(1 for result in asset_results if result.status == AssetStateClusterabilityStatus.SINGLE_STATE_FALLBACK.value))
    if clusterable_count >= int(cfg.min_clusterable_assets):
        status = AssetStateClusterabilityStatus.CLUSTERABLE_CANDIDATE.value
    elif fallback_count and fallback_count == len(asset_results):
        status = AssetStateClusterabilityStatus.SINGLE_STATE_FALLBACK.value
    elif any(result.status == AssetStateClusterabilityStatus.MISSING_FEATURES.value for result in asset_results):
        status = AssetStateClusterabilityStatus.MISSING_FEATURES.value
    elif any(result.status == AssetStateClusterabilityStatus.INSUFFICIENT_DATA.value for result in asset_results):
        status = AssetStateClusterabilityStatus.INSUFFICIENT_DATA.value
    else:
        status = AssetStateClusterabilityStatus.NOT_CLUSTERABLE.value
    return UniverseClusterabilityResult(
        schema_version=ASSET_STATE_SCHEMA_VERSION,
        axis=axis_value,
        band=band_value,
        status=status,
        asset_results=asset_results,
        policy=cfg,
    )


__all__ = [
    "CLUSTERABILITY_FALLBACK_STATUSES",
    "CLUSTERABILITY_STATUSES",
    "CLUSTERABILITY_STATUS_AXIS_NOT_CLUSTERABLE",
    "CLUSTERABILITY_STATUS_CLUSTERABLE_CANDIDATE",
    "CLUSTERABILITY_STATUS_FRAGMENTED_NOISE_CANDIDATE",
    "CLUSTERABILITY_STATUS_INSUFFICIENT_FINITE_ROWS",
    "CLUSTERABILITY_STATUS_INSUFFICIENT_HISTORY",
    "CLUSTERABILITY_STATUS_NEAR_FLAT_NEEDS_MORE_EVIDENCE",
    "CLUSTERABILITY_STATUS_UNKNOWN_ERROR",
    "CLUSTERABILITY_STATUS_VALID_FLAT_SINGLE_STATE",
    "FALLBACK_STATUS_AXIS_NOT_APPLICABLE",
    "FALLBACK_STATUS_INSUFFICIENT_DATA_NO_LABEL",
    "FALLBACK_STATUS_NEEDS_MANUAL_REVIEW",
    "FALLBACK_STATUS_NEUTRAL_FLAT_FALLBACK",
    "FALLBACK_STATUS_NO_FALLBACK_NEEDED",
    "AssetClusterabilityResult",
    "AssetStateClusterabilityAssessment",
    "AssetStateClusterabilityDiagnostics",
    "AssetStateClusterabilityPolicy",
    "AssetStateClusterabilityThresholdProfile",
    "AssetStateUniverseClusterabilityManifest",
    "UniverseClusterabilityResult",
    "default_clusterability_threshold_profiles",
    "evaluate_asset_clusterability",
    "evaluate_dataset_clusterability",
    "evaluate_universe_clusterability",
    "threshold_profile_for_axis",
]
