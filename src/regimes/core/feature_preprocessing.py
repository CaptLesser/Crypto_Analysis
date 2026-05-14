from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from src.regimes.core.foundation_contracts import (
    REGIME_LAYER_AXES,
    REGIME_LAYERS,
    REGIME_STUDY_BANDS,
    MissingnessPolicy,
    SourceArtifactLineage,
)

try:
    from sklearn.cluster import FeatureAgglomeration
    from sklearn.decomposition import FactorAnalysis, PCA
    from sklearn.preprocessing import PowerTransformer, QuantileTransformer, RobustScaler, StandardScaler

    _HAS_SKLEARN = True
except Exception:  # pragma: no cover - exercised only in minimal dependency environments
    FeatureAgglomeration = None  # type: ignore[assignment]
    FactorAnalysis = None  # type: ignore[assignment]
    PCA = None  # type: ignore[assignment]
    PowerTransformer = None  # type: ignore[assignment]
    QuantileTransformer = None  # type: ignore[assignment]
    RobustScaler = None  # type: ignore[assignment]
    StandardScaler = None  # type: ignore[assignment]
    _HAS_SKLEARN = False


REGIME_FEATURE_PREPROCESSING_SCHEMA_VERSION = 1
FEATURE_POOL_IMPLEMENTATION_STATUSES: tuple[str, ...] = ("implemented", "declared", "scaffold")
FEATURE_LEAKAGE_POLICIES: tuple[str, ...] = (
    "train_window_only",
    "no_forward_target_columns",
    "source_features_only",
    "diagnostics_only",
    "declared_not_implemented",
)
PREPROCESS_FIT_SCOPE_TRAIN_ONLY = "train_only"

_ALLOWED_FEATURE_AXES = tuple(
    dict.fromkeys(
        (
            "trend",
            "vol",
            "activity",
            "market",
            "relative",
            *(axis for axes in REGIME_LAYER_AXES.values() for axis in axes),
        )
    )
)


def _safe_float(value: object) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def _jsonable(value: Any) -> Any:
    if hasattr(value, "as_dict"):
        return value.as_dict()
    if hasattr(value, "to_metadata"):
        return value.to_metadata()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(val) for key, val in value.items()}
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, set):
        return sorted(_jsonable(item) for item in value)
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return _safe_float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.datetime64):
        return pd.Timestamp(value).isoformat()
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    if isinstance(value, BaseException):
        return str(value)
    if value is pd.NA:
        return None
    if isinstance(value, float):
        return _safe_float(value)
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    return str(value)


def _normalize_token(value: object, *, field_name: str) -> str:
    text = str(value).strip().lower()
    if not text:
        raise ValueError(f"Regime {field_name} must be non-empty")
    return text


def _require_members(values: Sequence[object], allowed: Sequence[str], *, field_name: str) -> tuple[str, ...]:
    normalized = tuple(_normalize_token(value, field_name=field_name) for value in values)
    if not normalized:
        raise ValueError(f"Regime {field_name} must include at least one value")
    invalid = [value for value in normalized if value not in allowed]
    if invalid:
        valid = ", ".join(str(item) for item in allowed)
        raise ValueError(f"Unsupported Regime {field_name} {invalid[0]!r}; expected one of: {valid}")
    return tuple(dict.fromkeys(normalized))


@dataclass(frozen=True)
class FeaturePoolContract:
    feature_family_name: str
    source_columns: tuple[str, ...]
    layer_compatibility: tuple[str, ...]
    axis_compatibility: tuple[str, ...]
    band_compatibility: tuple[str, ...]
    required_input_granularity: str
    missingness_policy: MissingnessPolicy
    leakage_policy: str
    source_lineage: tuple[SourceArtifactLineage, ...]
    implementation_status: str = "declared"
    schema_version: int = REGIME_FEATURE_PREPROCESSING_SCHEMA_VERSION

    def __post_init__(self) -> None:
        family = _normalize_token(self.feature_family_name, field_name="feature family name")
        if not self.source_columns:
            raise ValueError("Regime feature pool source_columns must be non-empty")
        layers = _require_members(self.layer_compatibility, REGIME_LAYERS, field_name="layer compatibility")
        axes = _require_members(self.axis_compatibility, _ALLOWED_FEATURE_AXES, field_name="axis compatibility")
        bands = _require_members(self.band_compatibility, REGIME_STUDY_BANDS, field_name="band compatibility")
        granularity = str(self.required_input_granularity).strip()
        if not granularity:
            raise ValueError("Regime required_input_granularity must be non-empty")
        leakage_policy = _require_members((self.leakage_policy,), FEATURE_LEAKAGE_POLICIES, field_name="leakage policy")[0]
        status = _require_members(
            (self.implementation_status,),
            FEATURE_POOL_IMPLEMENTATION_STATUSES,
            field_name="feature pool implementation status",
        )[0]
        if not self.source_lineage:
            raise ValueError("Regime feature pool source_lineage must be non-empty")
        object.__setattr__(self, "feature_family_name", family)
        object.__setattr__(self, "source_columns", tuple(str(column) for column in self.source_columns))
        object.__setattr__(self, "layer_compatibility", layers)
        object.__setattr__(self, "axis_compatibility", axes)
        object.__setattr__(self, "band_compatibility", bands)
        object.__setattr__(self, "required_input_granularity", granularity)
        object.__setattr__(self, "leakage_policy", leakage_policy)
        object.__setattr__(self, "source_lineage", tuple(self.source_lineage))
        object.__setattr__(self, "implementation_status", status)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "feature_family_name": self.feature_family_name,
            "source_columns": list(self.source_columns),
            "layer_compatibility": list(self.layer_compatibility),
            "axis_compatibility": list(self.axis_compatibility),
            "band_compatibility": list(self.band_compatibility),
            "required_input_granularity": self.required_input_granularity,
            "missingness_policy": self.missingness_policy.as_dict(),
            "leakage_policy": self.leakage_policy,
            "source_lineage": [lineage.as_dict() for lineage in self.source_lineage],
            "implementation_status": self.implementation_status,
        }

    def to_json(self, **kwargs: Any) -> str:
        return json.dumps(self.as_dict(), **kwargs)


def _missingness_policy() -> MissingnessPolicy:
    return MissingnessPolicy(
        policy="train_window_feature_filter_then_drop_nonfinite_rows",
        max_null_fraction=0.2,
        fail_closed=True,
    )


def _source_lineage(path: str, produced_by: str) -> tuple[SourceArtifactLineage, ...]:
    return (
        SourceArtifactLineage(
            artifact_kind="regime_feature_pool_definition",
            artifact_path=path,
            schema_version=REGIME_FEATURE_PREPROCESSING_SCHEMA_VERSION,
            produced_by=produced_by,
        ),
    )


def default_feature_pool_registry() -> dict[str, FeaturePoolContract]:
    asset_bands = ("micro", "meso", "macro")
    return {
        "asset_trend_manual_baseline": FeaturePoolContract(
            feature_family_name="asset_trend_manual_baseline",
            source_columns=("log_return", "macd_hist_12_26_9", "rsi_14", "adx_14"),
            layer_compatibility=("asset_state",),
            axis_compatibility=("trend",),
            band_compatibility=asset_bands,
            required_input_granularity="band_member_intervals",
            missingness_policy=_missingness_policy(),
            leakage_policy="train_window_only",
            source_lineage=_source_lineage("src/regimes/asset_state_test/features.py", "asset_state_test.feature_pool"),
            implementation_status="implemented",
        ),
        "asset_vol_manual_baseline": FeaturePoolContract(
            feature_family_name="asset_vol_manual_baseline",
            source_columns=("atr_14", "ret_std_20", "cv_20", "vol_osc_pct_14_28"),
            layer_compatibility=("asset_state",),
            axis_compatibility=("vol",),
            band_compatibility=asset_bands,
            required_input_granularity="band_member_intervals",
            missingness_policy=_missingness_policy(),
            leakage_policy="train_window_only",
            source_lineage=_source_lineage("src/regimes/asset_state_test/features.py", "asset_state_test.feature_pool"),
            implementation_status="implemented",
        ),
        "asset_activity_manual_baseline": FeaturePoolContract(
            feature_family_name="asset_activity_manual_baseline",
            source_columns=("trade_intensity", "avg_trade_size", "vroc_14", "prr"),
            layer_compatibility=("asset_state",),
            axis_compatibility=("activity",),
            band_compatibility=asset_bands,
            required_input_granularity="band_member_intervals",
            missingness_policy=_missingness_policy(),
            leakage_policy="train_window_only",
            source_lineage=_source_lineage("src/regimes/asset_state_test/features.py", "asset_state_test.feature_pool"),
            implementation_status="implemented",
        ),
        "market_state_declared_manifest": FeaturePoolContract(
            feature_family_name="market_state_declared_manifest",
            source_columns=(
                "advance_decline",
                "positive_return_share",
                "above_moving_average_share",
                "cross_sectional_return_std",
                "cross_sectional_vol_std",
                "return_iqr",
                "mean_pairwise_correlation",
                "median_pairwise_correlation",
                "correlation_breadth",
                "index_realized_vol",
                "median_asset_realized_vol",
                "vol_of_vol",
                "top_decile_contribution",
                "sector_leadership_share",
                "leader_rotation_rate",
            ),
            layer_compatibility=("market_state",),
            axis_compatibility=("market", "breadth", "dispersion", "correlation", "market_vol", "leadership"),
            band_compatibility=("micro", "meso", "macro", "pooled"),
            required_input_granularity="market_universe_membership_snapshot_plus_aggregate_features",
            missingness_policy=_missingness_policy(),
            leakage_policy="declared_not_implemented",
            source_lineage=_source_lineage("src/regimes/pathways.py", "market_state_feature_manifest"),
            implementation_status="declared",
        ),
        "relative_state_declared_manifest": FeaturePoolContract(
            feature_family_name="relative_state_declared_manifest",
            source_columns=(
                "rolling_beta_to_benchmark",
                "downside_beta",
                "upside_beta",
                "rolling_corr_to_benchmark",
                "rolling_corr_to_universe",
                "correlation_stability",
                "relative_return",
                "relative_momentum_rank",
                "excess_return_zscore",
                "distance_from_universe_median",
                "rank_volatility",
                "cross_sectional_zscore",
            ),
            layer_compatibility=("relative_state",),
            axis_compatibility=("relative", "beta", "correlation", "relative_strength", "relative_dispersion"),
            band_compatibility=("micro", "meso", "macro", "pooled"),
            required_input_granularity="relative_peer_basket_plus_benchmark_features",
            missingness_policy=_missingness_policy(),
            leakage_policy="declared_not_implemented",
            source_lineage=_source_lineage("src/regimes/pathways.py", "relative_state_feature_manifest"),
            implementation_status="declared",
        ),
    }


@dataclass(frozen=True)
class FeatureDropRecord:
    column: str
    reason: str
    metric_name: str | None = None
    metric_value: float | None = None
    reference_column: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "column": str(self.column),
            "reason": str(self.reason),
            "metric_name": self.metric_name,
            "metric_value": self.metric_value,
            "reference_column": self.reference_column,
        }


@dataclass(frozen=True)
class FeatureFilterConfig:
    max_missing_fraction: float = 0.2
    min_variance: float = 1e-12
    min_non_null_count: int = 2
    near_duplicate_tolerance: float = 1e-12
    mostly_zero_threshold: float = 0.98
    zero_epsilon: float = 0.0
    correlation_threshold: float | None = None

    def __post_init__(self) -> None:
        for name in ("max_missing_fraction", "mostly_zero_threshold"):
            value = float(getattr(self, name))
            if value < 0.0 or value > 1.0:
                raise ValueError(f"{name} must be between 0 and 1")
            object.__setattr__(self, name, value)
        if float(self.min_variance) < 0.0:
            raise ValueError("min_variance must be non-negative")
        if int(self.min_non_null_count) < 1:
            raise ValueError("min_non_null_count must be positive")
        if float(self.near_duplicate_tolerance) < 0.0:
            raise ValueError("near_duplicate_tolerance must be non-negative")
        if float(self.zero_epsilon) < 0.0:
            raise ValueError("zero_epsilon must be non-negative")
        if self.correlation_threshold is not None:
            threshold = float(self.correlation_threshold)
            if threshold <= 0.0 or threshold > 1.0:
                raise ValueError("correlation_threshold must be in (0, 1]")
            object.__setattr__(self, "correlation_threshold", threshold)
        object.__setattr__(self, "min_variance", float(self.min_variance))
        object.__setattr__(self, "min_non_null_count", int(self.min_non_null_count))
        object.__setattr__(self, "near_duplicate_tolerance", float(self.near_duplicate_tolerance))
        object.__setattr__(self, "zero_epsilon", float(self.zero_epsilon))

    def as_dict(self) -> dict[str, Any]:
        return {
            "max_missing_fraction": float(self.max_missing_fraction),
            "min_variance": float(self.min_variance),
            "min_non_null_count": int(self.min_non_null_count),
            "near_duplicate_tolerance": float(self.near_duplicate_tolerance),
            "mostly_zero_threshold": float(self.mostly_zero_threshold),
            "zero_epsilon": float(self.zero_epsilon),
            "correlation_threshold": self.correlation_threshold,
        }


@dataclass(frozen=True)
class FeatureFilterResult:
    input_columns: tuple[str, ...]
    retained_columns: tuple[str, ...]
    dropped_features: tuple[FeatureDropRecord, ...]
    before_shape: tuple[int, int]
    after_shape: tuple[int, int]
    missingness_summary: Mapping[str, Any]
    variance_summary: Mapping[str, Any]
    zero_share_summary: Mapping[str, Any]
    correlation_summary: Mapping[str, Any]
    config: FeatureFilterConfig
    schema_version: int = REGIME_FEATURE_PREPROCESSING_SCHEMA_VERSION

    @property
    def dropped_columns(self) -> tuple[str, ...]:
        return tuple(record.column for record in self.dropped_features)

    def to_metadata(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "input_columns": list(self.input_columns),
            "retained_columns": list(self.retained_columns),
            "dropped_columns": list(self.dropped_columns),
            "dropped_features": [record.as_dict() for record in self.dropped_features],
            "before_shape": list(self.before_shape),
            "after_shape": list(self.after_shape),
            "missingness_summary": _jsonable(dict(self.missingness_summary)),
            "variance_summary": _jsonable(dict(self.variance_summary)),
            "zero_share_summary": _jsonable(dict(self.zero_share_summary)),
            "correlation_summary": _jsonable(dict(self.correlation_summary)),
            "config": self.config.as_dict(),
        }


def _coerce_numeric(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    numeric = pd.DataFrame(index=frame.index)
    for column in columns:
        if column in frame.columns:
            numeric[str(column)] = pd.to_numeric(frame[str(column)], errors="coerce")
    return numeric.replace([np.inf, -np.inf], np.nan)


def filter_regime_feature_frame(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
    *,
    config: FeatureFilterConfig | None = None,
) -> FeatureFilterResult:
    cfg = config or FeatureFilterConfig()
    input_columns = tuple(dict.fromkeys(str(column) for column in feature_columns))
    numeric = _coerce_numeric(frame, input_columns)
    row_count = int(len(frame))
    dropped: list[FeatureDropRecord] = []
    dropped_by_column: dict[str, FeatureDropRecord] = {}
    missingness: dict[str, Any] = {}
    variance: dict[str, Any] = {}
    zero_share: dict[str, Any] = {}

    def drop(record: FeatureDropRecord) -> None:
        if record.column not in dropped_by_column:
            dropped_by_column[record.column] = record
            dropped.append(record)

    for column in input_columns:
        if column not in numeric.columns:
            missingness[column] = {
                "missing_count": row_count,
                "missing_fraction": 1.0 if row_count else None,
                "non_null_count": 0,
            }
            drop(FeatureDropRecord(column=column, reason="missing_source_column"))
            continue
        series = numeric[column]
        non_null = series.dropna()
        missing_count = int(row_count - len(non_null))
        missing_fraction = float(missing_count / row_count) if row_count else None
        missingness[column] = {
            "missing_count": missing_count,
            "missing_fraction": missing_fraction,
            "non_null_count": int(len(non_null)),
        }
        if int(len(non_null)) < int(cfg.min_non_null_count):
            drop(
                FeatureDropRecord(
                    column=column,
                    reason="insufficient_non_null",
                    metric_name="non_null_count",
                    metric_value=float(len(non_null)),
                )
            )
            continue
        if missing_fraction is not None and missing_fraction > float(cfg.max_missing_fraction):
            drop(
                FeatureDropRecord(
                    column=column,
                    reason="excessive_missingness",
                    metric_name="missing_fraction",
                    metric_value=missing_fraction,
                )
            )
            continue
        var_value = float(non_null.var(ddof=0))
        variance[column] = {
            "variance": var_value,
            "std": float(non_null.std(ddof=0)),
            "min": float(non_null.min()),
            "max": float(non_null.max()),
        }
        if var_value <= float(cfg.min_variance):
            drop(
                FeatureDropRecord(
                    column=column,
                    reason="low_variance",
                    metric_name="variance",
                    metric_value=var_value,
                )
            )
            continue
        zero_fraction = float((non_null.abs() <= float(cfg.zero_epsilon)).sum() / len(non_null))
        zero_share[column] = {
            "zero_fraction": zero_fraction,
            "zero_epsilon": float(cfg.zero_epsilon),
        }
        if zero_fraction >= float(cfg.mostly_zero_threshold):
            drop(
                FeatureDropRecord(
                    column=column,
                    reason="mostly_zero",
                    metric_name="zero_fraction",
                    metric_value=zero_fraction,
                )
            )

    retained = [column for column in input_columns if column not in dropped_by_column]
    final_retained: list[str] = []
    for column in retained:
        candidate = numeric[column]
        duplicate_of: str | None = None
        duplicate_diff: float | None = None
        for reference in final_retained:
            ref = numeric[reference]
            overlap = candidate.notna() & ref.notna()
            if not bool(overlap.any()):
                continue
            max_abs_diff = float(np.max(np.abs(candidate.loc[overlap].to_numpy() - ref.loc[overlap].to_numpy())))
            if max_abs_diff <= float(cfg.near_duplicate_tolerance):
                duplicate_of = reference
                duplicate_diff = max_abs_diff
                break
        if duplicate_of is not None:
            drop(
                FeatureDropRecord(
                    column=column,
                    reason="near_duplicate",
                    metric_name="max_abs_diff",
                    metric_value=duplicate_diff,
                    reference_column=duplicate_of,
                )
            )
        else:
            final_retained.append(column)

    correlation_summary: dict[str, Any] = {
        "correlation_threshold": cfg.correlation_threshold,
        "dropped_pairs": [],
    }
    if cfg.correlation_threshold is not None and len(final_retained) > 1:
        after_corr: list[str] = []
        for column in final_retained:
            redundant_with: str | None = None
            redundant_corr: float | None = None
            for reference in after_corr:
                overlap = numeric[column].notna() & numeric[reference].notna()
                if int(overlap.sum()) < 2:
                    continue
                corr = float(np.corrcoef(numeric.loc[overlap, column], numeric.loc[overlap, reference])[0, 1])
                if math.isfinite(corr) and abs(corr) >= float(cfg.correlation_threshold):
                    redundant_with = reference
                    redundant_corr = abs(corr)
                    break
            if redundant_with is not None:
                correlation_summary["dropped_pairs"].append(
                    {"column": column, "reference_column": redundant_with, "abs_correlation": redundant_corr}
                )
                drop(
                    FeatureDropRecord(
                        column=column,
                        reason="correlation_redundant",
                        metric_name="abs_correlation",
                        metric_value=redundant_corr,
                        reference_column=redundant_with,
                    )
                )
            else:
                after_corr.append(column)
        final_retained = after_corr

    retained_tuple = tuple(column for column in input_columns if column in set(final_retained))
    return FeatureFilterResult(
        input_columns=input_columns,
        retained_columns=retained_tuple,
        dropped_features=tuple(dropped),
        before_shape=(row_count, int(len(input_columns))),
        after_shape=(row_count, int(len(retained_tuple))),
        missingness_summary=missingness,
        variance_summary=variance,
        zero_share_summary=zero_share,
        correlation_summary=correlation_summary,
        config=cfg,
    )


class NoOpTransformer:
    def fit(self, x: np.ndarray, y: object | None = None) -> "NoOpTransformer":
        values = np.asarray(x, dtype=float)
        self.n_features_in_ = int(values.shape[1]) if values.ndim == 2 else 0
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(x, dtype=float)

    def fit_transform(self, x: np.ndarray, y: object | None = None) -> np.ndarray:
        return self.fit(x, y=y).transform(x)

    def get_params(self, deep: bool = False) -> dict[str, Any]:
        return {}


@dataclass(frozen=True)
class PreprocessorSpec:
    name: str
    family: str
    available: bool
    dependency: str | None = None
    default_params: Mapping[str, Any] = field(default_factory=dict)
    output_kind: str = "same_features"
    description: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "family": self.family,
            "available": bool(self.available),
            "dependency": self.dependency,
            "default_params": _jsonable(dict(self.default_params)),
            "output_kind": self.output_kind,
            "description": self.description,
        }


def preprocessing_registry() -> dict[str, PreprocessorSpec]:
    sklearn_available = bool(_HAS_SKLEARN)
    return {
        "noop": PreprocessorSpec(
            name="noop",
            family="identity",
            available=True,
            output_kind="same_features",
            description="No-op numeric feature passthrough.",
        ),
        "standard_scale": PreprocessorSpec(
            name="standard_scale",
            family="scaling",
            available=sklearn_available,
            dependency="sklearn.preprocessing.StandardScaler",
            output_kind="same_features",
            description="Mean-center and unit-variance scale using train-window statistics.",
        ),
        "robust_scale": PreprocessorSpec(
            name="robust_scale",
            family="scaling",
            available=sklearn_available,
            dependency="sklearn.preprocessing.RobustScaler",
            output_kind="same_features",
            description="Median/IQR scale using train-window statistics.",
        ),
        "quantile_transform": PreprocessorSpec(
            name="quantile_transform",
            family="monotonic_transform",
            available=sklearn_available,
            dependency="sklearn.preprocessing.QuantileTransformer",
            default_params={"output_distribution": "normal", "random_state": 17},
            output_kind="same_features",
            description="Quantile transform fitted only on train-window rows.",
        ),
        "power_transform": PreprocessorSpec(
            name="power_transform",
            family="monotonic_transform",
            available=sklearn_available,
            dependency="sklearn.preprocessing.PowerTransformer",
            default_params={"standardize": True},
            output_kind="same_features",
            description="Power transform fitted only on train-window rows.",
        ),
        "pca": PreprocessorSpec(
            name="pca",
            family="embedding",
            available=sklearn_available,
            dependency="sklearn.decomposition.PCA",
            default_params={"random_state": 17},
            output_kind="embedding",
            description="PCA embedding fitted only on train-window rows.",
        ),
        "factor_analysis": PreprocessorSpec(
            name="factor_analysis",
            family="embedding",
            available=sklearn_available,
            dependency="sklearn.decomposition.FactorAnalysis",
            default_params={"random_state": 17},
            output_kind="embedding",
            description="FactorAnalysis embedding fitted only on train-window rows.",
        ),
        "feature_agglomeration": PreprocessorSpec(
            name="feature_agglomeration",
            family="embedding",
            available=sklearn_available,
            dependency="sklearn.cluster.FeatureAgglomeration",
            output_kind="embedding",
            description="Feature agglomeration embedding fitted only on train-window rows.",
        ),
    }


def get_preprocessor_spec(name: str) -> PreprocessorSpec:
    key = _normalize_token(name, field_name="preprocessor name")
    registry = preprocessing_registry()
    if key not in registry:
        valid = ", ".join(sorted(registry))
        raise ValueError(f"Unsupported Regime preprocessor {name!r}; expected one of: {valid}")
    return registry[key]


def _component_count(params: Mapping[str, Any], *, n_samples: int, n_features: int, param_name: str = "n_components") -> int:
    requested = params.get(param_name)
    if requested is None:
        return max(1, min(2, int(n_samples), int(n_features)))
    return max(1, min(int(requested), int(n_samples), int(n_features)))


def _build_transformer(
    name: str,
    *,
    params: Mapping[str, Any] | None = None,
    n_samples: int,
    n_features: int,
) -> object:
    key = get_preprocessor_spec(name).name
    merged = {**dict(get_preprocessor_spec(key).default_params), **dict(params or {})}
    if key == "noop":
        return NoOpTransformer()
    spec = get_preprocessor_spec(key)
    if not spec.available:
        raise ValueError(f"Regime preprocessor {key!r} requires unavailable dependency {spec.dependency!r}")
    if key == "standard_scale":
        return StandardScaler(**merged)  # type: ignore[misc]
    if key == "robust_scale":
        return RobustScaler(**merged)  # type: ignore[misc]
    if key == "quantile_transform":
        merged["n_quantiles"] = max(1, min(int(merged.get("n_quantiles", 1000)), int(n_samples)))
        return QuantileTransformer(**merged)  # type: ignore[misc]
    if key == "power_transform":
        return PowerTransformer(**merged)  # type: ignore[misc]
    if key == "pca":
        merged["n_components"] = _component_count(merged, n_samples=n_samples, n_features=n_features)
        return PCA(**merged)  # type: ignore[misc]
    if key == "factor_analysis":
        merged["n_components"] = _component_count(merged, n_samples=n_samples, n_features=n_features)
        return FactorAnalysis(**merged)  # type: ignore[misc]
    if key == "feature_agglomeration":
        if int(n_features) < 2:
            raise ValueError("feature_agglomeration requires at least two retained features")
        merged["n_clusters"] = max(1, min(int(merged.get("n_clusters", 2)), int(n_features)))
        return FeatureAgglomeration(**merged)  # type: ignore[misc]
    raise ValueError(f"Unsupported Regime preprocessor {name!r}")


def _clean_matrix(frame: pd.DataFrame, columns: Sequence[str]) -> tuple[pd.DataFrame, np.ndarray, int]:
    selected = tuple(str(column) for column in columns)
    if not selected:
        return frame.iloc[0:0].copy(), np.empty((0, 0), dtype=float), int(len(frame))
    missing = [column for column in selected if column not in frame.columns]
    if missing:
        return frame.iloc[0:0].copy(), np.empty((0, len(selected)), dtype=float), int(len(frame))
    numeric = _coerce_numeric(frame, selected)
    finite_mask = numeric.notna().all(axis=1)
    clean = frame.loc[finite_mask].copy()
    values = numeric.loc[finite_mask, list(selected)].to_numpy(dtype=float)
    return clean, np.asarray(values, dtype=float), int(len(frame) - len(clean))


def _output_feature_names(transformer: object | None, preprocess_name: str, selected_columns: Sequence[str], x: np.ndarray) -> tuple[str, ...]:
    selected = tuple(str(column) for column in selected_columns)
    if transformer is not None and hasattr(transformer, "get_feature_names_out"):
        try:
            names = tuple(str(name) for name in transformer.get_feature_names_out(selected))  # type: ignore[attr-defined]
            if len(names) == int(x.shape[1]):
                return names
        except Exception:
            pass
    if int(x.shape[1]) == len(selected):
        return selected
    return tuple(f"{preprocess_name}_{idx}" for idx in range(int(x.shape[1])))


def _transformer_metadata(transformer: object | None) -> dict[str, Any]:
    if transformer is None:
        return {"class_name": None, "params": {}, "learned_attributes": {}}
    params: Mapping[str, Any] = {}
    if hasattr(transformer, "get_params"):
        try:
            params = transformer.get_params(deep=False)  # type: ignore[attr-defined]
        except Exception:
            params = {}
    learned: dict[str, Any] = {}
    for attr in (
        "n_features_in_",
        "n_components_",
        "n_components",
        "n_clusters",
        "n_quantiles_",
        "explained_variance_ratio_",
    ):
        if hasattr(transformer, attr):
            learned[attr] = _jsonable(getattr(transformer, attr))
    return {
        "class_name": type(transformer).__name__,
        "params": _jsonable(dict(params)),
        "learned_attributes": learned,
    }


@dataclass(frozen=True)
class TransformedFeatureMatrix:
    x: np.ndarray
    clean_frame: pd.DataFrame
    selected_columns: tuple[str, ...]
    output_feature_names: tuple[str, ...]
    metadata: Mapping[str, Any]

    def to_metadata(self) -> dict[str, Any]:
        return _jsonable(dict(self.metadata))


@dataclass(frozen=True)
class FittedRegimePreprocessor:
    x: np.ndarray
    clean_frame: pd.DataFrame
    input_columns: tuple[str, ...]
    selected_columns: tuple[str, ...]
    output_feature_names: tuple[str, ...]
    preprocess_name: str
    preprocessor_spec: PreprocessorSpec
    filter_result: FeatureFilterResult
    transformer: object | None = field(default=None, repr=False, compare=False)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def dropped_columns(self) -> tuple[str, ...]:
        return self.filter_result.dropped_columns

    def transform(self, frame: pd.DataFrame, *, window_role: str = "score") -> TransformedFeatureMatrix:
        return transform_regime_preprocessor(frame, self, window_role=window_role)

    def to_metadata(self) -> dict[str, Any]:
        return _jsonable(dict(self.metadata))


def _fit_role_is_train(role: object) -> bool:
    return str(role).strip().lower() in {"train", "training", "train_window"}


def fit_regime_preprocessor(
    train_frame: pd.DataFrame,
    feature_columns: Sequence[str],
    *,
    preprocess: str = "robust_scale",
    preprocess_params: Mapping[str, Any] | None = None,
    filter_config: FeatureFilterConfig | None = None,
    fit_window: Mapping[str, Any] | None = None,
    fit_window_role: str = "train",
    allow_full_window_fit: bool = False,
) -> FittedRegimePreprocessor:
    warnings: list[str] = []
    if not _fit_role_is_train(fit_window_role):
        if not allow_full_window_fit:
            raise ValueError("Regime preprocessing fit must use the train window; full-window fitting is rejected")
        warnings.append(f"non_train_fit_window_role:{fit_window_role}")
    spec = get_preprocessor_spec(preprocess)
    filter_result = filter_regime_feature_frame(train_frame, feature_columns, config=filter_config)
    selected = filter_result.retained_columns
    clean_frame, x_fit, dropped_rows = _clean_matrix(train_frame, selected)
    transformer: object | None = None
    if len(selected) and x_fit.shape[0] > 0:
        transformer = _build_transformer(
            spec.name,
            params=preprocess_params,
            n_samples=int(x_fit.shape[0]),
            n_features=int(x_fit.shape[1]),
        )
        x_out = np.asarray(transformer.fit_transform(x_fit), dtype=float)  # type: ignore[attr-defined]
    else:
        x_out = np.empty((0, len(selected)), dtype=float)
    output_names = _output_feature_names(transformer, spec.name, selected, x_out)
    metadata = {
        "schema_version": REGIME_FEATURE_PREPROCESSING_SCHEMA_VERSION,
        "fit_scope": PREPROCESS_FIT_SCOPE_TRAIN_ONLY,
        "fit_window_role": str(fit_window_role),
        "fit_window": _jsonable(dict(fit_window or {})),
        "warnings": warnings,
        "preprocess_name": spec.name,
        "preprocessor_spec": spec.as_dict(),
        "input_columns": list(filter_result.input_columns),
        "selected_columns": list(selected),
        "dropped_columns": list(filter_result.dropped_columns),
        "dropped_features": [record.as_dict() for record in filter_result.dropped_features],
        "before_shape": list(filter_result.before_shape),
        "after_filter_shape": list(filter_result.after_shape),
        "rows_before_clean": int(len(train_frame)),
        "rows_after_clean": int(len(clean_frame)),
        "dropped_row_count": int(dropped_rows),
        "fit_input_shape": [int(x_fit.shape[0]), int(x_fit.shape[1])],
        "fit_output_shape": [int(x_out.shape[0]), int(x_out.shape[1])],
        "output_feature_names": list(output_names),
        "filter_diagnostics": filter_result.to_metadata(),
        "transformer": _transformer_metadata(transformer),
    }
    return FittedRegimePreprocessor(
        x=x_out,
        clean_frame=clean_frame,
        input_columns=filter_result.input_columns,
        selected_columns=selected,
        output_feature_names=output_names,
        preprocess_name=spec.name,
        preprocessor_spec=spec,
        filter_result=filter_result,
        transformer=transformer,
        metadata=metadata,
    )


def transform_regime_preprocessor(
    frame: pd.DataFrame,
    fitted: FittedRegimePreprocessor,
    *,
    window_role: str = "score",
) -> TransformedFeatureMatrix:
    clean_frame, x_in, dropped_rows = _clean_matrix(frame, fitted.selected_columns)
    if x_in.shape[0] == 0 or x_in.shape[1] == 0:
        x_out = np.empty((0, len(fitted.output_feature_names)), dtype=float)
    elif fitted.transformer is None:
        x_out = x_in
    else:
        x_out = np.asarray(fitted.transformer.transform(x_in), dtype=float)  # type: ignore[attr-defined]
    output_names = _output_feature_names(fitted.transformer, fitted.preprocess_name, fitted.selected_columns, x_out)
    metadata = {
        "schema_version": REGIME_FEATURE_PREPROCESSING_SCHEMA_VERSION,
        "fit_scope": PREPROCESS_FIT_SCOPE_TRAIN_ONLY,
        "window_role": str(window_role),
        "preprocess_name": fitted.preprocess_name,
        "selected_columns": list(fitted.selected_columns),
        "output_feature_names": list(output_names),
        "rows_before_clean": int(len(frame)),
        "rows_after_clean": int(len(clean_frame)),
        "dropped_row_count": int(dropped_rows),
        "transform_input_shape": [int(x_in.shape[0]), int(x_in.shape[1])],
        "transform_output_shape": [int(x_out.shape[0]), int(x_out.shape[1])],
        "fitted_preprocess_metadata": fitted.to_metadata(),
    }
    return TransformedFeatureMatrix(
        x=x_out,
        clean_frame=clean_frame,
        selected_columns=fitted.selected_columns,
        output_feature_names=output_names,
        metadata=metadata,
    )


__all__ = [
    "FEATURE_LEAKAGE_POLICIES",
    "FEATURE_POOL_IMPLEMENTATION_STATUSES",
    "PREPROCESS_FIT_SCOPE_TRAIN_ONLY",
    "REGIME_FEATURE_PREPROCESSING_SCHEMA_VERSION",
    "FeatureDropRecord",
    "FeatureFilterConfig",
    "FeatureFilterResult",
    "FeaturePoolContract",
    "FittedRegimePreprocessor",
    "NoOpTransformer",
    "PreprocessorSpec",
    "TransformedFeatureMatrix",
    "default_feature_pool_registry",
    "filter_regime_feature_frame",
    "fit_regime_preprocessor",
    "get_preprocessor_spec",
    "preprocessing_registry",
    "transform_regime_preprocessor",
]
