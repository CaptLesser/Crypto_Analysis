from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from src.regimes.core.serialization import dumps_json, loads_json, require_json_object, to_jsonable
from src.regimes.market_state.contracts import (
    MARKET_STATE_SCHEMA_VERSION,
    MarketStateAxis,
    MarketStateBand,
    MarketStateSchemaVersion,
    _enum_value,
    _schema_version,
)
from src.regimes.market_state.feature_builder import (
    MARKET_FEATURE_BUILD_STATUS_READY,
    MarketFeatureBuildResult,
)
from src.regimes.market_state.feature_registry import (
    MarketFeatureRegistry,
    default_market_state_feature_registry,
)
from src.regimes.market_state.taxonomy import default_market_state_taxonomy


MARKET_STATE_CLUSTERABILITY_STATUS_CLUSTERABLE_CANDIDATE = "market_state_clusterable_candidate"
MARKET_STATE_CLUSTERABILITY_STATUS_INSUFFICIENT_CORE_BASKET = "insufficient_core_basket"
MARKET_STATE_CLUSTERABILITY_STATUS_INSUFFICIENT_BROAD_UNIVERSE = "insufficient_broad_universe"
MARKET_STATE_CLUSTERABILITY_STATUS_INSUFFICIENT_TIMESTAMPS = "insufficient_timestamps"
MARKET_STATE_CLUSTERABILITY_STATUS_INSUFFICIENT_FEATURE_COVERAGE = "insufficient_feature_coverage"
MARKET_STATE_CLUSTERABILITY_STATUS_LOW_VARIATION_MARKET_WINDOW = "low_variation_market_window"
MARKET_STATE_CLUSTERABILITY_STATUS_COVARIANCE_UNAVAILABLE = "covariance_unavailable"
MARKET_STATE_CLUSTERABILITY_STATUS_AXIS_NOT_CLUSTERABLE = "axis_not_clusterable"
MARKET_STATE_CLUSTERABILITY_STATUS_UNKNOWN_ERROR = "unknown_error"

MARKET_STATE_CLUSTERABILITY_STATUSES: tuple[str, ...] = (
    MARKET_STATE_CLUSTERABILITY_STATUS_CLUSTERABLE_CANDIDATE,
    MARKET_STATE_CLUSTERABILITY_STATUS_INSUFFICIENT_CORE_BASKET,
    MARKET_STATE_CLUSTERABILITY_STATUS_INSUFFICIENT_BROAD_UNIVERSE,
    MARKET_STATE_CLUSTERABILITY_STATUS_INSUFFICIENT_TIMESTAMPS,
    MARKET_STATE_CLUSTERABILITY_STATUS_INSUFFICIENT_FEATURE_COVERAGE,
    MARKET_STATE_CLUSTERABILITY_STATUS_LOW_VARIATION_MARKET_WINDOW,
    MARKET_STATE_CLUSTERABILITY_STATUS_COVARIANCE_UNAVAILABLE,
    MARKET_STATE_CLUSTERABILITY_STATUS_AXIS_NOT_CLUSTERABLE,
    MARKET_STATE_CLUSTERABILITY_STATUS_UNKNOWN_ERROR,
)

TIMESTAMP_COLUMN = "ts"


@dataclass(frozen=True)
class MarketStateClusterabilityPolicy:
    policy_id: str = "market_state_clusterability_default_v1"
    min_timestamp_count: int = 32
    min_core_asset_count: int = 2
    min_broad_asset_count: int = 3
    min_feature_count: int = 3
    min_finite_feature_count: int = 3
    min_feature_finite_fraction: float = 0.60
    min_overall_finite_fraction: float = 0.60
    max_missing_fraction: float = 0.40
    low_variance_threshold: float = 1e-12
    min_nonzero_variance_features: int = 2
    low_variance_feature_share_threshold: float = 0.90
    schema_version: int | MarketStateSchemaVersion = MARKET_STATE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for field_name in (
            "min_timestamp_count",
            "min_core_asset_count",
            "min_broad_asset_count",
            "min_feature_count",
            "min_finite_feature_count",
            "min_nonzero_variance_features",
        ):
            value = int(getattr(self, field_name))
            if value < 1:
                raise ValueError(f"Market-state clusterability {field_name} must be positive")
            object.__setattr__(self, field_name, value)
        for field_name in (
            "min_feature_finite_fraction",
            "min_overall_finite_fraction",
            "max_missing_fraction",
            "low_variance_feature_share_threshold",
        ):
            value = float(getattr(self, field_name))
            if value < 0.0 or value > 1.0:
                raise ValueError(f"Market-state clusterability {field_name} must be within [0, 1]")
            object.__setattr__(self, field_name, value)
        if float(self.low_variance_threshold) < 0.0:
            raise ValueError("Market-state clusterability low_variance_threshold must be non-negative")
        object.__setattr__(self, "schema_version", _schema_version(self.schema_version))
        object.__setattr__(self, "policy_id", str(self.policy_id).strip() or "market_state_clusterability_default_v1")
        object.__setattr__(self, "low_variance_threshold", float(self.low_variance_threshold))

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "policy_id": self.policy_id,
            "min_timestamp_count": int(self.min_timestamp_count),
            "min_core_asset_count": int(self.min_core_asset_count),
            "min_broad_asset_count": int(self.min_broad_asset_count),
            "min_feature_count": int(self.min_feature_count),
            "min_finite_feature_count": int(self.min_finite_feature_count),
            "min_feature_finite_fraction": float(self.min_feature_finite_fraction),
            "min_overall_finite_fraction": float(self.min_overall_finite_fraction),
            "max_missing_fraction": float(self.max_missing_fraction),
            "low_variance_threshold": float(self.low_variance_threshold),
            "min_nonzero_variance_features": int(self.min_nonzero_variance_features),
            "low_variance_feature_share_threshold": float(self.low_variance_feature_share_threshold),
            "fail_closed": True,
        }


@dataclass(frozen=True)
class MarketStateClusterabilityResult:
    status: str
    axis: str
    band: str
    diagnostics: Mapping[str, Any]
    blocking_reasons: Sequence[str] = ()
    fallback_hints: Sequence[str] = ()
    clusterer_feature_names: Sequence[str] = ()
    policy: MarketStateClusterabilityPolicy = field(default_factory=MarketStateClusterabilityPolicy)
    schema_version: int | MarketStateSchemaVersion = MARKET_STATE_SCHEMA_VERSION
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        status = str(self.status).strip().lower()
        if status not in MARKET_STATE_CLUSTERABILITY_STATUSES:
            raise ValueError(f"Unsupported market-state clusterability status {status!r}")
        object.__setattr__(self, "schema_version", _schema_version(self.schema_version))
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "blocking_reasons", tuple(str(reason) for reason in self.blocking_reasons))
        object.__setattr__(self, "fallback_hints", tuple(str(hint) for hint in self.fallback_hints))
        object.__setattr__(self, "clusterer_feature_names", tuple(str(feature) for feature in self.clusterer_feature_names))
        object.__setattr__(self, "diagnostics", to_jsonable(dict(self.diagnostics)))
        object.__setattr__(self, "metadata", to_jsonable(dict(self.metadata)))

    @property
    def clusterable_candidate(self) -> bool:
        return self.status == MARKET_STATE_CLUSTERABILITY_STATUS_CLUSTERABLE_CANDIDATE

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "artifact_kind": "market_state_clusterability_result",
            "status": self.status,
            "axis": self.axis,
            "band": self.band,
            "clusterable_candidate": self.clusterable_candidate,
            "diagnostics": to_jsonable(dict(self.diagnostics)),
            "blocking_reasons": list(self.blocking_reasons),
            "fallback_hints": list(self.fallback_hints),
            "clusterer_feature_names": list(self.clusterer_feature_names),
            "policy": self.policy.as_dict(),
            "metadata": to_jsonable(dict(self.metadata)),
            "clusterer_fit_allowed": self.clusterable_candidate,
            "production_labels_written": False,
        }

    def to_json(self, **kwargs: Any) -> str:
        return dumps_json(self.as_dict(), **kwargs)

    @classmethod
    def from_json(cls, text: str) -> dict[str, Any]:
        return require_json_object(loads_json(text), context="MarketStateClusterabilityResult JSON")


def evaluate_market_state_clusterability(
    feature_result: MarketFeatureBuildResult,
    *,
    axis: str | MarketStateAxis,
    band: str | MarketStateBand,
    policy: MarketStateClusterabilityPolicy | None = None,
    registry: MarketFeatureRegistry | None = None,
) -> MarketStateClusterabilityResult:
    cfg = policy or MarketStateClusterabilityPolicy()
    reg = registry or _registry_from_feature_result(feature_result)
    try:
        taxonomy = default_market_state_taxonomy()
        try:
            axis_value = _enum_value(axis, MarketStateAxis, field_name="axis")
            band_value = _enum_value(band, MarketStateBand, field_name="band")
            taxonomy.validate_axis_band(axis_value, band_value)
            axis_spec = taxonomy.axis_spec(axis_value)
        except Exception as exc:
            return _result(
                status=MARKET_STATE_CLUSTERABILITY_STATUS_AXIS_NOT_CLUSTERABLE,
                axis=str(axis),
                band=str(band),
                policy=cfg,
                diagnostics={"axis_band_valid": False, "error": str(exc)},
                blocking_reasons=(f"axis/band is not clusterable: {exc}",),
                fallback_hints=("do_not_fit_clusterer", "review_market_state_taxonomy"),
            )

        dataset_summary = dict(feature_result.dataset_summary or {})
        core_asset_count = int(dataset_summary.get("core_asset_count") or 0)
        broad_asset_count = int(dataset_summary.get("broad_asset_count") or 0)
        if core_asset_count < int(cfg.min_core_asset_count):
            return _result(
                status=MARKET_STATE_CLUSTERABILITY_STATUS_INSUFFICIENT_CORE_BASKET,
                axis=axis_value,
                band=band_value,
                policy=cfg,
                diagnostics={"core_asset_count": core_asset_count, "dataset_summary": dataset_summary},
                blocking_reasons=(f"core asset count {core_asset_count} < min_core_asset_count {cfg.min_core_asset_count}",),
                fallback_hints=("select_larger_core_basket", "do_not_fit_clusterer"),
            )
        if broad_asset_count < int(cfg.min_broad_asset_count):
            return _result(
                status=MARKET_STATE_CLUSTERABILITY_STATUS_INSUFFICIENT_BROAD_UNIVERSE,
                axis=axis_value,
                band=band_value,
                policy=cfg,
                diagnostics={"broad_asset_count": broad_asset_count, "dataset_summary": dataset_summary},
                blocking_reasons=(f"broad asset count {broad_asset_count} < min_broad_asset_count {cfg.min_broad_asset_count}",),
                fallback_hints=("expand_broad_universe", "do_not_fit_clusterer"),
            )
        if feature_result.status != MARKET_FEATURE_BUILD_STATUS_READY or not feature_result.usable:
            return _result(
                status=MARKET_STATE_CLUSTERABILITY_STATUS_INSUFFICIENT_FEATURE_COVERAGE,
                axis=axis_value,
                band=band_value,
                policy=cfg,
                diagnostics={"feature_build_status": feature_result.status, "feature_reason_codes": list(feature_result.reason_codes)},
                blocking_reasons=(f"feature build status is not ready: {feature_result.status}",),
                fallback_hints=("rebuild_market_features", "do_not_fit_clusterer"),
            )

        matrix = feature_result.feature_matrix.copy()
        if matrix.empty:
            return _result(
                status=MARKET_STATE_CLUSTERABILITY_STATUS_INSUFFICIENT_TIMESTAMPS,
                axis=axis_value,
                band=band_value,
                policy=cfg,
                diagnostics={"timestamp_count": 0},
                blocking_reasons=("feature matrix is empty",),
                fallback_hints=("increase_window", "do_not_fit_clusterer"),
            )

        axis_features = _axis_feature_names(feature_result, registry=reg, axis=axis_value, band=band_value)
        if len(axis_features) < int(cfg.min_feature_count):
            return _result(
                status=MARKET_STATE_CLUSTERABILITY_STATUS_AXIS_NOT_CLUSTERABLE,
                axis=axis_value,
                band=band_value,
                policy=cfg,
                diagnostics={"axis_feature_count": len(axis_features), "axis_features": list(axis_features)},
                blocking_reasons=("axis has too few registered market-state features for clustering",),
                fallback_hints=("review_axis_feature_registry", "do_not_fit_clusterer"),
            )

        numeric = matrix[list(axis_features)].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
        timestamp_count = int(numeric.shape[0])
        if timestamp_count < int(cfg.min_timestamp_count):
            return _result(
                status=MARKET_STATE_CLUSTERABILITY_STATUS_INSUFFICIENT_TIMESTAMPS,
                axis=axis_value,
                band=band_value,
                policy=cfg,
                diagnostics={"timestamp_count": timestamp_count, "axis_features": list(axis_features)},
                blocking_reasons=(f"timestamp count {timestamp_count} < min_timestamp_count {cfg.min_timestamp_count}",),
                fallback_hints=("increase_window", "do_not_fit_clusterer"),
            )

        feature_coverage = {
            feature: float(numeric[feature].notna().mean()) if timestamp_count else 0.0
            for feature in numeric.columns
        }
        finite_features = tuple(
            feature
            for feature, coverage in feature_coverage.items()
            if coverage >= float(cfg.min_feature_finite_fraction)
        )
        overall_finite_fraction = float(numeric.notna().to_numpy().mean()) if numeric.size else 0.0
        missing_fraction = 1.0 - overall_finite_fraction
        coverage_diagnostics = {
            "timestamp_count": timestamp_count,
            "axis_feature_count": len(axis_features),
            "finite_feature_count": len(finite_features),
            "feature_coverage": feature_coverage,
            "overall_finite_fraction": overall_finite_fraction,
            "missing_fraction": missing_fraction,
            "finite_features": list(finite_features),
        }
        if (
            len(finite_features) < int(cfg.min_finite_feature_count)
            or overall_finite_fraction < float(cfg.min_overall_finite_fraction)
            or missing_fraction > float(cfg.max_missing_fraction)
        ):
            return _result(
                status=MARKET_STATE_CLUSTERABILITY_STATUS_INSUFFICIENT_FEATURE_COVERAGE,
                axis=axis_value,
                band=band_value,
                policy=cfg,
                diagnostics=coverage_diagnostics,
                blocking_reasons=("market feature coverage or missingness failed clusterability policy",),
                fallback_hints=("rebuild_features_with_broader_coverage", "do_not_fit_clusterer"),
            )

        covariance_diagnostics = _covariance_availability(feature_result)
        if axis_spec.requires_covariance_correlation_features and not covariance_diagnostics["available"]:
            return _result(
                status=MARKET_STATE_CLUSTERABILITY_STATUS_COVARIANCE_UNAVAILABLE,
                axis=axis_value,
                band=band_value,
                policy=cfg,
                diagnostics={**coverage_diagnostics, "covariance_correlation": covariance_diagnostics},
                blocking_reasons=("axis requires covariance/correlation features but they are unavailable",),
                fallback_hints=("compute_core_basket_covariance_features", "do_not_fit_clusterer"),
            )

        usable = numeric[list(finite_features)].copy()
        variation_diagnostics = _variation_diagnostics(usable, cfg)
        if variation_diagnostics["low_variation_market_window"]:
            hints = ["do_not_fit_clusterer"]
            if axis_spec.allow_single_state_output:
                hints.append("single_state_output_allowed_by_axis_policy")
            else:
                hints.append("single_state_output_not_allowed_by_axis_policy")
            return _result(
                status=MARKET_STATE_CLUSTERABILITY_STATUS_LOW_VARIATION_MARKET_WINDOW,
                axis=axis_value,
                band=band_value,
                policy=cfg,
                diagnostics={
                    **coverage_diagnostics,
                    "covariance_correlation": covariance_diagnostics,
                    "variation": variation_diagnostics,
                    "axis_allows_single_state_output": bool(axis_spec.allow_single_state_output),
                },
                blocking_reasons=("market feature window has insufficient variation for clustering",),
                fallback_hints=tuple(hints),
            )

        return _result(
            status=MARKET_STATE_CLUSTERABILITY_STATUS_CLUSTERABLE_CANDIDATE,
            axis=axis_value,
            band=band_value,
            policy=cfg,
            diagnostics={
                **coverage_diagnostics,
                "covariance_correlation": covariance_diagnostics,
                "variation": variation_diagnostics,
                "core_asset_count": core_asset_count,
                "broad_asset_count": broad_asset_count,
            },
            blocking_reasons=(),
            fallback_hints=("fit_clusterer_allowed",),
            clusterer_feature_names=finite_features,
            metadata={
                "axis_policy": axis_spec.as_dict(),
                "filter_before_clusterer_fit": True,
            },
        )
    except Exception as exc:
        return _result(
            status=MARKET_STATE_CLUSTERABILITY_STATUS_UNKNOWN_ERROR,
            axis=str(axis),
            band=str(band),
            policy=cfg,
            diagnostics={"error": str(exc)},
            blocking_reasons=(f"market-state clusterability evaluation failed closed: {exc}",),
            fallback_hints=("manual_review", "do_not_fit_clusterer"),
        )


def market_state_clusterer_input_matrix(
    feature_result: MarketFeatureBuildResult,
    clusterability: MarketStateClusterabilityResult,
) -> pd.DataFrame:
    if not clusterability.clusterable_candidate:
        return pd.DataFrame()
    feature_names = tuple(feature for feature in clusterability.clusterer_feature_names if feature in feature_result.feature_matrix.columns)
    if not feature_names:
        return pd.DataFrame()
    columns = [TIMESTAMP_COLUMN] if TIMESTAMP_COLUMN in feature_result.feature_matrix.columns else []
    columns.extend(feature_names)
    return feature_result.feature_matrix[columns].copy()


def write_market_state_clusterability_report(
    path: str | Path,
    result: MarketStateClusterabilityResult,
) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(f".{out.name}.{os.getpid()}.tmp")
    tmp.write_text(result.to_json(indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, out)
    return out


def _registry_from_feature_result(feature_result: MarketFeatureBuildResult) -> MarketFeatureRegistry:
    if feature_result.registry:
        try:
            return MarketFeatureRegistry.from_dict(feature_result.registry)
        except Exception:
            pass
    return default_market_state_feature_registry()


def _axis_feature_names(
    feature_result: MarketFeatureBuildResult,
    *,
    registry: MarketFeatureRegistry,
    axis: str,
    band: str,
) -> tuple[str, ...]:
    available = set(str(feature) for feature in feature_result.feature_names)
    features: list[str] = []
    for spec in registry.families.values():
        if not spec.supports(axis=axis, band=band):
            continue
        features.extend(feature for feature in spec.output_features if feature in available)
    return tuple(dict.fromkeys(features))


def _covariance_availability(feature_result: MarketFeatureBuildResult) -> dict[str, Any]:
    diagnostics = dict(feature_result.covariance_correlation_diagnostics or {})
    correlation = _method_available(diagnostics.get("correlation"))
    ledoit = _method_available(diagnostics.get("ledoit_wolf"))
    oas = _method_available(diagnostics.get("oas"))
    covariance = bool(ledoit or oas)
    return {
        "available": bool(correlation and covariance),
        "correlation_available": bool(correlation),
        "covariance_available": bool(covariance),
        "ledoit_wolf_available": bool(ledoit),
        "oas_available": bool(oas),
        "raw_diagnostics": to_jsonable(diagnostics),
    }


def _method_available(payload: object) -> bool:
    if not isinstance(payload, Mapping):
        return False
    status = str(payload.get("status") or "").strip().lower()
    return status in {"computed", "partial"} and int(payload.get("computed_rows") or 0) > 0


def _variation_diagnostics(numeric: pd.DataFrame, policy: MarketStateClusterabilityPolicy) -> dict[str, Any]:
    if numeric.empty:
        return {
            "nonzero_variance_feature_count": 0,
            "low_variance_feature_count": 0,
            "low_variance_feature_share": None,
            "low_variation_market_window": True,
            "variance_summary": {},
        }
    variances = numeric.var(axis=0, skipna=True, ddof=0)
    low_variance = variances.fillna(0.0) <= float(policy.low_variance_threshold)
    low_count = int(low_variance.sum())
    nonzero_count = int((variances.fillna(0.0) > float(policy.low_variance_threshold)).sum())
    share = float(low_count / max(1, len(variances)))
    low_window = bool(
        nonzero_count < int(policy.min_nonzero_variance_features)
        or share >= float(policy.low_variance_feature_share_threshold)
    )
    return {
        "nonzero_variance_feature_count": nonzero_count,
        "low_variance_feature_count": low_count,
        "low_variance_feature_share": share,
        "low_variation_market_window": low_window,
        "variance_summary": {str(key): float(value) if np.isfinite(value) else None for key, value in variances.items()},
    }


def _result(
    *,
    status: str,
    axis: str,
    band: str,
    policy: MarketStateClusterabilityPolicy,
    diagnostics: Mapping[str, Any],
    blocking_reasons: Sequence[str],
    fallback_hints: Sequence[str],
    clusterer_feature_names: Sequence[str] = (),
    metadata: Mapping[str, Any] | None = None,
) -> MarketStateClusterabilityResult:
    return MarketStateClusterabilityResult(
        status=status,
        axis=axis,
        band=band,
        diagnostics=diagnostics,
        blocking_reasons=tuple(blocking_reasons),
        fallback_hints=tuple(fallback_hints),
        clusterer_feature_names=tuple(clusterer_feature_names),
        policy=policy,
        metadata=metadata or {},
    )


__all__ = [
    "MARKET_STATE_CLUSTERABILITY_STATUSES",
    "MARKET_STATE_CLUSTERABILITY_STATUS_AXIS_NOT_CLUSTERABLE",
    "MARKET_STATE_CLUSTERABILITY_STATUS_CLUSTERABLE_CANDIDATE",
    "MARKET_STATE_CLUSTERABILITY_STATUS_COVARIANCE_UNAVAILABLE",
    "MARKET_STATE_CLUSTERABILITY_STATUS_INSUFFICIENT_BROAD_UNIVERSE",
    "MARKET_STATE_CLUSTERABILITY_STATUS_INSUFFICIENT_CORE_BASKET",
    "MARKET_STATE_CLUSTERABILITY_STATUS_INSUFFICIENT_FEATURE_COVERAGE",
    "MARKET_STATE_CLUSTERABILITY_STATUS_INSUFFICIENT_TIMESTAMPS",
    "MARKET_STATE_CLUSTERABILITY_STATUS_LOW_VARIATION_MARKET_WINDOW",
    "MARKET_STATE_CLUSTERABILITY_STATUS_UNKNOWN_ERROR",
    "MarketStateClusterabilityPolicy",
    "MarketStateClusterabilityResult",
    "evaluate_market_state_clusterability",
    "market_state_clusterer_input_matrix",
    "write_market_state_clusterability_report",
]
