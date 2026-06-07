from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from src.regimes.core.serialization import to_jsonable
from src.regimes.core.test_branch_contracts import (
    MASK_INSUFFICIENT_HISTORY,
    MASK_LOW_VARIANCE_NEAR_FLAT,
    MASK_MISSING_REQUIRED_FEATURES,
)
from src.regimes.core.window_profiles import RegimeWindowProfile


MARKET_STATE_WINDOW_PROFILE_SCHEMA_VERSION = 1

MARKET_STATE_COVERAGE_INSUFFICIENT_ROWS = "insufficient_rows"
MARKET_STATE_COVERAGE_INSUFFICIENT_FINITE_COVERAGE = "insufficient_finite_coverage"
MARKET_STATE_COVERAGE_INSUFFICIENT_ACTIVITY_COVERAGE = "insufficient_activity_coverage"
MARKET_STATE_COVERAGE_INSUFFICIENT_CORRELATION_SAMPLE = "insufficient_correlation_sample"
MARKET_STATE_COVERAGE_LOW_VARIANCE_NEAR_FLAT = "low_variance_near_flat"
MARKET_STATE_COVERAGE_MISSING_REQUIRED_FEATURES = "missing_required_features"

MARKET_STATE_COVERAGE_REASON_CODES: tuple[str, ...] = (
    MARKET_STATE_COVERAGE_INSUFFICIENT_ROWS,
    MARKET_STATE_COVERAGE_INSUFFICIENT_FINITE_COVERAGE,
    MARKET_STATE_COVERAGE_INSUFFICIENT_ACTIVITY_COVERAGE,
    MARKET_STATE_COVERAGE_INSUFFICIENT_CORRELATION_SAMPLE,
    MARKET_STATE_COVERAGE_LOW_VARIANCE_NEAR_FLAT,
    MARKET_STATE_COVERAGE_MISSING_REQUIRED_FEATURES,
)


@dataclass(frozen=True)
class MarketStateCoverageGatePolicy:
    policy_id: str = "market_state_window_coverage_gate_v1"
    min_rows: int = 48
    min_median_finite_return_coverage_ratio: float = 0.60
    min_p10_finite_return_coverage_ratio: float = 0.35
    min_activity_coverage: float = 0.20
    min_correlation_sample_rows: int = 24
    low_variance_threshold: float = 1e-12
    require_fixed_universe: bool = False
    schema_version: int = MARKET_STATE_WINDOW_PROFILE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "policy_id", str(self.policy_id).strip() or "market_state_window_coverage_gate_v1")
        object.__setattr__(self, "min_rows", max(1, int(self.min_rows)))
        object.__setattr__(self, "min_correlation_sample_rows", max(1, int(self.min_correlation_sample_rows)))
        for field_name in (
            "min_median_finite_return_coverage_ratio",
            "min_p10_finite_return_coverage_ratio",
            "min_activity_coverage",
        ):
            value = min(1.0, max(0.0, float(getattr(self, field_name))))
            object.__setattr__(self, field_name, value)
        object.__setattr__(self, "low_variance_threshold", max(0.0, float(self.low_variance_threshold)))
        object.__setattr__(self, "require_fixed_universe", bool(self.require_fixed_universe))
        object.__setattr__(self, "schema_version", int(self.schema_version))

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "policy_id": self.policy_id,
            "min_rows": int(self.min_rows),
            "min_median_finite_return_coverage_ratio": float(self.min_median_finite_return_coverage_ratio),
            "min_p10_finite_return_coverage_ratio": float(self.min_p10_finite_return_coverage_ratio),
            "min_activity_coverage": float(self.min_activity_coverage),
            "min_correlation_sample_rows": int(self.min_correlation_sample_rows),
            "low_variance_threshold": float(self.low_variance_threshold),
            "require_fixed_universe": bool(self.require_fixed_universe),
            "fail_closed": True,
        }


@dataclass(frozen=True)
class MarketStateCoverageSummary:
    market_axis: str
    band: str
    window_profile: RegimeWindowProfile
    row_count: int
    start_ts: int | None
    end_ts: int | None
    source_tail_ts: int | None
    median_finite_return_coverage_ratio: float | None
    p10_finite_return_coverage_ratio: float | None
    min_finite_return_coverage_ratio: float | None
    median_active_finite_asset_count: float | None
    broad_denominator_count: int | None
    activity_coverage: float | None
    covariance_correlation_sample_available: bool
    covariance_correlation_sample_rows: int
    stable_feature_pool_available: bool
    speculative_feature_pool_available: bool
    required_feature_coverage: Mapping[str, float] = field(default_factory=dict)
    fixed_universe_required: bool = False
    schema_version: int = MARKET_STATE_WINDOW_PROFILE_SCHEMA_VERSION

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "market_axis": self.market_axis,
            "band": self.band,
            "window_profile_id": self.window_profile.window_profile_id,
            "window_profile": self.window_profile.as_dict(),
            "row_count": int(self.row_count),
            "start_ts": self.start_ts,
            "end_ts": self.end_ts,
            "source_tail_ts": self.source_tail_ts,
            "median_finite_return_coverage_ratio": self.median_finite_return_coverage_ratio,
            "p10_finite_return_coverage_ratio": self.p10_finite_return_coverage_ratio,
            "min_finite_return_coverage_ratio": self.min_finite_return_coverage_ratio,
            "median_active_finite_asset_count": self.median_active_finite_asset_count,
            "broad_denominator_count": self.broad_denominator_count,
            "activity_coverage": self.activity_coverage,
            "covariance_correlation_sample_available": bool(self.covariance_correlation_sample_available),
            "covariance_correlation_sample_rows": int(self.covariance_correlation_sample_rows),
            "stable_feature_pool_available": bool(self.stable_feature_pool_available),
            "speculative_feature_pool_available": bool(self.speculative_feature_pool_available),
            "required_feature_coverage": to_jsonable(dict(self.required_feature_coverage)),
            "fixed_universe_required": bool(self.fixed_universe_required),
        }


@dataclass(frozen=True)
class MarketStateCoverageGateResult:
    status: str
    passed: bool
    reason_codes: Sequence[str]
    summary: MarketStateCoverageSummary
    policy: MarketStateCoverageGatePolicy = field(default_factory=MarketStateCoverageGatePolicy)

    def as_dict(self) -> dict[str, Any]:
        return {
            "artifact_kind": "market_state_window_coverage_gate_result",
            "schema_version": MARKET_STATE_WINDOW_PROFILE_SCHEMA_VERSION,
            "status": self.status,
            "passed": bool(self.passed),
            "reason_codes": list(self.reason_codes),
            "summary": self.summary.as_dict(),
            "policy": self.policy.as_dict(),
            "production_writes_enabled": False,
        }


def default_market_state_window_profiles() -> tuple[RegimeWindowProfile, ...]:
    return (
        RegimeWindowProfile(window_profile_id="micro_recent_30d", band="micro", lookback_days=30),
        RegimeWindowProfile(window_profile_id="micro_recent_60d", band="micro", lookback_days=60),
        RegimeWindowProfile(window_profile_id="micro_recent_90d", band="micro", lookback_days=90),
        RegimeWindowProfile(window_profile_id="micro_recent_180d", band="micro", lookback_days=180),
        RegimeWindowProfile(window_profile_id="micro_recent_365d", band="micro", lookback_days=365),
        RegimeWindowProfile(window_profile_id="meso_recent_90d", band="meso", lookback_days=90),
        RegimeWindowProfile(window_profile_id="meso_recent_180d", band="meso", lookback_days=180),
        RegimeWindowProfile(window_profile_id="meso_recent_270d", band="meso", lookback_days=270),
        RegimeWindowProfile(window_profile_id="meso_recent_365d", band="meso", lookback_days=365),
        RegimeWindowProfile(window_profile_id="meso_recent_540d", band="meso", lookback_days=540),
        RegimeWindowProfile(window_profile_id="meso_recent_720d", band="meso", lookback_days=720),
        RegimeWindowProfile(window_profile_id="macro_recent_180d", band="macro", lookback_days=180),
        RegimeWindowProfile(window_profile_id="macro_recent_365d", band="macro", lookback_days=365),
        RegimeWindowProfile(window_profile_id="macro_recent_540d", band="macro", lookback_days=540),
        RegimeWindowProfile(window_profile_id="macro_recent_720d", band="macro", lookback_days=720),
        RegimeWindowProfile(window_profile_id="macro_recent_1080d", band="macro", lookback_days=1080),
        RegimeWindowProfile(
            window_profile_id="macro_recent_all_available_capped_1440d",
            band="macro",
            lookback_days=1440,
            row_cap=1440,
        ),
    )


def market_state_window_profiles_for_band(band: str) -> tuple[RegimeWindowProfile, ...]:
    normalized = str(band).strip().lower()
    return tuple(profile for profile in default_market_state_window_profiles() if profile.band == normalized)


def summarize_market_state_window_coverage(
    frame: pd.DataFrame,
    *,
    market_axis: str,
    band: str,
    window_profile: RegimeWindowProfile,
    required_features: Sequence[str] = (),
) -> MarketStateCoverageSummary:
    work = frame.copy()
    if "ts" in work.columns:
        work["ts"] = pd.to_numeric(work["ts"], errors="coerce")
        work = work.dropna(subset=["ts"]).sort_values("ts").reset_index(drop=True)
    row_count = int(work.shape[0])
    start_ts = int(work["ts"].min()) if row_count and "ts" in work.columns else window_profile.start_ts
    end_ts = int(work["ts"].max()) if row_count and "ts" in work.columns else window_profile.end_ts
    source_tail_ts = _max_numeric(work.get("source_tail_ts"))
    coverage_values = _finite_values(work, _coverage_ratio_columns(work))
    active_values = _finite_values(work, _active_count_columns(work))
    denominator_values = _finite_values(work, _denominator_columns(work))
    activity_values = _finite_values(work, _activity_columns(work))
    required_coverage = {
        str(feature): float(pd.to_numeric(work[feature], errors="coerce").replace([np.inf, -np.inf], np.nan).notna().mean())
        for feature in required_features
        if feature in work.columns and row_count
    }
    corr_columns = _correlation_columns(work)
    corr_rows = 0
    if corr_columns:
        corr_rows = int(
            pd.to_numeric(work[list(corr_columns)].stack(), errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .groupby(level=0)
            .count()
            .gt(0)
            .sum()
        )
    return MarketStateCoverageSummary(
        market_axis=str(market_axis),
        band=str(band).strip().lower(),
        window_profile=window_profile,
        row_count=row_count,
        start_ts=start_ts,
        end_ts=end_ts,
        source_tail_ts=source_tail_ts if source_tail_ts is not None else window_profile.source_tail_ts,
        median_finite_return_coverage_ratio=_percentile(coverage_values, 50),
        p10_finite_return_coverage_ratio=_percentile(coverage_values, 10),
        min_finite_return_coverage_ratio=float(np.min(coverage_values)) if coverage_values else None,
        median_active_finite_asset_count=_percentile(active_values, 50),
        broad_denominator_count=int(np.nanmax(denominator_values)) if denominator_values else None,
        activity_coverage=_percentile(activity_values, 50),
        covariance_correlation_sample_available=bool(corr_rows > 0),
        covariance_correlation_sample_rows=corr_rows,
        stable_feature_pool_available=_pool_available(work, "stable"),
        speculative_feature_pool_available=_pool_available(work, "speculative"),
        required_feature_coverage=required_coverage,
        fixed_universe_required=False,
    )


def evaluate_market_state_window_coverage(
    frame: pd.DataFrame,
    *,
    market_axis: str,
    band: str,
    window_profile: RegimeWindowProfile,
    required_features: Sequence[str] = (),
    policy: MarketStateCoverageGatePolicy | None = None,
) -> MarketStateCoverageGateResult:
    cfg = policy or MarketStateCoverageGatePolicy()
    summary = summarize_market_state_window_coverage(
        frame,
        market_axis=market_axis,
        band=band,
        window_profile=window_profile,
        required_features=required_features,
    )
    reasons: list[str] = []
    missing = [feature for feature in required_features if feature not in frame.columns]
    if missing:
        reasons.append(MARKET_STATE_COVERAGE_MISSING_REQUIRED_FEATURES)
    if int(summary.row_count) < int(cfg.min_rows):
        reasons.append(MARKET_STATE_COVERAGE_INSUFFICIENT_ROWS)
    if (
        summary.median_finite_return_coverage_ratio is not None
        and summary.median_finite_return_coverage_ratio < float(cfg.min_median_finite_return_coverage_ratio)
    ) or (
        summary.p10_finite_return_coverage_ratio is not None
        and summary.p10_finite_return_coverage_ratio < float(cfg.min_p10_finite_return_coverage_ratio)
    ):
        reasons.append(MARKET_STATE_COVERAGE_INSUFFICIENT_FINITE_COVERAGE)
    if summary.activity_coverage is not None and summary.activity_coverage < float(cfg.min_activity_coverage):
        reasons.append(MARKET_STATE_COVERAGE_INSUFFICIENT_ACTIVITY_COVERAGE)
    if _axis_requires_correlation(market_axis) and int(summary.covariance_correlation_sample_rows) < int(cfg.min_correlation_sample_rows):
        reasons.append(MARKET_STATE_COVERAGE_INSUFFICIENT_CORRELATION_SAMPLE)
    if _required_features_low_variance(frame, required_features, threshold=float(cfg.low_variance_threshold)):
        reasons.append(MARKET_STATE_COVERAGE_LOW_VARIANCE_NEAR_FLAT)
    reasons = list(dict.fromkeys(reasons))
    return MarketStateCoverageGateResult(
        status="passed" if not reasons else "blocked",
        passed=not reasons,
        reason_codes=tuple(reasons),
        summary=summary,
        policy=cfg,
    )


def normalize_market_state_coverage_reason(reason_code: str) -> str:
    aliases = {
        MARKET_STATE_COVERAGE_INSUFFICIENT_ROWS: MASK_INSUFFICIENT_HISTORY,
        MARKET_STATE_COVERAGE_LOW_VARIANCE_NEAR_FLAT: MASK_LOW_VARIANCE_NEAR_FLAT,
        MARKET_STATE_COVERAGE_MISSING_REQUIRED_FEATURES: MASK_MISSING_REQUIRED_FEATURES,
    }
    text = str(reason_code).strip().lower()
    return aliases.get(text, text)


def _coverage_ratio_columns(frame: pd.DataFrame) -> tuple[str, ...]:
    preferred = [
        column
        for column in frame.columns
        if str(column).endswith("_coverage_ratio") and not str(column).endswith("_min_coverage_ratio")
    ]
    return tuple(preferred)


def _active_count_columns(frame: pd.DataFrame) -> tuple[str, ...]:
    return tuple(column for column in frame.columns if str(column).endswith("_active_n"))


def _denominator_columns(frame: pd.DataFrame) -> tuple[str, ...]:
    return tuple(column for column in frame.columns if str(column).endswith("_requested_n"))


def _activity_columns(frame: pd.DataFrame) -> tuple[str, ...]:
    return tuple(
        column
        for column in frame.columns
        if "activity" in str(column).lower() and pd.api.types.is_numeric_dtype(frame[column])
    )


def _correlation_columns(frame: pd.DataFrame) -> tuple[str, ...]:
    return tuple(
        column
        for column in frame.columns
        if any(token in str(column).lower() for token in ("corr", "covariance"))
        and pd.api.types.is_numeric_dtype(frame[column])
    )


def _finite_values(frame: pd.DataFrame, columns: Sequence[str]) -> list[float]:
    values: list[float] = []
    for column in columns:
        if column not in frame.columns:
            continue
        series = pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        values.extend(float(value) for value in series.to_numpy(dtype=float))
    return values


def _pool_available(frame: pd.DataFrame, token: str) -> bool:
    columns = [column for column in frame.columns if token in str(column).lower()]
    if not columns:
        return False
    return bool(pd.to_numeric(frame[columns].stack(), errors="coerce").replace([np.inf, -np.inf], np.nan).notna().any())


def _percentile(values: Sequence[float], pct: float) -> float | None:
    clean = [float(value) for value in values if np.isfinite(float(value))]
    if not clean:
        return None
    return float(np.percentile(clean, float(pct)))


def _max_numeric(series: object) -> int | None:
    if series is None:
        return None
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return None
    return int(values.max())


def _axis_requires_correlation(axis: str) -> bool:
    text = str(axis).lower()
    return "correlation" in text or "covariance" in text


def _required_features_low_variance(frame: pd.DataFrame, features: Sequence[str], *, threshold: float) -> bool:
    present = [feature for feature in features if feature in frame.columns]
    if not present or frame.empty:
        return False
    numeric = frame[present].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    variances = numeric.var(axis=0, skipna=True, ddof=0).fillna(0.0)
    return bool((variances <= float(threshold)).all())


__all__ = [
    "MARKET_STATE_COVERAGE_INSUFFICIENT_ACTIVITY_COVERAGE",
    "MARKET_STATE_COVERAGE_INSUFFICIENT_CORRELATION_SAMPLE",
    "MARKET_STATE_COVERAGE_INSUFFICIENT_FINITE_COVERAGE",
    "MARKET_STATE_COVERAGE_INSUFFICIENT_ROWS",
    "MARKET_STATE_COVERAGE_LOW_VARIANCE_NEAR_FLAT",
    "MARKET_STATE_COVERAGE_MISSING_REQUIRED_FEATURES",
    "MARKET_STATE_COVERAGE_REASON_CODES",
    "MARKET_STATE_WINDOW_PROFILE_SCHEMA_VERSION",
    "MarketStateCoverageGatePolicy",
    "MarketStateCoverageGateResult",
    "MarketStateCoverageSummary",
    "default_market_state_window_profiles",
    "evaluate_market_state_window_coverage",
    "market_state_window_profiles_for_band",
    "normalize_market_state_coverage_reason",
    "summarize_market_state_window_coverage",
]
