from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd

from src.regimes.asset_state_test.contracts import FlatPreflightResult


def _safe_ratio(numer: int, denom: int) -> float:
    return float(numer / denom) if int(denom) > 0 else 1.0


def _feature_variance_summary(frame: pd.DataFrame, feature_columns: Sequence[str]) -> dict[str, object]:
    cols = [str(c) for c in feature_columns if str(c) in frame.columns]
    if not cols or frame.empty:
        return {
            "feature_count": int(len(cols)),
            "zero_variance_feature_count": int(len(cols)),
            "near_zero_variance_feature_count": int(len(cols)),
            "min_variance": None,
            "median_variance": None,
            "max_variance": None,
            "per_feature_variance": {},
        }
    numeric = frame[cols].apply(pd.to_numeric, errors="coerce")
    variances = numeric.var(ddof=0).replace([np.inf, -np.inf], np.nan)
    finite = variances.dropna()
    per_feature = {str(k): float(v) for k, v in finite.items()}
    near_zero = finite <= 1e-12
    return {
        "feature_count": int(len(cols)),
        "zero_variance_feature_count": int((finite == 0.0).sum()) + int(variances.isna().sum()),
        "near_zero_variance_feature_count": int(near_zero.sum()) + int(variances.isna().sum()),
        "min_variance": float(finite.min()) if not finite.empty else None,
        "median_variance": float(finite.median()) if not finite.empty else None,
        "max_variance": float(finite.max()) if not finite.empty else None,
        "per_feature_variance": per_feature,
    }


def _zero_movement_summary(frame: pd.DataFrame, feature_columns: Sequence[str]) -> dict[str, object]:
    movement_cols = [
        c
        for c in feature_columns
        if c in frame.columns
        and (
            str(c).endswith("_log_return")
            or "_d_close_" in str(c)
            or str(c).endswith("_roc_14")
            or str(c).endswith("_mom_14")
            or str(c).endswith("_range_hl")
            or str(c).endswith("_range_co")
            or str(c).endswith("_true_range")
            or str(c).endswith("_prr")
        )
    ]
    activity_cols = [
        c
        for c in feature_columns
        if c in frame.columns
        and (
            str(c).endswith("_trade_intensity")
            or str(c).endswith("_avg_trade_size")
            or str(c).endswith("_vroc_14")
            or str(c).endswith("_volume")
            or str(c).endswith("_trades")
        )
    ]
    out: dict[str, object] = {
        "movement_feature_count": int(len(movement_cols)),
        "activity_feature_count": int(len(activity_cols)),
        "zero_movement_fraction": 1.0 if not movement_cols or frame.empty else 0.0,
        "near_zero_movement_fraction": 1.0 if not movement_cols or frame.empty else 0.0,
        "low_activity_fraction": 0.0,
        "per_feature_zero_fraction": {},
    }
    if movement_cols and not frame.empty:
        numeric = frame[movement_cols].apply(pd.to_numeric, errors="coerce").abs()
        zero_mask = numeric <= 0.0
        near_zero_mask = numeric <= 1e-12
        out["zero_movement_fraction"] = float(zero_mask.all(axis=1).mean())
        out["near_zero_movement_fraction"] = float(near_zero_mask.all(axis=1).mean())
        out["per_feature_zero_fraction"] = {str(c): float(zero_mask[c].mean()) for c in movement_cols}
    if activity_cols and not frame.empty:
        activity = frame[activity_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        out["low_activity_fraction"] = float((activity.abs() <= 1e-12).all(axis=1).mean())
    return out


def run_flat_preflight(
    frame: pd.DataFrame,
    *,
    asset: str,
    axis: str,
    band: str,
    feature_columns: Sequence[str],
    train_start_ts: Optional[int] = None,
    train_end_ts: Optional[int] = None,
    max_missing_fraction: float = 0.35,
    min_complete_rows: int = 32,
    min_finite_rows: int = 32,
    min_nonzero_variance_features: int = 2,
    near_flat_fraction_threshold: float = 0.98,
    low_activity_fraction_threshold: float = 0.98,
) -> FlatPreflightResult:
    cols = [str(c) for c in feature_columns if str(c) in frame.columns]
    row_count = int(len(frame))
    if row_count == 0 or not cols:
        reason = "bad_or_missing_data"
        pass_flag = False
        return FlatPreflightResult(
            asset=str(asset),
            axis=str(axis),
            band=str(band),
            train_start_ts=train_start_ts,
            train_end_ts=train_end_ts,
            row_count=row_count,
            complete_row_count=0,
            finite_row_count=0,
            missing_fraction=1.0,
            variance_summary=_feature_variance_summary(frame, cols),
            zero_movement_summary=_zero_movement_summary(frame, cols),
            reason_code=reason,
            pass_flag=pass_flag,
            clusterable_candidate=False,
            included_in_fit=False,
            carried_as="not_clusterable",
            affected_axis_band=f"{axis}/{band}",
            near_flat_fraction_threshold=float(near_flat_fraction_threshold),
            near_flat_distance_to_threshold=None,
            zero_variance_feature_count=int(len(cols)),
            near_zero_variance_feature_count=int(len(cols)),
            near_zero_movement_fraction=None,
        )

    numeric = frame[cols].apply(pd.to_numeric, errors="coerce")
    complete_mask = numeric.notna().all(axis=1)
    finite_mask = np.isfinite(numeric.to_numpy(dtype=float)).all(axis=1)
    complete_row_count = int(complete_mask.sum())
    finite_row_count = int(finite_mask.sum())
    missing_fraction = float(numeric.isna().to_numpy().mean()) if row_count > 0 else 1.0
    variance_summary = _feature_variance_summary(frame.loc[finite_mask].copy(), cols)
    zero_summary = _zero_movement_summary(frame.loc[finite_mask].copy(), cols)
    feature_count = int(variance_summary.get("feature_count", 0) or 0)
    near_zero_count = int(variance_summary.get("near_zero_variance_feature_count", feature_count) or 0)
    zero_variance_count = int(variance_summary.get("zero_variance_feature_count", feature_count) or 0)
    nonzero_variance_features = int(max(0, feature_count - near_zero_count))
    near_flat_fraction = float(zero_summary.get("near_zero_movement_fraction", 0.0) or 0.0)
    low_activity_fraction = float(zero_summary.get("low_activity_fraction", 0.0) or 0.0)

    if missing_fraction > float(max_missing_fraction) or complete_row_count < int(min_complete_rows):
        reason = "bad_or_missing_data"
        carried_as = "not_clusterable"
    elif finite_row_count < int(min_finite_rows):
        reason = "bad_or_missing_data"
        carried_as = "not_clusterable"
    elif low_activity_fraction >= float(low_activity_fraction_threshold):
        reason = "low_activity"
        carried_as = "not_clusterable"
    elif near_flat_fraction >= float(near_flat_fraction_threshold):
        reason = "valid_flat_or_pegged"
        carried_as = "neutral_flat"
    elif nonzero_variance_features < int(min_nonzero_variance_features):
        reason = "insufficient_variance"
        carried_as = "not_clusterable"
    else:
        reason = "pass"
        carried_as = "cluster_candidate"

    pass_flag = reason == "pass"
    return FlatPreflightResult(
        asset=str(asset),
        axis=str(axis),
        band=str(band),
        train_start_ts=train_start_ts,
        train_end_ts=train_end_ts,
        row_count=row_count,
        complete_row_count=complete_row_count,
        finite_row_count=finite_row_count,
        missing_fraction=missing_fraction,
        variance_summary=variance_summary,
        zero_movement_summary=zero_summary,
        reason_code=reason,
        pass_flag=pass_flag,
        clusterable_candidate=pass_flag,
        included_in_fit=pass_flag,
        carried_as=carried_as,
        affected_axis_band=f"{axis}/{band}",
        near_flat_fraction_threshold=float(near_flat_fraction_threshold),
        near_flat_distance_to_threshold=float(float(near_flat_fraction_threshold) - near_flat_fraction),
        zero_variance_feature_count=zero_variance_count,
        near_zero_variance_feature_count=near_zero_count,
        near_zero_movement_fraction=near_flat_fraction,
    )
