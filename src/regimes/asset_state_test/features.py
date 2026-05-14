from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import RobustScaler, StandardScaler


FEATURE_POOLS: dict[str, tuple[str, ...]] = {
    "trend": (
        "log_return",
        "macd_hist_12_26_9",
        "rsi_14",
        "adx_14",
        "aroon_osc_25",
        "plus_di_14",
        "minus_di_14",
        "vi_plus_14",
        "vi_minus_14",
        "roc_14",
        "mom_14",
        "d_close_2",
        "d_close_3",
        "d_close_5",
        "d_close_10",
        "d_close_14",
        "d_close_20",
        "range_efficiency",
    ),
    "vol": (
        "atr_14",
        "ret_std_20",
        "cv_20",
        "vol_osc_pct_14_28",
        "true_range",
        "range_hl",
        "range_co",
        "var_20",
        "skew_20",
        "kurt_20",
        "q25_20",
        "q75_20",
        "squeeze_scalar",
    ),
    "activity": (
        "trade_intensity",
        "avg_trade_size",
        "vroc_14",
        "prr",
        "volume",
        "trades",
        "obv",
        "adl",
        "force_index",
        "chaikin_osc_3_10",
        "vpt",
        "eom_14",
        "pvi",
        "nvi",
    ),
}


@dataclass(frozen=True)
class PreprocessResult:
    x: np.ndarray
    clean_frame: pd.DataFrame
    selected_columns: tuple[str, ...]
    dropped_columns: tuple[str, ...]
    scaler_name: str
    variance_threshold: float
    selector: object | None = field(default=None, repr=False, compare=False)
    scaler: object | None = field(default=None, repr=False, compare=False)
    clip_bounds: dict[str, tuple[float, float]] = field(default_factory=dict)

    def to_metadata(self) -> dict[str, object]:
        return {
            "selected_columns": list(self.selected_columns),
            "dropped_columns": list(self.dropped_columns),
            "scaler": self.scaler_name,
            "variance_threshold": float(self.variance_threshold),
            "rows_after_dropna": int(len(self.clean_frame)),
            "feature_count": int(len(self.selected_columns)),
            "clipper": "winsor_p01_p99" if self.clip_bounds else None,
            "clip_bounds": {str(k): [float(v[0]), float(v[1])] for k, v in sorted(self.clip_bounds.items())},
        }


def feature_pool(axis: str, strategy: str = "manual_baseline") -> tuple[str, ...]:
    axis = str(axis)
    if axis not in FEATURE_POOLS:
        raise ValueError(f"Unsupported feature pool axis {axis!r}")
    if str(strategy) in {"manual_baseline", "variance_threshold_then_manual"}:
        if axis == "trend":
            return ("log_return", "macd_hist_12_26_9", "rsi_14", "adx_14")
        if axis == "vol":
            return ("atr_14", "ret_std_20", "cv_20", "vol_osc_pct_14_28")
        return ("trade_intensity", "avg_trade_size", "vroc_14", "prr")
    if str(strategy) == "trend_return_macd_compact":
        if axis != "trend":
            raise ValueError("trend_return_macd_compact is only supported for the trend axis")
        return ("log_return", "macd_hist_12_26_9")
    if str(strategy) == "trend_directional_compact":
        if axis != "trend":
            raise ValueError("trend_directional_compact is only supported for the trend axis")
        return ("log_return", "macd_hist_12_26_9", "roc_14", "mom_14", "range_efficiency")
    return FEATURE_POOLS[axis]


def select_feature_columns(
    frame: pd.DataFrame,
    *,
    axis: str,
    member_intervals: Sequence[int],
    feature_bases: Sequence[str] | None = None,
    strategy: str = "manual_baseline",
) -> tuple[str, ...]:
    bases = tuple(str(v) for v in (feature_bases or feature_pool(axis, strategy)))
    selected: list[str] = []
    for interval in member_intervals:
        prefix = f"i{int(interval)}_"
        for base in bases:
            col = f"{prefix}{base}"
            if col in frame.columns:
                selected.append(col)
    return tuple(dict.fromkeys(selected))


def preprocess_feature_frame(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
    *,
    preprocess: str = "robust_scale",
    variance_threshold: float = 1e-12,
) -> PreprocessResult:
    columns = tuple(str(c) for c in feature_columns)
    if not columns:
        return PreprocessResult(
            x=np.empty((0, 0), dtype=float),
            clean_frame=frame.iloc[0:0].copy(),
            selected_columns=(),
            dropped_columns=(),
            scaler_name=str(preprocess),
            variance_threshold=float(variance_threshold),
            selector=None,
            scaler=None,
        )
    clean = frame.dropna(subset=list(columns)).copy()
    if clean.empty:
        return PreprocessResult(
            x=np.empty((0, len(columns)), dtype=float),
            clean_frame=clean,
            selected_columns=columns,
            dropped_columns=(),
            scaler_name=str(preprocess),
            variance_threshold=float(variance_threshold),
            selector=None,
            scaler=None,
        )
    numeric = clean[list(columns)].astype(float)
    finite_mask = np.isfinite(numeric.to_numpy(dtype=float)).all(axis=1)
    clean = clean.loc[finite_mask].copy()
    numeric = numeric.loc[finite_mask]
    if clean.empty:
        return PreprocessResult(
            x=np.empty((0, len(columns)), dtype=float),
            clean_frame=clean,
            selected_columns=columns,
            dropped_columns=(),
            scaler_name=str(preprocess),
            variance_threshold=float(variance_threshold),
            selector=None,
            scaler=None,
        )
    clip_bounds: dict[str, tuple[float, float]] = {}
    if str(preprocess) == "winsor_p01_p99":
        for column in columns:
            values = numeric[column].to_numpy(dtype=float)
            lower = float(np.quantile(values, 0.01))
            upper = float(np.quantile(values, 0.99))
            clip_bounds[str(column)] = (lower, upper)
            numeric.loc[:, column] = np.clip(values, lower, upper)
    selector = VarianceThreshold(threshold=float(variance_threshold))
    try:
        selected_values = selector.fit_transform(numeric.to_numpy(dtype=float))
        keep_mask = selector.get_support()
    except ValueError:
        keep_mask = np.zeros(len(columns), dtype=bool)
        selected_values = np.empty((len(clean), 0), dtype=float)
    selected_columns = tuple(c for c, keep in zip(columns, keep_mask) if bool(keep))
    dropped_columns = tuple(c for c, keep in zip(columns, keep_mask) if not bool(keep))
    if selected_values.shape[1] == 0:
        return PreprocessResult(
            x=selected_values,
            clean_frame=clean,
            selected_columns=selected_columns,
            dropped_columns=dropped_columns,
            scaler_name=str(preprocess),
            variance_threshold=float(variance_threshold),
            selector=selector,
            scaler=None,
            clip_bounds=clip_bounds,
        )
    scaler = StandardScaler() if str(preprocess) == "standard_scale" else RobustScaler()
    x = scaler.fit_transform(selected_values)
    return PreprocessResult(
        x=np.asarray(x, dtype=float),
        clean_frame=clean,
        selected_columns=selected_columns,
        dropped_columns=dropped_columns,
        scaler_name=type(scaler).__name__,
        variance_threshold=float(variance_threshold),
        selector=selector,
        scaler=scaler,
        clip_bounds=clip_bounds,
    )


def transform_feature_frame(frame: pd.DataFrame, feature_columns: Sequence[str], fitted: PreprocessResult) -> PreprocessResult:
    columns = tuple(str(c) for c in feature_columns)
    selected_columns = tuple(str(c) for c in fitted.selected_columns)
    if not columns or not selected_columns:
        return PreprocessResult(
            x=np.empty((0, len(selected_columns)), dtype=float),
            clean_frame=frame.iloc[0:0].copy(),
            selected_columns=selected_columns,
            dropped_columns=tuple(str(c) for c in fitted.dropped_columns),
            scaler_name=fitted.scaler_name,
            variance_threshold=float(fitted.variance_threshold),
            selector=fitted.selector,
            scaler=fitted.scaler,
            clip_bounds=dict(fitted.clip_bounds),
        )
    missing = [c for c in selected_columns if c not in frame.columns]
    if missing:
        clean = frame.iloc[0:0].copy()
        return PreprocessResult(
            x=np.empty((0, len(selected_columns)), dtype=float),
            clean_frame=clean,
            selected_columns=selected_columns,
            dropped_columns=tuple(str(c) for c in fitted.dropped_columns),
            scaler_name=fitted.scaler_name,
            variance_threshold=float(fitted.variance_threshold),
            selector=fitted.selector,
            scaler=fitted.scaler,
            clip_bounds=dict(fitted.clip_bounds),
        )
    clean = frame.dropna(subset=list(selected_columns)).copy()
    if clean.empty:
        return PreprocessResult(
            x=np.empty((0, len(selected_columns)), dtype=float),
            clean_frame=clean,
            selected_columns=selected_columns,
            dropped_columns=tuple(str(c) for c in fitted.dropped_columns),
            scaler_name=fitted.scaler_name,
            variance_threshold=float(fitted.variance_threshold),
            selector=fitted.selector,
            scaler=fitted.scaler,
            clip_bounds=dict(fitted.clip_bounds),
        )
    numeric = clean[list(selected_columns)].astype(float)
    finite_mask = np.isfinite(numeric.to_numpy(dtype=float)).all(axis=1)
    clean = clean.loc[finite_mask].copy()
    numeric = numeric.loc[finite_mask]
    if clean.empty:
        return PreprocessResult(
            x=np.empty((0, len(selected_columns)), dtype=float),
            clean_frame=clean,
            selected_columns=selected_columns,
            dropped_columns=tuple(str(c) for c in fitted.dropped_columns),
            scaler_name=fitted.scaler_name,
            variance_threshold=float(fitted.variance_threshold),
            selector=fitted.selector,
            scaler=fitted.scaler,
            clip_bounds=dict(fitted.clip_bounds),
        )
    values = numeric.to_numpy(dtype=float).copy()
    if fitted.clip_bounds:
        for idx, column in enumerate(selected_columns):
            bounds = fitted.clip_bounds.get(str(column))
            if bounds is not None:
                values[:, idx] = np.clip(values[:, idx], float(bounds[0]), float(bounds[1]))
    if fitted.scaler is not None:
        values = fitted.scaler.transform(values)
    return PreprocessResult(
        x=np.asarray(values, dtype=float),
        clean_frame=clean,
        selected_columns=selected_columns,
        dropped_columns=tuple(str(c) for c in fitted.dropped_columns),
        scaler_name=fitted.scaler_name,
        variance_threshold=float(fitted.variance_threshold),
        selector=fitted.selector,
        scaler=fitted.scaler,
        clip_bounds=dict(fitted.clip_bounds),
    )
