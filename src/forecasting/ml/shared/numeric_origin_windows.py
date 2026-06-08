from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ProductionOriginArrays:
    ts_vec: np.ndarray
    y_vec: np.ndarray
    valid_target_idx: np.ndarray
    feat_cols: tuple[str, ...]
    feat_matrix: Optional[np.ndarray]
    factor_values: Optional[np.ndarray]
    array_builds: int = 1
    array_reuses: int = 0


@dataclass(frozen=True)
class ProductionOriginWindow:
    origin_ts: int
    valid_pos: int
    hist_idx: np.ndarray
    y_hist: np.ndarray
    ts_hist: np.ndarray
    x_hist: Optional[np.ndarray]
    x_last: Optional[np.ndarray]
    factor_hist: Optional[np.ndarray]
    factor_last: Optional[float]
    actual_value: float
    fit_bars: int

    @property
    def train_start_ts(self) -> int:
        return int(self.ts_hist[0])

    @property
    def train_end_ts(self) -> int:
        return int(self.ts_hist[-1])


@dataclass(frozen=True)
class ProductionOriginWindowResult:
    window: Optional[ProductionOriginWindow]
    reason: str = ""
    valid_pos: Optional[int] = None


def build_production_origin_arrays(
    *,
    frame: pd.DataFrame,
    target_col: str,
    selected_feature_columns: Sequence[str],
    use_dynamic_features: bool,
    as_float_array: Callable[[Any], np.ndarray],
    float_dtype: Any,
    factor_map: Optional[Dict[int, float]] = None,
    needs_factor_cache: bool = False,
    coerce_ts: bool = True,
) -> ProductionOriginArrays:
    if bool(coerce_ts):
        ts_vec = pd.to_numeric(frame["ts"], errors="coerce").fillna(-1).astype("int64").to_numpy()
    else:
        ts_vec = frame["ts"].astype("int64").to_numpy()
    y_vec = pd.to_numeric(frame[target_col], errors="coerce").to_numpy(dtype=float_dtype)
    valid_target_idx = np.flatnonzero(np.isfinite(y_vec))
    feat_cols = [str(col) for col in selected_feature_columns if str(col) in frame.columns]
    feat_matrix = None
    if bool(use_dynamic_features) and feat_cols:
        feat_frame = frame.loc[:, feat_cols].apply(pd.to_numeric, errors="coerce")
        feat_cols = [str(col) for col in feat_cols if feat_frame[str(col)].notna().any()]
        if feat_cols:
            feat_matrix = as_float_array(feat_frame.loc[:, feat_cols].to_numpy(dtype=float_dtype))
    factor_values = None
    if bool(needs_factor_cache):
        factor_values = as_float_array([(factor_map or {}).get(int(t), np.nan) for t in ts_vec])
    return ProductionOriginArrays(
        ts_vec=ts_vec,
        y_vec=y_vec,
        valid_target_idx=valid_target_idx,
        feat_cols=tuple(feat_cols),
        feat_matrix=feat_matrix,
        factor_values=factor_values,
    )


def prepare_production_origin_window(
    *,
    arrays: ProductionOriginArrays,
    idx: int,
    min_history_bars: int,
    history_bars: int,
    use_dynamic_features: bool,
    needs_factor_cache: bool,
    as_float_array: Callable[[Any], np.ndarray],
) -> ProductionOriginWindowResult:
    idx = int(idx)
    valid_pos = int(np.searchsorted(arrays.valid_target_idx, int(idx), side="right")) - 1
    if valid_pos < int(min_history_bars) - 1:
        return ProductionOriginWindowResult(window=None, reason="insufficient_valid_history", valid_pos=int(valid_pos))
    hist_start = max(0, valid_pos - int(history_bars) + 1)
    hist_idx = arrays.valid_target_idx[hist_start : valid_pos + 1]
    y_hist = as_float_array(arrays.y_vec[hist_idx])
    ts_hist = arrays.ts_vec[hist_idx]
    x_hist = None
    x_last = None
    if bool(use_dynamic_features):
        if arrays.feat_matrix is None:
            return ProductionOriginWindowResult(window=None, reason="missing_feature_matrix", valid_pos=int(valid_pos))
        fmat = arrays.feat_matrix[hist_idx]
        if not np.isfinite(fmat).any():
            return ProductionOriginWindowResult(window=None, reason="nonfinite_feature_history", valid_pos=int(valid_pos))
        med = np.nanmedian(fmat, axis=0)
        fmat = np.where(np.isfinite(fmat), fmat, med)
        x_hist = as_float_array(fmat)
        x_last = x_hist[-1]
    factor_hist = None
    factor_last = None
    if bool(needs_factor_cache):
        if arrays.factor_values is None:
            return ProductionOriginWindowResult(window=None, reason="missing_factor_cache", valid_pos=int(valid_pos))
        fh = arrays.factor_values[hist_idx]
        if not np.isfinite(fh).any():
            return ProductionOriginWindowResult(window=None, reason="nonfinite_factor_history", valid_pos=int(valid_pos))
        med_f = float(np.nanmedian(fh)) if np.isfinite(fh).any() else 0.0
        fh = np.where(np.isfinite(fh), fh, med_f)
        factor_hist = as_float_array(fh)
        factor_last = float(factor_hist[-1])
    return ProductionOriginWindowResult(
        valid_pos=int(valid_pos),
        window=ProductionOriginWindow(
            origin_ts=int(arrays.ts_vec[idx]),
            valid_pos=int(valid_pos),
            hist_idx=hist_idx,
            y_hist=y_hist,
            ts_hist=ts_hist,
            x_hist=x_hist,
            x_last=x_last,
            factor_hist=factor_hist,
            factor_last=factor_last,
            actual_value=float(arrays.y_vec[idx]),
            fit_bars=int(y_hist.size),
        )
    )
