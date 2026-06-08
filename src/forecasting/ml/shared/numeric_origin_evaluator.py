from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class OriginEvaluationInput:
    dataset: Any
    params: Dict[str, Any]
    idx_origin: int
    origin_ts: int
    origin_index: int
    y_hist: np.ndarray
    x_hist: Optional[np.ndarray] = None
    x_last: Optional[np.ndarray] = None
    factor_hist: Optional[np.ndarray] = None
    factor_last: Optional[float] = None


@dataclass(frozen=True)
class OriginEvaluationPayload:
    predictions: Tuple[float, ...]
    actuals: Tuple[float, ...]
    prediction_timestamps: Tuple[int, ...]

    @property
    def rows(self) -> int:
        return int(len(self.predictions))


def metric_values(payload: OriginEvaluationPayload) -> Tuple[Optional[float], Optional[float]]:
    if payload.rows <= 0:
        return None, None
    pred = np.asarray(payload.predictions, dtype=float)
    act = np.asarray(payload.actuals, dtype=float)
    return float(np.sqrt(np.mean((pred - act) ** 2))), float(np.mean(np.abs(pred - act)))


def evaluate_origin_predictions(
    *,
    dataset: Any,
    params: Dict[str, Any],
    ts_vec: np.ndarray,
    y_vec: np.ndarray,
    origins: Sequence[int],
    predict_origin: Callable[[OriginEvaluationInput], Dict[float, float]],
    feat_matrix: Optional[np.ndarray] = None,
    use_dynamic_features: bool = False,
    effective_history_bars: Optional[int] = None,
    trailing_history: bool = False,
    require_any_finite_feature: bool = False,
    needs_factor_cache: bool = False,
    factor_map: Optional[Dict[int, float]] = None,
    factor_values: Optional[np.ndarray] = None,
    min_valid_targets: int = 48,
) -> OriginEvaluationPayload:
    predictions: List[float] = []
    actuals: List[float] = []
    pred_ts: List[int] = []
    full_valid_target_idx = np.flatnonzero(np.isfinite(y_vec))
    for idx_origin, origin_ts in enumerate(origins):
        idx = int(np.searchsorted(ts_vec, int(origin_ts), side="right") - 1)
        if idx < 0:
            continue
        if trailing_history:
            y_hist_full = y_vec[: idx + 1]
            valid_target_idx = np.flatnonzero(np.isfinite(y_hist_full))
            if int(valid_target_idx.size) < int(min_valid_targets):
                continue
            bars = int(effective_history_bars) if effective_history_bars is not None else int(valid_target_idx.size)
            hist_idx = valid_target_idx[-bars:]
            y_hist = y_hist_full[hist_idx]
            feature_idx = hist_idx
        else:
            valid_pos = int(np.searchsorted(full_valid_target_idx, int(idx), side="right")) - 1
            if valid_pos < int(min_valid_targets) - 1:
                continue
            hist_idx = full_valid_target_idx[: valid_pos + 1]
            y_hist = y_vec[hist_idx]
            feature_idx = hist_idx
        x_hist = None
        x_last = None
        if bool(use_dynamic_features):
            if feat_matrix is None:
                continue
            if trailing_history:
                fmat = feat_matrix[: idx + 1][feature_idx]
            else:
                fmat = feat_matrix[feature_idx]
            if bool(require_any_finite_feature) and not np.isfinite(fmat).any():
                continue
            med = np.nanmedian(fmat, axis=0)
            fmat = np.where(np.isfinite(fmat), fmat, med)
            x_hist = fmat
            x_last = fmat[-1]
        factor_hist = None
        factor_last = None
        if bool(needs_factor_cache):
            ts_hist = ts_vec[hist_idx]
            if factor_values is not None:
                values = np.asarray(factor_values[hist_idx], dtype=float)
            else:
                values = np.asarray([(factor_map or {}).get(int(ts), np.nan) for ts in ts_hist], dtype=float)
            if not np.isfinite(values).any():
                continue
            med = float(np.nanmedian(values)) if np.isfinite(values).any() else 0.0
            factor_hist = np.where(np.isfinite(values), values, med)
            factor_last = float(factor_hist[-1])
        try:
            qvals = predict_origin(
                OriginEvaluationInput(
                    dataset=dataset,
                    params=dict(params),
                    idx_origin=int(idx_origin),
                    origin_ts=int(origin_ts),
                    origin_index=int(idx),
                    y_hist=y_hist,
                    x_hist=x_hist,
                    x_last=x_last,
                    factor_hist=factor_hist,
                    factor_last=factor_last,
                )
            )
        except Exception:
            continue
        y_true = y_vec[idx]
        if not np.isfinite(float(y_true)):
            continue
        predictions.append(float(qvals.get(0.5, np.nan)))
        actuals.append(float(y_true))
        pred_ts.append(int(origin_ts))
    return OriginEvaluationPayload(
        predictions=tuple(predictions),
        actuals=tuple(actuals),
        prediction_timestamps=tuple(pred_ts),
    )
