from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.forecasting.common.forecast_family_core import monotonic_quantiles, quantiles_from_samples, robust_sigma


def _safe_arr(y: Sequence[float]) -> np.ndarray:
    arr = np.asarray(y, dtype=float)
    return arr[np.isfinite(arr)]


def _draw_normal(mean: float, sigma: float, n: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    return rng.normal(float(mean), max(1e-8, float(sigma)), int(n))


def _draw_student_t(mean: float, sigma: float, df: float, n: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    tail_df = max(2.1, float(df))
    scale = max(1e-8, float(sigma)) * math.sqrt((tail_df - 2.0) / tail_df)
    return float(mean) + rng.standard_t(tail_df, int(n)) * scale


def _solve_linear_beta(X: np.ndarray, y: np.ndarray, ridge: float = 0.0) -> np.ndarray:
    XtX = X.T @ X
    rhs = X.T @ y
    if float(ridge) != 0.0:
        XtX = XtX + float(ridge) * np.eye(XtX.shape[0], dtype=float)
    try:
        return np.linalg.solve(XtX, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(XtX) @ rhs


def bayes_dlm_tvp_predict(
    y_hist: Sequence[float],
    horizon_bars: int,
    quantiles: Sequence[float],
    seasonal_period_bars: Optional[int],
    seed: int = 42,
    level_smoothing: float = 0.08,
    trend_smoothing: float = 0.08,
    seasonal_strength: float = 1.0,
    observation_scale: float = 1.0,
    X_hist: Optional[np.ndarray] = None,
    x_last: Optional[np.ndarray] = None,
    exogenous_scale: float = 0.35,
) -> Tuple[Dict[float, float], Dict[str, Any]]:
    y = _safe_arr(y_hist)
    if y.size < 16:
        raise RuntimeError("insufficient_history")

    level_alpha = min(0.5, max(0.01, float(level_smoothing)))
    trend_alpha = min(0.5, max(0.01, float(trend_smoothing)))
    level = float(y[0])
    trend = 0.0
    residuals: List[float] = []
    for i in range(1, y.size):
        prev_level = level
        projected = level + trend
        level = level_alpha * float(y[i]) + (1 - level_alpha) * projected
        trend = trend_alpha * (level - prev_level) + (1 - trend_alpha) * trend
        residuals.append(float(y[i] - projected))

    seasonal = 0.0
    if seasonal_period_bars is not None and int(seasonal_period_bars) > 1 and y.size > int(seasonal_period_bars) * 2:
        p = int(seasonal_period_bars)
        seasonal = float(y[-p] - np.mean(y[-2 * p : -p])) * max(0.0, float(seasonal_strength))

    pred = float(level + trend * int(horizon_bars) + seasonal)
    uses_exogenous = False
    if X_hist is not None and x_last is not None:
        X = np.asarray(X_hist, dtype=float)
        x0 = np.asarray(x_last, dtype=float).reshape(-1)
        if X.ndim == 2 and X.shape[0] == y.size and X.shape[1] == x0.size and X.shape[0] >= 16:
            x_med = np.nanmedian(X, axis=0)
            X = np.where(np.isfinite(X), X, x_med)
            x_mean = np.nanmean(X, axis=0)
            x_std = np.nanstd(X, axis=0)
            x_std = np.where(x_std < 1e-8, 1.0, x_std)
            Xn = (X - x_mean) / x_std
            xn = (np.where(np.isfinite(x0), x0, x_med) - x_mean) / x_std
            beta = _solve_linear_beta(Xn, y, ridge=1e-3)
            reg_pred = float(xn @ beta)
            pred = float((1.0 - max(0.0, min(0.9, float(exogenous_scale)))) * pred + max(0.0, min(0.9, float(exogenous_scale))) * reg_pred)
            uses_exogenous = True
    sigma = max(1e-6, robust_sigma(np.asarray(residuals[-max(32, int(math.sqrt(y.size))) :]))) * max(0.25, float(observation_scale))
    draws = _draw_normal(pred, sigma * math.sqrt(max(1, int(horizon_bars))), n=512, seed=seed)
    q = monotonic_quantiles(quantiles_from_samples(draws, quantiles), quantiles)
    return q, {"sigma": sigma, "trend": trend, "seasonal_adjust": seasonal, "level_smoothing": level_alpha, "trend_smoothing": trend_alpha, "uses_exogenous_features": bool(uses_exogenous)}


def bayes_stochastic_vol_predict(
    y_hist: Sequence[float],
    horizon_bars: int,
    quantiles: Sequence[float],
    seed: int = 42,
    persistence: float = 0.94,
    horizon_vol_scale: float = 0.15,
    innovation_scale: float = 1.0,
    heavy_tail_df: Optional[float] = None,
) -> Tuple[Dict[float, float], Dict[str, Any]]:
    y = _safe_arr(y_hist)
    if y.size < 32:
        raise RuntimeError("insufficient_history")

    y_pos = np.maximum(np.abs(y), 1e-10)
    z = np.log(y_pos)
    lam = min(0.995, max(0.50, float(persistence)))
    m = float(z[0])
    v = 0.0
    innovation = max(0.25, float(innovation_scale))
    for zi in z[1:]:
        m = lam * m + (1 - lam) * float(zi)
        v = lam * v + (1 - lam) * float((zi - m) ** 2)

    h = max(1, int(horizon_bars))
    pred_log = m
    pred_var = max(1e-8, v * (1.0 + max(0.0, float(horizon_vol_scale)) * h) * (innovation ** 2))
    draw_sigma = math.sqrt(pred_var)
    if heavy_tail_df is not None and float(heavy_tail_df) > 2.0:
        draws = np.exp(_draw_student_t(pred_log, draw_sigma, float(heavy_tail_df), n=768, seed=seed))
    else:
        draws = np.exp(_draw_normal(pred_log, draw_sigma, n=768, seed=seed))
    q = monotonic_quantiles(quantiles_from_samples(draws, quantiles), quantiles)
    return q, {"log_mean": pred_log, "log_var": pred_var, "persistence": lam, "heavy_tail_df": heavy_tail_df}


def bayes_regime_switch_predict(
    y_hist: Sequence[float],
    horizon_bars: int,
    quantiles: Sequence[float],
    seed: int = 42,
) -> Tuple[Dict[float, float], Dict[str, Any]]:
    y = _safe_arr(y_hist)
    if y.size < 64:
        raise RuntimeError("insufficient_history")

    dy = np.diff(y)
    rv = np.abs(dy)
    thr = float(np.median(rv) + 0.5 * np.std(rv))
    reg = (rv > thr).astype(int)
    if reg.size < 8:
        raise RuntimeError("insufficient_regimes")

    p11 = float(np.mean((reg[:-1] == 1) & (reg[1:] == 1)))
    p00 = float(np.mean((reg[:-1] == 0) & (reg[1:] == 0)))
    p11 = min(0.98, max(0.02, p11))
    p00 = min(0.98, max(0.02, p00))

    low = dy[reg == 0]
    high = dy[reg == 1]
    low_mu, low_sd = float(np.mean(low)) if low.size else 0.0, float(np.std(low)) if low.size else 1e-4
    high_mu, high_sd = float(np.mean(high)) if high.size else 0.0, float(np.std(high)) if high.size else 2e-4

    cur = int(reg[-1])
    h = max(1, int(horizon_bars))
    means = []
    sds = []
    for _ in range(h):
        if cur == 1:
            means.append(high_mu)
            sds.append(max(1e-6, high_sd))
            cur = 1 if p11 >= 0.5 else 0
        else:
            means.append(low_mu)
            sds.append(max(1e-6, low_sd))
            cur = 0 if p00 >= 0.5 else 1

    pred = float(y[-1] + np.sum(means))
    sigma = float(np.sqrt(np.sum(np.square(sds))))
    draws = _draw_normal(pred, sigma, n=640, seed=seed)
    q = monotonic_quantiles(quantiles_from_samples(draws, quantiles), quantiles)
    return q, {"threshold": thr, "p11": p11, "p00": p00}


def _ridge_fit_predict(X: np.ndarray, y: np.ndarray, x_last: np.ndarray, lam: float) -> Tuple[float, np.ndarray]:
    beta = _solve_linear_beta(X, y, ridge=float(lam))
    pred = float(x_last @ beta)
    return pred, beta


def bayes_dynamic_regression_shrinkage_predict(
    y_hist: Sequence[float],
    X_hist: np.ndarray,
    x_last: np.ndarray,
    quantiles: Sequence[float],
    seed: int = 42,
    global_shrinkage: float = 1.0,
    slab_scale: float = 1.0,
    feature_corr_weight: float = 1.0,
    coefficient_drift_scale: float = 0.0,
) -> Tuple[Dict[float, float], Dict[str, Any]]:
    y = _safe_arr(y_hist)
    if y.size < 64:
        raise RuntimeError("insufficient_history")
    if X_hist.ndim != 2 or X_hist.shape[0] != y.size:
        raise RuntimeError("bad_feature_matrix")

    X = np.asarray(X_hist, dtype=float)
    x0 = np.asarray(x_last, dtype=float).reshape(-1)
    if X.shape[1] != x0.size:
        raise RuntimeError("feature_dim_mismatch")

    X_mean = np.nanmean(X, axis=0)
    X_std = np.nanstd(X, axis=0)
    X_std = np.where(X_std < 1e-8, 1.0, X_std)
    Xn = np.where(np.isfinite(X), (X - X_mean) / X_std, 0.0)
    xn = np.where(np.isfinite(x0), (x0 - X_mean) / X_std, 0.0)

    corr = np.abs(np.nan_to_num(np.corrcoef(np.column_stack([Xn, y]), rowvar=False)[-1, :-1], nan=0.0))
    shrink = max(0.05, float(global_shrinkage))
    corr_weight = max(0.0, float(feature_corr_weight))
    slab = max(0.25, float(slab_scale))
    lam = shrink * (0.2 + corr_weight * (1.0 - np.clip(corr, 0.0, 1.0))) / slab
    lam_scalar = float(np.median(lam))

    pred, beta = _ridge_fit_predict(Xn, y, xn, lam=lam_scalar)
    fitted = Xn @ beta
    resid = y - fitted
    sigma = max(1e-6, robust_sigma(resid))
    if Xn.shape[0] >= 2:
        recent_drift = float(np.mean(np.abs(Xn[-1] - Xn[-2])))
        sigma *= (1.0 + max(0.0, float(coefficient_drift_scale)) * recent_drift)
    draws = _draw_normal(pred, sigma, n=512, seed=seed)
    q = monotonic_quantiles(quantiles_from_samples(draws, quantiles), quantiles)

    return q, {"lambda": lam_scalar, "active_coeffs": int(np.sum(np.abs(beta) > 1e-4)), "global_shrinkage": shrink, "slab_scale": slab}


def bayes_copula_dependency_predict(
    y_hist: Sequence[float],
    factor_hist: Sequence[float],
    factor_last: float,
    quantiles: Sequence[float],
    seed: int = 42,
    dependence_regularization: float = 1e-5,
    factor_weight_scale: float = 1.0,
    marginal_vol_scale: float = 1.0,
    tail_df: Optional[float] = None,
) -> Tuple[Dict[float, float], Dict[str, Any]]:
    y = _safe_arr(y_hist)
    f = np.asarray(factor_hist, dtype=float)
    if y.size < 48:
        raise RuntimeError("insufficient_history")
    if f.size != y.size:
        raise RuntimeError("factor_mismatch")

    f = np.where(np.isfinite(f), f, np.nan)
    mask = np.isfinite(f)
    if int(mask.sum()) < 32:
        raise RuntimeError("insufficient_factor_rows")
    yy = y[mask]
    ff = f[mask]

    reg = max(1e-8, float(dependence_regularization))
    X = np.column_stack([np.ones_like(ff), ff])
    beta = _solve_linear_beta(X, yy, ridge=reg)
    beta = beta.copy()
    beta[1] *= max(0.0, float(factor_weight_scale))
    pred = float(beta[0] + beta[1] * float(factor_last))
    resid = yy - (X @ beta)
    sigma = max(1e-6, robust_sigma(resid)) * max(0.25, float(marginal_vol_scale))
    if tail_df is not None and float(tail_df) > 2.0:
        draws = _draw_student_t(pred, sigma, float(tail_df), n=640, seed=seed)
    else:
        draws = _draw_normal(pred, sigma, n=640, seed=seed)
    q = monotonic_quantiles(quantiles_from_samples(draws, quantiles), quantiles)
    return q, {"beta0": float(beta[0]), "beta_factor": float(beta[1]), "dependence_regularization": reg, "tail_df": tail_df}


def bayes_tail_risk_predict(
    y_hist: Sequence[float],
    quantiles: Sequence[float],
    threshold_quantile: float = 0.1,
    min_tail_points: int = 8,
    shape_scale: float = 1.0,
    tail_scale_multiplier: float = 1.0,
    model_two_tails: bool = True,
) -> Tuple[Dict[float, float], Dict[str, Any]]:
    y = _safe_arr(y_hist)
    if y.size < 96:
        raise RuntimeError("insufficient_history")

    qcut = min(0.25, max(0.01, float(threshold_quantile)))
    lo_thr = float(np.quantile(y, qcut))
    hi_thr = float(np.quantile(y, 1.0 - qcut))
    lo_excess = lo_thr - y[y < lo_thr]
    hi_excess = y[y > hi_thr] - hi_thr
    min_points = max(4, int(min_tail_points))
    shape_adj = max(0.25, float(shape_scale))
    scale_mult = max(0.25, float(tail_scale_multiplier))

    def _gpd_quantile(excess: np.ndarray, p: float, base: float, sign: int) -> float:
        if excess.size < min_points:
            return float(np.quantile(y, p))
        mean_ex = float(np.mean(excess))
        var_ex = float(np.var(excess))
        raw_shape = 0.5 * (1.0 - (mean_ex**2) / max(var_ex, 1e-6))
        shape = max(-0.2, min(0.5, raw_shape * shape_adj))
        scale = max(1e-8, mean_ex * max(1e-6, (1.0 - shape)) * scale_mult)
        u = min(0.999, max(1e-5, p))
        if abs(shape) < 1e-4:
            q_ex = -scale * math.log(max(1e-8, 1.0 - u))
        else:
            q_ex = scale / shape * ((1.0 - u) ** (-shape) - 1.0)
        return float(base - q_ex if sign < 0 else base + q_ex)

    qvals: Dict[float, float] = {}
    for q in quantiles:
        qf = float(q)
        if model_two_tails and qf <= qcut:
            u = qf / qcut
            qvals[qf] = _gpd_quantile(lo_excess, u, lo_thr, sign=-1)
        elif model_two_tails and qf >= 1.0 - qcut:
            u = (qf - (1.0 - qcut)) / qcut
            qvals[qf] = _gpd_quantile(hi_excess, u, hi_thr, sign=1)
        elif (not model_two_tails) and qf >= 1.0 - qcut:
            u = (qf - (1.0 - qcut)) / qcut
            qvals[qf] = _gpd_quantile(hi_excess, u, hi_thr, sign=1)
        else:
            qvals[qf] = float(np.quantile(y, qf))

    qvals = monotonic_quantiles(qvals, quantiles)
    return qvals, {"lo_threshold": lo_thr, "hi_threshold": hi_thr, "lo_points": int(lo_excess.size), "hi_points": int(hi_excess.size), "threshold_quantile": qcut, "model_two_tails": bool(model_two_tails)}


def _esn_components(
    hidden: int,
    seed: int,
    input_scale: float = 0.5,
    recurrent_scale: float = 0.2,
    dropout: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, float]:
    rng = np.random.default_rng(int(seed))
    hidden = max(8, int(hidden))
    W_in = rng.normal(0.0, float(input_scale), size=(hidden, 1))
    W = rng.normal(0.0, float(recurrent_scale), size=(hidden, hidden))
    keep = 1.0 - min(0.8, max(0.0, float(dropout)))
    return W_in, W, keep


def _esn_step(
    state: np.ndarray,
    value: float,
    *,
    W_in: np.ndarray,
    W: np.ndarray,
    depth: int,
    keep: float,
) -> np.ndarray:
    x = np.asarray(state, dtype=float).copy()
    signal = W_in[:, 0] * float(value)
    for _ in range(max(1, int(depth))):
        x = np.tanh(W @ x + signal)
        if keep < 1.0:
            x = x * keep
    return x


def _esn_state_with_components(seq: np.ndarray, *, hidden: int, depth: int, W_in: np.ndarray, W: np.ndarray, keep: float) -> np.ndarray:
    x = np.zeros((max(8, int(hidden)),), dtype=float)
    states = []
    for v in seq:
        x = _esn_step(x, float(v), W_in=W_in, W=W, depth=depth, keep=keep)
        states.append(x.copy())
    return np.asarray(states, dtype=float)


def _esn_state(seq: np.ndarray, hidden: int, seed: int, input_scale: float = 0.5, recurrent_scale: float = 0.2, depth: int = 1, dropout: float = 0.0) -> np.ndarray:
    W_in, W, keep = _esn_components(hidden=hidden, seed=seed, input_scale=input_scale, recurrent_scale=recurrent_scale, dropout=dropout)
    return _esn_state_with_components(seq, hidden=hidden, depth=depth, W_in=W_in, W=W, keep=keep)


def _esn_last_state(seq: np.ndarray, hidden: int, seed: int, input_scale: float = 0.5, recurrent_scale: float = 0.2, depth: int = 1, dropout: float = 0.0) -> np.ndarray:
    W_in, W, keep = _esn_components(hidden=hidden, seed=seed, input_scale=input_scale, recurrent_scale=recurrent_scale, dropout=dropout)
    hidden = max(8, int(hidden))
    x = np.zeros((hidden,), dtype=float)
    for v in seq:
        x = _esn_step(x, float(v), W_in=W_in, W=W, depth=depth, keep=keep)
    return x


def neural_lstm_surrogate_predict(
    y_hist: Sequence[float],
    horizon_bars: int,
    quantiles: Sequence[float],
    seq_len: int,
    seed: int = 42,
    hidden_size: int = 32,
    num_layers: int = 1,
    dropout: float = 0.0,
    weight_decay: float = 1e-4,
    X_hist: Optional[np.ndarray] = None,
    x_last: Optional[np.ndarray] = None,
) -> Tuple[Dict[float, float], Dict[str, Any]]:
    y = _safe_arr(y_hist)
    if y.size < max(64, int(seq_len) // 2):
        raise RuntimeError("insufficient_history")
    seq = y[-int(seq_len) :]
    hidden = max(16, min(256, int(hidden_size)))
    layers = max(1, min(4, int(num_layers)))
    drop = 0.0 if layers <= 1 else min(0.6, max(0.0, float(dropout)))
    W_in, W, keep = _esn_components(hidden=hidden, seed=seed, dropout=drop)
    states = _esn_state_with_components(seq, hidden=hidden, depth=layers, W_in=W_in, W=W, keep=keep)
    if states.shape[0] < 8:
        raise RuntimeError("insufficient_states")
    X = states[:-1]
    t = seq[1:]
    extra_last = None
    if X_hist is not None and x_last is not None:
        X_extra = np.asarray(X_hist, dtype=float)
        x_extra_last = np.asarray(x_last, dtype=float).reshape(-1)
        if X_extra.ndim == 2 and X_extra.shape[0] == y.size and X_extra.shape[1] == x_extra_last.size:
            seq_extra = X_extra[-seq.size :]
            extra_rows = seq_extra[1:]
            extra_med = np.nanmedian(seq_extra, axis=0)
            extra_rows = np.where(np.isfinite(extra_rows), extra_rows, extra_med)
            extra_mean = np.nanmean(extra_rows, axis=0)
            extra_std = np.nanstd(extra_rows, axis=0)
            extra_std = np.where(extra_std < 1e-8, 1.0, extra_std)
            extra_rows = (extra_rows - extra_mean) / extra_std
            extra_last = np.where(np.isfinite(x_extra_last), x_extra_last, extra_med)
            extra_last = (extra_last - extra_mean) / extra_std
            X = np.column_stack([X, extra_rows])
    lam = 1e-3 * (1.0 + 250.0 * max(0.0, float(weight_decay)) + 2.0 * drop + 0.25 * (layers - 1))
    beta = _solve_linear_beta(X, t, ridge=lam)

    cur_state = states[-1].copy()
    preds = []
    for _ in range(max(1, int(horizon_bars))):
        feature_vec = cur_state
        if extra_last is not None:
            feature_vec = np.concatenate([feature_vec, extra_last], axis=0)
        yhat = float(feature_vec @ beta)
        preds.append(yhat)
        cur_state = _esn_step(cur_state, yhat, W_in=W_in, W=W, depth=layers, keep=keep)

    point = float(preds[-1])
    resid = t - (X @ beta)
    sigma = max(1e-6, robust_sigma(resid)) * (1.0 + drop + 10.0 * max(0.0, float(weight_decay)))
    draws = np.random.default_rng(seed).normal(point, sigma, size=512)
    q = monotonic_quantiles(quantiles_from_samples(draws, quantiles), quantiles)
    return q, {"hidden_size": hidden, "num_layers": layers, "dropout": drop, "seq_len": int(seq_len), "uses_exogenous_features": bool(extra_last is not None), "rollout_mode": "incremental_state_reuse"}


def neural_tcn_predict(
    y_hist: Sequence[float],
    horizon_bars: int,
    quantiles: Sequence[float],
    seq_len: int,
    seed: int = 42,
    channel_width: int = 32,
    dilation_depth: int = 6,
    kernel_size: int = 3,
    dropout: float = 0.0,
    X_hist: Optional[np.ndarray] = None,
    x_last: Optional[np.ndarray] = None,
) -> Tuple[Dict[float, float], Dict[str, Any]]:
    y = _safe_arr(y_hist)
    if y.size < max(96, int(seq_len) // 2):
        raise RuntimeError("insufficient_history")
    seq = y[-int(seq_len) :]
    depth = max(2, min(8, int(dilation_depth)))
    kernel = max(2, min(5, int(kernel_size)))
    channels = max(8, min(128, int(channel_width)))
    drop = min(0.6, max(0.0, float(dropout)))
    dilations = [2**i for i in range(depth)]
    max_d = max(dilations) * kernel
    if seq.size <= max_d + 4:
        raise RuntimeError("insufficient_seq_len")

    X_rows = []
    y_rows = []
    extra_last = None
    seq_extra = None
    extra_mean = None
    extra_std = None
    if X_hist is not None and x_last is not None:
        X_extra = np.asarray(X_hist, dtype=float)
        x_extra_last = np.asarray(x_last, dtype=float).reshape(-1)
        if X_extra.ndim == 2 and X_extra.shape[0] == y.size and X_extra.shape[1] == x_extra_last.size:
            seq_extra = X_extra[-seq.size :]
            extra_med = np.nanmedian(seq_extra, axis=0)
            extra_mean = np.nanmean(seq_extra, axis=0)
            extra_std = np.nanstd(seq_extra, axis=0)
            extra_std = np.where(extra_std < 1e-8, 1.0, extra_std)
            extra_last = np.where(np.isfinite(x_extra_last), x_extra_last, extra_med)
            extra_last = (extra_last - extra_mean) / extra_std
    for i in range(max_d, seq.size):
        feats = []
        for d in dilations:
            window = seq[max(0, i - d * kernel) : i : d]
            if window.size == 0:
                window = np.asarray([seq[i - d]], dtype=float)
            feats.append(float(np.mean(window)))
            feats.append(float(window[-1]))
        if seq_extra is not None and extra_mean is not None and extra_std is not None and extra_last is not None:
                current_extra = seq_extra[i]
                current_extra = np.where(np.isfinite(current_extra), current_extra, extra_last)
                current_extra = (current_extra - extra_mean) / extra_std
                feats.extend(current_extra.tolist())
        X_rows.append(feats)
        y_rows.append(float(seq[i]))
    X = np.asarray(X_rows, dtype=float)
    yy = np.asarray(y_rows, dtype=float)
    lam = 1e-3 * (1.0 + 0.02 * channels + 2.0 * drop)
    beta = _solve_linear_beta(X, yy, ridge=lam)

    hist = list(seq.astype(float))
    preds = []
    for _ in range(max(1, int(horizon_bars))):
        feats = []
        hist_arr = np.asarray(hist, dtype=float)
        for d in dilations:
            window = hist_arr[max(0, hist_arr.size - d * kernel) :: d]
            if window.size == 0:
                window = np.asarray([hist_arr[-d]], dtype=float)
            feats.append(float(np.mean(window[-kernel:])))
            feats.append(float(window[-1]))
        if extra_last is not None:
            feats.extend(np.asarray(extra_last, dtype=float).tolist())
        yhat = float(np.asarray(feats, dtype=float) @ beta)
        preds.append(yhat)
        hist.append(yhat)

    point = float(preds[-1])
    resid = yy - (X @ beta)
    sigma = max(1e-6, robust_sigma(resid)) * (1.0 + drop + 0.005 * channels)
    draws = np.random.default_rng(seed).normal(point, sigma, size=512)
    q = monotonic_quantiles(quantiles_from_samples(draws, quantiles), quantiles)
    return q, {"dilation_depth": depth, "kernel_size": kernel, "channel_width": channels, "seq_len": int(seq_len), "uses_exogenous_features": bool(extra_last is not None)}


def neural_nbeats_predict(
    y_hist: Sequence[float],
    horizon_bars: int,
    quantiles: Sequence[float],
    seq_len: int,
    seed: int = 42,
    num_stacks: int = 4,
    num_blocks: int = 2,
    layer_width: int = 128,
    generic_architecture: bool = True,
) -> Tuple[Dict[float, float], Dict[str, Any]]:
    y = _safe_arr(y_hist)
    if y.size < max(128, int(seq_len) // 2):
        raise RuntimeError("insufficient_history")
    seq = y[-int(seq_len) :]
    n = seq.size
    stacks = max(1, min(8, int(num_stacks)))
    blocks = max(1, min(6, int(num_blocks)))
    width = max(32, min(512, int(layer_width)))
    t = np.linspace(-1.0, 1.0, n)
    h = float(max(1, int(horizon_bars))) / float(n)
    tf = min(1.5, 1.0 + h)

    basis_cols = [np.ones(n), t, t * t]
    future_terms = [1.0, tf, tf * tf]
    if bool(generic_architecture):
        seasonal_count = max(1, min(6, stacks + blocks // 2))
        for k in range(1, seasonal_count + 1):
            basis_cols.append(np.sin(k * np.pi * t))
            basis_cols.append(np.cos(k * np.pi * t))
            future_terms.extend([math.sin(k * math.pi * tf), math.cos(k * math.pi * tf)])
    else:
        for k in range(1, min(4, stacks) + 1):
            basis_cols.append(t ** (k + 2))
            future_terms.append(tf ** (k + 2))
    X = np.column_stack(basis_cols)
    xf = np.asarray(future_terms, dtype=float)
    lam = 1e-3 * (1.0 + 0.002 * width + 0.2 * blocks)
    beta = _solve_linear_beta(X, seq, ridge=lam)

    point = float(xf @ beta)
    fitted = X @ beta
    resid = seq - fitted
    sigma = max(1e-6, robust_sigma(resid)) * (1.0 + 0.001 * width)
    draws = np.random.default_rng(seed).normal(point, sigma, size=512)
    q = monotonic_quantiles(quantiles_from_samples(draws, quantiles), quantiles)
    return q, {"num_stacks": stacks, "num_blocks": blocks, "layer_width": width, "generic_architecture": bool(generic_architecture), "seq_len": int(seq_len), "basis_terms": int(X.shape[1])}
