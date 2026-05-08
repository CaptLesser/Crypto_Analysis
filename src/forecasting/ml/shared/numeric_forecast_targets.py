from __future__ import annotations

import time
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

from src.forecasting.ml.shared.numeric_float_policy import DEFAULT_FLOAT_DTYPE


def add_elapsed(bucket: Dict[str, float], key: str, elapsed: float) -> None:
    bucket[str(key)] = float(bucket.get(str(key), 0.0) + float(elapsed))


def add_count(bucket: Dict[str, int], key: str, value: int) -> None:
    bucket[str(key)] = int(bucket.get(str(key), 0) + int(value))


def safe_log_return(close: np.ndarray) -> np.ndarray:
    prev = np.roll(close, 1)
    prev[0] = close[0]
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.log(np.where(prev > 0, close / prev, 1.0))
    out[~np.isfinite(out)] = 0.0
    out[0] = 0.0
    return out


def safe_true_range(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    comp1 = high - low
    comp2 = np.abs(high - prev_close)
    comp3 = np.abs(low - prev_close)
    out = np.maximum(comp1, np.maximum(comp2, comp3))
    out[~np.isfinite(out)] = 0.0
    return out


def future_window_views(values: np.ndarray, horizon_bars: int, block_start: int, block_stop: int) -> np.ndarray:
    return np.lib.stride_tricks.sliding_window_view(values[block_start + 1 : block_stop + horizon_bars], horizon_bars)


def compute_future_labels(
    ohlc: pd.DataFrame,
    horizon_bars: int,
    *,
    future_direction_deadzone: float,
    target_columns: Any = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    timing: Dict[str, float] = {
        "future_logret_direct_s": 0.0,
        "future_window_view_build_s": 0.0,
        "future_rv_s": 0.0,
        "future_true_range_s": 0.0,
        "future_drawdown_runup_s": 0.0,
        "future_range_efficiency_s": 0.0,
        "future_direction_finalize_s": 0.0,
        "future_labels_dataframe_build_s": 0.0,
    }
    counts: Dict[str, int] = {
        "future_valid_n": 0,
        "future_block_count": 0,
        "future_block_rows": 0,
    }
    n = len(ohlc)
    close = pd.to_numeric(ohlc["close"], errors="coerce").astype(DEFAULT_FLOAT_DTYPE).to_numpy(dtype=DEFAULT_FLOAT_DTYPE)
    high = pd.to_numeric(ohlc["high"], errors="coerce").astype(DEFAULT_FLOAT_DTYPE).to_numpy(dtype=DEFAULT_FLOAT_DTYPE)
    low = pd.to_numeric(ohlc["low"], errors="coerce").astype(DEFAULT_FLOAT_DTYPE).to_numpy(dtype=DEFAULT_FLOAT_DTYPE)
    ret = safe_log_return(close)
    tr = safe_true_range(high, low, close)

    requested = None
    if target_columns is not None:
        requested = {str(col) for col in target_columns if str(col)}
    need_logret = requested is None or "future_log_return" in requested or "future_direction" in requested or "future_range_efficiency" in requested
    emit_logret = requested is None or "future_log_return" in requested
    need_rv = requested is None or "future_realized_vol" in requested
    need_true_range = requested is None or "future_true_range" in requested
    need_mdd = requested is None or "future_max_drawdown" in requested
    need_mru = requested is None or "future_max_runup" in requested
    need_range_efficiency = requested is None or "future_range_efficiency" in requested
    need_direction = requested is None or "future_direction" in requested

    logret = np.full((n,), np.nan, dtype=DEFAULT_FLOAT_DTYPE) if need_logret else None
    rv = np.full((n,), np.nan, dtype=DEFAULT_FLOAT_DTYPE) if need_rv else None
    true_range = np.full((n,), np.nan, dtype=DEFAULT_FLOAT_DTYPE) if need_true_range else None
    mdd = np.full((n,), np.nan, dtype=DEFAULT_FLOAT_DTYPE) if need_mdd else None
    mru = np.full((n,), np.nan, dtype=DEFAULT_FLOAT_DTYPE) if need_mru else None
    range_efficiency = np.full((n,), np.nan, dtype=DEFAULT_FLOAT_DTYPE) if need_range_efficiency else None

    if horizon_bars > 0:
        valid_n = max(0, n - horizon_bars)
        counts["future_valid_n"] = int(valid_n)
        block_rows = 2048
        if valid_n > 0 and need_logret:
            t_logret = time.monotonic()
            close_now = close[:valid_n]
            close_future = close[horizon_bars:]
            close_positive = (close_now > 0.0) & (close_future > 0.0)
            with np.errstate(divide="ignore", invalid="ignore"):
                logret_vals = np.where(close_positive, np.log(close_future / close_now), np.nan)
            logret[:valid_n] = logret_vals
            add_elapsed(timing, "future_logret_direct_s", time.monotonic() - t_logret)

        for block_start in range(0, valid_n, block_rows):
            block_stop = min(valid_n, block_start + block_rows)
            add_count(counts, "future_block_count", 1)
            add_count(counts, "future_block_rows", int(block_stop - block_start))
            ret_windows = None
            tr_windows = None
            price_windows = None
            needs_ret_windows = bool(need_rv or need_range_efficiency)
            needs_tr_windows = bool(need_true_range)
            needs_price_windows = bool(need_mdd or need_mru)
            if needs_ret_windows or needs_tr_windows or needs_price_windows:
                t_windows = time.monotonic()
                if needs_ret_windows:
                    ret_windows = future_window_views(ret, horizon_bars, block_start, block_stop)
                if needs_tr_windows:
                    tr_windows = future_window_views(tr, horizon_bars, block_start, block_stop)
                if needs_price_windows:
                    price_windows = future_window_views(close, horizon_bars, block_start, block_stop)
                add_elapsed(timing, "future_window_view_build_s", time.monotonic() - t_windows)
            block_slice = slice(block_start, block_stop)
            close_block = close[block_slice]

            if need_rv:
                t_rv = time.monotonic()
                rv[block_slice] = np.std(ret_windows, axis=1, ddof=0)
                add_elapsed(timing, "future_rv_s", time.monotonic() - t_rv)

            if need_true_range:
                t_tr = time.monotonic()
                tr_sum = np.nansum(tr_windows, axis=1)
                true_vals = np.full((block_stop - block_start,), np.nan, dtype=DEFAULT_FLOAT_DTYPE)
                pos_mask = close_block > 0.0
                if np.any(pos_mask):
                    true_vals[pos_mask] = tr_sum[pos_mask] / close_block[pos_mask]
                true_range[block_slice] = true_vals
                add_elapsed(timing, "future_true_range_s", time.monotonic() - t_tr)

            if need_mdd or need_mru:
                t_draw = time.monotonic()
                finite_price_mask = np.all(np.isfinite(price_windows), axis=1)
                if np.any(finite_price_mask):
                    finite_prices = price_windows[finite_price_mask]
                    peak = np.maximum.accumulate(finite_prices, axis=1)
                    trough = np.minimum.accumulate(finite_prices, axis=1)
                    with np.errstate(divide="ignore", invalid="ignore"):
                        dd = np.where(peak != 0.0, finite_prices / peak - 1.0, np.nan)
                        ru = np.where(trough != 0.0, finite_prices / trough - 1.0, np.nan)
                    if need_mdd:
                        block_mdd = np.full((block_stop - block_start,), np.nan, dtype=DEFAULT_FLOAT_DTYPE)
                        dd_finite = np.any(np.isfinite(dd), axis=1)
                        if np.any(dd_finite):
                            idx = np.flatnonzero(finite_price_mask)[dd_finite]
                            block_mdd[idx] = np.nanmin(dd[dd_finite], axis=1)
                        mdd[block_slice] = block_mdd
                    if need_mru:
                        block_mru = np.full((block_stop - block_start,), np.nan, dtype=DEFAULT_FLOAT_DTYPE)
                        ru_finite = np.any(np.isfinite(ru), axis=1)
                        if np.any(ru_finite):
                            idx = np.flatnonzero(finite_price_mask)[ru_finite]
                            block_mru[idx] = np.nanmax(ru[ru_finite], axis=1)
                        mru[block_slice] = block_mru
                add_elapsed(timing, "future_drawdown_runup_s", time.monotonic() - t_draw)

            if need_range_efficiency:
                t_eff = time.monotonic()
                path_length = np.nansum(np.abs(ret_windows), axis=1)
                net_move = np.abs(logret[block_slice])
                eff_vals = np.full((block_stop - block_start,), np.nan, dtype=DEFAULT_FLOAT_DTYPE)
                eff_mask = np.isfinite(net_move) & np.isfinite(path_length) & (path_length > 1e-12)
                if np.any(eff_mask):
                    eff_vals[eff_mask] = net_move[eff_mask] / path_length[eff_mask]
                zero_path_mask = np.isfinite(net_move) & np.isfinite(path_length) & (path_length <= 1e-12)
                if np.any(zero_path_mask):
                    eff_vals[zero_path_mask] = 1.0
                range_efficiency[block_slice] = eff_vals
                add_elapsed(timing, "future_range_efficiency_s", time.monotonic() - t_eff)

    t_df = time.monotonic()
    out_dict: Dict[str, Any] = {}
    if emit_logret:
        out_dict["future_log_return"] = logret
    if need_rv:
        out_dict["future_realized_vol"] = rv
    if need_true_range:
        out_dict["future_true_range"] = true_range
    if need_mdd:
        out_dict["future_max_drawdown"] = mdd
    if need_mru:
        out_dict["future_max_runup"] = mru
    if need_range_efficiency:
        out_dict["future_range_efficiency"] = range_efficiency
    out = pd.DataFrame(out_dict)
    add_elapsed(timing, "future_labels_dataframe_build_s", time.monotonic() - t_df)
    if need_direction:
        t_dir = time.monotonic()
        direction = np.where(logret > future_direction_deadzone, 1, np.where(logret < -future_direction_deadzone, -1, 0))
        direction[~np.isfinite(logret)] = 0
        out["future_direction"] = direction.astype(int)
        add_elapsed(timing, "future_direction_finalize_s", time.monotonic() - t_dir)
    return out, {**timing, **counts}
