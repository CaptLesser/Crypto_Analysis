from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.forecasting.common.pipeline_parquet_utils import decide_range_from_disk_edges
from src.forecasting.ml.shared.numeric_float_policy import DEFAULT_FLOAT_DTYPE, as_default_float_array, default_float_full, default_float_nan_full
from src.forecasting.ml.shared.numeric_forecast_io import (
    NumericForecastIOConfig,
    chunk_end_ts_for_month,
    enqueue_stage_group_done,
    enqueue_stage_write_batch,
    get_start_ts,
    get_stop_ts,
    load_model_state,
    load_unit_feature_frame,
    merge_numeric_dicts,
    completed_edge_from_module_parquet,
    module_output_max_ts,
    save_model_state,
    stage_month_parts,
    ts_monotonic_unique,
)
from src.forecasting.ml.shared.numeric_forecast_targets import add_count, add_elapsed, compute_future_labels

P10_Z = -1.2815515655446004
P90_Z = 1.2815515655446004


class ConstantRegressor:
    def __init__(self, value: float):
        self.value = float(value)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.full((len(x),), self.value, dtype=DEFAULT_FLOAT_DTYPE)


@dataclass
class RegressionBundle:
    mean_model: Any
    residual_std: float
    diagnostics: Optional[Dict[str, Any]] = None


def _record_fit_diagnostics(model: Any, detail_counts: Dict[str, int]) -> None:
    diagnostics = getattr(model, "diagnostics", None)
    if not isinstance(diagnostics, dict):
        return
    for key, value in diagnostics.items():
        if not str(key).endswith("_count"):
            continue
        name = f"fit_diag_{str(key)}"
        if isinstance(value, bool):
            detail_counts[name] = int(detail_counts.get(name, 0) + int(value))
        elif isinstance(value, int):
            detail_counts[name] = int(detail_counts.get(name, 0) + int(value))


@dataclass
class UnitWork:
    asset: str
    interval: int
    horizon_minutes: int
    horizon_bars: int
    task: str
    training_window_months: int
    training_window_bars: int
    refit_cadence: Optional[str]
    source_start_ts: int
    work_start_ts: int
    work_end_ts: int
    forecast_resume_edge: int
    eval_resume_edge: int
    model_threads: int


@dataclass
class HorizonGroupWork:
    group_id: str
    asset: str
    interval: int
    horizon_minutes: int
    horizon_bars: int
    works: List[UnitWork]
    model_threads: int


@dataclass(frozen=True)
class NumericForecastEngineConfig:
    io_config: NumericForecastIOConfig
    task_label: Dict[str, str]
    task_short: Dict[str, str]
    future_label_columns: Sequence[str]
    future_direction_deadzone: float
    ts_floor_production: int
    progress_every_seconds: int
    select_feature_columns_fn: Callable[[Sequence[str], str, int, int, set[str]], List[str]]
    training_window_bars_for_pair_fn: Callable[[str, int, int], int]
    fit_model_fn: Callable[[np.ndarray, np.ndarray, Optional[Dict[str, Any]]], RegressionBundle]
    predict_model_fn: Callable[[RegressionBundle, np.ndarray], Tuple[np.ndarray, np.ndarray]]
    resolve_model_params_fn: Callable[..., Dict[str, Any]]
    model_profile_label_fn: Callable[..., str]
    should_refit_fn: Callable[[str, Optional[int], int], bool]
    env_int_fn: Callable[[str], Optional[int]]
    forecast_resume_edge_env: str
    eval_resume_edge_env: str
    source_start_env: str
    source_end_env: str
    work_start_env: str


def _required_future_label_columns_for_group(engine_config: NumericForecastEngineConfig, works: Sequence[UnitWork]) -> List[str]:
    required: List[str] = []
    seen: set[str] = set()
    for work in works:
        task = str(work.task)
        label_col = str(engine_config.task_label[task])
        if label_col and label_col not in seen:
            required.append(label_col)
            seen.add(label_col)
        if task == "log_return" and "future_direction" not in seen:
            required.append("future_direction")
            seen.add("future_direction")
    return required


def _required_task_label_columns_for_group(engine_config: NumericForecastEngineConfig, works: Sequence[UnitWork]) -> List[str]:
    required: List[str] = []
    seen: set[str] = set()
    for work in works:
        label_col = str(engine_config.task_label[str(work.task)])
        if label_col and label_col not in seen:
            required.append(label_col)
            seen.add(label_col)
    return required


def state_to_bundle(state: Optional[Dict[str, Any]]) -> Optional[RegressionBundle]:
    if not isinstance(state, dict):
        return None
    mean_model = state.get("mean_model")
    if mean_model is None:
        return None
    try:
        residual_std = float(state.get("residual_std", 0.0))
    except Exception:
        residual_std = 0.0
    return RegressionBundle(mean_model=mean_model, residual_std=float(residual_std))


def bundle_to_state(bundle: Optional[RegressionBundle]) -> Dict[str, Any]:
    if bundle is None:
        return {}
    return {"mean_model": bundle.mean_model, "residual_std": float(bundle.residual_std)}


def walk_forward_predict(
    engine_config: NumericForecastEngineConfig,
    *,
    df: pd.DataFrame,
    task: str,
    horizon_minutes: int,
    interval_minutes: int,
    horizon_bars: int,
    selected_window_bars: Optional[int] = None,
    refit_cadence: Optional[str] = None,
    initial_state: Optional[Dict[str, Any]] = None,
    process_from_ts: Optional[int] = None,
    progress_label: str = "",
    progress_every_seconds: Optional[int] = None,
    regressor_params: Optional[Dict[str, Any]] = None,
    prepared_x_cols: Optional[Sequence[str]] = None,
    prepared_x: Optional[np.ndarray] = None,
    prepared_ts: Optional[np.ndarray] = None,
    prepared_y: Optional[np.ndarray] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[int], Dict[str, Any]]:
    if df.empty:
        return pd.DataFrame(columns=["ts", "asset"]), pd.DataFrame(columns=["ts", "asset"]), [], {}

    label_base = engine_config.task_label[task]
    detail_timing: Dict[str, float] = {
        "wf_feature_select_s": 0.0,
        "wf_numpy_extract_s": 0.0,
        "wf_state_restore_s": 0.0,
        "wf_target_prepare_s": 0.0,
        "wf_coldstart_fit_s": 0.0,
        "wf_active_index_build_s": 0.0,
        "wf_month_split_build_s": 0.0,
        "wf_predict_core_s": 0.0,
        "wf_output_series_fill_s": 0.0,
        "wf_output_frame_build_s": 0.0,
        "wf_meta_pack_s": 0.0,
    }
    detail_counts: Dict[str, int] = {
        "wf_n_rows": 0,
        "wf_valid_stop": 0,
        "wf_active_rows": 0,
        "wf_pred_blocks": 0,
        "wf_pending_count": 0,
        "wf_first_real_idx": -1,
    }
    t_feature_select = time.monotonic()
    x_cols = (
        [str(col) for col in list(prepared_x_cols)]
        if prepared_x_cols is not None
        else engine_config.select_feature_columns_fn(df.columns, str(task), int(horizon_minutes), int(interval_minutes), {label_base, "future_direction"})
    )
    add_elapsed(detail_timing, "wf_feature_select_s", time.monotonic() - t_feature_select)
    t_state_restore = time.monotonic()
    state_x_cols = list(initial_state.get("x_cols", [])) if isinstance(initial_state, dict) else []
    if state_x_cols and all(c in df.columns for c in state_x_cols):
        x_cols = list(state_x_cols)
    add_elapsed(detail_timing, "wf_state_restore_s", time.monotonic() - t_state_restore)
    t_numpy_extract = time.monotonic()
    x = as_default_float_array(prepared_x) if prepared_x is not None else df[x_cols].to_numpy(dtype=DEFAULT_FLOAT_DTYPE)
    ts = np.asarray(prepared_ts, dtype=np.int64) if prepared_ts is not None else df["ts"].to_numpy(dtype=np.int64)
    add_elapsed(detail_timing, "wf_numpy_extract_s", time.monotonic() - t_numpy_extract)
    t_target_prepare = time.monotonic()
    y = as_default_float_array(prepared_y) if prepared_y is not None else pd.to_numeric(df[label_base], errors="coerce").to_numpy(dtype=DEFAULT_FLOAT_DTYPE)
    add_elapsed(detail_timing, "wf_target_prepare_s", time.monotonic() - t_target_prepare)

    n = len(df)
    detail_counts["wf_n_rows"] = int(n)
    pred_mean = default_float_full((n,), 0.0)
    pred_std = default_float_full((n,), 0.0)
    y_eval = default_float_nan_full((n,))
    pending_ts: List[int] = []

    if selected_window_bars is None:
        selected_w = int(engine_config.training_window_bars_for_pair_fn(task, int(horizon_minutes), int(interval_minutes)))
    else:
        selected_w = max(1, int(selected_window_bars))

    model = state_to_bundle(initial_state)
    if model is not None and state_x_cols and state_x_cols != x_cols:
        model = None
    last_progress_log = time.monotonic()
    diag_fit_total_count = 0
    diag_fit_total_s = 0.0
    diag_fit_coldstart_count = 0
    diag_refit_count = 0
    diag_refit_s = 0.0
    diag_state_time_mismatch_count = 0
    diag_state_time_mismatch_refit_count = 0
    diag_predict_count = 0
    diag_predict_s = 0.0
    last_refit_ts = None
    if isinstance(initial_state, dict) and initial_state.get("last_refit_ts") is not None:
        try:
            last_refit_ts = int(initial_state.get("last_refit_ts"))
        except Exception:
            last_refit_ts = None
    valid_stop = max(0, n - int(horizon_bars))
    detail_counts["wf_valid_stop"] = int(valid_stop)
    if valid_stop < n:
        pending_ts = [int(v) for v in ts[valid_stop:]]
    detail_counts["wf_pending_count"] = int(len(pending_ts))
    if np.any(np.isfinite(y)):
        y_eval[np.isfinite(y)] = y[np.isfinite(y)]

    first_fit_idx: Optional[int] = None
    if model is None:
        coldstart_idx = int(selected_w + int(horizon_bars) - 1)
        if coldstart_idx < valid_stop:
            eligible_end = coldstart_idx - int(horizon_bars) + 1
            start = eligible_end - int(selected_w)
            end = eligible_end
            tfit0 = time.monotonic()
            model = engine_config.fit_model_fn(x[start:end], y[start:end], regressor_params=regressor_params)
            _record_fit_diagnostics(model, detail_counts)
            fit_elapsed = float(time.monotonic() - tfit0)
            add_elapsed(detail_timing, "wf_coldstart_fit_s", fit_elapsed)
            diag_fit_total_count += 1
            diag_fit_total_s += fit_elapsed
            diag_fit_coldstart_count += 1
            first_fit_idx = int(coldstart_idx)
            last_refit_ts = int(ts[coldstart_idx])

    t_active_index = time.monotonic()
    active_start_idx = 0
    if process_from_ts is not None:
        active_start_idx = int(np.searchsorted(ts, int(process_from_ts), side="left"))
    if first_fit_idx is not None:
        active_start_idx = max(active_start_idx, int(first_fit_idx))
    if model is not None and last_refit_ts is not None and active_start_idx < valid_stop and int(last_refit_ts) > int(ts[active_start_idx]):
        diag_state_time_mismatch_count += 1
        model = None
        last_refit_ts = None
        correction_idx = max(int(active_start_idx), int(selected_w + int(horizon_bars) - 1))
        if correction_idx < valid_stop:
            eligible_end = int(correction_idx) - int(horizon_bars) + 1
            if eligible_end >= int(selected_w):
                start = eligible_end - int(selected_w)
                end = eligible_end
                trefit0 = time.monotonic()
                model = engine_config.fit_model_fn(x[start:end], y[start:end], regressor_params=regressor_params)
                _record_fit_diagnostics(model, detail_counts)
                refit_elapsed = float(time.monotonic() - trefit0)
                diag_fit_total_count += 1
                diag_fit_total_s += refit_elapsed
                diag_refit_count += 1
                diag_refit_s += refit_elapsed
                diag_state_time_mismatch_refit_count += 1
                first_fit_idx = int(correction_idx)
                active_start_idx = max(int(active_start_idx), int(first_fit_idx))
                last_refit_ts = int(ts[correction_idx])
    if model is not None and last_refit_ts is None and active_start_idx < valid_stop:
        last_refit_ts = int(ts[active_start_idx])
    add_elapsed(detail_timing, "wf_active_index_build_s", time.monotonic() - t_active_index)

    first_real_idx: Optional[int] = None
    progress_seconds = engine_config.progress_every_seconds if progress_every_seconds is None else int(progress_every_seconds)
    if model is not None and active_start_idx < valid_stop:
        active_idx = np.arange(active_start_idx, valid_stop, dtype=np.int64)
        detail_counts["wf_active_rows"] = int(len(active_idx))
        if len(active_idx) > 0:
            first_real_idx = int(active_idx[0])
            detail_counts["wf_first_real_idx"] = int(first_real_idx)
            t_month_split = time.monotonic()
            month_dt = pd.to_datetime(ts[active_idx], unit="s", utc=True)
            month_keys = month_dt.year.to_numpy(dtype=np.int64) * 100 + month_dt.month.to_numpy(dtype=np.int64)
            split_points = np.flatnonzero(month_keys[1:] != month_keys[:-1]) + 1
            block_starts = np.concatenate(([0], split_points))
            block_stops = np.concatenate((split_points, [len(active_idx)]))
            detail_counts["wf_pred_blocks"] = int(len(block_starts))
            add_elapsed(detail_timing, "wf_month_split_build_s", time.monotonic() - t_month_split)
            for block_start, block_stop in zip(block_starts, block_stops):
                idx_block = active_idx[int(block_start) : int(block_stop)]
                segment_start = 0
                while segment_start < len(idx_block):
                    idx_i = idx_block[segment_start]
                    ts_i = int(ts[idx_i])
                    if refit_cadence and engine_config.should_refit_fn(str(refit_cadence), last_refit_ts, ts_i):
                        eligible_end = int(idx_i) - int(horizon_bars) + 1
                        if eligible_end >= int(selected_w):
                            start = eligible_end - int(selected_w)
                            end = eligible_end
                            trefit0 = time.monotonic()
                            model = engine_config.fit_model_fn(x[start:end], y[start:end], regressor_params=regressor_params)
                            _record_fit_diagnostics(model, detail_counts)
                            refit_elapsed = float(time.monotonic() - trefit0)
                            diag_fit_total_count += 1
                            diag_fit_total_s += refit_elapsed
                            diag_refit_count += 1
                            diag_refit_s += refit_elapsed
                            last_refit_ts = int(ts_i)
                    next_segment_start = len(idx_block)
                    if refit_cadence and last_refit_ts is not None:
                        for check_pos in range(segment_start + 1, len(idx_block)):
                            if engine_config.should_refit_fn(str(refit_cadence), last_refit_ts, int(ts[idx_block[check_pos]])):
                                next_segment_start = check_pos
                                break
                    current_idx = idx_block[segment_start:next_segment_start]
                    tp0 = time.monotonic()
                    pm_block, ps_block = engine_config.predict_model_fn(model, x[current_idx])
                    elapsed = float(time.monotonic() - tp0)
                    add_elapsed(detail_timing, "wf_predict_core_s", elapsed)
                    pred_mean[current_idx] = pm_block
                    pred_std[current_idx] = ps_block
                    diag_predict_count += int(len(current_idx))
                    diag_predict_s += elapsed
                    segment_start = next_segment_start
                if progress_seconds > 0:
                    now = time.monotonic()
                    if (now - last_progress_log) >= float(progress_seconds):
                        pct = 100.0 * (float(idx_block[-1] + 1) / float(max(n, 1)))
                        engine_config.io_config.log_fn(
                            f"{engine_config.io_config.naming.log_prefix}[progress] {progress_label} row={int(idx_block[-1]) + 1}/{n} ({pct:.1f}%) selected_w={selected_w} model_ready=True"
                        )
                        last_progress_log = now

    t_output_series = time.monotonic()
    y_eval_series = pd.Series(y_eval, dtype=DEFAULT_FLOAT_DTYPE).ffill().fillna(0.0)
    pred_mean_series = pd.Series(pred_mean, dtype=DEFAULT_FLOAT_DTYPE).ffill().fillna(0.0)
    pred_std_series = pd.Series(pred_std, dtype=DEFAULT_FLOAT_DTYPE).ffill().fillna(0.0)
    pred_p10_series = pred_mean_series + (pred_std_series * P10_Z)
    pred_p90_series = pred_mean_series + (pred_std_series * P90_Z)
    add_elapsed(detail_timing, "wf_output_series_fill_s", time.monotonic() - t_output_series)

    t_output_frame = time.monotonic()
    pred_out = df[["ts", "asset"]].copy()
    pred_out["pred_mean"] = pred_mean_series
    pred_out["pred_std"] = pred_std_series
    pred_out["pred_p10"] = pred_p10_series
    pred_out["pred_p90"] = pred_p90_series

    eval_out = df[["ts", "asset"]].copy()
    eval_out[label_base] = y_eval_series
    if task == "log_return":
        dz = float(engine_config.future_direction_deadzone)
        direction = np.where(y_eval_series > dz, 1, np.where(y_eval_series < -dz, -1, 0)).astype(int)
        eval_out["future_direction"] = direction
    add_elapsed(detail_timing, "wf_output_frame_build_s", time.monotonic() - t_output_frame)

    t_meta = time.monotonic()
    meta = {
        "selected_window_bars": selected_w,
        "first_real_prediction_ts": int(ts[first_real_idx]) if first_real_idx is not None else None,
        "last_refit_ts": (int(last_refit_ts) if last_refit_ts is not None else None),
        "x_cols": x_cols,
        "active_model_state": bundle_to_state(model),
        "eval_window": [],
        "day_flags": [],
        "last_check_day": None,
        "diagnostics": {
            "fit_total_count": int(diag_fit_total_count),
            "fit_total_s": float(diag_fit_total_s),
            "fit_coldstart_count": int(diag_fit_coldstart_count),
            "refit_count": int(diag_refit_count),
            "refit_s": float(diag_refit_s),
            "state_time_mismatch_count": int(diag_state_time_mismatch_count),
            "state_time_mismatch_refit_count": int(diag_state_time_mismatch_refit_count),
            "predict_count": int(diag_predict_count),
            "predict_s": float(diag_predict_s),
        },
        "diagnostics_detail": {**detail_timing, **detail_counts},
    }
    add_elapsed(detail_timing, "wf_meta_pack_s", time.monotonic() - t_meta)
    meta["diagnostics_detail"] = {**detail_timing, **detail_counts}
    return pred_out, eval_out, sorted(set(int(t) for t in pending_ts)), meta


def build_unit_work(
    engine_config: NumericForecastEngineConfig,
    *,
    asset: str,
    interval: int,
    horizon_minutes: int,
    horizon_bars: int,
    task: str,
    training_window_months: int,
    training_window_bars: int,
    refit_cadence: Optional[str],
    model_threads: int,
    prefetched_context: Optional[Dict[str, Any]] = None,
) -> Optional[UnitWork]:
    io_config = engine_config.io_config
    prior_state: Dict[str, Any] = {}
    if prefetched_context is None:
        ohlc_edge = get_stop_ts(io_config, asset, interval)
        scalar_edge = io_config.feature_max_ts_fn(interval, asset, root=io_config.scalar_root)
        source_start = get_start_ts(io_config, asset, interval)
        forecast_edge = module_output_max_ts(
            io_config,
            root=io_config.parquet_root,
            interval=interval,
            asset=asset,
            store="forecast",
        )
        eval_edge = module_output_max_ts(
            io_config,
            root=io_config.parquet_root,
            interval=interval,
            asset=asset,
            store="eval",
        )
    else:
        ohlc_edge = prefetched_context.get("ohlc_edge")
        scalar_edge = prefetched_context.get("scalar_edge")
        source_start = prefetched_context.get("source_start")
        forecast_edge = prefetched_context.get("forecast_edge")
        eval_edge = prefetched_context.get("eval_edge")
    if ohlc_edge is None or scalar_edge is None or source_start is None:
        return None
    source_start_ts = int(source_start)
    source_edge = min(int(ohlc_edge), int(scalar_edge))
    source_start_override = engine_config.env_int_fn(engine_config.source_start_env)
    source_end_override = engine_config.env_int_fn(engine_config.source_end_env)
    forecast_edge_override = engine_config.env_int_fn(engine_config.forecast_resume_edge_env)
    eval_edge_override = engine_config.env_int_fn(engine_config.eval_resume_edge_env)
    work_start_override = engine_config.env_int_fn(engine_config.work_start_env)
    if source_start_override is not None:
        source_start_ts = max(int(source_start_ts), int(source_start_override))
    if source_end_override is not None:
        source_edge = min(int(source_edge), int(source_end_override))
    if int(source_edge) < int(source_start_ts):
        return None
    step_seconds = int(interval) * 60
    if forecast_edge_override is not None:
        forecast_edge = int(forecast_edge_override)
    else:
        forecast_edge = completed_edge_from_module_parquet(
            io_config,
            root=io_config.parquet_root,
            interval=int(interval),
            asset=str(asset),
            task=str(task),
            horizon_minutes=int(horizon_minutes),
            store="forecast",
            start_ts=max(int(source_start_ts), int(engine_config.ts_floor_production)),
            stop_ts=int(source_edge),
            step_seconds=int(step_seconds),
            allow_head_gap=True,
        )
    if eval_edge_override is not None:
        eval_edge = int(eval_edge_override)
    else:
        eval_edge = completed_edge_from_module_parquet(
            io_config,
            root=io_config.parquet_root,
            interval=int(interval),
            asset=str(asset),
            task=str(task),
            horizon_minutes=int(horizon_minutes),
            store="eval",
            start_ts=max(int(source_start_ts), int(engine_config.ts_floor_production)),
            stop_ts=int(source_edge),
            step_seconds=int(step_seconds),
            allow_head_gap=True,
        )
    _, _, forecast_reason = decide_range_from_disk_edges(
        asset=str(asset),
        interval_min=int(interval),
        downstream_max_ts=(int(forecast_edge) if forecast_edge is not None else None),
        upstream_min_ts=int(source_start_ts),
        upstream_max_ts=int(source_edge),
    )
    _, _, eval_reason = decide_range_from_disk_edges(
        asset=str(asset),
        interval_min=int(interval),
        downstream_max_ts=(int(eval_edge) if eval_edge is not None else None),
        upstream_min_ts=int(source_start_ts),
        upstream_max_ts=int(source_edge),
    )
    first_run_start_ts = max(int(source_start_ts), int(engine_config.ts_floor_production))
    first_run_resume_edge = int(first_run_start_ts) - int(step_seconds)
    forecast_resume_edge = (
        int(forecast_edge)
        if forecast_reason != "first_run" and forecast_edge is not None
        else int(first_run_resume_edge)
    )
    eval_resume_edge = (
        int(eval_edge)
        if eval_reason != "first_run" and eval_edge is not None
        else int(first_run_resume_edge)
    )
    prior_state = (
        dict(prefetched_context.get("prior_state") or {})
        if prefetched_context is not None and prefetched_context.get("prior_state") is not None
        else (load_model_state(io_config, asset=asset, interval=int(interval), horizon_minutes=int(horizon_minutes), task=task) or {})
    )
    pending_eval_from_ts = prior_state.get("eval_pending_from_ts")
    try:
        pending_eval_from_ts_i = int(pending_eval_from_ts) if pending_eval_from_ts is not None else None
    except Exception:
        pending_eval_from_ts_i = None
    if pending_eval_from_ts_i is not None:
        eval_resume_edge = min(int(eval_resume_edge), int(pending_eval_from_ts_i) - int(step_seconds))
    work_start_ts = int(forecast_resume_edge)
    if work_start_override is not None:
        work_start_ts = int(work_start_override)
    work_end_ts = int(source_edge)
    if int(work_end_ts) <= int(work_start_ts):
        io_config.log_fn(
            f"{io_config.naming.log_prefix}[edge-skip] asset={asset} k={interval} h={horizon_minutes}m task={task} forecast_edge={forecast_edge} eval_edge={eval_edge} source_edge={source_edge} resume_source=forecast_disk_edge"
        )
        return None
    return UnitWork(
        asset=asset,
        interval=int(interval),
        horizon_minutes=int(horizon_minutes),
        horizon_bars=int(horizon_bars),
        task=task,
        training_window_months=int(training_window_months),
        training_window_bars=int(training_window_bars),
        refit_cadence=(str(refit_cadence) if refit_cadence else None),
        source_start_ts=int(source_start_ts),
        work_start_ts=int(work_start_ts),
        work_end_ts=int(work_end_ts),
        forecast_resume_edge=int(forecast_resume_edge),
        eval_resume_edge=int(eval_resume_edge),
        model_threads=int(model_threads),
    )

def compute_group(engine_config: NumericForecastEngineConfig, work_group: HorizonGroupWork) -> Dict[str, Any]:
    io_config = engine_config.io_config
    if not work_group.works:
        return {"group_id": str(work_group.group_id), "results": []}
    first_work = work_group.works[0]
    group_label = f"asset={work_group.asset} k={int(work_group.interval)} h={int(work_group.horizon_minutes)}m tasks={','.join(str(w.task) for w in work_group.works)}"
    step_seconds = int(work_group.interval) * 60
    source_start_ts = int(first_work.source_start_ts)
    work_end_ts = int(first_work.work_end_ts)

    task_ctx: Dict[str, Dict[str, Any]] = {}
    group_detail_timing: Dict[str, float] = {
        "load_base_grid_s": 0.0,
        "ohlc_sort_dedup_s": 0.0,
        "ohlc_merge_ffill_s": 0.0,
        "scalar_sort_dedup_s": 0.0,
        "scalar_merge_ffill_s": 0.0,
        "load_finalize_sort_s": 0.0,
        "tail_reuse_filter_copy_s": 0.0,
        "fresh_load_call_s": 0.0,
        "feature_concat_sort_dedup_s": 0.0,
        "future_label_prepare_s": 0.0,
        "future_label_compute_s": 0.0,
        "future_label_assign_back_s": 0.0,
        "apply_head_floor_s": 0.0,
        "empty_chunk_fastpath_s": 0.0,
        "cache_tail_build_s": 0.0,
    }
    group_detail_counts: Dict[str, int] = {
        "chunk_rows_loaded": 0,
        "chunk_rows_reused_tail": 0,
        "chunk_rows_fresh": 0,
        "chunk_rows_after_floor": 0,
        "suffix_start_idx": 0,
        "suffix_rows_labeled": 0,
    }
    for work in sorted(work_group.works, key=lambda w: str(w.task)):
        unit_label = f"asset={work.asset} k={int(work.interval)} h={int(work.horizon_minutes)}m task={work.task}"
        regressor_params = engine_config.resolve_model_params_fn(
            task=work.task,
            model_threads=int(work_group.model_threads),
            interval_minutes=int(work.interval),
            horizon_minutes=int(work.horizon_minutes),
            training_window_months=int(work.training_window_months),
        )
        profile_label = engine_config.model_profile_label_fn(
            work.task,
            interval_minutes=int(work.interval),
            horizon_minutes=int(work.horizon_minutes),
            training_window_months=int(work.training_window_months),
        )
        io_config.log_fn(f"{io_config.naming.log_prefix}[unit-start] {unit_label} work=[{int(work.work_start_ts)},{int(work.work_end_ts)}] profile={profile_label} n_jobs={int(regressor_params.get('n_jobs', 0))}")
        task_ctx[str(work.task)] = {
            "work": work,
            "unit_label": unit_label,
            "regressor_params": regressor_params,
            "current_edge": int(work.work_start_ts),
            "forecast_write_edge": int(work.forecast_resume_edge),
            "eval_write_edge": int(work.eval_resume_edge),
            "state": load_model_state(io_config, asset=work.asset, interval=int(work.interval), horizon_minutes=int(work.horizon_minutes), task=work.task) or {},
            "rows_dropped_total": 0,
            "rows_written_total": 0,
            "pred_parts_all": [],
            "eval_parts_all": [],
            "ts_start_seen": None,
            "chunk_commit_count": 0,
            "fit_total_count_total": 0,
            "fit_total_s_total": 0.0,
            "fit_coldstart_count_total": 0,
            "refit_count_total": 0,
            "refit_s_total": 0.0,
            "predict_count_total": 0,
            "predict_s_total": 0.0,
            "parquet_write_s_total": 0.0,
            "model_save_count": 0,
            "model_save_s_total": 0.0,
            "detail_timing_totals": {"postpredict_rename_cols_s": 0.0, "postpredict_mask_filter_s": 0.0, "postpredict_month_group_prep_s": 0.0, "state_pack_s": 0.0, "chunk_commit_finalize_s": 0.0},
            "detail_count_totals": {"rows_pred_candidate": 0, "rows_eval_candidate": 0, "pred_month_groups": 0, "eval_month_groups": 0},
        }

    chunk_idx = 0
    chunk_load_count = 0
    data_load_s_total = 0.0
    read_ohlc_s_total = 0.0
    read_scalar_s_total = 0.0
    future_label_s_total = 0.0
    cached_feature_tail: Optional[pd.DataFrame] = None
    run_t0 = time.monotonic()

    def _active_contexts() -> List[Dict[str, Any]]:
        return [ctx for ctx in task_ctx.values() if int(ctx["current_edge"]) < int(work_end_ts)]

    while _active_contexts():
        active_ctx = _active_contexts()
        global_current_edge = min(int(ctx["current_edge"]) for ctx in active_ctx)
        chunk_idx += 1
        chunk_load_count += 1
        chunk_end_ts = chunk_end_ts_for_month(start_exclusive_ts=int(global_current_edge), end_inclusive_ts=int(work_end_ts), interval_minutes=int(work_group.interval))
        max_training_buffer_bars = max(int(ctx["work"].training_window_bars) + int(work_group.horizon_bars) for ctx in active_ctx)
        buffer_seconds = int(max_training_buffer_bars) * int(step_seconds)
        load_start_ts = max(int(source_start_ts), int(global_current_edge) - int(buffer_seconds))

        t0 = time.monotonic()
        reuse_tail = pd.DataFrame()
        if cached_feature_tail is not None and not cached_feature_tail.empty:
            t_reuse = time.monotonic()
            reuse_tail = cached_feature_tail[cached_feature_tail["ts"].astype("int64") >= int(load_start_ts)]
            add_elapsed(group_detail_timing, "tail_reuse_filter_copy_s", time.monotonic() - t_reuse)
            add_count(group_detail_counts, "chunk_rows_reused_tail", int(len(reuse_tail)))
        fresh_start_ts = int(load_start_ts)
        if not reuse_tail.empty:
            fresh_start_ts = max(int(load_start_ts), int(global_current_edge) + int(step_seconds))
        fresh_stats = {"load_total_s": 0.0, "read_ohlc_s": 0.0, "read_scalar_s": 0.0}
        fresh_df = pd.DataFrame()
        if int(fresh_start_ts) <= int(chunk_end_ts):
            t_fresh = time.monotonic()
            fresh_df, fresh_stats = load_unit_feature_frame(io_config, asset=work_group.asset, interval=int(work_group.interval), start_ts=int(fresh_start_ts), stop_ts=int(chunk_end_ts))
            add_elapsed(group_detail_timing, "fresh_load_call_s", time.monotonic() - t_fresh)
            add_count(group_detail_counts, "chunk_rows_fresh", int(len(fresh_df)))
        if reuse_tail.empty:
            feature_df = fresh_df
        elif fresh_df.empty:
            feature_df = reuse_tail.reset_index(drop=True)
        elif ts_monotonic_unique(reuse_tail) and ts_monotonic_unique(fresh_df) and int(reuse_tail["ts"].iloc[-1]) < int(fresh_df["ts"].iloc[0]):
            feature_df = pd.concat([reuse_tail, fresh_df], ignore_index=True)
        else:
            t_concat = time.monotonic()
            feature_df = pd.concat([reuse_tail, fresh_df], ignore_index=True).sort_values("ts").drop_duplicates(subset=["asset", "ts"], keep="last").reset_index(drop=True)
            add_elapsed(group_detail_timing, "feature_concat_sort_dedup_s", time.monotonic() - t_concat)
        add_count(group_detail_counts, "chunk_rows_loaded", int(len(feature_df)))
        label_t0 = time.monotonic()
        if feature_df.empty:
            df = feature_df
        else:
            df = feature_df.reset_index(drop=True)
            active_works = [ctx["work"] for ctx in active_ctx]
            required_future_label_columns = _required_future_label_columns_for_group(engine_config, active_works)
            required_task_label_columns = _required_task_label_columns_for_group(engine_config, active_works)
            missing_future_cols = [col for col in required_future_label_columns if col not in df.columns]
            if missing_future_cols:
                df = pd.concat(
                    [
                        df,
                        pd.DataFrame(
                            {
                                col: np.full((len(df),), np.nan, dtype=DEFAULT_FLOAT_DTYPE)
                                for col in missing_future_cols
                            }
                        ),
                    ],
                    axis=1,
                )
            label_ready_mask = df[required_task_label_columns].notna().all(axis=1)
            missing_idx = np.flatnonzero(~label_ready_mask.to_numpy(dtype=bool))
            if len(missing_idx) > 0:
                suffix_start_idx = int(missing_idx[0])
                add_count(group_detail_counts, "suffix_start_idx", int(suffix_start_idx))
                add_count(group_detail_counts, "suffix_rows_labeled", int(len(df) - suffix_start_idx))
                suffix_labels, future_detail = compute_future_labels(
                    df.loc[suffix_start_idx:, ["open", "high", "low", "close", "volume", "trades"]].reset_index(drop=True),
                    horizon_bars=int(work_group.horizon_bars),
                    future_direction_deadzone=float(engine_config.future_direction_deadzone),
                    target_columns=required_future_label_columns,
                )
                for key_name, value in future_detail.items():
                    if isinstance(value, float):
                        add_elapsed(group_detail_timing, key_name, float(value))
                    elif isinstance(value, int):
                        add_count(group_detail_counts, key_name, int(value))
                for col in required_future_label_columns:
                    df.loc[suffix_start_idx:, col] = suffix_labels[col].to_numpy()
        label_s = float(time.monotonic() - label_t0)
        data_load_s_total += float(fresh_stats.get("load_total_s", 0.0)) + label_s
        read_ohlc_s_total += float(fresh_stats.get("read_ohlc_s", 0.0))
        read_scalar_s_total += float(fresh_stats.get("read_scalar_s", 0.0))
        for key_name, value in fresh_stats.items():
            if key_name in {"load_total_s", "read_ohlc_s", "read_scalar_s"}:
                continue
            if str(key_name).endswith("_s") and isinstance(value, (int, float)):
                add_elapsed(group_detail_timing, str(key_name), float(value))
            elif isinstance(value, (int, float)):
                add_count(group_detail_counts, str(key_name), int(value))
        future_label_s_total += label_s
        io_config.log_fn(f"{io_config.naming.log_prefix}[group-load] {group_label} chunk={chunk_idx} load=[{int(load_start_ts)},{int(chunk_end_ts)}] rows={len(df)} load_s={time.monotonic() - t0:.1f}")
        add_count(group_detail_counts, "chunk_rows_after_floor", int(len(df)))
        if df.empty:
            cached_feature_tail = df
            for ctx in active_ctx:
                ctx["current_edge"] = int(chunk_end_ts)
            continue

        pending_chunk_commits: List[Dict[str, Any]] = []
        pending_pred_month_frames: Dict[Tuple[int, int], List[pd.DataFrame]] = {}
        pending_eval_month_frames: Dict[Tuple[int, int], List[pd.DataFrame]] = {}
        pending_pred_part_meta: Dict[Tuple[int, int], List[Dict[str, Any]]] = {}
        pending_eval_part_meta: Dict[Tuple[int, int], List[Dict[str, Any]]] = {}
        pending_pred_expected_cols: set[str] = set()
        pending_eval_expected_cols: set[str] = set()
        for ctx in active_ctx:
            work = ctx["work"]
            process_from_ts = int(ctx["current_edge"]) + int(step_seconds)
            if int(process_from_ts) > int(chunk_end_ts):
                ctx["current_edge"] = int(chunk_end_ts)
                continue
            pred_df, eval_df, tail_fill_ts, meta = walk_forward_predict(engine_config, df=df, task=work.task, horizon_minutes=int(work.horizon_minutes), interval_minutes=int(work.interval), horizon_bars=int(work.horizon_bars), selected_window_bars=int(work.training_window_bars), refit_cadence=work.refit_cadence, initial_state=ctx["state"], process_from_ts=int(process_from_ts), progress_label=str(ctx["unit_label"]), regressor_params=dict(ctx["regressor_params"]))
            pending_eval_from_ts = min((int(ts_i) for ts_i in tail_fill_ts), default=None)
            tshort = engine_config.task_short[work.task]
            pref = io_config.naming.prediction_prefix
            t_rename = time.monotonic()
            pred_df = pred_df.rename(columns={"pred_mean": f"{pref}_pred_mean_{tshort}_{int(work.horizon_minutes)}m", "pred_std": f"{pref}_pred_std_{tshort}_{int(work.horizon_minutes)}m", "pred_p10": f"{pref}_pred_p10_{tshort}_{int(work.horizon_minutes)}m", "pred_p90": f"{pref}_pred_p90_{tshort}_{int(work.horizon_minutes)}m"})
            label_col = engine_config.task_label[work.task]
            eval_cols = {label_col: f"{label_col}_{int(work.horizon_minutes)}m"}
            if work.task == "log_return":
                eval_cols["future_direction"] = f"future_direction_{int(work.horizon_minutes)}m"
            eval_df = eval_df.rename(columns=eval_cols)
            add_elapsed(ctx["detail_timing_totals"], "postpredict_rename_cols_s", time.monotonic() - t_rename)
            t_mask = time.monotonic()
            pred_ts = pred_df["ts"].astype("int64")
            eval_ts = eval_df["ts"].astype("int64")
            pred_mask = (
                (pred_ts >= int(engine_config.ts_floor_production))
                & (pred_ts > int(ctx["forecast_write_edge"]))
                & (pred_ts <= int(chunk_end_ts))
            )
            eval_mask = (
                (eval_ts >= int(engine_config.ts_floor_production))
                & (eval_ts > int(ctx["eval_write_edge"]))
                & (eval_ts <= int(chunk_end_ts))
            )
            to_write_pred = pred_df[pred_mask].copy()
            to_write_eval = eval_df[eval_mask].copy()
            ctx["rows_written_total"] += int(len(to_write_pred))
            add_elapsed(ctx["detail_timing_totals"], "postpredict_mask_filter_s", time.monotonic() - t_mask)
            ctx["detail_count_totals"]["rows_pred_candidate"] = int(ctx["detail_count_totals"].get("rows_pred_candidate", 0) + int(len(to_write_pred)))
            ctx["detail_count_totals"]["rows_eval_candidate"] = int(ctx["detail_count_totals"].get("rows_eval_candidate", 0) + int(len(to_write_eval)))
            t_month_group = time.monotonic()
            pred_month_groups = 0
            eval_month_groups = 0
            if not to_write_pred.empty:
                pred_write_cols = [c for c in to_write_pred.columns if c not in {"ts", "asset"}]
                pending_pred_expected_cols.update(pred_write_cols)
                ts_dt = pd.to_datetime(to_write_pred["ts"], unit="s", utc=True)
                to_write_pred["year"] = ts_dt.dt.year.astype(int)
                to_write_pred["month"] = ts_dt.dt.month.astype(int)
                for (y, m), grp in to_write_pred.groupby(["year", "month"], sort=True):
                    pred_month_groups += 1
                    month_key = (int(y), int(m))
                    month_df = grp.drop(columns=["year", "month"]).copy()
                    pending_pred_month_frames.setdefault(month_key, []).append(month_df)
                    pending_pred_part_meta.setdefault(month_key, []).append(
                        {
                            "horizon_minutes": int(work.horizon_minutes),
                            "task": str(work.task),
                            "store": "forecast",
                            "expected_cols": list(pred_write_cols),
                            "min_ts": int(month_df["ts"].min()),
                            "max_ts": int(month_df["ts"].max()),
                        }
                    )
            if not to_write_eval.empty:
                eval_write_cols = [c for c in to_write_eval.columns if c not in {"ts", "asset"}]
                pending_eval_expected_cols.update(eval_write_cols)
                ts_dt = pd.to_datetime(to_write_eval["ts"], unit="s", utc=True)
                to_write_eval["year"] = ts_dt.dt.year.astype(int)
                to_write_eval["month"] = ts_dt.dt.month.astype(int)
                for (y, m), grp in to_write_eval.groupby(["year", "month"], sort=True):
                    eval_month_groups += 1
                    month_key = (int(y), int(m))
                    month_df = grp.drop(columns=["year", "month"]).copy()
                    pending_eval_month_frames.setdefault(month_key, []).append(month_df)
                    pending_eval_part_meta.setdefault(month_key, []).append(
                        {
                            "horizon_minutes": int(work.horizon_minutes),
                            "task": str(work.task),
                            "store": "eval",
                            "expected_cols": list(eval_write_cols),
                            "min_ts": int(month_df["ts"].min()),
                            "max_ts": int(month_df["ts"].max()),
                        }
                    )
            add_elapsed(ctx["detail_timing_totals"], "postpredict_month_group_prep_s", time.monotonic() - t_month_group)
            ctx["detail_count_totals"]["pred_month_groups"] = int(ctx["detail_count_totals"].get("pred_month_groups", 0) + pred_month_groups)
            ctx["detail_count_totals"]["eval_month_groups"] = int(ctx["detail_count_totals"].get("eval_month_groups", 0) + eval_month_groups)
            diag = meta.get("diagnostics", {}) if isinstance(meta, dict) else {}
            ctx["fit_total_count_total"] += int(diag.get("fit_total_count", 0))
            ctx["fit_total_s_total"] += float(diag.get("fit_total_s", 0.0))
            ctx["fit_coldstart_count_total"] += int(diag.get("fit_coldstart_count", 0))
            ctx["refit_count_total"] += int(diag.get("refit_count", 0))
            ctx["refit_s_total"] += float(diag.get("refit_s", 0.0))
            ctx["predict_count_total"] += int(diag.get("predict_count", 0))
            ctx["predict_s_total"] += float(diag.get("predict_s", 0.0))
            detail_diag = meta.get("diagnostics_detail", {}) if isinstance(meta, dict) else {}
            for key_name, value in detail_diag.items():
                if isinstance(value, float):
                    add_elapsed(ctx["detail_timing_totals"], str(key_name), float(value))
                elif isinstance(value, int):
                    ctx["detail_count_totals"][str(key_name)] = int(ctx["detail_count_totals"].get(str(key_name), 0) + int(value))
            t_state_pack = time.monotonic()
            ctx["state"] = {**(meta.get("active_model_state", {}) if isinstance(meta, dict) else {}), "selected_window_bars": meta.get("selected_window_bars"), "eval_window": meta.get("eval_window", []), "day_flags": meta.get("day_flags", []), "last_check_day": meta.get("last_check_day"), "last_refit_ts": meta.get("last_refit_ts"), "eval_pending_from_ts": pending_eval_from_ts, "x_cols": meta.get("x_cols", [])}
            add_elapsed(ctx["detail_timing_totals"], "state_pack_s", time.monotonic() - t_state_pack)
            pending_chunk_commits.append({"ctx": ctx, "task_chunk_end_ts": int(chunk_end_ts), "rows_written": int(len(to_write_pred)), "pending_eval_from_ts": pending_eval_from_ts, "has_new_data": bool((not to_write_pred.empty) or (not to_write_eval.empty))})

        if pending_pred_month_frames:
            enqueue_stage_write_batch(str(work_group.group_id), stage_month_parts(io_config, month_frames=pending_pred_month_frames, group_id=str(work_group.group_id), interval=int(work_group.interval), asset=str(work_group.asset), horizon_minutes=int(work_group.horizon_minutes), task="__coalesced__", store="forecast", expected_cols=sorted(pending_pred_expected_cols), part_metadata_by_month=pending_pred_part_meta, chunk_idx=int(chunk_idx)))
        if pending_eval_month_frames:
            enqueue_stage_write_batch(str(work_group.group_id), stage_month_parts(io_config, month_frames=pending_eval_month_frames, group_id=str(work_group.group_id), interval=int(work_group.interval), asset=str(work_group.asset), horizon_minutes=int(work_group.horizon_minutes), task="__coalesced__", store="eval", expected_cols=sorted(pending_eval_expected_cols), part_metadata_by_month=pending_eval_part_meta, chunk_idx=int(chunk_idx)))
        for pending in pending_chunk_commits:
            ctx = pending["ctx"]
            work = ctx["work"]
            task_chunk_end_ts = int(pending["task_chunk_end_ts"])
            t_commit_finalize = time.monotonic()
            if pending["has_new_data"]:
                t_save0 = time.monotonic()
                save_model_state(io_config, asset=work.asset, interval=int(work.interval), horizon_minutes=int(work.horizon_minutes), task=work.task, state=ctx["state"])
                ctx["model_save_count"] += 1
                ctx["model_save_s_total"] += float(time.monotonic() - t_save0)
                if int(task_chunk_end_ts) > int(ctx["forecast_write_edge"]):
                    ctx["forecast_write_edge"] = int(task_chunk_end_ts)
                ctx["chunk_commit_count"] += 1
            pending_eval_from_ts = pending.get("pending_eval_from_ts")
            if pending_eval_from_ts is None:
                if int(task_chunk_end_ts) > int(ctx["eval_write_edge"]):
                    ctx["eval_write_edge"] = int(task_chunk_end_ts)
            else:
                next_eval_edge = int(pending_eval_from_ts) - int(work.interval) * 60
                if next_eval_edge < int(ctx["eval_write_edge"]):
                    ctx["eval_write_edge"] = int(next_eval_edge)
            ctx["current_edge"] = int(task_chunk_end_ts)
            add_elapsed(ctx["detail_timing_totals"], "chunk_commit_finalize_s", time.monotonic() - t_commit_finalize)
        next_tail_start_ts = max(int(source_start_ts), int(chunk_end_ts) - int(buffer_seconds))
        cached_feature_tail = df if df.empty or int(df["ts"].iloc[0]) >= int(next_tail_start_ts) else df[df["ts"].astype("int64") >= int(next_tail_start_ts)].copy()

    wall_s = float(time.monotonic() - run_t0)
    results: List[Dict[str, Any]] = []
    for task, ctx in sorted(task_ctx.items()):
        work = ctx["work"]
        detailed_timing = merge_numeric_dicts(group_detail_timing, ctx["detail_timing_totals"])
        detailed_counts = dict(group_detail_counts)
        detailed_counts.update(ctx["detail_count_totals"])
        accounted_s_total = float(sum(float(value) for value in detailed_timing.values() if isinstance(value, (int, float))))
        residual_s_total = float(wall_s - accounted_s_total)
        diag_summary = {
            "wall_s": wall_s,
            "chunk_loads": int(chunk_load_count),
            "chunk_commits": int(ctx["chunk_commit_count"]),
            "model_saves": int(ctx["model_save_count"]),
            "selected_window_bars": (int(ctx["state"].get("selected_window_bars")) if ctx["state"].get("selected_window_bars") is not None else None),
            "feature_count": len(ctx["state"].get("x_cols", []) or []),
            "data_load_s": float(data_load_s_total),
            "read_ohlc_s": float(read_ohlc_s_total),
            "read_scalar_s": float(read_scalar_s_total),
            "future_label_s": float(future_label_s_total),
            "fit_total_count": int(ctx["fit_total_count_total"]),
            "fit_total_s": float(ctx["fit_total_s_total"]),
            "fit_coldstart_count": int(ctx["fit_coldstart_count_total"]),
            "refit_count": int(ctx["refit_count_total"]),
            "refit_s": float(ctx["refit_s_total"]),
            "predict_count": int(ctx["predict_count_total"]),
            "predict_s": float(ctx["predict_s_total"]),
            "parquet_write_s": float(ctx["parquet_write_s_total"]),
            "model_save_s": float(ctx["model_save_s_total"]),
            **detailed_timing,
            **detailed_counts,
            "accounted_s_total": accounted_s_total,
            "residual_s_total": residual_s_total,
            "detailed_timing_nonoverlap_note": "Detailed timing fields are intended as non-overlapping instrumentation buckets within a unit where measured.",
            "residual_formula_used": "wall_s - accounted_s_total",
        }
        results.append({"unit_label": ctx["unit_label"], "asset": work.asset, "interval": int(work.interval), "horizon_minutes": int(work.horizon_minutes), "task": work.task, "rows_dropped": int(ctx["rows_dropped_total"]), "ts_start": ctx["ts_start_seen"], "rows_written": int(ctx["rows_written_total"]), "work_start_ts": int(work.work_start_ts), "work_end_ts": int(work.work_end_ts), "next_selected_window": ctx["state"].get("selected_window_bars"), "pred_parts": list(ctx["pred_parts_all"]), "eval_parts": list(ctx["eval_parts_all"]), "diag_summary": diag_summary})
    enqueue_stage_group_done(str(work_group.group_id))
    return {"group_id": str(work_group.group_id), "results": results}
