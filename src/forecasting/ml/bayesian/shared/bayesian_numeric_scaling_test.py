from __future__ import annotations

import argparse
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.forecasting.common.forecast_family_core import discover_edge_and_min, read_feature_window_columns, seasonality_info, write_json_atomic
from src.forecasting.common.ohlcvt_source import read_ohlcvt
from src.forecasting.common.path_config import resolve_path, selected_profile
from src.forecasting.common.stats_module_utils import NUMERIC_TASK_TO_TARGET_COLUMN, warm_seasonality_profiles
from src.forecasting.ml.bayesian.shared.bayesian_diagnostic_analysis import analyze_manifest_for_model
from src.forecasting.ml.bayesian.shared.bayesian_numeric_cohort import resolve_bayesian_cohort_assets
from src.forecasting.ml.bayesian.shared.bayesian_numeric_model_registry import BAYESIAN_NUMERIC_BRANCHES
from src.forecasting.ml.bayesian.shared.bayesian_runtime_bootstrap import configure_bayesian_thread_env
from src.forecasting.ml.bayesian.shared.bayesian_stage1_profile import resolve_execution_profile
from src.forecasting.ml.shared.numeric_forecast_targets import compute_future_labels
from src.forecasting.ml.shared.numeric_float_policy import DEFAULT_FLOAT_DTYPE, as_default_float_array
from src.forecasting.ml.shared.numeric_origin_windows import build_production_origin_arrays, prepare_production_origin_window
from src.forecasting.ml.shared.test_branch_function_telemetry import emit_event_for_path, telemetry_scope_for_path

DEFAULT_STAGE2_INTERVALS = (5, 15, 30, 60, 240, 720, 1440)
DEFAULT_TRAIN_WINDOWS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)
DEFAULT_FORECAST_DAYS = 28.0
MIN_STAGE2_TRAIN_BARS = 64


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _relative_run_dirs(intervals: Sequence[int], train_windows: Sequence[int]) -> List[Path]:
    return [
        Path(f"interval={int(interval)}m") / f"window={int(training_window)}m"
        for interval in sorted({int(value) for value in intervals})
        for training_window in train_windows
    ]


def _train_bars_for_window(interval_minutes: int, training_window_months: int) -> int:
    requested_bars = int(training_window_months) * 30 * 24 * 60 // max(1, int(interval_minutes))
    return max(MIN_STAGE2_TRAIN_BARS, int(requested_bars))


def _model_stage2_min_train_bars(module: Any) -> int:
    runtime_params = getattr(getattr(module, "MODULE_SPEC", None), "runtime_params", {}) or {}
    try:
        return max(MIN_STAGE2_TRAIN_BARS, int(runtime_params.get("stage2_min_train_bars", MIN_STAGE2_TRAIN_BARS)))
    except Exception:
        return MIN_STAGE2_TRAIN_BARS


def _stage2_history_start_ts(
    *,
    common_edge: int,
    interval_minutes: int,
    training_window_months: int,
    max_horizon_minutes: int,
    forecast_days: float,
    min_train_bars: int = MIN_STAGE2_TRAIN_BARS,
) -> int:
    interval = max(1, int(interval_minutes))
    interval_seconds = interval * 60
    train_bars = max(_train_bars_for_window(interval, int(training_window_months)), int(min_train_bars))
    horizon_bars = max(1, int(math.ceil(float(max_horizon_minutes) / float(interval))))
    eval_bars = max(1, int(math.ceil(float(forecast_days) * 24.0 * 60.0 / float(interval))))
    required_seconds = int(train_bars + horizon_bars + eval_bars + 1) * interval_seconds
    requested_seconds = int(max(1, int(training_window_months))) * 31 * 86400
    return int(common_edge) - max(int(requested_seconds), int(required_seconds))


def _load_existing_summary(run_dir: Path) -> Optional[Dict[str, Any]]:
    summary_path = Path(run_dir) / "run_summary.json"
    if not summary_path.exists():
        return None
    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _summary_complete(summary: Optional[Dict[str, Any]]) -> bool:
    if not isinstance(summary, dict):
        return False
    if not isinstance(summary.get("config"), dict):
        return False
    accuracy = summary.get("accuracy") or {}
    by_asset = accuracy.get("by_asset_target_horizon")
    return isinstance(by_asset, dict) and bool(by_asset)


def _discover_resume_run_root(output_dir: Path, planned_relative_run_dirs: Sequence[Path]) -> Optional[Path]:
    if not output_dir.exists():
        return None
    run_roots = sorted((path for path in output_dir.glob("run=*") if path.is_dir()), key=lambda path: path.name)
    for run_root in reversed(run_roots):
        matching_dirs = [run_root / rel_path for rel_path in planned_relative_run_dirs if (run_root / rel_path).exists()]
        if not matching_dirs:
            continue
        if any(not _summary_complete(_load_existing_summary(path)) for path in matching_dirs):
            return run_root.resolve()
        if any(not (run_root / rel_path / "run_summary.json").exists() for rel_path in planned_relative_run_dirs):
            return run_root.resolve()
    return None


def parse_int_csv(raw: str, default: Sequence[int]) -> List[int]:
    values = [part.strip() for part in str(raw).split(",") if part.strip()]
    return [int(v) for v in values] if values else [int(v) for v in default]


def parse_str_csv(raw: str, default: Sequence[str]) -> List[str]:
    values = [part.strip() for part in str(raw).split(",") if part.strip()]
    return [str(v) for v in values] if values else [str(v) for v in default]


def combos_from_feature_profile_json(feature_profile_json: Path, *, supported_tasks: Sequence[str], requested_intervals: Sequence[int], requested_tasks: Sequence[str], requested_horizons: Sequence[int]) -> List[Tuple[int, int, str]]:
    payload = json.loads(feature_profile_json.read_text(encoding="utf-8"))
    selections = payload.get("selections") or {}
    combos: List[Tuple[int, int, str]] = []
    interval_filter = {int(v) for v in requested_intervals}
    task_filter = {str(v) for v in requested_tasks}
    horizon_filter = {int(v) for v in requested_horizons}
    for key in selections.keys():
        parts = {}
        for token in str(key).split("|"):
            if "=" not in token:
                continue
            name, value = token.split("=", 1)
            parts[str(name)] = str(value)
        try:
            interval = int(parts.get("interval", "0"))
            horizon = int(parts.get("horizon", "0"))
            task = str(parts.get("task", "")).strip()
        except Exception:
            continue
        if interval <= 0 or horizon <= 0 or not task:
            continue
        if task not in set(str(task_name) for task_name in supported_tasks):
            continue
        if interval_filter and interval not in interval_filter:
            continue
        if task_filter and task not in task_filter:
            continue
        if horizon_filter and horizon not in horizon_filter:
            continue
        combos.append((int(interval), int(horizon), str(task)))
    return sorted(set(combos), key=lambda item: (item[0], item[1], item[2]))


def cohort_assets_from_feature_profile_json(feature_profile_json: Path) -> List[str]:
    payload = json.loads(feature_profile_json.read_text(encoding="utf-8"))
    return [str(asset) for asset in (payload.get("cohort_assets") or []) if str(asset)]


def selected_features_from_feature_profile_json(feature_profile_json: Path, *, interval: int, horizon: int, task: str) -> List[str]:
    profile = resolve_execution_profile(
        feature_profile_json,
        interval=int(interval),
        horizon=int(horizon),
        task=str(task),
        dynamic_feature_candidates=(),
        needs_dynamic_features=False,
        use_seasonality=False,
    )
    return list(profile.selected_features)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bayesian numeric Stage 2 diagnostic scaling test")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--profile", type=str, default=selected_profile(default="pipeline_test"))
    parser.add_argument("--parquet-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--intervals", type=str, default="")
    parser.add_argument("--tasks", type=str, default="")
    parser.add_argument("--horizon-minutes", type=str, default="")
    parser.add_argument("--train-window-months", type=str, default=",".join(str(v) for v in DEFAULT_TRAIN_WINDOWS))
    parser.add_argument("--forecast-days", type=float, default=DEFAULT_FORECAST_DAYS)
    parser.add_argument("--asset-count", type=int, default=8)
    parser.add_argument("--asset-workers", type=int, default=1)
    parser.add_argument("--model-threads", type=int, default=None)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--feature-profile-json", type=Path, default=None)
    parser.add_argument("--staged", action="store_true")
    return parser.parse_args()


def _source_ohlcvt_root(profile: Optional[str] = None) -> Path:
    resolved_profile = str(profile or selected_profile(default="pipeline_test"))
    return Path(resolve_path("source_ohlcvt_root", profile=resolved_profile, required=False) or Path("parquet"))


def _source_feature_root(profile: Optional[str] = None, fallback: Optional[Path] = None) -> Path:
    resolved_profile = str(profile or selected_profile(default="pipeline_test"))
    return Path(resolve_path("source_feature_root", profile=resolved_profile, required=False) or fallback or _source_ohlcvt_root(resolved_profile))


def _load_module(model_key: str):
    return __import__(f"src.forecasting.ml.bayesian.{model_key}.numerics", fromlist=["MODULE_SPEC"])


def _resolve_assets(feature_profile_json: Optional[Path], asset_count: int, *, parquet_root: Path, intervals: Sequence[int], seed: int) -> List[str]:
    if feature_profile_json is not None and feature_profile_json.exists():
        assets = cohort_assets_from_feature_profile_json(feature_profile_json)
        if assets:
            return assets
    return resolve_bayesian_cohort_assets(
        parquet_root=Path(parquet_root).resolve(),
        intervals=intervals,
        asset_count=int(asset_count),
        explicit_assets=(),
        seed=int(seed),
    )


def _load_asset_frame(
    module,
    asset: str,
    interval: int,
    start_ts: int,
    end_ts: int,
    task: str,
    selected_feature_columns: Optional[Sequence[str]] = None,
    base_ohlc_frame: Optional[pd.DataFrame] = None,
    parquet_root: Optional[Path] = None,
    feature_root: Optional[Path] = None,
    telemetry_path: Optional[Path] = None,
    model_key: str = "",
) -> pd.DataFrame:
    base_event = {
        "family": "Bayesian_Numeric",
        "model": str(model_key),
        "stage": "stage2",
        "combo_key": f"{int(interval)}:{str(task)}",
        "interval_minutes": int(interval),
        "task": str(task),
        "asset": str(asset),
        "module_name": __name__,
    }
    ohlc_columns = ["open", "high", "low", "close", "volume", "trades"]
    feature_columns = [NUMERIC_TASK_TO_TARGET_COLUMN[str(task)]]
    dynamic_feature_candidates = [str(c) for c in module.MODULE_SPEC.dynamic_feature_candidates]
    if selected_feature_columns is None:
        feature_columns.extend(dynamic_feature_candidates)
    else:
        selected_dynamic = [str(c) for c in selected_feature_columns if str(c)]
        feature_columns.extend(selected_dynamic)
    with telemetry_scope_for_path(
        telemetry_path,
        **base_event,
        function_name="read_ohlcvt",
        phase_name="source_read",
        parent_phase="diagnostics",
        source_path=str(Path(parquet_root or _source_ohlcvt_root()).resolve()),
    ) as scope:
        ohlc_frame = (
            base_ohlc_frame
            if base_ohlc_frame is not None
            else read_ohlcvt(
                root=Path(parquet_root or _source_ohlcvt_root()).resolve(),
                asset=str(asset),
                interval_min=int(interval),
                start_ts=int(start_ts),
                end_ts=int(end_ts),
                columns=["ts", "asset", *ohlc_columns],
            )
        )
        scope.set_output(ohlc_frame, reason_code=("source_read_empty" if ohlc_frame.empty else ""))
    with telemetry_scope_for_path(
        telemetry_path,
        **base_event,
        function_name="read_feature_window_columns",
        phase_name="feature_load",
        parent_phase="diagnostics",
        source_path=str(Path(feature_root or _source_feature_root(fallback=Path(parquet_root).resolve() if parquet_root is not None else None)).resolve()),
    ) as scope:
        feature_frame = read_feature_window_columns(
            root=Path(feature_root or _source_feature_root(fallback=Path(parquet_root).resolve() if parquet_root is not None else None)).resolve(),
            interval_minutes=int(interval),
            asset=str(asset),
            columns=list(dict.fromkeys(feature_columns)),
            start_ts=int(start_ts),
            end_ts=int(end_ts),
        )
        scope.set_output(feature_frame, reason_code=("feature_load_empty" if feature_frame.empty else ""))
    with telemetry_scope_for_path(
        telemetry_path,
        input_obj=ohlc_frame,
        required_columns=["ts", "asset", *ohlc_columns],
        key_columns=["ts", "asset"],
        **base_event,
        function_name="_load_asset_frame",
        phase_name="join",
        parent_phase="diagnostics",
    ) as scope:
        merged = ohlc_frame.merge(feature_frame, on=["ts", "asset"], how="outer", sort=True)
        scope.set_output(merged, reason_code=("join_empty" if merged.empty else ""))
    for column in [*ohlc_columns, *feature_columns]:
        if column not in merged.columns:
            merged[column] = np.nan
    return merged.sort_values("ts").reset_index(drop=True)


def _factor_maps(module, frames: Dict[str, pd.DataFrame], task: str, *, interval: int, horizon_minutes: int) -> Dict[str, Dict[int, float]]:
    if not module.MODULE_SPEC.needs_factor_cache:
        return {asset: {} for asset in frames}
    target_col = NUMERIC_TASK_TO_TARGET_COLUMN[str(task)]
    horizon_bars = max(1, int(horizon_minutes) // max(1, int(interval)))
    maps: Dict[str, Dict[int, float]] = {}
    for asset, frame in frames.items():
        labels, _ = compute_future_labels(
            frame.loc[:, ["high", "low", "close"]].reset_index(drop=True),
            int(horizon_bars),
            future_direction_deadzone=0.0,
        )
        series = pd.to_numeric(labels.get(target_col), errors="coerce") if target_col in labels.columns else pd.Series(dtype=float)
        ts_series = pd.to_numeric(frame["ts"], errors="coerce")
        maps[str(asset)] = {
            int(ts): float(val)
            for ts, val in zip(ts_series, series)
            if pd.notna(ts) and pd.notna(val)
        }
    out: Dict[str, Dict[int, float]] = {}
    for asset, frame in frames.items():
        local: Dict[int, float] = {}
        for ts in pd.to_numeric(frame["ts"], errors="coerce").dropna().astype("int64"):
            vals = [mapping[int(ts)] for other, mapping in maps.items() if other != asset and int(ts) in mapping]
            if vals:
                local[int(ts)] = float(np.mean(vals))
        out[asset] = local
    return out


def _origin_metrics(
    module,
    frame: pd.DataFrame,
    factor_map: Dict[int, float],
    *,
    task: str,
    interval: int,
    horizon_minutes: int,
    training_window_months: int,
    selected_feature_columns: Optional[Sequence[str]] = None,
    use_seasonality: Optional[bool] = None,
    parquet_root: Optional[Path] = None,
    telemetry_path: Optional[Path] = None,
    model_key: str = "",
    asset: str = "",
    forecast_days: float = DEFAULT_FORECAST_DAYS,
    min_train_bars: int = MIN_STAGE2_TRAIN_BARS,
) -> Tuple[List[float], List[float], List[int]]:
    asset_name = str(asset or (frame["asset"].iloc[0] if not frame.empty and "asset" in frame.columns else ""))
    base_event = {
        "family": "Bayesian_Numeric",
        "model": str(model_key),
        "stage": "stage2",
        "combo_key": f"{int(interval)}:{int(horizon_minutes)}:{str(task)}",
        "interval_minutes": int(interval),
        "horizon_minutes": int(horizon_minutes),
        "task": str(task),
        "asset": asset_name,
        "module_name": __name__,
    }
    with telemetry_scope_for_path(
        telemetry_path,
        input_obj=frame,
        required_columns=["high", "low", "close"],
        **base_event,
        function_name="compute_future_labels",
        phase_name="label_build",
        parent_phase="diagnostics",
    ) as scope:
        labels, _ = compute_future_labels(frame.loc[:, ["high", "low", "close"]].reset_index(drop=True), int(horizon_minutes) // int(interval), future_direction_deadzone=0.0)
        scope.set_output(labels, reason_code=("labels_empty" if labels.empty else ""))
    label_col = NUMERIC_TASK_TO_TARGET_COLUMN[str(task)]
    merged = frame.reset_index(drop=True).copy()
    if label_col in merged.columns and label_col in labels.columns:
        merged = merged.drop(columns=[label_col])
    merged = pd.concat([merged, labels.reset_index(drop=True)], axis=1)
    candidate_cols = [str(c) for c in module.MODULE_SPEC.dynamic_feature_candidates if str(c) in merged.columns]
    if selected_feature_columns is None:
        feat_cols = candidate_cols
    else:
        feat_cols = [str(c) for c in selected_feature_columns if str(c) in merged.columns]
    feat_cols = [str(c) for c in feat_cols if str(c) in merged.columns and merged[str(c)].notna().any()]
    use_dynamic_features = bool(module.MODULE_SPEC.needs_dynamic_features and feat_cols)
    origin_arrays = build_production_origin_arrays(
        frame=merged,
        target_col=str(label_col),
        selected_feature_columns=feat_cols,
        use_dynamic_features=use_dynamic_features,
        as_float_array=as_default_float_array,
        float_dtype=DEFAULT_FLOAT_DTYPE,
        factor_map=factor_map,
        needs_factor_cache=bool(module.MODULE_SPEC.needs_factor_cache),
        coerce_ts=True,
    )
    ts_vec = origin_arrays.ts_vec
    y_vec = origin_arrays.y_vec
    valid_target_idx = origin_arrays.valid_target_idx
    emit_event_for_path(
        telemetry_path,
        **base_event,
        function_name="_origin_metrics",
        phase_name="join",
        parent_phase="diagnostics",
        status="completed",
        input_rows=len(frame),
        output_rows=len(valid_target_idx),
        input_columns_count=len(frame.columns),
        output_columns_count=len(merged.columns),
        reason_code=("labels_empty" if len(valid_target_idx) == 0 else ""),
    )
    edge_ts = int(ts_vec[-1]) if len(ts_vec) else 0
    eval_start_ts = int(edge_ts - float(forecast_days) * 86400)
    train_bars = max(_train_bars_for_window(int(interval), int(training_window_months)), int(min_train_bars))
    preds: List[float] = []
    actuals: List[float] = []
    pred_ts: List[int] = []
    seasonal_period_bars = None
    if bool(module.MODULE_SPEC.use_seasonality if use_seasonality is None else use_seasonality):
        seas = seasonality_info(parquet_root=Path(parquet_root or _source_ohlcvt_root()).resolve(), interval_minutes=int(interval), asset=str(merged["asset"].iloc[0]))
        seasonal_period_bars = seas.get("seasonality_period_bars") if seas.get("seasonality_usable") else None
    loop_started = time.perf_counter()
    attempted_origins = 0
    failed_origins = 0
    skipped_origins = 0
    for idx in range(len(ts_vec)):
        origin_ts = int(ts_vec[idx])
        if origin_ts < int(eval_start_ts):
            continue
        window_result = prepare_production_origin_window(
            arrays=origin_arrays,
            idx=int(idx),
            min_history_bars=48,
            history_bars=int(train_bars),
            use_dynamic_features=use_dynamic_features,
            needs_factor_cache=bool(module.MODULE_SPEC.needs_factor_cache),
            as_float_array=as_default_float_array,
        )
        if window_result.window is None:
            skipped_origins += 1
            continue
        window = window_result.window
        try:
            attempted_origins += 1
            qvals, _meta = module.MODULE_SPEC.predict_fn(
                y_hist=window.y_hist,
                horizon_bars=int(horizon_minutes) // int(interval),
                quantiles=[0.1, 0.5, 0.9],
                seasonal_period_bars=(int(seasonal_period_bars) if seasonal_period_bars is not None else None),
                seed=17 + idx,
                model_params=dict(module.MODULE_SPEC.model_params),
                x_hist=window.x_hist,
                x_last=window.x_last,
                factor_hist=window.factor_hist,
                factor_last=window.factor_last,
            )
        except Exception:
            failed_origins += 1
            continue
        y_true = window.actual_value
        if not math.isfinite(float(y_true)):
            skipped_origins += 1
            continue
        preds.append(float(qvals.get(0.5, np.nan)))
        actuals.append(float(y_true))
        pred_ts.append(origin_ts)
    emit_event_for_path(
        telemetry_path,
        **base_event,
        function_name="_origin_metrics",
        phase_name="predict",
        parent_phase="diagnostics",
        status="completed",
        reason_code=("predict_returned_empty" if not preds else ""),
        elapsed_seconds=time.perf_counter() - loop_started,
        input_rows=len(valid_target_idx),
        output_rows=len(preds),
        asset_count=1,
    )
    return preds, actuals, pred_ts


def _baseline(actuals: Sequence[float]) -> Tuple[Optional[float], Optional[float]]:
    if len(actuals) < 2:
        return None, None
    series = pd.Series(list(actuals), dtype=float)
    baseline = series.shift(1).dropna()
    compare = pd.DataFrame({"y": series.iloc[1:].reset_index(drop=True), "baseline": baseline.reset_index(drop=True)})
    if compare.empty:
        return None, None
    err = compare["baseline"] - compare["y"]
    return float((err.abs()).mean()), float(np.sqrt((err.pow(2.0)).mean()))


def _partition_combo_pairs(interval_pairs: Sequence[Tuple[str, int]], shard_count: int) -> List[List[Tuple[str, int]]]:
    ordered_pairs = [(str(task), int(horizon)) for task, horizon in interval_pairs]
    if not ordered_pairs:
        return []
    resolved_shard_count = max(1, min(int(shard_count), len(ordered_pairs)))
    return [ordered_pairs[index::resolved_shard_count] for index in range(resolved_shard_count) if ordered_pairs[index::resolved_shard_count]]


def _score_combo_shard(payload: Dict[str, Any]) -> Dict[str, Any]:
    import os

    model_threads = payload.get("model_threads")
    if model_threads is not None:
        os.environ["BAYESIAN_NUMERIC_MODEL_THREADS"] = str(max(1, int(model_threads)))
    configure_bayesian_thread_env()
    module = _load_module(str(payload["model_key"]))
    feature_profile_json = Path(str(payload["feature_profile_json"])) if payload.get("feature_profile_json") else None
    interval = int(payload["interval"])
    history_start_ts = int(payload["history_start_ts"])
    common_edge = int(payload["common_edge"])
    training_window_months = int(payload["training_window_months"])
    assets = [str(asset) for asset in payload.get("assets", ())]
    parquet_root = Path(payload.get("parquet_root") or _source_ohlcvt_root()).resolve()
    feature_root = Path(payload.get("feature_root") or _source_feature_root(fallback=parquet_root)).resolve()
    telemetry_path = Path(str(payload["telemetry_path"])) if payload.get("telemetry_path") else None
    grouped_frame_cache: Dict[Tuple[str, Tuple[str, ...]], Tuple[Dict[str, pd.DataFrame], Dict[str, Dict[int, float]], Optional[bool]]] = {}
    base_ohlc_cache: Dict[str, pd.DataFrame] = {}
    accuracy_by_asset: Dict[str, Dict[str, Any]] = {}
    completed_combo_count = 0
    started = time.perf_counter()
    total_prediction_rows = 0

    for task, horizon_minutes in payload.get("task_horizons", ()):
        completed_combo_count += 1
        combo_key = f"{int(interval)}:{int(horizon_minutes)}:{str(task)}"
        combo_profile = (
            resolve_execution_profile(
                feature_profile_json,
                interval=int(interval),
                horizon=int(horizon_minutes),
                task=str(task),
                dynamic_feature_candidates=module.MODULE_SPEC.dynamic_feature_candidates,
                needs_dynamic_features=bool(module.MODULE_SPEC.needs_dynamic_features),
                use_seasonality=bool(module.MODULE_SPEC.use_seasonality),
            )
            if feature_profile_json is not None and feature_profile_json.exists()
            else None
        )
        selected_features = (
            list(combo_profile.selected_dynamic_feature_columns)
            if combo_profile is not None and combo_profile.use_dynamic_features
            else None
        )
        cache_key = (str(task), tuple(selected_features or ()))
        cached = grouped_frame_cache.get(cache_key)
        if cached is None:
            frames = {}
            for asset in assets:
                base_ohlc_frame = base_ohlc_cache.get(str(asset))
                if base_ohlc_frame is None:
                    base_ohlc_frame = read_ohlcvt(
                        root=parquet_root,
                        asset=str(asset),
                        interval_min=int(interval),
                        start_ts=int(history_start_ts),
                        end_ts=int(common_edge),
                        columns=["ts", "asset", "open", "high", "low", "close", "volume", "trades"],
                    )
                    base_ohlc_cache[str(asset)] = base_ohlc_frame
                frames[str(asset)] = _load_asset_frame(
                    module,
                    str(asset),
                    int(interval),
                    int(history_start_ts),
                    int(common_edge),
                    str(task),
                    selected_feature_columns=selected_features,
                    base_ohlc_frame=base_ohlc_frame,
                    parquet_root=parquet_root,
                    feature_root=feature_root,
                    telemetry_path=telemetry_path,
                    model_key=str(payload["model_key"]),
                )
            factor_maps = _factor_maps(module, frames, str(task), interval=int(interval), horizon_minutes=int(horizon_minutes))
            cached = (frames, factor_maps, (combo_profile.use_seasonality if combo_profile is not None else None))
            grouped_frame_cache[cache_key] = cached
        frames, factor_maps, cached_use_seasonality = cached
        for asset, frame in frames.items():
            if frame.empty:
                emit_event_for_path(
                    telemetry_path,
                    family="Bayesian_Numeric",
                    model=str(payload["model_key"]),
                    stage="stage2",
                    combo_key=combo_key,
                    interval_minutes=int(interval),
                    horizon_minutes=int(horizon_minutes),
                    task=str(task),
                    asset=str(asset),
                    function_name="_score_combo_shard",
                    module_name=__name__,
                    phase_name="source_read",
                    status="skipped",
                    reason_code="source_read_empty",
                    output_rows=0,
                )
                continue
            preds, actuals, pred_ts = _origin_metrics(
                module,
                frame,
                factor_maps.get(asset, {}),
                task=str(task),
                interval=int(interval),
                horizon_minutes=int(horizon_minutes),
                training_window_months=int(training_window_months),
                selected_feature_columns=selected_features,
                use_seasonality=cached_use_seasonality,
                parquet_root=parquet_root,
                telemetry_path=telemetry_path,
                model_key=str(payload["model_key"]),
                asset=str(asset),
                forecast_days=float(payload.get("forecast_days", DEFAULT_FORECAST_DAYS)),
                min_train_bars=int(payload.get("min_train_bars", MIN_STAGE2_TRAIN_BARS)),
            )
            if not preds:
                emit_event_for_path(
                    telemetry_path,
                    family="Bayesian_Numeric",
                    model=str(payload["model_key"]),
                    stage="stage2",
                    combo_key=combo_key,
                    interval_minutes=int(interval),
                    horizon_minutes=int(horizon_minutes),
                    task=str(task),
                    asset=str(asset),
                    function_name="_score_combo_shard",
                    module_name=__name__,
                    phase_name="predict",
                    status="completed",
                    reason_code="predict_returned_empty",
                    input_rows=len(frame),
                    output_rows=0,
                )
                continue
            total_prediction_rows += int(len(preds))
            pred = np.asarray(preds, dtype=float)
            act = np.asarray(actuals, dtype=float)
            mae = float(np.mean(np.abs(pred - act)))
            rmse = float(np.sqrt(np.mean((pred - act) ** 2)))
            baseline_mae, baseline_rmse = _baseline(actuals)
            accuracy_by_asset.setdefault(str(asset), {})[f"{task}:{int(horizon_minutes)}m"] = {
                "forecast_count": int(len(preds)),
                "rmse": rmse,
                "mae": mae,
                "baseline_type": "persistence" if baseline_rmse is not None else None,
                "baseline_rmse": baseline_rmse,
                "baseline_mae": baseline_mae,
                "first_prediction_ts": min(pred_ts),
                "last_prediction_ts": max(pred_ts),
            }
    emit_event_for_path(
        telemetry_path,
        family="Bayesian_Numeric",
        model=str(payload["model_key"]),
        stage="stage2",
        interval_minutes=int(interval),
        asset_count=len(assets),
        function_name="_score_combo_shard",
        module_name=__name__,
        phase_name="predict",
        parent_phase="diagnostics",
        status="completed",
        elapsed_seconds=time.perf_counter() - started,
        input_rows=int(completed_combo_count) * int(len(assets)),
        output_rows=total_prediction_rows,
        reason_code=("predict_returned_empty" if total_prediction_rows == 0 else ""),
    )
    return {
        "accuracy_by_asset": accuracy_by_asset,
        "grouped_load_cache_entries": int(len(grouped_frame_cache)),
        "completed_job_shards": 1,
        "completed_combo_jobs": int(completed_combo_count),
    }


def _clean_incomplete_run_dir(run_dir: Path) -> None:
    if not run_dir.exists():
        return
    summary = _load_existing_summary(run_dir)
    if _summary_complete(summary):
        return
    import shutil

    shutil.rmtree(run_dir)


def run_stage_for_model(model_key: str) -> Path:
    args = parse_args()
    profile = str(getattr(args, "profile", selected_profile(default="pipeline_test")))
    if args.parquet_root is None:
        args.parquet_root = _source_ohlcvt_root(profile)
    feature_root = _source_feature_root(profile, fallback=Path(args.parquet_root).resolve())
    if args.model_threads is not None:
        import os

        os.environ["BAYESIAN_NUMERIC_MODEL_THREADS"] = str(max(1, int(args.model_threads)))
    configure_bayesian_thread_env()
    if str(model_key) not in BAYESIAN_NUMERIC_BRANCHES:
        raise RuntimeError(f"unsupported Bayesian model key: {model_key}")
    module = _load_module(model_key)
    min_train_bars = _model_stage2_min_train_bars(module)
    output_dir = Path(args.output_dir).resolve()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    feature_profile_json = args.feature_profile_json.resolve() if args.feature_profile_json else None
    if bool(args.staged) and feature_profile_json is None:
        raise SystemExit("Staged Bayesian Stage-2 run blocked: missing --feature-profile-json.")
    requested_intervals = parse_int_csv(args.intervals, DEFAULT_STAGE2_INTERVALS)
    requested_tasks = parse_str_csv(args.tasks, module.MODULE_SPEC.default_tasks)
    requested_horizons = parse_int_csv(args.horizon_minutes, module.MODULE_SPEC.default_horizons)
    assets = _resolve_assets(
        feature_profile_json,
        int(args.asset_count),
        parquet_root=Path(args.parquet_root),
        intervals=requested_intervals,
        seed=int(args.seed),
    )
    if feature_profile_json is not None and feature_profile_json.exists():
        combos = combos_from_feature_profile_json(feature_profile_json, supported_tasks=module.MODULE_SPEC.default_tasks, requested_intervals=requested_intervals, requested_tasks=requested_tasks, requested_horizons=requested_horizons)
    else:
        combos = [(int(interval), int(horizon), str(task)) for interval in requested_intervals for horizon in requested_horizons for task in requested_tasks if int(horizon) % int(interval) == 0]
    seasonality_warmup: Dict[str, Any] = {"enabled": bool(module.MODULE_SPEC.use_seasonality), "warmed": 0, "failed": {}}
    if bool(module.MODULE_SPEC.use_seasonality) and combos:
        seasonality_warmup = warm_seasonality_profiles(
            parquet_root=Path(args.parquet_root),
            interval_minutes=sorted({int(c[0]) for c in combos}),
            assets=list(assets),
        )
        if seasonality_warmup.get("failed"):
            raise RuntimeError(f"Bayesian Stage 2 seasonality warmup failed: {seasonality_warmup['failed']}")
    train_windows = parse_int_csv(args.train_window_months, DEFAULT_TRAIN_WINDOWS)
    planned_relative_run_dirs = _relative_run_dirs(sorted({int(c[0]) for c in combos}), train_windows)
    resume_run_root = _discover_resume_run_root(output_dir, planned_relative_run_dirs)
    run_root = resume_run_root or (output_dir / f"run={run_id}")
    run_root.mkdir(parents=True, exist_ok=True)
    emit_event_for_path(
        run_root,
        family="Bayesian_Numeric",
        model=str(model_key),
        stage="stage2",
        function_name="run_stage_for_model",
        module_name=__name__,
        phase_name="source_root_resolution",
        status="completed",
        source_path=str(Path(args.parquet_root).resolve()),
        artifact_profile_source=(str(feature_profile_json) if feature_profile_json else ""),
    )
    emit_event_for_path(
        run_root,
        family="Bayesian_Numeric",
        model=str(model_key),
        stage="stage2",
        function_name="run_stage_for_model",
        module_name=__name__,
        phase_name="combo_planning",
        status="completed",
        asset_count=len(assets),
        input_rows=len(requested_intervals) * len(requested_horizons) * len(requested_tasks),
        output_rows=len(combos),
        reason_code=("no_assets" if not assets else "no_assets" if not combos else ""),
    )
    runs: List[Dict[str, Any]] = []
    for interval in sorted(set(int(c[0]) for c in combos)):
        interval_pairs = [(str(task), int(horizon)) for iv, horizon, task in combos if int(iv) == int(interval)]
        if not interval_pairs:
            continue
        interval_edges = []
        asset_min_ts: Dict[str, int] = {}
        for asset in assets:
            edge_ts, min_ts = discover_edge_and_min(asset=str(asset), interval_minutes=int(interval))
            if edge_ts is None or min_ts is None:
                raise RuntimeError(f"Stage 2 exact-history guard failed for interval={int(interval)}m asset={asset}: missing edge/min timestamp")
            interval_edges.append(int(edge_ts))
            asset_min_ts[str(asset)] = int(min_ts)
        common_edge = min(interval_edges)
        for training_window_months in train_windows:
            run_dir = run_root / f"interval={int(interval)}m" / f"window={int(training_window_months)}m"
            summary = _load_existing_summary(run_dir)
            if _summary_complete(summary):
                runs.append({"paths": {"run_summary": str((run_dir / "run_summary.json").resolve())}})
                continue
            _clean_incomplete_run_dir(run_dir)
            run_dir.mkdir(parents=True, exist_ok=True)
            accuracy_by_asset: Dict[str, Dict[str, Any]] = {}
            max_horizon_minutes = max(int(horizon) for _task, horizon in interval_pairs)
            history_start_ts = _stage2_history_start_ts(
                common_edge=int(common_edge),
                interval_minutes=int(interval),
                training_window_months=int(training_window_months),
                max_horizon_minutes=int(max_horizon_minutes),
                forecast_days=float(args.forecast_days),
                min_train_bars=int(min_train_bars),
            )
            insufficient_assets = [asset for asset, min_ts in asset_min_ts.items() if int(min_ts) > int(history_start_ts)]
            if insufficient_assets:
                raise RuntimeError(
                    f"Stage 2 exact-history guard failed for interval={int(interval)}m window={int(training_window_months)}m. Missing exact history for assets: {insufficient_assets}"
                )
            worker_budget = max(1, int(args.asset_workers))
            combo_shards = _partition_combo_pairs(interval_pairs, worker_budget)
            if worker_budget > 1 and len(combo_shards) > 1:
                shard_payloads = [
                    {
                        "model_key": str(model_key),
                        "feature_profile_json": (str(feature_profile_json) if feature_profile_json is not None else None),
                        "interval": int(interval),
                        "history_start_ts": int(history_start_ts),
                        "common_edge": int(common_edge),
                        "training_window_months": int(training_window_months),
                        "assets": list(assets),
                        "task_horizons": [(str(task), int(horizon)) for task, horizon in shard],
                        "parquet_root": str(Path(args.parquet_root).resolve()),
                        "feature_root": str(Path(feature_root).resolve()),
                        "model_threads": (max(1, int(args.model_threads)) if args.model_threads is not None else None),
                        "telemetry_path": str(run_root),
                        "forecast_days": float(args.forecast_days),
                        "min_train_bars": int(min_train_bars),
                    }
                    for shard in combo_shards
                ]
                with ProcessPoolExecutor(max_workers=min(worker_budget, len(combo_shards))) as executor:
                    shard_results = list(executor.map(_score_combo_shard, shard_payloads))
            else:
                shard_results = [
                    _score_combo_shard(
                        {
                            "model_key": str(model_key),
                            "feature_profile_json": (str(feature_profile_json) if feature_profile_json is not None else None),
                            "interval": int(interval),
                            "history_start_ts": int(history_start_ts),
                            "common_edge": int(common_edge),
                            "training_window_months": int(training_window_months),
                            "assets": list(assets),
                            "task_horizons": [(str(task), int(horizon)) for task, horizon in shard],
                            "parquet_root": str(Path(args.parquet_root).resolve()),
                            "feature_root": str(Path(feature_root).resolve()),
                            "model_threads": (max(1, int(args.model_threads)) if args.model_threads is not None else None),
                            "telemetry_path": str(run_root),
                            "forecast_days": float(args.forecast_days),
                            "min_train_bars": int(min_train_bars),
                        }
                    )
                    for shard in combo_shards
                ]
            grouped_cache_entries = 0
            completed_job_shards = 0
            completed_combo_jobs = 0
            for shard_result in shard_results:
                grouped_cache_entries += int(shard_result.get("grouped_load_cache_entries", 0))
                completed_job_shards += int(shard_result.get("completed_job_shards", 0))
                completed_combo_jobs += int(shard_result.get("completed_combo_jobs", 0))
                for asset, combo_payloads in (shard_result.get("accuracy_by_asset") or {}).items():
                    accuracy_by_asset.setdefault(str(asset), {}).update(dict(combo_payloads))
            if not accuracy_by_asset:
                raise RuntimeError(
                    f"Bayesian Stage 2 contract error: no eval metrics generated for interval={int(interval)}m "
                    f"window={int(training_window_months)}m "
                    f"(completed_job_shards={int(completed_job_shards)}, completed_combo_jobs={int(completed_combo_jobs)})."
                )
            quality = {
                "status": "eligible_complete",
                "reason": "metrics_generated",
                "requested_assets": int(len(assets)),
                "assets_with_metrics": int(len(accuracy_by_asset)),
                "completed_job_shards": int(completed_job_shards),
                "completed_combo_jobs": int(completed_combo_jobs),
            }
            summary = {
                "training_window_label": f"{int(training_window_months)}m",
                "training_window_months": int(training_window_months),
                "quality_status": str(quality["status"]),
                "quality_reason": str(quality["reason"]),
                "quality": quality,
                "config": {"interval": f"{int(interval)}m", "assets": list(assets), "asset_workers": max(1, int(args.asset_workers)), "model_threads": (max(1, int(args.model_threads)) if args.model_threads is not None else None), "worker_mode": "combo_job_process_shards", "job_shard_count": int(len(combo_shards)), "completed_job_shards": int(completed_job_shards), "completed_combo_jobs": int(completed_combo_jobs), "grouped_load_cache_entries": int(grouped_cache_entries), "history_start_ts": int(history_start_ts), "exact_history_required": True, "seed_ts": int(common_edge - int(args.forecast_days) * 86400), "accuracy_end_ts": int(common_edge), "forecast_target_month_start_utc": datetime.fromtimestamp(int(common_edge), tz=timezone.utc).replace(day=1, hour=0, minute=0, second=0, microsecond=0).isoformat()},
                "paths": {"run_root": str(run_dir)},
                "accuracy": {"by_asset_target_horizon": accuracy_by_asset},
            }
            run_summary = run_dir / "run_summary.json"
            write_json_atomic(run_summary, summary)
            emit_event_for_path(
                run_root,
                family="Bayesian_Numeric",
                model=str(model_key),
                stage="stage2",
                function_name="write_json_atomic",
                module_name=__name__,
                phase_name="write",
                status="completed",
                interval_minutes=int(interval),
                output_rows=len(accuracy_by_asset),
                asset_count=len(assets),
                output_path=str(run_summary),
            )
            runs.append({"paths": {"run_summary": str(run_summary)}})
    manifest = {"generated_utc": utc_now_iso(), "model_key": str(model_key), "feature_profile_json": (str(feature_profile_json) if feature_profile_json else None), "feature_profile_cohort_assets": list(assets), "selected_assets": list(assets), "seasonality_warmup": seasonality_warmup, "runs": runs}
    manifest_path = run_root / "diagnostic_manifest.json"
    write_json_atomic(manifest_path, manifest)
    emit_event_for_path(
        run_root,
        family="Bayesian_Numeric",
        model=str(model_key),
        stage="stage2",
        function_name="write_json_atomic",
        module_name=__name__,
        phase_name="artifact_handoff",
        status="completed",
        output_rows=len(runs),
        output_path=str(manifest_path),
        artifact_profile_source=(str(feature_profile_json) if feature_profile_json else ""),
    )
    if runs:
        analyze_manifest_for_model(str(model_key), manifest_path)
    return run_root


def main_for_model(model_key: str) -> None:
    run_root = run_stage_for_model(model_key)
    print(f"[{utc_now_iso()}] Bayesian Stage-2 diagnostic manifest written to {run_root}")


if __name__ == "__main__":
    main_for_model("dlm_tvp")
