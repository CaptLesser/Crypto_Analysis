from __future__ import annotations

import argparse
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.features.scalar_features import log as base_log
from src.forecasting.common.ml_module_utils import acquire_single_run_lock
from src.forecasting.common.runtime_config import cap_model_threads, get_model_threads, get_workers, log_resolved_runtime
from src.forecasting.common.stats_module_utils import (
    CAPABILITY_MATRIX,
    DEFAULT_BACKFILL_DAYS,
    DEFAULT_FEATURE_ROOT,
    DEFAULT_PARQUET_ROOT,
    DEFAULT_WORKERS,
    NUMERIC_TASK_TO_TARGET_COLUMN,
    default_stats_forecast_root,
    default_stats_log_file,
    default_stats_source_parquet_root,
    default_stats_state_root,
    interval_edge_ts,
    interval_min_ts,
    forecast_parts_tail_ts,
    make_stats_unit_key,
    read_feature_series_window,
    resolve_assets,
    select_make_do_window,
    utc_now_iso,
    write_forecast_parts,
    write_json_atomic,
)
from src.forecasting.stats.shared.stats_numeric_runner import StatsNumericModuleSpec, run_stats_numeric_module

try:
    from statsmodels.regression.quantile_regression import QuantReg  # type: ignore
except Exception:
    QuantReg = None  # pragma: no cover


LOG_FILE = default_stats_log_file("linear_quantile_reg.log")

PARQUET_ROOT = default_stats_source_parquet_root("PIPELINE_PARQUET_QR_ROOT", DEFAULT_PARQUET_ROOT)
FEATURE_ROOT = Path(os.getenv("PIPELINE_PARQUET_FEATURES_ROOT", str(DEFAULT_FEATURE_ROOT)))
FORECAST_ROOT = default_stats_forecast_root("PIPELINE_PARQUET_QR_ROOT", "Stats_QuantReg", DEFAULT_PARQUET_ROOT)
STATE_ROOT = default_stats_state_root(FORECAST_ROOT, "linear_quantile_reg")
try:
    STATE_ROOT.mkdir(parents=True, exist_ok=True)
except Exception:
    pass
MANIFEST_FILE = STATE_ROOT / "quantreg_run_manifest.json"
SKIPPED_FILE = STATE_ROOT / "quantreg_skipped.json"

FAMILY = "QuantReg"
DOMAIN = "Numerics"
FAMILY_TAG = "linear_quantile_reg"

DEFAULT_INTERVALS = [240, 1440]
DEFAULT_HORIZON_MINUTES = [1440, 4320, 10080]
MIN_TRAIN_BARS = 1024
TRAIN_WINDOWS_BARS = [2048, 4096, 8192]
DEFAULT_QUANTILES = [0.1, 0.5, 0.9]
LAGS = [1, 2, 3, 5, 8]


def log(msg: str) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    line = f"[{ts}] {msg}"
    print(line)
    try:
        with LOG_FILE.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass
    base_log(f"[linear_quantile_reg] {msg}")


def _parse_int_csv(raw: str, default_vals: Sequence[int]) -> List[int]:
    vals = [int(x.strip()) for x in str(raw).split(",") if x.strip()]
    return sorted(set(vals)) if vals else sorted(set(int(x) for x in default_vals))


def _parse_str_csv(raw: str, default_vals: Sequence[str]) -> List[str]:
    vals = [x.strip() for x in str(raw).split(",") if x.strip()]
    return sorted(set(vals)) if vals else sorted(set(str(x) for x in default_vals))


def _parse_quantiles(raw: str) -> List[float]:
    vals = [float(x.strip()) for x in str(raw).split(",") if x.strip()]
    q = vals if vals else list(DEFAULT_QUANTILES)
    q = sorted(set(float(x) for x in q if 0.0 < float(x) < 1.0))
    return q if q else list(DEFAULT_QUANTILES)


def _horizon_bars(horizon_minutes: int, interval_minutes: int) -> int:
    hm = int(horizon_minutes)
    iv = int(interval_minutes)
    if hm <= 0 or iv <= 0 or hm % iv != 0:
        raise ValueError(f"invalid horizon/interval pair: horizon={hm} interval={iv}")
    return hm // iv


def _build_supervised(y_vals: np.ndarray, lags: Sequence[int]) -> Tuple[np.ndarray, np.ndarray]:
    max_lag = int(max(lags))
    if len(y_vals) <= max_lag:
        return np.empty((0, len(lags) + 1), dtype=float), np.empty((0,), dtype=float)
    X: List[List[float]] = []
    y_out: List[float] = []
    for i in range(max_lag, len(y_vals)):
        row = [1.0]
        for lag in lags:
            row.append(float(y_vals[i - int(lag)]))
        X.append(row)
        y_out.append(float(y_vals[i]))
    return np.asarray(X, dtype=float), np.asarray(y_out, dtype=float)


def _fit_quant_models(y_train: pd.Series, quantiles: Sequence[float]) -> Dict[float, Any]:
    if QuantReg is None:
        raise RuntimeError("statsmodels unavailable")
    y_vals = np.asarray(y_train.astype(float))
    X, y_target = _build_supervised(y_vals=y_vals, lags=LAGS)
    if X.shape[0] < max(64, len(LAGS) * 8):
        raise RuntimeError("insufficient_supervised_rows")
    mod = QuantReg(y_target, X)
    out: Dict[float, Any] = {}
    for q in quantiles:
        out[float(q)] = mod.fit(q=float(q), max_iter=2000)
    return out


def _recursive_quantile_forecast(
    models: Dict[float, Any],
    y_train: pd.Series,
    quantiles: Sequence[float],
    steps: int,
) -> Dict[float, float]:
    hist = [float(x) for x in np.asarray(y_train.astype(float))]
    q_sorted = sorted(float(q) for q in quantiles)
    q_mid = min(q_sorted, key=lambda q: abs(q - 0.5))
    out_last = {q: float("nan") for q in q_sorted}
    for _ in range(int(steps)):
        row = np.asarray([1.0] + [hist[-int(l)] for l in LAGS], dtype=float).reshape(1, -1)
        step_vals = {q: float(models[q].predict(row)[0]) for q in q_sorted}
        for q in q_sorted:
            out_last[q] = step_vals[q]
        hist.append(step_vals[q_mid])
    return out_last


def _q_suffix(q: float) -> str:
    return f"p{int(round(float(q) * 100.0)):02d}"


def _process_unit(
    asset: str,
    interval: int,
    task: str,
    horizon_minutes: int,
    horizon_bars: int,
    backfill_days: int,
    predict_latest_only: bool,
    quantiles: Sequence[float],
) -> Dict[str, Any]:
    edge_ts = interval_edge_ts(asset=asset, interval_minutes=interval)
    if edge_ts is None:
        return {"unit_status": "skipped", "reason": "missing_edge_ts", "asset": asset}
    min_ts = interval_min_ts(asset=asset, interval_minutes=interval)
    if min_ts is None:
        return {"unit_status": "skipped", "reason": "missing_min_ts", "asset": asset, "edge_ts": int(edge_ts)}
    step = int(interval) * 60
    target_tail = int(edge_ts) - int(horizon_minutes) * 60
    if int(target_tail) < int(min_ts):
        return {"unit_status": "done", "asset": asset, "edge_ts": int(edge_ts), "rows": [], "fit_meta": {"no_closed_target": True}}
    dst_tail = forecast_parts_tail_ts(
        out_root=FORECAST_ROOT,
        interval_minutes=int(interval),
        family_tag=FAMILY_TAG,
        task=str(task),
        horizon_minutes=int(horizon_minutes),
        asset=str(asset),
    )
    if dst_tail is not None and int(dst_tail) > int(target_tail):
        raise RuntimeError(f"[hard-stop] asset={asset} interval={interval} task={task} h={horizon_minutes} dst_tail={dst_tail} > target_tail={target_tail}")
    if dst_tail is not None and int(dst_tail) == int(target_tail):
        return {"unit_status": "done", "asset": asset, "edge_ts": int(edge_ts), "rows": [], "fit_meta": {"at_edge": True}}

    target_col = NUMERIC_TASK_TO_TARGET_COLUMN[task]
    history_pad = max(TRAIN_WINDOWS_BARS) * int(interval) * 60 * 3
    origin_start = (
        max(int(min_ts), int(target_tail) - int(backfill_days) * 86400)
        if int(backfill_days) > 0 and dst_tail is None
        else (int(min_ts) if dst_tail is None else int(dst_tail) + int(step))
    )
    if int(origin_start) > int(target_tail):
        return {"unit_status": "done", "asset": asset, "edge_ts": int(edge_ts), "rows": [], "fit_meta": {"empty_range": True}}
    read_start = max(int(min_ts), int(origin_start) - int(history_pad))

    df = read_feature_series_window(
        root=FEATURE_ROOT,
        interval_minutes=interval,
        asset=asset,
        column=target_col,
        start_ts=int(read_start),
        end_ts=int(edge_ts),
        horizon_bars=int(horizon_bars),
    )
    if df.empty:
        return {"unit_status": "skipped", "reason": "missing_feature_rows", "asset": asset, "edge_ts": int(edge_ts)}

    ts = df["ts"].astype("int64").to_numpy()
    y = pd.to_numeric(df[target_col], errors="coerce")
    origins = sorted(set(int(t) for t in ts if int(origin_start) <= int(t) <= int(target_tail)))
    if not origins:
        return {"unit_status": "done", "asset": asset, "edge_ts": int(edge_ts), "rows": [], "fit_meta": {"no_origins": True}}
    if bool(predict_latest_only) and origins:
        origins = [int(origins[-1])]

    rows: List[Dict[str, Any]] = []
    skipped_origins = 0
    last_fit_meta: Dict[str, Any] = {}
    for origin_ts in origins:
        idx = int(np.searchsorted(ts, int(origin_ts), side="right") - 1)
        if idx < 0:
            skipped_origins += 1
            continue
        y_hist = y.iloc[: idx + 1]
        y_valid = y_hist.dropna()
        win = select_make_do_window(int(len(y_valid)), TRAIN_WINDOWS_BARS, MIN_TRAIN_BARS)
        if win is None:
            skipped_origins += 1
            continue
        y_train = y_valid.iloc[-int(win) :].astype(float)
        try:
            models = _fit_quant_models(y_train=y_train, quantiles=quantiles)
            fc = _recursive_quantile_forecast(models=models, y_train=y_train, quantiles=quantiles, steps=int(horizon_bars))
        except Exception:
            skipped_origins += 1
            continue
        row = {"ts": int(origin_ts), "asset": str(asset), "_stats_actual": float(y.iloc[idx]) if pd.notna(y.iloc[idx]) else None}
        for q in quantiles:
            row[f"qr_{task}_H{int(horizon_minutes)}_{_q_suffix(float(q))}"] = float(fc[float(q)])
        rows.append(row)
        last_fit_meta = {"selected_window_bars": int(win), "quantiles": [float(q) for q in quantiles]}

    if not rows:
        return {
            "unit_status": "skipped",
            "reason": "insufficient_data_for_any_origin",
            "asset": asset,
            "edge_ts": int(edge_ts),
            "skipped_origins": int(skipped_origins),
        }
    return {
        "unit_status": "done",
        "asset": asset,
        "edge_ts": int(edge_ts),
        "rows": rows,
        "fit_meta": {
            **last_fit_meta,
            "start_ts": int(origin_start),
            "target_tail_ts": int(target_tail),
            "dst_tail_ts": int(dst_tail) if dst_tail is not None else None,
        },
        "skipped_origins": int(skipped_origins),
    }


def main() -> None:
    run_stats_numeric_module(MODULE_SPEC)


def _add_quantile_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--quantiles", type=str, default=os.getenv("QR_QUANTILES", "0.1,0.5,0.9"))


def _resolved_quantiles(args: argparse.Namespace) -> List[float]:
    return sorted(set(_parse_quantiles(args.quantiles)) | {0.1, 0.5, 0.9})


MODULE_SPEC = StatsNumericModuleSpec(
    branch="quantreg",
    family=FAMILY,
    domain=DOMAIN,
    family_tag=FAMILY_TAG,
    model_id="stats_quantreg",
    model_version="2026-04-29",
    family_root_name="Stats_QuantReg",
    family_root_env="PIPELINE_PARQUET_QR_ROOT",
    forecast_root=FORECAST_ROOT,
    state_root=STATE_ROOT,
    manifest_file=MANIFEST_FILE,
    skipped_file=SKIPPED_FILE,
    default_intervals=DEFAULT_INTERVALS,
    default_horizons=DEFAULT_HORIZON_MINUTES,
    default_tasks=CAPABILITY_MATRIX["quantreg"]["numerics"],
    min_train_bars=MIN_TRAIN_BARS,
    train_windows_bars=TRAIN_WINDOWS_BARS,
    process_unit_fn=_process_unit,
    log_fn=log,
    dependency_check_fn=lambda: "statsmodels is required for linear_quantile_reg.py" if QuantReg is None else None,
    add_extra_args_fn=_add_quantile_args,
    extra_process_kwargs_fn=lambda args: {"quantiles": _resolved_quantiles(args)},
    manifest_extras_fn=lambda args: {"quantiles": _resolved_quantiles(args)},
)


if __name__ == "__main__":
    main()
