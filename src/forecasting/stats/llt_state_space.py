from __future__ import annotations

import argparse
import os
import time
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
    DEFAULT_CONF_ALPHA,
    DEFAULT_PARQUET_ROOT,
    DEFAULT_WORKERS,
    DEFAULT_FEATURE_ROOT,
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
    resolve_seasonality_profile,
    select_make_do_window,
    utc_now_iso,
    write_forecast_parts,
    write_json_atomic,
)
from src.forecasting.stats.shared.stats_numeric_runner import StatsNumericModuleSpec, run_stats_numeric_module

try:
    from statsmodels.tsa.statespace.structural import UnobservedComponents  # type: ignore
except Exception:
    UnobservedComponents = None  # pragma: no cover


LOG_FILE = default_stats_log_file("llt_state_space.log")

PARQUET_ROOT = default_stats_source_parquet_root("PIPELINE_PARQUET_LLT_ROOT", DEFAULT_PARQUET_ROOT)
FEATURE_ROOT = Path(os.getenv("PIPELINE_PARQUET_FEATURES_ROOT", str(DEFAULT_FEATURE_ROOT)))
FORECAST_ROOT = default_stats_forecast_root("PIPELINE_PARQUET_LLT_ROOT", "Stats_LLT", DEFAULT_PARQUET_ROOT)
STATE_ROOT = default_stats_state_root(FORECAST_ROOT, "llt_state_space")
try:
    STATE_ROOT.mkdir(parents=True, exist_ok=True)
except Exception:
    pass
MANIFEST_FILE = STATE_ROOT / "llt_run_manifest.json"
SKIPPED_FILE = STATE_ROOT / "llt_skipped.json"

FAMILY = "LLT"
DOMAIN = "Numerics"
FAMILY_TAG = "llt_state_space"

DEFAULT_INTERVALS = [60, 240, 1440]
DEFAULT_HORIZON_MINUTES = [1440, 4320, 10080]
MIN_TRAIN_BARS = 512
TRAIN_WINDOWS_BARS = [1024, 2048, 4096]


def log(msg: str) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    line = f"[{ts}] {msg}"
    print(line)
    try:
        with LOG_FILE.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass
    base_log(f"[llt_state_space] {msg}")


def _parse_int_csv(raw: str, default_vals: Sequence[int]) -> List[int]:
    vals = [int(x.strip()) for x in str(raw).split(",") if x.strip()]
    return sorted(set(vals)) if vals else sorted(set(int(x) for x in default_vals))


def _parse_str_csv(raw: str, default_vals: Sequence[str]) -> List[str]:
    vals = [x.strip() for x in str(raw).split(",") if x.strip()]
    return sorted(set(vals)) if vals else sorted(set(str(x) for x in default_vals))


def _horizon_bars(horizon_minutes: int, interval_minutes: int) -> int:
    hm = int(horizon_minutes)
    iv = int(interval_minutes)
    if hm <= 0 or iv <= 0 or hm % iv != 0:
        raise ValueError(f"invalid horizon/interval pair: horizon={hm} interval={iv}")
    return hm // iv


def _fit_predict_llt(
    y_train: pd.Series,
    steps: int,
    conf_alpha: float,
    seasonal_period_bars: Optional[int],
) -> Tuple[float, float, float, Dict[str, Any]]:
    if UnobservedComponents is None:
        raise RuntimeError("statsmodels unavailable")
    kwargs: Dict[str, Any] = {"level": "local linear trend"}
    if seasonal_period_bars is not None and int(seasonal_period_bars) > 1:
        kwargs["seasonal"] = int(seasonal_period_bars)
    fit_started = time.perf_counter()
    model = UnobservedComponents(endog=y_train.astype(float).to_numpy(), **kwargs)
    res = model.fit(disp=False)
    fit_elapsed_s = time.perf_counter() - fit_started
    pred = res.get_forecast(steps=int(steps))
    yhat = float(np.asarray(pred.predicted_mean)[-1])
    ci = pred.conf_int(alpha=float(conf_alpha))
    lo = float(np.asarray(ci)[:, 0][-1])
    hi = float(np.asarray(ci)[:, 1][-1])
    meta = {
        "aic": float(res.aic) if hasattr(res, "aic") and res.aic is not None else None,
        "bic": float(res.bic) if hasattr(res, "bic") and res.bic is not None else None,
        "converged": bool((getattr(res, "mle_retvals", {}) or {}).get("converged", True)),
        "fit_elapsed_s": float(fit_elapsed_s),
    }
    return yhat, lo, hi, meta


def _process_unit(
    asset: str,
    interval: int,
    task: str,
    horizon_minutes: int,
    horizon_bars: int,
    backfill_days: int,
    predict_latest_only: bool,
    conf_alpha: float,
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
        return {
            "unit_status": "skipped",
            "reason": "missing_feature_rows",
            "asset": asset,
            "edge_ts": int(edge_ts),
        }

    ts = df["ts"].astype("int64").to_numpy()
    y = pd.to_numeric(df[target_col], errors="coerce")
    origins = sorted(set(int(t) for t in ts if int(origin_start) <= int(t) <= int(target_tail)))
    if not origins:
        return {"unit_status": "done", "asset": asset, "edge_ts": int(edge_ts), "rows": [], "fit_meta": {"no_origins": True}}
    if bool(predict_latest_only) and origins:
        origins = [int(origins[-1])]

    seasonality = resolve_seasonality_profile(parquet_root=PARQUET_ROOT, interval_minutes=interval, asset=asset)
    seasonal_period = seasonality.seasonal_period_bars if seasonality.usable else None

    rows: List[Dict[str, Any]] = []
    skipped_origins = 0
    last_fit_meta: Dict[str, Any] = {}
    fit_elapsed_s_total = 0.0
    fit_count = 0
    for origin_ts in origins:
        idx = int(np.searchsorted(ts, int(origin_ts), side="right") - 1)
        if idx < 0:
            skipped_origins += 1
            continue
        y_hist = y.iloc[: idx + 1]
        y_valid = y_hist.dropna()
        win = select_make_do_window(
            valid_train_points=int(len(y_valid)),
            train_windows_bars=TRAIN_WINDOWS_BARS,
            min_train_bars=MIN_TRAIN_BARS,
        )
        if win is None:
            skipped_origins += 1
            continue
        y_train = y_valid.iloc[-int(win) :].astype(float)
        try:
            yhat, lo, hi, fit_meta = _fit_predict_llt(
                y_train=y_train,
                steps=int(horizon_bars),
                conf_alpha=float(conf_alpha),
                seasonal_period_bars=seasonal_period,
            )
        except Exception:
            skipped_origins += 1
            continue
        fit_count += 1
        fit_elapsed_s_total += float(fit_meta.get("fit_elapsed_s", 0.0) or 0.0)
        rows.append(
            {
                "ts": int(origin_ts),
                "asset": str(asset),
                "_stats_actual": float(y.iloc[idx]) if pd.notna(y.iloc[idx]) else None,
                f"llt_{task}_H{int(horizon_minutes)}_yhat": float(yhat),
                f"llt_{task}_H{int(horizon_minutes)}_lo": float(lo),
                f"llt_{task}_H{int(horizon_minutes)}_hi": float(hi),
            }
        )
        last_fit_meta = {
            "selected_window_bars": int(win),
            "seasonality_source": seasonality.source,
            "seasonality_used": bool(seasonality.usable),
            "seasonal_period_bars": int(seasonal_period) if seasonal_period else None,
            **fit_meta,
        }

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
            "origin_count": int(len(origins)),
            "fit_count": int(fit_count),
            "fit_elapsed_s_total": float(fit_elapsed_s_total),
            "seconds_per_origin": (float(fit_elapsed_s_total) / float(fit_count) if int(fit_count) > 0 else None),
            "forecast_rows": int(len(rows)),
        },
        "skipped_origins": int(skipped_origins),
    }


def main() -> None:
    run_stats_numeric_module(MODULE_SPEC)


MODULE_SPEC = StatsNumericModuleSpec(
    branch="llt",
    family=FAMILY,
    domain=DOMAIN,
    family_tag=FAMILY_TAG,
    model_id="stats_llt",
    model_version="2026-04-29",
    family_root_name="Stats_LLT",
    family_root_env="PIPELINE_PARQUET_LLT_ROOT",
    forecast_root=FORECAST_ROOT,
    state_root=STATE_ROOT,
    manifest_file=MANIFEST_FILE,
    skipped_file=SKIPPED_FILE,
    default_intervals=DEFAULT_INTERVALS,
    default_horizons=DEFAULT_HORIZON_MINUTES,
    default_tasks=CAPABILITY_MATRIX["llt"]["numerics"],
    min_train_bars=MIN_TRAIN_BARS,
    train_windows_bars=TRAIN_WINDOWS_BARS,
    process_unit_fn=_process_unit,
    log_fn=log,
    dependency_check_fn=lambda: "statsmodels is required for llt_state_space.py" if UnobservedComponents is None else None,
    supports_conf_alpha=True,
    extra_process_kwargs_fn=lambda args: {"conf_alpha": float(args.conf_alpha)},
    manifest_extras_fn=lambda args: {"conf_alpha": float(args.conf_alpha)},
)


if __name__ == "__main__":
    main()
