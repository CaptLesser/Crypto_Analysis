from __future__ import annotations

import argparse
import itertools
import os
import time
import warnings
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
    resolve_seasonality_profile,
    select_make_do_window,
    utc_now_iso,
    write_forecast_parts,
    write_json_atomic,
)
from src.forecasting.stats.shared.stats_model_utils import horizon_bars as _horizon_bars
from src.forecasting.stats.shared.stats_model_utils import parse_int_csv as _parse_int_csv
from src.forecasting.stats.shared.stats_model_utils import parse_str_csv as _parse_str_csv
from src.forecasting.stats.shared.stats_numeric_runner import StatsNumericModuleSpec, run_stats_numeric_module

try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX  # type: ignore
    from statsmodels.tools.sm_exceptions import ConvergenceWarning as StatsmodelsConvergenceWarning  # type: ignore
except Exception:
    SARIMAX = None  # pragma: no cover
    StatsmodelsConvergenceWarning = Warning  # type: ignore


LOG_FILE = default_stats_log_file("sarimax_forecaster.log")

PARQUET_ROOT = default_stats_source_parquet_root("PIPELINE_PARQUET_SARIMAX_ROOT", DEFAULT_PARQUET_ROOT)
FEATURE_ROOT = Path(os.getenv("PIPELINE_PARQUET_FEATURES_ROOT", str(DEFAULT_FEATURE_ROOT)))
FORECAST_ROOT = default_stats_forecast_root("PIPELINE_PARQUET_SARIMAX_ROOT", "Stats_SARIMAX", DEFAULT_PARQUET_ROOT)
STATE_ROOT = default_stats_state_root(FORECAST_ROOT, "sarimax_forecaster")
try:
    STATE_ROOT.mkdir(parents=True, exist_ok=True)
except Exception:
    pass
MANIFEST_FILE = STATE_ROOT / "sarimax_run_manifest.json"
SKIPPED_FILE = STATE_ROOT / "sarimax_skipped.json"

FAMILY = "SARIMAX"
DOMAIN = "Numerics"
FAMILY_TAG = "sarimax_forecaster"

DEFAULT_INTERVALS = [60, 240, 1440]
DEFAULT_HORIZON_MINUTES = [4320, 10080, 20160]
MIN_TRAIN_BARS = 768
TRAIN_WINDOWS_BARS = [1536, 3072, 6144]

P_MAX = int(os.getenv("SARIMAX_P_MAX", "3"))
Q_MAX = int(os.getenv("SARIMAX_Q_MAX", "3"))
D_MAX = int(os.getenv("SARIMAX_D_MAX", "1"))
P_SEAS_MAX = int(os.getenv("SARIMAX_P_SEAS_MAX", "1"))
Q_SEAS_MAX = int(os.getenv("SARIMAX_Q_SEAS_MAX", "1"))
D_SEAS_MAX = int(os.getenv("SARIMAX_D_SEAS_MAX", "1"))
MAX_SEARCH_CANDIDATES = int(os.getenv("SARIMAX_MAX_SEARCH_CANDIDATES", "64"))
FIT_METHOD = str(os.getenv("SARIMAX_FIT_METHOD", "lbfgs") or "lbfgs")
FIT_MAXITER = int(os.getenv("SARIMAX_FIT_MAXITER", "50"))
FIT_RETRY_METHOD = str(os.getenv("SARIMAX_FIT_RETRY_METHOD", FIT_METHOD) or FIT_METHOD)
FIT_RETRY_MAXITER = int(os.getenv("SARIMAX_FIT_RETRY_MAXITER", "200"))
FIT_RETRY_ENABLED = str(os.getenv("SARIMAX_FIT_RETRY_ENABLED", "1")).strip().lower() not in {"0", "false", "no", "off"}
WARNING_MESSAGE_SAMPLE_LIMIT = int(os.getenv("SARIMAX_WARNING_MESSAGE_SAMPLE_LIMIT", "3"))


def log(msg: str) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    line = f"[{ts}] {msg}"
    print(line)
    try:
        with LOG_FILE.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass
    base_log(f"[sarimax_forecaster] {msg}")


def _build_exog(n: int, seasonal_period_bars: Optional[int], start_index: int = 0) -> Optional[np.ndarray]:
    if seasonal_period_bars is None or int(seasonal_period_bars) <= 1:
        return None
    t = np.arange(int(start_index), int(start_index) + int(n), dtype=float)
    per = float(seasonal_period_bars)
    return np.column_stack([np.sin(2.0 * np.pi * t / per), np.cos(2.0 * np.pi * t / per)])


def _candidate_specs(use_seasonal: bool, seasonal_period: Optional[int]) -> List[Tuple[Tuple[int, int, int], Tuple[int, int, int, int]]]:
    orders = list(itertools.product(range(0, P_MAX + 1), range(0, D_MAX + 1), range(0, Q_MAX + 1)))
    seasonal_orders: List[Tuple[int, int, int, int]]
    if use_seasonal and seasonal_period is not None and int(seasonal_period) > 1:
        seasonal_orders = [
            (P, D, Q, int(seasonal_period))
            for P, D, Q in itertools.product(range(0, P_SEAS_MAX + 1), range(0, D_SEAS_MAX + 1), range(0, Q_SEAS_MAX + 1))
        ]
    else:
        seasonal_orders = [(0, 0, 0, 0)]
    out = [(o, so) for o in orders for so in seasonal_orders]
    out = sorted(out, key=lambda x: (x[0][0], x[0][1], x[0][2], x[1][0], x[1][1], x[1][2], x[1][3]))
    return out[: max(1, int(MAX_SEARCH_CANDIDATES))]


def _mle_retvals_payload(res: Any) -> Dict[str, Any]:
    retvals = getattr(res, "mle_retvals", {}) or {}
    if not isinstance(retvals, dict):
        return {}
    payload: Dict[str, Any] = {}
    for key in ("converged", "warnflag", "iterations", "fcalls", "gopt"):
        value = retvals.get(key)
        if value is None:
            continue
        if key == "gopt":
            try:
                arr = np.asarray(value, dtype=float)
                payload["gopt_abs_max"] = float(np.nanmax(np.abs(arr))) if arr.size else None
            except Exception:
                continue
            continue
        if isinstance(value, (bool, np.bool_)):
            payload[str(key)] = bool(value)
        elif isinstance(value, (int, np.integer)):
            payload[str(key)] = int(value)
        elif isinstance(value, (float, np.floating)):
            payload[str(key)] = float(value)
        else:
            payload[str(key)] = str(value)
    return payload


def _warning_messages(caught: Sequence[Any]) -> List[str]:
    out: List[str] = []
    for item in list(caught):
        try:
            if not issubclass(item.category, StatsmodelsConvergenceWarning):
                continue
        except Exception:
            continue
        message = str(item.message).strip()
        if message:
            out.append(message[:300])
        if len(out) >= max(0, int(WARNING_MESSAGE_SAMPLE_LIMIT)):
            break
    return out


def _fit_sarimax_once(model: Any, *, method: str, maxiter: int, start_params: Optional[Any] = None) -> Tuple[Any, Dict[str, Any]]:
    started = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", StatsmodelsConvergenceWarning)
        kwargs: Dict[str, Any] = {"disp": False, "method": str(method), "maxiter": int(maxiter)}
        if start_params is not None:
            kwargs["start_params"] = start_params
        res = model.fit(**kwargs)
    warning_count = sum(1 for item in caught if issubclass(item.category, StatsmodelsConvergenceWarning))
    retvals = _mle_retvals_payload(res)
    return res, {
        "method": str(method),
        "maxiter": int(maxiter),
        "elapsed_s": round(float(time.perf_counter() - started), 6),
        "convergence_warning_count": int(warning_count),
        "warning_messages": _warning_messages(caught),
        **retvals,
    }


def _fit_sarimax(model: Any) -> Tuple[Any, Dict[str, Any]]:
    res, initial_diag = _fit_sarimax_once(model, method=FIT_METHOD, maxiter=FIT_MAXITER)
    initial_converged = bool(initial_diag.get("converged", True))
    needs_retry = bool(FIT_RETRY_ENABLED) and (
        int(initial_diag.get("convergence_warning_count", 0) or 0) > 0 or not initial_converged
    )
    if not needs_retry:
        return res, {
            **initial_diag,
            "initial_convergence_warning_count": int(initial_diag.get("convergence_warning_count", 0) or 0),
            "initial_converged": initial_converged,
            "retry_count": 0,
            "retry_resolved": False,
        }
    start_params = getattr(res, "params", None)
    try:
        retry_res, retry_diag = _fit_sarimax_once(
            model,
            method=FIT_RETRY_METHOD,
            maxiter=max(int(FIT_RETRY_MAXITER), int(FIT_MAXITER)),
            start_params=start_params,
        )
    except Exception:
        return res, {
            **initial_diag,
            "initial_convergence_warning_count": int(initial_diag.get("convergence_warning_count", 0) or 0),
            "initial_converged": initial_converged,
            "retry_count": 1,
            "retry_failed": True,
            "retry_resolved": False,
        }
    retry_converged = bool(retry_diag.get("converged", True))
    retry_warning_count = int(retry_diag.get("convergence_warning_count", 0) or 0)
    initial_warning_count = int(initial_diag.get("convergence_warning_count", 0) or 0)
    retry_is_better = (retry_converged and not initial_converged) or retry_warning_count < initial_warning_count
    selected_res = retry_res if retry_is_better else res
    selected_diag = retry_diag if retry_is_better else initial_diag
    return selected_res, {
        **selected_diag,
        "initial_convergence_warning_count": initial_warning_count,
        "initial_converged": initial_converged,
        "retry_convergence_warning_count": retry_warning_count,
        "retry_converged": retry_converged,
        "retry_count": 1,
        "retry_resolved": bool(
            (not bool(selected_diag.get("convergence_warning_count", 0)) and bool(selected_diag.get("converged", True)))
            and (initial_warning_count > 0 or not initial_converged)
        ),
        "selected_retry": bool(retry_is_better),
    }


def _choose_spec(
    y_train: pd.Series,
    seasonal_period: Optional[int],
) -> Dict[str, Any]:
    if SARIMAX is None:
        raise RuntimeError("statsmodels unavailable")
    use_seasonal = seasonal_period is not None and int(seasonal_period) > 1
    exog = _build_exog(n=len(y_train), seasonal_period_bars=seasonal_period, start_index=0)
    best: Optional[Dict[str, Any]] = None
    search_started = time.perf_counter()
    candidate_count = 0
    failed_candidate_count = 0
    converged_candidate_count = 0
    nonconverged_candidate_count = 0
    search_warning_count = 0
    retry_count = 0
    retry_resolved_count = 0
    for order, sorder in _candidate_specs(use_seasonal=use_seasonal, seasonal_period=seasonal_period):
        candidate_count += 1
        try:
            model = SARIMAX(
                endog=y_train.astype(float).to_numpy(),
                exog=exog,
                order=order,
                seasonal_order=sorder,
                trend="c",
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            res, fit_diag = _fit_sarimax(model)
            converged = bool(fit_diag.get("converged", (getattr(res, "mle_retvals", {}) or {}).get("converged", True)))
            if converged:
                converged_candidate_count += 1
            else:
                nonconverged_candidate_count += 1
            search_warning_count += int(fit_diag.get("convergence_warning_count", 0) or 0)
            retry_count += int(fit_diag.get("retry_count", 0) or 0)
            retry_resolved_count += int(bool(fit_diag.get("retry_resolved", False)))
            cand = {
                "order": tuple(int(x) for x in order),
                "seasonal_order": tuple(int(x) for x in sorder),
                "aic": float(res.aic) if res.aic is not None else float("inf"),
                "bic": float(res.bic) if res.bic is not None else float("inf"),
                "converged": bool(converged),
                "spec_search_selected_convergence_warning_count": int(fit_diag.get("convergence_warning_count", 0) or 0),
                "spec_search_selected_retry_count": int(fit_diag.get("retry_count", 0) or 0),
            }
        except Exception:
            failed_candidate_count += 1
            continue
        if best is None:
            best = cand
            continue
        rank_new = (not bool(cand["converged"]), cand["aic"], cand["bic"], cand["order"], cand["seasonal_order"])
        rank_old = (not bool(best["converged"]), best["aic"], best["bic"], best["order"], best["seasonal_order"])
        if rank_new < rank_old:
            best = cand
    if best is None:
        raise RuntimeError("no_sarimax_spec_found")
    best.update(
        {
            "spec_search_candidate_count": int(candidate_count),
            "spec_search_failed_candidate_count": int(failed_candidate_count),
            "spec_search_converged_candidate_count": int(converged_candidate_count),
            "spec_search_nonconverged_candidate_count": int(nonconverged_candidate_count),
            "spec_search_convergence_warning_count": int(search_warning_count),
            "spec_search_retry_count": int(retry_count),
            "spec_search_retry_resolved_count": int(retry_resolved_count),
            "spec_search_elapsed_s": round(float(time.perf_counter() - search_started), 6),
        }
    )
    return best


def _fit_forecast(
    y_train: pd.Series,
    steps: int,
    conf_alpha: float,
    spec: Dict[str, Any],
    seasonal_period: Optional[int],
) -> Tuple[float, float, float, Dict[str, Any]]:
    order = tuple(int(x) for x in spec["order"])
    sorder = tuple(int(x) for x in spec["seasonal_order"])
    exog = _build_exog(n=len(y_train), seasonal_period_bars=seasonal_period, start_index=0)
    model = SARIMAX(
        endog=y_train.astype(float).to_numpy(),
        exog=exog,
        order=order,
        seasonal_order=sorder,
        trend="c",
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    res, fit_diag = _fit_sarimax(model)
    f_exog = _build_exog(n=int(steps), seasonal_period_bars=seasonal_period, start_index=len(y_train))
    pred = res.get_forecast(steps=int(steps), exog=f_exog)
    yhat = float(np.asarray(pred.predicted_mean)[-1])
    ci = pred.conf_int(alpha=float(conf_alpha))
    lo = float(np.asarray(ci)[:, 0][-1])
    hi = float(np.asarray(ci)[:, 1][-1])
    meta = {
        "order": list(order),
        "seasonal_order": list(sorder),
        "aic": float(res.aic) if res.aic is not None else None,
        "bic": float(res.bic) if res.bic is not None else None,
        "converged": bool(fit_diag.get("converged", (getattr(res, "mle_retvals", {}) or {}).get("converged", True))),
        "fit_convergence_warning_count": int(fit_diag.get("convergence_warning_count", 0) or 0),
        "fit_initial_convergence_warning_count": int(fit_diag.get("initial_convergence_warning_count", 0) or 0),
        "fit_initial_converged": bool(fit_diag.get("initial_converged", True)),
        "fit_retry_count": int(fit_diag.get("retry_count", 0) or 0),
        "fit_retry_resolved": bool(fit_diag.get("retry_resolved", False)),
        "fit_selected_retry": bool(fit_diag.get("selected_retry", False)),
        "fit_method": str(fit_diag.get("method", "")),
        "fit_maxiter": int(fit_diag.get("maxiter", 0) or 0),
        "fit_elapsed_s": float(fit_diag.get("elapsed_s", 0.0) or 0.0),
        "fit_warnflag": fit_diag.get("warnflag"),
        "fit_iterations": fit_diag.get("iterations"),
        "spec_search_convergence_warning_count": int(spec.get("spec_search_convergence_warning_count", 0) or 0),
        "spec_search_selected_convergence_warning_count": int(spec.get("spec_search_selected_convergence_warning_count", 0) or 0),
        "spec_search_candidate_count": int(spec.get("spec_search_candidate_count", 0) or 0),
        "spec_search_converged_candidate_count": int(spec.get("spec_search_converged_candidate_count", 0) or 0),
        "spec_search_nonconverged_candidate_count": int(spec.get("spec_search_nonconverged_candidate_count", 0) or 0),
        "spec_search_failed_candidate_count": int(spec.get("spec_search_failed_candidate_count", 0) or 0),
        "spec_search_retry_count": int(spec.get("spec_search_retry_count", 0) or 0),
        "spec_search_retry_resolved_count": int(spec.get("spec_search_retry_resolved_count", 0) or 0),
        "spec_search_elapsed_s": float(spec.get("spec_search_elapsed_s", 0.0) or 0.0),
        "convergence_warning_count": int(fit_diag.get("convergence_warning_count", 0) or 0) + int(spec.get("spec_search_convergence_warning_count", 0) or 0),
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
        return {"unit_status": "skipped", "reason": "missing_feature_rows", "asset": asset, "edge_ts": int(edge_ts)}

    ts = df["ts"].astype("int64").to_numpy()
    y = pd.to_numeric(df[target_col], errors="coerce")
    origins = sorted(set(int(t) for t in ts if int(origin_start) <= int(t) <= int(target_tail)))
    if not origins:
        return {"unit_status": "done", "asset": asset, "edge_ts": int(edge_ts), "rows": [], "fit_meta": {"no_origins": True}}
    if bool(predict_latest_only) and origins:
        origins = [int(origins[-1])]

    seasonality = resolve_seasonality_profile(parquet_root=PARQUET_ROOT, interval_minutes=interval, asset=asset)
    seasonal_period = seasonality.seasonal_period_bars if seasonality.usable else None

    spec_by_window: Dict[int, Dict[str, Any]] = {}
    rows: List[Dict[str, Any]] = []
    skipped_origins = 0
    last_fit_meta: Dict[str, Any] = {}
    spec_search_window_count = 0
    spec_search_elapsed_s_total = 0.0
    spec_search_candidate_count = 0
    spec_search_failed_candidate_count = 0
    spec_search_converged_candidate_count = 0
    spec_search_nonconverged_candidate_count = 0
    spec_search_convergence_warning_count = 0
    spec_search_retry_count = 0
    spec_search_retry_resolved_count = 0
    fit_elapsed_s_total = 0.0
    fit_retry_count = 0
    fit_retry_resolved_count = 0
    fit_initial_convergence_warning_count = 0
    fit_convergence_warning_count = 0
    nonconverged_fit_count = 0
    fit_exception_count = 0
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
        if int(win) not in spec_by_window:
            try:
                spec_by_window[int(win)] = _choose_spec(y_train=y_train, seasonal_period=seasonal_period)
                spec = spec_by_window[int(win)]
                spec_search_window_count += 1
                spec_search_elapsed_s_total += float(spec.get("spec_search_elapsed_s", 0.0) or 0.0)
                spec_search_candidate_count += int(spec.get("spec_search_candidate_count", 0) or 0)
                spec_search_failed_candidate_count += int(spec.get("spec_search_failed_candidate_count", 0) or 0)
                spec_search_converged_candidate_count += int(spec.get("spec_search_converged_candidate_count", 0) or 0)
                spec_search_nonconverged_candidate_count += int(spec.get("spec_search_nonconverged_candidate_count", 0) or 0)
                spec_search_convergence_warning_count += int(spec.get("spec_search_convergence_warning_count", 0) or 0)
                spec_search_retry_count += int(spec.get("spec_search_retry_count", 0) or 0)
                spec_search_retry_resolved_count += int(spec.get("spec_search_retry_resolved_count", 0) or 0)
            except Exception:
                skipped_origins += 1
                continue
        spec = spec_by_window[int(win)]
        try:
            yhat, lo, hi, fit_meta = _fit_forecast(
                y_train=y_train,
                steps=int(horizon_bars),
                conf_alpha=float(conf_alpha),
                spec=spec,
                seasonal_period=seasonal_period,
            )
        except Exception:
            skipped_origins += 1
            fit_exception_count += 1
            continue
        fit_elapsed_s_total += float(fit_meta.get("fit_elapsed_s", 0.0) or 0.0)
        fit_retry_count += int(fit_meta.get("fit_retry_count", 0) or 0)
        fit_retry_resolved_count += int(bool(fit_meta.get("fit_retry_resolved", False)))
        fit_initial_convergence_warning_count += int(fit_meta.get("fit_initial_convergence_warning_count", 0) or 0)
        fit_convergence_warning_count += int(fit_meta.get("fit_convergence_warning_count", 0) or 0)
        if not bool(fit_meta.get("converged", True)):
            nonconverged_fit_count += 1
        rows.append(
            {
                "ts": int(origin_ts),
                "asset": str(asset),
                "_stats_actual": float(y.iloc[idx]) if pd.notna(y.iloc[idx]) else None,
                f"sarimax_{task}_H{int(horizon_minutes)}_yhat": float(yhat),
                f"sarimax_{task}_H{int(horizon_minutes)}_lo": float(lo),
                f"sarimax_{task}_H{int(horizon_minutes)}_hi": float(hi),
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
            "fit_success_count": int(len(rows)),
            "fit_exception_count": int(fit_exception_count),
            "fit_elapsed_s_total": round(float(fit_elapsed_s_total), 6),
            "fit_elapsed_s_mean": round(float(fit_elapsed_s_total) / float(len(rows)), 6) if rows else None,
            "fit_retry_count": int(fit_retry_count),
            "fit_retry_resolved_count": int(fit_retry_resolved_count),
            "fit_initial_convergence_warning_count": int(fit_initial_convergence_warning_count),
            "fit_convergence_warning_count": int(fit_convergence_warning_count),
            "nonconverged_fit_count": int(nonconverged_fit_count),
            "spec_search_window_count": int(spec_search_window_count),
            "spec_search_elapsed_s_total": round(float(spec_search_elapsed_s_total), 6),
            "spec_search_candidate_count": int(spec_search_candidate_count),
            "spec_search_failed_candidate_count": int(spec_search_failed_candidate_count),
            "spec_search_converged_candidate_count": int(spec_search_converged_candidate_count),
            "spec_search_nonconverged_candidate_count": int(spec_search_nonconverged_candidate_count),
            "spec_search_retry_count": int(spec_search_retry_count),
            "spec_search_retry_resolved_count": int(spec_search_retry_resolved_count),
            "spec_search_convergence_warning_count": int(spec_search_convergence_warning_count),
            "convergence_warning_count": int(fit_convergence_warning_count) + int(spec_search_convergence_warning_count),
        },
        "skipped_origins": int(skipped_origins),
    }


def main() -> None:
    run_stats_numeric_module(MODULE_SPEC)


MODULE_SPEC = StatsNumericModuleSpec(
    branch="sarimax",
    family=FAMILY,
    domain=DOMAIN,
    family_tag=FAMILY_TAG,
    model_id="stats_sarimax",
    model_version="2026-04-29",
    family_root_name="Stats_SARIMAX",
    family_root_env="PIPELINE_PARQUET_SARIMAX_ROOT",
    forecast_root=FORECAST_ROOT,
    state_root=STATE_ROOT,
    manifest_file=MANIFEST_FILE,
    skipped_file=SKIPPED_FILE,
    default_intervals=DEFAULT_INTERVALS,
    default_horizons=DEFAULT_HORIZON_MINUTES,
    default_tasks=CAPABILITY_MATRIX["sarimax"]["numerics"],
    min_train_bars=MIN_TRAIN_BARS,
    train_windows_bars=TRAIN_WINDOWS_BARS,
    process_unit_fn=_process_unit,
    log_fn=log,
    dependency_check_fn=lambda: "statsmodels is required for sarimax_forecaster.py" if SARIMAX is None else None,
    supports_conf_alpha=True,
    extra_process_kwargs_fn=lambda args: {"conf_alpha": float(args.conf_alpha)},
    manifest_extras_fn=lambda args: {"conf_alpha": float(args.conf_alpha)},
)


if __name__ == "__main__":
    main()
