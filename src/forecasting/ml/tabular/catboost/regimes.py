from __future__ import annotations

import json
import math
import os
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss

from src.forecasting.common.ohlcvt_source import read_ohlcvt
from src.forecasting.common.path_config import resolve_path, selected_profile
from src.features.scalar_features import (
    OHLCVT_PARQUET_ROOT,
    PARQUET_COMPRESSION,
    PARQUET_ROW_GROUP,
    PARQUET_ROOT as SCALAR_PARQUET_ROOT,
    assets_present_in_features,
    list_assets_from_ohlcvt,
    log as base_log,
    ohlcvt_bounds_from_parquet,
)

from src.forecasting.common.ml_module_utils import (
    acquire_single_run_lock,
    apply_head_floor,
    get_ts_floor_2021,
    get_module_logger,
    horizon_bars_from_minutes as shared_horizon_bars_from_minutes,
    is_tail_index,
    prune_pending_ts,
    select_regime_feature_columns,
)
from src.forecasting.ml.shared.regime_forecast_io import read_regime_labels
from src.forecasting.ml.shared.regime_targets import (
    append_axis_label_columns,
    axis_label_available_column,
    axis_label_id_column,
    future_axis_targets,
)
from src.forecasting.ml.tabular.shared.regime_forecast_runner import (
    RegimeForecastIOConfig,
    RegimeForecastTask,
    RegimeForecastWorkItem,
    RegimeFamilyModuleSpec,
    apply_regime_unit_results,
    build_regime_arg_parser,
    dispatch_regime_work_items,
    group_regime_tasks,
    plan_regime_group_work,
    regime_family_output_root,
    regime_forecast_horizon_output_max_ts,
    regime_build_run_manifest,
    regime_pending_prefix,
    regime_pending_ts_for_unit,
    regime_read_state,
    regime_replace_pending_for_unit,
    regime_unit_meta,
    regime_unit_key,
    regime_write_month_parts,
    regime_write_run_manifest,
    regime_write_state,
    resolve_regime_io_config,
    resolve_regime_runtime_snapshot,
    resolve_regime_run_plan,
    resolve_regime_tasks,
)
from src.regimes.contracts import (
    REGIME_AXIS_ORDER,
    REGIME_DEFAULT_FORECAST_INTERVALS,
    RegimeAxisTarget,
    axis_id_to_label,
    axis_target,
    forecast_ceiling_interval,
)
from src.forecasting.common.runtime_config import log_resolved_runtime

try:
    from catboost import CatBoostClassifier  # type: ignore
except Exception:
    CatBoostClassifier = None  # pragma: no cover


PIPELINE_PROFILE = selected_profile()
LOG_DIR = Path(resolve_path("log_root", profile=PIPELINE_PROFILE, required=False) or Path("logs"))
LOG_FILE = LOG_DIR / "catboost_regimes.log"

PARQUET_ROOT = Path(
    os.getenv("PIPELINE_PARQUET_CB_REGIMES_ROOT")
    or resolve_path("output_parquet_root", profile=PIPELINE_PROFILE, required=False)
    or os.getenv("PIPELINE_PARQUET_ROOT", "parquet")
)
OUTPUT_ROOT = PARQUET_ROOT / "CatBoost_Regimes"
STATE_ROOT = OUTPUT_ROOT / "state"

WATERMARK_FILE = STATE_ROOT / "ml_watermarks.json"
PENDING_FILE = STATE_ROOT / "ml_pending.json"
PROGRESS_FILE = STATE_ROOT / "ml_progress.json"
MANIFEST_FILE = STATE_ROOT / "ml_run_manifest.json"
REGIME_IO_CONFIG: Optional[RegimeForecastIOConfig] = None

REGIME_ROOT = Path(
    resolve_path("source_regime_root", profile=PIPELINE_PROFILE, required=False)
    or os.getenv("PIPELINE_PARQUET_REGIME_ROOT", str(PARQUET_ROOT))
)
SCALAR_ROOT = Path(
    resolve_path("source_feature_root", profile=PIPELINE_PROFILE, required=False)
    or os.getenv("PIPELINE_PARQUET_FEATURES_ROOT", str(SCALAR_PARQUET_ROOT))
)

FAMILY = "CatBoost"
DOMAIN = "Regimes"
TASK = "future_regime_state"

DEFAULT_INTERVALS = list(REGIME_DEFAULT_FORECAST_INTERVALS)
DEFAULT_HORIZON_MINUTES = [30, 240, 720]
# Candidate Training Windows / Selection:
# TRAIN_WINDOWS is ordered in bars. At time T, feasible windows are those with
# eligible_samples(T) >= W. Selection starts as soon as min(TRAIN_WINDOWS) is
# feasible, then can be re-evaluated (scheduled and/or when new windows become
# feasible) to upgrade to larger windows if validation performance improves.
TRAIN_WINDOWS = [256, 512, 1024, 2048]
WINDOW_REEVAL_DAYS = int(os.getenv("CB_REGIME_WINDOW_REEVAL_DAYS", "7"))
WINDOW_SWITCH_EPS = float(os.getenv("CB_REGIME_WINDOW_SWITCH_EPS", "0.005"))
WINDOW_SWITCH_COOLDOWN_DAYS = int(os.getenv("CB_REGIME_WINDOW_COOLDOWN_DAYS", "14"))
NEUTRAL_CLASS = 1
CLASS_LABELS = [0, 1, 2]

REGIME_RUNNER_SPEC = RegimeFamilyModuleSpec(
    module_slug="catboost_regimes",
    family_name=FAMILY,
    prediction_prefix="cb",
    log_prefix="[cb]",
    parquet_root_env="PIPELINE_PARQUET_CB_REGIMES_ROOT",
    progress_seconds_env="CB_REGIME_PROGRESS_SECONDS",
    source_start_env="CB_REGIME_SOURCE_START_TS",
    source_end_env="CB_REGIME_SOURCE_END_TS",
    work_start_env="CB_REGIME_WORK_START_TS",
    forecast_resume_edge_env="CB_REGIME_FORECAST_RESUME_EDGE_TS",
    default_unit_workers=1,
    default_model_threads=4,
    max_logical_threads=32,
    thread_env_vars=("CB_REGIME_N_JOBS",),
    thread_param_name="thread_count",
    default_intervals=REGIME_DEFAULT_FORECAST_INTERVALS,
    default_horizon_minutes=tuple(DEFAULT_HORIZON_MINUTES),
    axes=REGIME_AXIS_ORDER,
)

REFIT_EVAL_BARS = int(os.getenv("CB_REGIME_REFIT_EVAL_BARS", "500"))
REFIT_CHECK_K = int(os.getenv("CB_REGIME_REFIT_K", "2"))
REFIT_CHECK_M = int(os.getenv("CB_REGIME_REFIT_M", "3"))
REFIT_MIN_DAILY_SAMPLES = int(os.getenv("CB_REGIME_REFIT_MIN_DAILY_SAMPLES", "64"))
REFIT_LOGLOSS_MARGIN = float(os.getenv("CB_REGIME_REFIT_LOGLOSS_MARGIN", "0.02"))
REFIT_ACC_MARGIN = float(os.getenv("CB_REGIME_REFIT_ACC_MARGIN", "0.01"))
TS_FLOOR_2021 = get_ts_floor_2021()
PENDING_PRUNE_BUFFER_BARS = int(os.getenv("ML_PENDING_PRUNE_BUFFER_BARS", "2"))
PENDING_MAX_RETAIN = int(os.getenv("ML_PENDING_MAX_RETAIN", "2000"))

CB_PARAMS = {
    "loss_function": "MultiClass",
    "iterations": int(os.getenv("CB_REGIME_N_ESTIMATORS", "200")),
    "depth": int(os.getenv("CB_REGIME_MAX_DEPTH", "4")),
    "learning_rate": float(os.getenv("CB_REGIME_LEARNING_RATE", "0.05")),
    "subsample": float(os.getenv("CB_REGIME_SUBSAMPLE", "0.9")),
    "l2_leaf_reg": float(os.getenv("CB_REGIME_REG_LAMBDA", "1.0")),
    "random_seed": int(os.getenv("CB_REGIME_RANDOM_STATE", "17")),
    "thread_count": int(os.getenv("CB_REGIME_N_JOBS", str(REGIME_RUNNER_SPEC.default_model_threads))),
    "verbose": False,
}


_LOGGER: Any = None


def log(msg: str) -> None:
    global _LOGGER
    if _LOGGER is None:
        _LOGGER = get_module_logger(
            REGIME_RUNNER_SPEC.module_slug,
            LOG_FILE,
            base_log_fn=lambda m: base_log(f"[{REGIME_RUNNER_SPEC.module_slug}] {m}"),
        )
    _LOGGER(msg)


def _apply_io_config(io_config: RegimeForecastIOConfig) -> RegimeForecastIOConfig:
    global REGIME_IO_CONFIG, PARQUET_ROOT, OUTPUT_ROOT, STATE_ROOT
    global WATERMARK_FILE, PENDING_FILE, PROGRESS_FILE, MANIFEST_FILE
    global LOG_DIR, LOG_FILE, _LOGGER
    global REGIME_ROOT, SCALAR_ROOT, OHLCVT_PARQUET_ROOT
    REGIME_IO_CONFIG = io_config
    PARQUET_ROOT = Path(io_config.parquet_root)
    OUTPUT_ROOT = regime_family_output_root(io_config, REGIME_RUNNER_SPEC)
    STATE_ROOT = Path(io_config.state_root)
    STATE_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_DIR = Path(io_config.log_root)
    LOG_FILE = LOG_DIR / "catboost_regimes.log"
    _LOGGER = None
    WATERMARK_FILE = STATE_ROOT / "ml_watermarks.json"
    PENDING_FILE = STATE_ROOT / "ml_pending.json"
    PROGRESS_FILE = STATE_ROOT / "ml_progress.json"
    MANIFEST_FILE = STATE_ROOT / "ml_run_manifest.json"
    REGIME_ROOT = Path(io_config.regime_label_root)
    SCALAR_ROOT = Path(io_config.scalar_root)
    OHLCVT_PARQUET_ROOT = Path(io_config.ohlc_root)
    return io_config


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def month_start_ts(year: int, month: int) -> int:
    return int(datetime(year, month, 1, 0, 0, 0, tzinfo=timezone.utc).timestamp())


def _next_month(y: int, m: int) -> Tuple[int, int]:
    return (y + 1, 1) if m == 12 else (y, m + 1)


def next_month_start_ts(year: int, month: int) -> int:
    y2, m2 = _next_month(year, month)
    return month_start_ts(y2, m2)


def iter_months_between(start_ts: int, end_ts: int) -> Iterable[Tuple[int, int]]:
    if end_ts is None or start_ts is None or end_ts < start_ts:
        return
    dt = datetime.fromtimestamp(start_ts, tz=timezone.utc)
    y, m = dt.year, dt.month
    while True:
        yield (y, m)
        y, m = _next_month(y, m)
        if month_start_ts(y, m) > end_ts:
            break


def horizon_bars_from_minutes(horizon_minutes: int, interval_minutes: int) -> int:
    return shared_horizon_bars_from_minutes(interval_minutes=interval_minutes, horizon_minutes=horizon_minutes)


def _unit_key(horizon_minutes: int, asset: str, interval: int) -> str:
    return regime_unit_key(
        family=FAMILY,
        domain=DOMAIN,
        task=TASK,
        horizon_minutes=int(horizon_minutes),
        asset=str(asset),
        interval=int(interval),
    )


def _unit_meta(horizon_minutes: int, horizon_bars: int, asset: str, interval: int) -> dict:
    return regime_unit_meta(
        REGIME_RUNNER_SPEC,
        family=FAMILY,
        domain=DOMAIN,
        task=TASK,
        horizon_minutes=int(horizon_minutes),
        horizon_bars=int(horizon_bars),
        asset=str(asset),
        interval=int(interval),
    )


def _read_state() -> Tuple[dict, dict, dict]:
    return regime_read_state(STATE_ROOT)

def _write_state(watermarks: dict, pending: dict, progress: dict) -> None:
    regime_write_state(STATE_ROOT, watermarks, pending, progress)

def _pending_prefix(horizon_minutes: int, asset: str, interval: int) -> str:
    return regime_pending_prefix(
        family=FAMILY,
        domain=DOMAIN,
        task=TASK,
        horizon_minutes=int(horizon_minutes),
        asset=str(asset),
        interval=int(interval),
    )


def _pending_ts_for_unit(pending: dict, horizon_minutes: int, asset: str, interval: int) -> List[int]:
    return regime_pending_ts_for_unit(
        pending,
        family=FAMILY,
        domain=DOMAIN,
        task=TASK,
        horizon_minutes=int(horizon_minutes),
        asset=str(asset),
        interval=int(interval),
    )


def _replace_pending_for_unit(
    pending: dict,
    horizon_minutes: int,
    horizon_bars: int,
    asset: str,
    interval: int,
    ts_list: Sequence[int],
) -> None:
    regime_replace_pending_for_unit(
        pending,
        REGIME_RUNNER_SPEC,
        family=FAMILY,
        domain=DOMAIN,
        task=TASK,
        horizon_minutes=int(horizon_minutes),
        horizon_bars=int(horizon_bars),
        asset=str(asset),
        interval=int(interval),
        ts_list=ts_list,
    )


def _read_monthly_filtered(
    base_dir: Path,
    table_dir: str,
    start_ts: int,
    end_ts: int,
    asset: str,
    columns: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    if str(table_dir).lower().startswith("ohlcvt_"):
        try:
            interval_min = int(str(table_dir).split("_", 1)[1])
        except Exception:
            interval_min = 0
        if interval_min > 0:
            out = read_ohlcvt(
                asset=str(asset),
                interval_min=interval_min,
                start_ts=int(start_ts),
                end_ts=int(end_ts),
                columns=list(columns) if columns else None,
                root=Path(base_dir),
            )
            if not out.empty and "ts" in out.columns:
                out["ts"] = pd.to_numeric(out["ts"], errors="coerce").astype("int64")
                if "asset" in out.columns:
                    out = out.sort_values("ts").drop_duplicates(subset=["asset", "ts"], keep="last")
            if out.empty:
                cols = list(columns) if columns else ["ts", "asset"]
                return pd.DataFrame(columns=cols)
            return out

    frames: List[pd.DataFrame] = []
    for y, m in iter_months_between(start_ts, end_ts):
        month_dirs = [
            base_dir / table_dir / f"asset={asset}" / f"year={y}" / f"month={m:02d}",
            base_dir / table_dir / f"year={y}" / f"month={m:02d}",
        ]
        files: List[Path] = []
        for month_dir in month_dirs:
            if month_dir.exists():
                files.extend(sorted(month_dir.glob("*.parquet")))
        for p in files:
            try:
                df = pd.read_parquet(p, columns=columns)
            except Exception:
                continue
            if "asset" not in df.columns or "ts" not in df.columns:
                continue
            ts_num = pd.to_numeric(df["ts"], errors="coerce")
            df = df[
                (df["asset"].astype(str) == str(asset))
                & ts_num.ge(int(start_ts))
                & ts_num.le(int(end_ts))
            ]
            if not df.empty:
                frames.append(df)
    if not frames:
        cols = list(columns) if columns else ["ts", "asset"]
        return pd.DataFrame(columns=cols)
    out = pd.concat(frames, ignore_index=True)
    out["ts"] = pd.to_numeric(out["ts"], errors="coerce")
    out = out[out["ts"].notna()].copy()
    out["ts"] = out["ts"].astype("int64")
    out = out.sort_values(["ts"]).drop_duplicates(subset=["asset", "ts"], keep="last")
    return out


def _infer_regime3(df: pd.DataFrame) -> pd.Series:
    if "regime_3" in df.columns:
        src = df["regime_3"]
        if np.issubdtype(src.dtype, np.number):
            out = pd.to_numeric(src, errors="coerce").round().astype("Int64")
            return out.clip(lower=0, upper=2)
        s = src.astype(str).str.lower().str.strip()
        mapping = {"low": 0, "neutral": 1, "high": 2, "0": 0, "1": 1, "2": 2}
        return s.map(mapping).astype("Int64")
    if "trend_label" in df.columns:
        s = df["trend_label"].astype(str).str.lower().str.strip()
        mapping = {
            "down": 0,
            "mr": 0,
            "mean_reversion": 0,
            "low": 0,
            "neutral": 1,
            "flat": 1,
            "unknown": 1,
            "up": 2,
            "cont": 2,
            "trend": 2,
            "high": 2,
        }
        return s.map(mapping).fillna(1).astype("Int64")
    if "vol_label" in df.columns:
        s = df["vol_label"].astype(str).str.lower().str.strip()
        mapping = {"low": 0, "normal": 1, "neutral": 1, "unknown": 1, "high": 2}
        return s.map(mapping).fillna(1).astype("Int64")
    return pd.Series([1] * len(df), index=df.index, dtype="Int64")


def _load_unit_frame(asset: str, interval: int, stop_ts: int) -> pd.DataFrame:
    min_ts, _max_ts = ohlcvt_bounds_from_parquet(interval, asset, root=OHLCVT_PARQUET_ROOT)
    if min_ts is None:
        return pd.DataFrame()
    effective_start_ts = max(int(min_ts), int(TS_FLOOR_2021))
    if int(stop_ts) < int(effective_start_ts):
        return pd.DataFrame()
    ohlc = _read_monthly_filtered(
        base_dir=OHLCVT_PARQUET_ROOT,
        table_dir=f"ohlcvt_{interval}",
        start_ts=int(effective_start_ts),
        end_ts=int(stop_ts),
        asset=asset,
        columns=["ts", "asset"],
    )
    if ohlc.empty:
        return pd.DataFrame()
    step = int(interval) * 60
    full_ts = pd.DataFrame({"ts": np.arange(int(effective_start_ts), int(stop_ts) + step, step, dtype=np.int64)})
    base = full_ts.merge(ohlc[["ts", "asset"]], on="ts", how="left")
    base["asset"] = base["asset"].fillna(asset).astype("string")

    scalars = _read_monthly_filtered(
        base_dir=SCALAR_ROOT,
        table_dir=f"scalar_features_{interval}",
        start_ts=int(effective_start_ts),
        end_ts=int(stop_ts),
        asset=asset,
        columns=None,
    )
    if scalars.empty:
        scalar_cols = ["bias_feature"]
        scalars = base[["ts", "asset"]].copy()
        scalars["bias_feature"] = 0.0
    else:
        scalar_cols = [c for c in scalars.columns if c not in {"asset", "ts"}]
        if not scalar_cols:
            scalar_cols = ["bias_feature"]
            scalars["bias_feature"] = 0.0
        for c in scalar_cols:
            scalars[c] = pd.to_numeric(scalars[c], errors="coerce")
        scalars = scalars.sort_values("ts").drop_duplicates(subset=["asset", "ts"], keep="last")

    regimes = read_regime_labels(
        base_dir=REGIME_ROOT,
        ceiling_interval=int(forecast_ceiling_interval(interval)),
        start_ts=int(effective_start_ts),
        end_ts=int(stop_ts),
        asset=asset,
        columns=None,
    )

    if regimes.empty:
        reg = base[["ts", "asset"]].copy()
        reg["regime_3_idx"] = NEUTRAL_CLASS
        reg = append_axis_label_columns(reg)
    else:
        reg = regimes[["ts", "asset"]].copy()
        reg["regime_3_idx"] = _infer_regime3(regimes).fillna(NEUTRAL_CLASS).astype("int64")
        for axis in REGIME_AXIS_ORDER:
            label_col = axis_target(axis).label_column
            if label_col in regimes.columns:
                reg[label_col] = regimes[label_col]
        reg = append_axis_label_columns(reg)
        reg = reg.sort_values("ts").drop_duplicates(subset=["asset", "ts"], keep="last")

    out = base.merge(scalars[["ts", "asset"] + scalar_cols], on=["ts", "asset"], how="left")
    regime_cols = ["regime_3_idx"]
    for axis in REGIME_AXIS_ORDER:
        regime_cols.extend([axis_label_id_column(axis), axis_label_available_column(axis)])
    out = out.merge(reg[["ts", "asset"] + regime_cols], on=["ts", "asset"], how="left")
    out["regime_3_idx"] = pd.to_numeric(out["regime_3_idx"], errors="coerce").fillna(NEUTRAL_CLASS).astype("int64")
    for axis in REGIME_AXIS_ORDER:
        id_col = axis_label_id_column(axis)
        available_col = axis_label_available_column(axis)
        out[id_col] = pd.to_numeric(out[id_col], errors="coerce").fillna(axis_target(axis).unknown_id).astype("int64")
        out[available_col] = out[available_col].fillna(False).astype(bool)
    out = out.sort_values("ts").reset_index(drop=True)
    out[scalar_cols] = out[scalar_cols].ffill().fillna(0.0)
    return out


class ConstantClassifier:
    def __init__(self, klass: int):
        self.klass = int(klass)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.full((len(x),), self.klass, dtype=np.int64)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        out = np.zeros((len(x), 3), dtype=float)
        out[:, self.klass] = 1.0
        return out


def _fit_classifier(x: np.ndarray, y: np.ndarray) -> Any:
    y = np.asarray(y, dtype=np.int64)
    classes = np.unique(y)
    if x.ndim != 2 or x.shape[1] == 0:
        vals, counts = np.unique(y, return_counts=True)
        if len(vals) == 0:
            return ConstantClassifier(NEUTRAL_CLASS)
        return ConstantClassifier(int(vals[np.argmax(counts)]))
    if len(classes) <= 1:
        return ConstantClassifier(int(classes[0]) if len(classes) == 1 else NEUTRAL_CLASS)
    if CatBoostClassifier is None:
        # xgboost unavailable: fallback constant majority to preserve pipeline continuity.
        vals, counts = np.unique(y, return_counts=True)
        return ConstantClassifier(int(vals[np.argmax(counts)]))
    model = CatBoostClassifier(**CB_PARAMS)
    model.fit(x, y, verbose=False)
    return model


def _aligned_probability_vector(model: Any, proba: np.ndarray) -> np.ndarray:
    raw = np.asarray(proba, dtype=float).reshape(-1)
    if raw.size == len(CLASS_LABELS):
        return raw
    out = np.zeros((len(CLASS_LABELS),), dtype=float)
    classes = getattr(model, "classes_", None)
    if classes is not None:
        for klass, value in zip(np.asarray(classes).reshape(-1).tolist(), raw.tolist()):
            try:
                idx = int(klass)
            except Exception:
                continue
            if 0 <= idx < len(out):
                out[idx] = float(value)
    elif raw.size:
        out[: min(len(out), raw.size)] = raw[: len(out)]
    total = float(out.sum())
    return out / total if total > 0.0 else out


def _predict_one(model: Any, x_row: np.ndarray) -> Tuple[int, np.ndarray]:
    proba = _aligned_probability_vector(model, np.asarray(model.predict_proba(x_row.reshape(1, -1)), dtype=float)[0])
    pred = int(np.argmax(proba))
    return pred, proba


@dataclass
class WindowSelection:
    selected_window_bars: Optional[int]
    selection_timestamp: Optional[int]
    best_accuracy: Optional[float]
    best_logloss: Optional[float]


def _evaluate_window_candidates(
    x: np.ndarray,
    y_future: np.ndarray,
    future_available: np.ndarray,
    eligible_end: int,
    feasible_windows: Sequence[int],
    selection_ts: int,
) -> WindowSelection:
    if eligible_end <= 0:
        return WindowSelection(None, None, None, None)

    best_w: Optional[int] = None
    best_acc = -1.0
    best_ll = float("inf")
    for w in sorted(set(int(z) for z in feasible_windows)):
        start = eligible_end - w
        end = eligible_end
        if start < 0 or end <= start:
            continue
        valid_idx = np.flatnonzero(np.asarray(future_available[start:end], dtype=bool)) + start
        if len(valid_idx) < int(w):
            continue
        valid_idx = valid_idx[-int(w):]
        xw = x[valid_idx]
        yw = y_future[valid_idx]
        if len(xw) < 10:
            continue
        split = int(len(xw) * 0.8)
        if split < 2 or split >= len(xw):
            continue
        x_tr, y_tr = xw[:split], yw[:split]
        x_va, y_va = xw[split:], yw[split:]
        model = _fit_classifier(x_tr, y_tr)
        pred = np.asarray(model.predict(x_va), dtype=np.int64)
        prob = np.asarray(model.predict_proba(x_va), dtype=float)
        acc = float(accuracy_score(y_va, pred))
        ll = float(log_loss(y_va, prob, labels=CLASS_LABELS))
        if acc > best_acc or (math.isclose(acc, best_acc) and ll < best_ll):
            best_w = int(w)
            best_acc = acc
            best_ll = ll

    return WindowSelection(
        selected_window_bars=best_w,
        selection_timestamp=int(selection_ts) if best_w is not None else None,
        best_accuracy=(best_acc if best_w is not None else None),
        best_logloss=(best_ll if best_w is not None else None),
    )


def _majority_baseline_metrics(y_true: np.ndarray) -> Tuple[float, float]:
    if len(y_true) == 0:
        return 0.0, float("inf")
    vals, counts = np.unique(y_true, return_counts=True)
    maj = int(vals[np.argmax(counts)])
    pred = np.full((len(y_true),), maj, dtype=np.int64)
    prob = np.zeros((len(y_true), 3), dtype=float)
    prob[:, maj] = 1.0
    acc = float(accuracy_score(y_true, pred))
    ll = float(log_loss(y_true, prob, labels=CLASS_LABELS))
    return acc, ll


def _should_refit(day_flags: deque[bool]) -> bool:
    if len(day_flags) < REFIT_CHECK_M:
        return False
    return sum(1 for x in day_flags if x) >= REFIT_CHECK_K


def _is_better_window(
    cand_acc: Optional[float],
    cand_ll: Optional[float],
    curr_acc: Optional[float],
    curr_ll: Optional[float],
) -> bool:
    if cand_acc is None:
        return False
    if curr_acc is None:
        return True
    return (float(cand_acc) - float(curr_acc)) > float(WINDOW_SWITCH_EPS)


def _walk_forward_predict_axis(
    df: pd.DataFrame,
    horizon_bars: int,
    target: RegimeAxisTarget,
    prior_selected_window: Optional[int] = None,
    prior_selection_timestamp: Optional[int] = None,
) -> Tuple[pd.DataFrame, List[int], Dict[str, Any]]:
    if df.empty:
        return pd.DataFrame(columns=["ts", "asset"]), [], {}
    label_source_columns = {"regime_3", "regime_3_idx"}
    for axis in REGIME_AXIS_ORDER:
        label_source_columns.add(axis_label_id_column(axis))
        label_source_columns.add(axis_label_available_column(axis))
    x_cols = select_regime_feature_columns(df.columns, label_source_columns=label_source_columns)
    x = df[x_cols].to_numpy(dtype=float)
    ts = df["ts"].to_numpy(dtype=np.int64)

    n = len(df)
    future = future_axis_targets(df, target, horizon_bars)
    y_future = future.label_ids
    future_available = future.available
    tail_pending = future.tail_pending

    selected_w = int(prior_selected_window) if prior_selected_window is not None else None
    selection_timestamp = int(prior_selection_timestamp) if prior_selection_timestamp is not None else None
    selected_acc: Optional[float] = None
    selected_ll: Optional[float] = None
    feasible_seen: set[int] = set()
    last_window_eval_day: Optional[str] = None
    window_upgrade_count = 0

    pred = np.full((n,), NEUTRAL_CLASS, dtype=np.int64)
    probs = np.zeros((n, 3), dtype=float)
    probs[:, NEUTRAL_CLASS] = 1.0
    kind = np.array(["head_fill"] * n, dtype=object)
    pending_ts: List[int] = []

    model: Optional[Any] = None
    first_real_idx: Optional[int] = None
    eval_window: deque[Tuple[int, np.ndarray]] = deque(maxlen=REFIT_EVAL_BARS)
    day_flags: deque[bool] = deque(maxlen=REFIT_CHECK_M)
    last_check_day: Optional[str] = None
    last_refit_ts: Optional[int] = None

    for i in range(n):
        tail_insufficient = bool(tail_pending[i]) or is_tail_index(i=i, n_rows=n, horizon_bars=horizon_bars)
        if tail_insufficient:
            if i > 0:
                pred[i] = pred[i - 1]
                probs[i] = probs[i - 1]
            kind[i] = "tail_fill"
            pending_ts.append(int(ts[i]))
            continue

        eligible_end = i - horizon_bars + 1
        eligible_valid_idx = np.flatnonzero(np.asarray(future_available[:max(0, eligible_end)], dtype=bool))
        eligible_count = int(len(eligible_valid_idx))
        day = datetime.fromtimestamp(int(ts[i]), tz=timezone.utc).date().isoformat()

        feasible = [w for w in TRAIN_WINDOWS if eligible_count >= int(w)]
        if feasible:
            new_feasible = set(int(w) for w in feasible) - feasible_seen
            if last_window_eval_day is None:
                schedule_due = True
            else:
                d0 = datetime.fromisoformat(last_window_eval_day).date()
                d1 = datetime.fromisoformat(day).date()
                schedule_due = (d1 - d0).days >= max(1, int(WINDOW_REEVAL_DAYS))
            selected_not_feasible = selected_w is None or int(selected_w) not in set(feasible)
            cooldown_active = False
            if (not selected_not_feasible) and selected_w is not None and selection_timestamp is not None:
                sel_day = datetime.fromtimestamp(int(selection_timestamp), tz=timezone.utc).date()
                cooldown_active = (datetime.fromisoformat(day).date() - sel_day).days < max(0, int(WINDOW_SWITCH_COOLDOWN_DAYS))
            should_eval = (selected_not_feasible or bool(new_feasible) or schedule_due) and not cooldown_active
            if should_eval:
                eval_sel = _evaluate_window_candidates(
                    x=x,
                    y_future=y_future,
                    future_available=future_available,
                    eligible_end=eligible_end,
                    feasible_windows=feasible,
                    selection_ts=int(ts[i]),
                )
                feasible_seen.update(set(int(w) for w in feasible))
                last_window_eval_day = day
                cand_w = eval_sel.selected_window_bars
                if cand_w is not None:
                    if selected_w is None:
                        selected_w = int(cand_w)
                        selected_acc = eval_sel.best_accuracy
                        selected_ll = eval_sel.best_logloss
                        selection_timestamp = int(eval_sel.selection_timestamp) if eval_sel.selection_timestamp is not None else selection_timestamp
                    else:
                        larger_or_equal = int(cand_w) >= int(selected_w)
                        improves = _is_better_window(eval_sel.best_accuracy, eval_sel.best_logloss, selected_acc, selected_ll)
                        if larger_or_equal and improves:
                            if int(cand_w) > int(selected_w):
                                window_upgrade_count += 1
                            selected_w = int(cand_w)
                            selected_acc = eval_sel.best_accuracy
                            selected_ll = eval_sel.best_logloss
                            selection_timestamp = int(eval_sel.selection_timestamp) if eval_sel.selection_timestamp is not None else selection_timestamp
                            model = None

        if model is None and selected_w is not None and eligible_count >= int(selected_w):
            train_idx = eligible_valid_idx[-int(selected_w):]
            model = _fit_classifier(x[train_idx], y_future[train_idx])

        if model is None:
            kind[i] = "head_fill"
            pred[i] = NEUTRAL_CLASS
            probs[i] = np.array([0.0, 1.0, 0.0], dtype=float)
            continue

        p, pvec = _predict_one(model, x[i])
        pred[i] = int(p)
        probs[i] = np.asarray(pvec, dtype=float)
        kind[i] = "real"
        if first_real_idx is None:
            first_real_idx = i

        matured_idx = i - horizon_bars
        if matured_idx >= 0 and kind[matured_idx] == "real" and bool(future_available[matured_idx]):
            y_t = int(y_future[matured_idx])
            eval_window.append((y_t, np.asarray(probs[matured_idx], dtype=float)))

        if last_check_day is None:
            last_check_day = day
        if day != last_check_day:
            last_check_day = day
            if len(eval_window) >= REFIT_MIN_DAILY_SAMPLES:
                y_true = np.asarray([r[0] for r in eval_window], dtype=np.int64)
                y_prob = np.vstack([r[1] for r in eval_window]).astype(float)
                y_hat = np.asarray(np.argmax(y_prob, axis=1), dtype=np.int64)
                acc = float(accuracy_score(y_true, y_hat))
                ll = float(log_loss(y_true, y_prob, labels=CLASS_LABELS))
                b_acc, b_ll = _majority_baseline_metrics(y_true)
                bad = (acc + REFIT_ACC_MARGIN < b_acc) or (ll > (b_ll + REFIT_LOGLOSS_MARGIN))
                day_flags.append(bool(bad))
                if _should_refit(day_flags) and selected_w is not None and eligible_count >= int(selected_w):
                    train_idx = eligible_valid_idx[-int(selected_w):]
                    model = _fit_classifier(x[train_idx], y_future[train_idx])
                    last_refit_ts = int(ts[i])
                    day_flags.clear()

    out = df[["ts", "asset"]].copy()
    out["pred"] = pred.astype(np.int64)
    out["prob_0"] = probs[:, 0].astype(float)
    out["prob_1"] = probs[:, 1].astype(float)
    out["prob_2"] = probs[:, 2].astype(float)
    out["target_available"] = future_available.astype(bool)
    out["kind"] = kind.astype(str)
    meta = {
        "selected_window_bars": selected_w,
        "selection_timestamp": int(selection_timestamp) if selection_timestamp is not None else None,
        "selected_window_best_accuracy": selected_acc,
        "selected_window_best_logloss": selected_ll,
        "window_upgrade_count": int(window_upgrade_count),
        "first_real_prediction_ts": int(ts[first_real_idx]) if first_real_idx is not None else None,
        "last_refit_ts": last_refit_ts,
        "x_cols": x_cols,
        "axis": target.axis,
        "target_available_count": int(np.asarray(future_available, dtype=bool).sum()),
        "target_unknown_count": int((~np.asarray(future_available, dtype=bool) & ~np.asarray(tail_pending, dtype=bool)).sum()),
        "tail_pending_count": int(np.asarray(tail_pending, dtype=bool).sum()),
    }
    return out, sorted(set(int(t) for t in pending_ts)), meta


def _walk_forward_predict(
    df: pd.DataFrame,
    horizon_bars: int,
    horizon_minutes: int,
    prior_selected_window: Optional[int] = None,
    prior_selection_timestamp: Optional[int] = None,
) -> Tuple[pd.DataFrame, List[int], Dict[str, Any]]:
    if df.empty:
        return pd.DataFrame(columns=["ts", "asset"]), [], {}

    out = df[["ts", "asset"]].copy()
    pending_ts: set[int] = set()
    axis_meta: Dict[str, Any] = {}
    compatibility_trend: Optional[pd.DataFrame] = None

    for axis in REGIME_AXIS_ORDER:
        target = axis_target(axis)
        axis_df, axis_pending, meta = _walk_forward_predict_axis(
            df=df,
            horizon_bars=int(horizon_bars),
            target=target,
            prior_selected_window=prior_selected_window,
            prior_selection_timestamp=prior_selection_timestamp,
        )
        pending_ts.update(int(t) for t in axis_pending)
        axis_meta[axis] = meta

        pred_ids = pd.to_numeric(axis_df["pred"], errors="coerce").fillna(target.unknown_id).astype("int64")
        out[target.prediction_id_column("cb", horizon_minutes)] = pred_ids
        out[target.prediction_column("cb", horizon_minutes)] = [
            axis_id_to_label(axis, int(label_id), unknown_label=target.unknown_label) for label_id in pred_ids.tolist()
        ]
        for idx, prob_col in enumerate(target.probability_columns("cb", horizon_minutes)):
            out[prob_col] = pd.to_numeric(axis_df[f"prob_{idx}"], errors="coerce").fillna(0.0).astype(float)
        future = future_axis_targets(df, target, horizon_bars)
        out[target.target_available_column("cb", horizon_minutes)] = future.available.astype(bool)
        out[target.prediction_kind_column("cb", horizon_minutes)] = axis_df["kind"].astype(str)

        if axis == "trend":
            compatibility_trend = axis_df

    if compatibility_trend is not None:
        out[f"cb_pred_regime3_{horizon_minutes}m"] = pd.to_numeric(
            compatibility_trend["pred"], errors="coerce"
        ).fillna(NEUTRAL_CLASS).astype("int64")
        out[f"cb_prob_regime3_low_{horizon_minutes}m"] = pd.to_numeric(
            compatibility_trend["prob_0"], errors="coerce"
        ).fillna(0.0).astype(float)
        out[f"cb_prob_regime3_neutral_{horizon_minutes}m"] = pd.to_numeric(
            compatibility_trend["prob_1"], errors="coerce"
        ).fillna(0.0).astype(float)
        out[f"cb_prob_regime3_high_{horizon_minutes}m"] = pd.to_numeric(
            compatibility_trend["prob_2"], errors="coerce"
        ).fillna(0.0).astype(float)

    trend_meta = axis_meta.get("trend", {}) if isinstance(axis_meta.get("trend", {}), dict) else {}
    meta = {
        "selected_window_bars": trend_meta.get("selected_window_bars"),
        "selection_timestamp": trend_meta.get("selection_timestamp"),
        "selected_window_best_accuracy": trend_meta.get("selected_window_best_accuracy"),
        "selected_window_best_logloss": trend_meta.get("selected_window_best_logloss"),
        "window_upgrade_count": trend_meta.get("window_upgrade_count"),
        "last_refit_ts": trend_meta.get("last_refit_ts"),
        "axis_meta": axis_meta,
    }
    return out, sorted(pending_ts), meta


def _write_month_parts(
    month_frames: Dict[Tuple[int, int], List[pd.DataFrame]],
    interval: int,
    horizon_minutes: int,
) -> List[dict]:
    compatibility_columns = [
        f"cb_pred_regime3_{horizon_minutes}m",
        f"cb_prob_regime3_low_{horizon_minutes}m",
        f"cb_prob_regime3_neutral_{horizon_minutes}m",
        f"cb_prob_regime3_high_{horizon_minutes}m",
    ]
    return regime_write_month_parts(
        month_frames,
        output_root=OUTPUT_ROOT,
        spec=REGIME_RUNNER_SPEC,
        interval=int(interval),
        horizon_minutes=int(horizon_minutes),
        compatibility_columns=compatibility_columns,
        parquet_compression=PARQUET_COMPRESSION,
        parquet_row_group=PARQUET_ROW_GROUP,
    )


def _resolve_task_records(
    intervals: Sequence[int],
    assets_arg: str,
    horizon_minutes_list: Sequence[int],
) -> Tuple[RegimeForecastTask, ...]:
    return resolve_regime_tasks(
        intervals=intervals,
        horizon_minutes_list=horizon_minutes_list,
        assets_arg=str(assets_arg),
        list_assets_from_ohlcvt_fn=lambda interval: list_assets_from_ohlcvt(int(interval), root=OHLCVT_PARQUET_ROOT),
        assets_present_in_features_fn=lambda interval: assets_present_in_features(int(interval), root=SCALAR_ROOT),
        horizon_bars_from_minutes_fn=horizon_bars_from_minutes,
    )


def _resolve_tasks(
    intervals: Sequence[int],
    assets_arg: str,
    horizon_minutes_list: Sequence[int],
) -> List[Tuple[str, int, int, int]]:
    return [
        (task.asset, task.interval, task.horizon_minutes, task.horizon_bars)
        for task in _resolve_task_records(intervals, assets_arg, horizon_minutes_list)
    ]


def _get_stop_ts(asset: str, interval: int) -> Optional[int]:
    _mn, mx = ohlcvt_bounds_from_parquet(interval, asset, root=OHLCVT_PARQUET_ROOT)
    return int(mx) if mx is not None else None


def _get_horizon_output_tail(asset: str, interval: int, horizon_minutes: int) -> Optional[int]:
    return regime_forecast_horizon_output_max_ts(
        OUTPUT_ROOT,
        spec=REGIME_RUNNER_SPEC,
        interval=int(interval),
        asset=str(asset),
        horizon_minutes=int(horizon_minutes),
    )


def main() -> None:
    parser = build_regime_arg_parser(
        REGIME_RUNNER_SPEC,
        description="Walk-forward CatBoost regime forecasting from canonical axis labels.",
    )
    args = parser.parse_args()
    plan = resolve_regime_run_plan(REGIME_RUNNER_SPEC, args)
    args.unit_workers = int(plan.unit_workers)
    _apply_io_config(
        resolve_regime_io_config(
            REGIME_RUNNER_SPEC,
            profile=str(getattr(args, "profile", PIPELINE_PROFILE) or PIPELINE_PROFILE),
            log_fn=log,
        )
    )
    runtime_snapshot = resolve_regime_runtime_snapshot(
        REGIME_RUNNER_SPEC,
        args,
        output_root=OUTPUT_ROOT,
        diagnostic_root=LOG_DIR,
    )
    resolved_model_threads = int(runtime_snapshot["model_threads"])
    CB_PARAMS["thread_count"] = int(resolved_model_threads)
    log_resolved_runtime(REGIME_RUNNER_SPEC.module_slug, resolved=runtime_snapshot)

    intervals = list(plan.intervals)
    horizon_minutes_list = list(plan.horizon_minutes)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")

    try:
        task_records = _resolve_task_records(
            intervals=intervals,
            assets_arg=args.assets,
            horizon_minutes_list=horizon_minutes_list,
        )
    except ValueError as exc:
        raise SystemExit(f"[cb][error] {exc}")
    tasks = [
        (task.asset, task.interval, task.horizon_minutes, task.horizon_bars)
        for task in task_records
    ]
    if not tasks:
        log("[cb] no tasks to run")
        return
    try:
        acquire_single_run_lock(STATE_ROOT, "catboost_regimes")
    except RuntimeError as exc:
        log(f"[cb][skip] {exc}")
        return

    watermarks, pending, progress = _read_state()
    expected_task_keys = [_unit_key(hm, a, k) for (a, k, hm, _hb) in tasks]
    if progress.get("tasks") != expected_task_keys or not isinstance(progress.get("completed"), list):
        progress = {
            "run_id": run_id,
            "mode": args.mode,
            "tasks": expected_task_keys,
            "completed": [],
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        _write_state(watermarks, pending, progress)
    completed = set(progress.get("completed", []))

    manifest_parts: List[dict] = []
    rows_dropped_pre_floor_total = 0
    ts_start_by_asset: Dict[str, int] = {}

    for task_group in group_regime_tasks(task_records):
        interval = int(task_group.interval)
        horizon_minutes = int(task_group.horizon_minutes)
        horizon_bars = int(task_group.horizon_bars)
        buffer_seconds = int(interval) * 60 * max(0, int(PENDING_PRUNE_BUFFER_BARS))
        work_plan = plan_regime_group_work(
            task_group,
            spec=REGIME_RUNNER_SPEC,
            family=FAMILY,
            domain=DOMAIN,
            task=TASK,
            watermarks=watermarks,
            pending=pending,
            completed_unit_keys=sorted(completed),
            get_stop_ts_fn=_get_stop_ts,
            get_horizon_output_tail_fn=_get_horizon_output_tail,
            pending_prune_buffer_bars=PENDING_PRUNE_BUFFER_BARS,
            pending_max_retain=(PENDING_MAX_RETAIN if PENDING_MAX_RETAIN > 0 else None),
        )
        completed_changed = False
        for ukey in work_plan.completed_unit_keys:
            if ukey not in completed:
                completed.add(ukey)
                completed_changed = True
        for ukey in work_plan.reopened_unit_keys:
            if ukey in completed:
                completed.discard(ukey)
                completed_changed = True
        if completed_changed:
            progress["completed"] = sorted(completed)
            progress["updated_at"] = datetime.now(timezone.utc).isoformat()
            _write_state(watermarks, pending, progress)
        work_items = list(work_plan.work_items)

        def _compute_one(item: RegimeForecastWorkItem) -> Dict[str, Any]:
            asset = str(item.asset)
            ukey = str(item.unit_key)
            stop_ts = int(item.stop_ts)
            prior_wm_i = item.prior_watermark_ts
            prior_sel_ts_i = item.prior_selection_timestamp
            prior_pending_set_i = set(int(ts) for ts in item.prior_pending_ts)
            prior_sel_w_i = item.prior_selected_window_bars
            df = _load_unit_frame(asset=asset, interval=interval, stop_ts=int(stop_ts))
            df, floor_meta = apply_head_floor(df, TS_FLOOR_2021, asset_col="asset", ts_col="ts")
            rows_dropped = int(floor_meta.get("rows_dropped_pre_floor", 0))
            asset_start_map = floor_meta.get("ts_start_by_asset", {})
            if df.empty:
                return {"ukey": ukey, "asset": asset, "empty": True, "rows_dropped": rows_dropped}

            pred_df, next_pending_ts, meta = _walk_forward_predict(
                df=df,
                horizon_bars=int(horizon_bars),
                horizon_minutes=int(horizon_minutes),
                prior_selected_window=prior_sel_w_i,
                prior_selection_timestamp=prior_sel_ts_i,
            )

            if args.mode == "backfill" or prior_wm_i is None:
                to_write = pred_df.copy()
            else:
                to_write = pred_df[(pred_df["ts"].astype(int) > int(prior_wm_i)) | (pred_df["ts"].astype(int).isin(prior_pending_set_i))].copy()

            source_ts_index = set(int(x) for x in df["ts"].astype(int).tolist())
            next_pending_clean = prune_pending_ts(
                next_pending_ts,
                last_written_ts=int(stop_ts),
                buffer_seconds=buffer_seconds,
                source_ts_index=source_ts_index,
                max_entries=(PENDING_MAX_RETAIN if PENDING_MAX_RETAIN > 0 else None),
            )

            rows_written_unit = int(len(to_write))
            asset_month_frames: Dict[Tuple[int, int], List[pd.DataFrame]] = {}
            if not to_write.empty:
                ts_dt = pd.to_datetime(to_write["ts"], unit="s", utc=True)
                to_write = to_write.copy()
                to_write["year"] = ts_dt.dt.year.astype(int)
                to_write["month"] = ts_dt.dt.month.astype(int)
                for (y, m), grp in to_write.groupby(["year", "month"], sort=True):
                    asset_month_frames.setdefault((int(y), int(m)), []).append(grp.drop(columns=["year", "month"]).copy())

            unit_meta = _unit_meta(horizon_minutes, horizon_bars, asset, interval)
            unit_meta.update(
                {
                    "ts_floor": int(TS_FLOOR_2021),
                    "ts_start": int(asset_start_map[str(asset)]) if isinstance(asset_start_map, dict) and str(asset) in asset_start_map else None,
                    "rows_dropped_pre_floor": int(rows_dropped),
                    "stop_ts": int(stop_ts),
                    "last_written_ts": int(stop_ts),
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                    "selected_window_bars": meta.get("selected_window_bars"),
                    "selection_timestamp": meta.get("selection_timestamp"),
                    "selected_window_best_accuracy": meta.get("selected_window_best_accuracy"),
                    "selected_window_best_logloss": meta.get("selected_window_best_logloss"),
                    "window_upgrade_count": meta.get("window_upgrade_count"),
                    "last_refit_ts": meta.get("last_refit_ts"),
                }
            )
            return {
                "ukey": ukey,
                "asset": asset,
                "rows_dropped": rows_dropped,
                "ts_start": unit_meta.get("ts_start"),
                "rows_written": rows_written_unit,
                "stop_ts": int(stop_ts),
                "unit_meta": unit_meta,
                "pending_clean": sorted(set(int(t) for t in next_pending_clean)),
                "asset_month_frames": asset_month_frames,
            }

        unit_results = dispatch_regime_work_items(
            work_items,
            unit_workers=int(args.unit_workers),
            compute_one_fn=_compute_one,
        )

        applied = apply_regime_unit_results(
            unit_results,
            spec=REGIME_RUNNER_SPEC,
            family=FAMILY,
            domain=DOMAIN,
            task=TASK,
            interval=interval,
            horizon_minutes=horizon_minutes,
            horizon_bars=horizon_bars,
            watermarks=watermarks,
            pending=pending,
            progress=progress,
            completed_unit_keys=completed,
            write_month_parts_fn=_write_month_parts,
            write_state_fn=_write_state,
            log_fn=log,
        )
        manifest_parts.extend(applied.manifest_parts)
        rows_dropped_pre_floor_total += applied.rows_dropped_pre_floor_total
        ts_start_by_asset.update(applied.ts_start_by_asset)
        completed = set(applied.completed_unit_keys)

    _write_state(watermarks, pending, progress)

    manifest = regime_build_run_manifest(
        REGIME_RUNNER_SPEC,
        plan,
        run_id=run_id,
        family=FAMILY,
        domain=DOMAIN,
        task=TASK,
        runtime_snapshot=runtime_snapshot,
        task_records=task_records,
        ts_floor=int(TS_FLOOR_2021),
        ts_start_by_asset=ts_start_by_asset,
        rows_dropped_pre_floor_total=rows_dropped_pre_floor_total,
        manifest_parts=manifest_parts,
    )
    regime_write_run_manifest(MANIFEST_FILE, manifest)
    log(f"[cb] run complete parts={len(manifest_parts)}")


if __name__ == "__main__":
    main()





