from __future__ import annotations

import argparse
import json
import math
import os
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from functools import lru_cache
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.forecasting.common.path_config import resolve_path, selected_profile
from src.forecasting.common.runtime_config import get_workers, log_resolved_runtime
from src.forecasting.ml.shared.numeric_forecast_io import (
    NumericForecastIOConfig,
    NumericForecastNamingConfig,
    expected_eval_columns,
    expected_forecast_columns,
    module_table,
    read_monthly_filtered,
    validated_module_month_parquet,
)
from src.forecasting.ml.tabular.shared.tabular_numeric_model_registry import (
    get_tabular_numeric_model_spec,
    load_model_task_metadata,
)


DEFAULT_MODEL_KEYS = ("xgboost", "lightgbm")
DEFAULT_TRAILING_WINDOWS_DAYS = (14, 30, 60, 90, 180)
DEFAULT_LOOKBACK_DAYS = 180
DEFAULT_CHUNK_DAYS = 14
DEFAULT_RANK_WINDOW_DAYS = 30
DEFAULT_RANK_MIN_PREDICTIONS = 10
DEFAULT_FLOOR_FRAC = 0.10
DEFAULT_UNIT_WORKERS = 4
DEFAULT_EPSILON_BY_TASK = {
    "log_return": 1e-4,
    "realized_vol": 1e-4,
    "true_range": 1e-4,
    "max_drawdown": 1e-4,
    "max_runup": 1e-4,
    "range_efficiency": 1e-3,
}
TRAILING_SUMMARY_COLUMNS = [
    "model_family",
    "asset",
    "interval_min",
    "task",
    "horizon_minutes",
    "window_days",
    "window_start_ts",
    "window_end_ts",
    "latest_realized_ts",
    "n_predictions",
    "hit_count_10",
    "hit_rate_10",
    "hit_count_20",
    "hit_rate_20",
    "hit_count_30",
    "hit_rate_30",
    "mae",
    "rmse",
    "median_abs_actual",
    "scale_floor",
    "epsilon_task",
    "floor_frac",
]
CHUNK_SUMMARY_COLUMNS = TRAILING_SUMMARY_COLUMNS + [
    "chunk_start_ts",
    "chunk_end_ts",
    "chunk_index_recent",
]
ROW_OUTPUT_COLUMNS = [
    "summary_family",
    "window_days",
    "chunk_start_ts",
    "chunk_end_ts",
    "chunk_index_recent",
    "model_family",
    "asset",
    "interval_min",
    "task",
    "horizon_minutes",
    "ts",
    "pred_value",
    "actual_value",
    "abs_error",
    "sq_error",
    "effective_scale",
    "relative_error_like",
    "hit_10",
    "hit_20",
    "hit_30",
]


@dataclass(frozen=True)
class FamilyRuntime:
    model_key: str
    parquet_root: Path
    naming: NumericForecastNamingConfig
    io_config: NumericForecastIOConfig
    numeric_tasks: List[str]
    task_short: Dict[str, str]
    task_label: Dict[str, str]


@dataclass(frozen=True)
class UnitDescriptor:
    asset: str
    interval_min: int
    task: str
    horizon_minutes: int


def _project_root() -> Path:
    return Path(__file__).resolve().parents[5]


def _pipeline_root() -> Path:
    profile = selected_profile()
    return Path(resolve_path("state_root", profile=profile, required=False) or resolve_path("output_parquet_root", profile=profile, required=False) or Path("."))


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _utc_now_run_token() -> str:
    return _utc_now().strftime("%Y%m%dT%H%M%SZ")


def _iso_now() -> str:
    return _utc_now().isoformat()


def _default_log_fn(msg: str) -> None:
    print(msg)


def _empty_ohlcvt(*args: Any, **kwargs: Any) -> pd.DataFrame:
    return pd.DataFrame()


def _empty_list_assets(*args: Any, **kwargs: Any) -> Sequence[str]:
    return []


def _none_ts(*args: Any, **kwargs: Any) -> Optional[int]:
    return None


def _parse_csv_ints(raw: str) -> List[int]:
    return [int(tok.strip()) for tok in str(raw).split(",") if tok.strip()]


def _parse_csv_strs(raw: str) -> List[str]:
    return [tok.strip() for tok in str(raw).split(",") if tok.strip()]


def _month_sort_key(path: Path) -> Tuple[int, int, str]:
    year = 0
    month = 0
    for part in path.parts:
        if part.startswith("year="):
            try:
                year = int(part.split("=", 1)[1])
            except Exception:
                year = 0
        elif part.startswith("month="):
            try:
                month = int(part.split("=", 1)[1])
            except Exception:
                month = 0
    return (year, month, path.name.lower())


@lru_cache(maxsize=8192)
def _parquet_columns(path_str: str) -> Tuple[str, ...]:
    try:
        return tuple(str(name) for name in pq.read_schema(path_str).names)
    except Exception:
        return ()


def task_epsilon_default(task: str) -> float:
    return float(DEFAULT_EPSILON_BY_TASK.get(str(task), 1e-4))


def build_family_runtime(model_key: str, *, project_root: Optional[Path] = None, log_fn: Optional[Callable[[str], None]] = None) -> FamilyRuntime:
    project_root = Path(project_root) if project_root else _project_root()
    log_fn = log_fn or _default_log_fn
    spec = get_tabular_numeric_model_spec(model_key)
    numeric_tasks, task_short, task_label = load_model_task_metadata(spec, project_root)
    profile = selected_profile()
    pipeline_root = _pipeline_root()
    parquet_root = Path(
        os.getenv(spec.parquet_root_env)
        or resolve_path("output_parquet_root", profile=profile, required=False)
        or os.getenv("PIPELINE_PARQUET_ROOT", "parquet")
    )
    ohlc_root = Path(resolve_path("source_ohlcvt_root", profile=profile, required=False) or parquet_root)
    scalar_root = Path(resolve_path("source_feature_root", profile=profile, required=False) or parquet_root)
    tmp_root = Path(resolve_path("tmp_root", profile=profile, required=False) or pipeline_root / "tmp")
    state_root = Path(resolve_path("state_root", profile=profile, required=False) or pipeline_root / "model_states")
    naming = NumericForecastNamingConfig(
        module_slug=spec.forecast_table_tag,
        forecast_table_tag=spec.forecast_table_tag,
        eval_table_tag=spec.eval_table_tag,
        prediction_prefix=spec.prediction_prefix,
        task_short=dict(task_short),
        task_label=dict(task_label),
        log_prefix=spec.progress_log_prefix,
    )
    io_config = NumericForecastIOConfig(
        naming=naming,
        parquet_root=parquet_root,
        staging_root=tmp_root / "recent_realized_accuracy_stage",
        state_root=state_root / "recent_realized_accuracy",
        scalar_root=scalar_root,
        ohlc_root=ohlc_root,
        parquet_compression=os.getenv("PIPELINE_PARQUET_COMPRESSION", "snappy"),
        parquet_row_group=int(os.getenv("PIPELINE_PARQUET_ROW_GROUP", "500000")),
        log_fn=log_fn,
        read_ohlcvt_fn=_empty_ohlcvt,
        list_assets_from_ohlcvt_fn=_empty_list_assets,
        first_ohlcvt_ts_fn=_none_ts,
        ohlcvt_max_ts_fn=_none_ts,
        feature_max_ts_fn=_none_ts,
    )
    return FamilyRuntime(
        model_key=str(spec.model_key),
        parquet_root=parquet_root,
        naming=naming,
        io_config=io_config,
        numeric_tasks=list(numeric_tasks),
        task_short=dict(task_short),
        task_label=dict(task_label),
    )


def discover_family_intervals(runtime: FamilyRuntime) -> List[int]:
    prefix = f"{runtime.naming.forecast_table_tag}_"
    intervals: set[int] = set()
    if not runtime.parquet_root.exists():
        return []
    for child in runtime.parquet_root.iterdir():
        if not child.is_dir() or not child.name.startswith(prefix):
            continue
        suffix = child.name[len(prefix) :]
        if suffix.isdigit():
            intervals.add(int(suffix))
    return sorted(intervals)


def discover_family_assets(runtime: FamilyRuntime, interval_min: int) -> List[str]:
    assets: set[str] = set()
    for store in ("forecast", "eval"):
        table_dir = runtime.parquet_root / module_table(runtime.io_config, store=store, interval=int(interval_min))
        if not table_dir.exists():
            continue
        for child in table_dir.glob("asset=*"):
            if child.is_dir() and child.name.startswith("asset="):
                asset = child.name.split("=", 1)[1]
                if asset:
                    assets.add(asset)
    return sorted(assets)


def _latest_asset_month_files(runtime: FamilyRuntime, *, interval_min: int, asset: str, store: str, max_files: int = 4) -> List[Path]:
    base = runtime.parquet_root / module_table(runtime.io_config, store=store, interval=int(interval_min)) / f"asset={asset}"
    if not base.exists():
        return []
    files = sorted(base.glob("year=*/month=*/*.parquet"), key=_month_sort_key, reverse=True)
    return [Path(p) for p in files[: max(1, int(max_files))]]


def enumerate_present_units_for_family(
    runtime: FamilyRuntime,
    *,
    intervals: Optional[Sequence[int]] = None,
    assets_filter: Optional[set[str]] = None,
    tasks_filter: Optional[set[str]] = None,
    horizons_filter: Optional[set[int]] = None,
) -> List[UnitDescriptor]:
    units: List[UnitDescriptor] = []
    use_intervals = [int(v) for v in (intervals or discover_family_intervals(runtime))]
    for interval_min in use_intervals:
        for asset in discover_family_assets(runtime, interval_min):
            if assets_filter and str(asset) not in assets_filter:
                continue
            forecast_cols: set[str] = set()
            eval_cols: set[str] = set()
            for path in _latest_asset_month_files(runtime, interval_min=interval_min, asset=asset, store="forecast"):
                try:
                    forecast_cols.update(_parquet_columns(str(path.resolve())))
                except Exception:
                    continue
            for path in _latest_asset_month_files(runtime, interval_min=interval_min, asset=asset, store="eval"):
                try:
                    eval_cols.update(_parquet_columns(str(path.resolve())))
                except Exception:
                    continue
            if not forecast_cols or not eval_cols:
                continue
            for task in runtime.numeric_tasks:
                if tasks_filter and str(task) not in tasks_filter:
                    continue
                tshort = runtime.task_short[str(task)]
                label = runtime.task_label[str(task)]
                pattern = re.compile(rf"^{re.escape(runtime.naming.prediction_prefix)}_pred_mean_{re.escape(tshort)}_(\d+)m$")
                horizons: set[int] = set()
                for col in forecast_cols:
                    match = pattern.match(str(col))
                    if match:
                        horizons.add(int(match.group(1)))
                for horizon_minutes in sorted(horizons):
                    if horizons_filter and int(horizon_minutes) not in horizons_filter:
                        continue
                    if f"{label}_{int(horizon_minutes)}m" not in eval_cols:
                        continue
                    units.append(UnitDescriptor(asset=str(asset), interval_min=int(interval_min), task=str(task), horizon_minutes=int(horizon_minutes)))
    return sorted(units, key=lambda item: (item.interval_min, item.horizon_minutes, item.task, item.asset))


def _load_asset_interval_frames(
    runtime: FamilyRuntime,
    *,
    asset: str,
    interval_min: int,
    analysis_start_ts: int,
    analysis_end_ts: int,
    units: Sequence[UnitDescriptor],
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[Tuple[str, int], str], Dict[Tuple[str, int], str]]:
    forecast_cols_map = {
        (str(unit.task), int(unit.horizon_minutes)): expected_forecast_columns(runtime.io_config, task=unit.task, horizon_minutes=unit.horizon_minutes)[0]
        for unit in units
    }
    eval_cols_map = {
        (str(unit.task), int(unit.horizon_minutes)): expected_eval_columns(runtime.io_config, task=unit.task, horizon_minutes=unit.horizon_minutes)[0]
        for unit in units
    }
    forecast_columns = ["ts", "asset"] + sorted(set(forecast_cols_map.values()))
    eval_columns = ["ts", "asset"] + sorted(set(eval_cols_map.values()))
    forecast = read_monthly_filtered(
        runtime.io_config,
        base_dir=runtime.parquet_root,
        table_dir=module_table(runtime.io_config, store="forecast", interval=int(interval_min)),
        start_ts=int(analysis_start_ts),
        end_ts=int(analysis_end_ts),
        asset=str(asset),
        columns=forecast_columns,
    )
    eval_df = read_monthly_filtered(
        runtime.io_config,
        base_dir=runtime.parquet_root,
        table_dir=module_table(runtime.io_config, store="eval", interval=int(interval_min)),
        start_ts=int(analysis_start_ts),
        end_ts=int(analysis_end_ts),
        asset=str(asset),
        columns=eval_columns,
    )
    return forecast, eval_df, forecast_cols_map, eval_cols_map


def _aligned_rows_from_shared_frames(
    runtime: FamilyRuntime,
    *,
    forecast: pd.DataFrame,
    eval_df: pd.DataFrame,
    asset: str,
    interval_min: int,
    task: str,
    horizon_minutes: int,
    analysis_start_ts: int,
    analysis_end_ts: int,
    forecast_col: str,
    eval_col: str,
) -> Tuple[pd.DataFrame, Optional[int]]:
    empty = ["model_family", "asset", "interval_min", "task", "horizon_minutes", "ts", "pred_value", "actual_value", "abs_error", "sq_error"]
    if forecast.empty or eval_df.empty or forecast_col not in forecast.columns or eval_col not in eval_df.columns:
        return pd.DataFrame(columns=empty), None
    sub_forecast = forecast.loc[:, ["ts", "asset", forecast_col]].copy()
    sub_eval = eval_df.loc[:, ["ts", "asset", eval_col]].copy()
    sub_forecast["ts"] = pd.to_numeric(sub_forecast["ts"], errors="coerce")
    sub_eval["ts"] = pd.to_numeric(sub_eval["ts"], errors="coerce")
    sub_forecast["pred_value"] = pd.to_numeric(sub_forecast[forecast_col], errors="coerce")
    sub_eval["actual_value"] = pd.to_numeric(sub_eval[eval_col], errors="coerce")
    out = sub_forecast.merge(sub_eval, on=["asset", "ts"], how="inner")
    out = out.dropna(subset=["ts", "pred_value", "actual_value"]).copy()
    if out.empty:
        return pd.DataFrame(columns=empty), None
    out["ts"] = out["ts"].astype("int64")
    latest_realized_ts = int(out["ts"].max())
    effective_start_ts = max(0, int(analysis_start_ts))
    effective_end_ts = min(int(analysis_end_ts), int(latest_realized_ts))
    out = out[(out["ts"] >= int(effective_start_ts)) & (out["ts"] <= int(effective_end_ts))].copy()
    if out.empty:
        return pd.DataFrame(columns=empty), int(latest_realized_ts)
    out["model_family"] = str(runtime.model_key)
    out["interval_min"] = int(interval_min)
    out["task"] = str(task)
    out["horizon_minutes"] = int(horizon_minutes)
    out["abs_error"] = (out["pred_value"] - out["actual_value"]).abs().astype(float)
    out["sq_error"] = ((out["pred_value"] - out["actual_value"]) ** 2.0).astype(float)
    return out[empty].sort_values("ts").reset_index(drop=True), int(latest_realized_ts)


def _process_asset_interval_group(
    runtime: FamilyRuntime,
    *,
    asset: str,
    interval_min: int,
    units: Sequence[UnitDescriptor],
    lookback_days: int,
    trailing_windows_days: Sequence[int],
    chunk_days: int,
    floor_frac: float,
) -> Dict[str, Any]:
    step_seconds = int(interval_min) * 60
    max_probe_days = max(int(lookback_days), max((int(v) for v in trailing_windows_days if int(v) > 0), default=0), int(chunk_days)) + 45
    probe_end_ts = int(_utc_now().timestamp())
    probe_start_ts = max(0, int(probe_end_ts - (int(max_probe_days) * 86400)))
    forecast_probe, eval_probe, forecast_cols_map, eval_cols_map = _load_asset_interval_frames(
        runtime,
        asset=str(asset),
        interval_min=int(interval_min),
        analysis_start_ts=int(probe_start_ts),
        analysis_end_ts=int(probe_end_ts),
        units=units,
    )
    candidate_latest: List[int] = []
    for unit in units:
        key = (str(unit.task), int(unit.horizon_minutes))
        forecast_col = forecast_cols_map[key]
        eval_col = eval_cols_map[key]
        if forecast_probe.empty or eval_probe.empty or forecast_col not in forecast_probe.columns or eval_col not in eval_probe.columns:
            continue
        merged = forecast_probe.loc[:, ["ts", "asset", forecast_col]].merge(eval_probe.loc[:, ["ts", "asset", eval_col]], on=["asset", "ts"], how="inner")
        merged["ts"] = pd.to_numeric(merged["ts"], errors="coerce")
        merged[forecast_col] = pd.to_numeric(merged[forecast_col], errors="coerce")
        merged[eval_col] = pd.to_numeric(merged[eval_col], errors="coerce")
        merged = merged.dropna(subset=["ts", forecast_col, eval_col])
        if not merged.empty:
            candidate_latest.append(int(merged["ts"].astype("int64").max()))
    if not candidate_latest:
        return {"trailing": [], "chunk": [], "rows": [], "processed_units": []}
    global_latest_realized_ts = int(max(candidate_latest))
    analysis_start_ts = max(0, int(global_latest_realized_ts - (int(lookback_days) * 86400) + int(step_seconds)))
    forecast, eval_df, forecast_cols_map, eval_cols_map = _load_asset_interval_frames(
        runtime,
        asset=str(asset),
        interval_min=int(interval_min),
        analysis_start_ts=int(analysis_start_ts),
        analysis_end_ts=int(global_latest_realized_ts),
        units=units,
    )
    trailing_frames: List[pd.DataFrame] = []
    chunk_frames: List[pd.DataFrame] = []
    row_frames: List[pd.DataFrame] = []
    processed_units: List[Dict[str, Any]] = []
    for unit in units:
        key = (str(unit.task), int(unit.horizon_minutes))
        rows_df, latest_realized_ts = _aligned_rows_from_shared_frames(
            runtime,
            forecast=forecast,
            eval_df=eval_df,
            asset=str(asset),
            interval_min=int(interval_min),
            task=str(unit.task),
            horizon_minutes=int(unit.horizon_minutes),
            analysis_start_ts=int(analysis_start_ts),
            analysis_end_ts=int(global_latest_realized_ts),
            forecast_col=forecast_cols_map[key],
            eval_col=eval_cols_map[key],
        )
        if rows_df.empty or latest_realized_ts is None:
            continue
        epsilon_task = task_epsilon_default(unit.task)
        trailing_df, trailing_rows = build_trailing_summary(
            rows_df,
            trailing_windows_days=trailing_windows_days,
            interval_min=int(interval_min),
            epsilon_task=float(epsilon_task),
            floor_frac=float(floor_frac),
        )
        chunk_df, chunk_rows = build_chunk_summary(
            rows_df,
            lookback_days=int(lookback_days),
            chunk_days=int(chunk_days),
            interval_min=int(interval_min),
            epsilon_task=float(epsilon_task),
            floor_frac=float(floor_frac),
        )
        if not trailing_df.empty:
            trailing_frames.append(trailing_df)
        if not chunk_df.empty:
            chunk_frames.append(chunk_df)
        if not trailing_rows.empty:
            row_frames.append(trailing_rows)
        if not chunk_rows.empty:
            row_frames.append(chunk_rows)
        processed_units.append(
            {
                "model_family": str(runtime.model_key),
                "asset": str(asset),
                "interval_min": int(interval_min),
                "task": str(unit.task),
                "horizon_minutes": int(unit.horizon_minutes),
                "latest_realized_ts": int(latest_realized_ts),
                "analysis_start_ts": int(max(0, int(latest_realized_ts - (int(lookback_days) * 86400) + int(step_seconds)))),
                "analysis_end_ts": int(latest_realized_ts),
                "aligned_rows": int(len(rows_df)),
                "epsilon_task": float(epsilon_task),
            }
        )
    return {
        "trailing": trailing_frames,
        "chunk": chunk_frames,
        "rows": row_frames,
        "processed_units": processed_units,
    }


def latest_populated_ts_for_unit(runtime: FamilyRuntime, *, asset: str, interval_min: int, task: str, horizon_minutes: int, store: str) -> Optional[int]:
    expected_cols = expected_eval_columns(runtime.io_config, task=task, horizon_minutes=horizon_minutes) if str(store) == "eval" else expected_forecast_columns(runtime.io_config, task=task, horizon_minutes=horizon_minutes)
    for path in _latest_asset_month_files(runtime, interval_min=interval_min, asset=asset, store=store, max_files=6):
        try:
            df = validated_module_month_parquet(
                runtime.io_config,
                path,
                asset=asset,
                store=str(store),
                interval=int(interval_min),
                task=str(task),
                horizon_minutes=int(horizon_minutes),
                expected_cols=expected_cols,
            )
        except Exception:
            continue
        if df is None or df.empty:
            continue
        mask = pd.Series(True, index=df.index)
        for col in expected_cols:
            mask = mask & pd.to_numeric(df[col], errors="coerce").notna()
        if not mask.any():
            continue
        ts = pd.to_numeric(df.loc[mask, "ts"], errors="coerce").dropna().astype("int64")
        if not ts.empty:
            return int(ts.max())
    return None


def load_aligned_unit_rows(
    runtime: FamilyRuntime,
    *,
    asset: str,
    interval_min: int,
    task: str,
    horizon_minutes: int,
    analysis_start_ts: int,
    analysis_end_ts: int,
) -> pd.DataFrame:
    forecast_col = expected_forecast_columns(runtime.io_config, task=task, horizon_minutes=horizon_minutes)[0]
    eval_col = expected_eval_columns(runtime.io_config, task=task, horizon_minutes=horizon_minutes)[0]
    forecast = read_monthly_filtered(
        runtime.io_config,
        base_dir=runtime.parquet_root,
        table_dir=module_table(runtime.io_config, store="forecast", interval=int(interval_min)),
        start_ts=int(analysis_start_ts),
        end_ts=int(analysis_end_ts),
        asset=str(asset),
        columns=["ts", "asset", forecast_col],
    )
    eval_df = read_monthly_filtered(
        runtime.io_config,
        base_dir=runtime.parquet_root,
        table_dir=module_table(runtime.io_config, store="eval", interval=int(interval_min)),
        start_ts=int(analysis_start_ts),
        end_ts=int(analysis_end_ts),
        asset=str(asset),
        columns=["ts", "asset", eval_col],
    )
    empty = ["model_family", "asset", "interval_min", "task", "horizon_minutes", "ts", "pred_value", "actual_value", "abs_error", "sq_error"]
    if forecast.empty or eval_df.empty:
        return pd.DataFrame(columns=empty)
    out = forecast.merge(eval_df, on=["asset", "ts"], how="inner")
    if out.empty:
        return pd.DataFrame(columns=empty)
    out["ts"] = pd.to_numeric(out["ts"], errors="coerce")
    out["pred_value"] = pd.to_numeric(out[forecast_col], errors="coerce")
    out["actual_value"] = pd.to_numeric(out[eval_col], errors="coerce")
    out = out.dropna(subset=["ts", "pred_value", "actual_value"]).copy()
    if out.empty:
        return pd.DataFrame(columns=empty)
    out["ts"] = out["ts"].astype("int64")
    out["model_family"] = str(runtime.model_key)
    out["interval_min"] = int(interval_min)
    out["task"] = str(task)
    out["horizon_minutes"] = int(horizon_minutes)
    out["abs_error"] = (out["pred_value"] - out["actual_value"]).abs().astype(float)
    out["sq_error"] = ((out["pred_value"] - out["actual_value"]) ** 2.0).astype(float)
    return out[empty].sort_values("ts").reset_index(drop=True)


def _summarize_slice(
    rows: pd.DataFrame,
    *,
    summary_family: str,
    epsilon_task: float,
    floor_frac: float,
    latest_realized_ts: int,
    window_days: Optional[int] = None,
    chunk_start_ts: Optional[int] = None,
    chunk_end_ts: Optional[int] = None,
    chunk_index_recent: Optional[int] = None,
) -> Tuple[Optional[Dict[str, Any]], pd.DataFrame]:
    if rows is None or rows.empty:
        return None, pd.DataFrame(columns=ROW_OUTPUT_COLUMNS)
    median_abs_actual = float(pd.to_numeric(rows["actual_value"], errors="coerce").abs().median())
    scale_floor = float(max(float(epsilon_task), float(floor_frac) * float(median_abs_actual)))
    detail = rows.copy()
    detail["effective_scale"] = np.maximum(detail["actual_value"].abs().to_numpy(dtype=float), float(scale_floor))
    detail["relative_error_like"] = detail["abs_error"].to_numpy(dtype=float) / detail["effective_scale"].to_numpy(dtype=float)
    detail["hit_10"] = detail["relative_error_like"] <= 0.10
    detail["hit_20"] = detail["relative_error_like"] <= 0.20
    detail["hit_30"] = detail["relative_error_like"] <= 0.30
    summary = {
        "model_family": str(detail["model_family"].iloc[0]),
        "asset": str(detail["asset"].iloc[0]),
        "interval_min": int(detail["interval_min"].iloc[0]),
        "task": str(detail["task"].iloc[0]),
        "horizon_minutes": int(detail["horizon_minutes"].iloc[0]),
        "window_days": (int(window_days) if window_days is not None else None),
        "window_start_ts": int(detail["ts"].min()),
        "window_end_ts": int(detail["ts"].max()),
        "latest_realized_ts": int(latest_realized_ts),
        "n_predictions": int(len(detail)),
        "hit_count_10": int(detail["hit_10"].sum()),
        "hit_rate_10": float(detail["hit_10"].mean()),
        "hit_count_20": int(detail["hit_20"].sum()),
        "hit_rate_20": float(detail["hit_20"].mean()),
        "hit_count_30": int(detail["hit_30"].sum()),
        "hit_rate_30": float(detail["hit_30"].mean()),
        "mae": float(detail["abs_error"].mean()),
        "rmse": float(math.sqrt(float(detail["sq_error"].mean()))),
        "median_abs_actual": float(median_abs_actual),
        "scale_floor": float(scale_floor),
        "epsilon_task": float(epsilon_task),
        "floor_frac": float(floor_frac),
        "chunk_start_ts": (int(chunk_start_ts) if chunk_start_ts is not None else None),
        "chunk_end_ts": (int(chunk_end_ts) if chunk_end_ts is not None else None),
        "chunk_index_recent": (int(chunk_index_recent) if chunk_index_recent is not None else None),
    }
    detail["summary_family"] = str(summary_family)
    detail["window_days"] = (int(window_days) if window_days is not None else None)
    detail["chunk_start_ts"] = (int(chunk_start_ts) if chunk_start_ts is not None else None)
    detail["chunk_end_ts"] = (int(chunk_end_ts) if chunk_end_ts is not None else None)
    detail["chunk_index_recent"] = (int(chunk_index_recent) if chunk_index_recent is not None else None)
    return summary, detail[ROW_OUTPUT_COLUMNS].reset_index(drop=True)


def build_trailing_summary(
    rows_df: pd.DataFrame,
    *,
    trailing_windows_days: Sequence[int],
    interval_min: int,
    epsilon_task: float,
    floor_frac: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if rows_df.empty:
        return pd.DataFrame(columns=TRAILING_SUMMARY_COLUMNS), pd.DataFrame(columns=ROW_OUTPUT_COLUMNS)
    latest_realized_ts = int(rows_df["ts"].max())
    step_seconds = int(interval_min) * 60
    summary_rows: List[Dict[str, Any]] = []
    detail_frames: List[pd.DataFrame] = []
    for window_days in sorted({int(v) for v in trailing_windows_days if int(v) > 0}):
        start_ts = int(latest_realized_ts - (int(window_days) * 86400) + step_seconds)
        slice_df = rows_df[rows_df["ts"].astype("int64") >= int(start_ts)].copy()
        summary, detail = _summarize_slice(
            slice_df,
            summary_family="trailing",
            epsilon_task=float(epsilon_task),
            floor_frac=float(floor_frac),
            latest_realized_ts=int(latest_realized_ts),
            window_days=int(window_days),
        )
        if summary is None:
            continue
        summary_rows.append(summary)
        detail_frames.append(detail)
    trailing_df = pd.DataFrame(summary_rows)
    if trailing_df.empty:
        trailing_df = pd.DataFrame(columns=TRAILING_SUMMARY_COLUMNS)
    else:
        trailing_df = trailing_df[TRAILING_SUMMARY_COLUMNS].sort_values(["model_family", "asset", "interval_min", "task", "horizon_minutes", "window_days"]).reset_index(drop=True)
    detail_df = pd.concat(detail_frames, ignore_index=True) if detail_frames else pd.DataFrame(columns=ROW_OUTPUT_COLUMNS)
    return trailing_df, detail_df


def build_chunk_summary(
    rows_df: pd.DataFrame,
    *,
    lookback_days: int,
    chunk_days: int,
    interval_min: int,
    epsilon_task: float,
    floor_frac: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if rows_df.empty:
        return pd.DataFrame(columns=CHUNK_SUMMARY_COLUMNS), pd.DataFrame(columns=ROW_OUTPUT_COLUMNS)
    latest_realized_ts = int(rows_df["ts"].max())
    step_seconds = int(interval_min) * 60
    lookback_seconds = int(lookback_days) * 86400
    chunk_seconds = int(chunk_days) * 86400
    global_start_ts = int(latest_realized_ts - lookback_seconds + step_seconds)
    summary_rows: List[Dict[str, Any]] = []
    detail_frames: List[pd.DataFrame] = []
    chunk_index_recent = 0
    chunk_end_ts = int(latest_realized_ts)
    min_ts = int(rows_df["ts"].min())
    while chunk_end_ts >= int(global_start_ts) and chunk_end_ts >= int(min_ts):
        raw_start = int(chunk_end_ts - chunk_seconds + step_seconds)
        chunk_start_ts = max(int(global_start_ts), int(raw_start))
        slice_df = rows_df[
            (rows_df["ts"].astype("int64") >= int(chunk_start_ts))
            & (rows_df["ts"].astype("int64") <= int(chunk_end_ts))
        ].copy()
        summary, detail = _summarize_slice(
            slice_df,
            summary_family="chunk",
            epsilon_task=float(epsilon_task),
            floor_frac=float(floor_frac),
            latest_realized_ts=int(latest_realized_ts),
            chunk_start_ts=int(chunk_start_ts),
            chunk_end_ts=int(chunk_end_ts),
            chunk_index_recent=int(chunk_index_recent),
        )
        if summary is not None:
            summary_rows.append(summary)
            detail_frames.append(detail)
        chunk_index_recent += 1
        chunk_end_ts = int(chunk_start_ts - step_seconds)
    chunk_df = pd.DataFrame(summary_rows)
    if chunk_df.empty:
        chunk_df = pd.DataFrame(columns=CHUNK_SUMMARY_COLUMNS)
    else:
        chunk_df = chunk_df[CHUNK_SUMMARY_COLUMNS].sort_values(["model_family", "asset", "interval_min", "task", "horizon_minutes", "chunk_index_recent"]).reset_index(drop=True)
    detail_df = pd.concat(detail_frames, ignore_index=True) if detail_frames else pd.DataFrame(columns=ROW_OUTPUT_COLUMNS)
    return chunk_df, detail_df


def build_ranked_view(
    trailing_summary_df: pd.DataFrame,
    *,
    rank_window_days: int,
    min_predictions_ranked: Optional[int],
) -> pd.DataFrame:
    if trailing_summary_df.empty:
        return pd.DataFrame(columns=TRAILING_SUMMARY_COLUMNS)
    ranked = trailing_summary_df[trailing_summary_df["window_days"].astype("Int64") == int(rank_window_days)].copy()
    if min_predictions_ranked is not None:
        ranked = ranked[ranked["n_predictions"].astype(int) >= int(min_predictions_ranked)].copy()
    return ranked.sort_values(
        ["hit_rate_20", "hit_rate_10", "mae", "rmse"],
        ascending=[False, False, True, True],
        kind="stable",
    ).reset_index(drop=True)


def write_accuracy_artifacts(
    output_dir: Path,
    *,
    trailing_df: pd.DataFrame,
    chunk_df: pd.DataFrame,
    ranked_df: pd.DataFrame,
    manifest: Dict[str, Any],
) -> Dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    trailing_path = output_dir / "recent_accuracy_trailing_summary.csv"
    chunk_path = output_dir / "recent_accuracy_chunk_summary.csv"
    ranked_path = output_dir / "recent_accuracy_ranked.csv"
    rows_path = output_dir / "recent_accuracy_rows.parquet"
    manifest_path = output_dir / "run_manifest.json"
    trailing_df.to_csv(trailing_path, index=False)
    chunk_df.to_csv(chunk_path, index=False)
    ranked_df.to_csv(ranked_path, index=False)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return {
        "recent_accuracy_trailing_summary.csv": str(trailing_path),
        "recent_accuracy_chunk_summary.csv": str(chunk_path),
        "recent_accuracy_ranked.csv": str(ranked_path),
        "recent_accuracy_rows.parquet": str(rows_path),
        "run_manifest.json": str(manifest_path),
    }


def _append_rows_parquet(
    writer: Optional[pq.ParquetWriter],
    rows_path: Path,
    rows_frames: Sequence[pd.DataFrame],
) -> tuple[Optional[pq.ParquetWriter], int]:
    non_empty = [frame for frame in rows_frames if frame is not None and not frame.empty]
    if not non_empty:
        return writer, 0
    rows_df = pd.concat(non_empty, ignore_index=True)
    if rows_df.empty:
        return writer, 0
    rows_df = rows_df.reindex(columns=ROW_OUTPUT_COLUMNS)
    float64_cols = [
        "pred_value",
        "actual_value",
        "abs_error",
        "sq_error",
        "effective_scale",
        "relative_error_like",
    ]
    int64_cols = [
        "window_days",
        "chunk_start_ts",
        "chunk_end_ts",
        "chunk_index_recent",
        "interval_min",
        "horizon_minutes",
        "ts",
    ]
    bool_cols = ["hit_10", "hit_20", "hit_30"]
    string_cols = ["summary_family", "model_family", "asset", "task"]
    for col in float64_cols:
        rows_df[col] = pd.to_numeric(rows_df[col], errors="coerce").astype("float64")
    for col in int64_cols:
        rows_df[col] = pd.to_numeric(rows_df[col], errors="coerce").astype("Int64")
    for col in bool_cols:
        rows_df[col] = rows_df[col].astype("boolean")
    for col in string_cols:
        rows_df[col] = rows_df[col].astype("string")
    table = pa.Table.from_pandas(rows_df, preserve_index=False)
    if writer is None:
        rows_path.parent.mkdir(parents=True, exist_ok=True)
        writer = pq.ParquetWriter(str(rows_path), table.schema)
    writer.write_table(table)
    return writer, int(len(rows_df))


def run_recent_realized_accuracy(
    *,
    model_keys: Sequence[str] = DEFAULT_MODEL_KEYS,
    intervals: Optional[Sequence[int]] = None,
    assets: Optional[Sequence[str]] = None,
    tasks: Optional[Sequence[str]] = None,
    horizons: Optional[Sequence[int]] = None,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    trailing_windows_days: Sequence[int] = DEFAULT_TRAILING_WINDOWS_DAYS,
    chunk_days: int = DEFAULT_CHUNK_DAYS,
    floor_frac: float = DEFAULT_FLOOR_FRAC,
    rank_window_days: int = DEFAULT_RANK_WINDOW_DAYS,
    min_predictions_ranked: Optional[int] = DEFAULT_RANK_MIN_PREDICTIONS,
    workers: Optional[int] = None,
    output_dir: Optional[Path] = None,
    project_root: Optional[Path] = None,
    log_fn: Optional[Callable[[str], None]] = None,
    include_rows_in_result: bool = False,
) -> Dict[str, Any]:
    project_root = Path(project_root) if project_root else _project_root()
    log_fn = log_fn or _default_log_fn
    assets_filter = {str(v) for v in assets} if assets else None
    tasks_filter = {str(v) for v in tasks} if tasks else None
    horizons_filter = {int(v) for v in horizons} if horizons else None
    interval_filter = [int(v) for v in intervals] if intervals else None
    resolved_workers = max(1, int(workers if workers is not None else DEFAULT_UNIT_WORKERS))
    log_resolved_runtime("recent_realized_accuracy", resolved={"unit_workers": int(resolved_workers), "writer_workers": 1})
    if output_dir is None:
        output_dir = project_root / "logs" / "diagnostics" / "tabular_recent_realized_accuracy" / f"run={_utc_now_run_token()}"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = output_dir / "recent_accuracy_rows.parquet"
    rows_writer: Optional[pq.ParquetWriter] = None
    rows_written = 0
    runtimes = [build_family_runtime(model_key, project_root=project_root, log_fn=log_fn) for model_key in model_keys]
    all_trailing: List[pd.DataFrame] = []
    all_chunk: List[pd.DataFrame] = []
    processed_units: List[Dict[str, Any]] = []
    for runtime in runtimes:
        units = enumerate_present_units_for_family(
            runtime,
            intervals=interval_filter,
            assets_filter=assets_filter,
            tasks_filter=tasks_filter,
            horizons_filter=horizons_filter,
        )
        log_fn(f"{runtime.naming.log_prefix}[recent_accuracy] discovered_units={len(units)} model_family={runtime.model_key}")
        grouped_units: Dict[Tuple[str, int], List[UnitDescriptor]] = {}
        for unit in units:
            grouped_units.setdefault((str(unit.asset), int(unit.interval_min)), []).append(unit)
        if not grouped_units:
            continue
        log_fn(
            f"{runtime.naming.log_prefix}[recent_accuracy] groups={len(grouped_units)} "
            f"model_family={runtime.model_key} unit_workers={int(resolved_workers)}"
        )
        if int(resolved_workers) <= 1 or len(grouped_units) <= 1:
            for (asset, interval_min), group_units in sorted(grouped_units.items(), key=lambda item: (item[0][1], item[0][0])):
                result = _process_asset_interval_group(
                    runtime,
                    asset=asset,
                    interval_min=interval_min,
                    units=group_units,
                    lookback_days=int(lookback_days),
                    trailing_windows_days=trailing_windows_days,
                    chunk_days=int(chunk_days),
                    floor_frac=float(floor_frac),
                )
                all_trailing.extend(result.get("trailing", []))
                all_chunk.extend(result.get("chunk", []))
                rows_writer, wrote_rows = _append_rows_parquet(rows_writer, rows_path, result.get("rows", []))
                rows_written += int(wrote_rows)
                processed_units.extend(result.get("processed_units", []))
        else:
            with ThreadPoolExecutor(max_workers=min(int(resolved_workers), len(grouped_units))) as executor:
                future_map = {
                    executor.submit(
                        _process_asset_interval_group,
                        runtime,
                        asset=asset,
                        interval_min=interval_min,
                        units=group_units,
                        lookback_days=int(lookback_days),
                        trailing_windows_days=trailing_windows_days,
                        chunk_days=int(chunk_days),
                        floor_frac=float(floor_frac),
                    ): (asset, interval_min)
                    for (asset, interval_min), group_units in grouped_units.items()
                }
                ordered_futures = sorted(
                    ((key, future) for future, key in future_map.items()),
                    key=lambda item: (item[0][1], item[0][0]),
                )
                for (_, _), future in ordered_futures:
                    result = future.result()
                    all_trailing.extend(result.get("trailing", []))
                    all_chunk.extend(result.get("chunk", []))
                    rows_writer, wrote_rows = _append_rows_parquet(rows_writer, rows_path, result.get("rows", []))
                    rows_written += int(wrote_rows)
                    processed_units.extend(result.get("processed_units", []))
    if rows_writer is not None:
        rows_writer.close()
    trailing_summary_df = pd.concat(all_trailing, ignore_index=True) if all_trailing else pd.DataFrame(columns=TRAILING_SUMMARY_COLUMNS)
    chunk_summary_df = pd.concat(all_chunk, ignore_index=True) if all_chunk else pd.DataFrame(columns=CHUNK_SUMMARY_COLUMNS)
    rows_output_df = pd.read_parquet(rows_path) if include_rows_in_result and rows_path.exists() else pd.DataFrame(columns=ROW_OUTPUT_COLUMNS)
    processed_units = sorted(
        processed_units,
        key=lambda row: (
            str(row.get("model_family")),
            int(row.get("interval_min", 0) or 0),
            int(row.get("horizon_minutes", 0) or 0),
            str(row.get("task")),
            str(row.get("asset")),
        ),
    )
    ranked_df = build_ranked_view(
        trailing_summary_df,
        rank_window_days=int(rank_window_days),
        min_predictions_ranked=min_predictions_ranked,
    )
    manifest = {
        "created_at_utc": _iso_now(),
        "project_root": str(project_root),
        "settings": {
            "model_keys": [str(v) for v in model_keys],
            "intervals": [int(v) for v in interval_filter] if interval_filter else [],
            "assets": sorted(assets_filter) if assets_filter else [],
            "tasks": sorted(tasks_filter) if tasks_filter else [],
            "horizons": sorted(int(v) for v in horizons_filter) if horizons_filter else [],
            "lookback_days": int(lookback_days),
            "trailing_windows_days": [int(v) for v in trailing_windows_days],
            "chunk_days": int(chunk_days),
            "floor_frac": float(floor_frac),
            "rank_window_days": int(rank_window_days),
            "min_predictions_ranked": (int(min_predictions_ranked) if min_predictions_ranked is not None else None),
            "workers": int(resolved_workers),
            "epsilon_by_task": {str(k): float(v) for k, v in DEFAULT_EPSILON_BY_TASK.items()},
        },
        "counts": {
            "processed_units": int(len(processed_units)),
            "trailing_rows": int(len(trailing_summary_df)),
            "chunk_rows": int(len(chunk_summary_df)),
            "row_audit_rows": int(rows_written),
            "ranked_rows": int(len(ranked_df)),
        },
        "processed_units": processed_units,
    }
    artifact_paths = write_accuracy_artifacts(
        Path(output_dir),
        trailing_df=trailing_summary_df,
        chunk_df=chunk_summary_df,
        ranked_df=ranked_df,
        manifest=manifest,
    )
    manifest["artifacts"] = artifact_paths
    (Path(output_dir) / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return {
        "output_dir": str(output_dir),
        "trailing_summary": trailing_summary_df,
        "chunk_summary": chunk_summary_df,
        "ranked": ranked_df,
        "rows": rows_output_df,
        "manifest": manifest,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize recent realized accuracy from tabular forecast/eval artifacts.")
    parser.add_argument("--models", type=str, default="xgboost,lightgbm", help="Comma-delimited model families.")
    parser.add_argument("--intervals", type=str, default="", help="Comma-delimited interval minutes.")
    parser.add_argument("--assets", type=str, default="", help="Comma-delimited assets.")
    parser.add_argument("--tasks", type=str, default="", help="Comma-delimited tasks.")
    parser.add_argument("--horizons", type=str, default="", help="Comma-delimited horizon minutes.")
    parser.add_argument("--lookback-days", type=int, default=DEFAULT_LOOKBACK_DAYS)
    parser.add_argument("--trailing-window-days", type=str, default="14,30,60,90,180")
    parser.add_argument("--chunk-days", type=int, default=DEFAULT_CHUNK_DAYS)
    parser.add_argument("--floor-frac", type=float, default=DEFAULT_FLOOR_FRAC)
    parser.add_argument("--rank-window-days", type=int, default=DEFAULT_RANK_WINDOW_DAYS)
    parser.add_argument("--rank-min-predictions", type=int, default=DEFAULT_RANK_MIN_PREDICTIONS)
    parser.add_argument("--workers", type=int, default=DEFAULT_UNIT_WORKERS)
    parser.add_argument("--output-dir", type=str, default="")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    run_recent_realized_accuracy(
        model_keys=_parse_csv_strs(args.models),
        intervals=_parse_csv_ints(args.intervals) if str(args.intervals).strip() else None,
        assets=_parse_csv_strs(args.assets) if str(args.assets).strip() else None,
        tasks=_parse_csv_strs(args.tasks) if str(args.tasks).strip() else None,
        horizons=_parse_csv_ints(args.horizons) if str(args.horizons).strip() else None,
        lookback_days=int(args.lookback_days),
        trailing_windows_days=_parse_csv_ints(args.trailing_window_days),
        chunk_days=int(args.chunk_days),
        floor_frac=float(args.floor_frac),
        rank_window_days=int(args.rank_window_days),
        min_predictions_ranked=int(args.rank_min_predictions) if args.rank_min_predictions is not None else None,
        workers=int(args.workers),
        output_dir=(Path(args.output_dir) if str(args.output_dir).strip() else None),
    )


if __name__ == "__main__":
    main()

