from __future__ import annotations

import json
import os
import pickle
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.forecasting.common.io_atomic import atomic_replace, sibling_temp_path
from src.forecasting.common.pipeline_parquet_utils import (
    PipelineValidationError,
    partition_max_ts,
    validate_strict_timegrid,
    validate_no_nan_columns,
)
from src.forecasting.common.sandbox_paths import assert_write_allowed


@dataclass(frozen=True)
class NumericForecastNamingConfig:
    module_slug: str
    forecast_table_tag: str
    eval_table_tag: str
    prediction_prefix: str
    task_short: Dict[str, str]
    task_label: Dict[str, str]
    log_prefix: str


@dataclass(frozen=True)
class NumericForecastIOConfig:
    naming: NumericForecastNamingConfig
    parquet_root: Path
    staging_root: Path
    state_root: Path
    scalar_root: Path
    ohlc_root: Path
    parquet_compression: str
    parquet_row_group: int
    log_fn: Callable[[str], None]
    read_ohlcvt_fn: Callable[..., pd.DataFrame]
    list_assets_from_ohlcvt_fn: Callable[[int], Sequence[str]]
    first_ohlcvt_ts_fn: Callable[..., Optional[int]]
    ohlcvt_max_ts_fn: Callable[..., Optional[int]]
    feature_max_ts_fn: Callable[..., Optional[int]]


_STAGE_WRITE_QUEUE: Any = None


def merge_numeric_dicts(*sources: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for src in sources:
        for key, value in src.items():
            if isinstance(value, bool):
                out[str(key)] = int(value)
            elif isinstance(value, (int, float)):
                out[str(key)] = float(out.get(str(key), 0.0) + float(value))
    return out


def ts_monotonic_unique(df: pd.DataFrame) -> bool:
    if df.empty or "ts" not in df.columns:
        return True
    ts = pd.to_numeric(df["ts"], errors="coerce").to_numpy(dtype=np.int64, copy=False)
    if len(ts) <= 1:
        return True
    return bool(np.all(ts[1:] > ts[:-1]))


def next_month(y: int, m: int) -> Tuple[int, int]:
    return (y + 1, 1) if m == 12 else (y, m + 1)


def month_start_ts(year: int, month: int) -> int:
    return int(datetime(year, month, 1, 0, 0, 0, tzinfo=timezone.utc).timestamp())


def iter_months_between(start_ts: int, end_ts: int) -> Iterable[Tuple[int, int]]:
    if end_ts is None or start_ts is None or end_ts < start_ts:
        return
    dt = datetime.fromtimestamp(start_ts, tz=timezone.utc)
    y, m = dt.year, dt.month
    while True:
        yield (y, m)
        y, m = next_month(y, m)
        if month_start_ts(y, m) > end_ts:
            break


def model_state_path(io_config: NumericForecastIOConfig, asset: str, interval: int, horizon_minutes: int, task: str) -> Path:
    return (
        io_config.state_root
        / f"interval={int(interval)}"
        / f"horizon={int(horizon_minutes)}m"
        / f"task={str(task)}"
        / f"asset={str(asset)}.pkl"
    )


def load_model_state(io_config: NumericForecastIOConfig, asset: str, interval: int, horizon_minutes: int, task: str) -> Optional[Dict[str, Any]]:
    p = model_state_path(io_config=io_config, asset=asset, interval=interval, horizon_minutes=horizon_minutes, task=task)
    if not p.exists():
        return None
    try:
        obj = pickle.loads(p.read_bytes())
    except Exception:
        return None
    if not isinstance(obj, dict):
        return None
    if int(obj.get("version", 0)) != 1:
        return None
    return obj


def save_model_state(io_config: NumericForecastIOConfig, asset: str, interval: int, horizon_minutes: int, task: str, state: Dict[str, Any]) -> None:
    p = model_state_path(io_config=io_config, asset=asset, interval=interval, horizon_minutes=horizon_minutes, task=task)
    assert_write_allowed(p, "numeric model state")
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(state)
    payload["version"] = 1
    payload["asset"] = str(asset)
    payload["interval"] = int(interval)
    payload["horizon_minutes"] = int(horizon_minutes)
    payload["task"] = str(task)
    tmp = sibling_temp_path(p, suffix=".pkl.tmp")
    assert_write_allowed(tmp, "numeric model state temp")
    tmp.write_bytes(pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL))
    atomic_replace(tmp, p)


def chunk_end_ts_for_month(start_exclusive_ts: int, end_inclusive_ts: int, interval_minutes: int) -> int:
    step = max(60, int(interval_minutes) * 60)
    first_ts = int(start_exclusive_ts) + step
    if first_ts > int(end_inclusive_ts):
        return int(end_inclusive_ts)
    dt = datetime.fromtimestamp(first_ts, tz=timezone.utc)
    next_y, next_m = next_month(dt.year, dt.month)
    month_end = month_start_ts(next_y, next_m) - step
    return int(min(int(end_inclusive_ts), int(month_end)))


def read_monthly_filtered(
    io_config: NumericForecastIOConfig,
    *,
    base_dir: Path,
    table_dir: str,
    start_ts: int,
    end_ts: int,
    asset: str,
    columns: Optional[Sequence[str]] = None,
    stats: Optional[Dict[str, float]] = None,
    stats_prefix: str = "",
) -> pd.DataFrame:
    def _stat_key(name: str) -> str:
        prefix = str(stats_prefix or "").strip()
        return f"{prefix}_{name}" if prefix else name

    def _add_stat(name: str, value: float) -> None:
        if stats is not None:
            key = _stat_key(name)
            stats[key] = float(stats.get(key, 0.0)) + float(value)

    def _set_stat_max(name: str, value: int) -> None:
        if stats is not None:
            key = _stat_key(name)
            stats[key] = float(max(float(stats.get(key, 0.0)), float(value)))

    def _set_stat_min_nonzero(name: str, value: int) -> None:
        if stats is None:
            return
        key = _stat_key(name)
        current = float(stats.get(key, 0.0))
        next_value = float(value)
        stats[key] = next_value if current <= 0 else float(min(current, next_value))

    def _month_files_for(year: int, month: int) -> List[Path]:
        paths: List[Path] = []
        month_dirs = [
            base_dir / table_dir / f"asset={asset}" / f"year={year}" / f"month={month:02d}",
            base_dir / table_dir / f"year={year}" / f"month={month:02d}",
        ]
        for month_dir in month_dirs:
            t_discover = time.monotonic()
            try:
                if month_dir.exists():
                    paths.extend(sorted(month_dir.glob("*.parquet")))
            finally:
                _add_stat("month_discovery_s", time.monotonic() - t_discover)
        return paths

    if str(table_dir).lower().startswith("ohlcvt_"):
        try:
            interval_min = int(str(table_dir).split("_", 1)[1])
        except Exception:
            interval_min = 0
        if interval_min > 0:
            discovered_files = 0
            for y, m in iter_months_between(start_ts, end_ts):
                discovered_files += len(_month_files_for(int(y), int(m)))
            _add_stat("month_files_discovered", float(discovered_files))
            out = io_config.read_ohlcvt_fn(
                asset=str(asset),
                interval_min=interval_min,
                start_ts=int(start_ts),
                end_ts=int(end_ts),
                columns=list(columns) if columns else None,
                root=Path(base_dir),
            )
            _add_stat("read_fn_calls", 1.0)
            _add_stat("month_files_read", float(discovered_files))
            _add_stat("rows_loaded", float(len(out)))
            _set_stat_max("columns_loaded", len(out.columns))
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
        files = _month_files_for(int(y), int(m))
        _add_stat("month_files_discovered", float(len(files)))
        for p in files:
            _add_stat("parquet_read_calls", 1.0)
            try:
                t_read = time.monotonic()
                df = pd.read_parquet(p, columns=columns)
                _add_stat("read_parquet_s", time.monotonic() - t_read)
            except Exception:
                continue
            _add_stat("month_files_read", 1.0)
            _add_stat("raw_rows_loaded", float(len(df)))
            _add_stat("raw_columns_loaded", float(len(df.columns)))
            _set_stat_max("raw_columns_loaded_max", len(df.columns))
            _set_stat_min_nonzero("raw_columns_loaded_min", len(df.columns))
            if "asset" not in df.columns or "ts" not in df.columns:
                continue
            t_filter = time.monotonic()
            ts_num = pd.to_numeric(df["ts"], errors="coerce")
            df = df[
                (df["asset"].astype(str) == str(asset))
                & ts_num.notna()
                & (ts_num.astype("int64") >= int(start_ts))
                & (ts_num.astype("int64") <= int(end_ts))
            ]
            _add_stat("filter_s", time.monotonic() - t_filter)
            if not df.empty:
                frames.append(df)
    if not frames:
        cols = list(columns) if columns else ["ts", "asset"]
        return pd.DataFrame(columns=cols)
    t_concat = time.monotonic()
    out = pd.concat(frames, ignore_index=True)
    _add_stat("concat_s", time.monotonic() - t_concat)
    t_sort = time.monotonic()
    out["ts"] = pd.to_numeric(out["ts"], errors="coerce").astype("int64")
    out = out.sort_values("ts").drop_duplicates(subset=["asset", "ts"], keep="last")
    _add_stat("read_sort_dedup_s", time.monotonic() - t_sort)
    return out


def load_unit_feature_frame(io_config: NumericForecastIOConfig, asset: str, interval: int, start_ts: int, stop_ts: int) -> Tuple[pd.DataFrame, Dict[str, float]]:
    stats = {
        "load_total_s": 0.0,
        "first_ohlcvt_ts_s": 0.0,
        "load_base_grid_s": 0.0,
        "read_ohlc_s": 0.0,
        "ohlc_sort_dedup_s": 0.0,
        "ohlc_merge_ffill_s": 0.0,
        "ohlc_month_discovery_s": 0.0,
        "ohlc_month_files_discovered": 0.0,
        "ohlc_month_files_read": 0.0,
        "ohlc_read_fn_calls": 0.0,
        "ohlc_parquet_read_calls": 0.0,
        "ohlc_rows_loaded": 0.0,
        "ohlc_columns_loaded": 0.0,
        "read_scalar_s": 0.0,
        "scalar_month_discovery_s": 0.0,
        "scalar_month_files_discovered": 0.0,
        "scalar_month_files_read": 0.0,
        "scalar_parquet_read_calls": 0.0,
        "scalar_raw_rows_loaded": 0.0,
        "scalar_raw_columns_loaded": 0.0,
        "scalar_raw_columns_loaded_max": 0.0,
        "scalar_raw_columns_loaded_min": 0.0,
        "scalar_read_parquet_s": 0.0,
        "scalar_filter_s": 0.0,
        "scalar_concat_s": 0.0,
        "scalar_sort_dedup_s": 0.0,
        "scalar_read_sort_dedup_s": 0.0,
        "scalar_numeric_coerce_s": 0.0,
        "scalar_merge_ffill_s": 0.0,
        "load_finalize_sort_s": 0.0,
    }
    t_load = time.monotonic()
    t_first = time.monotonic()
    min_ts = io_config.first_ohlcvt_ts_fn(interval, asset, root=io_config.ohlc_root)
    stats["first_ohlcvt_ts_s"] = float(time.monotonic() - t_first)
    if min_ts is None:
        stats["load_total_s"] = float(time.monotonic() - t_load)
        return pd.DataFrame(), stats
    effective_start_ts = max(int(min_ts), int(start_ts))
    if int(stop_ts) < int(effective_start_ts):
        stats["load_total_s"] = float(time.monotonic() - t_load)
        return pd.DataFrame(), stats
    step = int(interval) * 60
    t_base = time.monotonic()
    base = pd.DataFrame({"ts": np.arange(int(effective_start_ts), int(stop_ts) + step, step, dtype=np.int64)})
    base["asset"] = str(asset)
    stats["load_base_grid_s"] = float(time.monotonic() - t_base)

    t_ohlc = time.monotonic()
    ohlc = read_monthly_filtered(
        io_config,
        base_dir=io_config.ohlc_root,
        table_dir=f"ohlcvt_{interval}",
        start_ts=int(effective_start_ts),
        end_ts=int(stop_ts),
        asset=asset,
        columns=["ts", "asset", "open", "high", "low", "close", "volume", "trades"],
        stats=stats,
        stats_prefix="ohlc",
    )
    stats["read_ohlc_s"] += float(time.monotonic() - t_ohlc)
    if ohlc.empty:
        stats["load_total_s"] = float(time.monotonic() - t_load)
        return pd.DataFrame(), stats
    t_ohlc_sort = time.monotonic()
    if not ts_monotonic_unique(ohlc):
        ohlc = ohlc.sort_values("ts").drop_duplicates(subset=["asset", "ts"], keep="last")
    stats["ohlc_sort_dedup_s"] = float(time.monotonic() - t_ohlc_sort)
    t_ohlc_merge = time.monotonic()
    out = base.merge(ohlc.drop(columns=["asset"]), on="ts", how="left")
    for c in ["open", "high", "low", "close", "volume", "trades"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out[["open", "high", "low", "close", "volume", "trades"]] = out[["open", "high", "low", "close", "volume", "trades"]].ffill().fillna(0.0)
    stats["ohlc_merge_ffill_s"] = float(time.monotonic() - t_ohlc_merge)

    t_scalar = time.monotonic()
    scalars = read_monthly_filtered(
        io_config,
        base_dir=io_config.scalar_root,
        table_dir=f"scalar_features_{interval}",
        start_ts=int(effective_start_ts),
        end_ts=int(stop_ts),
        asset=asset,
        columns=None,
        stats=stats,
        stats_prefix="scalar",
    )
    stats["read_scalar_s"] += float(time.monotonic() - t_scalar)
    if scalars.empty:
        scalars = out[["ts", "asset"]].copy()
        scalars["bias_feature"] = 0.0
    scalar_cols = [c for c in scalars.columns if c not in {"ts", "asset"}]
    if not scalar_cols:
        scalar_cols = ["bias_feature"]
        scalars["bias_feature"] = 0.0
    t_scalar_coerce = time.monotonic()
    for c in scalar_cols:
        scalars[c] = pd.to_numeric(scalars[c], errors="coerce")
    stats["scalar_numeric_coerce_s"] = float(time.monotonic() - t_scalar_coerce)
    t_scalar_sort = time.monotonic()
    if not ts_monotonic_unique(scalars):
        scalars = scalars.sort_values("ts").drop_duplicates(subset=["asset", "ts"], keep="last")
    stats["scalar_sort_dedup_s"] = float(time.monotonic() - t_scalar_sort)
    t_scalar_merge = time.monotonic()
    out = out.merge(scalars[["ts", "asset"] + scalar_cols], on=["ts", "asset"], how="left")
    out[scalar_cols] = out[scalar_cols].ffill().fillna(0.0)
    stats["scalar_merge_ffill_s"] = float(time.monotonic() - t_scalar_merge)
    t_finalize = time.monotonic()
    out = out.sort_values("ts").reset_index(drop=True)
    stats["load_finalize_sort_s"] = float(time.monotonic() - t_finalize)
    stats["load_total_s"] = float(time.monotonic() - t_load)
    return out, stats


def get_stop_ts(io_config: NumericForecastIOConfig, asset: str, interval: int) -> Optional[int]:
    mx = io_config.ohlcvt_max_ts_fn(interval, asset, root=io_config.ohlc_root)
    return int(mx) if mx is not None else None


def get_start_ts(io_config: NumericForecastIOConfig, asset: str, interval: int) -> Optional[int]:
    mn = io_config.first_ohlcvt_ts_fn(interval, asset, root=io_config.ohlc_root)
    return int(mn) if mn is not None else None


def resolve_assets(io_config: NumericForecastIOConfig, intervals: Sequence[int], assets_arg: str) -> List[str]:
    if assets_arg.strip():
        return sorted({a.strip() for a in assets_arg.split(",") if a.strip()})
    assets: set[str] = set()
    for k in intervals:
        assets.update(str(x) for x in io_config.list_assets_from_ohlcvt_fn(k))
        feat_base = io_config.scalar_root / f"scalar_features_{int(k)}"
        if feat_base.exists():
            for p in feat_base.glob("asset=*"):
                if p.is_dir() and p.name.startswith("asset=") and len(p.name) > len("asset="):
                    assets.add(p.name.replace("asset=", ""))
    return sorted(assets)


def module_table(io_config: NumericForecastIOConfig, store: str, interval: int) -> str:
    tag = io_config.naming.eval_table_tag if store == "eval" else io_config.naming.forecast_table_tag
    return f"{tag}_{int(interval)}"


def expected_forecast_columns(io_config: NumericForecastIOConfig, task: str, horizon_minutes: int) -> List[str]:
    tshort = io_config.naming.task_short[str(task)]
    hm = int(horizon_minutes)
    pref = io_config.naming.prediction_prefix
    return [
        f"{pref}_pred_mean_{tshort}_{hm}m",
        f"{pref}_pred_std_{tshort}_{hm}m",
        f"{pref}_pred_p10_{tshort}_{hm}m",
        f"{pref}_pred_p90_{tshort}_{hm}m",
    ]


def expected_eval_columns(io_config: NumericForecastIOConfig, task: str, horizon_minutes: int) -> List[str]:
    hm = int(horizon_minutes)
    cols = [f"{io_config.naming.task_label[str(task)]}_{hm}m"]
    if str(task) == "log_return":
        cols.append(f"future_direction_{hm}m")
    return cols


def expected_store_columns(io_config: NumericForecastIOConfig, store: str, task: str, horizon_minutes: int) -> List[str]:
    if str(store) == "forecast":
        return expected_forecast_columns(io_config, task=task, horizon_minutes=horizon_minutes)
    if str(store) == "eval":
        return expected_eval_columns(io_config, task=task, horizon_minutes=horizon_minutes)
    raise ValueError(f"Unsupported store: {store}")


def month_part_path(io_config: NumericForecastIOConfig, root: Path, interval: int, asset: str, year: int, month: int, store: str) -> Path:
    return root / module_table(io_config, store=store, interval=interval) / f"asset={str(asset)}" / f"year={int(year)}" / f"month={int(month):02d}" / "part-000.parquet"


def module_output_max_ts(io_config: NumericForecastIOConfig, *, root: Path, interval: int, asset: str, store: str) -> Optional[int]:
    base = root / module_table(io_config, store=store, interval=interval) / f"asset={str(asset)}"
    return partition_max_ts(base, ts_column="ts")


def validated_existing_month_parquet(
    io_config: NumericForecastIOConfig,
    dst: Path,
    *,
    asset: str,
    store: str,
    interval: int,
    task: str,
    horizon_minutes: int,
) -> pd.DataFrame:
    if not dst.exists():
        return pd.DataFrame(columns=["ts", "asset"])
    try:
        df = pd.read_parquet(dst)
    except Exception as exc:
        raise RuntimeError(
            f"{io_config.naming.log_prefix}[schema-error] unreadable {store} parquet for asset={asset} k={int(interval)} h={int(horizon_minutes)}m task={task} path={dst}: {exc}"
        ) from exc
    if "ts" not in df.columns or "asset" not in df.columns:
        raise RuntimeError(
            f"{io_config.naming.log_prefix}[schema-error] missing key columns for asset={asset} k={int(interval)} h={int(horizon_minutes)}m task={task} path={dst} columns={list(df.columns)}"
        )
    if df.empty:
        return df.copy()
    ts_num = pd.to_numeric(df["ts"], errors="coerce")
    if ts_num.isna().any():
        bad_rows = int(ts_num.isna().sum())
        raise RuntimeError(
            f"{io_config.naming.log_prefix}[schema-error] invalid ts dtype for asset={asset} k={int(interval)} h={int(horizon_minutes)}m task={task} path={dst} bad_rows={bad_rows}"
        )
    out = df.copy()
    out["ts"] = ts_num.astype("int64")
    out["asset"] = out["asset"].astype(str)
    duplicate_mask = out.duplicated(subset=["asset", "ts"], keep=False)
    if duplicate_mask.any():
        dup_count = int(duplicate_mask.sum())
        raise RuntimeError(
            f"{io_config.naming.log_prefix}[schema-error] duplicate key rows for asset={asset} k={int(interval)} h={int(horizon_minutes)}m task={task} path={dst} duplicate_rows={dup_count}"
        )
    return out


def pair_columns_state(df: pd.DataFrame, expected_cols: Sequence[str]) -> str:
    present_count = sum(1 for col in expected_cols if col in df.columns)
    if present_count <= 0:
        return "absent"
    if present_count < len(expected_cols):
        return "partial"
    return "complete"


def validated_module_month_parquet(
    io_config: NumericForecastIOConfig,
    path: Path,
    *,
    asset: str,
    store: str,
    interval: int,
    task: str,
    horizon_minutes: int,
    expected_cols: Sequence[str],
) -> Optional[pd.DataFrame]:
    out = validated_existing_month_parquet(io_config, path, asset=asset, store=store, interval=interval, task=task, horizon_minutes=horizon_minutes)
    if out.empty:
        return None
    if pair_columns_state(out, expected_cols) != "complete":
        return None
    for col in expected_cols:
        col_num = pd.to_numeric(out[col], errors="coerce")
        non_null_mask = out[col].notna()
        bad_mask = non_null_mask & col_num.isna()
        if bad_mask.any():
            bad_rows = int(bad_mask.sum())
            raise RuntimeError(
                f"{io_config.naming.log_prefix}[schema-error] invalid {store} dtype for asset={asset} k={int(interval)} h={int(horizon_minutes)}m task={task} path={path} column={col} bad_rows={bad_rows}"
            )
    return out


def _row_needs_recompute(frame: pd.DataFrame) -> pd.Series:
    if "needs_recompute" in frame.columns:
        return frame["needs_recompute"].fillna(False).astype(bool)
    if "is_forward_filled" in frame.columns:
        return frame["is_forward_filled"].fillna(False).astype(bool)
    return pd.Series(False, index=frame.index)


def coalesce_keyed_frames(frames: Sequence[pd.DataFrame], expected_cols: Sequence[str]) -> pd.DataFrame:
    valid_frames = [frame for frame in frames if frame is not None and not frame.empty]
    if not valid_frames:
        return pd.DataFrame(columns=["asset", "ts", *list(expected_cols)])
    merged: Optional[pd.DataFrame] = None
    ordered_expected = list(expected_cols)
    for frame in valid_frames:
        current = frame.copy()
        if "asset" not in current.columns or "ts" not in current.columns:
            continue
        current["asset"] = current["asset"].astype(str)
        current["ts"] = pd.to_numeric(current["ts"], errors="coerce").astype("int64")
        for col in ordered_expected:
            if col not in current.columns:
                current[col] = np.nan
        current = current.sort_values(["ts", "asset"]).drop_duplicates(subset=["asset", "ts"], keep="last")
        current_i = current.set_index(["asset", "ts"])
        if merged is None:
            merged = current_i
            continue
        union_idx = merged.index.union(current_i.index)
        merged = merged.reindex(union_idx)
        for col in current_i.columns:
            incoming = current_i[col].reindex(union_idx)
            if col in merged.columns:
                merged[col] = incoming.combine_first(merged[col])
            else:
                merged[col] = incoming
    if merged is None:
        return pd.DataFrame(columns=["asset", "ts", *ordered_expected])
    out = merged.reset_index()
    ordered_cols = ["asset", "ts"] + [col for col in ordered_expected if col in out.columns]
    extra_cols = [col for col in out.columns if col not in ordered_cols]
    return out[ordered_cols + extra_cols].sort_values(["ts", "asset"]).reset_index(drop=True)


def _normalize_parquet_contract_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "asset" in out.columns:
        # Write as Arrow `string`, not pandas StringDtype / Arrow large_string.
        out["asset"] = out["asset"].astype(str).astype(object)
    if "ts" in out.columns:
        out["ts"] = pd.to_numeric(out["ts"], errors="coerce").astype("int64")
    return out


def _write_parquet_with_contract(
    df: pd.DataFrame,
    dst: Path,
    *,
    compression: str,
    row_group_size: int,
) -> None:
    normalized = _normalize_parquet_contract_frame(df)
    table = pa.Table.from_pandas(normalized, preserve_index=False)
    pq.write_table(table, dst, compression=compression, row_group_size=row_group_size)


def _validate_completed_write_regions(
    frame: pd.DataFrame,
    *,
    interval: int,
    asset: str,
    store: str,
    year: int,
    month: int,
    part_metadata: Sequence[Dict[str, Any]],
) -> None:
    if frame.empty or not part_metadata:
        return
    ts_series = pd.to_numeric(frame["ts"], errors="coerce")
    for meta in part_metadata:
        required_cols = [str(col) for col in meta.get("expected_cols", []) if str(col) in frame.columns]
        if not required_cols:
            required_cols = [str(col) for col in frame.columns if str(col) not in {"ts", "asset"}]
        if not required_cols:
            continue
        start_ts = meta.get("min_ts")
        end_ts = meta.get("max_ts")
        if start_ts is None or end_ts is None:
            continue
        mask = (ts_series >= int(start_ts)) & (ts_series <= int(end_ts))
        if not bool(mask.any()):
            continue
        scoped = frame.loc[mask, ["ts", *required_cols]].sort_values("ts").reset_index(drop=True)
        validate_no_nan_columns(
            scoped,
            columns=required_cols,
            context=(
                f"[write-contract] asset={str(asset)} k={int(interval)} "
                f"h={int(meta.get('horizon_minutes', 0) or 0)}m task={str(meta.get('task', 'unknown'))} "
                f"store={str(store)} context=asset-month={int(year):04d}-{int(month):02d}"
            ),
            ts_column="ts",
        )
        validate_strict_timegrid(
            scoped["ts"],
            interval_min=int(interval),
            context=(
                f"[write-contract] asset={str(asset)} k={int(interval)} "
                f"h={int(meta.get('horizon_minutes', 0) or 0)}m task={str(meta.get('task', 'unknown'))} "
                f"store={str(store)} context=asset-month={int(year):04d}-{int(month):02d}"
            ),
        )


def _trim_to_complete_value_region(
    frame: pd.DataFrame,
    *,
    interval: int,
    asset: str,
    store: str,
    year: int,
    month: int,
    required_cols: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    if frame.empty:
        return frame
    value_cols = [str(col) for col in (required_cols or []) if str(col) in frame.columns]
    if not value_cols:
        value_cols = [str(col) for col in frame.columns if str(col) not in {"asset", "ts"}]
    if not value_cols:
        return frame
    ordered = frame.sort_values(["ts", "asset"]).reset_index(drop=True)
    completeness = ordered.loc[:, value_cols].replace([np.inf, -np.inf], np.nan).notna().all(axis=1)
    if not bool(completeness.any()):
        return ordered.iloc[0:0].copy()
    first_complete_pos = int(np.where(completeness.to_numpy(dtype=bool, copy=False))[0][0])
    complete_region = ordered.iloc[first_complete_pos:].reset_index(drop=True)
    region_complete = complete_region.loc[:, value_cols].replace([np.inf, -np.inf], np.nan).notna().all(axis=1)
    if not bool(region_complete.all()):
        first_bad_pos = int(np.where(~region_complete.to_numpy(dtype=bool, copy=False))[0][0])
        first_bad_ts = int(pd.to_numeric(complete_region.iloc[first_bad_pos]["ts"], errors="coerce"))
        raise PipelineValidationError(
            (
                f"[write-contract] asset={str(asset)} k={int(interval)} store={str(store)} "
                f"context=asset-month={int(year):04d}-{int(month):02d}: incomplete row after complete head"
            ),
            first_bad_ts=first_bad_ts,
        )
    validate_strict_timegrid(
        complete_region["ts"],
        interval_min=int(interval),
        context=(
            f"[write-contract] asset={str(asset)} k={int(interval)} store={str(store)} "
            f"context=asset-month={int(year):04d}-{int(month):02d} complete-region"
        ),
    )
    return complete_region


def write_month_parts(
    io_config: NumericForecastIOConfig,
    month_frames: Dict[Tuple[int, int], List[pd.DataFrame]],
    *,
    root: Path,
    interval: int,
    asset: str,
    horizon_minutes: int,
    task: str,
    store: str,
    expected_cols: Optional[Sequence[str]] = None,
    part_metadata_by_month: Optional[Dict[Tuple[int, int], List[Dict[str, Any]]]] = None,
) -> List[dict]:
    if not month_frames:
        return []
    expected_cols = list(expected_cols) if expected_cols is not None else expected_store_columns(io_config, store=store, task=task, horizon_minutes=horizon_minutes)
    parts: List[dict] = []
    for (y, m), frames in sorted(month_frames.items()):
        if not frames:
            continue
        chunk = coalesce_keyed_frames(frames, expected_cols)
        dst = month_part_path(io_config, root=root, interval=interval, asset=asset, year=int(y), month=int(m), store=store)
        assert_write_allowed(dst, f"numeric {store} parquet")
        existing = validated_existing_month_parquet(io_config, dst, asset=asset, store=store, interval=interval, task=task, horizon_minutes=horizon_minutes)
        if not existing.empty:
            for col in expected_cols:
                if col not in existing.columns:
                    existing[col] = np.nan
            existing_i = existing.set_index(["asset", "ts"])
            chunk_i = chunk.set_index(["asset", "ts"])
            union_idx = existing_i.index.union(chunk_i.index)
            merged = existing_i.reindex(union_idx)
            for col in chunk_i.columns:
                incoming = chunk_i[col].reindex(union_idx)
                if col in merged.columns:
                    merged[col] = incoming.combine_first(merged[col])
                else:
                    merged[col] = incoming
            chunk = merged.reset_index()
            chunk = chunk.sort_values(["ts", "asset"]).drop_duplicates(subset=["asset", "ts"], keep="last")
        chunk = _trim_to_complete_value_region(
            chunk,
            interval=int(interval),
            asset=str(asset),
            store=str(store),
            year=int(y),
            month=int(m),
            required_cols=list(expected_cols),
        )
        if chunk.empty:
            dst.parent.mkdir(parents=True, exist_ok=True)
            tmp = sibling_temp_path(dst, suffix=".parquet.tmp")
            assert_write_allowed(tmp, f"numeric {store} parquet temp")
            _write_parquet_with_contract(
                chunk,
                tmp,
                compression=io_config.parquet_compression,
                row_group_size=io_config.parquet_row_group,
            )
            atomic_replace(tmp, dst)
            continue
        month_part_metadata = part_metadata_by_month.get((int(y), int(m)), []) if part_metadata_by_month else []
        _validate_completed_write_regions(
            chunk,
            interval=int(interval),
            asset=str(asset),
            store=str(store),
            year=int(y),
            month=int(m),
            part_metadata=month_part_metadata,
        )
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = sibling_temp_path(dst, suffix=".parquet.tmp")
        assert_write_allowed(tmp, f"numeric {store} parquet temp")
        _write_parquet_with_contract(
            chunk,
            tmp,
            compression=io_config.parquet_compression,
            row_group_size=io_config.parquet_row_group,
        )
        atomic_replace(tmp, dst)
        base_part = {
            "path": str(dst),
            "rows": int(len(chunk)),
            "assets": sorted(set(str(x) for x in chunk["asset"].astype(str).tolist())),
            "interval": int(interval),
            "horizon_minutes": int(horizon_minutes),
            "task": task,
            "store": store,
            "year": int(y),
            "month": int(m),
            "min_ts": int(chunk["ts"].min()),
            "max_ts": int(chunk["ts"].max()),
        }
        if not month_part_metadata:
            parts.append(dict(base_part))
            continue
        for meta in month_part_metadata:
            part = dict(base_part)
            part.update(meta)
            parts.append(part)
    return parts


def stage_month_parts(
    io_config: NumericForecastIOConfig,
    month_frames: Dict[Tuple[int, int], List[pd.DataFrame]],
    *,
    group_id: str,
    interval: int,
    asset: str,
    horizon_minutes: int,
    task: str,
    store: str,
    expected_cols: Sequence[str],
    part_metadata_by_month: Optional[Dict[Tuple[int, int], List[Dict[str, Any]]]] = None,
    chunk_idx: int,
) -> List[Dict[str, Any]]:
    staged: List[Dict[str, Any]] = []
    for (y, m), frames in sorted(month_frames.items()):
        if not frames:
            continue
        chunk = coalesce_keyed_frames(frames, expected_cols)
        stage_dir = io_config.staging_root / f"store={str(store)}" / f"asset={str(asset)}" / f"interval={int(interval)}" / f"horizon={int(horizon_minutes)}m" / f"year={int(y)}" / f"month={int(m):02d}"
        stage_path = stage_dir / f"chunk={int(chunk_idx):04d}-pid={os.getpid()}-ns={time.time_ns()}.parquet"
        assert_write_allowed(stage_path, "numeric staged parquet")
        stage_dir.mkdir(parents=True, exist_ok=True)
        _write_parquet_with_contract(
            chunk,
            stage_path,
            compression=io_config.parquet_compression,
            row_group_size=io_config.parquet_row_group,
        )
        staged.append(
            {
                "group_id": str(group_id),
                "stage_path": str(stage_path),
                "interval": int(interval),
                "asset": str(asset),
                "horizon_minutes": int(horizon_minutes),
                "task": str(task),
                "store": str(store),
                "year": int(y),
                "month": int(m),
                "chunk_idx": int(chunk_idx),
                "expected_cols": list(expected_cols),
                "part_metadata": list(part_metadata_by_month.get((int(y), int(m)), [])) if part_metadata_by_month else [],
            }
        )
    return staged


def init_stage_write_queue(write_queue: Any) -> None:
    global _STAGE_WRITE_QUEUE
    _STAGE_WRITE_QUEUE = write_queue


def enqueue_stage_write_batch(group_id: str, items: Sequence[Dict[str, Any]]) -> None:
    if not items:
        return
    if _STAGE_WRITE_QUEUE is None:
        raise RuntimeError("Stage write queue is not initialized.")
    _STAGE_WRITE_QUEUE.put({"kind": "write_batch", "group_id": str(group_id), "items": list(items)})


def enqueue_stage_group_done(group_id: str) -> None:
    if _STAGE_WRITE_QUEUE is None:
        raise RuntimeError("Stage write queue is not initialized.")
    _STAGE_WRITE_QUEUE.put({"kind": "group_done", "group_id": str(group_id)})


def merge_write_batch_state(io_config: NumericForecastIOConfig, group_state: Dict[str, Any], items: Sequence[Dict[str, Any]]) -> None:
    staged_groups: Dict[Tuple[int, str], List[Dict[str, Any]]] = {}
    for item in items:
        staged_groups.setdefault((int(item["chunk_idx"]), str(item["store"])), []).append(item)
    task_state = group_state.setdefault("task_state", {})
    for (_, store), grouped_items in sorted(staged_groups.items(), key=lambda pair: (pair[0][0], pair[0][1])):
        t0 = time.monotonic()
        flushed_parts: List[dict] = []
        touched_tasks = sorted({str(meta.get("task")) for item in grouped_items for meta in item.get("part_metadata", []) if meta.get("task") is not None})
        for item in grouped_items:
            stage_path = Path(str(item["stage_path"]))
            assert_write_allowed(stage_path, "numeric staged parquet read/delete")
            staged_df = pd.read_parquet(stage_path)
            month_key = (int(item["year"]), int(item["month"]))
            flushed_parts.extend(
                write_month_parts(
                    io_config,
                    month_frames={month_key: [staged_df]},
                    root=io_config.parquet_root,
                    interval=int(item["interval"]),
                    asset=str(item["asset"]),
                    horizon_minutes=int(item["horizon_minutes"]),
                    task=str(item["task"]),
                    store=str(item["store"]),
                    expected_cols=list(item.get("expected_cols", [])),
                    part_metadata_by_month={month_key: list(item.get("part_metadata", []))},
                )
            )
            assert_write_allowed(stage_path, "numeric staged parquet delete")
            try:
                stage_path.unlink(missing_ok=True)
            except Exception:
                pass
        write_share_s = float(time.monotonic() - t0) / len(touched_tasks) if touched_tasks else 0.0
        parts_by_task: Dict[str, List[dict]] = {}
        for part in flushed_parts:
            parts_by_task.setdefault(str(part.get("task")), []).append(part)
        for task in touched_tasks:
            info = task_state.setdefault(str(task), {"forecast_parts": [], "eval_parts": [], "forecast_write_s": 0.0, "eval_write_s": 0.0})
            if str(store) == "forecast":
                info["forecast_parts"].extend(parts_by_task.get(str(task), []))
                info["forecast_write_s"] = float(info["forecast_write_s"]) + float(write_share_s)
            else:
                info["eval_parts"].extend(parts_by_task.get(str(task), []))
                info["eval_write_s"] = float(info["eval_write_s"]) + float(write_share_s)


def stage_writer_loop(io_config: NumericForecastIOConfig, write_queue: Any, writer_state: Dict[str, Any], writer_cv: threading.Condition) -> None:
    while True:
        message = write_queue.get()
        kind = str(message.get("kind"))
        if kind == "stop":
            return
        group_id = str(message.get("group_id"))
        with writer_cv:
            state = writer_state.setdefault(group_id, {"done": False, "task_state": {}})
        try:
            if kind == "write_batch":
                merge_write_batch_state(io_config, state, list(message.get("items", []) or []))
                continue
            if kind == "group_done":
                with writer_cv:
                    state["done"] = True
                    writer_cv.notify_all()
        except Exception as exc:
            with writer_cv:
                state["done"] = True
                state["error"] = exc
                writer_state["__fatal__"] = {"group_id": group_id, "error": exc}
                writer_cv.notify_all()
            return


def finalize_group_results(io_config: NumericForecastIOConfig, group_payload: Dict[str, Any], writer_state: Dict[str, Any], writer_cv: threading.Condition) -> List[Dict[str, Any]]:
    group_id = str(group_payload.get("group_id"))
    with writer_cv:
        while True:
            fatal_state = writer_state.get("__fatal__")
            if isinstance(fatal_state, dict) and fatal_state.get("error") is not None:
                raise fatal_state["error"]
            state = writer_state.get(group_id)
            if state is not None and bool(state.get("done")):
                if state.get("error") is not None:
                    raise state["error"]
                group_state = writer_state.pop(group_id)
                break
            writer_cv.wait(timeout=0.5)
    task_state = group_state.get("task_state", {})
    results = [dict(res) for res in group_payload.get("results", [])]
    for res in results:
        task = str(res.get("task"))
        info = task_state.get(task, {})
        res["diag_summary"]["parquet_write_s"] = float(res["diag_summary"].get("parquet_write_s", 0.0)) + float(info.get("forecast_write_s", 0.0)) + float(info.get("eval_write_s", 0.0))
        res["pred_parts"].extend(list(info.get("forecast_parts", [])))
        res["eval_parts"].extend(list(info.get("eval_parts", [])))
        io_config.log_fn(f"{io_config.naming.log_prefix}[diag-summary] {res['unit_label']} {json.dumps(res['diag_summary'], sort_keys=True)}")
    return results


def completed_edge_from_module_parquet(
    io_config: NumericForecastIOConfig,
    *,
    root: Path,
    interval: int,
    asset: str,
    task: str,
    horizon_minutes: int,
    store: str,
    start_ts: int,
    stop_ts: int,
    step_seconds: int,
    allow_head_gap: bool = True,
    include_recompute: bool = False,
) -> Optional[int]:
    base = root / module_table(io_config, store=store, interval=interval) / f"asset={str(asset)}"
    if not base.exists():
        return None
    expected_cols = expected_store_columns(io_config, store=store, task=task, horizon_minutes=horizon_minutes)
    completed_ts: List[np.ndarray] = []
    for p in sorted(base.glob("year=*/month=*/*.parquet")):
        d = validated_module_month_parquet(io_config, p, asset=asset, store=store, interval=interval, task=task, horizon_minutes=horizon_minutes, expected_cols=expected_cols)
        if d is None or d.empty:
            continue
        asset_mask = d["asset"] == str(asset)
        ts_num = d["ts"]
        populated_mask = asset_mask
        for col in expected_cols:
            populated_mask = populated_mask & pd.to_numeric(d[col], errors="coerce").notna()
        if not bool(include_recompute):
            populated_mask = populated_mask & ~_row_needs_recompute(d)
        if not populated_mask.any():
            continue
        ts = ts_num.loc[populated_mask].astype("int64")
        ts = ts[(ts >= int(start_ts)) & (ts <= int(stop_ts))]
        if ts.empty:
            continue
        completed_ts.append(ts.to_numpy(dtype=np.int64))
    if not completed_ts:
        return None
    merged = np.unique(np.concatenate(completed_ts))
    expected = max(int(start_ts), int(merged[0])) if bool(allow_head_gap) else int(start_ts)
    last_complete: Optional[int] = None
    for ts_i in merged:
        cur = int(ts_i)
        if cur < expected:
            continue
        if cur != expected:
            break
        last_complete = cur
        expected += int(step_seconds)
        if last_complete >= int(stop_ts):
            break
    return last_complete
