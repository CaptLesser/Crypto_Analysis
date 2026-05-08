from __future__ import annotations

import ast
import json
import os
import re
import threading
import time
import uuid
from collections import deque
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from queue import Empty, Queue
from typing import Any, Callable, Dict, List, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from src.features.numeric_forecast_profiles import normalize_refit_cadence
from src.forecasting.common.io_atomic import atomic_replace, sibling_temp_path
from src.forecasting.common.sandbox_paths import assert_write_allowed
from src.forecasting.ml.shared.production_time import production_start_ts
from src.forecasting.ml.shared.numeric_runner_diagnostics import append_diagnostic_event, resource_snapshot
from src.forecasting.ml.shared.numeric_forecast_io import (
    NumericForecastIOConfig,
    NumericForecastNamingConfig,
    coalesce_keyed_frames,
    iter_months_between,
    module_table,
    month_part_path,
    validated_existing_month_parquet,
    write_month_parts,
)


def project_root() -> Path:
    return Path(__file__).resolve().parents[4]


def finalized_origin_indices(ts_vec: np.ndarray, *, start_ts: int, end_ts: int, predict_latest_only: bool) -> List[int]:
    indices = [int(idx) for idx, ts_val in enumerate(ts_vec) if int(start_ts) <= int(ts_val) <= int(end_ts)]
    if bool(predict_latest_only) and indices:
        indices = [indices[-1]]
    return indices


def partition_assets(assets: Sequence[str], shard_count: int) -> List[List[str]]:
    ordered_assets = [str(asset) for asset in assets if str(asset)]
    if not ordered_assets:
        return []
    resolved_shard_count = max(1, min(int(shard_count), len(ordered_assets)))
    return [ordered_assets[index::resolved_shard_count] for index in range(resolved_shard_count) if ordered_assets[index::resolved_shard_count]]


def parse_combo_list(raw: str) -> List[Tuple[int, int, str]]:
    combos: List[Tuple[int, int, str]] = []
    for token in [part.strip() for part in str(raw).split(",") if part.strip()]:
        interval, horizon, task = token.split(":", 2)
        combos.append((int(interval), int(horizon), str(task)))
    return sorted(set(combos), key=lambda item: (item[0], item[1], item[2]))


DEFAULT_NUMERIC_TASK_SHORT: Dict[str, str] = {
    "log_return": "lr",
    "realized_vol": "rv",
    "true_range": "tr",
    "max_drawdown": "mdd",
    "max_runup": "mru",
    "range_efficiency": "re",
    "direction": "dir",
}

DEFAULT_NUMERIC_TASK_LABEL: Dict[str, str] = {
    "log_return": "future_log_return",
    "realized_vol": "future_realized_vol",
    "true_range": "future_true_range",
    "max_drawdown": "future_max_drawdown",
    "max_runup": "future_max_runup",
    "range_efficiency": "future_range_efficiency",
    "direction": "future_direction",
}


def canonical_table_slug(raw: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", str(raw).strip()).strip("_").lower()
    return slug or "numeric_forecast"


def canonical_physical_naming(
    *,
    module_slug: str,
    prediction_prefix: Optional[str] = None,
    task_short: Optional[Dict[str, str]] = None,
    task_label: Optional[Dict[str, str]] = None,
    log_prefix: str = "",
) -> NumericForecastNamingConfig:
    slug = canonical_table_slug(module_slug)
    return NumericForecastNamingConfig(
        module_slug=slug,
        forecast_table_tag=slug,
        eval_table_tag=f"{slug}_eval",
        prediction_prefix=canonical_table_slug(prediction_prefix or slug),
        task_short={**DEFAULT_NUMERIC_TASK_SHORT, **dict(task_short or {})},
        task_label={**DEFAULT_NUMERIC_TASK_LABEL, **dict(task_label or {})},
        log_prefix=str(log_prefix or f"[{slug}]"),
    )


def canonical_physical_io_config(
    *,
    naming: NumericForecastNamingConfig,
    parquet_root: Path,
    staging_root: Path,
    state_root: Path,
    scalar_root: Optional[Path] = None,
    ohlc_root: Optional[Path] = None,
    parquet_compression: str = "zstd",
    parquet_row_group: int = 500000,
    log_fn: Optional[Callable[[str], None]] = None,
) -> NumericForecastIOConfig:
    empty_root = Path(parquet_root)
    return NumericForecastIOConfig(
        naming=naming,
        parquet_root=Path(parquet_root),
        staging_root=Path(staging_root),
        state_root=Path(state_root),
        scalar_root=Path(scalar_root or empty_root),
        ohlc_root=Path(ohlc_root or empty_root),
        parquet_compression=str(parquet_compression),
        parquet_row_group=int(parquet_row_group),
        log_fn=log_fn or (lambda _message: None),
        read_ohlcvt_fn=lambda **_kwargs: pd.DataFrame(),
        list_assets_from_ohlcvt_fn=lambda _interval: [],
        first_ohlcvt_ts_fn=lambda *_args, **_kwargs: None,
        ohlcvt_max_ts_fn=lambda *_args, **_kwargs: None,
        feature_max_ts_fn=lambda *_args, **_kwargs: None,
    )


def canonical_forecast_column(*, naming: NumericForecastNamingConfig, source_col: str, task: str, horizon_minutes: int) -> str:
    short = str(naming.task_short.get(str(task), canonical_table_slug(task)))
    source = canonical_table_slug(source_col)
    return f"{naming.prediction_prefix}_{source}_{short}_{int(horizon_minutes)}m"


def canonical_eval_actual_column(*, naming: NumericForecastNamingConfig, target_col: str, task: str, horizon_minutes: int) -> str:
    base = str(target_col or naming.task_label.get(str(task), str(task)))
    return f"{canonical_table_slug(base)}_{int(horizon_minutes)}m"


def canonical_eval_metric_column(*, naming: NumericForecastNamingConfig, source_col: str, task: str, horizon_minutes: int) -> str:
    short = str(naming.task_short.get(str(task), canonical_table_slug(task)))
    return f"{naming.prediction_prefix}_{canonical_table_slug(source_col)}_{short}_{int(horizon_minutes)}m"


def canonical_prediction_value_columns(rows: Sequence[Dict[str, Any]]) -> List[str]:
    preferred = ["pred_mean", "pred_std", "pred_p10", "pred_p50", "pred_p90"]
    seen = {str(col) for row in rows for col in row.keys()}
    ordered = [col for col in preferred if col in seen]
    extras = sorted(
        col
        for col in seen
        if col.startswith("pred_") and col not in set(ordered)
    )
    return [*ordered, *extras]


def canonicalize_forecast_rows(
    *,
    rows: Sequence[Dict[str, Any]],
    naming: NumericForecastNamingConfig,
    task: str,
    horizon_minutes: int,
    prediction_cols: Optional[Sequence[str]] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    prediction_cols = list(prediction_cols or canonical_prediction_value_columns(rows))
    expected_cols = [
        canonical_forecast_column(naming=naming, source_col=col, task=task, horizon_minutes=int(horizon_minutes))
        for col in prediction_cols
    ]
    out_rows: List[Dict[str, Any]] = []
    for row in rows:
        if row.get("asset") is None or row.get("ts") is None:
            continue
        out: Dict[str, Any] = {"asset": str(row["asset"]), "ts": int(row["ts"])}
        for source_col, dst_col in zip(prediction_cols, expected_cols):
            if source_col in row:
                out[dst_col] = row.get(source_col)
        out_rows.append(out)
    if not out_rows:
        return pd.DataFrame(columns=["asset", "ts", *expected_cols]), expected_cols
    return pd.DataFrame(out_rows), expected_cols


def canonicalize_eval_rows(
    *,
    rows: Sequence[Dict[str, Any]],
    naming: NumericForecastNamingConfig,
    task: str,
    horizon_minutes: int,
) -> Tuple[pd.DataFrame, List[str]]:
    metric_sources = ["error_p50", "abs_error_p50", "squared_error_p50"]
    expected_cols: List[str] = []
    out_rows: List[Dict[str, Any]] = []
    for row in rows:
        if row.get("asset") is None or row.get("ts") is None:
            continue
        target_col = str(row.get("target_col") or naming.task_label.get(str(task), str(task)))
        actual_col = canonical_eval_actual_column(
            naming=naming,
            target_col=target_col,
            task=task,
            horizon_minutes=int(horizon_minutes),
        )
        if actual_col not in expected_cols:
            expected_cols.append(actual_col)
        out: Dict[str, Any] = {"asset": str(row["asset"]), "ts": int(row["ts"])}
        if "actual" in row:
            out[actual_col] = row.get("actual")
        for source_col in metric_sources:
            if source_col in row:
                dst_col = canonical_eval_metric_column(
                    naming=naming,
                    source_col=source_col,
                    task=task,
                    horizon_minutes=int(horizon_minutes),
                )
                if dst_col not in expected_cols:
                    expected_cols.append(dst_col)
                out[dst_col] = row.get(source_col)
        out_rows.append(out)
    if not out_rows:
        return pd.DataFrame(columns=["asset", "ts", *expected_cols]), expected_cols
    return pd.DataFrame(out_rows), expected_cols


def _month_frames_by_asset(df: pd.DataFrame) -> Dict[str, Dict[Tuple[int, int], List[pd.DataFrame]]]:
    if df.empty:
        return {}
    work = df.copy()
    work["asset"] = work["asset"].astype(str)
    work["ts"] = pd.to_numeric(work["ts"], errors="coerce")
    work = work.dropna(subset=["asset", "ts"]).copy()
    if work.empty:
        return {}
    work["ts"] = work["ts"].astype("int64")
    dt = pd.to_datetime(work["ts"], unit="s", utc=True)
    work["_year"] = dt.dt.year.astype(int)
    work["_month"] = dt.dt.month.astype(int)
    grouped: Dict[str, Dict[Tuple[int, int], List[pd.DataFrame]]] = {}
    for (asset, year, month), grp in work.groupby(["asset", "_year", "_month"], sort=True):
        frame = grp.drop(columns=["_year", "_month"]).copy()
        grouped.setdefault(str(asset), {}).setdefault((int(year), int(month)), []).append(frame)
    return grouped


def write_canonical_physical_prediction_month_frames(
    *,
    io_config: NumericForecastIOConfig,
    interval_minutes: int,
    run_id: str,
    module_tag: str,
    task: str,
    horizon_minutes: int,
    month_frames: Dict[Tuple[int, int], List[pd.DataFrame]],
    store: str = "forecast",
    existing_key_cache: Optional[Dict[Any, Any]] = None,
) -> List[Dict[str, Any]]:
    del run_id, module_tag, existing_key_cache
    frames = [frame for frames in month_frames.values() for frame in frames if frame is not None and not frame.empty]
    if not frames:
        return []
    long_df = pd.concat(frames, ignore_index=True)
    if str(store) == "eval":
        canonical_df, expected_cols = canonicalize_eval_rows(
            rows=long_df.to_dict(orient="records"),
            naming=io_config.naming,
            task=str(task),
            horizon_minutes=int(horizon_minutes),
        )
    else:
        canonical_df, expected_cols = canonicalize_forecast_rows(
            rows=long_df.to_dict(orient="records"),
            naming=io_config.naming,
            task=str(task),
            horizon_minutes=int(horizon_minutes),
        )
    if canonical_df.empty or not expected_cols:
        return []
    parts: List[Dict[str, Any]] = []
    for asset, asset_month_frames in _month_frames_by_asset(canonical_df).items():
        part_metadata = {
            month_key: [
                {
                    "task": str(task),
                    "horizon_minutes": int(horizon_minutes),
                    "expected_cols": list(expected_cols),
                }
            ]
            for month_key in asset_month_frames
        }
        parts.extend(
            write_month_parts(
                io_config,
                month_frames=asset_month_frames,
                root=io_config.parquet_root,
                interval=int(interval_minutes),
                asset=str(asset),
                horizon_minutes=int(horizon_minutes),
                task=str(task),
                store=str(store),
                expected_cols=list(expected_cols),
                part_metadata_by_month=part_metadata,
            )
        )
    return parts


def write_canonical_physical_predictions(
    *,
    io_config: NumericForecastIOConfig,
    out_root: Path,
    interval_minutes: int,
    run_id: str,
    module_tag: str,
    task: str,
    horizon_minutes: int,
    df: pd.DataFrame,
    store: str = "forecast",
    existing_key_cache: Optional[Dict[Any, Any]] = None,
) -> List[Dict[str, Any]]:
    del out_root
    if df.empty:
        return []
    dt = pd.to_datetime(pd.to_numeric(df["ts"], errors="coerce"), unit="s", utc=True)
    work = df.copy()
    work["_year"] = dt.dt.year.astype("Int64")
    work["_month"] = dt.dt.month.astype("Int64")
    month_frames: Dict[Tuple[int, int], List[pd.DataFrame]] = {}
    for (year, month), grp in work.dropna(subset=["_year", "_month"]).groupby(["_year", "_month"], sort=True):
        month_frames.setdefault((int(year), int(month)), []).append(grp.drop(columns=["_year", "_month"]).copy())
    return write_canonical_physical_prediction_month_frames(
        io_config=io_config,
        interval_minutes=int(interval_minutes),
        run_id=str(run_id),
        module_tag=str(module_tag),
        task=str(task),
        horizon_minutes=int(horizon_minutes),
        month_frames=month_frames,
        store=str(store),
        existing_key_cache=existing_key_cache,
    )


def canonical_physical_output_tail_ts(
    *,
    io_config: NumericForecastIOConfig,
    interval_minutes: int,
    task: str,
    horizon_minutes: int,
    asset: str,
    store: str = "forecast",
) -> Optional[int]:
    base = io_config.parquet_root / module_table(io_config, store=store, interval=int(interval_minutes)) / f"asset={str(asset)}"
    if not base.exists():
        return None
    if str(store) == "eval":
        required_cols = [
            canonical_eval_actual_column(
                naming=io_config.naming,
                target_col=io_config.naming.task_label.get(str(task), str(task)),
                task=str(task),
                horizon_minutes=int(horizon_minutes),
            )
        ]
    else:
        required_cols = [
            canonical_forecast_column(
                naming=io_config.naming,
                source_col="pred_p50",
                task=str(task),
                horizon_minutes=int(horizon_minutes),
            )
        ]
    completed_ts: List[np.ndarray] = []
    for path in sorted(base.glob("year=*/month=*/*.parquet")):
        frame = validated_existing_month_parquet(
            io_config,
            path,
            asset=str(asset),
            store=str(store),
            interval=int(interval_minutes),
            task=str(task),
            horizon_minutes=int(horizon_minutes),
        )
        if frame.empty or not all(col in frame.columns for col in required_cols):
            continue
        mask = frame["asset"].astype(str).eq(str(asset))
        for col in required_cols:
            mask = mask & pd.to_numeric(frame[col], errors="coerce").notna()
        ts = pd.to_numeric(frame.loc[mask, "ts"], errors="coerce").dropna().astype("int64")
        if not ts.empty:
            completed_ts.append(ts.to_numpy(dtype=np.int64))
    if not completed_ts:
        return None
    merged = np.unique(np.concatenate(completed_ts))
    if merged.size == 0:
        return None
    step = int(interval_minutes) * 60
    expected = int(merged[0])
    last_complete: Optional[int] = None
    for ts_i in merged:
        cur = int(ts_i)
        if cur != expected:
            break
        last_complete = cur
        expected += int(step)
    return int(last_complete) if last_complete is not None else None


_CANONICAL_FORECAST_COL_RE = re.compile(
    r"^(?P<prefix>.+?)_pred_.+?_(?P<task_short>[a-z0-9]+)_(?P<horizon>\d+)m$",
    re.IGNORECASE,
)


def discover_existing_combo_specs_from_canonical_physical_output(
    *,
    io_config: NumericForecastIOConfig,
) -> Tuple[Tuple[int, int, str], ...]:
    combos: set[Tuple[int, int, str]] = set()
    root = Path(io_config.parquet_root)
    if not root.exists():
        return tuple()
    table_prefix = f"{io_config.naming.forecast_table_tag}_"
    task_by_short = {str(v): str(k) for k, v in io_config.naming.task_short.items()}
    for table_dir in sorted(root.glob(f"{table_prefix}*")):
        if not table_dir.is_dir():
            continue
        try:
            interval = int(str(table_dir.name)[len(table_prefix) :])
        except Exception:
            continue
        for path in sorted(table_dir.glob("asset=*/year=*/month=*/*.parquet")):
            try:
                cols = [str(name) for name in pq.read_schema(path).names]
            except Exception:
                continue
            for col in cols:
                match = _CANONICAL_FORECAST_COL_RE.match(str(col))
                if match is None:
                    continue
                task = task_by_short.get(str(match.group("task_short")), str(match.group("task_short")))
                combos.add((int(interval), int(match.group("horizon")), str(task)))
    return tuple(sorted(combos, key=lambda item: (item[0], item[1], item[2])))


def runtime_target_label_window(
    *,
    parquet_root: Path,
    asset: str,
    interval: int,
    horizon_minutes: Optional[int] = None,
    horizon_bars: Optional[int] = None,
    target_col: str,
    start_ts: int,
    end_ts: int,
    read_ohlcvt_fn: Callable[..., pd.DataFrame],
    compute_future_labels_fn: Callable[..., Tuple[pd.DataFrame, Any]],
    future_direction_deadzone: float = 0.0,
) -> pd.DataFrame:
    if horizon_bars is None:
        if horizon_minutes is None:
            resolved_horizon_bars = 1
        else:
            resolved_horizon_bars = max(1, int(horizon_minutes) // max(1, int(interval)))
    else:
        resolved_horizon_bars = max(1, int(horizon_bars))
    try:
        ohlc = read_ohlcvt_fn(
            root=parquet_root,
            asset=str(asset),
            interval_min=int(interval),
            start_ts=int(start_ts),
            end_ts=int(end_ts),
            columns=["ts", "asset", "high", "low", "close"],
        )
    except Exception:
        return pd.DataFrame(columns=["ts", "asset", str(target_col)])
    if ohlc.empty:
        return pd.DataFrame(columns=["ts", "asset", str(target_col)])
    try:
        labels, _stats = compute_future_labels_fn(
            ohlc.loc[:, ["high", "low", "close"]].reset_index(drop=True),
            int(resolved_horizon_bars),
            future_direction_deadzone=float(future_direction_deadzone),
            target_columns=[str(target_col)],
        )
    except TypeError:
        labels, _stats = compute_future_labels_fn(
            ohlc.loc[:, ["high", "low", "close"]].reset_index(drop=True),
            int(resolved_horizon_bars),
            future_direction_deadzone=float(future_direction_deadzone),
        )
    if str(target_col) not in labels.columns:
        return pd.DataFrame(columns=["ts", "asset", str(target_col)])
    out = pd.DataFrame(
        {
            "ts": pd.to_numeric(ohlc["ts"], errors="coerce"),
            "asset": ohlc["asset"].astype(str) if "asset" in ohlc.columns else str(asset),
            str(target_col): pd.to_numeric(labels[str(target_col)], errors="coerce"),
        }
    )
    out = out.dropna(subset=["ts"]).copy()
    if out.empty:
        return pd.DataFrame(columns=["ts", "asset", str(target_col)])
    out["ts"] = out["ts"].astype("int64")
    return out.loc[:, ["ts", "asset", str(target_col)]].sort_values("ts").reset_index(drop=True)


def overlay_runtime_target_labels(
    *,
    frame: pd.DataFrame,
    parquet_root: Path,
    asset: str,
    interval: int,
    horizon_minutes: int,
    target_col: str,
    start_ts: int,
    end_ts: int,
    read_ohlcvt_fn: Callable[..., pd.DataFrame],
    compute_future_labels_fn: Callable[..., Tuple[pd.DataFrame, Any]],
) -> pd.DataFrame:
    if not frame.empty and str(target_col) in frame.columns:
        existing = pd.to_numeric(frame[str(target_col)], errors="coerce")
        if existing.notna().any():
            return frame.sort_values("ts").reset_index(drop=True)
    runtime_labels = runtime_target_label_window(
        parquet_root=parquet_root,
        asset=asset,
        interval=interval,
        horizon_minutes=horizon_minutes,
        target_col=target_col,
        start_ts=start_ts,
        end_ts=end_ts,
        read_ohlcvt_fn=read_ohlcvt_fn,
        compute_future_labels_fn=compute_future_labels_fn,
    )
    if runtime_labels.empty:
        return frame
    if frame.empty:
        return runtime_labels.loc[:, ["ts", "asset", target_col]].sort_values("ts").reset_index(drop=True)
    merged = frame.drop(columns=[target_col], errors="ignore").merge(
        runtime_labels,
        on=["ts", "asset"],
        how="left",
        sort=True,
    )
    return merged.sort_values("ts").reset_index(drop=True)


def no_work_status_update(
    *,
    asset: str,
    interval: int,
    horizon_minutes: int,
    discover_edge_and_min_fn: Callable[..., Tuple[Optional[int], Optional[int]]],
) -> Dict[str, Any]:
    edge_ts, min_ts = discover_edge_and_min_fn(asset=str(asset), interval_minutes=int(interval))
    if edge_ts is None:
        return {"status": "skipped", "reason": "missing_edge_ts"}
    if min_ts is None:
        return {"status": "skipped", "reason": "missing_min_ts", "edge_ts": int(edge_ts)}
    target_tail = int(edge_ts) - int(horizon_minutes) * 60
    if int(target_tail) < int(min_ts):
        return {"status": "done", "reason": "no_closed_target"}
    return {"status": "done", "reason": "at_edge", "edge_ts": int(edge_ts)}


@dataclass(frozen=True)
class PlannedAssetWorkSpan:
    asset: str
    edge_ts: int
    target_tail_ts: int
    start_ts: int
    read_start_ts: int
    dst_tail_ts: Optional[int]


TailPlanningCacheKey = Tuple[str, str, str, int, int, str, bool]


class TailPlanningCache:
    """Run-local memo for exact output-tail planning questions."""

    def __init__(self) -> None:
        self._values: Dict[TailPlanningCacheKey, Optional[int]] = {}
        self._locks: Dict[TailPlanningCacheKey, threading.Lock] = {}
        self._guard = threading.Lock()

    def get_or_compute(self, key: TailPlanningCacheKey, compute: Callable[[], Optional[int]]) -> Optional[int]:
        with self._guard:
            if key in self._values:
                return self._values[key]
            lock = self._locks.get(key)
            if lock is None:
                lock = threading.Lock()
                self._locks[key] = lock
        with lock:
            with self._guard:
                if key in self._values:
                    return self._values[key]
            try:
                value = compute()
            except Exception:
                with self._guard:
                    if self._locks.get(key) is lock:
                        self._locks.pop(key, None)
                raise
            with self._guard:
                self._values[key] = value
                self._locks.pop(key, None)
            return value

    def __len__(self) -> int:
        with self._guard:
            return len(self._values)


def tail_planning_cache_key(
    *,
    namespace: str,
    out_root: Path,
    asset: str,
    interval_minutes: int,
    horizon_minutes: int,
    task: str,
    include_recompute: bool,
) -> TailPlanningCacheKey:
    return (
        str(namespace),
        str(Path(out_root).expanduser().resolve()),
        str(asset),
        int(interval_minutes),
        int(horizon_minutes),
        str(task),
        bool(include_recompute),
    )


def _cached_forecast_output_tail_ts(
    *,
    forecast_output_tail_ts_fn: Callable[..., Optional[int]],
    out_root: Path,
    interval_minutes: int,
    task: str,
    horizon_minutes: int,
    asset: str,
    include_recompute: bool,
    tail_cache: Optional[TailPlanningCache | MutableMapping[TailPlanningCacheKey, Optional[int]]],
    tail_cache_namespace: str,
) -> Optional[int]:
    kwargs: Dict[str, Any] = {
        "out_root": out_root,
        "interval_minutes": int(interval_minutes),
        "task": str(task),
        "horizon_minutes": int(horizon_minutes),
        "asset": str(asset),
    }
    if bool(include_recompute):
        kwargs["include_recompute"] = True
    if tail_cache is None:
        return forecast_output_tail_ts_fn(**kwargs)
    key = tail_planning_cache_key(
        namespace=str(tail_cache_namespace),
        out_root=Path(out_root),
        asset=str(asset),
        interval_minutes=int(interval_minutes),
        horizon_minutes=int(horizon_minutes),
        task=str(task),
        include_recompute=bool(include_recompute),
    )
    if isinstance(tail_cache, TailPlanningCache):
        return tail_cache.get_or_compute(key, lambda: forecast_output_tail_ts_fn(**kwargs))
    if key in tail_cache:
        return tail_cache[key]
    value = forecast_output_tail_ts_fn(**kwargs)
    tail_cache[key] = value
    return value


def planned_work_span_to_payload(span: PlannedAssetWorkSpan) -> Dict[str, Any]:
    return {
        "asset": str(span.asset),
        "edge_ts": int(span.edge_ts),
        "target_tail_ts": int(span.target_tail_ts),
        "start_ts": int(span.start_ts),
        "read_start_ts": int(span.read_start_ts),
        "dst_tail_ts": int(span.dst_tail_ts) if span.dst_tail_ts is not None else None,
    }


def planned_work_span_from_payload(payload: Dict[str, Any]) -> PlannedAssetWorkSpan:
    return PlannedAssetWorkSpan(
        asset=str(payload["asset"]),
        edge_ts=int(payload["edge_ts"]),
        target_tail_ts=int(payload["target_tail_ts"]),
        start_ts=int(payload["start_ts"]),
        read_start_ts=int(payload["read_start_ts"]),
        dst_tail_ts=(int(payload["dst_tail_ts"]) if payload.get("dst_tail_ts") is not None else None),
    )


def build_asset_shard_jobs(
    *,
    combo_key: Tuple[int, int, str],
    planned_assets: Sequence[str],
    worker_count: int,
    base_payload: Dict[str, Any],
    work_spans_by_asset: Optional[Dict[str, Any]] = None,
    work_span_payload_fn: Optional[Callable[[Any], Dict[str, Any]]] = None,
    include_public_shard_fields: bool = False,
) -> List[Dict[str, Any]]:
    shards = partition_assets(planned_assets, int(worker_count))
    jobs: List[Dict[str, Any]] = []
    span_map = dict(work_spans_by_asset or {})
    for shard_index, shard in enumerate(shards, start=1):
        work_spans: Dict[str, Any] = {}
        for asset_name in shard:
            span = span_map.get(str(asset_name))
            if span is None:
                work_spans[str(asset_name)] = None
            elif work_span_payload_fn is not None:
                work_spans[str(asset_name)] = work_span_payload_fn(span)
            elif isinstance(span, dict):
                work_spans[str(asset_name)] = dict(span)
            else:
                work_spans[str(asset_name)] = span
        job = {
            **dict(base_payload),
            "assets": list(shard),
            "work_spans": work_spans,
            "_combo_key": (int(combo_key[0]), int(combo_key[1]), str(combo_key[2])),
            "_shard_index": int(shard_index),
            "_shard_count": int(len(shards)),
        }
        if include_public_shard_fields:
            job["shard_index"] = int(shard_index)
            job["shard_count"] = int(len(shards))
        jobs.append(job)
    return jobs


def build_horizon_group_shard_jobs(
    *,
    combo_jobs: Sequence[Dict[str, Any]],
    worker_count: int,
) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[int, int], Dict[Tuple[int, int, str], Dict[str, Any]]] = {}
    for job in combo_jobs:
        combo_key = tuple(job.get("_combo_key", ()))
        if len(combo_key) != 3:
            continue
        interval, horizon_minutes, task = int(combo_key[0]), int(combo_key[1]), str(combo_key[2])
        entry = grouped.setdefault((interval, horizon_minutes), {}).setdefault(
            (interval, horizon_minutes, task),
            {"payload": None, "work_spans": {}, "assets": set()},
        )
        if entry["payload"] is None:
            payload = {key: value for key, value in dict(job).items() if not str(key).startswith("_")}
            payload.pop("assets", None)
            payload.pop("work_spans", None)
            entry["payload"] = payload
        for asset in job.get("assets", ()) or ():
            entry["assets"].add(str(asset))
        for asset, span in dict(job.get("work_spans") or {}).items():
            entry["work_spans"][str(asset)] = span

    group_jobs: List[Dict[str, Any]] = []
    for (interval, horizon_minutes), tasks in sorted(grouped.items(), key=lambda item: item[0]):
        group_assets = sorted({asset for entry in tasks.values() for asset in entry["assets"]})
        shards = partition_assets(group_assets, int(worker_count))
        for shard_index, shard in enumerate(shards, start=1):
            task_payloads: List[Dict[str, Any]] = []
            for combo_key, entry in sorted(tasks.items(), key=lambda item: item[0]):
                payload = dict(entry["payload"] or {})
                shard_spans = {}
                for asset in shard:
                    if str(asset) not in entry["work_spans"]:
                        continue
                    span = entry["work_spans"][str(asset)]
                    shard_spans[str(asset)] = dict(span) if isinstance(span, dict) else span
                if not shard_spans and not any(str(asset) in entry["assets"] for asset in shard):
                    continue
                payload["assets"] = list(shard)
                payload["work_spans"] = shard_spans
                task_payloads.append(payload)
            if not task_payloads:
                continue
            first = task_payloads[0]
            group_jobs.append(
                {
                    "run_id": str(first.get("run_id", "")),
                    "interval": int(interval),
                    "horizon_minutes": int(horizon_minutes),
                    "task": "__horizon_group__",
                    "assets": list(shard),
                    "task_payloads": task_payloads,
                    "parquet_root": str(first.get("parquet_root", "")),
                    "feature_root": str(first.get("feature_root", "")),
                    "_shard_index": int(shard_index),
                    "_shard_count": int(len(shards)),
                }
            )
    return group_jobs


def mark_prediction_row(
    row: Dict[str, Any],
    *,
    forward_filled: bool,
    needs_recompute: bool,
) -> Dict[str, Any]:
    out = dict(row)
    out["is_forward_filled"] = bool(forward_filled)
    out["needs_recompute"] = bool(needs_recompute)
    return out


def forward_fill_rows_to_edge(
    *,
    last_row: Optional[Dict[str, Any]],
    interval_minutes: int,
    edge_ts: int,
) -> List[Dict[str, Any]]:
    if not last_row or last_row.get("ts") is None:
        return []
    step = int(interval_minutes) * 60
    start_ts = int(last_row["ts"]) + int(step)
    if int(start_ts) > int(edge_ts):
        return []
    rows: List[Dict[str, Any]] = []
    for ts in range(int(start_ts), int(edge_ts) + 1, int(step)):
        row = dict(last_row)
        row["ts"] = int(ts)
        row["is_forward_filled"] = True
        row["needs_recompute"] = True
        rows.append(row)
    return rows


def prediction_eval_row(
    *,
    prediction_row: Dict[str, Any],
    actual_value: float,
    target_col: str,
) -> Dict[str, Any]:
    row = {
        "asset": str(prediction_row["asset"]),
        "ts": int(prediction_row["ts"]),
        "interval_min": int(prediction_row["interval_min"]),
        "horizon_min": int(prediction_row["horizon_min"]),
        "task": str(prediction_row["task"]),
        "run_id": str(prediction_row.get("run_id", "")),
        "model_id": str(prediction_row.get("model_id", "")),
        "model_version": str(prediction_row.get("model_version", "")),
        "target_col": str(target_col),
        "actual": float(actual_value),
        "is_forward_filled": bool(prediction_row.get("is_forward_filled", False)),
        "needs_recompute": bool(prediction_row.get("needs_recompute", False)),
    }
    for col in ("pred_p10", "pred_p50", "pred_p90"):
        if col in prediction_row:
            row[col] = float(prediction_row[col])
    pred_p50 = float(row.get("pred_p50", 0.0))
    err = pred_p50 - float(actual_value)
    row["error_p50"] = float(err)
    row["abs_error_p50"] = float(abs(err))
    row["squared_error_p50"] = float(err * err)
    return row


def plan_asset_work_span(
    *,
    asset: str,
    interval: int,
    horizon_minutes: int,
    task: str,
    mode: str,
    backfill_days: int,
    force: bool,
    fit_days: int,
    forecast_root: Path,
    discover_edge_and_min_fn: Callable[..., Tuple[Optional[int], Optional[int]]],
    forecast_output_tail_ts_fn: Callable[..., Optional[int]],
    decide_range_from_disk_edges_fn: Callable[..., Tuple[Optional[int], Any, str]],
    fit_window_start_fn: Callable[..., int],
    production_start_ts_value: Optional[int] = None,
    tail_cache: Optional[TailPlanningCache | MutableMapping[TailPlanningCacheKey, Optional[int]]] = None,
    tail_cache_namespace: str = "partitioned_forecast_output",
) -> Optional[PlannedAssetWorkSpan]:
    edge_ts, min_ts = discover_edge_and_min_fn(asset=str(asset), interval_minutes=int(interval))
    if edge_ts is None or min_ts is None:
        return None
    production_floor_ts = int(production_start_ts() if production_start_ts_value is None else production_start_ts_value)
    target_tail = int(edge_ts) - int(horizon_minutes) * 60
    effective_min_ts = max(int(min_ts), int(production_floor_ts))
    if int(target_tail) < int(effective_min_ts):
        return None
    dst_tail = None if bool(force) else _cached_forecast_output_tail_ts(
        forecast_output_tail_ts_fn=forecast_output_tail_ts_fn,
        out_root=forecast_root,
        interval_minutes=int(interval),
        task=str(task),
        horizon_minutes=int(horizon_minutes),
        asset=str(asset),
        include_recompute=False,
        tail_cache=tail_cache,
        tail_cache_namespace=str(tail_cache_namespace),
    )
    write_tail = None if bool(force) else _cached_forecast_output_tail_ts(
        forecast_output_tail_ts_fn=forecast_output_tail_ts_fn,
        out_root=forecast_root,
        interval_minutes=int(interval),
        task=str(task),
        horizon_minutes=int(horizon_minutes),
        asset=str(asset),
        include_recompute=True,
        tail_cache=tail_cache,
        tail_cache_namespace=str(tail_cache_namespace),
    )
    start_for_mode, _, resume_reason = decide_range_from_disk_edges_fn(
        asset=str(asset),
        interval_min=int(interval),
        downstream_max_ts=(int(dst_tail) if dst_tail is not None else None),
        upstream_min_ts=int(effective_min_ts),
        upstream_max_ts=int(target_tail),
        mode=str(mode),
        backfill_range=(max(int(effective_min_ts), int(target_tail) - int(backfill_days) * 86400), int(target_tail)) if str(mode) == "backfill" else None,
    )
    if resume_reason == "at_edge" and int(edge_ts) > int(target_tail):
        if write_tail is not None and int(write_tail) >= int(edge_ts):
            return None
        start_for_mode = int(target_tail)
        resume_reason = "edge_fill"
    if resume_reason in {"at_edge", "empty_backfill", "no_upstream", "no_upstream_head", "no_backfill_range"} or start_for_mode is None:
        return None
    start_for_mode = max(int(start_for_mode), int(effective_min_ts))
    if int(start_for_mode) > int(target_tail):
        return None
    read_start = fit_window_start_fn(edge_ts=int(start_for_mode), fit_days=int(fit_days), min_ts=int(min_ts))
    return PlannedAssetWorkSpan(
        asset=str(asset),
        edge_ts=int(edge_ts),
        target_tail_ts=int(target_tail),
        start_ts=int(start_for_mode),
        read_start_ts=int(read_start),
        dst_tail_ts=(int(dst_tail) if dst_tail is not None else None),
    )


def resolve_progress_every_seconds(env_name: str, *, default_seconds: int) -> int:
    raw = str(os.getenv(str(env_name), str(default_seconds))).strip()
    try:
        return max(5, int(raw))
    except Exception:
        return int(default_seconds)


def resolve_writer_retry_count(env_name: str = "NUMERIC_WRITER_RETRIES", *, default_retries: int = 3) -> int:
    raw = str(os.getenv(str(env_name), str(default_retries))).strip()
    try:
        return max(1, int(raw))
    except Exception:
        return int(default_retries)


def resolve_writer_retry_sleep_seconds(env_name: str = "NUMERIC_WRITER_RETRY_SECONDS", *, default_seconds: float = 1.0) -> float:
    raw = str(os.getenv(str(env_name), str(default_seconds))).strip()
    try:
        return max(0.1, float(raw))
    except Exception:
        return float(default_seconds)


def resolve_min_env_int(env_name: str, *, default_value: int, minimum: int) -> int:
    raw = str(os.getenv(str(env_name), str(default_value))).strip()
    try:
        return max(int(minimum), int(raw))
    except Exception:
        return int(default_value)


def spill_rows_chunk(
    *,
    rows: Sequence[Dict[str, Any]],
    staging_root: Path,
    module_tag: str,
    interval: int,
    horizon_minutes: int,
    task: str,
    asset: str,
    compression: str = "zstd",
) -> str:
    shard_rows_root = Path(staging_root) / "worker_row_batches"
    shard_rows_root.mkdir(parents=True, exist_ok=True)
    path = (
        shard_rows_root
        / f"{module_tag}_{int(interval)}_{int(horizon_minutes)}_{task}_{asset}_{uuid.uuid4().hex}.parquet"
    )
    pd.DataFrame(list(rows)).to_parquet(path, engine="pyarrow", compression=str(compression), index=False)
    return str(path)


def initial_partitioned_writer_state() -> Dict[str, Any]:
    return {"parts_by_combo": {}, "existing_key_cache": {}, "writer_stats": {}}


def start_partitioned_prediction_writer(
    *,
    module_tag: str,
    forecast_root: Path,
    writer_loop_fn: Callable[..., None],
) -> Tuple[Queue, Dict[str, Any], threading.Thread]:
    writer_queue: Queue = Queue()
    writer_state = initial_partitioned_writer_state()
    writer_thread = threading.Thread(
        target=writer_loop_fn,
        kwargs={"forecast_root": forecast_root, "write_queue": writer_queue, "writer_state": writer_state},
        name=f"{module_tag}_writer",
        daemon=True,
    )
    writer_thread.start()
    return writer_queue, writer_state, writer_thread


def partitioned_prediction_writer_loop(
    *,
    forecast_root: Path,
    write_queue: Any,
    writer_state: Dict[str, Any],
    write_partitioned_predictions_fn: Callable[..., List[Dict[str, Any]]],
    write_partitioned_prediction_month_frames_fn: Optional[Callable[..., List[Dict[str, Any]]]] = None,
) -> None:
    max_attempts = resolve_writer_retry_count()
    retry_sleep_seconds = resolve_writer_retry_sleep_seconds()

    def _assert_staged_rows_path_allowed(staged_rows_path: Path, action: str) -> None:
        assert_write_allowed(staged_rows_path, f"numeric writer staged rows {action}")

    def _load_staged_rows_frame(staged_rows_path: Path) -> pd.DataFrame:
        _assert_staged_rows_path_allowed(staged_rows_path, "read")
        suffix = str(staged_rows_path.suffix).lower()
        if suffix == ".parquet":
            return pd.read_parquet(staged_rows_path)
        return pd.read_json(staged_rows_path, orient="records", lines=True)

    def _unlink_staged_rows_path(staged_rows_path: Path) -> None:
        _assert_staged_rows_path_allowed(staged_rows_path, "delete")
        try:
            staged_rows_path.unlink()
        except Exception:
            pass

    def _drain_available_write_messages(first_message: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], bool]:
        messages = [first_message]
        saw_stop = False
        while True:
            try:
                next_message = write_queue.get_nowait()
            except Empty:
                break
            next_kind = str(next_message.get("kind"))
            if next_kind == "stop":
                saw_stop = True
                write_queue.task_done()
                break
            if next_kind in {"write_batch", "write_batch_file", "write_batch_files"}:
                messages.append(next_message)
            else:
                write_queue.task_done()
        return messages, saw_stop

    def _group_key(message: Dict[str, Any]) -> Optional[Tuple[Tuple[int, int, str], str, str, str]]:
        combo_key = tuple(message.get("combo_key", ()))
        if len(combo_key) != 3:
            return None
        interval, hm, task = combo_key
        return (
            (int(interval), int(hm), str(task)),
            str(message["run_id"]),
            str(message["module_tag"]),
            str(message.get("store", "forecast") or "forecast"),
        )

    def _append_frame_months(frame: pd.DataFrame, month_frames: Dict[Tuple[int, int], List[pd.DataFrame]]) -> None:
        if frame is None or frame.empty:
            return
        frame = frame.copy()
        frame["ts"] = pd.to_numeric(frame["ts"], errors="coerce")
        frame = frame.dropna(subset=["ts"]).copy()
        if frame.empty:
            return
        frame["ts"] = frame["ts"].astype("int64")
        dt = pd.to_datetime(frame["ts"], unit="s", utc=True)
        frame["year"] = dt.dt.year.astype(int)
        frame["month"] = dt.dt.month.astype(int)
        for (year, month), grp in frame.groupby(["year", "month"], sort=True):
            month_frames.setdefault((int(year), int(month)), []).append(
                grp.drop(columns=["year", "month"], errors="ignore").copy()
            )

    def _record_writer_success(
        *,
        grouped_messages: Sequence[Dict[str, Any]],
        staged_file_count: int,
        inline_row_count: int,
        parts: Sequence[Dict[str, Any]],
    ) -> None:
        stats = writer_state.setdefault("writer_stats", {})
        stats["messages_processed"] = int(stats.get("messages_processed", 0)) + int(len(grouped_messages))
        stats["coalesced_write_groups"] = int(stats.get("coalesced_write_groups", 0)) + 1
        stats["input_staged_files"] = int(stats.get("input_staged_files", 0)) + int(staged_file_count)
        stats["input_inline_rows"] = int(stats.get("input_inline_rows", 0)) + int(inline_row_count)
        stats["parts_written"] = int(stats.get("parts_written", 0)) + int(len(parts))
        writer_state["last_writer_success_at"] = time.time()

    while True:
        message = write_queue.get()
        messages_to_ack = [message]
        try:
            kind = str(message.get("kind"))
            if kind == "stop":
                return
            if kind not in {"write_batch", "write_batch_file", "write_batch_files"}:
                continue
            drained_messages, saw_stop = _drain_available_write_messages(message)
            messages_to_ack = list(drained_messages)
            grouped: Dict[Tuple[Tuple[int, int, str], str, str, str], List[Dict[str, Any]]] = {}
            for candidate in drained_messages:
                key = _group_key(candidate)
                if key is not None:
                    grouped.setdefault(key, []).append(candidate)
            for (combo_key, run_id, module_tag, store), grouped_messages in grouped.items():
                interval, hm, task = combo_key
                output_root = forecast_root if str(store) == "forecast" else Path(forecast_root) / str(store)
                last_error: Optional[Exception] = None
                for attempt in range(1, int(max_attempts) + 1):
                    staged_rows_paths: List[Path] = []
                    inline_frames: List[pd.DataFrame] = []
                    inline_row_count = 0
                    try:
                        for grouped_message in grouped_messages:
                            grouped_kind = str(grouped_message.get("kind"))
                            if grouped_kind == "write_batch":
                                rows = list(grouped_message.get("rows", []) or [])
                                if rows:
                                    inline_row_count += len(rows)
                                    inline_frames.append(pd.DataFrame(rows))
                            elif grouped_kind == "write_batch_file":
                                raw_path = str(grouped_message.get("staged_rows_path", "") or "").strip()
                                if raw_path:
                                    staged_rows_paths.append(Path(raw_path).resolve())
                            elif grouped_kind == "write_batch_files":
                                staged_rows_paths.extend(
                                    Path(str(path or "")).resolve()
                                    for path in list(grouped_message.get("staged_rows_paths", []) or [])
                                    if str(path or "").strip()
                                )
                        month_frames: Dict[Tuple[int, int], List[pd.DataFrame]] = {}
                        for staged_rows_path in staged_rows_paths:
                            _assert_staged_rows_path_allowed(staged_rows_path, "read/delete")
                            if not staged_rows_path.exists():
                                continue
                            frame = _load_staged_rows_frame(staged_rows_path)
                            if frame.empty:
                                _unlink_staged_rows_path(staged_rows_path)
                                continue
                            _append_frame_months(frame, month_frames)
                        for inline_frame in inline_frames:
                            _append_frame_months(inline_frame, month_frames)
                        if not month_frames:
                            for staged_rows_path in staged_rows_paths:
                                _unlink_staged_rows_path(staged_rows_path)
                            last_error = None
                            break
                        if write_partitioned_prediction_month_frames_fn is not None:
                            parts = write_partitioned_prediction_month_frames_fn(
                                out_root=output_root,
                                interval_minutes=int(interval),
                                run_id=str(run_id),
                                module_tag=str(module_tag),
                                task=str(task),
                                horizon_minutes=int(hm),
                                month_frames=month_frames,
                                existing_key_cache=writer_state.setdefault("existing_key_cache", {}),
                            )
                        else:
                            parts = write_partitioned_predictions_fn(
                                out_root=output_root,
                                interval_minutes=int(interval),
                                run_id=str(run_id),
                                module_tag=str(module_tag),
                                task=str(task),
                                horizon_minutes=int(hm),
                                df=pd.concat(
                                    [frame for frames in month_frames.values() for frame in frames],
                                    ignore_index=True,
                                ),
                                existing_key_cache=writer_state.setdefault("existing_key_cache", {}),
                            )
                        parts_by_store = writer_state.setdefault("parts_by_store", {})
                        parts_by_store.setdefault(str(store), {}).setdefault(combo_key, []).extend(parts)
                        if str(store) == "forecast":
                            writer_state.setdefault("parts_by_combo", {}).setdefault(combo_key, []).extend(parts)
                        for staged_rows_path in staged_rows_paths:
                            _unlink_staged_rows_path(staged_rows_path)
                        _record_writer_success(
                            grouped_messages=grouped_messages,
                            staged_file_count=len(staged_rows_paths),
                            inline_row_count=inline_row_count,
                            parts=parts,
                        )
                        last_error = None
                        break
                    except Exception as exc:
                        last_error = exc
                        writer_state["last_writer_retry"] = {
                            "combo_key": [int(interval), int(hm), str(task)],
                            "kind": "coalesced_write",
                            "messages": int(len(grouped_messages)),
                            "attempt": int(attempt),
                            "max_attempts": int(max_attempts),
                            "error": repr(exc),
                            "at_unix": float(time.time()),
                        }
                        if attempt >= int(max_attempts):
                            break
                        time.sleep(float(retry_sleep_seconds) * float(attempt))
                if last_error is not None:
                    raise last_error
            if saw_stop:
                return
        except Exception as exc:
            writer_state["__fatal__"] = exc
            return
        finally:
            for _message in messages_to_ack:
                write_queue.task_done()


def raise_writer_fatal(writer_state: Dict[str, Any], family_label: str) -> None:
    fatal = writer_state.get("__fatal__")
    if fatal is not None:
        raise RuntimeError(f"{family_label} writer failed: {fatal}") from fatal


def wait_for_writer_drain(*, writer_queue: Any, writer_thread: threading.Thread, writer_state: Dict[str, Any], family_label: str) -> None:
    while True:
        raise_writer_fatal(writer_state, family_label)
        unfinished = int(getattr(writer_queue, "unfinished_tasks", 0))
        if unfinished <= 0:
            return
        if not writer_thread.is_alive():
            raise RuntimeError(f"{family_label} writer stopped before draining queue; unfinished_tasks={unfinished}")
        time.sleep(0.5)


def validate_combo_completion(
    *,
    combo_key: Tuple[int, int, str],
    planned_assets: Sequence[str],
    combo_updates: Sequence[Tuple[str, Dict[str, Any]]],
    module_tag: str,
) -> None:
    planned_asset_set = {str(asset) for asset in planned_assets}
    if not planned_asset_set:
        return
    seen_assets: set[str] = set()
    actual_tails: Dict[str, int] = {}
    target_tails: Dict[str, int] = {}
    for ukey, upd in combo_updates:
        parts = str(ukey).split("|")
        asset = str(upd.get("asset") or (parts[4] if len(parts) >= 5 else "")).strip()
        if not asset:
            raise RuntimeError(f"[{module_tag}] combo validation failed combo={combo_key}: missing asset identity in unit update")
        seen_assets.add(asset)
        status = str(upd.get("status", "")).strip().lower()
        if status == "skipped":
            continue
        if status != "done":
            reason = str(upd.get("reason", status or "unknown"))
            raise RuntimeError(f"[{module_tag}] combo validation failed combo={combo_key} asset={asset}: status={reason}")
        meta = dict(upd.get("metadata") or {})
        rows = list(upd.get("rows", []) or [])
        row_count = int(meta.get("row_count", len(rows)) or 0)
        if row_count <= 0:
            raise RuntimeError(f"[{module_tag}] combo validation failed combo={combo_key} asset={asset}: no rows written for planned asset")
        if rows:
            row_ts = [int(row["ts"]) for row in rows if row.get("ts") is not None]
            if not row_ts:
                raise RuntimeError(f"[{module_tag}] combo validation failed combo={combo_key} asset={asset}: rows missing ts")
            actual_tails[asset] = int(max(row_ts))
        else:
            actual_tail = meta.get("actual_tail_ts")
            if actual_tail is None:
                raise RuntimeError(f"[{module_tag}] combo validation failed combo={combo_key} asset={asset}: missing actual_tail_ts")
            actual_tails[asset] = int(actual_tail)
        target_tail = meta.get("target_tail_ts")
        if target_tail is None:
            raise RuntimeError(f"[{module_tag}] combo validation failed combo={combo_key} asset={asset}: missing target_tail_ts")
        target_tails[asset] = int(target_tail)
        required_tail = int(meta.get("write_tail_ts", target_tail))
        if int(actual_tails[asset]) != int(required_tail):
            raise RuntimeError(
                f"[{module_tag}] combo validation failed combo={combo_key} asset={asset}: "
                f"actual_tail={int(actual_tails[asset])} required_tail={int(required_tail)} target_tail={int(target_tails[asset])}"
            )
    missing_assets = sorted(planned_asset_set.difference(seen_assets))
    if missing_assets:
        preview = ",".join(missing_assets[:10])
        raise RuntimeError(
            f"[{module_tag}] combo validation failed combo={combo_key}: "
            f"missing_updates_for_planned_assets={len(missing_assets)} sample={preview}"
        )
    if not actual_tails:
        return
    if len(set(actual_tails.values())) != 1:
        raise RuntimeError(
            f"[{module_tag}] combo validation failed combo={combo_key}: non-uniform actual tails "
            f"{sorted(set(int(v) for v in actual_tails.values()))[:10]}"
        )


def compute_market_factor_cache(
    *,
    parquet_root: Path,
    feature_root: Path,
    interval: int,
    horizon_minutes: int,
    task: str,
    assets: Sequence[str],
    start_ts: int,
    end_ts: int,
    target_col: str,
    read_feature_window_columns_fn: Callable[..., pd.DataFrame],
    overlay_runtime_target_labels_fn: Callable[..., pd.DataFrame],
    max_workers: int = 1,
    log_fn: Optional[Callable[[str], None]] = None,
    log_prefix: str = "",
) -> pd.DataFrame:
    started_at = time.monotonic()

    def _load_asset_component(asset_name: str) -> Optional[pd.DataFrame]:
        df = read_feature_window_columns_fn(
            root=feature_root,
            interval_minutes=int(interval),
            asset=str(asset_name),
            columns=[str(target_col)],
            start_ts=int(start_ts),
            end_ts=int(end_ts),
        )
        df = overlay_runtime_target_labels_fn(
            frame=df,
            parquet_root=parquet_root,
            asset=str(asset_name),
            interval=int(interval),
            horizon_minutes=int(horizon_minutes),
            target_col=str(target_col),
            start_ts=int(start_ts),
            end_ts=int(end_ts),
        )
        if df.empty or str(target_col) not in df.columns:
            return None
        d = df.loc[:, ["ts", str(target_col)]].copy()
        d["ts"] = pd.to_numeric(d["ts"], errors="coerce")
        d[str(target_col)] = pd.to_numeric(d[str(target_col)], errors="coerce")
        d = d.dropna(subset=["ts", str(target_col)])
        if d.empty:
            return None
        d["ts"] = d["ts"].astype("int64")
        return d.rename(columns={str(target_col): "market_factor_component"})

    ordered_assets = [str(asset) for asset in assets if str(asset)]
    frames: List[pd.DataFrame] = []
    resolved_workers = max(1, min(int(max_workers), len(ordered_assets))) if ordered_assets else 1
    if resolved_workers > 1 and len(ordered_assets) > 1:
        with ThreadPoolExecutor(max_workers=resolved_workers) as pool:
            loaded = list(pool.map(_load_asset_component, ordered_assets))
        frames = [frame for frame in loaded if frame is not None]
    else:
        for asset_name in ordered_assets:
            frame = _load_asset_component(asset_name)
            if frame is not None:
                frames.append(frame)

    if not frames:
        out = pd.DataFrame(columns=["ts", "market_factor"])
    else:
        all_df = pd.concat(frames, ignore_index=True)
        all_df["ts"] = pd.to_numeric(all_df["ts"], errors="coerce")
        all_df = all_df.dropna(subset=["ts", "market_factor_component"]).copy()
        if all_df.empty:
            out = pd.DataFrame(columns=["ts", "market_factor"])
        else:
            all_df["ts"] = all_df["ts"].astype("int64")
            out = (
                all_df.groupby("ts", as_index=False)["market_factor_component"]
                .mean()
                .rename(columns={"market_factor_component": "market_factor"})
                .sort_values("ts")
                .reset_index(drop=True)
            )
    if log_fn is not None:
        elapsed_s = time.monotonic() - started_at
        log_fn(
            f"{log_prefix}[factor-cache] k={int(interval)} h={int(horizon_minutes)}m task={task} "
            f"assets={len(ordered_assets)} rows={len(out)} elapsed_s={int(elapsed_s)} workers={int(resolved_workers)}"
        )
    return out


@dataclass(frozen=True)
class NumericTestedProductionArtifactScope:
    artifact_model_key: str
    handoff_path: Path
    feature_profile_json: Path
    cohort_assets: Tuple[str, ...]
    combo_specs: Tuple[Tuple[int, int, str], ...]
    combo_windows: Tuple[Tuple[int, int, str, int], ...]
    stage3_combo_results_path: Optional[Path]
    stage3_combo_specs: Tuple[Tuple[int, int, str], ...]
    tuned_params_by_combo: Dict[Tuple[int, int, str], Dict[str, Any]]


@dataclass(frozen=True)
class NumericExistingProductionScope:
    source_root: Path
    combo_specs: Tuple[Tuple[int, int, str], ...]


def resolve_model_state_root(forecast_root: Path, module_tag: str) -> Path:
    return (Path(forecast_root).resolve().parents[1] / "model_states" / str(module_tag)).resolve()


def unit_state_path(*, state_root: Path, asset: str, interval: int, horizon_minutes: int, task: str) -> Path:
    return (
        Path(state_root)
        / f"interval={int(interval)}"
        / f"horizon={int(horizon_minutes)}m"
        / f"task={str(task)}"
        / f"asset={str(asset)}.json"
    )


def load_unit_state(*, state_root: Path, asset: str, interval: int, horizon_minutes: int, task: str) -> Dict[str, Any]:
    path = unit_state_path(state_root=state_root, asset=asset, interval=interval, horizon_minutes=horizon_minutes, task=task)
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def save_unit_state(*, state_root: Path, asset: str, interval: int, horizon_minutes: int, task: str, state: Dict[str, Any]) -> None:
    path = unit_state_path(state_root=state_root, asset=asset, interval=interval, horizon_minutes=horizon_minutes, task=task)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(state)
    payload.update(
        {
            "version": 1,
            "asset": str(asset),
            "interval": int(interval),
            "horizon_minutes": int(horizon_minutes),
            "task": str(task),
        }
    )
    tmp = sibling_temp_path(path, suffix=".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    atomic_replace(tmp, path)


def staging_root(forecast_root: Path, module_tag: str) -> Path:
    root = Path(forecast_root) / "tmp" / f"{str(module_tag)}_stage"
    root.mkdir(parents=True, exist_ok=True)
    return root.resolve()


def artifact_model_key(model_id: str, *, prefix: str) -> str:
    raw = str(model_id).strip()
    return raw[len(prefix) :] if raw.startswith(prefix) else raw


def discover_numeric_tested_production_artifact_scope(
    *,
    project_root: Path,
    diagnostics_root_name: str,
    artifact_model_key: str,
    error_prefix: str,
    discover_tested_production_artifact_payload_fn: Callable[..., Optional[Dict[str, Any]]],
) -> Optional[NumericTestedProductionArtifactScope]:
    payload = discover_tested_production_artifact_payload_fn(
        project_root=project_root,
        diagnostics_root_name=diagnostics_root_name,
        artifact_model_key=artifact_model_key,
        error_prefix=error_prefix,
    )
    if payload is None:
        return None
    return NumericTestedProductionArtifactScope(
        artifact_model_key=artifact_model_key,
        handoff_path=payload["handoff_path"],
        feature_profile_json=payload["feature_profile_json"],
        cohort_assets=payload["cohort_assets"],
        combo_specs=payload["combo_specs"],
        combo_windows=payload["combo_windows"],
        stage3_combo_results_path=payload["stage3_combo_results_path"],
        stage3_combo_specs=payload["stage3_combo_specs"],
        tuned_params_by_combo=payload["tuned_params_by_combo"],
    )


def discover_numeric_existing_production_scope(
    *,
    manifest_path: Path,
    discover_existing_combo_specs_from_partitioned_output_fn: Callable[[Path], Tuple[Tuple[int, int, str], ...]],
) -> Optional[NumericExistingProductionScope]:
    forecast_root = manifest_path.parent
    combo_specs = discover_existing_combo_specs_from_partitioned_output_fn(forecast_root)
    if not combo_specs:
        return None
    return NumericExistingProductionScope(source_root=forecast_root.resolve(), combo_specs=combo_specs)


def build_numeric_family_manifest_payload(
    *,
    run_id: str,
    module_tag: str,
    model_id: str,
    model_version: str,
    family_name: str,
    combos: Sequence[Tuple[int, int, str]],
    combo_plans: Sequence[Dict[str, Any]],
    worker_budget: int,
    dispatch_slots: int,
    model_threads: int,
    mode: str,
    backfill_days: int,
    requested_fit_days: Optional[int],
    quantiles: Sequence[float],
    runtime_params: Dict[str, Any],
    requested_refit_cadence: str,
    predict_latest_only: bool,
    force: bool,
    overwrite_months: str,
    staging_root: Path,
    state_root: Path,
    tested_scope: Optional[NumericTestedProductionArtifactScope],
    production_scope: Optional[NumericExistingProductionScope],
    manifest_parts: Sequence[Dict[str, Any]],
    skipped_units: Dict[str, Dict[str, Any]],
    combo_count: int,
    job_shard_count: int,
    extra_fields: Optional[Dict[str, Any]] = None,
    finished_at: Optional[str] = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "run_id": run_id,
        "module": str(module_tag),
        "model_id": str(model_id),
        "model_version": str(model_version),
        "family": str(family_name),
        "domain": "Numerics",
        "combos": [{"interval": int(i), "horizon_minutes": int(h), "task": str(t)} for i, h, t in combos],
        "combo_plans": list(combo_plans),
        "intervals": sorted({int(i) for i, _, _ in combos}),
        "horizon_minutes": sorted({int(h) for _, h, _ in combos}),
        "tasks": sorted({str(t) for _, _, t in combos}),
        "workers": int(worker_budget),
        "dispatch_slots": int(dispatch_slots),
        "model_threads": int(model_threads),
        "worker_mode": "combo_job_process_shards",
        "writer_mode": "staged_combo_finalize",
        "staging_root": str(staging_root),
        "state_root": str(state_root),
        "pressure_guard": {
            "sample_interval_seconds": 5.0,
            "cpu_enter_pct": 92.0,
            "cpu_enter_samples": 3,
            "cpu_exit_pct": 80.0,
            "cpu_exit_samples": 3,
            "ram_enter_pct": 90.0,
            "ram_enter_samples": 2,
            "ram_exit_pct": 85.0,
            "ram_exit_samples": 3,
        },
        "job_shard_count": int(job_shard_count),
        "combo_count": int(combo_count),
        "mode": str(mode),
        "backfill_days": int(backfill_days),
        "fit_days": (int(requested_fit_days) if requested_fit_days is not None else None),
        "quantiles": [float(q) for q in quantiles],
        "runtime_params": dict(runtime_params),
        "refit_cadence": (normalize_refit_cadence(requested_refit_cadence) if requested_refit_cadence else None),
        "predict_latest_only": bool(predict_latest_only),
        "force": bool(force),
        "overwrite_months": str(overwrite_months),
        "default_run_profile": "production_profile_first",
        "tested_defaults": (
            {
                "artifact_model_key": tested_scope.artifact_model_key,
                "handoff_path": str(tested_scope.handoff_path),
                "feature_profile_json": str(tested_scope.feature_profile_json),
                "stage3_combo_results_path": (str(tested_scope.stage3_combo_results_path) if tested_scope.stage3_combo_results_path is not None else None),
                "combo_source": ("stage3" if tested_scope.stage3_combo_specs else "stage2"),
                "combo_windows": [f"{int(i)}:{int(h)}:{str(t)}@{int(m)}m" for i, h, t, m in tested_scope.combo_windows],
                "cohort_assets": list(tested_scope.cohort_assets),
            }
            if tested_scope is not None
            else None
        ),
        "production_scope_defaults": (
            {"source_root": str(production_scope.source_root), "combo_specs": [f"{int(i)}:{int(h)}:{str(t)}" for i, h, t in production_scope.combo_specs]}
            if production_scope is not None
            else None
        ),
    }
    if extra_fields:
        payload.update(dict(extra_fields))
    payload["parts"] = list(manifest_parts)
    payload["skipped_units"] = int(len(skipped_units))
    if finished_at is not None:
        payload["finished_at"] = str(finished_at)
    return payload


def execute_sharded_combo_jobs(
    *,
    combo_jobs: Sequence[Dict[str, Any]],
    combo_order: Sequence[Tuple[int, int, str]],
    dispatch_slots: int,
    module_name: str,
    module_tag: str,
    family_label: str,
    log_fn: Callable[[str], None],
    run_shard_fn: Callable[[Dict[str, Any]], List[Tuple[str, Dict[str, Any]]]],
    writer_queue: Any,
    writer_state: Dict[str, Any],
    make_unit_result_fn: Callable[[Dict[str, Any], Dict[str, Any]], Optional[Dict[str, Any]]],
    process_pool_init_retries: int,
    process_pool_init_retry_seconds: float,
    pressure_guard_factory: Callable[[str, Callable[[str], None]], Any],
    diagnostics_path: Optional[Path] = None,
    diagnostics_timestamp_fn: Optional[Callable[[], str]] = None,
) -> Tuple[Dict[Tuple[int, int, str], List[Tuple[str, Dict[str, Any]]]], List[Dict[str, Any]]]:
    updates_by_combo: Dict[Tuple[int, int, str], List[Tuple[str, Dict[str, Any]]]] = {combo_key: [] for combo_key in combo_order}
    unit_results: List[Dict[str, Any]] = []
    if not combo_jobs:
        return updates_by_combo, unit_results

    def _diag(event: str, payload: Dict[str, Any]) -> None:
        if diagnostics_path is None:
            return
        append_diagnostic_event(Path(diagnostics_path), event, payload, timestamp_fn=diagnostics_timestamp_fn)

    def _payload_diag_fields(payload: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "run_id": str(payload.get("run_id", "")),
            "module_tag": str(module_tag),
            "family_label": str(family_label),
            "interval": int(payload.get("interval", 0) or 0),
            "horizon_minutes": int(payload.get("horizon_minutes", 0) or 0),
            "task": str(payload.get("task", "")),
            "shard_index": int(payload.get("_shard_index", 0) or 0),
            "shard_count": int(payload.get("_shard_count", 0) or 0),
            "assets": int(len(payload.get("assets") or [])),
        }

    def _queue_update_rows(
        *,
        combo_key: Tuple[int, int, str],
        payload: Dict[str, Any],
        combo_updates: Sequence[Tuple[str, Dict[str, Any]]],
    ) -> Tuple[int, int, int, int]:
        staged_rows = [row for _ukey, upd in combo_updates for row in (upd.get("rows", []) if isinstance(upd.get("rows"), list) else [])]
        staged_row_files = [
            str(path)
            for _ukey, upd in combo_updates
            for path in (upd.get("staged_rows_paths", []) if isinstance(upd.get("staged_rows_paths"), list) else [])
            if str(path).strip()
        ]
        eval_rows = [row for _ukey, upd in combo_updates for row in (upd.get("eval_rows", []) if isinstance(upd.get("eval_rows"), list) else [])]
        eval_row_files = [
            str(path)
            for _ukey, upd in combo_updates
            for path in (upd.get("eval_staged_rows_paths", []) if isinstance(upd.get("eval_staged_rows_paths"), list) else [])
            if str(path).strip()
        ]
        if staged_rows:
            writer_queue.put({"kind": "write_batch", "combo_key": combo_key, "run_id": payload["run_id"], "module_tag": module_tag, "rows": staged_rows})
        if staged_row_files:
            writer_queue.put(
                {
                    "kind": "write_batch_files",
                    "combo_key": combo_key,
                    "run_id": payload["run_id"],
                    "module_tag": module_tag,
                    "staged_rows_paths": staged_row_files,
                }
            )
        if eval_rows:
            writer_queue.put({"kind": "write_batch", "store": "eval", "combo_key": combo_key, "run_id": payload["run_id"], "module_tag": module_tag, "rows": eval_rows})
        if eval_row_files:
            writer_queue.put(
                {
                    "kind": "write_batch_files",
                    "store": "eval",
                    "combo_key": combo_key,
                    "run_id": payload["run_id"],
                    "module_tag": module_tag,
                    "staged_rows_paths": eval_row_files,
                }
            )
        counts = (len(staged_rows), len(staged_row_files), len(eval_rows), len(eval_row_files))
        if any(counts):
            _diag(
                "writer_enqueue",
                {
                    **_payload_diag_fields(payload),
                    "forecast_rows": int(counts[0]),
                    "forecast_files": int(counts[1]),
                    "eval_rows": int(counts[2]),
                    "eval_files": int(counts[3]),
                    "queue_unfinished": int(getattr(writer_queue, "unfinished_tasks", 0)),
                },
            )
        return counts

    pressure_guard = pressure_guard_factory(
        module_name,
        lambda message: log_fn(f"[{module_tag}] {message}"),
    )
    pending_jobs = deque(combo_jobs)
    if dispatch_slots > 1:
        parallel_dispatch_done = False
        last_parallel_exc: Optional[Exception] = None
        completed_jobs_before_failure = 0
        for attempt in range(1, int(process_pool_init_retries) + 1):
            completed_jobs = 0
            try:
                with ProcessPoolExecutor(max_workers=dispatch_slots) as ex:
                    fut_map: Dict[Any, Dict[str, Any]] = {}
                    while pending_jobs or fut_map:
                        pressure_guard.refresh_if_due(force=not bool(fut_map))
                        while pending_jobs and len(fut_map) < int(dispatch_slots) and pressure_guard.should_admit_new_work(force_sample=False):
                            payload = pending_jobs.popleft()
                            log_fn(
                                f"[{module_tag}][dispatch] mode=parallel submit={int(completed_jobs) + len(fut_map) + 1}/{len(combo_jobs)} "
                                f"k={int(payload['interval'])} h={int(payload['horizon_minutes'])}m task={payload['task']} "
                                f"shard={int(payload['_shard_index'])}/{int(payload['_shard_count'])} assets={len(payload['assets'])}"
                            )
                            _diag(
                                "dispatch_submit",
                                {
                                    **_payload_diag_fields(payload),
                                    "mode": "parallel",
                                    "completed_jobs": int(completed_jobs),
                                    "inflight_jobs": int(len(fut_map)),
                                    "queued_jobs_remaining": int(len(pending_jobs)),
                                    "total_jobs": int(len(combo_jobs)),
                                    "writer_queue_unfinished": int(getattr(writer_queue, "unfinished_tasks", 0)),
                                    "resource": resource_snapshot(),
                                },
                            )
                            fut = ex.submit(run_shard_fn, {key: value for key, value in payload.items() if not str(key).startswith("_")})
                            payload["_submitted_monotonic"] = float(time.monotonic())
                            fut_map[fut] = payload
                        if fut_map:
                            done, _ = wait(set(fut_map.keys()), timeout=max(0.25, pressure_guard.seconds_until_next_sample()), return_when=FIRST_COMPLETED)
                            if not done:
                                pressure_guard.refresh_if_due(force=True)
                                continue
                            for fut in done:
                                payload = fut_map.pop(fut)
                                completed_jobs += 1
                                combo_key = payload["_combo_key"]
                                combo_updates = list(fut.result())
                                shard_elapsed_s = float(time.monotonic()) - float(payload.get("_submitted_monotonic", time.monotonic()))
                                updates_by_combo.setdefault(combo_key, []).extend(combo_updates)
                                raise_writer_fatal(writer_state, family_label)
                                staged_rows_count, staged_files_count, eval_rows_count, eval_files_count = _queue_update_rows(
                                    combo_key=combo_key,
                                    payload=payload,
                                    combo_updates=combo_updates,
                                )
                                for _ukey, upd in combo_updates:
                                    unit_result = make_unit_result_fn(payload, upd)
                                    if unit_result is not None:
                                        unit_results.append(unit_result)
                                log_fn(
                                    f"[{module_tag}][dispatch] mode=parallel completed={int(completed_jobs)}/{len(combo_jobs)} "
                                    f"k={int(payload['interval'])} h={int(payload['horizon_minutes'])}m task={payload['task']} "
                                    f"shard={int(payload['_shard_index'])}/{int(payload['_shard_count'])} "
                                    f"staged_rows={int(staged_rows_count)} staged_row_files={int(staged_files_count)} "
                                    f"eval_rows={int(eval_rows_count)} eval_row_files={int(eval_files_count)}"
                                )
                                done_units = int(sum(1 for _ukey, upd in combo_updates if str(upd.get("status", "")).lower() == "done"))
                                skipped_units_count = int(sum(1 for _ukey, upd in combo_updates if str(upd.get("status", "")).lower() == "skipped"))
                                _diag(
                                    "dispatch_complete",
                                    {
                                        **_payload_diag_fields(payload),
                                        "mode": "parallel",
                                        "completed_jobs": int(completed_jobs),
                                        "total_jobs": int(len(combo_jobs)),
                                        "updates": int(len(combo_updates)),
                                        "done_units": int(done_units),
                                        "skipped_units": int(skipped_units_count),
                                        "forecast_rows": int(staged_rows_count),
                                        "forecast_files": int(staged_files_count),
                                        "eval_rows": int(eval_rows_count),
                                        "eval_files": int(eval_files_count),
                                        "elapsed_s": round(float(shard_elapsed_s), 3),
                                        "writer_queue_unfinished": int(getattr(writer_queue, "unfinished_tasks", 0)),
                                        "writer_stats": dict(writer_state.get("writer_stats", {})),
                                        "resource": resource_snapshot(),
                                    },
                                )
                        elif pending_jobs:
                            pressure_guard.refresh_if_due(force=True)
                parallel_dispatch_done = True
                break
            except Exception as exc:
                last_parallel_exc = exc
                completed_jobs_before_failure = max(int(completed_jobs_before_failure), int(completed_jobs))
                if int(completed_jobs) > 0 or attempt >= int(process_pool_init_retries):
                    break
                log_fn(
                    f"[{module_tag}][runtime-retry] process pool init attempt={attempt}/{int(process_pool_init_retries)} "
                    f"failed; retrying in {float(process_pool_init_retry_seconds):.2f}s: {exc}"
                )
                time.sleep(float(process_pool_init_retry_seconds))
        if parallel_dispatch_done:
            return updates_by_combo, unit_results
        if last_parallel_exc is not None and int(completed_jobs_before_failure) > 0:
            raise RuntimeError(
                f"[{module_tag}] parallel shard execution failed after {int(completed_jobs_before_failure)} completed jobs: "
                f"{last_parallel_exc}"
            ) from last_parallel_exc
        log_fn(
            f"[{module_tag}][runtime-fallback] process pool unavailable after retries; forcing serial execution: {last_parallel_exc}"
        )
    for idx, payload in enumerate(combo_jobs, start=1):
        combo_key = payload["_combo_key"]
        log_fn(
            f"[{module_tag}][dispatch] mode=serial idx={idx}/{len(combo_jobs)} "
            f"k={int(payload['interval'])} h={int(payload['horizon_minutes'])}m task={payload['task']} "
            f"shard={int(payload['_shard_index'])}/{int(payload['_shard_count'])} assets={len(payload['assets'])}"
        )
        _diag(
            "dispatch_submit",
            {
                **_payload_diag_fields(payload),
                "mode": "serial",
                "completed_jobs": int(idx - 1),
                "inflight_jobs": 0,
                "queued_jobs_remaining": int(len(combo_jobs) - idx),
                "total_jobs": int(len(combo_jobs)),
                "writer_queue_unfinished": int(getattr(writer_queue, "unfinished_tasks", 0)),
                "resource": resource_snapshot(),
            },
        )
        shard_started = time.monotonic()
        combo_updates = run_shard_fn({key: value for key, value in payload.items() if not str(key).startswith("_")})
        shard_elapsed_s = float(time.monotonic()) - float(shard_started)
        updates_by_combo.setdefault(combo_key, []).extend(combo_updates)
        raise_writer_fatal(writer_state, family_label)
        staged_rows_count, staged_files_count, eval_rows_count, eval_files_count = _queue_update_rows(
            combo_key=combo_key,
            payload=payload,
            combo_updates=combo_updates,
        )
        for _ukey, upd in combo_updates:
            unit_result = make_unit_result_fn(payload, upd)
            if unit_result is not None:
                unit_results.append(unit_result)
        log_fn(
            f"[{module_tag}][dispatch] mode=serial completed={idx}/{len(combo_jobs)} "
            f"k={int(payload['interval'])} h={int(payload['horizon_minutes'])}m task={payload['task']} "
            f"shard={int(payload['_shard_index'])}/{int(payload['_shard_count'])} "
            f"staged_rows={int(staged_rows_count)} staged_row_files={int(staged_files_count)} "
            f"eval_rows={int(eval_rows_count)} eval_row_files={int(eval_files_count)}"
        )
        done_units = int(sum(1 for _ukey, upd in combo_updates if str(upd.get("status", "")).lower() == "done"))
        skipped_units_count = int(sum(1 for _ukey, upd in combo_updates if str(upd.get("status", "")).lower() == "skipped"))
        _diag(
            "dispatch_complete",
            {
                **_payload_diag_fields(payload),
                "mode": "serial",
                "completed_jobs": int(idx),
                "total_jobs": int(len(combo_jobs)),
                "updates": int(len(combo_updates)),
                "done_units": int(done_units),
                "skipped_units": int(skipped_units_count),
                "forecast_rows": int(staged_rows_count),
                "forecast_files": int(staged_files_count),
                "eval_rows": int(eval_rows_count),
                "eval_files": int(eval_files_count),
                "elapsed_s": round(float(shard_elapsed_s), 3),
                "writer_queue_unfinished": int(getattr(writer_queue, "unfinished_tasks", 0)),
                "writer_stats": dict(writer_state.get("writer_stats", {})),
                "resource": resource_snapshot(),
            },
        )
    return updates_by_combo, unit_results


def execute_grouped_horizon_jobs(
    *,
    group_jobs: Sequence[Dict[str, Any]],
    combo_order: Sequence[Tuple[int, int, str]],
    dispatch_slots: int,
    module_name: str,
    module_tag: str,
    family_label: str,
    log_fn: Callable[[str], None],
    run_group_shard_fn: Callable[[Dict[str, Any]], List[Tuple[Tuple[int, int, str], str, Dict[str, Any]]]],
    writer_queue: Any,
    writer_state: Dict[str, Any],
    make_unit_result_fn: Callable[[Dict[str, Any], Dict[str, Any]], Optional[Dict[str, Any]]],
    process_pool_init_retries: int,
    process_pool_init_retry_seconds: float,
    pressure_guard_factory: Callable[[str, Callable[[str], None]], Any],
    diagnostics_path: Optional[Path] = None,
    diagnostics_timestamp_fn: Optional[Callable[[], str]] = None,
) -> Tuple[Dict[Tuple[int, int, str], List[Tuple[str, Dict[str, Any]]]], List[Dict[str, Any]]]:
    updates_by_combo: Dict[Tuple[int, int, str], List[Tuple[str, Dict[str, Any]]]] = {combo_key: [] for combo_key in combo_order}
    unit_results: List[Dict[str, Any]] = []
    if not group_jobs:
        return updates_by_combo, unit_results

    def _diag(event: str, payload: Dict[str, Any]) -> None:
        if diagnostics_path is not None:
            append_diagnostic_event(Path(diagnostics_path), event, payload, timestamp_fn=diagnostics_timestamp_fn)

    def _payload_diag_fields(payload: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "run_id": str(payload.get("run_id", "")),
            "module_tag": str(module_tag),
            "family_label": str(family_label),
            "interval": int(payload.get("interval", 0) or 0),
            "horizon_minutes": int(payload.get("horizon_minutes", 0) or 0),
            "tasks": int(len(payload.get("task_payloads") or [])),
            "shard_index": int(payload.get("_shard_index", 0) or 0),
            "shard_count": int(payload.get("_shard_count", 0) or 0),
            "assets": int(len(payload.get("assets") or [])),
        }

    def _queue_updates(payload: Dict[str, Any], grouped_updates: Sequence[Tuple[Tuple[int, int, str], str, Dict[str, Any]]]) -> Tuple[int, int, int, int]:
        counts = [0, 0, 0, 0]
        by_combo: Dict[Tuple[int, int, str], List[Tuple[str, Dict[str, Any]]]] = {}
        for combo_key, ukey, upd in grouped_updates:
            key = (int(combo_key[0]), int(combo_key[1]), str(combo_key[2]))
            by_combo.setdefault(key, []).append((str(ukey), dict(upd)))
        for combo_key, combo_updates in by_combo.items():
            staged_rows = [row for _ukey, upd in combo_updates for row in (upd.get("rows", []) if isinstance(upd.get("rows"), list) else [])]
            staged_row_files = [
                str(path)
                for _ukey, upd in combo_updates
                for path in (upd.get("staged_rows_paths", []) if isinstance(upd.get("staged_rows_paths"), list) else [])
                if str(path).strip()
            ]
            eval_rows = [row for _ukey, upd in combo_updates for row in (upd.get("eval_rows", []) if isinstance(upd.get("eval_rows"), list) else [])]
            eval_row_files = [
                str(path)
                for _ukey, upd in combo_updates
                for path in (upd.get("eval_staged_rows_paths", []) if isinstance(upd.get("eval_staged_rows_paths"), list) else [])
                if str(path).strip()
            ]
            if staged_rows:
                writer_queue.put({"kind": "write_batch", "combo_key": combo_key, "run_id": payload["run_id"], "module_tag": module_tag, "rows": staged_rows})
            if staged_row_files:
                writer_queue.put({"kind": "write_batch_files", "combo_key": combo_key, "run_id": payload["run_id"], "module_tag": module_tag, "staged_rows_paths": staged_row_files})
            if eval_rows:
                writer_queue.put({"kind": "write_batch", "store": "eval", "combo_key": combo_key, "run_id": payload["run_id"], "module_tag": module_tag, "rows": eval_rows})
            if eval_row_files:
                writer_queue.put({"kind": "write_batch_files", "store": "eval", "combo_key": combo_key, "run_id": payload["run_id"], "module_tag": module_tag, "staged_rows_paths": eval_row_files})
            counts[0] += len(staged_rows)
            counts[1] += len(staged_row_files)
            counts[2] += len(eval_rows)
            counts[3] += len(eval_row_files)
        if any(counts):
            _diag(
                "writer_enqueue",
                {
                    **_payload_diag_fields(payload),
                    "forecast_rows": int(counts[0]),
                    "forecast_files": int(counts[1]),
                    "eval_rows": int(counts[2]),
                    "eval_files": int(counts[3]),
                    "queue_unfinished": int(getattr(writer_queue, "unfinished_tasks", 0)),
                },
            )
        return int(counts[0]), int(counts[1]), int(counts[2]), int(counts[3])

    def _record_group_result(payload: Dict[str, Any], grouped_updates: Sequence[Tuple[Tuple[int, int, str], str, Dict[str, Any]]]) -> None:
        for combo_key, ukey, upd in grouped_updates:
            key = (int(combo_key[0]), int(combo_key[1]), str(combo_key[2]))
            updates_by_combo.setdefault(key, []).append((str(ukey), dict(upd)))
            unit_payload = dict(payload)
            unit_payload["interval"] = int(key[0])
            unit_payload["horizon_minutes"] = int(key[1])
            unit_payload["task"] = str(key[2])
            unit_result = make_unit_result_fn(unit_payload, dict(upd))
            if unit_result is not None:
                unit_results.append(unit_result)

    pressure_guard = pressure_guard_factory(module_name, lambda message: log_fn(f"[{module_tag}] {message}"))
    pending_jobs = deque(group_jobs)
    if dispatch_slots > 1:
        last_parallel_exc: Optional[Exception] = None
        for attempt in range(1, int(process_pool_init_retries) + 1):
            completed_jobs = 0
            try:
                with ProcessPoolExecutor(max_workers=int(dispatch_slots)) as ex:
                    fut_map: Dict[Any, Dict[str, Any]] = {}
                    while pending_jobs or fut_map:
                        pressure_guard.refresh_if_due(force=not bool(fut_map))
                        while pending_jobs and len(fut_map) < int(dispatch_slots) and pressure_guard.should_admit_new_work(force_sample=False):
                            payload = pending_jobs.popleft()
                            log_fn(
                                f"[{module_tag}][dispatch] mode=parallel submit={int(completed_jobs) + len(fut_map) + 1}/{len(group_jobs)} "
                                f"k={int(payload['interval'])} h={int(payload['horizon_minutes'])}m tasks={len(payload.get('task_payloads') or [])} "
                                f"shard={int(payload['_shard_index'])}/{int(payload['_shard_count'])} assets={len(payload['assets'])}"
                            )
                            _diag("dispatch_submit", {**_payload_diag_fields(payload), "mode": "parallel", "total_jobs": int(len(group_jobs)), "resource": resource_snapshot()})
                            fut = ex.submit(run_group_shard_fn, {key: value for key, value in payload.items() if not str(key).startswith("_")})
                            payload["_submitted_monotonic"] = float(time.monotonic())
                            fut_map[fut] = payload
                        if fut_map:
                            done, _ = wait(set(fut_map.keys()), timeout=max(0.25, pressure_guard.seconds_until_next_sample()), return_when=FIRST_COMPLETED)
                            if not done:
                                pressure_guard.refresh_if_due(force=True)
                                continue
                            for fut in done:
                                payload = fut_map.pop(fut)
                                completed_jobs += 1
                                grouped_updates = list(fut.result())
                                _record_group_result(payload, grouped_updates)
                                raise_writer_fatal(writer_state, family_label)
                                counts = _queue_updates(payload, grouped_updates)
                                elapsed_s = float(time.monotonic()) - float(payload.get("_submitted_monotonic", time.monotonic()))
                                log_fn(
                                    f"[{module_tag}][dispatch] mode=parallel completed={int(completed_jobs)}/{len(group_jobs)} "
                                    f"k={int(payload['interval'])} h={int(payload['horizon_minutes'])}m tasks={len(payload.get('task_payloads') or [])} "
                                    f"shard={int(payload['_shard_index'])}/{int(payload['_shard_count'])} "
                                    f"staged_rows={counts[0]} staged_row_files={counts[1]} eval_rows={counts[2]} eval_row_files={counts[3]}"
                                )
                                _diag("dispatch_complete", {**_payload_diag_fields(payload), "mode": "parallel", "updates": int(len(grouped_updates)), "elapsed_s": round(float(elapsed_s), 3), "resource": resource_snapshot()})
                return updates_by_combo, unit_results
            except Exception as exc:
                last_parallel_exc = exc
                if attempt >= int(process_pool_init_retries):
                    break
                log_fn(f"[{module_tag}][runtime-retry] grouped process pool attempt={attempt}/{int(process_pool_init_retries)} failed; retrying in {float(process_pool_init_retry_seconds):.2f}s: {exc}")
                time.sleep(float(process_pool_init_retry_seconds))
        log_fn(f"[{module_tag}][runtime-fallback] grouped process pool unavailable; forcing serial execution: {last_parallel_exc}")
    for idx, payload in enumerate(group_jobs, start=1):
        log_fn(
            f"[{module_tag}][dispatch] mode=serial idx={idx}/{len(group_jobs)} "
            f"k={int(payload['interval'])} h={int(payload['horizon_minutes'])}m tasks={len(payload.get('task_payloads') or [])} "
            f"shard={int(payload['_shard_index'])}/{int(payload['_shard_count'])} assets={len(payload['assets'])}"
        )
        _diag("dispatch_submit", {**_payload_diag_fields(payload), "mode": "serial", "total_jobs": int(len(group_jobs)), "resource": resource_snapshot()})
        started = time.monotonic()
        grouped_updates = list(run_group_shard_fn({key: value for key, value in payload.items() if not str(key).startswith("_")}))
        _record_group_result(payload, grouped_updates)
        raise_writer_fatal(writer_state, family_label)
        counts = _queue_updates(payload, grouped_updates)
        elapsed_s = float(time.monotonic()) - float(started)
        log_fn(
            f"[{module_tag}][dispatch] mode=serial completed={idx}/{len(group_jobs)} "
            f"k={int(payload['interval'])} h={int(payload['horizon_minutes'])}m tasks={len(payload.get('task_payloads') or [])} "
            f"shard={int(payload['_shard_index'])}/{int(payload['_shard_count'])} "
            f"staged_rows={counts[0]} staged_row_files={counts[1]} eval_rows={counts[2]} eval_row_files={counts[3]}"
        )
        _diag("dispatch_complete", {**_payload_diag_fields(payload), "mode": "serial", "updates": int(len(grouped_updates)), "elapsed_s": round(float(elapsed_s), 3), "resource": resource_snapshot()})
    return updates_by_combo, unit_results


def combo_window_map(scope: Optional[Any]) -> Dict[Tuple[int, int, str], int]:
    if scope is None:
        return {}
    return {
        (int(interval), int(horizon), str(task)): int(training_window_months)
        for interval, horizon, task, training_window_months in scope.combo_windows
    }


def resolve_combo_fit_days(*, requested_fit_days: Optional[int], runtime_params: Dict[str, Any], tested_training_window_months: Optional[int], default_fit_days_fn: Callable[[Dict[str, Any]], int]) -> int:
    if requested_fit_days is not None:
        return max(1, int(requested_fit_days))
    if tested_training_window_months is not None and int(tested_training_window_months) > 0:
        return max(31, int(tested_training_window_months) * 31)
    return int(default_fit_days_fn(runtime_params))


def resolve_runner_model_threads(worker_count: int, configured_threads: int) -> int:
    return max(1, int(configured_threads))


def resolve_dispatch_slots(worker_count: int, queued_jobs: int) -> int:
    return max(1, min(max(1, int(worker_count)), max(1, int(queued_jobs)), 14))


def default_refit_cadence_for_interval(interval: int) -> str:
    if int(interval) <= 60:
        return "weekly"
    if int(interval) <= 240:
        return "biweekly"
    return "monthly"


def parse_best_params(raw: Any) -> Dict[str, Any]:
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return dict(raw)
    if not str(raw).strip():
        return {}
    try:
        parsed = ast.literal_eval(raw)
    except Exception:
        return {}
    return dict(parsed) if isinstance(parsed, dict) else {}


def load_stage3_combo_results(combo_results_path: Path) -> Tuple[Tuple[Tuple[int, int, str], ...], Dict[Tuple[int, int, str], Dict[str, Any]]]:
    if not combo_results_path.exists():
        return tuple(), {}
    df = pd.read_csv(combo_results_path)
    combos: List[Tuple[int, int, str]] = []
    tuned_params: Dict[Tuple[int, int, str], Dict[str, Any]] = {}
    for row in df.to_dict(orient="records"):
        try:
            combo = (int(row.get("interval")), int(row.get("horizon_minutes")), str(row.get("task")))
        except Exception:
            continue
        if combo[0] <= 0 or combo[1] <= 0 or not combo[2]:
            continue
        status = str(row.get("status", "") or "").strip().lower()
        if status == "ineligible":
            continue
        combos.append(combo)
        baseline_rmse = row.get("baseline_rmse")
        tuned_rmse = row.get("tuned_rmse")
        if pd.isna(baseline_rmse) or pd.isna(tuned_rmse):
            continue
        try:
            baseline_rmse_f = float(baseline_rmse)
            tuned_rmse_f = float(tuned_rmse)
        except Exception:
            continue
        if tuned_rmse_f <= baseline_rmse_f:
            best_params = parse_best_params(row.get("best_params"))
            if best_params:
                tuned_params[combo] = best_params
    return tuple(sorted(set(combos), key=lambda item: (item[0], item[1], item[2]))), tuned_params


def json_load_dict(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected JSON object in artifact file: {path}")
    return payload


_PARTITIONED_OUTPUT_RE = re.compile(r"^part-.+?_\d+-\d{6}-(?P<task>.+)-h(?P<horizon>\d+)m-.+\.parquet$", re.IGNORECASE)


def discover_existing_combo_specs_from_partitioned_output(out_root: Path) -> Tuple[Tuple[int, int, str], ...]:
    combos: set[Tuple[int, int, str]] = set()
    root = Path(out_root)
    if not root.exists():
        return tuple()
    for interval_dir in sorted(root.iterdir()):
        if not interval_dir.is_dir():
            continue
        try:
            interval = int(interval_dir.name)
        except Exception:
            continue
        for path in interval_dir.glob("year=*/month=*/*.parquet"):
            match = _PARTITIONED_OUTPUT_RE.match(path.name)
            if match is None:
                continue
            task = str(match.group("task"))
            horizon = int(match.group("horizon"))
            combos.add((int(interval), int(horizon), str(task)))
    return tuple(sorted(combos, key=lambda item: (item[0], item[1], item[2])))


def discover_existing_combo_windows_from_state_tree(
    *,
    state_root: Path,
    infer_training_window_months_fn: Callable[[int, int, str, Optional[int]], int],
    load_state_fn: Callable[[str, int, int, str], Dict[str, Any]],
) -> Tuple[Tuple[int, int, str, int], ...]:
    combos: set[Tuple[int, int, str, int]] = set()
    root = Path(state_root)
    if not root.exists():
        return tuple()
    for interval_dir in sorted(root.glob("interval=*")):
        try:
            interval = int(str(interval_dir.name).split("=", 1)[1])
        except Exception:
            continue
        for horizon_dir in sorted(interval_dir.glob("horizon=*m")):
            try:
                horizon_minutes = int(str(horizon_dir.name).split("=", 1)[1].rstrip("m"))
            except Exception:
                continue
            for task_dir in sorted(horizon_dir.glob("task=*")):
                task = str(task_dir.name).split("=", 1)[1]
                state_files = sorted(task_dir.glob("asset=*.pkl"))
                if not state_files:
                    continue
                sample_asset = str(state_files[0].stem).split("=", 1)[1]
                state = load_state_fn(str(sample_asset), int(interval), int(horizon_minutes), str(task)) or {}
                try:
                    selected_window_bars = int(state.get("selected_window_bars")) if state.get("selected_window_bars") is not None else None
                except Exception:
                    selected_window_bars = None
                combos.add(
                    (
                        int(interval),
                        int(horizon_minutes),
                        str(task),
                        int(infer_training_window_months_fn(int(interval), int(horizon_minutes), str(task), selected_window_bars)),
                    )
                )
    return tuple(sorted(combos, key=lambda item: (item[0], item[1], item[2], item[3])))


def discover_tested_production_artifact_payload(
    *,
    project_root: Path,
    diagnostics_root_name: str,
    artifact_model_key: str,
    error_prefix: str,
) -> Optional[Dict[str, Any]]:
    handoff_paths = sorted((project_root / "logs" / "diagnostics" / diagnostics_root_name).glob(f"run=*/{artifact_model_key}/stage2/run=*/stage3_survivor_handoff.json"))
    if not handoff_paths:
        return None
    handoff_path = handoff_paths[-1].resolve()
    payload = json_load_dict(handoff_path)
    feature_profile_raw = str(payload.get("feature_profile_json") or "").strip()
    if not feature_profile_raw:
        raise SystemExit(f"{error_prefix} tested production handoff exists but does not declare a Stage 1 feature profile artifact: {handoff_path}")
    feature_profile_json = Path(feature_profile_raw)
    feature_profile_json = (project_root / feature_profile_json).resolve() if not feature_profile_json.is_absolute() else feature_profile_json.resolve()
    if not feature_profile_json.exists():
        raise SystemExit(f"{error_prefix} tested production handoff exists but referenced Stage 1 feature profile artifact is missing: {feature_profile_json}")
    survivors = payload.get("survivors")
    if not isinstance(survivors, list) or not survivors:
        raise RuntimeError(f"Tested production handoff has no survivors: {handoff_path}")
    combo_specs = tuple(sorted({(int(row["interval_minutes"]), int(row["horizon_minutes"]), str(row["task"])) for row in survivors if isinstance(row, dict)}, key=lambda item: (item[0], item[1], item[2])))
    combo_windows = tuple(
        sorted(
            {
                (int(row["interval_minutes"]), int(row["horizon_minutes"]), str(row["task"]), int(row.get("training_window_months") or 0))
                for row in survivors
                if isinstance(row, dict)
            },
            key=lambda item: (item[0], item[1], item[2], item[3]),
        )
    )
    cohort_assets = tuple(sorted({str(asset) for asset in (payload.get("cohort_assets") or []) if str(asset).strip()}))
    stage3_combo_results_path = (handoff_path.parents[2] / "stage3" / "combo_results.csv").resolve()
    stage3_combo_specs, tuned_params_by_combo = load_stage3_combo_results(stage3_combo_results_path)
    return {
        "handoff_path": handoff_path,
        "feature_profile_json": feature_profile_json,
        "cohort_assets": cohort_assets,
        "combo_specs": combo_specs,
        "combo_windows": combo_windows,
        "stage3_combo_results_path": stage3_combo_results_path if stage3_combo_results_path.exists() else None,
        "stage3_combo_specs": stage3_combo_specs,
        "tuned_params_by_combo": tuned_params_by_combo,
    }
