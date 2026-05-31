from __future__ import annotations

import argparse
import ast
import importlib
import hashlib
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd
import psutil
import pyarrow.parquet as pq
from src.forecasting.common.path_config import resolve_path, selected_profile
from src.forecasting.ml.shared.numeric_cohort_common import (
    CLAMP_START_MONTH,
    CLAMP_START_YEAR,
    MonthKey,
    add_months,
    asset_months,
    common_recent_window,
    list_assets,
    months_between,
    month_seq,
    parse_asset_name,
)
from src.forecasting.ml.tabular.shared.tabular_diagnostic_analysis import analyze_manifest_for_model
from src.forecasting.ml.tabular.shared.tabular_numeric_model_registry import (
    get_tabular_numeric_model_spec,
    load_model_task_metadata,
    resolve_default_or_requested_combos as resolve_model_default_or_requested_combos,
    resolve_default_tasks,
    write_runtime_config_for_model,
)
from src.forecasting.ml.shared.test_branch_function_telemetry import emit_event_for_path
from src.forecasting.ml.shared.numeric_runner_diagnostics import emit_standard_numeric_diagnostic_packet


DEFAULT_MODEL_KEY = "xgboost"
CURRENT_MODEL_SPEC = get_tabular_numeric_model_spec(DEFAULT_MODEL_KEY)
DEFAULT_OUTPUT_DIR = CURRENT_MODEL_SPEC.diagnostics_output_dir
DEFAULT_TASKS = ""
DEFAULT_HORIZONS = ""
DEFAULT_TRAIN_WINDOWS = "1,2,3,4,5,6,7,8,9,10,11,12"
DEFAULT_FORECAST_DAYS = 30.0
DEFAULT_INTERVALS = ""
DEFAULT_ASSET_COUNT = 8
DEFAULT_SEARCH_BACK_MONTHS = 12
DEFAULT_SAMPLE_INTERVAL = 0.5
CLAMP_START_YEAR = 2025
CLAMP_START_MONTH = 1
MAX_LOGICAL_THREADS = 64
RUNTIME_PROFILES: Tuple[Tuple[int, int], ...] = ((8, 6),)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def month_label(months: Optional[int]) -> str:
    return "FULL" if months is None else f"{int(months)}m"


def month_start_utc_ts(year: int, month: int) -> int:
    return int(datetime(year, month, 1, 0, 0, 0, tzinfo=timezone.utc).timestamp())


def months_to_bars(months: int, interval_minutes: int = 1) -> int:
    return int(int(months) * 30 * 24 * 60 // int(interval_minutes))


def trailing_source_start_ts(seed_ts: int, interval_minutes: int, train_window_bars: int, max_horizon_bars: int) -> int:
    step_seconds = int(interval_minutes) * 60
    lookback_bars = max(1, int(train_window_bars) + int(max_horizon_bars) - 1)
    return int(seed_ts) - (int(lookback_bars) * int(step_seconds))


def parse_csv_str(raw: str) -> List[str]:
    return [x.strip() for x in str(raw).split(",") if x.strip()]


def parse_csv_int(raw: str) -> List[int]:
    out: List[int] = []
    for token in parse_csv_str(raw):
        try:
            value = int(token)
        except Exception:
            continue
        if value > 0:
            out.append(value)
    return sorted(set(out))


def parse_train_window_tokens(raw: str) -> List[Optional[int]]:
    out: List[Optional[int]] = []
    seen: set[object] = set()
    for token in parse_csv_str(raw):
        upper = str(token).upper()
        if upper == "FULL":
            key: object = "FULL"
            value: Optional[int] = None
        else:
            try:
                parsed = int(token)
            except Exception:
                continue
            if parsed <= 0:
                continue
            key = int(parsed)
            value = int(parsed)
        if key in seen:
            continue
        seen.add(key)
        out.append(value)
    if not out:
        raise ValueError(f"Invalid training windows: {raw}")
    return out


def combos_from_feature_profile_json(
    feature_profile_json: Path,
    *,
    supported_tasks: Sequence[str],
    requested_intervals: Sequence[int],
    requested_tasks: Sequence[str],
    requested_horizons: Sequence[int],
) -> List[Tuple[int, int, str]]:
    payload = json.loads(feature_profile_json.read_text(encoding="utf-8"))
    selections = payload.get("selections") or {}
    combos: List[Tuple[int, int, str]] = []
    allowed_tasks = {str(task) for task in supported_tasks}
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
        if task not in allowed_tasks:
            continue
        if interval_filter and interval not in interval_filter:
            continue
        if task_filter and task not in task_filter:
            continue
        if horizon_filter and horizon not in horizon_filter:
            continue
        combos.append((int(interval), int(horizon), str(task)))
    return sorted(set(combos), key=lambda item: (int(item[0]), int(item[1]), str(item[2])))


def cohort_assets_from_feature_profile_json(feature_profile_json: Path) -> Tuple[List[str], Dict[str, str]]:
    payload = json.loads(feature_profile_json.read_text(encoding="utf-8"))
    assets = [str(asset) for asset in (payload.get("cohort_assets") or []) if str(asset)]
    aliases = {str(key): str(value) for key, value in dict(payload.get("cohort_asset_aliases") or {}).items()}
    return assets, aliases


def resolve_default_or_requested_combos(
    *,
    requested_intervals: Sequence[int],
    requested_tasks: Sequence[str],
    requested_horizons: Sequence[int],
    runnable_tasks: Sequence[str],
) -> List[Tuple[int, int, str]]:
    return resolve_model_default_or_requested_combos(
        CURRENT_MODEL_SPEC,
        requested_intervals=requested_intervals,
        requested_tasks=requested_tasks,
        requested_horizons=requested_horizons,
        runnable_tasks=runnable_tasks,
    )


def runtime_profiles(max_logical_threads: int = MAX_LOGICAL_THREADS) -> List[Tuple[int, int]]:
    profiles: List[Tuple[int, int]] = []
    for workers, threads in RUNTIME_PROFILES:
        if int(workers) > int(max_logical_threads) or int(threads) > int(max_logical_threads):
            raise RuntimeError(
                f"Configured runtime profile {int(workers)}x{int(threads)} exceeds "
                f"max_logical_threads={int(max_logical_threads)}"
            )
        profiles.append((int(workers), int(threads)))
    return profiles


def runtime_profile_name(workers: int, threads: int) -> str:
    return f"{int(workers)}x{int(threads)}"


def load_module_task_metadata(project_root: Path) -> Tuple[List[str], Dict[str, str], Dict[str, str]]:
    return load_model_task_metadata(CURRENT_MODEL_SPEC, project_root)


def choose_assets(eligible_assets: Sequence[str], seed: int, asset_count: int) -> Tuple[List[str], Dict[str, str]]:
    assets = sorted({str(asset) for asset in eligible_assets})
    if not assets:
        return [], {}
    if int(asset_count) <= 0 or int(asset_count) >= len(assets):
        return assets, {}
    rng = random.Random(int(seed))
    sampled = list(assets)
    rng.shuffle(sampled)
    return sorted(sampled[: int(asset_count)]), {}


def max_asset_ts_from_monthly_table(table_root: Path, asset: str) -> Optional[int]:
    base = table_root / f"asset={asset}"
    if not base.exists():
        return None
    max_ts: Optional[int] = None
    for ydir in base.glob("year=*"):
        if not ydir.is_dir():
            continue
        for mdir in ydir.glob("month=*"):
            if not mdir.is_dir():
                continue
            for parquet_path in mdir.glob("*.parquet"):
                try:
                    frame = pd.read_parquet(parquet_path, columns=["ts"])
                except Exception:
                    continue
                if frame.empty or "ts" not in frame.columns:
                    continue
                values = pd.to_numeric(frame["ts"], errors="coerce").dropna()
                if values.empty:
                    continue
                current = int(values.astype("int64").max())
                if max_ts is None or current > max_ts:
                    max_ts = current
    return max_ts


def min_asset_ts_from_monthly_table(table_root: Path, asset: str) -> Optional[int]:
    base = table_root / f"asset={asset}"
    if not base.exists():
        return None
    min_ts: Optional[int] = None
    for ydir in base.glob("year=*"):
        if not ydir.is_dir():
            continue
        for mdir in ydir.glob("month=*"):
            if not mdir.is_dir():
                continue
            for parquet_path in mdir.glob("*.parquet"):
                try:
                    frame = pd.read_parquet(parquet_path, columns=["ts"])
                except Exception:
                    continue
                if frame.empty or "ts" not in frame.columns:
                    continue
                values = pd.to_numeric(frame["ts"], errors="coerce").dropna()
                if values.empty:
                    continue
                current = int(values.astype("int64").min())
                if min_ts is None or current < min_ts:
                    min_ts = current
    return min_ts


def max_asset_ts_for_month(table_root: Path, asset: str, month: MonthKey) -> Optional[int]:
    base = table_root / f"asset={asset}" / f"year={month.year}" / f"month={month.month:02d}"
    if not base.exists():
        return None
    max_ts: Optional[int] = None
    for parquet_path in base.glob("*.parquet"):
        try:
            frame = pd.read_parquet(parquet_path, columns=["ts"])
        except Exception:
            continue
        if frame.empty or "ts" not in frame.columns:
            continue
        values = pd.to_numeric(frame["ts"], errors="coerce").dropna()
        if values.empty:
            continue
        current = int(values.astype("int64").max())
        if max_ts is None or current > max_ts:
            max_ts = current
    return max_ts


def resolve_complete_observation_month(
    *,
    selected_assets: Sequence[str],
    ohlc_root: Path,
    start_month: MonthKey,
    max_backtrack_months: int,
    interval_minutes: int,
) -> Tuple[MonthKey, int, int]:
    bar_seconds = int(interval_minutes) * 60
    candidate = MonthKey(start_month.year, start_month.month)
    for back in range(max(1, int(max_backtrack_months))):
        month = add_months(candidate.year, candidate.month, -back)
        edges: List[int] = []
        complete = True
        for asset in selected_assets:
            edge = max_asset_ts_for_month(ohlc_root, asset, month)
            if edge is None:
                complete = False
                break
            edges.append(int(edge))
        if not complete or not edges:
            continue
        next_month = add_months(month.year, month.month, 1)
        expected_end_ts = int(month_start_utc_ts(next_month.year, next_month.month) - bar_seconds)
        actual_end_ts = int(min(edges))
        if actual_end_ts >= expected_end_ts:
            return month, expected_end_ts, actual_end_ts
    raise RuntimeError(
        "Could not resolve a fully complete fixed observation month across the selected cohort. "
        f"start_month={start_month.year:04d}-{start_month.month:02d} max_backtrack_months={int(max_backtrack_months)} interval={int(interval_minutes)}m"
    )


def write_runtime_config(runtime_cfg_path: Path, original_text: str, model_threads: int) -> None:
    write_runtime_config_for_model(CURRENT_MODEL_SPEC, runtime_cfg_path, original_text, model_threads)


def diagnostic_entry_command(python_exe: str) -> List[str]:
    if CURRENT_MODEL_SPEC.root_script_name:
        return [str(python_exe), str(CURRENT_MODEL_SPEC.root_script_name)]
    return [str(python_exe), "-m", str(CURRENT_MODEL_SPEC.module_import_path)]


def requested_pairs(
    tasks: Sequence[str],
    horizon_minutes: Sequence[int],
    task_short: Dict[str, str],
    allowed_pairs: Optional[Sequence[Tuple[str, int]]] = None,
) -> List[Tuple[str, int, str]]:
    out: List[Tuple[str, int, str]] = []
    allowed = {(str(task), int(horizon)) for task, horizon in allowed_pairs} if allowed_pairs is not None else None
    for task in tasks:
        short = task_short.get(str(task))
        if not short:
            continue
        for horizon in horizon_minutes:
            if allowed is not None and (str(task), int(horizon)) not in allowed:
                continue
            out.append((str(task), int(horizon), str(short)))
    return out


def seed_forecast_edge(
    parquet_out_root: Path,
    assets: Sequence[str],
    interval_minutes: int,
    seed_ts: int,
    tasks: Sequence[str],
    horizon_minutes: Sequence[int],
    task_short: Dict[str, str],
    allowed_pairs: Optional[Sequence[Tuple[str, int]]] = None,
) -> List[str]:
    table_root = parquet_out_root / f"{CURRENT_MODEL_SPEC.forecast_table_tag}_{int(interval_minutes)}"
    dt = datetime.fromtimestamp(seed_ts, tz=timezone.utc)
    row: Dict[str, object] = {"ts": int(seed_ts)}
    pairs: List[str] = []
    for task, horizon, short in requested_pairs(tasks, horizon_minutes, task_short, allowed_pairs=allowed_pairs):
        pairs.append(f"{task}:{int(horizon)}m")
        row[f"{CURRENT_MODEL_SPEC.prediction_prefix}_pred_mean_{short}_{int(horizon)}m"] = 0.0
        row[f"{CURRENT_MODEL_SPEC.prediction_prefix}_pred_std_{short}_{int(horizon)}m"] = 0.0
        row[f"{CURRENT_MODEL_SPEC.prediction_prefix}_pred_p10_{short}_{int(horizon)}m"] = 0.0
        row[f"{CURRENT_MODEL_SPEC.prediction_prefix}_pred_p90_{short}_{int(horizon)}m"] = 0.0
    for asset in assets:
        dst_dir = table_root / f"asset={asset}" / f"year={dt.year}" / f"month={dt.month:02d}"
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / "part-000.parquet"
        asset_row = dict(row)
        asset_row["asset"] = str(asset)
        pd.DataFrame([asset_row]).to_parquet(dst, engine="pyarrow", compression="snappy", index=False)
    return sorted(set(pairs))


def validate_seeded_forecast_edge(
    parquet_out_root: Path,
    assets: Sequence[str],
    interval_minutes: int,
    seed_ts: int,
    tasks: Sequence[str],
    horizon_minutes: Sequence[int],
    task_short: Dict[str, str],
    allowed_pairs: Optional[Sequence[Tuple[str, int]]] = None,
) -> Tuple[List[str], Dict[str, List[str]]]:
    dt = datetime.fromtimestamp(seed_ts, tz=timezone.utc)
    table_root = parquet_out_root / f"{CURRENT_MODEL_SPEC.forecast_table_tag}_{int(interval_minutes)}"
    requested = requested_pairs(tasks, horizon_minutes, task_short, allowed_pairs=allowed_pairs)
    required_cols: List[str] = []
    for _, horizon, short in requested:
        required_cols.extend(
            [
                f"{CURRENT_MODEL_SPEC.prediction_prefix}_pred_mean_{short}_{int(horizon)}m",
                f"{CURRENT_MODEL_SPEC.prediction_prefix}_pred_std_{short}_{int(horizon)}m",
                f"{CURRENT_MODEL_SPEC.prediction_prefix}_pred_p10_{short}_{int(horizon)}m",
                f"{CURRENT_MODEL_SPEC.prediction_prefix}_pred_p90_{short}_{int(horizon)}m",
            ]
        )
    missing_by_asset: Dict[str, List[str]] = {}
    for asset in assets:
        parquet_path = table_root / f"asset={asset}" / f"year={dt.year}" / f"month={dt.month:02d}" / "part-000.parquet"
        if not parquet_path.exists():
            missing_by_asset[str(asset)] = list(required_cols)
            continue
        try:
            frame = pd.read_parquet(parquet_path)
        except Exception:
            missing_by_asset[str(asset)] = list(required_cols)
            continue
        existing = set(str(col) for col in frame.columns)
        missing = [col for col in required_cols if col not in existing]
        if missing:
            missing_by_asset[str(asset)] = missing
    return sorted({f"{task}:{int(horizon)}m" for task, horizon, _ in requested}), missing_by_asset


def apply_nested_module_sandbox_roots(env: Dict[str, str], module_root: Path) -> None:
    if str(env.get("PIPELINE_SANDBOX_MODE", "") or "").strip().lower() not in {"1", "true", "yes", "on"}:
        return
    env["PIPELINE_SANDBOX_OUTPUT_ROOT"] = str(module_root)
    env["PIPELINE_SANDBOX_PARQUET_ROOT"] = str(module_root / "parquet")
    env["PIPELINE_SANDBOX_LOG_ROOT"] = str(module_root / "logs")
    env["PIPELINE_SANDBOX_STATE_ROOT"] = str(module_root / "model_states")
    env["PIPELINE_SANDBOX_TMP_ROOT"] = str(module_root / "tmp")
    env["PIPELINE_SANDBOX_CATBOOST_TRAIN_DIR"] = str(module_root / "tmp" / "catboost_train")


def parse_log_ts(line: str) -> Optional[float]:
    match = re.match(r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) UTC\]", line)
    if not match:
        return None
    try:
        return datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc).timestamp()
    except Exception:
        return None


def unit_key(asset: str, interval: int, horizon_minutes: int, task: str) -> str:
    return f"{asset}|{int(interval)}|{int(horizon_minutes)}|{task}"


def parse_module_log(module_log_path: Path) -> dict:
    out: dict = {
        "module_plan_line": None,
        "module_run_span_s": None,
        "run_start_ts": None,
        "run_end_ts": None,
        "dispatch_mode": None,
        "work_items": None,
        "unit_ranges": {},
        "unit_diag": {},
        "diag_aggregate": {},
        "max_parallel_active": 1,
        "multiple_units_active": False,
        "unit_start_ts": {},
        "unit_end_ts": {},
    }
    if not module_log_path.exists():
        return out
    lines = module_log_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    prefix = re.escape(CURRENT_MODEL_SPEC.progress_log_prefix)
    re_plan = re.compile(rf"{prefix}\[plan\].*total_units=(\d+)")
    re_dispatch = re.compile(rf"{prefix}\[dispatch\] mode=(serial|parallel)")
    re_unit_start = re.compile(rf"{prefix}\[unit-start\] asset=([^\s]+) k=(\d+) h=(\d+)m task=([^\s]+) work=\[(\d+),(\d+)\]")
    re_diag = re.compile(rf"{prefix}\[diag-summary\] asset=([^\s]+) k=(\d+) h=(\d+)m task=([^\s]+) (\{{.*\}})")
    re_done = re.compile(rf"{prefix} run complete")
    run_start_ts: Optional[float] = None
    run_end_ts: Optional[float] = None
    unit_start_ts: Dict[str, float] = {}
    unit_end_ts: Dict[str, float] = {}
    agg = defaultdict(float)
    for line in lines:
        ts = parse_log_ts(line)
        plan_match = re_plan.search(line)
        if plan_match:
            out["module_plan_line"] = line
            out["work_items"] = int(plan_match.group(1))
            if ts is not None and run_start_ts is None:
                run_start_ts = ts
        dispatch_match = re_dispatch.search(line)
        if dispatch_match:
            out["dispatch_mode"] = dispatch_match.group(1)
        start_match = re_unit_start.search(line)
        if start_match:
            key = unit_key(start_match.group(1), int(start_match.group(2)), int(start_match.group(3)), start_match.group(4))
            out["unit_ranges"][key] = [int(start_match.group(5)), int(start_match.group(6))]
            if ts is not None:
                unit_start_ts[key] = ts
        diag_match = re_diag.search(line)
        if diag_match:
            key = unit_key(diag_match.group(1), int(diag_match.group(2)), int(diag_match.group(3)), diag_match.group(4))
            try:
                diag = json.loads(diag_match.group(5))
            except Exception:
                continue
            out["unit_diag"][key] = {
                "asset": diag_match.group(1),
                "interval": int(diag_match.group(2)),
                "horizon_minutes": int(diag_match.group(3)),
                "task": diag_match.group(4),
                **diag,
            }
            for key_name, value in diag.items():
                if isinstance(value, (int, float)):
                    agg[key_name] += float(value)
            if ts is not None:
                unit_end_ts[key] = ts
        if re_done.search(line) and ts is not None:
            run_end_ts = ts
    if run_start_ts is not None and run_end_ts is not None:
        out["module_run_span_s"] = float(run_end_ts - run_start_ts)
    out["run_start_ts"] = float(run_start_ts) if run_start_ts is not None else None
    out["run_end_ts"] = float(run_end_ts) if run_end_ts is not None else None
    out["unit_start_ts"] = {key: float(value) for key, value in unit_start_ts.items()}
    out["unit_end_ts"] = {key: float(value) for key, value in unit_end_ts.items()}
    if agg:
        out["diag_aggregate"] = dict(agg)
    events: List[Tuple[float, int]] = []
    for key_name, start_ts in unit_start_ts.items():
        end_ts = unit_end_ts.get(key_name)
        if end_ts is None:
            continue
        events.append((start_ts, 1))
        events.append((end_ts, -1))
    if events:
        events.sort(key=lambda item: (item[0], -item[1]))
        current = 0
        maximum = 0
        for _, delta in events:
            current += delta
            if current > maximum:
                maximum = current
        out["max_parallel_active"] = int(maximum)
        out["multiple_units_active"] = bool(maximum >= 2)
    return out


def avg(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def stddev(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    mean = avg(values)
    return float(math.sqrt(sum((float(value) - mean) ** 2 for value in values) / len(values)))


def overlap_s(start_a: float, end_a: float, start_b: float, end_b: float) -> float:
    return max(0.0, min(float(end_a), float(end_b)) - max(float(start_a), float(start_b)))


def weighted_avg(samples: Sequence[Tuple[float, float]]) -> float:
    total_weight = sum(max(0.0, float(weight)) for _, weight in samples)
    if total_weight <= 0.0:
        return 0.0
    return float(sum(float(value) * max(0.0, float(weight)) for value, weight in samples) / total_weight)


def avg_by_horizon(unit_diag: Dict[str, dict], value_key: str) -> Dict[str, float]:
    buckets: Dict[str, List[float]] = defaultdict(list)
    for diag in unit_diag.values():
        horizon = diag.get("horizon_minutes")
        value = diag.get(value_key)
        if horizon is None or value is None:
            continue
        try:
            buckets[f"{int(horizon)}m"].append(float(value))
        except Exception:
            continue
    return {key: avg(values) for key, values in sorted(buckets.items(), key=lambda item: int(item[0].rstrip("m")))}


def append_log_lines(log_state: Dict[str, Any], lines: Sequence[str]) -> None:
    prefix = re.escape(CURRENT_MODEL_SPEC.progress_log_prefix)
    re_plan = re.compile(rf"{prefix}\[plan\].*total_units=(\d+)")
    re_unit_start = re.compile(rf"{prefix}\[unit-start\] asset=([^\s]+) k=(\d+) h=(\d+)m task=([^\s]+) work=\[(\d+),(\d+)\]")
    re_diag = re.compile(rf"{prefix}\[diag-summary\] asset=([^\s]+) k=(\d+) h=(\d+)m task=([^\s]+) (\{{.*\}})")
    re_done = re.compile(rf"{prefix} run complete")
    active_units: Dict[str, float] = log_state.setdefault("active_units", {})
    unit_start_ts: Dict[str, float] = log_state.setdefault("unit_start_ts", {})
    unit_end_ts: Dict[str, float] = log_state.setdefault("unit_end_ts", {})
    completed_units: set[str] = log_state.setdefault("completed_units", set())
    for line in lines:
        ts = parse_log_ts(line)
        plan_match = re_plan.search(line)
        if plan_match:
            log_state["work_items"] = int(plan_match.group(1))
            if ts is not None and log_state.get("run_start_ts") is None:
                log_state["run_start_ts"] = float(ts)
        start_match = re_unit_start.search(line)
        if start_match:
            key = unit_key(start_match.group(1), int(start_match.group(2)), int(start_match.group(3)), start_match.group(4))
            active_units[key] = float(ts) if ts is not None else float(log_state.get("elapsed_fallback_s", 0.0))
            if ts is not None:
                unit_start_ts[key] = float(ts)
        diag_match = re_diag.search(line)
        if diag_match:
            key = unit_key(diag_match.group(1), int(diag_match.group(2)), int(diag_match.group(3)), diag_match.group(4))
            active_units.pop(key, None)
            completed_units.add(key)
            if ts is not None:
                unit_end_ts[key] = float(ts)
        if re_done.search(line) and ts is not None:
            log_state["run_end_ts"] = float(ts)


def consume_module_log_delta(module_log: Path, log_state: Dict[str, Any], elapsed_fallback_s: float) -> None:
    log_state["elapsed_fallback_s"] = float(elapsed_fallback_s)
    if not module_log.exists():
        return
    with module_log.open("r", encoding="utf-8", errors="ignore") as handle:
        handle.seek(int(log_state.get("offset", 0)))
        chunk = handle.read()
        log_state["offset"] = int(handle.tell())
    text = f"{str(log_state.get('partial', ''))}{chunk}"
    if not text:
        return
    lines = text.splitlines(keepends=True)
    complete_lines: List[str] = []
    partial = ""
    for line in lines:
        if line.endswith("\n") or line.endswith("\r"):
            complete_lines.append(line.rstrip("\r\n"))
        else:
            partial = line
    log_state["partial"] = partial
    if complete_lines:
        append_log_lines(log_state, complete_lines)


def worker_state_snapshot(log_state: Dict[str, Any], worker_slots: int) -> Dict[str, int]:
    total_work_items = int(log_state.get("work_items") or 0)
    active_units = int(len(log_state.get("active_units", {})))
    completed_units = int(len(log_state.get("completed_units", set())))
    queued_work_items = max(0, total_work_items - completed_units - active_units)
    idle_worker_slots = max(0, int(worker_slots) - active_units)
    return {
        "queued_work_items": int(queued_work_items),
        "active_units": int(active_units),
        "idle_worker_slots": int(idle_worker_slots),
    }


def build_tail_report(
    *,
    wall_clock_s: float,
    unit_diag: Dict[str, dict],
    unit_start_ts: Dict[str, float],
    unit_end_ts: Dict[str, float],
    sample_records: Sequence[dict],
) -> Dict[str, Any]:
    if wall_clock_s <= 0.0:
        return {
            "distinct_horizons_alive_in_final_20pct": [],
            "avg_active_units_final_20pct": 0.0,
            "slowest_10_units": [],
        }
    final_window_start = float(wall_clock_s) * 0.8
    slowest_units = sorted(
        (
            {
                "asset": str(diag.get("asset")),
                "task": str(diag.get("task")),
                "horizon_minutes": int(diag.get("horizon_minutes", 0) or 0),
                "total_wall_s": float(diag.get("total_wall_s", 0.0) or 0.0),
            }
            for diag in unit_diag.values()
        ),
        key=lambda item: float(item["total_wall_s"]),
        reverse=True,
    )[:10]
    horizons_alive: set[str] = set()
    if unit_start_ts and unit_end_ts:
        origin = min(unit_start_ts.values())
        final_abs_start = origin + final_window_start
        final_abs_end = origin + float(wall_clock_s)
        for key_name, start_ts in unit_start_ts.items():
            end_ts = unit_end_ts.get(key_name)
            if end_ts is None or overlap_s(float(start_ts), float(end_ts), final_abs_start, final_abs_end) <= 0.0:
                continue
            diag = unit_diag.get(key_name) or {}
            horizon = diag.get("horizon_minutes")
            if horizon is not None:
                horizons_alive.add(f"{int(horizon)}m")
    final_samples: List[Tuple[float, float]] = []
    for record in sample_records:
        sample_start = float(record.get("sample_start_s", 0.0))
        sample_end = float(record.get("sample_end_s", 0.0))
        overlap = overlap_s(sample_start, sample_end, final_window_start, float(wall_clock_s))
        if overlap > 0.0:
            final_samples.append((float(record.get("active_units", 0.0) or 0.0), overlap))
    return {
        "distinct_horizons_alive_in_final_20pct": sorted(horizons_alive, key=lambda item: int(item.rstrip("m"))),
        "avg_active_units_final_20pct": weighted_avg(final_samples),
        "slowest_10_units": slowest_units,
    }


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(int(chunk_size))
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def parquet_probe(path: Path) -> Dict[str, Any]:
    probe: Dict[str, Any] = {
        "path": str(path),
        "exists": bool(path.exists()),
    }
    if not path.exists():
        return probe
    try:
        stat = path.stat()
        probe["size_bytes"] = int(stat.st_size)
        probe["mtime_utc"] = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat()
        probe["sha256"] = sha256_file(path)
    except Exception as exc:
        probe["stat_error"] = str(exc)
    try:
        pq_file = pq.ParquetFile(path)
        metadata = pq_file.metadata
        probe["parquet_readable"] = True
        probe["row_groups"] = int(metadata.num_row_groups) if metadata is not None else None
        probe["rows"] = int(metadata.num_rows) if metadata is not None else None
        probe["schema_columns"] = list(metadata.schema.names) if metadata is not None else None
    except Exception as exc:
        probe["parquet_readable"] = False
        probe["parquet_error"] = str(exc)
    return probe


def sibling_parquet_snapshot(path: Path, limit: int = 12) -> List[Dict[str, Any]]:
    parent = path.parent
    if not parent.exists():
        return []
    rows: List[Dict[str, Any]] = []
    for sibling in sorted(parent.glob("*.parquet")):
        try:
            stat = sibling.stat()
            rows.append(
                {
                    "name": sibling.name,
                    "size_bytes": int(stat.st_size),
                    "mtime_utc": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
                }
            )
        except Exception as exc:
            rows.append({"name": sibling.name, "error": str(exc)})
    rows.sort(key=lambda item: str(item.get("mtime_utc", "")), reverse=True)
    return rows[: int(limit)]


def recent_parquet_writes(root: Path, limit: int = 20) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not root.exists():
        return rows
    for path in root.rglob("*.parquet"):
        try:
            stat = path.stat()
            rows.append(
                {
                    "path": str(path),
                    "size_bytes": int(stat.st_size),
                    "mtime_utc": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
                }
            )
        except Exception as exc:
            rows.append({"path": str(path), "error": str(exc)})
    rows.sort(key=lambda item: str(item.get("mtime_utc", "")), reverse=True)
    return rows[: int(limit)]


def parse_schema_error_details(log_text: str) -> Optional[Dict[str, Any]]:
    match = re.search(
        rf"{re.escape(CURRENT_MODEL_SPEC.progress_log_prefix)}\[schema-error\]\s+unreadable\s+(forecast|eval)\s+parquet\s+for\s+asset=([^\s]+)\s+k=(\d+)\s+h=(\d+)m\s+task=([^\s]+)\s+path=([^\r\n:]+):\s*(.*)",
        log_text,
        flags=re.DOTALL,
    )
    if not match:
        return None
    return {
        "table_kind": str(match.group(1)),
        "asset": str(match.group(2)),
        "interval": int(match.group(3)),
        "horizon_minutes": int(match.group(4)),
        "task": str(match.group(5)),
        "path": str(match.group(6)).strip(),
        "error": str(match.group(7)).strip(),
    }


def build_failure_forensics(*, combined_log: Path, module_root: Path) -> Dict[str, Any]:
    forensic: Dict[str, Any] = {}
    try:
        log_text = combined_log.read_text(encoding="utf-8", errors="ignore")
    except Exception as exc:
        return {"log_read_error": str(exc)}
    schema_error = parse_schema_error_details(log_text)
    if schema_error is None:
        return {
            "schema_error": None,
            "recent_parquet_writes": recent_parquet_writes(module_root / "parquet"),
        }
    failing_path = Path(str(schema_error["path"]))
    forensic["schema_error"] = schema_error
    forensic["failing_parquet"] = parquet_probe(failing_path)
    forensic["sibling_parquet_files"] = sibling_parquet_snapshot(failing_path)
    forensic["recent_parquet_writes"] = recent_parquet_writes(module_root / "parquet")
    return forensic


def update_dwell(
    *,
    dwell_seconds: Dict[str, float],
    continuous_seconds: Dict[str, float],
    max_continuous_seconds: Dict[str, float],
    elapsed_s: float,
    value: float,
    total_prefix: str,
    thresholds: Sequence[int],
    continuous_threshold: int,
) -> None:
    for threshold in thresholds:
        key = f"{total_prefix}_ge_{int(threshold)}_s"
        if float(value) >= float(threshold):
            dwell_seconds[key] = float(dwell_seconds.get(key, 0.0) + float(elapsed_s))
        if int(threshold) == int(continuous_threshold):
            continuous_key = f"max_continuous_{total_prefix}_ge_{int(threshold)}_s"
            if float(value) >= float(threshold):
                continuous_seconds[continuous_key] = float(continuous_seconds.get(continuous_key, 0.0) + float(elapsed_s))
                max_continuous_seconds[continuous_key] = max(
                    float(max_continuous_seconds.get(continuous_key, 0.0)),
                    float(continuous_seconds[continuous_key]),
                )
            else:
                continuous_seconds[continuous_key] = 0.0


def _read_table_asset_range(
    table_root: Path,
    asset: str,
    start_exclusive_ts: int,
    end_inclusive_ts: Optional[int],
    columns: Sequence[str],
) -> pd.DataFrame:
    base = table_root / f"asset={asset}"
    if not base.exists():
        return pd.DataFrame(columns=list(columns))
    frames: List[pd.DataFrame] = []
    for year_dir in sorted(base.glob("year=*")):
        if not year_dir.is_dir():
            continue
        for month_dir in sorted(year_dir.glob("month=*")):
            if not month_dir.is_dir():
                continue
            for parquet_path in sorted(month_dir.glob("*.parquet")):
                read_cols = list(columns)
                if read_cols:
                    try:
                        available = set(pq.ParquetFile(parquet_path).schema.names)
                        read_cols = [col for col in read_cols if col in available]
                    except Exception:
                        pass
                    if "ts" not in read_cols:
                        read_cols = ["ts"] + read_cols
                    if "asset" not in read_cols:
                        read_cols = ["asset"] + read_cols
                try:
                    frame = pd.read_parquet(parquet_path, columns=read_cols)
                except Exception:
                    continue
                if frame.empty or "ts" not in frame.columns:
                    continue
                ts = pd.to_numeric(frame["ts"], errors="coerce")
                mask = ts.notna() & (ts.astype("int64") > int(start_exclusive_ts))
                if end_inclusive_ts is not None:
                    mask = mask & (ts.astype("int64") <= int(end_inclusive_ts))
                sliced = frame[mask].copy()
                if not sliced.empty:
                    sliced["ts"] = pd.to_numeric(sliced["ts"], errors="coerce").astype("int64")
                    frames.append(sliced)
    if not frames:
        return pd.DataFrame(columns=list(columns))
    out = pd.concat(frames, ignore_index=True)
    return out.sort_values("ts").drop_duplicates(subset=["asset", "ts"], keep="last")


def baseline_metrics_from_series(y: pd.Series) -> dict:
    yv = pd.to_numeric(y, errors="coerce").dropna().astype(float).reset_index(drop=True)
    if len(yv) >= 2:
        baseline = yv.shift(1)
        baseline_type = "persistence"
    elif len(yv) == 1:
        baseline = pd.Series([float(yv.mean())] * len(yv), dtype=float)
        baseline_type = "constant_mean"
    else:
        return {
            "baseline_type": None,
            "baseline_mae": None,
            "baseline_rmse": None,
            "y_std": None,
        }
    compare = pd.DataFrame({"y": yv, "baseline": baseline}).dropna()
    if compare.empty:
        return {
            "baseline_type": baseline_type,
            "baseline_mae": None,
            "baseline_rmse": None,
            "y_std": float(yv.std(ddof=0)) if len(yv) else None,
        }
    err = compare["baseline"] - compare["y"]
    return {
        "baseline_type": baseline_type,
        "baseline_mae": float(err.abs().mean()),
        "baseline_rmse": float(math.sqrt(float((err.pow(2.0)).mean()))),
        "y_std": float(yv.std(ddof=0)),
    }


def accuracy_from_frame(
    *,
    df: pd.DataFrame,
    y_col: str,
    pred_col: str,
    dir_col: Optional[str],
) -> dict:
    if df.empty:
        return {
            "forecast_count": 0,
            "baseline_type": None,
            "baseline_mae": None,
            "baseline_rmse": None,
            "y_std": None,
        }
    y_true = pd.to_numeric(df[y_col], errors="coerce").astype(float)
    y_pred = pd.to_numeric(df[pred_col], errors="coerce").astype(float)
    err = y_pred - y_true
    out = {
        "forecast_count": int(len(df)),
        "mae": float(err.abs().mean()),
        "rmse": float(math.sqrt(float((err * err).mean()))),
        **baseline_metrics_from_series(y_true),
    }
    if dir_col and dir_col in df.columns:
        true_dir = pd.to_numeric(df[dir_col], errors="coerce").fillna(0).astype(int)
        pred_dir = y_pred.apply(lambda v: 1 if v > 0.0001 else (-1 if v < -0.0001 else 0)).astype(int)
        out["directional_accuracy"] = float((pred_dir == true_dir).mean())
    return out


def compute_accuracy_metrics(
    *,
    run_parquet_root: Path,
    assets: Sequence[str],
    interval_minutes: int,
    start_exclusive_ts: int,
    end_inclusive_ts: Optional[int],
    tasks: Sequence[str],
    horizon_minutes: Sequence[int],
    task_short: Dict[str, str],
    task_label: Dict[str, str],
    allowed_pairs: Optional[Sequence[Tuple[str, int]]] = None,
) -> dict:
    pairs: List[Tuple[str, int, str, str, Optional[str]]] = []
    allowed = {(str(task), int(horizon)) for task, horizon in allowed_pairs} if allowed_pairs is not None else None
    for task in tasks:
        short = task_short.get(str(task))
        label = task_label.get(str(task))
        if not short or not label:
            continue
        for horizon in horizon_minutes:
            if allowed is not None and (str(task), int(horizon)) not in allowed:
                continue
            pairs.append(
                (
                    str(task),
                    int(horizon),
                    f"{CURRENT_MODEL_SPEC.prediction_prefix}_pred_mean_{short}_{int(horizon)}m",
                    f"{label}_{int(horizon)}m",
                    f"future_direction_{int(horizon)}m" if str(task) == "log_return" else None,
                )
            )
    forecast_cols = ["ts", "asset"]
    eval_cols = ["ts", "asset"]
    for _, _, mean_col, label_col, dir_col in pairs:
        forecast_cols.append(mean_col)
        eval_cols.append(label_col)
        if dir_col:
            eval_cols.append(dir_col)
    merged_frames: List[pd.DataFrame] = []
    for asset in assets:
        pred = _read_table_asset_range(
            run_parquet_root / f"{CURRENT_MODEL_SPEC.forecast_table_tag}_{int(interval_minutes)}",
            asset=asset,
            start_exclusive_ts=start_exclusive_ts,
            end_inclusive_ts=end_inclusive_ts,
            columns=sorted(set(forecast_cols)),
        )
        ev = _read_table_asset_range(
            run_parquet_root / f"{CURRENT_MODEL_SPEC.eval_table_tag}_{int(interval_minutes)}",
            asset=asset,
            start_exclusive_ts=start_exclusive_ts,
            end_inclusive_ts=end_inclusive_ts,
            columns=sorted(set(eval_cols)),
        )
        if pred.empty or ev.empty:
            continue
        merged = pred.merge(ev, on=["ts", "asset"], how="inner")
        if not merged.empty:
            merged_frames.append(merged)
    if not merged_frames:
        return {"forecast_count": 0, "by_target_horizon": {}, "by_asset_target_horizon": {}}
    all_df = pd.concat(merged_frames, ignore_index=True)
    by_pair: Dict[str, dict] = {}
    by_asset_pair: Dict[str, Dict[str, dict]] = {}
    total_n = 0
    total_abs = 0.0
    total_sq = 0.0
    dir_hit = 0
    dir_total = 0
    for task, horizon, mean_col, label_col, dir_col in pairs:
        pair_key = f"{task}:{int(horizon)}m"
        if mean_col not in all_df.columns or label_col not in all_df.columns:
            by_pair[pair_key] = {"forecast_count": 0}
            continue
        cols = ["ts", "asset", mean_col, label_col]
        if dir_col and dir_col in all_df.columns:
            cols.append(dir_col)
        pair_df = all_df[cols].copy()
        yt = pd.to_numeric(pair_df[label_col], errors="coerce")
        yp = pd.to_numeric(pair_df[mean_col], errors="coerce")
        pair_df = pair_df[yt.notna() & yp.notna()].copy()
        metrics = accuracy_from_frame(
            df=pair_df,
            y_col=label_col,
            pred_col=mean_col,
            dir_col=dir_col,
        )
        by_pair[pair_key] = metrics
        for asset, asset_df in pair_df.groupby("asset", sort=True):
            by_asset_pair.setdefault(str(asset), {})[pair_key] = accuracy_from_frame(
                df=asset_df,
                y_col=label_col,
                pred_col=mean_col,
                dir_col=dir_col,
            )
        count = int(metrics.get("forecast_count", 0) or 0)
        if count > 0:
            err = pd.to_numeric(pair_df[mean_col], errors="coerce").astype(float) - pd.to_numeric(
                pair_df[label_col], errors="coerce"
            ).astype(float)
            total_n += count
            total_abs += float(err.abs().sum())
            total_sq += float((err * err).sum())
            if dir_col and dir_col in pair_df.columns:
                true_dir = pd.to_numeric(pair_df[dir_col], errors="coerce").fillna(0).astype(int)
                pred_dir = pd.to_numeric(pair_df[mean_col], errors="coerce").astype(float).apply(
                    lambda v: 1 if v > 0.0001 else (-1 if v < -0.0001 else 0)
                )
                dir_hit += int((pred_dir.astype(int) == true_dir).sum())
                dir_total += int(len(pair_df))
    return {
        "forecast_count": int(total_n),
        "mae": float(total_abs / total_n) if total_n > 0 else None,
        "rmse": float(math.sqrt(total_sq / total_n)) if total_n > 0 else None,
        "directional_accuracy": float(dir_hit / dir_total) if dir_total > 0 else None,
        "by_target_horizon": by_pair,
        "by_asset_target_horizon": by_asset_pair,
    }


def verify_full_month_outputs(
    *,
    run_parquet_root: Path,
    assets: Sequence[str],
    start_exclusive_ts: int,
    end_inclusive_ts: int,
    interval_minutes: int,
    tasks: Sequence[str],
    horizon_minutes: Sequence[int],
    task_short: Dict[str, str],
    task_label: Dict[str, str],
    allowed_pairs: Optional[Sequence[Tuple[str, int]]] = None,
) -> dict:
    expected_rows = int(max(0, (int(end_inclusive_ts) - int(start_exclusive_ts)) // (60 * int(interval_minutes))))
    allowed = {(str(task), int(horizon)) for task, horizon in allowed_pairs} if allowed_pairs is not None else None
    per_unit: Dict[str, dict] = {}
    failures: List[dict] = []
    for asset in assets:
        for task in tasks:
            short = task_short.get(str(task))
            label = task_label.get(str(task))
            if not short or not label:
                continue
            for horizon in horizon_minutes:
                if allowed is not None and (str(task), int(horizon)) not in allowed:
                    continue
                pair_key = f"{asset}|{task}:{int(horizon)}m"
                pred_col = f"{CURRENT_MODEL_SPEC.prediction_prefix}_pred_mean_{short}_{int(horizon)}m"
                eval_col = f"{label}_{int(horizon)}m"
                pred = _read_table_asset_range(
                    run_parquet_root / f"{CURRENT_MODEL_SPEC.forecast_table_tag}_{int(interval_minutes)}",
                    asset=str(asset),
                    start_exclusive_ts=int(start_exclusive_ts),
                    end_inclusive_ts=int(end_inclusive_ts),
                    columns=["ts", "asset", pred_col],
                )
                ev = _read_table_asset_range(
                    run_parquet_root / f"{CURRENT_MODEL_SPEC.eval_table_tag}_{int(interval_minutes)}",
                    asset=str(asset),
                    start_exclusive_ts=int(start_exclusive_ts),
                    end_inclusive_ts=int(end_inclusive_ts),
                    columns=["ts", "asset", eval_col],
                )
                pred_non_null = int(pd.to_numeric(pred.get(pred_col), errors="coerce").notna().sum()) if pred_col in pred.columns else 0
                eval_non_null = int(pd.to_numeric(ev.get(eval_col), errors="coerce").notna().sum()) if eval_col in ev.columns else 0
                unit_result = {
                    "asset": str(asset),
                    "task": str(task),
                    "horizon_minutes": int(horizon),
                    "expected_rows": int(expected_rows),
                    "forecast_rows": int(len(pred)),
                    "forecast_non_null": int(pred_non_null),
                    "eval_rows": int(len(ev)),
                    "eval_non_null": int(eval_non_null),
                    "ok": bool(
                        int(len(pred)) == int(expected_rows)
                        and int(pred_non_null) == int(expected_rows)
                        and int(len(ev)) == int(expected_rows)
                        and int(eval_non_null) == int(expected_rows)
                    ),
                }
                per_unit[pair_key] = unit_result
                if not unit_result["ok"]:
                    failures.append(unit_result)
    return {
        "expected_rows_per_unit": int(expected_rows),
        "units_checked": int(len(per_unit)),
        "units_ok": int(sum(1 for value in per_unit.values() if bool(value.get("ok")))),
        "failures": failures,
        "per_unit": per_unit,
    }


def write_unit_artifacts(output_dir: Path, stage_result: dict) -> int:
    unit_diag = stage_result.get("unit_diag", {})
    by_asset_pair = ((stage_result.get("accuracy") or {}).get("by_asset_target_horizon") or {})
    resource = stage_result.get("resources") or {}
    training_window = stage_result.get("training_window_label")
    runtime_profile = stage_result.get("runtime_profile")
    interval_label = str(stage_result.get("config", {}).get("interval") or "")
    count = 0
    for _, diag in sorted(unit_diag.items()):
        asset = str(diag.get("asset"))
        horizon = int(diag.get("horizon_minutes"))
        task = str(diag.get("task"))
        pair_key = f"{task}:{int(horizon)}m"
        accuracy = ((by_asset_pair.get(asset) or {}).get(pair_key) or {})
        payload = {
            "generated_utc": utc_now_iso(),
            "run_status": "success" if stage_result.get("success") else "failed",
            "runtime_profile": runtime_profile,
            "interval": interval_label,
            "unit_workers": stage_result.get("config", {}).get("unit_workers"),
            "model_threads": stage_result.get("config", {}).get("model_threads"),
            "asset": asset,
            "training_window": training_window,
            "training_window_months": stage_result.get("training_window_months"),
            "training_window_bars": stage_result.get("training_window_bars"),
            "horizon_minutes": horizon,
            "task": task,
            "training_runtime_seconds": float(diag.get("fit_total_s", 0.0) or 0.0) + float(diag.get("refit_s", 0.0) or 0.0),
            "forecast_runtime_seconds": float(diag.get("predict_s", 0.0) or 0.0),
            "peak_memory_mb": float(resource.get("peak_proc_rss_gb", 0.0) or 0.0) * 1024.0,
            "cpu_utilization_estimate": float(resource.get("avg_proc_cpu_pct", 0.0) or 0.0),
            "training_rows": int(diag.get("selected_window_bars", 0) or 0),
            "feature_count": int(diag.get("feature_count", 0) or 0),
            "rmse": accuracy.get("rmse"),
            "mae": accuracy.get("mae"),
            "directional_accuracy": accuracy.get("directional_accuracy"),
            "forecast_count": accuracy.get("forecast_count"),
            "refit_count": int(diag.get("refit_count", 0) or 0),
            "run_wall_clock_s": stage_result.get("timing", {}).get("wall_clock_s"),
            "module_run_root": stage_result.get("paths", {}).get("run_root"),
            "run_summary_path": stage_result.get("paths", {}).get("run_summary"),
            "combined_log": stage_result.get("paths", {}).get("combined_log"),
            "module_log": stage_result.get("paths", {}).get("module_log"),
        }
        dst_dir = (
            output_dir
            / f"interval={interval_label}"
            / f"runtime_profile={runtime_profile}"
            / f"asset={asset}"
            / f"training_window={training_window}"
            / f"horizon={int(horizon)}m"
            / f"task={task}"
        )
        dst_dir.mkdir(parents=True, exist_ok=True)
        (dst_dir / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        count += 1
    return count


def emit_stage_function_telemetry(output_dir: Path, model_key: str, stage_result: dict) -> None:
    config = stage_result.get("config") or {}
    unit_diag = stage_result.get("unit_diag") or {}
    by_asset_pair = ((stage_result.get("accuracy") or {}).get("by_asset_target_horizon") or {})
    stage_name = f"stage{int(stage_result.get('stage_index', 0) or 0)}"
    interval_raw = str(config.get("interval") or "0m").rstrip("m")
    try:
        interval_minutes = int(interval_raw)
    except Exception:
        interval_minutes = None
    asset_count = len(config.get("assets") or [])
    forecast_rows = 0
    for combo_payloads in by_asset_pair.values():
        for metrics in (combo_payloads or {}).values():
            forecast_rows += int((metrics or {}).get("forecast_count", 0) or 0)
    selected_window_rows = sum(int((diag or {}).get("selected_window_bars", 0) or 0) for diag in unit_diag.values())
    for phase_name, seconds_key, function_name in (
        ("source_read", "load_s", "load_module_inputs"),
        ("label_build", "future_label_s", "compute_future_labels"),
        ("fit", "fit_s", "fit_models"),
        ("predict", "predict_s", "predict_outputs"),
        ("write", "write_s", "write_outputs"),
    ):
        total_seconds = sum(float((diag or {}).get(seconds_key, 0.0) or 0.0) for diag in unit_diag.values())
        emit_event_for_path(
            output_dir,
            family="Tabular_Numeric",
            model=str(model_key),
            stage=stage_name,
            interval_minutes=interval_minutes,
            asset_count=asset_count,
            function_name=function_name,
            module_name=__name__,
            phase_name=phase_name,
            parent_phase="diagnostics",
            status="completed",
            elapsed_seconds=total_seconds,
            input_rows=(selected_window_rows if phase_name in {"fit", "predict"} else len(unit_diag)),
            output_rows=(forecast_rows if phase_name in {"predict", "write"} else len(unit_diag)),
            reason_code=("predict_returned_empty" if phase_name == "predict" and forecast_rows == 0 else ""),
            output_path=str((stage_result.get("paths") or {}).get("run_summary", "")),
        )
    verification = stage_result.get("output_verification") or {}
    failures = list(verification.get("failures") or [])[:20]
    emit_event_for_path(
        output_dir,
        family="Tabular_Numeric",
        model=str(model_key),
        stage=stage_name,
        interval_minutes=interval_minutes,
        asset_count=asset_count,
        function_name="verify_full_month_outputs",
        module_name=__name__,
        phase_name="validation",
        parent_phase="diagnostics",
        status="completed",
        reason_code=("validation_dropped_all_rows" if failures else ""),
        input_rows=len(unit_diag),
        output_rows=max(0, len(unit_diag) - len(failures)),
        output_path=str((stage_result.get("paths") or {}).get("run_summary", "")),
    )
    for failure in failures:
        emit_event_for_path(
            output_dir,
            family="Tabular_Numeric",
            model=str(model_key),
            stage=stage_name,
            combo_key=f"{interval_minutes}:{int(failure.get('horizon_minutes', 0) or 0)}:{failure.get('task')}",
            interval_minutes=interval_minutes,
            horizon_minutes=int(failure.get("horizon_minutes", 0) or 0),
            task=str(failure.get("task") or ""),
            asset=str(failure.get("asset") or ""),
            function_name="verify_full_month_outputs",
            module_name=__name__,
            phase_name="validation",
            parent_phase="diagnostics",
            status="completed",
            reason_code="validation_dropped_all_rows",
            input_rows=int(failure.get("expected_rows", 0) or 0),
            output_rows=int(failure.get("forecast_rows", 0) or 0),
        )


def stage_run_dir(output_dir: Path, interval_minutes: int, workers: int, model_threads: int, train_window_months: Optional[int]) -> Path:
    runtime_profile = runtime_profile_name(int(workers), int(model_threads))
    training_window_label = month_label(train_window_months)
    interval_label = f"{int(interval_minutes)}m"
    return output_dir / "_runs" / f"interval={interval_label}" / f"runtime_profile={runtime_profile}" / f"training_window={training_window_label}"


def remove_tree_with_retries(path: Path, *, attempts: int = 5, delay_seconds: float = 0.5) -> None:
    last_error: Optional[Exception] = None
    for _ in range(max(1, int(attempts))):
        try:
            if path.exists():
                shutil.rmtree(path)
            return
        except FileNotFoundError:
            return
        except PermissionError as exc:
            last_error = exc
            time.sleep(max(0.0, float(delay_seconds)))
    if path.exists() and last_error is not None:
        tombstone = path.with_name(path.name + f".stale_{int(time.time())}")
        try:
            path.rename(tombstone)
            return
        except Exception:
            raise last_error


def cleanup_stage_outputs(
    *,
    output_dir: Path,
    interval_minutes: int,
    workers: int,
    model_threads: int,
    train_window_months: Optional[int],
    assets: Sequence[str],
) -> None:
    run_dir = stage_run_dir(output_dir, interval_minutes, workers, model_threads, train_window_months)
    if run_dir.exists():
        remove_tree_with_retries(run_dir)
    runtime_profile = runtime_profile_name(int(workers), int(model_threads))
    training_window_label = month_label(train_window_months)
    interval_label = f"{int(interval_minutes)}m"
    for asset in assets:
        asset_dir = (
            output_dir
            / f"interval={interval_label}"
            / f"runtime_profile={runtime_profile}"
            / f"asset={asset}"
            / f"training_window={training_window_label}"
        )
        if asset_dir.exists():
            remove_tree_with_retries(asset_dir)


def load_existing_stage_result(run_dir: Path) -> Optional[dict]:
    summary_path = run_dir / "run_summary.json"
    if not summary_path.exists():
        return None
    try:
        return json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def prune_stage2_module_parquet(run_results: Sequence[dict]) -> List[str]:
    pruned: List[str] = []
    for run in run_results:
        paths = run.get("paths") or {}
        module_root = Path(paths.get("run_root", ""))
        if not str(module_root):
            continue
        parquet_dir = module_root / "parquet"
        if parquet_dir.exists():
            remove_tree_with_retries(parquet_dir)
            if not parquet_dir.exists():
                pruned.append(str(parquet_dir.resolve()))
    return pruned


def stage_result_is_complete(stage_result: Optional[dict]) -> bool:
    if not stage_result:
        return False
    if not bool(stage_result.get("success")):
        return False
    return_code = stage_result.get("return_code")
    if return_code is None:
        return False
    if int(return_code) != 0:
        return False
    failures = ((stage_result.get("output_verification") or {}).get("failures") or [])
    if failures:
        return False
    return True


def discover_resume_output_dir(base_output_dir: Path, planned_relative_run_dirs: Sequence[Path]) -> Optional[Path]:
    if not base_output_dir.exists():
        return None
    run_roots = sorted((p for p in base_output_dir.glob("run=*") if p.is_dir()), key=lambda p: p.name)
    for run_root in reversed(run_roots):
        matching_stage_dirs = [run_root / rel_path for rel_path in planned_relative_run_dirs if (run_root / rel_path).exists()]
        if not matching_stage_dirs:
            continue
        manifest_path = run_root / "diagnostic_manifest.json"
        if not manifest_path.exists():
            return run_root.resolve()
        if any(not stage_result_is_complete(load_existing_stage_result(stage_dir)) for stage_dir in matching_stage_dirs):
            return run_root.resolve()
        if any(not (run_root / rel_path).exists() for rel_path in planned_relative_run_dirs):
            return run_root.resolve()
    return None


def write_manifest(output_dir: Path, manifest: dict) -> None:
    (output_dir / "diagnostic_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def run_stage(
    *,
    stage_index: int,
    project_root: Path,
    parquet_root: Path,
    output_dir: Path,
    runtime_cfg_path: Path,
    runtime_cfg_original: str,
    assets: Sequence[str],
    seed_ts: int,
    accuracy_end_ts: int,
    workers: int,
    model_threads: int,
    interval_minutes: int,
    train_window_months: Optional[int],
    train_window_bars: int,
    intervals_raw: str,
    combo_list_raw: str,
    tasks_raw: str,
    runnable_tasks: Sequence[str],
    horizon_minutes_raw: str,
    requested_horizons: Sequence[int],
    task_short: Dict[str, str],
    task_label: Dict[str, str],
    allowed_pairs: Sequence[Tuple[str, int]],
    python_exe: str,
    sample_interval_s: float,
    feature_profile_json: Optional[Path] = None,
) -> dict:
    runtime_profile = runtime_profile_name(int(workers), int(model_threads))
    training_window_label = month_label(train_window_months)
    interval_label = f"{int(interval_minutes)}m"
    run_dir = stage_run_dir(output_dir, int(interval_minutes), int(workers), int(model_threads), train_window_months)
    if run_dir.exists():
        existing_result = load_existing_stage_result(run_dir)
        if stage_result_is_complete(existing_result):
            print(
                f"[{utc_now_iso()}] stage={stage_index} interval={interval_label} profile={runtime_profile} "
                f"window={training_window_label} status=resume-skip "
                f"wall_clock_s={float((existing_result.get('timing') or {}).get('wall_clock_s', 0.0) or 0.0):.3f} "
                f"artifacts={int(existing_result.get('artifact_count', 0) or 0)}"
            )
            return existing_result
        print(
            f"[{utc_now_iso()}] stage={stage_index} interval={interval_label} profile={runtime_profile} "
            f"window={training_window_label} status=resume-rebuild"
        )
        cleanup_stage_outputs(
            output_dir=output_dir,
            interval_minutes=int(interval_minutes),
            workers=int(workers),
            model_threads=int(model_threads),
            train_window_months=train_window_months,
            assets=assets,
        )
    run_dir.mkdir(parents=True, exist_ok=False)
    module_root = run_dir / "module_root"
    (module_root / "logs").mkdir(parents=True, exist_ok=True)
    (module_root / "parquet").mkdir(parents=True, exist_ok=True)
    (module_root / "model_states").mkdir(parents=True, exist_ok=True)
    write_runtime_config(runtime_cfg_path, runtime_cfg_original, int(model_threads))
    seed_forecast_edge(
        module_root / "parquet",
        assets,
        int(interval_minutes),
        seed_ts,
        runnable_tasks,
        requested_horizons,
        task_short,
        allowed_pairs=allowed_pairs,
    )
    validated_pairs, missing_seed_cols = validate_seeded_forecast_edge(
        module_root / "parquet",
        assets,
        int(interval_minutes),
        seed_ts,
        runnable_tasks,
        requested_horizons,
        task_short,
        allowed_pairs=allowed_pairs,
    )
    if missing_seed_cols:
        raise RuntimeError(f"Seed validation failed for runtime_profile={runtime_profile} window={training_window_label}")
    env = os.environ.copy()
    env["PIPELINE_ROOT"] = str(module_root)
    env["PIPELINE_PARQUET_ROOT"] = str(parquet_root)
    env["PIPELINE_PARQUET_FEATURES_ROOT"] = str(parquet_root)
    env[CURRENT_MODEL_SPEC.parquet_root_env] = str(module_root / "parquet")
    apply_nested_module_sandbox_roots(env, module_root)
    env[CURRENT_MODEL_SPEC.train_windows_env] = str(int(train_window_bars))
    env[CURRENT_MODEL_SPEC.progress_seconds_env] = "600"
    max_horizon_bars = max(max(1, int(h) // int(interval_minutes)) for h in requested_horizons) if requested_horizons else 1
    env[CURRENT_MODEL_SPEC.source_end_env] = str(int(accuracy_end_ts))
    env[CURRENT_MODEL_SPEC.work_start_env] = str(int(seed_ts))
    env[CURRENT_MODEL_SPEC.forecast_resume_edge_env] = str(int(seed_ts))
    env[CURRENT_MODEL_SPEC.eval_resume_edge_env] = str(int(seed_ts))
    env[CURRENT_MODEL_SPEC.source_start_env] = str(
        trailing_source_start_ts(
            seed_ts=int(seed_ts),
            interval_minutes=int(interval_minutes),
            train_window_bars=int(train_window_bars),
            max_horizon_bars=int(max_horizon_bars),
        )
    )
    if feature_profile_json is not None:
        env["TABULAR_NUMERIC_FEATURE_SELECTION_FILE"] = str(feature_profile_json.resolve())
    cmd = [
        *diagnostic_entry_command(python_exe),
        "--intervals",
        str(intervals_raw),
        "--assets",
        ",".join(assets),
        "--combo-list",
        str(combo_list_raw),
        "--horizon-minutes",
        str(horizon_minutes_raw),
        "--tasks",
        str(tasks_raw),
        "--mode",
        "incremental",
        "--unit-workers",
        str(int(workers)),
    ]
    if train_window_months is not None:
        cmd.extend(["--train-window-months", str(int(train_window_months))])
    combined_log = run_dir / "combined.log"
    stage_started_utc = utc_now_iso()
    print(
        f"[{stage_started_utc}] stage={stage_index} interval={interval_label} profile={runtime_profile} "
        f"window={training_window_label} status=starting tasks={tasks_raw} horizons={horizon_minutes_raw}"
    )
    module_log = module_root / "logs" / str(CURRENT_MODEL_SPEC.log_file_name)
    with combined_log.open("w", encoding="utf-8") as logf:
        proc = subprocess.Popen(cmd, cwd=str(project_root), env=env, stdout=logf, stderr=subprocess.STDOUT, text=True)
        cpu_total: List[float] = []
        proc_cpu: List[float] = []
        proc_rss: List[float] = []
        proc_tree_cpu: List[float] = []
        proc_tree_rss: List[float] = []
        sys_mem: List[float] = []
        per_core_active_ge_50: List[float] = []
        per_core_active_ge_80: List[float] = []
        per_core_imbalance: List[float] = []
        worker_active_units: List[float] = []
        worker_idle_slots: List[float] = []
        io_read_total_bytes = 0.0
        io_write_total_bytes = 0.0
        io_reading_s = 0.0
        io_writing_s = 0.0
        idle_with_queue_s = 0.0
        fewer_than_half_active_s = 0.0
        dwell_seconds: Dict[str, float] = {
            "cpu_total_ge_80_s": 0.0,
            "cpu_total_ge_90_s": 0.0,
            "cpu_total_ge_95_s": 0.0,
            "proc_cpu_ge_80_s": 0.0,
            "proc_cpu_ge_90_s": 0.0,
            "proc_cpu_ge_95_s": 0.0,
            "sys_mem_ge_80_s": 0.0,
            "sys_mem_ge_90_s": 0.0,
        }
        continuous_seconds: Dict[str, float] = {
            "max_continuous_cpu_total_ge_90_s": 0.0,
            "max_continuous_proc_cpu_ge_90_s": 0.0,
            "max_continuous_sys_mem_ge_90_s": 0.0,
        }
        max_continuous_seconds: Dict[str, float] = {
            "max_continuous_cpu_total_ge_90_s": 0.0,
            "max_continuous_proc_cpu_ge_90_s": 0.0,
            "max_continuous_sys_mem_ge_90_s": 0.0,
        }
        peak_threads = 0
        psutil.cpu_percent(interval=None)
        psutil.cpu_percent(interval=None, percpu=True)
        try:
            parent: Optional[psutil.Process] = psutil.Process(proc.pid)
            parent.cpu_percent(interval=None)
        except Exception:
            parent = None
        last_tree_io: Optional[Tuple[float, float]] = None
        sample_records: List[dict] = []
        log_state: Dict[str, Any] = {
            "offset": 0,
            "partial": "",
            "work_items": 0,
            "active_units": {},
            "completed_units": set(),
            "unit_start_ts": {},
            "unit_end_ts": {},
            "run_start_ts": None,
            "run_end_ts": None,
        }
        t0 = time.monotonic()
        last_sample_ts = t0
        while proc.poll() is None:
            time.sleep(max(0.1, float(sample_interval_s)))
            sample_ts = time.monotonic()
            elapsed_sample_s = max(0.0, float(sample_ts - last_sample_ts))
            sample_start_s = max(0.0, float(last_sample_ts - t0))
            sample_end_s = max(0.0, float(sample_ts - t0))
            last_sample_ts = sample_ts
            current_percpu = [float(value) for value in psutil.cpu_percent(interval=None, percpu=True)]
            current_cpu_total = avg(current_percpu) if current_percpu else float(psutil.cpu_percent(interval=None))
            current_sys_mem = float(psutil.virtual_memory().percent)
            cpu_total.append(current_cpu_total)
            sys_mem.append(current_sys_mem)
            per_core_active_ge_50.append(float(sum(1 for value in current_percpu if float(value) >= 50.0)))
            per_core_active_ge_80.append(float(sum(1 for value in current_percpu if float(value) >= 80.0)))
            per_core_imbalance.append(stddev(current_percpu))
            proc_list: List[psutil.Process] = []
            if parent is not None:
                try:
                    if parent.is_running():
                        proc_list = [parent] + parent.children(recursive=True)
                except Exception:
                    proc_list = []
            current_cpu = 0.0
            current_rss = 0.0
            current_tree_cpu = 0.0
            current_tree_rss = 0.0
            current_threads = 0
            current_tree_read_bytes = 0.0
            current_tree_write_bytes = 0.0
            for child in proc_list:
                try:
                    child_cpu = float(child.cpu_percent(interval=None))
                    child_rss = float(child.memory_info().rss)
                    current_tree_cpu += child_cpu
                    current_tree_rss += child_rss
                    if parent is not None and int(child.pid) == int(parent.pid):
                        current_cpu = child_cpu
                        current_rss = child_rss
                    current_threads += int(child.num_threads())
                    io_counters = child.io_counters()
                    current_tree_read_bytes += float(getattr(io_counters, "read_bytes", 0.0) or 0.0)
                    current_tree_write_bytes += float(getattr(io_counters, "write_bytes", 0.0) or 0.0)
                except Exception:
                    continue
            proc_cpu.append(current_tree_cpu)
            proc_rss.append(current_tree_rss)
            proc_tree_cpu.append(current_tree_cpu)
            proc_tree_rss.append(current_tree_rss)
            peak_threads = max(peak_threads, current_threads)
            if last_tree_io is not None:
                read_delta = max(0.0, current_tree_read_bytes - float(last_tree_io[0]))
                write_delta = max(0.0, current_tree_write_bytes - float(last_tree_io[1]))
                io_read_total_bytes += read_delta
                io_write_total_bytes += write_delta
                if read_delta > 0.0:
                    io_reading_s += elapsed_sample_s
                if write_delta > 0.0:
                    io_writing_s += elapsed_sample_s
            last_tree_io = (current_tree_read_bytes, current_tree_write_bytes)
            consume_module_log_delta(module_log, log_state, sample_end_s)
            worker_state = worker_state_snapshot(log_state, int(workers))
            worker_active_units.append(float(worker_state["active_units"]))
            worker_idle_slots.append(float(worker_state["idle_worker_slots"]))
            if worker_state["queued_work_items"] > 0 and worker_state["idle_worker_slots"] > 0:
                idle_with_queue_s += elapsed_sample_s
            if float(worker_state["active_units"]) < (float(workers) / 2.0):
                fewer_than_half_active_s += elapsed_sample_s
            sample_records.append(
                {
                    "sample_start_s": float(sample_start_s),
                    "sample_end_s": float(sample_end_s),
                    "queued_work_items": int(worker_state["queued_work_items"]),
                    "active_units": int(worker_state["active_units"]),
                    "idle_worker_slots": int(worker_state["idle_worker_slots"]),
                }
            )
            update_dwell(
                dwell_seconds=dwell_seconds,
                continuous_seconds=continuous_seconds,
                max_continuous_seconds=max_continuous_seconds,
                elapsed_s=elapsed_sample_s,
                value=current_cpu_total,
                total_prefix="cpu_total",
                thresholds=(80, 90, 95),
                continuous_threshold=90,
            )
            update_dwell(
                dwell_seconds=dwell_seconds,
                continuous_seconds=continuous_seconds,
                max_continuous_seconds=max_continuous_seconds,
                elapsed_s=elapsed_sample_s,
                value=current_tree_cpu,
                total_prefix="proc_cpu",
                thresholds=(80, 90, 95),
                continuous_threshold=90,
            )
            update_dwell(
                dwell_seconds=dwell_seconds,
                continuous_seconds=continuous_seconds,
                max_continuous_seconds=max_continuous_seconds,
                elapsed_s=elapsed_sample_s,
                value=current_sys_mem,
                total_prefix="sys_mem",
                thresholds=(80, 90),
                continuous_threshold=90,
            )
        consume_module_log_delta(module_log, log_state, float(time.monotonic() - t0))
        wall_clock_s = float(time.monotonic() - t0)
        return_code = int(proc.returncode or 0)
    parsed = parse_module_log(module_log)
    accuracy = compute_accuracy_metrics(
        run_parquet_root=module_root / "parquet",
        assets=list(assets),
        interval_minutes=int(interval_minutes),
        start_exclusive_ts=int(seed_ts),
        end_inclusive_ts=int(accuracy_end_ts),
        tasks=list(runnable_tasks),
        horizon_minutes=list(requested_horizons),
        task_short=task_short,
        task_label=task_label,
        allowed_pairs=list(allowed_pairs),
    )
    output_verification = verify_full_month_outputs(
        run_parquet_root=module_root / "parquet",
        assets=list(assets),
        start_exclusive_ts=int(seed_ts),
        end_inclusive_ts=int(accuracy_end_ts),
        interval_minutes=int(interval_minutes),
        tasks=list(runnable_tasks),
        horizon_minutes=list(requested_horizons),
        task_short=task_short,
        task_label=task_label,
        allowed_pairs=list(allowed_pairs),
    )
    unit_diag: Dict[str, dict] = {}
    for key_name, diag in (parsed.get("unit_diag") or {}).items():
        enriched = dict(diag)
        if diag.get("selected_window_events"):
            try:
                enriched["selected_window_bars"] = int(diag["selected_window_events"][-1].get("to", train_window_bars))
            except Exception:
                enriched["selected_window_bars"] = int(train_window_bars)
        else:
            enriched["selected_window_bars"] = int(train_window_bars)
        enriched["load_s"] = float(diag.get("load_s", diag.get("data_load_s", 0.0)) or 0.0)
        enriched["future_label_s"] = float(diag.get("future_label_s", 0.0) or 0.0)
        enriched["fit_s"] = float(diag.get("fit_s", 0.0) or 0.0) or float(diag.get("fit_total_s", 0.0) or 0.0) + float(
            diag.get("refit_s", 0.0) or 0.0
        )
        enriched["predict_s"] = float(diag.get("predict_s", 0.0) or 0.0)
        enriched["write_s"] = float(diag.get("write_s", diag.get("parquet_write_s", 0.0)) or 0.0)
        enriched["total_wall_s"] = float(diag.get("total_wall_s", diag.get("wall_s", 0.0)) or 0.0)
        unit_diag[key_name] = enriched
    horizon_rollups = {
        "avg_unit_wall_s_by_horizon": avg_by_horizon(unit_diag, "total_wall_s"),
        "avg_future_label_s_by_horizon": avg_by_horizon(unit_diag, "future_label_s"),
        "avg_fit_s_by_horizon": avg_by_horizon(unit_diag, "fit_s"),
        "avg_predict_s_by_horizon": avg_by_horizon(unit_diag, "predict_s"),
        "avg_write_s_by_horizon": avg_by_horizon(unit_diag, "write_s"),
    }
    tail_report = build_tail_report(
        wall_clock_s=float(wall_clock_s),
        unit_diag=unit_diag,
        unit_start_ts=parsed.get("unit_start_ts", {}) or {},
        unit_end_ts=parsed.get("unit_end_ts", {}) or {},
        sample_records=sample_records,
    )
    failure_forensics = (
        build_failure_forensics(combined_log=combined_log, module_root=module_root) if int(return_code) != 0 else None
    )
    dwell_pct = {
        "cpu_total_ge_80_pct_run": (float(dwell_seconds["cpu_total_ge_80_s"]) / wall_clock_s * 100.0) if wall_clock_s > 0 else 0.0,
        "cpu_total_ge_90_pct_run": (float(dwell_seconds["cpu_total_ge_90_s"]) / wall_clock_s * 100.0) if wall_clock_s > 0 else 0.0,
        "cpu_total_ge_95_pct_run": (float(dwell_seconds["cpu_total_ge_95_s"]) / wall_clock_s * 100.0) if wall_clock_s > 0 else 0.0,
        "proc_cpu_ge_80_pct_run": (float(dwell_seconds["proc_cpu_ge_80_s"]) / wall_clock_s * 100.0) if wall_clock_s > 0 else 0.0,
        "proc_cpu_ge_90_pct_run": (float(dwell_seconds["proc_cpu_ge_90_s"]) / wall_clock_s * 100.0) if wall_clock_s > 0 else 0.0,
        "proc_cpu_ge_95_pct_run": (float(dwell_seconds["proc_cpu_ge_95_s"]) / wall_clock_s * 100.0) if wall_clock_s > 0 else 0.0,
        "sys_mem_ge_80_pct_run": (float(dwell_seconds["sys_mem_ge_80_s"]) / wall_clock_s * 100.0) if wall_clock_s > 0 else 0.0,
        "sys_mem_ge_90_pct_run": (float(dwell_seconds["sys_mem_ge_90_s"]) / wall_clock_s * 100.0) if wall_clock_s > 0 else 0.0,
    }
    result = {
        "stage_index": int(stage_index),
        "success": bool(return_code == 0),
        "return_code": int(return_code),
        "runtime_profile": runtime_profile,
        "training_window_label": training_window_label,
        "training_window_months": train_window_months,
        "training_window_bars": int(train_window_bars),
        "config": {
            "assets": list(assets),
            "unit_workers": int(workers),
            "model_threads": int(model_threads),
            "interval": interval_label,
            "tasks": list(runnable_tasks),
            "horizons": list(requested_horizons),
            "seed_ts": int(seed_ts),
            "accuracy_end_ts": int(accuracy_end_ts),
            "forecast_target_month_start_utc": datetime.fromtimestamp(seed_ts + (60 * int(interval_minutes)), tz=timezone.utc).isoformat(),
            "seeded_pairs": validated_pairs,
        },
        "timing": {
            "wall_clock_s": float(wall_clock_s),
            "module_run_span_s": parsed.get("module_run_span_s"),
        },
        "resources": {
            "sample_interval_s": float(sample_interval_s),
            "samples": len(cpu_total),
            "avg_cpu_total_pct": avg(cpu_total),
            "peak_cpu_total_pct": max(cpu_total) if cpu_total else 0.0,
            "avg_proc_cpu_pct": avg(proc_cpu),
            "peak_proc_cpu_pct": max(proc_cpu) if proc_cpu else 0.0,
            "avg_proc_rss_gb": avg(proc_rss) / (1024**3),
            "peak_proc_rss_gb": (max(proc_rss) if proc_rss else 0.0) / (1024**3),
            "avg_proc_tree_cpu_pct": avg(proc_tree_cpu),
            "peak_proc_tree_cpu_pct": max(proc_tree_cpu) if proc_tree_cpu else 0.0,
            "avg_proc_tree_rss_mb": avg(proc_tree_rss) / (1024**2),
            "peak_proc_tree_rss_mb": (max(proc_tree_rss) if proc_tree_rss else 0.0) / (1024**2),
            "peak_proc_threads": int(peak_threads),
            "avg_sys_mem_pct": avg(sys_mem),
            "peak_sys_mem_pct": max(sys_mem) if sys_mem else 0.0,
            "avg_active_cores_ge_50": avg(per_core_active_ge_50),
            "avg_active_cores_ge_80": avg(per_core_active_ge_80),
            "max_active_cores_ge_80": max(per_core_active_ge_80) if per_core_active_ge_80 else 0.0,
            "core_imbalance_index": avg(per_core_imbalance),
            "avg_active_units_sampled": avg(worker_active_units),
            "avg_idle_worker_slots": avg(worker_idle_slots),
            "pct_run_with_idle_workers_and_nonempty_queue": (float(idle_with_queue_s) / wall_clock_s * 100.0) if wall_clock_s > 0 else 0.0,
            "pct_run_with_fewer_than_half_workers_active": (float(fewer_than_half_active_s) / wall_clock_s * 100.0) if wall_clock_s > 0 else 0.0,
            "read_mb_total": float(io_read_total_bytes) / (1024**2),
            "write_mb_total": float(io_write_total_bytes) / (1024**2),
            "pct_run_reading": (float(io_reading_s) / wall_clock_s * 100.0) if wall_clock_s > 0 else 0.0,
            "pct_run_writing": (float(io_writing_s) / wall_clock_s * 100.0) if wall_clock_s > 0 else 0.0,
            "resource_dwell": {
                **dwell_seconds,
                **max_continuous_seconds,
                **dwell_pct,
            },
        },
        "concurrency": {
            "dispatch_mode": parsed.get("dispatch_mode"),
            "work_items": parsed.get("work_items"),
            "multiple_units_active": parsed.get("multiple_units_active"),
            "max_parallel_active": parsed.get("max_parallel_active"),
        },
        "diag_aggregate": parsed.get("diag_aggregate", {}),
        "horizon_rollups": horizon_rollups,
        "tail_report": tail_report,
        "failure_forensics": failure_forensics,
        "unit_diag": unit_diag,
        "unit_ranges": parsed.get("unit_ranges", {}),
        "accuracy": accuracy,
        "output_verification": output_verification,
        "paths": {
            "run_dir": str(run_dir),
            "run_root": str(module_root),
            "combined_log": str(combined_log),
            "module_log": str(module_log),
            "run_summary": str(run_dir / "run_summary.json"),
        },
    }
    (run_dir / "run_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    result["artifact_count"] = int(write_unit_artifacts(output_dir, result))
    packet_paths = emit_standard_numeric_diagnostic_packet(
        packet_root=run_dir / "standard_diagnostic_packet",
        run_result=result,
        mode="test",
        module_name=__name__,
        run_id=f"{CURRENT_MODEL_SPEC.model_key}_{runtime_profile}_{interval_label}_{training_window_label}",
    )
    result["standard_diagnostic_packet"] = packet_paths
    result.setdefault("paths", {})["standard_diagnostic_packet"] = str(run_dir / "standard_diagnostic_packet")
    (run_dir / "run_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    emit_stage_function_telemetry(output_dir, CURRENT_MODEL_SPEC.model_key, result)
    if return_code != 0:
        raise RuntimeError(
            "Diagnostic stage failed: "
            f"stage={int(stage_index)} interval={interval_label} profile={runtime_profile} window={training_window_label} "
            f"return_code={int(return_code)} combined_log={combined_log} module_log={module_log}"
        )
    if output_verification.get("failures"):
        first_failure = dict(output_verification["failures"][0])
        raise RuntimeError(
            "Diagnostic stage output verification failed: "
            f"stage={int(stage_index)} interval={interval_label} profile={runtime_profile} window={training_window_label} "
            f"asset={first_failure.get('asset')} task={first_failure.get('task')} "
            f"horizon={int(first_failure.get('horizon_minutes', 0) or 0)}m "
            f"expected_rows={int(first_failure.get('expected_rows', 0) or 0)} "
            f"forecast_rows={int(first_failure.get('forecast_rows', 0) or 0)} "
            f"forecast_non_null={int(first_failure.get('forecast_non_null', 0) or 0)} "
            f"eval_rows={int(first_failure.get('eval_rows', 0) or 0)} "
            f"eval_non_null={int(first_failure.get('eval_non_null', 0) or 0)}"
        )
    print(
        f"[{utc_now_iso()}] stage={stage_index} interval={interval_label} profile={runtime_profile} window={training_window_label} "
        f"status={'ok' if result['success'] else f'failed({return_code})'} "
        f"wall_clock_s={float(result['timing'].get('wall_clock_s', 0.0)):.3f} "
        f"module_run_span_s={float(result['timing'].get('module_run_span_s', 0.0) or 0.0):.3f} "
        f"artifacts={result['artifact_count']}"
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=f"Comprehensive diagnostic scaling test for {CURRENT_MODEL_SPEC.runtime_config_key}.")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[5])
    parser.add_argument("--profile", type=str, default=selected_profile(default="pipeline_test"))
    parser.add_argument("--parquet-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--intervals", type=str, default=DEFAULT_INTERVALS)
    parser.add_argument("--forecast-days", type=float, default=DEFAULT_FORECAST_DAYS)
    parser.add_argument("--tasks", type=str, default=DEFAULT_TASKS)
    parser.add_argument("--horizon-minutes", type=str, default=DEFAULT_HORIZONS)
    parser.add_argument("--train-window-months", type=str, default=DEFAULT_TRAIN_WINDOWS)
    parser.add_argument("--asset-count", type=int, default=DEFAULT_ASSET_COUNT)
    parser.add_argument("--seed", type=int, default=int(datetime.now(timezone.utc).strftime("%Y%m%d")))
    parser.add_argument("--search-back-months", type=int, default=DEFAULT_SEARCH_BACK_MONTHS)
    parser.add_argument("--sample-interval", type=float, default=DEFAULT_SAMPLE_INTERVAL)
    parser.add_argument("--python-exe", type=str, default=sys.executable)
    parser.add_argument("--feature-profile-json", type=Path, default=None, help="Feature selection artifact from Stage 1 to apply during diagnostics.")
    parser.add_argument(
        "--staged",
        action="store_true",
        help="Strict staged mode. Requires the Stage-1 feature-selection artifact and forbids silent fallback to default combo discovery.",
    )
    parser.add_argument("--resume-run", type=str, default="", help="Resume a specific existing run session id or run directory name.")
    parser.add_argument(
        "--no-resume-latest",
        action="store_true",
        help="Disable automatic resume of the latest matching incomplete run when no explicit resume target is supplied.",
    )
    args = parser.parse_args()
    if args.parquet_root is None:
        args.parquet_root = Path(resolve_path("source_ohlcvt_root", profile=str(args.profile), required=False) or Path("parquet"))
    return args


def main_for_model(model_key: str = DEFAULT_MODEL_KEY) -> None:
    global CURRENT_MODEL_SPEC, DEFAULT_OUTPUT_DIR
    CURRENT_MODEL_SPEC = get_tabular_numeric_model_spec(model_key)
    DEFAULT_OUTPUT_DIR = CURRENT_MODEL_SPEC.diagnostics_output_dir
    args = parse_args()
    project_root = args.project_root.resolve()
    parquet_root = args.parquet_root.resolve()
    base_output_dir = args.output_dir if args.output_dir.is_absolute() else (project_root / args.output_dir).resolve()
    runtime_cfg_path = project_root / "pipeline_runtime.json"
    runtime_cfg_original = runtime_cfg_path.read_text(encoding="utf-8-sig")
    supported_tasks, task_short, task_label = load_module_task_metadata(project_root)
    requested_tasks = parse_csv_str(args.tasks)
    requested_horizons = parse_csv_int(args.horizon_minutes)
    requested_intervals = parse_csv_int(args.intervals)
    runnable_tasks = [task for task in requested_tasks if task in supported_tasks and task in task_short and task in task_label]
    if not requested_tasks:
        runnable_tasks = [task for task in resolve_default_tasks(CURRENT_MODEL_SPEC, supported_tasks) if task in supported_tasks and task in task_short and task in task_label]
    unsupported_tasks = sorted(set(requested_tasks) - set(runnable_tasks))
    if not runnable_tasks:
        raise RuntimeError(
            f"No runnable tasks remain after filtering against {CURRENT_MODEL_SPEC.runtime_config_key} support. "
            f"requested={requested_tasks or resolve_default_tasks(CURRENT_MODEL_SPEC, supported_tasks)} supported={supported_tasks}"
        )
    staged_mode = bool(args.staged)
    feature_profile_json = args.feature_profile_json.resolve() if args.feature_profile_json else None
    feature_profile_cohort_assets: List[str] = []
    feature_profile_cohort_aliases: Dict[str, str] = {}
    if feature_profile_json is not None and feature_profile_json.exists():
        feature_profile_cohort_assets, feature_profile_cohort_aliases = cohort_assets_from_feature_profile_json(feature_profile_json)
    if staged_mode:
        if feature_profile_json is None:
            raise SystemExit(
                "Staged stage-2 run blocked: missing required Stage-1 feature-selection artifact. "
                "Pass --feature-profile-json <path>. Staged diagnostics must consume Stage-1 survivors and must not widen to defaults."
            )
        if not feature_profile_json.exists():
            raise SystemExit(
                "Staged stage-2 run blocked: Stage-1 feature-selection artifact not found at "
                f"{feature_profile_json}. Staged diagnostics must consume Stage-1 survivors and must not widen to defaults."
            )
        if not feature_profile_cohort_assets:
            raise SystemExit(
                "Staged stage-2 run blocked: Stage-1 feature-selection artifact does not contain persisted cohort assets. "
                "Regenerate Stage 1 with cohort persistence before running staged diagnostics."
            )
    resolved_combos: List[Tuple[int, int, str]]
    if feature_profile_json is not None and (staged_mode or (not requested_intervals and not requested_tasks and not requested_horizons)):
        resolved_combos = combos_from_feature_profile_json(
            feature_profile_json,
            supported_tasks=runnable_tasks,
            requested_intervals=requested_intervals,
            requested_tasks=requested_tasks,
            requested_horizons=requested_horizons,
        )
        if staged_mode and not resolved_combos:
            raise SystemExit(
                "Staged stage-2 run blocked: Stage-1 feature-selection artifact "
                f"{feature_profile_json} produced no surviving combos"
                f" for intervals={requested_intervals or 'ALL'} tasks={requested_tasks or 'ALL'} horizons={requested_horizons or 'ALL'}. "
                "Staged diagnostics must run only on Stage-1 survivors and must not widen to defaults."
            )
    else:
        resolved_combos = resolve_default_or_requested_combos(
            requested_intervals=requested_intervals,
            requested_tasks=requested_tasks,
            requested_horizons=requested_horizons,
            runnable_tasks=runnable_tasks,
        )
    if not resolved_combos:
        raise RuntimeError("No valid interval/horizon/task combinations remain after resolution.")
    combos_by_interval: Dict[int, List[Tuple[str, int]]] = {}
    for interval, horizon, task in resolved_combos:
        combos_by_interval.setdefault(int(interval), []).append((str(task), int(horizon)))
    window_tokens = parse_train_window_tokens(args.train_window_months)
    planned_relative_run_dirs: List[Path] = []
    runtime_matrix = list(runtime_profiles(MAX_LOGICAL_THREADS))
    for interval_minutes in sorted(combos_by_interval):
        for window_months in window_tokens:
            for workers, threads in runtime_matrix:
                planned_relative_run_dirs.append(
                    stage_run_dir(
                        Path("run=placeholder"),
                        int(interval_minutes),
                        int(workers),
                        int(threads),
                        window_months,
                    ).relative_to(Path("run=placeholder"))
                )
    resume_output_dir: Optional[Path] = None
    if args.resume_run:
        resume_name = str(args.resume_run).strip()
        if not resume_name.startswith("run="):
            resume_name = f"run={resume_name}"
        candidate = (base_output_dir / resume_name).resolve()
        if not candidate.exists():
            raise RuntimeError(f"Requested resume run does not exist: {candidate}")
        resume_output_dir = candidate
    elif not bool(args.no_resume_latest):
        latest_match = discover_resume_output_dir(
            base_output_dir,
            planned_relative_run_dirs,
        )
        if latest_match is not None:
            resume_output_dir = latest_match
    if resume_output_dir is not None:
        output_dir = resume_output_dir
        run_session_id = output_dir.name.split("=", 1)[-1]
        print(f"[{utc_now_iso()}] resuming diagnostic run at {output_dir}")
    else:
        run_session_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output_dir = (base_output_dir / f"run={run_session_id}").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    clamp_start = MonthKey(CLAMP_START_YEAR, CLAMP_START_MONTH)
    interval_details: List[dict] = []
    run_results: List[dict] = []
    analysis_outputs: Dict[str, Any] = {}
    host_info = {
        "logical_cpus": psutil.cpu_count(logical=True),
        "physical_cpus": psutil.cpu_count(logical=False),
        "total_ram_gb": round(psutil.virtual_memory().total / (1024**3), 3),
    }

    def current_manifest() -> dict:
        return {
            "generated_utc": utc_now_iso(),
            "script": str(Path(__file__).resolve()),
            "base_output_dir": str(base_output_dir),
            "output_dir": str(output_dir),
            "run_session_id": run_session_id,
            "project_root": str(project_root),
            "parquet_root": str(parquet_root),
            "host": host_info,
            "constraints": {
                "clamp_start": f"{CLAMP_START_YEAR:04d}-{CLAMP_START_MONTH:02d}-01",
                "intervals": [f"{int(interval)}m" for interval in sorted(combos_by_interval)],
                "forecast_style": "single_forward_simulation_only",
                "forecast_days_requested": float(args.forecast_days),
                "max_logical_threads": MAX_LOGICAL_THREADS,
            },
            "interval_details": interval_details,
            "selected_assets": sorted({asset for detail in interval_details for asset in detail.get("selected_assets", [])}),
            "seed": int(args.seed),
            "requested_tasks": requested_tasks,
            "runnable_tasks": runnable_tasks,
            "unsupported_not_run_tasks": unsupported_tasks,
            "resolved_combos": [
                {"interval_minutes": int(interval), "horizon_minutes": int(horizon), "task": str(task)}
                for interval, horizon, task in resolved_combos
            ],
            "interval_minutes": sorted({int(interval) for interval, _, _ in resolved_combos}),
            "horizon_minutes": sorted({int(horizon) for _, horizon, _ in resolved_combos}),
            "feature_profile_json": (str(args.feature_profile_json.resolve()) if args.feature_profile_json else None),
            "feature_profile_cohort_assets": list(feature_profile_cohort_assets),
            "runtime_profiles": [
                {
                    "runtime_profile": runtime_profile_name(int(workers), int(threads)),
                    "unit_workers": int(workers),
                    "model_threads": int(threads),
                }
                for workers, threads in runtime_matrix
            ],
            "analysis_outputs": dict(analysis_outputs),
            "runs": run_results,
        }

    write_manifest(output_dir, current_manifest())
    try:
        for interval_minutes in sorted(combos_by_interval):
            ohlc_root = parquet_root / f"ohlcvt_{int(interval_minutes)}"
            scalar_root = parquet_root / f"scalar_features_{int(interval_minutes)}"
            interval_pairs = list(combos_by_interval.get(int(interval_minutes), []))
            if not interval_pairs:
                continue
            final_month, eligible_assets = common_recent_window(
                ohlc_root=ohlc_root,
                scalar_root=scalar_root,
                min_assets=max(1, int(args.asset_count) if int(args.asset_count) > 0 else 1),
                window_months=13,
                search_back_months=int(args.search_back_months),
                clamp_start=clamp_start,
            )
            interval_asset_count = max(1, int(args.asset_count) if int(args.asset_count) > 0 else 1)
            if feature_profile_cohort_assets:
                selected_assets = [str(asset) for asset in feature_profile_cohort_assets]
                asset_alias_map = dict(feature_profile_cohort_aliases)
                missing_interval_assets = [asset for asset in selected_assets if asset not in set(eligible_assets)]
                if missing_interval_assets:
                    raise RuntimeError(
                        f"Stage-2 cohort is not eligible for interval={int(interval_minutes)}m. Missing assets: {missing_interval_assets}"
                    )
            else:
                selected_assets, asset_alias_map = choose_assets(eligible_assets, int(args.seed), int(args.asset_count))
            if not selected_assets:
                raise RuntimeError(f"No eligible assets discovered for interval={int(interval_minutes)}m")
            full_window_months = max(1, months_between(clamp_start, final_month))
            window_specs: List[Tuple[Optional[int], int]] = []
            for token in window_tokens:
                if token is None:
                    window_specs.append((None, months_to_bars(int(full_window_months), interval_minutes=int(interval_minutes))))
                else:
                    window_specs.append((int(token), months_to_bars(int(token), interval_minutes=int(interval_minutes))))
            asset_start_ts: Dict[str, int] = {}
            edges: List[int] = []
            for asset in eligible_assets:
                edge = max_asset_ts_for_month(ohlc_root, asset, final_month)
                if edge is None:
                    continue
                ohlc_start = min_asset_ts_from_monthly_table(ohlc_root, asset)
                scalar_start = min_asset_ts_from_monthly_table(scalar_root, asset)
                if ohlc_start is None or scalar_start is None:
                    continue
                asset_start_ts[str(asset)] = int(max(int(ohlc_start), int(scalar_start)))
                if asset in selected_assets:
                    edges.append(int(edge))
            bar_seconds = int(interval_minutes) * 60
            final_month, expected_accuracy_end_ts, actual_accuracy_end_ts = resolve_complete_observation_month(
                selected_assets=list(selected_assets),
                ohlc_root=ohlc_root,
                start_month=final_month,
                max_backtrack_months=max(1, int(args.search_back_months)),
                interval_minutes=int(interval_minutes),
            )
            accuracy_end_ts = int(expected_accuracy_end_ts)
            seed_ts = int(month_start_utc_ts(final_month.year, final_month.month) - bar_seconds)
            expected_forecast_rows_per_unit = int(max(0, (int(accuracy_end_ts) - int(seed_ts)) // bar_seconds))
            expected_forecast_days_actual = float(expected_forecast_rows_per_unit * int(interval_minutes)) / 1440.0
            interval_tasks = sorted({str(task) for task, _ in interval_pairs})
            interval_horizons = sorted({int(horizon) for _, horizon in interval_pairs})
            interval_details.append(
                {
                    "interval": f"{int(interval_minutes)}m",
                    "interval_minutes": int(interval_minutes),
                    "selected_assets": list(selected_assets),
                    "eligible_asset_pool": list(eligible_assets),
                    "asset_alias_resolution": dict(asset_alias_map),
                    "resolved_pairs": [f"{task}:{int(horizon)}m" for task, horizon in interval_pairs],
                    "forecast_target_month_start_utc": datetime.fromtimestamp(seed_ts + bar_seconds, tz=timezone.utc).isoformat(),
                    "forecast_rows_expected_per_unit": int(expected_forecast_rows_per_unit),
                    "forecast_days_actual": float(expected_forecast_days_actual),
                    "training_windows": [
                        {"label": month_label(months), "months": months, "bars": int(bars)}
                        for months, bars in window_specs
                    ],
                }
            )
            max_horizon_bars = max(max(1, int(horizon) // int(interval_minutes)) for _, horizon in interval_pairs)
            for window_months, window_bars in window_specs:
                required_start_ts = trailing_source_start_ts(
                    seed_ts=int(seed_ts),
                    interval_minutes=int(interval_minutes),
                    train_window_bars=int(window_bars),
                    max_horizon_bars=int(max_horizon_bars),
                )
                window_eligible_assets = [asset for asset in eligible_assets if int(asset_start_ts.get(str(asset), 10**18)) <= int(required_start_ts)]
                if len(window_eligible_assets) < int(interval_asset_count):
                    raise RuntimeError(
                        f"Insufficient exact-history assets for diagnostic stage: interval={int(interval_minutes)}m "
                        f"window={month_label(window_months)} required_assets={int(interval_asset_count)} available={len(window_eligible_assets)}"
                    )
                if feature_profile_cohort_assets:
                    window_selected_assets = [str(asset) for asset in selected_assets]
                    window_asset_alias_map = dict(asset_alias_map)
                    missing_window_assets = [asset for asset in window_selected_assets if asset not in set(window_eligible_assets)]
                    if missing_window_assets:
                        raise RuntimeError(
                            f"Stage-2 cohort lacks exact history for interval={int(interval_minutes)}m window={month_label(window_months)}. Missing assets: {missing_window_assets}"
                        )
                else:
                    window_selected_assets, window_asset_alias_map = choose_assets(window_eligible_assets, int(args.seed), int(args.asset_count))
                for workers, threads in runtime_profiles(MAX_LOGICAL_THREADS):
                    run_results.append(
                        run_stage(
                            stage_index=len(run_results) + 1,
                            project_root=project_root,
                            parquet_root=parquet_root,
                            output_dir=output_dir,
                            runtime_cfg_path=runtime_cfg_path,
                            runtime_cfg_original=runtime_cfg_original,
                            assets=window_selected_assets,
                            seed_ts=seed_ts,
                            accuracy_end_ts=accuracy_end_ts,
                            workers=int(workers),
                            model_threads=int(threads),
                            interval_minutes=int(interval_minutes),
                            train_window_months=window_months,
                            train_window_bars=int(window_bars),
                            intervals_raw=str(int(interval_minutes)),
                            combo_list_raw=",".join(
                                f"{int(interval_minutes)}:{int(horizon)}:{task}" for task, horizon in interval_pairs
                            ),
                            tasks_raw=",".join(interval_tasks),
                            runnable_tasks=interval_tasks,
                            horizon_minutes_raw=",".join(str(x) for x in interval_horizons),
                            requested_horizons=interval_horizons,
                            task_short=task_short,
                            task_label=task_label,
                            allowed_pairs=interval_pairs,
                            python_exe=args.python_exe,
                            sample_interval_s=float(args.sample_interval),
                            feature_profile_json=(args.feature_profile_json.resolve() if args.feature_profile_json else None),
                        )
                    )
                    write_manifest(output_dir, current_manifest())
                    if not bool(run_results[-1].get("success")):
                        raise RuntimeError(
                            "Diagnostic run aborted after first failed stage: "
                            f"stage={int(run_results[-1].get('stage_index', 0) or 0)} "
                            f"interval={run_results[-1].get('config', {}).get('interval')} "
                            f"profile={run_results[-1].get('runtime_profile')} "
                            f"window={run_results[-1].get('training_window_label')} "
                            f"return_code={int(run_results[-1].get('return_code', 0) or 0)}"
                        )
    finally:
        runtime_cfg_path.write_text(runtime_cfg_original, encoding="utf-8")
    write_manifest(output_dir, current_manifest())
    manifest_path = output_dir / "diagnostic_manifest.json"
    run_summary_paths = [Path((run.get("paths") or {}).get("run_summary", "")) for run in run_results]
    if manifest_path.exists() and run_results and all(path.exists() for path in run_summary_paths):
        analysis_outputs = analyze_manifest_for_model(model_key, manifest_path)
        pruned_module_parquet = prune_stage2_module_parquet(run_results)
        if pruned_module_parquet:
            analysis_outputs["pruned_module_parquet_roots"] = list(pruned_module_parquet)
        write_manifest(output_dir, current_manifest())
    print(f"[{utc_now_iso()}] diagnostic manifest written to {output_dir}")


def main() -> None:
    main_for_model(DEFAULT_MODEL_KEY)


if __name__ == "__main__":
    main()
