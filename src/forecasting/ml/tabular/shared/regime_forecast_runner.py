from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from src.features.scalar_features import (
    OHLCVT_PARQUET_ROOT,
    PARQUET_COMPRESSION,
    PARQUET_ROW_GROUP,
    PARQUET_ROOT as SCALAR_PARQUET_ROOT,
    assets_present_in_features,
    list_assets_from_ohlcvt,
)
from src.forecasting.common.io_atomic import atomic_replace, sibling_temp_path
from src.forecasting.common.path_config import resolve_path, selected_profile
from src.forecasting.common.pipeline_parquet_utils import (
    PipelineValidationError,
    partition_max_ts,
    validate_no_nan_columns,
    validate_strict_timegrid,
)
from src.forecasting.common.runtime_config import RUNTIME_CONFIG_PATH, cap_model_threads, get_model_threads, get_workers
from src.forecasting.common.sandbox_paths import SandboxOutputRoots, assert_write_allowed, resolve_sandbox_output_roots
from src.forecasting.common.ml_module_utils import (
    horizon_bars_from_minutes as shared_horizon_bars_from_minutes,
    make_unit_key,
    prune_pending_ts,
    read_ml_state,
    replace_pending_entries_for_unit,
    write_ml_state,
)
from src.regimes.contracts import (
    REGIME_AXIS_ORDER,
    REGIME_DEFAULT_FORECAST_INTERVALS,
    axis_target,
    forecast_ceiling_interval,
    forecast_output_columns,
    regime_forecast_part_path,
    safe_partition_value,
)


REGIME_THREAD_CAP_ENV_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")


def _regime_horizon_bars_from_minutes(horizon_minutes: int, interval_minutes: int) -> int:
    return shared_horizon_bars_from_minutes(interval_minutes=int(interval_minutes), horizon_minutes=int(horizon_minutes))


@dataclass(frozen=True)
class RegimeFamilyModuleSpec:
    module_slug: str
    family_name: str
    prediction_prefix: str
    log_prefix: str
    parquet_root_env: str
    progress_seconds_env: str = ""
    source_start_env: str = ""
    source_end_env: str = ""
    work_start_env: str = ""
    forecast_resume_edge_env: str = ""
    default_unit_workers: int = 1
    default_model_threads: int = 1
    max_logical_threads: int = 1
    thread_env_vars: Sequence[str] = ()
    thread_param_name: str = "n_jobs"
    default_intervals: Sequence[int] = REGIME_DEFAULT_FORECAST_INTERVALS
    default_horizon_minutes: Sequence[int] = (30, 240, 720)
    axes: Sequence[str] = REGIME_AXIS_ORDER
    select_feature_columns_fn: Optional[Callable[..., List[str]]] = None
    fit_classifier_fn: Optional[Callable[..., Any]] = None
    predict_classifier_fn: Optional[Callable[..., Any]] = None
    resolve_classifier_params_fn: Optional[Callable[..., Dict[str, Any]]] = None
    classifier_profile_label_fn: Optional[Callable[..., str]] = None
    default_training_window_months_for_combo_fn: Optional[Callable[[int, int, str], int]] = None
    training_window_bars_for_pair_fn: Optional[Callable[[str, int, int], int]] = None
    training_window_bars_from_months_fn: Optional[Callable[[int, int], int]] = None


@dataclass(frozen=True)
class RegimeForecastIOConfig:
    parquet_root: Path
    staging_root: Path
    state_root: Path
    log_root: Path
    scalar_root: Path
    ohlc_root: Path
    regime_label_root: Path
    parquet_compression: str = PARQUET_COMPRESSION
    parquet_row_group: int = PARQUET_ROW_GROUP
    log_fn: Callable[[str], None] = print


@dataclass(frozen=True)
class RegimeForecastPlan:
    intervals: tuple[int, ...]
    resolved_regime_ceiling_intervals: tuple[int, ...]
    horizon_minutes: tuple[int, ...]
    axes: tuple[str, ...]
    mode: str
    unit_workers: int


@dataclass(frozen=True)
class RegimeForecastTask:
    asset: str
    interval: int
    horizon_minutes: int
    horizon_bars: int


@dataclass(frozen=True)
class RegimeForecastTaskGroup:
    interval: int
    horizon_minutes: int
    horizon_bars: int
    assets: tuple[str, ...]


@dataclass(frozen=True)
class RegimeForecastWorkItem:
    asset: str
    unit_key: str
    interval: int
    horizon_minutes: int
    horizon_bars: int
    stop_ts: int
    prior_watermark_ts: Optional[int]
    prior_selection_timestamp: Optional[int]
    prior_selected_window_bars: Optional[int]
    prior_pending_ts: tuple[int, ...]


@dataclass(frozen=True)
class RegimeForecastWorkPlan:
    interval: int
    horizon_minutes: int
    horizon_bars: int
    assets: tuple[str, ...]
    work_items: tuple[RegimeForecastWorkItem, ...]
    completed_unit_keys: tuple[str, ...]
    reopened_unit_keys: tuple[str, ...]
    missing_source_unit_keys: tuple[str, ...]
    current_output_unit_keys: tuple[str, ...]


@dataclass(frozen=True)
class RegimeAppliedUnitResults:
    manifest_parts: tuple[dict, ...]
    rows_dropped_pre_floor_total: int
    ts_start_by_asset: dict[str, int]
    completed_unit_keys: tuple[str, ...]
    state_write_count: int


def dispatch_regime_work_items(
    work_items: Sequence[RegimeForecastWorkItem],
    *,
    unit_workers: int,
    compute_one_fn: Callable[[RegimeForecastWorkItem], Mapping[str, Any]],
) -> List[Mapping[str, Any]]:
    if int(unit_workers) <= 1 or len(work_items) <= 1:
        return [compute_one_fn(w) for w in work_items]

    unit_results: List[Mapping[str, Any]] = []
    max_workers = min(max(1, int(unit_workers)), len(work_items))
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(compute_one_fn, w) for w in work_items]
        for fut in as_completed(futures):
            unit_results.append(fut.result())
    return unit_results


def _csv_ints(raw: object, default: Sequence[int]) -> tuple[int, ...]:
    text = str(raw or "").strip()
    if not text:
        return tuple(int(x) for x in default)
    values = tuple(int(part.strip()) for part in text.split(",") if part.strip())
    return values or tuple(int(x) for x in default)


def _csv_axes(raw: object, default: Sequence[str]) -> tuple[str, ...]:
    text = str(raw or "").strip()
    axes = tuple(str(part.strip()) for part in text.split(",") if part.strip()) if text else tuple(str(axis) for axis in default)
    for axis in axes:
        axis_target(axis)
    return axes


def build_regime_arg_parser(spec: RegimeFamilyModuleSpec, *, description: Optional[str] = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=description or f"Walk-forward {spec.family_name} regime forecasting from canonical axis labels."
    )
    parser.add_argument("--profile", type=str, default=selected_profile())
    parser.add_argument("--intervals", type=str, default="")
    parser.add_argument("--assets", type=str, default="", help="Comma-delimited assets")
    parser.add_argument("--horizon-minutes", type=str, default="")
    parser.add_argument("--axes", type=str, default="", help="Comma-delimited Regime axes")
    parser.add_argument("--mode", type=str, default="incremental", choices=["incremental", "backfill"])
    parser.add_argument("--unit-workers", type=int, default=get_workers(spec.module_slug, "unit_workers", spec.default_unit_workers))
    return parser


def resolve_regime_run_plan(spec: RegimeFamilyModuleSpec, args: argparse.Namespace) -> RegimeForecastPlan:
    intervals = _csv_ints(getattr(args, "intervals", ""), spec.default_intervals)
    horizons = _csv_ints(getattr(args, "horizon_minutes", ""), spec.default_horizon_minutes)
    axes = _csv_axes(getattr(args, "axes", ""), spec.axes)
    return RegimeForecastPlan(
        intervals=tuple(int(interval) for interval in intervals),
        resolved_regime_ceiling_intervals=tuple(sorted({int(forecast_ceiling_interval(interval)) for interval in intervals})),
        horizon_minutes=tuple(int(horizon) for horizon in horizons),
        axes=tuple(str(axis) for axis in axes),
        mode=str(getattr(args, "mode", "incremental") or "incremental"),
        unit_workers=max(1, int(getattr(args, "unit_workers", spec.default_unit_workers) or spec.default_unit_workers)),
    )


def resolve_regime_assets(
    intervals: Sequence[int],
    assets_arg: str,
    *,
    list_assets_from_ohlcvt_fn: Callable[[int], Sequence[str]] = list_assets_from_ohlcvt,
    assets_present_in_features_fn: Callable[[int], Sequence[str]] = assets_present_in_features,
) -> tuple[str, ...]:
    if str(assets_arg).strip():
        return tuple(sorted({str(asset).strip() for asset in str(assets_arg).split(",") if str(asset).strip()}))
    assets: set[str] = set()
    for interval in intervals:
        assets.update(str(asset) for asset in list_assets_from_ohlcvt_fn(int(interval)))
        assets.update(str(asset) for asset in assets_present_in_features_fn(int(interval)))
    return tuple(sorted(assets))


def resolve_regime_tasks(
    intervals: Sequence[int],
    horizon_minutes_list: Sequence[int],
    *,
    assets_arg: str = "",
    list_assets_from_ohlcvt_fn: Callable[[int], Sequence[str]] = list_assets_from_ohlcvt,
    assets_present_in_features_fn: Callable[[int], Sequence[str]] = assets_present_in_features,
    horizon_bars_from_minutes_fn: Callable[[int, int], int] = _regime_horizon_bars_from_minutes,
) -> tuple[RegimeForecastTask, ...]:
    assets = resolve_regime_assets(
        intervals,
        assets_arg,
        list_assets_from_ohlcvt_fn=list_assets_from_ohlcvt_fn,
        assets_present_in_features_fn=assets_present_in_features_fn,
    )
    tasks: List[RegimeForecastTask] = []
    for asset in assets:
        for interval in intervals:
            for horizon_minutes in horizon_minutes_list:
                tasks.append(
                    RegimeForecastTask(
                        asset=str(asset),
                        interval=int(interval),
                        horizon_minutes=int(horizon_minutes),
                        horizon_bars=int(horizon_bars_from_minutes_fn(int(horizon_minutes), int(interval))),
                    )
                )
    return tuple(tasks)


def group_regime_tasks(tasks: Sequence[RegimeForecastTask]) -> tuple[RegimeForecastTaskGroup, ...]:
    groups: Dict[tuple[int, int, int], List[str]] = {}
    for task in tasks:
        key = (int(task.interval), int(task.horizon_minutes), int(task.horizon_bars))
        groups.setdefault(key, []).append(str(task.asset))
    out: List[RegimeForecastTaskGroup] = []
    for (interval, horizon_minutes, horizon_bars), assets in sorted(groups.items()):
        out.append(
            RegimeForecastTaskGroup(
                interval=int(interval),
                horizon_minutes=int(horizon_minutes),
                horizon_bars=int(horizon_bars),
                assets=tuple(sorted(set(str(asset) for asset in assets))),
            )
        )
    return tuple(out)


def plan_regime_group_work(
    group: RegimeForecastTaskGroup,
    *,
    spec: RegimeFamilyModuleSpec,
    family: str,
    domain: str,
    task: str,
    watermarks: Mapping[str, Any],
    pending: Mapping[str, Any],
    completed_unit_keys: Sequence[str],
    get_stop_ts_fn: Callable[[str, int], Optional[int]],
    get_horizon_output_tail_fn: Callable[[str, int, int], Optional[int]],
    pending_prune_buffer_bars: int = 0,
    pending_max_retain: Optional[int] = None,
) -> RegimeForecastWorkPlan:
    completed_in = {str(key) for key in completed_unit_keys}
    completed_out: set[str] = set()
    reopened: set[str] = set()
    missing_source: set[str] = set()
    current_output: set[str] = set()
    work_items: List[RegimeForecastWorkItem] = []
    units = watermarks.get("units", {}) if isinstance(watermarks, Mapping) else {}
    units = units if isinstance(units, Mapping) else {}

    interval = int(group.interval)
    horizon_minutes = int(group.horizon_minutes)
    horizon_bars = int(group.horizon_bars)
    buffer_seconds = int(interval) * 60 * max(0, int(pending_prune_buffer_bars))
    max_entries = int(pending_max_retain) if pending_max_retain is not None and int(pending_max_retain) > 0 else None

    for asset in group.assets:
        asset_name = str(asset)
        unit_key = regime_unit_key(
            family=family,
            domain=domain,
            task=task,
            horizon_minutes=horizon_minutes,
            asset=asset_name,
            interval=interval,
        )
        stop_ts = get_stop_ts_fn(asset_name, interval)
        if stop_ts is None:
            completed_out.add(unit_key)
            missing_source.add(unit_key)
            continue

        output_tail_ts = get_horizon_output_tail_fn(asset_name, interval, horizon_minutes)
        prior_pending_ts_raw = regime_pending_ts_for_unit(
            pending,
            family=family,
            domain=domain,
            task=task,
            horizon_minutes=horizon_minutes,
            asset=asset_name,
            interval=interval,
        )
        if output_tail_ts is not None and int(output_tail_ts) >= int(stop_ts) and not prior_pending_ts_raw:
            completed_out.add(unit_key)
            current_output.add(unit_key)
            continue
        if unit_key in completed_in:
            reopened.add(unit_key)

        unit_wm = units.get(unit_key, {}) if isinstance(units, Mapping) else {}
        unit_wm = unit_wm if isinstance(unit_wm, Mapping) else {}
        prior_pending_ts = tuple(
            prune_pending_ts(
                prior_pending_ts_raw,
                last_written_ts=(int(output_tail_ts) if output_tail_ts is not None else None),
                buffer_seconds=buffer_seconds,
                max_entries=max_entries,
            )
        )
        prior_selection_timestamp = unit_wm.get("selection_timestamp")
        prior_selected_window = unit_wm.get("selected_window_bars")
        work_items.append(
            RegimeForecastWorkItem(
                asset=asset_name,
                unit_key=unit_key,
                interval=interval,
                horizon_minutes=horizon_minutes,
                horizon_bars=horizon_bars,
                stop_ts=int(stop_ts),
                prior_watermark_ts=(int(output_tail_ts) if output_tail_ts is not None else None),
                prior_selection_timestamp=(
                    int(prior_selection_timestamp) if prior_selection_timestamp is not None else None
                ),
                prior_selected_window_bars=(int(prior_selected_window) if prior_selected_window is not None else None),
                prior_pending_ts=prior_pending_ts,
            )
        )

    return RegimeForecastWorkPlan(
        interval=interval,
        horizon_minutes=horizon_minutes,
        horizon_bars=horizon_bars,
        assets=tuple(str(asset) for asset in group.assets),
        work_items=tuple(work_items),
        completed_unit_keys=tuple(sorted(completed_out)),
        reopened_unit_keys=tuple(sorted(reopened)),
        missing_source_unit_keys=tuple(sorted(missing_source)),
        current_output_unit_keys=tuple(sorted(current_output)),
    )


def _positive_int_or_none(value: object) -> Optional[int]:
    try:
        out = int(value)  # type: ignore[arg-type]
    except Exception:
        return None
    return out if out > 0 else None


def _thread_override_from_env(spec: RegimeFamilyModuleSpec, env: Mapping[str, str]) -> tuple[Optional[int], Optional[str]]:
    for env_name in spec.thread_env_vars:
        raw = str(env.get(env_name, "") or "").strip()
        if not raw:
            continue
        resolved = _positive_int_or_none(raw)
        if resolved is not None:
            return int(resolved), str(env_name)
    return None, None


def resolve_regime_runtime_snapshot(
    spec: RegimeFamilyModuleSpec,
    args: argparse.Namespace,
    *,
    env: Optional[Mapping[str, str]] = None,
    output_root: Optional[Path] = None,
    diagnostic_root: Optional[Path] = None,
    writer_workers: int = 1,
    apply_thread_env: bool = True,
) -> Dict[str, Any]:
    source_env = env if env is not None else os.environ
    unit_workers = max(1, int(getattr(args, "unit_workers", spec.default_unit_workers) or spec.default_unit_workers))
    env_threads, env_name = _thread_override_from_env(spec, source_env)
    configured_threads = env_threads if env_threads is not None else get_model_threads(spec.module_slug, spec.default_model_threads)
    model_threads = cap_model_threads(
        workers=int(unit_workers),
        model_threads=int(configured_threads),
        max_logical_threads=int(spec.max_logical_threads),
    )
    thread_source = f"env:{env_name}" if env_name else "runtime_config"

    if apply_thread_env and hasattr(source_env, "__setitem__"):
        mutable_env = source_env  # type: ignore[assignment]
        for env_name_i in (*REGIME_THREAD_CAP_ENV_VARS, *tuple(spec.thread_env_vars)):
            mutable_env[str(env_name_i)] = str(int(model_threads))  # type: ignore[index]

    cpu_count = os.cpu_count() or 1
    cli_overrides = {
        "profile": str(getattr(args, "profile", selected_profile(env=source_env)) or ""),
        "intervals": str(getattr(args, "intervals", "") or ""),
        "assets": str(getattr(args, "assets", "") or ""),
        "horizon_minutes": str(getattr(args, "horizon_minutes", "") or ""),
        "axes": str(getattr(args, "axes", "") or ""),
        "mode": str(getattr(args, "mode", "") or ""),
        "unit_workers": int(unit_workers),
    }
    env_thread_caps = {
        str(name): str(source_env.get(str(name), "")) for name in (*REGIME_THREAD_CAP_ENV_VARS, *tuple(spec.thread_env_vars))
    }
    cpu_budget = int(unit_workers) * int(model_threads)
    return {
        "profile_name": str(cli_overrides["profile"]),
        "module_slug": str(spec.module_slug),
        "unit_workers": int(unit_workers),
        "model_threads": int(model_threads),
        "writer_workers": max(1, int(writer_workers)),
        "thread_param_name": str(spec.thread_param_name),
        "model_threads_source": thread_source,
        "configured_model_threads": int(configured_threads),
        "cpu_budget": int(cpu_budget),
        "cpu_count": int(cpu_count),
        "max_logical_threads": int(spec.max_logical_threads),
        "oversubscription_warning": bool(cpu_budget > int(spec.max_logical_threads)),
        "runtime_config_path": str(RUNTIME_CONFIG_PATH),
        "environment_thread_caps": env_thread_caps,
        "output_root": str(Path(output_root)) if output_root is not None else "",
        "diagnostic_root": str(Path(diagnostic_root)) if diagnostic_root is not None else "",
        "cli_overrides": cli_overrides,
    }


def regime_unit_meta(
    spec: RegimeFamilyModuleSpec,
    *,
    family: str,
    domain: str,
    task: str,
    horizon_minutes: int,
    horizon_bars: int,
    asset: str,
    interval: int,
) -> dict:
    return {
        "family": str(family),
        "domain": str(domain),
        "task": str(task),
        "module_slug": str(spec.module_slug),
        "prediction_prefix": str(spec.prediction_prefix),
        "axes": [str(axis) for axis in spec.axes],
        "horizon_minutes": int(horizon_minutes),
        "horizon_bars": int(horizon_bars),
        "asset": str(asset),
        "interval": int(interval),
        "requested_interval": int(interval),
        "resolved_regime_ceiling_interval": int(forecast_ceiling_interval(interval)),
    }


def regime_unit_key(
    *,
    family: str,
    domain: str,
    task: str,
    horizon_minutes: int,
    asset: str,
    interval: int,
) -> str:
    return make_unit_key(str(family), str(domain), str(task), int(horizon_minutes), str(asset), int(interval))


def regime_read_state(state_root: Path) -> tuple[dict, dict, dict]:
    return read_ml_state(Path(state_root), default_compaction={})


def regime_write_state(state_root: Path, watermarks: dict, pending: dict, progress: dict) -> None:
    root = Path(state_root)
    assert_write_allowed(root, "tabular regime model state root")
    write_ml_state(root, watermarks, pending, progress)


def regime_pending_prefix(
    *,
    family: str,
    domain: str,
    task: str,
    horizon_minutes: int,
    asset: str,
    interval: int,
) -> str:
    return f"{regime_unit_key(family=family, domain=domain, task=task, horizon_minutes=horizon_minutes, asset=asset, interval=interval)}|"


def regime_pending_ts_for_unit(
    pending: Mapping[str, Any],
    *,
    family: str,
    domain: str,
    task: str,
    horizon_minutes: int,
    asset: str,
    interval: int,
) -> List[int]:
    prefix = regime_pending_prefix(
        family=family,
        domain=domain,
        task=task,
        horizon_minutes=horizon_minutes,
        asset=asset,
        interval=interval,
    )
    entries = pending.get("entries", {}) if isinstance(pending, Mapping) else {}
    if not isinstance(entries, Mapping):
        return []
    out: List[int] = []
    for key, value in entries.items():
        if not str(key).startswith(prefix) or not isinstance(value, Mapping):
            continue
        try:
            out.append(int(value.get("ts")))
        except Exception:
            continue
    return sorted(set(out))


def regime_replace_pending_for_unit(
    pending: dict,
    spec: RegimeFamilyModuleSpec,
    *,
    family: str,
    domain: str,
    task: str,
    horizon_minutes: int,
    horizon_bars: int,
    asset: str,
    interval: int,
    ts_list: Sequence[int],
    reason: str = "tail_pending",
) -> None:
    unit_key = regime_unit_key(
        family=family,
        domain=domain,
        task=task,
        horizon_minutes=horizon_minutes,
        asset=asset,
        interval=interval,
    )
    unit_meta = regime_unit_meta(
        spec,
        family=family,
        domain=domain,
        task=task,
        horizon_minutes=int(horizon_minutes),
        horizon_bars=int(horizon_bars),
        asset=str(asset),
        interval=int(interval),
    )
    replace_pending_entries_for_unit(
        pending=pending,
        unit_key=unit_key,
        unit_meta=unit_meta,
        ts_list=ts_list,
        reason=str(reason),
    )


def apply_regime_unit_results(
    unit_results: Sequence[Mapping[str, Any]],
    *,
    spec: RegimeFamilyModuleSpec,
    family: str,
    domain: str,
    task: str,
    interval: int,
    horizon_minutes: int,
    horizon_bars: int,
    watermarks: dict,
    pending: dict,
    progress: dict,
    completed_unit_keys: Sequence[str],
    write_month_parts_fn: Callable[..., Sequence[Mapping[str, Any]]],
    write_state_fn: Callable[[dict, dict, dict], None],
    log_fn: Callable[[str], None] = print,
) -> RegimeAppliedUnitResults:
    completed = {str(key) for key in completed_unit_keys}
    manifest_parts: List[dict] = []
    rows_dropped_pre_floor_total = 0
    ts_start_by_asset: Dict[str, int] = {}
    state_write_count = 0

    for result in unit_results:
        ukey = str(result.get("ukey"))
        asset = str(result.get("asset"))
        rows_dropped_pre_floor_total += int(result.get("rows_dropped", 0) or 0)
        ts_start = result.get("ts_start")
        if ts_start is not None:
            ts_start_by_asset[asset] = int(ts_start)

        if result.get("empty"):
            completed.add(ukey)
            progress["completed"] = sorted(completed)
            progress["updated_at"] = datetime.now(timezone.utc).isoformat()
            write_state_fn(watermarks, pending, progress)
            state_write_count += 1
            continue

        parts = write_month_parts_fn(
            month_frames=result.get("asset_month_frames", {}),
            interval=int(interval),
            horizon_minutes=int(horizon_minutes),
        )
        manifest_parts.extend(dict(part) for part in parts)
        watermarks.setdefault("units", {})[ukey] = dict(result.get("unit_meta", {}))
        regime_replace_pending_for_unit(
            pending,
            spec,
            family=family,
            domain=domain,
            task=task,
            horizon_minutes=int(horizon_minutes),
            horizon_bars=int(horizon_bars),
            asset=asset,
            interval=int(interval),
            ts_list=result.get("pending_clean", []),
        )
        completed.add(ukey)
        progress["completed"] = sorted(completed)
        progress["updated_at"] = datetime.now(timezone.utc).isoformat()
        write_state_fn(watermarks, pending, progress)
        state_write_count += 1
        log_fn(
            f"{spec.log_prefix} asset={asset} k={int(interval)} h={int(horizon_minutes)}m({int(horizon_bars)}b) "
            f"stop_ts={int(result.get('stop_ts', 0) or 0)} rows_written={int(result.get('rows_written', 0) or 0)} "
            f"pending={len(result.get('pending_clean', []))}"
        )

    return RegimeAppliedUnitResults(
        manifest_parts=tuple(manifest_parts),
        rows_dropped_pre_floor_total=int(rows_dropped_pre_floor_total),
        ts_start_by_asset=dict(sorted(ts_start_by_asset.items())),
        completed_unit_keys=tuple(sorted(completed)),
        state_write_count=int(state_write_count),
    )


def regime_forecast_output_columns(
    spec: RegimeFamilyModuleSpec,
    horizon_minutes: int,
    *,
    compatibility_columns: Sequence[str] = (),
) -> List[str]:
    out = list(forecast_output_columns(spec.prediction_prefix, int(horizon_minutes)))
    for column in compatibility_columns:
        col = str(column)
        if col not in out:
            out.append(col)
    return out


def regime_forecast_output_part_path(
    output_root: Path,
    *,
    interval: int,
    asset: str,
    year: int,
    month: int,
) -> Path:
    return regime_forecast_part_path(Path(output_root), int(interval), str(asset), int(year), int(month))


def regime_family_output_root(io_config: RegimeForecastIOConfig, spec: RegimeFamilyModuleSpec) -> Path:
    return Path(io_config.parquet_root) / f"{spec.family_name}_Regimes"


def regime_forecast_output_max_ts(output_root: Path, *, interval: int, asset: str) -> Optional[int]:
    base = Path(output_root) / f"{int(interval)}" / f"asset={safe_partition_value(asset, field_name='asset')}"
    return partition_max_ts(base, ts_column="ts")


def regime_forecast_horizon_output_max_ts(
    output_root: Path,
    *,
    spec: RegimeFamilyModuleSpec,
    interval: int,
    asset: str,
    horizon_minutes: int,
) -> Optional[int]:
    base = Path(output_root) / f"{int(interval)}" / f"asset={safe_partition_value(asset, field_name='asset')}"
    expected_cols = list(forecast_output_columns(spec.prediction_prefix, int(horizon_minutes)))
    max_ts: Optional[int] = None
    for path in sorted(base.glob("year=*/month=*/part-000.parquet")):
        existing = regime_validated_existing_month_parquet(
            spec,
            path,
            asset=str(asset),
            interval=int(interval),
            horizon_minutes=int(horizon_minutes),
        )
        if existing.empty:
            continue
        missing_cols = [col for col in expected_cols if col not in existing.columns]
        if missing_cols:
            continue
        complete_mask = existing["asset"].astype(str).eq(str(asset))
        for col in expected_cols:
            complete_mask = complete_mask & existing[col].notna()
        if not bool(complete_mask.any()):
            continue
        ts_max = int(existing.loc[complete_mask, "ts"].astype("int64").max())
        max_ts = ts_max if max_ts is None else max(int(max_ts), ts_max)
    return max_ts


def regime_validated_existing_month_parquet(
    spec: RegimeFamilyModuleSpec,
    path: Path,
    *,
    asset: str,
    interval: int,
    horizon_minutes: int,
) -> pd.DataFrame:
    dst = Path(path)
    if not dst.exists():
        return pd.DataFrame(columns=["asset", "ts"])
    try:
        df = pd.read_parquet(dst)
    except Exception as exc:
        raise RuntimeError(
            f"{spec.log_prefix}[schema-error] unreadable Regime forecast parquet "
            f"asset={asset} k={int(interval)} h={int(horizon_minutes)}m path={dst}: {exc}"
        ) from exc
    if "ts" not in df.columns or "asset" not in df.columns:
        raise RuntimeError(
            f"{spec.log_prefix}[schema-error] missing key columns for Regime forecast parquet "
            f"asset={asset} k={int(interval)} h={int(horizon_minutes)}m path={dst} columns={list(df.columns)}"
        )
    if df.empty:
        return df.copy()
    out = df.copy()
    ts_num = pd.to_numeric(out["ts"], errors="coerce")
    if ts_num.isna().any():
        bad_rows = int(ts_num.isna().sum())
        raise RuntimeError(
            f"{spec.log_prefix}[schema-error] invalid ts values for Regime forecast parquet "
            f"asset={asset} k={int(interval)} h={int(horizon_minutes)}m path={dst} bad_rows={bad_rows}"
        )
    out["ts"] = ts_num.astype("int64")
    out["asset"] = out["asset"].astype(str)
    wrong_asset = out["asset"] != str(asset)
    if bool(wrong_asset.any()):
        bad_assets = sorted(set(str(x) for x in out.loc[wrong_asset, "asset"].astype(str).tolist()))
        raise RuntimeError(
            f"{spec.log_prefix}[schema-error] asset-partition mismatch for Regime forecast parquet "
            f"asset={asset} k={int(interval)} h={int(horizon_minutes)}m path={dst} bad_assets={bad_assets}"
        )
    duplicate_mask = out.duplicated(subset=["asset", "ts"], keep=False)
    if bool(duplicate_mask.any()):
        dup_count = int(duplicate_mask.sum())
        raise RuntimeError(
            f"{spec.log_prefix}[schema-error] duplicate key rows for Regime forecast parquet "
            f"asset={asset} k={int(interval)} h={int(horizon_minutes)}m path={dst} duplicate_rows={dup_count}"
        )
    return out.sort_values(["ts", "asset"]).reset_index(drop=True)


def regime_coalesce_keyed_frames(frames: Sequence[pd.DataFrame], expected_cols: Sequence[str]) -> pd.DataFrame:
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
        current["ts"] = pd.to_numeric(current["ts"], errors="coerce")
        current = current[current["ts"].notna()].copy()
        if current.empty:
            continue
        current["ts"] = current["ts"].astype("int64")
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


def _normalize_regime_parquet_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "asset" in out.columns:
        out["asset"] = out["asset"].astype(str).astype(object)
    if "ts" in out.columns:
        out["ts"] = pd.to_numeric(out["ts"], errors="coerce").astype("int64")
    return out


def _write_regime_parquet_with_contract(
    df: pd.DataFrame,
    dst: Path,
    *,
    compression: str,
    row_group_size: int,
) -> None:
    normalized = _normalize_regime_parquet_frame(df)
    normalized.to_parquet(
        dst,
        engine="pyarrow",
        compression=compression,
        index=False,
        row_group_size=int(row_group_size),
    )


def _trim_to_complete_regime_value_region(
    frame: pd.DataFrame,
    *,
    interval: int,
    asset: str,
    year: int,
    month: int,
    required_cols: Sequence[str],
) -> pd.DataFrame:
    if frame.empty:
        return frame
    value_cols = [str(col) for col in required_cols if str(col) in frame.columns]
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
                f"[regime-write-contract] asset={str(asset)} k={int(interval)} "
                f"context=asset-month={int(year):04d}-{int(month):02d}: incomplete row after complete head"
            ),
            first_bad_ts=first_bad_ts,
        )
    validate_strict_timegrid(
        complete_region["ts"],
        interval_min=int(interval),
        context=(
            f"[regime-write-contract] asset={str(asset)} k={int(interval)} "
            f"context=asset-month={int(year):04d}-{int(month):02d} complete-region"
        ),
    )
    return complete_region


def _validate_regime_completed_write_region(
    frame: pd.DataFrame,
    *,
    interval: int,
    asset: str,
    horizon_minutes: int,
    year: int,
    month: int,
    required_cols: Sequence[str],
    min_ts: Optional[int],
    max_ts: Optional[int],
) -> None:
    if frame.empty or min_ts is None or max_ts is None:
        return
    required = [str(col) for col in required_cols if str(col) in frame.columns]
    if not required:
        return
    ts_series = pd.to_numeric(frame["ts"], errors="coerce")
    mask = (ts_series >= int(min_ts)) & (ts_series <= int(max_ts))
    if not bool(mask.any()):
        return
    scoped = frame.loc[mask, ["ts", *required]].sort_values("ts").reset_index(drop=True)
    context = (
        f"[regime-write-contract] asset={str(asset)} k={int(interval)} "
        f"h={int(horizon_minutes)}m context=asset-month={int(year):04d}-{int(month):02d}"
    )
    validate_no_nan_columns(scoped, columns=required, context=context, ts_column="ts")
    validate_strict_timegrid(scoped["ts"], interval_min=int(interval), context=context)


def regime_write_month_parts(
    month_frames: Mapping[tuple[int, int], Sequence[pd.DataFrame]],
    *,
    output_root: Path,
    spec: RegimeFamilyModuleSpec,
    interval: int,
    horizon_minutes: int,
    compatibility_columns: Sequence[str] = (),
    parquet_compression: str = PARQUET_COMPRESSION,
    parquet_row_group: int = PARQUET_ROW_GROUP,
) -> List[dict]:
    if not month_frames:
        return []
    expected_cols = regime_forecast_output_columns(
        spec,
        int(horizon_minutes),
        compatibility_columns=compatibility_columns,
    )
    parts: List[dict] = []
    for (y, m), frames in sorted(month_frames.items()):
        if not frames:
            continue
        month_chunk = regime_coalesce_keyed_frames(list(frames), expected_cols)
        if month_chunk.empty or "asset" not in month_chunk.columns:
            continue
        for asset in sorted(set(str(x) for x in month_chunk["asset"].astype(str).tolist())):
            chunk = month_chunk[month_chunk["asset"].astype(str) == str(asset)].copy()
            if chunk.empty:
                continue
            min_input_ts = int(pd.to_numeric(chunk["ts"], errors="coerce").min())
            max_input_ts = int(pd.to_numeric(chunk["ts"], errors="coerce").max())
            dst = regime_forecast_output_part_path(
                output_root,
                interval=int(interval),
                asset=str(asset),
                year=int(y),
                month=int(m),
            )
            assert_write_allowed(dst, "tabular regime forecast parquet")
            existing = regime_validated_existing_month_parquet(
                spec,
                dst,
                asset=str(asset),
                interval=int(interval),
                horizon_minutes=int(horizon_minutes),
            )
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
            chunk = _trim_to_complete_regime_value_region(
                chunk,
                interval=int(interval),
                asset=str(asset),
                year=int(y),
                month=int(m),
                required_cols=expected_cols,
            )
            dst.parent.mkdir(parents=True, exist_ok=True)
            tmp = sibling_temp_path(dst, suffix=".parquet.tmp")
            assert_write_allowed(tmp, "tabular regime forecast parquet temp")
            if chunk.empty:
                _write_regime_parquet_with_contract(
                    chunk,
                    tmp,
                    compression=str(parquet_compression),
                    row_group_size=int(parquet_row_group),
                )
                atomic_replace(tmp, dst)
                continue
            _validate_regime_completed_write_region(
                chunk,
                interval=int(interval),
                asset=str(asset),
                horizon_minutes=int(horizon_minutes),
                year=int(y),
                month=int(m),
                required_cols=expected_cols,
                min_ts=min_input_ts,
                max_ts=max_input_ts,
            )
            _write_regime_parquet_with_contract(
                chunk,
                tmp,
                compression=str(parquet_compression),
                row_group_size=int(parquet_row_group),
            )
            atomic_replace(tmp, dst)
            parts.append(
                {
                    "path": str(dst),
                    "path_contract": "asset_part_000",
                    "rows": int(len(chunk)),
                    "asset": str(asset),
                    "assets": [str(asset)],
                    "interval": int(interval),
                    "horizon_minutes": int(horizon_minutes),
                    "year": int(y),
                    "month": int(m),
                    "min_ts": int(chunk["ts"].min()),
                    "max_ts": int(chunk["ts"].max()),
                }
            )
    return parts


def regime_manifest_base(
    spec: RegimeFamilyModuleSpec,
    plan: RegimeForecastPlan,
    *,
    run_id: str,
    family: str,
    domain: str,
    task: str,
    runtime_snapshot: Optional[Mapping[str, Any]] = None,
) -> dict:
    manifest = {
        "run_id": str(run_id),
        "mode": str(plan.mode),
        "family": str(family),
        "domain": str(domain),
        "task": str(task),
        "module_slug": str(spec.module_slug),
        "prediction_prefix": str(spec.prediction_prefix),
        "axes": list(plan.axes),
        "intervals": list(plan.intervals),
        "resolved_regime_ceiling_intervals": list(plan.resolved_regime_ceiling_intervals),
        "horizon_minutes": list(plan.horizon_minutes),
    }
    if runtime_snapshot is not None:
        manifest["runtime"] = dict(runtime_snapshot)
    return manifest


def _task_asset_and_horizon_bars(task: Any) -> tuple[str, int]:
    if isinstance(task, RegimeForecastTask):
        return str(task.asset), int(task.horizon_bars)
    if isinstance(task, Mapping):
        return str(task.get("asset")), int(task.get("horizon_bars"))
    return str(task[0]), int(task[3])


def regime_build_run_manifest(
    spec: RegimeFamilyModuleSpec,
    plan: RegimeForecastPlan,
    *,
    run_id: str,
    family: str,
    domain: str,
    task: str,
    runtime_snapshot: Optional[Mapping[str, Any]] = None,
    task_records: Sequence[Any] = (),
    ts_floor: int,
    ts_start_by_asset: Optional[Mapping[str, int]] = None,
    rows_dropped_pre_floor_total: int = 0,
    manifest_parts: Sequence[Mapping[str, Any]] = (),
    finished_at: Optional[str] = None,
) -> dict:
    manifest = regime_manifest_base(
        spec,
        plan,
        run_id=run_id,
        family=family,
        domain=domain,
        task=task,
        runtime_snapshot=runtime_snapshot,
    )
    task_pairs = [_task_asset_and_horizon_bars(row) for row in task_records]
    manifest.update(
        {
            "horizon_bars": sorted({int(horizon_bars) for _, horizon_bars in task_pairs}),
            "assets": sorted({str(asset) for asset, _ in task_pairs}),
            "ts_floor": int(ts_floor),
            "ts_start_by_asset": {
                str(asset): int(ts)
                for asset, ts in sorted((ts_start_by_asset or {}).items())
            },
            "rows_dropped_pre_floor_total": int(rows_dropped_pre_floor_total),
            "parts": [dict(part) for part in manifest_parts],
            "finished_at": str(finished_at or datetime.now(timezone.utc).isoformat()),
        }
    )
    return manifest


def regime_write_run_manifest(manifest_file: Path, manifest: Mapping[str, Any]) -> None:
    path = Path(manifest_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = sibling_temp_path(path)
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(dict(manifest), f, indent=2, sort_keys=True)
    atomic_replace(tmp, path)


def _sandbox_env_path(roots: SandboxOutputRoots, env_name: str, fallback: Path, kind: str, env: Mapping[str, str]) -> Path:
    raw = str(env.get(env_name, "") or "").strip()
    path = Path(raw) if raw else Path(fallback)
    assert_write_allowed(path, kind, roots=roots)
    return path


def _sandbox_resolution_env(env: Mapping[str, str]) -> Dict[str, str]:
    out = {str(key): str(value) for key, value in env.items()}
    raw_root = str(out.get("PIPELINE_SANDBOX_OUTPUT_ROOT", "") or "").strip()
    raw_pipeline_root = str(out.get("PIPELINE_ROOT", "") or "").strip()
    if raw_root and raw_pipeline_root:
        try:
            if Path(raw_root).expanduser().resolve() == Path(raw_pipeline_root).expanduser().resolve():
                out.pop("PIPELINE_ROOT", None)
        except Exception:
            pass
    return out


def resolve_regime_io_config(
    spec: RegimeFamilyModuleSpec,
    *,
    env: Optional[Mapping[str, str]] = None,
    profile: Optional[str] = None,
    log_fn: Callable[[str], None] = print,
) -> RegimeForecastIOConfig:
    source_env = env if env is not None else os.environ
    pipeline_profile = str(profile or selected_profile(env=source_env))
    path_config_env = dict(source_env)
    sandbox_roots = resolve_sandbox_output_roots(env=_sandbox_resolution_env(source_env))

    if sandbox_roots.enabled:
        log_root = _sandbox_env_path(sandbox_roots, "PIPELINE_SANDBOX_LOG_ROOT", sandbox_roots.log_root, "tabular regime log root", source_env)
        parquet_root = _sandbox_env_path(sandbox_roots, "PIPELINE_SANDBOX_PARQUET_ROOT", sandbox_roots.parquet_root, "tabular regime parquet root", source_env)
        state_base = _sandbox_env_path(sandbox_roots, "PIPELINE_SANDBOX_STATE_ROOT", sandbox_roots.state_root, "tabular regime state root", source_env)
        tmp_root = _sandbox_env_path(sandbox_roots, "PIPELINE_SANDBOX_TMP_ROOT", sandbox_roots.tmp_root, "tabular regime tmp root", source_env)
        state_root = state_base / "model_states" / spec.module_slug
        staging_root = tmp_root / f"{spec.module_slug}_stage"
        for path, kind in (
            (log_root, "tabular regime log root"),
            (state_root, "tabular regime model state root"),
            (staging_root, "tabular regime staging root"),
        ):
            assert_write_allowed(path, kind, roots=sandbox_roots)
    else:
        legacy_pipeline_root = Path(source_env["PIPELINE_ROOT"]) if str(source_env.get("PIPELINE_ROOT", "")).strip() else None
        log_root = resolve_path("log_root", profile=pipeline_profile, env=path_config_env, required=False) or (
            legacy_pipeline_root / "logs" if legacy_pipeline_root is not None else Path("logs")
        )
        state_base = resolve_path("state_root", profile=pipeline_profile, env=path_config_env, required=False) or (
            legacy_pipeline_root / "model_states" if legacy_pipeline_root is not None else Path("model_states")
        )
        tmp_root = resolve_path("tmp_root", profile=pipeline_profile, env=path_config_env, required=False) or (
            legacy_pipeline_root / "tmp" if legacy_pipeline_root is not None else Path("tmp")
        )
        parquet_root = Path(
            source_env.get(spec.parquet_root_env)
            or resolve_path("output_parquet_root", profile=pipeline_profile, env=path_config_env, required=False)
            or source_env.get("PIPELINE_PARQUET_ROOT", "parquet")
        )
        state_root = Path(state_base) / spec.module_slug
        staging_root = Path(tmp_root) / f"{spec.module_slug}_stage"

    scalar_root = Path(
        source_env.get("PIPELINE_SOURCE_FEATURES_ROOT")
        or source_env.get("PIPELINE_PARQUET_FEATURES_ROOT")
        or resolve_path("source_feature_root", profile=pipeline_profile, env=path_config_env, required=False)
        or SCALAR_PARQUET_ROOT
    )
    ohlc_root = Path(
        source_env.get("PIPELINE_SOURCE_OHLCVT_ROOT")
        or source_env.get("PIPELINE_SOURCE_PARQUET_ROOT")
        or resolve_path("source_ohlcvt_root", profile=pipeline_profile, env=path_config_env, required=False)
        or OHLCVT_PARQUET_ROOT
    )
    regime_label_root = Path(
        source_env.get("PIPELINE_SOURCE_REGIME_ROOT")
        or source_env.get("PIPELINE_PARQUET_REGIME_ROOT")
        or resolve_path("source_regime_root", profile=pipeline_profile, env=path_config_env, required=False)
        or parquet_root
    )

    for path in (log_root, state_root, staging_root):
        try:
            Path(path).mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

    return RegimeForecastIOConfig(
        parquet_root=Path(parquet_root),
        staging_root=Path(staging_root),
        state_root=Path(state_root),
        log_root=Path(log_root),
        scalar_root=Path(scalar_root),
        ohlc_root=Path(ohlc_root),
        regime_label_root=Path(regime_label_root),
        log_fn=log_fn,
    )
