from __future__ import annotations

import argparse
import os
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from src.forecasting.common.ohlcvt_source import read_ohlcvt
from src.forecasting.common.path_config import resolve_path, selected_profile
from src.features.numeric_forecast_profiles import normalize_refit_cadence, should_refit
from src.forecasting.ml.shared.numeric_runner_common import (
    NumericExistingProductionScope as ExistingProductionScope,
    NumericTestedProductionArtifactScope as TestedProductionArtifactScope,
    PlannedAssetWorkSpan,
    TailPlanningCache as _shared_TailPlanningCache,
    artifact_model_key as _shared_artifact_model_key,
    build_asset_shard_jobs as _shared_build_asset_shard_jobs,
    build_numeric_family_manifest_payload as _shared_build_numeric_family_manifest_payload,
    canonical_physical_io_config as _shared_canonical_physical_io_config,
    canonical_physical_naming as _shared_canonical_physical_naming,
    canonical_physical_output_tail_ts as _shared_canonical_physical_output_tail_ts,
    combo_window_map as _combo_window_map,
    compute_market_factor_cache as _shared_compute_market_factor_cache,
    discover_numeric_existing_production_scope as _shared_discover_numeric_existing_production_scope,
    discover_numeric_tested_production_artifact_scope as _shared_discover_numeric_tested_production_artifact_scope,
    default_refit_cadence_for_interval as _default_refit_cadence_for_interval,
    discover_existing_combo_specs_from_partitioned_output as _discover_existing_combo_specs_from_partitioned_output,
    discover_existing_combo_specs_from_canonical_physical_output as _shared_discover_existing_combo_specs_from_canonical_physical_output,
    discover_tested_production_artifact_payload as _shared_discover_tested_production_artifact_payload,
    build_horizon_group_shard_jobs as _shared_build_horizon_group_shard_jobs,
    execute_grouped_horizon_jobs as _shared_execute_grouped_horizon_jobs,
    execute_sharded_combo_jobs as _shared_execute_sharded_combo_jobs,
    finalized_origin_indices as _finalized_origin_indices,
    forward_fill_rows_to_edge as _shared_forward_fill_rows_to_edge,
    json_load_dict as _load_json_dict,
    load_stage3_combo_results as _load_stage3_combo_results,
    load_unit_state as _shared_load_unit_state,
    mark_prediction_row as _shared_mark_prediction_row,
    no_work_status_update as _shared_no_work_status_update,
    overlay_runtime_target_labels as _shared_overlay_runtime_target_labels,
    parse_combo_list as _parse_combo_list,
    parse_best_params as _parse_best_params,
    partition_assets as _partition_assets,
    partitioned_prediction_writer_loop as _shared_partitioned_prediction_writer_loop,
    plan_asset_work_span as _shared_plan_asset_work_span,
    planned_work_span_from_payload as _shared_planned_work_span_from_payload,
    planned_work_span_to_payload as _shared_planned_work_span_to_payload,
    prediction_eval_row as _shared_prediction_eval_row,
    project_root as _project_root,
    raise_writer_fatal as _shared_raise_writer_fatal,
    resolve_combo_fit_days as _shared_resolve_combo_fit_days,
    require_production_stream_scope_contract as _shared_require_production_stream_scope_contract,
    resolve_model_state_root as _shared_resolve_model_state_root,
    resolve_dispatch_slots as _resolve_dispatch_slots,
    resolve_min_env_int as _shared_resolve_min_env_int,
    resolve_production_stream_scope_contract as _shared_resolve_production_stream_scope_contract,
    resolve_progress_every_seconds as _shared_resolve_progress_every_seconds,
    resolve_runner_model_threads as _resolve_runner_model_threads,
    save_unit_state as _shared_save_unit_state,
    spill_rows_chunk as _shared_spill_rows_chunk,
    start_partitioned_prediction_writer as _shared_start_partitioned_prediction_writer,
    staging_root as _shared_staging_root,
    unit_state_path as _shared_unit_state_path,
    validate_combo_completion as _shared_validate_combo_completion,
    wait_for_writer_drain as _shared_wait_for_writer_drain,
    write_canonical_physical_prediction_month_frames as _shared_write_canonical_physical_prediction_month_frames,
    write_canonical_physical_predictions as _shared_write_canonical_physical_predictions,
)
from src.forecasting.ml.shared.numeric_runner_diagnostics import (
    append_diagnostic_event as _shared_append_diagnostic_event,
    diagnostics_file as _shared_diagnostics_file,
    reset_diagnostics_file as _shared_reset_diagnostics_file,
    resource_snapshot,
)
from src.forecasting.ml.shared.numeric_float_policy import DEFAULT_FLOAT_DTYPE, as_default_float_array
from src.forecasting.ml.shared.numeric_origin_windows import (
    build_production_origin_arrays,
    prepare_production_origin_window,
)
from src.forecasting.common.pipeline_parquet_utils import decide_range_from_disk_edges
from src.forecasting.common.forecast_family_core import (
    DEFAULT_BACKFILL_DAYS,
    DEFAULT_PARQUET_ROOT,
    DEFAULT_WORKERS,
    build_unit_context,
    default_common_roots,
    default_task_map,
    discover_edge_and_min,
    fit_window_start,
    forecast_output_tail_ts,
    load_assets,
    make_files,
    parse_quantiles,
    qcol,
    read_feature_window_columns,
    seasonality_info,
    task_target_col,
    utc_now_iso,
    write_json_atomic,
    write_partitioned_predictions,
    write_partitioned_prediction_month_frames,
)
from src.forecasting.common.runtime_config import DispatchPressureGuard, get_workers, log_resolved_runtime
from src.forecasting.common.sandbox_paths import SandboxOutputRoots, assert_write_allowed, resolve_sandbox_output_roots
from src.forecasting.common.stats_module_utils import NUMERIC_TASK_TO_TARGET_COLUMN
from src.forecasting.ml.bayesian.shared.bayesian_runtime_bootstrap import resolve_bayesian_model_threads
from src.forecasting.ml.bayesian.shared.bayesian_stage1_profile import resolve_execution_profile
from src.forecasting.ml.shared.numeric_forecast_targets import compute_future_labels

BayesianPredictFn = Callable[..., Tuple[Dict[float, float], Dict[str, Any]]]
BayesianBatchPredictFn = Callable[..., Sequence[Tuple[Dict[float, float], Dict[str, Any]]]]
_as_runtime_float_array = as_default_float_array
DEFAULT_PROGRESS_EVERY_SECONDS = 60
PROCESS_POOL_INIT_RETRIES = 3
PROCESS_POOL_INIT_RETRY_SECONDS = 1.0
DEFAULT_WORKER_ROW_BATCH_SIZE = 25000
DEFAULT_STATE_FLUSH_COMBO_CADENCE = 8
DEFAULT_ORIGIN_PREDICTION_BATCH_SIZE = 256


def _sandbox_resolution_env() -> Dict[str, str]:
    env = {str(key): str(value) for key, value in os.environ.items()}
    raw_root = str(env.get("PIPELINE_SANDBOX_OUTPUT_ROOT", "") or "").strip()
    raw_pipeline_root = str(env.get("PIPELINE_ROOT", "") or "").strip()
    if raw_root and raw_pipeline_root:
        try:
            if Path(raw_root).expanduser().resolve() == Path(raw_pipeline_root).expanduser().resolve():
                env.pop("PIPELINE_ROOT", None)
        except Exception:
            pass
    return env


def _sandbox_roots() -> SandboxOutputRoots:
    return resolve_sandbox_output_roots(env=_sandbox_resolution_env())


def _sandbox_env_path(roots: SandboxOutputRoots, env_name: str, fallback: Path, kind: str) -> Path:
    raw = str(os.getenv(env_name, "") or "").strip()
    path = Path(raw) if raw else Path(fallback)
    assert_write_allowed(path, kind, roots=roots)
    return path


def diagnostics_file(state_root: Path, branch: str) -> Path:
    roots = _sandbox_roots()
    if not roots.enabled:
        return _shared_diagnostics_file(state_root, branch)
    diagnostics_root = _sandbox_env_path(
        roots,
        "PIPELINE_SANDBOX_DIAGNOSTICS_ROOT",
        roots.diagnostics_root,
        "Bayesian diagnostics root",
    )
    return diagnostics_root / "bayesian_numeric_runner" / f"{str(branch)}_run_diagnostics.jsonl"


def append_diagnostic_event(path: Path, event: str, payload: Dict[str, Any], *, timestamp_fn: Any = None) -> None:
    assert_write_allowed(path, "Bayesian diagnostics JSONL", roots=_sandbox_roots())
    _shared_append_diagnostic_event(path, event, payload, timestamp_fn=timestamp_fn)


def reset_diagnostics_file(path: Path) -> None:
    assert_write_allowed(path, "Bayesian diagnostics JSONL reset", roots=_sandbox_roots())
    _shared_reset_diagnostics_file(path)


def _min_required_history_bars(*, interval_minutes: int, fit_days: int) -> int:
    fit_window_bars = int(fit_days) * 24 * 60 // max(1, int(interval_minutes))
    return max(48, int(fit_window_bars))


def _origin_prediction_batch_size(runtime_params: Dict[str, Any]) -> int:
    env = os.getenv("BAYES_NUMERIC_PREDICT_BATCH_SIZE")
    if env is not None:
        try:
            return max(1, int(env))
        except Exception:
            return DEFAULT_ORIGIN_PREDICTION_BATCH_SIZE
    return max(1, int(runtime_params.get("predict_batch_size", DEFAULT_ORIGIN_PREDICTION_BATCH_SIZE)))


@dataclass(frozen=True)
class BayesianNumericModuleSpec:
    module_tag: str
    model_id: str
    model_version: str
    family_root_name: str
    family_root_env: str
    default_intervals: Sequence[int]
    default_horizons: Sequence[int]
    default_tasks: Sequence[str]
    predict_fn: BayesianPredictFn
    predict_batch_fn: Optional[BayesianBatchPredictFn] = None
    use_seasonality: bool = False
    needs_dynamic_features: bool = False
    needs_factor_cache: bool = False
    dynamic_feature_candidates: Sequence[str] = ()
    model_params: Dict[str, Any] = field(default_factory=dict)
    runtime_params: Dict[str, Any] = field(default_factory=dict)
    resolve_model_params_fn: Callable[..., Dict[str, Any]] = field(default_factory=lambda: (lambda **_: {}))
    resolve_default_combo_specs_fn: Callable[[], List[Tuple[int, int, str]]] = field(default_factory=lambda: (lambda: []))
    progress_seconds_env: str = "BAYES_NUMERIC_PROGRESS_SECONDS"


def _state_root(forecast_root: Path, module_tag: str) -> Path:
    roots = _sandbox_roots()
    if not roots.enabled:
        return _shared_resolve_model_state_root(forecast_root, module_tag)
    state_root = _sandbox_env_path(
        roots,
        "PIPELINE_SANDBOX_STATE_ROOT",
        roots.state_root,
        "Bayesian model state root",
    ) / "bayesian_numeric_runner" / str(module_tag)
    assert_write_allowed(state_root, "Bayesian model state root", roots=roots)
    return state_root.resolve()


def _load_unit_state(*, state_root: Path, asset: str, interval: int, horizon_minutes: int, task: str) -> Dict[str, Any]:
    assert_write_allowed(
        _shared_unit_state_path(
            state_root=state_root,
            asset=asset,
            interval=interval,
            horizon_minutes=horizon_minutes,
            task=task,
        ),
        "Bayesian model state read",
        roots=_sandbox_roots(),
    )
    return _shared_load_unit_state(
        state_root=state_root,
        asset=asset,
        interval=interval,
        horizon_minutes=horizon_minutes,
        task=task,
    )


def _save_unit_state(*, state_root: Path, asset: str, interval: int, horizon_minutes: int, task: str, state: Dict[str, Any]) -> None:
    roots = _sandbox_roots()
    path = _shared_unit_state_path(
        state_root=state_root,
        asset=asset,
        interval=interval,
        horizon_minutes=horizon_minutes,
        task=task,
    )
    assert_write_allowed(path, "Bayesian model state", roots=roots)
    assert_write_allowed(path.with_name(f"{path.name}.{os.getpid()}.tmp"), "Bayesian model state temp", roots=roots)
    _shared_save_unit_state(
        state_root=state_root,
        asset=asset,
        interval=interval,
        horizon_minutes=horizon_minutes,
        task=task,
        state=state,
    )

def _resolve_seasonality_period(parquet_root: Path, interval: int, asset: str, enabled: bool) -> Dict[str, Any]:
    if not enabled:
        return {"seasonality_mode": "none", "seasonality_usable": False, "seasonality_period_bars": None, "seasonality_label": "none", "seasonality_path": None}
    return seasonality_info(parquet_root=parquet_root, interval_minutes=int(interval), asset=str(asset))


_overlay_runtime_target_labels = lambda **kwargs: _shared_overlay_runtime_target_labels(
    **kwargs,
    read_ohlcvt_fn=read_ohlcvt,
    compute_future_labels_fn=compute_future_labels,
)


def _origin_row_base(ctx: Any, origin_ts: int, train_start_ts: int, train_end_ts: int, seas: Dict[str, Any]) -> Dict[str, Any]:
    return {"asset": str(ctx.asset), "ts": int(origin_ts), "interval_min": int(ctx.interval_minutes), "horizon_min": int(ctx.horizon_minutes), "task": str(ctx.task), "run_id": str(ctx.run_id), "model_id": str(ctx.model_id), "model_version": str(ctx.model_version), "train_start_ts": int(train_start_ts), "train_end_ts": int(train_end_ts), "seasonality_mode": str(seas.get("seasonality_mode", "none")), "seasonality_label": seas.get("seasonality_label")}


def _plan_asset_work_span(**kwargs: Any) -> Optional[PlannedAssetWorkSpan]:
    canonical_io_config = kwargs.pop("canonical_io_config", None)
    if canonical_io_config is not None:
        kwargs.setdefault(
            "tail_cache_namespace",
            f"canonical_physical:{canonical_io_config.naming.forecast_table_tag}:forecast",
        )
    tail_fn = (
        (lambda **tail_kwargs: _shared_canonical_physical_output_tail_ts(
            io_config=canonical_io_config,
            interval_minutes=int(tail_kwargs["interval_minutes"]),
            task=str(tail_kwargs["task"]),
            horizon_minutes=int(tail_kwargs["horizon_minutes"]),
            asset=str(tail_kwargs["asset"]),
            include_recompute=bool(tail_kwargs.get("include_recompute", False)),
        ))
        if canonical_io_config is not None
        else forecast_output_tail_ts
    )
    return _shared_plan_asset_work_span(
        **kwargs,
        discover_edge_and_min_fn=discover_edge_and_min,
        forecast_output_tail_ts_fn=tail_fn,
        decide_range_from_disk_edges_fn=decide_range_from_disk_edges,
        fit_window_start_fn=fit_window_start,
    )


def _compute_factor_cache(**kwargs: Any) -> pd.DataFrame:
    return _shared_compute_market_factor_cache(
        **kwargs,
        target_col=NUMERIC_TASK_TO_TARGET_COLUMN[str(kwargs["task"])],
        read_feature_window_columns_fn=read_feature_window_columns,
        overlay_runtime_target_labels_fn=_overlay_runtime_target_labels,
    )


def _resolve_progress_every_seconds_for_spec(spec: BayesianNumericModuleSpec) -> int:
    return _shared_resolve_progress_every_seconds(
        str(spec.progress_seconds_env),
        default_seconds=DEFAULT_PROGRESS_EVERY_SECONDS,
    )


def _manifest_snapshot_payload(
    *,
    run_id: str,
    spec: BayesianNumericModuleSpec,
    combos: Sequence[Tuple[int, int, str]],
    combo_plans: Sequence[Dict[str, Any]],
    worker_budget: int,
    dispatch_slots: int,
    resolved_model_threads: int,
    args: argparse.Namespace,
    requested_fit_days: Optional[int],
    requested_refit_cadence: str,
    quantiles: Sequence[float],
    runtime_params: Dict[str, Any],
    staging_root: Path,
    state_root: Path,
    tested_scope: Optional[TestedProductionArtifactScope],
    production_scope: Optional[ExistingProductionScope],
    manifest_parts: Sequence[Dict[str, Any]],
    skipped_units: Dict[str, Dict[str, Any]],
    combo_count: int,
    job_shard_count: int,
    writer_stats: Optional[Dict[str, Any]] = None,
    finished_at: Optional[str] = None,
) -> Dict[str, Any]:
    return _shared_build_numeric_family_manifest_payload(
        run_id=run_id,
        module_tag=spec.module_tag,
        model_id=spec.model_id,
        model_version=spec.model_version,
        family_name="Stats_Bayes",
        combos=combos,
        combo_plans=combo_plans,
        worker_budget=worker_budget,
        dispatch_slots=dispatch_slots,
        model_threads=resolved_model_threads,
        mode=str(args.mode),
        backfill_days=int(args.backfill_days),
        requested_fit_days=requested_fit_days,
        quantiles=quantiles,
        runtime_params=runtime_params,
        requested_refit_cadence=requested_refit_cadence,
        predict_latest_only=bool(args.predict_latest_only),
        force=bool(args.force),
        overwrite_months=str(args.overwrite_months),
        staging_root=staging_root,
        state_root=state_root,
        tested_scope=tested_scope,
        production_scope=production_scope,
        manifest_parts=manifest_parts,
        skipped_units=skipped_units,
        combo_count=combo_count,
        job_shard_count=job_shard_count,
        extra_fields={
            "planned_work_items_total": int(sum(int(plan.get("planned_assets_with_work", 0) or 0) for plan in combo_plans)),
            "no_work_combo_count": int(sum(1 for plan in combo_plans if int(plan.get("planned_assets_with_work", 0) or 0) == 0)),
            "writer_stats": dict(writer_stats or {}),
            "diagnostics_jsonl": str(diagnostics_file(state_root, spec.module_tag)),
        },
        finished_at=finished_at,
    )


def _writer_loop(*, forecast_root: Path, write_queue: Any, writer_state: Dict[str, Any], canonical_io_config: Any = None) -> None:
    if canonical_io_config is not None:
        def _write_month_frames(**kwargs: Any) -> List[Dict[str, Any]]:
            store = "eval" if Path(kwargs.get("out_root", "")).name == "eval" else "forecast"
            return _shared_write_canonical_physical_prediction_month_frames(
                io_config=canonical_io_config,
                store=store,
                **{k: v for k, v in kwargs.items() if k != "out_root"},
            )

        def _write_predictions(**kwargs: Any) -> List[Dict[str, Any]]:
            store = "eval" if Path(kwargs.get("out_root", "")).name == "eval" else "forecast"
            return _shared_write_canonical_physical_predictions(
                io_config=canonical_io_config,
                store=store,
                **kwargs,
            )

        _shared_partitioned_prediction_writer_loop(
            forecast_root=forecast_root,
            write_queue=write_queue,
            writer_state=writer_state,
            write_partitioned_predictions_fn=_write_predictions,
            write_partitioned_prediction_month_frames_fn=_write_month_frames,
        )
        return

    _shared_partitioned_prediction_writer_loop(
        forecast_root=forecast_root,
        write_queue=write_queue,
        writer_state=writer_state,
        write_partitioned_predictions_fn=write_partitioned_predictions,
        write_partitioned_prediction_month_frames_fn=write_partitioned_prediction_month_frames,
    )


def _raise_writer_fatal(writer_state: Dict[str, Any], family_label: str) -> None:
    _shared_raise_writer_fatal(writer_state, family_label)


def _wait_for_writer_drain(*, writer_queue: queue.Queue, writer_thread: threading.Thread, writer_state: Dict[str, Any], family_label: str) -> None:
    _shared_wait_for_writer_drain(
        writer_queue=writer_queue,
        writer_thread=writer_thread,
        writer_state=writer_state,
        family_label=family_label,
    )


def _worker_row_batch_size() -> int:
    return _shared_resolve_min_env_int(
        "BAYESIAN_NUMERIC_WORKER_ROW_BATCH_SIZE",
        default_value=DEFAULT_WORKER_ROW_BATCH_SIZE,
        minimum=1000,
    )


def _state_flush_combo_cadence() -> int:
    return _shared_resolve_min_env_int(
        "BAYESIAN_NUMERIC_STATE_FLUSH_COMBOS",
        default_value=DEFAULT_STATE_FLUSH_COMBO_CADENCE,
        minimum=1,
    )


def _spill_rows_chunk(
    *,
    rows: Sequence[Dict[str, Any]],
    staging_root: Path,
    module_tag: str,
    interval: int,
    horizon_minutes: int,
    task: str,
    asset: str,
) -> str:
    roots = _sandbox_roots()
    shard_rows_root = Path(staging_root) / "worker_row_batches"
    assert_write_allowed(shard_rows_root, "Bayesian worker row-batch staging root", roots=roots)
    return _shared_spill_rows_chunk(
        rows=rows,
        staging_root=staging_root,
        module_tag=module_tag,
        interval=interval,
        horizon_minutes=horizon_minutes,
        task=task,
        asset=asset,
    )


def _persist_factor_cache(
    *,
    factor_cache: pd.DataFrame,
    staging_root: Path,
    module_tag: str,
    interval: int,
    horizon_minutes: int,
    task: str,
    start_ts: int,
    end_ts: int,
) -> Optional[str]:
    if factor_cache.empty:
        return None
    roots = _sandbox_roots()
    cache_root = staging_root / "factor_cache"
    assert_write_allowed(cache_root, "Bayesian factor cache root", roots=roots)
    cache_root.mkdir(parents=True, exist_ok=True)
    path = cache_root / f"{module_tag}_{interval}_{horizon_minutes}_{task}_{start_ts}_{end_ts}.parquet"
    assert_write_allowed(path, "Bayesian factor cache parquet", roots=roots)
    factor_cache.to_parquet(path, engine="pyarrow", compression="zstd", index=False)
    return str(path)


def _load_factor_cache(
    factor_cache_path: Optional[str],
) -> pd.DataFrame:
    if not factor_cache_path:
        return pd.DataFrame(columns=["ts", "market_factor"])
    path = Path(str(factor_cache_path))
    assert_write_allowed(path, "Bayesian factor cache read", roots=_sandbox_roots())
    if not path.exists():
        return pd.DataFrame(columns=["ts", "market_factor"])
    return pd.read_parquet(path)


def _factor_cache_map_from_frame(factor_cache: pd.DataFrame) -> Dict[int, float]:
    if factor_cache.empty:
        return {}
    factor_map: Dict[int, float] = {}
    for t, fval in zip(
        pd.to_numeric(factor_cache["ts"], errors="coerce"),
        pd.to_numeric(factor_cache["market_factor"], errors="coerce"),
    ):
        if pd.isna(t) or pd.isna(fval):
            continue
        factor_map[int(t)] = float(fval)
    return factor_map


def _factor_cache_artifact_path(
    *,
    staging_root: Path,
    module_tag: str,
    interval: int,
    horizon_minutes: int,
    task: str,
    start_ts: int,
    end_ts: int,
) -> Path:
    path = (
        staging_root
        / "factor_cache"
        / f"{module_tag}_{interval}_{horizon_minutes}_{task}_{start_ts}_{end_ts}.parquet"
    )
    assert_write_allowed(path, "Bayesian factor cache artifact path", roots=_sandbox_roots())
    return path


def _run_bayesian_asset(payload: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    model_threads = payload.get("model_threads")
    if model_threads is not None:
        os.environ["BAYESIAN_NUMERIC_MODEL_THREADS"] = str(max(1, int(model_threads)))
    from src.forecasting.ml.bayesian.shared.bayesian_runtime_bootstrap import configure_bayesian_thread_env

    configure_bayesian_thread_env()
    spec: BayesianNumericModuleSpec = payload["spec"]
    asset = str(payload["asset"])
    interval = int(payload["interval"])
    hm = int(payload["horizon_minutes"])
    task = str(payload["task"])
    target_col = str(payload["target_col"])
    run_id = str(payload["run_id"])
    quantiles = [float(q) for q in payload["quantiles"]]
    args_force = bool(payload["force"])
    args_mode = str(payload["mode"])
    args_backfill_days = int(payload["backfill_days"])
    args_fit_days = int(payload["fit_days"])
    args_predict_latest_only = bool(payload["predict_latest_only"])
    parquet_root = Path(str(payload["parquet_root"]))
    feature_root = Path(str(payload["feature_root"]))
    forecast_root = Path(str(payload["forecast_root"]))
    staging_root = Path(str(payload["staging_root"]))
    state_root = Path(str(payload["state_root"]))
    refit_cadence = str(payload["refit_cadence"])
    combo_model_params = dict(payload["combo_model_params"])
    runtime_params = dict(payload["runtime_params"])
    hb = int(payload["hb"])
    progress_every_seconds = max(5, int(payload.get("progress_every_seconds", DEFAULT_PROGRESS_EVERY_SECONDS)))
    allow_partial = bool(payload.get("allow_partial", False))
    combo_profile_payload = payload.get("combo_profile")
    combo_profile = None
    if isinstance(combo_profile_payload, dict):
        feature_profile_json = payload.get("feature_profile_json")
        combo_profile = resolve_execution_profile(
            Path(str(feature_profile_json)) if feature_profile_json else None,
            interval=int(interval),
            horizon=int(hm),
            task=str(task),
            dynamic_feature_candidates=spec.dynamic_feature_candidates,
            needs_dynamic_features=bool(spec.needs_dynamic_features),
            use_seasonality=bool(spec.use_seasonality),
        )
    factor_cache_map_payload = payload.get("_factor_cache_map")
    factor_cache_map = (
        {
            int(ts): float(value)
            for ts, value in dict(factor_cache_map_payload).items()
        }
        if isinstance(factor_cache_map_payload, dict)
        else None
    )
    factor_cache = (
        pd.DataFrame(columns=["ts", "market_factor"])
        if factor_cache_map is not None
        else _load_factor_cache(payload.get("factor_cache_path"))
    )

    ctx = build_unit_context(family="Stats_Bayes", domain="Numerics", module_tag=spec.module_tag, model_id=spec.model_id, model_version=spec.model_version, interval_minutes=int(interval), horizon_minutes=int(hm), task=str(task), target_col=str(target_col), asset=str(asset), run_id=run_id)
    ukey = ctx.ukey
    work_span_payload = payload.get("work_span")
    work_span = (
        _shared_planned_work_span_from_payload(work_span_payload)
        if isinstance(work_span_payload, dict)
        else _plan_asset_work_span(
            asset=str(asset),
            interval=int(interval),
            horizon_minutes=int(hm),
            task=str(task),
            mode=str(args_mode),
            backfill_days=int(args_backfill_days),
            force=bool(args_force),
            fit_days=int(args_fit_days),
            forecast_root=forecast_root,
        )
    )
    if work_span is None:
        return ukey, _shared_no_work_status_update(
            asset=str(asset),
            interval=int(interval),
            horizon_minutes=int(hm),
            discover_edge_and_min_fn=discover_edge_and_min,
        )
    edge_ts = int(work_span.edge_ts)
    target_tail = int(work_span.target_tail_ts)
    start_for_mode = int(work_span.start_ts)
    read_start = int(work_span.read_start_ts)
    dst_tail = (int(work_span.dst_tail_ts) if work_span.dst_tail_ts is not None else None)
    columns = [str(target_col)]
    if combo_profile is not None:
        columns.extend(str(c) for c in combo_profile.selected_dynamic_feature_columns)
    elif spec.needs_dynamic_features:
        columns.extend(str(c) for c in spec.dynamic_feature_candidates)
    preloaded_frame = payload.get("_preloaded_frame")
    if isinstance(preloaded_frame, pd.DataFrame):
        needed_cols = list(dict.fromkeys(["ts", "asset", *columns]))
        df = preloaded_frame.copy()
        for col in needed_cols:
            if col not in df.columns:
                df[col] = np.nan
        df = df.loc[:, needed_cols].copy()
        df["ts"] = pd.to_numeric(df["ts"], errors="coerce")
        df = df.dropna(subset=["ts"]).copy()
        if not df.empty:
            df["ts"] = df["ts"].astype("int64")
            df = df[(df["ts"] >= int(read_start)) & (df["ts"] <= int(edge_ts))].copy()
    else:
        df = read_feature_window_columns(root=feature_root, interval_minutes=int(interval), asset=str(asset), columns=columns, start_ts=int(read_start), end_ts=int(edge_ts))
        df = _overlay_runtime_target_labels(
            frame=df,
            parquet_root=parquet_root,
            asset=str(asset),
            interval=int(interval),
            horizon_minutes=int(hm),
            target_col=str(target_col),
            start_ts=int(read_start),
            end_ts=int(edge_ts),
        )
    if df.empty:
        return ukey, {"status": "skipped", "reason": "missing_feature_rows", "edge_ts": int(edge_ts)}
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
    seas = _resolve_seasonality_period(
        parquet_root=parquet_root,
        interval=int(interval),
        asset=str(asset),
        enabled=(combo_profile.use_seasonality if combo_profile is not None else spec.use_seasonality),
    )
    period_bars = seas.get("seasonality_period_bars") if seas.get("seasonality_usable") else None
    persisted_state = _load_unit_state(
        state_root=state_root,
        asset=str(asset),
        interval=int(interval),
        horizon_minutes=int(hm),
        task=str(task),
    )
    rows: List[Dict[str, Any]] = []
    staged_rows_paths: List[str] = []
    eval_rows: List[Dict[str, Any]] = []
    eval_staged_rows_paths: List[str] = []
    row_count = 0
    generated_row_count = 0
    eval_row_count = 0
    actual_tail_ts: Optional[int] = None
    last_row: Optional[Dict[str, Any]] = None
    last_eval_row: Optional[Dict[str, Any]] = None
    failed_origins = 0
    insufficient_history_origins = 0
    fit_meta: Dict[str, Any] = {}
    last_refit_ts = (
        int(persisted_state["last_refit_ts"])
        if persisted_state.get("last_refit_ts") is not None
        else None
    )
    refit_count = int(persisted_state.get("refit_count", 0) or 0)
    df = df.sort_values("ts").reset_index(drop=True)
    if combo_profile is not None:
        feat_cols = [str(c) for c in combo_profile.selected_dynamic_feature_columns if str(c) in df.columns]
    else:
        feat_cols = [c for c in spec.dynamic_feature_candidates if c in df.columns]
    use_dynamic_features = combo_profile.use_dynamic_features if combo_profile is not None else spec.needs_dynamic_features
    factor_map: Optional[Dict[int, float]] = None
    if spec.needs_factor_cache:
        factor_map = factor_cache_map if factor_cache_map is not None else _factor_cache_map_from_frame(factor_cache)
    origin_arrays = build_production_origin_arrays(
        frame=df,
        target_col=str(target_col),
        selected_feature_columns=feat_cols,
        use_dynamic_features=bool(use_dynamic_features),
        as_float_array=_as_runtime_float_array,
        float_dtype=DEFAULT_FLOAT_DTYPE,
        factor_map=factor_map,
        needs_factor_cache=bool(spec.needs_factor_cache),
        coerce_ts=False,
    )
    ts_vec = origin_arrays.ts_vec
    y_vec = origin_arrays.y_vec
    valid_target_idx = origin_arrays.valid_target_idx
    effective_start_ts = int(start_for_mode)
    min_history_bars = _min_required_history_bars(interval_minutes=int(interval), fit_days=int(args_fit_days))
    if int(valid_target_idx.size) >= int(min_history_bars):
        history_ready_idx = int(valid_target_idx[int(min_history_bars) - 1])
        effective_start_ts = max(int(effective_start_ts), int(ts_vec[history_ready_idx]))
    finalized_idx = _finalized_origin_indices(ts_vec, start_ts=int(effective_start_ts), end_ts=int(target_tail), predict_latest_only=bool(args_predict_latest_only))
    if not finalized_idx:
        return ukey, {"status": "done", "reason": "no_origins", "edge_ts": int(edge_ts), "rows": []}
    started_at = time.monotonic()
    last_progress_log = started_at
    row_batch_size = _worker_row_batch_size()
    origin_batch_size = _origin_prediction_batch_size(runtime_params)
    predict_batch_fn = spec.predict_batch_fn
    if predict_batch_fn is None:
        def predict_batch_fn(*, origin_batch: Sequence[Dict[str, Any]]) -> Sequence[Tuple[Dict[float, float], Dict[str, Any]]]:
            return [spec.predict_fn(**item) for item in origin_batch]

    origin_batch: List[Dict[str, Any]] = []

    def _flush_origin_batch() -> None:
        nonlocal rows, eval_rows, row_count, generated_row_count, eval_row_count
        nonlocal actual_tail_ts, last_row, last_eval_row, fit_meta, failed_origins
        if not origin_batch:
            return
        batch = list(origin_batch)
        origin_batch.clear()
        try:
            batch_results = list(predict_batch_fn(origin_batch=[item["predict_input"] for item in batch]))
        except Exception as exc:
            if allow_partial:
                failed_origins += int(len(batch))
                return
            first_origin = int(batch[0]["origin_ts"])
            raise RuntimeError(f"[{spec.module_tag}] origin batch failure asset={asset} interval={interval} horizon={hm} task={task} first_origin_ts={first_origin} batch_size={len(batch)} reason=predict_batch_fn_failed: {exc}") from exc
        if len(batch_results) != len(batch):
            raise RuntimeError(f"[{spec.module_tag}] predict_batch_fn returned {len(batch_results)} results for {len(batch)} origins")
        for item, result in zip(batch, batch_results):
            qvals, meta = result
            origin_ts = int(item["origin_ts"])
            row = _origin_row_base(ctx=ctx, origin_ts=origin_ts, train_start_ts=int(item["train_start_ts"]), train_end_ts=int(item["train_end_ts"]), seas=seas)
            for q in quantiles:
                row[qcol(float(q))] = float(qvals.get(float(q), np.nan))
            row = _shared_mark_prediction_row(row, forward_filled=False, needs_recompute=False)
            rows.append(row)
            row_count += 1
            generated_row_count += 1
            actual_tail_ts = origin_ts
            last_row = dict(row)
            actual_value = float(item["actual_value"])
            if np.isfinite(actual_value):
                eval_row = _shared_prediction_eval_row(
                    prediction_row=row,
                    actual_value=actual_value,
                    target_col=str(target_col),
                )
                eval_rows.append(eval_row)
                eval_row_count += 1
                last_eval_row = dict(eval_row)
            fit_meta = {
                "fit_bars": int(item["fit_bars"]),
                "quantiles": [float(q) for q in quantiles],
                "seed": 42,
                "seasonality_mode": seas.get("seasonality_mode"),
                "seasonality_label": seas.get("seasonality_label"),
                "seasonality_period_bars": seas.get("seasonality_period_bars"),
                "refit_cadence": str(refit_cadence),
                "refit_count": int(item["refit_count"]),
                "last_refit_ts": int(item["last_refit_ts"]) if item["last_refit_ts"] is not None else None,
                "origin_prediction_batch_size": int(origin_batch_size),
                "origin_prediction_batch_mode": "profile_batch" if spec.predict_batch_fn is not None else "compat_scalar_adapter",
                **(meta if isinstance(meta, dict) else {}),
            }
            if len(rows) >= int(row_batch_size):
                staged_rows_paths.append(
                    _spill_rows_chunk(
                        rows=rows,
                        staging_root=staging_root,
                        module_tag=spec.module_tag,
                        interval=int(interval),
                        horizon_minutes=int(hm),
                        task=str(task),
                        asset=str(asset),
                    )
                )
                rows = []
            if len(eval_rows) >= int(row_batch_size):
                eval_staged_rows_paths.append(
                    _spill_rows_chunk(
                        rows=eval_rows,
                        staging_root=staging_root,
                        module_tag=f"{spec.module_tag}_eval",
                        interval=int(interval),
                        horizon_minutes=int(hm),
                        task=str(task),
                        asset=str(asset),
                    )
                )
                eval_rows = []

    for idx_origin, idx in enumerate(finalized_idx):
        origin_ts = int(ts_vec[idx])
        fit_bars = max(64, int(min_history_bars))
        window_result = prepare_production_origin_window(
            arrays=origin_arrays,
            idx=int(idx),
            min_history_bars=int(min_history_bars),
            history_bars=int(fit_bars),
            use_dynamic_features=bool(use_dynamic_features),
            needs_factor_cache=bool(spec.needs_factor_cache),
            as_float_array=_as_runtime_float_array,
        )
        if window_result.window is None:
            reason = str(window_result.reason)
            failed_origins += 1
            if reason == "insufficient_valid_history":
                insufficient_history_origins += 1
                continue
            if allow_partial:
                continue
            raise RuntimeError(f"[{spec.module_tag}] origin failure asset={asset} interval={interval} horizon={hm} task={task} origin_ts={origin_ts} reason={reason}")
        window = window_result.window
        y_hist = window.y_hist
        ts_hist = window.ts_hist
        if last_refit_ts is None or should_refit(str(refit_cadence), last_refit_ts, int(origin_ts)):
            last_refit_ts = int(origin_ts)
            refit_count += 1
        origin_batch.append({
            "origin_ts": int(origin_ts),
            "train_start_ts": int(ts_hist[0]),
            "train_end_ts": int(ts_hist[-1]),
            "actual_value": float(window.actual_value),
            "fit_bars": int(y_hist.size),
            "refit_count": int(refit_count),
            "last_refit_ts": int(last_refit_ts) if last_refit_ts is not None else None,
            "predict_input": {
                "y_hist": y_hist,
                "horizon_bars": int(hb),
                "quantiles": quantiles,
                "seasonal_period_bars": int(period_bars) if period_bars is not None else None,
                "seed": 42 + idx_origin,
                "model_params": dict(combo_model_params),
                "x_hist": window.x_hist,
                "x_last": window.x_last,
                "factor_hist": window.factor_hist,
                "factor_last": window.factor_last,
            },
        })
        if len(origin_batch) >= int(origin_batch_size):
            _flush_origin_batch()
        now = time.monotonic()
        if (now - last_progress_log) >= float(progress_every_seconds):
            print(
                f"[{utc_now_iso()}] [{spec.module_tag}][progress] asset={asset} k={int(interval)} h={int(hm)}m task={task} "
                f"origin={int(idx_origin) + 1}/{len(finalized_idx)} rows={len(rows)} failed={int(failed_origins)} "
                f"elapsed_s={int(now - started_at)}",
                flush=True,
            )
            last_progress_log = now
    _flush_origin_batch()
    if row_count > 0 and last_row is not None:
        filled_rows = _shared_forward_fill_rows_to_edge(
            last_row=last_row,
            interval_minutes=int(interval),
            edge_ts=int(edge_ts),
        )
        if filled_rows:
            rows.extend(filled_rows)
            row_count += int(len(filled_rows))
            last_row = dict(filled_rows[-1])
        if last_eval_row is not None:
            filled_eval_rows = _shared_forward_fill_rows_to_edge(
                last_row=last_eval_row,
                interval_minutes=int(interval),
                edge_ts=int(edge_ts),
            )
            if filled_eval_rows:
                eval_rows.extend(filled_eval_rows)
                eval_row_count += int(len(filled_eval_rows))
                last_eval_row = dict(filled_eval_rows[-1])
    if failed_origins > 0 and row_count <= 0:
        skip_reason = "insufficient_valid_history" if int(insufficient_history_origins) == int(failed_origins) else "fit_failed_all_origins"
        return ukey, {
            "status": "skipped",
            "reason": skip_reason,
            "edge_ts": int(edge_ts),
            "metadata": {
                "failed_origins": int(failed_origins),
                "insufficient_history_origins": int(insufficient_history_origins),
                "target_tail_ts": int(target_tail),
                "seasonality_mode": seas.get("seasonality_mode"),
                "seasonality_label": seas.get("seasonality_label"),
            },
        }
    if rows:
        staged_rows_paths.append(
            _spill_rows_chunk(
                rows=rows,
                staging_root=staging_root,
                module_tag=spec.module_tag,
                interval=int(interval),
                horizon_minutes=int(hm),
                task=str(task),
                asset=str(asset),
            )
        )
        rows = []
    if eval_rows:
        eval_staged_rows_paths.append(
            _spill_rows_chunk(
                rows=eval_rows,
                staging_root=staging_root,
                module_tag=f"{spec.module_tag}_eval",
                interval=int(interval),
                horizon_minutes=int(hm),
                task=str(task),
                asset=str(asset),
            )
        )
        eval_rows = []
    if row_count > 0 and last_row is not None:
        _save_unit_state(
            state_root=state_root,
            asset=str(asset),
            interval=int(interval),
            horizon_minutes=int(hm),
            task=str(task),
            state={
                "run_id": str(run_id),
                "start_ts": int(effective_start_ts),
                "target_tail_ts": int(target_tail),
                "dst_tail_ts": (int(dst_tail) if dst_tail is not None else None),
                "last_refit_ts": (int(last_refit_ts) if last_refit_ts is not None else None),
                "refit_count": int(refit_count),
                "train_start_ts": int(last_row["train_start_ts"]),
                "train_end_ts": int(last_row["train_end_ts"]),
                "fit_meta": dict(fit_meta),
            },
        )
    return ukey, {
        "status": "done",
        "edge_ts": int(edge_ts),
        "rows": [],
        "staged_rows_paths": staged_rows_paths,
        "eval_rows": [],
        "eval_staged_rows_paths": eval_staged_rows_paths,
        "metadata": {
            "start_ts": int(effective_start_ts),
            "target_tail_ts": int(target_tail),
            "generated_tail_ts": int(actual_tail_ts) if actual_tail_ts is not None else None,
            "actual_tail_ts": int(edge_ts) if last_row is not None else None,
            "write_tail_ts": int(edge_ts),
            "row_count": int(row_count),
            "generated_row_count": int(generated_row_count),
            "eval_row_count": int(eval_row_count),
            "dst_tail_ts": int(dst_tail) if dst_tail is not None else None,
            "train_start_ts": int(last_row["train_start_ts"]) if last_row is not None else None,
            "train_end_ts": int(last_row["train_end_ts"]) if last_row is not None else None,
            "model_id": spec.model_id,
            "model_version": spec.model_version,
            "runtime_params": dict(runtime_params),
            "model_params": dict(combo_model_params),
            "hyperparams": {"fit_days": int(args_fit_days), "quantiles": [float(q) for q in quantiles], "refit_cadence": str(refit_cadence)},
            **fit_meta,
        },
    }


def _run_bayesian_asset_shard(payload: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
    work_spans = payload.get("work_spans") or {}
    shared_factor_cache_map: Optional[Dict[int, float]] = None
    if payload.get("factor_cache_path"):
        shared_factor_cache_map = _factor_cache_map_from_frame(_load_factor_cache(payload.get("factor_cache_path")))
    return [
        _run_bayesian_asset(
            {
                **payload,
                "asset": str(asset),
                "work_span": work_spans.get(str(asset)),
                "_factor_cache_map": shared_factor_cache_map,
            }
        )
        for asset in payload.get("assets", ())
    ]


def _run_bayesian_horizon_group_shard(payload: Dict[str, Any]) -> List[Tuple[Tuple[int, int, str], str, Dict[str, Any]]]:
    task_payloads = [dict(item) for item in (payload.get("task_payloads") or [])]
    if not task_payloads:
        return []
    out: List[Tuple[Tuple[int, int, str], str, Dict[str, Any]]] = []
    factor_maps: Dict[str, Optional[Dict[int, float]]] = {}
    for task_payload in task_payloads:
        task = str(task_payload["task"])
        factor_maps[task] = _factor_cache_map_from_frame(_load_factor_cache(task_payload.get("factor_cache_path"))) if task_payload.get("factor_cache_path") else None
    for asset in payload.get("assets", ()):
        asset = str(asset)
        active_payloads: List[Dict[str, Any]] = []
        read_start_values: List[int] = []
        edge_values: List[int] = []
        feature_cols: List[str] = []
        target_cols: List[str] = []
        for task_payload in task_payloads:
            work_span = (task_payload.get("work_spans") or {}).get(asset)
            if not isinstance(work_span, dict):
                continue
            active_payloads.append(task_payload)
            read_start_values.append(int(work_span["read_start_ts"]))
            edge_values.append(int(work_span["edge_ts"]))
            target_cols.append(str(task_payload["target_col"]))
            combo_profile_payload = task_payload.get("combo_profile")
            if isinstance(combo_profile_payload, dict):
                combo_profile = resolve_execution_profile(
                    Path(str(task_payload["feature_profile_json"])) if task_payload.get("feature_profile_json") else None,
                    interval=int(task_payload["interval"]),
                    horizon=int(task_payload["horizon_minutes"]),
                    task=str(task_payload["task"]),
                    dynamic_feature_candidates=task_payload["spec"].dynamic_feature_candidates,
                    needs_dynamic_features=bool(task_payload["spec"].needs_dynamic_features),
                    use_seasonality=bool(task_payload["spec"].use_seasonality),
                )
                feature_cols.extend(str(col) for col in combo_profile.selected_dynamic_feature_columns)
            elif task_payload["spec"].needs_dynamic_features:
                feature_cols.extend(str(col) for col in task_payload["spec"].dynamic_feature_candidates)
        if not active_payloads:
            continue
        read_start = min(read_start_values)
        edge_ts = max(edge_values)
        columns = list(dict.fromkeys([*target_cols, *feature_cols]))
        frame = read_feature_window_columns(
            root=Path(str(payload["feature_root"])),
            interval_minutes=int(payload["interval"]),
            asset=asset,
            columns=columns,
            start_ts=int(read_start),
            end_ts=int(edge_ts),
        )
        for target_col in dict.fromkeys(target_cols):
            frame = _overlay_runtime_target_labels(
                frame=frame,
                parquet_root=Path(str(payload["parquet_root"])),
                asset=asset,
                interval=int(payload["interval"]),
                horizon_minutes=int(payload["horizon_minutes"]),
                target_col=str(target_col),
                start_ts=int(read_start),
                end_ts=int(edge_ts),
            )
        for task_payload in active_payloads:
            task = str(task_payload["task"])
            combo_key = (int(task_payload["interval"]), int(task_payload["horizon_minutes"]), task)
            work_span = (task_payload.get("work_spans") or {}).get(asset)
            ukey, upd = _run_bayesian_asset(
                {
                    **task_payload,
                    "asset": asset,
                    "work_span": work_span,
                    "_factor_cache_map": factor_maps.get(task),
                    "_preloaded_frame": frame,
                }
            )
            out.append((combo_key, ukey, upd))
    return out


def _artifact_model_key(model_id: str) -> str:
    return _shared_artifact_model_key(model_id, prefix="bayes_")


def _resolve_combo_fit_days(*, requested_fit_days: Optional[int], runtime_params: Dict[str, Any], tested_training_window_months: Optional[int]) -> int:
    return _shared_resolve_combo_fit_days(
        requested_fit_days=requested_fit_days,
        runtime_params=runtime_params,
        tested_training_window_months=tested_training_window_months,
        default_fit_days_fn=lambda params: int(params.get("fit_days", os.getenv("BAYES_FIT_DAYS", "365"))),
    )


def _staging_root(forecast_root: Path, module_tag: str) -> Path:
    roots = _sandbox_roots()
    if not roots.enabled:
        return _shared_staging_root(forecast_root, module_tag)
    tmp_root = _sandbox_env_path(
        roots,
        "PIPELINE_SANDBOX_TMP_ROOT",
        roots.tmp_root,
        "Bayesian staging tmp root",
    )
    root = tmp_root / "bayesian_numeric_runner" / f"{str(module_tag)}_stage"
    assert_write_allowed(root, "Bayesian staging root", roots=roots)
    root.mkdir(parents=True, exist_ok=True)
    return root.resolve()


def discover_tested_production_artifact_scope(spec: BayesianNumericModuleSpec, project_root: Optional[Path] = None) -> Optional[TestedProductionArtifactScope]:
    return _shared_discover_numeric_tested_production_artifact_scope(
        project_root=(project_root or _project_root()).resolve(),
        diagnostics_root_name="bayesian_numeric_family_test_orchestrator",
        artifact_model_key=_artifact_model_key(spec.model_id),
        error_prefix=f"[{spec.module_tag}]",
        discover_tested_production_artifact_payload_fn=_shared_discover_tested_production_artifact_payload,
    )


def discover_existing_production_scope(spec: BayesianNumericModuleSpec, *, manifest_path: Path, project_root: Optional[Path] = None, canonical_io_config: Any = None) -> Optional[ExistingProductionScope]:
    if canonical_io_config is not None:
        combo_specs = _shared_discover_existing_combo_specs_from_canonical_physical_output(io_config=canonical_io_config)
        if combo_specs:
            return ExistingProductionScope(source_root=Path(canonical_io_config.parquet_root).resolve(), combo_specs=combo_specs)
    return _shared_discover_numeric_existing_production_scope(
        manifest_path=manifest_path,
        discover_existing_combo_specs_from_partitioned_output_fn=_discover_existing_combo_specs_from_partitioned_output,
    )


def run_bayesian_numeric_module(spec: BayesianNumericModuleSpec) -> None:
    parser = argparse.ArgumentParser(description=f"{spec.module_tag} Bayesian forecaster")
    parser.add_argument("--profile", type=str, default=selected_profile())
    parser.add_argument("--parquet-root", type=Path, default=None)
    parser.add_argument("--combo-list", type=str, default="")
    parser.add_argument("--assets", type=str, default="")
    parser.add_argument("--assets-file", type=str, default="")
    parser.add_argument("--workers", type=int, default=get_workers("bayesian_numeric_runner", "asset_workers", DEFAULT_WORKERS))
    parser.add_argument("--mode", type=str, choices=["incremental", "backfill"], default="incremental")
    parser.add_argument("--backfill_days", type=int, default=DEFAULT_BACKFILL_DAYS)
    parser.add_argument("--predict_latest_only", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--overwrite_months", type=str, default="")
    parser.add_argument("--fit_days", type=int, default=None)
    parser.add_argument("--refit-cadence", type=str, default="")
    parser.add_argument("--quantiles", type=str, default="0.1,0.5,0.9")
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    if args.parquet_root is None:
        args.parquet_root = resolve_path(
            "source_ohlcvt_root",
            profile=str(args.profile),
            required=True,
        )
    worker_budget = max(1, int(args.workers))

    runtime_params = dict(spec.runtime_params)
    requested_fit_days = args.fit_days
    requested_refit_cadence = str(args.refit_cadence or runtime_params.get("refit_cadence", "")).strip().lower()
    if requested_refit_cadence in {"", "auto"}:
        requested_refit_cadence = ""
    resolved_model_threads = _resolve_runner_model_threads(worker_budget, resolve_bayesian_model_threads())

    log_resolved_runtime("bayesian_numeric_runner", resolved={"asset_workers": int(worker_budget), "writer_workers": 1, "model_threads": int(resolved_model_threads), "module_tag": spec.module_tag})

    source_parquet_root = Path(args.parquet_root).resolve()
    _, feature_root, forecast_root = default_common_roots(spec.family_root_env, spec.family_root_name)
    staging_root = _staging_root(forecast_root, spec.module_tag)
    state_root = _state_root(forecast_root, spec.module_tag)
    assert_write_allowed(state_root, "Bayesian model state root", roots=_sandbox_roots())
    state_root.mkdir(parents=True, exist_ok=True)
    canonical_io_config = _shared_canonical_physical_io_config(
        naming=_shared_canonical_physical_naming(
            module_slug=spec.module_tag,
            prediction_prefix=spec.module_tag,
            log_prefix=f"[{spec.module_tag}]",
        ),
        parquet_root=forecast_root,
        staging_root=staging_root,
        state_root=state_root,
        scalar_root=feature_root,
        ohlc_root=source_parquet_root,
        log_fn=lambda message: print(message, flush=True),
    )
    files = make_files(forecast_root, "bayes_run_manifest.json", "bayes_skipped.json")
    quantiles = sorted(set(parse_quantiles(args.quantiles, default_vals=(0.1, 0.5, 0.9))) | {0.1, 0.5, 0.9})
    using_default_combo_selection = not str(args.combo_list).strip()
    production_scope = discover_existing_production_scope(spec, manifest_path=files.manifest_file, canonical_io_config=canonical_io_config) if using_default_combo_selection else None
    tested_scope = None
    tested_scope_error = None
    stream_contract = None
    if using_default_combo_selection:
        try:
            tested_scope = discover_tested_production_artifact_scope(spec)
        except BaseException as exc:
            tested_scope_error = exc
        stream_contract = _shared_require_production_stream_scope_contract(
            _shared_resolve_production_stream_scope_contract(
                family="bayesian",
                model=spec.module_tag,
                existing_scope=production_scope,
                tested_scope=tested_scope,
                tested_scope_error=tested_scope_error,
                production_paths_inspected=(forecast_root, files.manifest_file),
            )
        )
    if stream_contract is not None and stream_contract.mode == "locked_to_existing_production":
        print(
            f"[{utc_now_iso()}] [{spec.module_tag}] production-defaults root={production_scope.source_root if production_scope is not None else forecast_root} "
            f"combos={len(stream_contract.combo_specs)} "
            f"asset_scope=existing_production_scope scope_mode={stream_contract.mode} warnings={';'.join(stream_contract.warnings)}",
            flush=True,
        )
    elif stream_contract is not None and stream_contract.mode == "bootstrap_from_test" and tested_scope is not None:
        print(
            f"[{utc_now_iso()}] [{spec.module_tag}] tested-defaults handoff={tested_scope.handoff_path} "
            f"feature_profile_json={tested_scope.feature_profile_json} cohort_assets={len(tested_scope.cohort_assets)} "
            f"combo_source={'stage3' if tested_scope.stage3_combo_specs else 'stage2'} "
            f"combos={len(tested_scope.stage3_combo_specs or tested_scope.combo_specs)} "
            f"stage3_combo_results={tested_scope.stage3_combo_results_path} "
            f"asset_scope=full_production_universe scope_mode={stream_contract.mode}",
            flush=True,
        )
    assets = load_assets(intervals=spec.default_intervals, assets_arg=args.assets, assets_file=args.assets_file)
    if not assets:
        write_json_atomic(files.skipped_file, {"run_id": "none", "generated_at": utc_now_iso(), "units": {}})
        return

    combos = _parse_combo_list(args.combo_list) if str(args.combo_list).strip() else list(stream_contract.combo_specs if stream_contract is not None else ())
    combos = sorted({(int(i), int(h), str(t)) for i, h, t in combos if int(i) > 0 and int(h) > 0 and int(h) % int(i) == 0}, key=lambda item: (item[0], item[1], item[2]))
    tested_combo_window_months = _combo_window_map(tested_scope)
    run_id = os.getenv("RUN_ID", "") or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    diagnostics_path = diagnostics_file(state_root, spec.module_tag)
    reset_diagnostics_file(diagnostics_path)
    append_diagnostic_event(
        diagnostics_path,
        "run_start",
        {
            "run_id": run_id,
            "family": "bayesian_numeric",
            "module_tag": spec.module_tag,
            "assets": len(assets),
            "workers": int(worker_budget),
            "model_threads": int(resolved_model_threads),
            "mode": str(args.mode),
            "backfill_days": int(args.backfill_days),
            "predict_latest_only": bool(args.predict_latest_only),
            "force": bool(args.force),
            "resource": resource_snapshot(),
        },
        timestamp_fn=utc_now_iso,
    )
    manifest_parts: List[Dict[str, Any]] = []
    skipped_units: Dict[str, Dict[str, Any]] = {}
    combo_plans: List[Dict[str, Any]] = []
    planned_assets_by_combo: Dict[Tuple[int, int, str], List[str]] = {}

    task_map = default_task_map()
    combo_jobs: List[Dict[str, Any]] = []
    combo_order: List[Tuple[int, int, str]] = []
    progress_every_seconds = _resolve_progress_every_seconds_for_spec(spec)
    state_flush_combo_cadence = _state_flush_combo_cadence()
    planning_workers = max(1, min(int(worker_budget), max(1, len(assets))))
    tail_cache = _shared_TailPlanningCache()
    for interval, hm, task in combos:
        target_col = task_target_col(str(task), task_map)
        if not target_col:
            continue
        combo_key = (int(interval), int(hm), str(task))
        combo_order.append(combo_key)
        tested_training_window_months = tested_combo_window_months.get((int(interval), int(hm), str(task)))
        combo_fit_days = _resolve_combo_fit_days(
            requested_fit_days=requested_fit_days,
            runtime_params=runtime_params,
            tested_training_window_months=tested_training_window_months,
        )
        combo_profile = (
            resolve_execution_profile(
                tested_scope.feature_profile_json,
                interval=int(interval),
                horizon=int(hm),
                task=str(task),
                dynamic_feature_candidates=spec.dynamic_feature_candidates,
                needs_dynamic_features=bool(spec.needs_dynamic_features),
                use_seasonality=bool(spec.use_seasonality),
            )
            if tested_scope is not None
            else None
        )
        refit_cadence = normalize_refit_cadence(requested_refit_cadence) if requested_refit_cadence else _default_refit_cadence_for_interval(int(interval))
        combo_model_params = dict(spec.resolve_model_params_fn(task=str(task), interval_minutes=int(interval), horizon_minutes=int(hm)))
        if tested_scope is not None:
            combo_model_params.update(dict(tested_scope.tuned_params_by_combo.get((int(interval), int(hm), str(task))) or {}))
        plan_fn = lambda asset_name: _plan_asset_work_span(
            asset=str(asset_name),
            interval=int(interval),
            horizon_minutes=int(hm),
            task=str(task),
            mode=str(args.mode),
            backfill_days=int(args.backfill_days),
            force=bool(args.force),
            fit_days=int(combo_fit_days),
            forecast_root=forecast_root,
            canonical_io_config=canonical_io_config,
            tail_cache=tail_cache,
        )
        if planning_workers <= 1 or len(assets) <= 1:
            planned_spans_by_asset = {str(asset_name): plan_fn(asset_name) for asset_name in assets}
        else:
            try:
                with ThreadPoolExecutor(max_workers=int(planning_workers)) as planning_pool:
                    planned_results = list(planning_pool.map(plan_fn, assets))
                planned_spans_by_asset = {str(asset_name): planned_results[idx] for idx, asset_name in enumerate(assets)}
            except Exception as exc:
                print(
                    f"[{utc_now_iso()}] [{spec.module_tag}][runtime-fallback] planning pool unavailable; forcing serial planning: {exc}",
                    flush=True,
                )
                planned_spans_by_asset = {str(asset_name): plan_fn(asset_name) for asset_name in assets}
        planned_work_spans = [span for span in planned_spans_by_asset.values() if span is not None]
        combo_plan: Dict[str, Any] = {
            "interval": int(interval),
            "horizon_minutes": int(hm),
            "task": str(task),
            "training_window_months": (int(tested_training_window_months) if tested_training_window_months is not None else None),
            "fit_days": int(combo_fit_days),
            "fit_days_source": (
                "cli_override"
                if requested_fit_days is not None
                else "tested_training_window_months"
                if tested_training_window_months is not None
                else "runtime_default"
            ),
            "planned_assets_with_work": int(len(planned_work_spans)),
        }
        print(
            f"[{utc_now_iso()}] [{spec.module_tag}][group-start] k={int(interval)} h={int(hm)}m task={task} "
            f"assets_total={len(assets)} work_items={len(planned_work_spans)} fit_days={int(combo_fit_days)}",
            flush=True,
        )
        if not planned_work_spans:
            print(
                f"[{utc_now_iso()}] [{spec.module_tag}][group-noop] k={int(interval)} h={int(hm)}m task={task} "
                f"reason=already_at_edge assets_total={len(assets)}",
                flush=True,
            )
        factor_cache = pd.DataFrame(columns=["ts", "market_factor"])
        if spec.needs_factor_cache:
            if planned_work_spans:
                factor_cache_start_ts = min(int(span.read_start_ts) for span in planned_work_spans)
                factor_cache_end_ts = max(int(span.edge_ts) for span in planned_work_spans)
                factor_cache_path = _factor_cache_artifact_path(
                    staging_root=staging_root,
                    module_tag=spec.module_tag,
                    interval=int(interval),
                    horizon_minutes=int(hm),
                    task=str(task),
                    start_ts=int(factor_cache_start_ts),
                    end_ts=int(factor_cache_end_ts),
                )
                if factor_cache_path.exists():
                    factor_cache = _load_factor_cache(str(factor_cache_path))
                    print(
                        f"[{utc_now_iso()}] [{spec.module_tag}] [factor-cache-reuse] "
                        f"k={int(interval)} h={int(hm)}m task={task} "
                        f"rows={len(factor_cache)} path={factor_cache_path}",
                        flush=True,
                    )
                else:
                    factor_cache = _compute_factor_cache(
                        parquet_root=source_parquet_root,
                        feature_root=feature_root,
                        interval=int(interval),
                        horizon_minutes=int(hm),
                        task=str(task),
                        assets=assets,
                        start_ts=int(factor_cache_start_ts),
                        end_ts=int(factor_cache_end_ts),
                        max_workers=int(planning_workers),
                        log_fn=lambda message: print(f"[{utc_now_iso()}] [{spec.module_tag}] {message}", flush=True),
                        log_prefix="",
                    )
                combo_plan["factor_cache_start_ts"] = int(factor_cache_start_ts)
                combo_plan["factor_cache_end_ts"] = int(factor_cache_end_ts)
                combo_plan["factor_cache_rows"] = int(len(factor_cache))
            else:
                combo_plan["factor_cache_rows"] = 0
        combo_plans.append(combo_plan)
        hb = int(hm) // int(interval)
        planned_assets = [str(asset_name) for asset_name in assets if planned_spans_by_asset.get(str(asset_name)) is not None]
        planned_assets_by_combo[combo_key] = list(planned_assets)
        factor_cache_path = _persist_factor_cache(
            factor_cache=factor_cache,
            staging_root=staging_root,
            module_tag=spec.module_tag,
            interval=int(interval),
            horizon_minutes=int(hm),
            task=str(task),
            start_ts=int(combo_plan.get("factor_cache_start_ts", 0) or 0),
            end_ts=int(combo_plan.get("factor_cache_end_ts", 0) or 0),
        )
        shard_payload = {
            "spec": spec,
            "interval": int(interval),
            "horizon_minutes": int(hm),
            "task": str(task),
            "target_col": str(target_col),
            "run_id": str(run_id),
            "quantiles": [float(q) for q in quantiles],
            "force": bool(args.force),
            "mode": str(args.mode),
            "backfill_days": int(args.backfill_days),
            "fit_days": int(combo_fit_days),
            "predict_latest_only": bool(args.predict_latest_only),
            "parquet_root": str(source_parquet_root),
            "feature_root": str(feature_root),
            "forecast_root": str(forecast_root),
            "staging_root": str(staging_root),
            "state_root": str(state_root),
            "refit_cadence": str(refit_cadence),
            "combo_model_params": dict(combo_model_params),
            "runtime_params": dict(runtime_params),
            "hb": int(hb),
            "feature_profile_json": (str(tested_scope.feature_profile_json) if tested_scope is not None else None),
            "combo_profile": ({"enabled": True} if combo_profile is not None else None),
            "factor_cache_path": factor_cache_path,
            "model_threads": int(resolved_model_threads),
            "progress_every_seconds": int(progress_every_seconds),
            "allow_partial": bool(args.allow_partial),
        }
        combo_jobs.extend(
            _shared_build_asset_shard_jobs(
                combo_key=combo_key,
                planned_assets=planned_assets,
                worker_count=int(worker_budget),
                base_payload=shard_payload,
                work_spans_by_asset={
                    str(asset_name): planned_spans_by_asset[str(asset_name)]
                    for asset_name in planned_assets
                },
                work_span_payload_fn=_shared_planned_work_span_to_payload,
            )
        )

    group_jobs = _shared_build_horizon_group_shard_jobs(combo_jobs=combo_jobs, worker_count=int(worker_budget))
    dispatch_slots = _resolve_dispatch_slots(worker_budget, len(group_jobs)) if group_jobs else 0
    print(
        f"[{utc_now_iso()}] [{spec.module_tag}] production-plan combos={len(combo_order)} combo_jobs={len(combo_jobs)} "
        f"group_jobs={len(group_jobs)} worker_mode=horizon_group_process_shards assets={len(assets)} workers={int(worker_budget)} "
        f"dispatch_slots={int(dispatch_slots)} model_threads={int(resolved_model_threads)} "
        f"planning_workers={int(planning_workers)} progress_seconds={int(progress_every_seconds)} "
        f"state_flush_combo_cadence={int(state_flush_combo_cadence)} "
        f"planned_work_items_total={sum(int(plan.get('planned_assets_with_work', 0) or 0) for plan in combo_plans)}",
        flush=True,
    )
    append_diagnostic_event(
        diagnostics_path,
        "production_plan",
        {
            "run_id": run_id,
            "family": "bayesian_numeric",
            "module_tag": spec.module_tag,
            "combos": int(len(combo_order)),
            "combo_jobs": int(len(combo_jobs)),
            "group_jobs": int(len(group_jobs)),
            "assets": int(len(assets)),
            "workers": int(worker_budget),
            "dispatch_slots": int(dispatch_slots),
            "model_threads": int(resolved_model_threads),
            "planning_workers": int(planning_workers),
            "planned_work_items_total": int(sum(int(plan.get("planned_assets_with_work", 0) or 0) for plan in combo_plans)),
            "resource": resource_snapshot(),
        },
        timestamp_fn=utc_now_iso,
    )

    writer_queue, writer_state, writer_thread = _shared_start_partitioned_prediction_writer(
        module_tag=spec.module_tag,
        forecast_root=forecast_root,
        writer_loop_fn=lambda **kwargs: _writer_loop(canonical_io_config=canonical_io_config, **kwargs),
    )
    write_json_atomic(files.skipped_file, {"run_id": run_id, "generated_at": utc_now_iso(), "units": skipped_units})
    write_json_atomic(
        files.manifest_file,
        _manifest_snapshot_payload(
            run_id=run_id,
            spec=spec,
            combos=combos,
            combo_plans=combo_plans,
            worker_budget=worker_budget,
            dispatch_slots=dispatch_slots,
            resolved_model_threads=resolved_model_threads,
            args=args,
            requested_fit_days=requested_fit_days,
            requested_refit_cadence=requested_refit_cadence,
            quantiles=quantiles,
            runtime_params=runtime_params,
            staging_root=staging_root,
            state_root=state_root,
            tested_scope=tested_scope,
            production_scope=production_scope,
            manifest_parts=manifest_parts,
            skipped_units=skipped_units,
            combo_count=len(combo_order),
            job_shard_count=len(group_jobs),
            writer_stats=writer_state.get("writer_stats", {}),
        ),
    )
    def _make_unit_result(payload: Dict[str, Any], upd: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if str(upd.get("status", "")) != "done":
            return None
        meta = dict(upd.get("metadata") or {})
        return {
            "asset": str(upd.get("asset") or ""),
            "interval": int(upd.get("interval_min", payload["interval"])),
            "horizon_minutes": int(upd.get("horizon_min", payload["horizon_minutes"])),
            "task": str(upd.get("task", payload["task"])),
            "rows_written": int(meta.get("row_count", len(upd.get("rows", []) or [])) or 0),
            "work_start_ts": int(meta.get("start_ts") or 0),
            "work_end_ts": int(meta.get("target_tail_ts") or 0),
        }

    updates_by_combo, unit_results = _shared_execute_grouped_horizon_jobs(
        group_jobs=group_jobs,
        combo_order=combo_order,
        dispatch_slots=dispatch_slots,
        module_name="bayesian_numeric_runner",
        module_tag=spec.module_tag,
        family_label="Bayesian numeric",
        log_fn=lambda message: print(f"[{utc_now_iso()}] {message}", flush=True),
        run_group_shard_fn=_run_bayesian_horizon_group_shard,
        writer_queue=writer_queue,
        writer_state=writer_state,
        make_unit_result_fn=_make_unit_result,
        process_pool_init_retries=PROCESS_POOL_INIT_RETRIES,
        process_pool_init_retry_seconds=PROCESS_POOL_INIT_RETRY_SECONDS,
        pressure_guard_factory=lambda module_name, log_fn: DispatchPressureGuard(module_name=module_name, log_fn=log_fn),
        diagnostics_path=diagnostics_path,
        diagnostics_timestamp_fn=utc_now_iso,
    )

    print(
        f"[{utc_now_iso()}] [{spec.module_tag}][phase:writer-drain] "
        f"combo_jobs={len(combo_jobs)} queued_writer_messages={int(getattr(writer_queue, 'unfinished_tasks', 0))} "
        f"parts_written_so_far={sum(len(parts) for parts in writer_state.get('parts_by_combo', {}).values())}",
        flush=True,
    )
    _wait_for_writer_drain(
        writer_queue=writer_queue,
        writer_thread=writer_thread,
        writer_state=writer_state,
        family_label="Bayesian numeric",
    )
    append_diagnostic_event(
        diagnostics_path,
        "writer_drain_complete",
        {
            "run_id": run_id,
            "family": "bayesian_numeric",
            "module_tag": spec.module_tag,
            "writer_stats": dict(writer_state.get("writer_stats", {})),
            "resource": resource_snapshot(),
        },
        timestamp_fn=utc_now_iso,
    )
    print(f"[{utc_now_iso()}] [{spec.module_tag}][phase:writer-stop] writer_queue=drained", flush=True)
    writer_queue.put({"kind": "stop"})
    writer_thread.join()
    _raise_writer_fatal(writer_state, "Bayesian numeric")
    parts_by_combo: Dict[Tuple[int, int, str], List[Dict[str, Any]]] = dict(writer_state.get("parts_by_combo", {}))
    eval_parts_by_combo: Dict[Tuple[int, int, str], List[Dict[str, Any]]] = dict(
        writer_state.get("parts_by_store", {}).get("eval", {})
    )

    print(
        f"[{utc_now_iso()}] [{spec.module_tag}][phase:validate] combos={len(combo_order)}",
        flush=True,
    )
    for combo_index, (interval, hm, task) in enumerate(combo_order, start=1):
        unit_updates = updates_by_combo.get((int(interval), int(hm), str(task)), [])
        _shared_validate_combo_completion(
            combo_key=(int(interval), int(hm), str(task)),
            planned_assets=planned_assets_by_combo.get((int(interval), int(hm), str(task)), []),
            combo_updates=unit_updates,
            module_tag=spec.module_tag,
        )
        written = list(parts_by_combo.get((int(interval), int(hm), str(task)), []))
        manifest_parts.extend(written)
        print(
            f"[{utc_now_iso()}] [{spec.module_tag}] k={int(interval)} h={int(hm)}m task={task} "
            f"parts_written={len(written)} skipped_units={len([1 for _ukey, upd in unit_updates if str(upd.get('status', 'skipped')) == 'skipped'])}",
            flush=True,
        )
        for ukey, upd in unit_updates:
            if str(upd.get("status", "skipped")) == "skipped":
                skipped_units[ukey] = {"reason": str(upd.get("reason", "skipped")), "interval_min": int(interval), "horizon_min": int(hm), "task": str(task), "edge_ts": int(upd.get("edge_ts")) if upd.get("edge_ts") is not None else None}
        should_flush_state = (combo_index % int(state_flush_combo_cadence) == 0) or (combo_index == len(combo_order))
        if should_flush_state:
            write_json_atomic(files.skipped_file, {"run_id": run_id, "generated_at": utc_now_iso(), "units": skipped_units})
            write_json_atomic(
                files.manifest_file,
                _manifest_snapshot_payload(
                    run_id=run_id,
                    spec=spec,
                    combos=combos,
                    combo_plans=combo_plans,
                    worker_budget=worker_budget,
                    dispatch_slots=dispatch_slots,
                    resolved_model_threads=resolved_model_threads,
                    args=args,
                    requested_fit_days=requested_fit_days,
                    requested_refit_cadence=requested_refit_cadence,
                    quantiles=quantiles,
                    runtime_params=runtime_params,
                    staging_root=staging_root,
                    state_root=state_root,
                    tested_scope=tested_scope,
                    production_scope=production_scope,
                    manifest_parts=manifest_parts,
                    skipped_units=skipped_units,
                    combo_count=len(combo_order),
                    job_shard_count=len(group_jobs),
                    writer_stats=writer_state.get("writer_stats", {}),
                ),
            )

    print(
        f"[{utc_now_iso()}] [{spec.module_tag}][phase:manifest] unit_results={len(unit_results)} "
        f"manifest_parts={len(manifest_parts)} skipped_units={len(skipped_units)}",
        flush=True,
    )
    for res in unit_results:
        print(
            f"[{utc_now_iso()}] [{spec.module_tag}] asset={res.get('asset')} k={int(res.get('interval', 0) or 0)} "
            f"h={int(res.get('horizon_minutes', 0) or 0)}m task={res.get('task')} rows_written={int(res.get('rows_written', 0) or 0)} "
            f"work=[{int(res.get('work_start_ts', 0) or 0)},{int(res.get('work_end_ts', 0) or 0)}]",
            flush=True,
        )
    print(
        f"[{utc_now_iso()}] [{spec.module_tag}] run complete forecast_parts={sum(len(parts_by_combo.get(combo_key, [])) for combo_key in combo_order)} "
        f"eval_parts={sum(len(eval_parts_by_combo.get(combo_key, [])) for combo_key in combo_order)} assets_with_start={sum(1 for res in unit_results if int(res.get('work_start_ts', 0) or 0) > 0)} "
        f"no_work_combos={sum(1 for plan in combo_plans if int(plan.get('planned_assets_with_work', 0) or 0) == 0)}",
        flush=True,
    )
    append_diagnostic_event(
        diagnostics_path,
        "run_complete",
        {
            "run_id": run_id,
            "family": "bayesian_numeric",
            "module_tag": spec.module_tag,
            "forecast_parts": int(sum(len(parts_by_combo.get(combo_key, [])) for combo_key in combo_order)),
            "eval_parts": int(sum(len(eval_parts_by_combo.get(combo_key, [])) for combo_key in combo_order)),
            "unit_results": int(len(unit_results)),
            "skipped_units": int(len(skipped_units)),
            "writer_stats": dict(writer_state.get("writer_stats", {})),
            "resource": resource_snapshot(),
        },
        timestamp_fn=utc_now_iso,
    )

    write_json_atomic(files.skipped_file, {"run_id": run_id, "generated_at": utc_now_iso(), "units": skipped_units})
    write_json_atomic(
        files.manifest_file,
        _manifest_snapshot_payload(
            run_id=run_id,
            spec=spec,
            combos=combos,
            combo_plans=combo_plans,
            worker_budget=worker_budget,
            dispatch_slots=dispatch_slots,
            resolved_model_threads=resolved_model_threads,
            args=args,
            requested_fit_days=requested_fit_days,
            requested_refit_cadence=requested_refit_cadence,
            quantiles=quantiles,
            runtime_params=runtime_params,
            staging_root=staging_root,
            state_root=state_root,
            tested_scope=tested_scope,
            production_scope=production_scope,
            manifest_parts=manifest_parts,
            skipped_units=skipped_units,
            combo_count=len(combo_order),
            job_shard_count=len(group_jobs),
            writer_stats=writer_state.get("writer_stats", {}),
            finished_at=utc_now_iso(),
        ),
    )
