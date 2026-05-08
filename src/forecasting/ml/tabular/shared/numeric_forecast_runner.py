from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import threading
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from src.features.numeric_forecast_profiles import (
    format_combo_list,
    format_combo_window_list,
    format_combo_window_refit_list,
    should_refit,
)
from src.features.scalar_features import (
    OHLCVT_PARQUET_ROOT,
    PARQUET_COMPRESSION,
    PARQUET_ROW_GROUP,
    PARQUET_ROOT as SCALAR_PARQUET_ROOT,
    feature_max_ts_from_parquet,
    first_ohlcvt_ts_from_disk,
    list_assets_from_ohlcvt,
    ohlcvt_max_ts_from_parquet,
)
from src.forecasting.common.ml_module_utils import horizon_bars_from_minutes as shared_horizon_bars_from_minutes
from src.forecasting.common.ohlcvt_source import read_ohlcvt
from src.forecasting.common.path_config import require_pipeline_io, resolve_path, selected_profile
from src.forecasting.common.runtime_config import cap_model_threads, get_model_threads, get_workers, log_resolved_runtime
from src.forecasting.common.sandbox_paths import SandboxOutputRoots, assert_write_allowed, resolve_sandbox_output_roots
from src.forecasting.ml.shared.numeric_runtime_common import (
    ModuleLogFn,
    TabularTestedProductionArtifactScope as TestedProductionArtifactScope,
    create_queue_with_retry as _create_stage_queue_with_retry,
    deadzone_by_task as _deadzone_by_task,
    discover_tabular_existing_production_scope as _discover_tabular_existing_production_scope,
    discover_tabular_tested_production_artifact_scope as _discover_tabular_tested_production_artifact_scope,
    env_int as _env_int,
    resolve_planning_workers as _resolve_planning_workers,
)
from src.forecasting.ml.tabular.shared.numeric_forecast_cli import (
    horizons_for_interval,
    parse_combo_list,
    parse_combo_profile_list,
    parse_combo_window_list,
    parse_refit_cadence,
    parse_requested_tasks,
    parse_train_window_months,
    validate_requested_task_horizon_pairs,
)
from src.forecasting.ml.tabular.shared.numeric_forecast_engine import (
    HorizonGroupWork,
    NumericForecastEngineConfig,
    UnitWork,
    build_unit_work,
    bundle_to_state,
    compute_group,
    state_to_bundle,
    walk_forward_predict,
)
from src.forecasting.ml.shared.numeric_forecast_io import (
    NumericForecastIOConfig,
    NumericForecastNamingConfig,
    chunk_end_ts_for_month,
    coalesce_keyed_frames,
    expected_eval_columns,
    expected_forecast_columns,
    expected_store_columns,
    finalize_group_results,
    get_start_ts,
    get_stop_ts,
    init_stage_write_queue,
    load_model_state,
    load_unit_feature_frame,
    model_state_path,
    module_table,
    module_output_max_ts,
    month_part_path,
    pair_columns_state,
    read_monthly_filtered,
    resolve_assets,
    save_model_state,
    stage_month_parts,
    stage_writer_loop,
    validated_existing_month_parquet,
    validated_module_month_parquet,
    write_month_parts,
)
from src.forecasting.ml.shared.production_time import production_start_ts
from src.forecasting.ml.shared.numeric_forecast_targets import compute_future_labels, future_window_views, safe_log_return, safe_true_range
MIN_PLANNING_WORKERS = 14
STAGE_PARALLEL_INIT_RETRIES = 3
STAGE_PARALLEL_INIT_RETRY_SECONDS = 0.25


@dataclass(frozen=True)
class NumericFamilyModuleSpec:
    module_slug: str
    family_name: str
    prediction_prefix: str
    log_prefix: str
    parquet_root_env: str
    progress_seconds_env: str
    source_start_env: str
    source_end_env: str
    work_start_env: str
    forecast_resume_edge_env: str
    eval_resume_edge_env: str
    deadzone_env_prefix: str
    default_unit_workers: int
    default_model_threads: int
    max_logical_threads: int
    thread_env_vars: Sequence[str]
    thread_param_name: str
    numeric_tasks: Sequence[str]
    task_short: Dict[str, str]
    task_label: Dict[str, str]
    future_label_columns: Sequence[str]
    default_intervals: Sequence[int]
    default_horizon_minutes: Sequence[int]
    active_task_horizon_matrix: Dict[str, List[int]]
    normalize_refit_cadence_fn: Callable[[str], str]
    resolve_default_combo_profile_fn: Callable[[], Sequence[Tuple[int, int, str, int]]]
    resolve_default_refit_policy_fn: Callable[[Tuple[int, int, str]], Optional[str]]
    select_feature_columns_fn: Callable[[Sequence[str], str, int, int, set[str]], List[str]]
    fit_model_fn: Callable[..., Any]
    predict_model_fn: Callable[..., Any]
    resolve_regressor_params_fn: Callable[..., Dict[str, Any]]
    regressor_profile_label_fn: Callable[..., str]
    default_training_window_months_for_combo_fn: Callable[[int, int, str], int]
    training_window_bars_for_pair_fn: Callable[[str, int, int], int]
    training_window_bars_from_months_fn: Callable[[int, int], int]

def _infer_training_window_months_for_state(
    spec: NumericFamilyModuleSpec,
    interval: int,
    horizon_minutes: int,
    task: str,
    selected_window_bars: Optional[int],
) -> int:
    if selected_window_bars is not None:
        for months in range(1, 121):
            try:
                if int(spec.training_window_bars_from_months_fn(int(months), int(interval))) == int(selected_window_bars):
                    return int(months)
            except Exception:
                continue
    return int(spec.default_training_window_months_for_combo_fn(int(interval), int(horizon_minutes), str(task)))


def discover_existing_production_scope(spec: NumericFamilyModuleSpec, io_config: NumericForecastIOConfig) -> Optional[Tuple[Tuple[int, int, str, int], ...]]:
    return _discover_tabular_existing_production_scope(
        state_root=Path(io_config.state_root),
        infer_training_window_months_fn=lambda interval, horizon_minutes, task, selected_window_bars: _infer_training_window_months_for_state(
            spec,
            int(interval),
            int(horizon_minutes),
            str(task),
            selected_window_bars,
        ),
        load_state_fn=lambda asset, interval, horizon_minutes, task: load_model_state(
            io_config,
            asset=str(asset),
            interval=int(interval),
            horizon_minutes=int(horizon_minutes),
            task=str(task),
        )
        or {},
    )


def discover_tested_production_artifact_scope(spec: NumericFamilyModuleSpec, project_root: Optional[Path] = None) -> Optional[TestedProductionArtifactScope]:
    return _discover_tabular_tested_production_artifact_scope(
        module_slug=spec.module_slug,
        log_prefix=spec.log_prefix,
        project_root=project_root,
    )


def _sandbox_env_path(roots: SandboxOutputRoots, env_name: str, fallback: Path, kind: str) -> Path:
    raw = str(os.getenv(env_name, "") or "").strip()
    path = Path(raw) if raw else Path(fallback)
    assert_write_allowed(path, kind, roots=roots)
    return path


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


def build_numeric_family_module(spec: NumericFamilyModuleSpec) -> Dict[str, Any]:
    pipeline_profile = selected_profile()
    legacy_pipeline_root = Path(os.getenv("PIPELINE_ROOT")) if str(os.getenv("PIPELINE_ROOT", "")).strip() else None
    path_config_env = dict(os.environ)
    if legacy_pipeline_root is not None and not str(os.getenv("PIPELINE_TMP_ROOT", "")).strip():
        for name in ("TMPDIR", "TEMP", "TMP"):
            path_config_env.pop(name, None)
    pipeline_root = legacy_pipeline_root or resolve_path("tmp_root", profile=pipeline_profile, env=path_config_env, required=False) or Path(".")
    sandbox_roots = resolve_sandbox_output_roots(env=_sandbox_resolution_env())
    if sandbox_roots.enabled:
        log_dir = _sandbox_env_path(sandbox_roots, "PIPELINE_SANDBOX_LOG_ROOT", sandbox_roots.log_root, "tabular numeric log root")
        state_base = _sandbox_env_path(sandbox_roots, "PIPELINE_SANDBOX_STATE_ROOT", sandbox_roots.state_root, "tabular numeric state root")
        tmp_root = _sandbox_env_path(sandbox_roots, "PIPELINE_SANDBOX_TMP_ROOT", sandbox_roots.tmp_root, "tabular numeric tmp root")
        parquet_root = _sandbox_env_path(sandbox_roots, "PIPELINE_SANDBOX_PARQUET_ROOT", sandbox_roots.parquet_root, "tabular numeric parquet root")
        catboost_train_dir = _sandbox_env_path(
            sandbox_roots,
            "PIPELINE_SANDBOX_CATBOOST_TRAIN_DIR",
            sandbox_roots.catboost_train_dir,
            "tabular numeric CatBoost train dir",
        )
        state_root = state_base / "model_states" / spec.module_slug
        staging_root = tmp_root / f"{spec.module_slug}_stage"
        for path, kind in (
            (state_root, "tabular numeric model state root"),
            (staging_root, "tabular numeric staging root"),
            (catboost_train_dir, "tabular numeric CatBoost train dir"),
        ):
            assert_write_allowed(path, kind, roots=sandbox_roots)
        os.environ["TMP"] = str(tmp_root)
        os.environ["TEMP"] = str(tmp_root)
        os.environ["TMPDIR"] = str(tmp_root)
        if "catboost" in str(spec.module_slug).lower():
            os.environ["CATBOOST_TRAIN_DIR"] = str(catboost_train_dir)
    else:
        log_dir = resolve_path("log_root", profile=pipeline_profile, env=path_config_env, required=False) or (legacy_pipeline_root / "logs" if legacy_pipeline_root is not None else Path("logs"))
        state_base = resolve_path("state_root", profile=pipeline_profile, env=path_config_env, required=False) or (legacy_pipeline_root / "model_states" if legacy_pipeline_root is not None else Path("model_states"))
        tmp_root = resolve_path("tmp_root", profile=pipeline_profile, env=path_config_env, required=False) or (legacy_pipeline_root / "tmp" if legacy_pipeline_root is not None else Path("tmp"))
        state_root = state_base / spec.module_slug
        staging_root = tmp_root / f"{spec.module_slug}_stage"
        parquet_root = Path(
            os.getenv(spec.parquet_root_env)
            or resolve_path("output_parquet_root", profile=pipeline_profile, env=path_config_env, required=False)
            or Path("parquet")
        )
        catboost_train_dir = Path(os.getenv("CATBOOST_TRAIN_DIR", str(tmp_root / "catboost_numerics_stage" / "catboost_train")))
    for path in (log_dir, state_root, staging_root):
        try:
            path.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

    log_file = log_dir / f"{spec.module_slug}.log"
    scalar_root = Path(
        os.getenv("PIPELINE_SOURCE_FEATURES_ROOT")
        or resolve_path("source_feature_root", profile=pipeline_profile, env=path_config_env, required=False)
        or SCALAR_PARQUET_ROOT
    )
    ohlc_root = Path(
        os.getenv("PIPELINE_SOURCE_OHLCVT_ROOT")
        or os.getenv("PIPELINE_SOURCE_PARQUET_ROOT")
        or resolve_path("source_ohlcvt_root", profile=pipeline_profile, env=path_config_env, required=False)
        or OHLCVT_PARQUET_ROOT
    )
    progress_every_seconds = max(5, int(os.getenv(spec.progress_seconds_env, "60")))
    ts_floor_production = int(production_start_ts())
    deadzone_by_task = _deadzone_by_task(spec.deadzone_env_prefix)
    logger = ModuleLogFn(spec.module_slug, log_file)
    log = logger

    naming = NumericForecastNamingConfig(
        module_slug=spec.module_slug,
        forecast_table_tag=spec.module_slug,
        eval_table_tag=f"{spec.module_slug}_eval",
        prediction_prefix=spec.prediction_prefix,
        task_short=dict(spec.task_short),
        task_label=dict(spec.task_label),
        log_prefix=spec.log_prefix,
    )
    io_config = NumericForecastIOConfig(
        naming=naming,
        parquet_root=parquet_root,
        staging_root=staging_root,
        state_root=state_root,
        scalar_root=scalar_root,
        ohlc_root=ohlc_root,
        parquet_compression=PARQUET_COMPRESSION,
        parquet_row_group=PARQUET_ROW_GROUP,
        log_fn=log,
        read_ohlcvt_fn=read_ohlcvt,
        list_assets_from_ohlcvt_fn=list_assets_from_ohlcvt,
        first_ohlcvt_ts_fn=first_ohlcvt_ts_from_disk,
        ohlcvt_max_ts_fn=ohlcvt_max_ts_from_parquet,
        feature_max_ts_fn=feature_max_ts_from_parquet,
    )
    engine_config = NumericForecastEngineConfig(
        io_config=io_config,
        task_label=dict(spec.task_label),
        task_short=dict(spec.task_short),
        future_label_columns=list(spec.future_label_columns),
        future_direction_deadzone=deadzone_by_task["log_return"],
        ts_floor_production=ts_floor_production,
        progress_every_seconds=progress_every_seconds,
        select_feature_columns_fn=spec.select_feature_columns_fn,
        training_window_bars_for_pair_fn=spec.training_window_bars_for_pair_fn,
        fit_model_fn=spec.fit_model_fn,
        predict_model_fn=spec.predict_model_fn,
        resolve_model_params_fn=spec.resolve_regressor_params_fn,
        model_profile_label_fn=spec.regressor_profile_label_fn,
        should_refit_fn=should_refit,
        env_int_fn=_env_int,
        forecast_resume_edge_env=spec.forecast_resume_edge_env,
        eval_resume_edge_env=spec.eval_resume_edge_env,
        source_start_env=spec.source_start_env,
        source_end_env=spec.source_end_env,
        work_start_env=spec.work_start_env,
    )

    horizon_bars_from_minutes = lambda horizon_minutes, interval_minutes: shared_horizon_bars_from_minutes(interval_minutes=interval_minutes, horizon_minutes=horizon_minutes)
    parse_train_window_months_fn = lambda raw: parse_train_window_months(raw, error_prefix=f"{spec.log_prefix}[error]")
    parse_refit_cadence_fn = lambda raw: parse_refit_cadence(raw, normalize_refit_cadence=spec.normalize_refit_cadence_fn, error_prefix=f"{spec.log_prefix}[error]")
    parse_requested_tasks_fn = lambda raw: parse_requested_tasks(raw, numeric_tasks=spec.numeric_tasks, error_prefix=f"{spec.log_prefix}[error]")
    parse_combo_list_fn = lambda raw: parse_combo_list(raw, numeric_tasks=spec.numeric_tasks, active_task_horizon_matrix=spec.active_task_horizon_matrix, error_prefix=f"{spec.log_prefix}[error]")
    parse_combo_window_list_fn = lambda raw: parse_combo_window_list(raw, parse_combo_list_fn=parse_combo_list_fn, parse_train_window_months_fn=parse_train_window_months_fn, error_prefix=f"{spec.log_prefix}[error]")
    parse_combo_profile_list_fn = lambda raw: parse_combo_profile_list(raw, parse_combo_list_fn=parse_combo_list_fn, parse_train_window_months_fn=parse_train_window_months_fn, parse_refit_cadence_fn=parse_refit_cadence_fn, error_prefix=f"{spec.log_prefix}[error]")
    validate_requested_task_horizon_pairs_fn = lambda tasks, horizon_minutes: validate_requested_task_horizon_pairs(tasks, horizon_minutes, active_task_horizon_matrix=spec.active_task_horizon_matrix, error_prefix=f"{spec.log_prefix}[error]")
    horizons_for_interval_fn = lambda interval, horizon_minutes_list: horizons_for_interval(interval, horizon_minutes_list, horizon_bars_from_minutes=horizon_bars_from_minutes)
    read_monthly_filtered_fn = lambda **kwargs: read_monthly_filtered(io_config, **kwargs)
    load_unit_feature_frame_fn = lambda asset, interval, start_ts, stop_ts: load_unit_feature_frame(io_config, asset, interval, start_ts, stop_ts)
    get_start_ts_fn = lambda asset, interval: get_start_ts(io_config, asset, interval)
    get_stop_ts_fn = lambda asset, interval: get_stop_ts(io_config, asset, interval)
    resolve_assets_fn = lambda intervals, assets_arg: resolve_assets(io_config, intervals, assets_arg)
    def compute_future_labels_fn(ohlc, horizon_bars, *, target_columns=None):
        return compute_future_labels(
            ohlc,
            horizon_bars,
            future_direction_deadzone=deadzone_by_task["log_return"],
            target_columns=target_columns,
        )
    model_state_path_fn = lambda asset, interval, horizon_minutes, task: model_state_path(io_config, asset, interval, horizon_minutes, task)
    load_model_state_fn = lambda asset, interval, horizon_minutes, task: load_model_state(io_config, asset, interval, horizon_minutes, task)
    save_model_state_fn = lambda asset, interval, horizon_minutes, task, state: save_model_state(io_config, asset, interval, horizon_minutes, task, state)
    module_table_fn = lambda store, interval: module_table(io_config, store, interval)
    expected_forecast_columns_fn = lambda task, horizon_minutes: expected_forecast_columns(io_config, task, horizon_minutes)
    expected_eval_columns_fn = lambda task, horizon_minutes: expected_eval_columns(io_config, task, horizon_minutes)
    expected_store_columns_fn = lambda store, task, horizon_minutes: expected_store_columns(io_config, store, task, horizon_minutes)
    month_part_path_fn = lambda root, interval, asset, year, month, store: month_part_path(io_config, root, interval, asset, year, month, store)
    validated_existing_month_parquet_fn = lambda dst, **kwargs: validated_existing_month_parquet(io_config, dst, **kwargs)
    validated_module_month_parquet_fn = lambda path, **kwargs: validated_module_month_parquet(io_config, path, **kwargs)
    write_month_parts_fn = lambda month_frames, **kwargs: write_month_parts(io_config, month_frames, **kwargs)
    stage_month_parts_fn = lambda month_frames, **kwargs: stage_month_parts(io_config, month_frames, **kwargs)
    walk_forward_predict_fn = lambda **kwargs: walk_forward_predict(engine_config, **kwargs)
    build_unit_work_fn = lambda **kwargs: build_unit_work(engine_config, **kwargs)
    compute_group_fn = lambda work_group: compute_group(engine_config, work_group)
    finalize_group_results_fn = lambda group_payload, writer_state, writer_cv: finalize_group_results(io_config, group_payload, writer_state, writer_cv)
    discover_tested_production_artifact_scope_fn = lambda project_root=None: discover_tested_production_artifact_scope(spec, project_root=project_root)

    def main() -> None:
        parser = argparse.ArgumentParser(description=f"Walk-forward {spec.family_name} numeric forecasting.")
        parser.add_argument("--profile", type=str, default=pipeline_profile)
        parser.add_argument("--intervals", type=str, default="")
        parser.add_argument("--assets", type=str, default="", help="Comma-delimited assets")
        parser.add_argument("--combo-profile-list", type=str, default="", help="Comma-delimited interval:horizon:task@Nm@cadence tuples")
        parser.add_argument("--combo-list", type=str, default="", help="Comma-delimited interval:horizon:task triples")
        parser.add_argument("--combo-window-list", type=str, default="", help="Comma-delimited interval:horizon:task@Nm tuples")
        parser.add_argument("--horizon-minutes", type=str, default="")
        parser.add_argument("--train-window-months", type=str, default="")
        parser.add_argument("--refit-cadence", type=str, default="")
        parser.add_argument("--tasks", type=str, default="")
        parser.add_argument("--mode", type=str, default="incremental", choices=["incremental", "backfill"])
        parser.add_argument("--unit-workers", type=int, default=get_workers(spec.module_slug, "unit_workers", spec.default_unit_workers))
        args = parser.parse_args()
        require_pipeline_io(profile=str(args.profile or pipeline_profile))
        args.unit_workers = max(1, int(args.unit_workers))
        resolved_model_threads = cap_model_threads(
            workers=int(args.unit_workers),
            model_threads=get_model_threads(spec.module_slug, spec.default_model_threads),
            max_logical_threads=spec.max_logical_threads,
        )
        for env_name in spec.thread_env_vars:
            os.environ[env_name] = str(int(resolved_model_threads))
        log_resolved_runtime(spec.module_slug, resolved={"unit_workers": int(args.unit_workers), "model_threads": int(resolved_model_threads), "writer_workers": 1})

        explicit_combo_list = parse_combo_list_fn(args.combo_list)
        explicit_combo_profile_list = parse_combo_profile_list_fn(args.combo_profile_list)
        explicit_combo_window_list = parse_combo_window_list_fn(args.combo_window_list)
        explicit_train_window_months = parse_train_window_months_fn(args.train_window_months)
        explicit_refit_cadence = parse_refit_cadence_fn(args.refit_cadence)
        explicit_intervals = [int(x.strip()) for x in args.intervals.split(",") if x.strip()]
        explicit_horizons = [int(x.strip()) for x in args.horizon_minutes.split(",") if x.strip()]
        tasks = parse_requested_tasks_fn(args.tasks)
        using_default_combo_selection = not explicit_combo_profile_list and not explicit_combo_window_list and not explicit_combo_list and not explicit_intervals and not explicit_horizons and not str(args.tasks).strip()
        production_scope = discover_existing_production_scope(spec, io_config) if using_default_combo_selection else None
        tested_scope = discover_tested_production_artifact_scope_fn() if using_default_combo_selection and production_scope is None else None
        if production_scope is not None:
            log(
                f"{spec.log_prefix}[production-defaults] state_root={io_config.state_root.resolve()} "
                f"combos={len(production_scope)} asset_scope=existing_production_scope"
            )
        elif tested_scope is not None:
            os.environ.setdefault("TABULAR_NUMERIC_FEATURE_SELECTION_FILE", str(tested_scope.feature_profile_json))
            log(
                f"{spec.log_prefix}[tested-defaults] handoff={tested_scope.handoff_path} "
                f"feature_profile_json={tested_scope.feature_profile_json} "
                f"cohort_assets={len(tested_scope.cohort_assets)} combos={len(tested_scope.combo_windows)} "
                f"asset_scope=full_production_universe"
            )

        if explicit_combo_profile_list:
            resolved_combo_profiles = list(explicit_combo_profile_list)
            resolved_combo_windows = [(int(interval), int(hm), str(task), int(months)) for interval, hm, task, months, _ in resolved_combo_profiles]
            resolved_combos = [(int(interval), int(hm), str(task)) for interval, hm, task, _, _ in resolved_combo_profiles]
        elif explicit_combo_window_list:
            resolved_combo_windows = list(explicit_combo_window_list)
            resolved_combos = [(int(interval), int(hm), str(task)) for interval, hm, task, _ in resolved_combo_windows]
            resolved_combo_profiles = [(int(interval), int(hm), str(task), int(months), str(explicit_refit_cadence) if explicit_refit_cadence else None) for interval, hm, task, months in resolved_combo_windows]
        elif explicit_combo_list:
            resolved_combos = list(explicit_combo_list)
            resolved_combo_windows = [
                (
                    int(interval),
                    int(hm),
                    str(task),
                    int(explicit_train_window_months if explicit_train_window_months is not None else spec.default_training_window_months_for_combo_fn(interval, hm, task)),
                )
                for interval, hm, task in resolved_combos
            ]
            resolved_combo_profiles = [(int(interval), int(hm), str(task), int(months), str(explicit_refit_cadence) if explicit_refit_cadence else None) for interval, hm, task, months in resolved_combo_windows]
        elif using_default_combo_selection:
            resolved_combo_windows = list(production_scope if production_scope is not None else tested_scope.combo_windows if tested_scope is not None else spec.resolve_default_combo_profile_fn())
            resolved_combos = [(int(interval), int(hm), str(task)) for interval, hm, task, _ in resolved_combo_windows]
            if explicit_train_window_months is not None:
                resolved_combo_windows = [(int(interval), int(hm), str(task), int(explicit_train_window_months)) for interval, hm, task in resolved_combos]
            resolved_combo_profiles = [
                (
                    int(interval),
                    int(hm),
                    str(task),
                    int(months),
                    str(explicit_refit_cadence) if explicit_refit_cadence is not None else spec.resolve_default_refit_policy_fn((int(interval), int(hm), str(task))),
                )
                for interval, hm, task, months in resolved_combo_windows
            ]
        else:
            intervals = explicit_intervals or list(spec.default_intervals)
            horizon_minutes_requested = sorted(set(int(h) for h in (explicit_horizons or spec.default_horizon_minutes)))
            validate_requested_task_horizon_pairs_fn(tasks, horizon_minutes_requested)
            resolved_combos = [(int(interval), int(hm), str(task)) for interval in intervals for task in tasks for hm in spec.active_task_horizon_matrix.get(task, []) if int(hm) in horizon_minutes_requested]
            resolved_combo_windows = [
                (
                    int(interval),
                    int(hm),
                    str(task),
                    int(explicit_train_window_months if explicit_train_window_months is not None else spec.default_training_window_months_for_combo_fn(interval, hm, task)),
                )
                for interval, hm, task in resolved_combos
            ]
            resolved_combo_profiles = [(int(interval), int(hm), str(task), int(months), str(explicit_refit_cadence) if explicit_refit_cadence else None) for interval, hm, task, months in resolved_combo_windows]
        if not resolved_combos:
            raise SystemExit(f"{spec.log_prefix}[error] no active interval/horizon/task combinations after filtering.")

        intervals = sorted({int(interval) for interval, _, _ in resolved_combos})
        active_pairs = sorted({(str(task), int(hm)) for _, hm, task in resolved_combos}, key=lambda item: (item[1], item[0]))
        assets = resolve_assets_fn(intervals=intervals, assets_arg=args.assets)
        if not assets:
            log(f"{spec.log_prefix} no assets discovered")
            return

        all_units: List[Tuple[str, int, int, int, str, int, int, Optional[str]]] = []
        for asset in assets:
            for interval, hm, task, training_window_months, refit_cadence in resolved_combo_profiles:
                hb = horizon_bars_from_minutes(int(hm), int(interval))
                twb = spec.training_window_bars_from_months_fn(int(training_window_months), int(interval))
                all_units.append((asset, int(interval), int(hm), int(hb), str(task), int(training_window_months), int(twb), (str(refit_cadence) if refit_cadence else None)))
        planning_workers = _resolve_planning_workers(int(args.unit_workers), len(all_units), minimum_workers=MIN_PLANNING_WORKERS)

        log(
            f"{spec.log_prefix}[plan] assets={len(assets)} intervals={len(intervals)} active_pairs={len(active_pairs)} "
            f"tasks={len(tasks)} total_units={len(all_units)} unit_workers={int(args.unit_workers)} model_threads={int(resolved_model_threads)} "
            f"planning_workers={int(planning_workers)} "
            f"progress_seconds={int(progress_every_seconds)} resolved_combos={format_combo_list(resolved_combos)} "
            f"resolved_combo_windows={format_combo_window_list(resolved_combo_windows)} "
            f"resolved_combo_profiles={format_combo_window_refit_list(resolved_combo_profiles)}"
        )
        active_profile_map = {
            f"{int(interval)}:{int(hm)}:{str(task)}@{int(months)}m": {
                "profile": spec.regressor_profile_label_fn(
                    str(task),
                    interval_minutes=int(interval),
                    horizon_minutes=int(hm),
                    training_window_months=int(months),
                ),
                spec.thread_param_name: int(
                    spec.resolve_regressor_params_fn(
                        task=str(task),
                        model_threads=int(resolved_model_threads),
                        interval_minutes=int(interval),
                        horizon_minutes=int(hm),
                        training_window_months=int(months),
                    ).get(spec.thread_param_name, resolved_model_threads)
                ),
            }
            for interval, hm, task, months in resolved_combo_windows
        }
        log(f"{spec.log_prefix}[plan] active_regressor_profiles={json.dumps(active_profile_map, sort_keys=True)}")

        pair_counts: Dict[Tuple[int, int, int, str], int] = {}
        ordered_units = sorted(all_units, key=lambda x: (x[1], x[2], x[3], x[4], x[5], str(x[7] or ""), x[0]))
        for asset, interval, hm, hb, task, training_window_months, training_window_bars, refit_cadence in ordered_units:
            pair_key = (int(interval), int(hm), int(hb), str(task))
            pair_counts[pair_key] = pair_counts.get(pair_key, 0) + 1

        asset_interval_keys = sorted({(str(asset), int(interval)) for asset, interval, *_ in ordered_units})
        def _load_asset_interval_context(key: Tuple[str, int]) -> Tuple[Tuple[str, int], Dict[str, Optional[int]]]:
            asset_name, interval_minutes = key
            return key, {
                "source_start": get_start_ts(io_config, asset_name, int(interval_minutes)),
                "ohlc_edge": get_stop_ts(io_config, asset_name, int(interval_minutes)),
                "scalar_edge": io_config.feature_max_ts_fn(int(interval_minutes), asset_name, root=io_config.scalar_root),
                "forecast_edge": module_output_max_ts(io_config, root=io_config.parquet_root, interval=int(interval_minutes), asset=asset_name, store="forecast"),
                "eval_edge": module_output_max_ts(io_config, root=io_config.parquet_root, interval=int(interval_minutes), asset=asset_name, store="eval"),
            }

        asset_interval_context: Dict[Tuple[str, int], Dict[str, Optional[int]]] = {}
        preload_workers = _resolve_planning_workers(int(args.unit_workers), len(asset_interval_keys), minimum_workers=MIN_PLANNING_WORKERS)
        if preload_workers <= 1:
            for key in asset_interval_keys:
                k, payload = _load_asset_interval_context(key)
                asset_interval_context[k] = payload
        else:
            with ThreadPoolExecutor(max_workers=int(preload_workers)) as preload_pool:
                for key, payload in preload_pool.map(_load_asset_interval_context, asset_interval_keys):
                    asset_interval_context[key] = payload

        work_items: List[UnitWork] = []
        if planning_workers <= 1 or len(ordered_units) <= 1:
            for asset, interval, hm, hb, task, training_window_months, training_window_bars, refit_cadence in ordered_units:
                work = build_unit_work_fn(
                    asset=asset,
                    interval=int(interval),
                    horizon_minutes=int(hm),
                    horizon_bars=int(hb),
                    task=str(task),
                    training_window_months=int(training_window_months),
                    training_window_bars=int(training_window_bars),
                    refit_cadence=(str(refit_cadence) if refit_cadence else None),
                    model_threads=int(resolved_model_threads),
                    prefetched_context=dict(asset_interval_context.get((str(asset), int(interval)), {})),
                )
                if work is not None:
                    work_items.append(work)
        else:
            plan_jobs = [
                dict(
                    asset=asset,
                    interval=int(interval),
                    horizon_minutes=int(hm),
                    horizon_bars=int(hb),
                    task=str(task),
                    training_window_months=int(training_window_months),
                    training_window_bars=int(training_window_bars),
                    refit_cadence=(str(refit_cadence) if refit_cadence else None),
                    model_threads=int(resolved_model_threads),
                    prefetched_context=dict(asset_interval_context.get((str(asset), int(interval)), {})),
                )
                for asset, interval, hm, hb, task, training_window_months, training_window_bars, refit_cadence in ordered_units
            ]
            try:
                with ThreadPoolExecutor(max_workers=int(planning_workers)) as planning_pool:
                    for work in planning_pool.map(lambda job: build_unit_work_fn(**job), plan_jobs):
                        if work is not None:
                            work_items.append(work)
            except Exception as exc:
                log(f"{spec.log_prefix}[runtime-fallback] planning pool unavailable; forcing serial work planning: {exc}")
                for job in plan_jobs:
                    work = build_unit_work_fn(**job)
                    if work is not None:
                        work_items.append(work)
        for (interval, horizon_minutes, horizon_bars, task), asset_total in sorted(pair_counts.items()):
            queued = sum(1 for w in work_items if int(w.interval) == int(interval) and int(w.horizon_minutes) == int(horizon_minutes) and int(w.horizon_bars) == int(horizon_bars) and str(w.task) == str(task))
            log(f"{spec.log_prefix}[group-start] k={interval} h={horizon_minutes}m({horizon_bars}b) task={task} assets_total={asset_total} work_items={queued}")

        grouped_work_map: Dict[Tuple[str, int, int, int], List[UnitWork]] = {}
        for work in work_items:
            group_key = (str(work.asset), int(work.interval), int(work.horizon_minutes), int(work.horizon_bars))
            grouped_work_map.setdefault(group_key, []).append(work)
        group_work_items = [
            HorizonGroupWork(
                group_id=f"{asset}|{int(interval)}|{int(hm)}|{int(hb)}",
                asset=str(asset),
                interval=int(interval),
                horizon_minutes=int(hm),
                horizon_bars=int(hb),
                works=sorted(grouped_units, key=lambda w: str(w.task)),
                model_threads=int(resolved_model_threads),
            )
            for (asset, interval, hm, hb), grouped_units in sorted(grouped_work_map.items(), key=lambda item: (item[0][1], item[0][2], item[0][3], item[0][0]))
        ]

        total_work_items = len(work_items)
        mp_ctx = mp.get_context("spawn")
        write_queue, parallel_allowed = _create_stage_queue_with_retry(
            mp_ctx=mp_ctx,
            log=log,
            log_prefix=spec.log_prefix,
            retries=STAGE_PARALLEL_INIT_RETRIES,
            retry_seconds=STAGE_PARALLEL_INIT_RETRY_SECONDS,
        )
        init_stage_write_queue(write_queue)
        writer_state: Dict[str, Any] = {}
        writer_cv = threading.Condition()
        writer_thread = threading.Thread(
            target=stage_writer_loop,
            args=(io_config, write_queue, writer_state, writer_cv),
            name=f"{spec.prediction_prefix}_numeric_stage_writer",
            daemon=True,
        )
        writer_thread.start()
        unit_results: List[Dict[str, Any]] = []
        completed_units = 0
        try:
            if (not parallel_allowed) or args.unit_workers <= 1 or len(group_work_items) <= 1:
                for idx, gw in enumerate(group_work_items, start=1):
                    log(f"{spec.log_prefix}[dispatch] mode=serial idx={idx}/{len(group_work_items)} asset={gw.asset} k={gw.interval} h={gw.horizon_minutes}m tasks={','.join(w.task for w in gw.works)}")
                    group_results = finalize_group_results_fn(compute_group(engine_config, gw), writer_state, writer_cv)
                    unit_results.extend(group_results)
                    for res in group_results:
                        completed_units += 1
                        log(f"{spec.log_prefix}[dispatch] mode=serial completed={completed_units}/{total_work_items} asset={res.get('asset')} k={int(res.get('interval', 0) or 0)} h={int(res.get('horizon_minutes', 0) or 0)}m task={res.get('task')}")
            else:
                parallel_dispatch_done = False
                last_parallel_exc: Optional[Exception] = None
                for attempt in range(1, int(STAGE_PARALLEL_INIT_RETRIES) + 1):
                    try:
                        with ProcessPoolExecutor(max_workers=min(max(1, int(args.unit_workers)), len(group_work_items)), mp_context=mp_ctx, initializer=init_stage_write_queue, initargs=(write_queue,)) as ex:
                            fut_map = {ex.submit(compute_group, engine_config, gw): (gw.asset, int(gw.interval), int(gw.horizon_minutes), int(gw.horizon_bars)) for gw in group_work_items}
                            for fut in as_completed(fut_map):
                                _asset_done, _interval, _horizon_minutes, horizon_bars = fut_map[fut]
                                group_results = finalize_group_results_fn(fut.result(), writer_state, writer_cv)
                                unit_results.extend(group_results)
                                for res in group_results:
                                    completed_units += 1
                                    log(f"{spec.log_prefix}[dispatch] mode=parallel completed={completed_units}/{total_work_items} asset={res.get('asset')} k={int(res.get('interval', 0) or 0)} h={int(res.get('horizon_minutes', 0) or 0)}m({horizon_bars}b) task={res.get('task')}")
                        parallel_dispatch_done = True
                        break
                    except Exception as exc:
                        last_parallel_exc = exc
                        if attempt >= int(STAGE_PARALLEL_INIT_RETRIES):
                            break
                        log(
                            f"{spec.log_prefix}[runtime-retry] process pool init attempt={attempt}/{int(STAGE_PARALLEL_INIT_RETRIES)} "
                            f"failed; retrying in {STAGE_PARALLEL_INIT_RETRY_SECONDS:.2f}s: {exc}"
                        )
                        time.sleep(float(STAGE_PARALLEL_INIT_RETRY_SECONDS))
                if not parallel_dispatch_done:
                    log(f"{spec.log_prefix}[runtime-fallback] process pool unavailable after retries; forcing serial execution: {last_parallel_exc}")
                    for idx, gw in enumerate(group_work_items, start=1):
                        log(f"{spec.log_prefix}[dispatch] mode=serial idx={idx}/{len(group_work_items)} asset={gw.asset} k={gw.interval} h={gw.horizon_minutes}m tasks={','.join(w.task for w in gw.works)}")
                        group_results = finalize_group_results_fn(compute_group(engine_config, gw), writer_state, writer_cv)
                        unit_results.extend(group_results)
                        for res in group_results:
                            completed_units += 1
                            log(f"{spec.log_prefix}[dispatch] mode=serial completed={completed_units}/{total_work_items} asset={res.get('asset')} k={int(res.get('interval', 0) or 0)} h={int(res.get('horizon_minutes', 0) or 0)}m task={res.get('task')}")
        finally:
            write_queue.put({"kind": "stop"})
            writer_thread.join()

        for res in unit_results:
            if res.get("empty"):
                continue
            log(
                f"{spec.log_prefix} asset={res.get('asset')} k={int(res.get('interval', 0) or 0)} "
                f"h={int(res.get('horizon_minutes', 0) or 0)}m({horizon_bars_from_minutes(int(res.get('horizon_minutes', 0) or 0), int(res.get('interval', 1) or 1))}b) "
                f"task={res.get('task')} rows_written={int(res.get('rows_written', 0) or 0)} "
                f"work=[{int(res.get('work_start_ts', 0) or 0)},{int(res.get('work_end_ts', 0) or 0)}]"
            )
        log(
            f"{spec.log_prefix} run complete forecast_parts={sum(len(r.get('pred_parts', [])) for r in unit_results)} "
            f"eval_parts={sum(len(r.get('eval_parts', [])) for r in unit_results)} "
            f"rows_dropped_pre_floor_total={sum(int(r.get('rows_dropped', 0) or 0) for r in unit_results)} "
            f"assets_with_start={sum(1 for r in unit_results if r.get('ts_start') is not None)}"
        )

    return {
        "PIPELINE_ROOT": pipeline_root,
        "LOG_DIR": log_dir,
        "LOG_FILE": log_file,
        "STATE_ROOT": state_root,
        "PARQUET_ROOT": parquet_root,
        "STAGING_ROOT": staging_root,
        "SCALAR_ROOT": scalar_root,
        "FAMILY": spec.family_name,
        "DOMAIN": "Numerics",
        "PROGRESS_EVERY_SECONDS": progress_every_seconds,
        "TS_FLOOR_PRODUCTION": ts_floor_production,
        "DEADZONE_BY_TASK": deadzone_by_task,
        "log": log,
        "NAMING": naming,
        "IO_CONFIG": io_config,
        "ENGINE_CONFIG": engine_config,
        "_env_int": _env_int,
        "_horizon_bars_from_minutes": shared_horizon_bars_from_minutes,
        "horizon_bars_from_minutes": horizon_bars_from_minutes,
        "_parse_train_window_months": parse_train_window_months_fn,
        "_parse_refit_cadence": parse_refit_cadence_fn,
        "_parse_requested_tasks": parse_requested_tasks_fn,
        "_parse_combo_list": parse_combo_list_fn,
        "_parse_combo_window_list": parse_combo_window_list_fn,
        "_parse_combo_profile_list": parse_combo_profile_list_fn,
        "_validate_requested_task_horizon_pairs": validate_requested_task_horizon_pairs_fn,
        "_horizons_for_interval": horizons_for_interval_fn,
        "_read_monthly_filtered": read_monthly_filtered_fn,
        "_load_unit_feature_frame": load_unit_feature_frame_fn,
        "_get_start_ts": get_start_ts_fn,
        "_get_stop_ts": get_stop_ts_fn,
        "_resolve_assets": resolve_assets_fn,
        "_safe_log_return": safe_log_return,
        "_safe_true_range": safe_true_range,
        "_future_window_views": future_window_views,
        "_compute_future_labels": compute_future_labels_fn,
        "_model_state_path": model_state_path_fn,
        "_load_model_state": load_model_state_fn,
        "_save_model_state": save_model_state_fn,
        "_state_to_bundle": state_to_bundle,
        "_bundle_to_state": bundle_to_state,
        "_chunk_end_ts_for_month": chunk_end_ts_for_month,
        "_fit_regressor": spec.fit_model_fn,
        "_predict_block": spec.predict_model_fn,
        "_module_table": module_table_fn,
        "_expected_forecast_columns": expected_forecast_columns_fn,
        "_expected_eval_columns": expected_eval_columns_fn,
        "_expected_store_columns": expected_store_columns_fn,
        "_month_part_path": month_part_path_fn,
        "_validated_existing_month_parquet": validated_existing_month_parquet_fn,
        "_pair_columns_state": pair_columns_state,
        "_validated_module_month_parquet": validated_module_month_parquet_fn,
        "_coalesce_keyed_frames": coalesce_keyed_frames,
        "_write_month_parts": write_month_parts_fn,
        "_stage_month_parts": stage_month_parts_fn,
        "_init_stage_write_queue": init_stage_write_queue,
        "_walk_forward_predict": walk_forward_predict_fn,
        "_build_unit_work": build_unit_work_fn,
        "_compute_group": compute_group_fn,
        "_discover_existing_production_scope": lambda: discover_existing_production_scope(spec, io_config),
        "_discover_tested_production_artifact_scope": discover_tested_production_artifact_scope_fn,
        "_stage_writer_loop": lambda write_queue, writer_state, writer_cv: stage_writer_loop(io_config, write_queue, writer_state, writer_cv),
        "_finalize_group_results": finalize_group_results_fn,
        "main": main,
    }

