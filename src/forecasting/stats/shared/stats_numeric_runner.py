from __future__ import annotations

import argparse
import importlib
import os
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, ThreadPoolExecutor, wait
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import pandas as pd

from src.forecasting.common.ml_module_utils import acquire_single_run_lock as _shared_acquire_single_run_lock
from src.forecasting.common.forecast_family_core import (
    discover_edge_and_min,
    fit_window_start,
    forecast_output_tail_ts,
    horizon_bars,
    parse_int_csv,
    parse_str_csv,
)
from src.forecasting.common.pipeline_parquet_utils import decide_range_from_disk_edges
from src.forecasting.common.runtime_config import DispatchPressureGuard, cap_model_threads, get_model_threads, get_workers, log_resolved_runtime
from src.forecasting.common.stats_module_utils import (
    CAPABILITY_MATRIX,
    DEFAULT_BACKFILL_DAYS,
    DEFAULT_CONF_ALPHA,
    DEFAULT_WORKERS,
    NUMERIC_TASK_TO_TARGET_COLUMN,
    configure_stats_source_roots,
    make_stats_unit_key,
    resolved_stats_source_roots,
    resolve_assets,
    utc_now_iso,
    write_forecast_parts as _shared_write_forecast_parts,
    write_json_atomic as _shared_write_json_atomic,
)
from src.forecasting.ml.shared.numeric_runner_common import (
    NumericExistingProductionScope,
    NumericTestedProductionArtifactScope,
    TailPlanningCache,
    build_asset_shard_jobs,
    build_horizon_group_shard_jobs,
    canonical_physical_io_config,
    canonical_physical_naming,
    canonical_physical_output_tail_ts,
    discover_existing_combo_specs_from_canonical_physical_output,
    discover_existing_combo_specs_from_partitioned_output,
    discover_numeric_tested_production_artifact_scope,
    discover_tested_production_artifact_payload,
    execute_grouped_horizon_jobs,
    forward_fill_rows_to_edge,
    parse_combo_list,
    partitioned_prediction_writer_loop,
    plan_asset_work_span,
    prediction_eval_row,
    raise_writer_fatal,
    require_production_stream_scope_contract,
    resolve_dispatch_slots,
    resolve_production_stream_scope_contract,
    spill_rows_chunk,
    staging_root,
    start_partitioned_prediction_writer,
    validate_combo_completion,
    wait_for_writer_drain,
    write_canonical_physical_prediction_month_frames,
    write_canonical_physical_predictions,
)
from src.forecasting.ml.shared.numeric_runner_diagnostics import (
    append_diagnostic_event as _shared_append_diagnostic_event,
    diagnostics_file as _shared_diagnostics_file,
    reset_diagnostics_file as _shared_reset_diagnostics_file,
    resource_snapshot,
)
from src.forecasting.common.sandbox_paths import SandboxOutputRoots, assert_write_allowed, resolve_sandbox_output_roots

StatsProcessUnitFn = Callable[..., Dict[str, Any]]
StatsDependencyCheckFn = Callable[[], Optional[str]]
StatsLoggerFn = Callable[[str], None]
StatsExtraArgsFn = Callable[[argparse.ArgumentParser], None]
StatsExtraKwargsFn = Callable[[argparse.Namespace], Dict[str, Any]]
StatsManifestExtrasFn = Callable[[argparse.Namespace], Dict[str, Any]]
TestedProductionArtifactScope = NumericTestedProductionArtifactScope
ExistingProductionScope = NumericExistingProductionScope


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


def _stats_sandbox_log_fn(branch: str) -> StatsLoggerFn:
    roots = _sandbox_roots()
    log_path = _stats_sandbox_log_path(branch, roots=roots)

    def _log(message: str) -> None:
        line = f"[{utc_now_iso()}] {message}"
        print(line, flush=True)
        assert_write_allowed(log_path, "Stats log file", roots=_sandbox_roots())
        try:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
        except Exception:
            pass

    return _log


def _stats_sandbox_log_path(branch: str, *, roots: SandboxOutputRoots) -> Path:
    log_root = _sandbox_env_path(
        roots,
        "PIPELINE_SANDBOX_LOG_ROOT",
        roots.log_root,
        "Stats log root",
    ) / "stats_numeric_runner"
    assert_write_allowed(log_root, "Stats log root", roots=roots)
    log_path = log_root / f"{str(branch)}.log"
    assert_write_allowed(log_path, "Stats log file", roots=roots)
    return log_path.resolve()


def _rebase_branch_module_globals(spec: StatsNumericModuleSpec, *, forecast_root: Path, state_root: Path, roots: SandboxOutputRoots) -> None:
    try:
        module = importlib.import_module(str(spec.process_unit_fn.__module__))
    except Exception:
        return
    source_parquet_root = Path(
        os.getenv("PIPELINE_SOURCE_PARQUET_ROOT")
        or os.getenv("PIPELINE_PARQUET_ROOT", str(getattr(module, "PARQUET_ROOT", forecast_root)))
    ).resolve()
    source_feature_root = Path(
        os.getenv("PIPELINE_SOURCE_FEATURES_ROOT")
        or os.getenv("PIPELINE_PARQUET_FEATURES_ROOT", str(getattr(module, "FEATURE_ROOT", forecast_root)))
    ).resolve()
    assignments = {
        "PARQUET_ROOT": source_parquet_root,
        "FEATURE_ROOT": source_feature_root,
        "FORECAST_ROOT": forecast_root.resolve(),
        "STATE_ROOT": state_root.resolve(),
        "MANIFEST_FILE": (state_root / f"{str(spec.branch)}_run_manifest.json").resolve(),
        "SKIPPED_FILE": (state_root / f"{str(spec.branch)}_skipped.json").resolve(),
        "LOG_FILE": _stats_sandbox_log_path(spec.branch, roots=roots),
    }
    for name, value in assignments.items():
        try:
            setattr(module, name, value)
        except Exception:
            pass


def materialize_runtime_spec(spec: StatsNumericModuleSpec) -> StatsNumericModuleSpec:
    roots = _sandbox_roots()
    if not roots.enabled:
        return spec
    forecast_root = _sandbox_env_path(
        roots,
        "PIPELINE_SANDBOX_PARQUET_ROOT",
        roots.parquet_root,
        "Stats forecast parquet root",
    ) / str(spec.family_root_name)
    state_root = _sandbox_env_path(
        roots,
        "PIPELINE_SANDBOX_STATE_ROOT",
        roots.state_root,
        "Stats state root",
    ) / "stats_numeric_runner" / str(spec.family_tag)
    assert_write_allowed(forecast_root, "Stats forecast root", roots=roots)
    assert_write_allowed(state_root, "Stats state root", roots=roots)
    _rebase_branch_module_globals(spec, forecast_root=forecast_root, state_root=state_root, roots=roots)
    return replace(
        spec,
        forecast_root=forecast_root.resolve(),
        state_root=state_root.resolve(),
        manifest_file=(state_root / f"{str(spec.branch)}_run_manifest.json").resolve(),
        skipped_file=(state_root / f"{str(spec.branch)}_skipped.json").resolve(),
        log_fn=_stats_sandbox_log_fn(spec.branch),
    )


def diagnostics_file(state_root: Path, branch: str) -> Path:
    roots = _sandbox_roots()
    if not roots.enabled:
        return _shared_diagnostics_file(state_root, branch)
    diagnostics_root = _sandbox_env_path(
        roots,
        "PIPELINE_SANDBOX_DIAGNOSTICS_ROOT",
        roots.diagnostics_root,
        "Stats diagnostics root",
    )
    return diagnostics_root / "stats_numeric_runner" / f"{str(branch)}_run_diagnostics.jsonl"


def append_diagnostic_event(path: Path, event: str, payload: Dict[str, Any], *, timestamp_fn: Any = None) -> None:
    assert_write_allowed(path, "Stats diagnostics JSONL", roots=_sandbox_roots())
    _shared_append_diagnostic_event(path, event, payload, timestamp_fn=timestamp_fn)


def reset_diagnostics_file(path: Path) -> None:
    assert_write_allowed(path, "Stats diagnostics JSONL reset", roots=_sandbox_roots())
    _shared_reset_diagnostics_file(path)


def write_json_atomic(path: Path, payload: Dict[str, Any]) -> None:
    roots = _sandbox_roots()
    assert_write_allowed(path, "Stats runner JSON", roots=roots)
    assert_write_allowed(path.with_name(f"{path.name}.{os.getpid()}.tmp"), "Stats runner JSON temp", roots=roots)
    _shared_write_json_atomic(path, payload)


def write_forecast_parts(**kwargs: Any) -> List[Dict[str, Any]]:
    out_root = Path(kwargs["out_root"])
    assert_write_allowed(out_root, "Stats forecast parts root", roots=_sandbox_roots())
    return _shared_write_forecast_parts(**kwargs)


def _staging_root(forecast_root: Path, family_tag: str) -> Path:
    roots = _sandbox_roots()
    if not roots.enabled:
        return staging_root(forecast_root, family_tag)
    tmp_root = _sandbox_env_path(
        roots,
        "PIPELINE_SANDBOX_TMP_ROOT",
        roots.tmp_root,
        "Stats staging tmp root",
    )
    root = tmp_root / "stats_numeric_runner" / f"{str(family_tag)}_stage"
    assert_write_allowed(root, "Stats staging root", roots=roots)
    root.mkdir(parents=True, exist_ok=True)
    return root.resolve()


def _spill_rows_chunk(**kwargs: Any) -> str:
    stage_root = Path(kwargs["staging_root"])
    assert_write_allowed(stage_root / "worker_row_batches", "Stats worker row-batch staging root", roots=_sandbox_roots())
    return spill_rows_chunk(**kwargs)


def _acquire_single_run_lock(state_root: Path, run_name: str) -> Path:
    roots = _sandbox_roots()
    assert_write_allowed(state_root, "Stats lock state root", roots=roots)
    lock_path = Path(state_root) / f"{str(run_name)}.lock"
    assert_write_allowed(lock_path, "Stats lock", roots=roots)
    return _shared_acquire_single_run_lock(state_root, run_name)


@dataclass(frozen=True)
class StatsNumericModuleSpec:
    branch: str
    family: str
    domain: str
    family_tag: str
    model_id: str
    model_version: str
    family_root_name: str
    family_root_env: str
    forecast_root: Path
    state_root: Path
    manifest_file: Path
    skipped_file: Path
    default_intervals: Sequence[int]
    default_horizons: Sequence[int]
    default_tasks: Sequence[str]
    min_train_bars: int
    train_windows_bars: Sequence[int]
    process_unit_fn: StatsProcessUnitFn
    log_fn: StatsLoggerFn
    dependency_check_fn: StatsDependencyCheckFn = field(default=lambda: None)
    supports_conf_alpha: bool = False
    add_extra_args_fn: StatsExtraArgsFn = field(default=lambda parser: None)
    extra_process_kwargs_fn: StatsExtraKwargsFn = field(default=lambda args: {})
    manifest_extras_fn: StatsManifestExtrasFn = field(default=lambda args: {})
    progress_seconds_env: str = "STATS_NUMERIC_PROGRESS_SECONDS"


def _load_registered_module_spec(branch: str) -> StatsNumericModuleSpec:
    from src.forecasting.stats.shared.stats_numeric_model_registry import STATS_NUMERIC_ENTRYPOINTS

    module_name = STATS_NUMERIC_ENTRYPOINTS.get(str(branch))
    if not module_name:
        raise RuntimeError(f"No registered Stats numeric entrypoint for branch={branch!r}")
    module = importlib.import_module(module_name)
    module_spec = getattr(module, "MODULE_SPEC", None)
    if not isinstance(module_spec, StatsNumericModuleSpec):
        raise RuntimeError(f"Registered Stats entrypoint {module_name} does not expose MODULE_SPEC")
    return module_spec


def _default_tasks(spec: StatsNumericModuleSpec) -> List[str]:
    supported = CAPABILITY_MATRIX.get(spec.branch, {}).get("numerics", ())
    return list(spec.default_tasks or supported)


def _add_common_args(parser: argparse.ArgumentParser, spec: StatsNumericModuleSpec) -> None:
    parser.add_argument("--intervals", type=str, default=",".join(str(x) for x in spec.default_intervals))
    parser.add_argument("--horizons_minutes", type=str, default=",".join(str(x) for x in spec.default_horizons))
    parser.add_argument("--assets", type=str, default="")
    parser.add_argument("--tasks", type=str, default=",".join(_default_tasks(spec)))
    parser.add_argument("--combo-list", type=str, default="")
    parser.add_argument("--parquet-root", type=Path, default=None)
    parser.add_argument("--mode", type=str, default="backfill", choices=("backfill", "resume"))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--fit_days", type=int, default=None)
    parser.add_argument("--backfill_days", type=int, default=DEFAULT_BACKFILL_DAYS)
    parser.add_argument("--workers", type=int, default=get_workers("stats_forecasters", "asset_workers", DEFAULT_WORKERS))
    parser.add_argument("--model_threads", "--model-threads", dest="model_threads", type=int, default=None)
    parser.add_argument(
        "--execution_backend",
        "--execution-backend",
        dest="execution_backend",
        type=str,
        default=str(os.getenv("STATS_NUMERIC_EXECUTION_BACKEND", "process")).strip().lower() or "process",
        choices=("process", "thread"),
    )
    parser.add_argument("--predict_latest_only", action="store_true")
    parser.add_argument("--fill_to_edge", action="store_true")
    if spec.supports_conf_alpha:
        parser.add_argument("--conf_alpha", type=float, default=DEFAULT_CONF_ALPHA)
    spec.add_extra_args_fn(parser)


def _configure_runtime(args: argparse.Namespace) -> int:
    args.workers = max(1, int(args.workers))
    env_threads = str(os.getenv("STATS_NUMERIC_MODEL_THREADS", "")).strip()
    if args.model_threads is not None:
        resolved_model_threads = max(1, int(args.model_threads))
        args.model_threads_source = "cli_profile"
    elif env_threads:
        resolved_model_threads = max(1, int(env_threads))
        args.model_threads_source = "env_profile"
    else:
        resolved_model_threads = cap_model_threads(
            workers=int(args.workers),
            model_threads=get_model_threads("stats_forecasters", 2),
            max_logical_threads=32,
        )
        args.model_threads_source = "runtime_config_capped"
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = str(int(resolved_model_threads))
    log_resolved_runtime(
        "stats_forecasters",
        resolved={
            "asset_workers": int(args.workers),
            "model_threads": int(resolved_model_threads),
            "model_threads_source": str(args.model_threads_source),
            "writer_workers": 1,
        },
    )
    return int(resolved_model_threads)


def _apply_explicit_source_roots(args: argparse.Namespace, spec: StatsNumericModuleSpec) -> Dict[str, str]:
    if getattr(args, "parquet_root", None) is None:
        roots = resolved_stats_source_roots()
        setattr(args, "resolved_source_roots", roots)
        return roots

    source_root = Path(args.parquet_root).expanduser().resolve()
    args.parquet_root = source_root
    for name in (
        "PIPELINE_SOURCE_PARQUET_ROOT",
        "PIPELINE_SOURCE_OHLCVT_ROOT",
        "PIPELINE_SOURCE_FEATURES_ROOT",
        "PIPELINE_PARQUET_FEATURES_ROOT",
    ):
        os.environ[name] = str(source_root)
    roots = configure_stats_source_roots(parquet_root=source_root, feature_root=source_root)
    try:
        module = importlib.import_module(str(spec.process_unit_fn.__module__))
    except Exception:
        module = None
    if module is not None:
        for attr in ("PARQUET_ROOT", "FEATURE_ROOT"):
            try:
                setattr(module, attr, source_root)
            except Exception:
                pass
    setattr(args, "resolved_source_roots", roots)
    return roots


def _append_diagnostic_event(path: Path, event: str, payload: Dict[str, Any]) -> None:
    append_diagnostic_event(path, event, payload, timestamp_fn=utc_now_iso)


def _collect_rows_by_month(rows: Sequence[Dict[str, Any]]) -> Dict[Tuple[int, int], List[pd.DataFrame]]:
    if not rows:
        return {}
    dfw = pd.DataFrame(list(rows))
    dt = pd.to_datetime(dfw["ts"], unit="s", utc=True)
    dfw["year"] = dt.dt.year.astype(int)
    dfw["month"] = dt.dt.month.astype(int)
    out: Dict[Tuple[int, int], List[pd.DataFrame]] = {}
    for (year, month), grp in dfw.groupby(["year", "month"], sort=True):
        out.setdefault((int(year), int(month)), []).append(grp.drop(columns=["year", "month"]).copy())
    return out


def _writer_loop(*, forecast_root: Path, write_queue: Any, writer_state: Dict[str, Any], canonical_io_config: Any = None) -> None:
    if canonical_io_config is not None:
        def _write_month_frames(**kwargs: Any) -> List[Dict[str, Any]]:
            store = "eval" if Path(kwargs.get("out_root", "")).name == "eval" else "forecast"
            return write_canonical_physical_prediction_month_frames(
                io_config=canonical_io_config,
                store=store,
                **{k: v for k, v in kwargs.items() if k != "out_root"},
            )

        def _write_predictions(**kwargs: Any) -> List[Dict[str, Any]]:
            store = "eval" if Path(kwargs.get("out_root", "")).name == "eval" else "forecast"
            return write_canonical_physical_predictions(
                io_config=canonical_io_config,
                store=store,
                **kwargs,
            )

        partitioned_prediction_writer_loop(
            forecast_root=forecast_root,
            write_queue=write_queue,
            writer_state=writer_state,
            write_partitioned_predictions_fn=_write_predictions,
            write_partitioned_prediction_month_frames_fn=_write_month_frames,
        )
        return

    def _write_month_frames(
        *,
        out_root: Path,
        interval_minutes: int,
        run_id: str,
        module_tag: str,
        task: str,
        horizon_minutes: int,
        month_frames: Dict[Tuple[int, int], List[pd.DataFrame]],
        existing_key_cache: Optional[Dict[Any, Any]] = None,
    ) -> List[Dict[str, Any]]:
        del existing_key_cache
        return write_forecast_parts(
            monthly_frames=month_frames,
            out_root=out_root,
            family_tag=str(module_tag),
            interval_minutes=int(interval_minutes),
            task=str(task),
            horizon_minutes=int(horizon_minutes),
            run_id=str(run_id),
        )

    def _write_predictions(
        *,
        out_root: Path,
        interval_minutes: int,
        run_id: str,
        module_tag: str,
        task: str,
        horizon_minutes: int,
        df: pd.DataFrame,
        existing_key_cache: Optional[Dict[Any, Any]] = None,
    ) -> List[Dict[str, Any]]:
        del existing_key_cache
        return write_forecast_parts(
            monthly_frames=_collect_rows_by_month(df.to_dict(orient="records")),
            out_root=out_root,
            family_tag=str(module_tag),
            interval_minutes=int(interval_minutes),
            task=str(task),
            horizon_minutes=int(horizon_minutes),
            run_id=str(run_id),
        )

    partitioned_prediction_writer_loop(
        forecast_root=forecast_root,
        write_queue=write_queue,
        writer_state=writer_state,
        write_partitioned_predictions_fn=_write_predictions,
        write_partitioned_prediction_month_frames_fn=_write_month_frames,
    )


def _normalize_prediction_rows(
    rows: Sequence[Dict[str, Any]],
    *,
    spec: StatsNumericModuleSpec,
    interval: int,
    horizon_minutes: int,
    task: str,
    run_id: str,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    nonnegative_task = str(task) in {"realized_vol", "true_range", "max_runup"}
    for row in rows:
        d = dict(row)
        d["asset"] = str(d["asset"])
        d["ts"] = int(d["ts"])
        d.setdefault("interval_min", int(interval))
        d.setdefault("horizon_min", int(horizon_minutes))
        d.setdefault("task", str(task))
        d.setdefault("run_id", str(run_id))
        d.setdefault("model_id", str(spec.model_id))
        d.setdefault("model_version", str(spec.model_version))
        d.setdefault("is_forward_filled", False)
        d.setdefault("needs_recompute", False)
        if "pred_p50" not in d:
            yhat_cols = [str(col) for col in d if str(col).endswith("_yhat")]
            p50_cols = [str(col) for col in d if str(col).endswith("_p50")]
            source_col = yhat_cols[0] if yhat_cols else (p50_cols[0] if p50_cols else None)
            if source_col is not None:
                d["pred_p50"] = float(d[source_col])
        if "pred_p10" not in d:
            p10_cols = [str(col) for col in d if str(col).endswith("_p10")]
            lo_cols = [str(col) for col in d if str(col).endswith("_lo")]
            source_col = p10_cols[0] if p10_cols else (lo_cols[0] if lo_cols else None)
            if source_col is not None:
                d["pred_p10"] = float(d[source_col])
        if "pred_p90" not in d:
            p90_cols = [str(col) for col in d if str(col).endswith("_p90")]
            hi_cols = [str(col) for col in d if str(col).endswith("_hi")]
            source_col = p90_cols[0] if p90_cols else (hi_cols[0] if hi_cols else None)
            if source_col is not None:
                d["pred_p90"] = float(d[source_col])
        if nonnegative_task:
            for col in list(d):
                col_name = str(col)
                if col_name in {"pred_p10", "pred_p50", "pred_p90"} or col_name.endswith(("_yhat", "_lo", "_hi", "_p10", "_p50", "_p90", "_sigma", "_var")):
                    try:
                        d[col] = max(0.0, float(d[col]))
                    except Exception:
                        pass
        out.append(d)
    return out


def _public_prediction_row(row: Dict[str, Any]) -> Dict[str, Any]:
    return {str(k): v for k, v in row.items() if not str(k).startswith("_stats_")}


def _eval_rows_from_predictions(rows: Sequence[Dict[str, Any]], *, task: str) -> List[Dict[str, Any]]:
    target_col = NUMERIC_TASK_TO_TARGET_COLUMN.get(str(task), str(task))
    out: List[Dict[str, Any]] = []
    for row in rows:
        if "_stats_actual" not in row or "pred_p50" not in row:
            continue
        actual = pd.to_numeric(pd.Series([row.get("_stats_actual")]), errors="coerce").iloc[0]
        if pd.isna(actual):
            continue
        out.append(prediction_eval_row(prediction_row=row, actual_value=float(actual), target_col=target_col))
    return out


def _project_root() -> Path:
    return Path(os.getenv("PROJECT_ROOT", "") or Path.cwd()).resolve()


def _artifact_model_key(model_id: str) -> str:
    raw = str(model_id).strip()
    return raw[len("stats_") :] if raw.startswith("stats_") else raw


def discover_tested_production_artifact_scope(
    spec: StatsNumericModuleSpec,
    project_root: Optional[Path] = None,
) -> Optional[TestedProductionArtifactScope]:
    return discover_numeric_tested_production_artifact_scope(
        project_root=(project_root or _project_root()).resolve(),
        diagnostics_root_name="stats_numeric_family_test_orchestrator",
        artifact_model_key=_artifact_model_key(spec.model_id),
        error_prefix=f"[{spec.branch}]",
        discover_tested_production_artifact_payload_fn=discover_tested_production_artifact_payload,
    )


def discover_existing_production_scope(
    spec: StatsNumericModuleSpec,
    *,
    manifest_path: Optional[Path] = None,
    canonical_io_config: Any = None,
) -> Optional[ExistingProductionScope]:
    del manifest_path
    combo_specs = (
        discover_existing_combo_specs_from_canonical_physical_output(io_config=canonical_io_config)
        if canonical_io_config is not None
        else discover_existing_combo_specs_from_partitioned_output(spec.forecast_root)
    )
    if not combo_specs:
        return None
    return ExistingProductionScope(source_root=spec.forecast_root.resolve(), combo_specs=combo_specs)


def _snapshot_manifest(
    *,
    spec: StatsNumericModuleSpec,
    run_id: str,
    intervals: Sequence[int],
    horizons: Sequence[int],
    tasks: Sequence[str],
    combos: Sequence[Tuple[int, int, str]],
    combo_plans: Sequence[Dict[str, Any]],
    args: argparse.Namespace,
    resolved_model_threads: int,
    dispatch_slots: int,
    state_root: Path,
    stage_root: Path,
    manifest_parts: Sequence[Dict[str, Any]],
    unit_entries: Sequence[Dict[str, Any]],
    skipped_units: Dict[str, Dict[str, Any]],
    tested_scope: Optional[TestedProductionArtifactScope],
    production_scope: Optional[ExistingProductionScope],
    writer_stats: Optional[Dict[str, Any]] = None,
    job_shard_count: int = 0,
    finished_at: Optional[str] = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "run_id": run_id,
        "family": spec.family,
        "domain": spec.domain,
        "branch": spec.branch,
        "family_tag": spec.family_tag,
        "family_root_name": spec.family_root_name,
        "family_root_env": spec.family_root_env,
        "forecast_output_root": str(spec.forecast_root),
        "eval_output_root": str(spec.forecast_root / "eval"),
        "source_roots": dict(getattr(args, "resolved_source_roots", {}) or {}),
        "intervals": [int(x) for x in intervals],
        "horizon_minutes": [int(x) for x in horizons],
        "tasks": [str(x) for x in tasks],
        "combos": [{"interval": int(i), "horizon_minutes": int(h), "task": str(t)} for i, h, t in combos],
        "combo_plans": list(combo_plans),
        "combo_count": int(len(combos)),
        "workers": int(args.workers),
        "dispatch_slots": int(dispatch_slots),
        "model_threads": int(resolved_model_threads),
        "model_threads_source": str(getattr(args, "model_threads_source", "")),
        "execution_backend": str(getattr(args, "execution_backend", "process")),
        "worker_mode": f"horizon_group_{str(getattr(args, 'execution_backend', 'process'))}_shards",
        "writer_mode": "queued_partitioned_writer",
        "staging_root": str(stage_root),
        "state_root": str(state_root),
        "diagnostics_jsonl": str(diagnostics_file(state_root, spec.branch)),
        "job_shard_count": int(job_shard_count),
        "writer_stats": dict(writer_stats or {}),
        "backfill_days": int(args.backfill_days),
        "fit_days": int(args.fit_days) if args.fit_days is not None else None,
        "predict_latest_only": bool(args.predict_latest_only),
        "fill_to_edge": bool(args.fill_to_edge),
        "mode": str(args.mode),
        "force": bool(args.force),
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
        "min_train_bars": int(spec.min_train_bars),
        "train_windows_bars": [int(x) for x in spec.train_windows_bars],
        "parts": list(manifest_parts),
        "unit_entries": list(unit_entries),
        "skipped_units": len(skipped_units),
        **spec.manifest_extras_fn(args),
    }
    if finished_at is not None:
        payload["finished_at"] = str(finished_at)
    return payload


def _extend_rows_to_source_edge(rows: Sequence[Dict[str, Any]], *, interval_minutes: int, edge_ts: Optional[int]) -> List[Dict[str, Any]]:
    normalized = [dict(row) for row in rows]
    if edge_ts is None or not normalized:
        return normalized
    completed = [row for row in normalized if not bool(row.get("needs_recompute", False)) and not bool(row.get("is_forward_filled", False))]
    if not completed:
        return normalized
    last_row = max(completed, key=lambda row: int(row["ts"]))
    filled = forward_fill_rows_to_edge(last_row=last_row, interval_minutes=int(interval_minutes), edge_ts=int(edge_ts))
    return [*normalized, *filled]


def _run_stats_asset_shard_core(
    payload: Dict[str, Any],
    *,
    spec: StatsNumericModuleSpec,
    process_unit_fn: StatsProcessUnitFn,
    diagnostics_path: Optional[Path],
) -> List[Tuple[str, Dict[str, Any]]]:
    shard_started = time.perf_counter()
    updates: List[Tuple[str, Dict[str, Any]]] = []
    interval = int(payload["interval"])
    hm = int(payload["horizon_minutes"])
    task = str(payload["task"])
    hb = int(payload["horizon_bars"])
    run_id = str(payload["run_id"])
    spill_threshold = max(1, int(os.getenv("STATS_NUMERIC_ROW_SPILL_THRESHOLD", "5000")))
    done_count = 0
    skipped_count = 0
    forecast_row_count = 0
    eval_row_count = 0
    staged_file_count = 0
    slowest_unit: Dict[str, Any] = {}
    unit_summaries: List[Dict[str, Any]] = []
    top_units_per_shard = max(1, int(os.getenv("STATS_NUMERIC_DIAG_TOP_UNITS_PER_SHARD", "5")))
    extra_kwargs = dict(payload.get("extra_kwargs") or {})
    for asset in list(payload.get("assets") or []):
        unit_started = time.perf_counter()
        ukey = make_stats_unit_key(spec.family, spec.domain, task, hm, str(asset), interval)
        res = process_unit_fn(
            asset=str(asset),
            interval=int(interval),
            task=str(task),
            horizon_minutes=int(hm),
            horizon_bars=int(hb),
            backfill_days=int(payload["backfill_days"]),
            predict_latest_only=bool(payload.get("predict_latest_only", False)),
            **extra_kwargs,
        )
        status = str(res.get("unit_status"))
        edge_ts = res.get("edge_ts")
        unit_elapsed = float(time.perf_counter() - unit_started)
        if not slowest_unit or unit_elapsed > float(slowest_unit.get("elapsed_s", 0.0) or 0.0):
            slowest_unit = {"asset": str(asset), "elapsed_s": round(unit_elapsed, 3), "status": status}
        if status == "done":
            done_count += 1
            rows = _normalize_prediction_rows(
                res.get("rows", []) or [],
                spec=spec,
                interval=int(interval),
                horizon_minutes=int(hm),
                task=str(task),
                run_id=run_id,
            )
            if bool(payload.get("fill_to_edge", False)):
                rows = _extend_rows_to_source_edge(rows, interval_minutes=int(interval), edge_ts=(int(edge_ts) if edge_ts is not None else None))
            eval_rows = _eval_rows_from_predictions(rows, task=str(task))
            public_rows = [_public_prediction_row(row) for row in rows]
            forecast_row_count += int(len(public_rows))
            eval_row_count += int(len(eval_rows))
            fit_meta = res.get("fit_meta", {}) if isinstance(res.get("fit_meta"), dict) else {}
            unit_summaries.append(
                {
                    "asset": str(asset),
                    "status": "done",
                    "elapsed_s": round(float(unit_elapsed), 3),
                    "forecast_rows": int(len(public_rows)),
                    "eval_rows": int(len(eval_rows)),
                    "convergence_warning_count": int(fit_meta.get("convergence_warning_count", 0) or 0),
                    "nonconverged_fit_count": int(fit_meta.get("nonconverged_fit_count", 0) or 0),
                    "fit_retry_count": int(fit_meta.get("fit_retry_count", 0) or 0),
                    "retry_resolved_count": int(fit_meta.get("fit_retry_resolved_count", fit_meta.get("retry_resolved_count", 0)) or 0),
                    "fit_count": int(fit_meta.get("fit_count", fit_meta.get("fit_success_count", fit_meta.get("forecast_fit_count", 0))) or 0),
                    "origin_count": int(fit_meta.get("origin_count", fit_meta.get("forecast_origin_count", 0)) or 0),
                    "fit_elapsed_s_total": round(float(fit_meta.get("fit_elapsed_s_total", 0.0) or 0.0), 3),
                    "seconds_per_origin": (
                        round(float(fit_meta.get("seconds_per_origin")), 6)
                        if fit_meta.get("seconds_per_origin") is not None
                        else None
                    ),
                    "spec_search_elapsed_s": round(float(fit_meta.get("spec_search_elapsed_s", fit_meta.get("spec_search_elapsed_s_total", 0.0)) or 0.0), 3),
                    "spec_candidate_count": int(fit_meta.get("spec_candidate_count", fit_meta.get("spec_search_candidate_count", 0)) or 0),
                    "spec_failed_candidate_count": int(fit_meta.get("spec_failed_candidate_count", fit_meta.get("spec_search_failed_candidate_count", 0)) or 0),
                    "seasonality_used": bool(fit_meta.get("seasonality_used", fit_meta.get("seasonal_enabled", False))),
                    "seasonality_source": fit_meta.get("seasonality_source"),
                    "seasonal_period_bars": fit_meta.get("seasonal_period_bars"),
                    "order": fit_meta.get("order"),
                    "seasonal_order": fit_meta.get("seasonal_order"),
                    "selected_window_bars": fit_meta.get("selected_window_bars"),
                }
            )
            row_ts = [int(row["ts"]) for row in public_rows if row.get("ts") is not None]
            actual_tail = max(row_ts) if row_ts else None
            work_span = (payload.get("work_spans") or {}).get(str(asset)) or {}
            target_tail = work_span.get("target_tail_ts") or actual_tail or edge_ts
            upd: Dict[str, Any] = {
                "status": "done",
                "asset": str(asset),
                "interval_min": int(interval),
                "horizon_min": int(hm),
                "task": str(task),
                "fit_meta": fit_meta,
                "metadata": {
                    "row_count": int(len(public_rows)),
                    "eval_row_count": int(len(eval_rows)),
                    "elapsed_s": round(float(unit_elapsed), 3),
                    "edge_ts": int(edge_ts) if edge_ts is not None else None,
                    "actual_tail_ts": int(actual_tail) if actual_tail is not None else None,
                    "target_tail_ts": int(target_tail) if target_tail is not None else None,
                    "write_tail_ts": int(actual_tail) if actual_tail is not None else None,
                },
            }
            stage_root = Path(payload["stage_root"])
            if len(public_rows) >= spill_threshold:
                staged_file_count += 1
                upd["staged_rows_paths"] = [
                    _spill_rows_chunk(
                        rows=public_rows,
                        staging_root=stage_root,
                        module_tag=spec.family_tag,
                        interval=int(interval),
                        horizon_minutes=int(hm),
                        task=str(task),
                        asset=str(asset),
                    )
                ]
            else:
                upd["rows"] = public_rows
            if len(eval_rows) >= spill_threshold:
                staged_file_count += 1
                upd["eval_staged_rows_paths"] = [
                    _spill_rows_chunk(
                        rows=eval_rows,
                        staging_root=stage_root,
                        module_tag=f"{spec.family_tag}_eval",
                        interval=int(interval),
                        horizon_minutes=int(hm),
                        task=str(task),
                        asset=str(asset),
                    )
                ]
            else:
                upd["eval_rows"] = eval_rows
            updates.append((ukey, upd))
        elif status == "skipped":
            skipped_count += 1
            unit_summaries.append(
                {
                    "asset": str(res.get("asset") or asset),
                    "status": "skipped",
                    "elapsed_s": round(float(unit_elapsed), 3),
                    "reason": str(res.get("reason", "skipped")),
                    "forecast_rows": 0,
                    "eval_rows": 0,
                    "convergence_warning_count": 0,
                    "nonconverged_fit_count": 0,
                    "fit_retry_count": 0,
                    "retry_resolved_count": 0,
                    "fit_count": 0,
                    "origin_count": 0,
                    "fit_elapsed_s_total": 0.0,
                    "seconds_per_origin": None,
                    "spec_search_elapsed_s": 0.0,
                    "spec_candidate_count": 0,
                    "spec_failed_candidate_count": 0,
                    "seasonality_used": False,
                    "seasonality_source": None,
                    "seasonal_period_bars": None,
                    "order": None,
                    "seasonal_order": None,
                    "selected_window_bars": None,
                }
            )
            updates.append(
                (
                    ukey,
                    {
                        "status": "skipped",
                        "reason": str(res.get("reason", "skipped")),
                        "edge_ts": int(edge_ts) if edge_ts is not None else None,
                        "asset": str(res.get("asset") or asset),
                        "interval_min": int(interval),
                        "horizon_min": int(hm),
                        "task": str(task),
                        "metadata": {"elapsed_s": round(float(unit_elapsed), 3)},
                    },
                )
            )
        else:
            raise RuntimeError(f"[{spec.branch}] unit failed ukey={ukey} status={status or 'unknown'}")
    if diagnostics_path is not None:
        unit_elapsed_values = [float(row.get("elapsed_s", 0.0) or 0.0) for row in unit_summaries]
        unit_elapsed_summary = {
            "count": int(len(unit_elapsed_values)),
            "min_s": round(float(min(unit_elapsed_values)), 3) if unit_elapsed_values else None,
            "max_s": round(float(max(unit_elapsed_values)), 3) if unit_elapsed_values else None,
            "mean_s": round(float(sum(unit_elapsed_values) / len(unit_elapsed_values)), 3) if unit_elapsed_values else None,
            "forecast_rows_per_s": round(float(forecast_row_count) / float(sum(unit_elapsed_values)), 6) if sum(unit_elapsed_values) > 0.0 else None,
        }
        _append_diagnostic_event(
            diagnostics_path,
            "shard_finished",
            {
                "run_id": run_id,
                "branch": spec.branch,
                "interval": int(interval),
                "horizon_minutes": int(hm),
                "task": str(task),
                "shard_index": int(payload.get("_shard_index", payload.get("shard_index", 0)) or 0),
                "shard_count": int(payload.get("_shard_count", payload.get("shard_count", 0)) or 0),
                "assets": int(len(payload.get("assets") or [])),
                "done_units": int(done_count),
                "skipped_units": int(skipped_count),
                "forecast_rows": int(forecast_row_count),
                "eval_rows": int(eval_row_count),
                "staged_files": int(staged_file_count),
                "elapsed_s": round(float(time.perf_counter() - shard_started), 3),
                "slowest_unit": slowest_unit,
                "slowest_units": sorted(unit_summaries, key=lambda row: float(row.get("elapsed_s", 0.0) or 0.0), reverse=True)[:top_units_per_shard],
                "unit_elapsed_summary": unit_elapsed_summary,
                "resource": resource_snapshot(),
            },
        )
    return updates


def _run_stats_asset_shard_process(payload: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
    model_threads = max(1, int(payload.get("model_threads", 1) or 1))
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "STATS_NUMERIC_MODEL_THREADS"):
        os.environ[name] = str(model_threads)
    spec = _load_registered_module_spec(str(payload["branch"]))
    return _run_stats_asset_shard_core(
        payload,
        spec=spec,
        process_unit_fn=spec.process_unit_fn,
        diagnostics_path=(Path(str(payload["diagnostics_path"])) if str(payload.get("diagnostics_path", "")).strip() else None),
    )


def _run_stats_horizon_group_shard_core(
    payload: Dict[str, Any],
    *,
    spec: StatsNumericModuleSpec,
    process_unit_fn: StatsProcessUnitFn,
    diagnostics_path: Optional[Path],
) -> List[Tuple[Tuple[int, int, str], str, Dict[str, Any]]]:
    grouped_updates: List[Tuple[Tuple[int, int, str], str, Dict[str, Any]]] = []
    for task_payload in list(payload.get("task_payloads") or []):
        combo_key = (
            int(task_payload["interval"]),
            int(task_payload["horizon_minutes"]),
            str(task_payload["task"]),
        )
        for ukey, upd in _run_stats_asset_shard_core(
            dict(task_payload),
            spec=spec,
            process_unit_fn=process_unit_fn,
            diagnostics_path=diagnostics_path,
        ):
            grouped_updates.append((combo_key, ukey, upd))
    return grouped_updates


def _run_stats_horizon_group_shard_process(payload: Dict[str, Any]) -> List[Tuple[Tuple[int, int, str], str, Dict[str, Any]]]:
    model_threads = max(1, int(payload.get("model_threads", 1) or 1))
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "STATS_NUMERIC_MODEL_THREADS"):
        os.environ[name] = str(model_threads)
    spec = _load_registered_module_spec(str(payload["branch"]))
    return _run_stats_horizon_group_shard_core(
        payload,
        spec=spec,
        process_unit_fn=spec.process_unit_fn,
        diagnostics_path=(Path(str(payload["diagnostics_path"])) if str(payload.get("diagnostics_path", "")).strip() else None),
    )


def run_stats_numeric_module(spec: StatsNumericModuleSpec) -> None:
    spec = materialize_runtime_spec(spec)
    parser = argparse.ArgumentParser(description=f"{spec.family} frequentist numeric forecaster.")
    _add_common_args(parser, spec)
    args = parser.parse_args()
    source_roots = _apply_explicit_source_roots(args, spec)
    resolved_model_threads = _configure_runtime(args)

    missing_dependency = spec.dependency_check_fn()
    if missing_dependency:
        raise SystemExit(missing_dependency)

    supported = list(CAPABILITY_MATRIX[spec.branch]["numerics"])
    stage_root = _staging_root(spec.forecast_root, spec.family_tag)
    canonical_io_config = canonical_physical_io_config(
        naming=canonical_physical_naming(
            module_slug=spec.model_id,
            prediction_prefix=spec.model_id,
            log_prefix=f"[{spec.branch}]",
        ),
        parquet_root=spec.forecast_root,
        staging_root=stage_root,
        state_root=spec.state_root,
        log_fn=spec.log_fn,
    )
    spec.log_fn(
        f"[{spec.branch}] source_roots ohlcvt={source_roots.get('ohlcvt_root', '')} "
        f"scalar={source_roots.get('scalar_feature_root', '')} "
        f"edge={source_roots.get('edge_discovery_root', '')} "
        f"target={source_roots.get('target_label_root', '')}"
    )
    requested_combos = parse_combo_list(args.combo_list) if str(args.combo_list).strip() else []
    using_default_combo_selection = (
        not requested_combos
        and str(args.intervals) == ",".join(str(x) for x in spec.default_intervals)
        and str(args.horizons_minutes) == ",".join(str(x) for x in spec.default_horizons)
        and str(args.tasks) == ",".join(_default_tasks(spec))
    )
    production_scope = discover_existing_production_scope(spec, canonical_io_config=canonical_io_config) if using_default_combo_selection else None
    tested_scope = None
    tested_scope_error = None
    stream_contract = None
    if using_default_combo_selection:
        try:
            tested_scope = discover_tested_production_artifact_scope(spec)
        except BaseException as exc:
            tested_scope_error = exc
        stream_contract = require_production_stream_scope_contract(
            resolve_production_stream_scope_contract(
                family="stats",
                model=spec.branch,
                existing_scope=production_scope,
                tested_scope=tested_scope,
                tested_scope_error=tested_scope_error,
                production_paths_inspected=(spec.forecast_root, spec.state_root),
            )
        )
    if stream_contract is not None and stream_contract.mode == "locked_to_existing_production":
        spec.log_fn(
            f"[{spec.branch}] production-defaults root={production_scope.source_root if production_scope is not None else spec.forecast_root} "
            f"combos={len(stream_contract.combo_specs)} asset_scope=existing_production_scope "
            f"scope_mode={stream_contract.mode} warnings={';'.join(stream_contract.warnings)}"
        )
    elif stream_contract is not None and stream_contract.mode == "bootstrap_from_test" and tested_scope is not None:
        spec.log_fn(
            f"[{spec.branch}] tested-defaults handoff={tested_scope.handoff_path} "
            f"feature_profile_json={tested_scope.feature_profile_json} cohort_assets={len(tested_scope.cohort_assets)} "
            f"combo_source={'stage3' if tested_scope.stage3_combo_specs else 'stage2'} "
            f"combos={len(tested_scope.stage3_combo_specs or tested_scope.combo_specs)} "
            f"stage3_combo_results={tested_scope.stage3_combo_results_path} asset_scope=full_production_universe "
            f"scope_mode={stream_contract.mode}"
        )
    if requested_combos:
        combos = [(int(iv), int(hm), str(task)) for iv, hm, task in requested_combos if str(task) in supported]
        intervals = sorted({int(iv) for iv, _, _ in combos})
        horizons = sorted({int(hm) for _, hm, _ in combos})
        tasks = sorted({str(task) for _, _, task in combos})
    elif stream_contract is not None:
        scoped_combos = stream_contract.combo_specs
        combos = [(int(iv), int(hm), str(task)) for iv, hm, task in scoped_combos if str(task) in supported]
        intervals = sorted({int(iv) for iv, _, _ in combos})
        horizons = sorted({int(hm) for _, hm, _ in combos})
        tasks = sorted({str(task) for _, _, task in combos})
    else:
        intervals = parse_int_csv(args.intervals, spec.default_intervals)
        horizons = parse_int_csv(args.horizons_minutes, spec.default_horizons)
        tasks = [t for t in parse_str_csv(args.tasks, supported) if t in supported]
        if not tasks:
            tasks = supported
        combos = [(int(iv), int(hm), str(task)) for iv in intervals for hm in horizons for task in tasks]
    assets = resolve_assets(intervals=intervals, assets_arg=args.assets)
    if not assets:
        spec.log_fn(f"[{spec.branch}] no assets discovered")
        return

    lock_path: Optional[Path] = None
    try:
        lock_path = _acquire_single_run_lock(spec.state_root, spec.family_tag)
    except RuntimeError as exc:
        spec.log_fn(f"[{spec.branch}][skip] {exc}")
        return

    try:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        skipped_units: Dict[str, Dict[str, Any]] = {}
        manifest_parts: List[Dict[str, Any]] = []
        unit_entries: List[Dict[str, Any]] = []
        extra_kwargs = spec.extra_process_kwargs_fn(args)
        diagnostics_path = diagnostics_file(spec.state_root, spec.branch)
        reset_diagnostics_file(diagnostics_path)
        _append_diagnostic_event(
            diagnostics_path,
            "run_start",
            {
                "run_id": run_id,
                "branch": spec.branch,
                "assets": len(assets),
                "requested_combos": len(combos),
                "workers": int(args.workers),
                "model_threads": int(resolved_model_threads),
                "execution_backend": str(args.execution_backend),
                "mode": str(args.mode),
                "backfill_days": int(args.backfill_days),
                "predict_latest_only": bool(args.predict_latest_only),
                "force": bool(args.force),
                "source_roots": dict(source_roots),
                "resource": resource_snapshot(),
            },
        )
        combo_plans: List[Dict[str, Any]] = []
        combo_jobs: List[Dict[str, Any]] = []
        combo_order: List[Tuple[int, int, str]] = []
        planned_assets_by_combo: Dict[Tuple[int, int, str], List[str]] = {}
        planning_workers = max(1, min(int(args.workers), max(1, len(assets))))
        tail_cache = TailPlanningCache()

        for interval, hm, task in combos:
            plan_started = time.perf_counter()
            combo_key = (int(interval), int(hm), str(task))
            combo_order.append(combo_key)
            hb = horizon_bars(int(hm), int(interval))

            def _plan_one(asset_name: str) -> Optional[Any]:
                try:
                    return plan_asset_work_span(
                        asset=str(asset_name),
                        interval=int(interval),
                        horizon_minutes=int(hm),
                        task=str(task),
                        mode=("backfill" if str(args.mode) == "backfill" else "incremental"),
                        backfill_days=int(args.backfill_days),
                        force=bool(args.force),
                        fit_days=int(args.fit_days or DEFAULT_BACKFILL_DAYS),
                        forecast_root=spec.forecast_root,
                        discover_edge_and_min_fn=lambda **edge_kwargs: discover_edge_and_min(
                            root=Path(str(source_roots.get("edge_discovery_root") or source_roots.get("ohlcvt_root") or args.parquet_root)),
                            **edge_kwargs,
                        ),
                        forecast_output_tail_ts_fn=lambda **tail_kwargs: canonical_physical_output_tail_ts(
                            io_config=canonical_io_config,
                            interval_minutes=int(tail_kwargs["interval_minutes"]),
                            task=str(tail_kwargs["task"]),
                            horizon_minutes=int(tail_kwargs["horizon_minutes"]),
                            asset=str(tail_kwargs["asset"]),
                            include_recompute=bool(tail_kwargs.get("include_recompute", False)),
                        ),
                        decide_range_from_disk_edges_fn=decide_range_from_disk_edges,
                        fit_window_start_fn=fit_window_start,
                        tail_cache=tail_cache,
                        tail_cache_namespace=f"canonical_physical:{canonical_io_config.naming.forecast_table_tag}:forecast",
                    )
                except Exception:
                    return None

            if planning_workers <= 1 or len(assets) <= 1:
                planned_spans_by_asset = {str(asset_name): _plan_one(str(asset_name)) for asset_name in assets}
            else:
                try:
                    with ThreadPoolExecutor(max_workers=int(planning_workers)) as planning_pool:
                        planned_results = list(planning_pool.map(_plan_one, [str(asset_name) for asset_name in assets]))
                    planned_spans_by_asset = {str(asset_name): planned_results[idx] for idx, asset_name in enumerate(assets)}
                except Exception as exc:
                    spec.log_fn(f"[{spec.branch}][runtime-fallback] planning pool unavailable; forcing serial planning: {exc}")
                    planned_spans_by_asset = {str(asset_name): _plan_one(str(asset_name)) for asset_name in assets}

            if bool(args.force) or bool(args.predict_latest_only):
                planned_assets = [str(asset_name) for asset_name in assets]
            else:
                planned_assets = [str(asset_name) for asset_name in assets if planned_spans_by_asset.get(str(asset_name)) is not None]
            planned_assets_by_combo[combo_key] = list(planned_assets)
            combo_plan = {
                "interval": int(interval),
                "horizon_minutes": int(hm),
                "task": str(task),
                "assets_total": int(len(assets)),
                "planned_assets_with_work": int(len(planned_assets)),
                "skipped_at_edge_assets": int(max(0, len(assets) - len(planned_assets))),
                "horizon_bars": int(hb),
                "planning_elapsed_s": round(float(time.perf_counter() - plan_started), 3),
            }
            combo_plans.append(combo_plan)
            _append_diagnostic_event(
                diagnostics_path,
                "combo_planned",
                {
                    "run_id": run_id,
                    "branch": spec.branch,
                    **combo_plan,
                    "resource": resource_snapshot(),
                },
            )
            if not planned_assets:
                spec.log_fn(f"[{spec.branch}][group-noop] k={int(interval)} h={int(hm)}m task={task} reason=already_at_edge assets_total={len(assets)}")
                continue
            combo_jobs.extend(
                build_asset_shard_jobs(
                    combo_key=combo_key,
                    planned_assets=planned_assets,
                    worker_count=int(args.workers),
                    base_payload={
                        "interval": int(interval),
                        "horizon_minutes": int(hm),
                        "task": str(task),
                        "horizon_bars": int(hb),
                        "run_id": str(run_id),
                        "branch": str(spec.branch),
                        "backfill_days": int(args.backfill_days),
                        "predict_latest_only": bool(args.predict_latest_only),
                        "fill_to_edge": bool(args.fill_to_edge),
                        "extra_kwargs": dict(extra_kwargs),
                        "stage_root": str(stage_root),
                        "diagnostics_path": str(diagnostics_path),
                        "model_threads": int(resolved_model_threads),
                    },
                    work_spans_by_asset={
                        str(asset_name): (
                            {
                                "edge_ts": int(span.edge_ts),
                                "target_tail_ts": int(span.target_tail_ts),
                            }
                            if (span := planned_spans_by_asset.get(str(asset_name))) is not None
                            else None
                        )
                        for asset_name in planned_assets
                    },
                    include_public_shard_fields=True,
                )
            )
        group_jobs = build_horizon_group_shard_jobs(combo_jobs=combo_jobs, worker_count=int(args.workers))
        for group_job in group_jobs:
            group_job["branch"] = str(spec.branch)
            group_job["diagnostics_path"] = str(diagnostics_path)
            group_job["model_threads"] = int(resolved_model_threads)
        dispatch_slots = resolve_dispatch_slots(int(args.workers), len(group_jobs)) if group_jobs else 0

        def _queue_update_rows(combo_key: Tuple[int, int, str], updates: Sequence[Tuple[str, Dict[str, Any]]]) -> None:
            forecast_rows = [row for _ukey, upd in updates for row in (upd.get("rows", []) if isinstance(upd.get("rows"), list) else [])]
            forecast_files = [
                str(path)
                for _ukey, upd in updates
                for path in (upd.get("staged_rows_paths", []) if isinstance(upd.get("staged_rows_paths"), list) else [])
                if str(path).strip()
            ]
            eval_rows = [row for _ukey, upd in updates for row in (upd.get("eval_rows", []) if isinstance(upd.get("eval_rows"), list) else [])]
            eval_files = [
                str(path)
                for _ukey, upd in updates
                for path in (upd.get("eval_staged_rows_paths", []) if isinstance(upd.get("eval_staged_rows_paths"), list) else [])
                if str(path).strip()
            ]
            if forecast_rows:
                _append_diagnostic_event(
                    diagnostics_path,
                    "writer_enqueue",
                    {
                        "run_id": run_id,
                        "branch": spec.branch,
                        "store": "forecast",
                        "interval": int(combo_key[0]),
                        "horizon_minutes": int(combo_key[1]),
                        "task": str(combo_key[2]),
                        "rows": int(len(forecast_rows)),
                        "files": int(len(forecast_files)),
                        "queue_unfinished": int(getattr(writer_queue, "unfinished_tasks", 0)),
                    },
                )
                writer_queue.put({"kind": "write_batch", "combo_key": combo_key, "run_id": run_id, "module_tag": spec.family_tag, "rows": forecast_rows})
            if forecast_files:
                _append_diagnostic_event(
                    diagnostics_path,
                    "writer_enqueue",
                    {
                        "run_id": run_id,
                        "branch": spec.branch,
                        "store": "forecast",
                        "interval": int(combo_key[0]),
                        "horizon_minutes": int(combo_key[1]),
                        "task": str(combo_key[2]),
                        "rows": 0,
                        "files": int(len(forecast_files)),
                        "queue_unfinished": int(getattr(writer_queue, "unfinished_tasks", 0)),
                    },
                )
                writer_queue.put({"kind": "write_batch_files", "combo_key": combo_key, "run_id": run_id, "module_tag": spec.family_tag, "staged_rows_paths": forecast_files})
            if eval_rows:
                _append_diagnostic_event(
                    diagnostics_path,
                    "writer_enqueue",
                    {
                        "run_id": run_id,
                        "branch": spec.branch,
                        "store": "eval",
                        "interval": int(combo_key[0]),
                        "horizon_minutes": int(combo_key[1]),
                        "task": str(combo_key[2]),
                        "rows": int(len(eval_rows)),
                        "files": int(len(eval_files)),
                        "queue_unfinished": int(getattr(writer_queue, "unfinished_tasks", 0)),
                    },
                )
                writer_queue.put({"kind": "write_batch", "store": "eval", "combo_key": combo_key, "run_id": run_id, "module_tag": f"{spec.family_tag}_eval", "rows": eval_rows})
            if eval_files:
                _append_diagnostic_event(
                    diagnostics_path,
                    "writer_enqueue",
                    {
                        "run_id": run_id,
                        "branch": spec.branch,
                        "store": "eval",
                        "interval": int(combo_key[0]),
                        "horizon_minutes": int(combo_key[1]),
                        "task": str(combo_key[2]),
                        "rows": 0,
                        "files": int(len(eval_files)),
                        "queue_unfinished": int(getattr(writer_queue, "unfinished_tasks", 0)),
                    },
                )
                writer_queue.put(
                    {
                        "kind": "write_batch_files",
                        "store": "eval",
                        "combo_key": combo_key,
                        "run_id": run_id,
                        "module_tag": f"{spec.family_tag}_eval",
                        "staged_rows_paths": eval_files,
                    }
                )

        def _record_unit_update(ukey: str, upd: Dict[str, Any]) -> None:
            if str(upd.get("status")) == "done":
                metadata = dict(upd.get("metadata") or {})
                unit_entries.append(
                    {
                        "ukey": ukey,
                        "status": "done",
                        "edge_ts": int(metadata["edge_ts"]) if metadata.get("edge_ts") is not None else None,
                        "elapsed_s": float(metadata.get("elapsed_s", 0.0) or 0.0),
                        "forecast_rows": int(metadata.get("row_count", 0) or 0),
                        "eval_rows": int(metadata.get("eval_row_count", 0) or 0),
                        "fit_meta": upd.get("fit_meta", {}),
                    }
                )
            elif str(upd.get("status")) == "skipped":
                reason = str(upd.get("reason", "skipped"))
                skipped_units[ukey] = {
                    "reason": reason,
                    "edge_ts": upd.get("edge_ts"),
                    "asset": upd.get("asset"),
                    "task": upd.get("task"),
                    "interval": int(upd.get("interval_min", 0) or 0),
                    "horizon_minutes": int(upd.get("horizon_min", 0) or 0),
                    "elapsed_s": float((upd.get("metadata") or {}).get("elapsed_s", 0.0) or 0.0),
                }
                unit_entries.append(
                    {
                        "ukey": ukey,
                        "status": "skipped",
                        "edge_ts": upd.get("edge_ts"),
                        "elapsed_s": float((upd.get("metadata") or {}).get("elapsed_s", 0.0) or 0.0),
                        "fit_meta": {"reason": reason},
                    }
                )

        def _run_stats_asset_shard(payload: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
            return _run_stats_asset_shard_core(
                payload,
                spec=spec,
                process_unit_fn=spec.process_unit_fn,
                diagnostics_path=diagnostics_path,
            )

        write_json_atomic(
            spec.manifest_file,
            _snapshot_manifest(
                spec=spec,
                run_id=run_id,
                intervals=intervals,
                horizons=horizons,
                tasks=tasks,
                combos=combos,
                combo_plans=combo_plans,
                args=args,
                resolved_model_threads=resolved_model_threads,
                dispatch_slots=dispatch_slots,
                state_root=spec.state_root,
                stage_root=stage_root,
                manifest_parts=manifest_parts,
                unit_entries=unit_entries,
                skipped_units=skipped_units,
                tested_scope=tested_scope,
                production_scope=production_scope,
                job_shard_count=len(group_jobs),
            ),
        )

        writer_queue, writer_state, writer_thread = start_partitioned_prediction_writer(
            module_tag=spec.family_tag,
            forecast_root=spec.forecast_root,
            writer_loop_fn=lambda **kwargs: _writer_loop(canonical_io_config=canonical_io_config, **kwargs),
        )
        if group_jobs:
            spec.log_fn(
                f"[{spec.branch}] production-plan combos={len(combo_order)} combo_jobs={len(combo_jobs)} "
                f"group_jobs={len(group_jobs)} worker_mode=horizon_group_{str(args.execution_backend)}_shards assets={len(assets)} workers={int(args.workers)} "
                f"dispatch_slots={int(dispatch_slots)} model_threads={int(resolved_model_threads)}"
            )
        run_group_shard_fn = _run_stats_horizon_group_shard_process if str(args.execution_backend) == "process" else (
            lambda payload: _run_stats_horizon_group_shard_core(
                payload,
                spec=spec,
                process_unit_fn=spec.process_unit_fn,
                diagnostics_path=diagnostics_path,
            )
        )
        updates_by_combo, _unit_results = execute_grouped_horizon_jobs(
            group_jobs=group_jobs,
            combo_order=combo_order,
            dispatch_slots=dispatch_slots,
            module_name="stats_numeric_runner",
            module_tag=spec.family_tag,
            family_label="Stats numeric",
            log_fn=lambda message: spec.log_fn(f"[{spec.branch}] {message}"),
            run_group_shard_fn=run_group_shard_fn,
            writer_queue=writer_queue,
            writer_state=writer_state,
            make_unit_result_fn=lambda _payload, _upd: None,
            process_pool_init_retries=1,
            process_pool_init_retry_seconds=0.0,
            pressure_guard_factory=lambda module_name, log_fn: DispatchPressureGuard(module_name=module_name, log_fn=log_fn),
            diagnostics_path=diagnostics_path,
            diagnostics_timestamp_fn=utc_now_iso,
        )
        for combo_updates in updates_by_combo.values():
            for ukey, upd in combo_updates:
                _record_unit_update(ukey, upd)
        write_json_atomic(
            spec.manifest_file,
            _snapshot_manifest(
                spec=spec,
                run_id=run_id,
                intervals=intervals,
                horizons=horizons,
                tasks=tasks,
                combos=combos,
                combo_plans=combo_plans,
                args=args,
                resolved_model_threads=resolved_model_threads,
                dispatch_slots=dispatch_slots,
                state_root=spec.state_root,
                stage_root=stage_root,
                manifest_parts=manifest_parts,
                unit_entries=unit_entries,
                skipped_units=skipped_units,
                tested_scope=tested_scope,
                production_scope=production_scope,
                writer_stats=writer_state.get("writer_stats", {}),
                job_shard_count=len(group_jobs),
            ),
        )
        drain_started = time.perf_counter()
        _append_diagnostic_event(
            diagnostics_path,
            "writer_drain_start",
            {
                "run_id": run_id,
                "branch": spec.branch,
                "queue_unfinished": int(getattr(writer_queue, "unfinished_tasks", 0)),
                "writer_stats": dict(writer_state.get("writer_stats", {})),
                "resource": resource_snapshot(),
            },
        )
        wait_for_writer_drain(writer_queue=writer_queue, writer_thread=writer_thread, writer_state=writer_state, family_label="Stats numeric")
        _append_diagnostic_event(
            diagnostics_path,
            "writer_drain_complete",
            {
                "run_id": run_id,
                "branch": spec.branch,
                "elapsed_s": round(float(time.perf_counter() - drain_started), 3),
                "writer_stats": dict(writer_state.get("writer_stats", {})),
                "resource": resource_snapshot(),
            },
        )
        writer_queue.put({"kind": "stop"})
        writer_thread.join()
        raise_writer_fatal(writer_state, "Stats numeric")

        parts_by_combo: Dict[Tuple[int, int, str], List[Dict[str, Any]]] = dict(writer_state.get("parts_by_combo", {}))
        eval_parts_by_combo: Dict[Tuple[int, int, str], List[Dict[str, Any]]] = dict(writer_state.get("parts_by_store", {}).get("eval", {}))
        for interval, hm, task in combo_order:
            combo_key = (int(interval), int(hm), str(task))
            validate_combo_completion(
                combo_key=combo_key,
                planned_assets=planned_assets_by_combo.get(combo_key, []),
                combo_updates=updates_by_combo.get(combo_key, []),
                module_tag=spec.family_tag,
            )
            forecast_parts = list(parts_by_combo.get(combo_key, []))
            eval_parts = [{**part, "store": "eval"} for part in list(eval_parts_by_combo.get(combo_key, []))]
            manifest_parts.extend(forecast_parts)
            manifest_parts.extend(eval_parts)
            spec.log_fn(
                f"[{spec.branch}] combo complete interval={int(interval)} horizon={int(hm)} task={task} "
                f"forecast_parts={len(forecast_parts)} eval_parts={len(eval_parts)}"
            )
            _append_diagnostic_event(
                diagnostics_path,
                "combo_complete",
                {
                    "run_id": run_id,
                    "branch": spec.branch,
                    "interval": int(interval),
                    "horizon_minutes": int(hm),
                    "task": str(task),
                    "planned_assets": int(len(planned_assets_by_combo.get(combo_key, []))),
                    "updates": int(len(updates_by_combo.get(combo_key, []))),
                    "forecast_parts": int(len(forecast_parts)),
                    "eval_parts": int(len(eval_parts)),
                    "resource": resource_snapshot(),
                },
            )

        write_json_atomic(spec.skipped_file, {"run_id": run_id, "generated_at": utc_now_iso(), "units": skipped_units})
        write_json_atomic(
            spec.manifest_file,
            _snapshot_manifest(
                spec=spec,
                run_id=run_id,
                intervals=intervals,
                horizons=horizons,
                tasks=tasks,
                combos=combos,
                combo_plans=combo_plans,
                args=args,
                resolved_model_threads=resolved_model_threads,
                dispatch_slots=dispatch_slots,
                state_root=spec.state_root,
                stage_root=stage_root,
                manifest_parts=manifest_parts,
                unit_entries=unit_entries,
                skipped_units=skipped_units,
                tested_scope=tested_scope,
                production_scope=production_scope,
                writer_stats=writer_state.get("writer_stats", {}),
                job_shard_count=len(group_jobs),
                finished_at=utc_now_iso(),
            ),
        )
        _append_diagnostic_event(
            diagnostics_path,
            "run_complete",
            {
                "run_id": run_id,
                "branch": spec.branch,
                "parts": int(len(manifest_parts)),
                "skipped_units": int(len(skipped_units)),
                "unit_entries": int(len(unit_entries)),
                "writer_stats": dict(writer_state.get("writer_stats", {})),
                "diagnostics_path": str(diagnostics_path),
                "resource": resource_snapshot(),
            },
        )
        spec.log_fn(f"[{spec.branch}] run complete parts={len(manifest_parts)} skipped_units={len(skipped_units)}")
    finally:
        if lock_path is not None:
            assert_write_allowed(lock_path, "Stats lock release", roots=_sandbox_roots())
            try:
                Path(lock_path).unlink()
            except FileNotFoundError:
                pass
