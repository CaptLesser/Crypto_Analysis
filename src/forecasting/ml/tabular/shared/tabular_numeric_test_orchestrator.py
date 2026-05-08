from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from src.forecasting.common.sandbox_paths import assert_write_allowed, resolve_sandbox_output_roots
from src.forecasting.common.path_config import PathConfigError, require_pipeline_io
from src.forecasting.ml.shared.test_orchestrator_common import (
    add_sandbox_output_args,
    assert_test_branch_sandbox_launch,
    collect_stage3_outputs as _shared_collect_stage3_outputs,
    command_flag,
    discover_latest_stage2_manifest,
    family_logs_dir,
    finalize_sandbox_output_args,
    latest_incomplete_run as _shared_latest_incomplete_run,
    load_json_dict,
    remove_tree,
    resolve_run_root as _shared_resolve_run_root,
    run_complete as _shared_run_complete,
    stage0_candidates_path as _shared_stage0_candidates_path,
    stage0_complete as _shared_stage0_complete,
    stage0_dir,
    stage0_log_path,
    stage0_profile_path as _shared_stage0_profile_path,
    stage1_complete,
    stage1_meta_path,
    stage1_selection_path,
    stage2_complete,
    stage2_survivor_json_from_manifest,
    stage3_complete as _shared_stage3_complete,
    test_branch_child_env,
    test_branch_stage_tmp_root,
    utc_now_iso,
    utc_now_stamp,
    write_json_atomic,
    write_test_branch_health,
    HEALTH_REPORT_FILE,
)

MODEL_ORDER = ["xgboost", "lightgbm", "catboost", "random_forest", "elasticnet"]
TABULAR_STAGE0_ENTRYPOINT = "src.forecasting.ml.tabular.shared.tabular_numeric_stage0_profile"


@dataclass(frozen=True)
class TabularNumericTestModuleSpec:
    model_key: str
    display_name: str
    stage1_module: str
    stage2_module: str
    stage3_module: str


MODEL_SPECS: Dict[str, TabularNumericTestModuleSpec] = {
    "xgboost": TabularNumericTestModuleSpec(
        model_key="xgboost",
        display_name="XGBoost",
        stage1_module="src.forecasting.ml.tabular.xgboost.xgboost_feature_experiment",
        stage2_module="src.forecasting.ml.tabular.xgboost.xgboost_numeric_scaling_test",
        stage3_module="src.forecasting.ml.tabular.xgboost.xgboost_numeric_optuna_tuning",
    ),
    "lightgbm": TabularNumericTestModuleSpec(
        model_key="lightgbm",
        display_name="LightGBM",
        stage1_module="src.forecasting.ml.tabular.lightgbm.lightgbm_feature_experiment",
        stage2_module="src.forecasting.ml.tabular.lightgbm.lightgbm_numeric_scaling_test",
        stage3_module="src.forecasting.ml.tabular.lightgbm.lightgbm_numeric_optuna_tuning",
    ),
    "catboost": TabularNumericTestModuleSpec(
        model_key="catboost",
        display_name="CatBoost",
        stage1_module="src.forecasting.ml.tabular.catboost.catboost_feature_experiment",
        stage2_module="src.forecasting.ml.tabular.catboost.catboost_numeric_scaling_test",
        stage3_module="src.forecasting.ml.tabular.catboost.catboost_numeric_optuna_tuning",
    ),
    "random_forest": TabularNumericTestModuleSpec(
        model_key="random_forest",
        display_name="Random Forest",
        stage1_module="src.forecasting.ml.tabular.random_forest.random_forest_feature_experiment",
        stage2_module="src.forecasting.ml.tabular.random_forest.random_forest_numeric_scaling_test",
        stage3_module="src.forecasting.ml.tabular.random_forest.random_forest_numeric_optuna_tuning",
    ),
    "elasticnet": TabularNumericTestModuleSpec(
        model_key="elasticnet",
        display_name="ElasticNet",
        stage1_module="src.forecasting.ml.tabular.elasticnet.elasticnet_feature_experiment",
        stage2_module="src.forecasting.ml.tabular.elasticnet.elasticnet_numeric_scaling_test",
        stage3_module="src.forecasting.ml.tabular.elasticnet.elasticnet_numeric_optuna_tuning",
    ),
}

RUN_STATE_FILE = "tabular_numeric_test_orchestrator_state.json"
RUN_SUMMARY_FILE = "tabular_numeric_stage3_artifacts.json"
DEFAULT_OUTPUT_DIR = Path("logs") / "diagnostics" / "tabular_numeric_family_test_orchestrator"
STAGE0_PROFILE_JSON = "tabular_numeric_stage0_profile.json"
STAGE0_CANDIDATES_CSV = "tabular_numeric_stage0_candidates.csv"
REQUIRED_STAGE3_FILES = (
    "combo_results.csv",
    "optuna_trials.json",
    "unit_metrics.csv",
    "summary.md",
    "representative_samples.csv",
)


@dataclass(frozen=True)
class ModelPaths:
    root: Path
    stage1_dir: Path
    stage2_root: Path
    stage3_dir: Path
    stage3_storage_path: Path
    logs_dir: Path
    stage1_log: Path
    stage2_log: Path
    stage3_log: Path
def progress_line(message: str) -> None:
    print(f"[{utc_now_iso()}] {message}", flush=True)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sequential Stage-1/2/3 test orchestrator for the tabular ML numerics family.")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[5])
    parser.add_argument("--profile", type=str, default="pipeline_test")
    parser.add_argument("--parquet-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    add_sandbox_output_args(parser)
    parser.add_argument("--run-id", type=str, default="")
    parser.add_argument("--resume-run", type=str, default="")
    parser.add_argument("--no-resume-latest", action="store_true")
    parser.add_argument("--intervals", type=str, default="")
    parser.add_argument("--tasks", type=str, default="")
    parser.add_argument("--horizon-minutes", type=str, default="")
    parser.add_argument("--combo-list", type=str, default="")
    parser.add_argument("--assets", type=str, default="")
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--retry-delay-seconds", type=float, default=2.0)
    parser.add_argument("--stage0-task", type=str, default="")
    parser.add_argument("--stage0-interval", type=int, default=None)
    parser.add_argument("--stage0-horizon-minutes", type=int, default=None)
    parser.add_argument("--stage0-workers", type=str, default="")
    parser.add_argument("--stage0-threads", type=str, default="")
    parser.add_argument("--stage0-assets", type=str, default="")
    parser.add_argument("--stage0-backfill-days", type=int, default=None)
    parser.add_argument("--stage0-sample-seconds", type=float, default=None)
    parser.add_argument("--stage0-ram-cap-pct", type=float, default=None)
    parser.add_argument("--stage0-prune-run-artifacts", action="store_true")

    parser.add_argument("--stage1-asset-count", type=int, default=None)
    parser.add_argument("--stage1-seed", type=int, default=None)
    parser.add_argument("--stage1-train-window-months", type=int, default=None)
    parser.add_argument("--stage1-max-rows", type=int, default=None)
    parser.add_argument("--stage1-n-splits", type=int, default=None)
    parser.add_argument("--stage1-min-train-rows", type=int, default=None)
    parser.add_argument("--stage1-min-val-rows", type=int, default=None)
    parser.add_argument("--stage1-permutation-repeats", type=int, default=None)
    parser.add_argument("--stage1-top-k-features", type=int, default=None)
    parser.add_argument("--stage1-workers", type=int, default=None)
    parser.add_argument("--stage1-model-threads", type=int, default=None)

    parser.add_argument("--stage2-forecast-days", type=float, default=None)
    parser.add_argument("--stage2-train-window-months", type=str, default="")
    parser.add_argument("--stage2-asset-count", type=int, default=None)
    parser.add_argument("--stage2-seed", type=int, default=None)
    parser.add_argument("--stage2-search-back-months", type=int, default=None)
    parser.add_argument("--stage2-sample-interval", type=float, default=None)
    parser.add_argument("--stage2-python-exe", type=str, default="")

    parser.add_argument("--stage3-trials-per-combo", type=int, default=None)
    parser.add_argument("--stage3-model-threads", type=int, default=None)
    parser.add_argument("--stage3-sampler-seed", type=int, default=None)
    parser.add_argument("--stage3-parallel-workers", type=int, default=None)
    parser.add_argument("--stage3-trial-workers", type=int, default=None)
    parser.add_argument("--stage3-pruner-startup-trials", type=int, default=None)
    parser.add_argument("--stage3-pruner-warmup-steps", type=int, default=None)
    parser.add_argument("--stage3-timeout-seconds", type=int, default=None)
    parser.add_argument("--stage3-search-back-months", type=int, default=None)
    parser.add_argument("--stage3-history-window-months", type=int, default=None)
    parser.add_argument("--stage3-study-name-prefix", type=str, default="")
    parser.add_argument("--stage3-storage-root", type=Path, default=None)
    parser.add_argument("--stage3-quiet-progress", action="store_true")
    parser.add_argument("--stage3-strict-cpu-budget", action="store_true")
    parser.add_argument("--stage3-no-resume-study", action="store_true")
    parsed = parser.parse_args(argv)
    if parsed.parquet_root is None:
        try:
            parsed.parquet_root = require_pipeline_io(profile=str(parsed.profile or "pipeline_test")).source_ohlcvt_root
        except PathConfigError:
            parsed.parquet_root = Path("parquet")
    return finalize_sandbox_output_args(
        parsed,
        argv,
        default_output_dir=DEFAULT_OUTPUT_DIR,
        family_key="tabular",
    )


def model_paths(run_root: Path, model_key: str) -> ModelPaths:
    root = run_root / model_key
    logs_dir = root / "logs"
    return ModelPaths(
        root=root,
        stage1_dir=root / "stage1",
        stage2_root=root / "stage2",
        stage3_dir=root / "stage3",
        stage3_storage_path=root / "stage3_studies" / "optuna.sqlite3",
        logs_dir=logs_dir,
        stage1_log=logs_dir / "stage1.log",
        stage2_log=logs_dir / "stage2.log",
        stage3_log=logs_dir / "stage3.log",
    )


def subprocess_env(base_env: Dict[str, str], *, paths: ModelPaths, stage_name: str) -> Dict[str, str]:
    env = dict(base_env)
    temp_root = test_branch_stage_tmp_root(
        env,
        family_key="tabular",
        run_name=paths.root.parent.name,
        model_key=paths.root.name,
        stage_name=str(stage_name),
        fallback_root=(paths.root / "tmp" / str(stage_name)).resolve(),
    )
    env["TMP"] = str(temp_root)
    env["TEMP"] = str(temp_root)
    env["TMPDIR"] = str(temp_root)
    if paths.root.name == "catboost":
        env["CATBOOST_TRAIN_DIR"] = str((temp_root / "catboost_train").resolve())
    return env
def stage0_profile_path(run_root: Path) -> Path:
    return _shared_stage0_profile_path(run_root, STAGE0_PROFILE_JSON)


def stage0_candidates_path(run_root: Path) -> Path:
    return _shared_stage0_candidates_path(run_root, STAGE0_CANDIDATES_CSV)


def stage0_complete(run_root: Path) -> bool:
    return _shared_stage0_complete(run_root, STAGE0_PROFILE_JSON, STAGE0_CANDIDATES_CSV)
def stage3_complete(paths: ModelPaths) -> bool:
    return _shared_stage3_complete(paths, REQUIRED_STAGE3_FILES)


def run_complete(run_root: Path) -> bool:
    return _shared_run_complete(
        run_root,
        model_order=MODEL_ORDER,
        model_paths_fn=model_paths,
        stage0_profile_json_name=STAGE0_PROFILE_JSON,
        stage0_candidates_csv_name=STAGE0_CANDIDATES_CSV,
        required_stage3_files=REQUIRED_STAGE3_FILES,
    )


def latest_incomplete_run(base_output_dir: Path) -> Optional[Path]:
    return _shared_latest_incomplete_run(base_output_dir, run_complete)


def resolve_run_root(args: argparse.Namespace) -> Path:
    return _shared_resolve_run_root(args, latest_incomplete_run)


def run_logged_subprocess(command: Sequence[str], *, cwd: Path, env: Dict[str, str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as logf:
        logf.write(f"[{utc_now_iso()}] RUN {' '.join(command)}\n")
        logf.flush()
        completed = subprocess.run(
            list(command),
            cwd=str(cwd),
            env=env,
            stdout=logf,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        logf.write(f"[{utc_now_iso()}] EXIT {completed.returncode}\n")
        if completed.returncode != 0:
            raise RuntimeError(f"Command failed with exit code {completed.returncode}: {' '.join(command)}")


def run_with_retries(
    command: Sequence[str],
    *,
    cwd: Path,
    env: Dict[str, str],
    log_path: Path,
    max_attempts: int,
    retry_delay_seconds: float,
    cleanup_paths: Optional[Sequence[Path]] = None,
    progress_name: str,
) -> None:
    last_error: Optional[Exception] = None
    attempt_count = max(1, int(max_attempts))
    for attempt in range(1, attempt_count + 1):
        progress_line(f"{progress_name}: attempt {attempt}/{attempt_count}")
        try:
            run_logged_subprocess(command, cwd=cwd, env=env, log_path=log_path)
            progress_line(f"{progress_name}: success")
            return
        except Exception as exc:
            last_error = exc
            progress_line(f"{progress_name}: failed on attempt {attempt}/{attempt_count}: {exc}")
            if attempt >= attempt_count:
                break
            for cleanup_path in cleanup_paths or []:
                if cleanup_path.exists():
                    remove_tree(cleanup_path)
            progress_line(f"{progress_name}: retrying after {float(retry_delay_seconds):.1f}s")
            time.sleep(max(0.0, float(retry_delay_seconds)))
    if last_error is not None:
        raise last_error
    raise RuntimeError("retry loop exited without executing command")


def build_stage0_command(args: argparse.Namespace, run_root: Path) -> List[str]:
    command = [sys.executable, "-m", TABULAR_STAGE0_ENTRYPOINT]
    command_flag(command, "--project-root", args.project_root.resolve())
    command_flag(command, "--parquet-root", args.parquet_root.resolve())
    command_flag(command, "--output-dir", stage0_dir(run_root))
    command_flag(command, "--task", args.stage0_task or args.tasks)
    command_flag(command, "--interval", args.stage0_interval)
    command_flag(command, "--horizon-minutes", args.stage0_horizon_minutes)
    command_flag(command, "--workers", args.stage0_workers)
    command_flag(command, "--threads", args.stage0_threads)
    command_flag(command, "--assets", args.stage0_assets if str(args.stage0_assets).strip() else args.assets)
    command_flag(command, "--backfill-days", args.stage0_backfill_days)
    command_flag(command, "--sample-seconds", args.stage0_sample_seconds)
    command_flag(command, "--ram-cap-pct", args.stage0_ram_cap_pct)
    if bool(args.stage0_prune_run_artifacts):
        command.append("--prune-run-artifacts")
    return command


def load_stage0_profile(run_root: Path) -> Dict[str, Any]:
    return load_json_dict(stage0_profile_path(run_root))


def selected_stage0_profile(run_root: Path) -> Dict[str, Any]:
    payload = load_stage0_profile(run_root)
    selected = payload.get("selected_profile") if isinstance(payload, dict) else None
    return selected if isinstance(selected, dict) else {}


def build_stage1_command(spec: TabularNumericTestModuleSpec, args: argparse.Namespace, paths: ModelPaths) -> List[str]:
    stage0_profile = selected_stage0_profile(paths.root.parent)
    stage1_workers = args.stage1_workers
    if stage1_workers is None and stage0_profile.get("unit_workers") is not None:
        try:
            stage1_workers = int(stage0_profile["unit_workers"])
        except Exception:
            stage1_workers = None
    stage1_model_threads = args.stage1_model_threads
    if stage1_model_threads is None and stage0_profile.get("model_threads") is not None:
        try:
            stage1_model_threads = int(stage0_profile["model_threads"])
        except Exception:
            stage1_model_threads = None
    command = [sys.executable, "-m", spec.stage1_module]
    command_flag(command, "--project-root", args.project_root.resolve())
    command_flag(command, "--parquet-root", args.parquet_root.resolve())
    command_flag(command, "--output-dir", paths.stage1_dir)
    command_flag(command, "--intervals", args.intervals)
    command_flag(command, "--tasks", args.tasks)
    command_flag(command, "--horizon-minutes", args.horizon_minutes)
    command_flag(command, "--combo-list", args.combo_list)
    command_flag(command, "--assets", args.assets)
    command_flag(command, "--asset-count", args.stage1_asset_count)
    command_flag(command, "--seed", args.stage1_seed)
    command_flag(command, "--train-window-months", args.stage1_train_window_months)
    command_flag(command, "--max-rows", args.stage1_max_rows)
    command_flag(command, "--n-splits", args.stage1_n_splits)
    command_flag(command, "--min-train-rows", args.stage1_min_train_rows)
    command_flag(command, "--min-val-rows", args.stage1_min_val_rows)
    command_flag(command, "--permutation-repeats", args.stage1_permutation_repeats)
    command_flag(command, "--top-k-features", args.stage1_top_k_features)
    command_flag(command, "--workers", stage1_workers)
    command_flag(command, "--model-threads", stage1_model_threads)
    return command


def build_stage2_command(spec: TabularNumericTestModuleSpec, args: argparse.Namespace, paths: ModelPaths) -> List[str]:
    command = [sys.executable, "-m", spec.stage2_module, "--staged"]
    command_flag(command, "--project-root", args.project_root.resolve())
    command_flag(command, "--parquet-root", args.parquet_root.resolve())
    command_flag(command, "--output-dir", paths.stage2_root)
    command_flag(command, "--intervals", args.intervals)
    command_flag(command, "--tasks", args.tasks)
    command_flag(command, "--horizon-minutes", args.horizon_minutes)
    command_flag(command, "--assets", args.assets)
    command_flag(command, "--feature-profile-json", stage1_selection_path(paths))
    command_flag(command, "--forecast-days", args.stage2_forecast_days)
    command_flag(command, "--train-window-months", args.stage2_train_window_months)
    command_flag(command, "--asset-count", args.stage2_asset_count)
    command_flag(command, "--seed", args.stage2_seed)
    command_flag(command, "--search-back-months", args.stage2_search_back_months)
    command_flag(command, "--sample-interval", args.stage2_sample_interval)
    command_flag(command, "--python-exe", args.stage2_python_exe)
    return command


def build_stage3_command(spec: TabularNumericTestModuleSpec, args: argparse.Namespace, paths: ModelPaths, *, stage2_manifest: Path, survivor_json: Path, run_root: Path) -> List[str]:
    study_prefix = str(args.stage3_study_name_prefix).strip() or f"tabular_numeric_family_{run_root.name}_{spec.model_key}"
    storage_root = args.stage3_storage_root.resolve() if args.stage3_storage_root else paths.stage3_storage_path.parent.resolve()
    storage_path = (storage_root / spec.model_key / "optuna.sqlite3").resolve()
    assert_write_allowed(storage_path, "tabular test branch optuna storage", roots=resolve_sandbox_output_roots(args))
    stage0_profile = selected_stage0_profile(run_root)
    stage3_model_threads = args.stage3_model_threads
    if stage3_model_threads is None and stage0_profile.get("model_threads") is not None:
        try:
            stage3_model_threads = int(stage0_profile["model_threads"])
        except Exception:
            stage3_model_threads = None
    stage3_parallel_workers = args.stage3_parallel_workers
    if stage3_parallel_workers is None and stage0_profile.get("unit_workers") is not None:
        try:
            stage3_parallel_workers = int(stage0_profile["unit_workers"])
        except Exception:
            stage3_parallel_workers = None
    command = [sys.executable, "-m", spec.stage3_module, "--staged"]
    command_flag(command, "--stage2-manifest", stage2_manifest)
    command_flag(command, "--stage2-survivor-json", survivor_json)
    command_flag(command, "--output-dir", paths.stage3_dir)
    command_flag(command, "--assets", args.assets)
    command_flag(command, "--trials-per-combo", args.stage3_trials_per_combo)
    command_flag(command, "--model-threads", stage3_model_threads)
    command_flag(command, "--sampler-seed", args.stage3_sampler_seed)
    command_flag(command, "--study-name-prefix", study_prefix)
    command_flag(command, "--storage", f"sqlite:///{storage_path.as_posix()}")
    command_flag(command, "--parallel-workers", stage3_parallel_workers)
    command_flag(command, "--trial-workers", args.stage3_trial_workers)
    command_flag(command, "--pruner-startup-trials", args.stage3_pruner_startup_trials)
    command_flag(command, "--pruner-warmup-steps", args.stage3_pruner_warmup_steps)
    command_flag(command, "--timeout-seconds", args.stage3_timeout_seconds)
    command_flag(command, "--search-back-months", args.stage3_search_back_months)
    command_flag(command, "--history-window-months", args.stage3_history_window_months)
    if bool(args.stage3_quiet_progress):
        command.append("--quiet-progress")
    if bool(args.stage3_strict_cpu_budget):
        command.append("--strict-cpu-budget")
    if not bool(args.stage3_no_resume_study):
        command.append("--resume-study")
    return command


def collect_stage3_outputs(paths: ModelPaths) -> Dict[str, str]:
    return _shared_collect_stage3_outputs(paths, REQUIRED_STAGE3_FILES)


def collect_model_status(run_root: Path, model_key: str) -> Dict[str, Any]:
    paths = model_paths(run_root, model_key)
    stage2_manifest = discover_latest_stage2_manifest(paths.stage2_root)
    stage2_survivor = stage2_survivor_json_from_manifest(stage2_manifest)
    return {
        "model_key": model_key,
        "root": str(paths.root),
        "stage1": {
            "output_dir": str(paths.stage1_dir),
            "complete": stage1_complete(paths),
            "selection_json": str(stage1_selection_path(paths)) if stage1_selection_path(paths).exists() else None,
            "run_meta_json": str(stage1_meta_path(paths)) if stage1_meta_path(paths).exists() else None,
            "log_path": str(paths.stage1_log),
        },
        "stage2": {
            "output_root": str(paths.stage2_root),
            "complete": stage2_complete(paths),
            "diagnostic_manifest_json": str(stage2_manifest) if stage2_manifest is not None else None,
            "stage3_survivor_handoff_json": str(stage2_survivor) if stage2_survivor is not None else None,
            "log_path": str(paths.stage2_log),
        },
        "stage3": {
            "output_dir": str(paths.stage3_dir),
            "complete": stage3_complete(paths),
            "storage_path": str(paths.stage3_storage_path),
            "artifacts": collect_stage3_outputs(paths),
            "log_path": str(paths.stage3_log),
        },
    }


def collect_stage0_status(run_root: Path) -> Dict[str, Any]:
    payload = load_stage0_profile(run_root) if stage0_complete(run_root) else {}
    selected_profile = payload.get("selected_profile") if isinstance(payload, dict) else None
    return {
        "module": TABULAR_STAGE0_ENTRYPOINT,
        "output_dir": str(stage0_dir(run_root)),
        "complete": stage0_complete(run_root),
        "profile_json": str(stage0_profile_path(run_root)) if stage0_profile_path(run_root).exists() else None,
        "candidates_csv": str(stage0_candidates_path(run_root)) if stage0_candidates_path(run_root).exists() else None,
        "log_path": str(stage0_log_path(run_root)),
        "selected_profile": selected_profile if isinstance(selected_profile, dict) else None,
        "runtime_config_path": payload.get("runtime_config_path") if isinstance(payload, dict) else None,
    }


def write_run_state(run_root: Path, args: argparse.Namespace, *, active_model: Optional[str] = None, active_stage: Optional[str] = None) -> Dict[str, Any]:
    stage0_status = collect_stage0_status(run_root)
    model_statuses = {model_key: collect_model_status(run_root, model_key) for model_key in MODEL_ORDER}
    health = write_test_branch_health(
        run_root,
        family="Tabular_Numeric",
        stage0_status=stage0_status,
        models=model_statuses,
    )
    payload = {
        "generated_utc": utc_now_iso(),
        "run_root": str(run_root),
        "active_model": active_model,
        "active_stage": active_stage,
        "complete": run_complete(run_root),
        "args": {key: (str(value) if isinstance(value, Path) else value) for key, value in vars(args).items()},
        "stage0": stage0_status,
        "models": model_statuses,
        "health": {
            "status": health.get("status"),
            "error_count": health.get("error_count"),
            "warning_count": health.get("warning_count"),
            "health_report_json": str((run_root / HEALTH_REPORT_FILE).resolve()),
        },
    }
    write_json_atomic(run_root / RUN_STATE_FILE, payload)
    summary_payload = {
        "generated_utc": payload["generated_utc"],
        "run_root": str(run_root),
        "complete": payload["complete"],
        "stage0_profile": payload["stage0"]["selected_profile"],
        "health": payload["health"],
        "stage3_outputs": {
            model_key: payload["models"][model_key]["stage3"]["artifacts"]
            for model_key in MODEL_ORDER
            if payload["models"][model_key]["stage3"]["artifacts"]
        },
    }
    write_json_atomic(run_root / RUN_SUMMARY_FILE, summary_payload)
    return payload


def run_stage0(args: argparse.Namespace, run_root: Path, env: Dict[str, str]) -> None:
    progress_name = "Tabular Family Stage 0"
    if stage0_complete(run_root):
        progress_line(f"{progress_name}: already complete, skipping")
        return
    progress_line(f"{progress_name}: starting")
    if stage0_dir(run_root).exists():
        remove_tree(stage0_dir(run_root))
    write_run_state(run_root, args, active_stage="stage0")
    command = build_stage0_command(args, run_root)
    stage_env = dict(env)
    temp_root = test_branch_stage_tmp_root(
        stage_env,
        family_key="tabular",
        run_name=run_root.name,
        model_key="stage0",
        stage_name="stage0",
        fallback_root=(run_root / "tmp" / "stage0").resolve(),
    )
    stage_env["TMP"] = str(temp_root)
    stage_env["TEMP"] = str(temp_root)
    stage_env["TMPDIR"] = str(temp_root)
    run_with_retries(
        command,
        cwd=args.project_root.resolve(),
        env=stage_env,
        log_path=stage0_log_path(run_root),
        max_attempts=args.max_attempts,
        retry_delay_seconds=args.retry_delay_seconds,
        cleanup_paths=[stage0_dir(run_root)],
        progress_name=progress_name,
    )
    if not stage0_complete(run_root):
        raise RuntimeError("Stage 0 did not leave complete profile artifacts for tabular family")


def run_stage1(spec: TabularNumericTestModuleSpec, args: argparse.Namespace, run_root: Path, env: Dict[str, str]) -> None:
    paths = model_paths(run_root, spec.model_key)
    progress_name = f"{spec.display_name} Stage 1"
    if stage1_complete(paths):
        progress_line(f"{progress_name}: already complete, skipping")
        return
    progress_line(f"{progress_name}: starting")
    if paths.stage1_dir.exists():
        remove_tree(paths.stage1_dir)
    write_run_state(run_root, args, active_model=spec.model_key, active_stage="stage1")
    command = build_stage1_command(spec, args, paths)
    stage_env = subprocess_env(env, paths=paths, stage_name="stage1")
    run_with_retries(command, cwd=args.project_root.resolve(), env=stage_env, log_path=paths.stage1_log, max_attempts=args.max_attempts, retry_delay_seconds=args.retry_delay_seconds, cleanup_paths=[paths.stage1_dir], progress_name=progress_name)
    if not stage1_complete(paths):
        raise RuntimeError(f"Stage 1 did not leave complete artifacts for {spec.model_key}")


def run_stage2(spec: TabularNumericTestModuleSpec, args: argparse.Namespace, run_root: Path, env: Dict[str, str]) -> Path:
    paths = model_paths(run_root, spec.model_key)
    progress_name = f"{spec.display_name} Stage 2"
    manifest_path = discover_latest_stage2_manifest(paths.stage2_root)
    survivor_path = stage2_survivor_json_from_manifest(manifest_path)
    if manifest_path is not None and survivor_path is not None:
        progress_line(f"{progress_name}: already complete, skipping")
        return manifest_path
    progress_line(f"{progress_name}: starting")
    write_run_state(run_root, args, active_model=spec.model_key, active_stage="stage2")
    command = build_stage2_command(spec, args, paths)
    stage_env = subprocess_env(env, paths=paths, stage_name="stage2")
    run_with_retries(command, cwd=args.project_root.resolve(), env=stage_env, log_path=paths.stage2_log, max_attempts=args.max_attempts, retry_delay_seconds=args.retry_delay_seconds, progress_name=progress_name)
    manifest_path = discover_latest_stage2_manifest(paths.stage2_root)
    survivor_path = stage2_survivor_json_from_manifest(manifest_path)
    if manifest_path is None or survivor_path is None:
        raise RuntimeError(f"Stage 2 did not leave complete staged handoff artifacts for {spec.model_key}")
    return manifest_path


def run_stage3(spec: TabularNumericTestModuleSpec, args: argparse.Namespace, run_root: Path, env: Dict[str, str], *, stage2_manifest: Path) -> None:
    paths = model_paths(run_root, spec.model_key)
    progress_name = f"{spec.display_name} Stage 3"
    if stage3_complete(paths):
        progress_line(f"{progress_name}: already complete, skipping")
        return
    survivor_json = stage2_survivor_json_from_manifest(stage2_manifest)
    if survivor_json is None:
        raise RuntimeError(f"Missing Stage-2 survivor handoff for {spec.model_key}: {stage2_manifest}")
    progress_line(f"{progress_name}: starting")
    if paths.stage3_dir.exists():
        remove_tree(paths.stage3_dir)
    write_run_state(run_root, args, active_model=spec.model_key, active_stage="stage3")
    command = build_stage3_command(spec, args, paths, stage2_manifest=stage2_manifest, survivor_json=survivor_json, run_root=run_root)
    stage_env = subprocess_env(env, paths=paths, stage_name="stage3")
    run_with_retries(command, cwd=args.project_root.resolve(), env=stage_env, log_path=paths.stage3_log, max_attempts=args.max_attempts, retry_delay_seconds=args.retry_delay_seconds, cleanup_paths=[paths.stage3_dir], progress_name=progress_name)
    if not stage3_complete(paths):
        raise RuntimeError(f"Stage 3 did not leave complete artifacts for {spec.model_key}")


def run_orchestrator(args: argparse.Namespace) -> Path:
    run_root = resolve_run_root(args)
    run_root.mkdir(parents=True, exist_ok=True)
    env = test_branch_child_env(args, dict(os.environ))
    assert_test_branch_sandbox_launch(args, run_root, env, family_key="tabular")
    if args.resume_run:
        progress_line(f"Resuming orchestrator run at {run_root}")
    elif run_root.exists() and any(run_root.iterdir()):
        progress_line(f"Continuing orchestrator run at {run_root}")
    else:
        progress_line(f"Starting orchestrator run at {run_root}")
    write_run_state(run_root, args)
    run_stage0(args, run_root, env)
    write_run_state(run_root, args)
    total_models = len(MODEL_ORDER)
    for index, model_key in enumerate(MODEL_ORDER, start=1):
        spec = MODEL_SPECS[model_key]
        progress_line(f"[{index}/{total_models}] {spec.display_name}: entering family partition")
        run_stage1(spec, args, run_root, env)
        stage2_manifest = run_stage2(spec, args, run_root, env)
        run_stage3(spec, args, run_root, env, stage2_manifest=stage2_manifest)
        progress_line(f"[{index}/{total_models}] {spec.display_name}: complete")
        write_run_state(run_root, args)
    write_run_state(run_root, args)
    progress_line(f"Orchestrator run complete: {run_root}")
    return run_root


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    run_root = run_orchestrator(args)
    print(json.dumps({"run_root": str(run_root), "summary_json": str((run_root / RUN_SUMMARY_FILE).resolve())}, indent=2))


if __name__ == "__main__":
    main()
