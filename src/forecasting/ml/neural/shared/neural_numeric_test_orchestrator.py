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
from typing import Any, Dict, Optional, Sequence, Tuple

from src.forecasting.common.path_config import resolve_path, selected_profile

try:
    import psutil  # type: ignore
except Exception:
    psutil = None  # type: ignore

from src.forecasting.common.sandbox_paths import assert_write_allowed, resolve_sandbox_output_roots
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

from src.forecasting.ml.neural.shared.neural_numeric_model_registry import NEURAL_STAGE0_ENTRYPOINT

MODEL_ORDER = ["lstm", "tcn", "nbeats"]


@dataclass(frozen=True)
class NeuralNumericTestModuleSpec:
    model_key: str
    display_name: str
    stage1_module: str
    stage2_module: str
    stage3_module: str


MODEL_SPECS: Dict[str, NeuralNumericTestModuleSpec] = {
    "lstm": NeuralNumericTestModuleSpec(
        model_key="lstm",
        display_name="LSTM",
        stage1_module="src.forecasting.ml.neural.lstm.lstm_feature_experiment",
        stage2_module="src.forecasting.ml.neural.lstm.lstm_numeric_scaling_test",
        stage3_module="src.forecasting.ml.neural.lstm.lstm_numeric_optuna_tuning",
    ),
    "tcn": NeuralNumericTestModuleSpec(
        model_key="tcn",
        display_name="TCN",
        stage1_module="src.forecasting.ml.neural.tcn.tcn_feature_experiment",
        stage2_module="src.forecasting.ml.neural.tcn.tcn_numeric_scaling_test",
        stage3_module="src.forecasting.ml.neural.tcn.tcn_numeric_optuna_tuning",
    ),
    "nbeats": NeuralNumericTestModuleSpec(
        model_key="nbeats",
        display_name="N-BEATS",
        stage1_module="src.forecasting.ml.neural.nbeats.nbeats_feature_experiment",
        stage2_module="src.forecasting.ml.neural.nbeats.nbeats_numeric_scaling_test",
        stage3_module="src.forecasting.ml.neural.nbeats.nbeats_numeric_optuna_tuning",
    ),
}

RUN_STATE_FILE = "neural_numeric_test_orchestrator_state.json"
RUN_SUMMARY_FILE = "neural_numeric_stage3_artifacts.json"
DEFAULT_OUTPUT_DIR = Path("logs") / "diagnostics" / "neural_numeric_family_test_orchestrator"
STAGE0_PROFILE_JSON = "neural_numeric_stage0_profile.json"
STAGE0_CANDIDATES_CSV = "neural_numeric_stage0_candidates.csv"
REQUIRED_STAGE3_FILES = (
    "combo_results.csv",
    "optuna_trials.json",
    "unit_metrics.csv",
    "summary.md",
    "representative_samples.csv",
)
DEFAULT_MONITOR_INTERVAL_SECONDS = 30.0


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
def process_meta_path(log_path: Path) -> Path:
    return log_path.with_suffix(".process.json")


def write_process_meta(
    meta_path: Path,
    *,
    status: str,
    command: Sequence[str],
    cwd: Path,
    env: Dict[str, str],
    log_path: Path,
    pid: Optional[int] = None,
    started_utc: Optional[str] = None,
    finished_utc: Optional[str] = None,
    returncode: Optional[int] = None,
    error: Optional[str] = None,
    monitor_root: Optional[Path] = None,
) -> None:
    payload: Dict[str, Any] = {
        "status": str(status),
        "command": [str(part) for part in command],
        "cwd": str(cwd),
        "log_path": str(log_path),
        "pid": pid,
        "started_utc": started_utc,
        "finished_utc": finished_utc,
        "returncode": returncode,
        "error": error,
        "monitor_root": str(monitor_root) if monitor_root is not None else None,
        "env_hints": {
            "TMP": env.get("TMP"),
            "TEMP": env.get("TEMP"),
            "TMPDIR": env.get("TMPDIR"),
            "NEURAL_NUMERIC_MODEL_THREADS": env.get("NEURAL_NUMERIC_MODEL_THREADS"),
        },
    }
    write_json_atomic(meta_path, payload)


def _latest_file_info(root: Optional[Path]) -> Tuple[int, Optional[Path], Optional[float]]:
    if root is None or not root.exists():
        return 0, None, None
    file_count = 0
    latest_path: Optional[Path] = None
    latest_mtime: Optional[float] = None
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        file_count += 1
        try:
            mtime = float(path.stat().st_mtime)
        except Exception:
            continue
        if latest_mtime is None or mtime > latest_mtime:
            latest_mtime = mtime
            latest_path = path
    return file_count, latest_path, latest_mtime


def _monitor_root_snapshot(root: Optional[Path]) -> Dict[str, Any]:
    if root is None or not root.exists():
        return {"dir_count": 0, "dir_names": []}
    dir_names = sorted(path.name for path in root.iterdir() if path.is_dir())
    return {
        "dir_count": int(len(dir_names)),
        "dir_names": dir_names[:6],
    }


def _process_snapshot(pid: int) -> Dict[str, Any]:
    if psutil is None:
        return {}
    try:
        proc = psutil.Process(int(pid))
        children = proc.children(recursive=True)
        cpu_times = proc.cpu_times()
        rss_mb = float(proc.memory_info().rss) / (1024.0 * 1024.0)
        child_cpu_seconds = 0.0
        for child in children:
            try:
                child_times = child.cpu_times()
                child_cpu_seconds += float(getattr(child_times, "user", 0.0) + getattr(child_times, "system", 0.0))
            except Exception:
                continue
        return {
            "status": str(proc.status()),
            "cpu_seconds": float(getattr(cpu_times, "user", 0.0) + getattr(cpu_times, "system", 0.0)),
            "rss_mb": float(rss_mb),
            "child_count": int(len(children)),
            "child_cpu_seconds": float(child_cpu_seconds),
        }
    except Exception:
        return {}


def progress_line(message: str) -> None:
    print(f"[{utc_now_iso()}] {message}", flush=True)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sequential Stage-1/2/3 test orchestrator for the NeuralTS numerics family.")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[5])
    parser.add_argument("--profile", type=str, default=selected_profile(default="pipeline_test"))
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
    parser.add_argument("--stage0-fit-days", type=int, default=None)
    parser.add_argument("--stage0-seq-len", type=int, default=None)
    parser.add_argument("--stage0-batch-size", type=int, default=None)
    parser.add_argument("--stage0-epochs", type=int, default=None)
    parser.add_argument("--stage0-sample-seconds", type=float, default=None)
    parser.add_argument("--stage0-ram-cap-pct", type=float, default=None)
    parser.add_argument("--stage0-prune-run-artifacts", action="store_true")

    parser.add_argument("--stage1-asset-count", type=int, default=None)
    parser.add_argument("--stage1-train-window-months", type=int, default=None)

    parser.add_argument("--stage2-forecast-days", type=float, default=None)
    parser.add_argument("--stage2-train-window-months", type=str, default="")
    parser.add_argument("--stage2-asset-count", type=int, default=None)

    parser.add_argument("--stage3-trials-per-combo", type=int, default=None)
    parser.add_argument("--stage3-model-threads", type=int, default=None)
    parser.add_argument("--stage3-sampler-seed", type=int, default=None)
    parser.add_argument("--stage3-history-window-months", type=int, default=None)
    parser.add_argument("--stage3-study-name-prefix", type=str, default="")
    parser.add_argument("--stage3-storage-root", type=Path, default=None)
    parser.add_argument("--stage3-no-resume-study", action="store_true")
    args = parser.parse_args(argv)
    if args.parquet_root is None:
        args.parquet_root = Path(resolve_path("source_ohlcvt_root", profile=str(args.profile), required=False) or Path("parquet"))
    return finalize_sandbox_output_args(
        args,
        argv,
        default_output_dir=DEFAULT_OUTPUT_DIR,
        family_key="neural",
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
        family_key="neural",
        run_name=paths.root.parent.name,
        model_key=paths.root.name,
        stage_name=str(stage_name),
        fallback_root=(paths.root / "tmp" / str(stage_name)).resolve(),
    )
    env["TMP"] = str(temp_root)
    env["TEMP"] = str(temp_root)
    env["TMPDIR"] = str(temp_root)
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


def run_logged_subprocess(
    command: Sequence[str],
    *,
    cwd: Path,
    env: Dict[str, str],
    log_path: Path,
    progress_name: str,
    monitor_root: Optional[Path] = None,
    monitor_interval_seconds: float = DEFAULT_MONITOR_INTERVAL_SECONDS,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path = process_meta_path(log_path)
    started_utc = utc_now_iso()
    with log_path.open("a", encoding="utf-8") as logf:
        logf.write(f"[{utc_now_iso()}] RUN {' '.join(command)}\n")
        logf.flush()
        process = subprocess.Popen(list(command), cwd=str(cwd), env=env, stdout=logf, stderr=subprocess.STDOUT, text=True)
        write_process_meta(
            meta_path,
            status="running",
            command=command,
            cwd=cwd,
            env=env,
            log_path=log_path,
            pid=int(process.pid),
            started_utc=started_utc,
            monitor_root=monitor_root,
        )
        start_time = time.time()
        last_report = 0.0
        while True:
            returncode = process.poll()
            now = time.time()
            if returncode is not None:
                finished_utc = utc_now_iso()
                logf.write(f"[{finished_utc}] EXIT {int(returncode)}\n")
                logf.flush()
                if int(returncode) != 0:
                    write_process_meta(
                        meta_path,
                        status="failed",
                        command=command,
                        cwd=cwd,
                        env=env,
                        log_path=log_path,
                        pid=int(process.pid),
                        started_utc=started_utc,
                        finished_utc=finished_utc,
                        returncode=int(returncode),
                        error=f"Command failed with exit code {int(returncode)}: {' '.join(command)}",
                        monitor_root=monitor_root,
                    )
                    raise RuntimeError(f"Command failed with exit code {int(returncode)}: {' '.join(command)}")
                write_process_meta(
                    meta_path,
                    status="completed",
                    command=command,
                    cwd=cwd,
                    env=env,
                    log_path=log_path,
                    pid=int(process.pid),
                    started_utc=started_utc,
                    finished_utc=finished_utc,
                    returncode=int(returncode),
                    monitor_root=monitor_root,
                )
                return
            if now - last_report >= max(5.0, float(monitor_interval_seconds)):
                elapsed_seconds = int(now - start_time)
                log_size = log_path.stat().st_size if log_path.exists() else 0
                log_mtime = log_path.stat().st_mtime if log_path.exists() else None
                file_count, latest_path, latest_mtime = _latest_file_info(monitor_root)
                root_snapshot = _monitor_root_snapshot(monitor_root)
                proc_snapshot = _process_snapshot(int(process.pid))
                latest_age = None if latest_mtime is None else max(0, int(now - latest_mtime))
                log_age = None if log_mtime is None else max(0, int(now - log_mtime))
                latest_rel = None
                if latest_path is not None:
                    try:
                        latest_rel = str(latest_path.relative_to(monitor_root if monitor_root is not None else latest_path.parent))
                    except Exception:
                        latest_rel = str(latest_path)
                progress_line(
                    f"{progress_name}: pid={int(process.pid)} elapsed={elapsed_seconds}s "
                    f"proc_status={proc_snapshot.get('status', 'n/a')} cpu_s={proc_snapshot.get('cpu_seconds', 'n/a')} "
                    f"rss_mb={round(float(proc_snapshot['rss_mb']), 1) if 'rss_mb' in proc_snapshot else 'n/a'} "
                    f"child_count={proc_snapshot.get('child_count', 'n/a')} child_cpu_s={round(float(proc_snapshot['child_cpu_seconds']), 1) if 'child_cpu_seconds' in proc_snapshot else 'n/a'} "
                    f"log_size={int(log_size)}B log_age={log_age if log_age is not None else 'n/a'}s "
                    f"artifact_files={int(file_count)} latest_artifact_age={latest_age if latest_age is not None else 'n/a'}s "
                    f"latest_artifact={latest_rel if latest_rel is not None else 'n/a'} "
                    f"artifact_dirs={root_snapshot.get('dir_count', 0)} dirs={','.join(root_snapshot.get('dir_names', [])) if root_snapshot.get('dir_names') else 'n/a'}"
                )
                last_report = now
            time.sleep(1.0)


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
    monitor_root: Optional[Path] = None,
) -> None:
    last_error: Optional[Exception] = None
    attempt_count = max(1, int(max_attempts))
    for attempt in range(1, attempt_count + 1):
        progress_line(f"{progress_name}: attempt {attempt}/{attempt_count}")
        try:
            try:
                run_logged_subprocess(
                    command,
                    cwd=cwd,
                    env=env,
                    log_path=log_path,
                    progress_name=progress_name,
                    monitor_root=monitor_root,
                )
            except TypeError as exc:
                if "unexpected keyword argument" not in str(exc):
                    raise
                run_logged_subprocess(
                    command,
                    cwd=cwd,
                    env=env,
                    log_path=log_path,
                )
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


def build_stage0_command(args: argparse.Namespace, run_root: Path) -> list[str]:
    command = [sys.executable, "-m", NEURAL_STAGE0_ENTRYPOINT]
    command_flag(command, "--project-root", args.project_root.resolve())
    command_flag(command, "--parquet-root", args.parquet_root.resolve())
    command_flag(command, "--output-dir", stage0_dir(run_root))
    command_flag(command, "--task", args.stage0_task)
    command_flag(command, "--interval", args.stage0_interval)
    command_flag(command, "--horizon-minutes", args.stage0_horizon_minutes)
    command_flag(command, "--workers", args.stage0_workers)
    command_flag(command, "--threads", args.stage0_threads)
    command_flag(command, "--assets", args.stage0_assets if str(args.stage0_assets).strip() else args.assets)
    command_flag(command, "--backfill-days", args.stage0_backfill_days)
    command_flag(command, "--fit-days", args.stage0_fit_days)
    command_flag(command, "--seq-len", args.stage0_seq_len)
    command_flag(command, "--batch-size", args.stage0_batch_size)
    command_flag(command, "--epochs", args.stage0_epochs)
    command_flag(command, "--sample-seconds", args.stage0_sample_seconds)
    command_flag(command, "--ram-cap-pct", args.stage0_ram_cap_pct)
    if bool(args.stage0_prune_run_artifacts):
        command.append("--prune-run-artifacts")
    return command


def build_stage1_command(spec: NeuralNumericTestModuleSpec, args: argparse.Namespace, paths: ModelPaths) -> list[str]:
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
    command_flag(command, "--train-window-months", args.stage1_train_window_months)
    return command


def build_stage2_command(spec: NeuralNumericTestModuleSpec, args: argparse.Namespace, paths: ModelPaths) -> list[str]:
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
    run_root = paths.root.parent
    selected_profile = None
    if stage0_complete(run_root):
        payload = load_json_dict(stage0_profile_path(run_root))
        if isinstance(payload, dict):
            selected_profile = payload.get("selected_profile")
    if isinstance(selected_profile, dict):
        command_flag(command, "--asset-workers", selected_profile.get("asset_workers"))
        command_flag(command, "--model-threads", selected_profile.get("model_threads"))
    return command


def build_stage3_command(spec: NeuralNumericTestModuleSpec, args: argparse.Namespace, paths: ModelPaths, *, stage2_manifest: Path, survivor_json: Path, run_root: Path) -> list[str]:
    study_prefix = str(args.stage3_study_name_prefix).strip() or f"neural_numeric_family_{run_root.name}_{spec.model_key}"
    storage_path = resolve_stage3_storage_path(spec, args, paths)
    selected_threads = args.stage3_model_threads
    if selected_threads is None:
        stage0_payload = load_json_dict(stage0_profile_path(run_root))
        selected_profile = stage0_payload.get("selected_profile") if isinstance(stage0_payload, dict) else None
        if isinstance(selected_profile, dict) and selected_profile.get("model_threads") is not None:
            try:
                selected_threads = int(selected_profile["model_threads"])
            except Exception:
                selected_threads = None
    command = [sys.executable, "-m", spec.stage3_module, "--staged"]
    command_flag(command, "--stage2-manifest", stage2_manifest)
    command_flag(command, "--stage2-survivor-json", survivor_json)
    command_flag(command, "--output-dir", paths.stage3_dir)
    command_flag(command, "--assets", args.assets)
    command_flag(command, "--trials-per-combo", args.stage3_trials_per_combo)
    command_flag(command, "--model-threads", selected_threads)
    command_flag(command, "--sampler-seed", args.stage3_sampler_seed)
    command_flag(command, "--study-name-prefix", study_prefix)
    command_flag(command, "--storage", f"sqlite:///{storage_path.as_posix()}")
    command_flag(command, "--history-window-months", args.stage3_history_window_months)
    if not bool(args.stage3_no_resume_study):
        command.append("--resume-study")
    return command


def resolve_stage3_storage_path(spec: NeuralNumericTestModuleSpec, args: argparse.Namespace, paths: ModelPaths) -> Path:
    storage_root = args.stage3_storage_root.resolve() if args.stage3_storage_root else paths.stage3_storage_path.parent.resolve()
    storage_path = (storage_root / spec.model_key / "optuna.sqlite3").resolve()
    assert_write_allowed(storage_path, "neural test branch optuna storage", roots=resolve_sandbox_output_roots(args))
    return storage_path


def collect_stage3_outputs(paths: ModelPaths) -> Dict[str, str]:
    return _shared_collect_stage3_outputs(paths, REQUIRED_STAGE3_FILES)


def collect_stage0_status(run_root: Path) -> Dict[str, Any]:
    payload = load_json_dict(stage0_profile_path(run_root)) if stage0_complete(run_root) else {}
    selected_profile = payload.get("selected_profile") if isinstance(payload, dict) else None
    return {
        "module": NEURAL_STAGE0_ENTRYPOINT,
        "output_dir": str(stage0_dir(run_root)),
        "complete": stage0_complete(run_root),
        "profile_json": str(stage0_profile_path(run_root)) if stage0_profile_path(run_root).exists() else None,
        "candidates_csv": str(stage0_candidates_path(run_root)) if stage0_candidates_path(run_root).exists() else None,
        "log_path": str(stage0_log_path(run_root)),
        "selected_profile": selected_profile if isinstance(selected_profile, dict) else None,
        "runtime_config_path": payload.get("runtime_config_path") if isinstance(payload, dict) else None,
    }


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


def write_run_state(run_root: Path, args: argparse.Namespace, *, active_model: Optional[str] = None, active_stage: Optional[str] = None) -> Dict[str, Any]:
    stage0_status = collect_stage0_status(run_root)
    model_statuses = {model_key: collect_model_status(run_root, model_key) for model_key in MODEL_ORDER}
    health = write_test_branch_health(
        run_root,
        family="Neural_Numeric",
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
        "stage3_outputs": {model_key: payload["models"][model_key]["stage3"]["artifacts"] for model_key in MODEL_ORDER if payload["models"][model_key]["stage3"]["artifacts"]},
    }
    write_json_atomic(run_root / RUN_SUMMARY_FILE, summary_payload)
    return payload


def run_stage0(args: argparse.Namespace, run_root: Path, env: Dict[str, str]) -> None:
    progress_name = "Neural Family Stage 0"
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
        family_key="neural",
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
        monitor_root=stage0_dir(run_root),
    )
    if not stage0_complete(run_root):
        raise RuntimeError("Stage 0 did not leave complete profile artifacts for neural family")


def run_stage1(spec: NeuralNumericTestModuleSpec, args: argparse.Namespace, run_root: Path, env: Dict[str, str]) -> None:
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


def run_stage2(spec: NeuralNumericTestModuleSpec, args: argparse.Namespace, run_root: Path, env: Dict[str, str]) -> Path:
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


def run_stage3(spec: NeuralNumericTestModuleSpec, args: argparse.Namespace, run_root: Path, env: Dict[str, str], *, stage2_manifest: Path) -> None:
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
    resolve_stage3_storage_path(spec, args, paths).parent.mkdir(parents=True, exist_ok=True)
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
    assert_test_branch_sandbox_launch(args, run_root, env, family_key="neural")
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
