from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from src.forecasting.common.io_atomic import atomic_replace, sibling_temp_path
from src.forecasting.common.path_config import resolve_path, selected_profile
from src.forecasting.common.sandbox_paths import assert_write_allowed
from src.forecasting.common.stats_module_utils import resolve_seasonality_profile
from src.forecasting.ml.shared.numeric_cohort_common import (
    CLAMP_START_MONTH,
    CLAMP_START_YEAR,
    DEFAULT_COHORT_WINDOW_MONTHS,
    DEFAULT_SEARCH_BACK_MONTHS,
    FIXED_NUMERIC_FAMILY_COHORT_SYMBOLS,
    MonthKey,
    common_recent_window,
    select_representative_assets,
)
from src.forecasting.ml.shared.test_orchestrator_common import (
    HEALTH_REPORT_FILE,
    add_sandbox_output_args,
    assert_test_branch_sandbox_launch,
    finalize_sandbox_output_args,
    publish_canonical_family_profiles,
    test_branch_child_env,
    test_branch_stage_tmp_root,
    write_test_branch_health,
)
from src.forecasting.ml.shared.test_branch_function_telemetry import emit_event, emit_subprocess_event
from src.forecasting.stats.shared.stats_numeric_model_registry import (
    STATS_NUMERIC_BRANCHES,
    STATS_STAGE0_ENTRYPOINT,
    STATS_STAGE1_ENTRYPOINTS,
    STATS_STAGE2_ENTRYPOINTS,
    STATS_STAGE3_ENTRYPOINTS,
)

MODEL_ORDER = list(STATS_NUMERIC_BRANCHES)
RUN_STATE_FILE = "stats_numeric_test_orchestrator_state.json"
RUN_SUMMARY_FILE = "stats_numeric_stage3_artifacts.json"
DEFAULT_OUTPUT_DIR = Path("logs") / "diagnostics" / "stats_numeric_family_test_orchestrator"
DEFAULT_PARQUET_ROOT = Path(resolve_path("source_ohlcvt_root", profile=selected_profile(default="pipeline_test"), required=False) or Path("parquet"))
STAGE0_PROFILE_JSON = "stats_numeric_stage0_profile.json"
DEFAULT_STAGE1_ASSET_COUNT = len(FIXED_NUMERIC_FAMILY_COHORT_SYMBOLS)
REQUIRED_STAGE3_FILES = (
    "combo_results.csv",
    "optuna_trials.json",
    "unit_metrics.csv",
    "summary.md",
    "representative_samples.csv",
    "stage3_summary.json",
    "production_profile.json",
    "stage3_survivor_handoff.json",
)
REQUIRED_STAGE2_FILES = (
    "diagnostic_manifest.json",
    "stage3_survivor_handoff.json",
)


@dataclass(frozen=True)
class StatsNumericTestModuleSpec:
    model_key: str
    display_name: str
    stage1_module: str
    stage2_module: str
    stage3_module: str


MODEL_SPECS: Dict[str, StatsNumericTestModuleSpec] = {
    "sarimax": StatsNumericTestModuleSpec("sarimax", "SARIMAX", STATS_STAGE1_ENTRYPOINTS["sarimax"], STATS_STAGE2_ENTRYPOINTS["sarimax"], STATS_STAGE3_ENTRYPOINTS["sarimax"]),
    "llt": StatsNumericTestModuleSpec("llt", "LLT", STATS_STAGE1_ENTRYPOINTS["llt"], STATS_STAGE2_ENTRYPOINTS["llt"], STATS_STAGE3_ENTRYPOINTS["llt"]),
    "egarch": StatsNumericTestModuleSpec("egarch", "EGARCH", STATS_STAGE1_ENTRYPOINTS["egarch"], STATS_STAGE2_ENTRYPOINTS["egarch"], STATS_STAGE3_ENTRYPOINTS["egarch"]),
    "quantreg": StatsNumericTestModuleSpec("quantreg", "QuantReg", STATS_STAGE1_ENTRYPOINTS["quantreg"], STATS_STAGE2_ENTRYPOINTS["quantreg"], STATS_STAGE3_ENTRYPOINTS["quantreg"]),
}


@dataclass(frozen=True)
class ModelPaths:
    root: Path
    stage1_dir: Path
    stage2_dir: Path
    stage3_dir: Path
    logs_dir: Path
    stage1_log: Path
    stage2_log: Path
    stage3_log: Path


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def utc_now_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def write_json_atomic(path: Path, payload: Dict[str, Any]) -> None:
    assert_write_allowed(path, "stats test branch json")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = sibling_temp_path(path)
    assert_write_allowed(tmp, "stats test branch json temp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    atomic_replace(tmp, path)


def load_json_dict(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def progress_line(message: str) -> None:
    print(f"[{utc_now_iso()}] {message}", flush=True)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sequential Stage-0/1/2/3 test orchestrator for the frequentist stats family.")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--profile", type=str, default=selected_profile(default="pipeline_test"))
    parser.add_argument("--parquet-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    add_sandbox_output_args(parser)
    parser.add_argument("--run-id", type=str, default="")
    parser.add_argument("--resume-run", type=str, default="")
    parser.add_argument("--no-resume-latest", action="store_true")
    parser.add_argument("--combo-list", type=str, default="60:240:log_return,240:1440:realized_vol,1440:4320:true_range")
    parser.add_argument("--assets", type=str, default="")
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--retry-delay-seconds", type=float, default=2.0)
    parser.add_argument("--stage0-combo-list", type=str, default="")
    parser.add_argument("--stage0-workers", type=str, default="")
    parser.add_argument("--stage0-threads", type=str, default="")
    parser.add_argument("--stage0-assets", type=str, default="")
    parser.add_argument("--stage0-backfill-days", type=int, default=None)
    parser.add_argument("--stage0-fit-days", type=int, default=None)
    parser.add_argument("--stage0-sample-seconds", type=float, default=None)
    parser.add_argument("--stage0-ram-cap-pct", type=float, default=None)
    parser.add_argument("--stage0-prune-run-artifacts", action="store_true")
    parser.add_argument("--stage1-asset-count", type=int, default=DEFAULT_STAGE1_ASSET_COUNT)
    parser.add_argument("--stage1-seed", type=int, default=17)
    parser.add_argument("--stage1-workers", type=int, default=None)
    parser.add_argument("--stage1-backfill-days", type=int, default=7)
    parser.add_argument("--stage2-workers", type=int, default=None)
    parser.add_argument("--stage2-backfill-days", type=int, default=28)
    parser.add_argument("--stage3-workers", type=int, default=None)
    parser.add_argument("--stage3-backfill-days", type=int, default=14)
    parser.add_argument("--fit-days", type=int, default=180)
    parser.add_argument("--artifact-period-bars", type=int, default=0)
    parser.add_argument("--disable-seasonality-artifact-period", action="store_true")
    parser.add_argument("--seasonality-period-max-assets", type=int, default=3)
    args = parser.parse_args(argv)
    if args.parquet_root is None:
        args.parquet_root = Path(resolve_path("source_ohlcvt_root", profile=str(args.profile), required=False) or DEFAULT_PARQUET_ROOT)
    return finalize_sandbox_output_args(
        args,
        argv,
        default_output_dir=DEFAULT_OUTPUT_DIR,
        family_key="stats",
    )


def resolve_run_root(args: argparse.Namespace) -> Path:
    base = Path(args.output_dir)
    if str(args.resume_run).strip():
        return (base / f"run={str(args.resume_run).strip()}").resolve()
    if not bool(args.no_resume_latest):
        latest = latest_incomplete_run(base)
        if latest is not None:
            return latest.resolve()
    run_id = str(args.run_id).strip() or utc_now_stamp()
    return (base / f"run={run_id}").resolve()


def stage0_dir(run_root: Path) -> Path:
    return Path(run_root) / "stage0"


def stage0_profile_path(run_root: Path) -> Path:
    return stage0_dir(run_root) / STAGE0_PROFILE_JSON


def stage0_complete(run_root: Path) -> bool:
    payload = load_json_dict(stage0_profile_path(run_root))
    selected = payload.get("selected_profile")
    candidates = payload.get("candidates")
    if not isinstance(selected, dict) or not isinstance(candidates, list) or not candidates:
        return False
    if not (stage0_dir(run_root) / "stats_numeric_stage0_candidates.csv").exists():
        return False
    return any(isinstance(row, dict) and isinstance(row.get("branch_results"), list) and row.get("branch_results") for row in candidates)


def model_paths(run_root: Path, model_key: str) -> ModelPaths:
    root = Path(run_root) / str(model_key)
    logs = root / "logs"
    return ModelPaths(
        root=root,
        stage1_dir=root / "stage1",
        stage2_dir=root / "stage2",
        stage3_dir=root / "stage3",
        logs_dir=logs,
        stage1_log=logs / "stage1.log",
        stage2_log=logs / "stage2.log",
        stage3_log=logs / "stage3.log",
    )


def _stage_complete(path: Path, summary_name: str) -> bool:
    payload = load_json_dict(Path(path) / summary_name)
    if not payload or int(payload.get("returncode", -1)) != 0:
        return False
    status = str(payload.get("status", "") or "").strip()
    if status and status != "passed":
        return False
    if summary_name in {"stage1_summary.json", "stage2_summary.json"}:
        if status != "passed":
            return False
        if int(payload.get("forecast_rows", 0) or 0) <= 0:
            return False
        if not str(payload.get("manifest_path", "") or "").strip():
            return False
    return True


def stage1_complete(paths: ModelPaths) -> bool:
    return (
        _stage_complete(paths.stage1_dir, "stage1_summary.json")
        and (paths.stage1_dir / "feature_profile_selection.json").exists()
        and (paths.stage1_dir / "feature_experiment_run_meta.json").exists()
    )


def stage2_complete(paths: ModelPaths) -> bool:
    run_dir = latest_stage2_run_dir(paths)
    if run_dir is None:
        return False
    if not _stage_complete(run_dir, "stage2_summary.json"):
        return False
    return all((run_dir / name).exists() for name in REQUIRED_STAGE2_FILES)


def latest_stage2_run_dir(paths: ModelPaths) -> Optional[Path]:
    candidates = sorted((path.parent for path in paths.stage2_dir.glob("run=*/diagnostic_manifest.json")), key=lambda path: path.name)
    return candidates[-1] if candidates else None


def stage1_selection_path(paths: ModelPaths) -> Path:
    return paths.stage1_dir / "feature_profile_selection.json"


def stage2_manifest_path(paths: ModelPaths) -> Path:
    run_dir = latest_stage2_run_dir(paths)
    return (run_dir / "diagnostic_manifest.json") if run_dir is not None else (paths.stage2_dir / "diagnostic_manifest.json")


def stage2_survivor_path(paths: ModelPaths) -> Path:
    run_dir = latest_stage2_run_dir(paths)
    return (run_dir / "stage3_survivor_handoff.json") if run_dir is not None else (paths.stage2_dir / "stage3_survivor_handoff.json")


def stage3_complete(paths: ModelPaths) -> bool:
    if not (_stage_complete(paths.stage3_dir, "stage3_summary.json") and all((paths.stage3_dir / name).exists() for name in REQUIRED_STAGE3_FILES)):
        return False
    summary = load_json_dict(paths.stage3_dir / "stage3_summary.json")
    if str(summary.get("promotion_decision", "")).strip() == "not_promoted_without_walk_forward_quality":
        return False
    return True


def run_complete(run_root: Path) -> bool:
    if not stage0_complete(run_root):
        return False
    for model_key in MODEL_ORDER:
        paths = model_paths(run_root, model_key)
        if not (stage1_complete(paths) and stage2_complete(paths) and stage3_complete(paths)):
            return False
    return True


def latest_incomplete_run(base_output_dir: Path) -> Optional[Path]:
    if not Path(base_output_dir).exists():
        return None
    candidates = sorted((p for p in Path(base_output_dir).glob("run=*") if p.is_dir()), key=lambda p: p.name)
    for run_root in reversed(candidates):
        if not run_complete(run_root):
            return run_root.resolve()
    return None


def selected_stage0_profile(run_root: Path) -> Dict[str, Any]:
    payload = load_json_dict(stage0_profile_path(run_root))
    selected = payload.get("selected_profile")
    return selected if isinstance(selected, dict) else {}


def selected_stage0_profile_for_model(run_root: Path, model_key: str) -> Dict[str, Any]:
    payload = load_json_dict(stage0_profile_path(run_root))
    by_branch = payload.get("selected_profiles_by_branch")
    if isinstance(by_branch, dict):
        selected = by_branch.get(str(model_key))
        if isinstance(selected, dict):
            return selected
    return selected_stage0_profile(run_root)


def selected_stage0_assets(run_root: Path) -> list[str]:
    payload = load_json_dict(stage0_profile_path(run_root))
    assets = payload.get("assets")
    if not isinstance(assets, list):
        return []
    return [str(asset).strip() for asset in assets if str(asset).strip()]


def _profile_int(run_root: Path, key: str, default: int, *, model_key: Optional[str] = None) -> int:
    profile = selected_stage0_profile_for_model(run_root, str(model_key)) if model_key is not None else selected_stage0_profile(run_root)
    raw = profile.get(key)
    try:
        return max(1, int(raw))
    except Exception:
        return int(default)


def _parse_combo_intervals(raw: str) -> list[int]:
    intervals: list[int] = []
    for token in [part.strip() for part in str(raw).split(",") if part.strip()]:
        try:
            interval, _horizon, _task = token.split(":", 2)
            intervals.append(int(interval))
        except Exception:
            continue
    return sorted(set(intervals))


def _discover_assets_from_parquet_root(parquet_root: Path, intervals: Sequence[int]) -> list[str]:
    assets: set[str] = set()
    root = Path(parquet_root)
    for interval in intervals:
        for table_name in (f"ohlcvt_{int(interval)}", f"scalar_features_{int(interval)}"):
            table_root = root / table_name
            if not table_root.exists():
                continue
            for child in table_root.glob("asset=*"):
                if child.is_dir():
                    asset = child.name.split("=", 1)[1].strip()
                    if asset:
                        assets.add(asset)
    return sorted(assets)


def resolve_stats_cohort_assets(
    *,
    parquet_root: Path,
    intervals: Sequence[int],
    asset_count: int,
    seed: int,
    explicit_assets: Sequence[str] = (),
) -> List[str]:
    requested = [str(asset).strip() for asset in explicit_assets if str(asset).strip()]
    if requested:
        return requested
    interval_values = sorted({int(interval) for interval in intervals if int(interval) > 0})
    if not interval_values:
        return []
    clamp_start = MonthKey(CLAMP_START_YEAR, CLAMP_START_MONTH)
    eligible_sets: List[set[str]] = []
    for interval in interval_values:
        try:
            _end_month, eligible_assets = common_recent_window(
                ohlc_root=Path(parquet_root) / f"ohlcvt_{int(interval)}",
                scalar_root=Path(parquet_root) / f"scalar_features_{int(interval)}",
                min_assets=1,
                window_months=DEFAULT_COHORT_WINDOW_MONTHS,
                search_back_months=DEFAULT_SEARCH_BACK_MONTHS,
                clamp_start=clamp_start,
            )
        except Exception:
            return []
        eligible_sets.append({str(asset) for asset in eligible_assets})
    common_assets = sorted(set.intersection(*eligible_sets)) if eligible_sets else []
    if not common_assets:
        return []
    selected_assets, _alias_map = select_representative_assets(
        common_assets,
        seed=int(seed),
        asset_count=int(asset_count),
        required_symbols=FIXED_NUMERIC_FAMILY_COHORT_SYMBOLS,
    )
    return [str(asset) for asset in selected_assets]


def _resolve_stage1_assets(args: argparse.Namespace) -> List[str]:
    explicit = [part.strip() for part in str(args.assets).split(",") if part.strip()]
    if explicit:
        return explicit
    intervals = _parse_combo_intervals(str(args.combo_list))
    return resolve_stats_cohort_assets(
        parquet_root=Path(args.parquet_root),
        intervals=intervals,
        asset_count=int(args.stage1_asset_count),
        seed=int(args.stage1_seed),
    )


def _resolve_artifact_period_bars(args: argparse.Namespace, model_key: str, *, run_root: Optional[Path] = None) -> Optional[int]:
    if int(args.artifact_period_bars) > 1:
        return int(args.artifact_period_bars)
    if bool(args.disable_seasonality_artifact_period):
        return None
    if str(model_key) not in {"sarimax", "llt"}:
        return None
    intervals = _parse_combo_intervals(str(args.combo_list))
    if not intervals:
        return None
    assets_arg = str(args.assets).strip()
    if assets_arg:
        assets = [part.strip() for part in assets_arg.split(",") if part.strip()]
    elif run_root is not None:
        profile = load_json_dict(stage1_selection_path(model_paths(Path(run_root), str(model_key))))
        assets = [str(asset) for asset in list(profile.get("cohort_assets") or []) if str(asset)]
    else:
        assets = _resolve_stage1_assets(args)
    if not assets:
        return None
    assets = assets[: max(1, int(args.seasonality_period_max_assets))]
    counts: Dict[int, int] = {}
    for interval in intervals:
        for asset in assets:
            try:
                profile = resolve_seasonality_profile(parquet_root=Path(args.parquet_root), interval_minutes=int(interval), asset=str(asset))
            except Exception:
                continue
            if profile.usable and profile.seasonal_period_bars is not None and int(profile.seasonal_period_bars) > 1:
                counts[int(profile.seasonal_period_bars)] = int(counts.get(int(profile.seasonal_period_bars), 0)) + 1
    if not counts:
        return None
    return sorted(counts.items(), key=lambda item: (-int(item[1]), int(item[0])))[0][0]


def process_meta_path(log_path: Path) -> Path:
    return log_path.with_suffix(".process.json")


def subprocess_env(base_env: Dict[str, str], *, paths: ModelPaths, stage_name: str) -> Dict[str, str]:
    env = dict(base_env)
    temp_root = test_branch_stage_tmp_root(
        env,
        family_key="stats",
        run_name=paths.root.parent.name,
        model_key=paths.root.name,
        stage_name=str(stage_name),
        fallback_root=(paths.root / "tmp" / str(stage_name)).resolve(),
    )
    env["TMP"] = str(temp_root)
    env["TEMP"] = str(temp_root)
    env["TMPDIR"] = str(temp_root)
    env.setdefault("PYTHONFAULTHANDLER", "1")
    env.setdefault("PYTHONUNBUFFERED", "1")
    return env


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
) -> None:
    write_json_atomic(
        meta_path,
        {
            "status": str(status),
            "command": [str(part) for part in command],
            "cwd": str(cwd),
            "log_path": str(log_path),
            "pid": pid,
            "started_utc": started_utc,
            "finished_utc": finished_utc,
            "returncode": returncode,
            "error": error,
            "env_hints": {
                "TMP": env.get("TMP"),
                "TEMP": env.get("TEMP"),
                "TMPDIR": env.get("TMPDIR"),
                "PYTHONFAULTHANDLER": env.get("PYTHONFAULTHANDLER"),
                "PYTHONUNBUFFERED": env.get("PYTHONUNBUFFERED"),
                "OMP_NUM_THREADS": env.get("OMP_NUM_THREADS"),
                "MKL_NUM_THREADS": env.get("MKL_NUM_THREADS"),
                "OPENBLAS_NUM_THREADS": env.get("OPENBLAS_NUM_THREADS"),
            },
        },
    )


def run_logged_subprocess(command: Sequence[str], *, cwd: Path, env: Dict[str, str], log_path: Path, progress_name: str = "") -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    last_progress = started
    started_utc = utc_now_iso()
    meta_path = process_meta_path(log_path)
    with log_path.open("w", encoding="utf-8") as handle:
        proc = subprocess.Popen(list(command), cwd=str(cwd), env=env, stdout=handle, stderr=subprocess.STDOUT)
        write_process_meta(meta_path, status="running", command=command, cwd=cwd, env=env, log_path=log_path, pid=int(proc.pid), started_utc=started_utc)
        while proc.poll() is None:
            time.sleep(1.0)
            now = time.perf_counter()
            if now - last_progress >= 30.0:
                label = str(progress_name).strip() or str(command[2] if len(command) > 2 else "subprocess")
                progress_line(f"{label}: still running elapsed_s={now - started:.0f} log={log_path}")
                last_progress = now
        returncode = int(proc.wait())
    if int(returncode) != 0:
        error = f"Command failed with exit code {returncode}: {' '.join(str(part) for part in command)}"
        write_process_meta(meta_path, status="failed", command=command, cwd=cwd, env=env, log_path=log_path, pid=int(proc.pid), started_utc=started_utc, finished_utc=utc_now_iso(), returncode=returncode, error=error)
        exc = RuntimeError(error)
        emit_subprocess_event(
            log_path=log_path,
            command=command,
            status="failed",
            family="Stats_Frequentist",
            elapsed_seconds=time.perf_counter() - started,
            reason_code="exception",
            exception=exc,
        )
        raise exc
    write_process_meta(meta_path, status="completed", command=command, cwd=cwd, env=env, log_path=log_path, pid=int(proc.pid), started_utc=started_utc, finished_utc=utc_now_iso(), returncode=returncode)
    emit_subprocess_event(
        log_path=log_path,
        command=command,
        status="completed",
        family="Stats_Frequentist",
        elapsed_seconds=time.perf_counter() - started,
    )


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
    for attempt in range(1, max(1, int(max_attempts)) + 1):
        progress_line(f"{progress_name}: attempt {attempt}/{max(1, int(max_attempts))}")
        try:
            run_logged_subprocess(command, cwd=cwd, env=env, log_path=log_path, progress_name=progress_name)
            progress_line(f"{progress_name}: success")
            return
        except Exception as exc:
            last_error = exc
            progress_line(f"{progress_name}: failed on attempt {attempt}/{max(1, int(max_attempts))}: {exc}")
            if attempt >= max(1, int(max_attempts)):
                break
            for cleanup_path in cleanup_paths or []:
                assert_write_allowed(cleanup_path, "stats test branch cleanup")
                shutil.rmtree(cleanup_path, ignore_errors=True)
            progress_line(f"{progress_name}: retrying after {float(retry_delay_seconds):.1f}s")
            time.sleep(max(0.0, float(retry_delay_seconds)))
    if last_error is not None:
        raise last_error
    raise RuntimeError("retry loop exited without executing command")


def _run_stage0(args: argparse.Namespace, run_root: Path, env: Dict[str, str]) -> None:
    out_dir = stage0_dir(run_root)
    command = [
        sys.executable,
        "-m",
        STATS_STAGE0_ENTRYPOINT,
        "--project-root",
        str(args.project_root),
        "--parquet-root",
        str(args.parquet_root),
        "--output-dir",
        str(out_dir),
    ]
    stage0_combo_list = str(args.stage0_combo_list).strip() or str(args.combo_list)
    if stage0_combo_list:
        command.extend(["--combo-list", stage0_combo_list])
    if str(args.stage0_workers).strip():
        command.extend(["--workers", str(args.stage0_workers)])
    if str(args.stage0_threads).strip():
        command.extend(["--threads", str(args.stage0_threads)])
    if str(args.stage0_assets).strip():
        command.extend(["--assets", str(args.stage0_assets)])
    if args.stage0_backfill_days is not None:
        command.extend(["--backfill-days", str(int(args.stage0_backfill_days))])
    if args.stage0_fit_days is not None:
        command.extend(["--fit-days", str(int(args.stage0_fit_days))])
    if args.stage0_sample_seconds is not None:
        command.extend(["--sample-seconds", str(float(args.stage0_sample_seconds))])
    if args.stage0_ram_cap_pct is not None:
        command.extend(["--ram-cap-pct", str(float(args.stage0_ram_cap_pct))])
    if bool(args.stage0_prune_run_artifacts):
        command.append("--prune-run-artifacts")
    progress_line("Stats Family Stage 0: starting")
    stage_env = dict(env)
    temp_root = test_branch_stage_tmp_root(
        stage_env,
        family_key="stats",
        run_name=run_root.name,
        model_key="stage0",
        stage_name="stage0",
        fallback_root=(Path(run_root) / "tmp" / "stage0").resolve(),
    )
    stage_env["TMP"] = str(temp_root)
    stage_env["TEMP"] = str(temp_root)
    stage_env["TMPDIR"] = str(temp_root)
    run_with_retries(
        command,
        cwd=Path(args.project_root),
        env=stage_env,
        log_path=out_dir / "stage0.log",
        max_attempts=int(args.max_attempts),
        retry_delay_seconds=float(args.retry_delay_seconds),
        cleanup_paths=[out_dir],
        progress_name="Stats Family Stage 0",
    )
    if not stage0_complete(run_root):
        raise RuntimeError("Stage 0 did not leave complete profile artifacts for stats family")
    emit_event(run_root, run_id=run_root.name, family="Stats_Frequentist", model="family", stage="stage0", function_name="_run_stage0", module_name=STATS_STAGE0_ENTRYPOINT, phase_name="artifact_handoff", status="completed", output_path=str(stage0_profile_path(run_root)), artifact_profile_source=str(stage0_profile_path(run_root)))


def _stage_command(
    args: argparse.Namespace,
    module: str,
    output_dir: Path,
    workers: int,
    backfill_days: int,
    model_threads: int,
    predict_latest_only: bool,
    *,
    model_key: str,
    stage_name: str,
    run_root: Optional[Path] = None,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        module,
        "--project-root",
        str(args.project_root),
        "--parquet-root",
        str(args.parquet_root),
        "--output-dir",
        str(output_dir),
        "--workers",
        str(int(workers)),
        "--backfill-days",
        str(int(backfill_days)),
        "--fit-days",
        str(int(args.fit_days)),
        "--model-threads",
        str(int(model_threads)),
        "--force",
    ]
    if not (str(stage_name) == "stage3" and run_root is not None):
        command.extend(["--combo-list", str(args.combo_list)])
    assets = str(args.assets).strip()
    if not assets and str(stage_name) == "stage1":
        assets = ",".join(_resolve_stage1_assets(args))
    if assets:
        command.extend(["--assets", assets])
    if predict_latest_only:
        command.append("--predict-latest-only")
    if str(stage_name) == "stage3":
        artifact_period = _resolve_artifact_period_bars(args, model_key, run_root=run_root)
        if artifact_period is not None and int(artifact_period) > 1:
            command.extend(["--artifact-period-bars", str(int(artifact_period))])
    if str(stage_name) == "stage2":
        command.append("--staged")
        if run_root is not None:
            command.extend(["--feature-profile-json", str(stage1_selection_path(model_paths(Path(run_root), model_key)))])
    if str(stage_name) == "stage3" and run_root is not None:
        paths = model_paths(Path(run_root), model_key)
        command.append("--staged")
        command.extend(["--stage2-manifest", str(stage2_manifest_path(paths))])
        command.extend(["--stage2-survivor-json", str(stage2_survivor_path(paths))])
    return command


def _run_model_stage(args: argparse.Namespace, run_root: Path, model_key: str, stage_name: str, module: str, output_dir: Path, log_path: Path, env: Optional[Dict[str, str]] = None) -> None:
    profile_workers = _profile_int(run_root, "workers", 1, model_key=model_key)
    profile_threads = _profile_int(run_root, "threads", 1, model_key=model_key)
    if stage_name == "stage1":
        workers = profile_workers if args.stage1_workers is None else min(int(args.stage1_workers), profile_workers)
        backfill_days = int(args.stage1_backfill_days)
        predict_latest_only = False
    elif stage_name == "stage2":
        workers = profile_workers if args.stage2_workers is None else min(int(args.stage2_workers), profile_workers)
        backfill_days = int(args.stage2_backfill_days)
        predict_latest_only = False
    else:
        workers = profile_workers if args.stage3_workers is None else min(int(args.stage3_workers), profile_workers)
        backfill_days = int(args.stage3_backfill_days)
        predict_latest_only = True
    actual_output_dir = Path(output_dir)
    if str(stage_name) == "stage2" and not actual_output_dir.name.startswith("run="):
        actual_output_dir = actual_output_dir / f"run={utc_now_stamp()}"
    command = _stage_command(
        args,
        module,
        actual_output_dir,
        workers=max(1, int(workers)),
        backfill_days=backfill_days,
        model_threads=profile_threads,
        predict_latest_only=predict_latest_only,
        model_key=model_key,
        stage_name=stage_name,
        run_root=run_root,
    )
    progress_line(f"{MODEL_SPECS[model_key].display_name} {stage_name.title()}: starting")
    paths = model_paths(run_root, model_key)
    stage_env = subprocess_env(env if env is not None else test_branch_child_env(args, dict(os.environ)), paths=paths, stage_name=stage_name)
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        stage_env[name] = str(max(1, int(profile_threads)))
    stage_env["STATS_NUMERIC_MODEL_THREADS"] = str(max(1, int(profile_threads)))
    cleanup_paths = [actual_output_dir] if str(stage_name) in {"stage1", "stage3"} else None
    run_with_retries(
        command,
        cwd=Path(args.project_root),
        env=stage_env,
        log_path=log_path,
        max_attempts=int(args.max_attempts),
        retry_delay_seconds=float(args.retry_delay_seconds),
        cleanup_paths=cleanup_paths,
        progress_name=f"{MODEL_SPECS[model_key].display_name} {stage_name.title()}",
    )
    complete_fn = {"stage1": stage1_complete, "stage2": stage2_complete, "stage3": stage3_complete}[str(stage_name)]
    if not complete_fn(model_paths(run_root, model_key)):
        emit_event(run_root, run_id=run_root.name, family="Stats_Frequentist", model=model_key, stage=stage_name, function_name="_run_model_stage", module_name=module, phase_name="artifact_handoff", status="failed", reason_code="stage_artifact_missing", output_path=str(actual_output_dir))
        raise RuntimeError(f"{MODEL_SPECS[model_key].display_name} {stage_name.title()} did not produce complete stage artifacts")
    emit_event(run_root, run_id=run_root.name, family="Stats_Frequentist", model=model_key, stage=stage_name, function_name="_run_model_stage", module_name=module, phase_name="artifact_handoff", status="completed", output_path=str(actual_output_dir))
    progress_line(f"{MODEL_SPECS[model_key].display_name} {stage_name.title()}: success")


def write_state(run_root: Path, args: argparse.Namespace | None = None) -> None:
    stage0_status = {
        "complete": stage0_complete(run_root),
        "profile_path": str(stage0_profile_path(run_root)),
        "log_path": str(stage0_dir(run_root) / "stage0.log"),
        "selected_profile": selected_stage0_profile(run_root),
    }
    model_statuses: Dict[str, Any] = {}
    for model_key in MODEL_ORDER:
        paths = model_paths(run_root, model_key)
        stage3_artifacts = {
            name: str((paths.stage3_dir / name).resolve())
            for name in REQUIRED_STAGE3_FILES
            if (paths.stage3_dir / name).exists()
        }
        model_statuses[model_key] = {
            "stage1": {
                "complete": stage1_complete(paths),
                "path": str(paths.stage1_dir),
                "log_path": str(paths.stage1_log),
                "feature_profile_json": str(stage1_selection_path(paths)) if stage1_selection_path(paths).exists() else None,
            },
            "stage2": {
                "complete": stage2_complete(paths),
                "path": str(latest_stage2_run_dir(paths) or paths.stage2_dir),
                "log_path": str(paths.stage2_log),
                "diagnostic_manifest_json": str(stage2_manifest_path(paths)) if stage2_manifest_path(paths).exists() else None,
                "stage3_survivor_handoff_json": str(stage2_survivor_path(paths)) if stage2_survivor_path(paths).exists() else None,
            },
            "stage3": {
                "complete": stage3_complete(paths),
                "path": str(paths.stage3_dir),
                "log_path": str(paths.stage3_log),
                "artifacts": stage3_artifacts,
            },
        }
    health = write_test_branch_health(
        Path(run_root),
        family="Stats_Frequentist",
        stage0_status=stage0_status,
        models=model_statuses,
    )
    payload: Dict[str, Any] = {
        "family": "Stats_Frequentist",
        "complete": run_complete(run_root),
        "stage0": stage0_status,
        "models": model_statuses,
        "health": {
            "status": health.get("status"),
            "error_count": health.get("error_count"),
            "warning_count": health.get("warning_count"),
            "health_report_json": str((Path(run_root) / HEALTH_REPORT_FILE).resolve()),
        },
        "updated_at": utc_now_iso(),
    }
    if bool(payload["complete"]) and args is not None:
        payload["canonical_profiles"] = publish_canonical_family_profiles(
            args,
            run_root=run_root,
            diagnostics_root_name=DEFAULT_OUTPUT_DIR.name,
            model_order=MODEL_ORDER,
            model_paths_fn=model_paths,
            required_stage3_files=REQUIRED_STAGE3_FILES,
            family="Stats_Frequentist",
        )
    write_json_atomic(Path(run_root) / RUN_STATE_FILE, payload)


def write_summary(run_root: Path) -> None:
    outputs: Dict[str, Any] = {}
    for model_key in MODEL_ORDER:
        paths = model_paths(run_root, model_key)
        stage2_dir_for_summary = latest_stage2_run_dir(paths) or paths.stage2_dir
        outputs[model_key] = {
            "stage1": load_json_dict(paths.stage1_dir / "stage1_summary.json"),
            "stage2": load_json_dict(stage2_dir_for_summary / "stage2_summary.json"),
            "stage3": load_json_dict(paths.stage3_dir / "stage3_summary.json"),
        }
    health = load_json_dict(Path(run_root) / HEALTH_REPORT_FILE)
    write_json_atomic(
        Path(run_root) / RUN_SUMMARY_FILE,
        {
            "family": "Stats_Frequentist",
            "stage0_profile": selected_stage0_profile(run_root),
            "health": {
                "status": health.get("status"),
                "error_count": health.get("error_count"),
                "warning_count": health.get("warning_count"),
                "health_report_json": str((Path(run_root) / HEALTH_REPORT_FILE).resolve()),
            },
            "stage_outputs": outputs,
            "finished_at": utc_now_iso(),
        },
    )


def run_orchestrator(args: argparse.Namespace) -> Path:
    run_root = resolve_run_root(args)
    run_root.mkdir(parents=True, exist_ok=True)
    env = test_branch_child_env(args, dict(os.environ))
    assert_test_branch_sandbox_launch(args, run_root, env, family_key="stats")
    if run_root.exists() and any(run_root.iterdir()):
        progress_line(f"Continuing stats orchestrator run: {run_root}")
    else:
        progress_line(f"Starting stats orchestrator run: {run_root}")

    if stage0_complete(run_root):
        progress_line("Stats Family Stage 0: already complete, skipping")
        emit_event(run_root, run_id=run_root.name, family="Stats_Frequentist", model="family", stage="stage0", function_name="_run_stage0", module_name=STATS_STAGE0_ENTRYPOINT, phase_name="artifact_handoff", status="skipped", reason_code="stage_already_complete", output_path=str(stage0_profile_path(run_root)), artifact_profile_source=str(stage0_profile_path(run_root)))
    else:
        _run_stage0(args, run_root, env)
    write_state(run_root, args)

    for idx, model_key in enumerate(MODEL_ORDER, start=1):
        spec = MODEL_SPECS[model_key]
        paths = model_paths(run_root, model_key)
        progress_line(f"[{idx}/{len(MODEL_ORDER)}] {spec.display_name}: entering family partition")
        if stage1_complete(paths):
            progress_line(f"{spec.display_name} Stage 1: already complete, skipping")
            emit_event(run_root, run_id=run_root.name, family="Stats_Frequentist", model=model_key, stage="stage1", function_name="_run_model_stage", module_name=spec.stage1_module, phase_name="artifact_handoff", status="skipped", reason_code="stage_already_complete", output_path=str(paths.stage1_dir))
        else:
            _run_model_stage(args, run_root, model_key, "stage1", spec.stage1_module, paths.stage1_dir, paths.stage1_log, env)
        write_state(run_root)
        if stage2_complete(paths):
            progress_line(f"{spec.display_name} Stage 2: already complete, skipping")
            emit_event(run_root, run_id=run_root.name, family="Stats_Frequentist", model=model_key, stage="stage2", function_name="_run_model_stage", module_name=spec.stage2_module, phase_name="artifact_handoff", status="skipped", reason_code="stage_already_complete", output_path=str(stage2_survivor_path(paths)), artifact_profile_source=str(stage2_manifest_path(paths)))
        else:
            _run_model_stage(args, run_root, model_key, "stage2", spec.stage2_module, paths.stage2_dir, paths.stage2_log, env)
        write_state(run_root)
        if stage3_complete(paths):
            progress_line(f"{spec.display_name} Stage 3: already complete, skipping")
            emit_event(run_root, run_id=run_root.name, family="Stats_Frequentist", model=model_key, stage="stage3", function_name="_run_model_stage", module_name=spec.stage3_module, phase_name="artifact_handoff", status="skipped", reason_code="stage_already_complete", output_path=str(paths.stage3_dir))
        else:
            _run_model_stage(args, run_root, model_key, "stage3", spec.stage3_module, paths.stage3_dir, paths.stage3_log, env)
        write_state(run_root)

    write_summary(run_root)
    write_state(run_root, args)
    progress_line("Stats orchestrator run complete")
    return run_root.resolve()


def main(argv: Optional[Sequence[str]] = None) -> None:
    run_orchestrator(parse_args(argv))


if __name__ == "__main__":
    main()
