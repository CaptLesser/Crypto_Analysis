from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence

from src.forecasting.common.path_config import resolve_path, selected_profile
from src.forecasting.common.runtime_config import RUNTIME_CONFIG_PATH, load_runtime_config
from src.forecasting.ml.shared.stage0_profile_common import (
    candidate_key,
    dir_size_bytes,
    measure_branch_run as _shared_measure_branch_run,
    parse_int_csv as _parse_int_csv,
    parse_str_csv as _parse_str_csv,
    process_tree_cpu_percent as _process_tree_cpu_percent,
    process_tree_rss_bytes as _process_tree_rss_bytes,
    process_tree_write_bytes as _process_tree_write_bytes,
    resolve_combo_list as _resolve_combo_list,
    timestamp_utc as _timestamp,
    write_candidate_csv as _shared_write_candidate_csv,
)
from src.forecasting.ml.neural.shared.neural_numeric_cohort import FIXED_NEURAL_NUMERIC_COHORT, resolve_neural_cohort_assets
from src.forecasting.ml.neural.shared.neural_numeric_model_registry import (
    NEURAL_NUMERIC_BRANCHES,
    NEURAL_NUMERIC_ENTRYPOINTS,
    NEURAL_NUMERIC_FAMILY_ROOT_ENVS,
    NEURAL_NUMERIC_FAMILY_ROOT_NAMES,
)
from src.forecasting.ml.neural.shared.neural_runtime_bootstrap import neural_thread_env

try:
    import psutil  # type: ignore
except Exception:  # pragma: no cover
    psutil = None


SATURATION_SELECTION_TOLERANCE_RATIO = 0.03
_candidate_key = candidate_key


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Neural numerics Stage 0 worker profile sweep")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--profile", type=str, default=selected_profile(default="pipeline_test"))
    parser.add_argument("--parquet-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--task", type=str, default="")
    parser.add_argument("--interval", type=int, default=0)
    parser.add_argument("--horizon-minutes", type=int, default=0)
    parser.add_argument("--tasks", type=str, default="log_return,realized_vol")
    parser.add_argument("--intervals", type=str, default="30,60")
    parser.add_argument("--horizons", type=str, default="240,1440")
    parser.add_argument("--workers", type=str, default="6,8,10,12,14")
    parser.add_argument("--threads", type=str, default="4,6,8")
    parser.add_argument("--assets", type=str, default="")
    parser.add_argument("--backfill-days", type=int, default=30)
    parser.add_argument("--fit-days", type=int, default=180)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--sample-seconds", type=float, default=1.0)
    parser.add_argument("--ram-cap-pct", type=float, default=80.0)
    parser.add_argument("--prune-run-artifacts", action="store_true")
    args = parser.parse_args()
    if args.parquet_root is None:
        args.parquet_root = Path(resolve_path("source_ohlcvt_root", profile=str(args.profile), required=False) or Path("parquet"))
    return args

def _default_output_dir(project_root: Path) -> Path:
    return project_root / "logs" / "diagnostics" / "neural_numeric_stage0_profile" / f"run={_timestamp()}"


def _measure_branch_run(*, command: Sequence[str], env: Dict[str, str], cwd: Path, log_path: Path, output_dir: Path, sample_seconds: float) -> Dict[str, Any]:
    return _shared_measure_branch_run(
        command=command,
        env=env,
        cwd=cwd,
        log_path=log_path,
        output_dir=output_dir,
        sample_seconds=sample_seconds,
        psutil_module=psutil,
    )


def _run_candidate(*, args: argparse.Namespace, workers: int, threads: int, output_root: Path) -> Dict[str, Any]:
    candidate_root = output_root / _candidate_key(workers, threads)
    candidate_root.mkdir(parents=True, exist_ok=True)
    combo_list = _resolve_combo_list(args)
    resolved_assets = resolve_neural_cohort_assets(
        parquet_root=Path(args.parquet_root).resolve(),
        intervals=sorted({int(interval) for interval, _, _ in combo_list}),
        asset_count=len(_parse_str_csv(str(args.assets))) if str(args.assets).strip() else len(FIXED_NEURAL_NUMERIC_COHORT),
        explicit_assets=_parse_str_csv(str(args.assets)),
        seed=17,
    )
    assets_csv = ",".join(resolved_assets)
    branch_results: List[Dict[str, Any]] = []
    candidate_start = time.perf_counter()
    combo_list_arg = ",".join(f"{int(interval)}:{int(horizon)}:{str(task)}" for interval, horizon, task in combo_list)
    peak_system_ram_pct = 0.0
    peak_process_rss_bytes = 0
    peak_cpu_percent = 0.0
    total_write_bytes = 0
    total_output_bytes = 0

    for model_key in NEURAL_NUMERIC_BRANCHES:
        entrypoint = NEURAL_NUMERIC_ENTRYPOINTS[model_key]
        branch_parquet_root = candidate_root / "parquet" / model_key
        branch_output_dir = branch_parquet_root / NEURAL_NUMERIC_FAMILY_ROOT_NAMES[model_key]
        command = [
            sys.executable,
            "-m",
            entrypoint,
            "--parquet-root",
            str(Path(args.parquet_root).resolve()),
            "--combo-list",
            combo_list_arg,
            "--assets",
            assets_csv,
            "--workers",
            str(int(workers)),
            "--mode",
            "backfill",
            "--backfill_days",
            str(int(args.backfill_days)),
            "--fit_days",
            str(int(args.fit_days)),
            "--seq_len",
            str(int(args.seq_len)),
            "--batch_size",
            str(int(args.batch_size)),
            "--epochs",
            str(int(args.epochs)),
            "--force",
        ]
        env = dict(os.environ)
        env.update(neural_thread_env(int(threads)))
        env[NEURAL_NUMERIC_FAMILY_ROOT_ENVS[model_key]] = str(branch_parquet_root)
        env["RUN_ID"] = _candidate_key(workers, threads)
        branch_metrics = _measure_branch_run(
            command=command,
            env=env,
            cwd=Path(args.project_root),
            log_path=candidate_root / "logs" / f"{model_key}.log",
            output_dir=branch_output_dir,
            sample_seconds=float(args.sample_seconds),
        )
        branch_metrics["model_key"] = str(model_key)
        branch_results.append(branch_metrics)
        peak_system_ram_pct = max(peak_system_ram_pct, float(branch_metrics["peak_system_ram_pct"]))
        peak_process_rss_bytes = max(peak_process_rss_bytes, int(branch_metrics["peak_process_rss_bytes"]))
        peak_cpu_percent = max(peak_cpu_percent, float(branch_metrics["peak_cpu_percent"]))
        total_write_bytes += int(branch_metrics["process_write_bytes"])
        total_output_bytes += int(branch_metrics["output_bytes"])
        if bool(args.prune_run_artifacts):
            shutil.rmtree(branch_output_dir, ignore_errors=True)

    total_wall_seconds = float(time.perf_counter() - candidate_start)
    return {
        "workers": int(workers),
        "threads": int(threads),
        "candidate_key": _candidate_key(workers, threads),
        "task": (str(args.task) if str(args.task).strip() else ",".join(sorted({task for _, _, task in combo_list}))),
        "interval_minutes": (int(args.interval) if int(getattr(args, "interval", 0)) > 0 else min(interval for interval, _, _ in combo_list)),
        "horizon_minutes": (int(args.horizon_minutes) if int(getattr(args, "horizon_minutes", 0)) > 0 else max(horizon for _, horizon, _ in combo_list)),
        "combo_count": int(len(combo_list)),
        "combo_list": [{"interval_minutes": int(interval), "horizon_minutes": int(horizon), "task": str(task)} for interval, horizon, task in combo_list],
        "assets": list(resolved_assets),
        "total_wall_seconds": total_wall_seconds,
        "throughput_score": float((len(combo_list) * max(1, len([asset for asset in str(args.assets).split(',') if asset])) * max(1, len(NEURAL_NUMERIC_BRANCHES))) / max(total_wall_seconds, 1e-9)),
        "peak_system_ram_pct": peak_system_ram_pct,
        "peak_process_rss_bytes": int(peak_process_rss_bytes),
        "peak_cpu_percent": peak_cpu_percent,
        "total_process_write_bytes": int(total_write_bytes),
        "total_output_bytes": int(total_output_bytes),
        "branch_results": branch_results,
    }


def _select_best_candidate(results: Sequence[Dict[str, Any]], ram_cap_pct: float) -> Dict[str, Any]:
    within_cap = [result for result in results if float(result.get("peak_system_ram_pct", 0.0)) <= float(ram_cap_pct)]
    pool = within_cap if within_cap else list(results)
    best_wall = min(float(result.get("total_wall_seconds", 1e18)) for result in pool)
    near_best = [
        result
        for result in pool
        if float(result.get("total_wall_seconds", 1e18)) <= best_wall * (1.0 + SATURATION_SELECTION_TOLERANCE_RATIO)
    ]
    selection_pool = near_best if near_best else pool
    return max(
        selection_pool,
        key=lambda result: (
            float(result.get("throughput_score", 0.0)),
            int(result.get("workers", 0)),
            int(result.get("threads", 0)),
            float(result.get("peak_cpu_percent", 0.0)),
            -float(result.get("total_wall_seconds", 1e18)),
            -float(result.get("peak_system_ram_pct", 1e18)),
        ),
    )


def _write_candidate_csv(path: Path, results: Sequence[Dict[str, Any]]) -> None:
    fieldnames = [
        "candidate_key",
        "workers",
        "threads",
        "interval_minutes",
        "horizon_minutes",
        "task",
        "combo_count",
        "total_wall_seconds",
        "throughput_score",
        "peak_system_ram_pct",
        "peak_process_rss_bytes",
        "peak_cpu_percent",
        "total_process_write_bytes",
        "total_output_bytes",
    ]
    _shared_write_candidate_csv(path, results, fieldnames=fieldnames)


def _update_runtime_profile(workers: int, threads: int) -> None:
    cfg = load_runtime_config()
    modules = cfg.setdefault("modules", {})
    module_cfg = modules.setdefault("neural_numeric_runner", {})
    module_cfg["asset_workers"] = int(workers)
    module_cfg["model_threads"] = int(threads)
    RUNTIME_CONFIG_PATH.write_text(json.dumps(cfg, indent=2), encoding="utf-8")


def run_stage0(args: argparse.Namespace) -> Path:
    output_dir = Path(args.output_dir) if args.output_dir is not None else _default_output_dir(Path(args.project_root))
    output_dir.mkdir(parents=True, exist_ok=True)
    worker_values = _parse_int_csv(args.workers)
    thread_values = _parse_int_csv(args.threads)
    results: List[Dict[str, Any]] = []

    for workers in worker_values:
        for threads in thread_values:
            print(f"[stage0] profiling workers={workers} threads={threads}")
            result = _run_candidate(args=args, workers=int(workers), threads=int(threads), output_root=output_dir)
            results.append(result)

    selected = _select_best_candidate(results, float(args.ram_cap_pct))
    _update_runtime_profile(int(selected["workers"]), int(selected["threads"]))

    artifact = {
        "stage": 0,
        "family": "neural_numeric",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "task": (str(args.task) if str(args.task).strip() else None),
        "interval_minutes": (int(args.interval) if int(getattr(args, "interval", 0)) > 0 else None),
        "horizon_minutes": (int(args.horizon_minutes) if int(getattr(args, "horizon_minutes", 0)) > 0 else None),
        "tasks": sorted({task for _, _, task in _resolve_combo_list(args)}),
        "intervals": sorted({int(interval) for interval, _, _ in _resolve_combo_list(args)}),
        "horizons": sorted({int(horizon) for _, horizon, _ in _resolve_combo_list(args)}),
        "combo_count": int(len(_resolve_combo_list(args))),
        "assets": list(selected.get("assets") or []),
        "ram_cap_pct": float(args.ram_cap_pct),
        "candidates": results,
        "selected_profile": {
            "asset_workers": int(selected["workers"]),
            "model_threads": int(selected["threads"]),
            "candidate_key": str(selected["candidate_key"]),
            "total_wall_seconds": float(selected["total_wall_seconds"]),
            "peak_system_ram_pct": float(selected["peak_system_ram_pct"]),
        },
        "runtime_config_path": str(RUNTIME_CONFIG_PATH),
    }
    (output_dir / "neural_numeric_stage0_profile.json").write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    _write_candidate_csv(output_dir / "neural_numeric_stage0_candidates.csv", results)
    return output_dir


def main() -> None:
    args = parse_args()
    run_root = run_stage0(args)
    print(f"[stage0] completed {run_root}")


if __name__ == "__main__":
    main()
