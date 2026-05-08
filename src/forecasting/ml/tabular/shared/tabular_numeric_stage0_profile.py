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

from src.forecasting.common.path_config import PathConfigError, require_pipeline_io
from src.forecasting.common.runtime_config import RUNTIME_CONFIG_PATH, load_runtime_config
from src.forecasting.ml.shared.stage0_profile_common import (
    candidate_key,
    dir_size_bytes,
    measure_branch_run as _shared_measure_branch_run,
    parse_int_csv as _parse_int_csv,
    process_tree_cpu_percent as _process_tree_cpu_percent,
    process_tree_rss_bytes as _process_tree_rss_bytes,
    process_tree_write_bytes as _process_tree_write_bytes,
    timestamp_utc as _timestamp,
    write_candidate_csv as _shared_write_candidate_csv,
)
from src.forecasting.ml.shared.numeric_cohort_common import FIXED_NUMERIC_FAMILY_COHORT_SYMBOLS

try:
    import psutil  # type: ignore
except Exception:  # pragma: no cover
    psutil = None


TABULAR_STAGE0_BRANCH_SPECS = (
    {
        "model_key": "xgboost",
        "entrypoint": "src.forecasting.ml.tabular.xgboost.numerics",
        "module_slug": "xgboost_numerics",
        "root_env": "PIPELINE_PARQUET_XGB_NUMERICS_ROOT",
    },
    {
        "model_key": "lightgbm",
        "entrypoint": "src.forecasting.ml.tabular.lightgbm.numerics",
        "module_slug": "lightgbm_numerics",
        "root_env": "PIPELINE_PARQUET_LGB_NUMERICS_ROOT",
    },
    {
        "model_key": "catboost",
        "entrypoint": "src.forecasting.ml.tabular.catboost.numerics",
        "module_slug": "catboost_numerics",
        "root_env": "PIPELINE_PARQUET_CB_NUMERICS_ROOT",
    },
    {
        "model_key": "random_forest",
        "entrypoint": "src.forecasting.ml.tabular.random_forest.numerics",
        "module_slug": "random_forest_numerics",
        "root_env": "PIPELINE_PARQUET_RF_NUMERICS_ROOT",
    },
    {
        "model_key": "elasticnet",
        "entrypoint": "src.forecasting.ml.tabular.elasticnet.numerics",
        "module_slug": "elasticnet_numerics",
        "root_env": "PIPELINE_PARQUET_EN_NUMERICS_ROOT",
    },
)
_candidate_key = candidate_key


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tabular numerics Stage 0 worker profile sweep")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--profile", type=str, default="pipeline_test")
    parser.add_argument("--parquet-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--task", type=str, default="log_return")
    parser.add_argument("--interval", type=int, default=60)
    parser.add_argument("--horizon-minutes", type=int, default=60)
    parser.add_argument("--workers", type=str, default="6,8,10,12,14")
    parser.add_argument("--threads", type=str, default="4,6,8")
    parser.add_argument("--assets", type=str, default=",".join(FIXED_NUMERIC_FAMILY_COHORT_SYMBOLS))
    parser.add_argument("--backfill-days", type=int, default=30)
    parser.add_argument("--sample-seconds", type=float, default=1.0)
    parser.add_argument("--ram-cap-pct", type=float, default=80.0)
    parser.add_argument("--prune-run-artifacts", action="store_true")
    args = parser.parse_args()
    if args.parquet_root is None:
        try:
            args.parquet_root = require_pipeline_io(profile=str(args.profile or "pipeline_test")).source_ohlcvt_root
        except PathConfigError:
            args.parquet_root = Path("parquet")
    return args

def _default_output_dir(project_root: Path) -> Path:
    return project_root / "logs" / "diagnostics" / "tabular_numeric_stage0_profile" / f"run={_timestamp()}"


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


def _set_candidate_runtime_profile(*, workers: int, threads: int) -> None:
    cfg = load_runtime_config()
    modules = cfg.setdefault("modules", {})
    for spec in TABULAR_STAGE0_BRANCH_SPECS:
        module_cfg = modules.setdefault(str(spec["module_slug"]), {})
        module_cfg["unit_workers"] = int(workers)
        module_cfg["model_threads"] = int(threads)
    RUNTIME_CONFIG_PATH.write_text(json.dumps(cfg, indent=2), encoding="utf-8")


def _run_candidate(*, args: argparse.Namespace, workers: int, threads: int, output_root: Path) -> Dict[str, Any]:
    _set_candidate_runtime_profile(workers=int(workers), threads=int(threads))
    candidate_root = output_root / _candidate_key(workers, threads)
    candidate_root.mkdir(parents=True, exist_ok=True)
    assets_csv = str(args.assets)
    branch_results: List[Dict[str, Any]] = []
    candidate_start = time.perf_counter()
    peak_system_ram_pct = 0.0
    peak_process_rss_bytes = 0
    peak_cpu_percent = 0.0
    total_write_bytes = 0
    total_output_bytes = 0

    for spec in TABULAR_STAGE0_BRANCH_SPECS:
        branch_parquet_root = candidate_root / "parquet" / str(spec["model_key"])
        command = [
            sys.executable,
            "-m",
            str(spec["entrypoint"]),
            "--combo-list",
            f"{int(args.interval)}:{int(args.horizon_minutes)}:{str(args.task)}",
            "--assets",
            assets_csv,
            "--mode",
            "backfill",
            "--unit-workers",
            str(int(workers)),
        ]
        env = dict(os.environ)
        env["RUN_ID"] = _candidate_key(workers, threads)
        env[str(spec["root_env"])] = str(branch_parquet_root)
        branch_metrics = _measure_branch_run(
            command=command,
            env=env,
            cwd=Path(args.project_root),
            log_path=candidate_root / "logs" / f"{spec['model_key']}.log",
            output_dir=branch_parquet_root,
            sample_seconds=float(args.sample_seconds),
        )
        branch_metrics["model_key"] = str(spec["model_key"])
        branch_results.append(branch_metrics)
        peak_system_ram_pct = max(peak_system_ram_pct, float(branch_metrics["peak_system_ram_pct"]))
        peak_process_rss_bytes = max(peak_process_rss_bytes, int(branch_metrics["peak_process_rss_bytes"]))
        peak_cpu_percent = max(peak_cpu_percent, float(branch_metrics["peak_cpu_percent"]))
        total_write_bytes += int(branch_metrics["process_write_bytes"])
        total_output_bytes += int(branch_metrics["output_bytes"])
        if bool(args.prune_run_artifacts):
            shutil.rmtree(branch_parquet_root, ignore_errors=True)

    total_wall_seconds = float(time.perf_counter() - candidate_start)
    return {
        "unit_workers": int(workers),
        "model_threads": int(threads),
        "candidate_key": _candidate_key(workers, threads),
        "task": str(args.task),
        "interval_minutes": int(args.interval),
        "horizon_minutes": int(args.horizon_minutes),
        "assets": [asset for asset in str(args.assets).split(",") if asset],
        "total_wall_seconds": total_wall_seconds,
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
    return min(
        pool,
        key=lambda result: (
            float(result.get("total_wall_seconds", 1e18)),
            float(result.get("peak_system_ram_pct", 1e18)),
            int(result.get("unit_workers", 1e9)),
            int(result.get("model_threads", 1e9)),
        ),
    )


def _write_candidate_csv(path: Path, results: Sequence[Dict[str, Any]]) -> None:
    fieldnames = [
        "candidate_key",
        "unit_workers",
        "model_threads",
        "interval_minutes",
        "horizon_minutes",
        "task",
        "total_wall_seconds",
        "peak_system_ram_pct",
        "peak_process_rss_bytes",
        "peak_cpu_percent",
        "total_process_write_bytes",
        "total_output_bytes",
    ]
    _shared_write_candidate_csv(path, results, fieldnames=fieldnames)


def _update_runtime_profile(unit_workers: int, model_threads: int) -> None:
    _set_candidate_runtime_profile(workers=int(unit_workers), threads=int(model_threads))


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
    _update_runtime_profile(int(selected["unit_workers"]), int(selected["model_threads"]))

    artifact = {
        "stage": 0,
        "family": "tabular_numeric",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "task": str(args.task),
        "interval_minutes": int(args.interval),
        "horizon_minutes": int(args.horizon_minutes),
        "assets": [asset for asset in str(args.assets).split(",") if asset],
        "ram_cap_pct": float(args.ram_cap_pct),
        "candidates": results,
        "selected_profile": {
            "unit_workers": int(selected["unit_workers"]),
            "model_threads": int(selected["model_threads"]),
            "candidate_key": str(selected["candidate_key"]),
            "total_wall_seconds": float(selected["total_wall_seconds"]),
            "peak_system_ram_pct": float(selected["peak_system_ram_pct"]),
        },
        "runtime_config_path": str(RUNTIME_CONFIG_PATH),
    }
    (output_dir / "tabular_numeric_stage0_profile.json").write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    _write_candidate_csv(output_dir / "tabular_numeric_stage0_candidates.csv", results)
    return output_dir


def main() -> None:
    args = parse_args()
    require_pipeline_io(profile=str(args.profile or "pipeline_test"))
    run_root = run_stage0(args)
    print(f"[stage0] completed {run_root}")


if __name__ == "__main__":
    main()
