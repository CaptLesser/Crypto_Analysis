from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from src.forecasting.common.io_atomic import atomic_replace, sibling_temp_path
from src.forecasting.common.path_config import resolve_path, selected_profile
from src.forecasting.ml.shared.stage0_profile_common import (
    candidate_key,
    emit_stage0_event,
    emit_stage0_profile_artifacts,
    measure_branch_run,
    parse_int_csv,
    parse_str_csv,
    stage0_telemetry_scope,
    timestamp_utc,
    write_candidate_csv,
)
from src.forecasting.common.stats_module_utils import CAPABILITY_MATRIX
from src.forecasting.stats.shared.stats_numeric_model_registry import (
    STATS_NUMERIC_BRANCHES,
    STATS_NUMERIC_ENTRYPOINTS,
    STATS_NUMERIC_FAMILY_ROOT_ENVS,
    STATS_NUMERIC_FAMILY_ROOT_NAMES,
)

try:
    import psutil  # type: ignore
except Exception:  # pragma: no cover
    psutil = None


SATURATION_SELECTION_TOLERANCE_RATIO = 0.05
DEFAULT_WORKER_CANDIDATES = (6, 8, 10, 12, 14)
DEFAULT_THREAD_CANDIDATES = (4, 6, 8)
DEFAULT_COMBO_LIST = "60:240:log_return"


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _progress(message: str) -> None:
    print(f"[{datetime.now(timezone.utc).isoformat()}] {message}", flush=True)


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = sibling_temp_path(path)
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    atomic_replace(tmp, path)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Frequentist stats Stage 0 runtime profile sweep")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--profile", type=str, default=selected_profile(default="pipeline_test"))
    parser.add_argument("--parquet-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--workers", type=str, default=",".join(str(v) for v in DEFAULT_WORKER_CANDIDATES))
    parser.add_argument("--threads", type=str, default=",".join(str(v) for v in DEFAULT_THREAD_CANDIDATES))
    parser.add_argument("--combo-list", type=str, default=DEFAULT_COMBO_LIST)
    parser.add_argument("--assets", type=str, default="")
    parser.add_argument("--max-assets", type=int, default=1)
    parser.add_argument("--backfill-days", type=int, default=3)
    parser.add_argument("--fit-days", type=int, default=60)
    parser.add_argument("--sample-seconds", type=float, default=1.0)
    parser.add_argument("--progress-seconds", type=float, default=30.0)
    parser.add_argument("--ram-cap-pct", type=float, default=80.0)
    parser.add_argument("--prune-run-artifacts", action="store_true")
    args = parser.parse_args(argv)
    if args.parquet_root is None:
        args.parquet_root = Path(resolve_path("source_ohlcvt_root", profile=str(args.profile), required=False) or Path("parquet"))
    return args


def _default_output_dir(project_root: Path) -> Path:
    return Path(project_root) / "logs" / "diagnostics" / "stats_numeric_stage0_profile" / f"run={_stamp()}"


def _parse_combo_list(raw: str) -> List[tuple[int, int, str]]:
    combos: List[tuple[int, int, str]] = []
    for token in [part.strip() for part in str(raw).split(",") if part.strip()]:
        try:
            interval, horizon, task = token.split(":", 2)
            combos.append((int(interval), int(horizon), str(task)))
        except Exception:
            continue
    return sorted(set(combos), key=lambda item: (item[0], item[1], item[2]))


def _discover_assets_from_parquet_root(parquet_root: Path, intervals: Sequence[int], *, max_assets: int) -> List[str]:
    assets: set[str] = set()
    root = Path(parquet_root)
    for interval in intervals:
        for table_name in (f"ohlcvt_{int(interval)}", f"scalar_features_{int(interval)}"):
            table_root = root / table_name
            if not table_root.exists():
                continue
            for child in table_root.glob("asset=*"):
                if not child.is_dir():
                    continue
                asset = child.name.split("=", 1)[1].strip()
                if asset:
                    assets.add(asset)
    return sorted(assets)[: max(1, int(max_assets))]


def _resolve_assets(args: argparse.Namespace, combos: Sequence[tuple[int, int, str]]) -> List[str]:
    explicit = parse_str_csv(str(args.assets))
    if explicit:
        return explicit[: max(1, int(args.max_assets))]
    intervals = sorted({int(interval) for interval, _, _ in combos})
    return _discover_assets_from_parquet_root(Path(args.parquet_root), intervals, max_assets=int(args.max_assets))


def _select_best_candidate(candidates: Sequence[Dict[str, Any]], ram_cap_pct: float = 80.0) -> Dict[str, Any]:
    if not candidates:
        raise ValueError("no stats stage0 candidates")
    within_cap = [row for row in candidates if float(row.get("peak_system_ram_pct", 0.0)) <= float(ram_cap_pct)]
    pool = within_cap if within_cap else list(candidates)
    best_wall = min(float(row.get("total_wall_seconds", 1e18)) for row in pool)
    near_best = [
        row
        for row in pool
        if float(row.get("total_wall_seconds", 1e18)) <= best_wall * (1.0 + SATURATION_SELECTION_TOLERANCE_RATIO)
    ]
    selection_pool = near_best if near_best else pool
    return max(
        selection_pool,
        key=lambda row: (
            float(row.get("throughput_score", 0.0)),
            int(row.get("workers", 0)),
            int(row.get("threads", 0)),
            float(row.get("peak_cpu_percent", 0.0)),
            -float(row.get("total_wall_seconds", 1e18)),
            -float(row.get("peak_system_ram_pct", 1e18)),
        ),
    )


def _select_best_branch_profiles(candidates: Sequence[Dict[str, Any]], ram_cap_pct: float = 80.0) -> Dict[str, Dict[str, Any]]:
    branch_candidates: Dict[str, List[Dict[str, Any]]] = {}
    for row in candidates:
        for branch_result in row.get("branch_results", []) or []:
            model_key = str(branch_result.get("model_key", "")).strip()
            if not model_key:
                continue
            branch_candidates.setdefault(model_key, []).append(
                {
                    "candidate_key": row.get("candidate_key"),
                    "workers": int(row.get("workers", 1) or 1),
                    "threads": int(row.get("threads", 1) or 1),
                    "asset_workers": int(row.get("asset_workers", row.get("workers", 1)) or 1),
                    "model_threads": int(row.get("model_threads", row.get("threads", 1)) or 1),
                    "total_wall_seconds": float(branch_result.get("wall_seconds", 0.0) or 0.0),
                    "peak_system_ram_pct": float(branch_result.get("peak_system_ram_pct", row.get("peak_system_ram_pct", 0.0)) or 0.0),
                    "peak_cpu_percent": float(branch_result.get("peak_cpu_percent", 0.0) or 0.0),
                    "throughput_score": 1.0 / max(float(branch_result.get("wall_seconds", 0.0) or 0.0), 1e-9),
                }
            )
    return {
        model_key: {
            "workers": int(selected["workers"]),
            "threads": int(selected["threads"]),
            "asset_workers": int(selected["asset_workers"]),
            "model_threads": int(selected["model_threads"]),
            "candidate_key": str(selected["candidate_key"]),
            "total_wall_seconds": float(selected["total_wall_seconds"]),
            "peak_system_ram_pct": float(selected["peak_system_ram_pct"]),
        }
        for model_key, rows in sorted(branch_candidates.items())
        for selected in [_select_best_candidate(rows, ram_cap_pct=ram_cap_pct)]
    }


def _run_candidate(
    *,
    args: argparse.Namespace,
    workers: int,
    threads: int,
    output_root: Path,
    combos: Sequence[tuple[int, int, str]],
    assets: Sequence[str],
) -> Dict[str, Any]:
    candidate = candidate_key(int(workers), int(threads))
    candidate_root = Path(output_root) / candidate
    candidate_root.mkdir(parents=True, exist_ok=True)
    assets_csv = ",".join(str(asset) for asset in assets)
    branch_results: List[Dict[str, Any]] = []
    started = time.perf_counter()
    peak_system_ram_pct = 0.0
    peak_process_rss_bytes = 0
    peak_cpu_percent = 0.0
    total_write_bytes = 0
    total_output_bytes = 0

    for model_key in STATS_NUMERIC_BRANCHES:
        supported_tasks = set(str(task) for task in CAPABILITY_MATRIX.get(str(model_key), {}).get("numerics", ()))
        branch_combos = [combo for combo in combos if str(combo[2]) in supported_tasks]
        if not branch_combos and combos and supported_tasks:
            interval, horizon, _task = combos[0]
            branch_combos = [(int(interval), int(horizon), sorted(supported_tasks)[0])]
        if not branch_combos:
            _progress(f"Stats Stage 0 candidate={candidate} branch={model_key} skipped no compatible probe combos")
            emit_stage0_event(
                candidate_root,
                family="Stats_Numeric",
                model=str(model_key),
                function_name="_run_candidate",
                module_name=__name__,
                phase_name="combo_planning",
                parent_phase="profile_creation",
                status="skipped",
                reason_code="profile_missing",
                input_rows=len(combos),
                output_rows=0,
                asset_count=len(assets),
                output_path=str(candidate_root),
            )
            continue
        branch_parquet_root = candidate_root / "parquet" / str(model_key)
        branch_output_dir = branch_parquet_root / STATS_NUMERIC_FAMILY_ROOT_NAMES[str(model_key)]
        branch_combo_list_arg = ",".join(f"{int(interval)}:{int(horizon)}:{task}" for interval, horizon, task in branch_combos)
        command = [
            sys.executable,
            "-m",
            STATS_NUMERIC_ENTRYPOINTS[str(model_key)],
            "--parquet-root",
            str(Path(args.parquet_root).resolve()),
            "--combo-list",
            branch_combo_list_arg,
            "--assets",
            assets_csv,
            "--workers",
            str(int(workers)),
            "--backfill_days",
            str(int(args.backfill_days)),
            "--fit_days",
            str(int(args.fit_days)),
            "--predict_latest_only",
            "--force",
        ]
        env = dict(os.environ)
        env[STATS_NUMERIC_FAMILY_ROOT_ENVS[str(model_key)]] = str(branch_parquet_root)
        env["OMP_NUM_THREADS"] = str(max(1, int(threads)))
        env["MKL_NUM_THREADS"] = str(max(1, int(threads)))
        env["OPENBLAS_NUM_THREADS"] = str(max(1, int(threads)))
        _progress(
            f"Stats Stage 0 candidate={candidate} branch={model_key} starting "
            f"workers={int(workers)} threads={int(threads)} combos={len(branch_combos)} assets={len(assets)}"
        )
        branch_started = time.perf_counter()
        with stage0_telemetry_scope(
            candidate_root,
            family="Stats_Numeric",
            model=str(model_key),
            function_name="measure_branch_run",
            module_name=__name__,
            phase_name="profile_candidate_probe",
            parent_phase="profile_creation",
            combo_key=branch_combo_list_arg,
            asset_count=len(assets),
            source_path=str(args.parquet_root),
            output_path=str(branch_output_dir),
        ) as telemetry:
            metrics = measure_branch_run(
                command=command,
                env=env,
                cwd=Path(args.project_root),
                log_path=candidate_root / "logs" / f"{model_key}.log",
                output_dir=branch_output_dir,
                sample_seconds=float(args.sample_seconds),
                psutil_module=psutil,
            )
            telemetry.update(output_rows=1)
        metrics["model_key"] = str(model_key)
        metrics["combo_count"] = int(len(branch_combos))
        _progress(
            f"Stats Stage 0 candidate={candidate} branch={model_key} complete "
            f"wall_s={float(time.perf_counter() - branch_started):.1f} "
            f"output_mb={float(metrics['output_bytes']) / (1024 * 1024):.2f} "
            f"rss_mb={float(metrics['peak_process_rss_bytes']) / (1024 * 1024):.1f} "
            f"ram_pct={float(metrics['peak_system_ram_pct']):.1f}"
        )
        branch_results.append(metrics)
        peak_system_ram_pct = max(peak_system_ram_pct, float(metrics["peak_system_ram_pct"]))
        peak_process_rss_bytes = max(peak_process_rss_bytes, int(metrics["peak_process_rss_bytes"]))
        peak_cpu_percent = max(peak_cpu_percent, float(metrics["peak_cpu_percent"]))
        total_write_bytes += int(metrics["process_write_bytes"])
        total_output_bytes += int(metrics["output_bytes"])
        if bool(args.prune_run_artifacts):
            shutil.rmtree(branch_output_dir, ignore_errors=True)

    total_wall_seconds = float(time.perf_counter() - started)
    unit_count = max(1, int(len(combos)) * int(len(assets)) * int(len(STATS_NUMERIC_BRANCHES)))
    return {
        "candidate_key": candidate,
        "workers": int(workers),
        "threads": int(threads),
        "asset_workers": int(workers),
        "model_threads": int(threads),
        "total_threads": int(workers) * int(threads),
        "combo_count": int(len(combos)),
        "asset_count": int(len(assets)),
        "branch_count": int(len(STATS_NUMERIC_BRANCHES)),
        "total_wall_seconds": total_wall_seconds,
        "throughput_score": float(unit_count / max(total_wall_seconds, 1e-9)),
        "peak_system_ram_pct": peak_system_ram_pct,
        "peak_process_rss_bytes": int(peak_process_rss_bytes),
        "peak_cpu_percent": peak_cpu_percent,
        "total_process_write_bytes": int(total_write_bytes),
        "total_output_bytes": int(total_output_bytes),
        "branch_results": branch_results,
    }


def _write_candidates(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    write_candidate_csv(
        path,
        rows,
        fieldnames=[
            "candidate_key",
            "workers",
            "threads",
            "asset_workers",
            "model_threads",
            "total_threads",
            "combo_count",
            "asset_count",
            "branch_count",
            "total_wall_seconds",
            "throughput_score",
            "peak_system_ram_pct",
            "peak_process_rss_bytes",
            "peak_cpu_percent",
            "total_process_write_bytes",
            "total_output_bytes",
        ],
    )


def run_stage0(args: argparse.Namespace) -> Path:
    output_dir = Path(args.output_dir) if args.output_dir is not None else _default_output_dir(Path(args.project_root))
    output_dir.mkdir(parents=True, exist_ok=True)
    combos = _parse_combo_list(str(args.combo_list))
    emit_stage0_event(
        output_dir,
        family="Stats_Numeric",
        function_name="run_stage0",
        module_name=__name__,
        phase_name="combo_planning",
        status="completed" if combos else "skipped",
        reason_code="" if combos else "profile_missing",
        input_rows=1,
        output_rows=len(combos),
        source_path=str(args.parquet_root),
        output_path=str(output_dir),
    )
    if not combos:
        raise SystemExit("Stats Stage 0 requires at least one combo in --combo-list")
    assets = _resolve_assets(args, combos)
    emit_stage0_event(
        output_dir,
        family="Stats_Numeric",
        function_name="_resolve_assets",
        module_name=__name__,
        phase_name="asset_cohort_selection",
        status="completed" if assets else "skipped",
        reason_code="" if assets else "no_assets",
        input_rows=len(combos),
        output_rows=len(assets),
        asset_count=len(assets),
        source_path=str(args.parquet_root),
        output_path=str(output_dir),
    )
    if not assets:
        raise SystemExit("Stats Stage 0 could not resolve any assets for the probe run")

    results: List[Dict[str, Any]] = []
    worker_values = parse_int_csv(str(args.workers))
    thread_values = parse_int_csv(str(args.threads))
    total_candidates = int(len(worker_values) * len(thread_values))
    _progress(
        f"Stats Stage 0 probe sweep starting candidates={total_candidates} branches={len(STATS_NUMERIC_BRANCHES)} "
        f"combos={len(combos)} assets={len(assets)} output_dir={output_dir}"
    )
    candidate_idx = 0
    for workers in worker_values:
        for threads in thread_values:
            candidate_idx += 1
            _progress(f"Stats Stage 0 candidate {candidate_idx}/{total_candidates} starting workers={workers} threads={threads}")
            candidate_started = time.perf_counter()
            results.append(
                _run_candidate(
                    args=args,
                    workers=int(workers),
                    threads=int(threads),
                    output_root=output_dir,
                    combos=combos,
                    assets=assets,
                )
            )
            latest = results[-1]
            _progress(
                f"Stats Stage 0 candidate {candidate_idx}/{total_candidates} complete "
                f"wall_s={float(time.perf_counter() - candidate_started):.1f} "
                f"throughput={float(latest['throughput_score']):.4f} "
                f"ram_pct={float(latest['peak_system_ram_pct']):.1f}"
            )

    selected = _select_best_candidate(results, float(args.ram_cap_pct))
    selected_profiles_by_branch = _select_best_branch_profiles(results, float(args.ram_cap_pct))
    _progress(
        f"Stats Stage 0 selected candidate={selected['candidate_key']} "
        f"workers={int(selected['workers'])} threads={int(selected['threads'])} "
        f"wall_s={float(selected['total_wall_seconds']):.1f}"
    )
    payload = {
        "family": "Stats_Frequentist",
        "stage": "stage0",
        "branches": list(STATS_NUMERIC_BRANCHES),
        "selected_profile": {
            "workers": int(selected["workers"]),
            "threads": int(selected["threads"]),
            "asset_workers": int(selected["asset_workers"]),
            "model_threads": int(selected["model_threads"]),
            "candidate_key": str(selected["candidate_key"]),
            "total_wall_seconds": float(selected["total_wall_seconds"]),
            "peak_system_ram_pct": float(selected["peak_system_ram_pct"]),
        },
        "selected_profiles_by_branch": selected_profiles_by_branch,
        "candidates": results,
        "combo_list": [{"interval_minutes": int(iv), "horizon_minutes": int(hm), "task": str(task)} for iv, hm, task in combos],
        "assets": list(assets),
        "ram_cap_pct": float(args.ram_cap_pct),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    profile_path = output_dir / "stats_numeric_stage0_profile.json"
    candidates_path = output_dir / "stats_numeric_stage0_candidates.csv"
    _write_json(profile_path, payload)
    _write_candidates(candidates_path, results)
    emit_stage0_profile_artifacts(
        output_dir,
        family="Stats_Numeric",
        profile_path=profile_path,
        candidates_path=candidates_path,
        candidates=results,
        selected_profile=dict(payload.get("selected_profile") or {}),
        asset_count=len(payload.get("assets") or []),
        combo_count=len(combos),
    )
    return output_dir


def main(argv: Optional[Sequence[str]] = None) -> None:
    run_root = run_stage0(parse_args(argv))
    print(f"[stage0] completed {run_root}", flush=True)


if __name__ == "__main__":
    main()
