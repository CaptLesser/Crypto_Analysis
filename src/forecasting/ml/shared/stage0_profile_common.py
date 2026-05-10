from __future__ import annotations

import csv
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence

from src.forecasting.ml.shared.test_branch_function_telemetry import emit_event_for_path, telemetry_scope_for_path


def parse_int_csv(raw: str) -> List[int]:
    return [int(part.strip()) for part in str(raw).split(",") if part.strip()]


def parse_str_csv(raw: str) -> List[str]:
    return [str(part.strip()) for part in str(raw).split(",") if str(part).strip()]


def resolve_combo_list(args: Any) -> List[tuple[int, int, str]]:
    if int(getattr(args, "interval", 0)) > 0 and int(getattr(args, "horizon_minutes", 0)) > 0 and str(getattr(args, "task", "")).strip():
        return [(int(args.interval), int(args.horizon_minutes), str(args.task).strip())]
    intervals = parse_int_csv(str(args.intervals))
    horizons = parse_int_csv(str(args.horizons))
    tasks = parse_str_csv(str(args.tasks))
    combos = [
        (int(interval), int(horizon), str(task))
        for interval in intervals
        for horizon in horizons
        for task in tasks
        if int(interval) > 0 and int(horizon) > 0 and int(horizon) % int(interval) == 0
    ]
    return sorted(set(combos), key=lambda item: (item[0], item[1], item[2]))


def timestamp_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def dir_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for file_path in path.rglob("*"):
        if file_path.is_file():
            try:
                total += int(file_path.stat().st_size)
            except OSError:
                continue
    return total


def process_tree_rss_bytes(proc: Any) -> int:
    total = 0
    try:
        total += int(proc.memory_info().rss)
    except Exception:
        return 0
    try:
        children = proc.children(recursive=True)
    except Exception:
        children = []
    for child in children:
        try:
            total += int(child.memory_info().rss)
        except Exception:
            continue
    return total


def process_tree_cpu_percent(proc: Any) -> float:
    total = 0.0
    try:
        total += float(proc.cpu_percent(interval=None))
    except Exception:
        return 0.0
    try:
        children = proc.children(recursive=True)
    except Exception:
        children = []
    for child in children:
        try:
            total += float(child.cpu_percent(interval=None))
        except Exception:
            continue
    return total


def process_tree_write_bytes(proc: Any) -> int:
    total = 0
    targets = [proc]
    try:
        targets.extend(proc.children(recursive=True))
    except Exception:
        pass
    for target in targets:
        try:
            total += int(target.io_counters().write_bytes)
        except Exception:
            continue
    return total


def measure_branch_run(
    *,
    command: Sequence[str],
    env: Dict[str, str],
    cwd: Path,
    log_path: Path,
    output_dir: Path,
    sample_seconds: float,
    psutil_module: Any = None,
) -> Dict[str, Any]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(command, cwd=str(cwd), env=env, stdout=log_file, stderr=subprocess.STDOUT)
        peak_rss_bytes = 0
        peak_cpu_percent = 0.0
        peak_system_ram_pct = 0.0
        baseline_write_bytes = 0
        final_write_bytes = 0
        if psutil_module is not None:
            proc = psutil_module.Process(process.pid)
            try:
                proc.cpu_percent(interval=None)
                for child in proc.children(recursive=True):
                    child.cpu_percent(interval=None)
            except Exception:
                pass
            baseline_write_bytes = process_tree_write_bytes(proc)
            while process.poll() is None:
                time.sleep(max(0.2, float(sample_seconds)))
                peak_rss_bytes = max(peak_rss_bytes, process_tree_rss_bytes(proc))
                peak_cpu_percent = max(peak_cpu_percent, process_tree_cpu_percent(proc))
                try:
                    peak_system_ram_pct = max(peak_system_ram_pct, float(psutil_module.virtual_memory().percent))
                except Exception:
                    pass
            final_write_bytes = process_tree_write_bytes(proc)
            peak_rss_bytes = max(peak_rss_bytes, process_tree_rss_bytes(proc))
            try:
                peak_system_ram_pct = max(peak_system_ram_pct, float(psutil_module.virtual_memory().percent))
            except Exception:
                pass
        return_code = int(process.wait())
    wall_seconds = float(time.perf_counter() - start)
    output_bytes = dir_size_bytes(output_dir)
    if return_code != 0:
        raise RuntimeError(f"Command failed with exit code {return_code}: {' '.join(command)}")
    return {
        "wall_seconds": wall_seconds,
        "peak_process_rss_bytes": int(peak_rss_bytes),
        "peak_system_ram_pct": float(peak_system_ram_pct),
        "peak_cpu_percent": float(peak_cpu_percent),
        "process_write_bytes": max(0, int(final_write_bytes) - int(baseline_write_bytes)),
        "output_bytes": int(output_bytes),
        "log_path": str(log_path),
        "output_dir": str(output_dir),
    }


def candidate_key(workers: int, threads: int) -> str:
    return f"workers={int(workers)}_threads={int(threads)}"


def write_candidate_csv(path: Path, results: Sequence[Dict[str, Any]], *, fieldnames: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for result in results:
            writer.writerow({name: result.get(name) for name in fieldnames})


def emit_stage0_event(
    output_dir: Path,
    *,
    family: str,
    function_name: str,
    phase_name: str,
    model: str = "family",
    status: str = "completed",
    reason_code: str = "",
    **payload: Any,
) -> None:
    emit_event_for_path(
        Path(output_dir),
        family=str(family),
        model=str(model),
        stage="stage0",
        function_name=str(function_name),
        module_name=payload.pop("module_name", ""),
        phase_name=str(phase_name),
        status=str(status),
        reason_code=str(reason_code or ""),
        **payload,
    )


def stage0_telemetry_scope(
    output_dir: Path,
    *,
    family: str,
    function_name: str,
    phase_name: str,
    model: str = "family",
    status: str = "completed",
    reason_code: str = "",
    **payload: Any,
) -> Iterator[Any]:
    return telemetry_scope_for_path(
        Path(output_dir),
        family=str(family),
        model=str(model),
        stage="stage0",
        function_name=str(function_name),
        module_name=str(payload.pop("module_name", "")),
        phase_name=str(phase_name),
        status=str(status),
        reason_code=str(reason_code or ""),
        **payload,
    )


def emit_stage0_profile_artifacts(
    output_dir: Path,
    *,
    family: str,
    profile_path: Path,
    candidates_path: Path,
    candidates: Sequence[Dict[str, Any]],
    selected_profile: Optional[Dict[str, Any]],
    asset_count: int,
    combo_count: int = 1,
) -> None:
    profile_exists = Path(profile_path).exists()
    candidates_exists = Path(candidates_path).exists()
    valid = bool(selected_profile) and bool(candidates) and profile_exists and candidates_exists
    emit_stage0_event(
        output_dir,
        family=family,
        function_name="validate_stage0_profile",
        module_name=__name__,
        phase_name="profile_validation",
        status="completed" if valid else "failed",
        reason_code="" if valid else "profile_validation_failed",
        input_rows=len(candidates),
        output_rows=1 if valid else 0,
        asset_count=int(asset_count),
        output_path=str(profile_path),
    )
    emit_stage0_event(
        output_dir,
        family=family,
        function_name="write_stage0_profile",
        module_name=__name__,
        phase_name="write",
        status="completed" if profile_exists else "failed",
        reason_code="" if profile_exists else "profile_write_failed",
        input_rows=len(candidates),
        output_rows=1 if profile_exists else 0,
        asset_count=int(asset_count),
        output_path=str(profile_path),
    )
    emit_stage0_event(
        output_dir,
        family=family,
        function_name="run_stage0",
        module_name=__name__,
        phase_name="artifact_handoff",
        status="completed" if valid else "failed",
        reason_code="" if valid else "profile_validation_failed",
        input_rows=len(candidates),
        output_rows=int(combo_count),
        asset_count=int(asset_count),
        output_path=str(profile_path),
        artifact_profile_source=str(profile_path),
    )
