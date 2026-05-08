from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.forecasting.ml.production_orchestrator import __version__
from src.forecasting.common.path_config import PathConfigError, PipelineIOConfig, pipeline_io_env, require_pipeline_io, selected_profile
from src.forecasting.common.pipeline_parquet_utils import partition_max_ts
from src.forecasting.common.sandbox_paths import (
    SandboxOutputRoots,
    assert_write_allowed,
    resolve_sandbox_output_roots,
    sandbox_env_for_subprocess,
)
from src.forecasting.ml.production_orchestrator.common import (
    append_log_line,
    command_signature,
    ensure_serializable,
    load_json_dict,
    maybe_git_head,
    resolve_run_root,
    utc_now_iso,
    write_json_atomic,
)
from src.forecasting.ml.production_orchestrator.contracts import (
    build_contract_spec,
    snapshot_payload,
    validate_contract,
)
from src.forecasting.ml.production_orchestrator.heuristics import derive_heuristics
from src.forecasting.ml.production_orchestrator.registry import ProductionModuleSpec, mature_ml_modules
from src.forecasting.ml.production_orchestrator.telemetry import TelemetryRecorder


RETENTION_KEEP_RUN_MANIFESTS = 10
RETENTION_KEEP_SUCCESS_RUNS = 5
RETENTION_KEEP_FAILURE_RUNS = 20
DEFAULT_OUTPUT_DIR = Path("logs") / "diagnostics" / "production_numeric_orchestrator"


@dataclass
class OrchestratorArgs:
    project_root: Path
    output_dir: Path
    run_id: str = ""
    resume_run: str = ""
    no_resume_latest: bool = False
    sample_seconds: float = 5.0
    python_exe: str = sys.executable
    deep_disk_preflight: bool = False
    sandbox_output_root: Optional[Path] = None
    profile: str = "production"


def _module_dir(run_root: Path, module_key: str) -> Path:
    return run_root / "modules" / str(module_key)


def _pid_is_running(pid: int) -> bool:
    if int(pid) <= 0:
        return False
    if int(pid) == os.getpid():
        return True
    try:
        import psutil  # type: ignore

        return bool(psutil.pid_exists(int(pid)))
    except Exception:
        pass
    try:
        os.kill(int(pid), 0)
        return True
    except PermissionError:
        return True
    except OSError:
        return False


def _acquire_run_lock(run_root: Path) -> Path:
    lock_path = run_root / "orchestrator.lock"
    payload = {"pid": os.getpid(), "started_at": utc_now_iso()}
    while True:
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            existing = load_json_dict(lock_path)
            existing_pid = int(existing.get("pid", 0) or 0)
            if _pid_is_running(existing_pid):
                raise RuntimeError(
                    f"Production orchestrator run is already active for {run_root} "
                    f"(pid={existing_pid}, lock={lock_path})"
                )
            try:
                lock_path.unlink()
            except FileNotFoundError:
                pass
            continue
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True)
        return lock_path


def _release_run_lock(lock_path: Path) -> None:
    payload = load_json_dict(lock_path)
    lock_pid = int(payload.get("pid", 0) or 0)
    if lock_pid not in {0, os.getpid()}:
        return
    try:
        lock_path.unlink()
    except FileNotFoundError:
        pass


def _print_terminal_progress(message: str) -> None:
    print(message, flush=True)


def _write_module_summary_csv(run_root: Path, rows: List[Dict[str, Any]]) -> None:
    path = run_root / "module_run_summary.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run_id",
        "module_key",
        "family",
        "order_index",
        "entrypoint",
        "command",
        "status",
        "failure_type",
        "resumed",
        "resume_rejected",
        "resume_rejected_reason",
        "started_at",
        "finished_at",
        "runtime_seconds",
        "configured_workers",
        "configured_threads",
        "peak_child_count",
        "peak_thread_count",
        "peak_rss_mb",
        "avg_cpu_pct_tree",
        "peak_cpu_pct_tree",
        "avg_cpu_pct_system",
        "peak_cpu_pct_system",
        "avg_ram_pct_system",
        "peak_ram_pct_system",
        "rows_written",
        "parts_written",
        "assets_touched",
        "units_processed",
        "units_skipped",
        "resumed_units",
        "peak_rss_phase",
        "peak_thread_phase",
        "peak_cpu_phase",
        "last_progress_age_seconds",
        "last_artifact_change_age_seconds",
        "log_size_bytes_final",
        "artifact_file_count_final",
        "stalled_warning",
        "contract_completed",
        "heuristics",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


def _write_module_summary_json(run_root: Path, rows: List[Dict[str, Any]]) -> None:
    write_json_atomic(run_root / "module_run_summary.json", {"schema_version": 1, "rows": rows})


def _retained_runs(base_output_dir: Path) -> List[Path]:
    return sorted(
        (path for path in base_output_dir.glob("run=*") if path.is_dir()),
        key=lambda item: item.name,
        reverse=True,
    )


def _prune_old_samples(run_root: Path) -> None:
    for path in run_root.glob("modules/*/telemetry_samples.csv"):
        try:
            path.unlink()
        except Exception:
            continue


def _enforce_retention(base_output_dir: Path) -> None:
    runs = _retained_runs(base_output_dir)
    success_runs: List[Path] = []
    failure_runs: List[Path] = []
    for run_root in runs:
        manifest_path = run_root / "orchestrator_run_manifest.json"
        payload = {}
        if manifest_path.exists():
            try:
                payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            except Exception:
                payload = {}
        status = str(payload.get("status", "")).strip().lower()
        if status == "completed":
            success_runs.append(run_root)
        else:
            failure_runs.append(run_root)
    keep: set[Path] = set(runs[:RETENTION_KEEP_RUN_MANIFESTS])
    keep.update(success_runs[:RETENTION_KEEP_SUCCESS_RUNS])
    keep.update(failure_runs[:RETENTION_KEEP_FAILURE_RUNS])
    for run_root in runs:
        if run_root not in keep:
            shutil.rmtree(run_root, ignore_errors=True)
            continue
        if run_root in success_runs[RETENTION_KEEP_SUCCESS_RUNS:]:
            _prune_old_samples(run_root)


def _basic_resume_compatibility(
    previous: Dict[str, Any],
    current_snapshot: Dict[str, Any],
    predecessor_ok: bool,
) -> tuple[bool, Optional[str]]:
    if not predecessor_ok:
        return False, "stale_resume_state"
    if str(previous.get("module_key")) != str(current_snapshot.get("module_key")):
        return False, "module_identity_mismatch"
    if str(previous.get("command_signature")) != str(current_snapshot.get("command_signature")):
        return False, "command_signature_mismatch"
    prev_output_root = str(previous.get("contract_snapshot", {}).get("output_root", ""))
    new_output_root = str(current_snapshot.get("output_root", ""))
    if prev_output_root and new_output_root and prev_output_root != new_output_root:
        return False, "output_root_mismatch"
    return True, None


def _initial_manifest(run_root: Path, args: OrchestratorArgs) -> Dict[str, Any]:
    started_at = utc_now_iso()
    sandbox_roots = resolve_sandbox_output_roots(args)
    return {
        "schema_version": 1,
        "run_id": str(run_root.name),
        "generated_at": started_at,
        "started_at": started_at,
        "finished_at": None,
        "status": "running",
        "project_root": str(args.project_root.resolve()),
        "orchestrator_version": f"v1/{__version__}",
        "resume_source_run_id": None,
        "default_python_exe": str(args.python_exe),
        "sandbox": {
            "enabled": bool(sandbox_roots.enabled),
            "output_root": str(sandbox_roots.root) if sandbox_roots.enabled else None,
        },
        "module_execution_order": [
            {
                "module_key": str(spec.module_key),
                "family": str(spec.family),
                "entrypoint": str(spec.entrypoint),
                "command": spec.command(args.python_exe),
            }
            for spec in mature_ml_modules()
        ],
        "retention_policy": {
            "keep_run_manifests": RETENTION_KEEP_RUN_MANIFESTS,
            "keep_success_runs": RETENTION_KEEP_SUCCESS_RUNS,
            "keep_failure_runs": RETENTION_KEEP_FAILURE_RUNS,
        },
        "modules": [],
    }


def _load_or_init_manifest(run_root: Path, args: OrchestratorArgs) -> Dict[str, Any]:
    manifest_path = run_root / "orchestrator_run_manifest.json"
    if manifest_path.exists():
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                return payload
        except Exception:
            pass
    payload = _initial_manifest(run_root, args)
    write_json_atomic(manifest_path, payload)
    return payload


def _write_manifest(run_root: Path, manifest: Dict[str, Any]) -> None:
    write_json_atomic(run_root / "orchestrator_run_manifest.json", ensure_serializable(manifest))


def _replace_module_state(manifest: Dict[str, Any], module_state: Dict[str, Any]) -> None:
    manifest["modules"] = [
        item for item in manifest.get("modules", []) if str(item.get("module_key")) != str(module_state.get("module_key"))
    ]
    manifest["modules"].append(module_state)


def _configured_workers(contract_spec: Any) -> Optional[int]:
    for key in ("asset_workers", "unit_workers", "workers"):
        value = contract_spec.resolved_runtime_values.get(key)
        if value is not None:
            return int(value)
    return None


def _configured_threads(contract_spec: Any) -> Optional[int]:
    for key in ("model_threads", "threads"):
        value = contract_spec.resolved_runtime_values.get(key)
        if value is not None:
            return int(value)
    return None


def _empty_telemetry_summary(module_key: str, configured_workers: Optional[int], configured_threads: Optional[int]) -> Dict[str, Any]:
    return {
        "schema_version": 1,
        "module_key": str(module_key),
        "sample_count": 0,
        "sample_interval_seconds": 0.0,
        "runtime_seconds": 0.0,
        "python_exe": str(sys.executable),
        "command": [],
        "configured_runtime": {
            "workers": configured_workers,
            "threads": configured_threads,
        },
        "process_tree": {
            "peak_child_count": 0,
            "peak_thread_count": 0,
            "peak_rss_mb": 0.0,
            "avg_cpu_pct": 0.0,
            "peak_cpu_pct": 0.0,
        },
        "system": {
            "avg_cpu_pct": 0.0,
            "peak_cpu_pct": 0.0,
            "avg_ram_pct": 0.0,
            "peak_ram_pct": 0.0,
            "min_disk_free_gb": None,
        },
        "phase_peaks": {
            "peak_rss_phase": "unknown",
            "peak_thread_phase": "unknown",
            "peak_cpu_phase": "unknown",
        },
        "phase_durations": {},
        "last_progress_age_seconds": 0.0,
        "last_artifact_change_age_seconds": 0.0,
        "log_size_bytes_final": 0,
        "artifact_file_count_final": 0,
        "heuristics": [],
    }


def _runtime_contract_spec(contract_spec: Any, module_log: Path) -> Any:
    return replace(contract_spec, log_path=module_log.resolve())


def _matching_output_table_roots(output_root: Path, prefixes: List[str]) -> List[Path]:
    root = Path(output_root)
    if not root.exists():
        return []
    prefix_set = [str(prefix).strip() for prefix in prefixes if str(prefix).strip()]
    if not prefix_set:
        return []
    roots: List[Path] = []
    try:
        for child in root.iterdir():
            if not child.is_dir():
                continue
            name = child.name
            if any(name == prefix or name.startswith(f"{prefix}_") for prefix in prefix_set):
                roots.append(child)
    except Exception:
        return []
    return sorted(roots, key=lambda p: p.name)


def _interval_from_table_root(root: Path, prefixes: List[str]) -> Optional[int]:
    name = str(Path(root).name)
    for prefix in sorted((str(p) for p in prefixes if str(p)), key=len, reverse=True):
        if name == prefix:
            return None
        marker = f"{prefix}_"
        if not name.startswith(marker):
            continue
        raw = name[len(marker) :]
        try:
            return int(raw)
        except Exception:
            return None
    return None


def _source_parquet_root() -> Path:
    try:
        return require_pipeline_io(profile=selected_profile()).source_ohlcvt_root.resolve()
    except PathConfigError:
        raw = (
            os.getenv("PIPELINE_SOURCE_OHLCVT_ROOT")
            or os.getenv("PIPELINE_SOURCE_PARQUET_ROOT")
            or os.getenv("PIPELINE_PARQUET_ROOT")
        )
        if raw:
            return Path(raw).resolve()
        raise


_LATEST_PARTITION_MAX_TS_CACHE: Dict[str, Optional[int]] = {}


def _recursive_latest_partition_max_ts(root: Path) -> Optional[int]:
    cache_key = str(Path(root).resolve()).lower()
    if cache_key in _LATEST_PARTITION_MAX_TS_CACHE:
        return _LATEST_PARTITION_MAX_TS_CACHE[cache_key]
    base = Path(root)
    if not base.exists():
        _LATEST_PARTITION_MAX_TS_CACHE[cache_key] = None
        return None
    latest_key: Optional[tuple[int, int]] = None
    latest_dirs: List[Path] = []
    try:
        month_dirs = [path for path in base.rglob("month=*") if path.is_dir()]
    except Exception:
        _LATEST_PARTITION_MAX_TS_CACHE[cache_key] = None
        return None
    for month_dir in month_dirs:
        try:
            month = int(str(month_dir.name).split("=", 1)[1])
            year_dir = month_dir.parent
            year = int(str(year_dir.name).split("=", 1)[1])
        except Exception:
            continue
        key = (int(year), int(month))
        if latest_key is None or key > latest_key:
            latest_key = key
            latest_dirs = [month_dir]
        elif key == latest_key:
            latest_dirs.append(month_dir)
    if not latest_dirs:
        value = partition_max_ts(base, ts_column="ts")
        _LATEST_PARTITION_MAX_TS_CACHE[cache_key] = value
        return value
    max_ts: Optional[int] = None
    for month_dir in latest_dirs:
        for path in sorted(month_dir.glob("*.parquet"), key=lambda p: p.name.lower()):
            try:
                import pandas as pd

                frame = pd.read_parquet(path, columns=["ts"])
            except Exception:
                continue
            ts = pd.to_numeric(frame["ts"], errors="coerce").dropna().astype("int64") if "ts" in frame.columns else None
            if ts is None or ts.empty:
                continue
            cur = int(ts.max())
            max_ts = cur if max_ts is None else max(int(max_ts), cur)
    value = int(max_ts) if max_ts is not None else None
    _LATEST_PARTITION_MAX_TS_CACHE[cache_key] = value
    return value


def _asset_from_partition_path(path: Path) -> Optional[str]:
    for part in path.parts:
        if str(part).startswith("asset="):
            return str(part).split("=", 1)[1]
    return None


def _recursive_first_time_gap(root: Path, *, interval: int) -> Optional[str]:
    base = Path(root)
    if not base.exists():
        return None
    step = int(interval) * 60
    last_by_asset: dict[str, int] = {}
    try:
        parquet_files = sorted(base.rglob("*.parquet"), key=lambda p: str(p).lower())
    except Exception:
        return None
    for path in parquet_files:
        try:
            import pandas as pd

            try:
                frame = pd.read_parquet(path, columns=["asset", "ts"])
            except Exception:
                frame = pd.read_parquet(path, columns=["ts"])
        except Exception:
            continue
        if "ts" not in frame.columns or frame.empty:
            continue
        if "asset" not in frame.columns:
            asset_hint = _asset_from_partition_path(path)
            if asset_hint is None:
                continue
            frame["asset"] = str(asset_hint)
        frame = frame[["asset", "ts"]].copy()
        frame["asset"] = frame["asset"].astype(str)
        frame["ts"] = pd.to_numeric(frame["ts"], errors="coerce")
        frame = frame.dropna(subset=["ts"]).copy()
        if frame.empty:
            continue
        frame["ts"] = frame["ts"].astype("int64")
        frame = frame.sort_values(["asset", "ts"]).drop_duplicates(subset=["asset", "ts"], keep="last")
        for asset, ts_value in zip(frame["asset"], frame["ts"]):
            asset_key = str(asset)
            cur = int(ts_value)
            if cur % max(step, 1) != 0:
                return f"misaligned_output:{base.name}:asset={asset_key}:ts={cur}:step={step}"
            prev = last_by_asset.get(asset_key)
            if prev is not None and cur != int(prev) + int(step):
                return (
                    f"gap_output:{base.name}:asset={asset_key}:"
                    f"prev_ts={int(prev)}:ts={cur}:expected_step={step}"
                )
            last_by_asset[asset_key] = cur
    return None


def _tabular_disk_outputs_at_source_edge(contract_spec: Any, *, deep_gap_scan: bool = False) -> tuple[bool, str]:
    prefixes = list(getattr(contract_spec, "output_table_prefixes", []) or [])
    table_roots = _matching_output_table_roots(Path(contract_spec.output_root), prefixes)
    if not table_roots:
        return False, "no_output_tables"
    source_root = _source_parquet_root()
    checked = 0
    for table_root in table_roots:
        interval = _interval_from_table_root(table_root, prefixes)
        if interval is None:
            continue
        output_edge = _recursive_latest_partition_max_ts(table_root)
        if output_edge is None:
            return False, f"missing_output_edge:{table_root.name}"
        if bool(deep_gap_scan):
            gap_detail = _recursive_first_time_gap(table_root, interval=int(interval))
            if gap_detail is not None:
                return False, gap_detail
        source_edges = [
            edge
            for edge in (
                _recursive_latest_partition_max_ts(source_root / f"ohlcvt_{int(interval)}"),
                _recursive_latest_partition_max_ts(source_root / f"scalar_features_{int(interval)}"),
            )
            if edge is not None
        ]
        if not source_edges:
            return False, f"missing_source_edge:{int(interval)}"
        source_edge = min(int(edge) for edge in source_edges)
        if int(output_edge) < int(source_edge):
            return False, f"behind_source:{table_root.name}:{int(output_edge)}<{int(source_edge)}"
        checked += 1
    if checked <= 0:
        return False, "no_interval_tables"
    return True, f"tables_at_source_edge:{checked}"


def _disk_preflight_skip_reason(module_spec: ProductionModuleSpec, contract_spec: Any, *, deep_disk_preflight: bool = False) -> Optional[str]:
    contract_status, _, _ = validate_contract(contract_spec)
    if contract_status != "passed":
        return None
    if str(module_spec.family) == "tabular":
        ok, detail = _tabular_disk_outputs_at_source_edge(contract_spec, deep_gap_scan=bool(deep_disk_preflight))
        return f"disk_contract_passed,{detail}" if ok else None
    return "disk_contract_passed"


def _resume_contract_pass_status(prior_status: str) -> str:
    normalized = str(prior_status).strip().lower()
    if normalized in {"running", "failed", "contract_failed", "resume_rejected"}:
        return "completed"
    return "skipped"


def _should_halt_after_module(status: str) -> bool:
    return str(status).strip().lower() not in {"completed", "skipped"}


def _module_row_from_result(
    *,
    run_id: str,
    order_index: int,
    module_spec: ProductionModuleSpec,
    status: str,
    failure_type: Optional[str],
    resumed: bool,
    resume_rejected: bool,
    resume_rejected_reason: Optional[str],
    started_at: Optional[str],
    finished_at: Optional[str],
    telemetry_summary: Dict[str, Any],
    contract_result: Dict[str, Any],
) -> Dict[str, Any]:
    workload = dict(contract_result.get("workload_shape") or {})
    process_tree = dict(telemetry_summary.get("process_tree") or {})
    system = dict(telemetry_summary.get("system") or {})
    phase_peaks = dict(telemetry_summary.get("phase_peaks") or {})
    heuristics = list(telemetry_summary.get("heuristics") or [])
    stalled_warning = "stalled_warning" in heuristics
    return {
        "run_id": str(run_id),
        "module_key": str(module_spec.module_key),
        "family": str(module_spec.family),
        "order_index": int(order_index),
        "entrypoint": str(module_spec.entrypoint),
        "command": " ".join(telemetry_summary.get("command") or module_spec.command(telemetry_summary.get("python_exe", str(sys.executable)))),
        "status": str(status),
        "failure_type": failure_type,
        "resumed": bool(resumed),
        "resume_rejected": bool(resume_rejected),
        "resume_rejected_reason": resume_rejected_reason,
        "started_at": started_at,
        "finished_at": finished_at,
        "runtime_seconds": float(telemetry_summary.get("runtime_seconds", 0.0) or 0.0),
        "configured_workers": telemetry_summary.get("configured_runtime", {}).get("workers"),
        "configured_threads": telemetry_summary.get("configured_runtime", {}).get("threads"),
        "peak_child_count": int(process_tree.get("peak_child_count", 0) or 0),
        "peak_thread_count": int(process_tree.get("peak_thread_count", 0) or 0),
        "peak_rss_mb": float(process_tree.get("peak_rss_mb", 0.0) or 0.0),
        "avg_cpu_pct_tree": float(process_tree.get("avg_cpu_pct", 0.0) or 0.0),
        "peak_cpu_pct_tree": float(process_tree.get("peak_cpu_pct", 0.0) or 0.0),
        "avg_cpu_pct_system": float(system.get("avg_cpu_pct", 0.0) or 0.0),
        "peak_cpu_pct_system": float(system.get("peak_cpu_pct", 0.0) or 0.0),
        "avg_ram_pct_system": float(system.get("avg_ram_pct", 0.0) or 0.0),
        "peak_ram_pct_system": float(system.get("peak_ram_pct", 0.0) or 0.0),
        "rows_written": workload.get("rows_written"),
        "parts_written": workload.get("parts_written"),
        "assets_touched": workload.get("assets_touched"),
        "units_processed": workload.get("units_processed"),
        "units_skipped": workload.get("units_skipped"),
        "resumed_units": workload.get("resumed_units"),
        "peak_rss_phase": phase_peaks.get("peak_rss_phase"),
        "peak_thread_phase": phase_peaks.get("peak_thread_phase"),
        "peak_cpu_phase": phase_peaks.get("peak_cpu_phase"),
        "last_progress_age_seconds": float(telemetry_summary.get("last_progress_age_seconds", 0.0) or 0.0),
        "last_artifact_change_age_seconds": float(
            telemetry_summary.get("last_artifact_change_age_seconds", 0.0) or 0.0
        ),
        "log_size_bytes_final": int(telemetry_summary.get("log_size_bytes_final", 0) or 0),
        "artifact_file_count_final": int(telemetry_summary.get("artifact_file_count_final", 0) or 0),
        "stalled_warning": bool(stalled_warning),
        "contract_completed": bool(str(contract_result.get("status", "")).strip().lower() == "passed"),
        "heuristics": ",".join(str(item) for item in heuristics),
    }


def _write_contract_result(path: Path, contract_result: Dict[str, Any]) -> None:
    write_json_atomic(path, ensure_serializable(contract_result))


def _orchestrator_child_env(sandbox_roots: SandboxOutputRoots, io_config: Optional[PipelineIOConfig] = None) -> Dict[str, str]:
    if io_config is None:
        try:
            io_config = require_pipeline_io(profile=selected_profile())
        except PathConfigError:
            base_env = dict(os.environ)
        else:
            base_env = pipeline_io_env(io_config, os.environ)
    else:
        base_env = pipeline_io_env(io_config, os.environ)
    if not sandbox_roots.enabled:
        return base_env
    return sandbox_env_for_subprocess(sandbox_roots, base_env)


def _assert_sandbox_launch_paths(sandbox_roots: SandboxOutputRoots, run_root: Path, child_env: Dict[str, str]) -> None:
    if not sandbox_roots.enabled:
        return
    assert_write_allowed(run_root, "orchestrator run root", roots=sandbox_roots)
    sandbox_path_keys = (
        "PIPELINE_SANDBOX_OUTPUT_ROOT",
        "PIPELINE_SANDBOX_PARQUET_ROOT",
        "PIPELINE_SANDBOX_LOG_ROOT",
        "PIPELINE_SANDBOX_STATE_ROOT",
        "PIPELINE_SANDBOX_TMP_ROOT",
        "PIPELINE_SANDBOX_DIAGNOSTICS_ROOT",
        "PIPELINE_SANDBOX_MANIFEST_ROOT",
        "PIPELINE_SANDBOX_OPTUNA_ROOT",
        "PIPELINE_SANDBOX_CATBOOST_TRAIN_DIR",
        "PIPELINE_SANDBOX_REGIME_DEFINITION_ROOT",
        "PIPELINE_SANDBOX_RUNTIME_ARTIFACT_ROOT",
    )
    for key in sandbox_path_keys:
        value = child_env.get(key)
        if not value:
            raise RuntimeError(f"Missing sandbox subprocess path env: {key}")
        assert_write_allowed(Path(value), key, roots=sandbox_roots)
    assert_write_allowed(Path(child_env["PIPELINE_ROOT"]), "PIPELINE_ROOT", roots=sandbox_roots)
    assert_write_allowed(Path(child_env["CATBOOST_TRAIN_DIR"]), "CATBOOST_TRAIN_DIR", roots=sandbox_roots)


def _summarize_and_write_telemetry(
    *,
    module_spec: ProductionModuleSpec,
    telemetry_summary_path: Path,
    telemetry_summary: Dict[str, Any],
    configured_workers: Optional[int],
    configured_threads: Optional[int],
    runtime_seconds: float,
    python_exe: str,
    command: List[str],
) -> Dict[str, Any]:
    telemetry_summary["runtime_seconds"] = float(runtime_seconds)
    telemetry_summary["python_exe"] = str(python_exe)
    telemetry_summary["command"] = [str(part) for part in command]
    sample_count = int(telemetry_summary.get("sample_count", 0) or 0)
    heuristics: List[str]
    if sample_count <= 0:
        heuristics = ["telemetry_missing"]
    else:
        heuristics = derive_heuristics(
            {
                "runtime_seconds": float(runtime_seconds),
                "avg_cpu_pct_tree": float(telemetry_summary.get("process_tree", {}).get("avg_cpu_pct", 0.0) or 0.0),
                "peak_rss_mb": float(telemetry_summary.get("process_tree", {}).get("peak_rss_mb", 0.0) or 0.0),
                "peak_child_count": int(telemetry_summary.get("process_tree", {}).get("peak_child_count", 0) or 0),
                "peak_thread_count": int(
                    telemetry_summary.get("process_tree", {}).get("peak_thread_count", 0) or 0
                ),
                "last_progress_age_seconds": float(
                    telemetry_summary.get("last_progress_age_seconds", 0.0) or 0.0
                ),
                "last_artifact_change_age_seconds": float(
                    telemetry_summary.get("last_artifact_change_age_seconds", 0.0) or 0.0
                ),
            },
            configured_workers=configured_workers,
            configured_threads=configured_threads,
        )
    telemetry_summary.setdefault(
        "configured_runtime",
        {"workers": configured_workers, "threads": configured_threads},
    )
    telemetry_summary["heuristics"] = heuristics
    _write_contract_result(telemetry_summary_path, telemetry_summary)
    return telemetry_summary


def run_orchestrator(args: OrchestratorArgs) -> Path:
    io_config = require_pipeline_io(profile=args.profile)
    sandbox_roots = resolve_sandbox_output_roots(args)
    run_root = resolve_run_root(
        project_root=args.project_root.resolve(),
        output_dir=args.output_dir,
        run_id=args.run_id,
        resume_run=args.resume_run,
        no_resume_latest=bool(args.no_resume_latest),
    )
    child_env = _orchestrator_child_env(sandbox_roots, io_config)
    _assert_sandbox_launch_paths(sandbox_roots, run_root, child_env)
    run_root.mkdir(parents=True, exist_ok=True)
    run_lock_path = _acquire_run_lock(run_root)
    orchestrator_log = run_root / "orchestrator.log"
    manifest = _load_or_init_manifest(run_root, args)
    append_log_line(orchestrator_log, f"orchestrator start run_root={run_root}")
    if manifest.get("run_id") != run_root.name:
        manifest["run_id"] = run_root.name
    if manifest.get("resume_source_run_id") is None and manifest.get("modules"):
        manifest["resume_source_run_id"] = run_root.name
    if args.resume_run:
        manifest["status"] = "running"
        manifest["finished_at"] = None
        manifest.pop("halt_reason", None)
    _write_manifest(run_root, manifest)

    module_rows: List[Dict[str, Any]] = []
    previous_modules = {
        str(item.get("module_key")): item for item in manifest.get("modules", []) if isinstance(item, dict)
    }
    git_head = maybe_git_head(args.project_root.resolve())

    for order_index, module_spec in enumerate(mature_ml_modules(), start=1):
        module_dir = _module_dir(run_root, module_spec.module_key)
        module_dir.mkdir(parents=True, exist_ok=True)
        module_log = module_dir / "module.log"
        contract_snapshot_path = module_dir / "contract_snapshot.json"
        contract_validation_path = module_dir / "contract_validation.json"
        telemetry_samples_path = module_dir / "telemetry_samples.csv"
        telemetry_summary_path = module_dir / "telemetry_summary.json"

        command = module_spec.command(args.python_exe)
        contract_spec = build_contract_spec(module_spec, env=child_env)
        runtime_contract_spec = _runtime_contract_spec(contract_spec, module_log)
        snapshot = snapshot_payload(
            module_spec,
            contract_spec,
            command=command,
            cwd=args.project_root.resolve(),
            git_head=git_head,
            env=child_env,
        )
        signature = command_signature(
            {
                "module_key": module_spec.module_key,
                "command": command,
                "output_root": snapshot["output_root"],
                "runtime_config": snapshot["runtime_config"],
            }
        )
        snapshot["command_signature"] = signature
        write_json_atomic(contract_snapshot_path, ensure_serializable(snapshot))

        configured_workers = _configured_workers(contract_spec)
        configured_threads = _configured_threads(contract_spec)
        previous = previous_modules.get(module_spec.module_key, {})
        predecessor_ok = True
        resume_rejected = False
        resume_rejected_reason = None
        resumed = False
        prior_status = str(previous.get("status", "")).strip().lower()

        if previous:
            compatible, reject_reason = _basic_resume_compatibility(previous, snapshot, predecessor_ok)
            if not compatible:
                resume_rejected = True
                resume_rejected_reason = reject_reason
                append_log_line(
                    orchestrator_log,
                    f"module={module_spec.module_key} resume_rejected reason={reject_reason}",
                )
            else:
                if prior_status in {"running", "failed", "contract_failed", "resume_rejected"}:
                    resumed = True
                if prior_status in {"completed", "skipped", "running", "failed", "contract_failed", "resume_rejected"}:
                    previous_contract = load_json_dict(contract_validation_path)
                    contract_status, contract_failure_type, contract_result = validate_contract(runtime_contract_spec)
                    _write_contract_result(contract_validation_path, contract_result)
                    if contract_status == "passed":
                        started_at = str(previous.get("started_at") or utc_now_iso())
                        finished_at = utc_now_iso()
                        telemetry_summary = load_json_dict(telemetry_summary_path)
                        if not telemetry_summary:
                            telemetry_summary = _empty_telemetry_summary(
                                module_spec.module_key,
                                configured_workers,
                                configured_threads,
                            )
                        status_on_resume = _resume_contract_pass_status(prior_status)
                        telemetry_summary = _summarize_and_write_telemetry(
                            module_spec=module_spec,
                            telemetry_summary_path=telemetry_summary_path,
                            telemetry_summary=telemetry_summary,
                            configured_workers=configured_workers,
                            configured_threads=configured_threads,
                            runtime_seconds=float(telemetry_summary.get("runtime_seconds", 0.0) or 0.0),
                            python_exe=args.python_exe,
                            command=command,
                        )
                        module_state = {
                            "module_key": module_spec.module_key,
                            "family": module_spec.family,
                            "order_index": int(order_index),
                            "entrypoint": str(module_spec.entrypoint),
                            "command": command,
                            "status": status_on_resume,
                            "failure_type": None,
                            "active": False,
                            "resumed": bool(resumed),
                            "resume_rejected": False,
                            "resume_rejected_reason": None,
                            "started_at": started_at,
                            "finished_at": finished_at,
                            "runtime_seconds": float(telemetry_summary.get("runtime_seconds", 0.0) or 0.0),
                            "command_signature": signature,
                            "contract_snapshot": snapshot,
                            "contract_snapshot_path": str(contract_snapshot_path.relative_to(run_root)),
                            "contract_validation_path": str(contract_validation_path.relative_to(run_root)),
                            "telemetry_summary_path": str(telemetry_summary_path.relative_to(run_root)),
                            "telemetry_samples_path": str(telemetry_samples_path.relative_to(run_root)),
                            "status_history": list(previous.get("status_history") or []) + [status_on_resume],
                            "previous_contract_status": previous_contract.get("status"),
                        }
                        _replace_module_state(manifest, module_state)
                        _write_manifest(run_root, manifest)
                        append_log_line(orchestrator_log, f"module={module_spec.module_key} status={status_on_resume}")
                        _print_terminal_progress(
                            f"[orchestrator] module={module_spec.module_key} status={status_on_resume}"
                        )
                        module_rows.append(
                            _module_row_from_result(
                                run_id=run_root.name,
                                order_index=order_index,
                                module_spec=module_spec,
                                status=status_on_resume,
                                failure_type=None,
                                resumed=resumed,
                                resume_rejected=False,
                                resume_rejected_reason=None,
                                started_at=started_at,
                                finished_at=finished_at,
                                telemetry_summary=telemetry_summary,
                                contract_result=contract_result,
                            )
                        )
                        continue
                    resume_rejected = True
                    resume_rejected_reason = contract_failure_type or "stale_resume_state"

        disk_skip_reason = _disk_preflight_skip_reason(
            module_spec,
            contract_spec,
            deep_disk_preflight=bool(args.deep_disk_preflight),
        )
        if disk_skip_reason is not None:
            started_at = utc_now_iso()
            finished_at = started_at
            contract_status, contract_failure_type, contract_result = validate_contract(contract_spec)
            _write_contract_result(contract_validation_path, contract_result)
            telemetry_summary = _empty_telemetry_summary(
                module_spec.module_key,
                configured_workers,
                configured_threads,
            )
            telemetry_summary["runtime_seconds"] = 0.0
            telemetry_summary["python_exe"] = str(args.python_exe)
            telemetry_summary["command"] = command
            telemetry_summary["heuristics"] = ["disk_preflight_skip"]
            telemetry_summary["disk_preflight_reason"] = str(disk_skip_reason)
            _write_contract_result(telemetry_summary_path, telemetry_summary)
            module_state = {
                "module_key": module_spec.module_key,
                "family": module_spec.family,
                "order_index": int(order_index),
                "entrypoint": str(module_spec.entrypoint),
                "command": command,
                "status": "skipped",
                "failure_type": None,
                "active": False,
                "resumed": False,
                "resume_rejected": False,
                "resume_rejected_reason": None,
                "started_at": started_at,
                "finished_at": finished_at,
                "runtime_seconds": 0.0,
                "command_signature": signature,
                "contract_snapshot": snapshot,
                "contract_snapshot_path": str(contract_snapshot_path.relative_to(run_root)),
                "contract_validation_path": str(contract_validation_path.relative_to(run_root)),
                "telemetry_summary_path": str(telemetry_summary_path.relative_to(run_root)),
                "telemetry_samples_path": str(telemetry_samples_path.relative_to(run_root)),
                "status_history": ["skipped"],
                "disk_preflight_reason": str(disk_skip_reason),
                "contract_status": contract_status,
                "contract_failure_type": contract_failure_type,
            }
            _replace_module_state(manifest, module_state)
            _write_manifest(run_root, manifest)
            append_log_line(orchestrator_log, f"module={module_spec.module_key} status=skipped reason={disk_skip_reason}")
            _print_terminal_progress(
                f"[orchestrator] module={module_spec.module_key} status=skipped reason=disk_preflight"
            )
            module_rows.append(
                _module_row_from_result(
                    run_id=run_root.name,
                    order_index=order_index,
                    module_spec=module_spec,
                    status="skipped",
                    failure_type=None,
                    resumed=False,
                    resume_rejected=False,
                    resume_rejected_reason=None,
                    started_at=started_at,
                    finished_at=finished_at,
                    telemetry_summary=telemetry_summary,
                    contract_result=contract_result,
                )
            )
            continue

        started_at = utc_now_iso()
        module_state = {
            "module_key": module_spec.module_key,
            "family": module_spec.family,
            "order_index": int(order_index),
            "entrypoint": str(module_spec.entrypoint),
            "command": command,
            "status": "running",
            "failure_type": None,
            "active": True,
            "resumed": bool(resumed),
            "resume_rejected": bool(resume_rejected),
            "resume_rejected_reason": resume_rejected_reason,
            "started_at": started_at,
            "finished_at": None,
            "runtime_seconds": None,
            "command_signature": signature,
            "contract_snapshot": snapshot,
            "contract_snapshot_path": str(contract_snapshot_path.relative_to(run_root)),
            "contract_validation_path": str(contract_validation_path.relative_to(run_root)),
            "telemetry_summary_path": str(telemetry_summary_path.relative_to(run_root)),
            "telemetry_samples_path": str(telemetry_samples_path.relative_to(run_root)),
            "status_history": ["resume_rejected" if resume_rejected else ("resumed" if resumed else "running")],
        }
        _replace_module_state(manifest, module_state)
        _write_manifest(run_root, manifest)
        append_log_line(orchestrator_log, f"module={module_spec.module_key} launch command={' '.join(command)}")
        _print_terminal_progress(
            f"[orchestrator] module={module_spec.module_key} status=running order={order_index}/{len(mature_ml_modules())}"
        )

        final_status = "completed"
        failure_type: Optional[str] = None
        runtime_seconds = 0.0
        contract_result: Dict[str, Any] = {
            "schema_version": 1,
            "status": "failed",
            "failure_type": "process_exit",
            "checks": [],
            "workload_shape": {},
        }
        recorder: Optional[TelemetryRecorder] = None
        launch_started = time.monotonic()

        try:
            with module_log.open("a", encoding="utf-8") as handle:
                handle.write(f"[{utc_now_iso()}] RUN {' '.join(command)}\n")
                handle.flush()
                process = subprocess.Popen(
                    command,
                    cwd=str(args.project_root.resolve()),
                    env=child_env,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                recorder = TelemetryRecorder(
                    module_key=module_spec.module_key,
                    log_path=module_log,
                    anchor_roots=contract_spec.anchor_roots,
                    sample_interval_seconds=float(max(5.0, args.sample_seconds)),
                    pid=int(process.pid),
                    sample_callback=lambda sample, mk=module_spec.module_key: _print_terminal_progress(
                        f"[orchestrator] module={mk} elapsed={float(sample.get('elapsed_s', 0.0) or 0.0):.0f}s "
                        f"phase={sample.get('phase', 'unknown')} children={int(sample.get('child_count', 0) or 0)} "
                        f"threads={int(sample.get('thread_count_tree', 0) or 0)} "
                        f"rss_mb={float(sample.get('rss_mb_tree', 0.0) or 0.0):.1f} "
                        f"tree_cpu={float(sample.get('cpu_pct_tree', 0.0) or 0.0):.1f}% "
                        f"sys_cpu={float(sample.get('sys_cpu_pct', 0.0) or 0.0):.1f}%"
                    ),
                )
                recorder.loop_until_exit(process)
                returncode = int(process.wait())
                runtime_seconds = max(0.0, time.monotonic() - launch_started)
                handle.write(f"[{utc_now_iso()}] EXIT {returncode}\n")
                handle.flush()
                if returncode != 0:
                    final_status = "failed"
                    failure_type = "process_exit"
        except Exception as exc:
            runtime_seconds = max(0.0, time.monotonic() - launch_started)
            final_status = "failed"
            failure_type = "process_exit"
            append_log_line(orchestrator_log, f"module={module_spec.module_key} process_error={exc}")

        if recorder is not None:
            telemetry_summary = recorder.write_outputs(
                samples_path=telemetry_samples_path,
                summary_path=telemetry_summary_path,
                runtime_seconds=runtime_seconds,
                configured_workers=configured_workers,
                configured_threads=configured_threads,
                final_phase_durations={},
            )
        else:
            telemetry_summary = _empty_telemetry_summary(
                module_spec.module_key,
                configured_workers,
                configured_threads,
            )

        if final_status != "failed":
            contract_status, contract_failure_type, contract_result = validate_contract(runtime_contract_spec)
            _write_contract_result(contract_validation_path, contract_result)
            if contract_status != "passed":
                final_status = "contract_failed"
                failure_type = contract_failure_type or "contract_failed"
        else:
            contract_result["failure_type"] = failure_type
            _write_contract_result(contract_validation_path, contract_result)

        telemetry_summary = _summarize_and_write_telemetry(
            module_spec=module_spec,
            telemetry_summary_path=telemetry_summary_path,
            telemetry_summary=telemetry_summary,
            configured_workers=configured_workers,
            configured_threads=configured_threads,
            runtime_seconds=runtime_seconds,
            python_exe=args.python_exe,
            command=command,
        )
        finished_at = utc_now_iso()

        module_state.update(
            {
                "status": final_status,
                "failure_type": failure_type,
                "active": False,
                "finished_at": finished_at,
                "runtime_seconds": float(runtime_seconds),
                "status_history": [*module_state["status_history"], final_status],
            }
        )
        _replace_module_state(manifest, module_state)
        _write_manifest(run_root, manifest)
        append_log_line(
            orchestrator_log,
            f"module={module_spec.module_key} status={final_status} failure_type={failure_type}",
        )
        _print_terminal_progress(
            f"[orchestrator] module={module_spec.module_key} status={final_status} runtime_s={runtime_seconds:.1f}"
        )
        module_rows.append(
            _module_row_from_result(
                run_id=run_root.name,
                order_index=order_index,
                module_spec=module_spec,
                status=final_status,
                failure_type=failure_type,
                resumed=resumed,
                resume_rejected=resume_rejected,
                resume_rejected_reason=resume_rejected_reason,
                started_at=started_at,
                finished_at=finished_at,
                telemetry_summary=telemetry_summary,
                contract_result=contract_result,
            )
        )
        if _should_halt_after_module(final_status):
            manifest["halt_reason"] = {
                "module_key": str(module_spec.module_key),
                "status": str(final_status),
                "failure_type": failure_type,
            }
            _write_manifest(run_root, manifest)
            append_log_line(
                orchestrator_log,
                f"orchestrator halted module={module_spec.module_key} status={final_status} failure_type={failure_type}",
            )
            _print_terminal_progress(
                f"[orchestrator] halted_after module={module_spec.module_key} status={final_status}"
            )
            break

    overall_status = "completed" if all(
        row["status"] in {"completed", "skipped"} for row in module_rows
    ) else "failed"
    manifest["generated_at"] = utc_now_iso()
    manifest["finished_at"] = utc_now_iso()
    manifest["status"] = overall_status
    _write_manifest(run_root, manifest)
    _write_module_summary_csv(run_root, module_rows)
    _write_module_summary_json(run_root, module_rows)
    append_log_line(orchestrator_log, f"orchestrator finished status={overall_status}")
    _print_terminal_progress(f"[orchestrator] status={overall_status} run_root={run_root}")
    _enforce_retention(run_root.parent)
    _release_run_lock(run_lock_path)
    return run_root
