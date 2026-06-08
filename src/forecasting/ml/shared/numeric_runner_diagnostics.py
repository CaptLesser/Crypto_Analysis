from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from src.forecasting.common.concurrency import THREAD_ENV_VARS, effective_thread_snapshot
from src.forecasting.common.test_diagnostics import TestDiagnosticPacket

try:
    import psutil
except Exception:
    psutil = None


def diagnostics_file(state_root: Path, branch: str) -> Path:
    return Path(state_root) / f"{str(branch)}_run_diagnostics.jsonl"


WORKER_RESOURCE_TELEMETRY_SCHEMA_VERSION = 1


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    return str(value)


def resource_snapshot(*, include_thread_snapshot: bool = False) -> Dict[str, Any]:
    snapshot: Dict[str, Any] = {
        "monotonic_s": float(time.monotonic()),
        "pid": int(os.getpid()),
        "parent_pid": int(os.getppid()) if hasattr(os, "getppid") else None,
        "env_thread_caps": {name: os.environ.get(name) for name in THREAD_ENV_VARS},
    }
    if psutil is None:
        if include_thread_snapshot:
            snapshot["thread_snapshot"] = effective_thread_snapshot()
        return snapshot
    try:
        proc = psutil.Process(os.getpid())
        mem = proc.memory_info()
        io_counters = None
        try:
            io_counters = proc.io_counters()
        except Exception:
            io_counters = None
        snapshot.update(
            {
                "rss_mb": round(float(mem.rss) / 1024.0 / 1024.0, 3),
                "vms_mb": round(float(mem.vms) / 1024.0 / 1024.0, 3),
                "process_cpu_pct": float(proc.cpu_percent(interval=None)),
                "system_cpu_pct": float(psutil.cpu_percent(interval=None)),
                "system_ram_pct": float(psutil.virtual_memory().percent),
                "thread_count": int(proc.num_threads()),
            }
        )
        try:
            full = proc.memory_full_info()
            uss = getattr(full, "uss", None)
            if uss is not None:
                snapshot["uss_mb"] = round(float(uss) / 1024.0 / 1024.0, 3)
        except Exception:
            pass
        if io_counters is not None:
            snapshot["io"] = {
                "read_mb": round(float(getattr(io_counters, "read_bytes", 0)) / 1024.0 / 1024.0, 3),
                "write_mb": round(float(getattr(io_counters, "write_bytes", 0)) / 1024.0 / 1024.0, 3),
            }
    except Exception:
        pass
    if include_thread_snapshot:
        snapshot["thread_snapshot"] = effective_thread_snapshot()
    return snapshot


def worker_resource_telemetry_record(
    *,
    module: str,
    run_id: str,
    task_id: str,
    work_unit_id: Optional[str] = None,
    status: str = "completed",
    start_time_utc: Optional[str] = None,
    end_time_utc: Optional[str] = None,
    start_epoch_s: Optional[float] = None,
    end_epoch_s: Optional[float] = None,
    submitted_epoch_s: Optional[float] = None,
    resource_start: Optional[Mapping[str, Any]] = None,
    resource_end: Optional[Mapping[str, Any]] = None,
    phase_timings: Optional[Mapping[str, Any]] = None,
    identity: Optional[Mapping[str, Any]] = None,
    error: Optional[BaseException | str] = None,
) -> Dict[str, Any]:
    """Build the shared bounded worker/resource telemetry record.

    Children return this payload to the parent, which writes it through the
    existing diagnostics JSONL path. That keeps file writes centralized while
    still proving real worker process state.
    """
    start_resource = dict(resource_start or {})
    end_resource = dict(resource_end or {})
    start_epoch = float(start_epoch_s) if start_epoch_s is not None else None
    end_epoch = float(end_epoch_s) if end_epoch_s is not None else None
    duration_s = None
    if start_epoch is not None and end_epoch is not None:
        duration_s = max(0.0, end_epoch - start_epoch)
    queue_wait_s = None
    if submitted_epoch_s is not None and start_epoch is not None:
        queue_wait_s = max(0.0, float(start_epoch) - float(submitted_epoch_s))
    error_payload: Dict[str, Any] = {}
    if error is not None:
        if isinstance(error, BaseException):
            error_payload = {"error_type": type(error).__name__, "error_message": str(error)[:500]}
        else:
            error_payload = {"error_type": "error", "error_message": str(error)[:500]}
    record: Dict[str, Any] = {
        "schema_version": WORKER_RESOURCE_TELEMETRY_SCHEMA_VERSION,
        "module": str(module),
        "run_id": str(run_id),
        "task_id": str(task_id),
        "work_unit_id": str(work_unit_id or task_id),
        "worker_pid": int(end_resource.get("pid") or start_resource.get("pid") or os.getpid()),
        "parent_pid": end_resource.get("parent_pid") or start_resource.get("parent_pid"),
        "start_time_utc": str(start_time_utc or ""),
        "end_time_utc": str(end_time_utc or ""),
        "duration_s": round(float(duration_s), 6) if duration_s is not None else None,
        "elapsed_s": round(float(duration_s), 6) if duration_s is not None else None,
        "queue_wait_s": round(float(queue_wait_s), 6) if queue_wait_s is not None else None,
        "status": str(status),
        "process_thread_count": end_resource.get("thread_count") or start_resource.get("thread_count"),
        "rss_mb": end_resource.get("rss_mb") or start_resource.get("rss_mb"),
        "uss_mb": end_resource.get("uss_mb") or start_resource.get("uss_mb"),
        "env_thread_caps": end_resource.get("env_thread_caps") or start_resource.get("env_thread_caps") or {},
        "threadpool_snapshot": (
            (end_resource.get("thread_snapshot") or {}).get("threadpool_info")
            if isinstance(end_resource.get("thread_snapshot"), Mapping)
            else None
        )
        or (
            (start_resource.get("thread_snapshot") or {}).get("threadpool_info")
            if isinstance(start_resource.get("thread_snapshot"), Mapping)
            else None
        ),
        "numba_num_threads": (
            (end_resource.get("thread_snapshot") or {}).get("numba_num_threads")
            if isinstance(end_resource.get("thread_snapshot"), Mapping)
            else None
        )
        or (
            (start_resource.get("thread_snapshot") or {}).get("numba_num_threads")
            if isinstance(start_resource.get("thread_snapshot"), Mapping)
            else None
        ),
        "pyarrow_cpu_count": (
            (end_resource.get("thread_snapshot") or {}).get("pyarrow_cpu_count")
            if isinstance(end_resource.get("thread_snapshot"), Mapping)
            else None
        )
        or (
            (start_resource.get("thread_snapshot") or {}).get("pyarrow_cpu_count")
            if isinstance(start_resource.get("thread_snapshot"), Mapping)
            else None
        ),
        "pyarrow_io_thread_count": (
            (end_resource.get("thread_snapshot") or {}).get("pyarrow_io_thread_count")
            if isinstance(end_resource.get("thread_snapshot"), Mapping)
            else None
        )
        or (
            (start_resource.get("thread_snapshot") or {}).get("pyarrow_io_thread_count")
            if isinstance(start_resource.get("thread_snapshot"), Mapping)
            else None
        ),
        "resource_start": start_resource,
        "resource_end": end_resource,
        "phase_timings": dict(phase_timings or {}),
    }
    record.update(dict(identity or {}))
    record.update(error_payload)
    return _json_safe(record)


def append_diagnostic_event(path: Path, event: str, payload: Dict[str, Any], *, timestamp_fn: Any = None) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        ts_utc = timestamp_fn() if callable(timestamp_fn) else None
        row = {"ts_utc": ts_utc, "event": str(event), **dict(payload)}
        if row["ts_utc"] is None:
            row.pop("ts_utc", None)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True, default=str) + "\n")
    except Exception:
        return


def reset_diagnostics_file(path: Path) -> None:
    try:
        Path(path).unlink()
    except FileNotFoundError:
        return
    except Exception:
        return


def iter_diagnostic_events(path: Path) -> Iterable[Dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except Exception:
                continue
            if isinstance(payload, dict):
                yield payload


def _combo_key(row: Dict[str, Any]) -> Tuple[int, int, str]:
    return (int(row.get("interval", 0) or 0), int(row.get("horizon_minutes", 0) or 0), str(row.get("task", "")))


def summarize_diagnostics(path: Path, *, top_n: int = 10) -> Dict[str, Any]:
    n = max(1, int(top_n))
    event_total = 0
    event_counts: Dict[str, int] = {}
    combo_count = 0
    shard_count = 0
    writer_event_count = 0
    worker_event_count = 0
    max_rss_mb = 0.0
    max_worker_rss_mb = 0.0
    max_worker_uss_mb = 0.0
    max_worker_thread_count = 0
    max_system_ram_pct = 0.0
    latest_writer_stats: Dict[str, Any] = {}

    combo_elapsed: Dict[Tuple[int, int, str], float] = {}
    combo_rows_out: Dict[Tuple[int, int, str], int] = {}
    combo_done_units: Dict[Tuple[int, int, str], int] = {}
    combo_unit_elapsed: Dict[Tuple[int, int, str], float] = {}
    combo_unit_count: Dict[Tuple[int, int, str], int] = {}
    slowest_shards_ranked: List[Tuple[int, Dict[str, Any]]] = []
    slowest_units_ranked: List[Tuple[int, Dict[str, Any]]] = []
    slowest_worker_tasks_ranked: List[Tuple[int, Dict[str, Any]]] = []

    def _trim_slowest(rows: List[Tuple[int, Dict[str, Any]]]) -> List[Tuple[int, Dict[str, Any]]]:
        rows.sort(key=lambda item: (-float(item[1].get("elapsed_s", 0.0) or 0.0), int(item[0])))
        return rows[:n]

    for row in iter_diagnostic_events(path):
        seq = int(event_total)
        event_total += 1
        event = str(row.get("event", "unknown"))
        event_counts[event] = int(event_counts.get(event, 0)) + 1
        if event == "combo_complete":
            combo_count += 1
        if event.startswith("writer_"):
            writer_event_count += 1
        if event == "worker_resource":
            worker_event_count += 1
            telemetry = row.get("worker_telemetry") if isinstance(row.get("worker_telemetry"), dict) else row
            if isinstance(telemetry, dict):
                max_worker_rss_mb = max(max_worker_rss_mb, float(telemetry.get("rss_mb", 0.0) or 0.0))
                max_worker_uss_mb = max(max_worker_uss_mb, float(telemetry.get("uss_mb", 0.0) or 0.0))
                max_worker_thread_count = max(max_worker_thread_count, int(telemetry.get("process_thread_count", 0) or 0))
                slowest_worker_tasks_ranked.append((seq, dict(telemetry)))
                slowest_worker_tasks_ranked = _trim_slowest(slowest_worker_tasks_ranked)
        stats = row.get("writer_stats")
        if isinstance(stats, dict):
            latest_writer_stats = dict(stats)
        resource = row.get("resource")
        if isinstance(resource, dict):
            max_rss_mb = max(max_rss_mb, float(resource.get("rss_mb", 0.0) or 0.0))
            max_system_ram_pct = max(max_system_ram_pct, float(resource.get("system_ram_pct", 0.0) or 0.0))
        if event != "shard_finished":
            continue
        shard_count += 1
        key = _combo_key(row)
        elapsed_s = float(row.get("elapsed_s", 0.0) or 0.0)
        forecast_rows = int(row.get("forecast_rows", 0) or 0)
        done_units = int(row.get("done_units", 0) or 0)
        combo_elapsed[key] = combo_elapsed.get(key, 0.0) + elapsed_s
        combo_rows_out[key] = combo_rows_out.get(key, 0) + forecast_rows
        combo_done_units[key] = combo_done_units.get(key, 0) + done_units
        unit_summary = row.get("unit_elapsed_summary") if isinstance(row.get("unit_elapsed_summary"), dict) else {}
        unit_count = int(unit_summary.get("count", 0) or 0)
        unit_mean = float(unit_summary.get("mean_s", 0.0) or 0.0)
        if unit_count > 0 and unit_mean > 0.0:
            combo_unit_count[key] = combo_unit_count.get(key, 0) + unit_count
            combo_unit_elapsed[key] = combo_unit_elapsed.get(key, 0.0) + (unit_mean * float(unit_count))
        slowest_shards_ranked.append((seq, row))
        slowest_shards_ranked = _trim_slowest(slowest_shards_ranked)
        slowest_units = row.get("slowest_units") if isinstance(row.get("slowest_units"), list) else []
        if not slowest_units and isinstance(row.get("slowest_unit"), dict):
            slowest_units = [row.get("slowest_unit")]
        for unit_idx, unit in enumerate(slowest_units):
            if not isinstance(unit, dict):
                continue
            slowest_units_ranked.append(
                (
                    seq * 1000 + int(unit_idx),
                    {
                        "interval": key[0],
                        "horizon_minutes": key[1],
                        "task": key[2],
                        "shard_index": int(row.get("shard_index", 0) or 0),
                        **dict(unit),
                    },
                )
            )
        slowest_units_ranked = _trim_slowest(slowest_units_ranked)

    slowest_shards = [row for _seq, row in slowest_shards_ranked]
    slowest_combos = sorted(
        (
            {
                "interval": key[0],
                "horizon_minutes": key[1],
                "task": key[2],
                "shard_elapsed_s": round(float(elapsed), 3),
                "forecast_rows": int(combo_rows_out.get(key, 0)),
                "done_units": int(combo_done_units.get(key, 0)),
                "shard_seconds_per_forecast_row": (
                    round(float(elapsed) / float(combo_rows_out.get(key, 0)), 6)
                    if int(combo_rows_out.get(key, 0)) > 0
                    else None
                ),
                "mean_unit_elapsed_s": (
                    round(float(combo_unit_elapsed.get(key, 0.0)) / float(combo_unit_count.get(key, 0)), 3)
                    if int(combo_unit_count.get(key, 0)) > 0
                    else None
                ),
            }
            for key, elapsed in combo_elapsed.items()
        ),
        key=lambda row: float(row["shard_elapsed_s"]),
        reverse=True,
    )[:n]

    return {
        "path": str(Path(path)),
        "events": int(event_total),
        "event_counts": event_counts,
        "combo_count": int(combo_count),
        "shard_count": int(shard_count),
        "writer_event_count": int(writer_event_count),
        "worker_event_count": int(worker_event_count),
        "max_rss_mb": round(float(max_rss_mb), 3),
        "max_worker_rss_mb": round(float(max_worker_rss_mb), 3),
        "max_worker_uss_mb": round(float(max_worker_uss_mb), 3),
        "max_worker_thread_count": int(max_worker_thread_count),
        "max_system_ram_pct": round(float(max_system_ram_pct), 3),
        "latest_writer_stats": latest_writer_stats,
        "slowest_shards": [
            {
                "interval": int(row.get("interval", 0) or 0),
                "horizon_minutes": int(row.get("horizon_minutes", 0) or 0),
                "task": str(row.get("task", "")),
                "shard_index": int(row.get("shard_index", 0) or 0),
                "assets": int(row.get("assets", 0) or 0),
                "done_units": int(row.get("done_units", 0) or 0),
                "skipped_units": int(row.get("skipped_units", 0) or 0),
                "forecast_rows": int(row.get("forecast_rows", 0) or 0),
                "eval_rows": int(row.get("eval_rows", 0) or 0),
                "elapsed_s": round(float(row.get("elapsed_s", 0.0) or 0.0), 3),
                "slowest_unit": row.get("slowest_unit") if isinstance(row.get("slowest_unit"), dict) else {},
            }
            for row in slowest_shards
        ],
        "slowest_combos": slowest_combos,
        "slowest_units": [row for _seq, row in slowest_units_ranked],
        "slowest_worker_tasks": [row for _seq, row in slowest_worker_tasks_ranked],
    }


def emit_standard_numeric_diagnostic_packet(
    *,
    packet_root: Path,
    run_result: Mapping[str, Any],
    diagnostics_summary: Optional[Mapping[str, Any]] = None,
    mode: str = "test",
    module_name: str = "numeric_runner",
    run_id: Optional[str] = None,
    max_events: int = 100,
    max_samples: int = 100,
    max_top_offenders: int = 25,
) -> Dict[str, str]:
    """Emit a bounded standard Test diagnostic packet from existing numeric artifacts.

    This is additive: callers keep their existing run summary, JSONL diagnostics, and
    handoff manifests, and store the standard packet under a separate packet root.
    """
    result = dict(run_result)
    paths = dict(result.get("paths") or {})
    diagnostics = dict(diagnostics_summary or {})
    diagnostics_path = str(result.get("diagnostics_jsonl") or paths.get("diagnostics_jsonl") or "").strip()
    if not diagnostics:
        if diagnostics_path:
            try:
                diagnostics = summarize_diagnostics(Path(diagnostics_path), top_n=max_top_offenders)
            except Exception:
                diagnostics = {"path": diagnostics_path, "exists": False}

    packet = TestDiagnosticPacket.create(
        Path(packet_root),
        module_name=str(module_name),
        run_id=str(run_id or result.get("run_id") or Path(packet_root).parent.name),
        mode=str(mode),
        max_events=max_events,
        max_samples=max_samples,
        max_top_offenders=max_top_offenders,
    )
    config = dict(result.get("config") or {})
    resources = dict(result.get("resources") or {})
    concurrency = dict(result.get("concurrency") or {})
    writer_stats = dict(result.get("writer_stats") or diagnostics.get("latest_writer_stats") or {})
    packet.record_event(
        "numeric_run_summary",
        success=bool(result.get("success", True)),
        return_code=result.get("return_code"),
        runtime_profile=result.get("runtime_profile"),
        training_window_label=result.get("training_window_label"),
        wall_clock_s=(result.get("timing") or {}).get("wall_clock_s"),
        runner_diagnostics=diagnostics,
        writer_stats=writer_stats,
    )
    packet.record_sample(
        sample_type="resource_summary",
        name="process_tree",
        peak_proc_tree_rss_mb=resources.get("peak_proc_tree_rss_mb"),
        peak_proc_threads=resources.get("peak_proc_threads"),
        peak_cpu_total_pct=resources.get("peak_cpu_total_pct"),
        read_mb_total=resources.get("read_mb_total"),
        write_mb_total=resources.get("write_mb_total"),
    )
    packet.record_sample(
        sample_type="concurrency",
        name="requested_effective",
        requested_workers=config.get("unit_workers") or result.get("workers"),
        requested_model_threads=config.get("model_threads") or result.get("model_threads"),
        effective_workers=concurrency.get("effective_workers") or config.get("unit_workers") or result.get("workers"),
        effective_model_threads=concurrency.get("effective_model_threads") or config.get("model_threads") or result.get("model_threads"),
        dispatch_mode=concurrency.get("dispatch_mode"),
        max_parallel_active=concurrency.get("max_parallel_active"),
    )
    if writer_stats:
        packet.record_sample(sample_type="writer_stats", name="latest_writer_stats", **writer_stats)
    packet.record_sample(
        sample_type="worker_resource_summary",
        name="worker_resource",
        worker_event_count=diagnostics.get("worker_event_count"),
        max_worker_rss_mb=diagnostics.get("max_worker_rss_mb"),
        max_worker_uss_mb=diagnostics.get("max_worker_uss_mb"),
        max_worker_thread_count=diagnostics.get("max_worker_thread_count"),
    )

    for row in diagnostics.get("slowest_units") or []:
        if isinstance(row, Mapping):
            score = float(row.get("elapsed_s", 0.0) or 0.0)
            packet.record_top_offender(
                str(row.get("asset") or row.get("unit") or "unit"),
                score,
                category="slowest_unit",
                metadata=dict(row),
            )
    for row in diagnostics.get("slowest_shards") or []:
        if isinstance(row, Mapping):
            score = float(row.get("elapsed_s", 0.0) or 0.0)
            name = f"{row.get('interval')}:{row.get('horizon_minutes')}:{row.get('task')}:shard={row.get('shard_index')}"
            packet.record_top_offender(name, score, category="slowest_shard", metadata=dict(row))
    for row in diagnostics.get("slowest_worker_tasks") or []:
        if isinstance(row, Mapping):
            score = float(row.get("elapsed_s", row.get("duration_s", 0.0)) or 0.0)
            packet.record_top_offender(
                str(row.get("work_unit_id") or row.get("task_id") or "worker_task"),
                score,
                category="slowest_worker_task",
                metadata=dict(row),
            )
    if diagnostics_path:
        try:
            for row in iter_diagnostic_events(Path(diagnostics_path)):
                if str(row.get("event", "")) != "worker_resource":
                    continue
                packet.record_event("worker_resource", dict(row.get("worker_telemetry") or row))
        except Exception:
            pass

    accuracy = result.get("accuracy") if isinstance(result.get("accuracy"), Mapping) else {}
    verification = result.get("output_verification") if isinstance(result.get("output_verification"), Mapping) else {}
    row_counts = {
        "forecast_rows": int(sum(int(row.get("forecast_rows", 0) or 0) for row in diagnostics.get("slowest_shards") or [] if isinstance(row, Mapping))),
        "accuracy_rows": int(accuracy.get("rows", 0) or accuracy.get("row_count", 0) or 0),
        "verification_failure_count": len(verification.get("failures") or []) if isinstance(verification.get("failures"), list) else 0,
    }
    parity_status = "passed" if row_counts["verification_failure_count"] == 0 else "failed"
    packet.set_output_parity(
        status=parity_status,
        row_counts=row_counts,
        notes=["Numeric packet alignment preserves existing run_summary.json, diagnostics JSONL, and Stage 2 handoff manifests."],
    )
    packet.finalize(
        status="completed" if bool(result.get("success", True)) else "failed",
        run_summary={
            "legacy_run_summary_path": paths.get("run_summary"),
            "diagnostics_jsonl": result.get("diagnostics_jsonl") or paths.get("diagnostics_jsonl"),
            "requested_effective_concurrency": {
                "requested_workers": config.get("unit_workers") or result.get("workers"),
                "requested_model_threads": config.get("model_threads") or result.get("model_threads"),
                "effective_workers": concurrency.get("effective_workers") or config.get("unit_workers") or result.get("workers"),
                "effective_model_threads": concurrency.get("effective_model_threads") or config.get("model_threads") or result.get("model_threads"),
                "dispatch_mode": concurrency.get("dispatch_mode"),
                "max_parallel_active": concurrency.get("max_parallel_active"),
            },
            "writer_stats": writer_stats,
            "runner_diagnostics": diagnostics,
            "production_outputs_written": False,
        },
    )
    return {
        "run_summary": str(packet.paths.run_summary),
        "diagnostic_manifest": str(packet.paths.diagnostic_manifest),
        "diagnostic_events": str(packet.paths.diagnostic_events),
        "diagnostic_samples": str(packet.paths.diagnostic_samples),
        "top_offenders": str(packet.paths.top_offenders),
        "output_parity": str(packet.paths.output_parity),
    }


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Summarize numeric runner diagnostics JSONL.")
    parser.add_argument("path", type=Path)
    parser.add_argument("--top-n", type=int, default=10)
    args = parser.parse_args(argv)
    print(json.dumps(summarize_diagnostics(args.path, top_n=int(args.top_n)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
