from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

try:
    import psutil
except Exception:
    psutil = None


def diagnostics_file(state_root: Path, branch: str) -> Path:
    return Path(state_root) / f"{str(branch)}_run_diagnostics.jsonl"


def resource_snapshot() -> Dict[str, Any]:
    snapshot: Dict[str, Any] = {"monotonic_s": float(time.monotonic())}
    if psutil is None:
        return snapshot
    try:
        proc = psutil.Process(os.getpid())
        mem = proc.memory_info()
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
    except Exception:
        pass
    return snapshot


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
    for line in Path(path).read_text(encoding="utf-8").splitlines():
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
    events = list(iter_diagnostic_events(path))
    event_counts: Dict[str, int] = {}
    for row in events:
        event = str(row.get("event", "unknown"))
        event_counts[event] = int(event_counts.get(event, 0)) + 1

    shard_rows = [row for row in events if row.get("event") == "shard_finished"]
    combo_rows = [row for row in events if row.get("event") == "combo_complete"]
    writer_rows = [row for row in events if str(row.get("event", "")).startswith("writer_")]
    resource_rows = [row.get("resource") for row in events if isinstance(row.get("resource"), dict)]
    max_rss_mb = max((float(row.get("rss_mb", 0.0) or 0.0) for row in resource_rows), default=0.0)
    max_system_ram_pct = max((float(row.get("system_ram_pct", 0.0) or 0.0) for row in resource_rows), default=0.0)

    combo_elapsed: Dict[Tuple[int, int, str], float] = {}
    combo_rows_out: Dict[Tuple[int, int, str], int] = {}
    combo_done_units: Dict[Tuple[int, int, str], int] = {}
    combo_unit_elapsed: Dict[Tuple[int, int, str], float] = {}
    combo_unit_count: Dict[Tuple[int, int, str], int] = {}
    slowest_units_all: List[Dict[str, Any]] = []
    for row in shard_rows:
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
        slowest_units = row.get("slowest_units") if isinstance(row.get("slowest_units"), list) else []
        if not slowest_units and isinstance(row.get("slowest_unit"), dict):
            slowest_units = [row.get("slowest_unit")]
        for unit in slowest_units:
            if not isinstance(unit, dict):
                continue
            slowest_units_all.append(
                {
                    "interval": key[0],
                    "horizon_minutes": key[1],
                    "task": key[2],
                    "shard_index": int(row.get("shard_index", 0) or 0),
                    **dict(unit),
                }
            )

    slowest_shards = sorted(
        shard_rows,
        key=lambda row: float(row.get("elapsed_s", 0.0) or 0.0),
        reverse=True,
    )[: max(1, int(top_n))]
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
    )[: max(1, int(top_n))]

    latest_writer_stats: Dict[str, Any] = {}
    for row in reversed(events):
        stats = row.get("writer_stats")
        if isinstance(stats, dict):
            latest_writer_stats = dict(stats)
            break

    return {
        "path": str(Path(path)),
        "events": int(len(events)),
        "event_counts": event_counts,
        "combo_count": int(len(combo_rows)),
        "shard_count": int(len(shard_rows)),
        "writer_event_count": int(len(writer_rows)),
        "max_rss_mb": round(float(max_rss_mb), 3),
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
        "slowest_units": sorted(
            slowest_units_all,
            key=lambda row: float(row.get("elapsed_s", 0.0) or 0.0),
            reverse=True,
        )[: max(1, int(top_n))],
    }


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Summarize numeric runner diagnostics JSONL.")
    parser.add_argument("path", type=Path)
    parser.add_argument("--top-n", type=int, default=10)
    args = parser.parse_args(argv)
    print(json.dumps(summarize_diagnostics(args.path, top_n=int(args.top_n)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
