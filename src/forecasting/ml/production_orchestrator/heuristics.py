from __future__ import annotations

from typing import Dict, List


def derive_heuristics(summary: Dict[str, float], *, configured_workers: int | None, configured_threads: int | None) -> List[str]:
    heuristics: List[str] = []
    runtime_seconds = float(summary.get("runtime_seconds", 0.0) or 0.0)
    avg_cpu_tree = float(summary.get("avg_cpu_pct_tree", 0.0) or 0.0)
    peak_rss_mb = float(summary.get("peak_rss_mb", 0.0) or 0.0)
    peak_child_count = int(summary.get("peak_child_count", 0) or 0)
    peak_thread_count = int(summary.get("peak_thread_count", 0) or 0)
    last_progress_age = float(summary.get("last_progress_age_seconds", 0.0) or 0.0)
    last_artifact_age = float(summary.get("last_artifact_change_age_seconds", 0.0) or 0.0)

    if configured_workers is not None and int(configured_workers) >= 4 and peak_child_count <= 1:
        heuristics.append("suspected_serial_execution")
    if peak_rss_mb >= 4096.0 and avg_cpu_tree <= 20.0:
        heuristics.append("memory_swollen_cpu_idle")
    if configured_threads is not None and int(configured_threads) >= 4 and peak_thread_count <= 2:
        heuristics.append("unexpected_thread_collapse")
    if runtime_seconds >= 600.0 and avg_cpu_tree <= 25.0:
        heuristics.append("long_runtime_low_cpu")
    if max(last_progress_age, last_artifact_age) >= 180.0:
        heuristics.append("stalled_warning")
    return heuristics
