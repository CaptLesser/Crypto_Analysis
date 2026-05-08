from __future__ import annotations

import csv
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from src.forecasting.ml.production_orchestrator.common import latest_mtime_under_roots, utc_now_iso, write_json_atomic
from src.forecasting.ml.production_orchestrator.phases import infer_phase_from_text

try:
    import psutil  # type: ignore
except Exception:  # pragma: no cover
    psutil = None


@dataclass
class TelemetryRecorder:
    module_key: str
    log_path: Path
    anchor_roots: Sequence[Path]
    sample_interval_seconds: float = 5.0
    pid: Optional[int] = None
    sample_callback: Optional[Callable[[Dict[str, Any]], None]] = None

    def __post_init__(self) -> None:
        self._started_monotonic = time.monotonic()
        self._samples: List[Dict[str, Any]] = []
        self._log_offset = 0
        self._last_phase: str = "unknown"
        self._last_progress_monotonic = self._started_monotonic
        self._last_artifact_change_monotonic = self._started_monotonic
        self._peak_rss: Tuple[float, str] = (0.0, "unknown")
        self._peak_threads: Tuple[int, str] = (0, "unknown")
        self._peak_cpu: Tuple[float, str] = (0.0, "unknown")
        self._latest_artifact_mtime = latest_mtime_under_roots(self.anchor_roots)
        self._process_cache: Dict[int, Any] = {}
        self._primed_pids: set[int] = set()
        if psutil is not None:
            try:
                psutil.cpu_percent(interval=None)
                if self.pid is not None:
                    self._prime_process_tree()
            except Exception:
                pass

    def _process_tree(self) -> List[Any]:
        if psutil is None or self.pid is None:
            return []
        try:
            root = psutil.Process(int(self.pid))
            return [root, *root.children(recursive=True)]
        except Exception:
            return []

    def _prime_process_tree(self) -> None:
        for process in self._process_tree():
            try:
                pid = int(process.pid)
            except Exception:
                continue
            self._process_cache[pid] = process
            if pid in self._primed_pids:
                continue
            try:
                process.cpu_percent(interval=None)
                self._primed_pids.add(pid)
            except Exception:
                continue

    def _process_tree_stats(self) -> Dict[str, Any]:
        tree = self._process_tree()
        if not tree:
            return {
                "child_count": 0,
                "thread_count_tree": 0,
                "rss_mb_tree": 0.0,
                "cpu_pct_tree": 0.0,
            }
        child_count = max(0, len(tree) - 1)
        thread_count = 0
        rss_bytes = 0
        cpu_pct = 0.0
        live_pids: set[int] = set()
        for process in tree:
            try:
                pid = int(process.pid)
                live_pids.add(pid)
                cached = self._process_cache.get(pid)
                if cached is None:
                    self._process_cache[pid] = process
                    cached = process
                if pid not in self._primed_pids:
                    cached.cpu_percent(interval=None)
                    self._primed_pids.add(pid)
                thread_count += int(process.num_threads())
            except Exception:
                pass
            try:
                rss_bytes += int(process.memory_info().rss)
            except Exception:
                pass
            try:
                cpu_pct += float(cached.cpu_percent(interval=None))
            except Exception:
                pass
        stale_pids = [pid for pid in self._process_cache.keys() if pid not in live_pids]
        for pid in stale_pids:
            self._process_cache.pop(pid, None)
            self._primed_pids.discard(pid)
        return {
            "child_count": int(child_count),
            "thread_count_tree": int(thread_count),
            "rss_mb_tree": float(rss_bytes) / (1024.0 * 1024.0),
            "cpu_pct_tree": float(cpu_pct),
        }

    def _system_stats(self) -> Dict[str, Any]:
        if psutil is None:
            return {"sys_cpu_pct": 0.0, "sys_ram_pct": 0.0, "disk_free_gb": None}
        try:
            cpu_pct = float(psutil.cpu_percent(interval=None))
            ram_pct = float(psutil.virtual_memory().percent)
            disk_root = Path.cwd().anchor or "/"
            disk = psutil.disk_usage(str(disk_root))
            return {
                "sys_cpu_pct": float(cpu_pct),
                "sys_ram_pct": float(ram_pct),
                "disk_free_gb": float(disk.free) / (1024.0 ** 3),
            }
        except Exception:
            return {"sys_cpu_pct": 0.0, "sys_ram_pct": 0.0, "disk_free_gb": None}

    def _consume_log_markers(self) -> None:
        if not self.log_path.exists():
            return
        try:
            with self.log_path.open("r", encoding="utf-8", errors="ignore") as handle:
                handle.seek(self._log_offset)
                chunk = handle.read()
                self._log_offset = handle.tell()
        except Exception:
            return
        if not chunk:
            return
        lines = chunk.splitlines()
        phase = infer_phase_from_text(lines)
        if phase is not None:
            self._last_phase = str(phase)
            self._last_progress_monotonic = time.monotonic()

    def sample(self) -> None:
        now = time.monotonic()
        self._consume_log_markers()
        latest_artifact_mtime = latest_mtime_under_roots(self.anchor_roots)
        if latest_artifact_mtime is not None and (
            self._latest_artifact_mtime is None or float(latest_artifact_mtime) > float(self._latest_artifact_mtime)
        ):
            self._latest_artifact_mtime = float(latest_artifact_mtime)
            self._last_artifact_change_monotonic = now
        tree_stats = self._process_tree_stats()
        system_stats = self._system_stats()
        sample = {
            "sample_ts": utc_now_iso(),
            "elapsed_s": round(now - self._started_monotonic, 3),
            "phase": str(self._last_phase),
            **system_stats,
            "pid": int(self.pid) if self.pid is not None else None,
            **tree_stats,
        }
        self._samples.append(sample)
        if self.sample_callback is not None:
            try:
                self.sample_callback(dict(sample))
            except Exception:
                pass
        if float(tree_stats["rss_mb_tree"]) >= float(self._peak_rss[0]):
            self._peak_rss = (float(tree_stats["rss_mb_tree"]), str(self._last_phase))
        if int(tree_stats["thread_count_tree"]) >= int(self._peak_threads[0]):
            self._peak_threads = (int(tree_stats["thread_count_tree"]), str(self._last_phase))
        if float(tree_stats["cpu_pct_tree"]) >= float(self._peak_cpu[0]):
            self._peak_cpu = (float(tree_stats["cpu_pct_tree"]), str(self._last_phase))

    def loop_until_exit(self, process: Any) -> None:
        next_sample = time.monotonic()
        while True:
            returncode = process.poll()
            now = time.monotonic()
            if now >= next_sample:
                self.sample()
                next_sample = now + max(5.0, float(self.sample_interval_seconds))
            if returncode is not None:
                if not self._samples or (time.monotonic() - next_sample) < max(1.0, float(self.sample_interval_seconds)):
                    self.sample()
                return
            time.sleep(0.5)

    def write_outputs(self, *, samples_path: Path, summary_path: Path, runtime_seconds: float, configured_workers: Optional[int], configured_threads: Optional[int], final_phase_durations: Dict[str, float]) -> Dict[str, Any]:
        samples_path.parent.mkdir(parents=True, exist_ok=True)
        with samples_path.open("w", encoding="utf-8", newline="") as handle:
            fieldnames = [
                "sample_ts",
                "elapsed_s",
                "phase",
                "sys_cpu_pct",
                "sys_ram_pct",
                "disk_free_gb",
                "pid",
                "child_count",
                "thread_count_tree",
                "rss_mb_tree",
                "cpu_pct_tree",
            ]
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for sample in self._samples:
                writer.writerow({field: sample.get(field) for field in fieldnames})

        sample_count = len(self._samples)
        phase_durations: Dict[str, float] = {}
        previous_elapsed = 0.0
        for sample in self._samples:
            phase = str(sample.get("phase") or "unknown")
            elapsed = float(sample.get("elapsed_s", 0.0) or 0.0)
            delta = max(0.0, elapsed - previous_elapsed)
            phase_durations[phase] = float(phase_durations.get(phase, 0.0) + delta)
            previous_elapsed = elapsed
        for phase, duration in final_phase_durations.items():
            phase_durations[str(phase)] = float(duration)
        avg = lambda key: (sum(float(sample.get(key, 0.0) or 0.0) for sample in self._samples) / float(sample_count)) if sample_count else 0.0
        peak = lambda key: max((float(sample.get(key, 0.0) or 0.0) for sample in self._samples), default=0.0)
        min_nonnull = lambda key: min((float(sample.get(key)) for sample in self._samples if sample.get(key) is not None), default=None)
        artifact_file_count = 0
        for root in self.anchor_roots:
            if not root.exists():
                continue
            if root.is_file():
                artifact_file_count += 1
                continue
            try:
                artifact_file_count += sum(1 for path in root.rglob("*") if path.is_file())
            except Exception:
                continue
        last_progress_age = max(0.0, time.monotonic() - self._last_progress_monotonic)
        last_artifact_age = max(0.0, time.monotonic() - self._last_artifact_change_monotonic)
        summary = {
            "schema_version": 1,
            "module_key": str(self.module_key),
            "sample_count": int(sample_count),
            "sample_interval_seconds": float(max(5.0, self.sample_interval_seconds)),
            "runtime_seconds": float(runtime_seconds),
            "configured_runtime": {
                "workers": (int(configured_workers) if configured_workers is not None else None),
                "threads": (int(configured_threads) if configured_threads is not None else None),
            },
            "process_tree": {
                "peak_child_count": int(peak("child_count")),
                "peak_thread_count": int(peak("thread_count_tree")),
                "peak_rss_mb": float(peak("rss_mb_tree")),
                "avg_cpu_pct": float(avg("cpu_pct_tree")),
                "peak_cpu_pct": float(peak("cpu_pct_tree")),
            },
            "system": {
                "avg_cpu_pct": float(avg("sys_cpu_pct")),
                "peak_cpu_pct": float(peak("sys_cpu_pct")),
                "avg_ram_pct": float(avg("sys_ram_pct")),
                "peak_ram_pct": float(peak("sys_ram_pct")),
                "min_disk_free_gb": min_nonnull("disk_free_gb"),
            },
            "phase_peaks": {
                "peak_rss_phase": str(self._peak_rss[1]),
                "peak_thread_phase": str(self._peak_threads[1]),
                "peak_cpu_phase": str(self._peak_cpu[1]),
            },
            "phase_durations": phase_durations,
            "last_progress_age_seconds": float(last_progress_age),
            "last_artifact_change_age_seconds": float(last_artifact_age),
            "log_size_bytes_final": int(self.log_path.stat().st_size) if self.log_path.exists() else 0,
            "artifact_file_count_final": int(artifact_file_count),
        }
        write_json_atomic(summary_path, summary)
        return summary
