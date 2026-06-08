from __future__ import annotations

import json
import os
import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from src.forecasting.common.runtime_config import resolve_worker_setting
from src.regimes.core.serialization import to_jsonable


MARKET_STATE_TEST_RUNTIME_SCHEMA_VERSION = 1
MARKET_STATE_TEST_MODULE = "market_state_test"
MARKET_STATE_TEST_RUN_MANIFEST_FILENAME = "market_state_test_run_manifest.json"
MARKET_STATE_TEST_TELEMETRY_FILENAME = "market_state_test_telemetry_manifest.json"
MARKET_STATE_TEST_PROFILE_REGISTRY_VERSION = "test_candidate_v0_non_production"
MARKET_STATE_TEST_COVERAGE_POLICY_VERSION = "market_state_dynamic_coverage_v1"
MARKET_STATE_TEST_BATCHING_POLICY = "deterministic_axis_band_feature_method_tasks"
MARKET_STATE_TEST_CACHE_POLICY = "in_run_only_no_cross_run_persistence"
MARKET_STATE_TEST_DEFAULT_WORKERS = 6
MARKET_STATE_TEST_DEFAULT_WRITER_WORKERS = 1
THREAD_ENV_VARS: tuple[str, ...] = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")


def safe_token(value: object, *, fallback: str = "market_state_test") -> str:
    text = str(value).strip() or fallback
    for token in ("/", "\\", ":", "\x00"):
        text = text.replace(token, "_")
    return "".join(ch if ch.isalnum() or ch in "._=-" else "_" for ch in text)


def stable_cache_key(payload: Mapping[str, Any]) -> str:
    return json.dumps(to_jsonable(dict(payload)), sort_keys=True, separators=(",", ":"))


def observed_thread_env() -> dict[str, str | None]:
    return {name: os.environ.get(name) for name in THREAD_ENV_VARS}


def process_snapshot(label: str) -> dict[str, Any]:
    payload: dict[str, Any] = {"label": str(label), "monotonic_s": time.perf_counter()}
    try:
        import psutil  # type: ignore

        proc = psutil.Process(os.getpid())
        cpu = proc.cpu_times()
        payload.update(
            {
                "process_rss_mb": float(proc.memory_info().rss) / (1024.0 * 1024.0),
                "process_vms_mb": float(proc.memory_info().vms) / (1024.0 * 1024.0),
                "process_num_threads": int(proc.num_threads()),
                "process_cpu_seconds": float(cpu.user + cpu.system),
                "child_process_count": int(len(proc.children(recursive=True))),
            }
        )
    except Exception as exc:  # pragma: no cover - platform dependent
        payload["process_snapshot_error"] = type(exc).__name__
    return payload


@dataclass(frozen=True)
class MarketStateTestRuntimeConfig:
    run_id: str
    output_root: str
    market_state_test_workers: int = MARKET_STATE_TEST_DEFAULT_WORKERS
    writer_workers: int = MARKET_STATE_TEST_DEFAULT_WRITER_WORKERS
    batching_policy: str = MARKET_STATE_TEST_BATCHING_POLICY
    cache_policy: str = MARKET_STATE_TEST_CACHE_POLICY
    run_mode: str = "non-production Test"
    selected_window_policy: str = "manifest_split_policy"
    dynamic_coverage_policy_version: str = MARKET_STATE_TEST_COVERAGE_POLICY_VERSION
    schema_version: int = MARKET_STATE_TEST_RUNTIME_SCHEMA_VERSION
    profile_registry_version: str = MARKET_STATE_TEST_PROFILE_REGISTRY_VERSION
    thread_caps_enforced: bool = False
    observed_thread_env: Mapping[str, str | None] = field(default_factory=observed_thread_env)
    worker_source: Mapping[str, Any] = field(default_factory=dict)
    writer_source: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_manifest(cls, manifest: Any, *, write_root: Path) -> "MarketStateTestRuntimeConfig":
        worker = resolve_worker_setting(MARKET_STATE_TEST_MODULE, "market_state_test_workers", fallback=MARKET_STATE_TEST_DEFAULT_WORKERS)
        writer = resolve_worker_setting(MARKET_STATE_TEST_MODULE, "writer_workers", fallback=MARKET_STATE_TEST_DEFAULT_WRITER_WORKERS)
        return cls(
            run_id=safe_token(getattr(manifest, "study_id", "market_state_test")),
            output_root=str(write_root),
            market_state_test_workers=max(1, int(worker["value"])),
            writer_workers=max(1, int(writer["value"])),
            observed_thread_env=observed_thread_env(),
            worker_source=dict(worker),
            writer_source=dict(writer),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "artifact_kind": "market_state_test_runtime_config",
            "run_id": self.run_id,
            "run_mode": self.run_mode,
            "output_root": self.output_root,
            "market_state_test_workers": int(self.market_state_test_workers),
            "writer_workers": int(self.writer_workers),
            "writer_policy": "single_deterministic_final_writer",
            "batching_policy": self.batching_policy,
            "cache_policy": self.cache_policy,
            "selected_window_policy": self.selected_window_policy,
            "dynamic_coverage_policy_version": self.dynamic_coverage_policy_version,
            "profile_registry_version": self.profile_registry_version,
            "thread_caps_enforced": bool(self.thread_caps_enforced),
            "observed_thread_env": dict(self.observed_thread_env),
            "worker_source": dict(self.worker_source),
            "writer_source": dict(self.writer_source),
            "production_writes": False,
            "production_labels": False,
            "production_profile_selection": False,
            "production_promotion": False,
            "broad_all_to_all_pairwise": False,
            "cross_asset_peer_groups": False,
            "l2_order_book_sidecars": False,
        }


@dataclass
class MarketStateRuntimeCache:
    run_id: str
    assets_by_interval: dict[str, tuple[str, ...]] = field(default_factory=dict)
    universe_frames: dict[str, dict[str, Any]] = field(default_factory=dict)
    dataset_results: dict[str, Any] = field(default_factory=dict)
    feature_results: dict[str, Any] = field(default_factory=dict)
    clusterability_results: dict[str, Any] = field(default_factory=dict)
    clusterer_matrices: dict[str, Any] = field(default_factory=dict)
    forward_target_frames: dict[str, Any] = field(default_factory=dict)
    io_telemetry: dict[str, Mapping[str, Any]] = field(default_factory=dict)
    task_keys: dict[str, Mapping[str, Any]] = field(default_factory=dict)
    hits: Counter[str] = field(default_factory=Counter)
    misses: Counter[str] = field(default_factory=Counter)
    build_failures: Counter[str] = field(default_factory=Counter)
    build_seconds: Counter[str] = field(default_factory=Counter)
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)
    _build_locks: dict[tuple[str, str], threading.RLock] = field(default_factory=dict, repr=False)

    def get(self, store: str, key: str) -> Any | None:
        with self._lock:
            mapping = getattr(self, store)
            if key in mapping:
                self.hits[store] += 1
                return mapping[key]
            self.misses[store] += 1
            return None

    def set(self, store: str, key: str, value: Any, *, metadata: Mapping[str, Any] | None = None) -> Any:
        with self._lock:
            getattr(self, store)[key] = value
            if metadata is not None:
                self.task_keys[key] = dict(metadata)
            return value

    def get_or_build(
        self,
        store: str,
        key: str,
        builder: Callable[[], Any],
        *,
        metadata: Mapping[str, Any] | None = None,
        copy_for_return: Callable[[Any], Any] | None = None,
        io_telemetry: Callable[[Any, float, bool], Mapping[str, Any] | None] | None = None,
    ) -> Any:
        with self._lock:
            mapping = getattr(self, store)
            if key in mapping:
                self.hits[store] += 1
                value = mapping[key]
                return copy_for_return(value) if copy_for_return is not None else value
            lock = self._build_locks.setdefault((store, key), threading.RLock())
        with lock:
            with self._lock:
                mapping = getattr(self, store)
                if key in mapping:
                    self.hits[store] += 1
                    value = mapping[key]
                    return copy_for_return(value) if copy_for_return is not None else value
                self.misses[store] += 1
            started = time.perf_counter()
            try:
                value = builder()
            except Exception:
                with self._lock:
                    self.build_failures[store] += 1
                raise
            elapsed = max(0.0, time.perf_counter() - started)
            with self._lock:
                getattr(self, store)[key] = value
                self.build_seconds[store] += float(elapsed)
                if metadata is not None:
                    self.task_keys[key] = dict(metadata)
                if io_telemetry is not None:
                    payload = io_telemetry(value, elapsed, False)
                    if payload is not None:
                        self.io_telemetry[key] = dict(payload)
            return copy_for_return(value) if copy_for_return is not None else value

    def summary(self) -> dict[str, Any]:
        stores = (
            "assets_by_interval",
            "universe_frames",
            "dataset_results",
            "feature_results",
            "clusterability_results",
            "clusterer_matrices",
            "forward_target_frames",
        )
        with self._lock:
            return {
                "run_id": self.run_id,
                "cache_policy": MARKET_STATE_TEST_CACHE_POLICY,
                "store_entry_counts": {store: int(len(getattr(self, store))) for store in stores},
                "hits": {key: int(value) for key, value in sorted(self.hits.items())},
                "misses": {key: int(value) for key, value in sorted(self.misses.items())},
                "build_failures": {key: int(value) for key, value in sorted(self.build_failures.items())},
                "build_seconds": {key: float(value) for key, value in sorted(self.build_seconds.items())},
                "cache_key_count": int(len(self.task_keys)),
                "cache_key_samples": list(self.task_keys.values())[:10],
                "io_telemetry": list(self.io_telemetry.values())[:100],
                "cross_run_persistence": False,
            }

    def compact_summary(self) -> dict[str, Any]:
        stores = (
            "assets_by_interval",
            "universe_frames",
            "dataset_results",
            "feature_results",
            "clusterability_results",
            "clusterer_matrices",
            "forward_target_frames",
        )
        with self._lock:
            return {
                "run_id": self.run_id,
                "cache_policy": MARKET_STATE_TEST_CACHE_POLICY,
                "store_entry_counts": {store: int(len(getattr(self, store))) for store in stores},
                "hits": {key: int(value) for key, value in sorted(self.hits.items())},
                "misses": {key: int(value) for key, value in sorted(self.misses.items())},
                "build_failures": {key: int(value) for key, value in sorted(self.build_failures.items())},
                "build_seconds": {key: float(value) for key, value in sorted(self.build_seconds.items())},
                "cache_key_count": int(len(self.task_keys)),
                "io_telemetry_count": int(len(self.io_telemetry)),
                "cross_run_persistence": False,
            }


@dataclass
class MarketStateRunTelemetry:
    started_monotonic_s: float = field(default_factory=time.perf_counter)
    stage_seconds: Counter[str] = field(default_factory=Counter)
    stage_counts: Counter[str] = field(default_factory=Counter)
    warning_counts: Counter[str] = field(default_factory=Counter)
    warning_samples: dict[str, list[dict[str, str]]] = field(default_factory=dict)
    snapshots: list[dict[str, Any]] = field(default_factory=lambda: [process_snapshot("start")])
    artifact_write_seconds: float = 0.0

    def add_trial(self, trial: Any) -> None:
        runtime = dict(getattr(trial, "runtime", {}) or {})
        for stage, value in dict(runtime.get("phase_seconds") or {}).items():
            self.stage_seconds[str(stage)] += float(value)
            self.stage_counts[str(stage)] += int(dict(runtime.get("phase_counts") or {}).get(stage, 1))
        for stage, value in dict(runtime.get("warning_counts") or {}).items():
            self.warning_counts[str(stage)] += int(value)
        for stage, samples in dict(runtime.get("warning_samples") or {}).items():
            if not isinstance(samples, list):
                continue
            bucket = self.warning_samples.setdefault(str(stage), [])
            for sample in samples:
                if len(bucket) >= 5:
                    break
                if isinstance(sample, Mapping):
                    bucket.append({"category": str(sample.get("category", "")), "message": str(sample.get("message", ""))[:500]})

    def finish(self, label: str = "end") -> None:
        self.snapshots.append(process_snapshot(label))

    def as_dict(
        self,
        *,
        runtime_config: MarketStateTestRuntimeConfig,
        trials: Sequence[Any],
        cache: MarketStateRuntimeCache,
        task_count: int,
        artifact_paths: Mapping[str, str],
    ) -> dict[str, Any]:
        statuses = Counter(str(getattr(trial, "status", "unknown")) for trial in trials)
        rss_values = [
            float(snapshot["process_rss_mb"])
            for snapshot in self.snapshots
            if isinstance(snapshot.get("process_rss_mb"), (int, float))
        ]
        cpu_values = [
            float(snapshot["process_cpu_seconds"])
            for snapshot in self.snapshots
            if isinstance(snapshot.get("process_cpu_seconds"), (int, float))
        ]
        rows = []
        for trial in trials:
            runtime = dict(getattr(trial, "runtime", {}) or {})
            candidate = dict(getattr(trial, "candidate", {}) or {})
            rows.append(
                {
                    "trial_id": getattr(trial, "trial_id", None),
                    "status": getattr(trial, "status", None),
                    "band": candidate.get("band"),
                    "axis": candidate.get("axis"),
                    "feature_family_id": candidate.get("feature_family_id"),
                    "clusterer_family": (candidate.get("clusterer") or {}).get("family"),
                    "elapsed_s": runtime.get("elapsed_s"),
                    "row_count": runtime.get("row_count"),
                    "feature_count": runtime.get("feature_count"),
                    "warning_count_total": runtime.get("warning_count_total", 0),
                }
            )
        return {
            "schema_version": MARKET_STATE_TEST_RUNTIME_SCHEMA_VERSION,
            "artifact_kind": "market_state_test_telemetry_manifest",
            "run_id": runtime_config.run_id,
            "runtime_config": runtime_config.as_dict(),
            "elapsed_s": max(0.0, time.perf_counter() - self.started_monotonic_s),
            "stage_wall_seconds": {key: float(value) for key, value in sorted(self.stage_seconds.items())},
            "stage_counts": {key: int(value) for key, value in sorted(self.stage_counts.items())},
            "artifact_write_seconds": float(self.artifact_write_seconds),
            "process_snapshots": list(self.snapshots),
            "process_rss_mb_peak_observed": max(rss_values) if rss_values else None,
            "process_rss_mb_final_observed": rss_values[-1] if rss_values else None,
            "process_cpu_seconds_delta_observed": (max(cpu_values) - min(cpu_values)) if len(cpu_values) >= 2 else None,
            "observed_thread_env": observed_thread_env(),
            "thread_caps_enforced": False,
            "task_count": int(task_count),
            "worker_count": int(runtime_config.market_state_test_workers),
            "writer_workers": int(runtime_config.writer_workers),
            "cache_summary": cache.summary(),
            "trial_status_counts": {key: int(value) for key, value in sorted(statuses.items())},
            "warning_counts": {key: int(value) for key, value in sorted(self.warning_counts.items())},
            "warning_count_total": int(sum(self.warning_counts.values())),
            "warning_samples": {key: list(value) for key, value in sorted(self.warning_samples.items())},
            "method_band_axis_breakdown": rows,
            "artifact_paths": dict(artifact_paths),
            "production_writes": False,
            "production_labels": False,
            "final_profile_selection": False,
        }


def build_run_manifest(
    *,
    runtime_config: MarketStateTestRuntimeConfig,
    study_manifest: Mapping[str, Any],
    search_matrix: Mapping[str, Any],
    trials: Sequence[Any],
    cache: MarketStateRuntimeCache,
    artifact_paths: Mapping[str, str],
    validation_status: str = "not_run",
    finalization_status: str = "written",
) -> dict[str, Any]:
    planned = [dict(candidate) for candidate in search_matrix.get("candidates", [])] if isinstance(search_matrix, Mapping) else []
    completed: list[str] = []
    failed: list[str] = []
    skipped: list[str] = []
    degenerate: list[str] = []
    for trial in trials:
        status = str(getattr(trial, "status", "unknown"))
        trial_id = str(getattr(trial, "trial_id", ""))
        if status == "completed":
            completed.append(trial_id)
        elif status == "failed":
            failed.append(trial_id)
        elif "skipped" in status or "filtered" in status or "blocked" in status or "universe" in status:
            skipped.append(trial_id)
        elif "degenerate" in status:
            degenerate.append(trial_id)
    planned_task_ids = [str(task.get("candidate_id", "")) for task in planned]
    task_status_count = len(completed) + len(failed) + len(skipped) + len(degenerate)
    run_complete = bool(finalization_status == "written" and len(planned) == task_status_count and not failed)
    identity = {
        "run_id": runtime_config.run_id,
        "run_mode": runtime_config.run_mode,
        "schema_version": MARKET_STATE_TEST_RUNTIME_SCHEMA_VERSION,
        "profile_registry_version": runtime_config.profile_registry_version,
        "coverage_policy_version": runtime_config.dynamic_coverage_policy_version,
        "universe_manifest_id": str((study_manifest.get("universe_policy") or {}).get("policy_id") or study_manifest.get("study_id") or runtime_config.run_id),
        "window_policy": dict(study_manifest.get("split_policy") or {}),
        "source_tail_ts": study_manifest.get("end_ts"),
        "dynamic_coverage_policy": runtime_config.dynamic_coverage_policy_version,
        "production_flags": {
            "production_writes": False,
            "production_labels": False,
            "production_profile_selection": False,
            "production_promotion": False,
            "broad_all_to_all_pairwise": False,
            "cross_asset_peer_groups": False,
            "l2_order_book_sidecars": False,
        },
        "output_root": runtime_config.output_root,
        "cache_policy": runtime_config.cache_policy,
        "task_count": int(len(planned)),
        "planned_task_ids": planned_task_ids,
    }
    return {
        "schema_version": MARKET_STATE_TEST_RUNTIME_SCHEMA_VERSION,
        "artifact_kind": "market_state_test_run_manifest",
        "run_id": runtime_config.run_id,
        "manifest_identity": identity,
        "runtime_config": runtime_config.as_dict(),
        "study_manifest": dict(study_manifest),
        "planned_task_count": int(len(planned)),
        "planned_tasks": planned,
        "completed_tasks": completed,
        "failed_tasks": failed,
        "skipped_tasks": skipped,
        "degenerate_tasks": degenerate,
        "completed_task_ids": completed,
        "failed_task_ids": failed,
        "skipped_task_ids": skipped,
        "degenerate_task_ids": degenerate,
        "cache_summary": cache.summary(),
        "artifact_paths": dict(artifact_paths),
        "output_schema_version": int(study_manifest.get("schema_version", MARKET_STATE_TEST_RUNTIME_SCHEMA_VERSION)),
        "profile_registry_version": runtime_config.profile_registry_version,
        "production_flags": {
            "production_writes": False,
            "production_labels": False,
            "production_profile_selection": False,
            "production_promotion": False,
            "broad_all_to_all_pairwise": False,
            "cross_asset_peer_groups": False,
            "l2_order_book_sidecars": False,
        },
        "validation_status": validation_status,
        "finalization_status": finalization_status,
        "run_complete": run_complete,
        "incomplete_reason": None if run_complete else "planned tasks, task statuses, failures, or finalization status do not indicate a complete run",
        "resume_policy": "rerun_allowed_only_when runtime_config run_id and cache keys match; cross-run caches disabled",
        "cross_run_cache_persistence": False,
    }


def validate_market_state_run_manifest(
    manifest: Mapping[str, Any],
    *,
    expected_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    identity = dict(manifest.get("manifest_identity") or {})
    planned_ids = tuple(str(item) for item in identity.get("planned_task_ids") or ())
    completed = tuple(str(item) for item in manifest.get("completed_task_ids") or manifest.get("completed_tasks") or ())
    failed = tuple(str(item) for item in manifest.get("failed_task_ids") or manifest.get("failed_tasks") or ())
    skipped = tuple(str(item) for item in manifest.get("skipped_task_ids") or manifest.get("skipped_tasks") or ())
    degenerate = tuple(str(item) for item in manifest.get("degenerate_task_ids") or manifest.get("degenerate_tasks") or ())
    terminal = tuple(dict.fromkeys((*completed, *failed, *skipped, *degenerate)))
    production_flags = dict(identity.get("production_flags") or manifest.get("production_flags") or {})
    mismatches: dict[str, dict[str, Any]] = {}
    if expected_identity is not None:
        for field_name, expected in dict(expected_identity).items():
            observed = identity.get(field_name)
            if observed != expected:
                mismatches[field_name] = {"expected": expected, "observed": observed}
    production_safe = not any(bool(value) for value in production_flags.values())
    finalization_complete = str(manifest.get("finalization_status") or "") == "written"
    task_count = int(identity.get("task_count") or manifest.get("planned_task_count") or len(planned_ids))
    complete = bool(
        finalization_complete
        and production_safe
        and not failed
        and not mismatches
        and task_count == len(terminal)
        and (not planned_ids or set(terminal).issubset(set(planned_ids)))
    )
    reasons: list[str] = []
    if not finalization_complete:
        reasons.append("finalization_incomplete")
    if not production_safe:
        reasons.append("production_flags_not_false")
    if failed:
        reasons.append("failed_tasks_present")
    if mismatches:
        reasons.append("identity_mismatch")
    if task_count != len(terminal):
        reasons.append("task_status_incomplete")
    if planned_ids and not set(terminal).issubset(set(planned_ids)):
        reasons.append("unknown_terminal_task_ids")
    return {
        "schema_version": MARKET_STATE_TEST_RUNTIME_SCHEMA_VERSION,
        "artifact_kind": "market_state_run_manifest_validation",
        "complete": complete,
        "safe_to_reuse": complete,
        "identity": identity,
        "identity_mismatches": mismatches,
        "planned_task_count": task_count,
        "terminal_task_count": len(terminal),
        "finalization_complete": finalization_complete,
        "production_flags_safe": production_safe,
        "reasons": reasons,
    }


__all__ = [
    "MARKET_STATE_TEST_CACHE_POLICY",
    "MARKET_STATE_TEST_DEFAULT_WORKERS",
    "MARKET_STATE_TEST_DEFAULT_WRITER_WORKERS",
    "MARKET_STATE_TEST_RUN_MANIFEST_FILENAME",
    "MARKET_STATE_TEST_TELEMETRY_FILENAME",
    "MarketStateRunTelemetry",
    "MarketStateRuntimeCache",
    "MarketStateTestRuntimeConfig",
    "build_run_manifest",
    "observed_thread_env",
    "process_snapshot",
    "safe_token",
    "stable_cache_key",
    "validate_market_state_run_manifest",
]
