from __future__ import annotations

import hashlib
import json
import os
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.regimes.core.serialization import to_jsonable


CROSS_ASSET_STATE_FAMILY_SHARD_MANIFEST_KIND = "cross_asset_state_family_shard_manifest"
CROSS_ASSET_STATE_PROGRESS_HEARTBEAT_KIND = "cross_asset_state_execution_progress_heartbeat"
CROSS_ASSET_STATE_RUNTIME_TELEMETRY_EVENT_KIND = "cross_asset_state_runtime_telemetry_event"
CROSS_ASSET_STATE_RUNTIME_TELEMETRY_SCOPE = "runtime_telemetry_non_active"
CROSS_ASSET_STATE_RUNTIME_TELEMETRY_EVENTS_FILENAME = "cross_asset_state_runtime_telemetry.jsonl"
CROSS_ASSET_STATE_FAMILY_SHARD_MANIFEST_FILENAME = "family_shard_manifest.json"
CROSS_ASSET_STATE_FAMILY_SHARD_SCOPE = "family_shard"
CROSS_ASSET_STATE_DEFAULT_COMBINED_SCOPE = "default_test_branch_combined"
CROSS_ASSET_STATE_HEARTBEAT_SCOPE = "progress_heartbeat_non_active"

SHARD_STATUS_PENDING = "pending"
SHARD_STATUS_RUNNING = "running"
SHARD_STATUS_COMPLETE = "complete"
SHARD_STATUS_FAILED = "failed"
SHARD_STATUS_INCOMPLETE = "incomplete"
CROSS_ASSET_STATE_SHARD_STATUSES = (
    SHARD_STATUS_PENDING,
    SHARD_STATUS_RUNNING,
    SHARD_STATUS_COMPLETE,
    SHARD_STATUS_FAILED,
    SHARD_STATUS_INCOMPLETE,
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def stable_json_hash(payload: Mapping[str, Any] | Sequence[Any]) -> str:
    encoded = json.dumps(to_jsonable(payload), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_fingerprint(path: str | Path) -> dict[str, Any]:
    file_path = Path(path)
    if not file_path.exists() or not file_path.is_file():
        return {"exists": False, "size_bytes": None, "sha256": None}
    digest = hashlib.sha256()
    size = 0
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            size += len(chunk)
            digest.update(chunk)
    return {"exists": True, "size_bytes": int(size), "sha256": digest.hexdigest()}


def family_task_worker_id() -> str:
    return f"pid={os.getpid()}|thread={threading.get_ident()}"


def safe_family_token(family: str) -> str:
    token = re.sub(r"[^A-Za-z0-9_.=-]+", "_", str(family).strip())
    return token or "unknown_family"


def write_json_atomic(path: str | Path, payload: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f"{target.name}.tmp")
    tmp.write_text(json.dumps(to_jsonable(dict(payload)), indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(target)


def read_json_mapping(path: str | Path) -> dict[str, Any]:
    loaded = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(loaded, Mapping):
        raise ValueError(f"Expected JSON object at {path}")
    return dict(loaded)


def write_progress_heartbeat(
    progress_root: str | Path,
    *,
    run_id: str | None = None,
    family: str,
    shard_status: str,
    band: str | None = None,
    window_policy_id: str | None = None,
    asset_id: str | None = None,
    expected_cells: int = 0,
    completed_cells: int = 0,
    candidate_evaluations: int = 0,
    worker_id: str | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    progress_path = Path(progress_root)
    progress_path.mkdir(parents=True, exist_ok=True)
    worker = worker_id or family_task_worker_id()
    timestamp = utc_now_iso()
    payload: dict[str, Any] = {
        "artifact_kind": CROSS_ASSET_STATE_PROGRESS_HEARTBEAT_KIND,
        "artifact_scope": CROSS_ASSET_STATE_HEARTBEAT_SCOPE,
        "run_id": str(run_id or progress_path.parent.name or progress_path.name or "unknown_run"),
        "active_handoff_artifact": False,
        "final_artifact": False,
        "family": str(family),
        "band": str(band) if band is not None else None,
        "window_policy_id": str(window_policy_id) if window_policy_id is not None else None,
        "asset_id": str(asset_id) if asset_id is not None else None,
        "current_cell": {
            "asset_id": str(asset_id) if asset_id is not None else None,
            "relationship_feature_family": str(family),
            "band": str(band) if band is not None else None,
            "window_policy_id": str(window_policy_id) if window_policy_id is not None else None,
        },
        "shard_status": str(shard_status),
        "expected_cells": int(expected_cells),
        "completed_cells": int(completed_cells),
        "candidate_evaluations": int(candidate_evaluations),
        "worker_id": worker,
        "timestamp_utc": timestamp,
        "last_update_utc": timestamp,
        "production_approved": False,
        "production_writer_enabled": False,
        "production_labels_written": False,
        "production_outputs_written": False,
        "canonical_production_state_outputs_written": False,
    }
    for key, value in dict(extra or {}).items():
        if key in payload:
            payload[f"extra_{key}"] = value
            continue
        payload[key] = value
    heartbeat_path = progress_path / f"family={safe_family_token(family)}.heartbeat.json"
    events_path = progress_path / f"family={safe_family_token(family)}.progress.jsonl"
    write_json_atomic(heartbeat_path, payload)
    with events_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(to_jsonable(payload), sort_keys=True) + "\n")
    return {"heartbeat_path": str(heartbeat_path), "progress_events_path": str(events_path), "heartbeat": payload}


def write_runtime_telemetry_event(
    telemetry_root: str | Path,
    *,
    run_id: str,
    shard_id: str | None = None,
    worker_id: str | None = None,
    telemetry_level: str,
    status: str,
    family: str | None = None,
    band: str | None = None,
    window_policy_id: str | None = None,
    asset_id: str | None = None,
    cell_id: str | None = None,
    candidate_eval_count: int = 0,
    method_family: str | None = None,
    profile_type: str | None = None,
    elapsed_s: float = 0.0,
    cache_hit_count: int | None = None,
    cache_miss_count: int | None = None,
    cache_stats: Mapping[str, Any] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    telemetry_path = Path(telemetry_root)
    telemetry_path.mkdir(parents=True, exist_ok=True)
    worker = worker_id or family_task_worker_id()
    timestamp = utc_now_iso()
    family_text = None if family in (None, "") else str(family)
    band_text = None if band in (None, "") else str(band)
    asset_text = None if asset_id in (None, "") else str(asset_id)
    window_text = None if window_policy_id in (None, "") else str(window_policy_id)
    cell_text = cell_id or (
        f"{asset_text}|{family_text}|{band_text}|{window_text}"
        if asset_text is not None and family_text is not None and band_text is not None
        else None
    )
    payload: dict[str, Any] = {
        "artifact_kind": CROSS_ASSET_STATE_RUNTIME_TELEMETRY_EVENT_KIND,
        "artifact_scope": CROSS_ASSET_STATE_RUNTIME_TELEMETRY_SCOPE,
        "run_id": str(run_id),
        "shard_id": None if shard_id in (None, "") else str(shard_id),
        "worker_id": worker,
        "telemetry_level": str(telemetry_level),
        "family": family_text,
        "relationship_feature_family": family_text,
        "band": band_text,
        "window_policy_id": window_text,
        "asset_id": asset_text,
        "cell_id": cell_text,
        "candidate_eval_count": int(candidate_eval_count),
        "method_family": None if method_family in (None, "") else str(method_family),
        "profile_type": None if profile_type in (None, "") else str(profile_type),
        "elapsed_s": round(max(0.0, float(elapsed_s)), 6),
        "cache_hit_count": None if cache_hit_count is None else int(cache_hit_count),
        "cache_miss_count": None if cache_miss_count is None else int(cache_miss_count),
        "cache_stats": to_jsonable(dict(cache_stats or {})),
        "status": str(status),
        "timestamp": timestamp,
        "timestamp_utc": timestamp,
        "active_handoff_artifact": False,
        "not_active_handoff": True,
        "final_artifact": False,
        "partial_artifact": False,
        "incomplete_artifact": False,
        "active_handoff_eligible": False,
        "production_approved": False,
        "production_writer_enabled": False,
        "production_labels_written": False,
        "production_outputs_written": False,
        "canonical_production_state_outputs_written": False,
    }
    for key, value in dict(extra or {}).items():
        if key in payload:
            payload[f"extra_{key}"] = value
            continue
        payload[key] = value
    events_path = telemetry_path / CROSS_ASSET_STATE_RUNTIME_TELEMETRY_EVENTS_FILENAME
    with events_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(to_jsonable(payload), sort_keys=True) + "\n")
    return {"telemetry_events_path": str(events_path), "event": payload}


def aggregate_runtime_telemetry_events(
    event_paths: Sequence[str | Path],
    *,
    final_summary: bool = False,
) -> dict[str, Any]:
    events: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    for path_value in event_paths:
        path = Path(path_value)
        key = str(path)
        if key in seen_paths or not path.is_file():
            continue
        seen_paths.add(key)
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                text = line.strip()
                if not text:
                    continue
                try:
                    loaded = json.loads(text)
                except Exception:
                    continue
                if isinstance(loaded, Mapping):
                    events.append(dict(loaded))
    aggregation = {
        "artifact_kind": "cross_asset_state_runtime_telemetry_aggregation",
        "artifact_scope": "runtime_telemetry_final_summary" if final_summary else CROSS_ASSET_STATE_RUNTIME_TELEMETRY_SCOPE,
        "final_artifact": bool(final_summary),
        "active_handoff_artifact": False,
        "not_active_handoff": True,
        "event_count": len(events),
        "telemetry_event_paths": [str(Path(path)) for path in event_paths],
        "runtime_by_family": _runtime_group(events, ("family",)),
        "runtime_by_band_window": _runtime_group(events, ("band", "window_policy_id")),
        "runtime_by_method_profile_type": _runtime_group(events, ("method_family", "profile_type")),
        "candidate_eval_counts": {
            "total": sum(int(event.get("candidate_eval_count") or 0) for event in events),
            "by_family": _candidate_counts(events, ("family",)),
            "by_method_profile_type": _candidate_counts(events, ("method_family", "profile_type")),
        },
        "cache_stats": {
            "cache_hit_count": sum(int(event.get("cache_hit_count") or 0) for event in events),
            "cache_miss_count": sum(int(event.get("cache_miss_count") or 0) for event in events),
        },
        "worker_utilization_proxy": _worker_utilization_proxy(events),
        "production_write_allowed": False,
    }
    return aggregation


def _runtime_group(events: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for event in events:
        values = tuple(str(event.get(field) or "unknown") for field in fields)
        if all(value == "unknown" for value in values):
            continue
        key = "|".join(values)
        row = grouped.setdefault(
            key,
            {
                "event_count": 0,
                "elapsed_s_sum": 0.0,
                "elapsed_s_max": 0.0,
                "candidate_eval_count": 0,
            },
        )
        elapsed = max(0.0, float(event.get("elapsed_s") or 0.0))
        row["event_count"] = int(row["event_count"]) + 1
        row["elapsed_s_sum"] = round(float(row["elapsed_s_sum"]) + elapsed, 6)
        row["elapsed_s_max"] = round(max(float(row["elapsed_s_max"]), elapsed), 6)
        row["candidate_eval_count"] = int(row["candidate_eval_count"]) + int(event.get("candidate_eval_count") or 0)
    return grouped


def _candidate_counts(events: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for event in events:
        values = tuple(str(event.get(field) or "unknown") for field in fields)
        if all(value == "unknown" for value in values):
            continue
        key = "|".join(values)
        out[key] = int(out.get(key, 0)) + int(event.get("candidate_eval_count") or 0)
    return out


def _worker_utilization_proxy(events: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    worker_counts: dict[str, int] = {}
    worker_elapsed: dict[str, float] = {}
    for event in events:
        worker = str(event.get("worker_id") or "unknown")
        worker_counts[worker] = int(worker_counts.get(worker, 0)) + 1
        worker_elapsed[worker] = round(float(worker_elapsed.get(worker, 0.0)) + max(0.0, float(event.get("elapsed_s") or 0.0)), 6)
    return {
        "unique_worker_count": len(worker_counts),
        "event_count_by_worker": worker_counts,
        "elapsed_s_by_worker": worker_elapsed,
        "proxy_note": "Event counts and elapsed_s are observability proxies, not CPU accounting.",
    }


def build_family_run_fingerprint(
    *,
    family: str,
    bands: Sequence[str],
    feature_set_version: str,
    handoff_path: str | Path,
    eligibility_manifest_path: str | Path,
    eligibility_rows: Sequence[Mapping[str, Any]],
    max_valid_assets: int,
    sample_assets: Sequence[str],
    selection_engine_version: str,
) -> dict[str, Any]:
    family_rows = [
        dict(row)
        for row in eligibility_rows
        if str(row.get("relationship_feature_family")) == str(family)
        and str(row.get("band")) in {str(band) for band in bands}
    ]
    normalized_rows = sorted(family_rows, key=lambda row: json.dumps(to_jsonable(row), sort_keys=True))
    components = {
        "schema_version": 1,
        "family": str(family),
        "bands": [str(band) for band in bands],
        "feature_set_version": str(feature_set_version),
        "handoff_file": file_fingerprint(handoff_path),
        "eligibility_manifest_file": file_fingerprint(eligibility_manifest_path),
        "eligibility_rows_hash": stable_json_hash(normalized_rows),
        "eligibility_row_count": len(normalized_rows),
        "max_valid_assets": int(max_valid_assets),
        "sample_assets": [str(asset) for asset in sample_assets],
        "selection_engine_version": str(selection_engine_version),
    }
    return {"hash": stable_json_hash(components), "components": components}


def family_shard_manifest_payload(
    *,
    shard_status: str,
    family: str,
    bands: Sequence[str],
    run_fingerprint: Mapping[str, Any],
    run_root: str | Path,
    max_valid_assets: int,
    sample_assets: Sequence[str],
    summary: Mapping[str, Any] | None = None,
    runtime_telemetry: Mapping[str, Any] | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    status = str(shard_status)
    if status not in CROSS_ASSET_STATE_SHARD_STATUSES:
        raise ValueError(f"Unsupported Cross-Asset-State family shard status: {status}")
    summary_payload = dict(summary or {})
    counts = {
        "expected_cells": int(summary_payload.get("expected_cells") or 0),
        "selected_model_facing_cells": int(summary_payload.get("selected_model_facing_cells") or 0),
        "diagnostic_only_cells": int(summary_payload.get("diagnostic_only_cells") or 0),
        "masked_unavailable_cells": int(summary_payload.get("masked_unavailable_cells") or 0),
        "missing_cells": int(summary_payload.get("missing_cells") or 0),
    }
    paths = dict(summary_payload.get("paths") or {})
    return {
        "artifact_kind": CROSS_ASSET_STATE_FAMILY_SHARD_MANIFEST_KIND,
        "artifact_scope": CROSS_ASSET_STATE_FAMILY_SHARD_SCOPE,
        "created_at_utc": utc_now_iso(),
        "shard_status": status,
        "partial_artifact": status != SHARD_STATUS_COMPLETE,
        "incomplete_artifact": status != SHARD_STATUS_COMPLETE,
        "active_handoff_eligible": False,
        "active_handoff_artifact": False,
        "not_active_handoff": True,
        "final_artifact": False,
        "relationship_feature_family": str(family),
        "bands": [str(band) for band in bands],
        "run_root": str(run_root),
        "run_fingerprint_hash": str(run_fingerprint.get("hash") or ""),
        "run_fingerprint": to_jsonable(dict(run_fingerprint.get("components") or {})),
        "max_valid_assets": int(max_valid_assets),
        "sample_assets": [str(asset) for asset in sample_assets],
        "counts": counts,
        "paths": paths,
        "runtime_telemetry": dict(runtime_telemetry or {}),
        "error": error,
        "production_approved": False,
        "production_writer_enabled": False,
        "production_labels_written": False,
        "production_outputs_written": False,
        "canonical_production_state_outputs_written": False,
        "requires_human_approval_before_production": True,
    }


def validate_family_shard_manifest(
    manifest: Mapping[str, Any],
    *,
    manifest_path: str | Path | None = None,
    expected_family: str,
    expected_bands: Sequence[str],
    expected_fingerprint_hash: str,
    require_complete: bool = True,
) -> dict[str, Any]:
    reasons: list[str] = []
    status = str(manifest.get("shard_status") or "")
    if manifest.get("artifact_kind") != CROSS_ASSET_STATE_FAMILY_SHARD_MANIFEST_KIND:
        reasons.append("artifact_kind_invalid")
    if manifest.get("artifact_scope") != CROSS_ASSET_STATE_FAMILY_SHARD_SCOPE:
        reasons.append("artifact_scope_invalid")
    if manifest.get("active_handoff_artifact") is not False or manifest.get("not_active_handoff") is not True:
        reasons.append("family_shard_marked_active")
    if manifest.get("active_handoff_eligible") is not False:
        reasons.append("family_shard_active_handoff_eligible_not_false")
    if manifest.get("final_artifact") is not False:
        reasons.append("family_shard_final_artifact_not_false")
    if status != SHARD_STATUS_COMPLETE and manifest.get("partial_artifact") is not True:
        reasons.append("partial_artifact_marker_missing")
    if status == SHARD_STATUS_COMPLETE and manifest.get("partial_artifact") is True:
        reasons.append("complete_shard_marked_partial")
    if require_complete and status != SHARD_STATUS_COMPLETE:
        reasons.append("shard_status_not_complete")
    if status and status not in CROSS_ASSET_STATE_SHARD_STATUSES:
        reasons.append("shard_status_invalid")
    if str(manifest.get("relationship_feature_family") or "") != str(expected_family):
        reasons.append("relationship_feature_family_mismatch")
    if tuple(str(band) for band in manifest.get("bands") or ()) != tuple(str(band) for band in expected_bands):
        reasons.append("bands_mismatch")
    if str(manifest.get("run_fingerprint_hash") or "") != str(expected_fingerprint_hash):
        reasons.append("run_fingerprint_hash_mismatch")
    for field in (
        "production_approved",
        "production_writer_enabled",
        "production_labels_written",
        "production_outputs_written",
        "canonical_production_state_outputs_written",
    ):
        if manifest.get(field) is not False:
            reasons.append(f"{field}_not_false")
    if manifest.get("requires_human_approval_before_production") is not True:
        reasons.append("requires_human_approval_before_production_invalid")
    counts = manifest.get("counts") if isinstance(manifest.get("counts"), Mapping) else {}
    expected = int(counts.get("expected_cells") or 0)
    covered = (
        int(counts.get("selected_model_facing_cells") or 0)
        + int(counts.get("diagnostic_only_cells") or 0)
        + int(counts.get("masked_unavailable_cells") or 0)
        + int(counts.get("missing_cells") or 0)
    )
    if status == SHARD_STATUS_COMPLETE and expected != covered:
        reasons.append("count_completeness_mismatch")
    paths = manifest.get("paths") if isinstance(manifest.get("paths"), Mapping) else {}
    for key in ("summary_path", "selected_profiles_path"):
        value = paths.get(key)
        if not value:
            reasons.append(f"{key}_missing")
            continue
        if not Path(value).is_file():
            reasons.append(f"{key}_not_found")
    selected_path = paths.get("selected_profiles_path")
    if selected_path and Path(selected_path).is_file():
        try:
            selected_payload = read_json_mapping(selected_path)
        except Exception as exc:
            reasons.append(f"selected_profiles_malformed:{type(exc).__name__}")
        else:
            if selected_payload.get("artifact_scope") != CROSS_ASSET_STATE_FAMILY_SHARD_SCOPE:
                reasons.append("selected_profiles_artifact_scope_not_family_shard")
            if selected_payload.get("active_handoff_artifact") is not False or selected_payload.get("not_active_handoff") is not True:
                reasons.append("selected_profiles_marked_active")
            if selected_payload.get("single_active_nonproduction_handoff_artifact") not in (None, ""):
                reasons.append("selected_profiles_single_active_handoff_not_cleared")
            embedded = selected_payload.get("selection_engine_manifest_validation")
            if isinstance(embedded, Mapping) and embedded.get("passed") is not True:
                reasons.append("selected_profiles_validation_blocked")
    return {
        "artifact_kind": "cross_asset_state_family_shard_manifest_validation",
        "status": "passed" if not reasons else "blocked",
        "passed": not reasons,
        "reason_codes": list(dict.fromkeys(reasons)),
        "manifest_path": str(manifest_path) if manifest_path is not None else None,
        "relationship_feature_family": str(expected_family),
        "shard_status": status,
        "expected_fingerprint_hash": str(expected_fingerprint_hash),
        "production_write_allowed": False,
    }


__all__ = [
    "CROSS_ASSET_STATE_DEFAULT_COMBINED_SCOPE",
    "CROSS_ASSET_STATE_FAMILY_SHARD_MANIFEST_FILENAME",
    "CROSS_ASSET_STATE_FAMILY_SHARD_MANIFEST_KIND",
    "CROSS_ASSET_STATE_FAMILY_SHARD_SCOPE",
    "CROSS_ASSET_STATE_HEARTBEAT_SCOPE",
    "CROSS_ASSET_STATE_PROGRESS_HEARTBEAT_KIND",
    "CROSS_ASSET_STATE_RUNTIME_TELEMETRY_EVENT_KIND",
    "CROSS_ASSET_STATE_RUNTIME_TELEMETRY_EVENTS_FILENAME",
    "CROSS_ASSET_STATE_RUNTIME_TELEMETRY_SCOPE",
    "SHARD_STATUS_COMPLETE",
    "SHARD_STATUS_FAILED",
    "SHARD_STATUS_INCOMPLETE",
    "SHARD_STATUS_PENDING",
    "SHARD_STATUS_RUNNING",
    "aggregate_runtime_telemetry_events",
    "build_family_run_fingerprint",
    "family_shard_manifest_payload",
    "family_task_worker_id",
    "read_json_mapping",
    "safe_family_token",
    "stable_json_hash",
    "utc_now_iso",
    "validate_family_shard_manifest",
    "write_json_atomic",
    "write_progress_heartbeat",
    "write_runtime_telemetry_event",
]
