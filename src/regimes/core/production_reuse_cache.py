from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from src.regimes.core.serialization import to_jsonable


REGIME_PRODUCTION_REUSE_CACHE_SCHEMA_VERSION = 1
REGIME_PRODUCTION_REUSE_CACHE_ARTIFACT_KIND = "regime_production_reuse_cache_telemetry"
REGIME_PRODUCTION_PROFILE_LOOKUP_INDEX_ARTIFACT_KIND = "regime_production_profile_lookup_index"


@dataclass(frozen=True)
class RegimeProductionProfileLookupIndex:
    branch: str
    artifact_hash: str
    source_tail_fingerprint: str
    selected_by_grain: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    unavailable_by_grain: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    diagnostic_by_grain: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    duplicate_grain_keys: Sequence[str] = ()

    @property
    def selected_count(self) -> int:
        return len(self.selected_by_grain)

    @property
    def unavailable_count(self) -> int:
        return len(self.unavailable_by_grain)

    @property
    def diagnostic_count(self) -> int:
        return len(self.diagnostic_by_grain)

    def as_dict(self, *, include_records: bool = False) -> dict[str, Any]:
        payload = {
            "schema_version": REGIME_PRODUCTION_REUSE_CACHE_SCHEMA_VERSION,
            "artifact_kind": REGIME_PRODUCTION_PROFILE_LOOKUP_INDEX_ARTIFACT_KIND,
            "branch": self.branch,
            "artifact_hash": self.artifact_hash,
            "source_tail_fingerprint": self.source_tail_fingerprint,
            "selected_count": self.selected_count,
            "unavailable_count": self.unavailable_count,
            "diagnostic_count": self.diagnostic_count,
            "duplicate_grain_keys": list(self.duplicate_grain_keys),
            "index_valid": not bool(self.duplicate_grain_keys),
            "production_outputs_written": False,
            "production_labels_written": False,
            "canonical_production_state_outputs_written": False,
        }
        if include_records:
            payload.update(
                {
                    "selected_by_grain": to_jsonable(dict(self.selected_by_grain)),
                    "unavailable_by_grain": to_jsonable(dict(self.unavailable_by_grain)),
                    "diagnostic_by_grain": to_jsonable(dict(self.diagnostic_by_grain)),
                }
            )
        return payload


class RegimeProductionPlannerRunCache:
    """Explicit per-run cache for no-write Regime Production planning.

    Entries are invalidated by path stat fingerprints, validation config fingerprints,
    artifact hashes, and source-tail fingerprints. The cache does not cross process
    or run boundaries unless the caller intentionally reuses the object.
    """

    def __init__(self, *, cache_id: str = "regime_production_planner_run_cache") -> None:
        self.cache_id = str(cache_id or "regime_production_planner_run_cache")
        self._artifact_resolutions: dict[str, dict[str, Any]] = {}
        self._artifact_hashes: dict[str, dict[str, Any]] = {}
        self._profile_indexes: dict[str, RegimeProductionProfileLookupIndex] = {}
        self._relationship_input_checks: dict[str, tuple[Mapping[str, Any], ...]] = {}
        self._definition_state_plans: dict[str, Mapping[str, Any]] = {}
        self._timestamp_plans: dict[str, Mapping[str, Any]] = {}
        self._counters: dict[str, int] = {
            "artifact_resolution_hits": 0,
            "artifact_resolution_misses": 0,
            "artifact_hash_hits": 0,
            "artifact_hash_misses": 0,
            "profile_lookup_index_hits": 0,
            "profile_lookup_index_misses": 0,
            "relationship_input_index_hits": 0,
            "relationship_input_index_misses": 0,
            "definition_state_plan_hits": 0,
            "definition_state_plan_misses": 0,
            "timestamp_plan_hits": 0,
            "timestamp_plan_misses": 0,
        }

    def artifact_resolution(
        self,
        *,
        branch: str,
        path: str | Path,
        source: str,
        config_fingerprint: Mapping[str, Any] | str | None,
    ) -> Any | None:
        key = self._artifact_key(branch=branch, path=path, source=source, config_fingerprint=config_fingerprint)
        stat = _path_stat(path)
        entry = self._artifact_resolutions.get(key)
        if entry is not None and entry.get("path_stat") == stat:
            self._counters["artifact_resolution_hits"] += 1
            return entry["value"]
        self._counters["artifact_resolution_misses"] += 1
        return None

    def put_artifact_resolution(
        self,
        value: Any,
        *,
        branch: str,
        path: str | Path,
        source: str,
        config_fingerprint: Mapping[str, Any] | str | None,
        source_tail_fingerprint: str | None = None,
    ) -> None:
        key = self._artifact_key(branch=branch, path=path, source=source, config_fingerprint=config_fingerprint)
        self._artifact_resolutions[key] = {
            "path_stat": _path_stat(path),
            "source_tail_fingerprint": source_tail_fingerprint,
            "value": value,
        }

    def artifact_hash(self, path: str | Path, loader: Callable[[str | Path], str]) -> str:
        resolved = str(Path(path).resolve())
        stat = _path_stat(path)
        entry = self._artifact_hashes.get(resolved)
        if entry is not None and entry.get("path_stat") == stat:
            self._counters["artifact_hash_hits"] += 1
            return str(entry["value"])
        self._counters["artifact_hash_misses"] += 1
        value = str(loader(path))
        self._artifact_hashes[resolved] = {"path_stat": stat, "value": value}
        return value

    def profile_lookup_index(
        self,
        *,
        branch: str,
        artifact_hash: str,
        source_tail_fingerprint: str,
        config_fingerprint: Mapping[str, Any] | str | None,
        builder: Callable[[], RegimeProductionProfileLookupIndex],
    ) -> RegimeProductionProfileLookupIndex:
        key = _stable_fingerprint(
            {
                "kind": "profile_lookup_index",
                "branch": branch,
                "artifact_hash": artifact_hash,
                "source_tail_fingerprint": source_tail_fingerprint,
                "config_fingerprint": _config_fingerprint(config_fingerprint),
            }
        )
        cached = self._profile_indexes.get(key)
        if cached is not None:
            self._counters["profile_lookup_index_hits"] += 1
            return cached
        self._counters["profile_lookup_index_misses"] += 1
        built = builder()
        self._profile_indexes[key] = built
        return built

    def relationship_input_checks(
        self,
        *,
        manifest_fingerprint: str,
        env_fingerprint: Mapping[str, Any] | str | None,
        builder: Callable[[], tuple[Mapping[str, Any], ...]],
    ) -> tuple[Mapping[str, Any], ...]:
        key = _stable_fingerprint(
            {
                "kind": "relationship_input_checks",
                "manifest_fingerprint": manifest_fingerprint,
                "env_fingerprint": _config_fingerprint(env_fingerprint),
            }
        )
        cached = self._relationship_input_checks.get(key)
        if cached is not None:
            self._counters["relationship_input_index_hits"] += 1
            return tuple(copy.deepcopy(item) for item in cached)
        self._counters["relationship_input_index_misses"] += 1
        built = tuple(builder())
        self._relationship_input_checks[key] = tuple(copy.deepcopy(item) for item in built)
        return built

    def definition_state_plan(
        self,
        *,
        key_payload: Mapping[str, Any],
        builder: Callable[[], Mapping[str, Any]],
    ) -> Mapping[str, Any]:
        key = _stable_fingerprint({"kind": "definition_state_plan", "payload": to_jsonable(dict(key_payload))})
        cached = self._definition_state_plans.get(key)
        if cached is not None:
            self._counters["definition_state_plan_hits"] += 1
            return copy.deepcopy(cached)
        self._counters["definition_state_plan_misses"] += 1
        built = to_jsonable(dict(builder()))
        self._definition_state_plans[key] = copy.deepcopy(built)
        return built

    def timestamp_plan(
        self,
        *,
        key_payload: Mapping[str, Any],
        builder: Callable[[], Mapping[str, Any]],
    ) -> Mapping[str, Any]:
        key = _stable_fingerprint({"kind": "timestamp_plan", "payload": to_jsonable(dict(key_payload))})
        cached = self._timestamp_plans.get(key)
        if cached is not None:
            self._counters["timestamp_plan_hits"] += 1
            return copy.deepcopy(cached)
        self._counters["timestamp_plan_misses"] += 1
        built = to_jsonable(dict(builder()))
        self._timestamp_plans[key] = copy.deepcopy(built)
        return built

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": REGIME_PRODUCTION_REUSE_CACHE_SCHEMA_VERSION,
            "artifact_kind": REGIME_PRODUCTION_REUSE_CACHE_ARTIFACT_KIND,
            "cache_id": self.cache_id,
            "enabled": True,
            "scope": "explicit_per_run_object",
            "invalidation": {
                "active_artifact_resolution": "resolved_path_stat_plus_validation_config",
                "artifact_hash": "resolved_path_stat",
                "profile_lookup_index": "artifact_hash_plus_source_tail_fingerprint_plus_config",
                "relationship_input_index": "sidecar_path_fields_plus_env_root_fingerprint",
                "definition_state_plan": "target_profile_tail_known_at_refit_payload",
                "timestamp_plan": "branch_contract_source_tail_known_at_status_payload",
            },
            "counters": dict(self._counters),
            "entry_counts": {
                "artifact_resolution": len(self._artifact_resolutions),
                "artifact_hash": len(self._artifact_hashes),
                "profile_lookup_index": len(self._profile_indexes),
                "relationship_input_index": len(self._relationship_input_checks),
                "definition_state_plan": len(self._definition_state_plans),
                "timestamp_plan": len(self._timestamp_plans),
            },
            "production_outputs_written": False,
            "production_labels_written": False,
            "canonical_production_state_outputs_written": False,
        }

    def _artifact_key(
        self,
        *,
        branch: str,
        path: str | Path,
        source: str,
        config_fingerprint: Mapping[str, Any] | str | None,
    ) -> str:
        return _stable_fingerprint(
            {
                "kind": "artifact_resolution",
                "branch": str(branch),
                "path": str(Path(path).resolve()),
                "source": str(source),
                "config_fingerprint": _config_fingerprint(config_fingerprint),
            }
        )


def build_profile_lookup_index(
    *,
    branch: str,
    artifact_hash: str,
    target_fields: Sequence[str],
    selected_records: Sequence[Mapping[str, Any]] = (),
    unavailable_records: Sequence[Mapping[str, Any]] = (),
    diagnostic_records: Sequence[Mapping[str, Any]] = (),
) -> RegimeProductionProfileLookupIndex:
    selected, selected_duplicates = _records_by_grain(selected_records, target_fields)
    unavailable, unavailable_duplicates = _records_by_grain(unavailable_records, target_fields)
    diagnostic, diagnostic_duplicates = _records_by_grain(diagnostic_records, target_fields)
    duplicate_keys = tuple(dict.fromkeys((*selected_duplicates, *unavailable_duplicates, *diagnostic_duplicates)))
    return RegimeProductionProfileLookupIndex(
        branch=str(branch),
        artifact_hash=str(artifact_hash),
        source_tail_fingerprint=source_tail_fingerprint((*selected_records, *unavailable_records, *diagnostic_records)),
        selected_by_grain=selected,
        unavailable_by_grain=unavailable,
        diagnostic_by_grain=diagnostic,
        duplicate_grain_keys=duplicate_keys,
    )


def source_tail_fingerprint(records_or_manifest: Any) -> str:
    values: list[Any] = []
    if isinstance(records_or_manifest, Mapping):
        for field_name in ("source_tail_ts", "known_at_ts", "relationship_input_tail_ts", "relationship_known_at_ts"):
            if records_or_manifest.get(field_name) not in (None, ""):
                values.append((field_name, records_or_manifest.get(field_name)))
        for key in ("selected_profiles", "profiles", "masked_or_skipped_cells", "diagnostic_only_profiles"):
            value = records_or_manifest.get(key)
            if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                values.extend(_record_tail_values(value))
    elif isinstance(records_or_manifest, Sequence) and not isinstance(records_or_manifest, (str, bytes)):
        values.extend(_record_tail_values(records_or_manifest))
    return _stable_fingerprint(values)


def relationship_manifest_fingerprint(manifest: Mapping[str, Any]) -> str:
    payload = {
        "relationship_context_handoff_path": manifest.get("relationship_context_handoff_path"),
        "eligibility_manifest_path": manifest.get("eligibility_manifest_path"),
        "relationship_context_cadence_policy": to_jsonable(dict(manifest.get("relationship_context_cadence_policy") or {}))
        if isinstance(manifest.get("relationship_context_cadence_policy"), Mapping)
        else None,
        "source_tail_fingerprint": source_tail_fingerprint(manifest),
    }
    return _stable_fingerprint(payload)


def _records_by_grain(
    records: Sequence[Mapping[str, Any]],
    target_fields: Sequence[str],
) -> tuple[dict[str, Mapping[str, Any]], tuple[str, ...]]:
    by_grain: dict[str, Mapping[str, Any]] = {}
    duplicates: list[str] = []
    for row in records:
        key = grain_key(row, target_fields)
        if not key:
            continue
        if key in by_grain:
            duplicates.append(key)
            continue
        by_grain[key] = to_jsonable(dict(row))
    return by_grain, tuple(dict.fromkeys(duplicates))


def grain_key(row: Mapping[str, Any], target_fields: Sequence[str]) -> str:
    values = []
    for field_name in target_fields:
        value = row.get(field_name)
        if value in (None, ""):
            return ""
        values.append(str(value))
    return "|".join(values)


def _record_tail_values(records: Sequence[Any]) -> list[Any]:
    values: list[Any] = []
    for row in records:
        if not isinstance(row, Mapping):
            continue
        values.append(
            {
                "grain_hint": [
                    row.get("asset_id"),
                    row.get("axis") or row.get("market_axis"),
                    row.get("relationship_feature_family"),
                    row.get("band"),
                ],
                "source_tail_ts": row.get("source_tail_ts") or row.get("relationship_input_tail_ts"),
                "known_at_ts": row.get("known_at_ts") or row.get("relationship_known_at_ts"),
            }
        )
    return values


def _path_stat(path: str | Path) -> dict[str, Any]:
    resolved = Path(path)
    try:
        stat = resolved.stat()
    except FileNotFoundError:
        return {"exists": False, "path": str(resolved.resolve())}
    return {
        "exists": True,
        "path": str(resolved.resolve()),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _config_fingerprint(value: Mapping[str, Any] | str | None) -> str:
    if value is None:
        return "none"
    if isinstance(value, str):
        return value
    return _stable_fingerprint(to_jsonable(dict(value)))


def _stable_fingerprint(payload: Any) -> str:
    raw = json.dumps(to_jsonable(payload), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(raw).hexdigest()


__all__ = [
    "REGIME_PRODUCTION_PROFILE_LOOKUP_INDEX_ARTIFACT_KIND",
    "REGIME_PRODUCTION_REUSE_CACHE_ARTIFACT_KIND",
    "REGIME_PRODUCTION_REUSE_CACHE_SCHEMA_VERSION",
    "RegimeProductionPlannerRunCache",
    "RegimeProductionProfileLookupIndex",
    "build_profile_lookup_index",
    "grain_key",
    "relationship_manifest_fingerprint",
    "source_tail_fingerprint",
]
