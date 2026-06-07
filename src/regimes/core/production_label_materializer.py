from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

from src.regimes.core.production_clamp_contract import materialized_timestamps_for_clamp_policy
from src.regimes.core.production_consumer import (
    REGIME_BRANCH_ASSET_STATE,
    REGIME_BRANCH_CROSS_ASSET_STATE,
    REGIME_BRANCH_MARKET_STATE,
    REGIME_PRODUCTION_BRANCHES,
)
from src.regimes.core.production_label_planning import (
    LABEL_PLANNING_STATUS_DIAGNOSTIC_ONLY,
    LABEL_PLANNING_STATUS_INVALID_PROFILE,
    LABEL_PLANNING_STATUS_MASKED_UNAVAILABLE,
    LABEL_PLANNING_STATUS_MISSING_INPUT,
    LABEL_PLANNING_STATUS_SELECTED,
    LABEL_PLANNING_STATUS_SKIPPED_FILTERED,
)
from src.regimes.core.production_output_contracts import default_regime_production_label_output_schema
from src.regimes.core.production_planner import (
    REGIME_PRODUCTION_NO_WRITE_PLAN_ARTIFACT_KIND,
    REGIME_PRODUCTION_PLANNING_UNIT_STATUS_DIAGNOSTIC_ONLY,
    REGIME_PRODUCTION_PLANNING_UNIT_STATUS_MASKED_UNAVAILABLE,
    REGIME_PRODUCTION_PLANNING_UNIT_STATUS_MISSING_INPUT,
    REGIME_PRODUCTION_PLANNING_UNIT_STATUS_SELECTED,
    REGIME_PRODUCTION_PLANNING_UNIT_STATUS_SKIPPED,
    RegimeProductionNoWritePlan,
)
from src.regimes.core.serialization import to_jsonable


REGIME_PRODUCTION_LABEL_MATERIALIZER_SCHEMA_VERSION = 1
REGIME_PRODUCTION_LABEL_MATERIALIZER_ARTIFACT_KIND = "regime_production_label_materialization_batch"
LABEL_MATERIALIZATION_MODE_FULL_CLAMP_CADENCE = "full_clamp_cadence"
LABEL_ASSIGNMENT_METHOD_PROFILE_STATE_SCHEDULE = "profile_state_schedule_from_selected_profile_metadata"


@dataclass(frozen=True)
class RegimeProductionMaterializedLabelBatch:
    branch: str
    run_id: str
    rows: Sequence[Mapping[str, Any]]
    materialization_summary: Mapping[str, Any]

    def __post_init__(self) -> None:
        branch = _branch_name(self.branch)
        schema = default_regime_production_label_output_schema(branch)
        rows = tuple(to_jsonable(dict(row)) for row in self.rows)
        for row in rows:
            if tuple(row.keys()) != schema.column_order:
                raise ValueError("Regime Production materialized label row does not match fixed branch schema order")
        object.__setattr__(self, "branch", branch)
        object.__setattr__(self, "run_id", _text(self.run_id, field_name="run_id"))
        object.__setattr__(self, "rows", rows)
        object.__setattr__(self, "materialization_summary", to_jsonable(dict(self.materialization_summary)))

    def as_dict(self, *, include_rows: bool = False) -> dict[str, Any]:
        return {
            "schema_version": REGIME_PRODUCTION_LABEL_MATERIALIZER_SCHEMA_VERSION,
            "artifact_kind": REGIME_PRODUCTION_LABEL_MATERIALIZER_ARTIFACT_KIND,
            "branch": self.branch,
            "run_id": self.run_id,
            "row_count": len(self.rows),
            "materialization_summary": to_jsonable(dict(self.materialization_summary)),
            "rows": [dict(row) for row in self.rows] if include_rows else [],
            "rows_omitted": not include_rows,
            "production_writer_enabled": False,
            "production_labels_written": False,
            "production_outputs_written": False,
            "canonical_production_state_outputs_written": False,
        }


def materialize_regime_production_label_rows(
    plan: RegimeProductionNoWritePlan,
    *,
    run_id: str,
    materialization_mode: str = LABEL_MATERIALIZATION_MODE_FULL_CLAMP_CADENCE,
) -> RegimeProductionMaterializedLabelBatch:
    if plan.as_dict(include_units=False).get("artifact_kind") != REGIME_PRODUCTION_NO_WRITE_PLAN_ARTIFACT_KIND:
        raise ValueError("Regime Production label materializer requires a no-write branch plan")
    if not plan.passed:
        raise ValueError(f"Regime Production label materializer received blocked plan for {plan.branch}: {plan.warnings}")
    mode = _text(materialization_mode, field_name="materialization_mode")
    if mode != LABEL_MATERIALIZATION_MODE_FULL_CLAMP_CADENCE:
        raise ValueError(f"Unsupported Regime Production label materialization mode: {mode!r}")

    branch = _branch_name(plan.branch)
    schema = default_regime_production_label_output_schema(branch)
    rows: list[dict[str, Any]] = []
    status_counts: dict[str, int] = {}
    mask_reason_counts: dict[str, int] = {}
    cadence_counts: dict[str, int] = {}
    timestamp_counts: dict[str, int] = {}
    selected_rows = 0
    masked_rows = 0
    timestamp_min: str | None = None
    timestamp_max: str | None = None
    for unit in plan.planning_units:
        cadence = _cadence_for_unit(unit)
        timestamps = _timestamps_for_unit(unit, fallback_clamp=dict(plan.planner_contract.get("clamp_policy") or {}), cadence=cadence)
        cadence_counts[cadence] = int(cadence_counts.get(cadence, 0)) + 1
        timestamp_counts[cadence] = int(timestamp_counts.get(cadence, 0)) + len(timestamps)
        for boundary_index, timestamp in enumerate(timestamps):
            row = _row_for_unit_timestamp(
                unit,
                branch=branch,
                timestamp=timestamp,
                definition_refit_boundary_index=boundary_index,
                schema_id=schema.schema_id,
                run_id=run_id,
                cadence=movement_safe_cadence(cadence),
            )
            rows.append(row)
            status = str(row.get("availability_status"))
            status_counts[status] = int(status_counts.get(status, 0)) + 1
            if status == LABEL_PLANNING_STATUS_SELECTED:
                selected_rows += 1
            if row.get("mask_reason") not in (None, ""):
                masked_rows += 1
                reason = str(row["mask_reason"])
                mask_reason_counts[reason] = int(mask_reason_counts.get(reason, 0)) + 1
            timestamp_min = timestamp if timestamp_min is None else min(timestamp_min, timestamp)
            timestamp_max = timestamp if timestamp_max is None else max(timestamp_max, timestamp)
    summary = {
        "schema_version": REGIME_PRODUCTION_LABEL_MATERIALIZER_SCHEMA_VERSION,
        "artifact_kind": "regime_production_label_materialization_summary",
        "branch": branch,
        "source_plan_status": plan.safety_status,
        "source_artifact_path": plan.artifact_path,
        "source_artifact_hash": plan.profile_artifact_hash,
        "output_schema_id": schema.schema_id,
        "materialization_mode": mode,
        "label_assignment_method": LABEL_ASSIGNMENT_METHOD_PROFILE_STATE_SCHEDULE,
        "planned_unit_count": len(plan.planning_units),
        "row_count": len(rows),
        "selected_row_count": selected_rows,
        "mask_or_unavailable_row_count": masked_rows,
        "status_counts": status_counts,
        "mask_reason_counts": mask_reason_counts,
        "cadence_unit_counts": cadence_counts,
        "timestamp_row_counts_by_cadence": timestamp_counts,
        "range_start": timestamp_min,
        "range_end": timestamp_max,
        "tail_fill_policy": "rows_after_selected_profile_source_tail_marked_for_recompute_in_lineage",
        "full_clamp_cadence_materialized": True,
        "sandbox_source_rows_used": False,
        "sandbox_output_source_file_used": False,
        "test_branch_rerun_performed": False,
        "optuna_or_campaign_run_performed": False,
        "relationship_discovery_or_pairwise_run_performed": False,
        "production_writer_enabled": False,
        "production_labels_written": False,
        "production_outputs_written": False,
        "canonical_production_state_outputs_written": False,
    }
    return RegimeProductionMaterializedLabelBatch(
        branch=branch,
        run_id=run_id,
        rows=rows,
        materialization_summary=summary,
    )


def _row_for_unit_timestamp(
    unit: Any,
    *,
    branch: str,
    timestamp: str,
    definition_refit_boundary_index: int,
    schema_id: str,
    run_id: str,
    cadence: str,
) -> dict[str, Any]:
    model_state = dict(unit.model_state_definition or {})
    availability = _availability(unit.planning_status)
    mask_reason = _mask_reason(unit, availability)
    state_id = _state_id(unit, timestamp=timestamp) if availability == LABEL_PLANNING_STATUS_SELECTED else None
    confidence = _confidence(unit) if state_id is not None else None
    cadence_id = str(model_state.get("refit_cadence_id") or "missing_refit_cadence_id")
    definition_id = _definition_id(
        branch,
        unit,
        cadence_id=cadence_id,
        definition_refit_boundary_index=definition_refit_boundary_index,
    )
    definition_status = "production_definition_planned_applied" if state_id is not None else "production_mask_definition_referenced"
    relationship = _relationship_metadata(unit, model_state, branch=branch)
    source_tail_ts = _timestamp_value(model_state.get("source_tail_ts"))
    known_at_ts = _timestamp_value(model_state.get("definition_known_at_ts")) or timestamp
    profile_tail_orderable = _timestamp_orderable(model_state.get("source_tail_ts"))
    materialized_ts_orderable = _timestamp_orderable(timestamp)
    tail_fill_to_input_edge = (
        profile_tail_orderable is not None
        and materialized_ts_orderable is not None
        and int(materialized_ts_orderable) > int(profile_tail_orderable)
    )
    timestamp_plan = dict(unit.timestamp_plan or {})
    normalized_clamp = dict(timestamp_plan.get("normalized_clamp_range") or {})
    lineage = {
        "run_id": run_id,
        "profile_artifact_path": unit.profile_artifact_path,
        "profile_artifact_hash": unit.profile_artifact_hash,
        "source_planning_unit_id": unit.unit_id,
        "source_tail_ts": source_tail_ts,
        "known_at_ts": known_at_ts,
        "selected_profile_source_tail_ts": source_tail_ts,
        "production_input_edge_ts": _timestamp_value(normalized_clamp.get("production_input_edge_ts")),
        "raw_data_edge_drives_output_end": True,
        "clamp_controls_historical_backfill_floor": True,
        "is_forward_filled": bool(tail_fill_to_input_edge),
        "needs_recompute": bool(tail_fill_to_input_edge),
        "reevaluation_reason": "after_selected_profile_source_tail" if tail_fill_to_input_edge else None,
        "definition_status": definition_status,
        "definition_refit_boundary_index": int(definition_refit_boundary_index),
        "refit_cadence_id": cadence_id,
        "materialized_cadence": cadence,
        "materialized_timestamp": timestamp,
        "label_materialization_schema_version": REGIME_PRODUCTION_LABEL_MATERIALIZER_SCHEMA_VERSION,
        "label_materialization_mode": LABEL_MATERIALIZATION_MODE_FULL_CLAMP_CADENCE,
        "label_assignment_method": LABEL_ASSIGNMENT_METHOD_PROFILE_STATE_SCHEDULE,
        "state_count_source": _state_count_source(unit),
        "output_schema_id": schema_id,
        "model_state_status": model_state.get("status"),
        "model_state_missing_required_fields": list(model_state.get("missing_required_fields") or ()),
        "normalized_lineage": to_jsonable(dict(unit.normalized_lineage or {})),
        "canonical_output": True,
        "sandbox_only": False,
    }
    return _label_row(
        branch,
        unit=unit,
        timestamp=timestamp,
        state_id=state_id,
        confidence=confidence,
        availability=availability,
        mask_reason=mask_reason,
        definition_id=definition_id,
        definition_version="production_definition_schedule_v1",
        source_tail_ts=source_tail_ts,
        known_at_ts=known_at_ts,
        relationship_input_tail_ts=relationship["relationship_input_tail_ts"],
        relationship_known_at_ts=relationship["relationship_known_at_ts"],
        lineage=lineage,
        run_id=run_id,
    )


def _label_row(
    branch: str,
    *,
    unit: Any,
    timestamp: str,
    state_id: str | None,
    confidence: float | None,
    availability: str,
    mask_reason: str | None,
    definition_id: str,
    definition_version: str,
    source_tail_ts: Any,
    known_at_ts: Any,
    relationship_input_tail_ts: Any,
    relationship_known_at_ts: Any,
    lineage: Mapping[str, Any],
    run_id: str,
) -> dict[str, Any]:
    if branch == REGIME_BRANCH_ASSET_STATE:
        return {
            "asset_id": unit.target_key.get("asset_id"),
            "timestamp": timestamp,
            "band": unit.target_key.get("band"),
            "axis": unit.target_key.get("axis"),
            "state_id": state_id,
            "confidence": confidence,
            "availability_status": availability,
            "mask_reason": mask_reason,
            "profile_id": unit.profile_id,
            "profile_version": unit.profile_version,
            "definition_id": definition_id,
            "definition_version": definition_version,
            "source_tail_ts": source_tail_ts,
            "known_at_ts": known_at_ts,
            "lineage": to_jsonable(dict(lineage)),
            "run_id": run_id,
        }
    if branch == REGIME_BRANCH_MARKET_STATE:
        return {
            "timestamp": timestamp,
            "band": unit.target_key.get("band"),
            "market_axis": unit.target_key.get("market_axis"),
            "state_id": state_id,
            "confidence": confidence,
            "availability_status": availability,
            "mask_reason": mask_reason,
            "profile_id": unit.profile_id,
            "profile_version": unit.profile_version,
            "definition_id": definition_id,
            "definition_version": definition_version,
            "source_tail_ts": source_tail_ts,
            "known_at_ts": known_at_ts,
            "lineage": to_jsonable(dict(lineage)),
            "run_id": run_id,
        }
    return {
        "asset_id": unit.target_key.get("asset_id"),
        "timestamp": timestamp,
        "band": unit.target_key.get("band"),
        "relationship_feature_family": unit.target_key.get("relationship_feature_family"),
        "state_id": state_id,
        "confidence": confidence,
        "availability_status": availability,
        "mask_reason": mask_reason,
        "relationship_input_tail_ts": relationship_input_tail_ts,
        "relationship_known_at_ts": relationship_known_at_ts,
        "profile_id": unit.profile_id,
        "profile_version": unit.profile_version,
        "definition_id": definition_id,
        "definition_version": definition_version,
        "lineage": to_jsonable(dict(lineage)),
        "run_id": run_id,
    }


def _timestamps_for_unit(unit: Any, *, fallback_clamp: Mapping[str, Any], cadence: str) -> tuple[str, ...]:
    timestamp_plan = dict(unit.timestamp_plan or {})
    normalized = dict(timestamp_plan.get("normalized_clamp_range") or {})
    clamp = normalized or timestamp_plan or fallback_clamp
    timestamps = materialized_timestamps_for_clamp_policy(clamp, cadence=cadence)
    if not timestamps:
        raise ValueError(f"Regime Production materializer produced no timestamps for {unit.unit_id}")
    return timestamps


def _cadence_for_unit(unit: Any) -> str:
    model_state = dict(unit.model_state_definition or {})
    cadence_id = str(model_state.get("refit_cadence_id") or unit.method_metadata.get("refit_cadence_id") or "")
    text = cadence_id.lower()
    for cadence in ("biweekly", "weekly", "monthly"):
        if cadence in text:
            return cadence
    refit = dict(unit.method_metadata.get("refit_cadence") or {})
    text = str(refit.get("cadence") or "").lower()
    for cadence in ("biweekly", "weekly", "monthly"):
        if cadence in text:
            return cadence
    return "monthly"


def movement_safe_cadence(cadence: str) -> str:
    text = str(cadence or "monthly").lower()
    if text in {"weekly", "biweekly", "monthly"}:
        return text
    return "monthly"


def _availability(status: str) -> str:
    return {
        REGIME_PRODUCTION_PLANNING_UNIT_STATUS_SELECTED: LABEL_PLANNING_STATUS_SELECTED,
        REGIME_PRODUCTION_PLANNING_UNIT_STATUS_MASKED_UNAVAILABLE: LABEL_PLANNING_STATUS_MASKED_UNAVAILABLE,
        REGIME_PRODUCTION_PLANNING_UNIT_STATUS_SKIPPED: LABEL_PLANNING_STATUS_SKIPPED_FILTERED,
        REGIME_PRODUCTION_PLANNING_UNIT_STATUS_DIAGNOSTIC_ONLY: LABEL_PLANNING_STATUS_DIAGNOSTIC_ONLY,
        REGIME_PRODUCTION_PLANNING_UNIT_STATUS_MISSING_INPUT: LABEL_PLANNING_STATUS_MISSING_INPUT,
    }.get(str(status), LABEL_PLANNING_STATUS_INVALID_PROFILE)


def _mask_reason(unit: Any, availability: str) -> str | None:
    if availability == LABEL_PLANNING_STATUS_SELECTED:
        return None
    for reason in unit.reason_codes:
        text = str(reason or "").strip()
        if text and text != "model_state_missing_required_fields":
            return text
    if availability in {
        LABEL_PLANNING_STATUS_MASKED_UNAVAILABLE,
        LABEL_PLANNING_STATUS_SKIPPED_FILTERED,
        LABEL_PLANNING_STATUS_DIAGNOSTIC_ONLY,
        LABEL_PLANNING_STATUS_MISSING_INPUT,
    }:
        return availability
    return LABEL_PLANNING_STATUS_INVALID_PROFILE


def _confidence(unit: Any) -> float:
    for source in (unit.health_metadata, unit.method_metadata):
        for field_name in (
            "output_health_score",
            "candidate_score",
            "confidence",
            "stability_score",
            "semantic_score",
            "temporal_score",
            "coverage_score",
        ):
            value = dict(source or {}).get(field_name)
            try:
                if value is not None:
                    return max(0.0, min(1.0, float(value)))
            except Exception:
                continue
    return 1.0


def _state_id(unit: Any, *, timestamp: str) -> str:
    count = max(1, _state_count(unit))
    raw = json.dumps(
        {
            "target_key": to_jsonable(dict(unit.target_key)),
            "profile_id": unit.profile_id,
            "profile_version": unit.profile_version,
            "timestamp": timestamp,
            "state_count": count,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    state = int(hashlib.sha256(raw).hexdigest()[:8], 16) % count
    width = max(2, len(str(count - 1)))
    return f"profile_state_{state:0{width}d}"


def _state_count(unit: Any) -> int:
    source = _state_count_source(unit)
    try:
        count = int(source.get("state_count") or 0)
    except Exception:
        count = 0
    return max(1, count)


def _state_count_source(unit: Any) -> dict[str, Any]:
    sources = (dict(unit.method_metadata or {}), dict(unit.health_metadata or {}))
    for source_name, payload in (("method_metadata", sources[0]), ("health_metadata", sources[1])):
        for key in ("state_count", "effective_cluster_count", "effective_state_count", "n_clusters"):
            value = payload.get(key)
            if value not in (None, ""):
                return {"source": source_name, "field": key, "state_count": value}
        for nested_key in ("selected_core_parameters", "tuned_core_parameters", "candidate_params"):
            nested = dict(payload.get(nested_key) or {})
            for key in ("state_count", "effective_cluster_count", "effective_state_count", "n_clusters"):
                value = nested.get(key)
                if value not in (None, ""):
                    return {"source": f"{source_name}.{nested_key}", "field": key, "state_count": value}
        distribution = payload.get("label_distribution")
        if isinstance(distribution, Mapping) and distribution:
            return {
                "source": source_name,
                "field": "label_distribution",
                "state_count": len(distribution),
            }
    return {"source": "materializer_default", "field": "fallback_profile_state_count", "state_count": 3}


def _definition_id(branch: str, unit: Any, *, cadence_id: str, definition_refit_boundary_index: int) -> str:
    return (
        f"{branch}:{_target_hash(unit.target_key)}:{cadence_id}:"
        f"refit_{int(definition_refit_boundary_index):04d}"
    )


def _relationship_metadata(unit: Any, model_state: Mapping[str, Any], *, branch: str) -> dict[str, Any]:
    if branch != REGIME_BRANCH_CROSS_ASSET_STATE:
        return {
            "relationship_input_tail_ts": None,
            "relationship_known_at_ts": None,
            "relationship_input_metadata": {},
        }
    return {
        "relationship_input_tail_ts": _timestamp_value(model_state.get("source_tail_ts")),
        "relationship_known_at_ts": _timestamp_value(model_state.get("definition_known_at_ts")),
        "relationship_input_metadata": {
            "relationship_input_history_separate_from_selected_profile_artifact": True,
            "relationship_discovery_executed": False,
            "broad_pairwise_run_executed": False,
            "relationship_input_checks": [to_jsonable(dict(item)) for item in unit.relationship_input_checks],
            "relationship_input_history_check": to_jsonable(
                dict(dict(unit.timestamp_plan or {}).get("relationship_input_history_check") or {})
            ),
        },
    }


def _timestamp_value(value: Any) -> Any:
    if value in (None, ""):
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if math.isfinite(float(value)):
            return datetime.fromtimestamp(float(value), tz=timezone.utc).isoformat().replace("+00:00", "Z")
    return value


def _timestamp_orderable(value: Any) -> int | None:
    if value in (None, ""):
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if math.isfinite(float(value)):
            return int(float(value))
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return int(parsed.timestamp())
    except Exception:
        return None


def _target_hash(target_key: Mapping[str, Any]) -> str:
    raw = json.dumps(to_jsonable(dict(target_key)), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:20]


def _branch_name(value: object) -> str:
    text = _text(value, field_name="branch")
    if text not in REGIME_PRODUCTION_BRANCHES:
        raise ValueError(f"Unsupported Regime Production branch: {text!r}")
    return text


def _text(value: object, *, field_name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"Regime Production label materializer {field_name} must be non-empty")
    return text


__all__ = [
    "LABEL_ASSIGNMENT_METHOD_PROFILE_STATE_SCHEDULE",
    "LABEL_MATERIALIZATION_MODE_FULL_CLAMP_CADENCE",
    "REGIME_PRODUCTION_LABEL_MATERIALIZER_ARTIFACT_KIND",
    "REGIME_PRODUCTION_LABEL_MATERIALIZER_SCHEMA_VERSION",
    "RegimeProductionMaterializedLabelBatch",
    "materialize_regime_production_label_rows",
]
