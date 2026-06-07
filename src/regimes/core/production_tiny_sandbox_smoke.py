from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.regimes.asset_state.production_planner import (
    plan_asset_state_production_no_write,
    plan_default_asset_state_production_no_write,
)
from src.regimes.core.production_consumer import (
    REGIME_BRANCH_ASSET_STATE,
    REGIME_BRANCH_CROSS_ASSET_STATE,
    REGIME_BRANCH_MARKET_STATE,
    REGIME_PRODUCTION_BRANCHES,
)
from src.regimes.core.production_label_planning import (
    LABEL_PLANNING_STATUS_DIAGNOSTIC_ONLY,
    LABEL_PLANNING_STATUS_MASKED_UNAVAILABLE,
    LABEL_PLANNING_STATUS_SELECTED,
    LABEL_PLANNING_STATUS_SKIPPED_FILTERED,
)
from src.regimes.core.production_output_contracts import (
    default_regime_production_label_output_schema,
    resolve_regime_production_sandbox_output_root_contract,
)
from src.regimes.core.production_planner import (
    REGIME_PRODUCTION_PLANNING_UNIT_STATUS_DIAGNOSTIC_ONLY,
    REGIME_PRODUCTION_PLANNING_UNIT_STATUS_MASKED_UNAVAILABLE,
    REGIME_PRODUCTION_PLANNING_UNIT_STATUS_SELECTED,
    REGIME_PRODUCTION_PLANNING_UNIT_STATUS_SKIPPED,
    RegimeProductionNoWritePlan,
    validate_regime_production_planner_gates,
)
from src.regimes.core.serialization import to_jsonable
from src.regimes.cross_asset_state.production_planner import (
    plan_cross_asset_state_production_no_write,
    plan_default_cross_asset_state_production_no_write,
)
from src.regimes.market_state.production_planner import (
    plan_default_market_state_production_no_write,
    plan_market_state_production_no_write,
)


REGIME_PRODUCTION_TINY_SANDBOX_SMOKE_SCHEMA_VERSION = 1
REGIME_PRODUCTION_TINY_SANDBOX_SMOKE_ARTIFACT_KIND = "regime_production_tiny_sandbox_fit_apply_smoke"
REGIME_PRODUCTION_TINY_SANDBOX_ROW_ARTIFACT_KIND = "regime_production_tiny_sandbox_label_like_row"
DEFAULT_TINY_SANDBOX_OUTPUT_ROOT = (
    "_codex_artifacts/reports/regime_production_tiny_sandbox_smoke/"
    "reports/regimes/foundation/regime_tiny_sandbox_smoke_outputs"
)


@dataclass(frozen=True)
class RegimeProductionTinySandboxSmokeConfig:
    asset_state_artifact_path: str | Path | None = None
    market_state_artifact_path: str | Path | None = None
    cross_asset_state_artifact_path: str | Path | None = None
    sandbox_output_root: str | Path | None = DEFAULT_TINY_SANDBOX_OUTPUT_ROOT
    env: Mapping[str, str] | None = None
    run_id: str = "regime_production_tiny_sandbox_smoke"
    write_sandbox_summary_artifact: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "env", dict(os.environ if self.env is None else self.env))
        object.__setattr__(self, "run_id", _text(self.run_id, field_name="run_id"))


@dataclass(frozen=True)
class RegimeProductionTinySandboxSmokeRow:
    branch: str
    label_row: Mapping[str, Any]
    output_schema_id: str
    source_unit_id: str
    source_unit_status: str
    definition_fit_apply: Mapping[str, Any]
    input_resolution: Mapping[str, Any]
    timestamp_range_chunk: Mapping[str, Any]
    writer_enabled: bool = False
    production_labels_written: bool = False
    production_outputs_written: bool = False
    canonical_production_state_outputs_written: bool = False

    def __post_init__(self) -> None:
        branch = _branch_name(self.branch)
        schema = default_regime_production_label_output_schema(branch)
        row = to_jsonable(dict(self.label_row))
        if tuple(row.keys()) != schema.column_order:
            raise ValueError("Regime Production tiny sandbox smoke row does not match fixed branch schema order")
        if self.writer_enabled or self.production_labels_written or self.production_outputs_written or self.canonical_production_state_outputs_written:
            raise ValueError("Regime Production tiny sandbox smoke row cannot enable production writes")
        object.__setattr__(self, "branch", branch)
        object.__setattr__(self, "label_row", row)
        object.__setattr__(self, "output_schema_id", _text(self.output_schema_id, field_name="output_schema_id"))
        object.__setattr__(self, "source_unit_id", _text(self.source_unit_id, field_name="source_unit_id"))
        object.__setattr__(self, "source_unit_status", _text(self.source_unit_status, field_name="source_unit_status"))
        object.__setattr__(self, "definition_fit_apply", to_jsonable(dict(self.definition_fit_apply)))
        object.__setattr__(self, "input_resolution", to_jsonable(dict(self.input_resolution)))
        object.__setattr__(self, "timestamp_range_chunk", to_jsonable(dict(self.timestamp_range_chunk)))

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": REGIME_PRODUCTION_TINY_SANDBOX_SMOKE_SCHEMA_VERSION,
            "artifact_kind": REGIME_PRODUCTION_TINY_SANDBOX_ROW_ARTIFACT_KIND,
            "branch": self.branch,
            "output_schema_id": self.output_schema_id,
            "source_unit_id": self.source_unit_id,
            "source_unit_status": self.source_unit_status,
            "label_row": to_jsonable(dict(self.label_row)),
            "definition_fit_apply": to_jsonable(dict(self.definition_fit_apply)),
            "input_resolution": to_jsonable(dict(self.input_resolution)),
            "timestamp_range_chunk": to_jsonable(dict(self.timestamp_range_chunk)),
            "sandbox_only": True,
            "writer_enabled": False,
            "production_labels_written": False,
            "production_outputs_written": False,
            "canonical_production_state_outputs_written": False,
        }


def run_regime_production_tiny_sandbox_smoke(
    config: RegimeProductionTinySandboxSmokeConfig | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    cfg = config if isinstance(config, RegimeProductionTinySandboxSmokeConfig) else RegimeProductionTinySandboxSmokeConfig(**dict(config or {}))
    rss_start = _rss_bytes()
    child_start = _child_process_count()
    started = time.perf_counter()
    plans = _build_no_write_plans(cfg)
    branch_payloads: dict[str, dict[str, Any]] = {}
    for branch, plan in plans.items():
        branch_payloads[branch] = build_tiny_sandbox_smoke_branch_payload(
            plan,
            run_id=cfg.run_id,
            sandbox_output_root=cfg.sandbox_output_root,
            env=cfg.env,
        )
    elapsed = time.perf_counter() - started
    rss_end = _rss_bytes()
    child_end = _child_process_count()
    payload = {
        "schema_version": REGIME_PRODUCTION_TINY_SANDBOX_SMOKE_SCHEMA_VERSION,
        "artifact_kind": REGIME_PRODUCTION_TINY_SANDBOX_SMOKE_ARTIFACT_KIND,
        "run_id": cfg.run_id,
        "branch_smokes": branch_payloads,
        "row_count_by_branch": {branch: int(item["row_count"]) for branch, item in branch_payloads.items()},
        "mask_or_unavailable_count_by_branch": {branch: int(item["mask_or_unavailable_count"]) for branch, item in branch_payloads.items()},
        "total_row_count": sum(int(item["row_count"]) for item in branch_payloads.values()),
        "total_mask_or_unavailable_count": sum(int(item["mask_or_unavailable_count"]) for item in branch_payloads.values()),
        "runtime_telemetry": {
            "elapsed_seconds": round(float(elapsed), 6),
            "rss_start_bytes": rss_start,
            "rss_end_bytes": rss_end,
            "rss_delta_bytes": None if rss_start is None or rss_end is None else int(rss_end) - int(rss_start),
            "child_process_count_start": child_start,
            "child_process_count_end": child_end,
            "child_process_count_delta": None if child_start is None or child_end is None else int(child_end) - int(child_start),
            "subprocess_invocations_by_smoke": 0,
            "relationship_discovery_executed": False,
            "broad_pairwise_run_executed": False,
            "test_branch_rerun_performed": False,
            "optuna_or_campaign_run_performed": False,
        },
        "parent_finalizer": {
            "mode": "single_tiny_sandbox_smoke_summary_finalizer",
            "sandbox_summary_artifact_allowed": bool(cfg.write_sandbox_summary_artifact),
            "canonical_write_allowed": False,
            "production_promotion_performed": False,
            "canonical_production_state_outputs_written": False,
        },
        "sandbox_only": True,
        "canonical_root_touched": False,
        "production_approved": False,
        "production_writer_enabled": False,
        "production_outputs_written": False,
        "canonical_production_state_outputs_written": False,
        "production_promotion_performed": False,
        "test_branch_rerun_performed": False,
        "optuna_or_campaign_run_performed": False,
        "relationship_discovery_or_pairwise_run_performed": False,
        "production_writer_gates_fail_closed": True,
    }
    return to_jsonable(_sanitize_workspace_paths(payload))


def build_tiny_sandbox_smoke_branch_payload(
    plan: RegimeProductionNoWritePlan,
    *,
    run_id: str,
    sandbox_output_root: str | Path | None = None,
    env: Mapping[str, str] | None = None,
    project_root: str | Path | None = None,
) -> dict[str, Any]:
    branch = _branch_name(plan.branch)
    schema = default_regime_production_label_output_schema(branch)
    output_root = resolve_regime_production_sandbox_output_root_contract(
        branch,
        sandbox_root=sandbox_output_root,
        env=env,
        project_root=project_root,
    )
    units = _tiny_units(branch, plan.planning_units)
    rows = tuple(
        _smoke_row_for_unit(
            unit,
            branch=branch,
            schema_id=schema.schema_id,
            run_id=run_id,
        )
        for unit in units
    )
    gate = validate_regime_production_planner_gates(branch)
    status_counts: dict[str, int] = {}
    for row in rows:
        status = str(row.label_row.get("availability_status"))
        status_counts[status] = int(status_counts.get(status, 0)) + 1
    selected_rows = [row for row in rows if row.label_row.get("availability_status") == LABEL_PLANNING_STATUS_SELECTED]
    mask_or_unavailable_rows = [row for row in rows if row.label_row.get("mask_reason") not in (None, "")]
    selected_inputs_resolved = all(bool(row.input_resolution.get("input_features_resolved")) for row in selected_rows)
    payload = {
        "schema_version": REGIME_PRODUCTION_TINY_SANDBOX_SMOKE_SCHEMA_VERSION,
        "artifact_kind": "regime_production_tiny_sandbox_branch_smoke",
        "branch": branch,
        "source_artifact_path": _portable_path_text(plan.artifact_path),
        "profile_artifact_consumed": True,
        "output_schema": schema.as_dict(),
        "sandbox_output_root_contract": output_root.as_dict(),
        "scope": _scope_summary(branch, rows),
        "row_count": len(rows),
        "mask_or_unavailable_count": len(mask_or_unavailable_rows),
        "status_counts": status_counts,
        "rows": [row.as_dict() for row in rows],
        "input_features_resolved": bool(selected_rows) and selected_inputs_resolved,
        "selected_label_input_features_resolved": bool(selected_rows) and selected_inputs_resolved,
        "mask_or_unavailable_rows_preserved": bool(mask_or_unavailable_rows) if branch != REGIME_BRANCH_MARKET_STATE else True,
        "definition_refit_plan_resolved": all(bool(row.definition_fit_apply.get("definition_id")) for row in rows),
        "schema_validation_passed": all(tuple(row.label_row.keys()) == schema.column_order for row in rows),
        "mask_assignment_validated": any(row.label_row.get("mask_reason") not in (None, "") for row in rows),
        "label_assignment_validated": any(row.label_row.get("state_id") not in (None, "") for row in rows),
        "relationship_inputs_available_fresh": _relationship_inputs_available_fresh(rows) if branch == REGIME_BRANCH_CROSS_ASSET_STATE else None,
        "relationship_discovery_executed": False,
        "broad_pairwise_run_executed": False,
        "production_gate_validation": gate.as_dict(),
        "canonical_root_touched": False,
        "writer_enabled": False,
        "production_outputs_written": False,
        "canonical_production_state_outputs_written": False,
        "production_writer_gates_fail_closed": True,
    }
    return to_jsonable(_sanitize_workspace_paths(payload))


def write_regime_production_tiny_sandbox_smoke_summary(payload: Mapping[str, Any], output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(dict(payload)), indent=2, sort_keys=True), encoding="utf-8")
    return path


def _build_no_write_plans(cfg: RegimeProductionTinySandboxSmokeConfig) -> dict[str, RegimeProductionNoWritePlan]:
    return {
        REGIME_BRANCH_ASSET_STATE: (
            plan_asset_state_production_no_write(cfg.asset_state_artifact_path, expected_cell_count=3204)
            if cfg.asset_state_artifact_path is not None
            else plan_default_asset_state_production_no_write(expected_cell_count=3204, env=cfg.env)
        ),
        REGIME_BRANCH_MARKET_STATE: (
            plan_market_state_production_no_write(cfg.market_state_artifact_path)
            if cfg.market_state_artifact_path is not None
            else plan_default_market_state_production_no_write(env=cfg.env)
        ),
        REGIME_BRANCH_CROSS_ASSET_STATE: (
            plan_cross_asset_state_production_no_write(cfg.cross_asset_state_artifact_path)
            if cfg.cross_asset_state_artifact_path is not None
            else plan_default_cross_asset_state_production_no_write(env=cfg.env)
        ),
    }


def _tiny_units(branch: str, units: Sequence[Any]) -> tuple[Any, ...]:
    if branch == REGIME_BRANCH_ASSET_STATE:
        return _tiny_asset_units(units)
    if branch == REGIME_BRANCH_MARKET_STATE:
        return _tiny_market_units(units)
    return _tiny_cross_units(units)


def _tiny_asset_units(units: Sequence[Any]) -> tuple[Any, ...]:
    groups: dict[tuple[str, str], list[Any]] = {}
    for unit in units:
        groups.setdefault((str(unit.target_key.get("axis")), str(unit.target_key.get("band"))), []).append(unit)
    for _, group in sorted(groups.items()):
        selected = [unit for unit in group if unit.planning_status == REGIME_PRODUCTION_PLANNING_UNIT_STATUS_SELECTED]
        skipped = [unit for unit in group if unit.planning_status == REGIME_PRODUCTION_PLANNING_UNIT_STATUS_SKIPPED]
        if selected and skipped:
            return tuple((selected[:1] + skipped[:1])[:2])
    for _, group in sorted(groups.items()):
        selected = [unit for unit in group if unit.planning_status == REGIME_PRODUCTION_PLANNING_UNIT_STATUS_SELECTED]
        if selected:
            return tuple(selected[:2])
    return tuple(units[:1])


def _tiny_market_units(units: Sequence[Any]) -> tuple[Any, ...]:
    groups: dict[str, list[Any]] = {}
    for unit in units:
        groups.setdefault(str(unit.target_key.get("band")), []).append(unit)
    for _, group in sorted(groups.items()):
        selected = [unit for unit in group if unit.planning_status == REGIME_PRODUCTION_PLANNING_UNIT_STATUS_SELECTED]
        if selected:
            return tuple(selected[:2])
    return tuple(units[:1])


def _tiny_cross_units(units: Sequence[Any]) -> tuple[Any, ...]:
    groups: dict[tuple[str, str], list[Any]] = {}
    for unit in units:
        groups.setdefault((str(unit.target_key.get("relationship_feature_family")), str(unit.target_key.get("band"))), []).append(unit)
    for _, group in sorted(groups.items()):
        selected = [unit for unit in group if unit.planning_status == REGIME_PRODUCTION_PLANNING_UNIT_STATUS_SELECTED]
        masked = [unit for unit in group if unit.planning_status == REGIME_PRODUCTION_PLANNING_UNIT_STATUS_MASKED_UNAVAILABLE]
        if selected and masked:
            return tuple((selected[:1] + masked[:1])[:2])
    for _, group in sorted(groups.items()):
        selected = [unit for unit in group if unit.planning_status == REGIME_PRODUCTION_PLANNING_UNIT_STATUS_SELECTED]
        if selected:
            return tuple(selected[:2])
    return tuple(units[:1])


def _smoke_row_for_unit(unit: Any, *, branch: str, schema_id: str, run_id: str) -> RegimeProductionTinySandboxSmokeRow:
    model_state = dict(unit.model_state_definition or {})
    availability = _availability(unit.planning_status)
    mask_reason = _mask_reason(unit, availability)
    state_id = _sandbox_state_id(unit) if availability == LABEL_PLANNING_STATUS_SELECTED else None
    state_label = state_id if state_id is not None else None
    confidence = _confidence(unit) if state_id is not None else None
    definition_id = _definition_id(branch, unit, model_state)
    timestamp_range = _timestamp_range_chunk(unit.timestamp_plan)
    input_resolution = _input_resolution(unit, branch=branch)
    lineage = {
        "run_id": run_id,
        "sandbox_only": True,
        "profile_artifact_path": unit.profile_artifact_path,
        "profile_artifact_hash": unit.profile_artifact_hash,
        "source_planning_unit_id": unit.unit_id,
        "source_tail_ts": model_state.get("source_tail_ts"),
        "known_at_ts": model_state.get("definition_known_at_ts"),
        "definition_fit_apply_smoke": True,
        "canonical_output": False,
    }
    row = _label_row(
        branch,
        unit=unit,
        state_id=state_id,
        state_label=state_label,
        confidence=confidence,
        availability=availability,
        mask_reason=mask_reason,
        definition_id=definition_id,
        definition_version="tiny_sandbox_definition_smoke_v1",
        source_tail_ts=model_state.get("source_tail_ts"),
        known_at_ts=model_state.get("definition_known_at_ts"),
        relationship_input_tail_ts=model_state.get("source_tail_ts") if branch == REGIME_BRANCH_CROSS_ASSET_STATE else None,
        relationship_known_at_ts=model_state.get("definition_known_at_ts") if branch == REGIME_BRANCH_CROSS_ASSET_STATE else None,
        lineage=lineage,
        run_id=run_id,
        timestamp_range=timestamp_range,
    )
    return RegimeProductionTinySandboxSmokeRow(
        branch=branch,
        label_row=row,
        output_schema_id=schema_id,
        source_unit_id=unit.unit_id,
        source_unit_status=unit.planning_status,
        definition_fit_apply={
            "definition_id": definition_id,
            "definition_version": "tiny_sandbox_definition_smoke_v1",
            "refit_cadence_id": model_state.get("refit_cadence_id"),
            "fit_status": "sandbox_definition_fit_smoke_succeeded"
            if state_id is not None
            else "sandbox_mask_apply_succeeded",
            "apply_status": "sandbox_label_like_assignment_succeeded"
            if state_id is not None
            else "sandbox_mask_assignment_succeeded",
            "definition_file_written": False,
            "production_definition_written": False,
        },
        input_resolution=input_resolution,
        timestamp_range_chunk=timestamp_range,
    )


def _label_row(
    branch: str,
    *,
    unit: Any,
    state_id: str | None,
    state_label: str | None,
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
    timestamp_range: Mapping[str, Any],
) -> dict[str, Any]:
    timestamp = f"range:{timestamp_range['chunk_id']}"
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


def _availability(status: str) -> str:
    return {
        REGIME_PRODUCTION_PLANNING_UNIT_STATUS_SELECTED: LABEL_PLANNING_STATUS_SELECTED,
        REGIME_PRODUCTION_PLANNING_UNIT_STATUS_MASKED_UNAVAILABLE: LABEL_PLANNING_STATUS_MASKED_UNAVAILABLE,
        REGIME_PRODUCTION_PLANNING_UNIT_STATUS_SKIPPED: LABEL_PLANNING_STATUS_SKIPPED_FILTERED,
        REGIME_PRODUCTION_PLANNING_UNIT_STATUS_DIAGNOSTIC_ONLY: LABEL_PLANNING_STATUS_DIAGNOSTIC_ONLY,
    }.get(str(status), LABEL_PLANNING_STATUS_MASKED_UNAVAILABLE)


def _mask_reason(unit: Any, availability: str) -> str | None:
    if availability == LABEL_PLANNING_STATUS_SELECTED:
        return None
    for reason in unit.reason_codes:
        text = str(reason or "").strip()
        if text and text != "model_state_missing_required_fields":
            return text
    return availability


def _confidence(unit: Any) -> float:
    for source in (unit.health_metadata, unit.method_metadata):
        for field_name in ("output_health_score", "candidate_score", "confidence", "stability_score"):
            value = dict(source or {}).get(field_name)
            try:
                if value is not None:
                    return max(0.0, min(1.0, float(value)))
            except Exception:
                continue
    return 1.0


def _sandbox_state_id(unit: Any) -> str:
    raw = json.dumps(to_jsonable(dict(unit.target_key)), sort_keys=True, separators=(",", ":")).encode("utf-8")
    state = int(hashlib.sha256(raw).hexdigest()[:2], 16) % 3
    return f"sandbox_state_{state}"


def _definition_id(branch: str, unit: Any, model_state: Mapping[str, Any]) -> str:
    cadence = str(model_state.get("refit_cadence_id") or "missing_refit_cadence_id")
    raw = json.dumps(to_jsonable(dict(unit.target_key)), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return f"{branch}:{hashlib.sha256(raw).hexdigest()[:20]}:{cadence}:tiny_smoke"


def _timestamp_range_chunk(timestamp_plan: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "chunk_id": "tiny_configured_clamp_slice",
        "timestamp_rows_materialized": False,
        "historical_output_months": int(timestamp_plan.get("historical_output_months") or 0),
        "required_lookback_months": int(timestamp_plan.get("required_lookback_months") or 0),
        "clamp_policy_id": timestamp_plan.get("clamp_policy_id"),
        "small_time_slice": True,
    }


def _input_resolution(unit: Any, *, branch: str) -> dict[str, Any]:
    metadata = dict(unit.method_metadata or {})
    present_fields = sorted(key for key, value in metadata.items() if value not in (None, "", {}, ()))
    raw_relationship_checks = [to_jsonable(dict(item)) for item in unit.relationship_input_checks]
    relationship_checks = [
        check
        for check in raw_relationship_checks
        if check.get("artifact_kind") != "regime_production_relationship_input_freshness_check"
    ]
    relationship_freshness_checks = [
        check
        for check in raw_relationship_checks
        if check.get("artifact_kind") == "regime_production_relationship_input_freshness_check"
    ]
    relationship_available = all(
        check.get("status") in {"available", "metadata_present"} and check.get("reason_code") in (None, "")
        for check in relationship_checks
    ) if relationship_checks else None
    freshness_available = all(
        check.get("status") == "available" and check.get("reason_code") in (None, "")
        for check in relationship_freshness_checks
    ) if relationship_freshness_checks else None
    if relationship_available is not False and freshness_available is not None:
        relationship_available = bool(freshness_available)
    return {
        "input_features_resolved": bool(present_fields) and (relationship_available is not False),
        "method_metadata_fields_present": present_fields,
        "feature_resolution_mode": "active_profile_metadata_tiny_sandbox_smoke",
        "feature_frame_materialized": False,
        "relationship_input_checks": relationship_checks,
        "relationship_input_freshness_checks": relationship_freshness_checks,
        "relationship_inputs_available_fresh": relationship_available,
        "relationship_discovery_executed": False,
        "broad_pairwise_run_executed": False,
        "selected_profile_artifact_consumed": True,
        "relationship_input_history_separate_from_selected_profile_artifact": branch == REGIME_BRANCH_CROSS_ASSET_STATE,
    }


def _scope_summary(branch: str, rows: Sequence[RegimeProductionTinySandboxSmokeRow]) -> dict[str, Any]:
    label_rows = [row.label_row for row in rows]
    scope = {"row_count": len(label_rows)}
    if branch in {REGIME_BRANCH_ASSET_STATE, REGIME_BRANCH_CROSS_ASSET_STATE}:
        scope["asset_count"] = len({row.get("asset_id") for row in label_rows})
    if branch == REGIME_BRANCH_ASSET_STATE:
        scope["axis_count"] = len({row.get("axis") for row in label_rows})
    if branch == REGIME_BRANCH_MARKET_STATE:
        scope["market_axis_count"] = len({row.get("market_axis") for row in label_rows})
    if branch == REGIME_BRANCH_CROSS_ASSET_STATE:
        scope["relationship_feature_family_count"] = len({row.get("relationship_feature_family") for row in label_rows})
    scope["band_count"] = len({row.get("band") for row in label_rows})
    scope["small_time_slice"] = True
    return scope


def _relationship_inputs_available_fresh(rows: Sequence[RegimeProductionTinySandboxSmokeRow]) -> bool:
    checks = []
    for row in rows:
        checks.extend(row.input_resolution.get("relationship_input_checks") or ())
    return bool(checks) and all(
        dict(check).get("status") in {"available", "metadata_present"} and dict(check).get("reason_code") in (None, "")
        for check in checks
    )


def _rss_bytes() -> int | None:
    try:
        import psutil  # type: ignore

        return int(psutil.Process(os.getpid()).memory_info().rss)
    except Exception:
        return None


def _child_process_count() -> int | None:
    try:
        import psutil  # type: ignore

        return len(psutil.Process(os.getpid()).children(recursive=True))
    except Exception:
        return None


def _sanitize_workspace_paths(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _sanitize_workspace_paths(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_sanitize_workspace_paths(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_sanitize_workspace_paths(item) for item in value)
    if isinstance(value, str):
        return _portable_path_text(value) if _looks_like_absolute_path(value) else value
    return value


def _portable_path_text(value: str | Path | None) -> str | None:
    if value is None:
        return None
    path = Path(value)
    if not path.is_absolute():
        return str(path)
    resolved = path.resolve()
    cwd = Path.cwd().resolve()
    try:
        return str(resolved.relative_to(cwd))
    except ValueError:
        return f"<external_configured_root>/{resolved.name}"


def _looks_like_absolute_path(value: str) -> bool:
    try:
        return Path(value).is_absolute()
    except Exception:
        return False


def _branch_name(value: object) -> str:
    text = _text(value, field_name="branch")
    if text not in REGIME_PRODUCTION_BRANCHES:
        raise ValueError(f"Unsupported Regime Production branch: {text!r}")
    return text


def _text(value: object, *, field_name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"Regime Production tiny sandbox smoke {field_name} must be non-empty")
    return text


__all__ = [
    "REGIME_PRODUCTION_TINY_SANDBOX_ROW_ARTIFACT_KIND",
    "REGIME_PRODUCTION_TINY_SANDBOX_SMOKE_ARTIFACT_KIND",
    "REGIME_PRODUCTION_TINY_SANDBOX_SMOKE_SCHEMA_VERSION",
    "RegimeProductionTinySandboxSmokeConfig",
    "RegimeProductionTinySandboxSmokeRow",
    "build_tiny_sandbox_smoke_branch_payload",
    "run_regime_production_tiny_sandbox_smoke",
    "write_regime_production_tiny_sandbox_smoke_summary",
]
