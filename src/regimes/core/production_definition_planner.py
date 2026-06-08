from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.forecasting.common.path_config import PATH_KEYS, resolve_path
from src.regimes.asset_state.production_planner import (
    plan_asset_state_production_no_write,
    plan_default_asset_state_production_no_write,
)
from src.regimes.core.path_safety import validate_non_production_write_root, validate_report_root
from src.regimes.core.paths import is_relative_to, resolve_project_path
from src.regimes.core.production_consumer import (
    REGIME_BRANCH_ASSET_STATE,
    REGIME_BRANCH_CROSS_ASSET_STATE,
    REGIME_BRANCH_MARKET_STATE,
    REGIME_PRODUCTION_BRANCHES,
)
from src.regimes.core.production_clamp_contract import checkpoint_count_for_cadence
from src.regimes.core.production_output_contracts import CANONICAL_ROOT_KEYS
from src.regimes.core.production_planner import (
    REGIME_PRODUCTION_MODEL_STATE_STATUS_MISSING_INPUT,
    REGIME_PRODUCTION_MODEL_STATE_STATUS_PLANNED,
    REGIME_PRODUCTION_NO_WRITE_PLAN_ARTIFACT_KIND,
    RegimeProductionNoWritePlan,
    validate_regime_production_planner_gates,
)
from src.regimes.core.production_reuse_cache import RegimeProductionPlannerRunCache
from src.regimes.core.root_resolution import resolve_regime_production_write_root
from src.regimes.core.serialization import to_jsonable
from src.regimes.cross_asset_state.production_planner import (
    plan_cross_asset_state_production_no_write,
    plan_default_cross_asset_state_production_no_write,
)
from src.regimes.market_state.production_planner import (
    plan_default_market_state_production_no_write,
    plan_market_state_production_no_write,
)


REGIME_PRODUCTION_DEFINITION_PLANNER_SCHEMA_VERSION = 1
REGIME_PRODUCTION_DEFINITION_ROOT_ARTIFACT_KIND = "regime_production_definition_sandbox_root_contract"
REGIME_PRODUCTION_DEFINITION_RECORD_ARTIFACT_KIND = "regime_production_definition_schedule_record"
REGIME_PRODUCTION_DEFINITION_PLAN_ARTIFACT_KIND = "regime_production_definition_schedule_plan"
REGIME_PRODUCTION_DEFINITION_SUMMARY_ARTIFACT_KIND = "regime_production_definition_planner_summary"

DEFINITION_SANDBOX_ROOT_NAMESPACE = "regime_definition_sandbox_contracts"


@dataclass(frozen=True)
class RegimeProductionDefinitionPlannerConfig:
    asset_state_artifact_path: str | Path | None = None
    market_state_artifact_path: str | Path | None = None
    cross_asset_state_artifact_path: str | Path | None = None
    sandbox_definition_root: str | Path | None = None
    env: Mapping[str, str] | None = None
    run_id: str = "regime_production_definition_planner"
    include_definition_records: bool = False
    max_sample_records_per_branch: int = 5
    run_cache: RegimeProductionPlannerRunCache | None = None

    def __post_init__(self) -> None:
        if int(self.max_sample_records_per_branch) < 0:
            raise ValueError("Regime Production definition planner sample size cannot be negative")
        object.__setattr__(self, "env", dict(os.environ if self.env is None else self.env))
        object.__setattr__(self, "run_id", _text(self.run_id, field_name="run_id"))
        object.__setattr__(self, "max_sample_records_per_branch", int(self.max_sample_records_per_branch))


@dataclass(frozen=True)
class RegimeProductionDefinitionRootContract:
    branch: str
    sandbox_definition_root: str | Path
    root_source: str
    canonical_root_checks: Sequence[Mapping[str, Any]] = field(default_factory=tuple)
    writer_enabled: bool = False
    production_outputs_written: bool = False
    canonical_production_state_outputs_written: bool = False

    def __post_init__(self) -> None:
        branch = _branch_name(self.branch)
        root = validate_report_root(self.sandbox_definition_root, allow_foundation_descendant=True)
        root = validate_non_production_write_root(root)
        checks = tuple(to_jsonable(dict(item)) for item in self.canonical_root_checks)
        if any(bool(item.get("collision")) for item in checks):
            raise ValueError("Regime Production definition sandbox root collides with a canonical root")
        if self.writer_enabled or self.production_outputs_written or self.canonical_production_state_outputs_written:
            raise ValueError("Regime Production definition root contract cannot enable production writes")
        object.__setattr__(self, "branch", branch)
        object.__setattr__(self, "sandbox_definition_root", root)
        object.__setattr__(self, "root_source", _text(self.root_source, field_name="root_source"))
        object.__setattr__(self, "canonical_root_checks", checks)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": REGIME_PRODUCTION_DEFINITION_PLANNER_SCHEMA_VERSION,
            "artifact_kind": REGIME_PRODUCTION_DEFINITION_ROOT_ARTIFACT_KIND,
            "branch": self.branch,
            "sandbox_definition_root": _portable_path_text(self.sandbox_definition_root),
            "root_source": self.root_source,
            "root_kind": "noncanonical_sandbox_definition_root",
            "noncanonical": True,
            "separate_from_canonical_model_state_roots": True,
            "canonical_root_checks": [to_jsonable(dict(item)) for item in self.canonical_root_checks],
            "canonical_root_collision": False,
            "canonical_root_touched": False,
            "safe_to_delete_later_with_explicit_request": True,
            "writer_enabled": False,
            "production_outputs_written": False,
            "production_labels_written": False,
            "canonical_production_state_outputs_written": False,
        }


@dataclass(frozen=True)
class RegimeProductionDefinitionRecord:
    definition_id: str
    definition_version: str
    branch: str
    target_key: Mapping[str, Any]
    profile_id: str | None
    profile_version: str | None
    profile_artifact_path: str | None
    profile_artifact_hash: str
    refit_window_start: Any
    refit_window_end: Any
    refit_cadence_id: str
    cadence_checkpoint_index: int
    cadence_checkpoint_count: int
    definition_known_at_ts: Any
    source_tail_ts: Any
    status: str
    health_metadata: Mapping[str, Any]
    lineage: Mapping[str, Any]
    planned_file_path: str | Path
    planned_model_state_path: str | Path
    relationship_input_tail_ts: Any = None
    relationship_known_at_ts: Any = None
    relationship_input_lineage: Mapping[str, Any] = field(default_factory=dict)
    local_reselection_eligible: bool = False
    local_reselection_execution_performed: bool = False
    bar_level_mask_triggers_refit: bool = False

    def __post_init__(self) -> None:
        branch = _branch_name(self.branch)
        if int(self.cadence_checkpoint_index) < 0:
            raise ValueError("Regime Production definition checkpoint index cannot be negative")
        if int(self.cadence_checkpoint_count) <= 0:
            raise ValueError("Regime Production definition checkpoint count must be positive")
        if int(self.cadence_checkpoint_index) >= int(self.cadence_checkpoint_count):
            raise ValueError("Regime Production definition checkpoint index must be within checkpoint count")
        if self.local_reselection_execution_performed or self.bar_level_mask_triggers_refit:
            raise ValueError("Regime Production definition planner cannot execute reselection or per-bar refit")
        _validate_optional_order(self.refit_window_start, self.refit_window_end, context="refit window")
        _validate_optional_order(self.source_tail_ts, self.definition_known_at_ts, context="source tail and known-at")
        _validate_optional_order(self.relationship_input_tail_ts, self.relationship_known_at_ts, context="relationship source tail and known-at")
        object.__setattr__(self, "branch", branch)
        object.__setattr__(self, "definition_id", _text(self.definition_id, field_name="definition_id"))
        object.__setattr__(self, "definition_version", _text(self.definition_version, field_name="definition_version"))
        object.__setattr__(self, "target_key", to_jsonable(dict(self.target_key)))
        object.__setattr__(self, "profile_artifact_hash", _text(self.profile_artifact_hash, field_name="profile_artifact_hash"))
        object.__setattr__(self, "refit_cadence_id", _text(self.refit_cadence_id, field_name="refit_cadence_id"))
        object.__setattr__(self, "health_metadata", to_jsonable(dict(self.health_metadata)))
        object.__setattr__(self, "lineage", to_jsonable(dict(self.lineage)))
        object.__setattr__(self, "relationship_input_lineage", to_jsonable(dict(self.relationship_input_lineage)))

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": REGIME_PRODUCTION_DEFINITION_PLANNER_SCHEMA_VERSION,
            "artifact_kind": REGIME_PRODUCTION_DEFINITION_RECORD_ARTIFACT_KIND,
            "definition_id": self.definition_id,
            "definition_version": self.definition_version,
            "branch": self.branch,
            "target_key": to_jsonable(dict(self.target_key)),
            "profile_id": self.profile_id,
            "profile_version": self.profile_version,
            "profile_artifact_path": _portable_path_text(self.profile_artifact_path) if self.profile_artifact_path else None,
            "profile_artifact_hash": self.profile_artifact_hash,
            "refit_window_start": self.refit_window_start,
            "refit_window_end": self.refit_window_end,
            "refit_cadence_id": self.refit_cadence_id,
            "cadence_checkpoint_index": int(self.cadence_checkpoint_index),
            "cadence_checkpoint_count": int(self.cadence_checkpoint_count),
            "definition_known_at_ts": self.definition_known_at_ts,
            "source_tail_ts": self.source_tail_ts,
            "status": self.status,
            "health_metadata": to_jsonable(dict(self.health_metadata)),
            "lineage": to_jsonable(dict(self.lineage)),
            "planned_file_path": _portable_path_text(self.planned_file_path),
            "planned_model_state_path": _portable_path_text(self.planned_model_state_path),
            "relationship_input_tail_ts": self.relationship_input_tail_ts,
            "relationship_known_at_ts": self.relationship_known_at_ts,
            "relationship_input_lineage": to_jsonable(dict(self.relationship_input_lineage)),
            "relationship_input_history_separate_from_selected_profile_artifact": self.branch == REGIME_BRANCH_CROSS_ASSET_STATE,
            "model_state_persistence": "one_record_per_branch_target_key_cadence_checkpoint",
            "local_reselection_eligible": bool(self.local_reselection_eligible),
            "local_reselection_execution_performed": False,
            "bar_level_mask_triggers_refit": False,
            "definition_file_written": False,
            "model_state_record_written": False,
            "production_labels_written": False,
            "production_outputs_written": False,
            "canonical_production_state_outputs_written": False,
        }


@dataclass(frozen=True)
class RegimeProductionDefinitionSchedulePlan:
    branch: str
    source_no_write_plan: Mapping[str, Any]
    definition_root_contract: RegimeProductionDefinitionRootContract
    target_count: int
    definition_record_count: int
    cadence_checkpoint_counts: Mapping[str, int]
    target_count_by_cadence: Mapping[str, int]
    definition_count_by_cadence: Mapping[str, int]
    target_count_by_band: Mapping[str, int]
    definition_count_by_band: Mapping[str, int]
    model_state_status_counts: Mapping[str, int]
    relationship_tail_available_count: int = 0
    relationship_tail_missing_count: int = 0
    definition_records: Sequence[RegimeProductionDefinitionRecord] = ()
    definition_record_samples: Sequence[RegimeProductionDefinitionRecord] = ()
    warnings: Sequence[str] = ()
    production_gate_validation: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        branch = _branch_name(self.branch)
        source = to_jsonable(dict(self.source_no_write_plan))
        if source.get("artifact_kind") != REGIME_PRODUCTION_NO_WRITE_PLAN_ARTIFACT_KIND:
            raise ValueError("Regime Production definition schedule requires a no-write plan source")
        gate = to_jsonable(dict(self.production_gate_validation))
        if gate.get("production_write_allowed") or gate.get("writer_enabled") or gate.get("production_labels_allowed"):
            raise ValueError("Regime Production definition schedule received an open production gate")
        object.__setattr__(self, "branch", branch)
        object.__setattr__(self, "source_no_write_plan", source)
        object.__setattr__(self, "cadence_checkpoint_counts", to_jsonable(dict(self.cadence_checkpoint_counts)))
        object.__setattr__(self, "target_count_by_cadence", to_jsonable(dict(self.target_count_by_cadence)))
        object.__setattr__(self, "definition_count_by_cadence", to_jsonable(dict(self.definition_count_by_cadence)))
        object.__setattr__(self, "target_count_by_band", to_jsonable(dict(self.target_count_by_band)))
        object.__setattr__(self, "definition_count_by_band", to_jsonable(dict(self.definition_count_by_band)))
        object.__setattr__(self, "model_state_status_counts", to_jsonable(dict(self.model_state_status_counts)))
        object.__setattr__(self, "definition_records", tuple(self.definition_records))
        object.__setattr__(self, "definition_record_samples", tuple(self.definition_record_samples))
        object.__setattr__(self, "warnings", tuple(dict.fromkeys(str(warning) for warning in self.warnings)))
        object.__setattr__(self, "production_gate_validation", gate)

    def as_dict(self, *, include_records: bool = False) -> dict[str, Any]:
        records = self.definition_records if include_records else ()
        return {
            "schema_version": REGIME_PRODUCTION_DEFINITION_PLANNER_SCHEMA_VERSION,
            "artifact_kind": REGIME_PRODUCTION_DEFINITION_PLAN_ARTIFACT_KIND,
            "branch": self.branch,
            "source_no_write_plan": _sanitize_workspace_paths(to_jsonable(dict(self.source_no_write_plan))),
            "definition_root_contract": self.definition_root_contract.as_dict(),
            "target_count": int(self.target_count),
            "definition_record_count": int(self.definition_record_count),
            "cadence_checkpoint_counts": dict(self.cadence_checkpoint_counts),
            "target_count_by_cadence": dict(self.target_count_by_cadence),
            "definition_count_by_cadence": dict(self.definition_count_by_cadence),
            "target_count_by_band": dict(self.target_count_by_band),
            "definition_count_by_band": dict(self.definition_count_by_band),
            "model_state_status_counts": dict(self.model_state_status_counts),
            "relationship_tail_available_count": int(self.relationship_tail_available_count),
            "relationship_tail_missing_count": int(self.relationship_tail_missing_count),
            "definition_record_samples": [record.as_dict() for record in self.definition_record_samples],
            "definition_records": [record.as_dict() for record in records],
            "definition_records_omitted": not include_records,
            "model_state_persistence": "one_record_per_branch_target_key_cadence_checkpoint",
            "bar_level_mask_triggers_refit": False,
            "per_bar_refit_enabled": False,
            "local_reselection_execution_performed": False,
            "definition_files_written": False,
            "model_state_records_written": False,
            "production_labels_written": False,
            "production_outputs_written": False,
            "canonical_production_state_outputs_written": False,
            "production_gate_validation": to_jsonable(dict(self.production_gate_validation)),
            "production_writer_gates_fail_closed": True,
            "warnings": list(self.warnings),
        }


def run_regime_production_definition_schedule_planner(
    config: RegimeProductionDefinitionPlannerConfig | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    cfg = config if isinstance(config, RegimeProductionDefinitionPlannerConfig) else RegimeProductionDefinitionPlannerConfig(**dict(config or {}))
    run_cache = cfg.run_cache or RegimeProductionPlannerRunCache(cache_id=f"{cfg.run_id}_cache")
    plans = _build_no_write_plans(cfg, run_cache=run_cache)
    schedules = {
        branch: plan_regime_production_definition_schedule_from_no_write_plan(
            plan,
            sandbox_definition_root=cfg.sandbox_definition_root,
            env=cfg.env,
            include_definition_records=bool(cfg.include_definition_records),
            max_sample_records=cfg.max_sample_records_per_branch,
        )
        for branch, plan in plans.items()
    }
    payload = {
        "schema_version": REGIME_PRODUCTION_DEFINITION_PLANNER_SCHEMA_VERSION,
        "artifact_kind": REGIME_PRODUCTION_DEFINITION_SUMMARY_ARTIFACT_KIND,
        "run_id": cfg.run_id,
        "branch_schedules": {branch: schedule.as_dict(include_records=cfg.include_definition_records) for branch, schedule in schedules.items()},
        "definition_record_count_by_branch": {branch: int(schedule.definition_record_count) for branch, schedule in schedules.items()},
        "target_count_by_branch": {branch: int(schedule.target_count) for branch, schedule in schedules.items()},
        "total_definition_record_count": sum(int(schedule.definition_record_count) for schedule in schedules.values()),
        "total_target_count": sum(int(schedule.target_count) for schedule in schedules.values()),
        "cadence_assumptions": {
            branch: dict(schedule.cadence_checkpoint_counts)
            for branch, schedule in schedules.items()
        },
        "reuse_cache_telemetry": run_cache.as_dict(),
        "bar_level_mask_triggers_refit": False,
        "per_bar_refit_enabled": False,
        "local_reselection_execution_performed": False,
        "definition_files_written": False,
        "labels_generated": False,
        "production_approved": False,
        "production_writer_enabled": False,
        "production_labels_written": False,
        "production_outputs_written": False,
        "canonical_production_state_outputs_written": False,
        "production_promotion_performed": False,
        "relationship_discovery_or_pairwise_run_performed": False,
        "test_branch_rerun_performed": False,
        "production_writer_gates_fail_closed": True,
        "warnings": tuple(dict.fromkeys(warning for schedule in schedules.values() for warning in schedule.warnings)),
    }
    return to_jsonable(_sanitize_workspace_paths(payload))


def plan_regime_production_definition_schedule_from_no_write_plan(
    plan: RegimeProductionNoWritePlan,
    *,
    sandbox_definition_root: str | Path | None = None,
    env: Mapping[str, str] | None = None,
    project_root: str | Path | None = None,
    include_definition_records: bool = False,
    max_sample_records: int = 5,
) -> RegimeProductionDefinitionSchedulePlan:
    branch = _branch_name(plan.branch)
    root_contract = resolve_regime_production_definition_root_contract(
        branch,
        sandbox_definition_root=sandbox_definition_root,
        env=env,
        project_root=project_root,
    )
    plan_payload = plan.as_dict(include_units=False)
    default_historical_months = _historical_months_from_plan_payload(plan_payload)
    default_refit_cadence_id = _plan_refit_cadence_id(plan_payload)
    units = tuple(plan.planning_units)
    target_count = len(units)
    cadence_checkpoint_counts: dict[str, int] = {}
    target_count_by_cadence: dict[str, int] = {}
    definition_count_by_cadence: dict[str, int] = {}
    target_count_by_band: dict[str, int] = {}
    definition_count_by_band: dict[str, int] = {}
    model_state_status_counts: dict[str, int] = {}
    relationship_tail_available = 0
    relationship_tail_missing = 0
    definition_record_count = 0
    records: list[RegimeProductionDefinitionRecord] = []
    samples: list[RegimeProductionDefinitionRecord] = []
    warnings: list[str] = []
    sample_limit = max(0, int(max_sample_records))

    for unit in units:
        model_state = dict(unit.model_state_definition or {})
        cadence_id = str(model_state.get("refit_cadence_id") or default_refit_cadence_id)
        cadence = _cadence_from_id(cadence_id)
        checkpoints = cadence_checkpoint_counts.setdefault(
            cadence_id,
            _checkpoint_count(cadence, _historical_months_for_unit(unit, default_historical_months)),
        )
        target_count_by_cadence[cadence_id] = int(target_count_by_cadence.get(cadence_id, 0)) + 1
        definition_count_by_cadence[cadence_id] = int(definition_count_by_cadence.get(cadence_id, 0)) + checkpoints
        band = str(unit.target_key.get("band") or "unknown")
        target_count_by_band[band] = int(target_count_by_band.get(band, 0)) + 1
        definition_count_by_band[band] = int(definition_count_by_band.get(band, 0)) + checkpoints
        status = _definition_status(unit)
        model_state_status_counts[status] = int(model_state_status_counts.get(status, 0)) + 1
        relationship_tail = _relationship_tail_payload(unit, model_state)
        if branch == REGIME_BRANCH_CROSS_ASSET_STATE:
            if relationship_tail["relationship_input_tail_ts"] is None or relationship_tail["relationship_known_at_ts"] is None:
                relationship_tail_missing += 1
            else:
                relationship_tail_available += 1
        definition_record_count += checkpoints
        missing_fields = tuple(str(item) for item in model_state.get("missing_required_fields") or ())
        if missing_fields:
            warnings.append(f"{branch}_definition_model_state_missing_required_fields")
        should_materialize_records = include_definition_records or len(samples) < sample_limit
        if not should_materialize_records:
            continue
        for checkpoint_index in range(checkpoints):
            record = _definition_record_for_unit(
                unit,
                branch=branch,
                model_state=model_state,
                status=status,
                checkpoint_index=checkpoint_index,
                checkpoint_count=checkpoints,
                root=Path(root_contract.sandbox_definition_root),
                relationship_tail=relationship_tail,
            )
            if include_definition_records:
                records.append(record)
            if len(samples) < sample_limit:
                samples.append(record)
            if not include_definition_records and len(samples) >= sample_limit:
                break

    gate = validate_regime_production_planner_gates(branch)
    return RegimeProductionDefinitionSchedulePlan(
        branch=branch,
        source_no_write_plan=plan_payload,
        definition_root_contract=root_contract,
        target_count=target_count,
        definition_record_count=definition_record_count,
        cadence_checkpoint_counts=cadence_checkpoint_counts,
        target_count_by_cadence=target_count_by_cadence,
        definition_count_by_cadence=definition_count_by_cadence,
        target_count_by_band=target_count_by_band,
        definition_count_by_band=definition_count_by_band,
        model_state_status_counts=model_state_status_counts,
        relationship_tail_available_count=relationship_tail_available,
        relationship_tail_missing_count=relationship_tail_missing,
        definition_records=tuple(records),
        definition_record_samples=tuple(samples),
        warnings=tuple(dict.fromkeys(warnings)),
        production_gate_validation=gate.as_dict(),
    )


def resolve_regime_production_definition_root_contract(
    branch: str,
    *,
    sandbox_definition_root: str | Path | None = None,
    env: Mapping[str, str] | None = None,
    project_root: str | Path | None = None,
) -> RegimeProductionDefinitionRootContract:
    branch_name = _branch_name(branch)
    source_env = env if env is not None else os.environ
    if sandbox_definition_root is None:
        root, source = resolve_regime_production_write_root(
            None,
            env=source_env,
            project_root=project_root,
            subdir=f"{DEFINITION_SANDBOX_ROOT_NAMESPACE}/{branch_name}",
        )
    else:
        root, source = resolve_regime_production_write_root(
            sandbox_definition_root,
            env=source_env,
            project_root=project_root,
            allow_explicit_dry_test_override=True,
        )
    root = validate_non_production_write_root(root, project_root=project_root)
    checks = _canonical_root_checks(root, env=source_env, project_root=project_root)
    return RegimeProductionDefinitionRootContract(
        branch=branch_name,
        sandbox_definition_root=root,
        root_source=source,
        canonical_root_checks=checks,
    )


def write_regime_production_definition_planner_summary(
    payload: Mapping[str, Any],
    output_path: str | Path,
) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(dict(payload)), indent=2, sort_keys=True), encoding="utf-8")
    return path


def _build_no_write_plans(
    cfg: RegimeProductionDefinitionPlannerConfig,
    *,
    run_cache: RegimeProductionPlannerRunCache,
) -> dict[str, RegimeProductionNoWritePlan]:
    return {
        REGIME_BRANCH_ASSET_STATE: (
            plan_asset_state_production_no_write(cfg.asset_state_artifact_path, expected_cell_count=3204, run_cache=run_cache)
            if cfg.asset_state_artifact_path is not None
            else plan_default_asset_state_production_no_write(expected_cell_count=3204, env=cfg.env, run_cache=run_cache)
        ),
        REGIME_BRANCH_MARKET_STATE: (
            plan_market_state_production_no_write(cfg.market_state_artifact_path, run_cache=run_cache)
            if cfg.market_state_artifact_path is not None
            else plan_default_market_state_production_no_write(env=cfg.env, run_cache=run_cache)
        ),
        REGIME_BRANCH_CROSS_ASSET_STATE: (
            plan_cross_asset_state_production_no_write(cfg.cross_asset_state_artifact_path, run_cache=run_cache)
            if cfg.cross_asset_state_artifact_path is not None
            else plan_default_cross_asset_state_production_no_write(env=cfg.env, run_cache=run_cache)
        ),
    }


def _definition_record_for_unit(
    unit,
    *,
    branch: str,
    model_state: Mapping[str, Any],
    status: str,
    checkpoint_index: int,
    checkpoint_count: int,
    root: Path,
    relationship_tail: Mapping[str, Any],
) -> RegimeProductionDefinitionRecord:
    cadence_id = str(model_state.get("refit_cadence_id") or "missing_refit_cadence_id")
    target_hash = _target_hash(unit.target_key)
    definition_id = f"{branch}:{target_hash}:{cadence_id}:{int(checkpoint_index):04d}"
    planned_path = (
        root
        / f"cadence={_safe_path_part(cadence_id)}"
        / f"band={_safe_path_part(unit.target_key.get('band') or 'unknown')}"
        / f"target={target_hash}"
        / f"checkpoint={int(checkpoint_index):04d}"
        / "definition_plan.json"
    )
    planned_model_state_path = (
        root
        / "model_state"
        / f"cadence={_safe_path_part(cadence_id)}"
        / f"band={_safe_path_part(unit.target_key.get('band') or 'unknown')}"
        / f"target={target_hash}"
        / f"checkpoint={int(checkpoint_index):04d}"
        / "model_state.json"
    )
    missing_required_fields = tuple(str(item) for item in model_state.get("missing_required_fields") or ())
    lineage = dict(model_state.get("lineage") or {})
    lineage.update(
        {
            "source_planning_unit_id": unit.unit_id,
            "definition_schedule_planner": "regime_production_definition_schedule_planner_v1",
            "bar_level_mask_triggers_refit": False,
        }
    )
    return RegimeProductionDefinitionRecord(
        definition_id=definition_id,
        definition_version=f"definition_schedule_v1_checkpoint_{int(checkpoint_index):04d}",
        branch=branch,
        target_key=unit.target_key,
        profile_id=unit.profile_id,
        profile_version=unit.profile_version,
        profile_artifact_path=unit.profile_artifact_path,
        profile_artifact_hash=unit.profile_artifact_hash or str(model_state.get("profile_artifact_hash") or "unavailable"),
        refit_window_start=model_state.get("refit_window_start"),
        refit_window_end=model_state.get("refit_window_end"),
        refit_cadence_id=cadence_id,
        cadence_checkpoint_index=int(checkpoint_index),
        cadence_checkpoint_count=int(checkpoint_count),
        definition_known_at_ts=model_state.get("definition_known_at_ts"),
        source_tail_ts=model_state.get("source_tail_ts"),
        status=status,
        health_metadata=dict(model_state.get("health_metadata") or unit.health_metadata or {}),
        lineage=lineage,
        planned_file_path=planned_path,
        planned_model_state_path=planned_model_state_path,
        relationship_input_tail_ts=relationship_tail.get("relationship_input_tail_ts"),
        relationship_known_at_ts=relationship_tail.get("relationship_known_at_ts"),
        relationship_input_lineage=dict(relationship_tail.get("relationship_input_lineage") or {}),
        local_reselection_eligible=bool(missing_required_fields or status != REGIME_PRODUCTION_MODEL_STATE_STATUS_PLANNED),
        local_reselection_execution_performed=False,
        bar_level_mask_triggers_refit=False,
    )


def _definition_status(unit) -> str:
    model_state = dict(unit.model_state_definition or {})
    if model_state.get("missing_required_fields"):
        return REGIME_PRODUCTION_MODEL_STATE_STATUS_MISSING_INPUT
    return str(model_state.get("status") or REGIME_PRODUCTION_MODEL_STATE_STATUS_PLANNED)


def _relationship_tail_payload(unit, model_state: Mapping[str, Any]) -> dict[str, Any]:
    if unit.branch != REGIME_BRANCH_CROSS_ASSET_STATE:
        return {
            "relationship_input_tail_ts": None,
            "relationship_known_at_ts": None,
            "relationship_input_lineage": {},
        }
    lineage = dict(model_state.get("lineage") or {})
    return {
        "relationship_input_tail_ts": model_state.get("source_tail_ts"),
        "relationship_known_at_ts": model_state.get("definition_known_at_ts"),
        "relationship_input_lineage": {
            "relationship_context_id": lineage.get("relationship_context_id"),
            "relationship_snapshot_id": lineage.get("relationship_snapshot_id"),
            "relationship_context_cadence_policy_id": unit.method_metadata.get("relationship_context_cadence_policy_id"),
            "snapshot_cadence_days": unit.method_metadata.get("snapshot_cadence_days"),
            "relationship_input_checks": [to_jsonable(dict(item)) for item in unit.relationship_input_checks],
            "relationship_input_history_check": to_jsonable(
                dict(dict(unit.timestamp_plan or {}).get("relationship_input_history_check") or {})
            ),
            "relationship_input_history_separate_from_selected_profile_artifact": True,
        },
    }


def _historical_months_for_unit(unit, default_months: int) -> int:
    months = dict(unit.timestamp_plan or {}).get("historical_output_months")
    if months not in (None, ""):
        return max(1, int(months))
    return int(default_months)


def _historical_months_from_plan_payload(plan_payload: Mapping[str, Any]) -> int:
    months = dict(plan_payload.get("planner_contract", {}).get("clamp_policy", {}) or {}).get("historical_output_months")
    if months not in (None, ""):
        return max(1, int(months))
    return 13


def _plan_refit_cadence_id(plan_payload: Mapping[str, Any]) -> str:
    return str(plan_payload.get("planner_contract", {}).get("refit_cadence", {}).get("refit_cadence_id") or "missing_refit_cadence_id")


def _checkpoint_count(cadence: str, months: int) -> int:
    return checkpoint_count_for_cadence(cadence, months)


def _cadence_from_id(cadence_id: str) -> str:
    text = str(cadence_id).lower()
    for cadence in ("biweekly", "weekly", "monthly"):
        if cadence in text:
            return cadence
    return "monthly"


def _canonical_root_checks(
    sandbox_root: Path,
    *,
    env: Mapping[str, str],
    project_root: str | Path | None,
) -> tuple[dict[str, Any], ...]:
    checks: list[dict[str, Any]] = []
    root = Path(sandbox_root).resolve()
    for key in CANONICAL_ROOT_KEYS:
        configured = key in PATH_KEYS
        raw = resolve_path(key, env=env, required=False) if configured else None
        if raw is None:
            checks.append({"root_key": key, "configured": False, "collision": False})
            continue
        candidate = resolve_project_path(raw, project_root=project_root)
        collision = is_relative_to(root, candidate) or is_relative_to(candidate, root)
        checks.append({"root_key": key, "configured": True, "collision": bool(collision)})
    return tuple(checks)


def _target_hash(target_key: Mapping[str, Any]) -> str:
    raw = json.dumps(to_jsonable(dict(target_key)), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:20]


def _safe_path_part(value: object) -> str:
    text = str(value or "").strip().replace("/", "_").replace("\\", "_").replace(":", "_")
    return text or "unknown"


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


def _validate_optional_order(start: Any, end: Any, *, context: str) -> None:
    if start in (None, "") or end in (None, ""):
        return
    if _to_orderable(start, field_name=f"{context} start") > _to_orderable(end, field_name=f"{context} end"):
        raise ValueError(f"Regime Production definition planner {context} start must be <= end")


def _to_orderable(value: Any, *, field_name: str) -> float:
    if isinstance(value, bool) or value is None:
        raise ValueError(f"Regime Production definition planner {field_name} must be a timestamp")
    try:
        return float(value)
    except Exception:
        pass
    from datetime import datetime

    text = _text(value, field_name=field_name)
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()
    except Exception as exc:
        raise ValueError(f"Regime Production definition planner {field_name} must be numeric or ISO datetime") from exc


def _text(value: object, *, field_name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"Regime Production definition planner {field_name} must be non-empty")
    return text


__all__ = [
    "DEFINITION_SANDBOX_ROOT_NAMESPACE",
    "REGIME_PRODUCTION_DEFINITION_PLAN_ARTIFACT_KIND",
    "REGIME_PRODUCTION_DEFINITION_PLANNER_SCHEMA_VERSION",
    "REGIME_PRODUCTION_DEFINITION_RECORD_ARTIFACT_KIND",
    "REGIME_PRODUCTION_DEFINITION_ROOT_ARTIFACT_KIND",
    "REGIME_PRODUCTION_DEFINITION_SUMMARY_ARTIFACT_KIND",
    "RegimeProductionDefinitionPlannerConfig",
    "RegimeProductionDefinitionRecord",
    "RegimeProductionDefinitionRootContract",
    "RegimeProductionDefinitionSchedulePlan",
    "plan_regime_production_definition_schedule_from_no_write_plan",
    "resolve_regime_production_definition_root_contract",
    "run_regime_production_definition_schedule_planner",
    "write_regime_production_definition_planner_summary",
]
