from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.regimes.core.clamp_policy import RegimeClampPolicy
from src.regimes.core.production_consumer import (
    REGIME_BRANCH_ASSET_STATE,
    REGIME_BRANCH_CROSS_ASSET_STATE,
    REGIME_BRANCH_MARKET_STATE,
    REGIME_PRODUCTION_BRANCHES,
    REGIME_PRODUCTION_REASON_FAILED_HEALTH_GATE,
    REGIME_PRODUCTION_REASON_INSUFFICIENT,
    REGIME_PRODUCTION_REASON_INVALID_PROFILE,
    REGIME_PRODUCTION_REASON_MISSING_INPUT,
    REGIME_PRODUCTION_REASON_NOT_CLUSTERABLE,
    REGIME_PRODUCTION_STATUS_BLOCKED,
    REGIME_PRODUCTION_STATUS_READY_FOR_DRY_CONSUMPTION,
    validate_regime_production_gate,
)
from src.regimes.core.serialization import to_jsonable


REGIME_PRODUCTION_PLANNER_SCHEMA_VERSION = 1
REGIME_PRODUCTION_PLANNER_ARTIFACT_KIND = "regime_production_planner_contract"
REGIME_PRODUCTION_OUTPUT_GRAIN_ARTIFACT_KIND = "regime_production_output_grain_contract"
REGIME_PRODUCTION_CLAMP_POLICY_ARTIFACT_KIND = "regime_production_clamp_planning_policy"
REGIME_PRODUCTION_REFIT_CADENCE_ARTIFACT_KIND = "regime_production_refit_cadence_contract"
REGIME_PRODUCTION_LABEL_STATUS_ARTIFACT_KIND = "regime_production_label_status_contract"
REGIME_PRODUCTION_MODEL_STATE_ARTIFACT_KIND = "regime_production_model_state_definition_contract"
REGIME_PRODUCTION_RELATIONSHIP_INPUT_ARTIFACT_KIND = "regime_production_relationship_input_contract"
REGIME_PRODUCTION_NORMALIZED_LINEAGE_ARTIFACT_KIND = "regime_production_normalized_lineage"
REGIME_PRODUCTION_PLANNING_UNIT_ARTIFACT_KIND = "regime_production_no_write_planning_unit"
REGIME_PRODUCTION_NO_WRITE_PLAN_ARTIFACT_KIND = "regime_production_no_write_plan"

REGIME_PRODUCTION_LABEL_STATUS_VALID = "valid_label"
REGIME_PRODUCTION_LABEL_STATUS_LOW_CONFIDENCE = "low_confidence"
REGIME_PRODUCTION_LABEL_STATUS_UNKNOWN_NOISE = "unknown_noise"
REGIME_PRODUCTION_LABEL_STATUS_MASKED_UNAVAILABLE = "masked_unavailable"
REGIME_PRODUCTION_LABEL_STATUS_INSUFFICIENT = "insufficient"
REGIME_PRODUCTION_LABEL_STATUS_NOT_CLUSTERABLE = "not_clusterable"
REGIME_PRODUCTION_LABEL_STATUS_STALE_RELATIONSHIP_INPUT = "stale_relationship_input"
REGIME_PRODUCTION_LABEL_STATUS_FAILED_PROFILE_HEALTH = "failed_profile_health"
REGIME_PRODUCTION_LABEL_STATUS_MISSING_INPUT = "missing_input"
REGIME_PRODUCTION_LABEL_STATUS_INVALID_PROFILE = "invalid_profile"
REGIME_PRODUCTION_LABEL_STATUSES: tuple[str, ...] = (
    REGIME_PRODUCTION_LABEL_STATUS_VALID,
    REGIME_PRODUCTION_LABEL_STATUS_LOW_CONFIDENCE,
    REGIME_PRODUCTION_LABEL_STATUS_UNKNOWN_NOISE,
    REGIME_PRODUCTION_LABEL_STATUS_MASKED_UNAVAILABLE,
    REGIME_PRODUCTION_LABEL_STATUS_INSUFFICIENT,
    REGIME_PRODUCTION_LABEL_STATUS_NOT_CLUSTERABLE,
    REGIME_PRODUCTION_LABEL_STATUS_STALE_RELATIONSHIP_INPUT,
    REGIME_PRODUCTION_LABEL_STATUS_FAILED_PROFILE_HEALTH,
    REGIME_PRODUCTION_LABEL_STATUS_MISSING_INPUT,
    REGIME_PRODUCTION_LABEL_STATUS_INVALID_PROFILE,
)

REGIME_PRODUCTION_REFIT_CADENCE_WEEKLY = "weekly"
REGIME_PRODUCTION_REFIT_CADENCE_BIWEEKLY = "biweekly"
REGIME_PRODUCTION_REFIT_CADENCE_MONTHLY = "monthly"
REGIME_PRODUCTION_REFIT_CADENCES: tuple[str, ...] = (
    REGIME_PRODUCTION_REFIT_CADENCE_WEEKLY,
    REGIME_PRODUCTION_REFIT_CADENCE_BIWEEKLY,
    REGIME_PRODUCTION_REFIT_CADENCE_MONTHLY,
)

REGIME_PRODUCTION_MODEL_STATE_STATUS_PLANNED = "planned"
REGIME_PRODUCTION_MODEL_STATE_STATUS_ACTIVE = "active_definition"
REGIME_PRODUCTION_MODEL_STATE_STATUS_STALE = "stale_definition"
REGIME_PRODUCTION_MODEL_STATE_STATUS_BLOCKED = "blocked"
REGIME_PRODUCTION_MODEL_STATE_STATUS_MISSING_INPUT = REGIME_PRODUCTION_LABEL_STATUS_MISSING_INPUT
REGIME_PRODUCTION_MODEL_STATE_STATUS_INVALID_PROFILE = REGIME_PRODUCTION_LABEL_STATUS_INVALID_PROFILE
REGIME_PRODUCTION_MODEL_STATE_STATUS_FAILED_PROFILE_HEALTH = REGIME_PRODUCTION_LABEL_STATUS_FAILED_PROFILE_HEALTH
REGIME_PRODUCTION_MODEL_STATE_STATUSES: tuple[str, ...] = (
    REGIME_PRODUCTION_MODEL_STATE_STATUS_PLANNED,
    REGIME_PRODUCTION_MODEL_STATE_STATUS_ACTIVE,
    REGIME_PRODUCTION_MODEL_STATE_STATUS_STALE,
    REGIME_PRODUCTION_MODEL_STATE_STATUS_BLOCKED,
    REGIME_PRODUCTION_MODEL_STATE_STATUS_FAILED_PROFILE_HEALTH,
    REGIME_PRODUCTION_MODEL_STATE_STATUS_MISSING_INPUT,
    REGIME_PRODUCTION_MODEL_STATE_STATUS_INVALID_PROFILE,
)

REGIME_PRODUCTION_PLANNING_UNIT_STATUS_SELECTED = "selected"
REGIME_PRODUCTION_PLANNING_UNIT_STATUS_MASKED_UNAVAILABLE = "masked_unavailable"
REGIME_PRODUCTION_PLANNING_UNIT_STATUS_SKIPPED = "skipped_or_filtered"
REGIME_PRODUCTION_PLANNING_UNIT_STATUS_DIAGNOSTIC_ONLY = "diagnostic_only"
REGIME_PRODUCTION_PLANNING_UNIT_STATUS_MISSING_INPUT = "missing_input"
REGIME_PRODUCTION_PLANNING_UNIT_STATUS_INVALID_PROFILE = "invalid_profile"
REGIME_PRODUCTION_PLANNING_UNIT_STATUSES: tuple[str, ...] = (
    REGIME_PRODUCTION_PLANNING_UNIT_STATUS_SELECTED,
    REGIME_PRODUCTION_PLANNING_UNIT_STATUS_MASKED_UNAVAILABLE,
    REGIME_PRODUCTION_PLANNING_UNIT_STATUS_SKIPPED,
    REGIME_PRODUCTION_PLANNING_UNIT_STATUS_DIAGNOSTIC_ONLY,
    REGIME_PRODUCTION_PLANNING_UNIT_STATUS_MISSING_INPUT,
    REGIME_PRODUCTION_PLANNING_UNIT_STATUS_INVALID_PROFILE,
)

BRANCH_CLAMP_PATHWAYS: Mapping[str, str] = {
    REGIME_BRANCH_ASSET_STATE: "asset_state",
    REGIME_BRANCH_MARKET_STATE: "market_state",
    REGIME_BRANCH_CROSS_ASSET_STATE: "relative_state",
}

BRANCH_OUTPUT_GRAIN_FIELDS: Mapping[str, tuple[str, ...]] = {
    REGIME_BRANCH_ASSET_STATE: ("asset_id", "axis", "band", "timestamp"),
    REGIME_BRANCH_MARKET_STATE: ("market_axis", "band", "timestamp"),
    REGIME_BRANCH_CROSS_ASSET_STATE: ("asset_id", "relationship_feature_family", "band", "timestamp"),
}

BRANCH_TARGET_KEY_FIELDS: Mapping[str, tuple[str, ...]] = {
    REGIME_BRANCH_ASSET_STATE: ("asset_id", "axis", "band"),
    REGIME_BRANCH_MARKET_STATE: ("market_axis", "band"),
    REGIME_BRANCH_CROSS_ASSET_STATE: ("asset_id", "relationship_feature_family", "band"),
}

MODEL_STATE_REQUIRED_FIELDS: tuple[str, ...] = (
    "branch",
    "target_key",
    "grain_key",
    "profile_id",
    "profile_version",
    "profile_artifact_path",
    "profile_artifact_hash",
    "refit_window_start",
    "refit_window_end",
    "definition_known_at_ts",
    "source_tail_ts",
    "refit_cadence_id",
    "status",
    "health_metadata",
    "lineage",
)


@dataclass(frozen=True)
class RegimeProductionOutputGrainContract:
    branch: str
    grain_fields: Sequence[str]
    timestamp_field: str = "timestamp"
    label_field: str = "label_status"
    value_field: str = "label_value"

    def __post_init__(self) -> None:
        branch = _branch_name(self.branch)
        fields = _string_tuple(self.grain_fields, field_name="grain_fields")
        expected = BRANCH_OUTPUT_GRAIN_FIELDS[branch]
        if fields != expected:
            raise ValueError(
                f"Regime Production output grain for {branch!r} must be {expected!r}; received {fields!r}"
            )
        if self.timestamp_field not in fields:
            raise ValueError("Regime Production output grain must include timestamp field")
        object.__setattr__(self, "branch", branch)
        object.__setattr__(self, "grain_fields", fields)
        object.__setattr__(self, "timestamp_field", _text(self.timestamp_field, field_name="timestamp_field"))
        object.__setattr__(self, "label_field", _text(self.label_field, field_name="label_field"))
        object.__setattr__(self, "value_field", _text(self.value_field, field_name="value_field"))

    @property
    def grain_id(self) -> str:
        return " x ".join(self.grain_fields)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": REGIME_PRODUCTION_PLANNER_SCHEMA_VERSION,
            "artifact_kind": REGIME_PRODUCTION_OUTPUT_GRAIN_ARTIFACT_KIND,
            "branch": self.branch,
            "grain_id": self.grain_id,
            "grain_fields": list(self.grain_fields),
            "timestamp_field": self.timestamp_field,
            "label_field": self.label_field,
            "value_field": self.value_field,
            "production_labels_written": False,
            "production_outputs_written": False,
            "canonical_production_state_outputs_written": False,
        }


@dataclass(frozen=True)
class RegimeProductionClampPlanningPolicy:
    policy_id: str
    source_contract: str
    source_module: str
    core_clamp_contract: RegimeClampPolicy
    numeric_clamp_start_year: int
    numeric_clamp_start_month: int
    historical_output_months: int
    required_lookback_months: int
    applies_to_branches: Sequence[str]
    calendar_window_policy: str = "numeric_forecast_common_recent_window"
    runtime_boundaries_required: bool = True

    def __post_init__(self) -> None:
        branches = tuple(dict.fromkeys(_branch_name(branch) for branch in self.applies_to_branches))
        if not branches:
            raise ValueError("Regime Production clamp planning policy requires at least one branch")
        if int(self.numeric_clamp_start_month) < 1 or int(self.numeric_clamp_start_month) > 12:
            raise ValueError("Regime Production numeric clamp start month must be within 1..12")
        if int(self.historical_output_months) <= 0 or int(self.required_lookback_months) <= 0:
            raise ValueError("Regime Production clamp windows must be positive month counts")
        pathways = tuple(dict.fromkeys(BRANCH_CLAMP_PATHWAYS[branch] for branch in branches))
        if tuple(self.core_clamp_contract.applies_to_pathways) != pathways:
            raise ValueError("Regime Production clamp core contract pathways must match branch mapping")
        object.__setattr__(self, "policy_id", _text(self.policy_id, field_name="policy_id"))
        object.__setattr__(self, "source_contract", _text(self.source_contract, field_name="source_contract"))
        object.__setattr__(self, "source_module", _text(self.source_module, field_name="source_module"))
        object.__setattr__(self, "numeric_clamp_start_year", int(self.numeric_clamp_start_year))
        object.__setattr__(self, "numeric_clamp_start_month", int(self.numeric_clamp_start_month))
        object.__setattr__(self, "historical_output_months", int(self.historical_output_months))
        object.__setattr__(self, "required_lookback_months", int(self.required_lookback_months))
        object.__setattr__(self, "applies_to_branches", branches)
        object.__setattr__(self, "calendar_window_policy", _text(self.calendar_window_policy, field_name="calendar_window_policy"))

    @property
    def applies_to_pathways(self) -> tuple[str, ...]:
        return tuple(self.core_clamp_contract.applies_to_pathways)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": REGIME_PRODUCTION_PLANNER_SCHEMA_VERSION,
            "artifact_kind": REGIME_PRODUCTION_CLAMP_POLICY_ARTIFACT_KIND,
            "policy_id": self.policy_id,
            "source_contract": self.source_contract,
            "source_module": self.source_module,
            "calendar_window_policy": self.calendar_window_policy,
            "core_clamp_contract": self.core_clamp_contract.as_dict(),
            "numeric_clamp_start": {
                "year": int(self.numeric_clamp_start_year),
                "month": int(self.numeric_clamp_start_month),
            },
            "historical_output_months": int(self.historical_output_months),
            "required_lookback_months": int(self.required_lookback_months),
            "roughly_one_year_history_default": True,
            "runtime_boundaries_required": bool(self.runtime_boundaries_required),
            "applies_to_branches": list(self.applies_to_branches),
            "applies_to_pathways": list(self.applies_to_pathways),
            "production_outputs_written": False,
            "canonical_production_state_outputs_written": False,
        }


@dataclass(frozen=True)
class RegimeProductionRefitCadenceContract:
    branch: str
    refit_cadence_id: str
    cadence: str
    cadence_source: str
    interval_minutes: int | None = None
    cadence_mode: str = "calendar_based_refit"
    per_bar_refit_enabled: bool = False
    definitions_persist_across_walkthroughs: bool = True
    model_state_persistence: str = "one_record_per_branch_target_key"
    state_update_policy: str = "update_definition_on_cadence_only"

    def __post_init__(self) -> None:
        branch = _branch_name(self.branch)
        cadence = _text(self.cadence, field_name="cadence").lower()
        if cadence not in REGIME_PRODUCTION_REFIT_CADENCES:
            raise ValueError(f"Unsupported Regime Production refit cadence: {cadence!r}")
        if self.interval_minutes is not None and int(self.interval_minutes) <= 0:
            raise ValueError("Regime Production refit interval_minutes must be positive")
        if self.per_bar_refit_enabled:
            raise ValueError("Regime Production planner contracts cannot enable per-bar refit")
        object.__setattr__(self, "branch", branch)
        object.__setattr__(self, "refit_cadence_id", _text(self.refit_cadence_id, field_name="refit_cadence_id"))
        object.__setattr__(self, "cadence", cadence)
        object.__setattr__(self, "cadence_source", _text(self.cadence_source, field_name="cadence_source"))
        object.__setattr__(self, "interval_minutes", None if self.interval_minutes is None else int(self.interval_minutes))
        object.__setattr__(self, "cadence_mode", _text(self.cadence_mode, field_name="cadence_mode"))
        object.__setattr__(self, "per_bar_refit_enabled", False)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": REGIME_PRODUCTION_PLANNER_SCHEMA_VERSION,
            "artifact_kind": REGIME_PRODUCTION_REFIT_CADENCE_ARTIFACT_KIND,
            "branch": self.branch,
            "refit_cadence_id": self.refit_cadence_id,
            "cadence_mode": self.cadence_mode,
            "cadence": self.cadence,
            "cadence_source": self.cadence_source,
            "interval_minutes": self.interval_minutes,
            "calendar_based": True,
            "per_bar_refit_enabled": False,
            "definitions_persist_across_walkthroughs": bool(self.definitions_persist_across_walkthroughs),
            "model_state_persistence": self.model_state_persistence,
            "state_update_policy": self.state_update_policy,
            "production_writer_enabled": False,
        }


@dataclass(frozen=True)
class RegimeProductionLabelStatusContract:
    statuses: Sequence[str] = REGIME_PRODUCTION_LABEL_STATUSES

    def __post_init__(self) -> None:
        statuses = _string_tuple(self.statuses, field_name="statuses")
        missing = [status for status in REGIME_PRODUCTION_LABEL_STATUSES if status not in statuses]
        if missing:
            raise ValueError(f"Regime Production label status contract missing statuses: {missing!r}")
        object.__setattr__(self, "statuses", statuses)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": REGIME_PRODUCTION_PLANNER_SCHEMA_VERSION,
            "artifact_kind": REGIME_PRODUCTION_LABEL_STATUS_ARTIFACT_KIND,
            "statuses": list(self.statuses),
            "valid_label_status": REGIME_PRODUCTION_LABEL_STATUS_VALID,
            "unavailable_statuses": [
                REGIME_PRODUCTION_LABEL_STATUS_MASKED_UNAVAILABLE,
                REGIME_PRODUCTION_LABEL_STATUS_INSUFFICIENT,
                REGIME_PRODUCTION_LABEL_STATUS_NOT_CLUSTERABLE,
                REGIME_PRODUCTION_LABEL_STATUS_STALE_RELATIONSHIP_INPUT,
                REGIME_PRODUCTION_LABEL_STATUS_FAILED_PROFILE_HEALTH,
                REGIME_PRODUCTION_LABEL_STATUS_MISSING_INPUT,
                REGIME_PRODUCTION_LABEL_STATUS_INVALID_PROFILE,
            ],
            "preserve_downstream_shape": True,
            "missing_profile_or_cell_may_disappear": False,
            "production_labels_written": False,
        }


@dataclass(frozen=True)
class RegimeProductionModelStateDefinition:
    branch: str
    target_key: Mapping[str, Any]
    profile_id: str
    profile_version: str
    profile_artifact_path: str
    profile_artifact_hash: str
    refit_window_start: int | float | str
    refit_window_end: int | float | str
    definition_known_at_ts: int | float | str
    source_tail_ts: int | float | str
    refit_cadence_id: str
    status: str
    health_metadata: Mapping[str, Any]
    lineage: Mapping[str, Any]
    grain_key: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        branch = _branch_name(self.branch)
        target_key = to_jsonable(dict(self.target_key))
        expected_fields = BRANCH_TARGET_KEY_FIELDS[branch]
        missing = [field_name for field_name in expected_fields if field_name not in target_key or target_key.get(field_name) in (None, "")]
        if missing:
            raise ValueError(f"Regime Production model state target_key missing required fields: {missing!r}")
        grain_key = to_jsonable(dict(self.grain_key or target_key))
        if grain_key != target_key:
            raise ValueError("Regime Production model state grain_key must match target_key")
        status = _text(self.status, field_name="status")
        if status not in REGIME_PRODUCTION_MODEL_STATE_STATUSES:
            raise ValueError(f"Unsupported Regime Production model-state status: {status!r}")
        _validate_order(self.refit_window_start, self.refit_window_end, context="refit window")
        if _to_orderable(self.source_tail_ts, field_name="source_tail_ts") > _to_orderable(
            self.definition_known_at_ts,
            field_name="definition_known_at_ts",
        ):
            raise ValueError("Regime Production model state source_tail_ts must not exceed definition_known_at_ts")
        object.__setattr__(self, "branch", branch)
        object.__setattr__(self, "target_key", target_key)
        object.__setattr__(self, "grain_key", grain_key)
        object.__setattr__(self, "profile_id", _text(self.profile_id, field_name="profile_id"))
        object.__setattr__(self, "profile_version", _text(self.profile_version, field_name="profile_version"))
        object.__setattr__(self, "profile_artifact_path", _text(self.profile_artifact_path, field_name="profile_artifact_path"))
        object.__setattr__(self, "profile_artifact_hash", _text(self.profile_artifact_hash, field_name="profile_artifact_hash"))
        object.__setattr__(self, "refit_cadence_id", _text(self.refit_cadence_id, field_name="refit_cadence_id"))
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "health_metadata", to_jsonable(dict(self.health_metadata)))
        object.__setattr__(self, "lineage", to_jsonable(dict(self.lineage)))

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": REGIME_PRODUCTION_PLANNER_SCHEMA_VERSION,
            "artifact_kind": REGIME_PRODUCTION_MODEL_STATE_ARTIFACT_KIND,
            "branch": self.branch,
            "target_key": to_jsonable(dict(self.target_key)),
            "grain_key": to_jsonable(dict(self.grain_key or {})),
            "profile_id": self.profile_id,
            "profile_version": self.profile_version,
            "profile_artifact_path": self.profile_artifact_path,
            "profile_artifact_hash": self.profile_artifact_hash,
            "refit_window_start": self.refit_window_start,
            "refit_window_end": self.refit_window_end,
            "definition_known_at_ts": self.definition_known_at_ts,
            "source_tail_ts": self.source_tail_ts,
            "refit_cadence_id": self.refit_cadence_id,
            "status": self.status,
            "health_metadata": to_jsonable(dict(self.health_metadata)),
            "lineage": to_jsonable(dict(self.lineage)),
            "required_fields": list(MODEL_STATE_REQUIRED_FIELDS),
            "production_writer_enabled": False,
            "production_labels_written": False,
            "production_outputs_written": False,
            "canonical_production_state_outputs_written": False,
        }


@dataclass(frozen=True)
class RegimeProductionNormalizedLineage:
    branch: str
    profile_id: str | None
    profile_version: str | None
    selected_profile_artifact_path: str | Path | None
    selected_profile_artifact_hash: str | None
    source_tail_ts: Any = None
    known_at_ts: Any = None
    lineage_id: str | None = None
    normalized_row_lineage_id: str | None = None
    run_id: str | None = None
    source_run_reference: str | None = None
    manifest_schema_version: int | None = None
    branch_schema_policy: str | None = None
    raw_version_field: str | None = None
    raw_version_value: Any = None
    raw_lineage_id: Any = None
    raw_lineage_fields: Mapping[str, Any] = field(default_factory=dict)
    row_status: str = REGIME_PRODUCTION_PLANNING_UNIT_STATUS_SELECTED
    availability_reason_codes: Sequence[str] = ()
    lineage_reason_codes: Sequence[str] = ()

    def __post_init__(self) -> None:
        branch = _branch_name(self.branch)
        row_status = _text(self.row_status, field_name="row_status")
        if row_status not in REGIME_PRODUCTION_PLANNING_UNIT_STATUSES:
            raise ValueError(f"Unsupported Regime Production normalized lineage row_status: {row_status!r}")
        reasons = [str(reason) for reason in self.lineage_reason_codes if str(reason or "").strip()]
        if not str(self.profile_id or "").strip():
            reasons.append("profile_id_missing")
        if not str(self.profile_version or "").strip():
            reasons.append("profile_version_missing")
        if not str(self.selected_profile_artifact_path or "").strip():
            reasons.append("selected_profile_artifact_path_missing")
        if not str(self.selected_profile_artifact_hash or "").strip():
            reasons.append("selected_profile_artifact_hash_missing")
        if not str(self.lineage_id or self.normalized_row_lineage_id or "").strip():
            reasons.append("lineage_id_or_normalized_row_lineage_id_missing")
        if self.manifest_schema_version is None:
            reasons.append("manifest_schema_version_missing")
        if not str(self.branch_schema_policy or "").strip():
            reasons.append("branch_schema_policy_missing")
        if row_status == REGIME_PRODUCTION_PLANNING_UNIT_STATUS_SELECTED:
            if self.source_tail_ts in (None, ""):
                reasons.append("source_tail_ts_missing")
            if self.known_at_ts in (None, ""):
                reasons.append("known_at_ts_missing")
        if self.source_tail_ts not in (None, "") and self.known_at_ts not in (None, ""):
            try:
                if _to_orderable(self.source_tail_ts, field_name="source_tail_ts") > _to_orderable(
                    self.known_at_ts,
                    field_name="known_at_ts",
                ):
                    reasons.append("source_tail_ts_after_known_at_ts")
            except Exception:
                reasons.append("lineage_timestamp_invalid")
        if row_status != REGIME_PRODUCTION_PLANNING_UNIT_STATUS_SELECTED and not tuple(self.availability_reason_codes):
            reasons.append("unavailable_reason_code_missing")
        object.__setattr__(self, "branch", branch)
        object.__setattr__(self, "profile_id", None if self.profile_id is None else str(self.profile_id))
        object.__setattr__(self, "profile_version", None if self.profile_version is None else str(self.profile_version))
        object.__setattr__(
            self,
            "selected_profile_artifact_path",
            None if self.selected_profile_artifact_path is None else str(self.selected_profile_artifact_path),
        )
        object.__setattr__(
            self,
            "selected_profile_artifact_hash",
            None if self.selected_profile_artifact_hash is None else str(self.selected_profile_artifact_hash),
        )
        object.__setattr__(self, "lineage_id", None if self.lineage_id is None else str(self.lineage_id))
        object.__setattr__(
            self,
            "normalized_row_lineage_id",
            None if self.normalized_row_lineage_id is None else str(self.normalized_row_lineage_id),
        )
        object.__setattr__(self, "run_id", None if self.run_id is None else str(self.run_id))
        object.__setattr__(
            self,
            "source_run_reference",
            None if self.source_run_reference is None else str(self.source_run_reference),
        )
        object.__setattr__(self, "manifest_schema_version", None if self.manifest_schema_version is None else int(self.manifest_schema_version))
        object.__setattr__(self, "branch_schema_policy", None if self.branch_schema_policy is None else str(self.branch_schema_policy))
        object.__setattr__(self, "raw_version_field", None if self.raw_version_field is None else str(self.raw_version_field))
        object.__setattr__(self, "raw_lineage_fields", to_jsonable(dict(self.raw_lineage_fields)))
        object.__setattr__(self, "row_status", row_status)
        object.__setattr__(
            self,
            "availability_reason_codes",
            tuple(dict.fromkeys(str(reason) for reason in self.availability_reason_codes if str(reason or "").strip())),
        )
        object.__setattr__(self, "lineage_reason_codes", tuple(dict.fromkeys(reasons)))

    @property
    def passed(self) -> bool:
        return not self.lineage_reason_codes

    @property
    def lineage_status(self) -> str:
        if not self.passed:
            return "blocked"
        if self.row_status == REGIME_PRODUCTION_PLANNING_UNIT_STATUS_SELECTED:
            return "traceable"
        return "auditable_unavailable"

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": REGIME_PRODUCTION_PLANNER_SCHEMA_VERSION,
            "artifact_kind": REGIME_PRODUCTION_NORMALIZED_LINEAGE_ARTIFACT_KIND,
            "branch": self.branch,
            "profile_id": self.profile_id,
            "profile_version": self.profile_version,
            "lineage_id": self.lineage_id,
            "normalized_row_lineage_id": self.normalized_row_lineage_id,
            "source_tail_ts": self.source_tail_ts,
            "known_at_ts": self.known_at_ts,
            "selected_profile_artifact_path": self.selected_profile_artifact_path,
            "selected_profile_artifact_hash": self.selected_profile_artifact_hash,
            "run_id": self.run_id,
            "source_run_reference": self.source_run_reference,
            "manifest_schema_version": self.manifest_schema_version,
            "branch_schema_policy": self.branch_schema_policy,
            "raw_version_field": self.raw_version_field,
            "raw_version_value": to_jsonable(self.raw_version_value),
            "raw_lineage_id": to_jsonable(self.raw_lineage_id),
            "raw_lineage_fields": to_jsonable(dict(self.raw_lineage_fields)),
            "row_status": self.row_status,
            "availability_reason_codes": list(self.availability_reason_codes),
            "lineage_reason_codes": list(self.lineage_reason_codes),
            "lineage_status": self.lineage_status,
            "passed": self.passed,
            "production_writer_enabled": False,
            "production_labels_written": False,
            "production_outputs_written": False,
            "canonical_production_state_outputs_written": False,
        }


@dataclass(frozen=True)
class RegimeProductionRelationshipInputContract:
    input_id: str
    input_kind: str
    source_tail_ts: int | float | str | None = None
    known_at_ts: int | float | str | None = None
    branch: str = REGIME_BRANCH_CROSS_ASSET_STATE
    input_role: str = "time_indexed_relationship_data_input"
    selected_profile_artifact: bool = False
    recurring_input_data: bool = True
    relationship_discovery_allowed: bool = False
    relationship_discovery_requested: bool = False
    path: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        branch = _branch_name(self.branch)
        if branch != REGIME_BRANCH_CROSS_ASSET_STATE:
            raise ValueError("Regime Production relationship input contracts are Cross-Asset only")
        if self.selected_profile_artifact:
            raise ValueError("Relationship sidecars/features cannot be selected-profile artifacts")
        if self.relationship_discovery_allowed or self.relationship_discovery_requested:
            raise ValueError("Relationship discovery is outside the Production planner contract")
        if (self.source_tail_ts is None) != (self.known_at_ts is None):
            raise ValueError("Relationship inputs require source_tail_ts and known_at_ts together when either is present")
        if self.source_tail_ts is not None and self.known_at_ts is not None:
            if _to_orderable(self.source_tail_ts, field_name="source_tail_ts") > _to_orderable(self.known_at_ts, field_name="known_at_ts"):
                raise ValueError("Relationship input source_tail_ts must not exceed known_at_ts")
        object.__setattr__(self, "branch", branch)
        object.__setattr__(self, "input_id", _text(self.input_id, field_name="input_id"))
        object.__setattr__(self, "input_kind", _text(self.input_kind, field_name="input_kind"))
        object.__setattr__(self, "input_role", _text(self.input_role, field_name="input_role"))
        object.__setattr__(self, "selected_profile_artifact", False)
        object.__setattr__(self, "recurring_input_data", True)
        object.__setattr__(self, "relationship_discovery_allowed", False)
        object.__setattr__(self, "relationship_discovery_requested", False)
        object.__setattr__(self, "path", None if self.path is None else _text(self.path, field_name="path"))
        object.__setattr__(self, "metadata", to_jsonable(dict(self.metadata)))

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": REGIME_PRODUCTION_PLANNER_SCHEMA_VERSION,
            "artifact_kind": REGIME_PRODUCTION_RELATIONSHIP_INPUT_ARTIFACT_KIND,
            "branch": self.branch,
            "input_id": self.input_id,
            "input_kind": self.input_kind,
            "input_role": self.input_role,
            "path": self.path,
            "source_tail_ts": self.source_tail_ts,
            "known_at_ts": self.known_at_ts,
            "time_indexed_input_data": True,
            "recurring_input_data": True,
            "selected_profile_artifact": False,
            "relationship_discovery_allowed": False,
            "relationship_discovery_requested": False,
            "production_writer_enabled": False,
            "metadata": to_jsonable(dict(self.metadata)),
        }


@dataclass(frozen=True)
class RegimeProductionPlannerContract:
    branch: str
    output_grain: RegimeProductionOutputGrainContract
    clamp_policy: RegimeProductionClampPlanningPolicy
    refit_cadence: RegimeProductionRefitCadenceContract
    label_status_contract: RegimeProductionLabelStatusContract = field(default_factory=RegimeProductionLabelStatusContract)
    relationship_inputs: Sequence[RegimeProductionRelationshipInputContract] = ()
    production_approved: bool = False
    production_writer_enabled: bool = False
    production_labels_written: bool = False
    production_outputs_written: bool = False
    canonical_production_state_outputs_written: bool = False
    requires_human_approval_before_production: bool = True

    def __post_init__(self) -> None:
        branch = _branch_name(self.branch)
        if self.output_grain.branch != branch:
            raise ValueError("Regime Production planner output_grain branch mismatch")
        if self.refit_cadence.branch != branch:
            raise ValueError("Regime Production planner refit_cadence branch mismatch")
        if branch not in self.clamp_policy.applies_to_branches:
            raise ValueError("Regime Production planner clamp_policy does not apply to branch")
        relationship_inputs = tuple(self.relationship_inputs)
        if branch != REGIME_BRANCH_CROSS_ASSET_STATE and relationship_inputs:
            raise ValueError("Relationship input contracts are only valid for Cross-Asset Production planning")
        if any(item.branch != REGIME_BRANCH_CROSS_ASSET_STATE for item in relationship_inputs):
            raise ValueError("Relationship input contract branch mismatch")
        if (
            self.production_approved
            or self.production_writer_enabled
            or self.production_labels_written
            or self.production_outputs_written
            or self.canonical_production_state_outputs_written
        ):
            raise ValueError("Regime Production planner contracts cannot enable or record production writes")
        object.__setattr__(self, "branch", branch)
        object.__setattr__(self, "relationship_inputs", relationship_inputs)

    def as_dict(self) -> dict[str, Any]:
        gate = validate_regime_production_planner_gates(
            self.branch,
            requires_human_approval_before_production=bool(self.requires_human_approval_before_production),
        )
        return {
            "schema_version": REGIME_PRODUCTION_PLANNER_SCHEMA_VERSION,
            "artifact_kind": REGIME_PRODUCTION_PLANNER_ARTIFACT_KIND,
            "branch": self.branch,
            "output_grain": self.output_grain.as_dict(),
            "clamp_policy": self.clamp_policy.as_dict(),
            "refit_cadence": self.refit_cadence.as_dict(),
            "label_status_contract": self.label_status_contract.as_dict(),
            "model_state_required_fields": list(MODEL_STATE_REQUIRED_FIELDS),
            "model_state_persistence": "one_definition_record_per_branch_target_key",
            "selected_profile_artifact_contract": {
                "active_selected_profile_artifact_count": 1,
                "selected_profile_artifact_role": "settings_contract",
                "relationship_sidecars_are_selected_profile_artifacts": False,
            },
            "relationship_input_contracts": [item.as_dict() for item in self.relationship_inputs],
            "relationship_inputs_role": "time_indexed_input_data_not_selected_profile_artifacts"
            if self.branch == REGIME_BRANCH_CROSS_ASSET_STATE
            else None,
            "production_approved": False,
            "production_writer_enabled": False,
            "production_labels_written": False,
            "production_outputs_written": False,
            "canonical_production_state_outputs_written": False,
            "requires_human_approval_before_production": bool(self.requires_human_approval_before_production),
            "production_gate_validation": gate.as_dict(),
            "production_writer_gates_fail_closed": True,
        }


@dataclass(frozen=True)
class RegimeProductionPlanningUnit:
    branch: str
    target_key: Mapping[str, Any]
    output_grain_key: Mapping[str, Any]
    timestamp_plan: Mapping[str, Any]
    planning_status: str
    profile_id: str | None = None
    profile_version: str | None = None
    profile_artifact_path: str | None = None
    profile_artifact_hash: str | None = None
    model_state_definition: RegimeProductionModelStateDefinition | Mapping[str, Any] | None = None
    method_metadata: Mapping[str, Any] = field(default_factory=dict)
    health_metadata: Mapping[str, Any] = field(default_factory=dict)
    local_reselection_metadata: Mapping[str, Any] = field(default_factory=dict)
    normalized_lineage: RegimeProductionNormalizedLineage | Mapping[str, Any] | None = None
    relationship_input_checks: Sequence[Mapping[str, Any]] = ()
    reason_codes: Sequence[str] = ()
    warnings: Sequence[str] = ()

    def __post_init__(self) -> None:
        branch = _branch_name(self.branch)
        target_key = to_jsonable(dict(self.target_key))
        output_key = to_jsonable(dict(self.output_grain_key))
        timestamp_plan = to_jsonable(dict(self.timestamp_plan))
        status = _text(self.planning_status, field_name="planning_status")
        if status not in REGIME_PRODUCTION_PLANNING_UNIT_STATUSES:
            raise ValueError(f"Unsupported Regime Production planning unit status: {status!r}")
        expected_fields = BRANCH_TARGET_KEY_FIELDS[branch]
        missing = [field_name for field_name in expected_fields if target_key.get(field_name) in (None, "")]
        if missing:
            raise ValueError(f"Regime Production planning unit target_key missing required fields: {missing!r}")
        output_fields = BRANCH_OUTPUT_GRAIN_FIELDS[branch]
        missing_output = [field_name for field_name in output_fields if field_name not in output_key]
        if missing_output:
            raise ValueError(f"Regime Production planning unit output_grain_key missing fields: {missing_output!r}")
        if output_key.get("timestamp") in (None, ""):
            raise ValueError("Regime Production planning unit output_grain_key requires timestamp placeholder")
        if timestamp_plan.get("timestamps_materialized") is not False:
            raise ValueError("Regime Production planning units cannot materialize timestamp rows")
        model_state = self.model_state_definition
        if isinstance(model_state, RegimeProductionModelStateDefinition):
            model_state = model_state.as_dict()
        elif model_state is not None:
            model_state = to_jsonable(dict(model_state))
        normalized_lineage = self.normalized_lineage
        if isinstance(normalized_lineage, RegimeProductionNormalizedLineage):
            normalized_lineage = normalized_lineage.as_dict()
        elif normalized_lineage is not None:
            normalized_lineage = to_jsonable(dict(normalized_lineage))
        object.__setattr__(self, "branch", branch)
        object.__setattr__(self, "target_key", target_key)
        object.__setattr__(self, "output_grain_key", output_key)
        object.__setattr__(self, "timestamp_plan", timestamp_plan)
        object.__setattr__(self, "planning_status", status)
        object.__setattr__(self, "profile_id", None if self.profile_id is None else str(self.profile_id))
        object.__setattr__(self, "profile_version", None if self.profile_version is None else str(self.profile_version))
        object.__setattr__(self, "profile_artifact_path", None if self.profile_artifact_path is None else str(self.profile_artifact_path))
        object.__setattr__(self, "profile_artifact_hash", None if self.profile_artifact_hash is None else str(self.profile_artifact_hash))
        object.__setattr__(self, "model_state_definition", model_state)
        object.__setattr__(self, "method_metadata", to_jsonable(dict(self.method_metadata)))
        object.__setattr__(self, "health_metadata", to_jsonable(dict(self.health_metadata)))
        object.__setattr__(self, "local_reselection_metadata", to_jsonable(dict(self.local_reselection_metadata)))
        object.__setattr__(self, "normalized_lineage", normalized_lineage)
        object.__setattr__(self, "relationship_input_checks", tuple(to_jsonable(dict(item)) for item in self.relationship_input_checks))
        object.__setattr__(self, "reason_codes", tuple(dict.fromkeys(str(reason) for reason in self.reason_codes)))
        object.__setattr__(self, "warnings", tuple(dict.fromkeys(str(warning) for warning in self.warnings)))

    @property
    def unit_id(self) -> str:
        target = "|".join(str(self.target_key.get(field_name)) for field_name in BRANCH_TARGET_KEY_FIELDS[self.branch])
        return f"{self.branch}|{target}"

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": REGIME_PRODUCTION_PLANNER_SCHEMA_VERSION,
            "artifact_kind": REGIME_PRODUCTION_PLANNING_UNIT_ARTIFACT_KIND,
            "branch": self.branch,
            "unit_id": self.unit_id,
            "target_key": to_jsonable(dict(self.target_key)),
            "output_grain_key": to_jsonable(dict(self.output_grain_key)),
            "timestamp_plan": to_jsonable(dict(self.timestamp_plan)),
            "planning_status": self.planning_status,
            "profile_id": self.profile_id,
            "profile_version": self.profile_version,
            "profile_artifact_path": self.profile_artifact_path,
            "profile_artifact_hash": self.profile_artifact_hash,
            "model_state_definition": to_jsonable(dict(self.model_state_definition or {})) if self.model_state_definition is not None else None,
            "method_metadata": to_jsonable(dict(self.method_metadata)),
            "health_metadata": to_jsonable(dict(self.health_metadata)),
            "local_reselection_metadata": to_jsonable(dict(self.local_reselection_metadata)),
            "normalized_lineage": to_jsonable(dict(self.normalized_lineage or {})) if self.normalized_lineage is not None else None,
            "relationship_input_checks": [to_jsonable(dict(item)) for item in self.relationship_input_checks],
            "reason_codes": list(self.reason_codes),
            "warnings": list(self.warnings),
            "label_emitted": False,
            "production_label_record": None,
            "production_approved": False,
            "production_writer_enabled": False,
            "production_labels_written": False,
            "production_outputs_written": False,
            "canonical_production_state_outputs_written": False,
        }


@dataclass(frozen=True)
class RegimeProductionNoWritePlan:
    branch: str
    artifact_path: str | Path
    profile_artifact_hash: str
    consumer_validation: Mapping[str, Any]
    planner_contract: RegimeProductionPlannerContract | Mapping[str, Any]
    shared_dry_run_plan: Mapping[str, Any]
    planning_units: Sequence[RegimeProductionPlanningUnit]
    telemetry: Mapping[str, Any]
    job_matrix: Mapping[str, Any] = field(default_factory=dict)
    relationship_input_checks: Sequence[Mapping[str, Any]] = ()
    warnings: Sequence[str] = ()
    safety_status: str = REGIME_PRODUCTION_STATUS_READY_FOR_DRY_CONSUMPTION

    def __post_init__(self) -> None:
        branch = _branch_name(self.branch)
        units = tuple(self.planning_units)
        if any(unit.branch != branch for unit in units):
            raise ValueError("Regime Production no-write plan units must match plan branch")
        contract = self.planner_contract.as_dict() if isinstance(self.planner_contract, RegimeProductionPlannerContract) else to_jsonable(dict(self.planner_contract))
        if contract.get("production_writer_enabled") or contract.get("production_outputs_written") or contract.get("production_labels_written"):
            raise ValueError("Regime Production no-write plan received an open production contract")
        dry_plan = to_jsonable(dict(self.shared_dry_run_plan))
        if dry_plan.get("writer_enabled") or dry_plan.get("production_labels") or dry_plan.get("production_outputs_written"):
            raise ValueError("Regime Production no-write plan received an open dry-run writer gate")
        job_matrix = to_jsonable(dict(self.job_matrix or {}))
        if job_matrix:
            _validate_no_write_job_matrix(job_matrix, branch=branch, expected_unit_count=len(units))
        status = _text(self.safety_status, field_name="safety_status")
        if status not in {REGIME_PRODUCTION_STATUS_READY_FOR_DRY_CONSUMPTION, REGIME_PRODUCTION_STATUS_BLOCKED}:
            raise ValueError(f"Unsupported Regime Production no-write safety_status: {status!r}")
        lineage_blocked_units = tuple(
            unit.unit_id
            for unit in units
            if unit.planning_status == REGIME_PRODUCTION_PLANNING_UNIT_STATUS_SELECTED
            and not dict(unit.normalized_lineage or {}).get("passed")
        )
        clamp_blocked_units = tuple(
            unit.unit_id
            for unit in units
            if unit.planning_status == REGIME_PRODUCTION_PLANNING_UNIT_STATUS_SELECTED
            and _selected_unit_clamp_blocked(unit)
        )
        warnings = tuple(dict.fromkeys(str(warning) for warning in self.warnings))
        if lineage_blocked_units:
            status = REGIME_PRODUCTION_STATUS_BLOCKED
            warnings = tuple(
                dict.fromkeys(
                    (
                        *warnings,
                        "selected_unit_normalized_lineage_not_traceable",
                    )
                )
            )
        if clamp_blocked_units:
            status = REGIME_PRODUCTION_STATUS_BLOCKED
            warnings = tuple(
                dict.fromkeys(
                    (
                        *warnings,
                        "selected_unit_clamp_range_not_ready",
                    )
                )
            )
        object.__setattr__(self, "branch", branch)
        object.__setattr__(self, "artifact_path", str(self.artifact_path))
        object.__setattr__(self, "profile_artifact_hash", _text(self.profile_artifact_hash, field_name="profile_artifact_hash"))
        object.__setattr__(self, "consumer_validation", to_jsonable(dict(self.consumer_validation)))
        object.__setattr__(self, "planner_contract", contract)
        object.__setattr__(self, "shared_dry_run_plan", dry_plan)
        object.__setattr__(self, "planning_units", units)
        object.__setattr__(self, "telemetry", to_jsonable(dict(self.telemetry)))
        object.__setattr__(self, "job_matrix", job_matrix)
        object.__setattr__(self, "relationship_input_checks", tuple(to_jsonable(dict(item)) for item in self.relationship_input_checks))
        object.__setattr__(self, "warnings", warnings)
        object.__setattr__(self, "safety_status", status)

    @property
    def passed(self) -> bool:
        return self.safety_status == REGIME_PRODUCTION_STATUS_READY_FOR_DRY_CONSUMPTION

    def as_dict(self, *, include_units: bool = True) -> dict[str, Any]:
        unit_payloads = [unit.as_dict() for unit in self.planning_units] if include_units else []
        return {
            "schema_version": REGIME_PRODUCTION_PLANNER_SCHEMA_VERSION,
            "artifact_kind": REGIME_PRODUCTION_NO_WRITE_PLAN_ARTIFACT_KIND,
            "branch": self.branch,
            "status": self.safety_status,
            "passed": self.passed,
            "artifact_path": self.artifact_path,
            "profile_artifact_hash": self.profile_artifact_hash,
            "consumer_validation": to_jsonable(dict(self.consumer_validation)),
            "planner_contract": to_jsonable(dict(self.planner_contract)),
            "shared_dry_run_plan": to_jsonable(dict(self.shared_dry_run_plan)),
            "planning_unit_count": len(self.planning_units),
            "planning_units": unit_payloads,
            "planning_units_omitted": not include_units,
            "job_matrix": to_jsonable(dict(self.job_matrix)),
            "telemetry": to_jsonable(dict(self.telemetry)),
            "relationship_input_checks": [to_jsonable(dict(item)) for item in self.relationship_input_checks],
            "warnings": list(self.warnings),
            "parent_finalizer": {
                "mode": "single_no_write_finalizer",
                "dry_run_artifact_write_allowed": False,
                "production_write_allowed": False,
                "production_labels_written": False,
                "canonical_production_state_outputs_written": False,
            },
            "label_emitted": False,
            "production_approved": False,
            "production_writer_enabled": False,
            "production_labels_written": False,
            "production_outputs_written": False,
            "canonical_production_state_outputs_written": False,
            "production_promotion_performed": False,
            "production_writer_gates_fail_closed": True,
        }


def default_regime_production_output_grain(branch: str) -> RegimeProductionOutputGrainContract:
    branch_name = _branch_name(branch)
    return RegimeProductionOutputGrainContract(
        branch=branch_name,
        grain_fields=BRANCH_OUTPUT_GRAIN_FIELDS[branch_name],
    )


def validate_regime_production_output_grain(branch: str, grain_fields: Sequence[str]) -> RegimeProductionOutputGrainContract:
    return RegimeProductionOutputGrainContract(branch=branch, grain_fields=tuple(grain_fields))


def resolve_regime_production_clamp_policy(
    *,
    applies_to_branches: Sequence[str] = REGIME_PRODUCTION_BRANCHES,
) -> RegimeProductionClampPlanningPolicy:
    from src.forecasting.ml.shared.numeric_cohort_common import (
        CLAMP_START_MONTH,
        CLAMP_START_YEAR,
        DEFAULT_COHORT_WINDOW_MONTHS,
        DEFAULT_SEARCH_BACK_MONTHS,
    )

    branches = tuple(dict.fromkeys(_branch_name(branch) for branch in applies_to_branches))
    pathways = tuple(dict.fromkeys(BRANCH_CLAMP_PATHWAYS[branch] for branch in branches))
    core_clamp = RegimeClampPolicy(
        policy_id="regime_production_numeric_forecast_clamp_runtime_boundaries_v1",
        reason=(
            "Planner reuses Numeric Forecast common recent-window clamp defaults; "
            "runtime configuration must supply concrete output boundaries before any writes."
        ),
        applies_to_pathways=pathways,
    )
    return RegimeProductionClampPlanningPolicy(
        policy_id="regime_production_numeric_forecast_common_recent_window_v1",
        source_contract="numeric_forecast_common_recent_window",
        source_module="src.forecasting.ml.shared.numeric_cohort_common",
        core_clamp_contract=core_clamp,
        numeric_clamp_start_year=int(CLAMP_START_YEAR),
        numeric_clamp_start_month=int(CLAMP_START_MONTH),
        historical_output_months=int(DEFAULT_COHORT_WINDOW_MONTHS),
        required_lookback_months=int(DEFAULT_SEARCH_BACK_MONTHS),
        applies_to_branches=branches,
    )


def resolve_regime_production_refit_cadence(
    branch: str,
    *,
    interval_minutes: int | None = None,
    requested_cadence: str | None = None,
) -> RegimeProductionRefitCadenceContract:
    branch_name = _branch_name(branch)
    source = "regime_production_branch_default"
    interval = None if interval_minutes is None else int(interval_minutes)
    if requested_cadence is not None:
        cadence = _text(requested_cadence, field_name="requested_cadence").lower()
        source = "explicit_planner_contract_request"
    elif interval is not None:
        from src.forecasting.ml.shared.numeric_runner_common import default_refit_cadence_for_interval

        cadence = str(default_refit_cadence_for_interval(interval)).lower()
        source = "src.forecasting.ml.shared.numeric_runner_common.default_refit_cadence_for_interval"
    else:
        cadence = {
            REGIME_BRANCH_ASSET_STATE: REGIME_PRODUCTION_REFIT_CADENCE_BIWEEKLY,
            REGIME_BRANCH_MARKET_STATE: REGIME_PRODUCTION_REFIT_CADENCE_MONTHLY,
            REGIME_BRANCH_CROSS_ASSET_STATE: REGIME_PRODUCTION_REFIT_CADENCE_MONTHLY,
        }[branch_name]
    if cadence not in REGIME_PRODUCTION_REFIT_CADENCES:
        raise ValueError(f"Unsupported Regime Production refit cadence: {cadence!r}")
    return RegimeProductionRefitCadenceContract(
        branch=branch_name,
        refit_cadence_id=f"{branch_name}_{cadence}_calendar_refit_v1",
        cadence=cadence,
        cadence_source=source,
        interval_minutes=interval,
    )


def normalize_regime_production_label_status(value: object) -> str:
    text = str(value or "").strip().lower()
    aliases = {
        "": REGIME_PRODUCTION_LABEL_STATUS_INVALID_PROFILE,
        "valid": REGIME_PRODUCTION_LABEL_STATUS_VALID,
        "valid_labeled": REGIME_PRODUCTION_LABEL_STATUS_VALID,
        "valid_label": REGIME_PRODUCTION_LABEL_STATUS_VALID,
        "selected": REGIME_PRODUCTION_LABEL_STATUS_VALID,
        "low_confidence": REGIME_PRODUCTION_LABEL_STATUS_LOW_CONFIDENCE,
        "bar_low_confidence": REGIME_PRODUCTION_LABEL_STATUS_LOW_CONFIDENCE,
        "unknown": REGIME_PRODUCTION_LABEL_STATUS_UNKNOWN_NOISE,
        "noise": REGIME_PRODUCTION_LABEL_STATUS_UNKNOWN_NOISE,
        "unknown_noise": REGIME_PRODUCTION_LABEL_STATUS_UNKNOWN_NOISE,
        "unknown_or_noise": REGIME_PRODUCTION_LABEL_STATUS_UNKNOWN_NOISE,
        "bar_unknown_or_noise": REGIME_PRODUCTION_LABEL_STATUS_UNKNOWN_NOISE,
        "masked": REGIME_PRODUCTION_LABEL_STATUS_MASKED_UNAVAILABLE,
        "unavailable": REGIME_PRODUCTION_LABEL_STATUS_MASKED_UNAVAILABLE,
        "masked_unavailable": REGIME_PRODUCTION_LABEL_STATUS_MASKED_UNAVAILABLE,
        REGIME_PRODUCTION_REASON_INSUFFICIENT: REGIME_PRODUCTION_LABEL_STATUS_INSUFFICIENT,
        "insufficient_history": REGIME_PRODUCTION_LABEL_STATUS_INSUFFICIENT,
        "insufficient_rows": REGIME_PRODUCTION_LABEL_STATUS_INSUFFICIENT,
        REGIME_PRODUCTION_REASON_NOT_CLUSTERABLE: REGIME_PRODUCTION_LABEL_STATUS_NOT_CLUSTERABLE,
        "clusterability_filtered": REGIME_PRODUCTION_LABEL_STATUS_NOT_CLUSTERABLE,
        "non_clusterable": REGIME_PRODUCTION_LABEL_STATUS_NOT_CLUSTERABLE,
        "stale_relationship_input": REGIME_PRODUCTION_LABEL_STATUS_STALE_RELATIONSHIP_INPUT,
        "relationship_input_stale": REGIME_PRODUCTION_LABEL_STATUS_STALE_RELATIONSHIP_INPUT,
        "stale_relationship_snapshot": REGIME_PRODUCTION_LABEL_STATUS_STALE_RELATIONSHIP_INPUT,
        "stale_sidecar": REGIME_PRODUCTION_LABEL_STATUS_STALE_RELATIONSHIP_INPUT,
        REGIME_PRODUCTION_REASON_FAILED_HEALTH_GATE: REGIME_PRODUCTION_LABEL_STATUS_FAILED_PROFILE_HEALTH,
        "label_health_gate_failed": REGIME_PRODUCTION_LABEL_STATUS_FAILED_PROFILE_HEALTH,
        "failed_profile_health": REGIME_PRODUCTION_LABEL_STATUS_FAILED_PROFILE_HEALTH,
        REGIME_PRODUCTION_REASON_MISSING_INPUT: REGIME_PRODUCTION_LABEL_STATUS_MISSING_INPUT,
        "missing_features": REGIME_PRODUCTION_LABEL_STATUS_MISSING_INPUT,
        "missing_required_features": REGIME_PRODUCTION_LABEL_STATUS_MISSING_INPUT,
        "missing_required_family_fields": REGIME_PRODUCTION_LABEL_STATUS_MISSING_INPUT,
        REGIME_PRODUCTION_REASON_INVALID_PROFILE: REGIME_PRODUCTION_LABEL_STATUS_INVALID_PROFILE,
        "profile_invalid": REGIME_PRODUCTION_LABEL_STATUS_INVALID_PROFILE,
    }
    if text in aliases:
        return aliases[text]
    if text in REGIME_PRODUCTION_LABEL_STATUSES:
        return text
    return REGIME_PRODUCTION_LABEL_STATUS_INVALID_PROFILE


def validate_regime_production_planner_gates(
    branch: str,
    *,
    production_write_requested: bool = False,
    allow_production_writes: bool = False,
    requires_human_approval_before_production: bool = True,
):
    payload = {
        "production_approved": False,
        "production_writer_enabled": False,
        "production_labels_written": False,
        "production_outputs_written": False,
        "canonical_production_state_outputs_written": False,
        "requires_human_approval_before_production": bool(requires_human_approval_before_production),
    }
    return validate_regime_production_gate(
        payload,
        branch=branch,
        production_write_requested=bool(production_write_requested),
        allow_production_writes=bool(allow_production_writes),
        expected_requires_human_approval=bool(requires_human_approval_before_production),
    )


def build_regime_production_planner_contract(
    branch: str,
    *,
    interval_minutes: int | None = None,
    relationship_inputs: Sequence[RegimeProductionRelationshipInputContract] = (),
) -> RegimeProductionPlannerContract:
    branch_name = _branch_name(branch)
    return RegimeProductionPlannerContract(
        branch=branch_name,
        output_grain=default_regime_production_output_grain(branch_name),
        clamp_policy=resolve_regime_production_clamp_policy(applies_to_branches=(branch_name,)),
        refit_cadence=resolve_regime_production_refit_cadence(branch_name, interval_minutes=interval_minutes),
        relationship_inputs=tuple(relationship_inputs),
        requires_human_approval_before_production=branch_name != REGIME_BRANCH_MARKET_STATE,
    )


def _branch_name(value: object) -> str:
    text = _text(value, field_name="branch")
    aliases = {
        "asset": REGIME_BRANCH_ASSET_STATE,
        "asset-state": REGIME_BRANCH_ASSET_STATE,
        "asset_state_production": REGIME_BRANCH_ASSET_STATE,
        "market": REGIME_BRANCH_MARKET_STATE,
        "market-state": REGIME_BRANCH_MARKET_STATE,
        "market_state_production": REGIME_BRANCH_MARKET_STATE,
        "cross_asset": REGIME_BRANCH_CROSS_ASSET_STATE,
        "cross-asset-state": REGIME_BRANCH_CROSS_ASSET_STATE,
        "cross_asset_state_production": REGIME_BRANCH_CROSS_ASSET_STATE,
    }
    resolved = aliases.get(text, text)
    if resolved not in REGIME_PRODUCTION_BRANCHES:
        raise ValueError(f"Unsupported Regime Production branch: {value!r}")
    return resolved


def _text(value: object, *, field_name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"Regime Production planner {field_name} must be non-empty")
    return text


def _string_tuple(values: Sequence[object], *, field_name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValueError(f"Regime Production planner {field_name} must be a sequence")
    out = tuple(str(value).strip() for value in values if str(value).strip())
    if not out:
        raise ValueError(f"Regime Production planner {field_name} must be non-empty")
    return out


def _to_orderable(value: object, *, field_name: str) -> float:
    if isinstance(value, bool) or value is None:
        raise ValueError(f"Regime Production planner {field_name} must be a timestamp")
    try:
        return float(value)
    except Exception:
        pass
    from datetime import datetime

    text = _text(value, field_name=field_name)
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()
    except Exception as exc:
        raise ValueError(f"Regime Production planner {field_name} must be numeric or ISO datetime") from exc


def _validate_order(start: object, end: object, *, context: str) -> None:
    if _to_orderable(start, field_name=f"{context} start") > _to_orderable(end, field_name=f"{context} end"):
        raise ValueError(f"Regime Production planner {context} start must be <= end")


def _selected_unit_clamp_blocked(unit: RegimeProductionPlanningUnit) -> bool:
    timestamp_plan = dict(unit.timestamp_plan or {})
    normalized = dict(timestamp_plan.get("normalized_clamp_range") or {})
    if normalized and normalized.get("passed") is not True:
        return True
    relationship = dict(timestamp_plan.get("relationship_input_history_check") or {})
    if relationship and relationship.get("passed") is not True:
        return True
    return False


def _validate_no_write_job_matrix(job_matrix: Mapping[str, Any], *, branch: str, expected_unit_count: int) -> None:
    if job_matrix.get("artifact_kind") != "regime_production_job_matrix":
        raise ValueError("Regime Production no-write plan received an invalid job matrix artifact kind")
    if job_matrix.get("branch") != branch:
        raise ValueError("Regime Production no-write plan job matrix branch mismatch")
    if int(job_matrix.get("work_unit_count", -1)) != int(expected_unit_count):
        raise ValueError("Regime Production no-write plan job matrix work_unit_count mismatch")
    if int(job_matrix.get("writer_workers", 0)) != 1:
        raise ValueError("Regime Production no-write plan job matrix must enforce writer_workers=1")
    if job_matrix.get("workers_write_outputs") is not False:
        raise ValueError("Regime Production no-write plan job matrix cannot allow worker writes")
    parent_finalizer = dict(job_matrix.get("parent_finalizer") or {})
    if parent_finalizer.get("parent_single_finalizer") is not True:
        raise ValueError("Regime Production no-write plan job matrix requires parent_single_finalizer")
    if int(parent_finalizer.get("writer_workers", 0)) != 1:
        raise ValueError("Regime Production no-write plan job matrix parent writer_workers must equal 1")
    if parent_finalizer.get("production_write_allowed") or parent_finalizer.get("dry_run_artifact_write_allowed"):
        raise ValueError("Regime Production no-write plan job matrix finalizer write gates must stay closed")
    for field_name in (
        "label_rows_materialized",
        "production_labels_written",
        "production_outputs_written",
        "canonical_production_state_outputs_written",
    ):
        if job_matrix.get(field_name):
            raise ValueError(f"Regime Production no-write plan job matrix cannot set {field_name}")


__all__ = [
    "BRANCH_CLAMP_PATHWAYS",
    "BRANCH_OUTPUT_GRAIN_FIELDS",
    "BRANCH_TARGET_KEY_FIELDS",
    "MODEL_STATE_REQUIRED_FIELDS",
    "REGIME_PRODUCTION_CLAMP_POLICY_ARTIFACT_KIND",
    "REGIME_PRODUCTION_LABEL_STATUS_FAILED_PROFILE_HEALTH",
    "REGIME_PRODUCTION_LABEL_STATUS_INSUFFICIENT",
    "REGIME_PRODUCTION_LABEL_STATUS_INVALID_PROFILE",
    "REGIME_PRODUCTION_LABEL_STATUS_LOW_CONFIDENCE",
    "REGIME_PRODUCTION_LABEL_STATUS_MASKED_UNAVAILABLE",
    "REGIME_PRODUCTION_LABEL_STATUS_MISSING_INPUT",
    "REGIME_PRODUCTION_LABEL_STATUS_NOT_CLUSTERABLE",
    "REGIME_PRODUCTION_LABEL_STATUS_STALE_RELATIONSHIP_INPUT",
    "REGIME_PRODUCTION_LABEL_STATUS_UNKNOWN_NOISE",
    "REGIME_PRODUCTION_LABEL_STATUS_VALID",
    "REGIME_PRODUCTION_LABEL_STATUSES",
    "REGIME_PRODUCTION_MODEL_STATE_ARTIFACT_KIND",
    "REGIME_PRODUCTION_MODEL_STATE_STATUS_ACTIVE",
    "REGIME_PRODUCTION_MODEL_STATE_STATUS_BLOCKED",
    "REGIME_PRODUCTION_MODEL_STATE_STATUS_FAILED_PROFILE_HEALTH",
    "REGIME_PRODUCTION_MODEL_STATE_STATUS_INVALID_PROFILE",
    "REGIME_PRODUCTION_MODEL_STATE_STATUS_MISSING_INPUT",
    "REGIME_PRODUCTION_MODEL_STATE_STATUS_PLANNED",
    "REGIME_PRODUCTION_MODEL_STATE_STATUS_STALE",
    "REGIME_PRODUCTION_MODEL_STATE_STATUSES",
    "REGIME_PRODUCTION_NORMALIZED_LINEAGE_ARTIFACT_KIND",
    "REGIME_PRODUCTION_OUTPUT_GRAIN_ARTIFACT_KIND",
    "REGIME_PRODUCTION_NO_WRITE_PLAN_ARTIFACT_KIND",
    "REGIME_PRODUCTION_PLANNING_UNIT_ARTIFACT_KIND",
    "REGIME_PRODUCTION_PLANNING_UNIT_STATUS_DIAGNOSTIC_ONLY",
    "REGIME_PRODUCTION_PLANNING_UNIT_STATUS_INVALID_PROFILE",
    "REGIME_PRODUCTION_PLANNING_UNIT_STATUS_MASKED_UNAVAILABLE",
    "REGIME_PRODUCTION_PLANNING_UNIT_STATUS_MISSING_INPUT",
    "REGIME_PRODUCTION_PLANNING_UNIT_STATUS_SELECTED",
    "REGIME_PRODUCTION_PLANNING_UNIT_STATUS_SKIPPED",
    "REGIME_PRODUCTION_PLANNING_UNIT_STATUSES",
    "REGIME_PRODUCTION_PLANNER_ARTIFACT_KIND",
    "REGIME_PRODUCTION_PLANNER_SCHEMA_VERSION",
    "REGIME_PRODUCTION_REFIT_CADENCE_ARTIFACT_KIND",
    "REGIME_PRODUCTION_REFIT_CADENCE_BIWEEKLY",
    "REGIME_PRODUCTION_REFIT_CADENCE_MONTHLY",
    "REGIME_PRODUCTION_REFIT_CADENCE_WEEKLY",
    "REGIME_PRODUCTION_RELATIONSHIP_INPUT_ARTIFACT_KIND",
    "RegimeProductionClampPlanningPolicy",
    "RegimeProductionLabelStatusContract",
    "RegimeProductionModelStateDefinition",
    "RegimeProductionNormalizedLineage",
    "RegimeProductionNoWritePlan",
    "RegimeProductionOutputGrainContract",
    "RegimeProductionPlanningUnit",
    "RegimeProductionPlannerContract",
    "RegimeProductionRefitCadenceContract",
    "RegimeProductionRelationshipInputContract",
    "build_regime_production_planner_contract",
    "default_regime_production_output_grain",
    "normalize_regime_production_label_status",
    "resolve_regime_production_clamp_policy",
    "resolve_regime_production_refit_cadence",
    "validate_regime_production_output_grain",
    "validate_regime_production_planner_gates",
]
