from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from src.regimes.core.paths import resolve_project_path
from src.regimes.core.production_reuse_cache import RegimeProductionPlannerRunCache, source_tail_fingerprint
from src.regimes.core.serialization import to_jsonable
from src.regimes.core.test_branch_contracts import PRODUCTION_GATE_FIELDS


REGIME_PRODUCTION_CONSUMER_SCHEMA_VERSION = 1

REGIME_BRANCH_ASSET_STATE = "asset_state"
REGIME_BRANCH_MARKET_STATE = "market_state"
REGIME_BRANCH_CROSS_ASSET_STATE = "cross_asset_state"
REGIME_PRODUCTION_BRANCHES: tuple[str, ...] = (
    REGIME_BRANCH_ASSET_STATE,
    REGIME_BRANCH_MARKET_STATE,
    REGIME_BRANCH_CROSS_ASSET_STATE,
)

REGIME_PRODUCTION_STATUS_READY_FOR_DRY_CONSUMPTION = "ready_for_dry_consumption"
REGIME_PRODUCTION_STATUS_BLOCKED = "blocked"

REGIME_PRODUCTION_CELL_STATUS_SELECTED = "selected"
REGIME_PRODUCTION_CELL_STATUS_MASKED = "masked"
REGIME_PRODUCTION_CELL_STATUS_UNAVAILABLE = "unavailable"
REGIME_PRODUCTION_CELL_STATUS_SKIPPED = "skipped"
REGIME_PRODUCTION_CELL_STATUSES: tuple[str, ...] = (
    REGIME_PRODUCTION_CELL_STATUS_SELECTED,
    REGIME_PRODUCTION_CELL_STATUS_MASKED,
    REGIME_PRODUCTION_CELL_STATUS_UNAVAILABLE,
    REGIME_PRODUCTION_CELL_STATUS_SKIPPED,
)

REGIME_PRODUCTION_REASON_INSUFFICIENT = "insufficient"
REGIME_PRODUCTION_REASON_NOT_CLUSTERABLE = "not_clusterable"
REGIME_PRODUCTION_REASON_STALE_ARTIFACT = "stale_artifact"
REGIME_PRODUCTION_REASON_INVALID_PROFILE = "invalid_profile"
REGIME_PRODUCTION_REASON_MISSING_INPUT = "missing_input"
REGIME_PRODUCTION_REASON_FAILED_HEALTH_GATE = "failed_health_gate"
REGIME_PRODUCTION_MASK_REASON_CODES: tuple[str, ...] = (
    REGIME_PRODUCTION_REASON_INSUFFICIENT,
    REGIME_PRODUCTION_REASON_NOT_CLUSTERABLE,
    REGIME_PRODUCTION_REASON_STALE_ARTIFACT,
    REGIME_PRODUCTION_REASON_INVALID_PROFILE,
    REGIME_PRODUCTION_REASON_MISSING_INPUT,
    REGIME_PRODUCTION_REASON_FAILED_HEALTH_GATE,
)

REGIME_PRODUCTION_ACTIVE_ROOT_ENV = "PIPELINE_REGIME_PRODUCTION_ACTIVE_HANDOFF_ROOT"
REGIME_PRODUCTION_ACTIVE_INDEX_ENV = "PIPELINE_REGIME_PRODUCTION_ACTIVE_HANDOFF_INDEX"
DEFAULT_REGIME_PRODUCTION_ACTIVE_INDEX_PATH = Path("config/regimes/regime_production_active_selected_profiles.json")


@dataclass(frozen=True)
class RegimeProductionBranchPolicy:
    branch: str
    artifact_kinds: Sequence[str]
    profile_grain: str
    expected_output_grain: str
    active_filename: str
    manifest_path_env: str
    active_root_env: str
    active_artifact_glob: str
    schema_version_fields: Sequence[str] = ("schema_version",)
    schema_versions: Sequence[int] = (1,)
    schema_version_alias_fields: Mapping[str, Mapping[str, int]] = field(default_factory=dict)
    branch_schema_policy: str | None = None
    selected_records_key: str | None = "selected_profiles"
    diagnostic_records_key: str | None = None
    masked_records_key: str | None = "masked_or_skipped_cells"
    all_records_key: str | None = None
    selected_status_field: str | None = None
    selected_status_prefixes: Sequence[str] = ("selected",)
    selected_count_fields: Sequence[str] = ("selected_profile_count",)
    diagnostic_count_fields: Sequence[str] = ()
    masked_count_fields: Sequence[str] = ("masked_or_skipped_count", "masked_or_skipped_cell_count")
    skipped_count_fields: Sequence[str] = ()
    profile_count_fields: Sequence[str] = ("profile_count", "total_profile_count")
    expected_count_fields: Sequence[str] = ("expected_cell_count",)
    missing_count_fields: Sequence[str] = ("missing_cell_count",)
    grain_fields: Sequence[str] = ("profile_grain", "grain")
    require_profile_grain: bool = True
    branch_identity_fields: Mapping[str, str] = field(default_factory=dict)
    active_true_fields: Sequence[str] = ("active_handoff_artifact",)
    active_false_fields: Sequence[str] = ("not_active_handoff",)
    consumable_true_fields: Sequence[str] = ()
    single_active_field: str | None = "single_active_nonproduction_handoff_artifact"
    require_canonical_gate_field: bool = True
    expected_requires_human_approval: bool | None = True

    def __post_init__(self) -> None:
        branch = _non_empty_text(self.branch, field_name="branch")
        if branch not in REGIME_PRODUCTION_BRANCHES:
            raise ValueError(f"Unsupported Regime Production branch: {branch!r}")
        object.__setattr__(self, "branch", branch)
        for field_name in (
            "artifact_kinds",
            "schema_versions",
            "schema_version_fields",
            "selected_status_prefixes",
            "selected_count_fields",
            "diagnostic_count_fields",
            "masked_count_fields",
            "skipped_count_fields",
            "profile_count_fields",
            "expected_count_fields",
            "missing_count_fields",
            "grain_fields",
            "active_true_fields",
            "active_false_fields",
            "consumable_true_fields",
        ):
            value = getattr(self, field_name)
            object.__setattr__(self, field_name, tuple(value))
        object.__setattr__(self, "artifact_kinds", tuple(_non_empty_text(value, field_name="artifact_kind") for value in self.artifact_kinds))
        object.__setattr__(self, "schema_versions", tuple(int(value) for value in self.schema_versions))
        policy = self.branch_schema_policy or f"{branch}_selected_profile_manifest_policy_v1"
        object.__setattr__(self, "branch_schema_policy", _non_empty_text(policy, field_name="branch_schema_policy"))

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": REGIME_PRODUCTION_CONSUMER_SCHEMA_VERSION,
            "artifact_kind": "regime_production_branch_policy",
            "branch": self.branch,
            "artifact_kinds": list(self.artifact_kinds),
            "profile_grain": self.profile_grain,
            "expected_output_grain": self.expected_output_grain,
            "active_filename": self.active_filename,
            "manifest_path_env": self.manifest_path_env,
            "active_root_env": self.active_root_env,
            "active_artifact_glob": self.active_artifact_glob,
            "schema_version_fields": list(self.schema_version_fields),
            "schema_versions": list(self.schema_versions),
            "schema_version_alias_fields": to_jsonable(dict(self.schema_version_alias_fields)),
            "branch_schema_policy": self.branch_schema_policy,
            "production_write_allowed": False,
            "production_labels_allowed": False,
        }


@dataclass(frozen=True)
class RegimeProductionManifestVersion:
    branch: str
    manifest_schema_version: int | None
    raw_version_field: str | None = None
    raw_version_value: Any = None
    branch_schema_policy: str | None = None
    accepted_version_fields: Sequence[str] = ()

    @property
    def passed(self) -> bool:
        return self.manifest_schema_version is not None

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": REGIME_PRODUCTION_CONSUMER_SCHEMA_VERSION,
            "artifact_kind": "regime_production_manifest_version",
            "branch": self.branch,
            "manifest_schema_version": self.manifest_schema_version,
            "raw_version_field": self.raw_version_field,
            "raw_version_value": to_jsonable(self.raw_version_value),
            "branch_schema_policy": self.branch_schema_policy,
            "accepted_version_fields": list(self.accepted_version_fields),
        }


@dataclass(frozen=True)
class RegimeProductionBranchValidationContext:
    policy: RegimeProductionBranchPolicy
    artifact_path: Path | None
    manifest_version: RegimeProductionManifestVersion
    selected_records: Sequence[Mapping[str, Any]] = ()
    diagnostic_records: Sequence[Mapping[str, Any]] = ()
    masked_records: Sequence[Mapping[str, Any]] = ()
    skipped_records: Sequence[Mapping[str, Any]] = ()
    profile_records: Sequence[Mapping[str, Any]] = ()
    expected_cell_count: int = 0
    covered_cell_count: int = 0
    missing_cell_count: int = 0

    @property
    def branch(self) -> str:
        return self.policy.branch

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": REGIME_PRODUCTION_CONSUMER_SCHEMA_VERSION,
            "artifact_kind": "regime_production_branch_validation_context",
            "branch": self.branch,
            "artifact_path": str(self.artifact_path) if self.artifact_path is not None else None,
            "manifest_version": self.manifest_version.as_dict(),
            "selected_record_count": len(self.selected_records),
            "diagnostic_record_count": len(self.diagnostic_records),
            "masked_record_count": len(self.masked_records),
            "skipped_record_count": len(self.skipped_records),
            "profile_record_count": len(self.profile_records),
            "expected_cell_count": int(self.expected_cell_count),
            "covered_cell_count": int(self.covered_cell_count),
            "missing_cell_count": int(self.missing_cell_count),
        }


RegimeProductionBranchValidator = Callable[
    [Mapping[str, Any], RegimeProductionBranchValidationContext],
    Sequence[str],
]


@dataclass(frozen=True)
class RegimeProductionGateValidation:
    status: str
    branch: str
    reason_codes: Sequence[str] = ()
    production_consumption_allowed: bool = False
    production_write_allowed: bool = False
    writer_enabled: bool = False
    production_labels_allowed: bool = False

    @property
    def passed(self) -> bool:
        return self.status == REGIME_PRODUCTION_STATUS_READY_FOR_DRY_CONSUMPTION

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": REGIME_PRODUCTION_CONSUMER_SCHEMA_VERSION,
            "artifact_kind": "regime_production_gate_validation",
            "status": self.status,
            "passed": self.passed,
            "branch": self.branch,
            "reason_codes": list(self.reason_codes),
            "production_consumption_allowed": bool(self.production_consumption_allowed),
            "production_write_allowed": False,
            "writer_enabled": False,
            "production_labels_allowed": False,
            "production_writer_gates_fail_closed": True,
        }


@dataclass(frozen=True)
class RegimeProductionManifestValidation:
    status: str
    branch: str
    artifact_path: Path | None = None
    reason_codes: Sequence[str] = ()
    artifact_kind: str | None = None
    schema_version: int | None = None
    manifest_schema_version: int | None = None
    raw_version_field: str | None = None
    raw_version_value: Any = None
    branch_schema_policy: str | None = None
    manifest_version: RegimeProductionManifestVersion | None = None
    profile_grain: str | None = None
    expected_output_grain: str | None = None
    profile_count: int = 0
    selected_cell_count: int = 0
    diagnostic_cell_count: int = 0
    masked_unavailable_cell_count: int = 0
    skipped_cell_count: int = 0
    expected_cell_count: int = 0
    covered_cell_count: int = 0
    missing_cell_count: int = 0
    gate_validation: RegimeProductionGateValidation | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.status == REGIME_PRODUCTION_STATUS_READY_FOR_DRY_CONSUMPTION

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": REGIME_PRODUCTION_CONSUMER_SCHEMA_VERSION,
            "artifact_kind": "regime_production_manifest_validation",
            "status": self.status,
            "passed": self.passed,
            "branch": self.branch,
            "artifact_path": str(self.artifact_path) if self.artifact_path is not None else None,
            "reason_codes": list(self.reason_codes),
            "source_artifact_kind": self.artifact_kind,
            "source_schema_version": self.schema_version,
            "manifest_schema_version": self.manifest_schema_version,
            "raw_version_field": self.raw_version_field,
            "raw_version_value": to_jsonable(self.raw_version_value),
            "branch_schema_policy": self.branch_schema_policy,
            "manifest_version": self.manifest_version.as_dict() if self.manifest_version is not None else None,
            "profile_grain": self.profile_grain,
            "expected_output_grain": self.expected_output_grain,
            "profile_count": int(self.profile_count),
            "selected_cell_count": int(self.selected_cell_count),
            "diagnostic_cell_count": int(self.diagnostic_cell_count),
            "masked_unavailable_cell_count": int(self.masked_unavailable_cell_count),
            "skipped_cell_count": int(self.skipped_cell_count),
            "expected_cell_count": int(self.expected_cell_count),
            "covered_cell_count": int(self.covered_cell_count),
            "missing_cell_count": int(self.missing_cell_count),
            "gate_validation": self.gate_validation.as_dict() if self.gate_validation is not None else None,
            "metadata": to_jsonable(dict(self.metadata)),
            "production_write_allowed": False,
            "production_labels_written": False,
            "canonical_outputs_written": False,
        }


@dataclass(frozen=True)
class RegimeProductionArtifactResolution:
    status: str
    branch: str
    artifact_path: Path | None = None
    source: str | None = None
    reason_codes: Sequence[str] = ()
    validation: RegimeProductionManifestValidation | None = None
    manifest: Mapping[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.status == REGIME_PRODUCTION_STATUS_READY_FOR_DRY_CONSUMPTION

    def as_dict(self, *, include_manifest: bool = False) -> dict[str, Any]:
        payload = {
            "schema_version": REGIME_PRODUCTION_CONSUMER_SCHEMA_VERSION,
            "artifact_kind": "regime_production_artifact_resolution",
            "status": self.status,
            "passed": self.passed,
            "branch": self.branch,
            "artifact_path": str(self.artifact_path) if self.artifact_path is not None else None,
            "source": self.source,
            "reason_codes": list(self.reason_codes),
            "validation": self.validation.as_dict() if self.validation is not None else None,
            "production_write_allowed": False,
            "production_labels_written": False,
            "canonical_outputs_written": False,
        }
        if include_manifest:
            payload["manifest"] = to_jsonable(dict(self.manifest))
        return payload


@dataclass(frozen=True)
class RegimeProductionCellRecord:
    branch: str
    profile_grain: str
    status: str
    cell_key: Mapping[str, Any]
    profile_id: str | None = None
    reason_code: str | None = None
    reason: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        branch = _branch_name(self.branch)
        status = _non_empty_text(self.status, field_name="status")
        if status not in REGIME_PRODUCTION_CELL_STATUSES:
            raise ValueError(f"Unsupported Regime Production cell status: {status!r}")
        reason_code = None
        if status != REGIME_PRODUCTION_CELL_STATUS_SELECTED or self.reason_code is not None:
            reason_code = normalize_regime_production_mask_reason_code(self.reason_code)
        object.__setattr__(self, "branch", branch)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "cell_key", to_jsonable(dict(self.cell_key)))
        object.__setattr__(self, "reason_code", reason_code)
        object.__setattr__(self, "metadata", to_jsonable(dict(self.metadata)))

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": REGIME_PRODUCTION_CONSUMER_SCHEMA_VERSION,
            "artifact_kind": "regime_production_cell_record",
            "branch": self.branch,
            "profile_grain": self.profile_grain,
            "status": self.status,
            "cell_key": to_jsonable(dict(self.cell_key)),
            "profile_id": self.profile_id,
            "reason_code": self.reason_code,
            "reason": self.reason,
            "metadata": to_jsonable(dict(self.metadata)),
            "production_approved": False,
            "production_writer_enabled": False,
            "production_labels_written": False,
            "production_outputs_written": False,
            "canonical_production_state_outputs_written": False,
        }


@dataclass(frozen=True)
class RegimeProductionDryRunPlan:
    branch: str
    artifact_path: Path
    profile_grain: str
    expected_output_grain: str
    selected_cell_count: int
    masked_unavailable_cell_count: int
    expected_cell_count: int = 0
    diagnostic_cell_count: int = 0
    missing_cell_count: int = 0
    manifest_schema_version: int | None = None
    raw_version_field: str | None = None
    raw_version_value: Any = None
    branch_schema_policy: str | None = None
    planned_input_roots: Mapping[str, str] = field(default_factory=dict)
    planned_input_checks: Sequence[Mapping[str, Any]] = ()
    planned_output_root: str | None = None
    reason_codes: Sequence[str] = ()
    warnings: Sequence[str] = ()
    safety_status: str = REGIME_PRODUCTION_STATUS_READY_FOR_DRY_CONSUMPTION
    writer_enabled: bool = False
    production_labels: bool = False
    canonical_outputs_written: bool = False

    def __post_init__(self) -> None:
        if self.writer_enabled or self.production_labels or self.canonical_outputs_written:
            raise ValueError("Regime Production dry-run plan cannot enable writes, labels, or canonical outputs")

    @property
    def passed(self) -> bool:
        return self.safety_status == REGIME_PRODUCTION_STATUS_READY_FOR_DRY_CONSUMPTION

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": REGIME_PRODUCTION_CONSUMER_SCHEMA_VERSION,
            "artifact_kind": "regime_production_dry_run_plan",
            "branch": self.branch,
            "artifact_path": str(self.artifact_path),
            "profile_grain": self.profile_grain,
            "expected_output_grain": self.expected_output_grain,
            "manifest_schema_version": self.manifest_schema_version,
            "raw_version_field": self.raw_version_field,
            "raw_version_value": to_jsonable(self.raw_version_value),
            "branch_schema_policy": self.branch_schema_policy,
            "expected_cell_count": int(self.expected_cell_count),
            "selected_cell_count": int(self.selected_cell_count),
            "diagnostic_cell_count": int(self.diagnostic_cell_count),
            "masked_unavailable_cell_count": int(self.masked_unavailable_cell_count),
            "missing_cell_count": int(self.missing_cell_count),
            "planned_input_roots": dict(self.planned_input_roots),
            "planned_input_checks": [to_jsonable(dict(item)) for item in self.planned_input_checks],
            "planned_output_root": self.planned_output_root,
            "writer_enabled": False,
            "production_labels": False,
            "production_outputs_written": False,
            "canonical_production_state_outputs_written": False,
            "reason_codes": list(self.reason_codes),
            "warnings": list(self.warnings),
            "safety_status": self.safety_status,
            "safe_for_no_write_dry_consumption": self.passed,
        }


def default_regime_production_branch_policy(branch: str) -> RegimeProductionBranchPolicy:
    branch_name = _branch_name(branch)
    if branch_name == REGIME_BRANCH_ASSET_STATE:
        return RegimeProductionBranchPolicy(
            branch=branch_name,
            artifact_kinds=(
                "asset_state_test_selected_profiles_asset_specific_optuna_tuned_manifest",
                "asset_state_test_selected_profiles_asset_specific_manifest",
                "asset_state_test_selected_profiles_manifest",
            ),
            profile_grain="asset_id x axis x band",
            expected_output_grain="asset_id x axis x band x timestamp",
            active_filename="asset_state_selected_profiles.final_test.nonprod.json",
            manifest_path_env="PIPELINE_ASSET_STATE_SELECTED_PROFILE_MANIFEST",
            active_root_env="PIPELINE_ASSET_STATE_ACTIVE_HANDOFF_ROOT",
            active_artifact_glob="asset_state_selected_profiles*.nonprod.json",
            selected_records_key=None,
            masked_records_key=None,
            all_records_key="profiles",
            selected_status_field="selection_status",
            selected_status_prefixes=("selected",),
            selected_count_fields=("selected_profile_count",),
            skipped_count_fields=("skipped_or_filtered_profile_count",),
            require_profile_grain=False,
            branch_identity_fields={"selection_scope": "asset_state_final_test_branch"},
            active_true_fields=("active_nonproduction_test_branch_manifest", "production_handoff_artifact"),
            active_false_fields=(),
            single_active_field=None,
            require_canonical_gate_field=False,
            expected_requires_human_approval=True,
        )
    if branch_name == REGIME_BRANCH_MARKET_STATE:
        return RegimeProductionBranchPolicy(
            branch=branch_name,
            artifact_kinds=("market_state_test_branch_selected_profiles_nonprod",),
            profile_grain="market_axis x band",
            expected_output_grain="market_axis x band x timestamp",
            active_filename="market_state_selected_profiles.nonprod.json",
            manifest_path_env="PIPELINE_MARKET_STATE_SELECTED_PROFILE_MANIFEST",
            active_root_env="PIPELINE_MARKET_STATE_ACTIVE_HANDOFF_ROOT",
            active_artifact_glob="market_state_selected_profiles*.nonprod.json",
            selected_records_key="selected_profiles",
            masked_records_key="masked_or_skipped_cells",
            selected_count_fields=("selected_profile_count",),
            masked_count_fields=("masked_or_skipped_count",),
            branch_identity_fields={"selection_scope": "market_axis_band", "profile_grain": "market_axis x band"},
            active_true_fields=("active_handoff_artifact",),
            consumable_true_fields=("production_consumable",),
            require_canonical_gate_field=True,
            expected_requires_human_approval=False,
        )
    return RegimeProductionBranchPolicy(
        branch=branch_name,
        artifact_kinds=(
            "cross_asset_state_selected_profiles_selection_engine_nonprod",
            "cross_asset_state_selected_profiles_nonprod",
            "cross_asset_state_selected_profiles_mature_nonprod",
        ),
        profile_grain="asset_id x relationship_feature_family x band",
        expected_output_grain="asset_id x relationship_feature_family x band x timestamp",
        active_filename="cross_asset_state_selected_profiles.default_test_branch.nonprod.json",
        manifest_path_env="PIPELINE_CROSS_ASSET_STATE_SELECTED_PROFILE_MANIFEST",
        active_root_env="PIPELINE_CROSS_ASSET_STATE_ACTIVE_HANDOFF_ROOT",
        active_artifact_glob="cross_asset_state_selected_profiles*.nonprod.json",
        selected_records_key="selected_profiles",
        diagnostic_records_key="diagnostic_only_profiles",
        masked_records_key="masked_or_skipped_cells",
        all_records_key="profiles",
        selected_count_fields=("selected_model_facing_profile_count", "selected_profile_count"),
        diagnostic_count_fields=("diagnostic_only_profile_count",),
        masked_count_fields=("masked_or_skipped_cell_count", "masked_cell_count"),
        branch_identity_fields={"grain": "asset_id x relationship_feature_family x band"},
        active_true_fields=("active_handoff_artifact", "final_artifact"),
        active_false_fields=("not_active_handoff",),
        require_canonical_gate_field=True,
        expected_requires_human_approval=True,
        schema_version_alias_fields={"selection_engine_version": {"cross_asset_state_selection_engine_v1": 1}},
    )


def regime_production_branch_policies() -> dict[str, RegimeProductionBranchPolicy]:
    return {branch: default_regime_production_branch_policy(branch) for branch in REGIME_PRODUCTION_BRANCHES}


def resolve_active_selected_profile_artifact(
    branch: str,
    *,
    policy: RegimeProductionBranchPolicy | None = None,
    explicit_artifact_path: str | Path | None = None,
    env: Mapping[str, str] | None = None,
    project_root: str | Path | None = None,
    allow_explicit_artifact_override: bool = True,
    check_explicit_parent_ambiguity: bool = False,
    branch_validator: RegimeProductionBranchValidator | None = None,
    run_cache: RegimeProductionPlannerRunCache | None = None,
    cache_fingerprint: Mapping[str, Any] | str | None = None,
) -> RegimeProductionArtifactResolution:
    policy = policy or default_regime_production_branch_policy(branch)
    source_env = os.environ if env is None else env
    if explicit_artifact_path is not None and str(explicit_artifact_path).strip():
        if not allow_explicit_artifact_override:
            return _blocked_resolution(
                policy,
                artifact_path=Path(explicit_artifact_path),
                source="explicit_artifact_path",
                reasons=("explicit_artifact_override_not_allowed",),
            )
        path = resolve_project_path(explicit_artifact_path, project_root=project_root)
        if check_explicit_parent_ambiguity and path.parent.exists():
            candidates = _active_candidates_from_root(path.parent, policy)
            if len(candidates) > 1:
                return _blocked_resolution(
                    policy,
                    artifact_path=path.parent,
                    source="explicit_artifact_path",
                    reasons=("ambiguous_active_artifact",),
                )
        return _load_and_validate_candidate(
            path,
            policy,
            source="explicit_artifact_path",
            branch_validator=branch_validator,
            run_cache=run_cache,
            cache_fingerprint=cache_fingerprint,
        )

    configured_manifest = str(source_env.get(policy.manifest_path_env, "") or "").strip()
    if configured_manifest:
        return _load_and_validate_candidate(
            resolve_project_path(configured_manifest, project_root=project_root),
            policy,
            source=f"env.{policy.manifest_path_env}",
            branch_validator=branch_validator,
            run_cache=run_cache,
            cache_fingerprint=cache_fingerprint,
        )

    configured_root = str(source_env.get(policy.active_root_env, "") or "").strip()
    common_root = str(source_env.get(REGIME_PRODUCTION_ACTIVE_ROOT_ENV, "") or "").strip()
    if configured_root or common_root:
        root = resolve_project_path(configured_root or common_root, project_root=project_root)
        if common_root and not configured_root:
            root = root / policy.branch
        candidates = _active_candidates_from_root(root, policy)
        if len(candidates) > 1:
            return _blocked_resolution(
                policy,
                artifact_path=root,
                source=f"env.{policy.active_root_env if configured_root else REGIME_PRODUCTION_ACTIVE_ROOT_ENV}",
                reasons=("ambiguous_active_artifact",),
            )
        if not candidates:
            return _blocked_resolution(
                policy,
                artifact_path=root / policy.active_filename,
                source=f"env.{policy.active_root_env if configured_root else REGIME_PRODUCTION_ACTIVE_ROOT_ENV}",
                reasons=("manifest_missing",),
            )
        return _load_and_validate_candidate(
            candidates[0],
            policy,
            source=f"env.{policy.active_root_env if configured_root else REGIME_PRODUCTION_ACTIVE_ROOT_ENV}",
            branch_validator=branch_validator,
            run_cache=run_cache,
            cache_fingerprint=cache_fingerprint,
        )

    indexed = _candidate_from_active_index(policy, source_env=source_env, project_root=project_root)
    if indexed is not None:
        index_path, artifact_path, reasons = indexed
        if reasons:
            return _blocked_resolution(
                policy,
                artifact_path=index_path,
                source=f"active_index:{index_path}",
                reasons=tuple(reasons),
            )
        return _load_and_validate_candidate(
            artifact_path,
            policy,
            source=f"active_index:{index_path}",
            branch_validator=branch_validator,
            run_cache=run_cache,
            cache_fingerprint=cache_fingerprint,
        )

    return _blocked_resolution(policy, artifact_path=None, source=None, reasons=("active_artifact_not_configured",))


def validate_selected_profile_manifest_for_production(
    manifest: Mapping[str, Any],
    *,
    branch: str | None = None,
    policy: RegimeProductionBranchPolicy | None = None,
    artifact_path: str | Path | None = None,
    branch_validator: RegimeProductionBranchValidator | None = None,
) -> RegimeProductionManifestValidation:
    resolved_policy = policy or default_regime_production_branch_policy(_non_empty_text(branch, field_name="branch"))
    reasons: list[str] = []
    artifact_kind = str(manifest.get("artifact_kind") or "")
    if artifact_kind not in set(resolved_policy.artifact_kinds):
        reasons.append("artifact_kind_unsupported")
    manifest_version = normalize_regime_production_manifest_version(manifest, resolved_policy)
    if manifest_version.manifest_schema_version is None:
        reasons.append("schema_version_missing_or_invalid")
    elif manifest_version.manifest_schema_version not in set(int(value) for value in resolved_policy.schema_versions):
        reasons.append("schema_version_unsupported")

    profile_grain = _first_present_text(manifest, resolved_policy.grain_fields) or resolved_policy.profile_grain
    if resolved_policy.require_profile_grain and profile_grain != resolved_policy.profile_grain:
        reasons.append("profile_grain_invalid")
    for field_name, expected in resolved_policy.branch_identity_fields.items():
        if manifest.get(field_name) != expected:
            reasons.append(f"branch_identity_{field_name}_invalid")
    if artifact_path is not None and resolved_policy.single_active_field is not None:
        path_name = Path(artifact_path).name
        if manifest.get(resolved_policy.single_active_field) != path_name:
            reasons.append("single_active_nonproduction_handoff_artifact_invalid")
    for field_name in resolved_policy.active_true_fields:
        if manifest.get(field_name) is not True:
            reasons.append(f"{field_name}_missing_or_false")
    for field_name in resolved_policy.active_false_fields:
        if field_name in manifest and manifest.get(field_name) is not False:
            reasons.append(f"{field_name}_must_be_false")
    for field_name in resolved_policy.consumable_true_fields:
        if manifest.get(field_name) is not True:
            reasons.append(f"{field_name}_missing_or_false")
    reasons.extend(_stale_or_incomplete_reasons(manifest))

    gate = validate_regime_production_gate(
        manifest,
        branch=resolved_policy.branch,
        require_canonical=resolved_policy.require_canonical_gate_field,
        expected_requires_human_approval=resolved_policy.expected_requires_human_approval,
    )
    reasons.extend(f"gate:{reason}" for reason in gate.reason_codes)

    selected = _records_from_key(manifest, resolved_policy.selected_records_key, reasons)
    diagnostic = _records_from_key(manifest, resolved_policy.diagnostic_records_key, reasons)
    masked = _records_from_key(manifest, resolved_policy.masked_records_key, reasons)
    all_records = _records_from_key(manifest, resolved_policy.all_records_key, reasons)
    if not selected and all_records and resolved_policy.selected_status_field:
        selected = [
            row
            for row in all_records
            if any(str(row.get(resolved_policy.selected_status_field, "")).startswith(prefix) for prefix in resolved_policy.selected_status_prefixes)
        ]
    skipped = []
    if all_records and resolved_policy.selected_status_field:
        selected_ids = {id(row) for row in selected}
        skipped = [row for row in all_records if id(row) not in selected_ids]

    _check_declared_count(manifest, resolved_policy.selected_count_fields, len(selected), reasons, "selected_profile_count")
    _check_declared_count(manifest, resolved_policy.diagnostic_count_fields, len(diagnostic), reasons, "diagnostic_profile_count", required=False)
    _check_declared_count(manifest, resolved_policy.masked_count_fields, len(masked), reasons, "masked_or_skipped_count", required=False)
    _check_declared_count(manifest, resolved_policy.skipped_count_fields, len(skipped), reasons, "skipped_profile_count", required=False)
    profile_records = all_records or [*selected, *diagnostic, *masked, *skipped]
    _check_declared_count(manifest, resolved_policy.profile_count_fields, len(profile_records), reasons, "profile_count", required=False)

    selected_count = len(selected)
    diagnostic_count = len(diagnostic)
    masked_count = len(masked)
    skipped_count = len(skipped)
    covered_count = selected_count + diagnostic_count + masked_count + skipped_count
    expected_count = _first_present_int(manifest, resolved_policy.expected_count_fields)
    if expected_count is None:
        expected_count = len(profile_records) if profile_records else covered_count
    elif expected_count != covered_count and expected_count != len(profile_records):
        reasons.append("expected_cell_count_mismatch")
    missing_count = _first_present_int(manifest, resolved_policy.missing_count_fields) or 0
    if missing_count != 0:
        reasons.append("missing_cell_count_not_zero")

    context = RegimeProductionBranchValidationContext(
        policy=resolved_policy,
        artifact_path=Path(artifact_path) if artifact_path is not None else None,
        manifest_version=manifest_version,
        selected_records=tuple(selected),
        diagnostic_records=tuple(diagnostic),
        masked_records=tuple(masked),
        skipped_records=tuple(skipped),
        profile_records=tuple(profile_records),
        expected_cell_count=int(expected_count),
        covered_cell_count=covered_count,
        missing_cell_count=missing_count,
    )
    if branch_validator is not None:
        reasons.extend(str(reason) for reason in branch_validator(manifest, context))

    _validate_profile_record_fields(profile_records, reasons)
    status = REGIME_PRODUCTION_STATUS_BLOCKED if reasons else REGIME_PRODUCTION_STATUS_READY_FOR_DRY_CONSUMPTION
    return RegimeProductionManifestValidation(
        status=status,
        branch=resolved_policy.branch,
        artifact_path=Path(artifact_path) if artifact_path is not None else None,
        reason_codes=tuple(dict.fromkeys(reasons)),
        artifact_kind=artifact_kind or None,
        schema_version=manifest_version.manifest_schema_version,
        manifest_schema_version=manifest_version.manifest_schema_version,
        raw_version_field=manifest_version.raw_version_field,
        raw_version_value=manifest_version.raw_version_value,
        branch_schema_policy=manifest_version.branch_schema_policy,
        manifest_version=manifest_version,
        profile_grain=profile_grain,
        expected_output_grain=resolved_policy.expected_output_grain,
        profile_count=len(profile_records),
        selected_cell_count=selected_count,
        diagnostic_cell_count=diagnostic_count,
        masked_unavailable_cell_count=masked_count + skipped_count,
        skipped_cell_count=skipped_count,
        expected_cell_count=int(expected_count),
        covered_cell_count=covered_count,
        missing_cell_count=missing_count,
        gate_validation=gate,
        metadata={
            "active_filename": resolved_policy.active_filename,
            "selected_records_key": resolved_policy.selected_records_key,
            "masked_records_key": resolved_policy.masked_records_key,
            "all_records_key": resolved_policy.all_records_key,
            "schema_version_source": manifest_version.raw_version_field,
            "raw_version_field": manifest_version.raw_version_field,
            "raw_version_value": manifest_version.raw_version_value,
            "branch_schema_policy": manifest_version.branch_schema_policy,
            "branch_validator_hook_used": branch_validator is not None,
        },
    )


def validate_regime_production_gate(
    payload: Mapping[str, Any],
    *,
    branch: str,
    production_write_requested: bool = False,
    allow_production_writes: bool = False,
    require_canonical: bool = True,
    expected_requires_human_approval: bool | None = True,
) -> RegimeProductionGateValidation:
    branch_name = _branch_name(branch)
    reasons: list[str] = []
    fields = PRODUCTION_GATE_FIELDS if require_canonical else tuple(
        field for field in PRODUCTION_GATE_FIELDS if field != "canonical_production_state_outputs_written"
    )
    for field_name in fields:
        if field_name not in payload:
            reasons.append(f"{field_name}_missing")
            continue
        expected = expected_requires_human_approval if field_name == "requires_human_approval_before_production" else False
        if expected is not None and payload.get(field_name) is not bool(expected):
            reasons.append(f"{field_name}_invalid")
    if production_write_requested and not allow_production_writes:
        reasons.append("production_write_request_rejected")
    if allow_production_writes:
        reasons.append("production_write_enablement_not_supported_by_shared_spine")
    status = REGIME_PRODUCTION_STATUS_BLOCKED if reasons else REGIME_PRODUCTION_STATUS_READY_FOR_DRY_CONSUMPTION
    return RegimeProductionGateValidation(
        status=status,
        branch=branch_name,
        reason_codes=tuple(dict.fromkeys(reasons)),
        production_consumption_allowed=status == REGIME_PRODUCTION_STATUS_READY_FOR_DRY_CONSUMPTION,
        production_write_allowed=False,
        writer_enabled=False,
        production_labels_allowed=False,
    )


def normalize_regime_production_mask_reason_code(reason_code: object) -> str:
    text = str(reason_code or "").strip().lower()
    aliases = {
        "": REGIME_PRODUCTION_REASON_INVALID_PROFILE,
        "required_features_missing": REGIME_PRODUCTION_REASON_MISSING_INPUT,
        "missing_required_features": REGIME_PRODUCTION_REASON_MISSING_INPUT,
        "missing_features": REGIME_PRODUCTION_REASON_MISSING_INPUT,
        "missing_input": REGIME_PRODUCTION_REASON_MISSING_INPUT,
        "missing_required_family_fields": REGIME_PRODUCTION_REASON_MISSING_INPUT,
        "insufficient_rows": REGIME_PRODUCTION_REASON_INSUFFICIENT,
        "insufficient_history": REGIME_PRODUCTION_REASON_INSUFFICIENT,
        "insufficient_coverage": REGIME_PRODUCTION_REASON_INSUFFICIENT,
        "insufficient_finite_share": REGIME_PRODUCTION_REASON_INSUFFICIENT,
        "clusterability_filtered": REGIME_PRODUCTION_REASON_NOT_CLUSTERABLE,
        "non_clusterable": REGIME_PRODUCTION_REASON_NOT_CLUSTERABLE,
        "not_clusterable": REGIME_PRODUCTION_REASON_NOT_CLUSTERABLE,
        "low_variance_near_flat": REGIME_PRODUCTION_REASON_NOT_CLUSTERABLE,
        "stale_or_invalid_source": REGIME_PRODUCTION_REASON_STALE_ARTIFACT,
        "source_invalid": REGIME_PRODUCTION_REASON_STALE_ARTIFACT,
        "stale_artifact": REGIME_PRODUCTION_REASON_STALE_ARTIFACT,
        "invalid_profile": REGIME_PRODUCTION_REASON_INVALID_PROFILE,
        "profile_invalid": REGIME_PRODUCTION_REASON_INVALID_PROFILE,
        "failed_health_gate": REGIME_PRODUCTION_REASON_FAILED_HEALTH_GATE,
        "label_health_gate_failed": REGIME_PRODUCTION_REASON_FAILED_HEALTH_GATE,
    }
    return aliases.get(text, text if text in REGIME_PRODUCTION_MASK_REASON_CODES else REGIME_PRODUCTION_REASON_INVALID_PROFILE)


def production_cell_record(
    *,
    branch: str,
    profile_grain: str,
    status: str,
    cell_key: Mapping[str, Any],
    profile_id: str | None = None,
    reason_code: str | None = None,
    reason: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> RegimeProductionCellRecord:
    return RegimeProductionCellRecord(
        branch=branch,
        profile_grain=profile_grain,
        status=status,
        cell_key=cell_key,
        profile_id=profile_id,
        reason_code=reason_code,
        reason=reason,
        metadata=dict(metadata or {}),
    )


def build_regime_production_dry_run_plan(
    validation: RegimeProductionManifestValidation,
    *,
    planned_input_roots: Mapping[str, str | Path] | None = None,
    planned_input_checks: Sequence[Mapping[str, Any]] | None = None,
    planned_output_root: str | Path | None = None,
    project_root: str | Path | None = None,
    warnings: Sequence[str] = (),
) -> RegimeProductionDryRunPlan:
    if validation.artifact_path is None:
        raise ValueError("Regime Production dry-run plan requires a resolved artifact path")
    input_roots: dict[str, str] = {}
    for name, raw_path in dict(planned_input_roots or {}).items():
        if not str(name).strip():
            raise ValueError("Regime Production dry-run plan input root names must be non-empty")
        input_roots[str(name)] = str(resolve_project_path(raw_path, project_root=project_root))
    output = str(resolve_project_path(planned_output_root, project_root=project_root)) if planned_output_root is not None else None
    safety_status = REGIME_PRODUCTION_STATUS_READY_FOR_DRY_CONSUMPTION if validation.passed else REGIME_PRODUCTION_STATUS_BLOCKED
    return RegimeProductionDryRunPlan(
        branch=validation.branch,
        artifact_path=validation.artifact_path,
        profile_grain=validation.profile_grain or "",
        expected_output_grain=validation.expected_output_grain or "",
        selected_cell_count=validation.selected_cell_count,
        masked_unavailable_cell_count=validation.masked_unavailable_cell_count,
        expected_cell_count=validation.expected_cell_count,
        diagnostic_cell_count=validation.diagnostic_cell_count,
        missing_cell_count=validation.missing_cell_count,
        manifest_schema_version=validation.manifest_schema_version,
        raw_version_field=validation.raw_version_field,
        raw_version_value=validation.raw_version_value,
        branch_schema_policy=validation.branch_schema_policy,
        planned_input_roots=input_roots,
        planned_input_checks=tuple(dict(item) for item in (planned_input_checks or ())),
        planned_output_root=output,
        reason_codes=tuple(validation.reason_codes),
        warnings=tuple(warnings),
        safety_status=safety_status,
        writer_enabled=False,
        production_labels=False,
        canonical_outputs_written=False,
    )


def _active_candidates_from_root(root: Path, policy: RegimeProductionBranchPolicy) -> list[Path]:
    if root.is_file():
        return [root]
    if not root.exists() or not root.is_dir():
        return []
    candidates = sorted(path for path in root.glob(policy.active_artifact_glob) if path.is_file())
    exact = root / policy.active_filename
    if exact.is_file() and exact not in candidates:
        candidates.insert(0, exact)
    return candidates


def _candidate_from_active_index(
    policy: RegimeProductionBranchPolicy,
    *,
    source_env: Mapping[str, str],
    project_root: str | Path | None,
) -> tuple[Path, Path, tuple[str, ...]] | None:
    configured = str(source_env.get(REGIME_PRODUCTION_ACTIVE_INDEX_ENV, "") or "").strip()
    index_path = (
        resolve_project_path(configured, project_root=project_root)
        if configured
        else resolve_project_path(DEFAULT_REGIME_PRODUCTION_ACTIVE_INDEX_PATH, project_root=project_root)
    )
    if not index_path.exists():
        return None if not configured else (index_path, index_path, ("active_index_missing",))
    try:
        loaded = json.loads(index_path.read_text(encoding="utf-8"))
    except Exception:
        return index_path, index_path, ("active_index_malformed",)
    if not isinstance(loaded, Mapping):
        return index_path, index_path, ("active_index_malformed",)
    artifacts = loaded.get("active_selected_profile_artifacts")
    if not isinstance(artifacts, Mapping):
        return index_path, index_path, ("active_index_artifacts_missing",)
    raw_entry = artifacts.get(policy.branch)
    entries = raw_entry if isinstance(raw_entry, list) else [raw_entry] if raw_entry is not None else []
    entries = [entry for entry in entries if isinstance(entry, Mapping)]
    if not entries:
        return index_path, index_path, ("active_index_branch_missing",)
    active_entries = [entry for entry in entries if entry.get("active") is True]
    if len(active_entries) != 1:
        return index_path, index_path, ("active_index_branch_ambiguous",)
    entry = dict(active_entries[0])
    raw_path = str(entry.get("artifact_path") or "").strip()
    if not raw_path:
        return index_path, index_path, ("active_index_artifact_path_missing",)
    if Path(raw_path).is_absolute():
        return index_path, index_path, ("active_index_artifact_path_must_be_project_relative",)
    if entry.get("branch") not in (None, policy.branch):
        return index_path, index_path, ("active_index_branch_identity_invalid",)
    if entry.get("active_role") not in (None, "selected_profile_artifact"):
        return index_path, index_path, ("active_index_role_invalid",)
    return index_path, resolve_project_path(raw_path, project_root=project_root), ()


def _load_and_validate_candidate(
    path: Path,
    policy: RegimeProductionBranchPolicy,
    *,
    source: str,
    branch_validator: RegimeProductionBranchValidator | None = None,
    run_cache: RegimeProductionPlannerRunCache | None = None,
    cache_fingerprint: Mapping[str, Any] | str | None = None,
) -> RegimeProductionArtifactResolution:
    if not path.exists() or not path.is_file():
        return _blocked_resolution(policy, artifact_path=path, source=source, reasons=("manifest_missing",))
    if run_cache is not None:
        cached = run_cache.artifact_resolution(
            branch=policy.branch,
            path=path,
            source=source,
            config_fingerprint=cache_fingerprint,
        )
        if cached is not None:
            return cached
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(loaded, Mapping):
            raise ValueError("manifest root is not a JSON object")
        manifest = dict(loaded)
    except Exception as exc:
        return _blocked_resolution(
            policy,
            artifact_path=path,
            source=source,
            reasons=(f"manifest_malformed:{type(exc).__name__}",),
        )
    validation = validate_selected_profile_manifest_for_production(
        manifest,
        policy=policy,
        artifact_path=path,
        branch_validator=branch_validator,
    )
    status = REGIME_PRODUCTION_STATUS_READY_FOR_DRY_CONSUMPTION if validation.passed else REGIME_PRODUCTION_STATUS_BLOCKED
    resolution = RegimeProductionArtifactResolution(
        status=status,
        branch=policy.branch,
        artifact_path=path,
        source=source,
        reason_codes=tuple(validation.reason_codes),
        validation=validation,
        manifest=manifest if validation.passed else {},
    )
    if run_cache is not None:
        run_cache.put_artifact_resolution(
            resolution,
            branch=policy.branch,
            path=path,
            source=source,
            config_fingerprint=cache_fingerprint,
            source_tail_fingerprint=source_tail_fingerprint(manifest),
        )
    return resolution


def _blocked_resolution(
    policy: RegimeProductionBranchPolicy,
    *,
    artifact_path: Path | None,
    source: str | None,
    reasons: Sequence[str],
) -> RegimeProductionArtifactResolution:
    return RegimeProductionArtifactResolution(
        status=REGIME_PRODUCTION_STATUS_BLOCKED,
        branch=policy.branch,
        artifact_path=artifact_path,
        source=source,
        reason_codes=tuple(dict.fromkeys(str(reason) for reason in reasons)),
        validation=None,
        manifest={},
    )


def _records_from_key(manifest: Mapping[str, Any], key: str | None, reasons: list[str]) -> list[dict[str, Any]]:
    if key is None:
        return []
    if key not in manifest:
        return []
    records = manifest.get(key)
    if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
        reasons.append(f"{key}_not_sequence")
        return []
    return [dict(item) for item in records if isinstance(item, Mapping)]


def _check_declared_count(
    manifest: Mapping[str, Any],
    field_names: Sequence[str],
    actual: int,
    reasons: list[str],
    label: str,
    *,
    required: bool = True,
) -> None:
    if not field_names:
        return
    for field_name in field_names:
        if field_name not in manifest:
            continue
        value = _optional_int(manifest.get(field_name))
        if value is None:
            reasons.append(f"{label}_invalid")
        elif int(value) != int(actual):
            reasons.append(f"{label}_mismatch")
        return
    if required:
        reasons.append(f"{label}_missing")


def _validate_profile_record_fields(records: Sequence[Mapping[str, Any]], reasons: list[str]) -> None:
    for index, record in enumerate(records):
        if not str(record.get("profile_id") or "").strip():
            reasons.append(f"profile_{index}_profile_id_missing")
        for field_name in ("profile_version", "lineage_id"):
            if field_name in record and not str(record.get(field_name) or "").strip():
                reasons.append(f"profile_{index}_{field_name}_empty")
        source_present = "source_tail_ts" in record and record.get("source_tail_ts") not in (None, "")
        known_present = "known_at_ts" in record and record.get("known_at_ts") not in (None, "")
        if source_present != known_present:
            reasons.append(f"profile_{index}_lineage_ts_pair_incomplete")
        if source_present and known_present:
            source_tail = _to_orderable(record.get("source_tail_ts"))
            known_at = _to_orderable(record.get("known_at_ts"))
            if source_tail is None or known_at is None:
                reasons.append(f"profile_{index}_lineage_ts_invalid")
            elif source_tail > known_at:
                reasons.append(f"profile_{index}_source_tail_ts_after_known_at_ts")


def _stale_or_incomplete_reasons(manifest: Mapping[str, Any]) -> tuple[str, ...]:
    reasons: list[str] = []
    if manifest.get("stale_sandbox_manifest_used") is True:
        reasons.append("stale_sandbox_manifest_used")
    source_lineage = manifest.get("source_lineage")
    if isinstance(source_lineage, Mapping) and source_lineage.get("stale_sandbox_manifest_used") is True:
        reasons.append("stale_sandbox_source_lineage")
    if manifest.get("partial_artifact") is True:
        reasons.append("partial_artifact_not_active_handoff")
    if manifest.get("incomplete_artifact") is True:
        reasons.append("incomplete_artifact_not_active_handoff")
    if "combined_artifact_status" in manifest and str(manifest.get("combined_artifact_status") or "") != "complete":
        reasons.append("combined_artifact_status_not_complete")
    return tuple(reasons)


def _first_present_text(payload: Mapping[str, Any], fields: Sequence[str]) -> str | None:
    for field_name in fields:
        value = payload.get(field_name)
        if value not in (None, ""):
            return str(value)
    return None


def _first_present_int(payload: Mapping[str, Any], fields: Sequence[str]) -> int | None:
    for field_name in fields:
        if field_name not in payload:
            continue
        value = _optional_int(payload.get(field_name))
        if value is not None:
            return value
    return None


def normalize_regime_production_manifest_version(
    payload: Mapping[str, Any],
    policy: RegimeProductionBranchPolicy,
) -> RegimeProductionManifestVersion:
    accepted_fields = (*tuple(policy.schema_version_fields), *tuple(policy.schema_version_alias_fields.keys()))
    for field_name in policy.schema_version_fields:
        if field_name not in payload:
            continue
        raw_value = payload.get(field_name)
        direct = _optional_int(raw_value)
        return RegimeProductionManifestVersion(
            branch=policy.branch,
            manifest_schema_version=direct,
            raw_version_field=field_name,
            raw_version_value=raw_value,
            branch_schema_policy=policy.branch_schema_policy,
            accepted_version_fields=accepted_fields,
        )
    for field_name, aliases in policy.schema_version_alias_fields.items():
        if field_name in policy.schema_version_fields:
            continue
        raw_value = payload.get(field_name)
        if raw_value is None:
            continue
        resolved = aliases.get(str(raw_value))
        return RegimeProductionManifestVersion(
            branch=policy.branch,
            manifest_schema_version=None if resolved is None else int(resolved),
            raw_version_field=field_name,
            raw_version_value=raw_value,
            branch_schema_policy=policy.branch_schema_policy,
            accepted_version_fields=accepted_fields,
        )
    return RegimeProductionManifestVersion(
        branch=policy.branch,
        manifest_schema_version=None,
        raw_version_field=None,
        raw_version_value=None,
        branch_schema_policy=policy.branch_schema_policy,
        accepted_version_fields=accepted_fields,
    )


def _optional_int(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def _to_orderable(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _non_empty_text(value: object, *, field_name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"Regime Production {field_name} must be non-empty")
    return text


def _branch_name(value: str) -> str:
    text = _non_empty_text(value, field_name="branch")
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


__all__ = [
    "DEFAULT_REGIME_PRODUCTION_ACTIVE_INDEX_PATH",
    "REGIME_BRANCH_ASSET_STATE",
    "REGIME_BRANCH_CROSS_ASSET_STATE",
    "REGIME_BRANCH_MARKET_STATE",
    "REGIME_PRODUCTION_ACTIVE_INDEX_ENV",
    "REGIME_PRODUCTION_ACTIVE_ROOT_ENV",
    "REGIME_PRODUCTION_BRANCHES",
    "REGIME_PRODUCTION_CELL_STATUS_MASKED",
    "REGIME_PRODUCTION_CELL_STATUS_SELECTED",
    "REGIME_PRODUCTION_CELL_STATUS_SKIPPED",
    "REGIME_PRODUCTION_CELL_STATUS_UNAVAILABLE",
    "REGIME_PRODUCTION_MASK_REASON_CODES",
    "REGIME_PRODUCTION_REASON_FAILED_HEALTH_GATE",
    "REGIME_PRODUCTION_REASON_INSUFFICIENT",
    "REGIME_PRODUCTION_REASON_INVALID_PROFILE",
    "REGIME_PRODUCTION_REASON_MISSING_INPUT",
    "REGIME_PRODUCTION_REASON_NOT_CLUSTERABLE",
    "REGIME_PRODUCTION_REASON_STALE_ARTIFACT",
    "REGIME_PRODUCTION_STATUS_BLOCKED",
    "REGIME_PRODUCTION_STATUS_READY_FOR_DRY_CONSUMPTION",
    "RegimeProductionArtifactResolution",
    "RegimeProductionBranchValidationContext",
    "RegimeProductionBranchValidator",
    "RegimeProductionBranchPolicy",
    "RegimeProductionCellRecord",
    "RegimeProductionDryRunPlan",
    "RegimeProductionGateValidation",
    "RegimeProductionManifestValidation",
    "RegimeProductionManifestVersion",
    "RegimeProductionPlannerRunCache",
    "build_regime_production_dry_run_plan",
    "default_regime_production_branch_policy",
    "normalize_regime_production_manifest_version",
    "normalize_regime_production_mask_reason_code",
    "production_cell_record",
    "regime_production_branch_policies",
    "resolve_active_selected_profile_artifact",
    "validate_regime_production_gate",
    "validate_selected_profile_manifest_for_production",
]
