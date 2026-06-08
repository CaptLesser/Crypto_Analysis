from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.regimes.core.production_consumer import (
    REGIME_BRANCH_ASSET_STATE,
    REGIME_PRODUCTION_STATUS_READY_FOR_DRY_CONSUMPTION,
    RegimeProductionArtifactResolution,
    RegimeProductionBranchValidationContext,
    RegimeProductionPlannerRunCache,
    build_regime_production_dry_run_plan,
    default_regime_production_branch_policy,
    resolve_active_selected_profile_artifact,
)
from src.regimes.core.serialization import to_jsonable


ASSET_STATE_PRODUCTION_CONSUMER_SCHEMA_VERSION = 1
ASSET_STATE_CONSUMER_STATUS_READY_FOR_DRY_CONSUMPTION = "ready_for_dry_consumption"
ASSET_STATE_CONSUMER_STATUS_BLOCKED = "blocked"
ASSET_STATE_SELECTED_PROFILE_MANIFEST_ENV = "PIPELINE_ASSET_STATE_SELECTED_PROFILE_MANIFEST"


@dataclass(frozen=True)
class AssetStateProductionConsumerValidation:
    status: str
    manifest_path: Path
    sandbox_nonproduction_mode: bool = False
    production_mode_requested: bool = False
    reason_codes: Sequence[str] = ()
    expected_cell_count: int = 0
    selected_profile_count: int = 0
    skipped_or_filtered_count: int = 0
    covered_cell_count: int = 0
    production_write_allowed: bool = False
    canonical_outputs_written: bool = False
    normalized_manifest_version: Mapping[str, Any] = field(default_factory=dict)
    dry_run_plan: Mapping[str, Any] = field(default_factory=dict)
    shared_validation: Mapping[str, Any] = field(default_factory=dict)
    branch_validator_hook_used: bool = False
    manifest: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": ASSET_STATE_PRODUCTION_CONSUMER_SCHEMA_VERSION,
            "artifact_kind": "asset_state_production_consumer_validation",
            "status": self.status,
            "manifest_path": str(self.manifest_path),
            "sandbox_nonproduction_mode": bool(self.sandbox_nonproduction_mode),
            "production_mode_requested": bool(self.production_mode_requested),
            "reason_codes": list(self.reason_codes),
            "expected_cell_count": int(self.expected_cell_count),
            "selected_profile_count": int(self.selected_profile_count),
            "skipped_or_filtered_count": int(self.skipped_or_filtered_count),
            "covered_cell_count": int(self.covered_cell_count),
            "normalized_manifest_version": to_jsonable(dict(self.normalized_manifest_version)),
            "dry_run_plan": to_jsonable(dict(self.dry_run_plan)),
            "shared_validation": to_jsonable(dict(self.shared_validation)),
            "branch_validator_hook_used": bool(self.branch_validator_hook_used),
            "branch_local_resolution_deprecated": True,
            "production_write_allowed": False,
            "canonical_outputs_written": False,
            "production_labels_written": False,
            "production_promotion_performed": False,
            "never_runs_test_branch_from_production": True,
            "manifest": to_jsonable(dict(self.manifest)),
        }


def default_asset_state_selected_profiles_manifest_path(*, env: Mapping[str, str] | None = None) -> Path:
    source_env = os.environ if env is None else env
    configured = str(source_env.get(ASSET_STATE_SELECTED_PROFILE_MANIFEST_ENV, "")).strip()
    if configured:
        return Path(configured).expanduser()
    resolution = resolve_active_selected_profile_artifact(
        REGIME_BRANCH_ASSET_STATE,
        env=source_env,
        allow_explicit_artifact_override=False,
    )
    if resolution.artifact_path is not None:
        return resolution.artifact_path
    raise FileNotFoundError("Asset-State active selected-profile artifact is not configured")


def validate_asset_state_selected_profiles_for_consumption(
    manifest_path: str | Path,
    *,
    sandbox_nonproduction_mode: bool = False,
    production_mode_requested: bool = False,
    expected_cell_count: int | None = None,
    run_cache: RegimeProductionPlannerRunCache | None = None,
) -> AssetStateProductionConsumerValidation:
    path = Path(manifest_path)
    resolution = resolve_active_selected_profile_artifact(
        REGIME_BRANCH_ASSET_STATE,
        explicit_artifact_path=path,
        env={},
        branch_validator=_asset_state_branch_validator(expected_cell_count=expected_cell_count),
        run_cache=run_cache,
        cache_fingerprint={
            "branch": REGIME_BRANCH_ASSET_STATE,
            "expected_cell_count": expected_cell_count,
            "sandbox_nonproduction_mode": bool(sandbox_nonproduction_mode),
            "production_mode_requested": bool(production_mode_requested),
        },
    )
    return _from_shared_resolution(
        resolution,
        fallback_path=path,
        sandbox_nonproduction_mode=sandbox_nonproduction_mode,
        production_mode_requested=production_mode_requested,
    )


def validate_default_asset_state_selected_profiles_for_consumption(
    *,
    sandbox_nonproduction_mode: bool = False,
    production_mode_requested: bool = False,
    expected_cell_count: int | None = None,
    env: Mapping[str, str] | None = None,
    run_cache: RegimeProductionPlannerRunCache | None = None,
) -> AssetStateProductionConsumerValidation:
    resolution = resolve_active_selected_profile_artifact(
        REGIME_BRANCH_ASSET_STATE,
        env=env,
        allow_explicit_artifact_override=False,
        branch_validator=_asset_state_branch_validator(expected_cell_count=expected_cell_count),
        run_cache=run_cache,
        cache_fingerprint={
            "branch": REGIME_BRANCH_ASSET_STATE,
            "expected_cell_count": expected_cell_count,
            "sandbox_nonproduction_mode": bool(sandbox_nonproduction_mode),
            "production_mode_requested": bool(production_mode_requested),
            "env_manifest": str((env or {}).get(ASSET_STATE_SELECTED_PROFILE_MANIFEST_ENV, "")),
        },
    )
    return _from_shared_resolution(
        resolution,
        fallback_path=resolution.artifact_path or Path(default_regime_production_branch_policy(REGIME_BRANCH_ASSET_STATE).active_filename),
        sandbox_nonproduction_mode=sandbox_nonproduction_mode,
        production_mode_requested=production_mode_requested,
    )


def _from_shared_resolution(
    resolution: RegimeProductionArtifactResolution,
    *,
    fallback_path: Path,
    sandbox_nonproduction_mode: bool,
    production_mode_requested: bool,
) -> AssetStateProductionConsumerValidation:
    validation = resolution.validation
    manifest = dict(resolution.manifest)
    dry_run_plan: Mapping[str, Any] = {}
    if validation is not None and validation.artifact_path is not None:
        dry_run_plan = build_regime_production_dry_run_plan(validation).as_dict()
    normalized_version = validation.manifest_version.as_dict() if validation is not None and validation.manifest_version is not None else {}
    status = (
        ASSET_STATE_CONSUMER_STATUS_READY_FOR_DRY_CONSUMPTION
        if resolution.status == REGIME_PRODUCTION_STATUS_READY_FOR_DRY_CONSUMPTION
        else ASSET_STATE_CONSUMER_STATUS_BLOCKED
    )
    expose_manifest = bool(sandbox_nonproduction_mode) or status == ASSET_STATE_CONSUMER_STATUS_READY_FOR_DRY_CONSUMPTION
    return AssetStateProductionConsumerValidation(
        status=status,
        manifest_path=resolution.artifact_path or fallback_path,
        sandbox_nonproduction_mode=bool(sandbox_nonproduction_mode),
        production_mode_requested=bool(production_mode_requested),
        reason_codes=tuple(dict.fromkeys(str(reason) for reason in resolution.reason_codes)),
        expected_cell_count=validation.expected_cell_count if validation is not None else 0,
        selected_profile_count=validation.selected_cell_count if validation is not None else 0,
        skipped_or_filtered_count=validation.skipped_cell_count if validation is not None else 0,
        covered_cell_count=validation.covered_cell_count if validation is not None else 0,
        production_write_allowed=False,
        canonical_outputs_written=False,
        normalized_manifest_version=normalized_version,
        dry_run_plan=dry_run_plan,
        shared_validation=validation.as_dict() if validation is not None else {},
        branch_validator_hook_used=bool(validation and validation.metadata.get("branch_validator_hook_used")),
        manifest=manifest if expose_manifest else {},
    )


def _asset_state_branch_validator(*, expected_cell_count: int | None):
    def _validate(manifest: Mapping[str, Any], context: RegimeProductionBranchValidationContext) -> Sequence[str]:
        reasons: list[str] = []
        if expected_cell_count is not None and int(context.expected_cell_count) != int(expected_cell_count):
            reasons.append("asset_state_expected_cell_count_mismatch")
        if manifest.get("production_handoff_artifact_role") not in (None, "asset_state_profile_selection_manifest"):
            reasons.append("production_handoff_artifact_role_invalid")
        refit_safety = manifest.get("refit_label_safety")
        if refit_safety is not None and not isinstance(refit_safety, Mapping):
            reasons.append("refit_label_safety_invalid")
        for index, profile in enumerate(context.profile_records):
            if "window_profile_id" in profile and not str(profile.get("window_profile_id") or "").strip():
                reasons.append(f"profile_{index}_window_profile_id_empty")
            selected = str(profile.get("selection_status") or "").startswith("selected")
            status_label = "selected" if selected else "skipped_or_filtered"
            for field_name in ("source_tail_ts", "known_at_ts", "lineage_id", "profile_version"):
                if profile.get(field_name) in (None, ""):
                    reasons.append(f"asset_state_{status_label}_profile_{field_name}_missing")
            for field_name in ("refit_window_start", "refit_window_end"):
                if profile.get(field_name) in (None, ""):
                    reasons.append(f"asset_state_{status_label}_profile_{field_name}_missing")
            if not selected and profile.get("skipped_or_filtered_reason") in (None, ""):
                reasons.append("asset_state_skipped_or_filtered_profile_reason_missing")
        return tuple(dict.fromkeys(reasons))

    return _validate


__all__ = [
    "ASSET_STATE_CONSUMER_STATUS_BLOCKED",
    "ASSET_STATE_CONSUMER_STATUS_READY_FOR_DRY_CONSUMPTION",
    "ASSET_STATE_SELECTED_PROFILE_MANIFEST_ENV",
    "AssetStateProductionConsumerValidation",
    "default_asset_state_selected_profiles_manifest_path",
    "validate_asset_state_selected_profiles_for_consumption",
    "validate_default_asset_state_selected_profiles_for_consumption",
]
