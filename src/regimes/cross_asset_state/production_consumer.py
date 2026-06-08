from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.regimes.core.production_consumer import (
    REGIME_BRANCH_CROSS_ASSET_STATE,
    REGIME_PRODUCTION_STATUS_READY_FOR_DRY_CONSUMPTION,
    RegimeProductionArtifactResolution,
    RegimeProductionBranchValidationContext,
    RegimeProductionPlannerRunCache,
    build_regime_production_dry_run_plan,
    default_regime_production_branch_policy,
    resolve_active_selected_profile_artifact,
)
from src.regimes.core.root_resolution import (
    SOURCE_KIND_RELATIONSHIP_DISCOVERY,
    resolve_regime_production_sidecar_input_path,
)
from src.regimes.core.paths import resolve_project_root
from src.regimes.core.production_reuse_cache import relationship_manifest_fingerprint
from src.regimes.core.serialization import to_jsonable
from src.regimes.core.test_branch_contracts import PRODUCTION_GATE_FIELDS, validate_nonproduction_gate_flags
from src.regimes.cross_asset_state.profile_manifest import (
    CROSS_ASSET_STATE_SELECTED_PROFILE_MANIFEST_KIND,
    validate_cross_asset_state_profile_grain,
)
from src.regimes.cross_asset_state.execution_maturity import (
    CROSS_ASSET_STATE_DEFAULT_COMBINED_SCOPE,
    CROSS_ASSET_STATE_FAMILY_SHARD_MANIFEST_KIND,
    CROSS_ASSET_STATE_FAMILY_SHARD_SCOPE,
    CROSS_ASSET_STATE_HEARTBEAT_SCOPE,
    CROSS_ASSET_STATE_PROGRESS_HEARTBEAT_KIND,
    CROSS_ASSET_STATE_RUNTIME_TELEMETRY_EVENT_KIND,
    CROSS_ASSET_STATE_RUNTIME_TELEMETRY_SCOPE,
    SHARD_STATUS_COMPLETE,
    SHARD_STATUS_FAILED,
    SHARD_STATUS_INCOMPLETE,
    SHARD_STATUS_PENDING,
    SHARD_STATUS_RUNNING,
)


CROSS_ASSET_STATE_PRODUCTION_CONSUMER_SCHEMA_VERSION = 1
CROSS_ASSET_STATE_CONSUMER_STATUS_READY_FOR_DRY_CONSUMPTION = "ready_for_dry_consumption"
CROSS_ASSET_STATE_CONSUMER_STATUS_BLOCKED = "blocked"
CROSS_ASSET_STATE_SELECTED_PROFILE_SELECTION_ENGINE_MANIFEST_KIND = "cross_asset_state_selected_profiles_selection_engine_nonprod"
CROSS_ASSET_STATE_ACCEPTED_SELECTED_PROFILE_MANIFEST_KINDS: tuple[str, ...] = (
    CROSS_ASSET_STATE_SELECTED_PROFILE_MANIFEST_KIND,
    CROSS_ASSET_STATE_SELECTED_PROFILE_SELECTION_ENGINE_MANIFEST_KIND,
)
CROSS_ASSET_STATE_DEFAULT_SELECTED_PROFILES_FILENAME = "cross_asset_state_selected_profiles.default_test_branch.nonprod.json"
CROSS_ASSET_STATE_SELECTED_PROFILE_MANIFEST_ENV = "PIPELINE_CROSS_ASSET_STATE_SELECTED_PROFILE_MANIFEST"
CROSS_ASSET_STATE_ACTIVE_HANDOFF_ROOT = Path("_codex_artifacts") / "reports" / "cross_asset_state_active_handoff"
CROSS_ASSET_STATE_ACTIVE_SELECTED_PROFILES_PATH = (
    CROSS_ASSET_STATE_ACTIVE_HANDOFF_ROOT / CROSS_ASSET_STATE_DEFAULT_SELECTED_PROFILES_FILENAME
)

REQUIRED_ROW_METADATA_FIELDS: tuple[str, ...] = (
    "asset_id",
    "relationship_feature_family",
    "band",
    "feature_set_version",
    "support_definition_id",
    "support_size",
    "support_quality",
    "relationship_context_id",
    "source_tail_ts",
    "known_at_ts",
)
REQUIRED_MASK_METADATA_FIELDS: tuple[str, ...] = (*REQUIRED_ROW_METADATA_FIELDS, "mask_reason")


class CrossAssetStateProductionGateError(RuntimeError):
    pass


@dataclass(frozen=True)
class CrossAssetStateProductionConsumerValidation:
    status: str
    manifest_path: Path
    sandbox_nonproduction_mode: bool = False
    production_mode_requested: bool = False
    reason_codes: Sequence[str] = ()
    artifact_kind: str | None = None
    expected_cell_count: int = 0
    selected_model_facing_profile_count: int = 0
    diagnostic_only_profile_count: int = 0
    masked_or_skipped_cell_count: int = 0
    covered_cell_count: int = 0
    missing_cell_count: int = 0
    production_write_allowed: bool = False
    canonical_outputs_written: bool = False
    normalized_manifest_version: Mapping[str, Any] = field(default_factory=dict)
    dry_run_plan: Mapping[str, Any] = field(default_factory=dict)
    shared_validation: Mapping[str, Any] = field(default_factory=dict)
    relationship_input_checks: Sequence[Mapping[str, Any]] = ()
    branch_validator_hook_used: bool = False
    manifest: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": CROSS_ASSET_STATE_PRODUCTION_CONSUMER_SCHEMA_VERSION,
            "artifact_kind": "cross_asset_state_production_consumer_validation",
            "status": self.status,
            "manifest_path": str(self.manifest_path),
            "sandbox_nonproduction_mode": bool(self.sandbox_nonproduction_mode),
            "production_mode_requested": bool(self.production_mode_requested),
            "reason_codes": list(self.reason_codes),
            "source_artifact_kind": self.artifact_kind,
            "expected_cell_count": int(self.expected_cell_count),
            "selected_model_facing_profile_count": int(self.selected_model_facing_profile_count),
            "diagnostic_only_profile_count": int(self.diagnostic_only_profile_count),
            "masked_or_skipped_cell_count": int(self.masked_or_skipped_cell_count),
            "covered_cell_count": int(self.covered_cell_count),
            "missing_cell_count": int(self.missing_cell_count),
            "normalized_manifest_version": to_jsonable(dict(self.normalized_manifest_version)),
            "dry_run_plan": to_jsonable(dict(self.dry_run_plan)),
            "shared_validation": to_jsonable(dict(self.shared_validation)),
            "relationship_input_checks": [to_jsonable(dict(item)) for item in self.relationship_input_checks],
            "selected_profile_artifact_role": "single_active_selected_profile_artifact",
            "relationship_inputs_role": "time_indexed_data_inputs_not_selected_profile_artifacts",
            "branch_validator_hook_used": bool(self.branch_validator_hook_used),
            "branch_local_resolution_deprecated": True,
            "production_write_allowed": False,
            "canonical_outputs_written": False,
            "production_labels_written": False,
            "production_promotion_performed": False,
            "never_runs_test_branch_from_production": True,
            "manifest": to_jsonable(dict(self.manifest)),
        }


def default_cross_asset_state_selected_profiles_manifest_path(
    *,
    env: Mapping[str, str] | None = None,
) -> Path:
    """Return the configured active Cross-Asset-State selected-profile handoff path."""

    source_env = os.environ if env is None else env
    configured = str(source_env.get(CROSS_ASSET_STATE_SELECTED_PROFILE_MANIFEST_ENV, "")).strip()
    if configured:
        return Path(configured).expanduser()
    resolution = resolve_active_selected_profile_artifact(
        REGIME_BRANCH_CROSS_ASSET_STATE,
        env=source_env,
        allow_explicit_artifact_override=False,
    )
    if resolution.artifact_path is not None:
        return resolution.artifact_path
    raise FileNotFoundError("Cross-Asset-State active selected-profile artifact is not configured")


def validate_cross_asset_state_selected_profiles_for_consumption(
    manifest_path: str | Path,
    *,
    sandbox_nonproduction_mode: bool = False,
    production_mode_requested: bool = False,
    active_filename: str | None = None,
    run_cache: RegimeProductionPlannerRunCache | None = None,
) -> CrossAssetStateProductionConsumerValidation:
    path = Path(manifest_path)
    expected_active_filename = str(active_filename or path.name)
    policy = replace(
        default_regime_production_branch_policy(REGIME_BRANCH_CROSS_ASSET_STATE),
        active_filename=expected_active_filename,
    )
    resolution = resolve_active_selected_profile_artifact(
        REGIME_BRANCH_CROSS_ASSET_STATE,
        policy=policy,
        explicit_artifact_path=path,
        env={},
        check_explicit_parent_ambiguity=path.name == CROSS_ASSET_STATE_DEFAULT_SELECTED_PROFILES_FILENAME,
        branch_validator=_cross_asset_state_branch_validator(expected_active_filename=expected_active_filename),
        run_cache=run_cache,
        cache_fingerprint={
            "branch": REGIME_BRANCH_CROSS_ASSET_STATE,
            "expected_active_filename": expected_active_filename,
            "sandbox_nonproduction_mode": bool(sandbox_nonproduction_mode),
            "production_mode_requested": bool(production_mode_requested),
        },
    )
    return _from_shared_resolution(
        resolution,
        fallback_path=path,
        sandbox_nonproduction_mode=sandbox_nonproduction_mode,
        production_mode_requested=production_mode_requested,
        env={},
        run_cache=run_cache,
    )


def validate_default_cross_asset_state_selected_profiles_for_consumption(
    *,
    sandbox_nonproduction_mode: bool = False,
    production_mode_requested: bool = False,
    env: Mapping[str, str] | None = None,
    run_cache: RegimeProductionPlannerRunCache | None = None,
) -> CrossAssetStateProductionConsumerValidation:
    resolution = resolve_active_selected_profile_artifact(
        REGIME_BRANCH_CROSS_ASSET_STATE,
        env=env,
        allow_explicit_artifact_override=False,
        branch_validator=_cross_asset_state_branch_validator(
            expected_active_filename=CROSS_ASSET_STATE_DEFAULT_SELECTED_PROFILES_FILENAME,
        ),
        run_cache=run_cache,
        cache_fingerprint={
            "branch": REGIME_BRANCH_CROSS_ASSET_STATE,
            "expected_active_filename": CROSS_ASSET_STATE_DEFAULT_SELECTED_PROFILES_FILENAME,
            "sandbox_nonproduction_mode": bool(sandbox_nonproduction_mode),
            "production_mode_requested": bool(production_mode_requested),
            "env_manifest": str((env or {}).get(CROSS_ASSET_STATE_SELECTED_PROFILE_MANIFEST_ENV, "")),
        },
    )
    return _from_shared_resolution(
        resolution,
        fallback_path=resolution.artifact_path
        or Path(default_regime_production_branch_policy(REGIME_BRANCH_CROSS_ASSET_STATE).active_filename),
        sandbox_nonproduction_mode=sandbox_nonproduction_mode,
        production_mode_requested=production_mode_requested,
        env=env,
        run_cache=run_cache,
    )


def _from_shared_resolution(
    resolution: RegimeProductionArtifactResolution,
    *,
    fallback_path: Path,
    sandbox_nonproduction_mode: bool,
    production_mode_requested: bool,
    env: Mapping[str, str] | None,
    run_cache: RegimeProductionPlannerRunCache | None = None,
) -> CrossAssetStateProductionConsumerValidation:
    validation = resolution.validation
    manifest = dict(resolution.manifest)
    source_env = os.environ if env is None else env
    relationship_checks = _relationship_input_checks(manifest, env=source_env, run_cache=run_cache)
    planned_input_roots = {
        str(item["name"]): str(item["path"])
        for item in relationship_checks
        if item.get("path") not in (None, "")
    }
    warnings = [
        "cross_asset_relationship_inputs_are_time_indexed_data_inputs_not_selected_profile_artifacts",
        "relationship_discovery_execution_not_performed",
    ]
    warnings.extend(str(item["reason_code"]) for item in relationship_checks if item.get("status") != "available" and item.get("reason_code"))
    dry_run_plan: Mapping[str, Any] = {}
    if validation is not None and validation.artifact_path is not None:
        dry_run_plan = build_regime_production_dry_run_plan(
            validation,
            planned_input_roots=planned_input_roots,
            planned_input_checks=relationship_checks,
            warnings=warnings,
        ).as_dict()
    normalized_version = validation.manifest_version.as_dict() if validation is not None and validation.manifest_version is not None else {}
    status = (
        CROSS_ASSET_STATE_CONSUMER_STATUS_READY_FOR_DRY_CONSUMPTION
        if resolution.status == REGIME_PRODUCTION_STATUS_READY_FOR_DRY_CONSUMPTION
        else CROSS_ASSET_STATE_CONSUMER_STATUS_BLOCKED
    )
    expose_manifest = bool(sandbox_nonproduction_mode) or status == CROSS_ASSET_STATE_CONSUMER_STATUS_READY_FOR_DRY_CONSUMPTION
    return CrossAssetStateProductionConsumerValidation(
        status=status,
        manifest_path=resolution.artifact_path or fallback_path,
        sandbox_nonproduction_mode=bool(sandbox_nonproduction_mode),
        production_mode_requested=bool(production_mode_requested),
        reason_codes=_compat_cross_reason_codes(resolution.reason_codes),
        artifact_kind=validation.artifact_kind if validation is not None else None,
        expected_cell_count=validation.expected_cell_count if validation is not None else 0,
        selected_model_facing_profile_count=validation.selected_cell_count if validation is not None else 0,
        diagnostic_only_profile_count=validation.diagnostic_cell_count if validation is not None else 0,
        masked_or_skipped_cell_count=validation.masked_unavailable_cell_count if validation is not None else 0,
        covered_cell_count=validation.covered_cell_count if validation is not None else 0,
        missing_cell_count=validation.missing_cell_count if validation is not None else 0,
        production_write_allowed=False,
        canonical_outputs_written=False,
        normalized_manifest_version=normalized_version,
        dry_run_plan=dry_run_plan,
        shared_validation=validation.as_dict() if validation is not None else {},
        relationship_input_checks=tuple(relationship_checks),
        branch_validator_hook_used=bool(validation and validation.metadata.get("branch_validator_hook_used")),
        manifest=manifest if expose_manifest else {},
    )


def _cross_asset_state_branch_validator(*, expected_active_filename: str):
    def _validate(manifest: Mapping[str, Any], context: RegimeProductionBranchValidationContext) -> Sequence[str]:
        reasons: list[str] = []
        artifact_kind = str(manifest.get("artifact_kind") or "")
        if artifact_kind not in CROSS_ASSET_STATE_ACCEPTED_SELECTED_PROFILE_MANIFEST_KINDS:
            reasons.append("artifact_kind_invalid")
        reasons.extend(_partial_artifact_rejection_reasons(manifest))
        if (
            manifest.get("artifact_scope") in {CROSS_ASSET_STATE_FAMILY_SHARD_SCOPE, "execution_shard"}
            or manifest.get("active_handoff_artifact") is not True
            or manifest.get("not_active_handoff") is not False
        ):
            reasons.append("manifest_not_active_handoff")
        if context.artifact_path is not None and context.artifact_path.name != expected_active_filename:
            reasons.append("unexpected_selected_manifest_filename")
        if manifest.get("single_active_nonproduction_handoff_artifact") != expected_active_filename:
            reasons.append("single_active_nonproduction_handoff_artifact_invalid")
        if manifest.get("stale_sandbox_manifest_used") is not False:
            reasons.append("stale_sandbox_manifest_used_invalid")
        source_lineage = manifest.get("source_lineage")
        if isinstance(source_lineage, Mapping) and source_lineage.get("stale_sandbox_manifest_used") is True:
            reasons.append("stale_sandbox_source_lineage_marked_active")
        reasons.extend(
            validate_nonproduction_gate_flags(
                manifest,
                require_canonical=True,
                expected_requires_human_approval=True,
                prefix="manifest",
            )
        )

        selected = [dict(item) for item in context.selected_records]
        diagnostic = [dict(item) for item in context.diagnostic_records]
        masked = [dict(item) for item in context.masked_records]
        all_records = [dict(item) for item in context.profile_records]
        grain_validation = validate_cross_asset_state_profile_grain(all_records)
        if not grain_validation["passed"]:
            reasons.extend(str(reason) for reason in grain_validation["errors"])

        for index, record in enumerate(selected):
            _validate_row(
                record,
                REQUIRED_ROW_METADATA_FIELDS,
                reasons,
                prefix=f"selected_{index}",
                require_available_support=True,
                require_lineage_ts=True,
            )
        for index, record in enumerate(diagnostic):
            _validate_row(
                record,
                REQUIRED_ROW_METADATA_FIELDS,
                reasons,
                prefix=f"diagnostic_{index}",
                require_available_support=True,
                require_lineage_ts=True,
            )
        for index, record in enumerate(masked):
            _validate_row(
                record,
                REQUIRED_MASK_METADATA_FIELDS,
                reasons,
                prefix=f"masked_{index}",
                require_available_support=False,
                require_lineage_ts=False,
            )
            if str(record.get("profile_selection_status") or "") != "masked_unavailable":
                reasons.append(f"masked_{index}_profile_selection_status_invalid")
            if str(record.get("output_health_status") or "") != "masked_unavailable":
                reasons.append(f"masked_{index}_output_health_status_invalid")

        if artifact_kind == CROSS_ASSET_STATE_SELECTED_PROFILE_SELECTION_ENGINE_MANIFEST_KIND:
            from src.regimes.cross_asset_state.selection_engine import validate_cross_asset_state_selection_engine_manifest

            embedded = manifest.get("selection_engine_manifest_validation")
            if not isinstance(embedded, Mapping) or embedded.get("passed") is not True:
                reasons.append("selection_engine_manifest_validation_missing_or_blocked")
            selection_validation = validate_cross_asset_state_selection_engine_manifest(manifest)
            if not selection_validation["passed"]:
                reasons.extend(f"selection_engine_manifest:{reason}" for reason in selection_validation["reason_codes"])
        elif artifact_kind == CROSS_ASSET_STATE_SELECTED_PROFILE_MANIFEST_KIND:
            embedded = manifest.get("selected_profile_manifest_validation")
            if not isinstance(embedded, Mapping) or embedded.get("passed") is not True:
                reasons.append("selected_profile_manifest_validation_missing_or_blocked")
        return tuple(dict.fromkeys(reasons))

    return _validate


def _relationship_input_checks(
    manifest: Mapping[str, Any],
    *,
    env: Mapping[str, str] | None = None,
    project_root: str | Path | None = None,
    run_cache: RegimeProductionPlannerRunCache | None = None,
) -> tuple[Mapping[str, Any], ...]:
    if not manifest:
        return ()
    if run_cache is not None:
        source_env = os.environ if env is None else env
        return run_cache.relationship_input_checks(
            manifest_fingerprint=relationship_manifest_fingerprint(manifest),
            env_fingerprint={
                "relationship_discovery_root": str(source_env.get("PIPELINE_RELATIONSHIP_DISCOVERY_ROOT", "")),
                "project_root": str(project_root or ""),
            },
            builder=lambda: _relationship_input_checks_uncached(
                manifest,
                env=source_env,
                project_root=project_root,
            ),
        )
    return _relationship_input_checks_uncached(manifest, env=env, project_root=project_root)


def _relationship_input_checks_uncached(
    manifest: Mapping[str, Any],
    *,
    env: Mapping[str, str] | None = None,
    project_root: str | Path | None = None,
) -> tuple[Mapping[str, Any], ...]:
    if not manifest:
        return ()
    source_env = os.environ if env is None else env
    checks: list[Mapping[str, Any]] = []
    for name, field_name in (
        ("relationship_context_handoff", "relationship_context_handoff_path"),
        ("relationship_eligibility_manifest", "eligibility_manifest_path"),
    ):
        raw_path = manifest.get(field_name)
        if raw_path in (None, ""):
            checks.append(
                {
                    "name": name,
                    "field": field_name,
                    "input_role": "time_indexed_relationship_data_input",
                    "selected_profile_artifact": False,
                    "status": "not_declared",
                    "reason_code": "relationship_input_not_declared",
                    "execution_performed": False,
                }
            )
            continue
        resolution = resolve_regime_production_sidecar_input_path(
            SOURCE_KIND_RELATIONSHIP_DISCOVERY,
            raw_path,
            field_name=field_name,
            manifest=manifest,
            env=source_env,
            project_root=project_root,
            allow_branch_policy_manifest_file=True,
        )
        path = resolution.path
        available = path.exists()
        checks.append(
            {
                "name": name,
                "field": field_name,
                "path": _project_relative_path_text(path, project_root=project_root),
                "path_source": resolution.path_source,
                "root": _project_relative_path_text(resolution.root, project_root=project_root),
                "root_source": resolution.root_source,
                "configured_root_policy": resolution.configured_root_policy,
                "input_role": "time_indexed_relationship_data_input",
                "selected_profile_artifact": False,
                "status": "available" if available else "missing",
                "reason_code": None if available else "relationship_input_missing",
                "execution_performed": False,
            }
        )
    cadence = manifest.get("relationship_context_cadence_policy")
    if isinstance(cadence, Mapping):
        checks.append(
            {
                "name": "relationship_context_cadence_policy",
                "input_role": "freshness_metadata",
                "selected_profile_artifact": False,
                "status": "metadata_present",
                "execution_performed": False,
                "backfill_snapshot_count": len(cadence.get("backfill_snapshot_schedule") or ()),
            }
        )
    return tuple(checks)


def _project_relative_path_text(value: str | Path, *, project_root: str | Path | None = None) -> str:
    resolved = Path(value).resolve()
    project = resolve_project_root(project_root)
    try:
        return str(resolved.relative_to(project))
    except ValueError:
        return f"<external_configured_root>/{resolved.name}"


def _compat_cross_reason_codes(reasons: Sequence[str]) -> tuple[str, ...]:
    out: list[str] = []
    for reason in reasons:
        text = str(reason)
        out.append(text)
        if text == "ambiguous_active_artifact":
            out.append("multiple_active_handoff_artifacts_discoverable")
    return tuple(dict.fromkeys(out))


def validate_cross_asset_state_manifest_for_production(manifest: Mapping[str, Any] | str | Path) -> Mapping[str, Any]:
    payload: Mapping[str, Any]
    if isinstance(manifest, (str, Path)):
        path = Path(manifest)
        if not path.exists():
            raise CrossAssetStateProductionGateError(f"Cross-Asset-State manifest does not exist: {path}")
        loaded = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(loaded, Mapping):
            raise CrossAssetStateProductionGateError("Cross-Asset-State manifest must be a JSON object")
        payload = loaded
    else:
        payload = manifest

    artifact_kind = payload.get("artifact_kind")
    if artifact_kind in CROSS_ASSET_STATE_ACCEPTED_SELECTED_PROFILE_MANIFEST_KINDS:
        raise CrossAssetStateProductionGateError("Cross-Asset-State selected-profile handoff is dry-consumption only")
    if artifact_kind != "cross_asset_state_v1_shape_probe_summary":
        raise CrossAssetStateProductionGateError("Cross-Asset-State production consumer rejects unknown manifest kind")
    if payload.get("production_approved") is not True or payload.get("production_writer_enabled") is not True:
        raise CrossAssetStateProductionGateError("Cross-Asset-State production gates are closed")
    if payload.get("canonical_production_state_outputs_written") is not True:
        raise CrossAssetStateProductionGateError("Cross-Asset-State canonical production outputs are absent")
    return payload


def _blocked(
    path: Path,
    *,
    sandbox_nonproduction_mode: bool,
    production_mode_requested: bool,
    reasons: Sequence[str],
) -> CrossAssetStateProductionConsumerValidation:
    return CrossAssetStateProductionConsumerValidation(
        status=CROSS_ASSET_STATE_CONSUMER_STATUS_BLOCKED,
        manifest_path=path,
        sandbox_nonproduction_mode=bool(sandbox_nonproduction_mode),
        production_mode_requested=bool(production_mode_requested),
        reason_codes=tuple(dict.fromkeys(str(reason) for reason in reasons)),
        production_write_allowed=False,
        canonical_outputs_written=False,
    )


def _int_field(manifest: Mapping[str, Any], field_name: str, reasons: list[str]) -> int:
    try:
        return int(manifest[field_name])
    except KeyError:
        reasons.append(f"{field_name}_missing")
    except Exception:
        reasons.append(f"{field_name}_invalid")
    return 0


def _declared_count(
    manifest: Mapping[str, Any],
    field_names: Sequence[str],
    actual: int,
    reasons: list[str],
    *,
    reason_code: str,
    default_if_absent: int | None = None,
) -> int:
    for field_name in field_names:
        if field_name not in manifest:
            continue
        declared = _int_field(manifest, field_name, reasons)
        if declared != actual:
            reasons.append(reason_code)
        return declared
    if default_if_absent is None:
        reasons.append(f"{field_names[0]}_missing")
        return 0
    if default_if_absent != actual:
        reasons.append(reason_code)
    return int(default_if_absent)


def _validate_row(
    row: Mapping[str, Any],
    required_fields: Sequence[str],
    reasons: list[str],
    *,
    prefix: str,
    require_available_support: bool,
    require_lineage_ts: bool,
) -> None:
    for field_name in required_fields:
        if field_name not in row:
            reasons.append(f"{prefix}_{field_name}_missing")
        elif field_name == "mask_reason" and row.get(field_name) in (None, ""):
            reasons.append(f"{prefix}_{field_name}_missing")
    reasons.extend(
        validate_nonproduction_gate_flags(
            row,
            require_canonical=False,
            expected_requires_human_approval=True,
            prefix=prefix,
        )
    )
    if require_available_support:
        for field_name in ("support_definition_id", "support_size", "support_quality"):
            value = row.get(field_name)
            if value in (None, ""):
                reasons.append(f"{prefix}_{field_name}_empty")
    if require_lineage_ts and (row.get("source_tail_ts") in (None, "") or row.get("known_at_ts") in (None, "")):
        reasons.append(f"{prefix}_lineage_ts_missing")


def _partial_artifact_rejection_reasons(manifest: Mapping[str, Any]) -> tuple[str, ...]:
    reasons: list[str] = []
    artifact_kind = str(manifest.get("artifact_kind") or "")
    artifact_scope = str(manifest.get("artifact_scope") or "")
    shard_status = str(manifest.get("shard_status") or "")

    if artifact_kind == CROSS_ASSET_STATE_PROGRESS_HEARTBEAT_KIND or artifact_scope == CROSS_ASSET_STATE_HEARTBEAT_SCOPE:
        reasons.append("heartbeat_artifact_not_active_handoff")
    if artifact_kind == CROSS_ASSET_STATE_RUNTIME_TELEMETRY_EVENT_KIND or artifact_scope == CROSS_ASSET_STATE_RUNTIME_TELEMETRY_SCOPE:
        reasons.append("runtime_telemetry_not_active_handoff")
    if artifact_kind == CROSS_ASSET_STATE_FAMILY_SHARD_MANIFEST_KIND or artifact_scope == CROSS_ASSET_STATE_FAMILY_SHARD_SCOPE:
        reasons.append("family_shard_artifact_not_active_handoff")
    if artifact_kind == "cross_asset_state_execution_shard_manifest" or artifact_scope == "execution_shard":
        reasons.append("execution_shard_artifact_not_active_handoff")
    if shard_status == SHARD_STATUS_RUNNING:
        reasons.append("running_shard_not_active_handoff")
    elif shard_status in {SHARD_STATUS_PENDING, SHARD_STATUS_INCOMPLETE, SHARD_STATUS_FAILED}:
        reasons.append("incomplete_shard_not_active_handoff")
    if _contains_reason_code(manifest, "run_fingerprint_hash_mismatch"):
        reasons.append("stale_shard_fingerprint_mismatch")

    if artifact_kind in CROSS_ASSET_STATE_ACCEPTED_SELECTED_PROFILE_MANIFEST_KINDS:
        if artifact_scope != CROSS_ASSET_STATE_DEFAULT_COMBINED_SCOPE:
            reasons.append("combined_artifact_scope_invalid")
        if manifest.get("final_artifact") is not True:
            reasons.append("final_artifact_flag_missing_or_false")
        if manifest.get("partial_artifact") is True or manifest.get("incomplete_artifact") is True:
            reasons.append("incomplete_combined_artifact")
        if str(manifest.get("combined_artifact_status") or "") != SHARD_STATUS_COMPLETE:
            reasons.append("combined_artifact_status_not_complete")
        completeness = manifest.get("shard_completeness_metadata") or manifest.get("parent_finalizer_shard_completeness")
        if not isinstance(completeness, Mapping):
            reasons.append("shard_completeness_metadata_missing")
        elif completeness.get("passed") is not True or completeness.get("final_active_artifact_write_allowed") is not True:
            reasons.append("parent_finalizer_shard_completeness_blocked")
            reasons.append("incomplete_combined_artifact")
    return tuple(dict.fromkeys(reasons))


def _contains_reason_code(payload: Mapping[str, Any], reason_code: str) -> bool:
    stack: list[Any] = [payload]
    while stack:
        current = stack.pop()
        if isinstance(current, Mapping):
            reasons = current.get("reason_codes")
            if isinstance(reasons, (list, tuple, set)) and reason_code in {str(reason) for reason in reasons}:
                return True
            stack.extend(current.values())
        elif isinstance(current, (list, tuple)):
            stack.extend(current)
    return False


def _active_handoff_discovery_reasons(path: Path, *, expected_active_filename: str) -> tuple[str, ...]:
    if path.name != expected_active_filename or not path.parent.exists():
        return ()
    active_paths: list[Path] = []
    for candidate in path.parent.glob("cross_asset_state_selected_profiles*.nonprod.json"):
        if not candidate.is_file():
            continue
        try:
            loaded = json.loads(candidate.read_text(encoding="utf-8"))
        except Exception:
            continue
        if (
            isinstance(loaded, Mapping)
            and loaded.get("artifact_kind") in CROSS_ASSET_STATE_ACCEPTED_SELECTED_PROFILE_MANIFEST_KINDS
            and loaded.get("single_active_nonproduction_handoff_artifact") == candidate.name
            and loaded.get("stale_sandbox_manifest_used") is False
        ):
            active_paths.append(candidate)
    if len(active_paths) > 1:
        return ("multiple_active_handoff_artifacts_discoverable",)
    return ()


__all__ = [
    "CROSS_ASSET_STATE_ACTIVE_HANDOFF_ROOT",
    "CROSS_ASSET_STATE_ACTIVE_SELECTED_PROFILES_PATH",
    "CROSS_ASSET_STATE_CONSUMER_STATUS_BLOCKED",
    "CROSS_ASSET_STATE_CONSUMER_STATUS_READY_FOR_DRY_CONSUMPTION",
    "CROSS_ASSET_STATE_DEFAULT_SELECTED_PROFILES_FILENAME",
    "CROSS_ASSET_STATE_SELECTED_PROFILE_MANIFEST_ENV",
    "CROSS_ASSET_STATE_SELECTED_PROFILE_SELECTION_ENGINE_MANIFEST_KIND",
    "CrossAssetStateProductionConsumerValidation",
    "CrossAssetStateProductionGateError",
    "default_cross_asset_state_selected_profiles_manifest_path",
    "validate_cross_asset_state_manifest_for_production",
    "validate_cross_asset_state_selected_profiles_for_consumption",
    "validate_default_cross_asset_state_selected_profiles_for_consumption",
]
