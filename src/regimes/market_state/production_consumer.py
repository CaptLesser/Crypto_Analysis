from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.regimes.core.production_consumer import (
    REGIME_BRANCH_MARKET_STATE,
    REGIME_PRODUCTION_STATUS_READY_FOR_DRY_CONSUMPTION,
    RegimeProductionArtifactResolution,
    RegimeProductionBranchValidationContext,
    RegimeProductionPlannerRunCache,
    build_regime_production_dry_run_plan,
    default_regime_production_branch_policy,
    resolve_active_selected_profile_artifact,
)
from src.regimes.core.serialization import to_jsonable
from src.regimes.core.test_branch_contracts import (
    PRODUCTION_GATE_FIELDS,
    validate_nonproduction_gate_flags,
)
from src.regimes.market_state.axis_contracts import MARKET_STATE_V1_AXIS_IDS
from src.regimes.market_state.test_branch_campaign import (
    DEFAULT_BAND_INTERVALS,
    MARKET_STATE_SELECTED_PROFILES_FILENAME,
    MARKET_STATE_TEST_CAMPAIGN_ARTIFACT_KIND,
)


MARKET_STATE_PRODUCTION_CONSUMER_SCHEMA_VERSION = 1
MARKET_STATE_CONSUMER_STATUS_READY_FOR_CONSUMPTION = "ready_for_consumption"
MARKET_STATE_CONSUMER_STATUS_READY_FOR_SANDBOX_DRY_RUN = MARKET_STATE_CONSUMER_STATUS_READY_FOR_CONSUMPTION
MARKET_STATE_CONSUMER_STATUS_BLOCKED = "blocked"
MARKET_STATE_SELECTED_PROFILE_MANIFEST_ENV = "PIPELINE_MARKET_STATE_SELECTED_PROFILE_MANIFEST"
MARKET_STATE_ACTIVE_HANDOFF_ROOT = Path("_codex_artifacts") / "reports" / "market_state_active_handoff"
MARKET_STATE_ACTIVE_SELECTED_PROFILES_PATH = MARKET_STATE_ACTIVE_HANDOFF_ROOT / MARKET_STATE_SELECTED_PROFILES_FILENAME

REQUIRED_TOP_LEVEL_FLAGS: tuple[str, ...] = PRODUCTION_GATE_FIELDS

REQUIRED_PROFILE_FIELDS: tuple[str, ...] = (
    "profile_id",
    "market_axis",
    "band",
    "source_interval",
    "selected_method_profile",
    "selected_method_family",
    "selected_feature_pool",
    "selected_feature_set",
    "score_evidence_summary",
    "label_output_health_gate_summary",
    "source_tail_ts",
    "known_at_ts",
    "run_id",
    "trial_study_lineage",
    "selection_scope",
)

REQUIRED_MASK_FIELDS: tuple[str, ...] = (
    "axis",
    "band",
    "interval",
    "availability_status",
    "mask_reason_code",
    "reason",
    "profile_id",
)


def default_market_state_selected_profiles_manifest_path(
    *,
    env: Mapping[str, str] | None = None,
) -> Path:
    """Return the configured active Market-State selected-profile handoff path."""

    source_env = os.environ if env is None else env
    configured = str(source_env.get(MARKET_STATE_SELECTED_PROFILE_MANIFEST_ENV, "")).strip()
    if configured:
        return Path(configured).expanduser()
    resolution = resolve_active_selected_profile_artifact(
        REGIME_BRANCH_MARKET_STATE,
        env=source_env,
        allow_explicit_artifact_override=False,
    )
    if resolution.artifact_path is not None:
        return resolution.artifact_path
    raise FileNotFoundError("Market-State active selected-profile artifact is not configured")


@dataclass(frozen=True)
class MarketStateProductionConsumerValidation:
    status: str
    manifest_path: Path
    sandbox_nonproduction_mode: bool = False
    production_mode_requested: bool = False
    reason_codes: Sequence[str] = ()
    selected_profile_count: int = 0
    masked_or_skipped_count: int = 0
    covered_cell_count: int = 0
    expected_cell_count: int = 0
    missing_cells: Sequence[Mapping[str, Any]] = ()
    production_write_allowed: bool = False
    canonical_outputs_written: bool = False
    normalized_manifest_version: Mapping[str, Any] = field(default_factory=dict)
    dry_run_plan: Mapping[str, Any] = field(default_factory=dict)
    shared_validation: Mapping[str, Any] = field(default_factory=dict)
    branch_validator_hook_used: bool = False
    manifest: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": MARKET_STATE_PRODUCTION_CONSUMER_SCHEMA_VERSION,
            "artifact_kind": "market_state_production_consumer_validation",
            "status": self.status,
            "manifest_path": str(self.manifest_path),
            "sandbox_nonproduction_mode": bool(self.sandbox_nonproduction_mode),
            "production_mode_requested": bool(self.production_mode_requested),
            "reason_codes": list(self.reason_codes),
            "selected_profile_count": int(self.selected_profile_count),
            "masked_or_skipped_count": int(self.masked_or_skipped_count),
            "covered_cell_count": int(self.covered_cell_count),
            "expected_cell_count": int(self.expected_cell_count),
            "missing_cells": [dict(item) for item in self.missing_cells],
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


def validate_market_state_selected_profiles_for_consumption(
    manifest_path: str | Path,
    *,
    sandbox_nonproduction_mode: bool = False,
    production_mode_requested: bool = False,
    expected_band_intervals: Sequence[tuple[str, int]] = DEFAULT_BAND_INTERVALS,
    expected_axes: Sequence[str] = MARKET_STATE_V1_AXIS_IDS,
    run_cache: RegimeProductionPlannerRunCache | None = None,
) -> MarketStateProductionConsumerValidation:
    path = Path(manifest_path)
    branch_validator = _market_state_branch_validator(
        expected_band_intervals=expected_band_intervals,
        expected_axes=expected_axes,
    )
    resolution = resolve_active_selected_profile_artifact(
        REGIME_BRANCH_MARKET_STATE,
        explicit_artifact_path=path,
        env={},
        check_explicit_parent_ambiguity=True,
        branch_validator=branch_validator,
        run_cache=run_cache,
        cache_fingerprint={
            "branch": REGIME_BRANCH_MARKET_STATE,
            "expected_band_intervals": list(expected_band_intervals),
            "expected_axes": list(expected_axes),
            "sandbox_nonproduction_mode": bool(sandbox_nonproduction_mode),
            "production_mode_requested": bool(production_mode_requested),
        },
    )
    return _from_shared_resolution(
        resolution,
        fallback_path=path,
        sandbox_nonproduction_mode=sandbox_nonproduction_mode,
        production_mode_requested=production_mode_requested,
        expected_band_intervals=expected_band_intervals,
        expected_axes=expected_axes,
    )


def validate_default_market_state_selected_profiles_for_consumption(
    *,
    sandbox_nonproduction_mode: bool = False,
    production_mode_requested: bool = False,
    expected_band_intervals: Sequence[tuple[str, int]] = DEFAULT_BAND_INTERVALS,
    expected_axes: Sequence[str] = MARKET_STATE_V1_AXIS_IDS,
    env: Mapping[str, str] | None = None,
    run_cache: RegimeProductionPlannerRunCache | None = None,
) -> MarketStateProductionConsumerValidation:
    branch_validator = _market_state_branch_validator(
        expected_band_intervals=expected_band_intervals,
        expected_axes=expected_axes,
    )
    resolution = resolve_active_selected_profile_artifact(
        REGIME_BRANCH_MARKET_STATE,
        env=env,
        allow_explicit_artifact_override=False,
        branch_validator=branch_validator,
        run_cache=run_cache,
        cache_fingerprint={
            "branch": REGIME_BRANCH_MARKET_STATE,
            "expected_band_intervals": list(expected_band_intervals),
            "expected_axes": list(expected_axes),
            "sandbox_nonproduction_mode": bool(sandbox_nonproduction_mode),
            "production_mode_requested": bool(production_mode_requested),
            "env_manifest": str((env or {}).get(MARKET_STATE_SELECTED_PROFILE_MANIFEST_ENV, "")),
        },
    )
    return _from_shared_resolution(
        resolution,
        fallback_path=resolution.artifact_path or Path(default_regime_production_branch_policy(REGIME_BRANCH_MARKET_STATE).active_filename),
        sandbox_nonproduction_mode=sandbox_nonproduction_mode,
        production_mode_requested=production_mode_requested,
        expected_band_intervals=expected_band_intervals,
        expected_axes=expected_axes,
    )


def _blocked(
    path: Path,
    *,
    sandbox_nonproduction_mode: bool,
    production_mode_requested: bool,
    reasons: Sequence[str],
) -> MarketStateProductionConsumerValidation:
    return MarketStateProductionConsumerValidation(
        status=MARKET_STATE_CONSUMER_STATUS_BLOCKED,
        manifest_path=path,
        sandbox_nonproduction_mode=bool(sandbox_nonproduction_mode),
        production_mode_requested=bool(production_mode_requested),
        reason_codes=tuple(dict.fromkeys(str(reason) for reason in reasons)),
        production_write_allowed=False,
        canonical_outputs_written=False,
    )


def _from_shared_resolution(
    resolution: RegimeProductionArtifactResolution,
    *,
    fallback_path: Path,
    sandbox_nonproduction_mode: bool,
    production_mode_requested: bool,
    expected_band_intervals: Sequence[tuple[str, int]],
    expected_axes: Sequence[str],
) -> MarketStateProductionConsumerValidation:
    validation = resolution.validation
    manifest = dict(resolution.manifest)
    expected_cells = {(str(axis), str(band), int(interval)) for axis in expected_axes for band, interval in expected_band_intervals}
    selected = [dict(item) for item in manifest.get("selected_profiles") or () if isinstance(item, Mapping)]
    masked = [dict(item) for item in manifest.get("masked_or_skipped_cells") or () if isinstance(item, Mapping)]
    covered_cells = {
        (str(item.get("market_axis")), str(item.get("band")), int(item.get("source_interval") or 0))
        for item in selected
    } | {
        (str(item.get("axis")), str(item.get("band")), int(item.get("interval") or 0))
        for item in masked
    }
    missing_cells = [
        {"axis": axis, "band": band, "interval": interval}
        for axis, band, interval in sorted(expected_cells.difference(covered_cells))
    ]
    status = (
        MARKET_STATE_CONSUMER_STATUS_READY_FOR_SANDBOX_DRY_RUN
        if resolution.status == REGIME_PRODUCTION_STATUS_READY_FOR_DRY_CONSUMPTION
        else MARKET_STATE_CONSUMER_STATUS_BLOCKED
    )
    expose_manifest = bool(sandbox_nonproduction_mode) or status == MARKET_STATE_CONSUMER_STATUS_READY_FOR_SANDBOX_DRY_RUN
    dry_run_plan: Mapping[str, Any] = {}
    if validation is not None and validation.artifact_path is not None:
        dry_run_plan = build_regime_production_dry_run_plan(validation).as_dict()
    normalized_version = validation.manifest_version.as_dict() if validation is not None and validation.manifest_version is not None else {}
    return MarketStateProductionConsumerValidation(
        status=status,
        manifest_path=resolution.artifact_path or fallback_path,
        sandbox_nonproduction_mode=bool(sandbox_nonproduction_mode),
        production_mode_requested=bool(production_mode_requested),
        reason_codes=_compat_market_reason_codes(resolution.reason_codes),
        selected_profile_count=validation.selected_cell_count if validation is not None else 0,
        masked_or_skipped_count=validation.masked_unavailable_cell_count if validation is not None else 0,
        covered_cell_count=validation.covered_cell_count if validation is not None else len(covered_cells),
        expected_cell_count=validation.expected_cell_count if validation is not None else len(expected_cells),
        missing_cells=tuple(missing_cells),
        production_write_allowed=False,
        canonical_outputs_written=False,
        normalized_manifest_version=normalized_version,
        dry_run_plan=dry_run_plan,
        shared_validation=validation.as_dict() if validation is not None else {},
        branch_validator_hook_used=bool(validation and validation.metadata.get("branch_validator_hook_used")),
        manifest=manifest if expose_manifest else {},
    )


def _market_state_branch_validator(
    *,
    expected_band_intervals: Sequence[tuple[str, int]],
    expected_axes: Sequence[str],
):
    def _validate(manifest: Mapping[str, Any], context: RegimeProductionBranchValidationContext) -> Sequence[str]:
        reasons: list[str] = []
        if context.artifact_path is not None and context.artifact_path.name != MARKET_STATE_SELECTED_PROFILES_FILENAME:
            reasons.append("unexpected_selected_manifest_filename")
        if manifest.get("artifact_kind") != MARKET_STATE_TEST_CAMPAIGN_ARTIFACT_KIND:
            reasons.append("artifact_kind_invalid")
        if manifest.get("single_active_nonproduction_handoff_artifact") != MARKET_STATE_SELECTED_PROFILES_FILENAME:
            reasons.append("single_active_manifest_marker_missing")
        if manifest.get("test_branch_validation_status") != "passed":
            reasons.append("test_branch_validation_status_not_passed")
        if manifest.get("profile_selection_status") != "approved_by_test_branch":
            reasons.append("profile_selection_status_not_approved_by_test_branch")

        selected = [dict(item) for item in context.selected_records]
        masked = [dict(item) for item in context.masked_records]
        seen_profile_ids: set[str] = set()
        for profile in selected:
            _validate_required(profile, REQUIRED_PROFILE_FIELDS, reasons, prefix="profile")
            reasons.extend(
                validate_nonproduction_gate_flags(
                    profile,
                    require_canonical=False,
                    expected_requires_human_approval=False,
                    prefix="profile",
                )
            )
            profile_id = _non_empty(profile.get("profile_id"))
            if profile_id is None:
                reasons.append("profile_id_missing_or_empty")
            elif profile_id in seen_profile_ids:
                reasons.append(f"duplicate_profile_id:{profile_id}")
            else:
                seen_profile_ids.add(profile_id)
            if profile.get("availability_status") != "selected":
                reasons.append(f"profile_availability_invalid:{profile.get('profile_id')}")
            if profile.get("source_tail_ts") is None or profile.get("known_at_ts") is None:
                reasons.append(f"profile_lineage_ts_missing:{profile.get('profile_id')}")

        for mask in masked:
            _validate_required(mask, REQUIRED_MASK_FIELDS, reasons, prefix="mask")
            reasons.extend(
                validate_nonproduction_gate_flags(
                    mask,
                    require_canonical=False,
                    expected_requires_human_approval=False,
                    prefix="mask",
                )
            )
            if mask.get("availability_status") != "masked_unavailable":
                reasons.append(f"mask_availability_invalid:{mask.get('axis')}:{mask.get('band')}")
            if not _non_empty(mask.get("mask_reason_code")):
                reasons.append(f"mask_reason_code_missing:{mask.get('axis')}:{mask.get('band')}")

        expected_cells = {(str(axis), str(band), int(interval)) for axis in expected_axes for band, interval in expected_band_intervals}
        covered_cells = {
            (str(item.get("market_axis")), str(item.get("band")), int(item.get("source_interval") or 0))
            for item in selected
        } | {
            (str(item.get("axis")), str(item.get("band")), int(item.get("interval") or 0))
            for item in masked
        }
        if expected_cells.difference(covered_cells):
            reasons.append("axis_band_cells_missing")
        if isinstance(manifest.get("source_lineage"), Mapping) and manifest.get("source_lineage", {}).get("stale_sandbox_manifest_used") is True:
            reasons.append("shared_manifest:stale_sandbox_source_lineage_marked_active")
        return tuple(dict.fromkeys(reasons))

    return _validate


def _compat_market_reason_codes(reasons: Sequence[str]) -> tuple[str, ...]:
    out: list[str] = []
    for reason in reasons:
        text = str(reason)
        out.append(text)
        if text == "ambiguous_active_artifact":
            out.append("multiple_active_handoff_artifacts_discoverable")
        elif text == "production_consumable_missing_or_false":
            out.append("production_consumable_not_true")
    return tuple(dict.fromkeys(out))


def _validate_required(payload: Mapping[str, Any], required: Sequence[str], reasons: list[str], *, prefix: str) -> None:
    for field in required:
        if field not in payload:
            reasons.append(f"{prefix}_{field}_missing")


def _non_empty(value: object) -> str | None:
    text = str(value or "").strip()
    return text or None


__all__ = [
    "MARKET_STATE_ACTIVE_HANDOFF_ROOT",
    "MARKET_STATE_ACTIVE_SELECTED_PROFILES_PATH",
    "MARKET_STATE_CONSUMER_STATUS_BLOCKED",
    "MARKET_STATE_CONSUMER_STATUS_READY_FOR_CONSUMPTION",
    "MARKET_STATE_CONSUMER_STATUS_READY_FOR_SANDBOX_DRY_RUN",
    "MARKET_STATE_SELECTED_PROFILE_MANIFEST_ENV",
    "MarketStateProductionConsumerValidation",
    "default_market_state_selected_profiles_manifest_path",
    "validate_default_market_state_selected_profiles_for_consumption",
    "validate_market_state_selected_profiles_for_consumption",
]
