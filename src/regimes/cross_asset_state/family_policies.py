from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


FAMILY_STATUS_SELECTED_MODEL_FACING = "selected_model_facing"
FAMILY_STATUS_DIAGNOSTIC_ONLY = "diagnostic_only"
FAMILY_STATUS_MASKED_UNAVAILABLE = "masked_unavailable"
FAMILY_STATUS_NEEDS_REPAIR = "needs_repair"
FAMILY_STATUS_BLOCKED = "blocked"

CROSS_ASSET_STATE_FAMILY_POLICY_VERSION = "cross_asset_state_family_policy_v1"


@dataclass(frozen=True)
class CrossAssetStateFamilyPolicy:
    relationship_feature_family: str
    family_selection_status: str
    model_facing_eligible: bool
    reason: str
    evidence: Mapping[str, Any]
    recommended_next_action: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "relationship_feature_family": self.relationship_feature_family,
            "family_selection_status": self.family_selection_status,
            "model_facing_eligible": bool(self.model_facing_eligible),
            "reason": self.reason,
            "evidence": dict(self.evidence),
            "recommended_next_action": self.recommended_next_action,
            "production_approved": False,
            "production_writer_enabled": False,
        }


def classify_cross_asset_state_family(
    family: str,
    *,
    selected_count: int,
    masked_count: int,
    selected_profile_counts: Mapping[str, int],
    low_spread_mask_count: int = 0,
    questionable_count: int = 0,
    non_default_window_share: float = 0.0,
) -> CrossAssetStateFamilyPolicy:
    evidence = {
        "selected_count": int(selected_count),
        "masked_count": int(masked_count),
        "selected_profile_counts": dict(selected_profile_counts),
        "low_spread_mask_count": int(low_spread_mask_count),
        "questionable_count": int(questionable_count),
        "non_default_window_share": round(float(non_default_window_share), 6),
    }
    if selected_count <= 0:
        return CrossAssetStateFamilyPolicy(
            relationship_feature_family=family,
            family_selection_status=FAMILY_STATUS_BLOCKED,
            model_facing_eligible=False,
            reason="No coherent selected cells were available in the bounded run.",
            evidence=evidence,
            recommended_next_action="repair source values or masks before considering model-facing use",
        )
    if family == "residual_peer_signal":
        return CrossAssetStateFamilyPolicy(
            relationship_feature_family=family,
            family_selection_status=FAMILY_STATUS_SELECTED_MODEL_FACING,
            model_facing_eligible=True,
            reason="Positive-control family remains inspectable and has rule-threshold support.",
            evidence=evidence,
            recommended_next_action="wire leakage-safe economic diagnostics before final production consideration",
        )
    if family == "anchor_core_exposure":
        return CrossAssetStateFamilyPolicy(
            relationship_feature_family=family,
            family_selection_status=FAMILY_STATUS_SELECTED_MODEL_FACING,
            model_facing_eligible=True,
            reason="Plausible v1 axis, but saturation-aware scoring remains required before final Test Branch confidence.",
            evidence=evidence,
            recommended_next_action="add saturation-aware scoring and economic outcome checks",
        )
    if family == "relationship_concentration_entropy":
        return CrossAssetStateFamilyPolicy(
            relationship_feature_family=family,
            family_selection_status=FAMILY_STATUS_SELECTED_MODEL_FACING,
            model_facing_eligible=True,
            reason="Repaired variable-peer concentration/entropy features produced coherent selected cells; unavailable cells remain explicitly masked.",
            evidence=evidence,
            recommended_next_action="keep production gates closed until leakage-safe economic diagnostics are attached",
        )
    if family == "peer_strength_stability":
        return CrossAssetStateFamilyPolicy(
            relationship_feature_family=family,
            family_selection_status=FAMILY_STATUS_SELECTED_MODEL_FACING,
            model_facing_eligible=True,
            reason="Repaired variable-peer strength/stability features produced coherent selected cells; unavailable cells remain explicitly masked.",
            evidence=evidence,
            recommended_next_action="keep production gates closed until leakage-safe economic diagnostics are attached",
        )
    return CrossAssetStateFamilyPolicy(
        relationship_feature_family=family,
        family_selection_status=FAMILY_STATUS_NEEDS_REPAIR,
        model_facing_eligible=False,
        reason="No family-specific v1 policy is declared.",
        evidence=evidence,
        recommended_next_action="declare a source-owned family policy before model-facing use",
    )


def cross_asset_state_family_policy_manifest(policies: Sequence[CrossAssetStateFamilyPolicy]) -> dict[str, Any]:
    return {
        "artifact_kind": "cross_asset_state_family_policy_manifest",
        "family_policy_version": CROSS_ASSET_STATE_FAMILY_POLICY_VERSION,
        "families": [policy.as_dict() for policy in policies],
        "selected_model_facing_families": [
            policy.relationship_feature_family
            for policy in policies
            if policy.family_selection_status == FAMILY_STATUS_SELECTED_MODEL_FACING and policy.model_facing_eligible
        ],
        "diagnostic_only_families": [
            policy.relationship_feature_family
            for policy in policies
            if policy.family_selection_status in {FAMILY_STATUS_DIAGNOSTIC_ONLY, FAMILY_STATUS_NEEDS_REPAIR}
        ],
        "blocked_families": [
            policy.relationship_feature_family
            for policy in policies
            if policy.family_selection_status == FAMILY_STATUS_BLOCKED
        ],
        "production_approved": False,
        "production_writer_enabled": False,
    }


def validate_cross_asset_state_family_policy_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(manifest)
    families = [dict(item) for item in payload.get("families") or () if isinstance(item, Mapping)]
    reason_codes: list[str] = []
    if payload.get("artifact_kind") != "cross_asset_state_family_policy_manifest":
        reason_codes.append("artifact_kind_invalid")
    if payload.get("family_policy_version") != CROSS_ASSET_STATE_FAMILY_POLICY_VERSION:
        reason_codes.append("family_policy_version_invalid")
    if payload.get("production_approved") is not False or payload.get("production_writer_enabled") is not False:
        reason_codes.append("production_flags_not_fail_closed")
    statuses = {str(item.get("family_selection_status")) for item in families}
    if FAMILY_STATUS_SELECTED_MODEL_FACING not in statuses:
        reason_codes.append("no_model_facing_family_selected")
    for item in families:
        if item.get("family_selection_status") == FAMILY_STATUS_SELECTED_MODEL_FACING and item.get("model_facing_eligible") is not True:
            reason_codes.append(f"{item.get('relationship_feature_family')}_model_facing_status_mismatch")
        if item.get("production_approved") is not False or item.get("production_writer_enabled") is not False:
            reason_codes.append(f"{item.get('relationship_feature_family')}_production_flags_not_fail_closed")
    return {
        "artifact_kind": "cross_asset_state_family_policy_manifest_validation",
        "status": "passed" if not reason_codes else "blocked",
        "passed": not reason_codes,
        "reason_codes": reason_codes,
        "family_count": len(families),
        "production_write_allowed": False,
    }


__all__ = [
    "CROSS_ASSET_STATE_FAMILY_POLICY_VERSION",
    "FAMILY_STATUS_BLOCKED",
    "FAMILY_STATUS_DIAGNOSTIC_ONLY",
    "FAMILY_STATUS_MASKED_UNAVAILABLE",
    "FAMILY_STATUS_NEEDS_REPAIR",
    "FAMILY_STATUS_SELECTED_MODEL_FACING",
    "CrossAssetStateFamilyPolicy",
    "classify_cross_asset_state_family",
    "cross_asset_state_family_policy_manifest",
    "validate_cross_asset_state_family_policy_manifest",
]
