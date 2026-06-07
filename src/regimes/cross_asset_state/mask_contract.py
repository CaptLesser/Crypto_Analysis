from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from src.regimes.cross_asset_state.feature_families import CROSS_ASSET_STATE_SCHEMA_VERSION
from src.regimes.cross_asset_state.relationship_registry import RelationshipAvailabilityRecord


class CrossAssetStateMaskReason:
    RELATIONSHIP_SNAPSHOT_UNAVAILABLE = "relationship_snapshot_unavailable"
    ASSET_NOT_IN_RELATIONSHIP_SNAPSHOT = "asset_not_in_relationship_snapshot"
    MISSING_RELATIONSHIP_SNAPSHOT = "missing_relationship_snapshot"
    MISSING_ANCHOR = "missing_anchor"
    MISSING_CORE_BASKET = "missing_core_basket"
    NO_VIABLE_PEER_EDGES = "no_viable_peer_edges"
    INSUFFICIENT_PEER_HISTORY = "insufficient_peer_history"
    INSUFFICIENT_WINDOW_HISTORY = "insufficient_window_history"
    INSUFFICIENT_OVERLAP = "insufficient_overlap"
    ZERO_VARIANCE_DENOMINATOR = "zero_variance_denominator"
    INVALID_CORRELATION_WINDOW = "invalid_correlation_window"
    MISSING_REQUIRED_FAMILY_FIELDS = "missing_required_family_fields"
    INSUFFICIENT_ROWS = "insufficient_rows"
    INSUFFICIENT_VALID_UNMASKED_ROWS = "insufficient_valid_unmasked_rows"
    UNAVAILABLE_ZERO_MASKED = "unavailable_zero_masked"
    LOW_FEATURE_SPREAD = "low_feature_spread"
    CONSTANT_PEER_COUNT = "constant_peer_count"
    COMPRESSED_PEER_STRENGTH = "compressed_peer_strength"
    COMPRESSED_PEER_STABILITY = "compressed_peer_stability"
    LOW_ENTROPY_SPREAD = "low_entropy_spread"
    LOW_CONCENTRATION_SPREAD = "low_concentration_spread"
    INSUFFICIENT_EDGE_DIVERSITY = "insufficient_edge_diversity"
    NO_VARIABLE_PEER_SUPPORT = "no_variable_peer_support"
    INSUFFICIENT_PEER_SUPPORT = "insufficient_peer_support"
    INSUFFICIENT_CANDIDATE_EDGES = "insufficient_candidate_edges"
    NO_CANDIDATE_EDGES_AVAILABLE = "no_candidate_edges_available"
    NO_VALID_PEER_WEIGHTS = "no_valid_peer_weights"
    BELOW_MINIMUM_SUPPORT_FOR_ENTROPY = "below_minimum_support_for_entropy"
    BELOW_MINIMUM_SUPPORT_FOR_CONCENTRATION = "below_minimum_support_for_concentration"
    NO_PEER_WEIGHTS_AVAILABLE = "no_peer_weights_available"
    INSUFFICIENT_PRIOR_SNAPSHOT_FOR_CHURN = "insufficient_prior_snapshot_for_churn"
    ALL_PEER_WEIGHTS_EQUAL = "all_peer_weights_equal"
    LOW_EDGE_WEIGHT_SPREAD = "low_edge_weight_spread"
    UNSUPPORTED_SUPPORT_DEFINITION = "unsupported_support_definition"
    MISSING_REQUIRED_REPAIRED_COLUMN = "missing_required_repaired_column"
    NO_VIABLE_PROFILE = "no_viable_profile"
    FAMILY_DIAGNOSTIC_ONLY = "family_diagnostic_only"
    PROFILE_TYPE_NOT_SELECTION_ELIGIBLE = "profile_type_not_selection_eligible"
    ECONOMIC_PANEL_MISSING = "economic_panel_missing"
    NONFINITE_INPUT = "nonfinite_input"
    NONFINITE_OUTPUT = "nonfinite_output"
    NONFINITE_FAMILY_FEATURES = "nonfinite_family_features"
    DEGENERATE_LABELS = "degenerate_labels"
    DIAGNOSTIC_ONLY_PEER_METADATA = "diagnostic_only_peer_metadata"
    STALE_RELATIONSHIP_SNAPSHOT = "stale_relationship_snapshot"
    AMBIGUOUS_SNAPSHOT_RESOLUTION = "ambiguous_snapshot_resolution"
    NOT_APPLICABLE_FOR_FAMILY = "not_applicable_for_family"
    UNSUPPORTED_FEATURE_FAMILY = "unsupported_feature_family"
    PRODUCTION_GATE_CLOSED = "production_gate_closed"


CROSS_ASSET_STATE_MASK_REASONS: frozenset[str] = frozenset(
    value
    for name, value in vars(CrossAssetStateMaskReason).items()
    if name.isupper() and isinstance(value, str)
)


@dataclass(frozen=True)
class CrossAssetStateMaskedCell:
    asset_id: str
    relationship_feature_family: str
    band: str
    mask_reason: str
    ts: int | float | str | None = None
    relationship_snapshot_id: str | None = None
    known_at_ts: int | float | str | None = None
    source_tail_ts: int | float | str | None = None
    relationship_registry_mask_reason: str | None = None
    missing_columns: tuple[str, ...] = ()
    observed_rows: int = 0
    production_enabled: bool = False
    production_outputs_written: bool = False
    canonical_production_state_outputs_written: bool = False
    schema_version: int = CROSS_ASSET_STATE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.production_enabled or self.production_outputs_written or self.canonical_production_state_outputs_written:
            raise ValueError("Cross-Asset-State masked cells are sandbox/non-production only")
        reason = str(self.mask_reason).strip()
        if reason not in CROSS_ASSET_STATE_MASK_REASONS:
            raise ValueError(f"Unsupported Cross-Asset-State mask_reason {reason!r}")
        object.__setattr__(self, "asset_id", str(self.asset_id).strip())
        object.__setattr__(self, "relationship_feature_family", str(self.relationship_feature_family).strip())
        object.__setattr__(self, "band", str(self.band).strip().lower())
        object.__setattr__(self, "mask_reason", reason)
        object.__setattr__(self, "missing_columns", tuple(str(column) for column in self.missing_columns))
        object.__setattr__(self, "observed_rows", int(self.observed_rows))
        object.__setattr__(self, "production_enabled", False)
        object.__setattr__(self, "production_outputs_written", False)
        object.__setattr__(self, "canonical_production_state_outputs_written", False)

    def as_dict(self) -> dict[str, Any]:
        return {
            "artifact_kind": "cross_asset_state_masked_cell",
            "schema_version": int(self.schema_version),
            "asset_id": self.asset_id,
            "relationship_feature_family": self.relationship_feature_family,
            "band": self.band,
            "ts": self.ts,
            "selected_status": "masked_unavailable",
            "mask_reason": self.mask_reason,
            "relationship_registry_mask_reason": self.relationship_registry_mask_reason,
            "relationship_snapshot_id": self.relationship_snapshot_id,
            "known_at_ts": self.known_at_ts,
            "source_tail_ts": self.source_tail_ts,
            "missing_columns": list(self.missing_columns),
            "observed_rows": int(self.observed_rows),
            "shape_preserving": True,
            "production_enabled": False,
            "production_outputs_written": False,
            "canonical_production_state_outputs_written": False,
        }


def masked_cell_from_relationship_record(
    record: RelationshipAvailabilityRecord | Mapping[str, Any],
    *,
    relationship_feature_family: str,
    mask_reason: str | None = None,
    missing_columns: tuple[str, ...] = (),
    observed_rows: int = 0,
) -> CrossAssetStateMaskedCell:
    data = record.as_dict() if hasattr(record, "as_dict") else dict(record)
    relationship_reason = data.get("mask_reason")
    reason = mask_reason
    if reason is None:
        reason = (
            CrossAssetStateMaskReason.ASSET_NOT_IN_RELATIONSHIP_SNAPSHOT
            if relationship_reason == "asset_not_in_snapshot"
            else CrossAssetStateMaskReason.RELATIONSHIP_SNAPSHOT_UNAVAILABLE
        )
    return CrossAssetStateMaskedCell(
        asset_id=str(data.get("asset_id", "")),
        relationship_feature_family=relationship_feature_family,
        band=str(data.get("band", "")),
        ts=data.get("ts"),
        mask_reason=reason,
        relationship_snapshot_id=data.get("snapshot_id"),
        known_at_ts=data.get("known_at_ts"),
        source_tail_ts=data.get("source_tail_ts"),
        relationship_registry_mask_reason=str(relationship_reason) if relationship_reason else None,
        missing_columns=missing_columns,
        observed_rows=observed_rows,
    )
