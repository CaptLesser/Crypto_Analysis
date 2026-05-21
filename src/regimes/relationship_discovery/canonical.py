from __future__ import annotations

from dataclasses import dataclass
from typing import Any


PROCESS_1_CANONICAL_MODULE = "src.regimes.relationship_discovery"
PROCESS_2_CANONICAL_MODULE = "src.regimes.regime_features.cross_asset_features"

LEGACY_PAIRWISE_SCAFFOLD_MODULE = "src.regimes.regime_features.pairwise"
LEGACY_CROSS_ASSET_SCAFFOLD_MODULE = "src.regimes.regime_features.cross_asset"

RELATIONSHIP_DISCOVERY_V1_REPORT_SUBDIR = "relationship_discovery_v1"

ARTIFACT_METHOD_MANIFEST = "method_manifest"
ARTIFACT_REFIT_SNAPSHOT_MANIFEST = "refit_snapshot_manifest"
ARTIFACT_SELECTED_RELATIONSHIP_EDGES = "selected_relationship_edges"
ARTIFACT_ASSET_RELATIONSHIP_PROFILES = "asset_relationship_profiles"
ARTIFACT_RELATIONSHIP_STABILITY_SCORES = "relationship_stability_scores"
ARTIFACT_ISOLATED_ASSET_PROFILES = "isolated_asset_profiles"
ARTIFACT_EDGE_ALIAS_MANIFEST = "edge_alias_manifest"
ARTIFACT_RELATIONSHIP_SCOREBOARD = "relationship_scoreboard"

ARTIFACT_CROSS_ASSET_FEATURE_ROWS = "cross_asset_feature_rows"
ARTIFACT_CROSS_ASSET_FEATURE_MANIFEST = "cross_asset_feature_manifest"
ARTIFACT_RELATIONSHIP_FEATURE_CATALOG = "relationship_feature_catalog"
ARTIFACT_PROCESS1_TO_PROCESS2_HANDOFF_MANIFEST = "process1_to_process2_handoff_manifest"

RELATIONSHIP_FAMILY_MARKET_EXPOSURE = "market_exposure"
RELATIONSHIP_FAMILY_RESIDUAL_PEER = "residual_peer"
RELATIONSHIP_FAMILY_RISK_NEIGHBORHOOD = "risk_neighborhood"
RELATIONSHIP_FAMILY_ISOLATION_STATUS = "isolation_status"

FEATURE_CORR_TO_ANCHOR_PRIMARY = "corr_to_anchor_primary"
FEATURE_CORR_TO_ANCHOR_SECONDARY = "corr_to_anchor_secondary"
FEATURE_CORR_TO_CORE_BASKET = "corr_to_core_basket"
FEATURE_BETA_TO_CORE_BASKET = "beta_to_core_basket"
FEATURE_MARKET_MODE_EXPOSURE_SCORE = "market_mode_exposure_score"
FEATURE_RESIDUAL_PEER_SIGNAL_SCORE = "residual_peer_signal_score"
FEATURE_RELATIONSHIP_CONCENTRATION = "relationship_concentration"
FEATURE_RELATIONSHIP_ENTROPY = "relationship_entropy"
FEATURE_TOP_PEER_COUNT = "top_peer_count"
FEATURE_TOP_PEER_STABILITY_MEAN = "top_peer_stability_mean"
FEATURE_STRONGEST_PEER_SLOT_1_STRENGTH = "strongest_peer_slot_1_strength"
FEATURE_STRONGEST_PEER_SLOT_1_ALIAS = "strongest_peer_slot_1_alias"
FEATURE_ISOLATED_ASSET_SCORE = "isolated_asset_score"
FEATURE_PEER_SIGNAL_AVAILABILITY_STATUS = "peer_signal_availability_status"

V1_FEATURE_NAMES: tuple[str, ...] = (
    FEATURE_CORR_TO_ANCHOR_PRIMARY,
    FEATURE_CORR_TO_ANCHOR_SECONDARY,
    FEATURE_CORR_TO_CORE_BASKET,
    FEATURE_BETA_TO_CORE_BASKET,
    FEATURE_MARKET_MODE_EXPOSURE_SCORE,
    FEATURE_RELATIONSHIP_CONCENTRATION,
    FEATURE_RELATIONSHIP_ENTROPY,
    FEATURE_TOP_PEER_COUNT,
    FEATURE_ISOLATED_ASSET_SCORE,
    FEATURE_PEER_SIGNAL_AVAILABILITY_STATUS,
)

V1_IF_STABLE_FEATURE_NAMES: tuple[str, ...] = (
    FEATURE_RESIDUAL_PEER_SIGNAL_SCORE,
    FEATURE_TOP_PEER_STABILITY_MEAN,
    FEATURE_STRONGEST_PEER_SLOT_1_STRENGTH,
)

SIDECAR_FEATURE_NAMES: tuple[str, ...] = (FEATURE_STRONGEST_PEER_SLOT_1_ALIAS,)


@dataclass(frozen=True)
class RelationshipDiscoveryCanonicalOwnership:
    process_1_module: str = PROCESS_1_CANONICAL_MODULE
    process_2_module: str = PROCESS_2_CANONICAL_MODULE
    legacy_pairwise_module: str = LEGACY_PAIRWISE_SCAFFOLD_MODULE
    legacy_cross_asset_module: str = LEGACY_CROSS_ASSET_SCAFFOLD_MODULE

    def as_dict(self) -> dict[str, Any]:
        return {
            "artifact_kind": "relationship_discovery_canonical_ownership",
            "process_1_canonical_module": self.process_1_module,
            "process_2_canonical_module": self.process_2_module,
            "legacy_pairwise_scaffold_module": self.legacy_pairwise_module,
            "legacy_cross_asset_scaffold_module": self.legacy_cross_asset_module,
            "legacy_surfaces_remain_gated": True,
            "dynamic_peer_clusters_v1": False,
            "broad_all_to_all_production_v1": False,
            "production_enabled": False,
        }


def canonical_ownership() -> RelationshipDiscoveryCanonicalOwnership:
    return RelationshipDiscoveryCanonicalOwnership()


__all__ = [
    "ARTIFACT_ASSET_RELATIONSHIP_PROFILES",
    "ARTIFACT_CROSS_ASSET_FEATURE_MANIFEST",
    "ARTIFACT_CROSS_ASSET_FEATURE_ROWS",
    "ARTIFACT_EDGE_ALIAS_MANIFEST",
    "ARTIFACT_ISOLATED_ASSET_PROFILES",
    "ARTIFACT_METHOD_MANIFEST",
    "ARTIFACT_PROCESS1_TO_PROCESS2_HANDOFF_MANIFEST",
    "ARTIFACT_REFIT_SNAPSHOT_MANIFEST",
    "ARTIFACT_RELATIONSHIP_FEATURE_CATALOG",
    "ARTIFACT_RELATIONSHIP_SCOREBOARD",
    "ARTIFACT_RELATIONSHIP_STABILITY_SCORES",
    "ARTIFACT_SELECTED_RELATIONSHIP_EDGES",
    "FEATURE_BETA_TO_CORE_BASKET",
    "FEATURE_CORR_TO_ANCHOR_PRIMARY",
    "FEATURE_CORR_TO_ANCHOR_SECONDARY",
    "FEATURE_CORR_TO_CORE_BASKET",
    "FEATURE_ISOLATED_ASSET_SCORE",
    "FEATURE_MARKET_MODE_EXPOSURE_SCORE",
    "FEATURE_PEER_SIGNAL_AVAILABILITY_STATUS",
    "FEATURE_RELATIONSHIP_CONCENTRATION",
    "FEATURE_RELATIONSHIP_ENTROPY",
    "FEATURE_RESIDUAL_PEER_SIGNAL_SCORE",
    "FEATURE_STRONGEST_PEER_SLOT_1_ALIAS",
    "FEATURE_STRONGEST_PEER_SLOT_1_STRENGTH",
    "FEATURE_TOP_PEER_COUNT",
    "FEATURE_TOP_PEER_STABILITY_MEAN",
    "PROCESS_1_CANONICAL_MODULE",
    "PROCESS_2_CANONICAL_MODULE",
    "RELATIONSHIP_DISCOVERY_V1_REPORT_SUBDIR",
    "RELATIONSHIP_FAMILY_ISOLATION_STATUS",
    "RELATIONSHIP_FAMILY_MARKET_EXPOSURE",
    "RELATIONSHIP_FAMILY_RESIDUAL_PEER",
    "RELATIONSHIP_FAMILY_RISK_NEIGHBORHOOD",
    "SIDECAR_FEATURE_NAMES",
    "V1_FEATURE_NAMES",
    "V1_IF_STABLE_FEATURE_NAMES",
    "RelationshipDiscoveryCanonicalOwnership",
    "canonical_ownership",
]
