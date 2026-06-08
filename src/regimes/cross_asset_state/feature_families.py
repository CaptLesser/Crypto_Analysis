from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence


CROSS_ASSET_STATE_SCHEMA_VERSION = 1
CROSS_ASSET_STATE_FEATURE_SET_ORIGINAL = "cross_asset_v1_original"
CROSS_ASSET_STATE_FEATURE_SET_REPAIRED_SANDBOX = "cross_asset_v1_repaired_sandbox"
CROSS_ASSET_STATE_FEATURE_SET_REPAIRED_VARIABLE_PEER_SANDBOX = "cross_asset_v1_repaired_variable_peer_sandbox"

SUPPORTED_BANDS: tuple[str, ...] = ("meso", "macro")
SUPPORTED_FEATURE_SET_VERSIONS: tuple[str, ...] = (
    CROSS_ASSET_STATE_FEATURE_SET_ORIGINAL,
    CROSS_ASSET_STATE_FEATURE_SET_REPAIRED_SANDBOX,
    CROSS_ASSET_STATE_FEATURE_SET_REPAIRED_VARIABLE_PEER_SANDBOX,
)


@dataclass(frozen=True)
class CrossAssetStateFeatureFamilySpec:
    name: str
    required_columns: tuple[str, ...]
    method_family: str = "sandbox_kmeans_v1"
    model_facing_v1: bool = True
    feature_set_version: str = CROSS_ASSET_STATE_FEATURE_SET_ORIGINAL

    def __post_init__(self) -> None:
        if not self.name or not str(self.name).strip():
            raise ValueError("Cross-Asset-State feature family requires a name")
        columns = tuple(str(column).strip() for column in self.required_columns if str(column).strip())
        if not columns:
            raise ValueError("Cross-Asset-State feature family requires at least one feature column")
        object.__setattr__(self, "name", str(self.name).strip())
        object.__setattr__(self, "required_columns", columns)
        object.__setattr__(self, "method_family", str(self.method_family).strip())
        object.__setattr__(self, "model_facing_v1", bool(self.model_facing_v1))
        feature_set = str(self.feature_set_version).strip()
        if feature_set not in SUPPORTED_FEATURE_SET_VERSIONS:
            raise ValueError(f"Unsupported Cross-Asset-State feature_set_version {feature_set!r}")
        object.__setattr__(self, "feature_set_version", feature_set)

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "required_columns": list(self.required_columns),
            "method_family": self.method_family,
            "model_facing_v1": bool(self.model_facing_v1),
            "feature_set_version": self.feature_set_version,
        }


DEFAULT_CROSS_ASSET_STATE_FEATURE_FAMILIES: tuple[CrossAssetStateFeatureFamilySpec, ...] = (
    CrossAssetStateFeatureFamilySpec(
        name="anchor_core_exposure",
        required_columns=(
            "corr_to_anchor_primary",
            "corr_to_anchor_secondary",
            "corr_to_core_basket",
            "beta_to_core_basket",
        ),
    ),
    CrossAssetStateFeatureFamilySpec(
        name="peer_strength_stability",
        required_columns=(
            "strongest_peer_slot_1_strength",
            "top_peer_count",
            "top_peer_stability_mean",
        ),
    ),
    CrossAssetStateFeatureFamilySpec(
        name="relationship_concentration_entropy",
        required_columns=(
            "relationship_concentration",
            "relationship_entropy",
        ),
    ),
    CrossAssetStateFeatureFamilySpec(
        name="residual_peer_signal",
        required_columns=("residual_peer_signal_score",),
    ),
)

REPAIRED_SANDBOX_CROSS_ASSET_STATE_FEATURE_FAMILIES: tuple[CrossAssetStateFeatureFamilySpec, ...] = (
    CrossAssetStateFeatureFamilySpec(
        name="anchor_core_exposure",
        required_columns=(
            "corr_to_anchor_primary",
            "corr_to_anchor_secondary",
            "corr_to_core_basket",
            "beta_to_core_basket",
        ),
        feature_set_version=CROSS_ASSET_STATE_FEATURE_SET_REPAIRED_SANDBOX,
    ),
    CrossAssetStateFeatureFamilySpec(
        name="peer_strength_stability",
        required_columns=(
            "eligible_peer_count",
            "total_peer_strength",
            "avg_peer_strength",
            "top1_peer_strength",
            "top2_peer_strength",
            "top3_peer_strength",
            "top1_to_top3_ratio",
            "peer_strength_dispersion",
            "peer_strength_iqr",
            "peer_stability_dispersion",
            "persistence_weighted_peer_strength",
            "peer_membership_churn",
            "peer_set_jaccard_vs_prior_snapshot",
        ),
        feature_set_version=CROSS_ASSET_STATE_FEATURE_SET_REPAIRED_SANDBOX,
    ),
    CrossAssetStateFeatureFamilySpec(
        name="relationship_concentration_entropy",
        required_columns=(
            "max_peer_share",
            "top1_share",
            "top3_share",
            "hhi_all_eligible_peers",
            "effective_peer_count",
            "normalized_entropy_variable_support",
            "peer_weight_gini",
            "peer_count_to_50pct_mass",
            "peer_count_to_80pct_mass",
            "edge_weight_spread",
        ),
        feature_set_version=CROSS_ASSET_STATE_FEATURE_SET_REPAIRED_SANDBOX,
    ),
    CrossAssetStateFeatureFamilySpec(
        name="residual_peer_signal",
        required_columns=("residual_peer_signal_score",),
        feature_set_version=CROSS_ASSET_STATE_FEATURE_SET_REPAIRED_SANDBOX,
    ),
)

REPAIRED_VARIABLE_PEER_SANDBOX_CROSS_ASSET_STATE_FEATURE_FAMILIES: tuple[CrossAssetStateFeatureFamilySpec, ...] = (
    CrossAssetStateFeatureFamilySpec(
        name="anchor_core_exposure",
        required_columns=(
            "corr_to_anchor_primary",
            "corr_to_anchor_secondary",
            "corr_to_core_basket",
            "beta_to_core_basket",
        ),
        feature_set_version=CROSS_ASSET_STATE_FEATURE_SET_REPAIRED_VARIABLE_PEER_SANDBOX,
    ),
    CrossAssetStateFeatureFamilySpec(
        name="peer_strength_stability",
        required_columns=(
            "eligible_peer_count",
            "peer_count_above_threshold",
            "total_peer_strength",
            "avg_peer_strength",
            "median_peer_strength",
            "top1_peer_strength",
            "top2_peer_strength",
            "top1_to_total_ratio",
            "peer_strength_dispersion",
            "peer_strength_iqr",
            "peer_strength_slope_by_rank",
            "persistence_weighted_peer_strength",
        ),
        feature_set_version=CROSS_ASSET_STATE_FEATURE_SET_REPAIRED_VARIABLE_PEER_SANDBOX,
    ),
    CrossAssetStateFeatureFamilySpec(
        name="relationship_concentration_entropy",
        required_columns=(
            "max_peer_share",
            "top1_share",
            "hhi_all_eligible_peers",
            "effective_peer_count",
            "raw_entropy_variable_support",
            "normalized_entropy_variable_support",
            "peer_weight_gini",
            "peer_count_to_50pct_mass",
            "peer_count_to_80pct_mass",
            "edge_weight_spread",
            "edge_weight_iqr",
        ),
        feature_set_version=CROSS_ASSET_STATE_FEATURE_SET_REPAIRED_VARIABLE_PEER_SANDBOX,
    ),
    CrossAssetStateFeatureFamilySpec(
        name="residual_peer_signal",
        required_columns=("residual_peer_signal_score",),
        feature_set_version=CROSS_ASSET_STATE_FEATURE_SET_REPAIRED_VARIABLE_PEER_SANDBOX,
    ),
)


def default_feature_family_map(
    *,
    feature_set_version: str = CROSS_ASSET_STATE_FEATURE_SET_ORIGINAL,
) -> Mapping[str, CrossAssetStateFeatureFamilySpec]:
    feature_set = str(feature_set_version).strip()
    if feature_set == CROSS_ASSET_STATE_FEATURE_SET_ORIGINAL:
        specs = DEFAULT_CROSS_ASSET_STATE_FEATURE_FAMILIES
    elif feature_set == CROSS_ASSET_STATE_FEATURE_SET_REPAIRED_SANDBOX:
        specs = REPAIRED_SANDBOX_CROSS_ASSET_STATE_FEATURE_FAMILIES
    elif feature_set == CROSS_ASSET_STATE_FEATURE_SET_REPAIRED_VARIABLE_PEER_SANDBOX:
        specs = REPAIRED_VARIABLE_PEER_SANDBOX_CROSS_ASSET_STATE_FEATURE_FAMILIES
    else:
        raise ValueError(f"Unsupported Cross-Asset-State feature_set_version {feature_set!r}")
    return {spec.name: spec for spec in specs}


def resolve_feature_families(
    names: Sequence[str] | None = None,
    *,
    family_map: Mapping[str, CrossAssetStateFeatureFamilySpec] | None = None,
    feature_set_version: str = CROSS_ASSET_STATE_FEATURE_SET_ORIGINAL,
) -> tuple[CrossAssetStateFeatureFamilySpec, ...]:
    resolved = dict(family_map or default_feature_family_map(feature_set_version=feature_set_version))
    if names is None:
        return tuple(resolved.values())
    out: list[CrossAssetStateFeatureFamilySpec] = []
    for name in names:
        key = str(name).strip()
        if key not in resolved:
            raise ValueError(f"Unsupported Cross-Asset-State feature family {key!r}")
        out.append(resolved[key])
    return tuple(out)
