from __future__ import annotations

from typing import Any, Mapping, Sequence

from src.regimes.asset_state.feature_pools import AssetStateFeaturePoolRegistry, default_asset_state_feature_pool_registry
from src.regimes.asset_state.taxonomy import default_asset_state_taxonomy
from src.regimes.core.band_composites import (
    ALIGNMENT_POLICY_CEILING_BOUNDARY,
    BAND_COMPOSITE_SCHEMA_VERSION,
    BandCompositeSpec,
    band_composite_registry_as_dict,
    default_relationship_feature_permissions,
    resolve_band_composite_spec,
)


ASSET_STATE_BAND_COMPOSITE_POLICY_ID = "asset_state_band_composite_policy_v1"


def asset_state_band_composite_specs(
    *,
    feature_pool_registry: AssetStateFeaturePoolRegistry | None = None,
    relationship_feature_permissions: Mapping[str, Any] | None = None,
) -> tuple[BandCompositeSpec, ...]:
    taxonomy = default_asset_state_taxonomy()
    allowed_by_band = _allowed_feature_pools_by_band(feature_pool_registry or default_asset_state_feature_pool_registry())
    permissions = relationship_feature_permissions or default_relationship_feature_permissions()
    return tuple(
        BandCompositeSpec(
            band=band,
            ceiling_interval=int(spec.ceiling_interval_min),
            member_intervals=spec.member_intervals,
            output_cadence=int(spec.ceiling_interval_min),
            alignment_policy=ALIGNMENT_POLICY_CEILING_BOUNDARY,
            allowed_feature_families=allowed_by_band.get(band, ()),
            relationship_feature_permissions=permissions,
            metadata={
                "source": "asset_state_taxonomy",
                "train_days": int(spec.train_days),
                "validation_horizons_min": list(spec.validation_horizons_min),
                "preserves_existing_taxonomy_band_contract": True,
            },
        )
        for band, spec in sorted(taxonomy.bands.items())
    )


def asset_state_band_composite_policy(
    *,
    feature_pool_registry: AssetStateFeaturePoolRegistry | None = None,
) -> dict[str, Any]:
    specs = asset_state_band_composite_specs(feature_pool_registry=feature_pool_registry)
    payload = band_composite_registry_as_dict(specs)
    payload.update(
        {
            "policy_id": ASSET_STATE_BAND_COMPOSITE_POLICY_ID,
            "pathway": "asset_state",
            "schema_version": BAND_COMPOSITE_SCHEMA_VERSION,
            "pairwise_execution_enabled": False,
            "cross_asset_execution_enabled": False,
        }
    )
    return payload


def resolve_asset_state_band_composite(
    band: str,
    *,
    feature_pool_registry: AssetStateFeaturePoolRegistry | None = None,
) -> BandCompositeSpec:
    return resolve_band_composite_spec(
        band,
        specs=asset_state_band_composite_specs(feature_pool_registry=feature_pool_registry),
    )


def _allowed_feature_pools_by_band(registry: AssetStateFeaturePoolRegistry) -> dict[str, tuple[str, ...]]:
    payload: dict[str, list[str]] = {"micro": [], "meso": [], "macro": []}
    for spec in registry.pools.values():
        for band in spec.compatible_bands:
            payload[str(band)].append(spec.feature_pool_id)
    return {band: tuple(sorted(set(pool_ids))) for band, pool_ids in payload.items()}


__all__ = [
    "ASSET_STATE_BAND_COMPOSITE_POLICY_ID",
    "asset_state_band_composite_policy",
    "asset_state_band_composite_specs",
    "resolve_asset_state_band_composite",
]
