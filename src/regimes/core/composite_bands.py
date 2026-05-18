from __future__ import annotations

from typing import Any

from src.regimes.asset_state.taxonomy import default_asset_state_taxonomy
from src.regimes.core.band_composites import (
    ALIGNMENT_POLICY_CEILING_BOUNDARY,
    CROSS_ASSET_SUMMARY_FEATURES,
    PAIRWISE_RELATIONSHIP_FEATURES,
    BandCompositeSpec,
)
from src.regimes.market_state.taxonomy import default_market_state_taxonomy


ASSET_STATE_COMPOSITE_BAND_CONTRACT_ID = "asset_state_composite_band_contract_v1"
MARKET_STATE_COMPOSITE_BAND_CONTRACT_ID = "market_state_composite_band_contract_v1"


def asset_state_composite_band_contract() -> dict[str, Any]:
    from src.regimes.asset_state.band_policy import asset_state_band_composite_specs

    taxonomy = default_asset_state_taxonomy()
    return _contract(
        contract_id=ASSET_STATE_COMPOSITE_BAND_CONTRACT_ID,
        pathway="asset_state",
        bands=taxonomy.bands,
        composite_specs=asset_state_band_composite_specs(),
        key_columns=taxonomy.output_schema.key_columns,
        partition_columns=taxonomy.output_schema.partition_columns,
    )


def market_state_composite_band_contract() -> dict[str, Any]:
    from src.regimes.market_state.band_policy import market_state_band_composite_specs

    taxonomy = default_market_state_taxonomy()
    return _contract(
        contract_id=MARKET_STATE_COMPOSITE_BAND_CONTRACT_ID,
        pathway="market_state",
        bands=taxonomy.bands,
        composite_specs=market_state_band_composite_specs(),
        key_columns=taxonomy.output_schema.key_columns,
        partition_columns=taxonomy.output_schema.partition_columns,
    )


def validate_composite_band_contract_preserved(contract: dict[str, Any]) -> None:
    bands = contract.get("bands")
    if not isinstance(bands, dict) or set(bands) != {"micro", "meso", "macro"}:
        raise ValueError("Composite band contract must preserve micro, meso, and macro bands")
    expected = {
        "micro": {"ceiling_interval_min": 30, "member_intervals": [1, 5, 15, 30]},
        "meso": {"ceiling_interval_min": 240, "member_intervals": [60, 240]},
        "macro": {"ceiling_interval_min": 1440, "member_intervals": [720, 1440]},
    }
    for band, expected_payload in expected.items():
        payload = bands.get(band, {})
        if payload.get("ceiling_interval_min") != expected_payload["ceiling_interval_min"]:
            raise ValueError(f"Composite band contract changed {band} ceiling interval")
        if payload.get("member_intervals") != expected_payload["member_intervals"]:
            raise ValueError(f"Composite band contract changed {band} member intervals")
        if payload.get("composite_band") is not True:
            raise ValueError(f"Composite band contract must mark {band} as composite_band")
        if payload.get("output_cadence") != expected_payload["ceiling_interval_min"]:
            raise ValueError(f"Composite band contract changed {band} output cadence")
        if payload.get("alignment_policy") != ALIGNMENT_POLICY_CEILING_BOUNDARY:
            raise ValueError(f"Composite band contract changed {band} alignment policy")
        _validate_relationship_feature_permissions(payload.get("relationship_feature_permissions", {}), band=band)
    if contract.get("production_parquet_allowed") is not False:
        raise ValueError("Composite band contract must not allow production parquet writes")
    if "band" not in contract.get("key_columns", ()):
        raise ValueError("Composite band contract must preserve band in key columns")


def _contract(
    *,
    contract_id: str,
    pathway: str,
    bands: dict[str, Any],
    composite_specs: tuple[BandCompositeSpec, ...],
    key_columns: tuple[str, ...],
    partition_columns: tuple[str, ...],
) -> dict[str, Any]:
    specs = {spec.band: spec for spec in composite_specs}
    return {
        "artifact_kind": "regime_composite_band_contract",
        "contract_id": contract_id,
        "pathway": pathway,
        "bands": {
            band: {
                **specs[str(band)].as_dict(),
                "train_days": int(spec.train_days),
                "validation_horizons_min": list(spec.validation_horizons_min),
                "composite_band": True,
                "preserve_member_interval_identity": True,
                "preserve_ceiling_interval": True,
            }
            for band, spec in sorted(bands.items())
        },
        "key_columns": list(key_columns),
        "partition_columns": list(partition_columns),
        "band_key_preserved": "band" in set(key_columns),
        "ceiling_interval_min_preserved": "ceiling_interval_min" in set(key_columns),
        "production_parquet_allowed": False,
        "production_promotion_allowed": False,
    }


def _validate_relationship_feature_permissions(payload: Any, *, band: str) -> None:
    if not isinstance(payload, dict):
        raise ValueError(f"Composite band contract {band} relationship_feature_permissions must be a mapping")
    for family in (PAIRWISE_RELATIONSHIP_FEATURES, CROSS_ASSET_SUMMARY_FEATURES):
        permissions = payload.get(family)
        if not isinstance(permissions, dict):
            raise ValueError(f"Composite band contract {band} missing {family} permissions")
        if permissions.get("auto_inherit_member_intervals") is not False:
            raise ValueError(f"Composite band contract {band} must not auto-inherit member intervals for {family}")
        if permissions.get("execution_enabled") is not False:
            raise ValueError(f"Composite band contract {band} must keep {family} execution gated")
        if permissions.get("short_interval_execution_enabled") is not False:
            raise ValueError(f"Composite band contract {band} must keep short intervals gated for {family}")


__all__ = [
    "ASSET_STATE_COMPOSITE_BAND_CONTRACT_ID",
    "MARKET_STATE_COMPOSITE_BAND_CONTRACT_ID",
    "asset_state_composite_band_contract",
    "market_state_composite_band_contract",
    "validate_composite_band_contract_preserved",
]
