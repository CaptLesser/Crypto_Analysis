from __future__ import annotations

from src.regimes.asset_state.contracts import (
    ASSET_STATE_SCHEMA_VERSION,
    AssetStateAxis,
    AssetStateAxisSpec,
    AssetStateBand,
    AssetStateBandSpec,
    AssetStateOutputSchema,
    AssetStateTaxonomyManifest,
)


def default_asset_state_band_specs() -> tuple[AssetStateBandSpec, ...]:
    return (
        AssetStateBandSpec(
            band=AssetStateBand.MICRO,
            ceiling_interval_min=30,
            member_intervals=(1, 5, 15, 30),
            train_days=30,
            validation_horizons_min=(30, 60, 240),
        ),
        AssetStateBandSpec(
            band=AssetStateBand.MESO,
            ceiling_interval_min=240,
            member_intervals=(60, 240),
            train_days=180,
            validation_horizons_min=(240, 720, 1440),
        ),
        AssetStateBandSpec(
            band=AssetStateBand.MACRO,
            ceiling_interval_min=1440,
            member_intervals=(720, 1440),
            train_days=360,
            validation_horizons_min=(1440, 4320, 10080),
        ),
    )


def default_asset_state_axis_specs() -> tuple[AssetStateAxisSpec, ...]:
    all_bands = (AssetStateBand.MICRO, AssetStateBand.MESO, AssetStateBand.MACRO)
    return (
        AssetStateAxisSpec(
            axis_id=AssetStateAxis.TREND,
            purpose="Directional persistence: uptrend, downtrend, flat or neutral movement.",
            compatible_bands=all_bands,
            expected_feature_families=("asset_state_trend_core", "asset_state_directional_momentum"),
            expected_forward_validation_targets=("forward_return", "directional_hit_rate", "drawdown"),
            fallback_policy_hints={
                "policy": "allow_neutral_flat_single_state_when_movement_is_validly_near_zero",
                "neutral_label": "flat",
                "fail_closed_on_missing_directional_features": True,
            },
            allow_single_state_output=True,
            label_set=("downtrend", "flat", "uptrend"),
        ),
        AssetStateAxisSpec(
            axis_id=AssetStateAxis.VOLATILITY,
            purpose="Expansion and compression: low-vol compression, normal volatility, high-vol expansion, shock.",
            compatible_bands=all_bands,
            expected_feature_families=("asset_state_volatility_core", "asset_state_volatility_shape"),
            expected_forward_validation_targets=("realized_volatility", "forward_return", "drawdown"),
            fallback_policy_hints={
                "policy": "allow_normal_single_state_only_when_volatility_features_are_stable",
                "neutral_label": "normal",
                "fail_closed_on_missing_volatility_features": True,
            },
            allow_single_state_output=True,
            label_set=("compression", "normal", "expansion", "shock"),
        ),
        AssetStateAxisSpec(
            axis_id=AssetStateAxis.ACTIVITY,
            purpose="Participation and liquidity: volume, trade participation, liquidity participation, unusual activity.",
            compatible_bands=all_bands,
            expected_feature_families=("asset_state_activity_core", "asset_state_liquidity_participation"),
            expected_forward_validation_targets=("forward_return", "realized_volatility", "participation_continuity"),
            fallback_policy_hints={
                "policy": "allow_low_activity_single_state_when_activity_features_confirm_sparse_participation",
                "neutral_label": "normal",
                "fail_closed_on_missing_activity_features": True,
            },
            allow_single_state_output=True,
            label_set=("low_activity", "normal", "high_activity", "unusual_activity"),
        ),
        AssetStateAxisSpec(
            axis_id=AssetStateAxis.MEAN_REVERSION,
            purpose="Range-bound behavior, noisy chop, mean-reverting pressure, and trendlessness.",
            compatible_bands=all_bands,
            expected_feature_families=("asset_state_mean_reversion_core", "asset_state_chop_range"),
            expected_forward_validation_targets=("forward_return", "range_persistence", "realized_volatility"),
            fallback_policy_hints={
                "policy": "allow_chop_single_state_when_trend_and_range_features_show_valid_trendlessness",
                "neutral_label": "range_bound",
                "fail_closed_on_missing_reversion_features": True,
            },
            allow_single_state_output=True,
            label_set=("trendless", "range_bound", "mean_reverting", "breakout_prone"),
        ),
        AssetStateAxisSpec(
            axis_id=AssetStateAxis.DRAWDOWN,
            purpose="Downside pressure, stress, crash-like downside excursion, and recovery from stress.",
            compatible_bands=all_bands,
            expected_feature_families=("asset_state_drawdown_core", "asset_state_stress_downside"),
            expected_forward_validation_targets=("drawdown", "forward_return", "realized_volatility"),
            fallback_policy_hints={
                "policy": "allow_no_stress_single_state_when downside_features_have_no_material_excursion",
                "neutral_label": "no_stress",
                "fail_closed_on_missing_downside_features": True,
            },
            allow_single_state_output=True,
            label_set=("no_stress", "drawdown_pressure", "stress", "recovery"),
        ),
        AssetStateAxisSpec(
            axis_id=AssetStateAxis.RANGE_EFFICIENCY,
            purpose="Directional range efficiency versus noisy range, including runup and drawdown asymmetry.",
            compatible_bands=all_bands,
            expected_feature_families=("asset_state_range_efficiency_core", "asset_state_runup_drawdown_structure"),
            expected_forward_validation_targets=("forward_return", "drawdown", "range_efficiency_persistence"),
            fallback_policy_hints={
                "policy": "block_single_state_until efficiency_and_asymmetry_features_are_available",
                "neutral_label": "mixed",
                "fail_closed_on_missing_range_structure_features": True,
            },
            allow_single_state_output=False,
            label_set=("inefficient_chop", "mixed", "efficient_runup", "downside_asymmetric"),
        ),
    )


def default_asset_state_output_schema() -> AssetStateOutputSchema:
    return AssetStateOutputSchema(
        key_columns=("ts", "asset", "pathway", "axis", "band", "ceiling_interval_min"),
        partition_columns=("axis", "band", "asset", "year", "month"),
        state_columns=("regime_state_id", "regime_state_label", "regime_state_confidence", "regime_state_intensity"),
        diagnostic_columns=(
            "clusterability_status",
            "fallback_status",
            "profile_id",
            "profile_decision_status",
            "feature_schema_hash",
            "dataset_id",
            "non_production_artifact",
        ),
        production_flags={
            "production_output": False,
            "production_parquet_allowed": False,
            "production_regime_labels_allowed": False,
            "production_profile_promotion_allowed": False,
            "write_root_policy": "reports/regimes/foundation/asset_state_test only",
        },
    )


def default_asset_state_taxonomy() -> AssetStateTaxonomyManifest:
    axes = {spec.axis_id: spec for spec in default_asset_state_axis_specs()}
    bands = {spec.band: spec for spec in default_asset_state_band_specs()}
    return AssetStateTaxonomyManifest(
        axes=axes,
        bands=bands,
        output_schema=default_asset_state_output_schema(),
        metadata={
            "schema_version_note": f"asset-state Test foundation schema v{ASSET_STATE_SCHEMA_VERSION}",
            "production_boundary": "non-production foundation contract; no production Regime labels or parquet writes",
        },
    )


def validate_asset_state_axis_band(axis: str | AssetStateAxis, band: str | AssetStateBand) -> None:
    default_asset_state_taxonomy().validate_axis_band(axis, band)


__all__ = [
    "default_asset_state_axis_specs",
    "default_asset_state_band_specs",
    "default_asset_state_output_schema",
    "default_asset_state_taxonomy",
    "validate_asset_state_axis_band",
]
