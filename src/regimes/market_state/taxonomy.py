from __future__ import annotations

from src.regimes.market_state.contracts import (
    MARKET_STATE_SCHEMA_VERSION,
    MarketStateArtifactBoundary,
    MarketStateAxis,
    MarketStateAxisSpec,
    MarketStateBand,
    MarketStateBandSpec,
    MarketStateOutputSchema,
    MarketStateTaxonomyManifest,
)


def default_market_state_band_specs() -> tuple[MarketStateBandSpec, ...]:
    return (
        MarketStateBandSpec(
            band=MarketStateBand.MICRO,
            ceiling_interval_min=30,
            member_intervals=(1, 5, 15, 30),
            train_days=30,
            validation_horizons_min=(30, 60, 240),
        ),
        MarketStateBandSpec(
            band=MarketStateBand.MESO,
            ceiling_interval_min=240,
            member_intervals=(60, 240),
            train_days=180,
            validation_horizons_min=(240, 720, 1440),
        ),
        MarketStateBandSpec(
            band=MarketStateBand.MACRO,
            ceiling_interval_min=1440,
            member_intervals=(720, 1440),
            train_days=360,
            validation_horizons_min=(1440, 4320, 10080),
        ),
    )


def default_market_state_axis_specs() -> tuple[MarketStateAxisSpec, ...]:
    all_bands = (MarketStateBand.MICRO, MarketStateBand.MESO, MarketStateBand.MACRO)
    return (
        MarketStateAxisSpec(
            axis_id=MarketStateAxis.MARKET_RETURN,
            purpose="Broad market direction and risk-on/risk-off return behavior across the universe.",
            compatible_bands=all_bands,
            expected_feature_families=("market_return_aggregate", "core_basket_return_confirmation"),
            expected_validation_targets=("forward_market_return", "risk_on_off_continuation", "drawdown_breadth"),
            allow_single_state_output=True,
            requires_covariance_correlation_features=False,
            requires_broad_universe_aggregates=True,
            requires_core_basket_features=True,
            label_set=("risk_off", "neutral", "risk_on"),
        ),
        MarketStateAxisSpec(
            axis_id=MarketStateAxis.MARKET_VOLATILITY,
            purpose="Market-wide volatility compression, expansion, and shock behavior.",
            compatible_bands=all_bands,
            expected_feature_families=("market_realized_volatility", "volatility_of_volatility"),
            expected_validation_targets=("forward_market_realized_volatility", "volatility_regime_persistence", "shock_continuation"),
            allow_single_state_output=True,
            requires_covariance_correlation_features=False,
            requires_broad_universe_aggregates=True,
            requires_core_basket_features=True,
            label_set=("compression", "normal", "expansion", "shock"),
        ),
        MarketStateAxisSpec(
            axis_id=MarketStateAxis.MARKET_BREADTH,
            purpose="Participation breadth, advance/decline structure, and broad confirmation.",
            compatible_bands=all_bands,
            expected_feature_families=("market_breadth_participation", "advance_decline_structure"),
            expected_validation_targets=("forward_breadth_persistence", "forward_market_return", "participation_continuity"),
            allow_single_state_output=True,
            requires_covariance_correlation_features=False,
            requires_broad_universe_aggregates=True,
            requires_core_basket_features=False,
            label_set=("narrow_decline", "mixed", "broad_advance", "broad_decline"),
        ),
        MarketStateAxisSpec(
            axis_id=MarketStateAxis.MARKET_DISPERSION,
            purpose="Cross-sectional return and volatility spread, disagreement, and opportunity dispersion.",
            compatible_bands=all_bands,
            expected_feature_families=("market_cross_sectional_dispersion", "return_volatility_disagreement"),
            expected_validation_targets=("forward_dispersion", "forward_market_realized_volatility", "sector_rotation"),
            allow_single_state_output=False,
            requires_covariance_correlation_features=False,
            requires_broad_universe_aggregates=True,
            requires_core_basket_features=False,
            label_set=("compressed", "normal", "dispersed", "extreme_disagreement"),
        ),
        MarketStateAxisSpec(
            axis_id=MarketStateAxis.MARKET_CORRELATION,
            purpose="Correlation concentration, diversification failure, and correlation shock.",
            compatible_bands=all_bands,
            expected_feature_families=("market_correlation_structure", "covariance_concentration"),
            expected_validation_targets=("forward_correlation", "diversification_failure", "stress_continuation"),
            allow_single_state_output=False,
            requires_covariance_correlation_features=True,
            requires_broad_universe_aggregates=True,
            requires_core_basket_features=True,
            label_set=("diversified", "normal", "concentrated", "correlation_shock"),
        ),
        MarketStateAxisSpec(
            axis_id=MarketStateAxis.MARKET_LIQUIDITY_ACTIVITY,
            purpose="Broad activity, volume and trade participation, and concentration of market activity.",
            compatible_bands=all_bands,
            expected_feature_families=("market_liquidity_activity", "volume_trade_participation_concentration"),
            expected_validation_targets=("forward_activity_continuity", "liquidity_participation", "concentration_persistence"),
            allow_single_state_output=True,
            requires_covariance_correlation_features=False,
            requires_broad_universe_aggregates=True,
            requires_core_basket_features=True,
            label_set=("quiet", "normal", "active", "concentrated_activity"),
        ),
        MarketStateAxisSpec(
            axis_id=MarketStateAxis.MARKET_STRESS,
            purpose="Drawdown breadth, downside tail participation, and high-volatility/high-correlation stress.",
            compatible_bands=all_bands,
            expected_feature_families=("market_drawdown_breadth", "tail_stress_correlation_volatility"),
            expected_validation_targets=("forward_drawdown_breadth", "tail_loss_participation", "stress_persistence"),
            allow_single_state_output=True,
            requires_covariance_correlation_features=True,
            requires_broad_universe_aggregates=True,
            requires_core_basket_features=True,
            label_set=("calm", "fragile", "stress", "capitulation"),
        ),
    )


def default_market_state_artifact_boundary() -> MarketStateArtifactBoundary:
    return MarketStateArtifactBoundary()


def default_market_state_output_schema() -> MarketStateOutputSchema:
    return MarketStateOutputSchema(
        key_columns=("ts", "pathway", "universe", "axis", "band", "ceiling_interval_min"),
        partition_columns=("axis", "band", "universe", "year", "month"),
        state_columns=("market_state_id", "market_state_label", "market_state_confidence", "market_state_intensity"),
        diagnostic_columns=(
            "feature_schema_hash",
            "dataset_id",
            "profile_id",
            "validation_status",
            "member_asset_count",
            "contributing_asset_count",
            "non_production_artifact",
        ),
        artifact_boundary=default_market_state_artifact_boundary(),
    )


def default_market_state_taxonomy() -> MarketStateTaxonomyManifest:
    axes = {spec.axis_id: spec for spec in default_market_state_axis_specs()}
    bands = {spec.band: spec for spec in default_market_state_band_specs()}
    return MarketStateTaxonomyManifest(
        axes=axes,
        bands=bands,
        output_schema=default_market_state_output_schema(),
        artifact_boundary=default_market_state_artifact_boundary(),
        metadata={
            "schema_version_note": f"market-state taxonomy schema v{MARKET_STATE_SCHEMA_VERSION}",
            "pathway_boundary": "whole-market state labels only; no per-asset, relative, cross-asset, or production execution",
        },
    )


def validate_market_state_axis_band(axis: str | MarketStateAxis, band: str | MarketStateBand) -> None:
    default_market_state_taxonomy().validate_axis_band(axis, band)


__all__ = [
    "default_market_state_artifact_boundary",
    "default_market_state_axis_specs",
    "default_market_state_band_specs",
    "default_market_state_output_schema",
    "default_market_state_taxonomy",
    "validate_market_state_axis_band",
]
