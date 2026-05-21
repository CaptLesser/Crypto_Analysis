from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from src.regimes.core.serialization import to_jsonable


AXIS_MARKET_RETURN_STATE = "market_return_state"
AXIS_MARKET_VOLATILITY_STATE = "market_volatility_state"
AXIS_MARKET_BREADTH_STATE = "market_breadth_state"
AXIS_MARKET_DISPERSION_STATE = "market_dispersion_state"
AXIS_MARKET_CORRELATION_STATE = "market_correlation_state"
AXIS_MARKET_LIQUIDITY_ACTIVITY_STATE = "market_liquidity_activity_state"
AXIS_MARKET_STRESS_STATE = "market_stress_state"
AXIS_STABLE_PEG_STRESS_STATE = "stable_peg_stress_state"
AXIS_MARKET_SPECULATIVE_STATE = "market_speculative_state"
MARKET_STATE_V1_AXIS_IDS: tuple[str, ...] = (
    AXIS_MARKET_RETURN_STATE,
    AXIS_MARKET_VOLATILITY_STATE,
    AXIS_MARKET_BREADTH_STATE,
    AXIS_MARKET_DISPERSION_STATE,
    AXIS_MARKET_CORRELATION_STATE,
    AXIS_MARKET_LIQUIDITY_ACTIVITY_STATE,
    AXIS_MARKET_STRESS_STATE,
    AXIS_STABLE_PEG_STRESS_STATE,
    AXIS_MARKET_SPECULATIVE_STATE,
)

AXIS_PANEL_UNAVAILABLE_EMPTY = "empty_panel_unavailable"
AXIS_PANEL_UNAVAILABLE_REQUIRED_MISSING = "required_features_missing"
AXIS_PANEL_UNAVAILABLE_FAMILY_MISSING = "source_feature_family_missing"


@dataclass(frozen=True)
class MarketStateAxisContract:
    axis_id: str
    source_feature_families: Sequence[str]
    required_features: Sequence[str]
    optional_features: Sequence[str] = ()
    compatible_bands: Sequence[str] = ("micro", "meso", "macro")
    unavailable_behavior: str = AXIS_PANEL_UNAVAILABLE_REQUIRED_MISSING
    ordinal_feature: str | None = None
    ordinal_labels: Sequence[str] = ("low", "neutral", "high")
    ordinal_direction: str = "higher_is_higher_state"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        axis = _text(self.axis_id, "axis_id")
        families = _string_tuple(self.source_feature_families, "source_feature_families", require_non_empty=True)
        required = _string_tuple(self.required_features, "required_features", require_non_empty=True)
        optional = _string_tuple(self.optional_features, "optional_features", require_non_empty=False)
        bands = tuple(str(band).strip().lower() for band in self.compatible_bands if str(band).strip())
        if not bands:
            raise ValueError("Market-State axis contract compatible_bands must be non-empty")
        labels = _string_tuple(self.ordinal_labels, "ordinal_labels", require_non_empty=True)
        if len(labels) != 3:
            raise ValueError("Market-State axis contract ordinal_labels must contain three labels")
        direction = _text(self.ordinal_direction, "ordinal_direction")
        if direction not in {"higher_is_higher_state", "lower_is_higher_state"}:
            raise ValueError("Market-State axis contract ordinal_direction must be higher_is_higher_state or lower_is_higher_state")
        object.__setattr__(self, "axis_id", axis)
        object.__setattr__(self, "source_feature_families", families)
        object.__setattr__(self, "required_features", required)
        object.__setattr__(self, "optional_features", optional)
        object.__setattr__(self, "compatible_bands", bands)
        object.__setattr__(self, "unavailable_behavior", _text(self.unavailable_behavior, "unavailable_behavior"))
        object.__setattr__(self, "ordinal_feature", None if self.ordinal_feature is None else _text(self.ordinal_feature, "ordinal_feature"))
        object.__setattr__(self, "ordinal_labels", labels)
        object.__setattr__(self, "ordinal_direction", direction)
        object.__setattr__(self, "metadata", to_jsonable(dict(self.metadata)))

    @property
    def feature_columns(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys((*self.required_features, *self.optional_features)))

    def as_dict(self) -> dict[str, Any]:
        return {
            "axis_id": self.axis_id,
            "source_feature_families": list(self.source_feature_families),
            "required_features": list(self.required_features),
            "optional_features": list(self.optional_features),
            "compatible_bands": list(self.compatible_bands),
            "unavailable_behavior": self.unavailable_behavior,
            "ordinal_feature": self.ordinal_feature,
            "ordinal_labels": list(self.ordinal_labels),
            "ordinal_direction": self.ordinal_direction,
            "metadata": to_jsonable(dict(self.metadata)),
        }


def default_market_state_v1_axis_contracts() -> dict[str, MarketStateAxisContract]:
    contracts = (
        MarketStateAxisContract(
            axis_id=AXIS_MARKET_RETURN_STATE,
            source_feature_families=("market_return_summary",),
            required_features=("core_equal_weight_return", "anchor_equal_weight_return", "anchor_vs_core_leadership_spread"),
            optional_features=("core_liquidity_weighted_return", "broad_equal_weight_return", "core_return_zscore"),
            ordinal_feature="core_equal_weight_return",
            ordinal_labels=("low", "neutral", "high"),
        ),
        MarketStateAxisContract(
            axis_id=AXIS_MARKET_VOLATILITY_STATE,
            source_feature_families=("market_realized_volatility",),
            required_features=("core_realized_volatility", "core_median_asset_volatility"),
            optional_features=("core_asset_volatility_q25", "core_asset_volatility_q75", "core_asset_volatility_q90", "vol_expansion_zscore", "vol_of_vol", "core_range_based_volatility_fallback"),
            ordinal_feature="core_realized_volatility",
            ordinal_labels=("compression", "neutral", "expansion"),
        ),
        MarketStateAxisContract(
            axis_id=AXIS_MARKET_BREADTH_STATE,
            source_feature_families=("market_breadth",),
            required_features=("broad_advance_fraction", "broad_decline_fraction", "share_positive_return", "share_negative_return"),
            optional_features=("drawdown_recovery_breadth", "moving_average_breadth_2", "moving_average_breadth_3"),
            ordinal_feature="broad_advance_fraction",
            ordinal_labels=("weak", "mixed", "broad"),
        ),
        MarketStateAxisContract(
            axis_id=AXIS_MARKET_DISPERSION_STATE,
            source_feature_families=("market_dispersion",),
            required_features=("cross_sectional_return_std", "robust_return_dispersion_mad_or_iqr", "return_q90_q10_spread"),
            optional_features=("breadth_adjusted_dispersion", "core_cross_sectional_return_std", "broad_minus_core_dispersion"),
            ordinal_feature="cross_sectional_return_std",
            ordinal_labels=("diffuse", "mixed", "concentrated"),
        ),
        MarketStateAxisContract(
            axis_id=AXIS_MARKET_CORRELATION_STATE,
            source_feature_families=("market_covariance_summary",),
            required_features=("median_pairwise_correlation", "average_offdiag_correlation", "pc1_share"),
            optional_features=("correlation_q10", "correlation_q90", "spearman_median_pairwise_correlation", "top_eigenvalue", "absorption_ratio", "effective_rank", "correlation_distance_to_long_run_baseline"),
            ordinal_feature="pc1_share",
            ordinal_labels=("diffuse", "mixed", "concentrated"),
        ),
        MarketStateAxisContract(
            axis_id=AXIS_MARKET_LIQUIDITY_ACTIVITY_STATE,
            source_feature_families=("market_liquidity_activity",),
            required_features=("aggregate_dollar_volume", "median_dollar_volume", "activity_breadth"),
            optional_features=("trade_count_breadth", "volume_concentration", "trade_count_concentration", "amihud_illiquidity_proxy", "empty_no_trade_bar_share"),
            ordinal_feature="activity_breadth",
            ordinal_labels=("low", "neutral", "high"),
        ),
        MarketStateAxisContract(
            axis_id=AXIS_MARKET_STRESS_STATE,
            source_feature_families=("market_stress",),
            required_features=("core_market_drawdown", "broad_drawdown_breadth", "downside_semivariance"),
            optional_features=("high_vol_high_corr_coincidence", "turbulence_zscore"),
            ordinal_feature="core_market_drawdown",
            ordinal_labels=("low", "neutral", "high"),
            ordinal_direction="lower_is_higher_state",
        ),
        MarketStateAxisContract(
            axis_id=AXIS_STABLE_PEG_STRESS_STATE,
            source_feature_families=("stable_peg_stress",),
            required_features=("peg_deviation_abs", "stable_stress_breadth", "stable_panel_coverage"),
            optional_features=("stable_activity_share", "stable_volume_share", "stable_basis_zscore"),
            ordinal_feature="peg_deviation_abs",
            ordinal_labels=("low", "neutral", "high"),
        ),
        MarketStateAxisContract(
            axis_id=AXIS_MARKET_SPECULATIVE_STATE,
            source_feature_families=("speculative_satellite_sidecar",),
            required_features=(
                "speculative_return_dispersion",
                "speculative_vs_clean_broad_return_spread",
                "speculative_volume_share",
            ),
            optional_features=(
                "speculative_advance_fraction",
                "speculative_activity_breadth",
                "speculative_sample_count",
                "speculative_coverage",
            ),
            ordinal_feature="speculative_return_dispersion",
            ordinal_labels=("calm", "mixed", "heated"),
            metadata={"test_candidate_only": True, "production_promotion": False},
        ),
    )
    return {contract.axis_id: contract for contract in contracts}


def _string_tuple(values: Sequence[object], field_name: str, *, require_non_empty: bool) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        values = (values,)
    out = tuple(dict.fromkeys(_text(value, field_name) for value in values if str(value).strip()))
    if require_non_empty and not out:
        raise ValueError(f"Market-State axis contract {field_name} must be non-empty")
    return out


def _text(value: object, field_name: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"Market-State axis contract {field_name} must be non-empty")
    return text


__all__ = [
    "AXIS_MARKET_BREADTH_STATE",
    "AXIS_MARKET_CORRELATION_STATE",
    "AXIS_MARKET_DISPERSION_STATE",
    "AXIS_MARKET_LIQUIDITY_ACTIVITY_STATE",
    "AXIS_MARKET_RETURN_STATE",
    "AXIS_MARKET_SPECULATIVE_STATE",
    "AXIS_MARKET_STRESS_STATE",
    "AXIS_MARKET_VOLATILITY_STATE",
    "AXIS_PANEL_UNAVAILABLE_EMPTY",
    "AXIS_PANEL_UNAVAILABLE_FAMILY_MISSING",
    "AXIS_PANEL_UNAVAILABLE_REQUIRED_MISSING",
    "AXIS_STABLE_PEG_STRESS_STATE",
    "MARKET_STATE_V1_AXIS_IDS",
    "MarketStateAxisContract",
    "default_market_state_v1_axis_contracts",
]
