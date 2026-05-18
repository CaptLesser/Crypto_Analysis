from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from src.regimes.asset_state.contracts import (
    ASSET_STATE_AXIS_VALUES,
    ASSET_STATE_BAND_VALUES,
    ASSET_STATE_SCHEMA_VERSION,
    AssetStateAxis,
    AssetStateBand,
    AssetStateSchemaVersion,
    _enum_value,
    _mapping,
    _non_empty_text,
    _schema_version,
    _string_tuple,
)
from src.regimes.asset_state.taxonomy import default_asset_state_taxonomy
from src.regimes.core.feature_preprocessing import FeatureFilterConfig
from src.regimes.core.serialization import dumps_json, to_jsonable


ASSET_STATE_FEATURE_POOL_SCHEMA_VERSION = ASSET_STATE_SCHEMA_VERSION
FEATURE_POOL_STYLES: tuple[str, ...] = (
    "compact_axis",
    "broad_axis",
    "interaction_or_compressed_axis",
)
EMBEDDING_OPTIONS: tuple[str, ...] = (
    "none",
    "pca",
    "factor_analysis",
    "feature_agglomeration",
)
LEAKAGE_COLUMN_TOKENS: tuple[str, ...] = (
    "future_",
    "forward_",
    "_target",
    "target_",
    "_label",
    "label_",
    "regime_",
)


def _policy_float(
    policy: Mapping[str, Any],
    key: str,
    default: float,
) -> float:
    value = policy.get(key, default)
    try:
        return float(value)
    except Exception as exc:
        raise ValueError(f"Feature pool policy {key!r} must be numeric") from exc


def _policy_int(
    policy: Mapping[str, Any],
    key: str,
    default: int,
) -> int:
    value = policy.get(key, default)
    try:
        return int(value)
    except Exception as exc:
        raise ValueError(f"Feature pool policy {key!r} must be an integer") from exc


def _normalize_style(style: object) -> str:
    text = str(style).strip().lower()
    if text not in FEATURE_POOL_STYLES:
        raise ValueError(f"Unsupported asset-state feature pool style {text!r}; expected one of: {', '.join(FEATURE_POOL_STYLES)}")
    return text


def _normalize_embeddings(values: Sequence[object]) -> tuple[str, ...]:
    out = tuple(str(value).strip().lower() for value in values if str(value).strip())
    if not out:
        raise ValueError("Asset-state feature pool embedding_options must be non-empty")
    invalid = [value for value in out if value not in EMBEDDING_OPTIONS]
    if invalid:
        raise ValueError(f"Unsupported asset-state embedding option {invalid[0]!r}; expected one of: {', '.join(EMBEDDING_OPTIONS)}")
    return tuple(dict.fromkeys(out))


def _reject_leakage_risk_columns(columns: Sequence[str], *, field_name: str) -> None:
    bad = [
        column
        for column in columns
        if any(token in str(column).lower() for token in LEAKAGE_COLUMN_TOKENS)
    ]
    if bad:
        raise ValueError(f"Asset-state feature pool {field_name} contains leakage-risk columns: {bad}")


@dataclass(frozen=True)
class AssetStateFeaturePoolSpec:
    feature_pool_id: str
    axis: str | AssetStateAxis
    required_source_columns: Sequence[str]
    optional_source_columns: Sequence[str] = ()
    pending_supported_columns: Sequence[str] = ()
    sourced_from_ohlcvt_columns: Sequence[str] = ()
    validation_target_only_columns: Sequence[str] = ()
    unsupported_for_now_columns: Sequence[str] = ()
    compatible_bands: Sequence[str | AssetStateBand] = (
        AssetStateBand.MICRO,
        AssetStateBand.MESO,
        AssetStateBand.MACRO,
    )
    pool_style: str = "compact_axis"
    lookback_requirements: Mapping[str, Any] = field(default_factory=dict)
    missingness_policy: Mapping[str, Any] = field(
        default_factory=lambda: {
            "policy": "train_only_feature_filter_then_drop_nonfinite_rows",
            "max_missing_fraction": 0.2,
            "min_non_null_count": 2,
            "min_retained_columns": 2,
            "fail_closed": True,
        }
    )
    leakage_policy: Mapping[str, Any] = field(
        default_factory=lambda: {
            "policy": "source_features_only_no_forward_targets",
            "fit_scope": "train_only",
            "fail_closed": True,
        }
    )
    redundancy_policy: Mapping[str, Any] = field(
        default_factory=lambda: {
            "near_duplicate_tolerance": 1e-12,
            "correlation_threshold": 0.985,
            "mostly_zero_threshold": 0.98,
            "min_variance": 1e-12,
            "zero_epsilon": 0.0,
            "fail_closed": True,
        }
    )
    dimensionality_reduction_eligibility: Mapping[str, Any] = field(
        default_factory=lambda: {
            "eligible": True,
            "fit_scope": "train_only",
            "allowed_embeddings": list(EMBEDDING_OPTIONS),
            "selection_status": "metadata_only_not_selected",
        }
    )
    embedding_options: Sequence[str] = EMBEDDING_OPTIONS
    notes: str = ""
    schema_version: int | AssetStateSchemaVersion = ASSET_STATE_FEATURE_POOL_SCHEMA_VERSION

    def __post_init__(self) -> None:
        axis = _enum_value(self.axis, AssetStateAxis, field_name="axis")
        bands = tuple(_enum_value(band, AssetStateBand, field_name="compatible_bands") for band in self.compatible_bands)
        if not bands:
            raise ValueError("Asset-state feature pool compatible_bands must be non-empty")
        required = _string_tuple(self.required_source_columns, field_name="required_source_columns", require_non_empty=True)
        optional = _string_tuple(self.optional_source_columns, field_name="optional_source_columns")
        optional = tuple(column for column in optional if column not in set(required))
        pending = _string_tuple(self.pending_supported_columns, field_name="pending_supported_columns")
        pending = tuple(column for column in pending if column not in set(required).union(optional))
        sourced = _string_tuple(self.sourced_from_ohlcvt_columns, field_name="sourced_from_ohlcvt_columns")
        validation = _string_tuple(self.validation_target_only_columns, field_name="validation_target_only_columns")
        unsupported = _string_tuple(self.unsupported_for_now_columns, field_name="unsupported_for_now_columns")
        _reject_leakage_risk_columns(required, field_name="required_source_columns")
        _reject_leakage_risk_columns(optional, field_name="optional_source_columns")
        _reject_leakage_risk_columns(pending, field_name="pending_supported_columns")

        missingness = to_jsonable(_mapping(self.missingness_policy, field_name="missingness_policy"))
        leakage = to_jsonable(_mapping(self.leakage_policy, field_name="leakage_policy"))
        redundancy = to_jsonable(_mapping(self.redundancy_policy, field_name="redundancy_policy"))
        dimred = to_jsonable(_mapping(self.dimensionality_reduction_eligibility, field_name="dimensionality_reduction_eligibility"))
        lookbacks = to_jsonable(_mapping(self.lookback_requirements, field_name="lookback_requirements"))
        if not bool(missingness.get("fail_closed", False)):
            raise ValueError("Asset-state feature pool missingness_policy must fail closed")
        if not bool(leakage.get("fail_closed", False)):
            raise ValueError("Asset-state feature pool leakage_policy must fail closed")
        if not bool(redundancy.get("fail_closed", False)):
            raise ValueError("Asset-state feature pool redundancy_policy must fail closed")
        if str(leakage.get("fit_scope", "")).lower() != "train_only":
            raise ValueError("Asset-state feature pool leakage_policy fit_scope must be train_only")

        object.__setattr__(self, "schema_version", _schema_version(self.schema_version))
        object.__setattr__(self, "feature_pool_id", _non_empty_text(self.feature_pool_id, field_name="feature_pool_id"))
        object.__setattr__(self, "axis", axis)
        object.__setattr__(self, "compatible_bands", tuple(dict.fromkeys(bands)))
        object.__setattr__(self, "required_source_columns", required)
        object.__setattr__(self, "optional_source_columns", optional)
        object.__setattr__(self, "pending_supported_columns", pending)
        object.__setattr__(self, "sourced_from_ohlcvt_columns", sourced)
        object.__setattr__(self, "validation_target_only_columns", validation)
        object.__setattr__(self, "unsupported_for_now_columns", unsupported)
        object.__setattr__(self, "pool_style", _normalize_style(self.pool_style))
        object.__setattr__(self, "lookback_requirements", lookbacks)
        object.__setattr__(self, "missingness_policy", missingness)
        object.__setattr__(self, "leakage_policy", leakage)
        object.__setattr__(self, "redundancy_policy", redundancy)
        object.__setattr__(self, "dimensionality_reduction_eligibility", dimred)
        object.__setattr__(self, "embedding_options", _normalize_embeddings(self.embedding_options))
        object.__setattr__(self, "notes", str(self.notes).strip())

    @property
    def feature_bases(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys((*self.required_source_columns, *self.optional_source_columns)))

    @property
    def declared_candidate_bases(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys((*self.required_source_columns, *self.optional_source_columns, *self.pending_supported_columns)))

    @property
    def min_retained_columns(self) -> int:
        return max(1, _policy_int(self.missingness_policy, "min_retained_columns", len(self.required_source_columns)))

    def supports(self, *, axis: str | AssetStateAxis, band: str | AssetStateBand) -> bool:
        axis_value = _enum_value(axis, AssetStateAxis, field_name="axis")
        band_value = _enum_value(band, AssetStateBand, field_name="band")
        return axis_value == self.axis and band_value in self.compatible_bands

    def filter_config(self) -> FeatureFilterConfig:
        return FeatureFilterConfig(
            max_missing_fraction=_policy_float(self.missingness_policy, "max_missing_fraction", 0.2),
            min_variance=_policy_float(self.redundancy_policy, "min_variance", 1e-12),
            min_non_null_count=_policy_int(self.missingness_policy, "min_non_null_count", 2),
            near_duplicate_tolerance=_policy_float(self.redundancy_policy, "near_duplicate_tolerance", 1e-12),
            mostly_zero_threshold=_policy_float(self.redundancy_policy, "mostly_zero_threshold", 0.98),
            zero_epsilon=_policy_float(self.redundancy_policy, "zero_epsilon", 0.0),
            correlation_threshold=_policy_float(self.redundancy_policy, "correlation_threshold", 0.985)
            if self.redundancy_policy.get("correlation_threshold") is not None
            else None,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "feature_pool_id": self.feature_pool_id,
            "axis": self.axis,
            "compatible_bands": list(self.compatible_bands),
            "pool_style": self.pool_style,
            "required_source_columns": list(self.required_source_columns),
            "optional_source_columns": list(self.optional_source_columns),
            "pending_supported_columns": list(self.pending_supported_columns),
            "sourced_from_ohlcvt_columns": list(self.sourced_from_ohlcvt_columns),
            "validation_target_only_columns": list(self.validation_target_only_columns),
            "unsupported_for_now_columns": list(self.unsupported_for_now_columns),
            "feature_bases": list(self.feature_bases),
            "declared_candidate_bases": list(self.declared_candidate_bases),
            "lookback_requirements": to_jsonable(self.lookback_requirements),
            "missingness_policy": to_jsonable(self.missingness_policy),
            "leakage_policy": to_jsonable(self.leakage_policy),
            "redundancy_policy": to_jsonable(self.redundancy_policy),
            "dimensionality_reduction_eligibility": to_jsonable(self.dimensionality_reduction_eligibility),
            "embedding_options": list(self.embedding_options),
            "notes": self.notes,
        }

    def to_json(self, **kwargs: Any) -> str:
        return dumps_json(self.as_dict(), **kwargs)


@dataclass(frozen=True)
class AssetStateFeaturePoolRegistry:
    pools: Mapping[str, AssetStateFeaturePoolSpec | Mapping[str, Any]]
    artifact_kind: str = "asset_state_feature_pool_registry"
    schema_version: int | AssetStateSchemaVersion = ASSET_STATE_FEATURE_POOL_SCHEMA_VERSION

    def __post_init__(self) -> None:
        normalized: dict[str, AssetStateFeaturePoolSpec] = {}
        for key, value in self.pools.items():
            spec = value if isinstance(value, AssetStateFeaturePoolSpec) else AssetStateFeaturePoolSpec(**dict(value))
            if str(key) != spec.feature_pool_id:
                raise ValueError("Asset-state feature pool registry keys must match feature_pool_id")
            normalized[spec.feature_pool_id] = spec
        axes = set(default_asset_state_taxonomy().axes)
        represented_axes = {spec.axis for spec in normalized.values()}
        missing = sorted(axes.difference(represented_axes))
        if missing:
            raise ValueError(f"Asset-state feature pool registry is missing axes: {missing}")
        object.__setattr__(self, "schema_version", _schema_version(self.schema_version))
        object.__setattr__(self, "artifact_kind", _non_empty_text(self.artifact_kind, field_name="artifact_kind"))
        object.__setattr__(self, "pools", dict(sorted(normalized.items())))

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self.pools)

    def get(self, feature_pool_id: str) -> AssetStateFeaturePoolSpec:
        key = _non_empty_text(feature_pool_id, field_name="feature_pool_id")
        try:
            return self.pools[key]
        except KeyError as exc:
            raise ValueError(f"Unsupported asset-state feature pool {key!r}; expected one of: {', '.join(self.names)}") from exc

    def for_axis(
        self,
        axis: str | AssetStateAxis,
        *,
        band: str | AssetStateBand | None = None,
        style: str | None = None,
    ) -> tuple[AssetStateFeaturePoolSpec, ...]:
        axis_value = _enum_value(axis, AssetStateAxis, field_name="axis")
        band_value = None if band is None else _enum_value(band, AssetStateBand, field_name="band")
        style_value = None if style is None else _normalize_style(style)
        return tuple(
            spec
            for spec in self.pools.values()
            if spec.axis == axis_value
            and (band_value is None or band_value in spec.compatible_bands)
            and (style_value is None or spec.pool_style == style_value)
        )

    def default_for_axis(
        self,
        axis: str | AssetStateAxis,
        *,
        band: str | AssetStateBand,
        style: str = "compact_axis",
    ) -> AssetStateFeaturePoolSpec:
        matches = self.for_axis(axis, band=band, style=style)
        if matches:
            return matches[0]
        matches = self.for_axis(axis, band=band)
        if matches:
            return matches[0]
        axis_value = _enum_value(axis, AssetStateAxis, field_name="axis")
        band_value = _enum_value(band, AssetStateBand, field_name="band")
        raise ValueError(f"No asset-state feature pool registered for axis={axis_value!r} band={band_value!r}")

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "artifact_kind": self.artifact_kind,
            "pool_count": len(self.pools),
            "axes": list(ASSET_STATE_AXIS_VALUES),
            "bands": list(ASSET_STATE_BAND_VALUES),
            "pool_styles": list(FEATURE_POOL_STYLES),
            "embedding_options": list(EMBEDDING_OPTIONS),
            "pools": {key: spec.as_dict() for key, spec in self.pools.items()},
        }

    def to_json(self, **kwargs: Any) -> str:
        return dumps_json(self.as_dict(), **kwargs)


def _base_missingness_policy(min_retained: int = 2, max_missing: float = 0.2) -> dict[str, Any]:
    return {
        "policy": "train_only_feature_filter_then_drop_nonfinite_rows",
        "max_missing_fraction": max_missing,
        "min_non_null_count": 2,
        "min_retained_columns": min_retained,
        "fail_closed": True,
    }


def _base_redundancy_policy(correlation_threshold: float = 0.985) -> dict[str, Any]:
    return {
        "near_duplicate_tolerance": 1e-12,
        "correlation_threshold": correlation_threshold,
        "mostly_zero_threshold": 0.98,
        "min_variance": 1e-12,
        "zero_epsilon": 0.0,
        "fail_closed": True,
    }


def _lookbacks(*values: int) -> dict[str, Any]:
    return {
        "lookback_windows": sorted(set(int(value) for value in values if int(value) > 0)),
        "unit": "bars_or_source_feature_native_window",
        "notes": "Source scalar features must already have been computed without forward-looking data.",
    }


def _dimred(eligible: bool, options: Sequence[str] = EMBEDDING_OPTIONS) -> dict[str, Any]:
    return {
        "eligible": bool(eligible),
        "fit_scope": "train_only",
        "allowed_embeddings": list(options),
        "selection_status": "metadata_only_not_selected",
    }


def _pool(
    feature_pool_id: str,
    *,
    axis: AssetStateAxis,
    style: str,
    required: Sequence[str],
    optional: Sequence[str],
    pending: Sequence[str] = (),
    sourced_from_ohlcvt: Sequence[str] = (),
    validation_target_only: Sequence[str] = (),
    unsupported_for_now: Sequence[str] = (),
    lookbacks: Mapping[str, Any],
    min_retained: int,
    notes: str,
    dimred_eligible: bool = True,
) -> AssetStateFeaturePoolSpec:
    return AssetStateFeaturePoolSpec(
        feature_pool_id=feature_pool_id,
        axis=axis,
        pool_style=style,
        required_source_columns=required,
        optional_source_columns=optional,
        pending_supported_columns=pending,
        sourced_from_ohlcvt_columns=sourced_from_ohlcvt,
        validation_target_only_columns=validation_target_only,
        unsupported_for_now_columns=unsupported_for_now,
        lookback_requirements=lookbacks,
        missingness_policy=_base_missingness_policy(min_retained=min_retained),
        redundancy_policy=_base_redundancy_policy(),
        dimensionality_reduction_eligibility=_dimred(dimred_eligible),
        notes=notes,
    )


def default_asset_state_feature_pool_registry() -> AssetStateFeaturePoolRegistry:
    pools = (
        _pool(
            "trend_directionality_compact_axis",
            axis=AssetStateAxis.TREND,
            style="compact_axis",
            required=("log_return", "macd_hist_12_26_9", "rsi_14", "adx_14"),
            optional=("aroon_osc_25", "roc_14", "mom_14", "range_efficiency_20"),
            lookbacks=_lookbacks(9, 14, 25, 26),
            min_retained=3,
            notes="Compact directional persistence pool for trend, flatness, and directional strength.",
            dimred_eligible=False,
        ),
        _pool(
            "trend_directionality_broad_axis",
            axis=AssetStateAxis.TREND,
            style="broad_axis",
            required=("log_return", "rsi_14", "adx_14"),
            optional=(
                "macd_hist_12_26_9",
                "aroon_osc_25",
                "plus_di_14",
                "minus_di_14",
                "vi_plus_14",
                "vi_minus_14",
                "roc_14",
                "mom_14",
                "range_efficiency_20",
            ),
            pending=("trend_persistence_score_20",),
            validation_target_only=("forward_return_60m",),
            lookbacks=_lookbacks(9, 14, 25, 26),
            min_retained=5,
            notes="Broader directional pool that exposes trend strength, oscillator, and persistence variants to Test selection.",
        ),
        _pool(
            "trend_directionality_interaction_or_compressed_axis",
            axis=AssetStateAxis.TREND,
            style="interaction_or_compressed_axis",
            required=("log_return", "adx_14", "range_efficiency_20"),
            optional=("roc_14", "mom_14", "vi_plus_14", "vi_minus_14", "plus_di_14", "minus_di_14"),
            lookbacks=_lookbacks(14, 26),
            min_retained=3,
            notes="Compressed directional structure pool intended for train-only embedding or redundancy-aware feature compression tests.",
        ),
        _pool(
            "volatility_expansion_compact_axis",
            axis=AssetStateAxis.VOLATILITY,
            style="compact_axis",
            required=("atr_14", "ret_std_20", "true_range", "range_hl"),
            optional=("cv_20", "vol_osc_pct_14_28", "squeeze_scalar"),
            lookbacks=_lookbacks(14, 20, 28),
            min_retained=3,
            notes="Compact volatility compression/expansion pool using realized range and return dispersion.",
            dimred_eligible=False,
        ),
        _pool(
            "volatility_expansion_broad_axis",
            axis=AssetStateAxis.VOLATILITY,
            style="broad_axis",
            required=("atr_14", "ret_std_20", "true_range"),
            optional=("cv_20", "vol_osc_pct_14_28", "range_hl", "range_co", "var_20", "skew_20", "kurt_20", "q25_20", "q75_20", "squeeze_scalar"),
            lookbacks=_lookbacks(14, 20, 28),
            min_retained=5,
            notes="Broad volatility shape pool for expansion, compression, and shock separability tests.",
        ),
        _pool(
            "volatility_expansion_interaction_or_compressed_axis",
            axis=AssetStateAxis.VOLATILITY,
            style="interaction_or_compressed_axis",
            required=("ret_std_20", "atr_14", "squeeze_scalar"),
            optional=("cv_20", "var_20", "range_hl", "range_co", "vol_osc_pct_14_28"),
            pending=("volatility_compression_score_20",),
            lookbacks=_lookbacks(14, 20, 28),
            min_retained=3,
            notes="Compressed volatility pool emphasizing expansion/compression candidates and train-only embedding eligibility.",
        ),
        _pool(
            "activity_participation_compact_axis",
            axis=AssetStateAxis.ACTIVITY,
            style="compact_axis",
            required=("activity_state_score_20", "volume_zscore_20", "trades_zscore_20", "vroc_14"),
            optional=("trade_intensity", "avg_trade_size", "prr", "obv", "dollar_volume_proxy"),
            lookbacks=_lookbacks(14),
            min_retained=3,
            notes="Compact participation pool for activity, trade count, and volume impulse regimes.",
            dimred_eligible=False,
        ),
        _pool(
            "activity_participation_broad_axis",
            axis=AssetStateAxis.ACTIVITY,
            style="broad_axis",
            required=("activity_state_score_20", "volume_zscore_20", "trades_zscore_20"),
            optional=(
                "vroc_14",
                "trade_intensity",
                "avg_trade_size",
                "prr",
                "obv",
                "adl",
                "force_index",
                "chaikin_osc_3_10",
                "vpt",
                "eom_14",
                "pvi",
                "nvi",
                "dollar_volume_proxy",
                "volume_share_vs_rolling_20",
                "trade_count_intensity_zscore_20",
                "illiquidity_proxy_20",
            ),
            pending=("participation_imbalance_20",),
            sourced_from_ohlcvt=("volume", "trades"),
            unsupported_for_now=("order_book_depth", "open_interest"),
            lookbacks=_lookbacks(3, 10, 14),
            min_retained=5,
            notes="Broad activity and liquidity-participation pool for unusual participation and continuity tests.",
        ),
        _pool(
            "activity_participation_interaction_or_compressed_axis",
            axis=AssetStateAxis.ACTIVITY,
            style="interaction_or_compressed_axis",
            required=("activity_state_score_20", "dollar_volume_proxy", "prr"),
            optional=(
                "volume_zscore_20",
                "trades_zscore_20",
                "trade_intensity",
                "avg_trade_size",
                "force_index",
                "vpt",
                "eom_14",
                "pvi",
                "nvi",
            ),
            lookbacks=_lookbacks(14),
            min_retained=3,
            notes="Compressed participation pool intended to test volume/trade participation structure under redundancy guards.",
        ),
        _pool(
            "mean_reversion_chop_compact_axis",
            axis=AssetStateAxis.MEAN_REVERSION,
            style="compact_axis",
            required=("rsi_14", "zscore_20", "bollinger_pct_b_20", "choppiness_14"),
            optional=("stoch_k_14", "stoch_d_3", "cci_20", "williams_r_14"),
            lookbacks=_lookbacks(14, 20),
            min_retained=3,
            notes="Compact chop and mean-reversion pressure pool for range-bound and trendless behavior.",
            dimred_eligible=False,
        ),
        _pool(
            "mean_reversion_chop_broad_axis",
            axis=AssetStateAxis.MEAN_REVERSION,
            style="broad_axis",
            required=("rsi_14", "zscore_20", "choppiness_14"),
            optional=(
                "stoch_k_14",
                "stoch_d_3",
                "cci_20",
                "williams_r_14",
                "bollinger_pct_b_20",
                "bollinger_bandwidth_20",
                "hurst_100",
                "range_efficiency_20",
                "log_return",
            ),
            lookbacks=_lookbacks(14, 20, 64),
            min_retained=5,
            notes="Broad mean-reversion and chop pool for noisy range, trendlessness, and reversion tests.",
        ),
        _pool(
            "mean_reversion_chop_interaction_or_compressed_axis",
            axis=AssetStateAxis.MEAN_REVERSION,
            style="interaction_or_compressed_axis",
            required=("zscore_20", "bollinger_pct_b_20", "range_efficiency_20"),
            optional=("choppiness_14", "hurst_100", "log_return", "bollinger_bandwidth_20"),
            pending=("mean_reversion_pressure_20",),
            lookbacks=_lookbacks(14, 20, 64),
            min_retained=3,
            notes="Compressed range/reversion structure pool for train-only embedding tests.",
        ),
        _pool(
            "drawdown_stress_compact_axis",
            axis=AssetStateAxis.DRAWDOWN,
            style="compact_axis",
            required=("drawdown_from_rolling_high_20", "rolling_max_drawdown_20", "downside_vol_20", "ulcer_index_14"),
            optional=("downside_excursion_20", "atr_14", "ret_std_20"),
            lookbacks=_lookbacks(14, 20),
            min_retained=3,
            notes="Compact downside stress pool for drawdown pressure and recovery state tests.",
            dimred_eligible=False,
        ),
        _pool(
            "drawdown_stress_broad_axis",
            axis=AssetStateAxis.DRAWDOWN,
            style="broad_axis",
            required=("drawdown_from_rolling_high_20", "rolling_max_drawdown_20", "downside_vol_20"),
            optional=(
                "downside_excursion_20",
                "ulcer_index_14",
                "atr_14",
                "ret_std_20",
                "true_range",
                "log_return",
                "d_close_5",
                "d_close_20",
            ),
            lookbacks=_lookbacks(5, 14, 20),
            min_retained=5,
            notes="Broad downside excursion and stress pool for crash-like and recovery behavior tests.",
        ),
        _pool(
            "drawdown_stress_interaction_or_compressed_axis",
            axis=AssetStateAxis.DRAWDOWN,
            style="interaction_or_compressed_axis",
            required=("drawdown_from_rolling_high_20", "downside_excursion_20", "log_return"),
            optional=("rolling_max_drawdown_20", "downside_vol_20", "ulcer_index_14", "d_close_5", "d_close_20"),
            pending=("stress_recovery_score_20",),
            validation_target_only=("forward_drawdown", "forward_runup"),
            lookbacks=_lookbacks(5, 14, 20),
            min_retained=3,
            notes="Compressed downside asymmetry pool intended for stress/recovery embedding tests.",
        ),
        _pool(
            "range_efficiency_compact_axis",
            axis=AssetStateAxis.RANGE_EFFICIENCY,
            style="compact_axis",
            required=("range_efficiency_20", "true_range", "adx_14", "log_return"),
            optional=("range_hl", "range_co", "atr_14"),
            lookbacks=_lookbacks(14),
            min_retained=3,
            notes="Compact range-efficiency pool for efficient run versus noisy range behavior.",
            dimred_eligible=False,
        ),
        _pool(
            "range_efficiency_broad_axis",
            axis=AssetStateAxis.RANGE_EFFICIENCY,
            style="broad_axis",
            required=("range_efficiency_20", "true_range", "log_return"),
            optional=(
                "range_hl",
                "range_co",
                "atr_14",
                "adx_14",
                "path_efficiency_20",
                "directional_efficiency_20",
                "runup_drawdown_ratio_20",
                "close_location_value",
                "rolling_position_in_range_20",
            ),
            pending=("path_choppiness_ratio_20",),
            lookbacks=_lookbacks(14),
            min_retained=5,
            notes="Broad range-efficiency and runup/drawdown structure pool for asymmetry tests.",
        ),
        _pool(
            "range_efficiency_interaction_or_compressed_axis",
            axis=AssetStateAxis.RANGE_EFFICIENCY,
            style="interaction_or_compressed_axis",
            required=("range_efficiency_20", "runup_drawdown_ratio_20", "drawdown_from_rolling_high_20"),
            optional=(
                "path_efficiency_20",
                "close_location_value",
                "rolling_position_in_range_20",
                "adx_14",
                "atr_14",
                "log_return",
            ),
            lookbacks=_lookbacks(14),
            min_retained=3,
            notes="Compressed range and runup/drawdown asymmetry pool for train-only embedding tests.",
        ),
    )
    return AssetStateFeaturePoolRegistry({spec.feature_pool_id: spec for spec in pools})


def asset_state_feature_pool_reconciliation(
    source_feature_columns: Sequence[str],
    *,
    registry: AssetStateFeaturePoolRegistry | None = None,
) -> dict[str, Any]:
    """Compare declared Asset-State feature bases to an available scalar-feature schema."""
    from src.regimes.asset_state.feature_column_catalog import (
        FEATURE_COLUMN_STATUSES,
        FEATURE_COLUMN_STATUS_AVAILABLE,
        default_asset_state_feature_column_catalog,
    )

    active_registry = registry or default_asset_state_feature_pool_registry()
    catalog = default_asset_state_feature_column_catalog(source_feature_columns, source="asset_state_feature_pool_reconciliation")
    pool_payload: dict[str, Any] = {}
    missing_required_total: set[str] = set()
    missing_optional_total: set[str] = set()
    pending_total: set[str] = set()
    columns_by_status: dict[str, set[str]] = {status: set() for status in FEATURE_COLUMN_STATUSES}
    axis_payload: dict[str, dict[str, Any]] = {
        axis: {
            "axis": axis,
            "pool_ids": [],
            "usable_pool_ids": [],
            "blocked_pool_ids": [],
            "missing_required_columns": set(),
            "missing_optional_columns": set(),
            "pending_scalar_feature_columns": set(),
        }
        for axis in ASSET_STATE_AXIS_VALUES
    }
    for pool_id, spec in active_registry.pools.items():
        required_statuses = {base: catalog.classify(base).status for base in spec.required_source_columns}
        optional_statuses = {base: catalog.classify(base).status for base in spec.optional_source_columns}
        pending_statuses = {base: catalog.classify(base).status for base in spec.pending_supported_columns}
        sourced_statuses = {base: catalog.classify(base).status for base in spec.sourced_from_ohlcvt_columns}
        validation_statuses = {base: catalog.classify(base).status for base in spec.validation_target_only_columns}
        unsupported_statuses = {base: catalog.classify(base).status for base in spec.unsupported_for_now_columns}
        missing_required = tuple(base for base, status in required_statuses.items() if status != FEATURE_COLUMN_STATUS_AVAILABLE)
        missing_optional = tuple(base for base, status in optional_statuses.items() if status != FEATURE_COLUMN_STATUS_AVAILABLE)
        pending_columns = tuple(spec.pending_supported_columns)
        missing_required_total.update(missing_required)
        missing_optional_total.update(missing_optional)
        pending_total.update(pending_columns)
        for statuses in (
            required_statuses,
            optional_statuses,
            pending_statuses,
            sourced_statuses,
            validation_statuses,
            unsupported_statuses,
        ):
            for base, status in statuses.items():
                columns_by_status.setdefault(status, set()).add(base)
        usable = not missing_required
        axis_summary = axis_payload[spec.axis]
        axis_summary["pool_ids"].append(pool_id)
        axis_summary["missing_required_columns"].update(missing_required)
        axis_summary["missing_optional_columns"].update(missing_optional)
        axis_summary["pending_scalar_feature_columns"].update(pending_columns)
        if usable:
            axis_summary["usable_pool_ids"].append(pool_id)
        else:
            axis_summary["blocked_pool_ids"].append(pool_id)
        pool_payload[pool_id] = {
            "axis": spec.axis,
            "pool_style": spec.pool_style,
            "compatible_bands": list(spec.compatible_bands),
            "required_count": int(len(spec.required_source_columns)),
            "optional_count": int(len(spec.optional_source_columns)),
            "pending_supported_count": int(len(spec.pending_supported_columns)),
            "column_statuses": {
                "required": required_statuses,
                "optional": optional_statuses,
                "pending_supported": pending_statuses,
                "sourced_from_ohlcvt": sourced_statuses,
                "validation_target_only": validation_statuses,
                "unsupported_for_now": unsupported_statuses,
            },
            "available_required_columns": [base for base, status in required_statuses.items() if status == FEATURE_COLUMN_STATUS_AVAILABLE],
            "available_optional_columns": [base for base, status in optional_statuses.items() if status == FEATURE_COLUMN_STATUS_AVAILABLE],
            "missing_required_columns": list(missing_required),
            "missing_optional_columns": list(missing_optional),
            "pending_scalar_feature_columns": list(pending_columns),
            "required_reconciled": not missing_required,
            "usable_candidate_pool": usable,
        }
    axis_summaries = {
        axis: {
            "axis": axis,
            "status": "usable_alternatives" if payload["usable_pool_ids"] else "blocked",
            "pool_ids": list(payload["pool_ids"]),
            "usable_pool_ids": list(payload["usable_pool_ids"]),
            "blocked_pool_ids": list(payload["blocked_pool_ids"]),
            "usable_pool_count": int(len(payload["usable_pool_ids"])),
            "blocked_pool_count": int(len(payload["blocked_pool_ids"])),
            "missing_required_columns": sorted(payload["missing_required_columns"]),
            "missing_optional_columns": sorted(payload["missing_optional_columns"]),
            "pending_scalar_feature_columns": sorted(payload["pending_scalar_feature_columns"]),
            "axis_not_broken_if_any_usable_pool": bool(payload["usable_pool_ids"]),
        }
        for axis, payload in axis_payload.items()
    }
    return {
        "artifact_kind": "asset_state_feature_pool_reconciliation",
        "schema_version": int(ASSET_STATE_FEATURE_POOL_SCHEMA_VERSION),
        "pool_count": int(len(active_registry.pools)),
        "source_feature_count": int(len(set(source_feature_columns))),
        "all_required_reconciled": not missing_required_total,
        "all_axes_have_usable_pool": all(summary["usable_pool_count"] > 0 for summary in axis_summaries.values()),
        "missing_required_columns": sorted(missing_required_total),
        "missing_optional_columns": sorted(missing_optional_total),
        "pending_scalar_feature_columns": sorted(pending_total),
        "columns_by_status": {status: sorted(columns) for status, columns in sorted(columns_by_status.items()) if columns},
        "axis_summaries": axis_summaries,
        "pools": pool_payload,
        "production_profile_selection_enabled": False,
    }


__all__ = [
    "ASSET_STATE_FEATURE_POOL_SCHEMA_VERSION",
    "EMBEDDING_OPTIONS",
    "FEATURE_POOL_STYLES",
    "AssetStateFeaturePoolRegistry",
    "AssetStateFeaturePoolSpec",
    "asset_state_feature_pool_reconciliation",
    "default_asset_state_feature_pool_registry",
]
