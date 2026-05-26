from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from src.regimes.core.serialization import to_jsonable
from src.regimes.regime_features.contracts import (
    MARKET_REGIME_FEATURES,
    REGIME_FEATURES_SCHEMA_VERSION,
    RegimeFeatureKnownAtSpec,
    RegimeFeatureLineageSpec,
    market_regime_feature_output_schema,
)
from src.regimes.regime_features.universe import GlobalEligibilitySnapshot


TIMESTAMP_COLUMN = "ts"
ASSET_COLUMN = "asset"

MARKET_FEATURE_FAMILY_RETURN_SUMMARY = "market_return_summary"
MARKET_FEATURE_FAMILY_REALIZED_VOLATILITY = "market_realized_volatility"
MARKET_FEATURE_FAMILY_BREADTH = "market_breadth"
MARKET_FEATURE_FAMILY_DISPERSION = "market_dispersion"
MARKET_FEATURE_FAMILY_CORRELATION_SUMMARY = "market_correlation_summary"
MARKET_FEATURE_FAMILY_COVARIANCE_SUMMARY = "market_covariance_summary"
MARKET_FEATURE_FAMILY_LIQUIDITY_ACTIVITY = "market_liquidity_activity"
MARKET_FEATURE_FAMILY_STRESS = "market_stress"

PRIMITIVE_MARKET_FEATURE_FAMILIES: tuple[str, ...] = (
    MARKET_FEATURE_FAMILY_RETURN_SUMMARY,
    MARKET_FEATURE_FAMILY_REALIZED_VOLATILITY,
    MARKET_FEATURE_FAMILY_BREADTH,
    MARKET_FEATURE_FAMILY_DISPERSION,
    MARKET_FEATURE_FAMILY_CORRELATION_SUMMARY,
    MARKET_FEATURE_FAMILY_COVARIANCE_SUMMARY,
    MARKET_FEATURE_FAMILY_LIQUIDITY_ACTIVITY,
    MARKET_FEATURE_FAMILY_STRESS,
)

STATUS_COMPUTED = "computed"
STATUS_DISABLED = "disabled_by_feature_set"
STATUS_UNAVAILABLE_INSUFFICIENT_CORE_ASSETS = "unavailable_insufficient_core_assets"
STATUS_UNAVAILABLE_INSUFFICIENT_SAMPLES = "unavailable_insufficient_samples"
STATUS_UNAVAILABLE_MISSING_VOLUME = "unavailable_missing_volume"
STATUS_UNAVAILABLE_MISSING_TRADES = "unavailable_missing_trades"
STATUS_UNAVAILABLE_MISSING_DEPENDENCY = "unavailable_missing_dependency"
STATUS_SAMPLE_COMPUTED_SHRINKAGE_UNAVAILABLE = "sample_computed_shrinkage_unavailable_missing_dependency"

PRIMITIVE_MARKET_FEATURE_COLUMNS: tuple[str, ...] = (
    "market_return_equal_weight",
    "market_return_core_equal_weight",
    "market_return_median",
    "market_return_q05",
    "market_return_q10",
    "market_return_q25",
    "market_return_q75",
    "market_return_q90",
    "market_return_q95",
    "market_realized_volatility",
    "market_core_return_realized_volatility",
    "market_volatility_q10",
    "market_volatility_median",
    "market_volatility_q90",
    "market_volatility_asset_count",
    "share_assets_up",
    "share_assets_down",
    "positive_return_breadth",
    "negative_return_breadth",
    "finite_return_asset_count",
    "broad_universe_asset_count",
    "return_dispersion_std",
    "return_dispersion_iqr",
    "return_quantile_spread_q90_q10",
    "core_pairwise_corr_median",
    "core_pairwise_corr_mean",
    "core_pairwise_corr_q10",
    "core_pairwise_corr_q90",
    "core_pairwise_corr_sample_count",
    "core_pairwise_corr_coverage",
    "core_pairwise_corr_status",
    "covariance_summary_status",
    "covariance_sample_count",
    "covariance_trace",
    "covariance_avg_variance",
    "covariance_max_eigenvalue",
    "covariance_first_pc_concentration",
    "ledoit_wolf_covariance_trace",
    "ledoit_wolf_first_pc_concentration",
    "oas_covariance_trace",
    "oas_first_pc_concentration",
    "aggregate_volume",
    "aggregate_trades",
    "volume_status",
    "trades_status",
    "activity_breadth",
    "volume_activity_breadth",
    "trades_activity_breadth",
    "volume_concentration_hhi",
    "trades_concentration_hhi",
    "stress_down_participation",
    "downside_breadth",
    "high_vol_asset_share",
    "high_corr_high_vol_coincidence",
    "market_stress_status",
)

PRIMITIVE_MARKET_FEATURE_COLUMNS_BY_FAMILY: Mapping[str, tuple[str, ...]] = {
    MARKET_FEATURE_FAMILY_RETURN_SUMMARY: (
        "market_return_equal_weight",
        "market_return_core_equal_weight",
        "market_return_median",
        "market_return_q05",
        "market_return_q10",
        "market_return_q25",
        "market_return_q75",
        "market_return_q90",
        "market_return_q95",
    ),
    MARKET_FEATURE_FAMILY_REALIZED_VOLATILITY: (
        "market_realized_volatility",
        "market_core_return_realized_volatility",
        "market_volatility_q10",
        "market_volatility_median",
        "market_volatility_q90",
        "market_volatility_asset_count",
    ),
    MARKET_FEATURE_FAMILY_BREADTH: (
        "share_assets_up",
        "share_assets_down",
        "positive_return_breadth",
        "negative_return_breadth",
        "finite_return_asset_count",
        "broad_universe_asset_count",
    ),
    MARKET_FEATURE_FAMILY_DISPERSION: (
        "return_dispersion_std",
        "return_dispersion_iqr",
        "return_quantile_spread_q90_q10",
    ),
    MARKET_FEATURE_FAMILY_CORRELATION_SUMMARY: (
        "core_pairwise_corr_median",
        "core_pairwise_corr_mean",
        "core_pairwise_corr_q10",
        "core_pairwise_corr_q90",
        "core_pairwise_corr_sample_count",
        "core_pairwise_corr_coverage",
        "core_pairwise_corr_status",
    ),
    MARKET_FEATURE_FAMILY_COVARIANCE_SUMMARY: (
        "covariance_summary_status",
        "covariance_sample_count",
        "covariance_trace",
        "covariance_avg_variance",
        "covariance_max_eigenvalue",
        "covariance_first_pc_concentration",
        "ledoit_wolf_covariance_trace",
        "ledoit_wolf_first_pc_concentration",
        "oas_covariance_trace",
        "oas_first_pc_concentration",
    ),
    MARKET_FEATURE_FAMILY_LIQUIDITY_ACTIVITY: (
        "aggregate_volume",
        "aggregate_trades",
        "volume_status",
        "trades_status",
        "activity_breadth",
        "volume_activity_breadth",
        "trades_activity_breadth",
        "volume_concentration_hhi",
        "trades_concentration_hhi",
    ),
    MARKET_FEATURE_FAMILY_STRESS: (
        "stress_down_participation",
        "downside_breadth",
        "high_vol_asset_share",
        "high_corr_high_vol_coincidence",
        "market_stress_status",
    ),
}

UNAVAILABLE_MARKET_FEATURE_STATES: tuple[str, ...] = (
    STATUS_DISABLED,
    STATUS_UNAVAILABLE_INSUFFICIENT_CORE_ASSETS,
    STATUS_UNAVAILABLE_INSUFFICIENT_SAMPLES,
    STATUS_UNAVAILABLE_MISSING_VOLUME,
    STATUS_UNAVAILABLE_MISSING_TRADES,
    STATUS_UNAVAILABLE_MISSING_DEPENDENCY,
    STATUS_SAMPLE_COMPUTED_SHRINKAGE_UNAVAILABLE,
)


@dataclass(frozen=True)
class PrimitiveMarketFeatureConfig:
    feature_set_id: str
    interval: int
    band: str
    rolling_window: int = 20
    min_periods: int = 3
    universe_policy_id: str = "descriptive_global_eligibility"
    feature_families: Sequence[str] = field(default_factory=lambda: PRIMITIVE_MARKET_FEATURE_FAMILIES)
    enable_shrinkage_covariance: bool = True
    schema_version: int = REGIME_FEATURES_SCHEMA_VERSION
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        interval = int(self.interval)
        if interval <= 0:
            raise ValueError("Regime Feature market interval must be positive")
        if int(self.rolling_window) < 2:
            raise ValueError("Regime Feature market rolling_window must be at least 2")
        if int(self.min_periods) < 1:
            raise ValueError("Regime Feature market min_periods must be positive")
        families = _feature_family_tuple(self.feature_families)
        object.__setattr__(self, "feature_set_id", _text(self.feature_set_id, field_name="feature_set_id"))
        object.__setattr__(self, "interval", interval)
        object.__setattr__(self, "band", _text(self.band, field_name="band").lower())
        object.__setattr__(self, "rolling_window", int(self.rolling_window))
        object.__setattr__(self, "min_periods", int(self.min_periods))
        object.__setattr__(self, "universe_policy_id", _text(self.universe_policy_id, field_name="universe_policy_id"))
        object.__setattr__(self, "feature_families", families)
        object.__setattr__(self, "enable_shrinkage_covariance", bool(self.enable_shrinkage_covariance))
        object.__setattr__(self, "schema_version", int(self.schema_version))
        object.__setattr__(self, "metadata", to_jsonable(dict(self.metadata)))

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "artifact_kind": "primitive_market_feature_config",
            "feature_set_id": self.feature_set_id,
            "interval": int(self.interval),
            "band": self.band,
            "rolling_window": int(self.rolling_window),
            "min_periods": int(self.min_periods),
            "universe_policy_id": self.universe_policy_id,
            "feature_families": list(self.feature_families),
            "enable_shrinkage_covariance": bool(self.enable_shrinkage_covariance),
            "primitive_only": True,
            "pairwise_relationship_logic_finalized": False,
            "raw_pairwise_matrices_emitted": False,
            "metadata": to_jsonable(dict(self.metadata)),
        }


@dataclass
class PrimitiveMarketFeatureResult:
    config: PrimitiveMarketFeatureConfig
    feature_frame: pd.DataFrame = field(repr=False)
    known_at: RegimeFeatureKnownAtSpec
    lineage: RegimeFeatureLineageSpec
    universe_snapshot_id: str
    universe_snapshot_hash: str | None = None
    source_asset_count: int = 0
    diagnostics: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = REGIME_FEATURES_SCHEMA_VERSION

    @property
    def feature_columns(self) -> tuple[str, ...]:
        excluded = set(market_regime_feature_output_schema().required_columns)
        return tuple(column for column in self.feature_frame.columns if column not in excluded and column not in {"source_asset_count"})

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "artifact_kind": "primitive_market_regime_feature_result",
            "status": "ready" if not self.feature_frame.empty else "empty",
            "row_count": int(self.feature_frame.shape[0]),
            "feature_columns": list(self.feature_columns),
            "feature_count": int(len(self.feature_columns)),
            "feature_set_id": self.config.feature_set_id,
            "interval": int(self.config.interval),
            "band": self.config.band,
            "universe_snapshot_id": self.universe_snapshot_id,
            "universe_snapshot_hash": self.universe_snapshot_hash,
            "source_asset_count": int(self.source_asset_count),
            "known_at": self.known_at.as_dict(),
            "lineage": self.lineage.as_dict(),
            "diagnostics": to_jsonable(dict(self.diagnostics)),
            "production_outputs_written": False,
            "raw_pairwise_matrices_emitted": False,
            "pairwise_relationship_features_materialized": False,
        }


def build_primitive_market_regime_features(
    asset_frames: Mapping[str, pd.DataFrame],
    *,
    config: PrimitiveMarketFeatureConfig,
    universe_snapshot: Any | None = None,
    core_basket_assets: Sequence[str] = (),
    broad_universe_assets: Sequence[str] = (),
    source_partition_lineage: Sequence[Mapping[str, Any]] = (),
    generated_at_ts: int | None = None,
    source_data_kinds: Sequence[str] = ("ohlcvt",),
) -> PrimitiveMarketFeatureResult:
    if not isinstance(asset_frames, Mapping) or not asset_frames:
        raise ValueError("Regime Feature primitive market builder requires explicit asset_frames")
    normalized = {str(asset): _normalize_frame(frame, asset=str(asset)) for asset, frame in asset_frames.items()}
    normalized = {asset: frame for asset, frame in normalized.items() if not frame.empty}
    if not normalized:
        raise ValueError("Regime Feature primitive market builder requires at least one non-empty asset frame")

    universe = _resolve_universe_inputs(
        normalized_assets=tuple(sorted(normalized)),
        config=config,
        universe_snapshot=universe_snapshot,
        core_basket_assets=core_basket_assets,
        broad_universe_assets=broad_universe_assets,
    )
    core_assets = universe["core_assets"]
    broad_assets = universe["broad_assets"]
    universe_policy_id = universe["universe_policy_id"]
    universe_snapshot_id = universe["universe_snapshot_id"]
    universe_snapshot_hash = universe["universe_snapshot_hash"]

    return_assets = tuple(dict.fromkeys((*broad_assets, *core_assets)))
    all_returns = _return_panel(normalized, assets=return_assets, interval=int(config.interval))
    if all_returns.empty:
        raise ValueError("Regime Feature primitive market builder requires at least one return series")
    returns = all_returns.loc[:, [asset for asset in broad_assets if asset in all_returns.columns]]
    if returns.empty:
        raise ValueError("Regime Feature primitive market builder requires at least one broad return series")
    core_returns = all_returns.loc[:, [asset for asset in core_assets if asset in all_returns.columns]].reindex(returns.index)
    volume = _value_panel(normalized, assets=broad_assets, column="volume", index=returns.index)
    trades = _value_panel(normalized, assets=broad_assets, column="trades", index=returns.index)
    ts_values = returns.index.astype("int64")

    out = pd.DataFrame({TIMESTAMP_COLUMN: ts_values})
    out["interval"] = int(config.interval)
    out["band"] = config.band
    out["feature_set_id"] = config.feature_set_id
    out["universe_policy_id"] = universe_policy_id
    out["universe_snapshot_id"] = universe_snapshot_id
    out["schema_version"] = int(config.schema_version)

    rolling_asset_vol = _add_return_summary(out, returns=returns, core_returns=core_returns, config=config)
    _add_volatility_summary(out, returns=returns, core_returns=core_returns, rolling_asset_vol=rolling_asset_vol, config=config)
    _add_breadth(out, returns=returns, config=config)
    _add_dispersion(out, returns=returns, config=config)
    corr_summary = _rolling_correlation_summary(core_returns, config=config)
    for column in corr_summary.columns:
        out[column] = corr_summary[column].to_numpy()
    covariance_summary = _rolling_covariance_summary(core_returns, config=config)
    for column in covariance_summary.columns:
        out[column] = covariance_summary[column].to_numpy()
    _add_liquidity_activity(out, volume=volume, trades=trades, broad_asset_count=len(broad_assets), config=config)
    _add_stress(out, returns=returns, rolling_asset_vol=rolling_asset_vol, corr_summary=corr_summary, config=config)

    source_tail_ts = int(max(frame[TIMESTAMP_COLUMN].max() for frame in normalized.values()))
    feature_window_start = int(min(frame[TIMESTAMP_COLUMN].min() for frame in normalized.values()))
    generated = int(generated_at_ts if generated_at_ts is not None else max(int(time.time()), source_tail_ts))
    known_at = RegimeFeatureKnownAtSpec(
        ts=source_tail_ts,
        known_at_ts=generated,
        source_tail_ts=source_tail_ts,
        feature_available_at_ts=generated,
        no_lookahead_verified=True,
    )
    lineage_entries = tuple(source_partition_lineage) or (
        {
            "source_kind": "in_memory_asset_frames",
            "asset_count": int(len(normalized)),
            "interval": int(config.interval),
            "note": "caller supplied bounded frames",
        },
    )
    lineage = RegimeFeatureLineageSpec(
        artifact_family=MARKET_REGIME_FEATURES,
        feature_set_id=config.feature_set_id,
        interval=int(config.interval),
        band=config.band,
        source_data_kinds=source_data_kinds,
        source_partition_lineage=lineage_entries,
        source_tail_ts=source_tail_ts,
        feature_window_start=feature_window_start,
        feature_window_end=source_tail_ts,
        generated_at=generated,
        run_id=f"{config.feature_set_id}_{config.interval}_{config.band}",
        universe_snapshot_id=universe_snapshot_id,
        universe_snapshot_hash=universe_snapshot_hash,
        feature_registry_id="primitive_market_features_v1",
        calculation_policy=config.as_dict(),
    )
    out["known_at_ts"] = int(known_at.known_at_ts)
    out["source_tail_ts"] = int(known_at.source_tail_ts)
    out["lineage_id"] = lineage.lineage_id
    out = out[
        [
            TIMESTAMP_COLUMN,
            "interval",
            "band",
            "feature_set_id",
            "universe_policy_id",
            "universe_snapshot_id",
            "known_at_ts",
            "source_tail_ts",
            "lineage_id",
            "schema_version",
            *PRIMITIVE_MARKET_FEATURE_COLUMNS,
        ]
    ]
    market_regime_feature_output_schema().validate_columns(out.columns)
    diagnostics = {
        "primitive_market_feature_columns": list(PRIMITIVE_MARKET_FEATURE_COLUMNS),
        "feature_families": list(config.feature_families),
        "core_basket_assets": list(core_assets),
        "broad_universe_assets": list(broad_assets),
        "core_basket_asset_count": int(len(core_assets)),
        "broad_universe_asset_count": int(len(broad_assets)),
        "correlation_status_counts": _status_counts(out["core_pairwise_corr_status"]),
        "covariance_status_counts": _status_counts(out["covariance_summary_status"]),
        "volume_status_counts": _status_counts(out["volume_status"]),
        "trades_status_counts": _status_counts(out["trades_status"]),
        "core_basket_used_for_correlation_covariance": True,
        "broad_universe_used_for_breadth_dispersion_activity": True,
        "raw_pairwise_matrices_emitted": False,
        "pairwise_relationship_logic_finalized": False,
        "correlation_covariance_summaries_included": True,
    }
    return PrimitiveMarketFeatureResult(
        config=config,
        feature_frame=out,
        known_at=known_at,
        lineage=lineage,
        universe_snapshot_id=universe_snapshot_id,
        universe_snapshot_hash=universe_snapshot_hash,
        source_asset_count=len(normalized),
        diagnostics=diagnostics,
    )


def _resolve_universe_inputs(
    *,
    normalized_assets: Sequence[str],
    config: PrimitiveMarketFeatureConfig,
    universe_snapshot: Any | None,
    core_basket_assets: Sequence[str],
    broad_universe_assets: Sequence[str],
) -> dict[str, Any]:
    available = set(normalized_assets)
    if universe_snapshot is not None:
        core_assets = tuple(asset for asset in getattr(universe_snapshot, "core_basket_assets", ()) if asset in available)
        broad_assets = tuple(asset for asset in getattr(universe_snapshot, "broad_universe_assets", ()) if asset in available)
        if not core_assets and hasattr(universe_snapshot, "assets"):
            core_assets = tuple(asset for asset in getattr(universe_snapshot, "assets", ()) if asset in available)
        policy = getattr(universe_snapshot, "selection_policy", None)
        config_obj = getattr(universe_snapshot, "config", None)
        universe_policy_id = getattr(policy, "policy_id", None) or getattr(config_obj, "policy_id", None) or config.universe_policy_id
        snapshot_id = getattr(universe_snapshot, "snapshot_id", None) or "in_memory_universe_snapshot"
        snapshot_hash_obj = getattr(universe_snapshot, "snapshot_hash", None)
        universe_snapshot_hash = getattr(snapshot_hash_obj, "value", snapshot_hash_obj)
    else:
        core_assets = tuple(str(asset) for asset in core_basket_assets if str(asset) in available)
        broad_assets = tuple(str(asset) for asset in broad_universe_assets if str(asset) in available)
        universe_policy_id = config.universe_policy_id
        snapshot_id = "manual_or_in_memory_universe"
        universe_snapshot_hash = None
    if not broad_assets:
        broad_assets = tuple(sorted(available))
    if not core_assets:
        core_assets = broad_assets
    return {
        "core_assets": tuple(core_assets),
        "broad_assets": tuple(broad_assets),
        "universe_policy_id": str(universe_policy_id),
        "universe_snapshot_id": str(snapshot_id),
        "universe_snapshot_hash": None if universe_snapshot_hash is None else str(universe_snapshot_hash),
    }


def _add_return_summary(
    out: pd.DataFrame,
    *,
    returns: pd.DataFrame,
    core_returns: pd.DataFrame,
    config: PrimitiveMarketFeatureConfig,
) -> pd.DataFrame:
    if not _family_enabled(config, MARKET_FEATURE_FAMILY_RETURN_SUMMARY):
        for column in (
            "market_return_equal_weight",
            "market_return_core_equal_weight",
            "market_return_median",
            "market_return_q05",
            "market_return_q10",
            "market_return_q25",
            "market_return_q75",
            "market_return_q90",
            "market_return_q95",
        ):
            out[column] = np.nan
        return returns.rolling(int(config.rolling_window), min_periods=int(config.min_periods)).std(ddof=0)
    out["market_return_equal_weight"] = returns.mean(axis=1, skipna=True).to_numpy()
    out["market_return_core_equal_weight"] = core_returns.mean(axis=1, skipna=True).reindex(returns.index).to_numpy()
    out["market_return_median"] = returns.median(axis=1, skipna=True).to_numpy()
    for quantile, column in (
        (0.05, "market_return_q05"),
        (0.10, "market_return_q10"),
        (0.25, "market_return_q25"),
        (0.75, "market_return_q75"),
        (0.90, "market_return_q90"),
        (0.95, "market_return_q95"),
    ):
        out[column] = returns.quantile(quantile, axis=1, interpolation="linear").to_numpy()
    return returns.rolling(int(config.rolling_window), min_periods=int(config.min_periods)).std(ddof=0)


def _add_volatility_summary(
    out: pd.DataFrame,
    *,
    returns: pd.DataFrame,
    core_returns: pd.DataFrame,
    rolling_asset_vol: pd.DataFrame,
    config: PrimitiveMarketFeatureConfig,
) -> None:
    columns = (
        "market_realized_volatility",
        "market_core_return_realized_volatility",
        "market_volatility_q10",
        "market_volatility_median",
        "market_volatility_q90",
        "market_volatility_asset_count",
    )
    if not _family_enabled(config, MARKET_FEATURE_FAMILY_REALIZED_VOLATILITY):
        for column in columns:
            out[column] = np.nan
        return
    market_return = pd.Series(out["market_return_equal_weight"].to_numpy(), index=returns.index)
    core_market_return = core_returns.mean(axis=1, skipna=True).reindex(returns.index)
    out["market_realized_volatility"] = market_return.rolling(int(config.rolling_window), min_periods=int(config.min_periods)).std(ddof=0).to_numpy()
    out["market_core_return_realized_volatility"] = core_market_return.rolling(int(config.rolling_window), min_periods=int(config.min_periods)).std(ddof=0).to_numpy()
    out["market_volatility_q10"] = rolling_asset_vol.quantile(0.10, axis=1, interpolation="linear").to_numpy()
    out["market_volatility_median"] = rolling_asset_vol.median(axis=1, skipna=True).to_numpy()
    out["market_volatility_q90"] = rolling_asset_vol.quantile(0.90, axis=1, interpolation="linear").to_numpy()
    out["market_volatility_asset_count"] = rolling_asset_vol.notna().sum(axis=1).astype("int64").to_numpy()


def _add_breadth(out: pd.DataFrame, *, returns: pd.DataFrame, config: PrimitiveMarketFeatureConfig) -> None:
    if not _family_enabled(config, MARKET_FEATURE_FAMILY_BREADTH):
        for column in ("share_assets_up", "share_assets_down", "positive_return_breadth", "negative_return_breadth", "finite_return_asset_count", "broad_universe_asset_count"):
            out[column] = np.nan
        return
    denom = returns.notna().sum(axis=1).replace(0, np.nan)
    up = returns.gt(0.0).sum(axis=1).div(denom)
    down = returns.lt(0.0).sum(axis=1).div(denom)
    out["share_assets_up"] = up.to_numpy()
    out["share_assets_down"] = down.to_numpy()
    out["positive_return_breadth"] = up.to_numpy()
    out["negative_return_breadth"] = down.to_numpy()
    out["finite_return_asset_count"] = returns.notna().sum(axis=1).astype("int64").to_numpy()
    out["broad_universe_asset_count"] = int(returns.shape[1])


def _add_dispersion(out: pd.DataFrame, *, returns: pd.DataFrame, config: PrimitiveMarketFeatureConfig) -> None:
    if not _family_enabled(config, MARKET_FEATURE_FAMILY_DISPERSION):
        for column in ("return_dispersion_std", "return_dispersion_iqr", "return_quantile_spread_q90_q10"):
            out[column] = np.nan
        return
    q75 = returns.quantile(0.75, axis=1, interpolation="linear")
    q25 = returns.quantile(0.25, axis=1, interpolation="linear")
    q90 = returns.quantile(0.90, axis=1, interpolation="linear")
    q10 = returns.quantile(0.10, axis=1, interpolation="linear")
    out["return_dispersion_std"] = returns.std(axis=1, skipna=True, ddof=0).to_numpy()
    out["return_dispersion_iqr"] = (q75 - q25).to_numpy()
    out["return_quantile_spread_q90_q10"] = (q90 - q10).to_numpy()


def _rolling_correlation_summary(core_returns: pd.DataFrame, *, config: PrimitiveMarketFeatureConfig) -> pd.DataFrame:
    columns = (
        "core_pairwise_corr_median",
        "core_pairwise_corr_mean",
        "core_pairwise_corr_q10",
        "core_pairwise_corr_q90",
        "core_pairwise_corr_sample_count",
        "core_pairwise_corr_coverage",
        "core_pairwise_corr_status",
    )
    if not _family_enabled(config, MARKET_FEATURE_FAMILY_CORRELATION_SUMMARY):
        return _status_frame(core_returns.index, columns=columns, status_column="core_pairwise_corr_status", status=STATUS_DISABLED)
    records: list[dict[str, Any]] = []
    for ts in core_returns.index:
        window = core_returns.loc[:ts].tail(int(config.rolling_window)).dropna(axis=1, how="all")
        if window.shape[1] < 2:
            records.append(_corr_record(status=STATUS_UNAVAILABLE_INSUFFICIENT_CORE_ASSETS))
            continue
        sample_count = int(window.dropna(axis=0, how="all").shape[0])
        if sample_count < int(config.min_periods):
            records.append(_corr_record(status=STATUS_UNAVAILABLE_INSUFFICIENT_SAMPLES, sample_count=sample_count))
            continue
        corr = window.corr(min_periods=int(config.min_periods))
        values = _upper_triangle_values(corr.to_numpy(dtype="float64"))
        pair_count = int(window.shape[1] * (window.shape[1] - 1) / 2)
        finite_values = values[np.isfinite(values)]
        if finite_values.size == 0:
            records.append(_corr_record(status=STATUS_UNAVAILABLE_INSUFFICIENT_SAMPLES, sample_count=sample_count))
            continue
        records.append(
            {
                "core_pairwise_corr_median": float(np.nanmedian(finite_values)),
                "core_pairwise_corr_mean": float(np.nanmean(finite_values)),
                "core_pairwise_corr_q10": float(np.nanquantile(finite_values, 0.10)),
                "core_pairwise_corr_q90": float(np.nanquantile(finite_values, 0.90)),
                "core_pairwise_corr_sample_count": sample_count,
                "core_pairwise_corr_coverage": float(finite_values.size / max(1, pair_count)),
                "core_pairwise_corr_status": STATUS_COMPUTED,
            }
        )
    return pd.DataFrame(records, index=core_returns.index)


def _rolling_covariance_summary(core_returns: pd.DataFrame, *, config: PrimitiveMarketFeatureConfig) -> pd.DataFrame:
    columns = (
        "covariance_summary_status",
        "covariance_sample_count",
        "covariance_trace",
        "covariance_avg_variance",
        "covariance_max_eigenvalue",
        "covariance_first_pc_concentration",
        "ledoit_wolf_covariance_trace",
        "ledoit_wolf_first_pc_concentration",
        "oas_covariance_trace",
        "oas_first_pc_concentration",
    )
    if not _family_enabled(config, MARKET_FEATURE_FAMILY_COVARIANCE_SUMMARY):
        return _status_frame(core_returns.index, columns=columns, status_column="covariance_summary_status", status=STATUS_DISABLED)
    covariance_estimators = _covariance_estimators_available() if config.enable_shrinkage_covariance else None
    records: list[dict[str, Any]] = []
    for ts in core_returns.index:
        window = core_returns.loc[:ts].tail(int(config.rolling_window)).dropna(axis=1, how="all")
        if window.shape[1] < 2:
            records.append(_cov_record(status=STATUS_UNAVAILABLE_INSUFFICIENT_CORE_ASSETS))
            continue
        complete = window.dropna(axis=0, how="any")
        if complete.shape[0] < int(config.min_periods):
            records.append(_cov_record(status=STATUS_UNAVAILABLE_INSUFFICIENT_SAMPLES, sample_count=int(complete.shape[0])))
            continue
        values = complete.to_numpy(dtype="float64")
        sample_cov = np.cov(values, rowvar=False, ddof=0)
        base = _cov_metrics(sample_cov)
        base["covariance_sample_count"] = int(complete.shape[0])
        if covariance_estimators is None:
            base.update(
                {
                    "covariance_summary_status": STATUS_SAMPLE_COMPUTED_SHRINKAGE_UNAVAILABLE,
                    "ledoit_wolf_covariance_trace": np.nan,
                    "ledoit_wolf_first_pc_concentration": np.nan,
                    "oas_covariance_trace": np.nan,
                    "oas_first_pc_concentration": np.nan,
                }
            )
            records.append(base)
            continue
        ledoit_wolf_cls, oas_cls = covariance_estimators
        try:
            lw_cov = ledoit_wolf_cls().fit(values).covariance_
            oas_cov = oas_cls().fit(values).covariance_
            lw_metrics = _cov_metrics(lw_cov)
            oas_metrics = _cov_metrics(oas_cov)
            base.update(
                {
                    "covariance_summary_status": STATUS_COMPUTED,
                    "ledoit_wolf_covariance_trace": lw_metrics["covariance_trace"],
                    "ledoit_wolf_first_pc_concentration": lw_metrics["covariance_first_pc_concentration"],
                    "oas_covariance_trace": oas_metrics["covariance_trace"],
                    "oas_first_pc_concentration": oas_metrics["covariance_first_pc_concentration"],
                }
            )
        except Exception:
            base.update(
                {
                    "covariance_summary_status": STATUS_SAMPLE_COMPUTED_SHRINKAGE_UNAVAILABLE,
                    "ledoit_wolf_covariance_trace": np.nan,
                    "ledoit_wolf_first_pc_concentration": np.nan,
                    "oas_covariance_trace": np.nan,
                    "oas_first_pc_concentration": np.nan,
                }
            )
        records.append(base)
    return pd.DataFrame(records, index=core_returns.index)


def _add_liquidity_activity(
    out: pd.DataFrame,
    *,
    volume: pd.DataFrame,
    trades: pd.DataFrame,
    broad_asset_count: int,
    config: PrimitiveMarketFeatureConfig,
) -> None:
    columns = (
        "aggregate_volume",
        "aggregate_trades",
        "volume_status",
        "trades_status",
        "activity_breadth",
        "volume_activity_breadth",
        "trades_activity_breadth",
        "volume_concentration_hhi",
        "trades_concentration_hhi",
    )
    if not _family_enabled(config, MARKET_FEATURE_FAMILY_LIQUIDITY_ACTIVITY):
        for column in columns:
            out[column] = STATUS_DISABLED if column.endswith("_status") else np.nan
        return
    volume_available = bool(not volume.empty and volume.notna().any().any())
    trades_available = bool(not trades.empty and trades.notna().any().any())
    volume_positive = volume.fillna(0.0).gt(0.0) if not volume.empty else pd.DataFrame(index=out.index)
    trades_positive = trades.fillna(0.0).gt(0.0) if not trades.empty else pd.DataFrame(index=out.index)
    out["aggregate_volume"] = volume.sum(axis=1, skipna=True, min_count=1).to_numpy() if volume_available else np.nan
    out["aggregate_trades"] = trades.sum(axis=1, skipna=True, min_count=1).to_numpy() if trades_available else np.nan
    out["volume_status"] = STATUS_COMPUTED if volume_available else STATUS_UNAVAILABLE_MISSING_VOLUME
    out["trades_status"] = STATUS_COMPUTED if trades_available else STATUS_UNAVAILABLE_MISSING_TRADES
    denom = max(1, int(broad_asset_count))
    out["volume_activity_breadth"] = volume_positive.sum(axis=1).div(denom).to_numpy() if volume_available else np.nan
    out["trades_activity_breadth"] = trades_positive.sum(axis=1).div(denom).to_numpy() if trades_available else np.nan
    if volume_available and trades_available:
        out["activity_breadth"] = (volume_positive | trades_positive).sum(axis=1).div(denom).to_numpy()
    elif volume_available:
        out["activity_breadth"] = out["volume_activity_breadth"].to_numpy()
    elif trades_available:
        out["activity_breadth"] = out["trades_activity_breadth"].to_numpy()
    else:
        out["activity_breadth"] = np.nan
    out["volume_concentration_hhi"] = _concentration_hhi(volume).to_numpy() if volume_available else np.nan
    out["trades_concentration_hhi"] = _concentration_hhi(trades).to_numpy() if trades_available else np.nan


def _add_stress(
    out: pd.DataFrame,
    *,
    returns: pd.DataFrame,
    rolling_asset_vol: pd.DataFrame,
    corr_summary: pd.DataFrame,
    config: PrimitiveMarketFeatureConfig,
) -> None:
    columns = ("stress_down_participation", "downside_breadth", "high_vol_asset_share", "high_corr_high_vol_coincidence", "market_stress_status")
    if not _family_enabled(config, MARKET_FEATURE_FAMILY_STRESS):
        for column in columns:
            out[column] = STATUS_DISABLED if column == "market_stress_status" else np.nan
        return
    denom = returns.notna().sum(axis=1).replace(0, np.nan)
    downside = returns.lt(0.0).sum(axis=1).div(denom)
    high_vol_threshold = rolling_asset_vol.quantile(0.75, axis=1, interpolation="linear")
    high_vol = rolling_asset_vol.ge(high_vol_threshold, axis=0)
    high_vol_share = high_vol.sum(axis=1).div(rolling_asset_vol.notna().sum(axis=1).replace(0, np.nan))
    corr_median = pd.to_numeric(corr_summary.get("core_pairwise_corr_median"), errors="coerce")
    coincidence = pd.Series(np.nan, index=returns.index, dtype="float64")
    valid = corr_median.notna() & high_vol_share.notna()
    coincidence.loc[valid] = ((corr_median.loc[valid] >= 0.50) & (high_vol_share.loc[valid] >= 0.50)).astype(float)
    out["stress_down_participation"] = downside.to_numpy()
    out["downside_breadth"] = downside.to_numpy()
    out["high_vol_asset_share"] = high_vol_share.to_numpy()
    out["high_corr_high_vol_coincidence"] = coincidence.to_numpy()
    out["market_stress_status"] = np.where(valid.to_numpy(), STATUS_COMPUTED, STATUS_UNAVAILABLE_INSUFFICIENT_SAMPLES)


def _return_panel(frames: Mapping[str, pd.DataFrame], *, assets: Sequence[str], interval: int) -> pd.DataFrame:
    series: dict[str, pd.Series] = {}
    step_seconds = int(interval) * 60
    for asset in assets:
        frame = frames.get(str(asset))
        if frame is None or frame.empty:
            continue
        close = pd.to_numeric(frame["close"], errors="coerce").where(lambda values: values > 0)
        returns = np.log(close).diff()
        ts = pd.to_numeric(frame[TIMESTAMP_COLUMN], errors="coerce").astype("int64")
        valid_step = ts.diff().fillna(step_seconds).eq(step_seconds)
        returns = pd.Series(returns).where(valid_step)
        series[str(asset)] = pd.Series(returns.to_numpy(), index=pd.Index(ts, name=TIMESTAMP_COLUMN), name=str(asset))
    if not series:
        return pd.DataFrame()
    return pd.concat(series.values(), axis=1).sort_index()


def _value_panel(frames: Mapping[str, pd.DataFrame], *, assets: Sequence[str], column: str, index: pd.Index) -> pd.DataFrame:
    series: dict[str, pd.Series] = {}
    for asset in assets:
        frame = frames.get(str(asset))
        if frame is None or frame.empty or column not in frame.columns:
            continue
        series[str(asset)] = pd.Series(
            pd.to_numeric(frame[column], errors="coerce").to_numpy(),
            index=pd.Index(frame[TIMESTAMP_COLUMN].astype("int64"), name=TIMESTAMP_COLUMN),
            name=str(asset),
        )
    if not series:
        return pd.DataFrame(index=index)
    return pd.concat(series.values(), axis=1).reindex(index).sort_index()


def _normalize_frame(frame: pd.DataFrame, *, asset: str) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame(columns=[ASSET_COLUMN, TIMESTAMP_COLUMN, "close", "volume", "trades"])
    out = frame.copy()
    if ASSET_COLUMN not in out.columns:
        out[ASSET_COLUMN] = str(asset)
    for column in (TIMESTAMP_COLUMN, "close", "volume", "trades"):
        if column not in out.columns:
            out[column] = np.nan
    out = out[out[ASSET_COLUMN].astype(str) == str(asset)].copy()
    out[TIMESTAMP_COLUMN] = pd.to_numeric(out[TIMESTAMP_COLUMN], errors="coerce")
    out = out.dropna(subset=[TIMESTAMP_COLUMN]).copy()
    out[TIMESTAMP_COLUMN] = out[TIMESTAMP_COLUMN].astype("int64")
    for column in ("close", "volume", "trades"):
        out[column] = pd.to_numeric(out[column], errors="coerce")
    return out[[ASSET_COLUMN, TIMESTAMP_COLUMN, "close", "volume", "trades"]].drop_duplicates([ASSET_COLUMN, TIMESTAMP_COLUMN], keep="last").sort_values(TIMESTAMP_COLUMN).reset_index(drop=True)


def _corr_record(*, status: str, sample_count: int = 0) -> dict[str, Any]:
    return {
        "core_pairwise_corr_median": np.nan,
        "core_pairwise_corr_mean": np.nan,
        "core_pairwise_corr_q10": np.nan,
        "core_pairwise_corr_q90": np.nan,
        "core_pairwise_corr_sample_count": int(sample_count),
        "core_pairwise_corr_coverage": 0.0,
        "core_pairwise_corr_status": status,
    }


def _cov_record(*, status: str, sample_count: int = 0) -> dict[str, Any]:
    return {
        "covariance_summary_status": status,
        "covariance_sample_count": int(sample_count),
        "covariance_trace": np.nan,
        "covariance_avg_variance": np.nan,
        "covariance_max_eigenvalue": np.nan,
        "covariance_first_pc_concentration": np.nan,
        "ledoit_wolf_covariance_trace": np.nan,
        "ledoit_wolf_first_pc_concentration": np.nan,
        "oas_covariance_trace": np.nan,
        "oas_first_pc_concentration": np.nan,
    }


def _cov_metrics(cov: np.ndarray) -> dict[str, float]:
    matrix = np.asarray(cov, dtype="float64")
    if matrix.ndim != 2:
        return {
            "covariance_trace": np.nan,
            "covariance_avg_variance": np.nan,
            "covariance_max_eigenvalue": np.nan,
            "covariance_first_pc_concentration": np.nan,
        }
    eigenvalues = np.linalg.eigvalsh(matrix)
    trace = float(np.trace(matrix))
    max_eigen = float(np.nanmax(eigenvalues)) if eigenvalues.size else np.nan
    return {
        "covariance_trace": trace,
        "covariance_avg_variance": float(trace / max(1, matrix.shape[0])),
        "covariance_max_eigenvalue": max_eigen,
        "covariance_first_pc_concentration": float(max_eigen / trace) if math.isfinite(trace) and trace > 0 else np.nan,
    }


def _covariance_estimators_available() -> tuple[type[Any], type[Any]] | None:
    try:
        from sklearn.covariance import LedoitWolf, OAS
    except Exception:
        return None
    return LedoitWolf, OAS


def _upper_triangle_values(matrix: np.ndarray) -> np.ndarray:
    if matrix.ndim != 2 or matrix.shape[0] < 2:
        return np.asarray([], dtype="float64")
    mask = np.triu(np.ones(matrix.shape, dtype=bool), k=1)
    return np.asarray(matrix[mask], dtype="float64")


def _status_frame(index: pd.Index, *, columns: Sequence[str], status_column: str, status: str) -> pd.DataFrame:
    frame = pd.DataFrame(index=index)
    for column in columns:
        frame[column] = status if column == status_column else np.nan
    return frame


def _concentration_hhi(values: pd.DataFrame) -> pd.Series:
    def _row_hhi(row: pd.Series) -> float:
        positive = pd.to_numeric(row, errors="coerce").fillna(0.0).clip(lower=0.0)
        total = float(positive.sum())
        if total <= 0:
            return np.nan
        shares = positive / total
        return float(np.square(shares).sum())

    return values.apply(_row_hhi, axis=1)


def _family_enabled(config: PrimitiveMarketFeatureConfig, family: str) -> bool:
    return family in set(config.feature_families)


def _feature_family_tuple(values: Sequence[str]) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValueError("Regime Feature market feature_families must be a sequence")
    families = tuple(_text(value, field_name="feature_family") for value in values)
    if not families:
        raise ValueError("Regime Feature market feature_families must be non-empty")
    invalid = [family for family in families if family not in PRIMITIVE_MARKET_FEATURE_FAMILIES]
    if invalid:
        valid = ", ".join(PRIMITIVE_MARKET_FEATURE_FAMILIES)
        raise ValueError(f"Unsupported Regime Feature market feature families {invalid}; expected one of: {valid}")
    return tuple(dict.fromkeys(families))


def _status_counts(values: pd.Series) -> dict[str, int]:
    return {str(key): int(value) for key, value in values.value_counts(dropna=False).sort_index().items()}


def _text(value: object, *, field_name: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"Regime Feature market {field_name} must be non-empty")
    return text


__all__ = [
    "MARKET_FEATURE_FAMILY_BREADTH",
    "MARKET_FEATURE_FAMILY_CORRELATION_SUMMARY",
    "MARKET_FEATURE_FAMILY_COVARIANCE_SUMMARY",
    "MARKET_FEATURE_FAMILY_DISPERSION",
    "MARKET_FEATURE_FAMILY_LIQUIDITY_ACTIVITY",
    "MARKET_FEATURE_FAMILY_REALIZED_VOLATILITY",
    "MARKET_FEATURE_FAMILY_RETURN_SUMMARY",
    "MARKET_FEATURE_FAMILY_STRESS",
    "PRIMITIVE_MARKET_FEATURE_COLUMNS",
    "PRIMITIVE_MARKET_FEATURE_COLUMNS_BY_FAMILY",
    "PRIMITIVE_MARKET_FEATURE_FAMILIES",
    "PrimitiveMarketFeatureConfig",
    "PrimitiveMarketFeatureResult",
    "UNAVAILABLE_MARKET_FEATURE_STATES",
    "build_primitive_market_regime_features",
]
