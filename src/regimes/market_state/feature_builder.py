from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from src.regimes.asset_state.dataset import LEAKAGE_COLUMN_TOKENS
from src.regimes.core.lineage import RegimeLineageSpec
from src.regimes.core.serialization import dumps_json, to_jsonable
from src.regimes.market_state.contracts import MARKET_STATE_SCHEMA_VERSION, MarketStateSchemaVersion, _schema_version
from src.regimes.market_state.covariance_features import (
    MarketCovarianceFeatureConfig,
    compute_core_correlation_covariance_features,
)
from src.regimes.market_state.data_contracts import (
    MARKET_STATE_DATASET_STATUS_READY,
    MarketStateDatasetBuildResult,
)
from src.regimes.market_state.feature_registry import (
    MarketFeatureRegistry,
    default_market_state_feature_registry,
)


MARKET_FEATURE_BUILD_STATUS_READY = "ready"
MARKET_FEATURE_BUILD_STATUS_BLOCKED = "blocked"

MARKET_FEATURE_BUILD_REASON_DATASET_NOT_READY = "dataset_not_ready"
MARKET_FEATURE_BUILD_REASON_EMPTY_DATASET = "empty_market_state_dataset"
MARKET_FEATURE_BUILD_REASON_LEAKAGE_RISK_COLUMNS = "leakage_risk_columns"
MARKET_FEATURE_BUILD_REASON_MISSING_CORE_BASKET = "missing_core_basket"
MARKET_FEATURE_BUILD_REASON_MISSING_BROAD_UNIVERSE = "missing_broad_universe"
MARKET_FEATURE_BUILD_REASON_BUILDER_EXCEPTION = "builder_exception"

TIMESTAMP_COLUMN = "ts"
ASSET_COLUMN = "asset"


@dataclass(frozen=True)
class MarketFeatureBuilderConfig:
    rolling_window: int = 20
    min_periods: int = 5
    trend_window: int = 20
    high_correlation_threshold: float = 0.7
    drawdown_stress_threshold: float = -0.05
    high_vol_quantile: float = 0.75
    include_timestamp_column: bool = True
    schema_version: int | MarketStateSchemaVersion = MARKET_STATE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if int(self.rolling_window) < 2:
            raise ValueError("Market-state feature rolling_window must be at least 2")
        if int(self.min_periods) < 2:
            raise ValueError("Market-state feature min_periods must be at least 2")
        if int(self.trend_window) < 2:
            raise ValueError("Market-state feature trend_window must be at least 2")
        if not 0.0 <= float(self.high_correlation_threshold) <= 1.0:
            raise ValueError("Market-state feature high_correlation_threshold must be within [0, 1]")
        if not 0.0 <= float(self.high_vol_quantile) <= 1.0:
            raise ValueError("Market-state feature high_vol_quantile must be within [0, 1]")
        object.__setattr__(self, "schema_version", _schema_version(self.schema_version))
        object.__setattr__(self, "rolling_window", int(self.rolling_window))
        object.__setattr__(self, "min_periods", int(self.min_periods))
        object.__setattr__(self, "trend_window", int(self.trend_window))
        object.__setattr__(self, "high_correlation_threshold", float(self.high_correlation_threshold))
        object.__setattr__(self, "drawdown_stress_threshold", float(self.drawdown_stress_threshold))
        object.__setattr__(self, "high_vol_quantile", float(self.high_vol_quantile))
        object.__setattr__(self, "include_timestamp_column", bool(self.include_timestamp_column))

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "rolling_window": int(self.rolling_window),
            "min_periods": int(self.min_periods),
            "trend_window": int(self.trend_window),
            "high_correlation_threshold": float(self.high_correlation_threshold),
            "drawdown_stress_threshold": float(self.drawdown_stress_threshold),
            "high_vol_quantile": float(self.high_vol_quantile),
            "include_timestamp_column": bool(self.include_timestamp_column),
        }


@dataclass
class MarketFeatureBuildResult:
    status: str
    reason_codes: Sequence[str]
    feature_matrix: pd.DataFrame = field(default_factory=pd.DataFrame, repr=False)
    feature_names: Sequence[str] = ()
    family_diagnostics: Mapping[str, Any] = field(default_factory=dict)
    unavailable_features: Mapping[str, str] = field(default_factory=dict)
    covariance_correlation_diagnostics: Mapping[str, Any] = field(default_factory=dict)
    registry: Mapping[str, Any] = field(default_factory=dict)
    config: Mapping[str, Any] = field(default_factory=dict)
    dataset_summary: Mapping[str, Any] = field(default_factory=dict)
    lineage_metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int | MarketStateSchemaVersion = MARKET_STATE_SCHEMA_VERSION
    metadata: Mapping[str, Any] = field(default_factory=dict)
    message: str | None = None

    def __post_init__(self) -> None:
        self.schema_version = _schema_version(self.schema_version)
        self.reason_codes = tuple(dict.fromkeys(str(reason) for reason in self.reason_codes))
        self.feature_names = tuple(str(feature) for feature in self.feature_names)

    @property
    def usable(self) -> bool:
        return self.status == MARKET_FEATURE_BUILD_STATUS_READY

    @property
    def timestamp_count(self) -> int:
        return int(self.feature_matrix.shape[0])

    @property
    def feature_count(self) -> int:
        return int(len(self.feature_names))

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "artifact_kind": "market_state_feature_build_result",
            "status": self.status,
            "reason_codes": list(self.reason_codes),
            "timestamp_count": self.timestamp_count,
            "feature_count": self.feature_count,
            "feature_names": list(self.feature_names),
            "feature_schema_hash": _feature_schema_hash(self.feature_names),
            "feature_matrix_shape": [int(self.feature_matrix.shape[0]), int(self.feature_matrix.shape[1])],
            "family_diagnostics": to_jsonable(dict(self.family_diagnostics)),
            "unavailable_features": to_jsonable(dict(sorted(self.unavailable_features.items()))),
            "covariance_correlation_diagnostics": to_jsonable(dict(self.covariance_correlation_diagnostics)),
            "registry": to_jsonable(dict(self.registry)),
            "config": to_jsonable(dict(self.config)),
            "dataset_summary": to_jsonable(dict(self.dataset_summary)),
            "lineage_metadata": to_jsonable(dict(self.lineage_metadata)),
            "metadata": to_jsonable(dict(self.metadata)),
            "message": self.message,
            "raw_asset_dimension_clustering_matrix": False,
            "production_feature_store_writes": False,
        }

    def to_json(self, **kwargs: Any) -> str:
        return dumps_json(self.as_dict(), **kwargs)


def build_market_state_features(
    dataset: MarketStateDatasetBuildResult,
    *,
    registry: MarketFeatureRegistry | None = None,
    config: MarketFeatureBuilderConfig | None = None,
) -> MarketFeatureBuildResult:
    cfg = config or MarketFeatureBuilderConfig()
    reg = registry or default_market_state_feature_registry()
    try:
        if dataset.status != MARKET_STATE_DATASET_STATUS_READY or not dataset.usable:
            return _blocked_feature_result(
                dataset,
                reg,
                cfg,
                [MARKET_FEATURE_BUILD_REASON_DATASET_NOT_READY],
                message=f"market-state dataset is not ready: {dataset.status}",
            )
        leakage_columns = _leakage_columns((*dataset.panel.columns, *dataset.long_panel.columns))
        if leakage_columns:
            return _blocked_feature_result(
                dataset,
                reg,
                cfg,
                [MARKET_FEATURE_BUILD_REASON_LEAKAGE_RISK_COLUMNS],
                message=f"market-state feature input contains leakage-risk columns: {leakage_columns}",
            )
        if dataset.long_panel.empty or TIMESTAMP_COLUMN not in dataset.long_panel.columns:
            return _blocked_feature_result(
                dataset,
                reg,
                cfg,
                [MARKET_FEATURE_BUILD_REASON_EMPTY_DATASET],
                message="market-state feature builder requires a non-empty long_panel",
            )
        if not dataset.selected_core_assets:
            return _blocked_feature_result(
                dataset,
                reg,
                cfg,
                [MARKET_FEATURE_BUILD_REASON_MISSING_CORE_BASKET],
                message="market-state feature builder requires core basket assets",
            )
        if not dataset.selected_broad_assets:
            return _blocked_feature_result(
                dataset,
                reg,
                cfg,
                [MARKET_FEATURE_BUILD_REASON_MISSING_BROAD_UNIVERSE],
                message="market-state feature builder requires broad universe assets",
            )

        long_panel = dataset.long_panel.copy()
        long_panel[TIMESTAMP_COLUMN] = pd.to_numeric(long_panel[TIMESTAMP_COLUMN], errors="coerce").astype("Int64")
        long_panel = long_panel.dropna(subset=[TIMESTAMP_COLUMN]).copy()
        long_panel[TIMESTAMP_COLUMN] = long_panel[TIMESTAMP_COLUMN].astype("int64")
        timestamps = tuple(int(ts) for ts in sorted(long_panel[TIMESTAMP_COLUMN].unique()))
        feature_matrix = pd.DataFrame({TIMESTAMP_COLUMN: timestamps})

        core_assets = tuple(dataset.selected_core_assets)
        broad_assets = tuple(dataset.selected_broad_assets)
        core = long_panel.loc[long_panel[ASSET_COLUMN].astype(str).isin(core_assets)].copy()
        broad = long_panel.loc[long_panel[ASSET_COLUMN].astype(str).isin(broad_assets)].copy()

        broad_returns = _pivot_series(broad, "log_return", timestamps, broad_assets)
        core_returns = _pivot_series(core, "log_return", timestamps, core_assets)
        broad_close = _pivot_series(broad, "close", timestamps, broad_assets)
        broad_volume = _pivot_series(broad, "volume", timestamps, broad_assets)
        broad_trades = _pivot_series(broad, "trades", timestamps, broad_assets)
        core_volume = _pivot_series(core, "volume", timestamps, core_assets)
        core_trades = _pivot_series(core, "trades", timestamps, core_assets)
        broad_vol = _pivot_series(broad, "realized_volatility_proxy", timestamps, broad_assets)
        broad_drawdown = _pivot_series(broad, "drawdown_proxy", timestamps, broad_assets)
        broad_downside = _pivot_series(broad, "downside_proxy", timestamps, broad_assets)

        _add_return_summary(feature_matrix, broad_returns, core_returns, core_volume, core_trades)
        _add_realized_volatility(feature_matrix, broad_returns, broad_vol, cfg)
        _add_breadth(feature_matrix, broad_returns, broad_close, broad_drawdown, broad_downside, cfg)
        _add_dispersion(feature_matrix, broad_returns, broad_vol)

        cov_cfg = MarketCovarianceFeatureConfig(
            rolling_window=cfg.rolling_window,
            min_periods=cfg.min_periods,
            high_correlation_threshold=cfg.high_correlation_threshold,
        )
        cov_features, cov_diagnostics, cov_unavailable = compute_core_correlation_covariance_features(core_returns, config=cov_cfg)
        cov_features = cov_features.reset_index(drop=True)
        feature_matrix = pd.concat([feature_matrix.reset_index(drop=True), cov_features.reset_index(drop=True)], axis=1)

        _add_liquidity_activity(feature_matrix, broad_volume, broad_trades)
        _add_stress(feature_matrix, broad_returns, broad_drawdown, feature_matrix, cfg)

        feature_names = tuple(column for column in feature_matrix.columns if column != TIMESTAMP_COLUMN)
        coverage_metadata = _coverage_metadata_frame(dataset)
        if not coverage_metadata.empty:
            feature_matrix = feature_matrix.merge(coverage_metadata, on=TIMESTAMP_COLUMN, how="left")
        unavailable = dict(cov_unavailable)
        family_diagnostics = _family_diagnostics(reg, feature_matrix, unavailable)
        for family_id, diag in family_diagnostics.items():
            for feature in diag.get("unavailable_features", []):
                unavailable.setdefault(str(feature), str(diag.get("unavailable_reason") or "all_values_missing"))

        if not cfg.include_timestamp_column:
            feature_matrix = feature_matrix.drop(columns=[TIMESTAMP_COLUMN])
        return MarketFeatureBuildResult(
            status=MARKET_FEATURE_BUILD_STATUS_READY,
            reason_codes=(),
            feature_matrix=feature_matrix,
            feature_names=feature_names,
            family_diagnostics=family_diagnostics,
            unavailable_features=unavailable,
            covariance_correlation_diagnostics=cov_diagnostics,
            registry=reg.as_dict(),
            config=cfg.as_dict(),
            dataset_summary=_dataset_summary(dataset),
            lineage_metadata=_feature_lineage_metadata(
                dataset=dataset,
                feature_names=feature_names,
                config=cfg,
                registry=reg,
            ),
            metadata={
                "market_level_scope": "whole_market_timestamp_window",
                "core_basket_assets": list(core_assets),
                "broad_universe_assets": list(broad_assets),
                "core_basket_used_for_correlation_covariance": True,
                "broad_universe_used_for_breadth_dispersion_activity": True,
                "coverage_metadata_columns": [
                    column for column in feature_matrix.columns if _is_coverage_metadata_column(column)
                ],
                "coverage_metadata_storage": "feature_matrix_and_axis_panel_diagnostics",
                "raw_asset_dimension_clustering_matrix": False,
                "production_feature_store_writes": False,
            },
        )
    except ValueError:
        raise
    except Exception as exc:
        return _blocked_feature_result(
            dataset,
            reg,
            cfg,
            [MARKET_FEATURE_BUILD_REASON_BUILDER_EXCEPTION],
            message=f"market-state feature builder failed closed: {exc}",
        )


def _add_return_summary(
    out: pd.DataFrame,
    broad_returns: pd.DataFrame,
    core_returns: pd.DataFrame,
    core_volume: pd.DataFrame,
    core_trades: pd.DataFrame,
) -> None:
    out["market_return_equal_weight"] = broad_returns.mean(axis=1, skipna=True)
    out["market_return_core_activity_weighted"] = _activity_weighted_return(core_returns, core_volume, core_trades)
    out["market_return_median"] = broad_returns.median(axis=1, skipna=True)
    out["market_return_q10"] = broad_returns.quantile(0.10, axis=1, interpolation="linear")
    out["market_return_q25"] = broad_returns.quantile(0.25, axis=1, interpolation="linear")
    out["market_return_q75"] = broad_returns.quantile(0.75, axis=1, interpolation="linear")
    out["market_return_q90"] = broad_returns.quantile(0.90, axis=1, interpolation="linear")


def _add_realized_volatility(
    out: pd.DataFrame,
    broad_returns: pd.DataFrame,
    broad_vol: pd.DataFrame,
    cfg: MarketFeatureBuilderConfig,
) -> None:
    out["market_realized_vol_proxy"] = out["market_return_equal_weight"].rolling(
        int(cfg.rolling_window),
        min_periods=int(cfg.min_periods),
    ).std()
    out["market_asset_vol_median"] = broad_vol.median(axis=1, skipna=True)
    out["market_asset_vol_q75"] = broad_vol.quantile(0.75, axis=1, interpolation="linear")
    out["market_asset_vol_q90"] = broad_vol.quantile(0.90, axis=1, interpolation="linear")
    threshold = broad_vol.quantile(float(cfg.high_vol_quantile), axis=1, interpolation="linear")
    out["market_high_vol_share"] = _row_share(broad_vol.ge(threshold, axis=0), broad_vol.notna())
    if not bool(out["market_asset_vol_median"].notna().any()) and not broad_returns.empty:
        out["market_asset_vol_median"] = broad_returns.abs().median(axis=1, skipna=True)
        out["market_asset_vol_q75"] = broad_returns.abs().quantile(0.75, axis=1, interpolation="linear")
        out["market_asset_vol_q90"] = broad_returns.abs().quantile(0.90, axis=1, interpolation="linear")


def _add_breadth(
    out: pd.DataFrame,
    broad_returns: pd.DataFrame,
    broad_close: pd.DataFrame,
    broad_drawdown: pd.DataFrame,
    broad_downside: pd.DataFrame,
    cfg: MarketFeatureBuilderConfig,
) -> None:
    out["market_breadth_up_share"] = _row_share(broad_returns > 0.0, broad_returns.notna())
    trend = broad_close.rolling(int(cfg.trend_window), min_periods=int(cfg.min_periods)).mean()
    out["market_breadth_above_trend_share"] = _row_share(broad_close > trend, broad_close.notna() & trend.notna())
    out["market_breadth_drawdown_share"] = _row_share(broad_drawdown <= float(cfg.drawdown_stress_threshold), broad_drawdown.notna())
    stress_condition = (broad_drawdown <= float(cfg.drawdown_stress_threshold)) | (broad_downside < 0.0)
    out["market_breadth_stress_share"] = _row_share(stress_condition, broad_drawdown.notna() | broad_downside.notna())


def _add_dispersion(out: pd.DataFrame, broad_returns: pd.DataFrame, broad_vol: pd.DataFrame) -> None:
    out["market_return_dispersion_std"] = broad_returns.std(axis=1, skipna=True)
    out["market_return_quantile_spread_q90_q10"] = (
        broad_returns.quantile(0.90, axis=1, interpolation="linear")
        - broad_returns.quantile(0.10, axis=1, interpolation="linear")
    )
    out["market_tail_return_spread_q95_q05"] = (
        broad_returns.quantile(0.95, axis=1, interpolation="linear")
        - broad_returns.quantile(0.05, axis=1, interpolation="linear")
    )
    out["market_vol_dispersion_std"] = broad_vol.std(axis=1, skipna=True)
    out["market_vol_quantile_spread_q90_q10"] = (
        broad_vol.quantile(0.90, axis=1, interpolation="linear")
        - broad_vol.quantile(0.10, axis=1, interpolation="linear")
    )


def _add_liquidity_activity(out: pd.DataFrame, broad_volume: pd.DataFrame, broad_trades: pd.DataFrame) -> None:
    out["market_volume_total"] = broad_volume.sum(axis=1, min_count=1)
    out["market_trades_total"] = broad_trades.sum(axis=1, min_count=1)
    activity_present = (broad_volume > 0.0) | (broad_trades > 0.0)
    activity_valid = broad_volume.notna() | broad_trades.notna()
    out["market_activity_breadth_share"] = _row_share(activity_present, activity_valid)
    out["market_volume_top_asset_share"] = _top_share(broad_volume)
    out["market_trade_top_asset_share"] = _top_share(broad_trades)


def _add_stress(
    out: pd.DataFrame,
    broad_returns: pd.DataFrame,
    broad_drawdown: pd.DataFrame,
    features: pd.DataFrame,
    cfg: MarketFeatureBuilderConfig,
) -> None:
    out["market_downside_breadth_share"] = _row_share(broad_returns < 0.0, broad_returns.notna())
    out["market_high_vol_high_corr_coincidence"] = (
        features["market_high_vol_share"] * features["market_corr_high_share"]
        if "market_corr_high_share" in features.columns
        else np.nan
    )
    out["market_broad_drawdown_participation"] = _row_share(broad_drawdown <= float(cfg.drawdown_stress_threshold), broad_drawdown.notna())
    components = out[[
        "market_downside_breadth_share",
        "market_high_vol_high_corr_coincidence",
        "market_broad_drawdown_participation",
    ]]
    out["market_stress_composite"] = components.mean(axis=1, skipna=True)


def _pivot_series(
    long_panel: pd.DataFrame,
    column: str,
    timestamps: Sequence[int],
    assets: Sequence[str],
) -> pd.DataFrame:
    if column not in long_panel.columns:
        return pd.DataFrame(index=range(len(timestamps)), columns=list(assets), dtype=float)
    pivot = long_panel.pivot_table(index=TIMESTAMP_COLUMN, columns=ASSET_COLUMN, values=column, aggfunc="last")
    pivot = pivot.reindex(index=list(timestamps), columns=list(assets))
    return pivot.apply(pd.to_numeric, errors="coerce").reset_index(drop=True)


def _activity_weighted_return(returns: pd.DataFrame, volume: pd.DataFrame, trades: pd.DataFrame) -> pd.Series:
    if _has_positive_values(volume):
        return _weighted_row_mean(returns, volume)
    if _has_positive_values(trades):
        return _weighted_row_mean(returns, trades)
    return returns.mean(axis=1, skipna=True)


def _weighted_row_mean(values: pd.DataFrame, weights: pd.DataFrame) -> pd.Series:
    valid = values.notna() & weights.notna() & (weights > 0.0)
    numerator = values.where(valid).mul(weights.where(valid)).sum(axis=1, min_count=1)
    denominator = weights.where(valid).sum(axis=1, min_count=1)
    weighted = numerator / denominator
    return weighted.where(denominator > 0.0, values.mean(axis=1, skipna=True))


def _row_share(condition: pd.DataFrame, valid: pd.DataFrame) -> pd.Series:
    denom = valid.sum(axis=1)
    num = condition.where(valid, False).sum(axis=1)
    return (num / denom).where(denom > 0, np.nan)


def _top_share(values: pd.DataFrame) -> pd.Series:
    positive = values.where(values > 0.0)
    total = positive.sum(axis=1, min_count=1)
    top = positive.max(axis=1, skipna=True)
    return (top / total).where(total > 0.0, np.nan)


def _has_positive_values(values: pd.DataFrame) -> bool:
    return bool((values > 0.0).any().any()) if not values.empty else False


def _family_diagnostics(
    registry: MarketFeatureRegistry,
    feature_matrix: pd.DataFrame,
    unavailable: Mapping[str, str],
) -> dict[str, Any]:
    diagnostics: dict[str, Any] = {}
    for family_id, spec in registry.families.items():
        feature_names = tuple(feature for feature in spec.output_features if feature in feature_matrix.columns)
        finite_counts = {
            feature: int(pd.to_numeric(feature_matrix[feature], errors="coerce").notna().sum())
            for feature in feature_names
        }
        missing = [feature for feature, count in finite_counts.items() if count <= 0]
        if not feature_names or len(missing) == len(feature_names):
            status = "unavailable"
        elif missing:
            status = "partial"
        else:
            status = "computed"
        diagnostics[family_id] = {
            "status": status,
            "scope": spec.scope,
            "uses_core_basket": bool(spec.uses_core_basket),
            "uses_broad_universe": bool(spec.uses_broad_universe),
            "feature_names": list(feature_names),
            "finite_counts": finite_counts,
            "unavailable_features": missing,
            "unavailable_reason": "all_values_missing" if missing else None,
            "preexisting_unavailable_reasons": {
                feature: reason for feature, reason in unavailable.items() if feature in feature_names
            },
        }
    return diagnostics


def _coverage_metadata_frame(dataset: MarketStateDatasetBuildResult) -> pd.DataFrame:
    rows = dict(dataset.metadata).get("timestamp_bucket_coverage")
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(list(rows))
    if frame.empty or TIMESTAMP_COLUMN not in frame.columns:
        return pd.DataFrame()
    frame[TIMESTAMP_COLUMN] = pd.to_numeric(frame[TIMESTAMP_COLUMN], errors="coerce")
    frame = frame.dropna(subset=[TIMESTAMP_COLUMN]).copy()
    frame[TIMESTAMP_COLUMN] = frame[TIMESTAMP_COLUMN].astype("int64")
    metadata_columns = [column for column in frame.columns if column == TIMESTAMP_COLUMN or _is_coverage_metadata_column(str(column))]
    return frame[metadata_columns].drop_duplicates(subset=[TIMESTAMP_COLUMN], keep="last").reset_index(drop=True)


def _is_coverage_metadata_column(column: str) -> bool:
    return any(
        str(column).endswith(suffix)
        for suffix in (
            "_requested_n",
            "_active_n",
            "_coverage_ratio",
            "_min_active_n",
            "_min_coverage_ratio",
            "_coverage_pass",
            "_late_entry_count",
            "_warmup_excluded_count",
            "_stale_or_no_trade_excluded_count",
        )
    )


def _blocked_feature_result(
    dataset: MarketStateDatasetBuildResult,
    registry: MarketFeatureRegistry,
    config: MarketFeatureBuilderConfig,
    reason_codes: Sequence[str],
    *,
    message: str | None = None,
) -> MarketFeatureBuildResult:
    return MarketFeatureBuildResult(
        status=MARKET_FEATURE_BUILD_STATUS_BLOCKED,
        reason_codes=tuple(reason_codes),
        registry=registry.as_dict(),
        config=config.as_dict(),
        dataset_summary=_dataset_summary(dataset),
        message=message,
    )


def _dataset_summary(dataset: MarketStateDatasetBuildResult) -> dict[str, Any]:
    return {
        "status": dataset.status,
        "timestamp_count": int(dataset.timestamp_count),
        "core_asset_count": int(dataset.core_asset_count),
        "broad_asset_count": int(dataset.broad_asset_count),
        "selected_core_assets": list(dataset.selected_core_assets),
        "selected_broad_assets": list(dataset.selected_broad_assets),
        "panel_shape": [int(dataset.panel.shape[0]), int(dataset.panel.shape[1])],
        "long_panel_shape": [int(dataset.long_panel.shape[0]), int(dataset.long_panel.shape[1])],
    }


def _feature_lineage_metadata(
    *,
    dataset: MarketStateDatasetBuildResult,
    feature_names: Sequence[str],
    config: MarketFeatureBuilderConfig,
    registry: MarketFeatureRegistry,
) -> dict[str, Any]:
    source_lineage = tuple(item.as_dict() for item in dataset.source_partition_lineage)
    if not source_lineage:
        source_lineage = (
            {
                "source": "market_state_dataset_build_result",
                "row_count": int(dataset.timestamp_count),
                "panel_shape": [int(dataset.panel.shape[0]), int(dataset.panel.shape[1])],
            },
        )
    score_start, score_end = _timestamp_bounds(_dataset_timestamps(dataset))
    train_start, train_end = _timestamp_bounds(dataset.train_timestamps or _dataset_timestamps(dataset))
    source_tail_values = [
        int(item.max_ts)
        for item in dataset.source_partition_lineage
        if item.max_ts is not None
    ]
    source_tail_ts = max(source_tail_values) if source_tail_values else score_end
    known_at = dict(dataset.known_at_metadata or {})
    generated_at = max(int(known_at.get("known_at_ts", score_end)), int(source_tail_ts), int(score_end))
    return RegimeLineageSpec(
        pathway="market_state",
        axis="market_feature_builder",
        band=dataset.band,
        interval=int(dataset.interval),
        profile_id=_feature_schema_hash(feature_names),
        feature_family_id="market_state_feature_builder_v1",
        clusterer_family="not_applicable_feature_builder",
        source_data_kind="market_state_aligned_dataset",
        source_partition_lineage=source_lineage,
        source_tail_ts=source_tail_ts,
        train_window_start=train_start,
        train_window_end=train_end,
        score_window_start=score_start,
        score_window_end=score_end,
        generated_at=generated_at,
        run_id=f"market_state_feature_builder:{_feature_schema_hash(feature_names)}",
        metadata={
            "registry_id": registry.registry_id,
            "rolling_window": int(config.rolling_window),
            "feature_count": int(len(feature_names)),
            "production_feature_store_writes": False,
            "clustering_enabled": False,
        },
    ).as_dict()


def _dataset_timestamps(dataset: MarketStateDatasetBuildResult) -> tuple[int, ...]:
    if dataset.panel.empty or TIMESTAMP_COLUMN not in dataset.panel.columns:
        return ()
    values = pd.to_numeric(dataset.panel[TIMESTAMP_COLUMN], errors="coerce").dropna()
    return tuple(int(value) for value in values.astype("int64").tolist())


def _timestamp_bounds(timestamps: Sequence[int]) -> tuple[int, int]:
    if not timestamps:
        return 0, 0
    values = tuple(int(ts) for ts in timestamps)
    return min(values), max(values)


def _leakage_columns(columns: Iterable[Any]) -> tuple[str, ...]:
    blocked: list[str] = []
    for column in columns:
        name = str(column).lower()
        if name in {TIMESTAMP_COLUMN, ASSET_COLUMN}:
            continue
        if any(token in name for token in LEAKAGE_COLUMN_TOKENS):
            blocked.append(str(column))
    return tuple(dict.fromkeys(blocked))


def _feature_schema_hash(feature_names: Sequence[str]) -> str:
    payload = "\n".join(str(name) for name in feature_names).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def write_market_state_feature_build_report(
    path: str | Path,
    result: MarketFeatureBuildResult,
) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(f".{out.name}.{os.getpid()}.tmp")
    tmp.write_text(result.to_json(indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, out)
    return out


__all__ = [
    "MARKET_FEATURE_BUILD_REASON_BUILDER_EXCEPTION",
    "MARKET_FEATURE_BUILD_REASON_DATASET_NOT_READY",
    "MARKET_FEATURE_BUILD_REASON_EMPTY_DATASET",
    "MARKET_FEATURE_BUILD_REASON_LEAKAGE_RISK_COLUMNS",
    "MARKET_FEATURE_BUILD_REASON_MISSING_BROAD_UNIVERSE",
    "MARKET_FEATURE_BUILD_REASON_MISSING_CORE_BASKET",
    "MARKET_FEATURE_BUILD_STATUS_BLOCKED",
    "MARKET_FEATURE_BUILD_STATUS_READY",
    "MarketFeatureBuildResult",
    "MarketFeatureBuilderConfig",
    "build_market_state_features",
    "write_market_state_feature_build_report",
]
