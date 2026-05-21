from __future__ import annotations

import hashlib
import math
import time
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from src.regimes.core.serialization import dumps_json, to_jsonable
from src.regimes.relationship_discovery.contracts import (
    AssetRelationshipProfile,
    RelationshipEdge,
)
from src.regimes.relationship_discovery.data_panel import (
    RELATIONSHIP_DATA_STATUS_REAL_DATA_LOADED,
    RelationshipReturnPanelResult,
)
from src.regimes.relationship_discovery.scope import RelationshipDiscoveryUniverse


METHOD_RAW_ROLLING_CORR = "raw_rolling_correlation"
METHOD_BETA_TO_CORE = "beta_to_core_basket"
METHOD_RESIDUAL_RETURN = "residual_return_vs_core_basket"
METHOD_RESIDUAL_CORR = "residual_correlation_after_core_removal"
METHOD_VOLATILITY_SIMILARITY = "volatility_similarity"
METHOD_FEATURE_DISTANCE = "feature_distance_similarity"
METHOD_LEAD_LAG_DIAGNOSTIC = "lead_lag_correlation_diagnostic"

RELATIONSHIP_METHOD_STATUS_COMPUTED = "computed"
RELATIONSHIP_METHOD_STATUS_INSUFFICIENT = "insufficient"
RELATIONSHIP_METHOD_STATUS_UNAVAILABLE = "unavailable"
RELATIONSHIP_METHOD_STATUS_SKIPPED = "skipped"

RELATIONSHIP_METHOD_STATUSES: tuple[str, ...] = (
    RELATIONSHIP_METHOD_STATUS_COMPUTED,
    RELATIONSHIP_METHOD_STATUS_INSUFFICIENT,
    RELATIONSHIP_METHOD_STATUS_UNAVAILABLE,
    RELATIONSHIP_METHOD_STATUS_SKIPPED,
)

DEFAULT_METHODS: tuple[str, ...] = (
    METHOD_RAW_ROLLING_CORR,
    METHOD_BETA_TO_CORE,
    METHOD_RESIDUAL_RETURN,
    METHOD_RESIDUAL_CORR,
    METHOD_VOLATILITY_SIMILARITY,
    METHOD_FEATURE_DISTANCE,
)


@dataclass(frozen=True)
class RelationshipMethodComparisonConfig:
    methods: Sequence[str] = DEFAULT_METHODS
    window_days: Sequence[int] = (30, 90, 180)
    observation_windows: Sequence[int] | None = None
    min_observations: int = 20
    min_coverage: float = 0.75
    top_k_per_asset: int = 5
    max_pair_count: int = 2500
    include_lead_lag_diagnostic: bool = False
    max_lead_lag_assets: int = 8
    edge_threshold: float = 0.5

    def __post_init__(self) -> None:
        methods = tuple(dict.fromkeys(str(method).strip() for method in self.methods if str(method).strip()))
        if not methods:
            raise ValueError("Relationship Discovery method comparison requires at least one method")
        windows = None
        if self.observation_windows is not None:
            windows = tuple(dict.fromkeys(max(2, int(window)) for window in self.observation_windows))
            if not windows:
                raise ValueError("Relationship Discovery observation_windows must include at least one value")
        days = tuple(dict.fromkeys(max(1, int(day)) for day in self.window_days))
        object.__setattr__(self, "methods", methods)
        object.__setattr__(self, "window_days", days)
        object.__setattr__(self, "observation_windows", windows)
        object.__setattr__(self, "min_observations", max(2, int(self.min_observations)))
        object.__setattr__(self, "min_coverage", _share(self.min_coverage, field_name="min_coverage"))
        object.__setattr__(self, "top_k_per_asset", max(1, int(self.top_k_per_asset)))
        object.__setattr__(self, "max_pair_count", max(1, int(self.max_pair_count)))
        object.__setattr__(self, "include_lead_lag_diagnostic", bool(self.include_lead_lag_diagnostic))
        object.__setattr__(self, "max_lead_lag_assets", max(2, int(self.max_lead_lag_assets)))
        object.__setattr__(self, "edge_threshold", max(0.0, float(self.edge_threshold)))

    def as_dict(self) -> dict[str, Any]:
        return {
            "methods": list(self.methods),
            "window_days": list(self.window_days),
            "observation_windows": list(self.observation_windows) if self.observation_windows is not None else None,
            "min_observations": int(self.min_observations),
            "min_coverage": float(self.min_coverage),
            "top_k_per_asset": int(self.top_k_per_asset),
            "max_pair_count": int(self.max_pair_count),
            "include_lead_lag_diagnostic": bool(self.include_lead_lag_diagnostic),
            "max_lead_lag_assets": int(self.max_lead_lag_assets),
            "edge_threshold": float(self.edge_threshold),
        }


@dataclass
class RelationshipMethodResult:
    method_id: str
    method_family: str
    status: str
    interval: int
    window: int
    edges: Sequence[RelationshipEdge] = ()
    profiles: Sequence[AssetRelationshipProfile] = ()
    diagnostics: Mapping[str, Any] = field(default_factory=dict)
    runtime_summary: Mapping[str, Any] = field(default_factory=dict)
    reason_codes: Sequence[str] = ()
    message: str | None = None

    def __post_init__(self) -> None:
        status = str(self.status).strip().lower()
        if status not in RELATIONSHIP_METHOD_STATUSES:
            raise ValueError(f"Unsupported Relationship Discovery method status {status!r}")
        self.status = status
        self.interval = int(self.interval)
        self.window = int(self.window)
        self.reason_codes = tuple(dict.fromkeys(str(reason) for reason in self.reason_codes))
        self.edges = tuple(self.edges)
        self.profiles = tuple(self.profiles)

    @property
    def computed(self) -> bool:
        return self.status == RELATIONSHIP_METHOD_STATUS_COMPUTED

    def as_dict(self) -> dict[str, Any]:
        return {
            "artifact_kind": "relationship_method_comparison_result",
            "method_id": self.method_id,
            "method_family": self.method_family,
            "status": self.status,
            "reason_codes": list(self.reason_codes),
            "message": self.message,
            "interval": int(self.interval),
            "window": int(self.window),
            "edge_count": len(self.edges),
            "profile_count": len(self.profiles),
            "edges": [edge.as_dict() for edge in self.edges],
            "profiles": [profile.as_dict() for profile in self.profiles],
            "diagnostics": to_jsonable(dict(self.diagnostics)),
            "runtime_summary": to_jsonable(dict(self.runtime_summary)),
            "production_writes_enabled": False,
        }


@dataclass
class RelationshipMethodComparisonResult:
    status: str
    interval: int
    methods: Sequence[RelationshipMethodResult]
    diagnostics: Mapping[str, Any] = field(default_factory=dict)
    runtime_summary: Mapping[str, Any] = field(default_factory=dict)
    reason_codes: Sequence[str] = ()
    message: str | None = None

    def __post_init__(self) -> None:
        status = str(self.status).strip().lower()
        if status not in RELATIONSHIP_METHOD_STATUSES:
            raise ValueError(f"Unsupported Relationship Discovery comparison status {status!r}")
        self.status = status
        self.interval = int(self.interval)
        self.methods = tuple(self.methods)
        self.reason_codes = tuple(dict.fromkeys(str(reason) for reason in self.reason_codes))

    @property
    def edge_count(self) -> int:
        return sum(len(result.edges) for result in self.methods)

    @property
    def profile_count(self) -> int:
        return sum(len(result.profiles) for result in self.methods)

    def as_dict(self) -> dict[str, Any]:
        return {
            "artifact_kind": "relationship_method_comparison",
            "status": self.status,
            "reason_codes": list(self.reason_codes),
            "message": self.message,
            "interval": int(self.interval),
            "edge_count": int(self.edge_count),
            "profile_count": int(self.profile_count),
            "methods": [result.as_dict() for result in self.methods],
            "diagnostics": to_jsonable(dict(self.diagnostics)),
            "runtime_summary": to_jsonable(dict(self.runtime_summary)),
            "production_writes_enabled": False,
            "broad_production_all_to_all_enabled": False,
        }


def compare_relationship_methods(
    panel_result: RelationshipReturnPanelResult,
    universe: RelationshipDiscoveryUniverse,
    *,
    config: RelationshipMethodComparisonConfig | None = None,
    feature_panel: pd.DataFrame | None = None,
) -> RelationshipMethodComparisonResult:
    config = config or RelationshipMethodComparisonConfig()
    started = time.perf_counter()
    if panel_result.status != RELATIONSHIP_DATA_STATUS_REAL_DATA_LOADED or panel_result.return_panel.empty:
        return RelationshipMethodComparisonResult(
            status=RELATIONSHIP_METHOD_STATUS_INSUFFICIENT,
            interval=panel_result.interval,
            methods=(),
            reason_codes=("return_panel_unavailable",),
            message=f"return panel status is {panel_result.status}",
            diagnostics={"panel_status": panel_result.status},
            runtime_summary=_runtime(started),
        )

    returns = _return_matrix(panel_result.return_panel)
    assets = tuple(column for column in returns.columns if column in set(universe.selected_assets))
    returns = returns.loc[:, list(assets)]
    if len(assets) < int(config.min_observations * 0 + 2):
        return RelationshipMethodComparisonResult(
            status=RELATIONSHIP_METHOD_STATUS_INSUFFICIENT,
            interval=panel_result.interval,
            methods=(),
            reason_codes=("insufficient_assets",),
            message="fewer than two assets available in return panel",
            diagnostics={"assets": list(assets)},
            runtime_summary=_runtime(started),
        )
    pair_count = len(assets) * (len(assets) - 1) // 2
    if pair_count > int(config.max_pair_count):
        return RelationshipMethodComparisonResult(
            status=RELATIONSHIP_METHOD_STATUS_SKIPPED,
            interval=panel_result.interval,
            methods=(),
            reason_codes=("bounded_pair_count_exceeded",),
            message=f"bounded pair count {pair_count} exceeds max_pair_count {config.max_pair_count}",
            diagnostics={"asset_count": len(assets), "pair_count": pair_count, "max_pair_count": int(config.max_pair_count)},
            runtime_summary=_runtime(started),
        )

    windows = _windows_for(panel_result.interval, int(returns.shape[0]), config)
    if not windows:
        return RelationshipMethodComparisonResult(
            status=RELATIONSHIP_METHOD_STATUS_INSUFFICIENT,
            interval=panel_result.interval,
            methods=(),
            reason_codes=("insufficient_window_history",),
            message="no configured method window has enough observations",
            diagnostics={"available_observations": int(returns.shape[0]), "config": config.as_dict()},
            runtime_summary=_runtime(started),
        )

    core_assets = tuple(asset for asset in universe.core_assets if asset in returns.columns)
    anchors = tuple(asset for asset in universe.anchors if asset in returns.columns)
    method_results: list[RelationshipMethodResult] = []
    method_names = list(config.methods)
    if config.include_lead_lag_diagnostic:
        method_names.append(METHOD_LEAD_LAG_DIAGNOSTIC)
    for window in windows:
        window_frame = returns.tail(int(window)).copy()
        for method in method_names:
            method_results.append(
                _run_method(
                    method,
                    window_frame,
                    interval=panel_result.interval,
                    window=int(window),
                    anchors=anchors,
                    core_assets=core_assets,
                    config=config,
                    feature_panel=feature_panel,
                )
            )
    status = RELATIONSHIP_METHOD_STATUS_COMPUTED if any(result.computed for result in method_results) else RELATIONSHIP_METHOD_STATUS_INSUFFICIENT
    return RelationshipMethodComparisonResult(
        status=status,
        interval=panel_result.interval,
        methods=method_results,
        reason_codes=("relationship_methods_compared",) if status == RELATIONSHIP_METHOD_STATUS_COMPUTED else ("no_methods_computed",),
        diagnostics={
            "asset_count": len(assets),
            "pair_count": pair_count,
            "windows": list(windows),
            "methods_requested": list(method_names),
            "max_pair_count": int(config.max_pair_count),
        },
        runtime_summary=_runtime(started),
    )


def _run_method(
    method: str,
    frame: pd.DataFrame,
    *,
    interval: int,
    window: int,
    anchors: Sequence[str],
    core_assets: Sequence[str],
    config: RelationshipMethodComparisonConfig,
    feature_panel: pd.DataFrame | None,
) -> RelationshipMethodResult:
    started = time.perf_counter()
    method = str(method).strip()
    method_id = f"{method}_{interval}_{window}"
    lineage_id = _lineage_id(method_id, frame.columns, interval=interval, window=window)
    known_at_ts = int(frame.index.max()) if len(frame.index) else 0
    if int(frame.shape[0]) < int(config.min_observations):
        return _method_blocked(
            method_id,
            method,
            interval,
            window,
            RELATIONSHIP_METHOD_STATUS_INSUFFICIENT,
            ("insufficient_observations",),
            f"window has {frame.shape[0]} observations, below min_observations {config.min_observations}",
            started,
        )
    if method in {METHOD_BETA_TO_CORE, METHOD_RESIDUAL_RETURN, METHOD_RESIDUAL_CORR} and len(core_assets) < 2:
        return _method_blocked(
            method_id,
            method,
            interval,
            window,
            RELATIONSHIP_METHOD_STATUS_INSUFFICIENT,
            ("insufficient_core_assets",),
            "core basket methods require at least two core assets in the panel",
            started,
        )
    if method == METHOD_FEATURE_DISTANCE and feature_panel is None:
        return _method_blocked(
            method_id,
            method,
            interval,
            window,
            RELATIONSHIP_METHOD_STATUS_UNAVAILABLE,
            ("feature_panel_unavailable",),
            "feature-distance similarity requires an explicit feature panel; scalar metadata alone is not enough",
            started,
        )

    if method == METHOD_RAW_ROLLING_CORR:
        edge_values = _pairwise_corr_edges(frame, method_id=method_id, relationship_type=method, known_at_ts=known_at_ts, lineage_id=lineage_id, config=config)
    elif method == METHOD_RESIDUAL_CORR:
        residuals = _residuals_vs_core(frame, core_assets=core_assets)
        edge_values = _pairwise_corr_edges(
            residuals,
            method_id=method_id,
            relationship_type=method,
            known_at_ts=known_at_ts,
            lineage_id=lineage_id,
            config=config,
        )
    elif method == METHOD_VOLATILITY_SIMILARITY:
        edge_values = _volatility_similarity_edges(frame, method_id=method_id, known_at_ts=known_at_ts, lineage_id=lineage_id, config=config)
    elif method == METHOD_BETA_TO_CORE:
        edge_values = _beta_to_core_edges(frame, core_assets=core_assets, method_id=method_id, known_at_ts=known_at_ts, lineage_id=lineage_id, config=config)
    elif method == METHOD_RESIDUAL_RETURN:
        edge_values = _residual_return_edges(frame, core_assets=core_assets, method_id=method_id, known_at_ts=known_at_ts, lineage_id=lineage_id, config=config)
    elif method == METHOD_FEATURE_DISTANCE:
        edge_values = _feature_distance_edges(feature_panel, frame, method_id=method_id, known_at_ts=known_at_ts, lineage_id=lineage_id, config=config)
    elif method == METHOD_LEAD_LAG_DIAGNOSTIC:
        edge_values = _lead_lag_edges(frame, method_id=method_id, known_at_ts=known_at_ts, lineage_id=lineage_id, config=config)
    else:
        return _method_blocked(
            method_id,
            method,
            interval,
            window,
            RELATIONSHIP_METHOD_STATUS_UNAVAILABLE,
            ("unknown_method",),
            f"unknown relationship method {method!r}",
            started,
        )

    if not edge_values:
        return _method_blocked(
            method_id,
            method,
            interval,
            window,
            RELATIONSHIP_METHOD_STATUS_INSUFFICIENT,
            ("no_edges_after_filters",),
            "method produced no edge candidates after sample and coverage filters",
            started,
        )
    profiles = _profiles_for(frame, edge_values, method_id=method_id, interval=interval, window=window, anchors=anchors, core_assets=core_assets, known_at_ts=known_at_ts, lineage_id=lineage_id, threshold=config.edge_threshold)
    return RelationshipMethodResult(
        method_id=method_id,
        method_family=method,
        status=RELATIONSHIP_METHOD_STATUS_COMPUTED,
        interval=interval,
        window=window,
        edges=edge_values,
        profiles=profiles,
        diagnostics={
            "sample_count": int(frame.shape[0]),
            "asset_count": int(frame.shape[1]),
            "edge_count": int(len(edge_values)),
            "profile_count": int(len(profiles)),
            "min_observations": int(config.min_observations),
            "min_coverage": float(config.min_coverage),
        },
        runtime_summary=_runtime(started),
        reason_codes=("computed",),
    )


def _pairwise_corr_edges(
    frame: pd.DataFrame,
    *,
    method_id: str,
    relationship_type: str,
    known_at_ts: int,
    lineage_id: str,
    config: RelationshipMethodComparisonConfig,
) -> tuple[RelationshipEdge, ...]:
    candidates: dict[str, list[tuple[str, float, int, float]]] = {asset: [] for asset in frame.columns}
    for left, right in _pairs(frame.columns):
        pair = frame[[left, right]].dropna()
        sample_count = int(pair.shape[0])
        coverage = sample_count / max(1, int(frame.shape[0]))
        if sample_count < config.min_observations or coverage < config.min_coverage:
            continue
        if float(pair[left].std()) <= 0.0 or float(pair[right].std()) <= 0.0:
            continue
        value = _finite_or_none(pair[left].corr(pair[right]))
        if value is None:
            continue
        candidates[left].append((right, value, sample_count, coverage))
        candidates[right].append((left, value, sample_count, coverage))
    return _top_pair_edges(candidates, method_id=method_id, relationship_type=relationship_type, interval=_interval_from_method_id(method_id), window=_window_from_method_id(method_id), known_at_ts=known_at_ts, lineage_id=lineage_id, top_k=config.top_k_per_asset)


def _volatility_similarity_edges(
    frame: pd.DataFrame,
    *,
    method_id: str,
    known_at_ts: int,
    lineage_id: str,
    config: RelationshipMethodComparisonConfig,
) -> tuple[RelationshipEdge, ...]:
    candidates: dict[str, list[tuple[str, float, int, float]]] = {asset: [] for asset in frame.columns}
    stds = frame.std(skipna=True)
    for left, right in _pairs(frame.columns):
        pair = frame[[left, right]].dropna()
        sample_count = int(pair.shape[0])
        coverage = sample_count / max(1, int(frame.shape[0]))
        if sample_count < config.min_observations or coverage < config.min_coverage:
            continue
        left_std = float(stds[left])
        right_std = float(stds[right])
        denom = max(abs(left_std), abs(right_std))
        value = 1.0 if denom == 0 else max(0.0, 1.0 - abs(left_std - right_std) / denom)
        candidates[left].append((right, value, sample_count, coverage))
        candidates[right].append((left, value, sample_count, coverage))
    return _top_pair_edges(candidates, method_id=method_id, relationship_type=METHOD_VOLATILITY_SIMILARITY, interval=_interval_from_method_id(method_id), window=_window_from_method_id(method_id), known_at_ts=known_at_ts, lineage_id=lineage_id, top_k=config.top_k_per_asset)


def _beta_to_core_edges(
    frame: pd.DataFrame,
    *,
    core_assets: Sequence[str],
    method_id: str,
    known_at_ts: int,
    lineage_id: str,
    config: RelationshipMethodComparisonConfig,
) -> tuple[RelationshipEdge, ...]:
    basket = _core_basket(frame, core_assets)
    edges: list[RelationshipEdge] = []
    for asset in frame.columns:
        pair = pd.concat([frame[asset], basket], axis=1, keys=[asset, "core_basket"]).dropna()
        sample_count = int(pair.shape[0])
        coverage = sample_count / max(1, int(frame.shape[0]))
        if sample_count < config.min_observations or coverage < config.min_coverage:
            continue
        var = float(pair["core_basket"].var())
        if var <= 0:
            continue
        beta = float(pair[asset].cov(pair["core_basket"]) / var)
        edges.append(_edge(asset, "core_basket", METHOD_BETA_TO_CORE, beta, sample_count, coverage, method_id, known_at_ts, lineage_id))
    return tuple(edges)


def _residual_return_edges(
    frame: pd.DataFrame,
    *,
    core_assets: Sequence[str],
    method_id: str,
    known_at_ts: int,
    lineage_id: str,
    config: RelationshipMethodComparisonConfig,
) -> tuple[RelationshipEdge, ...]:
    basket = _core_basket(frame, core_assets)
    edges: list[RelationshipEdge] = []
    for asset in frame.columns:
        pair = pd.concat([frame[asset], basket], axis=1, keys=[asset, "core_basket"]).dropna()
        sample_count = int(pair.shape[0])
        coverage = sample_count / max(1, int(frame.shape[0]))
        if sample_count < config.min_observations or coverage < config.min_coverage:
            continue
        residual = _asset_residual(pair[asset], pair["core_basket"])
        value = float(residual.mean())
        edges.append(_edge(asset, "core_basket", METHOD_RESIDUAL_RETURN, value, sample_count, coverage, method_id, known_at_ts, lineage_id))
    return tuple(edges)


def _feature_distance_edges(
    feature_panel: pd.DataFrame | None,
    return_frame: pd.DataFrame,
    *,
    method_id: str,
    known_at_ts: int,
    lineage_id: str,
    config: RelationshipMethodComparisonConfig,
) -> tuple[RelationshipEdge, ...]:
    if feature_panel is None or feature_panel.empty or "asset" not in feature_panel.columns:
        return ()
    features = feature_panel.copy()
    feature_cols = [column for column in features.columns if column not in {"asset", "ts"} and pd.api.types.is_numeric_dtype(features[column])]
    if not feature_cols:
        return ()
    if "ts" in features.columns:
        min_ts = int(return_frame.index.min())
        max_ts = int(return_frame.index.max())
        features = features.loc[(pd.to_numeric(features["ts"], errors="coerce") >= min_ts) & (pd.to_numeric(features["ts"], errors="coerce") <= max_ts)].copy()
    grouped = features.groupby("asset", dropna=False)[feature_cols].mean(numeric_only=True)
    grouped = grouped.reindex(return_frame.columns).dropna(how="any")
    if grouped.shape[0] < 2:
        return ()
    standardized = (grouped - grouped.mean()) / grouped.std(ddof=0).replace(0, 1.0)
    candidates: dict[str, list[tuple[str, float, int, float]]] = {asset: [] for asset in standardized.index}
    for left, right in _pairs(standardized.index):
        dist = float(np.linalg.norm(standardized.loc[left].to_numpy() - standardized.loc[right].to_numpy()))
        value = 1.0 / (1.0 + dist)
        candidates[left].append((right, value, int(return_frame.shape[0]), 1.0))
        candidates[right].append((left, value, int(return_frame.shape[0]), 1.0))
    return _top_pair_edges(candidates, method_id=method_id, relationship_type=METHOD_FEATURE_DISTANCE, interval=_interval_from_method_id(method_id), window=_window_from_method_id(method_id), known_at_ts=known_at_ts, lineage_id=lineage_id, top_k=config.top_k_per_asset)


def _lead_lag_edges(
    frame: pd.DataFrame,
    *,
    method_id: str,
    known_at_ts: int,
    lineage_id: str,
    config: RelationshipMethodComparisonConfig,
) -> tuple[RelationshipEdge, ...]:
    scoped_assets = tuple(frame.columns[: int(config.max_lead_lag_assets)])
    candidates: dict[str, list[tuple[str, float, int, float]]] = {asset: [] for asset in scoped_assets}
    for left, right in _pairs(scoped_assets):
        pair = pd.concat([frame[left], frame[right].shift(1)], axis=1, keys=[left, right]).dropna()
        sample_count = int(pair.shape[0])
        coverage = sample_count / max(1, int(frame.shape[0]))
        if sample_count < config.min_observations or coverage < config.min_coverage:
            continue
        if float(pair[left].std()) <= 0.0 or float(pair[right].std()) <= 0.0:
            continue
        value = _finite_or_none(pair[left].corr(pair[right]))
        if value is None:
            continue
        candidates[left].append((f"{right}:lag1", value, sample_count, coverage))
    return _top_pair_edges(candidates, method_id=method_id, relationship_type=METHOD_LEAD_LAG_DIAGNOSTIC, interval=_interval_from_method_id(method_id), window=_window_from_method_id(method_id), known_at_ts=known_at_ts, lineage_id=lineage_id, top_k=config.top_k_per_asset)


def _profiles_for(
    frame: pd.DataFrame,
    edges: Sequence[RelationshipEdge],
    *,
    method_id: str,
    interval: int,
    window: int,
    anchors: Sequence[str],
    core_assets: Sequence[str],
    known_at_ts: int,
    lineage_id: str,
    threshold: float,
) -> tuple[AssetRelationshipProfile, ...]:
    edges_by_asset: dict[str, list[RelationshipEdge]] = {}
    for edge in edges:
        edges_by_asset.setdefault(edge.asset, []).append(edge)
    basket = _core_basket(frame, core_assets) if len(core_assets) >= 2 else pd.Series(index=frame.index, dtype="float64")
    profiles: list[AssetRelationshipProfile] = []
    for asset in frame.columns:
        asset_edges = edges_by_asset.get(asset, [])
        strengths = [abs(float(edge.value)) for edge in asset_edges]
        top_strength = max(strengths) if strengths else 0.0
        total = sum(strengths)
        probs = [value / total for value in strengths if total > 0 and value > 0]
        entropy = -sum(p * math.log(p) for p in probs) if probs else 0.0
        corr_primary = _corr(frame[asset], frame[anchors[0]]) if len(anchors) >= 1 and anchors[0] in frame.columns else 0.0
        corr_secondary = _corr(frame[asset], frame[anchors[1]]) if len(anchors) >= 2 and anchors[1] in frame.columns else 0.0
        beta = _beta(frame[asset], basket) if not basket.empty else 0.0
        residual = _asset_residual_pair(frame[asset], basket) if not basket.empty else pd.Series(dtype="float64")
        profiles.append(
            AssetRelationshipProfile(
                asset=asset,
                interval=interval,
                window=window,
                method_id=method_id,
                corr_to_anchor_primary=_bounded_corr(corr_primary),
                corr_to_anchor_secondary=_bounded_corr(corr_secondary),
                beta_to_core_basket=_finite(beta, default=0.0),
                residual_return_vs_core=_finite(float(residual.mean()) if not residual.empty else 0.0, default=0.0),
                residual_volatility_vs_core=max(0.0, _finite(float(residual.std()) if not residual.empty else 0.0, default=0.0)),
                top_relationship_strength=min(1.0, max(0.0, top_strength)),
                relationship_concentration=min(1.0, max(0.0, top_strength / total if total > 0 else 0.0)),
                relationship_entropy=max(0.0, entropy),
                relationship_count_above_threshold=sum(1 for value in strengths if value >= threshold),
                stability_summary={
                    "prototype_only": True,
                    "method_id": method_id,
                    "edge_count": len(asset_edges),
                    "threshold": float(threshold),
                },
                known_at_ts=known_at_ts,
                lineage_id=lineage_id,
            )
        )
    return tuple(profiles)


def _residuals_vs_core(frame: pd.DataFrame, *, core_assets: Sequence[str]) -> pd.DataFrame:
    basket = _core_basket(frame, core_assets)
    residuals = {}
    for asset in frame.columns:
        residuals[asset] = _asset_residual_pair(frame[asset], basket)
    return pd.DataFrame(residuals)


def _core_basket(frame: pd.DataFrame, core_assets: Sequence[str]) -> pd.Series:
    cols = [asset for asset in core_assets if asset in frame.columns]
    return frame[cols].mean(axis=1, skipna=True)


def _asset_residual_pair(asset_returns: pd.Series, basket: pd.Series) -> pd.Series:
    pair = pd.concat([asset_returns, basket], axis=1, keys=["asset", "basket"]).dropna()
    if pair.empty:
        return pd.Series(dtype="float64")
    return _asset_residual(pair["asset"], pair["basket"])


def _asset_residual(asset_returns: pd.Series, basket: pd.Series) -> pd.Series:
    var = float(basket.var())
    if var <= 0:
        return asset_returns - float(asset_returns.mean())
    beta = float(asset_returns.cov(basket) / var)
    alpha = float(asset_returns.mean() - beta * basket.mean())
    return asset_returns - (alpha + beta * basket)


def _top_pair_edges(
    candidates: Mapping[str, Sequence[tuple[str, float, int, float]]],
    *,
    method_id: str,
    relationship_type: str,
    interval: int,
    window: int,
    known_at_ts: int,
    lineage_id: str,
    top_k: int,
) -> tuple[RelationshipEdge, ...]:
    edges: list[RelationshipEdge] = []
    for asset, values in candidates.items():
        ranked = sorted(values, key=lambda item: (-abs(float(item[1])), str(item[0])))[: int(top_k)]
        for related, value, sample_count, coverage in ranked:
            edges.append(_edge(asset, related, relationship_type, value, sample_count, coverage, method_id, known_at_ts, lineage_id, interval=interval, window=window))
    return tuple(edges)


def _edge(
    asset: str,
    related: str,
    relationship_type: str,
    value: float,
    sample_count: int,
    coverage: float,
    method_id: str,
    known_at_ts: int,
    lineage_id: str,
    *,
    interval: int | None = None,
    window: int | None = None,
) -> RelationshipEdge:
    value = _finite(float(value), default=0.0)
    return RelationshipEdge(
        ts=known_at_ts,
        interval=interval if interval is not None else _interval_from_method_id(method_id),
        window=window if window is not None else _window_from_method_id(method_id),
        asset=asset,
        related_asset_or_benchmark=related,
        relationship_type=relationship_type,
        value=value,
        abs_value=abs(value),
        direction="positive" if value > 0 else "negative" if value < 0 else "neutral",
        sample_count=int(sample_count),
        coverage=min(1.0, max(0.0, float(coverage))),
        method_id=method_id,
        known_at_ts=known_at_ts,
        lineage_id=lineage_id,
    )


def _method_blocked(
    method_id: str,
    method_family: str,
    interval: int,
    window: int,
    status: str,
    reason_codes: Sequence[str],
    message: str,
    started: float,
) -> RelationshipMethodResult:
    return RelationshipMethodResult(
        method_id=method_id,
        method_family=method_family,
        status=status,
        interval=interval,
        window=window,
        reason_codes=reason_codes,
        message=message,
        runtime_summary=_runtime(started),
        diagnostics={"edge_count": 0, "profile_count": 0},
    )


def _windows_for(interval: int, available_observations: int, config: RelationshipMethodComparisonConfig) -> tuple[int, ...]:
    if config.observation_windows is not None:
        candidates = tuple(int(window) for window in config.observation_windows)
    else:
        observations_per_day = max(1, int(round(1440 / int(interval))))
        candidates = tuple(int(day) * observations_per_day for day in config.window_days)
    valid = [window for window in candidates if window <= int(available_observations) and window >= int(config.min_observations)]
    return tuple(dict.fromkeys(valid))


def _return_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "ts" in out.columns:
        out["ts"] = pd.to_numeric(out["ts"], errors="coerce")
        out = out.dropna(subset=["ts"]).copy()
        out["ts"] = out["ts"].astype("int64")
        out = out.set_index("ts")
    for column in out.columns:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    return out.sort_index()


def _pairs(values: Sequence[str]) -> tuple[tuple[str, str], ...]:
    items = tuple(str(value) for value in values)
    return tuple((items[i], items[j]) for i in range(len(items)) for j in range(i + 1, len(items)))


def _corr(left: pd.Series, right: pd.Series) -> float:
    pair = pd.concat([left, right], axis=1).dropna()
    if pair.shape[0] < 2:
        return 0.0
    if float(pair.iloc[:, 0].std()) <= 0.0 or float(pair.iloc[:, 1].std()) <= 0.0:
        return 0.0
    return _finite(float(pair.iloc[:, 0].corr(pair.iloc[:, 1])), default=0.0)


def _beta(asset_returns: pd.Series, basket: pd.Series) -> float:
    pair = pd.concat([asset_returns, basket], axis=1, keys=["asset", "basket"]).dropna()
    if pair.shape[0] < 2:
        return 0.0
    var = float(pair["basket"].var())
    return 0.0 if var <= 0 else float(pair["asset"].cov(pair["basket"]) / var)


def _bounded_corr(value: float) -> float:
    return min(1.0, max(-1.0, _finite(value, default=0.0)))


def _finite(value: float, *, default: float) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    if math.isnan(out) or math.isinf(out):
        return float(default)
    return out


def _finite_or_none(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    if math.isnan(out) or math.isinf(out):
        return None
    return out


def _share(value: object, *, field_name: str) -> float:
    try:
        out = float(value)
    except Exception as exc:
        raise ValueError(f"Relationship Discovery {field_name} must be numeric") from exc
    if out < 0.0 or out > 1.0:
        raise ValueError(f"Relationship Discovery {field_name} must be between 0 and 1")
    return out


def _lineage_id(method_id: str, assets: Sequence[str], *, interval: int, window: int) -> str:
    payload = {"method_id": method_id, "assets": list(assets), "interval": int(interval), "window": int(window)}
    return hashlib.sha256(dumps_json(payload).encode("utf-8")).hexdigest()[:16]


def _interval_from_method_id(method_id: str) -> int:
    return int(str(method_id).rsplit("_", 2)[-2])


def _window_from_method_id(method_id: str) -> int:
    return int(str(method_id).rsplit("_", 1)[-1])


def _runtime(started: float) -> dict[str, Any]:
    return {"runtime_seconds": float(max(0.0, time.perf_counter() - started))}


__all__ = [
    "DEFAULT_METHODS",
    "METHOD_BETA_TO_CORE",
    "METHOD_FEATURE_DISTANCE",
    "METHOD_LEAD_LAG_DIAGNOSTIC",
    "METHOD_RAW_ROLLING_CORR",
    "METHOD_RESIDUAL_CORR",
    "METHOD_RESIDUAL_RETURN",
    "METHOD_VOLATILITY_SIMILARITY",
    "RELATIONSHIP_METHOD_STATUS_COMPUTED",
    "RELATIONSHIP_METHOD_STATUS_INSUFFICIENT",
    "RELATIONSHIP_METHOD_STATUS_SKIPPED",
    "RELATIONSHIP_METHOD_STATUS_UNAVAILABLE",
    "RELATIONSHIP_METHOD_STATUSES",
    "RelationshipMethodComparisonConfig",
    "RelationshipMethodComparisonResult",
    "RelationshipMethodResult",
    "compare_relationship_methods",
]
