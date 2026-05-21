from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from src.regimes.core.serialization import to_jsonable
from src.regimes.relationship_discovery.scope import RelationshipDiscoveryUniverse


RELATIONSHIP_STORAGE_SCALE_SMALL = "small"
RELATIONSHIP_STORAGE_SCALE_MODERATE = "moderate"
RELATIONSHIP_STORAGE_SCALE_LARGE = "large"
RELATIONSHIP_STORAGE_SCALE_EXPLOSIVE = "explosive"

RELATIONSHIP_STORAGE_RECOMMEND_PERSIST = "persist"
RELATIONSHIP_STORAGE_RECOMMEND_PROTOTYPE = "prototype"
RELATIONSHIP_STORAGE_RECOMMEND_GATED = "gated"
RELATIONSHIP_STORAGE_RECOMMEND_AVOID = "avoid"


@dataclass(frozen=True)
class RelationshipStorageEstimateConfig:
    intervals: Sequence[int] = (240, 1440)
    windows: Sequence[int] = (30, 90, 180)
    refit_keys: int = 365
    metrics_per_pair: int = 6
    top_k_per_asset: int = 5
    full_broad_asset_count: int | None = None

    def __post_init__(self) -> None:
        intervals = tuple(dict.fromkeys(int(interval) for interval in self.intervals))
        windows = tuple(dict.fromkeys(int(window) for window in self.windows))
        if not intervals:
            raise ValueError("Relationship Discovery storage estimate requires at least one interval")
        if not windows:
            raise ValueError("Relationship Discovery storage estimate requires at least one window")
        if any(interval < 60 for interval in intervals):
            raise ValueError("Relationship Discovery storage estimate does not support sub-hour intervals")
        object.__setattr__(self, "intervals", intervals)
        object.__setattr__(self, "windows", windows)
        object.__setattr__(self, "refit_keys", max(1, int(self.refit_keys)))
        object.__setattr__(self, "metrics_per_pair", max(1, int(self.metrics_per_pair)))
        object.__setattr__(self, "top_k_per_asset", max(1, int(self.top_k_per_asset)))
        if self.full_broad_asset_count is not None:
            object.__setattr__(self, "full_broad_asset_count", max(1, int(self.full_broad_asset_count)))

    def as_dict(self) -> dict[str, Any]:
        return {
            "intervals": list(self.intervals),
            "windows": list(self.windows),
            "refit_keys": int(self.refit_keys),
            "metrics_per_pair": int(self.metrics_per_pair),
            "top_k_per_asset": int(self.top_k_per_asset),
            "full_broad_asset_count": self.full_broad_asset_count,
        }


@dataclass(frozen=True)
class RelationshipStorageEstimateRow:
    scenario: str
    asset_count: int
    pair_count: int
    intervals: Sequence[int]
    windows: Sequence[int]
    timestamps_or_refit_keys: int
    rows_per_metric: int
    total_rows_all_metrics: int
    expected_scale_class: str
    recommendation: str
    pair_policy: str
    notes: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "asset_count", max(0, int(self.asset_count)))
        object.__setattr__(self, "pair_count", max(0, int(self.pair_count)))
        object.__setattr__(self, "intervals", tuple(int(interval) for interval in self.intervals))
        object.__setattr__(self, "windows", tuple(int(window) for window in self.windows))
        object.__setattr__(self, "timestamps_or_refit_keys", max(1, int(self.timestamps_or_refit_keys)))
        object.__setattr__(self, "rows_per_metric", max(0, int(self.rows_per_metric)))
        object.__setattr__(self, "total_rows_all_metrics", max(0, int(self.total_rows_all_metrics)))

    def as_dict(self) -> dict[str, Any]:
        return {
            "scenario": self.scenario,
            "asset_count": int(self.asset_count),
            "pair_count": int(self.pair_count),
            "intervals": list(self.intervals),
            "windows": list(self.windows),
            "timestamps_or_refit_keys": int(self.timestamps_or_refit_keys),
            "rows_per_metric": int(self.rows_per_metric),
            "total_rows_all_metrics": int(self.total_rows_all_metrics),
            "expected_scale_class": self.expected_scale_class,
            "recommendation": self.recommendation,
            "pair_policy": self.pair_policy,
            "notes": self.notes,
        }


@dataclass(frozen=True)
class RelationshipStorageEstimateResult:
    estimates: Sequence[RelationshipStorageEstimateRow]
    config: RelationshipStorageEstimateConfig
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "estimates", tuple(self.estimates))

    def as_dict(self) -> dict[str, Any]:
        return {
            "artifact_kind": "relationship_storage_estimate",
            "estimates": [estimate.as_dict() for estimate in self.estimates],
            "config": self.config.as_dict(),
            "diagnostics": to_jsonable(dict(self.diagnostics)),
            "estimation_goal": "classify_storage_risk_not_precise_byte_accounting",
            "production_enabled": False,
        }


def estimate_relationship_storage(
    universe: RelationshipDiscoveryUniverse | None,
    *,
    config: RelationshipStorageEstimateConfig | None = None,
) -> RelationshipStorageEstimateResult:
    config = config or RelationshipStorageEstimateConfig()
    anchors = tuple(universe.anchors) if universe is not None else ()
    core = tuple(universe.core_assets) if universe is not None else ()
    broad_sample = tuple(universe.broad_sample_assets) if universe is not None else ()
    selected = tuple(universe.selected_assets) if universe is not None else tuple(dict.fromkeys((*anchors, *core, *broad_sample)))

    anchor_core_assets = tuple(dict.fromkeys((*anchors, *core)))
    bounded_assets = tuple(dict.fromkeys((*anchor_core_assets, *broad_sample))) or selected
    full_broad_count = config.full_broad_asset_count
    if full_broad_count is None:
        full_broad_count = max(250, len(bounded_assets))

    rows = (
        _row(
            "anchors_core_only",
            asset_count=len(anchor_core_assets),
            pair_count=len(anchor_core_assets) * max(1, len(anchors)),
            config=config,
            pair_policy="anchor/core asset to anchor benchmark facts",
            recommendation=RELATIONSHIP_STORAGE_RECOMMEND_PERSIST,
            notes="Suitable for durable manifests and later feature-generator consumption.",
        ),
        _row(
            "effective_core_internal_relationships",
            asset_count=len(core),
            pair_count=_undirected_pairs(len(core)),
            config=config,
            pair_policy="undirected all-to-all within effective core",
            recommendation=RELATIONSHIP_STORAGE_RECOMMEND_PERSIST,
            notes="Small enough to persist if method semantics remain stable.",
        ),
        _row(
            "bounded_broad_sample",
            asset_count=len(bounded_assets),
            pair_count=_undirected_pairs(len(bounded_assets)),
            config=config,
            pair_policy="undirected all-to-all within approved bounded scope",
            recommendation=RELATIONSHIP_STORAGE_RECOMMEND_PROTOTYPE,
            notes="Appropriate for discovery artifacts; avoid treating as dynamic peer production.",
        ),
        _row(
            "full_broad_all_to_all_hypothetical",
            asset_count=full_broad_count,
            pair_count=_undirected_pairs(full_broad_count),
            config=config,
            pair_policy="hypothetical undirected all-to-all across full broad universe",
            recommendation=RELATIONSHIP_STORAGE_RECOMMEND_AVOID,
            notes="Use only as a risk estimate; broad all-to-all production is outside this sprint.",
        ),
        _row(
            "top_k_selected_edges_hypothetical",
            asset_count=full_broad_count,
            pair_count=full_broad_count * int(config.top_k_per_asset),
            config=config,
            pair_policy="directed top-K selected edges per asset",
            recommendation=RELATIONSHIP_STORAGE_RECOMMEND_GATED,
            notes="Persist only behind selection gates and explicit non-peer-membership semantics.",
        ),
    )
    return RelationshipStorageEstimateResult(
        estimates=rows,
        config=config,
        diagnostics={
            "anchor_count": len(anchors),
            "core_count": len(core),
            "bounded_broad_sample_count": len(broad_sample),
            "bounded_selected_asset_count": len(bounded_assets),
            "full_broad_asset_count_assumption": int(full_broad_count),
        },
    )


def _row(
    scenario: str,
    *,
    asset_count: int,
    pair_count: int,
    config: RelationshipStorageEstimateConfig,
    pair_policy: str,
    recommendation: str,
    notes: str,
) -> RelationshipStorageEstimateRow:
    rows_per_metric = int(pair_count) * len(config.intervals) * len(config.windows) * int(config.refit_keys)
    total_rows = rows_per_metric * int(config.metrics_per_pair)
    scale = _scale_class(total_rows)
    final_recommendation = _recommendation(recommendation, scale=scale, scenario=scenario)
    return RelationshipStorageEstimateRow(
        scenario=scenario,
        asset_count=int(asset_count),
        pair_count=int(pair_count),
        intervals=config.intervals,
        windows=config.windows,
        timestamps_or_refit_keys=int(config.refit_keys),
        rows_per_metric=rows_per_metric,
        total_rows_all_metrics=total_rows,
        expected_scale_class=scale,
        recommendation=final_recommendation,
        pair_policy=pair_policy,
        notes=notes,
    )


def _scale_class(total_rows: int) -> str:
    if total_rows <= 10_000:
        return RELATIONSHIP_STORAGE_SCALE_SMALL
    if total_rows <= 1_000_000:
        return RELATIONSHIP_STORAGE_SCALE_MODERATE
    if total_rows <= 50_000_000:
        return RELATIONSHIP_STORAGE_SCALE_LARGE
    return RELATIONSHIP_STORAGE_SCALE_EXPLOSIVE


def _recommendation(default: str, *, scale: str, scenario: str) -> str:
    if scenario == "full_broad_all_to_all_hypothetical":
        return RELATIONSHIP_STORAGE_RECOMMEND_AVOID if scale in {RELATIONSHIP_STORAGE_SCALE_LARGE, RELATIONSHIP_STORAGE_SCALE_EXPLOSIVE} else RELATIONSHIP_STORAGE_RECOMMEND_GATED
    if scale == RELATIONSHIP_STORAGE_SCALE_EXPLOSIVE:
        return RELATIONSHIP_STORAGE_RECOMMEND_AVOID
    if scale == RELATIONSHIP_STORAGE_SCALE_LARGE:
        return RELATIONSHIP_STORAGE_RECOMMEND_GATED
    return default


def _undirected_pairs(asset_count: int) -> int:
    count = max(0, int(asset_count))
    return count * max(0, count - 1) // 2


__all__ = [
    "RELATIONSHIP_STORAGE_RECOMMEND_AVOID",
    "RELATIONSHIP_STORAGE_RECOMMEND_GATED",
    "RELATIONSHIP_STORAGE_RECOMMEND_PERSIST",
    "RELATIONSHIP_STORAGE_RECOMMEND_PROTOTYPE",
    "RELATIONSHIP_STORAGE_SCALE_EXPLOSIVE",
    "RELATIONSHIP_STORAGE_SCALE_LARGE",
    "RELATIONSHIP_STORAGE_SCALE_MODERATE",
    "RELATIONSHIP_STORAGE_SCALE_SMALL",
    "RelationshipStorageEstimateConfig",
    "RelationshipStorageEstimateResult",
    "RelationshipStorageEstimateRow",
    "estimate_relationship_storage",
]
