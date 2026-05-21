from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from src.regimes.core.serialization import to_jsonable
from src.regimes.relationship_discovery.contracts import RelationshipEdge, RelationshipStabilityScore


RELATIONSHIP_SELECTION_STATUS_STABLE_CANDIDATE = "stable_candidate"
RELATIONSHIP_SELECTION_STATUS_UNSTABLE_CANDIDATE = "unstable_candidate"
RELATIONSHIP_SELECTION_STATUS_INSUFFICIENT_COVERAGE = "insufficient_coverage"
RELATIONSHIP_SELECTION_STATUS_MARKET_MODE_ONLY = "market_mode_only"
RELATIONSHIP_SELECTION_STATUS_ISOLATED_ASSET = "isolated_asset"
RELATIONSHIP_SELECTION_STATUS_NEEDS_RESEARCH = "needs_research"

RELATIONSHIP_SELECTION_STATUSES: tuple[str, ...] = (
    RELATIONSHIP_SELECTION_STATUS_STABLE_CANDIDATE,
    RELATIONSHIP_SELECTION_STATUS_UNSTABLE_CANDIDATE,
    RELATIONSHIP_SELECTION_STATUS_INSUFFICIENT_COVERAGE,
    RELATIONSHIP_SELECTION_STATUS_MARKET_MODE_ONLY,
    RELATIONSHIP_SELECTION_STATUS_ISOLATED_ASSET,
    RELATIONSHIP_SELECTION_STATUS_NEEDS_RESEARCH,
)


@dataclass(frozen=True)
class RelationshipStabilityConfig:
    min_survival_count: int = 2
    min_survival_share: float = 0.5
    min_sign_stability: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "min_survival_count", max(1, int(self.min_survival_count)))
        object.__setattr__(self, "min_survival_share", _share(self.min_survival_share, field_name="min_survival_share"))
        object.__setattr__(self, "min_sign_stability", _share(self.min_sign_stability, field_name="min_sign_stability"))

    def as_dict(self) -> dict[str, Any]:
        return {
            "min_survival_count": int(self.min_survival_count),
            "min_survival_share": float(self.min_survival_share),
            "min_sign_stability": float(self.min_sign_stability),
        }


@dataclass(frozen=True)
class RelationshipStabilityResult:
    scores: Sequence[RelationshipStabilityScore]
    asset_statuses: Mapping[str, str] = field(default_factory=dict)
    stable_edges_per_asset: Mapping[str, int] = field(default_factory=dict)
    isolated_assets: Sequence[str] = ()
    noisy_assets: Sequence[str] = ()
    diagnostics: Mapping[str, Any] = field(default_factory=dict)
    config: RelationshipStabilityConfig = field(default_factory=RelationshipStabilityConfig)

    def __post_init__(self) -> None:
        object.__setattr__(self, "scores", tuple(self.scores))
        object.__setattr__(self, "isolated_assets", tuple(str(asset) for asset in self.isolated_assets))
        object.__setattr__(self, "noisy_assets", tuple(str(asset) for asset in self.noisy_assets))

    def as_dict(self) -> dict[str, Any]:
        return {
            "artifact_kind": "relationship_stability_summary",
            "status": "computed" if self.scores else "no_candidate_edges",
            "score_count": len(self.scores),
            "stable_edge_count": sum(1 for score in self.scores if score.activation_status == RELATIONSHIP_SELECTION_STATUS_STABLE_CANDIDATE),
            "unstable_edge_count": sum(1 for score in self.scores if score.activation_status == RELATIONSHIP_SELECTION_STATUS_UNSTABLE_CANDIDATE),
            "market_mode_edge_count": sum(1 for score in self.scores if score.activation_status == RELATIONSHIP_SELECTION_STATUS_MARKET_MODE_ONLY),
            "stable_edges_per_asset": {str(asset): int(count) for asset, count in sorted(self.stable_edges_per_asset.items())},
            "asset_statuses": {str(asset): str(status) for asset, status in sorted(self.asset_statuses.items())},
            "isolated_assets": list(self.isolated_assets),
            "isolated_asset_count": len(self.isolated_assets),
            "noisy_assets": list(self.noisy_assets),
            "noisy_asset_count": len(self.noisy_assets),
            "diagnostics": to_jsonable(dict(self.diagnostics)),
            "config": self.config.as_dict(),
            "prototype_non_production": True,
            "production_enabled": False,
            "final_peer_membership_claimed": False,
            "peer_group_created": False,
        }


def compute_relationship_stability(
    edges: Sequence[RelationshipEdge],
    *,
    all_assets: Sequence[str] = (),
    config: RelationshipStabilityConfig | None = None,
) -> RelationshipStabilityResult:
    config = config or RelationshipStabilityConfig()
    edge_rows = tuple(edge for edge in edges if isinstance(edge, RelationshipEdge))
    if not edge_rows:
        assets = tuple(dict.fromkeys(str(asset) for asset in all_assets if str(asset).strip()))
        return RelationshipStabilityResult(
            scores=(),
            asset_statuses={asset: RELATIONSHIP_SELECTION_STATUS_ISOLATED_ASSET for asset in assets},
            isolated_assets=assets,
            diagnostics={"reason_codes": ["no_candidate_edges"]},
            config=config,
        )

    rank_lookup = _rank_lookup(edge_rows)
    windows_by_method = _windows_by_method(edge_rows)
    grouped: dict[tuple[int, str, str, str], list[RelationshipEdge]] = {}
    for edge in edge_rows:
        grouped.setdefault(_stability_key(edge), []).append(edge)

    scores: list[RelationshipStabilityScore] = []
    for key, values in sorted(grouped.items()):
        interval, relationship_type, asset, related = key
        windows = windows_by_method.get((interval, relationship_type), set())
        observed_windows = {int(edge.window) for edge in values}
        survival_count = len(observed_windows)
        survival_share = survival_count / max(1, len(windows))
        strengths = [float(edge.abs_value) for edge in values]
        signs = [edge.direction for edge in values]
        sign_stability = _mode_share(signs)
        ranks = [rank_lookup[_candidate_key(edge)] for edge in values if _candidate_key(edge) in rank_lookup]
        rank_stability = _rank_stability(ranks)
        status = _activation_status(
            values[0],
            survival_count=survival_count,
            survival_share=survival_share,
            sign_stability=sign_stability,
            config=config,
        )
        scores.append(
            RelationshipStabilityScore(
                asset=asset,
                related_asset_or_benchmark=related,
                method_id=relationship_type,
                interval=interval,
                window=max(observed_windows),
                survival_count=survival_count,
                survival_share=min(1.0, max(0.0, survival_share)),
                mean_strength=_mean(strengths),
                strength_std=_std(strengths),
                sign_stability=sign_stability,
                rank_stability=rank_stability,
                activation_status=status,
            )
        )

    assets = tuple(dict.fromkeys([*(str(asset) for asset in all_assets if str(asset).strip()), *(edge.asset for edge in edge_rows)]))
    stable_counts = {asset: 0 for asset in assets}
    market_counts = {asset: 0 for asset in assets}
    edge_counts = {asset: 0 for asset in assets}
    for score in scores:
        edge_counts[score.asset] = edge_counts.get(score.asset, 0) + 1
        if score.activation_status == RELATIONSHIP_SELECTION_STATUS_STABLE_CANDIDATE:
            stable_counts[score.asset] = stable_counts.get(score.asset, 0) + 1
        if score.activation_status == RELATIONSHIP_SELECTION_STATUS_MARKET_MODE_ONLY:
            market_counts[score.asset] = market_counts.get(score.asset, 0) + 1

    asset_statuses: dict[str, str] = {}
    isolated_assets: list[str] = []
    noisy_assets: list[str] = []
    for asset in sorted(assets):
        if edge_counts.get(asset, 0) <= 0:
            asset_statuses[asset] = RELATIONSHIP_SELECTION_STATUS_ISOLATED_ASSET
            isolated_assets.append(asset)
        elif stable_counts.get(asset, 0) > 0:
            asset_statuses[asset] = RELATIONSHIP_SELECTION_STATUS_STABLE_CANDIDATE
        elif market_counts.get(asset, 0) > 0:
            asset_statuses[asset] = RELATIONSHIP_SELECTION_STATUS_MARKET_MODE_ONLY
            noisy_assets.append(asset)
        else:
            asset_statuses[asset] = RELATIONSHIP_SELECTION_STATUS_NEEDS_RESEARCH
            noisy_assets.append(asset)

    return RelationshipStabilityResult(
        scores=tuple(scores),
        asset_statuses=asset_statuses,
        stable_edges_per_asset={asset: count for asset, count in sorted(stable_counts.items()) if count > 0},
        isolated_assets=tuple(isolated_assets),
        noisy_assets=tuple(noisy_assets),
        diagnostics={
            "candidate_edge_count": len(edge_rows),
            "edge_survival_groups": len(scores),
            "relationship_type_count": len({score.method_id for score in scores}),
            "rank_stability_definition": "1 - normalized rank standard deviation across observed windows",
        },
        config=config,
    )


def _activation_status(
    edge: RelationshipEdge,
    *,
    survival_count: int,
    survival_share: float,
    sign_stability: float,
    config: RelationshipStabilityConfig,
) -> str:
    if _is_market_mode_edge(edge):
        return RELATIONSHIP_SELECTION_STATUS_MARKET_MODE_ONLY
    if (
        survival_count >= int(config.min_survival_count)
        and survival_share >= float(config.min_survival_share)
        and sign_stability >= float(config.min_sign_stability)
    ):
        return RELATIONSHIP_SELECTION_STATUS_STABLE_CANDIDATE
    return RELATIONSHIP_SELECTION_STATUS_UNSTABLE_CANDIDATE


def _is_market_mode_edge(edge: RelationshipEdge) -> bool:
    relationship_type = edge.relationship_type.lower()
    return edge.related_asset_or_benchmark == "core_basket" or relationship_type in {
        "beta_to_core_basket",
        "residual_return_vs_core_basket",
    }


def _stability_key(edge: RelationshipEdge) -> tuple[int, str, str, str]:
    return (int(edge.interval), edge.relationship_type, edge.asset, edge.related_asset_or_benchmark)


def _candidate_key(edge: RelationshipEdge) -> tuple[int, int, str, str, str]:
    return (int(edge.interval), int(edge.window), edge.method_id, edge.asset, edge.related_asset_or_benchmark)


def _rank_lookup(edges: Sequence[RelationshipEdge]) -> dict[tuple[int, int, str, str, str], int]:
    grouped: dict[tuple[int, int, str, str], list[RelationshipEdge]] = {}
    for edge in edges:
        grouped.setdefault((int(edge.interval), int(edge.window), edge.method_id, edge.asset), []).append(edge)
    ranks: dict[tuple[int, int, str, str, str], int] = {}
    for values in grouped.values():
        ordered = sorted(values, key=lambda edge: (-float(edge.abs_value), edge.related_asset_or_benchmark))
        for rank, edge in enumerate(ordered, start=1):
            ranks[_candidate_key(edge)] = rank
    return ranks


def _windows_by_method(edges: Sequence[RelationshipEdge]) -> dict[tuple[int, str], set[int]]:
    out: dict[tuple[int, str], set[int]] = {}
    for edge in edges:
        out.setdefault((int(edge.interval), edge.relationship_type), set()).add(int(edge.window))
    return out


def _mode_share(values: Sequence[str]) -> float:
    if not values:
        return 0.0
    counts: dict[str, int] = {}
    for value in values:
        counts[str(value)] = counts.get(str(value), 0) + 1
    return max(counts.values()) / len(values)


def _rank_stability(ranks: Sequence[int]) -> float:
    if not ranks:
        return 0.0
    if len(ranks) == 1:
        return 1.0
    max_rank = max(max(ranks), 1)
    normalized_std = _std([float(rank) for rank in ranks]) / max_rank
    return min(1.0, max(0.0, 1.0 - normalized_std))


def _mean(values: Sequence[float]) -> float:
    return sum(float(value) for value in values) / max(1, len(values))


def _std(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = _mean(values)
    variance = sum((float(value) - mean) ** 2 for value in values) / len(values)
    return math.sqrt(max(0.0, variance))


def _share(value: object, *, field_name: str) -> float:
    try:
        out = float(value)
    except Exception as exc:
        raise ValueError(f"Relationship Discovery {field_name} must be numeric") from exc
    if out < 0.0 or out > 1.0:
        raise ValueError(f"Relationship Discovery {field_name} must be between 0 and 1")
    return out


__all__ = [
    "RELATIONSHIP_SELECTION_STATUS_INSUFFICIENT_COVERAGE",
    "RELATIONSHIP_SELECTION_STATUS_ISOLATED_ASSET",
    "RELATIONSHIP_SELECTION_STATUS_MARKET_MODE_ONLY",
    "RELATIONSHIP_SELECTION_STATUS_NEEDS_RESEARCH",
    "RELATIONSHIP_SELECTION_STATUS_STABLE_CANDIDATE",
    "RELATIONSHIP_SELECTION_STATUS_UNSTABLE_CANDIDATE",
    "RELATIONSHIP_SELECTION_STATUSES",
    "RelationshipStabilityConfig",
    "RelationshipStabilityResult",
    "compute_relationship_stability",
]
