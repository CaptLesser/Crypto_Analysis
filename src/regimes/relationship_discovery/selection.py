from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.regimes.core.serialization import to_jsonable
from src.regimes.relationship_discovery.artifacts import (
    RelationshipArtifactWriteResult,
    write_relationship_json_artifact,
    write_relationship_jsonl_rows,
)
from src.regimes.relationship_discovery.contracts import RelationshipEdge
from src.regimes.relationship_discovery.methods import (
    RELATIONSHIP_METHOD_STATUS_COMPUTED,
    RelationshipMethodComparisonResult,
)
from src.regimes.relationship_discovery.stability import (
    RELATIONSHIP_SELECTION_STATUS_INSUFFICIENT_COVERAGE,
    RELATIONSHIP_SELECTION_STATUS_MARKET_MODE_ONLY,
    RELATIONSHIP_SELECTION_STATUS_NEEDS_RESEARCH,
    RELATIONSHIP_SELECTION_STATUS_STABLE_CANDIDATE,
    RELATIONSHIP_SELECTION_STATUS_UNSTABLE_CANDIDATE,
    RelationshipStabilityConfig,
    RelationshipStabilityResult,
    compute_relationship_stability,
)


RELATIONSHIP_SELECTION_RESULT_STATUS_COMPUTED = "computed"
RELATIONSHIP_SELECTION_RESULT_STATUS_NO_CANDIDATES = "no_candidates"


@dataclass(frozen=True)
class RelationshipEdgeSelectionConfig:
    min_coverage: float = 0.75
    min_sample_count: int = 20
    min_abs_strength: float = 0.5
    top_k_per_asset: int = 3
    require_sign_consistency: bool = False
    require_window_survival: bool = False
    min_survival_count: int = 2
    min_survival_share: float = 0.5
    relationship_types: Sequence[str] | None = None
    include_market_mode_edges: bool = True

    def __post_init__(self) -> None:
        relationship_types = None
        if self.relationship_types is not None:
            relationship_types = tuple(dict.fromkeys(str(value).strip().lower() for value in self.relationship_types if str(value).strip()))
        object.__setattr__(self, "min_coverage", _share(self.min_coverage, field_name="min_coverage"))
        object.__setattr__(self, "min_sample_count", max(1, int(self.min_sample_count)))
        object.__setattr__(self, "min_abs_strength", max(0.0, float(self.min_abs_strength)))
        object.__setattr__(self, "top_k_per_asset", max(1, int(self.top_k_per_asset)))
        object.__setattr__(self, "require_sign_consistency", bool(self.require_sign_consistency))
        object.__setattr__(self, "require_window_survival", bool(self.require_window_survival))
        object.__setattr__(self, "min_survival_count", max(1, int(self.min_survival_count)))
        object.__setattr__(self, "min_survival_share", _share(self.min_survival_share, field_name="min_survival_share"))
        object.__setattr__(self, "relationship_types", relationship_types)
        object.__setattr__(self, "include_market_mode_edges", bool(self.include_market_mode_edges))

    def as_dict(self) -> dict[str, Any]:
        return {
            "min_coverage": float(self.min_coverage),
            "min_sample_count": int(self.min_sample_count),
            "min_abs_strength": float(self.min_abs_strength),
            "top_k_per_asset": int(self.top_k_per_asset),
            "require_sign_consistency": bool(self.require_sign_consistency),
            "require_window_survival": bool(self.require_window_survival),
            "min_survival_count": int(self.min_survival_count),
            "min_survival_share": float(self.min_survival_share),
            "relationship_types": list(self.relationship_types) if self.relationship_types is not None else None,
            "include_market_mode_edges": bool(self.include_market_mode_edges),
        }

    def stability_config(self) -> RelationshipStabilityConfig:
        return RelationshipStabilityConfig(
            min_survival_count=self.min_survival_count,
            min_survival_share=self.min_survival_share,
            min_sign_stability=1.0,
        )


@dataclass(frozen=True)
class SelectedRelationshipEdge:
    edge: RelationshipEdge
    selection_status: str
    selection_rank: int
    reason_codes: Sequence[str] = ()
    survival_count: int = 1
    survival_share: float = 1.0
    sign_stability: float = 1.0
    rank_stability: float = 1.0
    prototype_non_production: bool = True
    final_peer_membership_claimed: bool = False
    peer_group_created: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "selection_status", str(self.selection_status).strip().lower())
        object.__setattr__(self, "selection_rank", max(0, int(self.selection_rank)))
        object.__setattr__(self, "reason_codes", tuple(dict.fromkeys(str(reason) for reason in self.reason_codes)))
        object.__setattr__(self, "survival_count", max(0, int(self.survival_count)))
        object.__setattr__(self, "survival_share", _share(self.survival_share, field_name="survival_share"))
        object.__setattr__(self, "sign_stability", _share(self.sign_stability, field_name="sign_stability"))
        object.__setattr__(self, "rank_stability", _share(self.rank_stability, field_name="rank_stability"))
        object.__setattr__(self, "prototype_non_production", True)
        object.__setattr__(self, "final_peer_membership_claimed", False)
        object.__setattr__(self, "peer_group_created", False)

    @property
    def selected(self) -> bool:
        if self.selection_status == RELATIONSHIP_SELECTION_STATUS_STABLE_CANDIDATE:
            return True
        if self.selection_status == RELATIONSHIP_SELECTION_STATUS_MARKET_MODE_ONLY:
            return True
        return False

    def as_dict(self) -> dict[str, Any]:
        payload = self.edge.as_dict()
        payload["source_artifact_kind"] = payload.pop("artifact_kind", "relationship_edge")
        payload.update(
            {
                "artifact_kind": "selected_relationship_edge",
                "selection_status": self.selection_status,
                "selection_rank": int(self.selection_rank),
                "selection_reason_codes": list(self.reason_codes),
                "survival_count": int(self.survival_count),
                "survival_share": float(self.survival_share),
                "sign_stability": float(self.sign_stability),
                "rank_stability": float(self.rank_stability),
                "prototype_non_production": True,
                "production_enabled": False,
                "final_peer_membership_claimed": False,
                "peer_group_created": False,
            }
        )
        return payload


@dataclass(frozen=True)
class RelationshipEdgeSelectionResult:
    status: str
    candidate_edges: Sequence[SelectedRelationshipEdge]
    selected_edges: Sequence[SelectedRelationshipEdge]
    stability_result: RelationshipStabilityResult
    diagnostics: Mapping[str, Any] = field(default_factory=dict)
    config: RelationshipEdgeSelectionConfig = field(default_factory=RelationshipEdgeSelectionConfig)

    def __post_init__(self) -> None:
        object.__setattr__(self, "candidate_edges", tuple(self.candidate_edges))
        object.__setattr__(self, "selected_edges", tuple(self.selected_edges))
        object.__setattr__(self, "status", str(self.status).strip().lower())

    def as_dict(self) -> dict[str, Any]:
        status_counts: dict[str, int] = {}
        for edge in self.candidate_edges:
            status_counts[edge.selection_status] = status_counts.get(edge.selection_status, 0) + 1
        return {
            "artifact_kind": "relationship_edge_selection_summary",
            "status": self.status,
            "candidate_edge_count": len(self.candidate_edges),
            "selected_edge_count": len(self.selected_edges),
            "status_counts": dict(sorted(status_counts.items())),
            "diagnostics": to_jsonable(dict(self.diagnostics)),
            "stability_summary": self.stability_result.as_dict(),
            "config": self.config.as_dict(),
            "prototype_non_production": True,
            "production_enabled": False,
            "final_peer_membership_claimed": False,
            "peer_group_created": False,
        }


@dataclass(frozen=True)
class RelationshipSelectionArtifactWriteResult:
    status: str
    output_root: Path
    writes: Sequence[RelationshipArtifactWriteResult]

    def as_dict(self) -> dict[str, Any]:
        written_paths = []
        for write in self.writes:
            written_paths.extend(str(path) for path in write.written_paths)
        return {
            "artifact_kind": "relationship_selection_artifact_write_result",
            "status": self.status,
            "output_root": str(self.output_root),
            "written_paths": written_paths,
            "write_count": len(self.writes),
            "production_enabled": False,
        }


def select_relationship_edges(
    comparisons: Sequence[RelationshipMethodComparisonResult] | Mapping[Any, RelationshipMethodComparisonResult] | Any,
    *,
    config: RelationshipEdgeSelectionConfig | None = None,
    all_assets: Sequence[str] = (),
) -> RelationshipEdgeSelectionResult:
    config = config or RelationshipEdgeSelectionConfig()
    candidate_edges = tuple(_candidate_edges(comparisons, config=config))
    stability = compute_relationship_stability(candidate_edges, all_assets=all_assets, config=config.stability_config())
    if not candidate_edges:
        return RelationshipEdgeSelectionResult(
            status=RELATIONSHIP_SELECTION_RESULT_STATUS_NO_CANDIDATES,
            candidate_edges=(),
            selected_edges=(),
            stability_result=stability,
            diagnostics={"reason_codes": ["no_candidate_edges"]},
            config=config,
        )

    rank_lookup = _rank_lookup(candidate_edges)
    stability_lookup = {(score.interval, score.method_id, score.asset, score.related_asset_or_benchmark): score for score in stability.scores}
    assessed: list[SelectedRelationshipEdge] = []
    for edge in _stable_order(candidate_edges):
        score = stability_lookup.get((int(edge.interval), edge.relationship_type, edge.asset, edge.related_asset_or_benchmark))
        rank = rank_lookup[_candidate_key(edge)]
        status, reasons = _selection_status(edge, rank=rank, stability_score=score, config=config)
        assessed.append(
            SelectedRelationshipEdge(
                edge=edge,
                selection_status=status,
                selection_rank=rank,
                reason_codes=reasons,
                survival_count=score.survival_count if score is not None else 1,
                survival_share=score.survival_share if score is not None else 1.0,
                sign_stability=score.sign_stability if score is not None else 1.0,
                rank_stability=score.rank_stability if score is not None else 1.0,
            )
        )

    selected = tuple(edge for edge in assessed if edge.selected)
    return RelationshipEdgeSelectionResult(
        status=RELATIONSHIP_SELECTION_RESULT_STATUS_COMPUTED,
        candidate_edges=tuple(assessed),
        selected_edges=selected,
        stability_result=stability,
        diagnostics={
            "candidate_edge_count": len(candidate_edges),
            "selected_edge_count": len(selected),
            "stable_candidate_count": sum(1 for edge in selected if edge.selection_status == RELATIONSHIP_SELECTION_STATUS_STABLE_CANDIDATE),
            "market_mode_only_count": sum(1 for edge in selected if edge.selection_status == RELATIONSHIP_SELECTION_STATUS_MARKET_MODE_ONLY),
        },
        config=config,
    )


def write_relationship_selection_artifacts(
    result: RelationshipEdgeSelectionResult,
    *,
    output_root: str | Path,
    artifact_dir: str | Path = "selected_edges",
    production_enabled: bool = False,
) -> RelationshipSelectionArtifactWriteResult:
    base = Path(artifact_dir)
    if base.is_absolute() or any(part in {"", ".."} for part in base.parts):
        raise ValueError("Relationship Discovery selection artifact_dir must stay within output root")
    writes = (
        write_relationship_json_artifact(
            result,
            output_root=output_root,
            relative_path=base / "selection_summary.json",
            production_enabled=production_enabled,
        ),
        write_relationship_jsonl_rows(
            result.candidate_edges,
            output_root=output_root,
            relative_path=base / "candidate_edges.jsonl",
            production_enabled=production_enabled,
        ),
        write_relationship_jsonl_rows(
            result.selected_edges,
            output_root=output_root,
            relative_path=base / "selected_edges.jsonl",
            production_enabled=production_enabled,
        ),
        write_relationship_json_artifact(
            result.stability_result,
            output_root=output_root,
            relative_path=base / "stability_summary.json",
            production_enabled=production_enabled,
        ),
        write_relationship_jsonl_rows(
            result.stability_result.scores,
            output_root=output_root,
            relative_path=base / "stability_scores.jsonl",
            production_enabled=production_enabled,
        ),
    )
    return RelationshipSelectionArtifactWriteResult(
        status="written",
        output_root=Path(output_root),
        writes=writes,
    )


def _selection_status(
    edge: RelationshipEdge,
    *,
    rank: int,
    stability_score: Any,
    config: RelationshipEdgeSelectionConfig,
) -> tuple[str, tuple[str, ...]]:
    reasons: list[str] = []
    if edge.sample_count < int(config.min_sample_count):
        reasons.append("below_min_sample_count")
    if float(edge.coverage) < float(config.min_coverage):
        reasons.append("below_min_coverage")
    if reasons:
        return RELATIONSHIP_SELECTION_STATUS_INSUFFICIENT_COVERAGE, tuple(reasons)

    market_mode = _is_market_mode_edge(edge)
    if float(edge.abs_value) < float(config.min_abs_strength):
        reasons.append("below_min_abs_strength")
        if market_mode and config.include_market_mode_edges:
            reasons.append("market_mode_fact_not_peer_candidate")
            return RELATIONSHIP_SELECTION_STATUS_MARKET_MODE_ONLY, tuple(reasons)
        return RELATIONSHIP_SELECTION_STATUS_NEEDS_RESEARCH, tuple(reasons)
    if rank > int(config.top_k_per_asset):
        return RELATIONSHIP_SELECTION_STATUS_NEEDS_RESEARCH, ("outside_top_k_per_asset",)

    if market_mode:
        if not config.include_market_mode_edges:
            return RELATIONSHIP_SELECTION_STATUS_NEEDS_RESEARCH, ("market_mode_excluded_by_config",)
        return RELATIONSHIP_SELECTION_STATUS_MARKET_MODE_ONLY, ("market_mode_fact_not_peer_candidate",)

    if stability_score is not None:
        if config.require_sign_consistency and float(stability_score.sign_stability) < 1.0:
            reasons.append("sign_consistency_failed")
        if config.require_window_survival:
            if int(stability_score.survival_count) < int(config.min_survival_count):
                reasons.append("window_survival_count_failed")
            if float(stability_score.survival_share) < float(config.min_survival_share):
                reasons.append("window_survival_share_failed")
    if reasons:
        return RELATIONSHIP_SELECTION_STATUS_UNSTABLE_CANDIDATE, tuple(reasons)
    return RELATIONSHIP_SELECTION_STATUS_STABLE_CANDIDATE, ("selected_by_configured_criteria",)


def _candidate_edges(
    comparisons: Sequence[RelationshipMethodComparisonResult] | Mapping[Any, RelationshipMethodComparisonResult] | Any,
    *,
    config: RelationshipEdgeSelectionConfig,
) -> tuple[RelationshipEdge, ...]:
    values = _comparison_values(comparisons)
    edges: list[RelationshipEdge] = []
    for comparison in values:
        if not isinstance(comparison, RelationshipMethodComparisonResult):
            continue
        for method in comparison.methods:
            if method.status != RELATIONSHIP_METHOD_STATUS_COMPUTED:
                continue
            for edge in method.edges:
                if config.relationship_types is not None and edge.relationship_type.lower() not in set(config.relationship_types):
                    continue
                edges.append(edge)
    return tuple(_stable_order(edges))


def _comparison_values(
    comparisons: Sequence[RelationshipMethodComparisonResult] | Mapping[Any, RelationshipMethodComparisonResult] | Any,
) -> tuple[RelationshipMethodComparisonResult, ...]:
    if isinstance(comparisons, RelationshipMethodComparisonResult):
        return (comparisons,)
    if isinstance(comparisons, Mapping):
        return tuple(value for value in comparisons.values() if isinstance(value, RelationshipMethodComparisonResult))
    if hasattr(comparisons, "comparisons"):
        nested = getattr(comparisons, "comparisons")
        return _comparison_values(nested)
    if isinstance(comparisons, Sequence) and not isinstance(comparisons, (str, bytes)):
        return tuple(value for value in comparisons if isinstance(value, RelationshipMethodComparisonResult))
    return ()


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


def _candidate_key(edge: RelationshipEdge) -> tuple[int, int, str, str, str]:
    return (int(edge.interval), int(edge.window), edge.method_id, edge.asset, edge.related_asset_or_benchmark)


def _stable_order(edges: Sequence[RelationshipEdge]) -> tuple[RelationshipEdge, ...]:
    return tuple(
        sorted(
            edges,
            key=lambda edge: (
                int(edge.interval),
                int(edge.window),
                edge.method_id,
                edge.asset,
                -float(edge.abs_value),
                edge.related_asset_or_benchmark,
            ),
        )
    )


def _is_market_mode_edge(edge: RelationshipEdge) -> bool:
    return edge.related_asset_or_benchmark == "core_basket" or edge.relationship_type in {
        "beta_to_core_basket",
        "residual_return_vs_core_basket",
    }


def _share(value: object, *, field_name: str) -> float:
    try:
        out = float(value)
    except Exception as exc:
        raise ValueError(f"Relationship Discovery {field_name} must be numeric") from exc
    if out < 0.0 or out > 1.0:
        raise ValueError(f"Relationship Discovery {field_name} must be between 0 and 1")
    return out


__all__ = [
    "RELATIONSHIP_SELECTION_RESULT_STATUS_COMPUTED",
    "RELATIONSHIP_SELECTION_RESULT_STATUS_NO_CANDIDATES",
    "RelationshipEdgeSelectionConfig",
    "RelationshipEdgeSelectionResult",
    "RelationshipSelectionArtifactWriteResult",
    "SelectedRelationshipEdge",
    "select_relationship_edges",
    "write_relationship_selection_artifacts",
]
