from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from src.regimes.core.serialization import to_jsonable
from src.regimes.relationship_discovery.methods import (
    RELATIONSHIP_METHOD_STATUS_COMPUTED,
    RELATIONSHIP_METHOD_STATUS_INSUFFICIENT,
    RELATIONSHIP_METHOD_STATUS_SKIPPED,
    RelationshipMethodComparisonConfig,
    RelationshipMethodComparisonResult,
    compare_relationship_methods,
)
from src.regimes.relationship_discovery.scope import (
    RELATIONSHIP_SCOPE_STATUS_REAL_DATA_LOADED,
    RelationshipDiscoveryScopeRequest,
    RelationshipDiscoveryScopeResult,
    build_relationship_discovery_scope,
)


RELATIONSHIP_PROTOTYPE_STATUS_COMPUTED = "computed"
RELATIONSHIP_PROTOTYPE_STATUS_SCOPE_UNAVAILABLE = "scope_unavailable"
RELATIONSHIP_PROTOTYPE_STATUS_INSUFFICIENT = "insufficient"
RELATIONSHIP_PROTOTYPE_STATUS_SKIPPED = "skipped"

RELATIONSHIP_PROTOTYPE_STATUSES: tuple[str, ...] = (
    RELATIONSHIP_PROTOTYPE_STATUS_COMPUTED,
    RELATIONSHIP_PROTOTYPE_STATUS_SCOPE_UNAVAILABLE,
    RELATIONSHIP_PROTOTYPE_STATUS_INSUFFICIENT,
    RELATIONSHIP_PROTOTYPE_STATUS_SKIPPED,
)


@dataclass(frozen=True)
class RelationshipDiscoveryPrototypeRequest:
    scope_request: RelationshipDiscoveryScopeRequest
    method_config: RelationshipMethodComparisonConfig = field(default_factory=RelationshipMethodComparisonConfig)
    feature_panel: pd.DataFrame | None = field(default=None, compare=False, repr=False)
    intervals: Sequence[int] | None = None

    def __post_init__(self) -> None:
        intervals = None
        if self.intervals is not None:
            intervals = tuple(dict.fromkeys(int(interval) for interval in self.intervals))
            if any(interval < 60 for interval in intervals):
                raise ValueError("Relationship Discovery prototype does not support sub-hour intervals")
        object.__setattr__(self, "intervals", intervals)

    def as_dict(self) -> dict[str, Any]:
        return {
            "scope_request": {
                "manifest_path": str(self.scope_request.manifest_path) if self.scope_request.manifest_path is not None else None,
                "market_state_universe_root": str(self.scope_request.market_state_universe_root)
                if self.scope_request.market_state_universe_root is not None
                else None,
                "ohlcvt_root": str(self.scope_request.ohlcvt_root) if self.scope_request.ohlcvt_root is not None else None,
                "scalar_feature_root": str(self.scope_request.scalar_feature_root) if self.scope_request.scalar_feature_root is not None else None,
                "primary_interval": int(self.scope_request.primary_interval),
                "confirmation_interval": int(self.scope_request.confirmation_interval),
                "include_60m_evidence_probe": bool(self.scope_request.include_60m_evidence_probe),
                "broad_sample_size": int(self.scope_request.broad_sample_size),
                "min_assets": int(self.scope_request.min_assets),
                "min_overlap": int(self.scope_request.min_overlap),
                "fixture_manifest_supplied": self.scope_request.fixture_manifest is not None,
            },
            "method_config": self.method_config.as_dict(),
            "feature_panel_supplied": self.feature_panel is not None,
            "intervals": list(self.intervals) if self.intervals is not None else None,
        }


@dataclass
class RelationshipDiscoveryPrototypeResult:
    status: str
    scope_result: RelationshipDiscoveryScopeResult
    comparisons: Mapping[int, RelationshipMethodComparisonResult] = field(default_factory=dict)
    reason_codes: Sequence[str] = ()
    message: str | None = None
    runtime_summary: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        status = str(self.status).strip().lower()
        if status not in RELATIONSHIP_PROTOTYPE_STATUSES:
            raise ValueError(f"Unsupported Relationship Discovery prototype status {status!r}")
        self.status = status
        self.reason_codes = tuple(dict.fromkeys(str(reason) for reason in self.reason_codes))

    @property
    def edge_count(self) -> int:
        return sum(result.edge_count for result in self.comparisons.values())

    @property
    def profile_count(self) -> int:
        return sum(result.profile_count for result in self.comparisons.values())

    def as_dict(self) -> dict[str, Any]:
        return {
            "artifact_kind": "relationship_discovery_prototype_result",
            "status": self.status,
            "reason_codes": list(self.reason_codes),
            "message": self.message,
            "edge_count": int(self.edge_count),
            "profile_count": int(self.profile_count),
            "scope_result": self.scope_result.as_dict(),
            "comparisons": {str(interval): result.as_dict() for interval, result in sorted(self.comparisons.items())},
            "runtime_summary": to_jsonable(dict(self.runtime_summary)),
            "metadata": to_jsonable(dict(self.metadata)),
            "production_writes_enabled": False,
            "peer_cluster_production_enabled": False,
            "broad_production_all_to_all_enabled": False,
        }


def run_relationship_discovery_prototype(request: RelationshipDiscoveryPrototypeRequest) -> RelationshipDiscoveryPrototypeResult:
    started = time.perf_counter()
    scope = build_relationship_discovery_scope(request.scope_request)
    if scope.status != RELATIONSHIP_SCOPE_STATUS_REAL_DATA_LOADED or scope.universe is None:
        return RelationshipDiscoveryPrototypeResult(
            status=RELATIONSHIP_PROTOTYPE_STATUS_SCOPE_UNAVAILABLE,
            scope_result=scope,
            reason_codes=tuple(scope.reason_codes) or ("scope_unavailable",),
            message=scope.message,
            runtime_summary=_runtime(started),
            metadata={"request": request.as_dict(), "production_writes_enabled": False},
        )

    intervals = tuple(request.intervals) if request.intervals is not None else tuple(sorted(scope.panel_results))
    comparisons: dict[int, RelationshipMethodComparisonResult] = {}
    for interval in intervals:
        panel = scope.panel_results.get(int(interval))
        if panel is None or not panel.usable:
            continue
        comparisons[int(interval)] = compare_relationship_methods(
            panel,
            scope.universe,
            config=request.method_config,
            feature_panel=request.feature_panel,
        )
    if not comparisons:
        status = RELATIONSHIP_PROTOTYPE_STATUS_INSUFFICIENT
        reason_codes = ("no_usable_interval_comparisons",)
        message = "no usable interval panels were available for method comparison"
    elif any(result.status == RELATIONSHIP_METHOD_STATUS_COMPUTED for result in comparisons.values()):
        status = RELATIONSHIP_PROTOTYPE_STATUS_COMPUTED
        reason_codes = ("relationship_prototype_computed",)
        message = None
    elif any(result.status == RELATIONSHIP_METHOD_STATUS_SKIPPED for result in comparisons.values()):
        status = RELATIONSHIP_PROTOTYPE_STATUS_SKIPPED
        reason_codes = ("relationship_prototype_skipped",)
        message = "all interval comparisons were skipped"
    else:
        status = RELATIONSHIP_PROTOTYPE_STATUS_INSUFFICIENT
        reason_codes = ("relationship_prototype_insufficient",)
        message = "no relationship method comparison produced computed outputs"
    return RelationshipDiscoveryPrototypeResult(
        status=status,
        scope_result=scope,
        comparisons=comparisons,
        reason_codes=reason_codes,
        message=message,
        runtime_summary=_runtime(started),
        metadata={"request": request.as_dict(), "production_writes_enabled": False},
    )


def _runtime(started: float) -> dict[str, Any]:
    return {"runtime_seconds": float(max(0.0, time.perf_counter() - started))}


__all__ = [
    "RELATIONSHIP_PROTOTYPE_STATUS_COMPUTED",
    "RELATIONSHIP_PROTOTYPE_STATUS_INSUFFICIENT",
    "RELATIONSHIP_PROTOTYPE_STATUS_SCOPE_UNAVAILABLE",
    "RELATIONSHIP_PROTOTYPE_STATUS_SKIPPED",
    "RELATIONSHIP_PROTOTYPE_STATUSES",
    "RelationshipDiscoveryPrototypeRequest",
    "RelationshipDiscoveryPrototypeResult",
    "run_relationship_discovery_prototype",
]
