"""Legacy gated pairwise scaffold.

Relationship Discovery v1 is canonical in ``src.regimes.relationship_discovery``.
This module remains a compatibility/scaffold surface for earlier Regime Feature
pairwise contracts. It must stay disabled by default: no broad all-to-all
production materialization, no dynamic peer discovery, and no production writes.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from src.regimes.core.serialization import dumps_json, loads_json, require_json_object, to_jsonable
from src.regimes.regime_features.contracts import (
    PAIRWISE_RELATIONSHIP_FEATURES,
    REGIME_FEATURES_SCHEMA_VERSION,
    RegimeFeatureArtifactBoundary,
    pairwise_relationship_output_schema,
)


RELATIONSHIP_ROLLING_CORR = "rolling_corr"
RELATIONSHIP_ROLLING_COV = "rolling_cov"
RELATIONSHIP_ROLLING_BETA = "rolling_beta"
RELATIONSHIP_RESIDUAL_CORR = "residual_corr"
RELATIONSHIP_LEAD_LAG_CORR = "lead_lag_corr"
RELATIONSHIP_RETURN_DISTANCE = "return_distance"
RELATIONSHIP_VOLATILITY_DISTANCE = "volatility_distance"
RELATIONSHIP_CORRELATION_DISTANCE = "correlation_distance"

PAIRWISE_RELATIONSHIP_TYPES: tuple[str, ...] = (
    RELATIONSHIP_ROLLING_CORR,
    RELATIONSHIP_ROLLING_COV,
    RELATIONSHIP_ROLLING_BETA,
    RELATIONSHIP_RESIDUAL_CORR,
    RELATIONSHIP_LEAD_LAG_CORR,
    RELATIONSHIP_RETURN_DISTANCE,
    RELATIONSHIP_VOLATILITY_DISTANCE,
    RELATIONSHIP_CORRELATION_DISTANCE,
)

PAIRWISE_SYMMETRIC_RELATIONSHIP_TYPES: tuple[str, ...] = (
    RELATIONSHIP_ROLLING_CORR,
    RELATIONSHIP_ROLLING_COV,
    RELATIONSHIP_RESIDUAL_CORR,
    RELATIONSHIP_RETURN_DISTANCE,
    RELATIONSHIP_VOLATILITY_DISTANCE,
    RELATIONSHIP_CORRELATION_DISTANCE,
)

PAIRWISE_DIRECTED_RELATIONSHIP_TYPES: tuple[str, ...] = (
    RELATIONSHIP_ROLLING_BETA,
    RELATIONSHIP_LEAD_LAG_CORR,
)

SCOPE_CORE_BASKET_ONLY = "core_basket_only"
SCOPE_BENCHMARK_ANCHOR_ONLY = "benchmark_anchor_only"
SCOPE_CORE_PLUS_ANCHORS = "core_plus_anchors"
SCOPE_TOP_K_PLACEHOLDER = "top_k_placeholder"
SCOPE_BROAD_ALL_TO_ALL_DISABLED = "broad_all_to_all_disabled"

PAIRWISE_RELATIONSHIP_SCOPES: tuple[str, ...] = (
    SCOPE_CORE_BASKET_ONLY,
    SCOPE_BENCHMARK_ANCHOR_ONLY,
    SCOPE_CORE_PLUS_ANCHORS,
    SCOPE_TOP_K_PLACEHOLDER,
    SCOPE_BROAD_ALL_TO_ALL_DISABLED,
)

PAIRWISE_MICRO_GATED_INTERVALS: tuple[int, ...] = (1, 5, 15)

PAIRWISE_SCOPE_NOT_APPROVED = "pairwise_scope_not_approved"
INTERVAL_NOT_APPROVED_FOR_PAIRWISE = "interval_not_approved_for_pairwise"
PEER_DISCOVERY_NOT_IMPLEMENTED = "peer_discovery_not_implemented"
PRODUCTION_PAIRWISE_WRITES_DISABLED = "production_pairwise_writes_disabled"

PLAN_STATUS_SCAFFOLD_READY = "scaffold_ready"
PLAN_STATUS_BLOCKED = "blocked"
PLAN_STATUS_FIXTURE_ONLY = "fixture_only"

PAIRWISE_LEGACY_SURFACE_STATUS = "legacy_scaffold_only"
PAIRWISE_PROCESS1_CANONICAL_MODULE = "src.regimes.relationship_discovery"


@dataclass(frozen=True)
class PairwiseRelationshipSpec:
    relationship_type: str
    window: int
    interval: int
    band: str
    directed: bool | None = None
    lag: int = 0
    min_periods: int | None = None
    schema_version: int = REGIME_FEATURES_SCHEMA_VERSION
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        relationship_type = _relationship_type(self.relationship_type)
        interval = int(self.interval)
        window = int(self.window)
        if interval <= 0:
            raise ValueError("Regime Feature pairwise interval must be positive")
        if window <= 1:
            raise ValueError("Regime Feature pairwise window must be greater than 1")
        min_periods = int(self.min_periods if self.min_periods is not None else min(window, max(2, window // 2)))
        if min_periods <= 1 or min_periods > window:
            raise ValueError("Regime Feature pairwise min_periods must be within [2, window]")
        inferred_directed = relationship_type in PAIRWISE_DIRECTED_RELATIONSHIP_TYPES
        if self.directed is not None and bool(self.directed) != inferred_directed:
            raise ValueError(f"Regime Feature pairwise directed flag mismatches relationship_type {relationship_type!r}")
        object.__setattr__(self, "relationship_type", relationship_type)
        object.__setattr__(self, "window", window)
        object.__setattr__(self, "interval", interval)
        object.__setattr__(self, "band", _text(self.band, field_name="band").lower())
        object.__setattr__(self, "directed", inferred_directed)
        object.__setattr__(self, "lag", int(self.lag))
        object.__setattr__(self, "min_periods", min_periods)
        object.__setattr__(self, "schema_version", int(self.schema_version))
        object.__setattr__(self, "metadata", to_jsonable(dict(self.metadata)))

    @property
    def spec_id(self) -> str:
        payload = {
            "relationship_type": self.relationship_type,
            "window": int(self.window),
            "interval": int(self.interval),
            "band": self.band,
            "directed": bool(self.directed),
            "lag": int(self.lag),
            "min_periods": int(self.min_periods or 0),
        }
        digest = hashlib.sha256(dumps_json(payload).encode("utf-8")).hexdigest()[:12]
        return f"{self.relationship_type}:{self.interval}:{self.band}:w{self.window}:lag{self.lag}:{digest}"

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "artifact_kind": "pairwise_relationship_spec",
            "spec_id": self.spec_id,
            "relationship_type": self.relationship_type,
            "window": int(self.window),
            "interval": int(self.interval),
            "band": self.band,
            "directed": bool(self.directed),
            "symmetric_pair_canonicalization": self.relationship_type in PAIRWISE_SYMMETRIC_RELATIONSHIP_TYPES,
            "lag": int(self.lag),
            "min_periods": int(self.min_periods or 0),
            "metadata": to_jsonable(dict(self.metadata)),
        }

    def to_json(self, **kwargs: Any) -> str:
        return dumps_json(self.as_dict(), **kwargs)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PairwiseRelationshipSpec":
        obj = require_json_object(payload, context="PairwiseRelationshipSpec")
        return cls(
            schema_version=obj.get("schema_version", REGIME_FEATURES_SCHEMA_VERSION),
            relationship_type=obj["relationship_type"],
            window=obj["window"],
            interval=obj["interval"],
            band=obj["band"],
            directed=obj.get("directed"),
            lag=obj.get("lag", 0),
            min_periods=obj.get("min_periods"),
            metadata=obj.get("metadata", {}),
        )

    @classmethod
    def from_json(cls, text: str) -> "PairwiseRelationshipSpec":
        return cls.from_dict(require_json_object(loads_json(text), context="PairwiseRelationshipSpec JSON"))


@dataclass(frozen=True)
class PairwiseRelationshipScopePolicy:
    scope: str
    interval: int
    band: str
    policy_id: str = "pairwise_scope_policy_v1"
    core_basket_assets: Sequence[str] = ()
    benchmark_anchors: Sequence[str] = ()
    top_k: int | None = None
    broad_all_to_all_enabled: bool = False
    micro_interval_pairwise_enabled: bool = False
    dynamic_peer_discovery_enabled: bool = False
    peer_clusters_enabled: bool = False
    materialization_enabled: bool = False
    production_enabled: bool = False
    schema_version: int = REGIME_FEATURES_SCHEMA_VERSION
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        scope = _member(self.scope, PAIRWISE_RELATIONSHIP_SCOPES, field_name="scope")
        interval = int(self.interval)
        if interval <= 0:
            raise ValueError("Regime Feature pairwise scope policy interval must be positive")
        if self.broad_all_to_all_enabled is not False:
            raise ValueError(f"{PAIRWISE_SCOPE_NOT_APPROVED}: Regime Feature pairwise broad_all_to_all_enabled must remain disabled by default")
        _require_disabled(self.dynamic_peer_discovery_enabled, field_name="dynamic_peer_discovery_enabled", status=PEER_DISCOVERY_NOT_IMPLEMENTED)
        _require_disabled(self.peer_clusters_enabled, field_name="peer_clusters_enabled", status=PEER_DISCOVERY_NOT_IMPLEMENTED)
        _require_disabled(self.production_enabled, field_name="production_enabled")
        top_k = None if self.top_k is None else int(self.top_k)
        if scope == SCOPE_TOP_K_PLACEHOLDER and (top_k is None or top_k <= 0):
            raise ValueError("Regime Feature pairwise top_k_placeholder scope requires positive top_k metadata")
        object.__setattr__(self, "scope", scope)
        object.__setattr__(self, "interval", interval)
        object.__setattr__(self, "band", _text(self.band, field_name="band").lower())
        object.__setattr__(self, "policy_id", _text(self.policy_id, field_name="policy_id"))
        object.__setattr__(self, "core_basket_assets", _string_tuple(self.core_basket_assets, field_name="core_basket_assets", require_non_empty=False))
        object.__setattr__(self, "benchmark_anchors", _string_tuple(self.benchmark_anchors, field_name="benchmark_anchors", require_non_empty=False))
        object.__setattr__(self, "top_k", top_k)
        object.__setattr__(self, "broad_all_to_all_enabled", False)
        object.__setattr__(self, "micro_interval_pairwise_enabled", bool(self.micro_interval_pairwise_enabled))
        object.__setattr__(self, "dynamic_peer_discovery_enabled", False)
        object.__setattr__(self, "peer_clusters_enabled", False)
        object.__setattr__(self, "materialization_enabled", bool(self.materialization_enabled))
        object.__setattr__(self, "production_enabled", False)
        object.__setattr__(self, "schema_version", int(self.schema_version))
        object.__setattr__(self, "metadata", to_jsonable(dict(self.metadata)))

    @property
    def blocked_reasons(self) -> tuple[str, ...]:
        reasons: list[str] = []
        if self.scope == SCOPE_BROAD_ALL_TO_ALL_DISABLED:
            reasons.append(PAIRWISE_SCOPE_NOT_APPROVED)
            reasons.append("broad_all_to_all_disabled")
        if int(self.interval) in PAIRWISE_MICRO_GATED_INTERVALS and not self.micro_interval_pairwise_enabled:
            reasons.append(INTERVAL_NOT_APPROVED_FOR_PAIRWISE)
            reasons.append("micro_interval_pairwise_disabled")
        if self.scope == SCOPE_TOP_K_PLACEHOLDER:
            reasons.append(PEER_DISCOVERY_NOT_IMPLEMENTED)
            reasons.append("top_k_peer_discovery_placeholder_only")
        if self.scope == SCOPE_CORE_BASKET_ONLY and len(self.core_basket_assets) < 2:
            reasons.append(PAIRWISE_SCOPE_NOT_APPROVED)
            reasons.append("insufficient_core_basket_assets")
        if self.scope in {SCOPE_BENCHMARK_ANCHOR_ONLY, SCOPE_CORE_PLUS_ANCHORS} and not self.benchmark_anchors:
            reasons.append(PAIRWISE_SCOPE_NOT_APPROVED)
            reasons.append("missing_benchmark_anchors")
        return tuple(dict.fromkeys(reasons))

    @property
    def candidate_pairs(self) -> tuple[tuple[str, str], ...]:
        if self.blocked_reasons:
            return ()
        if self.scope == SCOPE_CORE_BASKET_ONLY:
            return _undirected_pairs(self.core_basket_assets)
        if self.scope == SCOPE_BENCHMARK_ANCHOR_ONLY:
            return _directed_pairs(self.core_basket_assets, self.benchmark_anchors)
        if self.scope == SCOPE_CORE_PLUS_ANCHORS:
            return tuple(dict.fromkeys((*_undirected_pairs(self.core_basket_assets), *_directed_pairs(self.core_basket_assets, self.benchmark_anchors))))
        return ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "artifact_kind": "pairwise_relationship_scope_policy",
            "policy_id": self.policy_id,
            "scope": self.scope,
            "interval": int(self.interval),
            "band": self.band,
            "core_basket_assets": list(self.core_basket_assets),
            "benchmark_anchors": list(self.benchmark_anchors),
            "benchmark_anchors_metadata_only": True,
            "top_k": self.top_k,
            "candidate_pairs": [list(pair) for pair in self.candidate_pairs],
            "candidate_pair_count": int(len(self.candidate_pairs)),
            "blocked_reasons": list(self.blocked_reasons),
            "blocked_statuses": list(self.blocked_reasons),
            "broad_pairwise_all_to_all_enabled": False,
            "broad_all_to_all_enabled": False,
            "sub_hour_pairwise_enabled": False,
            "micro_interval_pairwise_enabled": bool(self.micro_interval_pairwise_enabled),
            "dynamic_peer_discovery_enabled": False,
            "peer_clusters_enabled": False,
            "materialization_enabled": bool(self.materialization_enabled),
            "production_pairwise_writes_enabled": False,
            "production_enabled": False,
            "metadata": to_jsonable(dict(self.metadata)),
        }

    def to_json(self, **kwargs: Any) -> str:
        return dumps_json(self.as_dict(), **kwargs)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PairwiseRelationshipScopePolicy":
        obj = require_json_object(payload, context="PairwiseRelationshipScopePolicy")
        return cls(
            schema_version=obj.get("schema_version", REGIME_FEATURES_SCHEMA_VERSION),
            policy_id=obj.get("policy_id", "pairwise_scope_policy_v1"),
            scope=obj["scope"],
            interval=obj["interval"],
            band=obj["band"],
            core_basket_assets=obj.get("core_basket_assets", ()),
            benchmark_anchors=obj.get("benchmark_anchors", ()),
            top_k=obj.get("top_k"),
            broad_all_to_all_enabled=bool(obj.get("broad_all_to_all_enabled", False)),
            micro_interval_pairwise_enabled=bool(obj.get("micro_interval_pairwise_enabled", False)),
            dynamic_peer_discovery_enabled=bool(obj.get("dynamic_peer_discovery_enabled", False)),
            peer_clusters_enabled=bool(obj.get("peer_clusters_enabled", False)),
            materialization_enabled=bool(obj.get("materialization_enabled", False)),
            production_enabled=bool(obj.get("production_enabled", False)),
            metadata=obj.get("metadata", {}),
        )

    @classmethod
    def from_json(cls, text: str) -> "PairwiseRelationshipScopePolicy":
        return cls.from_dict(require_json_object(loads_json(text), context="PairwiseRelationshipScopePolicy JSON"))


@dataclass(frozen=True)
class PairwiseRelationshipRow:
    ts: int | float | str
    interval: int
    band: str
    window: int
    asset: str
    related_asset_or_benchmark: str
    relationship_type: str
    value: float | int | str | None
    sample_count: int
    coverage: float
    known_at_ts: int | float | str
    lineage_id: str
    directed: bool | None = None
    scope: str = SCOPE_CORE_BASKET_ONLY
    schema_version: int = REGIME_FEATURES_SCHEMA_VERSION

    def __post_init__(self) -> None:
        relationship_type = _relationship_type(self.relationship_type)
        directed = relationship_type in PAIRWISE_DIRECTED_RELATIONSHIP_TYPES if self.directed is None else bool(self.directed)
        if directed != (relationship_type in PAIRWISE_DIRECTED_RELATIONSHIP_TYPES):
            raise ValueError(f"Regime Feature pairwise row directed flag mismatches relationship_type {relationship_type!r}")
        asset, related = canonical_pair(
            self.asset,
            self.related_asset_or_benchmark,
            relationship_type=relationship_type,
            directed=directed,
        )
        interval = int(self.interval)
        window = int(self.window)
        if interval <= 0:
            raise ValueError("Regime Feature pairwise row interval must be positive")
        if window <= 1:
            raise ValueError("Regime Feature pairwise row window must be greater than 1")
        sample_count = int(self.sample_count)
        if sample_count < 0:
            raise ValueError("Regime Feature pairwise row sample_count must be non-negative")
        object.__setattr__(self, "ts", _orderable(self.ts, field_name="ts"))
        object.__setattr__(self, "interval", interval)
        object.__setattr__(self, "band", _text(self.band, field_name="band").lower())
        object.__setattr__(self, "window", window)
        object.__setattr__(self, "asset", asset)
        object.__setattr__(self, "related_asset_or_benchmark", related)
        object.__setattr__(self, "relationship_type", relationship_type)
        object.__setattr__(self, "value", None if self.value is None else float(self.value))
        object.__setattr__(self, "sample_count", sample_count)
        object.__setattr__(self, "coverage", _coverage(self.coverage, field_name="coverage"))
        object.__setattr__(self, "known_at_ts", _orderable(self.known_at_ts, field_name="known_at_ts"))
        object.__setattr__(self, "lineage_id", _text(self.lineage_id, field_name="lineage_id"))
        object.__setattr__(self, "directed", directed)
        object.__setattr__(self, "scope", _member(self.scope, PAIRWISE_RELATIONSHIP_SCOPES, field_name="scope"))
        object.__setattr__(self, "schema_version", int(self.schema_version))

    @property
    def pair_key(self) -> str:
        arrow = "->" if self.directed else "<->"
        return f"{self.asset}{arrow}{self.related_asset_or_benchmark}"

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "artifact_kind": "pairwise_relationship_row",
            "ts": self.ts,
            "interval": int(self.interval),
            "band": self.band,
            "window": int(self.window),
            "asset": self.asset,
            "related_asset_or_benchmark": self.related_asset_or_benchmark,
            "pair_key": self.pair_key,
            "relationship_type": self.relationship_type,
            "value": self.value,
            "sample_count": int(self.sample_count),
            "coverage": float(self.coverage),
            "known_at_ts": self.known_at_ts,
            "lineage_id": self.lineage_id,
            "directed": bool(self.directed),
            "scope": self.scope,
        }

    def to_json(self, **kwargs: Any) -> str:
        return dumps_json(self.as_dict(), **kwargs)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PairwiseRelationshipRow":
        obj = require_json_object(payload, context="PairwiseRelationshipRow")
        return cls(
            schema_version=obj.get("schema_version", REGIME_FEATURES_SCHEMA_VERSION),
            ts=obj["ts"],
            interval=obj["interval"],
            band=obj["band"],
            window=obj["window"],
            asset=obj["asset"],
            related_asset_or_benchmark=obj["related_asset_or_benchmark"],
            relationship_type=obj["relationship_type"],
            value=obj.get("value"),
            sample_count=obj["sample_count"],
            coverage=obj["coverage"],
            known_at_ts=obj["known_at_ts"],
            lineage_id=obj["lineage_id"],
            directed=obj.get("directed"),
            scope=obj.get("scope", SCOPE_CORE_BASKET_ONLY),
        )

    @classmethod
    def from_json(cls, text: str) -> "PairwiseRelationshipRow":
        return cls.from_dict(require_json_object(loads_json(text), context="PairwiseRelationshipRow JSON"))


@dataclass(frozen=True)
class PairwiseRelationshipBuildPlan:
    plan_id: str
    specs: Sequence[PairwiseRelationshipSpec | Mapping[str, Any]]
    scope_policy: PairwiseRelationshipScopePolicy | Mapping[str, Any]
    materialization_enabled: bool = False
    fixture_only: bool = False
    production_enabled: bool = False
    schema_version: int = REGIME_FEATURES_SCHEMA_VERSION
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        specs = tuple(spec if isinstance(spec, PairwiseRelationshipSpec) else PairwiseRelationshipSpec.from_dict(spec) for spec in self.specs)
        if not specs:
            raise ValueError("Regime Feature pairwise build plan requires at least one relationship spec")
        policy = self.scope_policy if isinstance(self.scope_policy, PairwiseRelationshipScopePolicy) else PairwiseRelationshipScopePolicy.from_dict(self.scope_policy)
        for spec in specs:
            if spec.interval != policy.interval or spec.band != policy.band:
                raise ValueError("Regime Feature pairwise build plan specs must match scope policy interval/band")
        _require_disabled(self.production_enabled, field_name="production_enabled")
        if self.materialization_enabled and not self.fixture_only:
            raise ValueError(f"{PAIRWISE_SCOPE_NOT_APPROVED}: Regime Feature pairwise materialization requires fixture_only=True in this sprint")
        object.__setattr__(self, "plan_id", _text(self.plan_id, field_name="plan_id"))
        object.__setattr__(self, "specs", specs)
        object.__setattr__(self, "scope_policy", policy)
        object.__setattr__(self, "materialization_enabled", bool(self.materialization_enabled))
        object.__setattr__(self, "fixture_only", bool(self.fixture_only))
        object.__setattr__(self, "production_enabled", False)
        object.__setattr__(self, "schema_version", int(self.schema_version))
        object.__setattr__(self, "metadata", to_jsonable(dict(self.metadata)))

    @property
    def blocked_reasons(self) -> tuple[str, ...]:
        reasons = list(self.scope_policy.blocked_reasons)
        if self.fixture_only:
            reasons.append("fixture_only_not_production")
        if not self.materialization_enabled:
            reasons.append("materialization_disabled")
        return tuple(dict.fromkeys(reasons))

    @property
    def status(self) -> str:
        hard_blocks = [reason for reason in self.blocked_reasons if reason not in {"materialization_disabled", "fixture_only_not_production"}]
        if hard_blocks:
            return PLAN_STATUS_BLOCKED
        if self.fixture_only:
            return PLAN_STATUS_FIXTURE_ONLY
        return PLAN_STATUS_SCAFFOLD_READY

    @property
    def estimated_pair_count(self) -> int:
        return int(len(self.scope_policy.candidate_pairs) * len(self.specs))

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "artifact_kind": "pairwise_relationship_build_plan",
            "plan_id": self.plan_id,
            "artifact_family": PAIRWISE_RELATIONSHIP_FEATURES,
            "status": self.status,
            "specs": [spec.as_dict() for spec in self.specs],
            "scope_policy": self.scope_policy.as_dict(),
            "estimated_pair_count": int(self.estimated_pair_count),
            "blocked_reasons": list(self.blocked_reasons),
            "blocked_statuses": list(self.blocked_reasons),
            "output_schema": pairwise_relationship_output_schema().as_dict(),
            "artifact_boundary": RegimeFeatureArtifactBoundary(artifact_family=PAIRWISE_RELATIONSHIP_FEATURES).as_dict(),
            "broad_pairwise_all_to_all_enabled": False,
            "broad_all_to_all_enabled": False,
            "sub_hour_pairwise_enabled": False,
            "dynamic_peer_discovery_enabled": False,
            "peer_clusters_enabled": False,
            "materialization_enabled": bool(self.materialization_enabled),
            "fixture_only": bool(self.fixture_only),
            "production_pairwise_writes_enabled": False,
            "production_enabled": False,
            "metadata": to_jsonable(dict(self.metadata)),
        }

    def to_json(self, **kwargs: Any) -> str:
        return dumps_json(self.as_dict(), **kwargs)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PairwiseRelationshipBuildPlan":
        obj = require_json_object(payload, context="PairwiseRelationshipBuildPlan")
        return cls(
            schema_version=obj.get("schema_version", REGIME_FEATURES_SCHEMA_VERSION),
            plan_id=obj["plan_id"],
            specs=obj.get("specs", ()),
            scope_policy=obj["scope_policy"],
            materialization_enabled=bool(obj.get("materialization_enabled", False)),
            fixture_only=bool(obj.get("fixture_only", False)),
            production_enabled=bool(obj.get("production_enabled", False)),
            metadata=obj.get("metadata", {}),
        )

    @classmethod
    def from_json(cls, text: str) -> "PairwiseRelationshipBuildPlan":
        return cls.from_dict(require_json_object(loads_json(text), context="PairwiseRelationshipBuildPlan JSON"))


@dataclass(frozen=True)
class PairwiseRelationshipScaffold:
    scaffold_id: str = "pairwise_relationship_scaffold_v1"
    relationship_types: Sequence[str] = PAIRWISE_RELATIONSHIP_TYPES
    allowed_scopes: Sequence[str] = PAIRWISE_RELATIONSHIP_SCOPES
    materialization_enabled: bool = False
    broad_all_to_all_enabled: bool = False
    dynamic_peer_discovery_enabled: bool = False
    production_enabled: bool = False
    schema_version: int = REGIME_FEATURES_SCHEMA_VERSION
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        types = tuple(_relationship_type(value) for value in self.relationship_types)
        scopes = tuple(_member(value, PAIRWISE_RELATIONSHIP_SCOPES, field_name="allowed_scope") for value in self.allowed_scopes)
        _require_disabled(self.materialization_enabled, field_name="materialization_enabled")
        _require_disabled(self.broad_all_to_all_enabled, field_name="broad_all_to_all_enabled", status=PAIRWISE_SCOPE_NOT_APPROVED)
        _require_disabled(self.dynamic_peer_discovery_enabled, field_name="dynamic_peer_discovery_enabled", status=PEER_DISCOVERY_NOT_IMPLEMENTED)
        _require_disabled(self.production_enabled, field_name="production_enabled")
        object.__setattr__(self, "scaffold_id", _text(self.scaffold_id, field_name="scaffold_id"))
        object.__setattr__(self, "relationship_types", types)
        object.__setattr__(self, "allowed_scopes", scopes)
        object.__setattr__(self, "materialization_enabled", False)
        object.__setattr__(self, "broad_all_to_all_enabled", False)
        object.__setattr__(self, "dynamic_peer_discovery_enabled", False)
        object.__setattr__(self, "production_enabled", False)
        object.__setattr__(self, "schema_version", int(self.schema_version))
        object.__setattr__(self, "metadata", to_jsonable(dict(self.metadata)))

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "artifact_kind": "pairwise_relationship_feature_scaffold",
            "scaffold_id": self.scaffold_id,
            "artifact_family": PAIRWISE_RELATIONSHIP_FEATURES,
            "relationship_types": list(self.relationship_types),
            "directed_relationship_types": list(PAIRWISE_DIRECTED_RELATIONSHIP_TYPES),
            "symmetric_relationship_types": list(PAIRWISE_SYMMETRIC_RELATIONSHIP_TYPES),
            "output_schema": pairwise_relationship_output_schema().as_dict(),
            "artifact_boundary": RegimeFeatureArtifactBoundary(artifact_family=PAIRWISE_RELATIONSHIP_FEATURES).as_dict(),
            "allowed_scopes": list(self.allowed_scopes),
            "micro_gated_intervals": list(PAIRWISE_MICRO_GATED_INTERVALS),
            "default_block_statuses": [
                PAIRWISE_SCOPE_NOT_APPROVED,
                INTERVAL_NOT_APPROVED_FOR_PAIRWISE,
                PEER_DISCOVERY_NOT_IMPLEMENTED,
                PRODUCTION_PAIRWISE_WRITES_DISABLED,
            ],
            "materialization_enabled": False,
            "broad_pairwise_all_to_all_enabled": False,
            "broad_all_to_all_enabled": False,
            "sub_hour_pairwise_enabled": False,
            "dynamic_peer_discovery_enabled": False,
            "peer_clusters_enabled": False,
            "production_pairwise_writes_enabled": False,
            "production_enabled": False,
            "metadata": to_jsonable(dict(self.metadata)),
        }

    def to_json(self, **kwargs: Any) -> str:
        return dumps_json(self.as_dict(), **kwargs)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PairwiseRelationshipScaffold":
        obj = require_json_object(payload, context="PairwiseRelationshipScaffold")
        return cls(
            schema_version=obj.get("schema_version", REGIME_FEATURES_SCHEMA_VERSION),
            scaffold_id=obj["scaffold_id"],
            relationship_types=obj.get("relationship_types", PAIRWISE_RELATIONSHIP_TYPES),
            allowed_scopes=obj.get("allowed_scopes", PAIRWISE_RELATIONSHIP_SCOPES),
            materialization_enabled=bool(obj.get("materialization_enabled", False)),
            broad_all_to_all_enabled=bool(obj.get("broad_all_to_all_enabled", False)),
            dynamic_peer_discovery_enabled=bool(obj.get("dynamic_peer_discovery_enabled", False)),
            production_enabled=bool(obj.get("production_enabled", False)),
            metadata=obj.get("metadata", {}),
        )

    @classmethod
    def from_json(cls, text: str) -> "PairwiseRelationshipScaffold":
        return cls.from_dict(require_json_object(loads_json(text), context="PairwiseRelationshipScaffold JSON"))


def default_pairwise_relationship_scaffold() -> PairwiseRelationshipScaffold:
    return PairwiseRelationshipScaffold()


def pairwise_legacy_scaffold_status() -> dict[str, Any]:
    """Return the legacy pairwise routing contract without enabling execution."""

    return {
        "artifact_kind": "pairwise_legacy_scaffold_status",
        "surface_status": PAIRWISE_LEGACY_SURFACE_STATUS,
        "process1_canonical_module": PAIRWISE_PROCESS1_CANONICAL_MODULE,
        "compatibility_types_only": True,
        "broad_all_to_all_enabled": False,
        "dynamic_peer_discovery_enabled": False,
        "peer_clusters_enabled": False,
        "materialization_enabled": False,
        "production_enabled": False,
        "production_writer_exposed": False,
    }


def canonical_pair(
    asset: str,
    related_asset_or_benchmark: str,
    *,
    relationship_type: str,
    directed: bool | None = None,
) -> tuple[str, str]:
    relationship = _relationship_type(relationship_type)
    is_directed = relationship in PAIRWISE_DIRECTED_RELATIONSHIP_TYPES if directed is None else bool(directed)
    left = _text(asset, field_name="asset")
    right = _text(related_asset_or_benchmark, field_name="related_asset_or_benchmark")
    if left == right:
        raise ValueError("Regime Feature pairwise asset and related_asset_or_benchmark must differ")
    if is_directed:
        return left, right
    return tuple(sorted((left, right)))


def build_fixture_pairwise_rows(
    returns: pd.DataFrame,
    *,
    spec: PairwiseRelationshipSpec,
    scope_policy: PairwiseRelationshipScopePolicy,
    known_at_ts: int,
    lineage_id: str,
) -> tuple[PairwiseRelationshipRow, ...]:
    if not scope_policy.materialization_enabled:
        raise ValueError("Regime Feature fixture pairwise rows require materialization_enabled=True on scope policy")
    if spec.relationship_type != RELATIONSHIP_ROLLING_CORR:
        raise ValueError("Regime Feature fixture pairwise rows only implement rolling_corr schema validation")
    if spec.interval != scope_policy.interval or spec.band != scope_policy.band:
        raise ValueError("Regime Feature fixture pairwise spec and scope policy interval/band must match")
    if scope_policy.blocked_reasons:
        reasons = ", ".join(scope_policy.blocked_reasons)
        raise ValueError(f"Regime Feature fixture pairwise rows blocked by scope policy: {reasons}")
    frame = returns.copy()
    frame.index = pd.to_numeric(frame.index, errors="coerce").astype("int64")
    rows: list[PairwiseRelationshipRow] = []
    for asset, related in scope_policy.candidate_pairs:
        if asset not in frame.columns or related not in frame.columns:
            continue
        pair_frame = frame[[asset, related]].tail(int(spec.window)).dropna(how="any")
        sample_count = int(pair_frame.shape[0])
        value = None
        if sample_count >= int(spec.min_periods or 0):
            value = float(pair_frame[asset].corr(pair_frame[related]))
        rows.append(
            PairwiseRelationshipRow(
                ts=int(frame.index.max()),
                interval=int(spec.interval),
                band=spec.band,
                window=int(spec.window),
                asset=asset,
                related_asset_or_benchmark=related,
                relationship_type=spec.relationship_type,
                value=value,
                sample_count=sample_count,
                coverage=float(sample_count / max(1, int(spec.window))),
                known_at_ts=int(known_at_ts),
                lineage_id=lineage_id,
                scope=scope_policy.scope,
            )
        )
    return tuple(rows)


def require_pairwise_plan_approved(plan: PairwiseRelationshipBuildPlan | Mapping[str, Any]) -> None:
    resolved = plan if isinstance(plan, PairwiseRelationshipBuildPlan) else PairwiseRelationshipBuildPlan.from_dict(plan)
    if resolved.status == PLAN_STATUS_BLOCKED:
        reasons = ", ".join(resolved.blocked_reasons)
        raise ValueError(f"{PAIRWISE_SCOPE_NOT_APPROVED}: pairwise plan is not approved for execution: {reasons}")
    if not resolved.fixture_only:
        raise ValueError(f"{PAIRWISE_SCOPE_NOT_APPROVED}: pairwise computation is scaffold-only unless fixture_only=True")
    if not resolved.materialization_enabled:
        raise ValueError("materialization_disabled: pairwise materialization is disabled")


def _undirected_pairs(assets: Sequence[str]) -> tuple[tuple[str, str], ...]:
    normalized = sorted(dict.fromkeys(_string_tuple(assets, field_name="assets", require_non_empty=False)))
    pairs: list[tuple[str, str]] = []
    for idx, left in enumerate(normalized):
        for right in normalized[idx + 1 :]:
            pairs.append((left, right))
    return tuple(pairs)


def _directed_pairs(assets: Sequence[str], related_assets: Sequence[str]) -> tuple[tuple[str, str], ...]:
    left_assets = tuple(dict.fromkeys(_string_tuple(assets, field_name="assets", require_non_empty=False)))
    right_assets = tuple(dict.fromkeys(_string_tuple(related_assets, field_name="related_assets", require_non_empty=False)))
    return tuple((left, right) for left in left_assets for right in right_assets if left != right)


def _relationship_type(value: object) -> str:
    text = _text(value, field_name="relationship_type")
    if text not in PAIRWISE_RELATIONSHIP_TYPES:
        valid = ", ".join(PAIRWISE_RELATIONSHIP_TYPES)
        raise ValueError(f"Unsupported Regime Feature pairwise relationship_type {text!r}; expected one of: {valid}")
    return text


def _member(value: object, allowed: Sequence[str], *, field_name: str) -> str:
    text = _text(value, field_name=field_name)
    if text not in allowed:
        valid = ", ".join(allowed)
        raise ValueError(f"Unsupported Regime Feature pairwise {field_name} {text!r}; expected one of: {valid}")
    return text


def _string_tuple(values: Sequence[str], *, field_name: str, require_non_empty: bool) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValueError(f"Regime Feature pairwise {field_name} must be a sequence of strings")
    out = tuple(_text(value, field_name=field_name) for value in values)
    if require_non_empty and not out:
        raise ValueError(f"Regime Feature pairwise {field_name} must be non-empty")
    return out


def _coverage(value: Any, *, field_name: str) -> float:
    val = float(value)
    if val < 0.0 or val > 1.0:
        raise ValueError(f"Regime Feature pairwise {field_name} must be within [0, 1]")
    return val


def _orderable(value: Any, *, field_name: str) -> int | float | str:
    if value is None:
        raise ValueError(f"Regime Feature pairwise {field_name} must be non-empty")
    return value


def _text(value: object, *, field_name: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"Regime Feature pairwise {field_name} must be non-empty")
    return text


def _require_disabled(value: object, *, field_name: str, status: str | None = None) -> None:
    if value is not False:
        prefix = f"{status}: " if status else ""
        raise ValueError(f"{prefix}Regime Feature pairwise {field_name} must remain disabled in this sprint")


__all__ = [
    "INTERVAL_NOT_APPROVED_FOR_PAIRWISE",
    "PAIRWISE_DIRECTED_RELATIONSHIP_TYPES",
    "PAIRWISE_MICRO_GATED_INTERVALS",
    "PAIRWISE_RELATIONSHIP_SCOPES",
    "PAIRWISE_RELATIONSHIP_TYPES",
    "PAIRWISE_SCOPE_NOT_APPROVED",
    "PAIRWISE_SYMMETRIC_RELATIONSHIP_TYPES",
    "PEER_DISCOVERY_NOT_IMPLEMENTED",
    "PAIRWISE_LEGACY_SURFACE_STATUS",
    "PAIRWISE_PROCESS1_CANONICAL_MODULE",
    "PRODUCTION_PAIRWISE_WRITES_DISABLED",
    "RELATIONSHIP_CORRELATION_DISTANCE",
    "RELATIONSHIP_LEAD_LAG_CORR",
    "RELATIONSHIP_RESIDUAL_CORR",
    "RELATIONSHIP_RETURN_DISTANCE",
    "RELATIONSHIP_ROLLING_BETA",
    "RELATIONSHIP_ROLLING_CORR",
    "RELATIONSHIP_ROLLING_COV",
    "RELATIONSHIP_VOLATILITY_DISTANCE",
    "SCOPE_BENCHMARK_ANCHOR_ONLY",
    "SCOPE_BROAD_ALL_TO_ALL_DISABLED",
    "SCOPE_CORE_BASKET_ONLY",
    "SCOPE_CORE_PLUS_ANCHORS",
    "SCOPE_TOP_K_PLACEHOLDER",
    "PairwiseRelationshipBuildPlan",
    "PairwiseRelationshipRow",
    "PairwiseRelationshipScaffold",
    "PairwiseRelationshipScopePolicy",
    "PairwiseRelationshipSpec",
    "build_fixture_pairwise_rows",
    "canonical_pair",
    "default_pairwise_relationship_scaffold",
    "pairwise_legacy_scaffold_status",
    "require_pairwise_plan_approved",
]
