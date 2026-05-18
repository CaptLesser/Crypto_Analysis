"""Convenience builders for compact downstream Regime artifact contracts.

This module intentionally creates schema/spec scaffolding only. It does not
build forecast targets, implement Numerics integration, or materialize broad
downstream feature views.
"""

from __future__ import annotations

from typing import Any, Mapping

from src.regimes.core.clamp_policy import RegimeClampPolicy
from src.regimes.core.consumer_artifacts import (
    ASSET_STATE_COMPOSITION_SUMMARY,
    CROSS_ASSET_PEER_SUMMARY,
    MARKET_STATE_FEATURE_VIEW,
    NUMERICS_CANDIDATE_REGIME_FEATURE_EXPORT,
    REGIME_FORECAST_FEATURE_VIEW,
    REGIME_TRANSITION_SUMMARY,
    CompactRegimeArtifactSpec,
    CompactRegimeSchemaStub,
    compact_regime_schema_stub_by_kind,
    compact_regime_schema_stubs,
)
from src.regimes.core.known_at import KnownAtSpec
from src.regimes.core.lineage import RegimeLineageSpec


def downstream_schema_stubs() -> tuple[CompactRegimeSchemaStub, ...]:
    return compact_regime_schema_stubs()


def downstream_schema_stub(artifact_kind: str) -> CompactRegimeSchemaStub:
    stubs = compact_regime_schema_stub_by_kind()
    key = str(artifact_kind).strip().lower()
    if key not in stubs:
        raise ValueError(f"Unknown compact downstream artifact kind {artifact_kind!r}")
    return stubs[key]


def build_compact_downstream_artifact_spec(
    *,
    artifact_kind: str,
    pathway: str,
    interval: int,
    timestamp_key: str,
    known_at_policy: KnownAtSpec | Mapping[str, Any],
    lineage: RegimeLineageSpec | Mapping[str, Any],
    clamp_policy: RegimeClampPolicy | Mapping[str, Any],
    intended_consumer: str,
    axis: str | None = None,
    band: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> CompactRegimeArtifactSpec:
    return CompactRegimeArtifactSpec(
        artifact_kind=artifact_kind,
        pathway=pathway,
        axis=axis,
        band=band,
        interval=interval,
        timestamp_key=timestamp_key,
        known_at_policy=known_at_policy,
        lineage=lineage,
        clamp_policy=clamp_policy,
        intended_consumer=intended_consumer,
        metadata=metadata or {},
    )


__all__ = [
    "ASSET_STATE_COMPOSITION_SUMMARY",
    "CROSS_ASSET_PEER_SUMMARY",
    "MARKET_STATE_FEATURE_VIEW",
    "NUMERICS_CANDIDATE_REGIME_FEATURE_EXPORT",
    "REGIME_FORECAST_FEATURE_VIEW",
    "REGIME_TRANSITION_SUMMARY",
    "build_compact_downstream_artifact_spec",
    "downstream_schema_stub",
    "downstream_schema_stubs",
]
