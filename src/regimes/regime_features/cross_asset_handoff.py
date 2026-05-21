from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from src.regimes.core.forecaster_handoff import (
    ARTIFACT_KIND_CROSS_ASSET_FEATURE_ROWS,
    ARTIFACT_KIND_RELATIONSHIP_DISCOVERY_ARTIFACTS,
    PATHWAY_CROSS_ASSET,
    RegimeForecasterHandoffManifest,
    make_regime_forecaster_handoff_manifest,
    validate_regime_forecaster_handoff_manifest,
    write_regime_forecaster_handoff_manifest,
)


CROSS_ASSET_HANDOFF_PRODUCER = "src.regimes.regime_features.cross_asset_handoff"


def build_cross_asset_forecaster_handoff_manifest(
    *,
    artifact_kind: str,
    source_artifact_refs: Mapping[str, Any],
    output_artifact_refs: Mapping[str, Any],
    band: str,
    interval: int,
    known_at_ts: int | float | str,
    source_tail_ts: int | float | str,
    lineage_id: str,
    feature_family: str | None = None,
    asset: str | None = None,
    refit_key: str | None = None,
    profile_id: str | None = None,
    feature_profile_id: str | None = None,
    consumer_notes: str | None = None,
) -> RegimeForecasterHandoffManifest:
    return make_regime_forecaster_handoff_manifest(
        pathway=PATHWAY_CROSS_ASSET,
        artifact_kind=artifact_kind,
        feature_family=feature_family,
        band=band,
        interval=interval,
        asset=asset,
        refit_key=refit_key,
        profile_id=profile_id,
        feature_profile_id=feature_profile_id,
        source_artifact_refs=source_artifact_refs,
        output_artifact_refs=output_artifact_refs,
        known_at_ts=known_at_ts,
        source_tail_ts=source_tail_ts,
        lineage_id=lineage_id,
        consumer_notes=consumer_notes
        or "Cross-Asset sandbox handoff; no Cross-Asset regime labels, peer clusters, or production promotion.",
    )


def validate_cross_asset_forecaster_handoff_manifest(
    manifest: RegimeForecasterHandoffManifest | Mapping[str, Any],
    *,
    artifact_root: str | Path | None = None,
    report_root: str | Path | None = None,
    write_outputs: bool = False,
) -> RegimeForecasterHandoffManifest:
    resolved = validate_regime_forecaster_handoff_manifest(
        manifest,
        artifact_root=artifact_root,
        report_root=report_root,
        write_outputs=write_outputs,
    )
    if resolved.pathway != PATHWAY_CROSS_ASSET:
        raise ValueError("Cross-Asset forecaster handoff pathway must be cross_asset")
    if resolved.artifact_kind not in {
        ARTIFACT_KIND_RELATIONSHIP_DISCOVERY_ARTIFACTS,
        ARTIFACT_KIND_CROSS_ASSET_FEATURE_ROWS,
    }:
        raise ValueError("Cross-Asset forecaster handoff artifact_kind must be Relationship Discovery artifacts or feature rows")
    return resolved


def write_cross_asset_forecaster_handoff_manifest(
    manifest: RegimeForecasterHandoffManifest | Mapping[str, Any],
    *,
    output_root: str | Path,
    relative_path: str | Path,
    artifact_root: str | Path | None = None,
    report_root: str | Path | None = None,
    write_outputs: bool = True,
) -> Path:
    resolved = validate_cross_asset_forecaster_handoff_manifest(
        manifest,
        artifact_root=artifact_root or output_root,
        report_root=report_root,
        write_outputs=write_outputs,
    )
    return write_regime_forecaster_handoff_manifest(
        resolved,
        output_root=output_root,
        relative_path=relative_path,
        artifact_root=artifact_root or output_root,
        report_root=report_root,
        write_outputs=write_outputs,
    )


__all__ = [
    "CROSS_ASSET_HANDOFF_PRODUCER",
    "build_cross_asset_forecaster_handoff_manifest",
    "validate_cross_asset_forecaster_handoff_manifest",
    "write_cross_asset_forecaster_handoff_manifest",
]
