from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from src.regimes.core.forecaster_handoff import (
    ARTIFACT_KIND_ASSET_STATE_SANDBOX_LABELS,
    PATHWAY_ASSET_STATE,
    RegimeForecasterHandoffManifest,
    make_regime_forecaster_handoff_manifest,
    validate_regime_forecaster_handoff_manifest,
    write_regime_forecaster_handoff_manifest,
)


ASSET_STATE_HANDOFF_PRODUCER = "src.regimes.asset_state.handoff"


def build_asset_state_forecaster_handoff_manifest(
    *,
    source_artifact_refs: Mapping[str, Any],
    output_artifact_refs: Mapping[str, Any],
    band: str,
    interval: int,
    known_at_ts: int | float | str,
    source_tail_ts: int | float | str,
    lineage_id: str,
    axis: str | None = None,
    asset: str | None = None,
    refit_key: str | None = None,
    profile_id: str | None = None,
    feature_profile_id: str | None = None,
    consumer_notes: str | None = None,
) -> RegimeForecasterHandoffManifest:
    return make_regime_forecaster_handoff_manifest(
        pathway=PATHWAY_ASSET_STATE,
        artifact_kind=ARTIFACT_KIND_ASSET_STATE_SANDBOX_LABELS,
        axis=axis,
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
        consumer_notes=consumer_notes or "Asset-State sandbox output handoff; no forecaster implementation or production labels.",
    )


def validate_asset_state_forecaster_handoff_manifest(
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
    if resolved.pathway != PATHWAY_ASSET_STATE:
        raise ValueError("Asset-State forecaster handoff pathway must be asset_state")
    if resolved.artifact_kind != ARTIFACT_KIND_ASSET_STATE_SANDBOX_LABELS:
        raise ValueError("Asset-State forecaster handoff artifact_kind must be asset_state_sandbox_labels")
    return resolved


def write_asset_state_forecaster_handoff_manifest(
    manifest: RegimeForecasterHandoffManifest | Mapping[str, Any],
    *,
    output_root: str | Path,
    relative_path: str | Path,
    artifact_root: str | Path | None = None,
    report_root: str | Path | None = None,
    write_outputs: bool = True,
) -> Path:
    resolved = validate_asset_state_forecaster_handoff_manifest(
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
    "ASSET_STATE_HANDOFF_PRODUCER",
    "build_asset_state_forecaster_handoff_manifest",
    "validate_asset_state_forecaster_handoff_manifest",
    "write_asset_state_forecaster_handoff_manifest",
]
