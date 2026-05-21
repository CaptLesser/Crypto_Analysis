from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from src.regimes.core.forecaster_handoff import (
    ARTIFACT_KIND_MARKET_STATE_AXIS_PANEL,
    ARTIFACT_KIND_MARKET_STATE_FEATURE_PANEL,
    PATHWAY_MARKET_STATE,
    RegimeForecasterHandoffManifest,
    make_regime_forecaster_handoff_manifest,
    validate_regime_forecaster_handoff_manifest,
    write_regime_forecaster_handoff_manifest,
)


MARKET_STATE_HANDOFF_PRODUCER = "src.regimes.market_state.handoff"


def build_market_state_forecaster_handoff_manifest(
    *,
    artifact_kind: str,
    source_artifact_refs: Mapping[str, Any],
    output_artifact_refs: Mapping[str, Any],
    band: str,
    interval: int,
    known_at_ts: int | float | str,
    source_tail_ts: int | float | str,
    lineage_id: str,
    axis: str | None = None,
    feature_family: str | None = None,
    refit_key: str | None = None,
    profile_id: str | None = None,
    feature_profile_id: str | None = None,
    consumer_notes: str | None = None,
) -> RegimeForecasterHandoffManifest:
    return make_regime_forecaster_handoff_manifest(
        pathway=PATHWAY_MARKET_STATE,
        artifact_kind=artifact_kind,
        axis=axis,
        feature_family=feature_family,
        band=band,
        interval=interval,
        refit_key=refit_key,
        profile_id=profile_id,
        feature_profile_id=feature_profile_id,
        source_artifact_refs=source_artifact_refs,
        output_artifact_refs=output_artifact_refs,
        known_at_ts=known_at_ts,
        source_tail_ts=source_tail_ts,
        lineage_id=lineage_id,
        consumer_notes=consumer_notes or "Market-State sandbox handoff; no monolithic label or final profile selection.",
    )


def validate_market_state_forecaster_handoff_manifest(
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
    if resolved.pathway != PATHWAY_MARKET_STATE:
        raise ValueError("Market-State forecaster handoff pathway must be market_state")
    if resolved.artifact_kind not in {ARTIFACT_KIND_MARKET_STATE_AXIS_PANEL, ARTIFACT_KIND_MARKET_STATE_FEATURE_PANEL}:
        raise ValueError("Market-State forecaster handoff artifact_kind must be a Market-State feature or axis panel")
    return resolved


def write_market_state_forecaster_handoff_manifest(
    manifest: RegimeForecasterHandoffManifest | Mapping[str, Any],
    *,
    output_root: str | Path,
    relative_path: str | Path,
    artifact_root: str | Path | None = None,
    report_root: str | Path | None = None,
    write_outputs: bool = True,
) -> Path:
    resolved = validate_market_state_forecaster_handoff_manifest(
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
    "MARKET_STATE_HANDOFF_PRODUCER",
    "build_market_state_forecaster_handoff_manifest",
    "validate_market_state_forecaster_handoff_manifest",
    "write_market_state_forecaster_handoff_manifest",
]
