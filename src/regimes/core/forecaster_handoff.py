from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from src.regimes.core.artifact_refs import (
    make_artifact_ref,
    portable_ref_dict,
    resolve_artifact_ref,
    validate_portable_artifact_ref,
)
from src.regimes.core.serialization import to_jsonable


REGIME_FORECASTER_HANDOFF_SCHEMA_VERSION = 1
REGIME_FORECASTER_HANDOFF_ARTIFACT_KIND = "regime_forecaster_handoff_manifest"

PATHWAY_ASSET_STATE = "asset_state"
PATHWAY_MARKET_STATE = "market_state"
PATHWAY_CROSS_ASSET = "cross_asset"
REGIME_FORECASTER_HANDOFF_PATHWAYS: frozenset[str] = frozenset(
    {PATHWAY_ASSET_STATE, PATHWAY_MARKET_STATE, PATHWAY_CROSS_ASSET}
)

ARTIFACT_KIND_ASSET_STATE_SANDBOX_LABELS = "asset_state_sandbox_labels"
ARTIFACT_KIND_ASSET_STATE_FEATURE_MATRIX_OR_PANEL = "asset_state_feature_matrix_or_panel"
ARTIFACT_KIND_MARKET_STATE_FEATURE_PANEL = "market_state_feature_panel"
ARTIFACT_KIND_MARKET_STATE_AXIS_PANEL = "market_state_axis_panel"
ARTIFACT_KIND_RELATIONSHIP_DISCOVERY_ARTIFACTS = "relationship_discovery_artifacts"
ARTIFACT_KIND_CROSS_ASSET_FEATURE_ROWS = "cross_asset_feature_rows"
REGIME_FORECASTER_HANDOFF_PAYLOAD_KINDS: frozenset[str] = frozenset(
    {
        ARTIFACT_KIND_ASSET_STATE_SANDBOX_LABELS,
        ARTIFACT_KIND_ASSET_STATE_FEATURE_MATRIX_OR_PANEL,
        ARTIFACT_KIND_MARKET_STATE_FEATURE_PANEL,
        ARTIFACT_KIND_MARKET_STATE_AXIS_PANEL,
        ARTIFACT_KIND_RELATIONSHIP_DISCOVERY_ARTIFACTS,
        ARTIFACT_KIND_CROSS_ASSET_FEATURE_ROWS,
    }
)


def default_forecaster_handoff_boundary() -> dict[str, Any]:
    return {
        "classification": "sandbox_non_production",
        "production_enabled": False,
        "production_outputs_written": False,
        "production_labels_written": False,
        "production_promotion_performed": False,
        "forecaster_implemented": False,
    }


@dataclass(frozen=True)
class RegimeForecasterHandoffManifest:
    handoff_id: str
    pathway: str
    artifact_kind: str
    band: str
    interval: int
    source_artifact_refs: Mapping[str, Any]
    output_artifact_refs: Mapping[str, Any]
    known_at_ts: int | float | str
    source_tail_ts: int | float | str
    lineage_id: str
    axis: str | None = None
    feature_family: str | None = None
    asset: str | None = None
    refit_key: str | None = None
    profile_id: str | None = None
    feature_profile_id: str | None = None
    schema_version: int = REGIME_FORECASTER_HANDOFF_SCHEMA_VERSION
    artifact_boundary: Mapping[str, Any] = field(default_factory=default_forecaster_handoff_boundary)
    production_enabled: bool = False
    production_outputs_written: bool = False
    consumer_notes: str | None = None

    def __post_init__(self) -> None:
        if self.production_enabled is not False:
            raise ValueError("Regime forecaster handoff production_enabled must be false")
        if self.production_outputs_written is not False:
            raise ValueError("Regime forecaster handoff production_outputs_written must be false")
        object.__setattr__(self, "handoff_id", _text(self.handoff_id, field_name="handoff_id"))
        pathway = _text(self.pathway, field_name="pathway")
        if pathway not in REGIME_FORECASTER_HANDOFF_PATHWAYS:
            raise ValueError(f"Regime forecaster handoff pathway must be one of {sorted(REGIME_FORECASTER_HANDOFF_PATHWAYS)}")
        object.__setattr__(self, "pathway", pathway)
        payload_kind = _text(self.artifact_kind, field_name="artifact_kind")
        if payload_kind not in REGIME_FORECASTER_HANDOFF_PAYLOAD_KINDS:
            raise ValueError(
                f"Regime forecaster handoff artifact_kind must be one of {sorted(REGIME_FORECASTER_HANDOFF_PAYLOAD_KINDS)}"
            )
        object.__setattr__(self, "artifact_kind", payload_kind)
        object.__setattr__(self, "band", _text(self.band, field_name="band"))
        object.__setattr__(self, "interval", _positive_int(self.interval, field_name="interval"))
        object.__setattr__(self, "lineage_id", _text(self.lineage_id, field_name="lineage_id"))
        if _to_orderable(self.source_tail_ts, field_name="source_tail_ts") > _to_orderable(
            self.known_at_ts, field_name="known_at_ts"
        ):
            raise ValueError("Regime forecaster handoff source_tail_ts must not exceed known_at_ts")
        object.__setattr__(self, "schema_version", int(self.schema_version))
        if int(self.schema_version) != REGIME_FORECASTER_HANDOFF_SCHEMA_VERSION:
            raise ValueError("Regime forecaster handoff schema_version is unsupported")
        object.__setattr__(self, "source_artifact_refs", _refs(self.source_artifact_refs, field_name="source_artifact_refs"))
        object.__setattr__(self, "output_artifact_refs", _refs(self.output_artifact_refs, field_name="output_artifact_refs"))
        object.__setattr__(self, "artifact_boundary", _artifact_boundary(self.artifact_boundary))
        for name in ("axis", "feature_family", "asset", "refit_key", "profile_id", "feature_profile_id", "consumer_notes"):
            object.__setattr__(self, name, _optional_text(getattr(self, name)))

    def as_dict(self) -> dict[str, Any]:
        return {
            "handoff_id": self.handoff_id,
            "pathway": self.pathway,
            "artifact_kind": self.artifact_kind,
            "axis": self.axis,
            "feature_family": self.feature_family,
            "band": self.band,
            "interval": int(self.interval),
            "asset": self.asset,
            "refit_key": self.refit_key,
            "profile_id": self.profile_id,
            "feature_profile_id": self.feature_profile_id,
            "source_artifact_refs": to_jsonable(dict(self.source_artifact_refs)),
            "output_artifact_refs": to_jsonable(dict(self.output_artifact_refs)),
            "known_at_ts": self.known_at_ts,
            "source_tail_ts": self.source_tail_ts,
            "lineage_id": self.lineage_id,
            "schema_version": int(self.schema_version),
            "artifact_boundary": to_jsonable(dict(self.artifact_boundary)),
            "production_enabled": False,
            "production_outputs_written": False,
            "consumer_notes": self.consumer_notes,
        }


def make_regime_forecaster_handoff_manifest(
    *,
    pathway: str,
    artifact_kind: str,
    band: str,
    interval: int,
    source_artifact_refs: Mapping[str, Any],
    output_artifact_refs: Mapping[str, Any],
    known_at_ts: int | float | str,
    source_tail_ts: int | float | str,
    lineage_id: str,
    handoff_id: str | None = None,
    **kwargs: Any,
) -> RegimeForecasterHandoffManifest:
    payload = {
        "pathway": pathway,
        "artifact_kind": artifact_kind,
        "band": band,
        "interval": int(interval),
        "source_artifact_refs": to_jsonable(dict(source_artifact_refs)),
        "output_artifact_refs": to_jsonable(dict(output_artifact_refs)),
        "known_at_ts": known_at_ts,
        "source_tail_ts": source_tail_ts,
        "lineage_id": lineage_id,
    }
    resolved_id = handoff_id or "regime_forecaster_handoff_" + hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    return RegimeForecasterHandoffManifest(
        handoff_id=resolved_id,
        pathway=pathway,
        artifact_kind=artifact_kind,
        band=band,
        interval=interval,
        source_artifact_refs=source_artifact_refs,
        output_artifact_refs=output_artifact_refs,
        known_at_ts=known_at_ts,
        source_tail_ts=source_tail_ts,
        lineage_id=lineage_id,
        **kwargs,
    )


def validate_regime_forecaster_handoff_manifest(
    manifest: RegimeForecasterHandoffManifest | Mapping[str, Any],
    *,
    artifact_root: str | Path | None = None,
    report_root: str | Path | None = None,
    write_outputs: bool = False,
) -> RegimeForecasterHandoffManifest:
    resolved = manifest if isinstance(manifest, RegimeForecasterHandoffManifest) else RegimeForecasterHandoffManifest(**dict(manifest))
    for refs in (resolved.source_artifact_refs, resolved.output_artifact_refs):
        for ref in refs.values():
            validate_portable_artifact_ref(ref)
            if write_outputs:
                resolve_artifact_ref(ref, artifact_root=artifact_root, report_root=report_root, must_exist=True)
    if resolved.production_enabled is not False or resolved.production_outputs_written is not False:
        raise ValueError("Regime forecaster handoff production flags must be false")
    _artifact_boundary(resolved.artifact_boundary)
    return resolved


def write_regime_forecaster_handoff_manifest(
    manifest: RegimeForecasterHandoffManifest | Mapping[str, Any],
    *,
    output_root: str | Path,
    relative_path: str | Path,
    artifact_root: str | Path | None = None,
    report_root: str | Path | None = None,
    write_outputs: bool = True,
) -> Path:
    resolved = validate_regime_forecaster_handoff_manifest(
        manifest,
        artifact_root=artifact_root or output_root,
        report_root=report_root,
        write_outputs=write_outputs,
    )
    root = Path(output_root)
    rel = Path(relative_path)
    if rel.is_absolute() or any(part in {"", ".."} for part in rel.parts):
        raise ValueError("Regime forecaster handoff relative_path must stay within output root")
    path = (root / rel).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError("Regime forecaster handoff writer refusing to write outside output root") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(resolved.as_dict()), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def artifact_ref_for_handoff(
    path: str | Path,
    *,
    artifact_kind: str,
    artifact_root: str | Path,
    producer: str,
    known_at_ts: int | float | str | None = None,
    source_tail_ts: int | float | str | None = None,
) -> dict[str, Any]:
    return make_artifact_ref(
        path,
        artifact_kind=artifact_kind,
        artifact_root=artifact_root,
        producer=producer,
        known_at_ts=known_at_ts,
        source_tail_ts=source_tail_ts,
    ).as_dict()


def _refs(refs: Mapping[str, Any], *, field_name: str) -> dict[str, dict[str, Any]]:
    if not isinstance(refs, Mapping) or not refs:
        raise ValueError(f"Regime forecaster handoff {field_name} must be a non-empty mapping")
    return {str(name): portable_ref_dict(ref) for name, ref in refs.items()}


def _artifact_boundary(boundary: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(boundary, Mapping):
        raise ValueError("Regime forecaster handoff artifact_boundary must be a mapping")
    merged = default_forecaster_handoff_boundary()
    merged.update(dict(boundary))
    for key, value in merged.items():
        lower = str(key).lower()
        if ("production" in lower or lower in {"promoted", "promotion_allowed"}) and bool(value):
            raise ValueError(f"Regime forecaster handoff artifact_boundary cannot set {key}=true")
    if bool(merged.get("forecaster_implemented", False)):
        raise ValueError("Regime forecaster handoff must not mark forecaster_implemented true")
    return to_jsonable(merged)


def _text(value: object, *, field_name: str) -> str:
    text = str(value).strip()
    if not text or text == "None":
        raise ValueError(f"Regime forecaster handoff {field_name} must be non-empty")
    return text


def _optional_text(value: object | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _positive_int(value: object, *, field_name: str) -> int:
    try:
        out = int(value)
    except Exception as exc:
        raise ValueError(f"Regime forecaster handoff {field_name} must be an integer") from exc
    if out <= 0:
        raise ValueError(f"Regime forecaster handoff {field_name} must be positive")
    return out


def _to_orderable(value: object, *, field_name: str) -> float:
    if value is None or isinstance(value, bool):
        raise ValueError(f"Regime forecaster handoff {field_name} must be timestamp-compatible")
    try:
        return float(value)
    except Exception as exc:
        raise ValueError(f"Regime forecaster handoff {field_name} must be numeric") from exc


__all__ = [
    "ARTIFACT_KIND_ASSET_STATE_FEATURE_MATRIX_OR_PANEL",
    "ARTIFACT_KIND_ASSET_STATE_SANDBOX_LABELS",
    "ARTIFACT_KIND_CROSS_ASSET_FEATURE_ROWS",
    "ARTIFACT_KIND_MARKET_STATE_AXIS_PANEL",
    "ARTIFACT_KIND_MARKET_STATE_FEATURE_PANEL",
    "ARTIFACT_KIND_RELATIONSHIP_DISCOVERY_ARTIFACTS",
    "PATHWAY_ASSET_STATE",
    "PATHWAY_CROSS_ASSET",
    "PATHWAY_MARKET_STATE",
    "REGIME_FORECASTER_HANDOFF_ARTIFACT_KIND",
    "REGIME_FORECASTER_HANDOFF_PAYLOAD_KINDS",
    "REGIME_FORECASTER_HANDOFF_PATHWAYS",
    "REGIME_FORECASTER_HANDOFF_SCHEMA_VERSION",
    "RegimeForecasterHandoffManifest",
    "artifact_ref_for_handoff",
    "default_forecaster_handoff_boundary",
    "make_regime_forecaster_handoff_manifest",
    "validate_regime_forecaster_handoff_manifest",
    "write_regime_forecaster_handoff_manifest",
]
