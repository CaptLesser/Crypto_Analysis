from __future__ import annotations

"""Unified Regime forecaster handoff index contracts.

Index-level ArtifactRefs point to handoff manifest metadata and may omit
known_at_ts/source_tail_ts. Row-producing handoff manifests remain the causal
timestamp authority and reject missing or out-of-order causal timestamps.
"""

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.regimes.core.artifact_refs import make_artifact_ref, portable_ref_dict, resolve_artifact_ref
from src.regimes.core.forecaster_handoff import (
    REGIME_FORECASTER_HANDOFF_SCHEMA_VERSION,
    RegimeForecasterHandoffManifest,
    validate_regime_forecaster_handoff_manifest,
)
from src.regimes.core.serialization import to_jsonable


REGIME_FORECASTER_HANDOFF_INDEX_ARTIFACT_KIND = "regime_forecaster_handoff_index"


@dataclass(frozen=True)
class RegimeForecasterHandoffIndex:
    index_id: str
    handoff_manifest_refs: Mapping[str, Any]
    handoff_manifests: Sequence[Mapping[str, Any]] = ()
    schema_version: int = REGIME_FORECASTER_HANDOFF_SCHEMA_VERSION
    artifact_kind: str = REGIME_FORECASTER_HANDOFF_INDEX_ARTIFACT_KIND
    production_enabled: bool = False
    production_outputs_written: bool = False
    consumer_notes: str | None = None

    def __post_init__(self) -> None:
        if self.production_enabled is not False:
            raise ValueError("Regime forecaster handoff index production_enabled must be false")
        if self.production_outputs_written is not False:
            raise ValueError("Regime forecaster handoff index production_outputs_written must be false")
        object.__setattr__(self, "schema_version", int(self.schema_version))
        if int(self.schema_version) != REGIME_FORECASTER_HANDOFF_SCHEMA_VERSION:
            raise ValueError("Regime forecaster handoff index schema_version is unsupported")
        if str(self.artifact_kind) != REGIME_FORECASTER_HANDOFF_INDEX_ARTIFACT_KIND:
            raise ValueError("Regime forecaster handoff index artifact_kind is invalid")
        refs = {str(name): portable_ref_dict(ref) for name, ref in dict(self.handoff_manifest_refs).items()}
        if not refs:
            raise ValueError("Regime forecaster handoff index requires handoff_manifest_refs")
        object.__setattr__(self, "handoff_manifest_refs", refs)
        manifests = tuple(dict(item) for item in self.handoff_manifests)
        for manifest in manifests:
            validate_regime_forecaster_handoff_manifest(manifest, write_outputs=False)
        object.__setattr__(self, "handoff_manifests", manifests)

    def as_dict(self) -> dict[str, Any]:
        return {
            "artifact_kind": self.artifact_kind,
            "schema_version": int(self.schema_version),
            "index_id": self.index_id,
            "handoff_manifest_refs": to_jsonable(dict(self.handoff_manifest_refs)),
            "handoff_manifests": to_jsonable(list(self.handoff_manifests)),
            "production_enabled": False,
            "production_outputs_written": False,
            "consumer_notes": self.consumer_notes,
        }


def build_regime_forecaster_handoff_index(
    manifest_paths: Sequence[str | Path],
    *,
    artifact_root: str | Path,
    manifests: Sequence[RegimeForecasterHandoffManifest | Mapping[str, Any]] = (),
    index_id: str | None = None,
    producer: str = "src.regimes.core.handoff_index",
    consumer_notes: str | None = None,
) -> RegimeForecasterHandoffIndex:
    refs: dict[str, dict[str, Any]] = {}
    for path in manifest_paths:
        ref = make_artifact_ref(
            path,
            artifact_kind="regime_forecaster_handoff_manifest",
            artifact_root=artifact_root,
            producer=producer,
        )
        refs[Path(path).stem] = ref.as_dict()
    payload = {"refs": refs, "manifests": [m.as_dict() if isinstance(m, RegimeForecasterHandoffManifest) else dict(m) for m in manifests]}
    resolved_id = index_id or "regime_forecaster_handoff_index_" + hashlib.sha256(
        json.dumps(to_jsonable(payload), sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    return RegimeForecasterHandoffIndex(
        index_id=resolved_id,
        handoff_manifest_refs=refs,
        handoff_manifests=tuple(payload["manifests"]),
        consumer_notes=consumer_notes,
    )


def validate_regime_forecaster_handoff_index(
    index: RegimeForecasterHandoffIndex | Mapping[str, Any],
    *,
    artifact_root: str | Path | None = None,
    report_root: str | Path | None = None,
    write_outputs: bool = False,
) -> RegimeForecasterHandoffIndex:
    resolved = index if isinstance(index, RegimeForecasterHandoffIndex) else RegimeForecasterHandoffIndex(**dict(index))
    for ref in resolved.handoff_manifest_refs.values():
        if write_outputs:
            resolve_artifact_ref(ref, artifact_root=artifact_root, report_root=report_root, must_exist=True)
    return resolved


def write_regime_forecaster_handoff_index(
    index: RegimeForecasterHandoffIndex | Mapping[str, Any],
    *,
    output_root: str | Path,
    relative_path: str | Path,
    artifact_root: str | Path | None = None,
    report_root: str | Path | None = None,
    write_outputs: bool = True,
) -> Path:
    resolved = validate_regime_forecaster_handoff_index(
        index,
        artifact_root=artifact_root or output_root,
        report_root=report_root,
        write_outputs=write_outputs,
    )
    root = Path(output_root)
    rel = Path(relative_path)
    if rel.is_absolute() or any(part in {"", ".."} for part in rel.parts):
        raise ValueError("Regime forecaster handoff index relative_path must stay within output root")
    path = (root / rel).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError("Regime forecaster handoff index writer refusing to write outside output root") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(resolved.as_dict()), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


__all__ = [
    "REGIME_FORECASTER_HANDOFF_INDEX_ARTIFACT_KIND",
    "RegimeForecasterHandoffIndex",
    "build_regime_forecaster_handoff_index",
    "validate_regime_forecaster_handoff_index",
    "write_regime_forecaster_handoff_index",
]
