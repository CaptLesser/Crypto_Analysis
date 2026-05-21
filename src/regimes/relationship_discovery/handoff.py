from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.regimes.core.artifact_refs import (
    make_artifact_ref,
    resolve_artifact_ref,
    refs_by_artifact_id,
    validate_portable_artifact_ref,
    validate_portable_relative_path,
)
from src.regimes.core.contracts import require_schema_version
from src.regimes.core.serialization import to_jsonable
from src.regimes.relationship_discovery.schemas import (
    ASSET_RELATIONSHIP_PROFILES_SCHEMA_ID,
    EDGE_ALIAS_MANIFEST_SCHEMA_ID,
    ISOLATED_ASSET_PROFILES_SCHEMA_ID,
    METHOD_MANIFEST_SCHEMA_ID,
    REFIT_SNAPSHOT_MANIFEST_SCHEMA_ID,
    RELATIONSHIP_STABILITY_SCORES_SCHEMA_ID,
    SELECTED_RELATIONSHIP_EDGES_SCHEMA_ID,
)
from src.regimes.relationship_discovery.writer import RelationshipDiscoveryV1ArtifactWriteResult


PROCESS1_TO_PROCESS2_HANDOFF_SCHEMA_VERSION = 1
PROCESS1_TO_PROCESS2_HANDOFF_ARTIFACT_KIND = "process1_to_process2_handoff_manifest"

_REQUIRED_ARTIFACT_SCHEMA_IDS: tuple[str, ...] = (
    METHOD_MANIFEST_SCHEMA_ID,
    REFIT_SNAPSHOT_MANIFEST_SCHEMA_ID,
    SELECTED_RELATIONSHIP_EDGES_SCHEMA_ID,
    ASSET_RELATIONSHIP_PROFILES_SCHEMA_ID,
    RELATIONSHIP_STABILITY_SCORES_SCHEMA_ID,
    ISOLATED_ASSET_PROFILES_SCHEMA_ID,
    EDGE_ALIAS_MANIFEST_SCHEMA_ID,
)


@dataclass(frozen=True)
class RelationshipProcessHandoffManifest:
    handoff_id: str
    run_id: str
    refit_key: str
    interval: int
    window: int
    method_manifest_path: str
    method_manifest_hash: str
    refit_snapshot_manifest_path: str
    refit_snapshot_manifest_hash: str
    selected_relationship_edges_path: str
    selected_relationship_edges_hash: str
    asset_relationship_profiles_path: str
    asset_relationship_profiles_hash: str
    relationship_stability_scores_path: str
    relationship_stability_scores_hash: str
    isolated_asset_profiles_path: str
    isolated_asset_profiles_hash: str
    edge_alias_manifest_path: str
    edge_alias_manifest_hash: str
    eligible_feature_families: Sequence[str]
    known_at_ts: int | float | str
    source_tail_ts: int | float | str
    relationship_scoreboard_path: str | None = None
    relationship_scoreboard_hash: str | None = None
    schema_version: int = PROCESS1_TO_PROCESS2_HANDOFF_SCHEMA_VERSION
    production_enabled: bool = False
    artifact_kind: str = PROCESS1_TO_PROCESS2_HANDOFF_ARTIFACT_KIND
    diagnostics: Mapping[str, Any] = field(default_factory=dict)
    artifact_refs: Mapping[str, Any] = field(default_factory=dict)
    artifact_root: Path | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.production_enabled is not False:
            raise ValueError("Relationship Discovery handoff production_enabled must be false")
        object.__setattr__(self, "schema_version", require_schema_version(self.schema_version))
        object.__setattr__(self, "artifact_kind", _text(self.artifact_kind, field_name="artifact_kind"))
        object.__setattr__(self, "handoff_id", _text(self.handoff_id, field_name="handoff_id"))
        object.__setattr__(self, "run_id", _text(self.run_id, field_name="run_id"))
        object.__setattr__(self, "refit_key", _text(self.refit_key, field_name="refit_key"))
        object.__setattr__(self, "interval", _positive_int(self.interval, field_name="interval"))
        object.__setattr__(self, "window", _positive_int(self.window, field_name="window"))
        if _to_orderable(self.source_tail_ts, field_name="source_tail_ts") > _to_orderable(self.known_at_ts, field_name="known_at_ts"):
            raise ValueError("Relationship Discovery handoff source_tail_ts must not exceed known_at_ts")
        families = tuple(dict.fromkeys(_text(value, field_name="eligible_feature_family") for value in self.eligible_feature_families))
        if not families:
            raise ValueError("Relationship Discovery handoff requires eligible_feature_families")
        object.__setattr__(self, "eligible_feature_families", families)
        for field_name in (
            "method_manifest_path",
            "method_manifest_hash",
            "refit_snapshot_manifest_path",
            "refit_snapshot_manifest_hash",
            "selected_relationship_edges_path",
            "selected_relationship_edges_hash",
            "asset_relationship_profiles_path",
            "asset_relationship_profiles_hash",
            "relationship_stability_scores_path",
            "relationship_stability_scores_hash",
            "isolated_asset_profiles_path",
            "isolated_asset_profiles_hash",
            "edge_alias_manifest_path",
            "edge_alias_manifest_hash",
        ):
            object.__setattr__(self, field_name, _text(getattr(self, field_name), field_name=field_name))
        object.__setattr__(self, "relationship_scoreboard_path", _optional_text(self.relationship_scoreboard_path))
        object.__setattr__(self, "relationship_scoreboard_hash", _optional_text(self.relationship_scoreboard_hash))
        object.__setattr__(self, "artifact_refs", refs_by_artifact_id(dict(self.artifact_refs)) if self.artifact_refs else {})
        object.__setattr__(self, "artifact_root", None if self.artifact_root is None else Path(self.artifact_root))
        object.__setattr__(self, "diagnostics", to_jsonable(dict(self.diagnostics)))

    def as_dict(self) -> dict[str, Any]:
        return {
            "artifact_kind": self.artifact_kind,
            "handoff_id": self.handoff_id,
            "run_id": self.run_id,
            "refit_key": self.refit_key,
            "interval": int(self.interval),
            "window": int(self.window),
            "method_manifest_path": self.method_manifest_path,
            "method_manifest_hash": self.method_manifest_hash,
            "refit_snapshot_manifest_path": self.refit_snapshot_manifest_path,
            "refit_snapshot_manifest_hash": self.refit_snapshot_manifest_hash,
            "selected_relationship_edges_path": self.selected_relationship_edges_path,
            "selected_relationship_edges_hash": self.selected_relationship_edges_hash,
            "asset_relationship_profiles_path": self.asset_relationship_profiles_path,
            "asset_relationship_profiles_hash": self.asset_relationship_profiles_hash,
            "relationship_stability_scores_path": self.relationship_stability_scores_path,
            "relationship_stability_scores_hash": self.relationship_stability_scores_hash,
            "isolated_asset_profiles_path": self.isolated_asset_profiles_path,
            "isolated_asset_profiles_hash": self.isolated_asset_profiles_hash,
            "edge_alias_manifest_path": self.edge_alias_manifest_path,
            "edge_alias_manifest_hash": self.edge_alias_manifest_hash,
            "relationship_scoreboard_path": self.relationship_scoreboard_path,
            "relationship_scoreboard_hash": self.relationship_scoreboard_hash,
            "eligible_feature_families": list(self.eligible_feature_families),
            "known_at_ts": self.known_at_ts,
            "source_tail_ts": self.source_tail_ts,
            "schema_version": int(self.schema_version),
            "production_enabled": False,
            "production_outputs_written": False,
            "broad_all_to_all": False,
            "cross_asset_labels_written": False,
            "artifact_refs": to_jsonable(dict(self.artifact_refs)),
            "diagnostics": to_jsonable(dict(self.diagnostics)),
        }


def build_process1_to_process2_handoff_manifest(
    write_result: RelationshipDiscoveryV1ArtifactWriteResult,
    *,
    run_id: str,
    relationship_scoreboard_path: str | Path | None = None,
    write_outputs: bool = True,
) -> RelationshipProcessHandoffManifest:
    if write_outputs:
        validate_handoff_artifact_references(write_result, relationship_scoreboard_path=relationship_scoreboard_path)
    anchor_path = _required_path(write_result, EDGE_ALIAS_MANIFEST_SCHEMA_ID)
    refit_key = _partition_value(anchor_path, "refit_key")
    interval = int(_partition_value(anchor_path, "interval"))
    window = int(_partition_value(anchor_path, "window"))
    selected_path = _required_path(write_result, SELECTED_RELATIONSHIP_EDGES_SCHEMA_ID, refit_key=refit_key, interval=interval, window=window)
    snapshot_path = _required_path(write_result, REFIT_SNAPSHOT_MANIFEST_SCHEMA_ID, refit_key=refit_key, interval=interval)
    snapshot = _load_snapshot(snapshot_path)
    known_at_ts = snapshot.get("known_at_ts")
    source_tail_ts = snapshot.get("source_tail_ts")
    artifact_root = Path(write_result.output_root).resolve()
    refs = _artifact_refs_for_handoff(
        write_result,
        artifact_root=artifact_root,
        relationship_scoreboard_path=relationship_scoreboard_path,
        refit_key=refit_key,
        interval=interval,
        window=window,
        known_at_ts=known_at_ts,
        source_tail_ts=source_tail_ts,
    )
    handoff_payload = {
        "run_id": run_id,
        "refit_key": refit_key,
        "interval": interval,
        "window": window,
        "artifact_refs": refs,
    }
    scoreboard_path = Path(relationship_scoreboard_path) if relationship_scoreboard_path is not None else None
    return RelationshipProcessHandoffManifest(
        handoff_id=hashlib.sha256(json.dumps(handoff_payload, sort_keys=True).encode("utf-8")).hexdigest()[:16],
        run_id=run_id,
        refit_key=refit_key,
        interval=interval,
        window=window,
        method_manifest_path=_relative_artifact_path(_required_path(write_result, METHOD_MANIFEST_SCHEMA_ID), artifact_root),
        method_manifest_hash=_sha256(_required_path(write_result, METHOD_MANIFEST_SCHEMA_ID)),
        refit_snapshot_manifest_path=_relative_artifact_path(snapshot_path, artifact_root),
        refit_snapshot_manifest_hash=_sha256(snapshot_path),
        selected_relationship_edges_path=_relative_artifact_path(selected_path, artifact_root),
        selected_relationship_edges_hash=_sha256(selected_path),
        asset_relationship_profiles_path=_relative_artifact_path(_required_path(write_result, ASSET_RELATIONSHIP_PROFILES_SCHEMA_ID, refit_key=refit_key, interval=interval, window=window), artifact_root),
        asset_relationship_profiles_hash=_sha256(_required_path(write_result, ASSET_RELATIONSHIP_PROFILES_SCHEMA_ID, refit_key=refit_key, interval=interval, window=window)),
        relationship_stability_scores_path=_relative_artifact_path(_required_path(write_result, RELATIONSHIP_STABILITY_SCORES_SCHEMA_ID, refit_key=refit_key, interval=interval, window=window), artifact_root),
        relationship_stability_scores_hash=_sha256(_required_path(write_result, RELATIONSHIP_STABILITY_SCORES_SCHEMA_ID, refit_key=refit_key, interval=interval, window=window)),
        isolated_asset_profiles_path=_relative_artifact_path(_required_path(write_result, ISOLATED_ASSET_PROFILES_SCHEMA_ID, refit_key=refit_key, interval=interval, window=window), artifact_root),
        isolated_asset_profiles_hash=_sha256(_required_path(write_result, ISOLATED_ASSET_PROFILES_SCHEMA_ID, refit_key=refit_key, interval=interval, window=window)),
        edge_alias_manifest_path=_relative_artifact_path(_required_path(write_result, EDGE_ALIAS_MANIFEST_SCHEMA_ID, refit_key=refit_key, interval=interval, window=window), artifact_root),
        edge_alias_manifest_hash=_sha256(_required_path(write_result, EDGE_ALIAS_MANIFEST_SCHEMA_ID, refit_key=refit_key, interval=interval, window=window)),
        relationship_scoreboard_path=_relative_artifact_path(scoreboard_path, artifact_root) if scoreboard_path is not None else None,
        relationship_scoreboard_hash=_sha256(scoreboard_path) if scoreboard_path is not None else None,
        eligible_feature_families=("market_exposure", "residual_peer", "isolation_status"),
        known_at_ts=known_at_ts,
        source_tail_ts=source_tail_ts,
        artifact_refs=refs,
        artifact_root=artifact_root,
        diagnostics={
            "single_process2_entrypoint": True,
            "required_artifacts_validated": bool(write_outputs),
            "path_reference_standard": "artifact_refs_v1",
        },
    )


def validate_process1_to_process2_handoff_manifest(
    manifest: RelationshipProcessHandoffManifest | Mapping[str, Any],
    *,
    artifact_root: str | Path | None = None,
) -> None:
    payload = manifest.as_dict() if isinstance(manifest, RelationshipProcessHandoffManifest) else dict(manifest)
    runtime_root = Path(artifact_root) if artifact_root is not None else (manifest.artifact_root if isinstance(manifest, RelationshipProcessHandoffManifest) else None)
    if bool(payload.get("production_enabled", False)):
        raise ValueError("Relationship Discovery handoff production_enabled must be false")
    if bool(payload.get("production_outputs_written", False)):
        raise ValueError("Relationship Discovery handoff production_outputs_written must be false")
    if bool(payload.get("broad_all_to_all", False)):
        raise ValueError("Relationship Discovery handoff broad_all_to_all must be false")
    if bool(payload.get("cross_asset_labels_written", False)):
        raise ValueError("Relationship Discovery handoff cross_asset_labels_written must be false")
    if _to_orderable(payload.get("source_tail_ts"), field_name="source_tail_ts") > _to_orderable(payload.get("known_at_ts"), field_name="known_at_ts"):
        raise ValueError("Relationship Discovery handoff source_tail_ts must not exceed known_at_ts")
    for field_name in (
        "method_manifest_path",
        "refit_snapshot_manifest_path",
        "selected_relationship_edges_path",
        "asset_relationship_profiles_path",
        "relationship_stability_scores_path",
        "isolated_asset_profiles_path",
        "edge_alias_manifest_path",
    ):
        path = _resolve_handoff_path(payload, field_name, artifact_root=runtime_root)
        if not path.exists() or not path.is_file():
            raise ValueError(f"Relationship Discovery handoff referenced artifact missing: {field_name}")
    scoreboard = _optional_text(payload.get("relationship_scoreboard_path"))
    if scoreboard is not None:
        scoreboard_path = _resolve_handoff_path(payload, "relationship_scoreboard_path", artifact_root=runtime_root)
        if not scoreboard_path.is_file():
            raise ValueError("Relationship Discovery handoff referenced artifact missing: relationship_scoreboard_path")
    for ref in dict(payload.get("artifact_refs") or {}).values():
        validate_portable_artifact_ref(ref)


def validate_handoff_artifact_references(
    write_result: RelationshipDiscoveryV1ArtifactWriteResult,
    *,
    relationship_scoreboard_path: str | Path | None = None,
) -> None:
    by_schema = {write.schema_id: [] for write in write_result.writes}
    for write in write_result.writes:
        by_schema.setdefault(write.schema_id, []).append(write.path)
        if not write.path.exists() or not write.path.is_file():
            raise ValueError(f"Relationship Discovery handoff artifact is missing: {write.schema_id}")
    missing = [schema_id for schema_id in _REQUIRED_ARTIFACT_SCHEMA_IDS if schema_id not in by_schema]
    if missing:
        raise ValueError(f"Relationship Discovery handoff missing required artifacts: {missing}")
    if relationship_scoreboard_path is not None:
        path = Path(relationship_scoreboard_path)
        if not path.exists() or not path.is_file():
            raise ValueError("Relationship Discovery handoff relationship_scoreboard_path is missing")


def write_process1_to_process2_handoff_manifest(
    manifest: RelationshipProcessHandoffManifest,
    *,
    output_root: str | Path,
    relative_path: str | Path = "process1_to_process2_handoff_manifest.json",
) -> Path:
    root = Path(output_root)
    validate_process1_to_process2_handoff_manifest(manifest, artifact_root=root)
    rel = Path(relative_path)
    if rel.is_absolute() or any(part in {"", ".."} for part in rel.parts):
        raise ValueError("Relationship Discovery handoff relative_path must stay within output root")
    path = root / rel
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError("Relationship Discovery handoff writer refusing to write outside output root") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(manifest.as_dict()), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _artifact_refs_for_handoff(
    write_result: RelationshipDiscoveryV1ArtifactWriteResult,
    *,
    artifact_root: Path,
    relationship_scoreboard_path: str | Path | None,
    refit_key: str,
    interval: int,
    window: int,
    known_at_ts: int | float | str,
    source_tail_ts: int | float | str,
) -> dict[str, dict[str, Any]]:
    selected_paths = {
        METHOD_MANIFEST_SCHEMA_ID: _required_path(write_result, METHOD_MANIFEST_SCHEMA_ID),
        REFIT_SNAPSHOT_MANIFEST_SCHEMA_ID: _required_path(write_result, REFIT_SNAPSHOT_MANIFEST_SCHEMA_ID, refit_key=refit_key, interval=interval),
        SELECTED_RELATIONSHIP_EDGES_SCHEMA_ID: _required_path(write_result, SELECTED_RELATIONSHIP_EDGES_SCHEMA_ID, refit_key=refit_key, interval=interval, window=window),
        ASSET_RELATIONSHIP_PROFILES_SCHEMA_ID: _required_path(write_result, ASSET_RELATIONSHIP_PROFILES_SCHEMA_ID, refit_key=refit_key, interval=interval, window=window),
        RELATIONSHIP_STABILITY_SCORES_SCHEMA_ID: _required_path(write_result, RELATIONSHIP_STABILITY_SCORES_SCHEMA_ID, refit_key=refit_key, interval=interval, window=window),
        ISOLATED_ASSET_PROFILES_SCHEMA_ID: _required_path(write_result, ISOLATED_ASSET_PROFILES_SCHEMA_ID, refit_key=refit_key, interval=interval, window=window),
        EDGE_ALIAS_MANIFEST_SCHEMA_ID: _required_path(write_result, EDGE_ALIAS_MANIFEST_SCHEMA_ID, refit_key=refit_key, interval=interval, window=window),
    }
    refs = {
        schema_id: make_artifact_ref(
            path,
            artifact_kind=schema_id,
            artifact_root=artifact_root,
            producer="src.regimes.relationship_discovery.v1_runner",
            known_at_ts=known_at_ts,
            source_tail_ts=source_tail_ts,
        ).as_dict()
        for schema_id, path in selected_paths.items()
    }
    if relationship_scoreboard_path is not None:
        refs["relationship_scoreboard"] = make_artifact_ref(
            relationship_scoreboard_path,
            artifact_kind="relationship_scoreboard",
            artifact_root=artifact_root,
            producer="src.regimes.relationship_discovery.v1_runner",
            known_at_ts=known_at_ts,
            source_tail_ts=source_tail_ts,
        ).as_dict()
    return refs


def _relative_artifact_path(path: Path, artifact_root: Path) -> str:
    return validate_portable_relative_path(path.resolve().relative_to(artifact_root.resolve()).as_posix())


def _resolve_handoff_path(payload: Mapping[str, Any], field_name: str, *, artifact_root: str | Path | None) -> Path:
    ref_key = _ref_key_for_path_field(field_name)
    refs = payload.get("artifact_refs")
    if isinstance(refs, Mapping) and ref_key in refs:
        if artifact_root is None:
            raise ValueError("Relationship Discovery handoff artifact_root is required for portable artifact refs")
        return resolve_artifact_ref(refs[ref_key], artifact_root=artifact_root, must_exist=False)
    raw = _text(payload.get(field_name), field_name=field_name)
    path = Path(raw)
    if not path.is_absolute() and artifact_root is not None:
        path = Path(artifact_root) / validate_portable_relative_path(raw, field_name=field_name)
    return path.resolve()


def _ref_key_for_path_field(field_name: str) -> str:
    return {
        "method_manifest_path": METHOD_MANIFEST_SCHEMA_ID,
        "refit_snapshot_manifest_path": REFIT_SNAPSHOT_MANIFEST_SCHEMA_ID,
        "selected_relationship_edges_path": SELECTED_RELATIONSHIP_EDGES_SCHEMA_ID,
        "asset_relationship_profiles_path": ASSET_RELATIONSHIP_PROFILES_SCHEMA_ID,
        "relationship_stability_scores_path": RELATIONSHIP_STABILITY_SCORES_SCHEMA_ID,
        "isolated_asset_profiles_path": ISOLATED_ASSET_PROFILES_SCHEMA_ID,
        "edge_alias_manifest_path": EDGE_ALIAS_MANIFEST_SCHEMA_ID,
        "relationship_scoreboard_path": "relationship_scoreboard",
    }.get(field_name, field_name)


def _required_path(
    write_result: RelationshipDiscoveryV1ArtifactWriteResult,
    schema_id: str,
    *,
    refit_key: str | None = None,
    interval: int | None = None,
    window: int | None = None,
) -> Path:
    candidates = [write.path for write in write_result.writes if write.schema_id == schema_id]
    if refit_key is not None:
        candidates = [path for path in candidates if _partition_value(path, "refit_key") == refit_key]
    if interval is not None:
        candidates = [path for path in candidates if _partition_value(path, "interval") == str(int(interval))]
    if window is not None:
        candidates = [path for path in candidates if _partition_value(path, "window") == str(int(window))]
    candidates = sorted(candidates, key=lambda path: str(path))
    if not candidates:
        raise ValueError(f"Relationship Discovery handoff missing required artifact: {schema_id}")
    path = candidates[0]
    if not path.exists() or not path.is_file():
        raise ValueError(f"Relationship Discovery handoff referenced artifact missing: {schema_id}")
    return path


def _partition_value(path: Path, name: str) -> str:
    prefix = f"{name}="
    for part in path.parts:
        text = str(part)
        if text.startswith(prefix):
            return text[len(prefix) :]
    if name == "window":
        return ""
    raise ValueError(f"Relationship Discovery handoff path missing partition {name!r}: {path}")


def _load_snapshot(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("Relationship Discovery handoff snapshot artifact must be a JSON object")
    if bool(payload.get("production_enabled", False)):
        raise ValueError("Relationship Discovery handoff snapshot production_enabled must be false")
    return payload


def _sha256(path: Path | None) -> str | None:
    if path is None:
        return None
    if not path.exists() or not path.is_file():
        raise ValueError(f"Relationship Discovery handoff cannot hash missing artifact: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _text(value: object, *, field_name: str) -> str:
    text = str(value).strip()
    if not text or text == "None":
        raise ValueError(f"Relationship Discovery handoff {field_name} must be non-empty")
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
        raise ValueError(f"Relationship Discovery handoff {field_name} must be an integer") from exc
    if out <= 0:
        raise ValueError(f"Relationship Discovery handoff {field_name} must be positive")
    return out


def _to_orderable(value: object, *, field_name: str) -> float:
    if value is None or isinstance(value, bool):
        raise ValueError(f"Relationship Discovery handoff {field_name} must be timestamp-compatible")
    try:
        return float(value)
    except Exception as exc:
        raise ValueError(f"Relationship Discovery handoff {field_name} must be numeric") from exc


__all__ = [
    "PROCESS1_TO_PROCESS2_HANDOFF_ARTIFACT_KIND",
    "PROCESS1_TO_PROCESS2_HANDOFF_SCHEMA_VERSION",
    "RelationshipProcessHandoffManifest",
    "build_process1_to_process2_handoff_manifest",
    "validate_handoff_artifact_references",
    "validate_process1_to_process2_handoff_manifest",
    "write_process1_to_process2_handoff_manifest",
]
