from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.regimes.core.contracts import CANONICAL_SCHEMA_VERSION, require_json_mapping, require_non_empty_string, require_schema_version
from src.regimes.core.serialization import dumps_json, loads_json, require_json_object, to_jsonable


FEATURE_CACHE_SCHEMA_VERSION = CANONICAL_SCHEMA_VERSION
FEATURE_CACHE_MANIFEST_ARTIFACT_KIND = "regime_feature_cache_manifest"
FEATURE_CACHE_DECISION_ARTIFACT_KIND = "regime_feature_cache_decision"
FEATURE_CACHE_NOOP_WRITER_ARTIFACT_KIND = "regime_feature_cache_noop_writer"

CACHE_STATUS_VALID = "valid"
CACHE_STATUS_INVALID = "invalid"
CACHE_STATUSES: tuple[str, ...] = (CACHE_STATUS_VALID, CACHE_STATUS_INVALID)

CACHE_DECISION_REUSE = "reuse"
CACHE_DECISION_REBUILD = "rebuild"
CACHE_DECISION_INVALIDATE = "invalidate"
CACHE_DECISIONS: tuple[str, ...] = (
    CACHE_DECISION_REUSE,
    CACHE_DECISION_REBUILD,
    CACHE_DECISION_INVALIDATE,
)

FOUNDATION_FEATURE_CACHE_ROOT = Path("reports") / "regimes" / "foundation"


def _token(value: object, *, field_name: str) -> str:
    if isinstance(value, Enum):
        value = value.value
    return require_non_empty_string(value, field_name=field_name).lower()


def _string_tuple(values: Sequence[object], *, field_name: str, require_non_empty: bool = False) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"Regime feature cache {field_name} must be a sequence")
    out = tuple(str(value).strip() for value in values if str(value).strip())
    if require_non_empty and not out:
        raise ValueError(f"Regime feature cache {field_name} must include at least one value")
    return out


def _mapping_tuple(values: Sequence[Mapping[str, Any]], *, field_name: str) -> tuple[dict[str, Any], ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValueError(f"Regime feature cache {field_name} must be a sequence of JSON objects")
    out = tuple(require_json_object(value, context=f"Regime feature cache {field_name}") for value in values)
    if not out:
        raise ValueError(f"Regime feature cache {field_name} must include at least one value")
    return tuple(to_jsonable(value) for value in out)


def _normalized_window(window: Mapping[str, Any]) -> dict[str, Any]:
    obj = require_json_object(window, context="Regime feature cache train_window")
    if "start_ts" not in obj or "end_ts" not in obj:
        raise ValueError("Regime feature cache train_window requires start_ts and end_ts")
    return to_jsonable({"start_ts": obj["start_ts"], "end_ts": obj["end_ts"]})


def _canonical(value: Any) -> str:
    return dumps_json(value, indent=None, separators=(",", ":"), sort_keys=True)


def _signature(value: Any) -> str:
    digest = hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()
    return "sha256:" + digest


def _status(value: object) -> str:
    status = _token(value, field_name="cache status")
    if status not in CACHE_STATUSES:
        valid = ", ".join(CACHE_STATUSES)
        raise ValueError(f"Unsupported Regime feature cache status {status!r}; expected one of: {valid}")
    return status


def _decision_status(value: object) -> str:
    decision = _token(value, field_name="cache decision")
    if decision not in CACHE_DECISIONS:
        valid = ", ".join(CACHE_DECISIONS)
        raise ValueError(f"Unsupported Regime feature cache decision {decision!r}; expected one of: {valid}")
    return decision


def _resolve_root(path: str | Path) -> Path:
    root = Path(path).expanduser()
    if not root.is_absolute():
        root = Path.cwd() / root
    return root.resolve()


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        return path == root or path.is_relative_to(root)
    except AttributeError:  # pragma: no cover - compatibility only
        try:
            path.relative_to(root)
            return True
        except ValueError:
            return False


def validate_feature_cache_report_root(
    report_root: str | Path,
    *,
    project_root: str | Path | None = None,
) -> Path:
    root = _resolve_root(report_root)
    project = _resolve_root(project_root or Path.cwd())
    production_tokens = {"parquet", "regime_definitions", "model_states", "state"}
    if any(part.lower() in production_tokens for part in root.parts):
        raise ValueError("Regime feature cache report root is production-adjacent and is not allowed")
    for candidate in (
        project / "parquet",
        project / "regime_definitions",
        project / "model_states",
        project / "state",
    ):
        if _is_relative_to(root, _resolve_root(candidate)):
            raise ValueError("Regime feature cache report root is production-adjacent and is not allowed")
    normalized = tuple(part.lower() for part in root.parts)
    if len(normalized) < 3 or normalized[-3:] != ("reports", "regimes", "foundation"):
        raise ValueError("Regime feature cache report root must end with reports/regimes/foundation")
    return root


def _safe_child(root: Path, *parts: str) -> Path:
    candidate = root.joinpath(*parts).resolve()
    if not _is_relative_to(candidate, root):
        raise ValueError("Regime feature cache artifact path must stay under the report root")
    return candidate


@dataclass(frozen=True)
class FeatureCacheManifest:
    source_lineage: Sequence[Mapping[str, Any]]
    feature_family: str
    preprocessing_family: str
    train_window: Mapping[str, Any]
    cache_artifact_path: str | Path
    cache_id: str = "feature_cache"
    status: str = CACHE_STATUS_VALID
    invalidation_reasons: Sequence[str] = ()
    source_columns: Sequence[str] = ()
    selected_columns: Sequence[str] = ()
    shape_metadata: Mapping[str, Any] = field(default_factory=dict)
    preprocessing_metadata: Mapping[str, Any] = field(default_factory=dict)
    diagnostics: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = FEATURE_CACHE_SCHEMA_VERSION
    artifact_kind: str = FEATURE_CACHE_MANIFEST_ARTIFACT_KIND

    def __post_init__(self) -> None:
        schema_version = require_schema_version(self.schema_version)
        cache_id = require_non_empty_string(self.cache_id, field_name="cache_id")
        source_lineage = _mapping_tuple(self.source_lineage, field_name="source_lineage")
        feature_family = _token(self.feature_family, field_name="feature_family")
        preprocessing_family = _token(self.preprocessing_family, field_name="preprocessing_family")
        train_window = _normalized_window(self.train_window)
        status = _status(self.status)
        invalidation_reasons = _string_tuple(self.invalidation_reasons, field_name="invalidation_reasons")
        if status == CACHE_STATUS_INVALID and not invalidation_reasons:
            raise ValueError("Regime invalid feature cache manifest requires invalidation_reasons")
        cache_path = require_non_empty_string(self.cache_artifact_path, field_name="cache_artifact_path")
        object.__setattr__(self, "schema_version", schema_version)
        object.__setattr__(self, "cache_id", cache_id)
        object.__setattr__(self, "source_lineage", source_lineage)
        object.__setattr__(self, "feature_family", feature_family)
        object.__setattr__(self, "preprocessing_family", preprocessing_family)
        object.__setattr__(self, "train_window", train_window)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "invalidation_reasons", invalidation_reasons)
        object.__setattr__(self, "source_columns", _string_tuple(self.source_columns, field_name="source_columns"))
        object.__setattr__(self, "selected_columns", _string_tuple(self.selected_columns, field_name="selected_columns"))
        object.__setattr__(self, "cache_artifact_path", cache_path)
        object.__setattr__(self, "shape_metadata", require_json_mapping(self.shape_metadata, field_name="shape_metadata"))
        object.__setattr__(
            self,
            "preprocessing_metadata",
            require_json_mapping(self.preprocessing_metadata, field_name="preprocessing_metadata"),
        )
        object.__setattr__(self, "diagnostics", require_json_mapping(self.diagnostics, field_name="diagnostics"))

    @property
    def identity(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "source_lineage": list(self.source_lineage),
            "feature_family": self.feature_family,
            "preprocessing_family": self.preprocessing_family,
            "train_window": dict(self.train_window),
        }

    @property
    def source_lineage_signature(self) -> str:
        return _signature(list(self.source_lineage))

    @property
    def cache_key(self) -> str:
        return _signature(self.identity)

    @property
    def artifact_boundary(self) -> dict[str, Any]:
        return {
            "manifest_only": True,
            "sandbox_only": True,
            "cache_matrix_write_enabled": False,
            "production_cache_write_enabled": False,
            "production_writes_enabled": False,
            "production_outputs_written": False,
            "parquet_writes_enabled": False,
            "distributed_cache_coordination": False,
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "artifact_kind": self.artifact_kind,
            "cache_id": self.cache_id,
            "status": self.status,
            "cache_key": self.cache_key,
            "identity": self.identity,
            "source_lineage": list(self.source_lineage),
            "source_lineage_signature": self.source_lineage_signature,
            "feature_family": self.feature_family,
            "preprocessing_family": self.preprocessing_family,
            "train_window": dict(self.train_window),
            "source_columns": list(self.source_columns),
            "selected_columns": list(self.selected_columns),
            "shape_metadata": to_jsonable(self.shape_metadata),
            "preprocessing_metadata": to_jsonable(self.preprocessing_metadata),
            "cache_artifact_path": str(self.cache_artifact_path),
            "invalidation_reasons": list(self.invalidation_reasons),
            "diagnostics": to_jsonable(self.diagnostics),
            "artifact_boundary": self.artifact_boundary,
        }

    def to_json(self, **kwargs: Any) -> str:
        return dumps_json(self.as_dict(), **kwargs)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FeatureCacheManifest":
        obj = require_json_object(payload, context="Regime FeatureCacheManifest")
        return cls(
            schema_version=obj.get("schema_version", FEATURE_CACHE_SCHEMA_VERSION),
            artifact_kind=obj.get("artifact_kind", FEATURE_CACHE_MANIFEST_ARTIFACT_KIND),
            cache_id=obj.get("cache_id", "feature_cache"),
            status=obj.get("status", CACHE_STATUS_VALID),
            source_lineage=obj["source_lineage"],
            feature_family=obj["feature_family"],
            preprocessing_family=obj["preprocessing_family"],
            train_window=obj["train_window"],
            source_columns=obj.get("source_columns", ()),
            selected_columns=obj.get("selected_columns", ()),
            shape_metadata=obj.get("shape_metadata", {}),
            preprocessing_metadata=obj.get("preprocessing_metadata", {}),
            cache_artifact_path=obj["cache_artifact_path"],
            invalidation_reasons=obj.get("invalidation_reasons", ()),
            diagnostics=obj.get("diagnostics", {}),
        )

    @classmethod
    def from_json(cls, text: str) -> "FeatureCacheManifest":
        return cls.from_dict(require_json_object(loads_json(text), context="Regime FeatureCacheManifest JSON"))


@dataclass(frozen=True)
class FeatureCacheDecision:
    decision: str
    reasons: Sequence[str]
    cache_key: str | None = None
    requested_cache_key: str | None = None
    diagnostics: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = FEATURE_CACHE_SCHEMA_VERSION
    artifact_kind: str = FEATURE_CACHE_DECISION_ARTIFACT_KIND

    def __post_init__(self) -> None:
        schema_version = require_schema_version(self.schema_version)
        decision = _decision_status(self.decision)
        reasons = _string_tuple(self.reasons, field_name="decision reasons")
        if decision != CACHE_DECISION_REUSE and not reasons:
            raise ValueError("Regime feature cache rebuild/invalidate decisions require explicit reasons")
        if decision == CACHE_DECISION_REUSE and reasons:
            raise ValueError("Regime feature cache reuse decision cannot carry blocking reasons")
        object.__setattr__(self, "schema_version", schema_version)
        object.__setattr__(self, "decision", decision)
        object.__setattr__(self, "reasons", reasons)
        object.__setattr__(self, "cache_key", None if self.cache_key is None else str(self.cache_key))
        object.__setattr__(
            self,
            "requested_cache_key",
            None if self.requested_cache_key is None else str(self.requested_cache_key),
        )
        object.__setattr__(self, "diagnostics", require_json_mapping(self.diagnostics, field_name="diagnostics"))

    @property
    def can_reuse(self) -> bool:
        return self.decision == CACHE_DECISION_REUSE

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "artifact_kind": self.artifact_kind,
            "decision": self.decision,
            "can_reuse": bool(self.can_reuse),
            "reasons": list(self.reasons),
            "cache_key": self.cache_key,
            "requested_cache_key": self.requested_cache_key,
            "diagnostics": to_jsonable(self.diagnostics),
        }

    def to_json(self, **kwargs: Any) -> str:
        return dumps_json(self.as_dict(), **kwargs)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FeatureCacheDecision":
        obj = require_json_object(payload, context="Regime FeatureCacheDecision")
        return cls(
            schema_version=obj.get("schema_version", FEATURE_CACHE_SCHEMA_VERSION),
            artifact_kind=obj.get("artifact_kind", FEATURE_CACHE_DECISION_ARTIFACT_KIND),
            decision=obj["decision"],
            reasons=obj.get("reasons", ()),
            cache_key=obj.get("cache_key"),
            requested_cache_key=obj.get("requested_cache_key"),
            diagnostics=obj.get("diagnostics", {}),
        )

    @classmethod
    def from_json(cls, text: str) -> "FeatureCacheDecision":
        return cls.from_dict(require_json_object(loads_json(text), context="Regime FeatureCacheDecision JSON"))


def build_feature_cache_manifest(
    *,
    source_lineage: Sequence[Mapping[str, Any]],
    feature_family: str,
    preprocessing_family: str,
    train_window: Mapping[str, Any],
    cache_artifact_path: str | Path,
    cache_id: str = "feature_cache",
    source_columns: Sequence[str] = (),
    selected_columns: Sequence[str] = (),
    shape_metadata: Mapping[str, Any] | None = None,
    preprocessing_metadata: Mapping[str, Any] | None = None,
    diagnostics: Mapping[str, Any] | None = None,
) -> FeatureCacheManifest:
    return FeatureCacheManifest(
        cache_id=cache_id,
        source_lineage=source_lineage,
        feature_family=feature_family,
        preprocessing_family=preprocessing_family,
        train_window=train_window,
        source_columns=source_columns,
        selected_columns=selected_columns,
        shape_metadata=shape_metadata or {},
        preprocessing_metadata=preprocessing_metadata or {},
        cache_artifact_path=cache_artifact_path,
        diagnostics=diagnostics or {},
    )


def _requested_identity(
    *,
    source_lineage: Sequence[Mapping[str, Any]],
    feature_family: str,
    preprocessing_family: str,
    train_window: Mapping[str, Any],
    schema_version: int = FEATURE_CACHE_SCHEMA_VERSION,
) -> dict[str, Any]:
    return {
        "schema_version": int(schema_version),
        "source_lineage": list(_mapping_tuple(source_lineage, field_name="source_lineage")),
        "feature_family": _token(feature_family, field_name="feature_family"),
        "preprocessing_family": _token(preprocessing_family, field_name="preprocessing_family"),
        "train_window": _normalized_window(train_window),
    }


def can_reuse_cache(
    manifest: FeatureCacheManifest | Mapping[str, Any],
    *,
    source_lineage: Sequence[Mapping[str, Any]],
    feature_family: str,
    preprocessing_family: str,
    train_window: Mapping[str, Any],
    schema_version: int = FEATURE_CACHE_SCHEMA_VERSION,
) -> FeatureCacheDecision:
    current = manifest if isinstance(manifest, FeatureCacheManifest) else FeatureCacheManifest.from_dict(manifest)
    requested_identity = _requested_identity(
        schema_version=schema_version,
        source_lineage=source_lineage,
        feature_family=feature_family,
        preprocessing_family=preprocessing_family,
        train_window=train_window,
    )
    requested_lineage_signature = _signature(requested_identity["source_lineage"])
    requested_cache_key = _signature(requested_identity)
    invalidation_reasons: list[str] = []
    rebuild_reasons: list[str] = []
    diagnostics = {
        "checked_fields": [
            "schema_version",
            "source_lineage",
            "feature_family",
            "preprocessing_family",
            "train_window",
            "manifest_status",
            "artifact_boundary",
        ],
        "current_identity": current.identity,
        "requested_identity": requested_identity,
    }
    if current.status == CACHE_STATUS_INVALID:
        invalidation_reasons.append("manifest_status_invalid")
    if current.invalidation_reasons:
        invalidation_reasons.extend(f"invalidated:{reason}" for reason in current.invalidation_reasons)
    if current.artifact_boundary.get("production_outputs_written") is not False:
        invalidation_reasons.append("production_output_claim")
    if current.artifact_boundary.get("production_cache_write_enabled") is not False:
        invalidation_reasons.append("production_cache_write_enabled")
    if int(current.schema_version) != int(schema_version):
        rebuild_reasons.append("schema_version_mismatch")
    if current.source_lineage_signature != requested_lineage_signature:
        rebuild_reasons.append("source_lineage_mismatch")
    if current.feature_family != requested_identity["feature_family"]:
        rebuild_reasons.append("feature_family_mismatch")
    if current.preprocessing_family != requested_identity["preprocessing_family"]:
        rebuild_reasons.append("preprocessing_family_mismatch")
    if dict(current.train_window) != dict(requested_identity["train_window"]):
        rebuild_reasons.append("train_window_mismatch")
    if invalidation_reasons:
        return FeatureCacheDecision(
            decision=CACHE_DECISION_INVALIDATE,
            reasons=tuple(dict.fromkeys(invalidation_reasons)),
            cache_key=current.cache_key,
            requested_cache_key=requested_cache_key,
            diagnostics=diagnostics,
        )
    if rebuild_reasons:
        return FeatureCacheDecision(
            decision=CACHE_DECISION_REBUILD,
            reasons=tuple(dict.fromkeys(rebuild_reasons)),
            cache_key=current.cache_key,
            requested_cache_key=requested_cache_key,
            diagnostics=diagnostics,
        )
    return FeatureCacheDecision(
        decision=CACHE_DECISION_REUSE,
        reasons=(),
        cache_key=current.cache_key,
        requested_cache_key=requested_cache_key,
        diagnostics=diagnostics,
    )


def sandbox_feature_cache_noop_writer(manifest: FeatureCacheManifest | Mapping[str, Any]) -> dict[str, Any]:
    obj = manifest if isinstance(manifest, FeatureCacheManifest) else FeatureCacheManifest.from_dict(manifest)
    return {
        "schema_version": FEATURE_CACHE_SCHEMA_VERSION,
        "artifact_kind": FEATURE_CACHE_NOOP_WRITER_ARTIFACT_KIND,
        "status": "skipped_manifest_only",
        "cache_matrix_written": False,
        "cache_artifact_path": obj.cache_artifact_path,
        "cache_key": obj.cache_key,
        "reason": "sandbox_feature_cache_materialization_is_disabled",
        "artifact_boundary": obj.artifact_boundary,
    }


def write_sandbox_feature_cache_manifest(
    manifest: FeatureCacheManifest | Mapping[str, Any],
    *,
    report_root: str | Path = FOUNDATION_FEATURE_CACHE_ROOT,
    relative_path: str | Path = Path("feature_cache") / "feature_cache_manifest.json",
    project_root: str | Path | None = None,
) -> Path:
    obj = manifest if isinstance(manifest, FeatureCacheManifest) else FeatureCacheManifest.from_dict(manifest)
    root = validate_feature_cache_report_root(report_root, project_root=project_root)
    requested = Path(relative_path)
    if requested.is_absolute() or ".." in requested.parts:
        raise ValueError("Regime feature cache relative_path must stay under reports/regimes/foundation")
    if requested.suffix.lower() != ".json":
        raise ValueError("Regime feature cache manifest path must end in .json")
    path = _safe_child(root, *requested.parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dumps_json(obj.as_dict()) + "\n", encoding="utf-8")
    return path


__all__ = [
    "CACHE_DECISION_INVALIDATE",
    "CACHE_DECISION_REBUILD",
    "CACHE_DECISION_REUSE",
    "CACHE_DECISIONS",
    "CACHE_STATUS_INVALID",
    "CACHE_STATUS_VALID",
    "CACHE_STATUSES",
    "FEATURE_CACHE_DECISION_ARTIFACT_KIND",
    "FEATURE_CACHE_MANIFEST_ARTIFACT_KIND",
    "FEATURE_CACHE_NOOP_WRITER_ARTIFACT_KIND",
    "FEATURE_CACHE_SCHEMA_VERSION",
    "FOUNDATION_FEATURE_CACHE_ROOT",
    "FeatureCacheDecision",
    "FeatureCacheManifest",
    "build_feature_cache_manifest",
    "can_reuse_cache",
    "sandbox_feature_cache_noop_writer",
    "validate_feature_cache_report_root",
    "write_sandbox_feature_cache_manifest",
]
