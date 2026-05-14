"""Compatibility feature-cache manifest surface.

Canonical foundation cache identity and reuse decisions live in
``src.regimes.core.feature_cache``. This module remains for older diagnostics
that still exercise the first scaffold manifest shape.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field, replace
from datetime import date, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from src.regimes.core.artifacts import safe_path_part, write_json
from src.regimes.core.feature_preprocessing import FittedRegimePreprocessor, TransformedFeatureMatrix
from src.regimes.core.foundation_contracts import (
    REGIME_ASSET_SCOPES,
    REGIME_LAYER_AXES,
    REGIME_LAYERS,
    REGIME_STUDY_BANDS,
    SourceArtifactLineage,
    WindowBounds,
)
from src.regimes.core.pathway_artifacts import require_pathway_diagnostics_root


REGIME_FEATURE_CACHE_SCHEMA_VERSION = 1
FEATURE_CACHE_ARTIFACT_KIND = "regime_feature_cache_manifest"
FEATURE_CACHE_REUSE_DECISION_ARTIFACT_KIND = "regime_feature_cache_reuse_decision"
FEATURE_CACHE_MANIFEST_STATUS_VALID = "valid"
FEATURE_CACHE_MANIFEST_STATUS_INVALID = "invalid"
FEATURE_CACHE_WRITE_KINDS: tuple[str, ...] = ("sandbox", "diagnostics", "staged")
FEATURE_CACHE_LEAKAGE_GUARD_STATUSES: tuple[str, ...] = (
    "train_window_only",
    "validated_train_window_only",
    "warning_review_required",
    "blocked",
)


def _safe_float(value: object) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    return out if np.isfinite(out) else None


def _jsonable(value: Any) -> Any:
    if hasattr(value, "as_dict"):
        return value.as_dict()
    if hasattr(value, "to_metadata"):
        return value.to_metadata()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(val) for key, val in value.items()}
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, set):
        return sorted(_jsonable(item) for item in value)
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return _safe_float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.datetime64):
        return pd.Timestamp(value).isoformat()
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    if isinstance(value, float):
        return _safe_float(value)
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if value is pd.NA:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    return str(value)


def _normalize_token(value: object, *, field_name: str) -> str:
    token = str(value).strip().lower()
    if not token:
        raise ValueError(f"Regime feature cache {field_name} must be non-empty")
    return token


def _normalize_text(value: object, *, field_name: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"Regime feature cache {field_name} must be non-empty")
    return text


def _require_member(value: object, allowed: Sequence[str], *, field_name: str) -> str:
    token = _normalize_token(value, field_name=field_name)
    if token not in allowed:
        valid = ", ".join(str(item) for item in allowed)
        raise ValueError(f"Unsupported Regime feature cache {field_name} {token!r}; expected one of: {valid}")
    return token


def _unique_texts(values: Sequence[object], *, field_name: str, require_nonempty: bool = False) -> tuple[str, ...]:
    out = tuple(dict.fromkeys(str(value).strip() for value in values if str(value).strip()))
    if require_nonempty and not out:
        raise ValueError(f"Regime feature cache {field_name} must include at least one value")
    return out


def _window_dict(window: WindowBounds | Mapping[str, Any]) -> dict[str, Any]:
    if hasattr(window, "as_dict"):
        return window.as_dict()  # type: ignore[no-any-return]
    return {
        "start_ts": dict(window).get("start_ts"),
        "end_ts": dict(window).get("end_ts"),
    }


def _canonical(value: Any) -> str:
    return json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":"))


def _lineage_signature(lineage: Sequence[SourceArtifactLineage | Mapping[str, Any]]) -> str:
    payloads = [item.as_dict() if hasattr(item, "as_dict") else dict(item) for item in lineage]
    return _canonical(payloads)


def _fingerprint_signature(fingerprints: Sequence["SourceFileFingerprint" | Mapping[str, Any]]) -> str:
    payloads = [item.as_dict() if hasattr(item, "as_dict") else dict(item) for item in fingerprints]
    return _canonical(payloads)


def _hash_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


@dataclass(frozen=True)
class SourceFileFingerprint:
    path: str
    exists: bool
    size_bytes: int | None = None
    mtime_ns: int | None = None
    content_hash: str | None = None
    fingerprint_error: str | None = None
    schema_version: int = REGIME_FEATURE_CACHE_SCHEMA_VERSION

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "path": self.path,
            "exists": bool(self.exists),
            "size_bytes": self.size_bytes,
            "mtime_ns": self.mtime_ns,
            "content_hash": self.content_hash,
            "fingerprint_error": self.fingerprint_error,
        }


def fingerprint_source_path(path: Path | str, *, hash_file: bool = False) -> SourceFileFingerprint:
    raw_path = Path(path)
    try:
        stat = raw_path.stat()
        content_hash = _hash_file(raw_path) if bool(hash_file) and raw_path.is_file() else None
        return SourceFileFingerprint(
            path=str(raw_path),
            exists=True,
            size_bytes=int(stat.st_size),
            mtime_ns=int(stat.st_mtime_ns),
            content_hash=content_hash,
        )
    except FileNotFoundError:
        return SourceFileFingerprint(path=str(raw_path), exists=False)
    except Exception as exc:
        return SourceFileFingerprint(
            path=str(raw_path),
            exists=raw_path.exists(),
            fingerprint_error=f"{type(exc).__name__}: {exc}",
        )


def fingerprint_source_paths(paths: Sequence[Path | str], *, hash_files: bool = False) -> tuple[SourceFileFingerprint, ...]:
    return tuple(fingerprint_source_path(path, hash_file=hash_files) for path in paths)


@dataclass(frozen=True)
class FeatureCacheShapeMetadata:
    rows_before_filter: int
    features_before_filter: int
    rows_after_filter: int
    features_after_filter: int
    rows_after_transform: int
    features_after_transform: int
    schema_version: int = REGIME_FEATURE_CACHE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for field_name in (
            "rows_before_filter",
            "features_before_filter",
            "rows_after_filter",
            "features_after_filter",
            "rows_after_transform",
            "features_after_transform",
        ):
            value = int(getattr(self, field_name))
            if value < 0:
                raise ValueError(f"Regime feature cache {field_name} must be non-negative")
            object.__setattr__(self, field_name, value)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "rows_before_filter": int(self.rows_before_filter),
            "features_before_filter": int(self.features_before_filter),
            "rows_after_filter": int(self.rows_after_filter),
            "features_after_filter": int(self.features_after_filter),
            "rows_after_transform": int(self.rows_after_transform),
            "features_after_transform": int(self.features_after_transform),
        }


@dataclass(frozen=True)
class FeatureCacheManifest:
    layer: str
    axis: str
    band: str
    asset_scope: str
    source_input_paths: tuple[str, ...]
    source_file_fingerprints: tuple[SourceFileFingerprint, ...]
    source_lineage: tuple[SourceArtifactLineage, ...]
    feature_family: str
    preprocessing_family: str
    train_window: WindowBounds
    validation_score_window: WindowBounds
    shape_metadata: FeatureCacheShapeMetadata
    dropped_features: tuple[Mapping[str, Any], ...]
    transform_fitted_on_window: Mapping[str, Any]
    leakage_guard_status: str
    cache_artifact_path: str
    cache_write_kind: str
    preprocessing_metadata: Mapping[str, Any] = field(default_factory=dict)
    feature_filter_diagnostics: Mapping[str, Any] = field(default_factory=dict)
    dataset_scope: Mapping[str, Any] = field(default_factory=dict)
    invalidation_reasons: tuple[str, ...] = ()
    status: str = FEATURE_CACHE_MANIFEST_STATUS_VALID
    diagnostics: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = REGIME_FEATURE_CACHE_SCHEMA_VERSION
    artifact_kind: str = FEATURE_CACHE_ARTIFACT_KIND

    def __post_init__(self) -> None:
        layer = _require_member(self.layer, REGIME_LAYERS, field_name="layer")
        axis = _normalize_token(self.axis, field_name="axis")
        band = _require_member(self.band, REGIME_STUDY_BANDS, field_name="band")
        asset_scope = _require_member(self.asset_scope, REGIME_ASSET_SCOPES, field_name="asset scope")
        if axis not in REGIME_LAYER_AXES[layer]:
            valid = ", ".join(REGIME_LAYER_AXES[layer])
            raise ValueError(f"Unsupported Regime feature cache {layer} axis {axis!r}; expected one of: {valid}")
        if not self.source_input_paths:
            raise ValueError("Regime feature cache source_input_paths must be non-empty")
        if not self.source_file_fingerprints:
            raise ValueError("Regime feature cache source_file_fingerprints must be non-empty")
        if not self.source_lineage:
            raise ValueError("Regime feature cache source_lineage must be non-empty")
        feature_family = _normalize_token(self.feature_family, field_name="feature family")
        preprocessing_family = _normalize_token(self.preprocessing_family, field_name="preprocessing family")
        leakage_status = _require_member(
            self.leakage_guard_status,
            FEATURE_CACHE_LEAKAGE_GUARD_STATUSES,
            field_name="leakage guard status",
        )
        write_kind = _require_member(self.cache_write_kind, FEATURE_CACHE_WRITE_KINDS, field_name="cache write kind")
        cache_path = _normalize_text(self.cache_artifact_path, field_name="cache artifact path")
        status = _require_member(
            self.status,
            (FEATURE_CACHE_MANIFEST_STATUS_VALID, FEATURE_CACHE_MANIFEST_STATUS_INVALID),
            field_name="manifest status",
        )
        reasons = tuple(str(reason).strip() for reason in self.invalidation_reasons if str(reason).strip())
        if status == FEATURE_CACHE_MANIFEST_STATUS_INVALID and not reasons:
            raise ValueError("Regime invalid feature cache manifest requires invalidation_reasons")
        if status == FEATURE_CACHE_MANIFEST_STATUS_VALID and reasons:
            status = FEATURE_CACHE_MANIFEST_STATUS_INVALID
        object.__setattr__(self, "layer", layer)
        object.__setattr__(self, "axis", axis)
        object.__setattr__(self, "band", band)
        object.__setattr__(self, "asset_scope", asset_scope)
        object.__setattr__(self, "source_input_paths", tuple(str(path) for path in self.source_input_paths))
        object.__setattr__(self, "source_file_fingerprints", tuple(self.source_file_fingerprints))
        object.__setattr__(self, "source_lineage", tuple(self.source_lineage))
        object.__setattr__(self, "feature_family", feature_family)
        object.__setattr__(self, "preprocessing_family", preprocessing_family)
        object.__setattr__(self, "leakage_guard_status", leakage_status)
        object.__setattr__(self, "cache_artifact_path", cache_path)
        object.__setattr__(self, "cache_write_kind", write_kind)
        object.__setattr__(self, "dropped_features", tuple(dict(item) for item in self.dropped_features))
        object.__setattr__(self, "transform_fitted_on_window", dict(self.transform_fitted_on_window))
        object.__setattr__(self, "preprocessing_metadata", dict(self.preprocessing_metadata))
        object.__setattr__(self, "feature_filter_diagnostics", dict(self.feature_filter_diagnostics))
        object.__setattr__(self, "dataset_scope", dict(self.dataset_scope))
        object.__setattr__(self, "invalidation_reasons", reasons)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "diagnostics", dict(self.diagnostics))

    @property
    def source_lineage_signature(self) -> str:
        return _lineage_signature(self.source_lineage)

    @property
    def source_file_fingerprint_signature(self) -> str:
        return _fingerprint_signature(self.source_file_fingerprints)

    @property
    def artifact_boundary(self) -> dict[str, Any]:
        return {
            "write_mode": f"feature_cache_manifest_{self.cache_write_kind}",
            "manifest_only": True,
            "cache_matrix_write_enabled": False,
            "production_writes_enabled": False,
            "production_cache_write_enabled": False,
            "production_outputs_written": False,
            "parquet_writes_enabled": False,
            "definition_writes_enabled": False,
            "state_writes_enabled": False,
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "artifact_kind": self.artifact_kind,
            "status": self.status,
            "layer": self.layer,
            "axis": self.axis,
            "band": self.band,
            "asset_scope": self.asset_scope,
            "dataset_scope": _jsonable(dict(self.dataset_scope)),
            "source_input_paths": list(self.source_input_paths),
            "source_file_fingerprints": [fingerprint.as_dict() for fingerprint in self.source_file_fingerprints],
            "source_file_fingerprint_signature": self.source_file_fingerprint_signature,
            "source_lineage": [lineage.as_dict() for lineage in self.source_lineage],
            "source_lineage_signature": self.source_lineage_signature,
            "feature_family": self.feature_family,
            "preprocessing_family": self.preprocessing_family,
            "train_window": self.train_window.as_dict(),
            "validation_score_window": self.validation_score_window.as_dict(),
            "shape_metadata": self.shape_metadata.as_dict(),
            "rows_features_before_filter": [
                self.shape_metadata.rows_before_filter,
                self.shape_metadata.features_before_filter,
            ],
            "rows_features_after_filter": [
                self.shape_metadata.rows_after_filter,
                self.shape_metadata.features_after_filter,
            ],
            "rows_features_after_transform": [
                self.shape_metadata.rows_after_transform,
                self.shape_metadata.features_after_transform,
            ],
            "dropped_features": [_jsonable(dict(item)) for item in self.dropped_features],
            "transform_fitted_on_window": _jsonable(dict(self.transform_fitted_on_window)),
            "leakage_guard_status": self.leakage_guard_status,
            "cache_artifact_path": self.cache_artifact_path,
            "cache_write_kind": self.cache_write_kind,
            "preprocessing_metadata": _jsonable(dict(self.preprocessing_metadata)),
            "feature_filter_diagnostics": _jsonable(dict(self.feature_filter_diagnostics)),
            "invalidation_reasons": list(self.invalidation_reasons),
            "diagnostics": _jsonable(dict(self.diagnostics)),
            "artifact_boundary": self.artifact_boundary,
        }

    def to_json(self, **kwargs: Any) -> str:
        return json.dumps(self.as_dict(), **kwargs)


@dataclass(frozen=True)
class FeatureCacheReuseDecision:
    reusable: bool
    status: str
    reasons: tuple[str, ...]
    diagnostics: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = REGIME_FEATURE_CACHE_SCHEMA_VERSION
    artifact_kind: str = FEATURE_CACHE_REUSE_DECISION_ARTIFACT_KIND

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "artifact_kind": self.artifact_kind,
            "reusable": bool(self.reusable),
            "status": self.status,
            "reasons": list(self.reasons),
            "diagnostics": _jsonable(dict(self.diagnostics)),
        }


def build_feature_cache_manifest(
    *,
    feature_matrix: pd.DataFrame,
    fitted_preprocessor: FittedRegimePreprocessor,
    transformed_matrix: TransformedFeatureMatrix | None = None,
    layer: str,
    axis: str,
    band: str,
    asset_scope: str,
    source_input_paths: Sequence[Path | str],
    source_lineage: Sequence[SourceArtifactLineage],
    feature_family: str,
    train_window: WindowBounds,
    validation_score_window: WindowBounds,
    cache_artifact_path: Path | str,
    cache_write_kind: str = "diagnostics",
    dataset_scope: Mapping[str, Any] | None = None,
    leakage_guard_status: str | None = None,
    hash_source_files: bool = False,
) -> FeatureCacheManifest:
    transform = transformed_matrix
    x_transform = transform.x if transform is not None else fitted_preprocessor.x
    filter_result = fitted_preprocessor.filter_result
    preprocessing_metadata = fitted_preprocessor.to_metadata()
    fitted_window = dict(preprocessing_metadata.get("fit_window") or train_window.as_dict())
    warnings = tuple(str(item) for item in preprocessing_metadata.get("warnings", ()) or ())
    guard_status = leakage_guard_status or (
        "validated_train_window_only" if preprocessing_metadata.get("fit_scope") == "train_only" and not warnings else "warning_review_required"
    )
    diagnostics = {
        "manifest_builder": "build_feature_cache_manifest",
        "cache_matrix_materialized": False,
        "source_fingerprints_hash_files": bool(hash_source_files),
        "preprocessing_warnings": list(warnings),
    }
    return FeatureCacheManifest(
        layer=layer,
        axis=axis,
        band=band,
        asset_scope=asset_scope,
        source_input_paths=tuple(str(path) for path in source_input_paths),
        source_file_fingerprints=fingerprint_source_paths(source_input_paths, hash_files=hash_source_files),
        source_lineage=tuple(source_lineage),
        feature_family=feature_family,
        preprocessing_family=fitted_preprocessor.preprocess_name,
        train_window=train_window,
        validation_score_window=validation_score_window,
        shape_metadata=FeatureCacheShapeMetadata(
            rows_before_filter=int(filter_result.before_shape[0]),
            features_before_filter=int(filter_result.before_shape[1]),
            rows_after_filter=int(filter_result.after_shape[0]),
            features_after_filter=int(filter_result.after_shape[1]),
            rows_after_transform=int(x_transform.shape[0]),
            features_after_transform=int(x_transform.shape[1]) if x_transform.ndim == 2 else 0,
        ),
        dropped_features=tuple(record.as_dict() for record in filter_result.dropped_features),
        transform_fitted_on_window=fitted_window,
        leakage_guard_status=guard_status,
        cache_artifact_path=str(cache_artifact_path),
        cache_write_kind=cache_write_kind,
        preprocessing_metadata=preprocessing_metadata,
        feature_filter_diagnostics=filter_result.to_metadata(),
        dataset_scope=dict(dataset_scope or {}),
        diagnostics=diagnostics,
    )


def validate_feature_cache_reuse(
    manifest: FeatureCacheManifest | Mapping[str, Any],
    *,
    layer: str,
    axis: str,
    band: str,
    asset_scope: str,
    source_lineage: Sequence[SourceArtifactLineage | Mapping[str, Any]] | None = None,
    source_input_paths: Sequence[Path | str] | None = None,
    source_file_fingerprints: Sequence[SourceFileFingerprint | Mapping[str, Any]] | None = None,
    feature_family: str,
    preprocessing_family: str,
    train_window: WindowBounds | Mapping[str, Any],
    schema_version: int = REGIME_FEATURE_CACHE_SCHEMA_VERSION,
) -> FeatureCacheReuseDecision:
    payload = manifest.as_dict() if hasattr(manifest, "as_dict") else dict(manifest)
    reasons: list[str] = []
    diagnostics: dict[str, Any] = {
        "checked_fields": [
            "schema_version",
            "layer",
            "axis",
            "band",
            "asset_scope",
            "source_lineage",
            "source_input_paths",
            "source_file_fingerprints",
            "feature_family",
            "preprocessing_family",
            "train_window",
            "manifest_status",
        ]
    }
    if int(payload.get("schema_version", -1)) != int(schema_version):
        reasons.append("schema_version_mismatch")
    for field_name, requested in (
        ("layer", _normalize_token(layer, field_name="layer")),
        ("axis", _normalize_token(axis, field_name="axis")),
        ("band", _normalize_token(band, field_name="band")),
        ("asset_scope", _normalize_token(asset_scope, field_name="asset scope")),
        ("feature_family", _normalize_token(feature_family, field_name="feature family")),
        ("preprocessing_family", _normalize_token(preprocessing_family, field_name="preprocessing family")),
    ):
        if str(payload.get(field_name)) != requested:
            reasons.append(f"{field_name}_mismatch")
    if dict(payload.get("train_window") or {}) != _window_dict(train_window):
        reasons.append("train_window_mismatch")
    if source_lineage is not None:
        requested_signature = _lineage_signature(source_lineage)
        if str(payload.get("source_lineage_signature")) != requested_signature:
            reasons.append("source_lineage_mismatch")
        diagnostics["requested_source_lineage_signature"] = requested_signature
    if source_input_paths is not None:
        requested_paths = tuple(str(path) for path in source_input_paths)
        if tuple(str(path) for path in payload.get("source_input_paths", ())) != requested_paths:
            reasons.append("source_input_paths_mismatch")
    if source_file_fingerprints is not None:
        requested_fingerprint_signature = _fingerprint_signature(source_file_fingerprints)
        if str(payload.get("source_file_fingerprint_signature")) != requested_fingerprint_signature:
            reasons.append("source_file_fingerprints_mismatch")
        diagnostics["requested_source_file_fingerprint_signature"] = requested_fingerprint_signature
    if payload.get("status") != FEATURE_CACHE_MANIFEST_STATUS_VALID:
        reasons.append("manifest_status_invalid")
    if payload.get("artifact_boundary", {}).get("production_outputs_written") is not False:
        reasons.append("production_output_claim")
    reusable = not reasons
    return FeatureCacheReuseDecision(
        reusable=reusable,
        status="reuse_valid" if reusable else "rebuild_required",
        reasons=tuple(dict.fromkeys(reasons)),
        diagnostics=diagnostics,
    )


def mark_feature_cache_invalid(
    manifest: FeatureCacheManifest,
    reasons: Sequence[str],
    *,
    diagnostics: Mapping[str, Any] | None = None,
) -> FeatureCacheManifest:
    normalized = tuple(dict.fromkeys(str(reason).strip() for reason in reasons if str(reason).strip()))
    if not normalized:
        raise ValueError("Regime feature cache invalidation requires at least one reason")
    return replace(
        manifest,
        status=FEATURE_CACHE_MANIFEST_STATUS_INVALID,
        invalidation_reasons=tuple((*manifest.invalidation_reasons, *normalized)),
        diagnostics={**dict(manifest.diagnostics), **dict(diagnostics or {})},
    )


def feature_cache_manifest_diagnostic_path(
    diagnostics_root: Path,
    *,
    layer: str,
    axis: str,
    band: str,
    run_id: str,
    filename: str = "feature_cache_manifest.json",
) -> Path:
    return (
        Path(diagnostics_root)
        / "feature_cache_manifests"
        / safe_path_part(layer, context="Feature cache manifest layer")
        / safe_path_part(axis, context="Feature cache manifest axis")
        / safe_path_part(band, context="Feature cache manifest band")
        / safe_path_part(run_id, context="Feature cache manifest run id")
        / safe_path_part(filename, context="Feature cache manifest filename")
    )


def write_feature_cache_manifest_diagnostic(
    manifest: FeatureCacheManifest,
    *,
    diagnostics_root: Path,
    run_id: str,
    project_root: Path | None = None,
    env: Mapping[str, str] | None = None,
) -> Path:
    root_policy = require_pathway_diagnostics_root(Path(diagnostics_root), project_root=project_root, env=env)
    path = feature_cache_manifest_diagnostic_path(
        Path(diagnostics_root),
        layer=manifest.layer,
        axis=manifest.axis,
        band=manifest.band,
        run_id=run_id,
    )
    payload = manifest.as_dict()
    payload["diagnostics_root_policy"] = root_policy.as_dict()
    payload["actual_cache_matrix_written"] = False
    write_json(path, payload, write_kind="Regime feature cache manifest diagnostic")
    return path


def no_op_feature_cache_writer(manifest: FeatureCacheManifest) -> dict[str, Any]:
    return {
        "schema_version": REGIME_FEATURE_CACHE_SCHEMA_VERSION,
        "artifact_kind": "regime_feature_cache_no_op_writer",
        "status": "skipped_manifest_only",
        "cache_matrix_written": False,
        "cache_artifact_path": manifest.cache_artifact_path,
        "cache_write_kind": manifest.cache_write_kind,
        "reason": "high_volume_feature_cache_writes_not_enabled",
        "artifact_boundary": manifest.artifact_boundary,
    }


__all__ = [
    "FEATURE_CACHE_ARTIFACT_KIND",
    "FEATURE_CACHE_LEAKAGE_GUARD_STATUSES",
    "FEATURE_CACHE_MANIFEST_STATUS_INVALID",
    "FEATURE_CACHE_MANIFEST_STATUS_VALID",
    "FEATURE_CACHE_REUSE_DECISION_ARTIFACT_KIND",
    "FEATURE_CACHE_WRITE_KINDS",
    "REGIME_FEATURE_CACHE_SCHEMA_VERSION",
    "FeatureCacheManifest",
    "FeatureCacheReuseDecision",
    "FeatureCacheShapeMetadata",
    "SourceFileFingerprint",
    "build_feature_cache_manifest",
    "feature_cache_manifest_diagnostic_path",
    "fingerprint_source_path",
    "fingerprint_source_paths",
    "mark_feature_cache_invalid",
    "no_op_feature_cache_writer",
    "validate_feature_cache_reuse",
    "write_feature_cache_manifest_diagnostic",
]
