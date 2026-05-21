from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from src.regimes.core.artifact_refs import validate_portable_relative_path
from src.regimes.core.artifact_refs import resolve_artifact_ref, validate_portable_artifact_ref
from src.regimes.core.serialization import to_jsonable
from src.regimes.regime_features.cross_asset_feature_catalog import (
    CrossAssetRelationshipFeatureCatalog,
    default_cross_asset_relationship_feature_catalog,
    validate_cross_asset_relationship_feature_catalog,
)
from src.regimes.regime_features.cross_asset_feature_rows import (
    CrossAssetRelationshipFeatureManifest,
    CrossAssetRelationshipFeatureRow,
    build_cross_asset_feature_row_from_process1_profile,
    validate_cross_asset_feature_row,
)
from src.regimes.regime_features.cross_asset_feature_writer import (
    CrossAssetFeatureWriteResult,
    cross_asset_feature_expected_output_paths,
    validate_cross_asset_feature_output_root,
    write_cross_asset_feature_outputs,
)
CROSS_ASSET_FEATURE_GENERATOR_STATUS_COMPLETED = "completed"
CROSS_ASSET_FEATURE_GENERATOR_STATUS_NO_ROWS = "no_rows"
PROCESS1_TO_PROCESS2_HANDOFF_ARTIFACT_KIND = "process1_to_process2_handoff_manifest"
ASSET_RELATIONSHIP_PROFILES_SCHEMA_ID = "asset_relationship_profiles"
EDGE_ALIAS_MANIFEST_SCHEMA_ID = "edge_alias_manifest"
ISOLATED_ASSET_PROFILES_SCHEMA_ID = "isolated_asset_profiles"
REFIT_SNAPSHOT_MANIFEST_SCHEMA_ID = "refit_snapshot_manifest"
RELATIONSHIP_STABILITY_SCORES_SCHEMA_ID = "relationship_stability_scores"


@dataclass(frozen=True)
class CrossAssetFeatureGeneratorResult:
    status: str
    rows: Sequence[CrossAssetRelationshipFeatureRow]
    manifest: CrossAssetRelationshipFeatureManifest
    catalog: CrossAssetRelationshipFeatureCatalog
    write_result: CrossAssetFeatureWriteResult | None = None
    warnings: Sequence[str] = ()
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "artifact_kind": "cross_asset_feature_generator_result",
            "status": self.status,
            "row_count": len(self.rows),
            "rows": [row.as_dict() for row in self.rows],
            "manifest": self.manifest.as_dict(),
            "catalog": self.catalog.as_dict(),
            "write_result": self.write_result.as_dict() if self.write_result is not None else None,
            "warnings": list(self.warnings),
            "diagnostics": to_jsonable(dict(self.diagnostics)),
            "production_enabled": False,
            "production_outputs_written": False,
            "cross_asset_labels_written": False,
            "one_column_per_related_asset_allowed": False,
        }


def build_cross_asset_feature_rows_from_handoff(
    process1_to_process2_handoff_manifest: str | Path | Mapping[str, Any],
    *,
    output_root: str | Path | None = None,
    feature_catalog: CrossAssetRelationshipFeatureCatalog | Mapping[str, Any] | None = None,
    write_outputs: bool = False,
    prefer_parquet: bool = True,
    allow_json_fallback_for_tests: bool = False,
    production_enabled: bool = False,
    project_root: str | Path | None = None,
) -> CrossAssetFeatureGeneratorResult:
    if production_enabled is not False:
        raise ValueError("Cross-Asset feature generator production writes are disabled")
    handoff, manifest_base = _load_handoff(process1_to_process2_handoff_manifest)
    _validate_handoff_for_generator(handoff, base_path=manifest_base)
    catalog = _resolve_catalog(feature_catalog)
    snapshot = _read_json_object(_artifact_path(handoff, "refit_snapshot_manifest_path", base_path=manifest_base))
    profile_rows = _read_table(_artifact_path(handoff, "asset_relationship_profiles_path", base_path=manifest_base))
    stability_rows = _read_table(_artifact_path(handoff, "relationship_stability_scores_path", base_path=manifest_base))
    isolated_rows = _read_table(_artifact_path(handoff, "isolated_asset_profiles_path", base_path=manifest_base))
    warnings: list[str] = []
    alias_rows: list[dict[str, Any]] = []
    alias_path_text = str(handoff.get("edge_alias_manifest_path", "") or "").strip()
    if alias_path_text:
        alias_path = _resolve_artifact_path(alias_path_text, base_path=manifest_base)
        if alias_path.exists() and alias_path.is_file():
            alias_rows = _read_table(alias_path)
        else:
            warnings.append("optional edge_alias_manifest missing; sidecar alias availability set false")
    else:
        warnings.append("optional edge_alias_manifest not referenced; sidecar alias availability set false")

    isolated_by_key = {_row_key(row): row for row in isolated_rows}
    stability_by_asset = _stability_summary_by_asset(stability_rows)
    alias_flags = _alias_flags_by_asset(alias_rows)
    effective_start_ts = _required_value(snapshot, "effective_start_ts")
    effective_end_ts = _required_value(snapshot, "effective_end_ts")
    source_tail_ts = handoff.get("source_tail_ts", snapshot.get("source_tail_ts"))
    if source_tail_ts is None:
        raise ValueError("Cross-Asset feature generator requires source_tail_ts")

    rows: list[CrossAssetRelationshipFeatureRow] = []
    for profile in sorted(profile_rows, key=lambda row: (_text(row.get("asset"), field_name="asset"), _text(row.get("refit_key"), field_name="refit_key"))):
        key = _row_key(profile)
        isolated = isolated_by_key.get(key)
        if isolated is None:
            raise ValueError(f"Cross-Asset feature generator missing isolated_asset_profiles row for {key}")
        merged = dict(profile)
        merged.update(
            {
                "known_at_ts": handoff.get("known_at_ts"),
                "isolated_asset_score": isolated.get("isolated_asset_score", profile.get("isolated_asset_score", 1.0)),
                "peer_signal_availability_status": profile.get("peer_signal_availability_status", isolated.get("isolation_status", "unavailable")),
                "stable_edge_count": isolated.get("stable_edge_count", stability_by_asset.get(str(profile.get("asset")), {}).get("stable_edge_count", 0)),
                "candidate_edge_count": isolated.get("candidate_edge_count", stability_by_asset.get(str(profile.get("asset")), {}).get("candidate_edge_count", 0)),
            }
        )
        merged.update(alias_flags.get(str(profile.get("asset")), {}))
        row = build_cross_asset_feature_row_from_process1_profile(
            merged,
            handoff_id=_text(handoff.get("handoff_id"), field_name="handoff_id"),
            effective_start_ts=effective_start_ts,
            effective_end_ts=effective_end_ts,
            source_tail_ts=source_tail_ts,
            feature_catalog_id=catalog.catalog_id,
        )
        validate_cross_asset_feature_row(row)
        rows.append(row)

    feature_manifest_id = _feature_manifest_id(handoff, rows, catalog.catalog_id)
    if write_outputs:
        if output_root is None:
            raise ValueError("Cross-Asset feature generator output_root is required when write_outputs=True")
        write_result = write_cross_asset_feature_outputs(
            rows,
            catalog=catalog,
            output_root=output_root,
            handoff_id=_text(handoff.get("handoff_id"), field_name="handoff_id"),
            input_artifacts=_input_artifacts(handoff),
            refit_key=_text(handoff.get("refit_key"), field_name="refit_key"),
            interval=int(handoff.get("interval")),
            window=int(handoff.get("window")),
            known_at_ts=_required_value(handoff, "known_at_ts"),
            feature_manifest_id=feature_manifest_id,
            prefer_parquet=prefer_parquet,
            allow_json_fallback_for_tests=allow_json_fallback_for_tests,
            production_enabled=production_enabled,
            project_root=project_root,
        )
        manifest = write_result.manifest
    else:
        if output_root is not None:
            root = validate_cross_asset_feature_output_root(output_root, project_root=project_root, production_enabled=production_enabled)
            row_format = "parquet" if prefer_parquet else "jsonl"
            output_paths = cross_asset_feature_expected_output_paths(
                root,
                catalog_id=catalog.catalog_id,
                refit_key=_text(handoff.get("refit_key"), field_name="refit_key"),
                interval=int(handoff.get("interval")),
                window=int(handoff.get("window")),
                row_format=row_format,
            )
        else:
            output_paths = {}
        manifest = CrossAssetRelationshipFeatureManifest(
            feature_manifest_id=feature_manifest_id,
            handoff_id=_text(handoff.get("handoff_id"), field_name="handoff_id"),
            input_artifacts=_input_artifacts(handoff),
            output_paths={name: _relative_output_path(path, root) for name, path in output_paths.items()} if output_root is not None else {},
            feature_catalog_id=catalog.catalog_id,
            interval=int(handoff.get("interval")),
            window=int(handoff.get("window")),
            refit_key=_text(handoff.get("refit_key"), field_name="refit_key"),
            known_at_ts=_required_value(handoff, "known_at_ts"),
        )
        write_result = None

    return CrossAssetFeatureGeneratorResult(
        status=CROSS_ASSET_FEATURE_GENERATOR_STATUS_COMPLETED if rows else CROSS_ASSET_FEATURE_GENERATOR_STATUS_NO_ROWS,
        rows=tuple(rows),
        manifest=manifest,
        catalog=catalog,
        write_result=write_result,
        warnings=tuple(warnings),
        diagnostics={
            "asset_relationship_profile_rows": len(profile_rows),
            "relationship_stability_score_rows": len(stability_rows),
            "isolated_asset_profile_rows": len(isolated_rows),
            "edge_alias_manifest_rows": len(alias_rows),
            "write_outputs": bool(write_outputs),
            "production_enabled": False,
        },
    )


def _load_handoff(source: str | Path | Mapping[str, Any]) -> tuple[dict[str, Any], Path | None]:
    if isinstance(source, Mapping):
        return dict(source), None
    path = Path(source)
    payload = _read_json_object(path)
    return dict(payload), path.parent


def _validate_handoff_for_generator(handoff: Mapping[str, Any], *, base_path: Path | None) -> None:
    if handoff.get("artifact_kind") not in {None, PROCESS1_TO_PROCESS2_HANDOFF_ARTIFACT_KIND}:
        raise ValueError("Cross-Asset feature generator requires a Process 1 to Process 2 handoff manifest")
    for flag in ("production_enabled", "production_outputs_written", "broad_all_to_all", "cross_asset_labels_written"):
        if bool(handoff.get(flag, False)):
            raise ValueError(f"Cross-Asset feature generator handoff {flag} must be false")
    if _to_orderable(_required_value(handoff, "source_tail_ts"), field_name="source_tail_ts") > _to_orderable(_required_value(handoff, "known_at_ts"), field_name="known_at_ts"):
        raise ValueError("Cross-Asset feature generator source_tail_ts must not exceed known_at_ts")
    for field_name in (
        "handoff_id",
        "refit_key",
        "interval",
        "window",
        "known_at_ts",
        "source_tail_ts",
        "asset_relationship_profiles_path",
        "relationship_stability_scores_path",
        "isolated_asset_profiles_path",
        "refit_snapshot_manifest_path",
    ):
        _required_value(handoff, field_name)
    for field_name in (
        "asset_relationship_profiles_path",
        "relationship_stability_scores_path",
        "isolated_asset_profiles_path",
        "refit_snapshot_manifest_path",
    ):
        path = _artifact_path(handoff, field_name, base_path=base_path)
        if not path.exists() or not path.is_file():
            raise ValueError(f"Cross-Asset feature generator missing required artifact: {field_name}")


def _resolve_catalog(catalog: CrossAssetRelationshipFeatureCatalog | Mapping[str, Any] | None) -> CrossAssetRelationshipFeatureCatalog:
    if catalog is None:
        resolved = default_cross_asset_relationship_feature_catalog()
    elif isinstance(catalog, CrossAssetRelationshipFeatureCatalog):
        resolved = catalog
    else:
        resolved = CrossAssetRelationshipFeatureCatalog(
            entries=tuple(catalog.get("entries", ())),
            catalog_id=str(catalog.get("catalog_id") or "relationship_feature_catalog_v1"),
            schema_version=int(catalog.get("schema_version") or 1),
            artifact_kind=str(catalog.get("artifact_kind") or "relationship_feature_catalog"),
        )
    validate_cross_asset_relationship_feature_catalog(resolved)
    return resolved


def _artifact_path(handoff: Mapping[str, Any], field_name: str, *, base_path: Path | None) -> Path:
    refs = handoff.get("artifact_refs")
    ref_key = _ref_key_for_path_field(field_name)
    if isinstance(refs, Mapping) and ref_key in refs:
        if base_path is None:
            raise ValueError("Cross-Asset feature generator requires a manifest file base path for portable artifact refs")
        return resolve_artifact_ref(refs[ref_key], artifact_root=base_path, must_exist=False)
    return _resolve_artifact_path(_text(handoff.get(field_name), field_name=field_name), base_path=base_path)


def _resolve_artifact_path(raw_path: str, *, base_path: Path | None) -> Path:
    path = Path(raw_path)
    if not path.is_absolute() and base_path is not None:
        path = base_path / path
    return path.resolve()


def _relative_output_path(path: Path, root: Path) -> str:
    return validate_portable_relative_path(path.resolve().relative_to(root.resolve()).as_posix())


def _read_json_object(path: str | Path) -> Mapping[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("Cross-Asset feature generator expected a JSON object")
    return payload


def _read_table(path: str | Path) -> list[dict[str, Any]]:
    resolved = Path(path)
    suffix = resolved.suffix.lower()
    if suffix == ".parquet":
        return [dict(row) for row in pd.read_parquet(resolved).to_dict(orient="records")]
    if suffix == ".jsonl":
        rows = []
        for line in resolved.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, Mapping):
                raise ValueError("Cross-Asset feature generator JSONL rows must be objects")
            rows.append(dict(payload))
        return rows
    if suffix == ".json":
        payload = _read_json_object(resolved)
        if "rows" in payload:
            rows = payload["rows"]
            if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
                raise ValueError("Cross-Asset feature generator JSON table rows must be a sequence")
            return [dict(row) for row in rows]
        return [dict(payload)]
    raise ValueError(f"Cross-Asset feature generator unsupported artifact format: {resolved.suffix}")


def _row_key(row: Mapping[str, Any]) -> tuple[str, str, int, int]:
    return (
        _text(row.get("asset"), field_name="asset"),
        _text(row.get("refit_key"), field_name="refit_key"),
        int(row.get("interval")),
        int(row.get("window")),
    )


def _stability_summary_by_asset(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for row in rows:
        asset = str(row.get("asset", "")).strip()
        if not asset:
            continue
        summary = out.setdefault(asset, {"candidate_edge_count": 0, "stable_edge_count": 0})
        summary["candidate_edge_count"] += 1
        if bool(row.get("enough_history", False)) and str(row.get("activation_status", "")).lower() in {"active", "selected", "available"}:
            summary["stable_edge_count"] += 1
    return out


def _alias_flags_by_asset(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, bool]]:
    out: dict[str, dict[str, bool]] = {}
    for row in rows:
        if str(row.get("activation_status", "")).lower() not in {"active", "selected", "available"}:
            continue
        asset = str(row.get("asset", "")).strip()
        slot = str(row.get("slot", "")).strip()
        if not asset or not slot:
            continue
        flags = out.setdefault(asset, {})
        if slot == "strongest_peer_slot_1":
            flags["strongest_peer_slot_1_alias_available"] = True
        if slot == "strongest_peer_slot_2":
            flags["strongest_peer_slot_2_available"] = True
    return out


def _input_artifacts(handoff: Mapping[str, Any]) -> dict[str, Any]:
    refs = handoff.get("artifact_refs")
    if isinstance(refs, Mapping) and refs:
        return {
            key: validate_portable_artifact_ref(ref).as_dict()
            for key, ref in refs.items()
            if key
            in {
                REFIT_SNAPSHOT_MANIFEST_SCHEMA_ID,
                ASSET_RELATIONSHIP_PROFILES_SCHEMA_ID,
                RELATIONSHIP_STABILITY_SCORES_SCHEMA_ID,
                ISOLATED_ASSET_PROFILES_SCHEMA_ID,
                EDGE_ALIAS_MANIFEST_SCHEMA_ID,
            }
        }
    return {
        REFIT_SNAPSHOT_MANIFEST_SCHEMA_ID: handoff.get("refit_snapshot_manifest_path"),
        ASSET_RELATIONSHIP_PROFILES_SCHEMA_ID: handoff.get("asset_relationship_profiles_path"),
        RELATIONSHIP_STABILITY_SCORES_SCHEMA_ID: handoff.get("relationship_stability_scores_path"),
        ISOLATED_ASSET_PROFILES_SCHEMA_ID: handoff.get("isolated_asset_profiles_path"),
        EDGE_ALIAS_MANIFEST_SCHEMA_ID: handoff.get("edge_alias_manifest_path"),
    }


def _ref_key_for_path_field(field_name: str) -> str:
    return {
        "refit_snapshot_manifest_path": REFIT_SNAPSHOT_MANIFEST_SCHEMA_ID,
        "asset_relationship_profiles_path": ASSET_RELATIONSHIP_PROFILES_SCHEMA_ID,
        "relationship_stability_scores_path": RELATIONSHIP_STABILITY_SCORES_SCHEMA_ID,
        "isolated_asset_profiles_path": ISOLATED_ASSET_PROFILES_SCHEMA_ID,
        "edge_alias_manifest_path": EDGE_ALIAS_MANIFEST_SCHEMA_ID,
    }.get(field_name, field_name)


def _feature_manifest_id(handoff: Mapping[str, Any], rows: Sequence[CrossAssetRelationshipFeatureRow], catalog_id: str) -> str:
    payload = {
        "handoff_id": handoff.get("handoff_id"),
        "refit_key": handoff.get("refit_key"),
        "interval": handoff.get("interval"),
        "window": handoff.get("window"),
        "catalog_id": catalog_id,
        "assets": [row.asset for row in rows],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def _required_value(payload: Mapping[str, Any], field_name: str) -> Any:
    value = payload.get(field_name)
    if value is None or str(value).strip() == "":
        raise ValueError(f"Cross-Asset feature generator requires {field_name}")
    return value


def _text(value: object, *, field_name: str) -> str:
    text = str(value).strip()
    if not text or text == "None":
        raise ValueError(f"Cross-Asset feature generator {field_name} must be non-empty")
    return text


def _to_orderable(value: object, *, field_name: str) -> float:
    if value is None or isinstance(value, bool):
        raise ValueError(f"Cross-Asset feature generator {field_name} must be timestamp-compatible")
    try:
        return float(value)
    except Exception as exc:
        raise ValueError(f"Cross-Asset feature generator {field_name} must be numeric") from exc


__all__ = [
    "CROSS_ASSET_FEATURE_GENERATOR_STATUS_COMPLETED",
    "CROSS_ASSET_FEATURE_GENERATOR_STATUS_NO_ROWS",
    "CrossAssetFeatureGeneratorResult",
    "build_cross_asset_feature_rows_from_handoff",
]
