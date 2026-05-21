from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from src.regimes.core.artifact_refs import make_artifact_ref, validate_portable_relative_path
from src.regimes.core.path_safety import PRODUCTION_LIKE_WRITE_PARTS
from src.regimes.core.paths import has_path_parts, normalized_path_parts, resolve_project_path
from src.regimes.core.serialization import to_jsonable
from src.regimes.regime_features.cross_asset_feature_catalog import (
    CrossAssetRelationshipFeatureCatalog,
    RELATIONSHIP_FEATURE_CATALOG_ID,
    validate_cross_asset_relationship_feature_catalog,
)
from src.regimes.regime_features.cross_asset_feature_rows import (
    CROSS_ASSET_FEATURE_MANIFEST_ARTIFACT_KIND,
    CROSS_ASSET_FEATURE_ROWS_ARTIFACT_KIND,
    CrossAssetRelationshipFeatureManifest,
    CrossAssetRelationshipFeatureRow,
    validate_cross_asset_feature_manifest,
    validate_cross_asset_feature_row,
)


CROSS_ASSET_FEATURE_WRITE_STATUS_WRITTEN = "written"
CROSS_ASSET_FEATURE_WRITE_STATUS_NO_ROWS = "no_rows"


@dataclass(frozen=True)
class CrossAssetFeaturePartitionWrite:
    artifact_kind: str
    status: str
    format: str
    path: Path
    row_count: int
    fallback_reason: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "artifact_kind": self.artifact_kind,
            "status": self.status,
            "format": self.format,
            "path": str(self.path),
            "row_count": int(self.row_count),
            "fallback_reason": self.fallback_reason,
            "production_enabled": False,
            "production_outputs_written": False,
            "cross_asset_labels_written": False,
            "one_column_per_related_asset_allowed": False,
        }


@dataclass(frozen=True)
class CrossAssetFeatureWriteResult:
    status: str
    output_root: Path
    writes: Sequence[CrossAssetFeaturePartitionWrite]
    manifest: CrossAssetRelationshipFeatureManifest
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    @property
    def row_count(self) -> int:
        return sum(write.row_count for write in self.writes if write.artifact_kind == CROSS_ASSET_FEATURE_ROWS_ARTIFACT_KIND)

    @property
    def written_paths(self) -> tuple[Path, ...]:
        return tuple(write.path for write in self.writes)

    def as_dict(self) -> dict[str, Any]:
        return {
            "artifact_kind": "cross_asset_feature_write_result",
            "status": self.status,
            "output_root": str(self.output_root),
            "row_count": int(self.row_count),
            "writes": [write.as_dict() for write in self.writes],
            "manifest": self.manifest.as_dict(),
            "diagnostics": to_jsonable(dict(self.diagnostics)),
            "production_enabled": False,
            "production_outputs_written": False,
            "cross_asset_labels_written": False,
            "one_column_per_related_asset_allowed": False,
        }


def write_cross_asset_feature_outputs(
    rows: Sequence[CrossAssetRelationshipFeatureRow | Mapping[str, Any]],
    *,
    catalog: CrossAssetRelationshipFeatureCatalog,
    output_root: str | Path,
    handoff_id: str,
    input_artifacts: Mapping[str, Any],
    refit_key: str,
    interval: int,
    window: int,
    known_at_ts: int | float | str,
    feature_manifest_id: str,
    prefer_parquet: bool = True,
    allow_json_fallback_for_tests: bool = False,
    production_enabled: bool = False,
    project_root: str | Path | None = None,
) -> CrossAssetFeatureWriteResult:
    if production_enabled is not False:
        raise ValueError("Cross-Asset feature production writes are disabled")
    if not prefer_parquet and not allow_json_fallback_for_tests:
        raise ValueError("Cross-Asset feature JSONL fallback is test-only or dependency-missing")
    root = validate_cross_asset_feature_output_root(output_root, project_root=project_root, production_enabled=production_enabled)
    validate_cross_asset_relationship_feature_catalog(catalog)
    payload_rows = [_row_payload(row) for row in rows]
    for row in payload_rows:
        validate_cross_asset_feature_row(row)
    row_format = "parquet" if prefer_parquet else "jsonl"
    output_paths = cross_asset_feature_expected_output_paths(
        root,
        catalog_id=catalog.catalog_id,
        refit_key=refit_key,
        interval=interval,
        window=window,
        row_format=row_format,
    )
    writes: list[CrossAssetFeaturePartitionWrite] = []
    catalog_path = output_paths["relationship_feature_catalog"]
    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    catalog_path.write_text(json.dumps(to_jsonable(catalog.as_dict()), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    writes.append(_write_record("relationship_feature_catalog", "json", catalog_path, 1))

    if payload_rows:
        rows_path = output_paths["cross_asset_feature_rows"]
        rows_path.parent.mkdir(parents=True, exist_ok=True)
        if prefer_parquet:
            try:
                pd.DataFrame([_tabular_row(row) for row in payload_rows]).to_parquet(rows_path, index=False)
                writes.append(_write_record(CROSS_ASSET_FEATURE_ROWS_ARTIFACT_KIND, "parquet", rows_path, len(payload_rows)))
            except (ImportError, ModuleNotFoundError) as exc:
                if not allow_json_fallback_for_tests:
                    raise
                jsonl_path = rows_path.with_suffix(".jsonl")
                _write_jsonl(jsonl_path, payload_rows)
                writes.append(_write_record(CROSS_ASSET_FEATURE_ROWS_ARTIFACT_KIND, "jsonl", jsonl_path, len(payload_rows), fallback_reason=type(exc).__name__))
        else:
            _write_jsonl(rows_path, payload_rows)
            writes.append(_write_record(CROSS_ASSET_FEATURE_ROWS_ARTIFACT_KIND, "jsonl", rows_path, len(payload_rows), fallback_reason="test_jsonl_fallback"))

    manifest_path = output_paths["cross_asset_feature_manifest"]
    output_refs = {
        name: make_artifact_ref(
            path,
            artifact_kind=name,
            artifact_root=root,
            producer="src.regimes.regime_features.cross_asset_feature_writer",
            known_at_ts=known_at_ts,
        ).as_dict()
        for name, path in output_paths.items()
        if path.exists() and path.is_file()
    }
    manifest = CrossAssetRelationshipFeatureManifest(
        feature_manifest_id=feature_manifest_id,
        handoff_id=handoff_id,
        input_artifacts=to_jsonable(dict(input_artifacts)),
        output_paths={name: _relative_path(path, root) for name, path in output_paths.items()},
        output_artifact_refs=output_refs,
        feature_catalog_id=catalog.catalog_id,
        interval=interval,
        window=window,
        refit_key=refit_key,
        known_at_ts=known_at_ts,
    )
    validate_cross_asset_feature_manifest(manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(to_jsonable(manifest.as_dict()), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    writes.append(_write_record(CROSS_ASSET_FEATURE_MANIFEST_ARTIFACT_KIND, "json", manifest_path, 1))

    return CrossAssetFeatureWriteResult(
        status=CROSS_ASSET_FEATURE_WRITE_STATUS_WRITTEN if payload_rows else CROSS_ASSET_FEATURE_WRITE_STATUS_NO_ROWS,
        output_root=root,
        writes=tuple(writes),
        manifest=manifest,
        diagnostics={"row_format": row_format, "production_enabled": False},
    )


def cross_asset_feature_expected_output_paths(
    output_root: str | Path,
    *,
    catalog_id: str = RELATIONSHIP_FEATURE_CATALOG_ID,
    refit_key: str,
    interval: int,
    window: int,
    row_format: str = "parquet",
) -> dict[str, Path]:
    root = Path(output_root)
    if row_format not in {"parquet", "jsonl"}:
        raise ValueError("Cross-Asset feature row_format must be parquet or jsonl")
    return {
        "relationship_feature_catalog": root / "cross_asset_features" / "relationship_feature_catalog" / f"feature_catalog_id={catalog_id}" / "relationship_feature_catalog.json",
        "cross_asset_feature_rows": root / "cross_asset_features" / CROSS_ASSET_FEATURE_ROWS_ARTIFACT_KIND / f"interval={int(interval)}" / f"window={int(window)}" / f"refit_key={refit_key}" / f"part-000.{row_format}",
        "cross_asset_feature_manifest": root / "cross_asset_features" / CROSS_ASSET_FEATURE_MANIFEST_ARTIFACT_KIND / f"refit_key={refit_key}" / f"interval={int(interval)}" / "feature_manifest.json",
    }


def validate_cross_asset_feature_output_root(
    output_root: str | Path,
    *,
    project_root: str | Path | None = None,
    production_enabled: bool = False,
) -> Path:
    if production_enabled is not False:
        raise ValueError("Cross-Asset feature production writes are disabled")
    root = resolve_project_path(output_root, project_root=project_root)
    parts = set(normalized_path_parts(root))
    if parts.intersection(PRODUCTION_LIKE_WRITE_PARTS):
        raise ValueError("Cross-Asset feature output root is production-like and is not allowed")
    allowed_report_root = has_path_parts(root, ("reports", "regimes", "foundation"))
    allowed_codex_root = "_codex_artifacts" in parts
    allowed_sandbox_root = "sandbox" in parts
    if not (allowed_report_root or allowed_codex_root or allowed_sandbox_root):
        raise ValueError("Cross-Asset feature output root must be a report root or sandbox root")
    return root


def _row_payload(row: CrossAssetRelationshipFeatureRow | Mapping[str, Any]) -> dict[str, Any]:
    return row.as_dict() if isinstance(row, CrossAssetRelationshipFeatureRow) else dict(row)


def _write_record(
    artifact_kind: str,
    format_name: str,
    path: Path,
    row_count: int,
    *,
    fallback_reason: str | None = None,
) -> CrossAssetFeaturePartitionWrite:
    return CrossAssetFeaturePartitionWrite(
        artifact_kind=artifact_kind,
        status=CROSS_ASSET_FEATURE_WRITE_STATUS_WRITTEN,
        format=format_name,
        path=path,
        row_count=row_count,
        fallback_reason=fallback_reason,
    )


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.write_text("".join(json.dumps(to_jsonable(row), sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _tabular_row(row: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in row.items():
        if isinstance(value, Mapping) or isinstance(value, (list, tuple)):
            out[key] = json.dumps(to_jsonable(value), sort_keys=True)
        else:
            out[key] = value
    return out


def _relative_path(path: Path, root: Path) -> str:
    return validate_portable_relative_path(path.resolve().relative_to(root.resolve()).as_posix())


__all__ = [
    "CROSS_ASSET_FEATURE_WRITE_STATUS_NO_ROWS",
    "CROSS_ASSET_FEATURE_WRITE_STATUS_WRITTEN",
    "CrossAssetFeaturePartitionWrite",
    "CrossAssetFeatureWriteResult",
    "cross_asset_feature_expected_output_paths",
    "validate_cross_asset_feature_output_root",
    "write_cross_asset_feature_outputs",
]
