from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

from src.regimes.core.path_safety import PRODUCTION_LIKE_WRITE_PARTS
from src.regimes.core.paths import has_path_parts, normalized_path_parts, resolve_project_path
from src.regimes.core.serialization import to_jsonable
from src.regimes.relationship_discovery.artifacts import (
    RelationshipArtifactWriteResult,
    validate_relationship_discovery_report_root,
    write_relationship_json_artifact,
    write_relationship_jsonl_rows,
)
from src.regimes.relationship_discovery.aliases import build_edge_alias_manifest_rows
from src.regimes.relationship_discovery.contracts import (
    IsolatedAssetProfile,
    RelationshipScoreboard,
    RelationshipMethodManifest,
    RelationshipRefitSnapshot,
)
from src.regimes.relationship_discovery.methods import METHOD_FEATURE_DISTANCE, RelationshipMethodResult
from src.regimes.relationship_discovery.partitioning import (
    relationship_discovery_partition_contract,
    relationship_discovery_partition_path,
)
from src.regimes.relationship_discovery.prototype_runner import RelationshipDiscoveryPrototypeResult
from src.regimes.relationship_discovery.schemas import (
    METHOD_MANIFEST_SCHEMA_ID,
    REFIT_SNAPSHOT_MANIFEST_SCHEMA_ID,
    process1_artifact_schemas,
    validate_process1_artifact_rows,
)
from src.regimes.relationship_discovery.selection import RelationshipEdgeSelectionResult
from src.regimes.relationship_discovery.storage_estimate import (
    RelationshipStorageEstimateConfig,
    RelationshipStorageEstimateResult,
    estimate_relationship_storage,
)


RELATIONSHIP_PROTOTYPE_WRITE_STATUS_WRITTEN = "written"
RELATIONSHIP_PROTOTYPE_WRITE_STATUS_NO_ROWS = "no_rows"

RELATIONSHIP_V1_WRITE_STATUS_WRITTEN = "written"
RELATIONSHIP_V1_WRITE_STATUS_NO_ROWS = "no_rows"


@dataclass(frozen=True)
class RelationshipTabularWriteResult:
    logical_name: str
    status: str
    format: str
    path: Path | None
    row_count: int
    fallback_reason: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "logical_name": self.logical_name,
            "status": self.status,
            "format": self.format,
            "path": str(self.path) if self.path is not None else None,
            "row_count": int(self.row_count),
            "fallback_reason": self.fallback_reason,
            "production_enabled": False,
        }


@dataclass(frozen=True)
class RelationshipV1PartitionWrite:
    schema_id: str
    status: str
    format: str
    path: Path
    row_count: int
    fallback_reason: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_id": self.schema_id,
            "status": self.status,
            "format": self.format,
            "path": str(self.path),
            "row_count": int(self.row_count),
            "fallback_reason": self.fallback_reason,
            "production_enabled": False,
            "production_outputs_written": False,
            "broad_all_to_all": False,
            "cross_asset_labels_written": False,
        }


@dataclass(frozen=True)
class RelationshipDiscoveryV1ArtifactWriteResult:
    status: str
    output_root: Path
    writes: Sequence[RelationshipV1PartitionWrite]
    partition_contract: Mapping[str, str]

    @property
    def row_count(self) -> int:
        return sum(write.row_count for write in self.writes)

    @property
    def written_paths(self) -> tuple[Path, ...]:
        return tuple(write.path for write in self.writes)

    def as_dict(self) -> dict[str, Any]:
        return {
            "artifact_kind": "relationship_discovery_v1_artifact_write_result",
            "status": self.status,
            "output_root": str(self.output_root),
            "row_count": int(self.row_count),
            "writes": [write.as_dict() for write in self.writes],
            "partition_contract": dict(self.partition_contract),
            "production_enabled": False,
            "production_outputs_written": False,
            "broad_all_to_all": False,
            "cross_asset_labels_written": False,
        }


def write_relationship_discovery_v1_artifacts(
    rows_by_schema_id: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    output_root: str | Path,
    run_id: str,
    prefer_parquet: bool = True,
    allow_json_fallback_for_tests: bool = False,
    production_enabled: bool = False,
    project_root: str | Path | None = None,
) -> RelationshipDiscoveryV1ArtifactWriteResult:
    if production_enabled is not False:
        raise ValueError("Relationship Discovery v1 production writes are disabled")
    if not prefer_parquet and not allow_json_fallback_for_tests:
        raise ValueError("Relationship Discovery v1 JSONL fallback is test-only or dependency-missing")
    root = _validate_v1_partition_output_root(output_root, project_root=project_root, production_enabled=production_enabled)
    normalized = _normalize_rows(rows_by_schema_id)
    validate_process1_artifact_rows(normalized)
    writes: list[RelationshipV1PartitionWrite] = []
    for schema_id, rows in _ordered_schema_rows(normalized):
        if not rows:
            continue
        if schema_id in {METHOD_MANIFEST_SCHEMA_ID, REFIT_SNAPSHOT_MANIFEST_SCHEMA_ID}:
            writes.extend(_write_json_partitions(schema_id, rows, output_root=root, run_id=run_id))
        else:
            writes.extend(
                _write_row_partitions(
                    schema_id,
                    rows,
                    output_root=root,
                    run_id=run_id,
                    prefer_parquet=prefer_parquet,
                    allow_json_fallback_for_tests=allow_json_fallback_for_tests,
                )
            )
    return RelationshipDiscoveryV1ArtifactWriteResult(
        status=RELATIONSHIP_V1_WRITE_STATUS_WRITTEN if writes else RELATIONSHIP_V1_WRITE_STATUS_NO_ROWS,
        output_root=root,
        writes=tuple(writes),
        partition_contract=relationship_discovery_partition_contract(),
    )


@dataclass(frozen=True)
class RelationshipPrototypeArtifactWriteResult:
    status: str
    output_root: Path
    method_manifest: RelationshipArtifactWriteResult
    refit_snapshot_manifest: RelationshipArtifactWriteResult
    asset_relationship_profiles: RelationshipTabularWriteResult
    selected_relationship_edges: RelationshipTabularWriteResult
    relationship_stability_scores: RelationshipTabularWriteResult
    edge_alias_manifest: RelationshipTabularWriteResult
    isolated_asset_profiles: RelationshipTabularWriteResult
    relationship_scoreboard: RelationshipArtifactWriteResult
    storage_estimate: RelationshipArtifactWriteResult
    storage_estimate_result: RelationshipStorageEstimateResult

    def as_dict(self) -> dict[str, Any]:
        tabular = (
            self.asset_relationship_profiles,
            self.selected_relationship_edges,
            self.relationship_stability_scores,
            self.edge_alias_manifest,
            self.isolated_asset_profiles,
        )
        return {
            "artifact_kind": "relationship_prototype_artifact_write_result",
            "status": self.status,
            "output_root": str(self.output_root),
            "method_manifest": self.method_manifest.as_dict(),
            "refit_snapshot_manifest": self.refit_snapshot_manifest.as_dict(),
            "tabular_outputs": [item.as_dict() for item in tabular],
            "storage_estimate": self.storage_estimate.as_dict(),
            "relationship_scoreboard": self.relationship_scoreboard.as_dict(),
            "storage_estimate_result": self.storage_estimate_result.as_dict(),
            "production_enabled": False,
        }


@dataclass(frozen=True)
class RelationshipMethodManifestCollection:
    manifests: Sequence[RelationshipMethodManifest]
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "artifact_kind": "relationship_method_manifest_collection",
            "method_manifest_count": len(self.manifests),
            "manifests": [manifest.as_dict() for manifest in self.manifests],
            "diagnostics": to_jsonable(dict(self.diagnostics)),
            "production_enabled": False,
        }


def write_relationship_prototype_artifacts(
    prototype_result: RelationshipDiscoveryPrototypeResult,
    selection_result: RelationshipEdgeSelectionResult,
    *,
    output_root: str | Path,
    storage_config: RelationshipStorageEstimateConfig | None = None,
    prefer_parquet: bool = True,
    production_enabled: bool = False,
) -> RelationshipPrototypeArtifactWriteResult:
    root = _validate_prototype_output_root(output_root, production_enabled=production_enabled)
    refit_snapshot = _refit_snapshot(prototype_result)
    method_manifest = write_relationship_json_artifact(
        _method_manifest_collection(prototype_result),
        output_root=root,
        relative_path="method_manifest.json",
        production_enabled=production_enabled,
    )
    refit_manifest = write_relationship_json_artifact(
        _refit_snapshot(prototype_result),
        output_root=root,
        relative_path="refit_snapshot_manifest.json",
        production_enabled=production_enabled,
    )
    profile_rows = _with_source_tail_from_known_at(
        profile.as_dict() for comparison in prototype_result.comparisons.values() for method in comparison.methods for profile in method.profiles
    )
    selected_edge_rows = _with_source_tail_from_known_at(edge.as_dict() for edge in selection_result.selected_edges)
    stability_rows = _with_stability_metadata(selection_result, refit_snapshot=refit_snapshot)
    alias_rows = [row.as_dict() for row in build_edge_alias_manifest_rows(selection_result, refit_snapshot=refit_snapshot)]
    isolated_rows = _with_source_tail_from_known_at(row.as_dict() for row in _isolated_asset_profiles(selection_result, refit_snapshot=refit_snapshot))

    profiles_write = _write_tabular(
        profile_rows,
        output_root=root,
        stem="asset_relationship_profiles",
        logical_name="asset_relationship_profiles",
        prefer_parquet=prefer_parquet,
        production_enabled=production_enabled,
    )
    edges_write = _write_tabular(
        selected_edge_rows,
        output_root=root,
        stem="selected_relationship_edges",
        logical_name="selected_relationship_edges",
        prefer_parquet=prefer_parquet,
        production_enabled=production_enabled,
    )
    stability_write = _write_tabular(
        stability_rows,
        output_root=root,
        stem="relationship_stability_scores",
        logical_name="relationship_stability_scores",
        prefer_parquet=prefer_parquet,
        production_enabled=production_enabled,
    )
    aliases_write = _write_tabular(
        alias_rows,
        output_root=root,
        stem="edge_alias_manifest",
        logical_name="edge_alias_manifest",
        prefer_parquet=prefer_parquet,
        production_enabled=production_enabled,
    )
    isolated_write = _write_tabular(
        isolated_rows,
        output_root=root,
        stem="isolated_asset_profiles",
        logical_name="isolated_asset_profiles",
        prefer_parquet=prefer_parquet,
        production_enabled=production_enabled,
    )
    estimate = estimate_relationship_storage(prototype_result.scope_result.universe, config=storage_config)
    estimate_write = write_relationship_json_artifact(
        estimate,
        output_root=root,
        relative_path="storage_estimate.json",
        production_enabled=production_enabled,
    )
    scoreboard = _relationship_scoreboard(
        prototype_result,
        selection_result,
        refit_snapshot=refit_snapshot,
        artifact_paths={
            "method_manifest": _first_path(method_manifest, root=root),
            "refit_snapshot_manifest": _first_path(refit_manifest, root=root),
            "asset_relationship_profiles": _relative_path_or_none(profiles_write.path, root),
            "selected_relationship_edges": _relative_path_or_none(edges_write.path, root),
            "relationship_stability_scores": _relative_path_or_none(stability_write.path, root),
            "edge_alias_manifest": _relative_path_or_none(aliases_write.path, root),
            "isolated_asset_profiles": _relative_path_or_none(isolated_write.path, root),
            "storage_estimate": _first_path(estimate_write, root=root),
        },
    )
    scoreboard_write = write_relationship_json_artifact(
        scoreboard,
        output_root=root,
        relative_path="relationship_scoreboard.json",
        production_enabled=production_enabled,
    )
    row_count = sum(item.row_count for item in (profiles_write, edges_write, stability_write, aliases_write, isolated_write))
    return RelationshipPrototypeArtifactWriteResult(
        status=RELATIONSHIP_PROTOTYPE_WRITE_STATUS_WRITTEN if row_count > 0 else RELATIONSHIP_PROTOTYPE_WRITE_STATUS_NO_ROWS,
        output_root=root,
        method_manifest=method_manifest,
        refit_snapshot_manifest=refit_manifest,
        asset_relationship_profiles=profiles_write,
        selected_relationship_edges=edges_write,
        relationship_stability_scores=stability_write,
        edge_alias_manifest=aliases_write,
        isolated_asset_profiles=isolated_write,
        relationship_scoreboard=scoreboard_write,
        storage_estimate=estimate_write,
        storage_estimate_result=estimate,
    )


def _with_source_tail_from_known_at(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        enriched = dict(row)
        if "known_at_ts" in enriched and "source_tail_ts" not in enriched:
            enriched["source_tail_ts"] = enriched["known_at_ts"]
        out.append(enriched)
    return out


def _with_stability_metadata(
    selection_result: RelationshipEdgeSelectionResult,
    *,
    refit_snapshot: RelationshipRefitSnapshot,
) -> list[dict[str, Any]]:
    known_at: dict[tuple[str, str, str, int, int], str] = {}
    lineage: dict[tuple[str, str, str, int, int], str] = {}
    for selected in selection_result.candidate_edges:
        edge = selected.edge
        key = (edge.asset, edge.related_asset_or_benchmark, edge.method_id, int(edge.interval), int(edge.window))
        known_at.setdefault(key, str(edge.known_at_ts))
        lineage.setdefault(key, edge.lineage_id)

    rows: list[dict[str, Any]] = []
    for score in selection_result.stability_result.scores:
        row = dict(score.as_dict())
        key = (score.asset, score.related_asset_or_benchmark, score.method_id, int(score.interval), int(score.window))
        row_known_at = known_at.get(key, str(refit_snapshot.known_at_ts))
        row["known_at_ts"] = row_known_at
        row["source_tail_ts"] = row_known_at
        row["lineage_id"] = lineage.get(key, "relationship_stability_score")
        rows.append(row)
    return rows


def _validate_prototype_output_root(output_root: str | Path, *, production_enabled: bool) -> Path:
    root = validate_relationship_discovery_report_root(output_root, production_enabled=production_enabled)
    parts = set(normalized_path_parts(root))
    if parts.intersection(PRODUCTION_LIKE_WRITE_PARTS):
        raise ValueError("Relationship Discovery prototype artifact root is production-like and is not allowed")
    return root


def _validate_v1_partition_output_root(
    output_root: str | Path,
    *,
    project_root: str | Path | None,
    production_enabled: bool,
) -> Path:
    if production_enabled is not False:
        raise ValueError("Relationship Discovery v1 production writes are disabled")
    root = resolve_project_path(output_root, project_root=project_root)
    parts = set(normalized_path_parts(root))
    if parts.intersection(PRODUCTION_LIKE_WRITE_PARTS):
        raise ValueError("Relationship Discovery v1 artifact root is production-like and is not allowed")
    allowed_report_root = has_path_parts(root, ("reports", "regimes", "foundation"))
    allowed_codex_root = "_codex_artifacts" in parts
    allowed_sandbox_root = "sandbox" in parts
    if not (allowed_report_root or allowed_codex_root or allowed_sandbox_root):
        raise ValueError("Relationship Discovery v1 artifact root must be a report root or sandbox root")
    return root


def _normalize_rows(rows_by_schema_id: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, tuple[dict[str, Any], ...]]:
    if not isinstance(rows_by_schema_id, Mapping):
        raise ValueError("Relationship Discovery v1 writer requires rows_by_schema_id mapping")
    out: dict[str, tuple[dict[str, Any], ...]] = {}
    for schema_id, rows in rows_by_schema_id.items():
        if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence):
            raise ValueError(f"Relationship Discovery v1 rows for {schema_id} must be a sequence")
        out[str(schema_id)] = tuple(dict(row) for row in rows)
    return out


def _ordered_schema_rows(rows_by_schema_id: Mapping[str, Sequence[Mapping[str, Any]]]) -> tuple[tuple[str, tuple[dict[str, Any], ...]], ...]:
    schema_order = tuple(process1_artifact_schemas())
    out: list[tuple[str, tuple[dict[str, Any], ...]]] = []
    for schema_id in schema_order:
        rows = tuple(dict(row) for row in rows_by_schema_id.get(schema_id, ()))
        if rows:
            out.append((schema_id, rows))
    unknown = sorted(set(rows_by_schema_id).difference(schema_order))
    if unknown:
        raise ValueError(f"Unknown Relationship Discovery v1 schema ids: {unknown}")
    return tuple(out)


def _write_json_partitions(
    schema_id: str,
    rows: Sequence[Mapping[str, Any]],
    *,
    output_root: Path,
    run_id: str,
) -> tuple[RelationshipV1PartitionWrite, ...]:
    groups: dict[Path, list[dict[str, Any]]] = {}
    for row in rows:
        rel = relationship_discovery_partition_path(schema_id, row, run_id=run_id, format="parquet")
        groups.setdefault(rel, []).append(_with_boundary_flags(row))

    writes: list[RelationshipV1PartitionWrite] = []
    for rel, group_rows in sorted(groups.items(), key=lambda item: str(item[0])):
        path = _resolve_v1_partition_path(output_root, rel)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload: Mapping[str, Any]
        if schema_id == REFIT_SNAPSHOT_MANIFEST_SCHEMA_ID and len(group_rows) == 1:
            payload = group_rows[0]
        else:
            payload = {
                "artifact_kind": f"{schema_id}_collection",
                "schema_id": schema_id,
                "rows": group_rows,
                "row_count": len(group_rows),
                **_boundary_flags(),
            }
        path.write_text(json.dumps(to_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        writes.append(
            RelationshipV1PartitionWrite(
                schema_id=schema_id,
                status=RELATIONSHIP_V1_WRITE_STATUS_WRITTEN,
                format="json",
                path=path,
                row_count=len(group_rows),
            )
        )
    return tuple(writes)


def _write_row_partitions(
    schema_id: str,
    rows: Sequence[Mapping[str, Any]],
    *,
    output_root: Path,
    run_id: str,
    prefer_parquet: bool,
    allow_json_fallback_for_tests: bool,
) -> tuple[RelationshipV1PartitionWrite, ...]:
    groups: dict[Path, list[dict[str, Any]]] = {}
    target_format = "parquet" if prefer_parquet else "jsonl"
    for row in rows:
        rel = relationship_discovery_partition_path(schema_id, row, run_id=run_id, format=target_format)
        groups.setdefault(rel, []).append(_with_boundary_flags(row))

    writes: list[RelationshipV1PartitionWrite] = []
    for rel, group_rows in sorted(groups.items(), key=lambda item: str(item[0])):
        path = _resolve_v1_partition_path(output_root, rel)
        path.parent.mkdir(parents=True, exist_ok=True)
        if prefer_parquet:
            try:
                pd.DataFrame([_tabular_row(row) for row in group_rows]).to_parquet(path, index=False)
                writes.append(
                    RelationshipV1PartitionWrite(
                        schema_id=schema_id,
                        status=RELATIONSHIP_V1_WRITE_STATUS_WRITTEN,
                        format="parquet",
                        path=path,
                        row_count=len(group_rows),
                    )
                )
                continue
            except (ImportError, ModuleNotFoundError) as exc:
                if not allow_json_fallback_for_tests:
                    raise
                jsonl_path = path.with_suffix(".jsonl")
                _write_jsonl_payloads(jsonl_path, group_rows)
                writes.append(
                    RelationshipV1PartitionWrite(
                        schema_id=schema_id,
                        status=RELATIONSHIP_V1_WRITE_STATUS_WRITTEN,
                        format="jsonl",
                        path=jsonl_path,
                        row_count=len(group_rows),
                        fallback_reason=type(exc).__name__,
                    )
                )
                continue
        if not allow_json_fallback_for_tests:
            raise ValueError("Relationship Discovery v1 JSONL fallback is test-only or dependency-missing")
        _write_jsonl_payloads(path, group_rows)
        writes.append(
            RelationshipV1PartitionWrite(
                schema_id=schema_id,
                status=RELATIONSHIP_V1_WRITE_STATUS_WRITTEN,
                format="jsonl",
                path=path,
                row_count=len(group_rows),
                fallback_reason="test_jsonl_fallback",
            )
        )
    return tuple(writes)


def _write_jsonl_payloads(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.write_text("".join(json.dumps(to_jsonable(row), sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _resolve_v1_partition_path(root: Path, relative_path: Path) -> Path:
    if relative_path.is_absolute() or any(part in {"", ".."} for part in relative_path.parts):
        raise ValueError("Relationship Discovery v1 partition path must stay within output root")
    path = root / relative_path
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError("Relationship Discovery v1 writer refusing to write outside output root") from exc
    return path


def _with_boundary_flags(row: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(row)
    payload.update(_boundary_flags())
    return payload


def _boundary_flags() -> dict[str, bool]:
    return {
        "production_enabled": False,
        "production_outputs_written": False,
        "broad_all_to_all": False,
        "cross_asset_labels_written": False,
    }


def _write_tabular(
    rows: Sequence[Mapping[str, Any]],
    *,
    output_root: Path,
    stem: str,
    logical_name: str,
    prefer_parquet: bool,
    production_enabled: bool,
) -> RelationshipTabularWriteResult:
    if not rows:
        return RelationshipTabularWriteResult(
            logical_name=logical_name,
            status=RELATIONSHIP_PROTOTYPE_WRITE_STATUS_NO_ROWS,
            format="none",
            path=None,
            row_count=0,
        )
    if prefer_parquet:
        path = output_root / f"{stem}.parquet"
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            frame = pd.DataFrame([_tabular_row(row) for row in rows])
            frame.to_parquet(path, index=False)
            return RelationshipTabularWriteResult(
                logical_name=logical_name,
                status=RELATIONSHIP_PROTOTYPE_WRITE_STATUS_WRITTEN,
                format="parquet",
                path=path,
                row_count=len(rows),
            )
        except Exception as exc:
            fallback = write_relationship_jsonl_rows(
                rows,
                output_root=output_root,
                relative_path=f"{stem}.jsonl",
                production_enabled=production_enabled,
            )
            return RelationshipTabularWriteResult(
                logical_name=logical_name,
                status=fallback.status,
                format="jsonl",
                path=fallback.written_paths[0] if fallback.written_paths else None,
                row_count=fallback.row_count,
                fallback_reason=type(exc).__name__,
            )
    fallback = write_relationship_jsonl_rows(
        rows,
        output_root=output_root,
        relative_path=f"{stem}.jsonl",
        production_enabled=production_enabled,
    )
    return RelationshipTabularWriteResult(
        logical_name=logical_name,
        status=fallback.status,
        format="jsonl",
        path=fallback.written_paths[0] if fallback.written_paths else None,
        row_count=fallback.row_count,
    )


def _tabular_row(row: Mapping[str, Any]) -> dict[str, Any]:
    payload = to_jsonable(dict(row))
    out: dict[str, Any] = {}
    for key, value in payload.items():
        if isinstance(value, (dict, list, tuple, set)):
            out[str(key)] = json.dumps(to_jsonable(value), sort_keys=True, separators=(",", ":"))
        else:
            out[str(key)] = value
    return out


def _method_manifest_collection(prototype_result: RelationshipDiscoveryPrototypeResult) -> RelationshipMethodManifestCollection:
    manifests = []
    for method in _computed_methods(prototype_result):
        manifests.append(
            RelationshipMethodManifest(
                method_id=method.method_id,
                method_family=method.method_family,
                source_data="feature_panel" if method.method_family == METHOD_FEATURE_DISTANCE else "ohlcvt_returns",
                interval=method.interval,
                rolling_window=method.window,
                universe_scope="anchor_core_bounded_broad_sample",
                residualization_policy=_residualization_policy(method.method_family),
                normalization_policy="log_close_return",
                min_observations=int(method.diagnostics.get("min_observations") or 1),
                min_coverage=float(method.diagnostics.get("min_coverage") or 0.0),
                known_at_policy="closed_source_tail",
            )
        )
    return RelationshipMethodManifestCollection(
        manifests=tuple(manifests),
        diagnostics={
            "prototype_status": prototype_result.status,
            "comparison_intervals": sorted(int(interval) for interval in prototype_result.comparisons),
            "production_writes_enabled": False,
        },
    )


def _refit_snapshot(prototype_result: RelationshipDiscoveryPrototypeResult) -> RelationshipRefitSnapshot:
    universe = prototype_result.scope_result.universe
    if universe is None:
        raise ValueError("Relationship Discovery refit snapshot requires a universe scope")
    starts: list[int] = []
    ends: list[int] = []
    for panel in prototype_result.scope_result.panel_results.values():
        if panel.return_panel is None or panel.return_panel.empty or "ts" not in panel.return_panel.columns:
            continue
        ts = pd.to_numeric(panel.return_panel["ts"], errors="coerce").dropna()
        if ts.empty:
            continue
        starts.append(int(ts.min()))
        ends.append(int(ts.max()))
    snapshot_start = min(starts) if starts else 1
    snapshot_end = max(ends) if ends else snapshot_start
    manifest_ref = _portable_path_from_cwd(universe.manifest_path) if universe.manifest_path is not None else None
    return RelationshipRefitSnapshot(
        refit_key=f"{universe.manifest_id}_{snapshot_end}",
        snapshot_start=snapshot_start,
        snapshot_end=snapshot_end,
        known_at_ts=snapshot_end,
        eligible_assets=universe.selected_assets,
        anchors=universe.anchors,
        core_assets=universe.core_assets,
        broad_sample_assets=universe.broad_sample_assets,
        excluded_assets_with_reasons=_excluded_reasons(universe),
        universe_manifest_ref=manifest_ref,
        universe_manifest_hash=_manifest_hash(universe.manifest_path),
        source_tail_ts=snapshot_end,
    )


def _isolated_asset_profiles(
    selection_result: RelationshipEdgeSelectionResult,
    *,
    refit_snapshot: RelationshipRefitSnapshot,
) -> tuple[IsolatedAssetProfile, ...]:
    candidate_counts: dict[tuple[str, int, int], int] = {}
    stable_counts: dict[tuple[str, int, int], int] = {}
    lineage_lookup: dict[tuple[str, int, int], str] = {}
    known_at_lookup: dict[tuple[str, int, int], Any] = {}
    for selected in selection_result.candidate_edges:
        edge = selected.edge
        key = (edge.asset, int(edge.interval), int(edge.window))
        candidate_counts[key] = candidate_counts.get(key, 0) + 1
        lineage_lookup.setdefault(key, edge.lineage_id)
        known_at_lookup.setdefault(key, edge.known_at_ts)
        if selected.selected:
            stable_counts[key] = stable_counts.get(key, 0) + 1

    all_assets = sorted(selection_result.stability_result.asset_statuses)
    intervals = sorted({key[1] for key in candidate_counts} or {int(value) for value in refit_snapshot.as_dict().get("intervals", ()) if value})
    if not intervals:
        intervals = [0]
    windows = sorted({key[2] for key in candidate_counts} or {1})

    rows: list[IsolatedAssetProfile] = []
    for asset in all_assets:
        status = selection_result.stability_result.asset_statuses.get(asset, "isolated_asset")
        for interval in intervals:
            if interval <= 0:
                continue
            asset_windows = [window for window in windows if (asset, interval, window) in candidate_counts] or [max(windows)]
            for window in asset_windows:
                key = (asset, interval, window)
                candidate_count = candidate_counts.get(key, 0)
                stable_count = stable_counts.get(key, 0)
                unavailable = status in {"isolated_asset", "needs_research", "unstable_candidate"}
                rows.append(
                    IsolatedAssetProfile(
                        refit_key=refit_snapshot.refit_key,
                        asset=asset,
                        interval=interval,
                        window=window,
                        isolated_asset_score=1.0 if unavailable or candidate_count == 0 else 0.0,
                        peer_signal_availability_status="available" if stable_count > 0 else status,
                        reason_codes=(status,) if stable_count <= 0 else ("stable_relationship_available",),
                        stable_relationship_count=stable_count,
                        candidate_relationship_count=candidate_count,
                        known_at_ts=known_at_lookup.get(key, refit_snapshot.known_at_ts),
                        lineage_id=lineage_lookup.get(key, "isolated_asset_profile"),
                    )
                )
    return tuple(rows)


def _relationship_scoreboard(
    prototype_result: RelationshipDiscoveryPrototypeResult,
    selection_result: RelationshipEdgeSelectionResult,
    *,
    refit_snapshot: RelationshipRefitSnapshot,
    artifact_paths: Mapping[str, Any],
) -> RelationshipScoreboard:
    intervals = sorted(int(interval) for interval in prototype_result.comparisons)
    windows = sorted(
        {
            int(method.window)
            for comparison in prototype_result.comparisons.values()
            for method in comparison.methods
        }
    )
    return RelationshipScoreboard(
        refit_key=refit_snapshot.refit_key,
        status=selection_result.status,
        selected_edge_count=len(selection_result.selected_edges),
        candidate_edge_count=len(selection_result.candidate_edges),
        isolated_asset_count=len(selection_result.stability_result.isolated_assets),
        unstable_asset_count=len(selection_result.stability_result.noisy_assets),
        intervals=intervals,
        windows=windows,
        artifact_paths=artifact_paths,
        diagnostics={
            "prototype_status": prototype_result.status,
            "selection_diagnostics": selection_result.diagnostics,
            "production_enabled": False,
        },
    )


def _first_path(write: RelationshipArtifactWriteResult, *, root: str | Path | None = None) -> str | None:
    return _relative_path_or_none(write.written_paths[0] if write.written_paths else None, root)


def _relative_path_or_none(path: str | Path | None, root: str | Path | None) -> str | None:
    if path is None:
        return None
    target = Path(path)
    if root is None:
        return target.as_posix()
    return target.resolve().relative_to(Path(root).resolve()).as_posix()


def _portable_path_from_cwd(path: str | Path) -> str:
    target = Path(path)
    try:
        return target.resolve().relative_to(Path(".").resolve()).as_posix()
    except ValueError:
        return target.name


def _computed_methods(prototype_result: RelationshipDiscoveryPrototypeResult) -> tuple[RelationshipMethodResult, ...]:
    return tuple(
        method
        for comparison in prototype_result.comparisons.values()
        for method in comparison.methods
        if method.edges or method.profiles
    )


def _residualization_policy(method_family: str) -> str:
    if "residual" in method_family:
        return "core_basket_ols_residualization"
    if "beta_to_core" in method_family:
        return "core_basket_beta_estimate"
    return "none"


def _excluded_reasons(universe: Any) -> dict[str, str]:
    reasons: dict[str, str] = {}
    for asset in universe.stable_peg_assets_excluded:
        reasons[str(asset)] = "stable_or_peg_panel_excluded"
    for asset in universe.excluded_assets_blocked:
        reasons[str(asset)] = "market_state_excluded_blocked"
    for asset in universe.needs_review_assets_blocked:
        reasons[str(asset)] = "market_state_needs_review_blocked"
    return dict(sorted(reasons.items()))


def _manifest_hash(path: Path | None) -> str | None:
    if path is None or not path.exists() or not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


__all__ = [
    "RELATIONSHIP_PROTOTYPE_WRITE_STATUS_NO_ROWS",
    "RELATIONSHIP_PROTOTYPE_WRITE_STATUS_WRITTEN",
    "RELATIONSHIP_V1_WRITE_STATUS_NO_ROWS",
    "RELATIONSHIP_V1_WRITE_STATUS_WRITTEN",
    "RelationshipDiscoveryV1ArtifactWriteResult",
    "RelationshipMethodManifestCollection",
    "RelationshipPrototypeArtifactWriteResult",
    "RelationshipTabularWriteResult",
    "RelationshipV1PartitionWrite",
    "write_relationship_discovery_v1_artifacts",
    "write_relationship_prototype_artifacts",
]
