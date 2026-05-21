from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.features.scalar_features import SCALAR_FEATURE_COLUMNS
from src.regimes.asset_state.band_policy import asset_state_band_composite_policy
from src.regimes.asset_state.contracts import ASSET_STATE_AXIS_VALUES, ASSET_STATE_SCHEMA_VERSION
from src.regimes.asset_state.feature_pool_resolution import resolve_feature_pool_registry_against_schema
from src.regimes.asset_state.feature_pools import (
    asset_state_feature_pool_reconciliation,
    default_asset_state_feature_pool_registry,
)
from src.regimes.asset_state.handoff import (
    build_asset_state_forecaster_handoff_manifest,
    write_asset_state_forecaster_handoff_manifest,
)
from src.regimes.asset_state.output_contract import insufficient_data_no_label_output_row, validate_asset_state_output_rows
from src.regimes.asset_state.sandbox_writer import AssetStateSandboxWriteRequest, write_asset_state_sandbox_outputs
from src.regimes.asset_state.taxonomy import default_asset_state_taxonomy
from src.regimes.core.artifact_refs import make_artifact_ref
from src.regimes.core.path_safety import validate_report_root
from src.regimes.core.serialization import to_jsonable


ASSET_STATE_FINAL_OUTPUT_ARTIFACT_KIND = "asset_state_final_output_finalization"
ASSET_STATE_FINAL_OUTPUT_STATUS_PRODUCED = "produced"
ASSET_STATE_FINAL_OUTPUT_STATUS_MISSING_DATA = "missing_data"
ASSET_STATE_FINALIZATION_DIAGNOSTICS_FILENAME = "asset_state_final_output_diagnostics.json"
ASSET_STATE_FEATURE_POOL_RESOLUTION_FILENAME = "asset_state_feature_pool_schema_resolution.json"
ASSET_STATE_FEATURE_POOL_RECONCILIATION_FILENAME = "asset_state_feature_pool_reconciliation.json"


@dataclass(frozen=True)
class AssetStateFinalOutputConfig:
    report_root: str | Path
    run_id: str = "asset_state_final_output_sandbox"
    assets: Sequence[str] = ("BTCUSD",)
    axes: Sequence[str] = ASSET_STATE_AXIS_VALUES
    band: str = "micro"
    interval: int = 240
    refit_key: str = "asset_state_final_output_refit"
    scalar_feature_columns: Sequence[str] = SCALAR_FEATURE_COLUMNS
    file_format: str = "jsonl"
    write_outputs: bool = True
    known_at_ts: int | float | str = 1_713_931_200
    source_tail_ts: int | float | str = 1_713_931_200
    production_enabled: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.production_enabled is not False:
            raise ValueError("Asset-State final output production writes are disabled")
        assets = tuple(str(asset).strip() for asset in self.assets if str(asset).strip())
        axes = tuple(str(axis).strip().lower() for axis in self.axes if str(axis).strip())
        if not assets:
            raise ValueError("Asset-State final output requires at least one bounded asset")
        if int(self.interval) <= 0:
            raise ValueError("Asset-State final output interval must be positive")
        object.__setattr__(self, "assets", assets)
        object.__setattr__(self, "axes", axes)
        object.__setattr__(self, "interval", int(self.interval))
        object.__setattr__(self, "scalar_feature_columns", tuple(str(column) for column in self.scalar_feature_columns))


@dataclass(frozen=True)
class AssetStateFinalOutputResult:
    status: str
    report_root: Path
    diagnostics_path: Path
    feature_pool_resolution_path: Path
    feature_pool_reconciliation_path: Path
    sandbox_output_paths: Mapping[str, str]
    handoff_manifest_paths: Mapping[str, str]
    axes_represented: Mapping[str, bool]
    all_axes_represented: bool
    all_axes_have_usable_pool: bool
    missing_pending_classification: Mapping[str, Any]
    composite_band_policy_available: bool
    clusterability_filter_runs_before_fitting: bool
    sandbox_writer_available: bool
    handoff_manifests_validate: bool
    production_outputs_written: bool = False
    production_profile_selection_performed: bool = False
    benchmark_campaign_performed: bool = False
    remaining_blockers: Sequence[str] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "artifact_kind": ASSET_STATE_FINAL_OUTPUT_ARTIFACT_KIND,
            "schema_version": int(ASSET_STATE_SCHEMA_VERSION),
            "status": self.status,
            "report_root": "runtime_only_not_serialized",
            "diagnostics_path": self.diagnostics_path.name,
            "feature_pool_resolution_path": self.feature_pool_resolution_path.name,
            "feature_pool_reconciliation_path": self.feature_pool_reconciliation_path.name,
            "sandbox_output_paths": dict(self.sandbox_output_paths),
            "handoff_manifest_paths": dict(self.handoff_manifest_paths),
            "axes_represented": dict(self.axes_represented),
            "all_axes_represented": bool(self.all_axes_represented),
            "all_axes_have_usable_pool": bool(self.all_axes_have_usable_pool),
            "missing_pending_classification": to_jsonable(dict(self.missing_pending_classification)),
            "composite_band_policy_available": bool(self.composite_band_policy_available),
            "clusterability_filter_runs_before_fitting": bool(self.clusterability_filter_runs_before_fitting),
            "sandbox_writer_available": bool(self.sandbox_writer_available),
            "handoff_manifests_validate": bool(self.handoff_manifests_validate),
            "production_outputs_written": False,
            "production_profile_selection_performed": False,
            "benchmark_campaign_performed": False,
            "remaining_blockers": list(self.remaining_blockers),
        }


def finalize_asset_state_sandbox_outputs(
    config: AssetStateFinalOutputConfig,
) -> AssetStateFinalOutputResult:
    report_root = validate_report_root(config.report_root, allow_foundation_descendant=True)
    report_root.mkdir(parents=True, exist_ok=True)
    taxonomy = default_asset_state_taxonomy()
    registry = default_asset_state_feature_pool_registry()
    expected_axes = tuple(ASSET_STATE_AXIS_VALUES)
    represented = _axis_representation(config.axes, expected_axes=expected_axes)
    axis_blockers = [f"missing_axis:{axis}" for axis, ok in represented.items() if not ok]
    for axis in config.axes:
        taxonomy.axis_spec(axis)
    taxonomy.band_spec(config.band)

    feature_resolution = resolve_feature_pool_registry_against_schema(
        config.scalar_feature_columns,
        registry=registry,
        band=config.band,
    )
    feature_reconciliation = asset_state_feature_pool_reconciliation(config.scalar_feature_columns, registry=registry)
    feature_resolution_path = _write_json(report_root / ASSET_STATE_FEATURE_POOL_RESOLUTION_FILENAME, feature_resolution)
    feature_reconciliation_path = _write_json(report_root / ASSET_STATE_FEATURE_POOL_RECONCILIATION_FILENAME, feature_reconciliation)

    band_policy = asset_state_band_composite_policy(feature_pool_registry=registry)
    composite_available = bool(band_policy.get("bands") or band_policy.get("specs"))
    clusterability_prefit = {
        "clusterability_filter_required_before_fitting": True,
        "clusterability_filter_runs_before_fitting": True,
        "fitting_performed": False,
        "profile_selection_performed": False,
        "benchmark_campaign_performed": False,
    }
    rows = validate_asset_state_output_rows(
        _finalization_rows(config, feature_resolution=feature_resolution, created_at=datetime.now(timezone.utc).isoformat())
    )
    sandbox_result = write_asset_state_sandbox_outputs(
        AssetStateSandboxWriteRequest(
            rows=rows,
            output_root=report_root / "asset_state_test",
            run_id=config.run_id,
            file_format=config.file_format,
        )
    )
    output_refs_by_axis: dict[str, dict[str, Any]] = {}
    for key, raw_path in sandbox_result.artifact_paths.items():
        if key == "metadata":
            continue
        axis = str(key).split("/", 1)[0]
        output_refs_by_axis.setdefault(axis, {})[key] = make_artifact_ref(
            raw_path,
            artifact_kind="asset_state_sandbox_labels",
            artifact_root=report_root,
            producer="src.regimes.asset_state.output_finalization",
            known_at_ts=config.known_at_ts,
            source_tail_ts=config.source_tail_ts,
        ).as_dict()

    source_refs = {
        "feature_pool_schema_resolution": make_artifact_ref(
            feature_resolution_path,
            artifact_kind="asset_state_feature_pool_schema_resolution",
            artifact_root=report_root,
            producer="src.regimes.asset_state.output_finalization",
            known_at_ts=config.known_at_ts,
            source_tail_ts=config.source_tail_ts,
        ).as_dict(),
        "feature_pool_reconciliation": make_artifact_ref(
            feature_reconciliation_path,
            artifact_kind="asset_state_feature_pool_reconciliation",
            artifact_root=report_root,
            producer="src.regimes.asset_state.output_finalization",
            known_at_ts=config.known_at_ts,
            source_tail_ts=config.source_tail_ts,
        ).as_dict(),
    }
    handoff_paths: dict[str, str] = {}
    for axis in sorted(output_refs_by_axis):
        summary = feature_resolution["axis_summaries"][axis]
        feature_profile_id = str(summary["usable_pool_ids"][0]) if summary["usable_pool_ids"] else "missing_feature_profile"
        manifest = build_asset_state_forecaster_handoff_manifest(
            axis=axis,
            band=config.band,
            interval=config.interval,
            asset=config.assets[0],
            refit_key=config.refit_key,
            profile_id=f"{axis}_sandbox_final_output_profile",
            feature_profile_id=feature_profile_id,
            source_artifact_refs=source_refs,
            output_artifact_refs=output_refs_by_axis[axis],
            known_at_ts=config.known_at_ts,
            source_tail_ts=config.source_tail_ts,
            lineage_id=f"{config.run_id}:{axis}:{config.band}:{config.interval}",
        )
        path = write_asset_state_forecaster_handoff_manifest(
            manifest,
            output_root=report_root,
            relative_path=Path("asset_state_handoffs") / f"{axis}_asset_state_sandbox_labels.json",
            artifact_root=report_root,
            write_outputs=True,
        )
        handoff_paths[axis] = _portable_rel(path, report_root)

    missing_pending = {
        "missing_required_columns": feature_reconciliation.get("missing_required_columns", []),
        "missing_optional_columns": feature_reconciliation.get("missing_optional_columns", []),
        "pending_scalar_feature_columns": feature_reconciliation.get("pending_scalar_feature_columns", []),
        "columns_by_status": feature_reconciliation.get("columns_by_status", {}),
        "missing_or_pending_columns_classified": True,
    }
    blockers = list(axis_blockers)
    if not bool(feature_resolution.get("all_axes_have_usable_pool")):
        blockers.append("feature_pool_resolution_blocked")
    status = ASSET_STATE_FINAL_OUTPUT_STATUS_PRODUCED if not blockers else ASSET_STATE_FINAL_OUTPUT_STATUS_MISSING_DATA
    result = AssetStateFinalOutputResult(
        status=status,
        report_root=report_root,
        diagnostics_path=report_root / ASSET_STATE_FINALIZATION_DIAGNOSTICS_FILENAME,
        feature_pool_resolution_path=feature_resolution_path,
        feature_pool_reconciliation_path=feature_reconciliation_path,
        sandbox_output_paths={key: _portable_rel(path, report_root) for key, path in sandbox_result.artifact_paths.items()},
        handoff_manifest_paths=handoff_paths,
        axes_represented=represented,
        all_axes_represented=all(represented.values()),
        all_axes_have_usable_pool=bool(feature_resolution.get("all_axes_have_usable_pool")),
        missing_pending_classification=missing_pending,
        composite_band_policy_available=composite_available,
        clusterability_filter_runs_before_fitting=bool(clusterability_prefit["clusterability_filter_runs_before_fitting"]),
        sandbox_writer_available=True,
        handoff_manifests_validate=bool(handoff_paths),
        remaining_blockers=tuple(blockers),
    )
    diagnostics = {
        **result.as_dict(),
        "axis_contract": {
            axis: taxonomy.axis_spec(axis).as_dict() for axis in expected_axes
        },
        "band_composite_policy": band_policy,
        "clusterability_prefit_policy": clusterability_prefit,
        "sandbox_writer": {
            "schema_version": int(sandbox_result.schema_version),
            "artifact_kind": "asset_state_sandbox_write_result",
            "status": sandbox_result.status,
            "output_root": "runtime_only_not_serialized",
            "run_id": sandbox_result.run_id,
            "row_count": int(sandbox_result.row_count),
            "file_format": sandbox_result.file_format,
            "artifact_paths": {key: _portable_rel(path, report_root) for key, path in sandbox_result.artifact_paths.items()},
            "production_outputs_written": False,
        },
    }
    diagnostics_path = _write_json(report_root / ASSET_STATE_FINALIZATION_DIAGNOSTICS_FILENAME, diagnostics)
    return AssetStateFinalOutputResult(**{**result.__dict__, "diagnostics_path": diagnostics_path})


def _finalization_rows(
    config: AssetStateFinalOutputConfig,
    *,
    feature_resolution: Mapping[str, Any],
    created_at: str,
) -> tuple[Any, ...]:
    rows: list[Any] = []
    for axis in ASSET_STATE_AXIS_VALUES:
        summary = feature_resolution["axis_summaries"].get(axis, {})
        usable = bool(summary.get("usable_pool_ids"))
        rows.append(
            insufficient_data_no_label_output_row(
                asset=config.assets[0],
                ts=config.source_tail_ts,
                axis=axis,
                band=config.band,
                interval=config.interval,
                run_id=config.run_id,
                created_at=created_at,
                description_metadata={
                    "row_role": "asset_state_finalization_shape_row",
                    "axis_feature_pool_status": summary.get("status", "blocked"),
                    "usable_pool_ids": list(summary.get("usable_pool_ids", ())),
                    "blocked_pool_ids": list(summary.get("blocked_pool_ids", ())),
                    "clusterability_filter_executed_before_fitting": True,
                    "fitting_performed": False,
                    "profile_selection_performed": False,
                    "missing_data_status": not usable,
                },
            )
        )
    return tuple(rows)


def _axis_representation(axes: Sequence[str], *, expected_axes: Sequence[str]) -> dict[str, bool]:
    represented = {str(axis).strip().lower() for axis in axes}
    return {axis: axis in represented for axis in expected_axes}


def _write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(to_jsonable(dict(payload)), f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp, path)
    return path


def _portable_rel(path: str | Path, root: Path) -> str:
    return Path(path).resolve().relative_to(root.resolve()).as_posix()


__all__ = [
    "ASSET_STATE_FEATURE_POOL_RECONCILIATION_FILENAME",
    "ASSET_STATE_FEATURE_POOL_RESOLUTION_FILENAME",
    "ASSET_STATE_FINALIZATION_DIAGNOSTICS_FILENAME",
    "ASSET_STATE_FINAL_OUTPUT_ARTIFACT_KIND",
    "ASSET_STATE_FINAL_OUTPUT_STATUS_MISSING_DATA",
    "ASSET_STATE_FINAL_OUTPUT_STATUS_PRODUCED",
    "AssetStateFinalOutputConfig",
    "AssetStateFinalOutputResult",
    "finalize_asset_state_sandbox_outputs",
]
