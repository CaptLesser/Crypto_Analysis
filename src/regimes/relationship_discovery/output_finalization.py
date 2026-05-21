from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.regimes.core.artifact_refs import make_artifact_ref
from src.regimes.core.forecaster_handoff import (
    ARTIFACT_KIND_CROSS_ASSET_FEATURE_ROWS,
    ARTIFACT_KIND_RELATIONSHIP_DISCOVERY_ARTIFACTS,
)
from src.regimes.core.path_safety import validate_report_root
from src.regimes.core.serialization import to_jsonable
from src.regimes.regime_features.cross_asset import cross_asset_legacy_scaffold_status
from src.regimes.regime_features.cross_asset_feature_generator import build_cross_asset_feature_rows_from_handoff
from src.regimes.regime_features.cross_asset_feature_rows import (
    CROSS_ASSET_FEATURE_ROW_REQUIRED_FIELDS,
    validate_cross_asset_feature_row,
)
from src.regimes.regime_features.cross_asset_handoff import (
    build_cross_asset_forecaster_handoff_manifest,
    write_cross_asset_forecaster_handoff_manifest,
)
from src.regimes.regime_features.pairwise import pairwise_legacy_scaffold_status
from src.regimes.relationship_discovery.canonical import canonical_ownership
from src.regimes.relationship_discovery.defaults import default_relationship_discovery_policy
from src.regimes.relationship_discovery.handoff import (
    build_process1_to_process2_handoff_manifest,
    write_process1_to_process2_handoff_manifest,
)
from src.regimes.relationship_discovery.schemas import (
    ASSET_RELATIONSHIP_PROFILES_SCHEMA_ID,
    EDGE_ALIAS_MANIFEST_SCHEMA_ID,
    ISOLATED_ASSET_PROFILES_SCHEMA_ID,
    METHOD_MANIFEST_SCHEMA_ID,
    REFIT_SNAPSHOT_MANIFEST_SCHEMA_ID,
    RELATIONSHIP_STABILITY_SCORES_SCHEMA_ID,
    SELECTED_RELATIONSHIP_EDGES_SCHEMA_ID,
)
from src.regimes.relationship_discovery.writer import write_relationship_discovery_v1_artifacts


CROSS_ASSET_FINAL_OUTPUT_ARTIFACT_KIND = "cross_asset_final_output_finalization"
CROSS_ASSET_FINAL_OUTPUT_STATUS_PRODUCED = "produced"
CROSS_ASSET_FINAL_OUTPUT_STATUS_MISSING_DATA = "missing_data"
CROSS_ASSET_FINALIZATION_DIAGNOSTICS_FILENAME = "cross_asset_final_output_diagnostics.json"
CROSS_ASSET_POLICY_MANIFEST_FILENAME = "relationship_discovery_v1_policy_manifest.json"

PROCESS1_REQUIRED_ARTIFACTS: tuple[str, ...] = (
    METHOD_MANIFEST_SCHEMA_ID,
    REFIT_SNAPSHOT_MANIFEST_SCHEMA_ID,
    SELECTED_RELATIONSHIP_EDGES_SCHEMA_ID,
    ASSET_RELATIONSHIP_PROFILES_SCHEMA_ID,
    RELATIONSHIP_STABILITY_SCORES_SCHEMA_ID,
    ISOLATED_ASSET_PROFILES_SCHEMA_ID,
    EDGE_ALIAS_MANIFEST_SCHEMA_ID,
    "relationship_scoreboard",
)

PROCESS2_REQUIRED_ARTIFACTS: tuple[str, ...] = (
    "process1_to_process2_handoff_manifest",
    "relationship_feature_catalog",
    "cross_asset_feature_rows",
    "cross_asset_feature_manifest",
)


@dataclass(frozen=True)
class CrossAssetFinalOutputConfig:
    report_root: str | Path
    run_id: str = "cross_asset_final_output_sandbox"
    assets: Sequence[str] = ("BTCUSD", "ETHUSD", "SOLUSD")
    band: str = "meso"
    interval: int = 240
    window: int = 90
    refit_key: str = "cross_asset_final_output_refit"
    known_at_ts: int | float | str = 1_713_931_200
    source_tail_ts: int | float | str = 1_713_931_200
    prefer_parquet: bool = False
    allow_json_fallback_for_tests: bool = True
    selected_edge_count: int = 2
    production_enabled: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.production_enabled is not False:
            raise ValueError("Cross-Asset final output production writes are disabled")
        assets = tuple(str(asset).strip() for asset in self.assets if str(asset).strip())
        object.__setattr__(self, "assets", assets)
        object.__setattr__(self, "interval", int(self.interval))
        object.__setattr__(self, "window", int(self.window))
        object.__setattr__(self, "selected_edge_count", int(self.selected_edge_count))
        object.__setattr__(self, "band", str(self.band).strip().lower())


@dataclass(frozen=True)
class CrossAssetFinalOutputResult:
    status: str
    report_root: Path
    diagnostics_path: Path
    policy_manifest_path: Path
    process1_artifact_paths: Mapping[str, str]
    process2_artifact_paths: Mapping[str, str]
    handoff_manifest_paths: Mapping[str, str]
    relationship_discovery_canonical_process1: bool
    legacy_scaffolds_gated: bool
    v1_policies_verified: bool
    residual_selected_edge_path_exists: bool
    market_exposure_fields_exist: bool
    isolation_status_fields_exist: bool
    alias_slot_manifest_exists: bool
    feature_rows_are_row_based: bool
    feature_rows_validate: bool
    handoff_manifests_validate: bool
    cross_asset_regime_labels_created: bool = False
    peer_clusters_implemented: bool = False
    production_outputs_written: bool = False
    remaining_blockers: Sequence[str] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "artifact_kind": CROSS_ASSET_FINAL_OUTPUT_ARTIFACT_KIND,
            "schema_version": 1,
            "status": self.status,
            "report_root": "runtime_only_not_serialized",
            "diagnostics_path": self.diagnostics_path.name,
            "policy_manifest_path": _portable_rel(self.policy_manifest_path, self.report_root),
            "process1_artifact_paths": dict(self.process1_artifact_paths),
            "process2_artifact_paths": dict(self.process2_artifact_paths),
            "handoff_manifest_paths": dict(self.handoff_manifest_paths),
            "relationship_discovery_canonical_process1": bool(self.relationship_discovery_canonical_process1),
            "legacy_scaffolds_gated": bool(self.legacy_scaffolds_gated),
            "v1_policies_verified": bool(self.v1_policies_verified),
            "residual_selected_edge_path_exists": bool(self.residual_selected_edge_path_exists),
            "market_exposure_fields_exist": bool(self.market_exposure_fields_exist),
            "isolation_status_fields_exist": bool(self.isolation_status_fields_exist),
            "alias_slot_manifest_exists": bool(self.alias_slot_manifest_exists),
            "feature_rows_are_row_based": bool(self.feature_rows_are_row_based),
            "feature_rows_validate": bool(self.feature_rows_validate),
            "handoff_manifests_validate": bool(self.handoff_manifests_validate),
            "cross_asset_regime_labels_created": False,
            "peer_clusters_implemented": False,
            "production_outputs_written": False,
            "remaining_blockers": list(self.remaining_blockers),
        }


def finalize_cross_asset_feature_outputs(config: CrossAssetFinalOutputConfig) -> CrossAssetFinalOutputResult:
    report_root = validate_report_root(config.report_root, allow_foundation_descendant=True)
    report_root.mkdir(parents=True, exist_ok=True)
    finalization_root = _finalization_root(report_root)
    finalization_root.mkdir(parents=True, exist_ok=True)
    policy = default_relationship_discovery_policy()
    policy_manifest_path = _write_json(finalization_root / CROSS_ASSET_POLICY_MANIFEST_FILENAME, _policy_manifest(config, policy))
    blockers = _preflight_blockers(config)

    process1_paths: dict[str, str] = {}
    process2_paths: dict[str, str] = {}
    handoff_paths: dict[str, str] = {}
    feature_rows_payload: list[dict[str, Any]] = []
    feature_write_paths: tuple[Path, ...] = ()
    relationship_write_paths: tuple[Path, ...] = ()

    if not blockers:
        relationship_root = finalization_root / "relationship_discovery_v1"
        relationship_write = write_relationship_discovery_v1_artifacts(
            _relationship_rows(config),
            output_root=relationship_root,
            run_id=config.run_id,
            prefer_parquet=config.prefer_parquet,
            allow_json_fallback_for_tests=config.allow_json_fallback_for_tests,
            production_enabled=False,
        )
        relationship_write_paths = relationship_write.written_paths
        scoreboard_path = _write_json(relationship_root / "relationship_scoreboard.json", _scoreboard(config))
        process1_paths = _relationship_artifact_paths(relationship_write_paths, scoreboard_path, report_root=report_root)
        handoff = build_process1_to_process2_handoff_manifest(
            relationship_write,
            run_id=config.run_id,
            relationship_scoreboard_path=scoreboard_path,
        )
        handoff_path = write_process1_to_process2_handoff_manifest(handoff, output_root=relationship_root)
        cross_asset = build_cross_asset_feature_rows_from_handoff(
            handoff_path,
            output_root=finalization_root,
            write_outputs=True,
            prefer_parquet=config.prefer_parquet,
            allow_json_fallback_for_tests=config.allow_json_fallback_for_tests,
            production_enabled=False,
        )
        feature_rows_payload = [row.as_dict() for row in cross_asset.rows]
        feature_write_paths = cross_asset.write_result.written_paths if cross_asset.write_result is not None else ()
        process2_paths = _cross_asset_artifact_paths(
            handoff_path,
            feature_write_paths,
            report_root=report_root,
        )
        source_refs = {
            "policy_manifest": make_artifact_ref(
                policy_manifest_path,
                artifact_kind="relationship_discovery_v1_policy_manifest",
                artifact_root=report_root,
                producer="src.regimes.relationship_discovery.output_finalization",
                known_at_ts=config.known_at_ts,
                source_tail_ts=config.source_tail_ts,
            ).as_dict()
        }
        process1_refs = _refs_for_paths(
            {**process1_paths, "process1_to_process2_handoff_manifest": _portable_rel(handoff_path, report_root)},
            artifact_root=report_root,
            producer="src.regimes.relationship_discovery.output_finalization",
            known_at_ts=config.known_at_ts,
            source_tail_ts=config.source_tail_ts,
        )
        process2_refs = _refs_for_paths(
            process2_paths,
            artifact_root=report_root,
            producer="src.regimes.relationship_discovery.output_finalization",
            known_at_ts=config.known_at_ts,
            source_tail_ts=config.source_tail_ts,
        )
        relationship_manifest = build_cross_asset_forecaster_handoff_manifest(
            artifact_kind=ARTIFACT_KIND_RELATIONSHIP_DISCOVERY_ARTIFACTS,
            feature_family="relationship_discovery_v1",
            band=config.band,
            interval=config.interval,
            refit_key=config.refit_key,
            source_artifact_refs=source_refs,
            output_artifact_refs=process1_refs,
            known_at_ts=config.known_at_ts,
            source_tail_ts=config.source_tail_ts,
            lineage_id=f"{config.run_id}:relationship_discovery_v1",
        )
        relationship_manifest_path = write_cross_asset_forecaster_handoff_manifest(
            relationship_manifest,
            output_root=report_root,
            relative_path=_finalization_relative(report_root, "cross_asset_handoffs", "00_relationship_discovery_artifacts.json"),
            artifact_root=report_root,
            write_outputs=True,
        )
        feature_manifest = build_cross_asset_forecaster_handoff_manifest(
            artifact_kind=ARTIFACT_KIND_CROSS_ASSET_FEATURE_ROWS,
            feature_family="cross_asset_relationship_v1",
            band=config.band,
            interval=config.interval,
            refit_key=config.refit_key,
            feature_profile_id="cross_asset_relationship_v1_bounded_feature_rows",
            source_artifact_refs=process1_refs,
            output_artifact_refs=process2_refs,
            known_at_ts=config.known_at_ts,
            source_tail_ts=config.source_tail_ts,
            lineage_id=f"{config.run_id}:cross_asset_relationship_v1",
        )
        feature_manifest_path = write_cross_asset_forecaster_handoff_manifest(
            feature_manifest,
            output_root=report_root,
            relative_path=_finalization_relative(report_root, "cross_asset_handoffs", "01_cross_asset_feature_rows.json"),
            artifact_root=report_root,
            write_outputs=True,
        )
        handoff_paths = {
            "relationship_discovery_artifacts": _portable_rel(relationship_manifest_path, report_root),
            "cross_asset_feature_rows": _portable_rel(feature_manifest_path, report_root),
        }

    checks = _checks(config, process1_paths=process1_paths, process2_paths=process2_paths, feature_rows=feature_rows_payload)
    for name, ok in checks.items():
        if not ok:
            blockers.append(name)
    status = CROSS_ASSET_FINAL_OUTPUT_STATUS_PRODUCED if not blockers else CROSS_ASSET_FINAL_OUTPUT_STATUS_MISSING_DATA
    result = CrossAssetFinalOutputResult(
        status=status,
        report_root=report_root,
        diagnostics_path=finalization_root / CROSS_ASSET_FINALIZATION_DIAGNOSTICS_FILENAME,
        policy_manifest_path=policy_manifest_path,
        process1_artifact_paths=process1_paths,
        process2_artifact_paths=process2_paths,
        handoff_manifest_paths=handoff_paths,
        relationship_discovery_canonical_process1=checks["relationship_discovery_canonical_process1"],
        legacy_scaffolds_gated=checks["legacy_scaffolds_gated"],
        v1_policies_verified=checks["v1_policies_verified"],
        residual_selected_edge_path_exists=checks["residual_selected_edge_path_exists"],
        market_exposure_fields_exist=checks["market_exposure_fields_exist"],
        isolation_status_fields_exist=checks["isolation_status_fields_exist"],
        alias_slot_manifest_exists=checks["alias_slot_manifest_exists"],
        feature_rows_are_row_based=checks["feature_rows_are_row_based"],
        feature_rows_validate=checks["feature_rows_validate"],
        handoff_manifests_validate=bool(handoff_paths),
        remaining_blockers=tuple(dict.fromkeys(blockers)),
    )
    diagnostics = {
        **result.as_dict(),
        "canonical_ownership": canonical_ownership().as_dict(),
        "legacy_pairwise_scaffold": pairwise_legacy_scaffold_status(),
        "legacy_cross_asset_scaffold": cross_asset_legacy_scaffold_status(),
        "v1_policy": _policy_summary(policy),
        "process1_required_artifacts": list(PROCESS1_REQUIRED_ARTIFACTS),
        "process2_required_artifacts": list(PROCESS2_REQUIRED_ARTIFACTS),
        "relationship_write_count": len(relationship_write_paths),
        "cross_asset_feature_write_count": len(feature_write_paths),
        "cross_asset_feature_rows": feature_rows_payload,
        "required_feature_row_fields": list(CROSS_ASSET_FEATURE_ROW_REQUIRED_FIELDS),
        "artifact_boundary": {
            "production_enabled": False,
            "production_outputs_written": False,
            "cross_asset_regime_labels_created": False,
            "peer_clusters_implemented": False,
            "broad_all_to_all": False,
        },
    }
    diagnostics_path = _write_json(result.diagnostics_path, diagnostics)
    return CrossAssetFinalOutputResult(**{**result.__dict__, "diagnostics_path": diagnostics_path})


def _relationship_rows(config: CrossAssetFinalOutputConfig) -> dict[str, list[dict[str, Any]]]:
    anchor, peer_a, peer_b = config.assets[:3]
    base = {
        "refit_key": config.refit_key,
        "interval": config.interval,
        "window": config.window,
        "known_at_ts": str(config.known_at_ts),
        "source_tail_ts": str(config.source_tail_ts),
        "schema_version": 1,
    }
    return {
        METHOD_MANIFEST_SCHEMA_ID: [
            {
                "method_id": "residual_corr_selected_edge",
                "method_family": "residual_correlation",
                "relationship_family": "residual_peer",
                "source_data": "bounded_sandbox_relationship_discovery_v1",
                "interval": config.interval,
                "window": config.window,
                "k_policy": "k3_primary_with_k5_sensitivity",
                "residualization_policy": "residual_co_movement_selected_edge_path",
                "normalization_policy": "bounded_zscore_shape_proof",
                "thresholds": "bounded_v1_policy_thresholds",
                "universe_scope": "bounded_three_asset_non_production",
                "schema_version": 1,
                "generated_at": "2024-04-24T00:00:00+00:00",
                "run_id": config.run_id,
            }
        ],
        REFIT_SNAPSHOT_MANIFEST_SCHEMA_ID: [
            {
                **base,
                "effective_start_ts": str(int(config.source_tail_ts) - int(config.interval) * 60),
                "effective_end_ts": str(config.source_tail_ts),
                "source_tail_ts": str(config.source_tail_ts),
                "anchors": anchor,
                "core_assets": f"{anchor},{peer_a}",
                "broad_sample_assets": ",".join(config.assets),
                "excluded_assets_with_reasons": "",
                "universe_manifest_ref": "bounded_runtime_universe",
                "universe_manifest_hash": _hash_list(config.assets),
                "policy_id": "relationship_discovery_v1_bounded_final_output_policy",
            }
        ],
        SELECTED_RELATIONSHIP_EDGES_SCHEMA_ID: [
            _selected_edge(base, asset=anchor, related=peer_a, rank=1, value=0.82),
            _selected_edge(base, asset=anchor, related=peer_b, rank=2, value=0.63),
        ][: config.selected_edge_count],
        ASSET_RELATIONSHIP_PROFILES_SCHEMA_ID: [
            {
                **base,
                "asset": anchor,
                "corr_to_anchor_primary": 1.0,
                "corr_to_anchor_secondary": 0.72,
                "corr_to_core_basket": 0.85,
                "beta_to_core_basket": 1.05,
                "market_mode_exposure_score": 0.8,
                "residual_peer_signal_score": 0.45,
                "relationship_concentration": 0.4,
                "relationship_entropy": 0.6,
                "top_peer_count": min(config.selected_edge_count, 3),
                "top_peer_stability_mean": 0.7,
                "isolated_asset_score": 0.1,
                "peer_signal_availability_status": "available",
                "lineage_id": "bounded_cross_asset_relationship_lineage",
            }
        ],
        RELATIONSHIP_STABILITY_SCORES_SCHEMA_ID: [
            _stability(base, asset=anchor, related=peer_a, strength=0.82),
            _stability(base, asset=anchor, related=peer_b, strength=0.63),
        ][: config.selected_edge_count],
        ISOLATED_ASSET_PROFILES_SCHEMA_ID: [
            {
                **base,
                "asset": anchor,
                "isolation_status": "available",
                "isolated_asset_score": 0.1,
                "max_relationship_strength": 0.82,
                "stable_edge_count": config.selected_edge_count,
                "candidate_edge_count": config.selected_edge_count,
                "coverage": 1.0,
                "reason_codes": "",
                "lineage_id": "bounded_cross_asset_relationship_lineage",
            }
        ],
        EDGE_ALIAS_MANIFEST_SCHEMA_ID: [
            _alias(base, asset=anchor, related=peer_a, slot="strongest_peer_slot_1", rank=1, strength=0.82),
            _alias(base, asset=anchor, related=peer_b, slot="strongest_peer_slot_2", rank=2, strength=0.63),
        ][: config.selected_edge_count],
    }


def _selected_edge(base: Mapping[str, Any], *, asset: str, related: str, rank: int, value: float) -> dict[str, Any]:
    return {
        **base,
        "asset": asset,
        "related_asset_or_benchmark": related,
        "relationship_family": "residual_peer",
        "relationship_type": "residual_co_movement",
        "method_id": "residual_corr_selected_edge",
        "value": float(value),
        "abs_value": abs(float(value)),
        "direction": "positive",
        "rank": int(rank),
        "slot": f"strongest_peer_slot_{rank}",
        "selected_by_policy": True,
        "sample_count": 24,
        "coverage": 1.0,
        "stability_score": abs(float(value)),
        "activation_status": "active",
        "lineage_id": "bounded_cross_asset_relationship_lineage",
    }


def _stability(base: Mapping[str, Any], *, asset: str, related: str, strength: float) -> dict[str, Any]:
    return {
        **base,
        "asset": asset,
        "related_asset_or_benchmark": related,
        "method_id": "residual_corr_selected_edge",
        "survival_count": 2,
        "survival_share": 1.0,
        "mean_strength": float(strength),
        "strength_std": 0.0,
        "sign_stability": 1.0,
        "rank_stability": 1.0,
        "activation_status": "active",
        "enough_history": True,
        "stability_reason": "bounded_final_output_shape_proof",
        "lineage_id": "bounded_cross_asset_relationship_lineage",
    }


def _alias(base: Mapping[str, Any], *, asset: str, related: str, slot: str, rank: int, strength: float) -> dict[str, Any]:
    return {
        **base,
        "asset": asset,
        "slot": slot,
        "alias_name": slot,
        "related_asset": related,
        "relationship_family": "residual_peer",
        "method_id": "residual_corr_selected_edge",
        "strength": float(strength),
        "stability_score": float(strength),
        "activation_status": "active",
        "effective_start_ts": "1713916800",
        "effective_end_ts": str(base["known_at_ts"]),
        "lineage_id": "bounded_cross_asset_relationship_lineage",
    }


def _checks(
    config: CrossAssetFinalOutputConfig,
    *,
    process1_paths: Mapping[str, str],
    process2_paths: Mapping[str, str],
    feature_rows: Sequence[Mapping[str, Any]],
) -> dict[str, bool]:
    ownership = canonical_ownership().as_dict()
    pairwise = pairwise_legacy_scaffold_status()
    legacy_cross = cross_asset_legacy_scaffold_status()
    policy = default_relationship_discovery_policy()
    return {
        "relationship_discovery_canonical_process1": ownership["process_1_canonical_module"] == "src.regimes.relationship_discovery",
        "legacy_scaffolds_gated": (
            pairwise.get("production_enabled") is False
            and pairwise.get("broad_all_to_all_enabled") is False
            and legacy_cross.get("execution_enabled") is False
            and legacy_cross.get("cross_asset_clustering_enabled") is False
        ),
        "v1_policies_verified": _v1_policy_ok(policy, config),
        "residual_selected_edge_path_exists": SELECTED_RELATIONSHIP_EDGES_SCHEMA_ID in process1_paths,
        "market_exposure_fields_exist": bool(feature_rows)
        and all(
            field in row
            for row in feature_rows
            for field in (
                "corr_to_anchor_primary",
                "corr_to_anchor_secondary",
                "corr_to_core_basket",
                "beta_to_core_basket",
                "market_mode_exposure_score",
            )
        ),
        "isolation_status_fields_exist": bool(feature_rows)
        and all(
            field in row
            for row in feature_rows
            for field in ("isolated_asset_score", "peer_signal_availability_status", "stable_edge_count", "candidate_edge_count")
        ),
        "alias_slot_manifest_exists": EDGE_ALIAS_MANIFEST_SCHEMA_ID in process1_paths,
        "feature_rows_are_row_based": _feature_rows_are_row_based(feature_rows),
        "feature_rows_validate": bool(feature_rows) and all(_row_validates(row) for row in feature_rows),
        "process1_artifacts_complete": all(name in process1_paths for name in PROCESS1_REQUIRED_ARTIFACTS),
        "process2_artifacts_complete": all(name in process2_paths for name in PROCESS2_REQUIRED_ARTIFACTS),
    }


def _preflight_blockers(config: CrossAssetFinalOutputConfig) -> list[str]:
    blockers: list[str] = []
    if len(config.assets) < 3:
        blockers.append("missing_data:requires_at_least_three_assets_for_bounded_edges")
    if config.selected_edge_count <= 0:
        blockers.append("missing_data:no_selected_relationship_edges")
    if config.interval != 240 or config.band != "meso":
        blockers.append("policy_mismatch:bounded_persistent_interval_must_be_meso_240m")
    return blockers


def _v1_policy_ok(policy: Any, config: CrossAssetFinalOutputConfig) -> bool:
    interval_policy = policy.interval_policy
    k_policy = policy.k_policy
    return (
        int(interval_policy.primary_interval) == 240
        and int(interval_policy.confirmation_interval) == 1440
        and int(interval_policy.probe_interval) == 60
        and interval_policy.sub_hour_enabled is False
        and int(k_policy.primary_k) == 3
        and int(k_policy.sensitivity_k) == 5
        and policy.broad_all_to_all_enabled is False
        and policy.dynamic_peer_clusters_enabled is False
        and policy.cross_asset_regime_labels_enabled is False
        and config.interval == 240
    )


def _feature_rows_are_row_based(rows: Sequence[Mapping[str, Any]]) -> bool:
    if not rows:
        return False
    forbidden_prefixes = ("peer_", "related_asset_", "related_")
    allowed = {"peer_signal_availability_status"}
    for row in rows:
        if row.get("pathway") != "cross_asset":
            return False
        for name in row:
            lower = str(name).lower()
            if name in allowed:
                continue
            if lower.startswith(forbidden_prefixes) or lower.endswith(("_peer_alias", "_peer_strength")):
                return False
    return True


def _row_validates(row: Mapping[str, Any]) -> bool:
    try:
        validate_cross_asset_feature_row(row)
    except ValueError:
        return False
    return True


def _policy_manifest(config: CrossAssetFinalOutputConfig, policy: Any) -> dict[str, Any]:
    return {
        "artifact_kind": "relationship_discovery_v1_policy_manifest",
        "schema_version": 1,
        "run_id": config.run_id,
        "policy": _policy_summary(policy),
        "production_enabled": False,
        "production_outputs_written": False,
    }


def _policy_summary(policy: Any) -> dict[str, Any]:
    return {
        "interval_policy": policy.interval_policy.as_dict(),
        "k_policy": policy.k_policy.as_dict(),
        "method_policy": policy.method_policy.as_dict(),
        "satellite_policy": policy.satellite_policy.as_dict(),
        "production_enabled": False,
        "broad_all_to_all_enabled": False,
        "dynamic_peer_clusters_enabled": False,
        "cross_asset_regime_labels_enabled": False,
    }


def _scoreboard(config: CrossAssetFinalOutputConfig) -> dict[str, Any]:
    return {
        "artifact_kind": "relationship_scoreboard",
        "schema_version": 1,
        "run_id": config.run_id,
        "status": "bounded_cross_asset_final_output_shape_proof",
        "selected_edge_count": int(config.selected_edge_count),
        "relationship_family": "residual_peer",
        "production_enabled": False,
        "production_outputs_written": False,
        "broad_all_to_all": False,
        "cross_asset_labels_written": False,
    }


def _relationship_artifact_paths(paths: Sequence[Path], scoreboard_path: Path, *, report_root: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for path in paths:
        for schema_id in (
            METHOD_MANIFEST_SCHEMA_ID,
            REFIT_SNAPSHOT_MANIFEST_SCHEMA_ID,
            SELECTED_RELATIONSHIP_EDGES_SCHEMA_ID,
            ASSET_RELATIONSHIP_PROFILES_SCHEMA_ID,
            RELATIONSHIP_STABILITY_SCORES_SCHEMA_ID,
            ISOLATED_ASSET_PROFILES_SCHEMA_ID,
            EDGE_ALIAS_MANIFEST_SCHEMA_ID,
        ):
            if schema_id in path.parts:
                out[schema_id] = _portable_rel(path, report_root)
    out["relationship_scoreboard"] = _portable_rel(scoreboard_path, report_root)
    return out


def _cross_asset_artifact_paths(handoff_path: Path, paths: Sequence[Path], *, report_root: Path) -> dict[str, str]:
    out = {"process1_to_process2_handoff_manifest": _portable_rel(handoff_path, report_root)}
    for path in paths:
        parts = path.as_posix().split("/")
        if "relationship_feature_catalog" in parts:
            out["relationship_feature_catalog"] = _portable_rel(path, report_root)
        if "cross_asset_feature_rows" in parts:
            out["cross_asset_feature_rows"] = _portable_rel(path, report_root)
        if "cross_asset_feature_manifest" in parts:
            out["cross_asset_feature_manifest"] = _portable_rel(path, report_root)
    return out


def _refs_for_paths(
    rel_paths: Mapping[str, str],
    *,
    artifact_root: Path,
    producer: str,
    known_at_ts: int | float | str,
    source_tail_ts: int | float | str,
) -> dict[str, Any]:
    return {
        key: make_artifact_ref(
            artifact_root / rel_path,
            artifact_kind=key,
            artifact_root=artifact_root,
            producer=producer,
            known_at_ts=known_at_ts,
            source_tail_ts=source_tail_ts,
        ).as_dict()
        for key, rel_path in rel_paths.items()
    }


def _finalization_root(report_root: Path) -> Path:
    return report_root if report_root.name == "cross_asset_finalization" else report_root / "cross_asset_finalization"


def _finalization_relative(report_root: Path, *parts: str) -> Path:
    return _finalization_root(report_root).resolve().relative_to(report_root.resolve()).joinpath(*parts)


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


def _hash_list(values: Sequence[str]) -> str:
    return hashlib.sha256(json.dumps(list(values), sort_keys=True).encode("utf-8")).hexdigest()[:16]


__all__ = [
    "CROSS_ASSET_FINALIZATION_DIAGNOSTICS_FILENAME",
    "CROSS_ASSET_FINAL_OUTPUT_ARTIFACT_KIND",
    "CROSS_ASSET_FINAL_OUTPUT_STATUS_MISSING_DATA",
    "CROSS_ASSET_FINAL_OUTPUT_STATUS_PRODUCED",
    "CrossAssetFinalOutputConfig",
    "CrossAssetFinalOutputResult",
    "finalize_cross_asset_feature_outputs",
]
