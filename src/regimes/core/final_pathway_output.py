from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from src.regimes.asset_state.output_contract import clustered_asset_state_output_rows
from src.regimes.asset_state.sandbox_writer import AssetStateSandboxWriteRequest, write_asset_state_sandbox_outputs
from src.regimes.core.artifact_inventory import build_artifact_inventory, find_unsafe_path_strings
from src.regimes.core.artifact_refs import make_artifact_ref, validate_portable_artifact_ref
from src.regimes.core.forecaster_handoff import (
    ARTIFACT_KIND_ASSET_STATE_SANDBOX_LABELS,
    ARTIFACT_KIND_CROSS_ASSET_FEATURE_ROWS,
    ARTIFACT_KIND_MARKET_STATE_AXIS_PANEL,
    ARTIFACT_KIND_MARKET_STATE_FEATURE_PANEL,
    ARTIFACT_KIND_RELATIONSHIP_DISCOVERY_ARTIFACTS,
    PATHWAY_ASSET_STATE,
    PATHWAY_CROSS_ASSET,
    PATHWAY_MARKET_STATE,
    artifact_ref_for_handoff,
    make_regime_forecaster_handoff_manifest,
    write_regime_forecaster_handoff_manifest,
)
from src.regimes.core.handoff_index import (
    build_regime_forecaster_handoff_index,
    write_regime_forecaster_handoff_index,
)
from src.regimes.core.path_safety import resolve_regime_report_root, validate_report_root
from src.regimes.core.serialization import to_jsonable
from src.regimes.core.source_registry import RegimeSourceRegistryConfig, build_regime_source_registry
from src.regimes.market_state.axis_panel import MARKET_STATE_AXIS_PANEL_SCHEMA_ID
from src.regimes.market_state.feature_writer import (
    MARKET_STATE_FEATURE_WRITE_FORMAT_JSONL,
    MarketStateFeatureMaterializationRequest,
    write_market_state_v1_feature_materialization,
)
from src.regimes.regime_features.cross_asset_feature_generator import build_cross_asset_feature_rows_from_handoff
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


FINAL_PATHWAY_REPORT_SUBDIR = "final_pathway_output"
FINAL_REPORT_FILENAME = "final_regime_pathway_output_report.md"
FINAL_SUMMARY_FILENAME = "final_regime_pathway_output_summary.json"
FINAL_INVENTORY_FILENAME = "artifact_inventory.json"
FINAL_SOURCE_REGISTRY_FILENAME = "source_registry_diagnostics.json"
FINAL_FORECASTER_HANDOFF_INDEX_FILENAME = "forecaster_handoff_index.json"


@dataclass(frozen=True)
class FinalRegimePathwayOutputConfig:
    report_root: str | Path | None = None
    project_root: str | Path | None = None
    run_id: str = "final_regime_pathway_output_sandbox"
    assets: Sequence[str] = ("BTCUSD", "ETHUSD", "SOLUSD")
    interval: int = 240
    window: int = 12
    refit_key: str = "final_pathway_refit_2024_04_24"
    write_outputs: bool = True
    production_enabled: bool = False
    source_ohlcvt_root: str | Path | None = None
    source_feature_root: str | Path | None = None
    source_regime_feature_root: str | Path | None = None
    market_state_universe_manifest_root: str | Path | None = None
    relationship_discovery_root: str | Path | None = None
    universe_eligibility_snapshot_root: str | Path | None = None
    cli_args: Mapping[str, Any] = field(default_factory=dict)
    source_manifest: Mapping[str, Any] | str | Path | None = None
    runtime_config: Mapping[str, Any] | str | Path | None = None
    env: Mapping[str, str] | None = None
    profile: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.production_enabled is not False:
            raise ValueError("Final Regime pathway output runner production writes are disabled")
        object.__setattr__(self, "assets", tuple(str(asset).strip() for asset in self.assets if str(asset).strip()))
        if len(self.assets) < 3:
            raise ValueError("Final Regime pathway output runner requires at least three bounded assets")


@dataclass(frozen=True)
class FinalRegimePathwayOutputResult:
    verdict: str
    report_root: Path
    final_report_path: Path
    summary_path: Path
    inventory_path: Path
    source_registry_path: Path
    bounded_runner_succeeded: bool
    asset_state_outputs_produced: bool
    market_state_outputs_produced: bool
    cross_asset_feature_outputs_produced: bool
    unified_forecaster_handoff_manifests_produced: bool
    disk_safety_validation_passed: bool
    production_writes_or_promotions_performed: bool
    hardcoded_absolute_paths_introduced: bool
    artifact_counts: Mapping[str, int]
    remaining_blockers: Sequence[str] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "artifact_kind": "final_regime_pathway_output_result",
            "schema_version": 1,
            "verdict": self.verdict,
            "report_root": "runtime_only_not_serialized",
            "final_report_path": self.final_report_path.name,
            "summary_path": self.summary_path.name,
            "inventory_path": self.inventory_path.name,
            "source_registry_path": self.source_registry_path.name,
            "bounded_runner_succeeded": bool(self.bounded_runner_succeeded),
            "asset_state_outputs_produced": bool(self.asset_state_outputs_produced),
            "market_state_outputs_produced": bool(self.market_state_outputs_produced),
            "cross_asset_feature_outputs_produced": bool(self.cross_asset_feature_outputs_produced),
            "unified_forecaster_handoff_manifests_produced": bool(self.unified_forecaster_handoff_manifests_produced),
            "artifact_inventory_disk_safety_validation_passed": bool(self.disk_safety_validation_passed),
            "production_writes_or_promotions_performed": bool(self.production_writes_or_promotions_performed),
            "hardcoded_absolute_paths_introduced": bool(self.hardcoded_absolute_paths_introduced),
            "artifact_counts": dict(self.artifact_counts),
            "remaining_blockers": list(self.remaining_blockers),
        }


def default_final_pathway_output_report_root(
    *,
    project_root: str | Path | None = None,
) -> Path:
    return resolve_regime_report_root(FINAL_PATHWAY_REPORT_SUBDIR, project_root=project_root)


def run_final_regime_pathway_output(
    config: FinalRegimePathwayOutputConfig | None = None,
) -> FinalRegimePathwayOutputResult:
    cfg = config or FinalRegimePathwayOutputConfig()
    report_root = (
        validate_report_root(cfg.report_root, project_root=cfg.project_root, allow_foundation_descendant=True)
        if cfg.report_root is not None
        else default_final_pathway_output_report_root(project_root=cfg.project_root)
    )
    report_root.mkdir(parents=True, exist_ok=True)
    source_registry = build_regime_source_registry(
        RegimeSourceRegistryConfig(
            explicit_roots=_explicit_source_roots(cfg),
            cli_args=cfg.cli_args,
            manifest=cfg.source_manifest,
            runtime_config=cfg.runtime_config,
            env=cfg.env,
            profile=cfg.profile,
            project_root=cfg.project_root,
            output_root=report_root,
            assets=cfg.assets,
            intervals=(cfg.interval,),
        )
    )
    source_registry_path = _write_json(report_root / FINAL_SOURCE_REGISTRY_FILENAME, source_registry.as_dict())

    now = datetime.now(timezone.utc).isoformat()
    pipeline_manifest_path = _write_json(
        report_root / "pipeline_inputs" / "pipeline_input_manifest.json",
        _pipeline_input_manifest(cfg, created_at=now),
    )
    asset_result = write_asset_state_sandbox_outputs(
        AssetStateSandboxWriteRequest(
            rows=_asset_state_rows(cfg, created_at=now),
            output_root=report_root / "asset_state_test",
            run_id=cfg.run_id,
            file_format="jsonl",
        )
    )
    market_result = write_market_state_v1_feature_materialization(
        MarketStateFeatureMaterializationRequest(
            output_root=report_root / "market_state_axis_panels",
            run_id=cfg.run_id,
            market_feature_rows={"market_return_summary": _market_feature_rows(cfg)},
            axis_panel_rows={"market_return_state": _market_axis_rows(cfg)},
            universe_manifest_reference=_portable_ref(pipeline_manifest_path, report_root=report_root, kind="pipeline_input_manifest"),
            file_format=MARKET_STATE_FEATURE_WRITE_FORMAT_JSONL,
            metadata={"bounded_final_pathway_runner": True},
        )
    )
    relationship_root = report_root / "relationship_discovery_v1"
    relationship_write = write_relationship_discovery_v1_artifacts(
        _relationship_rows(cfg),
        output_root=relationship_root,
        run_id=cfg.run_id,
        prefer_parquet=False,
        allow_json_fallback_for_tests=True,
        production_enabled=False,
        project_root=cfg.project_root,
    )
    scoreboard_path = _write_json(
        relationship_root / "relationship_scoreboard.json",
        {
            "artifact_kind": "relationship_scoreboard",
            "schema_version": 1,
            "run_id": cfg.run_id,
            "status": "bounded_shape_proof",
            "selected_edge_count": 2,
            "production_enabled": False,
            "production_outputs_written": False,
            "broad_all_to_all": False,
        },
    )
    handoff = build_process1_to_process2_handoff_manifest(
        relationship_write,
        run_id=cfg.run_id,
        relationship_scoreboard_path=scoreboard_path,
    )
    handoff_path = write_process1_to_process2_handoff_manifest(handoff, output_root=relationship_root)
    cross_asset = build_cross_asset_feature_rows_from_handoff(
        handoff_path,
        output_root=relationship_root,
        write_outputs=True,
        prefer_parquet=False,
        allow_json_fallback_for_tests=True,
        production_enabled=False,
        project_root=cfg.project_root,
    )
    forecaster_handoff_paths = _write_forecaster_handoffs(
        cfg,
        report_root=report_root,
        pipeline_manifest_path=pipeline_manifest_path,
        asset_result=asset_result.as_dict(),
        market_result=market_result.as_dict(),
        relationship_write_paths=relationship_write.written_paths,
        handoff_path=handoff_path,
        cross_asset_write_paths=cross_asset.write_result.written_paths if cross_asset.write_result is not None else (),
    )
    forecaster_index = build_regime_forecaster_handoff_index(
        forecaster_handoff_paths,
        artifact_root=report_root,
        manifests=tuple(json.loads(path.read_text(encoding="utf-8")) for path in forecaster_handoff_paths),
        consumer_notes="Combined index for future Regime forecasters; no forecaster implementation is included.",
    )
    forecaster_index_path = write_regime_forecaster_handoff_index(
        forecaster_index,
        output_root=report_root,
        relative_path=FINAL_FORECASTER_HANDOFF_INDEX_FILENAME,
        artifact_root=report_root,
        write_outputs=True,
    )

    unified_handoff_path = _write_json(
        report_root / "unified_forecaster_handoff_manifest.json",
        _unified_forecaster_handoff(
            cfg,
            report_root=report_root,
            pipeline_manifest_path=pipeline_manifest_path,
            asset_result=asset_result.as_dict(),
            market_manifest_path=market_result.manifest_path,
            relationship_handoff_path=handoff_path,
            cross_asset_manifest=cross_asset.manifest.as_dict(),
            forecaster_handoff_index_path=forecaster_index_path,
        ),
    )
    inventory_payload = build_artifact_inventory(
        _files_under(report_root),
        artifact_root=report_root,
        report_root=report_root,
        inventory_id=f"{cfg.run_id}_inventory",
        producer="src.regimes.core.final_pathway_output",
    )
    inventory_path = _write_json(report_root / FINAL_INVENTORY_FILENAME, inventory_payload)

    unsafe_summary_paths = find_unsafe_path_strings(_summary_validation_payload(report_root))
    blockers = tuple(unsafe_summary_paths)
    artifact_counts = {
        "asset_state": len(asset_result.artifact_paths),
        "market_state": len(market_result.written_paths),
        "relationship_discovery": len(relationship_write.written_paths),
        "cross_asset": len(cross_asset.write_result.written_paths) if cross_asset.write_result is not None else 0,
        "forecaster_handoff_manifests": len(forecaster_handoff_paths),
        "forecaster_handoff_index": 1 if forecaster_index_path.is_file() else 0,
        "unified_handoff": 1 if unified_handoff_path.is_file() else 0,
        "inventory": 1 if inventory_path.is_file() else 0,
    }
    asset_ok = asset_result.row_count > 0 and all(Path(path).is_file() for path in asset_result.artifact_paths.values())
    market_ok = market_result.axis_panel_row_count > 0 and market_result.manifest_path is not None and market_result.manifest_path.is_file()
    cross_ok = cross_asset.write_result is not None and cross_asset.write_result.row_count > 0
    unified_ok = unified_handoff_path.is_file()
    safety_ok = not blockers
    passed = asset_ok and market_ok and cross_ok and unified_ok and safety_ok
    verdict = "PASSED" if passed else ("YELLOW" if asset_ok or market_ok or cross_ok else "RED")

    result = FinalRegimePathwayOutputResult(
        verdict=verdict,
        report_root=report_root,
        final_report_path=report_root / FINAL_REPORT_FILENAME,
        summary_path=report_root / FINAL_SUMMARY_FILENAME,
        inventory_path=inventory_path,
        source_registry_path=source_registry_path,
        bounded_runner_succeeded=passed,
        asset_state_outputs_produced=asset_ok,
        market_state_outputs_produced=market_ok,
        cross_asset_feature_outputs_produced=cross_ok,
        unified_forecaster_handoff_manifests_produced=unified_ok,
        disk_safety_validation_passed=safety_ok,
        production_writes_or_promotions_performed=False,
        hardcoded_absolute_paths_introduced=False,
        artifact_counts=artifact_counts,
        remaining_blockers=blockers,
    )
    summary_path = _write_json(report_root / FINAL_SUMMARY_FILENAME, result.as_dict())
    final_report_path = _write_report(report_root / FINAL_REPORT_FILENAME, result, cfg)
    return FinalRegimePathwayOutputResult(
        **{
            **result.__dict__,
            "summary_path": summary_path,
            "final_report_path": final_report_path,
        }
    )


def _pipeline_input_manifest(cfg: FinalRegimePathwayOutputConfig, *, created_at: str) -> dict[str, Any]:
    return {
        "artifact_kind": "final_regime_pathway_pipeline_input_manifest",
        "schema_version": 1,
        "run_id": cfg.run_id,
        "created_at": created_at,
        "assets": list(cfg.assets),
        "source_resolution": {
            "explicit_roots_supported": True,
            "cli_args_supported": True,
            "manifest_fields_supported": True,
            "runtime_config_supported": True,
            "path_config_and_env_supported": True,
        },
        "universe_eligibility": {
            asset: {"eligible": True, "reason": "bounded_sandbox_shape_proof"} for asset in cfg.assets
        },
        "pipeline_input_policy": {
            "local_repo_state_only": True,
            "production_enabled": False,
            "broad_all_to_all": False,
            "dynamic_peer_clusters": False,
            "l2_order_book_required": False,
        },
    }


def _asset_state_rows(cfg: FinalRegimePathwayOutputConfig, *, created_at: str) -> Sequence[Any]:
    return clustered_asset_state_output_rows(
        asset=cfg.assets[0],
        timestamps=(1_713_916_800, 1_713_931_200),
        labels=(0, 1),
        axis="trend",
        band="micro",
        interval=cfg.interval,
        profile_id="bounded_asset_state_shape_profile",
        feature_pool_id="bounded_asset_state_shape_features",
        clusterer_family="kmeans",
        assignment_policy="sandbox_shape_proof",
        refit_key=cfg.refit_key,
        run_id=cfg.run_id,
        created_at=created_at,
    )


def _market_feature_rows(cfg: FinalRegimePathwayOutputConfig) -> pd.DataFrame:
    ts = [1_713_916_800, 1_713_931_200]
    return pd.DataFrame(
        {
            "ts": ts,
            "interval": [cfg.interval, cfg.interval],
            "band": ["micro", "micro"],
            "feature_family_id": ["market_return_summary", "market_return_summary"],
            "feature_set_id": ["bounded_market_return_summary", "bounded_market_return_summary"],
            "known_at_ts": ts,
            "source_tail_ts": ts,
            "lineage_id": ["bounded_market_lineage", "bounded_market_lineage"],
            "schema_version": [1, 1],
            "universe_snapshot_id": [cfg.refit_key, cfg.refit_key],
            "universe_snapshot_hash": ["bounded_universe_hash", "bounded_universe_hash"],
            "known_at": [{"no_lookahead_verified": True}, {"no_lookahead_verified": True}],
            "core_equal_weight_return": [0.01, -0.005],
        }
    )


def _market_axis_rows(cfg: FinalRegimePathwayOutputConfig) -> pd.DataFrame:
    ts = [1_713_916_800, 1_713_931_200]
    return pd.DataFrame(
        {
            "ts": ts,
            "axis": ["market_return_state", "market_return_state"],
            "interval": [cfg.interval, cfg.interval],
            "band": ["micro", "micro"],
            "known_at_ts": ts,
            "source_tail_ts": ts,
            "lineage_id": ["bounded_market_lineage", "bounded_market_lineage"],
            "schema_version": [1, 1],
            "universe_snapshot_id": [cfg.refit_key, cfg.refit_key],
            "universe_snapshot_hash": ["bounded_universe_hash", "bounded_universe_hash"],
            "feature_schema_id": [MARKET_STATE_AXIS_PANEL_SCHEMA_ID, MARKET_STATE_AXIS_PANEL_SCHEMA_ID],
            "axis_panel_version": ["market_state_v1_axis_panel_v1", "market_state_v1_axis_panel_v1"],
            "known_at": [{"no_lookahead_verified": True}, {"no_lookahead_verified": True}],
            "core_equal_weight_return": [0.01, -0.005],
            "candidate_ordinal_state_is_final_label": [False, False],
            "composite_market_state_label_produced": [False, False],
        }
    )


def _relationship_rows(cfg: FinalRegimePathwayOutputConfig) -> dict[str, list[dict[str, Any]]]:
    asset, peer_a, peer_b = cfg.assets[:3]
    base = {
        "refit_key": cfg.refit_key,
        "interval": cfg.interval,
        "window": cfg.window,
        "known_at_ts": "1713931200",
        "source_tail_ts": "1713931200",
        "schema_version": 1,
    }
    return {
        METHOD_MANIFEST_SCHEMA_ID: [
            {
                "method_id": "bounded_residual_corr",
                "method_family": "residual_corr",
                "relationship_family": "residual_peer",
                "source_data": "bounded_sandbox_fixture",
                "interval": cfg.interval,
                "window": cfg.window,
                "k_policy": "fixed_small_k",
                "residualization_policy": "none_shape_proof",
                "normalization_policy": "zscore_shape_proof",
                "thresholds": "bounded_shape_thresholds",
                "universe_scope": "bounded_three_asset",
                "schema_version": 1,
                "generated_at": "2024-04-24T00:00:00+00:00",
                "run_id": cfg.run_id,
            }
        ],
        REFIT_SNAPSHOT_MANIFEST_SCHEMA_ID: [
            {
                **base,
                "effective_start_ts": "1713916800",
                "effective_end_ts": "1713931200",
                "source_tail_ts": "1713931200",
                "anchors": asset,
                "core_assets": f"{asset},{peer_a}",
                "broad_sample_assets": ",".join(cfg.assets),
                "excluded_assets_with_reasons": "",
                "universe_manifest_ref": "pipeline_inputs/pipeline_input_manifest.json",
                "universe_manifest_hash": "bounded_universe_hash",
                "policy_id": "bounded_final_pathway_policy",
            }
        ],
        SELECTED_RELATIONSHIP_EDGES_SCHEMA_ID: [
            _selected_edge(base, asset=asset, related=peer_a, rank=1, value=0.82),
            _selected_edge(base, asset=asset, related=peer_b, rank=2, value=0.63),
        ],
        ASSET_RELATIONSHIP_PROFILES_SCHEMA_ID: [
            {
                **base,
                "asset": asset,
                "corr_to_anchor_primary": 1.0,
                "corr_to_anchor_secondary": 0.72,
                "corr_to_core_basket": 0.85,
                "beta_to_core_basket": 1.05,
                "market_mode_exposure_score": 0.8,
                "residual_peer_signal_score": 0.45,
                "relationship_concentration": 0.4,
                "relationship_entropy": 0.6,
                "top_peer_count": 2,
                "top_peer_stability_mean": 0.7,
                "isolated_asset_score": 0.1,
                "peer_signal_availability_status": "available",
                "lineage_id": "bounded_relationship_lineage",
            }
        ],
        RELATIONSHIP_STABILITY_SCORES_SCHEMA_ID: [
            _stability(base, asset=asset, related=peer_a, strength=0.82),
            _stability(base, asset=asset, related=peer_b, strength=0.63),
        ],
        ISOLATED_ASSET_PROFILES_SCHEMA_ID: [
            {
                **base,
                "asset": asset,
                "isolation_status": "available",
                "isolated_asset_score": 0.1,
                "max_relationship_strength": 0.82,
                "stable_edge_count": 2,
                "candidate_edge_count": 2,
                "coverage": 1.0,
                "reason_codes": "",
                "lineage_id": "bounded_relationship_lineage",
            }
        ],
        EDGE_ALIAS_MANIFEST_SCHEMA_ID: [
            _alias(base, asset=asset, related=peer_a, slot="strongest_peer_slot_1", rank=1, strength=0.82),
            _alias(base, asset=asset, related=peer_b, slot="strongest_peer_slot_2", rank=2, strength=0.63),
        ],
    }


def _selected_edge(base: Mapping[str, Any], *, asset: str, related: str, rank: int, value: float) -> dict[str, Any]:
    return {
        **base,
        "asset": asset,
        "related_asset_or_benchmark": related,
        "relationship_family": "residual_peer",
        "relationship_type": "residual_corr",
        "method_id": "bounded_residual_corr",
        "value": float(value),
        "abs_value": abs(float(value)),
        "direction": "positive",
        "rank": int(rank),
        "slot": f"strongest_peer_slot_{rank}",
        "selected_by_policy": True,
        "sample_count": 12,
        "coverage": 1.0,
        "stability_score": abs(float(value)),
        "activation_status": "active",
        "lineage_id": "bounded_relationship_lineage",
    }


def _stability(base: Mapping[str, Any], *, asset: str, related: str, strength: float) -> dict[str, Any]:
    return {
        **base,
        "asset": asset,
        "related_asset_or_benchmark": related,
        "method_id": "bounded_residual_corr",
        "survival_count": 2,
        "survival_share": 1.0,
        "mean_strength": float(strength),
        "strength_std": 0.0,
        "sign_stability": 1.0,
        "rank_stability": 1.0,
        "activation_status": "active",
        "enough_history": True,
        "stability_reason": "bounded_shape_proof",
        "lineage_id": "bounded_relationship_lineage",
    }


def _alias(base: Mapping[str, Any], *, asset: str, related: str, slot: str, rank: int, strength: float) -> dict[str, Any]:
    return {
        **base,
        "asset": asset,
        "slot": slot,
        "alias_name": slot,
        "related_asset": related,
        "relationship_family": "residual_peer",
        "method_id": "bounded_residual_corr",
        "strength": float(strength),
        "stability_score": float(strength),
        "activation_status": "active",
        "effective_start_ts": "1713916800",
        "effective_end_ts": "1713931200",
        "lineage_id": "bounded_relationship_lineage",
    }


def _unified_forecaster_handoff(
    cfg: FinalRegimePathwayOutputConfig,
    *,
    report_root: Path,
    pipeline_manifest_path: Path,
    asset_result: Mapping[str, Any],
    market_manifest_path: Path | None,
    relationship_handoff_path: Path,
    cross_asset_manifest: Mapping[str, Any],
    forecaster_handoff_index_path: Path,
) -> dict[str, Any]:
    refs = {
        "pipeline_inputs": _portable_ref(pipeline_manifest_path, report_root=report_root, kind="pipeline_input_manifest"),
        "relationship_discovery_handoff": _portable_ref(relationship_handoff_path, report_root=report_root, kind="process1_to_process2_handoff_manifest"),
        "forecaster_handoff_index": _portable_ref(forecaster_handoff_index_path, report_root=report_root, kind="regime_forecaster_handoff_index"),
    }
    if market_manifest_path is not None:
        refs["market_state_feature_manifest"] = _portable_ref(market_manifest_path, report_root=report_root, kind="market_state_feature_manifest")
    for key, raw_path in dict(asset_result.get("artifact_paths") or {}).items():
        refs[f"asset_state_{key}"] = _portable_ref(Path(raw_path), report_root=report_root, kind="asset_state_sandbox_output")
    return {
        "artifact_kind": "unified_regime_forecaster_handoff_manifest",
        "schema_version": 1,
        "run_id": cfg.run_id,
        "artifact_refs": refs,
        "cross_asset_feature_manifest": to_jsonable(dict(cross_asset_manifest)),
        "consumer_contract": {
            "forecaster_ready_sandbox_only": True,
            "production_labels_available": False,
            "production_parquet_available": False,
            "numerics_exports_available": False,
        },
        "production_enabled": False,
        "production_outputs_written": False,
        "production_promotion_performed": False,
    }


def _write_forecaster_handoffs(
    cfg: FinalRegimePathwayOutputConfig,
    *,
    report_root: Path,
    pipeline_manifest_path: Path,
    asset_result: Mapping[str, Any],
    market_result: Mapping[str, Any],
    relationship_write_paths: Sequence[Path],
    handoff_path: Path,
    cross_asset_write_paths: Sequence[Path],
) -> tuple[Path, ...]:
    root = report_root / "forecaster_handoffs"
    source_refs = {
        "pipeline_inputs": _handoff_ref(
            pipeline_manifest_path,
            artifact_kind="pipeline_input_manifest",
            report_root=report_root,
            known_at_ts="1713931200",
            source_tail_ts="1713931200",
        )
    }
    asset_outputs = {
        str(key): _handoff_ref(
            Path(value),
            artifact_kind=ARTIFACT_KIND_ASSET_STATE_SANDBOX_LABELS,
            report_root=report_root,
            known_at_ts="1713931200",
            source_tail_ts="1713931200",
        )
        for key, value in dict(asset_result.get("artifact_paths") or {}).items()
        if Path(value).is_file()
    }
    market_written = [Path(path) for path in market_result.get("written_paths", ()) if Path(path).is_file()]
    market_feature_outputs = {
        f"market_feature_{idx}": _handoff_ref(
            path,
            artifact_kind=ARTIFACT_KIND_MARKET_STATE_FEATURE_PANEL,
            report_root=report_root,
            known_at_ts="1713931200",
            source_tail_ts="1713931200",
        )
        for idx, path in enumerate(market_written)
        if _has_path_part_prefix(path, "regime_features_market_")
    }
    market_axis_outputs = {
        f"market_axis_{idx}": _handoff_ref(
            path,
            artifact_kind=ARTIFACT_KIND_MARKET_STATE_AXIS_PANEL,
            report_root=report_root,
            known_at_ts="1713931200",
            source_tail_ts="1713931200",
        )
        for idx, path in enumerate(market_written)
        if _has_path_part_prefix(path, "market_state_axis_panel_")
    }
    relationship_outputs = {
        f"relationship_{idx}": _handoff_ref(
            path,
            artifact_kind=ARTIFACT_KIND_RELATIONSHIP_DISCOVERY_ARTIFACTS,
            report_root=report_root,
            known_at_ts="1713931200",
            source_tail_ts="1713931200",
        )
        for idx, path in enumerate(relationship_write_paths)
    }
    relationship_outputs["process1_to_process2_handoff"] = _handoff_ref(
        handoff_path,
        artifact_kind=ARTIFACT_KIND_RELATIONSHIP_DISCOVERY_ARTIFACTS,
        report_root=report_root,
        known_at_ts="1713931200",
        source_tail_ts="1713931200",
    )
    cross_asset_outputs = {
        f"cross_asset_{idx}": _handoff_ref(
            path,
            artifact_kind=ARTIFACT_KIND_CROSS_ASSET_FEATURE_ROWS,
            report_root=report_root,
            known_at_ts="1713931200",
            source_tail_ts="1713931200",
        )
        for idx, path in enumerate(cross_asset_write_paths)
    }
    manifests = (
        make_regime_forecaster_handoff_manifest(
            pathway=PATHWAY_ASSET_STATE,
            artifact_kind=ARTIFACT_KIND_ASSET_STATE_SANDBOX_LABELS,
            axis="trend",
            band="micro",
            interval=cfg.interval,
            asset=cfg.assets[0],
            refit_key=cfg.refit_key,
            profile_id="bounded_asset_state_shape_profile",
            source_artifact_refs=source_refs,
            output_artifact_refs=asset_outputs,
            known_at_ts="1713931200",
            source_tail_ts="1713931200",
            lineage_id="bounded_asset_state_lineage",
            consumer_notes="Asset-State sandbox labels only; not production labels.",
        ),
        make_regime_forecaster_handoff_manifest(
            pathway=PATHWAY_MARKET_STATE,
            artifact_kind=ARTIFACT_KIND_MARKET_STATE_FEATURE_PANEL,
            feature_family="market_return_summary",
            band="micro",
            interval=cfg.interval,
            refit_key=cfg.refit_key,
            feature_profile_id="bounded_market_return_summary",
            source_artifact_refs=source_refs,
            output_artifact_refs=market_feature_outputs,
            known_at_ts="1713931200",
            source_tail_ts="1713931200",
            lineage_id="bounded_market_lineage",
            consumer_notes="Market-State feature panel for future forecaster consumption.",
        ),
        make_regime_forecaster_handoff_manifest(
            pathway=PATHWAY_MARKET_STATE,
            artifact_kind=ARTIFACT_KIND_MARKET_STATE_AXIS_PANEL,
            axis="market_return_state",
            band="micro",
            interval=cfg.interval,
            refit_key=cfg.refit_key,
            source_artifact_refs=source_refs,
            output_artifact_refs=market_axis_outputs,
            known_at_ts="1713931200",
            source_tail_ts="1713931200",
            lineage_id="bounded_market_lineage",
            consumer_notes="Market-State axis panel; no composite production labels.",
        ),
        make_regime_forecaster_handoff_manifest(
            pathway=PATHWAY_CROSS_ASSET,
            artifact_kind=ARTIFACT_KIND_RELATIONSHIP_DISCOVERY_ARTIFACTS,
            feature_family="relationship_discovery_v1",
            band="micro",
            interval=cfg.interval,
            refit_key=cfg.refit_key,
            source_artifact_refs=source_refs,
            output_artifact_refs=relationship_outputs,
            known_at_ts="1713931200",
            source_tail_ts="1713931200",
            lineage_id="bounded_relationship_lineage",
            consumer_notes="Relationship Discovery artifacts only; no dynamic peer clusters or final labels.",
        ),
        make_regime_forecaster_handoff_manifest(
            pathway=PATHWAY_CROSS_ASSET,
            artifact_kind=ARTIFACT_KIND_CROSS_ASSET_FEATURE_ROWS,
            feature_family="cross_asset_relationship_v1",
            band="micro",
            interval=cfg.interval,
            refit_key=cfg.refit_key,
            source_artifact_refs=relationship_outputs,
            output_artifact_refs=cross_asset_outputs,
            known_at_ts="1713931200",
            source_tail_ts="1713931200",
            lineage_id="bounded_relationship_lineage",
            consumer_notes="Cross-Asset feature rows only; no Cross-Asset regime labels.",
        ),
    )
    written: list[Path] = []
    for idx, manifest in enumerate(manifests, start=1):
        path = write_regime_forecaster_handoff_manifest(
            manifest,
            output_root=report_root,
            relative_path=root.relative_to(report_root) / f"{idx:02d}_{manifest.pathway}_{manifest.artifact_kind}.json",
            artifact_root=report_root,
            write_outputs=True,
        )
        written.append(path)
    return tuple(written)


def _handoff_ref(
    path: Path,
    *,
    artifact_kind: str,
    report_root: Path,
    known_at_ts: int | float | str,
    source_tail_ts: int | float | str,
) -> dict[str, Any]:
    return artifact_ref_for_handoff(
        path,
        artifact_kind=artifact_kind,
        artifact_root=report_root,
        producer="src.regimes.core.final_pathway_output",
        known_at_ts=known_at_ts,
        source_tail_ts=source_tail_ts,
    )


def _has_path_part_prefix(path: Path, prefix: str) -> bool:
    return any(str(part).startswith(prefix) for part in Path(path).parts)


def _explicit_source_roots(cfg: FinalRegimePathwayOutputConfig) -> dict[str, Any]:
    return {
        key: value
        for key, value in {
            "ohlcvt": cfg.source_ohlcvt_root,
            "scalar_features": cfg.source_feature_root,
            "regime_features": cfg.source_regime_feature_root,
            "market_state_universe_manifest": cfg.market_state_universe_manifest_root,
            "relationship_discovery_artifacts": cfg.relationship_discovery_root,
            "universe_eligibility_snapshot": cfg.universe_eligibility_snapshot_root,
            "report_sandbox_output_root": cfg.report_root,
        }.items()
        if value is not None
    }


def _portable_ref(path: Path, *, report_root: Path, kind: str) -> dict[str, Any]:
    ref = make_artifact_ref(path, artifact_kind=kind, artifact_root=report_root, producer="src.regimes.core.final_pathway_output")
    validate_portable_artifact_ref(ref)
    return ref.as_dict()


def _summary_validation_payload(report_root: Path) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for path in report_root.rglob("*.json"):
        try:
            payload[str(path.relative_to(report_root).as_posix())] = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
    return payload


def _write_report(path: Path, result: FinalRegimePathwayOutputResult, cfg: FinalRegimePathwayOutputConfig) -> Path:
    blockers = [f"- {item}" for item in result.remaining_blockers] if result.remaining_blockers else ["- None."]
    lines = [
        "# Final Regime Pathway Output Report",
        "",
        f"Verdict: {result.verdict}",
        f"Run ID: `{cfg.run_id}`",
        "",
        "## Scope",
        "- Bounded, non-production Regime pathway shape proof.",
        "- No production labels, production parquet roots, promotions, broad benchmarks, dynamic peer clusters, final Cross-Asset labels, Regime forecasters, or Numerics exports were run.",
        "",
        "## Outputs",
        f"- Asset-State sandbox outputs produced: `{result.asset_state_outputs_produced}`",
        f"- Market-State axis-panel outputs produced: `{result.market_state_outputs_produced}`",
        f"- Cross-Asset feature outputs produced: `{result.cross_asset_feature_outputs_produced}`",
        f"- Unified forecaster handoff manifests produced: `{result.unified_forecaster_handoff_manifests_produced}`",
        f"- Artifact inventory / disk-safety validation passed: `{result.disk_safety_validation_passed}`",
        f"- Source registry diagnostics: `{result.source_registry_path.name}`",
        "",
        "## Artifact Counts",
        *[f"- {name}: `{count}`" for name, count in sorted(result.artifact_counts.items())],
        "",
        "## Safety",
        f"- Production writes or promotions performed: `{result.production_writes_or_promotions_performed}`",
        f"- Hardcoded absolute paths introduced: `{result.hardcoded_absolute_paths_introduced}`",
        "- Persistent handoff manifests use portable artifact refs and relative paths under this report root.",
        "",
        "## Files Inspected / Changed",
        "- Inspected existing Asset-State sandbox, Market-State feature, Relationship Discovery handoff, and Cross-Asset feature writer contracts.",
        "- Added portable artifact refs, inventory validation, and this final pathway output runner.",
        "",
        "## Remaining Blockers",
        *blockers,
    ]
    return _write_text(path, "\n".join(lines) + "\n")


def _write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(dict(payload)), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _write_text(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _files_under(root: Path) -> tuple[Path, ...]:
    return tuple(path for path in root.rglob("*") if path.is_file())


__all__ = [
    "FINAL_INVENTORY_FILENAME",
    "FINAL_PATHWAY_REPORT_SUBDIR",
    "FINAL_REPORT_FILENAME",
    "FINAL_SUMMARY_FILENAME",
    "FinalRegimePathwayOutputConfig",
    "FinalRegimePathwayOutputResult",
    "default_final_pathway_output_report_root",
    "run_final_regime_pathway_output",
]
