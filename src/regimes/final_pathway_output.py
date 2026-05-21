from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.regimes.asset_state.output_finalization import (
    ASSET_STATE_FINAL_OUTPUT_STATUS_PRODUCED,
    AssetStateFinalOutputConfig,
    finalize_asset_state_sandbox_outputs,
)
from src.regimes.core.artifact_inventory import build_artifact_inventory, validate_artifact_inventory
from src.regimes.core.disk_safety import validate_disk_safety_report
from src.regimes.core.handoff_index import (
    build_regime_forecaster_handoff_index,
    validate_regime_forecaster_handoff_index,
    write_regime_forecaster_handoff_index,
)
from src.regimes.core.path_safety import validate_report_root
from src.regimes.core.serialization import to_jsonable
from src.regimes.core.source_registry import (
    SOURCE_STATUS_FOUND,
    SOURCE_STATUS_PARTIAL,
    RegimeSourceRegistryConfig,
    build_regime_source_registry,
)
from src.regimes.core.test_branch_readiness import build_test_branch_readiness_matrix, write_test_branch_readiness_matrix
from src.regimes.final_pathway_contracts import (
    FINAL_REGIME_PATHWAY_STATUS_BLOCKED_BY_SCHEMA_GAP,
    FINAL_REGIME_PATHWAY_STATUS_BLOCKED_BY_SOURCE_RESOLUTION,
    FINAL_REGIME_PATHWAY_STATUS_BLOCKED_BY_WRITER_GAP,
    FINAL_REGIME_PATHWAY_STATUS_COMPLETED,
    FINAL_REGIME_PATHWAY_STATUS_COMPLETED_WITH_WARNINGS,
    FINAL_REGIME_PATHWAY_STATUS_FAILED,
    FINAL_REGIME_PATHWAY_STATUS_PARTIAL_MISSING_DATA,
    FinalRegimePathwayRunResult,
    FinalRegimePathwaySandboxConfig,
)
from src.regimes.market_state.output_finalization import (
    MARKET_STATE_FINAL_OUTPUT_STATUS_PRODUCED,
    MarketStateFinalOutputConfig,
    finalize_market_state_sandbox_outputs,
)
from src.regimes.relationship_discovery.output_finalization import (
    CROSS_ASSET_FINAL_OUTPUT_STATUS_PRODUCED,
    CrossAssetFinalOutputConfig,
    finalize_cross_asset_feature_outputs,
)


FINAL_REGIME_PATHWAY_RUN_RESULT_FILENAME = "final_regime_pathway_sandbox_run_result.json"
FINAL_REGIME_SOURCE_REGISTRY_FILENAME = "source_registry_diagnostics.json"
FINAL_REGIME_UNIVERSE_ELIGIBILITY_FILENAME = "pipeline_inputs/universe_eligibility_snapshot.json"
FINAL_REGIME_HANDOFF_INDEX_FILENAME = "forecaster_handoff_index.json"
FINAL_REGIME_ARTIFACT_INVENTORY_FILENAME = "artifact_inventory.json"


def run_final_regime_pathway_sandbox_output(
    config: FinalRegimePathwaySandboxConfig | Mapping[str, Any] | None = None,
    **kwargs: Any,
) -> FinalRegimePathwayRunResult:
    cfg = _coerce_config(config, kwargs)
    report_root = validate_report_root(cfg.report_root, allow_foundation_descendant=True)
    bounded_assets = _bounded_assets(cfg.assets, cap=cfg.bounded_asset_cap)

    if not cfg.write_outputs:
        return FinalRegimePathwayRunResult(
            status=FINAL_REGIME_PATHWAY_STATUS_COMPLETED_WITH_WARNINGS,
            report_root=report_root,
            run_id=cfg.run_id,
            warnings=("write_outputs_false_dry_run_only",),
            blockers=(),
            component_summaries={"bounded_inputs": _bounded_input_summary(cfg, bounded_assets)},
        )

    report_root.mkdir(parents=True, exist_ok=True)
    warnings: list[str] = []
    blockers: list[str] = []
    artifact_paths: dict[str, str] = {}
    component_summaries: dict[str, Any] = {"bounded_inputs": _bounded_input_summary(cfg, bounded_assets)}

    try:
        registry = build_regime_source_registry(_source_registry_config(cfg, bounded_assets, report_root))
        source_registry_path = _write_json(report_root / FINAL_REGIME_SOURCE_REGISTRY_FILENAME, registry.as_dict())
        artifact_paths["source_registry"] = _rel(source_registry_path, report_root)
        component_summaries["source_registry"] = registry.as_dict()

        if cfg.require_real_sources:
            missing_sources = [
                kind
                for kind, diagnostic in registry.diagnostics.items()
                if diagnostic.access_mode == "read" and diagnostic.status not in {SOURCE_STATUS_FOUND, SOURCE_STATUS_PARTIAL}
            ]
            if missing_sources:
                blockers.extend(f"source_resolution_missing:{kind}" for kind in missing_sources)
                result = _result(
                    cfg,
                    report_root,
                    status=FINAL_REGIME_PATHWAY_STATUS_BLOCKED_BY_SOURCE_RESOLUTION,
                    source_registry_path=_rel(source_registry_path, report_root),
                    artifact_paths=artifact_paths,
                    component_summaries=component_summaries,
                    warnings=warnings,
                    blockers=blockers,
                )
                _write_result(report_root, result)
                return result

        eligibility_path, eligibility_warning = _load_or_write_universe_eligibility_snapshot(
            cfg,
            report_root=report_root,
            bounded_assets=bounded_assets,
        )
        artifact_paths["universe_eligibility_snapshot"] = _rel(eligibility_path, report_root)
        if eligibility_warning:
            warnings.append(eligibility_warning)
            blockers.append(eligibility_warning)

        asset_root = report_root / "asset_state_finalization"
        market_root = report_root / "market_state_finalization"
        cross_root = report_root / "cross_asset_finalization"
        market_manifest_path, market_manifest_warning = _load_or_write_market_universe_manifest(cfg, report_root=report_root)
        if market_manifest_warning:
            warnings.append(market_manifest_warning)
            blockers.append(market_manifest_warning)

        asset_band = cfg.bands[0]
        asset_interval = cfg.intervals[0]
        market_band, market_interval = _select_market_band_interval(cfg.bands, cfg.intervals)

        asset_result = finalize_asset_state_sandbox_outputs(
            AssetStateFinalOutputConfig(
                report_root=asset_root,
                run_id=f"{cfg.run_id}_asset_state",
                assets=bounded_assets,
                band=asset_band,
                interval=asset_interval,
                write_outputs=True,
            )
        )
        market_result = finalize_market_state_sandbox_outputs(
            MarketStateFinalOutputConfig(
                report_root=market_root,
                universe_manifest_path=market_manifest_path,
                run_id=f"{cfg.run_id}_market_state",
                band=market_band,
                interval=market_interval,
            )
        )
        cross_result = finalize_cross_asset_feature_outputs(
            CrossAssetFinalOutputConfig(
                report_root=cross_root,
                run_id=f"{cfg.run_id}_cross_asset",
                assets=bounded_assets,
                band="meso",
                interval=240,
            )
        )

        component_summaries["asset_state"] = asset_result.as_dict()
        component_summaries["market_state"] = market_result.as_dict()
        component_summaries["cross_asset"] = cross_result.as_dict()
        _collect_component_paths("asset_state", asset_result.as_dict(), asset_root, report_root, artifact_paths)
        _collect_component_paths("market_state", market_result.as_dict(), market_root, report_root, artifact_paths)
        _collect_component_paths("cross_asset", cross_result.as_dict(), cross_root, report_root, artifact_paths)

        if asset_result.status != ASSET_STATE_FINAL_OUTPUT_STATUS_PRODUCED:
            blockers.extend(f"asset_state:{item}" for item in asset_result.remaining_blockers)
        if market_result.status != MARKET_STATE_FINAL_OUTPUT_STATUS_PRODUCED:
            blockers.extend(f"market_state:{item}" for item in market_result.remaining_blockers)
        if cross_result.status != CROSS_ASSET_FINAL_OUTPUT_STATUS_PRODUCED:
            blockers.extend(f"cross_asset:{item}" for item in cross_result.remaining_blockers)

        handoff_manifest_paths = _handoff_manifest_paths(asset_root, market_root, cross_root, asset_result, market_result, cross_result)
        handoff_index = build_regime_forecaster_handoff_index(
            handoff_manifest_paths,
            artifact_root=report_root,
            index_id=f"{cfg.run_id}_forecaster_handoff_index",
            consumer_notes="Unified non-production handoff index for final Regime pathway sandbox output.",
        )
        validate_regime_forecaster_handoff_index(handoff_index, artifact_root=report_root, write_outputs=True)
        handoff_index_path = write_regime_forecaster_handoff_index(
            handoff_index,
            output_root=report_root,
            relative_path=FINAL_REGIME_HANDOFF_INDEX_FILENAME,
            artifact_root=report_root,
            write_outputs=True,
        )
        artifact_paths["forecaster_handoff_index"] = _rel(handoff_index_path, report_root)
        component_summaries["forecaster_handoff_index"] = handoff_index.as_dict()

        readiness_path = write_test_branch_readiness_matrix(
            build_test_branch_readiness_matrix(),
            output_root=report_root,
        )
        artifact_paths["test_branch_readiness_matrix"] = _rel(readiness_path, report_root)

        status = _status_from_components(blockers=blockers, warnings=warnings)
        result_path = report_root / FINAL_REGIME_PATHWAY_RUN_RESULT_FILENAME
        inventory_path = report_root / FINAL_REGIME_ARTIFACT_INVENTORY_FILENAME
        pre_inventory_result = _result(
            cfg,
            report_root,
            status=status,
            source_registry_path=_rel(source_registry_path, report_root),
            universe_eligibility_snapshot_path=_rel(eligibility_path, report_root),
            forecaster_handoff_index_path=_rel(handoff_index_path, report_root),
            artifact_inventory_path=_rel(inventory_path, report_root),
            test_branch_readiness_matrix_path=_rel(readiness_path, report_root),
            asset_state_status=asset_result.status,
            market_state_status=market_result.status,
            cross_asset_status=cross_result.status,
            asset_state_outputs_produced=asset_result.status == ASSET_STATE_FINAL_OUTPUT_STATUS_PRODUCED,
            market_state_outputs_produced=market_result.status == MARKET_STATE_FINAL_OUTPUT_STATUS_PRODUCED,
            cross_asset_feature_outputs_produced=cross_result.status == CROSS_ASSET_FINAL_OUTPUT_STATUS_PRODUCED,
            unified_forecaster_handoff_manifests_produced=bool(handoff_manifest_paths),
            artifact_paths=artifact_paths,
            component_summaries=component_summaries,
            warnings=warnings,
            blockers=blockers,
        )
        _write_json(result_path, pre_inventory_result.as_dict())

        inventory = build_artifact_inventory(
            _inventory_file_paths(report_root),
            artifact_root=report_root,
            inventory_id=f"{cfg.run_id}_artifact_inventory",
            producer="src.regimes.final_pathway_output",
            report_root=report_root,
        )
        validate_artifact_inventory(inventory)
        validate_disk_safety_report(inventory["disk_safety_report"])
        _write_json(inventory_path, inventory)
        artifact_paths["artifact_inventory"] = _rel(inventory_path, report_root)
        component_summaries["artifact_inventory"] = {
            "artifact_count": inventory["artifact_count"],
            "disk_safety_validation": inventory["disk_safety_validation"],
            "risk_counts": inventory["disk_safety_report"]["risk_counts"],
        }

        final_status = _status_from_components(blockers=blockers, warnings=warnings)
        final_result = replace(
            pre_inventory_result,
            status=final_status,
            bounded_end_to_end_sandbox_runner_succeeded=final_status
            in {FINAL_REGIME_PATHWAY_STATUS_COMPLETED, FINAL_REGIME_PATHWAY_STATUS_COMPLETED_WITH_WARNINGS},
            artifact_inventory_disk_safety_validation_passed=True,
            artifact_paths=artifact_paths,
            component_summaries=component_summaries,
        )
        _write_json(result_path, final_result.as_dict())
        return final_result
    except ValueError as exc:
        message = str(exc)
        if "source" in message.lower() and "root" in message.lower():
            status = FINAL_REGIME_PATHWAY_STATUS_BLOCKED_BY_SOURCE_RESOLUTION
        elif "schema" in message.lower():
            status = FINAL_REGIME_PATHWAY_STATUS_BLOCKED_BY_SCHEMA_GAP
        elif "writer" in message.lower() or "write" in message.lower():
            status = FINAL_REGIME_PATHWAY_STATUS_BLOCKED_BY_WRITER_GAP
        else:
            status = FINAL_REGIME_PATHWAY_STATUS_FAILED
        result = _result(
            cfg,
            report_root,
            status=status,
            artifact_paths=artifact_paths,
            component_summaries=component_summaries,
            warnings=warnings,
            blockers=(*blockers, f"{status}:{message}"),
        )
        _write_result(report_root, result)
        return result


def _coerce_config(
    config: FinalRegimePathwaySandboxConfig | Mapping[str, Any] | None,
    kwargs: Mapping[str, Any],
) -> FinalRegimePathwaySandboxConfig:
    if config is None:
        return FinalRegimePathwaySandboxConfig(**dict(kwargs))
    if isinstance(config, FinalRegimePathwaySandboxConfig):
        return replace(config, **dict(kwargs)) if kwargs else config
    payload = {**dict(config), **dict(kwargs)}
    return FinalRegimePathwaySandboxConfig(**payload)


def _source_registry_config(
    cfg: FinalRegimePathwaySandboxConfig,
    bounded_assets: Sequence[str],
    report_root: Path,
) -> RegimeSourceRegistryConfig:
    payload = dict(cfg.source_registry_config or {})
    explicit_roots = {**dict(cfg.explicit_roots), **dict(payload.pop("explicit_roots", {}))}
    for owned_field in ("output_root", "assets", "intervals"):
        payload.pop(owned_field, None)
    return RegimeSourceRegistryConfig(
        **payload,
        explicit_roots=explicit_roots,
        output_root=report_root,
        assets=tuple(bounded_assets),
        intervals=tuple(cfg.intervals),
    )


def _load_or_write_universe_eligibility_snapshot(
    cfg: FinalRegimePathwaySandboxConfig,
    *,
    report_root: Path,
    bounded_assets: Sequence[str],
) -> tuple[Path, str | None]:
    if cfg.universe_eligibility_snapshot_path is not None:
        explicit = Path(cfg.universe_eligibility_snapshot_path)
        path = report_root / FINAL_REGIME_UNIVERSE_ELIGIBILITY_FILENAME
        if explicit.exists() and explicit.is_file():
            payload = _load_json_mapping(explicit) or _bounded_universe_eligibility_payload(
                cfg,
                bounded_assets,
                source="loaded_explicit_snapshot_unreadable_payload_replaced_with_bounded_shape",
            )
            payload = {**dict(payload), "source": "loaded_explicit_snapshot", "production_enabled": False, "production_outputs_written": False}
            _write_json(path, payload)
            return path, None
        _write_json(path, _bounded_universe_eligibility_payload(cfg, bounded_assets, source="bounded_fallback_after_missing_explicit_snapshot"))
        return path, "universe_eligibility_snapshot_missing_explicit_path"
    path = report_root / FINAL_REGIME_UNIVERSE_ELIGIBILITY_FILENAME
    _write_json(path, _bounded_universe_eligibility_payload(cfg, bounded_assets, source="bounded_sandbox_fixture"))
    return path, None


def _load_or_write_market_universe_manifest(
    cfg: FinalRegimePathwaySandboxConfig,
    *,
    report_root: Path,
) -> tuple[Path | None, str | None]:
    if cfg.market_universe_manifest_path is None:
        return None, None
    explicit = Path(cfg.market_universe_manifest_path)
    if not explicit.exists() or not explicit.is_file():
        return None, "market_universe_manifest_missing_explicit_path"
    payload = _load_json_mapping(explicit)
    if payload is None:
        return None, "market_universe_manifest_unreadable_explicit_path"
    path = report_root / "pipeline_inputs" / "market_state_universe_manifest.json"
    payload = {**dict(payload), "production_enabled": False}
    _write_json(path, payload)
    return path, None


def _bounded_universe_eligibility_payload(
    cfg: FinalRegimePathwaySandboxConfig,
    assets: Sequence[str],
    *,
    source: str,
) -> dict[str, Any]:
    return {
        "artifact_kind": "universe_eligibility_snapshot",
        "schema_version": 1,
        "run_id": cfg.run_id,
        "source": source,
        "assets": [{"asset": asset, "eligible": True, "reason_codes": ["bounded_sandbox_asset"]} for asset in assets],
        "asset_count": len(assets),
        "production_enabled": False,
        "production_outputs_written": False,
    }


def _bounded_assets(assets: Sequence[str], *, cap: int) -> tuple[str, ...]:
    bounded = tuple(str(asset).strip() for asset in assets if str(asset).strip())[: int(cap)]
    return bounded or ("BTCUSD",)


def _select_market_band_interval(bands: Sequence[str], intervals: Sequence[int]) -> tuple[str, int]:
    expected = {"micro": 60, "meso": 240, "macro": 1440}
    requested_intervals = {int(interval) for interval in intervals}
    for band in bands:
        key = str(band).strip().lower()
        if key in expected and expected[key] in requested_intervals:
            return key, expected[key]
    return "micro", 60


def _handoff_manifest_paths(
    asset_root: Path,
    market_root: Path,
    cross_root: Path,
    asset_result: Any,
    market_result: Any,
    cross_result: Any,
) -> tuple[Path, ...]:
    paths: list[Path] = []
    paths.extend(asset_root / rel_path for rel_path in asset_result.handoff_manifest_paths.values())
    paths.extend(market_root / rel_path for rel_path in market_result.handoff_manifest_paths.values())
    paths.extend(cross_root / rel_path for rel_path in cross_result.handoff_manifest_paths.values())
    return tuple(path for path in paths if path.exists())


def _collect_component_paths(
    prefix: str,
    payload: Mapping[str, Any],
    component_root: Path,
    report_root: Path,
    artifact_paths: dict[str, str],
) -> None:
    for key, value in payload.items():
        if key.endswith("_path") and isinstance(value, str) and value and value != "runtime_only_not_serialized":
            path = component_root / value
            if path.exists():
                artifact_paths[f"{prefix}.{key}"] = _rel(path, report_root)
        elif key.endswith("_paths") and isinstance(value, Mapping):
            for name, rel_path in value.items():
                if isinstance(rel_path, str) and rel_path:
                    path = component_root / rel_path
                    if path.exists():
                        artifact_paths[f"{prefix}.{key}.{name}"] = _rel(path, report_root)


def _status_from_components(*, blockers: Sequence[str], warnings: Sequence[str]) -> str:
    if blockers:
        return FINAL_REGIME_PATHWAY_STATUS_PARTIAL_MISSING_DATA
    if warnings:
        return FINAL_REGIME_PATHWAY_STATUS_COMPLETED_WITH_WARNINGS
    return FINAL_REGIME_PATHWAY_STATUS_COMPLETED


def _result(
    cfg: FinalRegimePathwaySandboxConfig,
    report_root: Path,
    *,
    status: str,
    source_registry_path: str | None = None,
    universe_eligibility_snapshot_path: str | None = None,
    forecaster_handoff_index_path: str | None = None,
    artifact_inventory_path: str | None = None,
    test_branch_readiness_matrix_path: str | None = None,
    asset_state_status: str | None = None,
    market_state_status: str | None = None,
    cross_asset_status: str | None = None,
    asset_state_outputs_produced: bool = False,
    market_state_outputs_produced: bool = False,
    cross_asset_feature_outputs_produced: bool = False,
    unified_forecaster_handoff_manifests_produced: bool = False,
    artifact_inventory_disk_safety_validation_passed: bool = False,
    warnings: Sequence[str] = (),
    blockers: Sequence[str] = (),
    artifact_paths: Mapping[str, str] | None = None,
    component_summaries: Mapping[str, Any] | None = None,
) -> FinalRegimePathwayRunResult:
    return FinalRegimePathwayRunResult(
        status=status,
        report_root=report_root,
        run_id=cfg.run_id,
        source_registry_path=source_registry_path,
        universe_eligibility_snapshot_path=universe_eligibility_snapshot_path,
        forecaster_handoff_index_path=forecaster_handoff_index_path,
        artifact_inventory_path=artifact_inventory_path,
        test_branch_readiness_matrix_path=test_branch_readiness_matrix_path,
        asset_state_status=asset_state_status,
        market_state_status=market_state_status,
        cross_asset_status=cross_asset_status,
        bounded_end_to_end_sandbox_runner_succeeded=status
        in {FINAL_REGIME_PATHWAY_STATUS_COMPLETED, FINAL_REGIME_PATHWAY_STATUS_COMPLETED_WITH_WARNINGS},
        asset_state_outputs_produced=asset_state_outputs_produced,
        market_state_outputs_produced=market_state_outputs_produced,
        cross_asset_feature_outputs_produced=cross_asset_feature_outputs_produced,
        unified_forecaster_handoff_manifests_produced=unified_forecaster_handoff_manifests_produced,
        artifact_inventory_disk_safety_validation_passed=artifact_inventory_disk_safety_validation_passed,
        warnings=tuple(warnings),
        blockers=tuple(blockers),
        artifact_paths=dict(artifact_paths or {}),
        component_summaries=dict(component_summaries or {}),
    )


def _bounded_input_summary(cfg: FinalRegimePathwaySandboxConfig, assets: Sequence[str]) -> dict[str, Any]:
    return {
        "assets": list(assets),
        "bounded_asset_cap": int(cfg.bounded_asset_cap),
        "intervals": [int(interval) for interval in cfg.intervals],
        "bands": list(cfg.bands),
        "start_ts": cfg.start_ts,
        "end_ts": cfg.end_ts,
        "clamp_policy": to_jsonable(dict(cfg.clamp_policy)),
        "write_outputs": bool(cfg.write_outputs),
        "production_enabled": False,
        "production_promotion_performed": False,
        "broad_benchmark_run": False,
        "full_universe_heavy_run": False,
        "broad_all_to_all_pairwise_run": False,
        "cross_asset_labels_written": False,
        "forecaster_training_run": False,
    }


def _inventory_file_paths(report_root: Path) -> tuple[Path, ...]:
    excluded = {FINAL_REGIME_ARTIFACT_INVENTORY_FILENAME}
    return tuple(sorted(path for path in report_root.rglob("*") if path.is_file() and path.name not in excluded))


def _write_result(report_root: Path, result: FinalRegimePathwayRunResult) -> Path:
    return _write_json(report_root / FINAL_REGIME_PATHWAY_RUN_RESULT_FILENAME, result.as_dict())


def _write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(to_jsonable(dict(payload)), f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp, path)
    return path


def _load_json_mapping(path: Path) -> Mapping[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, Mapping) else None


def _rel(path: str | Path, root: Path) -> str:
    return Path(path).resolve().relative_to(root.resolve()).as_posix()


__all__ = [
    "FINAL_REGIME_ARTIFACT_INVENTORY_FILENAME",
    "FINAL_REGIME_HANDOFF_INDEX_FILENAME",
    "FINAL_REGIME_PATHWAY_RUN_RESULT_FILENAME",
    "FINAL_REGIME_SOURCE_REGISTRY_FILENAME",
    "FINAL_REGIME_UNIVERSE_ELIGIBILITY_FILENAME",
    "run_final_regime_pathway_sandbox_output",
]
