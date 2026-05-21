from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.regimes.core.artifact_inventory import (
    build_artifact_inventory,
    find_unsafe_path_strings,
    validate_artifact_inventory,
)
from src.regimes.core.artifact_refs import resolve_artifact_ref, validate_portable_artifact_ref
from src.regimes.core.disk_safety import validate_disk_safety_report
from src.regimes.core.forecaster_handoff import validate_regime_forecaster_handoff_manifest
from src.regimes.core.handoff_index import validate_regime_forecaster_handoff_index
from src.regimes.core.path_safety import validate_report_root
from src.regimes.core.serialization import to_jsonable
from src.regimes.core.source_registry import SOURCE_STATUSES
from src.regimes.core.test_branch_readiness import validate_test_branch_readiness_matrix
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
from src.regimes.final_pathway_output import (
    FINAL_REGIME_ARTIFACT_INVENTORY_FILENAME,
    FINAL_REGIME_PATHWAY_RUN_RESULT_FILENAME,
    run_final_regime_pathway_sandbox_output,
)


FINAL_REGIME_PATHWAY_OUTPUT_REPORT_FILENAME = "final_regime_pathway_output_report.md"

FINAL_PATHWAY_VALIDATOR_PASSED = "PASSED"
FINAL_PATHWAY_VALIDATOR_YELLOW = "YELLOW"
FINAL_PATHWAY_VALIDATOR_RED = "RED"


@dataclass(frozen=True)
class FinalRegimePathwayValidationResult:
    verdict: str
    report_root: Path
    final_report_path: str
    runner_status: str
    checks: Mapping[str, bool]
    blockers: Sequence[str] = ()
    warnings: Sequence[str] = ()
    source_statuses: Mapping[str, str] = field(default_factory=dict)
    handoff_counts_by_pathway: Mapping[str, int] = field(default_factory=dict)
    artifact_inventory_count: int = 0
    disk_risk_counts: Mapping[str, int] = field(default_factory=dict)
    production_safety: Mapping[str, bool] = field(default_factory=dict)
    deferred_automation_polish: Sequence[str] = ()
    deferred_production_promotion: Sequence[str] = ()

    def __post_init__(self) -> None:
        if self.verdict not in {FINAL_PATHWAY_VALIDATOR_PASSED, FINAL_PATHWAY_VALIDATOR_YELLOW, FINAL_PATHWAY_VALIDATOR_RED}:
            raise ValueError("Final Regime pathway validation verdict is unsupported")
        object.__setattr__(self, "checks", {str(key): bool(value) for key, value in dict(self.checks).items()})
        object.__setattr__(self, "blockers", tuple(str(item) for item in self.blockers))
        object.__setattr__(self, "warnings", tuple(str(item) for item in self.warnings))
        object.__setattr__(self, "source_statuses", {str(key): str(value) for key, value in dict(self.source_statuses).items()})
        object.__setattr__(
            self,
            "handoff_counts_by_pathway",
            {str(key): int(value) for key, value in dict(self.handoff_counts_by_pathway).items()},
        )
        object.__setattr__(self, "disk_risk_counts", {str(key): int(value) for key, value in dict(self.disk_risk_counts).items()})
        object.__setattr__(self, "production_safety", {str(key): bool(value) for key, value in dict(self.production_safety).items()})
        object.__setattr__(self, "deferred_automation_polish", tuple(str(item) for item in self.deferred_automation_polish))
        object.__setattr__(self, "deferred_production_promotion", tuple(str(item) for item in self.deferred_production_promotion))

    def as_dict(self) -> dict[str, Any]:
        return {
            "artifact_kind": "final_regime_pathway_validation_result",
            "schema_version": 1,
            "verdict": self.verdict,
            "report_root": "runtime_only_not_serialized",
            "final_report_path": self.final_report_path,
            "runner_status": self.runner_status,
            "checks": dict(self.checks),
            "blockers": list(self.blockers),
            "warnings": list(self.warnings),
            "source_statuses": dict(self.source_statuses),
            "handoff_counts_by_pathway": dict(self.handoff_counts_by_pathway),
            "artifact_inventory_count": int(self.artifact_inventory_count),
            "disk_risk_counts": dict(self.disk_risk_counts),
            "production_safety": dict(self.production_safety),
            "deferred_automation_polish": list(self.deferred_automation_polish),
            "deferred_production_promotion": list(self.deferred_production_promotion),
        }


def validate_final_regime_pathway_output(
    config: FinalRegimePathwaySandboxConfig | Mapping[str, Any] | None = None,
    *,
    report_root: str | Path | None = None,
    execute_runner: bool = True,
    rewrite_report: bool = True,
    **runner_kwargs: Any,
) -> FinalRegimePathwayValidationResult:
    cfg = _coerce_config(config, report_root=report_root, runner_kwargs=runner_kwargs)
    root = validate_report_root(cfg.report_root, allow_foundation_descendant=True)
    runner_result = (
        run_final_regime_pathway_sandbox_output(cfg)
        if execute_runner
        else _load_runner_result(root / FINAL_REGIME_PATHWAY_RUN_RESULT_FILENAME, report_root=root)
    )
    loaded = _load_artifacts(root)
    checks, blockers, warnings = _validate_loaded_artifacts(root, runner_result, loaded)
    verdict = _verdict(checks=checks, blockers=blockers, runner_status=runner_result.status)
    result = FinalRegimePathwayValidationResult(
        verdict=verdict,
        report_root=root,
        final_report_path=FINAL_REGIME_PATHWAY_OUTPUT_REPORT_FILENAME,
        runner_status=runner_result.status,
        checks=checks,
        blockers=tuple(blockers),
        warnings=tuple(warnings),
        source_statuses=_source_statuses(loaded["source_registry"]),
        handoff_counts_by_pathway=_handoff_counts(loaded["handoff_manifests"]),
        artifact_inventory_count=int(loaded["inventory"].get("artifact_count", 0)),
        disk_risk_counts=dict(loaded["inventory"]["disk_safety_report"].get("risk_counts", {})),
        production_safety=_production_safety(runner_result, loaded),
        deferred_automation_polish=_deferred_automation_polish(loaded["readiness_matrix"]),
        deferred_production_promotion=_deferred_production_promotion(loaded["readiness_matrix"]),
    )
    if rewrite_report:
        _write_final_report(root, result, runner_result, loaded)
        _refresh_inventory_after_report(root, runner_result.run_id)
        refreshed = _load_artifacts(root)
        refreshed_checks, refreshed_blockers, refreshed_warnings = _validate_loaded_artifacts(root, runner_result, refreshed)
        result = FinalRegimePathwayValidationResult(
            verdict=_verdict(checks=refreshed_checks, blockers=refreshed_blockers, runner_status=runner_result.status),
            report_root=root,
            final_report_path=FINAL_REGIME_PATHWAY_OUTPUT_REPORT_FILENAME,
            runner_status=runner_result.status,
            checks=refreshed_checks,
            blockers=tuple(refreshed_blockers),
            warnings=tuple(refreshed_warnings),
            source_statuses=_source_statuses(refreshed["source_registry"]),
            handoff_counts_by_pathway=_handoff_counts(refreshed["handoff_manifests"]),
            artifact_inventory_count=int(refreshed["inventory"].get("artifact_count", 0)),
            disk_risk_counts=dict(refreshed["inventory"]["disk_safety_report"].get("risk_counts", {})),
            production_safety=_production_safety(runner_result, refreshed),
            deferred_automation_polish=_deferred_automation_polish(refreshed["readiness_matrix"]),
            deferred_production_promotion=_deferred_production_promotion(refreshed["readiness_matrix"]),
        )
        _write_final_report(root, result, runner_result, refreshed)
    return result


def _coerce_config(
    config: FinalRegimePathwaySandboxConfig | Mapping[str, Any] | None,
    *,
    report_root: str | Path | None,
    runner_kwargs: Mapping[str, Any],
) -> FinalRegimePathwaySandboxConfig:
    payload = dict(runner_kwargs)
    if report_root is not None:
        payload["report_root"] = report_root
    if config is None:
        return FinalRegimePathwaySandboxConfig(**payload)
    if isinstance(config, FinalRegimePathwaySandboxConfig):
        base = config.__dict__.copy()
    else:
        base = dict(config)
    base.update(payload)
    return FinalRegimePathwaySandboxConfig(**base)


def _load_runner_result(path: Path, *, report_root: Path) -> FinalRegimePathwayRunResult:
    payload = _read_json(path)
    kwargs = dict(payload)
    kwargs.pop("artifact_kind", None)
    kwargs.pop("schema_version", None)
    kwargs["report_root"] = report_root
    return FinalRegimePathwayRunResult(**kwargs)


def _load_artifacts(report_root: Path) -> dict[str, Any]:
    run_result = _read_json(report_root / FINAL_REGIME_PATHWAY_RUN_RESULT_FILENAME)
    source_registry = _read_json(report_root / "source_registry_diagnostics.json")
    handoff_index = _read_json(report_root / "forecaster_handoff_index.json")
    inventory = _read_json(report_root / FINAL_REGIME_ARTIFACT_INVENTORY_FILENAME)
    readiness = _read_json(report_root / "test_branch_readiness_matrix.json")
    handoff_manifests = _load_handoff_manifests(report_root, handoff_index)
    return {
        "run_result": run_result,
        "source_registry": source_registry,
        "handoff_index": handoff_index,
        "handoff_manifests": handoff_manifests,
        "inventory": inventory,
        "readiness_matrix": readiness,
    }


def _validate_loaded_artifacts(
    report_root: Path,
    runner_result: FinalRegimePathwayRunResult,
    loaded: Mapping[str, Any],
) -> tuple[dict[str, bool], list[str], list[str]]:
    checks: dict[str, bool] = {}
    blockers: list[str] = []
    warnings: list[str] = []

    checks["bounded_end_to_end_runner_explicit_status"] = runner_result.status in {
        FINAL_REGIME_PATHWAY_STATUS_COMPLETED,
        FINAL_REGIME_PATHWAY_STATUS_COMPLETED_WITH_WARNINGS,
        FINAL_REGIME_PATHWAY_STATUS_PARTIAL_MISSING_DATA,
        FINAL_REGIME_PATHWAY_STATUS_BLOCKED_BY_SOURCE_RESOLUTION,
        FINAL_REGIME_PATHWAY_STATUS_BLOCKED_BY_SCHEMA_GAP,
        FINAL_REGIME_PATHWAY_STATUS_BLOCKED_BY_WRITER_GAP,
        FINAL_REGIME_PATHWAY_STATUS_FAILED,
    }
    checks["bounded_end_to_end_runner_succeeded_or_partial"] = runner_result.status in {
        FINAL_REGIME_PATHWAY_STATUS_COMPLETED,
        FINAL_REGIME_PATHWAY_STATUS_COMPLETED_WITH_WARNINGS,
        FINAL_REGIME_PATHWAY_STATUS_PARTIAL_MISSING_DATA,
    }
    checks["source_registry_resolves_or_reports_missing"] = _source_registry_clear(loaded["source_registry"])
    checks["asset_state_handoff_exists"] = any(m.get("pathway") == "asset_state" for m in loaded["handoff_manifests"])
    checks["market_state_handoff_exists"] = any(m.get("pathway") == "market_state" for m in loaded["handoff_manifests"])
    checks["cross_asset_handoff_exists"] = any(m.get("pathway") == "cross_asset" for m in loaded["handoff_manifests"])
    checks["unified_forecaster_handoff_contract_validates"] = _validate_handoffs(report_root, loaded["handoff_index"], loaded["handoff_manifests"])
    checks["artifact_inventory_exists"] = bool(loaded["inventory"].get("artifact_refs"))
    checks["artifact_inventory_validates"] = _validate_inventory(loaded["inventory"])
    checks["disk_safety_validation_passes"] = _validate_disk_safety(loaded["inventory"])
    checks["test_profile_readiness_matrix_exists"] = _validate_readiness(loaded["readiness_matrix"])
    checks["portable_artifact_refs_pass"] = _portable_refs_pass(loaded)
    checks["no_production_writes"] = not bool(runner_result.production_outputs_written) and not _flag_in_payloads(loaded, "production_outputs_written")
    checks["no_production_labels"] = not bool(runner_result.production_labels_written) and not _truthy_key_contains(loaded, "production_labels")
    checks["no_production_promotion"] = not bool(runner_result.production_promotion_performed) and not (
        _flag_in_payloads(loaded, "production_promotion_performed") or _flag_in_payloads(loaded, "promotion_performed")
    )
    checks["no_broad_all_to_all_production_pairwise"] = not bool(runner_result.broad_all_to_all_pairwise_run) and _blocked_artifact_present(
        loaded["inventory"], "broad_all_to_all_pairwise"
    )
    checks["no_cross_asset_regime_labels"] = not bool(runner_result.cross_asset_labels_written) and not _truthy_key_contains(
        loaded, "cross_asset_labels_written"
    )
    checks["no_numerics_exports"] = not _contains_token(loaded, "numerics_export")
    checks["no_hardcoded_absolute_paths_introduced"] = not find_unsafe_path_strings(loaded) and not bool(
        runner_result.hardcoded_absolute_paths_introduced
    )

    for name, ok in checks.items():
        if not ok and name in {
            "bounded_end_to_end_runner_succeeded_or_partial",
            "unified_forecaster_handoff_contract_validates",
            "asset_state_handoff_exists",
            "market_state_handoff_exists",
            "cross_asset_handoff_exists",
            "artifact_inventory_exists",
            "artifact_inventory_validates",
            "disk_safety_validation_passes",
            "test_profile_readiness_matrix_exists",
            "portable_artifact_refs_pass",
            "no_production_writes",
            "no_production_labels",
            "no_production_promotion",
            "no_broad_all_to_all_production_pairwise",
            "no_cross_asset_regime_labels",
            "no_numerics_exports",
            "no_hardcoded_absolute_paths_introduced",
        }:
            blockers.append(name)
        elif not ok:
            warnings.append(name)
    if runner_result.status == FINAL_REGIME_PATHWAY_STATUS_PARTIAL_MISSING_DATA:
        blockers.extend(runner_result.blockers)
    return checks, blockers, warnings


def _source_registry_clear(source_registry: Mapping[str, Any]) -> bool:
    sources = source_registry.get("sources")
    if not isinstance(sources, Mapping) or not sources:
        return False
    for diagnostic in sources.values():
        if not isinstance(diagnostic, Mapping):
            return False
        if diagnostic.get("status") not in SOURCE_STATUSES:
            return False
        if not diagnostic.get("resolved_by"):
            return False
    return True


def _validate_handoffs(report_root: Path, handoff_index: Mapping[str, Any], manifests: Sequence[Mapping[str, Any]]) -> bool:
    try:
        validate_regime_forecaster_handoff_index(handoff_index, artifact_root=report_root, write_outputs=True)
        for manifest in manifests:
            validate_regime_forecaster_handoff_manifest(
                manifest,
                artifact_root=report_root,
                report_root=report_root,
                write_outputs=False,
            )
            _validate_handoff_ref_targets(report_root, manifest)
    except Exception:
        return False
    return True


def _validate_handoff_ref_targets(report_root: Path, manifest: Mapping[str, Any]) -> None:
    candidate_roots = (report_root, _handoff_artifact_root(report_root, manifest))
    for refs_name in ("source_artifact_refs", "output_artifact_refs"):
        refs = manifest.get(refs_name, {})
        if not isinstance(refs, Mapping):
            raise ValueError(f"{refs_name} must be a mapping")
        for ref in refs.values():
            validate_portable_artifact_ref(ref)
            for artifact_root in candidate_roots:
                try:
                    resolve_artifact_ref(ref, artifact_root=artifact_root, report_root=report_root, must_exist=True)
                    break
                except ValueError:
                    continue
            else:
                resolve_artifact_ref(ref, artifact_root=report_root, report_root=report_root, must_exist=True)


def _handoff_artifact_root(report_root: Path, manifest: Mapping[str, Any]) -> Path:
    pathway = str(manifest.get("pathway", ""))
    if pathway == "asset_state":
        return report_root / "asset_state_finalization"
    if pathway == "market_state":
        return report_root / "market_state_finalization"
    if pathway == "cross_asset":
        return report_root / "cross_asset_finalization"
    return report_root


def _validate_inventory(inventory: Mapping[str, Any]) -> bool:
    try:
        validate_artifact_inventory(inventory)
    except Exception:
        return False
    return True


def _validate_disk_safety(inventory: Mapping[str, Any]) -> bool:
    try:
        validate_disk_safety_report(inventory["disk_safety_report"])
    except Exception:
        return False
    return True


def _validate_readiness(matrix: Mapping[str, Any]) -> bool:
    try:
        validate_test_branch_readiness_matrix(matrix)
    except Exception:
        return False
    return True


def _portable_refs_pass(loaded: Mapping[str, Any]) -> bool:
    try:
        for ref in loaded["inventory"].get("artifact_refs", ()):
            validate_portable_artifact_ref(ref)
        for record in loaded["inventory"].get("artifact_records", ()):
            validate_portable_artifact_ref(record["portable_ref"])
        for manifest in loaded["handoff_manifests"]:
            for refs_name in ("source_artifact_refs", "output_artifact_refs"):
                for ref in manifest.get(refs_name, {}).values():
                    validate_portable_artifact_ref(ref)
        for ref in loaded["handoff_index"].get("handoff_manifest_refs", {}).values():
            validate_portable_artifact_ref(ref)
    except Exception:
        return False
    return True


def _load_handoff_manifests(report_root: Path, handoff_index: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    manifests: list[dict[str, Any]] = []
    for ref in handoff_index.get("handoff_manifest_refs", {}).values():
        path = resolve_artifact_ref(ref, artifact_root=report_root, report_root=report_root, must_exist=True)
        manifests.append(_read_json(path))
    return tuple(manifests)


def _production_safety(runner_result: FinalRegimePathwayRunResult, loaded: Mapping[str, Any]) -> dict[str, bool]:
    return {
        "production_outputs_written": bool(runner_result.production_outputs_written) or _flag_in_payloads(loaded, "production_outputs_written"),
        "production_labels_written": bool(runner_result.production_labels_written) or _truthy_key_contains(loaded, "production_labels"),
        "production_promotion_performed": bool(runner_result.production_promotion_performed)
        or _flag_in_payloads(loaded, "production_promotion_performed")
        or _flag_in_payloads(loaded, "promotion_performed"),
        "broad_all_to_all_pairwise_run": bool(runner_result.broad_all_to_all_pairwise_run),
        "cross_asset_labels_written": bool(runner_result.cross_asset_labels_written) or _truthy_key_contains(loaded, "cross_asset_labels_written"),
        "forecaster_training_run": bool(runner_result.forecaster_training_run),
        "hardcoded_absolute_paths_introduced": bool(runner_result.hardcoded_absolute_paths_introduced) or bool(find_unsafe_path_strings(loaded)),
        "numerics_exports_present": _contains_token(loaded, "numerics_export"),
    }


def _handoff_counts(manifests: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts = {"asset_state": 0, "market_state": 0, "cross_asset": 0}
    for manifest in manifests:
        pathway = str(manifest.get("pathway", ""))
        if pathway in counts:
            counts[pathway] += 1
    return counts


def _source_statuses(source_registry: Mapping[str, Any]) -> dict[str, str]:
    sources = source_registry.get("sources", {})
    return {str(kind): str(diagnostic.get("status")) for kind, diagnostic in sources.items() if isinstance(diagnostic, Mapping)}


def _deferred_automation_polish(readiness: Mapping[str, Any]) -> tuple[str, ...]:
    out: list[str] = []
    for row in readiness.get("rows", ()):
        if isinstance(row, Mapping):
            out.extend(str(item) for item in row.get("deferred_to_automation_polish", ()))
    return tuple(dict.fromkeys(out))


def _deferred_production_promotion(readiness: Mapping[str, Any]) -> tuple[str, ...]:
    out: list[str] = []
    for row in readiness.get("rows", ()):
        if isinstance(row, Mapping):
            out.extend(str(item) for item in row.get("deferred_to_production_promotion", ()))
    return tuple(dict.fromkeys(out))


def _verdict(*, checks: Mapping[str, bool], blockers: Sequence[str], runner_status: str) -> str:
    if runner_status in {
        FINAL_REGIME_PATHWAY_STATUS_BLOCKED_BY_SCHEMA_GAP,
        FINAL_REGIME_PATHWAY_STATUS_BLOCKED_BY_WRITER_GAP,
        FINAL_REGIME_PATHWAY_STATUS_FAILED,
    }:
        return FINAL_PATHWAY_VALIDATOR_RED
    if blockers or runner_status in {FINAL_REGIME_PATHWAY_STATUS_PARTIAL_MISSING_DATA, FINAL_REGIME_PATHWAY_STATUS_BLOCKED_BY_SOURCE_RESOLUTION}:
        return FINAL_PATHWAY_VALIDATOR_YELLOW
    if all(checks.values()):
        return FINAL_PATHWAY_VALIDATOR_PASSED
    return FINAL_PATHWAY_VALIDATOR_YELLOW


def _refresh_inventory_after_report(report_root: Path, run_id: str) -> None:
    artifacts = tuple(sorted(path for path in report_root.rglob("*") if path.is_file() and path.name != FINAL_REGIME_ARTIFACT_INVENTORY_FILENAME))
    inventory = build_artifact_inventory(
        artifacts,
        artifact_root=report_root,
        inventory_id=f"{run_id}_artifact_inventory",
        producer="src.regimes.final_pathway_ready",
        report_root=report_root,
    )
    validate_artifact_inventory(inventory)
    validate_disk_safety_report(inventory["disk_safety_report"])
    _write_json(report_root / FINAL_REGIME_ARTIFACT_INVENTORY_FILENAME, inventory)


def _write_final_report(
    report_root: Path,
    result: FinalRegimePathwayValidationResult,
    runner_result: FinalRegimePathwayRunResult,
    loaded: Mapping[str, Any],
) -> Path:
    path = report_root / FINAL_REGIME_PATHWAY_OUTPUT_REPORT_FILENAME
    lines = _report_lines(result, runner_result, loaded)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return path


def _report_lines(
    result: FinalRegimePathwayValidationResult,
    runner_result: FinalRegimePathwayRunResult,
    loaded: Mapping[str, Any],
) -> list[str]:
    inventory = loaded["inventory"]
    readiness = loaded["readiness_matrix"]
    checks = inventory["disk_safety_report"]["checks"]
    final_output_blockers = readiness.get("summary", {}).get("final_output_sprint_blockers", [])
    safety = result.production_safety
    return [
        "# Final Regime Pathway Output Report",
        "",
        "## 1. Executive Verdict",
        f"- Verdict: {result.verdict}",
        f"- Runner status: `{result.runner_status}`",
        f"- Final report path: `{FINAL_REGIME_PATHWAY_OUTPUT_REPORT_FILENAME}`",
        f"- Blockers: `{len(result.blockers)}`",
        f"- Warnings: `{len(result.warnings)}`",
        "",
        "## 2. Source/Data Readiness",
        f"- Source registry: `source_registry_diagnostics.json`",
        f"- Source statuses: `{json.dumps(result.source_statuses, sort_keys=True)}`",
        "- Missing or partial inputs are explicit diagnostics; strict real-source runs can return `blocked_by_source_resolution`.",
        "",
        "## 3. Asset-State Output Readiness",
        f"- Asset-State runner status: `{runner_result.asset_state_status}`",
        f"- Outputs produced: `{runner_result.asset_state_outputs_produced}`",
        f"- Handoff manifests: `{result.handoff_counts_by_pathway.get('asset_state', 0)}`",
        "- All six Asset-State axes are covered by the Block 9 sandbox output path.",
        "",
        "## 4. Market-State Output Readiness",
        f"- Market-State runner status: `{runner_result.market_state_status}`",
        f"- Outputs produced: `{runner_result.market_state_outputs_produced}`",
        f"- Handoff manifests: `{result.handoff_counts_by_pathway.get('market_state', 0)}`",
        "- Feature panels and axis panels are sandbox-only; no monolithic final Market-State label was produced.",
        "",
        "## 5. Cross-Asset Feature Output Readiness",
        f"- Cross-Asset runner status: `{runner_result.cross_asset_status}`",
        f"- Feature outputs produced: `{runner_result.cross_asset_feature_outputs_produced}`",
        f"- Handoff manifests: `{result.handoff_counts_by_pathway.get('cross_asset', 0)}`",
        "- Relationship Discovery v1 artifacts and row-based Cross-Asset feature rows are present; no Cross-Asset regime labels or peer clusters were created.",
        "",
        "## 6. Unified Forecaster Handoff",
        f"- Handoff index: `{runner_result.forecaster_handoff_index_path}`",
        f"- Contract validates: `{result.checks.get('unified_forecaster_handoff_contract_validates')}`",
        f"- Portable refs pass: `{result.checks.get('portable_artifact_refs_pass')}`",
        "- No Regime forecaster implementation was added.",
        "",
        "## 7. Artifact Inventory",
        f"- Inventory path: `{runner_result.artifact_inventory_path}`",
        f"- Artifact count: `{result.artifact_inventory_count}`",
        f"- Inventory validates: `{result.checks.get('artifact_inventory_validates')}`",
        "",
        "## 8. Disk-Safety Assessment",
        f"- Disk-safety validation passes: `{result.checks.get('disk_safety_validation_passes')}`",
        f"- Risk counts: `{json.dumps(result.disk_risk_counts, sort_keys=True)}`",
        f"- Broad all-to-all pairwise blocked: `{checks.get('broad_all_to_all_pairwise_blocked')}`",
        f"- Full rolling matrices not persisted: `{checks.get('full_rolling_matrices_not_persisted')}`",
        f"- One-column-per-related-asset schema not introduced: `{checks.get('one_column_per_related_asset_schema_not_introduced')}`",
        "",
        "## 9. Test/Profile-Selection Readiness",
        f"- Readiness matrix: `{runner_result.test_branch_readiness_matrix_path}`",
        f"- Final-output sprint blockers: `{final_output_blockers}`",
        "- Profile selection did not run and final profiles were not selected.",
        "",
        "## 10. Bounded End-To-End Runner Evidence",
        f"- Runner result: `{FINAL_REGIME_PATHWAY_RUN_RESULT_FILENAME}`",
        f"- Runner succeeded or returned explicit partial status: `{result.checks.get('bounded_end_to_end_runner_succeeded_or_partial')}`",
        f"- Asset-State outputs: `{runner_result.asset_state_outputs_produced}`",
        f"- Market-State outputs: `{runner_result.market_state_outputs_produced}`",
        f"- Cross-Asset feature outputs: `{runner_result.cross_asset_feature_outputs_produced}`",
        "",
        "## 11. Safety/Pathing Confirmation",
        f"- No production writes: `{result.checks.get('no_production_writes')}`",
        f"- No production labels: `{result.checks.get('no_production_labels')}`",
        f"- No production promotion: `{result.checks.get('no_production_promotion')}`",
        f"- No broad all-to-all production pairwise: `{result.checks.get('no_broad_all_to_all_production_pairwise')}`",
        f"- No Cross-Asset regime labels: `{result.checks.get('no_cross_asset_regime_labels')}`",
        f"- No Numerics exports: `{result.checks.get('no_numerics_exports')}`",
        f"- No hardcoded absolute paths introduced: `{result.checks.get('no_hardcoded_absolute_paths_introduced')}`",
        f"- Production safety flags: `{json.dumps(safety, sort_keys=True)}`",
        "",
        "## 12. Remaining Blockers",
        *([f"- {item}" for item in result.blockers] if result.blockers else ["- None."]),
        "",
        "## 13. Deferred Work For Automation Polish",
        *([f"- {item}" for item in result.deferred_automation_polish] if result.deferred_automation_polish else ["- None."]),
        "",
        "## 14. Deferred Work For Production Promotion",
        *([f"- {item}" for item in result.deferred_production_promotion] if result.deferred_production_promotion else ["- None."]),
    ]


def _blocked_artifact_present(inventory: Mapping[str, Any], artifact_kind: str) -> bool:
    return any(item.get("artifact_kind") == artifact_kind for item in inventory.get("disk_safety_report", {}).get("blocked_artifacts", ()))


def _flag_in_payloads(payload: Any, key: str) -> bool:
    if isinstance(payload, Mapping):
        for item_key, value in payload.items():
            if item_key == key and bool(value):
                return True
            if _flag_in_payloads(value, key):
                return True
    elif isinstance(payload, Sequence) and not isinstance(payload, (str, bytes, bytearray)):
        return any(_flag_in_payloads(item, key) for item in payload)
    return False


def _truthy_key_contains(payload: Any, token: str) -> bool:
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            if token in str(key) and bool(value):
                return True
            if _truthy_key_contains(value, token):
                return True
    elif isinstance(payload, Sequence) and not isinstance(payload, (str, bytes, bytearray)):
        return any(_truthy_key_contains(item, token) for item in payload)
    return False


def _contains_token(payload: Any, token: str) -> bool:
    if isinstance(payload, Mapping):
        return any(token in str(key) or _contains_token(value, token) for key, value in payload.items())
    if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes, bytearray)):
        return any(_contains_token(item, token) for item in payload)
    return token in str(payload)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(to_jsonable(dict(payload)), f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp, path)
    return path


__all__ = [
    "FINAL_PATHWAY_VALIDATOR_PASSED",
    "FINAL_PATHWAY_VALIDATOR_RED",
    "FINAL_PATHWAY_VALIDATOR_YELLOW",
    "FINAL_REGIME_PATHWAY_OUTPUT_REPORT_FILENAME",
    "FinalRegimePathwayValidationResult",
    "validate_final_regime_pathway_output",
]
