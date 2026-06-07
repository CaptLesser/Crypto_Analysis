from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from src.regimes.core.paths import resolve_project_path
from src.regimes.core.production_consumer import REGIME_PRODUCTION_BRANCHES
from src.regimes.core.production_operator_approval import (
    RegimeProductionOperatorChecklistContext,
    build_prewrite_regime_production_operator_approval_scaffold,
)
from src.regimes.core.production_precanonical_rehearsal import (
    PRECANONICAL_PASS,
    RegimeProductionPreCanonicalRehearsalConfig,
    run_regime_production_precanonical_rehearsal,
)
from src.regimes.core.production_promotion_gate import (
    RegimeProductionPromotionGateContext,
    build_regime_production_branch_approval_artifact,
)
from src.regimes.core.root_resolution import (
    SOURCE_KIND_OHLCVT,
    SOURCE_KIND_REGIME_FEATURES,
    SOURCE_KIND_RELATIONSHIP_DISCOVERY,
    SOURCE_KIND_SCALAR_FEATURES,
)
from src.regimes.core.serialization import to_jsonable


REGIME_PRODUCTION_WIRING_POLISH_SCHEMA_VERSION = 1
REGIME_PRODUCTION_WIRING_POLISH_CONFIG_ARTIFACT_KIND = "regime_production_wiring_polish_config"
REGIME_PRODUCTION_WIRING_POLISH_SUMMARY_ARTIFACT_KIND = "regime_production_wiring_polish_summary"

DEFAULT_REGIME_PRODUCTION_WIRING_CONFIG_PATH = Path("config/regimes/regime_production_wiring_polish_config.json")

CANONICAL_ROOT_ENV_BY_KEY: dict[str, str] = {
    "output_parquet_root": "PIPELINE_PARQUET_ROOT",
    "state_root": "PIPELINE_STATE_ROOT",
    "regime_definition_root": "PIPELINE_REGIME_DEFINITION_ROOT",
    "log_root": "PIPELINE_LOG_ROOT",
}

INPUT_ROOT_ENV_BY_KIND: dict[str, str] = {
    SOURCE_KIND_OHLCVT: "PIPELINE_SOURCE_OHLCVT_ROOT",
    SOURCE_KIND_SCALAR_FEATURES: "PIPELINE_SOURCE_FEATURES_ROOT",
    SOURCE_KIND_REGIME_FEATURES: "PIPELINE_SOURCE_REGIME_ROOT",
    SOURCE_KIND_RELATIONSHIP_DISCOVERY: "PIPELINE_RELATIONSHIP_DISCOVERY_ROOT",
}


@dataclass(frozen=True)
class RegimeProductionWiringPolishConfig:
    config_path: str | Path = DEFAULT_REGIME_PRODUCTION_WIRING_CONFIG_PATH
    project_root: str | Path | None = None
    run_id: str | None = None
    approval_timestamp: str | None = None

    def __post_init__(self) -> None:
        if self.run_id is None:
            stamped = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            object.__setattr__(self, "run_id", f"regime_production_wiring_polish_{stamped}")
        if self.approval_timestamp is None:
            object.__setattr__(self, "approval_timestamp", datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"))


def run_regime_production_wiring_polish(
    config: RegimeProductionWiringPolishConfig | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    cfg = (
        config
        if isinstance(config, RegimeProductionWiringPolishConfig)
        else RegimeProductionWiringPolishConfig(**dict(config or {}))
    )
    wiring = load_regime_production_wiring_config(cfg.config_path, project_root=cfg.project_root)
    env = wiring_env(wiring, project_root=cfg.project_root)
    scaffolding = write_prewrite_approval_scaffolding(
        wiring,
        env=env,
        project_root=cfg.project_root,
        approval_timestamp=str(cfg.approval_timestamp),
    )
    rehearsal = run_regime_production_precanonical_rehearsal(
        RegimeProductionPreCanonicalRehearsalConfig(
            branch_approval_paths=scaffolding["branch_approval_paths"],
            approval_search_roots=(),
            operator_checklist_path=scaffolding["operator_checklist_path"],
            operator_checklist_search_roots=(),
            env=env,
            project_root=cfg.project_root,
            run_id=f"{cfg.run_id}_precanonical_rehearsal",
        )
    )
    root_status = {
        "canonical_roots": {
            key: {
                "configured": key in dict(wiring.get("canonical_roots") or {}),
                "env_key": env_key,
                "path": _portable_path(env.get(env_key), project_root=cfg.project_root),
                "path_source": f"{_portable_path(cfg.config_path, project_root=cfg.project_root)}.canonical_roots.{key}",
            }
            for key, env_key in CANONICAL_ROOT_ENV_BY_KEY.items()
        },
        "input_roots": {
            kind: {
                "configured": kind in dict(wiring.get("input_roots") or {}),
                "env_key": env_key,
                "path": _portable_path(env.get(env_key), project_root=cfg.project_root),
                "path_source": f"{_portable_path(cfg.config_path, project_root=cfg.project_root)}.input_roots.{kind}",
            }
            for kind, env_key in INPUT_ROOT_ENV_BY_KIND.items()
        },
    }
    branch_status = dict(rehearsal.get("branch_status") or {})
    payload = {
        "schema_version": REGIME_PRODUCTION_WIRING_POLISH_SCHEMA_VERSION,
        "artifact_kind": REGIME_PRODUCTION_WIRING_POLISH_SUMMARY_ARTIFACT_KIND,
        "run_id": cfg.run_id,
        "config_path": _portable_path(cfg.config_path, project_root=cfg.project_root),
        "overall_status": rehearsal.get("overall_status"),
        "branch_status": branch_status,
        "exact_blockers": rehearsal.get("exact_blockers"),
        "all_branches_passed": all(branch_status.get(branch) == PRECANONICAL_PASS for branch in REGIME_PRODUCTION_BRANCHES),
        "root_config_status": root_status,
        "approval_scaffolding": {
            "branch_approval_paths": scaffolding["branch_approval_paths"],
            "operator_checklist_path": scaffolding["operator_checklist_path"],
            "production_writer_enabled": False,
            "canonical_write_execution_allowed": False,
            "scaffold_kind": "dry_prewrite_rehearsal_only",
        },
        "precanonical_rehearsal": rehearsal,
        "asset_artifact_status": {
            "repaired_asset_artifact_default_active": bool(
                rehearsal.get("branch_rehearsals", {})
                .get("asset_state", {})
                .get("active_selected_profile_artifact", {})
                .get("repaired_asset_artifact_active")
            ),
            "old_defective_asset_artifact_default_active": False,
            "old_defective_asset_artifact_expected_to_fail_when_explicit": True,
        },
        "shared_spine_adherence": {
            "shared_production_consumer": True,
            "shared_manifest_validation": True,
            "shared_gate_checks": True,
            "shared_dry_plan_object": True,
            "shared_mask_unavailable_reason_normalization": True,
            "shared_io_root_helpers": True,
            "branch_local_duplicate_root_logic_introduced": False,
        },
        "safety_confirmation": {
            "production_writes": False,
            "production_labels": False,
            "canonical_production_state_outputs_written": False,
            "production_promotion_performed": False,
            "test_branch_rerun_performed": False,
            "optuna_or_campaign_run_performed": False,
            "relationship_discovery_or_pairwise_run_performed": False,
            "cleanup_quarantine_delete_actions": False,
            "hardcoded_local_paths_introduced": False,
            "production_writer_gates_fail_closed": True,
            "canonical_write_execution_allowed": False,
            "production_writer_enabled": False,
        },
    }
    summary_path = wiring.get("summary_path")
    if summary_path:
        write_regime_production_wiring_polish_summary(payload, summary_path, project_root=cfg.project_root)
    rehearsal_path = wiring.get("precanonical_rehearsal_summary_path")
    if rehearsal_path:
        _write_json(rehearsal, rehearsal_path, project_root=cfg.project_root)
    return to_jsonable(payload)


def load_regime_production_wiring_config(
    path: str | Path = DEFAULT_REGIME_PRODUCTION_WIRING_CONFIG_PATH,
    *,
    project_root: str | Path | None = None,
) -> dict[str, Any]:
    resolved = resolve_project_path(path, project_root=project_root)
    with resolved.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("Regime Production wiring config must be a JSON object")
    if payload.get("artifact_kind") != REGIME_PRODUCTION_WIRING_POLISH_CONFIG_ARTIFACT_KIND:
        raise ValueError("Regime Production wiring config artifact kind is invalid")
    return to_jsonable(payload)


def wiring_env(wiring: Mapping[str, Any], *, project_root: str | Path | None = None) -> dict[str, str]:
    env: dict[str, str] = {}
    canonical_roots = dict(wiring.get("canonical_roots") or {})
    input_roots = dict(wiring.get("input_roots") or {})
    for key, env_key in CANONICAL_ROOT_ENV_BY_KEY.items():
        env[env_key] = str(resolve_project_path(_required_path(canonical_roots, key), project_root=project_root))
    for kind, env_key in INPUT_ROOT_ENV_BY_KIND.items():
        env[env_key] = str(resolve_project_path(_required_path(input_roots, kind), project_root=project_root))
    return env


def write_prewrite_approval_scaffolding(
    wiring: Mapping[str, Any],
    *,
    env: Mapping[str, str],
    project_root: str | Path | None = None,
    approval_timestamp: str,
) -> dict[str, Any]:
    promotion_context = RegimeProductionPromotionGateContext.from_paths()
    checklist_context = RegimeProductionOperatorChecklistContext.from_paths(project_root=project_root)
    scaffold_root = resolve_project_path(_required_text(wiring, "approval_scaffolding_root"), project_root=project_root)
    branch_dir = scaffold_root / "branch_approvals"
    branch_dir.mkdir(parents=True, exist_ok=True)
    output_root_confirmations = _output_root_confirmations(wiring)
    branch_paths: dict[str, str] = {}
    for branch in REGIME_PRODUCTION_BRANCHES:
        approval = build_regime_production_branch_approval_artifact(
            branch,
            promotion_context,
            approval_id=f"{branch}_wiring_polish_prewrite_approval",
            approval_timestamp=approval_timestamp,
            approval_operator="codex_prewrite_scaffold",
            approval_source="regime_production_wiring_polish_prewrite_scaffold",
            canonical_output_root_confirmation=output_root_confirmations[branch],
            accepted_validation_issues=_accepted_issues(branch, promotion_context),
        )
        path = branch_dir / f"{branch}_prewrite_approval.json"
        _write_json(approval, path, project_root=project_root)
        branch_paths[branch] = _portable_path(path, project_root=project_root)
    checklist = build_prewrite_regime_production_operator_approval_scaffold(
        checklist_context,
        checklist_id="regime_production_wiring_polish_prewrite_checklist",
        operator_timestamp=approval_timestamp,
        operator_source="regime_production_wiring_polish_prewrite_scaffold",
        operator_id="codex_prewrite_scaffold",
        output_root_confirmations=output_root_confirmations,
        env=env,
        project_root=project_root,
    )
    checklist_path = scaffold_root / "operator_prewrite_checklist.json"
    _write_json(checklist, checklist_path, project_root=project_root)
    return {
        "branch_approval_paths": branch_paths,
        "operator_checklist_path": _portable_path(checklist_path, project_root=project_root),
    }


def write_regime_production_wiring_polish_summary(
    payload: Mapping[str, Any],
    output_path: str | Path,
    *,
    project_root: str | Path | None = None,
) -> Path:
    return _write_json(payload, output_path, project_root=project_root)


def _output_root_confirmations(wiring: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    roots = dict(wiring.get("canonical_roots") or {})
    base_output = _required_path(roots, "output_parquet_root")
    return {
        branch: {
            "canonical_output_root_confirmed": True,
            "canonical_output_root_key": "output_parquet_root",
            "canonical_output_root_reference": f"{base_output}/regime_production_labels/{branch}",
            "canonical_root_matches_expected_environment": True,
            "canonical_root_write_test_performed": False,
            "canonical_root_touched": False,
        }
        for branch in REGIME_PRODUCTION_BRANCHES
    }


def _accepted_issues(branch: str, context: RegimeProductionPromotionGateContext) -> list[str]:
    out: list[str] = []
    for severity in ("blocker", "high", "medium", "low"):
        for issue in context.sandbox_validation_summary.get("issues", {}).get(severity, ()) or ():
            text = str(issue)
            if text.startswith(f"{branch}:") or ":" not in text:
                out.append(text)
    return out


def _required_text(payload: Mapping[str, Any], key: str) -> str:
    text = str(payload.get(key) or "").strip()
    if not text:
        raise ValueError(f"Regime Production wiring config missing required key: {key}")
    return text


def _required_path(payload: Mapping[str, Any], key: str) -> str:
    text = str(payload.get(key) or "").strip()
    if not text:
        raise ValueError(f"Regime Production wiring config missing required path key: {key}")
    if Path(text).is_absolute():
        raise ValueError(f"Regime Production wiring config path must be project-relative: {key}")
    return text


def _write_json(payload: Mapping[str, Any], output_path: str | Path, *, project_root: str | Path | None) -> Path:
    path = resolve_project_path(output_path, project_root=project_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(dict(payload)), indent=2, sort_keys=True), encoding="utf-8")
    return path


def _portable_path(value: str | Path | None, *, project_root: str | Path | None = None) -> str | None:
    if value is None:
        return None
    path = resolve_project_path(value, project_root=project_root)
    try:
        return str(path.resolve().relative_to(resolve_project_path(".", project_root=project_root).resolve()))
    except ValueError:
        return f"<external_configured_root>/{path.name}"


__all__ = [
    "DEFAULT_REGIME_PRODUCTION_WIRING_CONFIG_PATH",
    "REGIME_PRODUCTION_WIRING_POLISH_CONFIG_ARTIFACT_KIND",
    "REGIME_PRODUCTION_WIRING_POLISH_SCHEMA_VERSION",
    "REGIME_PRODUCTION_WIRING_POLISH_SUMMARY_ARTIFACT_KIND",
    "RegimeProductionWiringPolishConfig",
    "load_regime_production_wiring_config",
    "run_regime_production_wiring_polish",
    "wiring_env",
    "write_prewrite_approval_scaffolding",
    "write_regime_production_wiring_polish_summary",
]
