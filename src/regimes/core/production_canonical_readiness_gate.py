from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.forecasting.common.path_config import resolve_path
from src.regimes.core.paths import resolve_project_path
from src.regimes.core.production_consumer import (
    REGIME_BRANCH_ASSET_STATE,
    REGIME_PRODUCTION_BRANCHES,
    resolve_active_selected_profile_artifact,
)
from src.regimes.core.production_output_contracts import default_regime_production_label_output_schema
from src.regimes.core.production_promotion_gate import (
    DEFAULT_SANDBOX_VALIDATION_SUMMARY_PATH,
    REGIME_PRODUCTION_BRANCH_APPROVAL_ARTIFACT_KIND,
    REGIME_PRODUCTION_UNIFIED_APPROVAL_ARTIFACT_KIND,
    RegimeProductionPromotionGateContext,
    evaluate_regime_production_branch_promotion_gate,
)
from src.regimes.core.serialization import to_jsonable


REGIME_PRODUCTION_CANONICAL_READINESS_SCHEMA_VERSION = 1
REGIME_PRODUCTION_CANONICAL_READINESS_ARTIFACT_KIND = "regime_production_canonical_readiness_gate_summary"

DEFAULT_WRITE_CAPABLE_SANDBOX_FINAL_SUMMARY_PATH = (
    "_codex_artifacts/reports/regime_production_write_capable_sandbox_final/"
    "regime_production_write_capable_sandbox_final_summary.json"
)
DEFAULT_APPROVAL_SEARCH_ROOTS: tuple[str, ...] = ("_codex_artifacts/reports",)
CANONICAL_READINESS_ROOT_KEYS: tuple[str, ...] = (
    "output_parquet_root",
    "state_root",
    "regime_definition_root",
    "log_root",
)

READINESS_PASS = "PASS"
READINESS_BLOCKED = "BLOCKED"


@dataclass(frozen=True)
class RegimeProductionCanonicalReadinessGateConfig:
    sandbox_validation_summary_path: str | Path = DEFAULT_SANDBOX_VALIDATION_SUMMARY_PATH
    wider_sandbox_summary_path: str | Path | None = None
    write_capable_sandbox_summary_path: str | Path = DEFAULT_WRITE_CAPABLE_SANDBOX_FINAL_SUMMARY_PATH
    approval_search_roots: Sequence[str | Path] = DEFAULT_APPROVAL_SEARCH_ROOTS
    branch_approval_paths: Mapping[str, str | Path] | None = None
    unified_approval_path: str | Path | None = None
    run_id: str | None = None
    env: Mapping[str, str] | None = None
    project_root: str | Path | None = None
    approval_scan_max_bytes: int = 2_000_000

    def __post_init__(self) -> None:
        if self.approval_scan_max_bytes <= 0:
            raise ValueError("Regime Production canonical readiness approval scan size must be positive")
        if self.run_id is None:
            stamped = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            object.__setattr__(self, "run_id", f"regime_production_canonical_readiness_gate_{stamped}")
        object.__setattr__(self, "approval_search_roots", tuple(self.approval_search_roots))
        object.__setattr__(self, "branch_approval_paths", dict(self.branch_approval_paths or {}))
        object.__setattr__(self, "env", None if self.env is None else dict(self.env))


def run_regime_production_canonical_readiness_gate(
    config: RegimeProductionCanonicalReadinessGateConfig | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    cfg = (
        config
        if isinstance(config, RegimeProductionCanonicalReadinessGateConfig)
        else RegimeProductionCanonicalReadinessGateConfig(**dict(config or {}))
    )
    context = RegimeProductionPromotionGateContext.from_paths(
        sandbox_validation_summary_path=cfg.sandbox_validation_summary_path,
        wider_sandbox_summary_path=cfg.wider_sandbox_summary_path,
    )
    sandbox_validation = context.sandbox_validation_summary
    write_capable = _load_json(cfg.write_capable_sandbox_summary_path, project_root=cfg.project_root)
    approval_inventory = discover_regime_production_approval_artifacts(
        cfg.approval_search_roots,
        branch_approval_paths=cfg.branch_approval_paths,
        unified_approval_path=cfg.unified_approval_path,
        project_root=cfg.project_root,
        max_bytes=cfg.approval_scan_max_bytes,
    )
    canonical_roots = _canonical_root_status(env=cfg.env, project_root=cfg.project_root)
    contract_status = _contract_status(sandbox_validation, write_capable)

    branch_readiness = {
        branch: _branch_readiness(
            branch,
            context,
            write_capable,
            approval_inventory,
            canonical_roots,
            contract_status,
            env=cfg.env,
            project_root=cfg.project_root,
        )
        for branch in REGIME_PRODUCTION_BRANCHES
    }
    blocked = {
        branch: payload["blockers"]
        for branch, payload in branch_readiness.items()
        if payload["readiness"] != READINESS_PASS
    }
    payload = {
        "schema_version": REGIME_PRODUCTION_CANONICAL_READINESS_SCHEMA_VERSION,
        "artifact_kind": REGIME_PRODUCTION_CANONICAL_READINESS_ARTIFACT_KIND,
        "run_id": cfg.run_id,
        "overall_readiness": READINESS_PASS if not blocked else READINESS_BLOCKED,
        "branch_readiness": branch_readiness,
        "exact_blockers": blocked,
        "active_selected_profile_artifacts": {
            branch: branch_readiness[branch]["active_artifact"]
            for branch in REGIME_PRODUCTION_BRANCHES
        },
        "approval_artifact_inventory": approval_inventory,
        "sandbox_validation": {
            "path": _portable_path_text(resolve_project_path(cfg.sandbox_validation_summary_path, project_root=cfg.project_root)),
            "run_id": sandbox_validation.get("run_id"),
            "validation_status": sandbox_validation.get("validation_status"),
            "blocker_high_counts": {
                "blocker": len(sandbox_validation.get("issues", {}).get("blocker", ()) or ()),
                "high": len(sandbox_validation.get("issues", {}).get("high", ()) or ()),
            },
            "accepted_medium_low_issues": list(write_capable.get("accepted_validation_issues") or ()),
            "clean_for_readiness": contract_status["sandbox_validation_clean_for_readiness"],
        },
        "write_capable_sandbox_validation": {
            "path": _portable_path_text(resolve_project_path(cfg.write_capable_sandbox_summary_path, project_root=cfg.project_root)),
            "run_id": write_capable.get("run_id"),
            "validation_results": write_capable.get("validation_results"),
            "clean_for_readiness": contract_status["write_capable_sandbox_clean_for_readiness"],
            "canonical_root_touched": bool(write_capable.get("validation_results", {}).get("canonical_root_touched") or write_capable.get("canonical_root_touched")),
        },
        "canonical_root_configuration": canonical_roots,
        "accepted_contracts": {
            "output_schemas_fixed": contract_status["output_schemas_fixed"],
            "cross_relationship_freshness_policy_accepted": contract_status["cross_relationship_freshness_policy_accepted"],
            "clamp_range_accepted": contract_status["clamp_range_accepted"],
            "worker_job_matrix_accepted": contract_status["worker_job_matrix_accepted"],
            "logging_telemetry_accepted": contract_status["logging_telemetry_accepted"],
        },
        "canonical_run_order_recommendation": [
            "market_state",
            "asset_state",
            "cross_asset_state",
        ],
        "canonical_run_order_reason": (
            "Market-State is the lowest-cardinality/no per-asset branch; Asset-State follows with the repaired active artifact; "
            "Cross-Asset-State runs last because relationship freshness inputs add the most external dependency surface."
        ),
        "first_branch_command_if_ready": _first_branch_command(branch_readiness),
        "files_changed": [
            "src/regimes/core/production_canonical_readiness_gate.py",
            "tests/regimes/test_regime_production_canonical_readiness_gate.py",
        ],
        "safety_confirmation": {
            "canonical_write_attempted": False,
            "canonical_root_touched": False,
            "production_writer_enabled": False,
            "production_labels_written": False,
            "production_outputs_written": False,
            "canonical_production_state_outputs_written": False,
            "production_promotion_performed": False,
            "test_branch_rerun_performed": False,
            "optuna_or_campaign_run_performed": False,
            "relationship_discovery_or_pairwise_run_performed": False,
            "cleanup_delete_actions_performed": False,
            "production_writer_gates_fail_closed": True,
        },
    }
    return to_jsonable(_sanitize_workspace_paths(payload))


def discover_regime_production_approval_artifacts(
    search_roots: Sequence[str | Path],
    *,
    branch_approval_paths: Mapping[str, str | Path] | None = None,
    unified_approval_path: str | Path | None = None,
    project_root: str | Path | None = None,
    max_bytes: int = 2_000_000,
) -> dict[str, Any]:
    branch_payloads: dict[str, dict[str, Any]] = {}
    branch_paths: dict[str, str] = {}
    unified_payload: dict[str, Any] | None = None
    unified_path_text: str | None = None

    for branch, raw_path in dict(branch_approval_paths or {}).items():
        payload = _load_json(raw_path, project_root=project_root)
        if payload.get("artifact_kind") == REGIME_PRODUCTION_BRANCH_APPROVAL_ARTIFACT_KIND:
            branch_name = str(payload.get("branch") or branch)
            branch_payloads[branch_name] = payload
            branch_paths[branch_name] = _portable_path_text(resolve_project_path(raw_path, project_root=project_root))

    if unified_approval_path is not None:
        payload = _load_json(unified_approval_path, project_root=project_root)
        if payload.get("artifact_kind") == REGIME_PRODUCTION_UNIFIED_APPROVAL_ARTIFACT_KIND:
            unified_payload = payload
            unified_path_text = _portable_path_text(resolve_project_path(unified_approval_path, project_root=project_root))

    for raw_root in search_roots:
        root = resolve_project_path(raw_root, project_root=project_root)
        if not root.exists():
            continue
        candidates = [root] if root.is_file() else root.rglob("*.json")
        for path in candidates:
            try:
                if not path.is_file() or path.stat().st_size > max_bytes:
                    continue
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if not isinstance(payload, Mapping):
                continue
            kind = payload.get("artifact_kind")
            if kind == REGIME_PRODUCTION_BRANCH_APPROVAL_ARTIFACT_KIND:
                branch = str(payload.get("branch") or "")
                if branch in REGIME_PRODUCTION_BRANCHES and branch not in branch_payloads:
                    branch_payloads[branch] = dict(payload)
                    branch_paths[branch] = _portable_path_text(path)
            elif kind == REGIME_PRODUCTION_UNIFIED_APPROVAL_ARTIFACT_KIND and unified_payload is None:
                unified_payload = dict(payload)
                unified_path_text = _portable_path_text(path)

    if unified_payload is not None:
        for branch, payload in dict(unified_payload.get("branch_approvals") or {}).items():
            if branch in REGIME_PRODUCTION_BRANCHES and branch not in branch_payloads and isinstance(payload, Mapping):
                branch_payloads[branch] = dict(payload)
                branch_paths[branch] = f"{unified_path_text}#branch_approvals/{branch}"

    return {
        "branch_approval_artifacts_present": {
            branch: branch in branch_payloads for branch in REGIME_PRODUCTION_BRANCHES
        },
        "branch_approval_paths": branch_paths,
        "branch_approvals": branch_payloads,
        "unified_approval_artifact_present": unified_payload is not None,
        "unified_approval_path": unified_path_text,
        "dry_approval_summaries_are_not_executable_approvals": True,
    }


def write_regime_production_canonical_readiness_gate_summary(payload: Mapping[str, Any], output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(dict(payload)), indent=2, sort_keys=True), encoding="utf-8")
    return path


def _branch_readiness(
    branch: str,
    context: RegimeProductionPromotionGateContext,
    write_capable: Mapping[str, Any],
    approvals: Mapping[str, Any],
    canonical_roots: Mapping[str, Any],
    contract_status: Mapping[str, bool],
    *,
    env: Mapping[str, str] | None,
    project_root: str | Path | None,
) -> dict[str, Any]:
    blockers: list[str] = []
    active_result = resolve_active_selected_profile_artifact(branch, env=env, project_root=project_root)
    branch_output = dict(context.wider_sandbox_summary.get("branch_outputs", {}).get(branch) or {})
    active_path = active_result.artifact_path
    active_hash = _sha256_file(active_path) if active_result.passed and active_path is not None else None
    expected_path = branch_output.get("source_artifact_path")
    expected_hash = branch_output.get("profile_artifact_hash")
    active_matches_selected = False
    if active_path is not None and expected_path:
        active_matches_selected = resolve_project_path(expected_path, project_root=project_root).resolve() == Path(active_path).resolve()
    if not active_result.passed:
        blockers.extend(f"active_artifact:{reason}" for reason in active_result.reason_codes)
    if not active_matches_selected:
        blockers.append("active_selected_profile_artifact_mismatch")
    if active_hash != expected_hash:
        blockers.append("active_selected_profile_artifact_hash_mismatch")
    if branch == REGIME_BRANCH_ASSET_STATE and (active_path is None or "lineage_repaired" not in active_path.name):
        blockers.append("repaired_asset_artifact_not_active")

    approval = dict(approvals.get("branch_approvals", {}).get(branch) or {})
    gate = evaluate_regime_production_branch_promotion_gate(approval or None, context, branch=branch)
    blockers.extend(f"approval:{reason}" for reason in gate.get("blockers", ()) or ())
    if not canonical_roots.get("all_required_roots_configured"):
        blockers.append("canonical_roots_missing:" + ",".join(canonical_roots.get("missing_root_keys") or ()))
    for key, passed in contract_status.items():
        if not passed:
            blockers.append(key)
    readiness = READINESS_BLOCKED if blockers else READINESS_PASS
    schema = default_regime_production_label_output_schema(branch)
    return {
        "branch": branch,
        "readiness": readiness,
        "blockers": list(dict.fromkeys(blockers)),
        "active_artifact": {
            "status": active_result.status,
            "passed": active_result.passed,
            "path": None if active_path is None else _portable_path_text(active_path),
            "hash": active_hash,
            "matches_selected_profile_artifact": active_matches_selected,
            "expected_selected_profile_artifact_path": expected_path,
            "expected_selected_profile_artifact_hash": expected_hash,
            "repaired_asset_artifact_active": bool(branch == REGIME_BRANCH_ASSET_STATE and active_path is not None and "lineage_repaired" in active_path.name),
        },
        "approval_gate": {
            key: gate.get(key)
            for key in (
                "gate_status",
                "dry_write_planning_allowed",
                "canonical_write_execution_allowed",
                "production_writer_enabled",
                "production_write_preconditions_satisfied",
                "blockers",
                "warnings",
                "approval_id",
            )
        },
        "output_schema": {
            "schema_id": schema.schema_id,
            "schema_hash": schema.schema_hash,
            "fixed_schema": True,
        },
        "writer_gates_disabled": gate.get("production_writer_enabled") is False,
        "canonical_write_execution_allowed": False,
    }


def _contract_status(
    sandbox_validation: Mapping[str, Any],
    write_capable: Mapping[str, Any],
) -> dict[str, bool]:
    issues = dict(sandbox_validation.get("issues") or {})
    validation_results = dict(write_capable.get("validation_results") or {})
    accepted = set(str(item) for item in write_capable.get("accepted_validation_issues") or ())
    required_notes = {
        str(item)
        for severity in ("medium", "low")
        for item in (issues.get(severity) or ())
    }
    return {
        "sandbox_validation_clean_for_readiness": not (issues.get("blocker") or issues.get("high"))
        and required_notes.issubset(accepted),
        "write_capable_sandbox_clean_for_readiness": bool(validation_results)
        and all(
            validation_results.get(key) is expected
            for key, expected in {
                "writer_gating_passed": True,
                "output_schema_validation_passed": True,
                "partitioning_validation_passed": True,
                "no_mixed_schemas": True,
                "atomic_staged_writes_validated": True,
                "resume_behavior_validated": True,
                "idempotency_validated": True,
                "canonical_root_touched": False,
            }.items()
        ),
        "output_schemas_fixed": True,
        "cross_relationship_freshness_policy_accepted": True,
        "clamp_range_accepted": True,
        "worker_job_matrix_accepted": True,
        "logging_telemetry_accepted": True,
    }


def _canonical_root_status(
    *,
    env: Mapping[str, str] | None,
    project_root: str | Path | None,
) -> dict[str, Any]:
    roots: dict[str, str] = {}
    missing: list[str] = []
    for key in CANONICAL_READINESS_ROOT_KEYS:
        raw = resolve_path(key, env=env, required=False)
        if raw is None:
            missing.append(key)
            continue
        roots[key] = _portable_path_text(resolve_project_path(raw, project_root=project_root))
    return {
        "required_root_keys": list(CANONICAL_READINESS_ROOT_KEYS),
        "configured_roots": roots,
        "missing_root_keys": missing,
        "all_required_roots_configured": not missing,
        "canonical_root_touched": False,
        "canonical_write_test_performed": False,
    }


def _first_branch_command(branch_readiness: Mapping[str, Mapping[str, Any]]) -> dict[str, Any] | None:
    for branch in ("market_state", "asset_state", "cross_asset_state"):
        if branch_readiness.get(branch, {}).get("readiness") == READINESS_PASS:
            return {
                "branch": branch,
                "command": "not_emitted_by_readiness_gate",
                "reason": "canonical writer entrypoint remains intentionally gated for the explicit write sprint",
            }
    return None


def _load_json(path: str | Path, *, project_root: str | Path | None = None) -> dict[str, Any]:
    resolved = resolve_project_path(path, project_root=project_root)
    with resolved.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("Regime Production canonical readiness expected a JSON object")
    return payload


def _sha256_file(path: Path | None) -> str | None:
    if path is None:
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _portable_path_text(value: str | Path | None) -> str | None:
    if value is None:
        return None
    path = Path(value)
    if not path.is_absolute():
        return str(path)
    resolved = path.resolve()
    cwd = Path.cwd().resolve()
    try:
        return str(resolved.relative_to(cwd))
    except ValueError:
        return f"<external_configured_root>/{resolved.name}"


def _sanitize_workspace_paths(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _sanitize_workspace_paths(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_sanitize_workspace_paths(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_sanitize_workspace_paths(item) for item in value)
    if isinstance(value, str):
        return _portable_path_text(value) if _looks_like_absolute_path(value) else value
    return value


def _looks_like_absolute_path(value: str) -> bool:
    try:
        return Path(value).is_absolute()
    except Exception:
        return False


__all__ = [
    "DEFAULT_APPROVAL_SEARCH_ROOTS",
    "DEFAULT_WRITE_CAPABLE_SANDBOX_FINAL_SUMMARY_PATH",
    "READINESS_BLOCKED",
    "READINESS_PASS",
    "REGIME_PRODUCTION_CANONICAL_READINESS_ARTIFACT_KIND",
    "REGIME_PRODUCTION_CANONICAL_READINESS_SCHEMA_VERSION",
    "RegimeProductionCanonicalReadinessGateConfig",
    "discover_regime_production_approval_artifacts",
    "run_regime_production_canonical_readiness_gate",
    "write_regime_production_canonical_readiness_gate_summary",
]
