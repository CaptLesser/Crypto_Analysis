from __future__ import annotations

import csv
import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.forecasting.common.path_config import PathConfigError, resolve_path
from src.regimes.asset_state.production_planner import plan_asset_state_production_no_write
from src.regimes.core.path_safety import validate_non_production_write_root
from src.regimes.core.paths import is_relative_to, resolve_project_path
from src.regimes.core.production_consumer import (
    REGIME_BRANCH_ASSET_STATE,
    REGIME_BRANCH_CROSS_ASSET_STATE,
    REGIME_BRANCH_MARKET_STATE,
    REGIME_PRODUCTION_BRANCHES,
)
from src.regimes.core.production_definition_planner import (
    plan_regime_production_definition_schedule_from_no_write_plan,
)
from src.regimes.core.production_output_contracts import CANONICAL_ROOT_KEYS, LOG_TELEMETRY_NAMESPACE
from src.regimes.core.production_reuse_cache import RegimeProductionPlannerRunCache
from src.regimes.core.root_resolution import resolve_regime_production_write_root
from src.regimes.core.serialization import to_jsonable
from src.regimes.cross_asset_state.production_planner import plan_cross_asset_state_production_no_write
from src.regimes.market_state.production_planner import plan_market_state_production_no_write


REGIME_PRODUCTION_TELEMETRY_SCHEMA_VERSION = 1
REGIME_PRODUCTION_TELEMETRY_ROOT_ARTIFACT_KIND = "regime_production_telemetry_root_contract"
REGIME_PRODUCTION_BRANCH_TELEMETRY_ARTIFACT_KIND = "regime_production_branch_dry_planner_telemetry"
REGIME_PRODUCTION_DRY_TELEMETRY_ARTIFACT_KIND = "regime_production_dry_planner_telemetry"

TELEMETRY_JSON_FILENAME = "regime_production_dry_planner_telemetry.json"
TELEMETRY_BRANCH_CSV_FILENAME = "regime_production_branch_telemetry.csv"


@dataclass(frozen=True)
class RegimeProductionTelemetryConfig:
    run_id: str
    branches: Sequence[str] = REGIME_PRODUCTION_BRANCHES
    asset_state_artifact_path: str | Path | None = None
    market_state_artifact_path: str | Path | None = None
    cross_asset_state_artifact_path: str | Path | None = None
    telemetry_root: str | Path | None = None
    env: Mapping[str, str] | None = None
    project_root: str | Path | None = None
    allow_explicit_dry_test_override: bool = False
    write_json: bool = True
    write_csv: bool = True

    def __post_init__(self) -> None:
        branches = tuple(_branch_name(branch) for branch in self.branches)
        if not branches:
            raise ValueError("Regime Production telemetry requires at least one branch")
        object.__setattr__(self, "run_id", _text(self.run_id, field_name="run_id"))
        object.__setattr__(self, "branches", branches)
        object.__setattr__(self, "env", dict(os.environ if self.env is None else self.env))


@dataclass(frozen=True)
class RegimeProductionTelemetryRootContract:
    run_id: str
    telemetry_root: str | Path
    root_source: str
    canonical_root_checks: Sequence[Mapping[str, Any]] = field(default_factory=tuple)
    telemetry_writer_enabled: bool = True
    production_writer_enabled: bool = False
    production_outputs_written: bool = False
    canonical_production_state_outputs_written: bool = False

    def __post_init__(self) -> None:
        root = validate_non_production_write_root(self.telemetry_root)
        checks = tuple(to_jsonable(dict(item)) for item in self.canonical_root_checks)
        if any(bool(item.get("collision")) for item in checks):
            raise ValueError("Regime Production telemetry root collides with a canonical root")
        if self.production_writer_enabled or self.production_outputs_written or self.canonical_production_state_outputs_written:
            raise ValueError("Regime Production telemetry root cannot enable production writes")
        object.__setattr__(self, "run_id", _text(self.run_id, field_name="run_id"))
        object.__setattr__(self, "telemetry_root", root)
        object.__setattr__(self, "root_source", _text(self.root_source, field_name="root_source"))
        object.__setattr__(self, "canonical_root_checks", checks)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": REGIME_PRODUCTION_TELEMETRY_SCHEMA_VERSION,
            "artifact_kind": REGIME_PRODUCTION_TELEMETRY_ROOT_ARTIFACT_KIND,
            "run_id": self.run_id,
            "telemetry_root": _portable_path_text(self.telemetry_root),
            "root_source": self.root_source,
            "root_kind": "configured_log_telemetry_root",
            "canonical_root_checks": [to_jsonable(dict(item)) for item in self.canonical_root_checks],
            "canonical_root_collision": False,
            "telemetry_artifact_write_allowed": bool(self.telemetry_writer_enabled),
            "writer_enabled": False,
            "production_writer_enabled": False,
            "production_labels_written": False,
            "production_outputs_written": False,
            "canonical_production_state_outputs_written": False,
        }


def resolve_regime_production_telemetry_root_contract(
    *,
    run_id: str,
    telemetry_root: str | Path | None = None,
    env: Mapping[str, str] | None = None,
    project_root: str | Path | None = None,
    allow_explicit_dry_test_override: bool = False,
) -> RegimeProductionTelemetryRootContract:
    source_env = env if env is not None else os.environ
    if telemetry_root is not None and str(telemetry_root).strip():
        if not allow_explicit_dry_test_override:
            raise PathConfigError("Explicit Regime Production telemetry roots are allowed only for dry/smoke tests")
        root = resolve_project_path(telemetry_root, project_root=project_root)
        source = "explicit_dry_test_override"
    else:
        raw = resolve_path("log_root", env=source_env, required=True)
        if raw is None:
            raise PathConfigError("Regime Production telemetry log_root is not configured")
        root = resolve_project_path(raw, project_root=project_root) / LOG_TELEMETRY_NAMESPACE / f"run_id={_safe_path_part(run_id)}"
        source = "path_config.log_root/regime_production/run_id"
    root = validate_non_production_write_root(root, project_root=project_root)
    checks = _canonical_root_checks(root, env=source_env, project_root=project_root)
    return RegimeProductionTelemetryRootContract(
        run_id=run_id,
        telemetry_root=root,
        root_source=source,
        canonical_root_checks=checks,
    )


def run_regime_production_dry_planner_telemetry(
    config: RegimeProductionTelemetryConfig | Mapping[str, Any],
) -> dict[str, Any]:
    cfg = config if isinstance(config, RegimeProductionTelemetryConfig) else RegimeProductionTelemetryConfig(**dict(config))
    root_contract = resolve_regime_production_telemetry_root_contract(
        run_id=cfg.run_id,
        telemetry_root=cfg.telemetry_root,
        env=cfg.env,
        project_root=cfg.project_root,
        allow_explicit_dry_test_override=cfg.allow_explicit_dry_test_override,
    )
    run_cache = RegimeProductionPlannerRunCache(cache_id=f"{cfg.run_id}_telemetry_cache")
    phase_elapsed: dict[str, float] = {}
    rss_start = _rss_bytes()
    started = time.perf_counter()
    branch_records = []
    for branch in cfg.branches:
        plan_started = time.perf_counter()
        plan = _build_no_write_plan(branch, cfg=cfg, run_cache=run_cache)
        phase_elapsed[f"{branch}.dry_plan_seconds"] = _elapsed(plan_started)

        definition_started = time.perf_counter()
        schedule = plan_regime_production_definition_schedule_from_no_write_plan(
            plan,
            sandbox_definition_root=_definition_plan_contract_root(cfg, branch),
            env=cfg.env,
            project_root=cfg.project_root,
            include_definition_records=False,
            max_sample_records=0,
        )
        phase_elapsed[f"{branch}.definition_plan_seconds"] = _elapsed(definition_started)
        branch_records.append(_branch_telemetry_record(plan, schedule.as_dict(include_records=False), run_id=cfg.run_id))
    phase_elapsed["total_planning_seconds"] = _elapsed(started)
    payload = _telemetry_payload(
        run_id=cfg.run_id,
        root_contract=root_contract,
        branch_records=tuple(branch_records),
        phase_elapsed=phase_elapsed,
        rss_start=rss_start,
        rss_end=_rss_bytes(),
        cache_telemetry=run_cache.as_dict(),
    )
    write_started = time.perf_counter()
    artifact_paths = write_regime_production_dry_planner_telemetry(
        payload,
        root_contract=root_contract,
        write_json=bool(cfg.write_json),
        write_csv=bool(cfg.write_csv),
    )
    payload["elapsed_by_phase"]["telemetry_persist_seconds"] = _elapsed(write_started)
    payload["artifact_paths"] = artifact_paths
    if cfg.write_json:
        _write_json(Path(root_contract.telemetry_root) / TELEMETRY_JSON_FILENAME, payload)
    return payload


def write_regime_production_dry_planner_telemetry(
    payload: Mapping[str, Any],
    *,
    root_contract: RegimeProductionTelemetryRootContract,
    write_json: bool = True,
    write_csv: bool = True,
) -> dict[str, str]:
    root = Path(root_contract.telemetry_root)
    root.mkdir(parents=True, exist_ok=True)
    artifact_paths: dict[str, str] = {}
    if write_json:
        json_path = root / TELEMETRY_JSON_FILENAME
        _write_json(json_path, payload)
        artifact_paths["json"] = _portable_path_text(json_path)
    if write_csv:
        csv_path = root / TELEMETRY_BRANCH_CSV_FILENAME
        _write_branch_csv(csv_path, payload.get("branches") or ())
        artifact_paths["branch_csv"] = _portable_path_text(csv_path)
    return artifact_paths


def _build_no_write_plan(branch: str, *, cfg: RegimeProductionTelemetryConfig, run_cache: RegimeProductionPlannerRunCache):
    branch_name = _branch_name(branch)
    if branch_name == REGIME_BRANCH_ASSET_STATE:
        if cfg.asset_state_artifact_path is None:
            raise ValueError("Regime Production telemetry requires asset_state_artifact_path for Asset-State")
        return plan_asset_state_production_no_write(cfg.asset_state_artifact_path, expected_cell_count=3204, run_cache=run_cache)
    if branch_name == REGIME_BRANCH_MARKET_STATE:
        if cfg.market_state_artifact_path is None:
            raise ValueError("Regime Production telemetry requires market_state_artifact_path for Market-State")
        return plan_market_state_production_no_write(cfg.market_state_artifact_path, run_cache=run_cache)
    if cfg.cross_asset_state_artifact_path is None:
        raise ValueError("Regime Production telemetry requires cross_asset_state_artifact_path for Cross-Asset-State")
    return plan_cross_asset_state_production_no_write(cfg.cross_asset_state_artifact_path, run_cache=run_cache)


def _definition_plan_contract_root(cfg: RegimeProductionTelemetryConfig, branch: str) -> Path:
    root, _source = resolve_regime_production_write_root(
        None,
        env=cfg.env,
        project_root=cfg.project_root,
        subdir=f"regime_telemetry_definition_plan_contracts/run_id={_safe_path_part(cfg.run_id)}/{_branch_name(branch)}",
    )
    return root


def _branch_telemetry_record(plan: Any, schedule_payload: Mapping[str, Any], *, run_id: str) -> dict[str, Any]:
    payload = plan.as_dict(include_units=False)
    telemetry = dict(payload.get("telemetry") or {})
    consumer = dict(payload.get("consumer_validation") or {})
    dry_plan = dict(payload.get("shared_dry_run_plan") or {})
    planner_contract = dict(payload.get("planner_contract") or {})
    job_matrix = dict(payload.get("job_matrix") or {})
    source_tail_summary = _source_tail_known_at_summary(plan.planning_units)
    warning_codes = _normalize_codes((*payload.get("warnings", ()), *telemetry.get("warnings", ())))
    reason_codes = _normalize_codes(
        (
            *consumer.get("reason_codes", ()),
            *payload.get("warnings", ()),
            *source_tail_summary.get("reason_codes", ()),
        )
    )
    branch = payload["branch"]
    counts = _branch_counts(branch, telemetry, payload)
    record = {
        "schema_version": REGIME_PRODUCTION_TELEMETRY_SCHEMA_VERSION,
        "artifact_kind": REGIME_PRODUCTION_BRANCH_TELEMETRY_ARTIFACT_KIND,
        "branch": branch,
        "run_id": run_id,
        "active_artifact": {
            "path": _portable_path_text(payload.get("artifact_path")),
            "hash": payload.get("profile_artifact_hash"),
        },
        "clamp_range": _clamp_range_summary(planner_contract, plan.planning_units),
        "planned_cells": int(payload.get("planning_unit_count") or 0),
        "counts": counts,
        "definition_refit_plan": {
            "target_count": int(schedule_payload.get("target_count") or 0),
            "definition_record_count": int(schedule_payload.get("definition_record_count") or 0),
            "cadence_checkpoint_counts": dict(schedule_payload.get("cadence_checkpoint_counts") or {}),
            "target_count_by_cadence": dict(schedule_payload.get("target_count_by_cadence") or {}),
            "definition_count_by_cadence": dict(schedule_payload.get("definition_count_by_cadence") or {}),
            "model_state_status_counts": dict(schedule_payload.get("model_state_status_counts") or {}),
            "per_bar_refit_enabled": False,
            "definition_files_written": False,
            "model_state_records_written": False,
        },
        "input_roots": _input_roots(payload, dry_plan),
        "source_tail_known_at_summary": source_tail_summary,
        "worker_profile": _worker_profile_summary(job_matrix),
        "warnings": warning_codes,
        "reason_codes": reason_codes,
        "unit_reason_code_counts": _unit_reason_code_counts(plan.planning_units),
        "production_gates": {
            "planner_contract_gate": planner_contract.get("production_gate_validation"),
            "dry_run_writer_enabled": bool(dry_plan.get("writer_enabled")),
            "production_writer_enabled": bool(payload.get("production_writer_enabled")),
            "writer_enabled": bool(payload.get("production_writer_enabled")),
            "production_labels_written": bool(payload.get("production_labels_written")),
            "production_outputs_written": bool(payload.get("production_outputs_written")),
            "canonical_production_state_outputs_written": bool(payload.get("canonical_production_state_outputs_written")),
            "production_writer_gates_fail_closed": bool(payload.get("production_writer_gates_fail_closed")),
        },
        "writer_enabled": False,
        "production_writer_enabled": False,
        "production_labels_written": False,
        "production_outputs_written": False,
        "canonical_production_state_outputs_written": False,
    }
    return to_jsonable(record)


def _telemetry_payload(
    *,
    run_id: str,
    root_contract: RegimeProductionTelemetryRootContract,
    branch_records: Sequence[Mapping[str, Any]],
    phase_elapsed: Mapping[str, float],
    rss_start: int | None,
    rss_end: int | None,
    cache_telemetry: Mapping[str, Any],
) -> dict[str, Any]:
    warnings = _normalize_codes(reason for record in branch_records for reason in record.get("warnings", ()))
    reason_codes = _normalize_codes(reason for record in branch_records for reason in record.get("reason_codes", ()))
    return to_jsonable(
        {
            "schema_version": REGIME_PRODUCTION_TELEMETRY_SCHEMA_VERSION,
            "artifact_kind": REGIME_PRODUCTION_DRY_TELEMETRY_ARTIFACT_KIND,
            "run_id": run_id,
            "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "telemetry_root_contract": root_contract.as_dict(),
            "branch_count": len(branch_records),
            "branches": [dict(record) for record in branch_records],
            "elapsed_by_phase": {str(key): round(float(value), 6) for key, value in phase_elapsed.items()},
            "rss": {
                "rss_start_bytes": rss_start,
                "rss_end_bytes": rss_end,
                "rss_delta_bytes": None if rss_start is None or rss_end is None else int(rss_end) - int(rss_start),
            },
            "warnings": warnings,
            "reason_codes": reason_codes,
            "reuse_cache_telemetry": to_jsonable(dict(cache_telemetry)),
            "telemetry_rows_per_branch_only": True,
            "per_row_telemetry_emitted": False,
            "labels_generated": False,
            "production_approved": False,
            "writer_enabled": False,
            "production_writer_enabled": False,
            "production_labels_written": False,
            "production_outputs_written": False,
            "canonical_production_state_outputs_written": False,
            "production_promotion_performed": False,
            "production_writer_gates_fail_closed": True,
        }
    )


def _branch_counts(branch: str, telemetry: Mapping[str, Any], payload: Mapping[str, Any]) -> dict[str, int]:
    if branch == REGIME_BRANCH_ASSET_STATE:
        return {
            "planned": int(payload.get("planning_unit_count") or 0),
            "selected": int(telemetry.get("selected_unit_count") or 0),
            "masked_unavailable": int(telemetry.get("masked_unavailable_unit_count") or 0),
            "skipped_or_filtered": int(telemetry.get("skipped_or_filtered_unit_count") or 0),
            "unavailable": int(telemetry.get("masked_unavailable_unit_count") or 0) + int(telemetry.get("skipped_or_filtered_unit_count") or 0),
        }
    if branch == REGIME_BRANCH_MARKET_STATE:
        return {
            "planned": int(payload.get("planning_unit_count") or 0),
            "selected": int(telemetry.get("selected_unit_count") or 0),
            "masked_unavailable": int(telemetry.get("masked_unavailable_unit_count") or 0),
            "skipped_or_filtered": int(telemetry.get("masked_or_skipped_count") or 0),
            "unavailable": int(telemetry.get("masked_unavailable_unit_count") or 0),
        }
    return {
        "planned": int(payload.get("planning_unit_count") or 0),
        "selected": int(telemetry.get("selected_unit_count") or 0),
        "diagnostic_only": int(telemetry.get("diagnostic_only_unit_count") or 0),
        "masked_unavailable": int(telemetry.get("masked_unavailable_unit_count") or 0),
        "skipped_or_filtered": int(telemetry.get("masked_or_skipped_cell_count") or 0),
        "unavailable": int(telemetry.get("masked_unavailable_unit_count") or 0) + int(telemetry.get("missing_cell_count") or 0),
        "missing": int(telemetry.get("missing_cell_count") or 0),
    }


def _clamp_range_summary(planner_contract: Mapping[str, Any], units: Sequence[Any]) -> dict[str, Any]:
    policy = dict(planner_contract.get("clamp_policy") or {})
    ranges = []
    for unit in units:
        normalized = dict(dict(unit.timestamp_plan or {}).get("normalized_clamp_range") or {})
        if normalized:
            ranges.append(normalized)
    return {
        "policy_id": policy.get("policy_id"),
        "historical_output_months": policy.get("historical_output_months"),
        "required_lookback_months": policy.get("required_lookback_months"),
        "runtime_boundaries_required": policy.get("runtime_boundaries_required"),
        "output_start_min": _min_value(item.get("output_start_ts") for item in ranges),
        "output_end_max": _max_value(item.get("output_end_ts") for item in ranges),
        "clamp_passed_count": sum(1 for item in ranges if item.get("passed") is True),
        "clamp_blocked_count": sum(1 for item in ranges if item.get("passed") is not True),
    }


def _input_roots(payload: Mapping[str, Any], dry_plan: Mapping[str, Any]) -> dict[str, Any]:
    checks = [dict(item) for item in payload.get("relationship_input_checks") or ()]
    relationship_roots = {
        str(item.get("name") or item.get("field")): {
            "root": item.get("root"),
            "root_source": item.get("root_source"),
            "status": item.get("status"),
            "configured_root_policy": item.get("configured_root_policy"),
        }
        for item in checks
        if item.get("root") not in (None, "")
    }
    active_artifact = Path(str(payload.get("artifact_path") or "")).parent
    return {
        "active_artifact_parent": _portable_path_text(active_artifact),
        "dry_run_planned_input_roots": to_jsonable(dict(dry_plan.get("planned_input_roots") or {})),
        "relationship_input_roots": to_jsonable(relationship_roots),
    }


def _source_tail_known_at_summary(units: Sequence[Any]) -> dict[str, Any]:
    source_tails: list[Any] = []
    known_ats: list[Any] = []
    relationship_tails: list[Any] = []
    relationship_known_ats: list[Any] = []
    reasons: list[str] = []
    for unit in units:
        model_state = dict(unit.model_state_definition or {})
        source_tail = model_state.get("source_tail_ts")
        known_at = model_state.get("definition_known_at_ts")
        if source_tail in (None, ""):
            reasons.append("source_tail_missing")
        else:
            source_tails.append(source_tail)
        if known_at in (None, ""):
            reasons.append("known_at_missing")
        else:
            known_ats.append(known_at)
        relationship = dict(dict(unit.timestamp_plan or {}).get("relationship_input_history_check") or {})
        if relationship:
            if relationship.get("relationship_input_tail_ts") in (None, ""):
                reasons.append("relationship_input_tail_missing")
            else:
                relationship_tails.append(relationship.get("relationship_input_tail_ts"))
            if relationship.get("relationship_known_at_ts") in (None, ""):
                reasons.append("relationship_known_at_missing")
            else:
                relationship_known_ats.append(relationship.get("relationship_known_at_ts"))
    return {
        "unit_count": len(units),
        "source_tail_min": _min_value(source_tails),
        "source_tail_max": _max_value(source_tails),
        "source_tail_missing_count": len(units) - len(source_tails),
        "known_at_min": _min_value(known_ats),
        "known_at_max": _max_value(known_ats),
        "known_at_missing_count": len(units) - len(known_ats),
        "relationship_input_tail_min": _min_value(relationship_tails),
        "relationship_input_tail_max": _max_value(relationship_tails),
        "relationship_known_at_min": _min_value(relationship_known_ats),
        "relationship_known_at_max": _max_value(relationship_known_ats),
        "reason_codes": _normalize_codes(reasons),
    }


def _worker_profile_summary(job_matrix: Mapping[str, Any]) -> dict[str, Any]:
    profile = dict(job_matrix.get("worker_profile") or {})
    return {
        "workers": profile.get("workers"),
        "effective_workers": profile.get("effective_workers"),
        "model_threads": profile.get("model_threads"),
        "writer_workers": profile.get("writer_workers"),
        "backend": profile.get("backend"),
        "batch_size": profile.get("batch_size"),
        "grouping_fields": list(profile.get("grouping_fields") or ()),
        "parent_single_finalizer": bool(profile.get("parent_single_finalizer")),
        "workers_write_outputs": bool(profile.get("workers_write_outputs")),
    }


def _unit_reason_code_counts(units: Sequence[Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for unit in units:
        for reason in unit.reason_codes:
            text = str(reason or "").strip()
            if text:
                counts[text] = int(counts.get(text, 0)) + 1
    return dict(sorted(counts.items()))


def _write_branch_csv(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    fields = (
        "run_id",
        "branch",
        "active_artifact_path",
        "active_artifact_hash",
        "planned_cells",
        "selected",
        "masked_unavailable",
        "unavailable",
        "definition_record_count",
        "writer_enabled",
        "production_writer_enabled",
        "warning_count",
        "reason_code_count",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            counts = dict(record.get("counts") or {})
            definition = dict(record.get("definition_refit_plan") or {})
            active = dict(record.get("active_artifact") or {})
            writer.writerow(
                {
                    "run_id": record.get("run_id"),
                    "branch": record.get("branch"),
                    "active_artifact_path": active.get("path"),
                    "active_artifact_hash": active.get("hash"),
                    "planned_cells": record.get("planned_cells"),
                    "selected": counts.get("selected", 0),
                    "masked_unavailable": counts.get("masked_unavailable", 0),
                    "unavailable": counts.get("unavailable", 0),
                    "definition_record_count": definition.get("definition_record_count", 0),
                    "writer_enabled": record.get("writer_enabled"),
                    "production_writer_enabled": record.get("production_writer_enabled"),
                    "warning_count": len(record.get("warnings") or ()),
                    "reason_code_count": len(record.get("reason_codes") or ()),
                }
            )


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(to_jsonable(dict(payload)), indent=2, sort_keys=True), encoding="utf-8")


def _canonical_root_checks(
    telemetry_root: Path,
    *,
    env: Mapping[str, str],
    project_root: str | Path | None,
) -> tuple[dict[str, Any], ...]:
    checks: list[dict[str, Any]] = []
    root = Path(telemetry_root).resolve()
    for key in CANONICAL_ROOT_KEYS:
        raw = resolve_path(key, env=env, required=False)
        if raw is None:
            checks.append({"root_key": key, "configured": False, "collision": False})
            continue
        candidate = resolve_project_path(raw, project_root=project_root)
        collision = is_relative_to(root, candidate) or is_relative_to(candidate, root)
        checks.append({"root_key": key, "configured": True, "collision": bool(collision)})
    return tuple(checks)


def _rss_bytes() -> int | None:
    try:
        import psutil  # type: ignore

        return int(psutil.Process(os.getpid()).memory_info().rss)
    except Exception:
        return None


def _elapsed(started: float) -> float:
    return max(0.0, time.perf_counter() - float(started))


def _normalize_codes(values: Any) -> list[str]:
    out = []
    for value in values or ():
        text = str(value or "").strip()
        if text:
            out.append(text)
    return sorted(dict.fromkeys(out))


def _min_value(values: Any) -> Any:
    cleaned = [value for value in values or () if value not in (None, "")]
    if not cleaned:
        return None
    return min(cleaned, key=lambda value: (_orderable(value), str(value)))


def _max_value(values: Any) -> Any:
    cleaned = [value for value in values or () if value not in (None, "")]
    if not cleaned:
        return None
    return max(cleaned, key=lambda value: (_orderable(value), str(value)))


def _orderable(value: Any) -> float:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    text = str(value or "").strip()
    try:
        return float(text)
    except Exception:
        pass
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()
    except Exception:
        return 0.0


def _branch_name(value: object) -> str:
    text = _text(value, field_name="branch")
    aliases = {
        "asset": REGIME_BRANCH_ASSET_STATE,
        "asset-state": REGIME_BRANCH_ASSET_STATE,
        "market": REGIME_BRANCH_MARKET_STATE,
        "market-state": REGIME_BRANCH_MARKET_STATE,
        "cross_asset": REGIME_BRANCH_CROSS_ASSET_STATE,
        "cross-asset-state": REGIME_BRANCH_CROSS_ASSET_STATE,
    }
    resolved = aliases.get(text, text)
    if resolved not in REGIME_PRODUCTION_BRANCHES:
        raise ValueError(f"Unsupported Regime Production branch: {value!r}")
    return resolved


def _text(value: object, *, field_name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"Regime Production telemetry {field_name} must be non-empty")
    return text


def _safe_path_part(value: object) -> str:
    text = str(value or "").strip().replace("/", "_").replace("\\", "_").replace(":", "_").replace("|", "_")
    return text or "unknown"


def _portable_path_text(value: object) -> str | None:
    if value in (None, ""):
        return None
    path = Path(str(value)).resolve()
    cwd = Path.cwd().resolve()
    try:
        return str(path.relative_to(cwd))
    except ValueError:
        return f"<external_configured_root>/{path.name}"


__all__ = [
    "REGIME_PRODUCTION_BRANCH_TELEMETRY_ARTIFACT_KIND",
    "REGIME_PRODUCTION_DRY_TELEMETRY_ARTIFACT_KIND",
    "REGIME_PRODUCTION_TELEMETRY_ROOT_ARTIFACT_KIND",
    "REGIME_PRODUCTION_TELEMETRY_SCHEMA_VERSION",
    "TELEMETRY_BRANCH_CSV_FILENAME",
    "TELEMETRY_JSON_FILENAME",
    "RegimeProductionTelemetryConfig",
    "RegimeProductionTelemetryRootContract",
    "resolve_regime_production_telemetry_root_contract",
    "run_regime_production_dry_planner_telemetry",
    "write_regime_production_dry_planner_telemetry",
]
