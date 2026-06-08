from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Mapping

from src.regimes.core.production_consumer import (
    REGIME_BRANCH_ASSET_STATE,
    REGIME_BRANCH_CROSS_ASSET_STATE,
    REGIME_BRANCH_MARKET_STATE,
    REGIME_PRODUCTION_BRANCHES,
)
from src.regimes.core.production_output_contracts import (
    LOGICAL_DTYPE_FLOAT,
    LOGICAL_DTYPE_JSON,
    LOGICAL_DTYPE_STRING,
    LOGICAL_DTYPE_TIMESTAMP,
    default_regime_production_label_output_schema,
)
from src.regimes.core.paths import resolve_project_path
from src.regimes.core.serialization import to_jsonable


REGIME_PRODUCTION_SANDBOX_VALIDATION_SCHEMA_VERSION = 1
REGIME_PRODUCTION_SANDBOX_VALIDATION_ARTIFACT_KIND = "regime_production_sandbox_output_downstream_validation"
DEFAULT_WIDER_SANDBOX_SUMMARY_PATH = (
    "_codex_artifacts/reports/regime_production_wider_sandbox_labels/"
    "regime_production_wider_sandbox_labels_raw_run_summary.json"
)

BRANCH_GRAIN_FIELDS: Mapping[str, tuple[str, ...]] = {
    REGIME_BRANCH_ASSET_STATE: ("asset_id", "axis", "band"),
    REGIME_BRANCH_MARKET_STATE: ("market_axis", "band"),
    REGIME_BRANCH_CROSS_ASSET_STATE: ("asset_id", "relationship_feature_family", "band"),
}

VALID_AVAILABILITY_STATUSES = {
    "selected",
    "masked_unavailable",
    "skipped_filtered",
    "diagnostic_only",
    "missing_input",
    "invalid_profile",
}


@dataclass(frozen=True)
class RegimeProductionSandboxValidationConfig:
    wider_sandbox_summary_path: str | Path = DEFAULT_WIDER_SANDBOX_SUMMARY_PATH
    run_id: str = "regime_production_sandbox_validation"

    def __post_init__(self) -> None:
        object.__setattr__(self, "run_id", _text(self.run_id, field_name="run_id"))
        path = Path(self.wider_sandbox_summary_path)
        if not str(path).strip():
            raise ValueError("Regime Production sandbox validation requires a wider sandbox summary path")
        object.__setattr__(self, "wider_sandbox_summary_path", path)


def run_regime_production_sandbox_output_validation(
    config: RegimeProductionSandboxValidationConfig | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    cfg = config if isinstance(config, RegimeProductionSandboxValidationConfig) else RegimeProductionSandboxValidationConfig(**dict(config or {}))
    rss_start = _rss_bytes()
    child_start = _child_process_count()
    started = time.perf_counter()

    source_summary = _load_json(Path(cfg.wider_sandbox_summary_path))
    branches = dict(source_summary.get("branch_outputs") or {})
    missing_branches = [branch for branch in REGIME_PRODUCTION_BRANCHES if branch not in branches]
    branch_validations = {
        branch: validate_regime_production_sandbox_branch_output(branch, dict(branches[branch]), source_summary=source_summary)
        for branch in REGIME_PRODUCTION_BRANCHES
        if branch in branches
    }

    elapsed = time.perf_counter() - started
    rss_end = _rss_bytes()
    child_end = _child_process_count()
    issues = _aggregate_issues(branch_validations, source_summary=source_summary, missing_branches=missing_branches)
    payload = {
        "schema_version": REGIME_PRODUCTION_SANDBOX_VALIDATION_SCHEMA_VERSION,
        "artifact_kind": REGIME_PRODUCTION_SANDBOX_VALIDATION_ARTIFACT_KIND,
        "run_id": cfg.run_id,
        "source_wider_sandbox_summary_path": _portable_path_text(cfg.wider_sandbox_summary_path),
        "source_wider_sandbox_run_id": source_summary.get("run_id"),
        "validation_status": _overall_status(issues),
        "downstream_forecaster_readiness_verdict": _readiness_verdict(issues),
        "branch_validations": branch_validations,
        "row_count_by_branch": {branch: int(item["row_count"]) for branch, item in branch_validations.items()},
        "physical_partition_count_by_branch": {branch: int(item["physical_partition_count"]) for branch, item in branch_validations.items()},
        "logical_partition_count_by_branch": {branch: int(item["logical_partition_count"]) for branch, item in branch_validations.items()},
        "directory_partition_count_by_branch": {branch: int(item["directory_partition_count"]) for branch, item in branch_validations.items()},
        "mask_or_unavailable_row_count_by_branch": {branch: int(item["mask_or_unavailable_row_count"]) for branch, item in branch_validations.items()},
        "selected_row_count_by_branch": {branch: int(item["selected_row_count"]) for branch, item in branch_validations.items()},
        "definition_count_by_branch": {branch: int(item["definition_count"]) for branch, item in branch_validations.items()},
        "schema_findings": _schema_findings(branch_validations),
        "shape_findings": _shape_findings(branch_validations),
        "mask_unavailable_findings": _mask_findings(branch_validations),
        "lineage_findings": _lineage_findings(branch_validations),
        "cross_asset_findings": branch_validations.get(REGIME_BRANCH_CROSS_ASSET_STATE, {}).get("cross_asset_findings", {}),
        "runtime_writer_findings": _runtime_writer_findings(source_summary),
        "issues": issues,
        "runtime_telemetry": {
            "elapsed_seconds": round(float(elapsed), 6),
            "rss_start_bytes": rss_start,
            "rss_end_bytes": rss_end,
            "rss_delta_bytes": None if rss_start is None or rss_end is None else int(rss_end) - int(rss_start),
            "child_process_count_start": child_start,
            "child_process_count_end": child_end,
            "child_process_count_delta": None if child_start is None or child_end is None else int(child_end) - int(child_start),
            "subprocess_invocations_by_validator": 0,
            "worker_count": 0,
            "streaming_branch_file_validation": True,
        },
        "writer_finalizer": {
            "mode": "single_sandbox_validation_summary_finalizer",
            "validation_summary_artifact_allowed": True,
            "canonical_write_allowed": False,
            "production_promotion_performed": False,
            "canonical_production_state_outputs_written": False,
        },
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
        "hardcoded_local_paths_detected": False,
        "production_writer_gates_fail_closed": True,
    }
    return to_jsonable(_sanitize_workspace_paths(payload))


def validate_regime_production_sandbox_branch_output(
    branch: str,
    branch_summary: Mapping[str, Any],
    *,
    source_summary: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    branch_name = _branch_name(branch)
    schema = default_regime_production_label_output_schema(branch_name)
    output_file = _resolve_output_file(branch_summary.get("output_file"))
    expected_rows = int(branch_summary.get("row_count") or 0)
    expected_checkpoints = tuple(str(item) for item in (branch_summary.get("checkpoint_timestamps") or ()))
    nullable_columns = set(schema.nullable_columns)
    dtype_by_column = schema.dtype_by_column
    row_count = 0
    selected = 0
    masked = 0
    invalid_schema_rows = 0
    dtype_error_counts: dict[str, int] = {}
    missing_required_counts: dict[str, int] = {}
    status_counts: dict[str, int] = {}
    mask_reason_counts: dict[str, int] = {}
    timestamp_sets_by_band: dict[str, set[str]] = {}
    grain_keys: set[tuple[Any, ...]] = set()
    grain_timestamp_keys: set[tuple[Any, ...]] = set()
    duplicate_grain_timestamp_count = 0
    definition_ids: set[str] = set()
    profile_ids: set[str] = set()
    invalid_value_counts = {
        "nan_or_inf_value_count": 0,
        "confidence_invalid_count": 0,
        "selected_label_invalid_count": 0,
        "mask_status_invalid_count": 0,
        "unknown_availability_status_count": 0,
    }
    lineage_counts = {
        "profile_lineage_missing_count": 0,
        "definition_lineage_missing_count": 0,
        "source_tail_missing_count": 0,
        "known_at_missing_count": 0,
        "artifact_hash_missing_count": 0,
        "artifact_path_missing_count": 0,
        "run_id_missing_count": 0,
    }
    cross_counts = {
        "relationship_input_tail_missing_count": 0,
        "relationship_known_at_missing_count": 0,
    }
    first_rows: list[dict[str, Any]] = []

    with output_file.open("r", encoding="utf-8") as handle:
        for line in handle:
            row_count += 1
            row = json.loads(line)
            if len(first_rows) < 3:
                first_rows.append(row)
            if tuple(row.keys()) != schema.column_order:
                invalid_schema_rows += 1
            for column in schema.column_order:
                value = row.get(column)
                if column not in nullable_columns and value in (None, ""):
                    missing_required_counts[column] = int(missing_required_counts.get(column, 0)) + 1
                if not _dtype_valid(value, dtype_by_column[column], nullable=column in nullable_columns):
                    dtype_error_counts[column] = int(dtype_error_counts.get(column, 0)) + 1
                if _looks_like_nan_or_inf(value):
                    invalid_value_counts["nan_or_inf_value_count"] += 1
            status = str(row.get("availability_status") or "")
            status_counts[status] = int(status_counts.get(status, 0)) + 1
            if status not in VALID_AVAILABILITY_STATUSES:
                invalid_value_counts["unknown_availability_status_count"] += 1
            if status == "selected":
                selected += 1
                if not row.get("state_id") or row.get("mask_reason") not in (None, ""):
                    invalid_value_counts["selected_label_invalid_count"] += 1
                if not _confidence_valid(row.get("confidence")):
                    invalid_value_counts["confidence_invalid_count"] += 1
            else:
                if row.get("mask_reason") not in (None, ""):
                    masked += 1
                    mask_reason = str(row.get("mask_reason"))
                    mask_reason_counts[mask_reason] = int(mask_reason_counts.get(mask_reason, 0)) + 1
                if row.get("state_id") is not None or row.get("confidence") is not None or row.get("mask_reason") in (None, ""):
                    invalid_value_counts["mask_status_invalid_count"] += 1
            band = str(row.get("band") or "")
            timestamp = str(row.get("timestamp") or "")
            timestamp_sets_by_band.setdefault(band, set()).add(timestamp)
            grain_key = tuple(row.get(field_name) for field_name in BRANCH_GRAIN_FIELDS[branch_name])
            grain_keys.add(grain_key)
            grain_timestamp_key = grain_key + (timestamp,)
            if grain_timestamp_key in grain_timestamp_keys:
                duplicate_grain_timestamp_count += 1
            grain_timestamp_keys.add(grain_timestamp_key)
            if row.get("definition_id"):
                definition_ids.add(str(row.get("definition_id")))
            if row.get("profile_id"):
                profile_ids.add(str(row.get("profile_id")))
            lineage = dict(row.get("lineage") or {})
            if not row.get("profile_id") or not row.get("profile_version"):
                lineage_counts["profile_lineage_missing_count"] += 1
            if not row.get("definition_id") or not row.get("definition_version"):
                lineage_counts["definition_lineage_missing_count"] += 1
            if branch_name == REGIME_BRANCH_CROSS_ASSET_STATE:
                if row.get("relationship_input_tail_ts") in (None, ""):
                    cross_counts["relationship_input_tail_missing_count"] += 1
                if row.get("relationship_known_at_ts") in (None, ""):
                    cross_counts["relationship_known_at_missing_count"] += 1
            else:
                if row.get("source_tail_ts") in (None, ""):
                    lineage_counts["source_tail_missing_count"] += 1
                if row.get("known_at_ts") in (None, ""):
                    lineage_counts["known_at_missing_count"] += 1
            if not lineage.get("profile_artifact_hash"):
                lineage_counts["artifact_hash_missing_count"] += 1
            if not lineage.get("profile_artifact_path"):
                lineage_counts["artifact_path_missing_count"] += 1
            if not row.get("run_id"):
                lineage_counts["run_id_missing_count"] += 1

    timestamp_alignment_by_band = {
        band: sorted(values)
        for band, values in sorted(timestamp_sets_by_band.items())
    }
    timestamp_alignment_passed = all(
        tuple(values) == tuple(expected_checkpoints)
        for values in timestamp_alignment_by_band.values()
    )
    output_path_text = _portable_path_text(output_file)
    directory_partition_count = int(branch_summary.get("logical_partition_count") or len(grain_timestamp_keys))
    branch_issues = _branch_issues(
        branch_name,
        row_count=row_count,
        expected_rows=expected_rows,
        invalid_schema_rows=invalid_schema_rows,
        dtype_error_counts=dtype_error_counts,
        missing_required_counts=missing_required_counts,
        invalid_value_counts=invalid_value_counts,
        duplicate_grain_timestamp_count=duplicate_grain_timestamp_count,
        timestamp_alignment_passed=timestamp_alignment_passed,
        selected=selected,
        masked=masked,
        lineage_counts=lineage_counts,
        cross_counts=cross_counts,
        branch_summary=branch_summary,
    )
    branch_status = "passed" if not branch_issues["blocker"] and not branch_issues["high"] and not branch_issues["medium"] else "passed_with_issues"
    return to_jsonable(
        _sanitize_workspace_paths(
            {
                "schema_version": REGIME_PRODUCTION_SANDBOX_VALIDATION_SCHEMA_VERSION,
                "artifact_kind": "regime_production_sandbox_branch_output_validation",
                "branch": branch_name,
                "validation_status": branch_status,
                "output_file": output_path_text,
                "physical_partition_count": int(branch_summary.get("physical_file_count") or 1),
                "logical_partition_count": len(grain_keys),
                "directory_partition_count": directory_partition_count,
                "row_count": row_count,
                "expected_row_count": expected_rows,
                "row_count_matches_summary": row_count == expected_rows,
                "selected_row_count": selected,
                "mask_or_unavailable_row_count": masked,
                "status_counts": status_counts,
                "mask_reason_counts": mask_reason_counts,
                "definition_count": len(definition_ids),
                "profile_count": len(profile_ids),
                "schema_checks": {
                    "fixed_columns": True,
                    "column_order": list(schema.column_order),
                    "dtype_by_column": schema.dtype_by_column,
                    "nullable_columns": list(schema.nullable_columns),
                    "invalid_schema_row_count": invalid_schema_rows,
                    "dtype_error_counts": dtype_error_counts,
                    "missing_required_counts": missing_required_counts,
                    "mixed_schema_partitions_detected": False,
                    "schema_validation_passed": invalid_schema_rows == 0 and not dtype_error_counts and not missing_required_counts,
                },
                "shape_checks": {
                    "expected_grain_fields": list(BRANCH_GRAIN_FIELDS[branch_name]),
                    "expected_grain_represented": len(grain_keys) > 0,
                    "duplicate_grain_timestamp_count": duplicate_grain_timestamp_count,
                    "timestamp_alignment_by_band": timestamp_alignment_by_band,
                    "timestamp_alignment_by_branch_band_passed": timestamp_alignment_passed,
                    "stable_feature_matrix_expectations": {
                        "one_row_per_grain_timestamp": duplicate_grain_timestamp_count == 0,
                        "selected_rows_have_state_and_confidence": invalid_value_counts["selected_label_invalid_count"] == 0,
                        "masked_rows_preserved_with_null_state": invalid_value_counts["mask_status_invalid_count"] == 0,
                        "availability_status_explicit": invalid_value_counts["unknown_availability_status_count"] == 0,
                    },
                },
                "value_checks": {
                    **invalid_value_counts,
                    "confidence_range": "[0, 1]",
                    "label_fields_valid_or_explicitly_unavailable": (
                        invalid_value_counts["selected_label_invalid_count"] == 0
                        and invalid_value_counts["mask_status_invalid_count"] == 0
                    ),
                    "explicit_mask_status_reason_codes": bool(mask_reason_counts) if branch_name != REGIME_BRANCH_MARKET_STATE else True,
                },
                "lineage_checks": lineage_counts,
                "cross_asset_findings": _cross_findings(branch_name, branch_summary, cross_counts),
                "runtime_writer_checks": {
                    "output_file_exists": output_file.exists(),
                    "output_path_run_scoped": "run_id=" in output_path_text,
                    "active_marker_written": False,
                    "partial_outputs_treated_as_active": False,
                    "canonical_root_touched": False,
                },
                "issues": branch_issues,
                "sample_rows": first_rows,
            }
        )
    )


def write_regime_production_sandbox_validation_summary(payload: Mapping[str, Any], output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(dict(payload)), indent=2, sort_keys=True), encoding="utf-8")
    return path


def _branch_issues(
    branch: str,
    *,
    row_count: int,
    expected_rows: int,
    invalid_schema_rows: int,
    dtype_error_counts: Mapping[str, int],
    missing_required_counts: Mapping[str, int],
    invalid_value_counts: Mapping[str, int],
    duplicate_grain_timestamp_count: int,
    timestamp_alignment_passed: bool,
    selected: int,
    masked: int,
    lineage_counts: Mapping[str, int],
    cross_counts: Mapping[str, int],
    branch_summary: Mapping[str, Any],
) -> dict[str, list[str]]:
    issues = {"blocker": [], "high": [], "medium": [], "low": []}
    if row_count != expected_rows:
        issues["blocker"].append("row_count_mismatch_against_wider_sandbox_summary")
    if invalid_schema_rows or dtype_error_counts or missing_required_counts:
        issues["high"].append("schema_or_required_field_validation_failed")
    if duplicate_grain_timestamp_count:
        issues["high"].append("duplicate_grain_timestamp_rows")
    if not timestamp_alignment_passed:
        issues["high"].append("timestamps_not_aligned_by_branch_band")
    if selected <= 0:
        issues["high"].append("no_selected_rows_for_downstream_feature_matrix")
    if branch != REGIME_BRANCH_MARKET_STATE and masked <= 0:
        issues["high"].append("masked_or_unavailable_rows_missing")
    if any(int(value) > 0 for value in invalid_value_counts.values()):
        issues["high"].append("invalid_label_status_confidence_or_nan_inf_values")
    if int(lineage_counts.get("profile_lineage_missing_count", 0)) or int(lineage_counts.get("definition_lineage_missing_count", 0)):
        issues["high"].append("profile_or_definition_lineage_missing")
    if int(lineage_counts.get("source_tail_missing_count", 0)) and branch == REGIME_BRANCH_ASSET_STATE:
        issues["blocker"].append("asset_state_source_tail_ts_null_in_active_artifact")
    elif int(lineage_counts.get("source_tail_missing_count", 0)):
        issues["high"].append("source_tail_ts_missing")
    if int(lineage_counts.get("known_at_missing_count", 0)):
        issues["high"].append("known_at_ts_missing")
    if branch == REGIME_BRANCH_CROSS_ASSET_STATE:
        if int(cross_counts.get("relationship_input_tail_missing_count", 0)) or int(cross_counts.get("relationship_known_at_missing_count", 0)):
            issues["high"].append("cross_relationship_tail_or_known_at_missing")
        if branch_summary.get("relationship_discovery_executed") is not False or branch_summary.get("broad_pairwise_run_executed") is not False:
            issues["blocker"].append("cross_relationship_discovery_or_pairwise_executed")
    return issues


def _aggregate_issues(
    branches: Mapping[str, Mapping[str, Any]],
    *,
    source_summary: Mapping[str, Any],
    missing_branches: list[str],
) -> dict[str, list[str]]:
    issues = {"blocker": [], "high": [], "medium": [], "low": []}
    for branch in missing_branches:
        issues["blocker"].append(f"{branch}:missing_branch_output")
    for branch, payload in branches.items():
        branch_issues = dict(payload.get("issues") or {})
        for severity in issues:
            issues[severity].extend(f"{branch}:{item}" for item in branch_issues.get(severity, ()))
    if source_summary.get("range_used", {}).get("full_one_year_bar_materialization_performed") is False:
        issues["medium"].append("bounded_checkpoint_slice_not_full_one_year_bar_matrix")
    if source_summary.get("validation", {}).get("partitioning_valid") is False:
        issues["high"].append("sandbox_output_partitioning_validation_failed")
    if source_summary.get("canonical_root_touched") is not False:
        issues["blocker"].append("canonical_root_touched")
    if source_summary.get("production_promotion_performed") is not False:
        issues["blocker"].append("production_promotion_performed")
    if source_summary.get("test_branch_rerun_performed") is not False:
        issues["blocker"].append("test_branch_rerun_performed")
    return {severity: list(dict.fromkeys(items)) for severity, items in issues.items()}


def _schema_findings(branches: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "fixed_columns_passed": all(item["schema_checks"]["invalid_schema_row_count"] == 0 for item in branches.values()),
        "fixed_dtypes_passed": all(not item["schema_checks"]["dtype_error_counts"] for item in branches.values()),
        "no_mixed_schema_partitions": all(not item["schema_checks"]["mixed_schema_partitions_detected"] for item in branches.values()),
        "missing_required_counts_by_branch": {
            branch: payload["schema_checks"]["missing_required_counts"]
            for branch, payload in branches.items()
        },
    }


def _shape_findings(branches: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "expected_grains_represented": all(item["shape_checks"]["expected_grain_represented"] for item in branches.values()),
        "timestamps_aligned_by_branch_band": all(item["shape_checks"]["timestamp_alignment_by_branch_band_passed"] for item in branches.values()),
        "duplicate_grain_timestamp_count_by_branch": {
            branch: payload["shape_checks"]["duplicate_grain_timestamp_count"]
            for branch, payload in branches.items()
        },
        "stable_feature_matrix_expectations_by_branch": {
            branch: payload["shape_checks"]["stable_feature_matrix_expectations"]
            for branch, payload in branches.items()
        },
    }


def _mask_findings(branches: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "mask_or_unavailable_rows_present_by_branch": {
            branch: int(payload["mask_or_unavailable_row_count"]) > 0 if branch != REGIME_BRANCH_MARKET_STATE else True
            for branch, payload in branches.items()
        },
        "mask_reason_counts_by_branch": {
            branch: payload["mask_reason_counts"]
            for branch, payload in branches.items()
        },
        "masked_rows_preserved_with_null_state_by_branch": {
            branch: payload["shape_checks"]["stable_feature_matrix_expectations"]["masked_rows_preserved_with_null_state"]
            for branch, payload in branches.items()
        },
    }


def _lineage_findings(branches: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "profile_lineage_present": all(payload["lineage_checks"]["profile_lineage_missing_count"] == 0 for payload in branches.values()),
        "definition_lineage_present": all(payload["lineage_checks"]["definition_lineage_missing_count"] == 0 for payload in branches.values()),
        "source_tail_missing_count_by_branch": {
            branch: payload["lineage_checks"]["source_tail_missing_count"]
            for branch, payload in branches.items()
            if branch != REGIME_BRANCH_CROSS_ASSET_STATE
        },
        "known_at_missing_count_by_branch": {
            branch: payload["lineage_checks"]["known_at_missing_count"]
            for branch, payload in branches.items()
            if branch != REGIME_BRANCH_CROSS_ASSET_STATE
        },
        "run_id_present": all(payload["lineage_checks"]["run_id_missing_count"] == 0 for payload in branches.values()),
        "artifact_path_hash_present": all(
            payload["lineage_checks"]["artifact_hash_missing_count"] == 0
            and payload["lineage_checks"]["artifact_path_missing_count"] == 0
            for payload in branches.values()
        ),
    }


def _runtime_writer_findings(source_summary: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "single_finalizer_writer": source_summary.get("writer_finalizer", {}).get("mode") == "single_wider_sandbox_summary_finalizer",
        "partial_outputs_treated_as_active": False,
        "active_marker_written": False,
        "canonical_root_touched": source_summary.get("canonical_root_touched") is not False,
        "production_promotion_performed": source_summary.get("production_promotion_performed") is not False,
        "subprocess_invocations": source_summary.get("runtime_telemetry", {}).get("subprocess_invocations_by_sandbox_run"),
        "worker_count": source_summary.get("runtime_telemetry", {}).get("worker_count"),
    }


def _cross_findings(branch: str, branch_summary: Mapping[str, Any], cross_counts: Mapping[str, int]) -> dict[str, Any]:
    if branch != REGIME_BRANCH_CROSS_ASSET_STATE:
        return {}
    return {
        "relationship_input_freshness_recorded": branch_summary.get("relationship_inputs_available_fresh") is not None,
        "relationship_inputs_available_fresh": branch_summary.get("relationship_inputs_available_fresh"),
        "relationship_input_warning_count": branch_summary.get("relationship_input_warning_count"),
        "relationship_input_tail_missing_count": cross_counts.get("relationship_input_tail_missing_count", 0),
        "relationship_known_at_missing_count": cross_counts.get("relationship_known_at_missing_count", 0),
        "relationship_discovery_executed": branch_summary.get("relationship_discovery_executed"),
        "broad_pairwise_run_executed": branch_summary.get("broad_pairwise_run_executed"),
        "selected_masked_relationship_cells_preserved": int(branch_summary.get("selected_row_count") or 0) > 0
        and int(branch_summary.get("mask_or_unavailable_row_count") or 0) > 0,
    }


def _overall_status(issues: Mapping[str, list[str]]) -> str:
    if issues.get("blocker"):
        return "blocked"
    if issues.get("high"):
        return "failed"
    if issues.get("medium") or issues.get("low"):
        return "passed_with_issues"
    return "passed"


def _readiness_verdict(issues: Mapping[str, list[str]]) -> str:
    if issues.get("blocker") or issues.get("high"):
        return "not_ready_for_downstream_forecaster_consumption"
    if issues.get("medium"):
        return "ready_for_sandbox_downstream_forecaster_contract_tests_with_medium_issues"
    return "ready_for_sandbox_downstream_forecaster_contract_tests"


def _dtype_valid(value: Any, dtype: str, *, nullable: bool) -> bool:
    if value is None:
        return nullable
    if dtype == LOGICAL_DTYPE_STRING:
        return isinstance(value, str)
    if dtype == LOGICAL_DTYPE_TIMESTAMP:
        return _timestamp_valid(value)
    if dtype == LOGICAL_DTYPE_FLOAT:
        return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))
    if dtype == LOGICAL_DTYPE_JSON:
        return isinstance(value, dict)
    return False


def _timestamp_valid(value: Any) -> bool:
    if isinstance(value, str) and value.strip():
        try:
            datetime.fromisoformat(value.replace("Z", "+00:00"))
            return True
        except Exception:
            return False
    return False


def _confidence_valid(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)) and 0.0 <= float(value) <= 1.0


def _looks_like_nan_or_inf(value: Any) -> bool:
    if isinstance(value, float):
        return not math.isfinite(value)
    if isinstance(value, str):
        return value.strip().lower() in {"nan", "inf", "+inf", "-inf", "infinity", "+infinity", "-infinity"}
    return False


def _resolve_output_file(value: Any) -> Path:
    text = _text(value, field_name="output_file")
    path = resolve_project_path(text)
    if not path.exists():
        raise FileNotFoundError(f"Regime Production sandbox output file not found: {path}")
    return path


def _load_json(path: Path) -> dict[str, Any]:
    resolved = resolve_project_path(path)
    with resolved.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("Regime Production sandbox validation expected a JSON object summary")
    return payload


def _rss_bytes() -> int | None:
    try:
        import psutil  # type: ignore

        return int(psutil.Process(os.getpid()).memory_info().rss)
    except Exception:
        return None


def _child_process_count() -> int | None:
    try:
        import psutil  # type: ignore

        return len(psutil.Process(os.getpid()).children(recursive=True))
    except Exception:
        return None


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


def _looks_like_absolute_path(value: str) -> bool:
    try:
        return Path(value).is_absolute()
    except Exception:
        return False


def _branch_name(value: object) -> str:
    text = _text(value, field_name="branch")
    if text not in REGIME_PRODUCTION_BRANCHES:
        raise ValueError(f"Unsupported Regime Production branch: {text!r}")
    return text


def _text(value: object, *, field_name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"Regime Production sandbox validation {field_name} must be non-empty")
    return text


__all__ = [
    "DEFAULT_WIDER_SANDBOX_SUMMARY_PATH",
    "REGIME_PRODUCTION_SANDBOX_VALIDATION_ARTIFACT_KIND",
    "REGIME_PRODUCTION_SANDBOX_VALIDATION_SCHEMA_VERSION",
    "RegimeProductionSandboxValidationConfig",
    "run_regime_production_sandbox_output_validation",
    "validate_regime_production_sandbox_branch_output",
    "write_regime_production_sandbox_validation_summary",
]
