from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.regimes.core.production_consumer import (
    REGIME_BRANCH_ASSET_STATE,
    REGIME_BRANCH_CROSS_ASSET_STATE,
    REGIME_BRANCH_MARKET_STATE,
    REGIME_PRODUCTION_BRANCHES,
)
from src.regimes.core.production_output_contracts import (
    AVAILABILITY_FIELD,
    BRANCH_LABEL_GRAIN_FIELDS,
    LOGICAL_DTYPE_FLOAT,
    LOGICAL_DTYPE_JSON,
    LOGICAL_DTYPE_STRING,
    LOGICAL_DTYPE_TIMESTAMP,
    STATE_IDENTITY_FIELD,
    default_regime_production_label_output_schema,
)
from src.regimes.core.serialization import to_jsonable


REGIME_PRODUCTION_DOWNSTREAM_FORECASTER_SCHEMA_VERSION = 1
REGIME_PRODUCTION_DOWNSTREAM_FORECASTER_CHECK_ARTIFACT_KIND = (
    "regime_production_downstream_forecaster_integration_dry_check"
)

DOWNSTREAM_READINESS_PASS = "PASS"
DOWNSTREAM_READINESS_BLOCKED = "BLOCKED"

DEFAULT_MODEL_FEATURE_COLUMNS: tuple[str, ...] = (
    STATE_IDENTITY_FIELD,
    "confidence",
    AVAILABILITY_FIELD,
    "mask_reason",
)
MASK_OR_UNAVAILABLE_STATUSES: tuple[str, ...] = (
    "masked_unavailable",
    "skipped_filtered",
    "diagnostic_only",
    "missing_input",
    "invalid_profile",
)
SELECTED_STATUS = "selected"


@dataclass(frozen=True)
class RegimeProductionDownstreamForecasterDryCheck:
    branch: str
    row_count: int
    selected_row_count: int
    mask_or_unavailable_row_count: int
    required_columns: Sequence[str]
    join_key_columns: Sequence[str]
    model_feature_columns: Sequence[str]
    protected_metadata_columns: Sequence[str]
    timestamp_alignment_by_band: Mapping[str, Sequence[Any]]
    issues: Mapping[str, Sequence[str]]
    checks: Mapping[str, Any]
    sample_rows: Sequence[Mapping[str, Any]] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        object.__setattr__(self, "branch", _branch_name(self.branch))
        object.__setattr__(self, "required_columns", tuple(str(item) for item in self.required_columns))
        object.__setattr__(self, "join_key_columns", tuple(str(item) for item in self.join_key_columns))
        object.__setattr__(self, "model_feature_columns", tuple(str(item) for item in self.model_feature_columns))
        object.__setattr__(self, "protected_metadata_columns", tuple(str(item) for item in self.protected_metadata_columns))
        object.__setattr__(self, "timestamp_alignment_by_band", to_jsonable(dict(self.timestamp_alignment_by_band)))
        object.__setattr__(self, "issues", {key: tuple(value) for key, value in dict(self.issues).items()})
        object.__setattr__(self, "checks", to_jsonable(dict(self.checks)))
        object.__setattr__(self, "sample_rows", tuple(to_jsonable(dict(row)) for row in self.sample_rows))

    @property
    def downstream_readiness_verdict(self) -> str:
        return DOWNSTREAM_READINESS_BLOCKED if self.issues.get("blocker") or self.issues.get("high") else DOWNSTREAM_READINESS_PASS

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": REGIME_PRODUCTION_DOWNSTREAM_FORECASTER_SCHEMA_VERSION,
            "artifact_kind": REGIME_PRODUCTION_DOWNSTREAM_FORECASTER_CHECK_ARTIFACT_KIND,
            "branch": self.branch,
            "downstream_readiness_verdict": self.downstream_readiness_verdict,
            "row_count": int(self.row_count),
            "selected_row_count": int(self.selected_row_count),
            "mask_or_unavailable_row_count": int(self.mask_or_unavailable_row_count),
            "required_columns": list(self.required_columns),
            "join_key_columns": list(self.join_key_columns),
            "model_feature_columns": list(self.model_feature_columns),
            "protected_metadata_columns": list(self.protected_metadata_columns),
            "timestamp_alignment_by_band": to_jsonable(dict(self.timestamp_alignment_by_band)),
            "checks": to_jsonable(dict(self.checks)),
            "issues": {key: list(value) for key, value in self.issues.items()},
            "sample_rows": [to_jsonable(dict(row)) for row in self.sample_rows],
            "lineage_fields_model_facing": False,
            "model_training_invoked": False,
            "test_branch_rerun_performed": False,
            "canonical_write_allowed": False,
            "canonical_root_touched": False,
            "production_writer_enabled": False,
            "production_labels_written": False,
            "production_outputs_written": False,
            "canonical_production_state_outputs_written": False,
        }


def load_regime_production_sandbox_jsonl_as_forecaster_rows(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            loaded = json.loads(text)
            if not isinstance(loaded, Mapping):
                raise ValueError("Regime Production downstream forecaster input rows must be JSON objects")
            rows.append(dict(loaded))
    return rows


def validate_regime_production_downstream_forecaster_input(
    branch: str,
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_feature_timestamps_by_band: Mapping[str, Sequence[Any]] | None = None,
    expected_grain_keys: Sequence[Mapping[str, Any]] = (),
    candidate_model_feature_columns: Sequence[str] | None = None,
    explicitly_model_facing_metadata_columns: Sequence[str] = (),
    require_masked_or_unavailable_rows: bool = True,
) -> RegimeProductionDownstreamForecasterDryCheck:
    branch_name = _branch_name(branch)
    schema = default_regime_production_label_output_schema(branch_name)
    required_columns = tuple(schema.column_order)
    rows_payload = [dict(row) for row in rows]
    issues: dict[str, list[str]] = {"blocker": [], "high": [], "medium": [], "low": []}
    if not rows_payload:
        issues["blocker"].append("downstream_input_rows_missing")

    model_features = tuple(candidate_model_feature_columns or DEFAULT_MODEL_FEATURE_COLUMNS)
    protected_metadata = _protected_metadata_columns(schema.as_dict())
    explicit_metadata = {str(item) for item in explicitly_model_facing_metadata_columns}
    accidental_metadata_features = sorted(set(model_features).intersection(protected_metadata).difference(explicit_metadata))
    if accidental_metadata_features:
        issues["high"].append("protected_metadata_columns_selected_as_model_features")

    column_sets = [tuple(row.keys()) for row in rows_payload]
    missing_required: dict[str, int] = {}
    unexpected_columns: set[str] = set()
    column_order_mismatch_count = 0
    dtype_errors: dict[str, int] = {}
    nan_inf_count = 0
    selected_count = 0
    masked_count = 0
    selected_label_invalid = 0
    mask_row_invalid = 0
    confidence_invalid = 0
    lineage_missing = 0
    duplicate_join_key_count = 0
    join_keys_seen: set[tuple[Any, ...]] = set()
    grain_timestamp_keys: set[tuple[Any, ...]] = set()
    timestamp_sets_by_band: dict[str, set[int]] = {}
    sample_rows: list[dict[str, Any]] = []

    for row in rows_payload:
        if len(sample_rows) < 3:
            sample_rows.append(dict(row))
        if tuple(row.keys()) != required_columns:
            column_order_mismatch_count += 1
        row_columns = set(row)
        unexpected_columns.update(row_columns.difference(required_columns))
        for column in required_columns:
            if column not in row:
                missing_required[column] = int(missing_required.get(column, 0)) + 1
                continue
            value = row.get(column)
            if not _dtype_valid(value, schema.dtype_by_column[column], nullable=column in schema.nullable_columns):
                dtype_errors[column] = int(dtype_errors.get(column, 0)) + 1
            if _nan_or_inf(value):
                nan_inf_count += 1

        join_key = tuple(row.get(field_name) for field_name in BRANCH_LABEL_GRAIN_FIELDS[branch_name])
        if join_key in join_keys_seen:
            duplicate_join_key_count += 1
        join_keys_seen.add(join_key)
        grain_timestamp_keys.add(join_key)
        band = str(row.get("band") or "")
        timestamp = _timestamp_or_none(row.get("timestamp"))
        if timestamp is not None:
            timestamp_sets_by_band.setdefault(band, set()).add(timestamp)

        status = str(row.get(AVAILABILITY_FIELD) or "")
        if status == SELECTED_STATUS:
            selected_count += 1
            if not row.get(STATE_IDENTITY_FIELD) or row.get("mask_reason") not in (None, ""):
                selected_label_invalid += 1
            if not _confidence_valid(row.get("confidence")):
                confidence_invalid += 1
        elif status in MASK_OR_UNAVAILABLE_STATUSES:
            masked_count += 1
            if row.get(STATE_IDENTITY_FIELD) is not None or row.get("confidence") is not None or row.get("mask_reason") in (None, ""):
                mask_row_invalid += 1
        else:
            issues["high"].append("unknown_availability_status")

        if not _lineage_present(branch_name, row):
            lineage_missing += 1

    if missing_required or unexpected_columns or column_order_mismatch_count:
        issues["high"].append("required_columns_or_fixed_schema_invalid")
    if dtype_errors:
        issues["high"].append("column_dtype_validation_failed")
    if duplicate_join_key_count:
        issues["high"].append("duplicate_join_keys")
    if nan_inf_count:
        issues["high"].append("nan_or_inf_unexpected_values")
    if selected_label_invalid or confidence_invalid or mask_row_invalid:
        issues["high"].append("label_confidence_status_fields_not_forecaster_usable")
    if require_masked_or_unavailable_rows and masked_count == 0:
        issues["high"].append("masked_or_unavailable_rows_not_represented")
    if lineage_missing:
        issues["high"].append("required_lineage_fields_missing")

    timestamp_alignment = _timestamp_alignment(timestamp_sets_by_band, expected_feature_timestamps_by_band or {})
    if expected_feature_timestamps_by_band and not timestamp_alignment["passed"]:
        issues["high"].append("timestamp_alignment_with_feature_outputs_failed")
    expected_coverage = _expected_grain_coverage(
        branch_name,
        rows_payload,
        expected_grain_keys=expected_grain_keys,
        expected_feature_timestamps_by_band=expected_feature_timestamps_by_band or {},
    )
    if not expected_coverage["passed"]:
        issues["high"].append("expected_assets_or_cells_disappeared")

    checks = {
        "schema_checks": {
            "fixed_columns_passed": not missing_required and not unexpected_columns and column_order_mismatch_count == 0,
            "required_columns": list(required_columns),
            "missing_required_counts": missing_required,
            "unexpected_columns": sorted(unexpected_columns),
            "column_order_mismatch_count": column_order_mismatch_count,
            "dtype_error_counts": dtype_errors,
        },
        "join_key_checks": {
            "join_key_columns": list(BRANCH_LABEL_GRAIN_FIELDS[branch_name]),
            "duplicate_join_key_count": duplicate_join_key_count,
            "join_keys_usable": duplicate_join_key_count == 0,
        },
        "timestamp_alignment_checks": timestamp_alignment,
        "shape_checks": {
            "stable_row_feature_shape": not duplicate_join_key_count and expected_coverage["passed"],
            "expected_grain_coverage": expected_coverage,
            "no_disappearing_assets_or_cells": expected_coverage["passed"],
        },
        "mask_checks": {
            "masked_or_unavailable_rows_represented": masked_count > 0,
            "mask_or_unavailable_row_count": masked_count,
            "mask_row_invalid_count": mask_row_invalid,
        },
        "value_checks": {
            "nan_or_inf_unexpected_value_count": nan_inf_count,
            "selected_label_invalid_count": selected_label_invalid,
            "confidence_invalid_count": confidence_invalid,
            "label_confidence_status_fields_usable_as_features": selected_label_invalid == 0
            and confidence_invalid == 0
            and mask_row_invalid == 0,
        },
        "feature_column_checks": {
            "candidate_model_feature_columns": list(model_features),
            "protected_metadata_columns": sorted(protected_metadata),
            "accidental_metadata_feature_columns": accidental_metadata_features,
            "lineage_fields_model_facing": False,
            "raw_metadata_used_as_features": bool(accidental_metadata_features),
        },
        "lineage_checks": {
            "required_lineage_fields_present": lineage_missing == 0,
            "lineage_missing_row_count": lineage_missing,
            "lineage_fields_not_model_facing": not accidental_metadata_features,
        },
    }
    return RegimeProductionDownstreamForecasterDryCheck(
        branch=branch_name,
        row_count=len(rows_payload),
        selected_row_count=selected_count,
        mask_or_unavailable_row_count=masked_count,
        required_columns=required_columns,
        join_key_columns=BRANCH_LABEL_GRAIN_FIELDS[branch_name],
        model_feature_columns=model_features,
        protected_metadata_columns=tuple(sorted(protected_metadata)),
        timestamp_alignment_by_band={band: sorted(values) for band, values in timestamp_sets_by_band.items()},
        issues=issues,
        checks=checks,
        sample_rows=tuple(sample_rows),
    )


def _protected_metadata_columns(schema_payload: Mapping[str, Any]) -> set[str]:
    protected: set[str] = set()
    for field in schema_payload.get("fields") or ():
        if not isinstance(field, Mapping):
            continue
        role = str(field.get("role") or "")
        name = str(field.get("name") or "")
        if role in {"profile", "definition", "lineage", "lineage_time", "relationship_lineage_time"}:
            protected.add(name)
    return protected


def _timestamp_alignment(actual_by_band: Mapping[str, set[int]], expected_by_band: Mapping[str, Sequence[Any]]) -> dict[str, Any]:
    expected = {
        str(band): sorted(_timestamp(value, field_name=f"expected_timestamp[{band}]") for value in values)
        for band, values in expected_by_band.items()
    }
    actual = {str(band): sorted(values) for band, values in actual_by_band.items()}
    missing_by_band: dict[str, list[int]] = {}
    extra_by_band: dict[str, list[int]] = {}
    for band, expected_values in expected.items():
        actual_values = set(actual.get(band, ()))
        expected_set = set(expected_values)
        missing_by_band[band] = sorted(expected_set.difference(actual_values))
        extra_by_band[band] = sorted(actual_values.difference(expected_set))
    return {
        "passed": not expected or (all(not values for values in missing_by_band.values()) and all(not values for values in extra_by_band.values())),
        "actual_timestamps_by_band": actual,
        "expected_timestamps_by_band": expected,
        "missing_timestamps_by_band": missing_by_band,
        "extra_timestamps_by_band": extra_by_band,
    }


def _expected_grain_coverage(
    branch: str,
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_grain_keys: Sequence[Mapping[str, Any]],
    expected_feature_timestamps_by_band: Mapping[str, Sequence[Any]],
) -> dict[str, Any]:
    if not expected_grain_keys or not expected_feature_timestamps_by_band:
        return {"passed": True, "missing_grain_timestamp_count": 0, "missing_grain_timestamps": []}
    grain_fields = tuple(field for field in BRANCH_LABEL_GRAIN_FIELDS[branch] if field != "timestamp")
    available = {
        tuple(row.get(field) for field in (*grain_fields, "timestamp"))
        for row in rows
    }
    missing: list[dict[str, Any]] = []
    for grain in expected_grain_keys:
        band = str(grain.get("band") or "")
        for timestamp in expected_feature_timestamps_by_band.get(band, ()):
            expected_key = tuple(grain.get(field) for field in grain_fields) + (_timestamp(timestamp, field_name="expected_timestamp"),)
            if expected_key not in available:
                missing.append({"grain_key": to_jsonable(dict(grain)), "timestamp": timestamp})
    return {
        "passed": not missing,
        "missing_grain_timestamp_count": len(missing),
        "missing_grain_timestamps": missing[:20],
    }


def _lineage_present(branch: str, row: Mapping[str, Any]) -> bool:
    lineage = row.get("lineage")
    if not isinstance(lineage, Mapping):
        return False
    if not row.get("profile_id") or not row.get("profile_version") or not row.get("definition_id") or not row.get("definition_version"):
        return False
    if not row.get("run_id"):
        return False
    if branch == REGIME_BRANCH_CROSS_ASSET_STATE:
        return row.get("relationship_input_tail_ts") not in (None, "") and row.get("relationship_known_at_ts") not in (None, "")
    return row.get("source_tail_ts") not in (None, "") and row.get("known_at_ts") not in (None, "")


def _dtype_valid(value: Any, logical_dtype: str, *, nullable: bool) -> bool:
    if value in (None, ""):
        return bool(nullable)
    if logical_dtype == LOGICAL_DTYPE_STRING:
        return isinstance(value, str)
    if logical_dtype == LOGICAL_DTYPE_FLOAT:
        return _confidence_number(value, allow_none=False) is not None
    if logical_dtype == LOGICAL_DTYPE_TIMESTAMP:
        return _timestamp_or_none(value) is not None
    if logical_dtype == LOGICAL_DTYPE_JSON:
        return isinstance(value, Mapping)
    return False


def _confidence_valid(value: Any) -> bool:
    parsed = _confidence_number(value, allow_none=False)
    return parsed is not None and 0.0 <= parsed <= 1.0


def _confidence_number(value: Any, *, allow_none: bool) -> float | None:
    if value is None:
        return None if allow_none else None
    if isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except Exception:
        return None
    if math.isnan(parsed) or math.isinf(parsed):
        return None
    return parsed


def _nan_or_inf(value: Any) -> bool:
    if isinstance(value, float):
        return math.isnan(value) or math.isinf(value)
    if isinstance(value, str) and value.strip().lower() in {"nan", "inf", "+inf", "-inf", "infinity", "+infinity", "-infinity"}:
        return True
    return False


def _timestamp_or_none(value: Any) -> int | None:
    try:
        return _timestamp(value, field_name="timestamp")
    except ValueError:
        return None


def _timestamp(value: Any, *, field_name: str) -> int:
    if value in (None, "") or isinstance(value, bool):
        raise ValueError(f"Regime Production downstream forecaster {field_name} must be a timestamp")
    try:
        return int(float(value))
    except Exception:
        pass
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except Exception as exc:
        raise ValueError(f"Regime Production downstream forecaster {field_name} must be numeric or ISO datetime") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return int(parsed.timestamp())


def _branch_name(value: object) -> str:
    text = str(value or "").strip().lower().replace("-", "_")
    aliases = {
        "asset": REGIME_BRANCH_ASSET_STATE,
        "asset_state_production": REGIME_BRANCH_ASSET_STATE,
        "market": REGIME_BRANCH_MARKET_STATE,
        "market_state_production": REGIME_BRANCH_MARKET_STATE,
        "cross_asset": REGIME_BRANCH_CROSS_ASSET_STATE,
        "cross_asset_state_production": REGIME_BRANCH_CROSS_ASSET_STATE,
    }
    branch = aliases.get(text, text)
    if branch not in REGIME_PRODUCTION_BRANCHES:
        raise ValueError(f"Unsupported Regime Production branch: {value!r}")
    return branch


__all__ = [
    "DEFAULT_MODEL_FEATURE_COLUMNS",
    "DOWNSTREAM_READINESS_BLOCKED",
    "DOWNSTREAM_READINESS_PASS",
    "REGIME_PRODUCTION_DOWNSTREAM_FORECASTER_CHECK_ARTIFACT_KIND",
    "REGIME_PRODUCTION_DOWNSTREAM_FORECASTER_SCHEMA_VERSION",
    "RegimeProductionDownstreamForecasterDryCheck",
    "load_regime_production_sandbox_jsonl_as_forecaster_rows",
    "validate_regime_production_downstream_forecaster_input",
]
