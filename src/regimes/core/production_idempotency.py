from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.regimes.core.paths import resolve_project_path
from src.regimes.core.production_consumer import REGIME_PRODUCTION_BRANCHES
from src.regimes.core.serialization import to_jsonable


REGIME_PRODUCTION_IDEMPOTENCY_SCHEMA_VERSION = 1
REGIME_PRODUCTION_IDEMPOTENCY_KEY_ARTIFACT_KIND = "regime_production_idempotency_key"
REGIME_PRODUCTION_RERUN_EVALUATION_ARTIFACT_KIND = "regime_production_rerun_evaluation"

RERUN_ACTION_SKIP_COMPLETED = "skip_completed"
RERUN_ACTION_RECOMPUTE_REQUIRED = "recompute_required"
RERUN_ACTION_REJECT_PARTIAL = "reject_partial"
RERUN_ACTION_REJECT_SCHEMA_MISMATCH = "reject_schema_mismatch"
RERUN_ACTION_REJECT_MISSING_OUTPUTS = "reject_missing_outputs"

OUTPUT_COMPLETION_STATUS_COMPLETED = "completed"
OUTPUT_COMPLETION_STATUS_PARTIAL = "partial"


class RegimeProductionIdempotencyError(RuntimeError):
    """Raised when an existing Regime Production output cannot be safely reused."""


@dataclass(frozen=True)
class RegimeProductionIdempotencyKey:
    branch: str
    mode: str
    run_id: str
    output_root: str | Path
    clamp_range: Mapping[str, Any]
    output_schema_version: int
    output_schema_hash: str
    source_artifact_path: str | Path
    source_artifact_hash: str
    selected_profile_artifact_hash: str | None
    approval_artifact_hash: str
    source_tail_fingerprint: str
    selected_partitions: Sequence[str]
    root_kind: str = "sandbox_output_root"
    writer_scope: str = "label_output"

    def __post_init__(self) -> None:
        branch = _text(self.branch, field_name="branch")
        if branch not in REGIME_PRODUCTION_BRANCHES:
            raise ValueError(f"Unsupported Regime Production branch for idempotency: {branch!r}")
        partitions = tuple(sorted(_text(partition, field_name="selected_partition") for partition in self.selected_partitions))
        if not partitions:
            raise ValueError("Regime Production idempotency key requires selected_partitions")
        object.__setattr__(self, "branch", branch)
        object.__setattr__(self, "mode", _text(self.mode, field_name="mode"))
        object.__setattr__(self, "run_id", _text(self.run_id, field_name="run_id"))
        object.__setattr__(self, "clamp_range", to_jsonable(dict(self.clamp_range)))
        object.__setattr__(self, "output_schema_version", int(self.output_schema_version))
        object.__setattr__(self, "output_schema_hash", _text(self.output_schema_hash, field_name="output_schema_hash"))
        object.__setattr__(self, "source_artifact_hash", _text(self.source_artifact_hash, field_name="source_artifact_hash"))
        object.__setattr__(
            self,
            "selected_profile_artifact_hash",
            None if self.selected_profile_artifact_hash in (None, "") else str(self.selected_profile_artifact_hash),
        )
        object.__setattr__(self, "approval_artifact_hash", _text(self.approval_artifact_hash, field_name="approval_artifact_hash"))
        object.__setattr__(self, "source_tail_fingerprint", _text(self.source_tail_fingerprint, field_name="source_tail_fingerprint"))
        object.__setattr__(self, "selected_partitions", partitions)
        object.__setattr__(self, "root_kind", _text(self.root_kind, field_name="root_kind"))
        object.__setattr__(self, "writer_scope", _text(self.writer_scope, field_name="writer_scope"))

    @property
    def idempotency_key_hash(self) -> str:
        return stable_payload_hash(self._hash_payload())

    def as_dict(self) -> dict[str, Any]:
        payload = self._hash_payload()
        return {
            **payload,
            "run_id": self.run_id,
            "run_instance_id": self.run_id,
            "workload_key_excludes_run_id": True,
            "idempotency_key_hash": self.idempotency_key_hash,
            "production_writer_enabled": False,
            "canonical_write_execution_allowed": False,
            "production_outputs_written": False,
            "production_labels_written": False,
            "canonical_production_state_outputs_written": False,
        }

    def _hash_payload(self) -> dict[str, Any]:
        resolved_output_root = resolve_project_path(self.output_root)
        resolved_source_path = resolve_project_path(self.source_artifact_path)
        return {
            "schema_version": REGIME_PRODUCTION_IDEMPOTENCY_SCHEMA_VERSION,
            "artifact_kind": REGIME_PRODUCTION_IDEMPOTENCY_KEY_ARTIFACT_KIND,
            "branch": self.branch,
            "mode": self.mode,
            "root_kind": self.root_kind,
            "writer_scope": self.writer_scope,
            "output_root": _portable_path_text(resolved_output_root),
            "output_root_hash": stable_payload_hash({"output_root": str(resolved_output_root)}),
            "clamp_range": to_jsonable(dict(self.clamp_range)),
            "output_schema_version": int(self.output_schema_version),
            "output_schema_hash": self.output_schema_hash,
            "source_artifact_path": _portable_path_text(resolved_source_path),
            "source_artifact_path_hash": stable_payload_hash({"source_artifact_path": str(resolved_source_path)}),
            "source_artifact_hash": self.source_artifact_hash,
            "selected_profile_artifact_hash": self.selected_profile_artifact_hash,
            "approval_artifact_hash": self.approval_artifact_hash,
            "source_tail_fingerprint": self.source_tail_fingerprint,
            "selected_partitions": list(self.selected_partitions),
        }


def stable_payload_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(to_jsonable(dict(payload)), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def evaluate_regime_production_existing_output_manifest(
    manifest: Mapping[str, Any],
    *,
    expected_idempotency_key: RegimeProductionIdempotencyKey | Mapping[str, Any],
    expected_schema_hash: str,
    expected_partition_paths: Sequence[str | Path] = (),
) -> dict[str, Any]:
    payload = to_jsonable(dict(manifest))
    expected = (
        expected_idempotency_key.as_dict()
        if isinstance(expected_idempotency_key, RegimeProductionIdempotencyKey)
        else to_jsonable(dict(expected_idempotency_key))
    )
    reason_codes: list[str] = []
    action = RERUN_ACTION_SKIP_COMPLETED

    if bool(payload.get("partial_output_marker")):
        reason_codes.append("partial_output_marker_present")
    if payload.get("completion_status") != OUTPUT_COMPLETION_STATUS_COMPLETED:
        reason_codes.append("completion_status_not_completed")
    if payload.get("manifest_status") != OUTPUT_COMPLETION_STATUS_COMPLETED:
        reason_codes.append("manifest_status_not_completed")
    if payload.get("writer_finalization_status") != OUTPUT_COMPLETION_STATUS_COMPLETED:
        reason_codes.append("writer_finalization_status_not_completed")
    if reason_codes:
        action = RERUN_ACTION_REJECT_PARTIAL

    schema = dict(payload.get("output_schema") or {})
    if schema.get("schema_hash") != expected_schema_hash:
        reason_codes.append("output_schema_hash_mismatch")
        action = RERUN_ACTION_REJECT_SCHEMA_MISMATCH

    existing_key = dict(payload.get("idempotency") or {})
    existing_hash = existing_key.get("idempotency_key_hash") or payload.get("write_fingerprint")
    expected_hash = expected.get("idempotency_key_hash")
    if existing_hash != expected_hash:
        reason_codes.append("idempotency_key_hash_mismatch")
        if action == RERUN_ACTION_SKIP_COMPLETED:
            action = RERUN_ACTION_RECOMPUTE_REQUIRED

    missing_paths = [
        _portable_path_text(Path(path))
        for path in expected_partition_paths
        if not Path(path).exists()
    ]
    if missing_paths:
        reason_codes.append("declared_partition_file_missing")
        action = RERUN_ACTION_REJECT_MISSING_OUTPUTS

    return to_jsonable(
        {
            "schema_version": REGIME_PRODUCTION_IDEMPOTENCY_SCHEMA_VERSION,
            "artifact_kind": REGIME_PRODUCTION_RERUN_EVALUATION_ARTIFACT_KIND,
            "rerun_action": action,
            "can_skip_existing_completed_output": action == RERUN_ACTION_SKIP_COMPLETED,
            "requires_recompute": action == RERUN_ACTION_RECOMPUTE_REQUIRED,
            "reject_existing_output": action
            in {
                RERUN_ACTION_REJECT_PARTIAL,
                RERUN_ACTION_REJECT_SCHEMA_MISMATCH,
                RERUN_ACTION_REJECT_MISSING_OUTPUTS,
            },
            "reason_codes": list(dict.fromkeys(reason_codes)),
            "existing_idempotency_key_hash": existing_hash,
            "expected_idempotency_key_hash": expected_hash,
            "missing_partition_paths": missing_paths,
            "production_writer_enabled": False,
            "canonical_write_execution_allowed": False,
            "production_outputs_written": False,
            "production_labels_written": False,
            "canonical_production_state_outputs_written": False,
        }
    )


def assert_regime_production_completed_output_reusable(
    manifest: Mapping[str, Any],
    *,
    expected_idempotency_key: RegimeProductionIdempotencyKey | Mapping[str, Any],
    expected_schema_hash: str,
    expected_partition_paths: Sequence[str | Path] = (),
) -> dict[str, Any]:
    evaluation = evaluate_regime_production_existing_output_manifest(
        manifest,
        expected_idempotency_key=expected_idempotency_key,
        expected_schema_hash=expected_schema_hash,
        expected_partition_paths=expected_partition_paths,
    )
    if evaluation["rerun_action"] != RERUN_ACTION_SKIP_COMPLETED:
        raise RegimeProductionIdempotencyError(
            "Regime Production existing output is not reusable for rerun: "
            + ",".join(evaluation["reason_codes"])
        )
    return evaluation


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


def _text(value: object, *, field_name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"Regime Production idempotency {field_name} is required")
    return text


__all__ = [
    "OUTPUT_COMPLETION_STATUS_COMPLETED",
    "OUTPUT_COMPLETION_STATUS_PARTIAL",
    "REGIME_PRODUCTION_IDEMPOTENCY_KEY_ARTIFACT_KIND",
    "REGIME_PRODUCTION_IDEMPOTENCY_SCHEMA_VERSION",
    "REGIME_PRODUCTION_RERUN_EVALUATION_ARTIFACT_KIND",
    "RERUN_ACTION_RECOMPUTE_REQUIRED",
    "RERUN_ACTION_REJECT_MISSING_OUTPUTS",
    "RERUN_ACTION_REJECT_PARTIAL",
    "RERUN_ACTION_REJECT_SCHEMA_MISMATCH",
    "RERUN_ACTION_SKIP_COMPLETED",
    "RegimeProductionIdempotencyError",
    "RegimeProductionIdempotencyKey",
    "assert_regime_production_completed_output_reusable",
    "evaluate_regime_production_existing_output_manifest",
    "stable_payload_hash",
]
