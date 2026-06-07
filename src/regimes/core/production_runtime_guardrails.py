from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from src.regimes.core.production_consumer import (
    REGIME_BRANCH_ASSET_STATE,
    REGIME_BRANCH_CROSS_ASSET_STATE,
    REGIME_BRANCH_MARKET_STATE,
    REGIME_PRODUCTION_BRANCHES,
)
from src.regimes.core.serialization import to_jsonable


REGIME_PRODUCTION_RUNTIME_GUARDRAILS_SCHEMA_VERSION = 1
REGIME_PRODUCTION_RUNTIME_GUARDRAILS_ARTIFACT_KIND = "regime_production_runtime_resource_guardrails"

REGIME_PRODUCTION_RUNTIME_STATUS_PASSED = "passed"
REGIME_PRODUCTION_RUNTIME_STATUS_PASSED_WITH_WARNINGS = "passed_with_warnings"
REGIME_PRODUCTION_RUNTIME_STATUS_BLOCKED = "blocked"

REGIME_PRODUCTION_RUNTIME_ACTION_CONTINUE = "continue"
REGIME_PRODUCTION_RUNTIME_ACTION_CONTINUE_WITH_WARNINGS = "continue_with_warnings"
REGIME_PRODUCTION_RUNTIME_ACTION_ABORT_GRACEFULLY = "abort_gracefully"

DEFAULT_BRANCH_RUNTIME_BUDGETS: Mapping[str, Mapping[str, int]] = {
    REGIME_BRANCH_MARKET_STATE: {
        "max_workers": 4,
        "expected_runtime_seconds": 300,
        "max_runtime_seconds": 900,
        "expected_rss_mb": 1024,
        "max_rss_mb": 2048,
        "timeout_grace_seconds": 60,
        "long_pole_seconds": 120,
    },
    REGIME_BRANCH_ASSET_STATE: {
        "max_workers": 16,
        "expected_runtime_seconds": 3600,
        "max_runtime_seconds": 7200,
        "expected_rss_mb": 4096,
        "max_rss_mb": 8192,
        "timeout_grace_seconds": 300,
        "long_pole_seconds": 900,
    },
    REGIME_BRANCH_CROSS_ASSET_STATE: {
        "max_workers": 16,
        "expected_runtime_seconds": 7200,
        "max_runtime_seconds": 14400,
        "expected_rss_mb": 8192,
        "max_rss_mb": 16384,
        "timeout_grace_seconds": 300,
        "long_pole_seconds": 1200,
    },
}


@dataclass(frozen=True)
class RegimeProductionRuntimeBudget:
    branch: str
    max_workers: int
    expected_runtime_seconds: int
    max_runtime_seconds: int
    expected_rss_mb: int
    max_rss_mb: int
    timeout_grace_seconds: int
    long_pole_seconds: int
    policy_source: str = "source_owned_regime_production_runtime_budget"

    def __post_init__(self) -> None:
        branch = _branch_name(self.branch)
        positives = {
            "max_workers": int(self.max_workers),
            "expected_runtime_seconds": int(self.expected_runtime_seconds),
            "max_runtime_seconds": int(self.max_runtime_seconds),
            "expected_rss_mb": int(self.expected_rss_mb),
            "max_rss_mb": int(self.max_rss_mb),
            "timeout_grace_seconds": int(self.timeout_grace_seconds),
            "long_pole_seconds": int(self.long_pole_seconds),
        }
        for field_name, value in positives.items():
            if value <= 0:
                raise ValueError(f"Regime Production runtime budget {field_name} must be positive")
        if positives["expected_runtime_seconds"] > positives["max_runtime_seconds"]:
            raise ValueError("Regime Production expected runtime budget cannot exceed max runtime budget")
        if positives["expected_rss_mb"] > positives["max_rss_mb"]:
            raise ValueError("Regime Production expected RSS budget cannot exceed max RSS budget")
        object.__setattr__(self, "branch", branch)
        for field_name, value in positives.items():
            object.__setattr__(self, field_name, value)
        object.__setattr__(self, "policy_source", _text(self.policy_source, field_name="policy_source"))

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": REGIME_PRODUCTION_RUNTIME_GUARDRAILS_SCHEMA_VERSION,
            "branch": self.branch,
            "max_workers": int(self.max_workers),
            "expected_runtime_seconds": int(self.expected_runtime_seconds),
            "max_runtime_seconds": int(self.max_runtime_seconds),
            "expected_rss_mb": int(self.expected_rss_mb),
            "max_rss_mb": int(self.max_rss_mb),
            "timeout_grace_seconds": int(self.timeout_grace_seconds),
            "long_pole_seconds": int(self.long_pole_seconds),
            "policy_source": self.policy_source,
            "configured_budget_required_before_canonical_write": True,
        }


@dataclass(frozen=True)
class RegimeProductionRuntimeTelemetry:
    elapsed_seconds: float = 0.0
    rss_start_bytes: int | None = None
    rss_peak_bytes: int | None = None
    rss_end_bytes: int | None = None
    child_process_count_start: int | None = None
    child_process_count_end: int | None = None
    subprocess_invocation_count: int = 0
    phase_elapsed_seconds: Mapping[str, float] = field(default_factory=dict)
    orphan_child_process_count: int = 0
    timeout_triggered: bool = False
    graceful_abort_completed: bool = False
    partial_output_marker_present: bool = False
    active_output_pointer_updated: bool = False
    canonical_root_touched: bool = False

    def __post_init__(self) -> None:
        elapsed = max(0.0, float(self.elapsed_seconds))
        phases = {
            _text(key, field_name="phase_elapsed_seconds.key"): max(0.0, float(value))
            for key, value in dict(self.phase_elapsed_seconds or {}).items()
        }
        object.__setattr__(self, "elapsed_seconds", elapsed)
        object.__setattr__(self, "phase_elapsed_seconds", phases)
        object.__setattr__(self, "subprocess_invocation_count", max(0, int(self.subprocess_invocation_count)))
        object.__setattr__(self, "orphan_child_process_count", max(0, int(self.orphan_child_process_count)))

    @property
    def rss_peak_mb(self) -> float | None:
        peak = self.rss_peak_bytes
        if peak is None:
            values = tuple(value for value in (self.rss_start_bytes, self.rss_end_bytes) if value is not None)
            peak = max(values) if values else None
        if peak is None:
            return None
        return float(peak) / (1024.0 * 1024.0)

    @property
    def child_process_count_delta(self) -> int | None:
        if self.child_process_count_start is None or self.child_process_count_end is None:
            return None
        return int(self.child_process_count_end) - int(self.child_process_count_start)

    def as_dict(self) -> dict[str, Any]:
        return {
            "elapsed_seconds": round(float(self.elapsed_seconds), 6),
            "rss_start_bytes": self.rss_start_bytes,
            "rss_peak_bytes": self.rss_peak_bytes,
            "rss_end_bytes": self.rss_end_bytes,
            "rss_peak_mb": None if self.rss_peak_mb is None else round(float(self.rss_peak_mb), 6),
            "child_process_count_start": self.child_process_count_start,
            "child_process_count_end": self.child_process_count_end,
            "child_process_count_delta": self.child_process_count_delta,
            "subprocess_invocation_count": int(self.subprocess_invocation_count),
            "phase_elapsed_seconds": {str(key): round(float(value), 6) for key, value in self.phase_elapsed_seconds.items()},
            "orphan_child_process_count": int(self.orphan_child_process_count),
            "timeout_triggered": bool(self.timeout_triggered),
            "graceful_abort_completed": bool(self.graceful_abort_completed),
            "partial_output_marker_present": bool(self.partial_output_marker_present),
            "active_output_pointer_updated": bool(self.active_output_pointer_updated),
            "canonical_root_touched": bool(self.canonical_root_touched),
        }


@dataclass(frozen=True)
class RegimeProductionRuntimeGuardrailResult:
    branch: str
    status: str
    action: str
    runtime_budget: RegimeProductionRuntimeBudget
    runtime_telemetry: RegimeProductionRuntimeTelemetry
    worker_profile: Mapping[str, Any]
    reason_codes: Sequence[str]
    warning_codes: Sequence[str]

    def as_dict(self) -> dict[str, Any]:
        blocked = self.status == REGIME_PRODUCTION_RUNTIME_STATUS_BLOCKED
        return to_jsonable(
            {
                "schema_version": REGIME_PRODUCTION_RUNTIME_GUARDRAILS_SCHEMA_VERSION,
                "artifact_kind": REGIME_PRODUCTION_RUNTIME_GUARDRAILS_ARTIFACT_KIND,
                "branch": self.branch,
                "status": self.status,
                "action": self.action,
                "runtime_budget": self.runtime_budget.as_dict(),
                "runtime_telemetry": self.runtime_telemetry.as_dict(),
                "worker_profile": dict(self.worker_profile),
                "reason_codes": list(self.reason_codes),
                "warning_codes": list(self.warning_codes),
                "timeout_policy": {
                    "max_runtime_seconds": int(self.runtime_budget.max_runtime_seconds),
                    "timeout_grace_seconds": int(self.runtime_budget.timeout_grace_seconds),
                    "timeout_triggered": bool(self.runtime_telemetry.timeout_triggered),
                    "graceful_abort_required": bool(
                        "timeout_budget_exceeded" in self.reason_codes
                        or "timeout_triggered" in self.reason_codes
                    ),
                },
                "graceful_abort_policy": {
                    "abort_action": REGIME_PRODUCTION_RUNTIME_ACTION_ABORT_GRACEFULLY,
                    "release_or_mark_run_lock_recoverable": True,
                    "worker_children_must_be_reaped": True,
                    "partial_outputs_must_remain_non_active": True,
                    "active_output_pointer_update_allowed_after_abort": False,
                    "graceful_abort_completed": bool(self.runtime_telemetry.graceful_abort_completed),
                },
                "long_pole_detection": {
                    "threshold_seconds": int(self.runtime_budget.long_pole_seconds),
                    "warning_codes": [
                        code for code in self.warning_codes if code.startswith("long_pole_phase_detected")
                    ],
                },
                "partial_output_non_activation": {
                    "partial_output_marker_present": bool(self.runtime_telemetry.partial_output_marker_present),
                    "active_output_pointer_updated": bool(self.runtime_telemetry.active_output_pointer_updated),
                    "passed": not bool(self.runtime_telemetry.active_output_pointer_updated),
                },
                "subprocess_orphan_detection": {
                    "orphan_child_process_count": int(self.runtime_telemetry.orphan_child_process_count),
                    "child_process_count_delta": self.runtime_telemetry.child_process_count_delta,
                    "passed": (
                        int(self.runtime_telemetry.orphan_child_process_count) == 0
                        and (self.runtime_telemetry.child_process_count_delta in (0, None))
                    ),
                },
                "canonical_write_allowed": False,
                "canonical_root_touched": bool(self.runtime_telemetry.canonical_root_touched),
                "production_writer_enabled": False,
                "production_outputs_written": False,
                "production_labels_written": False,
                "canonical_production_state_outputs_written": False,
                "production_promotion_performed": False,
                "test_branch_rerun_performed": False,
                "optuna_or_campaign_run_performed": False,
                "relationship_discovery_or_pairwise_run_performed": False,
                "production_writer_gates_fail_closed": True,
                "blocked": blocked,
            }
        )


def default_regime_production_runtime_budget(branch: str) -> RegimeProductionRuntimeBudget:
    branch_name = _branch_name(branch)
    payload = DEFAULT_BRANCH_RUNTIME_BUDGETS[branch_name]
    return RegimeProductionRuntimeBudget(branch=branch_name, **dict(payload))


def validate_regime_production_runtime_guardrails(
    branch: str,
    worker_profile: Mapping[str, Any] | Any,
    runtime_telemetry: RegimeProductionRuntimeTelemetry | Mapping[str, Any],
    *,
    runtime_budget: RegimeProductionRuntimeBudget | Mapping[str, Any] | None = None,
) -> RegimeProductionRuntimeGuardrailResult:
    branch_name = _branch_name(branch)
    profile = _coerce_worker_profile(worker_profile)
    telemetry = (
        runtime_telemetry
        if isinstance(runtime_telemetry, RegimeProductionRuntimeTelemetry)
        else RegimeProductionRuntimeTelemetry(**dict(runtime_telemetry or {}))
    )
    budget = _coerce_runtime_budget(branch_name, runtime_budget)
    reason_codes: list[str] = []
    warning_codes: list[str] = []

    if str(profile.get("branch") or branch_name) != branch_name:
        reason_codes.append("worker_profile_branch_mismatch")
    if int(profile.get("workers") or 0) <= 0:
        reason_codes.append("worker_profile_workers_missing")
    elif int(profile.get("workers") or 0) > int(budget.max_workers):
        reason_codes.append("worker_count_exceeds_branch_max")
    if int(profile.get("writer_workers") or 0) != 1:
        reason_codes.append("writer_workers_not_single_finalizer")
    if profile.get("parent_single_finalizer") is not True:
        reason_codes.append("parent_single_finalizer_missing")
    if bool(profile.get("workers_write_outputs")):
        reason_codes.append("workers_write_outputs_enabled")
    if bool(profile.get("relationship_discovery_allowed")) or bool(profile.get("broad_pairwise_allowed")):
        reason_codes.append("relationship_discovery_or_pairwise_allowed")

    if telemetry.elapsed_seconds > float(budget.max_runtime_seconds):
        reason_codes.append("timeout_budget_exceeded")
    elif telemetry.elapsed_seconds > float(budget.expected_runtime_seconds):
        warning_codes.append("expected_runtime_budget_exceeded")
    if telemetry.timeout_triggered:
        reason_codes.append("timeout_triggered")
    if telemetry.rss_peak_mb is not None:
        if telemetry.rss_peak_mb > float(budget.max_rss_mb):
            reason_codes.append("rss_budget_exceeded")
        elif telemetry.rss_peak_mb > float(budget.expected_rss_mb):
            warning_codes.append("expected_rss_budget_exceeded")
    for phase, elapsed_seconds in telemetry.phase_elapsed_seconds.items():
        if elapsed_seconds > float(budget.long_pole_seconds):
            warning_codes.append(f"long_pole_phase_detected:{phase}")
    child_delta = telemetry.child_process_count_delta
    if telemetry.orphan_child_process_count > 0 or (child_delta is not None and child_delta > 0):
        reason_codes.append("orphan_child_process_detected")
    if telemetry.partial_output_marker_present:
        warning_codes.append("partial_output_marker_present")
    if telemetry.active_output_pointer_updated:
        reason_codes.append("partial_output_was_activated")
    if telemetry.canonical_root_touched:
        reason_codes.append("canonical_root_touched")

    reason_codes = _stable_codes(reason_codes)
    warning_codes = _stable_codes(warning_codes)
    if reason_codes:
        status = REGIME_PRODUCTION_RUNTIME_STATUS_BLOCKED
        action = REGIME_PRODUCTION_RUNTIME_ACTION_ABORT_GRACEFULLY
    elif warning_codes:
        status = REGIME_PRODUCTION_RUNTIME_STATUS_PASSED_WITH_WARNINGS
        action = REGIME_PRODUCTION_RUNTIME_ACTION_CONTINUE_WITH_WARNINGS
    else:
        status = REGIME_PRODUCTION_RUNTIME_STATUS_PASSED
        action = REGIME_PRODUCTION_RUNTIME_ACTION_CONTINUE
    return RegimeProductionRuntimeGuardrailResult(
        branch=branch_name,
        status=status,
        action=action,
        runtime_budget=budget,
        runtime_telemetry=telemetry,
        worker_profile=profile,
        reason_codes=tuple(reason_codes),
        warning_codes=tuple(warning_codes),
    )


def validate_regime_production_job_matrix_runtime_guardrails(
    job_matrix: Mapping[str, Any] | Any,
    runtime_telemetry: RegimeProductionRuntimeTelemetry | Mapping[str, Any],
    *,
    runtime_budget: RegimeProductionRuntimeBudget | Mapping[str, Any] | None = None,
) -> RegimeProductionRuntimeGuardrailResult:
    matrix = job_matrix.as_dict(include_work_units=False) if hasattr(job_matrix, "as_dict") else dict(job_matrix or {})
    branch = _branch_name(matrix.get("branch"))
    profile = dict(matrix.get("worker_profile") or {})
    if "parent_single_finalizer" not in profile and isinstance(matrix.get("parent_finalizer"), Mapping):
        profile["parent_single_finalizer"] = dict(matrix["parent_finalizer"]).get("parent_single_finalizer")
    if "workers_write_outputs" not in profile:
        profile["workers_write_outputs"] = matrix.get("workers_write_outputs")
    return validate_regime_production_runtime_guardrails(
        branch,
        profile,
        runtime_telemetry,
        runtime_budget=runtime_budget,
    )


def _coerce_worker_profile(value: Mapping[str, Any] | Any) -> dict[str, Any]:
    if hasattr(value, "as_dict"):
        payload = value.as_dict()
    else:
        payload = dict(value or {})
    parent = dict(payload.get("parent_finalizer") or {})
    if "parent_single_finalizer" not in payload and parent:
        payload["parent_single_finalizer"] = parent.get("parent_single_finalizer")
    if "writer_workers" not in payload and parent:
        payload["writer_workers"] = parent.get("writer_workers")
    return to_jsonable(payload)


def _coerce_runtime_budget(
    branch: str,
    value: RegimeProductionRuntimeBudget | Mapping[str, Any] | None,
) -> RegimeProductionRuntimeBudget:
    if value is None:
        return default_regime_production_runtime_budget(branch)
    if isinstance(value, RegimeProductionRuntimeBudget):
        if value.branch != branch:
            raise ValueError("Regime Production runtime budget branch mismatch")
        return value
    payload = dict(value)
    payload.setdefault("branch", branch)
    return RegimeProductionRuntimeBudget(**payload)


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
        raise ValueError(f"Unsupported Regime Production branch for runtime guardrails: {value!r}")
    return resolved


def _stable_codes(values: Sequence[str]) -> tuple[str, ...]:
    return tuple(sorted(dict.fromkeys(str(value).strip() for value in values if str(value).strip())))


def _text(value: object, *, field_name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"Regime Production runtime guardrails {field_name} must be non-empty")
    return text


__all__ = [
    "DEFAULT_BRANCH_RUNTIME_BUDGETS",
    "REGIME_PRODUCTION_RUNTIME_ACTION_ABORT_GRACEFULLY",
    "REGIME_PRODUCTION_RUNTIME_ACTION_CONTINUE",
    "REGIME_PRODUCTION_RUNTIME_ACTION_CONTINUE_WITH_WARNINGS",
    "REGIME_PRODUCTION_RUNTIME_GUARDRAILS_ARTIFACT_KIND",
    "REGIME_PRODUCTION_RUNTIME_GUARDRAILS_SCHEMA_VERSION",
    "REGIME_PRODUCTION_RUNTIME_STATUS_BLOCKED",
    "REGIME_PRODUCTION_RUNTIME_STATUS_PASSED",
    "REGIME_PRODUCTION_RUNTIME_STATUS_PASSED_WITH_WARNINGS",
    "RegimeProductionRuntimeBudget",
    "RegimeProductionRuntimeGuardrailResult",
    "RegimeProductionRuntimeTelemetry",
    "default_regime_production_runtime_budget",
    "validate_regime_production_job_matrix_runtime_guardrails",
    "validate_regime_production_runtime_guardrails",
]
