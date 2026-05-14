from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Sequence

from src.regimes.core.contracts import (
    CANONICAL_SCHEMA_VERSION,
    REGIME_CLASSIFICATION_VALUES,
    REGIME_RUN_STATUS_VALUES,
    RegimeClassification,
    RegimeLayer,
    RunStatus,
    StudyKey,
    coerce_study_key,
    require_json_mapping,
    require_non_empty_string,
    require_schema_version,
)
from src.regimes.core.serialization import dumps_json, loads_json, require_json_object, to_jsonable


REGIME_PROMOTION_GATE_SCHEMA_VERSION = CANONICAL_SCHEMA_VERSION

PROMOTION_STATUS_BLOCKED = "blocked"
PROMOTION_STATUS_WARNING = "warning"
PROMOTION_STATUS_ELIGIBLE_FOR_FURTHER_VALIDATION = "eligible_for_further_validation"
PROMOTION_STATUS_NOT_APPLICABLE = "not_applicable"
PROMOTION_GATE_STATUSES: tuple[str, ...] = (
    PROMOTION_STATUS_BLOCKED,
    PROMOTION_STATUS_WARNING,
    PROMOTION_STATUS_ELIGIBLE_FOR_FURTHER_VALIDATION,
    PROMOTION_STATUS_NOT_APPLICABLE,
)

PROMOTION_ACTION_PRODUCTION = "production_promotion"

FLAT_ASSET_POLICY_BLOCK = "block"
FLAT_ASSET_POLICY_ALLOW_SINGLE_STATE = "allow_single_state"
FLAT_ASSET_POLICIES: tuple[str, ...] = (
    FLAT_ASSET_POLICY_BLOCK,
    FLAT_ASSET_POLICY_ALLOW_SINGLE_STATE,
)

REQUIRED_SCOREBOARD_SECTIONS: tuple[str, ...] = (
    "internal_validity",
    "coverage_degeneracy",
    "runtime",
    "stability",
    "economic_separability",
)

NON_PRODUCTION_BLOCKING_CLASSIFICATIONS: tuple[str, ...] = (
    RegimeClassification.SANDBOX.value,
    RegimeClassification.SCAFFOLD.value,
    RegimeClassification.DIAGNOSTICS_ONLY.value,
    RegimeClassification.METADATA_ONLY.value,
)

BLOCKING_ARTIFACT_MARKERS: tuple[str, ...] = (
    "sandbox",
    "scaffold",
    "diagnostic",
    "diagnostics",
)

BLOCKING_METADATA_FLAGS: tuple[str, ...] = (
    "sandbox",
    "sandbox_enabled",
    "is_sandbox",
    "scaffold",
    "scaffold_only",
    "diagnostics_only",
    "diagnostic_only",
    "metadata_only",
)

BLOCKED_ALLOWED_NEXT_ACTIONS: tuple[str, ...] = (
    "inspect_blocking_reasons",
    "complete_missing_evidence",
    "rerun_outside_sandbox_or_scaffold",
)
WARNING_ALLOWED_NEXT_ACTIONS: tuple[str, ...] = (
    "review_warning_reasons",
    "run_additional_validation",
    "keep_out_of_production_until_reviewed",
)
ELIGIBLE_ALLOWED_NEXT_ACTIONS: tuple[str, ...] = (
    "run_external_validation",
    "prepare_human_promotion_review",
    "preserve_gate_result_as_evidence",
)
NOT_APPLICABLE_ALLOWED_NEXT_ACTIONS: tuple[str, ...] = (
    "choose_production_promotion_action",
)


def _text_or_none(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, Enum):
        value = value.value
    text = str(value).strip().lower()
    return text or None


def _normal_status(value: object) -> str:
    text = _text_or_none(value)
    if text not in PROMOTION_GATE_STATUSES:
        valid = ", ".join(PROMOTION_GATE_STATUSES)
        raise ValueError(f"Unsupported Regime promotion gate status {text!r}; expected one of: {valid}")
    return text


def _normal_flat_asset_policy(value: object) -> str:
    text = _text_or_none(value) or FLAT_ASSET_POLICY_BLOCK
    if text not in FLAT_ASSET_POLICIES:
        valid = ", ".join(FLAT_ASSET_POLICIES)
        raise ValueError(f"Unsupported Regime flat asset policy {text!r}; expected one of: {valid}")
    return text


def _normalize_string_tuple(values: Sequence[object] | None, *, field_name: str) -> tuple[str, ...]:
    if values is None:
        return ()
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValueError(f"Regime {field_name} must be a sequence of strings")
    return tuple(str(value).strip() for value in values if str(value).strip())


def _scoreboard_as_mapping(scoreboard: object | None) -> dict[str, Any] | None:
    if scoreboard is None:
        return None
    if hasattr(scoreboard, "as_dict"):
        return require_json_object(scoreboard.as_dict(), context="Regime promotion gate scoreboard")
    if isinstance(scoreboard, Mapping):
        return dict(scoreboard)
    raise ValueError("Regime promotion gate scoreboard must be a JSON object or expose as_dict")


def _scoreboard_sections(scoreboard: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if scoreboard is None:
        return None
    sections = scoreboard.get("sections")
    if isinstance(sections, Mapping):
        return dict(sections)
    if all(section in scoreboard for section in REQUIRED_SCOREBOARD_SECTIONS):
        return {section: scoreboard[section] for section in REQUIRED_SCOREBOARD_SECTIONS}
    return None


def _metadata_bool(metadata: Mapping[str, Any], key: str) -> bool:
    value = metadata.get(key)
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _metric_value(metrics: Mapping[str, Any], key: str) -> Any:
    value = metrics.get(key)
    if isinstance(value, Mapping) and "value" in value:
        return value.get("value")
    return value


def _bool_metric(metrics: Mapping[str, Any], key: str) -> bool:
    value = _metric_value(metrics, key)
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _int_metric(metrics: Mapping[str, Any], key: str) -> int | None:
    value = _metric_value(metrics, key)
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except Exception:
        return None


def _float_metric(metrics: Mapping[str, Any], key: str) -> float | None:
    value = _metric_value(metrics, key)
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except Exception:
        return None


def _path_has_sandbox_component(path: str) -> bool:
    normalized = path.replace("\\", "/").lower()
    return any(part == "sandbox" or part.endswith("_sandbox") for part in normalized.split("/"))


@dataclass(frozen=True)
class PromotionGateInput:
    gate_id: str
    study_key: StudyKey | Mapping[str, Any] | None = None
    scoreboard: Mapping[str, Any] | object | None = None
    artifact_kind: str | None = None
    artifact_classification: str | RegimeClassification | None = None
    artifact_metadata: Mapping[str, Any] = field(default_factory=dict)
    artifact_paths: Mapping[str, str] = field(default_factory=dict)
    run_status: str | RunStatus | None = None
    requested_action: str = PROMOTION_ACTION_PRODUCTION
    flat_asset_policy: str = FLAT_ASSET_POLICY_BLOCK
    schema_version: int = REGIME_PROMOTION_GATE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        schema_version = require_schema_version(self.schema_version)
        gate_id = require_non_empty_string(self.gate_id, field_name="promotion gate_id")
        study_key = None if self.study_key is None else coerce_study_key(self.study_key)
        metadata = require_json_mapping(self.artifact_metadata, field_name="promotion artifact_metadata")
        artifact_paths_raw = require_json_mapping(self.artifact_paths, field_name="promotion artifact_paths")
        artifact_paths = {
            require_non_empty_string(key, field_name="promotion artifact path key"): require_non_empty_string(
                value,
                field_name=f"promotion artifact path {key}",
            )
            for key, value in artifact_paths_raw.items()
        }
        requested_action = _text_or_none(self.requested_action) or PROMOTION_ACTION_PRODUCTION
        flat_asset_policy = _normal_flat_asset_policy(self.flat_asset_policy)
        object.__setattr__(self, "schema_version", schema_version)
        object.__setattr__(self, "gate_id", gate_id)
        object.__setattr__(self, "study_key", study_key)
        object.__setattr__(self, "scoreboard", _scoreboard_as_mapping(self.scoreboard))
        object.__setattr__(self, "artifact_kind", _text_or_none(self.artifact_kind))
        object.__setattr__(self, "artifact_classification", _text_or_none(self.artifact_classification))
        object.__setattr__(self, "artifact_metadata", to_jsonable(metadata))
        object.__setattr__(self, "artifact_paths", artifact_paths)
        object.__setattr__(self, "run_status", _text_or_none(self.run_status))
        object.__setattr__(self, "requested_action", requested_action)
        object.__setattr__(self, "flat_asset_policy", flat_asset_policy)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "gate_id": self.gate_id,
            "study_key": None if self.study_key is None else self.study_key.as_dict(),
            "scoreboard": to_jsonable(self.scoreboard),
            "artifact_kind": self.artifact_kind,
            "artifact_classification": self.artifact_classification,
            "artifact_metadata": to_jsonable(self.artifact_metadata),
            "artifact_paths": dict(self.artifact_paths),
            "run_status": self.run_status,
            "requested_action": self.requested_action,
            "flat_asset_policy": self.flat_asset_policy,
        }

    def to_json(self, **kwargs: Any) -> str:
        return dumps_json(self.as_dict(), **kwargs)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PromotionGateInput":
        obj = require_json_object(payload, context="Regime PromotionGateInput")
        return cls(
            schema_version=obj.get("schema_version", REGIME_PROMOTION_GATE_SCHEMA_VERSION),
            gate_id=obj["gate_id"],
            study_key=obj.get("study_key"),
            scoreboard=obj.get("scoreboard"),
            artifact_kind=obj.get("artifact_kind"),
            artifact_classification=obj.get("artifact_classification"),
            artifact_metadata=obj.get("artifact_metadata", {}),
            artifact_paths=obj.get("artifact_paths", {}),
            run_status=obj.get("run_status"),
            requested_action=obj.get("requested_action", PROMOTION_ACTION_PRODUCTION),
            flat_asset_policy=obj.get("flat_asset_policy", FLAT_ASSET_POLICY_BLOCK),
        )

    @classmethod
    def from_json(cls, text: str) -> "PromotionGateInput":
        return cls.from_dict(require_json_object(loads_json(text), context="Regime PromotionGateInput JSON"))


@dataclass(frozen=True)
class PromotionGateResult:
    gate_id: str
    status: str
    allowed_next_actions: Sequence[str]
    blocking_reasons: Sequence[str] = ()
    warning_reasons: Sequence[str] = ()
    checks: Mapping[str, Any] = field(default_factory=dict)
    input_summary: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = REGIME_PROMOTION_GATE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        schema_version = require_schema_version(self.schema_version)
        gate_id = require_non_empty_string(self.gate_id, field_name="promotion gate result gate_id")
        status = _normal_status(self.status)
        allowed_next_actions = _normalize_string_tuple(
            self.allowed_next_actions,
            field_name="promotion allowed_next_actions",
        )
        if not allowed_next_actions:
            raise ValueError("Regime promotion gate result requires allowed_next_actions")
        object.__setattr__(self, "schema_version", schema_version)
        object.__setattr__(self, "gate_id", gate_id)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "allowed_next_actions", allowed_next_actions)
        object.__setattr__(
            self,
            "blocking_reasons",
            _normalize_string_tuple(self.blocking_reasons, field_name="promotion blocking_reasons"),
        )
        object.__setattr__(
            self,
            "warning_reasons",
            _normalize_string_tuple(self.warning_reasons, field_name="promotion warning_reasons"),
        )
        object.__setattr__(self, "checks", to_jsonable(require_json_mapping(self.checks, field_name="promotion checks")))
        object.__setattr__(
            self,
            "input_summary",
            to_jsonable(require_json_mapping(self.input_summary, field_name="promotion input_summary")),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "gate_id": self.gate_id,
            "status": self.status,
            "allowed_next_actions": list(self.allowed_next_actions),
            "blocking_reasons": list(self.blocking_reasons),
            "warning_reasons": list(self.warning_reasons),
            "checks": to_jsonable(self.checks),
            "input_summary": to_jsonable(self.input_summary),
        }

    def to_json(self, **kwargs: Any) -> str:
        return dumps_json(self.as_dict(), **kwargs)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PromotionGateResult":
        obj = require_json_object(payload, context="Regime PromotionGateResult")
        return cls(
            schema_version=obj.get("schema_version", REGIME_PROMOTION_GATE_SCHEMA_VERSION),
            gate_id=obj["gate_id"],
            status=obj["status"],
            allowed_next_actions=obj["allowed_next_actions"],
            blocking_reasons=obj.get("blocking_reasons", ()),
            warning_reasons=obj.get("warning_reasons", ()),
            checks=obj.get("checks", {}),
            input_summary=obj.get("input_summary", {}),
        )

    @classmethod
    def from_json(cls, text: str) -> "PromotionGateResult":
        return cls.from_dict(require_json_object(loads_json(text), context="Regime PromotionGateResult JSON"))


def _classification_check(gate_input: PromotionGateInput) -> tuple[list[str], list[str], dict[str, Any]]:
    blocking: list[str] = []
    warnings: list[str] = []
    candidates: dict[str, str] = {}
    if gate_input.study_key is not None:
        candidates["study_key"] = gate_input.study_key.classification
    if gate_input.artifact_classification is not None:
        candidates["artifact_classification"] = gate_input.artifact_classification
    for key in ("classification", "production_classification", "artifact_classification"):
        value = _text_or_none(gate_input.artifact_metadata.get(key))
        if value is not None:
            candidates[f"metadata.{key}"] = value
    unknown = {source: value for source, value in candidates.items() if value not in REGIME_CLASSIFICATION_VALUES}
    if unknown:
        for source, value in sorted(unknown.items()):
            blocking.append(f"{source} has unsupported classification {value!r}")
    recognized = {value for value in candidates.values() if value in REGIME_CLASSIFICATION_VALUES}
    if not candidates:
        blocking.append("classification evidence is missing")
    elif len(recognized) > 1:
        blocking.append(f"classification evidence is inconsistent: {sorted(recognized)}")
    elif recognized:
        classification = next(iter(recognized))
        if classification in NON_PRODUCTION_BLOCKING_CLASSIFICATIONS:
            blocking.append(f"classification {classification!r} is not eligible for production promotion")
        elif classification == RegimeClassification.STAGED.value:
            warnings.append("staged classification requires further validation before production")
    return blocking, warnings, {"classifications": candidates}


def _artifact_safety_check(gate_input: PromotionGateInput) -> tuple[list[str], list[str], dict[str, Any]]:
    blocking: list[str] = []
    warnings: list[str] = []
    artifact_kind = gate_input.artifact_kind
    if artifact_kind is None:
        warnings.append("artifact_kind evidence is missing")
    elif any(marker in artifact_kind for marker in BLOCKING_ARTIFACT_MARKERS):
        blocking.append(f"artifact_kind {artifact_kind!r} is sandbox, scaffold, or diagnostics-only evidence")
    for key in BLOCKING_METADATA_FLAGS:
        if _metadata_bool(gate_input.artifact_metadata, key):
            blocking.append(f"artifact metadata flag {key!r} blocks production promotion")
    metadata_status = _text_or_none(gate_input.artifact_metadata.get("status"))
    if metadata_status in {"scaffold_only", "diagnostics_only", "metadata_only", "sandbox_only"}:
        blocking.append(f"artifact metadata status {metadata_status!r} blocks production promotion")
    write_kind = _text_or_none(gate_input.artifact_metadata.get("write_kind"))
    if write_kind in {"diagnostic", "diagnostics", "scaffold", "sandbox"}:
        blocking.append(f"artifact metadata write_kind {write_kind!r} blocks production promotion")
    sandbox_paths = {
        name: path for name, path in gate_input.artifact_paths.items() if _path_has_sandbox_component(path)
    }
    for name in sorted(sandbox_paths):
        blocking.append(f"artifact path {name!r} points under a sandbox root")
    return blocking, warnings, {"artifact_kind": artifact_kind, "sandbox_paths": sandbox_paths}


def _run_status_check(gate_input: PromotionGateInput) -> tuple[list[str], list[str], dict[str, Any]]:
    status = gate_input.run_status
    if status is None:
        return [], ["run_status evidence is missing"], {"run_status": None}
    if status not in REGIME_RUN_STATUS_VALUES:
        return [f"run_status {status!r} is unsupported"], [], {"run_status": status}
    if status != RunStatus.SUCCEEDED.value:
        return [f"run_status {status!r} is not a completed successful study"], [], {"run_status": status}
    return [], [], {"run_status": status}


def _scoreboard_completeness_check(
    gate_input: PromotionGateInput,
) -> tuple[list[str], list[str], dict[str, Any], dict[str, Any] | None]:
    blocking: list[str] = []
    warnings: list[str] = []
    sections = _scoreboard_sections(gate_input.scoreboard)
    section_statuses: dict[str, str | None] = {}
    if sections is None:
        return ["scoreboard evidence is missing or does not expose required sections"], warnings, {"sections": None}, None
    for section_name in REQUIRED_SCOREBOARD_SECTIONS:
        section = sections.get(section_name)
        if not isinstance(section, Mapping):
            blocking.append(f"scoreboard section {section_name!r} is missing or not a JSON object")
            section_statuses[section_name] = None
            continue
        status = _text_or_none(section.get("status"))
        section_statuses[section_name] = status
        if status is None:
            blocking.append(f"scoreboard section {section_name!r} is missing a status")
        elif status == "failed":
            blocking.append(f"scoreboard section {section_name!r} failed")
        elif section_name == "coverage_degeneracy" and status != "computed":
            blocking.append("coverage_degeneracy must be computed")
        elif status != "computed":
            warnings.append(f"scoreboard section {section_name!r} is {status!r}, not computed")
    return blocking, warnings, {"section_statuses": section_statuses}, sections


def _degeneracy_check(
    gate_input: PromotionGateInput,
    sections: Mapping[str, Any] | None,
) -> tuple[list[str], list[str], dict[str, Any]]:
    blocking: list[str] = []
    warnings: list[str] = []
    if sections is None:
        return blocking, warnings, {"coverage_metrics": None}
    coverage = sections.get("coverage_degeneracy")
    if not isinstance(coverage, Mapping):
        return blocking, warnings, {"coverage_metrics": None}
    metrics_raw = coverage.get("metrics")
    if not isinstance(metrics_raw, Mapping):
        blocking.append("coverage_degeneracy metrics are missing")
        return blocking, warnings, {"coverage_metrics": None}
    metrics = dict(metrics_raw)
    row_count = _int_metric(metrics, "row_count")
    effective_state_count = _int_metric(metrics, "effective_state_count")
    all_noise = _bool_metric(metrics, "all_noise_flag")
    all_unknown_or_null = _bool_metric(metrics, "all_unknown_or_null_flag")
    one_cluster = _bool_metric(metrics, "one_cluster_flag")
    tiny_share = _float_metric(metrics, "singleton_or_tiny_cluster_share")
    if row_count is None or row_count <= 0:
        blocking.append("coverage_degeneracy row_count is missing or non-positive")
    if all_noise:
        blocking.append("coverage_degeneracy is all noise")
    if all_unknown_or_null:
        blocking.append("coverage_degeneracy is all unknown or null labels")
    if effective_state_count is None:
        blocking.append("coverage_degeneracy effective_state_count is missing")
    elif effective_state_count <= 0:
        blocking.append("coverage_degeneracy has no effective state labels")
    if one_cluster:
        flat_allowed = (
            gate_input.flat_asset_policy == FLAT_ASSET_POLICY_ALLOW_SINGLE_STATE
            and gate_input.study_key is not None
            and gate_input.study_key.layer == RegimeLayer.ASSET_STATE.value
            and not all_noise
            and not all_unknown_or_null
            and effective_state_count == 1
        )
        if flat_allowed:
            warnings.append("single-state asset output accepted only for further validation by flat_asset_policy")
        else:
            blocking.append("coverage_degeneracy one-cluster output is blocked by flat_asset_policy")
    if tiny_share is not None and tiny_share >= 0.5:
        warnings.append("singleton_or_tiny_cluster_share is high")
    return (
        blocking,
        warnings,
        {
            "coverage_metrics": {
                "row_count": row_count,
                "effective_state_count": effective_state_count,
                "all_noise_flag": all_noise,
                "all_unknown_or_null_flag": all_unknown_or_null,
                "one_cluster_flag": one_cluster,
                "singleton_or_tiny_cluster_share": tiny_share,
            }
        },
    )


def _allowed_next_actions(status: str) -> tuple[str, ...]:
    if status == PROMOTION_STATUS_BLOCKED:
        return BLOCKED_ALLOWED_NEXT_ACTIONS
    if status == PROMOTION_STATUS_WARNING:
        return WARNING_ALLOWED_NEXT_ACTIONS
    if status == PROMOTION_STATUS_NOT_APPLICABLE:
        return NOT_APPLICABLE_ALLOWED_NEXT_ACTIONS
    return ELIGIBLE_ALLOWED_NEXT_ACTIONS


def evaluate_promotion_gate(gate_input: PromotionGateInput | Mapping[str, Any]) -> PromotionGateResult:
    gate = gate_input if isinstance(gate_input, PromotionGateInput) else PromotionGateInput.from_dict(gate_input)
    checks: dict[str, Any] = {}
    if gate.requested_action != PROMOTION_ACTION_PRODUCTION:
        return PromotionGateResult(
            gate_id=gate.gate_id,
            status=PROMOTION_STATUS_NOT_APPLICABLE,
            allowed_next_actions=_allowed_next_actions(PROMOTION_STATUS_NOT_APPLICABLE),
            warning_reasons=(f"requested_action {gate.requested_action!r} is outside this production promotion gate",),
            checks={"requested_action": gate.requested_action},
            input_summary=_input_summary(gate),
        )

    blocking_reasons: list[str] = []
    warning_reasons: list[str] = []

    block, warn, check = _classification_check(gate)
    blocking_reasons.extend(block)
    warning_reasons.extend(warn)
    checks["classification_safety"] = check

    block, warn, check = _artifact_safety_check(gate)
    blocking_reasons.extend(block)
    warning_reasons.extend(warn)
    checks["artifact_safety"] = check

    block, warn, check = _run_status_check(gate)
    blocking_reasons.extend(block)
    warning_reasons.extend(warn)
    checks["run_status"] = check

    block, warn, check, sections = _scoreboard_completeness_check(gate)
    blocking_reasons.extend(block)
    warning_reasons.extend(warn)
    checks["scoreboard_completeness"] = check

    block, warn, check = _degeneracy_check(gate, sections)
    blocking_reasons.extend(block)
    warning_reasons.extend(warn)
    checks["degeneracy"] = check

    blocking_reasons = sorted(dict.fromkeys(blocking_reasons))
    warning_reasons = sorted(dict.fromkeys(warning_reasons))
    if blocking_reasons:
        status = PROMOTION_STATUS_BLOCKED
    elif warning_reasons:
        status = PROMOTION_STATUS_WARNING
    else:
        status = PROMOTION_STATUS_ELIGIBLE_FOR_FURTHER_VALIDATION

    return PromotionGateResult(
        gate_id=gate.gate_id,
        status=status,
        allowed_next_actions=_allowed_next_actions(status),
        blocking_reasons=blocking_reasons,
        warning_reasons=warning_reasons,
        checks=checks,
        input_summary=_input_summary(gate),
    )


def _input_summary(gate: PromotionGateInput) -> dict[str, Any]:
    return {
        "study_id": None if gate.study_key is None else gate.study_key.study_id,
        "layer": None if gate.study_key is None else gate.study_key.layer,
        "axis": None if gate.study_key is None else gate.study_key.axis,
        "band": None if gate.study_key is None else gate.study_key.band,
        "classification": None if gate.study_key is None else gate.study_key.classification,
        "artifact_kind": gate.artifact_kind,
        "artifact_classification": gate.artifact_classification,
        "run_status": gate.run_status,
        "requested_action": gate.requested_action,
        "flat_asset_policy": gate.flat_asset_policy,
    }


__all__ = [
    "BLOCKED_ALLOWED_NEXT_ACTIONS",
    "ELIGIBLE_ALLOWED_NEXT_ACTIONS",
    "FLAT_ASSET_POLICIES",
    "FLAT_ASSET_POLICY_ALLOW_SINGLE_STATE",
    "FLAT_ASSET_POLICY_BLOCK",
    "NOT_APPLICABLE_ALLOWED_NEXT_ACTIONS",
    "NON_PRODUCTION_BLOCKING_CLASSIFICATIONS",
    "PROMOTION_ACTION_PRODUCTION",
    "PROMOTION_GATE_STATUSES",
    "PROMOTION_STATUS_BLOCKED",
    "PROMOTION_STATUS_ELIGIBLE_FOR_FURTHER_VALIDATION",
    "PROMOTION_STATUS_NOT_APPLICABLE",
    "PROMOTION_STATUS_WARNING",
    "REGIME_PROMOTION_GATE_SCHEMA_VERSION",
    "REQUIRED_SCOREBOARD_SECTIONS",
    "WARNING_ALLOWED_NEXT_ACTIONS",
    "PromotionGateInput",
    "PromotionGateResult",
    "evaluate_promotion_gate",
]
