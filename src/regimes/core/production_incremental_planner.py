from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

from src.regimes.core.production_consumer import (
    REGIME_BRANCH_ASSET_STATE,
    REGIME_BRANCH_CROSS_ASSET_STATE,
    REGIME_BRANCH_MARKET_STATE,
    REGIME_PRODUCTION_BRANCHES,
)
from src.regimes.core.production_planner import (
    REGIME_PRODUCTION_REFIT_CADENCE_BIWEEKLY,
    REGIME_PRODUCTION_REFIT_CADENCE_MONTHLY,
    REGIME_PRODUCTION_REFIT_CADENCE_WEEKLY,
    resolve_regime_production_refit_cadence,
)
from src.regimes.core.production_relationship_freshness import (
    RELATIONSHIP_FRESHNESS_STATUS_AVAILABLE,
    RELATIONSHIP_FRESHNESS_STATUS_MISSING,
    RELATIONSHIP_FRESHNESS_STATUS_STALE,
)
from src.regimes.core.serialization import to_jsonable


REGIME_PRODUCTION_INCREMENTAL_PLANNER_SCHEMA_VERSION = 1
REGIME_PRODUCTION_INCREMENTAL_PLAN_ARTIFACT_KIND = "regime_production_incremental_update_plan"

REGIME_PRODUCTION_INCREMENTAL_STATUS_NOOP = "noop_no_new_data"
REGIME_PRODUCTION_INCREMENTAL_STATUS_APPLY_EXISTING = "incremental_apply_existing_definition"
REGIME_PRODUCTION_INCREMENTAL_STATUS_REFIT_DEFINITION = "incremental_refit_definition"
REGIME_PRODUCTION_INCREMENTAL_STATUS_CROSS_RELATIONSHIP_UNAVAILABLE = "incremental_cross_relationship_unavailable"

REGIME_PRODUCTION_DEFINITION_ACTION_REUSE = "reuse_existing_persisted_definition"
REGIME_PRODUCTION_DEFINITION_ACTION_UPDATE = "update_definition_on_refit_cadence"

REGIME_PRODUCTION_RELATIONSHIP_ACTION_NOT_APPLICABLE = "not_applicable"
REGIME_PRODUCTION_RELATIONSHIP_ACTION_AVAILABLE = "relationship_inputs_available_for_new_range"
REGIME_PRODUCTION_RELATIONSHIP_ACTION_MASK_UNAVAILABLE = "mask_or_unavailable_new_range"

REGIME_PRODUCTION_INCREMENTAL_HEALTHY_DEFINITION_STATUSES: tuple[str, ...] = (
    "active",
    "available",
    "healthy",
    "planned",
    "valid",
)

_CADENCE_SECONDS: Mapping[str, int] = {
    REGIME_PRODUCTION_REFIT_CADENCE_WEEKLY: 7 * 24 * 60 * 60,
    REGIME_PRODUCTION_REFIT_CADENCE_BIWEEKLY: 14 * 24 * 60 * 60,
    REGIME_PRODUCTION_REFIT_CADENCE_MONTHLY: 31 * 24 * 60 * 60,
}


@dataclass(frozen=True)
class RegimeProductionIncrementalPlan:
    branch: str
    status: str
    source_tail_ts: Any
    last_output_tail_ts: Any
    source_tail_epoch_seconds: int
    last_output_tail_epoch_seconds: int
    source_tail_comparison: str
    next_required_timestamp: int | None
    output_range_start: int | None
    output_range_end: int | None
    output_range_closed: bool
    refit_cadence_id: str
    refit_cadence: str
    refit_cadence_seconds: int
    last_definition_refit_ts: Any = None
    last_definition_refit_epoch_seconds: int | None = None
    refit_boundary_ts: int | None = None
    refit_boundary_crossed: bool = False
    definition_action: str = REGIME_PRODUCTION_DEFINITION_ACTION_REUSE
    definition_health_status: str = "healthy"
    local_reselection_eligible: bool = False
    relationship_input_tail_ts: Any = None
    previous_relationship_input_tail_ts: Any = None
    relationship_input_tail_epoch_seconds: int | None = None
    previous_relationship_input_tail_epoch_seconds: int | None = None
    relationship_input_advanced: bool | None = None
    relationship_tail_comparison: str = REGIME_PRODUCTION_RELATIONSHIP_ACTION_NOT_APPLICABLE
    relationship_input_action: str = REGIME_PRODUCTION_RELATIONSHIP_ACTION_NOT_APPLICABLE
    relationship_freshness_status: str | None = None
    reason_codes: Sequence[str] = field(default_factory=tuple)
    warnings: Sequence[str] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        branch = _branch_name(self.branch)
        reasons = tuple(dict.fromkeys(str(item) for item in self.reason_codes if str(item or "").strip()))
        warnings = tuple(dict.fromkeys(str(item) for item in self.warnings if str(item or "").strip()))
        object.__setattr__(self, "branch", branch)
        object.__setattr__(self, "reason_codes", reasons)
        object.__setattr__(self, "warnings", warnings)

    @property
    def no_production_writes(self) -> bool:
        return True

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": REGIME_PRODUCTION_INCREMENTAL_PLANNER_SCHEMA_VERSION,
            "artifact_kind": REGIME_PRODUCTION_INCREMENTAL_PLAN_ARTIFACT_KIND,
            "branch": self.branch,
            "status": self.status,
            "source_tail_ts": to_jsonable(self.source_tail_ts),
            "last_output_tail_ts": to_jsonable(self.last_output_tail_ts),
            "source_tail_epoch_seconds": self.source_tail_epoch_seconds,
            "last_output_tail_epoch_seconds": self.last_output_tail_epoch_seconds,
            "source_tail_comparison": self.source_tail_comparison,
            "next_required_timestamp": self.next_required_timestamp,
            "output_range_start": self.output_range_start,
            "output_range_end": self.output_range_end,
            "output_range_closed": self.output_range_closed,
            "refit_cadence_id": self.refit_cadence_id,
            "refit_cadence": self.refit_cadence,
            "refit_cadence_seconds": self.refit_cadence_seconds,
            "last_definition_refit_ts": to_jsonable(self.last_definition_refit_ts),
            "last_definition_refit_epoch_seconds": self.last_definition_refit_epoch_seconds,
            "refit_boundary_ts": self.refit_boundary_ts,
            "refit_boundary_crossed": self.refit_boundary_crossed,
            "definition_action": self.definition_action,
            "definition_health_status": self.definition_health_status,
            "use_existing_persisted_definition": self.definition_action == REGIME_PRODUCTION_DEFINITION_ACTION_REUSE,
            "definition_update_planned": self.definition_action == REGIME_PRODUCTION_DEFINITION_ACTION_UPDATE,
            "local_reselection_eligible": self.local_reselection_eligible,
            "local_reselection_execution_performed": False,
            "relationship_input_tail_ts": to_jsonable(self.relationship_input_tail_ts),
            "previous_relationship_input_tail_ts": to_jsonable(self.previous_relationship_input_tail_ts),
            "relationship_input_tail_epoch_seconds": self.relationship_input_tail_epoch_seconds,
            "previous_relationship_input_tail_epoch_seconds": self.previous_relationship_input_tail_epoch_seconds,
            "relationship_input_advanced": self.relationship_input_advanced,
            "relationship_tail_comparison": self.relationship_tail_comparison,
            "relationship_input_action": self.relationship_input_action,
            "relationship_freshness_status": self.relationship_freshness_status,
            "relationship_input_history_separate_from_selected_profile_artifact": self.branch == REGIME_BRANCH_CROSS_ASSET_STATE,
            "relationship_discovery_executed": False,
            "broad_pairwise_run_executed": False,
            "reason_codes": list(self.reason_codes),
            "warnings": list(self.warnings),
            "production_writer_enabled": False,
            "production_labels_written": False,
            "production_outputs_written": False,
            "canonical_production_state_outputs_written": False,
        }


def detect_regime_production_output_tail(
    rows: Sequence[Mapping[str, Any]],
    *,
    timestamp_field: str = "timestamp",
) -> int | None:
    tails: list[int] = []
    for index, row in enumerate(rows):
        value = row.get(timestamp_field)
        if value in (None, ""):
            continue
        tails.append(_epoch_seconds(value, field_name=f"{timestamp_field}[{index}]"))
    return max(tails) if tails else None


def plan_regime_production_incremental_update(
    *,
    branch: str,
    source_tail_ts: Any,
    last_output_tail_ts: Any,
    last_definition_refit_ts: Any | None,
    interval_seconds: int,
    requested_cadence: str | None = None,
    definition_health_status: str = "healthy",
    relationship_input_tail_ts: Any | None = None,
    previous_relationship_input_tail_ts: Any | None = None,
    relationship_freshness_status: str | None = None,
) -> RegimeProductionIncrementalPlan:
    branch_name = _branch_name(branch)
    if int(interval_seconds) <= 0:
        raise ValueError("Regime Production incremental interval_seconds must be positive")
    interval = int(interval_seconds)
    source_tail = _epoch_seconds(source_tail_ts, field_name="source_tail_ts")
    last_output_tail = _epoch_seconds(last_output_tail_ts, field_name="last_output_tail_ts")
    if source_tail < last_output_tail:
        raise ValueError("Regime Production source_tail_ts precedes last production output tail")

    interval_minutes = max(1, interval // 60)
    cadence_contract = resolve_regime_production_refit_cadence(
        branch_name,
        interval_minutes=interval_minutes,
        requested_cadence=requested_cadence,
    )
    cadence_seconds = _CADENCE_SECONDS[cadence_contract.cadence]

    last_refit = None
    refit_boundary_ts = None
    refit_boundary_crossed = False
    reason_codes: list[str] = []
    warnings: list[str] = []
    if last_definition_refit_ts in (None, ""):
        reason_codes.append("persisted_definition_refit_checkpoint_missing")
        refit_boundary_crossed = source_tail > last_output_tail
    else:
        last_refit = _epoch_seconds(last_definition_refit_ts, field_name="last_definition_refit_ts")
        refit_boundary_ts = last_refit + cadence_seconds
        refit_boundary_crossed = source_tail >= refit_boundary_ts and source_tail > last_output_tail

    source_tail_comparison = "equal_to_last_output_tail" if source_tail == last_output_tail else "advanced_after_last_output_tail"
    output_range_start = None
    output_range_end = None
    next_required_timestamp = None
    status = REGIME_PRODUCTION_INCREMENTAL_STATUS_NOOP
    definition_action = REGIME_PRODUCTION_DEFINITION_ACTION_REUSE
    if source_tail > last_output_tail:
        next_required_timestamp = last_output_tail + interval
        output_range_start = next_required_timestamp
        output_range_end = source_tail
        if refit_boundary_crossed:
            status = REGIME_PRODUCTION_INCREMENTAL_STATUS_REFIT_DEFINITION
            definition_action = REGIME_PRODUCTION_DEFINITION_ACTION_UPDATE
            reason_codes.append("refit_cadence_boundary_crossed")
        else:
            status = REGIME_PRODUCTION_INCREMENTAL_STATUS_APPLY_EXISTING
            reason_codes.append("new_source_tail_within_existing_refit_window")
    else:
        reason_codes.append("no_new_source_tail")

    health_status = str(definition_health_status or "").strip().lower() or "missing"
    local_reselection_eligible = health_status not in REGIME_PRODUCTION_INCREMENTAL_HEALTHY_DEFINITION_STATUSES
    if local_reselection_eligible:
        reason_codes.append("definition_health_failed_local_reselection_eligible")
        warnings.append("definition_health_failed_reselection_is_eligible_but_not_executed")

    relationship_current = None
    relationship_previous = None
    relationship_advanced = None
    relationship_tail_comparison = REGIME_PRODUCTION_RELATIONSHIP_ACTION_NOT_APPLICABLE
    relationship_action = REGIME_PRODUCTION_RELATIONSHIP_ACTION_NOT_APPLICABLE
    freshness_status = relationship_freshness_status
    if branch_name == REGIME_BRANCH_CROSS_ASSET_STATE:
        relationship_current = (
            None
            if relationship_input_tail_ts in (None, "")
            else _epoch_seconds(relationship_input_tail_ts, field_name="relationship_input_tail_ts")
        )
        relationship_previous = (
            None
            if previous_relationship_input_tail_ts in (None, "")
            else _epoch_seconds(previous_relationship_input_tail_ts, field_name="previous_relationship_input_tail_ts")
        )
        if relationship_current is not None and relationship_previous is not None:
            relationship_advanced = relationship_current > relationship_previous
        if freshness_status is None:
            freshness_status = RELATIONSHIP_FRESHNESS_STATUS_MISSING if relationship_current is None else RELATIONSHIP_FRESHNESS_STATUS_AVAILABLE
        freshness_status = str(freshness_status).strip().lower()
        if relationship_current is None:
            relationship_tail_comparison = "missing_relationship_input_tail"
            relationship_action = REGIME_PRODUCTION_RELATIONSHIP_ACTION_MASK_UNAVAILABLE
            reason_codes.append("relationship_input_tail_ts_missing")
        elif relationship_current < source_tail:
            relationship_tail_comparison = "relationship_input_tail_lags_source_tail"
            if freshness_status == RELATIONSHIP_FRESHNESS_STATUS_AVAILABLE:
                freshness_status = RELATIONSHIP_FRESHNESS_STATUS_STALE
            relationship_action = REGIME_PRODUCTION_RELATIONSHIP_ACTION_MASK_UNAVAILABLE
            reason_codes.append("relationship_input_tail_lags_source_tail")
        else:
            relationship_tail_comparison = "relationship_input_tail_covers_source_tail"
            relationship_action = REGIME_PRODUCTION_RELATIONSHIP_ACTION_AVAILABLE
        if freshness_status in {RELATIONSHIP_FRESHNESS_STATUS_STALE, RELATIONSHIP_FRESHNESS_STATUS_MISSING}:
            relationship_action = REGIME_PRODUCTION_RELATIONSHIP_ACTION_MASK_UNAVAILABLE
            reason_codes.append("relationship_input_freshness_not_available")
            status = REGIME_PRODUCTION_INCREMENTAL_STATUS_CROSS_RELATIONSHIP_UNAVAILABLE
            warnings.append("cross_relationship_input_not_current_for_source_tail")

    return RegimeProductionIncrementalPlan(
        branch=branch_name,
        status=status,
        source_tail_ts=source_tail_ts,
        last_output_tail_ts=last_output_tail_ts,
        source_tail_epoch_seconds=source_tail,
        last_output_tail_epoch_seconds=last_output_tail,
        source_tail_comparison=source_tail_comparison,
        next_required_timestamp=next_required_timestamp,
        output_range_start=output_range_start,
        output_range_end=output_range_end,
        output_range_closed=output_range_start is not None and output_range_end is not None,
        refit_cadence_id=cadence_contract.refit_cadence_id,
        refit_cadence=cadence_contract.cadence,
        refit_cadence_seconds=cadence_seconds,
        last_definition_refit_ts=last_definition_refit_ts,
        last_definition_refit_epoch_seconds=last_refit,
        refit_boundary_ts=refit_boundary_ts,
        refit_boundary_crossed=refit_boundary_crossed,
        definition_action=definition_action,
        definition_health_status=health_status,
        local_reselection_eligible=local_reselection_eligible,
        relationship_input_tail_ts=relationship_input_tail_ts,
        previous_relationship_input_tail_ts=previous_relationship_input_tail_ts,
        relationship_input_tail_epoch_seconds=relationship_current,
        previous_relationship_input_tail_epoch_seconds=relationship_previous,
        relationship_input_advanced=relationship_advanced,
        relationship_tail_comparison=relationship_tail_comparison,
        relationship_input_action=relationship_action,
        relationship_freshness_status=freshness_status,
        reason_codes=tuple(reason_codes),
        warnings=tuple(warnings),
    )


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


def _epoch_seconds(value: Any, *, field_name: str) -> int:
    if value is None or value == "":
        raise ValueError(f"Regime Production incremental {field_name} is required")
    if isinstance(value, bool):
        raise ValueError(f"Regime Production incremental {field_name} must be a timestamp")
    if isinstance(value, (int, float)):
        return int(value)
    text = str(value).strip()
    if not text:
        raise ValueError(f"Regime Production incremental {field_name} is required")
    try:
        return int(float(text))
    except ValueError:
        pass
    iso_text = text[:-1] + "+00:00" if text.endswith("Z") else text
    try:
        parsed = datetime.fromisoformat(iso_text)
    except ValueError as exc:
        raise ValueError(f"Regime Production incremental {field_name} is not orderable") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return int(parsed.timestamp())


__all__ = [
    "REGIME_PRODUCTION_DEFINITION_ACTION_REUSE",
    "REGIME_PRODUCTION_DEFINITION_ACTION_UPDATE",
    "REGIME_PRODUCTION_INCREMENTAL_PLAN_ARTIFACT_KIND",
    "REGIME_PRODUCTION_INCREMENTAL_PLANNER_SCHEMA_VERSION",
    "REGIME_PRODUCTION_INCREMENTAL_STATUS_APPLY_EXISTING",
    "REGIME_PRODUCTION_INCREMENTAL_STATUS_CROSS_RELATIONSHIP_UNAVAILABLE",
    "REGIME_PRODUCTION_INCREMENTAL_STATUS_NOOP",
    "REGIME_PRODUCTION_INCREMENTAL_STATUS_REFIT_DEFINITION",
    "REGIME_PRODUCTION_RELATIONSHIP_ACTION_AVAILABLE",
    "REGIME_PRODUCTION_RELATIONSHIP_ACTION_MASK_UNAVAILABLE",
    "REGIME_PRODUCTION_RELATIONSHIP_ACTION_NOT_APPLICABLE",
    "RegimeProductionIncrementalPlan",
    "detect_regime_production_output_tail",
    "plan_regime_production_incremental_update",
]
