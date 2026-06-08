from __future__ import annotations

import calendar
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Mapping, Sequence

from src.forecasting.common.forecast_family_core import fit_window_start
from src.forecasting.ml.shared.numeric_cohort_common import (
    CLAMP_START_MONTH,
    CLAMP_START_YEAR,
    DEFAULT_COHORT_WINDOW_MONTHS,
    DEFAULT_SEARCH_BACK_MONTHS,
)
from src.forecasting.ml.shared.production_time import (
    PRODUCTION_START_TS_ENV,
    production_start_ts,
)
from src.regimes.core.production_consumer import (
    REGIME_BRANCH_ASSET_STATE,
    REGIME_BRANCH_CROSS_ASSET_STATE,
    REGIME_BRANCH_MARKET_STATE,
    REGIME_PRODUCTION_BRANCHES,
)
from src.regimes.core.serialization import to_jsonable


REGIME_PRODUCTION_CLAMP_RANGE_ARTIFACT_KIND = "regime_production_normalized_clamp_range"
REGIME_PRODUCTION_RELATIONSHIP_HISTORY_CHECK_ARTIFACT_KIND = "regime_production_relationship_history_clamp_check"
REGIME_PRODUCTION_CLAMP_RANGE_SCHEMA_VERSION = 1

CLAMP_RANGE_STATUS_READY = "range_ready"
CLAMP_RANGE_STATUS_AUDITABLE_UNAVAILABLE = "auditable_unavailable"
CLAMP_RANGE_STATUS_BLOCKED = "blocked"

RELATIONSHIP_HISTORY_STATUS_AVAILABLE = "available"
RELATIONSHIP_HISTORY_STATUS_UNAVAILABLE = "unavailable"

NUMERIC_CLAMP_POLICY_SOURCE_MODULES: tuple[str, ...] = (
    "src.forecasting.ml.shared.production_time.production_start_ts",
    "src.forecasting.ml.shared.numeric_cohort_common",
    "src.forecasting.common.forecast_family_core.fit_window_start",
)


@dataclass(frozen=True)
class RegimeProductionNormalizedClampRange:
    branch: str
    policy_id: str
    source_tail_ts: Any
    known_at_ts: Any
    production_input_edge_ts: Any
    row_status: str | None
    output_start_ts: int
    output_end_ts: int
    required_lookback_start_ts: int
    historical_output_months: int
    required_lookback_months: int
    source_tail_required: bool
    reason_codes: Sequence[str] = ()
    source_tail_orderable_ts: int | None = None
    known_at_orderable_ts: int | None = None
    production_input_edge_orderable_ts: int | None = None
    numeric_forecaster_policy_reused: bool = True
    numeric_policy_source_modules: Sequence[str] = NUMERIC_CLAMP_POLICY_SOURCE_MODULES
    production_start_ts_env: str = PRODUCTION_START_TS_ENV

    def __post_init__(self) -> None:
        branch = _branch_name(self.branch)
        reasons = tuple(dict.fromkeys(str(reason) for reason in self.reason_codes if str(reason or "").strip()))
        if int(self.historical_output_months) <= 0 or int(self.required_lookback_months) <= 0:
            raise ValueError("Regime Production clamp range month counts must be positive")
        if int(self.output_end_ts) < int(self.output_start_ts):
            raise ValueError("Regime Production clamp range output_end_ts must be >= output_start_ts")
        if int(self.required_lookback_start_ts) > int(self.output_start_ts):
            raise ValueError("Regime Production clamp range lookback must start no later than output_start_ts")
        object.__setattr__(self, "branch", branch)
        object.__setattr__(self, "policy_id", _text(self.policy_id, field_name="policy_id"))
        object.__setattr__(self, "output_start_ts", int(self.output_start_ts))
        object.__setattr__(self, "output_end_ts", int(self.output_end_ts))
        object.__setattr__(self, "required_lookback_start_ts", int(self.required_lookback_start_ts))
        object.__setattr__(self, "historical_output_months", int(self.historical_output_months))
        object.__setattr__(self, "required_lookback_months", int(self.required_lookback_months))
        object.__setattr__(self, "source_tail_required", bool(self.source_tail_required))
        object.__setattr__(self, "reason_codes", reasons)
        object.__setattr__(self, "numeric_policy_source_modules", tuple(str(item) for item in self.numeric_policy_source_modules))
        object.__setattr__(self, "production_start_ts_env", str(self.production_start_ts_env))

    @property
    def status(self) -> str:
        if not self.reason_codes:
            return CLAMP_RANGE_STATUS_READY
        if not self.source_tail_required:
            return CLAMP_RANGE_STATUS_AUDITABLE_UNAVAILABLE
        return CLAMP_RANGE_STATUS_BLOCKED

    @property
    def passed(self) -> bool:
        return self.status == CLAMP_RANGE_STATUS_READY

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": REGIME_PRODUCTION_CLAMP_RANGE_SCHEMA_VERSION,
            "artifact_kind": REGIME_PRODUCTION_CLAMP_RANGE_ARTIFACT_KIND,
            "branch": self.branch,
            "policy_id": self.policy_id,
            "source": "shared_regime_production_clamp_contract",
            "numeric_forecaster_policy_reused": bool(self.numeric_forecaster_policy_reused),
            "numeric_policy_source_modules": list(self.numeric_policy_source_modules),
            "production_start_ts_env": self.production_start_ts_env,
            "row_status": self.row_status,
            "source_tail_required": bool(self.source_tail_required),
            "source_tail_ts": to_jsonable(self.source_tail_ts),
            "source_tail_orderable_ts": self.source_tail_orderable_ts,
            "known_at_ts": to_jsonable(self.known_at_ts),
            "known_at_orderable_ts": self.known_at_orderable_ts,
            "production_input_edge_ts": to_jsonable(self.production_input_edge_ts),
            "production_input_edge_orderable_ts": self.production_input_edge_orderable_ts,
            "output_start_ts": int(self.output_start_ts),
            "output_start": _iso_from_ts(self.output_start_ts),
            "output_end_ts": int(self.output_end_ts),
            "output_end": _iso_from_ts(self.output_end_ts),
            "historical_output_months": int(self.historical_output_months),
            "required_lookback_months": int(self.required_lookback_months),
            "required_lookback_start_ts": int(self.required_lookback_start_ts),
            "required_lookback_start": _iso_from_ts(self.required_lookback_start_ts),
            "required_lookback_source": "src.forecasting.common.forecast_family_core.fit_window_start",
            "roughly_one_year_history_default": int(self.historical_output_months) == int(DEFAULT_COHORT_WINDOW_MONTHS),
            "up_to_one_year_refit_lookback_represented": int(self.required_lookback_months) >= int(DEFAULT_SEARCH_BACK_MONTHS),
            "raw_data_edge_drives_output_end": True,
            "clamp_controls_historical_backfill_floor": True,
            "status": self.status,
            "passed": self.passed,
            "reason_codes": list(self.reason_codes),
            "production_writer_enabled": False,
            "production_labels_written": False,
            "production_outputs_written": False,
            "canonical_production_state_outputs_written": False,
        }


def normalized_regime_production_clamp_range(
    *,
    branch: str,
    clamp_policy: Mapping[str, Any] | Any,
    source_tail_ts: Any = None,
    known_at_ts: Any = None,
    production_input_edge_ts: Any = None,
    row_status: str | None = None,
    source_tail_required: bool = True,
) -> RegimeProductionNormalizedClampRange:
    branch_name = _branch_name(branch)
    policy_payload = _policy_payload(clamp_policy)
    output_start_ts = _output_start_ts(policy_payload)
    historical_months = int(policy_payload.get("historical_output_months") or DEFAULT_COHORT_WINDOW_MONTHS)
    required_lookback_months = int(policy_payload.get("required_lookback_months") or DEFAULT_SEARCH_BACK_MONTHS)
    lookback_start_ts = int(
        fit_window_start(
            edge_ts=int(output_start_ts),
            fit_days=max(1, int(required_lookback_months)) * 31,
            min_ts=0,
        )
    )
    source_tail_value, source_reason = _optional_ts(source_tail_ts, field_name="source_tail_ts")
    known_at_value, known_reason = _optional_ts(known_at_ts, field_name="known_at_ts")
    input_edge_source = production_input_edge_ts
    input_edge_value, input_edge_reason = _optional_ts(input_edge_source, field_name="production_input_edge_ts")
    output_end_ts = int(input_edge_value) if input_edge_value is not None else _add_months_to_ts(output_start_ts, max(0, historical_months - 1))
    reasons: list[str] = []
    if source_reason:
        reasons.append(source_reason)
    if known_reason:
        reasons.append(known_reason)
    if input_edge_reason:
        reasons.append(input_edge_reason)
    if source_tail_value is None:
        if source_tail_required:
            reasons.append("source_tail_ts_missing_for_clamp_range")
    else:
        if int(source_tail_value) < int(output_start_ts):
            reasons.append("source_tail_before_clamp_output_start")
    if input_edge_value is None:
        if source_tail_required:
            reasons.append("production_input_edge_ts_missing_for_clamp_range")
    elif int(input_edge_value) < int(output_start_ts):
        reasons.append("production_input_edge_before_clamp_output_start")
    if source_tail_value is not None and known_at_value is not None and int(source_tail_value) > int(known_at_value):
        reasons.append("source_tail_after_known_at_for_clamp_range")
    return RegimeProductionNormalizedClampRange(
        branch=branch_name,
        policy_id=str(policy_payload.get("policy_id") or "regime_production_numeric_forecast_common_recent_window_v1"),
        source_tail_ts=source_tail_ts,
        known_at_ts=known_at_ts,
        production_input_edge_ts=input_edge_source,
        row_status=row_status,
        output_start_ts=output_start_ts,
        output_end_ts=output_end_ts,
        required_lookback_start_ts=lookback_start_ts,
        historical_output_months=historical_months,
        required_lookback_months=required_lookback_months,
        source_tail_required=bool(source_tail_required),
        reason_codes=tuple(reasons),
        source_tail_orderable_ts=source_tail_value,
        known_at_orderable_ts=known_at_value,
        production_input_edge_orderable_ts=input_edge_value,
    )


def relationship_input_history_check(
    *,
    branch: str,
    clamp_range: Mapping[str, Any] | RegimeProductionNormalizedClampRange,
    relationship_input_tail_ts: Any,
    relationship_known_at_ts: Any,
    snapshot_cadence_days: Any,
) -> dict[str, Any]:
    branch_name = _branch_name(branch)
    if branch_name != REGIME_BRANCH_CROSS_ASSET_STATE:
        raise ValueError("Regime Production relationship history checks are Cross-Asset only")
    clamp_payload = clamp_range.as_dict() if isinstance(clamp_range, RegimeProductionNormalizedClampRange) else dict(clamp_range)
    output_start_ts = int(clamp_payload["output_start_ts"])
    output_end_ts = int(clamp_payload["output_end_ts"])
    tail_value, tail_reason = _optional_ts(relationship_input_tail_ts, field_name="relationship_input_tail_ts")
    known_value, known_reason = _optional_ts(relationship_known_at_ts, field_name="relationship_known_at_ts")
    cadence_value, cadence_reason = _positive_int(snapshot_cadence_days, field_name="snapshot_cadence_days")
    reasons: list[str] = []
    for reason in (tail_reason, known_reason, cadence_reason):
        if reason:
            reasons.append(reason)
    if tail_value is None:
        reasons.append("relationship_input_tail_ts_missing_for_clamp_history")
    if known_value is None:
        reasons.append("relationship_known_at_ts_missing_for_clamp_history")
    if tail_value is not None and known_value is not None and int(tail_value) > int(known_value):
        reasons.append("relationship_input_tail_after_known_at")
    latest_acceptable_tail_ts = None
    if cadence_value is not None:
        latest_acceptable_tail_ts = int(output_end_ts) - int(cadence_value) * 86400
    covers_output_start = tail_value is not None and int(tail_value) >= int(output_start_ts)
    covers_output_end = tail_value is not None and int(tail_value) >= int(output_end_ts)
    within_cadence_for_output_end = (
        tail_value is not None
        and latest_acceptable_tail_ts is not None
        and int(tail_value) >= int(latest_acceptable_tail_ts)
    )
    if tail_value is not None and not covers_output_start:
        reasons.append("relationship_input_tail_before_clamp_output_start")
    if tail_value is not None and latest_acceptable_tail_ts is not None and not within_cadence_for_output_end:
        reasons.append("relationship_input_tail_stale_for_clamp_output_end")
    reasons = list(dict.fromkeys(reasons))
    return {
        "schema_version": REGIME_PRODUCTION_CLAMP_RANGE_SCHEMA_VERSION,
        "artifact_kind": REGIME_PRODUCTION_RELATIONSHIP_HISTORY_CHECK_ARTIFACT_KIND,
        "branch": branch_name,
        "source": "shared_regime_production_clamp_contract",
        "clamp_policy_id": clamp_payload.get("policy_id"),
        "output_start_ts": output_start_ts,
        "output_start": _iso_from_ts(output_start_ts),
        "output_end_ts": output_end_ts,
        "output_end": _iso_from_ts(output_end_ts),
        "relationship_input_tail_ts": to_jsonable(relationship_input_tail_ts),
        "relationship_input_tail_orderable_ts": tail_value,
        "relationship_known_at_ts": to_jsonable(relationship_known_at_ts),
        "relationship_known_at_orderable_ts": known_value,
        "snapshot_cadence_days": cadence_value,
        "latest_acceptable_tail_ts": latest_acceptable_tail_ts,
        "latest_acceptable_tail": _iso_from_ts(latest_acceptable_tail_ts) if latest_acceptable_tail_ts is not None else None,
        "covers_output_start": bool(covers_output_start),
        "covers_output_end": bool(covers_output_end),
        "within_cadence_for_output_end": bool(within_cadence_for_output_end),
        "status": RELATIONSHIP_HISTORY_STATUS_AVAILABLE if not reasons else RELATIONSHIP_HISTORY_STATUS_UNAVAILABLE,
        "passed": not reasons,
        "reason_codes": reasons,
        "relationship_input_history_separate_from_selected_profile_artifact": True,
        "relationship_discovery_executed": False,
        "broad_pairwise_run_executed": False,
        "selected_profile_artifact": False,
        "production_writer_enabled": False,
        "production_labels_written": False,
        "production_outputs_written": False,
        "canonical_production_state_outputs_written": False,
    }


def checkpoint_timestamps_for_clamp_policy(
    clamp_policy: Mapping[str, Any] | Any,
    *,
    checkpoint_count: int,
) -> tuple[str, ...]:
    policy_payload = _policy_payload(clamp_policy)
    start_ts = _output_start_ts(policy_payload)
    months = max(1, int(policy_payload.get("historical_output_months") or DEFAULT_COHORT_WINDOW_MONTHS))
    count = max(1, int(checkpoint_count))
    offsets = sorted({round(idx * (months - 1) / max(1, count - 1)) for idx in range(count)})
    while len(offsets) < count:
        offsets.append(min(months - 1, len(offsets)))
    return tuple(_iso_from_ts(_add_months_to_ts(start_ts, int(offset))) for offset in offsets[:count])


def materialized_timestamps_for_clamp_policy(
    clamp_policy: Mapping[str, Any] | Any,
    *,
    cadence: str = "monthly",
) -> tuple[str, ...]:
    policy_payload = _policy_payload(clamp_policy)
    start_ts = int(policy_payload.get("output_start_ts") or _output_start_ts(policy_payload))
    if policy_payload.get("output_end_ts") not in (None, ""):
        end_ts = int(policy_payload["output_end_ts"])
    else:
        months = max(1, int(policy_payload.get("historical_output_months") or DEFAULT_COHORT_WINDOW_MONTHS))
        end_ts = _add_months_to_ts(start_ts, max(0, months - 1))
    if end_ts < start_ts:
        raise ValueError("Regime Production materialized timestamp range end must be >= start")

    normalized = str(cadence or "monthly").strip().lower()
    if normalized == "weekly":
        return _fixed_day_timestamps(start_ts, end_ts, days=7)
    if normalized == "biweekly":
        return _fixed_day_timestamps(start_ts, end_ts, days=14)
    return _monthly_timestamps(start_ts, end_ts)


def checkpoint_count_for_cadence(cadence: str, historical_output_months: int) -> int:
    days = max(1, int(historical_output_months)) * 31
    normalized = str(cadence or "monthly").lower()
    if normalized == "weekly":
        return int(math.ceil(days / 7.0))
    if normalized == "biweekly":
        return int(math.ceil(days / 14.0))
    return max(1, int(historical_output_months))


def clamp_policy_window_summary(clamp_policy: Mapping[str, Any] | Any) -> dict[str, Any]:
    policy_payload = _policy_payload(clamp_policy)
    start_ts = _output_start_ts(policy_payload)
    months = int(policy_payload.get("historical_output_months") or DEFAULT_COHORT_WINDOW_MONTHS)
    lookbacks = int(policy_payload.get("required_lookback_months") or DEFAULT_SEARCH_BACK_MONTHS)
    end_ts = _add_months_to_ts(start_ts, max(0, months - 1))
    lookback_start = int(fit_window_start(edge_ts=start_ts, fit_days=max(1, lookbacks) * 31, min_ts=0))
    return {
        "clamp_policy_id": policy_payload.get("policy_id"),
        "historical_output_months": months,
        "required_lookback_months": lookbacks,
        "output_start_ts": int(start_ts),
        "output_start": _iso_from_ts(start_ts),
        "output_end_ts": int(end_ts),
        "output_end": _iso_from_ts(end_ts),
        "required_lookback_start_ts": int(lookback_start),
        "required_lookback_start": _iso_from_ts(lookback_start),
        "numeric_forecaster_policy_reused": True,
        "numeric_policy_source_modules": list(NUMERIC_CLAMP_POLICY_SOURCE_MODULES),
    }


def _policy_payload(clamp_policy: Mapping[str, Any] | Any) -> dict[str, Any]:
    if hasattr(clamp_policy, "as_dict"):
        return dict(clamp_policy.as_dict())
    return dict(clamp_policy or {})


def _output_start_ts(policy_payload: Mapping[str, Any]) -> int:
    numeric_start = dict(policy_payload.get("numeric_clamp_start") or {})
    year = int(numeric_start.get("year") or CLAMP_START_YEAR)
    month = int(numeric_start.get("month") or CLAMP_START_MONTH)
    numeric_start_ts = int(datetime(year, month, 1, tzinfo=timezone.utc).timestamp())
    return max(int(production_start_ts()), numeric_start_ts)


def _optional_ts(value: Any, *, field_name: str) -> tuple[int | None, str | None]:
    if value in (None, ""):
        return None, None
    if isinstance(value, bool):
        return None, f"{field_name}_invalid_for_clamp_range"
    try:
        return int(float(value)), None
    except Exception:
        pass
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp()), None
    except Exception:
        return None, f"{field_name}_invalid_for_clamp_range"


def _positive_int(value: Any, *, field_name: str) -> tuple[int | None, str | None]:
    if value in (None, ""):
        return None, f"{field_name}_missing_for_clamp_history"
    try:
        parsed = int(value)
    except Exception:
        return None, f"{field_name}_invalid_for_clamp_history"
    if parsed <= 0:
        return None, f"{field_name}_invalid_for_clamp_history"
    return parsed, None


def _add_months_to_ts(ts: int, months: int) -> int:
    dt = datetime.fromtimestamp(int(ts), tz=timezone.utc)
    month_index = (dt.year * 12) + dt.month - 1 + int(months)
    year = month_index // 12
    month = (month_index % 12) + 1
    day = min(dt.day, calendar.monthrange(year, month)[1])
    shifted = dt.replace(year=year, month=month, day=day)
    return int(shifted.timestamp())


def _monthly_timestamps(start_ts: int, end_ts: int) -> tuple[str, ...]:
    out: list[int] = []
    idx = 0
    while True:
        ts = _add_months_to_ts(start_ts, idx)
        if ts > end_ts:
            break
        out.append(ts)
        idx += 1
    if not out or out[-1] != int(end_ts):
        out.append(int(end_ts))
    return tuple(_iso_from_ts(ts) for ts in out)


def _fixed_day_timestamps(start_ts: int, end_ts: int, *, days: int) -> tuple[str, ...]:
    start = datetime.fromtimestamp(int(start_ts), tz=timezone.utc)
    end = datetime.fromtimestamp(int(end_ts), tz=timezone.utc)
    out: list[str] = []
    cursor = start
    step = timedelta(days=max(1, int(days)))
    while cursor <= end:
        out.append(cursor.isoformat().replace("+00:00", "Z"))
        cursor = cursor + step
    end_text = end.isoformat().replace("+00:00", "Z")
    if not out or out[-1] != end_text:
        out.append(end_text)
    return tuple(out)


def _iso_from_ts(ts: int | None) -> str | None:
    if ts is None:
        return None
    return datetime.fromtimestamp(int(ts), tz=timezone.utc).isoformat().replace("+00:00", "Z")


def _branch_name(value: object) -> str:
    text = _text(value, field_name="branch")
    aliases = {
        "asset": REGIME_BRANCH_ASSET_STATE,
        "asset-state": REGIME_BRANCH_ASSET_STATE,
        "asset_state_production": REGIME_BRANCH_ASSET_STATE,
        "market": REGIME_BRANCH_MARKET_STATE,
        "market-state": REGIME_BRANCH_MARKET_STATE,
        "market_state_production": REGIME_BRANCH_MARKET_STATE,
        "cross_asset": REGIME_BRANCH_CROSS_ASSET_STATE,
        "cross-asset-state": REGIME_BRANCH_CROSS_ASSET_STATE,
        "cross_asset_state_production": REGIME_BRANCH_CROSS_ASSET_STATE,
    }
    resolved = aliases.get(text, text)
    if resolved not in REGIME_PRODUCTION_BRANCHES:
        raise ValueError(f"Unsupported Regime Production branch: {value!r}")
    return resolved


def _text(value: object, *, field_name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"Regime Production clamp contract {field_name} must be non-empty")
    return text


__all__ = [
    "CLAMP_RANGE_STATUS_AUDITABLE_UNAVAILABLE",
    "CLAMP_RANGE_STATUS_BLOCKED",
    "CLAMP_RANGE_STATUS_READY",
    "NUMERIC_CLAMP_POLICY_SOURCE_MODULES",
    "REGIME_PRODUCTION_CLAMP_RANGE_ARTIFACT_KIND",
    "REGIME_PRODUCTION_CLAMP_RANGE_SCHEMA_VERSION",
    "REGIME_PRODUCTION_RELATIONSHIP_HISTORY_CHECK_ARTIFACT_KIND",
    "RELATIONSHIP_HISTORY_STATUS_AVAILABLE",
    "RELATIONSHIP_HISTORY_STATUS_UNAVAILABLE",
    "RegimeProductionNormalizedClampRange",
    "checkpoint_count_for_cadence",
    "checkpoint_timestamps_for_clamp_policy",
    "clamp_policy_window_summary",
    "materialized_timestamps_for_clamp_policy",
    "normalized_regime_production_clamp_range",
    "relationship_input_history_check",
]
