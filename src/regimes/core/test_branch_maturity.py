from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


METHOD_STATUS_SELECTABLE_PARTIAL = "selectable_partial"
METHOD_STATUS_DIAGNOSTIC_ONLY_RECOMMENDED = "diagnostic_only_recommended"
METHOD_STATUS_FALLBACK_ONLY = "fallback_only"
METHOD_STATUS_DIAGNOSTIC_ONLY_NOT_RUN = "diagnostic_only_not_run"

FILTER_INSUFFICIENT_ROWS = "insufficient_rows"
FILTER_INSUFFICIENT_VALID_UNMASKED_ROWS = "insufficient_valid_unmasked_rows"
FILTER_LOW_FEATURE_SPREAD = "low_feature_spread"
FILTER_ALL_MASKED = "all_masked"
FILTER_DOMINANT_STATE_COLLAPSE = "dominant_state_collapse"
FILTER_TINY_STATE_ONLY = "tiny_state_only"
FILTER_MISSING_RELATIONSHIP_CONTEXT = "missing_relationship_context"
FILTER_ASSET_NOT_IN_RELATIONSHIP_SNAPSHOT = "asset_not_in_relationship_snapshot"
FILTER_NO_VIABLE_RELATIONSHIP_VALUES = "no_viable_relationship_values"
FILTER_FAMILY_DIAGNOSTIC_ONLY = "family_diagnostic_only"
FILTER_PROFILE_TYPE_NOT_SELECTION_ELIGIBLE = "profile_type_not_selection_eligible"
FILTER_ECONOMIC_PANEL_MISSING = "economic_panel_missing"

TEST_BRANCH_FILTER_REASONS: tuple[str, ...] = (
    FILTER_INSUFFICIENT_ROWS,
    FILTER_INSUFFICIENT_VALID_UNMASKED_ROWS,
    FILTER_LOW_FEATURE_SPREAD,
    FILTER_ALL_MASKED,
    FILTER_DOMINANT_STATE_COLLAPSE,
    FILTER_TINY_STATE_ONLY,
    FILTER_MISSING_RELATIONSHIP_CONTEXT,
    FILTER_ASSET_NOT_IN_RELATIONSHIP_SNAPSHOT,
    FILTER_NO_VIABLE_RELATIONSHIP_VALUES,
    FILTER_FAMILY_DIAGNOSTIC_ONLY,
    FILTER_PROFILE_TYPE_NOT_SELECTION_ELIGIBLE,
    FILTER_ECONOMIC_PANEL_MISSING,
)


@dataclass(frozen=True)
class FeatureSpreadDiagnostics:
    feature_columns: Mapping[str, Mapping[str, float | None]]
    family_score_span: float
    family_score_iqr: float
    low_spread_warning: bool
    saturated_feature_warnings: Mapping[str, bool]
    warning_reasons: tuple[str, ...]
    family_transform_id: str | None = None

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "feature_columns": {
                str(column): dict(values)
                for column, values in self.feature_columns.items()
            },
            "family_score_span": round(float(self.family_score_span), 6),
            "family_score_iqr": round(float(self.family_score_iqr), 6),
            "low_spread_warning": bool(self.low_spread_warning),
            "saturated_feature_warnings": dict(self.saturated_feature_warnings),
            "warning_reasons": tuple(self.warning_reasons),
        }
        for column, flagged in self.saturated_feature_warnings.items():
            out[f"saturated_{column}_warning"] = bool(flagged)
        if self.family_transform_id is not None:
            out["family_transform_id"] = self.family_transform_id
        return out


def candidate_selection_status(*, health_passed: bool, selection_eligible: bool, diagnostic_only: bool) -> str:
    if not health_passed:
        return "candidate_failed_health"
    if diagnostic_only or not selection_eligible:
        return "candidate_diagnostic_only"
    return "candidate_ran"


def normalize_test_branch_filter_reason(reason: str | None) -> str | None:
    cleaned = str(reason or "").strip()
    if not cleaned:
        return None
    if cleaned in TEST_BRANCH_FILTER_REASONS:
        return cleaned
    aliases = {
        "relationship_concentration_entropy_low_raw_spread_guard": FILTER_LOW_FEATURE_SPREAD,
        "low_raw_spread_guard": FILTER_LOW_FEATURE_SPREAD,
        "birch_diagnostic_only_until_tuned_healthier": FILTER_PROFILE_TYPE_NOT_SELECTION_ELIGIBLE,
        "no_candidate_passed_output_health": FILTER_DOMINANT_STATE_COLLAPSE,
        "missing_feature_rows": FILTER_INSUFFICIENT_ROWS,
        "missing_required_field": FILTER_NO_VIABLE_RELATIONSHIP_VALUES,
        "insufficient_window_history": FILTER_INSUFFICIENT_ROWS,
    }
    return aliases.get(cleaned, cleaned)


def feature_spread_diagnostics(
    data: Any,
    *,
    score_values: Any,
    family_transform_id: str | None = None,
    raw_low_spread_columns: Sequence[str] = (),
    raw_span_epsilon: float = 0.005,
    raw_iqr_epsilon: float = 0.001,
    score_span_epsilon: float = 0.03,
    score_iqr_epsilon: float = 0.01,
    saturated_columns: Sequence[str] = (),
    saturation_abs_median_min: float = 0.85,
    saturation_iqr_max: float = 0.05,
) -> FeatureSpreadDiagnostics:
    pd = _pandas()
    columns: dict[str, dict[str, float | None]] = {}
    for column in getattr(data, "columns", ()):
        series = pd.to_numeric(data[column], errors="coerce").dropna()
        if series.empty:
            columns[str(column)] = {"min": None, "median": None, "max": None, "iqr": None, "span": None}
            continue
        q25 = float(series.quantile(0.25))
        q75 = float(series.quantile(0.75))
        columns[str(column)] = {
            "min": round(float(series.min()), 6),
            "median": round(float(series.median()), 6),
            "max": round(float(series.max()), 6),
            "iqr": round(q75 - q25, 6),
            "span": round(float(series.max() - series.min()), 6),
        }

    score = pd.to_numeric(score_values, errors="coerce").dropna()
    score_span = float(score.max() - score.min()) if len(score) else 0.0
    score_iqr = float(score.quantile(0.75) - score.quantile(0.25)) if len(score) else 0.0
    low_spread = bool(score_span < score_span_epsilon or score_iqr < score_iqr_epsilon)

    if raw_low_spread_columns:
        raw_stats = [columns.get(str(column), {}) for column in raw_low_spread_columns]
        if raw_stats and all(
            float(stats.get("span") or 0.0) < raw_span_epsilon
            and float(stats.get("iqr") or 0.0) < raw_iqr_epsilon
            for stats in raw_stats
        ):
            low_spread = True

    saturated_warnings: dict[str, bool] = {}
    for column in saturated_columns:
        if column not in getattr(data, "columns", ()):
            saturated_warnings[str(column)] = False
            continue
        series = pd.to_numeric(data[column], errors="coerce").dropna().abs()
        if series.empty:
            saturated_warnings[str(column)] = False
            continue
        iqr = float(series.quantile(0.75) - series.quantile(0.25))
        saturated_warnings[str(column)] = bool(float(series.median()) >= saturation_abs_median_min and iqr <= saturation_iqr_max)

    warning_reasons: list[str] = []
    if low_spread:
        warning_reasons.append(FILTER_LOW_FEATURE_SPREAD)
    warning_reasons.extend(
        f"saturated_{column}"
        for column, flagged in saturated_warnings.items()
        if flagged
    )
    return FeatureSpreadDiagnostics(
        feature_columns=columns,
        family_score_span=score_span,
        family_score_iqr=score_iqr,
        low_spread_warning=low_spread,
        saturated_feature_warnings=saturated_warnings,
        warning_reasons=tuple(dict.fromkeys(warning_reasons)),
        family_transform_id=family_transform_id,
    )


def filter_reason_record(
    *,
    reason_code: str,
    reason: str | None = None,
    source: str | None = None,
    profile_type: str | None = None,
    relationship_feature_family: str | None = None,
    selection_eligible: bool | None = None,
) -> dict[str, Any]:
    return {
        "reason_code": normalize_test_branch_filter_reason(reason_code),
        "reason": reason or reason_code,
        "source": source,
        "profile_type": profile_type,
        "relationship_feature_family": relationship_feature_family,
        "selection_eligible": selection_eligible,
    }


def _pandas() -> Any:
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("Regime Test Branch maturity diagnostics require pandas") from exc
    return pd


__all__ = [
    "FILTER_ALL_MASKED",
    "FILTER_ASSET_NOT_IN_RELATIONSHIP_SNAPSHOT",
    "FILTER_DOMINANT_STATE_COLLAPSE",
    "FILTER_ECONOMIC_PANEL_MISSING",
    "FILTER_FAMILY_DIAGNOSTIC_ONLY",
    "FILTER_INSUFFICIENT_ROWS",
    "FILTER_INSUFFICIENT_VALID_UNMASKED_ROWS",
    "FILTER_LOW_FEATURE_SPREAD",
    "FILTER_MISSING_RELATIONSHIP_CONTEXT",
    "FILTER_NO_VIABLE_RELATIONSHIP_VALUES",
    "FILTER_PROFILE_TYPE_NOT_SELECTION_ELIGIBLE",
    "FILTER_TINY_STATE_ONLY",
    "METHOD_STATUS_DIAGNOSTIC_ONLY_NOT_RUN",
    "METHOD_STATUS_DIAGNOSTIC_ONLY_RECOMMENDED",
    "METHOD_STATUS_FALLBACK_ONLY",
    "METHOD_STATUS_SELECTABLE_PARTIAL",
    "TEST_BRANCH_FILTER_REASONS",
    "FeatureSpreadDiagnostics",
    "candidate_selection_status",
    "feature_spread_diagnostics",
    "filter_reason_record",
    "normalize_test_branch_filter_reason",
]
