from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from src.features.scalar_features import SCALAR_FEATURE_COLUMNS


RAW_SOURCE_COLUMNS: tuple[str, ...] = ("open", "high", "low", "close", "volume", "trades")
KEY_COLUMNS: tuple[str, ...] = ("ts", "asset")
TARGET_HISTORY_COLUMNS: tuple[str, ...] = ("target_history",)
STRUCTURAL_TOKEN_PREFIXES: tuple[str, ...] = (
    "level_",
    "trend_",
    "seasonal",
    "seasonality_",
    "lag_",
)
STRUCTURAL_TOKEN_NAMES: set[str] = {
    "market_factor",
    "hour_of_day",
    "day_of_week",
    "short_term_trend",
    "volatility_regime",
    "max_drawdown",
    "max_runup",
    "range_efficiency",
    "realized_vol",
    "true_range",
}
MODEL_OUTPUT_PREFIXES: tuple[str, ...] = (
    "pred_",
    "prediction_",
    "forecast_",
    "yhat",
)
MODEL_OUTPUT_SUFFIXES: tuple[str, ...] = (
    "_yhat",
    "_pred",
    "_prediction",
)
VALIDATION_TARGET_TOKENS: tuple[str, ...] = (
    "actual",
    "error",
    "squared_error",
    "abs_error",
    "mae",
    "rmse",
    "pnl",
    "economic",
    "validation",
)

LEGACY_DYNAMIC_FEATURE_ALIASES: Dict[str, tuple[str, ...]] = {
    "macd": ("macd_12_26_9",),
    "macd_signal": ("macd_signal_12_26_9",),
    "macd_hist": ("macd_hist_12_26_9",),
    "ema_gap_12_26": ("macd_12_26_9",),
    "zscore_30": ("zscore_20",),
    "zscore_60": ("zscore_20",),
    "ret_std_14": ("ret_std_20",),
    "ret_std_30": ("ret_std_20",),
    "ret_std_60": ("ret_std_20",),
    "ret_std_120": ("ret_std_20",),
    "atr_30": ("atr_14", "atr_pct_14"),
    "range_efficiency_30": ("range_efficiency_20", "range_efficiency_50", "range_efficiency_100"),
    "volume_zscore_30": ("volume_zscore_20",),
}


@dataclass(frozen=True)
class CandidateUniverse:
    candidate_columns: tuple[str, ...]
    records: tuple[Dict[str, Any], ...]
    raw_source_candidates: tuple[str, ...]
    scalar_derived_candidates: tuple[str, ...]
    target_history_candidates: tuple[str, ...]
    structural_candidates: tuple[str, ...]
    excluded_candidates: tuple[Dict[str, Any], ...]
    stale_or_missing_candidates: tuple[Dict[str, Any], ...]
    alias_resolutions: tuple[Dict[str, str], ...]

    def to_artifact(self) -> Dict[str, Any]:
        return {
            "candidate_count": int(len(self.candidate_columns)),
            "candidate_columns": list(self.candidate_columns),
            "records": [dict(record) for record in self.records],
            "raw_source_candidates": list(self.raw_source_candidates),
            "scalar_derived_candidates": list(self.scalar_derived_candidates),
            "target_history_candidates": list(self.target_history_candidates),
            "structural_candidates": list(self.structural_candidates),
            "excluded_candidates": [dict(record) for record in self.excluded_candidates],
            "stale_or_missing_candidates": [dict(record) for record in self.stale_or_missing_candidates],
            "alias_resolutions": [dict(record) for record in self.alias_resolutions],
        }


def _dedupe(values: Iterable[str]) -> List[str]:
    out: List[str] = []
    seen: set[str] = set()
    for value in values:
        name = str(value).strip()
        if not name or name in seen:
            continue
        seen.add(name)
        out.append(name)
    return out


def _is_structural_token(name: str) -> bool:
    raw = str(name).strip()
    return raw in STRUCTURAL_TOKEN_NAMES or any(raw.startswith(prefix) for prefix in STRUCTURAL_TOKEN_PREFIXES)


def exclusion_reason(name: str) -> Optional[str]:
    raw = str(name).strip()
    low = raw.lower()
    if raw in KEY_COLUMNS:
        return "key_column"
    if low.startswith("future_"):
        return "future_label"
    if low.startswith("regime_") or "_regime_" in low or low.endswith("_regime"):
        return "downstream_regime_artifact"
    if low.endswith("_label") or "_label_" in low:
        return "label_column"
    if low.startswith(MODEL_OUTPUT_PREFIXES) or low.endswith(MODEL_OUTPUT_SUFFIXES):
        return "model_output_column"
    if any(token in low for token in VALIDATION_TARGET_TOKENS):
        return "validation_or_economic_target"
    return None


def is_dynamic_input_column(name: str) -> bool:
    raw = str(name).strip()
    return raw in RAW_SOURCE_COLUMNS or raw in set(SCALAR_FEATURE_COLUMNS)


def _resolve_alias(name: str, available: set[str]) -> Optional[str]:
    raw = str(name).strip()
    if raw in available:
        return raw
    for alias in LEGACY_DYNAMIC_FEATURE_ALIASES.get(raw, ()):
        if alias in available:
            return alias
    return None


def _overlay_names(
    preferred_feature_names: Sequence[str] = (),
    feature_blocks: Optional[Mapping[str, Sequence[str]]] = None,
) -> List[tuple[str, str]]:
    out: List[tuple[str, str]] = []
    for name in preferred_feature_names:
        if str(name).strip():
            out.append((str(name).strip(), "preferred_seed"))
    for block_name, block_values in dict(feature_blocks or {}).items():
        for name in block_values:
            if str(name).strip():
                out.append((str(name).strip(), f"feature_block:{str(block_name)}"))
    return out


def build_stage1_candidate_universe(
    *,
    loaded_frame_columns: Optional[Sequence[str]] = None,
    scalar_feature_columns: Sequence[str] = SCALAR_FEATURE_COLUMNS,
    include_raw_source: bool = True,
    model_family: str = "",
    model_key: str = "",
    preferred_feature_names: Sequence[str] = (),
    feature_blocks: Optional[Mapping[str, Sequence[str]]] = None,
    deny_columns: Sequence[str] = (),
    allow_columns: Optional[Sequence[str]] = None,
) -> CandidateUniverse:
    loaded = {str(col).strip() for col in (loaded_frame_columns or ()) if str(col).strip()}
    scalar_manifest = set(str(col).strip() for col in scalar_feature_columns if str(col).strip())
    available_dynamic = set(scalar_manifest)
    if loaded:
        available_dynamic.update(col for col in loaded if col in scalar_manifest)
    if include_raw_source:
        if loaded:
            available_dynamic.update(col for col in RAW_SOURCE_COLUMNS if col in loaded)
        else:
            available_dynamic.update(RAW_SOURCE_COLUMNS)
    deny = {str(col).strip() for col in deny_columns if str(col).strip()}
    allow = None if allow_columns is None else {str(col).strip() for col in allow_columns if str(col).strip()}

    records: List[Dict[str, Any]] = []
    excluded: List[Dict[str, Any]] = []
    raw_source: List[str] = []
    scalar_derived: List[str] = []
    candidate_cols: List[str] = []

    base_order = [*(RAW_SOURCE_COLUMNS if include_raw_source else ()), *scalar_feature_columns]
    for name in _dedupe(base_order):
        reason = exclusion_reason(name)
        if reason:
            excluded.append({"name": name, "source_type": "excluded/leakage", "reason": reason})
            continue
        if name in deny:
            excluded.append({"name": name, "source_type": "excluded/leakage", "reason": "model_family_deny"})
            continue
        if allow is not None and name not in allow:
            excluded.append({"name": name, "source_type": "excluded/leakage", "reason": "model_family_not_allowed"})
            continue
        if name not in available_dynamic:
            continue
        source_type = "raw_source" if name in RAW_SOURCE_COLUMNS else "scalar_derived"
        record = {
            "name": name,
            "source_type": source_type,
            "model_family": str(model_family),
            "model_key": str(model_key),
            "included": True,
        }
        records.append(record)
        candidate_cols.append(name)
        if source_type == "raw_source":
            raw_source.append(name)
        else:
            scalar_derived.append(name)

    alias_resolutions: List[Dict[str, str]] = []
    missing: List[Dict[str, Any]] = []
    target_history: List[str] = []
    structural: List[str] = []
    dynamic_available = set(candidate_cols)
    alias_available = set(dynamic_available)
    for overlay_name, overlay_source in _overlay_names(preferred_feature_names, feature_blocks):
        reason = exclusion_reason(overlay_name)
        if reason:
            record = {
                "name": overlay_name,
                "source_type": "excluded/leakage",
                "reason": reason,
                "overlay_source": overlay_source,
                "included": False,
            }
            excluded.append(record)
            records.append(record)
            continue
        if overlay_name in TARGET_HISTORY_COLUMNS:
            target_history.append(overlay_name)
            records.append(
                {
                    "name": overlay_name,
                    "source_type": "target_history",
                    "overlay_source": overlay_source,
                    "included": False,
                }
            )
            continue
        if _is_structural_token(overlay_name):
            structural.append(overlay_name)
            records.append(
                {
                    "name": overlay_name,
                    "source_type": "structural",
                    "overlay_source": overlay_source,
                    "included": False,
                }
            )
            continue
        actual = _resolve_alias(overlay_name, alias_available)
        if actual is not None:
            if actual != overlay_name:
                alias_resolutions.append(
                    {
                        "requested": overlay_name,
                        "resolved": actual,
                        "overlay_source": overlay_source,
                    }
                )
            continue
        record = {
            "requested": overlay_name,
            "name": overlay_name,
            "source_type": "missing/stale",
            "overlay_source": overlay_source,
            "reason": "not_in_scalar_manifest_or_raw_source",
            "included": False,
        }
        missing.append(record)
        records.append(record)

    return CandidateUniverse(
        candidate_columns=tuple(_dedupe(candidate_cols)),
        records=tuple(records),
        raw_source_candidates=tuple(_dedupe(raw_source)),
        scalar_derived_candidates=tuple(_dedupe(scalar_derived)),
        target_history_candidates=tuple(_dedupe(target_history)),
        structural_candidates=tuple(_dedupe(structural)),
        excluded_candidates=tuple(excluded),
        stale_or_missing_candidates=tuple(missing),
        alias_resolutions=tuple(alias_resolutions),
    )
