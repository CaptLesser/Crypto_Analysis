from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from src.regimes.core.serialization import to_jsonable
from src.regimes.market_state.axis_contracts import (
    AXIS_PANEL_UNAVAILABLE_EMPTY,
    AXIS_PANEL_UNAVAILABLE_FAMILY_MISSING,
    AXIS_PANEL_UNAVAILABLE_REQUIRED_MISSING,
    MARKET_STATE_V1_AXIS_IDS,
    MarketStateAxisContract,
    default_market_state_v1_axis_contracts,
)


MARKET_STATE_AXIS_PANEL_SCHEMA_ID = "market_state_v1_axis_panel_schema"
MARKET_STATE_AXIS_PANEL_VERSION = "market_state_v1_axis_panel_v1"
MARKET_STATE_AXIS_PANEL_STATUS_READY = "ready"
MARKET_STATE_AXIS_PANEL_STATUS_UNAVAILABLE = "unavailable"

IDENTITY_COLUMNS: tuple[str, ...] = (
    "ts",
    "interval",
    "band",
    "known_at_ts",
    "source_tail_ts",
    "schema_version",
    "universe_snapshot_id",
    "universe_snapshot_hash",
)


@dataclass(frozen=True)
class MarketStateAxisPanelConfig:
    rolling_zscore_window: int = 20
    rolling_percentile_window: int = 20
    include_derived_views: bool = True
    include_ordinal_candidates: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "rolling_zscore_window", max(2, int(self.rolling_zscore_window)))
        object.__setattr__(self, "rolling_percentile_window", max(2, int(self.rolling_percentile_window)))
        object.__setattr__(self, "include_derived_views", bool(self.include_derived_views))
        object.__setattr__(self, "include_ordinal_candidates", bool(self.include_ordinal_candidates))

    def as_dict(self) -> dict[str, Any]:
        return {
            "rolling_zscore_window": int(self.rolling_zscore_window),
            "rolling_percentile_window": int(self.rolling_percentile_window),
            "include_derived_views": bool(self.include_derived_views),
            "include_ordinal_candidates": bool(self.include_ordinal_candidates),
        }


@dataclass
class MarketStateAxisPanelAssemblyResult:
    status: str
    axis_panels: Mapping[str, pd.DataFrame] = field(default_factory=dict, repr=False)
    axis_status: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    contracts: Mapping[str, MarketStateAxisContract] = field(default_factory=dict)
    config: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def produced_axes(self) -> tuple[str, ...]:
        return tuple(axis for axis, frame in self.axis_panels.items() if not frame.empty)

    @property
    def axis_panel_count(self) -> int:
        return len(self.axis_panels)

    def as_dict(self) -> dict[str, Any]:
        return {
            "artifact_kind": "market_state_axis_panel_assembly_result",
            "status": self.status,
            "axis_panel_count": int(self.axis_panel_count),
            "produced_axes": list(self.produced_axes),
            "axis_status": {axis: dict(status) for axis, status in self.axis_status.items()},
            "contracts": {axis: contract.as_dict() for axis, contract in self.contracts.items()},
            "config": dict(self.config),
            "metadata": to_jsonable(dict(self.metadata)),
            "schema_id": MARKET_STATE_AXIS_PANEL_SCHEMA_ID,
            "production_labels_written": False,
            "production_parquet_written": False,
            "composite_market_state_label_produced": False,
        }


def assemble_market_state_v1_axis_panels(
    feature_rows: Mapping[str, pd.DataFrame] | Sequence[pd.DataFrame],
    *,
    contracts: Mapping[str, MarketStateAxisContract] | None = None,
    config: MarketStateAxisPanelConfig | None = None,
) -> MarketStateAxisPanelAssemblyResult:
    cfg = config or MarketStateAxisPanelConfig()
    axis_contracts = dict(contracts or default_market_state_v1_axis_contracts())
    by_family = _normalize_feature_rows(feature_rows)
    panels: dict[str, pd.DataFrame] = {}
    status: dict[str, Mapping[str, Any]] = {}
    for axis in MARKET_STATE_V1_AXIS_IDS:
        contract = axis_contracts[axis]
        panel, axis_status = _assemble_axis_panel(contract, by_family, cfg)
        panels[axis] = panel
        status[axis] = axis_status
    overall = MARKET_STATE_AXIS_PANEL_STATUS_READY if any(item["status"] == MARKET_STATE_AXIS_PANEL_STATUS_READY for item in status.values()) else MARKET_STATE_AXIS_PANEL_STATUS_UNAVAILABLE
    return MarketStateAxisPanelAssemblyResult(
        status=overall,
        axis_panels=panels,
        axis_status=status,
        contracts=axis_contracts,
        config=cfg.as_dict(),
        metadata={
            "candidate_ordinal_states_are_final_labels": False,
            "composite_market_state_label_produced": False,
            "monolithic_market_state_label_column": None,
            "production_writes_enabled": False,
        },
    )


def _assemble_axis_panel(
    contract: MarketStateAxisContract,
    rows_by_family: Mapping[str, pd.DataFrame],
    config: MarketStateAxisPanelConfig,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    family_frames = [rows_by_family.get(family, pd.DataFrame()).copy() for family in contract.source_feature_families]
    family_frames = [frame for frame in family_frames if not frame.empty]
    if not family_frames:
        return _unavailable_panel(contract, reason=AXIS_PANEL_UNAVAILABLE_FAMILY_MISSING), _axis_status(contract, AXIS_PANEL_UNAVAILABLE_FAMILY_MISSING)

    merged = _merge_family_frames(family_frames)
    if "band" in merged.columns:
        merged = merged.loc[merged["band"].astype(str).str.lower().isin(contract.compatible_bands)].copy()
    if merged.empty:
        return _unavailable_panel(contract, reason=AXIS_PANEL_UNAVAILABLE_EMPTY), _axis_status(contract, AXIS_PANEL_UNAVAILABLE_EMPTY)

    missing_required = [feature for feature in contract.required_features if feature not in merged.columns]
    if missing_required:
        return _unavailable_panel(contract, reason=AXIS_PANEL_UNAVAILABLE_REQUIRED_MISSING), _axis_status(
            contract,
            AXIS_PANEL_UNAVAILABLE_REQUIRED_MISSING,
            missing_required=missing_required,
        )

    for column in contract.optional_features:
        if column not in merged.columns:
            merged[column] = np.nan
    panel = _base_panel(merged, contract)
    if config.include_derived_views:
        panel = _add_derived_views(panel, contract, config)
    if config.include_ordinal_candidates and contract.ordinal_feature:
        panel = _add_candidate_ordinal_state(panel, contract)
    panel["candidate_ordinal_state_is_final_label"] = False
    panel["composite_market_state_label_produced"] = False
    return panel, {
        "status": MARKET_STATE_AXIS_PANEL_STATUS_READY,
        "axis_id": contract.axis_id,
        "row_count": int(panel.shape[0]),
        "source_feature_families": list(contract.source_feature_families),
        "required_features_present": list(contract.required_features),
        "optional_features_present": [column for column in contract.optional_features if column in merged.columns],
        "compatible_bands": list(contract.compatible_bands),
        "candidate_ordinal_states_are_final_labels": False,
    }


def _base_panel(merged: pd.DataFrame, contract: MarketStateAxisContract) -> pd.DataFrame:
    out = pd.DataFrame()
    out["ts"] = pd.to_numeric(merged["ts"], errors="coerce").astype("int64")
    out["axis"] = contract.axis_id
    for column in IDENTITY_COLUMNS:
        if column == "ts":
            continue
        out[column] = merged[column] if column in merged.columns else None
    out["feature_schema_id"] = MARKET_STATE_AXIS_PANEL_SCHEMA_ID
    out["axis_panel_version"] = MARKET_STATE_AXIS_PANEL_VERSION
    out["source_feature_families"] = ",".join(contract.source_feature_families)
    out["required_feature_count"] = len(contract.required_features)
    out["optional_feature_count"] = len(contract.optional_features)
    lineage_columns = [column for column in merged.columns if column == "lineage_id" or column.endswith("__lineage_id")]
    out["lineage_id"] = _join_lineage_ids(merged, lineage_columns)
    out["known_at"] = merged["known_at"] if "known_at" in merged.columns else None
    for feature in contract.feature_columns:
        out[feature] = pd.to_numeric(merged.get(feature), errors="coerce")
    for column in _coverage_metadata_columns(merged):
        out[column] = merged[column]
    return out.sort_values(["band", "ts"], na_position="last").reset_index(drop=True)


def _add_derived_views(
    panel: pd.DataFrame,
    contract: MarketStateAxisContract,
    config: MarketStateAxisPanelConfig,
) -> pd.DataFrame:
    out = panel.copy()
    for feature in contract.feature_columns:
        if feature not in out.columns:
            continue
        values = pd.to_numeric(out[feature], errors="coerce")
        out[f"{feature}_rolling_zscore"] = (
            values.groupby(out["band"], dropna=False)
            .transform(lambda series: _rolling_zscore(series, window=config.rolling_zscore_window))
            .to_numpy()
        )
        out[f"{feature}_rolling_percentile"] = (
            values.groupby(out["band"], dropna=False)
            .transform(lambda series: _rolling_percentile(series, window=config.rolling_percentile_window))
            .to_numpy()
        )
    return out


def _add_candidate_ordinal_state(panel: pd.DataFrame, contract: MarketStateAxisContract) -> pd.DataFrame:
    out = panel.copy()
    feature = str(contract.ordinal_feature)
    percentile_col = f"{feature}_rolling_percentile"
    values = pd.to_numeric(out[percentile_col] if percentile_col in out.columns else out[feature], errors="coerce")
    low_label, neutral_label, high_label = contract.ordinal_labels
    if contract.ordinal_direction == "lower_is_higher_state":
        low_label, high_label = high_label, low_label
    states = pd.Series(neutral_label, index=out.index, dtype=object)
    states.loc[values < 1.0 / 3.0] = low_label
    states.loc[values > 2.0 / 3.0] = high_label
    states.loc[values.isna()] = "unavailable"
    out["axis_candidate_state"] = states
    out["axis_candidate_state_source_feature"] = feature
    out["axis_candidate_state_policy"] = "rolling_percentile_tercile_descriptive_candidate"
    return out


def _merge_family_frames(frames: Sequence[pd.DataFrame]) -> pd.DataFrame:
    prepared: list[pd.DataFrame] = []
    for idx, frame in enumerate(frames):
        current = frame.copy()
        if "ts" not in current.columns:
            continue
        for column in ("interval", "band", "known_at_ts", "source_tail_ts", "schema_version", "universe_snapshot_id", "universe_snapshot_hash"):
            if column not in current.columns:
                current[column] = None
        suffix = "" if idx == 0 else f"__family{idx}"
        if suffix:
            rename = {
                column: f"{column}{suffix}"
                for column in current.columns
                if column not in ("ts", "interval", "band", "known_at_ts", "source_tail_ts", "schema_version", "universe_snapshot_id", "universe_snapshot_hash")
            }
            current = current.rename(columns=rename)
        prepared.append(current)
    if not prepared:
        return pd.DataFrame()
    merged = prepared[0]
    keys = ["ts", "interval", "band"]
    for frame in prepared[1:]:
        merged = merged.merge(frame, on=keys, how="outer", suffixes=("", "__dup"))
        for column in list(merged.columns):
            if column.endswith("__dup"):
                base = column[:-5]
                if base not in merged.columns:
                    merged = merged.rename(columns={column: base})
                else:
                    merged = merged.drop(columns=[column])
    return merged.sort_values(keys).reset_index(drop=True)


def _coverage_metadata_columns(frame: pd.DataFrame) -> tuple[str, ...]:
    return tuple(
        str(column)
        for column in frame.columns
        if any(
            str(column).endswith(suffix)
            for suffix in (
                "_requested_n",
                "_active_n",
                "_coverage_ratio",
                "_min_active_n",
                "_min_coverage_ratio",
                "_coverage_pass",
                "_late_entry_count",
                "_warmup_excluded_count",
                "_stale_or_no_trade_excluded_count",
            )
        )
    )


def _normalize_feature_rows(feature_rows: Mapping[str, pd.DataFrame] | Sequence[pd.DataFrame]) -> dict[str, pd.DataFrame]:
    if isinstance(feature_rows, Mapping):
        return {str(family): frame.copy() for family, frame in feature_rows.items() if isinstance(frame, pd.DataFrame)}
    out: dict[str, list[pd.DataFrame]] = {}
    for frame in feature_rows:
        if not isinstance(frame, pd.DataFrame) or frame.empty or "feature_family_id" not in frame.columns:
            continue
        for family, family_frame in frame.groupby(frame["feature_family_id"].astype(str), dropna=False):
            out.setdefault(str(family), []).append(family_frame.copy())
    return {family: pd.concat(frames, ignore_index=True) for family, frames in out.items()}


def _unavailable_panel(contract: MarketStateAxisContract, *, reason: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "axis": [contract.axis_id],
            "axis_panel_status": [MARKET_STATE_AXIS_PANEL_STATUS_UNAVAILABLE],
            "unavailable_reason": [reason],
            "feature_schema_id": [MARKET_STATE_AXIS_PANEL_SCHEMA_ID],
            "axis_panel_version": [MARKET_STATE_AXIS_PANEL_VERSION],
            "candidate_ordinal_state_is_final_label": [False],
            "composite_market_state_label_produced": [False],
        }
    )


def _axis_status(
    contract: MarketStateAxisContract,
    reason: str,
    *,
    missing_required: Sequence[str] = (),
) -> dict[str, Any]:
    return {
        "status": MARKET_STATE_AXIS_PANEL_STATUS_UNAVAILABLE,
        "axis_id": contract.axis_id,
        "reason": reason,
        "missing_required_features": list(missing_required),
        "source_feature_families": list(contract.source_feature_families),
        "unavailable_behavior": contract.unavailable_behavior,
    }


def _join_lineage_ids(frame: pd.DataFrame, lineage_columns: Sequence[str]) -> pd.Series:
    if not lineage_columns:
        return pd.Series([None] * len(frame), index=frame.index)
    return frame[list(lineage_columns)].astype(object).apply(
        lambda row: ",".join(dict.fromkeys(str(value) for value in row if pd.notna(value) and str(value).strip())),
        axis=1,
    )


def _rolling_zscore(series: pd.Series, *, window: int) -> pd.Series:
    mean = series.rolling(window=window, min_periods=window).mean()
    std = series.rolling(window=window, min_periods=window).std(ddof=0)
    return ((series - mean) / std.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)


def _rolling_percentile(series: pd.Series, *, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=1).apply(_last_percentile_rank, raw=False)


def _last_percentile_rank(values: pd.Series) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        return np.nan
    last = clean.iloc[-1]
    return float((clean <= last).mean())


__all__ = [
    "IDENTITY_COLUMNS",
    "MARKET_STATE_AXIS_PANEL_SCHEMA_ID",
    "MARKET_STATE_AXIS_PANEL_STATUS_READY",
    "MARKET_STATE_AXIS_PANEL_STATUS_UNAVAILABLE",
    "MARKET_STATE_AXIS_PANEL_VERSION",
    "MarketStateAxisPanelAssemblyResult",
    "MarketStateAxisPanelConfig",
    "assemble_market_state_v1_axis_panels",
]
