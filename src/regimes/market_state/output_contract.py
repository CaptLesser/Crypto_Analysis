from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from src.regimes.core.artifact_refs import make_artifact_ref, validate_portable_relative_path
from src.regimes.core.serialization import dumps_json, to_jsonable
from src.regimes.market_state.assignment_quality import (
    ASSIGNMENT_STATUS_LOW_CONFIDENCE,
    ASSIGNMENT_STATUS_NOT_SELECTED,
    ASSIGNMENT_STATUS_VALID,
    MARKET_STATE_ASSIGNMENT_STATUSES,
)
from src.regimes.market_state.clusterability import (
    MARKET_STATE_CLUSTERABILITY_STATUS_AXIS_NOT_CLUSTERABLE,
    MARKET_STATE_CLUSTERABILITY_STATUS_CLUSTERABLE_CANDIDATE,
    MARKET_STATE_CLUSTERABILITY_STATUS_INSUFFICIENT_TIMESTAMPS,
    MARKET_STATE_CLUSTERABILITY_STATUS_LOW_VARIATION_MARKET_WINDOW,
    MARKET_STATE_CLUSTERABILITY_STATUSES,
)
from src.regimes.market_state.contracts import (
    MARKET_STATE_PATHWAY_VALUE,
    MARKET_STATE_SCHEMA_VERSION,
    MarketStateAxis,
    MarketStateBand,
    MarketStatePathway,
    MarketStateSchemaVersion,
    _enum_value,
    _mapping,
    _non_empty_text,
    _schema_version,
    _string_tuple,
)
from src.regimes.market_state.feature_writer import (
    MARKET_STATE_FEATURE_WRITE_FORMAT_AUTO,
    MARKET_STATE_FEATURE_WRITE_FORMAT_JSONL,
    MARKET_STATE_FEATURE_WRITE_FORMAT_PARQUET,
    validate_market_state_feature_write_root,
)
from src.regimes.market_state.taxonomy import default_market_state_taxonomy


MARKET_STATE_SANDBOX_OUTPUT_BOUNDARY = "sandbox_non_production"

LABEL_SOURCE_CLUSTERED = "clustered"
LABEL_SOURCE_LOW_VARIATION_SINGLE_STATE_FALLBACK = "low_variation_single_state_fallback"
LABEL_SOURCE_INSUFFICIENT_DATA_NO_LABEL = "insufficient_data_no_label"
LABEL_SOURCE_AXIS_NOT_APPLICABLE = "axis_not_applicable"

MARKET_STATE_LABEL_SOURCES: tuple[str, ...] = (
    LABEL_SOURCE_CLUSTERED,
    LABEL_SOURCE_LOW_VARIATION_SINGLE_STATE_FALLBACK,
    LABEL_SOURCE_INSUFFICIENT_DATA_NO_LABEL,
    LABEL_SOURCE_AXIS_NOT_APPLICABLE,
)

FALLBACK_STATUS_NO_FALLBACK_NEEDED = "no_fallback_needed"
FALLBACK_STATUS_LOW_VARIATION_SINGLE_STATE = "low_variation_single_state"
FALLBACK_STATUS_INSUFFICIENT_DATA_NO_LABEL = "insufficient_data_no_label"
FALLBACK_STATUS_AXIS_NOT_APPLICABLE = "axis_not_applicable"

MARKET_STATE_FALLBACK_STATUSES: tuple[str, ...] = (
    FALLBACK_STATUS_NO_FALLBACK_NEEDED,
    FALLBACK_STATUS_LOW_VARIATION_SINGLE_STATE,
    FALLBACK_STATUS_INSUFFICIENT_DATA_NO_LABEL,
    FALLBACK_STATUS_AXIS_NOT_APPLICABLE,
)

MARKET_STATE_OUTPUT_REQUIRED_FIELDS: tuple[str, ...] = (
    "ts",
    "known_at_ts",
    "pathway",
    "axis",
    "band",
    "interval",
    "regime_id",
    "regime_label",
    "regime_label_source",
    "state_strength",
    "market_clusterability_status",
    "fallback_status",
    "profile_id",
    "feature_profile_id",
    "feature_family_id",
    "universe_policy_id",
    "core_basket_hash",
    "broad_universe_hash",
    "clusterer_family",
    "assignment_policy",
    "refit_key",
    "source_tail_ts",
    "lineage_id",
    "schema_version",
    "lineage",
    "description_metadata",
    "created_at",
    "run_id",
    "artifact_boundary",
)

MARKET_STATE_OUTPUT_PARTITION_FIELDS: tuple[str, ...] = ("run_id", "axis", "band", "interval")


def market_state_non_production_artifact_boundary() -> dict[str, Any]:
    return {
        "classification": MARKET_STATE_SANDBOX_OUTPUT_BOUNDARY,
        "pathway": MARKET_STATE_PATHWAY_VALUE,
        "market_level_labels_only": True,
        "asset_level_labels_allowed": False,
        "relative_cross_asset_execution_allowed": False,
        "production_output": False,
        "production_parquet_allowed": False,
        "production_regime_labels_allowed": False,
        "production_profile_promotion_allowed": False,
        "write_root_policy": "reports/regimes/foundation/market_state/sandbox_outputs only",
    }


@dataclass(frozen=True)
class MarketStateSandboxOutputSchema:
    required_fields: Sequence[str] = MARKET_STATE_OUTPUT_REQUIRED_FIELDS
    partition_fields: Sequence[str] = MARKET_STATE_OUTPUT_PARTITION_FIELDS
    pathway: str | MarketStatePathway = MarketStatePathway.MARKET_STATE
    artifact_kind: str = "market_state_sandbox_output_schema"
    artifact_boundary: Mapping[str, Any] = field(default_factory=market_state_non_production_artifact_boundary)
    schema_version: int | MarketStateSchemaVersion = MARKET_STATE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        required = _string_tuple(self.required_fields, field_name="required_fields", require_non_empty=True)
        missing = sorted(set(MARKET_STATE_OUTPUT_REQUIRED_FIELDS).difference(required))
        if missing:
            raise ValueError(f"Market-state sandbox output schema missing required fields: {missing}")
        object.__setattr__(self, "schema_version", _schema_version(self.schema_version))
        object.__setattr__(self, "pathway", _enum_value(self.pathway, MarketStatePathway, field_name="pathway"))
        object.__setattr__(self, "artifact_kind", _non_empty_text(self.artifact_kind, field_name="artifact_kind"))
        object.__setattr__(self, "required_fields", required)
        object.__setattr__(self, "partition_fields", _string_tuple(self.partition_fields, field_name="partition_fields", require_non_empty=True))
        object.__setattr__(self, "artifact_boundary", _artifact_boundary(self.artifact_boundary))

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "artifact_kind": self.artifact_kind,
            "pathway": self.pathway,
            "required_fields": list(self.required_fields),
            "partition_fields": list(self.partition_fields),
            "label_sources": list(MARKET_STATE_LABEL_SOURCES),
            "fallback_statuses": list(MARKET_STATE_FALLBACK_STATUSES),
            "storage_notes": {
                "description_metadata": "JSON object in contract; writer stores as JSON text for scalar storage.",
                "artifact_boundary": "JSON object in contract; writer stores as JSON text for scalar storage.",
            },
            "artifact_boundary": to_jsonable(dict(self.artifact_boundary)),
        }

    def to_json(self, **kwargs: Any) -> str:
        return dumps_json(self.as_dict(), **kwargs)


@dataclass(frozen=True)
class MarketStateOutputRow:
    ts: int | float | str
    axis: str | MarketStateAxis
    band: str | MarketStateBand
    interval: int
    regime_id: str | int | None
    regime_label: str | None
    regime_label_source: str
    state_strength: float | None
    market_clusterability_status: str
    fallback_status: str
    profile_id: str
    feature_family_id: str
    universe_policy_id: str
    core_basket_hash: str
    broad_universe_hash: str
    clusterer_family: str
    assignment_policy: str
    refit_key: str
    known_at_ts: int | float | str | None = None
    source_tail_ts: int | float | str | None = None
    lineage_id: str | None = None
    feature_profile_id: str | None = None
    lineage: Mapping[str, Any] = field(default_factory=dict)
    description_metadata: Mapping[str, Any] = field(default_factory=dict)
    run_id: str = "market_state_sandbox_run"
    created_at: str | None = None
    artifact_boundary: Mapping[str, Any] = field(default_factory=market_state_non_production_artifact_boundary)
    pathway: str | MarketStatePathway = MarketStatePathway.MARKET_STATE
    schema_version: int | MarketStateSchemaVersion = MARKET_STATE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        pathway = _enum_value(self.pathway, MarketStatePathway, field_name="pathway")
        if pathway != MARKET_STATE_PATHWAY_VALUE:
            raise ValueError("Market-state output pathway must be market_state")
        axis = _enum_value(self.axis, MarketStateAxis, field_name="axis")
        band = _enum_value(self.band, MarketStateBand, field_name="band")
        default_market_state_taxonomy().validate_axis_band(axis, band)
        interval = int(self.interval)
        if interval <= 0:
            raise ValueError("Market-state output interval must be positive")
        label_source = _label_source(self.regime_label_source)
        clusterability_status = _clusterability_status(self.market_clusterability_status)
        fallback_status = _fallback_status(self.fallback_status)
        state_strength = _state_strength(self.state_strength)
        _validate_label_contract(
            label_source=label_source,
            regime_id=self.regime_id,
            regime_label=self.regime_label,
            state_strength=state_strength,
            clusterability_status=clusterability_status,
            fallback_status=fallback_status,
        )
        object.__setattr__(self, "schema_version", _schema_version(self.schema_version))
        object.__setattr__(self, "ts", _non_empty_text(self.ts, field_name="ts"))
        object.__setattr__(
            self,
            "known_at_ts",
            _non_empty_text(self.known_at_ts if self.known_at_ts is not None else self.ts, field_name="known_at_ts"),
        )
        source_tail = _non_empty_text(self.source_tail_ts if self.source_tail_ts is not None else self.known_at_ts or self.ts, field_name="source_tail_ts")
        if _to_orderable(source_tail, field_name="source_tail_ts") > _to_orderable(self.known_at_ts, field_name="known_at_ts"):
            raise ValueError("Market-state output source_tail_ts must not exceed known_at_ts")
        object.__setattr__(self, "source_tail_ts", source_tail)
        object.__setattr__(self, "pathway", pathway)
        object.__setattr__(self, "axis", axis)
        object.__setattr__(self, "band", band)
        object.__setattr__(self, "interval", interval)
        object.__setattr__(self, "regime_id", None if self.regime_id is None else str(self.regime_id).strip())
        object.__setattr__(self, "regime_label", None if self.regime_label is None else str(self.regime_label).strip())
        object.__setattr__(self, "regime_label_source", label_source)
        object.__setattr__(self, "state_strength", state_strength)
        object.__setattr__(self, "market_clusterability_status", clusterability_status)
        object.__setattr__(self, "fallback_status", fallback_status)
        object.__setattr__(self, "profile_id", _non_empty_text(self.profile_id, field_name="profile_id"))
        object.__setattr__(self, "feature_family_id", _non_empty_text(self.feature_family_id, field_name="feature_family_id"))
        object.__setattr__(
            self,
            "feature_profile_id",
            _optional_text(self.feature_profile_id) or _non_empty_text(self.feature_family_id, field_name="feature_family_id"),
        )
        object.__setattr__(self, "universe_policy_id", _non_empty_text(self.universe_policy_id, field_name="universe_policy_id"))
        object.__setattr__(self, "core_basket_hash", _non_empty_text(self.core_basket_hash, field_name="core_basket_hash"))
        object.__setattr__(self, "broad_universe_hash", _non_empty_text(self.broad_universe_hash, field_name="broad_universe_hash"))
        object.__setattr__(self, "clusterer_family", _non_empty_text(self.clusterer_family, field_name="clusterer_family"))
        object.__setattr__(self, "assignment_policy", _non_empty_text(self.assignment_policy, field_name="assignment_policy"))
        object.__setattr__(self, "refit_key", _non_empty_text(self.refit_key, field_name="refit_key"))
        lineage = to_jsonable(_mapping(self.lineage, field_name="lineage"))
        object.__setattr__(self, "lineage", lineage)
        object.__setattr__(self, "lineage_id", _optional_text(self.lineage_id) or _lineage_id_from_mapping(lineage) or f"{self.run_id}:{axis}:{band}:{interval}:{self.refit_key}")
        object.__setattr__(self, "description_metadata", to_jsonable(_mapping(self.description_metadata, field_name="description_metadata")))
        object.__setattr__(self, "run_id", _non_empty_text(self.run_id, field_name="run_id"))
        object.__setattr__(self, "created_at", _created_at(self.created_at))
        object.__setattr__(self, "artifact_boundary", _artifact_boundary(self.artifact_boundary))

    def as_dict(self) -> dict[str, Any]:
        return {
            "ts": self.ts,
            "known_at_ts": self.known_at_ts,
            "pathway": self.pathway,
            "axis": self.axis,
            "band": self.band,
            "interval": int(self.interval),
            "regime_id": self.regime_id,
            "regime_label": self.regime_label,
            "regime_label_source": self.regime_label_source,
            "state_strength": self.state_strength,
            "market_clusterability_status": self.market_clusterability_status,
            "fallback_status": self.fallback_status,
            "profile_id": self.profile_id,
            "feature_profile_id": self.feature_profile_id,
            "feature_family_id": self.feature_family_id,
            "universe_policy_id": self.universe_policy_id,
            "core_basket_hash": self.core_basket_hash,
            "broad_universe_hash": self.broad_universe_hash,
            "clusterer_family": self.clusterer_family,
            "assignment_policy": self.assignment_policy,
            "refit_key": self.refit_key,
            "source_tail_ts": self.source_tail_ts,
            "lineage_id": self.lineage_id,
            "schema_version": int(self.schema_version),
            "lineage": to_jsonable(dict(self.lineage)),
            "description_metadata": to_jsonable(dict(self.description_metadata)),
            "created_at": self.created_at,
            "run_id": self.run_id,
            "artifact_boundary": to_jsonable(dict(self.artifact_boundary)),
        }

    def storage_dict(self) -> dict[str, Any]:
        payload = self.as_dict()
        payload["lineage"] = dumps_json(payload["lineage"], separators=(",", ":"))
        payload["description_metadata"] = dumps_json(payload["description_metadata"], separators=(",", ":"))
        payload["artifact_boundary"] = dumps_json(payload["artifact_boundary"], separators=(",", ":"))
        return payload


def default_market_state_sandbox_output_schema() -> MarketStateSandboxOutputSchema:
    return MarketStateSandboxOutputSchema()


def coerce_market_state_output_row(row: MarketStateOutputRow | Mapping[str, Any]) -> MarketStateOutputRow:
    if isinstance(row, MarketStateOutputRow):
        return row
    if not isinstance(row, Mapping):
        raise ValueError("Market-state output rows must be mappings or MarketStateOutputRow objects")
    payload = dict(row)
    for json_field in ("lineage", "description_metadata", "artifact_boundary"):
        if isinstance(payload.get(json_field), str):
            try:
                payload[json_field] = json.loads(payload[json_field])
            except json.JSONDecodeError as exc:
                raise ValueError(f"Market-state output {json_field} must be valid JSON when stored as text") from exc
    payload.setdefault("created_at", None)
    payload.setdefault("artifact_boundary", market_state_non_production_artifact_boundary())
    return MarketStateOutputRow(**payload)


def validate_market_state_output_rows(rows: Sequence[MarketStateOutputRow | Mapping[str, Any]]) -> tuple[MarketStateOutputRow, ...]:
    if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence):
        raise ValueError("Market-state output rows must be a non-empty sequence")
    normalized = tuple(coerce_market_state_output_row(row) for row in rows)
    if not normalized:
        raise ValueError("Market-state output rows must be non-empty")
    required = set(MARKET_STATE_OUTPUT_REQUIRED_FIELDS)
    for row in normalized:
        missing = sorted(required.difference(row.as_dict()))
        if missing:
            raise ValueError(f"Market-state output row missing required fields: {missing}")
    return normalized


def clustered_market_state_output_rows(
    *,
    timestamps: Sequence[int | float | str],
    labels: Sequence[str | int],
    axis: str | MarketStateAxis,
    band: str | MarketStateBand,
    interval: int,
    profile_id: str,
    feature_family_id: str,
    universe_policy_id: str,
    core_basket_hash: str,
    broad_universe_hash: str,
    clusterer_family: str,
    assignment_policy: str,
    refit_key: str,
    strengths: Sequence[float | None] | None = None,
    regime_label_map: Mapping[str | int, str] | None = None,
    run_id: str = "market_state_sandbox_run",
    created_at: str | None = None,
    description_metadata: Mapping[str, Any] | None = None,
) -> tuple[MarketStateOutputRow, ...]:
    if len(timestamps) != len(labels):
        raise ValueError("Market-state clustered output timestamps and labels must have equal length")
    if strengths is not None and len(strengths) != len(labels):
        raise ValueError("Market-state clustered output strengths and labels must have equal length")
    mapping = {str(key): str(value) for key, value in dict(regime_label_map or {}).items()}
    return tuple(
        MarketStateOutputRow(
            ts=ts,
            axis=axis,
            band=band,
            interval=interval,
            regime_id=str(label),
            regime_label=mapping.get(str(label), f"market_state_{label}"),
            regime_label_source=LABEL_SOURCE_CLUSTERED,
            state_strength=1.0 if strengths is None else strengths[idx],
            market_clusterability_status=MARKET_STATE_CLUSTERABILITY_STATUS_CLUSTERABLE_CANDIDATE,
            fallback_status=FALLBACK_STATUS_NO_FALLBACK_NEEDED,
            profile_id=profile_id,
            feature_family_id=feature_family_id,
            universe_policy_id=universe_policy_id,
            core_basket_hash=core_basket_hash,
            broad_universe_hash=broad_universe_hash,
            clusterer_family=clusterer_family,
            assignment_policy=assignment_policy,
            refit_key=refit_key,
            known_at_ts=ts,
            source_tail_ts=ts,
            lineage_id=f"{run_id}:{axis}:{band}:{interval}:{refit_key}",
            lineage=dict(description_metadata or {}).get("lineage", {}),
            description_metadata={"row_role": "clustered_market_state_label", **dict(description_metadata or {})},
            run_id=run_id,
            created_at=created_at,
        )
        for idx, (ts, label) in enumerate(zip(timestamps, labels))
    )


def low_variation_single_state_output_rows(
    *,
    timestamps: Sequence[int | float | str],
    axis: str | MarketStateAxis,
    band: str | MarketStateBand,
    interval: int,
    profile_id: str = "market_state_low_variation_fallback_profile",
    feature_family_id: str = "market_return_summary",
    universe_policy_id: str = "market_state_universe_policy_unresolved",
    core_basket_hash: str = "unresolved_core_basket",
    broad_universe_hash: str = "unresolved_broad_universe",
    run_id: str = "market_state_sandbox_run",
    created_at: str | None = None,
    description_metadata: Mapping[str, Any] | None = None,
) -> tuple[MarketStateOutputRow, ...]:
    return tuple(
        MarketStateOutputRow(
            ts=ts,
            axis=axis,
            band=band,
            interval=interval,
            regime_id="single_market_state",
            regime_label="single_market_state",
            regime_label_source=LABEL_SOURCE_LOW_VARIATION_SINGLE_STATE_FALLBACK,
            state_strength=1.0,
            market_clusterability_status=MARKET_STATE_CLUSTERABILITY_STATUS_LOW_VARIATION_MARKET_WINDOW,
            fallback_status=FALLBACK_STATUS_LOW_VARIATION_SINGLE_STATE,
            profile_id=profile_id,
            feature_family_id=feature_family_id,
            universe_policy_id=universe_policy_id,
            core_basket_hash=core_basket_hash,
            broad_universe_hash=broad_universe_hash,
            clusterer_family="none",
            assignment_policy="fallback_constant",
            refit_key="no_refit_low_variation_fallback",
            known_at_ts=ts,
            source_tail_ts=ts,
            lineage_id=f"{run_id}:{axis}:{band}:{interval}:no_refit_low_variation_fallback",
            lineage=dict(description_metadata or {}).get("lineage", {}),
            description_metadata={"row_role": "low_variation_single_state_fallback", **dict(description_metadata or {})},
            run_id=run_id,
            created_at=created_at,
        )
        for ts in timestamps
    )


def insufficient_market_state_no_label_output_row(
    *,
    ts: int | float | str,
    axis: str | MarketStateAxis,
    band: str | MarketStateBand,
    interval: int,
    market_clusterability_status: str = MARKET_STATE_CLUSTERABILITY_STATUS_INSUFFICIENT_TIMESTAMPS,
    run_id: str = "market_state_sandbox_run",
    created_at: str | None = None,
    description_metadata: Mapping[str, Any] | None = None,
) -> MarketStateOutputRow:
    return MarketStateOutputRow(
        ts=ts,
        axis=axis,
        band=band,
        interval=interval,
        regime_id=None,
        regime_label=None,
        regime_label_source=LABEL_SOURCE_INSUFFICIENT_DATA_NO_LABEL,
        state_strength=None,
        market_clusterability_status=market_clusterability_status,
        fallback_status=FALLBACK_STATUS_INSUFFICIENT_DATA_NO_LABEL,
        profile_id="no_market_state_profile_insufficient_data",
        feature_family_id="no_market_state_feature_family_insufficient_data",
        universe_policy_id="no_universe_policy_insufficient_data",
        core_basket_hash="no_core_basket_insufficient_data",
        broad_universe_hash="no_broad_universe_insufficient_data",
        clusterer_family="none",
        assignment_policy="no_assignment_insufficient_data",
        refit_key="no_refit_insufficient_data",
        known_at_ts=ts,
        source_tail_ts=ts,
        lineage_id=f"{run_id}:{axis}:{band}:{interval}:no_refit_insufficient_data",
        lineage=dict(description_metadata or {}).get("lineage", {}),
        description_metadata={"row_role": "insufficient_market_state_no_label", **dict(description_metadata or {})},
        run_id=run_id,
        created_at=created_at,
    )


def axis_not_applicable_market_state_output_row(
    *,
    ts: int | float | str,
    axis: str | MarketStateAxis,
    band: str | MarketStateBand,
    interval: int,
    run_id: str = "market_state_sandbox_run",
    created_at: str | None = None,
    description_metadata: Mapping[str, Any] | None = None,
) -> MarketStateOutputRow:
    return MarketStateOutputRow(
        ts=ts,
        axis=axis,
        band=band,
        interval=interval,
        regime_id=None,
        regime_label=None,
        regime_label_source=LABEL_SOURCE_AXIS_NOT_APPLICABLE,
        state_strength=None,
        market_clusterability_status=MARKET_STATE_CLUSTERABILITY_STATUS_AXIS_NOT_CLUSTERABLE,
        fallback_status=FALLBACK_STATUS_AXIS_NOT_APPLICABLE,
        profile_id="no_market_state_profile_axis_not_applicable",
        feature_family_id="no_market_state_feature_family_axis_not_applicable",
        universe_policy_id="no_universe_policy_axis_not_applicable",
        core_basket_hash="no_core_basket_axis_not_applicable",
        broad_universe_hash="no_broad_universe_axis_not_applicable",
        clusterer_family="none",
        assignment_policy="no_assignment_axis_not_applicable",
        refit_key="no_refit_axis_not_applicable",
        known_at_ts=ts,
        source_tail_ts=ts,
        lineage_id=f"{run_id}:{axis}:{band}:{interval}:no_refit_axis_not_applicable",
        lineage=dict(description_metadata or {}).get("lineage", {}),
        description_metadata={"row_role": "axis_not_applicable_market_state_no_label", **dict(description_metadata or {})},
        run_id=run_id,
        created_at=created_at,
    )


MARKET_STATE_WIDE_OUTPUT_SCHEMA_VERSION = 1
MARKET_STATE_WIDE_OUTPUT_ARTIFACT_KIND = "market_state_v1_wide_output_contract"
MARKET_STATE_WIDE_OUTPUT_STATUS_WRITTEN = "written"
MARKET_STATE_WIDE_OUTPUT_STATUS_NO_ROWS = "no_rows"
MARKET_STATE_WIDE_OUTPUT_FORMAT_AUTO = MARKET_STATE_FEATURE_WRITE_FORMAT_AUTO
MARKET_STATE_WIDE_OUTPUT_FORMAT_PARQUET = MARKET_STATE_FEATURE_WRITE_FORMAT_PARQUET
MARKET_STATE_WIDE_OUTPUT_FORMAT_JSONL = MARKET_STATE_FEATURE_WRITE_FORMAT_JSONL
MARKET_STATE_WIDE_OUTPUT_SENTINEL_STATE = "not_assigned"
MARKET_STATE_WIDE_OUTPUT_SENTINEL_ID = -1
MARKET_STATE_WIDE_OUTPUT_SENTINEL_SCORE = 0.0
MARKET_STATE_WIDE_OUTPUT_SENTINEL_CONFIDENCE = 0.0

MARKET_STATE_WIDE_OUTPUT_METADATA_COLUMNS: tuple[str, ...] = (
    "ts",
    "known_at_ts",
    "source_tail_ts",
    "band",
    "interval_min",
    "schema_version",
    "profile_id",
    "profile_version",
    "universe_manifest_id",
    "feature_bundle_id",
    "lineage_id",
)

MARKET_STATE_WIDE_OUTPUT_AXES: tuple[str, ...] = (
    "market_return_state",
    "market_volatility_state",
    "market_breadth_state",
    "market_dispersion_state",
    "market_correlation_state",
    "market_liquidity_activity_state",
    "market_stress_state",
    "stable_peg_stress_state",
    "market_speculative_state",
)

MARKET_STATE_WIDE_OUTPUT_KEY_COLUMNS: tuple[str, ...] = ("ts", "band", "profile_id")
MARKET_STATE_WIDE_OUTPUT_METADATA_ALIASES: Mapping[str, str] = {
    "interval": "interval_min",
    "universe_snapshot_id": "universe_manifest_id",
}
MARKET_STATE_WIDE_OUTPUT_FORBIDDEN_COLUMNS: tuple[str, ...] = (
    "market_state",
    "asset_market_state",
    "cross_asset_peer_label",
    "l2_order_book_state",
)
_WIDE_ASSIGNED_STATUSES: frozenset[str] = frozenset({ASSIGNMENT_STATUS_VALID, ASSIGNMENT_STATUS_LOW_CONFIDENCE})


def _wide_axis(value: object) -> str:
    axis = _non_empty_text(value, field_name="axis")
    if axis not in MARKET_STATE_WIDE_OUTPUT_AXES:
        raise ValueError(f"Unsupported Market-State wide output axis {axis!r}")
    return axis


def market_state_wide_axis_columns(axis: str) -> tuple[str, ...]:
    axis_name = _wide_axis(axis)
    return (
        axis_name,
        f"{axis_name}_id",
        f"{axis_name}_score",
        f"{axis_name}_confidence",
        f"{axis_name}_status",
    )


MARKET_STATE_WIDE_OUTPUT_REQUIRED_COLUMNS: tuple[str, ...] = (
    *MARKET_STATE_WIDE_OUTPUT_METADATA_COLUMNS,
    *(column for axis in MARKET_STATE_WIDE_OUTPUT_AXES for column in market_state_wide_axis_columns(axis)),
)


def market_state_wide_non_production_artifact_boundary() -> dict[str, Any]:
    return {
        "classification": MARKET_STATE_SANDBOX_OUTPUT_BOUNDARY,
        "contract_surface": "market_state_v1_wide_output",
        "row_grain": "ts_band_profile_id",
        "production_output": False,
        "production_parquet_allowed": False,
        "production_regime_labels_allowed": False,
        "production_profile_promotion_allowed": False,
        "asset_level_labels_allowed": False,
        "relative_cross_asset_execution_allowed": False,
        "broad_all_to_all_pairwise_allowed": False,
        "l2_order_book_sidecars_allowed": False,
        "composite_market_state_label_allowed": False,
        "write_root_policy": "sandbox_test_report_scoped_only",
    }


@dataclass(frozen=True)
class MarketStateWideOutputSchema:
    metadata_columns: Sequence[str] = MARKET_STATE_WIDE_OUTPUT_METADATA_COLUMNS
    axes: Sequence[str] = MARKET_STATE_WIDE_OUTPUT_AXES
    required_columns: Sequence[str] = MARKET_STATE_WIDE_OUTPUT_REQUIRED_COLUMNS
    allowed_statuses: Sequence[str] = MARKET_STATE_ASSIGNMENT_STATUSES
    schema_version: int = MARKET_STATE_WIDE_OUTPUT_SCHEMA_VERSION
    artifact_kind: str = MARKET_STATE_WIDE_OUTPUT_ARTIFACT_KIND
    artifact_boundary: Mapping[str, Any] = field(default_factory=market_state_wide_non_production_artifact_boundary)

    def __post_init__(self) -> None:
        metadata = _string_tuple(self.metadata_columns, field_name="metadata_columns", require_non_empty=True)
        axes = tuple(_wide_axis(axis) for axis in self.axes)
        required = _string_tuple(self.required_columns, field_name="required_columns", require_non_empty=True)
        statuses = _string_tuple(self.allowed_statuses, field_name="allowed_statuses", require_non_empty=True)
        missing_metadata = sorted(set(MARKET_STATE_WIDE_OUTPUT_METADATA_COLUMNS).difference(metadata))
        missing_axes = sorted(set(MARKET_STATE_WIDE_OUTPUT_AXES).difference(axes))
        missing_required = sorted(set(MARKET_STATE_WIDE_OUTPUT_REQUIRED_COLUMNS).difference(required))
        if missing_metadata:
            raise ValueError(f"Market-State wide output schema missing metadata columns: {missing_metadata}")
        if missing_axes:
            raise ValueError(f"Market-State wide output schema missing axes: {missing_axes}")
        if missing_required:
            raise ValueError(f"Market-State wide output schema missing required columns: {missing_required}")
        if set(statuses) != set(MARKET_STATE_ASSIGNMENT_STATUSES):
            raise ValueError("Market-State wide output schema statuses must match assignment-quality vocabulary")
        object.__setattr__(self, "metadata_columns", metadata)
        object.__setattr__(self, "axes", axes)
        object.__setattr__(self, "required_columns", required)
        object.__setattr__(self, "allowed_statuses", statuses)
        object.__setattr__(self, "schema_version", int(self.schema_version))
        object.__setattr__(self, "artifact_kind", _non_empty_text(self.artifact_kind, field_name="artifact_kind"))
        object.__setattr__(self, "artifact_boundary", _artifact_boundary(self.artifact_boundary))

    @property
    def dtype_expectations(self) -> dict[str, str]:
        dtypes = {
            "ts": "timestamp_numeric",
            "known_at_ts": "timestamp_numeric",
            "source_tail_ts": "timestamp_numeric",
            "band": "string",
            "interval_min": "positive_integer",
            "schema_version": "integer",
            "profile_id": "string",
            "profile_version": "string",
            "universe_manifest_id": "string",
            "feature_bundle_id": "string",
            "lineage_id": "string",
        }
        for axis in self.axes:
            dtypes.update(
                {
                    axis: "string",
                    f"{axis}_id": "integer",
                    f"{axis}_score": "finite_float_0_1",
                    f"{axis}_confidence": "finite_float_0_1",
                    f"{axis}_status": "assignment_status",
                }
            )
        return dtypes

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "artifact_kind": self.artifact_kind,
            "row_grain": list(MARKET_STATE_WIDE_OUTPUT_KEY_COLUMNS),
            "metadata_columns": list(self.metadata_columns),
            "axes": list(self.axes),
            "required_columns": list(self.required_columns),
            "allowed_statuses": list(self.allowed_statuses),
            "sentinel_policy": {
                "state": MARKET_STATE_WIDE_OUTPUT_SENTINEL_STATE,
                "state_id": MARKET_STATE_WIDE_OUTPUT_SENTINEL_ID,
                "score": MARKET_STATE_WIDE_OUTPUT_SENTINEL_SCORE,
                "confidence": MARKET_STATE_WIDE_OUTPUT_SENTINEL_CONFIDENCE,
                "masked_statuses": [status for status in self.allowed_statuses if status not in _WIDE_ASSIGNED_STATUSES],
            },
            "metadata_aliases": dict(MARKET_STATE_WIDE_OUTPUT_METADATA_ALIASES),
            "dtype_expectations": self.dtype_expectations,
            "forbidden_columns": list(MARKET_STATE_WIDE_OUTPUT_FORBIDDEN_COLUMNS),
            "artifact_boundary": to_jsonable(dict(self.artifact_boundary)),
        }

    def to_json(self, **kwargs: Any) -> str:
        return dumps_json(self.as_dict(), **kwargs)


@dataclass(frozen=True)
class MarketStateWideWriteRequest:
    output_root: str | Path
    run_id: str
    rows: pd.DataFrame | Sequence[Mapping[str, Any]]
    file_format: str = MARKET_STATE_WIDE_OUTPUT_FORMAT_AUTO
    production_enabled: bool = False
    write_manifest: bool = True
    schema: MarketStateWideOutputSchema | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "output_root", Path(self.output_root))
        object.__setattr__(self, "run_id", _wide_safe_part(self.run_id, field_name="run_id"))
        object.__setattr__(self, "file_format", _wide_file_format(self.file_format))
        object.__setattr__(self, "production_enabled", bool(self.production_enabled))
        object.__setattr__(self, "write_manifest", bool(self.write_manifest))
        object.__setattr__(self, "schema", self.schema or default_market_state_wide_output_schema())
        object.__setattr__(self, "metadata", to_jsonable(dict(self.metadata)))


@dataclass(frozen=True)
class MarketStateWideWriteResult:
    status: str
    output_root: Path
    run_id: str
    row_count: int = 0
    written_paths: Sequence[Path] = ()
    manifest_path: Path | None = None
    schema_path: Path | None = None
    file_format_counts: Mapping[str, int] = field(default_factory=dict)
    artifact_boundary: Mapping[str, Any] = field(default_factory=market_state_wide_non_production_artifact_boundary)
    manifest: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": MARKET_STATE_WIDE_OUTPUT_SCHEMA_VERSION,
            "artifact_kind": "market_state_v1_wide_output_write_result",
            "status": self.status,
            "output_root": str(self.output_root),
            "run_id": self.run_id,
            "row_count": int(self.row_count),
            "written_paths": [str(path) for path in self.written_paths],
            "manifest_path": str(self.manifest_path) if self.manifest_path is not None else None,
            "schema_path": str(self.schema_path) if self.schema_path is not None else None,
            "file_format_counts": dict(self.file_format_counts),
            "artifact_boundary": to_jsonable(dict(self.artifact_boundary)),
            "manifest": to_jsonable(dict(self.manifest)),
            "metadata": to_jsonable(dict(self.metadata)),
            "production_enabled": False,
            "production_outputs_written": False,
            "production_profile_promotion_allowed": False,
        }


def default_market_state_wide_output_schema() -> MarketStateWideOutputSchema:
    return MarketStateWideOutputSchema()


def build_market_state_wide_output_rows(
    row_metadata: pd.DataFrame | Sequence[Mapping[str, Any]],
    axis_assignments: Mapping[str, pd.DataFrame | Sequence[Mapping[str, Any]]] | None = None,
    *,
    schema: MarketStateWideOutputSchema | None = None,
) -> pd.DataFrame:
    contract = schema or default_market_state_wide_output_schema()
    base = _wide_frame(row_metadata)
    base = _apply_wide_metadata_aliases(base)
    if "schema_version" not in base.columns:
        base["schema_version"] = int(contract.schema_version)
    missing_metadata = [column for column in contract.metadata_columns if column not in base.columns]
    if missing_metadata:
        raise ValueError(f"Market-State wide row metadata missing required columns: {missing_metadata}")
    out = base.loc[:, list(contract.metadata_columns)].copy()
    for axis in contract.axes:
        out[axis] = MARKET_STATE_WIDE_OUTPUT_SENTINEL_STATE
        out[f"{axis}_id"] = MARKET_STATE_WIDE_OUTPUT_SENTINEL_ID
        out[f"{axis}_score"] = MARKET_STATE_WIDE_OUTPUT_SENTINEL_SCORE
        out[f"{axis}_confidence"] = MARKET_STATE_WIDE_OUTPUT_SENTINEL_CONFIDENCE
        out[f"{axis}_status"] = ASSIGNMENT_STATUS_NOT_SELECTED

    for axis, rows in (axis_assignments or {}).items():
        axis_name = _wide_axis(axis)
        if axis_name not in contract.axes:
            raise ValueError(f"Unsupported Market-State wide output axis {axis_name!r}")
        assignment = _normalize_wide_axis_assignments(axis_name, rows)
        if assignment.empty:
            continue
        merged = out.merge(assignment, on=list(MARKET_STATE_WIDE_OUTPUT_KEY_COLUMNS), how="left", suffixes=("", "__assignment"))
        for column in market_state_wide_axis_columns(axis_name):
            assignment_column = f"{column}__assignment"
            if assignment_column in merged.columns:
                merged[column] = merged[assignment_column].where(merged[assignment_column].notna(), merged[column])
        out = merged.drop(columns=[column for column in merged.columns if column.endswith("__assignment")])

    return validate_market_state_wide_output_rows(out, schema=contract)


def validate_market_state_wide_output_rows(
    rows: pd.DataFrame | Sequence[Mapping[str, Any]],
    *,
    schema: MarketStateWideOutputSchema | None = None,
) -> pd.DataFrame:
    contract = schema or default_market_state_wide_output_schema()
    frame = _apply_wide_metadata_aliases(_wide_frame(rows))
    missing = [column for column in contract.required_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Market-State wide output rows missing required columns: {missing}")
    forbidden = _wide_forbidden_columns(frame.columns, allowed=set(contract.required_columns))
    if forbidden:
        raise ValueError(f"Market-State wide output rows include forbidden columns: {forbidden}")
    out = frame.loc[:, list(contract.required_columns)].copy()
    if out.empty:
        return out
    _validate_wide_required_not_null(out, contract)
    _validate_wide_row_keys(out)
    _validate_wide_causal_metadata(out)
    out["interval_min"] = _positive_int_series(out["interval_min"], field_name="interval_min")
    out["schema_version"] = _positive_int_series(out["schema_version"], field_name="schema_version")
    if not out["schema_version"].eq(int(contract.schema_version)).all():
        raise ValueError("Market-State wide output rows must use the contract schema_version")
    for column in ("band", "profile_id", "profile_version", "universe_manifest_id", "feature_bundle_id", "lineage_id"):
        out[column] = out[column].map(lambda value: _non_empty_text(value, field_name=column))
    for axis in contract.axes:
        _validate_wide_axis_group(out, axis=axis, allowed_statuses=set(contract.allowed_statuses))
    return out


def write_market_state_wide_outputs(request: MarketStateWideWriteRequest) -> MarketStateWideWriteResult:
    root = validate_market_state_feature_write_root(request.output_root, production_enabled=bool(request.production_enabled))
    frame = validate_market_state_wide_output_rows(request.rows, schema=request.schema)
    boundary = market_state_wide_non_production_artifact_boundary()
    if frame.empty:
        return MarketStateWideWriteResult(
            status=MARKET_STATE_WIDE_OUTPUT_STATUS_NO_ROWS,
            output_root=root,
            run_id=request.run_id,
            artifact_boundary=boundary,
            metadata={"request_metadata": dict(request.metadata)},
        )

    written: list[Path] = []
    format_counts: dict[str, int] = {}
    for path, fmt in _write_wide_output_partitions(frame, root=root, run_id=request.run_id, file_format=request.file_format):
        written.append(path)
        format_counts[fmt] = format_counts.get(fmt, 0) + 1

    manifest_dir = root / "market_state_wide_output_manifests" / request.run_id
    _wide_ensure_within_root(manifest_dir, root)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    schema_path = manifest_dir / "schema.json"
    manifest_path = manifest_dir / "manifest.json"
    manifest = _wide_manifest_payload(
        request,
        frame=frame,
        root=root,
        written=written,
        file_format_counts=format_counts,
        artifact_boundary=boundary,
    )
    if request.write_manifest:
        _wide_write_json(schema_path, request.schema.as_dict())
        _wide_write_json(manifest_path, manifest)
        written.extend([schema_path, manifest_path])
    return MarketStateWideWriteResult(
        status=MARKET_STATE_WIDE_OUTPUT_STATUS_WRITTEN,
        output_root=root,
        run_id=request.run_id,
        row_count=int(frame.shape[0]),
        written_paths=tuple(written),
        manifest_path=manifest_path if request.write_manifest else None,
        schema_path=schema_path if request.write_manifest else None,
        file_format_counts=dict(sorted(format_counts.items())),
        artifact_boundary=boundary,
        manifest=manifest,
        metadata={"request_metadata": dict(request.metadata)},
    )


def _wide_frame(rows: pd.DataFrame | Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    if isinstance(rows, pd.DataFrame):
        return rows.copy()
    return pd.DataFrame([dict(row) for row in rows])


def _apply_wide_metadata_aliases(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for source, target in MARKET_STATE_WIDE_OUTPUT_METADATA_ALIASES.items():
        if target not in out.columns and source in out.columns:
            out[target] = out[source]
    return out


def _normalize_wide_axis_assignments(
    axis: str,
    rows: pd.DataFrame | Sequence[Mapping[str, Any]],
) -> pd.DataFrame:
    raw = _wide_frame(rows)
    if raw.empty:
        return raw
    missing_keys = [column for column in MARKET_STATE_WIDE_OUTPUT_KEY_COLUMNS if column not in raw.columns]
    if missing_keys:
        raise ValueError(f"Market-State wide axis assignment rows for {axis!r} missing key columns: {missing_keys}")
    status_column = _axis_assignment_column(raw, axis, "status", aliases=("assignment_status", "status"))
    state_column = _axis_assignment_column(raw, axis, "", aliases=("state", "regime_label", "label"))
    id_column = _axis_assignment_column(raw, axis, "id", aliases=("state_id", "regime_id", "label_id"))
    score_column = _axis_assignment_column(raw, axis, "score", aliases=("state_score", "assignment_score", "score"))
    confidence_column = _axis_assignment_column(raw, axis, "confidence", aliases=("assignment_confidence", "confidence"))
    required = {
        axis: state_column,
        f"{axis}_id": id_column,
        f"{axis}_score": score_column,
        f"{axis}_confidence": confidence_column,
        f"{axis}_status": status_column,
    }
    missing = [target for target, source in required.items() if source is None]
    if missing:
        raise ValueError(f"Market-State wide axis assignment rows for {axis!r} missing assignment columns: {missing}")
    out = raw.loc[:, list(MARKET_STATE_WIDE_OUTPUT_KEY_COLUMNS)].copy()
    for target, source in required.items():
        out[target] = raw[source]  # type: ignore[index]
    out[f"{axis}_status"] = out[f"{axis}_status"].map(lambda value: _assignment_status(value))
    masked = ~out[f"{axis}_status"].isin(_WIDE_ASSIGNED_STATUSES)
    out.loc[masked, axis] = MARKET_STATE_WIDE_OUTPUT_SENTINEL_STATE
    out.loc[masked, f"{axis}_id"] = MARKET_STATE_WIDE_OUTPUT_SENTINEL_ID
    out.loc[masked, f"{axis}_score"] = MARKET_STATE_WIDE_OUTPUT_SENTINEL_SCORE
    out.loc[masked, f"{axis}_confidence"] = MARKET_STATE_WIDE_OUTPUT_SENTINEL_CONFIDENCE
    if out.duplicated(list(MARKET_STATE_WIDE_OUTPUT_KEY_COLUMNS)).any():
        raise ValueError(f"Market-State wide axis assignment rows for {axis!r} contain duplicate ts/band/profile_id keys")
    return out


def _axis_assignment_column(frame: pd.DataFrame, axis: str, suffix: str, *, aliases: Sequence[str]) -> str | None:
    candidates = [axis if not suffix else f"{axis}_{suffix}", *aliases]
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
    return None


def _assignment_status(value: object) -> str:
    status = _non_empty_text(value, field_name="assignment_status").lower()
    if status not in MARKET_STATE_ASSIGNMENT_STATUSES:
        raise ValueError(f"Unsupported Market-State wide output assignment status {status!r}")
    return status


def _wide_forbidden_columns(columns: Sequence[str], *, allowed: set[str]) -> list[str]:
    forbidden: list[str] = []
    for column in columns:
        name = str(column)
        lowered = name.lower()
        if name in allowed:
            continue
        if lowered in MARKET_STATE_WIDE_OUTPUT_FORBIDDEN_COLUMNS:
            forbidden.append(name)
            continue
        if lowered.startswith("asset_market_state") or lowered.startswith("cross_asset_") or lowered.startswith("l2_"):
            forbidden.append(name)
            continue
        if "order_book" in lowered or "peer_label" in lowered:
            forbidden.append(name)
    return sorted(set(forbidden))


def _validate_wide_required_not_null(frame: pd.DataFrame, schema: MarketStateWideOutputSchema) -> None:
    null_columns = [column for column in schema.required_columns if frame[column].isna().any()]
    if null_columns:
        raise ValueError(f"Market-State wide output rows contain null required values: {null_columns}")
    for column in ("ts", "known_at_ts", "source_tail_ts", "lineage_id"):
        empty = frame[column].map(lambda value: not str(value).strip()).any()
        if empty:
            raise ValueError(f"Market-State wide output rows require non-empty {column}")


def _validate_wide_row_keys(frame: pd.DataFrame) -> None:
    if frame.duplicated(list(MARKET_STATE_WIDE_OUTPUT_KEY_COLUMNS)).any():
        raise ValueError("Market-State wide output rows must be unique by ts, band, and profile_id")


def _validate_wide_causal_metadata(frame: pd.DataFrame) -> None:
    ts = pd.to_numeric(frame["ts"], errors="coerce")
    known = pd.to_numeric(frame["known_at_ts"], errors="coerce")
    source_tail = pd.to_numeric(frame["source_tail_ts"], errors="coerce")
    if not np.isfinite(ts).all() or not np.isfinite(known).all() or not np.isfinite(source_tail).all():
        raise ValueError("Market-State wide output causal timestamps must be finite numeric values")
    if (source_tail > known).any():
        raise ValueError("Market-State wide output source_tail_ts must not exceed known_at_ts")


def _positive_int_series(series: pd.Series, *, field_name: str) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if not np.isfinite(numeric).all():
        raise ValueError(f"Market-State wide output {field_name} must be finite")
    ints = numeric.astype("int64")
    if (ints <= 0).any():
        raise ValueError(f"Market-State wide output {field_name} must be positive")
    if not (numeric == ints).all():
        raise ValueError(f"Market-State wide output {field_name} must be an integer")
    return ints


def _validate_wide_axis_group(frame: pd.DataFrame, *, axis: str, allowed_statuses: set[str]) -> None:
    status_column = f"{axis}_status"
    id_column = f"{axis}_id"
    score_column = f"{axis}_score"
    confidence_column = f"{axis}_confidence"
    frame[status_column] = frame[status_column].map(lambda value: _assignment_status(value))
    invalid = sorted(set(frame[status_column]).difference(allowed_statuses))
    if invalid:
        raise ValueError(f"Market-State wide output axis {axis!r} has invalid statuses: {invalid}")
    frame[id_column] = _int_series(frame[id_column], field_name=id_column)
    frame[score_column] = _bounded_float_series(frame[score_column], field_name=score_column)
    frame[confidence_column] = _bounded_float_series(frame[confidence_column], field_name=confidence_column)
    assigned = frame[status_column].isin(_WIDE_ASSIGNED_STATUSES)
    assigned_state_bad = frame.loc[assigned, axis].map(lambda value: _non_empty_text(value, field_name=axis) == MARKET_STATE_WIDE_OUTPUT_SENTINEL_STATE)
    if assigned_state_bad.any() or frame.loc[assigned, id_column].eq(MARKET_STATE_WIDE_OUTPUT_SENTINEL_ID).any():
        raise ValueError(f"Market-State wide output axis {axis!r} valid/low_confidence rows require real state assignments")
    masked = ~assigned
    if not masked.any():
        return
    masked_state = frame.loc[masked, axis].astype(str).eq(MARKET_STATE_WIDE_OUTPUT_SENTINEL_STATE).all()
    masked_id = frame.loc[masked, id_column].eq(MARKET_STATE_WIDE_OUTPUT_SENTINEL_ID).all()
    masked_score = frame.loc[masked, score_column].eq(MARKET_STATE_WIDE_OUTPUT_SENTINEL_SCORE).all()
    masked_confidence = frame.loc[masked, confidence_column].eq(MARKET_STATE_WIDE_OUTPUT_SENTINEL_CONFIDENCE).all()
    if not (masked_state and masked_id and masked_score and masked_confidence):
        raise ValueError(f"Market-State wide output axis {axis!r} masked rows must use explicit sentinel values")


def _int_series(series: pd.Series, *, field_name: str) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if not np.isfinite(numeric).all():
        raise ValueError(f"Market-State wide output {field_name} must be finite")
    ints = numeric.astype("int64")
    if not (numeric == ints).all():
        raise ValueError(f"Market-State wide output {field_name} must be an integer")
    return ints


def _bounded_float_series(series: pd.Series, *, field_name: str) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if not np.isfinite(numeric).all():
        raise ValueError(f"Market-State wide output {field_name} must be finite")
    floats = numeric.astype(float)
    if (floats < 0.0).any() or (floats > 1.0).any():
        raise ValueError(f"Market-State wide output {field_name} must be between 0 and 1")
    return floats


def _write_wide_output_partitions(
    frame: pd.DataFrame,
    *,
    root: Path,
    run_id: str,
    file_format: str,
) -> list[tuple[Path, str]]:
    work = _wide_with_time_partitions(frame)
    written: list[tuple[Path, str]] = []
    for (band, profile_id, year, month), group in work.groupby(["band", "profile_id", "_year", "_month"], sort=True):
        partition = (
            root
            / "market_state_wide_outputs"
            / f"run_id={_wide_safe_part(run_id, field_name='run_id')}"
            / f"band={_wide_safe_part(band, field_name='band')}"
            / f"profile_id={_wide_safe_part(profile_id, field_name='profile_id')}"
            / f"year={int(year):04d}"
            / f"month={int(month):02d}"
        )
        written.append(_wide_write_frame(group.drop(columns=["_year", "_month"], errors="ignore"), partition, root=root, file_format=file_format))
    return written


def _wide_write_frame(frame: pd.DataFrame, partition: Path, *, root: Path, file_format: str) -> tuple[Path, str]:
    _wide_ensure_within_root(partition, root)
    partition.mkdir(parents=True, exist_ok=True)
    if file_format == MARKET_STATE_WIDE_OUTPUT_FORMAT_JSONL:
        path = partition / "part-000.jsonl"
        _wide_write_jsonl_frame(path, frame)
        return path, MARKET_STATE_WIDE_OUTPUT_FORMAT_JSONL
    if file_format == MARKET_STATE_WIDE_OUTPUT_FORMAT_PARQUET:
        path = partition / "part-000.parquet"
        frame.to_parquet(path, index=False)
        return path, MARKET_STATE_WIDE_OUTPUT_FORMAT_PARQUET
    parquet_path = partition / "part-000.parquet"
    try:
        frame.to_parquet(parquet_path, index=False)
        return parquet_path, MARKET_STATE_WIDE_OUTPUT_FORMAT_PARQUET
    except Exception:
        jsonl_path = partition / "part-000.jsonl"
        _wide_write_jsonl_frame(jsonl_path, frame)
        return jsonl_path, MARKET_STATE_WIDE_OUTPUT_FORMAT_JSONL


def _wide_with_time_partitions(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.copy()
    ts = pd.to_numeric(work["ts"], errors="coerce")
    if not np.isfinite(ts).all():
        raise ValueError("Market-State wide output writer requires finite numeric ts values for partitioning")
    work["ts"] = ts.astype("int64")
    dt = pd.to_datetime(work["ts"], unit="s", utc=True)
    work["_year"] = dt.dt.year.astype("int64")
    work["_month"] = dt.dt.month.astype("int64")
    return work


def _wide_manifest_payload(
    request: MarketStateWideWriteRequest,
    *,
    frame: pd.DataFrame,
    root: Path,
    written: Sequence[Path],
    file_format_counts: Mapping[str, int],
    artifact_boundary: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": MARKET_STATE_WIDE_OUTPUT_SCHEMA_VERSION,
        "artifact_kind": "market_state_v1_wide_output_manifest",
        "run_id": request.run_id,
        "output_root": "runtime_only_not_serialized",
        "artifact_boundary": to_jsonable(dict(artifact_boundary)),
        "row_count": int(frame.shape[0]),
        "row_grain": list(MARKET_STATE_WIDE_OUTPUT_KEY_COLUMNS),
        "axes": list(request.schema.axes),
        "metadata_columns": list(request.schema.metadata_columns),
        "required_columns": list(request.schema.required_columns),
        "status_counts": _wide_status_counts(frame, axes=request.schema.axes),
        "frame_summary": _wide_frame_summary(frame),
        "written_paths": [_wide_relative_path(path, root) for path in written],
        "artifact_refs": [
            make_artifact_ref(
                path,
                artifact_kind="market_state_v1_wide_output_rows",
                artifact_root=root,
                producer="src.regimes.market_state.output_contract",
            ).as_dict()
            for path in written
            if path.exists() and path.is_file()
        ],
        "file_format_counts": dict(sorted(file_format_counts.items())),
        "metadata": to_jsonable(dict(request.metadata)),
        "production_enabled": False,
        "production_outputs_written": False,
        "production_profile_promotion_allowed": False,
        "composite_market_state_label_written": False,
    }


def _wide_status_counts(frame: pd.DataFrame, *, axes: Sequence[str]) -> dict[str, dict[str, int]]:
    return {
        axis: {str(status): int(count) for status, count in frame[f"{axis}_status"].value_counts(dropna=False).sort_index().items()}
        for axis in axes
    }


def _wide_frame_summary(frame: pd.DataFrame) -> dict[str, Any]:
    return {
        "bands": sorted(str(value) for value in frame["band"].dropna().unique()),
        "interval_minutes": sorted(int(value) for value in pd.to_numeric(frame["interval_min"], errors="coerce").dropna().unique()),
        "profile_ids": sorted(str(value) for value in frame["profile_id"].dropna().unique()),
        "profile_versions": sorted(str(value) for value in frame["profile_version"].dropna().unique()),
        "universe_manifest_ids": sorted(str(value) for value in frame["universe_manifest_id"].dropna().unique()),
        "feature_bundle_ids": sorted(str(value) for value in frame["feature_bundle_id"].dropna().unique()),
        "lineage_ids": sorted(str(value) for value in frame["lineage_id"].dropna().unique()),
        "known_at": {
            "min_known_at_ts": int(pd.to_numeric(frame["known_at_ts"], errors="coerce").min()),
            "max_known_at_ts": int(pd.to_numeric(frame["known_at_ts"], errors="coerce").max()),
        },
    }


def _wide_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(to_jsonable(dict(payload)), handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(tmp, path)


def _wide_write_jsonl_frame(path: Path, frame: pd.DataFrame) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in frame.to_dict(orient="records"):
            handle.write(json.dumps(to_jsonable(dict(row)), sort_keys=True))
            handle.write("\n")
    os.replace(tmp, path)


def _wide_file_format(value: str) -> str:
    text = str(value).strip().lower()
    valid = {
        MARKET_STATE_WIDE_OUTPUT_FORMAT_AUTO,
        MARKET_STATE_WIDE_OUTPUT_FORMAT_PARQUET,
        MARKET_STATE_WIDE_OUTPUT_FORMAT_JSONL,
    }
    if text not in valid:
        raise ValueError(f"Unsupported Market-State wide output file_format {value!r}")
    return text


def _wide_safe_part(value: object, *, field_name: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"Market-State wide output {field_name} must be non-empty")
    if any(part in text for part in ("/", "\\", ":", "\x00")):
        raise ValueError(f"Market-State wide output {field_name} contains unsafe path characters")
    return re.sub(r"[^A-Za-z0-9_.=-]+", "_", text)


def _wide_ensure_within_root(path: Path, root: Path) -> None:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"Market-State wide output writer refusing to write outside output root: {path}") from exc


def _wide_relative_path(path: Path, root: Path) -> str:
    return validate_portable_relative_path(path.resolve().relative_to(root.resolve()).as_posix())


def _validate_label_contract(
    *,
    label_source: str,
    regime_id: object,
    regime_label: object,
    state_strength: float | None,
    clusterability_status: str,
    fallback_status: str,
) -> None:
    if label_source == LABEL_SOURCE_CLUSTERED:
        if regime_id is None or regime_label is None or state_strength is None:
            raise ValueError("Clustered market-state rows require regime_id, regime_label, and state_strength")
        if clusterability_status != MARKET_STATE_CLUSTERABILITY_STATUS_CLUSTERABLE_CANDIDATE or fallback_status != FALLBACK_STATUS_NO_FALLBACK_NEEDED:
            raise ValueError("Clustered market-state rows require clusterable candidate and no fallback")
    elif label_source == LABEL_SOURCE_LOW_VARIATION_SINGLE_STATE_FALLBACK:
        if regime_id is None or regime_label is None or state_strength is None:
            raise ValueError("Low-variation fallback rows require fallback regime_id, regime_label, and state_strength")
        if clusterability_status != MARKET_STATE_CLUSTERABILITY_STATUS_LOW_VARIATION_MARKET_WINDOW or fallback_status != FALLBACK_STATUS_LOW_VARIATION_SINGLE_STATE:
            raise ValueError("Low-variation fallback rows require low_variation_market_window/low_variation_single_state")
    else:
        if regime_id is not None or regime_label is not None or state_strength is not None:
            raise ValueError("No-label market-state rows must not emit regime labels or state strength")
        expected = (
            FALLBACK_STATUS_INSUFFICIENT_DATA_NO_LABEL
            if label_source == LABEL_SOURCE_INSUFFICIENT_DATA_NO_LABEL
            else FALLBACK_STATUS_AXIS_NOT_APPLICABLE
        )
        if fallback_status != expected:
            raise ValueError(f"No-label market-state rows require fallback_status={expected}")


def _label_source(value: object) -> str:
    text = _non_empty_text(value, field_name="regime_label_source").lower()
    if text not in MARKET_STATE_LABEL_SOURCES:
        raise ValueError(f"Unsupported market-state regime_label_source {text!r}; expected one of: {', '.join(MARKET_STATE_LABEL_SOURCES)}")
    return text


def _clusterability_status(value: object) -> str:
    text = _non_empty_text(value, field_name="market_clusterability_status").lower()
    if text not in MARKET_STATE_CLUSTERABILITY_STATUSES:
        raise ValueError(f"Unsupported market-state clusterability status {text!r}")
    return text


def _fallback_status(value: object) -> str:
    text = _non_empty_text(value, field_name="fallback_status").lower()
    if text not in MARKET_STATE_FALLBACK_STATUSES:
        raise ValueError(f"Unsupported market-state fallback_status {text!r}")
    return text


def _state_strength(value: object) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except Exception as exc:
        raise ValueError("Market-state state_strength must be numeric when supplied") from exc
    if out < 0.0 or out > 1.0:
        raise ValueError("Market-state state_strength must be between 0 and 1")
    return out


def _created_at(value: object) -> str:
    if value is None:
        return datetime.now(timezone.utc).isoformat()
    return _non_empty_text(value, field_name="created_at")


def _optional_text(value: object | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _lineage_id_from_mapping(value: Mapping[str, Any]) -> str | None:
    for key in ("lineage_id", "run_id"):
        found = _optional_text(value.get(key))
        if found:
            return found
    return None


def _to_orderable(value: object, *, field_name: str) -> float:
    if value is None or isinstance(value, bool):
        raise ValueError(f"Market-state output {field_name} must be timestamp-compatible")
    try:
        return float(value)
    except Exception as exc:
        raise ValueError(f"Market-state output {field_name} must be numeric") from exc


def _artifact_boundary(value: Mapping[str, Any] | None) -> dict[str, Any]:
    boundary = market_state_non_production_artifact_boundary()
    boundary.update(_mapping(value, field_name="artifact_boundary") if value is not None else {})
    if str(boundary.get("classification", "")).strip().lower() != MARKET_STATE_SANDBOX_OUTPUT_BOUNDARY:
        raise ValueError("Market-state output artifact_boundary classification must be sandbox_non_production")
    for key in (
        "production_output",
        "production_parquet_allowed",
        "production_regime_labels_allowed",
        "production_profile_promotion_allowed",
        "asset_level_labels_allowed",
        "relative_cross_asset_execution_allowed",
    ):
        if bool(boundary.get(key)):
            raise ValueError(f"Market-state output artifact_boundary cannot set {key}=true")
    return to_jsonable(boundary)


__all__ = [
    "FALLBACK_STATUS_AXIS_NOT_APPLICABLE",
    "FALLBACK_STATUS_INSUFFICIENT_DATA_NO_LABEL",
    "FALLBACK_STATUS_LOW_VARIATION_SINGLE_STATE",
    "FALLBACK_STATUS_NO_FALLBACK_NEEDED",
    "LABEL_SOURCE_AXIS_NOT_APPLICABLE",
    "LABEL_SOURCE_CLUSTERED",
    "LABEL_SOURCE_INSUFFICIENT_DATA_NO_LABEL",
    "LABEL_SOURCE_LOW_VARIATION_SINGLE_STATE_FALLBACK",
    "MARKET_STATE_FALLBACK_STATUSES",
    "MARKET_STATE_LABEL_SOURCES",
    "MARKET_STATE_OUTPUT_PARTITION_FIELDS",
    "MARKET_STATE_OUTPUT_REQUIRED_FIELDS",
    "MARKET_STATE_SANDBOX_OUTPUT_BOUNDARY",
    "MARKET_STATE_WIDE_OUTPUT_AXES",
    "MARKET_STATE_WIDE_OUTPUT_ARTIFACT_KIND",
    "MARKET_STATE_WIDE_OUTPUT_FORMAT_AUTO",
    "MARKET_STATE_WIDE_OUTPUT_FORMAT_JSONL",
    "MARKET_STATE_WIDE_OUTPUT_FORMAT_PARQUET",
    "MARKET_STATE_WIDE_OUTPUT_FORBIDDEN_COLUMNS",
    "MARKET_STATE_WIDE_OUTPUT_KEY_COLUMNS",
    "MARKET_STATE_WIDE_OUTPUT_METADATA_ALIASES",
    "MARKET_STATE_WIDE_OUTPUT_METADATA_COLUMNS",
    "MARKET_STATE_WIDE_OUTPUT_REQUIRED_COLUMNS",
    "MARKET_STATE_WIDE_OUTPUT_SCHEMA_VERSION",
    "MARKET_STATE_WIDE_OUTPUT_SENTINEL_CONFIDENCE",
    "MARKET_STATE_WIDE_OUTPUT_SENTINEL_ID",
    "MARKET_STATE_WIDE_OUTPUT_SENTINEL_SCORE",
    "MARKET_STATE_WIDE_OUTPUT_SENTINEL_STATE",
    "MARKET_STATE_WIDE_OUTPUT_STATUS_NO_ROWS",
    "MARKET_STATE_WIDE_OUTPUT_STATUS_WRITTEN",
    "MarketStateOutputRow",
    "MarketStateSandboxOutputSchema",
    "MarketStateWideOutputSchema",
    "MarketStateWideWriteRequest",
    "MarketStateWideWriteResult",
    "axis_not_applicable_market_state_output_row",
    "build_market_state_wide_output_rows",
    "clustered_market_state_output_rows",
    "coerce_market_state_output_row",
    "default_market_state_sandbox_output_schema",
    "default_market_state_wide_output_schema",
    "insufficient_market_state_no_label_output_row",
    "low_variation_single_state_output_rows",
    "market_state_non_production_artifact_boundary",
    "market_state_wide_axis_columns",
    "market_state_wide_non_production_artifact_boundary",
    "validate_market_state_output_rows",
    "validate_market_state_wide_output_rows",
    "write_market_state_wide_outputs",
]
