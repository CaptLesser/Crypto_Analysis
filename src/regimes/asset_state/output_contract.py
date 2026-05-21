from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

from src.regimes.asset_state.clusterability import (
    CLUSTERABILITY_FALLBACK_STATUSES,
    CLUSTERABILITY_STATUSES,
    CLUSTERABILITY_STATUS_AXIS_NOT_CLUSTERABLE,
    CLUSTERABILITY_STATUS_CLUSTERABLE_CANDIDATE,
    CLUSTERABILITY_STATUS_INSUFFICIENT_FINITE_ROWS,
    CLUSTERABILITY_STATUS_INSUFFICIENT_HISTORY,
    CLUSTERABILITY_STATUS_VALID_FLAT_SINGLE_STATE,
    FALLBACK_STATUS_AXIS_NOT_APPLICABLE,
    FALLBACK_STATUS_INSUFFICIENT_DATA_NO_LABEL,
    FALLBACK_STATUS_NEUTRAL_FLAT_FALLBACK,
    FALLBACK_STATUS_NO_FALLBACK_NEEDED,
)
from src.regimes.asset_state.contracts import (
    ASSET_STATE_PATHWAY_VALUE,
    ASSET_STATE_SCHEMA_VERSION,
    AssetStateAxis,
    AssetStateBand,
    AssetStatePathway,
    AssetStateSchemaVersion,
    _enum_value,
    _mapping,
    _non_empty_text,
    _schema_version,
    _string_tuple,
)
from src.regimes.asset_state.taxonomy import default_asset_state_taxonomy
from src.regimes.core.known_at import KnownAtSpec
from src.regimes.core.lineage import RegimeLineageSpec
from src.regimes.core.serialization import dumps_json, to_jsonable


ASSET_STATE_SANDBOX_OUTPUT_ARTIFACT_KIND = "asset_state_sandbox_regime_output"
ASSET_STATE_SANDBOX_OUTPUT_BOUNDARY = "sandbox_non_production"

LABEL_SOURCE_CLUSTERED = "clustered"
LABEL_SOURCE_NEUTRAL_FLAT_FALLBACK = "neutral_flat_fallback"
LABEL_SOURCE_INSUFFICIENT_DATA_NO_LABEL = "insufficient_data_no_label"
LABEL_SOURCE_AXIS_NOT_APPLICABLE = "axis_not_applicable"

ASSET_STATE_LABEL_SOURCES: tuple[str, ...] = (
    LABEL_SOURCE_CLUSTERED,
    LABEL_SOURCE_NEUTRAL_FLAT_FALLBACK,
    LABEL_SOURCE_INSUFFICIENT_DATA_NO_LABEL,
    LABEL_SOURCE_AXIS_NOT_APPLICABLE,
)

ASSET_STATE_OUTPUT_REQUIRED_FIELDS: tuple[str, ...] = (
    "asset",
    "ts",
    "pathway",
    "axis",
    "band",
    "interval",
    "regime_id",
    "regime_label",
    "regime_label_source",
    "state_strength",
    "clusterability_status",
    "fallback_status",
    "profile_id",
    "feature_profile_id",
    "feature_pool_id",
    "clusterer_family",
    "assignment_policy",
    "refit_key",
    "known_at_ts",
    "source_tail_ts",
    "lineage_id",
    "schema_version",
    "description_metadata",
    "created_at",
    "run_id",
    "artifact_boundary",
)

ASSET_STATE_OUTPUT_PARTITION_FIELDS: tuple[str, ...] = (
    "run_id",
    "axis",
    "band",
    "interval",
    "asset",
)


def non_production_artifact_boundary() -> dict[str, Any]:
    return {
        "classification": ASSET_STATE_SANDBOX_OUTPUT_BOUNDARY,
        "production_output": False,
        "production_parquet_allowed": False,
        "production_regime_labels_allowed": False,
        "production_profile_promotion_allowed": False,
        "write_root_policy": "reports/regimes/foundation/asset_state_test/sandbox_outputs only",
    }


@dataclass(frozen=True)
class AssetStateSandboxOutputSchema:
    required_fields: Sequence[str] = ASSET_STATE_OUTPUT_REQUIRED_FIELDS
    partition_fields: Sequence[str] = ASSET_STATE_OUTPUT_PARTITION_FIELDS
    pathway: str | AssetStatePathway = AssetStatePathway.ASSET_STATE
    artifact_kind: str = "asset_state_sandbox_output_schema"
    artifact_boundary: Mapping[str, Any] = field(default_factory=non_production_artifact_boundary)
    schema_version: int | AssetStateSchemaVersion = ASSET_STATE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        required = _string_tuple(self.required_fields, field_name="required_fields", require_non_empty=True)
        partition = _string_tuple(self.partition_fields, field_name="partition_fields", require_non_empty=True)
        missing = sorted(set(ASSET_STATE_OUTPUT_REQUIRED_FIELDS).difference(required))
        if missing:
            raise ValueError(f"Asset-state sandbox output schema missing required fields: {missing}")
        boundary = _artifact_boundary(self.artifact_boundary)
        object.__setattr__(self, "schema_version", _schema_version(self.schema_version))
        object.__setattr__(self, "pathway", _enum_value(self.pathway, AssetStatePathway, field_name="pathway"))
        object.__setattr__(self, "artifact_kind", _non_empty_text(self.artifact_kind, field_name="artifact_kind"))
        object.__setattr__(self, "required_fields", required)
        object.__setattr__(self, "partition_fields", partition)
        object.__setattr__(self, "artifact_boundary", boundary)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "artifact_kind": self.artifact_kind,
            "pathway": self.pathway,
            "required_fields": list(self.required_fields),
            "partition_fields": list(self.partition_fields),
            "label_sources": list(ASSET_STATE_LABEL_SOURCES),
            "storage_notes": {
                "description_metadata": "JSON object in contract; writer stores as JSON text for parquet-compatible scalar storage.",
                "artifact_boundary": "JSON object in contract; writer stores as JSON text for parquet-compatible scalar storage.",
                "state_strength": "Nullable numeric confidence/strength field; no-label rows may be null.",
            },
            "artifact_boundary": to_jsonable(dict(self.artifact_boundary)),
        }

    def to_json(self, **kwargs: Any) -> str:
        return dumps_json(self.as_dict(), **kwargs)


@dataclass(frozen=True)
class AssetStateSandboxOutputRow:
    asset: str
    ts: int | float | str
    axis: str | AssetStateAxis
    band: str | AssetStateBand
    interval: int
    regime_id: str | int | None
    regime_label: str | None
    regime_label_source: str
    state_strength: float | None
    clusterability_status: str
    fallback_status: str
    profile_id: str
    feature_pool_id: str
    clusterer_family: str
    assignment_policy: str
    refit_key: str
    known_at_ts: int | float | str | None = None
    source_tail_ts: int | float | str | None = None
    lineage_id: str | None = None
    feature_profile_id: str | None = None
    description_metadata: Mapping[str, Any] = field(default_factory=dict)
    run_id: str = "asset_state_sandbox_run"
    created_at: str | None = None
    artifact_boundary: Mapping[str, Any] = field(default_factory=non_production_artifact_boundary)
    pathway: str | AssetStatePathway = AssetStatePathway.ASSET_STATE
    schema_version: int | AssetStateSchemaVersion = ASSET_STATE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        pathway = _enum_value(self.pathway, AssetStatePathway, field_name="pathway")
        if pathway != ASSET_STATE_PATHWAY_VALUE:
            raise ValueError("Asset-state sandbox output pathway must be asset_state")
        axis = _enum_value(self.axis, AssetStateAxis, field_name="axis")
        band = _enum_value(self.band, AssetStateBand, field_name="band")
        default_asset_state_taxonomy().validate_axis_band(axis, band)
        interval = int(self.interval)
        if interval <= 0:
            raise ValueError("Asset-state sandbox output interval must be positive")
        label_source = _label_source(self.regime_label_source)
        clusterability_status = _clusterability_status(self.clusterability_status)
        fallback_status = _fallback_status(self.fallback_status)
        state_strength = _state_strength(self.state_strength, label_source=label_source)
        if str(self.ts).strip() == "":
            raise ValueError("Asset-state sandbox output ts must be non-empty")
        _validate_label_contract(
            label_source=label_source,
            regime_id=self.regime_id,
            regime_label=self.regime_label,
            state_strength=state_strength,
            clusterability_status=clusterability_status,
            fallback_status=fallback_status,
        )
        object.__setattr__(self, "schema_version", _schema_version(self.schema_version))
        object.__setattr__(self, "asset", _non_empty_text(self.asset, field_name="asset"))
        object.__setattr__(self, "pathway", pathway)
        object.__setattr__(self, "axis", axis)
        object.__setattr__(self, "band", band)
        object.__setattr__(self, "interval", interval)
        object.__setattr__(self, "regime_id", None if self.regime_id is None else str(self.regime_id).strip())
        object.__setattr__(self, "regime_label", None if self.regime_label is None else str(self.regime_label).strip())
        object.__setattr__(self, "regime_label_source", label_source)
        object.__setattr__(self, "state_strength", state_strength)
        object.__setattr__(self, "clusterability_status", clusterability_status)
        object.__setattr__(self, "fallback_status", fallback_status)
        object.__setattr__(self, "profile_id", _non_empty_text(self.profile_id, field_name="profile_id"))
        object.__setattr__(self, "feature_pool_id", _non_empty_text(self.feature_pool_id, field_name="feature_pool_id"))
        object.__setattr__(
            self,
            "feature_profile_id",
            _optional_text(self.feature_profile_id) or _non_empty_text(self.feature_pool_id, field_name="feature_pool_id"),
        )
        object.__setattr__(self, "clusterer_family", _non_empty_text(self.clusterer_family, field_name="clusterer_family"))
        object.__setattr__(self, "assignment_policy", _non_empty_text(self.assignment_policy, field_name="assignment_policy"))
        object.__setattr__(self, "refit_key", _non_empty_text(self.refit_key, field_name="refit_key"))
        object.__setattr__(self, "run_id", _non_empty_text(self.run_id, field_name="run_id"))
        object.__setattr__(self, "created_at", _created_at(self.created_at))
        metadata = _description_metadata(self.description_metadata)
        known_at_ts = _resolve_known_at_ts(self.known_at_ts, metadata=metadata, ts=self.ts)
        source_tail_ts = _resolve_source_tail_ts(self.source_tail_ts, metadata=metadata, known_at_ts=known_at_ts)
        if _to_orderable(source_tail_ts, field_name="source_tail_ts") > _to_orderable(known_at_ts, field_name="known_at_ts"):
            raise ValueError("Asset-state sandbox output source_tail_ts must not exceed known_at_ts")
        object.__setattr__(self, "known_at_ts", known_at_ts)
        object.__setattr__(self, "source_tail_ts", source_tail_ts)
        object.__setattr__(self, "lineage_id", _resolve_lineage_id(self.lineage_id, metadata=metadata, row=self))
        object.__setattr__(self, "description_metadata", metadata)
        object.__setattr__(self, "artifact_boundary", _artifact_boundary(self.artifact_boundary))

    def as_dict(self) -> dict[str, Any]:
        return {
            "asset": self.asset,
            "ts": self.ts,
            "pathway": self.pathway,
            "axis": self.axis,
            "band": self.band,
            "interval": int(self.interval),
            "regime_id": self.regime_id,
            "regime_label": self.regime_label,
            "regime_label_source": self.regime_label_source,
            "state_strength": self.state_strength,
            "clusterability_status": self.clusterability_status,
            "fallback_status": self.fallback_status,
            "profile_id": self.profile_id,
            "feature_profile_id": self.feature_profile_id,
            "feature_pool_id": self.feature_pool_id,
            "clusterer_family": self.clusterer_family,
            "assignment_policy": self.assignment_policy,
            "refit_key": self.refit_key,
            "known_at_ts": self.known_at_ts,
            "source_tail_ts": self.source_tail_ts,
            "lineage_id": self.lineage_id,
            "schema_version": int(self.schema_version),
            "description_metadata": to_jsonable(dict(self.description_metadata)),
            "created_at": self.created_at,
            "run_id": self.run_id,
            "artifact_boundary": to_jsonable(dict(self.artifact_boundary)),
        }

    def storage_dict(self) -> dict[str, Any]:
        payload = self.as_dict()
        payload["description_metadata"] = dumps_json(payload["description_metadata"], separators=(",", ":"))
        payload["artifact_boundary"] = dumps_json(payload["artifact_boundary"], separators=(",", ":"))
        return payload


def default_asset_state_sandbox_output_schema() -> AssetStateSandboxOutputSchema:
    return AssetStateSandboxOutputSchema()


def coerce_asset_state_output_row(row: AssetStateSandboxOutputRow | Mapping[str, Any]) -> AssetStateSandboxOutputRow:
    if isinstance(row, AssetStateSandboxOutputRow):
        return row
    if not isinstance(row, Mapping):
        raise ValueError("Asset-state sandbox output rows must be mappings or AssetStateSandboxOutputRow objects")
    payload = dict(row)
    if "feature_pool_id" not in payload and "feature_profile_id" in payload:
        payload["feature_pool_id"] = payload.pop("feature_profile_id")
    if "state_strength" not in payload and "confidence" in payload:
        payload["state_strength"] = payload.pop("confidence")
    for json_field in ("description_metadata", "artifact_boundary"):
        if isinstance(payload.get(json_field), str):
            try:
                payload[json_field] = json.loads(payload[json_field])
            except json.JSONDecodeError as exc:
                raise ValueError(f"Asset-state sandbox output {json_field} must be valid JSON when stored as text") from exc
    payload.setdefault("created_at", None)
    payload.setdefault("artifact_boundary", non_production_artifact_boundary())
    return AssetStateSandboxOutputRow(**payload)


def validate_asset_state_output_rows(
    rows: Sequence[AssetStateSandboxOutputRow | Mapping[str, Any]],
) -> tuple[AssetStateSandboxOutputRow, ...]:
    if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence):
        raise ValueError("Asset-state sandbox output rows must be a non-empty sequence")
    normalized = tuple(coerce_asset_state_output_row(row) for row in rows)
    if not normalized:
        raise ValueError("Asset-state sandbox output rows must be non-empty")
    required = set(ASSET_STATE_OUTPUT_REQUIRED_FIELDS)
    for row in normalized:
        missing = sorted(required.difference(row.as_dict()))
        if missing:
            raise ValueError(f"Asset-state sandbox output row missing required fields: {missing}")
    return normalized


def clustered_asset_state_output_rows(
    *,
    asset: str,
    timestamps: Sequence[int | float | str],
    labels: Sequence[str | int],
    axis: str | AssetStateAxis,
    band: str | AssetStateBand,
    interval: int,
    profile_id: str,
    feature_pool_id: str,
    clusterer_family: str,
    assignment_policy: str,
    refit_key: str,
    strengths: Sequence[float | None] | None = None,
    regime_label_map: Mapping[str | int, str] | None = None,
    run_id: str = "asset_state_sandbox_run",
    created_at: str | None = None,
    description_metadata: Mapping[str, Any] | None = None,
) -> tuple[AssetStateSandboxOutputRow, ...]:
    if len(timestamps) != len(labels):
        raise ValueError("Asset-state clustered output timestamps and labels must have equal length")
    if strengths is not None and len(strengths) != len(labels):
        raise ValueError("Asset-state clustered output strengths and labels must have equal length")
    mapping = {str(key): str(value) for key, value in dict(regime_label_map or {}).items()}
    out: list[AssetStateSandboxOutputRow] = []
    for idx, (ts, label) in enumerate(zip(timestamps, labels)):
        label_text = str(label).strip()
        out.append(
            AssetStateSandboxOutputRow(
                asset=asset,
                ts=ts,
                axis=axis,
                band=band,
                interval=interval,
                regime_id=label_text,
                regime_label=mapping.get(label_text, f"state_{label_text}"),
                regime_label_source=LABEL_SOURCE_CLUSTERED,
                state_strength=1.0 if strengths is None else strengths[idx],
                clusterability_status=CLUSTERABILITY_STATUS_CLUSTERABLE_CANDIDATE,
                fallback_status=FALLBACK_STATUS_NO_FALLBACK_NEEDED,
                profile_id=profile_id,
                feature_pool_id=feature_pool_id,
                clusterer_family=clusterer_family,
                assignment_policy=assignment_policy,
                refit_key=refit_key,
                description_metadata={
                    "row_role": "active_clustered_label",
                    "label_index": idx,
                    **dict(description_metadata or {}),
                },
                run_id=run_id,
                created_at=created_at,
            )
        )
    return tuple(out)


def neutral_flat_fallback_output_rows(
    *,
    asset: str,
    timestamps: Sequence[int | float | str],
    axis: str | AssetStateAxis,
    band: str | AssetStateBand,
    interval: int,
    profile_id: str = "flat_fallback_profile",
    feature_pool_id: str = "flat_fallback_feature_pool",
    run_id: str = "asset_state_sandbox_run",
    created_at: str | None = None,
    description_metadata: Mapping[str, Any] | None = None,
) -> tuple[AssetStateSandboxOutputRow, ...]:
    return tuple(
        AssetStateSandboxOutputRow(
            asset=asset,
            ts=ts,
            axis=axis,
            band=band,
            interval=interval,
            regime_id="neutral_flat",
            regime_label="neutral_flat",
            regime_label_source=LABEL_SOURCE_NEUTRAL_FLAT_FALLBACK,
            state_strength=1.0,
            clusterability_status=CLUSTERABILITY_STATUS_VALID_FLAT_SINGLE_STATE,
            fallback_status=FALLBACK_STATUS_NEUTRAL_FLAT_FALLBACK,
            profile_id=profile_id,
            feature_pool_id=feature_pool_id,
            clusterer_family="none",
            assignment_policy="fallback_constant",
            refit_key="no_refit_flat_fallback",
            description_metadata={
                "row_role": "neutral_flat_fallback_label",
                **dict(description_metadata or {}),
            },
            run_id=run_id,
            created_at=created_at,
        )
        for ts in timestamps
    )


def insufficient_data_no_label_output_row(
    *,
    asset: str,
    ts: int | float | str,
    axis: str | AssetStateAxis,
    band: str | AssetStateBand,
    interval: int,
    clusterability_status: str = CLUSTERABILITY_STATUS_INSUFFICIENT_HISTORY,
    run_id: str = "asset_state_sandbox_run",
    created_at: str | None = None,
    description_metadata: Mapping[str, Any] | None = None,
) -> AssetStateSandboxOutputRow:
    if clusterability_status not in {CLUSTERABILITY_STATUS_INSUFFICIENT_HISTORY, CLUSTERABILITY_STATUS_INSUFFICIENT_FINITE_ROWS}:
        raise ValueError("Insufficient-data no-label rows require an insufficient-history or insufficient-finite clusterability status")
    return AssetStateSandboxOutputRow(
        asset=asset,
        ts=ts,
        axis=axis,
        band=band,
        interval=interval,
        regime_id=None,
        regime_label=None,
        regime_label_source=LABEL_SOURCE_INSUFFICIENT_DATA_NO_LABEL,
        state_strength=None,
        clusterability_status=clusterability_status,
        fallback_status=FALLBACK_STATUS_INSUFFICIENT_DATA_NO_LABEL,
        profile_id="no_profile_insufficient_data",
        feature_pool_id="no_feature_pool_insufficient_data",
        clusterer_family="none",
        assignment_policy="no_assignment_insufficient_data",
        refit_key="no_refit_insufficient_data",
        description_metadata={
            "row_role": "insufficient_data_no_label",
            **dict(description_metadata or {}),
        },
        run_id=run_id,
        created_at=created_at,
    )


def axis_not_applicable_output_row(
    *,
    asset: str,
    ts: int | float | str,
    axis: str | AssetStateAxis,
    band: str | AssetStateBand,
    interval: int,
    run_id: str = "asset_state_sandbox_run",
    created_at: str | None = None,
    description_metadata: Mapping[str, Any] | None = None,
) -> AssetStateSandboxOutputRow:
    return AssetStateSandboxOutputRow(
        asset=asset,
        ts=ts,
        axis=axis,
        band=band,
        interval=interval,
        regime_id=None,
        regime_label=None,
        regime_label_source=LABEL_SOURCE_AXIS_NOT_APPLICABLE,
        state_strength=None,
        clusterability_status=CLUSTERABILITY_STATUS_AXIS_NOT_CLUSTERABLE,
        fallback_status=FALLBACK_STATUS_AXIS_NOT_APPLICABLE,
        profile_id="no_profile_axis_not_applicable",
        feature_pool_id="no_feature_pool_axis_not_applicable",
        clusterer_family="none",
        assignment_policy="no_assignment_axis_not_applicable",
        refit_key="no_refit_axis_not_applicable",
        description_metadata={
            "row_role": "axis_not_applicable_no_label",
            **dict(description_metadata or {}),
        },
        run_id=run_id,
        created_at=created_at,
    )


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
            raise ValueError("Clustered asset-state output rows require regime_id, regime_label, and state_strength")
        if clusterability_status != CLUSTERABILITY_STATUS_CLUSTERABLE_CANDIDATE or fallback_status != FALLBACK_STATUS_NO_FALLBACK_NEEDED:
            raise ValueError("Clustered asset-state output rows require clusterable_candidate/no_fallback_needed")
    elif label_source == LABEL_SOURCE_NEUTRAL_FLAT_FALLBACK:
        if regime_id is None or regime_label is None or state_strength is None:
            raise ValueError("Neutral flat fallback rows require fallback regime_id, regime_label, and state_strength")
        if clusterability_status != CLUSTERABILITY_STATUS_VALID_FLAT_SINGLE_STATE or fallback_status != FALLBACK_STATUS_NEUTRAL_FLAT_FALLBACK:
            raise ValueError("Neutral flat fallback rows require valid_flat_single_state/neutral_flat_fallback")
    elif label_source == LABEL_SOURCE_INSUFFICIENT_DATA_NO_LABEL:
        if regime_id is not None or regime_label is not None or state_strength is not None:
            raise ValueError("Insufficient-data rows must not emit regime labels or state strength")
        if fallback_status != FALLBACK_STATUS_INSUFFICIENT_DATA_NO_LABEL:
            raise ValueError("Insufficient-data rows require insufficient_data_no_label fallback status")
    elif label_source == LABEL_SOURCE_AXIS_NOT_APPLICABLE:
        if regime_id is not None or regime_label is not None or state_strength is not None:
            raise ValueError("Axis-not-applicable rows must not emit regime labels or state strength")
        if fallback_status != FALLBACK_STATUS_AXIS_NOT_APPLICABLE:
            raise ValueError("Axis-not-applicable rows require axis_not_applicable fallback status")


def _label_source(value: object) -> str:
    text = _non_empty_text(value, field_name="regime_label_source").lower()
    if text not in ASSET_STATE_LABEL_SOURCES:
        raise ValueError(f"Unsupported asset-state regime_label_source {text!r}; expected one of: {', '.join(ASSET_STATE_LABEL_SOURCES)}")
    return text


def _clusterability_status(value: object) -> str:
    text = _non_empty_text(value, field_name="clusterability_status").lower()
    if text not in CLUSTERABILITY_STATUSES:
        raise ValueError(f"Unsupported asset-state clusterability_status {text!r}")
    return text


def _fallback_status(value: object) -> str:
    text = _non_empty_text(value, field_name="fallback_status").lower()
    if text not in CLUSTERABILITY_FALLBACK_STATUSES:
        raise ValueError(f"Unsupported asset-state fallback_status {text!r}")
    return text


def _state_strength(value: object, *, label_source: str) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except Exception as exc:
        raise ValueError("Asset-state state_strength must be numeric when supplied") from exc
    if out < 0.0 or out > 1.0:
        raise ValueError("Asset-state state_strength must be between 0 and 1")
    return out


def _created_at(value: object) -> str:
    if value is None:
        return datetime.now(timezone.utc).isoformat()
    return _non_empty_text(value, field_name="created_at")


def _description_metadata(value: Mapping[str, Any] | None) -> dict[str, Any]:
    metadata = to_jsonable(_mapping(value, field_name="description_metadata") if value is not None else {})
    if "regime_lineage" in metadata:
        metadata["regime_lineage"] = RegimeLineageSpec.from_dict(metadata["regime_lineage"]).as_dict()
    if "known_at" in metadata:
        metadata["known_at"] = KnownAtSpec.from_dict(metadata["known_at"]).as_dict()
    return metadata


def _optional_text(value: object | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _resolve_known_at_ts(value: object | None, *, metadata: Mapping[str, Any], ts: object) -> object:
    if value is not None:
        known_at = value
    elif isinstance(metadata.get("known_at"), Mapping) and metadata["known_at"].get("known_at_ts") is not None:
        known_at = metadata["known_at"]["known_at_ts"]
    else:
        known_at = ts
    if _to_orderable(known_at, field_name="known_at_ts") < _to_orderable(ts, field_name="ts"):
        raise ValueError("Asset-state sandbox output known_at_ts must be >= ts")
    return known_at


def _resolve_source_tail_ts(value: object | None, *, metadata: Mapping[str, Any], known_at_ts: object) -> object:
    if value is not None:
        return value
    if isinstance(metadata.get("known_at"), Mapping) and metadata["known_at"].get("source_tail_ts") is not None:
        return metadata["known_at"]["source_tail_ts"]
    if isinstance(metadata.get("regime_lineage"), Mapping) and metadata["regime_lineage"].get("source_tail_ts") is not None:
        return metadata["regime_lineage"]["source_tail_ts"]
    return known_at_ts


def _resolve_lineage_id(value: object | None, *, metadata: Mapping[str, Any], row: AssetStateSandboxOutputRow) -> str:
    explicit = _optional_text(value)
    if explicit is not None:
        return explicit
    lineage = metadata.get("regime_lineage")
    if isinstance(lineage, Mapping):
        run_id = _optional_text(lineage.get("run_id"))
        if run_id is not None:
            return run_id
    return f"{row.run_id}:{row.asset}:{row.axis}:{row.band}:{int(row.interval)}:{row.refit_key}"


def _to_orderable(value: object, *, field_name: str) -> float:
    if value is None or isinstance(value, bool):
        raise ValueError(f"Asset-state sandbox output {field_name} must be timestamp-compatible")
    try:
        return float(value)
    except Exception as exc:
        raise ValueError(f"Asset-state sandbox output {field_name} must be numeric") from exc


def _artifact_boundary(value: Mapping[str, Any] | None) -> dict[str, Any]:
    boundary = non_production_artifact_boundary()
    boundary.update(_mapping(value, field_name="artifact_boundary") if value is not None else {})
    if str(boundary.get("classification", "")).strip().lower() != ASSET_STATE_SANDBOX_OUTPUT_BOUNDARY:
        raise ValueError("Asset-state sandbox output artifact_boundary classification must be sandbox_non_production")
    for key in (
        "production_output",
        "production_parquet_allowed",
        "production_regime_labels_allowed",
        "production_profile_promotion_allowed",
    ):
        if bool(boundary.get(key)):
            raise ValueError(f"Asset-state sandbox output artifact_boundary cannot set {key}=true")
    return to_jsonable(boundary)


__all__ = [
    "ASSET_STATE_LABEL_SOURCES",
    "ASSET_STATE_OUTPUT_PARTITION_FIELDS",
    "ASSET_STATE_OUTPUT_REQUIRED_FIELDS",
    "ASSET_STATE_SANDBOX_OUTPUT_ARTIFACT_KIND",
    "ASSET_STATE_SANDBOX_OUTPUT_BOUNDARY",
    "LABEL_SOURCE_AXIS_NOT_APPLICABLE",
    "LABEL_SOURCE_CLUSTERED",
    "LABEL_SOURCE_INSUFFICIENT_DATA_NO_LABEL",
    "LABEL_SOURCE_NEUTRAL_FLAT_FALLBACK",
    "AssetStateSandboxOutputRow",
    "AssetStateSandboxOutputSchema",
    "axis_not_applicable_output_row",
    "clustered_asset_state_output_rows",
    "coerce_asset_state_output_row",
    "default_asset_state_sandbox_output_schema",
    "insufficient_data_no_label_output_row",
    "neutral_flat_fallback_output_rows",
    "non_production_artifact_boundary",
    "validate_asset_state_output_rows",
]
