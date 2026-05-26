from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from src.regimes.core.contracts import require_schema_version
from src.regimes.core.serialization import dumps_json, to_jsonable
from src.regimes.regime_features.cross_asset_feature_catalog import (
    CROSS_ASSET_RELATIONSHIP_FEATURE_SCHEMA_VERSION,
    DIAGNOSTIC_PROCESS2_FEATURES,
    RELATIONSHIP_FEATURE_CATALOG_ID,
    V1_FEATURE_FIELDS,
    default_cross_asset_relationship_feature_catalog,
)


CROSS_ASSET_FEATURE_ROWS_ARTIFACT_KIND = "cross_asset_feature_rows"
CROSS_ASSET_FEATURE_MANIFEST_ARTIFACT_KIND = "cross_asset_feature_manifest"
CROSS_ASSET_FEATURE_ROW_GRAIN = "one row per asset/refit_key/interval/window"
CROSS_ASSET_FEATURE_PATHWAY = "cross_asset"

SIDE_CAR_AVAILABILITY_FIELDS: tuple[str, ...] = (
    "strongest_peer_slot_1_alias_available",
    "strongest_peer_slot_2_available",
    "volatility_neighborhood_score_available",
    "residual_return_vs_core_available",
)

CROSS_ASSET_FEATURE_ROW_REQUIRED_FIELDS: tuple[str, ...] = (
    "pathway",
    "asset",
    "refit_key",
    "interval",
    "window",
    "effective_start_ts",
    "effective_end_ts",
    "known_at_ts",
    "source_tail_ts",
    "handoff_id",
    "lineage_id",
    "feature_catalog_id",
    "feature_family",
    *V1_FEATURE_FIELDS,
    *DIAGNOSTIC_PROCESS2_FEATURES,
    *SIDE_CAR_AVAILABILITY_FIELDS,
    "schema_version",
    "artifact_boundary",
)


@dataclass(frozen=True)
class CrossAssetRelationshipFeatureRow:
    asset: str
    refit_key: str
    interval: int
    window: int
    effective_start_ts: int | float | str
    effective_end_ts: int | float | str
    known_at_ts: int | float | str
    source_tail_ts: int | float | str
    handoff_id: str
    feature_family: str
    corr_to_anchor_primary: float
    corr_to_anchor_secondary: float
    corr_to_core_basket: float
    beta_to_core_basket: float
    market_mode_exposure_score: float
    isolated_asset_score: float
    peer_signal_availability_status: str
    stable_edge_count: int
    candidate_edge_count: int
    residual_peer_signal_score: float
    relationship_concentration: float
    relationship_entropy: float
    top_peer_count: int
    top_peer_stability_mean: float
    strongest_peer_slot_1_strength: float
    strongest_peer_slot_1_alias_available: bool = False
    strongest_peer_slot_2_available: bool = False
    volatility_neighborhood_score_available: bool = False
    residual_return_vs_core_available: bool = False
    feature_catalog_id: str = RELATIONSHIP_FEATURE_CATALOG_ID
    schema_version: int = CROSS_ASSET_RELATIONSHIP_FEATURE_SCHEMA_VERSION
    artifact_kind: str = CROSS_ASSET_FEATURE_ROWS_ARTIFACT_KIND
    pathway: str = CROSS_ASSET_FEATURE_PATHWAY
    lineage_id: str = "cross_asset_relationship_v1_lineage_unspecified"

    def __post_init__(self) -> None:
        object.__setattr__(self, "asset", _text(self.asset, field_name="asset"))
        object.__setattr__(self, "refit_key", _text(self.refit_key, field_name="refit_key"))
        object.__setattr__(self, "interval", _positive_int(self.interval, field_name="interval"))
        object.__setattr__(self, "window", _positive_int(self.window, field_name="window"))
        object.__setattr__(self, "handoff_id", _text(self.handoff_id, field_name="handoff_id"))
        pathway = _text(self.pathway, field_name="pathway")
        if pathway != CROSS_ASSET_FEATURE_PATHWAY:
            raise ValueError("Cross-Asset feature row pathway must be cross_asset")
        object.__setattr__(self, "pathway", pathway)
        object.__setattr__(self, "lineage_id", _text(self.lineage_id, field_name="lineage_id"))
        object.__setattr__(self, "feature_catalog_id", _text(self.feature_catalog_id, field_name="feature_catalog_id"))
        object.__setattr__(self, "feature_family", _text(self.feature_family, field_name="feature_family"))
        object.__setattr__(self, "schema_version", require_schema_version(self.schema_version))
        object.__setattr__(self, "artifact_kind", _text(self.artifact_kind, field_name="artifact_kind"))
        if _to_orderable(self.effective_end_ts, field_name="effective_end_ts") < _to_orderable(self.effective_start_ts, field_name="effective_start_ts"):
            raise ValueError("Cross-Asset feature row effective_end_ts must be >= effective_start_ts")
        if _to_orderable(self.source_tail_ts, field_name="source_tail_ts") > _to_orderable(self.known_at_ts, field_name="known_at_ts"):
            raise ValueError("Cross-Asset feature row source_tail_ts must not exceed known_at_ts")
        for name in (
            "corr_to_anchor_primary",
            "corr_to_anchor_secondary",
            "corr_to_core_basket",
            "beta_to_core_basket",
            "market_mode_exposure_score",
            "isolated_asset_score",
            "residual_peer_signal_score",
            "relationship_concentration",
            "relationship_entropy",
            "top_peer_stability_mean",
            "strongest_peer_slot_1_strength",
        ):
            object.__setattr__(self, name, _finite_float(getattr(self, name), field_name=name))
        for name in ("stable_edge_count", "candidate_edge_count", "top_peer_count"):
            object.__setattr__(self, name, _nonnegative_int(getattr(self, name), field_name=name))
        object.__setattr__(self, "peer_signal_availability_status", _text(self.peer_signal_availability_status, field_name="peer_signal_availability_status"))

    @property
    def artifact_boundary(self) -> dict[str, bool]:
        return {
            "production_enabled": False,
            "production_outputs_written": False,
            "broad_all_to_all": False,
            "cross_asset_labels_written": False,
            "one_column_per_related_asset_allowed": False,
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            "artifact_kind": self.artifact_kind,
            "pathway": self.pathway,
            "asset": self.asset,
            "refit_key": self.refit_key,
            "interval": int(self.interval),
            "window": int(self.window),
            "effective_start_ts": self.effective_start_ts,
            "effective_end_ts": self.effective_end_ts,
            "known_at_ts": self.known_at_ts,
            "source_tail_ts": self.source_tail_ts,
            "handoff_id": self.handoff_id,
            "lineage_id": self.lineage_id,
            "feature_catalog_id": self.feature_catalog_id,
            "feature_family": self.feature_family,
            "corr_to_anchor_primary": float(self.corr_to_anchor_primary),
            "corr_to_anchor_secondary": float(self.corr_to_anchor_secondary),
            "corr_to_core_basket": float(self.corr_to_core_basket),
            "beta_to_core_basket": float(self.beta_to_core_basket),
            "market_mode_exposure_score": float(self.market_mode_exposure_score),
            "isolated_asset_score": float(self.isolated_asset_score),
            "peer_signal_availability_status": self.peer_signal_availability_status,
            "stable_edge_count": int(self.stable_edge_count),
            "candidate_edge_count": int(self.candidate_edge_count),
            "residual_peer_signal_score": float(self.residual_peer_signal_score),
            "relationship_concentration": float(self.relationship_concentration),
            "relationship_entropy": float(self.relationship_entropy),
            "top_peer_count": int(self.top_peer_count),
            "top_peer_stability_mean": float(self.top_peer_stability_mean),
            "strongest_peer_slot_1_strength": float(self.strongest_peer_slot_1_strength),
            "strongest_peer_slot_1_alias_available": bool(self.strongest_peer_slot_1_alias_available),
            "strongest_peer_slot_2_available": bool(self.strongest_peer_slot_2_available),
            "volatility_neighborhood_score_available": bool(self.volatility_neighborhood_score_available),
            "residual_return_vs_core_available": bool(self.residual_return_vs_core_available),
            "schema_version": int(self.schema_version),
            "artifact_boundary": self.artifact_boundary,
            "production_enabled": False,
        }

    def to_json(self) -> str:
        return dumps_json(self.as_dict())


@dataclass(frozen=True)
class CrossAssetRelationshipFeatureManifest:
    feature_manifest_id: str
    handoff_id: str
    input_artifacts: Mapping[str, Any]
    output_paths: Mapping[str, Any]
    interval: int
    window: int
    refit_key: str
    known_at_ts: int | float | str
    output_artifact_refs: Mapping[str, Any] = field(default_factory=dict)
    feature_catalog_id: str = RELATIONSHIP_FEATURE_CATALOG_ID
    row_grain: str = CROSS_ASSET_FEATURE_ROW_GRAIN
    schema_version: int = CROSS_ASSET_RELATIONSHIP_FEATURE_SCHEMA_VERSION
    artifact_kind: str = CROSS_ASSET_FEATURE_MANIFEST_ARTIFACT_KIND

    def __post_init__(self) -> None:
        object.__setattr__(self, "feature_manifest_id", _text(self.feature_manifest_id, field_name="feature_manifest_id"))
        object.__setattr__(self, "handoff_id", _text(self.handoff_id, field_name="handoff_id"))
        object.__setattr__(self, "feature_catalog_id", _text(self.feature_catalog_id, field_name="feature_catalog_id"))
        object.__setattr__(self, "row_grain", _text(self.row_grain, field_name="row_grain"))
        object.__setattr__(self, "interval", _positive_int(self.interval, field_name="interval"))
        object.__setattr__(self, "window", _positive_int(self.window, field_name="window"))
        object.__setattr__(self, "refit_key", _text(self.refit_key, field_name="refit_key"))
        object.__setattr__(self, "schema_version", require_schema_version(self.schema_version))
        object.__setattr__(self, "artifact_kind", _text(self.artifact_kind, field_name="artifact_kind"))
        _to_orderable(self.known_at_ts, field_name="known_at_ts")

    def as_dict(self) -> dict[str, Any]:
        return {
            "artifact_kind": self.artifact_kind,
            "feature_manifest_id": self.feature_manifest_id,
            "handoff_id": self.handoff_id,
            "input_artifacts": to_jsonable(dict(self.input_artifacts)),
            "output_paths": to_jsonable(dict(self.output_paths)),
            "output_artifact_refs": to_jsonable(dict(self.output_artifact_refs)),
            "feature_catalog_id": self.feature_catalog_id,
            "row_grain": self.row_grain,
            "interval": int(self.interval),
            "window": int(self.window),
            "refit_key": self.refit_key,
            "known_at_ts": self.known_at_ts,
            "schema_version": int(self.schema_version),
            "production_enabled": False,
            "cross_asset_regime_classification_performed": False,
            "one_column_per_related_asset_allowed": False,
        }


def cross_asset_feature_rows_schema() -> dict[str, Any]:
    return {
        "artifact_kind": "cross_asset_feature_rows_schema",
        "schema_version": CROSS_ASSET_RELATIONSHIP_FEATURE_SCHEMA_VERSION,
        "schema_id": CROSS_ASSET_FEATURE_ROWS_ARTIFACT_KIND,
        "row_grain": CROSS_ASSET_FEATURE_ROW_GRAIN,
        "required_fields": list(CROSS_ASSET_FEATURE_ROW_REQUIRED_FIELDS),
        "v1_feature_fields": list(V1_FEATURE_FIELDS),
        "diagnostic_process2_feature_fields": list(DIAGNOSTIC_PROCESS2_FEATURES),
        "sidecar_availability_fields": list(SIDE_CAR_AVAILABILITY_FIELDS),
        "deferred_timestamp_grain_expansion": True,
        "one_column_per_related_asset_allowed": False,
        "production_enabled": False,
    }


def validate_cross_asset_feature_row(row: CrossAssetRelationshipFeatureRow | Mapping[str, Any]) -> None:
    payload = row.as_dict() if isinstance(row, CrossAssetRelationshipFeatureRow) else dict(row)
    _reject_dynamic_peer_identity_columns(payload)
    missing = [field for field in CROSS_ASSET_FEATURE_ROW_REQUIRED_FIELDS if field not in payload]
    if missing:
        raise ValueError(f"Cross-Asset feature row missing required fields: {missing}")
    if bool(payload.get("production_enabled", False)):
        raise ValueError("Cross-Asset feature row production_enabled must be false")
    if payload.get("pathway") != CROSS_ASSET_FEATURE_PATHWAY:
        raise ValueError("Cross-Asset feature row pathway must be cross_asset")
    boundary = payload.get("artifact_boundary")
    if not isinstance(boundary, Mapping):
        raise ValueError("Cross-Asset feature row artifact_boundary must be a mapping")
    if bool(boundary.get("production_enabled", False)):
        raise ValueError("Cross-Asset feature row artifact_boundary production_enabled must be false")
    if bool(boundary.get("production_outputs_written", False)):
        raise ValueError("Cross-Asset feature row artifact_boundary production_outputs_written must be false")
    if bool(boundary.get("broad_all_to_all", False)):
        raise ValueError("Cross-Asset feature row artifact_boundary broad_all_to_all must be false")
    if bool(boundary.get("cross_asset_labels_written", False)):
        raise ValueError("Cross-Asset feature row must not include Cross-Asset regime labels")
    if bool(boundary.get("one_column_per_related_asset_allowed", False)):
        raise ValueError("Cross-Asset feature row must not allow one-column-per-related-asset schema")
    for field in V1_FEATURE_FIELDS:
        if field == "top_peer_count":
            _nonnegative_int(payload[field], field_name=field)
        else:
            _finite_float(payload[field], field_name=field)
    for field in DIAGNOSTIC_PROCESS2_FEATURES:
        if field == "peer_signal_availability_status":
            _text(payload[field], field_name=field)
        elif field in {"stable_edge_count", "candidate_edge_count"}:
            _nonnegative_int(payload[field], field_name=field)
        else:
            _finite_float(payload[field], field_name=field)
    if _to_orderable(payload["source_tail_ts"], field_name="source_tail_ts") > _to_orderable(payload["known_at_ts"], field_name="known_at_ts"):
        raise ValueError("Cross-Asset feature row source_tail_ts must not exceed known_at_ts")


def validate_cross_asset_feature_manifest(manifest: CrossAssetRelationshipFeatureManifest | Mapping[str, Any]) -> None:
    payload = manifest.as_dict() if isinstance(manifest, CrossAssetRelationshipFeatureManifest) else dict(manifest)
    for field in (
        "feature_manifest_id",
        "handoff_id",
        "input_artifacts",
        "output_paths",
        "feature_catalog_id",
        "row_grain",
        "interval",
        "window",
        "refit_key",
        "known_at_ts",
        "schema_version",
    ):
        if field not in payload:
            raise ValueError(f"Cross-Asset feature manifest missing required field {field!r}")
    if bool(payload.get("production_enabled", False)):
        raise ValueError("Cross-Asset feature manifest production_enabled must be false")
    if bool(payload.get("one_column_per_related_asset_allowed", False)):
        raise ValueError("Cross-Asset feature manifest must not allow one-column-per-related-asset schema")


def build_cross_asset_feature_row_from_process1_profile(
    profile_row: Mapping[str, Any],
    *,
    handoff_id: str,
    effective_start_ts: int | float | str,
    effective_end_ts: int | float | str,
    source_tail_ts: int | float | str,
    feature_catalog_id: str = RELATIONSHIP_FEATURE_CATALOG_ID,
) -> CrossAssetRelationshipFeatureRow:
    catalog = default_cross_asset_relationship_feature_catalog()
    catalog.validate()
    return CrossAssetRelationshipFeatureRow(
        asset=str(profile_row["asset"]),
        refit_key=str(profile_row["refit_key"]),
        interval=int(profile_row["interval"]),
        window=int(profile_row["window"]),
        effective_start_ts=effective_start_ts,
        effective_end_ts=effective_end_ts,
        known_at_ts=profile_row["known_at_ts"],
        source_tail_ts=source_tail_ts,
        handoff_id=handoff_id,
        lineage_id=str(profile_row.get("lineage_id", f"{handoff_id}:cross_asset_relationship_v1")),
        feature_catalog_id=feature_catalog_id,
        feature_family="cross_asset_relationship_v1",
        corr_to_anchor_primary=float(profile_row.get("corr_to_anchor_primary", 0.0)),
        corr_to_anchor_secondary=float(profile_row.get("corr_to_anchor_secondary", 0.0)),
        corr_to_core_basket=float(profile_row.get("corr_to_core_basket", 0.0)),
        beta_to_core_basket=float(profile_row.get("beta_to_core_basket", 0.0)),
        market_mode_exposure_score=float(profile_row.get("market_mode_exposure_score", 0.0)),
        isolated_asset_score=float(profile_row.get("isolated_asset_score", 1.0)),
        peer_signal_availability_status=str(profile_row.get("peer_signal_availability_status", "unavailable")),
        stable_edge_count=int(profile_row.get("stable_edge_count", profile_row.get("top_peer_count", 0))),
        candidate_edge_count=int(profile_row.get("candidate_edge_count", profile_row.get("top_peer_count", 0))),
        residual_peer_signal_score=float(profile_row.get("residual_peer_signal_score", 0.0)),
        relationship_concentration=float(profile_row.get("relationship_concentration", 0.0)),
        relationship_entropy=float(profile_row.get("relationship_entropy", 0.0)),
        top_peer_count=int(profile_row.get("top_peer_count", 0)),
        top_peer_stability_mean=float(profile_row.get("top_peer_stability_mean", 0.0)),
        strongest_peer_slot_1_strength=float(profile_row.get("strongest_peer_slot_1_strength", profile_row.get("residual_peer_signal_score", 0.0))),
        strongest_peer_slot_1_alias_available=bool(profile_row.get("strongest_peer_slot_1_alias_available", False)),
        strongest_peer_slot_2_available=bool(profile_row.get("strongest_peer_slot_2_available", False)),
        volatility_neighborhood_score_available=bool(profile_row.get("volatility_neighborhood_score_available", False)),
        residual_return_vs_core_available=bool(profile_row.get("residual_return_vs_core_available", False)),
    )


def _text(value: object, *, field_name: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"Cross-Asset feature row {field_name} must be non-empty")
    return text


def _positive_int(value: object, *, field_name: str) -> int:
    out = _nonnegative_int(value, field_name=field_name)
    if out <= 0:
        raise ValueError(f"Cross-Asset feature row {field_name} must be positive")
    return out


def _nonnegative_int(value: object, *, field_name: str) -> int:
    try:
        out = int(value)
    except Exception as exc:
        raise ValueError(f"Cross-Asset feature row {field_name} must be an integer") from exc
    if out < 0:
        raise ValueError(f"Cross-Asset feature row {field_name} must be non-negative")
    return out


def _finite_float(value: object, *, field_name: str) -> float:
    try:
        out = float(value)
    except Exception as exc:
        raise ValueError(f"Cross-Asset feature row {field_name} must be numeric") from exc
    if out != out or out in {float("inf"), float("-inf")}:
        raise ValueError(f"Cross-Asset feature row {field_name} must be finite")
    return out


def _to_orderable(value: object, *, field_name: str) -> float:
    if value is None or isinstance(value, bool):
        raise ValueError(f"Cross-Asset feature row {field_name} must be timestamp-compatible")
    try:
        return float(value)
    except Exception as exc:
        raise ValueError(f"Cross-Asset feature row {field_name} must be numeric") from exc


def _reject_dynamic_peer_identity_columns(payload: Mapping[str, Any]) -> None:
    allowed_peer_slots = {
        "peer_signal_availability_status",
        "strongest_peer_slot_1_strength",
        "strongest_peer_slot_1_alias_available",
        "strongest_peer_slot_2_available",
    }
    for name in payload:
        lower = str(name).lower()
        if name in allowed_peer_slots:
            continue
        if lower.startswith(("peer_", "related_asset_", "related_")):
            raise ValueError("Cross-Asset feature rows must not include one-column-per-related-asset fields")
        if lower.endswith(("_peer_alias", "_peer_strength")):
            raise ValueError("Cross-Asset feature rows must use stable slot fields for peer identity metadata")


__all__ = [
    "CROSS_ASSET_FEATURE_MANIFEST_ARTIFACT_KIND",
    "CROSS_ASSET_FEATURE_ROWS_ARTIFACT_KIND",
    "CROSS_ASSET_FEATURE_ROW_GRAIN",
    "CROSS_ASSET_FEATURE_ROW_REQUIRED_FIELDS",
    "SIDE_CAR_AVAILABILITY_FIELDS",
    "CrossAssetRelationshipFeatureManifest",
    "CrossAssetRelationshipFeatureRow",
    "build_cross_asset_feature_row_from_process1_profile",
    "cross_asset_feature_rows_schema",
    "validate_cross_asset_feature_manifest",
    "validate_cross_asset_feature_row",
]
