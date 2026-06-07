from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.regimes.cross_asset_state.dataset_builder import (
    RelationshipValueAvailabilityIndex,
    build_relationship_value_availability_index,
    load_relationship_value_availability,
)
from src.regimes.cross_asset_state.feature_families import (
    CROSS_ASSET_STATE_SCHEMA_VERSION,
    default_feature_family_map,
)
from src.regimes.cross_asset_state.mask_contract import CrossAssetStateMaskReason
from src.regimes.cross_asset_state.relationship_registry import (
    RelationshipAvailabilityRecord,
    RelationshipSnapshotIndex,
    RelationshipSnapshotResolver,
    build_relationship_snapshot_index,
)


CROSS_ASSET_RELATIONSHIP_CONTEXT_SCHEMA_VERSION = 1
CROSS_ASSET_RELATIONSHIP_CONTEXT_ARTIFACT_KIND = "cross_asset_relationship_context_handoff"
DEFAULT_RELATIONSHIP_CONTEXT_CADENCE_POLICY_ID = "cross_asset_relationship_context_cadence_v1_meso14_macro30"
DEFAULT_SNAPSHOT_CADENCE_DAYS: Mapping[str, int] = {"meso": 14, "macro": 30}
DEFAULT_STALE_AFTER_DAYS: Mapping[str, int] = {"meso": 21, "macro": 45}
DEFAULT_STALE_SNAPSHOT_POLICY_ID = "mask_after_snapshot_cadence_plus_grace_v1"


@dataclass(frozen=True)
class CrossAssetRelationshipContextHandoff:
    relationship_context_id: str
    relationship_snapshot_roots: Sequence[str | Path]
    feature_roots: Sequence[str | Path]
    availability_sidecar_refs: Sequence[str | Path] = ()
    peer_metadata_refs: Sequence[Mapping[str, Any]] = ()
    future_outcome_panel_refs: Sequence[str | Path] = ()
    feature_set_version: str | None = None
    relationship_context_cadence_policy_id: str = DEFAULT_RELATIONSHIP_CONTEXT_CADENCE_POLICY_ID
    snapshot_cadence_days: Mapping[str, int] = field(default_factory=lambda: dict(DEFAULT_SNAPSHOT_CADENCE_DAYS))
    backfill_snapshot_schedule: Sequence[Mapping[str, Any]] = ()
    stale_snapshot_policy: Mapping[str, Any] = field(
        default_factory=lambda: {
            "policy_id": DEFAULT_STALE_SNAPSHOT_POLICY_ID,
            "stale_after_days_by_band": dict(DEFAULT_STALE_AFTER_DAYS),
            "action": "mask_unavailable",
            "mask_reason": CrossAssetStateMaskReason.STALE_RELATIONSHIP_SNAPSHOT,
        }
    )
    missing_snapshot_mask_reason: str = CrossAssetStateMaskReason.MISSING_RELATIONSHIP_SNAPSHOT
    no_future_graph_backfill: bool = True
    regime_feature_manifest_id: str | None = None
    created_at_utc: str | None = None
    schema_version: int = CROSS_ASSET_RELATIONSHIP_CONTEXT_SCHEMA_VERSION
    active_nonproduction_handoff: bool = True
    production_enabled: bool = False
    production_outputs_written: bool = False
    canonical_production_state_outputs_written: bool = False

    def __post_init__(self) -> None:
        if self.production_enabled or self.production_outputs_written or self.canonical_production_state_outputs_written:
            raise ValueError("Cross-Asset relationship context handoff is non-production only")
        if not self.active_nonproduction_handoff:
            raise ValueError("Cross-Asset relationship context handoff must be active non-production")
        object.__setattr__(self, "relationship_context_id", _text(self.relationship_context_id, "relationship_context_id"))
        object.__setattr__(self, "relationship_snapshot_roots", tuple(str(root) for root in self.relationship_snapshot_roots))
        object.__setattr__(self, "feature_roots", tuple(str(root) for root in self.feature_roots))
        object.__setattr__(self, "availability_sidecar_refs", tuple(str(root) for root in self.availability_sidecar_refs))
        object.__setattr__(self, "peer_metadata_refs", tuple(dict(ref) for ref in self.peer_metadata_refs))
        object.__setattr__(self, "future_outcome_panel_refs", tuple(str(ref) for ref in self.future_outcome_panel_refs))
        object.__setattr__(self, "feature_set_version", _optional_text(self.feature_set_version))
        cadence_policy_id = _optional_text(self.relationship_context_cadence_policy_id) or DEFAULT_RELATIONSHIP_CONTEXT_CADENCE_POLICY_ID
        object.__setattr__(self, "relationship_context_cadence_policy_id", _text(cadence_policy_id, "relationship_context_cadence_policy_id"))
        object.__setattr__(self, "snapshot_cadence_days", _cadence_days(self.snapshot_cadence_days or DEFAULT_SNAPSHOT_CADENCE_DAYS))
        object.__setattr__(self, "backfill_snapshot_schedule", tuple(dict(item) for item in self.backfill_snapshot_schedule))
        object.__setattr__(self, "stale_snapshot_policy", _stale_snapshot_policy(self.stale_snapshot_policy))
        missing_reason = _text(self.missing_snapshot_mask_reason, "missing_snapshot_mask_reason")
        if missing_reason != CrossAssetStateMaskReason.MISSING_RELATIONSHIP_SNAPSHOT:
            raise ValueError("Cross-Asset relationship context missing_snapshot_mask_reason must fail closed as missing_relationship_snapshot")
        object.__setattr__(self, "missing_snapshot_mask_reason", missing_reason)
        if self.no_future_graph_backfill is not True:
            raise ValueError("Cross-Asset relationship context handoff must guarantee no_future_graph_backfill")
        object.__setattr__(self, "no_future_graph_backfill", True)
        object.__setattr__(self, "regime_feature_manifest_id", _optional_text(self.regime_feature_manifest_id))
        object.__setattr__(self, "created_at_utc", _optional_text(self.created_at_utc))
        object.__setattr__(self, "production_enabled", False)
        object.__setattr__(self, "production_outputs_written", False)
        object.__setattr__(self, "canonical_production_state_outputs_written", False)

    def as_dict(self) -> dict[str, Any]:
        return {
            "artifact_kind": CROSS_ASSET_RELATIONSHIP_CONTEXT_ARTIFACT_KIND,
            "schema_version": int(self.schema_version),
            "relationship_context_id": self.relationship_context_id,
            "relationship_snapshot_roots": list(self.relationship_snapshot_roots),
            "feature_roots": list(self.feature_roots),
            "availability_sidecar_refs": list(self.availability_sidecar_refs),
            "peer_metadata_refs": list(self.peer_metadata_refs),
            "future_outcome_panel_refs": list(self.future_outcome_panel_refs),
            "feature_set_version": self.feature_set_version,
            "relationship_context_cadence_policy_id": self.relationship_context_cadence_policy_id,
            "snapshot_cadence_days": dict(self.snapshot_cadence_days),
            "backfill_snapshot_schedule": list(self.backfill_snapshot_schedule),
            "stale_snapshot_policy": dict(self.stale_snapshot_policy),
            "missing_snapshot_mask_reason": self.missing_snapshot_mask_reason,
            "no_future_graph_backfill": True,
            "live_refresh_policy": {
                "policy_id": self.relationship_context_cadence_policy_id,
                "refresh_cadence_days": dict(self.snapshot_cadence_days),
                "consume_policy": "most_recent_valid_known_at_snapshot_subject_to_staleness",
                "production_labels_write_gate": "fail_closed",
            },
            "regime_feature_manifest_id": self.regime_feature_manifest_id,
            "created_at_utc": self.created_at_utc,
            "required_identity_lineage_fields": [
                "relationship_context_id",
                "relationship_snapshot_id",
                "asset_id",
                "band",
                "known_at_ts",
                "source_tail_ts",
                "relationship_discovery_run_id",
                "regime_feature_manifest_id",
                "availability_sidecar_refs",
                "peer_metadata_refs",
                "future_outcome_panel_refs",
                "feature_set_version",
                "relationship_context_cadence_policy_id",
                "snapshot_cadence_days",
                "snapshot_valid_from_ts",
                "snapshot_valid_until_ts",
                "no_future_graph_backfill",
            ],
            "availability_semantics": [
                "value_status",
                "mask_reason",
                "field_level_availability",
                "family_level_availability",
                "row_cell_level_availability",
                "shape_preserving_unavailable_records",
            ],
            "active_nonproduction_handoff": True,
            "production_enabled": False,
            "production_outputs_written": False,
            "canonical_production_state_outputs_written": False,
        }

    def cadence_days_for_band(self, band: str) -> int | None:
        value = self.snapshot_cadence_days.get(str(band).lower())
        return int(value) if value is not None else None

    def stale_after_days_for_band(self, band: str) -> int | None:
        raw = dict(self.stale_snapshot_policy.get("stale_after_days_by_band") or {})
        value = raw.get(str(band).lower())
        return int(value) if value is not None else None

    def cadence_policy_as_dict(self) -> dict[str, Any]:
        return {
            "relationship_context_cadence_policy_id": self.relationship_context_cadence_policy_id,
            "snapshot_cadence_days": dict(self.snapshot_cadence_days),
            "backfill_snapshot_schedule": list(self.backfill_snapshot_schedule),
            "stale_snapshot_policy": dict(self.stale_snapshot_policy),
            "missing_snapshot_mask_reason": self.missing_snapshot_mask_reason,
            "no_future_graph_backfill": True,
        }

    def write_json(self, path: str | Path) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(self.as_dict(), indent=2, sort_keys=True), encoding="utf-8")
        return out


@dataclass(frozen=True)
class CrossAssetRelationshipContextRecord:
    relationship_context_id: str
    asset_id: str
    band: str
    ts: int | float | str
    context_status: str
    mask_reason: str | None = None
    relationship_snapshot_id: str | None = None
    known_at_ts: int | float | str | None = None
    source_tail_ts: int | float | str | None = None
    relationship_discovery_run_id: str | None = None
    regime_feature_manifest_id: str | None = None
    relationship_context_cadence_policy_id: str | None = None
    snapshot_cadence_days: int | None = None
    stale_snapshot_policy: Mapping[str, Any] = field(default_factory=dict)
    no_future_graph_backfill: bool = True
    snapshot_valid_from_ts: int | float | str | None = None
    snapshot_valid_until_ts: int | float | str | None = None
    availability_sidecar_refs: Sequence[str] = ()
    peer_metadata_refs: Sequence[Mapping[str, Any]] = ()
    field_availability: Sequence[Mapping[str, Any]] = ()
    family_availability: Sequence[Mapping[str, Any]] = ()
    schema_version: int = CROSS_ASSET_RELATIONSHIP_CONTEXT_SCHEMA_VERSION
    production_enabled: bool = False
    canonical_production_state_outputs_written: bool = False

    def __post_init__(self) -> None:
        if self.production_enabled or self.canonical_production_state_outputs_written:
            raise ValueError("Cross-Asset relationship context records are non-production only")
        object.__setattr__(self, "relationship_context_id", _text(self.relationship_context_id, "relationship_context_id"))
        object.__setattr__(self, "asset_id", _text(self.asset_id, "asset_id"))
        object.__setattr__(self, "band", _text(self.band, "band").lower())
        object.__setattr__(self, "context_status", _text(self.context_status, "context_status"))
        object.__setattr__(self, "mask_reason", _optional_text(self.mask_reason))
        object.__setattr__(self, "relationship_snapshot_id", _optional_text(self.relationship_snapshot_id))
        object.__setattr__(self, "relationship_discovery_run_id", _optional_text(self.relationship_discovery_run_id))
        object.__setattr__(self, "regime_feature_manifest_id", _optional_text(self.regime_feature_manifest_id))
        object.__setattr__(self, "relationship_context_cadence_policy_id", _optional_text(self.relationship_context_cadence_policy_id))
        object.__setattr__(self, "snapshot_cadence_days", _optional_positive_int(self.snapshot_cadence_days, "snapshot_cadence_days"))
        object.__setattr__(self, "stale_snapshot_policy", dict(self.stale_snapshot_policy or {}))
        object.__setattr__(self, "no_future_graph_backfill", bool(self.no_future_graph_backfill))
        object.__setattr__(self, "availability_sidecar_refs", tuple(str(ref) for ref in self.availability_sidecar_refs))
        object.__setattr__(self, "peer_metadata_refs", tuple(dict(ref) for ref in self.peer_metadata_refs))
        object.__setattr__(self, "field_availability", tuple(dict(record) for record in self.field_availability))
        object.__setattr__(self, "family_availability", tuple(dict(record) for record in self.family_availability))
        object.__setattr__(self, "production_enabled", False)
        object.__setattr__(self, "canonical_production_state_outputs_written", False)

    @property
    def shape_preserving(self) -> bool:
        return True

    def as_dict(self) -> dict[str, Any]:
        return {
            "artifact_kind": "cross_asset_relationship_context_record",
            "schema_version": int(self.schema_version),
            "relationship_context_id": self.relationship_context_id,
            "asset_id": self.asset_id,
            "band": self.band,
            "ts": self.ts,
            "context_status": self.context_status,
            "mask_reason": self.mask_reason,
            "relationship_snapshot_id": self.relationship_snapshot_id,
            "known_at_ts": self.known_at_ts,
            "source_tail_ts": self.source_tail_ts,
            "relationship_discovery_run_id": self.relationship_discovery_run_id,
            "regime_feature_manifest_id": self.regime_feature_manifest_id,
            "relationship_context_cadence_policy_id": self.relationship_context_cadence_policy_id,
            "snapshot_cadence_days": self.snapshot_cadence_days,
            "stale_snapshot_policy": dict(self.stale_snapshot_policy),
            "no_future_graph_backfill": bool(self.no_future_graph_backfill),
            "snapshot_valid_from_ts": self.snapshot_valid_from_ts,
            "snapshot_valid_until_ts": self.snapshot_valid_until_ts,
            "availability_sidecar_refs": list(self.availability_sidecar_refs),
            "peer_metadata_refs": list(self.peer_metadata_refs),
            "field_availability": list(self.field_availability),
            "family_availability": list(self.family_availability),
            "shape_preserving": True,
            "peer_metadata_model_facing": False,
            "production_enabled": False,
            "canonical_production_state_outputs_written": False,
        }


@dataclass(frozen=True)
class CrossAssetRelationshipContextResolver:
    handoff: CrossAssetRelationshipContextHandoff
    snapshot_resolver: RelationshipSnapshotResolver
    availability_frame: Any
    availability_index: RelationshipValueAvailabilityIndex | None = None

    @classmethod
    def from_handoff(cls, handoff: CrossAssetRelationshipContextHandoff, *, max_parquet_files_per_root: int | None = None) -> "CrossAssetRelationshipContextResolver":
        if not handoff.availability_sidecar_refs:
            raise ValueError("Cross-Asset relationship context handoff requires availability_sidecar_refs")
        availability = load_relationship_value_availability(handoff.availability_sidecar_refs)
        if availability is None or availability.empty:
            raise ValueError("Cross-Asset relationship context handoff availability sidecar is missing or empty")
        index = build_relationship_snapshot_index(
            (*handoff.relationship_snapshot_roots, *handoff.feature_roots),
            max_parquet_files_per_root=max_parquet_files_per_root,
        )
        availability_index = build_relationship_value_availability_index(availability)
        return cls(
            handoff=handoff,
            snapshot_resolver=RelationshipSnapshotResolver(index),
            availability_frame=availability,
            availability_index=availability_index,
        )

    def resolve(
        self,
        *,
        asset_id: str,
        band: str,
        ts: int | float | str,
        feature_families: Sequence[str] | None = None,
        feature_family_columns: Mapping[str, Sequence[str]] | None = None,
    ) -> CrossAssetRelationshipContextRecord:
        families = tuple(feature_families or default_feature_family_map().keys())
        family_columns = _family_columns(families, feature_family_columns)
        snapshot_record = self.snapshot_resolver.resolve(asset_id=asset_id, band=band, ts=ts, relationship_feature_family="relationship_context")
        if not snapshot_record.available:
            return self._masked_from_snapshot(snapshot_record, families=families, family_columns=family_columns)
        if self._snapshot_is_stale(snapshot_record):
            return self._masked_from_snapshot(
                snapshot_record,
                families=families,
                family_columns=family_columns,
                reason_override=CrossAssetStateMaskReason.STALE_RELATIONSHIP_SNAPSHOT,
            )
        snap = snapshot_record.as_dict()
        fields = self._field_availability(asset_id=asset_id, band=band, ts=ts, families=families, family_columns=family_columns)
        family_rows = self._family_availability(fields, families=families)
        masked_families = [row for row in family_rows if row["value_status"] == "masked_unavailable"]
        return CrossAssetRelationshipContextRecord(
            relationship_context_id=self._context_id(asset_id=asset_id, band=band, ts=ts, snapshot_id=str(snap["snapshot_id"])),
            asset_id=asset_id,
            band=band,
            ts=ts,
            context_status="masked_unavailable" if masked_families else "available",
            mask_reason=masked_families[0]["mask_reasons"][0] if masked_families and masked_families[0]["mask_reasons"] else None,
            relationship_snapshot_id=snap["snapshot_id"],
            known_at_ts=snap["known_at_ts"],
            source_tail_ts=snap["source_tail_ts"],
            relationship_discovery_run_id=snap["relationship_discovery_run_id"],
            regime_feature_manifest_id=self.handoff.regime_feature_manifest_id,
            relationship_context_cadence_policy_id=self.handoff.relationship_context_cadence_policy_id,
            snapshot_cadence_days=self.handoff.cadence_days_for_band(band),
            stale_snapshot_policy=self.handoff.stale_snapshot_policy,
            no_future_graph_backfill=self.handoff.no_future_graph_backfill,
            snapshot_valid_from_ts=snap.get("snapshot_valid_from_ts"),
            snapshot_valid_until_ts=snap.get("snapshot_valid_until_ts"),
            availability_sidecar_refs=self.handoff.availability_sidecar_refs,
            peer_metadata_refs=snap["peer_metadata_refs"] or self.handoff.peer_metadata_refs,
            field_availability=fields,
            family_availability=family_rows,
        )

    def _masked_from_snapshot(
        self,
        snapshot_record: RelationshipAvailabilityRecord,
        *,
        families: Sequence[str],
        family_columns: Mapping[str, tuple[str, ...]],
        reason_override: str | None = None,
    ) -> CrossAssetRelationshipContextRecord:
        snap = snapshot_record.as_dict()
        reason = reason_override or _registry_reason_to_context_reason(str(snap["mask_reason"]))
        family_rows = [
            {
                "relationship_feature_family": family,
                "value_status": "masked_unavailable",
                "mask_reasons": [reason],
                "field_count": len(family_columns[family]),
                "masked_field_count": len(family_columns[family]),
                "model_usable": False,
            }
            for family in families
        ]
        return CrossAssetRelationshipContextRecord(
            relationship_context_id=self._context_id(asset_id=str(snap["asset_id"]), band=str(snap["band"]), ts=snap["ts"], snapshot_id=str(snap.get("snapshot_id") or "masked")),
            asset_id=str(snap["asset_id"]),
            band=str(snap["band"]),
            ts=snap["ts"],
            context_status="masked_unavailable",
            mask_reason=reason,
            relationship_snapshot_id=snap.get("snapshot_id"),
            known_at_ts=snap.get("known_at_ts"),
            source_tail_ts=snap.get("source_tail_ts"),
            relationship_discovery_run_id=snap.get("relationship_discovery_run_id"),
            regime_feature_manifest_id=self.handoff.regime_feature_manifest_id,
            relationship_context_cadence_policy_id=self.handoff.relationship_context_cadence_policy_id,
            snapshot_cadence_days=self.handoff.cadence_days_for_band(str(snap["band"])),
            stale_snapshot_policy=self.handoff.stale_snapshot_policy,
            no_future_graph_backfill=self.handoff.no_future_graph_backfill,
            snapshot_valid_from_ts=snap.get("snapshot_valid_from_ts"),
            snapshot_valid_until_ts=snap.get("snapshot_valid_until_ts"),
            availability_sidecar_refs=self.handoff.availability_sidecar_refs,
            peer_metadata_refs=snap.get("peer_metadata_refs") or (),
            field_availability=(),
            family_availability=family_rows,
        )

    def _field_availability(
        self,
        *,
        asset_id: str,
        band: str,
        ts: int | float | str,
        families: Sequence[str],
        family_columns: Mapping[str, tuple[str, ...]],
    ) -> tuple[dict[str, Any], ...]:
        pd = _pandas()
        out: list[dict[str, Any]] = []
        for family in families:
            for field_name in family_columns[family]:
                if self.availability_index is not None:
                    scoped = self.availability_index.field_availability_rows(
                        asset=str(asset_id),
                        band=str(band),
                        family=str(family),
                        field_name=str(field_name),
                        ts=ts,
                    )
                else:
                    frame = self.availability_frame.copy()
                    frame = frame[frame["asset"].astype(str) == str(asset_id)]
                    frame = frame[frame["band"].astype(str) == str(band)]
                    if "ts" in frame.columns:
                        frame = frame[pd.to_numeric(frame["ts"], errors="coerce") <= float(ts)]
                    scoped = frame[frame["field_name"].astype(str) == str(field_name)] if "field_name" in frame.columns else pd.DataFrame()
                if scoped.empty:
                    out.append(
                        {
                            "field_name": field_name,
                            "relationship_feature_family": family,
                            "value_status": "masked_unavailable",
                            "mask_reason": CrossAssetStateMaskReason.MISSING_REQUIRED_FAMILY_FIELDS,
                            "model_usable": False,
                        }
                    )
                    continue
                usable = (
                    scoped[scoped["value_status"].astype(str).isin({"valid", "neutral_valid"})]
                    if "value_status" in scoped.columns
                    else pd.DataFrame()
                )
                selection_scope = usable if not usable.empty else scoped
                latest_idx = (
                    pd.to_numeric(selection_scope["ts"], errors="coerce").idxmax()
                    if "ts" in selection_scope.columns
                    else selection_scope.index[-1]
                )
                row = dict(selection_scope.loc[latest_idx])
                out.append(
                    {
                        "field_name": field_name,
                        "relationship_feature_family": family,
                        "value_status": str(row.get("value_status")),
                        "mask_reason": _optional_text(row.get("mask_reason")),
                        "model_usable": bool(row.get("value_status") in {"valid", "neutral_valid"}),
                        "numeric_value": row.get("numeric_value"),
                        "known_at_ts": row.get("known_at_ts"),
                        "source_tail_ts": row.get("source_tail_ts"),
                        "availability_selection_policy": "latest_model_usable_row_before_resolution"
                        if not usable.empty
                        else "latest_unavailable_row_before_resolution",
                    }
                )
        return tuple(out)

    def availability_index_stats(self) -> dict[str, Any]:
        if self.availability_index is None:
            return {
                "artifact_kind": "cross_asset_state_availability_index_telemetry",
                "index_built": False,
                "build_count": 0,
                "build_seconds": 0.0,
                "rows_indexed": 0,
                "lookup_count": 0,
                "miss_count": 0,
                "field_lookup_count": 0,
                "field_miss_count": 0,
                "logical_key_fields": [],
                "physical_key_columns": [],
                "optional_scope_fields": [],
            }
        return self.availability_index.stats()

    def _snapshot_is_stale(self, snapshot_record: RelationshipAvailabilityRecord) -> bool:
        snap = snapshot_record.as_dict()
        stale_after_days = self.handoff.stale_after_days_for_band(str(snap.get("band") or ""))
        if stale_after_days is None:
            return False
        known_at = _float_or_none(snap.get("known_at_ts"))
        ts = _float_or_none(snap.get("ts"))
        if known_at is None or ts is None:
            return True
        return ts - known_at > float(stale_after_days) * 86400.0

    @staticmethod
    def _family_availability(fields: Sequence[Mapping[str, Any]], *, families: Sequence[str]) -> tuple[dict[str, Any], ...]:
        out: list[dict[str, Any]] = []
        for family in families:
            rows = [dict(row) for row in fields if row.get("relationship_feature_family") == family]
            masked = [row for row in rows if row.get("value_status") == "masked_unavailable"]
            out.append(
                {
                    "relationship_feature_family": family,
                    "value_status": "masked_unavailable" if masked else "valid",
                    "mask_reasons": sorted({str(row.get("mask_reason")) for row in masked if row.get("mask_reason")}),
                    "field_count": len(rows),
                    "masked_field_count": len(masked),
                    "model_usable": not masked,
                }
            )
        return tuple(out)

    def _context_id(self, *, asset_id: str, band: str, ts: int | float | str, snapshot_id: str) -> str:
        raw = "|".join([self.handoff.relationship_context_id, str(asset_id), str(band), str(ts), str(snapshot_id)])
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def validate_relationship_context_handoff_for_production(handoff: Mapping[str, Any] | str | Path) -> Mapping[str, Any]:
    payload: Mapping[str, Any]
    if isinstance(handoff, (str, Path)):
        loaded = json.loads(Path(handoff).read_text(encoding="utf-8"))
        if not isinstance(loaded, Mapping):
            raise ValueError("Cross-Asset relationship context handoff must be a JSON object")
        payload = loaded
    else:
        payload = handoff
    if payload.get("artifact_kind") != CROSS_ASSET_RELATIONSHIP_CONTEXT_ARTIFACT_KIND:
        raise ValueError("Cross-Asset relationship context production consumer rejects unknown handoff kind")
    if payload.get("production_enabled") is not True or payload.get("canonical_production_state_outputs_written") is not True:
        raise ValueError("Cross-Asset relationship context production gates are closed")
    return payload


def _registry_reason_to_context_reason(reason: str) -> str:
    mapping = {
        "no_relationship_snapshot": CrossAssetStateMaskReason.MISSING_RELATIONSHIP_SNAPSHOT,
        "asset_not_in_snapshot": CrossAssetStateMaskReason.ASSET_NOT_IN_RELATIONSHIP_SNAPSHOT,
        "no_viable_anchor": CrossAssetStateMaskReason.MISSING_ANCHOR,
        "no_core_basket_relationship": CrossAssetStateMaskReason.MISSING_CORE_BASKET,
        "no_viable_peer_edges": CrossAssetStateMaskReason.NO_VIABLE_PEER_EDGES,
        "stale_snapshot": CrossAssetStateMaskReason.STALE_RELATIONSHIP_SNAPSHOT,
        "ambiguous_snapshot": CrossAssetStateMaskReason.AMBIGUOUS_SNAPSHOT_RESOLUTION,
    }
    return mapping.get(reason, CrossAssetStateMaskReason.RELATIONSHIP_SNAPSHOT_UNAVAILABLE)


def _family_columns(families: Sequence[str], overrides: Mapping[str, Sequence[str]] | None) -> dict[str, tuple[str, ...]]:
    defaults = default_feature_family_map()
    override_map = {
        str(family): tuple(str(column).strip() for column in columns if str(column).strip())
        for family, columns in dict(overrides or {}).items()
    }
    out: dict[str, tuple[str, ...]] = {}
    for family in families:
        key = str(family)
        if key not in defaults:
            raise ValueError(f"Unsupported Cross-Asset relationship context family {key!r}")
        out[key] = override_map.get(key) or defaults[key].required_columns
    return out


def _text(value: object, field_name: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"Cross-Asset relationship context {field_name} must be non-empty")
    return text


def _optional_text(value: object | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    return text


def _cadence_days(value: Mapping[str, Any]) -> dict[str, int]:
    out: dict[str, int] = {}
    for band, days in dict(value or {}).items():
        band_key = str(band).strip().lower()
        if not band_key:
            continue
        out[band_key] = _positive_int(days, f"snapshot_cadence_days[{band_key}]")
    if not out:
        raise ValueError("Cross-Asset relationship context snapshot_cadence_days must not be empty")
    return out


def _stale_snapshot_policy(value: Mapping[str, Any]) -> dict[str, Any]:
    raw = dict(value or {})
    policy_id = _text(raw.get("policy_id") or DEFAULT_STALE_SNAPSHOT_POLICY_ID, "stale_snapshot_policy.policy_id")
    action = _text(raw.get("action") or "mask_unavailable", "stale_snapshot_policy.action")
    if action != "mask_unavailable":
        raise ValueError("Cross-Asset relationship context stale_snapshot_policy must mask unavailable")
    reason = _text(raw.get("mask_reason") or CrossAssetStateMaskReason.STALE_RELATIONSHIP_SNAPSHOT, "stale_snapshot_policy.mask_reason")
    if reason != CrossAssetStateMaskReason.STALE_RELATIONSHIP_SNAPSHOT:
        raise ValueError("Cross-Asset relationship context stale_snapshot_policy mask_reason must be stale_relationship_snapshot")
    stale_after = _cadence_days(raw.get("stale_after_days_by_band") or DEFAULT_STALE_AFTER_DAYS)
    return {
        "policy_id": policy_id,
        "stale_after_days_by_band": stale_after,
        "action": action,
        "mask_reason": reason,
    }


def _positive_int(value: object, field_name: str) -> int:
    try:
        out = int(value)
    except Exception as exc:
        raise ValueError(f"Cross-Asset relationship context {field_name} must be an integer") from exc
    if out <= 0:
        raise ValueError(f"Cross-Asset relationship context {field_name} must be positive")
    return out


def _optional_positive_int(value: object | None, field_name: str) -> int | None:
    if value is None:
        return None
    return _positive_int(value, field_name)


def _float_or_none(value: object | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _pandas() -> Any:
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("Cross-Asset relationship context requires pandas") from exc
    return pd


__all__ = [
    "CROSS_ASSET_RELATIONSHIP_CONTEXT_ARTIFACT_KIND",
    "CROSS_ASSET_RELATIONSHIP_CONTEXT_SCHEMA_VERSION",
    "CrossAssetRelationshipContextHandoff",
    "CrossAssetRelationshipContextRecord",
    "CrossAssetRelationshipContextResolver",
    "validate_relationship_context_handoff_for_production",
]
