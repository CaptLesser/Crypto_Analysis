from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.regimes.core.serialization import to_jsonable


RELATIONSHIP_REGISTRY_SCHEMA_VERSION = 1
RELATIONSHIP_REGISTRY_ARTIFACT_KIND = "cross_asset_relationship_snapshot_registry"

DEFAULT_INTERVAL_BAND_MAP: Mapping[int, str] = {
    240: "meso",
    1440: "macro",
}

DEFAULT_RELATIONSHIP_FEATURE_FAMILIES: tuple[str, ...] = (
    "anchor_core_exposure",
    "peer_strength",
    "peer_stability",
    "relationship_concentration",
    "relationship_entropy",
    "residual_peer_signal",
    "decoupling_or_stress",
)


class RelationshipMaskReason:
    NO_RELATIONSHIP_SNAPSHOT = "no_relationship_snapshot"
    ASSET_NOT_IN_SNAPSHOT = "asset_not_in_snapshot"
    INSUFFICIENT_RELATIONSHIP_HISTORY = "insufficient_relationship_history"
    NO_VIABLE_ANCHOR = "no_viable_anchor"
    NO_VIABLE_PEER_EDGES = "no_viable_peer_edges"
    NO_CORE_BASKET_RELATIONSHIP = "no_core_basket_relationship"
    PEER_METADATA_DIAGNOSTIC_ONLY = "peer_metadata_diagnostic_only"
    SCHEMA_MISSING_REQUIRED_FIELDS = "schema_missing_required_fields"
    STALE_SNAPSHOT = "stale_snapshot"
    AMBIGUOUS_SNAPSHOT = "ambiguous_snapshot"
    SOURCE_TAIL_MISSING = "source_tail_missing"


MASK_REASONS: frozenset[str] = frozenset(
    value
    for name, value in vars(RelationshipMaskReason).items()
    if name.isupper() and isinstance(value, str)
)


@dataclass(frozen=True)
class PeerMetadataRef:
    artifact_kind: str
    path: str
    diagnostic_only: bool = True
    model_facing_v1: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "artifact_kind", _text(self.artifact_kind, "artifact_kind"))
        object.__setattr__(self, "path", _text(self.path, "path"))
        object.__setattr__(self, "diagnostic_only", bool(self.diagnostic_only))
        object.__setattr__(self, "model_facing_v1", bool(self.model_facing_v1))
        if self.model_facing_v1:
            raise ValueError("Cross-Asset-State peer metadata refs must remain non-model-facing in v1")

    def as_dict(self) -> dict[str, Any]:
        return {
            "artifact_kind": self.artifact_kind,
            "path": self.path,
            "diagnostic_only": bool(self.diagnostic_only),
            "model_facing_v1": False,
        }


@dataclass(frozen=True)
class RelationshipSnapshotRef:
    snapshot_id: str
    band: str
    interval: int
    refit_key: str | None = None
    known_at_ts: int | float | str | None = None
    source_tail_ts: int | float | str | None = None
    effective_start_ts: int | float | str | None = None
    effective_end_ts: int | float | str | None = None
    asset_ids: Sequence[str] = ()
    relationship_artifact_refs: Mapping[str, str] = field(default_factory=dict)
    peer_metadata_refs: Sequence[PeerMetadataRef | Mapping[str, Any]] = ()
    source_manifest_id: str | None = None
    relationship_discovery_run_id: str | None = None
    artifact_root: str | None = None
    schema_version: int = RELATIONSHIP_REGISTRY_SCHEMA_VERSION
    production_enabled: bool = False
    production_outputs_written: bool = False
    peer_group_model_facing_v1: bool = False

    def __post_init__(self) -> None:
        if self.production_enabled or self.production_outputs_written:
            raise ValueError("Cross-Asset relationship snapshot refs are sandbox/non-production only")
        if self.peer_group_model_facing_v1:
            raise ValueError("Cross-Asset relationship snapshot refs cannot expose peer_group as model-facing v1")
        object.__setattr__(self, "schema_version", _schema_version(self.schema_version))
        object.__setattr__(self, "snapshot_id", _text(self.snapshot_id, "snapshot_id"))
        object.__setattr__(self, "band", _text(self.band, "band").lower())
        object.__setattr__(self, "interval", _positive_int(self.interval, "interval"))
        object.__setattr__(self, "refit_key", _optional_text(self.refit_key))
        object.__setattr__(self, "known_at_ts", _optional_timestamp(self.known_at_ts, "known_at_ts"))
        object.__setattr__(self, "source_tail_ts", _optional_timestamp(self.source_tail_ts, "source_tail_ts"))
        object.__setattr__(self, "effective_start_ts", _optional_timestamp(self.effective_start_ts, "effective_start_ts"))
        object.__setattr__(self, "effective_end_ts", _optional_timestamp(self.effective_end_ts, "effective_end_ts"))
        if self.effective_start_ts is not None and self.effective_end_ts is not None:
            if _orderable(self.effective_start_ts, "effective_start_ts") > _orderable(self.effective_end_ts, "effective_end_ts"):
                raise ValueError("Cross-Asset relationship snapshot effective_start_ts must be <= effective_end_ts")
        if self.known_at_ts is not None and self.source_tail_ts is not None:
            if _orderable(self.source_tail_ts, "source_tail_ts") > _orderable(self.known_at_ts, "known_at_ts"):
                raise ValueError("Cross-Asset relationship snapshot source_tail_ts must not exceed known_at_ts")
        assets = tuple(dict.fromkeys(_text(asset, "asset_id") for asset in self.asset_ids if str(asset).strip()))
        object.__setattr__(self, "asset_ids", assets)
        refs = {str(key): _text(value, f"relationship_artifact_refs[{key}]") for key, value in dict(self.relationship_artifact_refs).items()}
        object.__setattr__(self, "relationship_artifact_refs", refs)
        peer_refs = tuple(ref if isinstance(ref, PeerMetadataRef) else PeerMetadataRef(**dict(ref)) for ref in self.peer_metadata_refs)
        object.__setattr__(self, "peer_metadata_refs", peer_refs)
        object.__setattr__(self, "source_manifest_id", _optional_text(self.source_manifest_id))
        object.__setattr__(self, "relationship_discovery_run_id", _optional_text(self.relationship_discovery_run_id))
        object.__setattr__(self, "artifact_root", _optional_text(self.artifact_root))
        object.__setattr__(self, "production_enabled", False)
        object.__setattr__(self, "production_outputs_written", False)
        object.__setattr__(self, "peer_group_model_facing_v1", False)

    @property
    def has_causal_lineage(self) -> bool:
        return self.known_at_ts is not None and self.source_tail_ts is not None

    def known_at_order(self) -> float:
        if self.known_at_ts is None:
            return float("-inf")
        return _orderable(self.known_at_ts, "known_at_ts")

    def as_dict(self) -> dict[str, Any]:
        return {
            "artifact_kind": "cross_asset_relationship_snapshot_ref",
            "schema_version": int(self.schema_version),
            "snapshot_id": self.snapshot_id,
            "band": self.band,
            "interval": int(self.interval),
            "refit_key": self.refit_key,
            "known_at_ts": self.known_at_ts,
            "source_tail_ts": self.source_tail_ts,
            "effective_start_ts": self.effective_start_ts,
            "effective_end_ts": self.effective_end_ts,
            "snapshot_valid_from_ts": self.effective_start_ts,
            "snapshot_valid_until_ts": self.effective_end_ts,
            "asset_ids": list(self.asset_ids),
            "asset_count": len(self.asset_ids),
            "relationship_artifact_refs": dict(self.relationship_artifact_refs),
            "peer_metadata_refs": [ref.as_dict() for ref in self.peer_metadata_refs],
            "source_manifest_id": self.source_manifest_id,
            "relationship_discovery_run_id": self.relationship_discovery_run_id,
            "artifact_root": self.artifact_root,
            "has_causal_lineage": self.has_causal_lineage,
            "production_enabled": False,
            "production_outputs_written": False,
            "peer_group_model_facing_v1": False,
        }


@dataclass(frozen=True)
class RelationshipAvailabilityRecord:
    asset_id: str
    band: str
    ts: int | float | str
    relationship_feature_family: str | None = None
    available: bool = False
    mask_reason: str | None = None
    snapshot: RelationshipSnapshotRef | Mapping[str, Any] | None = None
    relationship_artifact_refs: Mapping[str, str] = field(default_factory=dict)
    peer_metadata_refs: Sequence[PeerMetadataRef | Mapping[str, Any]] = ()
    schema_version: int = RELATIONSHIP_REGISTRY_SCHEMA_VERSION
    production_enabled: bool = False
    production_outputs_written: bool = False
    canonical_production_state_outputs_written: bool = False

    def __post_init__(self) -> None:
        if self.production_enabled or self.production_outputs_written or self.canonical_production_state_outputs_written:
            raise ValueError("Cross-Asset relationship availability records are sandbox/non-production only")
        object.__setattr__(self, "schema_version", _schema_version(self.schema_version))
        object.__setattr__(self, "asset_id", _text(self.asset_id, "asset_id"))
        object.__setattr__(self, "band", _text(self.band, "band").lower())
        object.__setattr__(self, "ts", _timestamp(self.ts, "ts"))
        object.__setattr__(self, "relationship_feature_family", _optional_text(self.relationship_feature_family))
        snap = self.snapshot
        if isinstance(snap, Mapping):
            snap = RelationshipSnapshotRef(**dict(snap))
        object.__setattr__(self, "snapshot", snap)
        available = bool(self.available)
        reason = _optional_text(self.mask_reason)
        if available:
            if snap is None:
                raise ValueError("Available Cross-Asset relationship records require a snapshot")
            reason = None
        else:
            if reason is None:
                raise ValueError("Unavailable Cross-Asset relationship records require a mask_reason")
            if reason not in MASK_REASONS:
                raise ValueError(f"Unsupported Cross-Asset relationship mask_reason {reason!r}")
        object.__setattr__(self, "available", available)
        object.__setattr__(self, "mask_reason", reason)
        refs = {str(key): _text(value, f"relationship_artifact_refs[{key}]") for key, value in dict(self.relationship_artifact_refs).items()}
        object.__setattr__(self, "relationship_artifact_refs", refs)
        peer_refs = tuple(ref if isinstance(ref, PeerMetadataRef) else PeerMetadataRef(**dict(ref)) for ref in self.peer_metadata_refs)
        object.__setattr__(self, "peer_metadata_refs", peer_refs)
        object.__setattr__(self, "production_enabled", False)
        object.__setattr__(self, "production_outputs_written", False)
        object.__setattr__(self, "canonical_production_state_outputs_written", False)

    def as_dict(self) -> dict[str, Any]:
        return {
            "artifact_kind": "cross_asset_relationship_availability_record",
            "schema_version": int(self.schema_version),
            "asset_id": self.asset_id,
            "band": self.band,
            "ts": self.ts,
            "relationship_feature_family": self.relationship_feature_family,
            "availability_status": "available" if self.available else "masked_unavailable",
            "available": bool(self.available),
            "mask_reason": self.mask_reason,
            "snapshot_id": self.snapshot.snapshot_id if isinstance(self.snapshot, RelationshipSnapshotRef) else None,
            "known_at_ts": self.snapshot.known_at_ts if isinstance(self.snapshot, RelationshipSnapshotRef) else None,
            "source_tail_ts": self.snapshot.source_tail_ts if isinstance(self.snapshot, RelationshipSnapshotRef) else None,
            "snapshot_valid_from_ts": self.snapshot.effective_start_ts if isinstance(self.snapshot, RelationshipSnapshotRef) else None,
            "snapshot_valid_until_ts": self.snapshot.effective_end_ts if isinstance(self.snapshot, RelationshipSnapshotRef) else None,
            "source_manifest_id": self.snapshot.source_manifest_id if isinstance(self.snapshot, RelationshipSnapshotRef) else None,
            "relationship_discovery_run_id": self.snapshot.relationship_discovery_run_id if isinstance(self.snapshot, RelationshipSnapshotRef) else None,
            "relationship_artifact_refs": dict(self.relationship_artifact_refs),
            "peer_metadata_refs": [ref.as_dict() for ref in self.peer_metadata_refs],
            "shape_preserving": True,
            "production_enabled": False,
            "production_outputs_written": False,
            "canonical_production_state_outputs_written": False,
        }


@dataclass(frozen=True)
class RelationshipSnapshotIndex:
    snapshots: Sequence[RelationshipSnapshotRef | Mapping[str, Any]]
    source_roots: Sequence[str] = ()
    schema_version: int = RELATIONSHIP_REGISTRY_SCHEMA_VERSION
    production_enabled: bool = False
    snapshots_by_band: Mapping[str, Sequence[RelationshipSnapshotRef]] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.production_enabled:
            raise ValueError("Cross-Asset relationship snapshot index production_enabled must remain false")
        object.__setattr__(self, "schema_version", _schema_version(self.schema_version))
        snapshots = tuple(snapshot if isinstance(snapshot, RelationshipSnapshotRef) else RelationshipSnapshotRef(**dict(snapshot)) for snapshot in self.snapshots)
        ids = [snapshot.snapshot_id for snapshot in snapshots]
        if len(ids) != len(set(ids)):
            raise ValueError("Cross-Asset relationship snapshot ids must be unique")
        object.__setattr__(self, "snapshots", snapshots)
        object.__setattr__(self, "source_roots", tuple(str(root) for root in self.source_roots))
        by_band: dict[str, list[RelationshipSnapshotRef]] = {}
        for snapshot in snapshots:
            by_band.setdefault(snapshot.band, []).append(snapshot)
        object.__setattr__(self, "snapshots_by_band", {band: tuple(rows) for band, rows in by_band.items()})
        object.__setattr__(self, "production_enabled", False)

    def list_available_snapshots(self, *, band: str | None = None) -> tuple[RelationshipSnapshotRef, ...]:
        if band is None:
            return tuple(self.snapshots)
        wanted = str(band).strip().lower()
        return tuple(self.snapshots_by_band.get(wanted, ()))

    def summary(self) -> dict[str, Any]:
        known = [snapshot.known_at_order() for snapshot in self.snapshots if snapshot.known_at_ts is not None]
        assets = sorted({asset for snapshot in self.snapshots for asset in snapshot.asset_ids})
        bands = sorted({snapshot.band for snapshot in self.snapshots})
        missing_lineage = [snapshot.snapshot_id for snapshot in self.snapshots if not snapshot.has_causal_lineage]
        cadence = "unknown_no_causal_snapshots"
        if len(known) == 1:
            cadence = "single_or_ad_hoc_snapshot"
        elif len(known) > 1:
            cadence = "refit_or_calendar_snapshot_family"
        return {
            "artifact_kind": "cross_asset_relationship_snapshot_index_summary",
            "schema_version": int(self.schema_version),
            "snapshot_count": len(self.snapshots),
            "indexed_snapshot_count": len(self.snapshots),
            "band_index_group_count": len(self.snapshots_by_band),
            "causal_snapshot_count": len(known),
            "missing_causal_lineage_count": len(missing_lineage),
            "missing_causal_lineage_snapshot_ids": missing_lineage[:50],
            "known_at_min": min(known) if known else None,
            "known_at_max": max(known) if known else None,
            "bands": bands,
            "asset_count": len(assets),
            "asset_sample": assets[:50],
            "cadence_inference": cadence,
            "backfill_1_to_2y_readiness": "not_ready" if len(known) < 2 else "needs_gap_analysis",
            "production_enabled": False,
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            "artifact_kind": RELATIONSHIP_REGISTRY_ARTIFACT_KIND,
            "schema_version": int(self.schema_version),
            "source_roots": list(self.source_roots),
            "snapshots": [snapshot.as_dict() for snapshot in self.snapshots],
            "summary": self.summary(),
            "production_enabled": False,
        }


@dataclass(frozen=True)
class RelationshipSnapshotResolver:
    index: RelationshipSnapshotIndex

    def resolve(
        self,
        *,
        asset_id: str,
        band: str,
        ts: int | float | str,
        relationship_feature_family: str | None = None,
    ) -> RelationshipAvailabilityRecord:
        asset = _text(asset_id, "asset_id")
        resolved_band = _text(band, "band").lower()
        resolved_ts = _timestamp(ts, "ts")
        candidates = self.index.list_available_snapshots(band=resolved_band)
        if not candidates:
            return self._masked(asset, resolved_band, resolved_ts, relationship_feature_family, RelationshipMaskReason.NO_RELATIONSHIP_SNAPSHOT)
        causal = [snapshot for snapshot in candidates if snapshot.has_causal_lineage]
        if not causal:
            return self._masked(asset, resolved_band, resolved_ts, relationship_feature_family, RelationshipMaskReason.SOURCE_TAIL_MISSING)
        orderable_ts = _orderable(resolved_ts, "ts")
        known = [snapshot for snapshot in causal if snapshot.known_at_order() <= orderable_ts]
        if not known:
            return self._masked(asset, resolved_band, resolved_ts, relationship_feature_family, RelationshipMaskReason.NO_RELATIONSHIP_SNAPSHOT)
        latest_known_at = max(snapshot.known_at_order() for snapshot in known)
        latest = [snapshot for snapshot in known if snapshot.known_at_order() == latest_known_at]
        if len(latest) > 1:
            merged = _merge_compatible_snapshot_refs(latest)
            if merged is None:
                return self._masked(asset, resolved_band, resolved_ts, relationship_feature_family, RelationshipMaskReason.AMBIGUOUS_SNAPSHOT)
            snapshot = merged
        else:
            snapshot = latest[0]
        if asset not in snapshot.asset_ids:
            return self._masked(asset, resolved_band, resolved_ts, relationship_feature_family, RelationshipMaskReason.ASSET_NOT_IN_SNAPSHOT, snapshot=snapshot)
        if not snapshot.relationship_artifact_refs:
            return self._masked(asset, resolved_band, resolved_ts, relationship_feature_family, RelationshipMaskReason.SCHEMA_MISSING_REQUIRED_FIELDS, snapshot=snapshot)
        return RelationshipAvailabilityRecord(
            asset_id=asset,
            band=resolved_band,
            ts=resolved_ts,
            relationship_feature_family=relationship_feature_family,
            available=True,
            snapshot=snapshot,
            relationship_artifact_refs=snapshot.relationship_artifact_refs,
            peer_metadata_refs=snapshot.peer_metadata_refs,
        )

    @staticmethod
    def _masked(
        asset_id: str,
        band: str,
        ts: int | float | str,
        relationship_feature_family: str | None,
        reason: str,
        *,
        snapshot: RelationshipSnapshotRef | None = None,
    ) -> RelationshipAvailabilityRecord:
        return RelationshipAvailabilityRecord(
            asset_id=asset_id,
            band=band,
            ts=ts,
            relationship_feature_family=relationship_feature_family,
            available=False,
            mask_reason=reason,
            snapshot=snapshot,
            relationship_artifact_refs=snapshot.relationship_artifact_refs if snapshot is not None else {},
            peer_metadata_refs=snapshot.peer_metadata_refs if snapshot is not None else (),
        )


def build_relationship_snapshot_index(
    roots: Sequence[str | Path],
    *,
    interval_band_map: Mapping[int, str] | None = None,
    max_parquet_files_per_root: int | None = None,
) -> RelationshipSnapshotIndex:
    band_map = dict(DEFAULT_INTERVAL_BAND_MAP)
    if interval_band_map:
        band_map.update({int(key): str(value).lower() for key, value in interval_band_map.items()})
    snapshots: list[RelationshipSnapshotRef] = []
    source_roots: list[str] = []
    for raw_root in roots:
        root = Path(raw_root)
        source_roots.append(str(root))
        if not root.exists():
            continue
        snapshots.extend(_snapshots_from_refit_manifests(root, band_map=band_map))
        snapshots.extend(_snapshots_from_diagnostic_sidecars(root, band_map=band_map))
        snapshots.extend(_snapshots_from_cross_asset_feature_rows(root, band_map=band_map, max_files=max_parquet_files_per_root))
    return RelationshipSnapshotIndex(snapshots=_dedupe_snapshots(snapshots), source_roots=tuple(source_roots))


def _snapshots_from_refit_manifests(root: Path, *, band_map: Mapping[int, str]) -> list[RelationshipSnapshotRef]:
    out: list[RelationshipSnapshotRef] = []
    for path in sorted(root.rglob("snapshot.json")):
        if "refit_snapshot_manifest" not in path.parts:
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, Mapping):
                continue
            interval = _positive_int(payload.get("interval"), "interval")
            refit_key = _optional_text(payload.get("refit_key"))
            known_at = payload.get("known_at_ts")
            source_tail = payload.get("source_tail_ts")
            assets = _assets_from_snapshot_payload(payload)
            snapshot_id = _snapshot_id("refit_snapshot", refit_key, interval, known_at, path)
            out.append(
                RelationshipSnapshotRef(
                    snapshot_id=snapshot_id,
                    band=_band_for_interval(interval, band_map),
                    interval=interval,
                    refit_key=refit_key,
                    known_at_ts=known_at,
                    source_tail_ts=source_tail,
                    effective_start_ts=payload.get("effective_start_ts", payload.get("snapshot_valid_from_ts", payload.get("snapshot_start"))),
                    effective_end_ts=payload.get("effective_end_ts", payload.get("snapshot_valid_until_ts", payload.get("snapshot_end"))),
                    asset_ids=assets,
                    relationship_artifact_refs={"refit_snapshot_manifest": str(path)},
                    source_manifest_id=_optional_text(payload.get("universe_manifest_ref")),
                    relationship_discovery_run_id=(
                        _optional_text(payload.get("relationship_policy_id"))
                        or _optional_text(payload.get("policy_id"))
                        or refit_key
                        or _run_id_from_path(path)
                    ),
                    artifact_root=str(root),
                )
            )
        except Exception:
            continue
    return out


def _snapshots_from_diagnostic_sidecars(root: Path, *, band_map: Mapping[int, str]) -> list[RelationshipSnapshotRef]:
    manifests = []
    if (root / "diagnostic_sidecars_manifest.json").is_file():
        manifests.append(root / "diagnostic_sidecars_manifest.json")
    manifests.extend(path for path in sorted(root.rglob("diagnostic_sidecars_manifest.json")) if path not in manifests)
    out: list[RelationshipSnapshotRef] = []
    for manifest_path in manifests:
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(payload, Mapping):
            continue
        paths = dict(payload.get("paths") or {})
        run_id = _optional_text(payload.get("run_id")) or _run_id_from_path(manifest_path)
        profile_path = _resolve_manifest_path(paths.get("asset_relationship_profiles"), manifest_path.parent)
        if profile_path is None or not profile_path.is_file():
            continue
        try:
            frame = _read_parquet(profile_path)
        except Exception:
            continue
        if "interval" not in frame.columns or "asset" not in frame.columns:
            continue
        group_cols = ["interval"]
        if "refit_key" in frame.columns:
            group_cols.append("refit_key")
        grouped = frame.groupby(group_cols, dropna=False)
        for key, group in grouped:
            key_values = key if isinstance(key, tuple) else (key,)
            interval = _positive_int(key_values[0], "interval")
            refit_key = str(key_values[1]) if len(key_values) > 1 and str(key_values[1]) != "nan" else run_id
            ts_values = _numeric_values(group["ts"]) if "ts" in group.columns else []
            refs = {name: str(_resolve_manifest_path(value, manifest_path.parent) or value) for name, value in paths.items() if name not in _peer_metadata_names()}
            peer_refs = [
                PeerMetadataRef(artifact_kind=name, path=str(_resolve_manifest_path(value, manifest_path.parent) or value))
                for name, value in paths.items()
                if name in _peer_metadata_names()
            ]
            out.append(
                RelationshipSnapshotRef(
                    snapshot_id=_snapshot_id("diagnostic_sidecar", run_id, refit_key, interval),
                    band=_band_for_interval(interval, band_map),
                    interval=interval,
                    refit_key=refit_key,
                    known_at_ts=_first_present(group, "known_at_ts"),
                    source_tail_ts=_first_present(group, "source_tail_ts"),
                    effective_start_ts=min(ts_values) if ts_values else None,
                    effective_end_ts=max(ts_values) if ts_values else None,
                    asset_ids=tuple(str(asset) for asset in sorted(group["asset"].dropna().astype(str).unique())),
                    relationship_artifact_refs=refs,
                    peer_metadata_refs=tuple(peer_refs),
                    source_manifest_id=run_id,
                    relationship_discovery_run_id=run_id,
                    artifact_root=str(manifest_path.parent),
                )
            )
    return out


def _snapshots_from_cross_asset_feature_rows(
    root: Path,
    *,
    band_map: Mapping[int, str],
    max_files: int | None,
) -> list[RelationshipSnapshotRef]:
    files = sorted(root.rglob("*.parquet"))
    if max_files is not None:
        files = files[: max(0, int(max_files))]
    grouped_refs: dict[tuple[str, int, str, str, str], dict[str, Any]] = {}
    for path in files:
        try:
            frame = _read_parquet(path)
        except Exception:
            continue
        required = {"asset", "interval", "known_at_ts", "source_tail_ts"}
        if not required.issubset(set(frame.columns)):
            continue
        band = _band_from_path(path) or _band_for_interval(_positive_int(frame["interval"].iloc[0], "interval"), band_map)
        group_cols = ["interval", "known_at_ts", "source_tail_ts"]
        for optional in ("refit_key", "feature_bundle_id", "relationship_policy_id"):
            if optional in frame.columns:
                group_cols.append(optional)
        for key, group in frame.groupby(group_cols, dropna=False):
            values = key if isinstance(key, tuple) else (key,)
            lookup = dict(zip(group_cols, values))
            interval = _positive_int(lookup["interval"], "interval")
            known_at = lookup["known_at_ts"]
            source_tail = lookup["source_tail_ts"]
            refit_key = _optional_text(lookup.get("refit_key")) or _optional_text(lookup.get("feature_bundle_id")) or "cross_asset_features"
            ts_values = _numeric_values(group["ts"]) if "ts" in group.columns else []
            group_key = (band, interval, str(known_at), str(source_tail), refit_key)
            item = grouped_refs.setdefault(
                group_key,
                {
                    "band": band,
                    "interval": interval,
                    "known_at_ts": known_at,
                    "source_tail_ts": source_tail,
                    "refit_key": refit_key,
                    "effective_start_ts": None,
                    "effective_end_ts": None,
                    "asset_ids": set(),
                    "paths": [],
                    "source_manifest_id": _optional_text(lookup.get("feature_bundle_id")) or _run_id_from_path(path),
                    "relationship_discovery_run_id": _optional_text(lookup.get("relationship_policy_id")),
                },
            )
            if ts_values:
                item["effective_start_ts"] = min(ts_values) if item["effective_start_ts"] is None else min(item["effective_start_ts"], min(ts_values))
                item["effective_end_ts"] = max(ts_values) if item["effective_end_ts"] is None else max(item["effective_end_ts"], max(ts_values))
            item["asset_ids"].update(str(asset) for asset in group["asset"].dropna().astype(str).unique())
            item["paths"].append(str(path))
    out: list[RelationshipSnapshotRef] = []
    for item in grouped_refs.values():
        path_hash = _snapshot_id(*item["paths"])
        out.append(
            RelationshipSnapshotRef(
                snapshot_id=_snapshot_id("cross_asset_feature_rows", item["refit_key"], item["interval"], item["known_at_ts"], item["source_tail_ts"]),
                band=item["band"],
                interval=item["interval"],
                refit_key=item["refit_key"],
                known_at_ts=item["known_at_ts"],
                source_tail_ts=item["source_tail_ts"],
                effective_start_ts=item["effective_start_ts"],
                effective_end_ts=item["effective_end_ts"],
                asset_ids=tuple(sorted(item["asset_ids"])),
                relationship_artifact_refs={"cross_asset_feature_rows": f"{len(item['paths'])} parquet partitions", "partition_set_hash": path_hash},
                source_manifest_id=item["source_manifest_id"],
                relationship_discovery_run_id=item["relationship_discovery_run_id"],
                artifact_root=str(root),
            )
        )
    return out


def _dedupe_snapshots(snapshots: Sequence[RelationshipSnapshotRef]) -> tuple[RelationshipSnapshotRef, ...]:
    out: dict[str, RelationshipSnapshotRef] = {}
    for snapshot in snapshots:
        existing = out.get(snapshot.snapshot_id)
        out[snapshot.snapshot_id] = snapshot if existing is None else _merge_snapshot_ref(existing, snapshot)
    return tuple(out.values())


def _merge_snapshot_ref(left: RelationshipSnapshotRef, right: RelationshipSnapshotRef) -> RelationshipSnapshotRef:
    snapshot_id = left.snapshot_id if left.snapshot_id == right.snapshot_id else _snapshot_id("merged_compatible_snapshot", left.snapshot_id, right.snapshot_id)
    return RelationshipSnapshotRef(
        snapshot_id=snapshot_id,
        band=left.band,
        interval=left.interval,
        refit_key=left.refit_key or right.refit_key,
        known_at_ts=left.known_at_ts if left.known_at_ts is not None else right.known_at_ts,
        source_tail_ts=left.source_tail_ts if left.source_tail_ts is not None else right.source_tail_ts,
        effective_start_ts=_min_optional_timestamp(left.effective_start_ts, right.effective_start_ts),
        effective_end_ts=_max_optional_timestamp(left.effective_end_ts, right.effective_end_ts),
        asset_ids=tuple(sorted(set(left.asset_ids) | set(right.asset_ids))),
        relationship_artifact_refs=_merge_relationship_artifact_refs(left.relationship_artifact_refs, right.relationship_artifact_refs),
        peer_metadata_refs=_merge_peer_metadata_refs(left.peer_metadata_refs, right.peer_metadata_refs),
        source_manifest_id=left.source_manifest_id or right.source_manifest_id,
        relationship_discovery_run_id=left.relationship_discovery_run_id or right.relationship_discovery_run_id,
        artifact_root=left.artifact_root or right.artifact_root,
    )


def _merge_compatible_snapshot_refs(snapshots: Sequence[RelationshipSnapshotRef]) -> RelationshipSnapshotRef | None:
    if not snapshots:
        return None
    merged = snapshots[0]
    for snapshot in snapshots[1:]:
        if not _snapshot_refs_are_compatible_for_merge(merged, snapshot):
            return None
        merged = _merge_snapshot_ref(merged, snapshot)
    return merged


def _snapshot_refs_are_compatible_for_merge(left: RelationshipSnapshotRef, right: RelationshipSnapshotRef) -> bool:
    same_time = (
        left.band == right.band
        and left.interval == right.interval
        and left.known_at_ts == right.known_at_ts
        and left.source_tail_ts == right.source_tail_ts
    )
    if not same_time:
        return False
    if left.relationship_discovery_run_id == right.relationship_discovery_run_id:
        return True
    return _is_refit_manifest_feature_row_pair(left, right)


def _is_refit_manifest_feature_row_pair(left: RelationshipSnapshotRef, right: RelationshipSnapshotRef) -> bool:
    left_keys = set(left.relationship_artifact_refs)
    right_keys = set(right.relationship_artifact_refs)
    return (
        "refit_snapshot_manifest" in left_keys
        and "cross_asset_feature_rows" in right_keys
    ) or (
        "cross_asset_feature_rows" in left_keys
        and "refit_snapshot_manifest" in right_keys
    )


def _merge_relationship_artifact_refs(left: Mapping[str, str], right: Mapping[str, str]) -> dict[str, str]:
    out = dict(left)
    for key, value in right.items():
        if key not in out or out[key] == value:
            out[key] = value
            continue
        suffix = 2
        merged_key = f"{key}_duplicate_{suffix}"
        while merged_key in out and out[merged_key] != value:
            suffix += 1
            merged_key = f"{key}_duplicate_{suffix}"
        out[merged_key] = value
    return out


def _merge_peer_metadata_refs(
    left: Sequence[PeerMetadataRef],
    right: Sequence[PeerMetadataRef],
) -> tuple[PeerMetadataRef, ...]:
    out: list[PeerMetadataRef] = []
    seen: set[str] = set()
    for ref in (*left, *right):
        key = json.dumps(ref.as_dict(), sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        out.append(ref)
    return tuple(out)


def _min_optional_timestamp(left: Any, right: Any) -> Any:
    values = [value for value in (left, right) if value is not None]
    if not values:
        return None
    return min(values, key=lambda value: _orderable(value, "effective_start_ts"))


def _max_optional_timestamp(left: Any, right: Any) -> Any:
    values = [value for value in (left, right) if value is not None]
    if not values:
        return None
    return max(values, key=lambda value: _orderable(value, "effective_end_ts"))


def _read_parquet(path: Path) -> Any:
    import pandas as pd

    return pd.read_parquet(path)


def _assets_from_snapshot_payload(payload: Mapping[str, Any]) -> tuple[str, ...]:
    assets: list[str] = []
    for field_name in ("eligible_assets", "broad_sample_assets", "core_assets", "anchors"):
        value = payload.get(field_name)
        if isinstance(value, str):
            assets.extend(item.strip() for item in value.split(",") if item.strip())
        elif isinstance(value, Sequence) and not isinstance(value, (bytes, str)):
            assets.extend(str(item).strip() for item in value if str(item).strip())
    return tuple(dict.fromkeys(assets))


def _first_present(frame: Any, column: str) -> Any:
    if column not in frame.columns:
        return None
    values = frame[column].dropna()
    if values.empty:
        return None
    return values.iloc[0]


def _numeric_values(series: Any) -> list[float]:
    try:
        import pandas as pd

        values = pd.to_numeric(series, errors="coerce").dropna()
        return [float(value) for value in values.tolist()]
    except Exception:
        return []


def _resolve_manifest_path(value: Any, base: Path) -> Path | None:
    if value is None:
        return None
    path = Path(str(value))
    if path.is_absolute():
        return path
    return base / path


def _peer_metadata_names() -> frozenset[str]:
    return frozenset({"peer_group_snapshots", "peer_membership_history", "peer_group_stability_scores"})


def _band_for_interval(interval: int, band_map: Mapping[int, str]) -> str:
    return str(band_map.get(int(interval), f"interval_{int(interval)}")).lower()


def _band_from_path(path: Path) -> str | None:
    for part in path.parts:
        if str(part).startswith("band="):
            return str(part).split("=", 1)[1].lower()
    return None


def _snapshot_id(*parts: Any) -> str:
    raw = json.dumps([str(part) for part in parts if part is not None], sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _run_id_from_path(path: Path) -> str | None:
    for part in reversed(path.parts):
        text = str(part)
        if text.startswith("regime_features_") or text.endswith("_run") or "refit" in text:
            return text
    return None


def _schema_version(value: Any) -> int:
    try:
        version = int(value)
    except Exception as exc:
        raise ValueError("Cross-Asset relationship registry schema_version must be an integer") from exc
    if version != RELATIONSHIP_REGISTRY_SCHEMA_VERSION:
        raise ValueError(f"Unsupported Cross-Asset relationship registry schema_version {version!r}")
    return version


def _text(value: Any, field_name: str) -> str:
    text = str(value).strip()
    if not text or text == "None":
        raise ValueError(f"Cross-Asset relationship registry {field_name} must be non-empty")
    return text


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text == "None" or text.lower() == "nan":
        return None
    return text


def _positive_int(value: Any, field_name: str) -> int:
    try:
        out = int(value)
    except Exception as exc:
        raise ValueError(f"Cross-Asset relationship registry {field_name} must be an integer") from exc
    if out <= 0:
        raise ValueError(f"Cross-Asset relationship registry {field_name} must be positive")
    return out


def _timestamp(value: Any, field_name: str) -> Any:
    if value is None or isinstance(value, bool):
        raise ValueError(f"Cross-Asset relationship registry {field_name} must be timestamp-compatible")
    _orderable(value, field_name)
    return _plain_scalar(value)


def _optional_timestamp(value: Any, field_name: str) -> Any:
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    _orderable(value, field_name)
    return _plain_scalar(value)


def _orderable(value: Any, field_name: str) -> float:
    if value is None or isinstance(value, bool):
        raise ValueError(f"Cross-Asset relationship registry {field_name} must be numeric")
    try:
        return float(value)
    except Exception as exc:
        raise ValueError(f"Cross-Asset relationship registry {field_name} must be numeric") from exc


def _plain_scalar(value: Any) -> Any:
    if hasattr(value, "item") and callable(value.item):
        try:
            return value.item()
        except Exception:
            return value
    return value
