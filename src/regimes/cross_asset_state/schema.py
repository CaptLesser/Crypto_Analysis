from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.regimes.cross_asset_state.feature_families import (
    CROSS_ASSET_STATE_FEATURE_SET_ORIGINAL,
    CROSS_ASSET_STATE_SCHEMA_VERSION,
    SUPPORTED_BANDS,
    SUPPORTED_FEATURE_SET_VERSIONS,
    default_feature_family_map,
)


@dataclass(frozen=True)
class CrossAssetStateProfileGrain:
    asset_id: str
    relationship_feature_family: str
    band: str

    def __post_init__(self) -> None:
        asset = str(self.asset_id).strip()
        family = str(self.relationship_feature_family).strip()
        band = str(self.band).strip().lower()
        if not asset:
            raise ValueError("Cross-Asset-State grain requires asset_id")
        if family not in default_feature_family_map():
            raise ValueError(f"Unsupported Cross-Asset-State feature family {family!r}")
        if band not in SUPPORTED_BANDS:
            raise ValueError(f"Unsupported Cross-Asset-State band {band!r}")
        object.__setattr__(self, "asset_id", asset)
        object.__setattr__(self, "relationship_feature_family", family)
        object.__setattr__(self, "band", band)

    def key(self) -> tuple[str, str, str]:
        return (self.asset_id, self.relationship_feature_family, self.band)

    def as_dict(self) -> dict[str, str]:
        return {
            "asset_id": self.asset_id,
            "relationship_feature_family": self.relationship_feature_family,
            "band": self.band,
        }


@dataclass(frozen=True)
class CrossAssetStateSelectedProfileRecord:
    asset_id: str
    relationship_feature_family: str
    band: str
    profile_id: str
    method_family: str
    feature_columns: tuple[str, ...]
    state_count: int
    label_health_status: str
    selected_status: str
    relationship_snapshot_id: str
    known_at_ts: int | float | str
    source_tail_ts: int | float | str
    observed_rows: int
    label_counts: dict[str, int]
    feature_set_version: str = CROSS_ASSET_STATE_FEATURE_SET_ORIGINAL
    support_definition_id: str = "original_fixed_top3_v1"
    support_size: int | None = None
    support_quality: str | None = None
    support_rank_max: int | None = None
    support_threshold: float | None = None
    support_fallback_path: str | None = None
    repaired_feature_manifest_id: str | None = None
    relationship_context_cadence_policy_id: str | None = None
    snapshot_cadence_days: int | None = None
    stale_snapshot_policy: dict[str, Any] | None = None
    no_future_graph_backfill: bool = True
    snapshot_valid_from_ts: int | float | str | None = None
    snapshot_valid_until_ts: int | float | str | None = None
    semantic_score_placeholder: float | None = None
    mask_reason: str | None = None
    peer_metadata_sidecar_status: str = "diagnostic_only_not_model_facing"
    production_enabled: bool = False
    production_outputs_written: bool = False
    canonical_production_state_outputs_written: bool = False
    schema_version: int = CROSS_ASSET_STATE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.production_enabled or self.production_outputs_written or self.canonical_production_state_outputs_written:
            raise ValueError("Cross-Asset-State selected profiles are sandbox/non-production only")
        CrossAssetStateProfileGrain(self.asset_id, self.relationship_feature_family, self.band)
        if not str(self.relationship_snapshot_id).strip():
            raise ValueError("Selected Cross-Asset-State profiles require relationship_snapshot_id lineage")
        columns = tuple(str(column).strip() for column in self.feature_columns if str(column).strip())
        if not columns:
            raise ValueError("Selected Cross-Asset-State profiles require feature_columns")
        feature_set = str(self.feature_set_version).strip()
        if feature_set not in SUPPORTED_FEATURE_SET_VERSIONS:
            raise ValueError(f"Unsupported Cross-Asset-State feature_set_version {feature_set!r}")
        object.__setattr__(self, "asset_id", str(self.asset_id).strip())
        object.__setattr__(self, "relationship_feature_family", str(self.relationship_feature_family).strip())
        object.__setattr__(self, "band", str(self.band).strip().lower())
        object.__setattr__(self, "profile_id", str(self.profile_id).strip())
        object.__setattr__(self, "method_family", str(self.method_family).strip())
        object.__setattr__(self, "feature_columns", columns)
        object.__setattr__(self, "feature_set_version", feature_set)
        object.__setattr__(self, "support_definition_id", str(self.support_definition_id).strip() or "original_fixed_top3_v1")
        object.__setattr__(self, "support_size", _optional_nonnegative_int(self.support_size, "support_size"))
        object.__setattr__(self, "support_quality", str(self.support_quality).strip() if self.support_quality else None)
        object.__setattr__(self, "support_rank_max", _optional_nonnegative_int(self.support_rank_max, "support_rank_max"))
        object.__setattr__(self, "support_threshold", _optional_float(self.support_threshold, "support_threshold"))
        object.__setattr__(self, "support_fallback_path", str(self.support_fallback_path).strip() if self.support_fallback_path else None)
        object.__setattr__(
            self,
            "repaired_feature_manifest_id",
            str(self.repaired_feature_manifest_id).strip() if self.repaired_feature_manifest_id else None,
        )
        object.__setattr__(
            self,
            "relationship_context_cadence_policy_id",
            str(self.relationship_context_cadence_policy_id).strip() if self.relationship_context_cadence_policy_id else None,
        )
        object.__setattr__(self, "snapshot_cadence_days", _optional_nonnegative_int(self.snapshot_cadence_days, "snapshot_cadence_days"))
        object.__setattr__(self, "stale_snapshot_policy", dict(self.stale_snapshot_policy or {}))
        object.__setattr__(self, "no_future_graph_backfill", bool(self.no_future_graph_backfill))
        object.__setattr__(self, "state_count", int(self.state_count))
        object.__setattr__(self, "observed_rows", int(self.observed_rows))
        object.__setattr__(self, "label_counts", {str(key): int(value) for key, value in self.label_counts.items()})
        object.__setattr__(self, "production_enabled", False)
        object.__setattr__(self, "production_outputs_written", False)
        object.__setattr__(self, "canonical_production_state_outputs_written", False)

    def as_dict(self) -> dict[str, Any]:
        return {
            "artifact_kind": "cross_asset_state_selected_profile",
            "schema_version": int(self.schema_version),
            "asset_id": self.asset_id,
            "relationship_feature_family": self.relationship_feature_family,
            "band": self.band,
            "feature_set_version": self.feature_set_version,
            "support_definition_id": self.support_definition_id,
            "support_size": self.support_size,
            "support_quality": self.support_quality,
            "support_rank_max": self.support_rank_max,
            "support_threshold": self.support_threshold,
            "support_fallback_path": self.support_fallback_path,
            "repaired_feature_manifest_id": self.repaired_feature_manifest_id,
            "profile_id": self.profile_id,
            "method_family": self.method_family,
            "feature_columns": list(self.feature_columns),
            "state_count": int(self.state_count),
            "label_health_status": self.label_health_status,
            "selected_status": self.selected_status,
            "mask_reason": self.mask_reason,
            "relationship_snapshot_id": self.relationship_snapshot_id,
            "known_at_ts": self.known_at_ts,
            "source_tail_ts": self.source_tail_ts,
            "relationship_context_cadence_policy_id": self.relationship_context_cadence_policy_id,
            "snapshot_cadence_days": self.snapshot_cadence_days,
            "stale_snapshot_policy": dict(self.stale_snapshot_policy or {}),
            "no_future_graph_backfill": bool(self.no_future_graph_backfill),
            "snapshot_valid_from_ts": self.snapshot_valid_from_ts,
            "snapshot_valid_until_ts": self.snapshot_valid_until_ts,
            "observed_rows": int(self.observed_rows),
            "label_counts": dict(self.label_counts),
            "semantic_score_placeholder": self.semantic_score_placeholder,
            "peer_metadata_sidecar_status": self.peer_metadata_sidecar_status,
            "shape_preserving": True,
            "production_enabled": False,
            "production_outputs_written": False,
            "canonical_production_state_outputs_written": False,
        }


def _optional_nonnegative_int(value: object | None, field_name: str) -> int | None:
    if value is None:
        return None
    try:
        out = int(value)
    except Exception as exc:
        raise ValueError(f"Cross-Asset-State selected profile {field_name} must be an integer") from exc
    if out < 0:
        raise ValueError(f"Cross-Asset-State selected profile {field_name} must be non-negative")
    return out


def _optional_float(value: object | None, field_name: str) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except Exception as exc:
        raise ValueError(f"Cross-Asset-State selected profile {field_name} must be numeric") from exc
