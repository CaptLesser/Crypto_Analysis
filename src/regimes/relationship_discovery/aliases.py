from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from src.regimes.core.contracts import require_schema_version
from src.regimes.relationship_discovery.canonical import RELATIONSHIP_FAMILY_RESIDUAL_PEER
from src.regimes.relationship_discovery.contracts import (
    EDGE_ALIAS_MANIFEST_ARTIFACT_KIND,
    RELATIONSHIP_DISCOVERY_SCHEMA_VERSION,
    RelationshipRefitSnapshot,
)
from src.regimes.relationship_discovery.methods import METHOD_RESIDUAL_CORR
from src.regimes.relationship_discovery.selection import (
    RelationshipEdgeSelectionResult,
    SelectedRelationshipEdge,
)


DEFAULT_EDGE_ALIAS_SLOTS: tuple[str, ...] = (
    "strongest_peer_slot_1",
    "strongest_peer_slot_2",
    "strongest_peer_slot_3",
)

EDGE_ALIAS_ACTIVATION_STATUS_ACTIVE = "active"
EDGE_ALIAS_ACTIVATION_STATUS_INACTIVE = "inactive"


@dataclass(frozen=True)
class EdgeAliasPolicy:
    slots: Sequence[str] = DEFAULT_EDGE_ALIAS_SLOTS
    relationship_types: Sequence[str] = (METHOD_RESIDUAL_CORR,)
    relationship_family: str = RELATIONSHIP_FAMILY_RESIDUAL_PEER
    hysteresis_enabled: bool = False
    hysteresis_min_improvement: float = 0.0

    def __post_init__(self) -> None:
        slots = tuple(dict.fromkeys(_text(slot, field_name="slot") for slot in self.slots))
        if not slots:
            raise ValueError("Relationship Discovery edge alias policy requires at least one slot")
        relationship_types = tuple(dict.fromkeys(_text(value, field_name="relationship_type").lower() for value in self.relationship_types))
        if not relationship_types:
            raise ValueError("Relationship Discovery edge alias policy requires at least one relationship_type")
        hysteresis_min_improvement = _non_negative(self.hysteresis_min_improvement, field_name="hysteresis_min_improvement")
        object.__setattr__(self, "slots", slots)
        object.__setattr__(self, "relationship_types", relationship_types)
        object.__setattr__(self, "relationship_family", _text(self.relationship_family, field_name="relationship_family"))
        object.__setattr__(self, "hysteresis_enabled", bool(self.hysteresis_enabled))
        object.__setattr__(self, "hysteresis_min_improvement", hysteresis_min_improvement)

    def as_dict(self) -> dict[str, Any]:
        return {
            "artifact_kind": "edge_alias_policy",
            "slots": list(self.slots),
            "relationship_types": list(self.relationship_types),
            "relationship_family": self.relationship_family,
            "hysteresis_enabled": bool(self.hysteresis_enabled),
            "hysteresis_min_improvement": float(self.hysteresis_min_improvement),
            "v1_behavior": "deterministic rank-based slot assignment at refit boundaries; no intra-refit identity changes",
            "identity_metadata_sidecar_only": True,
            "production_enabled": False,
        }


@dataclass(frozen=True)
class RelationshipEdgeAliasManifestRow:
    asset: str
    refit_key: str
    interval: int
    window: int
    slot: str
    alias_name: str
    related_asset: str
    relationship_family: str
    method_id: str
    strength: float
    stability_score: float
    activation_status: str
    effective_start_ts: int | float | str
    effective_end_ts: int | float | str
    known_at_ts: int | float | str
    source_tail_ts: int | float | str
    lineage_id: str
    schema_version: int = RELATIONSHIP_DISCOVERY_SCHEMA_VERSION
    artifact_kind: str = EDGE_ALIAS_MANIFEST_ARTIFACT_KIND

    def __post_init__(self) -> None:
        object.__setattr__(self, "schema_version", require_schema_version(self.schema_version))
        object.__setattr__(self, "artifact_kind", _text(self.artifact_kind, field_name="artifact_kind"))
        object.__setattr__(self, "asset", _text(self.asset, field_name="asset"))
        object.__setattr__(self, "refit_key", _text(self.refit_key, field_name="refit_key"))
        object.__setattr__(self, "interval", _positive_int(self.interval, field_name="interval"))
        object.__setattr__(self, "window", _positive_int(self.window, field_name="window"))
        object.__setattr__(self, "slot", _text(self.slot, field_name="slot"))
        object.__setattr__(self, "alias_name", _text(self.alias_name, field_name="alias_name"))
        object.__setattr__(self, "related_asset", _text(self.related_asset, field_name="related_asset"))
        object.__setattr__(self, "relationship_family", _text(self.relationship_family, field_name="relationship_family"))
        object.__setattr__(self, "method_id", _text(self.method_id, field_name="method_id"))
        object.__setattr__(self, "strength", _non_negative(self.strength, field_name="strength"))
        object.__setattr__(self, "stability_score", _share(self.stability_score, field_name="stability_score"))
        activation = _text(self.activation_status, field_name="activation_status").lower()
        if activation not in {EDGE_ALIAS_ACTIVATION_STATUS_ACTIVE, EDGE_ALIAS_ACTIVATION_STATUS_INACTIVE}:
            raise ValueError("Relationship Discovery edge alias activation_status must be active or inactive")
        object.__setattr__(self, "activation_status", activation)
        _timestamp(self.effective_start_ts, field_name="effective_start_ts")
        _timestamp(self.effective_end_ts, field_name="effective_end_ts")
        _timestamp(self.known_at_ts, field_name="known_at_ts")
        _timestamp(self.source_tail_ts, field_name="source_tail_ts")
        if _orderable(self.source_tail_ts, field_name="source_tail_ts") > _orderable(self.known_at_ts, field_name="known_at_ts"):
            raise ValueError("Relationship Discovery edge alias source_tail_ts must not exceed known_at_ts")
        object.__setattr__(self, "lineage_id", _text(self.lineage_id, field_name="lineage_id"))

    def as_dict(self) -> dict[str, Any]:
        return {
            "artifact_kind": self.artifact_kind,
            "asset": self.asset,
            "refit_key": self.refit_key,
            "interval": int(self.interval),
            "window": int(self.window),
            "slot": self.slot,
            "alias_name": self.alias_name,
            "related_asset": self.related_asset,
            "relationship_family": self.relationship_family,
            "method_id": self.method_id,
            "strength": float(self.strength),
            "stability_score": float(self.stability_score),
            "activation_status": self.activation_status,
            "effective_start_ts": self.effective_start_ts,
            "effective_end_ts": self.effective_end_ts,
            "known_at_ts": self.known_at_ts,
            "source_tail_ts": self.source_tail_ts,
            "lineage_id": self.lineage_id,
            "schema_version": int(self.schema_version),
            "identity_metadata_sidecar_only": True,
            "core_feature_column_required": False,
            "production_enabled": False,
        }


def build_edge_alias_manifest_rows(
    selection_result: RelationshipEdgeSelectionResult,
    *,
    refit_snapshot: RelationshipRefitSnapshot,
    policy: EdgeAliasPolicy | None = None,
) -> tuple[RelationshipEdgeAliasManifestRow, ...]:
    policy = policy or EdgeAliasPolicy()
    grouped: dict[tuple[str, int, int], list[SelectedRelationshipEdge]] = {}
    relationship_types = set(policy.relationship_types)
    for selected in selection_result.selected_edges:
        edge = selected.edge
        if edge.relationship_type.lower() not in relationship_types:
            continue
        grouped.setdefault((edge.asset, int(edge.interval), int(edge.window)), []).append(selected)

    rows: list[RelationshipEdgeAliasManifestRow] = []
    for (asset, interval, window), values in sorted(grouped.items()):
        ordered = sorted(
            values,
            key=lambda item: (
                int(item.selection_rank),
                -float(item.edge.abs_value),
                item.edge.related_asset_or_benchmark,
                item.edge.method_id,
            ),
        )
        for slot, selected in zip(policy.slots, ordered):
            edge = selected.edge
            rows.append(
                RelationshipEdgeAliasManifestRow(
                    asset=asset,
                    refit_key=refit_snapshot.refit_key,
                    interval=interval,
                    window=window,
                    slot=slot,
                    alias_name=f"{asset}_{slot}",
                    related_asset=edge.related_asset_or_benchmark,
                    relationship_family=policy.relationship_family,
                    method_id=edge.method_id,
                    strength=float(edge.abs_value),
                    stability_score=_stability_score(selected),
                    activation_status=EDGE_ALIAS_ACTIVATION_STATUS_ACTIVE if selected.selected else EDGE_ALIAS_ACTIVATION_STATUS_INACTIVE,
                    effective_start_ts=refit_snapshot.snapshot_start,
                    effective_end_ts=refit_snapshot.snapshot_end,
                    known_at_ts=edge.known_at_ts,
                    source_tail_ts=edge.known_at_ts,
                    lineage_id=edge.lineage_id,
                )
            )
    return tuple(rows)


def _stability_score(selected: SelectedRelationshipEdge) -> float:
    values = (
        float(selected.survival_share),
        float(selected.sign_stability),
        float(selected.rank_stability),
    )
    return min(1.0, max(0.0, sum(values) / len(values)))


def _text(value: object, *, field_name: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"Relationship Discovery edge alias {field_name} must be non-empty")
    return text


def _positive_int(value: object, *, field_name: str) -> int:
    try:
        out = int(value)
    except Exception as exc:
        raise ValueError(f"Relationship Discovery edge alias {field_name} must be an integer") from exc
    if out <= 0:
        raise ValueError(f"Relationship Discovery edge alias {field_name} must be positive")
    return out


def _non_negative(value: object, *, field_name: str) -> float:
    try:
        out = float(value)
    except Exception as exc:
        raise ValueError(f"Relationship Discovery edge alias {field_name} must be numeric") from exc
    if out < 0.0 or out in {float("inf"), float("-inf")} or out != out:
        raise ValueError(f"Relationship Discovery edge alias {field_name} must be finite and non-negative")
    return out


def _share(value: object, *, field_name: str) -> float:
    out = _non_negative(value, field_name=field_name)
    if out > 1.0:
        raise ValueError(f"Relationship Discovery edge alias {field_name} must be between 0 and 1")
    return out


def _timestamp(value: object, *, field_name: str) -> None:
    if value is None or isinstance(value, bool):
        raise ValueError(f"Relationship Discovery edge alias {field_name} must be timestamp-compatible")
    if isinstance(value, (int, float)):
        return
    _text(value, field_name=field_name)


def _orderable(value: object, *, field_name: str) -> float:
    _timestamp(value, field_name=field_name)
    try:
        return float(value)
    except Exception:
        return float(_text(value, field_name=field_name))


__all__ = [
    "DEFAULT_EDGE_ALIAS_SLOTS",
    "EDGE_ALIAS_ACTIVATION_STATUS_ACTIVE",
    "EDGE_ALIAS_ACTIVATION_STATUS_INACTIVE",
    "EdgeAliasPolicy",
    "RelationshipEdgeAliasManifestRow",
    "build_edge_alias_manifest_rows",
]
