from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from src.regimes.core.contracts import CANONICAL_SCHEMA_VERSION, require_schema_version
from src.regimes.core.serialization import dumps_json, loads_json, require_json_object, to_jsonable


REGIME_LINEAGE_ARTIFACT_KIND = "regime_lineage_spec"
REGIME_LINEAGE_SCHEMA_VERSION = CANONICAL_SCHEMA_VERSION
REGIME_LINEAGE_PATHWAYS: tuple[str, ...] = ("asset_state", "market_state", "relative_state")


@dataclass(frozen=True)
class RegimeLineageSpec:
    pathway: str
    axis: str
    band: str
    interval: int
    profile_id: str
    clusterer_family: str
    source_data_kind: str
    source_partition_lineage: Sequence[Mapping[str, Any]]
    source_tail_ts: int | float | str
    train_window_start: int | float | str
    train_window_end: int | float | str
    score_window_start: int | float | str
    score_window_end: int | float | str
    generated_at: int | float | str
    run_id: str
    feature_profile_id: str | None = None
    feature_family_id: str | None = None
    schema_version: int = REGIME_LINEAGE_SCHEMA_VERSION
    artifact_kind: str = REGIME_LINEAGE_ARTIFACT_KIND
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        pathway = _member(self.pathway, REGIME_LINEAGE_PATHWAYS, field_name="pathway")
        interval = int(self.interval)
        if interval <= 0:
            raise ValueError("Regime lineage interval must be positive")
        source_lineage = _mapping_tuple(self.source_partition_lineage, field_name="source_partition_lineage")
        if not source_lineage:
            raise ValueError("Regime lineage source_partition_lineage must include at least one entry")
        feature_profile = _optional_text(self.feature_profile_id)
        feature_family = _optional_text(self.feature_family_id)
        if not (feature_profile or feature_family):
            raise ValueError("Regime lineage requires feature_profile_id or feature_family_id")
        _validate_order(self.train_window_start, self.train_window_end, context="train window")
        _validate_order(self.score_window_start, self.score_window_end, context="score window")
        if _to_orderable(self.source_tail_ts, field_name="source_tail_ts") > _to_orderable(
            self.generated_at,
            field_name="generated_at",
        ):
            raise ValueError("Regime lineage source_tail_ts must not exceed generated_at")
        object.__setattr__(self, "schema_version", require_schema_version(self.schema_version))
        object.__setattr__(self, "artifact_kind", _text(self.artifact_kind, field_name="artifact_kind"))
        object.__setattr__(self, "pathway", pathway)
        object.__setattr__(self, "axis", _text(self.axis, field_name="axis").lower())
        object.__setattr__(self, "band", _text(self.band, field_name="band").lower())
        object.__setattr__(self, "interval", interval)
        object.__setattr__(self, "profile_id", _text(self.profile_id, field_name="profile_id"))
        object.__setattr__(self, "feature_profile_id", feature_profile)
        object.__setattr__(self, "feature_family_id", feature_family)
        object.__setattr__(self, "clusterer_family", _text(self.clusterer_family, field_name="clusterer_family"))
        object.__setattr__(self, "source_data_kind", _text(self.source_data_kind, field_name="source_data_kind"))
        object.__setattr__(self, "source_partition_lineage", source_lineage)
        object.__setattr__(self, "run_id", _text(self.run_id, field_name="run_id"))
        object.__setattr__(self, "metadata", to_jsonable(dict(self.metadata)))

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "artifact_kind": self.artifact_kind,
            "pathway": self.pathway,
            "axis": self.axis,
            "band": self.band,
            "interval": int(self.interval),
            "profile_id": self.profile_id,
            "feature_profile_id": self.feature_profile_id,
            "feature_family_id": self.feature_family_id,
            "clusterer_family": self.clusterer_family,
            "source_data_kind": self.source_data_kind,
            "source_partition_lineage": [dict(item) for item in self.source_partition_lineage],
            "source_tail_ts": self.source_tail_ts,
            "train_window_start": self.train_window_start,
            "train_window_end": self.train_window_end,
            "score_window_start": self.score_window_start,
            "score_window_end": self.score_window_end,
            "generated_at": self.generated_at,
            "run_id": self.run_id,
            "metadata": to_jsonable(dict(self.metadata)),
        }

    def to_json(self, **kwargs: Any) -> str:
        return dumps_json(self.as_dict(), **kwargs)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RegimeLineageSpec":
        obj = require_json_object(payload, context="RegimeLineageSpec")
        return cls(
            schema_version=obj.get("schema_version", REGIME_LINEAGE_SCHEMA_VERSION),
            artifact_kind=obj.get("artifact_kind", REGIME_LINEAGE_ARTIFACT_KIND),
            pathway=obj["pathway"],
            axis=obj["axis"],
            band=obj["band"],
            interval=obj["interval"],
            profile_id=obj["profile_id"],
            feature_profile_id=obj.get("feature_profile_id"),
            feature_family_id=obj.get("feature_family_id"),
            clusterer_family=obj["clusterer_family"],
            source_data_kind=obj["source_data_kind"],
            source_partition_lineage=obj["source_partition_lineage"],
            source_tail_ts=obj["source_tail_ts"],
            train_window_start=obj["train_window_start"],
            train_window_end=obj["train_window_end"],
            score_window_start=obj["score_window_start"],
            score_window_end=obj["score_window_end"],
            generated_at=obj["generated_at"],
            run_id=obj["run_id"],
            metadata=obj.get("metadata", {}),
        )

    @classmethod
    def from_json(cls, text: str) -> "RegimeLineageSpec":
        return cls.from_dict(require_json_object(loads_json(text), context="RegimeLineageSpec JSON"))


def _text(value: object, *, field_name: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"Regime lineage {field_name} must be non-empty")
    return text


def _optional_text(value: object | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _member(value: object, allowed: Sequence[str], *, field_name: str) -> str:
    text = _text(value, field_name=field_name).lower()
    if text not in allowed:
        valid = ", ".join(str(item) for item in allowed)
        raise ValueError(f"Unsupported Regime lineage {field_name} {text!r}; expected one of: {valid}")
    return text


def _mapping_tuple(values: Sequence[Mapping[str, Any]], *, field_name: str) -> tuple[dict[str, Any], ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValueError(f"Regime lineage {field_name} must be a sequence of JSON objects")
    out: list[dict[str, Any]] = []
    for value in values:
        if hasattr(value, "as_dict"):
            value = value.as_dict()
        if not isinstance(value, Mapping):
            raise ValueError(f"Regime lineage {field_name} entries must be JSON objects")
        out.append(to_jsonable(dict(value)))
    return tuple(out)


def _to_orderable(value: object, *, field_name: str) -> float:
    if isinstance(value, bool) or value is None:
        raise ValueError(f"Regime lineage {field_name} must be a timestamp")
    try:
        return float(value)
    except Exception:
        pass
    from datetime import datetime

    text = _text(value, field_name=field_name)
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()
    except Exception as exc:
        raise ValueError(f"Regime lineage {field_name} must be numeric or ISO datetime") from exc


def _validate_order(start: object, end: object, *, context: str) -> None:
    if _to_orderable(start, field_name=f"{context} start") > _to_orderable(end, field_name=f"{context} end"):
        raise ValueError(f"Regime lineage {context} start must be <= end")


__all__ = [
    "REGIME_LINEAGE_ARTIFACT_KIND",
    "REGIME_LINEAGE_PATHWAYS",
    "REGIME_LINEAGE_SCHEMA_VERSION",
    "RegimeLineageSpec",
]
