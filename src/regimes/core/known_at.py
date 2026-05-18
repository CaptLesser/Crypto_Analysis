from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Mapping

from src.regimes.core.contracts import CANONICAL_SCHEMA_VERSION, require_schema_version
from src.regimes.core.serialization import dumps_json, loads_json, require_json_object


KNOWN_AT_ARTIFACT_KIND = "regime_known_at_spec"
KNOWN_AT_SCHEMA_VERSION = CANONICAL_SCHEMA_VERSION


@dataclass(frozen=True)
class KnownAtSpec:
    ts: int | float | str
    known_at_ts: int | float | str
    source_tail_ts: int | float | str
    label_available_at_ts: int | float | str
    alignment_policy: str
    latency_policy: str
    no_lookahead_verified: bool
    schema_version: int = KNOWN_AT_SCHEMA_VERSION
    artifact_kind: str = KNOWN_AT_ARTIFACT_KIND

    def __post_init__(self) -> None:
        ts = _to_orderable(self.ts, field_name="ts")
        known_at = _to_orderable(self.known_at_ts, field_name="known_at_ts")
        source_tail = _to_orderable(self.source_tail_ts, field_name="source_tail_ts")
        label_available = _to_orderable(self.label_available_at_ts, field_name="label_available_at_ts")
        if known_at < ts:
            raise ValueError("Regime known_at_ts must be >= ts")
        if source_tail > known_at:
            raise ValueError("Regime source_tail_ts must not exceed known_at_ts")
        if label_available < known_at:
            raise ValueError("Regime label_available_at_ts must be >= known_at_ts")
        if self.no_lookahead_verified is not True:
            raise ValueError("Regime known-at contract requires no_lookahead_verified=true")
        object.__setattr__(self, "schema_version", require_schema_version(self.schema_version))
        object.__setattr__(self, "artifact_kind", _text(self.artifact_kind, field_name="artifact_kind"))
        object.__setattr__(self, "alignment_policy", _text(self.alignment_policy, field_name="alignment_policy"))
        object.__setattr__(self, "latency_policy", _text(self.latency_policy, field_name="latency_policy"))
        object.__setattr__(self, "no_lookahead_verified", True)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "artifact_kind": self.artifact_kind,
            "ts": self.ts,
            "known_at_ts": self.known_at_ts,
            "source_tail_ts": self.source_tail_ts,
            "label_available_at_ts": self.label_available_at_ts,
            "alignment_policy": self.alignment_policy,
            "latency_policy": self.latency_policy,
            "no_lookahead_verified": True,
        }

    def to_json(self, **kwargs: Any) -> str:
        return dumps_json(self.as_dict(), **kwargs)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "KnownAtSpec":
        obj = require_json_object(payload, context="KnownAtSpec")
        return cls(
            schema_version=obj.get("schema_version", KNOWN_AT_SCHEMA_VERSION),
            artifact_kind=obj.get("artifact_kind", KNOWN_AT_ARTIFACT_KIND),
            ts=obj["ts"],
            known_at_ts=obj["known_at_ts"],
            source_tail_ts=obj["source_tail_ts"],
            label_available_at_ts=obj["label_available_at_ts"],
            alignment_policy=obj["alignment_policy"],
            latency_policy=obj["latency_policy"],
            no_lookahead_verified=bool(obj["no_lookahead_verified"]),
        )

    @classmethod
    def from_json(cls, text: str) -> "KnownAtSpec":
        return cls.from_dict(require_json_object(loads_json(text), context="KnownAtSpec JSON"))


def _text(value: object, *, field_name: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"Regime known-at {field_name} must be non-empty")
    return text


def _to_orderable(value: object, *, field_name: str) -> float:
    if isinstance(value, bool) or value is None:
        raise ValueError(f"Regime known-at {field_name} must be a timestamp")
    try:
        return float(value)
    except Exception:
        pass
    text = _text(value, field_name=field_name)
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()
    except Exception as exc:
        raise ValueError(f"Regime known-at {field_name} must be numeric or ISO datetime") from exc


__all__ = [
    "KNOWN_AT_ARTIFACT_KIND",
    "KNOWN_AT_SCHEMA_VERSION",
    "KnownAtSpec",
]
