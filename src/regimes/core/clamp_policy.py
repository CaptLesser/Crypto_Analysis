from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Mapping, Sequence

from src.regimes.core.contracts import CANONICAL_SCHEMA_VERSION, require_schema_version
from src.regimes.core.lineage import REGIME_LINEAGE_PATHWAYS
from src.regimes.core.serialization import dumps_json, loads_json, require_json_object


REGIME_CLAMP_POLICY_ARTIFACT_KIND = "regime_clamp_policy"
REGIME_CLAMP_POLICY_SCHEMA_VERSION = CANONICAL_SCHEMA_VERSION
CLAMP_STATUS_CONFIGURED = "configured"
CLAMP_STATUS_NOT_CONFIGURED = "clamp_not_configured"


@dataclass(frozen=True)
class RegimeClampPolicy:
    policy_id: str
    reason: str
    applies_to_pathways: Sequence[str]
    historical_output_start: int | float | str | None = None
    historical_output_end: int | float | str | None = None
    live_start: int | float | str | None = None
    schema_version: int = REGIME_CLAMP_POLICY_SCHEMA_VERSION
    artifact_kind: str = REGIME_CLAMP_POLICY_ARTIFACT_KIND

    def __post_init__(self) -> None:
        pathways = tuple(dict.fromkeys(_pathway(pathway) for pathway in self.applies_to_pathways))
        if not pathways:
            raise ValueError("Regime clamp policy applies_to_pathways must include at least one pathway")
        configured = self.historical_output_start is not None
        if not configured and (self.historical_output_end is not None or self.live_start is not None):
            raise ValueError("Regime clamp policy requires historical_output_start when any clamp boundary is configured")
        if configured:
            start = _to_orderable(self.historical_output_start, field_name="historical_output_start")
            if self.historical_output_end is not None:
                end = _to_orderable(self.historical_output_end, field_name="historical_output_end")
                if end < start:
                    raise ValueError("Regime clamp historical_output_end must be >= historical_output_start")
            if self.live_start is not None:
                live = _to_orderable(self.live_start, field_name="live_start")
                if live < start:
                    raise ValueError("Regime clamp live_start must be >= historical_output_start")
        object.__setattr__(self, "schema_version", require_schema_version(self.schema_version))
        object.__setattr__(self, "artifact_kind", _text(self.artifact_kind, field_name="artifact_kind"))
        object.__setattr__(self, "policy_id", _text(self.policy_id, field_name="policy_id"))
        object.__setattr__(self, "reason", _text(self.reason, field_name="reason"))
        object.__setattr__(self, "applies_to_pathways", pathways)

    @property
    def status(self) -> str:
        return CLAMP_STATUS_CONFIGURED if self.historical_output_start is not None else CLAMP_STATUS_NOT_CONFIGURED

    def validate_output_ts(self, ts: int | float | str, *, pathway: str) -> None:
        target_pathway = _pathway(pathway)
        if target_pathway not in self.applies_to_pathways:
            raise ValueError(f"Regime clamp policy {self.policy_id!r} does not apply to pathway {target_pathway!r}")
        if self.status == CLAMP_STATUS_NOT_CONFIGURED:
            raise ValueError("Regime clamp policy is clamp_not_configured; explicit clamp config is required")
        value = _to_orderable(ts, field_name="ts")
        start = _to_orderable(self.historical_output_start, field_name="historical_output_start")
        if value < start:
            raise ValueError("Regime output timestamp is before historical_output_start")
        if self.historical_output_end is None:
            return
        end = _to_orderable(self.historical_output_end, field_name="historical_output_end")
        if value <= end:
            return
        if self.live_start is not None and value >= _to_orderable(self.live_start, field_name="live_start"):
            return
        raise ValueError("Regime output timestamp is outside configured historical/live clamp windows")

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "artifact_kind": self.artifact_kind,
            "policy_id": self.policy_id,
            "status": self.status,
            "historical_output_start": self.historical_output_start,
            "historical_output_end": self.historical_output_end,
            "live_start": self.live_start,
            "reason": self.reason,
            "applies_to_pathways": list(self.applies_to_pathways),
        }

    def to_json(self, **kwargs: Any) -> str:
        return dumps_json(self.as_dict(), **kwargs)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RegimeClampPolicy":
        obj = require_json_object(payload, context="RegimeClampPolicy")
        return cls(
            schema_version=obj.get("schema_version", REGIME_CLAMP_POLICY_SCHEMA_VERSION),
            artifact_kind=obj.get("artifact_kind", REGIME_CLAMP_POLICY_ARTIFACT_KIND),
            policy_id=obj["policy_id"],
            historical_output_start=obj.get("historical_output_start"),
            historical_output_end=obj.get("historical_output_end"),
            live_start=obj.get("live_start"),
            reason=obj["reason"],
            applies_to_pathways=obj["applies_to_pathways"],
        )

    @classmethod
    def from_json(cls, text: str) -> "RegimeClampPolicy":
        return cls.from_dict(require_json_object(loads_json(text), context="RegimeClampPolicy JSON"))


def validate_regime_output_ts_against_clamp(
    ts: int | float | str,
    *,
    pathway: str,
    clamp_policy: RegimeClampPolicy | Mapping[str, Any],
) -> None:
    policy = clamp_policy if isinstance(clamp_policy, RegimeClampPolicy) else RegimeClampPolicy.from_dict(clamp_policy)
    policy.validate_output_ts(ts, pathway=pathway)


def _text(value: object, *, field_name: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"Regime clamp {field_name} must be non-empty")
    return text


def _pathway(value: object) -> str:
    text = _text(value, field_name="pathway").lower()
    if text not in REGIME_LINEAGE_PATHWAYS:
        valid = ", ".join(REGIME_LINEAGE_PATHWAYS)
        raise ValueError(f"Unsupported Regime clamp pathway {text!r}; expected one of: {valid}")
    return text


def _to_orderable(value: object, *, field_name: str) -> float:
    if isinstance(value, bool) or value is None:
        raise ValueError(f"Regime clamp {field_name} must be a timestamp")
    try:
        return float(value)
    except Exception:
        pass
    text = _text(value, field_name=field_name)
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()
    except Exception as exc:
        raise ValueError(f"Regime clamp {field_name} must be numeric or ISO datetime") from exc


__all__ = [
    "CLAMP_STATUS_CONFIGURED",
    "CLAMP_STATUS_NOT_CONFIGURED",
    "REGIME_CLAMP_POLICY_ARTIFACT_KIND",
    "REGIME_CLAMP_POLICY_SCHEMA_VERSION",
    "RegimeClampPolicy",
    "validate_regime_output_ts_against_clamp",
]
