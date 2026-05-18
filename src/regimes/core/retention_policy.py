from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from src.regimes.core.contracts import CANONICAL_SCHEMA_VERSION, require_schema_version
from src.regimes.core.serialization import dumps_json, loads_json, require_json_object, to_jsonable


REGIME_RETENTION_POLICY_ARTIFACT_KIND = "regime_retention_policy"
REGIME_RETENTION_POLICY_SCHEMA_VERSION = CANONICAL_SCHEMA_VERSION
DEFAULT_NON_PRODUCTION_RETENTION = "manual_cleanup_only"
DEFAULT_PRODUCTION_OUTPUT_RETENTION = "no_delete_placeholder"


@dataclass(frozen=True)
class RegimeRetentionPolicy:
    diagnostic_artifact_retention: str | Mapping[str, Any] = DEFAULT_NON_PRODUCTION_RETENTION
    sandbox_output_retention: str | Mapping[str, Any] = DEFAULT_NON_PRODUCTION_RETENTION
    profile_manifest_retention: str | Mapping[str, Any] = DEFAULT_NON_PRODUCTION_RETENTION
    production_output_retention: str | Mapping[str, Any] = DEFAULT_PRODUCTION_OUTPUT_RETENTION
    max_intermediate_artifact_mb: int | float | None = None
    cleanup_enabled: bool = False
    schema_version: int = REGIME_RETENTION_POLICY_SCHEMA_VERSION
    artifact_kind: str = REGIME_RETENTION_POLICY_ARTIFACT_KIND

    def __post_init__(self) -> None:
        if self.max_intermediate_artifact_mb is not None:
            cap = float(self.max_intermediate_artifact_mb)
            if cap <= 0.0:
                raise ValueError("Regime retention max_intermediate_artifact_mb must be positive when supplied")
            object.__setattr__(self, "max_intermediate_artifact_mb", cap)
        _validate_production_retention_safe(self.production_output_retention)
        object.__setattr__(self, "schema_version", require_schema_version(self.schema_version))
        object.__setattr__(self, "artifact_kind", _text(self.artifact_kind, field_name="artifact_kind"))
        object.__setattr__(self, "cleanup_enabled", bool(self.cleanup_enabled))

    @property
    def production_deletion_enabled(self) -> bool:
        return False

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "artifact_kind": self.artifact_kind,
            "diagnostic_artifact_retention": to_jsonable(self.diagnostic_artifact_retention),
            "sandbox_output_retention": to_jsonable(self.sandbox_output_retention),
            "profile_manifest_retention": to_jsonable(self.profile_manifest_retention),
            "production_output_retention": to_jsonable(self.production_output_retention),
            "max_intermediate_artifact_mb": self.max_intermediate_artifact_mb,
            "cleanup_enabled": bool(self.cleanup_enabled),
            "production_deletion_enabled": False,
            "cleanup_job_implemented": False,
        }

    def to_json(self, **kwargs: Any) -> str:
        return dumps_json(self.as_dict(), **kwargs)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RegimeRetentionPolicy":
        obj = require_json_object(payload, context="RegimeRetentionPolicy")
        return cls(
            schema_version=obj.get("schema_version", REGIME_RETENTION_POLICY_SCHEMA_VERSION),
            artifact_kind=obj.get("artifact_kind", REGIME_RETENTION_POLICY_ARTIFACT_KIND),
            diagnostic_artifact_retention=obj.get("diagnostic_artifact_retention", DEFAULT_NON_PRODUCTION_RETENTION),
            sandbox_output_retention=obj.get("sandbox_output_retention", DEFAULT_NON_PRODUCTION_RETENTION),
            profile_manifest_retention=obj.get("profile_manifest_retention", DEFAULT_NON_PRODUCTION_RETENTION),
            production_output_retention=obj.get("production_output_retention", DEFAULT_PRODUCTION_OUTPUT_RETENTION),
            max_intermediate_artifact_mb=obj.get("max_intermediate_artifact_mb"),
            cleanup_enabled=bool(obj.get("cleanup_enabled", False)),
        )

    @classmethod
    def from_json(cls, text: str) -> "RegimeRetentionPolicy":
        return cls.from_dict(require_json_object(loads_json(text), context="RegimeRetentionPolicy JSON"))


def _text(value: object, *, field_name: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"Regime retention {field_name} must be non-empty")
    return text


def _validate_production_retention_safe(policy: str | Mapping[str, Any]) -> None:
    if isinstance(policy, Mapping):
        payload = dict(policy)
        for key in ("delete_enabled", "cleanup_enabled", "purge_enabled", "remove_enabled"):
            if bool(payload.get(key, False)):
                raise ValueError("Regime production_output_retention cannot enable deletion")
        return
    text = _text(policy, field_name="production_output_retention").lower()
    safe_values = {
        "no_delete_placeholder",
        "never_delete",
        "retain_forever",
        "manual_only",
        "manual_cleanup_only",
    }
    if text in safe_values:
        return
    unsafe_tokens = ("delete", "purge", "remove", "expire", "ttl")
    if any(token in text for token in unsafe_tokens):
        raise ValueError("Regime production_output_retention cannot enable deletion")


__all__ = [
    "DEFAULT_NON_PRODUCTION_RETENTION",
    "DEFAULT_PRODUCTION_OUTPUT_RETENTION",
    "REGIME_RETENTION_POLICY_ARTIFACT_KIND",
    "REGIME_RETENTION_POLICY_SCHEMA_VERSION",
    "RegimeRetentionPolicy",
]
