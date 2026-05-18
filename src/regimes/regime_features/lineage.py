from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Mapping, Sequence

from src.regimes.core.clamp_policy import RegimeClampPolicy
from src.regimes.core.contracts import CANONICAL_SCHEMA_VERSION, require_schema_version
from src.regimes.core.retention_policy import RegimeRetentionPolicy
from src.regimes.core.serialization import dumps_json, loads_json, require_json_object, to_jsonable


REGIME_FEATURES_LAYER = "regime_features"
REGIME_FEATURES_SCHEMA_VERSION = CANONICAL_SCHEMA_VERSION

REGIME_FEATURE_KNOWN_AT_ARTIFACT_KIND = "regime_feature_known_at_spec"
REGIME_FEATURE_LINEAGE_ARTIFACT_KIND = "regime_feature_lineage_spec"
REGIME_FEATURE_CLAMP_REF_ARTIFACT_KIND = "regime_feature_clamp_ref"
REGIME_FEATURE_RETENTION_HINT_ARTIFACT_KIND = "regime_feature_retention_hint"

MARKET_REGIME_FEATURES = "market_regime_features"
PAIRWISE_RELATIONSHIP_FEATURES = "pairwise_relationship_features"
CROSS_ASSET_SUMMARY_FEATURES = "cross_asset_summary_features"
ELIGIBILITY_SNAPSHOT = "eligibility_snapshot"
UNIVERSE_SNAPSHOT = "universe_snapshot"

REGIME_FEATURE_ARTIFACT_FAMILIES: tuple[str, ...] = (
    MARKET_REGIME_FEATURES,
    PAIRWISE_RELATIONSHIP_FEATURES,
    CROSS_ASSET_SUMMARY_FEATURES,
    ELIGIBILITY_SNAPSHOT,
    UNIVERSE_SNAPSHOT,
)

COMPLETED_FEATURE_STATUSES: frozenset[str] = frozenset({"ready", "completed", "written"})
NON_DESTRUCTIVE_PRODUCTION_RETENTION_VALUES: frozenset[str] = frozenset(
    {
        "no_delete_placeholder",
        "never_delete",
        "retain_forever",
        "manual_only",
        "manual_cleanup_only",
    }
)


@dataclass(frozen=True)
class RegimeFeatureKnownAtSpec:
    ts: int | float | str
    known_at_ts: int | float | str
    source_tail_ts: int | float | str
    feature_available_at_ts: int | float | str
    alignment_policy: str = "closed_source_tail"
    latency_policy: str = "same_batch_after_source_tail"
    no_lookahead_verified: bool = True
    build_status: str = "completed"
    schema_version: int = REGIME_FEATURES_SCHEMA_VERSION
    artifact_kind: str = REGIME_FEATURE_KNOWN_AT_ARTIFACT_KIND

    def __post_init__(self) -> None:
        ts = _to_orderable(self.ts, field_name="ts")
        known_at = _to_orderable(self.known_at_ts, field_name="known_at_ts")
        source_tail = _to_orderable(self.source_tail_ts, field_name="source_tail_ts")
        available = _to_orderable(self.feature_available_at_ts, field_name="feature_available_at_ts")
        if known_at < ts:
            raise ValueError("Regime Feature known_at_ts must be >= ts")
        if source_tail > known_at:
            raise ValueError("Regime Feature source_tail_ts must not exceed known_at_ts")
        if available < known_at:
            raise ValueError("Regime Feature feature_available_at_ts must be >= known_at_ts")
        status = _text(self.build_status, field_name="build_status").lower()
        if status in COMPLETED_FEATURE_STATUSES and self.no_lookahead_verified is not True:
            raise ValueError("Regime Feature completed artifacts require no_lookahead_verified=true")
        object.__setattr__(self, "schema_version", require_schema_version(self.schema_version))
        object.__setattr__(self, "artifact_kind", _text(self.artifact_kind, field_name="artifact_kind"))
        object.__setattr__(self, "alignment_policy", _text(self.alignment_policy, field_name="alignment_policy"))
        object.__setattr__(self, "latency_policy", _text(self.latency_policy, field_name="latency_policy"))
        object.__setattr__(self, "build_status", status)
        object.__setattr__(self, "no_lookahead_verified", bool(self.no_lookahead_verified))

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "artifact_kind": self.artifact_kind,
            "layer": REGIME_FEATURES_LAYER,
            "ts": self.ts,
            "known_at_ts": self.known_at_ts,
            "source_tail_ts": self.source_tail_ts,
            "feature_available_at_ts": self.feature_available_at_ts,
            "alignment_policy": self.alignment_policy,
            "latency_policy": self.latency_policy,
            "no_lookahead_verified": bool(self.no_lookahead_verified),
            "build_status": self.build_status,
        }

    def to_json(self, **kwargs: Any) -> str:
        return dumps_json(self.as_dict(), **kwargs)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RegimeFeatureKnownAtSpec":
        obj = require_json_object(payload, context="RegimeFeatureKnownAtSpec")
        return cls(
            schema_version=obj.get("schema_version", REGIME_FEATURES_SCHEMA_VERSION),
            artifact_kind=obj.get("artifact_kind", REGIME_FEATURE_KNOWN_AT_ARTIFACT_KIND),
            ts=obj["ts"],
            known_at_ts=obj["known_at_ts"],
            source_tail_ts=obj["source_tail_ts"],
            feature_available_at_ts=obj["feature_available_at_ts"],
            alignment_policy=obj["alignment_policy"],
            latency_policy=obj["latency_policy"],
            no_lookahead_verified=bool(obj["no_lookahead_verified"]),
            build_status=obj.get("build_status", "completed"),
        )

    @classmethod
    def from_json(cls, text: str) -> "RegimeFeatureKnownAtSpec":
        return cls.from_dict(require_json_object(loads_json(text), context="RegimeFeatureKnownAtSpec JSON"))


@dataclass(frozen=True)
class RegimeFeatureLineageSpec:
    artifact_family: str
    feature_set_id: str
    interval: int
    band: str
    source_data_kinds: Sequence[str]
    source_partition_lineage: Sequence[Mapping[str, Any]]
    source_tail_ts: int | float | str
    feature_window_start: int | float | str
    feature_window_end: int | float | str
    generated_at: int | float | str
    run_id: str
    universe_snapshot_id: str | None = None
    universe_snapshot_hash: str | None = None
    feature_registry_id: str | None = None
    calculation_policy: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = REGIME_FEATURES_SCHEMA_VERSION
    artifact_kind: str = REGIME_FEATURE_LINEAGE_ARTIFACT_KIND

    def __post_init__(self) -> None:
        artifact_family = _artifact_family(self.artifact_family)
        interval = int(self.interval)
        if interval <= 0:
            raise ValueError("Regime Feature lineage interval must be positive")
        source_kinds = _string_tuple(self.source_data_kinds, field_name="source_data_kinds", require_non_empty=True)
        source_lineage = _mapping_tuple(self.source_partition_lineage, field_name="source_partition_lineage")
        if not source_lineage:
            raise ValueError("Regime Feature lineage source_partition_lineage must include at least one entry")
        _validate_source_partition_lineage(source_lineage)
        _validate_order(self.feature_window_start, self.feature_window_end, context="feature calculation window")
        generated = _to_orderable(self.generated_at, field_name="generated_at")
        if _to_orderable(self.source_tail_ts, field_name="source_tail_ts") > generated:
            raise ValueError("Regime Feature lineage source_tail_ts must not exceed generated_at")
        object.__setattr__(self, "schema_version", require_schema_version(self.schema_version))
        object.__setattr__(self, "artifact_kind", _text(self.artifact_kind, field_name="artifact_kind"))
        object.__setattr__(self, "artifact_family", artifact_family)
        object.__setattr__(self, "feature_set_id", _text(self.feature_set_id, field_name="feature_set_id"))
        object.__setattr__(self, "interval", interval)
        object.__setattr__(self, "band", _text(self.band, field_name="band").lower())
        object.__setattr__(self, "source_data_kinds", source_kinds)
        object.__setattr__(self, "source_partition_lineage", source_lineage)
        object.__setattr__(self, "run_id", _text(self.run_id, field_name="run_id"))
        object.__setattr__(self, "universe_snapshot_id", _optional_text(self.universe_snapshot_id))
        object.__setattr__(self, "universe_snapshot_hash", _optional_text(self.universe_snapshot_hash))
        object.__setattr__(self, "feature_registry_id", _optional_text(self.feature_registry_id))
        object.__setattr__(self, "calculation_policy", to_jsonable(dict(self.calculation_policy)))

    @property
    def lineage_id(self) -> str:
        payload = {
            "artifact_family": self.artifact_family,
            "feature_set_id": self.feature_set_id,
            "interval": self.interval,
            "band": self.band,
            "source_tail_ts": self.source_tail_ts,
            "feature_window_start": self.feature_window_start,
            "feature_window_end": self.feature_window_end,
            "run_id": self.run_id,
            "universe_snapshot_id": self.universe_snapshot_id,
            "universe_snapshot_hash": self.universe_snapshot_hash,
            "feature_registry_id": self.feature_registry_id,
            "calculation_policy": self.calculation_policy,
        }
        return hashlib.sha256(dumps_json(payload).encode("utf-8")).hexdigest()[:16]

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "artifact_kind": self.artifact_kind,
            "layer": REGIME_FEATURES_LAYER,
            "lineage_id": self.lineage_id,
            "artifact_family": self.artifact_family,
            "feature_set_id": self.feature_set_id,
            "feature_registry_id": self.feature_registry_id,
            "interval": int(self.interval),
            "band": self.band,
            "source_data_kinds": list(self.source_data_kinds),
            "source_partition_lineage": [dict(item) for item in self.source_partition_lineage],
            "source_tail_ts": self.source_tail_ts,
            "feature_window_start": self.feature_window_start,
            "feature_window_end": self.feature_window_end,
            "generated_at": self.generated_at,
            "run_id": self.run_id,
            "universe_snapshot_id": self.universe_snapshot_id,
            "universe_snapshot_hash": self.universe_snapshot_hash,
            "calculation_policy": to_jsonable(dict(self.calculation_policy)),
            "production_outputs_written": False,
        }

    def to_json(self, **kwargs: Any) -> str:
        return dumps_json(self.as_dict(), **kwargs)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RegimeFeatureLineageSpec":
        obj = require_json_object(payload, context="RegimeFeatureLineageSpec")
        return cls(
            schema_version=obj.get("schema_version", REGIME_FEATURES_SCHEMA_VERSION),
            artifact_kind=obj.get("artifact_kind", REGIME_FEATURE_LINEAGE_ARTIFACT_KIND),
            artifact_family=obj["artifact_family"],
            feature_set_id=obj["feature_set_id"],
            feature_registry_id=obj.get("feature_registry_id"),
            interval=obj["interval"],
            band=obj["band"],
            source_data_kinds=obj["source_data_kinds"],
            source_partition_lineage=obj["source_partition_lineage"],
            source_tail_ts=obj["source_tail_ts"],
            feature_window_start=obj["feature_window_start"],
            feature_window_end=obj["feature_window_end"],
            generated_at=obj["generated_at"],
            run_id=obj["run_id"],
            universe_snapshot_id=obj.get("universe_snapshot_id"),
            universe_snapshot_hash=obj.get("universe_snapshot_hash"),
            calculation_policy=obj.get("calculation_policy", {}),
        )

    @classmethod
    def from_json(cls, text: str) -> "RegimeFeatureLineageSpec":
        return cls.from_dict(require_json_object(loads_json(text), context="RegimeFeatureLineageSpec JSON"))


@dataclass(frozen=True)
class RegimeFeatureClampRef:
    clamp_policy_id: str
    required: bool = False
    clamp_policy: RegimeClampPolicy | Mapping[str, Any] | None = None
    schema_version: int = REGIME_FEATURES_SCHEMA_VERSION
    artifact_kind: str = REGIME_FEATURE_CLAMP_REF_ARTIFACT_KIND

    def __post_init__(self) -> None:
        policy = None
        if self.clamp_policy is not None:
            policy = self.clamp_policy if isinstance(self.clamp_policy, RegimeClampPolicy) else RegimeClampPolicy.from_dict(self.clamp_policy)
            if policy.policy_id != str(self.clamp_policy_id):
                raise ValueError("Regime Feature clamp ref policy_id must match clamp_policy.policy_id")
        object.__setattr__(self, "schema_version", require_schema_version(self.schema_version))
        object.__setattr__(self, "artifact_kind", _text(self.artifact_kind, field_name="artifact_kind"))
        object.__setattr__(self, "clamp_policy_id", _text(self.clamp_policy_id, field_name="clamp_policy_id"))
        object.__setattr__(self, "required", bool(self.required))
        object.__setattr__(self, "clamp_policy", policy)

    @property
    def clamp_dates_hardcoded(self) -> bool:
        return False

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "artifact_kind": self.artifact_kind,
            "layer": REGIME_FEATURES_LAYER,
            "clamp_policy_id": self.clamp_policy_id,
            "required": bool(self.required),
            "clamp_policy": self.clamp_policy.as_dict() if self.clamp_policy is not None else None,
            "clamp_dates_hardcoded": False,
            "core_contract_referenced": "RegimeClampPolicy",
        }

    def to_json(self, **kwargs: Any) -> str:
        return dumps_json(self.as_dict(), **kwargs)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RegimeFeatureClampRef":
        obj = require_json_object(payload, context="RegimeFeatureClampRef")
        return cls(
            schema_version=obj.get("schema_version", REGIME_FEATURES_SCHEMA_VERSION),
            artifact_kind=obj.get("artifact_kind", REGIME_FEATURE_CLAMP_REF_ARTIFACT_KIND),
            clamp_policy_id=obj["clamp_policy_id"],
            required=bool(obj.get("required", False)),
            clamp_policy=obj.get("clamp_policy"),
        )

    @classmethod
    def from_json(cls, text: str) -> "RegimeFeatureClampRef":
        return cls.from_dict(require_json_object(loads_json(text), context="RegimeFeatureClampRef JSON"))


@dataclass(frozen=True)
class RegimeFeatureRetentionHint:
    diagnostic_artifact_retention: str | Mapping[str, Any] = "manual_cleanup_only"
    sandbox_output_retention: str | Mapping[str, Any] = "manual_cleanup_only"
    manifest_retention: str | Mapping[str, Any] = "manual_cleanup_only"
    production_output_retention: str | Mapping[str, Any] = "no_delete_placeholder"
    max_intermediate_artifact_mb: int | float | None = None
    cleanup_enabled: bool = False
    schema_version: int = REGIME_FEATURES_SCHEMA_VERSION
    artifact_kind: str = REGIME_FEATURE_RETENTION_HINT_ARTIFACT_KIND

    def __post_init__(self) -> None:
        _validate_non_destructive_retention(self.production_output_retention)
        core_policy = RegimeRetentionPolicy(
            diagnostic_artifact_retention=self.diagnostic_artifact_retention,
            sandbox_output_retention=self.sandbox_output_retention,
            profile_manifest_retention=self.manifest_retention,
            production_output_retention=self.production_output_retention,
            max_intermediate_artifact_mb=self.max_intermediate_artifact_mb,
            cleanup_enabled=bool(self.cleanup_enabled),
        )
        object.__setattr__(self, "schema_version", require_schema_version(self.schema_version))
        object.__setattr__(self, "artifact_kind", _text(self.artifact_kind, field_name="artifact_kind"))
        object.__setattr__(self, "cleanup_enabled", bool(self.cleanup_enabled))
        object.__setattr__(self, "_core_policy", core_policy)

    @property
    def production_deletion_enabled(self) -> bool:
        return False

    @property
    def cleanup_job_implemented(self) -> bool:
        return False

    @property
    def core_retention_policy(self) -> RegimeRetentionPolicy:
        return self._core_policy

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "artifact_kind": self.artifact_kind,
            "layer": REGIME_FEATURES_LAYER,
            "diagnostic_artifact_retention": to_jsonable(self.diagnostic_artifact_retention),
            "sandbox_output_retention": to_jsonable(self.sandbox_output_retention),
            "manifest_retention": to_jsonable(self.manifest_retention),
            "production_output_retention": to_jsonable(self.production_output_retention),
            "max_intermediate_artifact_mb": self.max_intermediate_artifact_mb,
            "cleanup_enabled": bool(self.cleanup_enabled),
            "production_deletion_enabled": False,
            "cleanup_job_implemented": False,
            "core_contract_referenced": "RegimeRetentionPolicy",
        }

    def to_json(self, **kwargs: Any) -> str:
        return dumps_json(self.as_dict(), **kwargs)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RegimeFeatureRetentionHint":
        obj = require_json_object(payload, context="RegimeFeatureRetentionHint")
        return cls(
            schema_version=obj.get("schema_version", REGIME_FEATURES_SCHEMA_VERSION),
            artifact_kind=obj.get("artifact_kind", REGIME_FEATURE_RETENTION_HINT_ARTIFACT_KIND),
            diagnostic_artifact_retention=obj.get("diagnostic_artifact_retention", "manual_cleanup_only"),
            sandbox_output_retention=obj.get("sandbox_output_retention", "manual_cleanup_only"),
            manifest_retention=obj.get("manifest_retention", "manual_cleanup_only"),
            production_output_retention=obj.get("production_output_retention", "no_delete_placeholder"),
            max_intermediate_artifact_mb=obj.get("max_intermediate_artifact_mb"),
            cleanup_enabled=bool(obj.get("cleanup_enabled", False)),
        )

    @classmethod
    def from_json(cls, text: str) -> "RegimeFeatureRetentionHint":
        return cls.from_dict(require_json_object(loads_json(text), context="RegimeFeatureRetentionHint JSON"))


def _validate_source_partition_lineage(entries: Sequence[Mapping[str, Any]]) -> None:
    for entry in entries:
        source_kind = str(entry.get("source_kind", "") or "").strip().lower()
        if source_kind in {"ohlcvt", "scalar_features"} and not str(entry.get("path", "") or "").strip():
            raise ValueError(f"Regime Feature lineage {source_kind} partition entries require path")
        if source_kind == "universe_snapshot" and not str(entry.get("snapshot_id", "") or "").strip():
            raise ValueError("Regime Feature lineage universe_snapshot entries require snapshot_id")


def _validate_non_destructive_retention(policy: str | Mapping[str, Any]) -> None:
    if isinstance(policy, Mapping):
        payload = dict(policy)
        for key in ("delete_enabled", "cleanup_enabled", "purge_enabled", "remove_enabled"):
            if bool(payload.get(key, False)):
                raise ValueError("Regime Feature production retention cannot enable deletion")
        return
    text = _text(policy, field_name="production_output_retention").lower()
    if text in NON_DESTRUCTIVE_PRODUCTION_RETENTION_VALUES:
        return
    unsafe_tokens = ("delete", "purge", "remove", "expire", "ttl")
    if any(token in text for token in unsafe_tokens):
        raise ValueError("Regime Feature production retention cannot enable deletion")


def _artifact_family(value: object) -> str:
    text = _text(value, field_name="artifact_family").lower()
    if text not in REGIME_FEATURE_ARTIFACT_FAMILIES:
        valid = ", ".join(REGIME_FEATURE_ARTIFACT_FAMILIES)
        raise ValueError(f"Unsupported Regime Feature artifact_family {text!r}; expected one of: {valid}")
    return text


def _mapping_tuple(values: Sequence[Mapping[str, Any]], *, field_name: str) -> tuple[dict[str, Any], ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValueError(f"Regime Feature {field_name} must be a sequence of JSON objects")
    out: list[dict[str, Any]] = []
    for value in values:
        if hasattr(value, "as_dict"):
            value = value.as_dict()
        if not isinstance(value, Mapping):
            raise ValueError(f"Regime Feature {field_name} entries must be JSON objects")
        out.append(to_jsonable(dict(value)))
    return tuple(out)


def _string_tuple(values: Sequence[str], *, field_name: str, require_non_empty: bool) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValueError(f"Regime Feature {field_name} must be a sequence of strings")
    out = tuple(_text(value, field_name=field_name) for value in values)
    if require_non_empty and not out:
        raise ValueError(f"Regime Feature {field_name} must be non-empty")
    return out


def _text(value: object, *, field_name: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"Regime Feature {field_name} must be non-empty")
    return text


def _optional_text(value: object | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _to_orderable(value: object, *, field_name: str) -> float:
    if isinstance(value, bool) or value is None:
        raise ValueError(f"Regime Feature {field_name} must be a timestamp")
    try:
        return float(value)
    except Exception:
        pass
    text = _text(value, field_name=field_name)
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()
    except Exception as exc:
        raise ValueError(f"Regime Feature {field_name} must be numeric or ISO datetime") from exc


def _validate_order(start: object, end: object, *, context: str) -> None:
    if _to_orderable(start, field_name=f"{context} start") > _to_orderable(end, field_name=f"{context} end"):
        raise ValueError(f"Regime Feature {context} start must be <= end")


__all__ = [
    "REGIME_FEATURES_LAYER",
    "REGIME_FEATURES_SCHEMA_VERSION",
    "REGIME_FEATURE_CLAMP_REF_ARTIFACT_KIND",
    "REGIME_FEATURE_KNOWN_AT_ARTIFACT_KIND",
    "REGIME_FEATURE_LINEAGE_ARTIFACT_KIND",
    "REGIME_FEATURE_RETENTION_HINT_ARTIFACT_KIND",
    "RegimeFeatureClampRef",
    "RegimeFeatureKnownAtSpec",
    "RegimeFeatureLineageSpec",
    "RegimeFeatureRetentionHint",
]
