from __future__ import annotations

import hashlib
import math
import time
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from src.regimes.core.serialization import dumps_json, loads_json, require_json_object, to_jsonable
from src.regimes.regime_features.contracts import (
    ELIGIBILITY_SNAPSHOT,
    REGIME_FEATURES_SCHEMA_VERSION,
    RegimeFeatureKnownAtSpec,
    RegimeFeatureLineageSpec,
)
from src.regimes.regime_features.eligibility import (
    ELIGIBILITY_STATUS_ELIGIBLE_OBSERVED,
    ELIGIBILITY_STATUS_INSUFFICIENT_DATA,
    ELIGIBILITY_STATUS_LIKELY_FLAT_OR_PEGGED,
    ELIGIBILITY_STATUS_LIKELY_INTERVAL_UNRELIABLE,
    ELIGIBILITY_STATUS_LIKELY_LOW_ACTIVITY,
    ELIGIBILITY_STATUS_LIKELY_SPARSE,
    ELIGIBILITY_STATUS_NEEDS_REVIEW,
    REGIME_FEATURE_ELIGIBILITY_STATUSES,
    RegimeFeatureEligibilityRow,
)


TIMESTAMP_COLUMN = "ts"
ASSET_COLUMN = "asset"
UNIVERSE_SCOPE = "universe"
CORE_BASKET_SCOPE = "core_basket"
BROAD_UNIVERSE_SCOPE = "broad_universe"
SNAPSHOT_HASH_ALGORITHM = "sha256"


def _default_core_excluded_statuses() -> tuple[str, ...]:
    return (
        ELIGIBILITY_STATUS_INSUFFICIENT_DATA,
        ELIGIBILITY_STATUS_LIKELY_FLAT_OR_PEGGED,
        ELIGIBILITY_STATUS_LIKELY_SPARSE,
        ELIGIBILITY_STATUS_LIKELY_LOW_ACTIVITY,
        ELIGIBILITY_STATUS_LIKELY_INTERVAL_UNRELIABLE,
    )


def _default_broad_excluded_statuses() -> tuple[str, ...]:
    return (ELIGIBILITY_STATUS_INSUFFICIENT_DATA,)


def _default_status_downweights() -> dict[str, float]:
    return {
        ELIGIBILITY_STATUS_ELIGIBLE_OBSERVED: 1.00,
        ELIGIBILITY_STATUS_NEEDS_REVIEW: 0.75,
        ELIGIBILITY_STATUS_LIKELY_LOW_ACTIVITY: 0.50,
        ELIGIBILITY_STATUS_LIKELY_SPARSE: 0.40,
        ELIGIBILITY_STATUS_LIKELY_INTERVAL_UNRELIABLE: 0.35,
        ELIGIBILITY_STATUS_LIKELY_FLAT_OR_PEGGED: 0.10,
        ELIGIBILITY_STATUS_INSUFFICIENT_DATA: 0.00,
    }


@dataclass(frozen=True)
class SnapshotHash:
    value: str
    algorithm: str = SNAPSHOT_HASH_ALGORITHM
    schema_version: int = REGIME_FEATURES_SCHEMA_VERSION

    def __post_init__(self) -> None:
        value = _text(self.value, field_name="snapshot_hash")
        algorithm = _text(self.algorithm, field_name="snapshot_hash algorithm").lower()
        if algorithm != SNAPSHOT_HASH_ALGORITHM:
            raise ValueError("Regime Feature snapshot hash algorithm must be sha256")
        if len(value) != 64 or any(char not in "0123456789abcdef" for char in value.lower()):
            raise ValueError("Regime Feature snapshot hash must be a 64 character hex sha256 digest")
        object.__setattr__(self, "value", value.lower())
        object.__setattr__(self, "algorithm", algorithm)
        object.__setattr__(self, "schema_version", int(self.schema_version))

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "SnapshotHash":
        digest = hashlib.sha256(dumps_json(to_jsonable(dict(payload))).encode("utf-8")).hexdigest()
        return cls(value=digest)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "artifact_kind": "regime_feature_snapshot_hash",
            "algorithm": self.algorithm,
            "value": self.value,
        }

    def to_json(self, **kwargs: Any) -> str:
        return dumps_json(self.as_dict(), **kwargs)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SnapshotHash":
        obj = require_json_object(payload, context="SnapshotHash")
        return cls(
            value=obj["value"],
            algorithm=obj.get("algorithm", SNAPSHOT_HASH_ALGORITHM),
            schema_version=obj.get("schema_version", REGIME_FEATURES_SCHEMA_VERSION),
        )

    @classmethod
    def from_json(cls, text: str) -> "SnapshotHash":
        return cls.from_dict(require_json_object(loads_json(text), context="SnapshotHash JSON"))


@dataclass(frozen=True)
class ExcludedAssetReason:
    asset: str
    scope: str
    reason: str
    status: str
    score: float
    diagnostics: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = REGIME_FEATURES_SCHEMA_VERSION

    def __post_init__(self) -> None:
        scope = _text(self.scope, field_name="excluded scope")
        if scope not in {UNIVERSE_SCOPE, CORE_BASKET_SCOPE, BROAD_UNIVERSE_SCOPE}:
            raise ValueError("Regime Feature excluded asset scope must be universe, core_basket, or broad_universe")
        status = _eligibility_status(self.status)
        object.__setattr__(self, "asset", _text(self.asset, field_name="excluded asset"))
        object.__setattr__(self, "scope", scope)
        object.__setattr__(self, "reason", _text(self.reason, field_name="excluded reason"))
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "score", _clamp01(self.score))
        object.__setattr__(self, "diagnostics", to_jsonable(dict(self.diagnostics)))
        object.__setattr__(self, "schema_version", int(self.schema_version))

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "artifact_kind": "regime_feature_excluded_asset_reason",
            "asset": self.asset,
            "scope": self.scope,
            "reason": self.reason,
            "status": self.status,
            "score": float(self.score),
            "diagnostics": to_jsonable(dict(self.diagnostics)),
        }

    def to_json(self, **kwargs: Any) -> str:
        return dumps_json(self.as_dict(), **kwargs)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ExcludedAssetReason":
        obj = require_json_object(payload, context="ExcludedAssetReason")
        return cls(
            schema_version=obj.get("schema_version", REGIME_FEATURES_SCHEMA_VERSION),
            asset=obj["asset"],
            scope=obj["scope"],
            reason=obj["reason"],
            status=obj["status"],
            score=obj["score"],
            diagnostics=obj.get("diagnostics", {}),
        )

    @classmethod
    def from_json(cls, text: str) -> "ExcludedAssetReason":
        return cls.from_dict(require_json_object(loads_json(text), context="ExcludedAssetReason JSON"))


@dataclass(frozen=True)
class UniverseSelectionPolicy:
    policy_id: str
    interval: int
    band: str
    refit_key: str = "foundation_skeletal"
    min_core_size: int = 1
    max_core_size: int = 24
    core_score_floor: float = 0.60
    broad_universe_score_floor: float = 0.20
    core_excluded_statuses: Sequence[str] = field(default_factory=_default_core_excluded_statuses)
    broad_excluded_statuses: Sequence[str] = field(default_factory=_default_broad_excluded_statuses)
    status_downweight_factors: Mapping[str, float] = field(default_factory=_default_status_downweights)
    benchmark_anchors: Sequence[str] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = REGIME_FEATURES_SCHEMA_VERSION

    def __post_init__(self) -> None:
        interval = int(self.interval)
        min_core = int(self.min_core_size)
        max_core = int(self.max_core_size)
        if interval <= 0:
            raise ValueError("Regime Feature universe selection interval must be positive")
        if min_core <= 0 or max_core < min_core:
            raise ValueError("Regime Feature universe selection core size bounds are invalid")
        status_downweights: dict[str, float] = {}
        for status, factor in dict(self.status_downweight_factors).items():
            status_downweights[_eligibility_status(status)] = _coverage(factor, field_name=f"status_downweight_factors[{status}]")
        for status in REGIME_FEATURE_ELIGIBILITY_STATUSES:
            status_downweights.setdefault(status, _default_status_downweights()[status])
        object.__setattr__(self, "policy_id", _text(self.policy_id, field_name="policy_id"))
        object.__setattr__(self, "refit_key", _text(self.refit_key, field_name="refit_key"))
        object.__setattr__(self, "interval", interval)
        object.__setattr__(self, "band", _text(self.band, field_name="band").lower())
        object.__setattr__(self, "min_core_size", min_core)
        object.__setattr__(self, "max_core_size", max_core)
        object.__setattr__(self, "core_score_floor", _coverage(self.core_score_floor, field_name="core_score_floor"))
        object.__setattr__(self, "broad_universe_score_floor", _coverage(self.broad_universe_score_floor, field_name="broad_universe_score_floor"))
        object.__setattr__(self, "core_excluded_statuses", _status_tuple(self.core_excluded_statuses, field_name="core_excluded_statuses"))
        object.__setattr__(self, "broad_excluded_statuses", _status_tuple(self.broad_excluded_statuses, field_name="broad_excluded_statuses"))
        object.__setattr__(self, "status_downweight_factors", to_jsonable(dict(sorted(status_downweights.items()))))
        object.__setattr__(self, "benchmark_anchors", _string_tuple(self.benchmark_anchors, field_name="benchmark_anchors", require_non_empty=False))
        object.__setattr__(self, "metadata", to_jsonable(dict(self.metadata)))
        object.__setattr__(self, "schema_version", int(self.schema_version))

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "artifact_kind": "regime_feature_universe_selection_policy",
            "policy_id": self.policy_id,
            "refit_key": self.refit_key,
            "interval": int(self.interval),
            "band": self.band,
            "min_core_size": int(self.min_core_size),
            "max_core_size": int(self.max_core_size),
            "core_score_floor": float(self.core_score_floor),
            "broad_universe_score_floor": float(self.broad_universe_score_floor),
            "core_excluded_statuses": list(self.core_excluded_statuses),
            "broad_excluded_statuses": list(self.broad_excluded_statuses),
            "status_downweight_factors": to_jsonable(dict(self.status_downweight_factors)),
            "benchmark_anchors": list(self.benchmark_anchors),
            "benchmark_anchors_metadata_only": True,
            "hardcoded_required_peers": [],
            "skeletal_policy": True,
            "final_core_basket_policy": False,
            "metadata": to_jsonable(dict(self.metadata)),
        }

    def to_json(self, **kwargs: Any) -> str:
        return dumps_json(self.as_dict(), **kwargs)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "UniverseSelectionPolicy":
        obj = require_json_object(payload, context="UniverseSelectionPolicy")
        return cls(
            schema_version=obj.get("schema_version", REGIME_FEATURES_SCHEMA_VERSION),
            policy_id=obj["policy_id"],
            refit_key=obj.get("refit_key", "foundation_skeletal"),
            interval=obj["interval"],
            band=obj["band"],
            min_core_size=obj.get("min_core_size", 1),
            max_core_size=obj.get("max_core_size", 24),
            core_score_floor=obj.get("core_score_floor", 0.60),
            broad_universe_score_floor=obj.get("broad_universe_score_floor", 0.20),
            core_excluded_statuses=obj.get("core_excluded_statuses", _default_core_excluded_statuses()),
            broad_excluded_statuses=obj.get("broad_excluded_statuses", _default_broad_excluded_statuses()),
            status_downweight_factors=obj.get("status_downweight_factors", _default_status_downweights()),
            benchmark_anchors=obj.get("benchmark_anchors", ()),
            metadata=obj.get("metadata", {}),
        )

    @classmethod
    def from_json(cls, text: str) -> "UniverseSelectionPolicy":
        return cls.from_dict(require_json_object(loads_json(text), context="UniverseSelectionPolicy JSON"))


def build_universe_selection_records(
    eligibility_rows: Sequence[RegimeFeatureEligibilityRow | Mapping[str, Any]],
    *,
    policy: UniverseSelectionPolicy,
) -> tuple[dict[str, Any], ...]:
    rows = tuple(_coerce_eligibility_row(row) for row in eligibility_rows)
    if not rows:
        raise ValueError("Regime Feature universe selection requires eligibility rows")
    seen_assets: set[str] = set()
    records: list[dict[str, Any]] = []
    for row in sorted(rows, key=lambda item: item.asset):
        if int(row.interval) != int(policy.interval) or row.band != policy.band:
            raise ValueError("Regime Feature universe selection policy interval/band must match eligibility rows")
        if row.asset in seen_assets:
            raise ValueError(f"Regime Feature universe selection duplicate eligibility asset {row.asset!r}")
        seen_assets.add(row.asset)
        records.append(_selection_record(row, policy=policy))
    return tuple(records)


def _selection_record(row: RegimeFeatureEligibilityRow, *, policy: UniverseSelectionPolicy) -> dict[str, Any]:
    coverage = _clamp01((row.coverage_diagnostics or {}).get("coverage_ratio", 0.0))
    finite_return = _clamp01((row.finite_return_coverage or {}).get("finite_return_ratio", 0.0))
    activity = _clamp01((row.activity_diagnostics or {}).get("activity_score", (row.activity_diagnostics or {}).get("activity_coverage_score", 0.0)))
    flat_risk = _clamp01((row.risk_diagnostics or {}).get("flat_or_pegged_risk", (row.movement_diagnostics or {}).get("flat_or_pegged_risk", 0.0)))
    gap_risk = _clamp01((row.interval_reliability_hints or {}).get("high_gap_ratio", 0.0))
    movement = _clamp01(1.0 - flat_risk)
    reliability = 0.0 if bool((row.interval_reliability_hints or {}).get("interval_unreliable_hint", False)) else _clamp01(1.0 - gap_risk)
    base_score = _clamp01((0.30 * coverage) + (0.25 * finite_return) + (0.20 * movement) + (0.15 * activity) + (0.10 * reliability))
    status_factor = _clamp01(dict(policy.status_downweight_factors).get(row.status, 1.0))
    effective_score = _clamp01(base_score * status_factor)
    core_blocked_reason = _scope_block_reason(
        status=row.status,
        score=effective_score,
        floor=float(policy.core_score_floor),
        excluded_statuses=policy.core_excluded_statuses,
    )
    broad_blocked_reason = _scope_block_reason(
        status=row.status,
        score=effective_score,
        floor=float(policy.broad_universe_score_floor),
        excluded_statuses=policy.broad_excluded_statuses,
    )
    return {
        "asset": row.asset,
        "status": row.status,
        "status_reasons": list(row.status_reasons),
        "base_score": base_score,
        "selection_score": effective_score,
        "core_candidate": core_blocked_reason is None,
        "broad_candidate": broad_blocked_reason is None,
        "core_blocked_reason": core_blocked_reason,
        "broad_blocked_reason": broad_blocked_reason,
        "coverage_ratio": coverage,
        "finite_return_ratio": finite_return,
        "activity_score": activity,
        "movement_score": movement,
        "reliability_score": reliability,
        "benchmark_anchor_metadata": row.asset in set(policy.benchmark_anchors),
        "diagnostics": {
            "coverage_diagnostics": to_jsonable(dict(row.coverage_diagnostics)),
            "finite_return_coverage": to_jsonable(dict(row.finite_return_coverage)),
            "movement_diagnostics": to_jsonable(dict(row.movement_diagnostics)),
            "activity_diagnostics": to_jsonable(dict(row.activity_diagnostics)),
            "risk_diagnostics": to_jsonable(dict(row.risk_diagnostics)),
            "interval_reliability_hints": to_jsonable(dict(row.interval_reliability_hints)),
            "scalar_feature_availability": to_jsonable(dict(row.scalar_feature_availability)),
            "ohlcvt_availability": to_jsonable(dict(row.ohlcvt_availability)),
        },
    }


def _scope_block_reason(*, status: str, score: float, floor: float, excluded_statuses: Sequence[str]) -> str | None:
    if status in set(excluded_statuses):
        if status == ELIGIBILITY_STATUS_INSUFFICIENT_DATA:
            return "insufficient_data_status"
        if status == ELIGIBILITY_STATUS_LIKELY_FLAT_OR_PEGGED:
            return "flat_or_pegged_status"
        if status == ELIGIBILITY_STATUS_LIKELY_SPARSE:
            return "sparse_status"
        if status == ELIGIBILITY_STATUS_LIKELY_LOW_ACTIVITY:
            return "low_activity_status"
        if status == ELIGIBILITY_STATUS_LIKELY_INTERVAL_UNRELIABLE:
            return "interval_unreliable_status"
        return f"{status}_status"
    if float(score) < float(floor):
        return "below_score_floor"
    return None


def _coerce_eligibility_row(row: RegimeFeatureEligibilityRow | Mapping[str, Any]) -> RegimeFeatureEligibilityRow:
    if isinstance(row, RegimeFeatureEligibilityRow):
        return row
    if isinstance(row, Mapping):
        return RegimeFeatureEligibilityRow.from_dict(row)
    raise ValueError("Regime Feature universe selection rows must be eligibility rows or row payload mappings")


@dataclass(frozen=True)
class GlobalEligibilitySnapshotConfig:
    policy_id: str
    interval: int
    band: str
    refit_key: str = "foundation_descriptive"
    min_history_rows: int = 1
    min_finite_return_ratio: float = 0.80
    max_zero_return_share: float = 0.95
    min_activity_coverage: float = 0.50
    min_median_abs_log_return: float = 1e-8
    min_realized_volatility: float = 1e-7
    max_tail_staleness_seconds: int | None = None
    max_core_basket_size: int = 24
    min_core_basket_size: int = 1
    broad_candidate_score_floor: float = 0.20
    generated_by: str = "regime_features.global_eligibility_snapshot"
    schema_version: int = REGIME_FEATURES_SCHEMA_VERSION
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        interval = int(self.interval)
        if interval <= 0:
            raise ValueError("Regime Feature eligibility interval must be positive")
        if int(self.min_history_rows) <= 0:
            raise ValueError("Regime Feature eligibility min_history_rows must be positive")
        if int(self.min_core_basket_size) <= 0 or int(self.max_core_basket_size) < int(self.min_core_basket_size):
            raise ValueError("Regime Feature eligibility core basket size bounds are invalid")
        object.__setattr__(self, "interval", interval)
        object.__setattr__(self, "band", _text(self.band, field_name="band").lower())
        object.__setattr__(self, "policy_id", _text(self.policy_id, field_name="policy_id"))
        object.__setattr__(self, "refit_key", _text(self.refit_key, field_name="refit_key"))
        object.__setattr__(self, "generated_by", _text(self.generated_by, field_name="generated_by"))
        object.__setattr__(self, "min_history_rows", int(self.min_history_rows))
        object.__setattr__(self, "max_core_basket_size", int(self.max_core_basket_size))
        object.__setattr__(self, "min_core_basket_size", int(self.min_core_basket_size))
        object.__setattr__(self, "min_finite_return_ratio", _coverage(self.min_finite_return_ratio, field_name="min_finite_return_ratio"))
        object.__setattr__(self, "max_zero_return_share", _coverage(self.max_zero_return_share, field_name="max_zero_return_share"))
        object.__setattr__(self, "min_activity_coverage", _coverage(self.min_activity_coverage, field_name="min_activity_coverage"))
        object.__setattr__(self, "broad_candidate_score_floor", _coverage(self.broad_candidate_score_floor, field_name="broad_candidate_score_floor"))
        object.__setattr__(self, "min_median_abs_log_return", float(self.min_median_abs_log_return))
        object.__setattr__(self, "min_realized_volatility", float(self.min_realized_volatility))
        object.__setattr__(self, "max_tail_staleness_seconds", None if self.max_tail_staleness_seconds is None else int(self.max_tail_staleness_seconds))
        object.__setattr__(self, "schema_version", int(self.schema_version))
        object.__setattr__(self, "metadata", to_jsonable(dict(self.metadata)))

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "artifact_kind": "regime_feature_global_eligibility_policy",
            "policy_id": self.policy_id,
            "refit_key": self.refit_key,
            "interval": int(self.interval),
            "band": self.band,
            "min_history_rows": int(self.min_history_rows),
            "min_finite_return_ratio": float(self.min_finite_return_ratio),
            "max_zero_return_share": float(self.max_zero_return_share),
            "min_activity_coverage": float(self.min_activity_coverage),
            "min_median_abs_log_return": float(self.min_median_abs_log_return),
            "min_realized_volatility": float(self.min_realized_volatility),
            "max_tail_staleness_seconds": self.max_tail_staleness_seconds,
            "max_core_basket_size": int(self.max_core_basket_size),
            "min_core_basket_size": int(self.min_core_basket_size),
            "broad_candidate_score_floor": float(self.broad_candidate_score_floor),
            "generated_by": self.generated_by,
            "descriptive_not_final_filter": True,
            "metadata": to_jsonable(dict(self.metadata)),
        }


@dataclass
class GlobalEligibilitySnapshot:
    config: GlobalEligibilitySnapshotConfig
    rows: pd.DataFrame = field(repr=False)
    core_basket_assets: Sequence[str]
    broad_universe_assets: Sequence[str]
    excluded_assets_with_reasons: Mapping[str, Sequence[str]]
    known_at: RegimeFeatureKnownAtSpec
    lineage: RegimeFeatureLineageSpec
    diagnostics: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = REGIME_FEATURES_SCHEMA_VERSION

    def __post_init__(self) -> None:
        self.core_basket_assets = tuple(str(asset) for asset in self.core_basket_assets)
        self.broad_universe_assets = tuple(str(asset) for asset in self.broad_universe_assets)
        self.excluded_assets_with_reasons = {
            str(asset): tuple(str(reason) for reason in reasons)
            for asset, reasons in dict(self.excluded_assets_with_reasons).items()
        }

    @property
    def snapshot_id(self) -> str:
        return f"{self.config.policy_id}:{self.config.refit_key}:{self.config.interval}:{self.config.band}:{self.snapshot_hash[:12]}"

    @property
    def snapshot_hash(self) -> str:
        payload = {
            "policy": self.config.as_dict(),
            "core": list(self.core_basket_assets),
            "broad": list(self.broad_universe_assets),
            "excluded": {asset: list(reasons) for asset, reasons in sorted(self.excluded_assets_with_reasons.items())},
            "source_tail_ts": self.known_at.source_tail_ts,
        }
        return hashlib.sha256(dumps_json(payload).encode("utf-8")).hexdigest()

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "artifact_kind": "regime_feature_global_eligibility_snapshot",
            "selection_policy_id": self.config.policy_id,
            "refit_key": self.config.refit_key,
            "interval": int(self.config.interval),
            "band": self.config.band,
            "snapshot_id": self.snapshot_id,
            "snapshot_hash": self.snapshot_hash,
            "known_at": self.known_at.as_dict(),
            "source_tail_ts": self.known_at.source_tail_ts,
            "core_basket_assets": list(self.core_basket_assets),
            "broad_universe_assets": list(self.broad_universe_assets),
            "excluded_assets_with_reasons": {asset: list(reasons) for asset, reasons in sorted(self.excluded_assets_with_reasons.items())},
            "eligibility_diagnostics": to_jsonable(dict(self.diagnostics)),
            "lineage": self.lineage.as_dict(),
            "descriptive_not_final_filter": True,
            "dynamic_peer_groups": [],
            "peer_group_discovery_enabled": False,
            "production_enabled": False,
        }


def build_global_eligibility_snapshot(
    asset_frames: Mapping[str, pd.DataFrame],
    *,
    config: GlobalEligibilitySnapshotConfig,
    scalar_metadata_by_asset: Mapping[str, Mapping[str, Any]] | None = None,
    source_partition_lineage: Sequence[Mapping[str, Any]] = (),
    generated_at_ts: int | None = None,
    source_data_kinds: Sequence[str] = ("ohlcvt",),
) -> GlobalEligibilitySnapshot:
    if not isinstance(asset_frames, Mapping) or not asset_frames:
        raise ValueError("Regime Feature eligibility snapshot requires explicit asset_frames")
    rows = [_profile_asset(asset, frame, config=config, scalar_metadata=(scalar_metadata_by_asset or {}).get(asset)) for asset, frame in sorted(asset_frames.items())]
    profile = pd.DataFrame(rows)
    source_tail_ts = _safe_int(profile["last_ts"].max()) if not profile.empty and "last_ts" in profile.columns else 0
    generated = int(generated_at_ts if generated_at_ts is not None else max(int(time.time()), source_tail_ts))
    profile = _score_profile(profile, config=config, generated_at_ts=generated)

    core_pool = profile[profile["core_candidate_flag"]].copy()
    core_pool = core_pool.sort_values(
        ["market_core_candidate_score", "activity_score", "coverage_ratio", "asset"],
        ascending=[False, False, False, True],
    )
    if int(core_pool.shape[0]) < int(config.min_core_basket_size):
        core_assets: tuple[str, ...] = ()
    else:
        core_assets = tuple(core_pool["asset"].head(int(config.max_core_basket_size)).astype(str).tolist())

    broad_pool = profile[profile["broad_universe_candidate_score"] >= float(config.broad_candidate_score_floor)].copy()
    broad_pool = broad_pool.sort_values(["broad_universe_candidate_score", "coverage_ratio", "asset"], ascending=[False, False, True])
    broad_assets = tuple(broad_pool["asset"].astype(str).tolist())

    excluded = {
        str(row["asset"]): tuple(row["exclusion_reasons"])
        for _, row in profile.iterrows()
        if row["exclusion_reasons"]
    }
    profile["exclusion_reasons"] = profile["exclusion_reasons"].map(lambda reasons: list(reasons))
    known_at = RegimeFeatureKnownAtSpec(
        ts=source_tail_ts,
        known_at_ts=generated,
        source_tail_ts=source_tail_ts,
        feature_available_at_ts=generated,
        no_lookahead_verified=True,
    )
    lineage_entries = tuple(source_partition_lineage) or (
        {
            "source_kind": "in_memory_asset_frames",
            "asset_count": int(len(asset_frames)),
            "interval": int(config.interval),
            "note": "caller supplied bounded frames",
        },
    )
    lineage = RegimeFeatureLineageSpec(
        artifact_family=ELIGIBILITY_SNAPSHOT,
        feature_set_id=config.policy_id,
        interval=int(config.interval),
        band=config.band,
        source_data_kinds=source_data_kinds,
        source_partition_lineage=lineage_entries,
        source_tail_ts=source_tail_ts,
        feature_window_start=_safe_int(profile["first_ts"].min()) if not profile.empty else source_tail_ts,
        feature_window_end=source_tail_ts,
        generated_at=generated,
        run_id=f"{config.policy_id}_{config.refit_key}",
        calculation_policy=config.as_dict(),
    )
    diagnostics = {
        "asset_count": int(profile.shape[0]),
        "core_candidate_count": int(profile["core_candidate_flag"].sum()) if not profile.empty else 0,
        "selected_core_count": int(len(core_assets)),
        "broad_candidate_count": int(len(broad_assets)),
        "excluded_asset_count": int(len(excluded)),
        "descriptive_not_final_filter": True,
        "dynamic_peer_groups_selected": False,
    }
    return GlobalEligibilitySnapshot(
        config=config,
        rows=profile,
        core_basket_assets=core_assets,
        broad_universe_assets=broad_assets,
        excluded_assets_with_reasons=excluded,
        known_at=known_at,
        lineage=lineage,
        diagnostics=diagnostics,
    )


def _profile_asset(asset: str, frame: pd.DataFrame, *, config: GlobalEligibilitySnapshotConfig, scalar_metadata: Mapping[str, Any] | None) -> dict[str, Any]:
    cleaned = _normalize_frame(frame, asset=asset)
    interval_seconds = int(config.interval) * 60
    if cleaned.empty:
        return {
            "asset": str(asset),
            "interval": int(config.interval),
            "band": config.band,
            "first_ts": None,
            "last_ts": None,
            "row_count": 0,
            "expected_row_count": 0,
            "coverage_ratio": 0.0,
            "finite_close_ratio": 0.0,
            "finite_return_ratio": 0.0,
            "zero_log_return_share": 1.0,
            "near_zero_log_return_share": 1.0,
            "median_abs_log_return": 0.0,
            "mean_abs_log_return": 0.0,
            "realized_volatility": 0.0,
            "finite_volume_ratio": 0.0,
            "finite_trades_ratio": 0.0,
            "zero_volume_share": 1.0,
            "zero_trades_share": 1.0,
            "median_volume": 0.0,
            "median_trades": 0.0,
            "raw_activity_value": 0.0,
            "activity_coverage_score": 0.0,
            "scalar_feature_partition_available": bool(scalar_metadata and scalar_metadata.get("available")),
            "scalar_feature_column_count": int((scalar_metadata or {}).get("column_count") or 0),
        }
    ts = pd.to_numeric(cleaned[TIMESTAMP_COLUMN], errors="coerce").dropna().astype("int64")
    first_ts = int(ts.min())
    last_ts = int(ts.max())
    expected = max(1, int((last_ts - first_ts) // interval_seconds) + 1) if last_ts >= first_ts else int(cleaned.shape[0])
    close = pd.to_numeric(cleaned["close"], errors="coerce")
    log_return = np.log(close).diff()
    valid_step = ts.diff().fillna(interval_seconds).eq(interval_seconds)
    log_return = log_return.where(valid_step)
    finite_return = log_return[np.isfinite(log_return)]
    volume = pd.to_numeric(cleaned.get("volume"), errors="coerce")
    trades = pd.to_numeric(cleaned.get("trades"), errors="coerce")
    return {
        "asset": str(asset),
        "interval": int(config.interval),
        "band": config.band,
        "first_ts": first_ts,
        "last_ts": last_ts,
        "row_count": int(cleaned.shape[0]),
        "expected_row_count": int(expected),
        "coverage_ratio": _clamp01(float(cleaned.shape[0]) / float(expected)),
        "finite_close_ratio": _finite_ratio(close),
        "finite_return_ratio": _finite_ratio(log_return),
        "zero_log_return_share": _share(np.isclose(finite_return, 0.0, atol=0.0)) if not finite_return.empty else 1.0,
        "near_zero_log_return_share": _share(np.isclose(finite_return, 0.0, atol=1e-8)) if not finite_return.empty else 1.0,
        "median_abs_log_return": float(np.nanmedian(np.abs(finite_return))) if not finite_return.empty else 0.0,
        "mean_abs_log_return": float(np.nanmean(np.abs(finite_return))) if not finite_return.empty else 0.0,
        "realized_volatility": float(np.nanstd(finite_return, ddof=0)) if not finite_return.empty else 0.0,
        "finite_volume_ratio": _finite_ratio(volume),
        "finite_trades_ratio": _finite_ratio(trades),
        "zero_volume_share": _share(volume.fillna(0.0).eq(0.0)) if not volume.empty else 1.0,
        "zero_trades_share": _share(trades.fillna(0.0).eq(0.0)) if not trades.empty else 1.0,
        "median_volume": float(np.nanmedian(volume)) if np.isfinite(volume).any() else 0.0,
        "median_trades": float(np.nanmedian(trades)) if np.isfinite(trades).any() else 0.0,
        "raw_activity_value": float(max(float(np.nanmedian(volume)) if np.isfinite(volume).any() else 0.0, 0.0) + max(float(np.nanmedian(trades)) if np.isfinite(trades).any() else 0.0, 0.0)),
        "activity_coverage_score": _clamp01(max(_finite_ratio(volume) * (1.0 - _share(volume.fillna(0.0).eq(0.0))), _finite_ratio(trades) * (1.0 - _share(trades.fillna(0.0).eq(0.0))))),
        "scalar_feature_partition_available": bool(scalar_metadata and scalar_metadata.get("available")),
        "scalar_feature_column_count": int((scalar_metadata or {}).get("column_count") or 0),
    }


def _score_profile(profile: pd.DataFrame, *, config: GlobalEligibilitySnapshotConfig, generated_at_ts: int) -> pd.DataFrame:
    out = profile.copy()
    max_activity = float(pd.to_numeric(out["raw_activity_value"], errors="coerce").max() or 0.0)
    out["activity_score"] = out["raw_activity_value"].map(lambda value: _clamp01(math.log1p(_num(value)) / math.log1p(max_activity)) if max_activity > 0 else 0.0)
    out["movement_score"] = out.apply(
        lambda row: 0.0
        if _num(row["zero_log_return_share"], default=1.0) > float(config.max_zero_return_share)
        else _clamp01(
            max(
                _num(row["median_abs_log_return"]) / max(float(config.min_median_abs_log_return), 1e-12),
                _num(row["realized_volatility"]) / max(float(config.min_realized_volatility), 1e-12),
            )
        ),
        axis=1,
    )
    out["latest_tail_age_seconds"] = out["last_ts"].map(lambda ts: None if pd.isna(ts) else max(0, int(generated_at_ts) - int(ts)))
    out["insufficient_history_flag"] = pd.to_numeric(out["row_count"], errors="coerce").fillna(0) < int(config.min_history_rows)
    out["insufficient_return_coverage_flag"] = pd.to_numeric(out["finite_return_ratio"], errors="coerce").fillna(0.0) < float(config.min_finite_return_ratio)
    out["insufficient_activity_flag"] = pd.to_numeric(out["activity_coverage_score"], errors="coerce").fillna(0.0) < float(config.min_activity_coverage)
    out["insufficient_movement_flag"] = pd.to_numeric(out["movement_score"], errors="coerce").fillna(0.0) <= 0.0
    out["stale_tail_flag"] = False
    if config.max_tail_staleness_seconds is not None:
        out["stale_tail_flag"] = pd.to_numeric(out["latest_tail_age_seconds"], errors="coerce").fillna(float("inf")) > int(config.max_tail_staleness_seconds)
    out["asset_state_suitability_prior"] = (
        0.30 * pd.to_numeric(out["coverage_ratio"], errors="coerce").fillna(0.0)
        + 0.25 * pd.to_numeric(out["finite_return_ratio"], errors="coerce").fillna(0.0)
        + 0.20 * pd.to_numeric(out["movement_score"], errors="coerce").fillna(0.0)
        + 0.15 * pd.to_numeric(out["activity_score"], errors="coerce").fillna(0.0)
        + 0.10 * pd.to_numeric(out["activity_coverage_score"], errors="coerce").fillna(0.0)
    ).map(_clamp01)
    out["market_core_candidate_score"] = (
        0.35 * pd.to_numeric(out["coverage_ratio"], errors="coerce").fillna(0.0)
        + 0.25 * pd.to_numeric(out["activity_score"], errors="coerce").fillna(0.0)
        + 0.20 * pd.to_numeric(out["finite_return_ratio"], errors="coerce").fillna(0.0)
        + 0.20 * pd.to_numeric(out["movement_score"], errors="coerce").fillna(0.0)
    ).map(_clamp01)
    out["broad_universe_candidate_score"] = (
        0.45 * pd.to_numeric(out["coverage_ratio"], errors="coerce").fillna(0.0)
        + 0.25 * pd.to_numeric(out["finite_return_ratio"], errors="coerce").fillna(0.0)
        + 0.15 * pd.to_numeric(out["movement_score"], errors="coerce").fillna(0.0)
        + 0.15 * pd.to_numeric(out["activity_coverage_score"], errors="coerce").fillna(0.0)
    ).map(_clamp01)
    out["cross_asset_relationship_candidate_score"] = (
        0.45 * pd.to_numeric(out["coverage_ratio"], errors="coerce").fillna(0.0)
        + 0.25 * pd.to_numeric(out["finite_return_ratio"], errors="coerce").fillna(0.0)
        + 0.20 * pd.to_numeric(out["movement_score"], errors="coerce").fillna(0.0)
        + 0.10 * pd.to_numeric(out["activity_score"], errors="coerce").fillna(0.0)
    ).map(_clamp01)
    out["flat_fallback_candidate_flag"] = out["insufficient_movement_flag"] & ~out["insufficient_history_flag"]
    out["core_candidate_flag"] = ~(
        out["insufficient_history_flag"]
        | out["insufficient_return_coverage_flag"]
        | out["insufficient_activity_flag"]
        | out["insufficient_movement_flag"]
        | out["stale_tail_flag"]
    )
    out["exclusion_reasons"] = out.apply(_exclusion_reasons, axis=1)
    return out.sort_values("asset").reset_index(drop=True)


def _exclusion_reasons(row: pd.Series) -> tuple[str, ...]:
    reasons: list[str] = []
    for flag, reason in (
        ("insufficient_history_flag", "insufficient_history"),
        ("insufficient_return_coverage_flag", "insufficient_return_coverage"),
        ("insufficient_activity_flag", "insufficient_activity"),
        ("insufficient_movement_flag", "insufficient_movement"),
        ("stale_tail_flag", "stale_tail"),
    ):
        if bool(row.get(flag, False)):
            reasons.append(reason)
    return tuple(reasons)


def _normalize_frame(frame: pd.DataFrame, *, asset: str) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame(columns=[ASSET_COLUMN, TIMESTAMP_COLUMN, "close", "volume", "trades"])
    out = frame.copy()
    if ASSET_COLUMN not in out.columns:
        out[ASSET_COLUMN] = str(asset)
    for column in (TIMESTAMP_COLUMN, "close", "volume", "trades"):
        if column not in out.columns:
            out[column] = np.nan
    out[TIMESTAMP_COLUMN] = pd.to_numeric(out[TIMESTAMP_COLUMN], errors="coerce")
    out = out.dropna(subset=[TIMESTAMP_COLUMN]).copy()
    out[TIMESTAMP_COLUMN] = out[TIMESTAMP_COLUMN].astype("int64")
    out[ASSET_COLUMN] = out[ASSET_COLUMN].fillna(str(asset)).astype(str)
    out = out[out[ASSET_COLUMN].astype(str) == str(asset)].copy()
    for column in ("close", "volume", "trades"):
        out[column] = pd.to_numeric(out[column], errors="coerce")
    return out[[ASSET_COLUMN, TIMESTAMP_COLUMN, "close", "volume", "trades"]].drop_duplicates([ASSET_COLUMN, TIMESTAMP_COLUMN], keep="last").sort_values(TIMESTAMP_COLUMN).reset_index(drop=True)


def _finite_ratio(series: pd.Series) -> float:
    if series is None or len(series) == 0:
        return 0.0
    return float(np.isfinite(pd.to_numeric(series, errors="coerce")).mean())


def _share(values: Any) -> float:
    if values is None or len(values) == 0:
        return 0.0
    return float(np.asarray(values, dtype=bool).mean())


def _safe_int(value: Any) -> int:
    if value is None or pd.isna(value):
        return 0
    return int(value)


def _coverage(value: Any, *, field_name: str) -> float:
    val = float(value)
    if val < 0.0 or val > 1.0:
        raise ValueError(f"Regime Feature eligibility {field_name} must be within [0, 1]")
    return val


def _clamp01(value: Any) -> float:
    try:
        val = float(value)
    except Exception:
        return 0.0
    if not math.isfinite(val):
        return 0.0
    return max(0.0, min(1.0, val))


def _num(value: Any, *, default: float = 0.0) -> float:
    try:
        val = float(value)
    except Exception:
        return float(default)
    if not math.isfinite(val):
        return float(default)
    return val


def _text(value: object, *, field_name: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"Regime Feature eligibility {field_name} must be non-empty")
    return text


def _string_tuple(values: Sequence[str], *, field_name: str, require_non_empty: bool) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValueError(f"Regime Feature universe selection {field_name} must be a sequence of strings")
    out = tuple(_text(value, field_name=field_name) for value in values)
    if require_non_empty and not out:
        raise ValueError(f"Regime Feature universe selection {field_name} must be non-empty")
    return out


def _status_tuple(values: Sequence[str], *, field_name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValueError(f"Regime Feature universe selection {field_name} must be a sequence of statuses")
    return tuple(_eligibility_status(value) for value in values)


def _eligibility_status(value: object) -> str:
    text = _text(value, field_name="eligibility status")
    if text not in REGIME_FEATURE_ELIGIBILITY_STATUSES:
        valid = ", ".join(REGIME_FEATURE_ELIGIBILITY_STATUSES)
        raise ValueError(f"Unsupported Regime Feature eligibility status {text!r}; expected one of: {valid}")
    return text


__all__ = [
    "BROAD_UNIVERSE_SCOPE",
    "CORE_BASKET_SCOPE",
    "SNAPSHOT_HASH_ALGORITHM",
    "UNIVERSE_SCOPE",
    "ExcludedAssetReason",
    "GlobalEligibilitySnapshot",
    "GlobalEligibilitySnapshotConfig",
    "SnapshotHash",
    "UniverseSelectionPolicy",
    "build_universe_selection_records",
    "build_global_eligibility_snapshot",
]
