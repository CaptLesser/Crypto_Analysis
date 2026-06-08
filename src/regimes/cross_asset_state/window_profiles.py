from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from src.regimes.core.window_profiles import RegimeWindowProfile


CROSS_ASSET_STATE_WINDOW_POLICY_SCHEMA_VERSION = 1
CROSS_ASSET_STATE_WINDOW_POLICY_ID = "cross_asset_state_window_policy_v1"
CROSS_ASSET_STATE_WINDOW_INSUFFICIENT_HISTORY = "insufficient_window_history"
CROSS_ASSET_STATE_WINDOW_FAMILIES: tuple[str, ...] = (
    "anchor_core_exposure",
    "peer_strength_stability",
    "relationship_concentration_entropy",
    "residual_peer_signal",
)
MESO_WINDOW_CANDIDATES: tuple[tuple[str, int | None], ...] = (
    ("current_default", None),
    ("meso_short", 90),
    ("meso_medium", 180),
    ("meso_long", 360),
)
MACRO_WINDOW_CANDIDATES: tuple[tuple[str, int | None], ...] = (
    ("current_default", None),
    ("macro_short", 180),
    ("macro_medium", 365),
    ("macro_long", 720),
)


@dataclass(frozen=True)
class CrossAssetStateWindowPolicy:
    window_policy_id: str
    relationship_feature_family: str
    band: str
    lookback_days: int | None
    min_rows: int = 8
    row_cap: int | None = None
    source_tail_anchor: bool = True
    window_candidate_name: str = ""
    rationale: str = ""
    schema_version: int = CROSS_ASSET_STATE_WINDOW_POLICY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        policy_id = str(self.window_policy_id).strip()
        family = str(self.relationship_feature_family).strip()
        band = str(self.band).strip().lower()
        lookback = None if self.lookback_days is None else int(self.lookback_days)
        min_rows = int(self.min_rows)
        row_cap = None if self.row_cap is None else int(self.row_cap)
        if not policy_id:
            raise ValueError("Cross-Asset-State window_policy_id must be non-empty")
        if not family:
            raise ValueError("Cross-Asset-State window policy requires relationship_feature_family")
        if band not in {"meso", "macro"}:
            raise ValueError(f"Unsupported Cross-Asset-State window band {band!r}")
        if lookback is not None and lookback <= 0:
            raise ValueError("Cross-Asset-State lookback_days must be positive when provided")
        if min_rows <= 0:
            raise ValueError("Cross-Asset-State window min_rows must be positive")
        if row_cap is not None and row_cap <= 0:
            raise ValueError("Cross-Asset-State window row_cap must be positive when provided")
        object.__setattr__(self, "window_policy_id", policy_id)
        object.__setattr__(self, "relationship_feature_family", family)
        object.__setattr__(self, "band", band)
        object.__setattr__(self, "lookback_days", lookback)
        object.__setattr__(self, "min_rows", min_rows)
        object.__setattr__(self, "row_cap", row_cap)
        object.__setattr__(self, "source_tail_anchor", bool(self.source_tail_anchor))
        object.__setattr__(self, "window_candidate_name", str(self.window_candidate_name or _candidate_name_from_policy_id(policy_id)).strip())

    @property
    def window_profile_id(self) -> str:
        return self.window_policy_id

    def as_regime_window_profile(self) -> RegimeWindowProfile:
        return RegimeWindowProfile(
            window_profile_id=self.window_policy_id,
            band=self.band,
            lookback_days=self.lookback_days,
            source_tail_anchor=self.source_tail_anchor,
            row_cap=self.row_cap,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "policy_set_id": CROSS_ASSET_STATE_WINDOW_POLICY_ID,
            "window_policy_id": self.window_policy_id,
            "window_profile_id": self.window_profile_id,
            "window_candidate_name": self.window_candidate_name,
            "relationship_feature_family": self.relationship_feature_family,
            "band": self.band,
            "lookback_days": None if self.lookback_days is None else int(self.lookback_days),
            "min_rows": int(self.min_rows),
            "row_cap": self.row_cap,
            "source_tail_anchor": bool(self.source_tail_anchor),
            "rationale": self.rationale,
            "production_approved": False,
            "production_writer_enabled": False,
        }


@dataclass(frozen=True)
class CrossAssetStateWindowCoverage:
    status: str
    passed: bool
    window_policy: CrossAssetStateWindowPolicy
    observed_rows: int
    min_rows: int
    start_ts: int | None
    end_ts: int | None
    source_tail_ts: int | None
    reason_code: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "passed": bool(self.passed),
            "reason_code": self.reason_code,
            "window_policy_id": self.window_policy.window_policy_id,
            "window_profile_id": self.window_policy.window_profile_id,
            "window_candidate_name": self.window_policy.window_candidate_name,
            "relationship_feature_family": self.window_policy.relationship_feature_family,
            "band": self.window_policy.band,
            "lookback_days": None if self.window_policy.lookback_days is None else int(self.window_policy.lookback_days),
            "observed_rows": int(self.observed_rows),
            "min_rows": int(self.min_rows),
            "start_ts": self.start_ts,
            "end_ts": self.end_ts,
            "source_tail_ts": self.source_tail_ts,
            "production_approved": False,
            "production_writer_enabled": False,
        }


def default_cross_asset_state_window_policies() -> tuple[CrossAssetStateWindowPolicy, ...]:
    policies: list[CrossAssetStateWindowPolicy] = []
    for family in CROSS_ASSET_STATE_WINDOW_FAMILIES:
        policies.extend(cross_asset_state_window_ladder_policies(relationship_feature_family=family, band="meso"))
        policies.extend(cross_asset_state_window_ladder_policies(relationship_feature_family=family, band="macro"))
    return tuple(policies)


def cross_asset_state_window_ladder_policies(
    *,
    relationship_feature_family: str,
    band: str,
) -> tuple[CrossAssetStateWindowPolicy, ...]:
    family = str(relationship_feature_family).strip()
    band_key = str(band).strip().lower()
    if band_key == "meso":
        ladder: tuple[tuple[str, int | None, str], ...] = tuple(
            (candidate, lookback, _window_candidate_rationale(candidate))
            for candidate, lookback in MESO_WINDOW_CANDIDATES
        )
    elif band_key == "macro":
        ladder = tuple(
            (candidate, lookback, _window_candidate_rationale(candidate))
            for candidate, lookback in MACRO_WINDOW_CANDIDATES
        )
    else:
        raise ValueError(f"Unsupported Cross-Asset-State window band {band!r}")
    policies: list[CrossAssetStateWindowPolicy] = []
    for candidate, lookback_days, rationale in ladder:
        policies.append(
            CrossAssetStateWindowPolicy(
                window_policy_id=f"cross_asset_{band_key}_{family}_{candidate}_v1",
                relationship_feature_family=family,
                band=band_key,
                lookback_days=lookback_days,
                source_tail_anchor=candidate != "current_default",
                window_candidate_name=candidate,
                rationale=rationale,
            )
        )
    return tuple(policies)


def cross_asset_state_window_policy_manifest() -> dict[str, Any]:
    policies = default_cross_asset_state_window_policies()
    return {
        "artifact_kind": "cross_asset_state_window_policy_manifest",
        "schema_version": CROSS_ASSET_STATE_WINDOW_POLICY_SCHEMA_VERSION,
        "policy_set_id": CROSS_ASSET_STATE_WINDOW_POLICY_ID,
        "current_default_behavior": (
            "Previous prototypes used all loaded rows after relationship availability filtering. "
            "The mini-test branch evaluates that current/default behavior plus short/medium/long source-tail anchored windows."
        ),
        "policies": [policy.as_dict() for policy in policies],
        "candidate_window_policy_questions": {
            "peer_strength_stability_fixed_window_risk": (
                "High. Weak or overly similar peer strength/stability states can be caused by too-short windows "
                "that overfit transient peers or too-long windows that flatten regime changes."
            ),
            "relationship_concentration_entropy_fixed_window_risk": (
                "High. Concentration and entropy are especially likely to look flat when a single fixed window "
                "hides spread/rank changes in dependency structure."
            ),
            "likely_window_sensitive_families": [
                "peer_strength_stability",
                "relationship_concentration_entropy",
                "anchor_core_exposure",
            ],
            "positive_control_family": "residual_peer_signal",
        },
        "production_approved": False,
        "production_writer_enabled": False,
    }


def validate_cross_asset_state_window_policy_manifest(manifest: Mapping[str, Any] | None = None) -> dict[str, Any]:
    payload = dict(manifest or cross_asset_state_window_policy_manifest())
    policies = [dict(policy) for policy in payload.get("policies") or () if isinstance(policy, Mapping)]
    reason_codes: list[str] = []
    if payload.get("artifact_kind") != "cross_asset_state_window_policy_manifest":
        reason_codes.append("artifact_kind_invalid")
    if payload.get("policy_set_id") != CROSS_ASSET_STATE_WINDOW_POLICY_ID:
        reason_codes.append("policy_set_id_invalid")
    if payload.get("production_approved") is not False or payload.get("production_writer_enabled") is not False:
        reason_codes.append("production_flags_not_fail_closed")
    expected: dict[tuple[str, str, str], int | None] = {}
    for family in CROSS_ASSET_STATE_WINDOW_FAMILIES:
        for band, candidates in (("meso", MESO_WINDOW_CANDIDATES), ("macro", MACRO_WINDOW_CANDIDATES)):
            for candidate, lookback in candidates:
                expected[(family, band, candidate)] = lookback
    observed: set[tuple[str, str, str]] = set()
    for index, policy in enumerate(policies):
        family = str(policy.get("relationship_feature_family", ""))
        band = str(policy.get("band", ""))
        candidate = str(policy.get("window_candidate_name", ""))
        key = (family, band, candidate)
        if key not in expected:
            reason_codes.append(f"policy_{index}_unexpected_family_band_candidate")
            continue
        observed.add(key)
        if policy.get("window_policy_id") in (None, ""):
            reason_codes.append(f"policy_{index}_window_policy_id_missing")
        if policy.get("lookback_days") != expected[key]:
            reason_codes.append(f"policy_{index}_lookback_days_invalid")
        if int(policy.get("min_rows") or 0) <= 0:
            reason_codes.append(f"policy_{index}_min_rows_invalid")
        if candidate == "current_default" and policy.get("source_tail_anchor") is not False:
            reason_codes.append(f"policy_{index}_current_default_anchor_invalid")
        if candidate != "current_default" and policy.get("source_tail_anchor") is not True:
            reason_codes.append(f"policy_{index}_source_tail_anchor_invalid")
        if policy.get("production_approved") is not False or policy.get("production_writer_enabled") is not False:
            reason_codes.append(f"policy_{index}_production_flags_not_fail_closed")
    missing = sorted(set(expected) - observed)
    if missing:
        reason_codes.append("required_window_candidates_missing")
    return {
        "artifact_kind": "cross_asset_state_window_policy_manifest_validation",
        "status": "passed" if not reason_codes else "blocked",
        "passed": not reason_codes,
        "policy_set_id": CROSS_ASSET_STATE_WINDOW_POLICY_ID,
        "policy_count": len(policies),
        "expected_policy_count": len(expected),
        "missing_policy_keys": ["|".join(key) for key in missing],
        "reason_codes": list(dict.fromkeys(reason_codes)),
        "production_write_allowed": False,
    }


def resolve_cross_asset_state_window_policy(
    *,
    relationship_feature_family: str,
    band: str,
    window_policy_id: str | None = None,
) -> CrossAssetStateWindowPolicy:
    family = str(relationship_feature_family).strip()
    band_key = str(band).strip().lower()
    policies = default_cross_asset_state_window_policies()
    if window_policy_id:
        for policy in policies:
            if policy.window_policy_id == str(window_policy_id):
                if policy.relationship_feature_family != family or policy.band != band_key:
                    raise ValueError("Cross-Asset-State window_policy_id does not match family/band")
                return policy
        raise ValueError(f"Unknown Cross-Asset-State window_policy_id {window_policy_id!r}")
    for policy in policies:
        if policy.relationship_feature_family == family and policy.band == band_key:
            return policy
    raise ValueError(f"No Cross-Asset-State window policy for {family!r}/{band_key!r}")


def apply_cross_asset_state_window_policy(
    frame: Any,
    policy: CrossAssetStateWindowPolicy,
    *,
    source_tail_ts: object | None,
    min_rows: int | None = None,
) -> tuple[Any, CrossAssetStateWindowCoverage]:
    pd = _pandas()
    if frame is None:
        return frame, _coverage(policy, 0, min_rows=min_rows, source_tail_ts=None, start_ts=None, end_ts=None, passed=False)
    work = frame.copy()
    if "ts" not in work.columns:
        return work.iloc[0:0].copy(), _coverage(policy, 0, min_rows=min_rows, source_tail_ts=None, start_ts=None, end_ts=None, passed=False)
    work["ts"] = pd.to_numeric(work["ts"], errors="coerce")
    work = work.dropna(subset=["ts"]).sort_values("ts").reset_index(drop=True)
    tail = _coerce_int(source_tail_ts)
    if tail is None:
        tail = _coerce_int(_max_numeric(work.get("source_tail_ts"))) or _coerce_int(work["ts"].max())
    resolved = policy.as_regime_window_profile().resolve(source_tail_ts=tail)
    if resolved.start_ts is not None:
        work = work[work["ts"] >= int(resolved.start_ts)]
    if resolved.end_ts is not None:
        work = work[work["ts"] <= int(resolved.end_ts)]
    if policy.row_cap is not None and len(work) > int(policy.row_cap):
        work = work.tail(int(policy.row_cap))
    observed = int(len(work))
    required = max(int(policy.min_rows), int(min_rows or 0))
    passed = observed >= required
    return work.copy(), _coverage(
        policy,
        observed,
        min_rows=required,
        source_tail_ts=tail,
        start_ts=resolved.start_ts,
        end_ts=resolved.end_ts,
        passed=passed,
    )


def family_window_sensitivity_summary() -> dict[str, Any]:
    return {
        "anchor_core_exposure": {
            "window_sensitive": True,
            "reason": "Primary correlation can saturate; beta/core-basket and secondary correlation should be checked across windows.",
            "eventual_test_branch_policies": [
                "current_default",
                "meso_short",
                "meso_medium",
                "meso_long",
                "macro_short",
                "macro_medium",
                "macro_long",
            ],
        },
        "peer_strength_stability": {
            "window_sensitive": True,
            "reason": "Peer count, strength, and stability can compress when the lookback is too short or too long.",
            "eventual_test_branch_policies": [
                "current_default",
                "meso_short",
                "meso_medium",
                "meso_long",
                "macro_short",
                "macro_medium",
                "macro_long",
            ],
        },
        "relationship_concentration_entropy": {
            "window_sensitive": True,
            "reason": "Concentration/entropy needs rank or spread changes over enough history to avoid flat states.",
            "eventual_test_branch_policies": [
                "current_default",
                "meso_short",
                "meso_medium",
                "meso_long",
                "macro_short",
                "macro_medium",
                "macro_long",
            ],
        },
        "residual_peer_signal": {
            "window_sensitive": "moderate",
            "reason": "Signed residual magnitude is the positive control; use shorter windows but verify macro persistence.",
            "eventual_test_branch_policies": [
                "current_default",
                "meso_short",
                "meso_medium",
                "meso_long",
                "macro_short",
                "macro_medium",
                "macro_long",
            ],
        },
    }


def _coverage(
    policy: CrossAssetStateWindowPolicy,
    observed_rows: int,
    *,
    min_rows: int | None,
    source_tail_ts: int | None,
    start_ts: int | None,
    end_ts: int | None,
    passed: bool,
) -> CrossAssetStateWindowCoverage:
    required = max(int(policy.min_rows), int(min_rows or 0))
    return CrossAssetStateWindowCoverage(
        status="passed" if passed else "masked_unavailable",
        passed=bool(passed),
        window_policy=policy,
        observed_rows=int(observed_rows),
        min_rows=required,
        start_ts=start_ts,
        end_ts=end_ts,
        source_tail_ts=source_tail_ts,
        reason_code=None if passed else CROSS_ASSET_STATE_WINDOW_INSUFFICIENT_HISTORY,
    )


def _coerce_int(value: object | None) -> int | None:
    if value is None:
        return None
    try:
        out = int(float(value))
    except Exception:
        return None
    return out


def _candidate_name_from_policy_id(policy_id: str) -> str:
    for token in (
        "current_default",
        "meso_short",
        "meso_medium",
        "meso_long",
        "macro_short",
        "macro_medium",
        "macro_long",
    ):
        if token in policy_id:
            return token
    return "custom"


def _window_candidate_rationale(candidate: str) -> str:
    return {
        "current_default": "Current available rows after relationship availability filtering.",
        "meso_short": "Short meso lookback for faster relationship changes.",
        "meso_medium": "Medium meso lookback balancing responsiveness and stability.",
        "meso_long": "Long meso lookback for persistent peer and concentration structure.",
        "macro_short": "Short macro lookback for recent dependency changes.",
        "macro_medium": "Medium macro lookback for annual relationship structure.",
        "macro_long": "Long macro lookback for durable peer and concentration structure.",
    }.get(candidate, "Custom Cross-Asset-State window candidate.")


def _max_numeric(series: object) -> object | None:
    if series is None:
        return None
    pd = _pandas()
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return None
    return values.max()


def _pandas() -> Any:
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("Cross-Asset-State window policy requires pandas") from exc
    return pd


__all__ = [
    "CROSS_ASSET_STATE_WINDOW_INSUFFICIENT_HISTORY",
    "CROSS_ASSET_STATE_WINDOW_FAMILIES",
    "CROSS_ASSET_STATE_WINDOW_POLICY_ID",
    "CROSS_ASSET_STATE_WINDOW_POLICY_SCHEMA_VERSION",
    "MACRO_WINDOW_CANDIDATES",
    "MESO_WINDOW_CANDIDATES",
    "CrossAssetStateWindowCoverage",
    "CrossAssetStateWindowPolicy",
    "apply_cross_asset_state_window_policy",
    "cross_asset_state_window_ladder_policies",
    "cross_asset_state_window_policy_manifest",
    "default_cross_asset_state_window_policies",
    "family_window_sensitivity_summary",
    "resolve_cross_asset_state_window_policy",
    "validate_cross_asset_state_window_policy_manifest",
]
