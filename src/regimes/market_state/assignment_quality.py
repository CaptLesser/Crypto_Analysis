from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from src.regimes.core.serialization import to_jsonable


ASSIGNMENT_STATUS_VALID = "valid"
ASSIGNMENT_STATUS_WARMUP_MASKED = "warmup_masked"
ASSIGNMENT_STATUS_INSUFFICIENT_HISTORY = "insufficient_history"
ASSIGNMENT_STATUS_INSUFFICIENT_UNIVERSE_COVERAGE = "insufficient_universe_coverage"
ASSIGNMENT_STATUS_NONFINITE_INPUT_BLOCKED = "nonfinite_input_blocked"
ASSIGNMENT_STATUS_NEAR_CONSTANT_INPUT_BLOCKED = "near_constant_input_blocked"
ASSIGNMENT_STATUS_LOW_CONFIDENCE = "low_confidence"
ASSIGNMENT_STATUS_MODEL_DEGENERATE = "model_degenerate"
ASSIGNMENT_STATUS_MODEL_UNAVAILABLE = "model_unavailable"
ASSIGNMENT_STATUS_NOT_SELECTED = "not_selected"

MARKET_STATE_ASSIGNMENT_STATUSES: tuple[str, ...] = (
    ASSIGNMENT_STATUS_VALID,
    ASSIGNMENT_STATUS_WARMUP_MASKED,
    ASSIGNMENT_STATUS_INSUFFICIENT_HISTORY,
    ASSIGNMENT_STATUS_INSUFFICIENT_UNIVERSE_COVERAGE,
    ASSIGNMENT_STATUS_NONFINITE_INPUT_BLOCKED,
    ASSIGNMENT_STATUS_NEAR_CONSTANT_INPUT_BLOCKED,
    ASSIGNMENT_STATUS_LOW_CONFIDENCE,
    ASSIGNMENT_STATUS_MODEL_DEGENERATE,
    ASSIGNMENT_STATUS_MODEL_UNAVAILABLE,
    ASSIGNMENT_STATUS_NOT_SELECTED,
)

CONFIDENCE_STRATEGY_KMEANS_DISTANCE_MARGIN = "kmeans_distance_margin"
CONFIDENCE_STRATEGY_POSTERIOR_PROBABILITY = "posterior_probability"
CONFIDENCE_STRATEGY_HDBSCAN_MEMBERSHIP_PROBABILITY = "hdbscan_membership_probability"
CONFIDENCE_STRATEGY_UNSUPPORTED_NATIVE_ASSIGNMENT = "unsupported_native_assignment"
CONFIDENCE_STRATEGY_DEFERRED_UNAVAILABLE = "deferred_unavailable"

STATE_SCORE_STRATEGY_ROBUST_PERCENTILE_INTENSITY = "robust_percentile_intensity"

KMEANS_CONFIDENCE_METHODS: frozenset[str] = frozenset(
    {
        "kmeans",
        "minibatch_kmeans",
        "pca_kmeans",
        "factor_analysis_kmeans",
        "spectral_embedding_kmeans",
        "birch_global_kmeans",
        "birch",
    }
)
POSTERIOR_CONFIDENCE_METHODS: frozenset[str] = frozenset(
    {
        "gaussian_mixture",
        "bayesian_gaussian_mixture",
        "pca_gaussian_mixture",
        "pca_bayesian_gaussian_mixture",
        "factor_analysis_gaussian_mixture",
    }
)
LESS_SUITABLE_ASSIGNMENT_METHODS: frozenset[str] = frozenset({"spectral_clustering", "agglomerative", "pca_agglomerative", "optics"})
HDBSCAN_CONFIDENCE_METHODS: frozenset[str] = frozenset({"hdbscan", "pca_hdbscan"})
DEFERRED_ASSIGNMENT_METHODS: frozenset[str] = frozenset()


@dataclass(frozen=True)
class MarketStateAssignmentQualityPolicy:
    min_history_rows: int = 20
    min_universe_coverage: float = 0.5
    near_constant_variance_threshold: float = 1e-12
    low_confidence_threshold: float = 0.20
    min_effective_states: int = 2
    status_column: str = "assignment_status"
    confidence_column: str = "assignment_confidence"
    state_score_column: str = "state_score"
    schema_version: int = 1
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "min_history_rows", max(1, int(self.min_history_rows)))
        object.__setattr__(self, "min_effective_states", max(1, int(self.min_effective_states)))
        object.__setattr__(self, "min_universe_coverage", _clamp01(float(self.min_universe_coverage)))
        object.__setattr__(self, "near_constant_variance_threshold", max(0.0, float(self.near_constant_variance_threshold)))
        object.__setattr__(self, "low_confidence_threshold", _clamp01(float(self.low_confidence_threshold)))
        object.__setattr__(self, "status_column", _text(self.status_column, field_name="status_column"))
        object.__setattr__(self, "confidence_column", _text(self.confidence_column, field_name="confidence_column"))
        object.__setattr__(self, "state_score_column", _text(self.state_score_column, field_name="state_score_column"))
        object.__setattr__(self, "schema_version", int(self.schema_version))
        object.__setattr__(self, "metadata", to_jsonable(dict(self.metadata)))

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "artifact_kind": "market_state_assignment_quality_policy",
            "min_history_rows": int(self.min_history_rows),
            "min_universe_coverage": float(self.min_universe_coverage),
            "near_constant_variance_threshold": float(self.near_constant_variance_threshold),
            "low_confidence_threshold": float(self.low_confidence_threshold),
            "min_effective_states": int(self.min_effective_states),
            "status_column": self.status_column,
            "confidence_column": self.confidence_column,
            "state_score_column": self.state_score_column,
            "metadata": to_jsonable(dict(self.metadata)),
            "production_labels_written": False,
            "final_profile_selection": False,
        }


def default_market_state_assignment_quality_policy() -> MarketStateAssignmentQualityPolicy:
    return MarketStateAssignmentQualityPolicy()


def market_state_assignment_status_vocabulary() -> dict[str, dict[str, Any]]:
    return {
        ASSIGNMENT_STATUS_VALID: {"terminal": False, "state_allowed": True, "meaning": "row has finite inputs and supported model assignment quality"},
        ASSIGNMENT_STATUS_WARMUP_MASKED: {"terminal": True, "state_allowed": False, "meaning": "row is inside configured warmup and must not emit an assignment"},
        ASSIGNMENT_STATUS_INSUFFICIENT_HISTORY: {"terminal": True, "state_allowed": False, "meaning": "feature bundle lacks enough history for assignment"},
        ASSIGNMENT_STATUS_INSUFFICIENT_UNIVERSE_COVERAGE: {"terminal": True, "state_allowed": False, "meaning": "row universe coverage is below assignment threshold"},
        ASSIGNMENT_STATUS_NONFINITE_INPUT_BLOCKED: {"terminal": True, "state_allowed": False, "meaning": "one or more required assignment inputs are NaN or infinite"},
        ASSIGNMENT_STATUS_NEAR_CONSTANT_INPUT_BLOCKED: {"terminal": True, "state_allowed": False, "meaning": "candidate feature bundle has no usable variation"},
        ASSIGNMENT_STATUS_LOW_CONFIDENCE: {"terminal": False, "state_allowed": True, "meaning": "row is assigned but confidence is below Test policy threshold"},
        ASSIGNMENT_STATUS_MODEL_DEGENERATE: {"terminal": True, "state_allowed": False, "meaning": "model labels collapse into too few effective states or excessive noise"},
        ASSIGNMENT_STATUS_MODEL_UNAVAILABLE: {"terminal": True, "state_allowed": False, "meaning": "method has no supported per-row assignment confidence in this contract"},
        ASSIGNMENT_STATUS_NOT_SELECTED: {"terminal": True, "state_allowed": False, "meaning": "axis/profile/method was not selected for this output contract"},
    }


def market_state_confidence_method_semantics() -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for method in sorted(KMEANS_CONFIDENCE_METHODS):
        out[method] = {
            "confidence_strategy": CONFIDENCE_STRATEGY_KMEANS_DISTANCE_MARGIN,
            "production_suitability": "suitable_if_profile_selected",
            "requires": ["cluster_centroids_or_distances", "finite_preprocessed_features"],
            "notes": "confidence is the distance margin between nearest and second nearest centroid in the assignment space",
        }
    for method in sorted(POSTERIOR_CONFIDENCE_METHODS):
        out[method] = {
            "confidence_strategy": CONFIDENCE_STRATEGY_POSTERIOR_PROBABILITY,
            "production_suitability": "suitable_if_profile_selected",
            "requires": ["posterior_probabilities"],
            "notes": "confidence is max posterior assignment probability",
        }
    for method in sorted(LESS_SUITABLE_ASSIGNMENT_METHODS):
        out[method] = {
            "confidence_strategy": CONFIDENCE_STRATEGY_UNSUPPORTED_NATIVE_ASSIGNMENT,
            "production_suitability": "less_suitable_for_production_assignment",
            "requires": [],
            "notes": "no durable native per-row out-of-sample confidence is defined in this contract",
        }
    for method in sorted(HDBSCAN_CONFIDENCE_METHODS):
        out[method] = {
            "confidence_strategy": CONFIDENCE_STRATEGY_HDBSCAN_MEMBERSHIP_PROBABILITY,
            "production_suitability": "test_eligible_with_density_noise_and_approximate_predict_limits",
            "requires": ["membership_probabilities", "finite_preprocessed_features", "runtime_compatibility_probe"],
            "notes": "confidence is HDBSCAN membership strength; noise labels remain valid model output but require explicit status/degeneracy review",
        }
    for method in sorted(DEFERRED_ASSIGNMENT_METHODS):
        out[method] = {
            "confidence_strategy": CONFIDENCE_STRATEGY_DEFERRED_UNAVAILABLE,
            "production_suitability": "deferred_until_runtime_compatibility_fixed_and_retested",
            "requires": [],
            "notes": "not relied on while the current HDBSCAN runtime compatibility issue remains open",
        }
    return out


def method_confidence_strategy(method_family: str) -> str:
    method = _method(method_family)
    if method in KMEANS_CONFIDENCE_METHODS:
        return CONFIDENCE_STRATEGY_KMEANS_DISTANCE_MARGIN
    if method in POSTERIOR_CONFIDENCE_METHODS:
        return CONFIDENCE_STRATEGY_POSTERIOR_PROBABILITY
    if method in HDBSCAN_CONFIDENCE_METHODS:
        return CONFIDENCE_STRATEGY_HDBSCAN_MEMBERSHIP_PROBABILITY
    if method in DEFERRED_ASSIGNMENT_METHODS:
        return CONFIDENCE_STRATEGY_DEFERRED_UNAVAILABLE
    return CONFIDENCE_STRATEGY_UNSUPPORTED_NATIVE_ASSIGNMENT


def apply_market_state_assignment_quality(
    frame: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
    method_family: str,
    labels: Sequence[object] | None = None,
    probabilities: Sequence[Sequence[float]] | np.ndarray | None = None,
    distances: Sequence[Sequence[float]] | np.ndarray | None = None,
    centroids: Sequence[Sequence[float]] | np.ndarray | None = None,
    ordinal_feature: str | None = None,
    ordinal_direction: str = "higher_is_higher_state",
    coverage_column: str | None = None,
    selected: bool = True,
    policy: MarketStateAssignmentQualityPolicy | None = None,
) -> pd.DataFrame:
    cfg = policy or default_market_state_assignment_quality_policy()
    if not isinstance(frame, pd.DataFrame):
        raise ValueError("Market-State assignment quality requires a DataFrame")
    out = frame.copy()
    features = _feature_columns(out, feature_columns)
    validate_causal_metadata(out)
    if not selected:
        out[cfg.status_column] = ASSIGNMENT_STATUS_NOT_SELECTED
        out[cfg.confidence_column] = None
        out[cfg.state_score_column] = None
        out["assignment_quality_metadata"] = [_metadata(method_family, cfg, selected=False)] * len(out)
        return out

    statuses = _input_statuses(out, features, coverage_column=coverage_column, policy=cfg)
    if _bundle_near_constant(out, features, statuses, policy=cfg):
        statuses = [ASSIGNMENT_STATUS_NEAR_CONSTANT_INPUT_BLOCKED if status == ASSIGNMENT_STATUS_VALID else status for status in statuses]

    confidences = _confidence_values(
        out,
        features=features,
        method_family=method_family,
        probabilities=probabilities,
        distances=distances,
        centroids=centroids,
        policy=cfg,
    )
    labels_arr = np.asarray(labels, dtype=object) if labels is not None else None
    if labels_arr is not None and len(labels_arr) != len(out):
        raise ValueError("Market-State assignment labels length must match frame rows")
    degenerate = _labels_degenerate(labels_arr, policy=cfg) if labels_arr is not None else False
    strategy = method_confidence_strategy(method_family)
    for idx, status in enumerate(tuple(statuses)):
        if status != ASSIGNMENT_STATUS_VALID:
            confidences[idx] = None
            continue
        if strategy in {CONFIDENCE_STRATEGY_UNSUPPORTED_NATIVE_ASSIGNMENT, CONFIDENCE_STRATEGY_DEFERRED_UNAVAILABLE}:
            statuses[idx] = ASSIGNMENT_STATUS_MODEL_UNAVAILABLE
            confidences[idx] = None
        elif degenerate:
            statuses[idx] = ASSIGNMENT_STATUS_MODEL_DEGENERATE
            confidences[idx] = None
        elif confidences[idx] is None or not np.isfinite(float(confidences[idx])):
            statuses[idx] = ASSIGNMENT_STATUS_MODEL_UNAVAILABLE
            confidences[idx] = None
        elif float(confidences[idx]) < float(cfg.low_confidence_threshold):
            statuses[idx] = ASSIGNMENT_STATUS_LOW_CONFIDENCE

    scores = state_score_values(
        out,
        ordinal_feature=ordinal_feature or features[0],
        ordinal_direction=ordinal_direction,
        statuses=statuses,
        policy=cfg,
    )
    out[cfg.status_column] = statuses
    out[cfg.confidence_column] = confidences
    out[cfg.state_score_column] = scores
    out["assignment_quality_metadata"] = [_metadata(method_family, cfg, selected=True, confidence_strategy=strategy)] * len(out)
    validate_assignment_quality_output(out, policy=cfg)
    return out


def validate_causal_metadata(frame: pd.DataFrame, required: Sequence[str] = ("ts", "known_at_ts", "source_tail_ts")) -> None:
    if not isinstance(frame, pd.DataFrame):
        raise ValueError("Market-State causal metadata validation requires a DataFrame")
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Market-State row-producing output missing causal fields: {missing}")
    for column in required:
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.isna().any() or not np.isfinite(values.to_numpy(dtype=float)).all():
            raise ValueError(f"Market-State causal field {column} must be finite and non-null")
    source_tail = pd.to_numeric(frame["source_tail_ts"], errors="coerce")
    known_at = pd.to_numeric(frame["known_at_ts"], errors="coerce")
    if (source_tail > known_at).any():
        raise ValueError("Market-State source_tail_ts must not exceed known_at_ts")


def validate_assignment_quality_output(
    frame: pd.DataFrame,
    *,
    policy: MarketStateAssignmentQualityPolicy | None = None,
) -> None:
    cfg = policy or default_market_state_assignment_quality_policy()
    validate_causal_metadata(frame)
    missing = [column for column in (cfg.status_column, cfg.confidence_column, cfg.state_score_column) if column not in frame.columns]
    if missing:
        raise ValueError(f"Market-State assignment quality output missing fields: {missing}")
    invalid_statuses = sorted(set(str(value) for value in frame[cfg.status_column].dropna()).difference(MARKET_STATE_ASSIGNMENT_STATUSES))
    if invalid_statuses:
        raise ValueError(f"Unsupported Market-State assignment statuses: {invalid_statuses}")
    active = frame[cfg.status_column].isin((ASSIGNMENT_STATUS_VALID, ASSIGNMENT_STATUS_LOW_CONFIDENCE))
    for column in (cfg.confidence_column, cfg.state_score_column):
        numeric = pd.to_numeric(frame[column], errors="coerce")
        if active.any():
            active_values = numeric.loc[active]
            if active_values.isna().any() or not np.isfinite(active_values.to_numpy(dtype=float)).all():
                raise ValueError(f"Market-State active assignment field {column} must be finite")
            if ((active_values < 0.0) | (active_values > 1.0)).any():
                raise ValueError(f"Market-State active assignment field {column} must be bounded in [0, 1]")
        masked_values = numeric.loc[~active & frame[column].notna()]
        if not masked_values.empty and not np.isfinite(masked_values.to_numpy(dtype=float)).all():
            raise ValueError(f"Market-State masked assignment field {column} cannot contain NaN/inf")


def state_score_values(
    frame: pd.DataFrame,
    *,
    ordinal_feature: str,
    ordinal_direction: str = "higher_is_higher_state",
    statuses: Sequence[str] | None = None,
    policy: MarketStateAssignmentQualityPolicy | None = None,
) -> list[float | None]:
    cfg = policy or default_market_state_assignment_quality_policy()
    feature = _text(ordinal_feature, field_name="ordinal_feature")
    if feature not in frame.columns:
        raise ValueError(f"Market-State state score feature {feature!r} is missing")
    values = pd.to_numeric(frame[feature], errors="coerce")
    active_statuses = tuple(statuses) if statuses is not None else (ASSIGNMENT_STATUS_VALID,) * len(frame)
    if len(active_statuses) != len(frame):
        raise ValueError("Market-State state score statuses length must match rows")
    active_mask = pd.Series([status in (ASSIGNMENT_STATUS_VALID, ASSIGNMENT_STATUS_LOW_CONFIDENCE) for status in active_statuses], index=frame.index)
    finite_active = values.loc[active_mask & values.notna() & np.isfinite(values.to_numpy(dtype=float))]
    scores: list[float | None] = [None] * len(frame)
    if finite_active.empty:
        return scores
    ranks = finite_active.rank(method="average", pct=True)
    if ordinal_direction == "lower_is_higher_state":
        ranks = 1.0 - ranks + (1.0 / max(1, len(ranks)))
    elif ordinal_direction != "higher_is_higher_state":
        raise ValueError("Market-State ordinal_direction must be higher_is_higher_state or lower_is_higher_state")
    for idx, value in ranks.clip(0.0, 1.0).items():
        scores[frame.index.get_loc(idx)] = float(value)
    return scores


def _input_statuses(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
    *,
    coverage_column: str | None,
    policy: MarketStateAssignmentQualityPolicy,
) -> list[str]:
    if len(frame) < int(policy.min_history_rows):
        return [ASSIGNMENT_STATUS_INSUFFICIENT_HISTORY] * len(frame)
    feature_values = frame[list(feature_columns)].apply(pd.to_numeric, errors="coerce")
    finite = np.isfinite(feature_values.to_numpy(dtype=float))
    statuses: list[str] = []
    coverage_values = pd.to_numeric(frame[coverage_column], errors="coerce") if coverage_column and coverage_column in frame.columns else None
    for idx in range(len(frame)):
        if idx < int(policy.min_history_rows) - 1:
            statuses.append(ASSIGNMENT_STATUS_WARMUP_MASKED)
        elif coverage_values is not None and (pd.isna(coverage_values.iloc[idx]) or float(coverage_values.iloc[idx]) < float(policy.min_universe_coverage)):
            statuses.append(ASSIGNMENT_STATUS_INSUFFICIENT_UNIVERSE_COVERAGE)
        elif not bool(finite[idx].all()):
            statuses.append(ASSIGNMENT_STATUS_NONFINITE_INPUT_BLOCKED)
        else:
            statuses.append(ASSIGNMENT_STATUS_VALID)
    return statuses


def _confidence_values(
    frame: pd.DataFrame,
    *,
    features: Sequence[str],
    method_family: str,
    probabilities: Sequence[Sequence[float]] | np.ndarray | None,
    distances: Sequence[Sequence[float]] | np.ndarray | None,
    centroids: Sequence[Sequence[float]] | np.ndarray | None,
    policy: MarketStateAssignmentQualityPolicy,
) -> list[float | None]:
    strategy = method_confidence_strategy(method_family)
    if strategy == CONFIDENCE_STRATEGY_POSTERIOR_PROBABILITY:
        if probabilities is None:
            return [None] * len(frame)
        probs = np.asarray(probabilities, dtype=float)
        if probs.ndim != 2 or probs.shape[0] != len(frame) or probs.shape[1] < 2:
            return [None] * len(frame)
        clean = np.where(np.isfinite(probs), probs, np.nan)
        return [_finite_clamp01(value) for value in np.nanmax(clean, axis=1)]
    if strategy == CONFIDENCE_STRATEGY_HDBSCAN_MEMBERSHIP_PROBABILITY:
        if probabilities is None:
            return [None] * len(frame)
        probs = np.asarray(probabilities, dtype=float)
        if probs.ndim == 2 and probs.shape[0] == len(frame):
            clean = np.where(np.isfinite(probs), probs, np.nan)
            return [_finite_clamp01(value) for value in np.nanmax(clean, axis=1)]
        if probs.ndim == 1 and probs.shape[0] == len(frame):
            return [_finite_clamp01(value) for value in probs]
        return [None] * len(frame)
    if strategy == CONFIDENCE_STRATEGY_KMEANS_DISTANCE_MARGIN:
        dist = _distance_matrix(frame, features=features, distances=distances, centroids=centroids)
        if dist is None or dist.ndim != 2 or dist.shape[0] != len(frame) or dist.shape[1] < 2:
            return [None] * len(frame)
        sorted_dist = np.sort(np.where(np.isfinite(dist), dist, np.nan), axis=1)
        nearest = sorted_dist[:, 0]
        second = sorted_dist[:, 1]
        margin = (second - nearest) / np.maximum(second, 1e-12)
        return [_finite_clamp01(value) for value in margin]
    return [None] * len(frame)


def _distance_matrix(
    frame: pd.DataFrame,
    *,
    features: Sequence[str],
    distances: Sequence[Sequence[float]] | np.ndarray | None,
    centroids: Sequence[Sequence[float]] | np.ndarray | None,
) -> np.ndarray | None:
    if distances is not None:
        return np.asarray(distances, dtype=float)
    if centroids is None:
        return None
    x = frame[list(features)].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    c = np.asarray(centroids, dtype=float)
    if c.ndim != 2 or x.ndim != 2 or c.shape[1] != x.shape[1]:
        return None
    return np.linalg.norm(x[:, None, :] - c[None, :, :], axis=2)


def _bundle_near_constant(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
    statuses: Sequence[str],
    *,
    policy: MarketStateAssignmentQualityPolicy,
) -> bool:
    active = [idx for idx, status in enumerate(statuses) if status == ASSIGNMENT_STATUS_VALID]
    if len(active) < int(policy.min_history_rows):
        return False
    values = frame.iloc[active][list(feature_columns)].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(values).all():
        return False
    variances = np.nanvar(values, axis=0)
    return bool(np.all(variances <= float(policy.near_constant_variance_threshold)))


def _labels_degenerate(labels: np.ndarray | None, *, policy: MarketStateAssignmentQualityPolicy) -> bool:
    if labels is None or labels.size == 0:
        return False
    observed = [str(label) for label in labels.tolist() if str(label) != "-1"]
    if len(set(observed)) < int(policy.min_effective_states):
        return True
    counts = pd.Series(observed, dtype=object).value_counts()
    if counts.empty:
        return True
    return bool((counts == 1).mean() > 0.5)


def _feature_columns(frame: pd.DataFrame, feature_columns: Sequence[str]) -> tuple[str, ...]:
    if isinstance(feature_columns, (str, bytes)) or not isinstance(feature_columns, Sequence):
        raise ValueError("Market-State assignment feature_columns must be a sequence")
    features = tuple(dict.fromkeys(_text(column, field_name="feature_column") for column in feature_columns))
    if not features:
        raise ValueError("Market-State assignment requires at least one feature column")
    missing = [column for column in features if column not in frame.columns]
    if missing:
        raise ValueError(f"Market-State assignment feature columns missing: {missing}")
    return features


def _metadata(
    method_family: str,
    policy: MarketStateAssignmentQualityPolicy,
    *,
    selected: bool,
    confidence_strategy: str | None = None,
) -> dict[str, Any]:
    method = _method(method_family)
    semantics = market_state_confidence_method_semantics().get(method, {})
    return {
        "artifact_kind": "market_state_assignment_quality_metadata",
        "method_family": method,
        "selected": bool(selected),
        "confidence_strategy": confidence_strategy or method_confidence_strategy(method),
        "state_score_strategy": STATE_SCORE_STRATEGY_ROBUST_PERCENTILE_INTENSITY,
        "method_semantics": semantics,
        "policy": policy.as_dict(),
        "production_labels_written": False,
        "final_profile_selection": False,
    }


def _method(value: object) -> str:
    return _text(value, field_name="method_family").lower()


def _text(value: object, *, field_name: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"Market-State assignment quality {field_name} must be non-empty")
    return text


def _clamp01(value: float) -> float:
    return float(min(1.0, max(0.0, value)))


def _finite_clamp01(value: object) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    if not np.isfinite(out):
        return None
    return _clamp01(out)


__all__ = [
    "ASSIGNMENT_STATUS_INSUFFICIENT_HISTORY",
    "ASSIGNMENT_STATUS_INSUFFICIENT_UNIVERSE_COVERAGE",
    "ASSIGNMENT_STATUS_LOW_CONFIDENCE",
    "ASSIGNMENT_STATUS_MODEL_DEGENERATE",
    "ASSIGNMENT_STATUS_MODEL_UNAVAILABLE",
    "ASSIGNMENT_STATUS_NEAR_CONSTANT_INPUT_BLOCKED",
    "ASSIGNMENT_STATUS_NONFINITE_INPUT_BLOCKED",
    "ASSIGNMENT_STATUS_NOT_SELECTED",
    "ASSIGNMENT_STATUS_VALID",
    "ASSIGNMENT_STATUS_WARMUP_MASKED",
    "CONFIDENCE_STRATEGY_DEFERRED_UNAVAILABLE",
    "CONFIDENCE_STRATEGY_HDBSCAN_MEMBERSHIP_PROBABILITY",
    "CONFIDENCE_STRATEGY_KMEANS_DISTANCE_MARGIN",
    "CONFIDENCE_STRATEGY_POSTERIOR_PROBABILITY",
    "CONFIDENCE_STRATEGY_UNSUPPORTED_NATIVE_ASSIGNMENT",
    "MARKET_STATE_ASSIGNMENT_STATUSES",
    "STATE_SCORE_STRATEGY_ROBUST_PERCENTILE_INTENSITY",
    "MarketStateAssignmentQualityPolicy",
    "apply_market_state_assignment_quality",
    "default_market_state_assignment_quality_policy",
    "market_state_assignment_status_vocabulary",
    "market_state_confidence_method_semantics",
    "method_confidence_strategy",
    "state_score_values",
    "validate_assignment_quality_output",
    "validate_causal_metadata",
]
