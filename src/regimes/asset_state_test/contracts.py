from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Optional, Sequence

from src.regimes.contracts import REGIME_AXIS_ORDER, REGIME_BANDS


STUDY_LAYER = "asset_state"
DEFAULT_AXIS = "trend"
DEFAULT_BAND = "micro"
DEFAULT_ASSETS: tuple[str, ...] = ("AAVEUSD", "XBTUSD")
DEFAULT_CYCLE5_ASSETS: tuple[str, ...] = ("AAVEUSD", "XBTUSD", "ADAUSD", "AI16ZUSD")
DEFAULT_FIRST_METHODS: tuple[str, ...] = ("hdbscan", "kmeans", "gaussian_mixture")
ADAPTER_READY_METHODS: tuple[str, ...] = (
    "hdbscan",
    "kmeans",
    "minibatch_kmeans",
    "gaussian_mixture",
    "bayesian_gaussian_mixture",
    "optics",
    "agglomerative",
    "birch",
)

FLAT_REASON_CODES: tuple[str, ...] = (
    "pass",
    "valid_flat_or_pegged",
    "insufficient_variance",
    "bad_or_missing_data",
    "low_activity",
    "axis_not_clusterable",
)

ARTIFACT_NAMES: tuple[str, ...] = (
    "trial_manifest.json",
    "candidate_scores.csv",
    "flat_preflight.csv",
    "cluster_diagnostics.json",
    "per_asset_summary.csv",
    "per_asset_summary.json",
    "asset_model_decisions.csv",
    "asset_model_decisions.json",
    "runtime_summary.json",
    "aggregate_summary.json",
    "experiment_config_snapshot.json",
    "command_log.txt",
    "artifact_validation.json",
)


def _json_default(value: object) -> object:
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, set):
        return sorted(str(v) for v in value)
    return str(value)


def stable_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, default=_json_default, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def safe_float(value: object, default: Optional[float] = None) -> Optional[float]:
    try:
        out = float(value)
    except Exception:
        return default
    if not math.isfinite(out):
        return default
    return out


@dataclass(frozen=True)
class StudyConfig:
    layer: str = STUDY_LAYER
    axis: str = DEFAULT_AXIS
    band: str = DEFAULT_BAND
    assets: tuple[str, ...] = DEFAULT_ASSETS
    train_start_ts: Optional[int] = None
    train_end_ts: Optional[int] = None
    eval_start_ts: Optional[int] = None
    eval_end_ts: Optional[int] = None
    methods: tuple[str, ...] = DEFAULT_FIRST_METHODS
    preprocess: str = "robust_scale"
    feature_strategy: str = "manual_baseline"
    feature_bases: tuple[str, ...] = ("log_return", "macd_hist_12_26_9", "rsi_14", "adx_14")
    member_intervals: tuple[int, ...] = (1, 5, 15, 30)
    random_state: int = 17
    workers: int = 1
    notes: str = ""

    def __post_init__(self) -> None:
        if self.layer != STUDY_LAYER:
            raise ValueError(f"Only {STUDY_LAYER!r} studies are supported, got {self.layer!r}")
        if self.axis not in REGIME_AXIS_ORDER:
            raise ValueError(f"Unsupported axis {self.axis!r}; expected one of {REGIME_AXIS_ORDER}")
        if self.band not in REGIME_BANDS:
            raise ValueError(f"Unsupported band {self.band!r}; expected one of {tuple(REGIME_BANDS)}")
        band = REGIME_BANDS[self.band]
        if tuple(int(v) for v in self.member_intervals) != tuple(int(v) for v in band.member_intervals):
            raise ValueError(f"Member intervals {self.member_intervals!r} do not match band {self.band!r}")
        if not self.assets:
            raise ValueError("At least one asset is required")
        if not self.methods:
            raise ValueError("At least one clustering method is required")

    @property
    def ceiling_interval_min(self) -> int:
        return int(REGIME_BANDS[self.band].ceiling_interval_min)

    @property
    def config_hash(self) -> str:
        return stable_hash(asdict(self))

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["ceiling_interval_min"] = self.ceiling_interval_min
        out["config_hash"] = self.config_hash
        return out


@dataclass(frozen=True)
class TrialConfig:
    study: StudyConfig
    method: str
    method_params: Mapping[str, Any] = field(default_factory=dict)
    preprocess: Optional[str] = None
    feature_strategy: Optional[str] = None
    trial_id: str = "manual"
    grid_family: Optional[str] = None
    grid_variant_id: Optional[str] = None

    def __post_init__(self) -> None:
        method = str(self.method).strip().lower()
        if method not in ADAPTER_READY_METHODS:
            raise ValueError(f"Unsupported method {self.method!r}; expected one of {ADAPTER_READY_METHODS}")

    @property
    def resolved_preprocess(self) -> str:
        return str(self.preprocess or self.study.preprocess)

    @property
    def resolved_feature_strategy(self) -> str:
        return str(self.feature_strategy or self.study.feature_strategy)

    @property
    def config_hash(self) -> str:
        return stable_hash(self.to_dict(include_hash=False))

    def to_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        out = {
            "trial_id": str(self.trial_id),
            "grid_family": self.grid_family,
            "grid_variant_id": self.grid_variant_id,
            "study": self.study.to_dict(),
            "method": str(self.method).strip().lower(),
            "method_params": dict(self.method_params),
            "preprocess": self.resolved_preprocess,
            "feature_strategy": self.resolved_feature_strategy,
        }
        if include_hash:
            out["trial_config_hash"] = self.config_hash
        return out


@dataclass(frozen=True)
class FlatPreflightResult:
    asset: str
    axis: str
    band: str
    train_start_ts: Optional[int]
    train_end_ts: Optional[int]
    row_count: int
    complete_row_count: int
    finite_row_count: int
    missing_fraction: float
    variance_summary: Mapping[str, Any]
    zero_movement_summary: Mapping[str, Any]
    reason_code: str
    pass_flag: bool
    clusterable_candidate: bool
    included_in_fit: bool
    carried_as: str
    affected_axis_band: str
    near_flat_fraction_threshold: Optional[float] = None
    near_flat_distance_to_threshold: Optional[float] = None
    zero_variance_feature_count: Optional[int] = None
    near_zero_variance_feature_count: Optional[int] = None
    near_zero_movement_fraction: Optional[float] = None

    def __post_init__(self) -> None:
        if self.reason_code not in FLAT_REASON_CODES:
            raise ValueError(f"Unsupported flat preflight reason code {self.reason_code!r}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TrialResult:
    trial_config: TrialConfig
    rows_fit: int
    feature_count: int
    labels: Sequence[int]
    diagnostics: Mapping[str, Any]
    elapsed_s: float
    status: str = "ok"
    error: Optional[str] = None

    @property
    def method(self) -> str:
        return str(self.trial_config.method).strip().lower()

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["trial_config"] = self.trial_config.to_dict()
        out["labels"] = [int(v) for v in self.labels]
        return out
