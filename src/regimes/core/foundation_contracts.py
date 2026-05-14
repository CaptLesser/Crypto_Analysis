"""Compatibility contracts retained for legacy foundation diagnostics.

Canonical foundation study code should use ``src.regimes.core.contracts``. This
module remains importable for older diagnostic reports and compatibility tests.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from src.regimes.contracts import REGIME_AXIS_ORDER, REGIME_BANDS


REGIME_FOUNDATION_SCHEMA_VERSION = 1

REGIME_LAYERS: tuple[str, ...] = ("asset_state", "market_state", "relative_state")
REGIME_STUDY_BANDS: tuple[str, ...] = (*tuple(REGIME_BANDS.keys()), "pooled")
REGIME_ASSET_SCOPES: tuple[str, ...] = (
    "single_asset",
    "asset_family",
    "universe",
    "market",
    "peer_group",
)
REGIME_PRODUCTION_CLASSIFICATIONS: tuple[str, ...] = (
    "production",
    "staged",
    "scaffold",
    "sandbox",
    "diagnostics_only",
)

MARKET_STATE_STUDY_AXES: tuple[str, ...] = (
    "market",
    "breadth",
    "dispersion",
    "correlation",
    "market_vol",
    "leadership",
)
RELATIVE_STATE_STUDY_AXES: tuple[str, ...] = (
    "relative",
    "beta",
    "correlation",
    "relative_strength",
    "relative_dispersion",
)
REGIME_LAYER_AXES: Mapping[str, tuple[str, ...]] = {
    "asset_state": tuple(REGIME_AXIS_ORDER),
    "market_state": MARKET_STATE_STUDY_AXES,
    "relative_state": RELATIVE_STATE_STUDY_AXES,
}
REGIME_LAYER_SCOPES: Mapping[str, tuple[str, ...]] = {
    "asset_state": ("single_asset", "asset_family", "universe"),
    "market_state": ("universe", "market"),
    "relative_state": ("single_asset", "asset_family", "universe", "peer_group"),
}


def _normalized_text(value: object, *, field_name: str) -> str:
    text = str(value).strip().lower()
    if not text:
        raise ValueError(f"Regime {field_name} must be non-empty")
    return text


def _require_member(value: str, allowed: Sequence[str], *, field_name: str) -> str:
    if value not in allowed:
        valid = ", ".join(str(item) for item in allowed)
        raise ValueError(f"Unsupported Regime {field_name} {value!r}; expected one of: {valid}")
    return value


def _as_jsonable(value: Any) -> Any:
    if hasattr(value, "as_dict"):
        return value.as_dict()
    if isinstance(value, Mapping):
        return {str(key): _as_jsonable(val) for key, val in value.items()}
    if isinstance(value, tuple):
        return [_as_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_as_jsonable(item) for item in value]
    return value


@dataclass(frozen=True)
class WindowBounds:
    start_ts: int | str | None
    end_ts: int | str | None

    def __post_init__(self) -> None:
        if self.start_ts is None or self.end_ts is None:
            return
        try:
            start = int(self.start_ts)
            end = int(self.end_ts)
        except Exception:
            return
        if start > end:
            raise ValueError("Regime window start_ts must be <= end_ts")

    def as_dict(self) -> dict[str, Any]:
        return {
            "start_ts": self.start_ts,
            "end_ts": self.end_ts,
        }


@dataclass(frozen=True)
class RegimeStudyIdentity:
    layer: str
    axis: str
    band: str
    asset_scope: str
    production_classification: str
    schema_version: int = REGIME_FOUNDATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        layer = _require_member(
            _normalized_text(self.layer, field_name="layer"),
            REGIME_LAYERS,
            field_name="layer",
        )
        axis = _normalized_text(self.axis, field_name="axis")
        band = _require_member(
            _normalized_text(self.band, field_name="band"),
            REGIME_STUDY_BANDS,
            field_name="band",
        )
        asset_scope = _require_member(
            _normalized_text(self.asset_scope, field_name="asset scope"),
            REGIME_ASSET_SCOPES,
            field_name="asset scope",
        )
        classification = _require_member(
            _normalized_text(self.production_classification, field_name="production classification"),
            REGIME_PRODUCTION_CLASSIFICATIONS,
            field_name="production classification",
        )
        _require_member(axis, REGIME_LAYER_AXES[layer], field_name=f"{layer} axis")
        _require_member(asset_scope, REGIME_LAYER_SCOPES[layer], field_name=f"{layer} asset scope")
        if classification == "production" and layer != "asset_state":
            raise ValueError("Regime production classification is currently allowed only for asset_state")
        if classification == "production" and band == "pooled":
            raise ValueError("Regime production classification requires a concrete micro/meso/macro band")
        object.__setattr__(self, "layer", layer)
        object.__setattr__(self, "axis", axis)
        object.__setattr__(self, "band", band)
        object.__setattr__(self, "asset_scope", asset_scope)
        object.__setattr__(self, "production_classification", classification)

    @property
    def not_production(self) -> bool:
        return self.production_classification != "production"

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "layer": self.layer,
            "axis": self.axis,
            "band": self.band,
            "asset_scope": self.asset_scope,
            "production_classification": self.production_classification,
            "not_production": bool(self.not_production),
        }

    def to_json(self, **kwargs: Any) -> str:
        return json.dumps(self.as_dict(), **kwargs)


@dataclass(frozen=True)
class SourceArtifactLineage:
    artifact_kind: str
    artifact_path: str
    schema_version: int | None = None
    content_hash: str | None = None
    produced_by: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not str(self.artifact_kind).strip():
            raise ValueError("Regime source artifact lineage artifact_kind must be non-empty")
        if not str(self.artifact_path).strip():
            raise ValueError("Regime source artifact lineage artifact_path must be non-empty")

    def as_dict(self) -> dict[str, Any]:
        return {
            "artifact_kind": str(self.artifact_kind),
            "artifact_path": str(self.artifact_path),
            "schema_version": self.schema_version,
            "content_hash": self.content_hash,
            "produced_by": self.produced_by,
            "metadata": _as_jsonable(dict(self.metadata)),
        }


@dataclass(frozen=True)
class FeatureMatrixShapeMetadata:
    row_count: int
    feature_count: int
    feature_columns: tuple[str, ...] = ()
    index_columns: tuple[str, ...] = ("ts",)

    def __post_init__(self) -> None:
        if int(self.row_count) < 0:
            raise ValueError("Regime feature matrix row_count must be non-negative")
        if int(self.feature_count) < 0:
            raise ValueError("Regime feature matrix feature_count must be non-negative")
        if self.feature_columns and int(self.feature_count) != len(self.feature_columns):
            raise ValueError("Regime feature_count must match feature_columns when feature_columns are supplied")
        object.__setattr__(self, "row_count", int(self.row_count))
        object.__setattr__(self, "feature_count", int(self.feature_count))
        object.__setattr__(self, "feature_columns", tuple(str(col) for col in self.feature_columns))
        object.__setattr__(self, "index_columns", tuple(str(col) for col in self.index_columns))

    def as_dict(self) -> dict[str, Any]:
        return {
            "row_count": int(self.row_count),
            "feature_count": int(self.feature_count),
            "feature_columns": list(self.feature_columns),
            "index_columns": list(self.index_columns),
        }


@dataclass(frozen=True)
class MissingnessPolicy:
    policy: str
    required_columns: tuple[str, ...] = ()
    max_null_fraction: float | None = None
    imputation: str | None = None
    fail_closed: bool = True

    def __post_init__(self) -> None:
        if not str(self.policy).strip():
            raise ValueError("Regime missingness policy must be non-empty")
        if self.max_null_fraction is not None:
            fraction = float(self.max_null_fraction)
            if fraction < 0.0 or fraction > 1.0:
                raise ValueError("Regime max_null_fraction must be between 0 and 1")
            object.__setattr__(self, "max_null_fraction", fraction)
        object.__setattr__(self, "required_columns", tuple(str(col) for col in self.required_columns))

    def as_dict(self) -> dict[str, Any]:
        return {
            "policy": str(self.policy),
            "required_columns": list(self.required_columns),
            "max_null_fraction": self.max_null_fraction,
            "imputation": self.imputation,
            "fail_closed": bool(self.fail_closed),
        }


@dataclass(frozen=True)
class TrainValidationSplitMetadata:
    train_window: WindowBounds
    validation_window: WindowBounds | None = None
    walk_forward_splits: tuple[Mapping[str, Any], ...] = ()
    split_policy: str = "explicit_window"

    def __post_init__(self) -> None:
        if not str(self.split_policy).strip():
            raise ValueError("Regime split_policy must be non-empty")
        object.__setattr__(self, "walk_forward_splits", tuple(dict(split) for split in self.walk_forward_splits))

    def as_dict(self) -> dict[str, Any]:
        return {
            "split_policy": str(self.split_policy),
            "train_window": self.train_window.as_dict(),
            "validation_window": None if self.validation_window is None else self.validation_window.as_dict(),
            "walk_forward_splits": [_as_jsonable(dict(split)) for split in self.walk_forward_splits],
        }


@dataclass(frozen=True)
class RegimeDatasetWindowContract:
    identity: RegimeStudyIdentity
    interval_minutes: int | None
    horizon_minutes: int | None
    window: WindowBounds
    source_lineage: tuple[SourceArtifactLineage, ...]
    feature_matrix_shape: FeatureMatrixShapeMetadata
    missingness_policy: MissingnessPolicy
    split_metadata: TrainValidationSplitMetadata
    asset: str | None = None
    assets: tuple[str, ...] = ()
    universe: str | None = None
    benchmark: str | None = None
    peer_assets: tuple[str, ...] = ()
    schema_version: int = REGIME_FOUNDATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.interval_minutes is not None and int(self.interval_minutes) <= 0:
            raise ValueError("Regime dataset interval_minutes must be positive when supplied")
        if self.horizon_minutes is not None and int(self.horizon_minutes) <= 0:
            raise ValueError("Regime dataset horizon_minutes must be positive when supplied")
        assets = tuple(str(asset).strip() for asset in self.assets if str(asset).strip())
        peer_assets = tuple(str(asset).strip() for asset in self.peer_assets if str(asset).strip())
        asset = None if self.asset is None else str(self.asset).strip()
        universe = None if self.universe is None else str(self.universe).strip()
        benchmark = None if self.benchmark is None else str(self.benchmark).strip()
        if self.identity.asset_scope == "single_asset" and not (asset or len(assets) == 1):
            raise ValueError("Regime single_asset dataset scope requires asset or exactly one assets entry")
        if self.identity.asset_scope == "asset_family" and not assets:
            raise ValueError("Regime asset_family dataset scope requires assets membership")
        if self.identity.asset_scope in {"universe", "market"} and not universe:
            raise ValueError(f"Regime {self.identity.asset_scope} dataset scope requires universe")
        if self.identity.asset_scope == "peer_group":
            if not asset:
                raise ValueError("Regime peer_group dataset scope requires asset")
            if not peer_assets:
                raise ValueError("Regime peer_group dataset scope requires peer_assets")
        if not self.source_lineage:
            raise ValueError("Regime dataset window requires at least one source artifact lineage entry")
        object.__setattr__(self, "interval_minutes", None if self.interval_minutes is None else int(self.interval_minutes))
        object.__setattr__(self, "horizon_minutes", None if self.horizon_minutes is None else int(self.horizon_minutes))
        object.__setattr__(self, "asset", asset or None)
        object.__setattr__(self, "assets", assets)
        object.__setattr__(self, "universe", universe or None)
        object.__setattr__(self, "benchmark", benchmark or None)
        object.__setattr__(self, "peer_assets", peer_assets)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "identity": self.identity.as_dict(),
            "asset": self.asset,
            "assets": list(self.assets),
            "asset_count": int(len(self.assets)),
            "universe": self.universe,
            "benchmark": self.benchmark,
            "peer_assets": list(self.peer_assets),
            "peer_asset_count": int(len(self.peer_assets)),
            "interval_minutes": self.interval_minutes,
            "horizon_minutes": self.horizon_minutes,
            "band": self.identity.band,
            "window": self.window.as_dict(),
            "source_lineage": [lineage.as_dict() for lineage in self.source_lineage],
            "feature_matrix_shape": self.feature_matrix_shape.as_dict(),
            "missingness_policy": self.missingness_policy.as_dict(),
            "split_metadata": self.split_metadata.as_dict(),
        }

    def to_json(self, **kwargs: Any) -> str:
        return json.dumps(self.as_dict(), **kwargs)


@dataclass(frozen=True)
class RegimeTrialResultContract:
    identity: RegimeStudyIdentity
    trial_id: str
    clusterer_family: str
    preprocessing_family: str
    feature_family: str
    feature_subset: tuple[str, ...]
    assignment_policy: str
    hyperparameters: Mapping[str, Any]
    random_seed: int | None
    fit_window: WindowBounds
    score_window: WindowBounds
    artifact_paths: Mapping[str, str]
    status: str
    failure_reason: str | None = None
    schema_version: int = REGIME_FOUNDATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for field_name in (
            "trial_id",
            "clusterer_family",
            "preprocessing_family",
            "feature_family",
            "assignment_policy",
            "status",
        ):
            if not str(getattr(self, field_name)).strip():
                raise ValueError(f"Regime trial {field_name} must be non-empty")
        status = _normalized_text(self.status, field_name="trial status")
        _require_member(status, ("ok", "failed", "skipped", "blocked", "diagnostics_only"), field_name="trial status")
        failure_reason = None if self.failure_reason is None else str(self.failure_reason).strip()
        if status in {"failed", "blocked"} and not failure_reason:
            raise ValueError("Regime failed or blocked trial result requires failure_reason")
        if status == "ok" and failure_reason:
            raise ValueError("Regime ok trial result cannot carry failure_reason")
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "failure_reason", failure_reason or None)
        object.__setattr__(self, "feature_subset", tuple(str(col) for col in self.feature_subset))
        object.__setattr__(self, "artifact_paths", {str(key): str(value) for key, value in self.artifact_paths.items()})
        if self.random_seed is not None:
            object.__setattr__(self, "random_seed", int(self.random_seed))

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "identity": self.identity.as_dict(),
            "trial_id": str(self.trial_id),
            "clusterer_family": str(self.clusterer_family),
            "preprocessing_family": str(self.preprocessing_family),
            "feature_family": str(self.feature_family),
            "feature_subset": list(self.feature_subset),
            "assignment_policy": str(self.assignment_policy),
            "hyperparameters": _as_jsonable(dict(self.hyperparameters)),
            "random_seed": self.random_seed,
            "fit_window": self.fit_window.as_dict(),
            "score_window": self.score_window.as_dict(),
            "artifact_paths": dict(self.artifact_paths),
            "status": self.status,
            "failure_reason": self.failure_reason,
        }

    def to_json(self, **kwargs: Any) -> str:
        return json.dumps(self.as_dict(), **kwargs)


@dataclass(frozen=True)
class ScoreMetricFamily:
    metrics: Mapping[str, Any] = field(default_factory=dict)
    status: str = "not_reported"

    def __post_init__(self) -> None:
        if not str(self.status).strip():
            raise ValueError("Regime score metric family status must be non-empty")

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": str(self.status),
            "metrics": _as_jsonable(dict(self.metrics)),
        }


@dataclass(frozen=True)
class RegimeScoreboardContract:
    identity: RegimeStudyIdentity
    trial_id: str
    internal_validity: ScoreMetricFamily
    stability: ScoreMetricFamily
    economic_separability: ScoreMetricFamily
    coverage_degeneracy: ScoreMetricFamily
    engineering_runtime: ScoreMetricFamily
    schema_version: int = REGIME_FOUNDATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not str(self.trial_id).strip():
            raise ValueError("Regime scoreboard trial_id must be non-empty")

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "identity": self.identity.as_dict(),
            "trial_id": str(self.trial_id),
            "metric_families": {
                "internal_validity": self.internal_validity.as_dict(),
                "stability": self.stability.as_dict(),
                "economic_separability": self.economic_separability.as_dict(),
                "coverage_degeneracy": self.coverage_degeneracy.as_dict(),
                "engineering_runtime": self.engineering_runtime.as_dict(),
            },
        }

    def to_json(self, **kwargs: Any) -> str:
        return json.dumps(self.as_dict(), **kwargs)


@dataclass(frozen=True)
class ArtifactReference:
    name: str
    path: str | None = None
    artifact_kind: str | None = None
    reason: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not str(self.name).strip():
            raise ValueError("Regime artifact reference name must be non-empty")

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": str(self.name),
            "path": None if self.path is None else str(self.path),
            "artifact_kind": None if self.artifact_kind is None else str(self.artifact_kind),
            "reason": None if self.reason is None else str(self.reason),
            "metadata": _as_jsonable(dict(self.metadata)),
        }


@dataclass(frozen=True)
class RegimeArtifactManifestContract:
    identity: RegimeStudyIdentity
    write_kind: str
    source_lineage: tuple[SourceArtifactLineage, ...]
    created_artifacts: tuple[ArtifactReference, ...] = ()
    blocked_artifacts: tuple[ArtifactReference, ...] = ()
    disabled_artifacts: tuple[ArtifactReference, ...] = ()
    production_outputs_written: bool = False
    status: str = "ok"
    schema_version: int = REGIME_FOUNDATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not str(self.write_kind).strip():
            raise ValueError("Regime artifact manifest write_kind must be non-empty")
        if not self.source_lineage:
            raise ValueError("Regime artifact manifest requires source lineage")
        if self.identity.not_production and bool(self.production_outputs_written):
            raise ValueError("Regime non-production artifact manifest cannot claim production_outputs_written")
        if not str(self.status).strip():
            raise ValueError("Regime artifact manifest status must be non-empty")

    @property
    def not_production_flags(self) -> dict[str, Any]:
        not_production = bool(self.identity.not_production)
        return {
            "not_production": not_production,
            "production_writes_enabled": not not_production,
            "production_outputs_written": bool(self.production_outputs_written),
            "classification": self.identity.production_classification,
            "reason": None if not not_production else f"classification={self.identity.production_classification}",
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "layer": self.identity.layer,
            "axis": self.identity.axis,
            "band": self.identity.band,
            "asset_scope": self.identity.asset_scope,
            "write_kind": str(self.write_kind),
            "production_classification": self.identity.production_classification,
            "status": str(self.status),
            "identity": self.identity.as_dict(),
            "source_lineage": [lineage.as_dict() for lineage in self.source_lineage],
            "created_artifacts": [artifact.as_dict() for artifact in self.created_artifacts],
            "blocked_artifacts": [artifact.as_dict() for artifact in self.blocked_artifacts],
            "disabled_artifacts": [artifact.as_dict() for artifact in self.disabled_artifacts],
            "not_production_flags": self.not_production_flags,
        }

    def to_json(self, **kwargs: Any) -> str:
        return json.dumps(self.as_dict(), **kwargs)


def classify_regime_study(
    *,
    layer: str,
    axis: str,
    band: str,
    asset_scope: str,
    production_classification: str,
) -> RegimeStudyIdentity:
    return RegimeStudyIdentity(
        layer=layer,
        axis=axis,
        band=band,
        asset_scope=asset_scope,
        production_classification=production_classification,
    )


__all__ = [
    "ArtifactReference",
    "FeatureMatrixShapeMetadata",
    "MARKET_STATE_STUDY_AXES",
    "MissingnessPolicy",
    "REGIME_ASSET_SCOPES",
    "REGIME_FOUNDATION_SCHEMA_VERSION",
    "REGIME_LAYERS",
    "REGIME_LAYER_AXES",
    "REGIME_LAYER_SCOPES",
    "REGIME_PRODUCTION_CLASSIFICATIONS",
    "REGIME_STUDY_BANDS",
    "RELATIVE_STATE_STUDY_AXES",
    "RegimeArtifactManifestContract",
    "RegimeDatasetWindowContract",
    "RegimeScoreboardContract",
    "RegimeStudyIdentity",
    "RegimeTrialResultContract",
    "ScoreMetricFamily",
    "SourceArtifactLineage",
    "TrainValidationSplitMetadata",
    "WindowBounds",
    "classify_regime_study",
]
