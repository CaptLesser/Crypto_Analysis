"""Legacy study-runner compatibility surface.

Canonical bounded foundation execution lives in
``src.regimes.studies.manifest`` and ``src.regimes.studies.single_trial``. This
module remains importable for older diagnostics and drift checks.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from src.forecasting.common.io_atomic import atomic_replace, sibling_temp_path
from src.regimes.core.artifacts import read_json, safe_path_part, write_json
from src.regimes.core.clusterer_adapters import (
    FIT_STATUS_FITTED,
    REGIME_CLUSTERER_ADAPTER_SCHEMA_VERSION,
    build_regime_clusterer_adapter,
    clusterer_adapter_registry,
)
from src.regimes.core.feature_preprocessing import (
    REGIME_FEATURE_PREPROCESSING_SCHEMA_VERSION,
    default_feature_pool_registry,
    fit_regime_preprocessor,
    preprocessing_registry,
    transform_regime_preprocessor,
)
from src.regimes.core.foundation_contracts import (
    REGIME_LAYER_AXES,
    REGIME_LAYER_SCOPES,
    REGIME_LAYERS,
    REGIME_STUDY_BANDS,
    RegimeStudyIdentity,
)
from src.regimes.core.pathway_artifacts import require_pathway_diagnostics_root
from src.regimes.core.scoring import REGIME_SCORING_SCHEMA_VERSION, RegimeTrialScoreInput, score_regime_trial
from src.regimes.core.splits import split_train_validation_frame


REGIME_STUDY_MANIFEST_SCHEMA_VERSION = 1
REGIME_SEARCH_SPACE_SCHEMA_VERSION = 1
REGIME_SINGLE_TRIAL_RUN_SCHEMA_VERSION = 1
REGIME_STUDY_NON_PRODUCTION_CLASSIFICATIONS: tuple[str, ...] = (
    "sandbox",
    "staged",
    "scaffold",
    "diagnostics_only",
)
REGIME_SINGLE_TRIAL_ARTIFACT_KIND = "regime_single_trial_run"
REGIME_SEARCH_SPACE_ARTIFACT_KIND = "regime_search_space"


def _safe_float(value: object) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def _jsonable(value: Any) -> Any:
    if hasattr(value, "as_dict"):
        return value.as_dict()
    if hasattr(value, "to_metadata"):
        return value.to_metadata()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(val) for key, val in value.items()}
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, set):
        return sorted(_jsonable(item) for item in value)
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return _safe_float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.datetime64):
        return pd.Timestamp(value).isoformat()
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    if isinstance(value, BaseException):
        return str(value)
    if value is pd.NA:
        return None
    if isinstance(value, float):
        return _safe_float(value)
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    return str(value)


def _normalize_token(value: object, *, field_name: str) -> str:
    text = str(value).strip().lower()
    if not text:
        raise ValueError(f"Regime study {field_name} must be non-empty")
    return text


def _require_members(values: Sequence[object], allowed: Sequence[str], *, field_name: str) -> tuple[str, ...]:
    normalized = tuple(_normalize_token(value, field_name=field_name) for value in values)
    if not normalized:
        raise ValueError(f"Regime study {field_name} must include at least one value")
    invalid = [value for value in normalized if value not in allowed]
    if invalid:
        valid = ", ".join(str(item) for item in allowed)
        raise ValueError(f"Unsupported Regime study {field_name} {invalid[0]!r}; expected one of: {valid}")
    return tuple(dict.fromkeys(normalized))


def _require_nonempty_mapping(value: Mapping[str, Any], *, field_name: str) -> dict[str, Any]:
    out = dict(value)
    if not out:
        raise ValueError(f"Regime study {field_name} must be non-empty")
    return out


@dataclass(frozen=True)
class RegimeStudyManifest:
    study_id: str
    layer: str
    axis: str
    band: str
    asset_scope: str
    feature_families_allowed: tuple[str, ...]
    preprocessing_families_allowed: tuple[str, ...]
    clusterer_families_allowed: tuple[str, ...]
    hyperparameter_search_spaces: Mapping[str, Any]
    split_policy: Mapping[str, Any]
    forward_target_horizons: tuple[int, ...]
    trial_budget: Mapping[str, Any]
    random_seed_policy: Mapping[str, Any]
    runtime_profile_name: str
    diagnostics_output_root: Path
    production_classification: str
    asset: str | None = None
    assets: tuple[str, ...] = ()
    universe: str | None = None
    benchmark: str | None = None
    production_write_prohibited: bool = True
    allow_production_writes: bool = False
    production_outputs_written: bool = False
    schema_version: int = REGIME_STUDY_MANIFEST_SCHEMA_VERSION

    def __post_init__(self) -> None:
        study_id = safe_path_part(str(self.study_id).strip(), context="Regime study id")
        layer = _require_members((self.layer,), REGIME_LAYERS, field_name="layer")[0]
        axis = _require_members((self.axis,), REGIME_LAYER_AXES[layer], field_name=f"{layer} axis")[0]
        band = _require_members((self.band,), REGIME_STUDY_BANDS, field_name="band")[0]
        asset_scope = _require_members((self.asset_scope,), REGIME_LAYER_SCOPES[layer], field_name=f"{layer} asset scope")[0]
        classification = _require_members(
            (self.production_classification,),
            REGIME_STUDY_NON_PRODUCTION_CLASSIFICATIONS,
            field_name="production classification",
        )[0]
        if not bool(self.production_write_prohibited):
            raise ValueError("Regime study manifests must prohibit production writes")
        if bool(self.allow_production_writes):
            raise ValueError("Regime study manifests cannot allow production writes")
        if bool(self.production_outputs_written):
            raise ValueError("Regime study manifests cannot claim production_outputs_written")
        RegimeStudyIdentity(
            layer=layer,
            axis=axis,
            band=band,
            asset_scope=asset_scope,
            production_classification=classification,
        )
        feature_registry = default_feature_pool_registry()
        feature_families = _require_members(
            self.feature_families_allowed,
            tuple(feature_registry),
            field_name="feature families allowed",
        )
        for family in feature_families:
            contract = feature_registry[family]
            if layer not in contract.layer_compatibility:
                raise ValueError(f"Feature family {family!r} is not compatible with layer {layer!r}")
            if axis not in contract.axis_compatibility:
                raise ValueError(f"Feature family {family!r} is not compatible with axis {axis!r}")
            if band not in contract.band_compatibility:
                raise ValueError(f"Feature family {family!r} is not compatible with band {band!r}")
        preprocessing_families = _require_members(
            self.preprocessing_families_allowed,
            tuple(preprocessing_registry()),
            field_name="preprocessing families allowed",
        )
        clusterer_families = _require_members(
            self.clusterer_families_allowed,
            tuple(clusterer_adapter_registry()),
            field_name="clusterer families allowed",
        )
        horizons = tuple(int(horizon) for horizon in self.forward_target_horizons)
        if any(horizon <= 0 for horizon in horizons):
            raise ValueError("Regime forward target horizons must be positive")
        budget = _require_nonempty_mapping(self.trial_budget, field_name="trial budget")
        if int(budget.get("max_trials", 1)) < 1:
            raise ValueError("Regime trial_budget.max_trials must be positive")
        seed_policy = _require_nonempty_mapping(self.random_seed_policy, field_name="random seed policy")
        if "base_seed" not in seed_policy:
            raise ValueError("Regime random_seed_policy requires base_seed")
        split_policy = _require_nonempty_mapping(self.split_policy, field_name="split policy")
        if not (
            "train_row_count" in split_policy
            or "train_fraction" in split_policy
            or {"train_start_ts", "train_end_ts"}.issubset(split_policy)
        ):
            raise ValueError("Regime split_policy requires train_row_count, train_fraction, or explicit train timestamps")
        search_spaces = dict(self.hyperparameter_search_spaces)
        if "clusterer_hyperparameters_by_family" not in search_spaces:
            raise ValueError("Regime hyperparameter_search_spaces requires clusterer_hyperparameters_by_family")
        runtime_profile = str(self.runtime_profile_name).strip()
        if not runtime_profile:
            raise ValueError("Regime runtime_profile_name must be non-empty")
        root = Path(self.diagnostics_output_root)
        object.__setattr__(self, "study_id", study_id)
        object.__setattr__(self, "layer", layer)
        object.__setattr__(self, "axis", axis)
        object.__setattr__(self, "band", band)
        object.__setattr__(self, "asset_scope", asset_scope)
        object.__setattr__(self, "feature_families_allowed", feature_families)
        object.__setattr__(self, "preprocessing_families_allowed", preprocessing_families)
        object.__setattr__(self, "clusterer_families_allowed", clusterer_families)
        object.__setattr__(self, "hyperparameter_search_spaces", search_spaces)
        object.__setattr__(self, "split_policy", split_policy)
        object.__setattr__(self, "forward_target_horizons", horizons)
        object.__setattr__(self, "trial_budget", budget)
        object.__setattr__(self, "random_seed_policy", seed_policy)
        object.__setattr__(self, "runtime_profile_name", runtime_profile)
        object.__setattr__(self, "diagnostics_output_root", root)
        object.__setattr__(self, "production_classification", classification)
        object.__setattr__(self, "asset", None if self.asset is None else str(self.asset).strip() or None)
        object.__setattr__(self, "assets", tuple(str(asset).strip() for asset in self.assets if str(asset).strip()))
        object.__setattr__(self, "universe", None if self.universe is None else str(self.universe).strip() or None)
        object.__setattr__(self, "benchmark", None if self.benchmark is None else str(self.benchmark).strip() or None)

    @property
    def scope(self) -> dict[str, Any]:
        return {
            "asset_scope": self.asset_scope,
            "asset": self.asset,
            "assets": list(self.assets),
            "asset_count": int(len(self.assets)),
            "universe": self.universe,
            "benchmark": self.benchmark,
        }

    @property
    def prohibition_flags(self) -> dict[str, bool]:
        return {
            "production_write_prohibited": bool(self.production_write_prohibited),
            "allow_production_writes": bool(self.allow_production_writes),
            "production_outputs_written": bool(self.production_outputs_written),
            "not_production": True,
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "study_id": self.study_id,
            "layer": self.layer,
            "axis": self.axis,
            "band": self.band,
            "asset_scope": self.asset_scope,
            "scope": self.scope,
            "feature_families_allowed": list(self.feature_families_allowed),
            "preprocessing_families_allowed": list(self.preprocessing_families_allowed),
            "clusterer_families_allowed": list(self.clusterer_families_allowed),
            "hyperparameter_search_spaces": _jsonable(dict(self.hyperparameter_search_spaces)),
            "split_policy": _jsonable(dict(self.split_policy)),
            "forward_target_horizons": list(self.forward_target_horizons),
            "trial_budget": _jsonable(dict(self.trial_budget)),
            "random_seed_policy": _jsonable(dict(self.random_seed_policy)),
            "runtime_profile_name": self.runtime_profile_name,
            "diagnostics_output_root": str(self.diagnostics_output_root),
            "production_classification": self.production_classification,
            "prohibition_flags": self.prohibition_flags,
        }

    def to_json(self, **kwargs: Any) -> str:
        return json.dumps(self.as_dict(), **kwargs)


def parse_regime_study_manifest(payload: Mapping[str, Any]) -> RegimeStudyManifest:
    return RegimeStudyManifest(
        study_id=str(payload.get("study_id", "")),
        layer=str(payload.get("layer", "")),
        axis=str(payload.get("axis", "")),
        band=str(payload.get("band", "")),
        asset_scope=str(payload.get("asset_scope", payload.get("scope", {}).get("asset_scope", ""))),
        feature_families_allowed=tuple(payload.get("feature_families_allowed", ())),
        preprocessing_families_allowed=tuple(payload.get("preprocessing_families_allowed", ())),
        clusterer_families_allowed=tuple(payload.get("clusterer_families_allowed", ())),
        hyperparameter_search_spaces=dict(payload.get("hyperparameter_search_spaces", {})),
        split_policy=dict(payload.get("split_policy", {})),
        forward_target_horizons=tuple(payload.get("forward_target_horizons", ())),
        trial_budget=dict(payload.get("trial_budget", {})),
        random_seed_policy=dict(payload.get("random_seed_policy", {})),
        runtime_profile_name=str(payload.get("runtime_profile_name", "")),
        diagnostics_output_root=Path(str(payload.get("diagnostics_output_root", ""))),
        production_classification=str(payload.get("production_classification", "")),
        asset=payload.get("asset") or payload.get("scope", {}).get("asset"),
        assets=tuple(payload.get("assets", payload.get("scope", {}).get("assets", ()))),
        universe=payload.get("universe") or payload.get("scope", {}).get("universe"),
        benchmark=payload.get("benchmark") or payload.get("scope", {}).get("benchmark"),
        production_write_prohibited=bool(
            payload.get("production_write_prohibited", payload.get("prohibition_flags", {}).get("production_write_prohibited", True))
        ),
        allow_production_writes=bool(
            payload.get("allow_production_writes", payload.get("prohibition_flags", {}).get("allow_production_writes", False))
        ),
        production_outputs_written=bool(
            payload.get("production_outputs_written", payload.get("prohibition_flags", {}).get("production_outputs_written", False))
        ),
        schema_version=int(payload.get("schema_version", REGIME_STUDY_MANIFEST_SCHEMA_VERSION)),
    )


def load_regime_study_manifest(path: Path | str) -> RegimeStudyManifest:
    payload = read_json(Path(path))
    if not payload:
        raise ValueError(f"Regime study manifest is empty or missing: {path}")
    return parse_regime_study_manifest(payload)


def build_regime_search_space(manifest: RegimeStudyManifest) -> dict[str, Any]:
    clusterer_registry = clusterer_adapter_registry()
    preprocessor_registry = preprocessing_registry()
    feature_registry = default_feature_pool_registry()
    manifest_spaces = dict(manifest.hyperparameter_search_spaces)
    clusterer_overrides = dict(manifest_spaces.get("clusterer_hyperparameters_by_family", {}))
    clusterer_spaces: dict[str, Any] = {}
    for family in manifest.clusterer_families_allowed:
        spec = clusterer_registry[family]
        clusterer_spaces[family] = {
            "family_name": family,
            "assignment_policy": spec.assignment_policy,
            "inductive_classification": spec.inductive_classification,
            "dependency_available": bool(spec.dependency_available),
            "tier": spec.tier,
            "default_hyperparameters": _jsonable(dict(spec.default_hyperparameters)),
            "hyperparameter_schema": _jsonable(dict(clusterer_overrides.get(family, spec.hyperparameter_schema))),
            "search_space_hook": spec.search_space_hook,
        }
    return {
        "schema_version": REGIME_SEARCH_SPACE_SCHEMA_VERSION,
        "artifact_kind": REGIME_SEARCH_SPACE_ARTIFACT_KIND,
        "study_id": manifest.study_id,
        "dimensions": {
            "feature_family": {
                "type": "categorical",
                "choices": list(manifest.feature_families_allowed),
                "metadata_by_choice": {
                    family: {
                        "implementation_status": feature_registry[family].implementation_status,
                        "source_columns": list(feature_registry[family].source_columns),
                    }
                    for family in manifest.feature_families_allowed
                },
            },
            "preprocessing_family": {
                "type": "categorical",
                "choices": list(manifest.preprocessing_families_allowed),
                "metadata_by_choice": {
                    family: preprocessor_registry[family].as_dict() for family in manifest.preprocessing_families_allowed
                },
            },
            "clusterer_family": {
                "type": "categorical",
                "choices": list(manifest.clusterer_families_allowed),
                "metadata_by_choice": {
                    family: clusterer_registry[family].as_dict() for family in manifest.clusterer_families_allowed
                },
            },
        },
        "conditional_spaces": {
            "clusterer_hyperparameters_by_family": clusterer_spaces,
        },
        "optuna_future_adapter": {
            "status": "stub_ready",
            "suggestion_order": ["feature_family", "preprocessing_family", "clusterer_family", "clusterer_hyperparameters"],
            "broad_optimization_enabled": False,
        },
    }


def _target_columns_for_horizons(frame: pd.DataFrame, horizons: Sequence[int]) -> list[str]:
    columns: list[str] = []
    generic = ("future_log_return", "future_return", "forward_return")
    for column in generic:
        if column in frame.columns and column not in columns:
            columns.append(column)
    for horizon in horizons:
        for column in (
            f"future_log_return_{int(horizon)}m",
            f"forward_return_{int(horizon)}m",
            f"future_realized_volatility_{int(horizon)}m",
            f"future_max_drawdown_{int(horizon)}m",
        ):
            if column in frame.columns and column not in columns:
                columns.append(column)
    for token in ("return", "vol", "drawdown"):
        for column in frame.columns:
            lowered = str(column).lower()
            if token in lowered and column not in columns:
                columns.append(str(column))
    return columns


def _split_frame(frame: pd.DataFrame, split_policy: Mapping[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    split = split_train_validation_frame(frame, split_policy)
    return split.train, split.validation, split.metadata


def _result_dir(manifest: RegimeStudyManifest, trial_id: str) -> Path:
    return Path(manifest.diagnostics_output_root) / "regime_studies" / safe_path_part(manifest.study_id) / safe_path_part(trial_id)


def _write_markdown(path: Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = sibling_temp_path(path)
    try:
        tmp.write_text(text, encoding="utf-8")
        atomic_replace(tmp, path)
    finally:
        try:
            if tmp.exists():
                tmp.unlink()
        except Exception:
            pass


def _markdown_summary(payload: Mapping[str, Any]) -> str:
    manifest = payload["manifest"]
    trial = payload["trial"]
    fit = trial["cluster_fit"]
    scoreboard = trial["scoreboard"]
    coverage = scoreboard["metric_families"]["coverage_degeneracy"]["metrics"]
    lines = [
        f"# Regime Single Trial {trial['trial_id']}",
        "",
        f"- study_id: {manifest['study_id']}",
        f"- layer/axis/band: {manifest['layer']} / {manifest['axis']} / {manifest['band']}",
        f"- feature_family: {trial['feature_family']}",
        f"- preprocessing_family: {trial['preprocessing_family']}",
        f"- clusterer_family: {trial['clusterer_family']}",
        f"- status: {trial['status']}",
        f"- fit_status: {fit['status']}",
        f"- effective_state_count: {coverage.get('effective_state_count')}",
        f"- noise_share: {coverage.get('noise_share')}",
        f"- production_outputs_written: {manifest['prohibition_flags']['production_outputs_written']}",
        "",
        "Artifacts are diagnostics-only foundation outputs.",
        "",
    ]
    return "\n".join(lines)


@dataclass(frozen=True)
class RegimeSingleTrialRunResult:
    payload: Mapping[str, Any]
    artifact_paths: Mapping[str, str]
    schema_version: int = REGIME_SINGLE_TRIAL_RUN_SCHEMA_VERSION

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "payload": _jsonable(dict(self.payload)),
            "artifact_paths": dict(self.artifact_paths),
        }


def run_regime_single_trial(
    manifest: RegimeStudyManifest,
    frame: pd.DataFrame,
    *,
    trial_id: str,
    feature_family: str,
    preprocessing_family: str,
    clusterer_family: str,
    clusterer_hyperparameters: Mapping[str, Any] | None = None,
    preprocessing_params: Mapping[str, Any] | None = None,
    write_outputs: bool = True,
    project_root: Path | None = None,
) -> RegimeSingleTrialRunResult:
    started = time.monotonic()
    if not manifest.production_write_prohibited or manifest.allow_production_writes or manifest.production_outputs_written:
        raise ValueError("Regime single-trial runner refuses manifests that permit production writes")
    diagnostics_policy = require_pathway_diagnostics_root(Path(manifest.diagnostics_output_root), project_root=project_root)
    feature_family_key = _require_members(
        (feature_family,),
        manifest.feature_families_allowed,
        field_name="trial feature family",
    )[0]
    preprocessing_key = _require_members(
        (preprocessing_family,),
        manifest.preprocessing_families_allowed,
        field_name="trial preprocessing family",
    )[0]
    clusterer_key = _require_members(
        (clusterer_family,),
        manifest.clusterer_families_allowed,
        field_name="trial clusterer family",
    )[0]
    feature_contract = default_feature_pool_registry()[feature_family_key]
    if feature_contract.implementation_status != "implemented":
        raise ValueError(f"Feature family {feature_family_key!r} is not implemented for deterministic trial execution")
    train_frame, validation_frame, split_metadata = _split_frame(frame, manifest.split_policy)
    fitted_preprocessor = fit_regime_preprocessor(
        train_frame,
        feature_contract.source_columns,
        preprocess=preprocessing_key,
        preprocess_params=preprocessing_params,
        fit_window=split_metadata,
        fit_window_role="train",
    )
    validation_transform = transform_regime_preprocessor(validation_frame, fitted_preprocessor, window_role="validation")
    adapter = build_regime_clusterer_adapter(clusterer_key, **dict(clusterer_hyperparameters or {}))
    fit_result = adapter.fit(fitted_preprocessor.x)
    assignment_payload: dict[str, Any] | None = None
    validation_refit_payload: dict[str, Any] | None = None
    if validation_transform.x.shape[0] > 0:
        if adapter.spec.supports_assign:
            assignment_payload = adapter.assign(validation_transform.x).as_dict()
        elif adapter.spec.supports_refit_recluster:
            validation_refit_payload = adapter.refit_recluster(validation_transform.x).as_dict()
    forward_columns = _target_columns_for_horizons(fitted_preprocessor.clean_frame, manifest.forward_target_horizons)
    forward_frame = fitted_preprocessor.clean_frame[forward_columns].copy() if forward_columns else None
    runtime_metadata = {
        "fit_time_s": fit_result.runtime_metadata.get("fit_time_s"),
        "elapsed_s": float(time.monotonic() - started),
        "row_count": int(fitted_preprocessor.x.shape[0]),
        "feature_count": int(fitted_preprocessor.x.shape[1]),
        "status": fit_result.status,
        "failure_reason": fit_result.failure_metadata.get("error") if fit_result.status != FIT_STATUS_FITTED else None,
        "retry_count": 0,
    }
    scoreboard = score_regime_trial(
        RegimeTrialScoreInput(
            trial_id=trial_id,
            labels=fit_result.labels,
            features=fitted_preprocessor.x,
            clusterer_family=clusterer_key,
            model=fit_result.estimator,
            forward_frame=forward_frame,
            runtime_metadata=runtime_metadata,
        )
    )
    status = "ok" if fit_result.status == FIT_STATUS_FITTED else "failed"
    result_dir = _result_dir(manifest, trial_id)
    artifact_paths = {
        "json": str(result_dir / "trial_result.json"),
        "markdown": str(result_dir / "trial_summary.md"),
    }
    payload = {
        "schema_version": REGIME_SINGLE_TRIAL_RUN_SCHEMA_VERSION,
        "artifact_kind": REGIME_SINGLE_TRIAL_ARTIFACT_KIND,
        "manifest": manifest.as_dict(),
        "diagnostics_root_policy": diagnostics_policy.as_dict(),
        "search_space": build_regime_search_space(manifest),
        "trial": {
            "trial_id": safe_path_part(trial_id, context="Regime trial id"),
            "status": status,
            "feature_family": feature_family_key,
            "preprocessing_family": preprocessing_key,
            "clusterer_family": clusterer_key,
            "clusterer_hyperparameters": _jsonable(dict(clusterer_hyperparameters or {})),
            "split_metadata": split_metadata,
            "feature_contract": feature_contract.as_dict(),
            "preprocess_metadata": fitted_preprocessor.to_metadata(),
            "validation_transform_metadata": validation_transform.to_metadata(),
            "cluster_fit": fit_result.as_dict(),
            "validation_assignment": assignment_payload,
            "validation_refit_recluster": validation_refit_payload,
            "labels": _jsonable(fit_result.labels),
            "scoreboard": scoreboard,
            "status_metadata": {
                "failure_metadata": _jsonable(dict(fit_result.failure_metadata)),
                "runtime_metadata": _jsonable(runtime_metadata),
            },
        },
        "schema_versions": {
            "manifest": REGIME_STUDY_MANIFEST_SCHEMA_VERSION,
            "search_space": REGIME_SEARCH_SPACE_SCHEMA_VERSION,
            "single_trial_run": REGIME_SINGLE_TRIAL_RUN_SCHEMA_VERSION,
            "feature_preprocessing": REGIME_FEATURE_PREPROCESSING_SCHEMA_VERSION,
            "clusterer_adapter": REGIME_CLUSTERER_ADAPTER_SCHEMA_VERSION,
            "scoreboard": REGIME_SCORING_SCHEMA_VERSION,
        },
        "artifact_boundary": {
            "diagnostics_only": True,
            "production_outputs_written": False,
            "production_write_prohibited": True,
        },
    }
    if write_outputs:
        write_json(
            Path(artifact_paths["json"]),
            _jsonable(payload),
            write_kind="Regime study single-trial diagnostic",
        )
        _write_markdown(Path(artifact_paths["markdown"]), _markdown_summary(payload))
    return RegimeSingleTrialRunResult(payload=payload, artifact_paths=artifact_paths)


__all__ = [
    "REGIME_SEARCH_SPACE_ARTIFACT_KIND",
    "REGIME_SEARCH_SPACE_SCHEMA_VERSION",
    "REGIME_SINGLE_TRIAL_ARTIFACT_KIND",
    "REGIME_SINGLE_TRIAL_RUN_SCHEMA_VERSION",
    "REGIME_STUDY_MANIFEST_SCHEMA_VERSION",
    "REGIME_STUDY_NON_PRODUCTION_CLASSIFICATIONS",
    "RegimeSingleTrialRunResult",
    "RegimeStudyManifest",
    "build_regime_search_space",
    "load_regime_study_manifest",
    "parse_regime_study_manifest",
    "run_regime_single_trial",
]
