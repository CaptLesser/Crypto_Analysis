from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.regimes.core.artifacts import safe_path_part, write_json
from src.regimes.core.foundation_contracts import REGIME_STUDY_BANDS, MissingnessPolicy, SourceArtifactLineage
from src.regimes.core.pathway_artifacts import require_pathway_diagnostics_root


RELATIVE_METADATA_SCHEMA_VERSION = 1
RELATIVE_STATE_PATHWAY = "relative_state"

RELATIVE_ALIGNMENT_FRAME_ARTIFACT_KIND = "relative_alignment_frame_contract"
RELATIVE_FEATURE_FAMILY_DECLARATION_ARTIFACT_KIND = "relative_feature_family_declaration"
RELATIVE_METADATA_DIAGNOSTIC_ARTIFACT_KIND = "relative_alignment_metadata_diagnostic"
RELATIVE_METADATA_STATUS = "metadata_only_scaffold"

RELATIVE_METADATA_PRODUCTION_CLASSIFICATIONS: tuple[str, ...] = ("scaffold", "metadata_only")
RELATIVE_TIMESTAMP_ALIGNMENT_POLICIES: tuple[str, ...] = (
    "exact_timestamp_intersection",
    "calendar_intersection",
    "asof_backward_with_tolerance",
    "left_primary_asset",
    "metadata_only_unspecified",
)
RELATIVE_FEATURE_IMPLEMENTATION_STATUSES: tuple[str, ...] = (
    "declared",
    "placeholder",
    "metadata_only",
)
RELATIVE_FEATURE_LEAKAGE_POLICIES: tuple[str, ...] = (
    "train_window_only",
    "metadata_only_no_forward_targets",
    "placeholder_not_implemented",
)
RELATIVE_FEATURE_AXES: tuple[str, ...] = (
    "relative",
    "beta",
    "correlation",
    "relative_strength",
    "relative_dispersion",
)

DEFAULT_RELATIVE_REQUIRED_FUTURE_SOURCES_OR_PROBES: tuple[str, ...] = (
    "primary_asset_aligned_price_or_return_source_probe",
    "market_benchmark_source_probe",
    "explicit_peer_group_or_peer_basket_registry_contract",
    "peer_asset_aligned_price_or_return_source_probe",
    "timestamp_calendar_and_asof_tolerance_contract",
    "stale_data_detection_policy_backed_by_source_timestamps",
    "cross_sectional_membership_snapshot_for_rank_features",
    "relative_alignment_frame_builder_and_validation_report",
)


def _utc_timestamp_text(value: str | None = None) -> str:
    return str(value or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"))


def _jsonable(value: Any) -> Any:
    if hasattr(value, "as_dict"):
        return value.as_dict()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(val) for key, val in value.items()}
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, set):
        return sorted(_jsonable(item) for item in value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            return None
        return value
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return str(value)


def _normalize_token(value: object, *, field_name: str) -> str:
    text = str(value).strip().lower()
    if not text:
        raise ValueError(f"Relative-state metadata {field_name} must be non-empty")
    return text


def _normalize_text(value: object, *, field_name: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"Relative-state metadata {field_name} must be non-empty")
    return text


def _require_member(value: object, allowed: Sequence[str], *, field_name: str) -> str:
    token = _normalize_token(value, field_name=field_name)
    if token not in allowed:
        valid = ", ".join(str(item) for item in allowed)
        raise ValueError(f"Unsupported relative-state metadata {field_name} {token!r}; expected one of: {valid}")
    return token


def _unique_texts(values: Sequence[object], *, field_name: str, require_nonempty: bool = True) -> tuple[str, ...]:
    normalized = tuple(_normalize_text(value, field_name=field_name) for value in values if str(value).strip())
    unique = tuple(dict.fromkeys(normalized))
    if require_nonempty and not unique:
        raise ValueError(f"Relative-state metadata {field_name} must include at least one value")
    return unique


def _require_members(values: Sequence[object], allowed: Sequence[str], *, field_name: str) -> tuple[str, ...]:
    members = tuple(_require_member(value, allowed, field_name=field_name) for value in values)
    if not members:
        raise ValueError(f"Relative-state metadata {field_name} must include at least one value")
    return tuple(dict.fromkeys(members))


def _relative_artifact_boundary(*, write_mode: str, alignment_frame_built: bool = False) -> dict[str, Any]:
    return {
        "write_mode": write_mode,
        "metadata_only": True,
        "production_writes_enabled": False,
        "parquet_writes_enabled": False,
        "state_writes_enabled": False,
        "definition_writes_enabled": False,
        "production_outputs_written": False,
        "source_reader_enabled": False,
        "downstream_readers_enabled": False,
        "alignment_frame_built": bool(alignment_frame_built),
    }


def _validate_relative_artifact_boundary(boundary: Mapping[str, Any], *, context: str) -> dict[str, Any]:
    checked = dict(boundary)
    forbidden_true_fields = (
        "production_writes_enabled",
        "parquet_writes_enabled",
        "state_writes_enabled",
        "definition_writes_enabled",
        "production_outputs_written",
        "source_reader_enabled",
        "downstream_readers_enabled",
    )
    for field_name in forbidden_true_fields:
        if checked.get(field_name) is not False:
            raise ValueError(f"{context} cannot enable {field_name}")
    if checked.get("metadata_only") is not True:
        raise ValueError(f"{context} must declare metadata_only=true")
    return checked


def _default_missingness_policy(columns: Sequence[str]) -> MissingnessPolicy:
    return MissingnessPolicy(
        policy="relative_alignment_metadata_required_columns_fail_closed",
        required_columns=tuple(str(column) for column in columns),
        max_null_fraction=0.1,
        fail_closed=True,
    )


def _relative_metadata_lineage(path: str, produced_by: str) -> tuple[SourceArtifactLineage, ...]:
    return (
        SourceArtifactLineage(
            artifact_kind="relative_state_metadata_contract",
            artifact_path=path,
            schema_version=RELATIVE_METADATA_SCHEMA_VERSION,
            produced_by=produced_by,
            metadata={"pathway": RELATIVE_STATE_PATHWAY, "metadata_only": True},
        ),
    )


@dataclass(frozen=True)
class RelativeAlignmentFrameContract:
    primary_asset: str
    market_benchmark_source: Mapping[str, Any]
    timestamp_alignment_policy: str
    interval_minutes: int
    band: str
    required_source_columns: tuple[str, ...]
    coverage_threshold: float
    missingness_policy: MissingnessPolicy
    stale_data_policy: Mapping[str, Any]
    lookback_windows: tuple[int, ...]
    source_lineage: tuple[SourceArtifactLineage, ...]
    comparison_universe: str | None = None
    peer_group: tuple[str, ...] = ()
    production_classification: str = "metadata_only"
    alignment_frame_built: bool = False
    dry_run_scaffold: bool = False
    schema_version: int = RELATIVE_METADATA_SCHEMA_VERSION
    artifact_kind: str = RELATIVE_ALIGNMENT_FRAME_ARTIFACT_KIND

    def __post_init__(self) -> None:
        primary_asset = _normalize_text(self.primary_asset, field_name="primary asset")
        comparison_universe = None
        if self.comparison_universe is not None and str(self.comparison_universe).strip():
            comparison_universe = _normalize_text(self.comparison_universe, field_name="comparison universe")
        peer_group = _unique_texts(self.peer_group, field_name="peer group", require_nonempty=False)
        if comparison_universe is None and not peer_group:
            raise ValueError("Relative-state metadata requires comparison_universe or peer_group")
        classification = _require_member(
            self.production_classification,
            RELATIVE_METADATA_PRODUCTION_CLASSIFICATIONS,
            field_name="production classification",
        )
        if bool(self.alignment_frame_built):
            if classification != "scaffold" or not bool(self.dry_run_scaffold):
                raise ValueError(
                    "Relative alignment frame can be marked built only for explicit scaffold dry-run metadata"
                )
        if int(self.interval_minutes) <= 0:
            raise ValueError("Relative-state metadata interval_minutes must be positive")
        band = _require_member(self.band, REGIME_STUDY_BANDS, field_name="band")
        alignment_policy = _require_member(
            self.timestamp_alignment_policy,
            RELATIVE_TIMESTAMP_ALIGNMENT_POLICIES,
            field_name="timestamp alignment policy",
        )
        source_columns = _unique_texts(self.required_source_columns, field_name="required source columns")
        coverage = float(self.coverage_threshold)
        if coverage < 0.0 or coverage > 1.0:
            raise ValueError("Relative-state metadata coverage_threshold must be between 0 and 1")
        if not self.market_benchmark_source:
            raise ValueError("Relative-state metadata market_benchmark_source must be non-empty")
        if not self.stale_data_policy:
            raise ValueError("Relative-state metadata stale_data_policy must be non-empty")
        lookbacks = tuple(int(window) for window in self.lookback_windows)
        if not lookbacks:
            raise ValueError("Relative-state metadata lookback_windows must be non-empty")
        if any(window <= 0 for window in lookbacks):
            raise ValueError("Relative-state metadata lookback_windows must be positive")
        if not self.source_lineage:
            raise ValueError("Relative-state metadata source_lineage must be non-empty")

        object.__setattr__(self, "primary_asset", primary_asset)
        object.__setattr__(self, "comparison_universe", comparison_universe)
        object.__setattr__(self, "peer_group", peer_group)
        object.__setattr__(self, "production_classification", classification)
        object.__setattr__(self, "interval_minutes", int(self.interval_minutes))
        object.__setattr__(self, "band", band)
        object.__setattr__(self, "timestamp_alignment_policy", alignment_policy)
        object.__setattr__(self, "required_source_columns", source_columns)
        object.__setattr__(self, "coverage_threshold", coverage)
        object.__setattr__(self, "lookback_windows", lookbacks)
        object.__setattr__(self, "source_lineage", tuple(self.source_lineage))
        object.__setattr__(
            self,
            "market_benchmark_source",
            {str(key): _jsonable(value) for key, value in self.market_benchmark_source.items()},
        )
        object.__setattr__(
            self,
            "stale_data_policy",
            {str(key): _jsonable(value) for key, value in self.stale_data_policy.items()},
        )

    @property
    def artifact_boundary(self) -> dict[str, Any]:
        write_mode = (
            "explicit_dry_run_scaffold_alignment_metadata"
            if bool(self.alignment_frame_built)
            else "relative_alignment_contract_metadata_only"
        )
        boundary = _relative_artifact_boundary(write_mode=write_mode, alignment_frame_built=self.alignment_frame_built)
        boundary["dry_run_scaffold"] = bool(self.dry_run_scaffold)
        return boundary

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "artifact_kind": self.artifact_kind,
            "pathway": RELATIVE_STATE_PATHWAY,
            "primary_asset": self.primary_asset,
            "comparison_universe": self.comparison_universe,
            "peer_group": list(self.peer_group),
            "peer_group_count": int(len(self.peer_group)),
            "market_benchmark_source": _jsonable(dict(self.market_benchmark_source)),
            "timestamp_alignment_policy": self.timestamp_alignment_policy,
            "interval_minutes": int(self.interval_minutes),
            "band": self.band,
            "required_source_columns": list(self.required_source_columns),
            "coverage_threshold": float(self.coverage_threshold),
            "missingness_policy": self.missingness_policy.as_dict(),
            "stale_data_policy": _jsonable(dict(self.stale_data_policy)),
            "lookback_windows": list(self.lookback_windows),
            "source_lineage": [lineage.as_dict() for lineage in self.source_lineage],
            "production_classification": self.production_classification,
            "not_production": True,
            "alignment_frame_built": bool(self.alignment_frame_built),
            "dry_run_scaffold": bool(self.dry_run_scaffold),
            "artifact_boundary": self.artifact_boundary,
        }

    def to_json(self, **kwargs: Any) -> str:
        return json.dumps(self.as_dict(), **kwargs)


@dataclass(frozen=True)
class RelativeFeatureFamilyDeclaration:
    feature_family_name: str
    required_source_columns: tuple[str, ...]
    derived_feature_columns: tuple[str, ...]
    axis_compatibility: tuple[str, ...]
    lookback_windows: tuple[int, ...]
    required_input_granularity: str
    missingness_policy: MissingnessPolicy
    leakage_policy: str
    source_lineage: tuple[SourceArtifactLineage, ...]
    implementation_status: str = "declared"
    layer_compatibility: tuple[str, ...] = (RELATIVE_STATE_PATHWAY,)
    band_compatibility: tuple[str, ...] = REGIME_STUDY_BANDS
    benchmark_required: bool = True
    peer_basket_required: bool = False
    notes: tuple[str, ...] = ()
    schema_version: int = RELATIVE_METADATA_SCHEMA_VERSION
    artifact_kind: str = RELATIVE_FEATURE_FAMILY_DECLARATION_ARTIFACT_KIND

    def __post_init__(self) -> None:
        family = _normalize_token(self.feature_family_name, field_name="feature family name")
        source_columns = _unique_texts(self.required_source_columns, field_name="required source columns")
        derived_columns = _unique_texts(self.derived_feature_columns, field_name="derived feature columns")
        axes = _require_members(self.axis_compatibility, RELATIVE_FEATURE_AXES, field_name="axis compatibility")
        layers = _require_members(self.layer_compatibility, (RELATIVE_STATE_PATHWAY,), field_name="layer compatibility")
        bands = _require_members(self.band_compatibility, REGIME_STUDY_BANDS, field_name="band compatibility")
        lookbacks = tuple(int(window) for window in self.lookback_windows)
        if any(window <= 0 for window in lookbacks):
            raise ValueError("Relative feature metadata lookback_windows must be positive")
        granularity = _normalize_text(self.required_input_granularity, field_name="required input granularity")
        leakage_policy = _require_member(
            self.leakage_policy,
            RELATIVE_FEATURE_LEAKAGE_POLICIES,
            field_name="leakage policy",
        )
        status = _require_member(
            self.implementation_status,
            RELATIVE_FEATURE_IMPLEMENTATION_STATUSES,
            field_name="feature implementation status",
        )
        if not self.source_lineage:
            raise ValueError("Relative feature metadata source_lineage must be non-empty")
        object.__setattr__(self, "feature_family_name", family)
        object.__setattr__(self, "required_source_columns", source_columns)
        object.__setattr__(self, "derived_feature_columns", derived_columns)
        object.__setattr__(self, "axis_compatibility", axes)
        object.__setattr__(self, "layer_compatibility", layers)
        object.__setattr__(self, "band_compatibility", bands)
        object.__setattr__(self, "lookback_windows", lookbacks)
        object.__setattr__(self, "required_input_granularity", granularity)
        object.__setattr__(self, "leakage_policy", leakage_policy)
        object.__setattr__(self, "implementation_status", status)
        object.__setattr__(self, "source_lineage", tuple(self.source_lineage))
        object.__setattr__(self, "notes", tuple(str(note) for note in self.notes))

    @property
    def artifact_boundary(self) -> dict[str, Any]:
        boundary = _relative_artifact_boundary(write_mode="relative_feature_family_declaration_metadata_only")
        boundary["feature_builder_enabled"] = False
        return boundary

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "artifact_kind": self.artifact_kind,
            "pathway": RELATIVE_STATE_PATHWAY,
            "feature_family_name": self.feature_family_name,
            "required_source_columns": list(self.required_source_columns),
            "derived_feature_columns": list(self.derived_feature_columns),
            "layer_compatibility": list(self.layer_compatibility),
            "axis_compatibility": list(self.axis_compatibility),
            "band_compatibility": list(self.band_compatibility),
            "lookback_windows": list(self.lookback_windows),
            "required_input_granularity": self.required_input_granularity,
            "missingness_policy": self.missingness_policy.as_dict(),
            "leakage_policy": self.leakage_policy,
            "source_lineage": [lineage.as_dict() for lineage in self.source_lineage],
            "implementation_status": self.implementation_status,
            "benchmark_required": bool(self.benchmark_required),
            "peer_basket_required": bool(self.peer_basket_required),
            "notes": list(self.notes),
            "artifact_boundary": self.artifact_boundary,
        }

    def to_json(self, **kwargs: Any) -> str:
        return json.dumps(self.as_dict(), **kwargs)


@dataclass(frozen=True)
class RelativeMetadataDiagnosticManifest:
    run_id: str
    alignment_contract: RelativeAlignmentFrameContract
    feature_families: tuple[RelativeFeatureFamilyDeclaration, ...] = ()
    required_future_sources_or_probes: tuple[str, ...] = DEFAULT_RELATIVE_REQUIRED_FUTURE_SOURCES_OR_PROBES
    diagnostics_root_policy: Mapping[str, Any] = field(default_factory=dict)
    created_at_utc: str | None = None
    schema_version: int = RELATIVE_METADATA_SCHEMA_VERSION
    artifact_kind: str = RELATIVE_METADATA_DIAGNOSTIC_ARTIFACT_KIND
    status: str = RELATIVE_METADATA_STATUS

    def __post_init__(self) -> None:
        run_id = _normalize_text(self.run_id, field_name="run id")
        feature_families = tuple(self.feature_families) or tuple(default_relative_feature_family_declarations().values())
        if not feature_families:
            raise ValueError("Relative metadata diagnostic manifest requires feature families")
        future_sources = _unique_texts(
            self.required_future_sources_or_probes,
            field_name="required future sources or probes",
        )
        boundary = self.artifact_boundary
        _validate_relative_artifact_boundary(boundary, context="Relative metadata diagnostic manifest")
        for feature in feature_families:
            _validate_relative_artifact_boundary(
                feature.artifact_boundary,
                context=f"Relative feature family {feature.feature_family_name}",
            )
        _validate_relative_artifact_boundary(
            self.alignment_contract.artifact_boundary,
            context="Relative alignment frame contract",
        )
        object.__setattr__(self, "run_id", run_id)
        object.__setattr__(self, "feature_families", feature_families)
        object.__setattr__(self, "required_future_sources_or_probes", future_sources)
        object.__setattr__(self, "created_at_utc", _utc_timestamp_text(self.created_at_utc))
        object.__setattr__(
            self,
            "diagnostics_root_policy",
            {str(key): _jsonable(value) for key, value in self.diagnostics_root_policy.items()},
        )

    @property
    def artifact_boundary(self) -> dict[str, Any]:
        boundary = _relative_artifact_boundary(
            write_mode="relative_metadata_diagnostic_json_only",
            alignment_frame_built=self.alignment_contract.alignment_frame_built,
        )
        boundary["production_readers_enabled"] = False
        boundary["real_alignment_builder_enabled"] = False
        return boundary

    def as_dict(self) -> dict[str, Any]:
        alignment_built = bool(self.alignment_contract.alignment_frame_built)
        build_guard = "explicit_dry_run_scaffold" if alignment_built else "not_built_metadata_only"
        return {
            "schema_version": int(self.schema_version),
            "artifact_kind": self.artifact_kind,
            "status": self.status,
            "pathway": RELATIVE_STATE_PATHWAY,
            "run_id": self.run_id,
            "created_at_utc": self.created_at_utc,
            "primary_asset": self.alignment_contract.primary_asset,
            "band": self.alignment_contract.band,
            "production_classification": self.alignment_contract.production_classification,
            "alignment_contract": self.alignment_contract.as_dict(),
            "relative_feature_families": [family.as_dict() for family in self.feature_families],
            "diagnostic_assertions": {
                "metadata_only": True,
                "alignment_frame_was_built": alignment_built,
                "alignment_frame_build_guard": build_guard,
                "production_writes_disabled": True,
                "source_readers_disabled": True,
                "downstream_readers_disabled": True,
                "production_readers_disabled": True,
            },
            "required_future_sources_or_probes": list(self.required_future_sources_or_probes),
            "diagnostics_root_policy": _jsonable(dict(self.diagnostics_root_policy)),
            "artifact_boundary": self.artifact_boundary,
        }

    def to_json(self, **kwargs: Any) -> str:
        return json.dumps(self.as_dict(), **kwargs)


def default_relative_feature_family_declarations() -> dict[str, RelativeFeatureFamilyDeclaration]:
    bands = tuple(REGIME_STUDY_BANDS)
    lineage = _relative_metadata_lineage(
        "metadata://relative_state/default_feature_family_declarations",
        "src.regimes.core.relative_metadata.default_relative_feature_family_declarations",
    )

    def policy(columns: Sequence[str]) -> MissingnessPolicy:
        return _default_missingness_policy(columns)

    return {
        "rolling_beta_to_market": RelativeFeatureFamilyDeclaration(
            feature_family_name="rolling_beta_to_market",
            required_source_columns=("asset_return", "benchmark_return"),
            derived_feature_columns=("rolling_beta_to_benchmark",),
            layer_compatibility=(RELATIVE_STATE_PATHWAY,),
            axis_compatibility=("beta", "relative"),
            band_compatibility=bands,
            lookback_windows=(20, 60, 120),
            required_input_granularity="aligned_primary_asset_and_benchmark_returns",
            missingness_policy=policy(("asset_return", "benchmark_return")),
            leakage_policy="train_window_only",
            source_lineage=lineage,
            implementation_status="declared",
        ),
        "residual_return": RelativeFeatureFamilyDeclaration(
            feature_family_name="residual_return",
            required_source_columns=("asset_return", "benchmark_return", "rolling_beta_to_benchmark"),
            derived_feature_columns=("residual_return",),
            layer_compatibility=(RELATIVE_STATE_PATHWAY,),
            axis_compatibility=("relative", "relative_strength", "beta"),
            band_compatibility=bands,
            lookback_windows=(20, 60, 120),
            required_input_granularity="aligned_primary_asset_and_benchmark_returns",
            missingness_policy=policy(("asset_return", "benchmark_return", "rolling_beta_to_benchmark")),
            leakage_policy="train_window_only",
            source_lineage=lineage,
            implementation_status="declared",
        ),
        "residual_volatility": RelativeFeatureFamilyDeclaration(
            feature_family_name="residual_volatility",
            required_source_columns=("residual_return",),
            derived_feature_columns=("residual_realized_volatility",),
            layer_compatibility=(RELATIVE_STATE_PATHWAY,),
            axis_compatibility=("relative", "relative_dispersion"),
            band_compatibility=bands,
            lookback_windows=(20, 60, 120),
            required_input_granularity="aligned_residual_return_frame",
            missingness_policy=policy(("residual_return",)),
            leakage_policy="train_window_only",
            source_lineage=lineage,
            implementation_status="declared",
        ),
        "rolling_correlation_to_market_peer_basket": RelativeFeatureFamilyDeclaration(
            feature_family_name="rolling_correlation_to_market_peer_basket",
            required_source_columns=("asset_return", "benchmark_return", "peer_basket_return"),
            derived_feature_columns=("rolling_corr_to_benchmark", "rolling_corr_to_peer_basket"),
            layer_compatibility=(RELATIVE_STATE_PATHWAY,),
            axis_compatibility=("correlation", "relative"),
            band_compatibility=bands,
            lookback_windows=(20, 60, 120),
            required_input_granularity="aligned_primary_benchmark_and_peer_basket_returns",
            missingness_policy=policy(("asset_return", "benchmark_return", "peer_basket_return")),
            leakage_policy="train_window_only",
            source_lineage=lineage,
            implementation_status="declared",
            peer_basket_required=True,
        ),
        "partial_correlation_placeholder": RelativeFeatureFamilyDeclaration(
            feature_family_name="partial_correlation_placeholder",
            required_source_columns=("asset_return", "benchmark_return", "peer_basket_return"),
            derived_feature_columns=("partial_corr_to_benchmark_conditional_peer_basket",),
            layer_compatibility=(RELATIVE_STATE_PATHWAY,),
            axis_compatibility=("correlation",),
            band_compatibility=bands,
            lookback_windows=(60, 120),
            required_input_granularity="aligned_primary_benchmark_and_peer_basket_returns",
            missingness_policy=policy(("asset_return", "benchmark_return", "peer_basket_return")),
            leakage_policy="placeholder_not_implemented",
            source_lineage=lineage,
            implementation_status="placeholder",
            peer_basket_required=True,
            notes=("Placeholder until partial-correlation source and numerical policy are specified.",),
        ),
        "cross_sectional_percentile_rank": RelativeFeatureFamilyDeclaration(
            feature_family_name="cross_sectional_percentile_rank",
            required_source_columns=("asset_return", "peer_group_returns", "peer_group_membership_snapshot"),
            derived_feature_columns=("cross_sectional_return_percentile", "cross_sectional_vol_percentile"),
            layer_compatibility=(RELATIVE_STATE_PATHWAY,),
            axis_compatibility=("relative", "relative_strength", "relative_dispersion"),
            band_compatibility=bands,
            lookback_windows=(20, 60),
            required_input_granularity="aligned_primary_and_peer_group_cross_section",
            missingness_policy=policy(("asset_return", "peer_group_returns", "peer_group_membership_snapshot")),
            leakage_policy="train_window_only",
            source_lineage=lineage,
            implementation_status="declared",
            peer_basket_required=True,
        ),
        "distance_to_peer_cluster_embedding_placeholder": RelativeFeatureFamilyDeclaration(
            feature_family_name="distance_to_peer_cluster_embedding_placeholder",
            required_source_columns=("relative_feature_matrix", "peer_cluster_embedding"),
            derived_feature_columns=("distance_to_peer_cluster_centroid",),
            layer_compatibility=(RELATIVE_STATE_PATHWAY,),
            axis_compatibility=("relative", "relative_dispersion"),
            band_compatibility=bands,
            lookback_windows=(60, 120),
            required_input_granularity="aligned_peer_group_embedding_frame",
            missingness_policy=policy(("relative_feature_matrix", "peer_cluster_embedding")),
            leakage_policy="placeholder_not_implemented",
            source_lineage=lineage,
            implementation_status="placeholder",
            peer_basket_required=True,
            notes=("Placeholder until peer-cluster embedding contracts and fit-window discipline are specified.",),
        ),
    }


def relative_alignment_metadata_diagnostic_manifest(
    *,
    run_id: str,
    alignment_contract: RelativeAlignmentFrameContract,
    feature_families: Sequence[RelativeFeatureFamilyDeclaration] | None = None,
    required_future_sources_or_probes: Sequence[str] = DEFAULT_RELATIVE_REQUIRED_FUTURE_SOURCES_OR_PROBES,
    diagnostics_root_policy: Mapping[str, Any] | None = None,
    created_at_utc: str | None = None,
) -> RelativeMetadataDiagnosticManifest:
    return RelativeMetadataDiagnosticManifest(
        run_id=run_id,
        alignment_contract=alignment_contract,
        feature_families=tuple(feature_families or ()),
        required_future_sources_or_probes=tuple(required_future_sources_or_probes),
        diagnostics_root_policy=dict(diagnostics_root_policy or {}),
        created_at_utc=created_at_utc,
    )


def relative_alignment_metadata_diagnostic_path(
    diagnostics_root: Path,
    *,
    primary_asset: str,
    run_id: str,
    filename: str = "relative_alignment_metadata.json",
) -> Path:
    return (
        Path(diagnostics_root)
        / "relative_alignment_metadata"
        / safe_path_part(primary_asset, context="Relative alignment metadata primary asset")
        / safe_path_part(run_id, context="Relative alignment metadata run id")
        / safe_path_part(filename, context="Relative alignment metadata filename")
    )


def write_relative_alignment_metadata_diagnostic(
    alignment_contract: RelativeAlignmentFrameContract,
    *,
    diagnostics_root: Path,
    run_id: str,
    feature_families: Sequence[RelativeFeatureFamilyDeclaration] | None = None,
    required_future_sources_or_probes: Sequence[str] = DEFAULT_RELATIVE_REQUIRED_FUTURE_SOURCES_OR_PROBES,
    project_root: Path | None = None,
    env: Mapping[str, str] | None = None,
    created_at_utc: str | None = None,
) -> Path:
    root_policy = require_pathway_diagnostics_root(
        Path(diagnostics_root),
        project_root=project_root,
        env=env,
    )
    manifest = relative_alignment_metadata_diagnostic_manifest(
        run_id=run_id,
        alignment_contract=alignment_contract,
        feature_families=feature_families,
        required_future_sources_or_probes=required_future_sources_or_probes,
        diagnostics_root_policy=root_policy.as_dict(),
        created_at_utc=created_at_utc,
    )
    path = relative_alignment_metadata_diagnostic_path(
        Path(diagnostics_root),
        primary_asset=alignment_contract.primary_asset,
        run_id=run_id,
    )
    write_json(path, manifest.as_dict(), write_kind="Regime relative-state metadata diagnostic")
    return path


__all__ = [
    "DEFAULT_RELATIVE_REQUIRED_FUTURE_SOURCES_OR_PROBES",
    "RELATIVE_ALIGNMENT_FRAME_ARTIFACT_KIND",
    "RELATIVE_FEATURE_AXES",
    "RELATIVE_FEATURE_FAMILY_DECLARATION_ARTIFACT_KIND",
    "RELATIVE_FEATURE_IMPLEMENTATION_STATUSES",
    "RELATIVE_FEATURE_LEAKAGE_POLICIES",
    "RELATIVE_METADATA_DIAGNOSTIC_ARTIFACT_KIND",
    "RELATIVE_METADATA_PRODUCTION_CLASSIFICATIONS",
    "RELATIVE_METADATA_SCHEMA_VERSION",
    "RELATIVE_METADATA_STATUS",
    "RELATIVE_STATE_PATHWAY",
    "RELATIVE_TIMESTAMP_ALIGNMENT_POLICIES",
    "RelativeAlignmentFrameContract",
    "RelativeFeatureFamilyDeclaration",
    "RelativeMetadataDiagnosticManifest",
    "default_relative_feature_family_declarations",
    "relative_alignment_metadata_diagnostic_manifest",
    "relative_alignment_metadata_diagnostic_path",
    "write_relative_alignment_metadata_diagnostic",
]
