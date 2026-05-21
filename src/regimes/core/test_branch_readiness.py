from __future__ import annotations

import importlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.regimes.core.path_safety import validate_report_root
from src.regimes.core.serialization import to_jsonable


TEST_BRANCH_READINESS_ARTIFACT_KIND = "regime_test_branch_readiness_matrix"
TEST_BRANCH_READINESS_FILENAME = "test_branch_readiness_matrix.json"

PATHWAY_ASSET_STATE = "asset_state"
PATHWAY_MARKET_STATE = "market_state"
PATHWAY_CROSS_ASSET = "cross_asset"

BLOCKS_FINAL_OUTPUT_SPRINT = "blocks final output sprint"
DEFER_AUTOMATION_POLISH = "can defer to automation polish"
DEFER_PRODUCTION_PROMOTION = "can defer to production promotion"
BLOCKER_CLASSIFICATIONS: frozenset[str] = frozenset(
    {BLOCKS_FINAL_OUTPUT_SPRINT, DEFER_AUTOMATION_POLISH, DEFER_PRODUCTION_PROMOTION}
)


@dataclass(frozen=True)
class TestBranchReadinessRow:
    pathway: str
    blocker_classification: str
    test_runner_exists: bool
    test_ready_manifest_exists: bool
    feature_profile_clusterer_candidate_comparison_exists: bool
    decision_buckets_supported: bool
    profile_manifest_exists: bool
    stability_scoring_exists: bool
    economic_scoring_exists: bool
    internal_scoring_exists: bool
    axis_specific_profile_decisions_supported: bool
    feature_profile_handoff_path_exists: bool
    relationship_feature_candidate_comparison_path_exists: bool
    regime_label_test_runner_absent_by_design: bool
    clear_next_step: str
    evidence_modules: Mapping[str, str] = field(default_factory=dict)
    gaps: Sequence[str] = ()
    deferred_to_automation_polish: Sequence[str] = ()
    deferred_to_production_promotion: Sequence[str] = ()

    def __post_init__(self) -> None:
        pathway = _non_empty(self.pathway, field_name="pathway")
        if pathway not in {PATHWAY_ASSET_STATE, PATHWAY_MARKET_STATE, PATHWAY_CROSS_ASSET}:
            raise ValueError("Test-branch readiness pathway is unsupported")
        classification = _non_empty(self.blocker_classification, field_name="blocker_classification")
        if classification not in BLOCKER_CLASSIFICATIONS:
            raise ValueError("Test-branch readiness blocker_classification is unsupported")
        object.__setattr__(self, "pathway", pathway)
        object.__setattr__(self, "blocker_classification", classification)
        object.__setattr__(self, "evidence_modules", to_jsonable(dict(self.evidence_modules)))
        object.__setattr__(self, "gaps", _string_tuple(self.gaps, field_name="gaps"))
        object.__setattr__(
            self,
            "deferred_to_automation_polish",
            _string_tuple(self.deferred_to_automation_polish, field_name="deferred_to_automation_polish"),
        )
        object.__setattr__(
            self,
            "deferred_to_production_promotion",
            _string_tuple(self.deferred_to_production_promotion, field_name="deferred_to_production_promotion"),
        )

    @property
    def blocks_final_output_sprint(self) -> bool:
        return self.blocker_classification == BLOCKS_FINAL_OUTPUT_SPRINT

    def as_dict(self) -> dict[str, Any]:
        return {
            "pathway": self.pathway,
            "blocker_classification": self.blocker_classification,
            "blocks_final_output_sprint": self.blocks_final_output_sprint,
            "test_runner_exists": bool(self.test_runner_exists),
            "test_ready_manifest_exists": bool(self.test_ready_manifest_exists),
            "feature_profile_clusterer_candidate_comparison_exists": bool(
                self.feature_profile_clusterer_candidate_comparison_exists
            ),
            "decision_buckets_supported": bool(self.decision_buckets_supported),
            "profile_manifest_exists": bool(self.profile_manifest_exists),
            "stability_scoring_exists": bool(self.stability_scoring_exists),
            "economic_scoring_exists": bool(self.economic_scoring_exists),
            "internal_scoring_exists": bool(self.internal_scoring_exists),
            "axis_specific_profile_decisions_supported": bool(self.axis_specific_profile_decisions_supported),
            "feature_profile_handoff_path_exists": bool(self.feature_profile_handoff_path_exists),
            "relationship_feature_candidate_comparison_path_exists": bool(
                self.relationship_feature_candidate_comparison_path_exists
            ),
            "regime_label_test_runner_absent_by_design": bool(self.regime_label_test_runner_absent_by_design),
            "clear_next_step": self.clear_next_step,
            "evidence_modules": to_jsonable(dict(self.evidence_modules)),
            "gaps": list(self.gaps),
            "deferred_to_automation_polish": list(self.deferred_to_automation_polish),
            "deferred_to_production_promotion": list(self.deferred_to_production_promotion),
        }


@dataclass(frozen=True)
class TestBranchReadinessMatrix:
    rows: Sequence[TestBranchReadinessRow | Mapping[str, Any]]
    schema_version: int = 1
    artifact_kind: str = TEST_BRANCH_READINESS_ARTIFACT_KIND
    production_enabled: bool = False
    profile_selection_run: bool = False
    final_profiles_selected: bool = False

    def __post_init__(self) -> None:
        if self.production_enabled is not False:
            raise ValueError("Test-branch readiness production_enabled must be false")
        if self.profile_selection_run is not False:
            raise ValueError("Test-branch readiness must not run profile selection")
        if self.final_profiles_selected is not False:
            raise ValueError("Test-branch readiness must not select final profiles")
        rows = tuple(row if isinstance(row, TestBranchReadinessRow) else _row_from_mapping(row) for row in self.rows)
        pathways = [row.pathway for row in rows]
        if sorted(pathways) != sorted({PATHWAY_ASSET_STATE, PATHWAY_MARKET_STATE, PATHWAY_CROSS_ASSET}):
            raise ValueError("Test-branch readiness matrix requires one row per pathway")
        object.__setattr__(self, "rows", tuple(sorted(rows, key=lambda row: row.pathway)))
        object.__setattr__(self, "schema_version", int(self.schema_version))
        if int(self.schema_version) != 1:
            raise ValueError("Test-branch readiness schema_version is unsupported")
        object.__setattr__(self, "artifact_kind", _non_empty(self.artifact_kind, field_name="artifact_kind"))

    def as_dict(self) -> dict[str, Any]:
        rows = [row.as_dict() for row in self.rows]
        final_blockers = [row["pathway"] for row in rows if row["blocks_final_output_sprint"]]
        return {
            "artifact_kind": self.artifact_kind,
            "schema_version": int(self.schema_version),
            "rows": rows,
            "summary": {
                "row_count": len(rows),
                "pathways": [row["pathway"] for row in rows],
                "final_output_sprint_blockers": final_blockers,
                "blocks_final_output_sprint": bool(final_blockers),
                "profile_selection_run": False,
                "final_profiles_selected": False,
                "production_profile_selection_enabled": False,
            },
            "production_enabled": False,
            "profile_selection_run": False,
            "final_profiles_selected": False,
        }


def build_test_branch_readiness_matrix() -> TestBranchReadinessMatrix:
    return TestBranchReadinessMatrix(
        rows=(
            _asset_state_row(),
            _market_state_row(),
            _cross_asset_row(),
        )
    )


def validate_test_branch_readiness_matrix(matrix: TestBranchReadinessMatrix | Mapping[str, Any]) -> TestBranchReadinessMatrix:
    resolved = matrix if isinstance(matrix, TestBranchReadinessMatrix) else TestBranchReadinessMatrix(rows=tuple(matrix.get("rows", ())))
    payload = resolved.as_dict()
    if bool(payload.get("production_enabled", False)):
        raise ValueError("Test-branch readiness production_enabled must be false")
    if bool(payload.get("profile_selection_run", False)) or bool(payload.get("final_profiles_selected", False)):
        raise ValueError("Test-branch readiness must not run or finalize profile selection")
    return resolved


def write_test_branch_readiness_matrix(
    matrix: TestBranchReadinessMatrix | Mapping[str, Any],
    *,
    output_root: str | Path,
    relative_path: str | Path = TEST_BRANCH_READINESS_FILENAME,
) -> Path:
    root = validate_report_root(output_root, allow_foundation_descendant=True)
    resolved = validate_test_branch_readiness_matrix(matrix)
    rel = Path(relative_path)
    if rel.is_absolute() or any(part in {"", ".."} for part in rel.parts):
        raise ValueError("Test-branch readiness relative_path must stay within output root")
    path = (root / rel).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError("Test-branch readiness writer refusing to write outside output root") from exc
    _write_json(path, resolved.as_dict())
    return path


def _asset_state_row() -> TestBranchReadinessRow:
    test_runner = _has_callable("src.regimes.asset_state.test_runner", "run_asset_state_test")
    matrix_builder = _has_callable("src.regimes.asset_state.test_runner", "build_asset_state_test_matrix")
    profile_registry = _has_callable("src.regimes.asset_state.profile_registry", "build_asset_state_profile_registry_from_trials")
    registry_module = _module("src.regimes.asset_state.profile_registry")
    buckets = tuple(getattr(registry_module, "PROFILE_REGISTRY_BUCKETS", ())) if registry_module is not None else ()
    return TestBranchReadinessRow(
        pathway=PATHWAY_ASSET_STATE,
        blocker_classification=DEFER_PRODUCTION_PROMOTION,
        test_runner_exists=test_runner,
        test_ready_manifest_exists=_has_callable("src.regimes.asset_state.study_manifest", "default_asset_state_study_manifest"),
        feature_profile_clusterer_candidate_comparison_exists=matrix_builder,
        decision_buckets_supported=_has_decision_buckets(buckets),
        profile_manifest_exists=profile_registry,
        stability_scoring_exists=_has_callable("src.regimes.asset_state.stability_validation", "validate_asset_state_stability")
        or _has_callable("src.regimes.core.stability", "score_precomputed_stability"),
        economic_scoring_exists=_has_callable("src.regimes.asset_state.economic_validation", "validate_asset_state_economic_separability")
        or _has_callable("src.regimes.core.economic", "score_economic_separability"),
        internal_scoring_exists=_has_callable("src.regimes.core.scoring", "score_internal_validity"),
        axis_specific_profile_decisions_supported=profile_registry,
        feature_profile_handoff_path_exists=_has_callable("src.regimes.asset_state.handoff", "build_asset_state_forecaster_handoff_manifest"),
        relationship_feature_candidate_comparison_path_exists=False,
        regime_label_test_runner_absent_by_design=False,
        clear_next_step="Automation polish can run bounded Asset-State Test studies and consume the non-production profile registry; production promotion remains deferred.",
        evidence_modules={
            "test_runner": "src.regimes.asset_state.test_runner.run_asset_state_test",
            "candidate_matrix": "src.regimes.asset_state.test_runner.build_asset_state_test_matrix",
            "profile_registry": "src.regimes.asset_state.profile_registry.build_asset_state_profile_registry_from_trials",
            "stability": "src.regimes.asset_state.stability_validation.validate_asset_state_stability",
            "economic": "src.regimes.asset_state.economic_validation.validate_asset_state_economic_separability",
            "internal": "src.regimes.core.scoring.score_internal_validity",
        },
        gaps=(),
        deferred_to_automation_polish=("run bounded test studies", "rank non-production profile candidates"),
        deferred_to_production_promotion=("freeze final profile manifest", "enable production promotion gate"),
    )


def _market_state_row() -> TestBranchReadinessRow:
    test_runner = _has_callable("src.regimes.market_state.test_runner", "run_market_state_test")
    matrix_builder = _has_callable("src.regimes.market_state.test_runner", "build_market_state_test_matrix")
    profile_registry = _has_callable("src.regimes.market_state.profile_registry", "build_market_state_profile_registry_from_trials")
    registry_module = _module("src.regimes.market_state.profile_registry")
    buckets = tuple(getattr(registry_module, "PROFILE_REGISTRY_BUCKETS", ())) if registry_module is not None else ()
    return TestBranchReadinessRow(
        pathway=PATHWAY_MARKET_STATE,
        blocker_classification=DEFER_PRODUCTION_PROMOTION,
        test_runner_exists=test_runner,
        test_ready_manifest_exists=_has_callable("src.regimes.market_state.study_manifest", "default_market_state_study_manifest"),
        feature_profile_clusterer_candidate_comparison_exists=matrix_builder,
        decision_buckets_supported=_has_decision_buckets(buckets),
        profile_manifest_exists=profile_registry,
        stability_scoring_exists=_has_callable("src.regimes.core.stability", "score_precomputed_stability"),
        economic_scoring_exists=_has_callable("src.regimes.core.economic", "score_economic_separability"),
        internal_scoring_exists=_has_callable("src.regimes.core.scoring", "score_internal_validity"),
        axis_specific_profile_decisions_supported=profile_registry,
        feature_profile_handoff_path_exists=_has_callable("src.regimes.market_state.handoff", "build_market_state_forecaster_handoff_manifest"),
        relationship_feature_candidate_comparison_path_exists=False,
        regime_label_test_runner_absent_by_design=False,
        clear_next_step="Automation polish can run bounded Market-State Test studies per axis and consume the non-production profile registry; production promotion remains deferred.",
        evidence_modules={
            "test_runner": "src.regimes.market_state.test_runner.run_market_state_test",
            "candidate_matrix": "src.regimes.market_state.test_runner.build_market_state_test_matrix",
            "profile_registry": "src.regimes.market_state.profile_registry.build_market_state_profile_registry_from_trials",
            "stability": "src.regimes.core.stability.score_precomputed_stability",
            "economic": "src.regimes.core.economic.score_economic_separability",
            "internal": "src.regimes.core.scoring.score_internal_validity",
        },
        gaps=(),
        deferred_to_automation_polish=("run bounded axis-specific test studies", "rank non-production market profile candidates"),
        deferred_to_production_promotion=("freeze final axis profile manifests", "enable production promotion gate"),
    )


def _cross_asset_row() -> TestBranchReadinessRow:
    feature_handoff = _has_callable("src.regimes.regime_features.cross_asset_handoff", "build_cross_asset_forecaster_handoff_manifest")
    catalog = _has_callable("src.regimes.regime_features.cross_asset_feature_catalog", "default_cross_asset_relationship_feature_catalog")
    generator = _has_callable("src.regimes.regime_features.cross_asset_feature_generator", "build_cross_asset_feature_rows_from_handoff")
    return TestBranchReadinessRow(
        pathway=PATHWAY_CROSS_ASSET,
        blocker_classification=DEFER_AUTOMATION_POLISH,
        test_runner_exists=False,
        test_ready_manifest_exists=feature_handoff,
        feature_profile_clusterer_candidate_comparison_exists=False,
        decision_buckets_supported=False,
        profile_manifest_exists=False,
        stability_scoring_exists=False,
        economic_scoring_exists=False,
        internal_scoring_exists=False,
        axis_specific_profile_decisions_supported=False,
        feature_profile_handoff_path_exists=feature_handoff,
        relationship_feature_candidate_comparison_path_exists=catalog and generator,
        regime_label_test_runner_absent_by_design=True,
        clear_next_step="Build a Cross-Asset feature-profile comparison and profile registry around relationship_feature_catalog plus cross_asset_feature_rows before any Cross-Asset regime-label Test runner.",
        evidence_modules={
            "feature_handoff": "src.regimes.regime_features.cross_asset_handoff.build_cross_asset_forecaster_handoff_manifest",
            "feature_catalog": "src.regimes.regime_features.cross_asset_feature_catalog.default_cross_asset_relationship_feature_catalog",
            "feature_generator": "src.regimes.regime_features.cross_asset_feature_generator.build_cross_asset_feature_rows_from_handoff",
            "label_runner": "absent_by_design",
        },
        gaps=(
            "no Cross-Asset feature-profile comparison registry yet",
            "no Cross-Asset profile decision buckets yet",
            "no Cross-Asset regime-label Test runner by design",
        ),
        deferred_to_automation_polish=(
            "create Cross-Asset feature-profile comparison harness",
            "create Cross-Asset profile registry and non-production decision buckets",
        ),
        deferred_to_production_promotion=("defer Cross-Asset regime-label runner and promotion until after feature-profile validation",),
    )


def _has_decision_buckets(buckets: Sequence[str]) -> bool:
    required = {
        "candidate_profiles",
        "watchlist_profiles",
        "rejected_profiles",
        "fallback_only_profiles",
        "blocked_profiles",
    }
    return required.issubset(set(buckets))


def _row_from_mapping(value: Mapping[str, Any]) -> TestBranchReadinessRow:
    payload = dict(value)
    payload.pop("blocks_final_output_sprint", None)
    return TestBranchReadinessRow(**payload)


def _has_callable(module_name: str, attr_name: str) -> bool:
    module = _module(module_name)
    return module is not None and callable(getattr(module, attr_name, None))


def _module(module_name: str) -> Any | None:
    try:
        return importlib.import_module(module_name)
    except Exception:
        return None


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(to_jsonable(dict(payload)), f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp, path)


def _non_empty(value: object, *, field_name: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"Test-branch readiness {field_name} must be non-empty")
    return text


def _string_tuple(values: Sequence[str], *, field_name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValueError(f"Test-branch readiness {field_name} must be a sequence")
    return tuple(_non_empty(value, field_name=field_name) for value in values)


__all__ = [
    "BLOCKS_FINAL_OUTPUT_SPRINT",
    "BLOCKER_CLASSIFICATIONS",
    "DEFER_AUTOMATION_POLISH",
    "DEFER_PRODUCTION_PROMOTION",
    "PATHWAY_ASSET_STATE",
    "PATHWAY_CROSS_ASSET",
    "PATHWAY_MARKET_STATE",
    "TEST_BRANCH_READINESS_ARTIFACT_KIND",
    "TEST_BRANCH_READINESS_FILENAME",
    "TestBranchReadinessMatrix",
    "TestBranchReadinessRow",
    "build_test_branch_readiness_matrix",
    "validate_test_branch_readiness_matrix",
    "write_test_branch_readiness_matrix",
]
