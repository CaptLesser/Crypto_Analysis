from __future__ import annotations

from src.regimes.studies.fixtures import synthetic_asset_state_fixture
from src.regimes.studies.foundation_smoke import (
    DEFAULT_FOUNDATION_SMOKE_REPORT_ROOT,
    FOUNDATION_SMOKE_ARTIFACT_KIND,
    FOUNDATION_SMOKE_SCHEMA_VERSION,
    FoundationSmokeResult,
    run_foundation_smoke,
)
from src.regimes.studies.manifest import (
    DEFAULT_STUDY_BUDGET,
    DEFAULT_STUDY_REPORT_ROOT,
    DEFAULT_STUDY_SPLIT_POLICY,
    REGIME_STUDY_MANIFEST_SCHEMA_VERSION,
    StudyManifest,
    default_asset_trend_manifest,
)
from src.regimes.studies.optuna_runner import (
    DEFAULT_OPTUNA_REPORT_ROOT,
    OPTUNA_AVAILABLE,
    OPTUNA_STUB_ARTIFACT_KIND,
    OPTUNA_STUB_SCHEMA_VERSION,
    OptunaStubStudyResult,
    OptunaUnavailableError,
    default_optuna_study_manifest,
    run_optuna_stub,
    validate_optuna_report_root,
)
from src.regimes.studies.search_space import StudySearchSpace, build_search_space
from src.regimes.studies.single_trial import SingleTrialResult, run_single_trial
from src.regimes.studies.small_panel_benchmark import (
    DEFAULT_SMALL_PANEL_BENCHMARK_ROOT,
    SMALL_PANEL_BENCHMARK_ARTIFACT_KIND,
    SMALL_PANEL_BENCHMARK_SCHEMA_VERSION,
    SmallPanelBenchmarkResult,
    run_small_panel_benchmark,
    synthetic_small_panel,
)


__all__ = [
    "DEFAULT_STUDY_BUDGET",
    "DEFAULT_FOUNDATION_SMOKE_REPORT_ROOT",
    "DEFAULT_OPTUNA_REPORT_ROOT",
    "DEFAULT_SMALL_PANEL_BENCHMARK_ROOT",
    "DEFAULT_STUDY_REPORT_ROOT",
    "DEFAULT_STUDY_SPLIT_POLICY",
    "FOUNDATION_SMOKE_ARTIFACT_KIND",
    "FOUNDATION_SMOKE_SCHEMA_VERSION",
    "FoundationSmokeResult",
    "OPTUNA_AVAILABLE",
    "OPTUNA_STUB_ARTIFACT_KIND",
    "OPTUNA_STUB_SCHEMA_VERSION",
    "OptunaStubStudyResult",
    "OptunaUnavailableError",
    "REGIME_STUDY_MANIFEST_SCHEMA_VERSION",
    "SMALL_PANEL_BENCHMARK_ARTIFACT_KIND",
    "SMALL_PANEL_BENCHMARK_SCHEMA_VERSION",
    "SingleTrialResult",
    "SmallPanelBenchmarkResult",
    "StudyManifest",
    "StudySearchSpace",
    "build_search_space",
    "default_asset_trend_manifest",
    "default_optuna_study_manifest",
    "run_foundation_smoke",
    "run_optuna_stub",
    "run_small_panel_benchmark",
    "run_single_trial",
    "synthetic_asset_state_fixture",
    "synthetic_small_panel",
    "validate_optuna_report_root",
]
