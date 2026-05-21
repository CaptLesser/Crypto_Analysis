from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.regimes.core.serialization import to_jsonable


FINAL_REGIME_PATHWAY_STATUS_COMPLETED = "completed"
FINAL_REGIME_PATHWAY_STATUS_COMPLETED_WITH_WARNINGS = "completed_with_warnings"
FINAL_REGIME_PATHWAY_STATUS_PARTIAL_MISSING_DATA = "partial_missing_data"
FINAL_REGIME_PATHWAY_STATUS_BLOCKED_BY_SOURCE_RESOLUTION = "blocked_by_source_resolution"
FINAL_REGIME_PATHWAY_STATUS_BLOCKED_BY_SCHEMA_GAP = "blocked_by_schema_gap"
FINAL_REGIME_PATHWAY_STATUS_BLOCKED_BY_WRITER_GAP = "blocked_by_writer_gap"
FINAL_REGIME_PATHWAY_STATUS_FAILED = "failed"

FINAL_REGIME_PATHWAY_RUN_STATUSES: frozenset[str] = frozenset(
    {
        FINAL_REGIME_PATHWAY_STATUS_COMPLETED,
        FINAL_REGIME_PATHWAY_STATUS_COMPLETED_WITH_WARNINGS,
        FINAL_REGIME_PATHWAY_STATUS_PARTIAL_MISSING_DATA,
        FINAL_REGIME_PATHWAY_STATUS_BLOCKED_BY_SOURCE_RESOLUTION,
        FINAL_REGIME_PATHWAY_STATUS_BLOCKED_BY_SCHEMA_GAP,
        FINAL_REGIME_PATHWAY_STATUS_BLOCKED_BY_WRITER_GAP,
        FINAL_REGIME_PATHWAY_STATUS_FAILED,
    }
)


@dataclass(frozen=True)
class FinalRegimePathwaySandboxConfig:
    report_root: str | Path = Path("reports") / "regimes" / "foundation" / "final_pathway_output"
    source_registry_config: Mapping[str, Any] | None = None
    explicit_roots: Mapping[str, Any] = field(default_factory=dict)
    market_universe_manifest_path: str | Path | None = None
    universe_eligibility_snapshot_path: str | Path | None = None
    assets: Sequence[str] = ("BTCUSD", "ETHUSD", "SOLUSD")
    bounded_asset_cap: int = 3
    start_ts: int | float | str | None = None
    end_ts: int | float | str | None = None
    clamp_policy: Mapping[str, Any] = field(default_factory=lambda: {"max_years": 2, "bounded_shape_proof": True})
    intervals: Sequence[int] = (60, 240)
    bands: Sequence[str] = ("micro", "meso")
    write_outputs: bool = True
    run_id: str = "final_regime_pathway_sandbox_output"
    require_real_sources: bool = False
    production_enabled: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.production_enabled is not False:
            raise ValueError("Final Regime pathway sandbox runner production_enabled must be false")
        cap = int(self.bounded_asset_cap)
        if cap <= 0:
            raise ValueError("Final Regime pathway sandbox runner bounded_asset_cap must be positive")
        assets = tuple(str(asset).strip() for asset in self.assets if str(asset).strip())
        if not assets:
            raise ValueError("Final Regime pathway sandbox runner requires at least one bounded asset")
        intervals = tuple(int(interval) for interval in self.intervals if int(interval) > 0)
        if not intervals:
            raise ValueError("Final Regime pathway sandbox runner requires at least one interval")
        bands = tuple(str(band).strip().lower() for band in self.bands if str(band).strip())
        if not bands:
            raise ValueError("Final Regime pathway sandbox runner requires at least one band")
        object.__setattr__(self, "assets", assets)
        object.__setattr__(self, "bounded_asset_cap", cap)
        object.__setattr__(self, "intervals", intervals)
        object.__setattr__(self, "bands", bands)
        object.__setattr__(self, "explicit_roots", dict(self.explicit_roots))
        object.__setattr__(self, "clamp_policy", dict(self.clamp_policy))
        object.__setattr__(self, "metadata", dict(self.metadata))
        if self.source_registry_config is not None:
            object.__setattr__(self, "source_registry_config", dict(self.source_registry_config))


@dataclass(frozen=True)
class FinalRegimePathwayRunResult:
    status: str
    report_root: Path
    run_id: str
    source_registry_path: str | None = None
    universe_eligibility_snapshot_path: str | None = None
    forecaster_handoff_index_path: str | None = None
    artifact_inventory_path: str | None = None
    test_branch_readiness_matrix_path: str | None = None
    asset_state_status: str | None = None
    market_state_status: str | None = None
    cross_asset_status: str | None = None
    bounded_end_to_end_sandbox_runner_succeeded: bool = False
    asset_state_outputs_produced: bool = False
    market_state_outputs_produced: bool = False
    cross_asset_feature_outputs_produced: bool = False
    unified_forecaster_handoff_manifests_produced: bool = False
    artifact_inventory_disk_safety_validation_passed: bool = False
    production_outputs_written: bool = False
    production_promotion_performed: bool = False
    production_labels_written: bool = False
    broad_benchmark_run: bool = False
    full_universe_heavy_run: bool = False
    broad_all_to_all_pairwise_run: bool = False
    cross_asset_labels_written: bool = False
    forecaster_training_run: bool = False
    hardcoded_absolute_paths_introduced: bool = False
    warnings: Sequence[str] = ()
    blockers: Sequence[str] = ()
    artifact_paths: Mapping[str, str] = field(default_factory=dict)
    component_summaries: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.status not in FINAL_REGIME_PATHWAY_RUN_STATUSES:
            raise ValueError(f"Unsupported final Regime pathway run status: {self.status!r}")
        for flag in (
            self.production_outputs_written,
            self.production_promotion_performed,
            self.production_labels_written,
            self.broad_benchmark_run,
            self.full_universe_heavy_run,
            self.broad_all_to_all_pairwise_run,
            self.cross_asset_labels_written,
            self.forecaster_training_run,
            self.hardcoded_absolute_paths_introduced,
        ):
            if flag:
                raise ValueError("Final Regime pathway run result may not enable production or blocked behavior flags")
        object.__setattr__(self, "warnings", tuple(str(item) for item in self.warnings))
        object.__setattr__(self, "blockers", tuple(str(item) for item in self.blockers))
        object.__setattr__(self, "artifact_paths", {str(key): str(value) for key, value in dict(self.artifact_paths).items()})
        object.__setattr__(self, "component_summaries", to_jsonable(dict(self.component_summaries)))

    def as_dict(self) -> dict[str, Any]:
        return {
            "artifact_kind": "final_regime_pathway_sandbox_run_result",
            "schema_version": 1,
            "status": self.status,
            "run_id": self.run_id,
            "report_root": "runtime_only_not_serialized",
            "source_registry_path": self.source_registry_path,
            "universe_eligibility_snapshot_path": self.universe_eligibility_snapshot_path,
            "forecaster_handoff_index_path": self.forecaster_handoff_index_path,
            "artifact_inventory_path": self.artifact_inventory_path,
            "test_branch_readiness_matrix_path": self.test_branch_readiness_matrix_path,
            "asset_state_status": self.asset_state_status,
            "market_state_status": self.market_state_status,
            "cross_asset_status": self.cross_asset_status,
            "bounded_end_to_end_sandbox_runner_succeeded": bool(self.bounded_end_to_end_sandbox_runner_succeeded),
            "asset_state_outputs_produced": bool(self.asset_state_outputs_produced),
            "market_state_outputs_produced": bool(self.market_state_outputs_produced),
            "cross_asset_feature_outputs_produced": bool(self.cross_asset_feature_outputs_produced),
            "unified_forecaster_handoff_manifests_produced": bool(self.unified_forecaster_handoff_manifests_produced),
            "artifact_inventory_disk_safety_validation_passed": bool(self.artifact_inventory_disk_safety_validation_passed),
            "production_outputs_written": False,
            "production_promotion_performed": False,
            "production_labels_written": False,
            "broad_benchmark_run": False,
            "full_universe_heavy_run": False,
            "broad_all_to_all_pairwise_run": False,
            "cross_asset_labels_written": False,
            "forecaster_training_run": False,
            "hardcoded_absolute_paths_introduced": False,
            "warnings": list(self.warnings),
            "blockers": list(self.blockers),
            "artifact_paths": to_jsonable(dict(self.artifact_paths)),
            "component_summaries": to_jsonable(dict(self.component_summaries)),
        }


__all__ = [
    "FINAL_REGIME_PATHWAY_RUN_STATUSES",
    "FINAL_REGIME_PATHWAY_STATUS_BLOCKED_BY_SCHEMA_GAP",
    "FINAL_REGIME_PATHWAY_STATUS_BLOCKED_BY_SOURCE_RESOLUTION",
    "FINAL_REGIME_PATHWAY_STATUS_BLOCKED_BY_WRITER_GAP",
    "FINAL_REGIME_PATHWAY_STATUS_COMPLETED",
    "FINAL_REGIME_PATHWAY_STATUS_COMPLETED_WITH_WARNINGS",
    "FINAL_REGIME_PATHWAY_STATUS_FAILED",
    "FINAL_REGIME_PATHWAY_STATUS_PARTIAL_MISSING_DATA",
    "FinalRegimePathwayRunResult",
    "FinalRegimePathwaySandboxConfig",
]
