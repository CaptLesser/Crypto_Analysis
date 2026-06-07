from __future__ import annotations

from dataclasses import dataclass
from typing import Any


SCORING_SCHEMA_ID = "cross_asset_state_scoring_v1_diagnostic"
SCORING_SCHEMA_VERSION = 1
ECONOMIC_DIAGNOSTIC_PENDING = "pending_not_computed"


@dataclass(frozen=True)
class CrossAssetStateDiagnosticScore:
    coverage_score: float
    output_health_score: float
    semantic_separation_score: float
    temporal_persistence_score: float
    economic_diagnostic_status: str = ECONOMIC_DIAGNOSTIC_PENDING
    economic_diagnostic_score: float | None = None
    runtime_seconds: float = 0.0
    hard_health_failure: bool = False

    @property
    def temporal_stability_score(self) -> float:
        return bounded_score(self.temporal_persistence_score)

    @property
    def runtime_tiebreak_score(self) -> float:
        # Runtime is a tiebreak only. It is intentionally excluded from total score.
        return bounded_score(1.0 / (1.0 + max(0.0, float(self.runtime_seconds))))

    @property
    def total_candidate_score(self) -> float:
        if self.hard_health_failure:
            return 0.0
        economic_ready = self.economic_diagnostic_score is not None and self.economic_diagnostic_status == "computed"
        if economic_ready:
            return round(
                0.25 * bounded_score(self.output_health_score)
                + 0.30 * bounded_score(self.semantic_separation_score)
                + 0.15 * self.temporal_stability_score
                + 0.15 * bounded_score(self.coverage_score)
                + 0.15 * bounded_score(float(self.economic_diagnostic_score)),
                6,
            )
        return round(
            0.30 * bounded_score(self.output_health_score)
            + 0.35 * bounded_score(self.semantic_separation_score)
            + 0.20 * self.temporal_stability_score
            + 0.15 * bounded_score(self.coverage_score),
            6,
        )

    @property
    def total_score(self) -> float:
        return self.total_candidate_score

    def as_dict(self) -> dict[str, Any]:
        return {
            "scoring_schema_id": SCORING_SCHEMA_ID,
            "scoring_schema_version": SCORING_SCHEMA_VERSION,
            "coverage_score": round(float(self.coverage_score), 6),
            "output_health_score": round(float(self.output_health_score), 6),
            "semantic_separation_score": round(float(self.semantic_separation_score), 6),
            "temporal_stability_score": round(float(self.temporal_stability_score), 6),
            "temporal_persistence_score": round(float(self.temporal_persistence_score), 6),
            "economic_diagnostic_status": self.economic_diagnostic_status,
            "economic_diagnostic_score": None
            if self.economic_diagnostic_score is None
            else round(float(self.economic_diagnostic_score), 6),
            "runtime_seconds": round(float(self.runtime_seconds), 6),
            "runtime_tiebreak_score": round(float(self.runtime_tiebreak_score), 6),
            "hard_health_failure": bool(self.hard_health_failure),
            "total_candidate_score": self.total_candidate_score,
            "total_score": self.total_candidate_score,
            "final_production_scoring": False,
        }


def bounded_score(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
