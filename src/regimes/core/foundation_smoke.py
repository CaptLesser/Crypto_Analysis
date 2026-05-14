"""Legacy foundation smoke compatibility surface.

Canonical bounded smoke execution lives in
``src.regimes.studies.foundation_smoke``. This module remains importable for
older CLI diagnostics and compatibility tests.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from src.forecasting.common.io_atomic import atomic_replace, sibling_temp_path
from src.regimes.core.artifacts import safe_path_part, write_json
from src.regimes.core.clusterer_adapters import clusterer_adapter_registry
from src.regimes.core.feature_preprocessing import default_feature_pool_registry
from src.regimes.core.foundation_contracts import (
    ArtifactReference,
    RegimeArtifactManifestContract,
    RegimeStudyIdentity,
    SourceArtifactLineage,
)
from src.regimes.core.paths import default_foundation_report_root, resolve_project_root
from src.regimes.core.pathway_artifacts import require_pathway_diagnostics_root
from src.regimes.core.study_runner import RegimeStudyManifest, run_regime_single_trial


REGIME_FOUNDATION_SMOKE_SCHEMA_VERSION = 1
REGIME_FOUNDATION_SMOKE_ARTIFACT_KIND = "regime_foundation_smoke"
REGIME_FOUNDATION_PROMOTION_GATE_ARTIFACT_KIND = "regime_foundation_promotion_gate"


def synthetic_foundation_smoke_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ts": list(range(8)),
            "log_return": [-0.05, -0.04, -0.06, 0.04, 0.05, 0.06, 0.07, 0.08],
            "macd_hist_12_26_9": [-1.2, -1.1, -1.3, 1.1, 1.2, 1.3, 1.4, 1.5],
            "rsi_14": [30.0, 31.0, 29.0, 70.0, 71.0, 72.0, 73.0, 74.0],
            "adx_14": [20.0, 21.0, 19.0, 40.0, 41.0, 42.0, 43.0, 44.0],
            "future_log_return": [-0.02, -0.03, -0.01, 0.02, 0.03, 0.04, 0.05, 0.06],
            "future_realized_volatility": [0.02, 0.021, 0.022, 0.05, 0.052, 0.051, 0.053, 0.054],
            "future_max_drawdown": [-0.04, -0.05, -0.03, -0.01, -0.011, -0.012, -0.013, -0.014],
        }
    )


def foundation_smoke_study_manifest(diagnostics_root: Path) -> RegimeStudyManifest:
    return RegimeStudyManifest(
        study_id="foundation_smoke_asset_trend_micro",
        layer="asset_state",
        axis="trend",
        band="micro",
        asset_scope="single_asset",
        asset="SMOKEUSD",
        feature_families_allowed=("asset_trend_manual_baseline",),
        preprocessing_families_allowed=("noop",),
        clusterer_families_allowed=("kmeans",),
        hyperparameter_search_spaces={
            "clusterer_hyperparameters_by_family": {
                "kmeans": {
                    "n_clusters": {"type": "integer", "choices": [2]},
                    "random_state": {"type": "integer", "choices": [17]},
                }
            }
        },
        split_policy={"train_row_count": 6, "walk_forward": {"enabled": False}},
        forward_target_horizons=(60,),
        trial_budget={"max_trials": 1, "timeout_s": 30, "single_trial_only": True},
        random_seed_policy={"base_seed": 17, "deterministic": True},
        runtime_profile_name="foundation_smoke",
        diagnostics_output_root=Path(diagnostics_root),
        production_classification="sandbox",
    )


def _source_lineage() -> tuple[SourceArtifactLineage, ...]:
    return (
        SourceArtifactLineage(
            artifact_kind="synthetic_regime_foundation_smoke_fixture",
            artifact_path="memory://regimes/foundation_smoke/synthetic_asset_trend_micro",
            schema_version=REGIME_FOUNDATION_SMOKE_SCHEMA_VERSION,
            content_hash="sha256:synthetic-foundation-smoke-v1",
            produced_by="src.regimes.core.foundation_smoke.synthetic_foundation_smoke_frame",
            metadata={"synthetic": True, "production_input": False},
        ),
    )


def foundation_smoke_output_dir(diagnostics_root: Path, run_id: str) -> Path:
    return Path(diagnostics_root) / "regime_foundation_smoke" / safe_path_part(run_id, context="foundation smoke run id")


def _promotion_gate(
    *,
    manifest: RegimeStudyManifest,
    trial_payload: Mapping[str, Any],
    artifact_manifest: RegimeArtifactManifestContract,
) -> dict[str, Any]:
    scoreboard_present = bool(trial_payload.get("trial", {}).get("scoreboard"))
    reasons = (
        "foundation_smoke_uses_synthetic_fixture",
        f"study_classification={manifest.production_classification}",
        "single_trial_is_not_method_selection_evidence",
        "production_writes_are_prohibited",
        "promotion_requires_external_review_and_broad_validation",
    )
    return {
        "schema_version": REGIME_FOUNDATION_SMOKE_SCHEMA_VERSION,
        "artifact_kind": REGIME_FOUNDATION_PROMOTION_GATE_ARTIFACT_KIND,
        "status": "production_promotion_refused",
        "production_promotion_allowed": False,
        "method_promotion_allowed": False,
        "reasons": list(reasons),
        "checked_flags": {
            "scoreboard_present": scoreboard_present,
            "artifact_manifest_not_production": artifact_manifest.not_production_flags["not_production"],
            "production_outputs_written": False,
            "synthetic_fixture": True,
            "single_trial_only": True,
        },
    }


def _artifact_manifest(
    *,
    identity: RegimeStudyIdentity,
    paths: Mapping[str, str],
) -> RegimeArtifactManifestContract:
    return RegimeArtifactManifestContract(
        identity=identity,
        write_kind="Regime foundation smoke sandbox diagnostics",
        source_lineage=_source_lineage(),
        created_artifacts=(
            ArtifactReference(name="trial_result", path=paths["trial_result"], artifact_kind="regime_single_trial_run"),
            ArtifactReference(name="scoreboard", path=paths["scoreboard"], artifact_kind="regime_trial_scoreboard"),
            ArtifactReference(
                name="promotion_gate",
                path=paths["promotion_gate"],
                artifact_kind=REGIME_FOUNDATION_PROMOTION_GATE_ARTIFACT_KIND,
            ),
            ArtifactReference(
                name="foundation_smoke_result",
                path=paths["foundation_smoke_result"],
                artifact_kind=REGIME_FOUNDATION_SMOKE_ARTIFACT_KIND,
            ),
            ArtifactReference(name="markdown_summary", path=paths["markdown"], artifact_kind="markdown_summary"),
        ),
        disabled_artifacts=(
            ArtifactReference(name="production_labels", reason="foundation smoke is sandbox synthetic only"),
            ArtifactReference(name="production_definition", reason="foundation smoke is not promotion evidence"),
            ArtifactReference(name="staged_method_profile", reason="method promotion is out of scope"),
        ),
        production_outputs_written=False,
        status="diagnostics_only",
    )


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
    gate = payload["promotion_gate"]
    trial = payload["trial_result"]["trial"]
    scoreboard = trial["scoreboard"]
    coverage = scoreboard["metric_families"]["coverage_degeneracy"]["metrics"]
    return "\n".join(
        [
            "# Regime Foundation Smoke",
            "",
            f"- run_id: {payload['run_id']}",
            "- layer/axis/band: asset_state / trend / micro",
            f"- feature_family: {trial['feature_family']}",
            f"- preprocessing_family: {trial['preprocessing_family']}",
            f"- clusterer_family: {trial['clusterer_family']}",
            f"- trial_status: {trial['status']}",
            f"- effective_state_count: {coverage.get('effective_state_count')}",
            f"- production_promotion_allowed: {gate['production_promotion_allowed']}",
            f"- production_outputs_written: {payload['artifact_manifest']['not_production_flags']['production_outputs_written']}",
            "",
            "This smoke is synthetic, deterministic, and sandbox-only.",
            "",
        ]
    )


@dataclass(frozen=True)
class RegimeFoundationSmokeResult:
    payload: Mapping[str, Any]
    artifact_paths: Mapping[str, str]
    schema_version: int = REGIME_FOUNDATION_SMOKE_SCHEMA_VERSION

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "payload": dict(self.payload),
            "artifact_paths": dict(self.artifact_paths),
        }


def run_regime_foundation_smoke(
    *,
    diagnostics_root: Path,
    run_id: str = "foundation_smoke",
    project_root: Path | None = None,
    write_outputs: bool = True,
) -> RegimeFoundationSmokeResult:
    if not clusterer_adapter_registry()["kmeans"].dependency_available:
        raise RuntimeError("Regime foundation smoke requires sklearn KMeans")
    diagnostics_root = Path(diagnostics_root)
    root_policy = require_pathway_diagnostics_root(diagnostics_root, project_root=project_root)
    manifest = foundation_smoke_study_manifest(diagnostics_root)
    frame = synthetic_foundation_smoke_frame()
    feature_contract = default_feature_pool_registry()["asset_trend_manual_baseline"]
    trial_id = safe_path_part(f"{run_id}_kmeans", context="foundation smoke trial id")
    trial_result = run_regime_single_trial(
        manifest,
        frame,
        trial_id=trial_id,
        feature_family="asset_trend_manual_baseline",
        preprocessing_family="noop",
        clusterer_family="kmeans",
        clusterer_hyperparameters={"n_clusters": 2, "n_init": 10, "random_state": 17},
        write_outputs=write_outputs,
        project_root=project_root,
    )
    out_dir = foundation_smoke_output_dir(diagnostics_root, run_id)
    paths = {
        "trial_result": trial_result.artifact_paths["json"],
        "trial_markdown": trial_result.artifact_paths["markdown"],
        "scoreboard": str(out_dir / "scoreboard.json"),
        "artifact_manifest": str(out_dir / "artifact_manifest.json"),
        "promotion_gate": str(out_dir / "promotion_gate.json"),
        "foundation_smoke_result": str(out_dir / "foundation_smoke_result.json"),
        "markdown": str(out_dir / "foundation_smoke_summary.md"),
    }
    identity = RegimeStudyIdentity(
        layer="asset_state",
        axis="trend",
        band="micro",
        asset_scope="single_asset",
        production_classification="sandbox",
    )
    artifact_manifest = _artifact_manifest(identity=identity, paths=paths)
    promotion_gate = _promotion_gate(
        manifest=manifest,
        trial_payload=trial_result.payload,
        artifact_manifest=artifact_manifest,
    )
    scoreboard = trial_result.payload["trial"]["scoreboard"]
    payload = {
        "schema_version": REGIME_FOUNDATION_SMOKE_SCHEMA_VERSION,
        "artifact_kind": REGIME_FOUNDATION_SMOKE_ARTIFACT_KIND,
        "status": "ok",
        "run_id": str(run_id),
        "diagnostics_root_policy": root_policy.as_dict(),
        "study_unit": identity.as_dict(),
        "feature_contract": feature_contract.as_dict(),
        "trial_result": trial_result.payload,
        "scoreboard": scoreboard,
        "artifact_manifest": artifact_manifest.as_dict(),
        "promotion_gate": promotion_gate,
        "artifact_paths": paths,
        "artifact_boundary": {
            "diagnostics_only": True,
            "synthetic_fixture": True,
            "production_outputs_written": False,
            "production_labels_written": False,
            "production_definitions_written": False,
            "method_promotion_written": False,
        },
    }
    if write_outputs:
        write_json(Path(paths["scoreboard"]), scoreboard, write_kind="Regime foundation smoke scoreboard")
        write_json(
            Path(paths["artifact_manifest"]),
            artifact_manifest.as_dict(),
            write_kind="Regime foundation smoke artifact manifest",
        )
        write_json(Path(paths["promotion_gate"]), promotion_gate, write_kind="Regime foundation smoke promotion gate")
        write_json(Path(paths["foundation_smoke_result"]), payload, write_kind="Regime foundation smoke result")
        _write_markdown(Path(paths["markdown"]), _markdown_summary(payload))
    return RegimeFoundationSmokeResult(payload=payload, artifact_paths=paths)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the tiny Regime foundation smoke.")
    parser.add_argument("--diagnostics-root", type=Path, default=default_foundation_report_root("legacy_foundation_smoke"))
    parser.add_argument("--run-id", type=str, default="foundation_smoke")
    parser.add_argument("--no-write", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    result = run_regime_foundation_smoke(
        diagnostics_root=Path(args.diagnostics_root),
        run_id=str(args.run_id),
        project_root=resolve_project_root(),
        write_outputs=not bool(args.no_write),
    )
    print(json.dumps(result.as_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "REGIME_FOUNDATION_PROMOTION_GATE_ARTIFACT_KIND",
    "REGIME_FOUNDATION_SMOKE_ARTIFACT_KIND",
    "REGIME_FOUNDATION_SMOKE_SCHEMA_VERSION",
    "RegimeFoundationSmokeResult",
    "foundation_smoke_output_dir",
    "foundation_smoke_study_manifest",
    "run_regime_foundation_smoke",
    "synthetic_foundation_smoke_frame",
]
