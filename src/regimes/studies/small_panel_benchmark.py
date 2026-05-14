from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from src.regimes.core.clusterer_adapters import clusterer_adapter_registry
from src.regimes.core.clusterer_base import ASSIGN_STATUS_ASSIGNED, FIT_STATUS_FITTED
from src.regimes.core.clusterer_registry import default_clusterer_registry
from src.regimes.core.contracts import (
    CANONICAL_SCHEMA_VERSION,
    RegimeAxis,
    RegimeBand,
    RegimeClassification,
    RegimeLayer,
    RunStatus,
    require_json_mapping,
    require_non_empty_string,
    require_schema_version,
)
from src.regimes.core.feature_registry import default_feature_family_registry
from src.regimes.core.flat_asset_policy import evaluate_flat_asset_policy
from src.regimes.core.preprocessing import fit_preprocessing_pipeline, transform_score_window_preprocessor
from src.regimes.core.promotion_gate import PROMOTION_STATUS_BLOCKED, PromotionGateInput, evaluate_promotion_gate
from src.regimes.core.scoreboard import build_regime_scoreboard
from src.regimes.core.serialization import dumps_json, loads_json, require_json_object, to_jsonable
from src.regimes.studies.manifest import StudyManifest
from src.regimes.studies.search_space import build_search_space


SMALL_PANEL_BENCHMARK_SCHEMA_VERSION = CANONICAL_SCHEMA_VERSION
SMALL_PANEL_BENCHMARK_ARTIFACT_KIND = "regime_small_panel_benchmark_summary"
SMALL_PANEL_FAMILY_ARTIFACT_KIND = "regime_small_panel_benchmark_family_artifact_manifest"
DEFAULT_SMALL_PANEL_BENCHMARK_ROOT = Path("reports") / "regimes" / "foundation" / "benchmarks"
SMALL_PANEL_SUMMARY_JSON = "small_panel_summary.json"
SMALL_PANEL_SUMMARY_MD = "small_panel_summary.md"
BENCHMARK_FAMILIES: tuple[str, ...] = (
    "kmeans",
    "minibatch_kmeans",
    "gaussian_mixture",
    "agglomerative",
)


def _resolve_root(report_root: str | Path) -> Path:
    root = Path(report_root).expanduser()
    if not root.is_absolute():
        root = Path.cwd() / root
    return root.resolve()


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        return path == root or path.is_relative_to(root)
    except AttributeError:  # pragma: no cover - Python compatibility only
        try:
            path.relative_to(root)
            return True
        except ValueError:
            return False


def _validate_benchmark_report_root(report_root: str | Path, *, project_root: str | Path | None = None) -> Path:
    root = _resolve_root(report_root)
    project = _resolve_root(project_root or Path.cwd())
    production_tokens = {"parquet", "regime_definitions", "model_states", "state"}
    if any(part.lower() in production_tokens for part in root.parts):
        raise ValueError("Regime small-panel benchmark report root is production-adjacent and is not allowed")
    for candidate in (
        project / "parquet",
        project / "regime_definitions",
        project / "model_states",
        project / "state",
    ):
        if _is_relative_to(root, _resolve_root(candidate)):
            raise ValueError("Regime small-panel benchmark report root is production-adjacent and is not allowed")
    normalized = tuple(part.lower() for part in root.parts)
    if len(normalized) < 4 or normalized[-4:] != ("reports", "regimes", "foundation", "benchmarks"):
        raise ValueError("Regime small-panel benchmark report root must end with reports/regimes/foundation/benchmarks")
    return root


def _safe_child(root: Path, *parts: str) -> Path:
    candidate = root.joinpath(*parts).resolve()
    if not _is_relative_to(candidate, root):
        raise ValueError("Regime small-panel benchmark artifact path must stay under report_root")
    return candidate


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dumps_json(payload) + "\n", encoding="utf-8")


def _write_markdown(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def synthetic_small_panel(*, seed: int = 17, periods: int = 12) -> pd.DataFrame:
    if int(periods) < 8:
        raise ValueError("Regime small-panel benchmark requires at least eight periods")
    rng = np.random.default_rng(int(seed))
    timestamps = pd.date_range("2026-01-01T00:00:00Z", periods=int(periods), freq="30min")
    rows: list[dict[str, Any]] = []
    for idx, timestamp in enumerate(timestamps):
        active_state = -1.0 if idx % 2 == 0 else 1.0
        volatile_state = -1.0 if idx % 3 == 0 else 1.0
        drift = float(idx) / max(float(periods - 1), 1.0)
        rows.append(
            {
                "timestamp": timestamp,
                "asset": "ACTIVEUSD",
                "panel_behavior": "active",
                "log_return": 0.030 * active_state + 0.002 * drift,
                "macd_hist_12_26_9": 1.15 * active_state + 0.04 * drift,
                "rsi_14": 50.0 + 18.0 * active_state - 0.2 * idx,
                "adx_14": 28.0 + 0.4 * idx,
                "atr_14": 0.040 + 0.001 * idx,
                "ret_std_20": 0.035 + 0.001 * idx,
                "cv_20": 1.00 + 0.02 * idx,
                "vol_osc_pct_14_28": 0.12 * active_state,
                "trade_intensity": 100.0 + 3.0 * idx,
                "avg_trade_size": 1.20 + 0.01 * idx,
                "vroc_14": 0.10 * active_state,
                "prr": 0.52 + 0.06 * active_state,
                "future_log_return": 0.026 * active_state,
                "future_realized_volatility": 0.040 + 0.001 * idx,
                "future_max_drawdown": -0.025 - 0.001 * (idx % 2),
            }
        )
        rows.append(
            {
                "timestamp": timestamp,
                "asset": "VOLUSD",
                "panel_behavior": "volatile",
                "log_return": 0.065 * volatile_state + float(rng.normal(0.0, 0.002)),
                "macd_hist_12_26_9": 1.80 * volatile_state + float(rng.normal(0.0, 0.01)),
                "rsi_14": 50.0 + 26.0 * volatile_state,
                "adx_14": 38.0 + 0.5 * (idx % 4),
                "atr_14": 0.095 + 0.003 * (idx % 4),
                "ret_std_20": 0.090 + 0.002 * (idx % 5),
                "cv_20": 2.10 + 0.04 * (idx % 4),
                "vol_osc_pct_14_28": 0.28 * volatile_state,
                "trade_intensity": 165.0 + 4.0 * (idx % 5),
                "avg_trade_size": 2.00 + 0.02 * (idx % 3),
                "vroc_14": 0.24 * volatile_state,
                "prr": 0.55 + 0.12 * volatile_state,
                "future_log_return": 0.050 * volatile_state,
                "future_realized_volatility": 0.100 + 0.003 * (idx % 4),
                "future_max_drawdown": -0.070 - 0.003 * (idx % 3),
            }
        )
        rows.append(
            {
                "timestamp": timestamp,
                "asset": "FLATUSD",
                "panel_behavior": "near_flat",
                "log_return": 0.0,
                "macd_hist_12_26_9": 0.0,
                "rsi_14": 50.0,
                "adx_14": 0.01,
                "atr_14": 0.0001,
                "ret_std_20": 0.0001,
                "cv_20": 0.0,
                "vol_osc_pct_14_28": 0.0,
                "trade_intensity": 0.0,
                "avg_trade_size": 0.0,
                "vroc_14": 0.0,
                "prr": 0.50,
                "future_log_return": 0.0,
                "future_realized_volatility": 0.0001,
                "future_max_drawdown": 0.0,
            }
        )
    return pd.DataFrame(rows).sort_values(["timestamp", "asset"]).reset_index(drop=True)


def small_panel_study_manifest(
    *,
    report_root: str | Path = DEFAULT_SMALL_PANEL_BENCHMARK_ROOT,
    seed: int = 17,
    candidate_families: Sequence[str] | None = None,
) -> StudyManifest:
    root = _validate_benchmark_report_root(report_root)
    families = tuple(candidate_families or _candidate_families())
    return StudyManifest(
        study_id="small_panel_asset_trend_micro_benchmark",
        layer=RegimeLayer.ASSET_STATE,
        axis=RegimeAxis.TREND,
        band=RegimeBand.MICRO,
        classification=RegimeClassification.SANDBOX,
        feature_families=("asset_state_trend_metadata_only",),
        preprocessing_options=("robust_scale",),
        candidate_clusterer_families=families,
        split_policy={"name": "deterministic_head_tail", "train_fraction": None, "train_rows": 24},
        budget={
            "max_trials": int(len(families)),
            "timeout_seconds": 120,
            "random_seed": int(seed),
            "tiny_cluster_threshold": 1,
        },
        report_root=root,
        metadata={
            "purpose": "small_panel_benchmark",
            "synthetic_panel": True,
            "production_outputs_written": False,
            "final_winner_freeze": False,
        },
    )


def _candidate_families() -> tuple[str, ...]:
    families = list(BENCHMARK_FAMILIES)
    hdbscan_spec = clusterer_adapter_registry().get("hdbscan")
    if hdbscan_spec is not None and hdbscan_spec.dependency_available:
        families.append("hdbscan")
    return tuple(families)


def _clusterer_params(family: str, *, seed: int) -> dict[str, Any]:
    if family == "kmeans":
        return {"n_clusters": 3, "n_init": 10, "random_state": int(seed)}
    if family == "minibatch_kmeans":
        return {"n_clusters": 3, "n_init": 10, "batch_size": 9, "random_state": int(seed)}
    if family == "gaussian_mixture":
        return {"n_components": 3, "covariance_type": "full", "random_state": int(seed)}
    if family == "agglomerative":
        return {"n_clusters": 3}
    if family == "hdbscan":
        return {"min_cluster_size": 3, "min_samples": 2, "prediction_data": True}
    return {}


def _split_panel(frame: pd.DataFrame, manifest: StudyManifest) -> tuple[pd.DataFrame, pd.DataFrame]:
    unique_timestamps = list(dict.fromkeys(frame["timestamp"].tolist()))
    train_rows = int(manifest.split_policy["train_rows"])
    rows_per_timestamp = int(frame.groupby("timestamp").size().median())
    train_timestamp_count = max(1, min(len(unique_timestamps) - 1, train_rows // max(rows_per_timestamp, 1)))
    train_timestamps = set(unique_timestamps[:train_timestamp_count])
    train = frame[frame["timestamp"].isin(train_timestamps)].copy()
    score = frame[~frame["timestamp"].isin(train_timestamps)].copy()
    return train, score


def _stability_perturbations(labels: Sequence[int]) -> tuple[dict[str, Any], ...]:
    return ({"name": "identity_precomputed_stub", "labels": [int(label) for label in labels]},)


def _score_value(scoreboard: Mapping[str, Any]) -> float:
    internal = scoreboard["sections"]["internal_validity"]["metrics"]
    silhouette = internal.get("silhouette", {})
    if isinstance(silhouette, Mapping) and silhouette.get("status") == "computed":
        try:
            return float(silhouette["value"])
        except Exception:
            return -999.0
    return -999.0


def _family_paths(root: Path, family: str) -> dict[str, Path]:
    family_root = _safe_child(root, "families", family)
    return {
        "artifact_manifest": _safe_child(family_root, "artifact_manifest.json"),
        "scoreboard": _safe_child(family_root, "scoreboard.json"),
        "promotion_gate": _safe_child(family_root, "promotion_gate.json"),
        "flat_policy": _safe_child(family_root, "flat_asset_policy.json"),
        "result": _safe_child(family_root, "result.json"),
    }


def _artifact_manifest(
    *,
    manifest: StudyManifest,
    family: str,
    paths: Mapping[str, Path],
    result_status: str,
) -> dict[str, Any]:
    return {
        "schema_version": SMALL_PANEL_BENCHMARK_SCHEMA_VERSION,
        "artifact_kind": SMALL_PANEL_FAMILY_ARTIFACT_KIND,
        "family_name": family,
        "study_id": manifest.study_id,
        "classification": manifest.classification,
        "status": result_status,
        "created_artifacts": {name: str(path) for name, path in paths.items()},
        "disabled_artifacts": {
            "production_labels": "small-panel benchmark is sandbox-only",
            "production_definition": "final production winner freeze is out of scope",
            "heavy_parallel_tuning": "strictly bounded benchmark",
        },
        "artifact_boundary": {
            "benchmark_only": True,
            "synthetic_panel": True,
            "production_outputs_written": False,
            "production_writes_enabled": False,
            "parquet_writes_enabled": False,
            "label_generation_changed": False,
        },
    }


def _promotion_gate(
    *,
    manifest: StudyManifest,
    family: str,
    scoreboard: Mapping[str, Any],
    paths: Mapping[str, Path],
) -> dict[str, Any]:
    return evaluate_promotion_gate(
        PromotionGateInput(
            gate_id=f"small_panel_{family}_promotion_gate",
            study_key=manifest.study_key,
            scoreboard=scoreboard,
            artifact_kind=SMALL_PANEL_FAMILY_ARTIFACT_KIND,
            artifact_classification=RegimeClassification.SANDBOX,
            artifact_metadata={
                "status": "sandbox_only",
                "sandbox": True,
                "diagnostics_only": True,
                "production_outputs_written": False,
                "study_runner": "src.regimes.studies.small_panel_benchmark",
            },
            artifact_paths={name: str(path) for name, path in paths.items()},
            run_status=RunStatus.SUCCEEDED,
        )
    ).as_dict()


def _asset_flat_policies(
    *,
    clean_frame: pd.DataFrame,
    selected_columns: Sequence[str],
    labels: Sequence[int],
    manifest: StudyManifest,
    family: str,
) -> dict[str, Any]:
    label_arr = np.asarray(labels, dtype=int)
    policies: dict[str, Any] = {}
    for asset in sorted(str(value) for value in clean_frame["asset"].dropna().unique()):
        mask = clean_frame["asset"].astype(str).to_numpy() == asset
        policies[asset] = evaluate_flat_asset_policy(
            clean_frame.loc[mask],
            selected_columns,
            labels=label_arr[mask],
            layer=manifest.layer,
            axis=manifest.axis,
            band=manifest.band,
            source_metadata={"asset": asset, "clusterer_family": family, "benchmark": "small_panel"},
        ).as_dict()
    return policies


def _family_result(
    *,
    family: str,
    family_index: int,
    manifest: StudyManifest,
    train_frame: pd.DataFrame,
    score_frame: pd.DataFrame,
    seed: int,
    root: Path,
) -> dict[str, Any]:
    feature_spec = default_feature_family_registry().get(manifest.feature_families[0])
    preprocessing = fit_preprocessing_pipeline(
        train_frame,
        feature_spec.required_source_columns,
        preprocess=manifest.preprocessing_options[0],
        fit_window={"split_policy": manifest.split_policy, "role": "train", "train_rows": int(len(train_frame))},
        fit_window_role="train",
    )
    score_matrix = transform_score_window_preprocessor(score_frame, preprocessing.fitted, window_role="score")
    clusterer = default_clusterer_registry().build(family, **_clusterer_params(family, seed=seed))
    fit_result = clusterer.fit(preprocessing.fitted.x)
    assign_result = clusterer.assign(score_matrix.x)
    status = "completed" if fit_result.status == FIT_STATUS_FITTED else "failed"
    scoreboard = build_regime_scoreboard(
        trial_id=f"small_panel_{family}",
        clusterer_family=family,
        labels=fit_result.labels,
        features=preprocessing.fitted.x,
        fit_result=fit_result,
        assignment_result=assign_result,
        stability_perturbations=_stability_perturbations(fit_result.labels),
        forward_frame=preprocessing.fitted.clean_frame,
        tiny_cluster_threshold=int(manifest.budget["tiny_cluster_threshold"]),
        metadata={"study_id": manifest.study_id, "benchmark": "small_panel", "family_order": family_index},
    ).as_dict()
    flat_overall = evaluate_flat_asset_policy(
        preprocessing.fitted.clean_frame,
        preprocessing.fitted.selected_columns,
        labels=fit_result.labels,
        layer=manifest.layer,
        axis=manifest.axis,
        band=manifest.band,
        source_metadata={"clusterer_family": family, "benchmark": "small_panel", "scope": "panel_train"},
    ).as_dict()
    flat_by_asset = _asset_flat_policies(
        clean_frame=preprocessing.fitted.clean_frame,
        selected_columns=preprocessing.fitted.selected_columns,
        labels=fit_result.labels,
        manifest=manifest,
        family=family,
    )
    paths = _family_paths(root, family)
    gate = _promotion_gate(manifest=manifest, family=family, scoreboard=scoreboard, paths=paths)
    artifact_manifest = _artifact_manifest(manifest=manifest, family=family, paths=paths, result_status=status)
    result = {
        "schema_version": SMALL_PANEL_BENCHMARK_SCHEMA_VERSION,
        "artifact_kind": "regime_small_panel_benchmark_family_result",
        "family_name": family,
        "family_order": int(family_index),
        "status": status,
        "hyperparameters": _clusterer_params(family, seed=seed),
        "fit_result": fit_result.as_dict(),
        "assignment_result": assign_result.as_dict(),
        "scoreboard": scoreboard,
        "flat_asset_policy": {"overall": flat_overall, "by_asset": flat_by_asset},
        "promotion_gate": gate,
        "artifact_manifest": artifact_manifest,
        "preprocessing": preprocessing.as_dict(),
        "artifact_paths": {name: str(path) for name, path in paths.items()},
    }
    return result


def _ranking(results: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result in results:
        scoreboard = result["scoreboard"]
        coverage = scoreboard["sections"]["coverage_degeneracy"]["metrics"]
        score = _score_value(scoreboard)
        rows.append(
            {
                "family_name": result["family_name"],
                "rank_score": score,
                "fit_status": result["fit_result"]["status"],
                "assignment_status": result["assignment_result"]["status"],
                "effective_state_count": coverage["effective_state_count"],
                "one_cluster_flag": coverage["one_cluster_flag"],
                "all_noise_flag": coverage["all_noise_flag"],
                "promotion_gate_status": result["promotion_gate"]["status"],
                "family_order": result["family_order"],
            }
        )
    return sorted(
        rows,
        key=lambda row: (
            0 if row["fit_status"] == FIT_STATUS_FITTED else 1,
            -float(row["rank_score"]),
            -int(row["effective_state_count"] or 0),
            int(row["family_order"]),
            str(row["family_name"]),
        ),
    )


def _markdown_summary(payload: Mapping[str, Any]) -> str:
    rows = [
        "# Regime Small-Panel Benchmark",
        "",
        f"- Run: `{payload['run_id']}`",
        f"- Seed: `{payload['seed']}`",
        f"- Scope: `{payload['study_manifest']['layer']} / {payload['study_manifest']['axis']} / {payload['study_manifest']['band']}`",
        f"- Families: `{', '.join(payload['candidate_families'])}`",
        f"- Promotion gates blocked: `{payload['all_promotion_gates_blocked']}`",
        "",
        "| Rank | Family | Score | Fit | Assign | States | Gate |",
        "| --- | --- | ---: | --- | --- | ---: | --- |",
    ]
    for idx, row in enumerate(payload["ranking"], start=1):
        rows.append(
            f"| {idx} | `{row['family_name']}` | {float(row['rank_score']):.6f} | "
            f"`{row['fit_status']}` | `{row['assignment_status']}` | "
            f"{row['effective_state_count']} | `{row['promotion_gate_status']}` |"
        )
    rows.extend(
        [
            "",
            "This benchmark is deterministic, synthetic, sandbox-only, and not a production winner freeze.",
        ]
    )
    return "\n".join(rows)


@dataclass(frozen=True)
class SmallPanelBenchmarkResult:
    summary: Mapping[str, Any]
    artifact_paths: Mapping[str, str]
    schema_version: int = SMALL_PANEL_BENCHMARK_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "schema_version", require_schema_version(self.schema_version))
        object.__setattr__(self, "summary", require_json_mapping(self.summary, field_name="small_panel_benchmark summary"))
        object.__setattr__(self, "artifact_paths", require_json_mapping(self.artifact_paths, field_name="small_panel_benchmark artifact_paths"))

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "summary": to_jsonable(self.summary),
            "artifact_paths": dict(self.artifact_paths),
        }

    def to_json(self, **kwargs: Any) -> str:
        return dumps_json(self.as_dict(), **kwargs)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SmallPanelBenchmarkResult":
        obj = require_json_object(payload, context="Regime SmallPanelBenchmarkResult")
        return cls(
            schema_version=obj.get("schema_version", SMALL_PANEL_BENCHMARK_SCHEMA_VERSION),
            summary=obj["summary"],
            artifact_paths=obj["artifact_paths"],
        )

    @classmethod
    def from_json(cls, text: str) -> "SmallPanelBenchmarkResult":
        return cls.from_dict(require_json_object(loads_json(text), context="Regime SmallPanelBenchmarkResult JSON"))


def run_small_panel_benchmark(
    *,
    report_root: str | Path = DEFAULT_SMALL_PANEL_BENCHMARK_ROOT,
    run_id: str = "small_panel_benchmark",
    seed: int = 17,
    panel: pd.DataFrame | None = None,
    write_outputs: bool = True,
    project_root: str | Path | None = None,
) -> SmallPanelBenchmarkResult:
    root = _validate_benchmark_report_root(report_root, project_root=project_root)
    run_token = require_non_empty_string(run_id, field_name="small-panel benchmark run_id")
    frame = panel.copy() if panel is not None else synthetic_small_panel(seed=int(seed))
    candidate_families = _candidate_families()
    manifest = small_panel_study_manifest(report_root=root, seed=int(seed), candidate_families=candidate_families)
    train_frame, score_frame = _split_panel(frame, manifest)
    family_results = [
        _family_result(
            family=family,
            family_index=idx,
            manifest=manifest,
            train_frame=train_frame,
            score_frame=score_frame,
            seed=int(seed),
            root=root,
        )
        for idx, family in enumerate(candidate_families, start=1)
    ]
    ranking = _ranking(family_results)
    summary_json_path = _safe_child(root, SMALL_PANEL_SUMMARY_JSON)
    summary_md_path = _safe_child(root, SMALL_PANEL_SUMMARY_MD)
    artifact_paths = {
        "summary_json": str(summary_json_path),
        "summary_markdown": str(summary_md_path),
        **{
            f"{result['family_name']}_{name}": path
            for result in family_results
            for name, path in result["artifact_paths"].items()
        },
    }
    summary = {
        "schema_version": SMALL_PANEL_BENCHMARK_SCHEMA_VERSION,
        "artifact_kind": SMALL_PANEL_BENCHMARK_ARTIFACT_KIND,
        "status": "completed",
        "run_id": run_token,
        "seed": int(seed),
        "study_manifest": manifest.as_dict(),
        "search_space": build_search_space(manifest).as_dict(),
        "panel_metadata": {
            "row_count": int(len(frame)),
            "asset_count": int(frame["asset"].nunique()),
            "assets": sorted(str(asset) for asset in frame["asset"].unique()),
            "behaviors": sorted(str(value) for value in frame["panel_behavior"].unique()),
            "train_row_count": int(len(train_frame)),
            "score_row_count": int(len(score_frame)),
        },
        "candidate_families": list(candidate_families),
        "ranking": ranking,
        "family_results": family_results,
        "all_promotion_gates_blocked": all(result["promotion_gate"]["status"] == PROMOTION_STATUS_BLOCKED for result in family_results),
        "near_flat_policy_statuses": {
            result["family_name"]: result["flat_asset_policy"]["by_asset"]["FLATUSD"]["status"]
            for result in family_results
        },
        "artifact_paths": artifact_paths,
        "artifact_boundary": {
            "benchmark_only": True,
            "synthetic_panel": True,
            "production_outputs_written": False,
            "production_writes_enabled": False,
            "production_labels_written": False,
            "production_definitions_written": False,
            "final_winner_freeze": False,
            "heavy_parallel_tuning": False,
        },
    }
    if not summary["all_promotion_gates_blocked"]:
        raise RuntimeError("Regime small-panel benchmark promotion gates must remain blocked")
    if write_outputs:
        for result in family_results:
            paths = {name: Path(path) for name, path in result["artifact_paths"].items()}
            _write_json(paths["scoreboard"], result["scoreboard"])
            _write_json(paths["promotion_gate"], result["promotion_gate"])
            _write_json(paths["flat_policy"], result["flat_asset_policy"])
            _write_json(paths["artifact_manifest"], result["artifact_manifest"])
            _write_json(paths["result"], result)
        _write_json(summary_json_path, summary)
        _write_markdown(summary_md_path, _markdown_summary(summary))
    return SmallPanelBenchmarkResult(summary=summary, artifact_paths=artifact_paths)


__all__ = [
    "BENCHMARK_FAMILIES",
    "DEFAULT_SMALL_PANEL_BENCHMARK_ROOT",
    "SMALL_PANEL_BENCHMARK_ARTIFACT_KIND",
    "SMALL_PANEL_BENCHMARK_SCHEMA_VERSION",
    "SMALL_PANEL_FAMILY_ARTIFACT_KIND",
    "SMALL_PANEL_SUMMARY_JSON",
    "SMALL_PANEL_SUMMARY_MD",
    "SmallPanelBenchmarkResult",
    "run_small_panel_benchmark",
    "small_panel_study_manifest",
    "synthetic_small_panel",
]
