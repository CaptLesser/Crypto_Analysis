from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from src.forecasting.common.io_atomic import atomic_replace, sibling_temp_path
from src.regimes.contracts import REGIME_AXIS_ORDER, REGIME_BANDS
from src.regimes.core.flat_preflight import (
    FLAT_STATUS_ACTIVE,
    FLAT_STATUS_AXIS_NOT_CLUSTERABLE,
    FLAT_STATUS_INSUFFICIENT_DATA,
    FLAT_STATUS_NEAR_FLAT_MORE_EVIDENCE,
    FLAT_STATUS_VALID_SINGLE_STATE,
    FlatPeggedPreflightResult,
    run_flat_pegged_preflight,
)
from src.regimes.core.paths import (
    is_production_adjacent_path,
    is_relative_to,
    resolve_project_path,
    resolve_project_root,
)


CLUSTERABILITY_SCHEMA_VERSION = 1
CLUSTERABILITY_ARTIFACT_KIND = "regime_asset_state_clusterability_preflight"
LABELS_ARTIFACT_KIND = "regime_asset_state_clusterability_labels"
DISCOVERY_ROOT_PARTS = ("reports", "regimes", "foundation", "discovery")
PANEL_STRATIFIED = "stratified"
PATHWAY_ASSET_STATE = "asset_state"

COHORT_ASSETS: Mapping[str, tuple[str, ...]] = {
    "liquid_major": ("XBTUSD",),
    "mid_liquidity": ("AAVEUSD",),
    "high_beta": ("ADAUSD",),
    "near_flat": ("TERMUSD",),
    "sparse_problematic": ("TEERUSD",),
}
COHORT_ORDER: tuple[str, ...] = tuple(COHORT_ASSETS)
BAND_ORDER: tuple[str, ...] = tuple(REGIME_BANDS)

AXIS_FEATURES: Mapping[str, tuple[str, ...]] = {
    "trend": ("log_return", "macd_hist_12_26_9", "rsi_14", "adx_14"),
    "vol": ("log_return", "atr_14", "ret_std_20", "cv_20", "vol_osc_pct_14_28"),
    "activity": ("log_return", "trade_intensity", "avg_trade_size", "vroc_14", "prr"),
}
AXIS_MOVEMENT_COLUMNS: Mapping[str, tuple[str, ...]] = {
    "trend": ("log_return", "macd_hist_12_26_9"),
    "vol": ("log_return", "atr_14", "ret_std_20"),
    "activity": ("log_return", "vroc_14", "prr"),
}
AXIS_ACTIVITY_COLUMNS: Mapping[str, tuple[str, ...]] = {
    "trend": (),
    "vol": (),
    "activity": ("trade_intensity", "avg_trade_size", "vroc_14"),
}
BAND_BASE_ROWS: Mapping[str, int] = {"micro": 72, "meso": 48, "macro": 36}


@dataclass(frozen=True)
class ClusterabilityRunResult:
    payload: Mapping[str, Any]
    rows: tuple[Mapping[str, Any], ...]
    artifact_paths: Mapping[str, str]


def _now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _jsonable(value: Any) -> Any:
    if hasattr(value, "as_dict"):
        return _jsonable(value.as_dict())
    if isinstance(value, Mapping):
        return {str(key): _jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, set):
        return sorted(_jsonable(item) for item in value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        out = float(value)
        return out if np.isfinite(out) else None
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    return value


def _stable_seed(*parts: object) -> int:
    digest = hashlib.sha256("|".join(str(part) for part in parts).encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % (2**32)


def _normalize_cohorts(cohorts: Sequence[str]) -> tuple[str, ...]:
    requested = tuple(str(cohort).strip() for cohort in cohorts if str(cohort).strip())
    unknown = sorted(set(requested).difference(COHORT_ASSETS))
    if unknown:
        valid = ", ".join(COHORT_ORDER)
        raise ValueError(f"unsupported clusterability cohort(s): {unknown}; expected one or more of: {valid}")
    selected = requested or COHORT_ORDER
    return tuple(cohort for cohort in COHORT_ORDER if cohort in set(selected))


def _normalize_bands(bands: Sequence[str]) -> tuple[str, ...]:
    requested = tuple(str(band).strip().lower() for band in bands if str(band).strip())
    unknown = sorted(set(requested).difference(REGIME_BANDS))
    if unknown:
        valid = ", ".join(BAND_ORDER)
        raise ValueError(f"unsupported clusterability band(s): {unknown}; expected one or more of: {valid}")
    selected = requested or BAND_ORDER
    return tuple(band for band in BAND_ORDER if band in set(selected))


def _normalize_seeds(seeds: Sequence[int]) -> tuple[int, ...]:
    selected = tuple(sorted({int(seed) for seed in seeds})) or (11,)
    return selected


def _has_discovery_root_parts(path: Path) -> bool:
    parts = tuple(str(part).lower() for part in path.resolve().parts)
    expected = tuple(DISCOVERY_ROOT_PARTS)
    width = len(expected)
    return any(parts[idx : idx + width] == expected for idx in range(max(len(parts) - width + 1, 0)))


def validate_clusterability_write_root(
    write_root: str | Path,
    *,
    project_root: str | Path | None = None,
) -> Path:
    root = resolve_project_path(write_root, project_root=project_root)
    project = resolve_project_root(project_root)
    if is_production_adjacent_path(root, project_root=project):
        raise ValueError("Clusterability preflight write root is production-adjacent and is not allowed")
    if not is_relative_to(root, project):
        raise ValueError("Clusterability preflight write root must stay under the project root")
    if not _has_discovery_root_parts(root):
        raise ValueError("Clusterability preflight write root must be under reports/regimes/foundation/discovery")
    return root


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = sibling_temp_path(path)
    try:
        tmp.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        atomic_replace(tmp, path)
    finally:
        try:
            if tmp.exists():
                tmp.unlink()
        except Exception:
            pass


def _write_text(path: Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = sibling_temp_path(path)
    try:
        tmp.write_text(text.rstrip() + "\n", encoding="utf-8")
        atomic_replace(tmp, path)
    finally:
        try:
            if tmp.exists():
                tmp.unlink()
        except Exception:
            pass


def _asset_records(cohorts: Sequence[str]) -> tuple[dict[str, str], ...]:
    rows: list[dict[str, str]] = []
    for cohort in _normalize_cohorts(cohorts):
        for asset in COHORT_ASSETS[cohort]:
            rows.append({"cohort": str(cohort), "asset": str(asset)})
    return tuple(rows)


def _rows_for(cohort: str, band: str) -> int:
    if cohort == "sparse_problematic":
        return 8
    return int(BAND_BASE_ROWS[str(band)])


def _sigma_for(cohort: str, axis: str) -> float:
    base = {
        "liquid_major": 0.008,
        "mid_liquidity": 0.014,
        "high_beta": 0.045,
        "near_flat": 0.0,
        "sparse_problematic": 0.02,
    }[str(cohort)]
    if axis == "activity":
        return base * 0.75
    if axis == "vol":
        return base * 1.25
    return base


def build_stratified_feature_frame(
    *,
    cohort: str,
    asset: str,
    axis: str,
    band: str,
    seed: int,
) -> pd.DataFrame:
    rows = _rows_for(cohort, band)
    rng = np.random.default_rng(_stable_seed(cohort, asset, axis, band, seed))
    sigma = _sigma_for(cohort, axis)
    if cohort == "near_flat":
        movement = np.zeros(rows, dtype=float)
        low_noise = np.zeros(rows, dtype=float)
    else:
        movement = rng.normal(0.0, sigma, rows)
        low_noise = rng.normal(0.0, max(sigma / 4.0, 1e-6), rows)
    if cohort == "high_beta":
        movement += rng.normal(0.0, sigma, rows)
    trend_level = np.cumsum(movement)
    activity_base = {
        "liquid_major": 140.0,
        "mid_liquidity": 65.0,
        "high_beta": 95.0,
        "near_flat": 0.0,
        "sparse_problematic": 18.0,
    }[str(cohort)]
    frame = pd.DataFrame(
        {
            "ts": np.arange(rows, dtype=np.int64) * int(REGIME_BANDS[str(band)].ceiling_interval_min) * 60,
            "asset": str(asset),
            "log_return": movement,
            "macd_hist_12_26_9": movement * 75.0 + low_noise,
            "rsi_14": 50.0 + np.clip(trend_level * 250.0, -25.0, 25.0),
            "adx_14": np.abs(movement) * 800.0 + np.abs(low_noise) * 20.0,
            "atr_14": np.abs(movement) * 3.0 + np.abs(low_noise),
            "ret_std_20": np.abs(rng.normal(max(sigma, 1e-6), max(sigma / 3.0, 1e-6), rows)),
            "cv_20": np.abs(rng.normal(max(sigma * 3.0, 1e-6), max(sigma / 2.0, 1e-6), rows)),
            "vol_osc_pct_14_28": rng.normal(0.0, max(sigma * 10.0, 1e-6), rows),
            "trade_intensity": activity_base + rng.normal(0.0, max(activity_base * 0.08, 1e-6), rows),
            "avg_trade_size": activity_base / 7.0 + rng.normal(0.0, max(activity_base * 0.01, 1e-6), rows),
            "vroc_14": rng.normal(0.0, max(sigma * 12.0, 1e-6), rows),
            "prr": movement * 2.0 + rng.normal(0.0, max(sigma / 5.0, 1e-6), rows),
        }
    )
    if cohort == "near_flat":
        for column in frame.columns:
            if column not in {"ts", "asset", "rsi_14"}:
                frame[column] = 0.0
        frame["rsi_14"] = 50.0
    if cohort == "sparse_problematic" and rows > 3:
        nan_columns = [column for column in AXIS_FEATURES[str(axis)] if column in frame.columns]
        frame.loc[frame.index[::3], nan_columns] = np.nan
    return frame


def _final_label_for_statuses(statuses: Sequence[str]) -> str:
    status_set = {str(status) for status in statuses}
    if FLAT_STATUS_AXIS_NOT_CLUSTERABLE in status_set:
        return "not_clusterable"
    if status_set and status_set.issubset({FLAT_STATUS_INSUFFICIENT_DATA}):
        return "insufficient_data"
    if status_set.intersection({FLAT_STATUS_VALID_SINGLE_STATE, FLAT_STATUS_NEAR_FLAT_MORE_EVIDENCE}):
        return "flat_fallback"
    if status_set == {FLAT_STATUS_ACTIVE}:
        return "clusterable"
    return "needs_review"


def _status_priority(status: str) -> int:
    return {
        FLAT_STATUS_AXIS_NOT_CLUSTERABLE: 0,
        FLAT_STATUS_INSUFFICIENT_DATA: 1,
        FLAT_STATUS_VALID_SINGLE_STATE: 2,
        FLAT_STATUS_NEAR_FLAT_MORE_EVIDENCE: 3,
        FLAT_STATUS_ACTIVE: 4,
    }.get(str(status), 5)


def _mean(values: Sequence[float | int | None]) -> float | None:
    clean = [float(value) for value in values if value is not None and np.isfinite(float(value))]
    if not clean:
        return None
    return float(np.mean(clean))


def _min(values: Sequence[float | int | None]) -> float | None:
    clean = [float(value) for value in values if value is not None and np.isfinite(float(value))]
    if not clean:
        return None
    return float(min(clean))


def _max(values: Sequence[float | int | None]) -> float | None:
    clean = [float(value) for value in values if value is not None and np.isfinite(float(value))]
    if not clean:
        return None
    return float(max(clean))


def _extract_metrics(result: FlatPeggedPreflightResult) -> dict[str, Any]:
    payload = result.as_dict()
    diagnostics = payload["diagnostics"]
    movement = diagnostics["movement_behavior"]
    diversity = diagnostics["sample_diversity"]
    variance = diagnostics["variance_behavior"]
    return {
        "status": payload["status"],
        "confidence_score": float(payload["confidence_score"]),
        "row_count": int(diagnostics["row_count"]),
        "finite_row_count": diversity.get("finite_row_count"),
        "unique_row_count": diversity.get("unique_row_count"),
        "effective_diversity_share": diversity.get("effective_diversity_share"),
        "duplicate_row_fraction": diversity.get("duplicate_row_fraction"),
        "near_zero_movement_fraction": movement.get("near_zero_movement_fraction"),
        "median_abs_movement": movement.get("median_abs_movement"),
        "missing_fraction": diagnostics.get("missing_fraction"),
        "near_constant_feature_count": variance.get("near_constant_feature_count"),
        "zero_variance_feature_count": variance.get("zero_variance_feature_count"),
        "warnings": payload.get("warnings", []),
        "recommended_action": payload["policy_decision"]["recommended_action"],
    }


def _row_for_asset_axis_band(
    *,
    cohort: str,
    asset: str,
    axis: str,
    band: str,
    seeds: Sequence[int],
) -> dict[str, Any]:
    seed_metrics: list[dict[str, Any]] = []
    for seed in seeds:
        frame = build_stratified_feature_frame(cohort=cohort, asset=asset, axis=axis, band=band, seed=int(seed))
        result = run_flat_pegged_preflight(
            frame,
            asset=asset,
            axis=axis,
            band=band,
            feature_columns=AXIS_FEATURES[str(axis)],
            movement_columns=AXIS_MOVEMENT_COLUMNS[str(axis)],
            activity_columns=AXIS_ACTIVITY_COLUMNS[str(axis)],
            source_metadata={
                "panel": PANEL_STRATIFIED,
                "cohort": str(cohort),
                "synthetic": True,
                "seed": int(seed),
            },
        )
        seed_metrics.append({"seed": int(seed), **_extract_metrics(result)})
    statuses = [str(item["status"]) for item in seed_metrics]
    status_counts = dict(sorted(Counter(statuses).items()))
    primary_status = sorted(status_counts, key=lambda status: (_status_priority(status), status))[0]
    final_label = _final_label_for_statuses(statuses)
    return {
        "pathway": PATHWAY_ASSET_STATE,
        "panel": PANEL_STRATIFIED,
        "cohort": str(cohort),
        "asset": str(asset),
        "axis": str(axis),
        "band": str(band),
        "seeds": [int(seed) for seed in seeds],
        "seed_count": int(len(seeds)),
        "status": str(primary_status),
        "status_counts": status_counts,
        "final_label": final_label,
        "clusterable_candidate": bool(final_label == "clusterable"),
        "flat_fallback_label": "neutral_flat" if final_label == "flat_fallback" else None,
        "row_count_min": int(_min([item["row_count"] for item in seed_metrics]) or 0),
        "row_count_max": int(_max([item["row_count"] for item in seed_metrics]) or 0),
        "finite_row_count_min": (
            None
            if _min([item.get("finite_row_count") for item in seed_metrics]) is None
            else int(_min([item.get("finite_row_count") for item in seed_metrics]) or 0)
        ),
        "unique_row_count_min": (
            None
            if _min([item.get("unique_row_count") for item in seed_metrics]) is None
            else int(_min([item.get("unique_row_count") for item in seed_metrics]) or 0)
        ),
        "effective_diversity_share_min": _min([item.get("effective_diversity_share") for item in seed_metrics]),
        "near_zero_movement_fraction_mean": _mean([item.get("near_zero_movement_fraction") for item in seed_metrics]),
        "near_zero_movement_fraction_max": _max([item.get("near_zero_movement_fraction") for item in seed_metrics]),
        "missing_fraction_max": _max([item.get("missing_fraction") for item in seed_metrics]),
        "near_constant_feature_count_max": (
            None
            if _max([item.get("near_constant_feature_count") for item in seed_metrics]) is None
            else int(_max([item.get("near_constant_feature_count") for item in seed_metrics]) or 0)
        ),
        "zero_variance_feature_count_max": (
            None
            if _max([item.get("zero_variance_feature_count") for item in seed_metrics]) is None
            else int(_max([item.get("zero_variance_feature_count") for item in seed_metrics]) or 0)
        ),
        "confidence_score_min": _min([item.get("confidence_score") for item in seed_metrics]),
        "confidence_score_mean": _mean([item.get("confidence_score") for item in seed_metrics]),
        "seed_metrics": seed_metrics,
    }


def build_clusterability_rows(
    *,
    pathway: str = PATHWAY_ASSET_STATE,
    panel: str = PANEL_STRATIFIED,
    cohorts: Sequence[str] = COHORT_ORDER,
    bands: Sequence[str] = BAND_ORDER,
    seeds: Sequence[int] = (11, 29),
) -> tuple[dict[str, Any], ...]:
    if str(pathway).strip() != PATHWAY_ASSET_STATE:
        raise ValueError("Clusterability preflight currently supports only pathway asset_state")
    if str(panel).strip() != PANEL_STRATIFIED:
        raise ValueError("Clusterability preflight must run only on panel stratified")
    normalized_cohorts = _normalize_cohorts(cohorts)
    normalized_bands = _normalize_bands(bands)
    normalized_seeds = _normalize_seeds(seeds)
    rows: list[dict[str, Any]] = []
    for record in _asset_records(normalized_cohorts):
        for axis in REGIME_AXIS_ORDER:
            for band in normalized_bands:
                rows.append(
                    _row_for_asset_axis_band(
                        cohort=record["cohort"],
                        asset=record["asset"],
                        axis=str(axis),
                        band=str(band),
                        seeds=normalized_seeds,
                    )
                )
    return tuple(rows)


def _summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_label = Counter(str(row["final_label"]) for row in rows)
    by_cohort: dict[str, dict[str, int]] = {}
    for row in rows:
        cohort = str(row["cohort"])
        by_cohort.setdefault(cohort, {})
        label = str(row["final_label"])
        by_cohort[cohort][label] = int(by_cohort[cohort].get(label, 0)) + 1
    return {
        "row_count": int(len(rows)),
        "asset_count": int(len({str(row["asset"]) for row in rows})),
        "axis_count": int(len({str(row["axis"]) for row in rows})),
        "band_count": int(len({str(row["band"]) for row in rows})),
        "final_label_counts": dict(sorted(by_label.items())),
        "cohort_label_counts": {cohort: dict(sorted(counts.items())) for cohort, counts in sorted(by_cohort.items())},
        "production_outputs_written": False,
        "production_labels_written": False,
        "promotion_allowed": False,
    }


def _artifact_paths(root: Path) -> dict[str, str]:
    return {
        "markdown": str(Path(root) / "clusterability_preflight.md"),
        "json": str(Path(root) / "clusterability_preflight.json"),
        "labels_jsonl": str(Path(root) / "clusterability_labels.jsonl"),
    }


def _markdown(payload: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        "# Clusterability Preflight",
        "",
        f"- Pathway: `{payload['pathway']}`",
        f"- Panel: `{payload['panel']}`",
        f"- Cohorts: `{', '.join(payload['cohorts'])}`",
        f"- Bands: `{', '.join(payload['bands'])}`",
        f"- Seeds: `{', '.join(str(seed) for seed in payload['seeds'])}`",
        f"- Rows: `{len(rows)}`",
        "- Production labels written: `false`",
        "- Promotion allowed: `false`",
        "",
        "| Cohort | Asset | Axis | Band | Final Label | Status | Near-Zero Movement Mean | Diversity Min |",
        "| --- | --- | --- | --- | --- | --- | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row['cohort']}`",
                    f"`{row['asset']}`",
                    f"`{row['axis']}`",
                    f"`{row['band']}`",
                    f"`{row['final_label']}`",
                    f"`{row['status']}`",
                    _format_metric(row.get("near_zero_movement_fraction_mean")),
                    _format_metric(row.get("effective_diversity_share_min")),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def _format_metric(value: object) -> str:
    if value is None:
        return "`null`"
    try:
        return f"`{float(value):.6f}`"
    except Exception:
        return f"`{value}`"


def _write_outputs(root: Path, payload: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    paths = _artifact_paths(root)
    _write_json(Path(paths["json"]), payload)
    labels_text = "\n".join(json.dumps(_jsonable(row), sort_keys=True) for row in rows)
    _write_text(Path(paths["labels_jsonl"]), labels_text)
    _write_text(Path(paths["markdown"]), _markdown(payload, rows))
    return paths


def run_clusterability_preflight(
    *,
    pathway: str = PATHWAY_ASSET_STATE,
    panel: str = PANEL_STRATIFIED,
    cohorts: Sequence[str] = COHORT_ORDER,
    bands: Sequence[str] = BAND_ORDER,
    seeds: Sequence[int] = (11, 29),
    write_root: str | Path = "reports/regimes/foundation/discovery",
    no_write: bool = False,
    project_root: str | Path | None = None,
) -> ClusterabilityRunResult:
    normalized_cohorts = _normalize_cohorts(cohorts)
    normalized_bands = _normalize_bands(bands)
    normalized_seeds = _normalize_seeds(seeds)
    rows = build_clusterability_rows(
        pathway=pathway,
        panel=panel,
        cohorts=normalized_cohorts,
        bands=normalized_bands,
        seeds=normalized_seeds,
    )
    root = resolve_project_path(write_root, project_root=project_root)
    paths = _artifact_paths(root)
    payload = {
        "schema_version": CLUSTERABILITY_SCHEMA_VERSION,
        "artifact_kind": CLUSTERABILITY_ARTIFACT_KIND,
        "created_at_utc": _now_utc(),
        "pathway": str(pathway),
        "panel": str(panel),
        "cohorts": list(normalized_cohorts),
        "bands": list(normalized_bands),
        "axes": list(REGIME_AXIS_ORDER),
        "seeds": list(normalized_seeds),
        "summary": _summary(rows),
        "artifact_boundary": {
            "diagnostics_only": True,
            "stratified_panel_only": True,
            "production_outputs_written": False,
            "production_labels_written": False,
            "label_generation_invoked": False,
            "clustering_invoked": False,
            "promotion_allowed": False,
        },
        "row_order": ["cohort_order", "asset", "axis_order", "band_order"],
        "rows": list(rows),
        "artifact_paths": paths if not no_write else {},
    }
    artifact_paths: Mapping[str, str] = {}
    if not no_write:
        root = validate_clusterability_write_root(root, project_root=project_root)
        artifact_paths = _write_outputs(root, payload, rows)
        payload = {**payload, "artifact_paths": dict(artifact_paths)}
    return ClusterabilityRunResult(payload=payload, rows=rows, artifact_paths=artifact_paths)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Deterministic asset-state clusterability preflight for a stratified panel.")
    parser.add_argument("--pathway", default=PATHWAY_ASSET_STATE, choices=(PATHWAY_ASSET_STATE,))
    parser.add_argument("--panel", default=PANEL_STRATIFIED, choices=(PANEL_STRATIFIED,))
    parser.add_argument("--cohorts", nargs="+", default=list(COHORT_ORDER), choices=COHORT_ORDER)
    parser.add_argument("--bands", nargs="+", default=list(BAND_ORDER), choices=BAND_ORDER)
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 29])
    parser.add_argument("--write-root", type=Path, default=Path("reports/regimes/foundation/discovery"))
    parser.add_argument("--no-write", action="store_true", help="Compute and print the summary without writing artifacts.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    result = run_clusterability_preflight(
        pathway=str(args.pathway),
        panel=str(args.panel),
        cohorts=tuple(args.cohorts),
        bands=tuple(args.bands),
        seeds=tuple(args.seeds),
        write_root=Path(args.write_root),
        no_write=bool(args.no_write),
    )
    print(json.dumps(_jsonable({"summary": result.payload["summary"], "artifact_paths": result.artifact_paths}), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CLUSTERABILITY_ARTIFACT_KIND",
    "CLUSTERABILITY_SCHEMA_VERSION",
    "COHORT_ASSETS",
    "ClusterabilityRunResult",
    "build_clusterability_rows",
    "build_stratified_feature_frame",
    "run_clusterability_preflight",
    "validate_clusterability_write_root",
]
