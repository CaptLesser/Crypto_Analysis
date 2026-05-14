from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


def _band_name(band: object) -> str:
    return str(getattr(band, "name", band))


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    return out if math.isfinite(out) else float(default)


def _series_to_float_dict(series: pd.Series) -> Dict[str, float]:
    return {str(k): _safe_float(v) for k, v in series.items()}


def _pairwise_corr_matrix(df: pd.DataFrame, cols: Sequence[str]) -> Dict[str, Dict[str, float]]:
    if not cols:
        return {}
    try:
        corr = df[list(cols)].astype(float).corr().fillna(0.0)
    except Exception:
        return {}
    out: Dict[str, Dict[str, float]] = {}
    for row_name in corr.index:
        row: Dict[str, float] = {}
        for col_name in corr.columns:
            row[str(col_name)] = _safe_float(corr.loc[row_name, col_name])
        out[str(row_name)] = row
    return out


def _hist_quantile_from_counts(counts: Sequence[int], q: float) -> float:
    if not counts:
        return 0.0
    total = int(sum(int(x) for x in counts))
    if total <= 0:
        return 0.0
    q = float(min(1.0, max(0.0, q)))
    target = int(math.ceil(q * total))
    if target <= 0:
        target = 1
    csum = 0
    for idx, count in enumerate(counts):
        csum += int(count)
        if csum >= target:
            return float(idx)
    return float(len(counts) - 1)


class RegimeDiagnosticCollector:
    def __init__(self, asset: str):
        self.asset = str(asset)
        self._rows: Dict[Tuple[str, str], Dict[str, Any]] = {}

    def _entry(self, band: str, category: str) -> Dict[str, Any]:
        key = (str(band), str(category))
        if key not in self._rows:
            self._rows[key] = {
                "asset": self.asset,
                "band": str(band),
                "category": str(category),
                "bars_total": 0,
                "flat_bar_count": 0,
                "centroid_candidate_rows": 0,
                "overridden_to_neutral_rows": 0,
                "final_unknown_rows": 0,
                "centroid_assigned_rows": 0,
                "centroid_left_unknown_rows": 0,
                "no_model_or_no_cluster_rows": 0,
                "incomplete_feature_rows": 0,
                "confidence_count": 0,
                "confidence_sum": 0.0,
                "confidence_hist": [0] * 101,
                "feature_variance": {},
                "feature_std": {},
                "pairwise_correlation_matrix": {},
                "clustering": {
                    "clusters_found": 0,
                    "training_noise_fraction": 1.0,
                    "cluster_sizes": {},
                },
            }
        return self._rows[key]

    def record_fit(
        self,
        band: object,
        category: str,
        train_df: pd.DataFrame,
        feature_cols: Sequence[str],
        model: Optional[dict[str, Any]],
    ) -> None:
        entry = self._entry(_band_name(band), category)
        if feature_cols and not train_df.empty:
            cols_present = [c for c in feature_cols if c in train_df.columns]
            if cols_present:
                numeric = train_df[cols_present].astype(float)
                entry["feature_variance"] = _series_to_float_dict(numeric.var(ddof=0))
                entry["feature_std"] = _series_to_float_dict(numeric.std(ddof=0))
                entry["pairwise_correlation_matrix"] = _pairwise_corr_matrix(numeric, cols_present)
        if model is None:
            entry["clustering"] = {
                "clusters_found": 0,
                "training_noise_fraction": 1.0,
                "cluster_sizes": {},
            }
            return
        train_meta = model.get("train_meta", {}) if isinstance(model, dict) else {}
        labels_raw = train_meta.get("labels", np.array([], dtype=int))
        labels = np.asarray(labels_raw, dtype=int) if labels_raw is not None else np.array([], dtype=int)
        n = int(labels.size)
        noise_count = int(np.sum(labels == -1)) if n > 0 else 0
        cluster_sizes: Dict[str, int] = {}
        for cid in sorted({int(v) for v in labels.tolist() if int(v) != -1}):
            cluster_sizes[str(cid)] = int(np.sum(labels == cid))
        entry["clustering"] = {
            "clusters_found": int(len(cluster_sizes)),
            "training_noise_fraction": float(noise_count / n) if n > 0 else 1.0,
            "cluster_sizes": cluster_sizes,
        }

    def record_assign(
        self,
        band: object,
        category: str,
        bars_total: int,
        flat_bar_count: int,
        centroid_candidate_rows: int,
        overridden_to_neutral_rows: int,
        final_unknown_rows: int,
        centroid_assigned_rows: int = 0,
        centroid_left_unknown_rows: int = 0,
        no_model_or_no_cluster_rows: int = 0,
        incomplete_feature_rows: int = 0,
        confidence_values: Optional[np.ndarray] = None,
    ) -> None:
        entry = self._entry(_band_name(band), category)
        entry["bars_total"] += int(bars_total)
        entry["flat_bar_count"] += int(flat_bar_count)
        entry["centroid_candidate_rows"] += int(centroid_candidate_rows)
        entry["overridden_to_neutral_rows"] += int(overridden_to_neutral_rows)
        entry["final_unknown_rows"] += int(final_unknown_rows)
        entry["centroid_assigned_rows"] += int(centroid_assigned_rows)
        entry["centroid_left_unknown_rows"] += int(centroid_left_unknown_rows)
        entry["no_model_or_no_cluster_rows"] += int(no_model_or_no_cluster_rows)
        entry["incomplete_feature_rows"] += int(incomplete_feature_rows)
        if confidence_values is not None and int(len(confidence_values)) > 0:
            vals = np.asarray(confidence_values, dtype=float)
            vals = vals[np.isfinite(vals)]
            if vals.size > 0:
                vals = np.clip(np.rint(vals), 0, 100).astype(int)
                hist = np.bincount(vals, minlength=101)
                entry["confidence_count"] += int(vals.size)
                entry["confidence_sum"] += float(vals.sum())
                cur_hist = entry.get("confidence_hist", [0] * 101)
                if not isinstance(cur_hist, list) or len(cur_hist) != 101:
                    cur_hist = [0] * 101
                entry["confidence_hist"] = [int(cur_hist[i]) + int(hist[i]) for i in range(101)]

    def to_json_ready(self) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for (_band, _category), entry in sorted(self._rows.items(), key=lambda kv: (kv[0][0], kv[0][1])):
            bars_total = int(entry["bars_total"])
            if bars_total <= 0:
                continue
            conf_count = int(entry.get("confidence_count", 0))
            conf_sum = float(entry.get("confidence_sum", 0.0))
            conf_hist = entry.get("confidence_hist", [0] * 101)
            if not isinstance(conf_hist, list) or len(conf_hist) != 101:
                conf_hist = [0] * 101
            rows.append(
                {
                    "asset": entry["asset"],
                    "band": entry["band"],
                    "category": entry["category"],
                    "bars_total": bars_total,
                    "flat_bar_fraction": float(entry["flat_bar_count"] / bars_total),
                    "feature_variance": dict(entry.get("feature_variance", {})),
                    "feature_std": dict(entry.get("feature_std", {})),
                    "pairwise_correlation_matrix": dict(entry.get("pairwise_correlation_matrix", {})),
                    "clustering": dict(entry.get("clustering", {})),
                    "assignments": {
                        "centroid_candidate_fraction": float(entry["centroid_candidate_rows"] / bars_total),
                        "overridden_to_neutral_fraction": float(entry["overridden_to_neutral_rows"] / bars_total),
                        "final_unknown_fraction": float(entry["final_unknown_rows"] / bars_total),
                        "centroid_assigned_fraction": float(entry["centroid_assigned_rows"] / bars_total),
                        "centroid_left_unknown_fraction": float(entry["centroid_left_unknown_rows"] / bars_total),
                        "no_model_or_no_cluster_fraction": float(entry["no_model_or_no_cluster_rows"] / bars_total),
                        "incomplete_feature_fraction": float(entry["incomplete_feature_rows"] / bars_total),
                        "mean_confidence": float(conf_sum / conf_count) if conf_count > 0 else 0.0,
                        "median_confidence": _hist_quantile_from_counts(conf_hist, 0.50) if conf_count > 0 else 0.0,
                        "p10_confidence": _hist_quantile_from_counts(conf_hist, 0.10) if conf_count > 0 else 0.0,
                        "p90_confidence": _hist_quantile_from_counts(conf_hist, 0.90) if conf_count > 0 else 0.0,
                    },
                }
            )
        return rows


DiagnosticCollector = RegimeDiagnosticCollector
