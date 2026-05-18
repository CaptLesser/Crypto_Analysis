from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd

from src.forecasting.common.forecast_family_core import discover_edge_and_min, read_feature_window_columns
from src.forecasting.common.stats_module_utils import NUMERIC_TASK_TO_TARGET_COLUMN
from src.forecasting.ml.shared.numeric_forecast_targets import compute_future_labels
from src.forecasting.ml.shared.test_branch_function_telemetry import emit_event_for_path, telemetry_scope_for_path
from src.forecasting.common.ohlcvt_source import read_ohlcvt
from src.forecasting.ml.shared.stage1_candidate_universe import LEGACY_DYNAMIC_FEATURE_ALIASES, RAW_SOURCE_COLUMNS


_STAGE1_SCORE_NEAR_TIE_TOLERANCE = 1e-12


@dataclass
class _StreamingFeatureStats:
    n: int = 0
    sum_x: float = 0.0
    sum_y: float = 0.0
    sum_x2: float = 0.0
    sum_y2: float = 0.0
    sum_xy: float = 0.0

    def add(self, x: np.ndarray, y: np.ndarray) -> None:
        if x.size <= 0:
            return
        self.n += int(x.size)
        self.sum_x += float(x.sum())
        self.sum_y += float(y.sum())
        self.sum_x2 += float(np.dot(x, x))
        self.sum_y2 += float(np.dot(y, y))
        self.sum_xy += float(np.dot(x, y))

    def x_variance_num(self) -> float:
        if self.n <= 0:
            return 0.0
        n = float(self.n)
        return float(self.sum_x2 - (self.sum_x * self.sum_x / n))

    def y_variance_num(self) -> float:
        if self.n <= 0:
            return 0.0
        n = float(self.n)
        return float(self.sum_y2 - (self.sum_y * self.sum_y / n))

    def corr(self) -> float | None:
        if self.n <= 1:
            return None
        n = float(self.n)
        cov_num = self.sum_xy - (self.sum_x * self.sum_y / n)
        x_var_num = self.x_variance_num()
        y_var_num = self.y_variance_num()
        denom = math.sqrt(max(0.0, x_var_num) * max(0.0, y_var_num))
        if denom <= 0.0:
            return None
        corr = cov_num / denom
        if not np.isfinite(corr):
            return None
        return float(corr)

    def score(self) -> float | None:
        corr = self.corr()
        return None if corr is None else float(abs(corr))


@dataclass(frozen=True)
class Stage1DynamicSelectionReport:
    selected_features: tuple[str, ...]
    target_column: str
    selection_method: str
    candidate_feature_count: int
    raw_source_candidate_count: int
    scalar_candidate_count: int
    scored_feature_count: int
    dropped_candidates: tuple[Dict[str, Any], ...]
    feature_scores: tuple[Dict[str, Any], ...]
    redundancy_groups: tuple[Dict[str, Any], ...]
    stability_summary: Dict[str, Any]
    resolved_roots: Dict[str, str]

    def to_artifact(self) -> Dict[str, Any]:
        return {
            "selected_features": list(self.selected_features),
            "target_column": str(self.target_column),
            "selection_method": str(self.selection_method),
            "candidate_feature_count": int(self.candidate_feature_count),
            "raw_source_candidate_count": int(self.raw_source_candidate_count),
            "scalar_candidate_count": int(self.scalar_candidate_count),
            "scored_feature_count": int(self.scored_feature_count),
            "dropped_candidates": [dict(record) for record in self.dropped_candidates],
            "feature_scores": [dict(record) for record in self.feature_scores],
            "redundancy_groups": [dict(record) for record in self.redundancy_groups],
            "stability_summary": dict(self.stability_summary),
            "resolved_roots": dict(self.resolved_roots),
        }


def _horizon_bars(horizon_minutes: int, interval_minutes: int) -> int:
    hm = int(horizon_minutes)
    iv = int(interval_minutes)
    if hm <= 0 or iv <= 0 or hm % iv != 0:
        raise ValueError(f"invalid horizon/interval pair: horizon={hm} interval={iv}")
    return hm // iv


def _candidate_pool(requested_feature_names: Sequence[str]) -> List[str]:
    pool: List[str] = []
    for feature_name in requested_feature_names:
        name = str(feature_name).strip()
        if not name:
            continue
        if name not in pool:
            pool.append(name)
        for alias in LEGACY_DYNAMIC_FEATURE_ALIASES.get(name, ()):
            alias_name = str(alias).strip()
            if alias_name and alias_name not in pool:
                pool.append(alias_name)
    return pool


def _split_candidate_pool(candidate_pool: Sequence[str]) -> tuple[List[str], List[str]]:
    raw_candidates = [str(col) for col in candidate_pool if str(col) in set(RAW_SOURCE_COLUMNS)]
    scalar_candidates = [str(col) for col in candidate_pool if str(col) not in set(RAW_SOURCE_COLUMNS)]
    return scalar_candidates, raw_candidates


def _resolve_actual_feature_name(requested_feature_name: str, available_columns: Iterable[str]) -> str | None:
    available = {str(column) for column in available_columns if str(column)}
    requested = str(requested_feature_name).strip()
    if requested in available:
        return requested
    for alias in LEGACY_DYNAMIC_FEATURE_ALIASES.get(requested, ()):
        alias_name = str(alias).strip()
        if alias_name in available:
            return alias_name
    return None


def _finite_corr(x: np.ndarray, y: np.ndarray) -> float | None:
    if x.size <= 1 or y.size <= 1 or x.size != y.size:
        return None
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) <= 1:
        return None
    xx = x[mask]
    yy = y[mask]
    x_var = float(np.var(xx))
    y_var = float(np.var(yy))
    if x_var <= 1e-18 or y_var <= 1e-18:
        return None
    corr = float(np.corrcoef(xx, yy)[0, 1])
    if not np.isfinite(corr):
        return None
    return corr


def _rank_corr_from_sample(sample: Dict[tuple[str, int], tuple[float, float]], min_rows: int) -> float | None:
    if len(sample) < int(min_rows):
        return None
    ordered = [sample[key] for key in sorted(sample)]
    x = np.asarray([pair[0] for pair in ordered], dtype=float)
    y = np.asarray([pair[1] for pair in ordered], dtype=float)
    if x.size < int(min_rows):
        return None
    x_rank = pd.Series(x).rank(method="average").to_numpy(dtype=float)
    y_rank = pd.Series(y).rank(method="average").to_numpy(dtype=float)
    corr = _finite_corr(x_rank, y_rank)
    return None if corr is None else float(abs(corr))


def _add_feature_samples(
    samples: Dict[str, Dict[tuple[str, int], tuple[float, float]]],
    *,
    feature_name: str,
    asset_name: str,
    ts_values: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    mask: np.ndarray,
    max_sample_rows: int,
) -> None:
    store = samples.setdefault(str(feature_name), {})
    remaining = int(max_sample_rows) - len(store)
    if remaining <= 0:
        return
    valid_idx = np.flatnonzero(mask)
    if valid_idx.size <= 0:
        return
    if valid_idx.size > remaining:
        take_positions = np.unique(np.linspace(0, valid_idx.size - 1, remaining, dtype=int))
        valid_idx = valid_idx[take_positions]
    for idx in valid_idx:
        ts_raw = ts_values[idx] if idx < ts_values.size else idx
        if pd.isna(ts_raw):
            ts_key = int(idx)
        else:
            ts_key = int(ts_raw)
        store[(str(asset_name), ts_key)] = (float(x[idx]), float(y[idx]))


def _add_time_fold_scores(
    fold_scores: Dict[str, List[float]],
    *,
    feature_name: str,
    x: np.ndarray,
    y: np.ndarray,
    min_feature_rows: int,
) -> None:
    if x.size <= 1:
        return
    min_fold_rows = max(8, int(min_feature_rows) // 4)
    for fold_idx in np.array_split(np.arange(x.size), 3):
        if fold_idx.size < min_fold_rows:
            continue
        corr = _finite_corr(x[fold_idx], y[fold_idx])
        if corr is not None:
            fold_scores.setdefault(str(feature_name), []).append(float(corr))


def _sequence_proxy_score(x: np.ndarray, y: np.ndarray, min_feature_rows: int) -> float | None:
    if x.size < max(8, int(min_feature_rows) // 2):
        return None
    window = min(8, max(2, x.size // 4))
    min_periods = min(3, window)
    rolling_mean = pd.Series(x).rolling(window=window, min_periods=min_periods).mean().to_numpy(dtype=float)
    rolling_delta = pd.Series(x).diff().rolling(window=window, min_periods=min_periods).mean().to_numpy(dtype=float)
    candidates: List[float] = []
    for proxy in (rolling_mean, rolling_delta):
        corr = _finite_corr(proxy, y)
        if corr is not None:
            candidates.append(abs(float(corr)))
    if not candidates:
        return None
    return float(max(candidates))


def _stability_metrics(scores: Sequence[float]) -> Dict[str, Any]:
    finite = [float(score) for score in scores if np.isfinite(float(score))]
    if not finite:
        return {
            "fold_score_count": 0,
            "fold_abs_mean": None,
            "fold_score_std": None,
            "fold_sign_consistency": None,
            "stable_fold_count": 0,
            "stability_score": None,
            "stable": False,
        }
    abs_values = [abs(score) for score in finite]
    positive = sum(1 for score in finite if score > 1e-12)
    negative = sum(1 for score in finite if score < -1e-12)
    directional = positive + negative
    sign_consistency = None if directional == 0 else float(max(positive, negative) / directional)
    fold_abs_mean = float(np.mean(abs_values))
    stability_score = fold_abs_mean * (sign_consistency if sign_consistency is not None else 0.0)
    stable_fold_count = int(sum(1 for value in abs_values if value >= 0.05))
    return {
        "fold_score_count": int(len(finite)),
        "fold_abs_mean": fold_abs_mean,
        "fold_score_std": float(np.std(finite)) if len(finite) > 1 else 0.0,
        "fold_sign_consistency": sign_consistency,
        "stable_fold_count": stable_fold_count,
        "stability_score": float(stability_score),
        "stable": bool(len(finite) >= 2 and stable_fold_count >= 2 and (sign_consistency or 0.0) >= 0.67),
    }


def _composite_score(
    *,
    pearson_abs: float | None,
    rank_corr_abs: float | None,
    stability_score: float | None,
    sequence_proxy_score: float | None,
    is_neural_family: bool,
) -> float | None:
    if is_neural_family:
        weighted = (
            (pearson_abs, 0.35),
            (rank_corr_abs, 0.30),
            (stability_score, 0.20),
            (sequence_proxy_score, 0.15),
        )
    else:
        weighted = (
            (pearson_abs, 0.45),
            (rank_corr_abs, 0.35),
            (stability_score, 0.20),
        )
    total_weight = 0.0
    score = 0.0
    for value, weight in weighted:
        if value is None or not np.isfinite(float(value)):
            continue
        total_weight += float(weight)
        score += float(value) * float(weight)
    if total_weight <= 0.0:
        return None
    return float(score / total_weight)


def _redundancy_groups(
    feature_scores: Sequence[Dict[str, Any]],
    samples: Dict[str, Dict[tuple[str, int], tuple[float, float]]],
    *,
    max_features: int,
    min_feature_rows: int,
    threshold: float = 0.95,
) -> List[Dict[str, Any]]:
    candidates = [str(record["feature"]) for record in feature_scores[: max(25, int(max_features) * 4)]]
    groups: List[Dict[str, Any]] = []
    assigned: set[str] = set()
    min_overlap = max(16, int(min_feature_rows) // 2)
    for idx, representative in enumerate(candidates):
        if representative in assigned:
            continue
        rep_sample = samples.get(representative, {})
        members: List[Dict[str, Any]] = []
        for member in candidates[idx + 1 :]:
            if member in assigned:
                continue
            member_sample = samples.get(member, {})
            common_keys = sorted(set(rep_sample).intersection(member_sample))
            if len(common_keys) < min_overlap:
                continue
            rep_values = np.asarray([rep_sample[key][0] for key in common_keys], dtype=float)
            member_values = np.asarray([member_sample[key][0] for key in common_keys], dtype=float)
            corr = _finite_corr(rep_values, member_values)
            if corr is None or abs(float(corr)) < float(threshold):
                continue
            members.append(
                {
                    "feature": member,
                    "abs_correlation": float(abs(corr)),
                    "overlap_rows": int(len(common_keys)),
                }
            )
            assigned.add(member)
        if members:
            groups.append(
                {
                    "representative": representative,
                    "members": members,
                    "threshold": float(threshold),
                }
            )
    return groups


def _empty_report(
    *,
    target_col: str,
    candidate_pool: Sequence[str],
    raw_candidate_pool: Sequence[str],
    dropped_candidates: Sequence[Dict[str, Any]],
    source_roots: Dict[str, str],
    reason: str,
) -> Stage1DynamicSelectionReport:
    return Stage1DynamicSelectionReport(
        selected_features=(),
        target_column=str(target_col),
        selection_method=str(reason),
        candidate_feature_count=int(len(candidate_pool)),
        raw_source_candidate_count=int(len(raw_candidate_pool)),
        scalar_candidate_count=int(len(candidate_pool) - len(raw_candidate_pool)),
        scored_feature_count=0,
        dropped_candidates=tuple(dict(record) for record in dropped_candidates),
        feature_scores=(),
        redundancy_groups=(),
        stability_summary={"stable_feature_count": 0, "features_with_fold_scores": 0},
        resolved_roots=dict(source_roots),
    )


def _maybe_report(
    selected: Sequence[str],
    report: Stage1DynamicSelectionReport,
    return_report: bool,
) -> List[str] | Stage1DynamicSelectionReport:
    if return_report:
        return report
    return [str(feature) for feature in selected]


def _sort_metric(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    return out if np.isfinite(out) else float(default)


def _feature_score_sort_key(record: Dict[str, Any]) -> tuple[float, float, int, float, int, int, str]:
    return (
        -_sort_metric(record.get("composite_score")),
        -_sort_metric(record.get("stability_score")),
        -int(record.get("stable_fold_count") or 0),
        -_sort_metric(record.get("fold_abs_mean")),
        -int(record.get("finite_rows") or 0),
        int(record.get("missing_row_estimate") or 0),
        str(record.get("feature") or ""),
    )


def _scores_need_near_tie_boundary_tiebreak(feature_scores: Sequence[Dict[str, Any]], max_features: int) -> bool:
    selected_count = max(1, int(max_features))
    if len(feature_scores) <= 1 or selected_count >= len(feature_scores):
        return False
    selected_boundary = _sort_metric(feature_scores[selected_count - 1].get("composite_score"), default=float("nan"))
    first_unselected = _sort_metric(feature_scores[selected_count].get("composite_score"), default=float("nan"))
    if not (np.isfinite(selected_boundary) and np.isfinite(first_unselected)):
        return False
    return abs(selected_boundary - first_unselected) <= _STAGE1_SCORE_NEAR_TIE_TOLERANCE


def select_stage1_dynamic_feature_columns(
    *,
    parquet_root: Path,
    asset_list: Sequence[str],
    interval_minutes: int,
    horizon_minutes: int,
    task: str,
    training_window_months: int,
    requested_feature_names: Sequence[str],
    max_features: int = 8,
    min_feature_rows: int = 64,
    telemetry_path: Path | None = None,
    family: str = "",
    model: str = "",
    stage: str = "stage1",
    combo_key: str | None = None,
    return_report: bool = False,
) -> List[str] | Stage1DynamicSelectionReport:
    started = time.perf_counter()
    resolved_root = Path(parquet_root).expanduser().resolve()
    source_roots = {
        "ohlcvt_root": str(resolved_root),
        "scalar_feature_root": str(resolved_root),
        "edge_discovery_root": str(resolved_root),
        "target_label_root": str(resolved_root),
    }
    base_event = {
        "family": family,
        "model": model,
        "stage": stage,
        "combo_key": combo_key,
        "interval_minutes": int(interval_minutes),
        "horizon_minutes": int(horizon_minutes),
        "task": str(task),
        "module_name": __name__,
        "asset_count": len(asset_list),
        "training_window_months": int(training_window_months),
        **source_roots,
    }
    requested = [str(name).strip() for name in requested_feature_names if str(name).strip()]
    candidate_pool = _candidate_pool(requested)
    feature_pool, raw_candidate_pool = _split_candidate_pool(candidate_pool)
    if not requested or not asset_list:
        target_col = str(NUMERIC_TASK_TO_TARGET_COLUMN.get(str(task), ""))
        report = _empty_report(
            target_col=target_col,
            candidate_pool=candidate_pool,
            raw_candidate_pool=raw_candidate_pool,
            dropped_candidates=(
                [
                    {
                        "feature": str(name),
                        "requested_feature": str(name),
                        "reason": "not_evaluated_no_assets" if not asset_list else "not_evaluated_no_requested_features",
                    }
                    for name in requested
                ]
            ),
            source_roots=source_roots,
            reason="skipped_no_assets" if not asset_list else "skipped_no_requested_features",
        )
        emit_event_for_path(
            telemetry_path,
            **base_event,
            function_name="select_stage1_dynamic_feature_columns",
            phase_name="feature_selection",
            status="skipped",
            reason_code="no_assets" if not asset_list else "required_columns_missing",
            elapsed_seconds=time.perf_counter() - started,
            output_rows=0,
            output_columns_count=0,
            candidate_feature_count=len(requested),
            selected_feature_count=0,
            dynamic_feature_count=0,
            source_path=str(resolved_root),
        )
        return _maybe_report([], report, bool(return_report))

    edge_candidates = []
    with telemetry_scope_for_path(
        telemetry_path,
        **base_event,
        function_name="discover_edge_and_min",
        phase_name="maturity_planning",
        parent_phase="feature_selection",
        source_path=str(resolved_root),
    ) as scope:
        for asset_name in asset_list:
            edge_ts, _min_ts = discover_edge_and_min(
                asset=str(asset_name),
                interval_minutes=int(interval_minutes),
                root=resolved_root,
            )
            if edge_ts is not None:
                edge_candidates.append(int(edge_ts))
        scope.set_output(output_rows=len(edge_candidates), reason_code=("no_mature_assets" if not edge_candidates else ""))
    if not edge_candidates:
        target_col = str(NUMERIC_TASK_TO_TARGET_COLUMN[str(task)])
        dropped = [
            {
                "feature": str(name),
                "requested_feature": str(name),
                "reason": "not_evaluated_no_mature_assets",
            }
            for name in requested
        ]
        report = _empty_report(
            target_col=target_col,
            candidate_pool=candidate_pool,
            raw_candidate_pool=raw_candidate_pool,
            dropped_candidates=dropped,
            source_roots=source_roots,
            reason="skipped_no_mature_assets",
        )
        return _maybe_report([], report, bool(return_report))

    common_edge_ts = min(edge_candidates)
    history_start_ts = int(common_edge_ts) - int(training_window_months) * 31 * 86400
    ohlc_columns = list(
        dict.fromkeys(
            [
                "ts",
                "asset",
                "high",
                "low",
                "close",
                *[col for col in RAW_SOURCE_COLUMNS if col in set(raw_candidate_pool)],
            ]
        )
    )
    target_col = str(NUMERIC_TASK_TO_TARGET_COLUMN[str(task)])
    horizon_bars = _horizon_bars(int(horizon_minutes), int(interval_minutes))
    per_feature_stats: Dict[str, _StreamingFeatureStats] = {}
    resolved_by_requested: Dict[str, str] = {}
    finite_rows_by_requested: Dict[str, int] = {}
    samples: Dict[str, Dict[tuple[str, int], tuple[float, float]]] = {}
    fold_scores: Dict[str, List[float]] = {}
    sequence_scores: Dict[str, List[float]] = {}
    feature_assets: Dict[str, set[str]] = {}
    is_neural_family = "neural" in str(family).lower()
    max_sample_rows = 5000

    for asset_name in asset_list:
        asset_event = {**base_event, "asset": str(asset_name)}
        with telemetry_scope_for_path(
            telemetry_path,
            **asset_event,
            function_name="read_ohlcvt",
            phase_name="source_read",
            parent_phase="feature_selection",
            source_path=str(resolved_root),
        ) as scope:
            ohlc_frame = read_ohlcvt(
                root=resolved_root,
                asset=str(asset_name),
                interval_min=int(interval_minutes),
                start_ts=int(history_start_ts),
                end_ts=int(common_edge_ts),
                columns=ohlc_columns,
            )
            scope.set_output(ohlc_frame, reason_code=("source_read_empty" if ohlc_frame.empty else ""))
        if ohlc_frame.empty:
            continue
        with telemetry_scope_for_path(
            telemetry_path,
            **asset_event,
            function_name="read_feature_window_columns",
            phase_name="feature_load",
            parent_phase="feature_selection",
            source_path=str(resolved_root),
        ) as scope:
            feature_frame = read_feature_window_columns(
                root=resolved_root,
                interval_minutes=int(interval_minutes),
                asset=str(asset_name),
                columns=feature_pool,
                start_ts=int(history_start_ts),
                end_ts=int(common_edge_ts),
            )
            scope.set_output(feature_frame, reason_code=("feature_load_empty" if feature_frame.empty else ""))
        with telemetry_scope_for_path(
            telemetry_path,
            input_obj=ohlc_frame,
            required_columns=["ts", "asset", "high", "low", "close"],
            key_columns=["ts", "asset"],
            **asset_event,
            function_name="merge_feature_frame",
            phase_name="join",
            parent_phase="feature_selection",
        ) as scope:
            if feature_frame.empty and not {"ts", "asset"}.issubset(set(feature_frame.columns)):
                feature_frame = ohlc_frame.loc[:, ["ts", "asset"]].copy()
            merged = ohlc_frame.merge(feature_frame, on=["ts", "asset"], how="left", sort=True).reset_index(drop=True)
            scope.set_output(merged, reason_code=("join_empty" if merged.empty else ""))
        with telemetry_scope_for_path(
            telemetry_path,
            input_obj=merged,
            required_columns=["high", "low", "close"],
            **asset_event,
            function_name="compute_future_labels",
            phase_name="label_build",
            parent_phase="feature_selection",
        ) as scope:
            labels, _stats = compute_future_labels(
                merged.loc[:, ["high", "low", "close"]].reset_index(drop=True),
                int(horizon_bars),
                future_direction_deadzone=0.0,
                target_columns=[target_col],
            )
            scope.set_output(labels, reason_code=("labels_empty" if labels.empty else ""))
        if target_col in merged.columns and target_col in labels.columns:
            merged = merged.drop(columns=[target_col])
        merged = pd.concat([merged, labels.reset_index(drop=True)], axis=1)
        if target_col not in merged.columns:
            emit_event_for_path(
                telemetry_path,
                **asset_event,
                function_name="select_stage1_dynamic_feature_columns",
                phase_name="feature_selection",
                parent_phase="label_build",
                status="skipped",
                reason_code="required_columns_missing",
                input_rows=len(merged),
                output_rows=0,
                required_columns_missing=[target_col],
            )
            continue
        y = pd.to_numeric(merged[target_col], errors="coerce").to_numpy(dtype=float)
        ts_values = pd.to_numeric(merged["ts"], errors="coerce").to_numpy(dtype=float) if "ts" in merged.columns else np.arange(len(merged), dtype=float)
        for requested_name in requested:
            actual_name = _resolve_actual_feature_name(requested_name, merged.columns)
            if actual_name is None or actual_name not in merged.columns:
                continue
            resolved_by_requested.setdefault(str(requested_name), str(actual_name))
            x = pd.to_numeric(merged[actual_name], errors="coerce").to_numpy(dtype=float)
            mask = np.isfinite(x) & np.isfinite(y)
            finite_rows = int(mask.sum())
            finite_rows_by_requested[str(requested_name)] = finite_rows_by_requested.get(str(requested_name), 0) + finite_rows
            if finite_rows <= 0:
                continue
            xx = x[mask]
            yy = y[mask]
            per_feature_stats.setdefault(str(actual_name), _StreamingFeatureStats()).add(xx, yy)
            feature_assets.setdefault(str(actual_name), set()).add(str(asset_name))
            _add_feature_samples(
                samples,
                feature_name=str(actual_name),
                asset_name=str(asset_name),
                ts_values=ts_values,
                x=x,
                y=y,
                mask=mask,
                max_sample_rows=max_sample_rows,
            )
            _add_time_fold_scores(
                fold_scores,
                feature_name=str(actual_name),
                x=xx,
                y=yy,
                min_feature_rows=int(min_feature_rows),
            )
            if is_neural_family:
                sequence_score = _sequence_proxy_score(xx, yy, int(min_feature_rows))
                if sequence_score is not None:
                    sequence_scores.setdefault(str(actual_name), []).append(float(sequence_score))

    dropped_candidates: List[Dict[str, Any]] = []
    seen_actuals: set[str] = set()
    for requested_name in requested:
        requested_key = str(requested_name)
        actual_name = resolved_by_requested.get(requested_key)
        source_type = "raw_source" if requested_key in set(RAW_SOURCE_COLUMNS) or actual_name in set(RAW_SOURCE_COLUMNS) else "scalar_derived"
        base_drop = {
            "requested_feature": requested_key,
            "feature": str(actual_name or requested_key),
            "source_type": source_type,
            "finite_rows": int(finite_rows_by_requested.get(requested_key, 0)),
        }
        if actual_name is None:
            dropped_candidates.append({**base_drop, "reason": "missing_from_loaded_frame"})
            continue
        stats = per_feature_stats.get(str(actual_name))
        if stats is None or int(stats.n) <= 0:
            dropped_candidates.append({**base_drop, "reason": "no_finite_feature_target_rows"})
            continue
        if int(stats.n) < int(min_feature_rows):
            dropped_candidates.append({**base_drop, "reason": "insufficient_finite_rows", "required_rows": int(min_feature_rows)})
            continue
        if stats.x_variance_num() <= 1e-12:
            dropped_candidates.append({**base_drop, "reason": "constant_or_near_constant"})
            continue
        if stats.y_variance_num() <= 1e-12:
            dropped_candidates.append({**base_drop, "reason": "target_constant_or_near_constant"})
            continue
        seen_actuals.add(str(actual_name))

    feature_score_records: List[Dict[str, Any]] = []
    for feature_name, stats in per_feature_stats.items():
        if str(feature_name) not in seen_actuals:
            continue
        pearson = stats.corr()
        pearson_abs = None if pearson is None else float(abs(pearson))
        rank_abs = _rank_corr_from_sample(samples.get(str(feature_name), {}), min_rows=max(8, min(int(min_feature_rows), max_sample_rows)))
        stability = _stability_metrics(fold_scores.get(str(feature_name), ()))
        stability_score = stability.get("stability_score")
        sequence_abs = None
        if is_neural_family:
            neural_scores = [float(score) for score in sequence_scores.get(str(feature_name), ()) if np.isfinite(float(score))]
            if neural_scores:
                sequence_abs = float(np.mean(neural_scores))
        composite = _composite_score(
            pearson_abs=pearson_abs,
            rank_corr_abs=rank_abs,
            stability_score=None if stability_score is None else float(stability_score),
            sequence_proxy_score=sequence_abs,
            is_neural_family=is_neural_family,
        )
        if composite is None:
            dropped_candidates.append(
                {
                    "requested_feature": str(feature_name),
                    "feature": str(feature_name),
                    "source_type": "raw_source" if str(feature_name) in set(RAW_SOURCE_COLUMNS) else "scalar_derived",
                    "finite_rows": int(stats.n),
                    "reason": "unscorable_relationship",
                }
            )
            continue
        feature_score_records.append(
            {
                "feature": str(feature_name),
                "source_type": "raw_source" if str(feature_name) in set(RAW_SOURCE_COLUMNS) else "scalar_derived",
                "composite_score": float(composite),
                "pearson_corr": None if pearson is None else float(pearson),
                "pearson_abs": pearson_abs,
                "rank_corr_abs": rank_abs,
                "sequence_proxy_score": sequence_abs,
                "finite_rows": int(stats.n),
                "finite_asset_count": int(len(feature_assets.get(str(feature_name), set()))),
                "x_variance_num": float(stats.x_variance_num()),
                "y_variance_num": float(stats.y_variance_num()),
                **stability,
            }
        )

    if not feature_score_records:
        report = _empty_report(
            target_col=target_col,
            candidate_pool=candidate_pool,
            raw_candidate_pool=raw_candidate_pool,
            dropped_candidates=dropped_candidates,
            source_roots=source_roots,
            reason="filtered_no_scored_features",
        )
        emit_event_for_path(
            telemetry_path,
            **base_event,
            function_name="select_stage1_dynamic_feature_columns",
            phase_name="feature_selection",
            status="completed",
            reason_code="feature_load_empty",
            elapsed_seconds=time.perf_counter() - started,
            input_columns_count=len(candidate_pool),
            output_columns_count=0,
            input_rows=sum(int(stats.n) for stats in per_feature_stats.values()),
            output_rows=0,
            candidate_feature_count=len(candidate_pool),
            raw_source_candidate_count=len(raw_candidate_pool),
            selected_feature_count=0,
            dynamic_feature_count=0,
            dropped_candidate_count=len(dropped_candidates),
            quality_reason="filtered_no_scored_features",
        )
        return _maybe_report([], report, bool(return_report))

    max_finite_rows = max(int(record.get("finite_rows") or 0) for record in feature_score_records)
    for record in feature_score_records:
        finite_rows = int(record.get("finite_rows") or 0)
        record["missing_row_estimate"] = int(max(0, max_finite_rows - finite_rows))
    feature_score_records.sort(key=_feature_score_sort_key)
    near_tie_boundary = _scores_need_near_tie_boundary_tiebreak(feature_score_records, int(max_features))
    scored_features: List[tuple[str, float, int]] = [
        (str(record["feature"]), float(record["composite_score"]), int(record["finite_rows"]))
        for record in feature_score_records
    ]
    selected: List[str] = []
    for feature_name, _score, _rows in scored_features:
        if feature_name not in selected:
            selected.append(feature_name)
        if len(selected) >= max(1, int(max_features)):
            break
    redundancy_groups = _redundancy_groups(
        feature_score_records,
        samples,
        max_features=int(max_features),
        min_feature_rows=int(min_feature_rows),
    )
    stable_feature_count = sum(1 for record in feature_score_records if bool(record.get("stable")))
    stability_summary = {
        "stable_feature_count": int(stable_feature_count),
        "features_with_fold_scores": int(sum(1 for record in feature_score_records if int(record.get("fold_score_count") or 0) > 0)),
        "folded_top_candidate_count": int(len(feature_score_records[: max(1, int(max_features))])),
    }
    selection_method = "composite_pearson_rank_fold_stability"
    if is_neural_family:
        selection_method = f"{selection_method}_sequence_proxy"
    if near_tie_boundary:
        selection_method = f"{selection_method}_near_tie_deterministic_boundary_fallback"
    report = Stage1DynamicSelectionReport(
        selected_features=tuple(selected),
        target_column=str(target_col),
        selection_method=selection_method,
        candidate_feature_count=int(len(candidate_pool)),
        raw_source_candidate_count=int(len(raw_candidate_pool)),
        scalar_candidate_count=int(len(candidate_pool) - len(raw_candidate_pool)),
        scored_feature_count=int(len(feature_score_records)),
        dropped_candidates=tuple(dict(record) for record in dropped_candidates),
        feature_scores=tuple(dict(record) for record in feature_score_records),
        redundancy_groups=tuple(dict(record) for record in redundancy_groups),
        stability_summary=stability_summary,
        resolved_roots=dict(source_roots),
    )
    emit_event_for_path(
        telemetry_path,
        **base_event,
        function_name="select_stage1_dynamic_feature_columns",
        phase_name="feature_selection",
        status="completed",
        elapsed_seconds=time.perf_counter() - started,
        input_columns_count=len(candidate_pool),
        output_columns_count=len(selected),
        input_rows=sum(int(rows) for _feature, _score, rows in scored_features),
        output_rows=len(selected),
        candidate_feature_count=len(candidate_pool),
        raw_source_candidate_count=len(raw_candidate_pool),
        scored_feature_count=len(feature_score_records),
        dropped_candidate_count=len(dropped_candidates),
        redundancy_group_count=len(redundancy_groups),
        selected_feature_count=len(selected),
        dynamic_feature_count=len(selected),
        warning_count=1 if near_tie_boundary else 0,
        quality_reason=selection_method,
    )
    return _maybe_report(selected, report, bool(return_report))
