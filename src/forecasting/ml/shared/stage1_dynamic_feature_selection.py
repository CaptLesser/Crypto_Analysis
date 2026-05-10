from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd

from src.forecasting.common.forecast_family_core import discover_edge_and_min, read_feature_window_columns
from src.forecasting.common.stats_module_utils import NUMERIC_TASK_TO_TARGET_COLUMN
from src.forecasting.ml.shared.numeric_forecast_targets import compute_future_labels
from src.forecasting.ml.shared.test_branch_function_telemetry import emit_event_for_path, telemetry_scope_for_path
from src.forecasting.common.ohlcvt_source import read_ohlcvt


# These aliases reflect current scalar_features schema names. They are intentionally
# conservative and only cover the legacy dynamic feature ids used by Bayes/Neural
# Stage-1 profiles.
LEGACY_DYNAMIC_FEATURE_ALIASES: Dict[str, tuple[str, ...]] = {
    "macd": ("macd_12_26_9",),
    "macd_signal": ("macd_signal_12_26_9",),
    "macd_hist": ("macd_hist_12_26_9",),
    "zscore_30": ("zscore_20",),
    "zscore_60": ("zscore_20",),
    "ret_std_14": ("ret_std_20",),
    "ret_std_30": ("ret_std_20",),
    "ret_std_60": ("ret_std_20",),
    "ret_std_120": ("ret_std_20",),
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
) -> List[str]:
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
    }
    requested = [str(name).strip() for name in requested_feature_names if str(name).strip()]
    if not requested or not asset_list:
        emit_event_for_path(
            telemetry_path,
            **base_event,
            function_name="select_stage1_dynamic_feature_columns",
            phase_name="feature_selection",
            status="skipped",
            reason_code="no_assets" if not asset_list else "required_columns_missing",
            output_rows=0,
            output_columns_count=0,
            source_path=str(parquet_root),
        )
        return []

    edge_candidates = []
    with telemetry_scope_for_path(
        telemetry_path,
        **base_event,
        function_name="discover_edge_and_min",
        phase_name="maturity_planning",
        parent_phase="feature_selection",
        source_path=str(parquet_root),
    ) as scope:
        for asset_name in asset_list:
            edge_ts, _min_ts = discover_edge_and_min(asset=str(asset_name), interval_minutes=int(interval_minutes))
            if edge_ts is not None:
                edge_candidates.append(int(edge_ts))
        scope.set_output(output_rows=len(edge_candidates), reason_code=("no_mature_assets" if not edge_candidates else ""))
    if not edge_candidates:
        return []

    common_edge_ts = min(edge_candidates)
    history_start_ts = int(common_edge_ts) - int(training_window_months) * 31 * 86400
    feature_pool = _candidate_pool(requested)
    target_col = str(NUMERIC_TASK_TO_TARGET_COLUMN[str(task)])
    horizon_bars = _horizon_bars(int(horizon_minutes), int(interval_minutes))
    per_feature_pairs: DefaultDict[str, List[tuple[float, float]]] = defaultdict(list)

    for asset_name in asset_list:
        asset_event = {**base_event, "asset": str(asset_name)}
        with telemetry_scope_for_path(
            telemetry_path,
            **asset_event,
            function_name="read_ohlcvt",
            phase_name="source_read",
            parent_phase="feature_selection",
            source_path=str(parquet_root),
        ) as scope:
            ohlc_frame = read_ohlcvt(
                root=Path(parquet_root),
                asset=str(asset_name),
                interval_min=int(interval_minutes),
                start_ts=int(history_start_ts),
                end_ts=int(common_edge_ts),
                columns=["ts", "asset", "high", "low", "close"],
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
            source_path=str(parquet_root),
        ) as scope:
            feature_frame = read_feature_window_columns(
                root=Path(parquet_root),
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
        for requested_name in requested:
            actual_name = _resolve_actual_feature_name(requested_name, merged.columns)
            if actual_name is None or actual_name not in merged.columns:
                continue
            x = pd.to_numeric(merged[actual_name], errors="coerce").to_numpy(dtype=float)
            mask = np.isfinite(x) & np.isfinite(y)
            if int(mask.sum()) < int(min_feature_rows):
                continue
            xx = x[mask]
            yy = y[mask]
            if np.unique(xx).size <= 1 or np.unique(yy).size <= 1:
                continue
            per_feature_pairs[actual_name].extend((float(xv), float(yv)) for xv, yv in zip(xx, yy))

    scored_features: List[tuple[str, float, int]] = []
    for feature_name, pairs in per_feature_pairs.items():
        if len(pairs) < int(min_feature_rows):
            continue
        xvals = np.asarray([pair[0] for pair in pairs], dtype=float)
        yvals = np.asarray([pair[1] for pair in pairs], dtype=float)
        if xvals.size < int(min_feature_rows) or np.unique(xvals).size <= 1 or np.unique(yvals).size <= 1:
            continue
        corr = np.corrcoef(xvals, yvals)[0, 1]
        if not np.isfinite(corr):
            continue
        scored_features.append((str(feature_name), float(abs(corr)), int(xvals.size)))

    if not scored_features:
        fallback_selected: List[str] = []
        for requested_name in requested:
            actual_name = _resolve_actual_feature_name(requested_name, per_feature_pairs.keys())
            if actual_name is not None and actual_name not in fallback_selected:
                fallback_selected.append(actual_name)
        selected = fallback_selected[: max(1, int(max_features))]
        emit_event_for_path(
            telemetry_path,
            **base_event,
            function_name="select_stage1_dynamic_feature_columns",
            phase_name="feature_selection",
            status="completed",
            reason_code=("feature_load_empty" if not selected else ""),
            input_columns_count=len(feature_pool),
            output_columns_count=len(selected),
        )
        return selected

    scored_features.sort(key=lambda item: (-item[1], -item[2], item[0]))
    selected: List[str] = []
    for feature_name, _score, _rows in scored_features:
        if feature_name not in selected:
            selected.append(feature_name)
        if len(selected) >= max(1, int(max_features)):
            break
    emit_event_for_path(
        telemetry_path,
        **base_event,
        function_name="select_stage1_dynamic_feature_columns",
        phase_name="feature_selection",
        status="completed",
        input_columns_count=len(feature_pool),
        output_columns_count=len(selected),
    )
    return selected
