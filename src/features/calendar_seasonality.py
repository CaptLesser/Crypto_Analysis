from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from src.forecasting.common.ohlcvt_source import list_assets_ohlcvt, ohlcvt_bounds, read_ohlcvt
from src.forecasting.common.io_atomic import atomic_replace, sibling_temp_path
from src.forecasting.common.path_config import require_pipeline_io, resolve_path, selected_profile


MODULE_VERSION = "1.2.0"
MODEL_FEATURE_SET_VERSION = "seasonality_feature_set_v1"
MANUAL_CONTEXT_VERSION = "seasonality_manual_context_v1"
PIPELINE_PROFILE = selected_profile()
DEFAULT_PARQUET_ROOT = Path(
    resolve_path("source_ohlcvt_root", profile=PIPELINE_PROFILE, required=False)
    or resolve_path("output_parquet_root", profile=PIPELINE_PROFILE, required=False)
    or Path("parquet")
)
DEFAULT_OUTPUT_ROOT = Path(
    resolve_path("output_parquet_root", profile=PIPELINE_PROFILE, required=False)
    or DEFAULT_PARQUET_ROOT
) / "seasonality"
PARQUET_COMPRESSION = os.getenv("PIPELINE_PARQUET_COMPRESSION", "snappy")

INTERVAL_TO_MIN = {"1H": 60, "4H": 240, "1D": 1440}
SUPPORTED_INTERVALS = tuple(INTERVAL_TO_MIN.keys())
SUPPORTED_INTERVAL_MINS = tuple(INTERVAL_TO_MIN[k] for k in SUPPORTED_INTERVALS)

CANDIDATE_PERIODS = {
    "1H": [24, 168, 720],
    "4H": [6, 42, 180],
    "1D": [7, 30, 365],
}
ASSET_MIN_HISTORY_YEARS = 2.0
DEFAULT_FEATURE_LOOKBACK_DAYS = 84
DEFAULT_FEATURE_MIN_BUCKET_SAMPLES = 4
RATIO_DENOMINATOR_EPS = 1e-12
RATIO_CLIP_MIN = 0.05
RATIO_CLIP_MAX = 20.0
RATIO_EXTREME_MIN = 0.10
RATIO_EXTREME_MAX = 10.0
SESSION_DEFINITIONS_UTC = {
    "Asia": "00:00-07:00 UTC rough crypto activity context",
    "Asia_Europe_overlap": "07:00-09:00 UTC rough overlap context",
    "Europe": "09:00-13:00 UTC rough Europe context",
    "Europe_US_overlap": "13:00-17:00 UTC rough overlap context",
    "US": "17:00-22:00 UTC rough US context",
    "Off_peak": "22:00-24:00 UTC rough lower activity context",
}
SCALAR_SEASONALITY_COLUMNS = [
    "true_range",
    "atr_pct_14",
    "activity_state_score_20",
    "illiquidity_proxy_20",
    "true_range_pct",
    "realized_volatility",
    "ret_std_20",
]
MODEL_FEATURE_COLUMNS = [
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "week_cycle_sin",
    "week_cycle_cos",
    "utc_weekend_flag",
    "is_asia_session",
    "is_europe_session",
    "is_us_session",
    "is_london_ny_overlap",
    "volume_vs_usual_bucket",
    "volume_vs_usual_bucket_clipped",
    "volume_vs_usual_bucket_log_ratio",
    "volume_vs_usual_bucket_quality_flag",
    "trades_vs_usual_bucket",
    "trades_vs_usual_bucket_clipped",
    "trades_vs_usual_bucket_log_ratio",
    "trades_vs_usual_bucket_quality_flag",
    "dollar_volume_vs_usual_bucket",
    "dollar_volume_vs_usual_bucket_clipped",
    "dollar_volume_vs_usual_bucket_log_ratio",
    "dollar_volume_vs_usual_bucket_quality_flag",
    "volatility_vs_usual_bucket",
    "volatility_vs_usual_bucket_clipped",
    "volatility_vs_usual_bucket_log_ratio",
    "volatility_vs_usual_bucket_quality_flag",
    "illiquidity_vs_usual_bucket",
    "illiquidity_vs_usual_bucket_clipped",
    "illiquidity_vs_usual_bucket_log_ratio",
    "illiquidity_vs_usual_bucket_quality_flag",
    "activity_state_vs_usual_bucket",
    "activity_state_vs_usual_bucket_clipped",
    "activity_state_vs_usual_bucket_log_ratio",
    "activity_state_vs_usual_bucket_quality_flag",
    "thin_liquidity_bucket_flag",
    "active_but_stressed_window_flag",
    "bucket_sample_count_asof",
    "bucket_min_samples_required",
    "bucket_sparse_flag",
    "bucket_fallback_used",
    "bucket_fallback_level",
    "seasonality_feature_quality_flag",
]
MANUAL_CONTEXT_COLUMNS = [
    "context_time_utc",
    "matched_artifact_time_utc",
    "context_freshness_status",
    "interval",
    "utc_hour",
    "utc_day_of_week",
    "is_weekend",
    "session_tag",
    "session_overlap_flag",
    "bucket_volume_vs_usual",
    "bucket_volume_vs_usual_clipped",
    "bucket_volume_quality_flag",
    "bucket_trades_vs_usual",
    "bucket_trades_vs_usual_clipped",
    "bucket_trades_quality_flag",
    "bucket_dollar_volume_vs_usual",
    "bucket_volatility_vs_usual",
    "bucket_volatility_vs_usual_clipped",
    "bucket_volatility_quality_flag",
    "bucket_activity_vs_usual",
    "bucket_liquidity_or_illiquidity_vs_usual",
    "thin_window_flag",
    "active_window_flag",
    "active_but_stressed_window_flag",
    "seasonal_volatility_regime_note",
    "seasonality_context_quality",
    "seasonality_context_quality_score",
    "seasonality_context_warnings",
    "bucket_sample_count_asof",
    "bucket_min_samples_required",
    "bucket_min_samples_met",
    "bucket_fallback_used",
    "bucket_fallback_level",
    "bucket_baseline_quality",
    "seasonal_target_confidence_adjustment",
    "seasonal_time_rule_note",
    "seasonality_summary_note",
]


def _boolean_arg(v: str | bool) -> bool:
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean: {v}")


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_date_start_ts(date_s: str) -> int:
    dt = datetime.strptime(date_s, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(dt.timestamp())


def _to_jsonable_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    v = float(x)
    if not np.isfinite(v):
        return None
    return v


def _mad(series: pd.Series) -> float:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return 0.0
    med = float(s.median())
    return float(np.median(np.abs(s.to_numpy(dtype=float) - med)))


def _robust_baseline(series: pd.Series, method: str = "median", trim_ratio: float = 0.1) -> float:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return 0.0
    if method == "trimmed_mean":
        vals = np.sort(s.to_numpy(dtype=float))
        n = len(vals)
        k = int(n * max(0.0, min(0.49, trim_ratio)))
        if (n - 2 * k) <= 0:
            return float(np.mean(vals))
        return float(np.mean(vals[k : n - k]))
    return float(s.median())


def _circular_smooth(values: np.ndarray, window: int) -> np.ndarray:
    n = int(len(values))
    if n <= 1:
        return values.copy()
    w = int(max(1, window))
    if w == 1:
        return values.copy()
    if w > n:
        w = n
    if w % 2 == 0:
        w = max(1, w - 1)
    if w <= 1:
        return values.copy()
    pad = w // 2
    ext = np.concatenate([values[-pad:], values, values[:pad]])
    kernel = np.ones(w, dtype=float) / float(w)
    out = np.convolve(ext, kernel, mode="valid")
    return out[:n]


def _iter_months_between(start_ts: int, end_ts: int) -> Iterable[Tuple[int, int]]:
    if end_ts < start_ts:
        return
    dt = datetime.fromtimestamp(int(start_ts), tz=timezone.utc)
    y = dt.year
    m = dt.month
    while True:
        yield y, m
        if m == 12:
            y += 1
            m = 1
        else:
            m += 1
        nxt = datetime(y, m, 1, tzinfo=timezone.utc)
        if int(nxt.timestamp()) > int(end_ts):
            break


def _min_asset_coverage_bars(interval_min: int, years: float = ASSET_MIN_HISTORY_YEARS) -> int:
    bars_per_day = (24 * 60) / float(interval_min)
    return int(np.ceil(365.25 * years * bars_per_day))


def _coverage_bars_from_bounds(min_ts: int, max_ts: int, interval_min: int) -> int:
    if max_ts < min_ts:
        return 0
    step = int(interval_min * 60)
    return int(((int(max_ts) - int(min_ts)) // step) + 1)


def _ohlc_path(root: Path, interval_min: int, year: int, month: int) -> Path:
    return root / f"ohlcvt_{interval_min}" / f"year={year}" / f"month={month:02d}" / f"part-ohlcvt_{interval_min}-{year}{month:02d}.parquet"


def _scalar_path(root: Path, interval_min: int, year: int, month: int) -> Path:
    return root / f"scalar_features_{interval_min}" / f"year={year}" / f"month={month:02d}" / f"part-scalar_features_{interval_min}-{year}{month:02d}.parquet"


def list_assets_for_interval(parquet_root: Path, interval_min: int) -> List[str]:
    return list_assets_ohlcvt(interval_min=int(interval_min), root=parquet_root)


def asset_bounds_for_interval(parquet_root: Path, interval_min: int) -> Dict[str, Dict[str, int]]:
    out: Dict[str, Dict[str, int]] = {}
    for asset in list_assets_for_interval(parquet_root, interval_min):
        mn, mx = ohlcvt_bounds(interval_min=int(interval_min), asset=str(asset), root=parquet_root)
        if mn is None or mx is None:
            continue
        out[str(asset)] = {"min_ts": int(mn), "max_ts": int(mx)}
    return out

def read_ohlc_asset_window(
    parquet_root: Path,
    interval_min: int,
    asset: str,
    start_ts: int,
    end_ts: int,
) -> pd.DataFrame:
    cols = ["asset", "ts", "high", "low", "close", "volume", "trades"]
    df = read_ohlcvt(
        asset=str(asset),
        interval_min=int(interval_min),
        start_ts=int(start_ts),
        end_ts=int(end_ts),
        columns=cols,
        root=parquet_root,
    )

    if df.empty:
        return pd.DataFrame(columns=cols)
    df["ts"] = pd.to_numeric(df["ts"], errors="coerce")
    for c in ("high", "low", "close", "volume", "trades"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["ts", "high", "low", "close"])
    df = df.sort_values("ts").drop_duplicates(subset=["ts"], keep="last").reset_index(drop=True)
    df["asset"] = asset
    return df


def read_scalar_feature_window(
    parquet_root: Path,
    interval_min: int,
    asset: str,
    start_ts: int,
    end_ts: int,
    requested_columns: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    paths: List[Path] = []
    for y, m in _iter_months_between(start_ts, end_ts):
        p = _scalar_path(parquet_root, interval_min, y, m)
        if p.exists():
            paths.append(p)
    cols_requested = list(dict.fromkeys(["asset", "ts", *(requested_columns or SCALAR_SEASONALITY_COLUMNS)]))
    if not paths:
        return pd.DataFrame(columns=[c for c in cols_requested if c != "asset"])

    try:
        import pyarrow.dataset as ds  # type: ignore

        dataset = ds.dataset([str(p) for p in paths], format="parquet")
        schema_cols = set(dataset.schema.names)
        cols = [c for c in cols_requested if c in schema_cols]
        if "asset" not in cols or "ts" not in cols:
            return pd.DataFrame(columns=[c for c in cols_requested if c != "asset"])
        filt = (
            (ds.field("asset") == asset)
            & (ds.field("ts") >= int(start_ts))
            & (ds.field("ts") <= int(end_ts))
        )
        tbl = dataset.to_table(filter=filt, columns=cols)
        df = tbl.to_pandas()
    except Exception:
        frames: List[pd.DataFrame] = []
        for p in paths:
            try:
                d = pd.read_parquet(p)
            except Exception:
                continue
            present = [c for c in cols_requested if c in d.columns]
            if "asset" not in present or "ts" not in present:
                continue
            d = d[present]
            d = d[(d["asset"] == asset) & (d["ts"] >= int(start_ts)) & (d["ts"] <= int(end_ts))]
            if not d.empty:
                frames.append(d)
        if not frames:
            return pd.DataFrame(columns=[c for c in cols_requested if c != "asset"])
        df = pd.concat(frames, ignore_index=True)

    if df.empty:
        return pd.DataFrame(columns=[c for c in cols_requested if c != "asset"])
    df["ts"] = pd.to_numeric(df["ts"], errors="coerce")
    for c in df.columns:
        if c not in {"asset", "ts"}:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["ts"]).sort_values("ts").drop_duplicates(subset=["ts"], keep="last")
    return df[[c for c in df.columns if c != "asset"]].reset_index(drop=True)


def read_scalar_true_range_window(
    parquet_root: Path,
    interval_min: int,
    asset: str,
    start_ts: int,
    end_ts: int,
) -> pd.DataFrame:
    df = read_scalar_feature_window(
        parquet_root=parquet_root,
        interval_min=interval_min,
        asset=asset,
        start_ts=start_ts,
        end_ts=end_ts,
        requested_columns=["true_range"],
    )
    return df[["ts", "true_range"]] if "true_range" in df.columns else pd.DataFrame(columns=["ts", "true_range"])


def _compute_true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = pd.to_numeric(df["close"], errors="coerce").shift(1)
    high = pd.to_numeric(df["high"], errors="coerce")
    low = pd.to_numeric(df["low"], errors="coerce")
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return pd.to_numeric(tr, errors="coerce")


def _build_calendar_columns(df: pd.DataFrame, interval_label: str) -> pd.DataFrame:
    out = df.copy()
    dt = pd.to_datetime(out["ts"], unit="s", utc=True)
    dow = dt.dt.dayofweek.astype(int)
    out["weekend_vs_weekday"] = (dow >= 5).astype(int)
    out["day_of_week"] = dow
    out["month_of_year"] = (dt.dt.month.astype(int) - 1)
    if interval_label == "1H":
        out["hour_of_week"] = dow * 24 + dt.dt.hour.astype(int)
    elif interval_label == "4H":
        out["hour_of_week"] = dow * 6 + (dt.dt.hour.astype(int) // 4)
    return out


def _component_stability(
    values: pd.Series,
    buckets: pd.Series,
    n_buckets: int,
    baseline: float,
    channel: str,
    wrap: bool,
    smoothing_window: int,
) -> float:
    df = pd.DataFrame({"v": values, "b": buckets}).dropna()
    if len(df) < max(32, n_buckets * 4):
        return 0.0
    n_splits = 4
    split_size = len(df) // n_splits
    if split_size < max(8, n_buckets // 2):
        return 0.0

    vectors: List[np.ndarray] = []
    for i in range(n_splits):
        s = i * split_size
        e = len(df) if i == (n_splits - 1) else (i + 1) * split_size
        d = df.iloc[s:e]
        if d.empty:
            continue
        grp = d.groupby("b")["v"]
        vec = np.full(n_buckets, np.nan, dtype=float)
        med = grp.median()
        for bi, vv in med.items():
            bi_i = int(bi)
            if bi_i < 0 or bi_i >= n_buckets:
                continue
            if channel == "mean":
                vec[bi_i] = float(vv - baseline)
            else:
                denom = baseline if baseline > 1e-12 else np.nan
                vec[bi_i] = float(vv / denom) if np.isfinite(denom) else 1.0
        if channel == "vol":
            vec = np.log(np.clip(np.nan_to_num(vec, nan=1.0, posinf=1.0, neginf=1.0), 1e-9, 1e9))
        else:
            vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
        if wrap and smoothing_window > 1:
            vec = _circular_smooth(vec, smoothing_window)
        vectors.append(vec)

    if len(vectors) < 2:
        return 0.0

    corrs: List[float] = []
    for i in range(len(vectors) - 1):
        a = vectors[i]
        b = vectors[i + 1]
        if np.std(a) < 1e-12 or np.std(b) < 1e-12:
            continue
        c = float(np.corrcoef(a, b)[0, 1])
        if np.isfinite(c):
            corrs.append((c + 1.0) * 0.5)
    if not corrs:
        return 0.0
    return float(np.clip(np.mean(corrs), 0.0, 1.0))


def _use_label(score: float, scope: str) -> str:
    if scope == "global":
        if score >= 0.75:
            return "high"
        if score >= 0.55:
            return "med"
        if score >= 0.35:
            return "low"
        return "none"
    if score >= 0.65:
        return "high"
    if score >= 0.45:
        return "med"
    if score >= 0.25:
        return "low"
    return "none"


def _component_quality(
    signal_to_noise: float,
    stability: float,
    coverage_bars: int,
    n_buckets: int,
    scope: str,
) -> Tuple[float, str, bool]:
    coverage_factor = min(1.0, float(coverage_bars) / float(max(1, n_buckets * 8)))
    snr_score = float(np.clip(signal_to_noise / 2.0, 0.0, 1.0))
    score = float(np.clip((0.45 * snr_score + 0.55 * stability) * coverage_factor, 0.0, 1.0))
    use = _use_label(score, scope)
    usable = use in {"high", "med"}
    return score, use, usable


def _top_k(values: np.ndarray, counts: np.ndarray, channel: str, k: int) -> List[Dict[str, Any]]:
    if len(values) == 0:
        return []
    if channel == "vol":
        mags = np.abs(np.log(np.clip(values, 1e-12, 1e12)))
    else:
        mags = np.abs(values)
    idx = np.argsort(-mags)
    out: List[Dict[str, Any]] = []
    for i in idx:
        if len(out) >= k:
            break
        if int(counts[i]) <= 0:
            continue
        out.append({"bucket": int(i), "value": _to_jsonable_float(values[i]), "bucket_count": int(counts[i])})
    return out

def _compute_component(
    df: pd.DataFrame,
    family: str,
    n_buckets: int,
    channel: str,
    baseline_method: str,
    smoothing_window: int,
    top_k_n: int,
    scope: str,
    apply_smoothing: bool,
) -> Dict[str, Any]:
    v_col = "log_return" if channel == "mean" else "true_range"
    d = df[["ts", family, v_col]].copy()
    d[v_col] = pd.to_numeric(d[v_col], errors="coerce")
    d[family] = pd.to_numeric(d[family], errors="coerce")
    d = d.dropna(subset=[family, v_col])
    if d.empty:
        neutral = 0.0 if channel == "mean" else 1.0
        return {
            "family": family,
            "channel": channel,
            "n_buckets": n_buckets,
            "values": np.full(n_buckets, neutral, dtype=float),
            "bucket_counts": np.zeros(n_buckets, dtype=int),
            "baseline": 0.0,
            "coverage_bars": 0,
            "coverage_start_ts": None,
            "coverage_end_ts": None,
            "signal_to_noise": 0.0,
            "stability_score": 0.0,
            "quality_score": 0.0,
            "recommended_use": "none",
            "usable": False,
            "top_k": [],
        }

    d[family] = d[family].astype(int)
    d = d[(d[family] >= 0) & (d[family] < n_buckets)]
    if d.empty:
        return {
            "family": family,
            "channel": channel,
            "n_buckets": n_buckets,
            "values": np.full(n_buckets, 0.0 if channel == "mean" else 1.0, dtype=float),
            "bucket_counts": np.zeros(n_buckets, dtype=int),
            "baseline": 0.0,
            "coverage_bars": 0,
            "coverage_start_ts": None,
            "coverage_end_ts": None,
            "signal_to_noise": 0.0,
            "stability_score": 0.0,
            "quality_score": 0.0,
            "recommended_use": "none",
            "usable": False,
            "top_k": [],
        }

    baseline = _robust_baseline(d[v_col], method=baseline_method)
    neutral = 0.0 if channel == "mean" else 1.0

    med = d.groupby(family)[v_col].median()
    cnt = d.groupby(family)[v_col].count()

    values = np.full(n_buckets, neutral, dtype=float)
    counts = np.zeros(n_buckets, dtype=int)

    for bi in range(n_buckets):
        c = int(cnt.get(bi, 0))
        counts[bi] = c
        if c <= 0:
            continue
        m = float(med.get(bi, np.nan))
        if channel == "mean":
            values[bi] = m - baseline
        else:
            denom = baseline if abs(baseline) > 1e-12 else np.nan
            values[bi] = (m / denom) if np.isfinite(denom) else 1.0

    wrap = family in {"day_of_week", "month_of_year", "hour_of_week"}
    if apply_smoothing and wrap and smoothing_window > 1:
        if channel == "mean":
            values = _circular_smooth(values, smoothing_window)
        else:
            values = np.exp(_circular_smooth(np.log(np.clip(values, 1e-9, 1e9)), smoothing_window))

    values = np.nan_to_num(values, nan=neutral, posinf=neutral, neginf=neutral)

    bucket_effect = np.zeros(len(d), dtype=float)
    for i, b in enumerate(d[family].to_numpy(dtype=int)):
        bucket_effect[i] = values[b]

    if channel == "mean":
        pred = baseline + bucket_effect
    else:
        pred = baseline * bucket_effect

    resid = d[v_col].to_numpy(dtype=float) - pred
    effect_series = pd.Series(bucket_effect)
    signal = _mad(effect_series)
    noise = _mad(pd.Series(resid))
    snr = float(signal / (noise + 1e-12)) if noise > 0 else (10.0 if signal > 0 else 0.0)

    stability = _component_stability(
        values=d[v_col],
        buckets=d[family],
        n_buckets=n_buckets,
        baseline=baseline,
        channel=channel,
        wrap=wrap,
        smoothing_window=smoothing_window if apply_smoothing else 1,
    )

    quality_score, use_label, usable = _component_quality(
        signal_to_noise=snr,
        stability=stability,
        coverage_bars=int(len(d)),
        n_buckets=n_buckets,
        scope=scope,
    )

    topk = _top_k(values=values, counts=counts, channel=channel, k=top_k_n)

    return {
        "family": family,
        "channel": channel,
        "n_buckets": n_buckets,
        "values": values,
        "bucket_counts": counts,
        "baseline": float(baseline),
        "coverage_bars": int(len(d)),
        "coverage_start_ts": int(d["ts"].min()) if not d.empty else None,
        "coverage_end_ts": int(d["ts"].max()) if not d.empty else None,
        "signal_to_noise": float(snr),
        "stability_score": float(stability),
        "quality_score": float(quality_score),
        "recommended_use": use_label,
        "usable": bool(usable),
        "top_k": topk,
    }


def _lag_autocorr(values: np.ndarray, lag: int) -> float:
    if lag <= 0 or len(values) <= lag:
        return 0.0
    a = values[lag:]
    b = values[:-lag]
    if len(a) < 8:
        return 0.0
    sa = float(np.std(a))
    sb = float(np.std(b))
    if sa < 1e-12 or sb < 1e-12:
        return 0.0
    c = float(np.corrcoef(a, b)[0, 1])
    if not np.isfinite(c):
        return 0.0
    return c


def _spectral_score(values: np.ndarray, period: int) -> float:
    x = values[np.isfinite(values)]
    n = len(x)
    if n < max(64, period * 2):
        return 0.0
    x = x - np.mean(x)
    fft = np.fft.rfft(x)
    power = np.abs(fft) ** 2
    freqs = np.fft.rfftfreq(n, d=1.0)
    if len(freqs) <= 2:
        return 0.0
    target = 1.0 / float(period)
    idx = int(np.argmin(np.abs(freqs - target)))
    lo = max(1, idx - 1)
    hi = min(len(power) - 1, idx + 1)
    local = float(np.sum(power[lo : hi + 1]))
    total = float(np.sum(power[1:]))
    if total <= 1e-12:
        return 0.0
    ratio = local / total
    return float(np.clip(ratio * 10.0, 0.0, 1.0))


def _period_stability(values: np.ndarray, period: int) -> float:
    n = len(values)
    if n < max(period * 6, 96):
        return 0.0
    n_splits = 4
    split_size = n // n_splits
    if split_size < period * 2:
        return 0.0
    acfs: List[float] = []
    for i in range(n_splits):
        s = i * split_size
        e = n if i == (n_splits - 1) else (i + 1) * split_size
        part = values[s:e]
        acfs.append(_lag_autocorr(part, period))
    arr = np.array(acfs, dtype=float)
    if arr.size == 0:
        return 0.0
    presence = float(np.mean(arr > 0.05))
    mean = float(np.mean(arr))
    std = float(np.std(arr))
    consistency = 0.0 if abs(mean) < 1e-9 else float(np.clip(1.0 - (std / (abs(mean) + 1e-9)), 0.0, 1.0))
    return float(np.clip(0.6 * presence + 0.4 * consistency, 0.0, 1.0))


def _period_quality(strength: float, stability: float, coverage_bars: int, period: int, scope: str) -> Tuple[float, str, bool]:
    coverage_factor = min(1.0, float(coverage_bars) / float(max(1, period * 10)))
    score = float(np.clip((0.6 * strength + 0.4 * stability) * coverage_factor, 0.0, 1.0))
    use = _use_label(score, scope)
    usable = use in {"high", "med"}
    return score, use, usable


def detect_period_candidates(
    series: pd.Series,
    interval_label: str,
    scope: str,
    period_channel: str = "vol",
) -> Tuple[List[Dict[str, Any]], Optional[int]]:
    s = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if s.empty:
        return [], None
    x = s.to_numpy(dtype=float)
    x = x - np.mean(x)

    candidates: List[Dict[str, Any]] = []
    for p in CANDIDATE_PERIODS[interval_label]:
        if len(x) < max(96, p * 6):
            continue
        acf = _lag_autocorr(x, p)
        spec = _spectral_score(x, p)
        strength = float(np.clip(0.7 * max(0.0, acf) + 0.3 * spec, 0.0, 1.0))
        stability = _period_stability(x, p)
        q_score, use, usable = _period_quality(strength, stability, len(x), p, scope)
        candidates.append(
            {
                "period_bars": int(p),
                "strength_score": float(strength),
                "stability_score": float(stability),
                "quality_score": float(q_score),
                "recommended_use": use,
                "usable": bool(usable),
                "coverage_bars": int(len(x)),
                "period_channel": period_channel,
                "method": "acf_plus_spectral_with_rolling_stability",
            }
        )

    candidates = sorted(
        candidates,
        key=lambda d: (float(d["quality_score"]), float(d["strength_score"]), float(d["stability_score"])),
        reverse=True,
    )

    recommended_period: Optional[int] = None
    for c in candidates:
        if c["usable"]:
            recommended_period = int(c["period_bars"])
            break

    return candidates, recommended_period


def _recommended_stability_score(candidates: Sequence[Dict[str, Any]], recommended_period: Optional[int]) -> float:
    if not candidates:
        return 0.0
    if recommended_period is not None:
        for c in candidates:
            if int(c.get("period_bars", -1)) == int(recommended_period):
                return float(c.get("stability_score", 0.0))
    return 0.0


def _choose_recommendation_channel(vol_recommended: Optional[int], return_recommended: Optional[int]) -> Tuple[Optional[int], str, Optional[str]]:
    if vol_recommended is not None:
        return int(vol_recommended), "volatility", "vol"
    if return_recommended is not None:
        return int(return_recommended), "return", "return"
    return None, "none", None


@dataclass
class ArtifactResult:
    scope: str
    interval: str
    asset: Optional[str]
    source: str
    source_priority: int
    bucket_components: Dict[Tuple[str, str], Dict[str, Any]]
    period_candidates: List[Dict[str, Any]]
    recommended_period_bars: Optional[int] = None
    vol_period_candidates: List[Dict[str, Any]] = field(default_factory=list)
    vol_recommended_period: Optional[int] = None
    vol_stability_score: float = 0.0
    return_period_candidates: List[Dict[str, Any]] = field(default_factory=list)
    return_recommended_period: Optional[int] = None
    return_stability_score: float = 0.0
    recommendation_channel: str = "none"
    recommended_period_channel: Optional[str] = None
    computed_from_assets_count: Optional[int] = None
    inclusion_rules: Optional[str] = None


def _component_families(interval_label: str) -> Dict[str, int]:
    families = {
        "weekend_vs_weekday": 2,
        "day_of_week": 7,
        "month_of_year": 12,
    }
    if interval_label == "1H":
        families["hour_of_week"] = 168
    elif interval_label == "4H":
        families["hour_of_week"] = 42
    return families

def _prepare_asset_series(
    parquet_root: Path,
    interval_label: str,
    asset: str,
    start_ts: int,
    end_ts: int,
    prefer_scalar_features_true_range: bool,
) -> pd.DataFrame:
    interval_min = INTERVAL_TO_MIN[interval_label]
    ohlc = read_ohlc_asset_window(parquet_root, interval_min, asset, start_ts, end_ts)
    if ohlc.empty:
        return pd.DataFrame()

    ohlc = ohlc.sort_values("ts").drop_duplicates(subset=["ts"], keep="last").reset_index(drop=True)
    tr_calc = _compute_true_range(ohlc)
    scalar_df = pd.DataFrame(columns=["ts"])

    if prefer_scalar_features_true_range:
        scalar_df = read_scalar_feature_window(parquet_root, interval_min, asset, start_ts, end_ts)
        if not scalar_df.empty and "true_range" in scalar_df.columns:
            merged = ohlc[["ts"]].merge(scalar_df[["ts", "true_range"]], on="ts", how="left")
            tr = pd.to_numeric(merged["true_range"], errors="coerce")
            tr = tr.fillna(tr_calc)
        else:
            tr = tr_calc
    else:
        tr = tr_calc

    out = ohlc[["asset", "ts", "high", "low", "close", "volume", "trades"]].copy()
    out["true_range"] = pd.to_numeric(tr, errors="coerce")
    close = pd.to_numeric(out["close"], errors="coerce")
    out["log_return"] = np.log(close / close.shift(1))
    out["dollar_volume"] = pd.to_numeric(out["volume"], errors="coerce") * close
    out["true_range_pct"] = out["true_range"] / close.replace(0.0, np.nan)
    if not scalar_df.empty:
        extra_cols = [
            c
            for c in SCALAR_SEASONALITY_COLUMNS
            if c in scalar_df.columns and c not in {"true_range", "true_range_pct"}
        ]
        merge_cols = ["ts", *extra_cols]
        if "true_range_pct" in scalar_df.columns:
            merge_cols.append("true_range_pct")
        if len(merge_cols) > 1:
            out = out.merge(scalar_df[list(dict.fromkeys(merge_cols))], on="ts", how="left", suffixes=("", "_scalar"))
            if "true_range_pct_scalar" in out.columns:
                scalar_tr_pct = pd.to_numeric(out["true_range_pct_scalar"], errors="coerce")
                out["true_range_pct"] = scalar_tr_pct.where(scalar_tr_pct.notna(), out["true_range_pct"])
                out = out.drop(columns=["true_range_pct_scalar"])
    out = _build_calendar_columns(out, interval_label)
    return out


def _calendar_bucket_for_interval(out: pd.DataFrame, interval_label: str) -> pd.Series:
    if interval_label == "1H":
        return out["utc_day_of_week"].astype(int) * 24 + out["utc_hour"].astype(int)
    if interval_label == "4H":
        return out["utc_day_of_week"].astype(int) * 6 + (out["utc_hour"].astype(int) // 4)
    return out["utc_day_of_week"].astype(int)


def _session_tag(hour: int) -> str:
    h = int(hour)
    if 13 <= h < 17:
        return "Europe_US_overlap"
    if 7 <= h < 9:
        return "Asia_Europe_overlap"
    if 0 <= h < 7:
        return "Asia"
    if 9 <= h < 13:
        return "Europe"
    if 17 <= h < 22:
        return "US"
    return "Off_peak"


def add_utc_calendar_session_features(df: pd.DataFrame, interval_label: str) -> pd.DataFrame:
    out = _build_calendar_columns(df, interval_label).copy()
    dt = pd.to_datetime(out["ts"], unit="s", utc=True)
    out["utc_hour"] = dt.dt.hour.astype(int)
    out["utc_day_of_week"] = dt.dt.dayofweek.astype(int)
    out["utc_weekend_flag"] = (out["utc_day_of_week"] >= 5).astype(int)
    out["is_weekend"] = out["utc_weekend_flag"].astype(bool)
    out["utc_month"] = dt.dt.month.astype(int)
    out["hour_sin"] = np.sin(2.0 * np.pi * out["utc_hour"].astype(float) / 24.0)
    out["hour_cos"] = np.cos(2.0 * np.pi * out["utc_hour"].astype(float) / 24.0)
    out["dow_sin"] = np.sin(2.0 * np.pi * out["utc_day_of_week"].astype(float) / 7.0)
    out["dow_cos"] = np.cos(2.0 * np.pi * out["utc_day_of_week"].astype(float) / 7.0)
    week_pos = out["utc_day_of_week"].astype(float) * 24.0 + out["utc_hour"].astype(float)
    out["week_cycle_sin"] = np.sin(2.0 * np.pi * week_pos / 168.0)
    out["week_cycle_cos"] = np.cos(2.0 * np.pi * week_pos / 168.0)
    out["session_tag"] = out["utc_hour"].map(_session_tag)
    out["is_asia_session"] = out["utc_hour"].between(0, 8, inclusive="left").astype(int)
    out["is_europe_session"] = out["utc_hour"].between(7, 16, inclusive="left").astype(int)
    out["is_us_session"] = out["utc_hour"].between(13, 22, inclusive="left").astype(int)
    out["is_london_ny_overlap"] = out["utc_hour"].between(13, 16, inclusive="both").astype(int)
    out["session_overlap_flag"] = out["session_tag"].isin({"Europe_US_overlap", "Asia_Europe_overlap"}).astype(int)
    out["seasonality_bucket"] = _calendar_bucket_for_interval(out, interval_label).astype(int)
    return out


def _rolling_mad(values: np.ndarray) -> float:
    vals = pd.Series(values).replace([np.inf, -np.inf], np.nan).dropna()
    if vals.empty:
        return np.nan
    med = float(vals.median())
    return float(np.median(np.abs(vals.to_numpy(dtype=float) - med)))


def _add_asof_bucket_baseline(
    df: pd.DataFrame,
    *,
    value_col: str,
    prefix: str,
    bucket_col: str,
    window_days: int,
    min_samples: int,
) -> pd.DataFrame:
    out = df.copy()
    median_col = f"{prefix}_bucket_median_asof"
    mean_col = f"{prefix}_bucket_mean_asof"
    mad_col = f"{prefix}_bucket_mad_asof"
    count_col = f"{prefix}_bucket_sample_count_asof"
    vs_col = f"{prefix}_vs_usual_bucket"
    z_col = f"{prefix}_bucket_z_robust"
    out[median_col] = np.nan
    out[mean_col] = np.nan
    out[mad_col] = np.nan
    out[count_col] = 0
    out[vs_col] = np.nan
    out[z_col] = np.nan
    if value_col not in out.columns:
        return out

    window_obs = max(int(min_samples), int(np.ceil(max(1, int(window_days)) / 7.0)))
    for _, idx in out.groupby(bucket_col, sort=False).groups.items():
        idx_list = list(idx)
        s = pd.to_numeric(out.loc[idx_list, value_col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        prior = s.shift(1)
        med = prior.rolling(window=window_obs, min_periods=int(min_samples)).median()
        mean = prior.rolling(window=window_obs, min_periods=int(min_samples)).mean()
        mad = prior.rolling(window=window_obs, min_periods=int(min_samples)).apply(_rolling_mad, raw=True)
        cnt = prior.rolling(window=window_obs, min_periods=1).count()
        out.loc[idx_list, median_col] = med.to_numpy(dtype=float)
        out.loc[idx_list, mean_col] = mean.to_numpy(dtype=float)
        out.loc[idx_list, mad_col] = mad.to_numpy(dtype=float)
        out.loc[idx_list, count_col] = cnt.fillna(0).astype(int).to_numpy()

    current = pd.to_numeric(out[value_col], errors="coerce")
    med = pd.to_numeric(out[median_col], errors="coerce")
    mad = pd.to_numeric(out[mad_col], errors="coerce")
    denom = med.where(med.abs() > RATIO_DENOMINATOR_EPS)
    out[vs_col] = current / denom
    robust_scale = 1.4826 * mad.where(mad > RATIO_DENOMINATOR_EPS)
    out[z_col] = (current - med) / robust_scale
    out[vs_col] = pd.to_numeric(out[vs_col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    out[z_col] = pd.to_numeric(out[z_col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    return out


def _add_asof_recent_baseline(
    df: pd.DataFrame,
    *,
    value_col: str,
    prefix: str,
    window_bars: int,
    min_samples: int,
) -> pd.DataFrame:
    out = df.copy()
    median_col = f"{prefix}_recent_median_asof"
    mad_col = f"{prefix}_recent_mad_asof"
    count_col = f"{prefix}_recent_sample_count_asof"
    fallback_ratio_col = f"{prefix}_vs_usual_recent_fallback"
    out[median_col] = np.nan
    out[mad_col] = np.nan
    out[count_col] = 0
    out[fallback_ratio_col] = np.nan
    if value_col not in out.columns:
        return out

    current = pd.to_numeric(out[value_col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    prior = current.shift(1)
    window = max(int(min_samples), int(window_bars))
    med = prior.rolling(window=window, min_periods=int(min_samples)).median()
    mad = prior.rolling(window=window, min_periods=int(min_samples)).apply(_rolling_mad, raw=True)
    cnt = prior.rolling(window=window, min_periods=1).count()
    denom = med.where(med.abs() > RATIO_DENOMINATOR_EPS)
    out[median_col] = med.to_numpy(dtype=float)
    out[mad_col] = mad.to_numpy(dtype=float)
    out[count_col] = cnt.fillna(0).astype(int).to_numpy()
    out[fallback_ratio_col] = (current / denom).replace([np.inf, -np.inf], np.nan)
    return out


def _add_asof_group_baseline(
    df: pd.DataFrame,
    *,
    value_col: str,
    prefix: str,
    group_col: str,
    suffix: str,
    window_obs: int,
    min_samples: int,
) -> pd.DataFrame:
    out = df.copy()
    median_col = f"{prefix}_{suffix}_median_asof"
    mad_col = f"{prefix}_{suffix}_mad_asof"
    count_col = f"{prefix}_{suffix}_sample_count_asof"
    ratio_col = f"{prefix}_vs_usual_{suffix}"
    out[median_col] = np.nan
    out[mad_col] = np.nan
    out[count_col] = 0
    out[ratio_col] = np.nan
    if value_col not in out.columns or group_col not in out.columns:
        return out

    window = max(int(min_samples), int(window_obs))
    for _, idx in out.groupby(group_col, sort=False).groups.items():
        idx_list = list(idx)
        s = pd.to_numeric(out.loc[idx_list, value_col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        prior = s.shift(1)
        med = prior.rolling(window=window, min_periods=int(min_samples)).median()
        mad = prior.rolling(window=window, min_periods=int(min_samples)).apply(_rolling_mad, raw=True)
        cnt = prior.rolling(window=window, min_periods=1).count()
        out.loc[idx_list, median_col] = med.to_numpy(dtype=float)
        out.loc[idx_list, mad_col] = mad.to_numpy(dtype=float)
        out.loc[idx_list, count_col] = cnt.fillna(0).astype(int).to_numpy()

    current = pd.to_numeric(out[value_col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    med = pd.to_numeric(out[median_col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    denom = med.where(med.abs() > RATIO_DENOMINATOR_EPS)
    out[ratio_col] = (current / denom).replace([np.inf, -np.inf], np.nan)
    return out


def _select_hierarchical_ratio(
    out: pd.DataFrame,
    *,
    prefix: str,
    levels: Sequence[Tuple[str, str, str, str]],
    min_samples: int,
    source_present: bool,
) -> pd.DataFrame:
    selected_ratio = pd.Series(np.nan, index=out.index, dtype=float)
    selected_denom = pd.Series(np.nan, index=out.index, dtype=float)
    selected_count = pd.Series(0, index=out.index, dtype=float)
    selected_level = pd.Series("null_no_baseline", index=out.index, dtype=object)
    selected_quality = pd.Series("ratio_source_missing" if not source_present else "ratio_source_missing", index=out.index, dtype=object)

    for level_name, ratio_col, denom_col, count_col in levels:
        ratio = pd.to_numeric(out.get(ratio_col, np.nan), errors="coerce").replace([np.inf, -np.inf], np.nan)
        denom = pd.to_numeric(out.get(denom_col, np.nan), errors="coerce").replace([np.inf, -np.inf], np.nan)
        count = pd.to_numeric(out.get(count_col, 0), errors="coerce").fillna(0)
        eligible = (
            source_present
            & selected_ratio.isna()
            & ratio.notna()
            & denom.abs().gt(RATIO_DENOMINATOR_EPS).fillna(False)
            & count.ge(int(min_samples))
        )
        selected_ratio = selected_ratio.where(~eligible, ratio)
        selected_denom = selected_denom.where(~eligible, denom)
        selected_count = selected_count.where(~eligible, count)
        selected_level = selected_level.where(~eligible, level_name)

    selected_quality = _ratio_quality_labels(
        selected_ratio,
        selected_denom,
        selected_count,
        min_samples,
        source_present=source_present,
    )
    out[f"{prefix}_vs_usual_selected"] = selected_ratio
    out[f"{prefix}_selected_baseline_asof"] = selected_denom
    out[f"{prefix}_selected_sample_count_asof"] = selected_count.fillna(0).astype(int)
    out[f"{prefix}_fallback_level"] = selected_level
    out[f"{prefix}_fallback_used"] = selected_level.ne(levels[0][0])
    out[f"{prefix}_selected_quality_flag"] = selected_quality
    return out


def _ratio_quality_labels(
    ratio: pd.Series,
    denom: pd.Series,
    count: pd.Series,
    min_samples: int,
    *,
    source_present: bool = True,
) -> pd.Series:
    ratio_num = pd.to_numeric(ratio, errors="coerce")
    denom_num = pd.to_numeric(denom, errors="coerce").abs()
    count_num = pd.to_numeric(count, errors="coerce").fillna(0).astype(float)
    labels = np.full(len(ratio_num), "ratio_valid", dtype=object)
    if not source_present:
        labels[:] = "ratio_source_missing"
    else:
        labels[ratio_num.isna().to_numpy()] = "ratio_source_missing"
    denom_small = (denom_num <= RATIO_DENOMINATOR_EPS).fillna(False).to_numpy()
    sparse = (count_num < int(min_samples)).to_numpy()
    labels[(labels != "ratio_source_missing") & denom_small] = "ratio_denominator_too_small"
    labels[(labels == "ratio_valid") & sparse] = "ratio_unreliable_sparse_bucket"
    extreme = ((ratio_num < RATIO_CLIP_MIN) | (ratio_num > RATIO_CLIP_MAX)).fillna(False).to_numpy()
    labels[(labels == "ratio_valid") & extreme] = "ratio_extreme_clipped"
    return pd.Series(labels, index=ratio.index)


def _add_ratio_safety_columns(
    out: pd.DataFrame,
    *,
    prefix: str,
    ratio_col: str,
    denominator_col: str,
    count_col: str,
    min_samples: int,
    source_present: bool = True,
) -> pd.DataFrame:
    if ratio_col not in out.columns:
        out[ratio_col] = np.nan
    ratio = pd.to_numeric(out[ratio_col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    denom = pd.to_numeric(out.get(denominator_col, np.nan), errors="coerce").replace([np.inf, -np.inf], np.nan)
    count = pd.to_numeric(out.get(count_col, 0), errors="coerce").fillna(0)
    quality = _ratio_quality_labels(ratio, denom, count, min_samples, source_present=source_present)
    clipped = ratio.clip(lower=RATIO_CLIP_MIN, upper=RATIO_CLIP_MAX)
    clipped = clipped.where(quality.ne("ratio_denominator_too_small") & quality.ne("ratio_source_missing"))
    out[f"{prefix}_vs_usual_bucket_clipped"] = clipped
    out[f"{prefix}_vs_usual_bucket_log_ratio"] = np.log(ratio.where(ratio > 0)).replace([np.inf, -np.inf], np.nan)
    out[f"{prefix}_vs_usual_bucket_clip_flag"] = (
        (ratio < RATIO_CLIP_MIN) | (ratio > RATIO_CLIP_MAX)
    ).fillna(False)
    out[f"{prefix}_vs_usual_bucket_extreme_flag"] = (
        (ratio < RATIO_EXTREME_MIN) | (ratio > RATIO_EXTREME_MAX)
    ).fillna(False)
    out[f"{prefix}_vs_usual_bucket_denominator_too_small_flag"] = quality.eq("ratio_denominator_too_small")
    out[f"{prefix}_vs_usual_bucket_sparse_flag"] = quality.eq("ratio_unreliable_sparse_bucket")
    out[f"{prefix}_vs_usual_bucket_quality_flag"] = quality
    return out


def _add_selected_ratio_safety_columns(out: pd.DataFrame, *, prefix: str) -> pd.DataFrame:
    ratio = pd.to_numeric(out.get(f"{prefix}_vs_usual_selected", np.nan), errors="coerce").replace([np.inf, -np.inf], np.nan)
    quality = pd.Series(out.get(f"{prefix}_selected_quality_flag", "ratio_source_missing"), index=out.index).astype(str)
    clipped = ratio.clip(lower=RATIO_CLIP_MIN, upper=RATIO_CLIP_MAX)
    clipped = clipped.where(quality.ne("ratio_denominator_too_small") & quality.ne("ratio_source_missing"))
    out[f"{prefix}_vs_usual_selected_clipped"] = clipped
    out[f"{prefix}_vs_usual_selected_log_ratio"] = np.log(ratio.where(ratio > 0)).replace([np.inf, -np.inf], np.nan)
    out[f"{prefix}_vs_usual_selected_clip_flag"] = (
        (ratio < RATIO_CLIP_MIN) | (ratio > RATIO_CLIP_MAX)
    ).fillna(False)
    out[f"{prefix}_vs_usual_selected_extreme_flag"] = (
        (ratio < RATIO_EXTREME_MIN) | (ratio > RATIO_EXTREME_MAX)
    ).fillna(False)
    out[f"{prefix}_vs_usual_selected_denominator_too_small_flag"] = quality.eq("ratio_denominator_too_small")
    out[f"{prefix}_vs_usual_selected_sparse_flag"] = quality.eq("ratio_unreliable_sparse_bucket")
    return out


def _label_quality(count: Any, min_samples: int, source_present: bool) -> str:
    if not source_present:
        return "source_missing"
    try:
        c = int(count)
    except Exception:
        c = 0
    return "ok" if c >= int(min_samples) else "insufficient_bucket_history"


def _ratio_label(value: Any, high: float = 1.5, low: float = 0.67) -> str:
    try:
        v = float(value)
    except Exception:
        return "unknown"
    if not np.isfinite(v):
        return "unknown"
    if v >= high:
        return "above_usual"
    if v <= low:
        return "below_usual"
    return "near_usual"


def build_model_feature_dataframe(
    series_df: pd.DataFrame,
    interval_label: str,
    *,
    bucket_window_days: int = DEFAULT_FEATURE_LOOKBACK_DAYS,
    min_bucket_samples: int = DEFAULT_FEATURE_MIN_BUCKET_SAMPLES,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    out = add_utc_calendar_session_features(series_df, interval_label)
    out = out.sort_values("ts").drop_duplicates(subset=["ts"], keep="last").reset_index(drop=True)
    out["seasonality_hour_bucket"] = out["utc_hour"].astype(int)
    out["seasonality_4h_slot_bucket"] = (out["utc_hour"].astype(int) // 4).astype(int)
    out["seasonality_session_weekend_bucket"] = out["session_tag"].astype(str) + "_weekend_" + out["utc_weekend_flag"].astype(str)
    out["seasonality_weekend_bucket"] = out["utc_weekend_flag"].astype(int)
    for c in out.columns:
        if c not in {"asset", "session_tag"} and is_numeric_dtype(out[c]):
            out[c] = pd.to_numeric(out[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
    if "atr_pct_14" not in out.columns:
        out["atr_pct_14"] = np.nan
    if "activity_state_score_20" not in out.columns:
        out["activity_state_score_20"] = np.nan
    if "illiquidity_proxy_20" not in out.columns:
        out["illiquidity_proxy_20"] = np.nan
    if "realized_volatility" not in out.columns:
        out["realized_volatility"] = pd.to_numeric(out.get("ret_std_20", np.nan), errors="coerce")

    baseline_specs = [
        ("volume", "volume"),
        ("trades", "trades"),
        ("dollar_volume", "dollar_volume"),
        ("true_range_pct", "true_range_pct"),
        ("atr_pct_14", "atr_pct"),
        ("realized_volatility", "realized_vol"),
        ("illiquidity_proxy_20", "illiquidity_proxy"),
        ("activity_state_score_20", "activity_state_score"),
        ("log_return", "return"),
    ]
    bars_per_day = max(1, int(round(1440 / INTERVAL_TO_MIN.get(interval_label, 1440))))
    recent_window_bars = max(int(min_bucket_samples), int(bucket_window_days) * bars_per_day)
    if interval_label == "1H":
        fallback_group_specs = [
            ("hour", "seasonality_hour_bucket", max(int(min_bucket_samples), int(bucket_window_days))),
            ("session_weekend", "seasonality_session_weekend_bucket", max(int(min_bucket_samples), int(bucket_window_days) * 4)),
        ]
        hierarchy_names = ["asset_hour_weekday", "asset_hour", "asset_session_weekend", "asset_recent_overall"]
    elif interval_label == "4H":
        fallback_group_specs = [
            ("4h_slot", "seasonality_4h_slot_bucket", max(int(min_bucket_samples), int(bucket_window_days))),
            ("session_weekend", "seasonality_session_weekend_bucket", max(int(min_bucket_samples), int(bucket_window_days))),
        ]
        hierarchy_names = ["asset_4h_slot_weekday", "asset_4h_slot", "asset_session_weekend", "asset_recent_overall"]
    else:
        fallback_group_specs = [
            ("weekend", "seasonality_weekend_bucket", max(int(min_bucket_samples), int(bucket_window_days))),
        ]
        hierarchy_names = ["asset_day_of_week", "asset_weekend", "asset_recent_overall"]
    for value_col, prefix in baseline_specs:
        out = _add_asof_bucket_baseline(
            out,
            value_col=value_col,
            prefix=prefix,
            bucket_col="seasonality_bucket",
            window_days=bucket_window_days,
            min_samples=min_bucket_samples,
        )
        for suffix, group_col, window_obs in fallback_group_specs:
            out = _add_asof_group_baseline(
                out,
                value_col=value_col,
                prefix=prefix,
                group_col=group_col,
                suffix=suffix,
                window_obs=window_obs,
                min_samples=min_bucket_samples,
            )
        out = _add_asof_recent_baseline(
            out,
            value_col=value_col,
            prefix=prefix,
            window_bars=recent_window_bars,
            min_samples=min_bucket_samples,
        )
    if "return_bucket_median_asof" in out.columns:
        out["return_vs_usual_bucket"] = (
            pd.to_numeric(out["log_return"], errors="coerce")
            - pd.to_numeric(out["return_bucket_median_asof"], errors="coerce")
        ).replace([np.inf, -np.inf], np.nan)

    out["volatility_vs_usual_bucket"] = out["atr_pct_vs_usual_bucket"].where(
        pd.to_numeric(out["atr_pct_vs_usual_bucket"], errors="coerce").notna(),
        out["true_range_pct_vs_usual_bucket"],
    )
    out["volatility_bucket_median_asof"] = out["atr_pct_bucket_median_asof"].where(
        pd.to_numeric(out["atr_pct_bucket_median_asof"], errors="coerce").notna(),
        out["true_range_pct_bucket_median_asof"],
    )
    out["volatility_bucket_sample_count_asof"] = out[["atr_pct_bucket_sample_count_asof", "true_range_pct_bucket_sample_count_asof"]].max(axis=1)
    out["volatility_bucket_z_robust"] = out["atr_pct_bucket_z_robust"].where(
        pd.to_numeric(out["atr_pct_bucket_z_robust"], errors="coerce").notna(),
        out["true_range_pct_bucket_z_robust"],
    )
    out["activity_state_vs_usual_bucket"] = out["activity_state_score_vs_usual_bucket"]
    out["illiquidity_vs_usual_bucket"] = out["illiquidity_proxy_vs_usual_bucket"]
    out["volatility_vs_usual_recent_fallback"] = out["atr_pct_vs_usual_recent_fallback"].where(
        pd.to_numeric(out["atr_pct_vs_usual_recent_fallback"], errors="coerce").notna(),
        out["true_range_pct_vs_usual_recent_fallback"],
    )
    out["volatility_recent_median_asof"] = out["atr_pct_recent_median_asof"].where(
        pd.to_numeric(out["atr_pct_recent_median_asof"], errors="coerce").notna(),
        out["true_range_pct_recent_median_asof"],
    )
    out["volatility_recent_sample_count_asof"] = out[["atr_pct_recent_sample_count_asof", "true_range_pct_recent_sample_count_asof"]].max(axis=1)
    out["activity_state_vs_usual_recent_fallback"] = out["activity_state_score_vs_usual_recent_fallback"]
    out["activity_state_recent_median_asof"] = out["activity_state_score_recent_median_asof"]
    out["activity_state_recent_sample_count_asof"] = out["activity_state_score_recent_sample_count_asof"]
    out["illiquidity_vs_usual_recent_fallback"] = out["illiquidity_proxy_vs_usual_recent_fallback"]
    out["illiquidity_recent_median_asof"] = out["illiquidity_proxy_recent_median_asof"]
    out["illiquidity_recent_sample_count_asof"] = out["illiquidity_proxy_recent_sample_count_asof"]
    for suffix, _group_col, _window_obs in fallback_group_specs:
        out[f"volatility_vs_usual_{suffix}"] = out[f"atr_pct_vs_usual_{suffix}"].where(
            pd.to_numeric(out[f"atr_pct_vs_usual_{suffix}"], errors="coerce").notna(),
            out[f"true_range_pct_vs_usual_{suffix}"],
        )
        out[f"volatility_{suffix}_median_asof"] = out[f"atr_pct_{suffix}_median_asof"].where(
            pd.to_numeric(out[f"atr_pct_{suffix}_median_asof"], errors="coerce").notna(),
            out[f"true_range_pct_{suffix}_median_asof"],
        )
        out[f"volatility_{suffix}_sample_count_asof"] = out[
            [f"atr_pct_{suffix}_sample_count_asof", f"true_range_pct_{suffix}_sample_count_asof"]
        ].max(axis=1)
        out[f"activity_state_vs_usual_{suffix}"] = out[f"activity_state_score_vs_usual_{suffix}"]
        out[f"activity_state_{suffix}_median_asof"] = out[f"activity_state_score_{suffix}_median_asof"]
        out[f"activity_state_{suffix}_sample_count_asof"] = out[f"activity_state_score_{suffix}_sample_count_asof"]
        out[f"illiquidity_vs_usual_{suffix}"] = out[f"illiquidity_proxy_vs_usual_{suffix}"]
        out[f"illiquidity_{suffix}_median_asof"] = out[f"illiquidity_proxy_{suffix}_median_asof"]
        out[f"illiquidity_{suffix}_sample_count_asof"] = out[f"illiquidity_proxy_{suffix}_sample_count_asof"]
    source_presence = {
        "volume": "volume" in out.columns and pd.to_numeric(out["volume"], errors="coerce").notna().any(),
        "trades": pd.to_numeric(out.get("trades", np.nan), errors="coerce").notna().any(),
        "dollar_volume": pd.to_numeric(out.get("dollar_volume", np.nan), errors="coerce").notna().any(),
        "volatility": pd.to_numeric(out["true_range_pct"], errors="coerce").notna().any()
        or pd.to_numeric(out["atr_pct_14"], errors="coerce").notna().any(),
        "illiquidity": pd.to_numeric(out.get("illiquidity_proxy_20", np.nan), errors="coerce").notna().any(),
        "activity_state": pd.to_numeric(out.get("activity_state_score_20", np.nan), errors="coerce").notna().any(),
        "return": pd.to_numeric(out.get("log_return", np.nan), errors="coerce").notna().any(),
    }
    prefix_level_sources = {
        "volume": ("volume", "volume"),
        "trades": ("trades", "trades"),
        "dollar_volume": ("dollar_volume", "dollar_volume"),
        "volatility": ("volatility", "volatility"),
        "illiquidity": ("illiquidity_proxy", "illiquidity"),
        "activity_state": ("activity_state_score", "activity_state"),
        "return": ("return", "return"),
    }
    for selected_prefix, (base_prefix, presence_key) in prefix_level_sources.items():
        exact_level = (
            hierarchy_names[0],
            f"{selected_prefix}_vs_usual_bucket" if selected_prefix in {"volatility", "activity_state", "illiquidity"} else f"{base_prefix}_vs_usual_bucket",
            f"{base_prefix}_bucket_median_asof",
            f"{base_prefix}_bucket_sample_count_asof",
        )
        levels: list[Tuple[str, str, str, str]] = [exact_level]
        for i, (suffix, _group_col, _window_obs) in enumerate(fallback_group_specs, start=1):
            ratio_name_prefix = selected_prefix if selected_prefix in {"volatility", "activity_state", "illiquidity"} else base_prefix
            median_name_prefix = base_prefix
            levels.append(
                (
                    hierarchy_names[i],
                    f"{ratio_name_prefix}_vs_usual_{suffix}",
                    f"{median_name_prefix}_{suffix}_median_asof",
                    f"{median_name_prefix}_{suffix}_sample_count_asof",
                )
            )
        recent_ratio_prefix = selected_prefix if selected_prefix in {"volatility", "activity_state", "illiquidity"} else base_prefix
        levels.append(
            (
                hierarchy_names[-1],
                f"{recent_ratio_prefix}_vs_usual_recent_fallback",
                f"{base_prefix}_recent_median_asof",
                f"{base_prefix}_recent_sample_count_asof",
            )
        )
        out = _select_hierarchical_ratio(
            out,
            prefix=selected_prefix,
            levels=levels,
            min_samples=min_bucket_samples,
            source_present=bool(source_presence[presence_key]),
        )
        out = _add_selected_ratio_safety_columns(out, prefix=selected_prefix)
    out["bucket_sample_count_asof"] = out[
        [
            "volume_bucket_sample_count_asof",
            "trades_bucket_sample_count_asof",
            "true_range_pct_bucket_sample_count_asof",
            "atr_pct_bucket_sample_count_asof",
            "illiquidity_proxy_bucket_sample_count_asof",
            "activity_state_score_bucket_sample_count_asof",
        ]
    ].max(axis=1)
    out["bucket_window_days"] = int(bucket_window_days)
    out["bucket_min_samples_required"] = int(min_bucket_samples)
    core_selected_count = out[
        [
            "volume_selected_sample_count_asof",
            "trades_selected_sample_count_asof",
            "volatility_selected_sample_count_asof",
        ]
    ].max(axis=1)
    out["bucket_min_samples_met"] = core_selected_count.astype(int) >= int(min_bucket_samples)
    out["bucket_sparse_flag"] = ~out["bucket_min_samples_met"].astype(bool)
    core_levels = out[["volume_fallback_level", "trades_fallback_level", "volatility_fallback_level"]].astype(str)
    out["bucket_fallback_used"] = core_levels.ne(hierarchy_names[0]).any(axis=1)
    out["bucket_fallback_level"] = core_levels.apply(
        lambda row: next((v for v in row if v != hierarchy_names[0]), hierarchy_names[0]),
        axis=1,
    )

    volume_present = source_presence["volume"]
    vol_present = pd.to_numeric(out["true_range_pct"], errors="coerce").notna().any()
    out["seasonality_feature_quality_flag"] = [
        _label_quality(c, min_bucket_samples, volume_present or vol_present) for c in out["bucket_sample_count_asof"]
    ]
    out["bucket_baseline_quality"] = out["seasonality_feature_quality_flag"]
    out["leakage_policy"] = "asof_strict_prior_same_bucket"
    out["feature_set_version"] = MODEL_FEATURE_SET_VERSION
    out["return_seasonality_diagnostic_only"] = True

    ratio_specs = [
        ("volume", "volume_vs_usual_bucket", "volume_bucket_median_asof", "volume_bucket_sample_count_asof", volume_present),
        ("trades", "trades_vs_usual_bucket", "trades_bucket_median_asof", "trades_bucket_sample_count_asof", pd.to_numeric(out.get("trades", np.nan), errors="coerce").notna().any()),
        (
            "dollar_volume",
            "dollar_volume_vs_usual_bucket",
            "dollar_volume_bucket_median_asof",
            "dollar_volume_bucket_sample_count_asof",
            pd.to_numeric(out.get("dollar_volume", np.nan), errors="coerce").notna().any(),
        ),
        (
            "volatility",
            "volatility_vs_usual_bucket",
            "atr_pct_bucket_median_asof",
            "atr_pct_bucket_sample_count_asof",
            vol_present or pd.to_numeric(out["atr_pct_14"], errors="coerce").notna().any(),
        ),
        (
            "illiquidity",
            "illiquidity_vs_usual_bucket",
            "illiquidity_proxy_bucket_median_asof",
            "illiquidity_proxy_bucket_sample_count_asof",
            pd.to_numeric(out.get("illiquidity_proxy_20", np.nan), errors="coerce").notna().any(),
        ),
        (
            "activity_state",
            "activity_state_vs_usual_bucket",
            "activity_state_score_bucket_median_asof",
            "activity_state_score_bucket_sample_count_asof",
            pd.to_numeric(out.get("activity_state_score_20", np.nan), errors="coerce").notna().any(),
        ),
        ("return", "return_vs_usual_bucket", "return_bucket_median_asof", "return_bucket_sample_count_asof", pd.to_numeric(out.get("log_return", np.nan), errors="coerce").notna().any()),
    ]
    for prefix, ratio_col, denom_col, count_col, present in ratio_specs:
        out = _add_ratio_safety_columns(
            out,
            prefix=prefix,
            ratio_col=ratio_col,
            denominator_col=denom_col,
            count_col=count_col,
            min_samples=min_bucket_samples,
            source_present=bool(present),
        )

    volume_vs = pd.to_numeric(out["volume_vs_usual_selected_clipped"], errors="coerce")
    trades_vs = pd.to_numeric(out["trades_vs_usual_selected_clipped"], errors="coerce")
    vol_vs = pd.to_numeric(out["volatility_vs_usual_selected_clipped"], errors="coerce")
    illiq_vs = pd.to_numeric(out["illiquidity_vs_usual_selected_clipped"], errors="coerce")
    activity_vs = pd.to_numeric(out["activity_state_vs_usual_selected_clipped"], errors="coerce")
    out["thin_liquidity_bucket_flag"] = ((volume_vs < 0.5) | (trades_vs < 0.5) | (illiq_vs > 1.5)).fillna(False)
    out["is_low_activity_window"] = ((volume_vs < 0.67) | (trades_vs < 0.67) | (activity_vs < 0.75)).fillna(False)
    out["is_active_window"] = ((volume_vs > 1.25) | (trades_vs > 1.25) | (activity_vs > 1.15)).fillna(False)
    out["active_but_stressed_window_flag"] = (out["is_active_window"] & ((vol_vs > 1.5) | (illiq_vs > 1.5))).fillna(False)
    out["liquidity_condition_bucket_label"] = np.select(
        [out["thin_liquidity_bucket_flag"], (illiq_vs > 1.25).fillna(False), (volume_vs > 1.25).fillna(False)],
        ["thin", "stressed", "active"],
        default="normal",
    )

    out["bucket_volume_vs_usual"] = out["volume_vs_usual_selected"]
    out["bucket_volume_vs_usual_clipped"] = out["volume_vs_usual_selected_clipped"]
    out["bucket_volume_quality_flag"] = out["volume_selected_quality_flag"]
    out["bucket_trades_vs_usual"] = out["trades_vs_usual_selected"]
    out["bucket_trades_vs_usual_clipped"] = out["trades_vs_usual_selected_clipped"]
    out["bucket_trades_quality_flag"] = out["trades_selected_quality_flag"]
    out["bucket_dollar_volume_vs_usual"] = out["dollar_volume_vs_usual_selected"]
    out["bucket_volatility_vs_usual"] = out["volatility_vs_usual_selected"]
    out["bucket_volatility_vs_usual_clipped"] = out["volatility_vs_usual_selected_clipped"]
    out["bucket_volatility_quality_flag"] = out["volatility_selected_quality_flag"]
    out["bucket_activity_vs_usual"] = out["activity_state_vs_usual_selected"]
    out["bucket_liquidity_or_illiquidity_vs_usual"] = out["illiquidity_vs_usual_selected"]
    out["thin_window_flag"] = out["thin_liquidity_bucket_flag"]
    out["active_window_flag"] = out["is_active_window"]
    out["seasonal_volatility_regime_note"] = np.select(
        [(vol_vs >= 1.5).fillna(False), (vol_vs <= 0.67).fillna(False)],
        ["volatility_above_usual_context_only", "volatility_below_usual_context_only"],
        default="volatility_near_usual_or_unknown_context_only",
    )
    clip_cols = [
        "volume_vs_usual_selected_clip_flag",
        "trades_vs_usual_selected_clip_flag",
        "dollar_volume_vs_usual_selected_clip_flag",
        "volatility_vs_usual_selected_clip_flag",
        "illiquidity_vs_usual_selected_clip_flag",
        "activity_state_vs_usual_selected_clip_flag",
    ]
    clipped_any = out[[c for c in clip_cols if c in out.columns]].any(axis=1) if clip_cols else pd.Series(False, index=out.index)
    low_core = out[
        [
            "volume_selected_quality_flag",
            "trades_selected_quality_flag",
            "volatility_selected_quality_flag",
        ]
    ].isin(["ratio_denominator_too_small", "ratio_source_missing", "ratio_unreliable_sparse_bucket"]).sum(axis=1)
    score = 1.0 - (0.25 * out["bucket_sparse_flag"].astype(float)) - (0.20 * clipped_any.astype(float)) - (0.15 * low_core.astype(float))
    out["seasonality_context_quality_score"] = score.clip(lower=0.0, upper=1.0).round(3)
    out["seasonality_context_quality"] = np.select(
        [out["seasonality_context_quality_score"] >= 0.75, out["seasonality_context_quality_score"] >= 0.45],
        ["high", "medium"],
        default="low",
    )
    warnings: list[str] = []
    for sparse, clipped, thin, stressed, low_count in zip(
        out["bucket_sparse_flag"].astype(bool),
        clipped_any.astype(bool),
        out["thin_liquidity_bucket_flag"].astype(bool),
        out["active_but_stressed_window_flag"].astype(bool),
        low_core.astype(int),
    ):
        parts: list[str] = []
        if sparse:
            parts.append("sparse_bucket_fallback_used")
        if clipped:
            parts.append("extreme_ratio_clipped")
        if thin:
            parts.append("thin_or_illiquid_window")
        if stressed:
            parts.append("active_but_stressed_window")
        if low_count:
            parts.append("low_quality_core_ratio")
        warnings.append(";".join(parts) if parts else "none")
    out["seasonality_context_warnings"] = warnings
    out["context_time_utc"] = pd.to_datetime(out["ts"], unit="s", utc=True).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    out["matched_artifact_time_utc"] = out["context_time_utc"]
    out["context_freshness_status"] = "artifact_row"
    out["interval"] = interval_label
    out["seasonal_target_confidence_adjustment"] = np.where(
        out["thin_window_flag"],
        "caution_thin_context_only",
        np.where(out["active_but_stressed_window_flag"], "caution_active_stressed_context_only", "none_context_only"),
    )
    out["seasonal_time_rule_note"] = np.where(
        out["session_overlap_flag"].astype(bool),
        "utc_session_overlap_context_only",
        "utc_24_7_crypto_context_only",
    )
    if interval_label == "1D":
        daily_warning = "daily_interval_hour_session_fields_not_applicable"
        out["utc_hour"] = np.nan
        out["session_tag"] = None
        out["session_overlap_flag"] = np.nan
        out["seasonal_time_rule_note"] = "utc_day_of_week_context_only"
    else:
        daily_warning = ""
    out["seasonality_summary_note"] = [
        f"volume={_ratio_label(v)}; volatility={_ratio_label(vol)}; liquidity={_ratio_label(ill, high=1.25, low=0.8)}; context_only"
        for v, vol, ill in zip(out["bucket_volume_vs_usual"], out["bucket_volatility_vs_usual"], out["bucket_liquidity_or_illiquidity_vs_usual"])
    ]
    if daily_warning:
        out["seasonality_context_warnings"] = np.where(
            out["seasonality_context_warnings"].astype(str).eq("none"),
            daily_warning,
            out["seasonality_context_warnings"].astype(str) + ";" + daily_warning,
        )

    missingness = {
        c: float(pd.to_numeric(out[c], errors="coerce").isna().mean())
        for c in MODEL_FEATURE_COLUMNS
        if c in out.columns and out[c].dtype.kind in {"f", "i", "u", "b"}
    }
    quality_flag_distribution = (
        out["seasonality_context_quality"].value_counts(dropna=False).astype(int).to_dict()
        if "seasonality_context_quality" in out.columns
        else {}
    )
    ratio_quality_distribution = {
        c: out[c].value_counts(dropna=False).astype(int).to_dict()
        for c in out.columns
        if c.endswith("_vs_usual_bucket_quality_flag")
    }
    clipped_ratio_counts = {
        c: int(pd.Series(out[c]).fillna(False).astype(bool).sum())
        for c in out.columns
        if c.endswith("_vs_usual_bucket_clip_flag")
    }
    extreme_ratio_counts = {
        c: int(pd.Series(out[c]).fillna(False).astype(bool).sum())
        for c in out.columns
        if c.endswith("_vs_usual_bucket_extreme_flag")
    }
    diagnostics = {
        "feature_set_version": MODEL_FEATURE_SET_VERSION,
        "module_version": MODULE_VERSION,
        "input_rows": int(len(series_df)),
        "output_rows": int(len(out)),
        "source_start_ts": int(out["ts"].min()) if not out.empty else None,
        "source_end_ts": int(out["ts"].max()) if not out.empty else None,
        "bucket_window_days": int(bucket_window_days),
        "bucket_min_samples": int(min_bucket_samples),
        "bucket_count": int(out["seasonality_bucket"].nunique()) if "seasonality_bucket" in out else 0,
        "buckets_failing_min_sample_rows": int((~out["bucket_min_samples_met"].astype(bool)).sum()) if not out.empty else 0,
        "bucket_sparse_rows": int(out["bucket_sparse_flag"].astype(bool).sum()) if not out.empty else 0,
        "bucket_fallback_used_rows": int(out["bucket_fallback_used"].astype(bool).sum()) if not out.empty else 0,
        "bucket_fallback_level_distribution": out["bucket_fallback_level"].value_counts(dropna=False).astype(int).to_dict(),
        "missingness_summary": missingness,
        "context_quality_distribution": quality_flag_distribution,
        "ratio_quality_distribution": ratio_quality_distribution,
        "clipped_ratio_counts": clipped_ratio_counts,
        "extreme_ratio_counts": extreme_ratio_counts,
        "near_zero_denominator_policy": f"denominators <= {RATIO_DENOMINATOR_EPS:g} are nulled and flagged",
        "ratio_clip_policy": {"min": RATIO_CLIP_MIN, "max": RATIO_CLIP_MAX},
        "leakage_safety_check_status": "passed_current_row_excluded_from_bucket_baseline",
        "return_seasonality_policy": "diagnostic_only",
        "multiple_testing_warning": "return seasonality remains diagnostic only; do not treat bucket return effects as validated directional signals",
        "session_definitions_utc": SESSION_DEFINITIONS_UTC,
        "static_full_history_baseline_warning": "legacy profile artifacts are descriptive; model_feature_set_v1 uses strict prior as-of bucket baselines",
        "model_facing_feature_columns": [c for c in MODEL_FEATURE_COLUMNS if c in out.columns],
        "manual_context_columns": [c for c in MANUAL_CONTEXT_COLUMNS if c in out.columns],
    }
    return out, diagnostics


def build_asset_model_feature_artifact(
    parquet_root: Path,
    interval_label: str,
    asset: str,
    start_ts: int,
    end_ts: int,
    prefer_scalar_features_true_range: bool,
    bucket_window_days: int,
    min_bucket_samples: int,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    series_df = _prepare_asset_series(
        parquet_root=parquet_root,
        interval_label=interval_label,
        asset=asset,
        start_ts=start_ts,
        end_ts=end_ts,
        prefer_scalar_features_true_range=prefer_scalar_features_true_range,
    )
    if series_df.empty:
        return pd.DataFrame(), {
            "asset": str(asset),
            "interval": interval_label,
            "feature_set_version": MODEL_FEATURE_SET_VERSION,
            "status": "empty_input",
            "input_rows": 0,
            "output_rows": 0,
        }
    features, diagnostics = build_model_feature_dataframe(
        series_df,
        interval_label,
        bucket_window_days=bucket_window_days,
        min_bucket_samples=min_bucket_samples,
    )
    diagnostics.update({"asset": str(asset), "interval": interval_label, "status": "built"})
    return features, diagnostics


def manual_context_dataframe(features: pd.DataFrame) -> pd.DataFrame:
    base_cols = ["asset", "ts", "feature_set_version", *MANUAL_CONTEXT_COLUMNS]
    cols = [c for c in base_cols if c in features.columns]
    out = features[cols].copy()
    out["manual_context_version"] = MANUAL_CONTEXT_VERSION
    out["manual_context_policy"] = "context_only_not_trade_trigger"
    return out


def build_asset_artifact(
    parquet_root: Path,
    interval_label: str,
    asset: str,
    start_ts: int,
    end_ts: int,
    prefer_scalar_features_true_range: bool,
    smoothing_window: int,
    top_k: int,
    baseline_method: str,
) -> Optional[ArtifactResult]:
    series_df = _prepare_asset_series(
        parquet_root=parquet_root,
        interval_label=interval_label,
        asset=asset,
        start_ts=start_ts,
        end_ts=end_ts,
        prefer_scalar_features_true_range=prefer_scalar_features_true_range,
    )
    if series_df.empty:
        return None

    families = _component_families(interval_label)
    comps: Dict[Tuple[str, str], Dict[str, Any]] = {}

    for family, n_buckets in families.items():
        for channel in ("mean", "vol"):
            use_smoothing = family == "hour_of_week" and smoothing_window > 1
            comp = _compute_component(
                df=series_df,
                family=family,
                n_buckets=n_buckets,
                channel=channel,
                baseline_method=baseline_method,
                smoothing_window=smoothing_window,
                top_k_n=top_k,
                scope="asset",
                apply_smoothing=use_smoothing,
            )
            comps[(family, channel)] = comp

    vol_period_candidates, vol_recommended_period = detect_period_candidates(
        series_df["true_range"],
        interval_label,
        scope="asset",
        period_channel="vol",
    )
    return_period_candidates, return_recommended_period = detect_period_candidates(
        series_df["log_return"],
        interval_label,
        scope="asset",
        period_channel="return",
    )
    period_candidates = sorted(
        list(vol_period_candidates) + list(return_period_candidates),
        key=lambda d: (float(d.get("quality_score", 0.0)), float(d.get("strength_score", 0.0)), float(d.get("stability_score", 0.0))),
        reverse=True,
    )
    rec_period, recommendation_channel, rec_period_channel = _choose_recommendation_channel(
        vol_recommended_period, return_recommended_period
    )
    vol_stability_score = _recommended_stability_score(vol_period_candidates, vol_recommended_period)
    return_stability_score = _recommended_stability_score(return_period_candidates, return_recommended_period)

    return ArtifactResult(
        scope="asset",
        interval=interval_label,
        asset=asset,
        source="asset_specific",
        source_priority=100,
        bucket_components=comps,
        period_candidates=period_candidates,
        vol_period_candidates=vol_period_candidates,
        vol_recommended_period=vol_recommended_period,
        vol_stability_score=vol_stability_score,
        return_period_candidates=return_period_candidates,
        return_recommended_period=return_recommended_period,
        return_stability_score=return_stability_score,
        recommendation_channel=recommendation_channel,
        recommended_period_bars=rec_period,
        recommended_period_channel=rec_period_channel,
    )


def _aggregate_global_components(asset_artifacts: Sequence[ArtifactResult], interval_label: str, top_k: int) -> Dict[Tuple[str, str], Dict[str, Any]]:
    families = _component_families(interval_label)
    out: Dict[Tuple[str, str], Dict[str, Any]] = {}

    for family, n_buckets in families.items():
        for channel in ("mean", "vol"):
            comp_list = [a.bucket_components.get((family, channel)) for a in asset_artifacts]
            comp_list = [c for c in comp_list if c is not None]
            if not comp_list:
                continue

            matrix = np.vstack([c["values"] for c in comp_list])
            counts = np.vstack([c["bucket_counts"] for c in comp_list])
            weights = np.array([max(0.05, float(c["quality_score"])) for c in comp_list], dtype=float)
            agg_values = np.average(matrix, axis=0, weights=weights)
            agg_counts = np.sum(counts, axis=0).astype(int)

            signal_to_noise = float(np.average([float(c["signal_to_noise"]) for c in comp_list], weights=weights))
            stability = float(np.median([float(c["stability_score"]) for c in comp_list]))
            coverage_bars = int(np.sum([int(c["coverage_bars"]) for c in comp_list]))
            start_vals = [c.get("coverage_start_ts") for c in comp_list if c.get("coverage_start_ts") is not None]
            end_vals = [c.get("coverage_end_ts") for c in comp_list if c.get("coverage_end_ts") is not None]
            baseline_vals = [float(c.get("baseline", np.nan)) for c in comp_list]
            baseline_vals = [b for b in baseline_vals if np.isfinite(b)]
            baseline = float(np.median(baseline_vals)) if baseline_vals else (0.0 if channel == "mean" else 1.0)

            quality_score, rec_use, usable = _component_quality(
                signal_to_noise=signal_to_noise,
                stability=stability,
                coverage_bars=coverage_bars,
                n_buckets=n_buckets,
                scope="global",
            )

            topk = _top_k(agg_values, agg_counts, channel, top_k)
            out[(family, channel)] = {
                "family": family,
                "channel": channel,
                "n_buckets": n_buckets,
                "values": np.nan_to_num(agg_values, nan=0.0 if channel == "mean" else 1.0),
                "bucket_counts": agg_counts,
                "baseline": baseline,
                "coverage_bars": coverage_bars,
                "coverage_start_ts": int(min(start_vals)) if start_vals else None,
                "coverage_end_ts": int(max(end_vals)) if end_vals else None,
                "signal_to_noise": signal_to_noise,
                "stability_score": stability,
                "quality_score": quality_score,
                "recommended_use": rec_use,
                "usable": usable,
                "top_k": topk,
            }
    return out


def _aggregate_global_periods(
    asset_artifacts: Sequence[ArtifactResult],
    interval_label: str,
    period_attr: str,
    period_channel: str,
) -> Tuple[List[Dict[str, Any]], Optional[int], float]:
    by_period: Dict[int, List[Dict[str, Any]]] = {p: [] for p in CANDIDATE_PERIODS[interval_label]}
    for art in asset_artifacts:
        for c in getattr(art, period_attr, []) or []:
            p = int(c["period_bars"])
            if p in by_period:
                by_period[p].append(c)

    out: List[Dict[str, Any]] = []
    for p in CANDIDATE_PERIODS[interval_label]:
        vals = by_period.get(p, [])
        if not vals:
            continue
        strengths = [float(v["strength_score"]) for v in vals]
        stabs = [float(v["stability_score"]) for v in vals]
        cov = [int(v["coverage_bars"]) for v in vals]
        strength = float(np.mean(strengths))
        stability = float(np.median(stabs))
        coverage = int(np.sum(cov))
        q_score, rec_use, usable = _period_quality(strength, stability, coverage, p, scope="global")
        out.append(
            {
                "period_bars": int(p),
                "strength_score": strength,
                "stability_score": stability,
                "quality_score": q_score,
                "recommended_use": rec_use,
                "usable": usable,
                "coverage_bars": coverage,
                "period_channel": period_channel,
                "assets_contributing": int(len(vals)),
                "method": "consensus_acf_plus_spectral_with_rolling_stability",
            }
        )

    out = sorted(out, key=lambda d: (d["quality_score"], d["strength_score"], d["stability_score"]), reverse=True)
    rec_period: Optional[int] = None
    rec_stability = 0.0
    for c in out:
        if c["usable"]:
            rec_period = int(c["period_bars"])
            rec_stability = float(c.get("stability_score", 0.0))
            break
    return out, rec_period, rec_stability


def build_global_artifact(
    asset_artifacts: Sequence[ArtifactResult],
    interval_label: str,
    computed_from_assets_count: int,
    inclusion_rules: str,
    top_k: int,
) -> Optional[ArtifactResult]:
    valid = [a for a in asset_artifacts if a is not None]
    if not valid:
        return None

    comp = _aggregate_global_components(valid, interval_label, top_k=top_k)
    vol_periods, vol_recommended_period, vol_stability_score = _aggregate_global_periods(
        valid, interval_label, period_attr="vol_period_candidates", period_channel="vol"
    )
    return_periods, return_recommended_period, return_stability_score = _aggregate_global_periods(
        valid, interval_label, period_attr="return_period_candidates", period_channel="return"
    )
    periods = sorted(
        list(vol_periods) + list(return_periods),
        key=lambda d: (float(d.get("quality_score", 0.0)), float(d.get("strength_score", 0.0)), float(d.get("stability_score", 0.0))),
        reverse=True,
    )
    rec_period, recommendation_channel, rec_period_channel = _choose_recommendation_channel(
        vol_recommended_period, return_recommended_period
    )

    return ArtifactResult(
        scope="global",
        interval=interval_label,
        asset=None,
        source="global",
        source_priority=10,
        bucket_components=comp,
        period_candidates=periods,
        vol_period_candidates=vol_periods,
        vol_recommended_period=vol_recommended_period,
        vol_stability_score=vol_stability_score,
        return_period_candidates=return_periods,
        return_recommended_period=return_recommended_period,
        return_stability_score=return_stability_score,
        recommendation_channel=recommendation_channel,
        recommended_period_bars=rec_period,
        recommended_period_channel=rec_period_channel,
        computed_from_assets_count=computed_from_assets_count,
        inclusion_rules=inclusion_rules,
    )


def _artifact_overall_summary(artifact: ArtifactResult) -> Dict[str, Any]:
    comp_scores = [float(c["quality_score"]) for c in artifact.bucket_components.values()]
    period_scores = [float(p["quality_score"]) for p in artifact.period_candidates]
    all_scores = comp_scores + period_scores
    score = float(np.mean(all_scores)) if all_scores else 0.0
    use = _use_label(score, scope=artifact.scope)
    usable = use in {"high", "med"}
    return {
        "overall_quality_score": score,
        "overall_recommended_use": use,
        "overall_usable": usable,
    }


def artifact_to_dataframe(artifact: ArtifactResult) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    summary = _artifact_overall_summary(artifact)
    vol_period_candidates_json = json.dumps(artifact.vol_period_candidates, sort_keys=True)
    return_period_candidates_json = json.dumps(artifact.return_period_candidates, sort_keys=True)

    for (family, channel), comp in artifact.bucket_components.items():
        top_k_json = json.dumps(comp.get("top_k", []), sort_keys=True)
        for bucket_idx in range(int(comp["n_buckets"])):
            rows.append(
                {
                    "record_type": "bucket",
                    "scope": artifact.scope,
                    "asset": artifact.asset,
                    "source": artifact.source,
                    "source_priority": artifact.source_priority,
                    "interval": artifact.interval,
                    "recommended_period_bars": artifact.recommended_period_bars,
                    "recommended_period_channel": artifact.recommended_period_channel,
                    "recommendation_channel": artifact.recommendation_channel,
                    "vol_recommended_period": artifact.vol_recommended_period,
                    "vol_stability_score": float(artifact.vol_stability_score),
                    "return_recommended_period": artifact.return_recommended_period,
                    "return_stability_score": float(artifact.return_stability_score),
                    "vol_period_candidates_json": vol_period_candidates_json,
                    "return_period_candidates_json": return_period_candidates_json,
                    "family": family,
                    "channel": channel,
                    "bucket": int(bucket_idx),
                    "bucket_count": int(comp["bucket_counts"][bucket_idx]),
                    "effect_value": float(comp["values"][bucket_idx]),
                    "baseline": float(comp["baseline"]),
                    "signal_to_noise": float(comp["signal_to_noise"]),
                    "stability_score": float(comp["stability_score"]),
                    "quality_score": float(comp["quality_score"]),
                    "recommended_use": comp["recommended_use"],
                    "usable": bool(comp["usable"]),
                    "coverage_bars": int(comp["coverage_bars"]),
                    "coverage_start_ts": comp["coverage_start_ts"],
                    "coverage_end_ts": comp["coverage_end_ts"],
                    "top_k_json": top_k_json,
                    "period_bars": None,
                    "period_channel": None,
                    "strength_score": None,
                    "method": None,
                    "computed_from_assets_count": artifact.computed_from_assets_count,
                    "inclusion_rules": artifact.inclusion_rules,
                    "overall_quality_score": summary["overall_quality_score"],
                    "overall_recommended_use": summary["overall_recommended_use"],
                    "overall_usable": summary["overall_usable"],
                }
            )

    for c in artifact.period_candidates:
        rows.append(
            {
                "record_type": "period_candidate",
                "scope": artifact.scope,
                "asset": artifact.asset,
                "source": artifact.source,
                "source_priority": artifact.source_priority,
                "interval": artifact.interval,
                "recommended_period_bars": artifact.recommended_period_bars,
                "recommended_period_channel": artifact.recommended_period_channel,
                "recommendation_channel": artifact.recommendation_channel,
                "vol_recommended_period": artifact.vol_recommended_period,
                "vol_stability_score": float(artifact.vol_stability_score),
                "return_recommended_period": artifact.return_recommended_period,
                "return_stability_score": float(artifact.return_stability_score),
                "vol_period_candidates_json": vol_period_candidates_json,
                "return_period_candidates_json": return_period_candidates_json,
                "family": None,
                "channel": None,
                "bucket": None,
                "bucket_count": None,
                "effect_value": None,
                "baseline": None,
                "signal_to_noise": None,
                "stability_score": float(c["stability_score"]),
                "quality_score": float(c["quality_score"]),
                "recommended_use": c["recommended_use"],
                "usable": bool(c["usable"]),
                "coverage_bars": int(c["coverage_bars"]),
                "coverage_start_ts": None,
                "coverage_end_ts": None,
                "top_k_json": None,
                "period_bars": int(c["period_bars"]),
                "period_channel": c.get("period_channel"),
                "strength_score": float(c["strength_score"]),
                "method": c.get("method"),
                "computed_from_assets_count": artifact.computed_from_assets_count,
                "inclusion_rules": artifact.inclusion_rules,
                "overall_quality_score": summary["overall_quality_score"],
                "overall_recommended_use": summary["overall_recommended_use"],
                "overall_usable": summary["overall_usable"],
            }
        )

    return pd.DataFrame(rows)


def _write_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = sibling_temp_path(path, suffix=".parquet.tmp")
    df.to_parquet(tmp, engine="pyarrow", index=False, compression=PARQUET_COMPRESSION)
    atomic_replace(tmp, path)


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = sibling_temp_path(path)
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    atomic_replace(tmp, path)


def _asset_output_path(output_root: Path, interval_label: str, asset: str) -> Path:
    return output_root / interval_label / "assets" / asset / "seasonality.parquet"


def _global_output_path(output_root: Path, interval_label: str) -> Path:
    return output_root / interval_label / "global" / "seasonality.parquet"


def _model_feature_output_path(output_root: Path, interval_label: str, asset: str) -> Path:
    return output_root / interval_label / "model_features_v1" / "assets" / asset / "seasonality_features.parquet"


def _manual_context_output_path(output_root: Path, interval_label: str, asset: str) -> Path:
    return output_root / interval_label / "manual_context_v1" / "assets" / asset / "seasonality_context.parquet"


def _feature_diagnostics_path(output_root: Path, interval_label: str) -> Path:
    return output_root / interval_label / "diagnostics" / f"{MODEL_FEATURE_SET_VERSION}.json"


def _manifest_path(output_root: Path, interval_label: str) -> Path:
    return output_root / interval_label / "manifest.json"


def _build_asset_task(
    parquet_root: Path,
    interval_label: str,
    asset: str,
    start_ts: int,
    end_ts: int,
    prefer_scalar_features_true_range: bool,
    smoothing_window: int,
    top_k: int,
    baseline_method: str,
) -> Tuple[str, Optional[ArtifactResult]]:
    art = build_asset_artifact(
        parquet_root=parquet_root,
        interval_label=interval_label,
        asset=asset,
        start_ts=start_ts,
        end_ts=end_ts,
        prefer_scalar_features_true_range=prefer_scalar_features_true_range,
        smoothing_window=smoothing_window,
        top_k=top_k,
        baseline_method=baseline_method,
    )
    return asset, art


def _run_single_interval(args: argparse.Namespace, interval_label: str) -> Dict[str, Any]:
    interval_min = INTERVAL_TO_MIN[interval_label]
    min_asset_coverage_bars = _min_asset_coverage_bars(interval_min, years=ASSET_MIN_HISTORY_YEARS)

    parquet_root = Path(args.parquet_root)
    output_root = Path(args.output_root)

    now_ts = int(datetime.now(timezone.utc).timestamp())
    asset_start_ts = now_ts - int(args.asset_lookback_days) * 86400
    global_start_ts = _parse_date_start_ts(args.global_start_date)

    all_assets = list_assets_for_interval(parquet_root, interval_min)
    bounds = asset_bounds_for_interval(parquet_root, interval_min)

    manifest: Dict[str, Any] = {
        "module": "calendar_seasonality",
        "version": MODULE_VERSION,
        "computed_at": _iso_now(),
        "interval": interval_label,
        "run_scope": args.run_scope,
        "parameters": {
            "global_start_date": args.global_start_date,
            "asset_lookback_days": int(args.asset_lookback_days),
            "workers": int(args.workers),
            "prefer_scalar_features_true_range": bool(args.prefer_scalar_features_true_range),
            "smoothing_window": int(args.smoothing_window),
            "top_k": int(args.top_k),
            "baseline_method": str(args.baseline_method),
            "emit_model_features": bool(args.emit_model_features),
            "feature_lookback_days": int(args.feature_lookback_days),
            "feature_min_bucket_samples": int(args.feature_min_bucket_samples),
            "candidate_periods": CANDIDATE_PERIODS[interval_label],
            "asset_min_history_years": ASSET_MIN_HISTORY_YEARS,
            "min_asset_coverage_bars": min_asset_coverage_bars,
        },
        "assets_produced": [],
        "assets_summary": {},
        "assets_skipped_insufficient_history_count": 0,
        "assets_skipped_insufficient_history": [],
        "model_feature_set": {
            "version": MODEL_FEATURE_SET_VERSION,
            "enabled": bool(args.emit_model_features),
            "assets_produced": [],
            "diagnostics_path": str(_feature_diagnostics_path(output_root, interval_label)),
        },
        "manual_context": {
            "version": MANUAL_CONTEXT_VERSION,
            "enabled": bool(args.emit_model_features),
            "assets_produced": [],
        },
        "global": {},
    }

    built_assets: Dict[str, ArtifactResult] = {}
    assets_eligible_for_asset_profile: List[str] = []
    skipped_insufficient_history: List[str] = []
    feature_diagnostics: List[Dict[str, Any]] = []

    if args.run_scope in {"asset", "both"}:
        for asset in all_assets:
            b = bounds.get(asset)
            if b is None:
                manifest["assets_summary"][asset] = {
                    "asset_profile_status": "insufficient_history_use_global",
                    "reason": "missing_ohlc_bounds",
                }
                continue
            coverage_bars = _coverage_bars_from_bounds(b["min_ts"], b["max_ts"], interval_min)
            if coverage_bars < min_asset_coverage_bars:
                manifest["assets_summary"][asset] = {
                    "asset_profile_status": "insufficient_history_use_global",
                    "coverage_bars": coverage_bars,
                    "required_min_coverage_bars": min_asset_coverage_bars,
                }
                skipped_insufficient_history.append(asset)
                continue
            assets_eligible_for_asset_profile.append(asset)

        workers = max(1, int(args.workers))
        if workers == 1:
            for asset in assets_eligible_for_asset_profile:
                _, art = _build_asset_task(
                    parquet_root,
                    interval_label,
                    asset,
                    asset_start_ts,
                    now_ts,
                    bool(args.prefer_scalar_features_true_range),
                    int(args.smoothing_window),
                    int(args.top_k),
                    str(args.baseline_method),
                )
                if art is None:
                    continue
                built_assets[asset] = art
        else:
            with ThreadPoolExecutor(max_workers=workers) as ex:
                futs = [
                    ex.submit(
                        _build_asset_task,
                        parquet_root,
                        interval_label,
                        asset,
                        asset_start_ts,
                        now_ts,
                        bool(args.prefer_scalar_features_true_range),
                        int(args.smoothing_window),
                        int(args.top_k),
                        str(args.baseline_method),
                    )
                    for asset in assets_eligible_for_asset_profile
                ]
                for fut in as_completed(futs):
                    asset, art = fut.result()
                    if art is None:
                        continue
                    built_assets[asset] = art

        for asset, art in sorted(built_assets.items()):
            df = artifact_to_dataframe(art)
            out = _asset_output_path(output_root, interval_label, asset)
            _write_parquet(df, out)
            summ = _artifact_overall_summary(art)
            manifest["assets_produced"].append(asset)
            manifest["assets_summary"][asset] = {
                "path": str(out),
                "asset_profile_status": "built",
                "recommended_period_bars": art.recommended_period_bars,
                "recommended_period_channel": art.recommended_period_channel,
                "recommendation_channel": art.recommendation_channel,
                "vol_recommended_period": art.vol_recommended_period,
                "vol_stability_score": art.vol_stability_score,
                "return_recommended_period": art.return_recommended_period,
                "return_stability_score": art.return_stability_score,
                "overall_recommended_use": summ["overall_recommended_use"],
                "overall_quality_score": summ["overall_quality_score"],
                "overall_usable": summ["overall_usable"],
            }
            if bool(args.emit_model_features):
                f_df, f_diag = build_asset_model_feature_artifact(
                    parquet_root=parquet_root,
                    interval_label=interval_label,
                    asset=asset,
                    start_ts=asset_start_ts,
                    end_ts=now_ts,
                    prefer_scalar_features_true_range=bool(args.prefer_scalar_features_true_range),
                    bucket_window_days=int(args.feature_lookback_days),
                    min_bucket_samples=int(args.feature_min_bucket_samples),
                )
                feature_diagnostics.append(f_diag)
                if not f_df.empty:
                    f_path = _model_feature_output_path(output_root, interval_label, asset)
                    _write_parquet(f_df, f_path)
                    c_df = manual_context_dataframe(f_df)
                    c_path = _manual_context_output_path(output_root, interval_label, asset)
                    _write_parquet(c_df, c_path)
                    manifest["assets_summary"][asset]["model_feature_path"] = str(f_path)
                    manifest["assets_summary"][asset]["manual_context_path"] = str(c_path)
                    manifest["assets_summary"][asset]["model_feature_rows"] = int(len(f_df))
                    manifest["model_feature_set"]["assets_produced"].append(asset)
                    manifest["manual_context"]["assets_produced"].append(asset)

        manifest["assets_skipped_insufficient_history_count"] = len(skipped_insufficient_history)
        manifest["assets_skipped_insufficient_history"] = sorted(skipped_insufficient_history)

    if bool(args.emit_model_features):
        rows = int(sum(int(d.get("output_rows", 0) or 0) for d in feature_diagnostics))
        missing_keys = sorted(
            {
                k
                for d in feature_diagnostics
                for k in (d.get("missingness_summary", {}) or {}).keys()
            }
        )
        missingness = {
            k: float(np.nanmean([float((d.get("missingness_summary", {}) or {}).get(k, np.nan)) for d in feature_diagnostics]))
            for k in missing_keys
        }
        diag_payload = {
            "generated_at_utc": _iso_now(),
            "interval": interval_label,
            "feature_set_version": MODEL_FEATURE_SET_VERSION,
            "manual_context_version": MANUAL_CONTEXT_VERSION,
            "asof_leakage_policy": "strictly_prior_same_calendar_bucket_current_row_excluded",
            "input_asset_count": int(len(feature_diagnostics)),
            "output_row_count": rows,
            "bucket_window_days": int(args.feature_lookback_days),
            "bucket_min_samples": int(args.feature_min_bucket_samples),
            "missingness_summary_mean_by_asset": missingness,
            "quality_flag_counts": {
                "ok_assets": int(sum(1 for d in feature_diagnostics if d.get("status") == "built")),
                "empty_input_assets": int(sum(1 for d in feature_diagnostics if d.get("status") == "empty_input")),
            },
            "return_seasonality_policy": "diagnostic_only_not_trusted_model_feature",
            "multiple_testing_warning": "return bucket diagnostics require out-of-sample validation and multiple-testing control before directional use",
            "static_full_history_baseline_warning": "legacy profile artifacts remain descriptive; model_features_v1 uses rolling as-of bucket baselines",
            "model_facing_feature_availability": MODEL_FEATURE_COLUMNS,
            "asset_diagnostics": feature_diagnostics,
        }
        d_path = _feature_diagnostics_path(output_root, interval_label)
        _write_json(d_path, diag_payload)

    if args.run_scope in {"global", "both"}:
        eligible_assets = [
            a for a in all_assets if a in bounds and bounds[a]["min_ts"] <= global_start_ts and bounds[a]["max_ts"] >= now_ts
        ]
        if not eligible_assets:
            eligible_assets = [
                a
                for a in all_assets
                if a in bounds and bounds[a]["min_ts"] <= global_start_ts and bounds[a]["max_ts"] >= (now_ts - interval_min * 60 * 2)
            ]

        assets_for_global: List[ArtifactResult] = []
        for asset in eligible_assets:
            art = build_asset_artifact(
                parquet_root=parquet_root,
                interval_label=interval_label,
                asset=asset,
                start_ts=global_start_ts,
                end_ts=now_ts,
                prefer_scalar_features_true_range=bool(args.prefer_scalar_features_true_range),
                smoothing_window=int(args.smoothing_window),
                top_k=int(args.top_k),
                baseline_method=str(args.baseline_method),
            )
            if art is not None:
                assets_for_global.append(art)

        inclusion_rules = (
            f"asset included if ohlcvt_{interval_min} coverage has min_ts <= {args.global_start_date} 00:00:00 UTC "
            f"and max_ts near now; no safe/noisy label filtering"
        )

        g_art = build_global_artifact(
            asset_artifacts=assets_for_global,
            interval_label=interval_label,
            computed_from_assets_count=len(assets_for_global),
            inclusion_rules=inclusion_rules,
            top_k=int(args.top_k),
        )

        if g_art is not None:
            g_df = artifact_to_dataframe(g_art)
            g_path = _global_output_path(output_root, interval_label)
            _write_parquet(g_df, g_path)
            g_sum = _artifact_overall_summary(g_art)
            manifest["global"] = {
                "path": str(g_path),
                "computed_from_assets_count": len(assets_for_global),
                "eligible_assets": eligible_assets,
                "included_assets": [a.asset for a in assets_for_global],
                "inclusion_rules": inclusion_rules,
                "inclusion_basis": "coverage_only",
                "seasonality_strength_filter_applied": False,
                "aggregation_policy": "consensus_first_per_asset_normalized_no_mega_asset_dominance",
                "recommended_period_bars": g_art.recommended_period_bars,
                "recommended_period_channel": g_art.recommended_period_channel,
                "recommendation_channel": g_art.recommendation_channel,
                "vol_recommended_period": g_art.vol_recommended_period,
                "vol_stability_score": g_art.vol_stability_score,
                "return_recommended_period": g_art.return_recommended_period,
                "return_stability_score": g_art.return_stability_score,
                "overall_recommended_use": g_sum["overall_recommended_use"],
                "overall_quality_score": g_sum["overall_quality_score"],
                "overall_usable": g_sum["overall_usable"],
            }
        else:
            manifest["global"] = {
                "path": None,
                "computed_from_assets_count": 0,
                "eligible_assets": eligible_assets,
                "included_assets": [],
                "inclusion_rules": inclusion_rules,
                "inclusion_basis": "coverage_only",
                "seasonality_strength_filter_applied": False,
                "aggregation_policy": "consensus_first_per_asset_normalized_no_mega_asset_dominance",
                "recommended_period_bars": None,
                "recommended_period_channel": None,
                "recommendation_channel": "none",
                "vol_recommended_period": None,
                "vol_stability_score": 0.0,
                "return_recommended_period": None,
                "return_stability_score": 0.0,
                "overall_recommended_use": "none",
                "overall_quality_score": 0.0,
                "overall_usable": False,
            }

    m_path = _manifest_path(output_root, interval_label)
    _write_json(m_path, manifest)
    manifest["manifest_path"] = str(m_path)
    return manifest


def run(args: argparse.Namespace) -> Dict[str, Any]:
    if args.interval in SUPPORTED_INTERVALS:
        interval_labels = [args.interval]
    else:
        interval_labels = list(SUPPORTED_INTERVALS)

    per_interval: Dict[str, Any] = {}
    for interval_label in interval_labels:
        per_interval[interval_label] = _run_single_interval(args, interval_label)

    if len(interval_labels) == 1:
        return per_interval[interval_labels[0]]

    return {
        "module": "calendar_seasonality",
        "version": MODULE_VERSION,
        "computed_at": _iso_now(),
        "run_scope": args.run_scope,
        "intervals_processed": interval_labels,
        "per_interval": per_interval,
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build calendar-only seasonality artifacts for downstream stats models.")
    p.add_argument("--interval", choices=[*SUPPORTED_INTERVALS, "all"], default="all")
    p.add_argument("--run_scope", choices=["asset", "global", "both"], default="both")
    p.add_argument("--global_start_date", type=str, default="2021-01-01")
    p.add_argument("--asset_lookback_days", type=int, default=730)
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) // 2))
    p.add_argument("--prefer_scalar_features_true_range", type=_boolean_arg, default=True)
    p.add_argument("--smoothing_window", type=int, default=5)
    p.add_argument("--top_k", type=int, default=10)
    p.add_argument("--baseline_method", choices=["median", "trimmed_mean"], default="median")
    p.add_argument("--emit_model_features", type=_boolean_arg, default=False)
    p.add_argument("--feature_lookback_days", type=int, default=DEFAULT_FEATURE_LOOKBACK_DAYS)
    p.add_argument("--feature_min_bucket_samples", type=int, default=DEFAULT_FEATURE_MIN_BUCKET_SAMPLES)
    p.add_argument("--parquet_root", type=str, default=str(DEFAULT_PARQUET_ROOT))
    p.add_argument("--output_root", type=str, default=str(DEFAULT_OUTPUT_ROOT))
    return p


def main() -> int:
    require_pipeline_io(profile=PIPELINE_PROFILE)
    parser = build_parser()
    args = parser.parse_args()
    manifest = run(args)
    if "per_interval" in manifest:
        per_interval_summary: Dict[str, Any] = {}
        for interval_label, m in (manifest.get("per_interval", {}) or {}).items():
            per_interval_summary[interval_label] = {
                "assets_produced": len((m.get("assets_produced", []) if isinstance(m, dict) else [])),
                "model_feature_assets_produced": len(
                    (
                        ((m.get("model_feature_set", {}) if isinstance(m, dict) else {}).get("assets_produced", []))
                        or []
                    )
                ),
                "global_computed_from_assets_count": int(
                    ((m.get("global", {}) if isinstance(m, dict) else {}).get("computed_from_assets_count", 0) or 0)
                ),
                "manifest_path": (m.get("manifest_path") if isinstance(m, dict) else None),
            }
        payload = {
            "status": "ok",
            "interval": "all",
            "run_scope": args.run_scope,
            "intervals_processed": manifest.get("intervals_processed", []),
            "per_interval": per_interval_summary,
        }
    else:
        payload = {
            "status": "ok",
            "interval": args.interval,
            "run_scope": args.run_scope,
            "assets_produced": len(manifest.get("assets_produced", [])),
            "model_feature_assets_produced": len((manifest.get("model_feature_set", {}) or {}).get("assets_produced", []) or []),
            "global_computed_from_assets_count": int(manifest.get("global", {}).get("computed_from_assets_count", 0) or 0),
            "manifest_path": manifest.get("manifest_path"),
        }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
