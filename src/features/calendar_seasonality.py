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
from src.forecasting.common.ohlcvt_source import list_assets_ohlcvt, ohlcvt_bounds, read_ohlcvt
from src.forecasting.common.io_atomic import atomic_replace, sibling_temp_path
from src.forecasting.common.path_config import require_pipeline_io, resolve_path, selected_profile


MODULE_VERSION = "1.0.0"
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
    cols = ["asset", "ts", "high", "low", "close"]
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
    for c in ("high", "low", "close"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["ts", "high", "low", "close"])
    df = df.sort_values("ts").drop_duplicates(subset=["ts"], keep="last").reset_index(drop=True)
    df["asset"] = asset
    return df


def read_scalar_true_range_window(
    parquet_root: Path,
    interval_min: int,
    asset: str,
    start_ts: int,
    end_ts: int,
) -> pd.DataFrame:
    paths: List[Path] = []
    for y, m in _iter_months_between(start_ts, end_ts):
        p = _scalar_path(parquet_root, interval_min, y, m)
        if p.exists():
            paths.append(p)
    if not paths:
        return pd.DataFrame(columns=["ts", "true_range"])

    cols = ["asset", "ts", "true_range"]
    try:
        import pyarrow.dataset as ds  # type: ignore

        dataset = ds.dataset([str(p) for p in paths], format="parquet")
        schema_cols = set(dataset.schema.names)
        if "true_range" not in schema_cols:
            return pd.DataFrame(columns=["ts", "true_range"])
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
            if "true_range" not in d.columns:
                continue
            d = d[["asset", "ts", "true_range"]]
            d = d[(d["asset"] == asset) & (d["ts"] >= int(start_ts)) & (d["ts"] <= int(end_ts))]
            if not d.empty:
                frames.append(d)
        if not frames:
            return pd.DataFrame(columns=["ts", "true_range"])
        df = pd.concat(frames, ignore_index=True)

    if df.empty:
        return pd.DataFrame(columns=["ts", "true_range"])
    df["ts"] = pd.to_numeric(df["ts"], errors="coerce")
    df["true_range"] = pd.to_numeric(df["true_range"], errors="coerce")
    df = df.dropna(subset=["ts"]).sort_values("ts").drop_duplicates(subset=["ts"], keep="last")
    return df[["ts", "true_range"]].reset_index(drop=True)


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

    if prefer_scalar_features_true_range:
        tr_df = read_scalar_true_range_window(parquet_root, interval_min, asset, start_ts, end_ts)
        if not tr_df.empty:
            merged = ohlc[["ts"]].merge(tr_df, on="ts", how="left")
            tr = pd.to_numeric(merged["true_range"], errors="coerce")
            tr = tr.fillna(tr_calc)
        else:
            tr = tr_calc
    else:
        tr = tr_calc

    out = ohlc[["asset", "ts", "high", "low", "close"]].copy()
    out["true_range"] = pd.to_numeric(tr, errors="coerce")
    close = pd.to_numeric(out["close"], errors="coerce")
    out["log_return"] = np.log(close / close.shift(1))
    out = _build_calendar_columns(out, interval_label)
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
            "candidate_periods": CANDIDATE_PERIODS[interval_label],
            "asset_min_history_years": ASSET_MIN_HISTORY_YEARS,
            "min_asset_coverage_bars": min_asset_coverage_bars,
        },
        "assets_produced": [],
        "assets_summary": {},
        "assets_skipped_insufficient_history_count": 0,
        "assets_skipped_insufficient_history": [],
        "global": {},
    }

    built_assets: Dict[str, ArtifactResult] = {}
    assets_eligible_for_asset_profile: List[str] = []
    skipped_insufficient_history: List[str] = []

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

        manifest["assets_skipped_insufficient_history_count"] = len(skipped_insufficient_history)
        manifest["assets_skipped_insufficient_history"] = sorted(skipped_insufficient_history)

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
            "global_computed_from_assets_count": int(manifest.get("global", {}).get("computed_from_assets_count", 0) or 0),
            "manifest_path": manifest.get("manifest_path"),
        }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
