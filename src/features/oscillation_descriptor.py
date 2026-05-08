from __future__ import annotations
import argparse
import math
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

from src.forecasting.common.ohlcvt_source import read_ohlcvt
try:
    from PyEMD import CEEMDAN  # type: ignore
except Exception:
    CEEMDAN = None

try:
    import hdbscan  # type: ignore
except Exception:
    hdbscan = None

try:
    from scipy.signal import find_peaks  # type: ignore
except Exception:
    find_peaks = None

from src.features.scalar_features import PARQUET_ROOT, OHLCVT_PARQUET_ROOT, log as base_log

LOOKBACK_DAYS_DEFAULT = 180
MIN_OSC_PCT = float(os.getenv("OSC_MIN_PCT", "0.02"))
DEFAULT_INTERVAL = "1m"
FALLBACK_INTERVAL = "5m"
OUTPUT_PATH = Path(os.getenv("OSC_OUTPUT_PATH", str(Path(__file__).with_name("oscillation_descriptors.csv"))))


def log(msg: str) -> None:
    base_log(f"[osc_desc] {msg}")


def month_range(start_ts: int, end_ts: int) -> List[Tuple[int, int]]:
    start_dt = pd.to_datetime(start_ts, unit="s", utc=True)
    end_dt = pd.to_datetime(end_ts, unit="s", utc=True)
    cur = pd.Timestamp(year=start_dt.year, month=start_dt.month, day=1, tz="UTC")
    end_marker = pd.Timestamp(year=end_dt.year, month=end_dt.month, day=1, tz="UTC")
    out = []
    while cur <= end_marker:
        out.append((cur.year, cur.month))
        cur = cur + pd.DateOffset(months=1)
    return out


def mad(series: pd.Series) -> float:
    s = series.dropna()
    if s.empty:
        return 0.0
    med = s.median()
    return float((s - med).abs().median())


def percentile_rank(value: float, population: np.ndarray) -> float:
    pop = population[~np.isnan(population)]
    if pop.size == 0 or math.isnan(value):
        return float("nan")
    return float((pop <= value).mean())


def load_scalar_features(asset: str, interval: str, start_ts: int, end_ts: int) -> pd.DataFrame:
    mins = int(interval.replace("m", ""))
    base = Path(PARQUET_ROOT) / f"scalar_features_{mins}"
    if not base.exists():
        return pd.DataFrame()
    cols = ["ts", "asset", "weighted_close"]
    dfs: List[pd.DataFrame] = []
    for y, m in month_range(start_ts, end_ts):
        p = base / f"year={y}/month={m:02d}"
        if not p.exists():
            continue
        for pq in p.glob("*.parquet"):
            try:
                df = pd.read_parquet(pq, columns=cols)
            except Exception:
                continue
            df = df[df["asset"] == asset]
            if df.empty:
                continue
            df = df[(df["ts"] >= start_ts) & (df["ts"] <= end_ts)]
            if not df.empty:
                dfs.append(df)
    if not dfs:
        return pd.DataFrame()
    out = pd.concat(dfs, ignore_index=True)
    return out.sort_values("ts")


def load_ohlc_volume(asset: str, interval: str, start_ts: int, end_ts: int) -> pd.DataFrame:
    mins = int(interval.replace("m", ""))
    cols = ["ts", "asset", "volume"]
    out = read_ohlcvt(
        asset=str(asset),
        interval_min=mins,
        start_ts=int(start_ts),
        end_ts=int(end_ts),
        columns=cols,
        root=Path(OHLCVT_PARQUET_ROOT),
    )
    if out.empty:
        return pd.DataFrame()
    return out.sort_values("ts")


def decompose(signal: np.ndarray) -> Tuple[str, List[np.ndarray]]:
    if CEEMDAN is None:
        raise RuntimeError("CEEMDAN not available; install PyEMD.")
    ce = CEEMDAN()
    imfs = list(ce(signal))
    return "CEEMDAN", imfs


def peak_indices(imf: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if find_peaks is None:
        return np.array([], dtype=int), np.array([], dtype=int)
    peaks, _ = find_peaks(imf)
    troughs, _ = find_peaks(-imf)
    return peaks, troughs


@dataclass
class OscEvent:
    start_ts: int
    end_ts: int
    amp_pct: float
    turnaround_seconds: float
    period_seconds: float
    volume_confirmation: float


def extract_events(
    imf: np.ndarray,
    ts: np.ndarray,
    price_raw: np.ndarray,
    v_sig: np.ndarray,
    v_all: np.ndarray,
    min_amp_pct: float,
    recent_cutoff: int,
) -> Tuple[List[OscEvent], int]:
    peaks, troughs = peak_indices(imf)
    turning = [(i, "peak") for i in peaks] + [(i, "trough") for i in troughs]
    turning = sorted(turning, key=lambda x: x[0])
    events: List[OscEvent] = []
    recent_count = 0

    # Period estimation based on same-type distances
    same_type_periods: List[float] = []
    last_peak = None
    last_trough = None
    for idx, typ in turning:
        if typ == "peak":
            if last_peak is not None:
                same_type_periods.append(ts[idx] - ts[last_peak])
            last_peak = idx
        else:
            if last_trough is not None:
                same_type_periods.append(ts[idx] - ts[last_trough])
            last_trough = idx
    est_period = np.median(same_type_periods) if same_type_periods else np.nan

    for (i, t1), (j, t2) in zip(turning[:-1], turning[1:]):
        start_ts = int(ts[i])
        end_ts = int(ts[j])
        if end_ts <= start_ts:
            continue
        if i >= len(price_raw) or j >= len(price_raw):
            continue
        p0 = price_raw[i]
        p1 = price_raw[j]
        if p0 is None or p1 is None or np.isnan(p0) or np.isnan(p1) or p0 == 0:
            continue
        amp_pct = abs(p1 - p0) / abs(p0)
        if amp_pct < min_amp_pct:
            continue
        ta_seconds = float(end_ts - start_ts)
        if math.isnan(est_period):
            period_seconds = ta_seconds * 2.0
        else:
            period_seconds = float(est_period)

        win_mask = (ts >= start_ts) & (ts <= end_ts)
        v_win_mean = float(np.nanmean(v_sig[win_mask])) if np.any(win_mask) else float("nan")
        vol_conf = percentile_rank(v_win_mean, v_all) if v_all.size > 0 else float("nan")

        events.append(OscEvent(start_ts, end_ts, amp_pct, ta_seconds, period_seconds, vol_conf))
        if end_ts >= recent_cutoff:
            recent_count += 1

    return events, recent_count


def imf_energy(imf: np.ndarray) -> float:
    if imf.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(imf))))


def build_imf_row(
    asset: str,
    interval: str,
    imf_idx: int,
    method: str,
    events: List[OscEvent],
    energy: float,
    window_start: int,
    window_end: int,
    recent_count: int,
) -> Dict[str, object]:
    cycle_count = len(events)
    low_support = 1 if cycle_count < 3 else 0
    if cycle_count == 0:
        med_amp = mad_amp = med_ta = mad_ta = med_period = mad_period = float("nan")
        med_vol = mad_vol = float("nan")
    else:
        amp_series = pd.Series([e.amp_pct for e in events])
        ta_series = pd.Series([e.turnaround_seconds for e in events])
        period_series = pd.Series([e.period_seconds for e in events])
        vol_series = pd.Series([e.volume_confirmation for e in events])
        med_amp = float(amp_series.median())
        mad_amp = mad(amp_series)
        med_ta = float(ta_series.median())
        mad_ta = mad(ta_series)
        med_period = float(period_series.median())
        mad_period = mad(period_series)
        med_vol = float(vol_series.median())
        mad_vol = mad(vol_series)

    amp_stability = float(min(mad_amp / med_amp, 10.0)) if med_amp and not math.isnan(med_amp) and med_amp != 0 else float("nan")
    period_stability = float(min(mad_period / med_period, 10.0)) if med_period and not math.isnan(med_period) and med_period != 0 else float("nan")

    return {
        "asset": asset,
        "interval": interval,
        "imf_index": imf_idx,
        "cycle_count": cycle_count,
        "median_amp_pct": med_amp,
        "mad_amp_pct": mad_amp,
        "median_turnaround_seconds": med_ta,
        "mad_turnaround_seconds": mad_ta,
        "median_period_seconds": med_period,
        "mad_period_seconds": mad_period,
        "amp_stability": amp_stability,
        "period_stability": period_stability,
        "recent_activity_14d": recent_count,
        "energy": energy,
        "median_volume_confirmation": med_vol,
        "mad_volume_confirmation": mad_vol,
        "decomp_method": method,
        "window_start_ts": window_start,
        "window_end_ts": window_end,
        "low_support": low_support,
    }


def cluster_imfs_asset(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    if hdbscan is None:
        raise RuntimeError("hdbscan is required for clustering.")
    features = [
        "median_amp_pct",
        "mad_amp_pct",
        "median_period_seconds",
        "mad_period_seconds",
        "median_turnaround_seconds",
        "mad_turnaround_seconds",
        "energy",
        "cycle_count",
        "median_volume_confirmation",
    ]
    x = df[features].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=float)
    if len(x) < 2:
        df["cluster_label"] = 0
        df["cluster_strength"] = np.nan
        df["outlier_score"] = np.nan
        return df
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    clusterer = hdbscan.HDBSCAN(min_cluster_size=3, min_samples=2, allow_single_cluster=True)
    labels = clusterer.fit_predict(x_scaled)
    strength = clusterer.probabilities_
    outlier = clusterer.outlier_scores_
    # Remap non-noise labels to consecutive starting at 0
    unique_labels = sorted(set(labels) - {-1})
    label_map = {lab: i for i, lab in enumerate(unique_labels)}
    mapped = [label_map.get(l, -1) for l in labels]
    df["cluster_label"] = mapped
    df["cluster_strength"] = strength
    df["outlier_score"] = outlier
    return df


def process_asset(asset: str, interval: str, lookback_days: int) -> pd.DataFrame:
    now_ts = int(pd.Timestamp.utcnow().timestamp())
    start_ts = now_ts - lookback_days * 24 * 3600

    # Primary interval attempt
    feat = load_scalar_features(asset, interval, start_ts, now_ts)
    vol = load_ohlc_volume(asset, interval, start_ts, now_ts)
    used_interval = interval
    if (feat.empty or vol.empty) and interval == DEFAULT_INTERVAL:
        # Fallback to 5m if 1m missing
        feat = load_scalar_features(asset, FALLBACK_INTERVAL, start_ts, now_ts)
        vol = load_ohlc_volume(asset, FALLBACK_INTERVAL, start_ts, now_ts)
        used_interval = FALLBACK_INTERVAL
    if feat.empty or vol.empty:
        log(f"{asset}: no data for interval {interval} (fallback {used_interval} also empty)")
        return pd.DataFrame()
    df = pd.merge(feat[["ts", "weighted_close"]], vol[["ts", "volume"]], on="ts", how="inner")
    df = df.sort_values("ts").dropna(subset=["weighted_close", "volume"])
    if df.empty:
        return pd.DataFrame()

    t = df["ts"].astype(int).to_numpy()
    p_raw = df["weighted_close"].astype(float).to_numpy()
    v_raw = df["volume"].astype(float).to_numpy()

    p_sig = np.log(p_raw)
    v_sig = np.log1p(v_raw)

    price_method, price_imfs = decompose(p_sig)
    _ = decompose(v_sig)
    decomp_method = price_method

    v_all = v_sig.copy()
    v_all = v_all[~np.isnan(v_all)]
    recent_cutoff = now_ts - 14 * 24 * 3600

    rows: List[Dict[str, object]] = []
    for idx, imf in enumerate(price_imfs):
        events, recent_count = extract_events(imf, t, p_raw, v_sig, v_all, MIN_OSC_PCT, recent_cutoff)
        energy = imf_energy(imf)
        row = build_imf_row(
            asset=asset,
            interval=used_interval,
            imf_idx=idx,
            method=decomp_method,
            events=events,
            energy=energy,
            window_start=int(t[0]),
            window_end=int(t[-1]),
            recent_count=recent_count,
        )
        rows.append(row)

    if not rows:
        return pd.DataFrame()
    df_rows = pd.DataFrame(rows)
    df_rows = cluster_imfs_asset(df_rows)
    return df_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute oscillation descriptors and clusters for assets.")
    parser.add_argument("--assets", type=str, default="", help="Comma list of assets; if empty, read from choppy_assets.csv")
    parser.add_argument("--interval", type=str, default=DEFAULT_INTERVAL, help='Interval like "1m" or "5m"')
    parser.add_argument("--lookback-days", type=int, default=LOOKBACK_DAYS_DEFAULT, help="Lookback window in days (default 180)")
    args = parser.parse_args()

    if CEEMDAN is None:
        raise RuntimeError("CEEMDAN is required; install PyEMD.")
    if hdbscan is None:
        raise RuntimeError("hdbscan is required for clustering.")

    interval = args.interval.lower()
    if interval not in ("1m", "5m"):
        log(f"unsupported interval={interval}; use 1m or 5m")
        return

    if args.assets:
        assets = [a.strip() for a in args.assets.split(",") if a.strip()]
    else:
        default_path = Path(__file__).with_name("choppy_assets.csv")
        if default_path.exists():
            try:
                df = pd.read_csv(default_path)
                assets = df["asset"].dropna().astype(str).tolist()
            except Exception:
                assets = []
        else:
            assets = []
    if not assets:
        log("no assets provided; exiting")
        return

    all_dfs: List[pd.DataFrame] = []
    for asset in assets:
        log(f"processing {asset} interval={interval}")
        rows = process_asset(asset, interval, args.lookback_days)
        if isinstance(rows, pd.DataFrame) and not rows.empty:
            all_dfs.append(rows)

    if not all_dfs:
        log("no oscillation rows produced; exiting")
        return

    df_out = pd.concat(all_dfs, ignore_index=True)
    df_out.to_csv(OUTPUT_PATH, index=False)
    log(f"wrote {len(df_out)} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
