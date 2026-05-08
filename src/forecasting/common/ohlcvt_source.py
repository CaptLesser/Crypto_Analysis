from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import pandas as pd

from src.forecasting.common.path_config import resolve_path


OHLCVT_COLUMNS = ["asset", "ts", "open", "high", "low", "close", "volume", "trades"]


def _default_ohlcvt_root() -> Optional[Path]:
    return resolve_path("source_ohlcvt_root", required=False)


def _parse_ym(value: str | Tuple[int, int] | None) -> Optional[Tuple[int, int]]:
    if value is None:
        return None
    if isinstance(value, tuple):
        y, m = int(value[0]), int(value[1])
        return y, m
    s = str(value).strip()
    parts = s.split("-")
    if len(parts) != 2:
        raise ValueError(f"Invalid YM value: {value}")
    return int(parts[0]), int(parts[1])


def _iter_months(start_ym: Tuple[int, int], end_ym: Tuple[int, int]) -> Iterable[Tuple[int, int]]:
    y, m = int(start_ym[0]), int(start_ym[1])
    ey, em = int(end_ym[0]), int(end_ym[1])
    while (y < ey) or (y == ey and m <= em):
        yield y, m
        if m == 12:
            y += 1
            m = 1
        else:
            m += 1


def _month_paths_for(
    *,
    interval_dir: Path,
    asset: str,
    year: int,
    month: int,
) -> List[Path]:
    new_month_dir = interval_dir / f"asset={asset}" / f"year={year}" / f"month={month:02d}"
    new_paths = sorted(new_month_dir.glob("*.parquet"))
    if new_paths:
        return new_paths

    old_month_dir = interval_dir / f"year={year}" / f"month={month:02d}"
    return sorted(old_month_dir.glob("*.parquet"))


def _discover_available_months(interval_dir: Path, asset: str) -> List[Tuple[int, int]]:
    months: set[Tuple[int, int]] = set()

    new_asset_dir = interval_dir / f"asset={asset}"
    if new_asset_dir.exists():
        for year_dir in new_asset_dir.glob("year=*"):
            y_txt = year_dir.name.replace("year=", "")
            if not y_txt.isdigit():
                continue
            y = int(y_txt)
            for month_dir in year_dir.glob("month=*"):
                m_txt = month_dir.name.replace("month=", "")
                if m_txt.isdigit():
                    months.add((y, int(m_txt)))

    for year_dir in interval_dir.glob("year=*"):
        y_txt = year_dir.name.replace("year=", "")
        if not y_txt.isdigit():
            continue
        y = int(y_txt)
        for month_dir in year_dir.glob("month=*"):
            m_txt = month_dir.name.replace("month=", "")
            if m_txt.isdigit():
                months.add((y, int(m_txt)))

    return sorted(months)


def list_month_partitions(
    *,
    family: str = "ohlcvt",
    interval_min: int,
    asset: str,
    start_ym: str | Tuple[int, int] | None = None,
    end_ym: str | Tuple[int, int] | None = None,
    root: Optional[Path] = None,
) -> List[Path]:
    fam = str(family).strip().lower()
    if fam != "ohlcvt":
        raise ValueError(f"Unsupported family: {family}")
    if not asset:
        raise ValueError("asset must be provided")

    root_path = Path(root) if root else _default_ohlcvt_root()
    if root_path is None:
        return []
    interval_dir = root_path / f"ohlcvt_{int(interval_min)}"
    if not interval_dir.exists():
        return []

    s_ym = _parse_ym(start_ym)
    e_ym = _parse_ym(end_ym)
    if s_ym and not e_ym:
        e_ym = s_ym
    if e_ym and not s_ym:
        s_ym = e_ym

    out: List[Path] = []
    if s_ym is not None and e_ym is not None:
        for y, m in _iter_months(s_ym, e_ym):
            out.extend(_month_paths_for(interval_dir=interval_dir, asset=asset, year=y, month=m))
        return out

    for y, m in _discover_available_months(interval_dir, asset):
        out.extend(_month_paths_for(interval_dir=interval_dir, asset=asset, year=y, month=m))
    return out


def read_ohlcvt(
    *,
    asset: str,
    interval_min: int,
    start_ts: Optional[int] = None,
    end_ts: Optional[int] = None,
    columns: Optional[Sequence[str]] = None,
    root: Optional[Path] = None,
) -> pd.DataFrame:
    if start_ts is not None and end_ts is not None and int(end_ts) < int(start_ts):
        return pd.DataFrame(columns=list(columns) if columns else OHLCVT_COLUMNS)

    start_ym: Optional[Tuple[int, int]] = None
    end_ym: Optional[Tuple[int, int]] = None
    if start_ts is not None:
        dt = pd.to_datetime(int(start_ts), unit="s", utc=True)
        start_ym = (int(dt.year), int(dt.month))
    if end_ts is not None:
        dt = pd.to_datetime(int(end_ts), unit="s", utc=True)
        end_ym = (int(dt.year), int(dt.month))

    paths = list_month_partitions(
        family="ohlcvt",
        interval_min=int(interval_min),
        asset=str(asset),
        start_ym=start_ym,
        end_ym=end_ym,
        root=root,
    )
    if not paths:
        return pd.DataFrame(columns=list(columns) if columns else OHLCVT_COLUMNS)

    selected_cols = list(dict.fromkeys(list(columns) if columns else OHLCVT_COLUMNS))
    required_cols = set(selected_cols)
    required_cols.add("asset")
    required_cols.add("ts")
    read_cols = [c for c in OHLCVT_COLUMNS if c in required_cols]

    try:
        import pyarrow.dataset as ds  # type: ignore

        filt = ds.field("asset") == str(asset)
        if start_ts is not None:
            filt = filt & (ds.field("ts") >= int(start_ts))
        if end_ts is not None:
            filt = filt & (ds.field("ts") <= int(end_ts))

        dataset = ds.dataset([str(p) for p in paths], format="parquet")
        table = dataset.to_table(filter=filt, columns=read_cols)
        df = table.to_pandas()
    except Exception:
        frames: List[pd.DataFrame] = []
        for p in paths:
            try:
                d = pd.read_parquet(p, columns=read_cols)
            except Exception:
                continue
            d = d[d["asset"].astype(str) == str(asset)]
            if start_ts is not None:
                d = d[d["ts"] >= int(start_ts)]
            if end_ts is not None:
                d = d[d["ts"] <= int(end_ts)]
            if not d.empty:
                frames.append(d)
        if not frames:
            return pd.DataFrame(columns=selected_cols)
        df = pd.concat(frames, ignore_index=True)

    if df.empty:
        return pd.DataFrame(columns=selected_cols)

    df = df.drop_duplicates(subset=["asset", "ts"], keep="last").sort_values(["asset", "ts"]).reset_index(drop=True)
    for c in selected_cols:
        if c not in df.columns:
            df[c] = pd.NA
    return df[selected_cols]


def list_assets_ohlcvt(interval_min: int, root: Optional[Path] = None) -> List[str]:
    root_path = Path(root) if root else _default_ohlcvt_root()
    if root_path is None:
        return []
    interval_dir = root_path / f"ohlcvt_{int(interval_min)}"
    if not interval_dir.exists():
        return []

    new_assets = sorted(
        {
            p.name.replace("asset=", "")
            for p in interval_dir.glob("asset=*")
            if p.is_dir() and p.name.startswith("asset=") and len(p.name) > len("asset=")
        }
    )
    if new_assets:
        return new_assets

    paths = sorted(interval_dir.rglob("*.parquet"))
    if not paths:
        return []
    try:
        import pyarrow.dataset as ds  # type: ignore

        dataset = ds.dataset([str(p) for p in paths], format="parquet")
        assets = dataset.to_table(columns=["asset"]).column(0).unique().to_pylist()
        return sorted({str(a) for a in assets if a is not None})
    except Exception:
        assets: set[str] = set()
        for p in paths:
            try:
                d = pd.read_parquet(p, columns=["asset"])
            except Exception:
                continue
            assets.update(d["asset"].dropna().astype(str).tolist())
        return sorted(assets)


def ohlcvt_bounds(interval_min: int, asset: str, root: Optional[Path] = None) -> Tuple[Optional[int], Optional[int]]:
    df = read_ohlcvt(asset=str(asset), interval_min=int(interval_min), columns=["ts"], root=root)
    if df.empty:
        return (None, None)
    return (int(df["ts"].min()), int(df["ts"].max()))
