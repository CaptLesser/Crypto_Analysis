from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

FIXED_NUMERIC_FAMILY_COHORT_SYMBOLS: Tuple[str, ...] = (
    "BTC",
    "SOL",
    "ETH",
    "ADA",
    "XRP",
    "LTC",
    "AVAX",
    "XMR",
)
COMMON_QUOTE_SUFFIXES: Tuple[str, ...] = (
    "USDT",
    "USDC",
    "USD",
    "EUR",
    "BTC",
    "ETH",
)
CLAMP_START_YEAR = 2025
CLAMP_START_MONTH = 1
DEFAULT_COHORT_WINDOW_MONTHS = 13
DEFAULT_SEARCH_BACK_MONTHS = 12
SYMBOL_CANONICAL_MAP: Dict[str, str] = {"XBT": "BTC"}


@dataclass(frozen=True)
class MonthKey:
    year: int
    month: int


def add_months(year: int, month: int, delta: int) -> MonthKey:
    idx = year * 12 + (month - 1) + int(delta)
    return MonthKey(idx // 12, idx % 12 + 1)


def month_seq(end_month: MonthKey, count: int) -> List[MonthKey]:
    return [add_months(end_month.year, end_month.month, -(count - 1 - i)) for i in range(count)]


def months_between(start: MonthKey, end_exclusive: MonthKey) -> int:
    return (end_exclusive.year * 12 + (end_exclusive.month - 1)) - (start.year * 12 + (start.month - 1))


def parse_asset_name(dirname: str) -> Optional[str]:
    if not dirname.startswith("asset="):
        return None
    value = dirname.split("=", 1)[1].strip()
    return value or None


def list_assets(table_root: Path) -> List[str]:
    if not table_root.exists():
        return []
    out: List[str] = []
    for path in table_root.iterdir():
        if path.is_dir():
            asset = parse_asset_name(path.name)
            if asset:
                out.append(asset)
    return sorted(set(out))


def asset_months(table_root: Path, asset: str) -> set[MonthKey]:
    base = table_root / f"asset={asset}"
    out: set[MonthKey] = set()
    if not base.exists():
        return out
    for ydir in base.glob("year=*"):
        if not ydir.is_dir():
            continue
        try:
            year = int(ydir.name.split("=", 1)[1])
        except Exception:
            continue
        for mdir in ydir.glob("month=*"):
            if not mdir.is_dir():
                continue
            try:
                month = int(mdir.name.split("=", 1)[1])
            except Exception:
                continue
            out.add(MonthKey(year, month))
    return out


def common_recent_window(
    *,
    ohlc_root: Path,
    scalar_root: Path,
    min_assets: int,
    window_months: int,
    search_back_months: int,
    clamp_start: MonthKey,
) -> Tuple[MonthKey, List[str]]:
    all_assets = sorted(set(list_assets(ohlc_root)).intersection(list_assets(scalar_root)))
    month_map: Dict[str, set[MonthKey]] = {}
    for asset in all_assets:
        months = asset_months(ohlc_root, asset).intersection(asset_months(scalar_root, asset))
        months = {m for m in months if months_between(clamp_start, add_months(m.year, m.month, 1)) > 0}
        if months:
            month_map[asset] = months
    now = datetime.now(timezone.utc)
    start_end_month = add_months(now.year, now.month, -1)
    for back in range(max(1, int(search_back_months))):
        end_month = add_months(start_end_month.year, start_end_month.month, -back)
        required = {
            m for m in month_seq(end_month, int(window_months))
            if months_between(clamp_start, add_months(m.year, m.month, 1)) > 0
        }
        eligible = [asset for asset, months in month_map.items() if required.issubset(months)]
        if len(eligible) >= int(min_assets):
            return end_month, sorted(eligible)
    raise RuntimeError(
        f"Could not find a common {int(window_months)}-month window with at least {int(min_assets)} assets."
    )


def asset_symbol(asset: str) -> str:
    normalized = re.sub(r"[^A-Z0-9]", "", str(asset).upper())
    for suffix in COMMON_QUOTE_SUFFIXES:
        if normalized.endswith(suffix) and len(normalized) > len(suffix):
            normalized = normalized[: -len(suffix)]
            break
    return SYMBOL_CANONICAL_MAP.get(normalized, normalized)


def candidate_rank(asset: str) -> Tuple[int, str]:
    upper = str(asset).upper()
    for idx, suffix in enumerate(COMMON_QUOTE_SUFFIXES):
        if upper.endswith(suffix):
            return (idx, upper)
    return (len(COMMON_QUOTE_SUFFIXES) + 1, upper)


def select_representative_assets(
    eligible_assets: Sequence[str],
    *,
    seed: int,
    asset_count: int,
    required_symbols: Sequence[str] = FIXED_NUMERIC_FAMILY_COHORT_SYMBOLS,
    preferred_optional_symbols: Sequence[str] = (),
) -> Tuple[List[str], Dict[str, str]]:
    assets = sorted({str(asset) for asset in eligible_assets})
    requested_symbols = [str(symbol).upper() for symbol in required_symbols]
    requested_count = len(requested_symbols)
    if int(asset_count) not in (0, requested_count):
        raise RuntimeError(
            f"Numeric family cohort is fixed at {requested_count} assets; received asset_count={int(asset_count)}."
        )
    by_symbol: Dict[str, List[str]] = {}
    for asset in assets:
        by_symbol.setdefault(asset_symbol(asset), []).append(str(asset))
    selected: List[str] = []
    alias_map: Dict[str, str] = {}

    def _pick_symbol(symbol: str) -> Optional[str]:
        candidates = [asset for asset in by_symbol.get(str(symbol).upper(), []) if asset not in selected]
        if not candidates:
            return None
        candidates = sorted(candidates, key=candidate_rank)
        return str(candidates[0])

    missing_symbols: List[str] = []
    for symbol in requested_symbols:
        chosen = _pick_symbol(symbol)
        if chosen is None:
            missing_symbols.append(symbol)
            continue
        selected.append(chosen)
        alias_map[symbol] = str(chosen)

    if missing_symbols:
        raise RuntimeError(
            "Fixed numeric family cohort could not be resolved from the eligible asset set. "
            f"Missing symbols: {missing_symbols}"
        )

    selected = sorted(selected)
    return selected, alias_map
