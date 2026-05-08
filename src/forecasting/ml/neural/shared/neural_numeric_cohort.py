from __future__ import annotations

from pathlib import Path
from typing import List, Sequence

from src.forecasting.ml.shared.numeric_cohort_common import (
    CLAMP_START_MONTH,
    CLAMP_START_YEAR,
    DEFAULT_COHORT_WINDOW_MONTHS,
    DEFAULT_SEARCH_BACK_MONTHS,
    MonthKey,
    common_recent_window,
    select_representative_assets,
)

FIXED_NEURAL_NUMERIC_COHORT = (
    "BTC",
    "SOL",
    "ETH",
    "ADA",
    "XRP",
    "LTC",
    "AVAX",
    "XMR",
)


def resolve_neural_cohort_assets(
    *,
    parquet_root: Path,
    intervals: Sequence[int],
    asset_count: int,
    explicit_assets: Sequence[str] = (),
    seed: int = 17,
) -> List[str]:
    requested_assets = [str(asset).strip() for asset in explicit_assets if str(asset).strip()]
    if requested_assets:
        return requested_assets
    interval_values = sorted({int(interval) for interval in intervals if int(interval) > 0})
    if not interval_values:
        raise RuntimeError("No intervals resolved for Neural cohort selection.")
    clamp_start = MonthKey(CLAMP_START_YEAR, CLAMP_START_MONTH)
    eligible_sets: List[set[str]] = []
    for interval_minutes in interval_values:
        _end_month, eligible_assets = common_recent_window(
            ohlc_root=parquet_root / f"ohlcvt_{int(interval_minutes)}",
            scalar_root=parquet_root / f"scalar_features_{int(interval_minutes)}",
            min_assets=1,
            window_months=DEFAULT_COHORT_WINDOW_MONTHS,
            search_back_months=DEFAULT_SEARCH_BACK_MONTHS,
            clamp_start=clamp_start,
        )
        eligible_sets.append({str(asset) for asset in eligible_assets})
    common_assets = sorted(set.intersection(*eligible_sets)) if eligible_sets else []
    if not common_assets:
        raise RuntimeError("No shared eligible assets were found across the requested Neural intervals.")
    selected_assets, _alias_map = select_representative_assets(
        common_assets,
        seed=int(seed),
        asset_count=int(asset_count),
        required_symbols=FIXED_NEURAL_NUMERIC_COHORT,
    )
    return [str(asset) for asset in selected_assets]
