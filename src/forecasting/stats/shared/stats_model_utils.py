from __future__ import annotations

from typing import List, Sequence

from src.forecasting.common.forecast_family_core import horizon_bars as _horizon_bars
from src.forecasting.common.forecast_family_core import parse_int_csv as _parse_int_csv
from src.forecasting.common.forecast_family_core import parse_str_csv as _parse_str_csv


def parse_int_csv(raw: str, default_vals: Sequence[int]) -> List[int]:
    return _parse_int_csv(raw, default_vals)


def parse_str_csv(raw: str, default_vals: Sequence[str]) -> List[str]:
    return _parse_str_csv(raw, default_vals)


def horizon_bars(horizon_minutes: int, interval_minutes: int) -> int:
    return _horizon_bars(horizon_minutes, interval_minutes)
