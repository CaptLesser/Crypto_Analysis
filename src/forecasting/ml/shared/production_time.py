from __future__ import annotations

import os
from datetime import datetime, timezone

DEFAULT_PRODUCTION_START_YEAR = 2025
DEFAULT_PRODUCTION_START_MONTH = 1
DEFAULT_PRODUCTION_START_DAY = 1
PRODUCTION_START_TS_ENV = "ML_NUMERIC_PRODUCTION_START_TS"


def default_production_start_ts() -> int:
    return int(
        datetime(
            DEFAULT_PRODUCTION_START_YEAR,
            DEFAULT_PRODUCTION_START_MONTH,
            DEFAULT_PRODUCTION_START_DAY,
            0,
            0,
            0,
            tzinfo=timezone.utc,
        ).timestamp()
    )


def production_start_ts() -> int:
    raw = str(os.getenv(PRODUCTION_START_TS_ENV, "")).strip()
    if raw:
        try:
            return int(raw)
        except Exception:
            pass
    return default_production_start_ts()
