from __future__ import annotations

import pandas as pd


def synthetic_asset_state_fixture(*, periods: int = 12, start: str = "2026-01-01T00:00:00Z") -> pd.DataFrame:
    if int(periods) < 6:
        raise ValueError("Regime synthetic fixture requires at least six periods")
    timestamps = pd.date_range(start, periods=int(periods), freq="30min")
    rows: list[dict[str, object]] = []
    for idx, timestamp in enumerate(timestamps):
        state = idx % 2
        offset = idx // 2
        if state == 0:
            row = {
                "log_return": -0.030 - 0.001 * offset,
                "macd_hist_12_26_9": -1.20 - 0.03 * offset,
                "rsi_14": 28.0 + 0.2 * offset,
                "adx_14": 31.0 + 0.1 * offset,
                "atr_14": 0.080 + 0.002 * offset,
                "ret_std_20": 0.060 + 0.001 * offset,
                "cv_20": 1.80 + 0.01 * offset,
                "vol_osc_pct_14_28": 0.24 + 0.005 * offset,
                "trade_intensity": 80.0 + offset,
                "avg_trade_size": 1.10 + 0.01 * offset,
                "vroc_14": -0.14 - 0.002 * offset,
                "prr": 0.42 + 0.002 * offset,
                "future_log_return": -0.022 - 0.001 * offset,
                "future_realized_volatility": 0.070 + 0.002 * offset,
                "future_max_drawdown": -0.055 - 0.001 * offset,
            }
        else:
            row = {
                "log_return": 0.034 + 0.001 * offset,
                "macd_hist_12_26_9": 1.15 + 0.03 * offset,
                "rsi_14": 69.0 - 0.2 * offset,
                "adx_14": 34.0 + 0.1 * offset,
                "atr_14": 0.030 + 0.001 * offset,
                "ret_std_20": 0.024 + 0.001 * offset,
                "cv_20": 0.88 + 0.01 * offset,
                "vol_osc_pct_14_28": -0.08 + 0.004 * offset,
                "trade_intensity": 140.0 + offset,
                "avg_trade_size": 1.70 + 0.01 * offset,
                "vroc_14": 0.18 + 0.002 * offset,
                "prr": 0.65 + 0.002 * offset,
                "future_log_return": 0.030 + 0.001 * offset,
                "future_realized_volatility": 0.030 + 0.001 * offset,
                "future_max_drawdown": -0.014 - 0.001 * offset,
            }
        rows.append(
            {
                "timestamp": timestamp,
                "asset": "XBTUSD",
                "synthetic_state": int(state),
                **row,
            }
        )
    return pd.DataFrame(rows)


__all__ = ["synthetic_asset_state_fixture"]
