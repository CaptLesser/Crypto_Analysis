from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd


CORRELATION_FEATURE_NAMES: tuple[str, ...] = (
    "market_corr_median_pairwise",
    "market_corr_q25_pairwise",
    "market_corr_q75_pairwise",
    "market_corr_high_share",
    "market_corr_first_pc_concentration",
)
COVARIANCE_FEATURE_NAMES: tuple[str, ...] = (
    "market_cov_ledoit_wolf_trace",
    "market_cov_ledoit_wolf_mean_variance",
    "market_cov_ledoit_wolf_mean_covariance",
    "market_cov_oas_trace",
    "market_cov_oas_mean_variance",
    "market_cov_oas_mean_covariance",
)


try:  # pragma: no cover - availability is environment-dependent.
    from sklearn.covariance import LedoitWolf, OAS

    SKLEARN_COVARIANCE_AVAILABLE = True
    SKLEARN_COVARIANCE_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - availability is environment-dependent.
    LedoitWolf = None  # type: ignore[assignment]
    OAS = None  # type: ignore[assignment]
    SKLEARN_COVARIANCE_AVAILABLE = False
    SKLEARN_COVARIANCE_IMPORT_ERROR = str(exc)


@dataclass(frozen=True)
class MarketCovarianceFeatureConfig:
    rolling_window: int = 20
    min_periods: int = 5
    high_correlation_threshold: float = 0.7

    def __post_init__(self) -> None:
        if int(self.rolling_window) < 2:
            raise ValueError("Market covariance rolling_window must be at least 2")
        if int(self.min_periods) < 2:
            raise ValueError("Market covariance min_periods must be at least 2")
        if not 0.0 <= float(self.high_correlation_threshold) <= 1.0:
            raise ValueError("Market covariance high_correlation_threshold must be within [0, 1]")


def compute_core_correlation_covariance_features(
    core_returns: pd.DataFrame,
    *,
    config: MarketCovarianceFeatureConfig | None = None,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, str]]:
    cfg = config or MarketCovarianceFeatureConfig()
    returns = core_returns.apply(pd.to_numeric, errors="coerce") if not core_returns.empty else pd.DataFrame()
    out = pd.DataFrame(index=returns.index)
    for name in (*CORRELATION_FEATURE_NAMES, *COVARIANCE_FEATURE_NAMES):
        out[name] = np.nan

    diagnostics: dict[str, Any] = {
        "core_asset_count": int(returns.shape[1]),
        "timestamp_count": int(returns.shape[0]),
        "rolling_window": int(cfg.rolling_window),
        "min_periods": int(cfg.min_periods),
        "high_correlation_threshold": float(cfg.high_correlation_threshold),
        "sklearn_covariance_available": bool(SKLEARN_COVARIANCE_AVAILABLE),
        "sklearn_covariance_import_error": SKLEARN_COVARIANCE_IMPORT_ERROR,
    }
    unavailable: dict[str, str] = {}
    if returns.empty or returns.shape[1] < 2:
        diagnostics.update(
            {
                "correlation": {"status": "unavailable", "computed_rows": 0, "reason": "fewer_than_two_core_assets"},
                "ledoit_wolf": {"status": "unavailable", "computed_rows": 0, "reason": "fewer_than_two_core_assets"},
                "oas": {"status": "unavailable", "computed_rows": 0, "reason": "fewer_than_two_core_assets"},
            }
        )
        for feature in out.columns:
            unavailable[str(feature)] = "fewer_than_two_core_assets"
        return out, diagnostics, unavailable

    corr_rows = 0
    lw_rows = 0
    oas_rows = 0
    for row_idx in range(len(returns)):
        start = max(0, row_idx - int(cfg.rolling_window) + 1)
        window = returns.iloc[start : row_idx + 1]
        enough_assets = tuple(column for column in window.columns if int(window[column].notna().sum()) >= int(cfg.min_periods))
        if len(enough_assets) >= 2:
            corr = window[list(enough_assets)].corr(min_periods=int(cfg.min_periods))
            pairwise = _upper_triangle_values(corr.to_numpy(dtype=float))
            pairwise = pairwise[np.isfinite(pairwise)]
            if pairwise.size:
                out.iat[row_idx, out.columns.get_loc("market_corr_median_pairwise")] = float(np.nanmedian(pairwise))
                out.iat[row_idx, out.columns.get_loc("market_corr_q25_pairwise")] = float(np.nanquantile(pairwise, 0.25))
                out.iat[row_idx, out.columns.get_loc("market_corr_q75_pairwise")] = float(np.nanquantile(pairwise, 0.75))
                out.iat[row_idx, out.columns.get_loc("market_corr_high_share")] = float((np.abs(pairwise) >= float(cfg.high_correlation_threshold)).mean())
                out.iat[row_idx, out.columns.get_loc("market_corr_first_pc_concentration")] = _first_pc_concentration(corr)
                corr_rows += 1

        clean = window.dropna(axis=0, how="any")
        if clean.shape[0] >= int(cfg.min_periods) and clean.shape[1] >= 2:
            values = clean.to_numpy(dtype=float)
            if SKLEARN_COVARIANCE_AVAILABLE and LedoitWolf is not None:
                try:
                    lw_cov = LedoitWolf().fit(values).covariance_
                    _set_covariance_summary(out, row_idx, "market_cov_ledoit_wolf", lw_cov)
                    lw_rows += 1
                except Exception:
                    pass
            if SKLEARN_COVARIANCE_AVAILABLE and OAS is not None:
                try:
                    oas_cov = OAS().fit(values).covariance_
                    _set_covariance_summary(out, row_idx, "market_cov_oas", oas_cov)
                    oas_rows += 1
                except Exception:
                    pass

    diagnostics["correlation"] = _method_status(corr_rows, len(returns))
    if SKLEARN_COVARIANCE_AVAILABLE:
        diagnostics["ledoit_wolf"] = _method_status(lw_rows, len(returns))
        diagnostics["oas"] = _method_status(oas_rows, len(returns))
    else:
        diagnostics["ledoit_wolf"] = {"status": "unavailable", "computed_rows": 0, "reason": "sklearn_covariance_unavailable"}
        diagnostics["oas"] = {"status": "unavailable", "computed_rows": 0, "reason": "sklearn_covariance_unavailable"}

    for feature in CORRELATION_FEATURE_NAMES:
        if not bool(out[feature].notna().any()):
            unavailable[feature] = "insufficient_core_return_window_for_correlation"
    for feature in COVARIANCE_FEATURE_NAMES:
        if not bool(out[feature].notna().any()):
            unavailable[feature] = (
                "sklearn_covariance_unavailable"
                if not SKLEARN_COVARIANCE_AVAILABLE
                else "insufficient_core_return_window_for_shrinkage_covariance"
            )
    return out, diagnostics, unavailable


def _upper_triangle_values(matrix: np.ndarray) -> np.ndarray:
    if matrix.ndim != 2 or matrix.shape[0] < 2:
        return np.asarray([], dtype=float)
    idx = np.triu_indices(matrix.shape[0], k=1)
    return matrix[idx]


def _first_pc_concentration(corr: pd.DataFrame) -> float:
    matrix = corr.to_numpy(dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] < 2:
        return np.nan
    matrix = np.where(np.isfinite(matrix), matrix, 0.0)
    np.fill_diagonal(matrix, 1.0)
    try:
        eigvals = np.linalg.eigvalsh(matrix)
    except np.linalg.LinAlgError:
        return np.nan
    eigvals = np.clip(eigvals, 0.0, None)
    total = float(eigvals.sum())
    return float(eigvals.max() / total) if total > 0.0 else np.nan


def _set_covariance_summary(out: pd.DataFrame, row_idx: int, prefix: str, covariance: np.ndarray) -> None:
    diag = np.diag(covariance)
    off_diag = _upper_triangle_values(covariance)
    out.iat[row_idx, out.columns.get_loc(f"{prefix}_trace")] = float(np.trace(covariance))
    out.iat[row_idx, out.columns.get_loc(f"{prefix}_mean_variance")] = float(np.mean(diag)) if diag.size else np.nan
    out.iat[row_idx, out.columns.get_loc(f"{prefix}_mean_covariance")] = float(np.mean(off_diag)) if off_diag.size else np.nan


def _method_status(computed_rows: int, total_rows: int) -> Mapping[str, Any]:
    if computed_rows <= 0:
        return {"status": "unavailable", "computed_rows": 0, "reason": "insufficient_window"}
    if computed_rows < total_rows:
        return {"status": "partial", "computed_rows": int(computed_rows)}
    return {"status": "computed", "computed_rows": int(computed_rows)}


__all__ = [
    "CORRELATION_FEATURE_NAMES",
    "COVARIANCE_FEATURE_NAMES",
    "MarketCovarianceFeatureConfig",
    "SKLEARN_COVARIANCE_AVAILABLE",
    "compute_core_correlation_covariance_features",
]
