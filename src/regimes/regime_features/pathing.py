from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from src.forecasting.common.path_config import resolve_path, selected_profile
from src.regimes.asset_state.dataset_builder import resolve_asset_state_scalar_feature_root
from src.regimes.core.path_safety import validate_report_root
from src.regimes.core.paths import resolve_project_path


REGIME_FEATURES_OUTPUT_ENV = "PIPELINE_REGIME_FEATURES_OUTPUT_ROOT"
REGIME_FEATURES_SOURCE_ENV = "PIPELINE_REGIME_FEATURES_SOURCE_ROOT"
REGIME_FEATURES_REPORT_SUBDIR = "regime_features"


@dataclass(frozen=True)
class RegimeFeatureSourceRoots:
    ohlcvt_root: Path | None
    scalar_feature_root: Path | None
    ohlcvt_source: str
    scalar_feature_source: str
    profile: str
    candidates: Mapping[str, Any] = field(default_factory=dict)

    @property
    def ohlcvt_found(self) -> bool:
        return self.ohlcvt_root is not None and self.ohlcvt_root.exists()

    @property
    def scalar_features_found(self) -> bool:
        return self.scalar_feature_root is not None and self.scalar_feature_root.exists()

    def as_dict(self) -> dict[str, Any]:
        return {
            "ohlcvt_root": str(self.ohlcvt_root) if self.ohlcvt_root is not None else None,
            "scalar_feature_root": str(self.scalar_feature_root) if self.scalar_feature_root is not None else None,
            "ohlcvt_source": self.ohlcvt_source,
            "scalar_feature_source": self.scalar_feature_source,
            "profile": self.profile,
            "ohlcvt_found": bool(self.ohlcvt_found),
            "scalar_features_found": bool(self.scalar_features_found),
            "candidates": dict(self.candidates),
        }


def resolve_regime_feature_source_roots(
    *,
    source_ohlcvt_root: str | Path | None = None,
    source_feature_root: str | Path | None = None,
    env: Mapping[str, str] | None = None,
    profile: str | None = None,
    project_root: str | Path | None = None,
) -> RegimeFeatureSourceRoots:
    source_env = env if env is not None else None
    resolved_profile = str(profile or selected_profile(env=source_env)).strip() or "production"
    ohlcvt_root, ohlcvt_source, ohlcvt_candidates = _resolve_ohlcvt_root(
        source_ohlcvt_root,
        env=env,
        profile=resolved_profile,
        project_root=project_root,
    )
    scalar_resolution = resolve_asset_state_scalar_feature_root(
        source_feature_root,
        profile=resolved_profile,
        env=env,
        include_default_parquet=True,
    )
    scalar_root = scalar_resolution.root if scalar_resolution.found else None
    scalar_source = scalar_resolution.source if scalar_resolution.root is not None else "unconfigured"
    return RegimeFeatureSourceRoots(
        ohlcvt_root=ohlcvt_root,
        scalar_feature_root=scalar_root,
        ohlcvt_source=ohlcvt_source,
        scalar_feature_source=scalar_source,
        profile=resolved_profile,
        candidates={
            "ohlcvt": ohlcvt_candidates,
            "scalar": scalar_resolution.as_dict(),
        },
    )


def validate_regime_feature_write_root(
    output_root: str | Path,
    *,
    project_root: str | Path | None = None,
    production_enabled: bool = False,
) -> Path:
    if production_enabled is not False:
        raise ValueError("Regime Feature production writes are disabled in foundation scaffolding")
    return validate_report_root(
        output_root,
        project_root=project_root,
        allow_foundation_descendant=True,
    )


def default_regime_feature_report_root(
    *,
    report_root: str | Path | None = None,
    env: Mapping[str, str] | None = None,
    project_root: str | Path | None = None,
) -> Path:
    if report_root is not None:
        return validate_regime_feature_write_root(report_root, project_root=project_root)
    source_env = env if env is not None else os.environ
    raw = str(source_env.get(REGIME_FEATURES_OUTPUT_ENV, "") or "").strip()
    if raw:
        return validate_regime_feature_write_root(raw, project_root=project_root)
    project = resolve_project_path(".", project_root=project_root).resolve()
    return validate_regime_feature_write_root(
        project / "reports" / "regimes" / "foundation" / REGIME_FEATURES_REPORT_SUBDIR,
        project_root=project_root,
    )


def market_feature_table_dir(interval: int) -> str:
    return f"regime_features_market_{int(interval)}"


def pairwise_feature_table_dir(interval: int) -> str:
    return f"regime_features_pairwise_{int(interval)}"


def cross_asset_feature_table_dir(interval: int) -> str:
    return f"regime_features_cross_asset_{int(interval)}"


def market_feature_month_dir(root: str | Path, *, interval: int, band: str, year: int, month: int) -> Path:
    return Path(root) / market_feature_table_dir(interval) / f"band={str(band)}" / f"year={int(year):04d}" / f"month={int(month):02d}"


def pairwise_feature_month_dir(
    root: str | Path,
    *,
    interval: int,
    band: str,
    relationship_type: str,
    window: int,
    year: int,
    month: int,
) -> Path:
    return (
        Path(root)
        / pairwise_feature_table_dir(interval)
        / f"relationship_type={str(relationship_type)}"
        / f"band={str(band)}"
        / f"window={int(window)}"
        / f"year={int(year):04d}"
        / f"month={int(month):02d}"
    )


def cross_asset_feature_month_dir(root: str | Path, *, interval: int, band: str, asset: str, year: int, month: int) -> Path:
    return (
        Path(root)
        / cross_asset_feature_table_dir(interval)
        / f"band={str(band)}"
        / f"asset={str(asset)}"
        / f"year={int(year):04d}"
        / f"month={int(month):02d}"
    )


def snapshot_dir(root: str | Path, *, policy_id: str, refit_key: str, interval: int, band: str) -> Path:
    return (
        Path(root)
        / "regime_feature_snapshots"
        / f"policy_id={str(policy_id)}"
        / f"refit_key={str(refit_key)}"
        / f"interval={int(interval)}"
        / f"band={str(band)}"
    )


def _resolve_ohlcvt_root(
    explicit_root: str | Path | None,
    *,
    env: Mapping[str, str] | None,
    profile: str,
    project_root: str | Path | None,
) -> tuple[Path | None, str, list[dict[str, Any]]]:
    source_env = env if env is not None else os.environ
    candidates: list[tuple[Path | None, str]] = []
    if explicit_root is not None and str(explicit_root).strip():
        candidates.append((resolve_project_path(explicit_root, project_root=project_root), "explicit_argument"))
    configured = resolve_path("source_ohlcvt_root", profile=profile, env=env, required=False)
    if configured is not None:
        candidates.append((configured, "path_config.source_ohlcvt_root"))
    raw_feature_source = str(source_env.get(REGIME_FEATURES_SOURCE_ENV, "") or "").strip()
    if raw_feature_source:
        candidates.append((Path(raw_feature_source).expanduser(), f"env.{REGIME_FEATURES_SOURCE_ENV}"))
    raw_pipeline_root = str(source_env.get("PIPELINE_ROOT", "") or "").strip()
    if raw_pipeline_root:
        candidates.append((Path(raw_pipeline_root).expanduser() / "parquet", "env.PIPELINE_ROOT/parquet"))
    raw_pipeline_parquet = str(source_env.get("PIPELINE_PARQUET", "") or "").strip()
    if raw_pipeline_parquet:
        candidates.append((Path(raw_pipeline_parquet).expanduser(), "env.PIPELINE_PARQUET"))
    candidates.append((resolve_project_path("parquet", project_root=project_root), "project_relative.parquet"))

    inspected: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for root, source in candidates:
        if root is None:
            continue
        resolved = root.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        intervals = _intervals_under(resolved, prefix="ohlcvt_") if resolved.exists() else ()
        inspected.append(
            {
                "source": source,
                "root": str(resolved),
                "exists": bool(resolved.exists()),
                "ohlcvt_intervals": list(intervals),
            }
        )
        if intervals:
            return resolved, source, inspected
    return None, "unconfigured_or_missing", inspected


def _intervals_under(root: Path, *, prefix: str) -> tuple[int, ...]:
    out: set[int] = set()
    if not root.exists() or not root.is_dir():
        return ()
    for child in root.iterdir():
        if not child.is_dir() or not child.name.startswith(prefix):
            continue
        value = child.name.removeprefix(prefix)
        if value.isdigit():
            out.add(int(value))
    return tuple(sorted(out))


__all__ = [
    "REGIME_FEATURES_OUTPUT_ENV",
    "REGIME_FEATURES_REPORT_SUBDIR",
    "REGIME_FEATURES_SOURCE_ENV",
    "RegimeFeatureSourceRoots",
    "cross_asset_feature_month_dir",
    "cross_asset_feature_table_dir",
    "default_regime_feature_report_root",
    "market_feature_month_dir",
    "market_feature_table_dir",
    "pairwise_feature_month_dir",
    "pairwise_feature_table_dir",
    "resolve_regime_feature_source_roots",
    "snapshot_dir",
    "validate_regime_feature_write_root",
]
