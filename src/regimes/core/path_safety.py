from __future__ import annotations

from pathlib import Path
from typing import Mapping

from src.regimes.core.paths import (
    FOUNDATION_REPORT_ROOT_PARTS,
    default_foundation_report_root,
    is_production_adjacent_path,
    normalized_path_parts,
    require_foundation_report_root,
    resolve_project_path,
    resolve_project_root,
)


PRODUCTION_LIKE_WRITE_PARTS: frozenset[str] = frozenset(
    {
        "live",
        "prod",
        "production",
        "production_outputs",
        "prod_outputs",
        "regime-labels",
        "regime_labels",
    }
)


def validate_report_root(
    report_root: str | Path,
    *,
    project_root: str | Path | None = None,
    allow_foundation_descendant: bool = True,
) -> Path:
    return require_foundation_report_root(
        report_root,
        project_root=project_root,
        allow_foundation_descendant=allow_foundation_descendant,
        error_prefix="Regime report root",
    )


def validate_non_production_write_root(
    write_root: str | Path,
    *,
    project_root: str | Path | None = None,
) -> Path:
    root = resolve_project_path(write_root, project_root=project_root)
    if is_production_adjacent_path(root, project_root=project_root):
        raise ValueError("Regime write root is production-adjacent and is not allowed")
    parts = normalized_path_parts(root)
    if any(part in PRODUCTION_LIKE_WRITE_PARTS for part in parts):
        raise ValueError("Regime write root is production-like and is not allowed")
    return root


def resolve_regime_report_root(
    *parts: str,
    report_root: str | Path | None = None,
    env: Mapping[str, str] | None = None,
    project_root: str | Path | None = None,
) -> Path:
    root = (
        resolve_project_path(report_root, project_root=project_root).joinpath(*(str(part) for part in parts))
        if report_root is not None
        else default_foundation_report_root(*parts, env=env, project_root=project_root)
    )
    return validate_report_root(root, project_root=project_root, allow_foundation_descendant=True)


def resolve_regime_data_root(
    data_root: str | Path | None = None,
    *,
    project_root: str | Path | None = None,
) -> Path:
    root = resolve_project_path(data_root, project_root=project_root) if data_root is not None else resolve_project_root(project_root) / "data" / "regimes"
    if is_production_adjacent_path(root, project_root=project_root):
        raise ValueError("Regime data root is production-adjacent and is not allowed")
    return root


__all__ = [
    "FOUNDATION_REPORT_ROOT_PARTS",
    "PRODUCTION_LIKE_WRITE_PARTS",
    "is_production_adjacent_path",
    "resolve_regime_data_root",
    "resolve_regime_report_root",
    "validate_non_production_write_root",
    "validate_report_root",
]
