from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping, Sequence

from src.forecasting.common.path_config import project_root as configured_project_root


REGIME_FOUNDATION_REPORT_ROOT_ENV = "PIPELINE_REGIME_FOUNDATION_REPORT_ROOT"
FOUNDATION_REPORT_ROOT_PARTS: tuple[str, ...] = ("reports", "regimes", "foundation")
PRODUCTION_ADJACENT_ROOT_NAMES: frozenset[str] = frozenset(
    {"parquet", "regime_definitions", "model_states", "state"}
)


def resolve_project_root(project_root: str | Path | None = None) -> Path:
    root = Path(project_root).expanduser() if project_root is not None else configured_project_root()
    return root.resolve()


def resolve_project_path(path: str | Path, *, project_root: str | Path | None = None) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = resolve_project_root(project_root) / candidate
    return candidate.resolve()


def is_relative_to(path: str | Path, root: str | Path) -> bool:
    try:
        Path(path).resolve().relative_to(Path(root).resolve())
        return True
    except ValueError:
        return False


def normalized_path_parts(path: str | Path) -> tuple[str, ...]:
    return tuple(part.lower() for part in Path(path).parts)


def has_path_parts(path: str | Path, expected_parts: Sequence[str]) -> bool:
    parts = normalized_path_parts(path)
    expected = tuple(str(part).lower() for part in expected_parts)
    width = len(expected)
    return any(parts[idx : idx + width] == expected for idx in range(max(len(parts) - width + 1, 0)))


def path_ends_with_parts(path: str | Path, expected_parts: Sequence[str]) -> bool:
    parts = normalized_path_parts(path)
    expected = tuple(str(part).lower() for part in expected_parts)
    return len(parts) >= len(expected) and parts[-len(expected) :] == expected


def is_production_adjacent_path(path: str | Path, *, project_root: str | Path | None = None) -> bool:
    root = resolve_project_path(path, project_root=project_root)
    if any(part.lower() in PRODUCTION_ADJACENT_ROOT_NAMES for part in root.parts):
        return True
    project = resolve_project_root(project_root)
    for name in PRODUCTION_ADJACENT_ROOT_NAMES:
        if is_relative_to(root, project / name):
            return True
    return False


def default_foundation_report_root(
    *parts: str,
    env: Mapping[str, str] | None = None,
    project_root: str | Path | None = None,
) -> Path:
    source_env = env if env is not None else os.environ
    raw_root = str(source_env.get(REGIME_FOUNDATION_REPORT_ROOT_ENV, "") or "").strip()
    base = (
        resolve_project_path(raw_root, project_root=project_root)
        if raw_root
        else resolve_project_root(project_root).joinpath(*FOUNDATION_REPORT_ROOT_PARTS)
    )
    return base.joinpath(*(str(part) for part in parts)).resolve()


def require_foundation_report_root(
    report_root: str | Path,
    *,
    project_root: str | Path | None = None,
    required_suffix: Sequence[str] = FOUNDATION_REPORT_ROOT_PARTS,
    allow_foundation_descendant: bool = False,
    error_prefix: str = "Regime foundation report root",
) -> Path:
    root = resolve_project_path(report_root, project_root=project_root)
    if is_production_adjacent_path(root, project_root=project_root):
        raise ValueError(f"{error_prefix} is production-adjacent and is not allowed")
    if allow_foundation_descendant:
        if not has_path_parts(root, FOUNDATION_REPORT_ROOT_PARTS):
            raise ValueError(f"{error_prefix} must be under reports/regimes/foundation")
    elif not path_ends_with_parts(root, required_suffix):
        suffix_text = "/".join(str(part) for part in required_suffix)
        raise ValueError(f"{error_prefix} must end with {suffix_text}")
    return root


__all__ = [
    "FOUNDATION_REPORT_ROOT_PARTS",
    "PRODUCTION_ADJACENT_ROOT_NAMES",
    "REGIME_FOUNDATION_REPORT_ROOT_ENV",
    "default_foundation_report_root",
    "has_path_parts",
    "is_production_adjacent_path",
    "is_relative_to",
    "normalized_path_parts",
    "path_ends_with_parts",
    "require_foundation_report_root",
    "resolve_project_path",
    "resolve_project_root",
]
