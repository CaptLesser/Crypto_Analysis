from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from src.forecasting.common.path_config import PathConfigError, resolve_path, selected_profile
from src.regimes.core.path_safety import validate_report_root
from src.regimes.core.paths import default_foundation_report_root, is_relative_to, resolve_project_path, resolve_project_root


SOURCE_KIND_OHLCVT = "ohlcvt"
SOURCE_KIND_SCALAR_FEATURES = "scalar_features"
SOURCE_KIND_REGIME_FEATURES = "regime_features"
SOURCE_KIND_MARKET_STATE_UNIVERSE = "market_state_universe_manifest"
SOURCE_KIND_RELATIONSHIP_DISCOVERY = "relationship_discovery_artifacts"
SOURCE_KIND_UNIVERSE_ELIGIBILITY = "universe_eligibility_snapshot"
SOURCE_KIND_REPORT_OUTPUT_ROOT = "report_sandbox_output_root"

REGIME_PRODUCTION_CONFIGURED_ROOTS_ONLY = "configured_roots_only"
REGIME_PRODUCTION_DRY_TEST_OVERRIDE_SOURCE = "explicit_dry_test_override"
REGIME_PRODUCTION_BRANCH_POLICY_SIDECAR_FILE = "branch_policy_manifest_sidecar_file_path"

READ_SOURCE_KINDS: frozenset[str] = frozenset(
    {
        SOURCE_KIND_OHLCVT,
        SOURCE_KIND_SCALAR_FEATURES,
        SOURCE_KIND_REGIME_FEATURES,
        SOURCE_KIND_MARKET_STATE_UNIVERSE,
        SOURCE_KIND_RELATIONSHIP_DISCOVERY,
        SOURCE_KIND_UNIVERSE_ELIGIBILITY,
    }
)
WRITE_SOURCE_KINDS: frozenset[str] = frozenset({SOURCE_KIND_REPORT_OUTPUT_ROOT})


ROOT_FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    SOURCE_KIND_OHLCVT: (SOURCE_KIND_OHLCVT, "source_ohlcvt_root", "ohlcvt_root", "parquet_root"),
    SOURCE_KIND_SCALAR_FEATURES: (SOURCE_KIND_SCALAR_FEATURES, "source_feature_root", "scalar_feature_root", "feature_root"),
    SOURCE_KIND_REGIME_FEATURES: (SOURCE_KIND_REGIME_FEATURES, "source_regime_root", "regime_feature_root", "regime_features_root"),
    SOURCE_KIND_MARKET_STATE_UNIVERSE: (
        SOURCE_KIND_MARKET_STATE_UNIVERSE,
        "market_state_universe_root",
        "market_state_universe_manifest_root",
        "manifest_root",
    ),
    SOURCE_KIND_RELATIONSHIP_DISCOVERY: (SOURCE_KIND_RELATIONSHIP_DISCOVERY, "relationship_discovery_root", "relationship_discovery_artifact_root"),
    SOURCE_KIND_UNIVERSE_ELIGIBILITY: (SOURCE_KIND_UNIVERSE_ELIGIBILITY, "universe_eligibility_root", "universe_eligibility_snapshot_root"),
    SOURCE_KIND_REPORT_OUTPUT_ROOT: (SOURCE_KIND_REPORT_OUTPUT_ROOT, "report_root", "output_root", "sandbox_output_root"),
}

ENV_ALIASES: dict[str, tuple[str, ...]] = {
    SOURCE_KIND_OHLCVT: ("PIPELINE_SOURCE_OHLCVT_ROOT", "PIPELINE_SOURCE_PARQUET_ROOT"),
    SOURCE_KIND_SCALAR_FEATURES: ("PIPELINE_SOURCE_FEATURES_ROOT", "PIPELINE_PARQUET_FEATURES_ROOT"),
    SOURCE_KIND_REGIME_FEATURES: ("PIPELINE_SOURCE_REGIME_ROOT", "PIPELINE_PARQUET_REGIME_ROOT"),
    SOURCE_KIND_MARKET_STATE_UNIVERSE: ("PIPELINE_MARKET_STATE_UNIVERSE_ROOT",),
    SOURCE_KIND_RELATIONSHIP_DISCOVERY: ("PIPELINE_RELATIONSHIP_DISCOVERY_ROOT",),
    SOURCE_KIND_UNIVERSE_ELIGIBILITY: ("PIPELINE_UNIVERSE_ELIGIBILITY_ROOT",),
    SOURCE_KIND_REPORT_OUTPUT_ROOT: ("PIPELINE_REGIME_FOUNDATION_REPORT_ROOT",),
}


@dataclass(frozen=True)
class RegimeProductionResolvedInputPath:
    source_kind: str
    path: Path
    path_source: str
    root: Path
    root_source: str
    configured_root_policy: str
    field_name: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_kind": self.source_kind,
            "path": str(self.path),
            "path_source": self.path_source,
            "root": str(self.root),
            "root_source": self.root_source,
            "configured_root_policy": self.configured_root_policy,
            "field_name": self.field_name,
        }


def resolve_regime_source_root_candidates(
    source_kind: str,
    *,
    explicit_roots: Mapping[str, Any] | None = None,
    cli_args: Mapping[str, Any] | None = None,
    manifest: Mapping[str, Any] | str | Path | None = None,
    runtime_config: Mapping[str, Any] | str | Path | None = None,
    env: Mapping[str, str] | None = None,
    profile: str | None = None,
    project_root: str | Path | None = None,
    include_project_defaults: bool = True,
) -> list[tuple[Path, str]]:
    kind = _source_kind(source_kind)
    source_env = env if env is not None else os.environ
    resolved_profile = str(profile or selected_profile(env=source_env)).strip() or "production"
    project = resolve_project_root(project_root)
    candidates: list[tuple[Path, str]] = []

    candidates.extend(_mapping_candidates(explicit_roots, kind, "explicit_argument", project_root=project))
    candidates.extend(_mapping_candidates(cli_args, kind, "cli_argument", project_root=project))
    candidates.extend(_mapping_candidates(_load_mapping(manifest), kind, "manifest_field", project_root=project))
    candidates.extend(_mapping_candidates(_load_mapping(runtime_config), kind, "runtime_config", project_root=project))
    candidates.extend(_path_config_candidates(kind, profile=resolved_profile, env=source_env, project_root=project))
    candidates.extend(_env_candidates(kind, env=source_env, project_root=project))

    if include_project_defaults:
        candidates.extend(_project_default_candidates(kind, project))

    return _dedupe_candidates(candidates)


def resolve_regime_write_root(
    output_root: str | Path | None,
    *,
    env: Mapping[str, str] | None = None,
    project_root: str | Path | None = None,
    subdir: str | None = None,
    allow_project_default: bool = True,
    explicit_source: str = "explicit_argument",
) -> tuple[Path, str]:
    if output_root is not None and str(output_root).strip():
        root = validate_report_root(output_root, project_root=project_root, allow_foundation_descendant=True)
        return root, explicit_source
    source_env = env if env is not None else os.environ
    raw = str(source_env.get("PIPELINE_REGIME_FOUNDATION_REPORT_ROOT", "") or "").strip()
    if raw:
        root = validate_report_root(raw, project_root=project_root, allow_foundation_descendant=True)
        return (root / subdir, "env.PIPELINE_REGIME_FOUNDATION_REPORT_ROOT/subdir") if subdir else (root, "env.PIPELINE_REGIME_FOUNDATION_REPORT_ROOT")
    if not allow_project_default:
        raise PathConfigError("Regime Production write root is not configured: PIPELINE_REGIME_FOUNDATION_REPORT_ROOT")
    root = default_foundation_report_root(*(subdir,) if subdir is not None else (), env=source_env, project_root=project_root)
    return validate_report_root(root, project_root=project_root, allow_foundation_descendant=True), "project_default.foundation_report_root"


def resolve_regime_production_write_root(
    output_root: str | Path | None,
    *,
    env: Mapping[str, str] | None = None,
    project_root: str | Path | None = None,
    subdir: str | None = None,
    allow_explicit_dry_test_override: bool = False,
) -> tuple[Path, str]:
    if output_root is not None and str(output_root).strip():
        if not allow_explicit_dry_test_override:
            raise PathConfigError("Explicit Regime Production write root overrides are allowed only for dry/smoke tests")
        return resolve_regime_write_root(
            output_root,
            env=env,
            project_root=project_root,
            subdir=subdir,
            allow_project_default=False,
            explicit_source=REGIME_PRODUCTION_DRY_TEST_OVERRIDE_SOURCE,
        )
    return resolve_regime_write_root(
        None,
        env=env,
        project_root=project_root,
        subdir=subdir,
        allow_project_default=False,
    )


def resolve_required_regime_production_source_root(
    source_kind: str,
    *,
    explicit_roots: Mapping[str, Any] | None = None,
    cli_args: Mapping[str, Any] | None = None,
    manifest: Mapping[str, Any] | str | Path | None = None,
    runtime_config: Mapping[str, Any] | str | Path | None = None,
    env: Mapping[str, str] | None = None,
    profile: str | None = None,
    project_root: str | Path | None = None,
) -> tuple[Path, str]:
    kind = _source_kind(source_kind)
    candidates = resolve_regime_source_root_candidates(
        kind,
        explicit_roots=explicit_roots,
        cli_args=cli_args,
        manifest=manifest,
        runtime_config=runtime_config,
        env=env,
        profile=profile,
        project_root=project_root,
        include_project_defaults=False,
    )
    if not candidates:
        raise PathConfigError(f"Regime Production source root is not configured for {kind}")
    return candidates[0]


def resolve_regime_production_sidecar_input_path(
    source_kind: str,
    raw_path: Any,
    *,
    field_name: str,
    explicit_roots: Mapping[str, Any] | None = None,
    cli_args: Mapping[str, Any] | None = None,
    manifest: Mapping[str, Any] | str | Path | None = None,
    runtime_config: Mapping[str, Any] | str | Path | None = None,
    env: Mapping[str, str] | None = None,
    profile: str | None = None,
    project_root: str | Path | None = None,
    allow_branch_policy_manifest_file: bool = False,
) -> RegimeProductionResolvedInputPath:
    kind = _source_kind(source_kind)
    text = str(raw_path or "").strip()
    if not text:
        raise PathConfigError(f"Regime Production sidecar input is not declared: {field_name}")
    project = resolve_project_root(project_root)
    path = resolve_project_path(text, project_root=project)
    candidates = resolve_regime_source_root_candidates(
        kind,
        explicit_roots=explicit_roots,
        cli_args=cli_args,
        manifest=manifest,
        runtime_config=runtime_config,
        env=env,
        profile=profile,
        project_root=project,
        include_project_defaults=False,
    )
    for root, source in candidates:
        if is_relative_to(path, root):
            return RegimeProductionResolvedInputPath(
                source_kind=kind,
                path=path,
                path_source=f"manifest_field.{field_name}",
                root=root,
                root_source=source,
                configured_root_policy=REGIME_PRODUCTION_CONFIGURED_ROOTS_ONLY,
                field_name=field_name,
            )
    if allow_branch_policy_manifest_file:
        return RegimeProductionResolvedInputPath(
            source_kind=kind,
            path=path,
            path_source=f"manifest_field.{field_name}",
            root=path.parent,
            root_source=f"manifest_field.{field_name}/parent",
            configured_root_policy=REGIME_PRODUCTION_BRANCH_POLICY_SIDECAR_FILE,
            field_name=field_name,
        )
    raise PathConfigError(f"Regime Production sidecar input root is not configured for {field_name}")


def _source_kind(value: str) -> str:
    text = str(value).strip()
    if text not in READ_SOURCE_KINDS and text not in WRITE_SOURCE_KINDS:
        raise ValueError(f"Unknown Regime source kind: {value!r}")
    return text


def _mapping_candidates(
    payload: Mapping[str, Any] | None,
    kind: str,
    source_prefix: str,
    *,
    project_root: Path,
) -> list[tuple[Path, str]]:
    if not payload:
        return []
    out: list[tuple[Path, str]] = []
    for key in ROOT_FIELD_ALIASES[kind]:
        value = _nested_lookup(payload, key)
        if value is None:
            continue
        path = _clean_path(value)
        if path is not None:
            out.append((resolve_project_path(path, project_root=project_root), f"{source_prefix}.{key}"))
    return out


def _path_config_candidates(kind: str, *, profile: str, env: Mapping[str, str], project_root: Path) -> list[tuple[Path, str]]:
    key_by_kind = {
        SOURCE_KIND_OHLCVT: "source_ohlcvt_root",
        SOURCE_KIND_SCALAR_FEATURES: "source_feature_root",
        SOURCE_KIND_REGIME_FEATURES: "source_regime_root",
    }
    key = key_by_kind.get(kind)
    if key is None:
        return []
    path = resolve_path(key, profile=profile, env=env, required=False)
    if path is None:
        return []
    return [(resolve_project_path(path, project_root=project_root), f"path_config.{key}")]


def _env_candidates(kind: str, *, env: Mapping[str, str], project_root: Path) -> list[tuple[Path, str]]:
    out: list[tuple[Path, str]] = []
    for env_name in ENV_ALIASES[kind]:
        path = _clean_path(env.get(env_name, ""))
        if path is not None:
            out.append((resolve_project_path(path, project_root=project_root), f"env.{env_name}"))
    pipeline_root = _clean_path(env.get("PIPELINE_ROOT", ""))
    if pipeline_root is not None and kind in {SOURCE_KIND_OHLCVT, SOURCE_KIND_SCALAR_FEATURES}:
        out.append((resolve_project_path(pipeline_root / "parquet", project_root=project_root), "env.PIPELINE_ROOT/parquet"))
    return out


def _project_default_candidates(kind: str, project_root: Path) -> list[tuple[Path, str]]:
    defaults = {
        SOURCE_KIND_OHLCVT: (project_root / "parquet", "project_relative.parquet"),
        SOURCE_KIND_SCALAR_FEATURES: (project_root / "parquet", "project_relative.parquet"),
        SOURCE_KIND_REGIME_FEATURES: (project_root / "reports" / "regimes" / "foundation" / "regime_features", "project_relative.foundation_regime_features"),
        SOURCE_KIND_MARKET_STATE_UNIVERSE: (project_root / "reports" / "regimes" / "foundation" / "market_state_universe", "project_relative.market_state_universe"),
        SOURCE_KIND_RELATIONSHIP_DISCOVERY: (project_root / "reports" / "regimes" / "foundation" / "relationship_discovery_v1", "project_relative.relationship_discovery_v1"),
        SOURCE_KIND_UNIVERSE_ELIGIBILITY: (project_root / "reports" / "regimes" / "foundation" / "universe_feature_policy", "project_relative.universe_feature_policy"),
        SOURCE_KIND_REPORT_OUTPUT_ROOT: (project_root / "reports" / "regimes" / "foundation", "project_relative.foundation_report_root"),
    }
    path, source = defaults[kind]
    return [(path.resolve(), source)]


def _load_mapping(source: Mapping[str, Any] | str | Path | None) -> Mapping[str, Any] | None:
    if source is None:
        return None
    if isinstance(source, Mapping):
        return source
    path = Path(source)
    if not path.exists() or not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, Mapping) else None


def _nested_lookup(payload: Mapping[str, Any], key: str) -> Any:
    if key in payload:
        return payload[key]
    for container_name in ("paths", "source_roots", "roots", "inputs", "outputs"):
        nested = payload.get(container_name)
        if isinstance(nested, Mapping) and key in nested:
            return nested[key]
    modules = payload.get("modules")
    if isinstance(modules, Mapping):
        for module_payload in modules.values():
            if isinstance(module_payload, Mapping):
                found = _nested_lookup(module_payload, key)
                if found is not None:
                    return found
    return None


def _clean_path(value: Any) -> Path | None:
    text = str(value or "").strip()
    if not text:
        return None
    return Path(text).expanduser()


def _dedupe_candidates(candidates: list[tuple[Path, str]]) -> list[tuple[Path, str]]:
    out: list[tuple[Path, str]] = []
    seen: set[Path] = set()
    for path, source in candidates:
        resolved = Path(path).resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        out.append((resolved, source))
    return out


__all__ = [
    "READ_SOURCE_KINDS",
    "REGIME_PRODUCTION_BRANCH_POLICY_SIDECAR_FILE",
    "REGIME_PRODUCTION_CONFIGURED_ROOTS_ONLY",
    "REGIME_PRODUCTION_DRY_TEST_OVERRIDE_SOURCE",
    "ROOT_FIELD_ALIASES",
    "SOURCE_KIND_MARKET_STATE_UNIVERSE",
    "SOURCE_KIND_OHLCVT",
    "SOURCE_KIND_REGIME_FEATURES",
    "SOURCE_KIND_RELATIONSHIP_DISCOVERY",
    "SOURCE_KIND_REPORT_OUTPUT_ROOT",
    "SOURCE_KIND_SCALAR_FEATURES",
    "SOURCE_KIND_UNIVERSE_ELIGIBILITY",
    "RegimeProductionResolvedInputPath",
    "WRITE_SOURCE_KINDS",
    "resolve_regime_production_sidecar_input_path",
    "resolve_regime_production_write_root",
    "resolve_required_regime_production_source_root",
    "resolve_regime_source_root_candidates",
    "resolve_regime_write_root",
]
