from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional, Sequence


class SandboxWritePathError(RuntimeError):
    """Raised when sandbox mode would write outside the sandbox root."""


@dataclass(frozen=True)
class SandboxOutputRoots:
    enabled: bool
    root: Path
    parquet_root: Path
    log_root: Path
    diagnostics_root: Path
    manifest_root: Path
    state_root: Path
    tmp_root: Path
    optuna_root: Path
    catboost_train_dir: Path
    regime_definition_root: Path
    runtime_artifact_root: Path


SANDBOX_ENV_MODE = "PIPELINE_SANDBOX_MODE"
SANDBOX_ENV_OUTPUT_ROOT = "PIPELINE_SANDBOX_OUTPUT_ROOT"

_TRUE_VALUES = {"1", "true", "yes", "on"}

_TABULAR_OUTPUT_ROOT_ENVS = (
    "PIPELINE_PARQUET_XGB_NUMERICS_ROOT",
    "PIPELINE_PARQUET_LGB_NUMERICS_ROOT",
    "PIPELINE_PARQUET_CB_NUMERICS_ROOT",
    "PIPELINE_PARQUET_RF_NUMERICS_ROOT",
    "PIPELINE_PARQUET_EN_NUMERICS_ROOT",
)

_BAYESIAN_OUTPUT_ROOT_ENVS = (
    "PIPELINE_PARQUET_BAYES_DLM_ROOT",
    "PIPELINE_PARQUET_BAYES_SV_ROOT",
    "PIPELINE_PARQUET_BAYES_DYNREG_ROOT",
    "PIPELINE_PARQUET_BAYES_COPULA_ROOT",
    "PIPELINE_PARQUET_BAYES_TAIL_ROOT",
)

_NEURAL_OUTPUT_ROOT_ENVS = (
    "PIPELINE_PARQUET_NEURALTS_LSTM_ROOT",
    "PIPELINE_PARQUET_NEURALTS_TCN_ROOT",
    "PIPELINE_PARQUET_NEURALTS_NBEATS_ROOT",
)

_STATS_OUTPUT_ROOT_ENVS = (
    "PIPELINE_PARQUET_SARIMAX_ROOT",
    "PIPELINE_PARQUET_LLT_ROOT",
    "PIPELINE_PARQUET_EGARCH_ROOT",
    "PIPELINE_PARQUET_QR_ROOT",
)

FAMILY_OUTPUT_ROOT_ENVS = (
    *_TABULAR_OUTPUT_ROOT_ENVS,
    *_BAYESIAN_OUTPUT_ROOT_ENVS,
    *_NEURAL_OUTPUT_ROOT_ENVS,
    *_STATS_OUTPUT_ROOT_ENVS,
)

_PRODUCTION_ARTIFACT_ROOT_ENVS = (
    "PIPELINE_ROOT",
    "PIPELINE_PARQUET_ROOT",
    "PIPELINE_PARQUET_FEATURES_ROOT",
    "PIPELINE_PARQUET_REGIME_ROOT",
    "PIPELINE_REGIME_DEFINITION_ROOT",
    *FAMILY_OUTPUT_ROOT_ENVS,
)


def _env_value(env: Mapping[str, str], key: str) -> str:
    return str(env.get(key, "") or "").strip()


def _is_truthy(raw: str) -> bool:
    return str(raw).strip().lower() in _TRUE_VALUES


def _resolve(path: Path) -> Path:
    return Path(path).expanduser().resolve()


def _disabled_roots() -> SandboxOutputRoots:
    empty = Path()
    return SandboxOutputRoots(
        enabled=False,
        root=empty,
        parquet_root=empty,
        log_root=empty,
        diagnostics_root=empty,
        manifest_root=empty,
        state_root=empty,
        tmp_root=empty,
        optuna_root=empty,
        catboost_train_dir=empty,
        regime_definition_root=empty,
        runtime_artifact_root=empty,
    )


def is_relative_to(path: Path, root: Path) -> bool:
    try:
        Path(path).resolve().relative_to(Path(root).resolve())
        return True
    except ValueError:
        return False


def default_production_roots(env: Optional[Mapping[str, str]] = None) -> tuple[Path, ...]:
    source_env = env if env is not None else os.environ
    roots: set[Path] = set()
    active_sandbox_root: Optional[Path] = None
    if _is_truthy(_env_value(source_env, SANDBOX_ENV_MODE)):
        raw_sandbox_root = _env_value(source_env, SANDBOX_ENV_OUTPUT_ROOT)
        if raw_sandbox_root:
            active_sandbox_root = _resolve(Path(raw_sandbox_root))

    def add_root(path: Path) -> None:
        resolved = _resolve(Path(path))
        if active_sandbox_root is not None and (
            resolved == active_sandbox_root or is_relative_to(resolved, active_sandbox_root)
        ):
            return
        roots.add(resolved)

    roots.add(_resolve(Path("D:/pipeline")))

    raw_pipeline_root = _env_value(source_env, "PIPELINE_ROOT")
    if raw_pipeline_root:
        pipeline_root = _resolve(Path(raw_pipeline_root))
        for path in (
            pipeline_root,
            Path(_env_value(source_env, "PIPELINE_PARQUET_ROOT") or pipeline_root / "parquet"),
            Path(_env_value(source_env, "PIPELINE_PARQUET_FEATURES_ROOT") or pipeline_root / "parquet"),
            Path(_env_value(source_env, "PIPELINE_PARQUET_REGIME_ROOT") or pipeline_root / "parquet"),
            Path(_env_value(source_env, "PIPELINE_REGIME_DEFINITION_ROOT") or pipeline_root / "regime_definitions"),
            pipeline_root / "logs",
            pipeline_root / "model_states",
            pipeline_root / "tmp",
        ):
            add_root(Path(path))
    for key in _PRODUCTION_ARTIFACT_ROOT_ENVS:
        raw = _env_value(source_env, key)
        if raw:
            add_root(Path(raw))
    return tuple(sorted(roots, key=lambda item: str(item).lower()))


def _validate_sandbox_root(root: Path, production_roots: Sequence[Path]) -> None:
    resolved = _resolve(root)
    for production_root in production_roots:
        prod = _resolve(production_root)
        if resolved == prod or is_relative_to(resolved, prod):
            raise SandboxWritePathError(
                f"Sandbox output root must not be inside a production root: {resolved} under {prod}"
            )


def roots_from_base(root: Path, *, env: Optional[Mapping[str, str]] = None) -> SandboxOutputRoots:
    resolved = _resolve(root)
    _validate_sandbox_root(resolved, default_production_roots(env))
    return SandboxOutputRoots(
        enabled=True,
        root=resolved,
        parquet_root=resolved / "parquet",
        log_root=resolved / "logs",
        diagnostics_root=resolved / "diagnostics",
        manifest_root=resolved / "manifests",
        state_root=resolved / "state",
        tmp_root=resolved / "tmp",
        optuna_root=resolved / "optuna",
        catboost_train_dir=resolved / "tmp" / "catboost_train",
        regime_definition_root=resolved / "regime_definitions",
        runtime_artifact_root=resolved / "runtime",
    )


def resolve_sandbox_output_roots(
    args: object | None = None,
    env: Optional[Mapping[str, str]] = None,
) -> SandboxOutputRoots:
    source_env = env if env is not None else os.environ
    arg_root = getattr(args, "sandbox_output_root", None) if args is not None else None
    raw_env_root = _env_value(source_env, SANDBOX_ENV_OUTPUT_ROOT)
    env_enabled = _is_truthy(_env_value(source_env, SANDBOX_ENV_MODE))
    raw_root = str(arg_root).strip() if arg_root else raw_env_root
    if not raw_root:
        if env_enabled:
            raise SandboxWritePathError(
                f"{SANDBOX_ENV_MODE}=1 requires {SANDBOX_ENV_OUTPUT_ROOT} to be set"
            )
        return _disabled_roots()
    return roots_from_base(Path(raw_root), env=source_env)


def _root_for_kind(roots: SandboxOutputRoots, kind: str) -> Path:
    normalized = str(kind).strip().lower().replace("-", "_")
    mapping = {
        "root": roots.root,
        "output": roots.root,
        "parquet": roots.parquet_root,
        "log": roots.log_root,
        "logs": roots.log_root,
        "diagnostic": roots.diagnostics_root,
        "diagnostics": roots.diagnostics_root,
        "manifest": roots.manifest_root,
        "manifests": roots.manifest_root,
        "state": roots.state_root,
        "tmp": roots.tmp_root,
        "temp": roots.tmp_root,
        "staging": roots.tmp_root,
        "cache": roots.tmp_root,
        "optuna": roots.optuna_root,
        "catboost": roots.catboost_train_dir,
        "catboost_train": roots.catboost_train_dir,
        "regime_definition": roots.regime_definition_root,
        "regime_definitions": roots.regime_definition_root,
        "runtime": roots.runtime_artifact_root,
        "runtime_artifact": roots.runtime_artifact_root,
    }
    try:
        return mapping[normalized]
    except KeyError as exc:
        raise KeyError(f"Unknown sandbox write path kind: {kind}") from exc


def sandbox_write_path(kind: str, *parts: object, roots: Optional[SandboxOutputRoots] = None) -> Path:
    resolved_roots = roots if roots is not None else resolve_sandbox_output_roots()
    if not resolved_roots.enabled:
        raise SandboxWritePathError("sandbox_write_path requires sandbox mode to be enabled")
    path = _root_for_kind(resolved_roots, kind)
    for part in parts:
        path = path / str(part)
    return path


def assert_write_allowed(
    path: Path,
    kind: str,
    *,
    roots: Optional[SandboxOutputRoots] = None,
) -> None:
    resolved_roots = roots if roots is not None else resolve_sandbox_output_roots()
    if not resolved_roots.enabled:
        return
    resolved = _resolve(Path(path))
    if resolved == resolved_roots.root or is_relative_to(resolved, resolved_roots.root):
        return
    raise SandboxWritePathError(f"Sandbox mode forbids {kind} write outside sandbox: {resolved}")


def assert_sandbox_write_path(path: Path, kind: str) -> None:
    assert_write_allowed(path, kind)


def _source_root_defaults(env: Mapping[str, str]) -> dict[str, str]:
    pipeline_root = _resolve(Path(_env_value(env, "PIPELINE_ROOT") or "."))
    parquet_root = _resolve(Path(_env_value(env, "PIPELINE_PARQUET_ROOT") or _env_value(env, "PIPELINE_SOURCE_PARQUET_ROOT") or pipeline_root / "parquet"))
    ohlcvt_root = _resolve(Path(_env_value(env, "PIPELINE_SOURCE_OHLCVT_ROOT") or parquet_root))
    features_root = _resolve(Path(_env_value(env, "PIPELINE_PARQUET_FEATURES_ROOT") or _env_value(env, "PIPELINE_SOURCE_FEATURES_ROOT") or parquet_root))
    regime_root = _resolve(Path(_env_value(env, "PIPELINE_PARQUET_REGIME_ROOT") or _env_value(env, "PIPELINE_SOURCE_REGIME_ROOT") or parquet_root))
    return {
        "PIPELINE_SOURCE_PARQUET_ROOT": str(parquet_root),
        "PIPELINE_SOURCE_OHLCVT_ROOT": str(ohlcvt_root),
        "PIPELINE_SOURCE_FEATURES_ROOT": str(features_root),
        "PIPELINE_SOURCE_REGIME_ROOT": str(regime_root),
        "PIPELINE_PARQUET_ROOT": str(parquet_root),
        "PIPELINE_PARQUET_FEATURES_ROOT": str(features_root),
    }


def sandbox_env_for_subprocess(
    roots: SandboxOutputRoots,
    source_env: Mapping[str, str],
) -> dict[str, str]:
    env = {str(key): str(value) for key, value in source_env.items()}
    if not roots.enabled:
        return env
    env.update(_source_root_defaults(source_env))
    env.update(
        {
            SANDBOX_ENV_MODE: "1",
            SANDBOX_ENV_OUTPUT_ROOT: str(roots.root),
            "PIPELINE_SANDBOX_PARQUET_ROOT": str(roots.parquet_root),
            "PIPELINE_SANDBOX_LOG_ROOT": str(roots.log_root),
            "PIPELINE_SANDBOX_STATE_ROOT": str(roots.state_root),
            "PIPELINE_SANDBOX_TMP_ROOT": str(roots.tmp_root),
            "PIPELINE_SANDBOX_DIAGNOSTICS_ROOT": str(roots.diagnostics_root),
            "PIPELINE_SANDBOX_MANIFEST_ROOT": str(roots.manifest_root),
            "PIPELINE_SANDBOX_OPTUNA_ROOT": str(roots.optuna_root),
            "PIPELINE_SANDBOX_CATBOOST_TRAIN_DIR": str(roots.catboost_train_dir),
            "PIPELINE_SANDBOX_REGIME_DEFINITION_ROOT": str(roots.regime_definition_root),
            "PIPELINE_SANDBOX_RUNTIME_ARTIFACT_ROOT": str(roots.runtime_artifact_root),
            "PIPELINE_ROOT": str(roots.root),
            "PIPELINE_LOG_ROOT": str(roots.log_root),
            "PIPELINE_STATE_ROOT": str(roots.state_root),
            "PIPELINE_TMP_ROOT": str(roots.tmp_root),
            "CATBOOST_TRAIN_DIR": str(roots.catboost_train_dir),
            "TMP": str(roots.tmp_root),
            "TEMP": str(roots.tmp_root),
            "TMPDIR": str(roots.tmp_root),
        }
    )
    for key in FAMILY_OUTPUT_ROOT_ENVS:
        env[key] = str(roots.parquet_root)
    env["PIPELINE_REGIME_DEFINITION_ROOT"] = str(roots.regime_definition_root)
    return env
