from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional, Sequence


class PathConfigError(RuntimeError):
    """Raised when required pipeline paths are not configured."""


PATH_KEYS = (
    "source_ohlcvt_root",
    "source_feature_root",
    "source_regime_root",
    "output_parquet_root",
    "log_root",
    "state_root",
    "tmp_root",
    "regime_definition_root",
)

REQUIRED_PATH_KEYS = PATH_KEYS

ENV_BY_KEY: dict[str, tuple[str, ...]] = {
    "source_ohlcvt_root": ("PIPELINE_SOURCE_OHLCVT_ROOT", "PIPELINE_SOURCE_PARQUET_ROOT"),
    "source_feature_root": ("PIPELINE_SOURCE_FEATURES_ROOT", "PIPELINE_PARQUET_FEATURES_ROOT"),
    "source_regime_root": ("PIPELINE_SOURCE_REGIME_ROOT", "PIPELINE_PARQUET_REGIME_ROOT"),
    "output_parquet_root": ("PIPELINE_PARQUET_ROOT",),
    "log_root": ("PIPELINE_LOG_ROOT",),
    "state_root": ("PIPELINE_STATE_ROOT",),
    "tmp_root": ("PIPELINE_TMP_ROOT", "TMPDIR", "TEMP", "TMP"),
    "regime_definition_root": ("PIPELINE_REGIME_DEFINITION_ROOT",),
}


@dataclass(frozen=True)
class PipelineIOConfig:
    profile: str
    source_ohlcvt_root: Path
    source_feature_root: Path
    source_regime_root: Path
    output_parquet_root: Path
    log_root: Path
    state_root: Path
    tmp_root: Path
    regime_definition_root: Path

    def as_env(self) -> dict[str, str]:
        return {
            "PIPELINE_PROFILE": self.profile,
            "PIPELINE_SOURCE_OHLCVT_ROOT": str(self.source_ohlcvt_root),
            "PIPELINE_SOURCE_PARQUET_ROOT": str(self.source_ohlcvt_root),
            "PIPELINE_SOURCE_FEATURES_ROOT": str(self.source_feature_root),
            "PIPELINE_SOURCE_REGIME_ROOT": str(self.source_regime_root),
            "PIPELINE_PARQUET_ROOT": str(self.output_parquet_root),
            "PIPELINE_PARQUET_FEATURES_ROOT": str(self.source_feature_root),
            "PIPELINE_PARQUET_REGIME_ROOT": str(self.source_regime_root),
            "PIPELINE_LOG_ROOT": str(self.log_root),
            "PIPELINE_STATE_ROOT": str(self.state_root),
            "PIPELINE_TMP_ROOT": str(self.tmp_root),
            "PIPELINE_REGIME_DEFINITION_ROOT": str(self.regime_definition_root),
        }


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def default_local_config_path() -> Path:
    return project_root() / "config" / "path_config.local.toml"


def selected_profile(default: str = "production", env: Optional[Mapping[str, str]] = None) -> str:
    source_env = env if env is not None else os.environ
    return str(source_env.get("PIPELINE_PROFILE", "") or default).strip() or default


def _read_config(path: Optional[Path] = None) -> dict:
    raw_path = path or Path(os.getenv("PIPELINE_PATH_CONFIG", "") or default_local_config_path())
    if not raw_path.exists():
        return {}
    with raw_path.open("rb") as f:
        payload = tomllib.load(f)
    return payload if isinstance(payload, dict) else {}


def _profile_paths(payload: Mapping[str, object], profile: str) -> Mapping[str, object]:
    profiles = payload.get("profiles", {})
    if not isinstance(profiles, Mapping):
        return {}
    profile_payload = profiles.get(str(profile), {})
    if not isinstance(profile_payload, Mapping):
        return {}
    paths = profile_payload.get("paths", {})
    return paths if isinstance(paths, Mapping) else {}


def _clean_path(value: object) -> Optional[Path]:
    raw = str(value or "").strip()
    if not raw:
        return None
    return Path(raw).expanduser()


def _env_path(key: str, env: Mapping[str, str]) -> Optional[Path]:
    for env_name in ENV_BY_KEY.get(key, ()):
        path = _clean_path(env.get(env_name, ""))
        if path is not None:
            return path
    return None


def resolve_path(
    key: str,
    *,
    profile: Optional[str] = None,
    overrides: Optional[Mapping[str, object]] = None,
    env: Optional[Mapping[str, str]] = None,
    config_path: Optional[Path] = None,
    required: bool = False,
) -> Optional[Path]:
    if key not in PATH_KEYS:
        raise KeyError(f"Unknown pipeline path key: {key}")
    source_env = env if env is not None else os.environ
    resolved_profile = str(profile or selected_profile(env=source_env))

    if overrides is not None:
        override_path = _clean_path(overrides.get(key, ""))
        if override_path is not None:
            return override_path

    payload = _read_config(config_path)
    profile_path = _clean_path(_profile_paths(payload, resolved_profile).get(key, ""))
    if profile_path is not None:
        return profile_path

    env_path = _env_path(key, source_env)
    if env_path is not None:
        return env_path

    if required:
        raise PathConfigError(f"Required pipeline path is not configured for profile '{resolved_profile}': {key}")
    return None


def resolve_pipeline_io(
    *,
    profile: Optional[str] = None,
    overrides: Optional[Mapping[str, object]] = None,
    env: Optional[Mapping[str, str]] = None,
    config_path: Optional[Path] = None,
    required: bool = True,
    required_keys: Sequence[str] = REQUIRED_PATH_KEYS,
) -> Optional[PipelineIOConfig]:
    source_env = env if env is not None else os.environ
    resolved_profile = str(profile or selected_profile(env=source_env))
    values: dict[str, Path] = {}
    missing: list[str] = []

    for key in PATH_KEYS:
        path = resolve_path(
            key,
            profile=resolved_profile,
            overrides=overrides,
            env=source_env,
            config_path=config_path,
            required=False,
        )
        if path is None:
            if key in set(required_keys):
                missing.append(key)
            continue
        values[key] = path

    if missing:
        if required:
            names = ", ".join(sorted(missing))
            raise PathConfigError(f"Required pipeline paths are not configured for profile '{resolved_profile}': {names}")
        return None

    if not all(key in values for key in PATH_KEYS):
        return None
    return PipelineIOConfig(profile=resolved_profile, **values)


def require_pipeline_io(
    *,
    profile: Optional[str] = None,
    overrides: Optional[Mapping[str, object]] = None,
    env: Optional[Mapping[str, str]] = None,
    config_path: Optional[Path] = None,
    required_keys: Sequence[str] = REQUIRED_PATH_KEYS,
) -> PipelineIOConfig:
    config = resolve_pipeline_io(
        profile=profile,
        overrides=overrides,
        env=env,
        config_path=config_path,
        required=True,
        required_keys=required_keys,
    )
    if config is None:
        resolved_profile = str(profile or selected_profile(env=env))
        raise PathConfigError(f"Required pipeline paths are not configured for profile '{resolved_profile}'")
    return config


def pipeline_io_env(config: PipelineIOConfig, base_env: Optional[Mapping[str, str]] = None) -> dict[str, str]:
    env = {str(k): str(v) for k, v in (base_env if base_env is not None else os.environ).items()}
    env.update(config.as_env())
    return env
