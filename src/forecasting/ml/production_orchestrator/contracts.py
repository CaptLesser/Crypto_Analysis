from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

from src.forecasting.common.path_config import resolve_pipeline_io, selected_profile
from src.forecasting.common.runtime_config import RUNTIME_CONFIG_PATH, load_runtime_config
from src.forecasting.ml.bayesian.shared.bayesian_numeric_model_registry import (
    BAYESIAN_NUMERIC_FAMILY_ROOT_ENVS,
    BAYESIAN_NUMERIC_FAMILY_ROOT_NAMES,
)
from src.forecasting.ml.neural.shared.neural_numeric_model_registry import (
    NEURAL_NUMERIC_FAMILY_ROOT_ENVS,
    NEURAL_NUMERIC_FAMILY_ROOT_NAMES,
)
from src.forecasting.ml.production_orchestrator.common import load_json_dict
from src.forecasting.stats.shared.stats_numeric_model_registry import (
    STATS_NUMERIC_FAMILY_ROOT_ENVS,
    STATS_NUMERIC_FAMILY_ROOT_NAMES,
)

STATS_MANIFEST_FILES = {
    "sarimax": "sarimax_run_manifest.json",
    "llt": "llt_run_manifest.json",
    "egarch": "egarch_run_manifest.json",
    "quantreg": "quantreg_run_manifest.json",
}

STATS_SKIPPED_FILES = {
    "sarimax": "sarimax_skipped.json",
    "llt": "llt_skipped.json",
    "egarch": "egarch_skipped.json",
    "quantreg": "quantreg_skipped.json",
}


@dataclass(frozen=True)
class ContractCheck:
    name: str
    target: str
    ok: bool
    detail: Optional[str] = None


@dataclass(frozen=True)
class ContractSpec:
    family: str
    module_key: str
    output_root: Path
    log_path: Path
    runtime_config_path: Optional[Path]
    runtime_config_key: Optional[str]
    resolved_runtime_values: Dict[str, Any]
    required_files: List[Path] = field(default_factory=list)
    required_nonempty_files: List[Path] = field(default_factory=list)
    required_json_files: List[Path] = field(default_factory=list)
    required_json_keys: Dict[str, List[str]] = field(default_factory=dict)
    required_nonempty_roots: List[Path] = field(default_factory=list)
    forbidden_globs: List[str] = field(default_factory=list)
    completion_markers: List[str] = field(default_factory=list)
    anchor_roots: List[Path] = field(default_factory=list)
    workload_roots: List[Path] = field(default_factory=list)
    cleanup_roots: List[Path] = field(default_factory=list)
    output_table_prefixes: List[str] = field(default_factory=list)


def _load_runtime_values(runtime_config_key: Optional[str]) -> tuple[Optional[Path], Dict[str, Any]]:
    if runtime_config_key is None:
        return None, {}
    payload = load_runtime_config()
    modules = payload.get("modules", {}) if isinstance(payload, dict) else {}
    module_cfg = modules.get(runtime_config_key, {}) if isinstance(modules, dict) else {}
    return RUNTIME_CONFIG_PATH, (dict(module_cfg) if isinstance(module_cfg, dict) else {})


def _log_contains_any_marker(log_path: Path, markers: Iterable[str]) -> bool:
    if not log_path.exists():
        return False
    try:
        text = log_path.read_text(encoding="utf-8", errors="ignore").lower()
    except Exception:
        return False
    return any(str(marker).strip().lower() in text for marker in markers if str(marker).strip())


def _has_nonempty_file_under(root: Path, suffix: str) -> bool:
    if not root.exists():
        return False
    for path in root.rglob(f"*{suffix}"):
        try:
            if path.is_file() and path.stat().st_size > 0:
                return True
        except Exception:
            continue
    return False


def _count_outputs(root: Path, suffix: str) -> int:
    if not root.exists():
        return 0
    count = 0
    for path in root.rglob(f"*{suffix}"):
        try:
            if path.is_file() and path.stat().st_size > 0:
                count += 1
        except Exception:
            continue
    return count


def _count_state_files(root: Path, suffix: str) -> int:
    return _count_outputs(root, suffix)


def _matching_table_roots(parquet_root: Path, prefixes: Iterable[str]) -> List[Path]:
    roots: List[Path] = []
    if not parquet_root.exists():
        return roots
    prefix_set = [str(prefix).strip() for prefix in prefixes if str(prefix).strip()]
    if not prefix_set:
        return roots
    try:
        for child in parquet_root.iterdir():
            if not child.is_dir():
                continue
            name = child.name
            if any(name == prefix or name.startswith(f"{prefix}_") for prefix in prefix_set):
                roots.append(child)
    except Exception:
        return roots
    return roots


def _env_get(env: Mapping[str, str], key: str, default: str = "") -> str:
    return str(env.get(key, default) or "").strip()


def _common_forecast_root(env: Mapping[str, str], family_root_env: str, family_root_name: str) -> Path:
    io_config = resolve_pipeline_io(profile=selected_profile(env=env), env=env, required=False)
    parquet_root = Path(
        _env_get(env, family_root_env)
        or _env_get(env, "PIPELINE_PARQUET_ROOT")
        or (io_config.output_parquet_root if io_config is not None else Path("parquet"))
    ).resolve()
    return (parquet_root / family_root_name).resolve()


def build_contract_spec(module_spec: Any, *, env: Optional[Mapping[str, str]] = None) -> ContractSpec:
    source_env = env if env is not None else os.environ
    io_config = resolve_pipeline_io(profile=selected_profile(env=source_env), env=source_env, required=False)
    output_root = Path(_env_get(source_env, "PIPELINE_SANDBOX_PARQUET_ROOT") or (io_config.output_parquet_root if io_config is not None else Path("parquet")))
    log_root = Path(_env_get(source_env, "PIPELINE_SANDBOX_LOG_ROOT") or (io_config.log_root if io_config is not None else Path("logs")))
    state_base = Path(_env_get(source_env, "PIPELINE_SANDBOX_STATE_ROOT") or (io_config.state_root if io_config is not None else Path("model_states")))
    tmp_root = Path(_env_get(source_env, "PIPELINE_SANDBOX_TMP_ROOT") or (io_config.tmp_root if io_config is not None else Path("tmp")))
    if str(module_spec.family) == "tabular":
        parquet_root = Path(
            _env_get(source_env, module_spec.parquet_root_env)
            or _env_get(source_env, "PIPELINE_PARQUET_ROOT")
            or output_root
        ).resolve()
        state_root = (state_base / module_spec.runtime_config_key).resolve()
        staging_root = (tmp_root / f"{module_spec.runtime_config_key}_stage").resolve()
        log_path = (log_root / f"{module_spec.runtime_config_key}.log").resolve()
        runtime_config_path, runtime_values = _load_runtime_values(module_spec.runtime_config_key)
        return ContractSpec(
            family=str(module_spec.family),
            module_key=str(module_spec.module_key),
            output_root=parquet_root,
            log_path=log_path,
            runtime_config_path=runtime_config_path,
            runtime_config_key=module_spec.runtime_config_key,
            resolved_runtime_values=runtime_values,
            required_nonempty_files=[log_path],
            required_nonempty_roots=[state_root],
            forbidden_globs=["*.tmp", "*.partial", "*.parquet.tmp", "*.pkl.tmp"],
            completion_markers=["run complete"],
            anchor_roots=[state_root, staging_root, log_path],
            workload_roots=[state_root],
            cleanup_roots=[staging_root],
            output_table_prefixes=[module_spec.runtime_config_key, f"{module_spec.runtime_config_key}_eval"],
        )
    if str(module_spec.family) == "bayesian":
        forecast_root = _common_forecast_root(
            source_env,
            BAYESIAN_NUMERIC_FAMILY_ROOT_ENVS[module_spec.module_key],
            BAYESIAN_NUMERIC_FAMILY_ROOT_NAMES[module_spec.module_key],
        )
        state_root = (forecast_root / "state").resolve()
        manifest_path = (state_root / "bayes_run_manifest.json").resolve()
        skipped_path = (state_root / "bayes_skipped.json").resolve()
        staging_root = (forecast_root / "tmp" / f"{module_spec.module_tag}_stage").resolve()
        runtime_config_path, runtime_values = _load_runtime_values("bayesian_numeric_runner")
        return ContractSpec(
            family=str(module_spec.family),
            module_key=str(module_spec.module_key),
            output_root=forecast_root.resolve(),
            log_path=Path(),
            runtime_config_path=runtime_config_path,
            runtime_config_key="bayesian_numeric_runner",
            resolved_runtime_values=runtime_values,
            required_files=[manifest_path, skipped_path],
            required_json_files=[manifest_path, skipped_path],
            required_json_keys={
                str(manifest_path): ["run_id", "parts", "dispatch_slots", "job_shard_count"],
                str(skipped_path): ["run_id", "units"],
            },
            forbidden_globs=["*.tmp", "*.partial", "*.parquet.tmp", "*.json.tmp"],
            completion_markers=["run complete"],
            anchor_roots=[forecast_root.resolve(), state_root, manifest_path, skipped_path],
            workload_roots=[forecast_root.resolve()],
            cleanup_roots=[staging_root],
        )
    if str(module_spec.family) == "stats":
        forecast_root = _common_forecast_root(
            source_env,
            STATS_NUMERIC_FAMILY_ROOT_ENVS[module_spec.module_key],
            STATS_NUMERIC_FAMILY_ROOT_NAMES[module_spec.module_key],
        )
        state_root = (forecast_root / "state").resolve()
        manifest_path = (state_root / STATS_MANIFEST_FILES[module_spec.module_key]).resolve()
        skipped_path = (state_root / STATS_SKIPPED_FILES[module_spec.module_key]).resolve()
        staging_root = (forecast_root / "tmp" / f"{module_spec.module_tag}_stage").resolve()
        runtime_config_path, runtime_values = _load_runtime_values("stats_numeric_runner")
        return ContractSpec(
            family=str(module_spec.family),
            module_key=str(module_spec.module_key),
            output_root=forecast_root.resolve(),
            log_path=Path(),
            runtime_config_path=runtime_config_path,
            runtime_config_key="stats_numeric_runner",
            resolved_runtime_values=runtime_values,
            required_files=[manifest_path, skipped_path],
            required_json_files=[manifest_path, skipped_path],
            required_json_keys={
                str(manifest_path): ["run_id", "parts", "unit_entries", "skipped_units", "forecast_output_root", "eval_output_root"],
                str(skipped_path): ["run_id", "units"],
            },
            required_nonempty_roots=[forecast_root.resolve()],
            forbidden_globs=["*.tmp", "*.partial", "*.parquet.tmp", "*.json.tmp"],
            completion_markers=["run complete"],
            anchor_roots=[forecast_root.resolve(), state_root, manifest_path, skipped_path],
            workload_roots=[forecast_root.resolve()],
            cleanup_roots=[staging_root],
        )
    forecast_root = _common_forecast_root(
        source_env,
        NEURAL_NUMERIC_FAMILY_ROOT_ENVS[module_spec.module_key],
        NEURAL_NUMERIC_FAMILY_ROOT_NAMES[module_spec.module_key],
    )
    state_root = (forecast_root / "state").resolve()
    manifest_path = (state_root / "neural_run_manifest.json").resolve()
    skipped_path = (state_root / "neural_skipped.json").resolve()
    staging_root = (forecast_root / "tmp" / f"{module_spec.module_tag}_stage").resolve()
    runtime_config_path, runtime_values = _load_runtime_values("neural_numeric_runner")
    return ContractSpec(
        family=str(module_spec.family),
        module_key=str(module_spec.module_key),
        output_root=forecast_root.resolve(),
        log_path=Path(),
        runtime_config_path=runtime_config_path,
        runtime_config_key="neural_numeric_runner",
        resolved_runtime_values=runtime_values,
        required_files=[manifest_path, skipped_path],
        required_json_files=[manifest_path, skipped_path],
        required_json_keys={
            str(manifest_path): ["run_id", "parts", "dispatch_slots", "job_shard_count"],
            str(skipped_path): ["run_id", "units"],
        },
        forbidden_globs=["*.tmp", "*.partial", "*.parquet.tmp", "*.json.tmp"],
        completion_markers=["run complete"],
        anchor_roots=[forecast_root.resolve(), state_root, manifest_path, skipped_path],
        workload_roots=[forecast_root.resolve()],
        cleanup_roots=[staging_root],
    )


def snapshot_payload(
    module_spec: Any,
    contract_spec: ContractSpec,
    *,
    command: List[str],
    cwd: Path,
    git_head: Optional[str],
    env: Optional[Mapping[str, str]] = None,
) -> Dict[str, Any]:
    expected_artifacts = {
        "required_files": [str(path) for path in contract_spec.required_files],
        "required_nonempty_files": [str(path) for path in contract_spec.required_nonempty_files],
        "required_json_files": [str(path) for path in contract_spec.required_json_files],
        "required_json_keys": dict(contract_spec.required_json_keys),
        "required_nonempty_roots": [str(path) for path in contract_spec.required_nonempty_roots],
        "output_table_prefixes": list(contract_spec.output_table_prefixes),
        "forbidden_globs": list(contract_spec.forbidden_globs),
    }
    return {
        "schema_version": 1,
        "module_key": str(module_spec.module_key),
        "family": str(module_spec.family),
        "command": [str(part) for part in command],
        "cwd": str(cwd),
        "output_root": str(contract_spec.output_root),
        "runtime_config": {
            "path": (str(contract_spec.runtime_config_path) if contract_spec.runtime_config_path is not None else None),
            "module_key": contract_spec.runtime_config_key,
            "resolved_values": dict(contract_spec.resolved_runtime_values),
        },
        "env_hints": dict(module_spec.env_hints(env)),
        "expected_artifacts": expected_artifacts,
        "git_head": git_head,
    }


def validate_contract(contract_spec: ContractSpec) -> tuple[str, Optional[str], Dict[str, Any]]:
    checks: List[ContractCheck] = []
    failure_type: Optional[str] = None
    for path in contract_spec.required_files:
        ok = path.exists()
        checks.append(ContractCheck(name="required_file_exists", target=str(path), ok=bool(ok)))
        if not ok and failure_type is None:
            failure_type = "artifact_missing"
    for path in contract_spec.required_nonempty_files:
        ok = path.exists() and path.is_file() and path.stat().st_size > 0
        checks.append(ContractCheck(name="required_file_nonempty", target=str(path), ok=bool(ok)))
        if not ok and failure_type is None:
            failure_type = "artifact_missing"
    for path in contract_spec.required_json_files:
        payload = load_json_dict(path)
        ok = bool(payload)
        checks.append(ContractCheck(name="required_json_exists", target=str(path), ok=bool(ok)))
        if not ok and failure_type is None:
            failure_type = "malformed_manifest"
    for path_str, keys in contract_spec.required_json_keys.items():
        payload = load_json_dict(Path(path_str))
        ok = all(key in payload for key in keys)
        checks.append(ContractCheck(name="required_json_keys", target=str(path_str), ok=bool(ok), detail=",".join(keys)))
        if not ok and failure_type is None:
            failure_type = "malformed_manifest"
    for root in contract_spec.required_nonempty_roots:
        suffix = ".parquet" if str(contract_spec.output_root) == str(root) else (".pkl" if "model_states" in str(root) else ".parquet")
        ok = _has_nonempty_file_under(root, suffix)
        checks.append(ContractCheck(name="required_nonempty_root", target=str(root), ok=bool(ok), detail=suffix))
        if not ok and failure_type is None:
            failure_type = "unexpected_empty_output"
    if contract_spec.output_table_prefixes:
        table_roots = _matching_table_roots(contract_spec.output_root, contract_spec.output_table_prefixes)
        ok = any(_has_nonempty_file_under(root, ".parquet") for root in table_roots)
        checks.append(
            ContractCheck(
                name="required_nonempty_output_tables",
                target=str(contract_spec.output_root),
                ok=bool(ok),
                detail=",".join(str(root.name) for root in table_roots),
            )
        )
        if not ok and failure_type is None:
            failure_type = "unexpected_empty_output"
    for root in contract_spec.anchor_roots:
        if not root.exists():
            continue
        for pattern in contract_spec.forbidden_globs:
            matches = [str(path) for path in root.rglob(pattern)]
            ok = len(matches) == 0
            checks.append(ContractCheck(name="forbidden_glob_absent", target=f"{root}::{pattern}", ok=bool(ok), detail=(matches[0] if matches else None)))
            if not ok and failure_type is None:
                failure_type = "contract_failed"
    if contract_spec.output_table_prefixes:
        for root in _matching_table_roots(contract_spec.output_root, contract_spec.output_table_prefixes):
            for pattern in contract_spec.forbidden_globs:
                matches = [str(path) for path in root.rglob(pattern)]
                ok = len(matches) == 0
                checks.append(
                    ContractCheck(
                        name="forbidden_glob_absent",
                        target=f"{root}::{pattern}",
                        ok=bool(ok),
                        detail=(matches[0] if matches else None),
                    )
                )
                if not ok and failure_type is None:
                    failure_type = "contract_failed"
    if contract_spec.completion_markers and contract_spec.log_path and contract_spec.log_path.is_file():
        ok = _log_contains_any_marker(contract_spec.log_path, contract_spec.completion_markers)
        checks.append(ContractCheck(name="completion_marker", target=str(contract_spec.log_path), ok=bool(ok), detail=",".join(contract_spec.completion_markers)))
        if not ok and failure_type is None:
            failure_type = "contract_failed"
    status = "passed" if all(check.ok for check in checks) else "failed"
    workload_shape = {
        "rows_written": None,
        "parts_written": None,
        "assets_touched": None,
        "units_processed": None,
        "units_skipped": None,
        "resumed_units": None,
    }
    if contract_spec.required_json_files:
        for path in contract_spec.required_json_files:
            payload = load_json_dict(path)
            if "parts" in payload and isinstance(payload.get("parts"), list):
                parts = list(payload.get("parts") or [])
                workload_shape["parts_written"] = int(len(parts))
                try:
                    workload_shape["rows_written"] = int(sum(int(part.get("rows", 0) or 0) for part in parts))
                except Exception:
                    workload_shape["rows_written"] = None
                try:
                    workload_shape["assets_touched"] = int(len({str(Path(str(part.get("path", ""))).parent.parent.parent.name).replace("asset=", "") for part in parts if part.get("path")}))
                except Exception:
                    workload_shape["assets_touched"] = None
            if "skipped_units" in payload:
                try:
                    skipped = payload.get("skipped_units")
                    workload_shape["units_skipped"] = int(skipped if isinstance(skipped, int) else len(payload.get("units", {})))
                except Exception:
                    workload_shape["units_skipped"] = None
    result = {
        "schema_version": 1,
        "status": str(status),
        "failure_type": failure_type,
        "checks": [
            {"name": check.name, "target": check.target, "ok": bool(check.ok), "detail": check.detail}
            for check in checks
        ],
        "workload_shape": workload_shape,
    }
    return status, failure_type, result
