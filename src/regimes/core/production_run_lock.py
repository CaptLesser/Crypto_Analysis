from __future__ import annotations

import hashlib
import json
import os
import socket
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from src.forecasting.common.io_atomic import atomic_replace, sibling_temp_path
from src.regimes.core.paths import resolve_project_path
from src.regimes.core.production_consumer import REGIME_PRODUCTION_BRANCHES
from src.regimes.core.serialization import to_jsonable


REGIME_PRODUCTION_RUN_LOCK_SCHEMA_VERSION = 1
REGIME_PRODUCTION_RUN_LOCK_ARTIFACT_KIND = "regime_production_run_lock"

REGIME_PRODUCTION_LOCK_MODE_SANDBOX = "sandbox"
REGIME_PRODUCTION_LOCK_MODE_CANONICAL = "canonical"
REGIME_PRODUCTION_LOCK_MODES: tuple[str, ...] = (
    REGIME_PRODUCTION_LOCK_MODE_SANDBOX,
    REGIME_PRODUCTION_LOCK_MODE_CANONICAL,
)

REGIME_PRODUCTION_LOCK_STATUS_ACTIVE = "active"
REGIME_PRODUCTION_LOCK_STATUS_RELEASED = "released"
REGIME_PRODUCTION_LOCK_STATUS_FAILED_RECOVERABLE = "failed_recoverable"
REGIME_PRODUCTION_LOCK_STATUS_STALE_RECOVERED = "stale_recovered"
REGIME_PRODUCTION_BLOCKING_LOCK_STATUSES: tuple[str, ...] = (REGIME_PRODUCTION_LOCK_STATUS_ACTIVE,)

DEFAULT_REGIME_PRODUCTION_STALE_LOCK_SECONDS = 6 * 60 * 60


class RegimeProductionRunLockError(RuntimeError):
    """Raised when a Regime Production run lock would violate fail-closed concurrency."""


@dataclass(frozen=True)
class RegimeProductionRunLockTarget:
    branch: str
    output_root: str | Path
    range_start: str
    range_end: str
    mode: str
    writer_scope: str = "label_output"
    root_kind: str = "sandbox_output_root"

    def __post_init__(self) -> None:
        branch = _text(self.branch, field_name="branch")
        if branch not in REGIME_PRODUCTION_BRANCHES:
            raise ValueError(f"Unsupported Regime Production branch for run lock: {branch!r}")
        mode = _text(self.mode, field_name="mode").lower()
        if mode not in REGIME_PRODUCTION_LOCK_MODES:
            raise ValueError(f"Unsupported Regime Production lock mode: {self.mode!r}")
        range_start = _text(self.range_start, field_name="range_start")
        range_end = _text(self.range_end, field_name="range_end")
        if range_start > range_end:
            raise ValueError("Regime Production lock range_start must be <= range_end")
        output_root = Path(self.output_root)
        if not str(output_root).strip():
            raise ValueError("Regime Production lock output_root is required")
        root_kind = _text(self.root_kind, field_name="root_kind")
        if mode == REGIME_PRODUCTION_LOCK_MODE_CANONICAL and "canonical" not in root_kind:
            root_kind = "canonical_output_root"
        if mode == REGIME_PRODUCTION_LOCK_MODE_SANDBOX and "sandbox" not in root_kind:
            root_kind = "sandbox_output_root"
        object.__setattr__(self, "branch", branch)
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "range_start", range_start)
        object.__setattr__(self, "range_end", range_end)
        object.__setattr__(self, "writer_scope", _text(self.writer_scope, field_name="writer_scope"))
        object.__setattr__(self, "root_kind", root_kind)

    def normalized(self, *, project_root: str | Path | None = None) -> dict[str, Any]:
        resolved_root = resolve_project_path(self.output_root, project_root=project_root)
        root_hash = _fingerprint({"output_root": str(resolved_root)})
        payload = {
            "branch": self.branch,
            "mode": self.mode,
            "writer_scope": self.writer_scope,
            "root_kind": self.root_kind,
            "output_root": _portable_path_text(resolved_root),
            "output_root_hash": root_hash,
            "range_start": self.range_start,
            "range_end": self.range_end,
        }
        payload["target_fingerprint"] = _fingerprint(payload)
        return payload


@dataclass(frozen=True)
class RegimeProductionRunLockHandle:
    target: RegimeProductionRunLockTarget
    lock_path: Path
    run_id: str
    owner: str
    target_fingerprint: str
    acquired: bool
    payload: Mapping[str, Any]
    recovered_stale_lock: bool = False

    def as_dict(self) -> dict[str, Any]:
        payload = dict(self.payload or {})
        return to_jsonable(
            {
                "schema_version": REGIME_PRODUCTION_RUN_LOCK_SCHEMA_VERSION,
                "artifact_kind": "regime_production_run_lock_handle",
                "branch": self.target.branch,
                "mode": self.target.mode,
                "run_id": self.run_id,
                "owner": self.owner,
                "lock_path": _portable_path_text(self.lock_path),
                "target_fingerprint": self.target_fingerprint,
                "acquired": bool(self.acquired),
                "recovered_stale_lock": bool(self.recovered_stale_lock),
                "production_writer_enabled": bool(payload.get("production_writer_enabled")),
                "canonical_write_execution_allowed": bool(payload.get("canonical_write_execution_allowed")),
                "canonical_root_touched": bool(payload.get("canonical_root_touched")),
                "production_outputs_written": bool(payload.get("production_outputs_written")),
                "production_labels_written": bool(payload.get("production_labels_written")),
                "canonical_label_outputs_written": bool(payload.get("canonical_label_outputs_written")),
                "canonical_production_state_outputs_written": bool(payload.get("canonical_production_state_outputs_written")),
            }
        )


def regime_production_run_lock_path(
    target: RegimeProductionRunLockTarget,
    *,
    lock_root: str | Path,
    project_root: str | Path | None = None,
) -> Path:
    lock_root_path = resolve_project_path(lock_root, project_root=project_root)
    normalized = target.normalized(project_root=project_root)
    key = str(normalized["target_fingerprint"]).split(":", 1)[-1]
    return (
        lock_root_path
        / f"mode={target.mode}"
        / f"branch={target.branch}"
        / f"range_start={_safe_path_part(target.range_start)}"
        / f"range_end={_safe_path_part(target.range_end)}"
        / f"{key}.json"
    )


def acquire_regime_production_run_lock(
    target: RegimeProductionRunLockTarget,
    *,
    lock_root: str | Path,
    run_id: str,
    owner: str,
    stale_after_seconds: int = DEFAULT_REGIME_PRODUCTION_STALE_LOCK_SECONDS,
    now_ts: str | datetime | None = None,
    project_root: str | Path | None = None,
    recover_stale: bool = True,
    production_writer_enabled: bool = False,
    canonical_write_execution_allowed: bool = False,
    canonical_root_touched: bool = False,
    production_outputs_written: bool = False,
    production_labels_written: bool = False,
    canonical_label_outputs_written: bool = False,
) -> RegimeProductionRunLockHandle:
    run_id_text = _text(run_id, field_name="run_id")
    owner_text = _text(owner, field_name="owner")
    stale_seconds = int(stale_after_seconds)
    if stale_seconds <= 0:
        raise ValueError("Regime Production stale_after_seconds must be positive")
    now = _coerce_datetime(now_ts)
    lock_path = regime_production_run_lock_path(target, lock_root=lock_root, project_root=project_root)
    normalized = target.normalized(project_root=project_root)
    payload = _active_lock_payload(
        target,
        normalized=normalized,
        lock_path=lock_path,
        run_id=run_id_text,
        owner=owner_text,
        stale_after_seconds=stale_seconds,
        now=now,
        production_writer_enabled=production_writer_enabled,
        canonical_write_execution_allowed=canonical_write_execution_allowed,
        canonical_root_touched=canonical_root_touched,
        production_outputs_written=production_outputs_written,
        production_labels_written=production_labels_written,
        canonical_label_outputs_written=canonical_label_outputs_written,
    )
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        recovery_claim_path = _claim_recovery_gate(lock_path, owner=owner_text, run_id=run_id_text, now=now)
        existing = _load_json(lock_path)
        try:
            if _lock_blocks_target(existing, now=now):
                raise RegimeProductionRunLockError(
                    "Regime Production run lock exists for "
                    f"branch={target.branch} mode={target.mode} range={target.range_start}..{target.range_end} "
                    f"root_hash={normalized['output_root_hash']} run_id={existing.get('run_id')!r}"
                )
            if _active_lock_is_stale(existing, now=now) and not recover_stale:
                raise RegimeProductionRunLockError(
                    "Regime Production run lock is stale but stale recovery is disabled: "
                    f"{_portable_path_text(lock_path)}"
                )
            recovered_stale = _active_lock_is_stale(existing, now=now)
            replacement = {
                **payload,
                "replaced_existing_lock": True,
                "replaced_existing_lock_status": existing.get("status"),
                "recovered_stale_lock": bool(recovered_stale),
                "recovery_claim_path": _portable_path_text(recovery_claim_path),
                "previous_lock": to_jsonable(existing),
            }
            _write_json_atomic(lock_path, replacement)
            return RegimeProductionRunLockHandle(
                target=target,
                lock_path=lock_path,
                run_id=run_id_text,
                owner=owner_text,
                target_fingerprint=str(normalized["target_fingerprint"]),
                acquired=True,
                payload=replacement,
                recovered_stale_lock=bool(recovered_stale),
            )
        finally:
            _release_recovery_gate(recovery_claim_path)
    with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return RegimeProductionRunLockHandle(
        target=target,
        lock_path=lock_path,
        run_id=run_id_text,
        owner=owner_text,
        target_fingerprint=str(normalized["target_fingerprint"]),
        acquired=True,
        payload=payload,
        recovered_stale_lock=False,
    )


def release_regime_production_run_lock(
    handle: RegimeProductionRunLockHandle,
    *,
    status: str = REGIME_PRODUCTION_LOCK_STATUS_RELEASED,
    reason: str = "success",
    now_ts: str | datetime | None = None,
    production_writer_enabled: bool | None = None,
    canonical_write_execution_allowed: bool | None = None,
    canonical_root_touched: bool | None = None,
    production_outputs_written: bool | None = None,
    production_labels_written: bool | None = None,
    canonical_label_outputs_written: bool | None = None,
) -> dict[str, Any]:
    status_text = _text(status, field_name="status")
    if status_text not in {REGIME_PRODUCTION_LOCK_STATUS_RELEASED, REGIME_PRODUCTION_LOCK_STATUS_FAILED_RECOVERABLE}:
        raise ValueError("Regime Production lock release status must be released or failed_recoverable")
    lock_path = Path(handle.lock_path)
    existing = _load_json(lock_path)
    existing_run_id = str(existing.get("run_id") or "")
    if existing.get("status") == REGIME_PRODUCTION_LOCK_STATUS_ACTIVE and existing_run_id != handle.run_id:
        raise RegimeProductionRunLockError(
            f"Regime Production lock owner mismatch for release: {existing_run_id!r} != {handle.run_id!r}"
        )
    now = _coerce_datetime(now_ts)
    payload = {
        **existing,
        "status": status_text,
        "active": False,
        "recoverable": True,
        "release_reason": _text(reason, field_name="reason"),
        "released_at_ts": _format_ts(now),
        "blocking_finalizer_commit": False,
        "production_writer_enabled": _flag_or_existing(existing, "production_writer_enabled", production_writer_enabled),
        "canonical_write_execution_allowed": _flag_or_existing(
            existing,
            "canonical_write_execution_allowed",
            canonical_write_execution_allowed,
        ),
        "canonical_root_touched": _flag_or_existing(existing, "canonical_root_touched", canonical_root_touched),
        "production_outputs_written": _flag_or_existing(existing, "production_outputs_written", production_outputs_written),
        "production_labels_written": _flag_or_existing(existing, "production_labels_written", production_labels_written),
        "canonical_label_outputs_written": _flag_or_existing(
            existing,
            "canonical_label_outputs_written",
            canonical_label_outputs_written,
        ),
        "canonical_production_state_outputs_written": False,
    }
    _write_json_atomic(lock_path, payload)
    return to_jsonable(payload)


def regime_production_run_lock_is_stale(
    payload: Mapping[str, Any],
    *,
    now_ts: str | datetime | None = None,
) -> bool:
    return _active_lock_is_stale(payload, now=_coerce_datetime(now_ts))


def _active_lock_payload(
    target: RegimeProductionRunLockTarget,
    *,
    normalized: Mapping[str, Any],
    lock_path: Path,
    run_id: str,
    owner: str,
    stale_after_seconds: int,
    now: datetime,
    production_writer_enabled: bool,
    canonical_write_execution_allowed: bool,
    canonical_root_touched: bool,
    production_outputs_written: bool,
    production_labels_written: bool,
    canonical_label_outputs_written: bool,
) -> dict[str, Any]:
    return {
        "schema_version": REGIME_PRODUCTION_RUN_LOCK_SCHEMA_VERSION,
        "artifact_kind": REGIME_PRODUCTION_RUN_LOCK_ARTIFACT_KIND,
        "branch": target.branch,
        "mode": target.mode,
        "writer_scope": target.writer_scope,
        "root_kind": target.root_kind,
        "run_id": run_id,
        "owner": owner,
        "owner_pid": os.getpid(),
        "owner_host": socket.gethostname(),
        "lock_path": _portable_path_text(lock_path),
        "status": REGIME_PRODUCTION_LOCK_STATUS_ACTIVE,
        "active": True,
        "recoverable": False,
        "acquired_at_ts": _format_ts(now),
        "stale_after_seconds": int(stale_after_seconds),
        "expires_at_ts": _format_ts(_from_epoch(now.timestamp() + int(stale_after_seconds))),
        "stale_lock_recovery_policy": "active locks older than stale_after_seconds may be replaced and preserved as previous_lock",
        "target": to_jsonable(dict(normalized)),
        "target_fingerprint": normalized["target_fingerprint"],
        "output_root": normalized["output_root"],
        "output_root_hash": normalized["output_root_hash"],
        "range_start": target.range_start,
        "range_end": target.range_end,
        "parent_single_finalizer": True,
        "writer_workers": 1,
        "workers_write_outputs": False,
        "blocking_finalizer_commit": True,
        "sandbox_lock_separate_from_canonical_lock": True,
        "production_writer_enabled": bool(production_writer_enabled),
        "canonical_write_execution_allowed": bool(canonical_write_execution_allowed),
        "canonical_root_touched": bool(canonical_root_touched),
        "production_outputs_written": bool(production_outputs_written),
        "production_labels_written": bool(production_labels_written),
        "canonical_label_outputs_written": bool(canonical_label_outputs_written),
        "canonical_production_state_outputs_written": False,
        "production_promotion_performed": False,
        "test_branch_rerun_performed": False,
        "optuna_or_campaign_run_performed": False,
        "relationship_discovery_or_pairwise_run_performed": False,
    }


def _lock_blocks_target(payload: Mapping[str, Any], *, now: datetime) -> bool:
    status = str(payload.get("status") or "")
    if status not in REGIME_PRODUCTION_BLOCKING_LOCK_STATUSES:
        return False
    return not _active_lock_is_stale(payload, now=now)


def _active_lock_is_stale(payload: Mapping[str, Any], *, now: datetime) -> bool:
    status = str(payload.get("status") or "")
    if status != REGIME_PRODUCTION_LOCK_STATUS_ACTIVE:
        return False
    acquired_raw = payload.get("acquired_at_ts")
    stale_raw = payload.get("stale_after_seconds")
    if acquired_raw in (None, "") or stale_raw in (None, ""):
        return True
    try:
        acquired = _coerce_datetime(acquired_raw)
        stale_seconds = int(stale_raw)
    except Exception:
        return True
    return (now.timestamp() - acquired.timestamp()) >= stale_seconds


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = sibling_temp_path(path, suffix=".json.tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(to_jsonable(dict(payload)), handle, indent=2, sort_keys=True)
        handle.write("\n")
    atomic_replace(tmp, path)


def _claim_recovery_gate(lock_path: Path, *, owner: str, run_id: str, now: datetime) -> Path:
    claim_path = lock_path.with_name(f"{lock_path.name}.recovery_claim")
    payload = {
        "schema_version": REGIME_PRODUCTION_RUN_LOCK_SCHEMA_VERSION,
        "artifact_kind": "regime_production_run_lock_recovery_claim",
        "owner": owner,
        "run_id": run_id,
        "owner_pid": os.getpid(),
        "claimed_at_ts": _format_ts(now),
        "lock_path": _portable_path_text(lock_path),
        "production_writer_enabled": False,
        "canonical_write_execution_allowed": False,
    }
    while True:
        try:
            fd = os.open(str(claim_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            existing = _load_json(claim_path)
            if not _recovery_claim_is_stale(existing, now=now):
                raise RegimeProductionRunLockError(
                    f"Regime Production run lock recovery already active: {_portable_path_text(claim_path)}"
                )
            try:
                claim_path.unlink()
            except FileNotFoundError:
                pass
            continue
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        return claim_path


def _release_recovery_gate(claim_path: Path) -> None:
    try:
        claim_path.unlink()
    except FileNotFoundError:
        pass


def _flag_or_existing(existing: Mapping[str, Any], field: str, value: bool | None) -> bool:
    if value is None:
        return bool(existing.get(field))
    return bool(value)


def _recovery_claim_is_stale(payload: Mapping[str, Any], *, now: datetime) -> bool:
    claimed_raw = payload.get("claimed_at_ts")
    if claimed_raw in (None, ""):
        return True
    try:
        claimed = _coerce_datetime(claimed_raw)
    except Exception:
        return True
    return (now.timestamp() - claimed.timestamp()) >= DEFAULT_REGIME_PRODUCTION_STALE_LOCK_SECONDS


def _load_json(path: Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise RegimeProductionRunLockError(f"Regime Production run lock is not a JSON object: {path}")
    return payload


def _fingerprint(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(to_jsonable(dict(payload)), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _format_ts(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _coerce_datetime(value: str | datetime | None) -> datetime:
    if value is None:
        return datetime.now(timezone.utc)
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
    text = _text(value, field_name="timestamp")
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _from_epoch(value: float) -> datetime:
    return datetime.fromtimestamp(float(value), tz=timezone.utc)


def _safe_path_part(value: str) -> str:
    cleaned = []
    for char in str(value):
        cleaned.append(char if char.isalnum() or char in {"-", "_", "."} else "_")
    return "".join(cleaned).strip("._") or "value"


def _portable_path_text(value: str | Path | None) -> str | None:
    if value is None:
        return None
    path = Path(value)
    if not path.is_absolute():
        return str(path)
    resolved = path.resolve()
    cwd = Path.cwd().resolve()
    try:
        return str(resolved.relative_to(cwd))
    except ValueError:
        return f"<external_configured_root>/{resolved.name}"


def _text(value: object, *, field_name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"Regime Production run lock {field_name} is required")
    return text


__all__ = [
    "DEFAULT_REGIME_PRODUCTION_STALE_LOCK_SECONDS",
    "REGIME_PRODUCTION_LOCK_MODE_CANONICAL",
    "REGIME_PRODUCTION_LOCK_MODE_SANDBOX",
    "REGIME_PRODUCTION_LOCK_STATUS_ACTIVE",
    "REGIME_PRODUCTION_LOCK_STATUS_FAILED_RECOVERABLE",
    "REGIME_PRODUCTION_LOCK_STATUS_RELEASED",
    "REGIME_PRODUCTION_LOCK_STATUS_STALE_RECOVERED",
    "REGIME_PRODUCTION_RUN_LOCK_ARTIFACT_KIND",
    "REGIME_PRODUCTION_RUN_LOCK_SCHEMA_VERSION",
    "RegimeProductionRunLockError",
    "RegimeProductionRunLockHandle",
    "RegimeProductionRunLockTarget",
    "acquire_regime_production_run_lock",
    "regime_production_run_lock_is_stale",
    "regime_production_run_lock_path",
    "release_regime_production_run_lock",
]
