from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from src.forecasting.common.io_atomic import atomic_replace, sibling_temp_path


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def utc_now_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def write_json_atomic(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = sibling_temp_path(path)
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    atomic_replace(tmp, path)


def load_json_dict(path: Path) -> Dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def command_signature(payload: Dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return f"sha256:{hashlib.sha256(raw).hexdigest()}"


def ensure_serializable(value: Any) -> Any:
    if is_dataclass(value):
        return ensure_serializable(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): ensure_serializable(inner) for key, inner in value.items()}
    if isinstance(value, (list, tuple)):
        return [ensure_serializable(inner) for inner in value]
    return value


def remove_tree(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)


def remove_file(path: Path) -> None:
    if path.exists():
        path.unlink()


def resolve_output_dir(project_root: Path, output_dir: Path) -> Path:
    return output_dir.resolve() if output_dir.is_absolute() else (project_root / output_dir).resolve()


def latest_incomplete_run(base_output_dir: Path) -> Optional[Path]:
    if not base_output_dir.exists():
        return None
    candidates = sorted((path for path in base_output_dir.glob("run=*") if path.is_dir()), key=lambda item: item.name)
    for path in reversed(candidates):
        manifest = load_json_dict(path / "orchestrator_run_manifest.json")
        if str(manifest.get("status", "")).strip().lower() not in {"completed", "failed"}:
            return path.resolve()
    return None


def resolve_run_root(*, project_root: Path, output_dir: Path, run_id: str, resume_run: str, no_resume_latest: bool) -> Path:
    base_output_dir = resolve_output_dir(project_root, output_dir)
    if str(resume_run).strip():
        run_name = str(resume_run).strip()
        if not run_name.startswith("run="):
            run_name = f"run={run_name}"
        run_root = (base_output_dir / run_name).resolve()
        if not run_root.exists():
            raise RuntimeError(f"Requested resume run does not exist: {run_root}")
        return run_root
    if str(run_id).strip():
        run_name = str(run_id).strip()
        if not run_name.startswith("run="):
            run_name = f"run={run_name}"
        return (base_output_dir / run_name).resolve()
    if not bool(no_resume_latest):
        latest = latest_incomplete_run(base_output_dir)
        if latest is not None:
            return latest
    return (base_output_dir / f"run={utc_now_stamp()}").resolve()


def maybe_git_head(project_root: Path) -> Optional[str]:
    head_path = project_root / ".git" / "HEAD"
    if not head_path.exists():
        return None
    try:
        raw = head_path.read_text(encoding="utf-8").strip()
    except Exception:
        return None
    if raw.startswith("ref:"):
        ref = raw.split(":", 1)[1].strip()
        ref_path = project_root / ".git" / Path(ref)
        try:
            return ref_path.read_text(encoding="utf-8").strip() or None
        except Exception:
            return None
    return raw or None


def append_log_line(log_path: Path, message: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"[{utc_now_iso()}] {message}\n")


def latest_mtime_under_roots(roots: Iterable[Path], *, max_entries: int = 5000) -> Optional[float]:
    latest: Optional[float] = None
    seen = 0
    stack = [Path(root) for root in roots if Path(root).exists()]
    while stack:
        current = stack.pop()
        try:
            stat = current.stat()
            latest = float(stat.st_mtime) if latest is None else max(float(latest), float(stat.st_mtime))
        except Exception:
            pass
        if current.is_dir():
            try:
                children = list(current.iterdir())
            except Exception:
                continue
            for child in children:
                if seen >= max_entries:
                    return latest
                seen += 1
                stack.append(child)
    return latest
