from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, is_dataclass
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import Any, Mapping


def to_jsonable(value: Any) -> Any:
    if hasattr(value, "as_dict"):
        return to_jsonable(value.as_dict())
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value) and not isinstance(value, type):
        return to_jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [to_jsonable(item) for item in value]
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    if isinstance(value, set):
        return sorted(to_jsonable(item) for item in value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    if isinstance(value, BaseException):
        return str(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return str(value)


def dumps_json(value: Any, **kwargs: Any) -> str:
    options = {"indent": 2, "sort_keys": True}
    options.update(kwargs)
    return json.dumps(to_jsonable(value), **options)


def loads_json(text: str) -> Any:
    if not str(text).strip():
        raise ValueError("Regime JSON payload must be non-empty")
    return json.loads(text)


def require_json_object(payload: Any, *, context: str) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise ValueError(f"{context} must be a JSON object")
    return dict(payload)


def require_known_fields(
    payload: Mapping[str, Any],
    *,
    required: set[str],
    optional: set[str],
    context: str,
) -> dict[str, Any]:
    obj = require_json_object(payload, context=context)
    missing = sorted(required.difference(obj))
    if missing:
        raise ValueError(f"{context} missing required fields: {', '.join(missing)}")
    unknown = sorted(set(obj).difference(required).difference(optional))
    if unknown:
        raise ValueError(f"{context} has unexpected fields: {', '.join(unknown)}")
    return obj


__all__ = [
    "dumps_json",
    "loads_json",
    "require_json_object",
    "require_known_fields",
    "to_jsonable",
]
