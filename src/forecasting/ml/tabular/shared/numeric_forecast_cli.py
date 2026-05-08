from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple


def parse_train_window_months(raw: str, *, error_prefix: str) -> Optional[int]:
    value = str(raw).strip()
    if not value:
        return None
    if value.lower().endswith("m"):
        value = value[:-1].strip()
    try:
        months = int(value)
    except Exception:
        raise SystemExit(f"{error_prefix} invalid train window months: {raw}")
    if months <= 0:
        raise SystemExit(f"{error_prefix} invalid train window months: {raw}")
    return int(months)


def parse_refit_cadence(
    raw: str,
    *,
    normalize_refit_cadence: Callable[[str], str],
    error_prefix: str,
) -> Optional[str]:
    value = str(raw).strip()
    if not value:
        return None
    try:
        return normalize_refit_cadence(value)
    except ValueError:
        raise SystemExit(f"{error_prefix} unsupported refit cadence: {raw}")


def parse_requested_tasks(raw: str, *, numeric_tasks: Sequence[str], error_prefix: str) -> List[str]:
    requested = [x.strip() for x in str(raw).split(",") if x.strip()]
    if not requested:
        return list(numeric_tasks)
    unsupported = sorted(set(requested) - set(numeric_tasks))
    if unsupported:
        raise SystemExit(f"{error_prefix} unsupported tasks requested: " + ",".join(unsupported) + " | supported=" + ",".join(numeric_tasks))
    return requested


def parse_combo_list(
    raw: str,
    *,
    numeric_tasks: Sequence[str],
    active_task_horizon_matrix: Dict[str, Sequence[int]],
    error_prefix: str,
) -> List[Tuple[int, int, str]]:
    requested = [token.strip() for token in str(raw).split(",") if token.strip()]
    out: List[Tuple[int, int, str]] = []
    for token in requested:
        parts = token.split(":")
        if len(parts) != 3:
            raise SystemExit(f"{error_prefix} invalid combo token: {token}")
        try:
            interval = int(parts[0])
            horizon = int(parts[1])
        except Exception:
            raise SystemExit(f"{error_prefix} invalid combo token: {token}")
        task = parts[2].strip()
        if task not in numeric_tasks:
            raise SystemExit(f"{error_prefix} unsupported combo task: {task}")
        supported = set(active_task_horizon_matrix.get(task, []))
        if int(horizon) not in supported:
            raise SystemExit(f"{error_prefix} unsupported combo task/horizon: {task}:{int(horizon)}m")
        out.append((int(interval), int(horizon), str(task)))
    return out


def parse_combo_window_list(
    raw: str,
    *,
    parse_combo_list_fn: Callable[[str], List[Tuple[int, int, str]]],
    parse_train_window_months_fn: Callable[[str], Optional[int]],
    error_prefix: str,
) -> List[Tuple[int, int, str, int]]:
    requested = [token.strip() for token in str(raw).split(",") if token.strip()]
    out: List[Tuple[int, int, str, int]] = []
    for token in requested:
        combo_part, sep, months_part = token.partition("@")
        if not sep:
            raise SystemExit(f"{error_prefix} invalid combo-window token: {token}")
        combo = parse_combo_list_fn(combo_part)
        if len(combo) != 1:
            raise SystemExit(f"{error_prefix} invalid combo-window token: {token}")
        months = parse_train_window_months_fn(months_part)
        if months is None:
            raise SystemExit(f"{error_prefix} invalid combo-window token: {token}")
        interval, horizon, task = combo[0]
        out.append((int(interval), int(horizon), str(task), int(months)))
    return out


def parse_combo_profile_list(
    raw: str,
    *,
    parse_combo_list_fn: Callable[[str], List[Tuple[int, int, str]]],
    parse_train_window_months_fn: Callable[[str], Optional[int]],
    parse_refit_cadence_fn: Callable[[str], Optional[str]],
    error_prefix: str,
) -> List[Tuple[int, int, str, int, str]]:
    requested = [token.strip() for token in str(raw).split(",") if token.strip()]
    out: List[Tuple[int, int, str, int, str]] = []
    for token in requested:
        parts = [part.strip() for part in token.split("@")]
        if len(parts) != 3:
            raise SystemExit(f"{error_prefix} invalid combo-profile token: {token}")
        combo = parse_combo_list_fn(parts[0])
        if len(combo) != 1:
            raise SystemExit(f"{error_prefix} invalid combo-profile token: {token}")
        months = parse_train_window_months_fn(parts[1])
        cadence = parse_refit_cadence_fn(parts[2])
        if months is None or cadence is None:
            raise SystemExit(f"{error_prefix} invalid combo-profile token: {token}")
        interval, horizon, task = combo[0]
        out.append((int(interval), int(horizon), str(task), int(months), str(cadence)))
    return out


def validate_requested_task_horizon_pairs(
    tasks: Sequence[str],
    horizon_minutes: Sequence[int],
    *,
    active_task_horizon_matrix: Dict[str, Sequence[int]],
    error_prefix: str,
) -> None:
    unsupported: List[str] = []
    for task in tasks:
        supported = set(active_task_horizon_matrix.get(str(task), []))
        for hm in horizon_minutes:
            if int(hm) not in supported:
                unsupported.append(f"{task}:{int(hm)}m")
    if unsupported:
        raise SystemExit(f"{error_prefix} unsupported task/horizon pairs requested: " + ",".join(sorted(unsupported)))


def horizons_for_interval(
    interval: int,
    horizon_minutes_list: Sequence[int],
    *,
    horizon_bars_from_minutes: Callable[[int, int], int],
) -> List[Tuple[int, int]]:
    out: List[Tuple[int, int]] = []
    for hm in horizon_minutes_list:
        hb = horizon_bars_from_minutes(int(hm), int(interval))
        out.append((int(hm), int(hb)))
    return sorted(set(out), key=lambda x: x[0])
