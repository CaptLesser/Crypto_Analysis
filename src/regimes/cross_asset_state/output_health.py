from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


HEALTH_FAILURE_INSUFFICIENT_ROWS = "insufficient_valid_rows"
HEALTH_FAILURE_NONFINITE_VALUES = "nonfinite_values"
HEALTH_FAILURE_ALL_MASK_OUTPUT = "all_mask_output"
HEALTH_FAILURE_ALL_ONE_STATE_COLLAPSE = "all_one_state_collapse"
HEALTH_FAILURE_INSUFFICIENT_STATES = "insufficient_state_count"
HEALTH_FAILURE_DOMINANT_STATE = "dominant_state_failure"
HEALTH_FAILURE_TINY_STATE = "tiny_state_failure"
HEALTH_FAILURE_SHAPE_NOT_PRESERVED = "shape_not_preserved"
HEALTH_WARNING_BIRCH_TOO_FEW_SUBCLUSTERS = "birch_too_few_subclusters"


@dataclass(frozen=True)
class CrossAssetStateOutputHealth:
    row_count: int
    valid_row_count: int
    state_count: int
    dominant_state_share: float
    tiny_state_count: int
    nonfinite_count: int = 0
    shape_preserved: bool = True
    min_rows: int = 8
    min_states: int = 2
    max_dominant_state_share: float = 0.95
    min_tiny_state_share: float = 0.03
    failure_reasons: tuple[str, ...] = ()
    warning_reasons: tuple[str, ...] = ()

    @property
    def passed(self) -> bool:
        return (
            self.shape_preserved
            and self.valid_row_count >= self.min_rows
            and self.state_count >= self.min_states
            and self.nonfinite_count == 0
            and self.tiny_state_count == 0
            and self.dominant_state_share <= self.max_dominant_state_share
        )

    @property
    def failure_type(self) -> str | None:
        return self.failure_reasons[0] if self.failure_reasons else None

    def as_dict(self) -> dict[str, Any]:
        return {
            "row_count": int(self.row_count),
            "valid_row_count": int(self.valid_row_count),
            "state_count": int(self.state_count),
            "dominant_state_share": round(float(self.dominant_state_share), 6),
            "tiny_state_count": int(self.tiny_state_count),
            "nonfinite_count": int(self.nonfinite_count),
            "shape_preserved": bool(self.shape_preserved),
            "passed": bool(self.passed),
            "failure_reasons": list(self.failure_reasons),
            "warning_reasons": list(self.warning_reasons),
            "failure_type": self.failure_type,
            "max_dominant_state_share": round(float(self.max_dominant_state_share), 6),
            "min_tiny_state_share": round(float(self.min_tiny_state_share), 6),
            "min_rows": int(self.min_rows),
            "min_states": int(self.min_states),
        }


def evaluate_output_health(
    labels: list[str],
    *,
    row_count: int,
    nonfinite_count: int = 0,
    shape_preserved: bool = True,
    warning_reasons: Sequence[str] = (),
) -> CrossAssetStateOutputHealth:
    counts: dict[str, int] = {}
    for label in labels:
        counts[str(label)] = counts.get(str(label), 0) + 1
    valid = len(labels)
    dominant = max(counts.values()) / valid if valid and counts else 1.0
    tiny = sum(1 for count in counts.values() if valid and count / valid < 0.03)
    failure_reasons = _failure_reasons(
        row_count=row_count,
        valid_row_count=valid,
        state_count=len(counts),
        dominant_state_share=dominant,
        tiny_state_count=tiny,
        nonfinite_count=nonfinite_count,
        shape_preserved=shape_preserved,
    )
    return CrossAssetStateOutputHealth(
        row_count=int(row_count),
        valid_row_count=int(valid),
        state_count=len(counts),
        dominant_state_share=dominant,
        tiny_state_count=tiny,
        nonfinite_count=int(nonfinite_count),
        shape_preserved=shape_preserved,
        failure_reasons=tuple(failure_reasons),
        warning_reasons=tuple(dict.fromkeys(str(reason) for reason in warning_reasons if str(reason).strip())),
    )


def label_counts(labels: list[str]) -> Mapping[str, int]:
    out: dict[str, int] = {}
    for label in labels:
        key = str(label)
        out[key] = out.get(key, 0) + 1
    return dict(sorted(out.items()))


def _failure_reasons(
    *,
    row_count: int,
    valid_row_count: int,
    state_count: int,
    dominant_state_share: float,
    tiny_state_count: int,
    nonfinite_count: int,
    shape_preserved: bool,
) -> tuple[str, ...]:
    reasons: list[str] = []
    if not shape_preserved:
        reasons.append(HEALTH_FAILURE_SHAPE_NOT_PRESERVED)
    if valid_row_count < 8:
        reasons.append(HEALTH_FAILURE_INSUFFICIENT_ROWS)
    if nonfinite_count:
        reasons.append(HEALTH_FAILURE_NONFINITE_VALUES)
    if row_count and valid_row_count == 0:
        reasons.append(HEALTH_FAILURE_ALL_MASK_OUTPUT)
    if valid_row_count and state_count == 1:
        reasons.append(HEALTH_FAILURE_ALL_ONE_STATE_COLLAPSE)
    elif state_count < 2:
        reasons.append(HEALTH_FAILURE_INSUFFICIENT_STATES)
    if dominant_state_share > 0.95:
        reasons.append(HEALTH_FAILURE_DOMINANT_STATE)
    if tiny_state_count:
        reasons.append(HEALTH_FAILURE_TINY_STATE)
    return tuple(dict.fromkeys(reasons))
