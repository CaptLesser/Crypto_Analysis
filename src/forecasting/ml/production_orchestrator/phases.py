from __future__ import annotations

from typing import Iterable, Optional


PHASES = ("startup", "load", "fit", "predict", "write", "finalize", "unknown")

_PATTERN_TO_PHASE = (
    ("[phase:start", "startup"),
    ("production-plan", "startup"),
    ("[plan]", "startup"),
    ("group-start", "load"),
    ("read_scalar", "load"),
    ("read_ohlc", "load"),
    ("load_total_s", "load"),
    ("[phase:dispatch]", "predict"),
    ("[dispatch]", "predict"),
    ("coldstart_fit", "fit"),
    ("fit_total_s", "fit"),
    ("refit", "fit"),
    ("predict_count", "predict"),
    ("wf_predict", "predict"),
    ("pred_", "predict"),
    ("[phase:writer-drain]", "write"),
    ("[phase:writer-stop]", "write"),
    ("parts_written", "write"),
    ("parquet_write", "write"),
    ("write_batch", "write"),
    ("[phase:validate]", "finalize"),
    ("[phase:manifest]", "finalize"),
    ("run complete", "finalize"),
    ("finished_at", "finalize"),
)


def infer_phase_from_text(lines: Iterable[str]) -> Optional[str]:
    latest: Optional[str] = None
    for raw_line in lines:
        line = str(raw_line).strip().lower()
        if not line:
            continue
        for pattern, phase in _PATTERN_TO_PHASE:
            if pattern in line:
                latest = str(phase)
    return latest
