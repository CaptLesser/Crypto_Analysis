from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.regimes.core.serialization import to_jsonable


REGIME_WINDOW_PROFILE_SCHEMA_VERSION = 1
SECONDS_PER_DAY = 86_400


@dataclass(frozen=True)
class RegimeWindowProfile:
    window_profile_id: str
    band: str
    lookback_days: int | None
    source_tail_anchor: bool = True
    row_cap: int | None = None
    partition_cap: int | None = None
    start_ts: int | None = None
    end_ts: int | None = None
    source_tail_ts: int | None = None
    schema_version: int = REGIME_WINDOW_PROFILE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        profile_id = str(self.window_profile_id).strip()
        band = str(self.band).strip().lower()
        if not profile_id:
            raise ValueError("Regime window_profile_id must be non-empty")
        if not band:
            raise ValueError("Regime window band must be non-empty")
        lookback = None if self.lookback_days is None else int(self.lookback_days)
        if lookback is not None and lookback <= 0:
            raise ValueError("Regime window lookback_days must be positive when provided")
        row_cap = None if self.row_cap is None else int(self.row_cap)
        partition_cap = None if self.partition_cap is None else int(self.partition_cap)
        if row_cap is not None and row_cap <= 0:
            raise ValueError("Regime window row_cap must be positive when provided")
        if partition_cap is not None and partition_cap <= 0:
            raise ValueError("Regime window partition_cap must be positive when provided")
        start_ts = None if self.start_ts is None else int(self.start_ts)
        end_ts = None if self.end_ts is None else int(self.end_ts)
        source_tail_ts = None if self.source_tail_ts is None else int(self.source_tail_ts)
        if start_ts is not None and end_ts is not None and start_ts > end_ts:
            raise ValueError("Regime window start_ts cannot be after end_ts")
        object.__setattr__(self, "window_profile_id", profile_id)
        object.__setattr__(self, "band", band)
        object.__setattr__(self, "lookback_days", lookback)
        object.__setattr__(self, "row_cap", row_cap)
        object.__setattr__(self, "partition_cap", partition_cap)
        object.__setattr__(self, "start_ts", start_ts)
        object.__setattr__(self, "end_ts", end_ts)
        object.__setattr__(self, "source_tail_ts", source_tail_ts)
        object.__setattr__(self, "schema_version", int(self.schema_version))

    def resolve(self, *, source_tail_ts: int | None = None) -> "RegimeWindowProfile":
        tail = self.source_tail_ts if self.source_tail_ts is not None else source_tail_ts
        if tail is None:
            return self
        tail_int = int(tail)
        start = self.start_ts
        end = self.end_ts
        if self.source_tail_anchor:
            end = tail_int if end is None else min(int(end), tail_int)
            if self.lookback_days is not None and start is None:
                start = int(end) - int(self.lookback_days) * SECONDS_PER_DAY
        return RegimeWindowProfile(
            window_profile_id=self.window_profile_id,
            band=self.band,
            lookback_days=self.lookback_days,
            source_tail_anchor=self.source_tail_anchor,
            row_cap=self.row_cap,
            partition_cap=self.partition_cap,
            start_ts=start,
            end_ts=end,
            source_tail_ts=tail_int,
            schema_version=self.schema_version,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "window_profile_id": self.window_profile_id,
            "band": self.band,
            "lookback_days": self.lookback_days,
            "source_tail_anchor": bool(self.source_tail_anchor),
            "row_cap": self.row_cap,
            "partition_cap": self.partition_cap,
            "source_tail_ts": self.source_tail_ts,
            "start_ts": self.start_ts,
            "end_ts": self.end_ts,
        }


def coerce_window_profile(value: RegimeWindowProfile | dict[str, Any]) -> RegimeWindowProfile:
    if isinstance(value, RegimeWindowProfile):
        return value
    return RegimeWindowProfile(**dict(value))


def window_profile_rows(profiles: tuple[RegimeWindowProfile, ...]) -> tuple[dict[str, Any], ...]:
    return tuple(to_jsonable(profile.as_dict()) for profile in profiles)


__all__ = [
    "REGIME_WINDOW_PROFILE_SCHEMA_VERSION",
    "SECONDS_PER_DAY",
    "RegimeWindowProfile",
    "coerce_window_profile",
    "window_profile_rows",
]
