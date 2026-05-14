from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence


UNKNOWN_LABEL = "unknown"
UNKNOWN_CLUSTER_ID = -1
NEUTRAL_LABEL_ID = 1


@dataclass(frozen=True)
class RegimeAxisContract:
    name: str
    labels: tuple[str, ...]
    neutral_label: str

    @property
    def label_to_id(self) -> Dict[str, int]:
        return {label: idx for idx, label in enumerate(self.labels)}

    @property
    def id_to_label(self) -> Dict[int, str]:
        return {idx: label for idx, label in enumerate(self.labels)}

    @property
    def cluster_column(self) -> str:
        return f"{self.name}_cluster_id"

    @property
    def label_column(self) -> str:
        return f"{self.name}_label"

    @property
    def confidence_column(self) -> str:
        return f"{self.name}_confidence_pct"

    @property
    def intensity_column(self) -> str:
        return f"{self.name}_intensity_pct"

    @property
    def output_columns(self) -> tuple[str, ...]:
        return (
            self.cluster_column,
            self.label_column,
            self.confidence_column,
            self.intensity_column,
        )


@dataclass(frozen=True)
class RegimeBandContract:
    name: str
    ceiling_interval_min: int
    member_intervals: tuple[int, ...]
    train_days: int

    @property
    def table_dir(self) -> str:
        return regime_table_dir(self.ceiling_interval_min)


@dataclass(frozen=True)
class RegimeAxisTarget:
    axis: str
    labels: tuple[str, ...]
    neutral_label: str
    unknown_label: str = UNKNOWN_LABEL
    unknown_id: int = NEUTRAL_LABEL_ID

    @property
    def label_column(self) -> str:
        return REGIME_AXES[self.axis].label_column

    @property
    def label_to_id(self) -> Dict[str, int]:
        return {label: idx for idx, label in enumerate(self.labels)}

    @property
    def id_to_label(self) -> Dict[int, str]:
        return {idx: label for idx, label in enumerate(self.labels)}

    def normalize_label(self, label: object, *, allow_unknown: bool = True) -> str:
        return normalize_axis_label(self.axis, label, allow_unknown=allow_unknown)

    def label_id(self, label: object) -> int:
        return axis_label_to_id(self.axis, label, unknown_id=self.unknown_id)

    def prediction_column(self, prefix: str, horizon_minutes: int) -> str:
        return forecast_prediction_column(prefix, self.axis, horizon_minutes)

    def prediction_id_column(self, prefix: str, horizon_minutes: int) -> str:
        return forecast_prediction_id_column(prefix, self.axis, horizon_minutes)

    def probability_columns(self, prefix: str, horizon_minutes: int) -> tuple[str, ...]:
        return forecast_probability_columns(prefix, self.axis, horizon_minutes)

    def target_available_column(self, prefix: str, horizon_minutes: int) -> str:
        return forecast_target_available_column(prefix, self.axis, horizon_minutes)

    def prediction_kind_column(self, prefix: str, horizon_minutes: int) -> str:
        return forecast_prediction_kind_column(prefix, self.axis, horizon_minutes)

    def output_columns(self, prefix: str, horizon_minutes: int) -> tuple[str, ...]:
        return forecast_axis_output_columns(prefix, self.axis, horizon_minutes)


REGIME_AXIS_ORDER: tuple[str, ...] = ("trend", "vol", "activity")
REGIME_AXES: Mapping[str, RegimeAxisContract] = {
    "trend": RegimeAxisContract(name="trend", labels=("down", "flat", "up"), neutral_label="flat"),
    "vol": RegimeAxisContract(name="vol", labels=("low", "normal", "high"), neutral_label="normal"),
    "activity": RegimeAxisContract(name="activity", labels=("low", "normal", "high"), neutral_label="normal"),
}

REGIME_AXIS_TARGETS: Mapping[str, RegimeAxisTarget] = {
    axis: RegimeAxisTarget(axis=axis, labels=contract.labels, neutral_label=contract.neutral_label)
    for axis, contract in REGIME_AXES.items()
}

REGIME_BANDS: Mapping[str, RegimeBandContract] = {
    "micro": RegimeBandContract(name="micro", ceiling_interval_min=30, member_intervals=(1, 5, 15, 30), train_days=30),
    "meso": RegimeBandContract(name="meso", ceiling_interval_min=240, member_intervals=(60, 240), train_days=180),
    "macro": RegimeBandContract(name="macro", ceiling_interval_min=1440, member_intervals=(720, 1440), train_days=360),
}

REGIME_CEILING_TO_BAND: Mapping[int, RegimeBandContract] = {
    band.ceiling_interval_min: band for band in REGIME_BANDS.values()
}

REGIME_INTERVAL_TO_BAND: Mapping[int, RegimeBandContract] = {
    interval: band for band in REGIME_BANDS.values() for interval in band.member_intervals
}

REGIME_CANONICAL_CEILING_INTERVALS: tuple[int, ...] = tuple(
    band.ceiling_interval_min for band in REGIME_BANDS.values()
)
REGIME_DEFAULT_FORECAST_INTERVALS: tuple[int, ...] = REGIME_CANONICAL_CEILING_INTERVALS

BASE_LABEL_OUTPUT_COLUMNS: tuple[str, ...] = (
    "ts",
    "asset",
    "band",
    "ceiling_interval_min",
)

LABEL_DIAGNOSTIC_COLUMNS: tuple[str, ...] = (
    "feature_schema_hash",
)

REQUIRED_LABEL_COLUMNS: tuple[str, ...] = (
    *BASE_LABEL_OUTPUT_COLUMNS,
    *(column for axis_name in REGIME_AXIS_ORDER for column in REGIME_AXES[axis_name].output_columns),
    *LABEL_DIAGNOSTIC_COLUMNS,
)


def regime_table_dir(ceiling_interval_min: int) -> str:
    return f"regimes_{int(ceiling_interval_min)}"


def regime_label_month_dir(root: Path, ceiling_interval_min: int, asset: str, year: int, month: int) -> Path:
    return (
        Path(root)
        / regime_table_dir(int(ceiling_interval_min))
        / f"asset={str(asset)}"
        / f"year={int(year)}"
        / f"month={int(month):02d}"
    )


def regime_forecast_month_dir(root: Path, interval_min: int, asset: str, year: int, month: int) -> Path:
    return (
        Path(root)
        / f"{int(interval_min)}"
        / f"asset={str(asset)}"
        / f"year={int(year)}"
        / f"month={int(month):02d}"
    )


def regime_forecast_part_path(root: Path, interval_min: int, asset: str, year: int, month: int) -> Path:
    return regime_forecast_month_dir(root, interval_min, asset, year, month) / "part-000.parquet"


def band_for_ceiling(ceiling_interval_min: int) -> RegimeBandContract:
    try:
        return REGIME_CEILING_TO_BAND[int(ceiling_interval_min)]
    except KeyError as exc:
        valid = ", ".join(str(k) for k in sorted(REGIME_CEILING_TO_BAND))
        raise ValueError(f"Unsupported Regime ceiling interval {int(ceiling_interval_min)}; expected one of: {valid}") from exc


def band_for_member_interval(interval_min: int) -> RegimeBandContract:
    try:
        return REGIME_INTERVAL_TO_BAND[int(interval_min)]
    except KeyError as exc:
        valid = ", ".join(str(k) for k in sorted(REGIME_INTERVAL_TO_BAND))
        raise ValueError(f"Unsupported Regime member interval {int(interval_min)}; expected one of: {valid}") from exc


def forecast_ceiling_interval(interval_min: int) -> int:
    return int(band_for_member_interval(int(interval_min)).ceiling_interval_min)


def is_canonical_ceiling_interval(interval_min: int) -> bool:
    return int(interval_min) in REGIME_CEILING_TO_BAND


def normalize_axis_label(axis: str, label: object, *, allow_unknown: bool = True) -> str:
    axis_contract = REGIME_AXES[str(axis)]
    value = str(label).strip().lower()
    if value in axis_contract.label_to_id:
        return value
    if allow_unknown and value == UNKNOWN_LABEL:
        return UNKNOWN_LABEL
    raise ValueError(f"Unsupported {axis} Regime label: {label!r}")


def axis_label_to_id(axis: str, label: object, *, unknown_id: int = NEUTRAL_LABEL_ID) -> int:
    normalized = normalize_axis_label(axis, label, allow_unknown=True)
    if normalized == UNKNOWN_LABEL:
        return int(unknown_id)
    return int(REGIME_AXES[str(axis)].label_to_id[normalized])


def axis_id_to_label(axis: str, label_id: int, *, unknown_label: str = UNKNOWN_LABEL) -> str:
    return REGIME_AXES[str(axis)].id_to_label.get(int(label_id), str(unknown_label))


def required_label_columns(columns: Iterable[str] = ()) -> tuple[str, ...]:
    requested = tuple(str(c) for c in columns)
    if not requested:
        return REQUIRED_LABEL_COLUMNS
    required = {"ts", "asset", *requested}
    return tuple(c for c in REQUIRED_LABEL_COLUMNS if c in required)


def all_label_output_columns() -> tuple[str, ...]:
    return REQUIRED_LABEL_COLUMNS


def axis_label_columns(axes: Sequence[str] = REGIME_AXIS_ORDER) -> tuple[str, ...]:
    return tuple(REGIME_AXES[str(axis)].label_column for axis in axes)


def axis_target(axis: str) -> RegimeAxisTarget:
    try:
        return REGIME_AXIS_TARGETS[str(axis)]
    except KeyError as exc:
        valid = ", ".join(REGIME_AXIS_ORDER)
        raise ValueError(f"Unsupported Regime axis target {axis!r}; expected one of: {valid}") from exc


def forecast_prediction_column(prefix: str, axis: str, horizon_minutes: int) -> str:
    axis_target(axis)
    return f"{str(prefix)}_pred_{str(axis)}_{int(horizon_minutes)}m"


def forecast_prediction_id_column(prefix: str, axis: str, horizon_minutes: int) -> str:
    axis_target(axis)
    return f"{str(prefix)}_pred_{str(axis)}_id_{int(horizon_minutes)}m"


def forecast_probability_columns(prefix: str, axis: str, horizon_minutes: int) -> tuple[str, ...]:
    target = axis_target(axis)
    return tuple(f"{str(prefix)}_prob_{target.axis}_{label}_{int(horizon_minutes)}m" for label in target.labels)


def forecast_target_available_column(prefix: str, axis: str, horizon_minutes: int) -> str:
    axis_target(axis)
    return f"{str(prefix)}_target_available_{str(axis)}_{int(horizon_minutes)}m"


def forecast_prediction_kind_column(prefix: str, axis: str, horizon_minutes: int) -> str:
    axis_target(axis)
    return f"{str(prefix)}_prediction_kind_{str(axis)}_{int(horizon_minutes)}m"


def forecast_axis_output_columns(prefix: str, axis: str, horizon_minutes: int) -> tuple[str, ...]:
    return (
        forecast_prediction_column(prefix, axis, horizon_minutes),
        forecast_prediction_id_column(prefix, axis, horizon_minutes),
        *forecast_probability_columns(prefix, axis, horizon_minutes),
        forecast_target_available_column(prefix, axis, horizon_minutes),
        forecast_prediction_kind_column(prefix, axis, horizon_minutes),
    )


def forecast_output_columns(
    prefix: str,
    horizon_minutes: int,
    axes: Sequence[str] = REGIME_AXIS_ORDER,
) -> tuple[str, ...]:
    return tuple(
        column
        for axis in axes
        for column in forecast_axis_output_columns(prefix, str(axis), int(horizon_minutes))
    )
