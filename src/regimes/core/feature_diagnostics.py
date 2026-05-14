from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from src.regimes.core.contracts import CANONICAL_SCHEMA_VERSION, require_schema_version
from src.regimes.core.serialization import dumps_json, loads_json, require_known_fields, require_json_object, to_jsonable


@dataclass(frozen=True)
class FeatureDropDiagnostic:
    column: str
    reason: str
    metric_name: str | None = None
    metric_value: float | None = None
    reference_column: str | None = None

    def __post_init__(self) -> None:
        column = str(self.column).strip()
        reason = str(self.reason).strip()
        if not column:
            raise ValueError("Regime feature drop diagnostic column must be non-empty")
        if not reason:
            raise ValueError("Regime feature drop diagnostic reason must be non-empty")
        object.__setattr__(self, "column", column)
        object.__setattr__(self, "reason", reason)
        if self.metric_value is not None:
            object.__setattr__(self, "metric_value", float(self.metric_value))
        if self.metric_name is not None:
            object.__setattr__(self, "metric_name", str(self.metric_name).strip() or None)
        if self.reference_column is not None:
            object.__setattr__(self, "reference_column", str(self.reference_column).strip() or None)

    def as_dict(self) -> dict[str, Any]:
        return {
            "column": self.column,
            "reason": self.reason,
            "metric_name": self.metric_name,
            "metric_value": self.metric_value,
            "reference_column": self.reference_column,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FeatureDropDiagnostic":
        obj = require_known_fields(
            payload,
            required={"column", "reason"},
            optional={"metric_name", "metric_value", "reference_column"},
            context="Regime FeatureDropDiagnostic",
        )
        return cls(
            column=obj["column"],
            reason=obj["reason"],
            metric_name=obj.get("metric_name"),
            metric_value=obj.get("metric_value"),
            reference_column=obj.get("reference_column"),
        )


@dataclass(frozen=True)
class FeatureSelectionDiagnostics:
    input_columns: Sequence[str]
    retained_columns: Sequence[str]
    dropped_features: Sequence[FeatureDropDiagnostic | Mapping[str, Any]]
    before_shape: Sequence[int]
    after_shape: Sequence[int]
    missingness_summary: Mapping[str, Any] = field(default_factory=dict)
    variance_summary: Mapping[str, Any] = field(default_factory=dict)
    zero_share_summary: Mapping[str, Any] = field(default_factory=dict)
    duplicate_summary: Mapping[str, Any] = field(default_factory=dict)
    correlation_summary: Mapping[str, Any] = field(default_factory=dict)
    config: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = CANONICAL_SCHEMA_VERSION

    def __post_init__(self) -> None:
        schema_version = require_schema_version(self.schema_version)
        input_columns = tuple(str(column).strip() for column in self.input_columns if str(column).strip())
        retained_columns = tuple(str(column).strip() for column in self.retained_columns if str(column).strip())
        if not input_columns:
            raise ValueError("Regime feature selection diagnostics input_columns must be non-empty")
        before_shape = tuple(int(value) for value in self.before_shape)
        after_shape = tuple(int(value) for value in self.after_shape)
        if len(before_shape) != 2 or len(after_shape) != 2:
            raise ValueError("Regime feature selection diagnostics shapes must be two-element sequences")
        dropped = tuple(
            item if isinstance(item, FeatureDropDiagnostic) else FeatureDropDiagnostic.from_dict(item)
            for item in self.dropped_features
        )
        dropped_columns = {record.column for record in dropped}
        retained_set = set(retained_columns)
        overlap = retained_set.intersection(dropped_columns)
        if overlap:
            raise ValueError(f"Regime feature selection diagnostics retained and dropped overlap: {sorted(overlap)[0]}")
        object.__setattr__(self, "schema_version", schema_version)
        object.__setattr__(self, "input_columns", input_columns)
        object.__setattr__(self, "retained_columns", retained_columns)
        object.__setattr__(self, "dropped_features", dropped)
        object.__setattr__(self, "before_shape", before_shape)
        object.__setattr__(self, "after_shape", after_shape)
        object.__setattr__(self, "missingness_summary", dict(self.missingness_summary))
        object.__setattr__(self, "variance_summary", dict(self.variance_summary))
        object.__setattr__(self, "zero_share_summary", dict(self.zero_share_summary))
        object.__setattr__(self, "duplicate_summary", dict(self.duplicate_summary))
        object.__setattr__(self, "correlation_summary", dict(self.correlation_summary))
        object.__setattr__(self, "config", dict(self.config))

    @property
    def dropped_columns(self) -> tuple[str, ...]:
        return tuple(record.column for record in self.dropped_features)

    @property
    def drop_reasons_by_column(self) -> dict[str, str]:
        return {record.column: record.reason for record in self.dropped_features}

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "input_columns": list(self.input_columns),
            "retained_columns": list(self.retained_columns),
            "dropped_columns": list(self.dropped_columns),
            "dropped_features": [record.as_dict() for record in self.dropped_features],
            "before_shape": list(self.before_shape),
            "after_shape": list(self.after_shape),
            "missingness_summary": to_jsonable(self.missingness_summary),
            "variance_summary": to_jsonable(self.variance_summary),
            "zero_share_summary": to_jsonable(self.zero_share_summary),
            "duplicate_summary": to_jsonable(self.duplicate_summary),
            "correlation_summary": to_jsonable(self.correlation_summary),
            "config": to_jsonable(self.config),
        }

    def to_json(self, **kwargs: Any) -> str:
        return dumps_json(self.as_dict(), **kwargs)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FeatureSelectionDiagnostics":
        obj = require_known_fields(
            payload,
            required={
                "schema_version",
                "input_columns",
                "retained_columns",
                "dropped_features",
                "before_shape",
                "after_shape",
            },
            optional={
                "dropped_columns",
                "missingness_summary",
                "variance_summary",
                "zero_share_summary",
                "duplicate_summary",
                "correlation_summary",
                "config",
            },
            context="Regime FeatureSelectionDiagnostics",
        )
        return cls(
            schema_version=obj["schema_version"],
            input_columns=obj["input_columns"],
            retained_columns=obj["retained_columns"],
            dropped_features=obj["dropped_features"],
            before_shape=obj["before_shape"],
            after_shape=obj["after_shape"],
            missingness_summary=obj.get("missingness_summary", {}),
            variance_summary=obj.get("variance_summary", {}),
            zero_share_summary=obj.get("zero_share_summary", {}),
            duplicate_summary=obj.get("duplicate_summary", {}),
            correlation_summary=obj.get("correlation_summary", {}),
            config=obj.get("config", {}),
        )

    @classmethod
    def from_json(cls, text: str) -> "FeatureSelectionDiagnostics":
        return cls.from_dict(require_json_object(loads_json(text), context="Regime FeatureSelectionDiagnostics JSON"))

    @classmethod
    def from_filter_result(cls, filter_result: Any) -> "FeatureSelectionDiagnostics":
        dropped = []
        duplicate_pairs = []
        for record in getattr(filter_result, "dropped_features", ()):
            diagnostic = FeatureDropDiagnostic(
                column=getattr(record, "column"),
                reason=getattr(record, "reason"),
                metric_name=getattr(record, "metric_name", None),
                metric_value=getattr(record, "metric_value", None),
                reference_column=getattr(record, "reference_column", None),
            )
            dropped.append(diagnostic)
            if diagnostic.reason == "near_duplicate":
                duplicate_pairs.append(
                    {
                        "column": diagnostic.column,
                        "reference_column": diagnostic.reference_column,
                        "metric_name": diagnostic.metric_name,
                        "metric_value": diagnostic.metric_value,
                    }
                )
        config = getattr(filter_result, "config", {})
        if hasattr(config, "as_dict"):
            config_payload = config.as_dict()
        else:
            config_payload = dict(config)
        return cls(
            schema_version=int(getattr(filter_result, "schema_version", CANONICAL_SCHEMA_VERSION)),
            input_columns=tuple(getattr(filter_result, "input_columns")),
            retained_columns=tuple(getattr(filter_result, "retained_columns")),
            dropped_features=tuple(dropped),
            before_shape=tuple(getattr(filter_result, "before_shape")),
            after_shape=tuple(getattr(filter_result, "after_shape")),
            missingness_summary=dict(getattr(filter_result, "missingness_summary", {})),
            variance_summary=dict(getattr(filter_result, "variance_summary", {})),
            zero_share_summary=dict(getattr(filter_result, "zero_share_summary", {})),
            duplicate_summary={"dropped_pairs": duplicate_pairs},
            correlation_summary=dict(getattr(filter_result, "correlation_summary", {})),
            config=config_payload,
        )


__all__ = [
    "FeatureDropDiagnostic",
    "FeatureSelectionDiagnostics",
]
