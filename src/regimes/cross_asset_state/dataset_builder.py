from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from src.regimes.cross_asset_state.mask_contract import CrossAssetStateMaskReason


MASK_REASON_MAP: Mapping[str, str] = {
    "missing_relationship_snapshot": CrossAssetStateMaskReason.MISSING_RELATIONSHIP_SNAPSHOT,
    "asset_not_in_relationship_snapshot": CrossAssetStateMaskReason.ASSET_NOT_IN_RELATIONSHIP_SNAPSHOT,
    "missing_anchor": CrossAssetStateMaskReason.MISSING_ANCHOR,
    "missing_core_basket": CrossAssetStateMaskReason.MISSING_CORE_BASKET,
    "no_viable_peer_edges": CrossAssetStateMaskReason.NO_VIABLE_PEER_EDGES,
    "insufficient_peer_history": CrossAssetStateMaskReason.INSUFFICIENT_PEER_HISTORY,
    "insufficient_window_history": CrossAssetStateMaskReason.INSUFFICIENT_WINDOW_HISTORY,
    "insufficient_overlap": CrossAssetStateMaskReason.INSUFFICIENT_OVERLAP,
    "zero_variance_denominator": CrossAssetStateMaskReason.ZERO_VARIANCE_DENOMINATOR,
    "invalid_correlation_window": CrossAssetStateMaskReason.INVALID_CORRELATION_WINDOW,
    "missing_required_field": CrossAssetStateMaskReason.MISSING_REQUIRED_FAMILY_FIELDS,
    "insufficient_valid_unmasked_rows": CrossAssetStateMaskReason.INSUFFICIENT_VALID_UNMASKED_ROWS,
    "unavailable_zero_masked": CrossAssetStateMaskReason.UNAVAILABLE_ZERO_MASKED,
    "low_feature_spread": CrossAssetStateMaskReason.LOW_FEATURE_SPREAD,
    "constant_peer_count": CrossAssetStateMaskReason.CONSTANT_PEER_COUNT,
    "compressed_peer_strength": CrossAssetStateMaskReason.COMPRESSED_PEER_STRENGTH,
    "compressed_peer_stability": CrossAssetStateMaskReason.COMPRESSED_PEER_STABILITY,
    "low_entropy_spread": CrossAssetStateMaskReason.LOW_ENTROPY_SPREAD,
    "low_concentration_spread": CrossAssetStateMaskReason.LOW_CONCENTRATION_SPREAD,
    "insufficient_edge_diversity": CrossAssetStateMaskReason.INSUFFICIENT_EDGE_DIVERSITY,
    "no_variable_peer_support": CrossAssetStateMaskReason.NO_VARIABLE_PEER_SUPPORT,
    "insufficient_peer_support": CrossAssetStateMaskReason.INSUFFICIENT_PEER_SUPPORT,
    "insufficient_candidate_edges": CrossAssetStateMaskReason.INSUFFICIENT_CANDIDATE_EDGES,
    "no_candidate_edges_available": CrossAssetStateMaskReason.NO_CANDIDATE_EDGES_AVAILABLE,
    "no_valid_peer_weights": CrossAssetStateMaskReason.NO_VALID_PEER_WEIGHTS,
    "below_minimum_support_for_entropy": CrossAssetStateMaskReason.BELOW_MINIMUM_SUPPORT_FOR_ENTROPY,
    "below_minimum_support_for_concentration": CrossAssetStateMaskReason.BELOW_MINIMUM_SUPPORT_FOR_CONCENTRATION,
    "no_peer_weights_available": CrossAssetStateMaskReason.NO_PEER_WEIGHTS_AVAILABLE,
    "insufficient_prior_snapshot_for_churn": CrossAssetStateMaskReason.INSUFFICIENT_PRIOR_SNAPSHOT_FOR_CHURN,
    "all_peer_weights_equal": CrossAssetStateMaskReason.ALL_PEER_WEIGHTS_EQUAL,
    "low_edge_weight_spread": CrossAssetStateMaskReason.LOW_EDGE_WEIGHT_SPREAD,
    "unsupported_support_definition": CrossAssetStateMaskReason.UNSUPPORTED_SUPPORT_DEFINITION,
    "missing_required_repaired_column": CrossAssetStateMaskReason.MISSING_REQUIRED_REPAIRED_COLUMN,
    "no_viable_profile": CrossAssetStateMaskReason.NO_VIABLE_PROFILE,
    "family_diagnostic_only": CrossAssetStateMaskReason.FAMILY_DIAGNOSTIC_ONLY,
    "profile_type_not_selection_eligible": CrossAssetStateMaskReason.PROFILE_TYPE_NOT_SELECTION_ELIGIBLE,
    "economic_panel_missing": CrossAssetStateMaskReason.ECONOMIC_PANEL_MISSING,
    "nonfinite_input": CrossAssetStateMaskReason.NONFINITE_INPUT,
    "nonfinite_output": CrossAssetStateMaskReason.NONFINITE_OUTPUT,
    "diagnostic_only_peer_metadata": CrossAssetStateMaskReason.DIAGNOSTIC_ONLY_PEER_METADATA,
    "stale_relationship_snapshot": CrossAssetStateMaskReason.STALE_RELATIONSHIP_SNAPSHOT,
    "ambiguous_snapshot_resolution": CrossAssetStateMaskReason.AMBIGUOUS_SNAPSHOT_RESOLUTION,
    "not_applicable_for_family": CrossAssetStateMaskReason.NOT_APPLICABLE_FOR_FAMILY,
}


def load_relationship_value_availability(roots: Sequence[str | Path]) -> Any | None:
    pd = _pandas()
    frames = []
    for root in roots:
        root_path = Path(root)
        if not root_path.exists():
            continue
        if root_path.is_file():
            paths = [root_path]
        else:
            paths = [
                *root_path.rglob("*relationship_value_availability*.parquet"),
                *root_path.rglob("*relationship_value_availability*.csv"),
            ]
        for path in sorted(paths):
            try:
                frame = pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path)
            except Exception:
                continue
            if not frame.empty:
                frames.append(frame)
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


@dataclass
class RelationshipValueAvailabilityIndex:
    frame: Any
    _groups: dict[tuple[str, str, str, str], Any] = field(default_factory=dict)
    _scoped_groups: dict[tuple[str, ...], Any] = field(default_factory=dict)
    physical_key_columns: tuple[str, ...] = ()
    logical_key_fields: tuple[str, ...] = ()
    optional_scope_fields: tuple[str, ...] = ()
    indexed_row_count: int = 0
    build_seconds: float = 0.0
    build_count: int = 1
    lookup_count: int = 0
    hit_count: int = 0
    miss_count: int = 0
    bad_ts_filtered_count: int = 0
    field_lookup_count: int = 0
    field_hit_count: int = 0
    field_miss_count: int = 0
    scoped_lookup_count: int = 0
    scoped_hit_count: int = 0
    scoped_fallback_count: int = 0

    @classmethod
    def from_frame(cls, frame: Any | None) -> "RelationshipValueAvailabilityIndex | None":
        if frame is None or getattr(frame, "empty", True):
            return None
        started = time.perf_counter()
        index = cls(frame=frame)
        asset_col = _asset_column(frame)
        if asset_col is None or "band" not in frame.columns or "relationship_feature_family" not in frame.columns:
            index.build_seconds = round(time.perf_counter() - started, 6)
            return index
        field_col = "field_name" if "field_name" in frame.columns else None
        optional_cols = tuple(
            column
            for column in (
                "window_policy_id",
                "known_at_ts",
                "source_tail_ts",
                "relationship_snapshot_id",
                "snapshot_id",
            )
            if column in frame.columns
        )
        index.physical_key_columns = tuple(
            column for column in (asset_col, "band", "relationship_feature_family", field_col, *optional_cols) if column
        )
        index.logical_key_fields = tuple(
            dict.fromkeys(
                (
                    "asset_id",
                    "band",
                    "relationship_feature_family",
                    "field_name",
                    *optional_cols,
                )
            )
        )
        index.optional_scope_fields = optional_cols
        indexed = 0
        for row_index, row in frame.iterrows():
            field_name = str(row.get(field_col)) if field_col else ""
            key = (
                str(row.get(asset_col)),
                str(row.get("band")),
                str(row.get("relationship_feature_family")),
                field_name,
            )
            index._groups.setdefault(key, []).append(row_index)
            scoped_key = (*key, *(_scoped_value(row.get(column)) for column in optional_cols))
            padded_key = (*scoped_key, *("",) * (8 - len(scoped_key)))
            index._scoped_groups.setdefault(padded_key, []).append(row_index)
            indexed += 1
        index.indexed_row_count = indexed
        index.build_seconds = round(time.perf_counter() - started, 6)
        return index

    def filter_frame_to_available_family_rows(
        self,
        frame: Any,
        *,
        asset: str,
        band: str,
        family: str,
        required_columns: Sequence[str],
        window_policy_id: str | None = None,
    ) -> tuple[Any, str | None]:
        self.lookup_count += 1
        if frame is None or getattr(frame, "empty", True):
            self.miss_count += 1
            return frame, None
        row_indexes = self._row_indexes(
            asset=asset,
            band=band,
            family=family,
            required_columns=required_columns,
            window_policy_id=window_policy_id,
        )
        if not row_indexes:
            self.miss_count += 1
            return frame, None
        self.hit_count += 1
        scoped = self.frame.loc[row_indexes]
        pd = _pandas()
        bad = scoped[scoped["value_status"].astype(str) == "masked_unavailable"] if "value_status" in scoped.columns else pd.DataFrame()
        if bad.empty:
            return frame, None
        reason = _dominant_reason(bad)
        if "ts" not in frame.columns or "ts" not in bad.columns:
            self.bad_ts_filtered_count += len(frame)
            return frame.iloc[0:0].copy(), reason
        bad_ts = set(pd.to_numeric(bad["ts"], errors="coerce").dropna().astype("int64").tolist())
        mask = pd.to_numeric(frame["ts"], errors="coerce").astype("Int64").isin(bad_ts)
        self.bad_ts_filtered_count += int(mask.sum())
        filtered = frame[~mask].copy()
        return filtered, reason

    def field_availability_rows(
        self,
        *,
        asset: str,
        band: str,
        family: str,
        field_name: str,
        ts: int | float | str,
        window_policy_id: str | None = None,
        known_at_ts: int | float | str | None = None,
        source_tail_ts: int | float | str | None = None,
        relationship_snapshot_id: str | None = None,
    ) -> Any:
        self.field_lookup_count += 1
        rows = self._row_indexes(
            asset=asset,
            band=band,
            family=family,
            required_columns=(field_name,),
            window_policy_id=window_policy_id,
            known_at_ts=known_at_ts,
            source_tail_ts=source_tail_ts,
            relationship_snapshot_id=relationship_snapshot_id,
        )
        if not rows:
            self.field_miss_count += 1
            return self.frame.iloc[0:0]
        self.field_hit_count += 1
        scoped = self.frame.loc[list(rows)]
        if "ts" not in scoped.columns:
            return scoped
        pd = _pandas()
        return scoped[pd.to_numeric(scoped["ts"], errors="coerce") <= float(ts)]

    def _row_indexes(
        self,
        *,
        asset: str,
        band: str,
        family: str,
        required_columns: Sequence[str],
        window_policy_id: str | None = None,
        known_at_ts: int | float | str | None = None,
        source_tail_ts: int | float | str | None = None,
        relationship_snapshot_id: str | None = None,
    ) -> list[Any]:
        scoped_values = {
            "window_policy_id": window_policy_id,
            "known_at_ts": known_at_ts,
            "source_tail_ts": source_tail_ts,
            "relationship_snapshot_id": relationship_snapshot_id,
            "snapshot_id": relationship_snapshot_id,
        }
        scope_supplied = any(scoped_values.get(field) is not None for field in self.optional_scope_fields)
        row_indexes: list[Any] = []
        for column in required_columns:
            row_indexes.extend(self._groups.get((str(asset), str(band), str(family), str(column)), ()))
        row_indexes = list(dict.fromkeys(row_indexes))
        if scope_supplied:
            self.scoped_lookup_count += 1
            scoped_rows: list[Any] = []
            for column in required_columns:
                core_key = (str(asset), str(band), str(family), str(column))
                suffix = tuple(_scoped_value(scoped_values.get(field)) for field in self.optional_scope_fields)
                scoped_key = (*core_key, *suffix)
                padded_key = (*scoped_key, *("",) * (8 - len(scoped_key)))
                scoped_rows.extend(self._scoped_groups.get(padded_key, ()))
            if scoped_rows:
                self.scoped_hit_count += 1
                return scoped_rows
            scoped_rows = self._filter_core_rows_by_supplied_scope(row_indexes, scoped_values)
            if scoped_rows:
                self.scoped_hit_count += 1
                return scoped_rows
            self.scoped_fallback_count += 1
        return row_indexes

    def _filter_core_rows_by_supplied_scope(self, row_indexes: Sequence[Any], scoped_values: Mapping[str, Any]) -> list[Any]:
        if not row_indexes:
            return []
        scoped = self.frame.loc[list(row_indexes)]
        mask = None
        for field in self.optional_scope_fields:
            value = scoped_values.get(field)
            if value is None or field not in scoped.columns:
                continue
            current = scoped[field].astype(str) == str(value)
            mask = current if mask is None else mask & current
        if mask is None:
            return []
        return list(scoped[mask].index)

    def stats(self) -> dict[str, Any]:
        return {
            "artifact_kind": "cross_asset_state_availability_index_telemetry",
            "source_owned_index": True,
            "index_built": True,
            "build_count": int(self.build_count),
            "build_seconds": float(self.build_seconds),
            "source_row_count": int(len(self.frame)),
            "rows_indexed": int(self.indexed_row_count),
            "indexed_row_count": int(self.indexed_row_count),
            "group_count": int(len(self._groups)),
            "scoped_group_count": int(len(self._scoped_groups)),
            "physical_key_columns": list(self.physical_key_columns),
            "logical_key_fields": list(self.logical_key_fields),
            "optional_scope_fields": list(self.optional_scope_fields),
            "lookup_count": int(self.lookup_count),
            "hit_count": int(self.hit_count),
            "miss_count": int(self.miss_count),
            "bad_ts_filtered_count": int(self.bad_ts_filtered_count),
            "field_lookup_count": int(self.field_lookup_count),
            "field_hit_count": int(self.field_hit_count),
            "field_miss_count": int(self.field_miss_count),
            "scoped_lookup_count": int(self.scoped_lookup_count),
            "scoped_hit_count": int(self.scoped_hit_count),
            "scoped_fallback_count": int(self.scoped_fallback_count),
        }


def build_relationship_value_availability_index(availability: Any | None) -> RelationshipValueAvailabilityIndex | None:
    return RelationshipValueAvailabilityIndex.from_frame(availability)


@dataclass
class CrossAssetFeaturePanelMatrixCache:
    """Per-run in-memory cache for Cross-Asset feature panels and window matrices."""

    _feature_panels: dict[tuple[Any, ...], tuple[Any, str | None]] = field(default_factory=dict)
    _dataset_matrices: dict[tuple[Any, ...], tuple[Any, Any]] = field(default_factory=dict)
    feature_panel_lookup_count: int = 0
    feature_panel_hit_count: int = 0
    feature_panel_miss_count: int = 0
    feature_panel_build_count: int = 0
    feature_panel_build_seconds: float = 0.0
    dataset_matrix_lookup_count: int = 0
    dataset_matrix_hit_count: int = 0
    dataset_matrix_miss_count: int = 0
    dataset_matrix_build_count: int = 0
    dataset_matrix_build_seconds: float = 0.0

    key_fields: tuple[str, ...] = (
        "asset_id",
        "band",
        "relationship_feature_family",
        "window_policy_id",
        "feature_set_version",
        "relationship_context_id",
        "relationship_snapshot_id",
        "known_at_ts",
        "source_tail_ts",
        "required_columns",
        "source_frame_signature",
    )

    def filtered_feature_panel(
        self,
        frame: Any,
        availability: Any | None,
        *,
        asset: str,
        band: str,
        family: str,
        required_columns: Sequence[str],
        window_policy_id: str | None,
        feature_set_version: str,
        relationship_context_id: str | None,
        relationship_snapshot_id: str | None,
        known_at_ts: int | float | str | None,
        source_tail_ts: int | float | str | None,
    ) -> tuple[Any, str | None]:
        key = self._key(
            cache_layer="feature_panel",
            frame=frame,
            asset=asset,
            band=band,
            family=family,
            required_columns=required_columns,
            window_policy_id=window_policy_id,
            feature_set_version=feature_set_version,
            relationship_context_id=relationship_context_id,
            relationship_snapshot_id=relationship_snapshot_id,
            known_at_ts=known_at_ts,
            source_tail_ts=source_tail_ts,
        )
        self.feature_panel_lookup_count += 1
        cached = self._feature_panels.get(key)
        if cached is not None:
            self.feature_panel_hit_count += 1
            return cached
        self.feature_panel_miss_count += 1
        started = time.perf_counter()
        panel, reason = filter_frame_to_available_family_rows(
            frame,
            availability,
            asset=asset,
            band=band,
            family=family,
            required_columns=required_columns,
            window_policy_id=window_policy_id,
        )
        self.feature_panel_build_seconds += time.perf_counter() - started
        self.feature_panel_build_count += 1
        cached_panel = self._safe_panel(panel)
        self._feature_panels[key] = (cached_panel, reason)
        return cached_panel, reason

    def dataset_matrix(
        self,
        frame: Any,
        *,
        asset: str,
        band: str,
        family: str,
        required_columns: Sequence[str],
        window_policy_id: str | None,
        feature_set_version: str,
        relationship_context_id: str | None,
        relationship_snapshot_id: str | None,
        known_at_ts: int | float | str | None,
        source_tail_ts: int | float | str | None,
        matrix_role: str,
        min_rows: int,
        build_fn: Callable[[], tuple[Any, Any]],
    ) -> tuple[Any, Any]:
        key = self._key(
            cache_layer=f"dataset_matrix:{matrix_role}",
            frame=frame,
            asset=asset,
            band=band,
            family=family,
            required_columns=required_columns,
            window_policy_id=window_policy_id,
            feature_set_version=feature_set_version,
            relationship_context_id=relationship_context_id,
            relationship_snapshot_id=relationship_snapshot_id,
            known_at_ts=known_at_ts,
            source_tail_ts=source_tail_ts,
            extra=(("min_rows", int(min_rows)),),
        )
        self.dataset_matrix_lookup_count += 1
        cached = self._dataset_matrices.get(key)
        if cached is not None:
            self.dataset_matrix_hit_count += 1
            return cached
        self.dataset_matrix_miss_count += 1
        started = time.perf_counter()
        matrix, metadata = build_fn()
        self.dataset_matrix_build_seconds += time.perf_counter() - started
        self.dataset_matrix_build_count += 1
        cached_matrix = self._safe_panel(matrix)
        self._dataset_matrices[key] = (cached_matrix, metadata)
        return cached_matrix, metadata

    def stats(self) -> dict[str, Any]:
        return {
            "artifact_kind": "cross_asset_state_feature_panel_matrix_cache_telemetry",
            "cache_scope": "per_run_in_memory",
            "disk_cache_enabled": False,
            "key_fields": list(self.key_fields),
            "stale_reuse_guard": {
                "feature_set_version_in_key": True,
                "relationship_context_id_in_key": True,
                "relationship_snapshot_id_in_key": True,
                "window_policy_id_in_key": True,
                "known_at_ts_in_key": True,
                "source_tail_ts_in_key": True,
                "source_frame_signature_in_key": True,
                "cross_run_reuse_enabled": False,
            },
            "mask_semantics": {
                "masked_unavailable_rows_are_filtered_not_zero_filled": True,
                "known_at_source_tail_preserved": True,
            },
            "feature_panel_cache": {
                "lookup_count": int(self.feature_panel_lookup_count),
                "hit_count": int(self.feature_panel_hit_count),
                "miss_count": int(self.feature_panel_miss_count),
                "build_count": int(self.feature_panel_build_count),
                "build_seconds": round(float(self.feature_panel_build_seconds), 6),
                "entry_count": int(len(self._feature_panels)),
            },
            "dataset_matrix_cache": {
                "lookup_count": int(self.dataset_matrix_lookup_count),
                "hit_count": int(self.dataset_matrix_hit_count),
                "miss_count": int(self.dataset_matrix_miss_count),
                "build_count": int(self.dataset_matrix_build_count),
                "build_seconds": round(float(self.dataset_matrix_build_seconds), 6),
                "entry_count": int(len(self._dataset_matrices)),
            },
        }

    def _key(
        self,
        *,
        cache_layer: str,
        frame: Any,
        asset: str,
        band: str,
        family: str,
        required_columns: Sequence[str],
        window_policy_id: str | None,
        feature_set_version: str,
        relationship_context_id: str | None,
        relationship_snapshot_id: str | None,
        known_at_ts: int | float | str | None,
        source_tail_ts: int | float | str | None,
        extra: Sequence[tuple[str, Any]] = (),
    ) -> tuple[Any, ...]:
        return (
            str(cache_layer),
            str(asset),
            str(band),
            str(family),
            _cache_text(window_policy_id),
            str(feature_set_version),
            _cache_text(relationship_context_id),
            _cache_text(relationship_snapshot_id),
            _cache_text(known_at_ts),
            _cache_text(source_tail_ts),
            tuple(str(column) for column in required_columns),
            _frame_signature(frame),
            tuple((str(key), _cache_text(value)) for key, value in extra),
        )

    @staticmethod
    def _safe_panel(frame: Any) -> Any:
        if frame is None:
            return None
        if hasattr(frame, "copy"):
            return frame.copy(deep=False)
        return frame


def filter_frame_to_available_family_rows(
    frame: Any,
    availability: Any | None,
    *,
    asset: str,
    band: str,
    family: str,
    required_columns: Sequence[str],
    window_policy_id: str | None = None,
) -> tuple[Any, str | None]:
    if frame is None or availability is None:
        return frame, None
    if isinstance(availability, RelationshipValueAvailabilityIndex):
        return availability.filter_frame_to_available_family_rows(
            frame,
            asset=asset,
            band=band,
            family=family,
            required_columns=required_columns,
            window_policy_id=window_policy_id,
        )
    if availability.empty:
        return frame, None
    pd = _pandas()
    scoped = availability.copy()
    asset_col = _asset_column(scoped)
    if asset_col is not None:
        scoped = scoped[scoped[asset_col].astype(str) == str(asset)]
    if "band" in scoped.columns:
        scoped = scoped[scoped["band"].astype(str) == str(band)]
    if "relationship_feature_family" in scoped.columns:
        scoped = scoped[scoped["relationship_feature_family"].astype(str) == str(family)]
    if "field_name" in scoped.columns:
        scoped = scoped[scoped["field_name"].astype(str).isin([str(column) for column in required_columns])]
    if window_policy_id is not None and "window_policy_id" in scoped.columns:
        window_scoped = scoped[scoped["window_policy_id"].astype(str) == str(window_policy_id)]
        if not window_scoped.empty:
            scoped = window_scoped
    if scoped.empty:
        return frame, None
    bad = scoped[scoped["value_status"].astype(str) == "masked_unavailable"] if "value_status" in scoped.columns else pd.DataFrame()
    if bad.empty:
        return frame, None
    reason = _dominant_reason(bad)
    if "ts" not in frame.columns or "ts" not in bad.columns:
        return frame.iloc[0:0].copy(), reason
    bad_ts = set(pd.to_numeric(bad["ts"], errors="coerce").dropna().astype("int64").tolist())
    filtered = frame[~pd.to_numeric(frame["ts"], errors="coerce").astype("Int64").isin(bad_ts)].copy()
    return filtered, reason


def _dominant_reason(frame: Any) -> str:
    if "mask_reason" not in frame.columns:
        return CrossAssetStateMaskReason.RELATIONSHIP_SNAPSHOT_UNAVAILABLE
    reasons = [str(value) for value in frame["mask_reason"].dropna().tolist() if str(value).strip()]
    if not reasons:
        return CrossAssetStateMaskReason.RELATIONSHIP_SNAPSHOT_UNAVAILABLE
    counts = {reason: reasons.count(reason) for reason in set(reasons)}
    return MASK_REASON_MAP.get(max(counts, key=counts.get), CrossAssetStateMaskReason.RELATIONSHIP_SNAPSHOT_UNAVAILABLE)


def _asset_column(frame: Any) -> str | None:
    if "asset" in frame.columns:
        return "asset"
    if "asset_id" in frame.columns:
        return "asset_id"
    return None


def _scoped_value(value: Any) -> str:
    if value is None:
        return ""
    text = str(value)
    if not text or text.lower() == "nan":
        return ""
    return text


def _cache_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value)
    if not text or text.lower() == "nan":
        return ""
    return text


def _frame_signature(frame: Any) -> tuple[Any, ...]:
    if frame is None:
        return ("none",)
    columns = tuple(str(column) for column in getattr(frame, "columns", ()))
    row_count = int(len(frame)) if hasattr(frame, "__len__") else 0
    values: list[tuple[str, str, str]] = []
    for column in ("ts", "known_at_ts", "source_tail_ts"):
        if column not in columns:
            continue
        try:
            series = _pandas().to_numeric(frame[column], errors="coerce").dropna()
        except Exception:
            continue
        if getattr(series, "empty", True):
            values.append((column, "", ""))
            continue
        values.append((column, str(series.min()), str(series.max())))
    return ("frame", row_count, columns, tuple(values))


def _pandas() -> Any:
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("Cross-Asset-State dataset builder requires pandas") from exc
    return pd
