from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from src.forecasting.common.path_config import resolve_path, selected_profile
from src.regimes.core.feature_preprocessing import (
    FeatureFilterConfig,
    fit_regime_preprocessor,
    transform_regime_preprocessor,
)
from src.regimes.core.paths import (
    is_production_adjacent_path,
    require_foundation_report_root,
)

from .data_contracts import (
    DATASET_BUILD_STATUS_READY,
    DATASET_REASON_BUILDER_EXCEPTION,
    DATASET_REASON_EMPTY_WINDOW,
    DATASET_REASON_INSUFFICIENT_HOLDOUT_ROWS,
    DATASET_REASON_INSUFFICIENT_TRAIN_ROWS,
    DATASET_REASON_INSUFFICIENT_VALIDATION_ROWS,
    DATASET_REASON_LEAKAGE_RISK_COLUMNS,
    DATASET_REASON_MALFORMED_PARTITION,
    DATASET_REASON_MISSING_ASSET_PARTITION,
    DATASET_REASON_MISSING_FEATURE_COLUMNS,
    DATASET_REASON_MISSING_INTERVAL_ROOT,
    DATASET_REASON_MISSING_PARTITION,
    DATASET_REASON_MISSING_SOURCE_ROOT,
    DATASET_REASON_NONFINITE_FEATURE_MATRIX,
    DATASET_REASON_NO_RETAINED_TRAIN_FEATURES,
    DATASET_REASON_PREPROCESSING_FAILED,
    DATASET_REASON_UNSAFE_DIAGNOSTICS_ROOT,
    AssetStateSplitSpec,
    AssetStateStudyDatasetRequest,
    DatasetBuildResult,
    PartitionLineage,
    SplitMatrix,
    blocked_dataset_result,
    empty_split_matrix,
)
from .dataset import LEAKAGE_COLUMN_TOKENS
from .feature_registry import (
    AssetStateFeaturePoolSpec,
    canonical_interval_feature_name,
    default_asset_state_feature_pool_registry as default_dataset_feature_pool_registry,
)
from .feature_pools import default_asset_state_feature_pool_registry as default_rich_feature_pool_registry
from .taxonomy import default_asset_state_taxonomy


TIMESTAMP_COLUMN = "ts"
ASSET_COLUMN = "asset"

SCALAR_FEATURE_STATUS_REAL_PROJECT_DATA_FOUND = "real_project_data_found"
SCALAR_FEATURE_STATUS_ROOT_NOT_CONFIGURED = "scalar_feature_root_not_configured"
SCALAR_FEATURE_STATUS_ROOT_MISSING = "scalar_feature_root_missing"
SCALAR_FEATURE_STATUS_INTERVALS_NOT_FOUND = "scalar_feature_intervals_not_found"
SCALAR_FEATURE_STATUS_ASSETS_NOT_FOUND = "scalar_feature_assets_not_found"
SCALAR_FEATURE_STATUS_PARTITIONS_MISSING = "scalar_feature_partitions_missing"
DATASET_PROBE_STATUS_BUILD_SUCCEEDED = "dataset_build_succeeded"
DATASET_PROBE_STATUS_BUILD_FAILED = "dataset_build_failed"


@dataclass(frozen=True)
class ScalarFeatureRootResolution:
    status: str
    root: Path | None
    source: str
    configured: bool
    intervals: tuple[int, ...] = ()
    message: str | None = None

    @property
    def found(self) -> bool:
        return self.status == SCALAR_FEATURE_STATUS_REAL_PROJECT_DATA_FOUND and self.root is not None

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "root": str(self.root) if self.root is not None else None,
            "source": self.source,
            "configured": bool(self.configured),
            "intervals": list(self.intervals),
            "message": self.message,
        }


def resolve_asset_state_scalar_feature_root(
    source_feature_root: str | Path | None = None,
    *,
    manifest: Any | None = None,
    profile: str | None = None,
    config_path: str | Path | None = None,
    env: Mapping[str, str] | None = None,
    include_default_parquet: bool = True,
) -> ScalarFeatureRootResolution:
    """Resolve the read-only scalar-feature root using project path conventions."""
    source_env = env if env is not None else os.environ
    candidates = _scalar_feature_root_candidates(
        source_feature_root=source_feature_root,
        manifest=manifest,
        profile=profile,
        config_path=config_path,
        env=source_env,
        include_default_parquet=include_default_parquet,
    )
    if not candidates:
        return ScalarFeatureRootResolution(
            status=SCALAR_FEATURE_STATUS_ROOT_NOT_CONFIGURED,
            root=None,
            source="unconfigured",
            configured=False,
            message="scalar feature root was not supplied and no project path configuration was available",
        )

    first_missing: tuple[Path, str, bool] | None = None
    first_no_intervals: tuple[Path, str, bool] | None = None
    for root, source, configured in candidates:
        if not root.exists() or not root.is_dir():
            if first_missing is None:
                first_missing = (root, source, configured)
            continue
        intervals = discover_scalar_feature_intervals(root)
        if intervals:
            return ScalarFeatureRootResolution(
                status=SCALAR_FEATURE_STATUS_REAL_PROJECT_DATA_FOUND,
                root=root,
                source=source,
                configured=configured,
                intervals=intervals,
                message="scalar feature intervals discovered",
            )
        if first_no_intervals is None:
            first_no_intervals = (root, source, configured)

    if first_no_intervals is not None:
        root, source, configured = first_no_intervals
        return ScalarFeatureRootResolution(
            status=SCALAR_FEATURE_STATUS_INTERVALS_NOT_FOUND,
            root=root,
            source=source,
            configured=configured,
            intervals=(),
            message="scalar feature root exists but no scalar_features_<interval> directories were found",
        )
    root, source, configured = first_missing
    return ScalarFeatureRootResolution(
        status=SCALAR_FEATURE_STATUS_ROOT_MISSING,
        root=root,
        source=source,
        configured=configured,
        intervals=(),
        message="configured scalar feature root does not exist",
    )


def discover_scalar_feature_intervals(source_feature_root: str | Path) -> tuple[int, ...]:
    """Return available scalar feature intervals without recursive history scans."""
    root = Path(source_feature_root)
    intervals: set[int] = set()

    for parent in _scalar_feature_parent_roots(root):
        if parent.name.startswith("scalar_features_"):
            parsed = _parse_interval_dir_name(parent.name)
            if parsed is not None and parent.exists():
                intervals.add(parsed)
            continue
        if not parent.exists() or not parent.is_dir():
            continue
        for child in parent.iterdir():
            if not child.is_dir():
                continue
            parsed = _parse_interval_dir_name(child.name)
            if parsed is not None:
                intervals.add(parsed)

    return tuple(sorted(intervals))


def discover_scalar_feature_assets(
    source_feature_root: str | Path,
    interval: int,
) -> tuple[str, ...]:
    """Return assets available for one interval using direct partition directories."""
    assets: set[str] = set()
    for interval_root in _interval_root_candidates(Path(source_feature_root), interval):
        if not interval_root.exists() or not interval_root.is_dir():
            continue
        for child in interval_root.iterdir():
            if child.is_dir() and child.name.startswith("asset="):
                value = child.name.split("=", 1)[1]
                if value:
                    assets.add(value)
    return tuple(sorted(assets))


def resolve_scalar_feature_partitions(
    source_feature_root: str | Path,
    *,
    interval: int,
    asset: str,
    start_ts: int | None = None,
    end_ts: int | None = None,
) -> tuple[Path, ...]:
    """Resolve parquet partitions for one asset and interval using bounded listing."""
    paths: list[Path] = []
    seen: set[Path] = set()
    for interval_root in _interval_root_candidates(Path(source_feature_root), interval):
        if not interval_root.exists() or not interval_root.is_dir():
            continue
        asset_root = interval_root / f"asset={asset}"
        candidates: Iterable[Path]
        if asset_root.exists() and asset_root.is_dir():
            candidates = _asset_partition_parquet_paths(asset_root, start_ts=start_ts, end_ts=end_ts)
        else:
            candidates = interval_root.glob("*.parquet")
        for path in candidates:
            if path.is_file():
                resolved = path.resolve()
                if resolved not in seen:
                    paths.append(path)
                    seen.add(resolved)
    return tuple(sorted(paths))


def resolve_scalar_feature_source_tail_ts(
    source_feature_root: str | Path,
    *,
    interval: int,
    asset: str,
) -> int | None:
    """Return the latest observed ts for one asset/interval using newest partitions first."""
    partitions = resolve_scalar_feature_partitions(source_feature_root, interval=int(interval), asset=str(asset))
    for path in sorted(partitions, key=lambda item: str(item), reverse=True):
        try:
            frame = pd.read_parquet(path, columns=[TIMESTAMP_COLUMN])
        except Exception:
            continue
        if frame.empty or TIMESTAMP_COLUMN not in frame.columns:
            continue
        values = pd.to_numeric(frame[TIMESTAMP_COLUMN], errors="coerce").dropna()
        if values.empty:
            continue
        return int(values.max())
    return None


def resolve_asset_state_band_source_tail_ts(
    source_feature_root: str | Path,
    *,
    asset: str,
    band: str,
) -> int | None:
    """Return the safe common source tail for an Asset-State band.

    The band tail is the minimum latest timestamp across member intervals, so a
    window ending at this tail is supported by every required member interval.
    """
    band_spec = default_asset_state_taxonomy().band_spec(str(band))
    tails: list[int] = []
    for interval in tuple(int(value) for value in band_spec.member_intervals):
        tail = resolve_scalar_feature_source_tail_ts(source_feature_root, interval=int(interval), asset=str(asset))
        if tail is None:
            return None
        tails.append(int(tail))
    return min(tails) if tails else None


def build_asset_state_study_dataset(
    request: AssetStateStudyDatasetRequest,
) -> DatasetBuildResult:
    """Build a train/validation/holdout scalar-feature panel for one asset-axis-band study."""
    try:
        diagnostics_root = _validate_diagnostics_root(request) if request.write_diagnostics else None
        root_resolution = resolve_asset_state_scalar_feature_root(request.source_feature_root)
        source_root = root_resolution.root
        if source_root is None:
            return _maybe_write_diagnostics(
                blocked_dataset_result(
                    request,
                    [DATASET_REASON_MISSING_SOURCE_ROOT],
                    message=root_resolution.message,
                    metadata={"scalar_feature_root_resolution": root_resolution.as_dict()},
                ),
                diagnostics_root,
            )
        if not source_root.exists() or not source_root.is_dir():
            return _maybe_write_diagnostics(
                blocked_dataset_result(
                    request,
                    [DATASET_REASON_MISSING_SOURCE_ROOT],
                    message=f"source feature root does not exist: {source_root}",
                    metadata={"scalar_feature_root_resolution": root_resolution.as_dict()},
                ),
                diagnostics_root,
            )

        taxonomy = default_asset_state_taxonomy()
        taxonomy.validate_axis_band(request.axis_id, request.band_id)
        band_spec = taxonomy.band_spec(request.band_id)
        feature_pool = _resolve_feature_pool(request)
        member_intervals = tuple(int(i) for i in band_spec.member_intervals)
        frames: list[pd.DataFrame] = []
        lineage: list[PartitionLineage] = []
        missing_intervals: list[int] = []
        missing_asset_intervals: list[int] = []
        missing_partition_intervals: list[int] = []

        for interval in member_intervals:
            interval_roots = tuple(
                p for p in _interval_root_candidates(source_root, interval) if p.exists() and p.is_dir()
            )
            if not interval_roots:
                missing_intervals.append(interval)
                if request.require_all_member_intervals:
                    continue
                else:
                    continue

            if request.asset not in discover_scalar_feature_assets(source_root, interval):
                has_direct_parquet = any(any(root.glob("*.parquet")) for root in interval_roots)
                if not has_direct_parquet:
                    missing_asset_intervals.append(interval)
                    if request.require_all_member_intervals:
                        continue
                    else:
                        continue

            part_paths = resolve_scalar_feature_partitions(
                source_root,
                interval=interval,
                asset=request.asset,
                start_ts=request.start_ts,
                end_ts=request.end_ts,
            )
            if not part_paths:
                missing_partition_intervals.append(interval)
                if request.require_all_member_intervals:
                    continue
                else:
                    continue

            try:
                interval_frame, interval_lineage = _read_interval_partitions(
                    interval=interval,
                    asset=request.asset,
                    paths=part_paths,
                    interval_roots=interval_roots,
                    start_ts=request.start_ts,
                    end_ts=request.end_ts,
                    feature_pool=feature_pool,
                )
            except LeakageColumnError as exc:
                return _maybe_write_diagnostics(
                    blocked_dataset_result(
                        request,
                        [DATASET_REASON_LEAKAGE_RISK_COLUMNS],
                        message=str(exc),
                        partition_lineage=lineage,
                    ),
                    diagnostics_root,
                )
            except MalformedPartitionError as exc:
                return _maybe_write_diagnostics(
                    blocked_dataset_result(
                        request,
                        [DATASET_REASON_MALFORMED_PARTITION],
                        message=str(exc),
                        partition_lineage=lineage,
                    ),
                    diagnostics_root,
                )

            if interval_frame.empty:
                missing_partition_intervals.append(interval)
                continue
            frames.append(interval_frame)
            lineage.extend(interval_lineage)

        reason_codes: list[str] = []
        if missing_intervals:
            reason_codes.append(DATASET_REASON_MISSING_INTERVAL_ROOT)
        if missing_asset_intervals:
            reason_codes.append(DATASET_REASON_MISSING_ASSET_PARTITION)
        if missing_partition_intervals:
            reason_codes.append(DATASET_REASON_MISSING_PARTITION)
        if request.require_all_member_intervals and reason_codes:
            return _maybe_write_diagnostics(
                blocked_dataset_result(
                    request,
                    reason_codes,
                    message="required scalar feature interval partitions are missing",
                    partition_lineage=lineage,
                    metadata={
                        "missing_interval_roots": missing_intervals,
                        "missing_asset_intervals": missing_asset_intervals,
                        "missing_partition_intervals": missing_partition_intervals,
                    },
                ),
                diagnostics_root,
            )
        if not frames:
            return _maybe_write_diagnostics(
                blocked_dataset_result(
                    request,
                    reason_codes or [DATASET_REASON_EMPTY_WINDOW],
                    message="no scalar feature rows were loaded for the requested window",
                    partition_lineage=lineage,
                ),
                diagnostics_root,
            )

        panel = _merge_interval_frames(frames)
        panel = _filter_window(panel, request.start_ts, request.end_ts)
        if panel.empty:
            return _maybe_write_diagnostics(
                blocked_dataset_result(
                    request,
                    [DATASET_REASON_EMPTY_WINDOW],
                    message="loaded partitions contain no rows in the requested window",
                    partition_lineage=lineage,
                ),
                diagnostics_root,
            )

        feature_columns = _feature_columns_for_panel(panel, feature_pool, member_intervals)
        if not feature_columns:
            return _maybe_write_diagnostics(
                blocked_dataset_result(
                    request,
                    [DATASET_REASON_MISSING_FEATURE_COLUMNS],
                    message="no expected scalar feature columns are available for the requested axis and band",
                    partition_lineage=lineage,
                    missingness_summary=_missingness_summary(panel, ()),
                ),
                diagnostics_root,
            )

        train_frame, validation_frame, holdout_frame = _split_panel(panel, request.split)
        split_counts_before = {
            "raw": int(panel.shape[0]),
            "train_before_cleaning": int(train_frame.shape[0]),
            "validation_before_cleaning": int(validation_frame.shape[0]),
            "holdout_before_cleaning": int(holdout_frame.shape[0]),
        }
        if train_frame.shape[0] < request.min_train_rows:
            return _maybe_write_diagnostics(
                blocked_dataset_result(
                    request,
                    [DATASET_REASON_INSUFFICIENT_TRAIN_ROWS],
                    message="train split has too few rows before preprocessing",
                    partition_lineage=lineage,
                    missingness_summary=_missingness_summary(panel, feature_columns, train_frame=train_frame),
                    metadata=split_counts_before,
                ),
                diagnostics_root,
            )
        if validation_frame.shape[0] < request.min_validation_rows:
            return _maybe_write_diagnostics(
                blocked_dataset_result(
                    request,
                    [DATASET_REASON_INSUFFICIENT_VALIDATION_ROWS],
                    message="validation split has too few rows before preprocessing",
                    partition_lineage=lineage,
                    missingness_summary=_missingness_summary(panel, feature_columns, train_frame=train_frame),
                    metadata=split_counts_before,
                ),
                diagnostics_root,
            )
        if request.min_holdout_rows and holdout_frame.shape[0] < request.min_holdout_rows:
            return _maybe_write_diagnostics(
                blocked_dataset_result(
                    request,
                    [DATASET_REASON_INSUFFICIENT_HOLDOUT_ROWS],
                    message="holdout split has too few rows before preprocessing",
                    partition_lineage=lineage,
                    missingness_summary=_missingness_summary(panel, feature_columns, train_frame=train_frame),
                    metadata=split_counts_before,
                ),
                diagnostics_root,
            )

        try:
            fitted = fit_regime_preprocessor(
                train_frame,
                feature_columns=feature_columns,
                preprocess=request.preprocess,
                preprocess_params=dict(request.preprocess_params),
                filter_config=FeatureFilterConfig(min_non_null_count=2),
                fit_window_role="train",
            )
            validation_matrix = transform_regime_preprocessor(
                validation_frame,
                fitted,
                window_role="validation",
            )
            holdout_matrix = (
                transform_regime_preprocessor(
                    holdout_frame,
                    fitted,
                    window_role="holdout",
                )
                if not holdout_frame.empty
                else None
            )
        except Exception as exc:
            return _maybe_write_diagnostics(
                blocked_dataset_result(
                    request,
                    [DATASET_REASON_PREPROCESSING_FAILED],
                    message=f"train-only preprocessing failed: {exc}",
                    partition_lineage=lineage,
                    missingness_summary=_missingness_summary(panel, feature_columns, train_frame=train_frame),
                ),
                diagnostics_root,
            )

        if fitted.x.shape[1] < request.min_feature_count:
            return _maybe_write_diagnostics(
                blocked_dataset_result(
                    request,
                    [DATASET_REASON_NO_RETAINED_TRAIN_FEATURES],
                    message="train-fitted feature filters retained too few columns",
                    partition_lineage=lineage,
                    missingness_summary=_missingness_summary(panel, feature_columns, train_frame=train_frame),
                    metadata=fitted.to_metadata(),
                ),
                diagnostics_root,
            )
        if fitted.x.shape[0] < request.min_train_rows:
            return _maybe_write_diagnostics(
                blocked_dataset_result(
                    request,
                    [DATASET_REASON_INSUFFICIENT_TRAIN_ROWS],
                    message="train split has too few rows after train-fitted preprocessing",
                    partition_lineage=lineage,
                    missingness_summary=_missingness_summary(panel, feature_columns, train_frame=train_frame),
                    metadata=fitted.to_metadata(),
                ),
                diagnostics_root,
            )
        if validation_matrix.x.shape[0] < request.min_validation_rows:
            return _maybe_write_diagnostics(
                blocked_dataset_result(
                    request,
                    [DATASET_REASON_INSUFFICIENT_VALIDATION_ROWS],
                    message="validation split has too few rows after train-fitted preprocessing",
                    partition_lineage=lineage,
                    missingness_summary=_missingness_summary(panel, feature_columns, train_frame=train_frame),
                    metadata=validation_matrix.metadata,
                ),
                diagnostics_root,
            )
        if request.min_holdout_rows and (
            holdout_matrix is None or holdout_matrix.x.shape[0] < request.min_holdout_rows
        ):
            return _maybe_write_diagnostics(
                blocked_dataset_result(
                    request,
                    [DATASET_REASON_INSUFFICIENT_HOLDOUT_ROWS],
                    message="holdout split has too few rows after train-fitted preprocessing",
                    partition_lineage=lineage,
                    missingness_summary=_missingness_summary(panel, feature_columns, train_frame=train_frame),
                ),
                diagnostics_root,
            )

        if not _finite_matrix(fitted.x) or not _finite_matrix(validation_matrix.x):
            return _maybe_write_diagnostics(
                blocked_dataset_result(
                    request,
                    [DATASET_REASON_NONFINITE_FEATURE_MATRIX],
                    message="non-finite values remain after preprocessing",
                    partition_lineage=lineage,
                    missingness_summary=_missingness_summary(panel, feature_columns, train_frame=train_frame),
                ),
                diagnostics_root,
            )
        if holdout_matrix is not None and not _finite_matrix(holdout_matrix.x):
            return _maybe_write_diagnostics(
                blocked_dataset_result(
                    request,
                    [DATASET_REASON_NONFINITE_FEATURE_MATRIX],
                    message="non-finite values remain in holdout after preprocessing",
                    partition_lineage=lineage,
                    missingness_summary=_missingness_summary(panel, feature_columns, train_frame=train_frame),
                ),
                diagnostics_root,
            )

        horizon_min = int(request.forward_horizon_min or band_spec.ceiling_interval_min)
        target_frame = _compute_forward_targets(
            panel,
            target_interval=int(band_spec.ceiling_interval_min),
            horizon_min=horizon_min,
        )

        train_split = SplitMatrix(
            name="train",
            x=fitted.x,
            timestamps=_timestamps_from_frame(fitted.clean_frame),
            frame=fitted.clean_frame,
            targets=None,
            row_count_before_cleaning=int(train_frame.shape[0]),
            dropped_rows=int(train_frame.shape[0] - fitted.x.shape[0]),
            metadata={
                "fit_scope": "train_only",
                "targets_computed": False,
                "preprocessor": fitted.to_metadata(),
            },
        )
        validation_split = SplitMatrix(
            name="validation",
            x=validation_matrix.x,
            timestamps=_timestamps_from_frame(validation_matrix.clean_frame),
            frame=validation_matrix.clean_frame,
            targets=_align_targets(validation_matrix.clean_frame, target_frame),
            row_count_before_cleaning=int(validation_frame.shape[0]),
            dropped_rows=int(validation_frame.shape[0] - validation_matrix.x.shape[0]),
            metadata={
                "transform_scope": "train_fitted_preprocessor",
                "targets_computed": True,
                "target_horizon_min": horizon_min,
            },
        )
        holdout_split = (
            SplitMatrix(
                name="holdout",
                x=holdout_matrix.x,
                timestamps=_timestamps_from_frame(holdout_matrix.clean_frame),
                frame=holdout_matrix.clean_frame,
                targets=_align_targets(holdout_matrix.clean_frame, target_frame),
                row_count_before_cleaning=int(holdout_frame.shape[0]),
                dropped_rows=int(holdout_frame.shape[0] - holdout_matrix.x.shape[0]),
                metadata={
                    "transform_scope": "train_fitted_preprocessor",
                    "targets_computed": True,
                    "target_horizon_min": horizon_min,
                },
            )
            if holdout_matrix is not None
            else empty_split_matrix("holdout")
        )

        dropped_rows = {
            "train": train_split.dropped_rows,
            "validation": validation_split.dropped_rows,
            "holdout": holdout_split.dropped_rows,
        }
        row_counts = {
            **split_counts_before,
            "train": train_split.row_count,
            "validation": validation_split.row_count,
            "holdout": holdout_split.row_count,
        }
        result = DatasetBuildResult(
            status=DATASET_BUILD_STATUS_READY,
            reason_codes=(),
            asset=request.asset,
            axis=request.axis_id,
            band=request.band_id,
            schema_version=request.schema_version,
            train=train_split,
            validation=validation_split,
            holdout=holdout_split,
            feature_columns=tuple(feature_columns),
            output_feature_names=tuple(fitted.output_feature_names),
            scalar_feature_columns_available=tuple(sorted(c for c in panel.columns if c not in {ASSET_COLUMN, TIMESTAMP_COLUMN})),
            partition_lineage=tuple(lineage),
            missingness_summary=_missingness_summary(panel, feature_columns, train_frame=train_frame),
            row_counts=row_counts,
            dropped_rows=dropped_rows,
            feature_count=int(fitted.x.shape[1]),
            metadata={
                "request": request.as_dict(),
                "asset_metadata": {
                    "asset": request.asset,
                    "member_intervals_min": list(member_intervals),
                    "source_feature_root": str(source_root),
                    "source_feature_root_resolution": root_resolution.as_dict(),
                },
                "split_policy": request.split.as_dict(),
                "preprocessing_policy": {
                    "fit_scope": "train_only",
                    "feature_filters_fit_scope": "train_only",
                    "validation_transform_scope": "train_fitted_preprocessor",
                    "holdout_transform_scope": "train_fitted_preprocessor",
                },
                "target_policy": {
                    "computed_for_train": False,
                    "computed_for_validation": True,
                    "computed_for_holdout": True,
                    "horizon_min": horizon_min,
                    "target_interval_min": int(band_spec.ceiling_interval_min),
                },
            },
        )
        return _maybe_write_diagnostics(result, diagnostics_root)
    except ValueError:
        raise
    except Exception as exc:
        return blocked_dataset_result(
            request,
            [DATASET_REASON_BUILDER_EXCEPTION],
            message=f"asset-state dataset builder failed closed: {exc}",
        )


def build_asset_state_dataset(
    request: AssetStateStudyDatasetRequest,
) -> DatasetBuildResult:
    """Compatibility alias for the Block 2 study dataset builder."""
    return build_asset_state_study_dataset(request)


class LeakageColumnError(ValueError):
    pass


class MalformedPartitionError(ValueError):
    pass


def _scalar_feature_parent_roots(root: Path) -> tuple[Path, ...]:
    candidates = [root]
    if root.name != "scalar_features":
        candidates.append(root / "scalar_features")
    candidates.append(root / "model_states" / "scalar_features")
    return tuple(dict.fromkeys(candidates))


def _scalar_feature_root_candidates(
    *,
    source_feature_root: str | Path | None,
    manifest: Any | None,
    profile: str | None,
    config_path: str | Path | None,
    env: Mapping[str, str],
    include_default_parquet: bool,
) -> tuple[tuple[Path, str, bool], ...]:
    candidates: list[tuple[Path, str, bool]] = []

    def add(value: object, source: str, configured: bool) -> None:
        path = _clean_candidate_root(value)
        if path is not None:
            candidates.append((path, source, bool(configured)))

    add(source_feature_root, "explicit_argument", True)
    if source_feature_root is None and manifest is not None:
        add(getattr(manifest, "source_feature_root", None), "manifest_field", True)

    resolved_profile = str(profile or selected_profile(env=env))
    resolved_config_path = Path(config_path) if config_path is not None else None
    add(
        resolve_path(
            "source_feature_root",
            profile=resolved_profile,
            env=env,
            config_path=resolved_config_path,
            required=False,
        ),
        "path_config.source_feature_root",
        True,
    )
    add(
        resolve_path(
            "output_parquet_root",
            profile=resolved_profile,
            env=env,
            config_path=resolved_config_path,
            required=False,
        ),
        "path_config.output_parquet_root",
        True,
    )

    pipeline_root = _clean_candidate_root(env.get("PIPELINE_ROOT", ""))
    if pipeline_root is not None:
        add(pipeline_root / "parquet", "env.PIPELINE_ROOT/parquet", True)

    if include_default_parquet:
        add(Path("parquet"), "default_relative_parquet", False)

    return _dedupe_root_candidates(candidates)


def _clean_candidate_root(value: object) -> Path | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    return Path(raw).expanduser()


def _dedupe_root_candidates(candidates: Sequence[tuple[Path, str, bool]]) -> tuple[tuple[Path, str, bool], ...]:
    out: list[tuple[Path, str, bool]] = []
    seen: set[str] = set()
    for path, source, configured in candidates:
        try:
            key = str(path.resolve(strict=False)).lower()
        except Exception:
            key = str(path).lower()
        if key in seen:
            continue
        out.append((path, source, configured))
        seen.add(key)
    return tuple(out)


def _parse_interval_dir_name(name: str) -> int | None:
    prefix = "scalar_features_"
    if not name.startswith(prefix):
        return None
    suffix = name[len(prefix) :]
    try:
        value = int(suffix)
    except ValueError:
        return None
    return value if value > 0 else None


def _interval_root_candidates(source_root: Path, interval: int) -> tuple[Path, ...]:
    interval_name = f"scalar_features_{int(interval)}"
    if source_root.name == interval_name:
        return (source_root,)
    return tuple(parent / interval_name for parent in _scalar_feature_parent_roots(source_root))


def _asset_partition_parquet_paths(
    asset_root: Path,
    *,
    start_ts: int | None,
    end_ts: int | None,
) -> tuple[Path, ...]:
    paths: list[Path] = []
    if start_ts is not None and end_ts is not None:
        for year, month in _months_in_window(start_ts, end_ts):
            for month_label in {f"{month:02d}", str(month)}:
                month_root = asset_root / f"year={year}" / f"month={month_label}"
                if month_root.exists() and month_root.is_dir():
                    paths.extend(sorted(month_root.glob("*.parquet")))
        return tuple(sorted(dict.fromkeys(paths)))

    for year_root in sorted(asset_root.glob("year=*")):
        if not year_root.is_dir():
            continue
        for month_root in sorted(year_root.glob("month=*")):
            if month_root.is_dir():
                paths.extend(sorted(month_root.glob("*.parquet")))
    paths.extend(sorted(asset_root.glob("*.parquet")))
    return tuple(sorted(dict.fromkeys(paths)))


def _months_in_window(start_ts: int | None, end_ts: int | None) -> tuple[tuple[int, int], ...]:
    if start_ts is None and end_ts is None:
        return ()
    start_dt = datetime.fromtimestamp(int(start_ts or 0), tz=timezone.utc)
    end_dt = datetime.fromtimestamp(int((end_ts or start_ts or 0) - 1), tz=timezone.utc)
    months: list[tuple[int, int]] = []
    year = start_dt.year
    month = start_dt.month
    while (year, month) <= (end_dt.year, end_dt.month):
        months.append((year, month))
        if month == 12:
            year += 1
            month = 1
        else:
            month += 1
    return tuple(months)


def _resolve_feature_pool(request: AssetStateStudyDatasetRequest) -> AssetStateFeaturePoolSpec:
    registry = default_dataset_feature_pool_registry()
    if request.feature_pool_id:
        try:
            return registry.get(request.feature_pool_id)
        except ValueError:
            rich_pool = default_rich_feature_pool_registry().get(str(request.feature_pool_id))
            return AssetStateFeaturePoolSpec(
                pool_id=rich_pool.feature_pool_id,
                axis=rich_pool.axis,
                feature_bases=tuple(
                    dict.fromkeys(
                        (
                            *tuple(rich_pool.required_source_columns),
                            *tuple(rich_pool.optional_source_columns),
                        )
                    )
                ),
                compatible_bands=rich_pool.compatible_bands,
                expected_source_kind="scalar_feature_parquet",
                missingness_policy=rich_pool.missingness_policy,
                leakage_policy=str(rich_pool.leakage_policy.get("policy", "source_features_only_no_forward_targets")),
            )
    return registry.default_for_axis(request.axis_id, band=request.band_id)


def _read_interval_partitions(
    *,
    interval: int,
    asset: str,
    paths: Sequence[Path],
    interval_roots: Sequence[Path],
    start_ts: int | None,
    end_ts: int | None,
    feature_pool: AssetStateFeaturePoolSpec,
) -> tuple[pd.DataFrame, tuple[PartitionLineage, ...]]:
    frames: list[pd.DataFrame] = []
    lineage: list[PartitionLineage] = []
    for path in paths:
        frame = pd.read_parquet(path)
        if TIMESTAMP_COLUMN not in frame.columns:
            raise MalformedPartitionError(f"partition missing ts column: {path}")
        leakage_columns = _leakage_columns(frame.columns)
        if leakage_columns:
            raise LeakageColumnError(
                f"partition contains leakage-risk columns {leakage_columns}: {path}"
            )
        frame = frame.copy()
        frame[TIMESTAMP_COLUMN] = pd.to_numeric(frame[TIMESTAMP_COLUMN], errors="coerce")
        frame = frame.dropna(subset=[TIMESTAMP_COLUMN])
        frame[TIMESTAMP_COLUMN] = frame[TIMESTAMP_COLUMN].astype("int64")
        if ASSET_COLUMN not in frame.columns:
            partition_asset = _partition_value(path, "asset")
            frame[ASSET_COLUMN] = partition_asset or asset
        frame = frame[frame[ASSET_COLUMN].astype(str) == asset]
        frame = _filter_window(frame, start_ts, end_ts)
        root = _nearest_interval_root(path, interval_roots)
        lineage.append(
            PartitionLineage(
                interval=int(interval),
                asset=asset,
                path=str(path),
                root=str(root),
                row_count=int(frame.shape[0]),
                min_ts=_min_ts(frame),
                max_ts=_max_ts(frame),
                columns=tuple(str(c) for c in frame.columns),
                year=_partition_int_value(path, "year"),
                month=_partition_int_value(path, "month"),
            )
        )
        if frame.empty:
            continue
        frames.append(_canonicalize_interval_frame(frame, interval, feature_pool))

    if not frames:
        return pd.DataFrame(), tuple(lineage)
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values(TIMESTAMP_COLUMN).drop_duplicates(
        subset=[ASSET_COLUMN, TIMESTAMP_COLUMN],
        keep="last",
    )
    return combined.reset_index(drop=True), tuple(lineage)


def _canonicalize_interval_frame(
    frame: pd.DataFrame,
    interval: int,
    feature_pool: AssetStateFeaturePoolSpec,
) -> pd.DataFrame:
    output = frame[[ASSET_COLUMN, TIMESTAMP_COLUMN]].copy()
    selected: set[str] = set()
    for base in feature_pool.feature_bases:
        canonical = canonical_interval_feature_name(interval, base)
        source = canonical if canonical in frame.columns else base if base in frame.columns else None
        if source is None:
            continue
        output[canonical] = pd.to_numeric(frame[source], errors="coerce")
        selected.add(canonical)

    for target_source in ("close", "log_return"):
        canonical = canonical_interval_feature_name(interval, target_source)
        source = canonical if canonical in frame.columns else target_source if target_source in frame.columns else None
        if source is not None and canonical not in output.columns:
            output[canonical] = pd.to_numeric(frame[source], errors="coerce")

    if not selected:
        return output
    return output


def _merge_interval_frames(frames: Sequence[pd.DataFrame]) -> pd.DataFrame:
    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.merge(frame, on=[ASSET_COLUMN, TIMESTAMP_COLUMN], how="outer")
    return (
        merged.sort_values(TIMESTAMP_COLUMN)
        .drop_duplicates(subset=[ASSET_COLUMN, TIMESTAMP_COLUMN], keep="last")
        .reset_index(drop=True)
    )


def _filter_window(
    frame: pd.DataFrame,
    start_ts: int | None,
    end_ts: int | None,
) -> pd.DataFrame:
    if frame.empty or TIMESTAMP_COLUMN not in frame.columns:
        return frame
    mask = pd.Series(True, index=frame.index)
    if start_ts is not None:
        mask &= frame[TIMESTAMP_COLUMN].astype("int64") >= int(start_ts)
    if end_ts is not None:
        mask &= frame[TIMESTAMP_COLUMN].astype("int64") < int(end_ts)
    return frame.loc[mask].copy()


def _feature_columns_for_panel(
    panel: pd.DataFrame,
    feature_pool: AssetStateFeaturePoolSpec,
    member_intervals: Sequence[int],
) -> tuple[str, ...]:
    columns: list[str] = []
    for interval in member_intervals:
        for base in feature_pool.feature_bases:
            canonical = canonical_interval_feature_name(interval, base)
            if canonical in panel.columns:
                columns.append(canonical)
    return tuple(dict.fromkeys(columns))


def _split_panel(
    panel: pd.DataFrame,
    split: AssetStateSplitSpec,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    panel = panel.sort_values(TIMESTAMP_COLUMN).reset_index(drop=True)
    if split.uses_explicit_windows:
        train = _window_split(panel, split.train_start_ts, split.train_end_ts)
        validation = _window_split(panel, split.validation_start_ts, split.validation_end_ts)
        holdout = _window_split(panel, split.holdout_start_ts, split.holdout_end_ts)
        return train, validation, holdout

    n = int(panel.shape[0])
    total = split.train_fraction + split.validation_fraction + split.holdout_fraction
    train_n = int(math.floor(n * (split.train_fraction / total)))
    validation_n = int(math.floor(n * (split.validation_fraction / total)))
    if train_n >= n and n > 0:
        train_n = n - 1
    validation_start = train_n
    validation_end = min(n, validation_start + validation_n)
    train = panel.iloc[:train_n].copy()
    validation = panel.iloc[validation_start:validation_end].copy()
    holdout = panel.iloc[validation_end:].copy()
    return train, validation, holdout


def _window_split(
    panel: pd.DataFrame,
    start_ts: int | None,
    end_ts: int | None,
) -> pd.DataFrame:
    if start_ts is None or end_ts is None:
        return pd.DataFrame(columns=panel.columns)
    return _filter_window(panel, int(start_ts), int(end_ts))


def _compute_forward_targets(
    panel: pd.DataFrame,
    *,
    target_interval: int,
    horizon_min: int,
) -> pd.DataFrame:
    ordered = panel.sort_values(TIMESTAMP_COLUMN).reset_index(drop=True)
    result = ordered[[TIMESTAMP_COLUMN]].copy()
    target_name = f"forward_return_{int(horizon_min)}m"
    steps = max(1, int(round(float(horizon_min) / float(target_interval))))
    close_col = canonical_interval_feature_name(target_interval, "close")
    return_col = canonical_interval_feature_name(target_interval, "log_return")
    if close_col in ordered.columns:
        close = pd.to_numeric(ordered[close_col], errors="coerce").astype(float)
        close = close.where(close > 0.0, np.nan)
        result[target_name] = np.log(close.shift(-steps) / close)
    elif return_col in ordered.columns:
        returns = pd.to_numeric(ordered[return_col], errors="coerce").astype(float)
        forward = pd.Series(0.0, index=ordered.index)
        for i in range(1, steps + 1):
            forward = forward + returns.shift(-i)
        result[target_name] = forward
    else:
        result[target_name] = np.nan
    return result


def _align_targets(frame: pd.DataFrame, target_frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or target_frame.empty or TIMESTAMP_COLUMN not in frame.columns:
        return pd.DataFrame(columns=[TIMESTAMP_COLUMN, *[c for c in target_frame.columns if c != TIMESTAMP_COLUMN]])
    aligned = frame[[TIMESTAMP_COLUMN]].merge(target_frame, on=TIMESTAMP_COLUMN, how="left")
    return aligned.reset_index(drop=True)


def _missingness_summary(
    panel: pd.DataFrame,
    feature_columns: Sequence[str],
    *,
    train_frame: pd.DataFrame | None = None,
) -> dict[str, Any]:
    if not feature_columns:
        return {
            "feature_columns": [],
            "overall_missing_fraction": {},
            "train_missing_fraction": {},
        }
    overall = {
        col: float(panel[col].isna().mean()) if col in panel.columns and len(panel) else 1.0
        for col in feature_columns
    }
    train = {}
    if train_frame is not None:
        train = {
            col: float(train_frame[col].isna().mean()) if col in train_frame.columns and len(train_frame) else 1.0
            for col in feature_columns
        }
    return {
        "feature_columns": list(feature_columns),
        "overall_missing_fraction": overall,
        "train_missing_fraction": train,
        "overall_max_missing_fraction": max(overall.values()) if overall else None,
        "train_max_missing_fraction": max(train.values()) if train else None,
    }


def _leakage_columns(columns: Iterable[Any]) -> tuple[str, ...]:
    blocked: list[str] = []
    for col in columns:
        name = str(col).lower()
        if any(token in name for token in LEAKAGE_COLUMN_TOKENS):
            blocked.append(str(col))
    return tuple(blocked)


def _finite_matrix(x: np.ndarray) -> bool:
    return bool(np.isfinite(x).all())


def _timestamps_from_frame(frame: pd.DataFrame) -> tuple[int, ...]:
    if frame.empty or TIMESTAMP_COLUMN not in frame.columns:
        return ()
    return tuple(int(v) for v in frame[TIMESTAMP_COLUMN].tolist())


def _min_ts(frame: pd.DataFrame) -> int | None:
    if frame.empty or TIMESTAMP_COLUMN not in frame.columns:
        return None
    return int(frame[TIMESTAMP_COLUMN].min())


def _max_ts(frame: pd.DataFrame) -> int | None:
    if frame.empty or TIMESTAMP_COLUMN not in frame.columns:
        return None
    return int(frame[TIMESTAMP_COLUMN].max())


def _partition_value(path: Path, key: str) -> str | None:
    for part in path.parts:
        prefix = f"{key}="
        if part.startswith(prefix):
            return part[len(prefix) :]
    return None


def _partition_int_value(path: Path, key: str) -> int | None:
    value = _partition_value(path, key)
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _nearest_interval_root(path: Path, interval_roots: Sequence[Path]) -> Path:
    resolved = path.resolve()
    best = interval_roots[0] if interval_roots else path.parent
    best_len = -1
    for root in interval_roots:
        try:
            root_resolved = root.resolve()
        except FileNotFoundError:
            continue
        try:
            resolved.relative_to(root_resolved)
        except ValueError:
            continue
        size = len(root_resolved.parts)
        if size > best_len:
            best = root
            best_len = size
    return best


def _validate_diagnostics_root(request: AssetStateStudyDatasetRequest) -> Path:
    if request.diagnostics_root is None:
        raise ValueError("diagnostics_root is required when write_diagnostics=True")
    diagnostics_root = Path(request.diagnostics_root)
    if is_production_adjacent_path(diagnostics_root):
        raise ValueError(
            f"{DATASET_REASON_UNSAFE_DIAGNOSTICS_ROOT}: diagnostics root is production-adjacent: {diagnostics_root}"
        )
    safe_root = require_foundation_report_root(
        diagnostics_root,
        allow_foundation_descendant=True,
    )
    if "asset_state_test" not in {part.lower() for part in safe_root.parts}:
        raise ValueError(
            f"{DATASET_REASON_UNSAFE_DIAGNOSTICS_ROOT}: diagnostics root must be under asset_state_test foundation reports"
        )
    return safe_root


def _maybe_write_diagnostics(
    result: DatasetBuildResult,
    diagnostics_root: Path | None,
) -> DatasetBuildResult:
    if diagnostics_root is None:
        return result
    diagnostics_root.mkdir(parents=True, exist_ok=True)
    path = diagnostics_root / "asset_state_dataset_build_result.json"
    _write_json_atomic(path, result.as_dict())
    result.diagnostics_path = str(path)
    return result


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp, path)
