from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from hashlib import sha256
from time import perf_counter
from typing import Any, Mapping

import numpy as np
from sklearn.cluster import OPTICS, cluster_optics_dbscan, cluster_optics_xi


OPTICS_REUSE_SCHEMA_VERSION = 1

_ASSIGNMENT_KEYS = {
    "assignment_k",
    "assignment_threshold_quantile",
    "assignment_threshold_multiplier",
    "unknown_label",
    "exclude_noise_from_assignment",
}
_EXTRACTION_KEYS = {
    "cluster_method",
    "eps",
    "xi",
    "predecessor_correction",
    "min_cluster_size",
}
_BASE_KEYS = {
    "algorithm",
    "leaf_size",
    "max_eps",
    "metric",
    "metric_params",
    "min_samples",
    "n_jobs",
    "p",
}


@dataclass(frozen=True)
class OpticsBaseFitKey:
    x_hash: str
    base_parameters: tuple[tuple[str, Any], ...]
    source_tail_ts: str | None = None
    prepared_context_key: str | None = None
    schema_version: int = OPTICS_REUSE_SCHEMA_VERSION

    def stable_id(self) -> str:
        payload = repr((self.schema_version, self.x_hash, self.base_parameters, self.source_tail_ts, self.prepared_context_key))
        return sha256(payload.encode("utf-8")).hexdigest()


@dataclass
class OpticsBaseFit:
    key: OpticsBaseFitKey
    estimator: OPTICS
    fit_elapsed_s: float
    row_count: int


_BASE_FIT_CACHE: "OrderedDict[str, OpticsBaseFit]" = OrderedDict()


def optics_base_parameters(params: Mapping[str, Any]) -> dict[str, Any]:
    """Return parameters that change the OPTICS reachability/order fit."""
    base: dict[str, Any] = {"min_samples": int(params.get("min_samples", 5))}
    for key in sorted(_BASE_KEYS):
        if key in params:
            base[key] = params[key]
    return base


def optics_extraction_parameters(params: Mapping[str, Any]) -> dict[str, Any]:
    """Return parameters that only extract labels from an existing OPTICS base fit."""
    extraction: dict[str, Any] = {
        "cluster_method": str(params.get("cluster_method", "xi")),
        "xi": float(params.get("xi", 0.05)),
        "predecessor_correction": bool(params.get("predecessor_correction", True)),
    }
    if "eps" in params:
        extraction["eps"] = float(params["eps"])
    if "min_cluster_size" in params:
        extraction["min_cluster_size"] = params["min_cluster_size"]
    return extraction


def optics_assignment_parameters(params: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): params[key] for key in sorted(_ASSIGNMENT_KEYS) if key in params}


def optics_matrix_hash(x: np.ndarray) -> str:
    arr = np.ascontiguousarray(np.asarray(x, dtype=float))
    digest = sha256()
    digest.update(str(arr.shape).encode("ascii"))
    digest.update(arr.dtype.str.encode("ascii"))
    digest.update(arr.view(np.uint8))
    return digest.hexdigest()


def build_optics_base_fit_key(
    x: np.ndarray,
    params: Mapping[str, Any],
    *,
    source_tail_ts: str | None = None,
    prepared_context_key: str | None = None,
) -> OpticsBaseFitKey:
    base = optics_base_parameters(params)
    return OpticsBaseFitKey(
        x_hash=optics_matrix_hash(x),
        base_parameters=tuple(sorted(base.items())),
        source_tail_ts=source_tail_ts,
        prepared_context_key=prepared_context_key,
    )


def fit_optics_base(
    x: np.ndarray,
    params: Mapping[str, Any],
    *,
    source_tail_ts: str | None = None,
    prepared_context_key: str | None = None,
) -> OpticsBaseFit:
    arr = np.ascontiguousarray(np.asarray(x, dtype=float))
    base_params = optics_base_parameters(params)
    key = build_optics_base_fit_key(
        arr,
        params,
        source_tail_ts=source_tail_ts,
        prepared_context_key=prepared_context_key,
    )
    started = perf_counter()
    estimator = OPTICS(**base_params)
    estimator.fit(arr)
    return OpticsBaseFit(
        key=key,
        estimator=estimator,
        fit_elapsed_s=float(perf_counter() - started),
        row_count=int(arr.shape[0]),
    )


def fit_optics_base_cached(
    x: np.ndarray,
    params: Mapping[str, Any],
    *,
    source_tail_ts: str | None = None,
    prepared_context_key: str | None = None,
    max_entries: int = 32,
) -> tuple[OpticsBaseFit, bool]:
    arr = np.ascontiguousarray(np.asarray(x, dtype=float))
    key = build_optics_base_fit_key(
        arr,
        params,
        source_tail_ts=source_tail_ts,
        prepared_context_key=prepared_context_key,
    )
    cache_id = key.stable_id()
    cached = _BASE_FIT_CACHE.get(cache_id)
    if cached is not None:
        _BASE_FIT_CACHE.move_to_end(cache_id)
        return cached, True
    base_fit = fit_optics_base(
        arr,
        params,
        source_tail_ts=source_tail_ts,
        prepared_context_key=prepared_context_key,
    )
    _BASE_FIT_CACHE[cache_id] = base_fit
    _BASE_FIT_CACHE.move_to_end(cache_id)
    while len(_BASE_FIT_CACHE) > int(max_entries):
        _BASE_FIT_CACHE.popitem(last=False)
    return base_fit, False


def clear_optics_base_fit_cache() -> None:
    _BASE_FIT_CACHE.clear()


def extract_optics_labels(base_fit: OpticsBaseFit, params: Mapping[str, Any]) -> np.ndarray:
    extraction = optics_extraction_parameters(params)
    method = str(extraction.get("cluster_method", "xi"))
    estimator = base_fit.estimator
    if method == "dbscan":
        eps = float(extraction.get("eps", np.inf))
        labels = cluster_optics_dbscan(
            reachability=estimator.reachability_,
            core_distances=estimator.core_distances_,
            ordering=estimator.ordering_,
            eps=eps,
        )
        return np.asarray(labels, dtype=int)
    if method == "xi":
        labels, _clusters = cluster_optics_xi(
            reachability=estimator.reachability_,
            predecessor=estimator.predecessor_,
            ordering=estimator.ordering_,
            min_samples=optics_base_parameters(dict(base_fit.key.base_parameters))["min_samples"],
            min_cluster_size=extraction.get("min_cluster_size"),
            xi=float(extraction.get("xi", 0.05)),
            predecessor_correction=bool(extraction.get("predecessor_correction", True)),
        )
        return np.asarray(labels, dtype=int)
    raise ValueError(f"Unsupported OPTICS extraction method {method!r}")


def fit_extract_optics_labels(x: np.ndarray, params: Mapping[str, Any]) -> np.ndarray:
    return extract_optics_labels(fit_optics_base(x, params), params)


def optics_reuse_boundaries() -> dict[str, Any]:
    return {
        "schema_version": OPTICS_REUSE_SCHEMA_VERSION,
        "base_fit_parameters": sorted(_BASE_KEYS),
        "extraction_parameters": sorted(_EXTRACTION_KEYS),
        "assignment_parameters": sorted(_ASSIGNMENT_KEYS),
        "reuse_semantics": "exact only when matrix/prepared context/source tail and base-fit parameters match",
    }


__all__ = [
    "OPTICS_REUSE_SCHEMA_VERSION",
    "OpticsBaseFit",
    "OpticsBaseFitKey",
    "build_optics_base_fit_key",
    "clear_optics_base_fit_cache",
    "extract_optics_labels",
    "fit_extract_optics_labels",
    "fit_optics_base",
    "fit_optics_base_cached",
    "optics_assignment_parameters",
    "optics_base_parameters",
    "optics_extraction_parameters",
    "optics_matrix_hash",
    "optics_reuse_boundaries",
]
