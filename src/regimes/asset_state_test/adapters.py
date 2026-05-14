from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

import numpy as np
from sklearn.cluster import AgglomerativeClustering, Birch, KMeans, MiniBatchKMeans, OPTICS
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture

from src.regimes.asset_state_test.contracts import ADAPTER_READY_METHODS

try:
    import hdbscan  # type: ignore
except Exception:  # pragma: no cover - optional dependency guard
    hdbscan = None  # type: ignore


def _patch_hdbscan_check_array_compat() -> None:
    if hdbscan is None:
        return
    try:
        import inspect
        import sklearn.utils.validation as sklearn_validation

        hdbscan_module = hdbscan.hdbscan_  # type: ignore[attr-defined]
        current = getattr(hdbscan_module, "check_array", None)
        if current is None:
            return
        if "force_all_finite" in inspect.signature(current).parameters:
            return

        def check_array_compat(*args: Any, force_all_finite: Any = None, **kwargs: Any) -> Any:
            if force_all_finite is not None and "ensure_all_finite" not in kwargs:
                kwargs["ensure_all_finite"] = force_all_finite
            return sklearn_validation.check_array(*args, **kwargs)

        hdbscan_module.check_array = check_array_compat
    except Exception:
        return


@dataclass(frozen=True)
class ClusterFitResult:
    method: str
    labels: np.ndarray
    probabilities: Optional[np.ndarray]
    confidence: Optional[np.ndarray]
    noise_mask: np.ndarray
    fitted_metadata: Mapping[str, Any]
    method_params: Mapping[str, Any]
    runtime_stats: Mapping[str, Any]
    estimator: Any = field(repr=False, compare=False)

    @property
    def cluster_count(self) -> int:
        return int(len({int(v) for v in self.labels.tolist() if int(v) != -1}))


@dataclass(frozen=True)
class ClusterAssignmentResult:
    method: str
    labels: np.ndarray
    probabilities: Optional[np.ndarray]
    confidence: Optional[np.ndarray]
    noise_mask: np.ndarray
    status: str
    supported: bool
    error: Optional[str]
    metadata: Mapping[str, Any]

    @property
    def assigned_count(self) -> int:
        return int(np.asarray(self.labels, dtype=int).size)


class ClustererAdapter:
    method: str = "base"
    supports_predict: bool = False
    supports_probabilities: bool = False
    supports_noise: bool = False

    def __init__(self, **params: Any) -> None:
        self.params = dict(params)

    def fit(self, x: np.ndarray) -> ClusterFitResult:
        raise NotImplementedError

    def predict(self, x: np.ndarray) -> np.ndarray:
        if not hasattr(self, "estimator"):
            raise RuntimeError("Adapter has not been fitted")
        estimator = getattr(self, "estimator")
        if not hasattr(estimator, "predict"):
            raise NotImplementedError(f"{self.method} does not support predict")
        return np.asarray(estimator.predict(x), dtype=int)

    def assign(self, x: np.ndarray) -> np.ndarray:
        return self.predict(x)

    def assign_result(self, x: np.ndarray) -> ClusterAssignmentResult:
        try:
            labels = self.assign(x)
            probabilities = None
            estimator = getattr(self, "estimator", None)
            if estimator is not None and hasattr(estimator, "predict_proba"):
                probabilities = np.asarray(estimator.predict_proba(x), dtype=float)
            return _assignment_result(
                method=self.method,
                labels=labels,
                probabilities=probabilities,
                status="assigned",
                supported=True,
                error=None,
                metadata={"assignment_method": "predict"},
            )
        except NotImplementedError as exc:
            return _assignment_result(
                method=self.method,
                labels=np.empty(0, dtype=int),
                probabilities=None,
                status="unsupported",
                supported=False,
                error=str(exc),
                metadata={"assignment_method": "unsupported"},
            )
        except Exception as exc:
            return _assignment_result(
                method=self.method,
                labels=np.empty(0, dtype=int),
                probabilities=None,
                status="failed",
                supported=hasattr(getattr(self, "estimator", None), "predict"),
                error=str(exc),
                metadata={"assignment_method": "predict"},
            )


def _confidence_from_probabilities(probabilities: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if probabilities is None:
        return None
    arr = np.asarray(probabilities, dtype=float)
    if arr.ndim == 2:
        return np.nanmax(arr, axis=1)
    return arr


def _noise_mask(labels: np.ndarray) -> np.ndarray:
    return np.asarray(labels, dtype=int) == -1


def _assignment_result(
    *,
    method: str,
    labels: np.ndarray,
    probabilities: Optional[np.ndarray],
    status: str,
    supported: bool,
    error: Optional[str],
    metadata: Mapping[str, Any],
) -> ClusterAssignmentResult:
    labels = np.asarray(labels, dtype=int)
    probs = None if probabilities is None else np.asarray(probabilities, dtype=float)
    return ClusterAssignmentResult(
        method=str(method),
        labels=labels,
        probabilities=probs,
        confidence=_confidence_from_probabilities(probs),
        noise_mask=_noise_mask(labels),
        status=str(status),
        supported=bool(supported),
        error=error,
        metadata=dict(metadata),
    )


def _result(
    *,
    method: str,
    estimator: Any,
    labels: np.ndarray,
    probabilities: Optional[np.ndarray],
    params: Mapping[str, Any],
    started: float,
    extra_metadata: Optional[Mapping[str, Any]] = None,
) -> ClusterFitResult:
    labels = np.asarray(labels, dtype=int)
    probs = None if probabilities is None else np.asarray(probabilities, dtype=float)
    metadata = {
        "n_samples": int(labels.size),
        "cluster_count": int(len({int(v) for v in labels.tolist() if int(v) != -1})),
        "noise_count": int(np.sum(labels == -1)),
        "supports_predict": bool(hasattr(estimator, "predict")),
    }
    if extra_metadata:
        metadata.update(dict(extra_metadata))
    return ClusterFitResult(
        method=str(method),
        labels=labels,
        probabilities=probs,
        confidence=_confidence_from_probabilities(probs),
        noise_mask=_noise_mask(labels),
        fitted_metadata=metadata,
        method_params=dict(params),
        runtime_stats={"fit_elapsed_s": float(time.monotonic() - started)},
        estimator=estimator,
    )


class KMeansAdapter(ClustererAdapter):
    method = "kmeans"
    supports_predict = True

    def fit(self, x: np.ndarray) -> ClusterFitResult:
        started = time.monotonic()
        params = {"n_clusters": 3, "n_init": 10, "random_state": 17, **self.params}
        estimator = KMeans(**params)
        labels = estimator.fit_predict(x)
        self.estimator = estimator
        return _result(method=self.method, estimator=estimator, labels=labels, probabilities=None, params=params, started=started, extra_metadata={"inertia": float(estimator.inertia_)})


class MiniBatchKMeansAdapter(ClustererAdapter):
    method = "minibatch_kmeans"
    supports_predict = True

    def fit(self, x: np.ndarray) -> ClusterFitResult:
        started = time.monotonic()
        params = {"n_clusters": 3, "n_init": 10, "random_state": 17, "batch_size": 256, **self.params}
        estimator = MiniBatchKMeans(**params)
        labels = estimator.fit_predict(x)
        self.estimator = estimator
        return _result(method=self.method, estimator=estimator, labels=labels, probabilities=None, params=params, started=started, extra_metadata={"inertia": float(estimator.inertia_)})


class GaussianMixtureAdapter(ClustererAdapter):
    method = "gaussian_mixture"
    supports_predict = True
    supports_probabilities = True

    def fit(self, x: np.ndarray) -> ClusterFitResult:
        started = time.monotonic()
        params = {"n_components": 3, "covariance_type": "full", "random_state": 17, **self.params}
        estimator = GaussianMixture(**params)
        estimator.fit(x)
        labels = estimator.predict(x)
        probabilities = estimator.predict_proba(x)
        self.estimator = estimator
        return _result(
            method=self.method,
            estimator=estimator,
            labels=labels,
            probabilities=probabilities,
            params=params,
            started=started,
            extra_metadata={"bic": float(estimator.bic(x)), "aic": float(estimator.aic(x))},
        )


class BayesianGaussianMixtureAdapter(ClustererAdapter):
    method = "bayesian_gaussian_mixture"
    supports_predict = True
    supports_probabilities = True

    def fit(self, x: np.ndarray) -> ClusterFitResult:
        started = time.monotonic()
        params = {"n_components": 6, "covariance_type": "full", "random_state": 17, **self.params}
        estimator = BayesianGaussianMixture(**params)
        estimator.fit(x)
        labels = estimator.predict(x)
        probabilities = estimator.predict_proba(x)
        self.estimator = estimator
        return _result(method=self.method, estimator=estimator, labels=labels, probabilities=probabilities, params=params, started=started)


class HDBSCANAdapter(ClustererAdapter):
    method = "hdbscan"
    supports_probabilities = True
    supports_noise = True

    def fit(self, x: np.ndarray) -> ClusterFitResult:
        if hdbscan is None:
            raise RuntimeError("hdbscan is required for the HDBSCAN adapter")
        _patch_hdbscan_check_array_compat()
        started = time.monotonic()
        params = {
            "min_cluster_size": 5,
            "min_samples": 1,
            "allow_single_cluster": True,
            "prediction_data": True,
            "cluster_selection_method": "eom",
            **self.params,
        }
        estimator = hdbscan.HDBSCAN(**params)
        labels = estimator.fit_predict(x)
        probabilities = getattr(estimator, "probabilities_", None)
        persistence = getattr(estimator, "cluster_persistence_", None)
        self.estimator = estimator
        return _result(
            method=self.method,
            estimator=estimator,
            labels=labels,
            probabilities=probabilities,
            params=params,
            started=started,
            extra_metadata={"cluster_persistence": [] if persistence is None else [float(v) for v in persistence]},
        )

    def assign_result(self, x: np.ndarray) -> ClusterAssignmentResult:
        if hdbscan is None:
            return _assignment_result(
                method=self.method,
                labels=np.empty(0, dtype=int),
                probabilities=None,
                status="unsupported",
                supported=False,
                error="hdbscan is not installed",
                metadata={"assignment_method": "hdbscan.approximate_predict"},
            )
        estimator = getattr(self, "estimator", None)
        if estimator is None:
            return _assignment_result(
                method=self.method,
                labels=np.empty(0, dtype=int),
                probabilities=None,
                status="failed",
                supported=True,
                error="Adapter has not been fitted",
                metadata={"assignment_method": "hdbscan.approximate_predict"},
            )
        approximate_predict = getattr(hdbscan, "approximate_predict", None)
        if approximate_predict is None:
            return _assignment_result(
                method=self.method,
                labels=np.empty(0, dtype=int),
                probabilities=None,
                status="unsupported",
                supported=False,
                error="hdbscan.approximate_predict is unavailable",
                metadata={"assignment_method": "hdbscan.approximate_predict"},
            )
        try:
            labels, strengths = approximate_predict(estimator, x)
            return _assignment_result(
                method=self.method,
                labels=np.asarray(labels, dtype=int),
                probabilities=np.asarray(strengths, dtype=float),
                status="assigned",
                supported=True,
                error=None,
                metadata={"assignment_method": "hdbscan.approximate_predict"},
            )
        except Exception as exc:
            return _assignment_result(
                method=self.method,
                labels=np.empty(0, dtype=int),
                probabilities=None,
                status="failed",
                supported=True,
                error=str(exc),
                metadata={"assignment_method": "hdbscan.approximate_predict"},
            )


class OPTICSAdapter(ClustererAdapter):
    method = "optics"
    supports_noise = True

    def fit(self, x: np.ndarray) -> ClusterFitResult:
        started = time.monotonic()
        params = {"min_samples": 5, "xi": 0.05, **self.params}
        estimator = OPTICS(**params)
        labels = estimator.fit_predict(x)
        reachability = getattr(estimator, "reachability_", None)
        self.estimator = estimator
        metadata = {}
        if reachability is not None:
            finite = np.asarray(reachability, dtype=float)
            finite = finite[np.isfinite(finite)]
            metadata["reachability_mean"] = float(finite.mean()) if finite.size else None
            metadata["reachability_p90"] = float(np.quantile(finite, 0.9)) if finite.size else None
        return _result(method=self.method, estimator=estimator, labels=labels, probabilities=None, params=params, started=started, extra_metadata=metadata)


class AgglomerativeAdapter(ClustererAdapter):
    method = "agglomerative"

    def fit(self, x: np.ndarray) -> ClusterFitResult:
        started = time.monotonic()
        params = {"n_clusters": 3, **self.params}
        estimator = AgglomerativeClustering(**params)
        labels = estimator.fit_predict(x)
        self.estimator = estimator
        return _result(method=self.method, estimator=estimator, labels=labels, probabilities=None, params=params, started=started)


class BirchAdapter(ClustererAdapter):
    method = "birch"
    supports_predict = True

    def fit(self, x: np.ndarray) -> ClusterFitResult:
        started = time.monotonic()
        params = {"n_clusters": 3, **self.params}
        estimator = Birch(**params)
        labels = estimator.fit_predict(x)
        self.estimator = estimator
        return _result(method=self.method, estimator=estimator, labels=labels, probabilities=None, params=params, started=started)


_ADAPTERS: dict[str, type[ClustererAdapter]] = {
    "hdbscan": HDBSCANAdapter,
    "kmeans": KMeansAdapter,
    "minibatch_kmeans": MiniBatchKMeansAdapter,
    "gaussian_mixture": GaussianMixtureAdapter,
    "bayesian_gaussian_mixture": BayesianGaussianMixtureAdapter,
    "optics": OPTICSAdapter,
    "agglomerative": AgglomerativeAdapter,
    "birch": BirchAdapter,
}


def supported_methods() -> tuple[str, ...]:
    return ADAPTER_READY_METHODS


def build_clusterer_adapter(method: str, **params: Any) -> ClustererAdapter:
    key = str(method).strip().lower()
    try:
        cls = _ADAPTERS[key]
    except KeyError as exc:
        raise ValueError(f"Unsupported clustering method {method!r}; expected one of {tuple(_ADAPTERS)}") from exc
    return cls(**params)
