from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

from src.regimes.asset_state_test.contracts import (
    ADAPTER_READY_METHODS,
    DEFAULT_ASSETS,
    DEFAULT_AXIS,
    DEFAULT_BAND,
    DEFAULT_FIRST_METHODS,
    STUDY_LAYER,
)


@dataclass(frozen=True)
class StudyManifest:
    path: Path
    layer: str
    axes: tuple[str, ...]
    bands: tuple[str, ...]
    first_cycle_axis: str
    first_cycle_band: str
    first_cycle_assets: tuple[str, ...]
    first_cycle_methods: tuple[str, ...]
    adapter_ready_methods: tuple[str, ...]
    raw_sha256: str

    def to_dict(self) -> dict[str, object]:
        return {
            "path": str(self.path),
            "layer": self.layer,
            "axes": list(self.axes),
            "bands": list(self.bands),
            "first_cycle_axis": self.first_cycle_axis,
            "first_cycle_band": self.first_cycle_band,
            "first_cycle_assets": list(self.first_cycle_assets),
            "first_cycle_methods": list(self.first_cycle_methods),
            "adapter_ready_methods": list(self.adapter_ready_methods),
            "raw_sha256": self.raw_sha256,
        }


_METHOD_PATTERNS = {
    "hdbscan": r"\bHDBSCAN\b",
    "kmeans": r"\bKMeans\b",
    "minibatch_kmeans": r"\bMiniBatchKMeans\b",
    "gaussian_mixture": r"\bGaussianMixture\b",
    "bayesian_gaussian_mixture": r"\bBayesianGaussianMixture\b",
    "optics": r"\bOPTICS\b",
    "agglomerative": r"\bAgglomerativeClustering\b",
    "birch": r"\bBirch\b",
}


def load_study_manifest(path: Path | str) -> StudyManifest:
    manifest_path = Path(path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Study manifest not found: {manifest_path}")
    text = manifest_path.read_text(encoding="utf-8")
    raw_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    layer = STUDY_LAYER if "asset_state" in text else STUDY_LAYER
    axes = tuple(axis for axis in ("trend", "vol", "activity") if re.search(rf"\b{axis}\b", text))
    bands = tuple(band for band in ("micro", "meso", "macro") if re.search(rf"\b{band}\b", text))
    assets = tuple(dict.fromkeys(re.findall(r"\b[A-Z0-9]{2,12}USD\b", text)))
    first_assets = tuple(asset for asset in assets if asset in {"AAVEUSD", "XBTUSD", "ADAUSD", "AI16ZUSD"}) or DEFAULT_ASSETS
    detected_methods = tuple(method for method, pattern in _METHOD_PATTERNS.items() if re.search(pattern, text))
    first_methods = tuple(method for method in DEFAULT_FIRST_METHODS if method in detected_methods) or DEFAULT_FIRST_METHODS
    return StudyManifest(
        path=manifest_path,
        layer=layer,
        axes=axes or ("trend", "vol", "activity"),
        bands=bands or ("micro", "meso", "macro"),
        first_cycle_axis=DEFAULT_AXIS,
        first_cycle_band=DEFAULT_BAND,
        first_cycle_assets=first_assets,
        first_cycle_methods=first_methods,
        adapter_ready_methods=tuple(method for method in ADAPTER_READY_METHODS if method in detected_methods) or ADAPTER_READY_METHODS,
        raw_sha256=raw_hash,
    )
