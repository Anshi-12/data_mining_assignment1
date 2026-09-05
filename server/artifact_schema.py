from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ArtifactLayout:
    """File contract planned for the real production scorer and retraining loop.

    The binary objects are intentionally separate from metadata so later training can
    update atomically and the API can validate schema/version compatibility.
    """

    root: Path
    manifest_name: str = "manifest.json"
    scaler_name: str = "robust_scaler.joblib"
    detector_name: str = "isolation_forest.joblib"
    calibration_name: str = "score_calibration.json"
    search_history_name: str = "search_history.json"
    benchmarks_name: str = "benchmarks.json"
    manifold_name: str = "manifold.json"
    top_anomalies_name: str = "top_anomalies.json"

    @property
    def manifest(self) -> Path:
        return self.root / self.manifest_name

    @property
    def scaler(self) -> Path:
        return self.root / self.scaler_name

    @property
    def detector(self) -> Path:
        return self.root / self.detector_name

    @property
    def calibration(self) -> Path:
        return self.root / self.calibration_name

    @property
    def search_history(self) -> Path:
        return self.root / self.search_history_name

    @property
    def benchmarks(self) -> Path:
        return self.root / self.benchmarks_name

    @property
    def manifold(self) -> Path:
        return self.root / self.manifold_name

    @property
    def top_anomalies(self) -> Path:
        return self.root / self.top_anomalies_name
