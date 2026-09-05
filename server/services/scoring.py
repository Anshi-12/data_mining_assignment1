from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from server.contracts import FeatureAttribution, ScoreResult, TelemetryVector

API_TO_FEATURE = {
    "network_bytes_in": "NetworkBytesIn",
    "network_bytes_out": "NetworkBytesOut",
    "cpu_utilization": "CPUUtilization",
    "memory_pressure": "MemoryPressure",
    "latency_ms": "LatencyMs",
    "error_rate": "ErrorRate",
    "request_velocity": "RequestVelocity",
    "auth_failures": "AuthFailures",
    "entropy_score": "EntropyScore",
    "disk_iops": "DiskIOPS",
}
FEATURE_TO_API = {v: k for k, v in API_TO_FEATURE.items()}


@dataclass(slots=True)
class ProductionScorer:
    scaler: Any
    model: Any
    calibration: dict[str, Any]
    manifest: dict[str, Any]

    @property
    def name(self) -> str:
        return "persisted-robustscaler-isolationforest"

    def _vector(self, observation: TelemetryVector) -> np.ndarray:
        payload = observation.model_dump()
        ordered: list[float] = []
        for feature in self.manifest["feature_order"]:
            api_name = FEATURE_TO_API.get(feature)
            if api_name is None:
                raise ValueError(f"Manifest feature '{feature}' has no API field mapping.")
            ordered.append(float(payload[api_name]))
        return np.asarray(ordered, dtype=float)

    def score(self, observation: TelemetryVector) -> ScoreResult:
        raw_vector = self._vector(observation)
        scaled = self.scaler.transform(raw_vector.reshape(1, -1))
        raw_score = float(-self.model.decision_function(scaled)[0])
        threat_index = float(
            np.interp(
                raw_score,
                self.calibration["score_quantiles"],
                self.calibration["percentiles"],
                left=0.0,
                right=1.0,
            )
            * 100.0
        )
        raw_threshold = float(self.calibration["raw_threshold"])
        threat_threshold = float(self.calibration["threat_threshold"])
        flagged = bool(raw_score >= raw_threshold)

        centers = np.asarray(self.scaler.center_, dtype=float)
        scales = np.asarray(self.scaler.scale_, dtype=float)
        safe_scales = np.where(np.abs(scales) < 1e-12, 1.0, scales)
        deviations = np.abs(raw_vector - centers) / safe_scales
        attributions = [
            FeatureAttribution(
                feature=feature,
                value=float(raw_vector[i]),
                robust_center=float(centers[i]),
                robust_scale=float(scales[i]),
                iqr_deviation=float(deviations[i]),
            )
            for i, feature in sorted(
                enumerate(self.manifest["feature_order"]),
                key=lambda pair: float(deviations[pair[0]]),
                reverse=True,
            )
        ]
        return ScoreResult(
            raw_anomaly_score=raw_score,
            threat_index=threat_index,
            raw_threshold=raw_threshold,
            threat_threshold=threat_threshold,
            is_anomaly=flagged,
            verdict="above_threshold" if flagged else "below_threshold",
            scorer_version=str(self.manifest["schema_version"]),
            model_type=str(self.manifest["production_model"]),
            attribution=attributions,
        )
