from __future__ import annotations

from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field


class HealthResponse(BaseModel):
    status: str = "ok"
    service: str
    phase: int = 6
    ml_ready: bool
    ml_reason: str | None = None
    scorer: str
    model_type: str | None = None
    selected_hyperparameters: dict[str, Any] | None = None
    artifact_schema_version: str


class TelemetryVector(BaseModel):
    """One telemetry observation. Labels are intentionally absent."""

    model_config = ConfigDict(extra="forbid")

    network_bytes_in: float
    network_bytes_out: float
    cpu_utilization: float = Field(ge=0)
    memory_pressure: float = Field(ge=0)
    latency_ms: float = Field(ge=0)
    error_rate: float = Field(ge=0)
    request_velocity: float = Field(ge=0)
    auth_failures: float = Field(ge=0)
    entropy_score: float
    disk_iops: float = Field(ge=0)


class FeatureAttribution(BaseModel):
    feature: str
    value: float
    robust_center: float
    robust_scale: float
    iqr_deviation: float


class ScoreResult(BaseModel):
    raw_anomaly_score: float
    threat_index: float = Field(ge=0, le=100)
    raw_threshold: float
    threat_threshold: float
    is_anomaly: bool
    verdict: str
    scorer_version: str
    model_type: str
    attribution_method: str = "separate_robust_iqr_deviation"
    attribution: list[FeatureAttribution]


class RetrainRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    rows: int = Field(default=2_000, ge=500, le=5_000)
    contamination: float = Field(default=0.035, gt=0.0, lt=0.2)
    seed: int = 42
    validation_fraction: float = Field(default=0.25, ge=0.1, le=0.4)


class RetrainResponse(BaseModel):
    status: str
    rows_used: int
    validation_rows: int
    search_runs: int
    selected_params: dict[str, Any]
    selection_reason: str
    selected_metrics: dict[str, float]
    artifact_schema_version: str
    message: str


class Scorer(Protocol):
    @property
    def name(self) -> str: ...

    def score(self, observation: TelemetryVector) -> ScoreResult: ...
