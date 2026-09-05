"""Deterministic locally synthesized server telemetry for anomaly detection.

This module intentionally reproduces the executable data-generation logic from the
reference project while making the feature/ground-truth boundary explicit.
Ground-truth columns are for evaluation only and must never be provided to
unsupervised detector ``fit`` calls.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
import pandas as pd

DEFAULT_ROWS: Final[int] = 10_000
DEFAULT_CONTAMINATION: Final[float] = 0.035
DEFAULT_SEED: Final[int] = 42

FEATURE_NAMES: Final[tuple[str, ...]] = (
    "NetworkBytesIn",
    "NetworkBytesOut",
    "CPUUtilization",
    "MemoryPressure",
    "LatencyMs",
    "ErrorRate",
    "RequestVelocity",
    "AuthFailures",
    "EntropyScore",
    "DiskIOPS",
)

LABEL_COLUMNS: Final[tuple[str, str]] = ("is_anomaly", "archetype")

FEATURE_API_FIELDS: Final[dict[str, str]] = {
    "NetworkBytesIn": "network_bytes_in",
    "NetworkBytesOut": "network_bytes_out",
    "CPUUtilization": "cpu_utilization",
    "MemoryPressure": "memory_pressure",
    "LatencyMs": "latency_ms",
    "ErrorRate": "error_rate",
    "RequestVelocity": "request_velocity",
    "AuthFailures": "auth_failures",
    "EntropyScore": "entropy_score",
    "DiskIOPS": "disk_iops",
}

FEATURE_INPUT_BOUNDS: Final[dict[str, dict[str, float]]] = {
    "NetworkBytesIn": {"min": 0.0, "max": 8_000_000.0, "step": 10_000.0},
    "NetworkBytesOut": {"min": 0.0, "max": 2_000_000.0, "step": 10_000.0},
    "CPUUtilization": {"min": 0.0, "max": 100.0, "step": 1.0},
    "MemoryPressure": {"min": 0.0, "max": 100.0, "step": 1.0},
    "LatencyMs": {"min": 0.0, "max": 8_500.0, "step": 10.0},
    "ErrorRate": {"min": 0.0, "max": 0.65, "step": 0.005},
    "RequestVelocity": {"min": 0.0, "max": 4_500.0, "step": 10.0},
    "AuthFailures": {"min": 0.0, "max": 100.0, "step": 1.0},
    "EntropyScore": {"min": 0.0, "max": 1.0, "step": 0.01},
    "DiskIOPS": {"min": 0.0, "max": 5_000.0, "step": 10.0},
}

NORMAL_ARCHETYPE: Final[str] = "Normal Operations"
ANOMALY_ARCHETYPES: Final[tuple[str, ...]] = (
    "Volumetric DDoS Attack",
    "Stealth Credential Infiltration",
    "Resource Exhaustion & Memory Leak",
    "Subspace Correlation Breakdown",
)

FEATURE_CATALOG: Final[tuple[dict[str, str], ...]] = (
    {
        "feature": "NetworkBytesIn",
        "unit": "Bytes/sec",
        "normal_range": "20K - 800K",
        "description": "Inbound network throughput",
    },
    {
        "feature": "NetworkBytesOut",
        "unit": "Bytes/sec",
        "normal_range": "15K - 1M",
        "description": "Outbound egress bandwidth",
    },
    {
        "feature": "CPUUtilization",
        "unit": "%",
        "normal_range": "5% - 75%",
        "description": "Server compute core load",
    },
    {
        "feature": "MemoryPressure",
        "unit": "%",
        "normal_range": "20% - 78%",
        "description": "RAM allocation percentage",
    },
    {
        "feature": "LatencyMs",
        "unit": "ms",
        "normal_range": "8ms - 150ms",
        "description": "End-to-end API response latency",
    },
    {
        "feature": "ErrorRate",
        "unit": "Ratio",
        "normal_range": "0.0 - 0.04",
        "description": "HTTP 5xx server fault frequency",
    },
    {
        "feature": "RequestVelocity",
        "unit": "Req/sec",
        "normal_range": "30 - 250",
        "description": "Client traffic request volume",
    },
    {
        "feature": "AuthFailures",
        "unit": "Count",
        "normal_range": "0 - 3",
        "description": "Failed authentication handshakes",
    },
    {
        "feature": "EntropyScore",
        "unit": "0.0 - 1.0",
        "normal_range": "0.15 - 0.70",
        "description": "Payload bitstream randomness",
    },
    {
        "feature": "DiskIOPS",
        "unit": "IOPS",
        "normal_range": "20 - 800",
        "description": "Block device read/write throughput",
    },
)


@dataclass(frozen=True, slots=True)
class DatasetStatistics:
    rows: int
    feature_count: int
    anomaly_count: int
    normal_count: int
    contamination: float
    archetype_counts: dict[str, int]
    archetype_shares_of_anomalies: dict[str, float]


@dataclass(frozen=True, slots=True)
class TelemetryDataset:
    """One generated dataset with a hard feature/ground-truth separation."""

    features: pd.DataFrame
    labels: pd.DataFrame
    catalog: tuple[dict[str, str], ...]
    statistics: DatasetStatistics
    seed: int
    requested_contamination: float

    def combined_frame(self) -> pd.DataFrame:
        """Return features plus labels for inspection/export, never detector fitting."""
        return pd.concat([self.features.copy(), self.labels.copy()], axis=1)


def expected_archetype_counts(n_samples: int, contamination: float) -> dict[str, int]:
    """Return the exact counts produced by the reference generator allocation rule."""
    n_anomalies = int(n_samples * contamination)
    n_per_arch = n_anomalies // 4
    # The reference implementation gives the fourth archetype the remaining count
    # after allocating three equal blocks.
    fourth_count = n_anomalies - (n_per_arch * 3)
    return {
        ANOMALY_ARCHETYPES[0]: n_per_arch,
        ANOMALY_ARCHETYPES[1]: n_per_arch,
        ANOMALY_ARCHETYPES[2]: n_per_arch,
        ANOMALY_ARCHETYPES[3]: fourth_count,
    }


def _validate_generation_request(n_samples: int, contamination: float) -> None:
    if n_samples < 1:
        raise ValueError("n_samples must be at least 1.")
    if not 0.0 <= contamination < 1.0:
        raise ValueError("contamination must be in the range [0.0, 1.0).")
    if 0 < int(n_samples * contamination) < 4:
        raise ValueError(
            "The requested dataset produces fewer than four anomaly rows; "
            "increase n_samples or contamination so all four archetypes can be represented."
        )


def generate_anomaly_dataset(
    n_samples: int = DEFAULT_ROWS,
    contamination: float = DEFAULT_CONTAMINATION,
    random_state: int = DEFAULT_SEED,
) -> TelemetryDataset:
    """Generate reproducible 10-D server telemetry and evaluation-only labels.

    The normal distributions and four anomaly archetypes mirror the reference
    executable generator. Data is synthesized locally; nothing is downloaded.
    """
    _validate_generation_request(n_samples, contamination)

    # RandomState deliberately mirrors np.random.seed + module-level draws in the
    # reference implementation while avoiding mutation of global RNG state.
    rng = np.random.RandomState(random_state)
    n_anomalies = int(n_samples * contamination)
    n_normal = n_samples - n_anomalies

    # Normal operations.
    bytes_in = np.clip(rng.lognormal(mean=11.5, sigma=0.6, size=n_normal), 20_000, 800_000)
    bytes_out = np.clip(bytes_in * rng.uniform(0.6, 1.4, size=n_normal), 15_000, 1_000_000)
    cpu_util = np.clip(rng.beta(a=2, b=5, size=n_normal) * 100.0, 5.0, 75.0)
    mem_press = np.clip(rng.beta(a=3, b=3, size=n_normal) * 100.0, 20.0, 78.0)
    latency = np.clip(rng.exponential(scale=35.0, size=n_normal) + 12.0, 8.0, 150.0)
    error_rate = np.clip(rng.exponential(scale=0.005, size=n_normal), 0.0, 0.04)
    req_velocity = np.clip(
        rng.poisson(lam=120, size=n_normal) + rng.normal(0, 15, size=n_normal),
        30,
        250,
    )
    auth_failures = rng.choice([0, 1, 2, 3], size=n_normal, p=[0.75, 0.18, 0.05, 0.02])
    entropy_score = np.clip(rng.normal(0.45, 0.10, size=n_normal), 0.15, 0.70)
    disk_iops = np.clip(cpu_util * 8.5 + rng.normal(50, 25, size=n_normal), 20, 800)

    normal_matrix = np.column_stack(
        [
            bytes_in,
            bytes_out,
            cpu_util,
            mem_press,
            latency,
            error_rate,
            req_velocity,
            auth_failures,
            entropy_score,
            disk_iops,
        ]
    )

    expected = expected_archetype_counts(n_samples, contamination)
    n_a1 = expected[ANOMALY_ARCHETYPES[0]]
    n_a2 = expected[ANOMALY_ARCHETYPES[1]]
    n_a3 = expected[ANOMALY_ARCHETYPES[2]]
    n_a4 = expected[ANOMALY_ARCHETYPES[3]]

    # 1. Volumetric DDoS.
    m_a1 = np.column_stack(
        [
            rng.uniform(2_500_000, 8_000_000, n_a1),
            rng.uniform(500_000, 2_000_000, n_a1),
            rng.uniform(85.0, 99.5, n_a1),
            rng.uniform(70.0, 92.0, n_a1),
            rng.uniform(450.0, 1_800.0, n_a1),
            rng.uniform(0.08, 0.35, n_a1),
            rng.uniform(1_200, 4_500, n_a1),
            rng.poisson(lam=5, size=n_a1),
            rng.uniform(0.80, 0.98, n_a1),
            rng.uniform(1_500, 4_000, n_a1),
        ]
    )

    # 2. Credential stuffing / stealth infiltration.
    m_a2 = np.column_stack(
        [
            rng.uniform(50_000, 200_000, n_a2),
            rng.uniform(30_000, 150_000, n_a2),
            rng.uniform(20.0, 45.0, n_a2),
            rng.uniform(40.0, 60.0, n_a2),
            rng.uniform(25.0, 80.0, n_a2),
            rng.uniform(0.02, 0.08, n_a2),
            rng.uniform(80, 200, n_a2),
            rng.uniform(25, 95, n_a2),
            rng.uniform(0.91, 0.99, n_a2),
            rng.uniform(50, 200, n_a2),
        ]
    )

    # 3. Resource exhaustion / memory leak.
    m_a3 = np.column_stack(
        [
            rng.uniform(80_000, 300_000, n_a3),
            rng.uniform(60_000, 250_000, n_a3),
            rng.uniform(92.0, 100.0, n_a3),
            rng.uniform(96.5, 99.9, n_a3),
            rng.uniform(2_800.0, 8_500.0, n_a3),
            rng.uniform(0.25, 0.65, n_a3),
            rng.uniform(40, 110, n_a3),
            rng.choice([0, 1], size=n_a3),
            rng.uniform(0.40, 0.65, n_a3),
            rng.uniform(2_200, 5_000, n_a3),
        ]
    )

    # 4. CPU-vs-Disk IOPS subspace correlation breakdown.
    m_a4 = np.column_stack(
        [
            rng.uniform(100_000, 400_000, n_a4),
            rng.uniform(80_000, 350_000, n_a4),
            rng.uniform(88.0, 98.0, n_a4),
            rng.uniform(30.0, 50.0, n_a4),
            rng.uniform(120.0, 350.0, n_a4),
            rng.uniform(0.01, 0.05, n_a4),
            rng.uniform(150, 300, n_a4),
            rng.choice([0, 1, 2], size=n_a4),
            rng.uniform(0.35, 0.55, n_a4),
            rng.uniform(0.0, 10.0, n_a4),
        ]
    )

    anomaly_matrix = np.vstack([m_a1, m_a2, m_a3, m_a4]) if n_anomalies else np.empty((0, 10))
    feature_matrix = np.vstack([normal_matrix, anomaly_matrix])
    anomaly_labels = np.concatenate(
        [np.zeros(n_normal, dtype=int), np.ones(n_anomalies, dtype=int)]
    )
    archetype_labels = (
        [NORMAL_ARCHETYPE] * n_normal
        + [ANOMALY_ARCHETYPES[0]] * n_a1
        + [ANOMALY_ARCHETYPES[1]] * n_a2
        + [ANOMALY_ARCHETYPES[2]] * n_a3
        + [ANOMALY_ARCHETYPES[3]] * n_a4
    )

    indices = np.arange(n_samples)
    rng.shuffle(indices)

    features = pd.DataFrame(feature_matrix[indices], columns=FEATURE_NAMES)
    labels = pd.DataFrame(
        {
            "is_anomaly": anomaly_labels[indices],
            "archetype": [archetype_labels[i] for i in indices],
        }
    )

    anomaly_count = int(labels["is_anomaly"].sum())
    archetype_counts = {
        name: int((labels["archetype"] == name).sum()) for name in ANOMALY_ARCHETYPES
    }
    statistics = DatasetStatistics(
        rows=n_samples,
        feature_count=len(FEATURE_NAMES),
        anomaly_count=anomaly_count,
        normal_count=n_samples - anomaly_count,
        contamination=(anomaly_count / n_samples),
        archetype_counts=archetype_counts,
        archetype_shares_of_anomalies={
            name: (count / anomaly_count if anomaly_count else 0.0)
            for name, count in archetype_counts.items()
        },
    )

    return TelemetryDataset(
        features=features,
        labels=labels,
        catalog=FEATURE_CATALOG,
        statistics=statistics,
        seed=random_state,
        requested_contamination=contamination,
    )
