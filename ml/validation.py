"""Validation helpers for synthesized telemetry datasets."""

from __future__ import annotations

from dataclasses import dataclass

from ml.data_loader import (
    ANOMALY_ARCHETYPES,
    FEATURE_NAMES,
    TelemetryDataset,
    expected_archetype_counts,
)


@dataclass(frozen=True, slots=True)
class ValidationResult:
    valid: bool
    checks: tuple[str, ...]


def validate_dataset(dataset: TelemetryDataset) -> ValidationResult:
    """Validate the structural and contamination invariants for one dataset."""
    checks: list[str] = []
    rows = dataset.statistics.rows

    if dataset.features.shape != (rows, len(FEATURE_NAMES)):
        raise ValueError(
            f"Feature matrix shape {dataset.features.shape} does not match expected "
            f"({rows}, {len(FEATURE_NAMES)})."
        )
    checks.append(f"feature matrix shape = {dataset.features.shape}")

    if tuple(dataset.features.columns) != FEATURE_NAMES:
        raise ValueError("Feature columns do not match the canonical 10-feature schema.")
    checks.append("feature schema = canonical 10 telemetry features")

    if set(dataset.labels.columns) != {"is_anomaly", "archetype"}:
        raise ValueError("Ground-truth labels must remain structurally separate from features.")
    checks.append("ground-truth labels are separate from feature matrix")

    expected_anomalies = int(rows * dataset.requested_contamination)
    if dataset.statistics.anomaly_count != expected_anomalies:
        raise ValueError(
            f"Expected {expected_anomalies} anomalies but found {dataset.statistics.anomaly_count}."
        )
    checks.append(
        f"anomaly count = {dataset.statistics.anomaly_count} "
        f"({dataset.statistics.contamination:.2%})"
    )

    expected_counts = expected_archetype_counts(rows, dataset.requested_contamination)
    if dataset.statistics.archetype_counts != expected_counts:
        raise ValueError(
            "Anomaly archetype counts differ from the deterministic allocation rule: "
            f"expected {expected_counts}, found {dataset.statistics.archetype_counts}."
        )
    checks.append("all four anomaly archetypes match expected counts")

    if expected_anomalies >= 4:
        missing = [
            name
            for name in ANOMALY_ARCHETYPES
            if dataset.statistics.archetype_counts.get(name, 0) <= 0
        ]
        if missing:
            raise ValueError(f"Missing anomaly archetypes: {', '.join(missing)}")
        checks.append("all four anomaly archetypes are present")

    if dataset.features.isna().any().any():
        raise ValueError("Synthesized feature matrix unexpectedly contains missing values.")
    checks.append("feature matrix contains no missing values")

    return ValidationResult(valid=True, checks=tuple(checks))
