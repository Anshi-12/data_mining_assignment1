from __future__ import annotations

import pandas as pd

from ml.data_loader import (
    ANOMALY_ARCHETYPES,
    DEFAULT_CONTAMINATION,
    FEATURE_NAMES,
    generate_anomaly_dataset,
    expected_archetype_counts,
)
from ml.preprocessing import build_robust_scaler
from ml.validation import validate_dataset


def test_default_dataset_shape_and_label_separation() -> None:
    dataset = generate_anomaly_dataset()

    assert dataset.features.shape == (10_000, 10)
    assert dataset.labels.shape == (10_000, 2)
    assert tuple(dataset.features.columns) == FEATURE_NAMES
    assert "is_anomaly" not in dataset.features.columns
    assert "archetype" not in dataset.features.columns
    assert set(dataset.labels.columns) == {"is_anomaly", "archetype"}


def test_generation_is_deterministic_for_same_seed() -> None:
    left = generate_anomaly_dataset(n_samples=2_000, random_state=42)
    right = generate_anomaly_dataset(n_samples=2_000, random_state=42)

    pd.testing.assert_frame_equal(left.features, right.features)
    pd.testing.assert_frame_equal(left.labels, right.labels)


def test_expected_contamination_and_archetype_proportions() -> None:
    dataset = generate_anomaly_dataset(n_samples=10_000, contamination=DEFAULT_CONTAMINATION)
    expected_counts = expected_archetype_counts(10_000, DEFAULT_CONTAMINATION)

    assert dataset.statistics.anomaly_count == 350
    assert dataset.statistics.contamination == 0.035
    assert dataset.statistics.archetype_counts == expected_counts
    assert expected_counts == {
        ANOMALY_ARCHETYPES[0]: 87,
        ANOMALY_ARCHETYPES[1]: 87,
        ANOMALY_ARCHETYPES[2]: 87,
        ANOMALY_ARCHETYPES[3]: 89,
    }


def test_development_row_count_is_parameterized_and_validates() -> None:
    dataset = generate_anomaly_dataset(n_samples=2_000, contamination=0.035, random_state=42)
    result = validate_dataset(dataset)

    assert dataset.features.shape == (2_000, 10)
    assert dataset.statistics.anomaly_count == 70
    assert result.valid is True


def test_all_four_anomaly_archetypes_are_present() -> None:
    dataset = generate_anomaly_dataset(n_samples=2_000)
    observed = set(dataset.labels.loc[dataset.labels["is_anomaly"] == 1, "archetype"])

    assert observed == set(ANOMALY_ARCHETYPES)


def test_robust_scaler_is_returned_unfitted() -> None:
    scaler = build_robust_scaler()

    assert not hasattr(scaler, "center_")
    assert not hasattr(scaler, "scale_")


def test_feature_catalog_has_units_for_all_features() -> None:
    dataset = generate_anomaly_dataset(n_samples=2_000)

    assert [item["feature"] for item in dataset.catalog] == list(FEATURE_NAMES)
    assert all(item["unit"] for item in dataset.catalog)
    assert all(item["normal_range"] for item in dataset.catalog)
