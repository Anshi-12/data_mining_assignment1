from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pytest

from ml.data_loader import FEATURE_NAMES, generate_anomaly_dataset
from ml.training import load_production_artifacts, score_fixed_vector, train_phase3


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    artifact_dir = tmp_path_factory.mktemp("phase3_artifacts")
    result = train_phase3(rows=1_200, seed=42, artifact_dir=artifact_dir, top_n=25)
    return result, Path(artifact_dir)


def test_scaler_is_fit_on_training_rows_only(trained):
    result, artifact_dir = trained
    dataset = generate_anomaly_dataset(n_samples=1_200, random_state=42)
    scaler = joblib.load(artifact_dir / "robust_scaler.joblib")

    train = dataset.features.iloc[list(result.split.training_indices)]
    expected_training_medians = np.median(train.to_numpy(dtype=float), axis=0)
    full_medians = np.median(dataset.features.to_numpy(dtype=float), axis=0)

    assert np.allclose(scaler.center_, expected_training_medians)
    # The proof would be vacuous if this particular split happened to have every
    # median identical to the full dataset. At least one learned center must differ.
    assert np.any(np.abs(scaler.center_ - full_medians) > 1e-9)


def test_labels_are_absent_from_all_fitting_inputs(trained):
    result, artifact_dir = trained
    manifest = json.loads((artifact_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["feature_order"] == list(FEATURE_NAMES)
    assert set(manifest["feature_order"]).isdisjoint({"is_anomaly", "archetype"})
    assert manifest["ground_truth_usage"].startswith("validation metrics")
    assert result.split.training_rows + result.split.validation_rows == 1_200


def test_all_five_detectors_produce_valid_metrics(trained):
    result, _ = trained
    assert {r.model_name for r in result.benchmark_results} == {
        "Isolation Forest",
        "Local Outlier Factor",
        "One-Class SVM",
        "Robust Mahalanobis",
        "NumPy Autoencoder",
    }
    for metric in result.benchmark_results:
        for value in (
            metric.roc_auc,
            metric.pr_auc,
            metric.precision,
            metric.recall,
            metric.f1,
        ):
            assert 0.0 <= value <= 1.0
        assert metric.fit_seconds >= 0.0
        assert metric.inference_ms_per_row >= 0.0


def test_search_history_contains_only_real_measured_runs(trained):
    result, artifact_dir = trained
    payload = json.loads((artifact_dir / "search_history.json").read_text(encoding="utf-8"))
    assert len(result.search_history) == 12
    assert len(payload["runs"]) == 12
    assert all(run["measured"] is True for run in payload["runs"])
    assert all(run["fit_seconds"] > 0.0 for run in payload["runs"])
    assert payload["selected_params"] == result.selected_params


def test_artifacts_round_trip_and_reproduce_fixed_vector_score(trained):
    _, artifact_dir = trained
    scaler_a, model_a, calibration_a, manifest = load_production_artifacts(artifact_dir)
    dataset = generate_anomaly_dataset(n_samples=1_200, random_state=42)
    vector = dataset.features.iloc[11].to_numpy(dtype=float)
    score_a = score_fixed_vector(
        vector,
        scaler=scaler_a,
        model=model_a,
        calibration_payload=calibration_a,
    )

    scaler_b, model_b, calibration_b, _ = load_production_artifacts(artifact_dir)
    score_b = score_fixed_vector(
        vector,
        scaler=scaler_b,
        model=model_b,
        calibration_payload=calibration_b,
    )
    assert np.allclose(score_a, score_b, atol=1e-12)
    assert manifest["feature_order"] == list(FEATURE_NAMES)


def test_expected_phase3_artifacts_exist(trained):
    _, artifact_dir = trained
    expected = {
        "robust_scaler.joblib",
        "isolation_forest.joblib",
        "score_calibration.json",
        "search_history.json",
        "manifest.json",
        "benchmarks.json",
        "manifold.json",
        "top_anomalies.json",
    }
    assert expected.issubset({p.name for p in artifact_dir.iterdir()})
