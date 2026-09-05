from __future__ import annotations

import numpy as np
import pandas as pd

from crispdm_studio.config import CONFIG
from crispdm_studio.ingestion import ingest_csv
from crispdm_studio.modeling import run_modeling
from crispdm_studio.preparation import prepare_dataset
from crispdm_studio.understanding import profile_dataset
from crispdm_studio.understanding.task_inference import TaskType


def _inputs(df: pd.DataFrame):
    ds = ingest_csv(filename="model.csv", mime_type="text/csv", data=df.to_csv(index=False).encode(), config=CONFIG)
    understanding = profile_dataset(ds)
    preparation = prepare_dataset(ds.dataframe, understanding)
    return understanding, preparation


def test_classification_path_and_baseline_present():
    rng = np.random.default_rng(1)
    n = 100
    x = rng.normal(size=n)
    df = pd.DataFrame({"feature": x, "group": np.where(x > 0, "east", "west"), "churn": (x > 0.2).astype(int)})
    understanding, preparation = _inputs(df)
    result = run_modeling(preparation.structural_dataframe, understanding, preparation, "churn")

    assert not result.skipped
    assert result.task_type is TaskType.CLASSIFICATION
    assert result.baseline is not None
    assert {m.name for m in result.model_results} == {"Dummy baseline", "Logistic Regression", "Random Forest", "HistGradientBoosting"}
    assert result.split is not None and result.split.stratified
    assert "balanced_accuracy" in result.model_results[1].test_metrics


def test_regression_path_and_required_metrics():
    rng = np.random.default_rng(2)
    n = 110
    x = rng.normal(size=n)
    y = 3.5 * x + rng.normal(scale=0.25, size=n)
    df = pd.DataFrame({"feature": x, "segment": np.where(x > 0, "a", "b"), "revenue": y})
    understanding, preparation = _inputs(df)
    result = run_modeling(preparation.structural_dataframe, understanding, preparation, "revenue")

    assert not result.skipped
    assert result.task_type is TaskType.REGRESSION
    assert result.baseline is not None
    assert {m.name for m in result.model_results} == {"Dummy baseline", "Ridge", "Random Forest", "HistGradientBoosting"}
    for model in result.model_results:
        assert {"mae", "rmse", "r2"}.issubset(model.test_metrics)


def test_cross_validation_reports_mean_and_std():
    rng = np.random.default_rng(3)
    n = 100
    x = rng.normal(size=n)
    df = pd.DataFrame({"x": x, "z": rng.normal(size=n), "label": np.where(x + rng.normal(scale=.3, size=n) > 0, "yes", "no")})
    understanding, preparation = _inputs(df)
    result = run_modeling(preparation.structural_dataframe, understanding, preparation, "label")

    assert result.cv_folds >= 2
    assert all(model.cv_metrics for model in result.model_results)
    metric = next(m for m in result.model_results[0].cv_metrics if m.name == "accuracy")
    assert np.isfinite(metric.mean)
    assert metric.std >= 0


def test_preprocessing_statistics_are_fit_on_training_rows_only():
    rng = np.random.default_rng(9)
    n = 120
    feature = rng.normal(loc=20, scale=7, size=n)
    feature[[5, 17, 73]] = np.nan
    label = np.array([0, 1] * (n // 2))
    df = pd.DataFrame({"measure": feature, "category": np.where(np.arange(n) % 3 == 0, "a", "b"), "label": label})
    understanding, preparation = _inputs(df)

    original_transformer = preparation.preprocessing_spec.build_transformer()
    assert not hasattr(original_transformer, "transformers_")

    result = run_modeling(preparation.structural_dataframe, understanding, preparation, "label")
    assert not result.skipped and result.split is not None
    fitted = result.model_results[1].fitted_pipeline.named_steps["preprocess"]
    numeric_pipe = fitted.named_transformers_["numeric"]
    learned_median = float(numeric_pipe.named_steps["imputer"].statistics_[0])
    expected_train_median = float(preparation.structural_dataframe.loc[list(result.split.train_indices), "measure"].median())

    assert learned_median == expected_train_median
    assert not hasattr(original_transformer, "transformers_")
    assert "label" not in result.feature_columns


def test_original_phase3_transformer_remains_unfitted_after_modeling():
    rng = np.random.default_rng(4)
    n = 80
    df = pd.DataFrame({"x": rng.normal(size=n), "label": [0, 1] * 40})
    understanding, preparation = _inputs(df)
    original = preparation.preprocessing_spec.build_transformer()
    _ = run_modeling(preparation.structural_dataframe, understanding, preparation, "label")
    assert not hasattr(original, "transformers_")


def test_no_confirmed_target_skips_cleanly():
    df = pd.DataFrame({"x": np.linspace(0, 1, 30), "group": ["a", "b"] * 15})
    understanding, preparation = _inputs(df)
    result = run_modeling(preparation.structural_dataframe, understanding, preparation, None)
    assert result.skipped
    assert "explicit" in result.skip_reason.lower() or "confirmed" in result.skip_reason.lower()
    assert not result.model_results
