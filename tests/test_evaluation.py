from __future__ import annotations

from dataclasses import replace
from functools import lru_cache
import numpy as np
import pandas as pd

from crispdm_studio.clustering import run_clustering
from crispdm_studio.config import CONFIG
from crispdm_studio.eda import run_eda
from crispdm_studio.evaluation import evaluate_modeling
from crispdm_studio.evaluation.metrics import baseline_improvements, reasoned_selection
from crispdm_studio.ingestion import ingest_csv
from crispdm_studio.modeling import run_modeling
from crispdm_studio.modeling.comparison import CVMetric
from crispdm_studio.preparation import prepare_dataset
from crispdm_studio.reporting.chart_export import chart_to_html
from crispdm_studio.understanding import profile_dataset
from crispdm_studio.understanding.quality import QualityWarning, WarningSeverity


def _context(df: pd.DataFrame, target: str):
    ds = ingest_csv(filename="eval.csv", mime_type="text/csv", data=df.to_csv(index=False).encode(), config=CONFIG)
    understanding = profile_dataset(ds)
    preparation = prepare_dataset(ds.dataframe, understanding)
    frame = preparation.structural_dataframe
    eda = run_eda(frame, understanding)
    clustering = run_clustering(frame, understanding)
    modeling = run_modeling(frame, understanding, preparation, target)
    return understanding, preparation, eda, clustering, modeling


@lru_cache(maxsize=1)
def _classification_context():
    rng = np.random.default_rng(12)
    n = 140
    x = rng.normal(size=n)
    z = rng.normal(size=n)
    label = (x + 0.35 * z + rng.normal(scale=0.25, size=n) > 0).astype(int)
    df = pd.DataFrame({"signal": x, "secondary": z, "group": np.where(z > 0, "east", "west"), "label": label})
    return _context(df, "label")


def test_baseline_improvement_math():
    *_, modeling = _classification_context()
    improvements = baseline_improvements(modeling)
    logistic = next(x for x in improvements if x.model_name == "Logistic Regression")
    baseline = modeling.baseline.test_metrics[modeling.primary_metric]
    model = next(m for m in modeling.model_results if m.name == "Logistic Regression")
    expected = model.test_metrics[modeling.primary_metric] - baseline
    assert np.isclose(logistic.absolute_improvement, expected)
    assert logistic.beats_baseline == (expected > 0)


def test_reasoned_selection_prefers_stable_model_over_flashy_unstable_one():
    *_, modeling = _classification_context()
    updated = []
    for model in modeling.model_results:
        if model.name == "Dummy baseline":
            updated.append(replace(model, test_metrics={**model.test_metrics, "balanced_accuracy": 0.50}, cv_metrics=(CVMetric("balanced_accuracy", 0.50, 0.01),)))
        elif model.name == "Logistic Regression":
            updated.append(replace(model, test_metrics={**model.test_metrics, "balanced_accuracy": 0.85}, cv_metrics=(CVMetric("balanced_accuracy", 0.83, 0.02),), overfitting_flag=False, overfitting_reason=None))
        elif model.name == "Random Forest":
            updated.append(replace(model, test_metrics={**model.test_metrics, "balanced_accuracy": 0.99}, cv_metrics=(CVMetric("balanced_accuracy", 0.55, 0.25),), overfitting_flag=True, overfitting_reason="large gap"))
        else:
            updated.append(replace(model, test_metrics={**model.test_metrics, "balanced_accuracy": 0.70}, cv_metrics=(CVMetric("balanced_accuracy", 0.68, 0.10),)))
    synthetic = replace(modeling, model_results=tuple(updated))
    selected, scores, _ = reasoned_selection(synthetic)
    assert selected is not None
    assert selected.name == "Logistic Regression"
    rf = next(x for x in scores if x.model_name == "Random Forest")
    logistic = next(x for x in scores if x.model_name == "Logistic Regression")
    assert logistic.composite_score > rf.composite_score


def test_permutation_importance_and_confusion_chart_present_without_refit():
    understanding, preparation, eda, clustering, modeling = _classification_context()
    before = next(m for m in modeling.model_results if m.name == "Logistic Regression").fitted_pipeline.named_steps["preprocess"].named_transformers_["numeric"].named_steps["imputer"].statistics_.copy()
    result = evaluate_modeling(modeling, eda, clustering, understanding)
    assert not result.skipped
    assert result.permutation_importance
    assert {x.feature for x in result.permutation_importance}.issubset(set(modeling.feature_columns))
    assert any(chart.chart_id == "evaluation:confusion_matrix" for chart in result.charts)
    assert any(chart.chart_id == "evaluation:permutation_importance" for chart in result.charts)
    assert "plotly" in chart_to_html(result.charts[0]).lower()
    after = next(m for m in modeling.model_results if m.name == "Logistic Regression").fitted_pipeline.named_steps["preprocess"].named_transformers_["numeric"].named_steps["imputer"].statistics_
    assert np.array_equal(before, after)
    original = preparation.preprocessing_spec.build_transformer()
    assert not hasattr(original, "transformers_")


def test_regression_has_residual_chart():
    rng = np.random.default_rng(13)
    n = 150
    x = rng.normal(size=n)
    y = 4 * x + rng.normal(scale=0.35, size=n)
    df = pd.DataFrame({"x": x, "z": rng.normal(size=n), "revenue": y})
    understanding, _, eda, clustering, modeling = _context(df, "revenue")
    result = evaluate_modeling(modeling, eda, clustering, understanding)
    assert not result.skipped
    assert any(chart.chart_id == "evaluation:residuals" for chart in result.charts)


def test_recommendation_reflects_quality_and_clustering_caveats():
    understanding, _, eda, clustering, modeling = _classification_context()
    warning = QualityWarning("fixture_warning", WarningSeverity.WARNING, "Fixture has material missingness that needs review.")
    understanding = replace(understanding, quality_warnings=(warning,))
    clustering = replace(clustering, status="skipped", chosen_k=None, silhouette=None, selection_reason="weak separation in fixture")
    result = evaluate_modeling(modeling, eda, clustering, understanding)
    text = result.recommendation.full_text.lower()
    assert "material missingness" in text
    assert "weak separation" in text
    assert "next" in text


def test_no_model_beats_baseline_returns_no_defensible_model():
    understanding, _, eda, clustering, modeling = _classification_context()
    updated = []
    for model in modeling.model_results:
        if model.is_baseline:
            updated.append(replace(model, test_metrics={**model.test_metrics, "balanced_accuracy": 0.80}, cv_metrics=(CVMetric("balanced_accuracy", 0.80, 0.01),)))
        else:
            updated.append(replace(model, test_metrics={**model.test_metrics, "balanced_accuracy": 0.70}, cv_metrics=(CVMetric("balanced_accuracy", 0.70, 0.03),)))
    modeling = replace(modeling, model_results=tuple(updated))
    result = evaluate_modeling(modeling, eda, clustering, understanding)
    assert result.skipped
    assert result.selected_model is None
    assert "dummy baseline" in result.recommendation.what_we_learned.lower()
