import numpy as np
import pandas as pd

from crispdm_studio.clustering import run_clustering
from crispdm_studio.ingestion import ingest_csv
from crispdm_studio.config import CONFIG
from crispdm_studio.preparation import prepare_dataset
from crispdm_studio.understanding import profile_dataset


def _understanding(df: pd.DataFrame):
    raw = df.to_csv(index=False).encode()
    ds = ingest_csv(filename="fixture.csv", mime_type="text/csv", data=raw, config=CONFIG)
    u = profile_dataset(ds)
    prep = prepare_dataset(ds.dataframe, u)
    return u, prep.structural_dataframe, prep


def test_well_separated_clusters_recovered():
    rng = np.random.default_rng(7)
    a = rng.normal(loc=(-4, -4), scale=0.35, size=(30, 2))
    b = rng.normal(loc=(4, 4), scale=0.35, size=(30, 2))
    df = pd.DataFrame(np.vstack([a, b]), columns=["x", "y"])
    u, structural, _ = _understanding(df)
    result = run_clustering(structural, u)
    assert not result.skipped
    assert result.chosen_k == 2
    assert result.silhouette > 0.8
    assert len(result.profiles) == 2
    assert sum(p.size for p in result.profiles) == len(structural)


def test_too_few_rows_skipped():
    df = pd.DataFrame({"x": [0, 1, 2, 3], "group": ["a", "a", "b", "b"]})
    u, structural, _ = _understanding(df)
    result = run_clustering(structural, u)
    assert result.skipped
    assert "at least" in result.selection_reason.lower()


def test_degenerate_features_skipped():
    # Constant columns are removed structurally, leaving no clustering features.
    df = pd.DataFrame({"constant": ["x"] * 10, "other": [1] * 10})
    u, structural, _ = _understanding(df)
    result = run_clustering(structural, u)
    assert result.skipped
    assert "no usable" in result.selection_reason.lower() or "zero usable" in result.selection_reason.lower()


def test_silhouette_selects_best_k():
    rng = np.random.default_rng(3)
    chunks = [rng.normal(loc=(v, v), scale=0.2, size=(25, 2)) for v in (-5, 0, 5)]
    df = pd.DataFrame(np.vstack(chunks), columns=["feature_a", "feature_b"])
    u, structural, _ = _understanding(df)
    result = run_clustering(structural, u, max_k=5)
    assert not result.skipped
    assert result.chosen_k == max(result.candidates, key=lambda c: c.silhouette).k
    assert result.chosen_k == 3


def test_segment_stats_and_descriptions_are_data_derived():
    rng = np.random.default_rng(10)
    low = pd.DataFrame({"spend": rng.normal(10, 0.3, 20), "visits": rng.normal(2, 0.1, 20), "tier": ["basic"] * 20})
    high = pd.DataFrame({"spend": rng.normal(100, 0.3, 20), "visits": rng.normal(10, 0.1, 20), "tier": ["premium"] * 20})
    df = pd.concat([low, high], ignore_index=True)
    u, structural, _ = _understanding(df)
    result = run_clustering(structural, u)
    assert not result.skipped
    assert len(result.segment_descriptions) == result.chosen_k
    medians = sorted(p.numeric_medians["spend"] for p in result.profiles)
    assert medians[0] < 20 and medians[-1] > 90
    texts = " ".join(d.text for d in result.segment_descriptions)
    assert "spend median" in texts


def test_supervised_transformer_remains_unfitted():
    rng = np.random.default_rng(2)
    df = pd.DataFrame({"x": np.r_[rng.normal(-3, .2, 15), rng.normal(3, .2, 15)], "cat": ["a"]*15 + ["b"]*15})
    u, structural, prep = _understanding(df)
    supervised = prep.preprocessing_spec.build_transformer()
    _ = run_clustering(structural, u)
    assert not hasattr(supervised, "transformers_")

def test_clustering_chart_reuses_shared_export_helper():
    from crispdm_studio.reporting.chart_export import chart_to_html
    rng = np.random.default_rng(12)
    df = pd.DataFrame({
        "a": np.r_[rng.normal(-3, .2, 20), rng.normal(3, .2, 20)],
        "b": np.r_[rng.normal(-3, .2, 20), rng.normal(3, .2, 20)],
    })
    u, structural, _ = _understanding(df)
    result = run_clustering(structural, u)
    assert result.charts
    assert "plotly" in chart_to_html(result.charts[0]).lower()
