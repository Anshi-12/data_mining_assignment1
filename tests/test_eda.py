from __future__ import annotations

import numpy as np
import pandas as pd

from crispdm_studio.eda import FindingKind, run_eda
from crispdm_studio.models import DatasetOverview, IngestedDataset, UploadMetadata
from crispdm_studio.preparation import prepare_dataset
from crispdm_studio.reporting.chart_export import chart_to_html
from crispdm_studio.understanding import profile_dataset


def _understanding(frame: pd.DataFrame):
    ingested = IngestedDataset(
        dataframe=frame,
        metadata=UploadMetadata("fixture.csv", "text/csv", 100, "utf-8", ","),
        overview=DatasetOverview(len(frame), len(frame.columns), 0, 0, 0, 0, 0, 0, 0),
    )
    return profile_dataset(ingested)


def _eda(frame: pd.DataFrame):
    understanding = _understanding(frame)
    preparation = prepare_dataset(frame, understanding)
    return run_eda(preparation.structural_dataframe, understanding)


def test_numeric_only_adapts_and_computes_stats_correlations_outliers():
    frame = pd.DataFrame(
        {
            "x": np.arange(1, 31, dtype=float) ** 2,
            "y": (np.arange(1, 31, dtype=float) ** 2) * 2,
            "skewed": [1.0] * 20 + list(np.linspace(2, 20, 10)),
        }
    )
    result = _eda(frame)

    assert set(result.numeric_columns) == {"x", "y", "skewed"}
    assert result.categorical_columns == ()
    assert result.correlations
    assert result.correlations[0].correlation > 0.99
    skewed = next(item for item in result.numeric_distributions if item.column == "skewed")
    assert skewed.skewness is not None and skewed.skewness > 0
    assert skewed.outlier_count > 0
    assert any(item.analysis == "Categorical distributions" for item in result.skipped_analyses)


def test_categorical_only_adapts_and_computes_imbalance_and_association():
    frame = pd.DataFrame(
        {
            "segment": ["A"] * 16 + ["B"] * 4,
            "outcome": ["yes"] * 14 + ["no"] * 2 + ["no"] * 4,
            "batch": [f"b{i % 10}" for i in range(20)],
            "wave": ["w0"] * 10 + ["w1"] * 10,
        }
    )
    result = _eda(frame)

    assert result.numeric_columns == ()
    assert {"segment", "outcome"}.issubset(set(result.categorical_columns))
    segment = next(item for item in result.categorical_distributions if item.column == "segment")
    assert segment.dominant_share == 0.8
    assert result.categorical_associations
    assert any(item.analysis == "Numeric distributions" for item in result.skipped_analyses)


def test_mixed_data_computes_numeric_categorical_relationship_and_reusable_chart():
    frame = pd.DataFrame(
        {
            "amount": [10, 11, 12, 13, 14, 40, 41, 42, 43, 44, 45, 46],
            "group": ["low"] * 5 + ["high"] * 7,
            "flag": ["n", "n", "n", "y", "n", "y", "y", "y", "y", "y", "n", "y"],
        }
    )
    result = _eda(frame)

    assert result.numeric_categorical_relationships
    strongest = result.numeric_categorical_relationships[0]
    assert strongest.eta_squared > 0.5
    assert any(f.kind is FindingKind.NUMERIC_CATEGORICAL for f in result.findings)
    assert result.charts
    fragment = chart_to_html(result.charts[0])
    assert "plotly" in fragment.lower()
    assert result.charts[0].to_plotly_json()["data"]


def test_high_missingness_is_reported_without_imputation():
    frame = pd.DataFrame(
        {
            "value": [1.0, None, None, None, 5.0, None, None, 8.0, None, None],
            "group": ["a", "a", None, "b", None, None, "a", "b", None, None],
            "row_token": [f"r{i}" for i in range(10)],
        }
    )
    understanding = _understanding(frame)
    preparation = prepare_dataset(frame, understanding)
    assert preparation.structural_dataframe.isna().sum().sum() > 0

    result = run_eda(preparation.structural_dataframe, understanding)
    missing = {column: ratio for column, _, ratio in result.missingness}
    assert missing["value"] == 0.7
    assert missing["group"] == 0.5
    assert any(f.kind is FindingKind.MISSINGNESS for f in result.findings)
    assert any(chart.section == "Missingness" for chart in result.charts)


def test_findings_contain_computed_numeric_values_not_fixed_text():
    frame = pd.DataFrame({"income": [1, 1, 1, 1, 2, 3, 10, 20, 50, 100], "group": ["a", "b"] * 5})
    result = _eda(frame)
    summary = next(item for item in result.numeric_distributions if item.column == "income")
    finding = next(item for item in result.findings if item.finding_id == "numeric:income")

    assert summary.skewness is not None
    assert f"{summary.skewness:.2f}" in finding.text
    assert f"{summary.q95:,.4g}" in finding.text


def test_constant_columns_removed_before_eda_do_not_generate_fake_analysis():
    frame = pd.DataFrame({"constant": [1] * 12, "category": ["a", "b"] * 6})
    understanding = _understanding(frame)
    preparation = prepare_dataset(frame, understanding)
    assert "constant" not in preparation.structural_dataframe.columns
    result = run_eda(preparation.structural_dataframe, understanding)
    assert "constant" not in result.numeric_columns


def test_static_export_helper_uses_the_same_stored_figure(monkeypatch):
    frame = pd.DataFrame({"value": [1, 2, 4, 8, 16, 32], "group": ["a", "a", "b", "b", "a", "b"]})
    result = _eda(frame)
    chart = result.charts[0]
    calls = {}

    def fake_to_image(*, format, width, height, scale):
        calls.update({"format": format, "width": width, "height": height, "scale": scale})
        return b"same-stored-figure"

    monkeypatch.setattr(chart.figure, "to_image", fake_to_image)
    from crispdm_studio.reporting.chart_export import chart_to_png_bytes

    exported = chart_to_png_bytes(chart, width=640, height=480, scale=1.0)
    assert exported == b"same-stored-figure"
    assert calls == {"format": "png", "width": 640, "height": 480, "scale": 1.0}
