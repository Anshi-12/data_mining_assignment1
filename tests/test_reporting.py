from __future__ import annotations

from dataclasses import replace
from functools import lru_cache
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd

from crispdm_studio.clustering import run_clustering
from crispdm_studio.config import CONFIG
from crispdm_studio.eda import run_eda
from crispdm_studio.evaluation import evaluate_modeling
from crispdm_studio.ingestion import ingest_csv
from crispdm_studio.modeling import run_modeling
from crispdm_studio.preparation import prepare_dataset
from crispdm_studio.reporting import build_report, build_report_context, render_html_report
from crispdm_studio.reporting.chart_export import ChartExportError
from crispdm_studio.reporting.pdf_report import render_pdf_report
from crispdm_studio.understanding import profile_dataset


@lru_cache(maxsize=1)
def _computed_results():
    rng = np.random.default_rng(23)
    n = 90
    signal = rng.normal(size=n)
    secondary = rng.normal(size=n)
    label = (signal + 0.25 * secondary + rng.normal(scale=0.35, size=n) > 0).astype(int)
    frame = pd.DataFrame(
        {
            "signal": signal,
            "secondary": secondary,
            "region": np.where(secondary > 0, "east", "west"),
            "label": label,
        }
    )
    dataset = ingest_csv(
        filename="report.csv",
        mime_type="text/csv",
        data=frame.to_csv(index=False).encode(),
        config=CONFIG,
    )
    understanding = profile_dataset(dataset)
    preparation = prepare_dataset(dataset.dataframe, understanding)
    eda = run_eda(preparation.structural_dataframe, understanding)
    clustering = run_clustering(preparation.structural_dataframe, understanding)
    modeling = run_modeling(preparation.structural_dataframe, understanding, preparation, "label")
    evaluation = evaluate_modeling(modeling, eda, clustering, understanding)
    return understanding, preparation, eda, clustering, modeling, evaluation


def test_html_always_renders_and_contains_all_report_sections():
    context = build_report_context(*_computed_results())
    html = render_html_report(context)
    for heading in (
        "Executive summary",
        "Business Understanding",
        "Data Understanding",
        "Data Preparation",
        "Exploratory Data Analysis",
        "Clustering",
        "Predictive Modeling",
        "Model Comparison",
        "Evaluation",
        "Limitations",
        "Recommendation",
        "Technical appendix",
    ):
        assert heading in html
    assert "plotly" in html.lower()
    assert _computed_results()[2].findings[0].text in html
    assert _computed_results()[5].recommendation.what_we_learned in html


def test_report_context_reuses_exact_stored_chart_objects():
    understanding, preparation, eda, clustering, modeling, evaluation = _computed_results()
    context = build_report_context(understanding, preparation, eda, clustering, modeling, evaluation)
    stored = list(eda.charts) + list(clustering.charts) + list(evaluation.charts)
    by_id = {chart.chart_id: chart for chart in stored}
    assert context.charts
    for item in context.charts:
        assert item.source_object is by_id[item.chart_id]


def test_report_generation_does_not_refit(monkeypatch):
    # All model fitting happened before this patch. Report generation must remain
    # read-only; any accidental Pipeline.fit call is therefore a hard failure.
    results = _computed_results()
    import sklearn.pipeline

    def fail_fit(*args, **kwargs):
        raise AssertionError("reporting must never fit a pipeline")

    monkeypatch.setattr(sklearn.pipeline.Pipeline, "fit", fail_fit)
    context = build_report_context(*results)
    html = render_html_report(context)
    assert html.startswith("<!doctype html>")


def test_skipped_clustering_section_renders_honestly():
    understanding, preparation, eda, clustering, modeling, evaluation = _computed_results()
    skipped = replace(
        clustering,
        status="skipped",
        chosen_k=None,
        silhouette=None,
        selection_reason="fixture has no meaningful separation",
        profiles=(),
        segment_descriptions=(),
    )
    context = build_report_context(understanding, preparation, eda, skipped, modeling, evaluation)
    html = render_html_report(context)
    assert "Clustering skipped." in html
    assert "fixture has no meaningful separation" in html


def test_pdf_chart_fallback_is_clean_when_static_renderer_missing(monkeypatch):
    context = build_report_context(*_computed_results())
    html = render_html_report(context)

    import crispdm_studio.reporting.pdf_report as module

    def no_static(*args, **kwargs):
        raise ChartExportError("no Chrome in fixture")

    class FakeHTML:
        def __init__(self, *, string):
            self.string = string
        def write_pdf(self):
            assert "Chart unavailable in this PDF" in self.string
            return b"%PDF-fake-styled"

    monkeypatch.setattr(module, "chart_to_png_bytes", no_static)
    monkeypatch.setitem(sys.modules, "weasyprint", SimpleNamespace(HTML=FakeHTML))
    result = render_pdf_report(html, context)
    assert result.pdf_bytes.startswith(b"%PDF")
    assert result.static_charts_placeholder == len(context.charts)
    assert result.fidelity == "chart-fallback"
    assert any("plotly_get_chrome" in (notice.install_hint or "") for notice in result.notices)


def test_pdf_text_fallback_is_produced_if_weasyprint_fails(monkeypatch):
    context = build_report_context(*_computed_results())
    html = render_html_report(context)
    import crispdm_studio.reporting.pdf_report as module

    monkeypatch.setattr(module, "chart_to_png_bytes", lambda chart: b"fake-png")

    class BrokenHTML:
        def __init__(self, *, string):
            pass
        def write_pdf(self):
            raise RuntimeError("missing native PDF runtime")

    monkeypatch.setitem(sys.modules, "weasyprint", SimpleNamespace(HTML=BrokenHTML))
    result = render_pdf_report(html, context)
    assert result.pdf_bytes.startswith(b"%PDF-1.4")
    assert result.fidelity == "text-only-fallback"
    assert result.weasyprint_succeeded is False
    assert any(notice.component == "html_to_pdf" for notice in result.notices)
