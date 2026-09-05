"""Read-only report context assembled exclusively from already-computed CRISP-DM results."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from markupsafe import Markup

from crispdm_studio.clustering.engine import ClusteringResult
from crispdm_studio.eda.engine import EDAResult
from crispdm_studio.evaluation import EvaluationResult
from crispdm_studio.modeling.comparison import ModelingResult
from crispdm_studio.preparation.preprocessing import PreparationResult
from crispdm_studio.reporting.chart_export import chart_to_html
from crispdm_studio.understanding.profiler import UnderstandingResult


@dataclass(frozen=True, slots=True)
class ReportChart:
    chart_id: str
    section: str
    title: str
    alt_text: str
    interactive_html: Markup
    static_marker: str
    source_object: Any


@dataclass(frozen=True, slots=True)
class ReportContext:
    understanding: UnderstandingResult
    preparation: PreparationResult
    eda: EDAResult
    clustering: ClusteringResult
    modeling: ModelingResult
    evaluation: EvaluationResult
    charts: tuple[ReportChart, ...]
    executive_points: tuple[str, ...]
    limitations: tuple[str, ...]


def _unique_charts(*groups: Iterable[Any]) -> tuple[Any, ...]:
    seen: set[str] = set()
    charts: list[Any] = []
    for group in groups:
        for chart in group:
            chart_id = str(chart.chart_id)
            if chart_id not in seen:
                seen.add(chart_id)
                charts.append(chart)
    return tuple(charts)


def _report_charts(eda: EDAResult, clustering: ClusteringResult, evaluation: EvaluationResult) -> tuple[ReportChart, ...]:
    stored = _unique_charts(eda.charts, clustering.charts, evaluation.charts)
    result: list[ReportChart] = []
    for index, chart in enumerate(stored):
        # First chart embeds Plotly.js so the downloaded HTML is self-contained and
        # requires neither network access nor a system browser dependency.
        fragment = chart_to_html(chart, include_plotlyjs=True if index == 0 else False)
        result.append(
            ReportChart(
                chart_id=str(chart.chart_id),
                section=str(chart.section),
                title=str(chart.title),
                alt_text=str(chart.alt_text),
                interactive_html=Markup(fragment),
                static_marker=f"<!--STATIC_CHART:{chart.chart_id}-->",
                source_object=chart,
            )
        )
    return tuple(result)


def build_report_context(
    understanding: UnderstandingResult,
    preparation: PreparationResult,
    eda: EDAResult,
    clustering: ClusteringResult,
    modeling: ModelingResult,
    evaluation: EvaluationResult,
) -> ReportContext:
    """Assemble report data without recomputing analysis or fitting anything."""
    executive: list[str] = list(understanding.objective.known_facts)
    executive.extend(finding.text for finding in eda.findings[:3])
    if clustering.skipped:
        executive.append(f"Clustering: {clustering.selection_reason}")
    else:
        executive.extend(item.text for item in clustering.segment_descriptions[:2])
    executive.append(evaluation.recommendation.what_we_learned)

    limitations: list[str] = [warning.message for warning in understanding.quality_warnings]
    limitations.extend(f"{item.analysis}: {item.reason}" for item in eda.skipped_analyses)
    if clustering.skipped:
        limitations.append(f"Clustering: {clustering.selection_reason}")
    if modeling.skipped and modeling.skip_reason:
        limitations.append(f"Supervised modeling: {modeling.skip_reason}")
    if evaluation.skipped and evaluation.skip_reason:
        limitations.append(f"Evaluation: {evaluation.skip_reason}")
    limitations.append(evaluation.recommendation.limitations)

    # Preserve order while removing duplicate strings.
    limitations = list(dict.fromkeys(item for item in limitations if item))
    return ReportContext(
        understanding=understanding,
        preparation=preparation,
        eda=eda,
        clustering=clustering,
        modeling=modeling,
        evaluation=evaluation,
        charts=_report_charts(eda, clustering, evaluation),
        executive_points=tuple(executive),
        limitations=tuple(limitations),
    )
