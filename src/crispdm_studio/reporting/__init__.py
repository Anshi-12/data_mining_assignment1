"""Phase 8 reporting public API."""
from __future__ import annotations

from dataclasses import dataclass

from crispdm_studio.clustering.engine import ClusteringResult
from crispdm_studio.eda.engine import EDAResult
from crispdm_studio.evaluation import EvaluationResult
from crispdm_studio.modeling.comparison import ModelingResult
from crispdm_studio.preparation.preprocessing import PreparationResult
from crispdm_studio.reporting.context import ReportContext, build_report_context
from crispdm_studio.reporting.html_report import render_html_report
from crispdm_studio.reporting.pdf_report import ExportNotice, PDFResult, render_pdf_report
from crispdm_studio.understanding.profiler import UnderstandingResult


@dataclass(frozen=True, slots=True)
class ReportResult:
    context: ReportContext
    html: str
    pdf: PDFResult

    @property
    def html_bytes(self) -> bytes:
        return self.html.encode("utf-8")

    @property
    def pdf_bytes(self) -> bytes:
        return self.pdf.pdf_bytes


def build_report(
    understanding: UnderstandingResult,
    preparation: PreparationResult,
    eda: EDAResult,
    clustering: ClusteringResult,
    modeling: ModelingResult,
    evaluation: EvaluationResult,
) -> ReportResult:
    """Render both formats solely from previously computed result objects."""
    context = build_report_context(understanding, preparation, eda, clustering, modeling, evaluation)
    canonical_html = render_html_report(context)
    pdf = render_pdf_report(canonical_html, context)
    return ReportResult(context, canonical_html, pdf)


__all__ = [
    "ExportNotice",
    "PDFResult",
    "ReportContext",
    "ReportResult",
    "build_report",
    "build_report_context",
    "render_html_report",
    "render_pdf_report",
]
