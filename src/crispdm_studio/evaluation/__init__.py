"""Phase 7 evaluation public API. Evaluation never refits a model."""
from __future__ import annotations

from dataclasses import dataclass

from crispdm_studio.clustering.engine import ClusteringResult
from crispdm_studio.eda.engine import EDAResult
from crispdm_studio.evaluation.diagnostics import EvaluationChart, error_diagnostic_chart, permutation_importance_chart
from crispdm_studio.evaluation.metrics import (
    BaselineImprovement, ClassImbalanceDiagnostic, PermutationImportanceItem, SelectionScore,
    baseline_improvements, class_imbalance, compute_permutation_importance, reasoned_selection,
)
from crispdm_studio.evaluation.recommendations import Recommendation, build_recommendation
from crispdm_studio.modeling.comparison import ModelResult, ModelingResult
from crispdm_studio.understanding.profiler import UnderstandingResult


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    skipped: bool
    skip_reason: str | None
    selected_model_name: str | None
    selected_model: ModelResult | None
    selection_reason: str
    baseline_improvements: tuple[BaselineImprovement, ...]
    selection_scores: tuple[SelectionScore, ...]
    imbalance: ClassImbalanceDiagnostic
    permutation_importance: tuple[PermutationImportanceItem, ...]
    charts: tuple[EvaluationChart, ...]
    recommendation: Recommendation


def evaluate_modeling(modeling: ModelingResult, eda: EDAResult, clustering: ClusteringResult, understanding: UnderstandingResult) -> EvaluationResult:
    """Evaluate already-fitted Phase 6 results without any estimator or transformer fit."""
    imbalance = class_imbalance(modeling)
    selected, scores, reason = reasoned_selection(modeling)
    improvements = baseline_improvements(modeling)
    importance: tuple[PermutationImportanceItem, ...] = ()
    charts: list[EvaluationChart] = []
    if selected is not None:
        importance = compute_permutation_importance(modeling, selected)
        importance_chart = permutation_importance_chart(importance)
        error_chart = error_diagnostic_chart(modeling, selected)
        if importance_chart:
            charts.append(importance_chart)
        if error_chart:
            charts.append(error_chart)
    recommendation = build_recommendation(modeling, selected, scores, imbalance, understanding, clustering, reason)
    skipped = selected is None
    return EvaluationResult(skipped, reason if skipped else None, selected.name if selected else None, selected, reason, improvements, scores, imbalance, importance, tuple(charts), recommendation)


__all__ = ["EvaluationResult", "EvaluationChart", "evaluate_modeling"]
