"""Reusable Phase 7 diagnostic charts built from stored held-out predictions."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from sklearn.metrics import confusion_matrix

from crispdm_studio.evaluation.metrics import PermutationImportanceItem
from crispdm_studio.modeling.comparison import ModelResult, ModelingResult
from crispdm_studio.understanding.task_inference import TaskType


@dataclass(frozen=True, slots=True)
class EvaluationChart:
    chart_id: str
    section: str
    title: str
    figure: go.Figure
    source_columns: tuple[str, ...]
    alt_text: str

    def to_plotly_json(self) -> dict[str, Any]:
        return self.figure.to_plotly_json()

    def to_html(self, *, include_plotlyjs: str | bool = "cdn") -> str:
        return pio.to_html(self.figure, full_html=False, include_plotlyjs=include_plotlyjs, config={"displaylogo": False, "responsive": True})


def permutation_importance_chart(items: tuple[PermutationImportanceItem, ...]) -> EvaluationChart | None:
    if not items:
        return None
    ordered = list(reversed(items[:20]))
    fig = go.Figure(go.Bar(x=[x.importance_mean for x in ordered], y=[x.feature for x in ordered], orientation="h", error_x={"type": "data", "array": [x.importance_std for x in ordered], "visible": True}))
    fig.update_layout(title="Held-out permutation importance", xaxis_title="Decrease in evaluation score when permuted", yaxis_title="Feature")
    return EvaluationChart("evaluation:permutation_importance", "Feature importance", "Held-out permutation importance", fig, tuple(x.feature for x in items), "Permutation importance on the held-out test set for the selected fitted pipeline.")


def error_diagnostic_chart(modeling: ModelingResult, selected: ModelResult) -> EvaluationChart | None:
    if modeling.X_test is None or modeling.y_test is None:
        return None
    pred = selected.fitted_pipeline.predict(modeling.X_test)
    if modeling.task_type is TaskType.CLASSIFICATION:
        labels = np.unique(np.concatenate([modeling.y_test.to_numpy(), np.asarray(pred)]))
        matrix = confusion_matrix(modeling.y_test, pred, labels=labels)
        fig = go.Figure(go.Heatmap(z=matrix, x=[str(x) for x in labels], y=[str(x) for x in labels], colorbar={"title": "Count"}, hovertemplate="Predicted=%{x}<br>Actual=%{y}<br>Count=%{z}<extra></extra>"))
        fig.update_layout(title="Held-out confusion matrix", xaxis_title="Predicted", yaxis_title="Actual")
        return EvaluationChart("evaluation:confusion_matrix", "Error diagnostics", "Held-out confusion matrix", fig, (), "Confusion matrix for selected model predictions on the held-out test set.")
    residuals = modeling.y_test.to_numpy(dtype=float) - np.asarray(pred, dtype=float)
    fig = go.Figure(go.Scatter(x=np.asarray(pred, dtype=float), y=residuals, mode="markers", name="Residual"))
    fig.add_hline(y=0, line_dash="dash")
    fig.update_layout(title="Held-out residual diagnostic", xaxis_title="Predicted value", yaxis_title="Residual (actual - predicted)")
    return EvaluationChart("evaluation:residuals", "Error diagnostics", "Held-out residual diagnostic", fig, (), "Residuals versus predictions for the selected regression pipeline on the held-out test set.")
