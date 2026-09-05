"""Reusable clustering charts for interactive UI and static report export."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import plotly.graph_objects as go
import plotly.io as pio


@dataclass(frozen=True, slots=True)
class ClusteringChart:
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


def silhouette_chart(candidates) -> ClusteringChart:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[c.k for c in candidates], y=[c.silhouette for c in candidates], mode="lines+markers", name="Silhouette"))
    fig.update_layout(title="Silhouette by candidate k", xaxis_title="k", yaxis_title="Silhouette score")
    return ClusteringChart("cluster:silhouette", "diagnostics", "Silhouette by candidate k", fig, (), "Silhouette scores for each evaluated cluster count.")


def inertia_chart(candidates) -> ClusteringChart:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[c.k for c in candidates], y=[c.inertia for c in candidates], mode="lines+markers", name="Inertia"))
    fig.update_layout(title="K-Means inertia by candidate k", xaxis_title="k", yaxis_title="Inertia")
    return ClusteringChart("cluster:inertia", "diagnostics", "K-Means inertia by candidate k", fig, (), "Within-cluster sum of squares for each evaluated k.")


def pca_chart(x, y, labels, explained) -> ClusteringChart:
    fig = go.Figure()
    for cluster in sorted(set(labels)):
        idx = [i for i, label in enumerate(labels) if label == cluster]
        fig.add_trace(go.Scatter(x=[x[i] for i in idx], y=[y[i] for i in idx], mode="markers", name=f"Cluster {cluster}"))
    fig.update_layout(title="2D PCA projection of clustered observations", xaxis_title=f"PC1 ({explained[0]:.1%} variance)", yaxis_title=f"PC2 ({explained[1]:.1%} variance)")
    return ClusteringChart("cluster:pca", "projection", "2D PCA projection", fig, (), "PCA projection of observations colored by chosen K-Means cluster.")
