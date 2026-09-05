"""Reusable chart objects for EDA UI and later report export.

The EDA engine constructs each Plotly figure exactly once. UI code renders the
stored figure interactively; reporting code can export the same stored figure to
PNG/SVG/HTML without recomputing statistics or rebuilding charts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import plotly.graph_objects as go
import plotly.io as pio


@dataclass(frozen=True, slots=True)
class EDAChart:
    chart_id: str
    section: str
    title: str
    figure: go.Figure
    source_columns: tuple[str, ...]
    alt_text: str

    def to_plotly_json(self) -> dict[str, Any]:
        """Return a JSON-compatible Plotly specification for persistence/export."""
        return self.figure.to_plotly_json()

    def to_html(self, *, include_plotlyjs: str | bool = "cdn") -> str:
        """Return an embeddable interactive HTML fragment from the stored figure."""
        return pio.to_html(
            self.figure,
            full_html=False,
            include_plotlyjs=include_plotlyjs,
            config={"displaylogo": False, "responsive": True},
        )


def numeric_distribution_chart(column: str, values) -> EDAChart:
    figure = go.Figure()
    figure.add_trace(go.Histogram(x=values, nbinsx=30, name=column))
    figure.update_layout(
        title=f"Distribution of {column}",
        xaxis_title=column,
        yaxis_title="Count",
        bargap=0.05,
    )
    return EDAChart(
        chart_id=f"numeric_distribution:{column}",
        section="Numeric distributions",
        title=f"Distribution of {column}",
        figure=figure,
        source_columns=(column,),
        alt_text=f"Histogram showing the observed distribution of {column}.",
    )


def numeric_box_chart(column: str, values) -> EDAChart:
    figure = go.Figure(go.Box(y=values, name=column, boxpoints="outliers"))
    figure.update_layout(title=f"Outlier diagnostic for {column}", yaxis_title=column)
    return EDAChart(
        chart_id=f"outliers:{column}",
        section="Outlier diagnostics",
        title=f"Outlier diagnostic for {column}",
        figure=figure,
        source_columns=(column,),
        alt_text=f"Box plot for {column} with Plotly outlier points visible.",
    )


def categorical_distribution_chart(column: str, labels, counts) -> EDAChart:
    figure = go.Figure(go.Bar(x=list(labels), y=list(counts), name=column))
    figure.update_layout(
        title=f"Top observed values of {column}",
        xaxis_title=column,
        yaxis_title="Count",
    )
    return EDAChart(
        chart_id=f"categorical_distribution:{column}",
        section="Categorical distributions",
        title=f"Top observed values of {column}",
        figure=figure,
        source_columns=(column,),
        alt_text=f"Bar chart of the most common observed values in {column}.",
    )


def missingness_chart(columns, counts, ratios) -> EDAChart:
    figure = go.Figure(
        go.Bar(
            x=list(columns),
            y=[ratio * 100 for ratio in ratios],
            customdata=list(counts),
            hovertemplate="%{x}<br>Missing: %{customdata:,}<br>Share: %{y:.1f}%<extra></extra>",
        )
    )
    figure.update_layout(
        title="Missingness by column",
        xaxis_title="Column",
        yaxis_title="Missing rows (%)",
    )
    return EDAChart(
        chart_id="missingness:columns",
        section="Missingness",
        title="Missingness by column",
        figure=figure,
        source_columns=tuple(columns),
        alt_text="Bar chart showing the percentage of missing rows in each column with missing values.",
    )


def correlation_heatmap(columns, matrix) -> EDAChart:
    figure = go.Figure(
        data=go.Heatmap(
            z=matrix,
            x=list(columns),
            y=list(columns),
            zmin=-1,
            zmax=1,
            colorscale="RdBu",
            reversescale=True,
            colorbar={"title": "Pearson r"},
            hovertemplate="%{x} vs %{y}<br>r=%{z:.2f}<extra></extra>",
        )
    )
    figure.update_layout(title="Numeric correlation matrix")
    return EDAChart(
        chart_id="correlation:matrix",
        section="Correlation analysis",
        title="Numeric correlation matrix",
        figure=figure,
        source_columns=tuple(columns),
        alt_text="Heatmap of pairwise Pearson correlations among usable numeric columns.",
    )


def numeric_categorical_box_chart(numeric: str, categorical: str, frame) -> EDAChart:
    figure = go.Figure()
    for label, group in frame.groupby(categorical, observed=True):
        figure.add_trace(go.Box(y=group[numeric], name=str(label), boxpoints=False))
    figure.update_layout(
        title=f"{numeric} by {categorical}",
        xaxis_title=categorical,
        yaxis_title=numeric,
        showlegend=False,
    )
    return EDAChart(
        chart_id=f"numcat:{numeric}:{categorical}",
        section="Numeric ↔ categorical relationships",
        title=f"{numeric} by {categorical}",
        figure=figure,
        source_columns=(numeric, categorical),
        alt_text=f"Box plots comparing {numeric} across levels of {categorical}.",
    )


def categorical_association_chart(left: str, right: str, table) -> EDAChart:
    figure = go.Figure(
        data=go.Heatmap(
            z=table.to_numpy(),
            x=[str(value) for value in table.columns],
            y=[str(value) for value in table.index],
            colorbar={"title": "Count"},
            hovertemplate=f"{right}=%{{x}}<br>{left}=%{{y}}<br>Count=%{{z}}<extra></extra>",
        )
    )
    figure.update_layout(
        title=f"Observed counts: {left} × {right}",
        xaxis_title=right,
        yaxis_title=left,
    )
    return EDAChart(
        chart_id=f"catassoc:{left}:{right}",
        section="Categorical associations",
        title=f"Observed counts: {left} × {right}",
        figure=figure,
        source_columns=(left, right),
        alt_text=f"Contingency heatmap of observed counts for {left} and {right}.",
    )
