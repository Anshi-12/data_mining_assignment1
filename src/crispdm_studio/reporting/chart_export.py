"""Shared static/HTML export helpers for reusable analysis chart objects.

Any analysis chart that exposes ``figure``, ``chart_id`` and ``to_html`` can use
these helpers. Phase 4 EDA and Phase 5 clustering therefore share one export path.
"""
from __future__ import annotations

from pathlib import Path
from typing import Protocol, Any


class ExportableChart(Protocol):
    chart_id: str
    figure: Any
    def to_html(self, *, include_plotlyjs: str | bool = "cdn") -> str: ...


class ChartExportError(RuntimeError):
    """Raised when Plotly's static renderer cannot export a stored chart."""


def chart_to_png_bytes(chart: ExportableChart, *, width: int = 1200, height: int = 700, scale: float = 1.5) -> bytes:
    try:
        return chart.figure.to_image(format="png", width=width, height=height, scale=scale)
    except Exception as exc:
        raise ChartExportError(
            f"Static export failed for chart '{chart.chart_id}'. Ensure Kaleido/Chrome requirements are available."
        ) from exc


def chart_to_svg_bytes(chart: ExportableChart, *, width: int = 1200, height: int = 700) -> bytes:
    try:
        return chart.figure.to_image(format="svg", width=width, height=height)
    except Exception as exc:
        raise ChartExportError(
            f"Static export failed for chart '{chart.chart_id}'. Ensure Kaleido/Chrome requirements are available."
        ) from exc


def chart_to_html(chart: ExportableChart, *, include_plotlyjs: str | bool = "cdn") -> str:
    return chart.to_html(include_plotlyjs=include_plotlyjs)


def export_chart_png(chart: ExportableChart, destination: str | Path) -> Path:
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(chart_to_png_bytes(chart))
    return path
