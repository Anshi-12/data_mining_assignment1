"""Canonical single-template HTML report generation."""
from __future__ import annotations

from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from crispdm_studio.reporting.context import ReportContext

PROJECT_ROOT = Path(__file__).resolve().parents[3]
TEMPLATE_DIR = PROJECT_ROOT / "templates"
CSS_PATH = PROJECT_ROOT / "assets" / "report.css"


def _environment() -> Environment:
    env = Environment(
        loader=FileSystemLoader(str(TEMPLATE_DIR)),
        autoescape=select_autoescape(("html", "xml")),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    env.filters["pct"] = lambda value: "—" if value is None else f"{float(value):.1%}"
    env.filters["num"] = lambda value: "—" if value is None else f"{float(value):,.4g}"
    env.filters["metric"] = lambda value: "—" if value is None else f"{float(value):.3f}"
    return env


def render_html_report(context: ReportContext) -> str:
    """Render the canonical report HTML. This path has no system dependencies."""
    template = _environment().get_template("report.html.j2")
    css = CSS_PATH.read_text(encoding="utf-8")
    return template.render(report=context, report_css=css)
