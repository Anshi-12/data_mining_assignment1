"""Robust PDF generation derived from the canonical HTML report.

The canonical HTML always succeeds without Chrome/WeasyPrint. For PDF, static chart
slots in that same HTML are filled from the exact stored chart figures. If Kaleido /
Chrome is unavailable, each failed chart gets an explicit placeholder. If WeasyPrint
itself is unavailable or cannot initialize its native dependencies, a small pure-
Python text PDF is generated from the same canonical HTML so the download still works.
"""
from __future__ import annotations

import base64
from dataclasses import dataclass
from html import unescape
from html.parser import HTMLParser
import re
from textwrap import wrap

from crispdm_studio.reporting.chart_export import ChartExportError, chart_to_png_bytes
from crispdm_studio.reporting.context import ReportContext


@dataclass(frozen=True, slots=True)
class ExportNotice:
    component: str
    message: str
    install_hint: str | None = None


@dataclass(frozen=True, slots=True)
class PDFResult:
    pdf_bytes: bytes
    fidelity: str
    notices: tuple[ExportNotice, ...]
    static_charts_rendered: int
    static_charts_placeholder: int
    weasyprint_succeeded: bool


def _escape_html(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _inject_static_charts(canonical_html: str, context: ReportContext) -> tuple[str, list[ExportNotice], int, int]:
    html = canonical_html
    notices: list[ExportNotice] = []
    rendered = 0
    placeholders = 0
    static_runtime_error: ChartExportError | None = None
    for item in context.charts:
        try:
            if static_runtime_error is not None:
                raise static_runtime_error
            png = chart_to_png_bytes(item.source_object)
            uri = "data:image/png;base64," + base64.b64encode(png).decode("ascii")
            replacement = (
                f'<figure class="chart-static"><img src="{uri}" alt="{_escape_html(item.alt_text)}">'
                f'<figcaption>{_escape_html(item.title)}</figcaption></figure>'
            )
            rendered += 1
        except ChartExportError as exc:
            # A missing Chrome/Kaleido runtime is global, so after the first static
            # export fails we stop retrying every chart and fill all remaining slots.
            static_runtime_error = exc
            replacement = (
                '<div class="chart-static chart-fallback">'
                f'<strong>Chart unavailable in this PDF:</strong> {_escape_html(item.title)}<br>'
                f'<span>{_escape_html(item.alt_text)}</span></div>'
            )
            placeholders += 1
            if not any(n.component == "static_chart" for n in notices):
                notices.append(
                    ExportNotice(
                        component="static_chart",
                        message=str(exc),
                        install_hint="Install Google Chrome/Chromium or run `plotly_get_chrome` so Kaleido can render full-fidelity PDF charts.",
                    )
                )
        html = html.replace(item.static_marker, replacement)
    return html, notices, rendered, placeholders


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag in {"script", "style"}:
            self._skip += 1
        if tag in {"p", "li", "h1", "h2", "h3", "h4", "tr", "br", "section", "div"}:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style"} and self._skip:
            self._skip -= 1
        if tag in {"p", "li", "h1", "h2", "h3", "h4", "tr", "section", "div"}:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._skip:
            cleaned = re.sub(r"\s+", " ", data).strip()
            if cleaned:
                self.parts.append(cleaned + " ")

    def text(self) -> str:
        raw = unescape("".join(self.parts))
        lines = [re.sub(r"\s+", " ", line).strip() for line in raw.splitlines()]
        return "\n".join(line for line in lines if line)


def _pdf_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _minimal_text_pdf(canonical_html: str) -> bytes:
    """Generate a valid, dependency-free Helvetica text PDF from canonical HTML."""
    parser = _TextExtractor()
    parser.feed(canonical_html)
    source_lines: list[str] = []
    for paragraph in parser.text().splitlines():
        source_lines.extend(wrap(paragraph, width=92) or [""])
        source_lines.append("")
    if not source_lines:
        source_lines = ["CRISP-DM report"]

    lines_per_page = 52
    pages = [source_lines[i : i + lines_per_page] for i in range(0, len(source_lines), lines_per_page)]
    objects: list[bytes] = []

    # Object numbers: 1 catalog, 2 pages, 3 font, then pairs (page, content).
    page_object_numbers = [4 + i * 2 for i in range(len(pages))]
    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    kids = " ".join(f"{n} 0 R" for n in page_object_numbers)
    objects.append(f"<< /Type /Pages /Kids [{kids}] /Count {len(pages)} >>".encode("ascii"))
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    for page_index, lines in enumerate(pages):
        page_num = page_object_numbers[page_index]
        content_num = page_num + 1
        objects.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 3 0 R >> >> /Contents {content_num} 0 R >>".encode("ascii")
        )
        commands = ["BT", "/F1 9 Tf", "11 TL", "50 750 Td"]
        first = True
        for line in lines:
            encoded = line.encode("latin-1", "replace").decode("latin-1")
            if not first:
                commands.append("T*")
            commands.append(f"({_pdf_escape(encoded)}) Tj")
            first = False
        commands.append("ET")
        stream = "\n".join(commands).encode("latin-1")
        objects.append(b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream")

    output = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0]
    for number, obj in enumerate(objects, start=1):
        offsets.append(len(output))
        output.extend(f"{number} 0 obj\n".encode("ascii"))
        output.extend(obj)
        output.extend(b"\nendobj\n")
    xref = len(output)
    output.extend(f"xref\n0 {len(objects)+1}\n".encode("ascii"))
    output.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        output.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    output.extend(
        f"trailer\n<< /Size {len(objects)+1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode("ascii")
    )
    return bytes(output)


def render_pdf_report(canonical_html: str, context: ReportContext) -> PDFResult:
    """Create PDF from canonical HTML, degrading safely when optional runtimes fail."""
    pdf_html, notices, rendered, placeholders = _inject_static_charts(canonical_html, context)
    try:
        # Lazy import is essential: HTML export must remain usable even if WeasyPrint
        # or its native platform dependencies are unavailable.
        from weasyprint import HTML

        pdf = HTML(string=pdf_html).write_pdf()
        fidelity = "full" if placeholders == 0 else "chart-fallback"
        return PDFResult(pdf, fidelity, tuple(notices), rendered, placeholders, True)
    except Exception as exc:
        notices.append(
            ExportNotice(
                component="html_to_pdf",
                message=f"WeasyPrint could not render the styled PDF ({type(exc).__name__}: {exc}). A text-only PDF fallback was produced instead.",
                install_hint="Install/repair WeasyPrint and its platform GTK/Pango libraries for styled PDF output.",
            )
        )
        return PDFResult(
            _minimal_text_pdf(pdf_html),
            "text-only-fallback",
            tuple(notices),
            rendered,
            placeholders,
            False,
        )
