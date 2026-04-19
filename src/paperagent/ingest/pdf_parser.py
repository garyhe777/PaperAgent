from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import fitz

from paperagent.config import Settings


class BasePDFMarkdownConverter(ABC):
    """Shared interface for all PDF-to-Markdown backends."""

    backend_name: str = "base"

    @abstractmethod
    def convert(self, pdf_path: Path) -> tuple[str, str]:
        """Return a paper title and markdown text."""

    def infer_title(self, markdown_text: str, fallback: str) -> str:
        for line in markdown_text.splitlines():
            stripped = line.strip()
            if stripped.startswith("## "):
                return stripped[3:].strip()
            if stripped.startswith("# "):
                return stripped[2:].strip()
        return fallback


class PDFMarkdownConverter(BasePDFMarkdownConverter):
    """Convert a PDF into markdown with lightweight heading heuristics."""

    backend_name = "pymupdf"

    def convert(self, pdf_path: Path) -> tuple[str, str]:
        document = fitz.open(pdf_path)
        markdown_lines: list[str] = []
        title = pdf_path.stem

        for page_index, page in enumerate(document, start=1):
            markdown_lines.append(f"<!-- page: {page_index} -->")
            page_dict = page.get_text("dict")
            for block in page_dict.get("blocks", []):
                if "lines" not in block:
                    continue
                text_parts: list[str] = []
                max_size = 0.0
                for line in block["lines"]:
                    line_text = "".join(
                        span.get("text", "")
                        for span in line.get("spans", [])
                        if span.get("text", "").strip()
                    ).strip()
                    if line_text:
                        text_parts.append(line_text)
                    for span in line.get("spans", []):
                        max_size = max(max_size, float(span.get("size", 0)))
                if not text_parts:
                    continue
                text = " ".join(text_parts).strip()
                if page_index == 1 and not title and len(text) > 10:
                    title = text
                markdown_lines.append(self._format_line(text, max_size))
            markdown_lines.append("")
        markdown_text = "\n".join(markdown_lines).strip() + "\n"
        normalized_title = self.infer_title(markdown_text, fallback=title)
        return normalized_title, markdown_text

    def _format_line(self, text: str, font_size: float) -> str:
        if len(text) < 120 and font_size >= 15:
            return f"## {text}"
        if len(text) < 80 and text.isupper():
            return f"## {text.title()}"
        if len(text) < 80 and text.endswith(":"):
            return f"### {text.rstrip(':')}"
        return text


class DatalabPDFMarkdownConverter(BasePDFMarkdownConverter):
    """Convert a PDF into markdown through the Datalab hosted parser."""

    backend_name = "datalab"

    def __init__(self, api_key: str, mode: str = "balanced") -> None:
        if not api_key:
            raise ValueError("Datalab backend requires PAPERAGENT_DATALAB_API_KEY.")
        self.api_key = api_key
        self.mode = mode

    def convert(self, pdf_path: Path) -> tuple[str, str]:
        try:
            from datalab_sdk import ConvertOptions, DatalabClient
        except ModuleNotFoundError as exc:  # pragma: no cover - depends on optional package
            raise ModuleNotFoundError(
                "Datalab backend requires the optional 'datalab_sdk' package in the active environment."
            ) from exc

        client = DatalabClient(api_key=self.api_key)
        options = ConvertOptions(
            output_format="markdown",
            mode=self.mode,
        )
        result = client.convert(str(pdf_path), options=options)
        markdown_text = (result.markdown or "").strip()
        if not markdown_text:
            raise ValueError("Datalab returned empty markdown.")
        markdown_text = markdown_text + "\n"
        title = self.infer_title(markdown_text, fallback=pdf_path.stem)
        return title, markdown_text


def build_pdf_markdown_converter(
    settings: Settings,
    backend: str | None = None,
) -> BasePDFMarkdownConverter:
    selected_backend = (backend or settings.pdf_backend).strip().lower()
    if selected_backend == "pymupdf":
        return PDFMarkdownConverter()
    if selected_backend == "datalab":
        return DatalabPDFMarkdownConverter(
            api_key=settings.datalab_api_key or "",
            mode=settings.datalab_mode,
        )
    raise ValueError(
        f"Unsupported pdf backend '{selected_backend}'. Expected one of: pymupdf, datalab."
    )
