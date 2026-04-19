from __future__ import annotations

from pathlib import Path

import fitz


class PDFMarkdownConverter:
    """Convert a PDF into markdown with lightweight heading heuristics."""

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
        normalized_title = self._infer_title(markdown_text, fallback=title)
        return normalized_title, markdown_text

    def _format_line(self, text: str, font_size: float) -> str:
        if len(text) < 120 and font_size >= 15:
            return f"## {text}"
        if len(text) < 80 and text.isupper():
            return f"## {text.title()}"
        if len(text) < 80 and text.endswith(":"):
            return f"### {text.rstrip(':')}"
        return text

    def _infer_title(self, markdown_text: str, fallback: str) -> str:
        for line in markdown_text.splitlines():
            stripped = line.strip()
            if stripped.startswith("## "):
                return stripped[3:].strip()
        return fallback
