from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class SectionBlock:
    section_title: str
    page_number: int
    content: str


def split_markdown_into_sections(markdown_text: str) -> list[SectionBlock]:
    sections: list[SectionBlock] = []
    current_title = "Document"
    current_page = 1
    buffer: list[str] = []

    def flush() -> None:
        nonlocal buffer
        content = "\n".join(line for line in buffer if line.strip()).strip()
        if content:
            sections.append(
                SectionBlock(
                    section_title=current_title,
                    page_number=current_page,
                    content=content,
                )
            )
        buffer = []

    for raw_line in markdown_text.splitlines():
        line = raw_line.rstrip()
        if line.startswith("<!-- page:"):
            flush()
            try:
                current_page = int(line.removeprefix("<!-- page:").removesuffix(" -->").strip())
            except ValueError:
                current_page = current_page
            continue
        if line.startswith("#"):
            flush()
            current_title = line.lstrip("#").strip() or current_title
            continue
        buffer.append(line)
    flush()
    return sections


def chunk_sections(
    sections: list[SectionBlock],
    chunk_size: int,
    chunk_overlap: int,
) -> list[SectionBlock]:
    chunks: list[SectionBlock] = []
    for section in sections:
        text = section.content.strip()
        if len(text) <= chunk_size:
            chunks.append(section)
            continue
        start = 0
        while start < len(text):
            end = min(len(text), start + chunk_size)
            window = text[start:end].strip()
            if window:
                chunks.append(
                    SectionBlock(
                        section_title=section.section_title,
                        page_number=section.page_number,
                        content=window,
                    )
                )
            if end == len(text):
                break
            start = max(end - chunk_overlap, start + 1)
    return chunks
