from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal


SourceType = Literal["pdf", "url"]
IngestStatus = Literal["pending", "processing", "completed", "failed"]


@dataclass(slots=True)
class PaperRecord:
    paper_id: str
    title: str
    source_type: SourceType
    source_value: str
    pdf_path: str
    md_path: str
    ingest_status: IngestStatus
    error_message: str | None
    created_at: datetime
    updated_at: datetime


@dataclass(slots=True)
class ChunkRecord:
    paper_id: str
    chunk_id: str
    section_title: str
    page_number: int
    content: str
    token_count: int


@dataclass(slots=True)
class RetrievalResult:
    paper_id: str
    chunk_id: str
    content: str
    section_title: str
    page_number: int
    score: float
    source: str


@dataclass(slots=True)
class AgentEvent:
    event_type: str
    message: str
    payload: dict = field(default_factory=dict)


@dataclass(slots=True)
class SlideOutline:
    title: str
    bullets: list[str]
    notes: str = ""


@dataclass(slots=True)
class DeckOutline:
    paper_id: str
    title: str
    slides: list[SlideOutline]
