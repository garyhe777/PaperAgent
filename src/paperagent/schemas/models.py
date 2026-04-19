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
class PaperProfileRecord:
    paper_id: str
    abstract_text: str
    short_summary: str
    keywords: list[str]
    profile_status: str
    profile_error: str | None
    profile_updated_at: datetime


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
class PaperCatalogResult:
    paper_id: str
    title: str
    short_summary: str
    keywords: list[str]
    score: float


@dataclass(slots=True)
class AgentEvent:
    event_type: str
    message: str
    payload: dict = field(default_factory=dict)


@dataclass(slots=True)
class ChatSessionRecord:
    session_id: str
    paper_id: str | None
    title: str
    style: str
    created_at: datetime
    updated_at: datetime


@dataclass(slots=True)
class ChatMessageRecord:
    message_id: int
    session_id: str
    message_type: str
    content: str
    raw_json: str
    created_at: datetime


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


@dataclass(slots=True)
class SlidePlan:
    slide_type: str
    title: str
    goal: str
    questions_to_search: list[str]
    layout_hint: str
    visual_intent: str


@dataclass(slots=True)
class DeckPlan:
    paper_id: str
    title: str
    audience: str
    slides: list[SlidePlan]


@dataclass(slots=True)
class SlideContent:
    slide_type: str
    title: str
    bullets: list[str]
    notes: str
    citations: list[str]
    layout_hint: str = ""
    visual_intent: str = ""


@dataclass(slots=True)
class RenderResult:
    ppt_path: str
    slide_count: int
    renderer: str
