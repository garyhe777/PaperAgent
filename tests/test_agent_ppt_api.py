from __future__ import annotations

import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from fastapi.testclient import TestClient

from paperagent.schemas.models import ChunkRecord, PaperProfileRecord, PaperRecord
from paperagent.services import ServiceContainer
from paperagent.web.api import create_app


def test_chat_agent_streams_answer(container: ServiceContainer, sample_pdf: Path):
    result = container.ingest_service.ingest(pdf_path=sample_pdf)
    events = list(
        container.chat_agent.ask(
            paper_id=result["paper_id"],
            question="Explain the method and experiments",
            style="beginner",
        )
    )
    event_types = [event.event_type for event in events]
    assert "agent_started" in event_types
    assert "rag_hit" in event_types
    assert "final_answer_done" in event_types
    session_events = [event for event in events if event.event_type == "session_created"]
    assert session_events
    session_id = session_events[0].payload["session_id"]
    assert container.chat_session_repository.get_session(str(session_id)) is not None
    assert container.chat_message_repository.list_messages(str(session_id))


def test_general_chat_skips_retrieval_for_greeting(container: ServiceContainer):
    events = list(
        container.chat_agent.ask(
            paper_id=None,
            question="hello",
            style="beginner",
        )
    )
    event_types = [event.event_type for event in events]
    assert "agent_started" in event_types
    assert "rag_hit" not in event_types
    assert "final_answer_done" in event_types


def test_general_chat_can_search_database_when_needed(container: ServiceContainer, sample_pdf: Path):
    result = container.ingest_service.ingest(pdf_path=sample_pdf)
    events = list(
        container.chat_agent.ask(
            paper_id=None,
            question="Explain the method in the indexed papers",
            style="beginner",
        )
    )
    event_types = [event.event_type for event in events]
    assert result["paper_id"]
    assert "rag_hit" in event_types


def test_general_chat_can_search_catalog_without_chunk_rag(container: ServiceContainer):
    _seed_catalog_paper(
        container,
        paper_id="tag-wm",
        title="TAG-WM: Tamper-Aware Generative Image Watermarking",
        abstract="A watermarking paper for tamper-aware generation.",
        summary="A paper profile for TAG-WM.",
        keywords=["watermark", "tamper-aware"],
        chunks=[],
    )
    events = list(
        container.chat_agent.ask(
            paper_id=None,
            question="有哪些 watermark 相关论文？",
            style="beginner",
        )
    )
    event_types = [event.event_type for event in events]
    tool_names = [event.payload.get("tool_name") for event in events if event.event_type == "tool_called"]
    assert "search_papers" in tool_names
    assert "catalog_hit" in event_types
    assert "rag_hit" not in event_types


def test_general_chat_can_search_catalog_then_scoped_context(container: ServiceContainer):
    _seed_catalog_paper(
        container,
        paper_id="tag-wm",
        title="TAG-WM: Tamper-Aware Generative Image Watermarking",
        abstract="A watermarking paper for tamper-aware generation.",
        summary="A paper profile for TAG-WM.",
        keywords=["watermark", "tamper-aware"],
        chunks=[
            ChunkRecord(
                paper_id="tag-wm",
                chunk_id="tag-wm-0001",
                section_title="Method",
                page_number=1,
                content="TAG-WM uses diffusion inversion sensitivity to detect tampering in generated images.",
                token_count=12,
            )
        ],
    )
    events = list(
        container.chat_agent.ask(
            paper_id=None,
            question="TAG-WM 的方法是什么？",
            style="beginner",
        )
    )
    tool_names = [event.payload.get("tool_name") for event in events if event.event_type == "tool_called"]
    event_types = [event.event_type for event in events]
    assert "search_papers" in tool_names
    assert "search_paper_context" in tool_names
    assert "catalog_hit" in event_types
    assert "rag_hit" in event_types


def test_scoped_paper_chat_skips_catalog_lookup(container: ServiceContainer):
    _seed_catalog_paper(
        container,
        paper_id="tag-wm",
        title="TAG-WM: Tamper-Aware Generative Image Watermarking",
        abstract="A watermarking paper for tamper-aware generation.",
        summary="A paper profile for TAG-WM.",
        keywords=["watermark", "tamper-aware"],
        chunks=[
            ChunkRecord(
                paper_id="tag-wm",
                chunk_id="tag-wm-0001",
                section_title="Method",
                page_number=1,
                content="TAG-WM uses diffusion inversion sensitivity to detect tampering in generated images.",
                token_count=12,
            )
        ],
    )
    events = list(
        container.chat_agent.ask(
            paper_id="tag-wm",
            question="Explain the method",
            style="beginner",
        )
    )
    tool_names = [event.payload.get("tool_name") for event in events if event.event_type == "tool_called"]
    assert "search_paper_context" in tool_names
    assert "search_papers" not in tool_names


def test_session_context_is_persisted_across_turns(container: ServiceContainer, sample_pdf: Path):
    ingest_result = container.ingest_service.ingest(pdf_path=sample_pdf)
    first_events = list(
        container.chat_agent.ask(
            paper_id=ingest_result["paper_id"],
            question="Explain the method",
            style="beginner",
        )
    )
    session_id = str(
        next(event for event in first_events if event.event_type == "session_created").payload["session_id"]
    )

    second_events = list(
        container.chat_agent.ask(
            paper_id=None,
            session_id=session_id,
            question="Continue and summarize the experiment part",
            style="beginner",
        )
    )
    assert any(event.event_type == "session_loaded" for event in second_events)
    persisted_messages = container.chat_message_repository.list_message_records(session_id)
    assert len(persisted_messages) >= 4


def test_ppt_generation_creates_files(container: ServiceContainer, sample_pdf: Path):
    result = container.ingest_service.ingest(pdf_path=sample_pdf)
    deck = container.ppt_service.generate(result["paper_id"])
    assert Path(deck["ppt_path"]).exists()
    assert Path(deck["outline_path"]).exists()
    assert deck["slide_count"] >= 3


def test_ingest_url_and_api_endpoints(container: ServiceContainer, sample_pdf: Path, tmp_path: Path):
    handler = partial(SimpleHTTPRequestHandler, directory=str(sample_pdf.parent))
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        url = f"http://127.0.0.1:{server.server_port}/{sample_pdf.name}"
        ingest_result = container.ingest_service.ingest(url=url)
        assert ingest_result["status"] == "completed"

        client = TestClient(create_app(container))
        papers_response = client.get("/papers")
        assert papers_response.status_code == 200
        assert papers_response.json()

        with sample_pdf.open("rb") as handle:
            upload_response = client.post(
                "/ingest/upload",
                files={"file": (sample_pdf.name, handle, "application/pdf")},
            )
        assert upload_response.status_code == 200

        stream_response = client.post(
            "/chat/stream",
            json={
                "session_id": None,
                "paper_id": ingest_result["paper_id"],
                "question": "What problem does the paper solve?",
                "style": "beginner",
            },
        )
        assert stream_response.status_code == 200
        assert "text/event-stream" in stream_response.headers["content-type"]
        assert "data:" in stream_response.text
    finally:
        server.shutdown()
        server.server_close()


def _seed_catalog_paper(
    container: ServiceContainer,
    paper_id: str,
    title: str,
    abstract: str,
    summary: str,
    keywords: list[str],
    chunks: list[ChunkRecord],
) -> None:
    from datetime import datetime

    now = datetime.utcnow()
    container.paper_repository.upsert_paper(
        PaperRecord(
            paper_id=paper_id,
            title=title,
            source_type="pdf",
            source_value=f"{paper_id}.pdf",
            pdf_path=f"{paper_id}.pdf",
            md_path=f"{paper_id}.md",
            ingest_status="completed",
            error_message=None,
            created_at=now,
            updated_at=now,
        )
    )
    container.paper_repository.upsert_profile(
        PaperProfileRecord(
            paper_id=paper_id,
            abstract_text=abstract,
            short_summary=summary,
            keywords=keywords,
            profile_status="completed",
            profile_error=None,
            profile_updated_at=now,
        )
    )
    if chunks:
        container.chunk_repository.replace_chunks(paper_id, chunks)
        container.retrieval_service.index_paper(paper_id, chunks)
