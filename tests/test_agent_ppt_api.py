from __future__ import annotations

import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from fastapi.testclient import TestClient

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
