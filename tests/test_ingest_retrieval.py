from __future__ import annotations

from pathlib import Path

from paperagent.services import ServiceContainer


def test_ingest_local_pdf(container: ServiceContainer, sample_pdf: Path):
    result = container.ingest_service.ingest(pdf_path=sample_pdf)
    assert result["status"] == "completed"
    assert Path(result["md_path"]).exists()
    paper = container.paper_repository.get_paper(result["paper_id"])
    assert paper is not None
    assert paper.ingest_status == "completed"
    chunks = container.chunk_repository.list_chunks(result["paper_id"])
    assert chunks


def test_hybrid_retrieval_returns_chunks(container: ServiceContainer, sample_pdf: Path):
    result = container.ingest_service.ingest(pdf_path=sample_pdf)
    hits = container.retrieval_service.retrieve(
        query="What is the method contribution?",
        paper_id=result["paper_id"],
        top_k=3,
    )
    assert hits
    assert all(hit.paper_id == result["paper_id"] for hit in hits)
    assert any("method" in hit.content.lower() or "method" in hit.section_title.lower() for hit in hits)
