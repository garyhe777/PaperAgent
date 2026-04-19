from __future__ import annotations

from pathlib import Path

from paperagent.ingest.pdf_parser import BasePDFMarkdownConverter
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


def test_ingest_can_override_pdf_backend(
    container: ServiceContainer,
    sample_pdf: Path,
    monkeypatch,
):
    calls: list[str] = []

    class FakeDatalabConverter(BasePDFMarkdownConverter):
        backend_name = "datalab"

        def convert(self, pdf_path: Path) -> tuple[str, str]:
            calls.append(str(pdf_path))
            return (
                "Fake Datalab Title",
                "\n".join(
                    [
                        "## Fake Datalab Title",
                        "",
                        "### Method",
                        "This markdown came from the fake datalab backend.",
                    ]
                )
                + "\n",
            )

    monkeypatch.setattr(
        "paperagent.ingest.service.build_pdf_markdown_converter",
        lambda settings, backend=None: FakeDatalabConverter(),
    )

    result = container.ingest_service.ingest(pdf_path=sample_pdf, pdf_backend="datalab")
    assert result["status"] == "completed"
    assert result["pdf_backend"] == "datalab"
    assert calls
    markdown = Path(result["md_path"]).read_text(encoding="utf-8")
    assert "fake datalab backend" in markdown.lower()
