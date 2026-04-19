from __future__ import annotations

from pathlib import Path

from paperagent.ingest.profile_service import PaperProfileService
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


def test_profile_service_extracts_abstract(container: ServiceContainer):
    markdown = "\n".join(
        [
            "## Sample Paper",
            "",
            "## Abstract",
            "This paper studies robust watermarking for diffusion models.",
            "It improves inversion quality and preserves image fidelity.",
            "",
            "## Introduction",
            "Intro text.",
        ]
    )
    profile_service = PaperProfileService(container.settings)
    abstract = profile_service.extract_abstract(markdown)
    assert "robust watermarking" in abstract
    assert "Introduction" not in abstract


def test_ingest_generates_profile(container: ServiceContainer, sample_pdf: Path, monkeypatch):
    markdown_text = "\n".join(
        [
            "## PaperAgent Benchmark Study",
            "",
            "## Abstract",
            "This paper studies a paper assistant for beginners.",
            "It combines retrieval and explanation generation.",
            "",
            "## Introduction",
            "Intro text.",
        ]
    )

    class FakeConverter(BasePDFMarkdownConverter):
        backend_name = "pymupdf"

        def convert(self, pdf_path: Path) -> tuple[str, str]:
            return "PaperAgent Benchmark Study", markdown_text

    monkeypatch.setattr(
        "paperagent.ingest.service.build_pdf_markdown_converter",
        lambda settings, backend=None: FakeConverter(),
    )
    result = container.ingest_service.ingest(pdf_path=sample_pdf)
    profile = container.paper_repository.get_profile(result["paper_id"])
    assert profile is not None
    assert profile.profile_status == "completed"
    assert profile.abstract_text
    assert profile.short_summary
    assert profile.keywords


def test_ingest_without_abstract_marks_profile_empty(
    container: ServiceContainer,
    tmp_path: Path,
    monkeypatch,
):
    pdf_path = tmp_path / "no_abstract.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 fake")
    markdown_path = tmp_path / "fake.md"
    markdown_text = "\n".join(
        [
            "## Sample Paper",
            "",
            "## Introduction",
            "No abstract is present here.",
        ]
    )

    class FakeConverter(BasePDFMarkdownConverter):
        backend_name = "pymupdf"

        def convert(self, pdf_path: Path) -> tuple[str, str]:
            return "Sample Paper", markdown_text

    monkeypatch.setattr(
        "paperagent.ingest.service.build_pdf_markdown_converter",
        lambda settings, backend=None: FakeConverter(),
    )
    result = container.ingest_service.ingest(pdf_path=pdf_path)

    assert result["status"] == "completed"
    profile = container.paper_repository.get_profile(result["paper_id"])
    assert profile is not None
    assert profile.profile_status == "empty"
    assert profile.short_summary == ""


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
