from __future__ import annotations

from pathlib import Path

import fitz
import pytest

from paperagent.config import Settings
from paperagent.services import ServiceContainer


@pytest.fixture()
def sample_pdf(tmp_path: Path) -> Path:
    pdf_path = tmp_path / "sample_paper.pdf"
    document = fitz.open()
    page = document.new_page()
    text = "\n".join(
        [
            "PaperAgent Benchmark Study",
            "Abstract",
            "This paper studies a paper assistant for beginners.",
            "Introduction",
            "We solve the problem of understanding long papers quickly.",
            "Method",
            "Our method combines a retriever, an agent, and a deck generator.",
            "Experiments",
            "We evaluate on arXiv-style documents and compare explanation quality.",
            "Conclusion",
            "The system is useful but still limited by parsing quality.",
        ]
    )
    page.insert_text((72, 72), text, fontsize=16)
    document.save(pdf_path)
    document.close()
    return pdf_path


@pytest.fixture()
def test_settings(tmp_path: Path) -> Settings:
    settings = Settings(
        data_dir=tmp_path / "data",
        storage_dir=tmp_path / "data" / "storage",
        database_path=tmp_path / "data" / "paperagent.db",
        chroma_dir=tmp_path / "data" / "chroma",
        bm25_dir=tmp_path / "data" / "bm25",
        deck_dir=tmp_path / "data" / "decks",
        pdf_backend="pymupdf",
        llm_backend="mock",
        embedding_backend="hash",
    )
    settings.ensure_directories()
    return settings


@pytest.fixture()
def container(test_settings: Settings) -> ServiceContainer:
    return ServiceContainer(test_settings)
