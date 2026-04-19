from __future__ import annotations

from pathlib import Path

import pytest

from paperagent.schemas.models import DeckContent, SlideContent
from paperagent.services import ServiceContainer


def test_ppt_service_generates_files_from_structured_content(
    container: ServiceContainer,
    sample_pdf: Path,
    monkeypatch,
):
    ingest_result = container.ingest_service.ingest(pdf_path=sample_pdf)

    def fake_detect_runtime():
        return {"available": True}

    def fake_execute_builder(builder_script: Path, render_config_path: Path, work_dir: Path):
        config = __import__("json").loads(render_config_path.read_text(encoding="utf-8"))
        Path(config["output_path"]).write_bytes(b"fake-pptx")

    monkeypatch.setattr(container.ppt_render_service, "_detect_runtime", fake_detect_runtime)
    monkeypatch.setattr(container.ppt_render_service, "_execute_builder", fake_execute_builder)

    deck = DeckContent(
        paper_id=ingest_result["paper_id"],
        title="Sample Deck",
        audience="beginner",
        slides=[
            SlideContent(
                slide_type="title",
                title="Paper Overview",
                bullets=["What problem the paper studies"],
                notes="Open the deck.",
                citations=[],
            ),
            SlideContent(
                slide_type="method",
                title="Core Method",
                bullets=["Summarize the main method"],
                notes="Explain the pipeline.",
                citations=[],
            ),
            SlideContent(
                slide_type="conclusion",
                title="Takeaways",
                bullets=["Summarize the conclusion"],
                notes="Close clearly.",
                citations=[],
            ),
        ],
    )

    result = container.ppt_service.generate_from_content(deck)

    assert Path(result["ppt_path"]).exists()
    assert Path(result["content_path"]).exists()
    assert result["renderer"] == "skill"
    assert result["slide_count"] == 3


def test_ppt_service_validates_slide_count(container: ServiceContainer):
    deck = DeckContent(
        paper_id="paper-1",
        title="Too Small",
        audience="beginner",
        slides=[
            SlideContent(
                slide_type="title",
                title="Only One",
                bullets=["One bullet"],
                notes="",
                citations=[],
            )
        ],
    )

    with pytest.raises(ValueError, match="between 3 and 8 slides"):
        container.ppt_service.generate_from_content(deck)


def test_ppt_service_validates_slide_bullets(container: ServiceContainer):
    deck = DeckContent(
        paper_id="paper-1",
        title="Bad Bullets",
        audience="beginner",
        slides=[
            SlideContent(slide_type="title", title="One", bullets=["ok"], notes="", citations=[]),
            SlideContent(slide_type="content", title="Two", bullets=[], notes="", citations=[]),
            SlideContent(slide_type="content", title="Three", bullets=["ok"], notes="", citations=[]),
        ],
    )

    with pytest.raises(ValueError, match="between 1 and 6 bullets"):
        container.ppt_service.generate_from_content(deck)
