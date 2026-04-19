from __future__ import annotations

from datetime import datetime

from paperagent.schemas.models import PaperRecord
from paperagent.services import ServiceContainer


def test_ppt_planner_returns_structured_plan(container: ServiceContainer):
    result = container.ingest_service.ingest(pdf_path=_make_sample_pdf(container))
    plan = container.ppt_planning_service.plan(result["paper_id"])
    assert plan.paper_id == result["paper_id"]
    assert 5 <= len(plan.slides) <= 7
    assert plan.slides[0].slide_type == "title"
    assert any(slide.slide_type == "method" for slide in plan.slides)


def test_ppt_planner_degrades_without_profile(container: ServiceContainer):
    now = datetime.utcnow()
    container.paper_repository.upsert_paper(
        PaperRecord(
            paper_id="paper-no-profile",
            title="Paper Without Profile",
            source_type="pdf",
            source_value="paper.pdf",
            pdf_path="paper.pdf",
            md_path="paper.md",
            ingest_status="completed",
            error_message=None,
            created_at=now,
            updated_at=now,
        )
    )
    plan = container.ppt_planning_service.plan("paper-no-profile")
    assert plan.title == "Paper Without Profile"
    assert plan.slides


def _make_sample_pdf(container: ServiceContainer):
    # Reuse the ingest fixture structure indirectly by creating a tiny local PDF path
    from pathlib import Path
    import fitz

    pdf_path = container.settings.data_dir / "planner_sample.pdf"
    document = fitz.open()
    page = document.new_page()
    page.insert_text(
        (72, 72),
        "\n".join(
            [
                "Sample Planning Paper",
                "Abstract",
                "This paper presents a planner-friendly benchmark.",
                "Introduction",
                "The paper focuses on structure and evidence.",
            ]
        ),
        fontsize=16,
    )
    document.save(Path(pdf_path))
    document.close()
    return Path(pdf_path)
