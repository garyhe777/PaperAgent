from __future__ import annotations

from pathlib import Path

from paperagent.schemas.models import DeckPlan, SlidePlan
from paperagent.services import ServiceContainer


def test_ppt_enrichment_uses_rag_and_returns_citations(container: ServiceContainer, sample_pdf: Path):
    result = container.ingest_service.ingest(pdf_path=sample_pdf)
    plan = DeckPlan(
        paper_id=result["paper_id"],
        title=result["title"],
        audience="beginner",
        slides=[
            SlidePlan(
                slide_type="method",
                title="Core Method",
                goal="Explain the core method.",
                questions_to_search=["What is the core method proposed in this paper?"],
                layout_hint="method",
                visual_intent="method diagram",
            )
        ],
    )

    slides = container.ppt_enrichment_service.enrich(plan, result["paper_id"])
    assert slides
    assert slides[0].bullets
    assert slides[0].citations


def test_ppt_enrichment_gracefully_handles_missing_evidence(container: ServiceContainer):
    plan = DeckPlan(
        paper_id="paper-1",
        title="Missing Evidence Deck",
        audience="beginner",
        slides=[
            SlidePlan(
                slide_type="experiments",
                title="Experiments",
                goal="Explain experiments.",
                questions_to_search=["A query that will not match anything"],
                layout_hint="experiment",
                visual_intent="results summary",
            )
        ],
    )

    slides = container.ppt_enrichment_service.enrich(plan, "paper-1")
    assert slides
    assert slides[0].bullets
