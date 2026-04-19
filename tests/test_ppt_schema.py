from __future__ import annotations

from dataclasses import asdict

from paperagent.schemas.models import DeckPlan, RenderResult, SlideContent, SlidePlan


def test_ppt_planning_schemas_are_serializable():
    slide_plan = SlidePlan(
        slide_type="method",
        title="Core Method",
        goal="Explain the proposed method",
        questions_to_search=["What is the core method?"],
        layout_hint="two-column",
        visual_intent="method diagram",
    )
    deck_plan = DeckPlan(
        paper_id="paper-1",
        title="Sample Deck",
        audience="beginner",
        slides=[slide_plan],
    )
    slide_content = SlideContent(
        slide_type="method",
        title="Core Method",
        bullets=["Bullet 1", "Bullet 2"],
        notes="Explain the key steps.",
        citations=["paper-1-0001"],
        layout_hint="two-column",
        visual_intent="method diagram",
    )
    render_result = RenderResult(
        ppt_path="output.pptx",
        slide_count=1,
        renderer="skill",
    )

    assert asdict(deck_plan)["slides"][0]["slide_type"] == "method"
    assert asdict(slide_content)["citations"] == ["paper-1-0001"]
    assert asdict(render_result)["renderer"] == "skill"
