from __future__ import annotations

from dataclasses import asdict

from paperagent.schemas.models import DeckContent, RenderResult, SlideContent


def test_ppt_content_schemas_are_serializable():
    slide_content = SlideContent(
        slide_type="method",
        title="Core Method",
        bullets=["Bullet 1", "Bullet 2"],
        notes="Explain the key steps.",
        citations=["paper-1-0001"],
        layout_hint="two-column",
        visual_intent="method diagram",
    )
    deck_content = DeckContent(
        paper_id="paper-1",
        title="Sample Deck",
        audience="beginner",
        slides=[slide_content],
    )
    render_result = RenderResult(
        ppt_path="output.pptx",
        slide_count=1,
        renderer="skill",
    )

    assert asdict(deck_content)["slides"][0]["slide_type"] == "method"
    assert asdict(slide_content)["citations"] == ["paper-1-0001"]
    assert asdict(render_result)["renderer"] == "skill"
