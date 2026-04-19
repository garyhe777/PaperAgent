from __future__ import annotations

import json
from pathlib import Path

from pptx import Presentation
from pptx.util import Inches

from paperagent.config import Settings
from paperagent.schemas.models import DeckPlan, SlideContent
from paperagent.storage.repositories import PaperRepository


class PPTService:
    def __init__(
        self,
        settings: Settings,
        paper_repository: PaperRepository,
        ppt_planning_service,
        ppt_enrichment_service,
    ) -> None:
        self.settings = settings
        self.paper_repository = paper_repository
        self.ppt_planning_service = ppt_planning_service
        self.ppt_enrichment_service = ppt_enrichment_service

    def generate(self, paper_id: str, template_path: Path | None = None) -> dict:
        paper = self.paper_repository.get_paper(paper_id)
        if not paper:
            raise ValueError(f"Paper {paper_id} not found.")

        deck_plan = self.ppt_planning_service.plan(paper_id=paper_id)
        deck_content = self.ppt_enrichment_service.enrich(deck_plan, paper_id=paper_id)

        deck_dir = self.settings.deck_dir / paper_id
        deck_dir.mkdir(parents=True, exist_ok=True)
        plan_path = deck_dir / "deck_plan.json"
        content_path = deck_dir / "deck_content.json"
        ppt_path = deck_dir / "paper_briefing.pptx"
        plan_path.write_text(self._plan_to_json(deck_plan), encoding="utf-8")
        content_path.write_text(self._content_to_json(paper_id, deck_plan.title, deck_content), encoding="utf-8")
        self._render_ppt(deck_plan.title, deck_content, ppt_path, template_path=template_path)
        return {
            "paper_id": paper_id,
            "title": deck_plan.title,
            "ppt_path": str(ppt_path),
            "plan_path": str(plan_path),
            "content_path": str(content_path),
            "slide_count": len(deck_content),
        }

    def _render_ppt(
        self,
        deck_title: str,
        slides: list[SlideContent],
        output_path: Path,
        template_path: Path | None,
    ) -> None:
        presentation = Presentation(str(template_path)) if template_path else Presentation()
        for index, slide_content in enumerate(slides):
            if index == 0 or slide_content.slide_type == "title":
                self._add_title_slide(presentation, deck_title, slide_content)
            else:
                self._add_content_slide(presentation, slide_content)
        presentation.save(str(output_path))

    def _add_title_slide(self, presentation: Presentation, deck_title: str, slide_content: SlideContent) -> None:
        layout = presentation.slide_layouts[0] if len(presentation.slide_layouts) > 0 else presentation.slide_layouts[0]
        slide = presentation.slides.add_slide(layout)
        slide.shapes.title.text = slide_content.title or deck_title
        if len(slide.placeholders) > 1:
            subtitle = slide.placeholders[1]
            subtitle.text = "\n".join(slide_content.bullets[:3]) if slide_content.bullets else "Paper briefing"

    def _add_content_slide(self, presentation: Presentation, slide_content: SlideContent) -> None:
        layout = presentation.slide_layouts[1] if len(presentation.slide_layouts) > 1 else presentation.slide_layouts[0]
        slide = presentation.slides.add_slide(layout)
        slide.shapes.title.text = slide_content.title
        bullet_lines = slide_content.bullets[:5]
        if len(slide.placeholders) > 1:
            text_frame = slide.placeholders[1].text_frame
            text_frame.clear()
            for index, bullet in enumerate(bullet_lines):
                paragraph = text_frame.paragraphs[0] if index == 0 else text_frame.add_paragraph()
                paragraph.text = bullet
        else:
            text_box = slide.shapes.add_textbox(Inches(1), Inches(1.5), Inches(8), Inches(4.5))
            frame = text_box.text_frame
            for index, bullet in enumerate(bullet_lines):
                paragraph = frame.paragraphs[0] if index == 0 else frame.add_paragraph()
                paragraph.text = bullet

    def _plan_to_json(self, deck_plan: DeckPlan) -> str:
        payload = {
            "paper_id": deck_plan.paper_id,
            "title": deck_plan.title,
            "audience": deck_plan.audience,
            "slides": [
                {
                    "type": slide.slide_type,
                    "title": slide.title,
                    "goal": slide.goal,
                    "questions_to_search": slide.questions_to_search,
                    "layout_hint": slide.layout_hint,
                    "visual_intent": slide.visual_intent,
                }
                for slide in deck_plan.slides
            ],
        }
        return json.dumps(payload, ensure_ascii=False, indent=2)

    def _content_to_json(self, paper_id: str, title: str, slides: list[SlideContent]) -> str:
        payload = {
            "paper_id": paper_id,
            "title": title,
            "slides": [
                {
                    "type": slide.slide_type,
                    "title": slide.title,
                    "bullets": slide.bullets,
                    "notes": slide.notes,
                    "citations": slide.citations,
                    "layout_hint": slide.layout_hint,
                    "visual_intent": slide.visual_intent,
                }
                for slide in slides
            ],
        }
        return json.dumps(payload, ensure_ascii=False, indent=2)
