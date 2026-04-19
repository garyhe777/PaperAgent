from __future__ import annotations

import json

from paperagent.config import Settings
from paperagent.schemas.models import DeckContent, SlideContent
from paperagent.storage.repositories import PaperRepository


class PPTService:
    def __init__(
        self,
        settings: Settings,
        paper_repository: PaperRepository,
        ppt_render_service,
    ) -> None:
        self.settings = settings
        self.paper_repository = paper_repository
        self.ppt_render_service = ppt_render_service

    def generate_from_content(self, deck_content: DeckContent) -> dict:
        normalized = self._normalize_deck_content(deck_content)
        paper = self.paper_repository.get_paper(normalized.paper_id)
        if not paper:
            raise ValueError(f"Paper {normalized.paper_id} not found.")

        deck_dir = self.settings.deck_dir / normalized.paper_id
        deck_dir.mkdir(parents=True, exist_ok=True)
        content_path = deck_dir / "deck_content.json"
        content_path.write_text(self._content_to_json(normalized), encoding="utf-8")

        render_result = self.ppt_render_service.render(
            paper_id=normalized.paper_id,
            deck_title=normalized.title,
            slides=normalized.slides,
        )
        return {
            "paper_id": normalized.paper_id,
            "title": normalized.title,
            "audience": normalized.audience,
            "content_path": str(content_path),
            "ppt_path": render_result.ppt_path,
            "slide_count": render_result.slide_count,
            "renderer": render_result.renderer,
        }

    def _normalize_deck_content(self, deck_content: DeckContent) -> DeckContent:
        paper_id = deck_content.paper_id.strip()
        if not paper_id:
            raise ValueError("Deck content must include a paper_id.")

        title = deck_content.title.strip()
        if not title:
            raise ValueError("Deck content must include a title.")

        audience = deck_content.audience.strip() or "beginner"
        if not 3 <= len(deck_content.slides) <= 8:
            raise ValueError("Deck content must contain between 3 and 8 slides.")

        normalized_slides: list[SlideContent] = []
        for index, slide in enumerate(deck_content.slides, start=1):
            slide_title = slide.title.strip()
            if not slide_title:
                raise ValueError(f"Slide {index} must include a title.")

            bullets = [item.strip() for item in slide.bullets if item.strip()]
            if not 1 <= len(bullets) <= 6:
                raise ValueError(f"Slide {index} must include between 1 and 6 bullets.")

            citations: list[str] = []
            for citation in slide.citations:
                normalized_citation = citation.strip()
                if normalized_citation and normalized_citation not in citations:
                    citations.append(normalized_citation)

            normalized_slides.append(
                SlideContent(
                    slide_type=slide.slide_type.strip() or "content",
                    title=slide_title,
                    bullets=bullets,
                    notes=slide.notes.strip(),
                    citations=citations,
                    layout_hint=slide.layout_hint.strip(),
                    visual_intent=slide.visual_intent.strip(),
                )
            )

        return DeckContent(
            paper_id=paper_id,
            title=title,
            audience=audience,
            slides=normalized_slides,
        )

    def _content_to_json(self, deck_content: DeckContent) -> str:
        payload = {
            "paper_id": deck_content.paper_id,
            "title": deck_content.title,
            "audience": deck_content.audience,
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
                for slide in deck_content.slides
            ],
        }
        return json.dumps(payload, ensure_ascii=False, indent=2)
