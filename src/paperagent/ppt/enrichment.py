from __future__ import annotations

import json
import re

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from paperagent.agent.prompts import PromptLoader
from paperagent.config import Settings
from paperagent.schemas.models import DeckPlan, RetrievalResult, SlideContent, SlidePlan


class PPTEnrichmentService:
    def __init__(self, settings: Settings, retrieval_service) -> None:
        self.settings = settings
        self.retrieval_service = retrieval_service
        self.prompt_loader = PromptLoader()

    def enrich(self, deck_plan: DeckPlan, paper_id: str) -> list[SlideContent]:
        slides: list[SlideContent] = []
        for slide in deck_plan.slides:
            evidence = self._collect_evidence(slide, paper_id)
            slides.append(self._build_slide_content(slide, evidence))
        return slides

    def _collect_evidence(self, slide: SlidePlan, paper_id: str) -> list[RetrievalResult]:
        evidence: list[RetrievalResult] = []
        for query in slide.questions_to_search[:2]:
            hits = self.retrieval_service.retrieve(
                query=query,
                paper_id=paper_id,
                top_k=3,
            )
            evidence.extend(hits[:3])
        deduped: dict[str, RetrievalResult] = {}
        for item in evidence:
            if item.chunk_id not in deduped:
                deduped[item.chunk_id] = item
        return list(deduped.values())[:3]

    def _build_slide_content(self, slide: SlidePlan, evidence: list[RetrievalResult]) -> SlideContent:
        if self.settings.llm_backend == "mock":
            return self._mock_slide_content(slide, evidence)

        prompt = self.prompt_loader.load("ppt_slide_writer_system.txt")
        llm = self._chat_model(temperature=0.2)
        response = llm.invoke(
            [
                SystemMessage(content=prompt),
                HumanMessage(
                    content=json.dumps(
                        {
                            "slide_type": slide.slide_type,
                            "title": slide.title,
                            "goal": slide.goal,
                            "layout_hint": slide.layout_hint,
                            "visual_intent": slide.visual_intent,
                            "evidence": [
                                {
                                    "chunk_id": item.chunk_id,
                                    "section_title": item.section_title,
                                    "page_number": item.page_number,
                                    "content": item.content,
                                }
                                for item in evidence
                            ],
                        },
                        ensure_ascii=False,
                    )
                ),
            ]
        )
        payload = self._safe_json_loads(str(response.content))
        bullets = [str(item).strip() for item in payload.get("bullets", []) if str(item).strip()]
        notes = str(payload.get("notes", "")).strip()
        citations = [str(item).strip() for item in payload.get("citations", []) if str(item).strip()]
        if not bullets:
            return self._mock_slide_content(slide, evidence)
        return SlideContent(
            slide_type=slide.slide_type,
            title=str(payload.get("title", slide.title)).strip() or slide.title,
            bullets=bullets[:5],
            notes=notes or "Use the evidence to explain this slide clearly.",
            citations=citations or [item.chunk_id for item in evidence],
            layout_hint=slide.layout_hint,
            visual_intent=slide.visual_intent,
        )

    def _mock_slide_content(self, slide: SlidePlan, evidence: list[RetrievalResult]) -> SlideContent:
        if not evidence:
            bullets = ["当前没有足够证据支撑这一页，可在后续追问中补充更多细节。"]
            citations: list[str] = []
        else:
            bullets = []
            for item in evidence[:3]:
                sentence = item.content.replace("\n", " ").strip()
                sentence = re.split(r"(?<=[.!?])\s+", sentence)[0].strip()
                bullets.append(f"{sentence[:160]}")
            citations = [item.chunk_id for item in evidence[:3]]
        return SlideContent(
            slide_type=slide.slide_type,
            title=slide.title,
            bullets=bullets,
            notes=f"Goal: {slide.goal}",
            citations=citations,
            layout_hint=slide.layout_hint,
            visual_intent=slide.visual_intent,
        )

    def _safe_json_loads(self, text: str) -> dict:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                return json.loads(match.group(0))
            raise

    def _chat_model(self, temperature: float) -> ChatOpenAI:
        return ChatOpenAI(
            model=self.settings.llm_model,
            api_key=self.settings.llm_api_key,
            base_url=self.settings.llm_base_url,
            temperature=temperature,
        )
