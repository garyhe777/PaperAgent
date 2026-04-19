from __future__ import annotations

import json
import re

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from paperagent.agent.prompts import PromptLoader
from paperagent.config import Settings
from paperagent.schemas.models import DeckPlan, SlidePlan
from paperagent.storage.repositories import PaperRepository


class PPTPlanningService:
    def __init__(self, settings: Settings, paper_repository: PaperRepository) -> None:
        self.settings = settings
        self.paper_repository = paper_repository
        self.prompt_loader = PromptLoader()

    def plan(self, paper_id: str, audience: str = "beginner", slide_count: int = 6) -> DeckPlan:
        paper = self.paper_repository.get_paper(paper_id)
        if not paper:
            raise ValueError(f"Paper {paper_id} not found.")
        profile = self.paper_repository.get_profile(paper_id)
        if self.settings.llm_backend == "mock":
            return self._mock_plan(
                paper_id=paper_id,
                title=paper.title,
                audience=audience,
                short_summary=profile.short_summary if profile else "",
            )

        prompt = self.prompt_loader.load("ppt_planner_system.txt")
        llm = self._chat_model(temperature=0)
        response = llm.invoke(
            [
                SystemMessage(content=prompt),
                HumanMessage(
                    content=json.dumps(
                        {
                            "paper_id": paper_id,
                            "title": paper.title,
                            "audience": audience,
                            "slide_count": max(5, min(slide_count, 7)),
                            "abstract_text": profile.abstract_text if profile else "",
                            "short_summary": profile.short_summary if profile else "",
                            "keywords": profile.keywords if profile else [],
                        },
                        ensure_ascii=False,
                    )
                ),
            ]
        )
        payload = self._safe_json_loads(str(response.content))
        return self._payload_to_plan(paper_id=paper_id, title=paper.title, audience=audience, payload=payload)

    def _mock_plan(self, paper_id: str, title: str, audience: str, short_summary: str) -> DeckPlan:
        slides = [
            SlidePlan(
                slide_type="title",
                title=title,
                goal="Introduce the paper topic and framing.",
                questions_to_search=[],
                layout_hint="title",
                visual_intent="hero title slide",
            ),
            SlidePlan(
                slide_type="background",
                title="Problem & Motivation",
                goal="Explain why this paper matters.",
                questions_to_search=["What problem does this paper solve?"],
                layout_hint="content",
                visual_intent="problem framing",
            ),
            SlidePlan(
                slide_type="method",
                title="Core Method",
                goal="Explain the main method simply.",
                questions_to_search=["What is the core method proposed in this paper?"],
                layout_hint="method",
                visual_intent="method pipeline",
            ),
            SlidePlan(
                slide_type="experiments",
                title="Experiments & Results",
                goal="Summarize evaluation and key results.",
                questions_to_search=["What experiments and results are most important in this paper?"],
                layout_hint="experiment",
                visual_intent="results summary",
            ),
            SlidePlan(
                slide_type="conclusion",
                title="Conclusion & Limitations",
                goal="Close with contributions and limitations.",
                questions_to_search=["What are the main conclusions and limitations of this paper?"],
                layout_hint="conclusion",
                visual_intent="closing summary",
            ),
        ]
        if short_summary:
            slides.insert(
                1,
                SlidePlan(
                    slide_type="takeaway",
                    title="Quick Takeaway",
                    goal="Give a short high-level summary before details.",
                    questions_to_search=[],
                    layout_hint="content",
                    visual_intent="summary card",
                ),
            )
        return DeckPlan(
            paper_id=paper_id,
            title=title,
            audience=audience,
            slides=slides[:6],
        )

    def _payload_to_plan(self, paper_id: str, title: str, audience: str, payload: dict) -> DeckPlan:
        raw_slides = payload.get("slides", [])
        slides: list[SlidePlan] = []
        for item in raw_slides:
            slide_type = str(item.get("type", "content")).strip().lower()
            slides.append(
                SlidePlan(
                    slide_type=slide_type,
                    title=str(item.get("title", "")).strip() or slide_type.title(),
                    goal=str(item.get("goal", "")).strip() or "Explain this part clearly.",
                    questions_to_search=[
                        str(question).strip()
                        for question in item.get("questions_to_search", [])
                        if str(question).strip()
                    ],
                    layout_hint=str(item.get("layout_hint", "content")).strip() or "content",
                    visual_intent=str(item.get("visual_intent", "")).strip() or slide_type,
                )
            )
        if not slides:
            return self._mock_plan(paper_id=paper_id, title=title, audience=audience, short_summary="")
        return DeckPlan(
            paper_id=paper_id,
            title=str(payload.get("deck_title", title)).strip() or title,
            audience=audience,
            slides=slides[:7],
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
