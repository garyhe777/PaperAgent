from __future__ import annotations

import json
import re
from datetime import datetime

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from paperagent.agent.prompts import PromptLoader
from paperagent.config import Settings
from paperagent.schemas.models import PaperProfileRecord


class PaperProfileService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.prompt_loader = PromptLoader()

    def build_profile(self, paper_id: str, title: str, markdown_text: str) -> PaperProfileRecord:
        abstract_text = self.extract_abstract(markdown_text)
        now = datetime.utcnow()
        if not abstract_text:
            return PaperProfileRecord(
                paper_id=paper_id,
                abstract_text="",
                short_summary="",
                keywords=[],
                profile_status="empty",
                profile_error=None,
                profile_updated_at=now,
            )

        try:
            short_summary, keywords = self._generate_summary_and_keywords(title=title, abstract_text=abstract_text)
            return PaperProfileRecord(
                paper_id=paper_id,
                abstract_text=abstract_text,
                short_summary=short_summary,
                keywords=keywords,
                profile_status="completed",
                profile_error=None,
                profile_updated_at=now,
            )
        except Exception as exc:  # noqa: BLE001
            return PaperProfileRecord(
                paper_id=paper_id,
                abstract_text=abstract_text,
                short_summary="",
                keywords=[],
                profile_status="failed",
                profile_error=str(exc),
                profile_updated_at=now,
            )

    def extract_abstract(self, markdown_text: str) -> str:
        lines = [line.rstrip() for line in markdown_text.splitlines()]
        abstract_start: int | None = None
        section_markers = {
            "introduction",
            "background",
            "relatedwork",
            "method",
            "methods",
            "approach",
            "experiments",
            "results",
            "evaluation",
            "discussion",
            "conclusion",
            "limitations",
            "references",
        }
        for index, line in enumerate(lines):
            normalized = re.sub(r"[^a-z0-9]", "", line.lower())
            if normalized in {"abstract", "##abstract", "#abstract"}:
                abstract_start = index + 1
                break
        if abstract_start is None:
            return ""

        collected: list[str] = []
        for line in lines[abstract_start:]:
            stripped = line.strip()
            if not stripped:
                if collected:
                    break
                continue
            normalized = re.sub(r"[^a-z0-9]", "", stripped.lower().lstrip("#").strip())
            if normalized in section_markers and collected:
                break
            if stripped.startswith("<!--"):
                continue
            collected.append(stripped.lstrip("#").strip())
        abstract_text = " ".join(collected).strip()
        return re.sub(r"\s+", " ", abstract_text)

    def _generate_summary_and_keywords(self, title: str, abstract_text: str) -> tuple[str, list[str]]:
        if self.settings.llm_backend == "mock":
            return self._mock_profile(title, abstract_text)

        llm = self._chat_model(temperature=0)
        system_prompt = self.prompt_loader.load("paper_profile_system.txt")
        response = llm.invoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(
                    content=json.dumps(
                        {
                            "title": title,
                            "abstract": abstract_text,
                        },
                        ensure_ascii=False,
                    )
                ),
            ]
        )
        payload = self._safe_json_loads(str(response.content))
        summary = str(payload.get("short_summary", "")).strip()
        keywords = [str(item).strip() for item in payload.get("keywords", []) if str(item).strip()]
        if not summary:
            raise ValueError("Profile generation returned empty short_summary.")
        return summary, keywords[:8]

    def _mock_profile(self, title: str, abstract_text: str) -> tuple[str, list[str]]:
        sentences = re.split(r"(?<=[.!?])\s+", abstract_text.strip())
        summary = " ".join(sentence for sentence in sentences[:2] if sentence).strip() or abstract_text[:240]
        tokens = re.findall(r"[A-Za-z][A-Za-z0-9_-]+", f"{title} {abstract_text}".lower())
        stopwords = {
            "this",
            "paper",
            "study",
            "studies",
            "with",
            "from",
            "that",
            "into",
            "their",
            "using",
            "based",
            "towards",
            "approach",
            "method",
        }
        keywords: list[str] = []
        for token in tokens:
            if token in stopwords or len(token) < 4 or token in keywords:
                continue
            keywords.append(token)
            if len(keywords) >= 6:
                break
        return summary, keywords

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
