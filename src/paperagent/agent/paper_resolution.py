from __future__ import annotations

import re
from dataclasses import dataclass


PPT_TRIGGER_TOKENS = (
    "ppt",
    "slide",
    "slides",
    "presentation",
    "deck",
    "幻灯片",
    "演示",
    "汇报",
)


@dataclass(slots=True)
class ResolvedPaperTarget:
    paper_id: str | None
    paper_title: str | None
    source: str


def is_ppt_request(text: str) -> bool:
    normalized = text.strip().lower()
    if not normalized:
        return False
    return any(token in normalized for token in PPT_TRIGGER_TOKENS)


def resolve_ppt_target(
    *,
    prompt: str,
    scoped_paper_id: str | None,
    paper_repository,
    paper_catalog_service,
) -> ResolvedPaperTarget:
    if scoped_paper_id:
        paper = paper_repository.get_paper(scoped_paper_id)
        if paper:
            return ResolvedPaperTarget(
                paper_id=paper.paper_id,
                paper_title=paper.title,
                source="scoped",
            )

    normalized_prompt = prompt.strip().lower()
    papers = paper_repository.list_papers()

    for paper in sorted(papers, key=lambda item: len(item.paper_id), reverse=True):
        if paper.paper_id.lower() in normalized_prompt:
            return ResolvedPaperTarget(
                paper_id=paper.paper_id,
                paper_title=paper.title,
                source="paper_id",
            )

    prompt_alpha = normalize_search_text(normalized_prompt)
    for paper in papers:
        normalized_title = normalize_search_text(paper.title)
        if normalized_title and normalized_title in prompt_alpha:
            return ResolvedPaperTarget(
                paper_id=paper.paper_id,
                paper_title=paper.title,
                source="title",
            )

    if paper_catalog_service:
        catalog_hits = paper_catalog_service.search_papers(prompt, top_k=1)
        if catalog_hits:
            top_hit = catalog_hits[0]
            return ResolvedPaperTarget(
                paper_id=top_hit.paper_id,
                paper_title=top_hit.title,
                source="catalog",
            )

    return ResolvedPaperTarget(
        paper_id=None,
        paper_title=None,
        source="unresolved",
    )


def normalize_search_text(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))
