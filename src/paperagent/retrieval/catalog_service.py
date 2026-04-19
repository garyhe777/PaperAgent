from __future__ import annotations

import re
from collections import Counter

from rank_bm25 import BM25Okapi

from paperagent.schemas.models import PaperCatalogResult
from paperagent.storage.repositories import PaperRepository


class PaperCatalogSearchService:
    def __init__(self, paper_repository: PaperRepository) -> None:
        self.paper_repository = paper_repository

    def search_papers(self, query: str, top_k: int = 5) -> list[PaperCatalogResult]:
        papers = self.paper_repository.list_papers()
        if not papers:
            return []

        candidates: list[tuple[str, str, str, list[str]]] = []
        for paper in papers:
            profile = self.paper_repository.get_profile(paper.paper_id)
            summary = profile.short_summary if profile else ""
            keywords = profile.keywords if profile else []
            search_text = " ".join(
                [
                    paper.title,
                    profile.abstract_text if profile else "",
                    summary,
                    " ".join(keywords),
                ]
            ).strip()
            candidates.append((paper.paper_id, paper.title, summary, keywords, search_text))

        tokenized_corpus = [self._tokenize(item[4]) for item in candidates]
        if not any(tokenized_corpus):
            return []
        bm25 = BM25Okapi(tokenized_corpus)
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        scores = bm25.get_scores(query_tokens)
        query_counter = Counter(query_tokens)
        ranked = sorted(
            enumerate(scores),
            key=lambda item: (item[1], self._title_overlap_bonus(candidates[item[0]][1], query_counter)),
            reverse=True,
        )[:top_k]

        results: list[PaperCatalogResult] = []
        for index, score in ranked:
            paper_id, title, summary, keywords, _ = candidates[index]
            results.append(
                PaperCatalogResult(
                    paper_id=paper_id,
                    title=title,
                    short_summary=summary,
                    keywords=keywords,
                    score=round(float(score), 4),
                )
            )
        return results

    def _tokenize(self, text: str) -> list[str]:
        return re.findall(r"\w+", text.lower())

    def _title_overlap_bonus(self, title: str, query_counter: Counter[str]) -> int:
        title_tokens = set(self._tokenize(title))
        return sum(query_counter[token] for token in title_tokens if token in query_counter)
