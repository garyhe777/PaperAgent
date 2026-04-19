from __future__ import annotations

import json
import re
from pathlib import Path

import chromadb
from rank_bm25 import BM25Okapi

from paperagent.config import Settings
from paperagent.retrieval.embeddings import build_embedding_provider
from paperagent.schemas.models import ChunkRecord, RetrievalResult
from paperagent.storage.repositories import ChunkRepository


class HybridRetrievalService:
    def __init__(self, settings: Settings, chunk_repository: ChunkRepository) -> None:
        self.settings = settings
        self.chunk_repository = chunk_repository
        self.client = chromadb.PersistentClient(path=str(settings.chroma_dir))
        self.embedding_provider = build_embedding_provider(settings)

    def index_paper(self, paper_id: str, chunks: list[ChunkRecord]) -> None:
        collection = self._get_collection(paper_id)
        collection.delete(where={"paper_id": paper_id})
        global_collection = self._get_collection(None)
        global_collection.delete(where={"paper_id": paper_id})
        texts = [chunk.content for chunk in chunks]
        embeddings = self.embedding_provider.embed_documents(texts)
        metadatas = [
            {
                "paper_id": chunk.paper_id,
                "section_title": chunk.section_title,
                "page_number": chunk.page_number,
            }
            for chunk in chunks
        ]
        ids = [chunk.chunk_id for chunk in chunks]
        collection.upsert(ids=ids, documents=texts, embeddings=embeddings, metadatas=metadatas)
        global_collection.upsert(ids=ids, documents=texts, embeddings=embeddings, metadatas=metadatas)
        self._write_bm25_index(paper_id, chunks)

    def retrieve(self, query: str, paper_id: str | None = None, top_k: int | None = None) -> list[RetrievalResult]:
        top_k = top_k or self.settings.default_top_k
        chunks = self.chunk_repository.list_chunks(paper_id) if paper_id else self.chunk_repository.list_all_chunks()
        if not chunks:
            return []

        vector_hits = self._vector_search(query, paper_id, top_k)
        bm25_hits = self._bm25_search(query, paper_id, top_k)

        merged: dict[str, RetrievalResult] = {}
        score_map: dict[str, float] = {}
        for source_results in [vector_hits, bm25_hits]:
            max_score = max((result.score for result in source_results), default=1.0) or 1.0
            for result in source_results:
                normalized = result.score / max_score
                score_map[result.chunk_id] = score_map.get(result.chunk_id, 0.0) + normalized
                if result.chunk_id not in merged:
                    merged[result.chunk_id] = result

        ranked = sorted(
            merged.values(),
            key=lambda item: (score_map.get(item.chunk_id, 0.0), -item.page_number),
            reverse=True,
        )
        final_results: list[RetrievalResult] = []
        for result in ranked[:top_k]:
            final_results.append(
                RetrievalResult(
                    paper_id=result.paper_id,
                    chunk_id=result.chunk_id,
                    content=result.content,
                    section_title=result.section_title,
                    page_number=result.page_number,
                    score=round(score_map.get(result.chunk_id, result.score), 4),
                    source="hybrid",
                )
            )
        return final_results

    def _vector_search(self, query: str, paper_id: str | None, top_k: int) -> list[RetrievalResult]:
        collection = self._get_collection(paper_id)
        query_kwargs = {
            "query_embeddings": [self.embedding_provider.embed_query(query)],
            "n_results": top_k,
        }
        if paper_id:
            query_kwargs["where"] = {"paper_id": paper_id}
        response = collection.query(**query_kwargs)
        ids = response.get("ids", [[]])[0]
        documents = response.get("documents", [[]])[0]
        metadatas = response.get("metadatas", [[]])[0]
        distances = response.get("distances", [[]])[0]
        results: list[RetrievalResult] = []
        for chunk_id, content, metadata, distance in zip(ids, documents, metadatas, distances):
            resolved_paper_id = str(metadata.get("paper_id", paper_id or "unknown"))
            results.append(
                RetrievalResult(
                    paper_id=resolved_paper_id,
                    chunk_id=chunk_id,
                    content=content,
                    section_title=str(metadata.get("section_title", "Document")),
                    page_number=int(metadata.get("page_number", 1)),
                    score=max(0.0, 1.0 - float(distance)),
                    source="vector",
                )
            )
        return results

    def _bm25_search(self, query: str, paper_id: str | None, top_k: int) -> list[RetrievalResult]:
        if paper_id:
            bm25_path = self._bm25_path(paper_id)
            if not bm25_path.exists():
                return []
            payload = json.loads(bm25_path.read_text(encoding="utf-8"))
        else:
            chunks = self.chunk_repository.list_all_chunks()
            if not chunks:
                return []
            payload = {
                "tokens": [self._tokenize(chunk.content) for chunk in chunks],
                "chunks": [
                    {
                        "paper_id": chunk.paper_id,
                        "chunk_id": chunk.chunk_id,
                        "section_title": chunk.section_title,
                        "page_number": chunk.page_number,
                        "content": chunk.content,
                    }
                    for chunk in chunks
                ],
            }
        corpus_tokens = payload["tokens"]
        bm25 = BM25Okapi(corpus_tokens)
        query_tokens = self._tokenize(query)
        scores = bm25.get_scores(query_tokens)
        paired = sorted(
            enumerate(scores),
            key=lambda item: item[1],
            reverse=True,
        )[:top_k]
        results: list[RetrievalResult] = []
        for index, score in paired:
            chunk = payload["chunks"][index]
            results.append(
                RetrievalResult(
                    paper_id=str(chunk.get("paper_id", paper_id or "unknown")),
                    chunk_id=chunk["chunk_id"],
                    content=chunk["content"],
                    section_title=chunk["section_title"],
                    page_number=int(chunk["page_number"]),
                    score=float(score),
                    source="bm25",
                )
            )
        return results

    def _write_bm25_index(self, paper_id: str, chunks: list[ChunkRecord]) -> None:
        payload = {
            "tokens": [self._tokenize(chunk.content) for chunk in chunks],
            "chunks": [
                {
                    "chunk_id": chunk.chunk_id,
                    "section_title": chunk.section_title,
                    "page_number": chunk.page_number,
                    "content": chunk.content,
                }
                for chunk in chunks
            ],
        }
        path = self._bm25_path(paper_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _tokenize(self, text: str) -> list[str]:
        return re.findall(r"\w+", text.lower())

    def _get_collection(self, paper_id: str | None):
        return self.client.get_or_create_collection(name=self._collection_name(paper_id))

    def _collection_name(self, paper_id: str | None) -> str:
        if paper_id is None:
            return "paper_global"
        return "paper_" + re.sub(r"[^a-zA-Z0-9_-]", "_", paper_id)

    def _bm25_path(self, paper_id: str) -> Path:
        return self.settings.bm25_dir / f"{paper_id}.json"
