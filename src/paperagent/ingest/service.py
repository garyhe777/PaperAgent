from __future__ import annotations

import hashlib
import importlib
import json
import shutil
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

import httpx

from paperagent.config import Settings
from paperagent.ingest.chunking import chunk_sections, split_markdown_into_sections
from paperagent.ingest.pdf_parser import build_pdf_markdown_converter
from paperagent.schemas.models import ChunkRecord, PaperRecord
from paperagent.storage.repositories import ChunkRepository, PaperRepository


class IngestService:
    def __init__(
        self,
        settings: Settings,
        paper_repository: PaperRepository,
        chunk_repository: ChunkRepository,
        retrieval_service,
    ) -> None:
        self.settings = settings
        self.paper_repository = paper_repository
        self.chunk_repository = chunk_repository
        self.retrieval_service = retrieval_service

    def ingest(
        self,
        pdf_path: Path | None = None,
        url: str | None = None,
        override_title: str | None = None,
        pdf_backend: str | None = None,
    ) -> dict:
        if not pdf_path and not url:
            raise ValueError("Either pdf_path or url must be provided.")
        if pdf_path and url:
            raise ValueError("Provide only one source: pdf_path or url.")

        source_value = str(pdf_path) if pdf_path else str(url)
        source_type = "pdf" if pdf_path else "url"
        selected_backend = (pdf_backend or self.settings.pdf_backend).strip().lower()
        paper_id = self._paper_id_for(source_value)
        paper_dir = self.settings.storage_dir / paper_id
        paper_dir.mkdir(parents=True, exist_ok=True)

        now = datetime.utcnow()
        record = PaperRecord(
            paper_id=paper_id,
            title=override_title or paper_id,
            source_type=source_type,
            source_value=source_value,
            pdf_path=str(paper_dir / "paper.pdf"),
            md_path=str(paper_dir / "paper.md"),
            ingest_status="processing",
            error_message=None,
            created_at=now,
            updated_at=now,
        )
        existing = self.paper_repository.get_paper(paper_id)
        if (
            existing
            and Path(existing.pdf_path).exists()
            and Path(existing.md_path).exists()
            and self._cached_backend_for(paper_dir) == selected_backend
        ):
            return {
                "paper_id": existing.paper_id,
                "title": existing.title,
                "status": "cached",
                "pdf_path": existing.pdf_path,
                "md_path": existing.md_path,
                "pdf_backend": selected_backend,
            }

        self.paper_repository.upsert_paper(record)

        try:
            final_pdf_path = paper_dir / "paper.pdf"
            if pdf_path:
                shutil.copyfile(pdf_path, final_pdf_path)
            else:
                self._download_pdf(url or "", final_pdf_path)
            converter = build_pdf_markdown_converter(self.settings, backend=selected_backend)
            title, markdown_text = converter.convert(final_pdf_path)
            markdown_path = paper_dir / "paper.md"
            markdown_path.write_text(markdown_text, encoding="utf-8")
            self._write_ingest_meta(paper_dir, selected_backend)

            sections = split_markdown_into_sections(markdown_text)
            chunk_blocks = chunk_sections(
                sections=sections,
                chunk_size=self.settings.chunk_size,
                chunk_overlap=self.settings.chunk_overlap,
            )
            chunks = [
                ChunkRecord(
                    paper_id=paper_id,
                    chunk_id=f"{paper_id}-{index:04d}",
                    section_title=block.section_title,
                    page_number=block.page_number,
                    content=block.content,
                    token_count=len(block.content.split()),
                )
                for index, block in enumerate(chunk_blocks, start=1)
            ]
            self.chunk_repository.replace_chunks(paper_id, chunks)
            self.retrieval_service.index_paper(paper_id, chunks)

            completed = PaperRecord(
                paper_id=paper_id,
                title=override_title or title,
                source_type=source_type,
                source_value=source_value,
                pdf_path=str(final_pdf_path),
                md_path=str(markdown_path),
                ingest_status="completed",
                error_message=None,
                created_at=existing.created_at if existing else now,
                updated_at=datetime.utcnow(),
            )
            self.paper_repository.upsert_paper(completed)
            return {
                "paper_id": paper_id,
                "title": completed.title,
                "status": completed.ingest_status,
                "pdf_path": completed.pdf_path,
                "md_path": completed.md_path,
                "chunk_count": len(chunks),
                "pdf_backend": selected_backend,
            }
        except Exception as exc:  # noqa: BLE001
            failed = PaperRecord(
                paper_id=paper_id,
                title=override_title or record.title,
                source_type=source_type,
                source_value=source_value,
                pdf_path=record.pdf_path,
                md_path=record.md_path,
                ingest_status="failed",
                error_message=str(exc),
                created_at=existing.created_at if existing else now,
                updated_at=datetime.utcnow(),
            )
            self.paper_repository.upsert_paper(failed)
            raise

    def doctor(self) -> list[tuple[str, str, str]]:
        report: list[tuple[str, str, str]] = []
        for module_name in ["fitz", "chromadb", "rank_bm25", "pptx", "fastapi"]:
            try:
                importlib.import_module(module_name)
                report.append((module_name, "OK", "module available"))
            except ModuleNotFoundError as exc:
                report.append((module_name, "MISSING", str(exc)))

        report.append(
            (
                "database",
                "OK" if self.settings.database_path.parent.exists() else "MISSING",
                str(self.settings.database_path),
            )
        )
        report.append(
            (
                "llm_backend",
                "OK",
                json.dumps(
                    {
                        "backend": self.settings.llm_backend,
                        "model": self.settings.llm_model,
                        "base_url": self.settings.llm_base_url,
                    },
                    ensure_ascii=False,
                ),
            )
        )
        report.append(
            (
                "embedding_backend",
                "OK",
                json.dumps(
                    {
                        "backend": self.settings.embedding_backend,
                        "model": self.settings.embedding_model,
                    },
                    ensure_ascii=False,
                ),
            )
        )
        datalab_enabled = self.settings.pdf_backend == "datalab"
        try:
            importlib.import_module("datalab_sdk")
            datalab_module_status = "OK"
            datalab_module_detail = "module available"
        except ModuleNotFoundError as exc:
            datalab_module_status = "MISSING"
            datalab_module_detail = str(exc)

        report.append(("datalab_sdk", datalab_module_status, datalab_module_detail))
        report.append(
            (
                "pdf_backend",
                "OK",
                json.dumps(
                    {
                        "backend": self.settings.pdf_backend,
                        "datalab_mode": self.settings.datalab_mode,
                    },
                    ensure_ascii=False,
                ),
            )
        )
        report.append(
            (
                "datalab_api_key",
                "OK" if self.settings.datalab_api_key else ("MISSING" if datalab_enabled else "OPTIONAL"),
                "configured" if self.settings.datalab_api_key else "not configured",
            )
        )
        return report

    def _download_pdf(self, url: str, output_path: Path) -> None:
        with httpx.Client(timeout=30.0, follow_redirects=True, trust_env=False) as client:
            response = client.get(url)
            response.raise_for_status()
            output_path.write_bytes(response.content)

    def _paper_id_for(self, source_value: str) -> str:
        parsed = urlparse(source_value)
        base = Path(parsed.path).stem if parsed.scheme else Path(source_value).stem
        digest = hashlib.sha1(source_value.encode("utf-8")).hexdigest()[:8]
        safe = "".join(char if char.isalnum() else "-" for char in base.lower()).strip("-")
        return f"{safe or 'paper'}-{digest}"

    def _cached_backend_for(self, paper_dir: Path) -> str | None:
        meta_path = paper_dir / "ingest_meta.json"
        if not meta_path.exists():
            return self.settings.pdf_backend
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None
        backend = str(meta.get("pdf_backend", "")).strip().lower()
        return backend or None

    def _write_ingest_meta(self, paper_dir: Path, pdf_backend: str) -> None:
        meta_path = paper_dir / "ingest_meta.json"
        meta_path.write_text(
            json.dumps({"pdf_backend": pdf_backend}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
