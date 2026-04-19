from __future__ import annotations

import json
from pathlib import Path
from tempfile import NamedTemporaryFile

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel


class IngestRequest(BaseModel):
    pdf: str | None = None
    url: str | None = None
    title: str | None = None


class ChatRequest(BaseModel):
    paper_id: str
    question: str
    style: str = "beginner"


class PPTRequest(BaseModel):
    paper_id: str
    template_path: str | None = None


def create_app(container) -> FastAPI:
    app = FastAPI(title="PaperAgent API")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.get("/papers")
    def list_papers() -> list[dict]:
        return [
            {
                "paper_id": paper.paper_id,
                "title": paper.title,
                "ingest_status": paper.ingest_status,
            }
            for paper in container.paper_repository.list_papers()
        ]

    @app.get("/papers/{paper_id}")
    def get_paper(paper_id: str) -> dict:
        paper = container.paper_repository.get_paper(paper_id)
        if not paper:
            raise HTTPException(status_code=404, detail="Paper not found")
        return {
            "paper_id": paper.paper_id,
            "title": paper.title,
            "source_type": paper.source_type,
            "source_value": paper.source_value,
            "pdf_path": paper.pdf_path,
            "md_path": paper.md_path,
            "ingest_status": paper.ingest_status,
            "error_message": paper.error_message,
        }

    @app.post("/ingest")
    def ingest(request: IngestRequest) -> dict:
        try:
            return container.ingest_service.ingest(
                pdf_path=Path(request.pdf) if request.pdf else None,
                url=request.url,
                override_title=request.title,
            )
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/ingest/upload")
    async def ingest_upload(file: UploadFile = File(...)) -> dict:
        temp_path: Path | None = None
        suffix = Path(file.filename or "paper.pdf").suffix or ".pdf"
        with NamedTemporaryFile(delete=False, suffix=suffix) as handle:
            handle.write(await file.read())
            temp_path = Path(handle.name)
        try:
            return container.ingest_service.ingest(pdf_path=temp_path)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        finally:
            if temp_path and temp_path.exists():
                temp_path.unlink()

    @app.post("/ppt")
    def generate_ppt(request: PPTRequest) -> dict:
        try:
            return container.ppt_service.generate(
                paper_id=request.paper_id,
                template_path=Path(request.template_path) if request.template_path else None,
            )
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/chat/stream")
    def chat_stream(request: ChatRequest) -> StreamingResponse:
        def event_stream():
            for event in container.chat_agent.ask(
                paper_id=request.paper_id,
                question=request.question,
                style=request.style,
            ):
                yield f"data: {json.dumps({'event_type': event.event_type, 'message': event.message, 'payload': event.payload}, ensure_ascii=False)}\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    return app
