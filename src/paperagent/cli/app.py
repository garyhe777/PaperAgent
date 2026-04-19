from __future__ import annotations

import json
import re
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from paperagent.config import get_settings
from paperagent.logging_utils import configure_logging
from paperagent.schemas.models import AgentEvent
from paperagent.services import ServiceContainer
from paperagent.web.api import create_app

app = typer.Typer(help="PaperAgent CLI")
chat_app = typer.Typer(help="Chat with an imported paper")
db_app = typer.Typer(help="Inspect the local database")
app.add_typer(chat_app, name="chat")
app.add_typer(db_app, name="db")
console = Console()


def get_container() -> ServiceContainer:
    settings = get_settings()
    configure_logging(settings.log_level)
    return ServiceContainer(settings)


@app.command()
def doctor() -> None:
    container = get_container()
    report = container.ingest_service.doctor()
    table = Table(title="PaperAgent Health Check")
    table.add_column("Check")
    table.add_column("Status")
    table.add_column("Details")
    for name, status, details in report:
        table.add_row(name, status, details)
    console.print(table)


@app.command()
def ingest(
    pdf: Path | None = typer.Option(None, help="Local PDF path"),
    url: str | None = typer.Option(None, help="Remote PDF URL"),
    title: str | None = typer.Option(None, help="Optional custom paper title"),
    pdf_backend: str | None = typer.Option(
        None,
        "--pdf-backend",
        help="PDF to Markdown backend: pymupdf or datalab",
    ),
) -> None:
    container = get_container()
    selected_source = str(pdf or url or "")
    selected_backend = pdf_backend or container.settings.pdf_backend
    status_message = (
        f"Ingesting {selected_source or 'paper'} with backend '{selected_backend}'..."
    )
    with console.status(status_message, spinner="dots"):
        result = container.ingest_service.ingest(
            pdf_path=pdf,
            url=url,
            override_title=title,
            pdf_backend=pdf_backend,
        )
    console.print_json(json.dumps(result))


@chat_app.command("ask")
def chat_ask(
    paper_id: str | None = typer.Option(None, help="Optional paper identifier"),
    session_id: str | None = typer.Option(None, help="Optional chat session identifier"),
    question: str = typer.Option(..., help="Question to ask"),
    style: str = typer.Option("beginner", help="Answer style"),
) -> None:
    container = get_container()
    for event in container.chat_agent.ask(
        paper_id=paper_id,
        question=question,
        style=style,
        session_id=session_id,
    ):
        render_event(event)


@chat_app.command("interactive")
def chat_interactive(
    paper_id: str | None = typer.Option(None, help="Optional paper identifier"),
    session_id: str | None = typer.Option(None, help="Optional chat session identifier"),
    style: str = typer.Option("beginner", help="Answer style"),
) -> None:
    container = get_container()
    current_session_id = session_id
    if paper_id:
        console.print(f"[bold green]Start chatting with paper {paper_id}[/bold green]")
    else:
        console.print("[bold green]Start general chat. The agent may search the paper database if needed.[/bold green]")
    while True:
        question = typer.prompt("question", prompt_suffix=" > ")
        if question.strip().lower() in {"exit", "quit"}:
            break
        for event in container.chat_agent.ask(
            paper_id=paper_id,
            question=question,
            style=style,
            session_id=current_session_id,
        ):
            if event.payload.get("session_id"):
                current_session_id = str(event.payload["session_id"])
            render_event(event)


@app.command("pptgen")
def generate_ppt(
    paper_id: str | None = typer.Option(None, help="Optional paper identifier"),
    prompt: str | None = typer.Option(
        None,
        help='Optional natural language request, e.g. "对 tag-wm 做 ppt"',
    ),
    template: Path | None = typer.Option(None, help="Existing PPTX template"),
) -> None:
    container = get_container()
    resolved_paper_id = _resolve_pptgen_paper_id(
        container=container,
        paper_id=paper_id,
        prompt=prompt,
    )
    result = container.ppt_service.generate(paper_id=resolved_paper_id, template_path=template)
    console.print_json(json.dumps(result))


@db_app.command("papers")
def db_papers() -> None:
    container = get_container()
    table = Table(title="Imported Papers")
    table.add_column("paper_id")
    table.add_column("title")
    table.add_column("status")
    table.add_column("profile")
    table.add_column("pdf")
    for paper in container.paper_repository.list_papers():
        profile = container.paper_repository.get_profile(paper.paper_id)
        table.add_row(
            paper.paper_id,
            paper.title,
            paper.ingest_status,
            profile.profile_status if profile else "missing",
            paper.pdf_path,
        )
    console.print(table)


@db_app.command("profiles")
def db_profiles() -> None:
    container = get_container()
    table = Table(title="Paper Profiles")
    table.add_column("paper_id")
    table.add_column("status")
    table.add_column("keywords")
    table.add_column("short_summary")
    for profile in container.paper_repository.list_profiles():
        table.add_row(
            profile.paper_id,
            profile.profile_status,
            ", ".join(profile.keywords[:5]),
            profile.short_summary[:120],
        )
    console.print(table)


@db_app.command("chunks")
def db_chunks(paper_id: str = typer.Option(..., help="Paper identifier")) -> None:
    container = get_container()
    table = Table(title=f"Chunks for {paper_id}")
    table.add_column("chunk_id")
    table.add_column("section")
    table.add_column("page")
    table.add_column("preview")
    for chunk in container.chunk_repository.list_chunks(paper_id):
        table.add_row(chunk.chunk_id, chunk.section_title, str(chunk.page_number), chunk.content[:80])
    console.print(table)


@app.command()
def serve_api(host: str = "127.0.0.1", port: int = 8000) -> None:
    import uvicorn

    uvicorn.run(create_app(get_container()), host=host, port=port)


def render_event(event: AgentEvent) -> None:
    if event.event_type == "final_answer_stream":
        console.print(event.message, end="")
        return
    if event.event_type == "final_answer_done":
        console.print("")
        return
    console.print(f"[cyan]{event.event_type}[/cyan]: {event.message}")


def _resolve_pptgen_paper_id(
    container: ServiceContainer,
    paper_id: str | None,
    prompt: str | None,
) -> str:
    if paper_id:
        return paper_id
    if not prompt or not prompt.strip():
        raise typer.BadParameter("Provide --paper-id or --prompt for pptgen.")

    normalized_prompt = prompt.strip().lower()
    papers = container.paper_repository.list_papers()

    # First prefer exact paper_id mentions in natural language.
    for paper in sorted(papers, key=lambda item: len(item.paper_id), reverse=True):
        if paper.paper_id.lower() in normalized_prompt:
            return paper.paper_id

    # Then try exact title containment with loose normalization.
    prompt_alpha = _normalize_search_text(normalized_prompt)
    for paper in papers:
        if _normalize_search_text(paper.title) and _normalize_search_text(paper.title) in prompt_alpha:
            return paper.paper_id

    # Finally fall back to catalog search over title/summary/keywords.
    catalog_hits = container.paper_catalog_service.search_papers(prompt, top_k=1)
    if catalog_hits:
        return catalog_hits[0].paper_id
    raise typer.BadParameter(f"Could not infer a paper from prompt: {prompt}")


def _normalize_search_text(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


if __name__ == "__main__":
    app()
