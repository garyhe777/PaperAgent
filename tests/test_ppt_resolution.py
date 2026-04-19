from __future__ import annotations

from datetime import datetime

from paperagent.agent.paper_resolution import is_ppt_request, resolve_ppt_target
from paperagent.schemas.models import PaperProfileRecord, PaperRecord
from paperagent.services import ServiceContainer


def test_ppt_request_detection_matches_chat_triggers():
    assert is_ppt_request("给这篇论文做个 PPT")
    assert is_ppt_request("Please build presentation slides for this paper")
    assert not is_ppt_request("Explain the method")


def test_ppt_resolution_prefers_scoped_paper(container: ServiceContainer):
    _seed_paper(
        container,
        paper_id="tag-wm",
        title="TAG-WM: Tamper-Aware Generative Image Watermarking",
    )
    _seed_paper(
        container,
        paper_id="roar",
        title="ROAR: Reducing Inversion Error in Generative Image Watermarking",
    )
    resolved = resolve_ppt_target(
        prompt="请给 ROAR 做个 PPT",
        scoped_paper_id="tag-wm",
        paper_repository=container.paper_repository,
        paper_catalog_service=container.paper_catalog_service,
    )
    assert resolved.paper_id == "tag-wm"
    assert resolved.source == "scoped"


def test_ppt_resolution_falls_back_to_catalog_search(container: ServiceContainer):
    _seed_paper(
        container,
        paper_id="roar",
        title="ROAR: Reducing Inversion Error in Generative Image Watermarking",
        summary="A paper about reducing inversion error in watermarking.",
        keywords=["watermarking", "inversion"],
    )
    resolved = resolve_ppt_target(
        prompt="帮我给 reducing inversion error 做个ppt",
        scoped_paper_id=None,
        paper_repository=container.paper_repository,
        paper_catalog_service=container.paper_catalog_service,
    )
    assert resolved.paper_id == "roar"
    assert resolved.source == "catalog"


def test_ppt_resolution_returns_unresolved_when_no_paper_matches(container: ServiceContainer):
    resolved = resolve_ppt_target(
        prompt="帮我做个 PPT",
        scoped_paper_id=None,
        paper_repository=container.paper_repository,
        paper_catalog_service=container.paper_catalog_service,
    )
    assert resolved.paper_id is None
    assert resolved.source == "unresolved"


def _seed_paper(
    container: ServiceContainer,
    paper_id: str,
    title: str,
    summary: str = "",
    keywords: list[str] | None = None,
) -> None:
    now = datetime.utcnow()
    container.paper_repository.upsert_paper(
        PaperRecord(
            paper_id=paper_id,
            title=title,
            source_type="pdf",
            source_value=f"{paper_id}.pdf",
            pdf_path=f"{paper_id}.pdf",
            md_path=f"{paper_id}.md",
            ingest_status="completed",
            error_message=None,
            created_at=now,
            updated_at=now,
        )
    )
    container.paper_repository.upsert_profile(
        PaperProfileRecord(
            paper_id=paper_id,
            abstract_text="",
            short_summary=summary,
            keywords=keywords or [],
            profile_status="completed",
            profile_error=None,
            profile_updated_at=now,
        )
    )
