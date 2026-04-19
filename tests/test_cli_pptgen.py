from __future__ import annotations

from datetime import datetime

import pytest

from paperagent.cli.app import _resolve_pptgen_paper_id
from paperagent.schemas.models import PaperProfileRecord, PaperRecord
from paperagent.services import ServiceContainer


def test_pptgen_prompt_resolves_explicit_paper_id(container: ServiceContainer):
    _seed_paper(container, paper_id="tag-wm", title="TAG-WM: Tamper-Aware Generative Image Watermarking")
    resolved = _resolve_pptgen_paper_id(container, paper_id=None, prompt="对 tag-wm 做 ppt")
    assert resolved == "tag-wm"


def test_pptgen_prompt_falls_back_to_catalog_search(container: ServiceContainer):
    _seed_paper(
        container,
        paper_id="roar",
        title="ROAR: Reducing Inversion Error in Generative Image Watermarking",
        summary="A paper about reducing inversion error in watermarking.",
        keywords=["watermarking", "inversion"],
    )
    resolved = _resolve_pptgen_paper_id(container, paper_id=None, prompt="帮我给 reducing inversion error 做个ppt")
    assert resolved == "roar"


def test_pptgen_requires_paper_id_or_prompt(container: ServiceContainer):
    with pytest.raises(Exception):
        _resolve_pptgen_paper_id(container, paper_id=None, prompt=None)


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
