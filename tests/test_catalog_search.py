from __future__ import annotations

from datetime import datetime, timedelta

from paperagent.schemas.models import PaperProfileRecord, PaperRecord
from paperagent.services import ServiceContainer


def test_catalog_search_returns_matching_papers(container: ServiceContainer):
    now = datetime.utcnow()
    container.paper_repository.upsert_paper(
        PaperRecord(
            paper_id="tag-wm",
            title="TAG-WM: Tamper-Aware Generative Image Watermarking",
            source_type="pdf",
            source_value="tag.pdf",
            pdf_path="tag.pdf",
            md_path="tag.md",
            ingest_status="completed",
            error_message=None,
            created_at=now,
            updated_at=now,
        )
    )
    container.paper_repository.upsert_profile(
        PaperProfileRecord(
            paper_id="tag-wm",
            abstract_text="This paper studies watermarking for generative image models.",
            short_summary="A watermarking paper for diffusion image generation.",
            keywords=["watermarking", "diffusion", "tamper-aware"],
            profile_status="completed",
            profile_error=None,
            profile_updated_at=now,
        )
    )
    container.paper_repository.upsert_paper(
        PaperRecord(
            paper_id="roar",
            title="ROAR: Reducing Inversion Error in Generative Image Watermarking",
            source_type="pdf",
            source_value="roar.pdf",
            pdf_path="roar.pdf",
            md_path="roar.md",
            ingest_status="completed",
            error_message=None,
            created_at=now + timedelta(seconds=1),
            updated_at=now + timedelta(seconds=1),
        )
    )
    container.paper_repository.upsert_profile(
        PaperProfileRecord(
            paper_id="roar",
            abstract_text="This paper reduces inversion error in watermarking pipelines.",
            short_summary="It improves inversion quality for watermarking systems.",
            keywords=["watermarking", "inversion", "image"],
            profile_status="completed",
            profile_error=None,
            profile_updated_at=now + timedelta(seconds=1),
        )
    )

    results = container.paper_catalog_service.search_papers("tamper aware watermarking", top_k=2)
    assert results
    assert results[0].paper_id == "tag-wm"
    assert results[0].short_summary
    assert "watermarking" in results[0].keywords


def test_catalog_search_falls_back_to_title_without_profile(container: ServiceContainer):
    now = datetime.utcnow()
    container.paper_repository.upsert_paper(
        PaperRecord(
            paper_id="seal",
            title="SEAL: Semantic Aware Image Watermarking",
            source_type="pdf",
            source_value="seal.pdf",
            pdf_path="seal.pdf",
            md_path="seal.md",
            ingest_status="completed",
            error_message=None,
            created_at=now,
            updated_at=now,
        )
    )

    results = container.paper_catalog_service.search_papers("semantic watermarking", top_k=3)
    assert results
    assert results[0].paper_id == "seal"
