from __future__ import annotations

from datetime import datetime

from paperagent.schemas.models import PaperProfileRecord
from paperagent.services import ServiceContainer


def test_database_initializes_paper_profiles_table(container: ServiceContainer):
    with container.database.connect() as connection:
        row = connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'paper_profiles'"
        ).fetchone()
    assert row is not None


def test_paper_repository_can_roundtrip_profile(container: ServiceContainer):
    profile = PaperProfileRecord(
        paper_id="paper-1",
        abstract_text="This paper studies watermarking.",
        short_summary="A short summary.",
        keywords=["watermark", "diffusion"],
        profile_status="completed",
        profile_error=None,
        profile_updated_at=datetime.utcnow(),
    )
    container.paper_repository.upsert_profile(profile)

    loaded = container.paper_repository.get_profile("paper-1")
    assert loaded is not None
    assert loaded.paper_id == "paper-1"
    assert loaded.keywords == ["watermark", "diffusion"]
    assert loaded.profile_status == "completed"
