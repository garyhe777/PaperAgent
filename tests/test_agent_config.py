from __future__ import annotations

from paperagent.config import Settings
from paperagent.services import ServiceContainer


def test_agent_max_tool_iterations_comes_from_settings(tmp_path):
    settings = Settings(
        data_dir=tmp_path / "data",
        storage_dir=tmp_path / "data" / "storage",
        database_path=tmp_path / "data" / "paperagent.db",
        chroma_dir=tmp_path / "data" / "chroma",
        bm25_dir=tmp_path / "data" / "bm25",
        deck_dir=tmp_path / "data" / "decks",
        pdf_backend="pymupdf",
        llm_backend="mock",
        embedding_backend="hash",
        agent_max_tool_iterations=11,
    )
    settings.ensure_directories()

    container = ServiceContainer(settings)
    assert container.chat_agent.max_tool_iterations == 11
