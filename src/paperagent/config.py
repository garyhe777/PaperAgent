from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Centralized settings loaded from environment variables or a .env file."""

    model_config = SettingsConfigDict(
        env_prefix="PAPERAGENT_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "PaperAgent"
    data_dir: Path = Field(default=Path(".paperagent_data"))
    storage_dir: Path = Field(default=Path(".paperagent_data/storage"))
    database_path: Path = Field(default=Path(".paperagent_data/paperagent.db"))
    chroma_dir: Path = Field(default=Path(".paperagent_data/chroma"))
    bm25_dir: Path = Field(default=Path(".paperagent_data/bm25"))
    deck_dir: Path = Field(default=Path(".paperagent_data/decks"))
    log_level: str = "INFO"
    pdf_backend: str = "datalab"

    llm_backend: str = "openai"
    llm_model: str = "gpt-4.1-mini"
    llm_base_url: str | None = None
    llm_api_key: str | None = None

    embedding_backend: str = "hash"
    embedding_model: str = "text-embedding-3-small"
    embedding_base_url: str | None = None
    embedding_api_key: str | None = None

    datalab_api_key: str | None = None
    datalab_mode: str = "balanced"

    default_top_k: int = 5
    agent_max_tool_iterations: int = 8
    chunk_size: int = 1400
    chunk_overlap: int = 200

    def ensure_directories(self) -> None:
        for path in [
            self.data_dir,
            self.storage_dir,
            self.chroma_dir,
            self.bm25_dir,
            self.deck_dir,
        ]:
            path.mkdir(parents=True, exist_ok=True)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_directories()
    return settings
