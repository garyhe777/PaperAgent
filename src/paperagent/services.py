from __future__ import annotations

from functools import cached_property

from paperagent.agent.service import PaperChatAgent
from paperagent.config import Settings
from paperagent.ingest.service import IngestService
from paperagent.ppt.service import PPTService
from paperagent.retrieval.service import HybridRetrievalService
from paperagent.storage.database import Database
from paperagent.storage.repositories import ChunkRepository, PaperRepository


class ServiceContainer:
    """Dependency container used by CLI, tests, and the future API layer."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    @cached_property
    def database(self) -> Database:
        database = Database(self.settings.database_path)
        database.initialize()
        return database

    @cached_property
    def paper_repository(self) -> PaperRepository:
        return PaperRepository(self.database)

    @cached_property
    def chunk_repository(self) -> ChunkRepository:
        return ChunkRepository(self.database)

    @cached_property
    def retrieval_service(self) -> HybridRetrievalService:
        return HybridRetrievalService(
            settings=self.settings,
            chunk_repository=self.chunk_repository,
        )

    @cached_property
    def ingest_service(self) -> IngestService:
        return IngestService(
            settings=self.settings,
            paper_repository=self.paper_repository,
            chunk_repository=self.chunk_repository,
            retrieval_service=self.retrieval_service,
        )

    @cached_property
    def chat_agent(self) -> PaperChatAgent:
        return PaperChatAgent(
            settings=self.settings,
            paper_repository=self.paper_repository,
            retrieval_service=self.retrieval_service,
        )

    @cached_property
    def ppt_service(self) -> PPTService:
        return PPTService(
            settings=self.settings,
            paper_repository=self.paper_repository,
            chunk_repository=self.chunk_repository,
        )
