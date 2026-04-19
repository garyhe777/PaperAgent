from __future__ import annotations

from datetime import datetime
from sqlite3 import Row

from paperagent.schemas.models import ChunkRecord, PaperRecord
from paperagent.storage.database import Database


def _paper_from_row(row: Row) -> PaperRecord:
    return PaperRecord(
        paper_id=row["paper_id"],
        title=row["title"],
        source_type=row["source_type"],
        source_value=row["source_value"],
        pdf_path=row["pdf_path"],
        md_path=row["md_path"],
        ingest_status=row["ingest_status"],
        error_message=row["error_message"],
        created_at=datetime.fromisoformat(row["created_at"]),
        updated_at=datetime.fromisoformat(row["updated_at"]),
    )


def _chunk_from_row(row: Row) -> ChunkRecord:
    return ChunkRecord(
        paper_id=row["paper_id"],
        chunk_id=row["chunk_id"],
        section_title=row["section_title"],
        page_number=row["page_number"],
        content=row["content"],
        token_count=row["token_count"],
    )


class PaperRepository:
    def __init__(self, database: Database) -> None:
        self.database = database

    def upsert_paper(self, paper: PaperRecord) -> None:
        with self.database.connect() as connection:
            connection.execute(
                """
                INSERT INTO papers (
                    paper_id, title, source_type, source_value, pdf_path, md_path,
                    ingest_status, error_message, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(paper_id) DO UPDATE SET
                    title=excluded.title,
                    source_type=excluded.source_type,
                    source_value=excluded.source_value,
                    pdf_path=excluded.pdf_path,
                    md_path=excluded.md_path,
                    ingest_status=excluded.ingest_status,
                    error_message=excluded.error_message,
                    updated_at=excluded.updated_at
                """,
                (
                    paper.paper_id,
                    paper.title,
                    paper.source_type,
                    paper.source_value,
                    paper.pdf_path,
                    paper.md_path,
                    paper.ingest_status,
                    paper.error_message,
                    paper.created_at.isoformat(),
                    paper.updated_at.isoformat(),
                ),
            )

    def get_paper(self, paper_id: str) -> PaperRecord | None:
        with self.database.connect() as connection:
            row = connection.execute(
                "SELECT * FROM papers WHERE paper_id = ?",
                (paper_id,),
            ).fetchone()
        return _paper_from_row(row) if row else None

    def list_papers(self) -> list[PaperRecord]:
        with self.database.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM papers ORDER BY created_at DESC"
            ).fetchall()
        return [_paper_from_row(row) for row in rows]


class ChunkRepository:
    def __init__(self, database: Database) -> None:
        self.database = database

    def replace_chunks(self, paper_id: str, chunks: list[ChunkRecord]) -> None:
        with self.database.connect() as connection:
            connection.execute("DELETE FROM chunks WHERE paper_id = ?", (paper_id,))
            connection.executemany(
                """
                INSERT INTO chunks (paper_id, chunk_id, section_title, page_number, content, token_count)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        chunk.paper_id,
                        chunk.chunk_id,
                        chunk.section_title,
                        chunk.page_number,
                        chunk.content,
                        chunk.token_count,
                    )
                    for chunk in chunks
                ],
            )

    def list_chunks(self, paper_id: str) -> list[ChunkRecord]:
        with self.database.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM chunks WHERE paper_id = ? ORDER BY chunk_id",
                (paper_id,),
            ).fetchall()
        return [_chunk_from_row(row) for row in rows]
