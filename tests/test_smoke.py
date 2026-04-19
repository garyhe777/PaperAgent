from paperagent.config import Settings
from paperagent.storage.database import Database


def test_settings_create_directories(tmp_path):
    settings = Settings(
        data_dir=tmp_path / "data",
        storage_dir=tmp_path / "data" / "storage",
        database_path=tmp_path / "data" / "paperagent.db",
        chroma_dir=tmp_path / "data" / "chroma",
        bm25_dir=tmp_path / "data" / "bm25",
        deck_dir=tmp_path / "data" / "decks",
    )
    settings.ensure_directories()
    assert settings.database_path.parent.exists()
    assert settings.storage_dir.exists()


def test_database_initialize(tmp_path):
    database_path = tmp_path / "paperagent.db"
    database = Database(database_path)
    database.initialize()
    assert database_path.exists()
