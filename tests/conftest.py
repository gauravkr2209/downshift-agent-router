"""Test configuration and fixtures."""

from collections.abc import Generator
from pathlib import Path

import pytest
from sqlmodel import Session, SQLModel, create_engine

import server.app.config as config_mod
import server.app.db as db_mod


@pytest.fixture(autouse=True)
def setup_test_db(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Generator[Session, None, None]:
    """Provide a fresh SQLite database file for each test using monkeypatch."""
    test_db_file = tmp_path / "test.db"
    test_engine = create_engine(
        f"sqlite:///{test_db_file}",
        connect_args={"check_same_thread": False},
    )

    monkeypatch.setattr(config_mod.settings, "database_url", f"sqlite:///{test_db_file}")
    monkeypatch.setattr(db_mod, "engine", test_engine)

    SQLModel.metadata.create_all(test_engine)

    with Session(test_engine) as session:
        yield session

    test_engine.dispose()
