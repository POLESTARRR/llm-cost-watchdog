"""Shared fixtures: every test runs against a temp DB, never data/usage.db."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import tracker  # noqa: E402

SAMPLE_PATH = Path(__file__).resolve().parent.parent / "sample_usage.json"


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    """Redirect all DB access to a fresh temp file via WATCHDOG_DB_PATH."""
    db_path = str(tmp_path / "test_usage.db")
    monkeypatch.setenv("WATCHDOG_DB_PATH", db_path)
    tracker.init_db(db_path)
    return db_path


@pytest.fixture
def sample_db(temp_db):
    """Temp DB preloaded with sample_usage.json (35 events, 2 injected anomalies)."""
    tracker.batch_load(str(SAMPLE_PATH), db_path=temp_db)
    return temp_db
