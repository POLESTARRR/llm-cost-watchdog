"""Tracker: persistence, schema migration, and batch loading.

The migration tests matter most, a cost watchdog that drops your spend
history on upgrade has destroyed the only thing it exists to keep.
"""

import sqlite3

import pytest

from src import tracker
from src.usage_schema import UsageEvent

# The v1 table, before provider / cache columns existed.
V1_SCHEMA = """
CREATE TABLE usage_events (
    id TEXT PRIMARY KEY,
    timestamp TEXT NOT NULL,
    model TEXT NOT NULL,
    project_tag TEXT NOT NULL,
    input_tokens INTEGER NOT NULL,
    output_tokens INTEGER NOT NULL,
    cost_usd REAL NOT NULL,
    latency_ms REAL NOT NULL,
    prompt_preview TEXT,
    success INTEGER NOT NULL,
    error TEXT
);
"""


def _make_v1_db(path, rows=3):
    conn = sqlite3.connect(path)
    conn.executescript(V1_SCHEMA)
    for i in range(rows):
        conn.execute(
            "INSERT INTO usage_events VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (f"id-{i}", f"2026-08-0{i + 1}T10:00:00+00:00", "gemini-flash-latest",
             "legacy-project", 100, 50, 0.001, 500.0, "old event", 1, None),
        )
    conn.commit()
    conn.close()


# --- migration -----------------------------------------------------------


def test_migration_preserves_existing_rows(tmp_path):
    db = str(tmp_path / "v1.db")
    _make_v1_db(db, rows=5)

    tracker.init_db(db)

    events = tracker.get_events(db_path=db)
    assert len(events) == 5
    assert {e.project_tag for e in events} == {"legacy-project"}


def test_migration_adds_the_new_columns(tmp_path):
    db = str(tmp_path / "v1.db")
    _make_v1_db(db)

    tracker.init_db(db)

    conn = sqlite3.connect(db)
    cols = {r[1] for r in conn.execute("PRAGMA table_info(usage_events)")}
    conn.close()
    assert {"provider", "cached_input_tokens", "cache_write_tokens"} <= cols


def test_migrated_rows_get_safe_defaults(tmp_path):
    db = str(tmp_path / "v1.db")
    _make_v1_db(db)
    tracker.init_db(db)

    event = tracker.get_events(db_path=db)[0]
    assert event.provider == "unknown"
    assert event.cached_input_tokens == 0
    assert event.cache_write_tokens == 0


def test_migration_is_idempotent(tmp_path):
    """init_db runs on every call, a second run must not error or duplicate."""
    db = str(tmp_path / "v1.db")
    _make_v1_db(db, rows=2)

    for _ in range(3):
        tracker.init_db(db)

    assert len(tracker.get_events(db_path=db)) == 2


def test_can_write_new_events_to_a_migrated_db(tmp_path):
    db = str(tmp_path / "v1.db")
    _make_v1_db(db, rows=1)
    tracker.init_db(db)

    tracker.log_usage(
        UsageEvent(
            model="claude-opus-5", provider="anthropic", project_tag="new",
            input_tokens=500, output_tokens=100, cached_input_tokens=200,
            cache_write_tokens=50, cost_usd=0.01, latency_ms=900.0,
        ),
        db_path=db,
    )

    events = tracker.get_events(db_path=db)
    assert len(events) == 2
    new = next(e for e in events if e.project_tag == "new")
    assert (new.provider, new.cached_input_tokens, new.cache_write_tokens) == ("anthropic", 200, 50)


# --- round-tripping ------------------------------------------------------


def test_log_and_read_round_trip(temp_db):
    tracker.log_usage(
        UsageEvent(
            model="gpt-5.6-sol", provider="openai", project_tag="p",
            input_tokens=1000, output_tokens=200, cached_input_tokens=400,
            cache_write_tokens=100, cost_usd=0.05, latency_ms=1234.5,
            prompt_preview="hello", success=True,
        ),
        db_path=temp_db,
    )
    e = tracker.get_events(db_path=temp_db)[0]
    assert (e.model, e.provider) == ("gpt-5.6-sol", "openai")
    assert (e.cached_input_tokens, e.cache_write_tokens) == (400, 100)
    assert e.latency_ms == pytest.approx(1234.5)


def test_failed_events_are_persisted_with_their_error(temp_db):
    tracker.log_usage(
        UsageEvent(
            model="claude-opus-5", provider="anthropic", input_tokens=0,
            output_tokens=0, cost_usd=0.0, latency_ms=250.0,
            success=False, error="429 rate limit",
        ),
        db_path=temp_db,
    )
    e = tracker.get_events(db_path=temp_db)[0]
    assert e.success is False and "429" in e.error


def test_get_events_filters_by_project(temp_db):
    for tag in ("a", "a", "b"):
        tracker.log_usage(
            UsageEvent(model="claude-opus-5", provider="anthropic", project_tag=tag,
                       input_tokens=1, output_tokens=1, cost_usd=0.0, latency_ms=1.0),
            db_path=temp_db,
        )
    assert len(tracker.get_events(project_tag="a", db_path=temp_db)) == 2


# --- batch load ----------------------------------------------------------


def test_batch_load_infers_provider_when_absent(tmp_path, temp_db):
    import json

    path = tmp_path / "events.json"
    path.write_text(json.dumps([
        {"model": "claude-opus-5", "input_tokens": 100, "output_tokens": 50, "latency_ms": 500},
        {"model": "gpt-5-nano", "input_tokens": 100, "output_tokens": 50, "latency_ms": 500},
    ]))

    tracker.batch_load(str(path), db_path=temp_db)
    assert {e.provider for e in tracker.get_events(db_path=temp_db)} == {"anthropic", "openai"}


def test_batch_load_computes_cost_when_absent(tmp_path, temp_db):
    import json

    path = tmp_path / "events.json"
    path.write_text(json.dumps([
        {"model": "claude-opus-5", "input_tokens": 10_000, "output_tokens": 1000,
         "cached_input_tokens": 8000, "latency_ms": 500}
    ]))

    tracker.batch_load(str(path), db_path=temp_db)
    event = tracker.get_events(db_path=temp_db)[0]
    # Cheaper than the uncached price, because the cached subset was honored.
    from src.pricing import calculate_cost
    assert event.cost_usd == pytest.approx(calculate_cost("claude-opus-5", 10_000, 1000, 8000))
    assert event.cost_usd < calculate_cost("claude-opus-5", 10_000, 1000)
