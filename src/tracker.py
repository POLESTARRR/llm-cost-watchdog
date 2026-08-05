"""
SQLite-backed logging and retrieval of UsageEvents.

Also runnable as a CLI for manual logging (calls made outside our own
wrapped code) and for batch-loading sample_usage.json during testing:

    python -m src.tracker --log-manual --model gemini-flash-latest --cost 0.002 --project teaching-workshop
    python -m src.tracker --batch-load sample_usage.json
"""

import argparse
import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

from src.pricing import calculate_cost
from src.usage_schema import UsageEvent

DEFAULT_DB_PATH = str(Path(__file__).resolve().parent.parent / "data" / "usage.db")


def resolve_db_path(db_path: str | None = None) -> str:
    """Resolve the DB path at call time, so tests and containers can redirect it.

    Precedence: explicit argument > WATCHDOG_DB_PATH env var > DEFAULT_DB_PATH.
    Resolved per-call rather than bound as a default argument, which would
    freeze the path at import time.
    """
    return db_path or os.environ.get("WATCHDOG_DB_PATH") or DEFAULT_DB_PATH

TABLE_SCHEMA = """
CREATE TABLE IF NOT EXISTS usage_events (
    id TEXT PRIMARY KEY,
    timestamp TEXT NOT NULL,
    model TEXT NOT NULL,
    provider TEXT NOT NULL DEFAULT 'unknown',
    project_tag TEXT NOT NULL,
    input_tokens INTEGER NOT NULL,
    output_tokens INTEGER NOT NULL,
    cached_input_tokens INTEGER NOT NULL DEFAULT 0,
    cache_write_tokens INTEGER NOT NULL DEFAULT 0,
    cost_usd REAL NOT NULL,
    latency_ms REAL NOT NULL,
    prompt_preview TEXT,
    success INTEGER NOT NULL,
    error TEXT
);
"""

# Indexes are created AFTER migrations run — an index on a column that a
# pre-migration table doesn't have yet would fail the whole script.
INDEX_SCHEMA = """
CREATE INDEX IF NOT EXISTS idx_usage_events_timestamp ON usage_events(timestamp);
CREATE INDEX IF NOT EXISTS idx_usage_events_model ON usage_events(model);
CREATE INDEX IF NOT EXISTS idx_usage_events_provider ON usage_events(provider);
CREATE INDEX IF NOT EXISTS idx_usage_events_project ON usage_events(project_tag);
"""

# Columns added after v1 shipped. A watchdog that loses your cost history on
# upgrade defeats its own purpose, so migrate in place instead of recreating.
_MIGRATIONS = [
    ("provider", "TEXT NOT NULL DEFAULT 'unknown'"),
    ("cached_input_tokens", "INTEGER NOT NULL DEFAULT 0"),
    ("cache_write_tokens", "INTEGER NOT NULL DEFAULT 0"),
]


@contextmanager
def _connect(db_path: str | None = None):
    resolved = resolve_db_path(db_path)
    Path(resolved).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(resolved)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db(db_path: str | None = None) -> None:
    """Create the usage_events table, and migrate an older one in place."""
    with _connect(db_path) as conn:
        conn.executescript(TABLE_SCHEMA)

        existing = {row["name"] for row in conn.execute("PRAGMA table_info(usage_events)")}
        for column, ddl in _MIGRATIONS:
            if column not in existing:
                conn.execute(f"ALTER TABLE usage_events ADD COLUMN {column} {ddl}")

        conn.executescript(INDEX_SCHEMA)


def log_usage(event: UsageEvent, db_path: str | None = None) -> None:
    """Persist a single UsageEvent to SQLite."""
    init_db(db_path)
    with _connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO usage_events
                (id, timestamp, model, provider, project_tag,
                 input_tokens, output_tokens, cached_input_tokens, cache_write_tokens,
                 cost_usd, latency_ms, prompt_preview, success, error)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event.id,
                event.timestamp,
                event.model,
                event.provider,
                event.project_tag,
                event.input_tokens,
                event.output_tokens,
                event.cached_input_tokens,
                event.cache_write_tokens,
                event.cost_usd,
                event.latency_ms,
                event.prompt_preview,
                int(event.success),
                event.error,
            ),
        )


def get_events(
    start_date: str | None = None,
    end_date: str | None = None,
    project_tag: str | None = None,
    db_path: str | None = None,
) -> list[UsageEvent]:
    """Retrieve UsageEvents, optionally filtered by ISO date range and project_tag.

    start_date/end_date are inclusive ISO-8601 timestamp strings. If omitted,
    that bound is unlimited.
    """
    init_db(db_path)
    query = "SELECT * FROM usage_events WHERE 1=1"
    params: list = []
    if start_date:
        query += " AND timestamp >= ?"
        params.append(start_date)
    if end_date:
        query += " AND timestamp <= ?"
        params.append(end_date)
    if project_tag:
        query += " AND project_tag = ?"
        params.append(project_tag)
    query += " ORDER BY timestamp ASC"

    with _connect(db_path) as conn:
        rows = conn.execute(query, params).fetchall()

    return [
        UsageEvent(
            id=row["id"],
            timestamp=row["timestamp"],
            model=row["model"],
            provider=_row_get(row, "provider", "unknown"),
            project_tag=row["project_tag"],
            input_tokens=row["input_tokens"],
            output_tokens=row["output_tokens"],
            cached_input_tokens=_row_get(row, "cached_input_tokens", 0),
            cache_write_tokens=_row_get(row, "cache_write_tokens", 0),
            cost_usd=row["cost_usd"],
            latency_ms=row["latency_ms"],
            prompt_preview=row["prompt_preview"] or "",
            success=bool(row["success"]),
            error=row["error"],
        )
        for row in rows
    ]


def _row_get(row: sqlite3.Row, key: str, default):
    """Read a column that may not exist on a pre-migration row."""
    try:
        value = row[key]
    except IndexError:
        return default
    return default if value is None else value


def _period_start(period: str) -> str:
    """Return the ISO start timestamp for a named period, anchored to now (UTC)."""
    now = datetime.now(timezone.utc)
    if period == "today":
        start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    elif period == "week":
        start = now - timedelta(days=7)
    elif period == "month":
        start = now - timedelta(days=30)
    elif period == "all_time":
        start = datetime(1970, 1, 1, tzinfo=timezone.utc)
    else:
        raise ValueError(f"Unknown period: {period}")
    return start.isoformat()


def get_events_for_period(period: str, project_tag: str | None = None, db_path: str | None = None) -> list[UsageEvent]:
    """Convenience wrapper: get_events() for a named period ('today'|'week'|'month'|'all_time')."""
    start = _period_start(period)
    return get_events(start_date=start, project_tag=project_tag, db_path=db_path)


def batch_load(json_path: str, db_path: str | None = None) -> int:
    """Load events from a sample_usage.json-style file for testing.

    Each entry may omit cost_usd (computed via pricing.py), timestamp
    (defaults to now), and id (auto-generated). Returns the count loaded.
    """
    with open(json_path) as f:
        raw_events = json.load(f)

    count = 0
    for raw in raw_events:
        model = raw["model"]
        cached = raw.get("cached_input_tokens", 0)
        written = raw.get("cache_write_tokens", 0)

        cost_usd = raw.get("cost_usd")
        if cost_usd is None:
            cost_usd = calculate_cost(
                model, raw["input_tokens"], raw["output_tokens"], cached, written
            )

        event = UsageEvent(
            model=model,
            provider=raw.get("provider") or _safe_provider(model),
            project_tag=raw.get("project_tag", "default"),
            input_tokens=raw["input_tokens"],
            output_tokens=raw["output_tokens"],
            cached_input_tokens=cached,
            cache_write_tokens=written,
            cost_usd=cost_usd,
            latency_ms=raw["latency_ms"],
            prompt_preview=UsageEvent.make_preview(raw.get("prompt_preview", "")),
            success=raw.get("success", True),
            error=raw.get("error"),
            **({"timestamp": raw["timestamp"]} if "timestamp" in raw else {}),
        )
        log_usage(event, db_path=db_path)
        count += 1
    return count


def _safe_provider(model: str) -> str:
    from src.providers import infer_provider

    try:
        return infer_provider(model)
    except Exception:
        return "unknown"


def _main() -> None:
    parser = argparse.ArgumentParser(description="Manually log usage events or batch-load sample data.")
    parser.add_argument("--log-manual", action="store_true", help="Log a single manual usage event")
    parser.add_argument("--model", type=str, help="Model name")
    parser.add_argument("--cost", type=float, help="Cost in USD")
    parser.add_argument("--tokens", type=int, default=0, help="Total tokens (logged as output_tokens)")
    parser.add_argument("--project", type=str, default="default", help="Project tag")
    parser.add_argument("--note", type=str, default="", help="Optional note, stored as prompt_preview")
    parser.add_argument("--batch-load", type=str, help="Path to a sample_usage.json-style file to load")
    parser.add_argument("--db-path", type=str, default=None, help="Override the SQLite DB path")
    args = parser.parse_args()

    if args.log_manual:
        if not args.model or args.cost is None:
            parser.error("--log-manual requires --model and --cost")
        event = UsageEvent(
            model=args.model,
            provider=_safe_provider(args.model),
            project_tag=args.project,
            input_tokens=0,
            output_tokens=args.tokens,
            cost_usd=args.cost,
            latency_ms=0.0,
            prompt_preview=UsageEvent.make_preview(args.note or "manual entry"),
            success=True,
        )
        log_usage(event, db_path=args.db_path)
        print(f"logged manual entry {event.id} | model={event.model} cost=${event.cost_usd:.6f} project={event.project_tag}")
    elif args.batch_load:
        n = batch_load(args.batch_load, db_path=args.db_path)
        print(f"loaded {n} events from {args.batch_load}")
    else:
        parser.print_help()


if __name__ == "__main__":
    _main()
