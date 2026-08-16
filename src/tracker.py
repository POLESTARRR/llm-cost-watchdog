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
    cache_write_1h_tokens INTEGER NOT NULL DEFAULT 0,
    cost_usd REAL NOT NULL,
    latency_ms REAL NOT NULL,
    prompt_preview TEXT,
    prompt_hash TEXT,
    service_tier TEXT NOT NULL DEFAULT 'standard',
    success INTEGER NOT NULL,
    error TEXT,
    source TEXT NOT NULL DEFAULT 'live'
);
"""

# Indexes are created AFTER migrations run — an index on a column that a
# pre-migration table doesn't have yet would fail the whole script.
INDEX_SCHEMA = """
CREATE INDEX IF NOT EXISTS idx_usage_events_timestamp ON usage_events(timestamp);
CREATE INDEX IF NOT EXISTS idx_usage_events_model ON usage_events(model);
CREATE INDEX IF NOT EXISTS idx_usage_events_provider ON usage_events(provider);
CREATE INDEX IF NOT EXISTS idx_usage_events_project ON usage_events(project_tag);
CREATE INDEX IF NOT EXISTS idx_usage_events_source ON usage_events(source);
CREATE INDEX IF NOT EXISTS idx_usage_events_prompt_hash ON usage_events(prompt_hash);
"""

# Columns added after v1 shipped. A watchdog that loses your cost history on
# upgrade defeats its own purpose, so migrate in place instead of recreating.
_MIGRATIONS = [
    ("provider", "TEXT NOT NULL DEFAULT 'unknown'"),
    ("cached_input_tokens", "INTEGER NOT NULL DEFAULT 0"),
    ("cache_write_tokens", "INTEGER NOT NULL DEFAULT 0"),
    ("source", "TEXT NOT NULL DEFAULT 'live'"),
    ("cache_write_1h_tokens", "INTEGER NOT NULL DEFAULT 0"),
    ("prompt_hash", "TEXT"),
    ("service_tier", "TEXT NOT NULL DEFAULT 'standard'"),
]

VALID_SOURCES = ("live", "demo", "manual", "subscription")

# The rows representing money that actually left your account, per token.
# Anything *reporting billed spend* filters to this.
#
# `subscription` is deliberately NOT here. Claude Code usage on a Pro/Max plan
# is real tokens doing real work, but it is covered by a flat monthly fee — no
# per-token charge ever occurred. Counting it as billed spend would overstate
# money actually spent by the entire build cost, which is the largest number in
# this database. It is reported instead as list-price-equivalent *value*; see
# LIST_PRICE_SOURCES and analyzer.subscription_roi().
BILLED_SOURCES = "live,manual"

# Every row that represents real token consumption at real published rates,
# whether or not it was metered. This is the honest denominator for "what did
# this work cost to produce" — it just isn't a claim about your bank balance.
LIST_PRICE_SOURCES = "live,manual,subscription"

# The subset of billed rows that represents an ongoing *run rate*: calls this
# wrapper made and measured itself. `manual` rows are real money too, but they
# are backfilled after the fact from an existing record (a Claude Code
# transcript), so they are history, not a rate — importing five build
# transcripts in one afternoon says nothing about what next week will cost.
#
# Anything that compares spend to a *weekly budget*, or extrapolates a burn
# rate, uses this instead of BILLED_SOURCES. Build cost stays fully visible in
# the totals and the per-project breakdown; it just isn't treated as recurring.
RUNTIME_SOURCES = "live"


_turso_conn = None  # process-wide singleton -- see below


@contextmanager
def _connect(db_path: str | None = None):
    # A remote Turso database, when configured, always wins over any local
    # db_path -- there's exactly one remote DB per deployment, so per-call
    # path overrides (used by tests and CLI tools) don't apply to it. Local
    # dev and the test suite never set TURSO_DATABASE_URL, so this branch
    # is inert for them. See src/turso_backend.py for why a wrapper is
    # needed at all rather than using libsql's connection directly.
    #
    # The Turso connection is cached process-wide and never closed here:
    # opening a TursoConnection does a real network sync, and this codebase
    # calls _connect() once per query (not once per request), so reconnecting
    # every call paid that sync twice per query. Confirmed against the real
    # deployment: find_waste() alone issues ~9 queries and was timing out at
    # 45s+ from reconnect overhead alone before this was cached. Writes made
    # outside this process (e.g. a direct migration against the Turso HTTP
    # API) won't be visible until this process restarts -- an accepted
    # tradeoff since normal writes all go through this same cached connection.
    turso_url = os.environ.get("TURSO_DATABASE_URL")
    if turso_url:
        global _turso_conn
        if _turso_conn is None:
            from src.turso_backend import TursoConnection
            _turso_conn = TursoConnection(turso_url, os.environ["TURSO_AUTH_TOKEN"])
        conn = _turso_conn
        should_close = False
    else:
        resolved = resolve_db_path(db_path)
        Path(resolved).parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(resolved)
        conn.row_factory = sqlite3.Row
        should_close = True
    try:
        yield conn
        conn.commit()
    finally:
        if should_close:
            conn.close()


def init_db(db_path: str | None = None) -> None:
    """Create the usage_events table, and migrate an older one in place."""
    with _connect(db_path) as conn:
        conn.executescript(TABLE_SCHEMA)

        existing = {row["name"] for row in conn.execute("PRAGMA table_info(usage_events)")}
        added = []
        for column, ddl in _MIGRATIONS:
            if column not in existing:
                conn.execute(f"ALTER TABLE usage_events ADD COLUMN {column} {ddl}")
                added.append(column)

        # ADD COLUMN ... DEFAULT 'live' stamps every pre-existing row as live,
        # which would assert that seeded demo rows were real billed calls. The
        # backfill runs once, only in the transaction that added the column.
        if "source" in added:
            _backfill_source(conn)

        conn.executescript(INDEX_SCHEMA)


def _backfill_source(conn: sqlite3.Connection) -> None:
    """Classify rows that predate the `source` column.

    There is no stored flag to read, so this infers provenance from two
    fingerprints the writers left behind:

      * Real calls are timestamped with `datetime.now()`, which carries
        microseconds. Seeded rows were authored by hand at round minutes.
      * Real calls measure latency with a clock, so it is never exactly zero.
        Manual entries hardcode `latency_ms=0.0`.

    Anything that fails both tests is treated as demo data. That direction is
    deliberate: mislabelling a real call as demo understates spend and is
    visible, while the reverse invents money you never spent.
    """
    conn.execute(
        """
        UPDATE usage_events SET source = CASE
            WHEN latency_ms = 0 AND success = 1 THEN 'manual'
            WHEN instr(timestamp, '.') > 0     THEN 'live'
            ELSE 'demo'
        END
        """
    )


_INSERT_SQL = """
INSERT INTO usage_events
    (id, timestamp, model, provider, project_tag,
     input_tokens, output_tokens, cached_input_tokens, cache_write_tokens,
     cache_write_1h_tokens, cost_usd, latency_ms, prompt_preview, prompt_hash,
     service_tier, success, error, source)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""


def _insert_params(event: UsageEvent) -> tuple:
    """Column values for _INSERT_SQL, in order.

    Shared by log_usage and log_usage_many so the column list can only ever be
    wrong in one place — adding a column to one writer and not the other was
    the obvious failure mode of the previous duplicated version.
    """
    return (
        event.id,
        event.timestamp,
        event.model,
        event.provider,
        event.project_tag,
        event.input_tokens,
        event.output_tokens,
        event.cached_input_tokens,
        event.cache_write_tokens,
        event.cache_write_1h_tokens,
        event.cost_usd,
        event.latency_ms,
        event.prompt_preview,
        event.prompt_hash,
        event.service_tier,
        int(event.success),
        event.error,
        event.source,
    )


def log_usage(event: UsageEvent, db_path: str | None = None) -> None:
    """Persist a single UsageEvent to SQLite."""
    init_db(db_path)
    with _connect(db_path) as conn:
        conn.execute(_INSERT_SQL, _insert_params(event))


def log_usage_many(events: list[UsageEvent], db_path: str | None = None) -> None:
    """Persist multiple UsageEvents in one connection and one commit.

    log_usage() opens a fresh connection per call (via init_db() *and* its
    own _connect()) -- against local SQLite that's negligible, but against
    Turso each connection does a real network sync, so importing N events
    one-by-one costs 2N remote round-trips. This does exactly one connect
    and one commit for the whole batch, confirmed against the real Render
    deployment: a 1225-row import went from a projected multi-hour runtime
    (26s/row) to a single-digit-seconds batch.
    """
    init_db(db_path)
    with _connect(db_path) as conn:
        for event in events:
            conn.execute(_INSERT_SQL, _insert_params(event))


def get_events(
    start_date: str | None = None,
    end_date: str | None = None,
    project_tag: str | None = None,
    source: str | None = None,
    db_path: str | None = None,
) -> list[UsageEvent]:
    """Retrieve UsageEvents, optionally filtered by ISO date range, project, and source.

    start_date/end_date are inclusive ISO-8601 timestamp strings. If omitted,
    that bound is unlimited.

    `source` accepts one of VALID_SOURCES, a comma-separated set of them
    ("live,manual" — every row you were actually billed for), or None/"all"
    for everything.
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
    wanted = parse_sources(source)
    if wanted is not None:
        query += f" AND source IN ({','.join('?' * len(wanted))})"
        params.extend(wanted)
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
            cache_write_1h_tokens=_row_get(row, "cache_write_1h_tokens", 0),
            cost_usd=row["cost_usd"],
            latency_ms=row["latency_ms"],
            prompt_preview=row["prompt_preview"] or "",
            prompt_hash=_row_get(row, "prompt_hash", None),
            service_tier=_row_get(row, "service_tier", "standard"),
            success=bool(row["success"]),
            error=row["error"],
            source=_row_get(row, "source", "live"),
        )
        for row in rows
    ]


def parse_sources(source: str | None) -> tuple[str, ...] | None:
    """Normalise a source filter into a tuple of valid sources, or None for all.

    Returns None (meaning "no filter") for None and "all", so callers can pass
    a query-string value straight through without special-casing it.
    """
    if source is None:
        return None
    cleaned = source.strip().lower()
    if not cleaned or cleaned == "all":
        return None

    wanted = tuple(part.strip() for part in cleaned.split(",") if part.strip())
    invalid = [s for s in wanted if s not in VALID_SOURCES]
    if invalid:
        raise ValueError(f"invalid source(s) {invalid}; expected any of {list(VALID_SOURCES)} or 'all'")
    return wanted


def source_totals(period: str = "all_time", db_path: str | None = None) -> dict:
    """Cost and call count per provenance class, for the period.

    This is what makes the headline number honest: it says how much of the
    total was really billed vs. seeded for the demo.
    """
    events = get_events_for_period(period, db_path=db_path)
    cost: dict[str, float] = {}
    calls: dict[str, int] = {}
    for e in events:
        cost[e.source] = round(cost.get(e.source, 0.0) + e.cost_usd, 6)
        calls[e.source] = calls.get(e.source, 0) + 1

    billed = round(cost.get("live", 0.0) + cost.get("manual", 0.0), 6)
    subscription = round(cost.get("subscription", 0.0), 6)
    total = round(sum(cost.values()), 6)
    return {
        "period": period,
        "cost_by_source": cost,
        "calls_by_source": calls,
        "total_cost_usd": total,
        # Money metered and charged per token.
        "billed_cost_usd": billed,
        # Real tokens under a flat-fee plan: list-price value, not money spent.
        "subscription_cost_usd": subscription,
        "has_subscription_data": calls.get("subscription", 0) > 0,
        # Everything real, priced at list — billed + subscription. The honest
        # answer to "what did this work cost to produce".
        "list_price_cost_usd": round(billed + subscription, 6),
        "demo_cost_usd": round(cost.get("demo", 0.0), 6),
        "has_demo_data": calls.get("demo", 0) > 0,
        "demo_percent_of_total": round((cost.get("demo", 0.0) / total) * 100, 2) if total else 0.0,
    }


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


def get_events_for_period(
    period: str,
    project_tag: str | None = None,
    source: str | None = None,
    db_path: str | None = None,
) -> list[UsageEvent]:
    """Convenience wrapper: get_events() for a named period ('today'|'week'|'month'|'all_time')."""
    start = _period_start(period)
    return get_events(start_date=start, project_tag=project_tag, source=source, db_path=db_path)


def batch_load(json_path: str, db_path: str | None = None, source: str = "demo") -> int:
    """Load events from a sample_usage.json-style file for testing.

    Each entry may omit cost_usd (computed via pricing.py), timestamp
    (defaults to now), and id (auto-generated). Returns the count loaded.

    Rows land as `demo` unless told otherwise: anything arriving from a JSON
    file was authored, not billed, and the dashboard needs to be able to say so.
    """
    with open(json_path) as f:
        raw_events = json.load(f)

    count = 0
    for raw in raw_events:
        model = raw["model"]
        cached = raw.get("cached_input_tokens", 0)
        written = raw.get("cache_write_tokens", 0)
        written_1h = raw.get("cache_write_1h_tokens", 0)
        tier = raw.get("service_tier", "standard")

        cost_usd = raw.get("cost_usd")
        if cost_usd is None:
            cost_usd = calculate_cost(
                model, raw["input_tokens"], raw["output_tokens"], cached, written,
                cache_write_1h_tokens=written_1h, service_tier=tier,
            )

        event = UsageEvent(
            model=model,
            provider=raw.get("provider") or _safe_provider(model),
            project_tag=raw.get("project_tag", "default"),
            input_tokens=raw["input_tokens"],
            output_tokens=raw["output_tokens"],
            cached_input_tokens=cached,
            cache_write_tokens=written,
            cache_write_1h_tokens=written_1h,
            cost_usd=cost_usd,
            latency_ms=raw["latency_ms"],
            prompt_preview=UsageEvent.make_preview(raw.get("prompt_preview", "")),
            prompt_hash=raw.get("prompt_hash"),
            service_tier=tier,
            success=raw.get("success", True),
            error=raw.get("error"),
            source=raw.get("source", source),
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


def purge_source(source: str, db_path: str | None = None) -> int:
    """Delete every row with the given provenance. Returns rows removed.

    Exists so seeded data can be thrown away once there is real traffic to
    look at, without touching billed history.
    """
    if source not in VALID_SOURCES:
        raise ValueError(f"invalid source {source!r}; expected one of {list(VALID_SOURCES)}")
    init_db(db_path)
    with _connect(db_path) as conn:
        cursor = conn.execute("DELETE FROM usage_events WHERE source = ?", (source,))
        return cursor.rowcount


def _main() -> None:
    parser = argparse.ArgumentParser(description="Manually log usage events or batch-load sample data.")
    parser.add_argument("--log-manual", action="store_true", help="Log a single manual usage event")
    parser.add_argument("--model", type=str, help="Model name")
    parser.add_argument("--cost", type=float, help="Cost in USD")
    parser.add_argument("--tokens", type=int, default=0, help="Total tokens (logged as output_tokens)")
    parser.add_argument("--project", type=str, default="default", help="Project tag")
    parser.add_argument("--note", type=str, default="", help="Optional note, stored as prompt_preview")
    parser.add_argument("--batch-load", type=str, help="Path to a sample_usage.json-style file to load")
    parser.add_argument("--source", type=str, default=None, choices=list(VALID_SOURCES),
                        help="Provenance to stamp on loaded/logged rows (default: demo for --batch-load, manual for --log-manual)")
    parser.add_argument("--purge", type=str, default=None, choices=list(VALID_SOURCES),
                        help="Delete every row with this provenance, e.g. --purge demo")
    parser.add_argument("--provenance", action="store_true", help="Show the cost/call split by source")
    parser.add_argument("--period", type=str, default="all_time", help="Period for --provenance")
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
            source=args.source or "manual",
        )
        log_usage(event, db_path=args.db_path)
        print(f"logged {event.source} entry {event.id} | model={event.model} cost=${event.cost_usd:.6f} project={event.project_tag}")
    elif args.batch_load:
        n = batch_load(args.batch_load, db_path=args.db_path, source=args.source or "demo")
        print(f"loaded {n} events from {args.batch_load} as source={args.source or 'demo'}")
    elif args.purge:
        n = purge_source(args.purge, db_path=args.db_path)
        print(f"purged {n} row(s) with source={args.purge}")
    elif args.provenance:
        totals = source_totals(args.period, db_path=args.db_path)
        print(json.dumps(totals, indent=2))
    else:
        parser.print_help()


if __name__ == "__main__":
    _main()
