"""
Provenance: every row records whether it represents real money.

These tests exist because the dashboard once reported $0.4768 of "spend" when
$0.4702 of it was seeded sample data that was never billed, a cost tracker
confidently reporting fiction. The rules encoded here:

  * a row's source is always one of live / demo / manual
  * the seeder cannot produce rows that claim to be live
  * anything that enforces or projects real spend counts billed rows only
  * migrating an old DB classifies its rows instead of blanket-stamping them
"""

import sqlite3

import pytest

from src import tracker
from src.analyzer import check_budget_status, compute_report, project_burn_rate, provider_breakdown
from src.guard import check_guards
from src.tracker import (
    BILLED_SOURCES,
    RUNTIME_SOURCES,
    VALID_SOURCES,
    batch_load,
    get_events_for_period,
    log_usage,
    parse_sources,
    purge_source,
    source_totals,
)
from src.usage_schema import UsageEvent
from src.waste import find_waste


def _event(cost=0.01, source="live", **kw):
    defaults = dict(
        model="gemini-flash-latest",
        provider="google",
        project_tag="p",
        input_tokens=100,
        output_tokens=50,
        cost_usd=cost,
        latency_ms=120.0,
        source=source,
    )
    defaults.update(kw)
    return UsageEvent(**defaults)


# --- defaults and round-tripping ------------------------------------------

def test_usage_event_defaults_to_live():
    assert _event().source == "live"


def test_source_survives_a_write_read_cycle(temp_db):
    for src in VALID_SOURCES:
        log_usage(_event(source=src), db_path=temp_db)

    got = {e.source for e in get_events_for_period("all_time", db_path=temp_db)}
    assert got == set(VALID_SOURCES)


def test_invalid_source_is_rejected_by_the_schema():
    with pytest.raises(Exception):
        _event(source="fabricated")


# --- the seeder must never claim to be real -------------------------------

def test_batch_load_marks_rows_as_demo(temp_db, tmp_path):
    path = tmp_path / "s.json"
    path.write_text('[{"model":"gemini-flash-latest","input_tokens":10,'
                    '"output_tokens":5,"latency_ms":100}]')

    batch_load(str(path), db_path=temp_db)

    events = get_events_for_period("all_time", db_path=temp_db)
    assert [e.source for e in events] == ["demo"]


def test_sample_usage_json_loads_entirely_as_demo(sample_db):
    events = get_events_for_period("all_time", db_path=sample_db)
    assert events, "fixture should not be empty"
    assert {e.source for e in events} == {"demo"}


def test_batch_load_source_can_be_overridden(temp_db, tmp_path):
    path = tmp_path / "s.json"
    path.write_text('[{"model":"gemini-flash-latest","input_tokens":10,'
                    '"output_tokens":5,"latency_ms":100}]')

    batch_load(str(path), db_path=temp_db, source="manual")

    assert get_events_for_period("all_time", db_path=temp_db)[0].source == "manual"


# --- filtering -------------------------------------------------------------

@pytest.mark.parametrize("value,expected", [
    (None, None),
    ("all", None),
    ("", None),
    ("live", ("live",)),
    ("LIVE", ("live",)),
    (" live, manual ", ("live", "manual")),
    ("live,manual,demo", ("live", "manual", "demo")),
])
def test_parse_sources(value, expected):
    assert parse_sources(value) == expected


@pytest.mark.parametrize("bad", ["nope", "live,nope", "seeded"])
def test_parse_sources_rejects_unknown(bad):
    with pytest.raises(ValueError):
        parse_sources(bad)


def test_get_events_filters_by_source(temp_db):
    log_usage(_event(cost=1.0, source="live"), db_path=temp_db)
    log_usage(_event(cost=2.0, source="demo"), db_path=temp_db)
    log_usage(_event(cost=4.0, source="manual"), db_path=temp_db)

    def total(src):
        return sum(e.cost_usd for e in get_events_for_period("all_time", source=src, db_path=temp_db))

    assert total("live") == 1.0
    assert total("demo") == 2.0
    assert total(BILLED_SOURCES) == 5.0
    assert total(None) == 7.0
    assert total("all") == 7.0


# --- the headline number tells the truth -----------------------------------

def test_source_totals_separates_billed_from_seeded(temp_db):
    log_usage(_event(cost=0.10, source="live"), db_path=temp_db)
    log_usage(_event(cost=0.05, source="manual"), db_path=temp_db)
    log_usage(_event(cost=9.85, source="demo"), db_path=temp_db)

    t = source_totals("all_time", db_path=temp_db)

    assert t["total_cost_usd"] == 10.0
    assert t["billed_cost_usd"] == 0.15
    assert t["demo_cost_usd"] == 9.85
    assert t["has_demo_data"] is True
    assert t["demo_percent_of_total"] == 98.5
    assert t["calls_by_source"] == {"live": 1, "manual": 1, "demo": 1}


def test_source_totals_reports_no_demo_data_when_clean(temp_db):
    log_usage(_event(cost=0.10, source="live"), db_path=temp_db)

    t = source_totals("all_time", db_path=temp_db)

    assert t["has_demo_data"] is False
    assert t["demo_percent_of_total"] == 0.0
    assert t["billed_cost_usd"] == t["total_cost_usd"]


def test_report_breaks_down_by_source(temp_db):
    log_usage(_event(cost=0.25, source="live"), db_path=temp_db)
    log_usage(_event(cost=0.75, source="demo"), db_path=temp_db)

    report = compute_report("all_time")

    assert report.breakdown_by_source == {"live": 0.25, "demo": 0.75}
    assert report.calls_by_source == {"live": 1, "demo": 1}
    assert report.source_filter is None


def test_report_honours_a_source_filter(temp_db):
    log_usage(_event(cost=0.25, source="live"), db_path=temp_db)
    log_usage(_event(cost=0.75, source="demo"), db_path=temp_db)

    report = compute_report("all_time", source="live")

    assert report.total_cost_usd == 0.25
    assert report.total_calls == 1
    assert report.source_filter == "live"


def test_provider_breakdown_exposes_live_call_count(temp_db):
    log_usage(_event(source="demo", model="claude-sonnet-5", provider="anthropic"), db_path=temp_db)
    log_usage(_event(source="demo", model="claude-sonnet-5", provider="anthropic"), db_path=temp_db)
    log_usage(_event(source="live", provider="google"), db_path=temp_db)

    rows = {r["provider"]: r for r in provider_breakdown("all_time")}

    # The point of this field: anthropic's numbers exist, but none are real.
    assert rows["anthropic"]["live_calls"] == 0
    assert rows["anthropic"]["calls_by_source"] == {"demo": 2}
    assert rows["google"]["live_calls"] == 1


# --- demo data must not spend a real budget --------------------------------

def test_budget_ignores_demo_rows_by_default(temp_db, monkeypatch):
    monkeypatch.setenv("WEEKLY_BUDGET_USD", "1.00")
    log_usage(_event(cost=5.00, source="demo"), db_path=temp_db)
    log_usage(_event(cost=0.10, source="live"), db_path=temp_db)

    status = check_budget_status("weekly")

    # $5 of seeded data would read as 500% of a $1 budget.
    assert status["status"] == "under"
    assert status["spend_usd"] == 0.10
    assert status["source_filter"] == RUNTIME_SOURCES


def test_budget_can_be_asked_to_include_demo_rows(temp_db, monkeypatch):
    monkeypatch.setenv("WEEKLY_BUDGET_USD", "1.00")
    log_usage(_event(cost=5.00, source="demo"), db_path=temp_db)

    assert check_budget_status("weekly", source="all")["status"] == "over"


# --- backfilled build cost is real money, but not a weekly run rate ---------

def test_budget_ignores_backfilled_build_cost_by_default(temp_db, monkeypatch):
    """Importing a Claude Code transcript must not read as this week's spend.

    `manual` rows are real money, unlike demo rows — but they are reconstructed
    from an existing record after the fact, so a bulk import of several
    projects' build cost would otherwise report "over budget" for a week in
    which nothing new was actually run.
    """
    monkeypatch.setenv("WEEKLY_BUDGET_USD", "1.00")
    log_usage(_event(cost=50.00, source="manual"), db_path=temp_db)
    log_usage(_event(cost=0.10, source="live"), db_path=temp_db)

    status = check_budget_status("weekly")

    assert status["status"] == "under"
    assert status["spend_usd"] == 0.10

    # ...but it is still real money, and still counted as such on request.
    assert check_budget_status("weekly", source=BILLED_SOURCES)["status"] == "over"


def test_backfilled_build_cost_does_not_block_live_calls(temp_db, monkeypatch):
    """A guardrail cannot prevent spend that already happened last week."""
    monkeypatch.setenv("WATCHDOG_GUARD_MODE", "block")
    monkeypatch.setenv("WEEKLY_BUDGET_USD", "1.00")
    log_usage(_event(cost=50.00, source="manual"), db_path=temp_db)

    verdict = check_guards()

    assert verdict.allowed
    assert "weekly_budget" not in verdict.triggered


def test_guard_does_not_trip_on_seeded_spend(temp_db, monkeypatch):
    monkeypatch.setenv("WATCHDOG_GUARD_MODE", "block")
    monkeypatch.setenv("WEEKLY_BUDGET_USD", "1.00")
    log_usage(_event(cost=99.0, source="demo"), db_path=temp_db)

    verdict = check_guards(project_tag="p")

    # Loading sample data must never be enough to refuse a real call.
    assert verdict.allowed is True
    assert "weekly_budget" not in verdict.triggered


def test_guard_still_trips_on_real_spend(temp_db, monkeypatch):
    monkeypatch.setenv("WATCHDOG_GUARD_MODE", "block")
    monkeypatch.setenv("WEEKLY_BUDGET_USD", "1.00")
    log_usage(_event(cost=99.0, source="live"), db_path=temp_db)

    verdict = check_guards(project_tag="p")

    assert verdict.allowed is False
    assert "weekly_budget" in verdict.triggered


def test_burn_rate_projects_from_billed_rows_only(temp_db):
    log_usage(_event(cost=50.0, source="demo"), db_path=temp_db)

    burn = project_burn_rate("week")

    assert burn["spend_so_far_usd"] == 0.0
    assert burn["confidence"] == "none"


def test_waste_totals_respect_the_source_filter(temp_db):
    log_usage(_event(cost=1.0, source="demo"), db_path=temp_db)
    log_usage(_event(cost=0.5, source="live"), db_path=temp_db)

    assert find_waste("all_time", source="live")["total_spend_usd"] == 0.5
    assert find_waste("all_time")["total_spend_usd"] == 1.5


# --- purging ---------------------------------------------------------------

def test_purge_demo_leaves_billed_history_intact(temp_db):
    log_usage(_event(cost=1.0, source="demo"), db_path=temp_db)
    log_usage(_event(cost=2.0, source="demo"), db_path=temp_db)
    log_usage(_event(cost=0.5, source="live"), db_path=temp_db)
    log_usage(_event(cost=0.25, source="manual"), db_path=temp_db)

    removed = purge_source("demo", db_path=temp_db)

    assert removed == 2
    remaining = get_events_for_period("all_time", db_path=temp_db)
    assert sorted(e.source for e in remaining) == ["live", "manual"]


def test_purge_rejects_an_unknown_source(temp_db):
    with pytest.raises(ValueError):
        purge_source("everything", db_path=temp_db)


# --- migrating a pre-provenance database -----------------------------------

def _legacy_db(path):
    """Build a DB shaped like the pre-provenance schema, with no `source`."""
    conn = sqlite3.connect(path)
    conn.execute("""
        CREATE TABLE usage_events (
            id TEXT PRIMARY KEY, timestamp TEXT NOT NULL, model TEXT NOT NULL,
            provider TEXT NOT NULL DEFAULT 'unknown', project_tag TEXT NOT NULL,
            input_tokens INTEGER NOT NULL, output_tokens INTEGER NOT NULL,
            cached_input_tokens INTEGER NOT NULL DEFAULT 0,
            cache_write_tokens INTEGER NOT NULL DEFAULT 0,
            cost_usd REAL NOT NULL, latency_ms REAL NOT NULL,
            prompt_preview TEXT, success INTEGER NOT NULL, error TEXT
        )""")
    rows = [
        # microseconds + real latency -> a measured call
        ("a", "2026-08-05T18:29:52.049793+00:00", 1200.5, 1),
        # round minute + authored latency -> seeded
        ("b", "2026-08-03T01:21:00+00:00", 900.0, 1),
        # zero latency -> hand-entered, no clock was involved
        ("c", "2026-08-05T18:30:05.076484+00:00", 0.0, 1),
        # a failed row with a round timestamp is still seeded, not manual
        ("d", "2026-08-02T09:00:00+00:00", 0.0, 0),
    ]
    for rid, ts, latency, ok in rows:
        conn.execute(
            "INSERT INTO usage_events VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (rid, ts, "gemini-flash-latest", "google", "p", 10, 5, 0, 0, 0.01, latency, "x", ok, None),
        )
    conn.commit()
    conn.close()


def test_migration_classifies_legacy_rows_instead_of_stamping_them(tmp_path, monkeypatch):
    db = str(tmp_path / "legacy.db")
    _legacy_db(db)
    monkeypatch.setenv("WATCHDOG_DB_PATH", db)

    tracker.init_db(db)

    got = {e.id: e.source for e in get_events_for_period("all_time", db_path=db)}
    assert got == {"a": "live", "b": "demo", "c": "manual", "d": "demo"}


def test_migration_preserves_every_legacy_row(tmp_path, monkeypatch):
    db = str(tmp_path / "legacy.db")
    _legacy_db(db)
    monkeypatch.setenv("WATCHDOG_DB_PATH", db)

    tracker.init_db(db)

    assert len(get_events_for_period("all_time", db_path=db)) == 4


def test_migration_is_idempotent_and_does_not_reclassify(tmp_path, monkeypatch):
    db = str(tmp_path / "legacy.db")
    _legacy_db(db)
    monkeypatch.setenv("WATCHDOG_DB_PATH", db)
    tracker.init_db(db)

    # A live row written after migration has zero latency only by coincidence;
    # re-running init_db must not re-run the backfill and relabel it.
    log_usage(_event(cost=0.02, source="live", latency_ms=0.0), db_path=db)
    tracker.init_db(db)
    tracker.init_db(db)

    sources = sorted(e.source for e in get_events_for_period("all_time", db_path=db))
    assert sources == ["demo", "demo", "live", "live", "manual"]
