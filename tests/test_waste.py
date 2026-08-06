"""Waste detection — spend that bought nothing, and spend you could stop."""

import pytest

from src.tracker import log_usage
from src.usage_schema import UsageEvent
from src.waste import (
    find_cache_opportunities,
    find_duplicate_calls,
    find_model_switch_savings,
    find_overpowered_calls,
    find_retry_waste,
    find_waste,
)


def _log(db, model="claude-sonnet-5", provider="anthropic", project="p",
         inp=2000, out=300, cached=0, cost=0.02, latency=1000.0,
         preview="Summarize the quarterly engineering report and list risks",
         success=True):
    log_usage(
        UsageEvent(
            model=model, provider=provider, project_tag=project,
            input_tokens=inp, output_tokens=out, cached_input_tokens=cached,
            cost_usd=cost, latency_ms=latency, prompt_preview=preview, success=success,
        ),
        db_path=db,
    )


# --- retry waste ---------------------------------------------------------


def test_retry_waste_counts_failures_and_time(temp_db):
    for _ in range(3):
        _log(temp_db, success=False, cost=0.0, latency=500.0)
    for _ in range(7):
        _log(temp_db)

    w = find_retry_waste("all_time")
    assert w["failed_calls"] == 3
    assert w["total_calls"] == 10
    assert w["failure_rate"] == pytest.approx(0.3)
    assert w["wasted_seconds"] == pytest.approx(1.5)


def test_retry_waste_attributes_failures_to_a_model(temp_db):
    for _ in range(4):
        _log(temp_db, model="gemini-flash-latest", provider="google", success=False, cost=0.0)
    w = find_retry_waste("all_time")
    assert w["failures_by_model"] == {"gemini-flash-latest": 4}
    assert "gemini-flash-latest" in w["recommendation"]


def test_high_failure_rate_is_called_structural(temp_db):
    for _ in range(8):
        _log(temp_db, success=False, cost=0.0)
    for _ in range(2):
        _log(temp_db)
    assert "structural" in find_retry_waste("all_time")["recommendation"]


def test_no_failures_says_so(temp_db):
    _log(temp_db)
    w = find_retry_waste("all_time")
    assert w["failed_calls"] == 0
    assert "No failed calls" in w["recommendation"]


# --- duplicate calls -----------------------------------------------------


def test_identical_prompts_are_flagged(temp_db):
    for _ in range(4):
        _log(temp_db, preview="Score this job posting against my resume and rank the fit")

    rows = find_duplicate_calls("all_time")
    assert len(rows) == 1
    assert rows[0]["times_sent"] == 4
    # Only the repeats are avoidable — the first send was necessary.
    assert rows[0]["avoidable_cost_usd"] == pytest.approx(rows[0]["total_cost_usd"] * 0.75)


def test_distinct_prompts_are_not_flagged(temp_db):
    for i in range(4):
        _log(temp_db, preview=f"A genuinely different question number {i} about the codebase")
    assert find_duplicate_calls("all_time") == []


def test_short_previews_are_ignored(temp_db):
    """A truncated shared header shouldn't look like a duplicate prompt."""
    for _ in range(5):
        _log(temp_db, preview="Hi")
    assert find_duplicate_calls("all_time") == []


def test_duplicates_are_scoped_per_model(temp_db):
    same = "Explain the caching strategy used in this repository in detail"
    _log(temp_db, model="claude-sonnet-5", preview=same)
    _log(temp_db, model="claude-haiku-4-5", preview=same)
    assert find_duplicate_calls("all_time") == []  # one each, not a repeat


# --- cache opportunities -------------------------------------------------


def test_repeated_large_uncached_prompts_are_flagged(temp_db):
    for _ in range(5):
        _log(temp_db, inp=8000, cached=0)

    rows = find_cache_opportunities("all_time")
    assert len(rows) == 1
    assert rows[0]["estimated_saving_usd"] > 0
    assert rows[0]["cache_hit_rate"] == 0.0


def test_already_cached_traffic_is_not_flagged(temp_db):
    for _ in range(5):
        _log(temp_db, inp=8000, cached=7000)   # 87% hit rate
    assert find_cache_opportunities("all_time") == []


def test_small_prompts_are_not_worth_caching(temp_db):
    for _ in range(5):
        _log(temp_db, inp=100)
    assert find_cache_opportunities("all_time") == []


def test_single_large_call_is_not_a_cache_opportunity(temp_db):
    """Caching only pays when the prefix is reused."""
    _log(temp_db, inp=50_000)
    assert find_cache_opportunities("all_time") == []


# --- over-powered models -------------------------------------------------


def test_frontier_model_on_trivial_work_is_flagged(temp_db):
    for _ in range(3):
        _log(temp_db, model="claude-opus-5", inp=200, out=50, cost=0.002)

    rows = find_overpowered_calls("all_time")
    assert len(rows) == 1
    assert rows[0]["suggested_model"] == "claude-haiku-4-5"
    assert rows[0]["estimated_saving_usd"] > 0
    assert "heuristic" in rows[0]["confidence"]


def test_frontier_model_on_substantial_work_is_not_flagged(temp_db):
    _log(temp_db, model="claude-opus-5", inp=5000, out=2000, cost=0.08)
    assert find_overpowered_calls("all_time") == []


def test_cheap_model_is_never_flagged_as_overpowered(temp_db):
    for _ in range(3):
        _log(temp_db, model="claude-haiku-4-5", inp=100, out=20, cost=0.0002)
    assert find_overpowered_calls("all_time") == []


# --- aggregate -----------------------------------------------------------


def test_find_waste_totals_recoverable_spend(temp_db):
    for _ in range(5):
        _log(temp_db, inp=8000, cost=0.05,
             preview="Summarize the full engineering handbook chapter by chapter")

    w = find_waste("all_time")
    assert w["recoverable_usd"] > 0
    assert 0 < w["recoverable_percent"] <= 100
    assert w["top_action"]


def test_find_waste_on_clean_data_reports_nothing_to_do(temp_db):
    for i in range(3):
        _log(temp_db, inp=200, out=50, model="claude-haiku-4-5", cost=0.0002,
             preview=f"A distinct short question number {i} that repeats nothing")

    w = find_waste("all_time")
    assert w["recoverable_usd"] == 0.0
    assert "No obvious waste" in w["top_action"]


def test_find_waste_on_empty_db_is_safe(temp_db):
    w = find_waste("all_time")
    assert w["total_spend_usd"] == 0.0
    assert w["recoverable_usd"] == 0.0
    assert w["recoverable_percent"] == 0.0


# --- model switch ----------------------------------------------------------


def test_model_switch_finds_real_savings(temp_db):
    for _ in range(2):
        _log(temp_db, model="claude-opus-5", inp=200_000, out=40_000, cost=2.0,
             preview="Large real build-session turn re-sending accumulated context")

    rows = find_model_switch_savings("all_time")

    assert len(rows) == 1
    row = rows[0]
    assert row["from_model"] == "claude-opus-5"
    assert row["to_model"] == "claude-sonnet-5"
    assert row["calls_repriced"] == 2
    assert row["estimated_saving_usd"] > 1.0
    assert "claude-opus-5" in row["action"] and "claude-sonnet-5" in row["action"]


def test_model_switch_below_threshold_is_not_reported(temp_db):
    _log(temp_db, model="claude-opus-5", inp=1000, out=200, cost=0.01,
         preview="One small opus call, nowhere near the $1 reporting floor")

    assert find_model_switch_savings("all_time") == []


def test_model_switch_skips_models_without_a_configured_sibling(temp_db):
    _log(temp_db, model="claude-haiku-4-5", inp=200_000, out=40_000, cost=2.0,
         preview="Haiku is already the cheapest tier, nothing to downgrade to")

    assert find_model_switch_savings("all_time") == []


def test_model_switch_only_counts_successful_calls(temp_db):
    _log(temp_db, model="claude-opus-5", inp=200_000, out=40_000, cost=2.0,
         success=False,
         preview="A failed opus call should not be repriced as a switch opportunity")

    assert find_model_switch_savings("all_time") == []


def test_find_waste_surfaces_model_switch_as_top_action_when_largest(temp_db):
    for i in range(3):
        _log(temp_db, model="claude-opus-5", inp=200_000, out=40_000, cost=2.0,
             preview=f"Distinct real build-session turn number {i}, not a repeat")

    w = find_waste("all_time")

    assert "model_switches" in w
    assert w["model_switches"], "expected at least one model-switch finding"
    assert "claude-sonnet-5" in w["top_action"]


def test_find_waste_model_switches_key_present_even_when_empty(temp_db):
    w = find_waste("all_time")
    assert w["model_switches"] == []
    assert "model_switches" in w["recoverable_by_category"]
