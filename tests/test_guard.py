"""Spend guardrails — the only part of this project that stops a call.

These tests matter disproportionately: a guard that fails open silently is
worse than no guard, because you believe you're protected.
"""

from datetime import datetime, timezone

import pytest

from src.guard import (
    MODE_BLOCK,
    MODE_OFF,
    MODE_WARN,
    BudgetExceededError,
    check_guards,
    enforce,
    guard_status,
)
from src.tracker import log_usage
from src.usage_schema import UsageEvent


def _spend(db, cost, project="default", ts=None):
    log_usage(
        UsageEvent(
            model="claude-opus-5", provider="anthropic", project_tag=project,
            input_tokens=10, output_tokens=10, cost_usd=cost, latency_ms=100.0,
            **({"timestamp": ts} if ts else {}),
        ),
        db_path=db,
    )


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for var in ("WATCHDOG_GUARD_MODE", "WATCHDOG_PROJECT_CAPS",
                "WATCHDOG_MAX_CALLS_PER_MIN", "WATCHDOG_MAX_COST_PER_CALL"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("WEEKLY_BUDGET_USD", "5.00")


# --- modes ---------------------------------------------------------------


def test_default_mode_is_warn(temp_db):
    assert check_guards().mode == MODE_WARN


def test_off_mode_allows_everything(temp_db, monkeypatch):
    monkeypatch.setenv("WATCHDOG_GUARD_MODE", MODE_OFF)
    _spend(temp_db, 999.0)
    verdict = check_guards()
    assert verdict.allowed is True
    assert verdict.triggered == []


def test_unknown_mode_falls_back_to_warn(temp_db, monkeypatch):
    """A typo in config must not silently disable protection."""
    monkeypatch.setenv("WATCHDOG_GUARD_MODE", "blocck")
    assert check_guards().mode == MODE_WARN


# --- weekly budget -------------------------------------------------------


def test_under_budget_is_allowed(temp_db):
    _spend(temp_db, 1.0)
    verdict = check_guards()
    assert verdict.allowed is True
    assert verdict.triggered == []


def test_over_budget_trips_in_warn_mode_but_still_allows(temp_db, monkeypatch):
    monkeypatch.setenv("WATCHDOG_GUARD_MODE", MODE_WARN)
    _spend(temp_db, 6.0)
    verdict = check_guards()
    assert "weekly_budget" in verdict.triggered
    assert verdict.allowed is True     # warn mode does not stop the call


def test_over_budget_blocks_in_block_mode(temp_db, monkeypatch):
    monkeypatch.setenv("WATCHDOG_GUARD_MODE", MODE_BLOCK)
    _spend(temp_db, 6.0)
    assert check_guards().allowed is False


def test_enforce_raises_only_in_block_mode(temp_db, monkeypatch):
    _spend(temp_db, 6.0)

    monkeypatch.setenv("WATCHDOG_GUARD_MODE", MODE_WARN)
    enforce()  # must not raise

    monkeypatch.setenv("WATCHDOG_GUARD_MODE", MODE_BLOCK)
    with pytest.raises(BudgetExceededError):
        enforce()


def test_blocked_error_carries_the_verdict(temp_db, monkeypatch):
    monkeypatch.setenv("WATCHDOG_GUARD_MODE", MODE_BLOCK)
    _spend(temp_db, 6.0)
    with pytest.raises(BudgetExceededError) as exc:
        enforce()
    assert "weekly_budget" in exc.value.verdict.triggered
    assert "budget" in str(exc.value)


# --- per-project caps ----------------------------------------------------


def test_project_cap_trips_independently_of_the_global_budget(temp_db, monkeypatch):
    monkeypatch.setenv("WATCHDOG_PROJECT_CAPS", "job-search-agent:0.50")
    _spend(temp_db, 0.75, project="job-search-agent")   # global budget is $5, fine

    assert "project_cap" in check_guards(project_tag="job-search-agent").triggered
    assert check_guards(project_tag="other-project").triggered == []


def test_project_cap_only_counts_that_project(temp_db, monkeypatch):
    monkeypatch.setenv("WATCHDOG_PROJECT_CAPS", "a:1.00")
    _spend(temp_db, 5.0, project="b")
    assert "project_cap" not in check_guards(project_tag="a").triggered


def test_malformed_project_caps_are_ignored_not_fatal(temp_db, monkeypatch):
    """A typo in config must not break the call it was meant to protect."""
    monkeypatch.setenv("WATCHDOG_PROJECT_CAPS", "broken,a:notanumber,good:1.00")
    _spend(temp_db, 2.0, project="good")
    assert "project_cap" in check_guards(project_tag="good").triggered


# --- runaway-loop circuit breaker ---------------------------------------


def test_circuit_breaker_trips_on_call_burst(temp_db, monkeypatch):
    """The scenario: an agent loop with no exit condition. Rate is the early
    signal — cost lags far behind volume on cheap models."""
    monkeypatch.setenv("WATCHDOG_MAX_CALLS_PER_MIN", "10")
    now = datetime.now(timezone.utc).isoformat()
    for _ in range(12):
        _spend(temp_db, 0.0001, ts=now)

    assert "rate_limit" in check_guards().triggered


def test_circuit_breaker_ignores_older_calls(temp_db, monkeypatch):
    monkeypatch.setenv("WATCHDOG_MAX_CALLS_PER_MIN", "5")
    old = "2026-08-01T10:00:00+00:00"
    for _ in range(20):
        _spend(temp_db, 0.0001, ts=old)

    assert "rate_limit" not in check_guards().triggered


def test_circuit_breaker_can_be_disabled(temp_db, monkeypatch):
    monkeypatch.setenv("WATCHDOG_MAX_CALLS_PER_MIN", "0")
    now = datetime.now(timezone.utc).isoformat()
    for _ in range(50):
        _spend(temp_db, 0.0001, ts=now)

    assert "rate_limit" not in check_guards().triggered


# --- per-call cost ceiling ----------------------------------------------


def test_implausibly_expensive_call_is_flagged(temp_db, monkeypatch):
    monkeypatch.setenv("WATCHDOG_MAX_COST_PER_CALL", "0.10")
    verdict = check_guards(
        model="claude-opus-5", estimated_input_tokens=500_000, estimated_output_tokens=10_000
    )
    assert "per_call_cost" in verdict.triggered


def test_normal_sized_call_is_not_flagged(temp_db, monkeypatch):
    monkeypatch.setenv("WATCHDOG_MAX_COST_PER_CALL", "1.00")
    verdict = check_guards(
        model="claude-opus-5", estimated_input_tokens=2000, estimated_output_tokens=500
    )
    assert "per_call_cost" not in verdict.triggered


# --- status --------------------------------------------------------------


def test_guard_status_reports_headroom(temp_db):
    _spend(temp_db, 1.25)
    status = guard_status()
    assert status["weekly_spend_usd"] == pytest.approx(1.25)
    assert status["weekly_headroom_usd"] == pytest.approx(3.75)
    assert status["weekly_percent_used"] == pytest.approx(25.0)
    assert status["mode"] in (MODE_OFF, MODE_WARN, MODE_BLOCK)
    assert status["mode_meaning"]


def test_guard_status_lists_configured_caps(temp_db, monkeypatch):
    monkeypatch.setenv("WATCHDOG_PROJECT_CAPS", "a:1.00,b:2.50")
    assert guard_status()["project_caps"] == {"a": 1.00, "b": 2.50}
