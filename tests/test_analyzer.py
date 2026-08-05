"""Analyzer: reports, anomaly-detection accuracy, budget boundaries,
burn-rate projection, provider breakdown, and model-switch what-ifs."""

import json
from datetime import datetime, timedelta, timezone

import pytest

from src import tracker
from src.analyzer import (
    check_budget_status,
    compute_report,
    flag_anomalies,
    project_burn_rate,
    provider_breakdown,
    what_if_switched,
)
from src.usage_schema import UsageEvent

from .conftest import SAMPLE_PATH


def _log(db, model="test-model", cost=0.001, latency=1000, project="test",
         provider="anthropic", cached=0, written=0, success=True, ts=None):
    tracker.log_usage(
        UsageEvent(
            model=model,
            provider=provider,
            project_tag=project,
            input_tokens=100,
            output_tokens=100,
            cached_input_tokens=cached,
            cache_write_tokens=written,
            cost_usd=cost,
            latency_ms=latency,
            success=success,
            **({"timestamp": ts} if ts else {}),
        ),
        db_path=db,
    )


# --- anomaly detection ---------------------------------------------------


def test_flags_injected_cost_spike(temp_db):
    for _ in range(20):
        _log(temp_db, cost=0.001, latency=1000)
    _log(temp_db, cost=0.005, latency=1000)  # 5x cost, normal latency

    anomalies = flag_anomalies(3.0)
    assert len(anomalies) == 1
    assert "cost" in anomalies[0].reason


def test_flags_injected_latency_spike(temp_db):
    for _ in range(20):
        _log(temp_db, cost=0.001, latency=1000)
    _log(temp_db, cost=0.001, latency=8000)  # 8x latency, normal cost

    anomalies = flag_anomalies(3.0)
    assert len(anomalies) == 1
    assert "latency" in anomalies[0].reason


def test_does_not_flag_normal_variance(temp_db):
    for i in range(25):
        wobble = 1 + (i % 5) * 0.15  # up to 1.6x — well under 3x
        _log(temp_db, cost=0.001 * wobble, latency=1000 * wobble)
    assert flag_anomalies(3.0) == []


def test_first_event_for_a_model_is_never_flagged(temp_db):
    _log(temp_db, model="brand-new", cost=99.0, latency=99999)
    assert flag_anomalies(3.0) == []


def test_threshold_multiplier_is_respected(temp_db):
    for _ in range(20):
        _log(temp_db, cost=0.001, latency=1000)
    _log(temp_db, cost=0.004, latency=1000)  # 4x

    assert len(flag_anomalies(3.0)) == 1  # 4x > 3x -> flagged
    assert flag_anomalies(5.0) == []      # 4x < 5x -> not flagged


def test_comparison_is_per_model_not_global(temp_db):
    """An expensive model must not be flagged merely for costing more than a
    cheap one — the baseline is per-model."""
    for _ in range(20):
        _log(temp_db, model="cheap-model", cost=0.0001, latency=500)
        _log(temp_db, model="pricey-model", cost=0.05, latency=3000)
    _log(temp_db, model="pricey-model", cost=0.05, latency=3000)

    assert flag_anomalies(3.0) == []


def test_failed_calls_do_not_poison_the_baseline(temp_db):
    """A 429 logs cost=0 / low latency. Including failures would drag the
    rolling average down and make the next normal call look anomalous."""
    for _ in range(20):
        _log(temp_db, cost=0.001, latency=1000)
    for _ in range(10):
        _log(temp_db, cost=0.0, latency=50, success=False)
    _log(temp_db, cost=0.001, latency=1000)  # a perfectly normal call

    assert flag_anomalies(3.0) == []


def test_catches_all_injected_anomalies_in_sample_data(sample_db):
    """The eval that matters: exactly the planted spikes in sample_usage.json,
    and no false positives on the rest."""
    anomalies = flag_anomalies(3.0)
    assert len(anomalies) == 3

    reasons = " ".join(a.reason for a in anomalies)
    assert "cost" in reasons and "latency" in reasons

    events = {e.id: e for e in tracker.get_events(db_path=sample_db)}
    flagged = [events[a.usage_event_id] for a in anomalies]
    assert any(e.latency_ms == 21500 for e in flagged)       # latency spike
    assert any(e.input_tokens == 48000 for e in flagged)     # cost spike
    assert any(e.input_tokens == 310000 for e in flagged)    # long-context spike


def test_long_context_call_is_flagged_high_severity(sample_db):
    """The 310k-token call trips both cost and latency, so it must be 'high'."""
    events = {e.id: e for e in tracker.get_events(db_path=sample_db)}
    hits = [a for a in flag_anomalies(3.0) if events[a.usage_event_id].input_tokens == 310000]
    assert hits and hits[0].severity == "high"


# --- reports -------------------------------------------------------------


def test_report_totals_and_breakdowns(sample_db):
    report = compute_report("all_time")
    raw = json.load(open(SAMPLE_PATH))

    assert report.total_calls == len(raw)
    assert report.failed_calls == sum(1 for e in raw if not e.get("success", True))
    assert len(report.breakdown_by_provider) == len({e["provider"] for e in raw})

    for breakdown in (report.breakdown_by_model, report.breakdown_by_project, report.breakdown_by_provider):
        assert sum(breakdown.values()) == pytest.approx(report.total_cost_usd, abs=1e-5)


def test_report_counts_cache_savings(temp_db):
    _log(temp_db, model="claude-opus-5", cached=10_000)
    assert compute_report("all_time").cache_savings_usd > 0


def test_report_on_empty_db_is_zeroed(temp_db):
    report = compute_report("all_time")
    assert report.total_calls == 0
    assert report.total_cost_usd == 0.0
    assert report.cache_savings_usd == 0.0


# --- budget boundaries ---------------------------------------------------


def test_budget_under(temp_db, monkeypatch):
    monkeypatch.setenv("WEEKLY_BUDGET_USD", "10.00")
    _log(temp_db, cost=1.0)
    status = check_budget_status("weekly")
    assert status["status"] == "under"
    assert status["percent_used"] == pytest.approx(10.0)


def test_budget_near_at_80_percent_boundary(temp_db, monkeypatch):
    monkeypatch.setenv("WEEKLY_BUDGET_USD", "10.00")
    _log(temp_db, cost=8.0)
    assert check_budget_status("weekly")["status"] == "near"


def test_budget_just_under_80_percent_is_still_under(temp_db, monkeypatch):
    monkeypatch.setenv("WEEKLY_BUDGET_USD", "10.00")
    _log(temp_db, cost=7.99)
    assert check_budget_status("weekly")["status"] == "under"


def test_budget_over_at_100_percent_boundary(temp_db, monkeypatch):
    monkeypatch.setenv("WEEKLY_BUDGET_USD", "10.00")
    _log(temp_db, cost=10.0)
    status = check_budget_status("weekly")
    assert status["status"] == "over"
    assert status["remaining_usd"] == pytest.approx(0.0)


def test_budget_over_returns_negative_remaining(temp_db, monkeypatch):
    monkeypatch.setenv("WEEKLY_BUDGET_USD", "5.00")
    _log(temp_db, cost=7.5)
    assert check_budget_status("weekly")["remaining_usd"] < 0


# --- burn rate -----------------------------------------------------------


def test_burn_rate_on_empty_db_is_safe(temp_db):
    burn = project_burn_rate("all_time")
    assert burn["daily_burn_usd"] == 0.0
    assert burn["on_track"] is True
    assert burn["confidence"] == "none"


def test_burn_rate_projects_from_observed_span(temp_db, monkeypatch):
    """Rate is spend divided by the span the data actually covers: $4 spread
    across a 4-day span is $1/day, projecting to $7 for a week."""
    monkeypatch.setenv("WEEKLY_BUDGET_USD", "10.00")
    now = datetime.now(timezone.utc)
    _log(temp_db, cost=2.0, ts=(now - timedelta(days=4)).isoformat())
    _log(temp_db, cost=2.0, ts=now.isoformat())

    burn = project_burn_rate("all_time")
    assert burn["observed_days"] == pytest.approx(4.0, rel=0.02)
    assert burn["daily_burn_usd"] == pytest.approx(1.0, rel=0.05)
    assert burn["projected_weekly_usd"] == pytest.approx(7.0, rel=0.05)


def test_burn_rate_flags_off_track_when_projection_exceeds_budget(temp_db, monkeypatch):
    monkeypatch.setenv("WEEKLY_BUDGET_USD", "1.00")
    now = datetime.now(timezone.utc)
    for day in range(4):
        _log(temp_db, cost=1.0, ts=(now - timedelta(days=3 - day)).isoformat())

    burn = project_burn_rate("all_time")
    assert burn["on_track"] is False
    assert burn["days_until_budget_exhausted"] is not None


def test_burn_rate_confidence_scales_with_evidence(temp_db):
    now = datetime.now(timezone.utc)
    _log(temp_db, cost=0.1, ts=(now - timedelta(minutes=5)).isoformat())
    _log(temp_db, cost=0.1, ts=now.isoformat())
    assert project_burn_rate("all_time")["confidence"] == "low"

    for i in range(30):
        _log(temp_db, cost=0.1, ts=(now - timedelta(days=5) + timedelta(hours=i * 4)).isoformat())
    assert project_burn_rate("all_time")["confidence"] == "high"


def test_burn_rate_excludes_failed_calls(temp_db):
    now = datetime.now(timezone.utc)
    for i in range(5):
        _log(temp_db, cost=0.0, success=False, ts=(now - timedelta(hours=i)).isoformat())
    assert project_burn_rate("all_time")["calls_observed"] == 0


# --- provider breakdown --------------------------------------------------


def test_provider_breakdown_groups_and_sorts_by_cost(sample_db):
    rows = provider_breakdown("all_time")
    assert {r["provider"] for r in rows} == {"anthropic", "openai", "google"}
    costs = [r["total_cost_usd"] for r in rows]
    assert costs == sorted(costs, reverse=True)


def test_provider_breakdown_computes_cache_hit_rate(temp_db):
    _log(temp_db, provider="anthropic", cached=50)  # input_tokens is 100
    row = next(r for r in provider_breakdown("all_time") if r["provider"] == "anthropic")
    assert row["cache_hit_rate"] == pytest.approx(0.5)


def test_provider_breakdown_counts_failures_separately(temp_db):
    _log(temp_db, provider="google", success=True)
    _log(temp_db, provider="google", success=False)
    row = next(r for r in provider_breakdown("all_time") if r["provider"] == "google")
    assert row["calls"] == 2 and row["failed_calls"] == 1


# --- what-if model switch ------------------------------------------------


def test_what_if_switch_to_cheaper_model_reports_savings(sample_db):
    result = what_if_switched("claude-sonnet-5", "claude-haiku-4-5", "all_time")
    assert result["calls_repriced"] > 0
    assert result["verdict"] == "cheaper"
    assert result["savings_usd"] > 0


def test_what_if_switch_to_pricier_model_reports_increase(sample_db):
    result = what_if_switched("claude-haiku-4-5", "claude-opus-5", "all_time")
    assert result["verdict"] == "more expensive"
    assert result["savings_usd"] < 0


def test_what_if_with_no_matching_traffic_is_handled(sample_db):
    result = what_if_switched("claude-fable-5", "claude-haiku-4-5", "all_time")
    assert result["calls_repriced"] == 0
    assert "note" in result


# --- digest --------------------------------------------------------------


def test_generate_digest_runs_without_crashing(sample_db, tmp_path, monkeypatch):
    """The digest must still produce output and save a report when the LLM
    call fails — a scheduled run that loses everything to a 429 is broken."""
    from src import digest

    monkeypatch.setattr(digest, "REPORTS_DIR", tmp_path / "reports")

    def boom(*args, **kwargs):
        raise RuntimeError("simulated API failure")

    monkeypatch.setattr(digest, "call_llm", boom)

    text = digest.generate_digest("all_time")
    assert text and "$" in text

    saved = list((tmp_path / "reports").glob("*_digest.json"))
    assert len(saved) == 1
    record = json.load(open(saved[0]))
    assert record["llm_written"] is False
    assert len(record["anomalies"]) == 3
