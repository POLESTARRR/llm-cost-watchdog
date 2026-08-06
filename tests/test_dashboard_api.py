"""API endpoint tests for the dashboard service via FastAPI's TestClient."""

import pytest
from fastapi.testclient import TestClient

from dashboard.app import app

PERIODS = ["today", "week", "month", "all_time"]


@pytest.fixture
def client(sample_db):
    """TestClient bound to a temp DB preloaded with sample_usage.json."""
    return TestClient(app)


def test_health(client):
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json() == {"status": "ok"}


# --- /report -------------------------------------------------------------


def test_report_default_period(client):
    body = client.get("/report").json()
    assert set(body) >= {
        "period", "total_cost_usd", "total_calls", "failed_calls",
        "cache_savings_usd", "breakdown_by_model", "breakdown_by_project",
        "breakdown_by_provider",
    }
    assert body["period"] == "week"


@pytest.mark.parametrize("period", PERIODS)
def test_report_accepts_all_valid_periods(client, period):
    assert client.get(f"/report?period={period}").json()["period"] == period


def test_report_rejects_invalid_period(client):
    assert client.get("/report?period=fortnight").status_code == 422


def test_report_all_time_sees_all_sample_events(client):
    assert client.get("/report?period=all_time").json()["total_calls"] == 44


def test_report_breaks_down_by_all_three_providers(client):
    body = client.get("/report?period=all_time").json()
    assert set(body["breakdown_by_provider"]) == {"anthropic", "openai", "google"}


# --- /anomalies ----------------------------------------------------------


def test_anomalies_shape(client):
    body = client.get("/anomalies").json()
    assert len(body) == 3  # the three planted spikes
    for a in body:
        assert set(a) >= {"id", "usage_event_id", "reason", "severity", "detected_at"}
        assert a["severity"] in {"low", "medium", "high"}


def test_anomalies_threshold_is_tunable(client):
    assert client.get("/anomalies?threshold_multiplier=500").json() == []


# --- /budget and /burn-rate ---------------------------------------------


def test_budget_shape(client):
    body = client.get("/budget").json()
    assert set(body) >= {"status", "percent_used", "remaining_usd", "spend_usd", "limit_usd"}
    assert body["status"] in {"under", "near", "over"}


def test_burn_rate_shape(client):
    body = client.get("/burn-rate?period=all_time").json()
    assert set(body) >= {
        "daily_burn_usd", "projected_weekly_usd", "budget_limit_usd",
        "on_track", "confidence", "calls_observed",
    }
    assert body["confidence"] in {"none", "low", "medium", "high"}


def test_burn_rate_rejects_invalid_period(client):
    assert client.get("/burn-rate?period=nope").status_code == 422


# --- /providers and /compare --------------------------------------------


def test_providers_reports_config_and_breakdown(client):
    body = client.get("/providers?period=all_time").json()
    assert set(body["configured"]) == {"google", "anthropic", "openai"}
    assert {r["provider"] for r in body["breakdown"]} == {"anthropic", "openai", "google"}
    for row in body["breakdown"]:
        assert set(row) >= {
            "provider", "total_cost_usd", "calls", "failed_calls",
            "cache_hit_rate", "avg_latency_ms", "models_used",
        }


def test_compare_returns_models_cheapest_first(client):
    rows = client.get("/compare?input_tokens=5000&output_tokens=1000").json()
    costs = [r["cost_usd"] for r in rows]
    assert costs == sorted(costs)
    assert rows[0]["vs_cheapest"] == 1.0


# --- /calls --------------------------------------------------------------


def test_calls_returns_recent_first_and_respects_limit(client):
    body = client.get("/calls?period=all_time&limit=5").json()
    assert len(body) == 5

    timestamps = [c["timestamp"] for c in body]
    assert timestamps == sorted(timestamps, reverse=True)

    assert set(body[0]) >= {
        "id", "timestamp", "model", "provider", "project_tag",
        "input_tokens", "output_tokens", "cached_input_tokens",
        "cost_usd", "latency_ms", "success",
    }


def test_calls_never_exposes_full_prompts(client):
    """prompt_preview is capped at 80 chars — a privacy guarantee, so assert it."""
    for call in client.get("/calls?period=all_time&limit=44").json():
        assert len(call["prompt_preview"]) <= 80


# --- static frontend -----------------------------------------------------


def test_index_html_is_served(client):
    res = client.get("/")
    assert res.status_code == 200
    assert "text/html" in res.headers["content-type"]
    assert "LLM Cost Watchdog" in res.text


def test_index_defines_palette_vars_on_root_not_viz_root(client):
    """Regression: scoping the custom properties to .viz-root left `body`'s
    var() unresolved and rendered white-on-white text."""
    html = client.get("/").text
    assert ":root {" in html
    assert ".viz-root {\n    min-height: 100vh;\n    background: var(--surface-0);" in html


# --- provenance ----------------------------------------------------------


def test_provenance_endpoint_reports_the_split(client):
    body = client.get("/provenance").json()
    assert set(body) >= {
        "cost_by_source", "calls_by_source", "total_cost_usd",
        "billed_cost_usd", "demo_cost_usd", "has_demo_data",
        "demo_percent_of_total",
    }
    # The fixture is sample_usage.json, so all of it is seeded.
    assert body["has_demo_data"] is True
    assert body["billed_cost_usd"] == 0.0
    assert body["demo_percent_of_total"] == 100.0


@pytest.mark.parametrize("period", PERIODS)
def test_provenance_accepts_all_valid_periods(client, period):
    assert client.get(f"/provenance?period={period}").json()["period"] == period


def test_report_source_filter_excludes_demo(client):
    everything = client.get("/report?period=all_time").json()
    live_only = client.get("/report?period=all_time&source=live").json()

    assert everything["total_calls"] > 0
    assert live_only["total_calls"] == 0
    assert live_only["total_cost_usd"] == 0.0
    assert live_only["source_filter"] == "live"


def test_report_source_all_is_the_same_as_omitting_it(client):
    a = client.get("/report?period=all_time").json()
    b = client.get("/report?period=all_time&source=all").json()
    assert a["total_cost_usd"] == b["total_cost_usd"]


@pytest.mark.parametrize("source", ["live", "demo", "manual", "live,manual", "all"])
def test_endpoints_accept_valid_sources(client, source):
    for path in ["/report", "/calls", "/waste", "/providers", "/budget", "/burn-rate", "/anomalies"]:
        sep = "&" if "?" in path else "?"
        res = client.get(f"{path}{sep}source={source}")
        assert res.status_code == 200, f"{path} rejected source={source}"


@pytest.mark.parametrize("bad", ["seeded", "live,seeded", "nope"])
def test_endpoints_reject_invalid_sources(client, bad):
    res = client.get(f"/report?source={bad}")
    assert res.status_code == 422


def test_budget_defaults_to_billed_rows_only(client):
    """The fixture is 100% demo data, so a budget over real money reads $0."""
    body = client.get("/budget").json()
    assert body["spend_usd"] == 0.0
    assert body["source_filter"] == "live,manual"


def test_budget_includes_demo_when_explicitly_asked(client):
    body = client.get("/budget?source=all").json()
    assert body["spend_usd"] > 0.0


def test_calls_carry_their_source(client):
    calls = client.get("/calls?period=all_time&limit=5").json()
    assert calls
    assert all(c["source"] == "demo" for c in calls)


def test_provider_rows_expose_live_call_counts(client):
    rows = client.get("/providers?period=all_time").json()["breakdown"]
    assert rows
    # Nothing in the fixture was ever billed, so no provider has live traffic.
    assert all(r["live_calls"] == 0 for r in rows)
    assert all("calls_by_source" in r for r in rows)


def test_index_has_a_source_filter_and_provenance_banner(client):
    html = client.get("/").text
    assert 'id="source"' in html
    assert 'id="provenance"' in html
    assert 'value="live,manual"' in html
