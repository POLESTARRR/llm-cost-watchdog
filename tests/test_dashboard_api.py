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
    # `ollama` is listed like any other provider but, unlike the hosted three,
    # its configured-ness depends on whether a local server happens to be up,
    # so this asserts membership rather than a fixed True/False.
    assert set(body["configured"]) == {"google", "anthropic", "openai", "ollama"}
    # The breakdown covers providers with recorded traffic; the sample fixture
    # predates local models, so ollama is absent here by construction.
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
    """prompt_preview is capped at 80 chars, a privacy guarantee, so assert it."""
    for call in client.get("/calls?period=all_time&limit=44").json():
        assert len(call["prompt_preview"]) <= 80


# --- static frontend -----------------------------------------------------


def test_index_html_is_served(client):
    res = client.get("/")
    assert res.status_code == 200
    assert "text/html" in res.headers["content-type"]
    assert "LLM Cost Gateway" in res.text


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


def test_budget_defaults_to_runtime_rows_only(client):
    """The fixture is 100% demo data, so a budget over real money reads $0.

    The default is `live` rather than `live,manual`: a weekly budget describes
    an ongoing run rate, and backfilled build-cost imports are history.
    """
    body = client.get("/budget").json()
    assert body["spend_usd"] == 0.0
    assert body["source_filter"] == "live"


def test_budget_includes_demo_when_explicitly_asked(client, temp_db):
    """Asking for every source must count demo rows the default excludes.

    Seeds its own row rather than relying on sample_usage.json: the fixture's
    timestamps are fixed, so as the file ages out of the trailing 7-day window
    a weekly budget over it reads $0 regardless of the source filter, and this
    test silently stopped exercising the behaviour it names.
    """
    from src.tracker import log_usage
    from src.usage_schema import UsageEvent

    log_usage(
        UsageEvent(
            model="claude-opus-5", provider="anthropic", project_tag="demo-proj",
            input_tokens=1000, output_tokens=100, cost_usd=0.5,
            latency_ms=100.0, source="demo",
        ),
        db_path=temp_db,
    )

    default_body = client.get("/budget").json()
    all_body = client.get("/budget?source=all").json()

    assert default_body["spend_usd"] == 0.0  # demo row excluded by default
    assert all_body["spend_usd"] > 0.0       # and counted when asked for


def test_budget_explains_why_it_reads_zero(client, temp_db):
    """A budget over metered spend reads $0 when everything was covered by a
    subscription. Showing only 0% next to real recorded activity reads as a
    broken widget, so the response must say why."""
    from src.tracker import log_usage
    from src.usage_schema import UsageEvent

    log_usage(
        UsageEvent(
            model="claude-opus-5", provider="anthropic", project_tag="build",
            input_tokens=1000, output_tokens=100, cost_usd=25.0,
            latency_ms=100.0, source="subscription",
        ),
        db_path=temp_db,
    )
    body = client.get("/budget").json()
    assert body["spend_usd"] == 0.0
    assert "note" in body
    assert body["uncounted_cost_usd"] == pytest.approx(25.0)
    assert body["uncounted_calls"] == 1


def test_budget_omits_the_note_when_there_is_metered_spend(client, temp_db):
    from src.tracker import log_usage
    from src.usage_schema import UsageEvent

    log_usage(
        UsageEvent(
            model="claude-opus-5", provider="anthropic", project_tag="app",
            input_tokens=1000, output_tokens=100, cost_usd=1.0,
            latency_ms=100.0, source="live",
        ),
        db_path=temp_db,
    )
    assert "note" not in client.get("/budget").json()


def test_roi_endpoint(client):
    body = client.get("/roi").json()
    assert "list_price_value_usd" in body


def test_router_endpoint_reports_configuration(client):
    body = client.get("/router").json()
    assert set(body) >= {"strategy", "groups", "cooldowns", "available_strategies"}


def test_pricing_drift_endpoint(client):
    body = client.get("/pricing-drift").json()
    assert "checked" in body
    assert "drifted" in body


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
    # The two kinds of row the page makes claims about must be separately
    # selectable: measured build work, and traffic the gateway itself served.
    assert 'value="subscription,manual"' in html
    assert 'value="live"' in html


def test_index_defaults_to_all_time(client):
    """The default period must not exclude the data the page is about.

    "Last 7 days" was the default while the measured work spanned a fixed twelve
    days in the past, so the page opened on a near-empty slice of a full ledger
    and read as a broken or fabricated dashboard. A default that hides the
    subject is worse than no default.
    """
    html = client.get("/").text
    assert '<option value="all_time" selected>' in html
    assert '<option value="week" selected>' not in html


# --- /import (remote ingest for build-cost imports) ------------------------


@pytest.fixture
def import_key(monkeypatch):
    """Enable the /import endpoint for a test by patching the module-level
    IMPORT_KEY directly, it's read from the environment once at import
    time, so a plain monkeypatch.setenv after that point has no effect.
    """
    import dashboard.app as dashboard_app
    monkeypatch.setattr(dashboard_app, "IMPORT_KEY", "test-import-key")
    return "test-import-key"


def _import_event(**overrides):
    event = {
        "model": "claude-sonnet-5",
        "provider": "anthropic",
        "project_tag": "some-project-build",
        "input_tokens": 10_000,
        "output_tokens": 2_000,
    }
    event.update(overrides)
    return event


def test_import_disabled_without_key(client, monkeypatch):
    # Explicitly force the "not configured" state, relying on the ambient
    # environment not having WATCHDOG_IMPORT_KEY set is fragile (exactly
    # this test broke the moment a real deployment key was added to .env
    # for manual testing).
    import dashboard.app as dashboard_app
    monkeypatch.setattr(dashboard_app, "IMPORT_KEY", None)

    res = client.post("/import", json={"events": [_import_event()]})
    assert res.status_code == 403


def test_import_rejects_wrong_key(client, import_key):
    res = client.post(
        "/import", json={"events": [_import_event()]},
        headers={"X-Watchdog-Import-Key": "wrong-key"},
    )
    assert res.status_code == 401


def test_import_rejects_missing_key_header(client, import_key):
    res = client.post("/import", json={"events": [_import_event()]})
    assert res.status_code == 401


def test_import_logs_events_with_correct_key(client, import_key):
    res = client.post(
        "/import", json={"events": [_import_event()]},
        headers={"X-Watchdog-Import-Key": import_key},
    )
    assert res.status_code == 200
    body = res.json()
    assert body["logged"] == 1
    assert body["total_cost_usd"] > 0

    calls = client.get("/calls?period=all_time&limit=5&source=manual").json()
    assert any(c["project_tag"] == "some-project-build" for c in calls)


def test_import_recomputes_cost_server_side(client, import_key):
    """The request body has no cost field at all, cost always comes from
    this project's own pricing table, never the caller."""
    from src.pricing import calculate_cost

    res = client.post(
        "/import", json={"events": [_import_event(input_tokens=50_000, output_tokens=5_000)]},
        headers={"X-Watchdog-Import-Key": import_key},
    )
    expected = calculate_cost("claude-sonnet-5", 50_000, 5_000)
    assert res.json()["total_cost_usd"] == pytest.approx(expected, rel=1e-6)


def test_import_skips_models_with_no_pricing(client, import_key):
    res = client.post(
        "/import", json={"events": [_import_event(model="not-a-real-model")]},
        headers={"X-Watchdog-Import-Key": import_key},
    )
    body = res.json()
    assert body["logged"] == 0
    assert body["skipped_unpriced"] == 1


def test_import_failed_call_costs_nothing(client, import_key):
    res = client.post(
        "/import", json={"events": [_import_event(success=False, output_tokens=0)]},
        headers={"X-Watchdog-Import-Key": import_key},
    )
    assert res.json()["total_cost_usd"] == 0.0


def test_import_defaults_source_to_manual(client, import_key):
    client.post(
        "/import", json={"events": [_import_event()]},
        headers={"X-Watchdog-Import-Key": import_key},
    )
    calls = client.get("/calls?period=all_time&limit=5&source=manual").json()
    matching = [c for c in calls if c["project_tag"] == "some-project-build"]
    assert matching and matching[0]["source"] == "manual"


# --- routes added after the catch-all static mount --------------------------
#
# Regression: /shadow and /complexity were originally appended to the end of
# app.py, i.e. AFTER `app.mount("/", StaticFiles(...))`. The mount matches every
# path, so both endpoints 404'd while looking perfectly correct in the source.
# These tests fail if anyone appends a route to the bottom of the file again.


def test_complexity_endpoint_is_reachable_and_explains_itself(client):
    body = client.get("/complexity", params={"prompt": "reformat this JSON"}).json()
    assert body["tier"] == "trivial"
    assert body["signals"]


def test_complexity_endpoint_rejects_an_empty_prompt(client):
    assert client.get("/complexity", params={"prompt": ""}).status_code == 422


def test_shadow_endpoint_is_reachable(client):
    body = client.get("/shadow").json()
    assert body["total_comparisons"] == 0
    assert "NOT a saving" in body["note"]


def test_gateway_is_mounted_on_the_same_app(client):
    """One process serves the proxy and the UI that reads what it recorded."""
    assert client.get("/v1/models").status_code == 200


# --- /validation (dated classifier-validation artifact) --------------------


def test_validation_reports_the_shipped_artifact(client):
    """The published result must carry both verdicts and the date it was made."""
    body = client.get("/validation").json()
    if not body.get("available"):
        # A clone without the artifact is a supported state, not a failure.
        assert "reason" in body
        return

    assert body["n_prompts"] > 0
    assert body["generated_at"]
    # Both questions are always answered. Ranking well is not permitted to stand
    # in for routing well, which is the distinction the whole analysis exists on.
    assert "passes" in body["ranking"]
    assert "passes" in body["routing"]
    assert set(body["tiers"]) <= {"trivial", "moderate", "complex"}


def test_validation_tier_shares_are_a_partition(client):
    body = client.get("/validation").json()
    if not body.get("available"):
        return
    total = sum(t["share_percent"] for t in body["tiers"].values())
    assert total == pytest.approx(100.0, abs=0.5)


def test_index_hides_validation_until_it_loads(client):
    """The section must ship hidden: an empty study is worse than no study."""
    html = client.get("/").text
    assert 'id="validation"' in html
    assert 'id="validation" hidden' in html
