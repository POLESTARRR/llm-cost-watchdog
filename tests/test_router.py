"""Routing: model groups, cooldowns, history-based selection, and simulation."""

import pytest

from src.router import (
    RoutingError,
    active_cooldowns,
    clear_cooldown,
    model_groups,
    model_stats,
    resolve_group,
    router_status,
    select,
    simulate_routing,
    start_cooldown,
)
from src.tracker import log_usage
from src.usage_schema import UsageEvent


@pytest.fixture
def groups(monkeypatch):
    monkeypatch.setenv("WATCHDOG_GROUP_FAST", "gemini-flash-lite-latest,claude-haiku-4-5,gpt-5-nano")
    monkeypatch.setenv("WATCHDOG_GROUP_SMART", "claude-opus-5,gpt-5.6-sol")
    # Treat every provider as configured so selection tests exercise the
    # strategy, not the local machine's credentials.
    monkeypatch.setattr(
        "src.providers.configured_providers",
        lambda: {"google": True, "anthropic": True, "openai": True},
    )
    return None


def _event(model, *, cost=0.01, latency=100.0, success=True, project="p"):
    return UsageEvent(
        model=model, provider="anthropic", project_tag=project,
        input_tokens=1000, output_tokens=100, cost_usd=cost,
        latency_ms=latency, success=success, source="live",
    )


# --- groups --------------------------------------------------------------


def test_groups_read_from_environment(groups):
    assert set(model_groups()) >= {"fast", "smart"}
    assert "claude-haiku-4-5" in resolve_group("fast")


def test_unknown_group_names_the_env_var_to_set(groups):
    with pytest.raises(RoutingError, match="WATCHDOG_GROUP_NOPE"):
        resolve_group("nope")


def test_groups_are_read_at_call_time_not_import_time(monkeypatch):
    """A group added after import must be visible without a restart."""
    monkeypatch.setenv("WATCHDOG_GROUP_LATEBOUND", "claude-haiku-4-5")
    assert "latebound" in model_groups()


# --- cooldowns -----------------------------------------------------------


def test_cooldown_benches_a_model(temp_db):
    start_cooldown("claude-haiku-4-5", seconds=60, db_path=temp_db)
    benched = active_cooldowns(db_path=temp_db)
    assert "claude-haiku-4-5" in benched
    assert benched["claude-haiku-4-5"]["seconds_remaining"] > 0


def test_expired_cooldown_is_not_reported(temp_db):
    start_cooldown("claude-haiku-4-5", seconds=-1, db_path=temp_db)
    assert active_cooldowns(db_path=temp_db) == {}


def test_cooldown_is_idempotent_and_clearable(temp_db):
    start_cooldown("gpt-5-nano", seconds=60, db_path=temp_db)
    start_cooldown("gpt-5-nano", seconds=60, db_path=temp_db)
    assert len(active_cooldowns(db_path=temp_db)) == 1
    clear_cooldown("gpt-5-nano", db_path=temp_db)
    assert active_cooldowns(db_path=temp_db) == {}


def test_cooling_model_is_excluded_from_selection(groups, temp_db):
    cheapest = select("fast", strategy="cheapest", db_path=temp_db).model
    start_cooldown(cheapest, seconds=60, db_path=temp_db)

    decision = select("fast", strategy="cheapest", db_path=temp_db)
    assert decision.model != cheapest
    assert cheapest in decision.excluded
    assert "cooling down" in decision.excluded[cheapest]


def test_every_member_cooling_is_an_error_not_a_silent_pick(groups, temp_db):
    for model in resolve_group("fast"):
        start_cooldown(model, seconds=60, db_path=temp_db)
    with pytest.raises(RoutingError, match="no usable model"):
        select("fast", db_path=temp_db)


# --- strategy ------------------------------------------------------------


def test_cheapest_picks_the_cheapest_member(groups, temp_db):
    decision = select("fast", strategy="cheapest", db_path=temp_db)
    assert decision.model == "gpt-5-nano"  # cheapest in the pricing table
    assert decision.strategy == "cheapest"
    assert "cheapest" in decision.basis


def test_unknown_strategy_is_rejected(groups, temp_db):
    with pytest.raises(RoutingError, match="unknown strategy"):
        select("fast", strategy="vibes", db_path=temp_db)


def test_history_strategy_falls_back_to_price_without_enough_data(groups, temp_db):
    """One lucky call must not decide routing for everything after it."""
    log_usage(_event("claude-haiku-4-5", latency=1.0), db_path=temp_db)
    decision = select("fast", strategy="lowest-latency", db_path=temp_db)
    assert "fell back to cheapest" in decision.basis


def test_lowest_latency_uses_measured_history(groups, temp_db):
    for _ in range(5):
        log_usage(_event("claude-haiku-4-5", latency=50.0), db_path=temp_db)
        log_usage(_event("gpt-5-nano", latency=9000.0), db_path=temp_db)

    decision = select("fast", strategy="lowest-latency", db_path=temp_db)
    assert decision.model == "claude-haiku-4-5"
    assert "lowest measured latency" in decision.basis


def test_lowest_failure_prefers_the_reliable_model(groups, temp_db):
    for _ in range(5):
        log_usage(_event("claude-haiku-4-5", success=True), db_path=temp_db)
        log_usage(_event("gpt-5-nano", success=False), db_path=temp_db)

    decision = select("fast", strategy="lowest-failure", db_path=temp_db)
    assert decision.model == "claude-haiku-4-5"


def test_failed_calls_do_not_pollute_measured_latency(temp_db):
    """A 429 logs cost=0 and near-zero latency; counting it would make a
    failing model look like the fastest one."""
    for _ in range(4):
        log_usage(_event("gpt-5-nano", latency=0.0, success=False), db_path=temp_db)
    log_usage(_event("gpt-5-nano", latency=500.0, success=True), db_path=temp_db)

    stats = model_stats()["gpt-5-nano"]
    assert stats["avg_latency_ms"] == 500.0
    assert stats["failure_rate"] == pytest.approx(0.8)


def test_decision_records_why(groups, temp_db):
    d = select("fast", strategy="cheapest", db_path=temp_db).as_dict()
    assert set(d) == {"group", "model", "strategy", "candidates", "excluded", "basis"}
    assert d["model"] in d["candidates"]


# --- pre-call context check ----------------------------------------------


def test_model_with_unknown_context_window_is_not_guessed_at(groups, temp_db, monkeypatch):
    """No window data must mean 'leave it alone', never 'assume it fits'."""
    monkeypatch.setattr("src.pricing_drift.context_windows", lambda refresh=False: {})
    decision = select("fast", estimated_input_tokens=10_000_000, db_path=temp_db)
    assert decision.model  # still routed, nothing excluded on window grounds
    assert not any("window" in r for r in decision.excluded.values())


def test_prompt_too_large_for_a_window_excludes_that_model(groups, temp_db, monkeypatch):
    monkeypatch.setattr(
        "src.pricing_drift.context_windows",
        lambda refresh=False: {"gpt-5-nano": 1000, "claude-haiku-4-5": 200_000},
    )
    decision = select("fast", estimated_input_tokens=50_000, db_path=temp_db)
    assert decision.model != "gpt-5-nano"
    assert "window" in decision.excluded["gpt-5-nano"]


# --- simulation ----------------------------------------------------------


def test_simulate_routing_reprices_real_traffic(groups, temp_db):
    for _ in range(3):
        log_usage(_event("claude-opus-5", cost=1.0), db_path=temp_db)

    result = simulate_routing("fast", "cheapest", period="all_time")
    assert result["calls_repriced"] == 3
    assert result["actual_cost_usd"] == pytest.approx(3.0)
    assert result["simulated_cost_usd"] < result["actual_cost_usd"]
    assert result["verdict"] == "cheaper"
    assert sum(result["routed_to"].values()) == 3


def test_simulate_routing_excludes_failed_calls(groups, temp_db):
    log_usage(_event("claude-opus-5", cost=1.0), db_path=temp_db)
    log_usage(_event("claude-opus-5", cost=0.0, success=False), db_path=temp_db)
    assert simulate_routing("fast", "cheapest", period="all_time")["calls_repriced"] == 1


def test_simulate_routing_on_empty_period_is_not_an_error(groups, temp_db):
    result = simulate_routing("fast", "cheapest", period="today")
    assert result["calls_repriced"] == 0
    assert "note" in result


def test_simulation_states_its_own_caveat(groups, temp_db):
    log_usage(_event("claude-opus-5", cost=1.0), db_path=temp_db)
    result = simulate_routing("fast", "cheapest", period="all_time")
    assert "quality" in result["caveat"] or "judge" in result["caveat"]


# --- status --------------------------------------------------------------


def test_router_status_reports_configuration(groups, temp_db):
    start_cooldown("gpt-5-nano", seconds=60, db_path=temp_db)
    status = router_status(db_path=temp_db)
    assert "fast" in status["groups"]
    assert "gpt-5-nano" in status["cooldowns"]
    assert status["strategy"] in status["available_strategies"]


def test_router_status_flags_unpriced_group_members(monkeypatch, temp_db):
    monkeypatch.setenv("WATCHDOG_GROUP_ODD", "claude-haiku-4-5,not-a-real-model")
    assert "not-a-real-model" in router_status(db_path=temp_db)["unpriced_members"]
