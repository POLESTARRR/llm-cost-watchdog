"""Pricing correctness: per-provider rates, cache accounting, and the two
provider-specific billing rules that most cost trackers get wrong."""

import pytest

from src.pricing import (
    CACHE_WRITE_MULTIPLIER_1H,
    CACHE_WRITE_MULTIPLIER_5M,
    LONG_CONTEXT_INPUT_MULTIPLIER,
    LONG_CONTEXT_OUTPUT_MULTIPLIER,
    LONG_CONTEXT_THRESHOLD,
    PRICING_TABLE,
    calculate_cost,
    compare_models,
    estimate_cost,
    get_rates,
)
from src.providers import infer_provider


# --- table integrity -----------------------------------------------------


@pytest.mark.parametrize("model", list(PRICING_TABLE))
def test_every_model_has_all_three_rates(model):
    rates = PRICING_TABLE[model]
    assert set(rates) == {"input", "cached_input", "output"}
    assert all(v > 0 for v in rates.values())


@pytest.mark.parametrize("model", list(PRICING_TABLE))
def test_cached_input_is_cheaper_than_input(model):
    """A cached read that costs the same as a fresh read isn't a cache."""
    rates = PRICING_TABLE[model]
    assert rates["cached_input"] < rates["input"]


@pytest.mark.parametrize("model", list(PRICING_TABLE))
def test_every_priced_model_routes_to_a_provider(model):
    assert infer_provider(model) in {"anthropic", "openai", "google"}


def test_all_three_providers_are_represented():
    providers = {infer_provider(m) for m in PRICING_TABLE}
    assert providers == {"anthropic", "openai", "google"}


# --- basic cost math -----------------------------------------------------


@pytest.mark.parametrize("model", list(PRICING_TABLE))
def test_cost_matches_hand_computation(model):
    r = PRICING_TABLE[model]
    expected = (2000 / 1000) * r["input"] + (1000 / 1000) * r["output"]
    assert calculate_cost(model, 2000, 1000) == pytest.approx(expected)


def test_zero_tokens_is_zero():
    assert calculate_cost("claude-opus-5", 0, 0) == 0.0


def test_unknown_model_falls_back_without_raising():
    assert calculate_cost("some-unreleased-model", 1000, 500) > 0
    assert get_rates("some-unreleased-model")["input"] > 0


# --- cache accounting ----------------------------------------------------


def test_cached_tokens_reduce_cost():
    full = calculate_cost("claude-opus-5", 10_000, 1000)
    cached = calculate_cost("claude-opus-5", 10_000, 1000, cached_input_tokens=8000)
    assert cached < full


def test_cached_subset_cannot_exceed_total_input():
    """Guards against a provider reporting more cached tokens than prompt."""
    a = calculate_cost("claude-opus-5", 1000, 100, cached_input_tokens=5000)
    b = calculate_cost("claude-opus-5", 1000, 100, cached_input_tokens=1000)
    assert a == pytest.approx(b)


def test_fully_cached_prompt_bills_entirely_at_cached_rate():
    r = PRICING_TABLE["claude-sonnet-5"]
    cost = calculate_cost("claude-sonnet-5", 5000, 0, cached_input_tokens=5000)
    assert cost == pytest.approx((5000 / 1000) * r["cached_input"])


# --- rule 1: OpenAI long-context surcharge -------------------------------


def test_long_context_surcharge_applies_above_threshold():
    """>272k input bills the WHOLE request at 2x input / 1.5x output."""
    n = LONG_CONTEXT_THRESHOLD + 1
    r = PRICING_TABLE["gpt-5.6-sol"]
    expected = (n / 1000) * r["input"] * LONG_CONTEXT_INPUT_MULTIPLIER + (
        1000 / 1000
    ) * r["output"] * LONG_CONTEXT_OUTPUT_MULTIPLIER
    assert calculate_cost("gpt-5.6-sol", n, 1000) == pytest.approx(expected)


def test_no_surcharge_at_or_below_threshold():
    n = LONG_CONTEXT_THRESHOLD
    r = PRICING_TABLE["gpt-5.6-sol"]
    expected = (n / 1000) * r["input"] + (1000 / 1000) * r["output"]
    assert calculate_cost("gpt-5.6-sol", n, 1000) == pytest.approx(expected)


def test_surcharge_makes_effective_rate_double():
    under = calculate_cost("gpt-5.6-sol", 200_000, 0) / 200_000
    over = calculate_cost("gpt-5.6-sol", 300_000, 0) / 300_000
    assert over == pytest.approx(under * LONG_CONTEXT_INPUT_MULTIPLIER, rel=0.01)


def test_surcharge_does_not_apply_to_other_providers():
    """Anthropic has no long-context surcharge, applying one would overstate."""
    r = PRICING_TABLE["claude-sonnet-5"]
    n = LONG_CONTEXT_THRESHOLD + 50_000
    assert calculate_cost("claude-sonnet-5", n, 0) == pytest.approx((n / 1000) * r["input"])


# --- rule 2: cache-write premium -----------------------------------------


def test_cache_writes_cost_more_than_plain_input_on_gpt56():
    plain = calculate_cost("gpt-5.6-sol", 10_000, 0)
    written = calculate_cost("gpt-5.6-sol", 10_000, 0, cache_write_tokens=10_000)
    assert written == pytest.approx(plain * CACHE_WRITE_MULTIPLIER_5M)


def test_cache_write_ordering_write_gt_plain_gt_read():
    plain = calculate_cost("gpt-5.6-sol", 10_000, 500)
    written = calculate_cost("gpt-5.6-sol", 10_000, 500, cache_write_tokens=8000)
    read = calculate_cost("gpt-5.6-sol", 10_000, 500, cached_input_tokens=8000)
    assert written > plain > read


def test_anthropic_cache_writes_are_surcharged():
    """Regression: Claude was missing from CACHE_WRITE_BILLED_MODELS, so every
    Anthropic cache write billed at a flat 1.0x. That understated this repo's
    own imported build cost by 14.2%."""
    plain = calculate_cost("claude-sonnet-5", 10_000, 0)
    written = calculate_cost("claude-sonnet-5", 10_000, 0, cache_write_tokens=10_000)
    assert written == pytest.approx(plain * CACHE_WRITE_MULTIPLIER_5M)
    assert written > plain


def test_one_hour_cache_writes_cost_more_than_five_minute():
    """Regression: a single 1.25x multiplier priced 1h writes at the 5m rate.
    Real Claude Code traffic is 100% 1h ephemeral."""
    write_5m = calculate_cost("claude-opus-5", 10_000, 0, cache_write_tokens=10_000)
    write_1h = calculate_cost(
        "claude-opus-5", 10_000, 0, cache_write_tokens=10_000, cache_write_1h_tokens=10_000
    )
    plain = calculate_cost("claude-opus-5", 10_000, 0)
    assert write_5m == pytest.approx(plain * CACHE_WRITE_MULTIPLIER_5M)
    assert write_1h == pytest.approx(plain * CACHE_WRITE_MULTIPLIER_1H)
    assert write_1h > write_5m


def test_partial_ttl_split_bills_each_portion_at_its_own_rate():
    r = PRICING_TABLE["claude-opus-5"]
    total = calculate_cost(
        "claude-opus-5", 10_000, 0, cache_write_tokens=10_000, cache_write_1h_tokens=4000
    )
    expected = (
        (6000 / 1000) * r["input"] * CACHE_WRITE_MULTIPLIER_5M
        + (4000 / 1000) * r["input"] * CACHE_WRITE_MULTIPLIER_1H
    )
    assert total == pytest.approx(expected)


def test_one_hour_subset_cannot_exceed_total_cache_writes():
    """An over-large 1h figure must clamp, not invent cost."""
    clamped = calculate_cost(
        "claude-opus-5", 10_000, 0, cache_write_tokens=5000, cache_write_1h_tokens=99_999
    )
    exact = calculate_cost(
        "claude-opus-5", 10_000, 0, cache_write_tokens=5000, cache_write_1h_tokens=5000
    )
    assert clamped == pytest.approx(exact)


def test_batch_tier_bills_at_half():
    standard = calculate_cost("claude-opus-5", 10_000, 1000)
    batch = calculate_cost("claude-opus-5", 10_000, 1000, service_tier="batch")
    assert batch == pytest.approx(standard * 0.5)


def test_unknown_service_tier_bills_at_full_price():
    """Overcharging is the visible error; a new tier name must not crash."""
    standard = calculate_cost("claude-opus-5", 10_000, 1000)
    assert calculate_cost("claude-opus-5", 10_000, 1000, service_tier="flex") == pytest.approx(standard)
    assert calculate_cost("claude-opus-5", 10_000, 1000, service_tier=None) == pytest.approx(standard)


def test_cached_and_written_subsets_do_not_double_bill():
    """cached + written + uncached must partition input_tokens exactly."""
    total = calculate_cost("gpt-5.6-sol", 10_000, 0, cached_input_tokens=4000, cache_write_tokens=3000)
    r = PRICING_TABLE["gpt-5.6-sol"]
    expected = (
        (3000 / 1000) * r["input"]                                  # uncached remainder
        + (4000 / 1000) * r["cached_input"]                         # cache reads
        + (3000 / 1000) * r["input"] * CACHE_WRITE_MULTIPLIER_5M    # cache writes
    )
    assert total == pytest.approx(expected)


# --- comparison helpers --------------------------------------------------


def test_compare_models_sorts_cheapest_first():
    rows = compare_models(5000, 1000)
    costs = [r["cost_usd"] for r in rows]
    assert costs == sorted(costs)


def test_compare_models_reports_multiple_vs_cheapest():
    rows = compare_models(5000, 1000)
    assert rows[0]["vs_cheapest"] == pytest.approx(1.0)
    assert rows[-1]["vs_cheapest"] > 1.0


def test_compare_models_can_be_restricted_to_named_models():
    rows = compare_models(1000, 100, ["claude-opus-5", "gpt-5-nano"])
    assert {r["model"] for r in rows} == {"claude-opus-5", "gpt-5-nano"}


def test_estimate_cost_flags_unpriced_models():
    assert estimate_cost("claude-opus-5", 100, 10)["priced"] is True
    assert estimate_cost("not-a-real-model", 100, 10)["priced"] is False
