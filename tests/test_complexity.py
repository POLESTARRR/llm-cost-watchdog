"""The classifier's job is to be *predictably* wrong, not never wrong.

These tests pin the behaviour that routing depends on: mechanical work goes
cheap, reasoning work goes expensive, and anything unclear escalates rather
than saving a fraction of a cent.
"""

import pytest

from src.complexity import (
    TIER_COMPLEX,
    TIER_MODERATE,
    TIER_TRIVIAL,
    classify,
    tier_index,
)

TRIVIAL_PROMPTS = [
    "reformat this JSON",
    "rename the variable foo to bar",
    "sort this list alphabetically",
    "what does the -f flag do",
    "add a docstring to this function",
]

COMPLEX_PROMPTS = [
    "Refactor this module so the storage layer can be swapped without touching the callers, "
    "and explain the trade-offs of each approach you considered.",
    "Why is this test flaky? Walk me through the possible race conditions step by step.",
    "Design a schema migration strategy that works with zero downtime.",
]


@pytest.mark.parametrize("prompt", TRIVIAL_PROMPTS)
def test_mechanical_work_is_trivial(prompt):
    assert classify(prompt).tier == TIER_TRIVIAL


@pytest.mark.parametrize("prompt", COMPLEX_PROMPTS)
def test_reasoning_work_is_complex(prompt):
    assert classify(prompt).tier == TIER_COMPLEX


def test_ambiguity_escalates_rather_than_saving_money():
    """The asymmetry that justifies the thresholds.

    A bare question with no signal either way must not land in the cheap tier.
    Being wrong downward costs an hour; being wrong upward costs a fraction of
    a cent.
    """
    assert classify("Can you look at the user service?").tier == TIER_MODERATE


def test_empty_prompt_escalates_instead_of_looking_easy():
    v = classify("")
    assert v.tier == TIER_MODERATE
    assert "empty prompt" in v.signals


def test_stack_trace_is_complex_without_any_verb():
    """An error dump is debugging work even when nothing asks a question."""
    prompt = (
        "Traceback (most recent call last):\n"
        '  File "app.py", line 12, in <module>\n'
        "    main()\n"
        "ZeroDivisionError: division by zero"
    )
    assert classify(prompt).tier == TIER_COMPLEX


def test_long_prompts_escalate():
    assert classify("summarize this: " + ("word " * 2000)).tier == TIER_COMPLEX


def test_every_verdict_explains_itself():
    """A routing decision nobody can audit is a routing decision nobody trusts."""
    v = classify("Refactor the auth module and explain why")
    assert v.signals
    assert all(isinstance(s, str) and s for s in v.signals)
    assert str(v).startswith(v.tier)


def test_trivial_verbs_can_be_outweighed_by_real_complexity():
    """A mechanical verb inside a hard request must not drag it to the cheap tier."""
    prompt = (
        "Rename the handler, then refactor the dispatch layer so retries are idempotent, "
        "and explain why the current design deadlocks under concurrency. " + "context " * 400
    )
    assert classify(prompt).tier == TIER_COMPLEX


class TestTierIndex:
    def test_maps_tiers_across_a_full_ladder(self):
        assert tier_index(TIER_TRIVIAL, 3) == 0
        assert tier_index(TIER_MODERATE, 3) == 1
        assert tier_index(TIER_COMPLEX, 3) == 2

    def test_degrades_when_the_group_is_smaller_than_the_tiers(self):
        assert tier_index(TIER_COMPLEX, 1) == 0
        assert tier_index(TIER_TRIVIAL, 2) == 0
        assert tier_index(TIER_COMPLEX, 2) == 1

    def test_never_indexes_out_of_range(self):
        for n in range(1, 6):
            for tier in (TIER_TRIVIAL, TIER_MODERATE, TIER_COMPLEX):
                assert 0 <= tier_index(tier, n) < n


# --- context size: the signal the words cannot carry ----------------------


def test_same_words_route_differently_by_how_much_there_is_to_read():
    """The point of the whole context signal, in one assertion.

    "fix the login bug" is three words either way. In a fresh session it is a
    small job; four hundred thousand tokens deep it is a request to change
    something inside a system the model has to read first. Measured traffic says
    the second case runs far longer, and nothing in the wording says so.
    """
    from src.complexity import classify

    early = classify("fix the login bug", context_tokens=1_000)
    late = classify("fix the login bug", context_tokens=600_000)

    assert early.tier == "trivial"
    assert late.tier == "complex"
    assert late.score > early.score


def test_small_context_never_demotes_a_hard_request():
    """Regression: the fallback test caught this the day it was introduced.

    A small conversation is weak evidence of an easy request. It must not cancel
    strong evidence of a hard one, or the first architectural question of a
    session gets sent to a mid-tier model because nothing had been said yet.
    """
    from src.complexity import classify

    v = classify(
        "Design a migration strategy and explain the trade-offs.",
        context_tokens=1_000,
    )
    assert v.tier == "complex"
    assert any("no discount" in s for s in v.signals)


def test_large_context_escalates_even_with_no_hard_words():
    from src.complexity import classify

    v = classify("carry on", context_tokens=500_000)
    assert v.tier == "complex"


def test_context_is_optional_and_omitting_it_keeps_word_only_behaviour():
    from src.complexity import classify

    assert classify("reformat this JSON").tier == classify(
        "reformat this JSON", context_tokens=None
    ).tier
