"""Subscription provenance, prompt hashing, and the corrected cache-write TTL.

These cover the three corrections that changed this project's own headline
number, so each test names the wrong behaviour it exists to prevent.
"""

import json

import pytest

from src.analyzer import check_budget_status, subscription_roi
from src.guard import check_guards
from src.tracker import BILLED_SOURCES, LIST_PRICE_SOURCES, log_usage, source_totals
from src.usage_schema import UsageEvent
from src.waste import find_duplicate_calls


def _sub_event(cost=10.0, project="build", prompt_hash=None, preview="x" * 60):
    return UsageEvent(
        model="claude-opus-5", provider="anthropic", project_tag=project,
        input_tokens=1000, output_tokens=100, cost_usd=cost, latency_ms=500.0,
        prompt_preview=preview, prompt_hash=prompt_hash, source="subscription",
    )


# --- provenance ----------------------------------------------------------


def test_subscription_is_not_billed_spend():
    """The bug this prevents: reporting flat-fee usage as money charged."""
    assert "subscription" not in BILLED_SOURCES
    assert "subscription" in LIST_PRICE_SOURCES


def test_source_totals_separates_value_from_billed_spend(temp_db):
    log_usage(_sub_event(cost=10.0), db_path=temp_db)
    log_usage(
        UsageEvent(model="claude-opus-5", provider="anthropic", project_tag="live",
                   input_tokens=100, output_tokens=10, cost_usd=1.0, latency_ms=10.0,
                   source="live"),
        db_path=temp_db,
    )
    totals = source_totals("all_time", db_path=temp_db)

    assert totals["billed_cost_usd"] == pytest.approx(1.0)      # only the metered call
    assert totals["subscription_cost_usd"] == pytest.approx(10.0)
    assert totals["list_price_cost_usd"] == pytest.approx(11.0)
    assert totals["has_subscription_data"] is True


def test_budget_ignores_subscription_rows(temp_db):
    """A weekly budget describes metered money. Flat-fee usage cannot exhaust
    it, otherwise importing a build transcript would report you over budget
    on spend that was never charged."""
    log_usage(_sub_event(cost=9999.0), db_path=temp_db)
    assert check_budget_status("weekly")["spend_usd"] == 0.0


def test_guardrails_ignore_subscription_rows(temp_db):
    """A guardrail stops the *next* call; it cannot un-spend a flat fee."""
    log_usage(_sub_event(cost=9999.0), db_path=temp_db)
    result = check_guards(project_tag="build", model="claude-opus-5", estimated_input_tokens=10)
    assert result.allowed


# --- subscription ROI ----------------------------------------------------


def test_roi_reports_value_not_spend(temp_db, monkeypatch):
    monkeypatch.setenv("WATCHDOG_SUBSCRIPTION_USD_PER_MONTH", "20.00")
    for _ in range(3):
        log_usage(_sub_event(cost=10.0), db_path=temp_db)

    roi = subscription_roi("all_time")
    assert roi["list_price_value_usd"] == pytest.approx(30.0)
    assert roi["calls"] == 3
    assert roi["roi_multiple"] > 1
    assert "not money spent" in roi["note"]


def test_roi_is_empty_without_subscription_rows(temp_db):
    assert subscription_roi("all_time")["list_price_value_usd"] == 0.0


def test_roi_never_charges_less_than_a_days_subscription(temp_db):
    """A single call must not divide by ~zero elapsed time and report an
    absurd multiple."""
    log_usage(_sub_event(cost=10.0), db_path=temp_db)
    roi = subscription_roi("all_time")
    assert roi["subscription_cost_usd"] > 0


# --- prompt hashing ------------------------------------------------------


def test_hash_distinguishes_prompts_sharing_a_preview():
    """The documented civil-prep false positive: a long fixed instruction
    template made 22 different questions look identical at 80 characters."""
    template = "You are a careful assistant. Follow the rubric exactly. " * 3
    a = UsageEvent.make_hash(template + "What is the capital of France?")
    b = UsageEvent.make_hash(template + "Explain the water cycle.")

    assert UsageEvent.make_preview(template + "Q1") == UsageEvent.make_preview(template + "Q2")
    assert a != b


def test_identical_prompts_hash_identically():
    assert UsageEvent.make_hash("same prompt") == UsageEvent.make_hash("same prompt")


def test_hash_of_empty_prompt_is_none():
    assert UsageEvent.make_hash("") is None


def test_duplicate_detection_uses_hash_over_preview(temp_db):
    shared = "y" * 60
    log_usage(_sub_event(prompt_hash="hash-a", preview=shared), db_path=temp_db)
    log_usage(_sub_event(prompt_hash="hash-b", preview=shared), db_path=temp_db)

    # Same preview, different prompts -> not duplicates.
    assert find_duplicate_calls(period="all_time", source="subscription") == []


def test_duplicate_detection_flags_genuinely_repeated_prompts(temp_db):
    shared = "z" * 60
    for _ in range(3):
        log_usage(_sub_event(prompt_hash="same-hash", preview=shared), db_path=temp_db)

    rows = find_duplicate_calls(period="all_time", source="subscription")
    assert len(rows) == 1
    assert rows[0]["times_sent"] == 3
    assert rows[0]["matched_on"] == "hash"


def test_unhashed_rows_still_match_on_preview_and_say_so(temp_db):
    """Rows predating the column must keep working, and must be labelled so a
    preview match can be read with the appropriate suspicion."""
    shared = "w" * 60
    for _ in range(2):
        log_usage(_sub_event(prompt_hash=None, preview=shared), db_path=temp_db)

    rows = find_duplicate_calls(period="all_time", source="subscription")
    assert rows and rows[0]["matched_on"] == "preview"


def test_hashed_and_unhashed_rows_never_share_a_group(temp_db):
    shared = "v" * 60
    log_usage(_sub_event(prompt_hash="h", preview=shared), db_path=temp_db)
    log_usage(_sub_event(prompt_hash=None, preview=shared), db_path=temp_db)
    assert find_duplicate_calls(period="all_time", source="subscription") == []


# --- importer ------------------------------------------------------------


def _transcript(tmp_path, turns):
    path = tmp_path / "session.jsonl"
    with open(path, "w") as f:
        for rec in turns:
            f.write(json.dumps(rec) + "\n")
    return str(path)


def test_importer_reads_the_ttl_split_and_prices_1h_writes(tmp_path, temp_db):
    from scripts.import_claude_code_usage import load
    from src.pricing import calculate_cost

    turns = [
        {"type": "user", "timestamp": "2026-08-01T10:00:00.000Z",
         "message": {"content": [{"type": "text", "text": "hello"}]}},
        {"type": "assistant", "timestamp": "2026-08-01T10:00:05.000Z",
         "message": {"id": "msg_1", "model": "claude-opus-5", "usage": {
             "input_tokens": 10, "output_tokens": 100,
             "cache_read_input_tokens": 500, "cache_creation_input_tokens": 1000,
             "cache_creation": {"ephemeral_1h_input_tokens": 1000,
                                "ephemeral_5m_input_tokens": 0},
             "service_tier": "standard"}}},
    ]
    stats = load(_transcript(tmp_path, turns), "proj", db_path=temp_db)

    expected = calculate_cost(
        "claude-opus-5", 1510, 100, 500, 1000,
        cache_write_1h_tokens=1000, service_tier="standard",
    )
    assert stats["logged"] == 1
    assert stats["total_cost"] == pytest.approx(expected, rel=1e-6)


def test_importer_derives_latency_from_timestamps(tmp_path, temp_db):
    """Regression: every imported row carried latency_ms=0, so latency was
    unmeasurable for the largest dataset in the project."""
    from scripts.import_claude_code_usage import load
    from src.tracker import get_events

    turns = [
        {"type": "user", "timestamp": "2026-08-01T10:00:00.000Z",
         "message": {"content": [{"type": "text", "text": "hello"}]}},
        {"type": "assistant", "timestamp": "2026-08-01T10:00:03.500Z",
         "message": {"id": "msg_1", "model": "claude-opus-5",
                     "usage": {"input_tokens": 10, "output_tokens": 10}}},
    ]
    load(_transcript(tmp_path, turns), "proj", db_path=temp_db)

    event = get_events(db_path=temp_db)[0]
    assert event.latency_ms == pytest.approx(3500.0)
    assert event.prompt_hash is not None


def test_importer_discards_implausible_idle_gaps(tmp_path, temp_db):
    """An overnight gap is idle time, not latency, recording it would wreck
    the anomaly detector's rolling averages."""
    from scripts.import_claude_code_usage import load
    from src.tracker import get_events

    turns = [
        {"type": "user", "timestamp": "2026-08-01T10:00:00.000Z",
         "message": {"content": [{"type": "text", "text": "hi"}]}},
        {"type": "assistant", "timestamp": "2026-08-01T18:00:00.000Z",
         "message": {"id": "msg_1", "model": "claude-opus-5",
                     "usage": {"input_tokens": 10, "output_tokens": 10}}},
    ]
    load(_transcript(tmp_path, turns), "proj", db_path=temp_db)
    assert get_events(db_path=temp_db)[0].latency_ms == 0.0


def test_importer_defaults_to_subscription_provenance(tmp_path, temp_db):
    from scripts.import_claude_code_usage import load
    from src.tracker import get_events

    turns = [
        {"type": "user", "timestamp": "2026-08-01T10:00:00.000Z",
         "message": {"content": [{"type": "text", "text": "hi"}]}},
        {"type": "assistant", "timestamp": "2026-08-01T10:00:01.000Z",
         "message": {"id": "msg_1", "model": "claude-opus-5",
                     "usage": {"input_tokens": 10, "output_tokens": 10}}},
    ]
    load(_transcript(tmp_path, turns), "proj", db_path=temp_db)
    assert get_events(db_path=temp_db)[0].source == "subscription"


def test_importer_checkpoints_are_per_project(tmp_path, temp_db):
    """One working directory can hold several projects; a shared checkpoint
    would let the first import mark the others' turns as already done."""
    from scripts.import_claude_code_usage import load

    turns = [
        {"type": "assistant", "timestamp": "2026-08-01T10:00:01.000Z",
         "message": {"id": "msg_1", "model": "claude-opus-5",
                     "usage": {"input_tokens": 10, "output_tokens": 10}}},
    ]
    path = _transcript(tmp_path, turns)
    assert load(path, "project-a", db_path=temp_db)["logged"] == 1
    assert load(path, "project-b", db_path=temp_db)["logged"] == 1   # not blocked
    assert load(path, "project-a", db_path=temp_db)["logged"] == 0   # already done
