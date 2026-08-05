"""call_llm: the tracked wrapper. Retry behavior, and the guarantee that no
call escapes untracked."""

import pytest

from src import utils
from src.providers.base import LLMResponse
from src.tracker import get_events


class _FakeProvider:
    """Fails with `fail_with` for the first `fail_times` calls, then succeeds."""

    def __init__(self, fail_times=0, fail_with=None, response=None):
        self.fail_times = fail_times
        self.fail_with = fail_with or RuntimeError("boom")
        self.response = response or LLMResponse(
            text="ok", input_tokens=100, output_tokens=20, cached_input_tokens=40
        )
        self.calls = 0

    def complete(self, prompt, model, temperature):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise self.fail_with
        return self.response


class RateLimitError(Exception):
    """Named to match what the rate-limit sniffer looks for."""


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    """Never actually back off during tests."""
    monkeypatch.setattr(utils.time, "sleep", lambda _s: None)


def _install(monkeypatch, provider):
    monkeypatch.setattr(utils, "get_provider", lambda model: provider)


# --- happy path ----------------------------------------------------------


def test_successful_call_returns_text_and_logs_one_event(temp_db, monkeypatch):
    _install(monkeypatch, _FakeProvider())

    text = utils.call_llm("hi", model="claude-opus-5", project_tag="proj")
    assert text == "ok"

    events = get_events(db_path=temp_db)
    assert len(events) == 1
    e = events[0]
    assert e.success is True
    assert e.provider == "anthropic"       # inferred from the model ID
    assert e.project_tag == "proj"
    assert e.cached_input_tokens == 40
    assert e.cost_usd > 0


def test_cost_reflects_the_cached_discount(temp_db, monkeypatch):
    """A call with cached tokens must cost less than the same call without."""
    _install(monkeypatch, _FakeProvider(
        response=LLMResponse(text="ok", input_tokens=10_000, output_tokens=100, cached_input_tokens=9000)))
    utils.call_llm("hi", model="claude-opus-5")
    cheap = get_events(db_path=temp_db)[0].cost_usd

    from src.pricing import calculate_cost
    assert cheap < calculate_cost("claude-opus-5", 10_000, 100)


def test_prompt_is_truncated_to_a_preview(temp_db, monkeypatch):
    _install(monkeypatch, _FakeProvider())
    utils.call_llm("x" * 500, model="claude-opus-5")
    assert len(get_events(db_path=temp_db)[0].prompt_preview) == 80


# --- failures are still tracked -----------------------------------------


def test_non_retryable_failure_is_logged_then_raised(temp_db, monkeypatch):
    _install(monkeypatch, _FakeProvider(fail_times=99, fail_with=ValueError("bad request")))

    with pytest.raises(ValueError):
        utils.call_llm("hi", model="claude-opus-5")

    events = get_events(db_path=temp_db)
    assert len(events) == 1
    assert events[0].success is False
    assert "bad request" in events[0].error


def test_non_retryable_failure_is_not_retried(temp_db, monkeypatch):
    provider = _FakeProvider(fail_times=99, fail_with=ValueError("bad request"))
    _install(monkeypatch, provider)

    with pytest.raises(ValueError):
        utils.call_llm("hi", model="claude-opus-5")

    assert provider.calls == 1  # no wasted retries on a 400-class error


# --- retry on rate limits ------------------------------------------------


def test_rate_limit_is_retried_and_can_succeed(temp_db, monkeypatch):
    provider = _FakeProvider(fail_times=2, fail_with=RateLimitError("429 slow down"))
    _install(monkeypatch, provider)

    assert utils.call_llm("hi", model="claude-opus-5") == "ok"
    assert provider.calls == 3


def test_every_retry_attempt_is_logged(temp_db, monkeypatch):
    """Retry volume is itself a cost signal, so failed attempts are kept."""
    _install(monkeypatch, _FakeProvider(fail_times=2, fail_with=RateLimitError("429")))

    utils.call_llm("hi", model="claude-opus-5")

    events = get_events(db_path=temp_db)
    assert len(events) == 3
    assert [e.success for e in events] == [False, False, True]


def test_retries_are_exhausted_then_raised(temp_db, monkeypatch):
    provider = _FakeProvider(fail_times=99, fail_with=RateLimitError("429"))
    _install(monkeypatch, provider)

    with pytest.raises(RateLimitError):
        utils.call_llm("hi", model="claude-opus-5", max_retries=2)

    assert provider.calls == 3  # initial + 2 retries
    assert len(get_events(db_path=temp_db)) == 3


def test_retries_can_be_disabled(temp_db, monkeypatch):
    provider = _FakeProvider(fail_times=99, fail_with=RateLimitError("429"))
    _install(monkeypatch, provider)

    with pytest.raises(RateLimitError):
        utils.call_llm("hi", model="claude-opus-5", max_retries=0)

    assert provider.calls == 1


# --- rate-limit detection ------------------------------------------------


@pytest.mark.parametrize("exc", [
    RateLimitError("too many requests"),
    Exception("HTTP 429 Too Many Requests"),
    Exception("You exceeded your current quota"),
    type("ResourceExhausted", (Exception,), {})("resource exhausted"),
])
def test_detects_rate_limits_across_provider_sdks(exc):
    assert utils._is_rate_limit(exc) is True


@pytest.mark.parametrize("exc", [
    ValueError("invalid model"),
    Exception("HTTP 401 unauthorized"),
    Exception("connection reset"),
])
def test_does_not_treat_other_errors_as_rate_limits(exc):
    assert utils._is_rate_limit(exc) is False


def test_backoff_grows_and_is_capped():
    ceilings = [max(utils._backoff_delay(i) for _ in range(200)) for i in range(3)]
    assert ceilings[0] < ceilings[1] < ceilings[2]
    assert all(d <= utils.MAX_BACKOFF_SECONDS for d in ceilings)
