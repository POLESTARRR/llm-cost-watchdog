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


@pytest.fixture(autouse=True)
def _no_fallback(monkeypatch):
    """Isolate retry behavior from cross-provider fallback. Tests that want
    fallback opt back in via the `fallback_on` fixture."""
    monkeypatch.setenv("WATCHDOG_FALLBACK", "off")


@pytest.fixture
def fallback_on(monkeypatch):
    monkeypatch.setenv("WATCHDOG_FALLBACK", "on")


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


# --- cross-provider fallback --------------------------------------------


def test_falls_over_to_another_provider_when_rate_limited(
    temp_db, monkeypatch, fallback_on
):
    """A 429 on one provider should not fail the call when another provider
    has credentials. That is the whole point of tracking three of them."""
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    primary = _FakeProvider(fail_times=99, fail_with=RateLimitError("429"))
    backup = _FakeProvider(response=LLMResponse(text="from backup", input_tokens=10, output_tokens=5))

    def route(model):
        return primary if model.startswith("claude") else backup

    monkeypatch.setattr(utils, "get_provider", route)

    assert utils.call_llm("hi", model="claude-opus-5", max_retries=0) == "from backup"
    assert backup.calls == 1


def test_fallback_records_both_the_failure_and_the_success(
    temp_db, monkeypatch, fallback_on
):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    primary = _FakeProvider(fail_times=99, fail_with=RateLimitError("429"))
    backup = _FakeProvider()
    monkeypatch.setattr(
        utils, "get_provider", lambda m: primary if m.startswith("claude") else backup
    )

    utils.call_llm("hi", model="claude-opus-5", max_retries=0)

    events = get_events(db_path=temp_db)
    assert [e.success for e in events] == [False, True]
    assert events[0].provider == "anthropic"    # the one that was exhausted
    assert events[1].provider == "google"       # the one that answered


def test_no_fallback_on_non_rate_limit_errors(temp_db, monkeypatch, fallback_on):
    """A 400 fails the same way everywhere, retrying elsewhere just burns
    another call."""
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    primary = _FakeProvider(fail_times=99, fail_with=ValueError("bad request"))
    backup = _FakeProvider()
    monkeypatch.setattr(
        utils, "get_provider", lambda m: primary if m.startswith("claude") else backup
    )

    with pytest.raises(ValueError):
        utils.call_llm("hi", model="claude-opus-5", max_retries=0)
    assert backup.calls == 0


def test_fallback_can_be_disabled(temp_db, monkeypatch):
    monkeypatch.setenv("WATCHDOG_FALLBACK", "off")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    primary = _FakeProvider(fail_times=99, fail_with=RateLimitError("429"))
    backup = _FakeProvider()
    monkeypatch.setattr(
        utils, "get_provider", lambda m: primary if m.startswith("claude") else backup
    )

    with pytest.raises(RateLimitError):
        utils.call_llm("hi", model="claude-opus-5", max_retries=0)
    assert backup.calls == 0


def test_fallback_candidates_exclude_the_failing_provider(monkeypatch, fallback_on):
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    candidates = utils._fallback_candidates(exclude_provider="anthropic")
    assert "claude-haiku-4-5" not in candidates      # same provider that failed
    assert "gemini-flash-lite-latest" in candidates  # configured, different provider
    assert "gpt-5-nano" not in candidates            # no credentials
