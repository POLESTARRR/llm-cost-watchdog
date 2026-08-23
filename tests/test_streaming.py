"""Streaming, and the reason it exists at all: an honest time-to-first-token.

The shortcut this code deliberately refuses is emitting one buffered chunk and
recording the total duration as a TTFT. These tests pin the honest behaviour:
TTFT is measured at the first byte of real content, total latency keeps the
meaning it has on every non-streamed call, and a stream that dies partway is
recorded as a failure rather than a cheap success.
"""

import time

import pytest

from src.providers.base import LLMResponse, ProviderError, StreamChunk, supports_streaming
from src.utils import CallResult, stream_llm


class FakeStreamingProvider:
    name = "test"

    def __init__(self, deltas=("Hel", "lo ", "world"), first_token_delay=0.05, fail_at=None):
        self.deltas, self.delay, self.fail_at = deltas, first_token_delay, fail_at

    def is_configured(self):
        return True

    def complete(self, prompt, model, temperature):
        return LLMResponse(text="".join(self.deltas), input_tokens=10, output_tokens=3)

    def complete_stream(self, prompt, model, temperature):
        time.sleep(self.delay)
        for i, d in enumerate(self.deltas):
            if self.fail_at is not None and i == self.fail_at:
                raise RuntimeError("provider died mid-stream")
            yield StreamChunk(text_delta=d)
        yield StreamChunk(done=True, response=LLMResponse(
            text="".join(self.deltas), input_tokens=10, output_tokens=3))


@pytest.fixture
def streaming(monkeypatch, temp_db):
    monkeypatch.setenv("WATCHDOG_GUARD_MODE", "off")

    def install(provider):
        monkeypatch.setattr("src.utils.get_provider", lambda m: provider)
        return provider

    return install


def _drain(gen):
    """Split a stream_llm generator into (text_deltas, CallResult)."""
    deltas, result = [], None
    for item in gen:
        (deltas.append(item) if isinstance(item, str) else None)
        if isinstance(item, CallResult):
            result = item
    return deltas, result


def test_capability_is_answerable_without_making_a_call():
    """The gateway refuses up front rather than failing mid-response."""
    assert supports_streaming(FakeStreamingProvider())

    class NoStream:
        name = "x"

        def is_configured(self):
            return True

        def complete(self, p, model, temperature):
            return LLMResponse(text="", input_tokens=0, output_tokens=0)

    assert not supports_streaming(NoStream())


def test_deltas_arrive_before_the_result(streaming):
    streaming(FakeStreamingProvider())
    deltas, result = _drain(stream_llm("hi", model="gemini-flash-latest"))
    assert deltas == ["Hel", "lo ", "world"]
    assert result.text == "Hello world"


def test_ttft_is_measured_and_is_less_than_total_latency(streaming):
    """The whole point. A synthetic single chunk would make these equal."""
    streaming(FakeStreamingProvider(first_token_delay=0.05))
    _, result = _drain(stream_llm("hi", model="gemini-flash-latest"))
    event = result.event
    assert event.ttft_ms is not None
    assert event.ttft_ms >= 50          # the injected delay really happened
    assert event.ttft_ms < event.latency_ms


def test_latency_keeps_its_usual_meaning(streaming):
    """Streamed and non-streamed calls must stay comparable on latency_ms."""
    streaming(FakeStreamingProvider())
    _, result = _drain(stream_llm("hi", model="gemini-flash-latest"))
    assert result.event.latency_ms > 0
    assert result.event.success is True


def test_usage_comes_from_the_provider_not_a_guess(streaming):
    streaming(FakeStreamingProvider())
    _, result = _drain(stream_llm("hi", model="gemini-flash-latest"))
    assert (result.event.input_tokens, result.event.output_tokens) == (10, 3)


def test_a_mid_stream_failure_is_logged_as_a_failure(streaming, temp_db):
    """A dead stream must not land in the ledger as a free successful call."""
    from src.tracker import get_events_for_period

    streaming(FakeStreamingProvider(fail_at=1))
    with pytest.raises(RuntimeError, match="died mid-stream"):
        _drain(stream_llm("hi", model="gemini-flash-latest"))

    events = get_events_for_period("all_time")
    assert len(events) == 1
    assert events[0].success is False
    assert "died mid-stream" in events[0].error


def test_a_stream_with_no_usage_block_is_not_recorded_as_free(streaming, temp_db):
    """Inventing zeroes would read as a real call that cost nothing."""
    from src.tracker import get_events_for_period

    class NoTerminal(FakeStreamingProvider):
        def complete_stream(self, prompt, model, temperature):
            yield StreamChunk(text_delta="partial")

    streaming(NoTerminal())
    with pytest.raises(ProviderError, match="without reporting usage"):
        _drain(stream_llm("hi", model="gemini-flash-latest"))

    assert get_events_for_period("all_time")[0].success is False


def test_guardrails_run_before_the_first_byte(streaming, monkeypatch):
    """A stream that has started cannot be un-spent."""
    from src.guard import BudgetExceededError, GuardVerdict

    streaming(FakeStreamingProvider())

    def _block(**kwargs):
        raise BudgetExceededError(
            GuardVerdict(allowed=False, mode="block", triggered=["weekly_budget"], message="over")
        )

    monkeypatch.setattr("src.utils.enforce", _block)
    with pytest.raises(BudgetExceededError):
        _drain(stream_llm("hi", model="gemini-flash-latest"))


def test_streams_do_not_fail_over(streaming, monkeypatch):
    """Re-running the prompt elsewhere would splice two answers into one body."""
    calls = []

    class RateLimited(FakeStreamingProvider):
        def complete_stream(self, prompt, model, temperature):
            calls.append(model)
            raise RuntimeError("429 rate limit exceeded")
            yield  # pragma: no cover

    streaming(RateLimited())
    with pytest.raises(RuntimeError, match="429"):
        _drain(stream_llm("hi", model="gemini-flash-latest"))
    assert len(calls) == 1
