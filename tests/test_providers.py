"""Provider routing, credential detection, and usage normalization.

The normalization tests are the important ones: each vendor reports token
usage in a different shape, and getting Anthropic's in particular wrong is
the classic way to silently under-report spend.
"""

import pytest

from src.providers import ProviderError, get_provider, infer_provider
from src.providers.anthropic_provider import AnthropicProvider, _accepts_temperature
from src.providers.base import LLMResponse
from src.providers.gemini import GeminiProvider
from src.providers.openai_provider import OpenAIProvider


# --- routing -------------------------------------------------------------


@pytest.mark.parametrize(
    "model,expected",
    [
        ("claude-opus-5", "anthropic"),
        ("claude-sonnet-5", "anthropic"),
        ("claude-haiku-4-5", "anthropic"),
        ("gpt-5.6-sol", "openai"),
        ("gpt-5-nano", "openai"),
        ("o3", "openai"),
        ("o4-mini", "openai"),
        ("gemini-flash-latest", "google"),
        ("gemma-4-31b-it", "google"),
    ],
)
def test_infer_provider_routes_by_prefix(model, expected):
    assert infer_provider(model) == expected


def test_unknown_model_prefix_raises_with_a_useful_message():
    with pytest.raises(ProviderError) as exc:
        infer_provider("llama-4-70b")
    assert "llama-4-70b" in str(exc.value)


def test_get_provider_returns_the_matching_adapter():
    assert isinstance(get_provider("claude-opus-5"), AnthropicProvider)
    assert isinstance(get_provider("gpt-5.6-sol"), OpenAIProvider)
    assert isinstance(get_provider("gemini-flash-latest"), GeminiProvider)


# --- credential detection ------------------------------------------------


def test_is_configured_follows_the_env_var(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert AnthropicProvider().is_configured() is False
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    assert AnthropicProvider().is_configured() is True


def test_complete_raises_when_unconfigured(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ProviderError):
        OpenAIProvider().complete("hi", model="gpt-5-nano", temperature=0.3)


# --- usage normalization -------------------------------------------------


class _Obj:
    """Minimal stand-in for an SDK response object (attribute access)."""

    def __init__(self, **kw):
        self.__dict__.update(kw)


def test_anthropic_input_tokens_include_the_cached_portion(monkeypatch):
    """Anthropic's usage.input_tokens is the UNCACHED remainder. If we don't
    add cache_read + cache_creation back in, the prompt is under-counted."""
    fake = _Obj(
        stop_reason="end_turn",
        content=[_Obj(type="text", text="hello")],
        usage=_Obj(
            input_tokens=200,               # uncached remainder only
            cache_read_input_tokens=800,
            cache_creation_input_tokens=100,
            output_tokens=50,
        ),
    )
    client = _Obj(messages=_Obj(create=lambda **kw: fake))
    p = AnthropicProvider()
    monkeypatch.setattr(p, "_get_client", lambda: client)

    result = p.complete("hi", model="claude-opus-5", temperature=0.3)
    assert result.input_tokens == 1100          # 200 + 800 + 100, not 200
    assert result.cached_input_tokens == 800
    assert result.cache_write_tokens == 100
    assert result.output_tokens == 50
    assert result.text == "hello"


def test_anthropic_refusal_yields_empty_text_not_a_crash(monkeypatch):
    """A refusal returns HTTP 200 with possibly-empty content; indexing
    content[0] unconditionally would raise."""
    fake = _Obj(
        stop_reason="refusal",
        content=[],
        usage=_Obj(input_tokens=10, cache_read_input_tokens=0,
                   cache_creation_input_tokens=0, output_tokens=0),
    )
    client = _Obj(messages=_Obj(create=lambda **kw: fake))
    p = AnthropicProvider()
    monkeypatch.setattr(p, "_get_client", lambda: client)

    result = p.complete("hi", model="claude-opus-5", temperature=0.3)
    assert result.text == ""
    assert result.input_tokens == 10


def test_anthropic_omits_temperature_on_models_that_reject_it():
    """Sampling params were removed on Opus 4.7+/5, Sonnet 5, Fable 5 — sending
    them returns a 400."""
    assert _accepts_temperature("claude-opus-5") is False
    assert _accepts_temperature("claude-sonnet-5") is False
    assert _accepts_temperature("claude-fable-5") is False
    assert _accepts_temperature("claude-haiku-4-5") is True


def test_anthropic_does_not_send_temperature_for_opus5(monkeypatch):
    seen = {}

    def _create(**kw):
        seen.update(kw)
        return _Obj(
            stop_reason="end_turn",
            content=[_Obj(type="text", text="ok")],
            usage=_Obj(input_tokens=1, cache_read_input_tokens=0,
                       cache_creation_input_tokens=0, output_tokens=1),
        )

    p = AnthropicProvider()
    monkeypatch.setattr(p, "_get_client", lambda: _Obj(messages=_Obj(create=_create)))
    p.complete("hi", model="claude-opus-5", temperature=0.7)
    assert "temperature" not in seen


def test_openai_input_tokens_already_include_cached(monkeypatch):
    """Unlike Anthropic, OpenAI's input_tokens is the full prompt — summing
    the cached subset back in would double-count it."""
    fake = _Obj(
        output_text="hi there",
        usage=_Obj(
            input_tokens=1000,   # already includes the 700 cached
            input_tokens_details=_Obj(cached_tokens=700, cache_write_tokens=0),
            output_tokens=120,
        ),
    )
    p = OpenAIProvider()
    monkeypatch.setattr(p, "_get_client", lambda: _Obj(responses=_Obj(create=lambda **kw: fake)))

    result = p.complete("hi", model="gpt-5.6-sol", temperature=0.3)
    assert result.input_tokens == 1000
    assert result.cached_input_tokens == 700


def test_openai_handles_missing_usage_detail_fields(monkeypatch):
    """The detail objects and their fields are Optional on the SDK models."""
    fake = _Obj(
        output_text="hi",
        usage=_Obj(input_tokens=50, input_tokens_details=None, output_tokens=10),
    )
    p = OpenAIProvider()
    monkeypatch.setattr(p, "_get_client", lambda: _Obj(responses=_Obj(create=lambda **kw: fake)))

    result = p.complete("hi", model="gpt-5-nano", temperature=0.3)
    assert result.cached_input_tokens == 0
    assert result.cache_write_tokens == 0


def test_gemini_normalizes_usage_metadata(monkeypatch):
    fake = _Obj(
        text="hello",
        usage_metadata=_Obj(prompt_token_count=300, candidates_token_count=90),
    )
    fake_model = _Obj(generate_content=lambda *a, **kw: fake)
    fake_genai = _Obj(GenerativeModel=lambda m: fake_model)

    p = GeminiProvider()
    monkeypatch.setattr(p, "_ensure_configured", lambda: fake_genai)

    result = p.complete("hi", model="gemini-flash-latest", temperature=0.3)
    assert (result.input_tokens, result.output_tokens) == (300, 90)
    assert result.cached_input_tokens == 0


def test_llm_response_defaults_cache_fields_to_zero():
    r = LLMResponse(text="x", input_tokens=10, output_tokens=5)
    assert r.cached_input_tokens == 0
    assert r.cache_write_tokens == 0
