"""Anthropic Claude adapter (official `anthropic` SDK).

Reads usage off the Messages API response:
  usage.input_tokens               — uncached input, billed at full rate
  usage.cache_read_input_tokens    — served from prompt cache, ~0.1x rate
  usage.cache_creation_input_tokens— written to cache, ~1.25x rate
  usage.output_tokens

Note that `input_tokens` is the *uncached remainder*, not the total prompt —
total prompt = input_tokens + cache_read + cache_creation. Getting this wrong
is the classic way to under-report Anthropic spend, so we normalize it here:
`input_tokens` on LLMResponse is the full billable prompt, with the cached
portion carried separately so pricing.py can bill each at its own rate.
"""

import os

from src.providers.base import LLMResponse, ProviderError

_client = None


class AnthropicProvider:
    name = "anthropic"

    def is_configured(self) -> bool:
        return bool(os.environ.get("ANTHROPIC_API_KEY"))

    def _get_client(self):
        global _client
        import anthropic

        if not self.is_configured():
            raise ProviderError("ANTHROPIC_API_KEY is not set")
        if _client is None:
            _client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        return _client

    def complete(self, prompt: str, model: str, temperature: float) -> LLMResponse:
        client = self._get_client()
        # Current Claude models (Opus 5 / Sonnet 5 / Fable 5, and the 4.7+ family)
        # reject `temperature` — it was removed from the API. Only send it to
        # models that still accept it.
        kwargs = {
            "model": model,
            "max_tokens": 4096,
            "messages": [{"role": "user", "content": prompt}],
        }
        if _accepts_temperature(model):
            kwargs["temperature"] = temperature

        response = client.messages.create(**kwargs)

        # A refusal returns HTTP 200 with stop_reason="refusal" and possibly
        # empty content — check before indexing, or this raises IndexError.
        if response.stop_reason == "refusal":
            text = ""
        else:
            text = "".join(b.text for b in response.content if b.type == "text")

        usage = response.usage
        cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
        cache_write = getattr(usage, "cache_creation_input_tokens", 0) or 0

        return LLMResponse(
            text=text,
            # Full billable prompt, not just the uncached remainder.
            input_tokens=usage.input_tokens + cache_read + cache_write,
            output_tokens=usage.output_tokens,
            cached_input_tokens=cache_read,
            cache_write_tokens=cache_write,
        )


def _accepts_temperature(model: str) -> bool:
    """Sampling params were removed on Opus 4.7+, Opus 5, Sonnet 5, and Fable 5."""
    removed_prefixes = (
        "claude-opus-5",
        "claude-opus-4-7",
        "claude-opus-4-8",
        "claude-sonnet-5",
        "claude-fable-5",
        "claude-mythos-5",
    )
    return not model.startswith(removed_prefixes)
