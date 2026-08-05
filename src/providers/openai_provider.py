"""OpenAI adapter (official `openai` SDK, Responses API).

Reads usage off the response:
  usage.input_tokens                          — total prompt tokens
  usage.input_tokens_details.cached_tokens    — subset served from cache
  usage.output_tokens

Unlike Anthropic, OpenAI's `input_tokens` already includes the cached
portion, so no summing is needed — we just split out the cached subset for
per-rate billing.
"""

import os

from src.providers.base import LLMResponse, ProviderError

_client = None


class OpenAIProvider:
    name = "openai"

    def is_configured(self) -> bool:
        return bool(os.environ.get("OPENAI_API_KEY"))

    def _get_client(self):
        global _client
        import openai

        if not self.is_configured():
            raise ProviderError("OPENAI_API_KEY is not set")
        if _client is None:
            _client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        return _client

    def complete(self, prompt: str, model: str, temperature: float) -> LLMResponse:
        client = self._get_client()
        kwargs = {"model": model, "input": prompt}
        # Reasoning models (o-series, gpt-5 reasoning tiers) reject temperature.
        if _accepts_temperature(model):
            kwargs["temperature"] = temperature

        response = client.responses.create(**kwargs)

        usage = response.usage
        # These detail objects and their fields are Optional on the SDK models,
        # so `or 0` rather than a bare getattr default.
        details = getattr(usage, "input_tokens_details", None)
        cached = (getattr(details, "cached_tokens", None) or 0) if details else 0
        written = (getattr(details, "cache_write_tokens", None) or 0) if details else 0

        return LLMResponse(
            text=response.output_text,
            # Already includes the cached/written portions.
            input_tokens=usage.input_tokens,
            # Already includes reasoning tokens — do not add them again.
            output_tokens=usage.output_tokens,
            cached_input_tokens=cached,
            cache_write_tokens=written,
        )


def _accepts_temperature(model: str) -> bool:
    """Reasoning models reject sampling parameters."""
    return not model.startswith(("o1", "o3", "o4", "gpt-5"))
