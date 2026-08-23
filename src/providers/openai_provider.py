"""OpenAI adapter (official `openai` SDK, Responses API).

Reads usage off the response:
  usage.input_tokens, total prompt tokens
  usage.input_tokens_details.cached_tokens, subset served from cache
  usage.output_tokens

Unlike Anthropic, OpenAI's `input_tokens` already includes the cached
portion, so no summing is needed. We just split out the cached subset for
per-rate billing.
"""

import os
from collections.abc import Iterator

from src.providers.base import LLMResponse, ProviderError, StreamChunk

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
            # Already includes reasoning tokens, do not add them again.
            output_tokens=usage.output_tokens,
            cached_input_tokens=cached,
            cache_write_tokens=written,
        )


    def complete_stream(
        self, prompt: str, model: str, temperature: float
    ) -> Iterator[StreamChunk]:
        """Stream via the Responses API.

        Usage arrives only on the terminal `response.completed` event, so the
        deltas are forwarded as they come and the counts are read at the end.
        """
        client = self._get_client()
        kwargs = {"model": model, "input": prompt, "stream": True}
        if _accepts_temperature(model):
            kwargs["temperature"] = temperature

        text_parts: list[str] = []
        usage = None

        for event in client.responses.create(**kwargs):
            etype = getattr(event, "type", "")
            if etype == "response.output_text.delta":
                delta = getattr(event, "delta", "") or ""
                if delta:
                    text_parts.append(delta)
                    yield StreamChunk(text_delta=delta)
            elif etype == "response.completed":
                usage = getattr(event.response, "usage", None)

        details = getattr(usage, "input_tokens_details", None) if usage else None
        cached = (getattr(details, "cached_tokens", None) or 0) if details else 0
        written = (getattr(details, "cache_write_tokens", None) or 0) if details else 0

        yield StreamChunk(done=True, response=LLMResponse(
            text="".join(text_parts),
            input_tokens=getattr(usage, "input_tokens", 0) or 0 if usage else 0,
            output_tokens=getattr(usage, "output_tokens", 0) or 0 if usage else 0,
            cached_input_tokens=cached,
            cache_write_tokens=written,
        ))


def _accepts_temperature(model: str) -> bool:
    """Reasoning models reject sampling parameters."""
    return not model.startswith(("o1", "o3", "o4", "gpt-5"))


class _OpenAIChatMixin:
    """Chat + tools via the Chat Completions API.

    `complete()` uses the Responses API, which is the newer surface and the
    right default for plain text. Tool calling goes through Chat Completions
    instead because that is the API whose request and response shapes the
    gateway already speaks, OpenAI's own format is this project's wire format,
    so no translation is needed in either direction.
    """

    def complete_chat(self, messages, model, temperature, tools=None):
        client = self._get_client()
        kwargs = {"model": model, "messages": messages}
        if tools:
            kwargs["tools"] = tools
        if _accepts_temperature(model):
            kwargs["temperature"] = temperature

        response = client.chat.completions.create(**kwargs)
        choice = response.choices[0]
        message = choice.message

        tool_calls = None
        if message.tool_calls:
            tool_calls = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in message.tool_calls
            ]

        usage = response.usage
        details = getattr(usage, "prompt_tokens_details", None)
        cached = (getattr(details, "cached_tokens", None) or 0) if details else 0

        return LLMResponse(
            text=message.content or "",
            input_tokens=usage.prompt_tokens,
            output_tokens=usage.completion_tokens,
            cached_input_tokens=cached,
            tool_calls=tool_calls,
            finish_reason=choice.finish_reason or "stop",
        )


OpenAIProvider.complete_chat = _OpenAIChatMixin.complete_chat
