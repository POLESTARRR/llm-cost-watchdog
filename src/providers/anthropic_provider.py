"""Anthropic Claude adapter (official `anthropic` SDK).

Reads usage off the Messages API response:
  usage.input_tokens, uncached input, billed at full rate
  usage.cache_read_input_tokens, served from prompt cache, ~0.1x rate
  usage.cache_creation_input_tokens, written to cache, ~1.25x rate
  usage.output_tokens

Note that `input_tokens` is the *uncached remainder*, not the total prompt
total prompt = input_tokens + cache_read + cache_creation. Getting this wrong
is the classic way to under-report Anthropic spend, so we normalize it here:
`input_tokens` on LLMResponse is the full billable prompt, with the cached
portion carried separately so pricing.py can bill each at its own rate.
"""

import json
import os
from collections.abc import Iterator

from src.providers.base import LLMResponse, ProviderError, StreamChunk

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
        # reject `temperature`. It was removed from the API. Only send it to
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
        # empty content, check before indexing, or this raises IndexError.
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


    def complete_stream(
        self, prompt: str, model: str, temperature: float
    ) -> Iterator[StreamChunk]:
        """Stream via the Messages API.

        Anthropic splits usage across two events: `message_start` carries the
        input counts (including the cache breakdown) and `message_delta` carries
        the final output count. Both must be collected before the final chunk
        can report a complete, correctly-priced usage block, which is why the
        input numbers are stashed rather than emitted as they arrive.
        """
        client = self._get_client()
        kwargs = {
            "model": model,
            "max_tokens": 4096,
            "messages": [{"role": "user", "content": prompt}],
        }
        if _accepts_temperature(model):
            kwargs["temperature"] = temperature

        text_parts: list[str] = []
        in_tokens = cache_read = cache_write = out_tokens = 0

        with client.messages.stream(**kwargs) as stream:
            for event in stream:
                etype = getattr(event, "type", "")
                if etype == "message_start":
                    u = event.message.usage
                    in_tokens = u.input_tokens
                    cache_read = getattr(u, "cache_read_input_tokens", 0) or 0
                    cache_write = getattr(u, "cache_creation_input_tokens", 0) or 0
                elif etype == "content_block_delta":
                    delta = getattr(event.delta, "text", "") or ""
                    if delta:
                        text_parts.append(delta)
                        yield StreamChunk(text_delta=delta)
                elif etype == "message_delta":
                    out_tokens = getattr(event.usage, "output_tokens", 0) or 0

        yield StreamChunk(done=True, response=LLMResponse(
            text="".join(text_parts),
            # Full billable prompt, matching complete()'s normalization.
            input_tokens=in_tokens + cache_read + cache_write,
            output_tokens=out_tokens,
            cached_input_tokens=cache_read,
            cache_write_tokens=cache_write,
        ))


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


def _to_anthropic_tools(tools: list[dict]) -> list[dict]:
    """OpenAI tool definitions -> Anthropic's shape.

    The two differ only in nesting and one key name, but the difference is not
    optional: Anthropic rejects an OpenAI-shaped tool outright.
    """
    out = []
    for tool in tools or []:
        fn = tool.get("function", tool)
        out.append({
            "name": fn.get("name", ""),
            "description": fn.get("description", ""),
            "input_schema": fn.get("parameters") or {"type": "object", "properties": {}},
        })
    return out


def _to_anthropic_messages(messages: list[dict]) -> tuple[str | None, list[dict]]:
    """Split out the system prompt and translate tool turns.

    Anthropic takes `system` as a top-level parameter rather than a message,
    and represents tool results as a `tool_result` content block inside a USER
    message, where OpenAI uses a dedicated `tool` role. Getting this wrong
    produces a 400 rather than a subtly worse answer, which is at least loud.
    """
    system: str | None = None
    out: list[dict] = []

    for msg in messages:
        role, content = msg.get("role"), msg.get("content")

        if role == "system":
            system = content if system is None else f"{system}\n\n{content}"
        elif role == "tool":
            out.append({"role": "user", "content": [{
                "type": "tool_result",
                "tool_use_id": msg.get("tool_call_id", ""),
                "content": content or "",
            }]})
        elif role == "assistant" and msg.get("tool_calls"):
            blocks = []
            if content:
                blocks.append({"type": "text", "text": content})
            for call in msg["tool_calls"]:
                fn = call.get("function", {})
                args = fn.get("arguments") or "{}"
                blocks.append({
                    "type": "tool_use",
                    "id": call.get("id", ""),
                    "name": fn.get("name", ""),
                    "input": json.loads(args) if isinstance(args, str) else args,
                })
            out.append({"role": "assistant", "content": blocks})
        else:
            out.append({"role": role, "content": content or ""})

    return system, out


class _AnthropicChatMixin:
    def complete_chat(self, messages, model, temperature, tools=None):
        client = self._get_client()
        system, converted = _to_anthropic_messages(messages)

        kwargs = {"model": model, "max_tokens": 4096, "messages": converted}
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = _to_anthropic_tools(tools)
        if _accepts_temperature(model):
            kwargs["temperature"] = temperature

        response = client.messages.create(**kwargs)

        text = "".join(b.text for b in response.content if b.type == "text")
        tool_calls = [
            {
                "id": b.id,
                "type": "function",
                # Anthropic hands back a decoded object; the OpenAI wire format
                # this gateway speaks specifies a JSON string.
                "function": {"name": b.name, "arguments": json.dumps(b.input)},
            }
            for b in response.content if b.type == "tool_use"
        ] or None

        usage = response.usage
        cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
        cache_write = getattr(usage, "cache_creation_input_tokens", 0) or 0

        return LLMResponse(
            text=text,
            input_tokens=usage.input_tokens + cache_read + cache_write,
            output_tokens=usage.output_tokens,
            cached_input_tokens=cache_read,
            cache_write_tokens=cache_write,
            tool_calls=tool_calls,
            finish_reason="tool_calls" if tool_calls else "stop",
        )


AnthropicProvider.complete_chat = _AnthropicChatMixin.complete_chat
