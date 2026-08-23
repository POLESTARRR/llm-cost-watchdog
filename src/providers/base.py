"""
The provider interface every LLM backend implements.

The whole point of this layer: a cost watchdog that only watches one vendor
isn't a cost watchdog. Each provider adapter takes a prompt and returns the
same normalized shape, so tracker/analyzer/digest/MCP never learn that
Gemini reports `usage_metadata.prompt_token_count` while Anthropic reports
`usage.input_tokens` and OpenAI reports `usage.input_tokens` with cached
tokens nested under `input_tokens_details`.

To add a provider: implement `Provider`, register it in registry.py, and add
its models to pricing.py. Nothing else changes.
"""

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Protocol


@dataclass
class LLMResponse:
    """Normalized result of a single LLM call, across all providers.

    `input_tokens` is the FULL billable prompt. `cached_input_tokens` and
    `cache_write_tokens` are disjoint subsets of it:
      - cached  = served from cache, billed at a discount (~0.1x)
      - written = newly stored in cache, billed at a premium on some models
    Providers without prompt caching report 0 for both.

    `output_tokens` includes reasoning tokens where the provider emits them
    every provider counts reasoning inside output, so callers must not add
    them separately.
    """

    text: str
    input_tokens: int
    output_tokens: int
    cached_input_tokens: int = 0
    cache_write_tokens: int = 0
    # Tool calls the model wants executed, normalized to OpenAI's shape:
    # [{"id": ..., "type": "function", "function": {"name": ..., "arguments": "<json str>"}}]
    # `arguments` is a JSON *string*, not an object, because that is what the
    # OpenAI wire format specifies and what every client parsing it expects.
    # Vendors that hand back a decoded object get re-encoded in their adapter.
    tool_calls: list[dict] | None = None
    # "stop" | "tool_calls" | "length". Callers must not infer this from whether
    # `tool_calls` is empty: a model can emit text and tool calls together.
    finish_reason: str = "stop"


@dataclass
class StreamChunk:
    """One piece of a streamed response.

    Every chunk but the last carries a `text_delta` and nothing else. The final
    chunk carries `done=True` and the complete `LLMResponse`, because usage
    numbers only exist once the provider has finished, that is a fact about
    every vendor's streaming API, not a shortcut taken here.

    This shape is what lets streaming stay honest in the ledger: the wrapper
    logs when the stream closes, using real token counts, and separately
    records the genuinely-measured time to first token.
    """

    text_delta: str = ""
    done: bool = False
    response: "LLMResponse | None" = None


class Provider(Protocol):
    """What every provider adapter must implement."""

    name: str

    def complete(self, prompt: str, model: str, temperature: float) -> LLMResponse:
        """Make one completion call and return normalized usage."""
        ...

    def is_configured(self) -> bool:
        """True if this provider has the credentials it needs."""
        ...


class ChatProvider(Provider, Protocol):
    """A provider that accepts real message history and tool definitions.

    `complete()` takes a single flattened prompt, which is enough for most
    traffic and keeps the wrapper's API small. It cannot express two things
    that agent workloads depend on: a conversation containing tool results, and
    a request that offers the model tools to call. Flattening those into a
    string loses the structure the provider needs to answer correctly.

    Kept optional and separate so "this vendor cannot do tools here" is
    answerable before a request is made, rather than discovered by a client
    receiving an empty completion.
    """

    def complete_chat(
        self,
        messages: list[dict],
        model: str,
        temperature: float,
        tools: list[dict] | None = None,
    ) -> LLMResponse:
        """Complete against real messages, optionally offering tools."""
        ...


def supports_tools(provider: Provider) -> bool:
    return callable(getattr(provider, "complete_chat", None))


class StreamingProvider(Provider, Protocol):
    """A provider that can also stream. Optional; see `supports_streaming()`.

    Kept as a separate protocol rather than a method on `Provider` returning
    None, so "this vendor cannot stream" is answerable without making a call,
    which is what lets the gateway refuse up front instead of failing mid-response.
    """

    def complete_stream(
        self, prompt: str, model: str, temperature: float
    ) -> Iterator[StreamChunk]:
        """Yield text deltas, then a final chunk carrying full usage."""
        ...


def supports_streaming(provider: Provider) -> bool:
    return callable(getattr(provider, "complete_stream", None))


class ProviderError(RuntimeError):
    """Raised when a provider is unusable (missing key, unknown model, etc.)."""
