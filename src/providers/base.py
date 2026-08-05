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

    `output_tokens` includes reasoning tokens where the provider emits them —
    every provider counts reasoning inside output, so callers must not add
    them separately.
    """

    text: str
    input_tokens: int
    output_tokens: int
    cached_input_tokens: int = 0
    cache_write_tokens: int = 0


class Provider(Protocol):
    """What every provider adapter must implement."""

    name: str

    def complete(self, prompt: str, model: str, temperature: float) -> LLMResponse:
        """Make one completion call and return normalized usage."""
        ...

    def is_configured(self) -> bool:
        """True if this provider has the credentials it needs."""
        ...


class ProviderError(RuntimeError):
    """Raised when a provider is unusable (missing key, unknown model, etc.)."""
