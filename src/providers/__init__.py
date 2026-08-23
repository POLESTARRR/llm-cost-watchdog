"""Provider registry: model ID -> provider adapter.

`call_llm(prompt, model="claude-opus-5")` should just work without the caller
naming a provider, so the model ID prefix does the routing. This is the one
place that knows which vendor owns which naming scheme.
"""

from dotenv import load_dotenv

from src.providers.anthropic_provider import AnthropicProvider
from src.providers.base import (
    LLMResponse,
    Provider,
    ProviderError,
    StreamChunk,
    supports_streaming,
    supports_tools,
)
from src.providers.gemini import GeminiProvider
from src.providers.ollama import OllamaProvider
from src.providers.openai_provider import OpenAIProvider

# Load .env here too, not just in utils, provider credential checks must give
# the same answer regardless of which module the caller imported first.
load_dotenv()

_PROVIDERS: dict[str, Provider] = {
    "google": GeminiProvider(),
    "anthropic": AnthropicProvider(),
    "openai": OpenAIProvider(),
    "ollama": OllamaProvider(),
}

# Ordered longest-prefix-first so "gpt-oss" can't shadow "gpt-" style entries.
#
# `ollama/` is first and is the only *namespaced* prefix here, deliberately.
# Local servers host models whose bare names already belong to hosted vendors
# (Ollama serves `gemma`, so does Google), so local ids are explicitly
# qualified rather than guessed at. `ollama/gemma3` and `gemma-3` then route
# to different providers without either rule needing to know about the other.
_PREFIX_ROUTES: list[tuple[str, str]] = [
    ("ollama/", "ollama"),
    ("gemini", "google"),
    ("gemma", "google"),
    ("claude", "anthropic"),
    ("gpt", "openai"),
    ("o1", "openai"),
    ("o3", "openai"),
    ("o4", "openai"),
]


def infer_provider(model: str) -> str:
    """Return the provider name that owns `model`, by ID prefix."""
    for prefix, provider in _PREFIX_ROUTES:
        if model.startswith(prefix):
            return provider
    raise ProviderError(
        f"Cannot infer provider for model {model!r}. "
        f"Known prefixes: {sorted({p for p, _ in _PREFIX_ROUTES})}"
    )


def get_provider(model: str) -> Provider:
    """Return the adapter instance for `model`."""
    return _PROVIDERS[infer_provider(model)]


def configured_providers() -> dict[str, bool]:
    """Which providers currently have credentials available."""
    return {name: p.is_configured() for name, p in _PROVIDERS.items()}


__all__ = [
    "LLMResponse",
    "Provider",
    "ProviderError",
    "StreamChunk",
    "supports_streaming",
    "supports_tools",
    "configured_providers",
    "get_provider",
    "infer_provider",
]
