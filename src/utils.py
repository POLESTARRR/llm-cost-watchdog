"""
Shared utilities: env loading, logging, JSON helpers, and the call_llm()
wrapper — the core product of this project.

Every LLM call anywhere in this codebase (including this project's own
digest-writing calls) routes through call_llm(), so cost, tokens, latency,
and cache usage are tracked automatically and identically across providers.
Drop this one function into any project and that project becomes tracked.
"""

import json
import logging
import os
import random
import time
from pathlib import Path

from dotenv import load_dotenv

from src.pricing import DEFAULT_MODEL, calculate_cost
from src.providers import ProviderError, get_provider, infer_provider
from src.tracker import log_usage
from src.usage_schema import UsageEvent

load_dotenv()

WEEKLY_BUDGET_USD = float(os.environ.get("WEEKLY_BUDGET_USD", "5.00"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("llm-cost-watchdog")

# Rate limits are the single most common transient LLM failure — this project
# hit them repeatedly during its own development. Retrying with jittered
# exponential backoff turns a hard failure into a slow success.
MAX_RETRIES = int(os.environ.get("LLM_MAX_RETRIES", "3"))
BASE_BACKOFF_SECONDS = 1.0
MAX_BACKOFF_SECONDS = 30.0


def _is_rate_limit(exc: Exception) -> bool:
    """Detect a rate-limit/overload error without importing every SDK.

    Each provider raises its own class (google's ResourceExhausted,
    anthropic.RateLimitError, openai.RateLimitError), so match on the class
    name and message rather than coupling this module to all three SDKs.
    """
    name = type(exc).__name__.lower()
    if any(k in name for k in ("ratelimit", "resourceexhausted", "overloaded")):
        return True
    text = str(exc).lower()
    return "429" in text or "rate limit" in text or "quota" in text


def _backoff_delay(attempt: int) -> float:
    """Exponential backoff with full jitter, capped."""
    ceiling = min(BASE_BACKOFF_SECONDS * (2**attempt), MAX_BACKOFF_SECONDS)
    return random.uniform(0, ceiling)


def call_llm(
    prompt: str,
    temperature: float = 0.3,
    model: str = DEFAULT_MODEL,
    project_tag: str = "default",
    max_retries: int | None = None,
) -> str:
    """Call any supported LLM, track the call, and return the response text.

    The provider is inferred from the model ID — `claude-*` goes to Anthropic,
    `gpt-*`/`o*` to OpenAI, `gemini-*` to Google — so callers never name a
    provider.

    Every call is logged as a UsageEvent before this function returns or
    raises, including failures. Retries on rate limits are logged as separate
    failed events, so a call that succeeded on its third attempt shows all
    three: the two 429s and the success. That is deliberate — retry volume is
    itself a cost signal.
    """
    retries = MAX_RETRIES if max_retries is None else max_retries
    provider_name = _safe_infer(model)
    provider = get_provider(model)
    prompt_preview = UsageEvent.make_preview(prompt)

    last_exc: Exception | None = None

    for attempt in range(retries + 1):
        start = time.perf_counter()
        try:
            result = provider.complete(prompt, model=model, temperature=temperature)
            latency_ms = (time.perf_counter() - start) * 1000

            cost_usd = calculate_cost(
                model,
                result.input_tokens,
                result.output_tokens,
                result.cached_input_tokens,
                result.cache_write_tokens,
            )

            log_usage(
                UsageEvent(
                    model=model,
                    provider=provider_name,
                    project_tag=project_tag,
                    input_tokens=result.input_tokens,
                    output_tokens=result.output_tokens,
                    cached_input_tokens=result.cached_input_tokens,
                    cache_write_tokens=result.cache_write_tokens,
                    cost_usd=cost_usd,
                    latency_ms=latency_ms,
                    prompt_preview=prompt_preview,
                    success=True,
                )
            )
            logger.info(
                "call_llm ok | %s/%s project=%s cost=$%.6f latency=%.0fms tokens=%d/%d cached=%d",
                provider_name, model, project_tag, cost_usd, latency_ms,
                result.input_tokens, result.output_tokens, result.cached_input_tokens,
            )
            return result.text

        except Exception as exc:
            latency_ms = (time.perf_counter() - start) * 1000
            last_exc = exc

            log_usage(
                UsageEvent(
                    model=model,
                    provider=provider_name,
                    project_tag=project_tag,
                    input_tokens=0,
                    output_tokens=0,
                    cost_usd=0.0,
                    latency_ms=latency_ms,
                    prompt_preview=prompt_preview,
                    success=False,
                    error=str(exc)[:500],
                )
            )

            retryable = _is_rate_limit(exc) and attempt < retries
            if not retryable:
                logger.error(
                    "call_llm failed | %s/%s project=%s latency=%.0fms error=%s",
                    provider_name, model, project_tag, latency_ms, str(exc)[:200],
                )
                raise

            delay = _backoff_delay(attempt)
            logger.warning(
                "call_llm rate-limited | %s/%s attempt %d/%d, retrying in %.1fs",
                provider_name, model, attempt + 1, retries, delay,
            )
            time.sleep(delay)

    raise last_exc  # pragma: no cover - loop always returns or raises


def _safe_infer(model: str) -> str:
    try:
        return infer_provider(model)
    except ProviderError:
        return "unknown"


def read_json(path: str) -> object:
    with open(path) as f:
        return json.load(f)


def write_json(path: str, data: object) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
