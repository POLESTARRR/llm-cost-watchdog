"""
Shared utilities: env loading, logging, JSON helpers, and the call_llm()
wrapper, the core product of this project.

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
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from src.guard import enforce
from src.pricing import DEFAULT_MODEL, calculate_cost
from src.providers import (
    LLMResponse,
    ProviderError,
    configured_providers,
    get_provider,
    infer_provider,
)
from src.tracker import log_usage
from src.usage_schema import UsageEvent

load_dotenv()

WEEKLY_BUDGET_USD = float(os.environ.get("WEEKLY_BUDGET_USD", "5.00"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("llm-cost-gateway")

# Rate limits are the single most common transient LLM failure. This project
# hit them repeatedly during its own development. Retrying with jittered
# exponential backoff turns a hard failure into a slow success.
MAX_RETRIES = int(os.environ.get("LLM_MAX_RETRIES", "3"))
BASE_BACKOFF_SECONDS = 1.0
MAX_BACKOFF_SECONDS = 30.0

# When the requested provider is rate-limited and you have credentials for
# another one, failing the call outright is a choice, and the wrong one.
# These are the cheap, broadly-available substitutes to fall back to, in
# order. Set WATCHDOG_FALLBACK=off to disable.
FALLBACK_MODELS = [
    "gemini-flash-lite-latest",   # google
    "claude-haiku-4-5",           # anthropic
    "gpt-5-nano",                 # openai
]


def _fallback_enabled() -> bool:
    return os.environ.get("WATCHDOG_FALLBACK", "on").strip().lower() != "off"


def _fallback_candidates(exclude_provider: str) -> list[str]:
    """Configured models from providers other than the one that just failed."""
    if not _fallback_enabled():
        return []
    configured = configured_providers()
    out = []
    for model in FALLBACK_MODELS:
        try:
            provider = infer_provider(model)
        except ProviderError:
            continue
        if provider != exclude_provider and configured.get(provider):
            out.append(model)
    return out


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


@dataclass
class CallResult:
    """A completed call: the text, plus everything the ledger recorded about it.

    `call_llm()` returns only the text, which is the right default for the
    common case and is what every existing caller expects. The gateway needs
    the token counts and cost to fill in an OpenAI-shaped `usage` block, and
    re-reading the row it just wrote would be both racy and absurd, so the
    detailed variant hands the event back directly.
    """

    text: str
    model: str
    event: UsageEvent
    routing: dict | None = None
    # Populated only by the chat/tools path. `text` can be empty while these
    # are set: a model that decides to call a tool has nothing to say yet.
    tool_calls: list[dict] | None = None
    finish_reason: str = "stop"


def call_llm(
    prompt: str,
    temperature: float = 0.3,
    model: str = DEFAULT_MODEL,
    project_tag: str = "default",
    max_retries: int | None = None,
    skip_guards: bool = False,
    model_group: str | None = None,
) -> str:
    """Call any supported LLM, track the call, and return the response text.

    Thin wrapper over `call_llm_detailed()`; see it for the full behaviour.
    """
    return call_llm_detailed(
        prompt, temperature, model, project_tag, max_retries, skip_guards, model_group
    ).text


def call_llm_detailed(
    prompt: str,
    temperature: float = 0.3,
    model: str = DEFAULT_MODEL,
    project_tag: str = "default",
    max_retries: int | None = None,
    skip_guards: bool = False,
    model_group: str | None = None,
) -> CallResult:
    """Call any supported LLM, track the call, and return text plus usage.

    The provider is inferred from the model ID, `claude-*` goes to Anthropic,
    `gpt-*`/`o*` to OpenAI, `gemini-*` to Google, so callers never name a
    provider.

    Pass `model_group` instead of `model` to let src/router.py choose from a
    declared group using recorded history (cost, latency, failure rate),
    skipping any member that is cooling down after a rate limit. The chosen
    model is logged, so a routing decision can be audited against the same
    ledger that informed it.

    Every call is logged as a UsageEvent before this function returns or
    raises, including failures. Retries on rate limits are logged as separate
    failed events, so a call that succeeded on its third attempt shows all
    three: the two 429s and the success. That is deliberate, retry volume is
    itself a cost signal.
    """
    retries = MAX_RETRIES if max_retries is None else max_retries

    group_members: list[str] = []
    routing: dict | None = None
    if model_group:
        from src.router import RoutingError, select

        try:
            # The prompt is passed so `strategy=complexity` can read it. Every
            # other strategy ignores it; none of them retain it.
            decision = select(
                model_group, estimated_input_tokens=len(prompt) // 4, prompt=prompt
            )
        except RoutingError:
            logger.error("call_llm: routing failed for group %r", model_group)
            raise
        model = decision.model
        group_members = decision.candidates
        routing = decision.as_dict()
        logger.info(
            "call_llm routed | group=%s -> %s via %s (%s)",
            model_group, model, decision.strategy, decision.basis,
        )

    # Pre-flight guardrails. In `block` mode this raises BudgetExceededError
    # before any request goes out, the one place this project stops spend
    # rather than reporting it. `skip_guards` lets the digest still report on
    # a session that has already tripped a limit.
    if not skip_guards:
        enforce(
            project_tag=project_tag,
            model=model,
            # ~4 chars/token is rough, but it only needs to catch a call that
            # is orders of magnitude too big, not price it precisely.
            estimated_input_tokens=len(prompt) // 4,
        )

    try:
        return _attempt_model(prompt, temperature, model, project_tag, retries, routing)
    except Exception as exc:
        # Only a rate limit / exhausted quota is worth failing over. A 400 or
        # an auth error will fail identically on every provider, so retrying
        # elsewhere just burns another call.
        if not _is_rate_limit(exc):
            raise

        # Bench the exhausted model so the *next* call skips it instead of
        # rediscovering the same 429. Cheap insurance, and the one piece of
        # state that makes repeated routing better than repeated guessing.
        _bench(model)

        # Within a group, the rest of the group is the natural fallback set,
        # the caller already declared those models interchangeable. Outside
        # one, fall back to the cross-provider defaults.
        if group_members:
            candidates = [m for m in group_members if m != model]
        else:
            candidates = _fallback_candidates(exclude_provider=_safe_infer(model))
        if not candidates:
            raise

        for alt in candidates:
            logger.warning(
                "call_llm falling back | %s exhausted, trying %s instead", model, alt
            )
            # Record the substitution in the decision trail. Without this the
            # caller sees a response from a model the routing record never
            # names, which reads as a routing bug and is in fact failover
            # working. A decision record that omits the fallback is worse than
            # no record: it is a confident wrong answer about what happened.
            alt_routing = dict(routing) if routing else {}
            alt_routing["fell_back_from"] = model
            alt_routing["fell_back_reason"] = "rate limit / quota exhausted"
            try:
                return _attempt_model(prompt, temperature, alt, project_tag, 0, alt_routing)
            except Exception as alt_exc:
                if not _is_rate_limit(alt_exc):
                    raise
                _bench(alt)
                continue

        logger.error("call_llm: every provider exhausted (tried %s)", [model, *candidates])
        raise


def _attempt_model(
    prompt: str,
    temperature: float,
    model: str,
    project_tag: str,
    retries: int,
    routing: dict | None = None,
) -> CallResult:
    """Try one model, retrying that model on rate limits. Logs every attempt."""
    provider_name = _safe_infer(model)
    provider = get_provider(model)
    prompt_preview = UsageEvent.make_preview(prompt)
    prompt_hash = UsageEvent.make_hash(prompt)
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

            event = UsageEvent(
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
                    prompt_hash=prompt_hash,
                    success=True,
            )
            log_usage(event)
            logger.info(
                "call_llm ok | %s/%s project=%s cost=$%.6f latency=%.0fms tokens=%d/%d cached=%d",
                provider_name, model, project_tag, cost_usd, latency_ms,
                result.input_tokens, result.output_tokens, result.cached_input_tokens,
            )
            return CallResult(text=result.text, model=model, event=event, routing=routing)

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
                    prompt_hash=prompt_hash,
                    success=False,
                    error=str(exc)[:500],
                )
            )

            if not (_is_rate_limit(exc) and attempt < retries):
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


def _bench(model: str) -> None:
    """Put a rate-limited model on cooldown, best-effort.

    Never allowed to fail the call: cooldown is an optimisation for the next
    request, and losing a response because the bookkeeping raised would be a
    strictly worse outcome than not benching.
    """
    try:
        from src.router import start_cooldown

        start_cooldown(model, reason="rate limit")
    except Exception:  # pragma: no cover - defensive
        logger.debug("could not record cooldown for %s", model, exc_info=True)


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


def stream_llm(
    prompt: str,
    temperature: float = 0.3,
    model: str = DEFAULT_MODEL,
    project_tag: str = "default",
    skip_guards: bool = False,
    model_group: str | None = None,
):
    """Stream a completion, yielding text deltas, then log the call.

    Yields `str` deltas as they arrive and, as its final value, a `CallResult`
    carrying the complete text and the logged `UsageEvent`. Callers that only
    want text can ignore the last item by type.

    The reason this exists rather than the gateway faking a stream: streaming
    is the one path where **time to first token is a real, measurable number**,
    and the ledger should hold the real one. A single synthetic chunk emitted
    after a blocking call would record a TTFT equal to the total duration,
    which is not a measurement of anything.

    `latency_ms` keeps its usual meaning, total time to completion, so streamed
    and non-streamed calls stay comparable. `ttft_ms` is added alongside it.

    Guardrails run before the first byte, exactly as in the blocking path: a
    budget block must prevent a call, and a stream that has already started
    cannot be un-spent.

    Streams do NOT fail over. Once bytes have reached the client, silently
    re-running the prompt on another model would emit two different answers
    into one response body. A rate limit mid-stream is raised, not papered over.
    """
    from src.providers import supports_streaming

    group_members: list[str] = []
    routing: dict | None = None
    if model_group:
        from src.router import select

        decision = select(
            model_group, estimated_input_tokens=len(prompt) // 4, prompt=prompt
        )
        model = decision.model
        group_members = decision.candidates
        routing = decision.as_dict()

    provider_name = _safe_infer(model)
    provider = get_provider(model)
    if not supports_streaming(provider):
        raise ProviderError(
            f"provider {provider_name!r} has no streaming implementation for {model!r}"
        )

    if not skip_guards:
        enforce(
            project_tag=project_tag,
            model=model,
            estimated_input_tokens=len(prompt) // 4,
        )

    prompt_preview = UsageEvent.make_preview(prompt)
    prompt_hash = UsageEvent.make_hash(prompt)
    start = time.perf_counter()
    ttft_ms: float | None = None
    final: LLMResponse | None = None

    try:
        for chunk in provider.complete_stream(prompt, model=model, temperature=temperature):
            if chunk.text_delta and ttft_ms is None:
                # Measured once, at the first byte of actual content. Chunks
                # carrying only metadata must not count as "first token".
                ttft_ms = (time.perf_counter() - start) * 1000
            if chunk.done:
                final = chunk.response
                break
            if chunk.text_delta:
                yield chunk.text_delta
    except Exception as exc:
        latency_ms = (time.perf_counter() - start) * 1000
        log_usage(
            UsageEvent(
                model=model, provider=provider_name, project_tag=project_tag,
                input_tokens=0, output_tokens=0, cost_usd=0.0,
                latency_ms=latency_ms, ttft_ms=ttft_ms,
                prompt_preview=prompt_preview, prompt_hash=prompt_hash,
                success=False, error=str(exc)[:500],
            )
        )
        logger.error("stream_llm failed | %s/%s: %s", provider_name, model, str(exc)[:200])
        raise

    latency_ms = (time.perf_counter() - start) * 1000

    if final is None:
        # The provider ended without a terminal chunk, so there are no usage
        # numbers. Log the failure rather than inventing zeroes that would read
        # as a free call.
        log_usage(
            UsageEvent(
                model=model, provider=provider_name, project_tag=project_tag,
                input_tokens=0, output_tokens=0, cost_usd=0.0,
                latency_ms=latency_ms, ttft_ms=ttft_ms,
                prompt_preview=prompt_preview, prompt_hash=prompt_hash,
                success=False, error="stream ended without a usage block",
            )
        )
        raise ProviderError(f"{provider_name} stream ended without reporting usage")

    cost_usd = calculate_cost(
        model, final.input_tokens, final.output_tokens,
        final.cached_input_tokens, final.cache_write_tokens,
    )
    event = UsageEvent(
        model=model, provider=provider_name, project_tag=project_tag,
        input_tokens=final.input_tokens, output_tokens=final.output_tokens,
        cached_input_tokens=final.cached_input_tokens,
        cache_write_tokens=final.cache_write_tokens,
        cost_usd=cost_usd, latency_ms=latency_ms, ttft_ms=ttft_ms,
        prompt_preview=prompt_preview, prompt_hash=prompt_hash, success=True,
    )
    log_usage(event)
    logger.info(
        "stream_llm ok | %s/%s project=%s cost=$%.6f ttft=%sms total=%.0fms tokens=%d/%d",
        provider_name, model, project_tag, cost_usd,
        f"{ttft_ms:.0f}" if ttft_ms is not None else "n/a",
        latency_ms, final.input_tokens, final.output_tokens,
    )

    yield CallResult(text=final.text, model=model, event=event, routing=routing)


def call_chat_detailed(
    messages: list[dict],
    temperature: float = 0.3,
    model: str = DEFAULT_MODEL,
    project_tag: str = "default",
    tools: list[dict] | None = None,
    skip_guards: bool = False,
    model_group: str | None = None,
) -> CallResult:
    """Complete against real message history, optionally offering tools.

    The structured counterpart to `call_llm_detailed()`. Agent traffic needs
    two things a single flattened prompt cannot express: a conversation that
    contains tool results, and a request that offers the model tools to call.

    Everything else is identical, same guardrails before dispatch, same
    routing, same pricing, same ledger row, so a tool-using call is as tracked
    and as governed as any other. `result.event` is the logged UsageEvent and
    `result.text` may be empty when the model chose to call a tool instead of
    answering; the tool calls themselves live on `result.tool_calls`.

    No failover. Falling back mid-conversation risks a second model inventing
    a tool call the first one never made, against tools it was never shown.
    """
    from src.providers import supports_tools

    routing: dict | None = None
    if model_group:
        from src.router import select

        # Routing reads the last user turn: it is what the model is actually
        # being asked to do, and joining the whole history would let a long
        # transcript escalate a trivial follow-up.
        last_user = next(
            (m.get("content") for m in reversed(messages)
             if m.get("role") == "user" and isinstance(m.get("content"), str)),
            None,
        )
        decision = select(
            model_group,
            estimated_input_tokens=sum(len(str(m.get("content") or "")) for m in messages) // 4,
            prompt=last_user,
        )
        model = decision.model
        routing = decision.as_dict()

    provider_name = _safe_infer(model)
    provider = get_provider(model)
    if not supports_tools(provider):
        raise ProviderError(
            f"provider {provider_name!r} has no tool-calling implementation for {model!r}"
        )

    approx_prompt = "\n".join(str(m.get("content") or "") for m in messages)
    if not skip_guards:
        enforce(
            project_tag=project_tag,
            model=model,
            estimated_input_tokens=len(approx_prompt) // 4,
        )

    prompt_preview = UsageEvent.make_preview(approx_prompt)
    prompt_hash = UsageEvent.make_hash(approx_prompt)
    start = time.perf_counter()

    try:
        result = provider.complete_chat(
            messages, model=model, temperature=temperature, tools=tools
        )
    except Exception as exc:
        latency_ms = (time.perf_counter() - start) * 1000
        log_usage(
            UsageEvent(
                model=model, provider=provider_name, project_tag=project_tag,
                input_tokens=0, output_tokens=0, cost_usd=0.0, latency_ms=latency_ms,
                prompt_preview=prompt_preview, prompt_hash=prompt_hash,
                success=False, error=str(exc)[:500],
            )
        )
        logger.error("call_chat failed | %s/%s: %s", provider_name, model, str(exc)[:200])
        raise

    latency_ms = (time.perf_counter() - start) * 1000
    cost_usd = calculate_cost(
        model, result.input_tokens, result.output_tokens,
        result.cached_input_tokens, result.cache_write_tokens,
    )
    event = UsageEvent(
        model=model, provider=provider_name, project_tag=project_tag,
        input_tokens=result.input_tokens, output_tokens=result.output_tokens,
        cached_input_tokens=result.cached_input_tokens,
        cache_write_tokens=result.cache_write_tokens,
        cost_usd=cost_usd, latency_ms=latency_ms,
        prompt_preview=prompt_preview, prompt_hash=prompt_hash, success=True,
    )
    log_usage(event)
    logger.info(
        "call_chat ok | %s/%s project=%s cost=$%.6f latency=%.0fms tokens=%d/%d tools=%d",
        provider_name, model, project_tag, cost_usd, latency_ms,
        result.input_tokens, result.output_tokens, len(result.tool_calls or []),
    )

    return CallResult(
        text=result.text, model=model, event=event, routing=routing,
        tool_calls=result.tool_calls, finish_reason=result.finish_reason,
    )
