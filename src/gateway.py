"""OpenAI-compatible gateway: adoption without a code change.

Until this existed, using this project meant rewriting every LLM call in your
codebase to import `call_llm()`. That is a real migration, and it is why, after
this repo had a ledger, a router, guardrails, waste detection and 360 passing
tests, it had recorded exactly **zero** live calls, including from its author's
own thirteen other projects. The integration cost exceeded the benefit for
every single one of them.

The importer, meanwhile, collected thousands of real rows, precisely because it
required no integration at all: it read files that already existed. This module
applies that lesson to the live path.

    export OPENAI_BASE_URL=http://localhost:8000/v1
    export OPENAI_API_KEY=wd-myproject       # names the project in the ledger

Any application built on the OpenAI SDK, or on anything that speaks its wire
format, is now tracked, routed, budget-enforced and cooled-down without one
line of its source changing. Point `model` at a declared model group
(`"group:fast"`) and it is routed; name a model directly and it is not.

**Streaming is real streaming.** `stream: true` returns server-sent events in
OpenAI's chunk format, and the wrapper underneath measures time to first token
from the first byte of actual content, recording it in the ledger alongside
(never instead of) total latency. The shortcut this deliberately avoids is
emitting one buffered chunk and calling the whole duration a TTFT, which would
put a number in the ledger that measures nothing.

Streams do not fail over. Once bytes have reached the client, quietly re-running
the prompt elsewhere would splice two different answers into one response body,
so a mid-stream failure arrives as a terminal `error` event instead.

**Tool calling is supported** on providers that implement it (Anthropic,
OpenAI, Ollama; not Gemini here). Requests offering `tools`, or replaying a
history that already contains tool results, take a structured path that passes
real messages through instead of flattening them, because a `tool` role has no
textual equivalent a provider will interpret correctly.

The one combination still refused is **streaming plus tools**, with the reason:
reassembling a tool call from partial argument deltas is real work this has not
done yet, and guessing at half-parsed JSON would be worse than saying no.
"""

import json
import logging
import os
import time
import uuid

from fastapi import APIRouter, Header, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from src.guard import BudgetExceededError
from src.router import RoutingError, model_groups

logger = logging.getLogger("llm-cost-watchdog")

router = APIRouter(prefix="/v1", tags=["gateway"])

# Route to a declared model group by asking for `group:<name>` as the model.
# A prefix rather than bare group names so a group called "gpt-4" can never
# shadow a real model id.
GROUP_PREFIX = "group:"

# API keys are how a multi-project gateway attributes spend. The convention is
# `wd-<project>`: the suffix becomes the project_tag, so per-project caps in
# guard.py apply to gateway traffic exactly as they do to direct calls.
KEY_PREFIX = "wd-"
DEFAULT_PROJECT = os.environ.get("WATCHDOG_GATEWAY_PROJECT", "gateway")

# Optional shared secret. When set, requests must present it (or a wd-* key) or
# be refused. Unset means open, which is correct for a localhost tool and
# dangerous the moment it is exposed, hence the warning in /v1/models.
GATEWAY_KEY = os.environ.get("WATCHDOG_GATEWAY_KEY")


class ChatMessage(BaseModel):
    role: str
    # `content` is `str | list | None` in the real API (multimodal parts, tool
    # results). Accepting the loose type and flattening is more robust than
    # rejecting shapes we could reasonably handle.
    content: str | list | None = None
    # Present when an agent replays its own history back: the assistant turn
    # that requested a tool, and the tool turn answering it. Without these the
    # second round-trip of every agent loop would be rejected as malformed.
    tool_calls: list[dict] | None = None
    tool_call_id: str | None = None
    name: str | None = None


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    temperature: float = 0.3
    stream: bool = False
    tools: list | None = None
    # Accepted and ignored, listing them keeps a strict client from erroring on
    # an unexpected-field response, while making clear they have no effect.
    max_tokens: int | None = None
    top_p: float | None = None
    user: str | None = Field(default=None)


def _flatten(messages: list[ChatMessage]) -> str:
    """Collapse a chat transcript into the single prompt the wrapper takes.

    Lossy by construction: the providers underneath have real multi-turn APIs
    and this throws that structure away. It is the honest cost of putting one
    uniform wrapper in front of three vendors, and it is bounded, role labels
    are preserved so the model still sees who said what.
    """
    parts: list[str] = []
    for m in messages:
        content = m.content
        if isinstance(content, list):
            # Multimodal content parts: keep the text, drop images. A local 3B
            # model has nowhere to put an image anyway.
            content = " ".join(
                p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"
            )
        if not content:
            continue
        parts.append(content if m.role == "user" else f"[{m.role}]\n{content}")
    return "\n\n".join(parts).strip()


def _project_from_key(authorization: str | None) -> str:
    """Derive the ledger's project tag from the bearer token.

    Attribution is the whole reason a gateway beats a wrapper for more than one
    caller: the ledger already breaks cost down by project, and this is what
    supplies that dimension without the caller passing anything extra.
    """
    if not authorization:
        return DEFAULT_PROJECT
    token = authorization.removeprefix("Bearer ").strip()
    if token.startswith(KEY_PREFIX):
        return token[len(KEY_PREFIX):] or DEFAULT_PROJECT
    return DEFAULT_PROJECT


def _authorized(authorization: str | None) -> bool:
    if not GATEWAY_KEY:
        return True
    if not authorization:
        return False
    token = authorization.removeprefix("Bearer ").strip()
    return token == GATEWAY_KEY or token.startswith(KEY_PREFIX)


@router.get("/models")
def list_models():
    """OpenAI-shaped model list. Many clients call this on startup.

    Declared groups are listed alongside real models so `group:fast` shows up
    in a client's model picker as a first-class choice.
    """
    from src.pricing import PRICING_TABLE
    from src.providers import _PROVIDERS, configured_providers

    configured = configured_providers()
    data = []

    for model in PRICING_TABLE:
        from src.providers import ProviderError, infer_provider

        try:
            owner = infer_provider(model)
        except ProviderError:
            continue
        if configured.get(owner):
            data.append({"id": model, "object": "model", "owned_by": owner})

    # Locally-pulled models are discovered, not enumerated: the table can't
    # know what someone has on their own disk.
    ollama = _PROVIDERS.get("ollama")
    if ollama is not None and ollama.is_configured():
        data += [{"id": m, "object": "model", "owned_by": "ollama"} for m in ollama.available_models()]

    data += [
        {"id": f"{GROUP_PREFIX}{name}", "object": "model", "owned_by": "watchdog-router"}
        for name in model_groups()
    ]

    body = {"object": "list", "data": data}
    if not GATEWAY_KEY:
        body["warning"] = (
            "This gateway has no WATCHDOG_GATEWAY_KEY set and will serve any caller that can "
            "reach it. That is fine on localhost and unsafe anywhere else."
        )
    return body


@router.post("/chat/completions")
async def chat_completions(
    body: ChatCompletionRequest,
    request: Request,
    authorization: str | None = Header(default=None),
):
    """The endpoint. Accepts OpenAI's request shape, returns its response shape."""
    from src.utils import call_llm_detailed

    if not _authorized(authorization):
        raise HTTPException(status_code=401, detail="invalid api key")

    # Refuse what isn't supported, at the door, with the reason. See module docstring.
    project = _project_from_key(authorization)
    model_group = None
    model = body.model
    if model.startswith(GROUP_PREFIX):
        model_group = model[len(GROUP_PREFIX):]
        model = None

    # Structured path: tools offered, or a history that already contains tool
    # results or assistant tool calls. Flattening any of those to a string
    # would discard the exact structure the provider needs.
    if _needs_chat_path(body):
        if body.stream:
            raise HTTPException(
                status_code=400,
                detail=(
                    "streaming tool calls is not supported. Text streaming is; a streamed "
                    "tool call would have to be reassembled from partial argument deltas, "
                    "and this gateway does not do that yet. Set stream=false when passing tools."
                ),
            )
        return _chat_with_tools(body, project, model, model_group)

    prompt = _flatten(body.messages)
    if not prompt:
        raise HTTPException(status_code=400, detail="no text content in messages")

    if body.stream:
        return StreamingResponse(
            _sse(prompt, body, project, model, model_group),
            media_type="text/event-stream",
            # Proxies that buffer would defeat the point of streaming at all.
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    try:
        result = (
            call_llm_detailed(prompt, temperature=body.temperature,
                              project_tag=project, model_group=model_group)
            if model_group else
            call_llm_detailed(prompt, temperature=body.temperature,
                              project_tag=project, model=model)
        )
    except BudgetExceededError as exc:
        # 429 rather than 402: this is a self-imposed quota that will clear,
        # and every OpenAI client already knows how to back off on a 429.
        raise HTTPException(
            status_code=429,
            detail={"message": exc.verdict.message, "guardrail": exc.verdict.triggered,
                    "type": "budget_exceeded"},
        ) from exc
    except RoutingError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 - surface provider errors as 502
        logger.error("gateway call failed: %s", exc)
        raise HTTPException(status_code=502, detail=f"upstream provider error: {exc}") from exc

    event = result.event

    # Fire-and-forget shadow comparison. Deliberately after the response is
    # built, and swallowing everything: a quality experiment must never be able
    # to fail a request that already succeeded.
    _maybe_shadow(prompt, result, project)

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": result.model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": result.text},
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": event.input_tokens,
            "completion_tokens": event.output_tokens,
            "total_tokens": event.input_tokens + event.output_tokens,
            "prompt_tokens_details": {"cached_tokens": event.cached_input_tokens},
        },
        # Non-standard, namespaced so a strict client ignores it. This is the
        # gateway's whole value proposition made visible in-band: what the call
        # cost, and if it was routed, exactly why it went where it did.
        "x_watchdog": {
            "cost_usd": event.cost_usd,
            "latency_ms": round(event.latency_ms, 1),
            "project": project,
            "routing": result.routing,
        },
    }


def _maybe_shadow(prompt: str, result, project: str) -> None:
    try:
        from src import shadow

        if not shadow.enabled() or not shadow.should_shadow(prompt):
            return
        shadow.run_shadow(
            prompt=prompt,
            real_model=result.model,
            real_response=result.text,
            real_cost_usd=result.event.cost_usd,
            real_latency_ms=result.event.latency_ms,
            project_tag=project,
        )
    except Exception:  # pragma: no cover - defensive by design
        logger.debug("shadow comparison failed", exc_info=True)


def _sse_event(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


def _sse(prompt: str, body: "ChatCompletionRequest", project: str,
         model: str | None, model_group: str | None):
    """Server-sent events in OpenAI's streaming shape.

    Real streaming, not a single buffered chunk dressed up as one. The wrapper
    underneath measures time to first token from the first byte of content and
    records it separately from total latency, so what lands in the ledger is a
    measurement rather than a restatement of the total duration.

    Errors reach the client as a terminal `error` event rather than a status
    code: headers are long gone by the time a provider fails mid-stream, so a
    500 is not available and silence would be indistinguishable from a short
    answer.
    """
    from src.utils import stream_llm

    completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())
    sent_model = model or (model_group or "unknown")

    def frame(delta: dict, finish: str | None = None) -> dict:
        return {
            "id": completion_id, "object": "chat.completion.chunk",
            "created": created, "model": sent_model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        }

    try:
        stream = stream_llm(
            prompt, temperature=body.temperature, project_tag=project,
            model_group=model_group, **({"model": model} if model else {}),
        )
        yield _sse_event(frame({"role": "assistant"}))

        result = None
        for item in stream:
            if isinstance(item, str):
                yield _sse_event(frame({"content": item}))
            else:
                result = item

        yield _sse_event(frame({}, finish="stop"))

        if result is not None:
            event = result.event
            # A usage frame on the final event, matching OpenAI's
            # stream_options.include_usage behaviour, plus the cost data that
            # is the reason this gateway exists.
            yield _sse_event({
                "id": completion_id, "object": "chat.completion.chunk",
                "created": created, "model": result.model, "choices": [],
                "usage": {
                    "prompt_tokens": event.input_tokens,
                    "completion_tokens": event.output_tokens,
                    "total_tokens": event.input_tokens + event.output_tokens,
                },
                "x_watchdog": {
                    "cost_usd": event.cost_usd,
                    "latency_ms": round(event.latency_ms, 1),
                    "ttft_ms": round(event.ttft_ms, 1) if event.ttft_ms is not None else None,
                    "project": project,
                    "routing": result.routing,
                },
            })
    except BudgetExceededError as exc:
        yield _sse_event({"error": {"message": exc.verdict.message,
                                    "type": "budget_exceeded",
                                    "guardrail": exc.verdict.triggered}})
    except Exception as exc:  # noqa: BLE001
        logger.error("gateway stream failed: %s", exc)
        yield _sse_event({"error": {"message": str(exc), "type": "upstream_error"}})

    yield "data: [DONE]\n\n"


def _needs_chat_path(body: "ChatCompletionRequest") -> bool:
    """Whether this request must preserve message structure.

    True when tools are offered, or when the history already contains a tool
    result or an assistant turn carrying tool calls. Those cannot survive
    flattening: a `tool` role has no textual equivalent the provider will
    interpret correctly.
    """
    if body.tools:
        return True
    return any(
        m.role == "tool" or getattr(m, "tool_calls", None) for m in body.messages
    )


def _chat_with_tools(body: "ChatCompletionRequest", project: str,
                     model: str | None, model_group: str | None):
    """Tool-calling path: real messages in, OpenAI tool_calls out."""
    from src.providers import ProviderError
    from src.utils import call_chat_detailed

    messages = [m.model_dump(exclude_none=True) for m in body.messages]

    try:
        result = call_chat_detailed(
            messages, temperature=body.temperature, project_tag=project,
            tools=body.tools, model_group=model_group,
            **({"model": model} if model else {}),
        )
    except BudgetExceededError as exc:
        raise HTTPException(
            status_code=429,
            detail={"message": exc.verdict.message, "guardrail": exc.verdict.triggered,
                    "type": "budget_exceeded"},
        ) from exc
    except ProviderError as exc:
        # A provider with no tool implementation is a 400: the request is
        # unanswerable as asked, and retrying it unchanged will not help.
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RoutingError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.error("gateway tool call failed: %s", exc)
        raise HTTPException(status_code=502, detail=f"upstream provider error: {exc}") from exc

    event = result.event
    message: dict = {"role": "assistant", "content": result.text or None}
    if result.tool_calls:
        message["tool_calls"] = result.tool_calls

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": result.model,
        "choices": [{"index": 0, "message": message, "finish_reason": result.finish_reason}],
        "usage": {
            "prompt_tokens": event.input_tokens,
            "completion_tokens": event.output_tokens,
            "total_tokens": event.input_tokens + event.output_tokens,
            "prompt_tokens_details": {"cached_tokens": event.cached_input_tokens},
        },
        "x_watchdog": {
            "cost_usd": event.cost_usd,
            "latency_ms": round(event.latency_ms, 1),
            "project": project,
            "routing": result.routing,
        },
    }
