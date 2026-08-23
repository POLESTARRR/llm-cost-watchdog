"""Ollama adapter: locally-hosted models, zero marginal cost.

This is the first provider whose "credential" is not an API key but a running
process. `is_configured()` probes the server rather than reading an env var,
because a key that exists tells you nothing about whether a local daemon is up.

Why a local tier belongs in a cost router at all: every other provider here
trades quality for a *lower* price. This one trades quality for **no price**.
That changes the shape of the routing problem rather than just extending it,
and the change is instructive:

    A $0.00 model makes the `cheapest` strategy degenerate. It wins every
    comparison, for every prompt, forever.

That is not a bug to patch out. It is the clearest possible demonstration that
price alone was never a routing policy, which is why `strategy="complexity"`
(src/complexity.py) exists and why local models are only ever reachable through
it or through an explicit model id.

The cost that *is* real here is latency and the machine it runs on. Measured on
an M-series laptop with llama3.2:3b: ~5s cold model load, then ~25 tokens/sec.
The ledger records that latency like any other call, so "free" is never
reported without the number that qualifies it.

Model ids carry an explicit `ollama/` prefix (`ollama/llama3.2:3b`). Bare names
would collide with hosted providers, Ollama serves `gemma` and Google does too,
and the registry routes on prefix.
"""

import json
import os
from collections.abc import Iterator

import httpx

from src.providers.base import LLMResponse, ProviderError, StreamChunk

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")

# Local generation is slow but free; a timeout short enough for a hosted API
# would abort perfectly healthy local calls on a laptop under load.
OLLAMA_TIMEOUT_S = float(os.environ.get("OLLAMA_TIMEOUT_S", "300"))

# The probe in is_configured() runs on the hot path of every routing decision
# (`_configured_models`), so it must fail fast when nothing is listening.
_PROBE_TIMEOUT_S = 1.0

MODEL_PREFIX = "ollama/"


class OllamaProvider:
    name = "ollama"

    def is_configured(self) -> bool:
        """True if an Ollama server is reachable. Never raises.

        Called during routing to filter candidates, so an unreachable server
        must read as "not available" rather than propagating a connection
        error out of what is supposed to be a preference calculation.
        """
        try:
            return httpx.get(f"{OLLAMA_HOST}/api/tags", timeout=_PROBE_TIMEOUT_S).is_success
        except httpx.HTTPError:
            return False

    def available_models(self) -> list[str]:
        """Model ids pulled on the local server, prefixed to match the registry."""
        try:
            resp = httpx.get(f"{OLLAMA_HOST}/api/tags", timeout=_PROBE_TIMEOUT_S)
            resp.raise_for_status()
        except httpx.HTTPError:
            return []
        return [MODEL_PREFIX + m["name"] for m in resp.json().get("models", [])]

    def complete(self, prompt: str, model: str, temperature: float) -> LLMResponse:
        if not self.is_configured():
            raise ProviderError(
                f"no Ollama server reachable at {OLLAMA_HOST}. Start one with `ollama serve`, "
                f"or set OLLAMA_HOST."
            )

        payload = {
            "model": model.removeprefix(MODEL_PREFIX),
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature},
        }
        try:
            resp = httpx.post(
                f"{OLLAMA_HOST}/api/generate", json=payload, timeout=OLLAMA_TIMEOUT_S
            )
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise ProviderError(_explain_status(exc, payload["model"])) from exc
        except httpx.HTTPError as exc:
            raise ProviderError(f"Ollama request failed: {exc}") from exc

        data = resp.json()
        # Ollama names these differently from every hosted provider but means
        # the same thing: tokens of prompt processed, tokens generated. Both are
        # absent on some older builds, hence the defaults.
        return LLMResponse(
            text=data.get("response", ""),
            input_tokens=data.get("prompt_eval_count", 0) or 0,
            output_tokens=data.get("eval_count", 0) or 0,
            # No server-side prompt caching locally. Reporting 0 is accurate,
            # not a placeholder: there is no cache to read from or write to.
            cached_input_tokens=0,
            cache_write_tokens=0,
        )

    def complete_stream(
        self, prompt: str, model: str, temperature: float
    ) -> Iterator[StreamChunk]:
        """Stream from /api/generate.

        Ollama emits newline-delimited JSON, one object per token, and the
        object with `done: true` carries the counts. That matches StreamChunk's
        shape exactly, so nothing has to be buffered to fake a usage block.
        """
        if not self.is_configured():
            raise ProviderError(f"no Ollama server reachable at {OLLAMA_HOST}")

        payload = {
            "model": model.removeprefix(MODEL_PREFIX),
            "prompt": prompt,
            "stream": True,
            "options": {"temperature": temperature},
        }
        text_parts: list[str] = []
        with httpx.stream(
            "POST", f"{OLLAMA_HOST}/api/generate", json=payload, timeout=OLLAMA_TIMEOUT_S
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line.strip():
                    continue
                data = json.loads(line)
                delta = data.get("response", "")
                if delta:
                    text_parts.append(delta)
                    yield StreamChunk(text_delta=delta)
                if data.get("done"):
                    yield StreamChunk(done=True, response=LLMResponse(
                        text="".join(text_parts),
                        input_tokens=data.get("prompt_eval_count", 0) or 0,
                        output_tokens=data.get("eval_count", 0) or 0,
                    ))
                    return

    def complete_chat(
        self,
        messages: list[dict],
        model: str,
        temperature: float,
        tools: list[dict] | None = None,
    ) -> LLMResponse:
        """Chat completion with real message history and optional tools.

        Uses /api/chat rather than /api/generate, which is the endpoint that
        understands roles and tool results at all.

        One normalization matters: Ollama returns tool-call `arguments` as a
        decoded JSON **object**, while the OpenAI wire format specifies a JSON
        **string**. Clients parsing the gateway's response call json.loads on
        that field, so handing them an object breaks them. Re-encoding here
        keeps the difference inside the adapter, which is what the adapter
        layer is for.
        """
        if not self.is_configured():
            raise ProviderError(f"no Ollama server reachable at {OLLAMA_HOST}")

        payload: dict = {
            "model": model.removeprefix(MODEL_PREFIX),
            "messages": [_to_ollama_message(m) for m in messages],
            "stream": False,
            "options": {"temperature": temperature},
        }
        if tools:
            payload["tools"] = tools

        try:
            resp = httpx.post(f"{OLLAMA_HOST}/api/chat", json=payload, timeout=OLLAMA_TIMEOUT_S)
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise ProviderError(_explain_status(exc, payload["model"])) from exc
        except httpx.HTTPError as exc:
            raise ProviderError(f"Ollama request failed: {exc}") from exc

        data = resp.json()
        message = data.get("message", {}) or {}

        tool_calls = None
        if raw_calls := message.get("tool_calls"):
            tool_calls = []
            for i, call in enumerate(raw_calls):
                fn = call.get("function", {}) or {}
                args = fn.get("arguments", {})
                tool_calls.append({
                    # Not every build returns an id; a stable synthetic one is
                    # better than None, which some clients reject outright.
                    "id": call.get("id") or f"call_{i}",
                    "type": "function",
                    "function": {
                        "name": fn.get("name", ""),
                        "arguments": args if isinstance(args, str) else json.dumps(args),
                    },
                })

        return LLMResponse(
            text=message.get("content", "") or "",
            input_tokens=data.get("prompt_eval_count", 0) or 0,
            output_tokens=data.get("eval_count", 0) or 0,
            tool_calls=tool_calls,
            finish_reason="tool_calls" if tool_calls else "stop",
        )


def _explain_status(exc: httpx.HTTPStatusError, model: str) -> str:
    """Turn an Ollama HTTP error into something actionable.

    404 and 400 mean genuinely different things and the earlier version of this
    reported both as "pull the model", which sent a real debugging session
    chasing a model that was already installed. A 400 is a malformed request;
    Ollama puts the actual reason in the body, so surface it.
    """
    status = exc.response.status_code
    if status == 404:
        return f"Ollama has no model {model!r}. Pull it with `ollama pull {model}`."
    try:
        detail = exc.response.json().get("error") or exc.response.text
    except Exception:  # pragma: no cover - non-JSON error body
        detail = exc.response.text
    return f"Ollama rejected the request ({status}): {detail}"


def _to_ollama_message(message: dict) -> dict:
    """Translate one OpenAI-shaped message into what Ollama's /api/chat accepts.

    The round trip is asymmetric, and this is the bug it exists to fix: Ollama
    *returns* tool-call `arguments` as a decoded object, and also *requires* an
    object on the way back in, while the OpenAI wire format specifies a JSON
    string in both directions. The response side was normalized from the start;
    the request side was not, so every agent loop worked on turn one and failed
    on turn two with "Value looks like object, but can't find closing '}'".

    Also drops `content: null`, which OpenAI uses for an assistant turn that
    only called a tool, and which Ollama does not expect.
    """
    out = {k: v for k, v in message.items() if v is not None}

    if calls := out.get("tool_calls"):
        converted = []
        for call in calls:
            call = dict(call)
            fn = dict(call.get("function", {}))
            args = fn.get("arguments")
            if isinstance(args, str):
                try:
                    fn["arguments"] = json.loads(args)
                except json.JSONDecodeError:
                    # A model emitted unparseable arguments. Forwarding the raw
                    # string lets Ollama reject it with its own message rather
                    # than this layer inventing an interpretation.
                    pass
            call["function"] = fn
            converted.append(call)
        out["tool_calls"] = converted

    return out
