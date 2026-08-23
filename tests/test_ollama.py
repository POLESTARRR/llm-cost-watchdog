"""The local provider, and the pricing rule that makes it honest.

The interesting assertions here are about *zero*: a local model must price at
exactly nothing, and must do so for models nobody has added to the table, or
the ledger reports spend that never happened.
"""

import json

import httpx
import pytest

from src.pricing import calculate_cost, get_rates, is_local_model
from src.providers import infer_provider
from src.providers.base import ProviderError
from src.providers.ollama import OllamaProvider

# A real /api/generate response, captured from a local llama3.2:3b.
GENERATE_RESPONSE = {
    "model": "llama3.2:3b",
    "response": "def reverse_string(s):\n    return s[::-1]",
    "done": True,
    "prompt_eval_count": 33,
    "eval_count": 52,
    "total_duration": 7434177709,
}


@pytest.fixture(autouse=True)
def _clear_probe_cache():
    """The reachability probe is cached for 30s; tests must not inherit it."""
    import src.providers.ollama as mod

    mod._probe_cache = None
    yield
    mod._probe_cache = None


@pytest.fixture
def provider():
    return OllamaProvider()


class FakeResponse:
    """Minimal stand-in for httpx.Response.

    httpx's own MockTransport builds responses that aren't bound to a request,
    so `raise_for_status()` refuses to run on them. The adapter only ever
    touches three attributes, so a stub is both sufficient and clearer about
    what the contract actually is.
    """

    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self._payload = payload or {}

    @property
    def is_success(self):
        return 200 <= self.status_code < 300

    def raise_for_status(self):
        if not self.is_success:
            raise httpx.HTTPStatusError("error", request=None, response=self)

    @property
    def text(self):
        return json.dumps(self._payload)

    def json(self):
        return self._payload


class TestRouting:
    def test_namespaced_prefix_routes_to_ollama(self):
        assert infer_provider("ollama/llama3.2:3b") == "ollama"

    def test_bare_vendor_names_still_route_to_their_vendor(self):
        """The collision this prefix exists to prevent: Ollama also serves Gemma."""
        assert infer_provider("gemma-3") == "google"
        assert infer_provider("ollama/gemma3") == "ollama"


class TestPricing:
    def test_local_models_cost_exactly_zero(self):
        assert calculate_cost("ollama/llama3.2:3b", 100_000, 50_000) == 0.0

    def test_unlisted_local_models_are_free_not_guessed(self):
        """The bug this prevents: an unlisted model hitting the fallback rate
        and being reported as real spend that never occurred."""
        assert calculate_cost("ollama/some-model-nobody-added", 10_000, 10_000) == 0.0
        assert get_rates("ollama/anything") == {"input": 0.0, "cached_input": 0.0, "output": 0.0}

    def test_unlisted_hosted_models_still_fall_back_to_a_real_rate(self):
        """The zero-rate rule must not leak to hosted models."""
        assert calculate_cost("gpt-9-unreleased", 10_000, 10_000) > 0

    def test_is_local_model(self):
        assert is_local_model("ollama/llama3.2:3b")
        assert not is_local_model("claude-opus-5")


class TestComplete:
    def test_maps_ollama_token_names_onto_the_normalized_shape(self, provider, monkeypatch):
        sent = {}

        monkeypatch.setattr(httpx, "get", lambda url, **k: FakeResponse(200, {"models": []}))

        def fake_post(url, **kwargs):
            sent.update(kwargs.get("json") or {})
            return FakeResponse(200, GENERATE_RESPONSE)

        monkeypatch.setattr(httpx, "post", fake_post)

        r = provider.complete("reverse a string", model="ollama/llama3.2:3b", temperature=0.3)
        # The namespace prefix is ours, not Ollama's; it must be stripped.
        assert sent["model"] == "llama3.2:3b"
        assert sent["stream"] is False
        assert r.input_tokens == 33
        assert r.output_tokens == 52
        assert r.text.startswith("def reverse_string")
        # No server-side cache exists locally; 0 is accurate, not a placeholder.
        assert r.cached_input_tokens == 0
        assert r.cache_write_tokens == 0

    def test_unreachable_server_reads_as_unconfigured_rather_than_raising(self, provider, monkeypatch):
        """is_configured() runs inside routing; a connection error there would
        turn a preference calculation into a crash."""
        def boom(*a, **k):
            raise httpx.ConnectError("nothing listening")

        monkeypatch.setattr(httpx, "get", boom)
        assert provider.is_configured() is False

    def test_complete_explains_a_missing_server(self, provider, monkeypatch):
        monkeypatch.setattr(httpx, "get", lambda *a, **k: (_ for _ in ()).throw(httpx.ConnectError("x")))
        with pytest.raises(ProviderError, match="ollama serve"):
            provider.complete("hi", model="ollama/llama3.2:3b", temperature=0.3)


class TestToolCallRoundTrip:
    """The asymmetry that broke every agent loop on turn two.

    Ollama RETURNS tool-call arguments as a decoded object and also REQUIRES an
    object on the way back in. The OpenAI wire format this gateway speaks
    specifies a JSON string in both directions. Normalizing only the response
    made turn one work and turn two fail with Ollama's memorable complaint:
    "Value looks like object, but can't find closing '}' symbol".
    """

    def test_inbound_arguments_are_decoded_to_an_object(self):
        from src.providers.ollama import _to_ollama_message

        out = _to_ollama_message({
            "role": "assistant", "content": None,
            "tool_calls": [{"id": "c1", "type": "function", "function": {
                "name": "get_weather", "arguments": '{"city": "Paris"}'}}],
        })
        assert out["tool_calls"][0]["function"]["arguments"] == {"city": "Paris"}

    def test_null_content_is_dropped(self):
        """OpenAI uses content:null for a turn that only called a tool."""
        from src.providers.ollama import _to_ollama_message

        assert "content" not in _to_ollama_message({"role": "assistant", "content": None})

    def test_unparseable_arguments_are_forwarded_untouched(self):
        """Let Ollama reject it with its own message rather than guessing."""
        from src.providers.ollama import _to_ollama_message

        out = _to_ollama_message({
            "role": "assistant",
            "tool_calls": [{"function": {"name": "f", "arguments": "{not json"}}],
        })
        assert out["tool_calls"][0]["function"]["arguments"] == "{not json"

    def test_outbound_arguments_are_encoded_to_a_string(self, provider, monkeypatch):
        monkeypatch.setattr(httpx, "get", lambda url, **k: FakeResponse(200, {"models": []}))
        monkeypatch.setattr(httpx, "post", lambda url, **k: FakeResponse(200, {
            "message": {"role": "assistant", "content": "", "tool_calls": [
                {"id": "call_x", "function": {"name": "get_weather",
                                              "arguments": {"city": "Paris"}}}]},
            "prompt_eval_count": 152, "eval_count": 13,
        }))
        r = provider.complete_chat(
            [{"role": "user", "content": "weather?"}],
            model="ollama/llama3.2:3b", temperature=0.3, tools=[{"type": "function"}],
        )
        assert r.finish_reason == "tool_calls"
        assert r.tool_calls[0]["function"]["arguments"] == '{"city": "Paris"}'

    def test_a_missing_tool_call_id_gets_a_synthetic_one(self, provider, monkeypatch):
        """Some builds omit it, and clients reject a null id."""
        monkeypatch.setattr(httpx, "get", lambda url, **k: FakeResponse(200, {"models": []}))
        monkeypatch.setattr(httpx, "post", lambda url, **k: FakeResponse(200, {
            "message": {"tool_calls": [{"function": {"name": "f", "arguments": {}}}]},
            "prompt_eval_count": 1, "eval_count": 1,
        }))
        r = provider.complete_chat([{"role": "user", "content": "x"}],
                                   model="ollama/llama3.2:3b", temperature=0.3)
        assert r.tool_calls[0]["id"] == "call_0"


class TestErrorMessages:
    """404 and 400 mean different things; reporting both as 'pull the model'
    sent a real debugging session chasing an already-installed model."""

    def test_404_says_pull_the_model(self, provider, monkeypatch):
        from src.providers.ollama import _explain_status

        exc = httpx.HTTPStatusError("x", request=None, response=FakeResponse(404))
        assert "ollama pull" in _explain_status(exc, "llama3.2:3b")

    def test_400_surfaces_ollamas_own_reason(self):
        from src.providers.ollama import _explain_status

        resp = FakeResponse(400, {"error": "Value looks like object"})
        exc = httpx.HTTPStatusError("x", request=None, response=resp)
        msg = _explain_status(exc, "llama3.2:3b")
        assert "Value looks like object" in msg
        assert "pull" not in msg


def test_the_probe_is_cached_so_routing_does_not_pay_for_it(provider, monkeypatch):
    """On a deployed host with no Ollama, an uncached probe is a per-request tax."""
    calls = []

    def counting_get(url, **kwargs):
        calls.append(url)
        return FakeResponse(200, {"models": []})

    monkeypatch.setattr(httpx, "get", counting_get)
    for _ in range(5):
        assert provider.is_configured() is True
    assert len(calls) == 1
