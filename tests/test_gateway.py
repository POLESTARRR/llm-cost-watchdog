"""The gateway's contract: OpenAI's wire format in, this project's ledger out.

The tests that matter most here are the *refusals*. A proxy that silently drops
streaming or tool calls is worse than one that never existed, because the
failure surfaces three layers away in someone else's code.
"""

import pytest
from fastapi.testclient import TestClient

from src.gateway import _flatten, _project_from_key
from src.usage_schema import UsageEvent
from src.utils import CallResult


@pytest.fixture
def client(temp_db, monkeypatch):
    monkeypatch.setenv("WATCHDOG_GUARD_MODE", "off")
    from dashboard.app import app

    return TestClient(app)


@pytest.fixture
def fake_call(monkeypatch):
    """Replace the wrapper so no network call happens, keeping the real shape."""
    calls = {}

    def _fake(prompt, temperature=0.3, project_tag="default", model=None, model_group=None, **kw):
        calls.update(prompt=prompt, project_tag=project_tag, model=model, model_group=model_group)
        event = UsageEvent(
            model=model or "routed-model", provider="test", project_tag=project_tag,
            input_tokens=11, output_tokens=7, cost_usd=0.00042, latency_ms=123.4,
        )
        return CallResult(text="hello from the model", model=event.model, event=event)

    monkeypatch.setattr("src.utils.call_llm_detailed", _fake)
    return calls


def _body(**over):
    body = {"model": "claude-haiku-4-5", "messages": [{"role": "user", "content": "hi"}]}
    body.update(over)
    return body


TOOL = {"type": "function", "function": {"name": "get_weather", "description": "weather",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}}


@pytest.fixture
def fake_chat(monkeypatch):
    """Replace the structured wrapper, keeping its real return shape."""
    calls = {}

    def _fake(messages, temperature=0.3, project_tag="default", tools=None,
              model=None, model_group=None, **kw):
        calls.update(messages=messages, tools=tools, model=model, model_group=model_group)
        event = UsageEvent(
            model=model or "routed-model", provider="test", project_tag=project_tag,
            input_tokens=152, output_tokens=13, cost_usd=0.001, latency_ms=90.0,
        )
        return CallResult(
            text="", model=event.model, event=event,
            tool_calls=[{"id": "call_1", "type": "function",
                         "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'}}],
            finish_reason="tool_calls",
        )

    monkeypatch.setattr("src.utils.call_chat_detailed", _fake)
    return calls


class TestToolCalling:
    def test_tools_take_the_structured_path_and_return_tool_calls(self, client, fake_chat):
        r = client.post("/v1/chat/completions", json=_body(tools=[TOOL]))
        assert r.status_code == 200
        choice = r.json()["choices"][0]
        assert choice["finish_reason"] == "tool_calls"
        call = choice["message"]["tool_calls"][0]
        assert call["function"]["name"] == "get_weather"
        # Arguments must be a JSON *string*: clients call json.loads on it.
        assert isinstance(call["function"]["arguments"], str)

    def test_messages_are_passed_through_not_flattened(self, client, fake_chat):
        client.post("/v1/chat/completions", json=_body(tools=[TOOL]))
        assert isinstance(fake_chat["messages"], list)
        assert fake_chat["messages"][0]["role"] == "user"
        assert fake_chat["tools"] == [TOOL]

    def test_a_tool_result_turn_takes_the_structured_path_without_tools(self, client, fake_chat):
        """The second round-trip of every agent loop. A `tool` role has no
        textual equivalent, so it must never be flattened."""
        r = client.post("/v1/chat/completions", json=_body(messages=[
            {"role": "user", "content": "weather in Paris?"},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "call_1", "type": "function",
                 "function": {"name": "get_weather", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "call_1", "content": "18C, clear"},
        ]))
        assert r.status_code == 200
        assert [m["role"] for m in fake_chat["messages"]] == ["user", "assistant", "tool"]

    def test_streaming_with_tools_is_refused_with_the_reason(self, client, fake_chat):
        r = client.post("/v1/chat/completions", json=_body(tools=[TOOL], stream=True))
        assert r.status_code == 400
        assert "streaming tool calls is not supported" in r.json()["detail"]
        assert "argument deltas" in r.json()["detail"]

    def test_a_provider_without_tool_support_is_a_400(self, client, monkeypatch):
        from src.providers import ProviderError

        def _boom(*a, **k):
            raise ProviderError("provider 'google' has no tool-calling implementation")

        monkeypatch.setattr("src.utils.call_chat_detailed", _boom)
        r = client.post("/v1/chat/completions", json=_body(tools=[TOOL]))
        assert r.status_code == 400
        assert "no tool-calling implementation" in r.json()["detail"]


class TestRefusals:
    """Unsupported features must fail loudly, at the door, with the reason."""

    def test_empty_messages_are_rejected(self, client, fake_call):
        r = client.post("/v1/chat/completions", json=_body(messages=[{"role": "user", "content": ""}]))
        assert r.status_code == 400


class TestCompletion:
    def test_returns_openai_response_shape(self, client, fake_call):
        r = client.post("/v1/chat/completions", json=_body())
        assert r.status_code == 200
        b = r.json()
        assert b["object"] == "chat.completion"
        assert b["choices"][0]["message"]["role"] == "assistant"
        assert b["choices"][0]["message"]["content"] == "hello from the model"
        assert b["usage"] == {
            "prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18,
            "prompt_tokens_details": {"cached_tokens": 0},
        }

    def test_reports_cost_in_a_namespaced_extension(self, client, fake_call):
        """The value proposition, visible in-band without breaking strict clients."""
        b = client.post("/v1/chat/completions", json=_body()).json()
        assert b["x_watchdog"]["cost_usd"] == 0.00042
        assert b["x_watchdog"]["latency_ms"] == 123.4

    def test_group_prefix_routes_instead_of_naming_a_model(self, client, fake_call):
        client.post("/v1/chat/completions", json=_body(model="group:fast"))
        assert fake_call["model_group"] == "fast"
        assert fake_call["model"] is None

    def test_a_plain_model_id_is_not_routed(self, client, fake_call):
        client.post("/v1/chat/completions", json=_body(model="claude-haiku-4-5"))
        assert fake_call["model_group"] is None
        assert fake_call["model"] == "claude-haiku-4-5"

    def test_budget_block_surfaces_as_429(self, client, monkeypatch):
        from src.guard import BudgetExceededError, GuardVerdict

        def _boom(*a, **k):
            raise BudgetExceededError(
                GuardVerdict(allowed=False, mode="block", triggered=["weekly_budget"],
                             message="weekly spend has reached the budget")
            )

        monkeypatch.setattr("src.utils.call_llm_detailed", _boom)
        r = client.post("/v1/chat/completions", json=_body())
        # 429, not 402: a self-imposed quota clears, and every OpenAI client
        # already knows how to back off on a 429.
        assert r.status_code == 429
        assert r.json()["detail"]["type"] == "budget_exceeded"


class TestAttribution:
    """Per-project attribution is why a gateway beats a wrapper for >1 caller."""

    def test_api_key_suffix_becomes_the_project_tag(self, client, fake_call):
        client.post("/v1/chat/completions", json=_body(),
                    headers={"Authorization": "Bearer wd-checkout-service"})
        assert fake_call["project_tag"] == "checkout-service"

    @pytest.mark.parametrize("header,expected", [
        (None, "gateway"),
        ("Bearer sk-something-else", "gateway"),
        ("Bearer wd-", "gateway"),
        ("Bearer wd-alpha", "alpha"),
    ])
    def test_project_derivation(self, header, expected):
        assert _project_from_key(header) == expected


class TestFlatten:
    def test_preserves_who_said_what(self):
        out = _flatten([
            type("M", (), {"role": "system", "content": "be terse"})(),
            type("M", (), {"role": "user", "content": "hello"})(),
        ])
        assert "[system]" in out and "be terse" in out
        # User content is not labelled; it is the prompt.
        assert out.endswith("hello")

    def test_drops_images_but_keeps_text_from_multimodal_parts(self):
        msg = type("M", (), {"role": "user", "content": [
            {"type": "text", "text": "what is this"},
            {"type": "image_url", "image_url": {"url": "data:..."}},
        ]})()
        assert _flatten([msg]) == "what is this"

    def test_skips_empty_content(self):
        msgs = [
            type("M", (), {"role": "user", "content": None})(),
            type("M", (), {"role": "user", "content": "real"})(),
        ]
        assert _flatten(msgs) == "real"


class TestModelsEndpoint:
    def test_lists_groups_alongside_models(self, client, monkeypatch):
        monkeypatch.setenv("WATCHDOG_GROUP_FAST", "claude-haiku-4-5,gpt-5-nano")
        b = client.get("/v1/models").json()
        ids = {m["id"] for m in b["data"]}
        assert "group:fast" in ids

    def test_warns_when_unauthenticated_and_open(self, client):
        b = client.get("/v1/models").json()
        assert "warning" in b and "unsafe" in b["warning"]


class TestDeployedSafety:
    """A public gateway with no key spends the owner's quota for strangers.

    The failure this guards against is silent and expensive, so the default
    flips with the environment instead of staying permissive everywhere.
    """

    def test_deployed_without_a_key_refuses_completions(self, client, fake_call, monkeypatch):
        monkeypatch.setenv("PORT", "10000")          # Render/Railway/Fly/Heroku all set this
        monkeypatch.setattr("src.gateway.GATEWAY_KEY", None)
        r = client.post("/v1/chat/completions", json=_body())
        assert r.status_code == 503
        assert "WATCHDOG_GATEWAY_KEY" in r.json()["detail"]

    def test_deployed_with_a_key_serves_normally(self, client, fake_call, monkeypatch):
        monkeypatch.setenv("PORT", "10000")
        monkeypatch.setattr("src.gateway.GATEWAY_KEY", "secret")
        r = client.post("/v1/chat/completions", json=_body(),
                        headers={"Authorization": "Bearer secret"})
        assert r.status_code == 200

    def test_localhost_without_a_key_still_works(self, client, fake_call, monkeypatch):
        """No key locally is convenience, not a vulnerability."""
        monkeypatch.delenv("PORT", raising=False)
        monkeypatch.delenv("TURSO_DATABASE_URL", raising=False)
        monkeypatch.setattr("src.gateway.GATEWAY_KEY", None)
        assert client.post("/v1/chat/completions", json=_body()).status_code == 200

    def test_the_dashboard_still_works_when_completions_are_refused(self, client, monkeypatch):
        """Refusing to spend money must not take the read-only views down."""
        monkeypatch.setenv("PORT", "10000")
        monkeypatch.setattr("src.gateway.GATEWAY_KEY", None)
        assert client.get("/report").status_code == 200
        assert "DEPLOYED WITHOUT" in client.get("/v1/models").json()["warning"]
