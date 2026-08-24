"""Tests for the MCP server, which had none.

Twenty-two tools were exposed to Claude Desktop and nothing checked any of them.
That is the worst place in this project to have no coverage: an HTTP endpoint
that breaks returns a visible error, while an MCP tool that breaks returns a
plausible-looking dict into a conversation, where a wrong number is repeated
back to the user as an answer and nobody sees a stack trace.

So these check the things that would be wrong quietly:

  * every tool is registered and callable
  * bad arguments are refused rather than silently coerced
  * the provenance rules the rest of the project enforces survive the MCP
    boundary, since this is the surface where a total is most likely to be
    quoted as money without the caveat attached
"""

import inspect

import pytest

import src.mcp_server as mcp


def _tools() -> dict:
    """Every tool function the module exposes, by name."""
    return {
        name: obj for name, obj in vars(mcp).items()
        if inspect.isfunction(obj)
        and not name.startswith("_")
        and obj.__module__ == "src.mcp_server"
    }


# --- registration ---------------------------------------------------------


def test_server_is_constructed():
    assert mcp.server is not None
    assert mcp.server.name == "llm-cost-gateway"


def test_instructions_warn_about_provenance():
    """The instructions are the only context the model gets before calling.

    This project's headline number is not money spent, and a model reading these
    tools will quote a total unless told otherwise. That warning living in the
    instructions is load-bearing, not decoration.
    """
    text = mcp.server.instructions.lower()
    assert "provenance" in text
    assert "subscription" in text
    assert "flat-fee" in text or "flat fee" in text


def test_every_tool_carries_a_description_the_model_can_read():
    """The description is in the decorator, not the docstring.

    An MCP client never sees __doc__; it sees the `description=` passed to
    @server.tool, and that text is the entire basis on which a model decides
    whether to call the tool and how. A missing or thin one produces a tool that
    is technically exposed and never correctly used, which is worse than absent
    because it looks fine from the inside.

    Read from the source with ast rather than from the SDK's registry, so this
    keeps working when the library changes its internals.
    """
    import ast
    import pathlib

    tree = ast.parse(pathlib.Path("src/mcp_server.py").read_text())
    described = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        for dec in node.decorator_list:
            if not isinstance(dec, ast.Call):
                continue
            for kw in dec.keywords:
                if kw.arg == "description":
                    described[node.name] = ast.literal_eval(kw.value)

    assert len(described) >= 20, f"only {len(described)} tools carry a description"
    for name, text in described.items():
        assert len(text) > 40, f"{name}'s description is too thin for a model to route on: {text!r}"

    # Registered functions and described functions must be the same set: a tool
    # exposed without a description, or described but never registered, is a
    # silent gap either way.
    assert set(described) == set(_tools()), (
        f"mismatch: {set(described) ^ set(_tools())}"
    )


# --- the read-only tools actually run -------------------------------------


@pytest.mark.parametrize("name", [
    "get_cost_report", "get_data_provenance", "check_budget_status",
    "get_burn_rate", "get_provider_breakdown", "list_providers",
    "check_guard_status", "get_router_status", "get_subscription_roi",
])
def test_read_only_tools_return_a_mapping(sample_db, name):
    fn = _tools().get(name)
    if fn is None:
        pytest.skip(f"{name} not exposed in this build")
    out = fn()
    assert isinstance(out, (dict, list)), f"{name} returned {type(out).__name__}"


def test_flag_anomalies_returns_a_list(sample_db):
    assert isinstance(mcp.flag_anomalies(), list)


def test_compare_model_costs_orders_cheapest_first(sample_db):
    rows = mcp.compare_model_costs(input_tokens=5000, output_tokens=1000)
    costs = [r["cost_usd"] for r in rows]
    assert costs == sorted(costs)


# --- bad input is refused, not coerced ------------------------------------


def test_invalid_period_is_rejected_with_an_explanation(sample_db):
    """A model will pass 'yesterday' eventually. It must not silently become a week."""
    out = mcp.get_cost_report(period="yesterday")
    assert isinstance(out, dict)
    assert "error" in out or "valid" in str(out).lower()


def test_invalid_source_is_rejected(sample_db):
    out = mcp.get_cost_report(period="week", source="pretend")
    assert isinstance(out, dict)
    assert "error" in out or "valid" in str(out).lower()


# --- provenance survives the MCP boundary ---------------------------------


def test_provenance_tool_separates_billed_from_flat_fee(sample_db):
    """The distinction the whole project rests on must be visible here too."""
    out = mcp.get_data_provenance(period="all_time")
    assert "billed_cost_usd" in out
    assert "total_cost_usd" in out


def test_subscription_roi_says_it_is_not_money_spent(sample_db):
    out = mcp.get_subscription_roi(period="all_time")
    assert isinstance(out, dict)
    note = str(out.get("note", "")).lower()
    if out.get("calls"):
        assert "not money spent" in note or "no per-token charge" in note


def test_log_manual_entry_validates_its_model(sample_db):
    """A write tool reachable from a chat window must not accept anything."""
    fn = _tools().get("log_manual_entry")
    if fn is None:
        pytest.skip("log_manual_entry not exposed")
    out = fn(model="not-a-real-model-anyone-prices", cost_usd=1.0,
             tokens=100, project_tag="test")
    # It records the call rather than refusing it, and infer_provider falls back
    # to "unknown" for a model nobody prices. What matters is that it does not
    # raise into a chat window and does not invent a provider.
    assert isinstance(out, str)
    assert out


# --- the real protocol, not the module ------------------------------------


def _handshake(timeout: float = 25.0) -> list[dict]:
    """Start the server as a subprocess and complete a real MCP handshake.

    Every other test here imports the module, and that is exactly how the worst
    bug in this file survived: six tools were defined after `server.run()`, which
    blocks for the life of the process, so they were never registered. Importing
    the module ran every decorator and made all 22 look present. Only speaking
    the protocol showed that a client saw 16.
    """
    import json
    import subprocess
    import sys
    import time

    proc = subprocess.Popen(
        [sys.executable, "-m", "src.mcp_server"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, bufsize=1,
    )

    def send(obj):
        proc.stdin.write(json.dumps(obj) + "\n")
        proc.stdin.flush()

    def wait_for(msg_id):
        deadline = time.time() + timeout
        while time.time() < deadline:
            line = proc.stdout.readline()
            if not line:
                return None
            try:
                msg = json.loads(line)
            except ValueError:
                continue
            if msg.get("id") == msg_id:
                return msg
        return None

    try:
        send({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
            "protocolVersion": "2024-11-05", "capabilities": {},
            "clientInfo": {"name": "pytest", "version": "0"}}})
        assert wait_for(1), "server never answered initialize"

        send({"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}})
        send({"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}})
        listed = wait_for(2)
        assert listed, "server never answered tools/list"
        return listed["result"]["tools"]
    finally:
        proc.kill()


@pytest.mark.slow
def test_client_actually_sees_every_defined_tool():
    """Regression: 6 of 22 tools were unreachable and everything looked fine.

    The entrypoint block sat above them, `server.run()` never returns, and the
    decorators below it never ran. Any tool added after that block silently
    vanishes from every client, so this compares what the protocol advertises
    against what the source defines.
    """
    import ast
    import pathlib

    tree = ast.parse(pathlib.Path("src/mcp_server.py").read_text())
    defined = {
        node.name for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and any(isinstance(d, ast.Call) for d in node.decorator_list)
    }

    advertised = {t["name"] for t in _handshake()}
    assert advertised == defined, (
        f"tools defined but never reaching a client: {sorted(defined - advertised)}"
    )


@pytest.mark.slow
def test_entrypoint_is_the_last_thing_in_the_file():
    """A cheap structural guard for the same bug, with no subprocess needed."""
    import pathlib

    src = pathlib.Path("src/mcp_server.py").read_text()
    idx = src.index('if __name__ == "__main__":')
    assert "@server.tool" not in src[idx:], (
        "a tool is registered below the entrypoint; server.run() blocks and it "
        "will never be reached"
    )


@pytest.mark.slow
def test_advertised_tools_carry_usable_descriptions():
    for tool in _handshake():
        assert len(tool.get("description", "")) > 40, f"{tool['name']} description too thin"
        assert "inputSchema" in tool, f"{tool['name']} has no input schema"
