#!/usr/bin/env python3
"""Six live proofs, run through the official OpenAI SDK. Nothing is mocked.

`gateway_demo.py` uses urllib, which proves the wire format is right. This uses
`openai-python` itself, unpatched, because that is the actual claim: an existing
application does not have to change to sit behind this gateway. A hand-rolled
HTTP client can be made to work with a wire format that is merely close enough;
a real SDK cannot.

    # terminal 1
    WATCHDOG_GROUP_LADDER=ollama/llama3.2:3b,gemini-flash-lite-latest,gemini-pro-latest \
    WATCHDOG_ROUTING_STRATEGY=complexity \
    venv/bin/uvicorn dashboard.app:app --port 8000

    # terminal 2
    venv/bin/python scripts/proof.py

Proof 5 (the guardrail) needs the server started with `WATCHDOG_GUARD_MODE=block`
and a tiny `WEEKLY_BUDGET_USD`, so it is skipped unless --prove-guard is passed;
it is the one check that requires the server to be configured to refuse work.

With only Ollama running, every proof but the routing tiers still works and the
whole run costs $0.00.
"""

import argparse
import json
import sys
import time

try:
    import openai
    from openai import OpenAI
except ImportError:  # pragma: no cover
    sys.exit("pip install openai")

RULE = "=" * 74


def header(n: int, text: str) -> None:
    print(f"\n{RULE}\nPROOF {n} — {text}\n{RULE}")


def proof_1_sdk(client, model):
    header(1, "the official OpenAI SDK talks to this gateway, unmodified")
    r = client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": "Reverse 'hello' in Python. One line."}]
    )
    wd = r.model_extra["x_watchdog"]
    print(f"  client   : openai-python {openai.__version__}  (no patches, no shims)")
    print(f"  model    : {r.model}")
    print(f"  answer   : {(r.choices[0].message.content or '').strip()[:60]}")
    print(f"  usage    : {r.usage.total_tokens} tokens")
    print(f"  cost     : ${wd['cost_usd']}")
    print(f"  project  : {wd['project']}   <- derived from the API key, not the code")


def proof_2_routing(client, group):
    header(2, "same client, same code: the router picks the model per prompt")
    prompts = [
        "reformat this JSON: {'a':1}",
        "Can you take a look at the user service?",
        "Design a zero-downtime schema migration and explain the trade-offs.",
    ]
    for prompt in prompts:
        try:
            r = client.chat.completions.create(
                model=group, messages=[{"role": "user", "content": prompt}]
            )
        except openai.APIStatusError as exc:
            print(f"  {prompt[:46]:<48} -> {exc.status_code}: {exc.response.text[:60]}")
            continue
        wd = r.model_extra["x_watchdog"]
        rt = wd.get("routing") or {}
        tier = (rt.get("complexity") or {}).get("tier", "?")
        print(f"  {prompt[:60]}")
        print(f"     tier={tier:<9} -> {r.model:<26} ${wd['cost_usd']:.6f}  {wd['latency_ms']:.0f}ms")
        if rt.get("fell_back_from"):
            print(f"     (fell back from {rt['fell_back_from']}: {rt['fell_back_reason']})")


def proof_3_streaming(client, model):
    header(3, "real streaming, with a time-to-first-token that is measured")
    t0, first, n, text, wd = time.time(), None, 0, "", None
    for chunk in client.chat.completions.create(
        model=model, stream=True,
        messages=[{"role": "user", "content": "Count 1 to 8, comma separated."}],
    ):
        if chunk.choices and chunk.choices[0].delta.content:
            if first is None:
                first = time.time() - t0
            n += 1
            text += chunk.choices[0].delta.content
        if chunk.model_extra and chunk.model_extra.get("x_watchdog"):
            wd = chunk.model_extra["x_watchdog"]

    print(f"  chunks the SDK received : {n}    <- a faked stream would be 1")
    print(f"  client-observed TTFT    : {first * 1000:.0f}ms")
    if wd and wd.get("ttft_ms"):
        print(f"  server-recorded ttft_ms : {wd['ttft_ms']}ms   <- this is what lands in the ledger")
        print(f"  server latency_ms       : {wd['latency_ms']}ms")
        print(f"  TTFT as % of total      : {wd['ttft_ms'] / wd['latency_ms']:.0%}"
              f"    (a single buffered chunk would read 100%)")
    print(f"  text                    : {text.strip()[:50]}")


def proof_4_tools(client, model):
    header(4, "a real agent loop: tool call out, tool result back, final answer")
    tool = {"type": "function", "function": {
        "name": "get_weather", "description": "Get current weather for a city",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                       "required": ["city"]}}}

    messages = [{"role": "user", "content": "What's the weather in Paris? Use the tool."}]
    r = client.chat.completions.create(model=model, messages=messages, tools=[tool])
    choice = r.choices[0]

    if not choice.message.tool_calls:
        print(f"  the model answered without calling a tool: {choice.message.content[:60]}")
        return

    call = choice.message.tool_calls[0]
    print(f"  TURN 1  finish_reason={choice.finish_reason}")
    print(f"          tool={call.function.name}  args={call.function.arguments!r}")
    # The assertion behind the round-trip bug: arguments must be a JSON *string*
    # the SDK can parse, not an object the provider happened to hand back.
    print(f"          SDK parsed it -> {json.loads(call.function.arguments)}")
    print(f"          ${r.model_extra['x_watchdog']['cost_usd']}  {r.usage.total_tokens} tok")

    messages += [
        choice.message.model_dump(exclude_none=True),
        {"role": "tool", "tool_call_id": call.id, "content": "18C, clear skies"},
    ]
    r2 = client.chat.completions.create(model=model, messages=messages, tools=[tool])
    print(f"  TURN 2  finish_reason={r2.choices[0].finish_reason}")
    print(f"          answer: {(r2.choices[0].message.content or '').strip()[:70]}")
    print(f"          ${r2.model_extra['x_watchdog']['cost_usd']}  {r2.usage.total_tokens} tok")


def proof_5_guardrail(client, model):
    header(5, "the guardrail BLOCKS, it does not merely report")
    try:
        client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "this call must never reach a provider"}],
        )
        print("  !! the call went through. Start the server with WATCHDOG_GUARD_MODE=block")
        print("     and a tiny WEEKLY_BUDGET_USD for this proof to mean anything.")
    except openai.RateLimitError as exc:
        detail = exc.response.json()["detail"]
        print(f"  HTTP {exc.status_code}   (429, not 402: a self-imposed quota that will clear)")
        print(f"  type      : {detail['type']}")
        print(f"  guardrail : {detail['guardrail']}")
        print(f"  message   : {detail['message']}")
        print("  -> the stock SDK raised its ordinary RateLimitError. No custom handling.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:8000")
    ap.add_argument("--model", default="ollama/llama3.2:3b")
    ap.add_argument("--group", default="group:ladder")
    ap.add_argument("--key", default="wd-proof-demo")
    ap.add_argument("--prove-guard", action="store_true",
                    help="run the guardrail proof (needs WATCHDOG_GUARD_MODE=block)")
    args = ap.parse_args()

    client = OpenAI(base_url=f"{args.base}/v1", api_key=args.key, max_retries=0, timeout=600)

    try:
        client.models.list()
    except Exception as exc:
        print(f"no gateway at {args.base} ({exc}). See this file's docstring.")
        return 1

    if args.prove_guard:
        proof_5_guardrail(client, args.model)
        return 0

    proof_1_sdk(client, args.model)
    proof_2_routing(client, args.group)
    proof_3_streaming(client, args.model)
    proof_4_tools(client, args.model)

    print(f"\n{RULE}")
    print("PROOF 6 — everything above is now in the ledger")
    print(RULE)
    print(f"  {args.base}/calls?source=live      every call, with cost and latency")
    print(f"  {args.base}/report?source=live     totals by model and project")
    print(f"  {args.base}/router                 groups, cooldowns, active strategy")
    print(f"  {args.base}/shadow                 cheap-vs-real quality comparisons")
    print("\n  Guardrail proof (needs a server started to refuse work):")
    print("    WATCHDOG_GUARD_MODE=block WEEKLY_BUDGET_USD=0.0001 uvicorn dashboard.app:app")
    print("    venv/bin/python scripts/proof.py --prove-guard")
    return 0


if __name__ == "__main__":
    sys.exit(main())
