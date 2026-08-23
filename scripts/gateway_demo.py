#!/usr/bin/env python3
"""End-to-end proof that the gateway routes, prices and records real traffic.

Run it against a live server and it makes **real calls** through the
OpenAI-compatible endpoint, then prints what the ledger recorded. Nothing here
is mocked, which is the point: every other demo in this repo replays stored
data, and stored data is exactly what could not tell you whether the live path
worked.

    # terminal 1
    WATCHDOG_GROUP_LADDER=ollama/llama3.2:3b,gemini-flash-lite-latest,gemini-pro-latest \
    WATCHDOG_ROUTING_STRATEGY=complexity \
    venv/bin/python -m uvicorn dashboard.app:app --port 8000

    # terminal 2
    venv/bin/python scripts/gateway_demo.py

Requires at least one configured provider. With only Ollama running it still
works end to end and every call costs $0.00, which is the cheapest possible way
to prove the path is real.
"""

import argparse
import json
import sys
import urllib.error
import urllib.request

# Deliberately spans the classifier's range: two that should reach the cheap
# tier, two that should not, and one that is genuinely ambiguous.
PROMPTS = [
    "reformat this JSON: {'a':1,'b':2}",
    "rename the variable `tmp` to `buffer`",
    "Can you take a look at the user service?",
    "Why is this test flaky? Walk me through the possible race conditions.",
    "Design a zero-downtime schema migration strategy and explain the trade-offs "
    "of each approach you considered.",
]


def call(base: str, model: str, prompt: str, key: str, timeout: float) -> dict:
    req = urllib.request.Request(
        f"{base}/v1/chat/completions",
        data=json.dumps({"model": model, "messages": [{"role": "user", "content": prompt}]}).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.load(resp)
    except urllib.error.HTTPError as exc:
        return {"error": json.loads(exc.read() or b"{}").get("detail", str(exc)), "status": exc.code}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:8000")
    ap.add_argument("--model", default="group:ladder", help="a model id, or group:<name>")
    ap.add_argument("--key", default="wd-gateway-demo", help="wd-<project> names the ledger project")
    ap.add_argument("--timeout", type=float, default=300)
    args = ap.parse_args()

    try:
        urllib.request.urlopen(f"{args.base}/health", timeout=5)
    except Exception as exc:
        print(f"no server at {args.base} ({exc}). Start one first, see this file's docstring.")
        return 1

    print(f"gateway: {args.base}   model: {args.model}\n")
    total_cost = 0.0
    fallbacks = 0

    for prompt in PROMPTS:
        body = call(args.base, args.model, prompt, args.key, args.timeout)
        print(f"  {prompt[:64]}")

        if "error" in body:
            print(f"    REFUSED ({body['status']}): {body['error']}\n")
            continue

        wd = body["x_watchdog"]
        routing = wd.get("routing") or {}
        tier = (routing.get("complexity") or {}).get("tier", "-")
        sent, landed = routing.get("model", body["model"]), body["model"]
        total_cost += wd["cost_usd"]

        line = f"    tier={tier:<8} -> {landed}"
        if routing.get("fell_back_from"):
            fallbacks += 1
            # Worth printing loudly: a response from a model the routing record
            # doesn't name looks like a bug until you can see the substitution.
            line += f"   (FELL BACK from {sent}: {routing['fell_back_reason']})"
        print(line)
        print(f"    ${wd['cost_usd']:.6f}  {wd['latency_ms']:.0f}ms  "
              f"{body['usage']['total_tokens']} tok  project={wd['project']}")
        print(f"    {body['choices'][0]['message']['content'][:70].strip()}...\n")

    print(f"total: ${total_cost:.6f} across {len(PROMPTS)} calls, {fallbacks} fallback(s)")
    print(f"\nledger:   {args.base}/calls?source=live")
    print(f"routing:  {args.base}/router")
    print(f"shadow:   {args.base}/shadow")
    return 0


if __name__ == "__main__":
    sys.exit(main())
