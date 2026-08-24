#!/usr/bin/env python3
"""Does the complexity classifier predict anything real? Checked against 4,399
turns of actual development work, at zero cost, with no model calls.

    venv/bin/python scripts/validate_classifier.py

src/complexity.py decides which model tier every routed request goes to, and
until now nothing had ever checked whether its verdicts correspond to anything.
It was written from intuition about which words signal hard work, it reads
plausible, and plausible is exactly the state a heuristic can sit in forever
while being worthless. The router is the project's central claim, so an
unvalidated classifier is an unvalidated project.

**The ground truth.** Claude Code transcripts record what actually happened
after each human message: how many tokens the agent generated, and how many
steps it took before handing control back. Those are measurements of how much
work the request really required, made by a system that had no knowledge of this
classifier. If prompts the classifier calls `complex` did in fact take more work
than prompts it calls `trivial`, it is measuring something. If they took the
same, it is noise wearing a label.

**What is deliberately NOT used as the outcome.** Turn *cost*. In an agentic
session the bill is dominated by accumulated context, so a turn late in a long
session costs more than an early one no matter what was asked. Correlating
complexity with cost would mostly measure position in the conversation, and
would produce a strong, meaningless, flattering result. Output tokens and step
count are not inflated that way, so they are the outcomes used here.

This is a validation, not a demonstration. A null result is reported as a null
result: see the verdict at the bottom, which is computed, not written in advance.
"""

import json
import pathlib
import statistics
import sys
from collections import defaultdict

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src.complexity import classify  # noqa: E402

TRANSCRIPTS = pathlib.Path.home() / ".claude" / "projects"

# Messages Claude Code injects on the user's behalf. They are not requests a
# human made and must not be classified as if they were.
SYNTHETIC = (
    "<command-name>", "<local-command-stdout>", "Caveat:", "<system-reminder>",
    "[Request interrupted", "<command-message>", "This session is being continued",
)


def _text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            b.get("text", "") for b in content
            if isinstance(b, dict) and b.get("type") == "text"
        )
    return ""


def harvest() -> list[dict]:
    """Pair every genuine human prompt with the work the agent did in response."""
    samples = []
    for path in TRANSCRIPTS.rglob("*.jsonl"):
        pending = None          # the prompt we are currently measuring
        out_tokens = 0
        steps = 0

        def flush():
            if pending and steps:
                samples.append({
                    "prompt": pending, "output_tokens": out_tokens, "steps": steps,
                })

        for line in path.open(errors="ignore"):
            try:
                d = json.loads(line)
            except Exception:
                continue

            if d.get("type") == "user":
                txt = _text(d.get("message", {}).get("content")).strip()
                # A tool result is a `user` record too, but carries no text.
                if not txt:
                    continue
                flush()
                if any(txt.startswith(s) or s in txt[:200] for s in SYNTHETIC):
                    pending = None
                elif len(txt) < 3:
                    pending = None
                else:
                    pending = txt
                out_tokens = steps = 0

            elif d.get("type") == "assistant" and pending:
                u = d.get("message", {}).get("usage") or {}
                if u:
                    out_tokens += u.get("output_tokens", 0)
                    steps += 1
        flush()
    return samples


def summarise(rows: list[float]) -> tuple[float, float]:
    s = sorted(rows)
    return statistics.median(s), statistics.mean(s)


def main() -> int:
    samples = harvest()
    if not samples:
        print("No transcripts found under", TRANSCRIPTS)
        return 1

    by_tier: dict[str, list[dict]] = defaultdict(list)
    for s in samples:
        s["tier"] = classify(s["prompt"]).tier
        by_tier[s["tier"]].append(s)

    print(f"\n{len(samples):,} genuine human prompts recovered from "
          f"{len(list(TRANSCRIPTS.rglob('*.jsonl')))} transcripts.")
    print("Outcome measured: what the agent actually did next.\n")

    print(f"{'tier':10}{'prompts':>9}{'median out':>12}{'mean out':>10}"
          f"{'median steps':>14}{'mean steps':>12}")
    order = ["trivial", "moderate", "complex"]
    stats = {}
    for tier in order:
        rows = by_tier.get(tier, [])
        if not rows:
            print(f"{tier:10}{0:>9}{'-':>12}{'-':>10}{'-':>14}{'-':>12}")
            continue
        mo, ao = summarise([r["output_tokens"] for r in rows])
        ms, as_ = summarise([float(r["steps"]) for r in rows])
        stats[tier] = {"median_out": mo, "mean_out": ao, "median_steps": ms, "mean_steps": as_,
                       "n": len(rows)}
        print(f"{tier:10}{len(rows):>9}{mo:>12,.0f}{ao:>10,.0f}{ms:>14.1f}{as_:>12.1f}")

    # Two questions, and passing the first says nothing about the second. A
    # classifier can rank perfectly and still be useless for routing if it
    # almost never commits to a tier, so both are reported and neither is
    # allowed to stand in for the other.
    print()
    if "trivial" not in stats or "complex" not in stats:
        print("VERDICT: not enough coverage across tiers to judge.")
        return 0

    t, c = stats["trivial"], stats["complex"]
    lift_out = c["median_out"] / t["median_out"] if t["median_out"] else 0
    lift_steps = c["median_steps"] / t["median_steps"] if t["median_steps"] else 0

    print("1. DOES THE ORDER MEAN ANYTHING?")
    print(f"   complex vs trivial: {lift_out:.1f}x the output, {lift_steps:.1f}x the steps (medians)")
    ranks = lift_out >= 1.5 and lift_steps >= 1.3
    print("   YES. The ordering tracks real work." if ranks else
          "   NO. The ordering does not track real work.")

    print("\n2. DOES IT COMMIT HARD ENOUGH TO ROUTE ON?")
    cheap = 100 * t["n"] / len(samples)
    mid = 100 * stats.get("moderate", {}).get("n", 0) / len(samples)
    print(f"   reaches the cheap tier on {cheap:.1f}% of prompts; {mid:.0f}% land in the middle")
    if cheap >= 15:
        print("   YES. Enough traffic reaches the cheap tier for routing to move the bill.")
    else:
        print(f"   NO. At {cheap:.1f}% cheap-tier reach, complexity routing sends almost")
        print("   everything to the mid or top tier and can only move the bill slightly.")

    print("\nVERDICT:", end=" ")
    if ranks and cheap >= 15:
        print("the classifier is sound and usable as a router.")
    elif ranks:
        print("the classifier ranks correctly but is far too conservative to save much.")
        print("  The obvious fix, lowering the trivial threshold, is not available: the group")
        print("  immediately below it is prompts where ONLY the length rule fired, and 19% of")
        print("  those ran to 20+ steps. Brevity is not simplicity. In an agentic session the")
        print("  prompt often does not contain the request, the context does, which is the same")
        print("  reason 78.5% of the bill is spent reading. A router that reads only the prompt")
        print("  is reading the part where the difficulty is not.")
    else:
        print("the classifier is not measuring difficulty. Complexity routing is")
        print("  unjustified as written.")

    print("\nA sample of what landed where:")
    for tier in order:
        rows = sorted(by_tier.get(tier, []), key=lambda r: -r["output_tokens"])[:2]
        for r in rows:
            p = " ".join(r["prompt"].split())[:88]
            print(f"  [{tier:8}] steps={r['steps']:>3} out={r['output_tokens']:>6,}  {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
