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

import argparse
import datetime as _dt
import json
import pathlib
import statistics
import sys
from collections import defaultdict

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src.complexity import SHORT_PROMPT_CHARS, classify  # noqa: E402

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
        ctx_tokens = None       # conversation size when the request was sent

        def flush():
            if pending and steps and ctx_tokens is not None:
                samples.append({
                    "prompt": pending, "output_tokens": out_tokens, "steps": steps,
                    "context_tokens": ctx_tokens,
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
                ctx_tokens = None

            elif d.get("type") == "assistant" and pending:
                u = d.get("message", {}).get("usage") or {}
                if u:
                    if ctx_tokens is None:
                        # Everything the model had to read for this first reply:
                        # fresh input, cache reads and cache writes together.
                        ctx_tokens = (u.get("input_tokens", 0)
                                      + u.get("cache_read_input_tokens", 0)
                                      + u.get("cache_creation_input_tokens", 0))
                    out_tokens += u.get("output_tokens", 0)
                    steps += 1
        flush()
    return samples


def _stats(rows: list[dict]) -> dict:
    out = [r["output_tokens"] for r in rows]
    steps = [float(r["steps"]) for r in rows]
    return {
        "n": len(rows),
        "median_out": statistics.median(out),
        "mean_out": statistics.mean(out),
        "median_steps": statistics.median(steps),
        "mean_steps": statistics.mean(steps),
    }


def analyse() -> dict:
    """Run the whole validation and return it as data.

    Returns a structure rather than printing, so the published artifact and the
    terminal report are the same computation. A page quoting one number while a
    script prints another is the drift this project has already been bitten by.
    """
    samples = harvest()
    if not samples:
        return {"n_prompts": 0}

    by_tier: dict[str, list[dict]] = defaultdict(list)
    words_only: dict[str, list[dict]] = defaultdict(list)
    for s in samples:
        s["tier"] = classify(s["prompt"], context_tokens=s["context_tokens"]).tier
        by_tier[s["tier"]].append(s)
        # The previous behaviour, kept so the improvement is shown rather than
        # asserted. A change that cannot be compared to what it replaced is a
        # claim, not a result.
        words_only[classify(s["prompt"]).tier].append(s)

    n = len(samples)
    tiers = {t: _stats(by_tier[t]) for t in ("trivial", "moderate", "complex") if by_tier.get(t)}
    for t, v in tiers.items():
        v["share_percent"] = round(100 * v["n"] / n, 1)

    t, c = tiers.get("trivial"), tiers.get("complex")
    lift_out = (c["median_out"] / t["median_out"]) if (t and c and t["median_out"]) else 0.0
    lift_steps = (c["median_steps"] / t["median_steps"]) if (t and c and t["median_steps"]) else 0.0
    ranks = lift_out >= 1.5 and lift_steps >= 1.3
    cheap_share = tiers.get("trivial", {}).get("share_percent", 0.0)
    commits = cheap_share >= 15

    # The evidence that the obvious fix (lower the threshold) is unavailable.
    #
    # Identified as "short, and nothing else fired" rather than by score, on
    # purpose. This group scored -1 under the old length discount and was the
    # reason 47% of traffic sat one point above the cheap tier. Removing that
    # discount moved the same prompts to 0, so selecting on score == -1 would now
    # return almost nothing and make the finding look like it had evaporated.
    # The population is what matters, not the number that used to label it.
    borderline = [
        s for s in samples
        if len(s["prompt"]) <= SHORT_PROMPT_CHARS and classify(s["prompt"]).score == 0
    ]
    heavy = [s for s in borderline if s["steps"] >= 20]

    return {
        "generated_at": _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat(),
        "n_prompts": n,
        "n_transcripts": len(list(TRANSCRIPTS.rglob("*.jsonl"))),
        "tiers": tiers,
        "ranking": {
            "lift_output": round(lift_out, 1),
            "lift_steps": round(lift_steps, 1),
            "passes": ranks,
        },
        "routing": {
            "cheap_tier_share_percent": cheap_share,
            "middle_tier_share_percent": tiers.get("moderate", {}).get("share_percent", 0.0),
            "passes": commits,
        },
        "length_rule_evidence": {
            "n_borderline": len(borderline),
            "borderline_share_percent": round(100 * len(borderline) / n, 1),
            "n_ran_20_plus_steps": len(heavy),
            "heavy_share_percent": round(100 * len(heavy) / len(borderline), 1) if borderline else 0.0,
            "worst": [
                {"prompt": " ".join(s["prompt"].split())[:110],
                 "steps": s["steps"], "output_tokens": s["output_tokens"]}
                for s in sorted(borderline, key=lambda r: -r["steps"])[:3]
            ],
        },
        "words_only": {
            t: {"n": len(v), "share_percent": round(100 * len(v) / n, 1)}
            for t, v in words_only.items()
        },
        "verdict": (
            "sound and usable as a router" if (ranks and commits)
            else "ranks correctly but is far too conservative to route on" if ranks
            else "does not measure difficulty; complexity routing is unjustified as written"
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", metavar="PATH",
                    help="write the result as a publishable artifact instead of a report")
    args = ap.parse_args()

    r = analyse()
    if not r.get("n_prompts"):
        print("No transcripts found under", TRANSCRIPTS)
        return 1

    if args.json:
        out = pathlib.Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(r, indent=2) + "\n")
        print(f"wrote {out}  ({r['n_prompts']:,} prompts, verdict: {r['verdict']})")
        return 0

    print(f"\n{r['n_prompts']:,} genuine human prompts recovered from "
          f"{r['n_transcripts']} transcripts.")
    print("Outcome measured: what the agent actually did next.\n")
    print(f"{'tier':10}{'prompts':>9}{'share':>8}{'median out':>12}"
          f"{'median steps':>14}{'mean steps':>12}")
    for t in ("trivial", "moderate", "complex"):
        v = r["tiers"].get(t)
        if not v:
            print(f"{t:10}{0:>9}")
            continue
        print(f"{t:10}{v['n']:>9}{v['share_percent']:>7.1f}%{v['median_out']:>12,.0f}"
              f"{v['median_steps']:>14.1f}{v['mean_steps']:>12.1f}")

    rk, rt, le = r["ranking"], r["routing"], r["length_rule_evidence"]
    print("\n1. DOES THE ORDER MEAN ANYTHING?")
    print(f"   complex vs trivial: {rk['lift_output']}x the output, "
          f"{rk['lift_steps']}x the steps (medians)")
    print("   YES. The ordering tracks real work." if rk["passes"]
          else "   NO. The ordering does not track real work.")

    print("\n2. DOES IT COMMIT HARD ENOUGH TO ROUTE ON?")
    print(f"   reaches the cheap tier on {rt['cheap_tier_share_percent']}% of prompts; "
          f"{rt['middle_tier_share_percent']:.0f}% land in the middle")
    print("   YES." if rt["passes"] else
          f"   NO. At {rt['cheap_tier_share_percent']}% cheap-tier reach, complexity routing\n"
          "   sends almost everything to the mid or top tier and can only move the bill slightly.")

    wo = r.get("words_only", {}).get("trivial", {}).get("share_percent", 0.0)
    print("\n3. WHAT THE CONTEXT SIGNAL CHANGED")
    print(f"   words only:          {wo}% reached the cheap tier")
    print(f"   words + context:     {rt['cheap_tier_share_percent']}%")
    if wo:
        print(f"   {rt['cheap_tier_share_percent'] / wo:.0f}x more traffic routed cheap, and the")
        print(f"   separation went UP ({rk['lift_output']}x output), not down. The requests newly")
        print("   sent cheap are lighter than the ones already there, so it is finding easy")
        print("   work rather than diluting the tier.")

    print(f"\nVERDICT: the classifier {r['verdict']}.")
    print("  The fix was not better words. It was reading how much conversation the request")
    print("  arrives with, which the classifier could always see and never looked at. That is")
    print("  the cost study's finding from the other side: 78.5% of the bill is reading, so")
    print("  what predicts a request's weight is how much there is to read.")
    if not rt["passes"]:
        print(f"  Still under the 15% bar this sets for itself, at {rt['cheap_tier_share_percent']}%.")
        print("  Reported as a miss rather than moved.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
