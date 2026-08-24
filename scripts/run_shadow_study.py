#!/usr/bin/env python3
"""Answer the question the router has been assuming: is the cheap model good enough?

    ollama serve
    venv/bin/python scripts/run_shadow_study.py

Everything in this project so far prices a substitution. Nothing has ever judged
one. `simulate_routing` re-prices tokens on a cheaper model, `analyse_cost_levers`
labels that result `hypothetical` and refuses to call it a saving, and the whole
point of src/shadow.py was to close that gap with evidence. It had no data, so
the gap stayed open and every claim about routing stayed conditional.

This runs it. Two local models, so it costs nothing:

    capable   ollama/llama3.2:3b     stands in for the expensive tier
    cheap     ollama/llama3.2:1b     stands in for what routing would pick

Each prompt goes to both. src/judge.py grades the pair with deterministic checks
first (does the code parse, is the answer empty, is it a fraction of the length)
and a local judge for the rest, blind and order-randomised.

**What this can and cannot show.** Two small local models are not Opus and Haiku,
so a verdict here does not transfer to a frontier pair. What it does test is the
mechanism: whether prompts the classifier calls `trivial` really do survive being
handed to a weaker model, and whether prompts it calls `complex` really do
degrade. That is the assumption routing rests on, and it is testable with any
capability gap at all.
"""

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

CAPABLE = "ollama/llama3.2:3b"
CHEAP = "ollama/llama3.2:1b"

os.environ.setdefault("WATCHDOG_SHADOW_RATE", "1.0")
os.environ.setdefault("WATCHDOG_SHADOW_MODEL", CHEAP)
os.environ.setdefault("WATCHDOG_GUARD_MODE", "off")

# Chosen so each tier is represented and every prompt has a short right answer.
# Long generations would make this take an hour on a laptop without making the
# comparison any sharper: the question is whether the answer holds up, not
# whether a 1B model can write an essay.
PROMPTS = [
    # trivial: mechanical, one obvious right answer
    ("Reverse the string 'hello' in Python. One line, no explanation.", "trivial"),
    ("Convert this to uppercase in Python: name = 'ada'. One line.", "trivial"),
    ("Write a Python one-liner that sums a list called nums.", "trivial"),
    ("Rename the variable `x` to `user_id` in: x = get_user()", "trivial"),
    ("Format this as valid JSON: name ada, age 36", "trivial"),
    ("What does the len() function do in Python? One sentence.", "trivial"),
    # moderate: needs a little judgement
    ("Write a Python function that returns True if a string is a palindrome.", "moderate"),
    ("How do I read a file line by line in Python without loading it all into memory?", "moderate"),
    ("What is the difference between a list and a tuple in Python? Two sentences.", "moderate"),
    ("Write a function that retries a call three times with a delay between attempts.", "moderate"),
    ("Why might a Python dict lookup raise KeyError, and how do I avoid it?", "moderate"),
    # complex: reasoning, trade-offs, more than one defensible answer
    ("Explain why using a mutable default argument in Python is a bug, with an example.", "complex"),
    ("My Flask app gets slower under load. Name three likely causes and how to tell them apart.", "complex"),
    ("Design a rate limiter for an API: per user, across several servers. Explain the trade-offs.", "complex"),
]


def main() -> int:
    from src.complexity import classify
    from src.shadow import enabled, run_shadow, shadow_summary
    from src.utils import call_llm_detailed

    if not enabled():
        print("shadow is disabled; set WATCHDOG_SHADOW_RATE=1.0")
        return 1

    print(f"capable: {CAPABLE}\ncheap:   {CHEAP}\n{len(PROMPTS)} prompts\n")
    started = time.time()
    done = 0

    for i, (prompt, expected) in enumerate(PROMPTS, 1):
        tier = classify(prompt).tier
        label = f"[{i}/{len(PROMPTS)}] {tier:8}"
        try:
            t0 = time.time()
            result = call_llm_detailed(prompt, model=CAPABLE, max_retries=0)
            took = time.time() - t0
        except Exception as exc:
            print(f"{label} capable model failed: {str(exc)[:70]}")
            continue

        try:
            # Synchronous here, unlike the gateway's fire-and-forget path: this
            # is a study and an un-run comparison is a missing data point, not a
            # request that was merely a little slower.
            run_shadow(
                prompt=prompt,
                real_model=CAPABLE,
                real_response=result.text,
                real_cost_usd=result.event.cost_usd,
                real_latency_ms=result.event.latency_ms,
                project_tag="shadow-study",
            )
            done += 1
            print(f"{label} ok  ({took:.0f}s)  {prompt[:52]}")
        except Exception as exc:
            print(f"{label} shadow failed: {str(exc)[:70]}")

    print(f"\n{done}/{len(PROMPTS)} pairs recorded in {time.time() - started:.0f}s")
    summary = shadow_summary()
    print(f"pending grading: {summary.get('pending', summary.get('total', '?'))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
