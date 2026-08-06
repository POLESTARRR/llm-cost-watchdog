"""
Waste detection — spend that bought you nothing, and spend you could stop
buying tomorrow.

A cost report tells you *where* money went. It doesn't tell you which of it
was avoidable. These four checks do, and each one ends in a concrete action
rather than an observation:

  1. Retry waste      — failed attempts, and the latency you paid for them
  2. Duplicate calls  — the same prompt sent repeatedly
  3. Cache misses     — repeated long prefixes never cached
  4. Over-powered     — a frontier model doing trivial-length work

Everything here is computed from data already in SQLite. No API calls, no
extra cost to run.
"""

from collections import Counter, defaultdict

from src.pricing import PRICING_TABLE, calculate_cost, get_rates
from src.tracker import get_events_for_period

# A prompt preview is 80 chars; two calls sharing one are near-certainly the
# same prompt. Short previews (a truncated system header) are excluded to
# avoid false positives on prompts that merely start alike.
_MIN_PREVIEW_FOR_DUPLICATE = 40

# Below this, a call is "trivial-length" — a frontier model is likely overkill.
_TRIVIAL_OUTPUT_TOKENS = 120
_TRIVIAL_INPUT_TOKENS = 800

# Models considered frontier-tier for the over-powered check.
_FRONTIER_PREFIXES = ("claude-opus", "claude-fable", "gpt-5.6-sol", "gpt-5.5", "gemini-pro")

# Cheapest sensible substitute per provider, for the "you could have used"
# suggestion. Deliberately conservative — a real downgrade suggestion, not
# the absolute cheapest model on the market.
_DOWNGRADE = {
    "anthropic": "claude-haiku-4-5",
    "openai": "gpt-5-nano",
    "google": "gemini-flash-lite-latest",
}


def find_retry_waste(period: str = "week", source: str | None = None) -> dict:
    """Money and time spent on calls that returned nothing.

    Failed calls usually cost $0 directly, but they are not free: they burn
    latency, they consume rate-limit budget, and a high failure rate on one
    model is a signal to switch or back off harder.
    """
    events = get_events_for_period(period, source=source)
    failed = [e for e in events if not e.success]

    by_model: dict[str, int] = Counter(e.model for e in failed)
    by_project: dict[str, int] = Counter(e.project_tag for e in failed)

    wasted_ms = sum(e.latency_ms for e in failed)
    wasted_cost = sum(e.cost_usd for e in failed)

    return {
        "period": period,
        "failed_calls": len(failed),
        "total_calls": len(events),
        "failure_rate": round(len(failed) / len(events), 4) if events else 0.0,
        "wasted_cost_usd": round(wasted_cost, 6),
        "wasted_seconds": round(wasted_ms / 1000, 1),
        "failures_by_model": dict(by_model),
        "failures_by_project": dict(by_project),
        "recommendation": _retry_recommendation(len(failed), len(events), by_model),
    }


def _retry_recommendation(failed: int, total: int, by_model: Counter) -> str:
    if not failed:
        return "No failed calls in this period."
    rate = failed / total if total else 0
    worst = by_model.most_common(1)[0][0] if by_model else "unknown"
    if rate > 0.25:
        return (
            f"{rate:.0%} of calls failed — most on {worst}. That is high enough to be "
            f"structural, not transient: check quota tier, or route this traffic to another provider."
        )
    if rate > 0.05:
        return (
            f"{rate:.0%} of calls failed, mostly on {worst}. Worth raising backoff "
            f"(LLM_MAX_RETRIES) or spreading load across providers."
        )
    return f"{failed} failed call(s) on {worst} — within normal transient range."


def find_duplicate_calls(
    period: str = "week", min_repeats: int = 2, source: str | None = None
) -> list[dict]:
    """Identical prompts sent more than once.

    The cheapest token is the one you don't send. A prompt repeated verbatim
    is either a missing local cache or a loop re-asking the same question.
    """
    events = [
        e for e in get_events_for_period(period, source=source)
        if e.success and len(e.prompt_preview) >= _MIN_PREVIEW_FOR_DUPLICATE
    ]

    groups: dict[tuple[str, str], list] = defaultdict(list)
    for e in events:
        groups[(e.model, e.prompt_preview)].append(e)

    rows = []
    for (model, preview), items in groups.items():
        if len(items) < min_repeats:
            continue
        total = sum(e.cost_usd for e in items)
        # Everything after the first send is the avoidable part.
        avoidable = total - (total / len(items))
        rows.append({
            "model": model,
            "provider": items[0].provider,
            "prompt_preview": preview,
            "times_sent": len(items),
            "total_cost_usd": round(total, 6),
            "avoidable_cost_usd": round(avoidable, 6),
            "projects": sorted({e.project_tag for e in items}),
        })

    rows.sort(key=lambda r: r["avoidable_cost_usd"], reverse=True)
    return rows


def find_cache_opportunities(
    period: str = "week", min_input_tokens: int = 1024, source: str | None = None
) -> list[dict]:
    """Repeated large prompts that were never served from cache.

    Prompt caching is the biggest single lever on real LLM spend, and it is
    opt-in on every provider — which means it is usually just left off. This
    finds the traffic where turning it on would actually pay, and prices the
    saving using the model's real cached rate.
    """
    events = [
        e for e in get_events_for_period(period, source=source)
        if e.success and e.input_tokens >= min_input_tokens
    ]

    groups: dict[tuple[str, str], list] = defaultdict(list)
    for e in events:
        groups[(e.model, e.project_tag)].append(e)

    rows = []
    for (model, project), items in groups.items():
        if len(items) < 2:
            continue  # caching needs a repeat to pay off
        already_cached = sum(e.cached_input_tokens for e in items)
        total_input = sum(e.input_tokens for e in items)
        if total_input and already_cached / total_input > 0.5:
            continue  # already caching well

        rates = get_rates(model)
        # Conservative: assume only the repeats (not the first call) could be
        # served from cache, and only the portion not already cached.
        cacheable = (total_input - already_cached) * (len(items) - 1) / len(items)
        saving = (cacheable / 1000) * (rates["input"] - rates["cached_input"])
        if saving < 1e-6:
            continue

        rows.append({
            "model": model,
            "project_tag": project,
            "calls": len(items),
            "total_input_tokens": total_input,
            "currently_cached_tokens": already_cached,
            "cache_hit_rate": round(already_cached / total_input, 4) if total_input else 0.0,
            "estimated_saving_usd": round(saving, 6),
            "action": f"Add prompt caching to the shared prefix of {project}'s {model} calls.",
        })

    rows.sort(key=lambda r: r["estimated_saving_usd"], reverse=True)
    return rows


def find_overpowered_calls(period: str = "week", source: str | None = None) -> list[dict]:
    """Frontier models doing work a cheap model would have handled.

    Heuristic, and labeled as one: short input AND short output on an
    expensive model. It cannot know the task was easy — only that the shape
    of the call looks trivial. Treat it as a prompt to review, not a verdict.
    """
    events = [
        e for e in get_events_for_period(period, source=source)
        if e.success
        and e.model.startswith(_FRONTIER_PREFIXES)
        and e.output_tokens <= _TRIVIAL_OUTPUT_TOKENS
        and e.input_tokens <= _TRIVIAL_INPUT_TOKENS
    ]

    groups: dict[tuple[str, str], list] = defaultdict(list)
    for e in events:
        groups[(e.model, e.project_tag)].append(e)

    rows = []
    for (model, project), items in groups.items():
        alt = _DOWNGRADE.get(items[0].provider)
        if not alt or alt not in PRICING_TABLE:
            continue

        actual = sum(e.cost_usd for e in items)
        alternative = sum(
            calculate_cost(alt, e.input_tokens, e.output_tokens, e.cached_input_tokens)
            for e in items
        )
        if alternative >= actual:
            continue

        rows.append({
            "model": model,
            "project_tag": project,
            "calls": len(items),
            "actual_cost_usd": round(actual, 6),
            "suggested_model": alt,
            "suggested_cost_usd": round(alternative, 6),
            "estimated_saving_usd": round(actual - alternative, 6),
            "basis": (
                f"{len(items)} call(s) averaged "
                f"{sum(e.input_tokens for e in items)//len(items)} in / "
                f"{sum(e.output_tokens for e in items)//len(items)} out — trivial-length work "
                f"for a frontier model."
            ),
            "confidence": "heuristic — review before switching; short output does not always mean easy task",
        })

    rows.sort(key=lambda r: r["estimated_saving_usd"], reverse=True)
    return rows


def find_waste(period: str = "week", source: str | None = None) -> dict:
    """Run every waste check and total the recoverable spend."""
    retries = find_retry_waste(period, source=source)
    duplicates = find_duplicate_calls(period, source=source)
    cache_ops = find_cache_opportunities(period, source=source)
    overpowered = find_overpowered_calls(period, source=source)

    by_category = {
        "duplicate_calls": sum(r["avoidable_cost_usd"] for r in duplicates),
        "cache_opportunities": sum(r["estimated_saving_usd"] for r in cache_ops),
        "overpowered_calls": sum(r["estimated_saving_usd"] for r in overpowered),
    }

    total = compute_total(period, source=source)

    # The categories OVERLAP — a duplicated call is also a cache opportunity,
    # and both may also be over-powered. Summing them naively can exceed total
    # spend (observed: 114%), which is obviously wrong and would make the whole
    # number untrustworthy. So this is reported as an upper bound, capped at
    # actual spend, with the per-category detail alongside for the real picture.
    naive_sum = sum(by_category.values())
    recoverable = min(naive_sum, total)

    pct = round((recoverable / total) * 100, 2) if total else 0.0

    return {
        "period": period,
        "source_filter": source,
        "total_spend_usd": round(total, 6),
        "recoverable_usd": round(recoverable, 6),
        "recoverable_percent": pct,
        "recoverable_is_upper_bound": True,
        "recoverable_by_category": {k: round(v, 6) for k, v in by_category.items()},
        "note": (
            "Categories overlap (a duplicated call is also a cache opportunity), "
            "so the total is an upper bound capped at actual spend, not a sum of "
            "independent savings."
        ),
        "retry_waste": retries,
        "duplicate_calls": duplicates,
        "cache_opportunities": cache_ops,
        "overpowered_calls": overpowered,
        "top_action": _top_action(duplicates, cache_ops, overpowered, retries),
    }


def compute_total(period: str, source: str | None = None) -> float:
    return sum(e.cost_usd for e in get_events_for_period(period, source=source))


def _top_action(duplicates, cache_ops, overpowered, retries) -> str:
    """The single highest-value thing to do, in plain language."""
    candidates = []
    if cache_ops:
        c = cache_ops[0]
        candidates.append((c["estimated_saving_usd"],
                           f"Enable prompt caching for {c['project_tag']}'s {c['model']} calls "
                           f"(~${c['estimated_saving_usd']:.4f} this period)."))
    if duplicates:
        d = duplicates[0]
        candidates.append((d["avoidable_cost_usd"],
                           f"Cache or dedupe a prompt sent {d['times_sent']}x to {d['model']} "
                           f"(~${d['avoidable_cost_usd']:.4f} this period)."))
    if overpowered:
        o = overpowered[0]
        candidates.append((o["estimated_saving_usd"],
                           f"Consider {o['suggested_model']} instead of {o['model']} for "
                           f"{o['project_tag']} (~${o['estimated_saving_usd']:.4f} this period)."))

    if candidates:
        return max(candidates, key=lambda c: c[0])[1]
    if retries["failed_calls"]:
        return retries["recommendation"]
    return "No obvious waste found in this period."
