"""
MCP server exposing the watchdog over stdio, for local Claude Desktop use.

Run directly:  python -m src.mcp_server
"""

from mcp.server.mcpserver import MCPServer

from src.analyzer import check_budget_status as _check_budget_status
from src.analyzer import compute_report
from src.analyzer import flag_anomalies as _flag_anomalies
from src.analyzer import project_burn_rate as _project_burn_rate
from src.analyzer import provider_breakdown as _provider_breakdown
from src.analyzer import subscription_roi as _subscription_roi
from src.analyzer import what_if_switched as _what_if_switched
from src.guard import guard_status as _guard_status
from src.pricing import PRICING_TABLE, compare_models
from src.pricing_drift import check_drift as _check_drift
from src.providers import configured_providers, infer_provider
from src.router import router_status as _router_status
from src.router import simulate_routing as _simulate_routing
from src.waste import find_waste as _find_waste
from src.tracker import VALID_SOURCES, log_usage
from src.tracker import source_totals as _source_totals
from src.usage_schema import UsageEvent

server = MCPServer(
    name="llm-cost-gateway",
    instructions=(
        "Tracks LLM API cost, tokens, latency, and prompt-cache usage across "
        "Anthropic, OpenAI, and Google. Use these tools to report spend, check "
        "budget and burn rate, surface anomalous calls, compare model pricing, "
        "and log calls made outside the tracked wrapper. Every row carries a "
        "provenance ('live' = a real billed API call, 'demo' = seeded sample "
        "data, 'manual' = hand-entered, 'subscription' = real tokens covered by "
        "a flat-fee plan rather than metered billing), use get_data_provenance "
        "before quoting a total as real money, and prefer get_subscription_roi "
        "when the usage came from a Claude Pro/Max plan, because no per-token "
        "charge occurred for those rows."
    ),
)

_VALID_PERIODS = {"today", "week", "month", "all_time"}


def _bad_period(period: str) -> dict:
    return {"error": f"invalid period {period!r}; expected one of {sorted(_VALID_PERIODS)}"}


def _bad_source(source: str) -> dict:
    return {"error": f"invalid source {source!r}; expected any of {list(VALID_SOURCES)}, "
                     f"a comma-separated set of them, or 'all'"}


def _check_source(source: str | None) -> dict | None:
    """Return an error dict if `source` is unusable, else None."""
    if source is None:
        return None
    parts = [p.strip() for p in source.strip().lower().split(",") if p.strip()]
    if parts == ["all"] or not parts:
        return None
    return None if all(p in VALID_SOURCES for p in parts) else _bad_source(source)


@server.tool(
    description="Get an LLM cost report for a period. period: 'today' | 'week' | 'month' | "
                "'all_time'. Returns total cost, call count, failures, prompt-cache savings, "
                "and breakdowns by model, project, provider, and data source. Pass "
                "source='live' for real billed calls only, 'live,manual' for everything you "
                "were charged for, or leave unset to include seeded demo data. Always read "
                "breakdown_by_source before describing the total as real spend."
)
def get_cost_report(period: str = "week", source: str | None = None) -> dict:
    if period not in _VALID_PERIODS:
        return _bad_period(period)
    if (err := _check_source(source)) is not None:
        return err
    return compute_report(period, source=source).model_dump()


@server.tool(
    description="Show how much of the recorded spend is real. Splits cost and call count by "
                "provenance: 'live' (an actual billed API call), 'manual' (hand-entered, real "
                "but unverified), and 'demo' (seeded sample data that cost nothing). Use this "
                "to answer 'is this number real?' and before quoting any total as money spent."
)
def get_data_provenance(period: str = "all_time") -> dict:
    if period not in _VALID_PERIODS:
        return _bad_period(period)
    return _source_totals(period)


@server.tool(
    description="Check current spend against the configured weekly budget. Returns status "
                "('under' | 'near' | 'over'), percent used, and remaining USD."
)
def check_budget_status() -> dict:
    return _check_budget_status("weekly")


@server.tool(
    description="Project whether current spend will exceed the budget. Returns daily burn rate, "
                "projected weekly total, the date the budget runs out, and a confidence level "
                "based on how much data the projection rests on."
)
def get_burn_rate(period: str = "week") -> dict:
    if period not in _VALID_PERIODS:
        return _bad_period(period)
    return _project_burn_rate(period)


@server.tool(
    description="Find calls that were unusually expensive or slow compared to the trailing "
                "20-call rolling average for the same model. Returns each flagged call with a reason."
)
def flag_anomalies(threshold_multiplier: float = 3.0) -> list[dict]:
    return [a.model_dump() for a in _flag_anomalies(threshold_multiplier)]


@server.tool(
    description="Break spend down by provider (anthropic / openai / google): cost, call volume, "
                "tokens, prompt-cache hit rate, average latency, and which models were used. "
                "Each row reports live_calls, if that is 0 the provider's numbers are seeded "
                "demo data and its adapter has never run against a live endpoint."
)
def get_provider_breakdown(period: str = "week", source: str | None = None) -> list[dict]:
    if period not in _VALID_PERIODS:
        return [_bad_period(period)]
    if (err := _check_source(source)) is not None:
        return [err]
    return _provider_breakdown(period, source=source)


@server.tool(
    description="Estimate and compare what a call of a given size would cost across models, "
                "cheapest first. Makes no API calls, pure pricing arithmetic. Optionally pass "
                "a list of model IDs to restrict the comparison."
)
def compare_model_costs(
    input_tokens: int,
    output_tokens: int,
    models: list[str] | None = None,
) -> list[dict]:
    unknown = [m for m in (models or []) if m not in PRICING_TABLE]
    rows = compare_models(input_tokens, output_tokens, models)
    if unknown:
        rows.append({"warning": f"not in pricing table, priced at fallback rates: {unknown}"})
    return rows


@server.tool(
    description="Re-price this period's real traffic for one model as if it had run on another. "
                "Answers 'should I switch models?' using your own token counts rather than a benchmark."
)
def what_if_switched(from_model: str, to_model: str, period: str = "week") -> dict:
    if period not in _VALID_PERIODS:
        return _bad_period(period)
    return _what_if_switched(from_model, to_model, period)


@server.tool(
    description="List which LLM providers are configured with credentials, and which models "
                "the watchdog knows how to price."
)
def list_providers() -> dict:
    grouped: dict[str, list[str]] = {}
    for model in PRICING_TABLE:
        try:
            grouped.setdefault(infer_provider(model), []).append(model)
        except Exception:
            grouped.setdefault("unknown", []).append(model)
    return {"configured": configured_providers(), "priced_models_by_provider": grouped}


@server.tool(
    description="Find spend that bought you nothing or could be stopped: failed-call waste, "
                "duplicate prompts, missed prompt-caching opportunities, frontier models doing "
                "trivial work, and real traffic on a model with a cheaper same-vendor sibling "
                "(re-priced on your actual calls, not gated by call length). Each finding ends "
                "in a concrete action."
)
def find_waste(period: str = "week", source: str | None = None) -> dict:
    if period not in _VALID_PERIODS:
        return _bad_period(period)
    if (err := _check_source(source)) is not None:
        return err
    return _find_waste(period, source=source)


@server.tool(
    description="Check the spend guardrails: current mode (off/warn/block), how much budget "
                "headroom is left, call-rate against the runaway-loop circuit breaker, and any "
                "per-project caps. Use this to ask 'how much runway do I have?' before being blocked."
)
def check_guard_status() -> dict:
    return _guard_status()


@server.tool(
    description="Log an LLM call made outside the tracked call_llm() wrapper, e.g. one made "
                "through a web UI or another tool. Provider is inferred from the model ID. "
                "Recorded with source='manual': real spend, but reported rather than measured."
)
def log_manual_entry(
    model: str,
    cost_usd: float,
    tokens: int,
    project_tag: str,
    note: str = "",
) -> str:
    try:
        provider = infer_provider(model)
    except Exception:
        provider = "unknown"

    event = UsageEvent(
        model=model,
        provider=provider,
        project_tag=project_tag,
        input_tokens=0,
        output_tokens=tokens,
        cost_usd=cost_usd,
        latency_ms=0.0,
        prompt_preview=UsageEvent.make_preview(note or "manual entry"),
        success=True,
        source="manual",
    )
    log_usage(event)
    return f"✓ logged as source=manual | {model} ${cost_usd:.6f} project={project_tag}"


@server.tool(
    description="What a flat-fee Claude Pro/Max subscription returned in list-price API "
                "value. Use this instead of get_cost_report when asked 'what did my "
                "subscription get me' or 'was it worth it'. Returns the list-price value of "
                "tokens consumed, the subscription cost for the span they cover, and the "
                "ratio. Critically: no per-token charge occurred for these rows, so this is "
                "value delivered, not money spent, never describe it as billed spend."
)
def get_subscription_roi(period: str = "all_time") -> dict:
    if period not in _VALID_PERIODS:
        return _bad_period(period)
    return _subscription_roi(period)


@server.tool(
    description="Show the router's current configuration: declared model groups, the active "
                "routing strategy, which models are cooling down after a rate limit and for "
                "how long, and any group member missing from the pricing table. Read-only, "
                "dispatches nothing."
)
def get_router_status() -> dict:
    return _router_status()


@server.tool(
    description="Replay recorded traffic through a routing policy and re-price it, to answer "
                "'what would this policy have cost?' without running it. group is a declared "
                "model group; strategy is 'cheapest' | 'lowest-latency' | 'lowest-failure' | "
                "'shuffle'. Re-prices each real call's actual token counts on whichever group "
                "member the strategy would have picked. The result prices a switch, it does "
                "not judge whether the alternative model's output would have been as good."
)
def simulate_routing(
    group: str, strategy: str = "cheapest", period: str = "week", source: str | None = None
) -> dict:
    if period not in _VALID_PERIODS:
        return _bad_period(period)
    if (err := _check_source(source)) is not None:
        return err
    try:
        return _simulate_routing(group, strategy, period=period, source=source)
    except Exception as exc:
        return {"error": str(exc)}


@server.tool(
    description="Check this project's hardcoded pricing table against LiteLLM's public, "
                "community-maintained price map and report any rate that disagrees. Use when "
                "asked whether the pricing is current or why a cost figure looks off. Reports "
                "drift only, it never rewrites a rate, because silently re-pricing recorded "
                "history is worse than a stale number. Set refresh=true to re-download."
)
def check_pricing_drift(refresh: bool = False) -> dict:
    return _check_drift(refresh=refresh)


if __name__ == "__main__":
    server.run(transport="stdio")


@server.tool(
    description="Classify how hard a prompt is, the input the 'complexity' routing strategy "
                "uses to pick a model tier. Returns tier ('trivial' | 'moderate' | 'complex'), "
                "a score, and every heuristic rule that fired with what it contributed. Free "
                "and instant: it is pure heuristics and calls no model. Use it to understand "
                "why a prompt was routed where it was, or to check a routing decision you "
                "think was wrong."
)
def classify_prompt_complexity(prompt: str) -> dict:
    from src.complexity import classify

    return classify(prompt).as_dict()


@server.tool(
    description="Report the shadow-comparison dataset: real prompts that were answered by a "
                "real model AND re-answered by a cheap local one, grouped by complexity tier. "
                "This is the only thing here that speaks to routing QUALITY rather than cost. "
                "Note 'unverified_savings_usd' is what those calls cost on the real model, "
                "i.e. what routing them locally would have avoided, and is NOT a saving until "
                "'scored' covers them and 'acceptance_rate' says the cheap answers held up."
)
def get_shadow_summary() -> dict:
    from src.shadow import shadow_summary

    return shadow_summary()


@server.tool(
    description="List shadow comparisons that have not been graded yet, so a human or a judge "
                "can decide whether the cheap model's answer was good enough. Each row holds "
                "the real prompt, the real model's answer, and the local model's answer. "
                "Grade one with record_shadow_verdict."
)
def list_pending_shadow_reviews(limit: int = 10, tier: str | None = None) -> list[dict]:
    from src.shadow import pending_review

    return pending_review(limit=limit, tier=tier)


@server.tool(
    description="Grade one shadow comparison. verdict must be 'acceptable' (the cheap model's "
                "answer would have been fine) or 'inadequate' (it would not). This is what "
                "turns collected comparisons into an acceptance rate per complexity tier, and "
                "therefore into evidence about where cheap routing is safe."
)
def record_shadow_verdict(shadow_id: str, verdict: str, scored_by: str = "claude") -> dict:
    from src.shadow import record_verdict

    try:
        record_verdict(shadow_id, verdict, scored_by=scored_by)
    except ValueError as exc:
        return {"error": str(exc)}
    return {"ok": True, "id": shadow_id, "verdict": verdict}


@server.tool(
    description="Grade ungraded shadow comparisons automatically and record the verdicts. "
                "Deterministic checks run first (does the cheap answer's code parse, is it "
                "empty, is it drastically shorter) and are trusted absolutely; where none "
                "applies, a local model judges blind, with the two answers in randomised "
                "order. Local-judge verdicts are TRIAGE, not evidence: a small model grading "
                "a small model. Filter on scored_by before quoting an acceptance rate. Free: "
                "the judge runs locally."
)
def grade_shadow_comparisons(limit: int = 20, tier: str | None = None, use_llm: bool = True) -> dict:
    from src.judge import grade_pending

    return grade_pending(limit=limit, tier=tier, use_llm=use_llm)
