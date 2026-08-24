"""
Model-group routing driven by this project's own recorded traffic.

`call_llm()` already picks a provider from the model id and fails over on a
429. That is failover, not routing: the fallback order is a hardcoded list and
nothing remembers that a provider was rate-limited thirty seconds ago.

This adds the missing half, model *groups*, cooldowns, and a routing strategy
but the strategy is the point, not the mechanism. A router that picks the
cheapest deployment from a published price list is a solved problem (LiteLLM's
Router does it, among others). What that router cannot do is answer "which
model actually served my traffic fastest, at what real cost, with what failure
rate", because it has no memory. This one reads the same SQLite ledger every
other part of this project reads, so a routing decision is made from measured
history and can be audited against it afterwards.

    WATCHDOG_GROUP_FAST=gemini-flash-lite-latest,claude-haiku-4-5,gpt-5-nano
    WATCHDOG_GROUP_SMART=claude-opus-5,gpt-5.6-sol
    WATCHDOG_ROUTING_STRATEGY=cheapest      # or lowest-latency | lowest-failure | shuffle

    from src.utils import call_llm
    call_llm(prompt, model_group="fast", project_tag="my-app")

The honest limits, stated up front: this is a single-process, single-user
router. There is no Redis, no cross-process coordination, no per-deployment
TPM/RPM accounting, and no concurrency control, those matter for a shared
gateway serving many callers and are deliberately absent here. Cooldown state
lives in SQLite so it survives a restart, which is as far as a personal tool
needs to go.
"""

import os
import random
import time
from dataclasses import dataclass

from src.pricing import PRICING_TABLE, estimate_cost
from src.tracker import _connect, init_db
from src.usage_schema import UsageEvent

# How long a model sits out after tripping a rate limit. Long enough that a
# per-minute quota has a chance to reset; short enough that one bad minute
# doesn't bench a model for an hour.
DEFAULT_COOLDOWN_SECONDS = int(os.environ.get("WATCHDOG_COOLDOWN_SECONDS", "60"))

STRATEGIES = ("cheapest", "lowest-latency", "lowest-failure", "shuffle", "complexity")


# The name of the group synthesized when nobody has declared one.
DEFAULT_GROUP = "ladder"

# How many tiers a synthesized ladder has. Three matches the classifier's
# trivial/moderate/complex, so every tier is reachable.
DEFAULT_LADDER_SIZE = 3


def default_strategy() -> str:
    """The configured strategy, resolved at call time.

    Read late for the same reason `model_groups()` is: this module already
    promises that changing configuration does not require a restart, and a
    strategy pinned at import time quietly broke that promise. It cost a
    confusing debugging session, a running gateway kept routing `cheapest`
    while its own /router endpoint reported `complexity`, because the endpoint
    read the environment and the router read a stale module global.
    """
    # `complexity` rather than `cheapest`, because a group containing a free
    # local model makes `cheapest` degenerate: it sends architecture questions
    # to a 3B model forever, since nothing is cheaper than $0. Asking a group
    # to route at all implies wanting the right tier, not always the bottom one.
    return os.environ.get("WATCHDOG_ROUTING_STRATEGY", "complexity")

# How much history a history-based strategy considers. Short enough to react to
# a provider degrading today, long enough not to swing on a single call.
HISTORY_WINDOW_DAYS = int(os.environ.get("WATCHDOG_ROUTING_WINDOW_DAYS", "7"))

# A model needs at least this many recorded calls before its measured latency
# or failure rate is trusted. Below it, one unlucky call would decide routing
# for everything that follows.
MIN_CALLS_FOR_HISTORY = 3

_COOLDOWN_SCHEMA = """
CREATE TABLE IF NOT EXISTS model_cooldowns (
    model TEXT PRIMARY KEY,
    until_ts REAL NOT NULL,
    reason TEXT
);
"""


class RoutingError(RuntimeError):
    """No candidate could be selected for a group."""


@dataclass
class RoutingDecision:
    """Why a call went where it went. Logged alongside the call itself."""
    group: str
    model: str
    strategy: str
    candidates: list[str]
    excluded: dict[str, str]
    basis: str
    # Only populated by the `complexity` strategy: the classifier's full
    # verdict, so a questionable route can be traced to the signals that
    # produced it rather than just the tier it landed on.
    complexity: dict | None = None

    def as_dict(self) -> dict:
        out = {
            "group": self.group, "model": self.model, "strategy": self.strategy,
            "candidates": self.candidates, "excluded": self.excluded, "basis": self.basis,
        }
        if self.complexity is not None:
            out["complexity"] = self.complexity
        return out


# --- groups --------------------------------------------------------------


def model_groups(include_default: bool = True) -> dict[str, list[str]]:
    """Groups declared as WATCHDOG_GROUP_<NAME>=model,model,model.

    Read from the environment at call time rather than import time so a test or
    a running process can change them without a restart, the same reason
    tracker.resolve_db_path() resolves late.

    When nothing is declared, a `ladder` group is synthesized from whatever
    providers are actually configured. A gateway that requires four environment
    variables before `group:ladder` resolves is not a gateway anyone will try:
    this project's own deployment advertised that group on its front page while
    returning 400 for it, because the variables were never set on the host.
    An explicitly declared group of the same name always wins.
    """
    groups: dict[str, list[str]] = {}
    for key, value in os.environ.items():
        if not key.startswith("WATCHDOG_GROUP_"):
            continue
        name = key[len("WATCHDOG_GROUP_"):].strip().lower()
        members = [m.strip() for m in value.split(",") if m.strip()]
        if name and members:
            groups[name] = members

    if include_default and not groups:
        if ladder := default_ladder():
            groups[DEFAULT_GROUP] = ladder
    return groups


def default_ladder(size: int = DEFAULT_LADDER_SIZE) -> list[str]:
    """A cheapest-to-most-capable ladder built from configured providers.

    Price order stands in for capability order, which is defensible here for
    the same reason it is inside a hand-declared group: these are all general
    chat models, and within one vendor's line-up price tracks capability
    closely enough to pick a tier.

    Returns [] when nothing is configured, so callers get "no groups" rather
    than a group whose every member is unreachable.
    """
    from src.pricing import PRICING_TABLE, estimate_cost
    from src.providers import ProviderError, _PROVIDERS, configured_providers, infer_provider

    available = configured_providers()
    candidates: list[str] = []

    for model in PRICING_TABLE:
        try:
            owner = infer_provider(model)
        except ProviderError:
            continue
        if available.get(owner):
            candidates.append(model)

    # Locally-served models are discovered rather than enumerated, and belong at
    # the bottom of the ladder because they cost nothing per token.
    #
    # Availability comes from the same `configured_providers()` view used for
    # hosted models above, not from calling the adapter's own probe: two
    # different sources of truth for "is this usable" is how a caller ends up
    # with a ladder whose rungs disagree with what /providers reports.
    if available.get("ollama"):
        ollama = _PROVIDERS.get("ollama")
        if ollama is not None:
            candidates = ollama.available_models()[:1] + candidates

    if not candidates:
        return []

    priced = sorted(candidates, key=lambda m: estimate_cost(m, 1000, 500)["cost_usd"])
    if len(priced) <= size:
        return priced

    # Take the cheapest, the dearest, and evenly spaced rungs between, so the
    # ladder spans the real price range instead of clustering at one end.
    idx = sorted({round(i * (len(priced) - 1) / (size - 1)) for i in range(size)})
    return [priced[i] for i in idx]


def resolve_group(group: str) -> list[str]:
    groups = model_groups()
    if group not in groups:
        raise RoutingError(
            f"unknown model group {group!r}; define WATCHDOG_GROUP_{group.upper()}=model,model. "
            f"Known groups: {sorted(groups) or 'none'}"
        )
    return groups[group]


# --- cooldowns -----------------------------------------------------------


def _init_cooldowns(db_path: str | None = None) -> None:
    init_db(db_path)
    with _connect(db_path) as conn:
        conn.executescript(_COOLDOWN_SCHEMA)


def start_cooldown(
    model: str, seconds: int = DEFAULT_COOLDOWN_SECONDS, reason: str = "rate limit",
    db_path: str | None = None,
) -> None:
    """Bench a model until `seconds` from now.

    Persisted rather than held in memory: a rate limit that a restart forgets
    is a rate limit you hit again immediately.
    """
    _init_cooldowns(db_path)
    with _connect(db_path) as conn:
        conn.execute(
            "INSERT INTO model_cooldowns (model, until_ts, reason) VALUES (?, ?, ?) "
            "ON CONFLICT(model) DO UPDATE SET until_ts=excluded.until_ts, reason=excluded.reason",
            (model, time.time() + seconds, reason),
        )


def active_cooldowns(db_path: str | None = None) -> dict[str, dict]:
    """Models currently benched, with seconds remaining."""
    _init_cooldowns(db_path)
    now = time.time()
    with _connect(db_path) as conn:
        rows = conn.execute(
            "SELECT model, until_ts, reason FROM model_cooldowns WHERE until_ts > ?", (now,)
        ).fetchall()
    return {
        r["model"]: {"seconds_remaining": round(r["until_ts"] - now, 1), "reason": r["reason"]}
        for r in rows
    }


def clear_cooldown(model: str, db_path: str | None = None) -> None:
    _init_cooldowns(db_path)
    with _connect(db_path) as conn:
        conn.execute("DELETE FROM model_cooldowns WHERE model = ?", (model,))


# --- measured history ----------------------------------------------------


def model_stats(
    period_days: int = HISTORY_WINDOW_DAYS,
    source: str | None = None,
    events: list[UsageEvent] | None = None,
) -> dict[str, dict]:
    """Per-model cost, latency and failure rate from recorded traffic.

    This is what makes the routing strategies non-generic: they are computed
    from calls that actually happened, not from a published price list.
    """
    from src.tracker import get_events_for_period

    if events is None:
        period = "week" if period_days <= 7 else "month"
        events = get_events_for_period(period, source=source)

    stats: dict[str, dict] = {}
    for e in events:
        s = stats.setdefault(e.model, {
            "calls": 0, "failures": 0, "cost_usd": 0.0,
            "latency_total": 0.0, "latency_samples": 0,
        })
        s["calls"] += 1
        s["cost_usd"] += e.cost_usd
        if not e.success:
            s["failures"] += 1
        elif e.latency_ms > 0:
            s["latency_total"] += e.latency_ms
            s["latency_samples"] += 1

    for model, s in stats.items():
        s["failure_rate"] = round(s["failures"] / s["calls"], 4) if s["calls"] else 0.0
        s["avg_latency_ms"] = (
            round(s["latency_total"] / s["latency_samples"], 1) if s["latency_samples"] else None
        )
        s["avg_cost_usd"] = round(s["cost_usd"] / s["calls"], 8) if s["calls"] else 0.0
        s["cost_usd"] = round(s["cost_usd"], 6)
        del s["latency_total"], s["latency_samples"]
    return stats


# --- selection -----------------------------------------------------------


def _configured_models(candidates: list[str]) -> tuple[list[str], dict[str, str]]:
    """Drop candidates whose provider has no credentials configured."""
    from src.providers import ProviderError, configured_providers, infer_provider

    available = configured_providers()
    keep, excluded = [], {}
    for model in candidates:
        try:
            provider = infer_provider(model)
        except ProviderError:
            excluded[model] = "unknown provider"
            continue
        if not available.get(provider):
            excluded[model] = f"{provider} not configured"
            continue
        keep.append(model)
    return keep, excluded


def select(
    group: str,
    strategy: str | None = None,
    estimated_input_tokens: int = 0,
    estimated_output_tokens: int = 500,
    require_configured: bool = True,
    db_path: str | None = None,
    prompt: str | None = None,
) -> RoutingDecision:
    """Pick one model from `group`, and record why.

    Order of filtering matters: hard constraints first (provider configured,
    not benched, prompt actually fits), then the strategy ranks whatever
    survives. A strategy is a preference, never a reason to dispatch a call
    that cannot succeed.
    """
    strategy = strategy or default_strategy()
    if strategy not in STRATEGIES:
        raise RoutingError(f"unknown strategy {strategy!r}; expected one of {list(STRATEGIES)}")

    members = resolve_group(group)
    excluded: dict[str, str] = {}

    candidates = list(members)
    if require_configured:
        candidates, excluded = _configured_models(candidates)

    benched = active_cooldowns(db_path=db_path)
    for model in list(candidates):
        if model in benched:
            excluded[model] = f"cooling down ({benched[model]['seconds_remaining']}s left)"
            candidates.remove(model)

    # Pre-call context check: a prompt that cannot fit is not a routing
    # preference, it is a guaranteed failure. Window data comes from the public
    # price map; models with no known window are left alone rather than guessed.
    if estimated_input_tokens:
        from src.pricing_drift import context_windows

        windows = context_windows()
        for model in list(candidates):
            limit = windows.get(model)
            if limit and estimated_input_tokens > limit:
                excluded[model] = f"prompt {estimated_input_tokens} tok exceeds {limit} tok window"
                candidates.remove(model)

    if not candidates:
        raise RoutingError(
            f"no usable model in group {group!r}. Excluded: {excluded or 'nothing configured'}"
        )

    stats = model_stats()
    chosen, basis, verdict = _rank(
        candidates, strategy, stats, estimated_input_tokens, estimated_output_tokens, prompt
    )

    return RoutingDecision(
        group=group, model=chosen, strategy=strategy,
        candidates=candidates, excluded=excluded, basis=basis, complexity=verdict,
    )


def _rank(
    candidates: list[str], strategy: str, stats: dict[str, dict],
    in_tokens: int, out_tokens: int, prompt: str | None = None,
) -> tuple[str, str, dict | None]:
    """Return (model, human-readable reason, complexity verdict or None)."""
    if strategy == "shuffle":
        pick = random.choice(candidates)
        return pick, f"random pick from {len(candidates)} candidate(s)", None

    if strategy == "complexity":
        return _rank_by_complexity(candidates, prompt, in_tokens, out_tokens)

    if strategy == "cheapest":
        priced = {m: estimate_cost(m, in_tokens or 1000, out_tokens)["cost_usd"] for m in candidates}
        pick = min(priced, key=priced.get)
        return pick, f"cheapest for {in_tokens or 1000}/{out_tokens} tokens (${priced[pick]:.6f})", None

    # History-based strategies fall back to price when there isn't enough
    # measured traffic to be worth trusting, an unmeasured model is not
    # thereby the best one.
    measured = {
        m: stats[m] for m in candidates
        if m in stats and stats[m]["calls"] >= MIN_CALLS_FOR_HISTORY
    }
    if not measured:
        priced = {m: estimate_cost(m, in_tokens or 1000, out_tokens)["cost_usd"] for m in candidates}
        pick = min(priced, key=priced.get)
        return pick, f"no model has {MIN_CALLS_FOR_HISTORY}+ recorded calls; fell back to cheapest", None

    if strategy == "lowest-latency":
        timed = {m: s["avg_latency_ms"] for m, s in measured.items() if s["avg_latency_ms"]}
        if not timed:
            pick = min(measured, key=lambda m: measured[m]["avg_cost_usd"])
            return pick, "no measured latency; fell back to lowest measured cost", None
        pick = min(timed, key=timed.get)
        return pick, f"lowest measured latency ({timed[pick]:.0f}ms over {measured[pick]['calls']} calls)", None

    # lowest-failure: ties broken by measured cost, so the cheapest of several
    # equally-reliable models wins rather than an arbitrary one.
    pick = min(measured, key=lambda m: (measured[m]["failure_rate"], measured[m]["avg_cost_usd"]))
    s = measured[pick]
    return pick, f"lowest measured failure rate ({s['failure_rate']:.1%} over {s['calls']} calls)", None


def _rank_by_complexity(
    candidates: list[str], prompt: str | None, in_tokens: int, out_tokens: int,
) -> tuple[str, str, dict | None]:
    """Pick by what the prompt is asking for, not by what the models cost.

    The candidate list is sorted cheapest-first and the tier indexes into it.
    Sorting by price is what makes this work without any extra configuration:
    a model group is already a set the user declared interchangeable, so its
    price order is a usable stand-in for its capability order.

    Without a prompt this cannot do its job, and guessing would be worse than
    admitting it: it degrades to the middle tier and says so in the basis, so
    the fallback is visible in the decision record rather than silent.
    """
    from src.complexity import classify, tier_index

    ordered = sorted(
        candidates, key=lambda m: estimate_cost(m, in_tokens or 1000, out_tokens)["cost_usd"]
    )

    if prompt is None:
        pick = ordered[tier_index("moderate", len(ordered))]
        return pick, "no prompt supplied; defaulted to the middle of the group", None

    # `in_tokens` is the size of the whole request, conversation history
    # included, so it is exactly the context measurement the classifier wants.
    # It was already being threaded through here to price the candidates and was
    # never shown to the thing deciding difficulty, which is why the cheap tier
    # was reachable on 1.8% of real traffic: the classifier could only read the
    # words, and on measured data the words are the half that does not carry the
    # request. Passed as None when unknown, which keeps word-only behaviour.
    verdict = classify(prompt, context_tokens=in_tokens or None)
    pick = ordered[tier_index(verdict.tier, len(ordered))]
    basis = (
        f"classified {verdict.tier} (score {verdict.score:+d}"
        + (f", {in_tokens:,} tokens of context" if in_tokens else "")
        + f") -> tier {tier_index(verdict.tier, len(ordered)) + 1} of {len(ordered)} by price"
    )
    return pick, basis, verdict.as_dict()


# --- counterfactual ------------------------------------------------------


def simulate_routing(
    group: str, strategy: str, period: str = "week", source: str | None = None,
) -> dict:
    """Replay recorded traffic through a routing policy and re-price it.

    This is the feature the ledger makes possible and a stateless router
    cannot have: instead of arguing about which policy is cheaper, run last
    week's real calls through it and read the number. It is `what_if_switched`
    generalised from one model to a policy.

    Every call in the period is re-priced on whichever group member the
    strategy would have chosen, using that call's real token counts. The
    result is a bound, not a promise, it assumes the alternative model would
    have produced a comparable response, which is exactly the assumption a
    human should be making the decision about.
    """
    from src.tracker import get_events_for_period

    members = resolve_group(group)
    events = [e for e in get_events_for_period(period, source=source) if e.success]
    if not events:
        return {
            "group": group, "strategy": strategy, "period": period,
            "calls_repriced": 0, "note": "no successful calls in this period",
        }

    stats = model_stats(events=events)
    actual = sum(e.cost_usd for e in events)
    simulated = 0.0
    routed_to: dict[str, int] = {}

    for e in events:
        # No prompt is passed, and none can be: the ledger stores an 80-char
        # preview and a hash, never the prompt itself. So `complexity` cannot
        # be simulated against history, it degrades to the middle tier and the
        # caveat below says so outright rather than returning a number whose
        # basis the caller would have to go read the source to distrust.
        pick, _, _ = _rank(members, strategy, stats, e.input_tokens, e.output_tokens)
        routed_to[pick] = routed_to.get(pick, 0) + 1
        simulated += estimate_cost(pick, e.input_tokens, e.output_tokens)["cost_usd"]

    delta = actual - simulated
    return {
        "group": group,
        "strategy": strategy,
        "period": period,
        "calls_repriced": len(events),
        "actual_cost_usd": round(actual, 6),
        "simulated_cost_usd": round(simulated, 6),
        "savings_usd": round(delta, 6),
        "savings_percent": round((delta / actual) * 100, 2) if actual else 0.0,
        "routed_to": routed_to,
        "verdict": "cheaper" if delta > 0 else ("same" if abs(delta) < 1e-9 else "more expensive"),
        "caveat": (
            "Re-prices real token counts on the model the policy would have picked. "
            "Assumes comparable output quality, it prices the switch, it does not judge it."
            + (
                " NOTE: `complexity` routes on prompt text, which the ledger does not store, "
                "so this simulation fell back to the middle tier for every call and does NOT "
                "represent what that strategy would really do. Use src/shadow.py to measure it "
                "on live traffic instead."
                if strategy == "complexity" else ""
            )
        ),
    }


def router_status(db_path: str | None = None) -> dict:
    """Everything the router would do right now, without dispatching anything."""
    groups = model_groups()
    declared = model_groups(include_default=False)
    return {
        "strategy": default_strategy(),
        "available_strategies": list(STRATEGIES),
        "groups": groups,
        # A synthesized ladder must never look like something the operator
        # chose; they need to know it will change if their credentials do.
        "groups_are_default": not declared,
        "cooldowns": active_cooldowns(db_path=db_path),
        "cooldown_seconds": DEFAULT_COOLDOWN_SECONDS,
        "history_window_days": HISTORY_WINDOW_DAYS,
        "unpriced_members": sorted({
            m for members in groups.values() for m in members if m not in PRICING_TABLE
        }),
    }
