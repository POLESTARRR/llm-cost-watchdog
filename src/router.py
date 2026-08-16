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

STRATEGIES = ("cheapest", "lowest-latency", "lowest-failure", "shuffle")
DEFAULT_STRATEGY = os.environ.get("WATCHDOG_ROUTING_STRATEGY", "cheapest")

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

    def as_dict(self) -> dict:
        return {
            "group": self.group, "model": self.model, "strategy": self.strategy,
            "candidates": self.candidates, "excluded": self.excluded, "basis": self.basis,
        }


# --- groups --------------------------------------------------------------


def model_groups() -> dict[str, list[str]]:
    """Groups declared as WATCHDOG_GROUP_<NAME>=model,model,model.

    Read from the environment at call time rather than import time so a test or
    a running process can change them without a restart, the same reason
    tracker.resolve_db_path() resolves late.
    """
    groups: dict[str, list[str]] = {}
    for key, value in os.environ.items():
        if not key.startswith("WATCHDOG_GROUP_"):
            continue
        name = key[len("WATCHDOG_GROUP_"):].strip().lower()
        members = [m.strip() for m in value.split(",") if m.strip()]
        if name and members:
            groups[name] = members
    return groups


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
) -> RoutingDecision:
    """Pick one model from `group`, and record why.

    Order of filtering matters: hard constraints first (provider configured,
    not benched, prompt actually fits), then the strategy ranks whatever
    survives. A strategy is a preference, never a reason to dispatch a call
    that cannot succeed.
    """
    strategy = strategy or DEFAULT_STRATEGY
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
    chosen, basis = _rank(candidates, strategy, stats, estimated_input_tokens, estimated_output_tokens)

    return RoutingDecision(
        group=group, model=chosen, strategy=strategy,
        candidates=candidates, excluded=excluded, basis=basis,
    )


def _rank(
    candidates: list[str], strategy: str, stats: dict[str, dict],
    in_tokens: int, out_tokens: int,
) -> tuple[str, str]:
    """Return (model, human-readable reason)."""
    if strategy == "shuffle":
        pick = random.choice(candidates)
        return pick, f"random pick from {len(candidates)} candidate(s)"

    if strategy == "cheapest":
        priced = {m: estimate_cost(m, in_tokens or 1000, out_tokens)["cost_usd"] for m in candidates}
        pick = min(priced, key=priced.get)
        return pick, f"cheapest for {in_tokens or 1000}/{out_tokens} tokens (${priced[pick]:.6f})"

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
        return pick, f"no model has {MIN_CALLS_FOR_HISTORY}+ recorded calls; fell back to cheapest"

    if strategy == "lowest-latency":
        timed = {m: s["avg_latency_ms"] for m, s in measured.items() if s["avg_latency_ms"]}
        if not timed:
            pick = min(measured, key=lambda m: measured[m]["avg_cost_usd"])
            return pick, "no measured latency; fell back to lowest measured cost"
        pick = min(timed, key=timed.get)
        return pick, f"lowest measured latency ({timed[pick]:.0f}ms over {measured[pick]['calls']} calls)"

    # lowest-failure: ties broken by measured cost, so the cheapest of several
    # equally-reliable models wins rather than an arbitrary one.
    pick = min(measured, key=lambda m: (measured[m]["failure_rate"], measured[m]["avg_cost_usd"]))
    s = measured[pick]
    return pick, f"lowest measured failure rate ({s['failure_rate']:.1%} over {s['calls']} calls)"


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
        pick, _ = _rank(members, strategy, stats, e.input_tokens, e.output_tokens)
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
        ),
    }


def router_status(db_path: str | None = None) -> dict:
    """Everything the router would do right now, without dispatching anything."""
    groups = model_groups()
    return {
        "strategy": DEFAULT_STRATEGY,
        "available_strategies": list(STRATEGIES),
        "groups": groups,
        "cooldowns": active_cooldowns(db_path=db_path),
        "cooldown_seconds": DEFAULT_COOLDOWN_SECONDS,
        "history_window_days": HISTORY_WINDOW_DAYS,
        "unpriced_members": sorted({
            m for members in groups.values() for m in members if m not in PRICING_TABLE
        }),
    }
