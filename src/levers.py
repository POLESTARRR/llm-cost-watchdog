"""Which optimisation would actually have paid, ranked on real recorded traffic.

`find_waste` answers "what was avoidable?" and answers it one check at a time.
That framing has a blind spot this project fell into: it reports every category
it can measure, without saying which one is *worth doing*, so the loudest number
wins attention rather than the largest one. Run against this repo's own ledger it
leads with a model-switch finding, and a model switch is not where this traffic's
money is.

The levers, measured on 4,186 real calls (764M input tokens, 2.9M output):

    prompt caching        already saving 84%   REALISED, no quality risk
    switch to Haiku       would save     77%   HYPOTHETICAL, large quality risk
    switch to Sonnet      would save     31%   HYPOTHETICAL, moderate risk
    shorter outputs       caps at        11%   output is 11% of spend

Two things follow, and neither is visible from a per-category waste report:

**Caching is the dominant lever and it is already pulled.** Reporting a
hypothetical model switch as the top recommendation, next to a realised saving
four times its size, inverts the actual priority. A cost tool should say "the
thing you already did is what worked."

**Cost here is 78% input tokens.** Every "use a cheaper model" recommendation is
really a claim about input pricing, on a workload whose outputs are rounding
error. That reframes what a router is for: on agentic traffic, routing changes
who reads your context, not who writes your answer, and the reading is the bill.

Each lever is labelled `realised` or `hypothetical`, and every hypothetical one
carries the assumption it depends on. A saving that requires the cheap model to
have been good enough is not a saving until something has checked, which is what
src/shadow.py is for.
"""

from dataclasses import dataclass, field

from src.pricing import PRICING_TABLE, calculate_cost
from src.tracker import get_events_for_period
from src.usage_schema import UsageEvent

# Same-vendor tiers, cheapest last. Only used for the model-switch lever, and
# only within a vendor: pricing is comparable there in a way it is not across
# vendors with different tokenizers and context behaviour.
CHEAPER_SIBLINGS = {
    "claude-opus-5": ["claude-sonnet-5", "claude-haiku-4-5"],
    "claude-opus-4-8": ["claude-sonnet-5", "claude-haiku-4-5"],
    "claude-fable-5": ["claude-sonnet-5", "claude-haiku-4-5"],
    "claude-sonnet-5": ["claude-haiku-4-5"],
    "gpt-5.6-sol": ["gpt-5.6-terra", "gpt-5.6-luna"],
    "gpt-5.6-terra": ["gpt-5.6-luna"],
    "gemini-pro-latest": ["gemini-flash-latest", "gemini-flash-lite-latest"],
    "gemini-flash-latest": ["gemini-flash-lite-latest"],
}


@dataclass
class Lever:
    """One way the bill could be, or already is, smaller."""

    name: str
    saving_usd: float
    percent: float
    # `realised` means it already happened and is in the actual number.
    # `hypothetical` means it requires a change nobody has validated.
    status: str
    detail: str
    assumption: str = ""

    def as_dict(self) -> dict:
        d = {
            "lever": self.name,
            "saving_usd": round(self.saving_usd, 2),
            "percent_of_spend": round(self.percent, 1),
            "status": self.status,
            "detail": self.detail,
        }
        if self.assumption:
            d["assumption"] = self.assumption
        return d


@dataclass
class LeverReport:
    total_spend_usd: float
    calls: int
    input_share_percent: float
    output_share_percent: float
    levers: list[Lever] = field(default_factory=list)

    def as_dict(self) -> dict:
        realised = [x for x in self.levers if x.status == "realised"]
        return {
            "total_spend_usd": round(self.total_spend_usd, 2),
            "calls": self.calls,
            "cost_shape": {
                "input_percent": round(self.input_share_percent, 1),
                "output_percent": round(self.output_share_percent, 1),
            },
            "levers": [x.as_dict() for x in self.levers],
            "biggest_lever": self.levers[0].name if self.levers else None,
            "already_realised_usd": round(sum(x.saving_usd for x in realised), 2),
            "note": (
                "Levers are ranked by size, not by novelty. A `realised` lever is already "
                "reflected in total_spend_usd, it is what the bill would have been WITHOUT it. "
                "A `hypothetical` lever is not a saving until its assumption is checked; "
                "re-pricing tokens on a cheaper model prices a switch, it does not judge one."
            ),
        }


def _cost(e: UsageEvent, model: str | None = None) -> float:
    return calculate_cost(
        model or e.model, e.input_tokens, e.output_tokens,
        e.cached_input_tokens, e.cache_write_tokens, e.cache_write_1h_tokens,
    )


def _cost_shape(events: list[UsageEvent]) -> tuple[float, float]:
    """Split spend into the part paid for reading and the part for writing."""
    inp = out = 0.0
    for e in events:
        rates = PRICING_TABLE.get(e.model)
        if not rates:
            continue
        cached = min(e.cached_input_tokens, e.input_tokens)
        written = min(e.cache_write_tokens, e.input_tokens - cached)
        uncached = e.input_tokens - cached - written
        inp += (uncached / 1000) * rates["input"]
        inp += (cached / 1000) * rates["cached_input"]
        inp += (written / 1000) * rates["input"] * 1.25
        out += (e.output_tokens / 1000) * rates["output"]
    return inp, out


def analyse_levers(period: str = "all_time", source: str | None = None) -> LeverReport:
    """Rank every measurable cost lever against recorded traffic."""
    events = [e for e in get_events_for_period(period, source=source) if e.success]
    actual = sum(_cost(e) for e in events)

    report = LeverReport(total_spend_usd=actual, calls=len(events),
                         input_share_percent=0.0, output_share_percent=0.0)
    if not events or actual <= 0:
        return report

    inp, out = _cost_shape(events)
    report.input_share_percent = 100 * inp / actual
    report.output_share_percent = 100 * out / actual

    levers: list[Lever] = []

    # --- 1. prompt caching, already realised ------------------------------
    uncached_total = sum(
        calculate_cost(e.model, e.input_tokens, e.output_tokens, 0, 0) for e in events
    )
    cache_saving = uncached_total - actual
    if cache_saving > 0:
        hit = sum(e.cached_input_tokens for e in events)
        total_in = sum(e.input_tokens for e in events) or 1
        levers.append(Lever(
            name="prompt caching",
            saving_usd=cache_saving,
            percent=100 * cache_saving / uncached_total,
            status="realised",
            detail=(
                f"{100 * hit / total_in:.0f}% of input tokens were served from cache. "
                f"The same traffic uncached would have cost ${uncached_total:,.2f} "
                f"instead of ${actual:,.2f}."
            ),
        ))

    # --- 2. model tier, hypothetical --------------------------------------
    by_target: dict[str, float] = {}
    for e in events:
        for sibling in CHEAPER_SIBLINGS.get(e.model, []):
            by_target[sibling] = by_target.get(sibling, 0.0) + (_cost(e) - _cost(e, sibling))

    for target, saving in sorted(by_target.items(), key=lambda kv: -kv[1])[:2]:
        if saving <= 0:
            continue
        levers.append(Lever(
            name=f"move eligible traffic to {target}",
            saving_usd=saving,
            percent=100 * saving / actual,
            status="hypothetical",
            detail=f"Re-prices each call's real token counts on {target}.",
            assumption=(
                f"that {target} would have produced acceptable output for this work. "
                "Unverified: nothing here has compared the answers."
            ),
        ))

    # --- 3. output length, bounded by the shape ---------------------------
    # Worth stating even though it is small, because "make it terser" is the
    # instinctive first suggestion and the data says it cannot matter much.
    levers.append(Lever(
        name="shorter outputs",
        saving_usd=out,
        percent=100 * out / actual,
        status="hypothetical",
        detail=(
            f"Output tokens account for ${out:,.2f} of ${actual:,.2f}. Even generating "
            f"nothing at all caps this lever at {100 * out / actual:.0f}%."
        ),
        assumption="that the outputs contained anything removable at all.",
    ))

    # --- 4. cache TTL choice ----------------------------------------------
    # 1-hour writes bill 2.0x input, 5-minute writes 1.25x. Real money when a
    # workload writes a lot of cache, invisible in any per-call view.
    ttl_extra = 0.0
    for e in events:
        if e.cache_write_1h_tokens and (rates := PRICING_TABLE.get(e.model)):
            ttl_extra += (e.cache_write_1h_tokens / 1000) * rates["input"] * (2.0 - 1.25)
    if ttl_extra > 0:
        levers.append(Lever(
            name="1-hour cache TTL premium",
            saving_usd=ttl_extra,
            percent=100 * ttl_extra / actual,
            status="hypothetical",
            detail=(
                f"${ttl_extra:,.2f} was the premium for writing to a 1-hour cache instead "
                "of the 5-minute one (2.0x input vs 1.25x)."
            ),
            assumption="that a 5-minute TTL would have hit often enough to be worth it.",
        ))

    levers.sort(key=lambda x: -x.saving_usd)
    report.levers = levers
    return report
