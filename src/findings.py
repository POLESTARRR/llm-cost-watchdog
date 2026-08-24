"""What the ledger says about the cost of building software with an agent.

Every other module here answers an operational question: what did I spend, am I
over budget, which model should this prompt go to. This one answers the question
a *reader* has, which is different and had no home in the codebase until now:

    What does it actually cost to build real software with an AI agent,
    and where does the money go?

The dashboard could not answer that, and it is the only question its data is
uniquely qualified to answer. 4,407 recorded turns across 13 projects that were
actually shipped is not a benchmark or a synthetic replay, it is a bill. The
panels that existed instead (spend by model, budget status, projected weekly)
are the right panels for someone operating the tool and the wrong ones for
someone deciding whether the tool is worth their attention, because none of them
state a finding. They state inventory.

The finding, computed below rather than asserted:

**You do not pay an agent to write. You pay it to read.** This portfolio's
traffic reads 292 tokens for every token it writes, and 78.5% of the money went
to input. The instinct that follows from "the AI writes my code" is to ask for
shorter answers, and output is 10.9% of the bill, so that instinct is capped at
a tenth of the spend before it starts. What matters instead is who reads your
context and how often it is cached, which is precisely what a router decides.

Everything here is derived from the events table at call time. Nothing is
hardcoded, so a page rendering these numbers cannot drift away from the database
behind it, which is the specific failure that made the earlier headline dishonest.
"""

from dataclasses import dataclass, field

from src.pricing import PRICING_TABLE, calculate_cost
from src.tracker import get_events_for_period
from src.usage_schema import UsageEvent


@dataclass
class ProjectCost:
    """One shipped project and what the agent cost to build it."""

    project: str
    turns: int
    cost_usd: float
    input_tokens: int
    output_tokens: int

    def as_dict(self) -> dict:
        return {
            "project": self.project,
            "turns": self.turns,
            "cost_usd": round(self.cost_usd, 2),
            "cost_per_turn_usd": round(self.cost_usd / self.turns, 4) if self.turns else 0.0,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "read_write_ratio": round(self.input_tokens / max(self.output_tokens, 1)),
        }


@dataclass
class Findings:
    turns: int = 0
    projects: int = 0
    cost_usd: float = 0.0
    first_day: str = ""
    last_day: str = ""

    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0

    read_cost_usd: float = 0.0
    write_cost_usd: float = 0.0
    cache_ttl_premium_usd: float = 0.0

    uncached_cost_usd: float = 0.0

    median_turn_usd: float = 0.0
    p99_turn_usd: float = 0.0
    priciest: dict = field(default_factory=dict)

    by_project: list[ProjectCost] = field(default_factory=list)
    by_model: list[dict] = field(default_factory=list)

    def as_dict(self) -> dict:
        cost = self.cost_usd or 1.0
        saved = self.uncached_cost_usd - self.cost_usd

        # The three split components are all priced from token counts through
        # one pricing table, so their sum is internally consistent and the
        # percentages below always total 100. `cost_usd` is the figure stored on
        # each row, and on every real path the two agree exactly (verified:
        # 679.54 + 91.88 + 94.06 = 865.48, the recorded total to the cent).
        #
        # They can only diverge if a row was written with a cost that its own
        # tokens do not imply, which the import endpoint prevents by recomputing
        # server-side. Dividing by the priced sum rather than by `cost_usd` means
        # that even then the reader sees a coherent breakdown of the priced work
        # instead of three percentages that quietly fail to add up.
        priced = self.read_cost_usd + self.cache_ttl_premium_usd + self.write_cost_usd
        priced = priced or 1.0
        return {
            "headline": {
                "cost_usd": round(self.cost_usd, 2),
                "turns": self.turns,
                "projects": self.projects,
                "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens,
                "cost_per_project_usd": round(self.cost_usd / self.projects, 2) if self.projects else 0.0,
                "first_day": self.first_day,
                "last_day": self.last_day,
            },
            # The finding. Ratio first because it is the part that surprises.
            "read_vs_write": {
                "ratio": round(self.input_tokens / max(self.output_tokens, 1)),
                "read_cost_usd": round(self.read_cost_usd, 2),
                "write_cost_usd": round(self.write_cost_usd, 2),
                "cache_ttl_premium_usd": round(self.cache_ttl_premium_usd, 2),
                "read_percent": round(100 * self.read_cost_usd / priced, 1),
                "write_percent": round(100 * self.write_cost_usd / priced, 1),
                "cache_ttl_premium_percent": round(100 * self.cache_ttl_premium_usd / priced, 1),
                "priciest_turn": self.priciest,
            },
            "caching": {
                "saved_usd": round(saved, 2),
                "uncached_cost_usd": round(self.uncached_cost_usd, 2),
                "percent_saved": round(100 * saved / self.uncached_cost_usd, 1) if self.uncached_cost_usd else 0.0,
                "multiple": round(self.uncached_cost_usd / cost, 1),
                "hit_rate_percent": round(100 * self.cached_tokens / max(self.input_tokens, 1), 1),
            },
            "turn_shape": {
                "median_usd": round(self.median_turn_usd, 4),
                "p99_usd": round(self.p99_turn_usd, 4),
            },
            "by_project": [p.as_dict() for p in self.by_project],
            "by_model": self.by_model,
        }


def _cost_split(events: list[UsageEvent]) -> tuple[float, float, float]:
    """Split the bill three ways: reading, the cache-TTL premium, and writing.

    Not a token count scaled by a single rate. A cached read bills at roughly a
    tenth of the uncached rate and a cache *write* bills at a premium, so a split
    that ignores cache state on a 97%-cached workload is off by most of the bill.

    The third component exists because two of them did not add up to the total
    and the gap was 10.6%, which is far too large to write off as rounding. It
    is the surcharge for writing to a one-hour cache rather than a five-minute
    one: 2.0x the input rate against 1.25x. On this workload it comes to $91.88,
    within three dollars of everything the agent generated all month. Reporting
    it as an unexplained remainder would have been the same class of vagueness
    this page exists to remove, so it is named and priced on its own.
    """
    read = write = ttl_premium = 0.0
    for e in events:
        rates = PRICING_TABLE.get(e.model)
        if not rates:
            continue
        cached = min(e.cached_input_tokens, e.input_tokens)
        written = min(e.cache_write_tokens, e.input_tokens - cached)
        uncached = e.input_tokens - cached - written
        read += (uncached / 1000) * rates["input"]
        read += (cached / 1000) * rates["cached_input"]
        read += (written / 1000) * rates["input"] * 1.25
        write += (e.output_tokens / 1000) * rates["output"]
        ttl_premium += (e.cache_write_1h_tokens / 1000) * rates["input"] * (2.0 - 1.25)
    return read, ttl_premium, write


# The measured build work: turns a coding agent actually spent shipping the
# projects, recovered from local session transcripts. Deliberately NOT including
# `live`, which is traffic the gateway itself served. Folding the two together
# put four demo calls to Gemini next to 4,407 turns of Claude and reported "16
# projects", and it is the same category error that made a reader ask why a page
# about an OpenAI-compatible endpoint was full of Anthropic models. They are two
# different claims about two different things, so they stay separate and are
# labelled separately wherever they appear.
PORTFOLIO_SOURCES = "subscription,manual"


def load_turns(source: str = PORTFOLIO_SOURCES) -> list[UsageEvent]:
    """Every successful turn for `source`, fetched once.

    Exists because /findings was issuing four separate full reads of the same
    4,399 rows: one for the findings, one per counterfactual, one for the ROI.
    Against a local SQLite file that is invisible. Against a remote Turso
    database each one is a network round-trip, and the endpoint took 7.8s, which
    is 7.8s of blank page where the entire study is supposed to be. Callers that
    need several views of the same traffic now read it once and pass it down.
    """
    return [e for e in get_events_for_period("all_time", source=source) if e.success]


def compute_findings(
    source: str = PORTFOLIO_SOURCES,
    events: list[UsageEvent] | None = None,
) -> Findings:
    """Summarise every real turn in the ledger into a set of stated findings.

    Defaults to the measured build work. `demo` is excluded and never silently
    mixed in: a seeded row inside a headline is the exact defect that made the
    previous version of this page untrustworthy.
    """
    events = load_turns(source) if events is None else events
    f = Findings()
    if not events:
        return f

    f.turns = len(events)
    f.cost_usd = sum(e.cost_usd for e in events)
    days = sorted(e.timestamp[:10] for e in events)
    f.first_day, f.last_day = days[0], days[-1]

    f.input_tokens = sum(e.input_tokens for e in events)
    f.output_tokens = sum(e.output_tokens for e in events)
    f.cached_tokens = sum(e.cached_input_tokens for e in events)

    f.read_cost_usd, f.cache_ttl_premium_usd, f.write_cost_usd = _cost_split(events)
    f.uncached_cost_usd = sum(
        calculate_cost(e.model, e.input_tokens, e.output_tokens, 0, 0) for e in events
    )

    costs = sorted(e.cost_usd for e in events)
    f.median_turn_usd = costs[len(costs) // 2]
    f.p99_turn_usd = costs[min(int(len(costs) * 0.99), len(costs) - 1)]

    # The single turn that best illustrates the ratio: it is expensive because
    # of what it read, and what it wrote back is almost free.
    top = max(events, key=lambda e: e.cost_usd)
    f.priciest = {
        "cost_usd": round(top.cost_usd, 2),
        "input_tokens": top.input_tokens,
        "output_tokens": top.output_tokens,
        "model": top.model,
        "project": top.project_tag,
    }

    grouped: dict[str, ProjectCost] = {}
    for e in events:
        p = grouped.get(e.project_tag)
        if p is None:
            p = grouped[e.project_tag] = ProjectCost(e.project_tag, 0, 0.0, 0, 0)
        p.turns += 1
        p.cost_usd += e.cost_usd
        p.input_tokens += e.input_tokens
        p.output_tokens += e.output_tokens
    f.by_project = sorted(grouped.values(), key=lambda p: -p.cost_usd)
    f.projects = len(f.by_project)

    models: dict[str, dict] = {}
    for e in events:
        m = models.setdefault(e.model, {"model": e.model, "turns": 0, "cost_usd": 0.0})
        m["turns"] += 1
        m["cost_usd"] += e.cost_usd
    f.by_model = sorted(
        ({**m, "cost_usd": round(m["cost_usd"], 2)} for m in models.values()),
        key=lambda m: -m["cost_usd"],
    )

    return f


def counterfactual(
    model: str,
    source: str = PORTFOLIO_SOURCES,
    events: list[UsageEvent] | None = None,
) -> dict:
    """Re-price every recorded turn as if one model had served all of it.

    Deliberately named `counterfactual` and not `savings`. It prices a
    substitution using each turn's real token counts; it does not claim the
    substitute would have produced usable work, and on a workload that is 78%
    reading, a cheaper model reads the same tokens for less rather than reading
    fewer of them. Whether the answers would have held up is what src/shadow.py
    measures, and nothing here should be read as having measured it.
    """
    events = load_turns(source) if events is None else events
    if not events:
        return {"model": model, "cost_usd": 0.0, "actual_usd": 0.0, "delta_percent": 0.0}

    actual = sum(e.cost_usd for e in events)
    hypothetical = sum(
        calculate_cost(
            model, e.input_tokens, e.output_tokens,
            e.cached_input_tokens, e.cache_write_tokens, e.cache_write_1h_tokens,
        )
        for e in events
    )
    return {
        "model": model,
        "cost_usd": round(hypothetical, 2),
        "actual_usd": round(actual, 2),
        "delta_percent": round(100 * (hypothetical - actual) / actual, 1) if actual else 0.0,
        "caveat": (
            "Prices a substitution on real token counts. It does not judge whether "
            "the substitute's answers would have been good enough."
        ),
    }
