"""
FastAPI backing service for the local dashboard, which also serves the
static single-file frontend.

    uvicorn dashboard.app:app --reload --port 8000

The MCP server is the core deliverable; this exists so the same data can be
inspected in a browser without Claude Desktop.
"""

import datetime as _dt
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

load_dotenv()

from contextlib import asynccontextmanager

from src.analyzer import (
    check_budget_status,
    compute_report,
    flag_anomalies,
    project_burn_rate,
    provider_breakdown,
    subscription_roi,
)
from src.gateway import router as gateway_router
from src.guard import guard_status
from src.pricing import PRICING_TABLE, calculate_cost, compare_models
from src.pricing_drift import check_drift
from src.router import router_status
from src.waste import find_waste
from src.providers import configured_providers
from src.tracker import get_events_for_period, init_db, log_usage_many, parse_sources, source_totals
from src.usage_schema import Source, UsageEvent

STATIC_DIR = Path(__file__).resolve().parent / "static"
PERIOD_RE = "^(today|week|month|all_time)$"

# "all" | any comma-separated combination of the valid sources.
_SRC = "live|demo|manual|subscription"
SOURCE_RE = f"^(all|({_SRC})(,({_SRC}))*)$"

# Set to enable remote import; unset means the endpoint is disabled entirely,
# not merely unauthenticated. See scripts/import_claude_code_usage.py --remote-url.
IMPORT_KEY = os.environ.get("WATCHDOG_IMPORT_KEY")


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(title="LLM Cost Gateway", version="2.2.0", lifespan=lifespan)

# The OpenAI-compatible gateway. Mounted on the same app as the dashboard on
# purpose: one process, one port, one SQLite file, so `uvicorn dashboard.app:app`
# gives you the proxy AND the UI that reads what the proxy recorded.
app.include_router(gateway_router)


def _source(value: str | None) -> str | None:
    """Validate a source filter at the edge, so a bad one 400s instead of 500s."""
    try:
        parse_sources(value)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return value


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/report")
def report(
    period: str = Query("week", pattern=PERIOD_RE),
    source: str | None = Query(None, pattern=SOURCE_RE),
) -> dict:
    return compute_report(period, source=_source(source)).model_dump()


@app.get("/provenance")
def provenance(period: str = Query("all_time", pattern=PERIOD_RE)) -> dict:
    """How much of the recorded spend is real. Drives the dashboard's demo-data banner."""
    return source_totals(period)


@app.get("/anomalies")
def anomalies(
    threshold_multiplier: float = 3.0,
    source: str | None = Query(None, pattern=SOURCE_RE),
) -> list[dict]:
    return [a.model_dump() for a in flag_anomalies(threshold_multiplier, source=_source(source))]


@app.get("/budget")
def budget(source: str | None = Query(None, pattern=SOURCE_RE)) -> dict:
    # Unset means billed-only here, not "everything", a budget describes real
    # money, so check_budget_status() owns that default.
    if source is None:
        return check_budget_status("weekly")
    return check_budget_status("weekly", source=_source(source))


@app.get("/burn-rate")
def burn_rate(
    period: str = Query("week", pattern=PERIOD_RE),
    source: str | None = Query(None, pattern=SOURCE_RE),
) -> dict:
    if source is None:
        return project_burn_rate(period)
    return project_burn_rate(period, source=_source(source))


@app.get("/providers")
def providers(
    period: str = Query("week", pattern=PERIOD_RE),
    source: str | None = Query(None, pattern=SOURCE_RE),
) -> dict:
    return {
        "configured": configured_providers(),
        "breakdown": provider_breakdown(period, source=_source(source)),
    }


@app.get("/compare")
def compare(input_tokens: int = 5000, output_tokens: int = 1000) -> list[dict]:
    return compare_models(input_tokens, output_tokens)


@app.get("/waste")
def waste(
    period: str = Query("week", pattern=PERIOD_RE),
    source: str | None = Query(None, pattern=SOURCE_RE),
) -> dict:
    return find_waste(period, source=_source(source))


@app.get("/guard")
def guard() -> dict:
    return guard_status()


@app.get("/roi")
def roi(period: str = Query("all_time", pattern=PERIOD_RE)) -> dict:
    """List-price value of flat-fee usage, and what the plan cost over that span.

    The budget widget reads $0 whenever spend is covered by a subscription
    rather than metered per token, which is arithmetically right and visually
    alarming. This is the number that actually means something for that case.
    """
    return subscription_roi(period)


@app.get("/router")
def router() -> dict:
    """Declared model groups, active strategy, and anything cooling down."""
    return router_status()


@app.get("/pricing-drift")
def pricing_drift(refresh: bool = False) -> dict:
    """Local rates that disagree with the public price map. Never rewrites one."""
    return check_drift(refresh=refresh)


@app.get("/calls")
def calls(
    period: str = Query("week", pattern=PERIOD_RE),
    limit: int = 25,
    source: str | None = Query(None, pattern=SOURCE_RE),
) -> list[dict]:
    """Most recent calls first, for the dashboard's activity table."""
    events = get_events_for_period(period, source=_source(source))
    return [e.model_dump() for e in reversed(events)][:limit]


class ImportEvent(BaseModel):
    """One usage row from a remote import client. Deliberately mirrors
    UsageEvent's fields rather than reusing it directly. This is a
    request-body contract at a trust boundary, not an internal type, and it
    omits `id` and `cost_usd`: the id is server-assigned, and cost is always
    recomputed server-side (see import_events) rather than trusted from the
    caller, so a leaked key can misreport tokens but can't forge a cost.
    """
    model: str
    provider: str = "unknown"
    project_tag: str = "default"
    input_tokens: int
    output_tokens: int
    cached_input_tokens: int = 0
    cache_write_tokens: int = 0
    cache_write_1h_tokens: int = 0
    service_tier: str = "standard"
    latency_ms: float = 0.0
    prompt_preview: str = ""
    prompt_hash: str | None = None
    success: bool = True
    error: str | None = None
    source: Source = "manual"
    timestamp: str | None = None  # None -> now, at insert time


class ImportPayload(BaseModel):
    events: list[ImportEvent]


@app.post("/import")
def import_events(
    payload: ImportPayload,
    x_watchdog_import_key: str | None = Header(default=None),
) -> dict:
    """Remote sink for scripts/import_claude_code_usage.py --remote-url, so a
    deployed dashboard's data can be updated without redeploying: run the
    import script anywhere with this URL and key, and the live site reflects
    it on the next page load.

    Disabled entirely (403) unless WATCHDOG_IMPORT_KEY is set in this
    deployment's environment. There is no such thing as a default-open
    write endpoint here. Cost is always recomputed from tokens via this
    project's own pricing table, never trusted from the request body.
    """
    if not IMPORT_KEY:
        raise HTTPException(
            status_code=403,
            detail="Remote import is disabled on this deployment (WATCHDOG_IMPORT_KEY is not set).",
        )
    if x_watchdog_import_key != IMPORT_KEY:
        raise HTTPException(status_code=401, detail="Missing or invalid X-Watchdog-Import-Key header.")

    skipped_unpriced = 0
    total_cost = 0.0
    to_insert: list[UsageEvent] = []
    for e in payload.events:
        if e.model not in PRICING_TABLE:
            skipped_unpriced += 1
            continue
        cost = (
            calculate_cost(
                e.model, e.input_tokens, e.output_tokens, e.cached_input_tokens,
                e.cache_write_tokens, cache_write_1h_tokens=e.cache_write_1h_tokens,
                service_tier=e.service_tier,
            )
            if e.success else 0.0
        )
        event = UsageEvent(
            model=e.model,
            provider=e.provider,
            project_tag=e.project_tag,
            input_tokens=e.input_tokens,
            output_tokens=e.output_tokens,
            cached_input_tokens=e.cached_input_tokens,
            cache_write_tokens=e.cache_write_tokens,
            cache_write_1h_tokens=e.cache_write_1h_tokens,
            service_tier=e.service_tier,
            cost_usd=cost,
            latency_ms=e.latency_ms,
            prompt_preview=e.prompt_preview,
            prompt_hash=e.prompt_hash,
            success=e.success,
            error=e.error,
            source=e.source,
            **({"timestamp": e.timestamp} if e.timestamp else {}),
        )
        to_insert.append(event)
        total_cost += cost

    # One connection, one commit for the whole batch, see log_usage_many's
    # docstring for why this matters against a remote (Turso) database.
    log_usage_many(to_insert)

    return {"logged": len(to_insert), "skipped_unpriced": skipped_unpriced, "total_cost_usd": round(total_cost, 6)}


# Mounted last so the API routes above take precedence over the static catch-all.
@app.get("/shadow")
def shadow_endpoint():
    """Quality evidence for cheap-model routing, grouped by complexity tier."""
    from src.shadow import shadow_summary

    return shadow_summary()


@app.post("/shadow/grade")
def grade_shadows_endpoint(
    limit: int = Query(20, ge=1, le=200),
    tier: str | None = Query(None, pattern="^(trivial|moderate|complex)$"),
    use_llm: bool = Query(True),
):
    """Grade pending shadow comparisons. Deterministic checks first, then a
    local judge. POST because it writes verdicts."""
    from src.judge import grade_pending

    return grade_pending(limit=limit, tier=tier, use_llm=use_llm)


@app.get("/levers")
def levers_endpoint(
    period: str = Query("all_time", pattern=PERIOD_RE),
    source: str | None = Query(None, pattern=SOURCE_RE),
):
    """Rank every cost lever by size, separating realised from hypothetical."""
    from src.levers import analyse_levers

    return analyse_levers(period=period, source=source).as_dict()


@app.get("/findings")
def findings_endpoint():
    """What the ledger says about the cost of building software with an agent.

    Scoped to the measured build work, never mixed with gateway traffic. Every
    number is derived at call time so the page rendering it cannot drift from
    the database behind it.
    """
    from src.findings import compute_findings, counterfactual, load_turns

    # One read, shared by every computation below. Each of these used to fetch
    # the same 4,399 rows for itself, which is free against local SQLite and
    # four network round-trips against Turso.
    turns = load_turns()
    data = compute_findings(events=turns).as_dict()
    data["counterfactuals"] = [
        counterfactual(m, events=turns) for m in ("claude-sonnet-5", "claude-haiku-4-5")
    ]
    # What was actually paid, alongside what the tokens would have cost metered.
    # Carried here so the page cannot render the list price as if it were a
    # charge. That conflation is the specific dishonesty `Source.subscription`
    # was introduced to prevent, and the headline drifted back into it once.
    data["actually_paid"] = subscription_roi("all_time", events=turns)
    return data


@app.get("/sessions")
def sessions_endpoint() -> dict:
    """Your own Claude Code sessions, priced. Local only, by nature.

    This is the ccost CLI's data served over HTTP, so the thing you can act on
    appears where you are already looking instead of only in a terminal.

    It reads ~/.claude/projects, which exists on the machine that did the work
    and nowhere else. On a deployed host there is nothing to read and this says
    so rather than returning an empty shape that looks like "you have no
    sessions". Nothing leaves this process either way: the transcripts are read,
    priced, and summarised in memory.
    """
    import sys

    scripts_dir = Path(__file__).resolve().parent.parent / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))

    try:
        import ccost
    except ImportError as exc:  # pragma: no cover - defensive
        return {"available": False, "reason": f"ccost unavailable: {exc}"}

    def _example() -> dict:
        """The shipped snapshot, plainly marked as somebody else's numbers.

        Without this a deployed copy showed a study and no tool, so the only
        thing on the page a reader could act on was invisible to everyone except
        the one person whose laptop it ran on. `is_example` is not decoration:
        the panel renders differently on it, because presenting another
        machine's session as the reader's own would be a straightforward lie.
        """
        snap = Path(__file__).resolve().parent.parent / "data" / "ccost_snapshot.json"
        if not snap.exists():
            return {"available": False,
                    "reason": "No Claude Code sessions on this machine, and no example shipped."}
        try:
            return {"available": True, "is_example": True, **json.loads(snap.read_text())}
        except (OSError, ValueError) as exc:
            return {"available": False, "reason": f"Example unreadable: {exc}"}

    if not ccost.TRANSCRIPTS.exists():
        return _example()

    sessions = ccost.read_sessions()
    if not sessions:
        return _example()

    current = max(sessions, key=lambda s: s["mtime"])
    turns = current["turns"]
    cutoff = _dt.datetime.now(_dt.timezone.utc) - _dt.timedelta(days=7)
    recent = [
        s for s in sessions
        if _dt.datetime.fromtimestamp(s["mtime"], tz=_dt.timezone.utc) >= cutoff
    ]

    by_project: dict[str, float] = {}
    for s in recent:
        by_project[s["project"]] = by_project.get(s["project"], 0.0) + s["cost"]

    return {
        "available": True,
        "is_example": False,
        "current": {
            "project": current["project"],
            "turns": len(turns),
            "cost_usd": round(current["cost"], 2),
            "context_now": turns[-1],
            "context_start": turns[0],
            "growth": round(turns[-1] / turns[0], 1) if turns[0] else 1.0,
            "should_restart": turns[-1] >= ccost.RESTART_SUGGEST_TOKENS,
        },
        "week": {
            "sessions": len(recent),
            "turns": sum(len(s["turns"]) for s in recent),
            "cost_usd": round(sum(s["cost"] for s in recent), 2),
            "tokens_read": sum(sum(s["turns"]) for s in recent),
            "grown_past_threshold": sum(
                1 for s in recent if s["turns"] and s["turns"][-1] >= ccost.RESTART_SUGGEST_TOKENS
            ),
            "by_project": sorted(
                ({"project": p, "cost_usd": round(c, 2)} for p, c in by_project.items()),
                key=lambda r: -r["cost_usd"],
            )[:8],
        },
        "threshold_tokens": ccost.RESTART_SUGGEST_TOKENS,
    }


@app.get("/validation")
def validation_endpoint() -> dict:
    """Whether the complexity classifier predicts real work. A dated artifact.

    Not computed on request, unlike every other endpoint here. The evidence is
    Claude Code transcripts under ~/.claude/projects, which exist on the machine
    that did the work and nowhere else; a deployed host has no access to them
    and never will. So the analysis runs locally and ships its result:

        venv/bin/python scripts/validate_classifier.py --json data/classifier_validation.json

    That makes it the one number on this site that can go stale, which is why it
    carries `generated_at` and why the page prints that date next to it. Missing
    file is a normal state (a fresh clone, someone else's deployment) and is
    reported as such rather than as an error.
    """
    path = Path(__file__).resolve().parent.parent / "data" / "classifier_validation.json"
    if not path.exists():
        return {"available": False, "reason": "No validation has been run for this deployment."}
    try:
        return {"available": True, **json.loads(path.read_text())}
    except (OSError, ValueError) as exc:
        return {"available": False, "reason": f"Validation artifact unreadable: {exc}"}


@app.get("/complexity")
def complexity_endpoint(prompt: str = Query(..., min_length=1, max_length=20000)):
    """Classify a prompt without calling anything. Free, instant, explainable.

    Exposed so the classifier can be interrogated directly: paste a prompt you
    think was misrouted and see precisely which rules fired.
    """
    from src.complexity import classify

    return classify(prompt).as_dict()


app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
