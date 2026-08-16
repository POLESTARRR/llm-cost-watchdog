"""
FastAPI backing service for the local dashboard, which also serves the
static single-file frontend.

    uvicorn dashboard.app:app --reload --port 8000

The MCP server is the core deliverable; this exists so the same data can be
inspected in a browser without Claude Desktop.
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

load_dotenv()

from src.analyzer import (
    check_budget_status,
    compute_report,
    flag_anomalies,
    project_burn_rate,
    provider_breakdown,
)
from src.guard import guard_status
from src.pricing import PRICING_TABLE, calculate_cost, compare_models
from src.waste import find_waste
from src.providers import configured_providers
from src.tracker import get_events_for_period, log_usage_many, parse_sources, source_totals
from src.usage_schema import Source, UsageEvent

STATIC_DIR = Path(__file__).resolve().parent / "static"
PERIOD_RE = "^(today|week|month|all_time)$"

# "all" | any comma-separated combination of the valid sources.
_SRC = "live|demo|manual|subscription"
SOURCE_RE = f"^(all|({_SRC})(,({_SRC}))*)$"

# Set to enable remote import; unset means the endpoint is disabled entirely,
# not merely unauthenticated. See scripts/import_claude_code_usage.py --remote-url.
IMPORT_KEY = os.environ.get("WATCHDOG_IMPORT_KEY")

app = FastAPI(title="LLM Cost Watchdog", version="2.1.0")


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
app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
