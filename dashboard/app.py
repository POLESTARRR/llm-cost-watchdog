"""
FastAPI backing service for the local dashboard, which also serves the
static single-file frontend.

    uvicorn dashboard.app:app --reload --port 8000

The MCP server is the core deliverable; this exists so the same data can be
inspected in a browser without Claude Desktop.
"""

from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles

from src.analyzer import (
    check_budget_status,
    compute_report,
    flag_anomalies,
    project_burn_rate,
    provider_breakdown,
)
from src.guard import guard_status
from src.pricing import compare_models
from src.waste import find_waste
from src.providers import configured_providers
from src.tracker import get_events_for_period, parse_sources, source_totals

STATIC_DIR = Path(__file__).resolve().parent / "static"
PERIOD_RE = "^(today|week|month|all_time)$"

# "all" | "live" | "demo" | "manual" | any comma-separated combination.
SOURCE_RE = "^(all|(live|demo|manual)(,(live|demo|manual))*)$"

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
    # Unset means billed-only here, not "everything" — a budget describes real
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


# Mounted last so the API routes above take precedence over the static catch-all.
app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
