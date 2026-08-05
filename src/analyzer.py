"""
Reports, anomaly detection, budget checks, and burn-rate projection over
logged UsageEvents.

Everything here reads from SQLite and returns plain data — no LLM calls, no
provider SDKs. That separation is why the digest, the MCP server, and the
dashboard can all share this layer without duplicating logic.
"""

import os
from collections import defaultdict
from datetime import datetime, timedelta, timezone

from src.pricing import calculate_cost, get_rates
from src.tracker import get_events_for_period
from src.usage_schema import AnomalyFlag, CostReport

ROLLING_WINDOW = 20

_PERIOD_DAYS = {"today": 1, "week": 7, "month": 30}


def compute_report(period: str = "week") -> CostReport:
    """Aggregate events for a period into totals and breakdowns.

    Breaks down by model, project, and provider, and computes what prompt
    caching actually saved — the counterfactual cost if every cached token
    had been billed at the full input rate.
    """
    events = get_events_for_period(period)

    total_cost = 0.0
    cache_savings = 0.0
    failed = 0
    by_model: dict[str, float] = defaultdict(float)
    by_project: dict[str, float] = defaultdict(float)
    by_provider: dict[str, float] = defaultdict(float)

    for e in events:
        total_cost += e.cost_usd
        by_model[e.model] += e.cost_usd
        by_project[e.project_tag] += e.cost_usd
        by_provider[e.provider] += e.cost_usd
        if not e.success:
            failed += 1

        if e.cached_input_tokens:
            rates = get_rates(e.model)
            cache_savings += (e.cached_input_tokens / 1000) * (
                rates["input"] - rates["cached_input"]
            )

    return CostReport(
        period=period,
        total_cost_usd=round(total_cost, 6),
        total_calls=len(events),
        failed_calls=failed,
        cache_savings_usd=round(cache_savings, 6),
        breakdown_by_model={k: round(v, 6) for k, v in by_model.items()},
        breakdown_by_project={k: round(v, 6) for k, v in by_project.items()},
        breakdown_by_provider={k: round(v, 6) for k, v in by_provider.items()},
        anomalies=[],
    )


def flag_anomalies(threshold_multiplier: float = 3.0, period: str = "all_time") -> list[AnomalyFlag]:
    """Flag calls whose cost OR latency exceeds `threshold_multiplier` times
    the trailing rolling average for that specific model.

    For each model, events are processed in chronological order. Each event is
    compared against the average cost/latency of the ROLLING_WINDOW (20)
    events immediately preceding it for that same model:

        is_anomaly = (event.cost_usd > threshold_multiplier * rolling_avg_cost)
                     or (event.latency_ms > threshold_multiplier * rolling_avg_latency)

    Comparison is per-model, so a legitimately expensive Opus call is never
    flagged merely for costing more than a Haiku call. The first event for a
    model has no prior history and is never flagged. Failed calls are excluded
    from both the baseline and the flagging — a 429 with 0 cost and 200ms
    latency would otherwise drag the rolling average down and cause the next
    normal call to look anomalous.
    """
    events = [e for e in get_events_for_period(period) if e.success]

    by_model: dict[str, list] = defaultdict(list)
    for e in events:
        by_model[e.model].append(e)

    anomalies: list[AnomalyFlag] = []

    for model, model_events in by_model.items():
        history: list = []
        for event in model_events:
            if history:
                window = history[-ROLLING_WINDOW:]
                rolling_avg_cost = sum(h.cost_usd for h in window) / len(window)
                rolling_avg_latency = sum(h.latency_ms for h in window) / len(window)

                cost_anomaly = rolling_avg_cost > 0 and event.cost_usd > threshold_multiplier * rolling_avg_cost
                latency_anomaly = rolling_avg_latency > 0 and event.latency_ms > threshold_multiplier * rolling_avg_latency

                if cost_anomaly or latency_anomaly:
                    reasons = []
                    if cost_anomaly:
                        reasons.append(
                            f"cost ${event.cost_usd:.6f} is {event.cost_usd / rolling_avg_cost:.1f}x "
                            f"the rolling avg ${rolling_avg_cost:.6f} for {model}"
                        )
                    if latency_anomaly:
                        reasons.append(
                            f"latency {event.latency_ms:.0f}ms is {event.latency_ms / rolling_avg_latency:.1f}x "
                            f"the rolling avg {rolling_avg_latency:.0f}ms for {model}"
                        )
                    severity = "high" if (cost_anomaly and latency_anomaly) else "medium"
                    anomalies.append(
                        AnomalyFlag(
                            usage_event_id=event.id,
                            reason="; ".join(reasons),
                            severity=severity,
                        )
                    )
            history.append(event)

    return anomalies


def check_budget_status(period: str = "weekly") -> dict:
    """Compare current period spend against the configured budget.

    "near" = spend is at or over 80% of the limit but under 100%.
    """
    limit_usd = float(os.environ.get("WEEKLY_BUDGET_USD", "5.00"))

    period_map = {"daily": "today", "weekly": "week", "monthly": "month"}
    report = compute_report(period_map.get(period, "week"))

    spend = report.total_cost_usd
    percent_used = round((spend / limit_usd) * 100, 2) if limit_usd > 0 else 0.0

    if spend >= limit_usd:
        status = "over"
    elif spend >= 0.8 * limit_usd:
        status = "near"
    else:
        status = "under"

    return {
        "status": status,
        "percent_used": percent_used,
        "remaining_usd": round(limit_usd - spend, 6),
        "spend_usd": round(spend, 6),
        "limit_usd": limit_usd,
        "period": period,
    }


def project_burn_rate(period: str = "week") -> dict:
    """Project when current spend will exhaust the weekly budget.

    Answers the question a raw total can't: "am I on track?" Uses the observed
    spend rate over `period` to extrapolate a daily rate and a projected
    7-day total, then estimates the date the budget runs out.

    Extrapolation is only as good as the window — a burst of activity in a
    short window projects a misleadingly high rate, so `confidence` reports
    how much data the projection rests on.
    """
    events = [e for e in get_events_for_period(period) if e.success]
    limit_usd = float(os.environ.get("WEEKLY_BUDGET_USD", "5.00"))

    if not events:
        # Same key set as the populated path — a caller reading
        # `calls_observed` must not KeyError just because there was no traffic.
        return {
            "period": period,
            "observed_days": 0.0,
            "calls_observed": 0,
            "spend_so_far_usd": 0.0,
            "daily_burn_usd": 0.0,
            "projected_weekly_usd": 0.0,
            "budget_limit_usd": limit_usd,
            "projected_percent_of_budget": 0.0,
            "days_until_budget_exhausted": None,
            "exhaustion_date": None,
            "on_track": True,
            "confidence": "none",
            "note": "No successful calls in this period — nothing to project from.",
        }

    total = sum(e.cost_usd for e in events)

    # Span the data actually covers, not the nominal period — 3 days of data
    # in a 7-day window should produce a 3-day rate, not a 7-day one.
    timestamps = sorted(e.timestamp for e in events)
    first = _parse_ts(timestamps[0])
    last = _parse_ts(timestamps[-1])
    observed_days = max((last - first).total_seconds() / 86400, 1 / 24)  # floor at 1h

    daily = total / observed_days
    projected_weekly = daily * 7
    pct = round((projected_weekly / limit_usd) * 100, 2) if limit_usd > 0 else 0.0

    days_left = None
    exhaustion_date = None
    if daily > 0 and limit_usd > 0:
        remaining = max(limit_usd - total, 0.0)
        days_left = round(remaining / daily, 2)
        exhaustion_date = (datetime.now(timezone.utc) + timedelta(days=days_left)).strftime("%Y-%m-%d")

    if observed_days >= 3 and len(events) >= 20:
        confidence = "high"
    elif observed_days >= 1 and len(events) >= 5:
        confidence = "medium"
    else:
        confidence = "low"

    return {
        "period": period,
        "observed_days": round(observed_days, 2),
        "calls_observed": len(events),
        "spend_so_far_usd": round(total, 6),
        "daily_burn_usd": round(daily, 6),
        "projected_weekly_usd": round(projected_weekly, 6),
        "budget_limit_usd": limit_usd,
        "projected_percent_of_budget": pct,
        "days_until_budget_exhausted": days_left,
        "exhaustion_date": exhaustion_date,
        "on_track": projected_weekly <= limit_usd,
        "confidence": confidence,
    }


def provider_breakdown(period: str = "week") -> list[dict]:
    """Per-provider cost, call volume, tokens, latency, and cache hit rate.

    The view that only exists once tracking is multi-provider: which vendor is
    actually costing you money, and which one is slow.
    """
    events = get_events_for_period(period)
    grouped: dict[str, list] = defaultdict(list)
    for e in events:
        grouped[e.provider].append(e)

    rows = []
    for provider, items in grouped.items():
        ok = [e for e in items if e.success]
        in_tokens = sum(e.input_tokens for e in ok)
        cached = sum(e.cached_input_tokens for e in ok)
        rows.append(
            {
                "provider": provider,
                "total_cost_usd": round(sum(e.cost_usd for e in items), 6),
                "calls": len(items),
                "failed_calls": len(items) - len(ok),
                "input_tokens": in_tokens,
                "output_tokens": sum(e.output_tokens for e in ok),
                "cached_input_tokens": cached,
                "cache_hit_rate": round(cached / in_tokens, 4) if in_tokens else 0.0,
                "avg_latency_ms": round(sum(e.latency_ms for e in ok) / len(ok), 1) if ok else 0.0,
                "models_used": sorted({e.model for e in items}),
            }
        )

    rows.sort(key=lambda r: r["total_cost_usd"], reverse=True)
    return rows


def what_if_switched(from_model: str, to_model: str, period: str = "week") -> dict:
    """Re-price this period's real traffic as if it had run on another model.

    Turns "should I switch models?" from a guess into arithmetic, using your
    own token counts rather than a vendor's benchmark.
    """
    events = [
        e for e in get_events_for_period(period) if e.success and e.model == from_model
    ]
    if not events:
        return {
            "from_model": from_model,
            "to_model": to_model,
            "calls_repriced": 0,
            "note": f"No successful {from_model} calls in this period.",
        }

    actual = sum(e.cost_usd for e in events)
    hypothetical = sum(
        calculate_cost(
            to_model, e.input_tokens, e.output_tokens, e.cached_input_tokens, e.cache_write_tokens
        )
        for e in events
    )
    delta = actual - hypothetical

    return {
        "from_model": from_model,
        "to_model": to_model,
        "period": period,
        "calls_repriced": len(events),
        "actual_cost_usd": round(actual, 6),
        "hypothetical_cost_usd": round(hypothetical, 6),
        "savings_usd": round(delta, 6),
        "savings_percent": round((delta / actual) * 100, 2) if actual else 0.0,
        "verdict": "cheaper" if delta > 0 else ("more expensive" if delta < 0 else "identical"),
    }


def _parse_ts(value: str) -> datetime:
    dt = datetime.fromisoformat(value)
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
