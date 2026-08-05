"""
Spend guardrails — the part that stops money being spent, rather than
reporting on it afterwards.

Observability tells you what a runaway agent loop cost you. A guardrail
stops it at call 40 instead of call 4,000. That difference is the whole
point of this module: everything else in this project is read-only history,
this is the one piece that intervenes.

Three independent protections, each of which can be off / warn / block:

  1. Budget      — spend for the period vs. your configured limit
  2. Circuit     — too many calls in a short window (a loop that won't stop)
  3. Per-call    — a single call whose estimated cost is implausibly large

Configuration (all optional, all via env):

    WATCHDOG_GUARD_MODE=warn      # off | warn | block   (default: warn)
    WEEKLY_BUDGET_USD=5.00
    WATCHDOG_MAX_CALLS_PER_MIN=60
    WATCHDOG_MAX_COST_PER_CALL=1.00
    WATCHDOG_PROJECT_CAPS=job-search-agent:2.00,teaching-workshop:1.00

Default mode is `warn`, deliberately. A tracking wrapper that silently
starts refusing calls would be worse than the problem it solves — you opt
into blocking.
"""

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from src.pricing import calculate_cost
from src.tracker import get_events_for_period

logger = logging.getLogger("llm-cost-watchdog")

MODE_OFF, MODE_WARN, MODE_BLOCK = "off", "warn", "block"


class BudgetExceededError(RuntimeError):
    """Raised when a guardrail blocks a call in `block` mode."""

    def __init__(self, verdict: "GuardVerdict"):
        self.verdict = verdict
        super().__init__(verdict.message)


@dataclass
class GuardVerdict:
    allowed: bool
    mode: str
    triggered: list[str] = field(default_factory=list)
    message: str = ""
    details: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "allowed": self.allowed,
            "mode": self.mode,
            "triggered": self.triggered,
            "message": self.message,
            **self.details,
        }


def _mode() -> str:
    mode = os.environ.get("WATCHDOG_GUARD_MODE", MODE_WARN).strip().lower()
    return mode if mode in (MODE_OFF, MODE_WARN, MODE_BLOCK) else MODE_WARN


def _project_caps() -> dict[str, float]:
    """Parse `WATCHDOG_PROJECT_CAPS=proj:1.50,other:0.25` into a dict.

    A malformed entry is skipped with a warning rather than crashing the
    call it was meant to protect.
    """
    caps: dict[str, float] = {}
    raw = os.environ.get("WATCHDOG_PROJECT_CAPS", "").strip()
    if not raw:
        return caps
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        name, _, value = chunk.rpartition(":")
        try:
            caps[name.strip()] = float(value)
        except ValueError:
            logger.warning("ignoring malformed WATCHDOG_PROJECT_CAPS entry: %r", chunk)
    return caps


def check_guards(
    project_tag: str = "default",
    model: str | None = None,
    estimated_input_tokens: int = 0,
    estimated_output_tokens: int = 0,
) -> GuardVerdict:
    """Evaluate every guardrail against current state. Never raises.

    Called before an LLM request goes out. Returns a verdict; it is the
    caller's job (call_llm) to honor `allowed`.
    """
    mode = _mode()
    if mode == MODE_OFF:
        return GuardVerdict(allowed=True, mode=mode, message="guards disabled")

    triggered: list[str] = []
    details: dict = {}
    reasons: list[str] = []

    events = get_events_for_period("week")
    spend = sum(e.cost_usd for e in events)
    limit = float(os.environ.get("WEEKLY_BUDGET_USD", "5.00"))

    # --- 1. weekly budget ------------------------------------------------
    details["weekly_spend_usd"] = round(spend, 6)
    details["weekly_limit_usd"] = limit
    if limit > 0 and spend >= limit:
        triggered.append("weekly_budget")
        reasons.append(f"weekly spend ${spend:.4f} has reached the ${limit:.2f} budget")

    # --- 2. per-project cap ----------------------------------------------
    cap = _project_caps().get(project_tag)
    if cap is not None:
        project_spend = sum(e.cost_usd for e in events if e.project_tag == project_tag)
        details["project_spend_usd"] = round(project_spend, 6)
        details["project_cap_usd"] = cap
        if project_spend >= cap:
            triggered.append("project_cap")
            reasons.append(
                f"project '{project_tag}' spend ${project_spend:.4f} has reached its ${cap:.2f} cap"
            )

    # --- 3. runaway-loop circuit breaker ---------------------------------
    # The scenario this exists for: an agent loop with no exit condition,
    # discovered the next morning. Rate is the only early signal — cost
    # lags far behind call volume on cheap models.
    max_per_min = int(os.environ.get("WATCHDOG_MAX_CALLS_PER_MIN", "60"))
    if max_per_min > 0:
        cutoff = datetime.now(timezone.utc) - timedelta(minutes=1)
        recent = sum(1 for e in events if _parse_ts(e.timestamp) >= cutoff)
        details["calls_last_minute"] = recent
        details["max_calls_per_minute"] = max_per_min
        if recent >= max_per_min:
            triggered.append("rate_limit")
            reasons.append(f"{recent} calls in the last minute (limit {max_per_min}) — possible runaway loop")

    # --- 4. implausibly expensive single call ----------------------------
    max_call = float(os.environ.get("WATCHDOG_MAX_COST_PER_CALL", "1.00"))
    if model and max_call > 0 and (estimated_input_tokens or estimated_output_tokens):
        est = calculate_cost(model, estimated_input_tokens, estimated_output_tokens)
        details["estimated_call_cost_usd"] = est
        details["max_cost_per_call_usd"] = max_call
        if est > max_call:
            triggered.append("per_call_cost")
            reasons.append(f"this call is estimated at ${est:.4f}, over the ${max_call:.2f} per-call limit")

    if not triggered:
        return GuardVerdict(allowed=True, mode=mode, message="ok", details=details)

    message = "; ".join(reasons)
    allowed = mode != MODE_BLOCK
    return GuardVerdict(
        allowed=allowed,
        mode=mode,
        triggered=triggered,
        message=message,
        details=details,
    )


def enforce(
    project_tag: str = "default",
    model: str | None = None,
    estimated_input_tokens: int = 0,
    estimated_output_tokens: int = 0,
) -> GuardVerdict:
    """Check guardrails and act: log a warning, or raise in `block` mode."""
    verdict = check_guards(project_tag, model, estimated_input_tokens, estimated_output_tokens)

    if verdict.triggered:
        if verdict.allowed:
            logger.warning("guardrail (warn): %s", verdict.message)
        else:
            logger.error("guardrail (block): %s", verdict.message)
            raise BudgetExceededError(verdict)

    return verdict


def guard_status() -> dict:
    """Current guardrail configuration and how close each is to tripping.

    Exposed as an MCP tool so you can ask "how much runway do I have?"
    without waiting to be blocked.
    """
    verdict = check_guards()
    d = verdict.details
    spend = d.get("weekly_spend_usd", 0.0)
    limit = d.get("weekly_limit_usd", 0.0)

    return {
        "mode": verdict.mode,
        "mode_meaning": {
            MODE_OFF: "guards disabled — nothing is checked",
            MODE_WARN: "over-limit calls are logged but still sent",
            MODE_BLOCK: "over-limit calls raise BudgetExceededError and are not sent",
        }[verdict.mode],
        "currently_tripped": verdict.triggered,
        "weekly_spend_usd": spend,
        "weekly_limit_usd": limit,
        "weekly_headroom_usd": round(limit - spend, 6),
        "weekly_percent_used": round((spend / limit) * 100, 2) if limit else 0.0,
        "calls_last_minute": d.get("calls_last_minute", 0),
        "max_calls_per_minute": d.get("max_calls_per_minute"),
        "max_cost_per_call_usd": float(os.environ.get("WATCHDOG_MAX_COST_PER_CALL", "1.00")),
        "project_caps": _project_caps(),
    }


def _parse_ts(value: str) -> datetime:
    dt = datetime.fromisoformat(value)
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
