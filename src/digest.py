"""
The weekly agentic loop: pull a cost report + anomalies, ask an LLM to turn
them into a short plain-language digest, save it, and return the text.

This is "agentic" in a narrow, honest sense: it makes an autonomous judgment
call every week, what's normal, what's worth flagging, what's worth doing
about it, without a human framing the question each time. It is not a
multi-step tool-using agent; it's a single well-documented function that
reads structured data and writes a summary. That's the right amount of
machinery for this problem.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from src.analyzer import compute_report, flag_anomalies
from src.utils import call_llm

logger = logging.getLogger("llm-cost-gateway")

REPORTS_DIR = Path(__file__).resolve().parent.parent / "data" / "reports"

DIGEST_MODEL = "gemini-flash-lite-latest"


def _build_prompt(report, anomalies) -> str:
    anomaly_lines = (
        "\n".join(f"- [{a.severity}] {a.reason}" for a in anomalies)
        if anomalies
        else "- none"
    )
    return f"""You are a cost-observability assistant. Write a short, plain-language
weekly digest of LLM API usage based on the data below. Cover exactly these
four things, each in 1-2 sentences:
1. Total spend for the period
2. The single biggest cost driver (by model or project)
3. Any anomalies worth investigating (or say none were found)
4. One concrete recommendation

Keep it under 150 words total. No headers, no bullet points, just plain prose.

PERIOD: {report.period}
TOTAL COST: ${report.total_cost_usd:.6f}
TOTAL CALLS: {report.total_calls}
BREAKDOWN BY MODEL: {json.dumps(report.breakdown_by_model)}
BREAKDOWN BY PROJECT: {json.dumps(report.breakdown_by_project)}

ANOMALIES:
{anomaly_lines}
"""


def _fallback_summary(report, anomalies, exc: Exception) -> str:
    """Deterministic digest used when the LLM call is unavailable."""
    if report.breakdown_by_project:
        top_project = max(report.breakdown_by_project.items(), key=lambda kv: kv[1])
        driver = f"{top_project[0]} (${top_project[1]:.6f})"
    else:
        driver = "no activity"

    lines = [
        f"[Generated without LLM: {type(exc).__name__}]",
        f"Over the last {report.period}, {report.total_calls} calls cost ${report.total_cost_usd:.6f}.",
        f"Biggest cost driver: {driver}.",
    ]
    if anomalies:
        lines.append(f"{len(anomalies)} anomaly/anomalies flagged:")
        lines.extend(f"  - [{a.severity}] {a.reason}" for a in anomalies)
    else:
        lines.append("No anomalies flagged.")
    return "\n".join(lines)


def generate_digest(period: str = "week") -> str:
    """Generate, save, and return the weekly digest text for `period`.

    Saves a JSON record to data/reports/{date}_digest.json containing the
    underlying report, anomalies, and the generated digest text, so the
    digest is auditable after the fact and not just a disappearing string.
    """
    report = compute_report(period)
    anomalies = flag_anomalies()

    prompt = _build_prompt(report, anomalies)
    # A scheduled digest shouldn't lose the whole run to a transient API error
    # (rate limit, outage). Fall back to a deterministic summary so the report
    # still gets written and the numbers are still visible.
    try:
        digest_text = call_llm(prompt, temperature=0.3, model=DIGEST_MODEL, project_tag="cost-watchdog-self")
        llm_written = True
    except Exception as exc:
        logger.warning("digest LLM call failed, falling back to plain summary: %s", str(exc)[:200])
        digest_text = _fallback_summary(report, anomalies, exc)
        llm_written = False

    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / f"{date_str}_digest.json"

    record = {
        "date": date_str,
        "period": period,
        "digest_text": digest_text,
        "llm_written": llm_written,
        "report": report.model_dump(),
        "anomalies": [a.model_dump() for a in anomalies],
    }
    with open(report_path, "w") as f:
        json.dump(record, f, indent=2, default=str)

    logger.info(
        "digest generated | period=%s total_cost=$%.6f total_calls=%d anomalies=%d saved_to=%s",
        period, report.total_cost_usd, report.total_calls, len(anomalies), report_path,
    )

    return digest_text
