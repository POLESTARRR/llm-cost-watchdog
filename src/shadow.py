"""Shadow comparison: measuring whether the cheap model would have been enough.

Every cost tool in this space, including the rest of this project, stops at the
same line. `simulate_routing` says it outright in its own `caveat`: *"it prices
the switch, it does not judge it."* Re-pricing last week's tokens on a cheaper
model tells you what you would have paid. It cannot tell you whether you would
have accepted the answer, and a saving you had to redo by hand is not a saving.

This closes that gap the only way it can honestly be closed: by running the
cheap model on **the same real prompt** and keeping both answers.

    real call  ──► frontier model ──► answer returned to the caller (unchanged)
         │
         └─shadow─► local model ────► answer stored, never returned

Four properties make this affordable to leave switched on:

**The user never waits.** The shadow runs after the real response is already
in the caller's hands. A slow or broken local model cannot delay or fail a real
request; the shadow is wrapped so its exceptions never escape.

**It is free.** The shadow target is a local model, so the comparison costs
nothing per token. The frontier call was going to happen regardless. This is
the whole reason the technique is practical here and isn't elsewhere, a
shadow against a second paid API doubles your bill to study your bill.

**It samples.** `WATCHDOG_SHADOW_RATE=0.05` shadows one call in twenty. Enough
to accumulate a real distribution over a week of ordinary work, cheap enough in
wall-clock CPU that a laptop stays usable.

**It stores prompts, and that is a deliberate exception.** The ledger's rule is
that full prompts are never written to disk (`usage_schema.make_preview`), and
that rule stands. But a quality comparison you cannot re-read is not evidence,
so shadow records live in their own table, are opt-in, and are documented here
as the one place where prompt text is retained. Delete them with
`purge_shadows()`; they are never mixed into `usage_events`.

Scoring is left to a separate, later pass on purpose. Collect first, judge
second: the judgement rubric will change as you look at the data, and rerunning
a scorer over stored text is free while re-collecting the traffic is not.
"""

import json
import logging
import os
import random
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone

from src.tracker import _connect, init_db

logger = logging.getLogger("llm-cost-watchdog")

# Fraction of eligible calls to shadow. 0 disables the feature entirely.
SHADOW_RATE = float(os.environ.get("WATCHDOG_SHADOW_RATE", "0"))

# The model the shadow runs against. Local by default, because a shadow that
# bills per token defeats the purpose.
SHADOW_MODEL = os.environ.get("WATCHDOG_SHADOW_MODEL", "ollama/llama3.2:3b")

# Prompts longer than this are skipped: a 3B local model will not produce a
# meaningful comparison against a 30k-token context, and generating one costs
# minutes of CPU to learn nothing.
MAX_SHADOW_PROMPT_CHARS = int(os.environ.get("WATCHDOG_SHADOW_MAX_CHARS", "8000"))

_SHADOW_SCHEMA = """
CREATE TABLE IF NOT EXISTS shadow_comparisons (
    id TEXT PRIMARY KEY,
    timestamp TEXT NOT NULL,
    project_tag TEXT,
    prompt TEXT NOT NULL,
    prompt_chars INTEGER,
    complexity_tier TEXT,
    complexity_score INTEGER,
    complexity_signals TEXT,
    real_model TEXT NOT NULL,
    real_response TEXT,
    real_cost_usd REAL,
    real_latency_ms REAL,
    shadow_model TEXT NOT NULL,
    shadow_response TEXT,
    shadow_cost_usd REAL,
    shadow_latency_ms REAL,
    shadow_error TEXT,
    -- Filled in later by a scoring pass, deliberately not at capture time.
    verdict TEXT,
    scored_by TEXT
);
CREATE INDEX IF NOT EXISTS idx_shadow_tier ON shadow_comparisons(complexity_tier);
"""


@dataclass
class ShadowResult:
    id: str
    shadow_model: str
    shadow_latency_ms: float
    shadow_error: str | None = None


def _init(db_path: str | None = None) -> None:
    init_db(db_path)
    with _connect(db_path) as conn:
        conn.executescript(_SHADOW_SCHEMA)


def enabled() -> bool:
    return SHADOW_RATE > 0


def should_shadow(prompt: str, rate: float | None = None) -> bool:
    """Whether this particular call gets shadowed.

    Sampling is random rather than every-Nth so a periodic workload (a cron job
    firing the same prompt hourly) cannot land permanently on or off the sample
    and skew the whole dataset toward one kind of work.
    """
    r = SHADOW_RATE if rate is None else rate
    if r <= 0:
        return False
    if len(prompt) > MAX_SHADOW_PROMPT_CHARS:
        return False
    return random.random() < r


def run_shadow(
    prompt: str,
    real_model: str,
    real_response: str,
    real_cost_usd: float,
    real_latency_ms: float,
    project_tag: str = "default",
    shadow_model: str | None = None,
    db_path: str | None = None,
) -> ShadowResult | None:
    """Run the cheap model on the same prompt and store both answers.

    Never raises. This runs after the real response has been handed back, so
    any failure here is a lost data point and nothing more, treating it as a
    request failure would make an observability feature into an outage.
    """
    from src.complexity import classify

    target = shadow_model or SHADOW_MODEL

    # Comparing a model against itself measures nothing, and the resulting rows
    # are worse than absent: they land in the acceptance rate as if they were
    # evidence about cheap-vs-expensive. This is not hypothetical, the first
    # real run collected two of them, because the "real" call had already
    # failed over to the local model and the shadow then re-ran the same model.
    if target == real_model:
        logger.debug("skipping shadow: real and shadow model are both %s", target)
        return None

    _init(db_path)

    verdict = classify(prompt)
    row_id = str(uuid.uuid4())
    shadow_text: str | None = None
    shadow_cost = 0.0
    shadow_error: str | None = None

    start = time.perf_counter()
    try:
        from src.pricing import calculate_cost
        from src.providers import get_provider

        result = get_provider(target).complete(prompt, model=target, temperature=0.3)
        shadow_text = result.text
        shadow_cost = calculate_cost(target, result.input_tokens, result.output_tokens)
    except Exception as exc:  # noqa: BLE001 - deliberate: see docstring
        shadow_error = f"{type(exc).__name__}: {exc}"
        logger.warning("shadow call failed (real call unaffected): %s", shadow_error)
    latency_ms = (time.perf_counter() - start) * 1000

    with _connect(db_path) as conn:
        conn.execute(
            """INSERT INTO shadow_comparisons (
                id, timestamp, project_tag, prompt, prompt_chars,
                complexity_tier, complexity_score, complexity_signals,
                real_model, real_response, real_cost_usd, real_latency_ms,
                shadow_model, shadow_response, shadow_cost_usd, shadow_latency_ms, shadow_error
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                row_id, datetime.now(timezone.utc).isoformat(), project_tag,
                prompt, len(prompt),
                verdict.tier, verdict.score, json.dumps(verdict.signals),
                real_model, real_response, real_cost_usd, real_latency_ms,
                target, shadow_text, shadow_cost, latency_ms, shadow_error,
            ),
        )

    return ShadowResult(
        id=row_id, shadow_model=target, shadow_latency_ms=latency_ms, shadow_error=shadow_error
    )


def shadow_summary(db_path: str | None = None) -> dict:
    """What the shadow dataset shows so far, grouped by complexity tier.

    Reports what was *collected*, and separately how much has been *scored*.
    Keeping those apart matters: 400 unscored comparisons look like evidence
    and are not, they are raw material. The savings figure is the money that
    would not have been spent had these calls gone local, and it is explicitly
    labelled as unverified until the scoring pass says the answers held up.
    """
    _init(db_path)
    with _connect(db_path) as conn:
        rows = conn.execute(
            """SELECT complexity_tier AS tier, COUNT(*) AS n,
                      SUM(shadow_error IS NOT NULL) AS failures,
                      SUM(real_cost_usd) AS real_cost,
                      AVG(real_latency_ms) AS real_ms,
                      AVG(shadow_latency_ms) AS shadow_ms,
                      SUM(verdict IS NOT NULL) AS scored,
                      SUM(verdict = 'acceptable') AS acceptable
               FROM shadow_comparisons GROUP BY complexity_tier"""
        ).fetchall()
        total = conn.execute("SELECT COUNT(*) c FROM shadow_comparisons").fetchone()["c"]

    by_tier = {}
    for r in rows:
        scored = r["scored"] or 0
        by_tier[r["tier"] or "unknown"] = {
            "comparisons": r["n"],
            "shadow_failures": r["failures"] or 0,
            "real_cost_usd": round(r["real_cost"] or 0.0, 6),
            "avg_real_latency_ms": round(r["real_ms"] or 0.0, 1),
            "avg_shadow_latency_ms": round(r["shadow_ms"] or 0.0, 1),
            "scored": scored,
            "acceptable": r["acceptable"] or 0,
            "acceptance_rate": (
                round((r["acceptable"] or 0) / scored, 3) if scored else None
            ),
        }

    return {
        "total_comparisons": total,
        "shadow_model": SHADOW_MODEL,
        "sample_rate": SHADOW_RATE,
        "by_tier": by_tier,
        "unverified_savings_usd": round(sum(t["real_cost_usd"] for t in by_tier.values()), 6),
        "note": (
            "`unverified_savings_usd` is what these calls cost on the real model, i.e. what "
            "routing them locally would have avoided. It is NOT a saving until `scored` "
            "covers them and `acceptance_rate` says the local answers were good enough."
        ),
    }


def pending_review(limit: int = 20, tier: str | None = None, db_path: str | None = None) -> list[dict]:
    """Unscored comparisons, for a human or a judge to grade."""
    _init(db_path)
    sql = "SELECT * FROM shadow_comparisons WHERE verdict IS NULL AND shadow_error IS NULL"
    params: list = []
    if tier:
        sql += " AND complexity_tier = ?"
        params.append(tier)
    sql += " ORDER BY timestamp DESC LIMIT ?"
    params.append(limit)
    with _connect(db_path) as conn:
        return [dict(r) for r in conn.execute(sql, params).fetchall()]


def record_verdict(
    shadow_id: str, verdict: str, scored_by: str = "human", db_path: str | None = None
) -> None:
    """Grade one comparison. `verdict` is 'acceptable' or 'inadequate'."""
    if verdict not in ("acceptable", "inadequate"):
        raise ValueError("verdict must be 'acceptable' or 'inadequate'")
    _init(db_path)
    with _connect(db_path) as conn:
        conn.execute(
            "UPDATE shadow_comparisons SET verdict = ?, scored_by = ? WHERE id = ?",
            (verdict, scored_by, shadow_id),
        )


def purge_shadows(db_path: str | None = None) -> int:
    """Delete every stored comparison, including the prompt text they hold."""
    _init(db_path)
    with _connect(db_path) as conn:
        n = conn.execute("SELECT COUNT(*) c FROM shadow_comparisons").fetchone()["c"]
        conn.execute("DELETE FROM shadow_comparisons")
    return n
