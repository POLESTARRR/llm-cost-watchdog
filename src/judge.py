"""Grading shadow comparisons: was the cheap answer good enough?

`src/shadow.py` collects pairs. This turns pairs into a number, which is the
step that converts "we could have saved $X" into "we could have saved $X and
here is the acceptance rate that says whether that was real."

Two graders, tried in that order, because they fail differently:

**Deterministic checks run first and are trusted absolutely.** If the expensive
answer contains Python that parses and the cheap one contains Python that does
not, no further judgement is required or wanted. These checks are cheap, exactly
reproducible, and cannot be talked into anything. They only fire when they
apply, which is a minority of prompts, and they abstain rather than guess.

**A local LLM judge handles the rest**, and is treated as weak evidence that it
is. It runs on the same free local model the shadow used, so grading a week of
comparisons costs nothing. The obvious objection, that a 3B model is being asked
to grade a 3B model, is real and is why:

  - the judge is asked a **narrow comparative** question ("would a developer
    have to redo this?"), not an open-ended quality score, because a small model
    is far better at comparison than at calibration;
  - it never sees which answer came from which model, so it cannot defer to a
    brand name it recognises;
  - the two answers are presented in a **randomised order**, since position bias
    is the best-documented failure mode of LLM judges;
  - its verdicts are recorded as `scored_by="local-judge"`, so any reporting can
    separate them from human grades, and a disagreement is findable later.

The honest summary: deterministic checks are evidence, the local judge is a
triage filter that tells you where to spend your own attention. `WATCHDOG_JUDGE_MODEL`
can point at a stronger model if you have budget for it, and then the same
pipeline produces stronger evidence without any other change.
"""

import ast
import json
import logging
import os
import random
import re

logger = logging.getLogger("llm-cost-gateway")

JUDGE_MODEL = os.environ.get("WATCHDOG_JUDGE_MODEL", "ollama/llama3.2:3b")

ACCEPTABLE, INADEQUATE = "acceptable", "inadequate"

_FENCE = re.compile(r"```(\w+)?\n(.*?)```", re.DOTALL)

_JUDGE_PROMPT = """You are comparing two answers to the same question.

QUESTION:
{prompt}

ANSWER A:
{a}

ANSWER B:
{b}

Would a developer who received ANSWER {target} have to redo the work, compared \
to receiving the other answer?

Reply with exactly one word: EQUIVALENT if answer {target} is about as useful, \
or WORSE if it is meaningfully less useful, wrong, or incomplete.
"""


def _code_blocks(text: str) -> list[tuple[str, str]]:
    return [(m.group(1) or "", m.group(2)) for m in _FENCE.finditer(text or "")]


def _python_parses(text: str) -> bool | None:
    """True/False if this text contains Python, None if it contains none.

    None means "this check does not apply", which is distinct from False and
    must not be collapsed into it: a prose answer with no code is not a
    syntax error.
    """
    blocks = [code for lang, code in _code_blocks(text) if lang in ("", "python", "py")]
    if not blocks:
        return None
    for code in blocks:
        try:
            ast.parse(code)
        except SyntaxError:
            return False
        except ValueError:
            return False
    return True


def _json_parses(text: str) -> bool | None:
    blocks = [code for lang, code in _code_blocks(text) if lang == "json"]
    if not blocks:
        return None
    for code in blocks:
        try:
            json.loads(code)
        except json.JSONDecodeError:
            return False
    return True


def deterministic_verdict(real: str, shadow: str) -> tuple[str, str] | None:
    """Grade without a model, or abstain.

    Returns `(verdict, reason)` or None when no check applies. Abstaining is the
    common case and the correct one, most prompts have no mechanically checkable
    property, and inventing one would be worse than deferring to the judge.
    """
    if not (shadow or "").strip():
        return INADEQUATE, "the cheap model returned an empty answer"

    for name, check in (("Python", _python_parses), ("JSON", _json_parses)):
        real_ok, shadow_ok = check(real), check(shadow)
        if shadow_ok is False and real_ok is not False:
            return INADEQUATE, f"the cheap answer's {name} does not parse and the real one's does"
        if shadow_ok is False and real_ok is False:
            return None  # both broken: not a difference between the models

    # A drastically shorter answer to a question the expensive model treated as
    # substantial. Deliberately conservative: 5x, and only above a floor, so a
    # legitimately terse correct answer is not punished for being efficient.
    real_len, shadow_len = len((real or "").strip()), len((shadow or "").strip())
    if real_len > 400 and shadow_len * 5 < real_len:
        return INADEQUATE, f"cheap answer is {real_len // max(shadow_len, 1)}x shorter ({shadow_len} vs {real_len} chars)"

    return None


def llm_verdict(prompt: str, real: str, shadow: str, model: str | None = None) -> tuple[str, str]:
    """Ask the local model which answer is weaker. Blind and order-randomised."""
    from src.utils import call_llm

    target_is_a = random.random() < 0.5
    a, b = (shadow, real) if target_is_a else (real, shadow)
    target = "A" if target_is_a else "B"

    reply = call_llm(
        _JUDGE_PROMPT.format(prompt=prompt[:2000], a=a[:2000], b=b[:2000], target=target),
        model=model or JUDGE_MODEL,
        project_tag="shadow-judge",
        temperature=0.0,
        # The judge must be able to run when the budget it is reporting on has
        # already tripped: refusing to grade because you overspent is backwards.
        skip_guards=True,
    ).strip().upper()

    if "EQUIVALENT" in reply:
        return ACCEPTABLE, f"local judge: equivalent (answer {target} was the cheap one)"
    if "WORSE" in reply:
        return INADEQUATE, f"local judge: worse (answer {target} was the cheap one)"
    # An unparseable verdict is not a pass. Defaulting to `acceptable` would
    # quietly inflate the acceptance rate every time the judge rambled.
    return INADEQUATE, f"local judge gave no clear verdict ({reply[:60]!r}); counted as inadequate"


def grade_pending(limit: int = 20, tier: str | None = None, use_llm: bool = True,
                  db_path: str | None = None) -> dict:
    """Grade ungraded comparisons and record the verdicts.

    Deterministic checks first; the LLM judge only where they abstain.
    """
    from src.shadow import pending_review, record_verdict

    rows = pending_review(limit=limit, tier=tier, db_path=db_path)
    graded, results = 0, []

    for row in rows:
        real, shadow = row["real_response"] or "", row["shadow_response"] or ""
        outcome = deterministic_verdict(real, shadow)
        scorer = "deterministic"

        if outcome is None:
            if not use_llm:
                continue
            try:
                outcome = llm_verdict(row["prompt"], real, shadow)
                scorer = "local-judge"
            except Exception as exc:  # noqa: BLE001
                logger.warning("judge failed for %s: %s", row["id"], exc)
                continue

        verdict, reason = outcome
        record_verdict(row["id"], verdict, scored_by=scorer, db_path=db_path)
        graded += 1
        results.append({
            "id": row["id"], "tier": row["complexity_tier"],
            "verdict": verdict, "scored_by": scorer, "reason": reason,
        })

    return {
        "graded": graded,
        "remaining": max(len(rows) - graded, 0),
        "results": results,
        "note": (
            "Deterministic verdicts are evidence. Local-judge verdicts are triage: a small "
            "model grading a small model. Filter on scored_by before quoting an acceptance rate."
        ),
    }
