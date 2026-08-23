"""Prompt complexity classification, the input to content-aware routing.

Every other routing strategy in this project ranks models by something
*measured about the model*: its price, its latency, its failure rate. None of
them look at what is being asked. That is the gap this closes, and it is the
one routing input that cannot be read off a price list or a ledger.

    "reformat this JSON"                -> trivial   -> the free local model
    "why is this test flaky?"            -> moderate  -> a mid-tier model
    "redesign this module's boundaries"  -> complex   -> the frontier model

Three commitments shape the implementation:

**It is heuristic, not a model call.** A classifier that costs an API call to
decide which API to call is a tax on every request, and on cheap prompts it
costs more than it saves. This runs in microseconds and spends nothing, so the
router's overhead is never the reason routing was a bad idea. That constraint
is a feature: it forces the signals to be legible.

**It reports its reasoning.** `signals` lists every rule that fired and what it
contributed. A routing decision that sent your architecture question to a 3B
model must be auditable after the fact, otherwise the only way to discover a
misroute is to notice the answer was bad.

**It is deliberately biased upward.** Misrouting a hard prompt to a weak model
produces a confidently wrong answer that costs you an hour. Misrouting an easy
prompt to a strong model costs a fraction of a cent. Those errors are not
symmetric and the thresholds do not pretend they are: ambiguous prompts escalate.

Known limits, stated rather than buried: this reads English, it reads the
prompt only (not the conversation it sits in), and its verb lists are a
judgement call rather than a trained boundary. It is a cheap prior, not an
oracle. `src/shadow.py` exists to measure how good a prior it actually is,
against real traffic, instead of assuming.
"""

import re
from dataclasses import dataclass, field

TIER_TRIVIAL, TIER_MODERATE, TIER_COMPLEX = "trivial", "moderate", "complex"
TIERS = (TIER_TRIVIAL, TIER_MODERATE, TIER_COMPLEX)

# Score boundaries. Tuned so an unadorned short question lands in `moderate`
# rather than `trivial`: reaching the cheap tier should require positive
# evidence of simplicity, not merely the absence of evidence of difficulty.
TRIVIAL_MAX_SCORE = -2
COMPLEX_MIN_SCORE = 3

# Verbs that describe reshaping a system rather than answering about it. These
# are the strongest single signal available from a prompt's surface, an
# architectural request almost always contains one.
# Stems rather than whole words: "migrate" would miss "migration", and a
# classifier that depends on which inflection someone happened to type is
# brittle in a way nobody would ever debug.
_COMPLEX_VERBS = (
    "architect", "design", "refactor", "restructur", "migrat",
    "optimi", "debug", "diagnos", "investigat", "trade-off", "tradeoff",
    "scalab", "benchmark", "profil", "audit", "evaluat", "concurren",
    "why does", "why is", "why did", "root cause", "race condition",
    "memory leak", "deadlock", "vulnerab", "bottleneck", "idempot",
)

# Verbs describing a mechanical transformation with one obvious right answer.
_TRIVIAL_VERBS = (
    "format", "reformat", "indent", "prettify", "rename", "capitalize",
    "lowercase", "uppercase", "sort", "reverse", "count", "convert",
    "translate", "spell", "list the", "what does", "what is the syntax",
    "boilerplate", "scaffold", "stub", "getter", "setter", "docstring",
    "add a comment", "add comments", "typo",
)

# Phrasings that promise multi-step reasoning regardless of the verb used.
_MULTI_STEP = (
    "step by step", "first,", "then,", "after that", "and then", "finally,",
    "explain why", "walk me through", "pros and cons", "options",
    "alternative", "instead of", "should i", "which approach", "best way",
)

_CODE_FENCE = re.compile(r"```")
_QUESTION = re.compile(r"\?")
# A stack trace or error dump is a debugging prompt even with no verb at all.
_TRACEBACK = re.compile(
    r"traceback \(most recent call last\)|at [\w.$]+\([\w.]+\.java:\d+\)|"
    r"^\s+at .+:\d+:\d+$|error:|exception:|panic:|segmentation fault",
    re.IGNORECASE | re.MULTILINE,
)

# Length thresholds in characters. Long prompts carry more context to reconcile;
# very short ones are usually lookups.
SHORT_PROMPT_CHARS = 120
LONG_PROMPT_CHARS = 1500
VERY_LONG_PROMPT_CHARS = 6000


@dataclass
class ComplexityVerdict:
    """A tier, the score behind it, and every rule that moved that score."""

    tier: str
    score: int
    signals: list[str] = field(default_factory=list)
    prompt_chars: int = 0

    def as_dict(self) -> dict:
        return {
            "tier": self.tier,
            "score": self.score,
            "signals": self.signals,
            "prompt_chars": self.prompt_chars,
        }

    def __str__(self) -> str:
        return f"{self.tier} (score {self.score:+d}: {'; '.join(self.signals) or 'no signals'})"


def _count_hits(haystack: str, needles) -> list[str]:
    return [n for n in needles if n in haystack]


def classify(prompt: str) -> ComplexityVerdict:
    """Score a prompt's difficulty and bucket it into a tier.

    Positive score means harder. Every rule that fires appends a human-readable
    line to `signals`, so the total can always be reconstructed by hand.
    """
    text = (prompt or "").strip()
    lowered = text.lower()
    chars = len(text)
    score = 0
    signals: list[str] = []

    if not text:
        # An empty prompt is not "easy", it is malformed. Escalating costs a
        # fraction of a cent; a local model silently answering nothing does not
        # surface the bug.
        return ComplexityVerdict(tier=TIER_MODERATE, score=0, signals=["empty prompt"], prompt_chars=0)

    # --- vocabulary ------------------------------------------------------
    # Scored before length, because whether a prompt is *about* something hard
    # is a far stronger signal than how many characters it took to say it.
    hard_signal = False

    if hits := _count_hits(lowered, _COMPLEX_VERBS):
        score += 2 * len(hits[:2])  # capped: three synonyms are not three reasons
        signals.append(f"reasoning verbs {hits[:2]} +{2 * len(hits[:2])}")
        hard_signal = True

    if hits := _count_hits(lowered, _TRIVIAL_VERBS):
        score -= 2 * len(hits[:2])
        signals.append(f"mechanical verbs {hits[:2]} -{2 * len(hits[:2])}")

    if hits := _count_hits(lowered, _MULTI_STEP):
        score += 2
        signals.append(f"multi-step phrasing {hits[:2]} +2")
        hard_signal = True

    # --- structure -------------------------------------------------------
    if _TRACEBACK.search(text):
        score += 3
        signals.append("stack trace or error output +3")
        hard_signal = True

    fences = len(_CODE_FENCE.findall(text))
    if fences >= 4:
        # Two or more separate code blocks: usually "reconcile these", which is
        # comparison work, not transformation work.
        score += 2
        signals.append(f"{fences // 2} code blocks +2")
    elif fences >= 2:
        score += 1
        signals.append("contains a code block +1")

    if len(_QUESTION.findall(text)) >= 3:
        score += 1
        signals.append("several questions in one prompt +1")

    # --- length ----------------------------------------------------------
    # Length is the weakest signal here and is scored last, because it is the
    # one most likely to be misleading. "Design a zero-downtime migration
    # strategy" is 50 characters and is not easy work.
    #
    # So the short-prompt discount is *gated*: brevity only argues for the
    # cheap tier when nothing else already argued against it. Long prompts
    # still escalate unconditionally, since the asymmetry runs that way, a
    # long prompt is rarely trivial, but a short one is often hard.
    if chars >= VERY_LONG_PROMPT_CHARS:
        score += 3
        signals.append(f"very long prompt ({chars} chars) +3")
    elif chars >= LONG_PROMPT_CHARS:
        score += 2
        signals.append(f"long prompt ({chars} chars) +2")
    elif chars <= SHORT_PROMPT_CHARS:
        if hard_signal:
            signals.append(f"short prompt ({chars} chars) but a complexity signal fired; no discount")
        else:
            score -= 1
            signals.append(f"short prompt ({chars} chars) -1")

    # --- bucket ----------------------------------------------------------
    if score >= COMPLEX_MIN_SCORE:
        tier = TIER_COMPLEX
    elif score <= TRIVIAL_MAX_SCORE:
        tier = TIER_TRIVIAL
    else:
        tier = TIER_MODERATE

    return ComplexityVerdict(tier=tier, score=score, signals=signals, prompt_chars=chars)


def tier_index(tier: str, n_candidates: int) -> int:
    """Map a tier onto a position in a capability-ordered candidate list.

    The list is ordered cheapest-first, and price is used as a *proxy* for
    capability. That proxy is only defensible inside a group the user curated
    to be interchangeable, which is exactly what a model group is, it is not a
    claim that price predicts quality across vendors in general.

    With fewer candidates than tiers the mapping degrades sensibly rather than
    erroring: two candidates give you a cheap one and a capable one, and one
    candidate makes the whole question moot.
    """
    if n_candidates <= 1:
        return 0
    if tier == TIER_TRIVIAL:
        return 0
    if tier == TIER_COMPLEX:
        return n_candidates - 1
    return (n_candidates - 1) // 2
