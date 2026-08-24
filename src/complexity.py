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

# Score boundaries, set from measured outcomes rather than from taste.
#
# `TRIVIAL_MAX_SCORE` was -2 on the reasoning that reaching the cheap tier
# should require positive evidence of simplicity. That argument is sound and the
# number it produced was wrong, which is the failure mode of every unvalidated
# heuristic: it sounded careful, so nobody checked it.
#
# scripts/validate_classifier.py checks it, against 1,068 real prompts from
# Claude Code transcripts whose outcome is known: how many tokens the agent
# actually generated in reply, and how many steps it took. Neither is inflated
# by accumulated context the way per-turn cost is, so neither rewards a late
# turn for being late. What that measurement found:
#
#     score  n     median output   median steps
#      -2    19        2,920            4        <- routed cheap
#      -1   499        3,473            5        <- routed mid
#       0   323        9,045            9        <- routed mid
#
# The boundary sat between -2 and -1, where real work differs by 1.2x. The
# actual cliff is between -1 and 0, where it differs by 2.6x, holds at every
# quartile, and survives a permutation test at p < 0.0005. So the old threshold
# split a homogeneous group and left the real seam uncut, which stranded 47% of
# all traffic one point above a tier it belonged in and made the cheap tier
# almost unreachable: 89% of prompts were classified `moderate`.
#
# Moving it to -1 was the obvious conclusion and it is wrong, which the existing
# tests caught: they assert that ambiguity must escalate rather than save money,
# and that assertion survives this data. The -1 group is not a group of easy
# requests, it is the group where **only the length penalty fired**. Nothing
# about those prompts says "simple"; they are merely short. And in an agentic
# session a short prompt routinely means "continue the large thing you are
# doing", so the context carries the difficulty and the wording hides it:
#
#     "Please run this application for me."   ->  216 steps,  57,017 output
#     "let's start from where we live."       ->  158 steps,  94,637 output
#
# 19% of score -1 prompts ran to 20 steps or more. A median says route them
# cheap; that tail says a cheap model would be handed the session's heaviest
# work on no evidence at all, and the cost of being wrong is asymmetric because
# rework is more expensive than the tier upgrade would have been.
#
# So the boundary stays where positive evidence of simplicity is required. The
# measurement's real target is the length penalty below, not this line.
TRIVIAL_MAX_SCORE = -2

# Left at 3. The same measurement shows the complex end separating cleanly
# (score >= 3 runs 13,533 median output against 9,045 at score 0), but the
# per-score samples above 3 are small (n = 8 to 32) and non-monotonic, so there
# is no evidence here that would justify moving it. Unvalidated is not the same
# as wrong, and a threshold is not improved by adjusting it without a reason.
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

# Length thresholds in characters, set from measured outcomes rather than from
# an intuition about what "long" means.
#
# LONG_PROMPT_CHARS was 1500 and VERY_LONG was 6000, which is roughly six times
# too high. Across 1,081 real requests, grouped by how much the person typed:
#
#     <60 chars   ->  4 steps,  2,734 output
#     60-120      ->  6 steps,  4,832
#     120-250     ->  7 steps,  6,860
#     250-500     -> 10 steps, 11,444
#     500-1000    ->  7 steps, 12,584
#     1000+       -> 12 steps, 14,675
#
# The work has already quadrupled by 250 characters, and past that it exceeds
# the median of the complex tier itself. 200/350 was chosen by sweeping the pair
# against these outcomes rather than by picking round numbers: it gives the
# widest separation between tiers (7.9x output, against 7.3x at 250/500) while
# keeping the cheap tier reachable, and it is nowhere near the old 1500/6000. With the old
# numbers a detailed 600-character bug report, several systems and symptoms
# described,
# earned no escalation at all and landed in the middle tier with a score of
# zero. That is the single most common way a real user meets this classifier,
# and it got it wrong every time.
SHORT_PROMPT_CHARS = 120
LONG_PROMPT_CHARS = 200
VERY_LONG_PROMPT_CHARS = 350

# Context-size thresholds, in tokens already in the conversation when the
# request arrives. These are the strongest signal available and the classifier
# ignored them entirely until now, which was the defect behind its 1.8%
# cheap-tier reach.
#
# The words in a prompt are a guess about difficulty. The size of the
# conversation carrying it is a measurement of one, and this project's own data
# says so. Across 1,079 real requests, grouped by how much context existed at
# the moment they were sent:
#
#     context      median steps    median output
#      38K              4             2,143
#     138K              6             5,086
#     271K              8             7,245
#     519K              8             8,403
#
# Monotonic, and available on every single request rather than on the 1.8% where
# a mechanical verb happens to appear. It is also the same fact the cost study
# found from the other direction: 78.5% of the bill is reading, so the thing
# that predicts a request's cost is how much there is to read.
#
# Set between the first and second quartiles, and again above the third, since
# those are where the outcome actually steps rather than at round numbers.
SMALL_CONTEXT_TOKENS = 60_000
LARGE_CONTEXT_TOKENS = 250_000
HUGE_CONTEXT_TOKENS = 450_000


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


def _count_hits(haystack: str, needles, whole_word: bool = False) -> list[str]:
    """Find which needles appear, with matching tightness set by the direction.

    Plain substring matching was fine for the complex verbs, where open stems are
    the point ("migrat" must catch "migration"), and quietly wrong for the
    trivial ones, where it fires inside unrelated words and makes a request look
    EASIER than it is. Real examples from the lists as written: "count" matched
    "account" and "discount", "sort" matched "assortment", "stub" matched
    "stubborn", "spell" matched "misspell".

    That asymmetry is why the tightness is a parameter rather than a constant.
    A complex verb that over-matches escalates a request and costs a fraction of
    a cent. A trivial verb that over-matches sends real work to the cheapest
    model available, so those are anchored at both ends and allowed only the
    ordinary inflections a person would actually type.
    """
    if whole_word:
        def pattern(n: str) -> str:
            stem = re.escape(n)
            # English drops a silent trailing e before -ing: rename -> renaming,
            # translate -> translating. Without this the anchored form misses
            # the most natural way to phrase the request.
            if n.endswith("e"):
                return r"\b(?:%s(?:s|d|ed)?|%sing)\b" % (stem, re.escape(n[:-1]))
            return r"\b%s(?:s|es|ed|ing)?\b" % stem

        return [n for n in needles if re.search(pattern(n), haystack)]
    return [n for n in needles if re.search(r"\b" + re.escape(n), haystack)]


def classify(prompt: str, context_tokens: int | None = None) -> ComplexityVerdict:
    """Score a request's difficulty and bucket it into a tier.

    Positive score means harder. Every rule that fires appends a human-readable
    line to `signals`, so the total can always be reconstructed by hand.

    `context_tokens` is how much conversation already exists when the request
    arrives, which the caller knows and the prompt text does not reveal. Passing
    it is strongly preferred: on measured traffic it is the only signal that
    fires on every request, and reading the words alone reached the cheap tier
    on 1.8% of them. Omitting it keeps the old word-only behaviour, which is
    correct for a first message and for callers that genuinely cannot count.
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

    if hits := _count_hits(lowered, _TRIVIAL_VERBS, whole_word=True):
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
        # No discount, in either direction. The gate above used to allow one
        # when no hard signal had fired, on the theory that brevity with nothing
        # else to say is weak evidence of simplicity. Measured against 1,068 real
        # prompts it is not evidence of anything: this branch alone produced 47%
        # of all traffic at score -1, and 19% of that group ran to 20 steps or
        # more, including "Please run this application for me." at 216 steps.
        #
        # Brevity in an agentic session usually means the context, not the
        # sentence, carries the request. Length still escalates upward above,
        # because that asymmetry does hold: a long prompt is rarely trivial.
        # Prompts that are genuinely simple still reach the cheap tier on the
        # mechanical-verb evidence that actually distinguishes them.
        signals.append(f"short prompt ({chars} chars); no discount, brevity is not simplicity")

    # --- how much is there to read ---------------------------------------
    # Deliberately last, so it adjusts a verdict the words have already argued
    # for rather than overwriting it. A short mechanical request inside a huge
    # session is still not free, and an architectural question at the start of
    # an empty one is still hard.
    if context_tokens is not None:
        if context_tokens >= HUGE_CONTEXT_TOKENS:
            score += 3
            signals.append(f"huge context ({context_tokens:,} tokens) +3")
        elif context_tokens >= LARGE_CONTEXT_TOKENS:
            score += 2
            signals.append(f"large context ({context_tokens:,} tokens) +2")
        elif context_tokens <= SMALL_CONTEXT_TOKENS:
            # Gated, exactly as the short-prompt discount is, and for the same
            # reason. A small conversation is weak evidence of an easy request
            # and it must not cancel strong evidence of a hard one: "Design a
            # migration strategy and explain the trade-offs" is the first thing
            # someone types in a fresh session, and demoting it to a mid model
            # because nothing had been said yet is precisely the misroute this
            # module promises not to make. The existing fallback test caught it.
            #
            # The asymmetry only runs one way. A large context escalates
            # unconditionally above, because a lot to read is never free.
            if hard_signal:
                signals.append(
                    f"small context ({context_tokens:,} tokens) but a complexity signal fired; "
                    "no discount"
                )
            else:
                score -= 2
                signals.append(f"small context ({context_tokens:,} tokens) -2")

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
