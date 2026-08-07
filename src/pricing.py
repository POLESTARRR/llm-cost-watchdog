"""
Multi-provider pricing table and cost calculator.

Rates are USD **per 1,000 tokens**, derived from published per-million
pricing. Three rates per model, because a prompt is not billed uniformly:

  input          — uncached prompt tokens, full rate
  cached_input   — prompt tokens served from a provider-side cache
  output         — generated tokens

The cached rate is the reason this file has three columns instead of two.
Prompt caching is the single biggest lever on real LLM spend — Anthropic
cache reads bill at ~0.1x input, OpenAI at ~0.1x — and a tracker that
prices every input token at the full rate will overstate a cache-heavy
workload's cost several times over. Almost no cost dashboard models this.

Adding a model is one entry. Adding a *provider* is one entry plus an
adapter in src/providers/ — see providers/base.py.

Prices verified against provider pricing pages as of 2026-08. Rerun
`python -m src.pricing --check` to print the table for review.
"""

from src.providers import infer_provider

# USD per 1,000 tokens.
PRICING_TABLE: dict[str, dict[str, float]] = {
    # --- Anthropic -------------------------------------------------------
    # $/MTok: Fable 5 10/50 · Opus 5 & 4.8 5/25 · Sonnet 5 3/15 · Haiku 4.5 1/5
    # Cache reads bill at ~0.1x the input rate.
    "claude-fable-5": {"input": 0.010, "cached_input": 0.0010, "output": 0.050},
    "claude-opus-5": {"input": 0.005, "cached_input": 0.0005, "output": 0.025},
    "claude-opus-4-8": {"input": 0.005, "cached_input": 0.0005, "output": 0.025},
    "claude-sonnet-5": {"input": 0.003, "cached_input": 0.0003, "output": 0.015},
    "claude-haiku-4-5": {"input": 0.001, "cached_input": 0.0001, "output": 0.005},

    # --- OpenAI ----------------------------------------------------------
    # GPT-5.6 family (current gen). `gpt-5.6` aliases to sol.
    # These carry two extra billing rules — see LONG_CONTEXT_MODELS and
    # CACHE_WRITE_MULTIPLIER below.
    "gpt-5.6-sol": {"input": 0.005, "cached_input": 0.0005, "output": 0.030},
    "gpt-5.6-terra": {"input": 0.002, "cached_input": 0.0002, "output": 0.012},
    "gpt-5.6-luna": {"input": 0.0002, "cached_input": 0.00002, "output": 0.0012},
    # Prior generations, still callable.
    "gpt-5.4": {"input": 0.0025, "cached_input": 0.00025, "output": 0.015},
    "gpt-5.4-mini": {"input": 0.00075, "cached_input": 0.000075, "output": 0.0045},
    "gpt-5.4-nano": {"input": 0.0002, "cached_input": 0.00002, "output": 0.00125},
    "gpt-5-mini": {"input": 0.00025, "cached_input": 0.000025, "output": 0.002},
    "gpt-5-nano": {"input": 0.00005, "cached_input": 0.000005, "output": 0.0004},

    # --- Google ----------------------------------------------------------
    # "-latest" aliases: the pinned gemini-1.5-* generation is retired, and
    # aliases survive Google rolling models forward.
    "gemini-flash-latest": {"input": 0.0003, "cached_input": 0.000075, "output": 0.0025},
    "gemini-pro-latest": {"input": 0.00125, "cached_input": 0.0003125, "output": 0.010},
    "gemini-flash-lite-latest": {"input": 0.0001, "cached_input": 0.000025, "output": 0.0004},
    # Verified against ai.google.dev/gemini-api/docs/pricing 2026-08-07:
    # $0.25 / $0.025 / $1.50 per 1M tokens (input / cached input / output).
    "gemini-3.1-flash-lite": {"input": 0.00025, "cached_input": 0.000025, "output": 0.0015},
}

DEFAULT_MODEL = "gemini-flash-lite-latest"

# Used when a model isn't in the table, so a tracking call never crashes on an
# unknown model — we'd rather log an approximate cost than lose the event.
_FALLBACK_RATES = {"input": 0.001, "cached_input": 0.0001, "output": 0.005}

# --- Provider-specific billing rules ------------------------------------
#
# Two rules that most cost dashboards silently get wrong. Both are OpenAI
# GPT-5.6-family behaviors; keeping them as data (not buried in an if) means
# a new model with the same rules is still a one-line addition.

# Prompts above this many input tokens are surcharged for the ENTIRE request,
# not just the tokens past the threshold. Missing this understates a
# long-context workload by ~2x.
LONG_CONTEXT_THRESHOLD = 272_000
LONG_CONTEXT_INPUT_MULTIPLIER = 2.0
LONG_CONTEXT_OUTPUT_MULTIPLIER = 1.5
LONG_CONTEXT_MODELS = ("gpt-5.6",)

# Writing to the prompt cache is not free on these models: it bills at a
# premium over the uncached input rate. Cache writes are what make the first
# call of a cached workload *more* expensive, not less.
CACHE_WRITE_MULTIPLIER = 1.25
CACHE_WRITE_BILLED_MODELS = ("gpt-5.6",)


def get_rates(model: str) -> dict[str, float]:
    """Return the per-1k rates for `model`, falling back rather than raising."""
    return PRICING_TABLE.get(model, _FALLBACK_RATES)


def has_long_context_surcharge(model: str, input_tokens: int) -> bool:
    return model.startswith(LONG_CONTEXT_MODELS) and input_tokens > LONG_CONTEXT_THRESHOLD


def bills_cache_writes(model: str) -> bool:
    return model.startswith(CACHE_WRITE_BILLED_MODELS)


def calculate_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
    cached_input_tokens: int = 0,
    cache_write_tokens: int = 0,
) -> float:
    """USD cost of one call.

    `input_tokens` is the FULL prompt including any cached or cache-written
    portion; `cached_input_tokens` and `cache_write_tokens` are subsets of it.
    Each subset bills at its own rate and the remainder at the full input rate,
    so passing a token in both `input_tokens` and a subset argument is correct
    and expected.

    `output_tokens` should already include reasoning tokens — every provider
    reports them inside the output count, so adding them separately
    double-counts.
    """
    rates = get_rates(model)

    in_mult = out_mult = 1.0
    if has_long_context_surcharge(model, input_tokens):
        in_mult = LONG_CONTEXT_INPUT_MULTIPLIER
        out_mult = LONG_CONTEXT_OUTPUT_MULTIPLIER

    # Subsets can't exceed the whole, and can't overlap each other.
    cached = min(cached_input_tokens, input_tokens)
    written = min(cache_write_tokens, input_tokens - cached)
    uncached = input_tokens - cached - written

    cost = (uncached / 1000) * rates["input"] * in_mult
    cost += (cached / 1000) * rates["cached_input"] * in_mult
    if written:
        write_rate = rates["input"] * (CACHE_WRITE_MULTIPLIER if bills_cache_writes(model) else 1.0)
        cost += (written / 1000) * write_rate * in_mult
    cost += (output_tokens / 1000) * rates["output"] * out_mult

    return round(cost, 8)


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> dict:
    """What one call would cost on `model`, with no cache assumed.

    Used by the compare_model_costs MCP tool to answer "what would this
    prompt cost on each model?" without making any API calls.
    """
    return {
        "model": model,
        "provider": _safe_provider(model),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_usd": calculate_cost(model, input_tokens, output_tokens),
        "priced": model in PRICING_TABLE,
    }


def compare_models(
    input_tokens: int, output_tokens: int, models: list[str] | None = None
) -> list[dict]:
    """Price the same call across models, cheapest first."""
    targets = models or list(PRICING_TABLE)
    rows = [estimate_cost(m, input_tokens, output_tokens) for m in targets]
    rows.sort(key=lambda r: r["cost_usd"])

    if rows:
        cheapest = rows[0]["cost_usd"] or 1e-12
        for r in rows:
            r["vs_cheapest"] = round(r["cost_usd"] / cheapest, 2)
    return rows


def _safe_provider(model: str) -> str:
    try:
        return infer_provider(model)
    except Exception:
        return "unknown"


def models_by_provider() -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for model in PRICING_TABLE:
        grouped.setdefault(_safe_provider(model), []).append(model)
    return grouped


if __name__ == "__main__":
    print(f"{'model':<28} {'provider':<10} {'in':>10} {'cached':>10} {'out':>10}")
    for model, r in PRICING_TABLE.items():
        print(
            f"{model:<28} {_safe_provider(model):<10} "
            f"{r['input']:>10.6f} {r['cached_input']:>10.6f} {r['output']:>10.6f}"
        )
