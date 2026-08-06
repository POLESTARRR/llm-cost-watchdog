# LLM Cost & Observability Watchdog

A personal cost, latency, and prompt-cache observability layer for LLM API calls across **Anthropic, OpenAI, and Google** — exposed to Claude Desktop as an **MCP server**.

Wrap every LLM call you make in one tracked function, store it, watch it for anomalies and budget overruns, and get an autonomous weekly digest of what you spent and what looked wrong.

There is no chat UI. You talk to it through Claude Desktop ("what did I spend this week?", "would Haiku have been cheaper?"), or open the local dashboard in a browser.

---

## 1. The problem

Every LLM API call has a real cost and a real latency, and almost nobody tracks either until the bill arrives. Portfolio projects show what an agent *can do* — they rarely show what it *costs to run*.

The ones that do track cost usually track it **for one vendor**, and price it **wrong**, because real LLM billing is not `tokens × rate`:

- **A prompt is not billed uniformly.** Cache reads bill at ~10% of the input rate; cache *writes* bill at a **premium** on some models. A tracker that prices every input token at the full rate can overstate a cache-heavy workload several times over — or miss that its first call is the expensive one.
- **Anthropic's `usage.input_tokens` is the uncached remainder**, not the whole prompt. Reading it directly silently under-counts every cached call.
- **OpenAI's GPT-5.6 family surcharges long context.** Past 272K input tokens the *entire* request bills at 2× input / 1.5× output — not just the excess.

This project models all of that. One wrapper function, `call_llm()`, records cost, tokens, latency, and cache usage for every call — successful or failed, on any provider — into SQLite. Everything else reads from that one table.

---

## 2. Architecture

```
   your other projects                     this project's own digest calls
            │                                            │
            └───────────────────┬────────────────────────┘
                                ▼
                    src/utils.py :: call_llm()
   guardrails (can BLOCK) · infers provider from model ID · retries 429s with
   backoff · falls over to another provider · prices it · logs EVERY call
                                │
              ┌─────────────────┼─────────────────┐
              ▼                 ▼                 ▼
      providers/gemini   providers/anthropic  providers/openai
              └─────────────────┼─────────────────┘
                     normalized LLMResponse
                (text, input, output, cached, cache_write)
                                │
                                ▼
                 src/tracker.py ──►  data/usage.db  (SQLite)
                                │     one row per call
                                ▼
              src/analyzer.py · guard.py · waste.py
   reports · anomalies · budget · burn rate · guardrails · waste finding
                                │
            ┌───────────────────┼───────────────────┐
            ▼                   ▼                   ▼
     src/digest.py       src/mcp_server.py    dashboard/app.py
   weekly agentic loop    12 tools, stdio      FastAPI + static HTML
                          → Claude Desktop     → localhost:8000
```

Two design decisions carry the project:

**The wrapper is the product.** `call_llm()` logs before it returns, so there is no code path where a call happens and isn't tracked — including calls that fail, which is exactly when you most want the record.

**The provider adapter is the seam.** Each vendor reports usage in a different shape; the adapters normalize to one `LLMResponse`. Tracker, analyzer, digest, dashboard, and MCP never learn which vendor a call came from. Adding a provider is one adapter + one pricing entry.

---

## 3. The anomaly-detection formula

No black box, no ML. For each model, events are walked in chronological order and each call is compared to the trailing average of the **last 20 calls for that same model**:

```python
window = history[-20:]
rolling_avg_cost    = sum(h.cost_usd   for h in window) / len(window)
rolling_avg_latency = sum(h.latency_ms for h in window) / len(window)

is_anomaly = (event.cost_usd  > threshold_multiplier * rolling_avg_cost) \
          or (event.latency_ms > threshold_multiplier * rolling_avg_latency)
```

- Default `threshold_multiplier` is **3.0**.
- Comparison is **per-model**, so a legitimately expensive Opus call is never flagged just for costing more than a Haiku call.
- **Failed calls are excluded** from both the baseline and the flagging. A 429 logs `cost=0` and low latency; leaving those in drags the rolling average down and makes the *next normal call* look anomalous.
- The **first call for a model is never flagged** — there is no baseline yet.
- Severity is `high` when cost *and* latency both trip, otherwise `medium`.

Implementation: [`src/analyzer.py`](src/analyzer.py) → `flag_anomalies()`.

---

## 4. How to run locally

```bash
git clone <your-repo-url> llm-cost-watchdog
cd llm-cost-watchdog

python3 -m venv venv
source venv/bin/activate            # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env                # add whichever provider keys you use
```

You only need keys for providers you actually call — a missing key disables that provider, it doesn't break the tracker.

Load the synthetic sample data so there's something to look at:

```bash
python -m src.tracker --batch-load sample_usage.json
```

These rows land tagged `source=demo`. They are never counted against your
budget, they can't trip a guardrail, and the dashboard states on its face how
much of the displayed total they account for — see [§8a Provenance](#8a-provenance--is-this-number-real).
Once you have real traffic of your own, `python -m src.tracker --purge demo`
removes them and leaves billed history untouched.

Then any of:

```bash
# CLI: log a call made outside the wrapper (provider inferred from model ID)
python -m src.tracker --log-manual --model claude-opus-5 --cost 0.002 --project teaching-workshop

# Weekly digest (also the cron entrypoint)
python scripts/weekly_digest.py

# Dashboard at http://localhost:8000
uvicorn dashboard.app:app --reload --port 8000

# Or containerized
docker compose up                       # DASHBOARD_PORT=8001 docker compose up  if 8000 is taken

# Tests
pytest tests/ -q
```

### Connecting Claude Desktop to the MCP server

Add this to `claude_desktop_config.json`
(macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "llm-cost-watchdog": {
      "command": "/absolute/path/to/llm-cost-watchdog/venv/bin/python",
      "args": ["-m", "src.mcp_server"],
      "cwd": "/absolute/path/to/llm-cost-watchdog"
    }
  }
}
```

Use absolute paths, and point `command` at the **venv's** Python so dependencies resolve. Restart Claude Desktop, then ask *"what did I spend this week?"*.

### Scheduling the weekly digest

```bash
crontab -e
# Every Monday at 9am
0 9 * * MON cd /path/to/llm-cost-watchdog && venv/bin/python scripts/weekly_digest.py
```

Each run appends to `data/reports/digest_runs.log` and writes `data/reports/{date}_digest.json`.

---

## 5. Using it in your other projects

The whole point of the wrapper is that it's portable:

```python
from src.utils import call_llm

# Provider is inferred from the model ID — no provider argument anywhere.
summary = call_llm(prompt, model="claude-opus-5",  project_tag="job-search-agent")
draft   = call_llm(prompt, model="gpt-5.6-luna",   project_tag="teaching-workshop")
quick   = call_llm(prompt, model="gemini-flash-latest", project_tag="job-search-agent")
```

Tag each project distinctly and the cost breakdown tells you which of your projects is actually expensive.

---

## 6. Guardrails — the part that stops spend

Everything else in this project is read-only history. Guardrails are the one
piece that intervenes, because knowing what a runaway agent loop cost you is
much worse than having it stopped at call 40.

Three independent protections, set via `.env`:

| Guard | Catches | Setting |
|---|---|---|
| **Weekly budget** | steady overspend | `WEEKLY_BUDGET_USD` |
| **Per-project cap** | one project eating the whole budget | `WATCHDOG_PROJECT_CAPS=proj:2.00` |
| **Circuit breaker** | a loop with no exit condition | `WATCHDOG_MAX_CALLS_PER_MIN` |
| **Per-call ceiling** | one implausibly huge prompt | `WATCHDOG_MAX_COST_PER_CALL` |

```bash
WATCHDOG_GUARD_MODE=warn    # off | warn | block
```

**Default is `warn`, deliberately.** A tracking wrapper that silently starts
refusing calls is worse than the problem it solves — you opt into `block`.
In `block` mode, `call_llm()` raises `BudgetExceededError` *before* the
request goes out.

The circuit breaker exists for one specific scenario: an agent loop with no
exit condition, discovered the next morning. **Call rate is the early signal**
— on cheap models, cost lags far behind volume, so a budget check alone would
not catch it until thousands of calls in.

### Provider fallback

With credentials for more than one provider, a 429 shouldn't fail the call —
it should fail *over*:

```bash
WATCHDOG_FALLBACK=on        # default
```

When the requested provider is rate-limited, `call_llm()` retries it with
backoff, then routes to a cheap model on a *different* configured provider.
Both the failure and the eventual success are logged, so the fallback shows up
in your cost history rather than hiding. Only rate limits trigger it — a 400
fails identically everywhere, so retrying elsewhere just burns another call.

---

## 7. Finding waste

`find_waste` answers the question a cost report can't: **which of this spend
was avoidable?** Four checks, all computed from data already in SQLite — no
API calls, free to run:

| Check | Finds |
|---|---|
| **Retry waste** | Failed calls, the time they burned, and whether the failure rate is transient or structural |
| **Duplicate calls** | The same prompt sent repeatedly — a missing cache or a loop re-asking |
| **Cache opportunities** | Repeated large prompts never served from cache, priced at the model's real cached rate |
| **Over-powered calls** | A frontier model doing trivial-length work, with a costed alternative |

On this repo's own sample data it reports **24.5% of spend as recoverable**,
with the top action spelled out: *"Enable prompt caching for
cost-watchdog-self's claude-sonnet-5 calls (~$0.0936 this period)."*

**The total is an upper bound, not a sum.** The categories overlap — a
duplicated call is also a cache opportunity — so naively adding them can
exceed 100% of spend (this actually happened: 114%). It is capped at actual
spend and reported with per-category detail alongside.

---

## 8. MCP tools reference

| Tool | Arguments | Returns |
|---|---|---|
| `get_cost_report` | `period`: `today`\|`week`\|`month`\|`all_time`, `source?` | Total cost, calls, failures, cache savings, breakdowns by model / project / provider / source |
| `get_data_provenance` | `period` | How much of the recorded spend is real: cost and calls split by `live` / `manual` / `demo` |
| `check_budget_status` | none | `under`/`near`/`over`, percent used, remaining USD. Counts billed rows only |
| `get_burn_rate` | `period` | Daily burn, projected weekly total, budget-exhaustion date, confidence |
| `flag_anomalies` | `threshold_multiplier` (default `3.0`), `source?` | Flagged calls with reason and severity |
| `get_provider_breakdown` | `period`, `source?` | Per-provider cost, calls, tokens, cache hit rate, avg latency, models used, and `live_calls` |
| `compare_model_costs` | `input_tokens`, `output_tokens`, `models?` | What one call would cost on each model, cheapest first. No API calls made |
| `what_if_switched` | `from_model`, `to_model`, `period` | Re-prices your real traffic on another model |
| `find_waste` | `period`, `source?` | Retry waste, duplicate prompts, missed caching, over-powered models — each with a concrete action |
| `check_guard_status` | none | Guard mode, budget headroom, call-rate vs. the circuit breaker, per-project caps |
| `list_providers` | none | Which providers have credentials; which models are priced |
| `log_manual_entry` | `model`, `cost_usd`, `tokens`, `project_tag`, `note` | Confirmation; the row is recorded with `source=manual` |

`source` accepts `live`, `demo`, `manual`, any comma-separated combination (`live,manual`), or `all`. Omitting it includes everything — but the report still returns `breakdown_by_source`, so a total inflated by seeded data can never present itself as billed spend.

Example exchanges in Claude Desktop:

> **"Am I on track for the week?"** → `get_burn_rate()`
> ```json
> {"daily_burn_usd": 0.0637, "projected_weekly_usd": 0.446,
>  "budget_limit_usd": 5.0, "on_track": true,
>  "exhaustion_date": "2026-10-07", "confidence": "high"}
> ```

> **"Would Haiku have been cheaper than Sonnet?"** → `what_if_switched("claude-sonnet-5", "claude-haiku-4-5")`
> ```json
> {"calls_repriced": 9, "actual_cost_usd": 0.3221,
>  "hypothetical_cost_usd": 0.1074, "savings_usd": 0.2147,
>  "savings_percent": 66.67, "verdict": "cheaper"}
> ```

> **"Was anything weird this week?"** → `flag_anomalies()`
> ```json
> [{"reason": "cost $0.130840 is 295.3x the rolling avg for gpt-5.6-luna; latency 14800ms is 11.6x the rolling avg",
>   "severity": "high"}]
> ```

> **"Is any of this real?"** → `get_data_provenance()`
> ```json
> {"total_cost_usd": 0.476797, "billed_cost_usd": 0.006593,
>  "demo_cost_usd": 0.470204, "demo_percent_of_total": 98.62,
>  "calls_by_source": {"demo": 44, "manual": 4, "live": 5}}
> ```

---

## 8a. Provenance — is this number real?

A cost tracker whose figures can't be traced back to a billed API call is worse
than no tracker: it reports confident fiction. This repo ships with
`sample_usage.json` so the dashboard isn't empty on first run, which means the
headline figure is mostly invented until you generate traffic of your own.

Rather than hide that, every row carries a `source`:

| Source | Meaning | Counts toward budget? |
|---|---|---|
| `live` | A real HTTP request through `call_llm()`. Latency measured, tokens from the provider's own usage block | Yes |
| `manual` | Hand-entered via `log_manual_entry` or the CLI. Real spend, but reported rather than measured | Yes |
| `demo` | Seeded from a JSON file. Never billed, cost nothing | **No** |

Three consequences worth stating:

1. **Guardrails count billed rows only.** Loading `sample_usage.json` must never
   be enough to exhaust your budget and start refusing real calls. `check_guards()`
   and `check_budget_status()` filter to `live,manual`.
2. **The dashboard says so on its face.** When demo rows are present it renders a
   banner reading *"98.62% of the spend shown is seeded demo data"*, and the
   **Data** filter switches to billed-only. Each row in the activity table wears
   its badge; each provider bar shows `live_calls`, so a provider whose adapter
   has never run for real is visibly labeled **NO LIVE CALLS**.
3. **Upgrading an old DB classifies rather than assumes.** `ALTER TABLE … DEFAULT
   'live'` would stamp every pre-existing row as real. The migration instead
   infers provenance from two fingerprints the writers left behind — measured
   calls carry microsecond timestamps and non-zero latency; seeded rows were
   authored at round minutes — and defaults to `demo` on ambiguity, because
   understating spend is visible while overstating it invents money.

Once you have real traffic, drop the samples:

```bash
python -m src.tracker --provenance      # what's real right now
python -m src.tracker --purge demo      # delete seeded rows, keep billed history
```

---

## 9. Why this runs locally, not deployed

This is a personal MCP server, and running it locally over stdio is the standard, correct way to run one — not a limitation. Claude Desktop launches the process directly; there is no network hop, no hosting bill, and no reason for your personal spend data to leave your machine. The SQLite database and generated reports are gitignored for the same reason. The Docker setup exists to make the *dashboard* trivially demoable, not because the system needs a server. Total hosting cost: $0, by design.

---

## 10. Tech stack

- **Provider SDKs** — `anthropic`, `openai`, `google-generativeai`. Each behind an adapter implementing one `Provider` protocol, so the rest of the codebase is vendor-agnostic.
- **SQLite** — structured, numeric, time-series data. A vector DB would be the wrong tool: there is nothing here to search semantically. Schema changes migrate **in place** rather than recreating, because a cost tracker that drops your history on upgrade has destroyed the only thing it exists to keep.
- **Pydantic** — one schema shared by tracker, analyzer, MCP server, and dashboard, so the shape can't drift between them.
- **MCP Python SDK** — the core deliverable; stdio transport for local Claude Desktop integration.
- **FastAPI + vanilla JS** — the dashboard. One HTML file, no React, no npm, no build step, on purpose.
- **Docker Compose** — one-command dashboard run with `data/` mounted so history survives restarts.
- **pytest** — the eval suite below.
- **No orchestration framework.** The agentic loop is a plain documented function. LangChain here would be machinery without a job.

---

## 11. Eval results

`pytest tests/ -q` → **220 passed**.

| Suite | Tests | Covers |
|---|---:|---|
| `test_pricing.py` | 82 | Rate table integrity, cache accounting, long-context surcharge, cache-write premium, model comparison |
| `test_analyzer.py` | 29 | Anomaly accuracy, budget boundaries, burn rate, provider breakdown, what-if, digest |
| `test_providers.py` | 21 | Model→provider routing, credential detection, per-vendor usage normalization |
| `test_dashboard_api.py` | 20 | All 8 endpoints, response shapes, invalid input, privacy cap |
| `test_call_llm.py` | 22 | Retry/backoff, **cross-provider fallback**, failure tracking, cost correctness |
| `test_tracker.py` | 10 | Persistence, **in-place migration from the v1 schema**, batch load |
| `test_guard.py` | 18 | Guard modes, budget/project/rate/per-call trips, fail-safe on bad config |
| `test_waste.py` | 18 | All four waste checks, plus the overlap cap that stops >100% claims |

**Anomaly detection** — against `sample_usage.json` (44 synthetic events, 6 models, 3 providers, 3 planted spikes):

| Metric | Result |
|---|---|
| Planted anomalies caught | **3 / 3** |
| False positives on the other 41 events | **0** |
| Cost spike detected at | 29.3× the model's rolling average |
| Latency spike detected at | 18.6× the model's rolling average |
| Long-context spike | caught as `high` (tripped cost **and** latency) |

Detection is also verified against controlled fixtures: a 5× cost spike and an 8× latency spike each caught in isolation; normal variance up to 1.6× ignored; a 4× spike flagged at `3.0` and correctly *not* at `5.0`; an expensive model never flagged merely for being pricier than a cheap one; and a burst of failed calls does not poison the baseline.

**Cost calculation** — asserted against hand-computed values for **every** model in the pricing table, plus: cached tokens always cheaper than uncached; a fully-cached prompt billed entirely at the cached rate; cached + written + uncached partitioning `input_tokens` exactly with no double-billing; the long-context surcharge applying above 272K and **not** at or below it, and **not** to other providers; cache writes surcharged only on models that bill for them.

**Migration** — a v1-schema database is migrated in place with all rows preserved, safe defaults on old rows, idempotent re-runs, and new-format writes accepted afterward. Verified on the real development database: 37/37 rows preserved.

**Privacy** — `prompt_preview` is asserted to never exceed 80 characters anywhere it is returned.

---

## 12. The agentic digest loop

`src/digest.py :: generate_digest()` runs weekly and:

1. Pulls the `CostReport` for the period
2. Pulls anomalies
3. Sends both to the LLM **through `call_llm()` itself** — so the digest-writing call is tracked in the same table it's reporting on
4. Saves report + anomalies + generated text to `data/reports/{date}_digest.json`

It is agentic in an honest, narrow sense: it makes a judgment call every week — what's normal, what's worth flagging, what to do about it — without a human framing the question. It is *not* a multi-step tool-using agent and doesn't pretend to be.

**Fallback behavior:** if the LLM call fails, the digest does **not** abort. It writes a deterministic plain-text summary and records `"llm_written": false`. A scheduled job that loses its whole run to a transient 429 is a broken job.

### Postmortem

*Pending.* Reserved for a real incident — what the watchdog caught on live traffic, what was actually happening, and what changed as a result. It will not be filled with a hypothetical.

---

## Known gaps

Stated plainly rather than hidden:

- **Pricing is a snapshot.** Rates were verified against provider pricing pages in August 2026 and are hardcoded. There is no automatic refresh; when a vendor changes prices, `src/pricing.py` needs an edit. `python -m src.pricing` prints the whole table for review.
- **Live-tested against Google only.** The Anthropic and OpenAI adapters are unit-tested against captured response shapes but have not been exercised against a live endpoint in this repo — no keys were configured. The Gemini path has made real, tracked calls. This is not just a note in a README: `get_provider_breakdown` reports `live_calls: 0` for both, and the dashboard labels their bars **NO LIVE CALLS**.
- **Most of the shipped data is seeded.** Out of the box the DB is `sample_usage.json` plus whatever you generate. At the time of writing that is 44 demo rows ($0.4702) against 5 live calls and 4 manual entries ($0.0066) — 98.62% demo. Run `python -m src.tracker --provenance` for the current split, and `--purge demo` to clear it.
- **The digest's LLM path is exercised via its fallback.** The free-tier quota was exhausted during development, so the deterministic summary is what's been observed end-to-end. Both paths are tested.
- **Burn rate extrapolates linearly** from the observed span. A burst in a short window projects a misleadingly high rate — which is why every projection carries a `confidence` field.
