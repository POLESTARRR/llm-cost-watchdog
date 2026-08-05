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
     infers provider from model ID · measures latency · retries 429s
      with jittered backoff · prices via pricing.py · logs EVERY call
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
                         src/analyzer.py
     reports · anomalies · budget · burn rate · provider breakdown · what-if
                                │
            ┌───────────────────┼───────────────────┐
            ▼                   ▼                   ▼
     src/digest.py       src/mcp_server.py    dashboard/app.py
   weekly agentic loop     9 tools, stdio      FastAPI + static HTML
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

## 6. MCP tools reference

| Tool | Arguments | Returns |
|---|---|---|
| `get_cost_report` | `period`: `today`\|`week`\|`month`\|`all_time` | Total cost, calls, failures, cache savings, breakdowns by model / project / provider |
| `check_budget_status` | none | `under`/`near`/`over`, percent used, remaining USD |
| `get_burn_rate` | `period` | Daily burn, projected weekly total, budget-exhaustion date, confidence |
| `flag_anomalies` | `threshold_multiplier` (default `3.0`) | Flagged calls with reason and severity |
| `get_provider_breakdown` | `period` | Per-provider cost, calls, tokens, cache hit rate, avg latency, models used |
| `compare_model_costs` | `input_tokens`, `output_tokens`, `models?` | What one call would cost on each model, cheapest first. No API calls made |
| `what_if_switched` | `from_model`, `to_model`, `period` | Re-prices your real traffic on another model |
| `list_providers` | none | Which providers have credentials; which models are priced |
| `log_manual_entry` | `model`, `cost_usd`, `tokens`, `project_tag`, `note` | `"✓ logged"` |

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

---

## 7. Why this runs locally, not deployed

This is a personal MCP server, and running it locally over stdio is the standard, correct way to run one — not a limitation. Claude Desktop launches the process directly; there is no network hop, no hosting bill, and no reason for your personal spend data to leave your machine. The SQLite database and generated reports are gitignored for the same reason. The Docker setup exists to make the *dashboard* trivially demoable, not because the system needs a server. Total hosting cost: $0, by design.

---

## 8. Tech stack

- **Provider SDKs** — `anthropic`, `openai`, `google-generativeai`. Each behind an adapter implementing one `Provider` protocol, so the rest of the codebase is vendor-agnostic.
- **SQLite** — structured, numeric, time-series data. A vector DB would be the wrong tool: there is nothing here to search semantically. Schema changes migrate **in place** rather than recreating, because a cost tracker that drops your history on upgrade has destroyed the only thing it exists to keep.
- **Pydantic** — one schema shared by tracker, analyzer, MCP server, and dashboard, so the shape can't drift between them.
- **MCP Python SDK** — the core deliverable; stdio transport for local Claude Desktop integration.
- **FastAPI + vanilla JS** — the dashboard. One HTML file, no React, no npm, no build step, on purpose.
- **Docker Compose** — one-command dashboard run with `data/` mounted so history survives restarts.
- **pytest** — the eval suite below.
- **No orchestration framework.** The agentic loop is a plain documented function. LangChain here would be machinery without a job.

---

## 9. Eval results

`pytest tests/ -q` → **179 passed**.

| Suite | Tests | Covers |
|---|---:|---|
| `test_pricing.py` | 82 | Rate table integrity, cache accounting, long-context surcharge, cache-write premium, model comparison |
| `test_analyzer.py` | 29 | Anomaly accuracy, budget boundaries, burn rate, provider breakdown, what-if, digest |
| `test_providers.py` | 21 | Model→provider routing, credential detection, per-vendor usage normalization |
| `test_dashboard_api.py` | 20 | All 8 endpoints, response shapes, invalid input, privacy cap |
| `test_call_llm.py` | 17 | Retry/backoff, failure tracking, cost correctness, rate-limit detection |
| `test_tracker.py` | 10 | Persistence, **in-place migration from the v1 schema**, batch load |

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

## 10. The agentic digest loop

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
- **Live-tested against Google only.** The Anthropic and OpenAI adapters are unit-tested against captured response shapes but have not been exercised against a live endpoint in this repo — no keys were configured. The Gemini path has made real, tracked calls.
- **The digest's LLM path is exercised via its fallback.** The free-tier quota was exhausted during development, so the deterministic summary is what's been observed end-to-end. Both paths are tested.
- **Burn rate extrapolates linearly** from the observed span. A burst in a short window projects a misleadingly high rate — which is why every projection carries a `confidence` field.
