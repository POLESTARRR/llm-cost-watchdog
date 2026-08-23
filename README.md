# LLM Cost Gateway

A self-hosted **OpenAI-compatible gateway** that routes each request to the cheapest model that can actually handle it, enforces budgets before money is spent, and records every call in a ledger you own, across **Anthropic, OpenAI, Google, and locally-hosted models**.

Adoption is one environment variable:

```bash
export OPENAI_BASE_URL=http://localhost:8000/v1
export OPENAI_API_KEY=wd-my-project        # the suffix names the project in the ledger
```

Any app built on the OpenAI SDK is now tracked, routed, budget-enforced and failed-over **without one line of its source changing**. Ask it for `group:ladder` instead of a model name and it classifies your prompt and picks the tier: mechanical work goes to a free local model, architectural work goes to the frontier one.

It is also an **MCP server**, so you can ask Claude Desktop *"what did I spend this week?"*, and a browser dashboard.

### Why the gateway exists

The earlier version of this project was a Python wrapper you imported. It had a ledger, a router, guardrails, waste detection and 360 passing tests, and it had recorded exactly **zero** live calls, including from its author's own thirteen other projects. Adopting it meant rewriting every LLM call you had; that price was higher than the benefit, every time, for everyone.

The one component that did collect real data was the importer, precisely because it required no integration at all: it read files that already existed. The gateway is that lesson applied to the live path.

### What changed, and what didn't

The gateway is **purely additive**. Every capability the watchdog had still
works, through the same code paths, and the wrapper it was built around is still
a supported entry point, it is now the gateway's engine rather than its only
front door.

| | Watchdog (before) | Gateway (now) |
|---|---|---|
| **Adoption cost** | rewrite every LLM call to `call_llm()` | one env var, or import the wrapper as before |
| **Entry points** | Python import | Python import **+** `/v1/chat/completions` **+** MCP **+** dashboard |
| **Providers** | Anthropic · OpenAI · Google | + **Ollama** (local, $0.00/token) |
| **Routing strategies** | cheapest · lowest-latency · lowest-failure · shuffle | + **complexity** (reads the prompt, not the ledger) |
| **Streaming** | — | real SSE, all 4 providers, **measured TTFT** |
| **Tool calling** | — | Anthropic · OpenAI · Ollama, full agent loop |
| **Quality evidence** | none (`simulate_routing` priced switches, never judged them) | **shadow comparison + grading** |
| **Cost attribution** | per project, set in code | per project, **from the API key** |
| **MCP tools** | 16 | **21** |
| **HTTP endpoints** | 15 | **20** |
| **Test files / tests** | 13 / 360 | **19 / 462** |
| **Live rows in the ledger** | **0** | real traffic, with real failures |

Everything below carries over untouched: the [three-rate pricing model](#1-the-problem)
(cache reads, cache writes, the 1-hour TTL split, OpenAI's long-context
surcharge), [anomaly detection](#3-the-anomaly-detection-formula),
[guardrails](#6-guardrails-the-part-that-stops-spend),
[waste finding](#7-finding-waste), [provenance](#8a-provenance--is-this-number-real),
subscription-vs-billed accounting, pricing-drift checks, the weekly agentic
digest, Turso persistence, and the Claude Code transcript importer.

Three things were **changed** rather than added, each because the live path
exposed a defect the ledger alone could not:

- **`WATCHDOG_ROUTING_STRATEGY` now resolves at call time.** It was read at
  import time while `model_groups()` was read at call time, so a running gateway
  routed `cheapest` while its own `/router` endpoint reported `complexity`.
- **Fallbacks are recorded.** A 429 that failed over within a group returned a
  model the decision record never named, which reads as a routing bug and is in
  fact failover working. `fell_back_from` closes that.
- **`UsageEvent` gained `ttft_ms`.** Added alongside `latency_ms`, never
  replacing it, so streamed and non-streamed calls stay comparable.

---

## 1. The problem

Every LLM API call has a real cost and a real latency, and almost nobody tracks either until the bill arrives. Portfolio projects show what an agent *can do*. They rarely show what it *costs to run*.

The ones that do track cost usually track it **for one vendor**, and price it **wrong**, because real LLM billing is not `tokens × rate`:

- **A prompt is not billed uniformly.** Cache reads bill at ~10% of the input rate; cache *writes* bill at a **premium** on some models. A tracker that prices every input token at the full rate can overstate a cache-heavy workload several times over, or miss that its first call is the expensive one.
- **Anthropic's `usage.input_tokens` is the uncached remainder**: not the whole prompt. Reading it directly silently under-counts every cached call.
- **OpenAI's GPT-5.6 family surcharges long context.** Past 272K input tokens the *entire* request bills at 2× input / 1.5× output, not just the excess.

This project models all of that. One wrapper function, `call_llm()`, records cost, tokens, latency, and cache usage for every call, successful or failed, on any provider, into SQLite. Everything else reads from that one table.

---

## 2. Architecture

```
  ANY OpenAI-SDK app          projects that import the wrapper       digest
  (OPENAI_BASE_URL=…)                      │                            │
            │                              │                            │
            ▼                              │                            │
   src/gateway.py  /v1/chat/completions     │                            │
   OpenAI wire format · api key → project   │                            │
   SSE streaming · tool calls · real TTFT  │                            │
            └──────────────┬───────────────┴────────────────────────────┘
                           ▼
                 src/utils.py :: call_llm()
   guardrails (can BLOCK) · infers provider from model ID · retries 429s with
   backoff · falls over (and RECORDS the substitution) · prices it · logs ALL
                           │
            ┌──────────────┴───── routing (src/router.py) ─────┐
            │   cheapest · lowest-latency · lowest-failure ·   │
            │   shuffle · complexity ◄── src/complexity.py     │
            │                            reads the PROMPT,     │
            │                            not the ledger        │
            └──────────────┬──────────────────────────────────┘
                           ▼
   providers/  gemini · anthropic · openai · ollama (local, $0.00)
                           │
                normalized LLMResponse
                           │
                           ▼
            src/tracker.py ──►  data/usage.db  (SQLite)
                           │     one row per call
              ┌────────────┼────────────┬──────────────┐
              ▼            ▼            ▼              ▼
        analyzer.py    guard.py     waste.py      shadow.py
      anomalies      budgets      avoidable     was the cheap
      burn rate      caps         spend         model good enough?
                                                      │
                                                  judge.py
                                              deterministic first,
                                              local judge as triage
                           │
            ┌──────────────┼──────────────┐
            ▼              ▼              ▼
     src/digest.py   src/mcp_server.py  dashboard/app.py
   weekly loop        20 tools, stdio    FastAPI + gateway + HTML
                      → Claude Desktop   → localhost:8000
```

Three design decisions carry the project:

**Nothing is tracked that isn't intercepted.** `call_llm()` logs before it returns, so there is no code path where a call happens and isn't tracked, including calls that fail, which is exactly when you most want the record.

**The gateway is the product; the wrapper is its engine.** Both entry points converge on the same function, so routing, guardrails, pricing and the ledger are identical whichever you use. The gateway exists because the wrapper's adoption cost, rewriting your calls, was the thing actually preventing the ledger from ever seeing real traffic.

**The provider adapter is the seam.** Each vendor reports usage in a different shape; the adapters normalize to one `LLMResponse`. Tracker, analyzer, digest, dashboard, and MCP never learn which vendor a call came from. Adding a provider is one adapter + one pricing entry, which is exactly what adding local models turned out to be.

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
- The **first call for a model is never flagged**. There is no baseline yet.
- Severity is `high` when cost *and* latency both trip, otherwise `medium`.

Implementation: [`src/analyzer.py`](src/analyzer.py) → `flag_anomalies()`.

---

## 4. How to run locally

```bash
git clone <your-repo-url> llm-cost-gateway
cd llm-cost-gateway

python3 -m venv venv
source venv/bin/activate            # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env                # add whichever provider keys you use
```

> **If you move or rename the project directory, rebuild the venv**
> (`rm -rf venv && python3 -m venv venv && pip install -r requirements.txt`).
> A virtualenv hardcodes absolute paths in its console scripts, so after a move
> `venv/bin/uvicorn` fails with a confusing "no such file" naming the *old*
> path while `venv/bin/python` still works. This bit this repo twice.

You only need keys for providers you actually call. A missing key disables that provider; it doesn't break the tracker.

The fastest way to see it work costs nothing and needs no API key at all:

```bash
ollama serve &                      # https://ollama.com
ollama pull llama3.2:3b
python -m uvicorn dashboard.app:app --port 8000 &
python scripts/gateway_demo.py      # real calls, real ledger rows, $0.00
```

If you would rather look at populated dashboards before making any calls, there
is a seeded sample project:

```bash
python -m src.tracker --batch-load demo_job_search_agent.json
```

(`sample_usage.json` is a *separate* file, it's the pytest fixture with
planted anomalies, not demo content; don't load it into a DB you're using
day to day.)

Those rows land tagged `source=demo`. They are never counted against your
budget, they can't trip a guardrail, and the dashboard states on its face how
much of the displayed total they account for, see [§8a Provenance](#8a-provenance--is-this-number-real).
`python -m src.tracker --purge demo` removes them and leaves billed history
untouched. **The shipped `data/usage.db` contains no demo rows**; it is real
imported Claude Code usage plus whatever live traffic you generate.

Then any of:

```bash
# CLI: log a call made outside the wrapper (provider inferred from model ID)
python -m src.tracker --log-manual --model claude-opus-5 --cost 0.002 --project teaching-workshop

# Weekly digest (also the cron entrypoint)
python scripts/weekly_digest.py

# Dashboard AND the OpenAI-compatible gateway, same process, http://localhost:8000
python -m uvicorn dashboard.app:app --reload --port 8000

# Prove the live path end to end with real calls (see §13)
python scripts/proof.py             # six proofs, via the official OpenAI SDK
python scripts/gateway_demo.py      # the same path over raw HTTP

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

# Provider is inferred from the model ID; no provider argument anywhere.
summary = call_llm(prompt, model="claude-opus-5",  project_tag="job-search-agent")
draft   = call_llm(prompt, model="gpt-5.6-luna",   project_tag="teaching-workshop")
quick   = call_llm(prompt, model="gemini-flash-latest", project_tag="job-search-agent")
```

Tag each project distinctly and the cost breakdown tells you which of your projects is actually expensive.

---

## 6. Guardrails: the part that stops spend

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
refusing calls is worse than the problem it solves. You opt into `block`.
In `block` mode, `call_llm()` raises `BudgetExceededError` *before* the
request goes out.

The circuit breaker exists for one specific scenario: an agent loop with no
exit condition, discovered the next morning. **Call rate is the early signal**
on cheap models, cost lags far behind volume, so a budget check alone would
not catch it until thousands of calls in.

### Provider fallback

With credentials for more than one provider, a 429 shouldn't fail the call
it should fail *over*:

```bash
WATCHDOG_FALLBACK=on        # default
```

When the requested provider is rate-limited, `call_llm()` retries it with
backoff, then routes to a cheap model on a *different* configured provider.
Both the failure and the eventual success are logged, so the fallback shows up
in your cost history rather than hiding. Only rate limits trigger it; a 400
fails identically everywhere, so retrying elsewhere just burns another call.

### Routing, choosing the model from recorded history

Failover reacts to a failure. Routing decides *before* the call, and this one
decides from the ledger.

```bash
WATCHDOG_GROUP_FAST=gemini-flash-lite-latest,claude-haiku-4-5,gpt-5-nano
WATCHDOG_GROUP_SMART=claude-opus-5,gpt-5.6-sol
WATCHDOG_ROUTING_STRATEGY=cheapest    # cheapest | lowest-latency | lowest-failure | shuffle
```

```python
call_llm(prompt, model_group="fast", project_tag="my-app")
```

Four strategies. `cheapest` prices the call on each member. The other three,
`lowest-latency`, `lowest-failure`, `shuffle`, read `src/router.py`'s
`model_stats()`, computed from **calls that actually happened**: measured
latency, measured failure rate, measured cost per call.

That is the whole argument for this existing at all. A generic router picks the
cheapest deployment from a published price list; every router does that.
Picking the model that *actually served your traffic fastest last week* needs a
ledger, and this project already had one.

Three details that matter more than the strategy list:

- **Hard constraints filter before the strategy ranks.** Provider not
  configured, model cooling down, prompt larger than the model's context
  window, all removed first. A preference is never a reason to dispatch a call
  that cannot succeed. Context windows come from the public price map (this
  project's own table prices calls, it doesn't size them); a model with no
  known window is left alone rather than guessed at.
- **Unmeasured is not the same as good.** A model needs `MIN_CALLS_FOR_HISTORY`
  recorded calls before its history is trusted, and a history strategy with
  nothing to go on falls back to price and says so. One lucky call must not
  decide routing for everything after it. Failed calls are excluded from
  measured latency for the same reason the anomaly detector excludes them, a
  429 logs near-zero latency, and counting it would make a failing model look
  like the fastest one.
- **Cooldowns persist.** A 429 benches that model in SQLite, not in memory: a
  rate limit a restart forgets is a rate limit you hit again immediately.

Every decision records *why*, chosen model, strategy, candidates considered,
what was excluded and for what reason, so a routing choice can be audited
against the same history that informed it.

**`simulate_routing` is the part nothing else can do.** Replay real recorded
traffic through a policy and re-price it, before adopting it:

```
simulate_routing(group="fast", strategy="cheapest", period="week")
→ {"calls_repriced": 412, "actual_cost_usd": 6.41,
   "simulated_cost_usd": 1.83, "savings_percent": 71.4,
   "routed_to": {"gpt-5-nano": 412}, "verdict": "cheaper"}
```

It re-prices each call's *real* token counts on whichever member the strategy
would have picked. It prices a switch; it does not judge one, the assumption
that the cheaper model's output would have been good enough is exactly the
judgment a human should be making, and the tool says so in its own `caveat`
field rather than burying it here.

**What this deliberately is not.** Single process, single user. No Redis, no
proxy server, no per-deployment TPM/RPM accounting, no concurrency control.
Those matter for a shared gateway serving many callers; LiteLLM's Router
already does them well, and reimplementing them badly here would trade the one
thing this project has, the ledger, for a worse copy of something that
exists.

---

## 6a. The gateway: adoption without a code change

`src/gateway.py` mounts an OpenAI-compatible endpoint on the same FastAPI app as
the dashboard, so one process gives you the proxy and the UI that reads what the
proxy recorded.

```bash
export OPENAI_BASE_URL=http://localhost:8000/v1
export OPENAI_API_KEY=wd-checkout-service
```

Ask for a model by name and it is called directly. Ask for `group:<name>` and it
is routed. The API key's suffix becomes the ledger's `project_tag`, which is
what makes the existing per-project caps in `guard.py` apply to gateway traffic
without the caller passing anything extra.

Every response carries a namespaced extension a strict client will ignore:

```json
"x_watchdog": {
  "cost_usd": 0.000456, "latency_ms": 5411.0, "project": "checkout-service",
  "routing": {"complexity": {"tier": "complex", "score": 4}, "basis": "..."}
}
```

**Streaming is real streaming.** `stream: true` returns server-sent events in
OpenAI's chunk format, and the wrapper measures time to first token at the first
byte of actual content, recording it *alongside* total latency rather than
replacing it. Verified against a live local model: **4811ms TTFT against 6074ms
total**, over 22 separate chunks. The shortcut this refuses is emitting one
buffered chunk and calling the whole duration a TTFT, which would put a number
in the ledger that measures nothing.

Streams do not fail over. Once bytes have reached the client, re-running the
prompt elsewhere would splice two different answers into one response body, so a
mid-stream failure arrives as a terminal `error` event.

**Tool calling works, and so does the loop.** Requests offering `tools`, or
replaying a history that already contains tool results, take a structured path
that passes real messages through instead of flattening them, because a `tool`
role has no textual equivalent a provider will interpret correctly. Supported on
Anthropic, OpenAI and Ollama; not Gemini here.

```
TURN 1: tool_calls | get_weather {"city": "Paris"}     $0.00  173 tok
TURN 2: stop       | "The current weather in Paris is 18 degrees Celsius…"  $0.00  115 tok
```

That two-turn loop is real output against `llama3.2:3b`, and getting it working
surfaced a bug worth naming: **Ollama returns tool-call `arguments` as a decoded
object and also requires an object inbound, while the OpenAI wire format
specifies a JSON string in both directions.** Normalizing only the response made
turn one work and turn two fail with *"Value looks like object, but can't find
closing '}' symbol"*. Every agent loop would have broken on its second call.

**Still refused: streaming *plus* tools**, with the reason. Reassembling a tool
call from partial argument deltas is real work this hasn't done, and guessing at
half-parsed JSON would be worse than saying no.

A budget block surfaces as **429**, not 402: it is a self-imposed quota that
will clear, and every OpenAI client already knows how to back off on a 429.

## 6b. Local models: the zero-cost tier

`ollama/<model>` routes to a local Ollama server. It is the first provider here
whose "credential" is a running process rather than an API key, so
`is_configured()` probes the server instead of reading an env var.

Local models are zero-rated **by prefix, not by table entry**. Every other model
has to be enumerated because its price is a fact about a vendor's price list; a
local model's price is a fact about where it runs. Without that rule an unlisted
local model would hit `_FALLBACK_RATES` and be reported as spend that never
happened, which is exactly what the provenance system exists to prevent.

Adding a $0.00 model surfaces something worth stating plainly: **it makes
`cheapest` degenerate.** Local wins every comparison, for every prompt, forever.
That is not a bug to patch out. It is the clearest demonstration available that
price alone was never a routing policy, which is what the next section is for.

The cost that *is* real here is latency. Measured on an M-series laptop with
`llama3.2:3b`: ~5s cold model load, then ~25 tokens/sec. The ledger records that
like any other call, so "free" is never reported without the number qualifying it.

## 6c. Complexity routing: the strategy that reads the prompt

Every other strategy ranks models by something measured *about the model*. This
one looks at what is being asked:

```
"reformat this JSON"                    -> trivial   -> free local model
"Can you look at the user service?"     -> moderate  -> mid-tier
"Design a zero-downtime migration…"     -> complex   -> frontier
```

Three commitments shape it:

- **It is heuristic, not a model call.** A classifier that costs an API call to
  decide which API to call is a tax on every request, and on cheap prompts it
  costs more than it saves. This runs in microseconds and spends nothing.
- **It reports its reasoning.** Every rule that fired is listed with what it
  contributed, so a route to a 3B model is auditable after the fact. `GET
  /complexity?prompt=…` classifies anything without calling a model.
- **It is deliberately biased upward.** Misrouting a hard prompt to a weak model
  costs you an hour; misrouting an easy one to a strong model costs a fraction
  of a cent. Ambiguous prompts escalate, and the short-prompt discount is
  suppressed whenever any complexity signal fired, because *"design a
  zero-downtime migration strategy"* is fifty characters and is not easy work.

Tiers index into the group's **price ladder**, so no extra configuration is
needed: a model group is already a set you declared interchangeable, which is
the only context where price is a defensible proxy for capability.

`simulate_routing` **cannot** model this strategy, and says so in its own
`caveat` rather than returning a number you would have to read the source to
distrust: the ledger stores an 80-char preview and a hash, never prompt text.
Measuring complexity routing requires live traffic, which is what §6d is for.

## 6d. Shadow comparison: was the cheap model good enough?

`simulate_routing` already admitted the limit: *"it prices the switch, it does
not judge it."* Re-pricing tokens tells you what you would have paid, never
whether you would have accepted the answer, and a saving you had to redo by hand
is not a saving.

`src/shadow.py` closes that the only way it honestly can: run the cheap model on
**the same real prompt** and keep both answers.

```bash
WATCHDOG_SHADOW_RATE=0.05     # shadow one call in twenty
```

- **The user never waits.** The shadow runs after the real response is already
  returned, and every exception is swallowed. A quality experiment must never be
  able to fail a request that already succeeded.
- **It is free.** The shadow target is local; the frontier call was happening
  anyway. A shadow against a second paid API doubles your bill to study your bill.
- **It stores prompts, and that is the one deliberate exception** to this
  project's never-log-full-prompts rule. Evidence you cannot re-read is not
  evidence. Shadow rows live in their own table, are opt-in, and
  `purge_shadows()` deletes them.

Collection and judgement are separate on purpose. `GET /shadow` reports
`unverified_savings_usd` alongside `scored` and `acceptance_rate`, and refuses to
call the first one a saving until the others cover it. Four hundred ungraded
comparisons look like evidence and are not; they are raw material.

### Grading (`src/judge.py`)

`POST /shadow/grade` turns pairs into a number, with two graders that fail
differently:

**Deterministic checks run first and are trusted absolutely.** If the expensive
answer's Python parses and the cheap one's does not, no further judgement is
wanted. They abstain rather than guess, which is the common case, and they
abstain when *both* answers are broken, because that is not a difference between
the models.

**A local LLM judge handles the rest, and is labelled as the weak evidence it
is.** The obvious objection, that a 3B model is grading a 3B model, is real and
shapes the design: the judge is asked a narrow comparative question rather than
for a quality score, it never learns which answer came from which model, the two
answers are presented in **randomised order** because position bias is the
best-documented LLM-judge failure mode, and an unparseable verdict counts as
*inadequate* rather than quietly inflating the acceptance rate.

Verdicts record `scored_by`, so `deterministic` and `local-judge` grades can
always be separated. `WATCHDOG_JUDGE_MODEL` points the same pipeline at a
stronger model if you have budget for one.

First real numbers, from the five-call demo with shadowing on:

| tier | comparisons | real cost | real latency | local latency | acceptance |
|---|---|---|---|---|---|
| moderate | 1 | $0.000026 | 4973ms | 4159ms | 0% |
| complex | 1 | $0.000551 | 5974ms | **34219ms** | 0% |

Two data points prove nothing about acceptance, and the table is here mostly to
show what the pipeline produces rather than to claim a result. The latency
column is already the more interesting one: on the complex prompt the free local
model took **5.7x longer** than the paid one it was replacing. That is the real
cost of "free", and it is exactly the number a cost-only analysis never shows.

One guard the first real run forced: **a shadow against the same model is
skipped.** The primary call had already failed over to the local model, so the
shadow re-ran that same model and produced two rows of pure noise that would
have landed in the acceptance rate as if they were evidence.

---

## 7. Finding waste

`find_waste` answers the question a cost report can't: **which of this spend
was avoidable?** Five checks, all computed from data already in SQLite, no
API calls, free to run:

| Check | Finds |
|---|---|
| **Model switch** | Real spend on a model with a cheaper same-vendor sibling, re-priced on YOUR actual traffic via `what_if_switched`, not gated by call length, so it catches substantial work too |
| **Retry waste** | Failed calls, the time they burned, and whether the failure rate is transient or structural |
| **Duplicate calls** | The same prompt sent repeatedly: a missing cache or a loop re-asking |
| **Cache opportunities** | Repeated large prompts never served from cache, priced at the model's real cached rate |
| **Over-powered calls** | A frontier model doing trivial-length work, with a costed alternative |

Against this repo's own real, imported Claude Code build-cost data (§8d) it
reports **~60% of spend as recoverable**, with the top action sourced from a
real re-pricing of real traffic: *"581 real claude-sonnet-5 call(s) cost
$54.83; the same traffic on claude-haiku-4-5 would have cost $18.28 (67%
less)."*, not a caching tip this time but a genuine model-tier finding.

**The total is an upper bound, not a sum.** The categories overlap, a
duplicated call is also a cache opportunity, so naively adding them can
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
| `find_waste` | `period`, `source?` | Retry waste, duplicate prompts, missed caching, over-powered models. Each with a concrete action |
| `check_guard_status` | none | Guard mode, budget headroom, call-rate vs. the circuit breaker, per-project caps |
| `list_providers` | none | Which providers have credentials; which models are priced |
| `log_manual_entry` | `model`, `cost_usd`, `tokens`, `project_tag`, `note` | Confirmation; the row is recorded with `source=manual` |
| `get_subscription_roi` | `period` | List-price value of flat-fee usage, the subscription cost for the span it covers, and the ratio. Never describes it as billed spend |
| `get_router_status` | none | Declared model groups, active strategy, models cooling down and for how long, group members missing from the pricing table |
| `simulate_routing` | `group`, `strategy`, `period`, `source?` | Replays real traffic through a routing policy and re-prices it. Prices a switch; does not judge it |
| `check_pricing_drift` | `refresh` | Every local rate that disagrees with the public price map. Reports drift, never rewrites a rate |

`source` accepts `live`, `demo`, `manual`, `subscription`, any comma-separated combination (`live,manual`), or `all`. Omitting it includes everything, but the report still returns `breakdown_by_source`, so a total inflated by seeded data can never present itself as billed spend.

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
> {"total_cost_usd": 130.635655, "billed_cost_usd": 130.434414,
>  "demo_cost_usd": 0.201241, "demo_percent_of_total": 0.15,
>  "calls_by_source": {"demo": 10, "live": 71, "manual": 929}}
> ```

---

## 8a. Provenance, is this number real?

A cost tracker whose figures can't be traced back to a billed API call is worse
than no tracker: it reports confident fiction. This repo ships with one
illustrative fake project (`demo_job_search_agent.json`, a job-search agent
no such project exists, it's shaped like one to be a believable demo) so the
dashboard isn't empty on first run. Everything else in the DB, `civil-prep`
and `cost-watchdog-self` below, is real traffic, not invented.

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
   infers provenance from two fingerprints the writers left behind, measured
   calls carry microsecond timestamps and non-zero latency; seeded rows were
   authored at round minutes, and defaults to `demo` on ambiguity, because
   understating spend is visible while overstating it invents money.

Once you have real traffic, drop the samples:

```bash
python -m src.tracker --provenance      # what's real right now
python -m src.tracker --purge demo      # delete seeded rows, keep billed history
```

---

## 8b. Real integration: tracking [civil-prep](https://github.com/POLESTARRR/civil-prep)

The rest of this README's examples are the shipped fake project. This section
isn't, it's what happened when this tracker was pointed at a second, unrelated
real project: **civil-prep**, a UPSC current-affairs RAG system with its own
retrieval store, its own eval harness, and its own `call_llm()` wrapper (Gemini
only, no relation to this project's multi-provider one).

civil-prep's own `src/utils.py::call_llm()` doesn't log anywhere. It just
returns text. To measure it without changing its behavior, its `call_llm` was
wrapped for one run to capture what the Gemini SDK *itself* reports on
`response.usage_metadata` (`prompt_token_count`, `candidates_token_count`)
not an estimate from `len(prompt) // 4`. That is the provider's own accounting, and
those captured events were logged into this tracker's DB with
`source="live"`. The questions were civil-prep's own committed
`eval_questions.json`; the source documents were its own committed
`data/raw/` articles. Nothing here is synthetic.

**Every one of civil-prep's 25 committed eval questions, full pipeline, all four GS papers:**

| Call type | Calls | Input tokens | Output tokens | Cost |
|---|--:|--:|--:|--:|
| Answer generation (retrieval + Gemini) | 22 | 13,477 | 1,199 | $0.001827 |
| Faithfulness judge | 22 | 11,771 | 66 | $0.001203 |
| Relevance judge | 22 | 11,664 | 66 | $0.001193 |
| **Total** | **66** | **36,912** | **1,331** | **$0.004224** |

Only 22 of 25 questions got an answer call; the other 3 fell below civil-prep's
own 0.50 retrieval-confidence threshold and **correctly refused** rather than
guessing (`ask.py`'s `CONFIDENCE_THRESHOLD` gate; see the code excerpt in
§1). Those 3 still went through the judge, because the faithfulness rubric explicitly
scores an honest "I don't have enough information" as 1.0, and this run's
judge agreed on all three, real evidence the refusal path is doing its job.

At ~$0.0002/question end-to-end (retrieval + answer + both judge calls), running
every real eval question this system has cost **under half a cent total**.
That's not a rounding trick, Gemini Flash-Lite genuinely is that cheap, and a
tracker that reported a bigger number here would be reporting something false.

**Something the waste detector caught that's worth stating honestly:** with a
smaller sample it flagged the answer-generation and judge calls as "duplicate
prompts." They aren't, they're 22 *different* questions and answers. The
`find_duplicate_calls` heuristic groups by an 80-character `prompt_preview`
(see `src/waste.py`'s `_MIN_PREVIEW_FOR_DUPLICATE`), and civil-prep's prompts
open with a long fixed instruction template before the part that actually
varies per call. Against a system with static, template-heavy prompts, an
80-char preview isn't enough to tell two calls apart, a real limitation this
run surfaced, not a bug it hid. The fix, hash the full prompt instead of previewing it, is now implemented:
every row carries a `prompt_hash` (SHA-256, never the text), duplicate
detection groups on it, and each finding reports `matched_on: "hash" | "preview"`
so a preview-matched row from before the column existed can be read with the
appropriate suspicion.

**A second real, useful finding from the full run:** every one of the 44 judge
calls (22 faithfulness + 22 relevance) scored a perfect 1.00. Across 22 varied
questions spanning four different GS papers, a judge that never disagrees is
itself worth being suspicious of, either civil-prep's retrieval is genuinely
that precise (plausible; it only answers when confidence clears 0.50, so the
sample is pre-filtered toward easy cases), or the judge model is too lenient
to catch a subtly wrong answer. `eval.py`'s own `F1_THRESHOLD = 0.70` CI gate
would not currently catch a regression, because nothing in this dataset scores
below 1.0. The honest fix is to run the judge against at least one
deliberately wrong answer and confirm it scores near 0.0. That check doesn't
exist yet in civil-prep's suite.

Combined with `cost-watchdog-self`. This project's own real digest, run for
real across every period it supports (`today`, `week`, `month`, `all_time`):
5 genuine calls, 2,173 tokens, $0.000399, on `gemini-flash-lite-latest` since
that's the only provider with a live key here (would route to `claude-opus-5`
if `ANTHROPIC_API_KEY` were set, see `DIGEST_MODEL` in `src/digest.py`), the
dashboard now carries three tracks side by side: one openly fake project for
demo purposes, and two real ones. `civil-prep`'s calls show `source=live` and
count toward its own budget; the fake `job-search-agent` data does not.

Worth being explicit about why `cost-watchdog-self` is the *smallest* real
number here, not the largest: the digest is one summarization call per
period. civil-prep runs a 3-call pipeline (retrieval + answer + two judges)
for each of 25 questions, 66 calls. A single-call feature will always cost
less than a 66-call pipeline; running the digest more times than it's
actually useful to run just to outrank civil-prep's number would be exactly
the kind of padding this section exists to refuse.

---

## 8c. Portfolio survey: does this hold up across every real project?

To check whether "small real numbers" is specific to civil-prep or a pattern,
every other public repo on the same GitHub account was pulled via the GitHub
API (`api.github.com/users/POLESTARRR/repos`) and checked for whether it
calls an LLM provider **at all**, by grepping for provider SDK imports and
API patterns, not by assumption:

| Repo | Language | Calls an LLM (Anthropic/OpenAI/Google)? | Real tokens tracked |
|---|---|---|---|
| [`civil-prep`](https://github.com/POLESTARRR/civil-prep) | Python | **Yes**, Gemini, via its own `call_llm()` | 36,912 in / 1,331 out, $0.004224 |
| `smart-clinic-project` | Java | **No.** `TokenService.java` is JWT auth (`io.jsonwebtoken`), unrelated to LLM tokens despite the filename. Grepped the whole repo for provider SDK patterns, zero matches. | $0, not applicable |
| `EMOTION-DETECTION` | Python | **No.** Calls IBM Watson's NLP emotion-classification endpoint (`sn-watson-emotion.labs.skills.network`) via plain `requests`, a hosted classifier, not a token-billed LLM. | $0, not applicable |
| `CODE-OF-CONDUCT` | Shell | **No.** A community-health template repo; its one script is a `bc`-based simple-interest calculator, unrelated to AI entirely. | $0, not applicable |

This is the honest answer to "why is civil-prep's number so small": it isn't
that the tracking is fake, it's that **most of a real portfolio doesn't call
an LLM at all**, and the one project that does uses a model priced at
$0.10/$0.40 per million tokens. Both facts are real. Neither should be hidden
to make a chart look busier.

**What this means practically, project by project:**

- **civil-prep**: already efficient (Flash-Lite, confidence-gated refusal,
  citations on every answer). The one real gap the run surfaced: the
  LLM-judge eval has no negative test (see §8b above), add one deliberately
  wrong answer to `eval_questions.json`'s test harness so a future regression
  in retrieval quality can actually be caught by CI, not just a drop that
  happens to also stay under a judge that's never seen a wrong answer.
- **smart-clinic-project**: a pure CRUD hospital-management app today.
  If an LLM feature were added later (e.g. drafting a visit summary from
  structured appointment data), this tracker's `call_llm()` wrapper would
  drop in with one import, same as civil-prep did. Today no LLM cost exists to
  optimize because none is spent.
- **EMOTION-DETECTION**: deliberately not migrated to an LLM here: IBM
  Watson's classifier is free at this scale and purpose-built for the task;
  swapping in an LLM would trade a $0 hosted classifier for a per-call cost
  to do the same job, which is the opposite of what this project argues for.
- **CODE-OF-CONDUCT**: a governance template; there is no scenario where
  this should call an LLM, and it doesn't.

That asymmetry. One project spending real, small, trackable money and three
spending exactly nothing, is what a real portfolio actually looks like. The
dashboard's `civil-prep` project tag reflects it without embellishment.

---

## 8d. What it actually cost to *build* these projects with Claude Code

Everything above is *runtime* cost, what civil-prep and this project's own
digest spend when they run. There's a second real number this project can
answer that nothing above captures: what did it cost, in real Anthropic
tokens, to **build** them in the first place? Claude Code writes a local
transcript for every session (`~/.claude/projects/<project>/<session-id>.jsonl`),
and every assistant turn in it carries the actual `usage` block from the real
API response Claude received, the same authoritative source every other
`source=live` row in this project is priced from. `scripts/import_claude_code_usage.py`
reads that transcript directly and logs each turn as `source="manual"` (real
spend, reconstructed from an existing record rather than measured live by
this project's own wrapper, Claude Code isn't calling itself through this
codebase, so `call_llm()` never sees these turns), with the turn's real
historical timestamp, deduped by the API response's own message id so
re-running on a transcript that's grown only imports what's new.

```bash
python scripts/import_claude_code_usage.py \
    --session ~/.claude/projects/-path-to-project/<uuid>.jsonl \
    --project-tag my-project-build
```

`scripts/import_all_projects.py` runs this across every tracked project in one
pass, the mapping from transcript folder to project tag lives in that file, so
adding a new project is a one-line edit rather than a remembered command.

| Project | Turns | Tokens | List-price value |
|---|--:|--:|--:|
| `p2-jsw` | 974 | 295,518,870 | $165.58 |
| `scalp-log` | 346 | 132,020,815 | $117.01 |
| `last-kilometre` | 488 | 136,024,281 | $109.36 |
| `ibs` | 275 | 97,090,581 | $81.18 |
| `brainstorm` | 251 | 86,761,551 | $70.28 |
| `gtm` | 402 | 84,190,432 | $64.14 |
| `umbra` | 275 | 57,635,295 | $48.65 |
| `civil-prep` | 417 | 89,582,158 | $48.29 |
| `llm-cost-watchdog` (this repo) | 155 | 34,741,474 | $30.71 |
| `prahar` | 193 | 31,599,518 | $22.28 |
| `saans` | 51 | 33,653,826 | $21.38 |
| `clip2cart` | 201 | 31,017,243 | $14.21 |
| `p1-ura` | 158 | 15,109,021 | $7.26 |
| **Total** | **4,186** | **1,124,945,065** | **$800.33** |

### That column says "value", not "cost", and the distinction is the point

These sessions ran on a **Claude Pro subscription**. The tokens are real, the
work is real, and the rate they are priced at is the real published API rate,
but **no per-token charge ever occurred**. Calling $800.33 "money spent" would
overstate actual spend by the entire table, which is precisely the "confident
fiction" [§8a](#8a-provenance--is-this-number-real) exists to prevent.

So these rows carry a fourth provenance, `subscription`, and it is deliberately
**excluded from `BILLED_SOURCES`**. `check_budget_status()` and the guardrails
ignore it, a flat fee cannot exhaust a metered budget, and a guardrail cannot
un-spend it. `get_data_provenance` reports `billed_cost_usd: 0.0` alongside
`list_price_cost_usd: 800.33`, and both figures are true.

The ratio is the interesting number, and it needs no exaggeration:

```bash
python -c "from src.analyzer import subscription_roi; print(subscription_roi())"
```

> **$800.33 of list-price API value, over an 11.3-day span, against $7.55 of
> subscription time, a 106× multiple.**

The denominator is prorated to the span the usage actually covers rather than a
full month, because charging a two-day burst a full month's fee would flatter
the ratio dishonestly. Tune it with `WATCHDOG_SUBSCRIPTION_USD_PER_MONTH`.

Against the ~$0.0046 these projects cost to *run*, build cost still dwarfs
runtime by roughly five orders of magnitude. Both numbers are real; they answer
different questions, and now the middle number, what was actually charged, is
reported separately from both.
Building an app with an agentic coding tool costs vastly more than running
the finished app, because building means many long turns re-sending a large,
growing context (this is also why 97% of the Anthropic spend above is
cache-read, not fresh input, Claude Code aggressively caches conversation
history, and the dashboard's own "saved by caching" figure now reflects that
directly). Neither figure should stand in for the other, and this project
tracks both rather than picking the one that looks better.

**Caveats, stated plainly:** these rows carry `latency_ms=0`, the transcript
doesn't record per-turn wall-clock time, and zero is the honest value for
"not measured," not a guess. The cache-write premium uses this project's
existing flat `CACHE_WRITE_MULTIPLIER = 1.25`, which is accurate for
Anthropic's 5-minute ephemeral cache but understates the real premium on
1-hour ephemeral writes (most of what these sessions used), meaning the true
Anthropic invoice is likely somewhat *higher* than $130.43, not lower. And the
`llm-cost-watchdog-build` session's transcript also contains this section's
own writing and the earlier civil-prep integration work (§8b/§8c). It is the
literal Claude Code history of this repo, not a hand-curated subset.

---

## 8e. Adding a project after the dashboard is already deployed

The workflow above assumes the importer runs next to the database. Once the
*dashboard* is deployed to a public URL (§9) that's no longer true, the
deployed copy has its own database, separate from whatever's on your laptop.
`POST /import` closes that gap: point `import_claude_code_usage.py` at the
live URL instead of a local path, and a deployed dashboard picks up a new
project's real build cost immediately, no redeploy.

```bash
python scripts/import_claude_code_usage.py \
    --session ~/.claude/projects/-path-to-new-project/<uuid>.jsonl \
    --project-tag new-project-build \
    --remote-url https://your-dashboard.example.com \
    --import-key "$WATCHDOG_IMPORT_KEY"
```

The endpoint is closed by default, `WATCHDOG_IMPORT_KEY` unset means every
request 403s, not merely-unauthenticated. Set it as a platform secret on the
deployment (never commit it), and give the same value to `--import-key`
locally. A wrong or missing key 401s. Cost is **always recomputed server-side**
from `PRICING_TABLE`, the request body carries tokens and model, never a
cost figure, so a leaked key can misreport volume but can't forge a dollar
amount. Same message-id checkpoint and same-vendor dedup as the local path,
so a partial network failure just means the next run picks up where it left
off, never double-logs.

---

## 8f. Persistence on a free host: SQLite locally, Turso when deployed

Render's free tier has no persistent disk. Every restart wipes anything
written to the local filesystem, which would silently erase all tracked
history. `src/turso_backend.py` swaps the storage backend to
[Turso](https://turso.tech) (a hosted, SQLite-compatible database with a free
tier) whenever `TURSO_DATABASE_URL` is set, with zero changes anywhere else in
the codebase, `tracker.py`'s `_connect()` is the only integration point.

This needed a real wrapper, not a drop-in swap: libsql's Python client returns
plain tuples instead of `sqlite3.Row` (no `row["column"]` access, used
everywhere in this codebase), has no `.row_factory` to opt into that, and its
cursor isn't directly iterable the way `sqlite3.Cursor` is, three gaps
`_TursoRow`/`_Cursor` close, each confirmed against a **real** Turso database
rather than assumed from docs (the docs I could reach gave inconsistent
package names and incomplete examples; guessing here would have meant
silently wrong query results, not just an ImportError).

**Verified for real, escalating to the exact production path:**
1. Bare `libsql.connect()` + raw SQL against a live Turso database (caught the
   tuple-vs-Row and iteration gaps)
2. The actual `tracker.py`/`analyzer.py` functions (`init_db`, `log_usage`,
   `get_events_for_period`, `compute_report`) imported and run for real against
   that database
3. The literal production container, built from this repo's own Dockerfile,
   `linux/amd64` (Render's real architecture, not this dev machine's arm64),
   running the real `uvicorn` `CMD`, with `curl` against `/health`,
   `/provenance`, `/import`, and `/report` for real

`libsql` is deliberately **not** in `requirements.txt`. Its only prebuilt
wheels are macOS and `manylinux x86_64`; there's no `linux/arm64` wheel and no
macOS wheel for this repo's own dev interpreter (Python 3.14), so it fails to
build from source on this exact machine (confirmed: no Rust toolchain, no
`cc` linker). It's installed as a separate `RUN pip install` line in the
Dockerfile instead, which only ever builds for `linux/amd64`, where a real
wheel exists and the install is fast. Local dev and every test in this repo
run on plain `sqlite3` unconditionally; `TURSO_DATABASE_URL` is read only
inside `_connect()`, never at import time, so nothing locally needs `libsql`
installed at all.

**Where the credentials live, and why not `.env`:** Turso's URL and auth token
are in `.env.render`, not `.env`. `load_dotenv()` loads `.env` into every local
process including pytest, putting a remote-database switch in it means local
dev and the test suite start silently hitting the real production database
the instant it's set. This broke 155 tests immediately when first tried, which
is exactly the failure mode `.env.render` (never loaded automatically, purely
a reference for pasting into Render's environment-variable UI) exists to
prevent. Both files are gitignored; the actual secrets are never committed.

---

## 9. Local by default, deployable when you want a URL to share

The MCP server always runs locally over stdio. That part is correct as-is, not
a limitation to fix. Claude Desktop launches the process directly; there is no
network hop, no hosting bill, and no reason your personal spend data needs to
leave your machine for that half of this project. The SQLite database and
generated reports stay gitignored for the same reason.

The *dashboard* is a separate concern: `dashboard/app.py` is a normal FastAPI
app, deployable anywhere that runs Python (Render, Railway, Fly.io, a VPS) with
persistent storage for `data/usage.db`. Set `WATCHDOG_IMPORT_KEY` there as a
platform secret to enable §8e's remote-import path, so it can be updated with
a new project's real numbers after every deploy, not just at deploy time. The
Docker setup makes a local run of it trivially demoable even without deploying
anywhere; deploying it is opt-in, not required.

---

## 10. Tech stack

- **Provider SDKs**, `anthropic`, `openai`, `google-generativeai`. Each behind an adapter implementing one `Provider` protocol, so the rest of the codebase is vendor-agnostic.
- **SQLite**: structured, numeric, time-series data. A vector DB would be the wrong tool: there is nothing here to search semantically. Schema changes migrate **in place** rather than recreating, because a cost tracker that drops your history on upgrade has destroyed the only thing it exists to keep.
- **Pydantic**. One schema shared by tracker, analyzer, MCP server, and dashboard, so the shape can't drift between them.
- **MCP Python SDK**: the core deliverable; stdio transport for local Claude Desktop integration.
- **FastAPI + vanilla JS**: the dashboard. One HTML file, no React, no npm, no build step, on purpose.
- **Docker Compose**: one-command dashboard run with `data/` mounted so history survives restarts.
- **pytest**: the eval suite below.
- **No orchestration framework.** The agentic loop is a plain documented function. LangChain here would be machinery without a job.

---

## 11. Eval results

`pytest tests/ -q` → **347 passed**.

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

**Anomaly detection**: against `sample_usage.json` (44 synthetic events, 6 models, 3 providers, 3 planted spikes):

| Metric | Result |
|---|---|
| Planted anomalies caught | **3 / 3** |
| False positives on the other 41 events | **0** |
| Cost spike detected at | 29.3× the model's rolling average |
| Latency spike detected at | 18.6× the model's rolling average |
| Long-context spike | caught as `high` (tripped cost **and** latency) |

Detection is also verified against controlled fixtures: a 5× cost spike and an 8× latency spike each caught in isolation; normal variance up to 1.6× ignored; a 4× spike flagged at `3.0` and correctly *not* at `5.0`; an expensive model never flagged merely for being pricier than a cheap one; and a burst of failed calls does not poison the baseline.

**Cost calculation**: asserted against hand-computed values for **every** model in the pricing table, plus: cached tokens always cheaper than uncached; a fully-cached prompt billed entirely at the cached rate; cached + written + uncached partitioning `input_tokens` exactly with no double-billing; the long-context surcharge applying above 272K and **not** at or below it, and **not** to other providers; cache writes surcharged only on models that bill for them.

**Migration**: a v1-schema database is migrated in place with all rows preserved, safe defaults on old rows, idempotent re-runs, and new-format writes accepted afterward. Verified on the real development database: 37/37 rows preserved.

**Privacy**, `prompt_preview` is asserted to never exceed 80 characters anywhere it is returned.

---

## 12. The agentic digest loop

`src/digest.py :: generate_digest()` runs weekly and:

1. Pulls the `CostReport` for the period
2. Pulls anomalies
3. Sends both to the LLM **through `call_llm()` itself**, so the digest-writing call is tracked in the same table it's reporting on
4. Saves report + anomalies + generated text to `data/reports/{date}_digest.json`

It is agentic in an honest, narrow sense: it makes a judgment call every week, what's normal, what's worth flagging, what to do about it, without a human framing the question. It is *not* a multi-step tool-using agent and doesn't pretend to be.

**Fallback behavior:** if the LLM call fails, the digest does **not** abort. It writes a deterministic plain-text summary and records `"llm_written": false`. A scheduled job that loses its whole run to a transient 429 is a broken job.

### Postmortem: the tracker was under-reporting its own cost by 17%

Not a hypothetical, and not a flattering one, the bug was in this project's
pricing engine, and it made every number this repo has ever published too
small.

**Symptom.** A review of `src/pricing.py` against a real Claude Code transcript
showed cache-write tokens being priced at the plain input rate.

**Root cause, two bugs stacked.**

1. `CACHE_WRITE_BILLED_MODELS = ("gpt-5.6",)`. Anthropic models were never in
   that tuple, so `bills_cache_writes("claude-opus-5")` returned `False` and
   every Claude cache write billed at **1.0×** input, no premium at all. The
   test suite asserted this was correct
   (`test_cache_writes_not_surcharged_on_models_without_the_rule`), so the bug
   was pinned in place by a passing test.
2. There was a single `CACHE_WRITE_MULTIPLIER = 1.25`, which is the
   **5-minute** ephemeral rate. Claude Code caches with a **1-hour** TTL, which
   bills at **2.0×**.

**Evidence.** Every assistant turn in a Claude Code transcript carries the
split explicitly, and it is unambiguous:

```json
"cache_creation": {"ephemeral_1h_input_tokens": 10443, "ephemeral_5m_input_tokens": 0}
```

Across all 4,186 imported turns: **26,746,714 cache-write tokens, 100% of them
1-hour.** Not a majority, all of them.

**Impact.** Cache writes were priced at $116.40 and should have been $232.80.
The corrected corpus totals **$800.33**; under the old engine it read
**$683.93**. A **17.0% understatement**, on the single largest figure this
project reports.

**Fix.** Split the multiplier by TTL (`CACHE_WRITE_MULTIPLIER_5M` / `_1H`), add
`claude` to the billed-models tuple, thread a `cache_write_1h_tokens` subset
through `calculate_cost()` and the schema, and read the real split from the
transcript in the importer. The test that asserted the wrong behaviour was
rewritten into two regression tests that assert the right one.

**What this changes about how the project is built.** A hand-verified pricing
table is a liability that looks like an asset, it is confidently wrong and
nothing contradicts it. `src/pricing_drift.py` now reconciles every rate
against a public, community-maintained price map and reports disagreement
without ever rewriting a rate. It earned its place on its first run by catching
a second, unrelated error: the Sonnet 5 entry misses the **introductory
pricing** ($2/$10 per MTok, active through 2026-08-31) and is over-priced
against it today.

**The honest caveat.** The 1h/5m split is now read from the data, but the
project still has no automatic date-ranged pricing, so introductory and
promotional rates are not modelled, see Known gaps.

---

## 13. The live trace: what actually happened the first time

Real output from `scripts/gateway_demo.py`, five real calls through the running
gateway against a three-tier ladder
(`ollama/llama3.2:3b, gemini-flash-lite-latest, gemini-pro-latest`):

```
reformat this JSON: {'a':1,'b':2}
   tier=trivial  -> ollama/llama3.2:3b          $0.000000   2786ms
Can you take a look at the user service?
   tier=moderate -> gemini-flash-lite-latest    $0.000057   1160ms
Why is this test flaky? Walk me through the possible race conditions.
   tier=complex  -> ollama/llama3.2:3b          $0.000000  22763ms
                    (FELL BACK from gemini-pro-latest: rate limit / quota exhausted)
Design a zero-downtime schema migration strategy…
   tier=complex  -> gemini-flash-lite-latest    $0.000456   5411ms

total: $0.000512 across 5 calls, 1 fallback
```

Three things in that trace are worth more than the cost number:

**It produced this project's first real failure data.** Before the gateway, the
ledger held 4,186 real rows with **zero** failures, because they were imported
from transcripts of calls that had already succeeded. The `lowest-failure`
routing strategy had, quite literally, nothing to rank on. The first live run
recorded four genuine 429s, a persisted cooldown, and a within-group fallback.

**The fallback initially looked like a routing bug.** The response named
`ollama/llama3.2:3b` while the decision record named `gemini-pro-latest`, with
nothing connecting them. The fix was not to the router, which was correct, but to
the record: a fallback now writes `fell_back_from` and `fell_back_reason` into the
decision trail. A decision record that omits the substitution is worse than none,
it is a confident wrong answer about what happened.

**The last call went to the mid tier despite classifying `complex`,** because
`gemini-pro-latest` was still benched from the previous call's 429. That is hard
constraints filtering before the strategy ranks, working exactly as §6 describes,
and it is visible only because the run was real.

---

## Known gaps

Stated plainly rather than hidden:

- **The complexity classifier is a cheap prior, not an oracle.** It reads English,
  reads the prompt alone rather than the conversation around it, and its verb
  lists are a judgement call rather than a trained boundary. `src/shadow.py`
  exists to measure how good a prior it actually is; until shadow data is
  collected and *graded*, the tier boundaries are reasoning, not evidence.
- **`simulate_routing` cannot evaluate `complexity`.** The ledger stores no
  prompt text by design, so that combination degrades to the middle tier and
  labels itself as not representative.
- **Streaming and tool calling cannot be combined.** Each works alone; together
  the gateway 400s, because reassembling a tool call from partial argument
  deltas has not been done. For a streaming agent client that is a real
  limitation.
- **Gemini has no tool-calling adapter here.** Anthropic, OpenAI and Ollama do.
  Gemini's function-calling shape differs enough to need its own translation
  layer, and writing one that has never been run against the live API would be
  guessing in public.
- **The Anthropic and OpenAI streaming and tool paths are unit-tested against
  captured response shapes, not live endpoints,** because no keys are
  configured. Only the Ollama paths have been exercised end to end for real.
- **The local judge is a 3B model grading a 3B model.** Blind, order-randomised,
  and treated as triage rather than evidence, but the limitation is structural
  and no amount of prompt care removes it. Deterministic verdicts are the part
  that carries weight; filter on `scored_by` before quoting any acceptance rate.
- **The gateway is unauthenticated unless `WATCHDOG_GATEWAY_KEY` is set,** and
  `/v1/models` says so in a `warning` field. Fine on localhost, unsafe anywhere
  else.
- **Shadow rows store full prompt text.** The one deliberate exception to the
  never-log-prompts rule, opt-in, isolated to its own table, deletable.

- **Pricing is a snapshot, now with a second opinion.** Rates are still hardcoded and still need an edit when a vendor moves them, but `python -m src.pricing_drift` reconciles every mapped rate against a public, community-maintained price map and reports disagreement. It never rewrites a rate, silently re-pricing recorded history is worse than a stale number.
- **No date-ranged pricing.** The table holds one rate per model, so introductory and promotional pricing is not modelled. The drift checker currently flags Sonnet 5 for exactly this reason: the table carries $3/$15 per MTok while the introductory $2/$10 is in effect through 2026-08-31, so Sonnet-5 traffic in that window is over-priced here. Fixing it properly means date-ranged rates, which is a real change rather than a table edit.
- **Live-tested against Google and Ollama only.** The Anthropic and OpenAI adapters are unit-tested against captured response shapes but have not been exercised against a live endpoint in this repo, no keys were configured. The Gemini and local paths have made real, tracked calls through the gateway. This is not just a note in a README: `get_provider_breakdown` reports `live_calls: 0` for both, and the dashboard labels their bars **NO LIVE CALLS**.
- **The Gemini free-tier quota is exhausted,** which is why the live trace in §13 shows `gemini-pro-latest` returning 429 and falling back. That made for better evidence than a clean run would have, but it does mean the frontier tier of the demo ladder is not currently reachable.
- **The shipped DB mixes one fake project with real ones.** `job-search-agent` is illustrative (`demo_job_search_agent.json`, `source=demo`, ~$19.4, never billed, scaled up from an earlier ~$0.20 revision purely so it isn't dwarfed to invisibility next to real numbers 1000x its size; the labeling, badges, and filtering are unchanged, it's still openly fake). `llm-cost-watchdog-build` / `civil-prep-build` are real *build-time* Claude Code usage imported from local session transcripts (`source=manual`, ~$130 combined, see [§8d](#8d-what-it-actually-cost-to-build-these-projects-with-claude-code)). The small real *runtime* rows from [§8b](#8b-real-integration-tracking-civil-prep)/[§8c](#8c-portfolio-survey-does-this-hold-up-across-every-real-project) (`civil-prep` / `cost-watchdog-self`, `source=live`, a few thousandths of a dollar) were removed from the shipped DB, several orders of magnitude smaller than build cost, they cluttered the Cost-by-Project view without changing the conclusion. The methodology and code are unchanged and reproducible; §8b/§8c describe what actually happened when they ran, they just aren't sitting in the current snapshot. Run `python -m src.tracker --provenance` for the live numbers, and `--purge demo` to drop the fake project entirely.
- **The digest's LLM path is exercised via its fallback.** The free-tier quota was exhausted during development, so the deterministic summary is what's been observed end-to-end. Both paths are tested.
- **Burn rate extrapolates linearly** from the observed span. A burst in a short window projects a misleadingly high rate, which is why every projection carries a `confidence` field.
