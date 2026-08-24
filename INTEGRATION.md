# Integration Guide: Using the Gateway with Your Project

## The simplest way: one environment variable

Any application using the OpenAI Python SDK (or compatible wire format) can route through the gateway without code changes.

### Step 1: Set two environment variables

```bash
export OPENAI_BASE_URL=https://llmcostwatchdog.onrender.com/v1
export OPENAI_API_KEY=wd-<your-project-name>
```

The `wd-` prefix tells the gateway which project this is (used in reports and filtering).

### Step 2: Run your code normally

```python
from openai import OpenAI

client = OpenAI()  # Reads OPENAI_BASE_URL and OPENAI_API_KEY from env
response = client.chat.completions.create(
    model="group:ladder",  # Or any specific model: "gemini-flash-latest", etc
    messages=[{"role": "user", "content": "Your prompt"}]
)
```

**That's it.** Every call is now:
- Routed based on complexity (if using `group:ladder`)
- Recorded in the live ledger
- Measured for cost, latency, and TTFT
- Compared against a cheap model for quality (shadow comparison)

### Step 3: Watch the live dashboard

```
https://llmcostwatchdog.onrender.com/calls?source=live
```

Each call appears here with:
- Model chosen
- Complexity tier assigned
- Cost
- Latency + TTFT
- Token counts
- Project tag

## Which model should I use?

### `group:ladder` (recommended for most code)
Routes automatically by complexity:
- **Trivial** (e.g., "reverse 'hello'") → cheapest model
- **Moderate** (code review, refactoring) → mid-tier
- **Complex** (design docs, architecture) → most capable

Example:
```python
model="group:ladder"
```

### Direct model names
Skip routing, use a specific model:
```python
model="gemini-flash-latest"    # Google's fast model
model="claude-sonnet-5"         # Anthropic's balanced model
model="gpt-4o-mini"            # OpenAI's small model
model="ollama/llama3.2:3b"     # Local (free, slow)
```

## Verifying it works

### Quick test
```python
from openai import OpenAI

client = OpenAI(
    base_url="https://llmcostwatchdog.onrender.com/v1",
    api_key="wd-test"
)

r = client.chat.completions.create(
    model="group:ladder",
    messages=[{"role": "user", "content": "Say 'ok'"}]
)
print(r.choices[0].message.content)
```

### Full test (6 proofs)
```bash
venv/bin/python scripts/proof.py --base https://llmcostwatchdog.onrender.com --key wd-test
```

Shows:
1. SDK compatibility (unmodified OpenAI SDK)
2. Routing (same code, different models per prompt)
3. Real streaming with TTFT
4. Tool calling (agent loops)
5. Guardrail blocking
6. Live ledger capture

## Data that appears in the dashboard

### Per-call metrics
- **Cost**: real billed cost including caching and tier
- **Latency**: wall-clock time end-to-end
- **TTFT**: time to first token (streaming only)
- **Complexity tier**: assigned by the router
- **Model chosen**: what the router picked
- **Project**: extracted from your API key (wd-<project>)

### Aggregated analysis
- **By model**: total cost and calls per model
- **By project**: cost per project
- **Cost levers**: prompt caching savings, model-switch hypotheticals, output length bounds
- **Anomalies**: unusual calls (very high latency, high cost per token)
- **Budget**: weekly spend vs cap

## Common questions

**Q: Does the gateway modify my prompts?**
No. Prompts are passed through unchanged unless tools are offered (which requires a structured message path, not flattening).

**Q: What if a model fails?**
The gateway retries within the same model group's tier. If the entire tier fails, it falls back to the next tier and logs the reason.

**Q: Can I use this with non-OpenAI models?**
Yes. Google Gemini is pre-configured. Anthropic and OpenAI are supported. Add any model by setting env vars and updating `src/providers/`.

**Q: What if I don't want the dashboard seeing my prompts?**
Shadow comparison (quality measurement) stores full prompt text locally. It never leaves your deployment. Disable shadow with `WATCHDOG_SHADOW_RATE=0`.

**Q: How much does it cost?**
$0 on Render free tier. The gateway itself is free; you pay for the LLM calls it routes (at published model prices).
