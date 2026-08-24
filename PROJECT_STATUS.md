# Where this project actually stands

Last verified: 2026-08-25. Every figure below was checked, not remembered.

## What you can use today

```bash
pip install -e .
ccost                  # what your current session is costing
ccost week             # the last seven days, by project
ccost projects         # every project you have worked on
ccost report           # a self-contained HTML file to share
ccost install-hook     # get told mid-session, without asking
```

Reads **Claude Code**, **OpenAI Codex** and **GitHub Copilot** from local logs.
No API key, no account, nothing leaves the machine.

## What works, verified

| | |
|---|---|
| Tests | **575 passing** |
| CLI | runs from any directory after `pip install -e .` |
| MCP server | **22 tools**, verified over a real initialize + tools/list handshake |
| Claude Code hook | fires correctly, throttled, never fails a session |
| HTML report | 8.4 KB, **zero external references**, renders offline |
| Deployed site | 23 projects, 8,765 turns, $1,521.56, all sections rendering |
| CI | tests on push, liveness check twice daily |
| Quality study | 14 comparisons graded blind, published including the bad result |

## The finding it is built on

Measured across 23 projects and 8,765 turns:

| | | |
|---|---:|---:|
| Reading context | **$1,212.68** | 79.7% |
| One-hour cache-write premium | $145.79 | 9.6% |
| Generating output | $163.09 | 10.7% |

**314 tokens read per token written.** Context grows 5.6x across a long session
and never shrinks, which is why the advice is "start a new session", and why
that advice is worth automating.

## And the answer to the question underneath all of it

Every cost figure prices a swap to a cheaper model. None of them ask whether the
cheap answer was any good. So both models ran the same 14 prompts and the pairs
were graded blind:

| tier the router assigned | compared | held up |
|---|---:|---:|
| trivial | 5 | **20%** |
| moderate | 9 | 11% |
| complex | 2 | 0% |

Two of sixteen survived. Asked to reverse `'hello'`, the cheap model answered
`'hell'`. The gap between a 3B and a 1B is wider than between a frontier model
and its cheap sibling, so the percentages do not transfer, but the direction
does: **routing on price without measuring quality is a way to save money by
getting worse answers.**

These are list prices. The work ran on a flat plan costing **$12.92** over the
same span, and the site never presents the larger number as money spent.

## What is honestly incomplete

- **The gateway has no real users.** It works (SDK compatibility, streaming,
  tool calling, guardrails all verified with live traffic) and nobody routes
  through it. It solves a problem its author does not have.
- **The router reaches the cheap tier on 10% of requests**, under the 15% bar
  the validator sets for itself. Reported as a miss rather than moved. See
  [/validation](https://llmcostwatchdog.onrender.com/validation).
- **The quality study is small.** 14 graded comparisons between a 3B and a 1B
  local model. The ordering it found is meaningful; the percentages do not
  transfer to a frontier pair, and the page says so.
- **`WATCHDOG_IMPORT_KEY` on Render does not match the local one**, so
  `auto_sync.sh` cannot push. The deployed ledger was last synced directly
  against Turso. Fix by copying the value from Render into `.env`.

## What is deliberately not claimed

- Not machine learning. The classifier is a scored heuristic and is described as one.
- Not a production system. It is deployed and working; it has no users.
- Not "saves X%". Cost concentration is measured; savings are not.
