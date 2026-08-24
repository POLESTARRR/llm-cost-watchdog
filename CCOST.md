# ccost

**What your AI coding assistant is costing you, and when to start over.**

```bash
python scripts/ccost.py            # what is happening right now
python scripts/ccost.py week       # the last seven days
python scripts/ccost.py projects   # every project, ranked
```

No setup. No API key. No account. It reads the session logs Claude Code already
writes to `~/.claude/projects`, and never sends anything anywhere.

## What it tells you

```
Most recent session  P2-JSW · just now
  1269 turns, $201 of model time at list prices
  context now 478K tokens per turn, up from 47K at the start (10.2x)

  Consider starting a new session for your next task.
  Each further turn here reads 478K tokens before it answers.
  Roughly $0.230 per turn now against $0.043 fresh, about 5x.
```

## Why "start a new session" is the whole point

The expensive part of agentic coding is not the code the model writes. It is the
context it reads before writing. Measured across 4,399 real requests: **78.5% of
the bill is input**, and the model reads **287 tokens for every one it produces**.

Context accumulates inside a session and never shrinks. Measured across 43 long
sessions, grouped by how far through each session a turn was:

| position in session | context per turn | vs the start |
|---|---:|---:|
| first 10% | 60,058 | 1.0x |
| 40-50% | 276,452 | 4.6x |
| last 10% | 337,322 | **5.6x** |

**The same question costs five times more at the end of a long session than at
the beginning**, and nothing in the interface tells you.

The advice is not "have shorter conversations". It is: when you finish a piece of
work, start a new session for the next one. That costs nothing, changes nothing
about how you work, and is invisible until someone measures it.

## Notes

- **List prices, not your bill.** On a flat monthly plan no per-token charge
  occurs. These numbers are what the same tokens would cost through a metered
  API, which is the only way to compare one session against another.
- **Cache-aware.** Cache reads bill at roughly a tenth of the input rate and
  cache writes at a premium. A tool that prices every input token at full rate
  overstates a cache-heavy workload several times over.
- **Unpriceable models are skipped, not guessed.** If your transcripts contain a
  model newer than the price table, its turns are left out rather than invented.
