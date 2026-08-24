#!/usr/bin/env python3
"""ccost: what your AI coding assistant is costing you, and when to start over.

    ccost              what is happening right now
    ccost week         the last seven days
    ccost projects     every project, ranked

Reads the session logs Claude Code already writes to ~/.claude/projects. Nothing
to install into your workflow, no API key, no account, and it never sends
anything anywhere. That is deliberate: the predecessor to this tool required you
to route your traffic through a proxy, and in six months not one person did,
including its author. A tool that reads files you already have is one you will
actually run.

**Why the "start a fresh session" advice is the point.**

The expensive part of agentic coding is not the code the model writes, it is the
context it reads before writing. Measured across 4,399 real requests: 78.5% of
the bill is input, and the model reads 287 tokens for every one it produces.

Context accumulates inside a session and never shrinks. Measured across 43 long
sessions, grouped by how far through the session each turn was:

    first 10% of turns     60,058 tokens of context      1.0x
    40-50%                276,452                        4.6x
    last 10%              337,322                        5.6x

The same question costs five times more at the end of a long session than at the
start, and nothing in the interface tells you. That is what this reports.

The advice is not "have shorter conversations". It is "when you finish a piece
of work, start a new session for the next one", which costs nothing and is
invisible until someone measures it.
"""

import json
import os
import sqlite3
import pathlib
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src.pricing import PRICING_TABLE, calculate_cost  # noqa: E402

TRANSCRIPTS = pathlib.Path.home() / ".claude" / "projects"
CODEX_SESSIONS = pathlib.Path.home() / ".codex" / "sessions"
COPILOT_DB = pathlib.Path.home() / ".copilot" / "session-store.db"

# Context per turn beyond which a fresh session is worth suggesting. Set at the
# point the measurement above shows the cost has roughly tripled, not at a round
# number: below this, restarting saves less than the cost of losing your thread.
RESTART_SUGGEST_TOKENS = 200_000

BOLD, DIM, RESET = "\033[1m", "\033[2m", "\033[0m"
GREEN, YELLOW, RED = "\033[32m", "\033[33m", "\033[31m"
if not sys.stdout.isatty() or os.environ.get("NO_COLOR"):
    BOLD = DIM = RESET = GREEN = YELLOW = RED = ""


def money(v: float) -> str:
    if v >= 100:
        return f"${v:,.0f}"
    if v >= 1:
        return f"${v:,.2f}"
    return f"${v:.3f}"


def human(n: int) -> str:
    if n >= 1_000_000_000:
        return f"{n / 1_000_000_000:.2f}B"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.0f}K"
    return str(n)


def project_name(encoded: str) -> str:
    """Turn an encoded transcript directory into something worth reading.

    Claude Code names these after the working directory with every separator
    flattened to a dash, so they arrive as
    "-Users-dhruvsharma-Desktop-RES-PROJECTS-P2-JSW". Printing the whole path and
    truncating it left "Desktop/RES/PROJECTS/Mum/R", which cuts mid-word and
    hides the only part that identifies the project. The tail is the name.

    The username is dropped positionally rather than by matching a known string,
    so this works on anyone's machine.
    """
    parts = [p for p in encoded.split("-") if p]
    if parts and parts[0] == "Users":
        parts = parts[2:]          # drop "Users" and whoever's home this is
    # Containers that group projects but never name one.
    while parts and parts[0] in ("Desktop", "Documents", "Downloads", "RES", "PROJECTS"):
        parts.pop(0)
    if not parts:
        return "~"                 # the home directory itself
    return "-".join(parts[-3:])


def _model_of(rec: dict) -> str:
    """Map a transcript's model id onto a priced one.

    Transcripts carry dated ids like claude-sonnet-5-20260114. The pricing table
    is keyed on the family. An unpriced model is skipped rather than guessed at,
    because inventing a rate is how a cost tool starts lying.
    """
    m = (rec.get("message", {}) or {}).get("model") or ""
    if m in PRICING_TABLE:
        return m
    for known in PRICING_TABLE:
        if m.startswith(known):
            return known
    return ""


def read_codex_sessions() -> list[dict]:
    """OpenAI Codex sessions, from ~/.codex/sessions.

    Codex records the same thing Claude Code does under a different shape: each
    `token_count` event carries input, cached, cache-write and output counts, so
    it prices through the identical three-rate model with no special cases.

    `last_token_usage` is the turn; `total_token_usage` is the running total for
    the session. Summing the running total would count every turn again for each
    turn that followed it, which on a long session overstates by orders of
    magnitude, so only the per-turn figure is read.
    """
    sessions = []
    if not CODEX_SESSIONS.exists():
        return sessions

    for path in sorted(CODEX_SESSIONS.rglob("*.jsonl")):
        turns, cost, model, cwd = [], 0.0, "", ""
        for line in path.open(errors="ignore"):
            try:
                rec = json.loads(line)
            except Exception:
                continue
            payload = rec.get("payload") or {}

            if not model:
                model = payload.get("model") or model
            if not cwd:
                cwd = payload.get("cwd") or cwd

            if payload.get("type") != "token_count":
                continue
            use = (payload.get("info") or {}).get("last_token_usage") or {}
            if not use:
                continue

            cached = use.get("cached_input_tokens", 0)
            written = use.get("cache_write_input_tokens", 0)
            fresh = max(use.get("input_tokens", 0) - cached - written, 0)
            # Reasoning tokens are billed as output and are not included in
            # output_tokens, so leaving them out undercounts a reasoning model.
            out = use.get("output_tokens", 0) + use.get("reasoning_output_tokens", 0)
            ctx = fresh + cached + written
            if ctx <= 0:
                continue

            priced = model if model in PRICING_TABLE else ""
            if not priced:
                for known in PRICING_TABLE:
                    if model.startswith(known):
                        priced = known
                        break
            if priced:
                cost += calculate_cost(priced, ctx, out, cached, written)
            turns.append(ctx)

        if turns:
            name = pathlib.Path(cwd).name if cwd else path.parent.name
            sessions.append({
                "tool": "codex", "project": name or "codex", "file": path.name,
                "turns": turns, "cost": cost, "priced": True,
                "mtime": path.stat().st_mtime,
            })
    return sessions


def read_copilot_sessions() -> list[dict]:
    """GitHub Copilot CLI sessions, from its local SQLite store.

    Copilot records what was said and never how many tokens it took: the `turns`
    table is user_message, assistant_response, timestamp. It is a flat-fee
    product with no per-token accounting exposed locally, so there is nothing
    here to price and guessing a number would be worse than reporting none.

    These sessions are therefore carried with `priced: False` and counted as
    activity only. A cost tool that quietly invents the number it exists to
    report has failed at the one thing it is for.
    """
    if not COPILOT_DB.exists():
        return []
    try:
        conn = sqlite3.connect(f"file:{COPILOT_DB}?mode=ro", uri=True)
    except sqlite3.Error:
        return []

    sessions = []
    try:
        rows = conn.execute(
            "SELECT s.id, s.cwd, COUNT(t.id), MAX(t.timestamp) "
            "FROM sessions s LEFT JOIN turns t ON t.session_id = s.id GROUP BY s.id"
        ).fetchall()
    except sqlite3.Error:
        return []
    finally:
        conn.close()

    for sid, cwd, n_turns, last_ts in rows:
        if not n_turns:
            continue
        sessions.append({
            "tool": "copilot",
            "project": pathlib.Path(cwd).name if cwd else "copilot",
            "file": str(sid), "turns": [], "cost": 0.0, "priced": False,
            "message_count": n_turns,
            "mtime": COPILOT_DB.stat().st_mtime,
        })
    return sessions


def read_all_sessions() -> list[dict]:
    """Every assistant on this machine that leaves something readable behind."""
    return read_sessions() + read_codex_sessions() + read_copilot_sessions()


def read_sessions() -> list[dict]:
    """Every session on disk, with its turns priced."""
    sessions = []
    for path in sorted(TRANSCRIPTS.rglob("*.jsonl")):
        turns, cost, first, last = [], 0.0, None, None
        for line in path.open(errors="ignore"):
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if rec.get("type") != "assistant":
                continue
            u = (rec.get("message", {}) or {}).get("usage") or {}
            if not u:
                continue
            model = _model_of(rec)
            if not model:
                continue

            cached = u.get("cache_read_input_tokens", 0)
            written = u.get("cache_creation_input_tokens", 0)
            fresh = u.get("input_tokens", 0)
            out = u.get("output_tokens", 0)
            ctx = fresh + cached + written

            cost += calculate_cost(model, fresh + cached + written, out, cached, written)
            turns.append(ctx)
            ts = rec.get("timestamp")
            if ts:
                first = first or ts
                last = ts

        if turns:
            sessions.append({
                "tool": "claude-code", "priced": True,
                "project": project_name(path.parent.name),
                "file": path.name,
                "turns": turns,
                "cost": cost,
                "first": first,
                "last": last,
                "mtime": path.stat().st_mtime,
            })
    return sessions


def _parse(ts: str | None):
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return None


# --- commands -------------------------------------------------------------


def cmd_now(sessions: list[dict]) -> None:
    """The session you are in, and whether it has got expensive."""
    if not sessions:
        print("No Claude Code sessions found under ~/.claude/projects")
        return
    # Only a session with token counts has a context size to advise on. Copilot
    # publishes none, so it can never be the subject of "your context has grown".
    priced = [x for x in sessions if x.get("priced") and x["turns"]]
    if not priced:
        print("No session with token counts. Copilot does not publish them; "
              "Claude Code and Codex do.")
        return
    s = max(priced, key=lambda x: x["mtime"])
    turns = s["turns"]
    ctx = turns[-1]
    start = turns[0] if turns else 0
    growth = ctx / start if start else 1.0

    when = datetime.fromtimestamp(s["mtime"], tz=timezone.utc)
    ago = datetime.now(timezone.utc) - when
    mins = int(ago.total_seconds() // 60)
    ago_s = "just now" if mins < 2 else f"{mins} min ago" if mins < 90 else f"{mins // 60}h ago"

    print(f"\n{BOLD}Most recent session{RESET}  "
          f"{DIM}{s['project']} · {s.get('tool', 'claude-code')} · {ago_s}{RESET}")
    print(f"  {len(turns)} turns, {money(s['cost'])} of model time at list prices")
    print(f"  context now {BOLD}{human(ctx)} tokens{RESET} per turn, "
          f"up from {human(start)} at the start ({growth:.1f}x)")

    if ctx >= RESTART_SUGGEST_TOKENS:
        colour = RED if ctx >= 350_000 else YELLOW
        # What the next turn costs now, against what it would cost fresh.
        here = calculate_cost("claude-sonnet-5", ctx, 1500, int(ctx * 0.95), 0)
        fresh = calculate_cost("claude-sonnet-5", start, 1500, int(start * 0.95), 0)
        print(f"\n  {colour}Consider starting a new session for your next task.{RESET}")
        print(f"  Each further turn here reads {human(ctx)} tokens before it answers.")
        if fresh > 0:
            print(f"  Roughly {money(here)} per turn now against {money(fresh)} fresh, "
                  f"about {here / fresh:.0f}x.")
        print(f"  {DIM}Context never shrinks inside a session. Finishing a piece of work is "
              f"the natural place to start over.{RESET}")
    else:
        print(f"\n  {GREEN}Context is still small. Nothing to do.{RESET}")


def cmd_week(sessions: list[dict]) -> None:
    cutoff = datetime.now(timezone.utc) - timedelta(days=7)
    recent = [s for s in sessions if datetime.fromtimestamp(s["mtime"], tz=timezone.utc) >= cutoff]
    if not recent:
        print("\nNothing in the last seven days.")
        return

    priced = [s for s in recent if s.get("priced")]
    unpriced = [s for s in recent if not s.get("priced")]
    total = sum(s["cost"] for s in priced)
    turns = sum(len(s["turns"]) for s in priced)
    read = sum(sum(s["turns"]) for s in priced)

    by_tool: dict[str, int] = defaultdict(int)
    for s in recent:
        by_tool[s.get("tool", "claude-code")] += 1

    print(f"\n{BOLD}Last 7 days{RESET}")
    print(f"  {len(recent)} sessions across "
          f"{', '.join(f'{k} ({v})' for k, v in sorted(by_tool.items()))}")
    print(f"  {turns:,} turns with token counts")
    print(f"  {money(total)} of model time at list prices")
    print(f"  {human(read)} tokens read {DIM}(this is where the money goes){RESET}")

    by_project: dict[str, float] = defaultdict(float)
    for s in priced:
        by_project[s["project"]] += s["cost"]
    print(f"\n  {DIM}by project{RESET}")
    for proj, c in sorted(by_project.items(), key=lambda kv: -kv[1])[:8]:
        bar = "#" * max(1, int(28 * c / max(total, 1e-9)))
        print(f"    {proj[:26]:26} {money(c):>9}  {DIM}{bar}{RESET}")

    if unpriced:
        msgs = sum(s.get("message_count", 0) for s in unpriced)
        print(f"\n  {DIM}Also {len(unpriced)} Copilot session(s), {msgs} messages. "
              f"Copilot publishes no token counts locally, so these are counted "
              f"but not priced.{RESET}")

    heavy = [s for s in priced if s["turns"] and s["turns"][-1] >= RESTART_SUGGEST_TOKENS]
    if heavy:
        print(f"\n  {YELLOW}{len(heavy)} of these sessions grew past "
              f"{human(RESTART_SUGGEST_TOKENS)} tokens of context.{RESET}")
        print(f"  {DIM}Those turns each re-read a large codebase before answering.{RESET}")


def cmd_projects(sessions: list[dict]) -> None:
    by: dict[str, dict] = defaultdict(
        lambda: {"cost": 0.0, "turns": 0, "read": 0, "sessions": 0, "tools": set()})
    for s in (x for x in sessions if x.get("priced")):
        by[s["project"]]["tools"].add(s.get("tool", "claude-code"))
        b = by[s["project"]]
        b["cost"] += s["cost"]
        b["turns"] += len(s["turns"])
        b["read"] += sum(s["turns"])
        b["sessions"] += 1

    print(f"\n{BOLD}Every project{RESET}  {DIM}list prices, all time{RESET}")
    print(f"  {'project':28}{'cost':>10}{'turns':>8}{'sessions':>10}{'read':>9}")
    for proj, b in sorted(by.items(), key=lambda kv: -kv[1]["cost"])[:25]:
        print(f"  {proj[:26]:28}{money(b['cost']):>10}{b['turns']:>8}"
              f"{b['sessions']:>10}{human(b['read']):>9}")
    total = sum(b["cost"] for b in by.values())
    print(f"  {'':28}{money(total):>10}  {DIM}total{RESET}")


def snapshot(sessions: list[dict]) -> dict:
    """A shareable example of what this tool says, with no prompt text in it.

    The panel it feeds is local by nature: it reads ~/.claude/projects, which
    exists on the machine that did the work and nowhere else. That left the
    deployed site showing a study and no tool at all, so the one thing a reader
    could act on was invisible to every reader who was not its author.

    This ships the shape of the answer instead. Project names, counts and prices
    only. No prompts, no file paths, no code: nothing here reveals what was being
    worked on, which is the property that makes it safe to publish at all.
    """
    import datetime as dt

    priced = [s for s in sessions if s.get("priced") and s["turns"]] or sessions
    current = max(priced, key=lambda s: s["mtime"])
    turns = current["turns"]
    cutoff = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=7)
    recent = [s for s in sessions
              if dt.datetime.fromtimestamp(s["mtime"], tz=dt.timezone.utc) >= cutoff]

    by: dict[str, float] = {}
    for s in recent:
        by[s["project"]] = by.get(s["project"], 0.0) + s["cost"]

    return {
        "generated_at": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat(),
        "current": {
            "project": current["project"],
            "turns": len(turns),
            "cost_usd": round(current["cost"], 2),
            "context_now": turns[-1],
            "context_start": turns[0],
            "growth": round(turns[-1] / turns[0], 1) if turns[0] else 1.0,
            "should_restart": turns[-1] >= RESTART_SUGGEST_TOKENS,
        },
        "week": {
            "sessions": len(recent),
            "turns": sum(len(s["turns"]) for s in recent),
            "cost_usd": round(sum(s["cost"] for s in recent), 2),
            "tokens_read": sum(sum(s["turns"]) for s in recent),
            "grown_past_threshold": sum(
                1 for s in recent if s["turns"] and s["turns"][-1] >= RESTART_SUGGEST_TOKENS),
            "by_project": sorted(
                ({"project": p, "cost_usd": round(c, 2)} for p, c in by.items()),
                key=lambda r: -r["cost_usd"])[:8],
        },
        "threshold_tokens": RESTART_SUGGEST_TOKENS,
    }


def main() -> int:
    if not TRANSCRIPTS.exists():
        print(f"No Claude Code data at {TRANSCRIPTS}")
        return 1

    cmd = (sys.argv[1] if len(sys.argv) > 1 else "now").lower()
    if cmd == "--snapshot":
        out = pathlib.Path(sys.argv[2] if len(sys.argv) > 2 else "data/ccost_snapshot.json")
        sessions = read_all_sessions()
        if not sessions:
            print("nothing to snapshot")
            return 1
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(snapshot(sessions), indent=2) + "\n")
        print(f"wrote {out}")
        return 0
    if cmd in ("-h", "--help", "help"):
        print(__doc__)
        return 0

    sessions = read_all_sessions()
    if not sessions:
        print("No priced sessions found. Models in your transcripts may be newer "
              "than this tool's price table.")
        return 1

    if cmd == "now":
        cmd_now(sessions)
    elif cmd == "week":
        cmd_week(sessions)
    elif cmd in ("projects", "project"):
        cmd_projects(sessions)
    else:
        print(f"Unknown command {cmd!r}. Try: now, week, projects")
        return 1
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
