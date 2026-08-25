#!/usr/bin/env python3
"""ccost: what your AI coding assistant is costing you, and when to start over.

    ccost              what is happening right now
    ccost week         the last seven days
    ccost projects     every project, ranked
    ccost report       a shareable HTML file, no server needed
    ccost install-hook warn me automatically during a session
    ccost hook         (internal: the Claude Code Stop-hook itself)

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

    ranked = sorted(by.items(), key=lambda kv: -kv[1]["cost"])

    # Everything, unless the list is genuinely unwieldy. This used to cut at 25
    # silently, which hid exactly the projects worth noticing: a new one has
    # almost no cost yet, so it sorts last and vanished on the day you started
    # it. Truncation is now both generous and announced.
    LIMIT = 40
    shown, hidden = ranked[:LIMIT], ranked[LIMIT:]

    print(f"\n{BOLD}Every project{RESET}  {DIM}list prices, all time · "
          f"{len(ranked)} project{'s' if len(ranked) != 1 else ''}{RESET}")
    print(f"  {'project':28}{'cost':>10}{'turns':>8}{'sessions':>10}{'read':>9}")
    for proj, b in shown:
        print(f"  {proj[:26]:28}{money(b['cost']):>10}{b['turns']:>8}"
              f"{b['sessions']:>10}{human(b['read']):>9}")
    if hidden:
        print(f"  {DIM}... and {len(hidden)} smaller "
              f"({money(sum(b['cost'] for _, b in hidden))} between them){RESET}")
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



REPORT_CSS = """
:root{color-scheme:light dark;--bg:#f4f4f2;--card:#fcfcfb;--line:#e2e2dd;
--ink:#0b0b0b;--dim:#52514e;--mute:#7a7975;--read:#2a78d6;--write:#eb6834;--ok:#007a4d;--warn:#9a6700}
@media(prefers-color-scheme:dark){:root{--bg:#111110;--card:#1a1a19;--line:#33322f;
--ink:#fff;--dim:#c3c2b7;--mute:#91908a;--read:#3987e5;--write:#d95926;--ok:#3fb27f;--warn:#d9a441}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);
font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;padding:2rem 1.25rem 4rem}
.wrap{max-width:940px;margin:0 auto}h1{font-size:1.35rem;margin:0 0 .3rem;letter-spacing:-.01em}
.sub{color:var(--dim);font-size:.9rem;margin-bottom:1.6rem}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:1rem 1.15rem;margin-bottom:1rem}
.card.warn{border-left:3px solid var(--warn)}.card.ok{border-left:3px solid var(--ok)}
h2{font-size:.72rem;text-transform:uppercase;letter-spacing:.09em;color:var(--mute);margin:0 0 .6rem;font-weight:600}
.figs{display:flex;gap:2.2rem;flex-wrap:wrap;margin:.2rem 0 .9rem}
.figs .n{font-size:1.6rem;font-weight:650;letter-spacing:-.02em;font-variant-numeric:tabular-nums}
.figs .l{font-size:.78rem;color:var(--dim)}
.split{display:flex;height:40px;border-radius:8px;overflow:hidden;border:1px solid var(--line);margin:.3rem 0 .5rem}
.split div{display:flex;align-items:center;padding:0 .7rem;font-size:.82rem;font-weight:600;color:#fff;min-width:0;overflow:hidden}
.key{display:flex;gap:1.3rem;flex-wrap:wrap;font-size:.8rem;color:var(--mute)}
.sw{display:inline-block;width:10px;height:10px;border-radius:3px;margin-right:.35rem}
table{width:100%;border-collapse:collapse;font-size:.85rem}
th,td{text-align:left;padding:.45rem .6rem;border-bottom:1px solid var(--line);white-space:nowrap}
th{font-size:.7rem;text-transform:uppercase;letter-spacing:.04em;color:var(--mute);font-weight:500}
td.n{text-align:right;font-variant-numeric:tabular-nums}tr:last-child td{border-bottom:none}
.note{color:var(--mute);font-size:.8rem;line-height:1.6;margin:.6rem 0 0}
.scroll{overflow-x:auto}
"""



def context_growth_curve(sessions: list[dict], min_turns: int = 20) -> list[int]:
    """Median context per turn at each tenth of a session, across long sessions.

    This is the shape the whole tool argues from, and it is worth drawing rather
    than asserting. Bucketing by *position* rather than by turn number is what
    makes sessions of different lengths comparable: the tenth turn of a 20-turn
    session and the hundredth of a 200-turn one are the same point in the arc.
    """
    import statistics
    from collections import defaultdict

    buckets: dict[int, list[int]] = defaultdict(list)
    for s in sessions:
        turns = s.get("turns") or []
        if len(turns) < min_turns:
            continue
        n = len(turns)
        for i, ctx in enumerate(turns):
            buckets[min(int(10 * i / n), 9)].append(ctx)
    if len(buckets) < 10:
        return []
    return [int(statistics.median(buckets[i])) for i in range(10)]


def _growth_svg(curve: list[int]) -> str:
    """An inline SVG bar chart. No library, no script, no external request.

    Drawn as a string because the report has to survive being emailed, opened
    from a file:// URL with the network off, and read years from now. Anything
    that needs fetching fails all three.
    """
    if not curve:
        return ""
    peak = max(curve) or 1
    W, H, PAD = 640, 150, 22
    bar = (W - 2 * PAD) / len(curve) - 6
    bars, labels = [], []
    for i, v in enumerate(curve):
        h = max(3, (v / peak) * (H - 46))
        x = PAD + i * ((W - 2 * PAD) / len(curve))
        y = H - 22 - h
        shade = 0.35 + 0.65 * (v / peak)
        bars.append(
            f'<rect x="{x:.0f}" y="{y:.0f}" width="{bar:.0f}" height="{h:.0f}" '
            f'rx="3" fill="currentColor" opacity="{shade:.2f}"><title>'
            f'{i * 10}-{i * 10 + 10}% through a session: {v:,} tokens</title></rect>'
        )
        if i in (0, 4, 9):
            labels.append(
                f'<text x="{x + bar / 2:.0f}" y="{H - 6}" font-size="10" '
                f'text-anchor="middle" fill="currentColor" opacity=".55">'
                f'{"start" if i == 0 else "halfway" if i == 4 else "end"}</text>'
            )
    return (f'<svg viewBox="0 0 {W} {H}" width="100%" role="img" '
            f'aria-label="Context per message grows across a session">'
            + "".join(bars) + "".join(labels) + "</svg>")


def build_report(sessions: list[dict]) -> str:
    """One self-contained HTML file: no server, no key, no network.

    The dashboard needs a deployment, a hosted database and a shared secret
    before anyone sees a number, which is three walls in front of a person who
    just wants to look at their own usage. This is the same content as a file
    you can double-click, email, or drop on any static host.

    Everything is inlined. It opens with the network off, and it stays readable
    years after this repo does, because there is nothing for it to depend on.
    """
    import datetime as dt
    import html as _html

    priced = [s for s in sessions if s.get("priced") and s["turns"]]
    unpriced = [s for s in sessions if not s.get("priced")]
    e = _html.escape

    total = sum(s["cost"] for s in priced)
    turns = sum(len(s["turns"]) for s in priced)
    read = sum(sum(s["turns"]) for s in priced)

    by_proj: dict[str, dict] = {}
    for s in priced:
        b = by_proj.setdefault(s["project"], {"cost": 0.0, "turns": 0, "read": 0, "n": 0,
                                              "tools": set()})
        b["cost"] += s["cost"]; b["turns"] += len(s["turns"])
        b["read"] += sum(s["turns"]); b["n"] += 1
        b["tools"].add(s.get("tool", "claude-code"))

    cur = max(priced, key=lambda s: s["mtime"]) if priced else None
    rows = "".join(
        f"<tr><td>{e(p)}</td><td>{e(', '.join(sorted(b['tools'])))}</td>"
        f"<td class=n>{money(b['cost'])}</td><td class=n>{b['turns']:,}</td>"
        f"<td class=n>{b['n']}</td><td class=n>{human(b['read'])}</td></tr>"
        for p, b in sorted(by_proj.items(), key=lambda kv: -kv[1]["cost"])
    )

    growth_card = ""
    if cur:
        ctx, start = cur["turns"][-1], cur["turns"][0]
        grew = ctx >= RESTART_SUGGEST_TOKENS
        growth_card = f"""
    <div class="card {'warn' if grew else 'ok'}">
      <h2>Most recent session</h2>
      <div class=figs>
        <div><div class=n>{human(ctx)}</div><div class=l>tokens read per message</div></div>
        <div><div class=n>{(ctx / start if start else 1):.1f}x</div><div class=l>more than at the start</div></div>
        <div><div class=n>{len(cur['turns']):,}</div><div class=l>messages</div></div>
        <div><div class=n>{money(cur['cost'])}</div><div class=l>{e(cur.get('tool', ''))} &middot; {e(cur['project'])}</div></div>
      </div>
      {'<p class=note><b>Start a new session for your next task.</b> Every further message here reads '
       + human(ctx) + ' tokens before it answers, against ' + human(start)
       + ' at the start. Context never shrinks inside a session, so finishing a piece of work is the '
         'natural place to start over. It costs nothing.</p>'
       if grew else '<p class=note>Context is still small. Nothing to do.</p>'}
    </div>"""

    unpriced_note = ""
    if unpriced:
        msgs = sum(s.get("message_count", 0) for s in unpriced)
        unpriced_note = (f"<p class=note>Also {len(unpriced)} Copilot session(s), {msgs} messages. "
                         f"Copilot publishes no token counts locally, so these are counted but "
                         f"never priced: inventing the number would defeat the point.</p>")

    when = dt.datetime.now().strftime("%d %b %Y")
    return f"""<!doctype html><html lang=en><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>AI coding cost report</title><style>{REPORT_CSS}</style></head><body><div class=wrap>
<h1>What my AI coding assistants cost</h1>
<div class=sub>Generated {when} from local session logs. Nothing here was uploaded anywhere.</div>
{growth_card}
<div class=card>
  <h2>All time</h2>
  <div class=figs>
    <div><div class=n>{money(total)}</div><div class=l>at pay-per-use rates</div></div>
    <div><div class=n>{len(by_proj)}</div><div class=l>projects</div></div>
    <div><div class=n>{turns:,}</div><div class=l>messages with token counts</div></div>
    <div><div class=n>{human(read)}</div><div class=l>tokens read</div></div>
  </div>
  <p class=note><b>These are list prices, not a bill.</b> On a flat monthly plan no per-token
  charge occurs. This is what the same tokens would have cost through a metered API, which is
  the only way to compare one session against another.</p>
  {unpriced_note}
</div>
<div class=card>
  <h2>By project</h2>
  <div class=scroll><table>
    <thead><tr><th>project</th><th>assistant</th><th class=n>cost</th>
    <th class=n>messages</th><th class=n>sessions</th><th class=n>read</th></tr></thead>
    <tbody>{rows}</tbody>
  </table></div>
</div>
<div class=card>
  <h2>Context grows for as long as a session lasts</h2>
  <div style="color:var(--read)">{_growth_svg(context_growth_curve(priced))}</div>
  {(
    "<p class=note style='margin-top:.4rem'>Median context per message at each tenth of a "
    "session, across your " + str(sum(1 for s in priced if len(s['turns']) >= 20)) +
    " longest sessions. By the end, each message carries <b>" +
    f"{max(context_growth_curve(priced)) / max(context_growth_curve(priced)[0], 1):.1f}x</b>"
    " what it carried at the start. Nothing you do makes it go back down, which is why "
    "finishing a piece of work and starting fresh is the only lever here that costs nothing."
    "</p>"
  ) if context_growth_curve(priced) else ""}
</div>
<div class=card>
  <h2>Why reading is the number that matters</h2>
  <p class=note style="margin-top:0">Across these sessions the assistant read
  <b>{(read / max(sum(1 for s in priced for _ in s['turns']), 1)):,.0f} tokens per message</b> on average.
  The expensive part of agentic coding is not the code it writes, it is the context it reads
  before writing, and that grows for as long as a session lasts. Finishing a piece of work and
  starting a new session is the one lever that costs nothing to pull.</p>
  <p class=note>Generated by <code>ccost</code>. Run <code>python scripts/ccost.py report</code>
  to rebuild this file from your own logs.</p>
</div>
</div></body></html>
"""



# Where the hook remembers what it has already said. Without this it would warn
# on every single turn once the threshold is crossed, which is how a useful
# warning becomes something people turn off.
HOOK_STATE = pathlib.Path.home() / ".cache" / "ccost" / "hook-state.json"

# Warn again only after context has grown by this much since the last warning.
HOOK_REWARN_TOKENS = 150_000


def session_context(transcript: pathlib.Path) -> tuple[int, int, int]:
    """(context now, context at the start, turns) for one transcript file.

    Reads the real token counts the assistant recorded rather than estimating
    from file size. A transcript is mostly text that was never sent as input,
    so bytes-divided-by-four is wrong by a wide and unpredictable margin, and a
    warning that fires at the wrong time is worse than none.
    """
    first = last = 0
    turns = 0
    try:
        with transcript.open(errors="ignore") as fh:
            for line in fh:
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                if rec.get("type") != "assistant":
                    continue
                u = (rec.get("message") or {}).get("usage") or {}
                if not u:
                    continue
                ctx = (u.get("input_tokens", 0)
                       + u.get("cache_read_input_tokens", 0)
                       + u.get("cache_creation_input_tokens", 0))
                if ctx <= 0:
                    continue
                if not first:
                    first = ctx
                last = ctx
                turns += 1
    except OSError:
        return 0, 0, 0
    return last, first, turns


def cmd_hook() -> int:
    """Claude Code Stop-hook: warn mid-session when context has grown expensive.

    Registered against the `Stop` event, which fires after each assistant turn
    and hands over `transcript_path` on stdin. Printing a JSON object with
    `systemMessage` surfaces the text to the user in the session itself, so the
    advice arrives while it is still actionable rather than the next time
    somebody remembers to run a CLI.

    Always exits 0. A cost tool must never be the reason a session breaks, so
    every failure path here is silent: a missing file, unreadable JSON, an
    unexpected payload shape all end in "say nothing and get out of the way".
    """
    try:
        payload = json.loads(sys.stdin.read() or "{}")
    except ValueError:
        return 0

    path = payload.get("transcript_path")
    if not path:
        return 0
    transcript = pathlib.Path(path)
    if not transcript.exists():
        return 0

    ctx, start, turns = session_context(transcript)
    if ctx < RESTART_SUGGEST_TOKENS or turns < 5:
        return 0

    # Throttle per session, so a long session gets a handful of nudges rather
    # than one per turn.
    session = payload.get("session_id") or transcript.stem
    state = {}
    try:
        state = json.loads(HOOK_STATE.read_text())
    except (OSError, ValueError):
        pass
    if ctx - state.get(session, 0) < HOOK_REWARN_TOKENS:
        return 0

    state[session] = ctx
    try:
        HOOK_STATE.parent.mkdir(parents=True, exist_ok=True)
        # Keep the file small: only sessions still being warned about.
        HOOK_STATE.write_text(json.dumps(dict(list(state.items())[-50:])))
    except OSError:
        pass

    growth = f"{ctx / start:.0f}x" if start else "considerably"
    here = calculate_cost("claude-sonnet-5", ctx, 1500, int(ctx * 0.95), 0)
    fresh = calculate_cost("claude-sonnet-5", start or 1, 1500, int((start or 1) * 0.95), 0)
    ratio = f", about {here / fresh:.0f}x" if fresh > 0 else ""

    print(json.dumps({"systemMessage":
        f"ccost: this session now reads {human(ctx)} tokens per message, {growth} more "
        f"than when it started{ratio}. Context never shrinks inside a session, so "
        f"finishing this piece of work and starting a new one is the cheapest thing "
        f"available. Run `ccost` for detail."}))
    return 0



CLAUDE_SETTINGS = pathlib.Path.home() / ".claude" / "settings.json"


def cmd_install_hook(remove: bool = False) -> int:
    """Register (or remove) the Stop hook in ~/.claude/settings.json.

    Editing that file by hand is a small task that people reasonably decline to
    do, and a warning nobody installs is a warning that does not exist. This
    merges into the existing settings rather than replacing them, because that
    file also holds the user's model and effort preferences and losing those to
    a convenience command would be a poor trade.
    """
    import shutil

    settings = {}
    if CLAUDE_SETTINGS.exists():
        try:
            settings = json.loads(CLAUDE_SETTINGS.read_text())
        except ValueError:
            print(f"{CLAUDE_SETTINGS} is not valid JSON. Not touching it.")
            return 1

    command = f"{sys.executable} {pathlib.Path(__file__).resolve()} hook"
    entry = {
        "matcher": "*",
        "hooks": [{"type": "command", "command": command, "timeout": 10}],
    }

    hooks = settings.setdefault("hooks", {})
    stop = hooks.setdefault("Stop", [])
    # Identify our own entry by the "ccost" in its command, so re-running
    # updates in place instead of stacking duplicates.
    stop[:] = [g for g in stop if "ccost" not in json.dumps(g)]

    if remove:
        if not stop:
            hooks.pop("Stop", None)
        if not hooks:
            settings.pop("hooks", None)
    else:
        stop.append(entry)

    if CLAUDE_SETTINGS.exists():
        shutil.copy2(CLAUDE_SETTINGS, CLAUDE_SETTINGS.with_suffix(".json.ccost-backup"))
    CLAUDE_SETTINGS.parent.mkdir(parents=True, exist_ok=True)
    CLAUDE_SETTINGS.write_text(json.dumps(settings, indent=2) + "\n")

    if remove:
        print(f"removed the ccost hook from {CLAUDE_SETTINGS}")
    else:
        print(f"installed into {CLAUDE_SETTINGS}")
        print(f"  backup: {CLAUDE_SETTINGS.with_suffix('.json.ccost-backup').name}")
        print(f"\n  From now on, when a session's context passes "
              f"{human(RESTART_SUGGEST_TOKENS)} tokens per message,")
        print("  it will tell you, in the session, without being asked.")
        print("  Start a new Claude Code session for it to take effect.")
        print("\n  Undo any time with: ccost install-hook --remove")
    return 0


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

    if cmd in ("install-hook", "uninstall-hook"):
        return cmd_install_hook(remove=(cmd == "uninstall-hook" or "--remove" in sys.argv))
    if cmd == "hook":
        return cmd_hook()
    if cmd == "report":
        out = pathlib.Path(sys.argv[2] if len(sys.argv) > 2 else "ai-cost-report.html")
        out.write_text(build_report(sessions))
        print(f"wrote {out.resolve()}")
        print("  open it, email it, or drop it on any static host. No server, no key.")
        return 0
    if cmd == "now":
        cmd_now(sessions)
    elif cmd == "week":
        cmd_week(sessions)
    elif cmd in ("projects", "project"):
        cmd_projects(sessions)
    else:
        print(f"Unknown command {cmd!r}. Try: now, week, projects, report")
        return 1
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
