#!/usr/bin/env python3
"""
Import every tracked project's Claude Code build usage in one pass.

`import_claude_code_usage.py` handles one transcript at a time. This is the
driver that knows *which* transcripts belong to which project, so adding a new
project to the dashboard is a one-line edit here instead of a remembered
command. Run it after building anything new:

    python scripts/import_all_projects.py             # import everything
    python scripts/import_all_projects.py --dry-run   # show the plan only
    python scripts/import_all_projects.py --rebuild   # purge + reimport clean

Two things make this more than a for-loop:

**Folders move; transcripts don't.** Claude Code encodes a transcript path from
the working directory at the time, so a project that has since been renamed or
relocated still lives under its old encoded name. `~/Desktop/CART` is now
`~/Desktop/C2C`, and `~/Desktop/Sebhorric dermatitis` is now `~/Desktop/scalp
log`. The mapping below is therefore historical, not a listing of what is on
disk today, and must not be "fixed" by pointing it at current paths.

**One working directory can hold several projects.** `~/Desktop/CART/C2C
PORTFOLIO` held Saans, LastKilometre, Tollgate and FlatmatePlan at once, so its
single transcript interleaves all four and cannot be imported under one tag.
Those turns are attributed individually by the absolute file paths appearing in
each turn's tool calls — see `_attribute()`.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.import_claude_code_usage import load  # noqa: E402
from src.tracker import get_events, purge_source  # noqa: E402

TRANSCRIPT_ROOT = Path.home() / ".claude" / "projects"

# One encoded transcript directory -> one project tag.
# Keys are the *historical* encoded working directory (see module docstring).
PROJECTS: dict[str, str] = {
    "-Users-dhruvsharma-Desktop-P2-JSW-llm-cost-watchdog": "llm-cost-watchdog",
    "-Users-dhruvsharma-Desktop-P2": "p2-jsw",
    "-Users-dhruvsharma-upsc-rag-assistant": "civil-prep",
    "-Users-dhruvsharma-Desktop-CART": "clip2cart",
    "-Users-dhruvsharma-Desktop-PMP-LastKilometre": "last-kilometre",
    "-Users-dhruvsharma-Desktop-umbra": "umbra",
    "-Users-dhruvsharma-Desktop-prahar": "prahar",
    "-Users-dhruvsharma-Desktop-GTM": "gtm",
    "-Users-dhruvsharma-Desktop-Sebhorric-dermatitis": "scalp-log",
    "-Users-dhruvsharma-Desktop-Brainstorm-ibs": "ibs",
    "-Users-dhruvsharma-Desktop-Brainstorm": "brainstorm",
    "-Users-dhruvsharma-Desktop-P1-URA": "p1-ura",
}

# Transcript directories that interleave several projects. Maps a path
# fragment appearing in a turn's tool calls to the project tag it belongs to.
MIXED_PROJECTS: dict[str, dict[str, str]] = {
    "-Users-dhruvsharma-Desktop-CART-C2C-PORTFOLIO": {
        "/Saans": "saans",
        "/LastKilometre": "last-kilometre",
    },
}

# Deliberately never imported. Tollgate was an LLM cost-router the user
# rejected as unoriginal; FlatmatePlan was dropped and is gone from disk.
# Neither is part of the portfolio, so their turns are attributed nowhere
# rather than silently folded into a neighbouring project's cost.
EXCLUDED_FRAGMENTS = ("/Tollgate", "/FlatmatePlan")

# The Saans work begins at this user prompt, ~14 minutes before a `Saans/`
# folder exists on disk — so path attribution alone cannot see it. After this
# instant Saans is the only project in that transcript. Timestamps are only
# safe to use here because this one boundary was verified by hand; everything
# else is attributed by path, since Tollgate and LastKilometre overlap in time.
SAANS_CUTOVER = "2026-08-08T15:56:51Z"


def _tool_paths(record: dict) -> list[str]:
    """Absolute-ish path strings mentioned anywhere in a record's tool calls."""
    found: list[str] = []
    msg = record.get("message") or {}
    content = msg.get("content")
    if not isinstance(content, list):
        return found
    for block in content:
        if not isinstance(block, dict) or block.get("type") != "tool_use":
            continue
        blob = json.dumps(block.get("input") or {})
        found.append(blob)
    return found


def _attribute(transcript: Path, fragments: dict[str, str]) -> dict[str, set]:
    """Map message-id -> project tag for a transcript holding several projects.

    Attribution carries forward: a turn that touches no file belongs to
    whichever project the conversation was last working on. Timestamps are not
    used as a general signal because two of these projects overlap in time.
    """
    by_project: dict[str, set] = {tag: set() for tag in fragments.values()}
    current: str | None = None

    with open(transcript, errors="ignore") as f:
        for line in f:
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue

            ts = d.get("timestamp") or ""
            blobs = " ".join(_tool_paths(d))

            if any(frag in blobs for frag in EXCLUDED_FRAGMENTS):
                current = None  # excluded project — attribute nothing
            else:
                for frag, tag in fragments.items():
                    if frag in blobs:
                        current = tag
                        break
            if ts >= SAANS_CUTOVER and "saans" in by_project:
                current = "saans"

            if d.get("type") == "assistant" and current:
                mid = (d.get("message") or {}).get("id")
                if mid:
                    by_project[current].add(mid)

    return by_project


def _transcripts(dir_name: str) -> list[Path]:
    d = TRANSCRIPT_ROOT / dir_name
    return sorted(d.glob("*.jsonl")) if d.is_dir() else []


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="Show what would be imported, write nothing")
    ap.add_argument("--rebuild", action="store_true",
                    help="Purge existing subscription rows and their checkpoints first, "
                         "so everything is re-priced with the current pricing table")
    ap.add_argument("--db-path", default=None)
    ap.add_argument("--source", default="subscription", choices=["subscription", "manual"])
    args = ap.parse_args()

    if args.rebuild and not args.dry_run:
        removed = purge_source(args.source, db_path=args.db_path)
        stale = 0
        for d in list(PROJECTS) + list(MIXED_PROJECTS):
            for cp in (TRANSCRIPT_ROOT / d).glob(".*.imported.json"):
                cp.unlink()
                stale += 1
        print(f"rebuild: purged {removed} '{args.source}' row(s), cleared {stale} checkpoint(s)\n")

    grand_cost, grand_turns = 0.0, 0
    rows: list[tuple[str, int, float]] = []

    for dir_name, tag in PROJECTS.items():
        files = _transcripts(dir_name)
        if not files:
            print(f"  !  {tag:<20} no transcript found at {dir_name}")
            continue
        turns, cost = 0, 0.0
        for f in files:
            if args.dry_run:
                print(f"  ~  {tag:<20} would import {f.name}")
                continue
            st = load(str(f), tag, db_path=args.db_path, source=args.source)
            turns += st["logged"]
            cost += st["total_cost"]
        if not args.dry_run:
            rows.append((tag, turns, cost))
            grand_turns += turns
            grand_cost += cost
            print(f"  +  {tag:<20} {turns:>5} turns  ${cost:>9.4f}")

    # Mixed directories: attribute per turn, then import each project's subset.
    for dir_name, fragments in MIXED_PROJECTS.items():
        files = _transcripts(dir_name)
        if not files:
            continue
        for f in files:
            assigned = _attribute(f, fragments)
            for tag, ids in assigned.items():
                if args.dry_run:
                    print(f"  ~  {tag:<20} would import {len(ids)} attributed turns from {f.name}")
                    continue
                if not ids:
                    continue
                st = load(str(f), tag, db_path=args.db_path, source=args.source,
                          only_message_ids=ids)
                rows.append((tag, st["logged"], st["total_cost"]))
                grand_turns += st["logged"]
                grand_cost += st["total_cost"]
                print(f"  +  {tag:<20} {st['logged']:>5} turns  ${st['total_cost']:>9.4f}  (attributed)")

    if args.dry_run:
        return

    print(f"\n  =  {'TOTAL':<20} {grand_turns:>5} turns  ${grand_cost:>9.4f}")
    tracked = {e.project_tag for e in get_events(source=args.source, db_path=args.db_path)}
    print(f"\n  {len(tracked)} project(s) now tracked: {', '.join(sorted(tracked))}")


if __name__ == "__main__":
    main()
