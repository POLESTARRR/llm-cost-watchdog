#!/usr/bin/env python3
"""
Import every tracked project's Claude Code build usage in one pass.

`import_claude_code_usage.py` handles one transcript at a time. This is the
driver that knows *which* transcripts belong to which project, so adding a new
project to the dashboard is a one-line edit here instead of a remembered
command. Run it after building anything new:

    python scripts/import_all_projects.py             # import everything, locally
    python scripts/import_all_projects.py --dry-run   # show the plan only
    python scripts/import_all_projects.py --rebuild   # purge + reimport clean

    # Push the same rows to a deployed dashboard instead of the local DB.
    # Independent checkpoint from the local run, see _checkpoint_path's
    # docstring in import_claude_code_usage.py, so this can run any time
    # without re-sending turns the local DB already has, or vice versa.
    python scripts/import_all_projects.py --remote-url https://your-dashboard.example.com

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
each turn's tool calls, see `_attribute()`.
"""
import argparse
import json
import os
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
# folder exists on disk, so path attribution alone cannot see it. After this
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
                current = None  # excluded project, attribute nothing
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


def discover_projects() -> dict[str, str]:
    """Every transcript directory on disk, mapped to a project tag.

    PROJECTS below is a hand-written map and was the only source of truth, which
    meant a project existed to the dashboard only if someone remembered to add a
    line for it. Twenty-two did not, including the largest one by weekly cost.
    That is exactly the kind of quiet omission the site is supposed to be
    incapable of: a total that is wrong because of what was never counted.

    So discovery is now automatic and the hand-written entries become overrides
    rather than the whole list. They still matter and cannot be derived:

      * a directory encodes the working directory as it was, so a project that
        has since been renamed or moved still lives under its old name, and only
        a human knows that ~/Desktop/CART is now ~/Desktop/C2C.
      * MIXED_PROJECTS covers one directory holding several projects at once,
        which no amount of path parsing can separate.

    Anything not named in either map is imported under a tag derived from the
    tail of its path, which is right often enough to be worth doing and is
    always better than being absent.
    """
    found: dict[str, str] = {}
    if not TRANSCRIPT_ROOT.exists():
        return found

    for d in sorted(TRANSCRIPT_ROOT.iterdir()):
        if not d.is_dir() or not any(d.glob("*.jsonl")):
            continue
        if d.name in PROJECTS or d.name in MIXED_PROJECTS:
            continue          # an explicit mapping always wins
        parts = [x for x in d.name.split("-") if x]
        if parts and parts[0] == "Users":
            parts = parts[2:]                      # drop "Users" and the username
        while parts and parts[0] in ("Desktop", "Documents", "Downloads", "RES", "PROJECTS"):
            parts.pop(0)
        tag = "-".join(parts[-3:]).lower() if parts else "home"
        found[d.name] = tag or "home"
    return found


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="Show what would be imported, write nothing")
    ap.add_argument("--rebuild", action="store_true",
                    help="Purge existing subscription rows and their checkpoints first, "
                         "so everything is re-priced with the current pricing table")
    ap.add_argument("--db-path", default=None)
    ap.add_argument("--source", default="subscription", choices=["subscription", "manual"])
    ap.add_argument("--remote-url", default=None,
                    help="Push to a deployed dashboard's /import endpoint instead of writing "
                         "locally, e.g. https://llmcostwatchdog.onrender.com. Uses its own "
                         "per-destination checkpoint, so this can run independently of, and "
                         "as often as, the local import.")
    ap.add_argument("--import-key", default=None,
                    help="WATCHDOG_IMPORT_KEY on the remote deployment. Defaults to the "
                         "WATCHDOG_IMPORT_KEY environment variable.")
    args = ap.parse_args()

    import_key = args.import_key or os.environ.get("WATCHDOG_IMPORT_KEY")
    if args.remote_url and not args.dry_run and not import_key:
        ap.error("--remote-url requires --import-key or a WATCHDOG_IMPORT_KEY env var")

    if args.rebuild and args.remote_url:
        # purge_source() only ever touches the local SQLite file, there is no
        # remote-purge endpoint, and there should not be: wiping a deployed
        # dashboard's history from a CLI flag is exactly the kind of
        # destructive action that needs its own explicit, confirmed step.
        ap.error("--rebuild purges the local DB; it has no effect with --remote-url. "
                 "Rebuild locally first, then push with --remote-url alone.")

    if args.rebuild and not args.dry_run:
        removed = purge_source(args.source, db_path=args.db_path)
        stale = 0
        for d in list(PROJECTS) + list(MIXED_PROJECTS) + list(discover_projects()):
            for cp in (TRANSCRIPT_ROOT / d).glob(".*.imported.json"):
                # Rebuilding the local DB must only clear *local* checkpoints.
                # A remote checkpoint (`.remote-<slug>.imported.json`) tracks a
                # different destination's history; wiping it here would make
                # the next --remote-url run resend already-pushed rows, and
                # POST /import mints a fresh UUID per row, so that duplicates
                # them on the remote instead of no-op'ing.
                if ".remote-" in cp.name:
                    continue
                cp.unlink()
                stale += 1
        print(f"rebuild: purged {removed} '{args.source}' row(s), cleared {stale} local checkpoint(s)\n")

    grand_cost, grand_turns = 0.0, 0
    rows: list[tuple[str, int, float]] = []

    # Explicit mappings first, then everything else found on disk. A project
    # must not be missing from the totals merely because nobody added a line.
    all_projects = {**PROJECTS, **discover_projects()}
    discovered_only = set(discover_projects())
    if discovered_only:
        print(f"  {len(PROJECTS)} mapped, {len(discovered_only)} discovered automatically\n")

    for dir_name, tag in all_projects.items():
        files = _transcripts(dir_name)
        if not files:
            print(f"  !  {tag:<20} no transcript found at {dir_name}")
            continue
        turns, cost = 0, 0.0
        for f in files:
            if args.dry_run:
                print(f"  ~  {tag:<20} would import {f.name}")
                continue
            st = load(str(f), tag, db_path=args.db_path, source=args.source,
                      remote_url=args.remote_url, import_key=import_key)
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
                          only_message_ids=ids, remote_url=args.remote_url, import_key=import_key)
                rows.append((tag, st["logged"], st["total_cost"]))
                grand_turns += st["logged"]
                grand_cost += st["total_cost"]
                print(f"  +  {tag:<20} {st['logged']:>5} turns  ${st['total_cost']:>9.4f}  (attributed)")

    if args.dry_run:
        return

    sink = args.remote_url or "local DB"
    print(f"\n  =  {'TOTAL':<20} {grand_turns:>5} turns  ${grand_cost:>9.4f}  -> {sink}")

    # The post-import "what's tracked now" check reads db_path directly, which
    # is meaningless for a remote push: db_path names the *local* file, not
    # the dashboard that was just pushed to. Verify that case with --provenance
    # against the deployed URL instead (see the README's remote-import section).
    if not args.remote_url:
        tracked = {e.project_tag for e in get_events(source=args.source, db_path=args.db_path)}
        print(f"\n  {len(tracked)} project(s) now tracked: {', '.join(sorted(tracked))}")


if __name__ == "__main__":
    main()
