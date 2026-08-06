#!/usr/bin/env python3
"""
Import real Claude Code build-time usage into the watchdog.

Claude Code writes a local JSONL transcript per session (default location:
`~/.claude/projects/<encoded-cwd>/<session-id>.jsonl`), and every assistant
turn in it carries the real `usage` block from the actual Anthropic API
response it received — input/output tokens, cache-read tokens, cache-write
tokens, the works. That is the authoritative source for "what did it cost,
in real tokens, to build this project with Claude Code" — a number this
project's own call_llm() wrapper can never see, because Claude Code isn't
calling itself through this codebase.

This script reads that transcript directly and logs each real turn as a
`source="manual"` event (real spend, reconstructed from an existing record
rather than measured live) under whatever project tag you give it. Dedupes
by the API response's own message id, so re-running on the same transcript
is a no-op the second time. Timestamps are the turn's real historical
timestamps, not "now".

    python scripts/import_claude_code_usage.py \\
        --session ~/.claude/projects/-path-to-project/<uuid>.jsonl \\
        --project-tag my-project-build

    # Find your session files:
    ls ~/.claude/projects/*/*.jsonl
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pricing import calculate_cost, PRICING_TABLE
from src.tracker import get_events, log_usage
from src.usage_schema import UsageEvent

# Claude Code's own model ids, mapped to this project's pricing table keys.
# Extend as new models appear in a transcript; anything not listed here (or
# not in PRICING_TABLE) is skipped and reported, not guessed at.
MODEL_MAP = {
    "claude-sonnet-5": "claude-sonnet-5",
    "claude-opus-5": "claude-opus-5",
    "claude-haiku-4-5-20251001": "claude-haiku-4-5",
    "claude-opus-4-8": "claude-opus-4-8",
}


def _preview(msg: dict) -> str:
    for block in msg.get("content", []) or []:
        if block.get("type") == "text" and block.get("text", "").strip():
            return UsageEvent.make_preview(block["text"].strip())
        if block.get("type") == "tool_use":
            return UsageEvent.make_preview(f"[tool call: {block.get('name', '?')}]")
    return "[no text content]"


def _checkpoint_path(transcript_path: str) -> Path:
    """Where we remember which message ids from this transcript are already
    logged, so re-running on a transcript that's grown (the session
    continued) only imports the new turns instead of re-logging everything.
    """
    p = Path(transcript_path)
    return p.parent / f".{p.stem}.imported.json"


def load(path: str, project_tag: str, db_path: str | None = None) -> dict:
    checkpoint = _checkpoint_path(path)
    seen_ids = set(json.loads(checkpoint.read_text())) if checkpoint.exists() else set()
    already_imported = len(seen_ids)
    logged = 0
    skipped_no_price = 0
    skipped_synthetic = 0
    total_cost = 0.0
    total_tokens = 0

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if d.get("type") != "assistant":
                continue

            msg = d.get("message", {})
            mid = msg.get("id")
            if not mid or mid in seen_ids:
                continue
            seen_ids.add(mid)

            usage = msg.get("usage")
            model_raw = msg.get("model")
            if not usage or not model_raw:
                continue
            if model_raw not in MODEL_MAP:
                skipped_synthetic += 1
                continue

            model = MODEL_MAP[model_raw]
            if model not in PRICING_TABLE:
                skipped_no_price += 1
                continue

            cache_read = usage.get("cache_read_input_tokens", 0) or 0
            cache_write = usage.get("cache_creation_input_tokens", 0) or 0
            raw_input = usage.get("input_tokens", 0) or 0
            output = usage.get("output_tokens", 0) or 0
            full_input = raw_input + cache_read + cache_write

            cost = calculate_cost(model, full_input, output, cache_read, cache_write)

            event = UsageEvent(
                model=model,
                provider="anthropic",
                project_tag=project_tag,
                input_tokens=full_input,
                output_tokens=output,
                cached_input_tokens=cache_read,
                cache_write_tokens=cache_write,
                cost_usd=cost,
                latency_ms=0.0,  # not recorded per-turn in the transcript
                prompt_preview=_preview(msg),
                success=True,
                source="manual",
                timestamp=d.get("timestamp"),
            )
            log_usage(event, db_path=db_path)
            logged += 1
            total_cost += cost
            total_tokens += full_input + output

    checkpoint.write_text(json.dumps(sorted(seen_ids)))

    return {
        "logged": logged,
        "already_imported": already_imported,
        "skipped_no_price": skipped_no_price,
        "skipped_unknown_model": skipped_synthetic,
        "total_cost": round(total_cost, 6),
        "total_tokens": total_tokens,
    }


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--session", required=True, help="Path to a Claude Code session .jsonl file")
    parser.add_argument("--project-tag", required=True, help="Project tag to log these events under")
    parser.add_argument("--db-path", default=None, help="Override the SQLite DB path")
    args = parser.parse_args()

    path = Path(args.session).expanduser()
    if not path.exists():
        parser.error(f"session file not found: {path}")

    stats = load(str(path), args.project_tag, db_path=args.db_path)
    print(f"logged={stats['logged']} new turns  tokens={stats['total_tokens']:,}  cost=${stats['total_cost']:.4f}"
          + (f"  ({stats['already_imported']} already imported, skipped)" if stats["already_imported"] else ""))
    if stats["skipped_no_price"]:
        print(f"  ({stats['skipped_no_price']} turns skipped: model not in PRICING_TABLE)")
    if stats["skipped_unknown_model"]:
        print(f"  ({stats['skipped_unknown_model']} turns skipped: unrecognized/synthetic model id)")


if __name__ == "__main__":
    _main()
