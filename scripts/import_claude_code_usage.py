#!/usr/bin/env python3
"""
Import real Claude Code build-time usage into the watchdog.

Claude Code writes a local JSONL transcript per session (default location:
`~/.claude/projects/<encoded-cwd>/<session-id>.jsonl`), and every assistant
turn in it carries the real `usage` block from the actual Anthropic API
response it received, input/output tokens, cache-read tokens, cache-write
tokens, the works. That is the authoritative source for "what did it cost,
in real tokens, to build this project with Claude Code", a number this
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

By default this writes straight to the local SQLite file. Pass --remote-url
(plus --import-key) to push to a deployed dashboard's /import endpoint
instead, e.g. a live site updates the moment you run this, no redeploy:

    python scripts/import_claude_code_usage.py \\
        --session ~/.claude/projects/-path-to-project/<uuid>.jsonl \\
        --project-tag my-project-build \\
        --remote-url https://your-dashboard.example.com \\
        --import-key "$WATCHDOG_IMPORT_KEY"
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pricing import calculate_cost, PRICING_TABLE
from src.tracker import log_usage
from src.usage_schema import UsageEvent

# Render's free tier plus a real Turso sync per batch is measurably slower
# than local SQLite. 200 events per POST regularly exceeded a 30s client
# timeout there, well after the batch was fully staged, so the whole import
# looked like a hard failure a few hundred turns in even though most of the
# work had already succeeded. Smaller batches, and a generous timeout on top,
# trade a few extra round trips for actually finishing on a slow host.
_REMOTE_BATCH_SIZE = 40
_REMOTE_TIMEOUT_SECONDS = 90.0

# Claude Code's own model ids, mapped to this project's pricing table keys.
# Extend as new models appear in a transcript; anything not listed here (or
# not in PRICING_TABLE) is skipped and reported, not guessed at.
MODEL_MAP = {
    "claude-sonnet-5": "claude-sonnet-5",
    "claude-opus-5": "claude-opus-5",
    "claude-haiku-4-5-20251001": "claude-haiku-4-5",
    "claude-opus-4-8": "claude-opus-4-8",
}


# A derived latency above this is not a measurement of anything, it means the
# session was idle (laptop closed, user away) between the prompt and the reply.
# Dropped to 0.0 ("not measured") rather than recorded, because a 4-hour
# "latency" would wreck the anomaly detector's rolling averages.
_MAX_PLAUSIBLE_LATENCY_S = 600.0


def _preview(msg: dict) -> str:
    for block in msg.get("content", []) or []:
        if block.get("type") == "text" and block.get("text", "").strip():
            return UsageEvent.make_preview(block["text"].strip())
        if block.get("type") == "tool_use":
            return UsageEvent.make_preview(f"[tool call: {block.get('name', '?')}]")
    return "[no text content]"


def _parse_ts(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return None


def _derive_latency_ms(prev_ts: str | None, turn_ts: str | None) -> float:
    """Wall-clock between the prompt being sent and the reply being recorded.

    The transcript has no explicit per-turn latency field, but every record is
    timestamped. The gap from the preceding record (the user message, or the
    tool result that unblocked the model) to this assistant turn is the time
    the request actually took, it excludes user think time, because a user
    message is stamped when it is *sent*, not when it was started.

    This is a derived figure and labelled as one. It is an upper bound: it
    includes client-side overhead alongside real API latency. Returns 0.0,
    the project's existing "not measured" value, when it cannot be derived
    or is implausibly large.
    """
    a, b = _parse_ts(prev_ts), _parse_ts(turn_ts)
    if a is None or b is None:
        return 0.0
    seconds = (b - a).total_seconds()
    if seconds <= 0 or seconds > _MAX_PLAUSIBLE_LATENCY_S:
        return 0.0
    return round(seconds * 1000, 2)


def _prompt_text(record: dict) -> str:
    """Flatten a user/tool-result record into the text the model was given.

    Used only to hash, never stored. Lets duplicate detection compare whole
    prompts instead of an 80-character preview, which could not tell two calls
    apart when they shared a long fixed instruction template.
    """
    msg = record.get("message") or {}
    content = msg.get("content")
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for block in content or []:
        if not isinstance(block, dict):
            parts.append(str(block))
            continue
        if block.get("type") == "text":
            parts.append(block.get("text", ""))
        elif block.get("type") == "tool_result":
            inner = block.get("content")
            parts.append(inner if isinstance(inner, str) else json.dumps(inner, sort_keys=True))
    return "\n".join(p for p in parts if p)


def _checkpoint_path(transcript_path: str, project_tag: str, sink: str = "local") -> Path:
    """Where we remember which message ids from this transcript are already
    logged, so re-running on a transcript that's grown (the session
    continued) only imports the new turns instead of re-logging everything.

    Keyed by project tag as well as transcript: one working directory can hold
    several projects, and a single shared checkpoint would let the first
    project's import mark the other projects' turns as already-done.

    Also keyed by `sink` for anything other than local ("local" or a slug of
    the remote URL). The local SQLite file and a deployed dashboard's database
    are genuinely different destinations with independent history, they do not
    stay in sync with each other. A single shared checkpoint would mean the
    first import (say, local) marks every message id as done, and a later
    --remote-url push against the same transcript would see them as
    already-imported and send nothing, even though the remote database has
    never seen them.

    `sink="local"` keeps the pre-existing filename (no suffix) so checkpoints
    already on disk from local imports remain valid, only a non-local sink
    gets a distinct suffix, since that is the new case this exists to cover.
    """
    p = Path(transcript_path)
    suffix = "" if sink == "local" else f".{sink}"
    return p.parent / f".{p.stem}.{project_tag}{suffix}.imported.json"


def _sink_slug(remote_url: str | None) -> str:
    """Filesystem-safe identifier for a push destination, for the checkpoint name."""
    if not remote_url:
        return "local"
    import re

    return "remote-" + re.sub(r"[^a-zA-Z0-9]+", "-", remote_url.strip()).strip("-").lower()


def _push_remote(
    events: list[dict],
    message_ids: list[str],
    remote_url: str,
    import_key: str,
    checkpoint: Path,
    seen_ids: set[str],
) -> dict:
    """POST events to a deployed dashboard's /import endpoint, in batches.

    `events[i]` and `message_ids[i]` are the same turn, the caller appends
    both in lockstep, so a batch slice of one is the matching slice of the
    other. The checkpoint is persisted after *each* batch the server accepts,
    not once at the end of the whole file. Only checkpointing at the end meant
    a failure on, say, batch 6 of 10 discarded the fact that batches 1 to 5
    had already landed durably on the server, so the next run resent them and
    `/import` mints a fresh UUID per row, no dedup, so that duplicated rows
    that were never actually missing.

    Raises on any non-2xx response rather than silently dropping a batch, a
    partial import that looks successful is worse than a loud failure. A
    timeout is retried a few times with backoff before giving up: on a slow
    host (free-tier compute plus a real Turso sync per batch) a single slow
    request is common and does not mean the batch failed, only that it hasn't
    finished yet. Retrying is safe for the same reason incremental
    checkpointing is needed at all: a request this client sees as timed out
    either never reached the server or the server is still working on it,
    either way nothing has been double counted yet.
    """
    logged = total_cost = skipped = 0
    total_batches = (len(events) + _REMOTE_BATCH_SIZE - 1) // _REMOTE_BATCH_SIZE
    with httpx.Client(base_url=remote_url.rstrip("/"), timeout=_REMOTE_TIMEOUT_SECONDS) as client:
        for batch_num, i in enumerate(range(0, len(events), _REMOTE_BATCH_SIZE), start=1):
            batch = events[i:i + _REMOTE_BATCH_SIZE]
            batch_ids = message_ids[i:i + _REMOTE_BATCH_SIZE]
            last_exc: Exception | None = None
            for attempt in range(3):
                try:
                    resp = client.post(
                        "/import",
                        json={"events": batch},
                        headers={"X-Watchdog-Import-Key": import_key},
                    )
                    resp.raise_for_status()
                    body = resp.json()
                    logged += body["logged"]
                    skipped += body["skipped_unpriced"]
                    total_cost += body["total_cost_usd"]
                    seen_ids.update(batch_ids)
                    checkpoint.write_text(json.dumps(sorted(seen_ids)))
                    break
                except httpx.TimeoutException as exc:
                    last_exc = exc
                    print(f"    batch {batch_num}/{total_batches} timed out "
                          f"(attempt {attempt + 1}/3), retrying...")
                    time.sleep(2 * (attempt + 1))
            else:
                raise last_exc
    return {"logged": logged, "skipped_no_price": skipped, "total_cost": total_cost}


def load(
    path: str,
    project_tag: str,
    db_path: str | None = None,
    remote_url: str | None = None,
    import_key: str | None = None,
    source: str = "subscription",
    only_message_ids: set[str] | None = None,
) -> dict:
    """Import one transcript's assistant turns.

    `only_message_ids`, when given, restricts the import to those API message
    ids, used when one transcript interleaves several projects and each
    project's turns have been attributed separately upstream.
    """
    checkpoint = _checkpoint_path(path, project_tag, sink=_sink_slug(remote_url))
    seen_ids = set(json.loads(checkpoint.read_text())) if checkpoint.exists() else set()
    already_imported = len(seen_ids)
    new_ids: list[str] = []
    pending: list[dict] = []  # for --remote-url; local writes happen inline below
    pending_ids: list[str] = []  # message id for pending[i], same index, see _push_remote
    skipped_no_price = 0
    skipped_synthetic = 0
    logged = 0
    total_cost = 0.0
    total_tokens = 0

    # The record immediately before an assistant turn is the prompt that
    # produced it, used for the derived latency and the prompt hash.
    prev_ts: str | None = None
    prev_hash: str | None = None
    latency_derived = 0

    with open(path, errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue

            if d.get("type") != "assistant":
                # Remember the prompt side so the next assistant turn can be
                # timed and hashed against it.
                if d.get("type") == "user":
                    prev_ts = d.get("timestamp") or prev_ts
                    prev_hash = UsageEvent.make_hash(_prompt_text(d))
                continue

            msg = d.get("message", {})
            mid = msg.get("id")
            turn_ts = d.get("timestamp")
            if not mid or mid in seen_ids or mid in new_ids:
                prev_ts = turn_ts or prev_ts
                continue
            if only_message_ids is not None and mid not in only_message_ids:
                prev_ts = turn_ts or prev_ts
                continue

            usage = msg.get("usage")
            model_raw = msg.get("model")
            if not usage or not model_raw:
                prev_ts = turn_ts or prev_ts
                continue
            if model_raw not in MODEL_MAP:
                skipped_synthetic += 1
                prev_ts = turn_ts or prev_ts
                continue

            model = MODEL_MAP[model_raw]
            if model not in PRICING_TABLE:
                skipped_no_price += 1
                prev_ts = turn_ts or prev_ts
                continue

            cache_read = usage.get("cache_read_input_tokens", 0) or 0
            cache_write = usage.get("cache_creation_input_tokens", 0) or 0
            raw_input = usage.get("input_tokens", 0) or 0
            output = usage.get("output_tokens", 0) or 0
            full_input = raw_input + cache_read + cache_write

            # The TTL split is what makes the cache-write premium correct:
            # 1-hour writes bill at 2.0x input, 5-minute at 1.25x. Claude Code
            # uses 1-hour caching almost exclusively, so pricing the whole
            # figure at the 5-minute rate understated real cost by ~14%.
            creation = usage.get("cache_creation") or {}
            cache_write_1h = creation.get("ephemeral_1h_input_tokens", 0) or 0
            # Trust the flat total as authoritative; the split is a subset of it.
            cache_write_1h = min(cache_write_1h, cache_write)
            tier = usage.get("service_tier") or "standard"

            latency_ms = _derive_latency_ms(prev_ts, turn_ts)
            if latency_ms:
                latency_derived += 1

            cost = calculate_cost(
                model, full_input, output, cache_read, cache_write,
                cache_write_1h_tokens=cache_write_1h, service_tier=tier,
            )
            total_cost += cost
            total_tokens += full_input + output
            new_ids.append(mid)

            row = {
                "model": model, "provider": "anthropic", "project_tag": project_tag,
                "input_tokens": full_input, "output_tokens": output,
                "cached_input_tokens": cache_read, "cache_write_tokens": cache_write,
                "cache_write_1h_tokens": cache_write_1h, "service_tier": tier,
                "latency_ms": latency_ms, "prompt_preview": _preview(msg),
                "prompt_hash": prev_hash, "success": True, "source": source,
                "timestamp": turn_ts,
            }

            if remote_url:
                pending.append(row)
                pending_ids.append(mid)
            else:
                log_usage(UsageEvent(cost_usd=cost, **row), db_path=db_path)
            logged += 1
            prev_ts = turn_ts or prev_ts

    # Only commit the checkpoint after a write actually succeeds, a failed
    # remote push must leave these turns eligible for retry on the next run,
    # not silently marked as already-imported. For the remote path that
    # happens incrementally per batch inside _push_remote, since seen_ids is
    # the same set object passed in below, so it is already up to date here
    # even if a later batch in this same file then fails.
    if remote_url and pending:
        _push_remote(pending, pending_ids, remote_url, import_key, checkpoint, seen_ids)
    else:
        seen_ids.update(new_ids)
        checkpoint.write_text(json.dumps(sorted(seen_ids)))

    return {
        "logged": logged,
        "already_imported": already_imported,
        "skipped_no_price": skipped_no_price,
        "skipped_unknown_model": skipped_synthetic,
        "total_cost": round(total_cost, 6),
        "total_tokens": total_tokens,
        "latency_derived": latency_derived,
    }


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--session", required=True, help="Path to a Claude Code session .jsonl file")
    parser.add_argument("--project-tag", required=True, help="Project tag to log these events under")
    parser.add_argument("--db-path", default=None, help="Override the local SQLite DB path (ignored with --remote-url)")
    parser.add_argument("--remote-url", default=None,
                         help="Push to a deployed dashboard's /import endpoint instead of writing locally, "
                              "e.g. https://your-dashboard.example.com")
    parser.add_argument("--import-key", default=None,
                         help="Value of WATCHDOG_IMPORT_KEY on the remote deployment. "
                              "Defaults to the WATCHDOG_IMPORT_KEY env var if not given.")
    parser.add_argument("--source", default="subscription",
                         choices=["subscription", "manual"],
                         help="Provenance for the imported rows. Default 'subscription': "
                              "Claude Code usage on a Pro/Max plan is real tokens covered "
                              "by a flat fee, so it is list-price value rather than metered "
                              "spend. Use 'manual' only if these turns were billed per token "
                              "against an API key.")
    args = parser.parse_args()

    path = Path(args.session).expanduser()
    if not path.exists():
        parser.error(f"session file not found: {path}")

    import_key = args.import_key or os.environ.get("WATCHDOG_IMPORT_KEY")
    if args.remote_url and not import_key:
        parser.error("--remote-url requires --import-key (or a WATCHDOG_IMPORT_KEY env var)")

    stats = load(str(path), args.project_tag, db_path=args.db_path,
                 remote_url=args.remote_url, import_key=import_key, source=args.source)
    sink = f"remote: {args.remote_url}" if args.remote_url else "local DB"
    print(f"[{sink}] logged={stats['logged']} new turns  tokens={stats['total_tokens']:,}  cost=${stats['total_cost']:.4f}"
          + (f"  ({stats['already_imported']} already imported, skipped)" if stats["already_imported"] else ""))
    if stats["latency_derived"]:
        print(f"  ({stats['latency_derived']} turns carry a latency derived from transcript timestamps)")
    if stats["skipped_no_price"]:
        print(f"  ({stats['skipped_no_price']} turns skipped: model not in PRICING_TABLE)")
    if stats["skipped_unknown_model"]:
        print(f"  ({stats['skipped_unknown_model']} turns skipped: unrecognized/synthetic model id)")


if __name__ == "__main__":
    _main()
