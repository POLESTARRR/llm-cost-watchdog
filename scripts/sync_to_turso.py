#!/usr/bin/env python3
"""Push the local ledger straight to the deployed database.

    python scripts/sync_to_turso.py            # sync
    python scripts/sync_to_turso.py --dry-run  # show what would change

The other route to the live site is `import_all_projects.py --remote-url`, which
posts to the deployment's /import endpoint and needs WATCHDOG_IMPORT_KEY on both
ends to agree. On this machine they do not, so that path returns 401 and the
published site silently stops tracking new work.

This removes the middleman. Turso credentials already sit in .env.render, and
the deployed app reads the same database, so writing to it directly is the same
operation with one fewer secret to keep in sync. It also means the sync does not
depend on the web service being awake, which on a free tier it usually is not.

**Additive, never-delete.** Only subscription rows are touched. Live rows on the
remote are gateway traffic the local database never saw, and replacing them would
delete real history to publish a copy of something else. Each subscription turn is
identified by a content fingerprint (model + provider + project_tag + timestamp +
token counts + prompt_hash), and only genuinely-new turns are inserted. The
remote may have rows that no local transcript file still exists for (deleted
projects, other machines' imports); those are never removed. This script is
idempotent and safe to re-run.
"""

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import dotenv_values  # noqa: E402

from src.tracker import get_events_for_period  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
COLS = ("id", "timestamp", "model", "provider", "project_tag", "input_tokens",
        "output_tokens", "cached_input_tokens", "cache_write_tokens",
        "cache_write_1h_tokens", "cost_usd", "latency_ms", "ttft_ms",
        "prompt_preview", "prompt_hash", "service_tier", "success", "error", "source")
BATCH = 150


def credentials() -> tuple[str, str]:
    cfg = {**dotenv_values(ROOT / ".env"), **dotenv_values(ROOT / ".env.render")}
    url, token = cfg.get("TURSO_DATABASE_URL"), cfg.get("TURSO_AUTH_TOKEN")
    if not url or not token:
        sys.exit("No TURSO_DATABASE_URL / TURSO_AUTH_TOKEN in .env or .env.render")
    return url.replace("libsql://", "https://").rstrip("/"), token


def run(url: str, token: str, stmts: list, timeout: int = 180) -> list:
    reqs = [{"type": "execute", "stmt": s if isinstance(s, dict) else {"sql": s}} for s in stmts]
    reqs.append({"type": "close"})
    body = json.dumps({"requests": reqs}).encode()
    req = urllib.request.Request(
        url + "/v2/pipeline", body,
        {"Authorization": f"Bearer {token}", "Content-Type": "application/json"})
    res = json.load(urllib.request.urlopen(req, timeout=timeout))
    out = []
    for i, item in enumerate(res["results"][:-1]):
        if item.get("type") == "error":
            raise SystemExit(f"statement {i} failed: {item}")
        r = item["response"]["result"]
        rows = [[c.get("value") for c in row] for row in r["rows"]]
        out.append(rows if rows else r.get("affected_row_count"))
    return out


def arg(v):
    """Turso's wire format needs an explicit type tag per value."""
    if v is None:
        return {"type": "null"}
    if isinstance(v, bool):
        return {"type": "integer", "value": str(int(v))}
    if isinstance(v, int):
        return {"type": "integer", "value": str(v)}
    if isinstance(v, float):
        return {"type": "float", "value": v}
    return {"type": "text", "value": str(v)}


FINGERPRINT_COLS = ("model", "provider", "project_tag", "timestamp",
                    "input_tokens", "output_tokens", "prompt_hash")


def _fingerprint(event) -> str:
    """Stable content fingerprint for dedup — identifies a turn by its
    transcript-authoritative fields. Random UUIDs are not usable because
    the nightly --rebuild mints fresh ones each run."""
    parts = []
    for c in FINGERPRINT_COLS:
        v = getattr(event, c, None)
        parts.append(str(v) if v is not None else "")
    return "|".join(parts)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="(unused, kept for run-sync.sh compat)")
    args = ap.parse_args()

    url, token = credentials()
    events = [e for e in get_events_for_period("all_time", source="subscription") if e.success]
    local_total = sum(e.cost_usd for e in events)

    if not events:
        print("local ledger has no subscription rows; nothing to sync")
        return 0

    try:
        before = run(url, token,
                     ["SELECT source, COUNT(*), ROUND(SUM(cost_usd),2) "
                      "FROM usage_events GROUP BY 1"])[0]
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        print(f"cannot reach the remote database: {exc}")
        return 1

    # If the table is empty, run() returns an int (affected_row_count), not a list.
    # Treat that as "no rows found".
    if not isinstance(before, list):
        before = []

    remote_sub = next((r for r in before if r[0] == "subscription"), None)
    remote_n = int(remote_sub[1]) if remote_sub else 0
    print(f"remote: {remote_n:,} subscription rows"
          f"{f' (${remote_sub[2]})' if remote_sub else ''}")
    print(f"local : {len(events):,} rows (${local_total:,.2f})")

    # Fetch all remote fingerprints (content columns only — lightweight).
    # Values come back as strings from Turso's HTTP API.
    remote_fps = set()
    try:
        raw = run(url, token,
                  [f"SELECT {','.join(FINGERPRINT_COLS)} "
                   f"FROM usage_events WHERE source='subscription'"])[0]
        if isinstance(raw, list):
            for row in raw:
                # Turso returns SQL NULL as None; normalize identically to the
                # local "" so a null prompt_hash does not look like a new turn.
                remote_fps.add("|".join("" if v is None else str(v) for v in row))
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        print(f"cannot read remote fingerprints: {exc}")
        return 1

    # Find genuinely new local turns.
    new_events = [e for e in events if _fingerprint(e) not in remote_fps]

    if not new_events:
        print("all local turns already on remote; nothing to insert")
        return 0

    new_cost = sum(e.cost_usd for e in new_events)
    print(f"new turns to insert: {len(new_events):,} (${new_cost:,.2f})")

    if args.dry_run:
        return 0

    # Insert only the new rows, never delete anything.
    head = f"INSERT INTO usage_events ({','.join(COLS)}) VALUES "
    placeholders = "(" + ",".join("?" * len(COLS)) + ")"
    for i in range(0, len(new_events), BATCH):
        chunk = new_events[i:i + BATCH]
        params = []
        for e in chunk:
            d = e.model_dump()
            for c in COLS:
                v = d.get(c)
                params.append(arg(1 if v else 0) if c == "success" else arg(v))
        run(url, token, [{"sql": head + ",".join([placeholders] * len(chunk)), "args": params}])

    # Verify the insert landed.
    after = run(url, token, ["SELECT COUNT(*), ROUND(SUM(cost_usd),2) "
                             "FROM usage_events WHERE source='subscription'"])[0]
    n, cost = int(after[0][0]), float(after[0][1])
    if n < remote_n:
        print(f"VERIFY FAILED: remote dropped from {remote_n:,} to {n:,} rows")
        return 1
    print(f"synced: inserted {len(new_events):,} turns; "
          f"remote now {n:,} subscription rows (${cost:,.2f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
