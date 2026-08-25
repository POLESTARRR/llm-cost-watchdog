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

**Only subscription rows are touched.** Live rows on the remote are gateway
traffic the local database never saw, and replacing them would delete real
history to publish a copy of something else. Safe to re-run: the local ledger is
a superset and is itself rebuildable from ~/.claude/projects at any time.
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="publish even if it would shrink the remote substantially")
    args = ap.parse_args()

    url, token = credentials()
    events = [e for e in get_events_for_period("all_time", source="subscription") if e.success]
    local_total = sum(e.cost_usd for e in events)

    if not events:
        # Publishing an empty ledger over a populated one would be the single
        # most destructive thing this script could do, and an empty local
        # database is far more likely to mean "not imported yet" than "the work
        # was deleted".
        print("local ledger has no subscription rows; refusing to empty the remote")
        return 1

    try:
        before = run(url, token,
                     ["SELECT source, COUNT(*), ROUND(SUM(cost_usd),2) "
                      "FROM usage_events GROUP BY 1"])[0]
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        print(f"cannot reach the remote database: {exc}")
        return 1

    remote_sub = next((r for r in before if r[0] == "subscription"), None)
    remote_n = int(remote_sub[1]) if remote_sub else 0
    print(f"remote: {remote_n:,} subscription rows"
          f"{f' (${remote_sub[2]})' if remote_sub else ''}")
    print(f"local : {len(events):,} rows (${local_total:,.2f})")

    if remote_n == len(events):
        print("already in sync")
        return 0

    # Refuse a collapse. The empty-ledger check above is not enough: a second
    # clone of this repo, with its own fresh database, imported 33 rows and
    # published them over 9,033, because the importer's checkpoints live beside
    # the transcripts and are shared between copies while the databases are not.
    # The clone was told everything had already been imported, imported almost
    # nothing, and passed a guard that only asked whether it had *something*.
    #
    # Any large shrink is far more likely to be a misconfigured copy than a
    # genuine deletion, so it stops and makes a human look.
    if remote_n and len(events) < remote_n * 0.9 and not args.force:
        print(f"\nREFUSING: local has {len(events):,} rows against the remote's {remote_n:,}.")
        print("A drop this large usually means this copy's database is incomplete,")
        print("not that the work was deleted. Nothing was changed.")
        print("\nIf the shrink is real and intended, pass --force.")
        return 1
    if args.dry_run:
        print(f"would replace {remote_n:,} remote rows with {len(events):,} local ones")
        return 0

    run(url, token, ["DELETE FROM usage_events WHERE source='subscription'"])

    head = f"INSERT INTO usage_events ({','.join(COLS)}) VALUES "
    placeholders = "(" + ",".join("?" * len(COLS)) + ")"
    for i in range(0, len(events), BATCH):
        chunk = events[i:i + BATCH]
        params = []
        for e in chunk:
            d = e.model_dump()
            for c in COLS:
                v = d.get(c)
                params.append(arg(1 if v else 0) if c == "success" else arg(v))
        run(url, token, [{"sql": head + ",".join([placeholders] * len(chunk)), "args": params}])

    after = run(url, token, ["SELECT COUNT(*), ROUND(SUM(cost_usd),2) "
                             "FROM usage_events WHERE source='subscription'"])[0]
    n, cost = int(after[0][0]), float(after[0][1])
    if n != len(events) or abs(cost - round(local_total, 2)) > 0.05:
        print(f"VERIFY FAILED: remote has {n:,} rows / ${cost}, local {len(events):,} / "
              f"${local_total:,.2f}")
        return 1
    print(f"synced and verified: {n:,} rows, ${cost:,.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
