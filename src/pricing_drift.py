"""
Reconcile this project's hardcoded PRICING_TABLE against a public price map.

The README's first stated known gap is that pricing is a snapshot: rates were
verified by hand and there is no automatic refresh, so a vendor price change
silently makes every historical figure wrong. This closes that gap without
surrendering the table.

**The public map is a second opinion, not a source of truth.** LiteLLM
publishes a community-maintained `model_prices_and_context_window.json` used by
its router for cost-based routing. It is broad and current, but it does not
model the two rules this project exists to get right, the GPT-5.6 long-context
surcharge, and the 5-minute vs 1-hour cache-write TTL split that was
understating this repo's own build cost by 17%. So this never overwrites a
local rate. It reports disagreement and leaves the decision to a human, which
is the only safe direction: a bad auto-update would silently re-price years of
history.

    python -m src.pricing_drift              # compare against the cached map
    python -m src.pricing_drift --refresh    # re-download first

Offline by default, and offline-safe: with no cached map and no network it
reports that it could not check rather than failing the caller.
"""

import json
from pathlib import Path

from src.pricing import PRICING_TABLE

PRICE_MAP_URL = (
    "https://raw.githubusercontent.com/BerriAI/litellm/main/"
    "model_prices_and_context_window.json"
)

CACHE_PATH = Path(__file__).resolve().parent.parent / "data" / "litellm_price_map.json"

# Our model id -> the id used in the public map. Only entries that genuinely
# refer to the same model belong here; a wrong mapping produces a confident
# false drift report, which is worse than no report.
MODEL_ALIASES: dict[str, str] = {
    "claude-opus-4-8": "claude-opus-4-8",
    "claude-sonnet-5": "claude-sonnet-5",
    "claude-haiku-4-5": "claude-haiku-4-5",
    "claude-opus-5": "claude-opus-5",
    "gemini-flash-latest": "gemini/gemini-flash-latest",
    "gemini-pro-latest": "gemini/gemini-pro-latest",
    "gemini-flash-lite-latest": "gemini/gemini-flash-lite-latest",
}

# Rates below this differ by less than a rounding artifact in the source data.
_TOLERANCE = 1e-9


def load_price_map(refresh: bool = False) -> dict | None:
    """Return the public price map, or None if it is unavailable.

    Returning None rather than raising is deliberate: a drift check is a
    reporting nicety, and a network blip must never break a cost calculation.
    """
    if refresh or not CACHE_PATH.exists():
        try:
            import httpx

            resp = httpx.get(PRICE_MAP_URL, timeout=20.0, follow_redirects=True)
            resp.raise_for_status()
            CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
            CACHE_PATH.write_text(resp.text)
        except Exception:
            if not CACHE_PATH.exists():
                return None
    try:
        return json.loads(CACHE_PATH.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _per_1k(value: float | None) -> float | None:
    """The map quotes per-token costs; this project quotes per 1,000 tokens."""
    return None if value is None else value * 1000


def check_drift(refresh: bool = False) -> dict:
    """Compare every mapped model's rates against the public map.

    Returns a report rather than printing, so the MCP server and the dashboard
    can surface the same finding the CLI does.
    """
    price_map = load_price_map(refresh=refresh)
    if price_map is None:
        return {
            "checked": False,
            "reason": "public price map unavailable (no cache, and the download failed)",
            "drifted": [],
            "unmapped": sorted(PRICING_TABLE),
        }

    drifted: list[dict] = []
    agreed: list[str] = []
    unmapped: list[str] = []

    for model, local in PRICING_TABLE.items():
        remote_key = MODEL_ALIASES.get(model)
        remote = price_map.get(remote_key) if remote_key else None
        if not remote:
            unmapped.append(model)
            continue

        comparisons = {
            "input": _per_1k(remote.get("input_cost_per_token")),
            "cached_input": _per_1k(remote.get("cache_read_input_token_cost")),
            "output": _per_1k(remote.get("output_cost_per_token")),
        }
        deltas = {
            field: {"ours": local[field], "theirs": theirs}
            for field, theirs in comparisons.items()
            if theirs is not None and abs(local[field] - theirs) > _TOLERANCE
        }
        if deltas:
            drifted.append({"model": model, "public_id": remote_key, "fields": deltas})
        else:
            agreed.append(model)

    return {
        "checked": True,
        "source": PRICE_MAP_URL,
        "models_compared": len(agreed) + len(drifted),
        "agreed": sorted(agreed),
        "drifted": drifted,
        # Not an error: this project prices models the public map has never
        # heard of, and the map carries thousands we do not use.
        "unmapped": sorted(unmapped),
    }


def context_windows(refresh: bool = False) -> dict[str, int]:
    """Max input tokens per model, from the public map.

    This project's own table has no context-window column, it prices calls, it
    doesn't size them. The router uses this for its pre-call check, so a prompt
    that cannot fit is never dispatched (and, on GPT-5.6, so traffic can be
    steered away from the 272K long-context surcharge before it is incurred).
    """
    price_map = load_price_map(refresh=refresh) or {}
    out: dict[str, int] = {}
    for model, remote_key in MODEL_ALIASES.items():
        remote = price_map.get(remote_key) or {}
        limit = remote.get("max_input_tokens")
        if isinstance(limit, int) and limit > 0:
            out[model] = limit
    return out


def _main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Check PRICING_TABLE against the public price map.")
    ap.add_argument("--refresh", action="store_true", help="Re-download the price map first")
    args = ap.parse_args()

    report = check_drift(refresh=args.refresh)
    if not report["checked"]:
        print(f"could not check: {report['reason']}")
        raise SystemExit(0)

    print(f"compared {report['models_compared']} model(s) against {report['source']}\n")
    if not report["drifted"]:
        print("  no drift: every mapped rate agrees with the public map")
    for row in report["drifted"]:
        print(f"  DRIFT  {row['model']}  (public id: {row['public_id']})")
        for field, d in row["fields"].items():
            direction = "under" if d["ours"] < d["theirs"] else "over"
            print(f"           {field:<13} ours={d['ours']:.6f}  theirs={d['theirs']:.6f}  ({direction}-priced here)")
    if report["unmapped"]:
        print(f"\n  {len(report['unmapped'])} model(s) not in the public map: {', '.join(report['unmapped'])}")


if __name__ == "__main__":
    _main()
