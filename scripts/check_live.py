#!/usr/bin/env python3
"""Is the published site still telling the truth? Runs anywhere, needs no browser.

    python scripts/check_live.py                          # the deployed site
    python scripts/check_live.py http://127.0.0.1:8010    # a local one

scripts/check_page_renders.sh is the stronger check and needs Chrome, so it runs
on a laptop and not in CI. This is the portable half: it asks the endpoints the
page depends on whether they still answer, and whether their answers are
internally consistent.

The point is rot, not deployment. Once nobody is actively working on this, the
ways it breaks are quiet ones: a host sleeps and never wakes, a database
credential expires, a shipped artifact stops being copied into an image, a
number drifts away from the number beside it. Every one of those leaves a page
that still returns HTTP 200 and says something false, which is the only failure
mode this project actually cares about.

Exits non-zero on the first inconsistency, so a scheduled job can page you.
"""

import json
import sys
import urllib.error
import urllib.request

DEFAULT = "https://llmcostwatchdog.onrender.com"
TIMEOUT = 90          # a sleeping free-tier host takes a while to wake


def get(base: str, path: str) -> dict:
    with urllib.request.urlopen(base + path, timeout=TIMEOUT) as r:
        return json.load(r)


def main() -> int:
    base = (sys.argv[1] if len(sys.argv) > 1 else DEFAULT).rstrip("/")
    print(f"checking {base}\n")
    problems: list[str] = []

    def check(label: str, ok: bool, detail: str = "") -> None:
        print(f"  {'ok  ' if ok else 'FAIL'} {label:38} {detail}")
        if not ok:
            problems.append(label)

    # 1. The host is awake at all.
    try:
        get(base, "/health")
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        print(f"  FAIL host unreachable: {exc}")
        return 1
    check("host responds", True)

    # 2. The study still has its data. A dropped database credential shows up
    #    here as zeroes rather than as an error.
    try:
        f = get(base, "/findings")
    except Exception as exc:
        check("findings endpoint", False, str(exc))
        return 1

    h = f.get("headline", {})
    check("ledger has rows", h.get("turns", 0) > 0, f"{h.get('turns', 0):,} turns")
    check("ledger has cost", h.get("cost_usd", 0) > 0, f"${h.get('cost_usd', 0):,.2f}")
    check("projects present", h.get("projects", 0) > 0, f"{h.get('projects', 0)} projects")

    # 3. The cost split must still account for everything. Three components that
    #    stop summing to 100 means the pricing model and the ledger have drifted
    #    apart, which is exactly the class of quiet lie this page exists to avoid.
    rw = f.get("read_vs_write", {})
    total = (rw.get("read_percent", 0) + rw.get("cache_ttl_premium_percent", 0)
             + rw.get("write_percent", 0))
    check("cost split sums to 100%", abs(total - 100) < 1.0, f"{total:.1f}%")

    # 4. List price and what was actually paid must both be present. The headline
    #    is only honest as a pair.
    paid = f.get("actually_paid", {})
    check("actually-paid figure present",
          "subscription_cost_usd" in paid or paid.get("calls", 0) == 0,
          f"${paid.get('subscription_cost_usd', 0):,.2f} paid")

    # 5. Shipped artifacts survived the build. Both have vanished from an image
    #    before, silently, because a Dockerfile did not copy data/.
    try:
        v = get(base, "/validation")
        check("router validation shipped", bool(v.get("available")),
              v.get("verdict", v.get("reason", ""))[:44])
    except Exception as exc:
        check("router validation shipped", False, str(exc))

    try:
        s = get(base, "/sessions")
        check("tool example shipped", bool(s.get("available")),
              "example" if s.get("is_example") else "live data")
    except Exception as exc:
        check("tool example shipped", False, str(exc))

    # 6. The page itself, and the script that fills it.
    try:
        with urllib.request.urlopen(base + "/", timeout=TIMEOUT) as r:
            html = r.read().decode("utf-8", "replace")
        check("page served", "<title>" in html, f"{len(html):,} bytes")
        check("study markup present", 'id="study-nums"' in html)
        check("tool panel present", 'id="tool"' in html)
    except Exception as exc:
        check("page served", False, str(exc))

    print()
    if problems:
        print(f"{len(problems)} problem(s): " + ", ".join(problems))
        return 1
    print("everything consistent")
    return 0


if __name__ == "__main__":
    sys.exit(main())
