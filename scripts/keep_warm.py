#!/usr/bin/env python3
"""Ping the gateway every 15 minutes to keep it from sleeping on Render free tier.

    # Add to Render as a background worker:
    python scripts/keep_warm.py

    # Or run locally in the background:
    python scripts/keep_warm.py --base http://localhost:8000

The gateway sleeps after 15 minutes of inactivity on Render free tier,
taking ~13s to wake up on the next request. A simple keep-alive ping
every 14 minutes prevents sleep without consuming meaningful resources.
"""

import argparse
import time
import sys

try:
    import httpx
except ImportError:
    sys.exit("pip install httpx")


def keep_warm(base_url: str, interval_seconds: int = 840, timeout_s: int = 10) -> None:
    """Ping the gateway every interval_seconds (default 14 min)."""
    client = httpx.Client(timeout=timeout_s)

    while True:
        try:
            response = client.get(f"{base_url}/health")
            if response.is_success:
                print(f"[{time.strftime('%H:%M:%S')}] ping ok")
            else:
                print(f"[{time.strftime('%H:%M:%S')}] ping failed: {response.status_code}")
        except Exception as e:
            print(f"[{time.strftime('%H:%M:%S')}] error: {e}")

        time.sleep(interval_seconds)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="https://llmcostwatchdog.onrender.com",
                    help="gateway base URL (default: production Render URL)")
    ap.add_argument("--interval", type=int, default=840,
                    help="seconds between pings (default 840 = 14 min)")
    args = ap.parse_args()

    print(f"keeping {args.base} warm every {args.interval}s")
    keep_warm(args.base, interval_seconds=args.interval)
