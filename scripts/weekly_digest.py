#!/usr/bin/env python3
"""
Standalone entrypoint for the weekly digest, intended for cron:

    0 9 * * MON cd /path/to/llm-cost-watchdog && venv/bin/python scripts/weekly_digest.py

Appends a run record to data/reports/digest_runs.log so scheduled runs leave
an audit trail even when nobody is watching stdout.
"""

import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.analyzer import check_budget_status  # noqa: E402
from src.digest import generate_digest  # noqa: E402

LOG_PATH = Path(__file__).resolve().parent.parent / "data" / "reports" / "digest_runs.log"


def main() -> int:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(LOG_PATH), logging.StreamHandler(sys.stdout)],
    )
    logger = logging.getLogger("weekly_digest")

    started = datetime.now(timezone.utc)
    logger.info("weekly digest run started")

    try:
        digest_text = generate_digest("week")
        budget = check_budget_status("weekly")
        logger.info(
            "budget: %s (%.2f%% of $%.2f used, $%.6f remaining)",
            budget["status"], budget["percent_used"], budget["limit_usd"], budget["remaining_usd"],
        )
        if budget["status"] in ("near", "over"):
            logger.warning("BUDGET ALERT: weekly spend is %s the configured limit", budget["status"])

        print()
        print("=" * 60)
        print(digest_text)
        print("=" * 60)

        elapsed = (datetime.now(timezone.utc) - started).total_seconds()
        logger.info("weekly digest run finished in %.1fs", elapsed)
        return 0

    except Exception:
        logger.exception("weekly digest run failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
