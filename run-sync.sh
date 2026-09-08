#!/bin/bash
# The scheduled sync, running from a location macOS will let it read.
#
# The working copy of this project lives under ~/Desktop, and macOS refuses
# background agents access to Desktop, Documents and Downloads without Full Disk
# Access, which only a human clicking through System Settings can grant. So the
# job was installed, reported success, and never ran.
#
# It turns out Desktop was incidental. Verified with a probe agent: a launchd job
# CAN read ~/.claude/projects and ~/code, and cannot read ~/Desktop. The sync
# needs the transcripts, some code, and the network. None of that has to be on
# the Desktop, so this is a clone that lives where the job is allowed to look.
#
# It pulls from GitHub first, so it tracks whatever was last pushed rather than
# drifting into a stale fork of itself.
set -uo pipefail
cd "$(dirname "$0")" || exit 1

echo "=== $(date '+%Y-%m-%d %H:%M:%S') ==="
git pull --quiet --ff-only origin main 2>/dev/null && echo "updated from GitHub" || echo "could not pull; running the code already here"

# --rebuild, not a plain import. The importer records a checkpoint per
# transcript *next to the transcript*, so every copy of this repo shares them
# while each keeps its own database. A fresh clone is therefore told everything
# has already been imported, imports almost nothing, and ends up with a nearly
# empty ledger that looks legitimate. That is exactly what happened here: 33
# rows were published over 9,033.
#
# Rebuilding purges this copy's rows and re-reads every transcript, so the
# result depends only on what is on disk and not on what some other copy did
# earlier. It costs a couple of minutes and removes a whole class of silent
# corruption.
venv/bin/python scripts/import_all_projects.py --rebuild || exit 1

# The 8pm fire often lands right as the machine wakes from sleep, before DNS
# is back up, and fails with "nodename nor servname provided" even though the
# rebuild above (no network needed) succeeded. Retrying a few times a few
# seconds apart covers that without masking a real outage.
synced=0
for attempt in 1 2 3 4 5; do
    if venv/bin/python scripts/sync_to_turso.py; then
        synced=1
        break
    fi
    echo "sync attempt $attempt failed"
    [ "$attempt" -lt 5 ] && sleep 15
done
[ "$synced" -eq 1 ] || exit 1
echo "sync complete"
