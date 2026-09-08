#!/bin/bash
# Check the deployed dashboard and page (macOS notification) when it lies.
#
# Runs on a schedule via launchd and needs no special access: the site is
# public, so unlike the transcript sync it does not need to read ~/.claude.
# A copy lives in ~/code/ccost-sync (a launchd-readable location), not on the
# Desktop where macOS blocks background agents.
#
# Fires only after two consecutive failures. The laptop sleeps and drops off
# wifi between fires, and one transient blip should not make you duck. Two
# checks roughly an hour apart both failing is a real problem worth a page.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

LOG="data/health_check.log"
STRIKES="data/.health_check_strikes"
MAX_STRIKES=2

log() { printf "%s  %s\n" "$(date '+%Y-%m-%d %H:%M:%S')" "$*" >> "$LOG"; }

if venv/bin/python scripts/check_live.py > /dev/null 2>&1; then
    rm -f "$STRIKES"
    log "ok"
    exit 0
fi

# Failed. Count it, and only page when consecutive failures reach the limit.
strikes=0
if [ -f "$STRIKES" ]; then
    strikes=$(<"$STRIKES")
fi
strikes=$((strikes + 1))
echo "$strikes" > "$STRIKES"

if [ "$strikes" -lt "$MAX_STRIKES" ]; then
    log "check failed ($strikes/$MAX_STRIKES); standing by"
    exit 1     # transient or not, exit non-zero is still the truth
fi

log "check failed $MAX_STRIKES consecutive times — paging"
osascript -e 'display notification "llmcostwatchdog.onrender.com has failed its health check twice in a row. Run: bash scripts/health_check.sh" with title "LLM Cost Watchdog"' > /dev/null 2>&1
exit 1